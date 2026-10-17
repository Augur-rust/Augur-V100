"""Function-tool protocol for the local Qwen template. Never executes tools.

Tool calls are exposed only after native EOF and argument validation. Their
arguments may be split into API deltas, but unchecked model XML is never an
executable call. Exact continuation retains original consumed token IDs.
"""
from __future__ import annotations

import copy
import json
import re
import uuid

import qwen35_chat as C

NAME = re.compile(r'[A-Za-z0-9_-]{1,64}\Z')
CALL_ID = re.compile(r'[A-Za-z0-9_-]{1,128}\Z')
MAX_CALLS = 128
MAX_TOOL_BYTES = 256 * 1024


def ordered(value):
    """API argument member order is irrelevant; make template rendering stable."""
    if isinstance(value,dict):return {key:ordered(value[key]) for key in sorted(value)}
    if isinstance(value,list):return [ordered(item) for item in value]
    return value


def validate_schema(schema):
    from jsonschema import Draft202012Validator
    C.require(type(schema) is dict and schema.get('type') == 'object',
              'tool parameters must be an object JSON schema')
    # Resolution must never perform IO. Local definitions are supported;
    # external and dynamic references are explicitly unsupported.
    def visit(node, depth=0):
        C.require(depth <= 32, 'tool schema nesting exceeds 32')
        if isinstance(node, dict):
            for key,value in node.items():
                if key=='properties' and isinstance(value,dict):
                    C.require(all(isinstance(name,str) and name and not re.search(r'[<>\r\n=]',name)
                                  for name in value),'parameter names cannot contain template delimiters')
                if key == '$ref':
                    C.require(isinstance(value,str) and value.startswith('#/'), 'external schema references are forbidden')
                C.require(key not in ('$dynamicRef','$recursiveRef','$id'), 'dynamic schema resolution is unsupported')
                visit(value,depth+1)
        elif isinstance(node,list):
            for value in node: visit(value,depth+1)
    visit(schema)
    Draft202012Validator.check_schema(schema)


def policy(value):
    tools=value.get('tools')
    tools=[] if tools is None else tools
    C.require(type(tools) is list and len(tools)<=128,'tools must be an array of at most 128 functions')
    normalized=[]; names=set()
    for tool in tools:
        C.require(type(tool) is dict and set(tool)=={'type','function'} and tool['type']=='function',
                  'only function tools are supported')
        f=tool['function']
        C.require(type(f) is dict and set(f)<={'name','description','parameters','strict'}
                  and isinstance(f.get('name'),str) and NAME.fullmatch(f['name']), 'invalid function definition/name')
        C.require(f['name'] not in names,'duplicate function name')
        names.add(f['name'])
        C.require('description' not in f or isinstance(f['description'],str),'function description must be text')
        C.require(f.get('strict') is None or f.get('strict') is False,
                  'strict constrained generation is not implemented; use strict=false or omit it')
        f=copy.deepcopy(f)
        f.setdefault('parameters',dict(type='object',properties={},additionalProperties=False))
        try:
            validate_schema(f['parameters'])
        except Exception as error:
            raise C.ProtocolError('invalid or unsupported tool schema: '+str(error)) from error
        normalized.append(dict(type='function',function=f))
    C.require(len(json.dumps(normalized,ensure_ascii=False,allow_nan=False).encode())<=MAX_TOOL_BYTES,
              'tool definitions exceed byte limit')
    choice=value.get('tool_choice','auto' if tools else 'none')
    if isinstance(choice,dict):
        C.require(set(choice)=={'type','function'} and choice['type']=='function'
                  and type(choice['function']) is dict and set(choice['function'])=={'name'}
                  and choice['function']['name'] in names,'forced function is not declared')
    else:
        C.require(choice in ('none','auto','required'),'unsupported tool_choice')
        C.require(bool(tools) or choice=='none','tool_choice requires declared tools')
    parallel=value.get('parallel_tool_calls',True)
    C.require(type(parallel) is bool,'parallel_tool_calls must be boolean')
    return dict(tools=normalized,tool_choice=copy.deepcopy(choice),parallel_tool_calls=parallel)


def safe_content(text,role,empty=False):
    if text is None and empty:text=''
    if type(text) is list:
        C.require(text and all(type(p) is dict and set(p)=={'type','text'} and p['type']=='text'
                  and isinstance(p['text'],str) for p in text),'only text content parts are supported')
        text=''.join(p['text'] for p in text)
    C.require(isinstance(text,str) and (empty or text.strip()),'text content required')
    C.require(not re.search(r'<\|[^<>\r\n]*\|>|</?think>|</?tool_(?:call|response)>',text),
              'literal model/tool control delimiters are not accepted in message content')
    return dict(role=role,content=text,reasoning_content='',tool_calls=[])


def arguments(name,args,tools):
    from jsonschema import Draft202012Validator
    f=next((t['function'] for t in tools if t['function']['name']==name),None)
    C.require(f is not None,'model called an undeclared function')
    C.require(type(args) is dict,'function arguments must be a JSON object')
    errors=iter(Draft202012Validator(f['parameters']).iter_errors(args))
    first=next(errors,None)
    C.require(first is None,'function arguments violate schema'+(' at '+str(list(first.path)) if first else ''))
    return f


def historical_arguments(name,args):
    """Validate recorded data without granting current tool execution authority.

    Request tools describe the next generation. Completed history may mention
    retired/renamed tools or an older schema for a still-present name. Current
    schemas cannot validate those records; call/result pairing is checked by
    normalize_messages. Generated calls still use arguments()/parse_block().
    """
    C.require(isinstance(name,str) and NAME.fullmatch(name), 'invalid historical function name')
    C.require(type(args) is dict, 'historical function arguments must be a JSON object')
    C.require(all(isinstance(key,str) and key and not re.search(r'[<>\r\n=]',key)
                  for key in args), 'historical parameter names cannot contain template delimiters')


def normalize_messages(messages,tools):
    C.require(type(messages) is list and 1<=len(messages)<=129,'1..129 messages required')
    result=[]; used_ids=set(); pending={}; tool_group={}; expected='user'; user_seen=False
    for index,m in enumerate(messages):
        C.require(type(m) is dict and isinstance(m.get('role'),str),'message object and role required')
        role=m['role']
        if role in ('system','developer'):
            C.require(index==0 and set(m)=={'role','content'},'one leading system/developer message required')
            result.append(safe_content(m['content'],'system'));continue
        if role=='tool':
            C.require(set(m)=={'role','content','tool_call_id'} and m['tool_call_id'] in pending
                      and m['tool_call_id'] not in tool_group,'unknown, duplicate or missing tool_call_id')
            item=safe_content(m['content'],'tool',empty=True)
            item['tool_call_id']=m['tool_call_id'];tool_group[m['tool_call_id']]=item
            if len(tool_group)==len(pending):
                # The supplied template has no call-ID field in tool_response.
                # Canonicalize result order to assistant call order.
                result.extend(tool_group[id] for id in pending)
                pending={};tool_group={};expected='assistant'
            continue
        C.require(not pending,'all outstanding tool calls require one result before another message')
        C.require(role==expected,'invalid user/assistant/tool message sequence')
        if role=='user':
            C.require(set(m)=={'role','content'},'unknown user message fields')
            result.append(safe_content(m['content'],'user'));expected='assistant';user_seen=True
        else:
            C.require(set(m)<={'role','content','tool_calls'},'unknown assistant message fields')
            calls=m.get('tool_calls')
            calls=[] if calls is None else calls
            C.require(type(calls) is list and len(calls)<=MAX_CALLS,'invalid assistant tool_calls')
            item=safe_content(m.get('content'),'assistant',empty=True)
            if calls:
                # Output.finish strips the separator before generated tool XML.
                # SSE clients have already received it; normalize their replay
                # identically so it can retain the original native token history.
                item['content']=item['content'].rstrip()
            for call in calls:
                C.require(type(call) is dict and set(call)=={'id','type','function'}
                          and call['type']=='function' and isinstance(call['id'],str)
                          and CALL_ID.fullmatch(call['id']) and call['id'] not in used_ids,'invalid/reused tool call ID')
                f=call['function']
                C.require(type(f) is dict and set(f)=={'name','arguments'} and isinstance(f['arguments'],str),
                          'API function arguments must be a JSON string')
                args=ordered(C.strict_json(f['arguments']));historical_arguments(f['name'],args)
                item['tool_calls'].append(dict(id=call['id'],type='function',function=dict(name=f['name'],arguments=args)))
                used_ids.add(call['id']);pending[call['id']]=True
            result.append(item);expected='user' if not calls else 'tool'
    C.require(user_seen and not pending and result[-1]['role'] in ('user','tool'),
              'conversation must end in a user message or a complete group of tool results')
    return result


def parse_block(block,tools):
    match=re.fullmatch(r'<tool_call>\s*<function=([A-Za-z0-9_-]{1,64})>\s*(.*?)\s*</function>\s*</tool_call>',block,re.S)
    C.require(match is not None,'malformed Qwen tool-call block')
    name,body=match.groups()
    f=next((t['function'] for t in tools if t['function']['name']==name),None)
    C.require(f is not None,'model called an undeclared function')
    args={};position=0
    for part in re.finditer(r'<parameter=([^<>\r\n=]+)>\n?(.*?)\n?</parameter>',body,re.S):
        C.require(not body[position:part.start()].strip(),'malformed parameter content')
        key,raw=part.groups();C.require(key not in args,'duplicate generated parameter')
        C.require(not re.search(r'</?(?:tool_call|function|parameter)(?:>|=)',raw),'nested tool markup in parameter')
        # The template prints strings without JSON quotes. Prefer a valid raw
        # string even through local $ref/anyOf, otherwise parse a JSON value.
        # Validation against the full schema still checks missing/unknown keys.
        from jsonschema import Draft202012Validator
        probe=dict(f['parameters'],required=[key])
        probe.pop('minProperties',None);probe.pop('dependentRequired',None)
        property_schema=f['parameters'].get('properties',{}).get(key,{})
        probe['properties']={key:property_schema};probe.pop('allOf',None)
        if Draft202012Validator(probe).is_valid({key:raw}):
            value=raw
        else:
            try:
                value=C.strict_json('{"value":'+raw+'}')['value']
            except ValueError:
                value=raw
        args[key]=value;position=part.end()
    C.require(not body[position:].strip(),'unparsed parameter content')
    arguments(name,args,tools)
    return dict(name=name,arguments=ordered(args))


class Output:
    """Preserve ordinary text streaming; hold each call until native EOF."""
    def __init__(self,policy,emit):
        self.policy,self.emit=policy,emit
        self.buffer='';self.content='';self.calls=[];self.in_calls=False;self.total=0

    def push(self,text):
        self.total+=len(text.encode());C.require(self.total<=4*1024*1024,'tool response exceeds byte limit')
        self.buffer+=text
        while True:
            start=self.buffer.find('<tool_call>')
            if start<0:
                if self.in_calls:
                    C.require(not self.buffer.strip() or '<tool_call>'.startswith(self.buffer.strip()),
                              'text after a tool call is unsupported by the model template')
                    return
                keep=0
                for n in range(1,min(len(self.buffer),len('<tool_call>')-1)+1):
                    if '<tool_call>'.startswith(self.buffer[-n:]):keep=n
                ready=self.buffer[:-keep] if keep else self.buffer
                self.buffer=self.buffer[-keep:] if keep else ''
                if ready:self.content+=ready;self.emit(ready)
                return
            prefix=self.buffer[:start]
            C.require(not self.in_calls or not prefix.strip(),'content between tool calls')
            if not self.in_calls and prefix:self.content+=prefix;self.emit(prefix)
            self.in_calls=True;self.buffer=self.buffer[start:]
            end=self.buffer.find('</tool_call>')
            if end<0:return
            end+=len('</tool_call>')
            C.require(self.policy['tool_choice']!='none','tool call forbidden by tool_choice=none')
            f=parse_block(self.buffer[:end],self.policy['tools'])
            choice=self.policy['tool_choice']
            C.require(not isinstance(choice,dict) or f['name']==choice['function']['name'],'forced function differs')
            C.require(len(self.calls)<MAX_CALLS and (self.policy['parallel_tool_calls'] or not self.calls),
                      'generated tool-call count exceeds policy')
            self.calls.append(dict(id='call_'+uuid.uuid4().hex,type='function',function=f))
            self.buffer=self.buffer[end:]

    def finish(self,stop):
        if self.in_calls:
            C.require(not self.buffer.strip() and stop=='eos','incomplete/truncated tool-call response')
        else:
            C.require(not self.buffer,'incomplete tool-call delimiter')
        choice=self.policy['tool_choice']
        C.require(bool(self.calls) or choice not in ('required',) and not isinstance(choice,dict),
                  'required function call was not generated')
        return self.content.rstrip() if self.calls else self.content,self.calls

    def publish(self):
        for index,call in enumerate(self.calls):
            args=json.dumps(call['function']['arguments'],ensure_ascii=False,separators=(',',':'),allow_nan=False)
            self.emit(dict(tool_calls=[dict(index=index,id=call['id'],type='function',
                function=dict(name=call['function']['name'],arguments=''))]))
            for offset in range(0,len(args),128):
                self.emit(dict(tool_calls=[dict(index=index,function=dict(arguments=args[offset:offset+128]))]))


def public_calls(calls):
    return [dict(id=c['id'],type='function',function=dict(name=c['function']['name'],arguments=json.dumps(
        c['function']['arguments'],ensure_ascii=False,separators=(',',':'),allow_nan=False))) for c in calls]


class Ledger(C.Ledger):
    def __init__(self,template,policy):
        super().__init__(template,'resident')
        self.tool_policy=copy.deepcopy(policy)

    def render(self,messages,generation=True):
        tools=self.tool_policy['tools'] if self.tool_policy['tool_choice']!='none' else None
        if isinstance(self.tool_policy['tool_choice'],dict):
            name=self.tool_policy['tool_choice']['function']['name']
            tools=[t for t in tools if t['function']['name']==name]
        # tool_choice is an explicit prompt policy and a checked postcondition,
        # not grammar-constrained decoding. Auto leaves the native template
        # entirely unchanged.
        instruction=None
        if self.tool_policy['tool_choice']=='required':
            instruction='This request requires at least one function call before any final answer.'
        elif isinstance(self.tool_policy['tool_choice'],dict):
            instruction='This request requires a function call to '+name+'. Do not answer without that call.'
        if self.tool_policy['parallel_tool_calls'] is False and tools:
            instruction=(instruction+'\n' if instruction else '')+'Call at most one function in each assistant response.'
        if instruction:
            messages=copy.deepcopy(messages)
            if messages[0]['role']=='system':messages[0]['content']+='\n\n'+instruction
            else:messages.insert(0,dict(role='system',content=instruction,reasoning_content='',tool_calls=[]))
        return self.template.render(messages,tools=tools,add_generation_prompt=generation)

    def plan_messages(self,messages,output_reserve=134):
        C.require(type(output_reserve) is int and output_reserve>=134,'invalid tool continuation output reserve')
        C.require(not self.poisoned,'session is uncertain')
        full=self.render(messages)
        rid='turn-'+uuid.uuid4().hex
        if self.terminal is None:
            ids=self.template.encode(full)
            lo,hi,_=C.PROFILES[self.profile]
            C.require(lo<=len(ids)<=hi and len(full.encode())<=4*1024*1024,'initial tool input extent')
            payload=dict(request_id=rid,session_id=self.session_id,prompt=full);path='/v1/session/open'
        else:
            C.require(messages[:len(self.messages)]==self.messages,'tool history mismatch')
            pending=self.terminal['session']['pending_token']
            C.require(self.terminal['stop_reason']=='eos' and pending==self.template.end,
                      'tool history append requires completed message EOS')
            canonical=self.render(self.messages,generation=False)
            C.require(canonical.endswith('<|im_end|>\n') and full.startswith(canonical),'template rewrote tool history')
            suffix=full[len(canonical)-1:]
            suffix_ids=self.template.encode(suffix)
            C.require(0<len(suffix_ids)<=2048 and len(suffix.encode())<=16384,'tool result suffix exceeds native limit')
            ids=self.history+[pending]+suffix_ids
            C.require(len(ids)+output_reserve<=C.PROFILES[self.profile][2],'tool continuation reserve exhausted')
            payload=dict(request_id=rid,session_id=self.session_id,revision=self.terminal['session']['revision'],
                exact_history=self.history.copy(),suffix=suffix);path='/v1/session/append'
            # Preserve the raw model XML and original exact token history. Never
            # replace consumed tokens with re-serialized API argument strings.
            full=self.prefix+self.last_text+'<|im_end|>'+suffix
        return dict(path=path,payload=payload,input_ids=ids,messages=copy.deepcopy(messages),prefix=full)
