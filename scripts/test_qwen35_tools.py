"""CPU protocol and real TCP tests for native function-tool Chat Completions."""
import copy
import http.client
import json
from pathlib import Path
import threading
import unittest

import qwen35_api as A
import qwen35_chat as C
import qwen35_tools as T
from test_qwen35_api import Backend,MODEL,KEY,request
from test_qwen35_chat import Template

TOOLS=[dict(type='function',function=dict(name='lookup',description='Read an entry',parameters=dict(
    type='object',properties=dict(key=dict(type='string'),count=dict(type='integer')),
    required=['key'],additionalProperties=False)))]
RAW='<tool_call>\n<function=lookup>\n<parameter=key>\nalpha\n</parameter>\n<parameter=count>\n2\n</parameter>\n</function>\n</tool_call>'


class ToolTemplate(Template):
    def render(self,messages,tools=None,add_generation_prompt=True):
        header='TOOLS '+json.dumps(tools,sort_keys=True)+'\n' if tools else ''
        out=header
        for m in messages:
            content=m['content']
            for call in m.get('tool_calls',[]):
                f=call['function'];content+='<tool_call>\n<function='+f['name']+'>\n'
                for k,v in f['arguments'].items():
                    content+='<parameter='+k+'>\n'+(v if isinstance(v,str) else json.dumps(v))+'\n</parameter>\n'
                content+='</function>\n</tool_call>'
            out+='<|im_start|>'+m['role']+'\n'+content+'<|im_end|>\n'
        return out+('<|im_start|>assistant\n' if add_generation_prompt else '')


class ParsingTests(unittest.TestCase):
    def policy(self,**kw):return T.policy(dict(tools=TOOLS,**kw))

    def test_plain_history_accepts_optional_empty_calls_without_tools(self):
        plain=[dict(role='user',content='Hi'),dict(role='assistant',content='Hello'),
               dict(role='user',content='Continue')]
        expected=A.validate_request(request(plain),MODEL)
        for value in (None,[]):
            messages=copy.deepcopy(plain);messages[1]['tool_calls']=value
            actual=A.validate_request(request(messages),MODEL)
            self.assertEqual(actual,expected)
            self.assertNotIn('tool_policy',actual)
        for value in (False,0,'',{},[dict(id='c1')]):
            messages=copy.deepcopy(plain);messages[1]['tool_calls']=value
            with self.assertRaises(A.ApiError):A.validate_request(request(messages),MODEL)

    def test_every_chunk_split_and_argument_values(self):
        for split in range(len(RAW)+1):
            emitted=[];o=T.Output(self.policy(),emitted.append)
            o.push(RAW[:split]);o.push(RAW[split:]);content,calls=o.finish('eos')
            self.assertEqual(content,'');self.assertEqual(calls[0]['function']['arguments'],{'key':'alpha','count':2})
            self.assertEqual(emitted,[],'calls must wait for complete native EOF')
            o.publish();self.assertEqual(emitted[0]['tool_calls'][0]['index'],0)
            self.assertEqual(json.loads(''.join(d['tool_calls'][0]['function']['arguments'] for d in emitted)),{'key':'alpha','count':2})

    def test_character_at_a_time_parallel_calls(self):
        o=T.Output(self.policy(),lambda _:None)
        for ch in RAW+'\n'+RAW:o.push(ch)
        _,calls=o.finish('eos');self.assertEqual(len(calls),2);self.assertNotEqual(calls[0]['id'],calls[1]['id'])

    def test_policy_rejections(self):
        for kw,raw in (({'tool_choice':'none'},RAW),({'parallel_tool_calls':False},RAW+'\n'+RAW)):
            with self.assertRaises(ValueError):T.Output(self.policy(**kw),lambda _:None).push(raw)
        for bad in (dict(tools=TOOLS*2),dict(tools=TOOLS,tool_choice={'type':'function','function':{'name':'other'}}),
                    dict(tools=TOOLS,parallel_tool_calls=1),dict(tools=[dict(type='function',function=dict(TOOLS[0]['function'],strict=True))])):
            with self.assertRaises(ValueError):T.policy(bad)

    def test_truncated_call_never_publishes(self):
        for text,stop in ((RAW[:-1],'eos'),(RAW,'max_new_tokens')):
            emitted=[];o=T.Output(self.policy(),emitted.append);o.push(text)
            with self.assertRaises(ValueError):o.finish(stop)
            self.assertEqual(emitted,[])

    def test_required_missing_call_and_named_function(self):
        o=T.Output(self.policy(tool_choice='required'),lambda _:None);o.push('I guessed')
        with self.assertRaises(ValueError):o.finish('eos')
        p=self.policy(tool_choice={'type':'function','function':{'name':'lookup'}})
        o=T.Output(p,lambda _:None);o.push(RAW);self.assertEqual(len(o.finish('eos')[1]),1)

    def test_plain_text_unchanged(self):
        emitted=[];o=T.Output(self.policy(),emitted.append)
        for s in ('Hello ','world'):o.push(s)
        self.assertEqual(o.finish('eos'),('Hello world',[]));self.assertEqual(emitted,['Hello ','world'])

    def test_invalid_generated_args(self):
        for bad in (RAW.replace('lookup>','unknown>'),RAW.replace('\n2\n','\ntrue\n'),
            RAW.replace('<parameter=count>','<parameter=extra>'),RAW.replace('</function>','<parameter=key>\nbeta\n</parameter>\n</function>'),
            RAW+'bad suffix'):
            with self.subTest(bad=bad),self.assertRaises(ValueError):T.Output(self.policy(),lambda _:None).push(bad)

    def test_external_ref_never_resolves(self):
        bad=copy.deepcopy(TOOLS);bad[0]['function']['parameters']['properties']['key']={'$ref':'https://invalid.example/schema'}
        with self.assertRaises(ValueError):T.policy(dict(tools=bad))

    def test_local_ref_and_nested_schema(self):
        tools=copy.deepcopy(TOOLS)
        s=tools[0]['function']['parameters'];s['$defs']={'s':{'type':'string'}};s['properties']['key']={'$ref':'#/$defs/s'}
        p=T.policy(dict(tools=tools));o=T.Output(p,lambda _:None);o.push(RAW)
        self.assertEqual(o.finish('eos')[1][0]['function']['arguments']['key'],'alpha')
        o=T.Output(p,lambda _:None);o.push(RAW.replace('\nalpha\n','\n123\n'))
        self.assertEqual(o.finish('eos')[1][0]['function']['arguments']['key'],'123')

    def test_api_argument_order_is_not_history_identity(self):
        args='{"count":2,"key":"alpha"}'
        messages=[dict(role='user',content='Go'),dict(role='assistant',content=None,tool_calls=[dict(
            id='c1',type='function',function=dict(name='lookup',arguments=args))]),
            dict(role='tool',tool_call_id='c1',content='yes')]
        normalized=T.normalize_messages(messages,TOOLS)
        self.assertEqual(list(normalized[1]['tool_calls'][0]['function']['arguments']),['count','key'])
        o=T.Output(self.policy(),lambda _:None);o.push(RAW)
        self.assertEqual(list(o.finish('eos')[1][0]['function']['arguments']),['count','key'])

    def test_malformed_message_call_type_is_not_ignored(self):
        with self.assertRaises(ValueError):
            T.normalize_messages([dict(role='user',content='Go'),dict(role='assistant',content='X',tool_calls=0),
                dict(role='user',content='Again')],TOOLS)

    def test_retired_history_tool_does_not_require_current_declaration(self):
        call=dict(id='old1',type='function',function=dict(name='browser_session',arguments='{"action":"open"}'))
        messages=[dict(role='user',content='Inspect the page'),dict(role='assistant',content=None,tool_calls=[call]),
                  dict(role='tool',tool_call_id='old1',content='The browser CLI was unavailable'),
                  dict(role='assistant',content='The browser action could not run.'),dict(role='user',content='Continue')]
        current=[dict(type='function',function=dict(name='bsk_session',parameters=dict(type='object',properties={}))) ]
        for tools in (current,[],TOOLS):
            with self.subTest(tools=tools):
                normalized=A.validate_request(request(messages,tools=tools),MODEL)
                self.assertEqual(normalized['messages'][1]['tool_calls'][0]['function'],
                                 dict(name='browser_session',arguments={'action':'open'}))
                self.assertEqual([t['function']['name'] for t in normalized['tool_policy']['tools']],
                                 [t['function']['name'] for t in tools])
        self.assertEqual(messages[1]['tool_calls'][0]['function']['arguments'],'{"action":"open"}')

    def test_historical_schema_is_not_revalidated_against_current_schema(self):
        call=dict(id='old1',type='function',function=dict(name='lookup',arguments='{"old_parameter":42}'))
        messages=[dict(role='user',content='Earlier task'),dict(role='assistant',content=None,tool_calls=[call]),
                  dict(role='tool',tool_call_id='old1',content='completed')]
        self.assertEqual(T.normalize_messages(messages,TOOLS)[1]['tool_calls'][0]['function']['arguments'],
                         {'old_parameter':42})
        with self.assertRaisesRegex(ValueError,'violate schema'):
            T.arguments('lookup',{'old_parameter':42},TOOLS)

    def test_invalid_historical_names_json_and_result_pairs_still_reject(self):
        call=dict(id='old1',type='function',function=dict(name='retired_tool',arguments='{}'))
        valid=[dict(role='user',content='Earlier task'),dict(role='assistant',content=None,tool_calls=[call]),
               dict(role='tool',tool_call_id='old1',content='completed')]
        bad=[]
        for name in (None,42,'bad>name','x\nfunction',''):
            case=copy.deepcopy(valid);case[1]['tool_calls'][0]['function']['name']=name;bad.append(case)
        for args in ('[]','null','{"a":NaN}','{"a":1,"a":2}','{"bad>key":1}','{"bad\\nkey":1}'):
            case=copy.deepcopy(valid);case[1]['tool_calls'][0]['function']['arguments']=args;bad.append(case)
        bad.append(valid[:-1])
        case=copy.deepcopy(valid);case[2]['tool_call_id']='different';bad.append(case)
        for case in bad:
            with self.subTest(case=case),self.assertRaises(ValueError):T.normalize_messages(case,[])

    def test_retired_tool_cannot_be_newly_generated(self):
        emitted=[];output=T.Output(self.policy(),emitted.append)
        with self.assertRaisesRegex(ValueError,'undeclared function'):
            output.push(RAW.replace('lookup>','browser_session>'))
        self.assertEqual(emitted,[])

    def test_history_ids_order_and_missing_results(self):
        calls=[dict(id='c1',type='function',function=dict(name='lookup',arguments='{"key":"a"}')),
               dict(id='c2',type='function',function=dict(name='lookup',arguments='{"key":"b"}'))]
        msgs=[dict(role='user',content='Go'),dict(role='assistant',content=None,tool_calls=calls),
              dict(role='tool',tool_call_id='c2',content='B'),dict(role='tool',tool_call_id='c1',content='A')]
        result=T.normalize_messages(msgs,TOOLS)
        self.assertEqual([m['tool_call_id'] for m in result[-2:]],['c1','c2'])
        for bad in (msgs[:-1],msgs+[dict(role='tool',tool_call_id='c1',content='again')],
                    msgs[:2]+[dict(role='tool',tool_call_id='foreign',content='X')]):
            with self.assertRaises(ValueError):T.normalize_messages(bad,TOOLS)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.backend=Backend();self.backend.template=ToolTemplate();self.backend.text=RAW
        def factory():
            t=self.backend.factory();health=t.health
            def changed():
                h=health();h['session_limits'].update(initial_output_limit=4096,continuation_output_limit=4096)
                return h
            t.health=changed;return t
        self.controller=A.Controller(self.backend.template,'http://127.0.0.1',MODEL,transport_factory=factory)
        self.addCleanup(self.controller.close)

    def complete(self,messages=None,**options):
        result={};emitted=[]
        def deliver(meta,req,text,finish,usage):result.update(meta=meta,request=req,text=text,finish=finish,usage=usage)
        self.controller.complete(request(messages,tools=TOOLS,**options),lambda *_:None,emitted.append,deliver,lambda _:lambda:None)
        result['emitted']=emitted;return result

    def test_actual_tool_result_reuses_original_ids_not_canonical_xml(self):
        # The model whitespace differs from the canonical serialization.
        self.backend.text=RAW.replace('<function=lookup>\n','<function=lookup>\n\n')
        first=self.complete();self.assertEqual(first['finish'],'tool_calls')
        before=list(self.controller.current.history);session=self.controller.current.session_id
        calls=first['request']['response_tool_calls']
        messages=[dict(role='user',content='Hi'),dict(role='assistant',content=None,tool_calls=calls),
                  dict(role='tool',tool_call_id=calls[0]['id'],content='{"value":17}')]
        self.backend.text='Value is 17.'
        second=self.complete(messages)
        self.assertEqual(second['meta']['session_mode'],'exact_history_append')
        self.assertEqual(self.controller.current.session_id,session)
        self.assertEqual(self.backend.requests[-1][1]['exact_history'],before)
        self.assertEqual(self.backend.loads,1)

    def test_extended_tool_result_code_budget_preserves_raw_history(self):
        self.backend.request_budgets=True
        def factory():
            transport=self.backend.factory();health=transport.health
            def extended():
                value=health();value['session_limits'].update(initial_output_limit=4096,
                    continuation_output_limit=128,continuation_max_output_limit=1024)
                return value
            transport.health=extended;return transport
        self.controller=A.Controller(self.backend.template,'http://127.0.0.1',MODEL,transport_factory=factory)
        self.addCleanup(self.controller.close)
        first=self.complete();history=list(self.controller.current.history);session=self.controller.current.session_id
        calls=first['request']['response_tool_calls']
        messages=[dict(role='user',content='Hi'),dict(role='assistant',content=None,tool_calls=calls),
                  dict(role='tool',tool_call_id=calls[0]['id'],content='actual compiler diagnostic')]
        self.backend.text='X'*180
        result=self.complete(messages,max_tokens=512)
        self.assertEqual(result['finish'],'stop')
        self.assertEqual(result['text'],'X'*180)
        self.assertEqual(self.backend.requests[-1][1]['max_new_tokens'],512)
        self.assertEqual(self.backend.requests[-1][1]['exact_history'],history)
        self.assertEqual(self.controller.current.session_id,session)
        self.assertEqual(self.backend.loads,1)

    def test_schema_change_reopens_instead_of_reusing_kv(self):
        first=self.complete();calls=first['request']['response_tool_calls']
        messages=[dict(role='user',content='Hi'),dict(role='assistant',content=None,tool_calls=calls),
                  dict(role='tool',tool_call_id=calls[0]['id'],content='done')]
        self.backend.text='Done'
        result=self.complete(messages,tool_choice='none')
        self.assertEqual(result['meta']['session_mode'],'canonical_full_history_open')
        self.assertEqual(self.backend.closes,1)

    def test_invalid_call_closes_completed_native_session(self):
        self.backend.text=RAW.replace('lookup>','other>')
        with self.assertRaises(ValueError):self.complete()
        self.assertIsNone(self.controller.current)
        self.assertTrue(self.controller.faulted)

    def test_tcp_stream_and_nonstream_tool_shapes(self):
        server=A.Server(('127.0.0.1',0),self.controller,KEY)
        thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True);thread.start()
        try:
            for streaming in (False,True):
                conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5)
                body=request(tools=TOOLS,stream=streaming)
                conn.request('POST','/v1/chat/completions',json.dumps(body),{'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
                r=conn.getresponse();raw=r.read().decode();conn.close();self.assertEqual(r.status,200,raw)
                if not streaming:
                    value=json.loads(raw);self.assertEqual(value['choices'][0]['finish_reason'],'tool_calls')
                    self.assertIsNone(value['choices'][0]['message']['content'])
                    self.assertEqual(json.loads(value['choices'][0]['message']['tool_calls'][0]['function']['arguments']),{'key':'alpha','count':2})
                else:
                    events=[json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
                    deltas=[e['choices'][0]['delta'] for e in events if e['choices']]
                    args=''.join(c['function'].get('arguments','') for d in deltas for c in d.get('tool_calls',[]))
                    self.assertEqual(json.loads(args),{'key':'alpha','count':2});self.assertIn('data: [DONE]',raw)
                    self.assertEqual(events[-1]['choices'][0]['finish_reason'],'tool_calls')
                    self.assertNotIn('<tool_call>',raw)
        finally:
            server.shutdown();thread.join(2);server.server_close()

    def test_tcp_history_survives_tool_rename_and_removal(self):
        old_call=dict(id='old_browser',type='function',function=dict(name='browser_session',arguments='{"action":"open"}'))
        messages=[dict(role='user',content='Earlier task'),dict(role='assistant',content=None,tool_calls=[old_call]),
                  dict(role='tool',tool_call_id='old_browser',content='The CLI was unavailable'),
                  dict(role='assistant',content='The earlier browser action failed.'),dict(role='user',content='Continue')]
        current=[dict(type='function',function=dict(name='bsk_session',parameters=dict(type='object',properties={}))) ]
        self.backend.text='Ready to continue.'
        server=A.Server(('127.0.0.1',0),self.controller,KEY)
        thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True);thread.start()
        try:
            for tools in (current,[]):
                for streaming in (False,True):
                    with self.subTest(tools=bool(tools),streaming=streaming):
                        connection=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=5)
                        connection.request('POST','/v1/chat/completions',json.dumps(request(messages,tools=tools,stream=streaming)),
                                           {'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
                        response=connection.getresponse();raw=response.read().decode();connection.close()
                        self.assertEqual(response.status,200,raw)
                        if streaming:self.assertIn('data: [DONE]',raw)
                        else:self.assertEqual(json.loads(raw)['choices'][0]['message']['content'],'Ready to continue.')
                        self.assertEqual(self.controller.current.tool_policy['tools'],T.policy({'tools':tools})['tools'])
                        self.assertEqual(self.controller.current.messages[1]['tool_calls'][0]['function']['name'],'browser_session')
        finally:
            server.shutdown();thread.join(2);server.server_close()

    def test_disconnect_does_not_adopt_pending_call(self):
        self.backend.mode='truncated_http'
        with self.assertRaises(ValueError):self.complete()
        self.assertIsNone(self.controller.current)

    def test_budget_truncated_tool_never_exposes_call(self):
        self.backend.mode='budget'
        with self.assertRaises(ValueError):self.complete()
        self.assertIsNone(self.controller.current)

    def test_client_failed_tool_publication_closes_native_session(self):
        def broken(*_):raise OSError('client RST')
        with self.assertRaises(OSError):
            self.controller.complete(request(tools=TOOLS),lambda *_:None,lambda _:None,broken,lambda _:lambda:None)
        self.assertIsNone(self.controller.current)
        self.assertGreater(self.backend.closes,0)


class RealTemplateTests(unittest.TestCase):
    @unittest.skipUnless(Path('F:/qwen3.8 27B/tokenizer.json').is_file(),'local model tokenizer required')
    def test_plain_assistant_history_optional_fields_are_render_local(self):
        template=C.ChatTemplate(Path('F:/qwen3.8 27B'))
        messages=[dict(role='user',content='Write a function'),
                  dict(role='assistant',content='def identity(x): return x'),
                  dict(role='user',content='Test it')]
        before=copy.deepcopy(messages)
        normalized=[C.ChatTemplate.message(m['role'],m['content']) for m in messages]
        self.assertEqual(template.render(messages),template.render(normalized))
        self.assertEqual(messages,before)
        self.assertEqual(template.render(messages,add_generation_prompt=False),
                         template.render(normalized,add_generation_prompt=False))

    @unittest.skipUnless(Path('F:/qwen3.8 27B/tokenizer.json').is_file(),'local model tokenizer required')
    def test_native_template_and_marker_decoder(self):
        template=C.ChatTemplate(Path('F:/qwen3.8 27B'))
        policy=T.policy(dict(tools=TOOLS));ledger=T.Ledger(template,policy)
        messages=T.normalize_messages([dict(role='user',content='Lookup alpha')],TOOLS)
        plan=ledger.plan_messages(messages)
        self.assertIn('<tools>',plan['payload']['prompt'])
        self.assertEqual(template.tokenizer.decode(template.encode(RAW),skip_special_tokens=True),RAW)
        call=dict(id='c1',type='function',function=dict(name='lookup',arguments='{"key":"alpha"}'))
        msgs=T.normalize_messages([dict(role='user',content='Lookup alpha'),dict(role='assistant',content=None,tool_calls=[call]),
            dict(role='tool',tool_call_id='c1',content='{"value":17}')],TOOLS)
        rendered=ledger.render(msgs)
        self.assertIn('<parameter=key>\nalpha\n</parameter>',rendered)
        self.assertIn('<tool_response>\n{"value":17}\n</tool_response>',rendered)


if __name__=='__main__':unittest.main()
