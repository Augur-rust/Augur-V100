/** Run against the actual installed DSH runtime; only inference is scripted. */
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {pathToFileURL} from 'node:url';
const install=process.argv[2];
if(!install) throw Error('Usage: node test.mjs DSH_INSTALL [receipt.json]');
const req=createRequire(path.join(install,'resources/app/package.json'));
const load=async name=>import(pathToFileURL(req.resolve(name)).href);
const names=['@deepseek-ai/cordis','@deepseek-ai/cordis-plugin-loader','@deepseek-ai/cordis-plugin-include','@deepseek-ai/dsh-llm','@deepseek-ai/dsh-session','@deepseek-ai/dsh-session-projection','@deepseek-ai/dsh-system-prompt','@deepseek-ai/dsh-tools','@deepseek-ai/dsh-agent','@deepseek-ai/dsh-agent-loop','@deepseek-ai/dsh-llm-retry'];
const mods=new Map(await Promise.all(names.map(async name=>[name,await load(name)])));
let source=await fs.readFile(new URL('./index.mjs',import.meta.url),'utf8');
for(const name of ['@deepseek-ai/dsh-llm','zod']) source=source.replace("from '"+name+"'",'from '+JSON.stringify(pathToFileURL(req.resolve(name)).href));
source=source.replace("from './single-flight.mjs'",'from '+JSON.stringify(new URL('./single-flight.mjs',import.meta.url).href));
const recovery=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const {Context}=mods.get('@deepseek-ai/cordis');
const {LlmAdapter,createUserMessage,ToolCallId,resolveRetryPolicy}=mods.get('@deepseek-ai/dsh-llm');
const {SessionId}=mods.get('@deepseek-ai/dsh-session');
const root=await fs.mkdtemp(path.join(os.tmpdir(),'augur-recovery-'));
const text=(text,kind='stop')=>[{type:'block-start',index:0,blockType:'text'},{type:'text-delta',index:0,text},{type:'block-end',index:0,block:{type:'text',text}},{type:'finish',reason:{kind}}];
const loop=()=>[{type:'block-start',index:0,blockType:'reasoning'},{type:'reasoning-delta',index:0,text:'PRIVATE_FAILED_REASONING'},{type:'block-end',index:0,block:{type:'reasoning',text:'PRIVATE_FAILED_REASONING'}},{type:'finish',reason:{kind:'error',failure:{code:'PI_AI_ERROR',message:recovery.REPETITION}}}];
const thought=()=>[{type:'block-start',index:0,blockType:'reasoning'},{type:'reasoning-delta',index:0,text:'still thinking'},{type:'block-end',index:0,block:{type:'reasoning',text:'still thinking'}},{type:'finish',reason:{kind:'max-tokens'}}];
const tool=(id,failed=false)=>[{type:'block-start',index:0,blockType:'tool-call'},{type:'tool-call-delta',index:0,id:ToolCallId(id),name:'work',argumentsDelta:'{}'},{type:'block-end',index:0,block:{type:'tool-call',id:ToolCallId(id),name:'work',arguments:'{}'}},failed?loop().at(-1):{type:'finish',reason:{kind:'tool-calls'}}];
const records=[];
async function run(label,script,verify,options={}) {
  const ctx=new Context();ctx.baseUrl=pathToFileURL(root).href+'/';
  let adapter,agent,toolRuns=0;
  try {
    await ctx.plugin(mods.get('@deepseek-ai/cordis-plugin-loader').default);
    ctx.loader.builtins.include=mods.get('@deepseek-ai/cordis-plugin-include').default;
    const entries=[...names.slice(3), '@augur/dsh-reasoning-recovery'];
    const modules=new Map(entries.map(name=>[name,name==='@augur/dsh-reasoning-recovery'?recovery:mods.get(name).default??mods.get(name)]));
    ctx.loader.internal={version:'v2',async import(specifier){assert(modules.has(specifier),specifier);return modules.get(specifier);}};
    const yaml=path.join(root,label+'.yml');await fs.writeFile(yaml,entries.map(name=>'- name: '+JSON.stringify(name)).join('\n').replaceAll('\n','\n')+'\n');
    await ctx.loader.create({name:'cordis:include',config:{path:pathToFileURL(yaml).href}});await ctx.loader.await();
    assert.deepEqual([...ctx.loader.entries()].filter(e=>!e.disabled&&!e.fiber).map(e=>e.options.name),[]);
    adapter=new class extends LlmAdapter {
      requests=[];
      providerRetryPolicy(){return resolveRetryPolicy({mode:'normal',maxRetries:1,retryableCodes:['RATE_LIMIT','SERVER'],backoff:{initialDelayMs:1,maxDelayMs:1,jitterRatio:0}},'test');}
      async resolveModel(provider,id){return {provider,id,name:id,reasoning:{efforts:['off','low','high','xhigh'].map(id=>({id,name:id})),defaultEffort:'xhigh'}};}
      async *stream(request){this.requests.push(request);const action=script[this.requests.length-1];assert(action,'unexpected extra request');const chunks=typeof action==='function'?await action(request,agent):action;for(const chunk of chunks)yield chunk;}
    };
    ctx.llm.registerAdapter(['augur-local','other'],adapter);
    ctx.tools.register({name:'work',description:'Record one completed action',parameters:{type:'object',properties:{},additionalProperties:false},output:{schema:{type:'object',properties:{done:{type:'boolean'}},required:['done'],additionalProperties:false},render:value=>[{type:'text',text:JSON.stringify(value)}]},async execute(){toolRuns++;return {done:true};}});
    agent=await ctx.agentLoop.create(SessionId(label),{provider:options.provider??'augur-local',model:'quasar-27b',reasoningEffort:'xhigh',maxTokens:249036});
    agent.followup(createUserMessage({content:[{type:'text',text:'ORIGINAL TASK: create the requested deliverable.'}],source:{kind:'user'}}));
    if(options.concurrent) await options.concurrent(ctx,agent);
    await agent.whenIdle();
    if(!adapter.requests.length) console.error(JSON.stringify(agent.session.snapshotEvents(),null,2));
    await verify({ctx,agent,adapter,toolRuns,events:agent.session.snapshotEvents()});
    records.push({case:label,status:'PASS',requests:adapter.requests.length});
  } finally { await ctx.fiber.dispose(); }
}
const notices=events=>events.filter(e=>e.type==='user/message'&&e.data.source?.kind==='augur-reasoning-recovery');
const turns=events=>events.filter(e=>e.type==='turn/start');
await run('same-turn-recovery',[loop(),text('recovered'),text('normal next turn')],async ({agent,adapter,events})=>{
 assert.equal(adapter.requests.length,2);assert.equal(turns(events).length,1);assert.equal(notices(events).length,1);
 assert.equal(adapter.requests[1].reasoningEffort,'low');assert.equal(adapter.requests[1].maxTokens,8192);
 const body=JSON.stringify(adapter.requests[1].messages);assert(body.includes('ORIGINAL TASK'));assert(body.includes('Harness recovery notice'));assert(!body.includes('PRIVATE_FAILED_REASONING'));
 assert.equal(events.filter(e=>e.type==='assistant/attempt').length,1);assert.equal(events.filter(e=>e.type==='assistant/message').length,1);
 agent.followup(createUserMessage({content:[{type:'text',text:'next task'}],source:{kind:'user'}}));await agent.whenIdle();
 assert.equal(adapter.requests[2].reasoningEffort,'xhigh');assert.equal(adapter.requests[2].maxTokens,249036);
});
await run('second-stage',[loop(),thought(),text('recovered without thinking')],({adapter,events})=>{assert.equal(notices(events).length,2);assert.equal(turns(events).length,1);assert.equal(adapter.requests[2].reasoningEffort,'off');assert.equal(adapter.requests[2].maxTokens,16384);assert.equal(events.filter(e=>e.type==='assistant/message').length,1);});
await run('bounded-fallback',[loop(),loop(),loop(),text('new user task')],async ({agent,adapter,events})=>{assert.equal(adapter.requests.length,3);assert.equal(notices(events).length,2);assert.equal(events.filter(e=>e.type==='assistant/message').length,0);agent.followup(createUserMessage({content:[{type:'text',text:'fresh user task'}],source:{kind:'user'}}));await agent.whenIdle();assert.equal(adapter.requests[3].reasoningEffort,'xhigh');assert.equal(adapter.requests[3].maxTokens,249036);});
await run('completed-tool-preserved',[tool('first'),loop(),tool('second'),text('done')],({adapter,events,toolRuns})=>{assert.equal(toolRuns,2);assert.equal(turns(events).length,1);assert(adapter.requests[2].messages.some(m=>m.role==='tool'));assert.equal(adapter.requests[3].reasoningEffort,'xhigh');assert.equal(adapter.requests[3].maxTokens,249036);});
await run('failed-tool-not-executed',[tool('partial',true),text('recovered')],({toolRuns,events})=>{assert.equal(toolRuns,0);assert.equal(notices(events).length,1);});
await run('truncated-recovery-not-success',[loop(),text('incomplete','max-tokens'),thought()],({events,adapter})=>{assert.equal(adapter.requests.length,3);assert.equal(events.filter(e=>e.type==='assistant/message').length,0);});
await run('normal-path',[text('ordinary answer')],({adapter,events})=>{assert.equal(adapter.requests.length,1);assert.equal(adapter.requests[0].reasoningEffort,'xhigh');assert.equal(notices(events).length,0);});
await run('unrelated-error',[{type:'finish',reason:{kind:'error',failure:{code:'PI_AI_ERROR',message:'Native request failed; no completion was committed.'}}}].map(c=>[c]),({adapter,events})=>{assert.equal(adapter.requests.length,1);assert.equal(notices(events).length,0);});
await run('other-provider',[loop()],({adapter,events})=>{assert.equal(adapter.requests.length,1);assert.equal(notices(events).length,0);},{provider:'other'});
await run('user-cancel',[(request,agent)=>{agent.cancel({kind:'user'});return loop();}],({adapter,events})=>{assert.equal(adapter.requests.length,1);assert.equal(notices(events).length,0);});
assert(recovery.isLoopFailure({code:'PI_AI_ERROR',message:'422 '+JSON.stringify({error:{code:'reasoning_repetition',message:recovery.REPETITION}})}));
assert(!recovery.isLoopFailure({code:'PI_AI_ERROR',message:'unrelated '+recovery.REPETITION}));
// Replay counters fold from durable notices; failed attempts and compaction do not reset them.
let state=recovery.initialState();state=recovery.fold(state,{type:'turn/start',data:{turn:3}});
for(const attempt of [1,2])state=recovery.fold(state,{type:'user/message',data:{source:{kind:'augur-reasoning-recovery',recovery:{turn:3,step:1,attempt,base:{provider:'augur-local',model:'quasar-27b'}}}}});
state=recovery.fold(state,{type:'compaction',data:{}});assert.equal(state.count,2);
let releaseFirst,enteredFirst;
const firstEntered=new Promise(resolve=>enteredFirst=resolve);
const firstRelease=new Promise(resolve=>releaseFirst=resolve);
await run('multiple-sessions-queue',[
 async()=>{enteredFirst();await firstRelease;return text('first finished');},text('next session finished')
],({adapter})=>{assert.equal(adapter.requests.length,2);assert.equal(adapter.requests[1].sessionId,'queued-third');},{concurrent:async(ctx)=>{
 await firstEntered;
 const pending=[];
 for(const id of ['queued-cancel','queued-third']){
  const a=await ctx.agentLoop.create(SessionId(id),{provider:'augur-local',model:'quasar-27b'});
  a.followup(createUserMessage({content:[{type:'text',text:'new session request'}],source:{kind:'user'}}));pending.push(a);
 }
 await new Promise(resolve=>setTimeout(resolve,20));
 pending[0].cancel({kind:'user'});releaseFirst();
 await Promise.all(pending.map(a=>a.whenIdle()));
}});
const busy={type:'finish',reason:{kind:'error',failure:{code:'INVALID_REQUEST',message:'409: '+JSON.stringify({message:'One generation is already active.',code:'single_user_busy'})}}};
await run('busy-drain-retry',[[{type:'usage',usage:{inputTokens:0,outputTokens:0}},busy],text('owner drained')],({adapter,events})=>{assert.equal(adapter.requests.length,2);assert.equal(notices(events).length,0);assert.equal(events.filter(e=>e.type==='assistant/attempt').length,0);});
const receipt={status:'PASS',inference:'scripted; installed real DSH Loader, AgentLoop, session, tools, retry and projections',cases:records};
if(process.argv[3])await fs.writeFile(process.argv[3],JSON.stringify(receipt,null,2)+'\n');
console.log(JSON.stringify(receipt,null,2));
// root is our unique temporary fixture only; never contains real user sessions.
await fs.rm(root,{recursive:true});
