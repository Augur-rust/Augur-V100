import assert from 'node:assert/strict';
import {SingleFlight,serialize,isBusyFailure} from './single-flight.mjs';
const busy={type:'finish',reason:{kind:'error',failure:{code:'INVALID_REQUEST',message:'409: '+JSON.stringify({message:'One generation is already active.',code:'single_user_busy'})}}};
assert(isBusyFailure(busy));
const queue=new SingleFlight(),first=await queue.acquire();
const cancel=new AbortController();let entered=false;
const second=queue.acquire(cancel.signal);const rejected=assert.rejects(second);
const third=queue.acquire().then(release=>{entered=true;release();});
cancel.abort(new Error('user stopped queued request'));await rejected;assert(!entered);first();await third;assert(!queue.active);
let calls=0;async function* adapter(){calls++;if(calls===1)yield {type:'usage',usage:{inputTokens:0,outputTokens:0}};yield calls===1?busy:{type:'finish',reason:{kind:'stop'}};}
const output=[];for await(const frame of serialize(queue,{},adapter,{delayMs:1,busyWaitMs:100}))output.push(frame);
assert.equal(calls,2);assert.equal(output.length,1);assert.equal(output[0].reason.kind,'stop');assert(!queue.active);
calls=0;async function* partial(){calls++;yield {type:'text-delta',index:0,text:'partial'};yield busy;}
for await(const _ of serialize(queue,{},partial,{delayMs:1,busyWaitMs:100})){}assert.equal(calls,1);
let maxActive=0,active=0;async function* work(){active++;maxActive=Math.max(maxActive,active);try{await new Promise(r=>setTimeout(r,10));yield {type:'finish',reason:{kind:'stop'}};}finally{active--;}}
await Promise.all(Array.from({length:5},async()=>{for await(const _ of serialize(queue,{},work)){};}));assert.equal(maxActive,1);
const stop=new AbortController();async function* occupied(){stop.abort(new Error('user stop'));yield busy;}
await assert.rejects(async()=>{for await(const _ of serialize(queue,{signal:stop.signal},occupied)){};});assert(!queue.active);
const release=await queue.acquire(),pending=queue.acquire();const closed=assert.rejects(pending);queue.close();await closed;release();
console.log(JSON.stringify({status:'PASS',cases:['FIFO','queued cancellation','pre-admission 409 retry','no replay after output','five requests serialize','busy wait cancellation','disposal drains waiters']}));
