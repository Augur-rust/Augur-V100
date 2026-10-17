/** One cancellable FIFO for all local Augur requests, including compaction. */
export class SingleFlight {
  active=false;
  waiting=[];
  disposed=false;
  acquire(signal) {
    if(this.disposed || signal?.aborted) return Promise.reject(signal?.reason ?? new Error('Augur request queue stopped'));
    return new Promise((resolve,reject)=>{
      const ticket={signal,resolve,reject,abort:null};
      ticket.abort=()=>{const i=this.waiting.indexOf(ticket);if(i>=0)this.waiting.splice(i,1);reject(signal.reason);};
      signal?.addEventListener('abort',ticket.abort,{once:true});
      this.waiting.push(ticket);this.pump();
    });
  }
  pump() {
    if(this.active || this.disposed) return;
    const ticket=this.waiting.shift();if(!ticket)return;
    ticket.signal?.removeEventListener('abort',ticket.abort);
    if(ticket.signal?.aborted){ticket.reject(ticket.signal.reason);this.pump();return;}
    this.active=true;let released=false;
    ticket.resolve(()=>{if(released)return;released=true;this.active=false;this.pump();});
  }
  close() {
    this.disposed=true;
    for(const ticket of this.waiting.splice(0)){ticket.signal?.removeEventListener('abort',ticket.abort);ticket.reject(new Error('Augur request queue stopped'));}
  }
}
export function isBusyFailure(chunk) {
  if(chunk.type!=='finish' || chunk.reason.kind!=='error')return false;
  const failure=chunk.reason.failure;
  if(!['INVALID_REQUEST','PI_AI_ERROR','single_user_busy'].includes(failure.code))return false;
  try {const value=JSON.parse(failure.message.trim().replace(/^409\s*:?\s*/,''));const body=value.error??value;return body.code==='single_user_busy'&&body.message==='One generation is already active.';}catch{return false;}
}
function wait(ms,signal) {
  if(signal?.aborted)return Promise.reject(signal.reason);
  return new Promise((resolve,reject)=>{
    const done=()=>{signal?.removeEventListener('abort',abort);resolve();};
    const timer=setTimeout(done,ms);
    const abort=()=>{clearTimeout(timer);signal?.removeEventListener('abort',abort);reject(signal.reason);};
    signal?.addEventListener('abort',abort,{once:true});
  });
}
export async function* serialize(queue,options,next,{busyWaitMs=30000,delayMs=250}={}) {
  const release=await queue.acquire(options.signal);
  try {
    const deadline=Date.now()+busyWaitMs;
    while(true) {
      options.signal?.throwIfAborted();
      let yielded=false,retry=false,pending=[];
      for await(const chunk of next()) {
        // Never replay a request which published output. A busy response is
        // pre-admission only; wait briefly for an external owner's drain.
        if(!yielded && isBusyFailure(chunk) && Date.now()<deadline){retry=true;break;}
        if(!yielded && chunk.type==='usage'){pending.push(chunk);if(pending.length>8){for(const frame of pending)yield frame;pending=[];yielded=true;}continue;}
        for(const frame of pending)yield frame;pending=[];
        yielded=true;yield chunk;
      }
      if(!retry){for(const frame of pending)yield frame;return;}
      await wait(Math.min(delayMs,Math.max(0,deadline-Date.now())),options.signal);
    }
  } finally {release();}
}
