'use strict';
const params=new URLSearchParams(location.hash.slice(1));
const key=params.get('key')||sessionStorage.getItem('augur-setup-key')||'';
if(key)sessionStorage.setItem('augur-setup-key',key);
if(params.get('lang'))setLang(params.get('lang'));
history.replaceState(null,'',location.pathname);
const monitor=location.pathname==='/open/monitor';
applyI18n();
document.getElementById('title').textContent=T(monitor?'svc_monitor':'svc_chat');
document.getElementById('setup').href='/#key='+encodeURIComponent(key);
let progress=null,anchor=0;
setInterval(()=>{document.getElementById('elapsed').textContent=progress?T('svc_elapsed',{s:(progress.elapsed_s+(progress.status==='running'?(performance.now()-anchor)/1000:0)).toFixed(1)}):'';},100);
async function connect(){
  try{
    const response=await fetch('/api/state',{headers:{Authorization:'Bearer '+key},cache:'no-store',signal:AbortSignal.timeout(10000)});
    if(!response.ok)throw new Error(T('svc_expired'));
    const state=await response.json();progress=state.progress;anchor=performance.now();
    if(state.language_chosen&&state.language!==LANG){setLang(state.language);document.getElementById('title').textContent=T(monitor?'svc_monitor':'svc_chat');}
    const url=monitor?state.monitor_url:(state.running&&state.phase==='running'&&state.ready?state.ready.browser_url:null);
    const valid=monitor?/^http:\/\/127\.0\.0\.1:\d+\/$/:/^http:\/\/127\.0\.0\.1:\d+\/#key=[A-Za-z0-9_-]+$/;
    // The language travels in the fragment: chat and monitor are separate origins with their own storage.
    if(url&&valid.test(url)){location.replace(url+(monitor?'#':'&')+'lang='+LANG);return;}
    document.getElementById('message').textContent=state.operation?state.message:(state.error||T('svc_not_started'));
  }catch(error){document.getElementById('message').textContent=error.message;}
  setTimeout(connect,1000);
}
connect();
