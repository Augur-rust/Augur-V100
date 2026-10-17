'use strict';
const $=id=>document.getElementById(id);
let key=new URLSearchParams(location.hash.slice(1)).get('key')||sessionStorage.getItem('augur-setup-key')||'';
if(key)sessionStorage.setItem('augur-setup-key',key);
history.replaceState(null,'',location.pathname);
const fields=['model_dir','draft_source_dir','draft_quantized_dir','quant_output_dir','context_tokens','default_output_tokens','port','vision_residency','cache_enabled','cache_dir','export_env'];
const stages=['queued','validating','stopping','saving','loading','monitor','ready','quantizing','rekey'];
let languageKnown=false;
let loaded=false,busy=false,running=false,lastOperation=null,progress=null,progressAt=0,submitting=false,refreshing=false,epoch=0;
async function api(path,body){const options={headers:{Authorization:'Bearer '+key},cache:'no-store',signal:AbortSignal.timeout(15000)};if(body!==undefined){options.method='POST';options.headers['Content-Type']='application/json';options.body=JSON.stringify(body);}const response=await fetch(path,options);const value=await response.json();if(!response.ok)throw new Error(value.error||T('request_failed'));return value;}
function config(){const value={schema:1};for(const id of fields){const input=$(id);value[id]=input.type==='checkbox'?input.checked:input.type==='number'||id==='context_tokens'?Number(input.value):input.value;}return value;}
function fill(value){for(const id of fields){if($(id).type==='checkbox')$(id).checked=value[id];else $(id).value=value[id];}outputLimit();}
// The output ceiling follows the KV choice (95% of it). Choosing a capacity selects that ceiling;
// the user may still lower the default by hand.
function outputLimit(){const limit=Math.floor(Number($('context_tokens').value)*95/100);$('default_output_tokens').max=limit;$('output_hint').textContent=T('output_max',{n:limit.toLocaleString('en-US')});if(Number($('default_output_tokens').value)>limit)$('default_output_tokens').value=limit;}
function matchCapacity(){outputLimit();$('default_output_tokens').value=$('default_output_tokens').max;}
function controls(){for(const input of document.querySelectorAll('form input,form select,form button'))input.disabled=busy||submitting||!loaded;$('stop').disabled=busy||submitting||!running;}
function setProgress(value){progress=value;progressAt=performance.now();drawProgress();}
function drawProgress(){
  $('progress').hidden=!progress;
  if(!progress){$('restart').textContent=T('restart');return;}
  const active=progress.status==='running',elapsed=progress.elapsed_s+(active?(performance.now()-progressAt)/1000:0);
  $('spinner').hidden=!active;
  $('progress_stage').textContent=active?T(stages.includes(progress.stage)?'st_'+progress.stage:'st_working'):T(progress.status==='failed'?'failed':'complete');
  $('elapsed').textContent=T(active?'elapsed_run':'elapsed_total',{s:elapsed.toFixed(1)});
  $('progress_detail').textContent=active?T('detail_run',{s:(progress.stage_elapsed_s+(performance.now()-progressAt)/1000).toFixed(1)}):T(progress.status==='failed'?'detail_failed':'detail_done');
  $('restart').textContent=active&&progress.action==='restart'?T('restarting',{s:elapsed.toFixed(1)}):T('restart');
}
function hideLinks(){for(const id of ['chat','monitor']){$(id).hidden=true;$(id).removeAttribute('href');}}
function serviceLink(id,available){const link=$(id);link.hidden=!available;if(available)link.href='/open/'+id+'#key='+encodeURIComponent(key)+'&lang='+LANG;else link.removeAttribute('href');}
async function refresh(){
  if(refreshing||submitting)return;
  refreshing=true;const revision=epoch;
  try{
    const s=await api('/api/state');if(submitting||revision!==epoch)return;
    running=s.running;busy=!!s.operation;
    if(!languageKnown){languageKnown=true;if(s.language_chosen){if(s.language!==LANG)setLang(s.language);$('language').value=LANG;}else welcome();}
    if(!loaded||((lastOperation==='quantize'||lastOperation==='restart')&&!busy&&!s.error))fill(s.config);
    loaded=true;lastOperation=s.operation;setProgress(s.progress||null);
    $('badge').textContent=T(busy?'busy':running?'running':'stopped');
    $('phase').textContent=T('ph_'+s.phase)!=='ph_'+s.phase?T('ph_'+s.phase):s.phase;
    $('message').textContent=s.message;$('error').textContent=s.error||'';$('config_location').textContent=T('config_at')+s.config_path;
    const url=s.ready&&s.ready.browser_url;
    serviceLink('chat',!!(running&&s.phase==='running'&&url&&/^http:\/\/127\.0\.0\.1:\d+\/#key=/.test(url)));
    serviceLink('monitor',!!(s.monitor_url&&/^http:\/\/127\.0\.0\.1:\d+\/$/.test(s.monitor_url)));
    $('monitor_error').textContent=s.monitor_error||'';controls();
    if(!busy&&(s.progress?.action!==shownConnection||!connection))loadConnection(s.progress?.action);
  }catch(error){
    if(submitting||revision!==epoch)return;
    hideLinks();loaded=false;$('badge').textContent=T('lost');$('phase').textContent=T('reconnecting');
    $('error').textContent=error.message+T('lost_detail');controls();
  }finally{refreshing=false;}
}
// The key is fetched behind the setup bearer and kept out of the DOM until shown or copied.
let connection=null,shownConnection=null,revealed=false;
async function loadConnection(after){shownConnection=after;try{connection=await api('/api/connection');}catch{return;}
  drawConnection();}
function drawConnection(){if(!connection)return;
  $('conn_base_url').textContent=connection.base_url;$('conn_model').textContent=connection.model;$('conn_key_file').textContent=connection.key_file;
  $('conn_key').textContent=revealed?connection.api_key:'••••••••';
  $('env_status').textContent=T(connection.env_exported?'env_on':'env_off');
  $('reveal_key').textContent=T(revealed?'hide':'show');}
function copyText(id){const text=id==='conn_key'?connection?.api_key:$(id).textContent;if(!text)return;navigator.clipboard.writeText(text).then(()=>{$('message').textContent=T('copied');},()=>{$('error').textContent=T('clipboard');});}
async function action(path,body){
  epoch++;submitting=true;busy=true;controls();$('error').textContent='';
  const name=({'/api/save-restart':'restart','/api/validate':'validate','/api/quantize':'quantize','/api/stop':'stop'})[path];
  setProgress({action:name,stage:'queued',status:'running',elapsed_s:0,stage_elapsed_s:0});
  try{const result=await api(path,body);lastOperation=result.operation;}
  catch(error){$('error').textContent=error.message;busy=false;setProgress({...progress,status:'failed',elapsed_s:(performance.now()-progressAt)/1000});}
  finally{submitting=false;controls();}
  await refresh();
}
$('context_tokens').addEventListener('change',matchCapacity);
$('configuration').addEventListener('submit',event=>{event.preventDefault();if(running&&!confirm(T('confirm_restart')))return;action('/api/save-restart',config());});
for(const button of document.querySelectorAll('[data-copy]'))button.onclick=()=>copyText(button.dataset.copy);
$('reveal_key').onclick=()=>{revealed=!revealed;$('reveal_key').textContent=T(revealed?'hide':'show');$('reveal_key').setAttribute('aria-pressed',String(revealed));$('conn_key').textContent=revealed&&connection?connection.api_key:'••••••••';};
$('rekey').onclick=()=>{if(confirm(T('confirm_rekey',{r:running?T('confirm_rekey_running'):''}))){connection=null;action('/api/rekey',{});}};
$('validate').onclick=()=>action('/api/validate',config());
$('quantize').onclick=()=>action('/api/quantize',config());
$('quit').onclick=async()=>{if(!confirm(T(running?'confirm_quit_running':'confirm_quit')))return;try{await api('/api/quit',{});epoch++;submitting=true;loaded=false;controls();hideLinks();$('badge').textContent=T('quitting');$('phase').textContent=T('quitting_phase');$('message').textContent=T('quit_done');$('error').textContent='';}catch(error){$('error').textContent=error.message;}};
$('stop').onclick=()=>{if(confirm(T('confirm_stop')))action('/api/stop',{});};
for(const button of document.querySelectorAll('[data-browse]'))button.onclick=async()=>{try{button.disabled=true;const v=await api('/api/browse',{});if(v.path)$(button.dataset.browse).value=v.path;}catch(error){$('error').textContent=error.message;}finally{controls();}};
// Language: chosen once on first start (Simplified Chinese preselected), then a saved preference.
// The server keeps it so launcher messages, chat and monitor follow the same choice.
async function chooseLanguage(value){setLang(value);$('language').value=LANG;outputLimit();drawProgress();drawConnection();
  try{await api('/api/language',{language:LANG});}catch(error){$('error').textContent=error.message;}refresh();}
function welcome(){const dialog=$('welcome');if(!dialog.showModal){chooseLanguage(LANG);return;}
  for(const input of dialog.querySelectorAll('input[name=welcome_lang]')){input.checked=input.value===LANG;input.onchange=()=>{setLang(input.value);};}
  dialog.onclose=()=>chooseLanguage(dialog.querySelector('input[name=welcome_lang]:checked')?.value||'zh-CN');dialog.showModal();}
$('language').onchange=()=>chooseLanguage($('language').value);
$('language').value=LANG;applyI18n();controls();refresh();setInterval(refresh,1000);setInterval(drawProgress,100);
