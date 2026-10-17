'use strict';
const $=id=>document.getElementById(id),fmt=(n,d=1)=>Number.isFinite(n)?n.toLocaleString('en-US',{maximumFractionDigits:d,minimumFractionDigits:d}):'—',pct=n=>Number.isFinite(n)?fmt(n*100)+'%':'—';
const phaseText=()=>({failed_or_cancelled:T('请求失败或取消'),idle:T('待机'),prefill:'Prefill',decode:T('生成中'),offline:T('引擎离线'),running_without_token_observer:T('生成中 · 等待计数器'),starting:T('正在连接'),sampling_error:T('采样异常')});
// A browser-local visibility boundary keeps original receipts available for audits.
const historyStorageKey='augur.monitor.requests.cleared-through.v1';
let lastMetrics=null,clearedThrough=0;
const decodeLabel=mode=>({sampling:T('采样 · gate0.45 · 固定 B8'),greedy:T('贪婪'),unknown:T('历史未标注')}[mode]||T('历史未标注'));
function readHistoryBoundary(){try{const n=Number(localStorage.getItem(historyStorageKey));return Number.isFinite(n)&&n>0?n:0}catch{return 0}}
clearedThrough=readHistoryBoundary();
function visibleRequests(data){const mode=$('mode-filter').value||'all';return (data?.recent||[]).filter(row=>(mode==='all'||(row.decode_mode||'unknown')===mode)&&(!Number.isFinite(row.completed_unix_s)||row.completed_unix_s>clearedThrough))}
function clearRequestHistory(){
  if(!lastMetrics||!visibleRequests(lastMetrics).length)return;
  clearedThrough=Math.max(clearedThrough,lastMetrics.updated_unix_s||0,...(lastMetrics.recent||[]).map(row=>Number.isFinite(row.completed_unix_s)?row.completed_unix_s:0));
  let persisted=true;
  try{localStorage.setItem(historyStorageKey,String(clearedThrough))}catch{persisted=false}
  render(lastMetrics);
  $('history-status').textContent=persisted?T('历史列表已清空，新完成请求会继续显示。原始账单、模型对话和缓存保留。'):T('当前页面的历史列表已清空；浏览器禁止保存设置，刷新后可能恢复。');
}
$('clear-history').addEventListener('click',clearRequestHistory);
$('mode-filter').addEventListener('change',()=>{if(lastMetrics)render(lastMetrics)});
window.addEventListener('storage',event=>{if(event.key===historyStorageKey||event.key===null){clearedThrough=readHistoryBoundary();if(lastMetrics)render(lastMetrics)}});
function path(values,w,h,max){let result='',open=false;values.forEach((v,i)=>{if(!Number.isFinite(v)){open=false;return}const x=values.length<2?w:i*w/(values.length-1),y=h-Math.min(v/max,1)*h;result+=(open?'L':'M')+x.toFixed(1)+','+y.toFixed(1);open=true;});return result;}
function charts(rows,mode){const g=rows.map(x=>x.gpu),t=rows.map(x=>(x.decode_mode||'unknown')===mode?x.tps:null),top=Math.max(40,Math.ceil(Math.max(...t.filter(Number.isFinite),1)/20)*20);let grid='';for(let n=0;n<=4;n++){const y=12+n*48;grid+='<line x1="32" x2="854" y1="'+y+'" y2="'+y+'" stroke="#2c3441"/><text x="0" y="'+(y+4)+'">'+(100-n*25)+'</text><text x="867" y="'+(y+4)+'">'+Math.round(top*(1-n/4))+'</text>';}$('chart').innerHTML=grid+'<g transform="translate(32,12)"><path d="'+path(g,822,192,100)+'" fill="none" stroke="#7f9cc4" stroke-width="1.6"/><path d="'+path(t,822,192,top)+'" fill="none" stroke="#d4b062" stroke-width="2.4"/></g>';$('tps-mini').innerHTML='<path d="'+path(t.slice(-40),280,38,top)+'" fill="none" stroke="#d4b062" stroke-width="2"/>';}
function cell(text,sub){const td=document.createElement('td');td.textContent=text;if(sub){const el=document.createElement('small');el.textContent=sub;td.append(el)}return td;}
function prefillCell(row){const detail=row.prefill_rate_status==='not_executed'?T('未执行新 prefill · 状态复用'):Number.isFinite(row.prefill_tokens)?T('{n} 新 token / Native {s} s',{n:fmt(row.prefill_tokens,0),s:fmt(row.prefill_s,3)}):T('缺少可信 token 计数');return cell(Number.isFinite(row.prefill_tps)?fmt(row.prefill_tps)+' TPS':'—',detail);}
function renderPrefill(row){
  const executed=row?.prefill_rate_status==='measured'&&Number.isFinite(row.prefill_tps);
  $('prefill-tps').textContent=executed?fmt(row.prefill_tps):'—';
  $('prefill-detail').textContent=!row?T('等待完整请求账单'):row.prefill_rate_status==='not_executed'?T('完整缓存命中 · 未执行新 prefill'):executed?T('新计算输入 token / Native 执行时间'):T('本次请求缺少匹配的 Native 计数或计时');
  $('prefill-tokens').textContent=Number.isFinite(row?.prefill_tokens)?fmt(row.prefill_tokens,0)+' token':'—';
  $('prefill-time').textContent=Number.isFinite(row?.prefill_s)?fmt(row.prefill_s,3)+' s':'—';
}
function renderContext(context,stale){
  const c=context||{},valid=Number.isInteger(c.used_tokens)&&Number.isInteger(c.capacity_tokens)&&c.capacity_tokens>0&&c.used_tokens>=0&&c.used_tokens<=c.capacity_tokens&&!stale;
  const capacity=Number.isInteger(c.capacity_tokens)?c.capacity_tokens:null;
  $('context-used').textContent=valid?fmt(c.used_tokens,0):'—';
  $('context-capacity').textContent=' / '+fmt(capacity,0)+' token';
  $('context-percent').textContent=valid?pct(c.used_tokens/c.capacity_tokens):'—';
  const scope={prefill_input:T('输入已计入 · Prefill 中'),live:T('当前请求 · 实时增长'),last_completed:T('最近完成请求'),failed_or_cancelled:T('请求已取消 · 等待新账单')};
  $('context-scope').textContent=stale?T('采样已过期'):(!c.engine_online?T('引擎已停 · '):'')+(scope[c.scope]||T('等待上下文账单'));
  const bar=$('context-progress');bar.classList.toggle('compression-due',valid&&c.compression_due);bar.classList.toggle('unknown',!valid);
  $('context-fill').style.width=(valid?100*c.used_tokens/c.capacity_tokens:0)+'%';
  if(valid){bar.setAttribute('aria-valuenow',c.used_tokens);bar.setAttribute('aria-valuemax',c.capacity_tokens);bar.setAttribute('aria-valuetext',fmt(c.used_tokens,0)+' / '+fmt(c.capacity_tokens,0)+' token');}else{bar.removeAttribute('aria-valuenow');bar.removeAttribute('aria-valuemax');bar.setAttribute('aria-valuetext',T('上下文占用未知'));}
  $('context-remaining').textContent=!valid?T('等待可信上下文计数'):c.compression_due?T('已到压缩提示线 · 请由客户端在请求边界压缩'):T('距压缩提示线还剩 {n} token',{n:fmt(c.remaining_to_compression_tokens,0)});
  $('context-threshold').textContent=T('95% 压缩提示线 · {n} token',{n:fmt(c.compression_threshold_tokens,0)});
  $('context-memory').textContent=T('KV 显存预分配：')+(Number.isFinite(c.kv_allocated_bytes)?fmt(c.kv_allocated_bytes/1073741824,2)+' GiB':'—')+(!c.engine_online?T(' · 引擎已停止'):T(' · 与 token 使用比例分开计量'));
}
function render(data){lastMetrics=data;const age=Date.now()/1000-data.updated_unix_s,stale=age>8;const g=data.gpu||{},l=data.live,r=data.latest;$('connection').textContent=stale?T('采样已过期'):data.engine_online?T('引擎在线'):T('引擎离线');$('updated').textContent=new Date(data.updated_unix_s*1000).toLocaleTimeString();$('gpu-name').textContent=g.name||T('GPU 不可用');$('gpu').textContent=fmt(g.utilization,0);$('gpu-bar').style.width=Math.max(0,Math.min(g.utilization||0,100))+'%';$('gpu-detail').textContent=fmt(g.temperature_c,0)+' °C · '+fmt(g.power_w,0)+' W';$('memory').textContent=fmt(Number.isFinite(g.used_mib)?g.used_mib/1024:null,2);$('mem-bar').style.width=(g.used_mib/g.total_mib*100||0)+'%';$('mem-detail').textContent=T('总计 {t} GiB · 可用 {f} GiB',{t:fmt(Number.isFinite(g.total_mib)?g.total_mib/1024:null,1),f:fmt(Number.isFinite(g.free_mib)?g.free_mib/1024:null,2)});const vr=r?.vision_residency;if(vr&&data.engine_online){$('mem-detail').textContent+=' · Vision '+(vr.gpu_weights_resident?'GPU '+fmt(vr.resident_bytes/1048576,0)+' MiB':T('CPU 待命'))+T('（最近完成请求）');}const running=l&&['prefill','decode'].includes(l.status);$('decode-mode').textContent=decodeLabel(l?.decode_mode||r?.decode_mode||'unknown');$('tps').textContent=stale?'—':fmt(running?l.rolling_5s_tps:null);$('tps-detail').textContent=l?(running?T('原生 token / 近 5 秒墙钟'):T('待机 · 无活动生成')):T('等待 token 观测器连接');renderPrefill(r);renderContext(data.context,stale);$('phase').textContent=phaseText()[data.status]||data.status;$('tokens').textContent=fmt(running?l.output_tokens:null,0);$('elapsed').textContent=running?fmt(l.elapsed_s)+' s':'—';$('ttft').textContent=running&&Number.isFinite(l.ttft_s)?fmt(l.ttft_s*1000,0)+' ms':'—';$('request-id').textContent=running?l.request_id:(data.active?.request_id||T('暂无活动请求'));const body=$('requests');body.replaceChildren();const rows=visibleRequests(data);$('clear-history').disabled=rows.length===0;for(const row of rows){const tr=document.createElement('tr');tr.append(cell(new Date(row.completed_unix_s*1000).toLocaleTimeString(),row.request_id?.slice(-16)),cell(decodeLabel(row.decode_mode),row.sampling_preset||''),cell(fmt(row.output_tokens,0)),cell(fmt(row.native_generation_tps)),cell(fmt(row.http_output_tps)),prefillCell(row),cell(pct(row.input_cache_coverage),fmt(row.cached_tokens,0)+' / '+fmt(row.input_tokens,0)),cell(pct(row.accepted_over_proposed)),cell(pct(row.accepted_over_checked)),cell(fmt(row.mean_accepted_drafts,2),T('{n} 轮',{n:fmt(row.rounds,0)})),cell(row.stop));body.append(tr)}if(!rows.length){const tr=document.createElement('tr'),td=cell(clearedThrough?T('历史记录已清空，等待新的完成请求。'):T('暂无已完成请求。'));td.colSpan=11;td.className='empty-history';tr.append(td);body.append(tr)}charts(data.history||[],l?.decode_mode||r?.decode_mode||'unknown');$('errors').textContent=(data.errors||[]).join(' · ');}
async function poll(){try{const r=await fetch('/api/metrics',{cache:'no-store'});if(!r.ok)throw Error('HTTP '+r.status);render(await r.json());}catch(e){$('connection').textContent=T('监控连接中断');$('tps').textContent='—';renderContext(null,true);$('errors').textContent=String(e.message)}finally{setTimeout(poll,2000)}}$('ui-language').value=LANG;$('ui-language').onchange=()=>{setLang($('ui-language').value);if(lastMetrics)render(lastMetrics)};
applyI18n();poll();
