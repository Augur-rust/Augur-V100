'use strict';
// Monitor strings are written in Simplified Chinese (the default); EN maps each to English.
// T() returns the Chinese text unless English is selected. The launcher passes #lang=…,
// and this origin remembers the choice.
const EN={
'Augur · 运行监控':'Augur · Monitor','运行监控':'Monitor','正在连接':'Connecting','每一个 token 的去向。':'Where every token goes.',
'GPU、Prefill、Decode 与上下文账单，每 2 秒更新，只读观测。':'GPU, prefill, decode and context bills, refreshed every 2 seconds. Read-only.',
'最近采样':'Last sample','GPU 利用率':'GPU utilization','实时':'Live','显存占用':'Memory used','近 5 秒':'Last 5 s','最近完成':'Last completed',
'等待 token 观测器连接':'Waiting for the token observer','等待完整请求账单':'Waiting for a complete request bill','新计算输入':'New input computed','Native 耗时':'Native time',
'KV 上下文占用':'KV context usage','等待上下文账单':'Waiting for a context bill','等待输入 token 计数':'Waiting for the input token count','95% 压缩提示线':'95% compaction line',
'KV 显存预分配：—':'KV memory reserved: —','TPS 最近趋势':'Recent TPS trend','KV 上下文使用进度':'KV context usage',
'输入、历史、思考、工具与答案共用上下文；计数跟随当前请求增长，切换会话或压缩后可以下降。DSH 在请求边界按客户端估算判断压缩，实际触发时刻可能与引擎 token 账单略有差异；此进度条不会截断正在生成的内容。':'Input, history, thinking, tools and answer share the context. The count grows with the current request and can drop after switching chats or compacting. DSH decides compaction at request boundaries from its own estimate, so its timing can differ slightly from the engine bill. This bar never truncates a generation in progress.',
'GPU 与 Decode TPS 趋势':'GPU and decode TPS','最近五分钟 GPU 利用率和生成 TPS 曲线':'GPU utilization and decode TPS over the last five minutes','最近 5 分钟，GPU 左轴，Decode TPS 右轴':'Last 5 minutes. GPU on the left axis, decode TPS on the right','现在':'Now',
'当前请求':'Current request','解码模式待确认':'Decode mode pending','待机':'Idle','已输出 token':'Output tokens','请求耗时':'Elapsed','首 token 耗时':'First token','暂无活动请求':'No active request',
'TPS 使用原生校验后的 token 计数，含思考、不含 EOS。DFlash 接受率仅在完整收据提交后更新。':'TPS uses natively verified token counts, including thinking and excluding EOS. DFlash acceptance updates only after a complete receipt is committed.',
'最近完成请求':'Recent requests','解码模式':'Decode mode','全部（逐条分开）':'All (each row separate)','采样':'Sampling','贪婪':'Greedy','历史未标注':'Unlabeled (history)',
'缓存覆盖 = 复用输入 / 总输入':'Cache coverage = reused input / total input','清空历史记录':'Clear history','清空本浏览器中的请求列表；保留原始账单、模型对话和缓存':'Clear the request list in this browser; receipts, conversations and caches are kept',
'完成时间 / 请求':'Completed / request','输出 token':'Output tokens','原生生成 TPS':'Native decode TPS','HTTP 全程 TPS':'HTTP end-to-end TPS','输入缓存覆盖':'Input cache coverage','草稿接受率':'Draft acceptance','已检查草稿命中率':'Checked-draft hit rate','每轮接受草稿':'Accepted drafts / round','结束原因':'Stop reason',
'Prefill TPS = 实际新计算输入 token / Native 执行时间；不计缓存读写、工作区准备与释放、HTTP 开销。Native 执行包含输入上传、前向、head 与草稿特征捕获；完整缓存命中显示未执行，缺少匹配的原生计时显示 —。草稿接受率 = 接受 / 全部提出；已检查草稿命中率 = 接受 / 实际检查。前面拒绝后的未检查后缀不算预测错误。输入缓存覆盖率与草稿指标独立；固定 B8 每轮提出 7 个草稿。统计均为各条已完成请求。采样与贪婪分别标注，不合并接受率或 TPS；历史无模式证据的记录标为未标注。':'Prefill TPS = newly computed input tokens / native execution time, excluding cache I/O, workspace setup and release, and HTTP. Native execution covers input upload, forward pass, head and draft feature capture. A full cache hit shows as not executed; a missing native timer shows —. Draft acceptance = accepted / proposed; checked-draft hit rate = accepted / actually checked, so an unchecked suffix after a rejection is not a misprediction. Input cache coverage is independent of the draft metrics. Fixed B8 proposes 7 drafts per round. Every figure is per completed request. Sampling and greedy are labeled separately and never pooled; rows without mode evidence are marked unlabeled.',
'QUASAR 27B NVFP4，DFlash Q8 草稿，固定 B8':'QUASAR 27B NVFP4, DFlash Q8 draft, fixed B8','仅本机，不展示提示词、token 内容或密钥':'Local only. No prompts, token text or keys are shown',
'请求失败或取消':'Request failed or cancelled','生成中':'Generating','引擎离线':'Engine offline','生成中 · 等待计数器':'Generating · waiting for counters','采样异常':'Sampling error',
'采样 · gate0.45 · 固定 B8':'Sampling · gate0.45 · fixed B8',
'历史列表已清空，新完成请求会继续显示。原始账单、模型对话和缓存保留。':'History cleared; new requests keep appearing. Receipts, conversations and caches are kept.',
'当前页面的历史列表已清空；浏览器禁止保存设置，刷新后可能恢复。':'History cleared on this page. The browser blocked saving the setting, so it may return after a reload.',
'未执行新 prefill · 状态复用':'No new prefill · state reused','{n} 新 token / Native {s} s':'{n} new tokens / native {s} s','缺少可信 token 计数':'No trusted token count',
'完整缓存命中 · 未执行新 prefill':'Full cache hit · no new prefill','新计算输入 token / Native 执行时间':'New input tokens / native execution time','本次请求缺少匹配的 Native 计数或计时':'This request has no matching native count or timer',
'输入已计入 · Prefill 中':'Input counted · prefilling','当前请求 · 实时增长':'Current request · live','请求已取消 · 等待新账单':'Request cancelled · waiting for a new bill',
'采样已过期':'Sample stale','引擎已停 · ':'Engine stopped · ','上下文占用未知':'Context usage unknown','等待可信上下文计数':'Waiting for a trusted context count',
'已到压缩提示线 · 请由客户端在请求边界压缩':'At the compaction line · compact on the client at a request boundary','距压缩提示线还剩 {n} token':'{n} tokens to the compaction line',
'95% 压缩提示线 · {n} token':'95% compaction line · {n} tokens','KV 显存预分配：':'KV memory reserved: ',' · 引擎已停止':' · engine stopped',' · 与 token 使用比例分开计量':' · measured separately from token usage',
'引擎在线':'Engine online','GPU 不可用':'GPU unavailable','总计 {t} GiB · 可用 {f} GiB':'{t} GiB total · {f} GiB free','（最近完成请求）':' (last completed request)','CPU 待命':'CPU standby',
'原生 token / 近 5 秒墙钟':'Native tokens / last 5 s wall clock','待机 · 无活动生成':'Idle · no active generation','{n} 轮':'{n} rounds',
'历史记录已清空，等待新的完成请求。':'History cleared; waiting for new completed requests.','暂无已完成请求。':'No completed requests yet.','监控连接中断':'Monitor disconnected','界面语言':'Language'};
// Storage may be blocked by browser privacy settings; the page still works in the default language.
const langGet=()=>{try{return localStorage.getItem('augur-lang')}catch{return null}};
const langSet=value=>{try{localStorage.setItem('augur-lang',value)}catch{}};
let LANG=(()=>{const hinted=new URLSearchParams(location.hash.slice(1)).get('lang');if(hinted)langSet(hinted==='en'?'en':'zh-CN');return (hinted||langGet())==='en'?'en':'zh-CN'})();
if(location.hash)history.replaceState(null,'',location.pathname);
function T(zh,vars){let text=LANG==='en'?(EN[zh]??zh):zh;for(const [k,v] of Object.entries(vars||{}))text=text.replace('{'+k+'}',v);return text}
// Static markup is authored in Chinese; remember each original so switching back is exact.
const originals=new WeakMap();
function applyI18n(){
  document.documentElement.lang=LANG;document.title=T('Augur · 运行监控');
  const walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
  for(let node=walker.nextNode();node;node=walker.nextNode()){
    if(node.parentElement?.closest('[data-live]'))continue;
    if(!originals.has(node))originals.set(node,node.nodeValue);
    const zh=originals.get(node),core=zh.trim();if(!core||!(core in EN))continue;
    node.nodeValue=zh.replace(core,T(core));}
  for(const el of document.querySelectorAll('[title],[aria-label]'))for(const name of ['title','aria-label']){
    if(!el.hasAttribute(name))continue;const key=name==='title'?'i18nTitle':'i18nAria';if(!el.dataset[key])el.dataset[key]=el.getAttribute(name);el.setAttribute(name,T(el.dataset[key]));}
}
function setLang(value){LANG=value==='en'?'en':'zh-CN';langSet(LANG);applyI18n()}
