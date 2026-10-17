'use strict';
// Interface strings for the setup and service pages. Simplified Chinese is the default;
// the launcher stores the user's choice in preferences.json and reports it in /api/state.
const I18N={
'zh-CN':{
title:'Augur 本地服务配置',tagline:'本地推理引擎',connecting:'连接中',reading:'正在读取配置',open_chat:'打开聊天',open_monitor:'打开实时监控',
strip:'每轮由主模型一次验证 8 个位置。草稿按真实概率提案，英文思考平均每轮通过约 3.6 个 token。',
rail_foot:'QUASAR 27B NVFP4 主模型，DFlash 草稿，固定 B8。服务只监听本机回环地址。',
h1:'配置一次，留在本机。',intro:'选好模型目录和上下文容量，保存后引擎会在这台机器上加载，并给出聊天和监控入口。',
s1:'模型与草稿',model_dir:'QUASAR 主模型目录',model_dir_ph:'原始 NVFP4 主模型目录',draft_source_dir:'DFlash 原始 BF16 目录',draft_source_ph:'包含 config.json 与 model.safetensors',
quant_output_dir:'量化输出目录',browse:'浏览',quant_hint:'默认写入 release 的 user-data/models。Rust 原生 CPU 量化生成 Q8_0 权重，并内置 FP32 gate0.45；原始权重保留，已有目录不会被覆盖。',
quantize:'量化并选用',draft_quantized_dir:'正在选用的量化草稿',draft_quantized_ph:'量化成功后自动填写，也可以选择已有结果',
s2:'上下文与服务',context:'KV 上下文容量',output:'默认单次输出上限',port:'模型 API 端口',vision:'Vision 显存策略',vision_on_demand:'按需加载（默认）',vision_auto:'优先常驻，内存紧张时回收',
context_hint:'KV 容量由输入、历史、思考和答案共用，不是额外的输出额度。修改容量需要重启，已有对话不会被自动截断。请求自带输出额度时以请求为准，但不超过容量的 95%。',
output_max:'最大 {n} tokens',
s3:'本地持久化缓存',cache_toggle:'保存可恢复的输入前缀，重启后复用',cache_dir:'缓存目录',cache_hint:'缓存含对话 token 与模型状态，请放在私有本机磁盘。256K 建议预留至少 24 GiB；不兼容的旧缓存不会被自动删除。',
s4:'接入其他 Agent',model_name:'模型名',copy:'复制',show:'显示',hide:'隐藏',export_env:'写入当前用户环境变量 <code>AUGUR_API_KEY</code>、<code>AUGUR_BASE_URL</code>、<code>AUGUR_MODEL</code>',
env_on:'已写入当前用户环境变量。',env_off:'未写入环境变量；需要时复制密钥粘贴到 Agent。',
key_hint:'聊天页会自动登录，无需输入密钥。密钥只保存在本机 <code id="conn_key_file"></code> 和上面的环境变量里，不在发布包中，不会写进日志。新打开的终端和 Agent 才能读到环境变量。怀疑泄露时点“更换 API Key”，旧密钥立即失效。',
rekey:'更换 API Key',validate:'检查配置',restart:'保存并重启服务',stop:'关闭服务',quit:'退出 Augur',
footer:'关闭服务会中断当前生成并释放显存，配置页继续保留。退出 Augur 会先关闭服务，再结束后台启动器。关闭浏览器标签不会停止引擎；再次双击“启动服务.exe”会回到这个页面。',
language:'界面语言',
st_queued:'等待处理',st_validating:'校验模型与配置',st_stopping:'关闭旧服务并释放显存',st_saving:'保存配置',st_loading:'加载模型、DFlash 与 KV',st_monitor:'连接实时监控',st_ready:'服务就绪',st_quantizing:'检查与量化 DFlash',st_rekey:'更换 API Key',st_working:'正在处理',
failed:'操作失败',complete:'操作完成',elapsed_run:'已用时 {s} 秒',elapsed_total:'总耗时 {s} 秒',
detail_run:'当前阶段已用 {s} 秒 · 页面会自动更新，请保持启动器运行。',detail_failed:'请查看下方错误信息；原操作已结束。',detail_done:'本次操作已结束，服务入口已按当前状态更新。',
restarting:'正在启动 / 重启 · {s} 秒',busy:'正在处理',running:'服务运行中',stopped:'服务已关闭',
ph_stopped:'服务已关闭',ph_starting:'正在启动',ph_running:'服务运行中',ph_stopping:'正在释放显存',ph_startup_failed:'启动失败',ph_cleanup_unconfirmed:'资源清理待确认',ph_service_exited:'服务已退出',
config_at:'配置保存在：',lost:'连接中断',reconnecting:'正在重新连接配置服务',lost_detail:'。请保持启动器运行；连接恢复后自动更新。',request_failed:'请求失败',
copied:'已复制到剪贴板。',clipboard:'无法访问剪贴板，请点“显示”后手动复制。',
confirm_restart:'重启会中断当前生成并清空活跃 GPU 会话。继续吗？',confirm_rekey:'更换后旧密钥立即失效{r}。继续吗？',confirm_rekey_running:'，服务会重启并中断当前生成',
confirm_quit_running:'退出会关闭模型服务、中断当前生成并释放显存。继续吗？',confirm_quit:'退出 Augur 后台启动器？',confirm_stop:'关闭模型服务并中断当前生成？配置页将继续保留。',
quitting:'正在退出',quitting_phase:'正在关闭服务并退出',quit_done:'退出完成后可以关闭此页面；再次双击“启动服务.exe”即可重新打开。',
welcome_title:'选择界面语言',welcome_body:'之后可以随时在页面右上角切换。聊天页和监控页会跟随这个选择。',welcome_ok:'继续',
svc_title:'Augur · 连接服务',svc_chat:'连接模型聊天',svc_monitor:'连接实时监控',svc_wait:'正在获取当前服务入口…',svc_back:'返回配置页',
svc_expired:'配置页授权已失效，请从启动器重新打开配置页。',svc_not_started:'服务尚未启动。请回到配置页点击“保存并重启服务”；就绪后此页会自动连接。',svc_elapsed:'操作已用时 {s} 秒'},
en:{
title:'Augur local service setup',tagline:'Local inference engine',connecting:'Connecting',reading:'Reading settings',open_chat:'Open chat',open_monitor:'Open live monitor',
strip:'Each round the target model verifies 8 positions at once. The draft proposes from real probabilities; English thinking passes about 3.6 tokens per round.',
rail_foot:'QUASAR 27B NVFP4 target, DFlash draft, fixed B8. The service listens on loopback only.',
h1:'Set up once. Stays on this machine.',intro:'Choose the model folders and context capacity. After saving, the engine loads on this machine and the chat and monitor links appear.',
s1:'Model and draft',model_dir:'QUASAR target model folder',model_dir_ph:'Original NVFP4 target folder',draft_source_dir:'Original DFlash BF16 folder',draft_source_ph:'Contains config.json and model.safetensors',
quant_output_dir:'Quantization output folder',browse:'Browse',quant_hint:'Defaults to the release user-data/models folder. The native Rust CPU quantizer writes Q8_0 weights with an embedded FP32 gate0.45. Original weights are kept and existing folders are never overwritten.',
quantize:'Quantize and use',draft_quantized_dir:'Quantized draft in use',draft_quantized_ph:'Filled in after quantization, or choose an existing result',
s2:'Context and service',context:'KV context capacity',output:'Default output limit',port:'Model API port',vision:'Vision memory policy',vision_on_demand:'Load on demand (default)',vision_auto:'Keep resident, release under memory pressure',
context_hint:'Input, history, thinking and answer share the KV capacity; it is not an extra output allowance. Changing it needs a restart and never truncates existing chats. A request\'s own output limit wins, capped at 95% of capacity.',
output_max:'Up to {n} tokens',
s3:'Local persistent cache',cache_toggle:'Keep reusable input prefixes across restarts',cache_dir:'Cache folder',cache_hint:'The cache holds conversation tokens and model state; keep it on a private local disk. Allow at least 24 GiB for 256K. Incompatible old caches are never deleted automatically.',
s4:'Connect other agents',model_name:'Model',copy:'Copy',show:'Show',hide:'Hide',export_env:'Write user environment variables <code>AUGUR_API_KEY</code>, <code>AUGUR_BASE_URL</code>, <code>AUGUR_MODEL</code>',
env_on:'Written to your user environment.',env_off:'Not in the environment; copy the key into your agent when needed.',
key_hint:'The chat page signs in by itself. The key lives only in <code id="conn_key_file"></code> on this machine and in the variables above; it is not in the release package and never logged. Only terminals and agents opened afterwards see the variables. If you think it leaked, click "Replace API key"; the old key stops working at once.',
rekey:'Replace API key',validate:'Check settings',restart:'Save and restart',stop:'Stop service',quit:'Quit Augur',
footer:'Stopping the service interrupts generation and frees GPU memory; this page stays. Quit Augur stops the service, then ends the background launcher. Closing the browser tab does not stop the engine; double-click 启动服务.exe to come back here.',
language:'Language',
st_queued:'Queued',st_validating:'Checking model and settings',st_stopping:'Stopping and freeing GPU memory',st_saving:'Saving settings',st_loading:'Loading model, DFlash and KV',st_monitor:'Connecting monitor',st_ready:'Service ready',st_quantizing:'Checking and quantizing DFlash',st_rekey:'Replacing API key',st_working:'Working',
failed:'Operation failed',complete:'Operation complete',elapsed_run:'{s} s elapsed',elapsed_total:'{s} s total',
detail_run:'This step: {s} s · The page updates by itself; keep the launcher running.',detail_failed:'See the error below; the operation has ended.',detail_done:'Done. Service links reflect the current state.',
restarting:'Starting / restarting · {s} s',busy:'Working',running:'Service running',stopped:'Service stopped',
ph_stopped:'Service stopped',ph_starting:'Starting',ph_running:'Service running',ph_stopping:'Freeing GPU memory',ph_startup_failed:'Startup failed',ph_cleanup_unconfirmed:'Cleanup unconfirmed',ph_service_exited:'Service exited',
config_at:'Settings saved at: ',lost:'Disconnected',reconnecting:'Reconnecting to the launcher',lost_detail:'. Keep the launcher running; this page updates when it is back.',request_failed:'Request failed',
copied:'Copied to clipboard.',clipboard:'Clipboard unavailable; click "Show" and copy by hand.',
confirm_restart:'Restarting interrupts generation and clears active GPU sessions. Continue?',confirm_rekey:'The old key stops working at once{r}. Continue?',confirm_rekey_running:', and the service restarts, interrupting generation',
confirm_quit_running:'Quitting stops the model service, interrupts generation and frees GPU memory. Continue?',confirm_quit:'Quit the Augur background launcher?',confirm_stop:'Stop the model service and interrupt generation? This page stays open.',
quitting:'Quitting',quitting_phase:'Stopping the service and quitting',quit_done:'You can close this page once it finishes. Double-click 启动服务.exe to open it again.',
welcome_title:'Choose your language',welcome_body:'You can switch any time from the top right. The chat and monitor pages follow this choice.',welcome_ok:'Continue',
svc_title:'Augur · Connecting',svc_chat:'Connecting to chat',svc_monitor:'Connecting to the monitor',svc_wait:'Getting the current service link…',svc_back:'Back to setup',
svc_expired:'Setup authorization expired; open the setup page from the launcher again.',svc_not_started:'The service has not started. Go back and click "Save and restart"; this page connects once it is ready.',svc_elapsed:'Operation {s} s'}};
let LANG=(()=>{try{return localStorage.getItem('augur-lang')}catch{return null}})()==='en'?'en':'zh-CN';
function T(key,vars){let text=(I18N[LANG]||I18N['zh-CN'])[key]??I18N['zh-CN'][key]??key;for(const [k,v] of Object.entries(vars||{}))text=text.replace('{'+k+'}',v);return text;}
// Static markup: data-i18n (text), data-i18n-html (trusted strings above), data-i18n-ph (placeholder).
function applyI18n(){
  document.documentElement.lang=LANG;
  for(const el of document.querySelectorAll('[data-i18n]'))el.textContent=T(el.dataset.i18n);
  for(const el of document.querySelectorAll('[data-i18n-html]'))el.innerHTML=T(el.dataset.i18nHtml);
  for(const el of document.querySelectorAll('[data-i18n-ph]'))el.placeholder=T(el.dataset.i18nPh);
  const title=document.querySelector('title[data-i18n-title]');if(title)document.title=T(title.dataset.i18nTitle);
}
function setLang(value){LANG=value==='en'?'en':'zh-CN';try{localStorage.setItem('augur-lang',LANG);}catch{}applyI18n();}
