'use strict';
// Chat page strings. Simplified Chinese is the default; the launcher passes the saved choice
// as #lang=… and this origin remembers it. Thinking language is a separate model setting.
const I18N = {
  'zh-CN': {
    new_chat: '新对话', conversations: '对话列表', local_gpu: '本机 GPU', experimental: '实验性', connecting: '正在连接',
    export: '导出对话', settings: '设置', conversation: '对话内容', message: '消息', placeholder: '给 Augur 发消息',
    attach: '添加图片', tools: '工具', think: '思考', image_detail: '图片细节', detail_auto: '自动', detail_high: '高', detail_low: '低',
    stop: '停止生成', send: '发送', api_key: 'API Key', output_tokens: '输出 token 上限', thinking_language: '思考语言',
    thinking_en: '英文（默认）', thinking_zh: '中文', ui_language: '界面语言', done: '完成',
    stop_service: '关闭服务', stop_service_confirm: '关闭服务会中断当前生成、释放显存并退出后端；此页面将无法再发送消息，需要重新双击“启动服务.exe”。确定关闭吗？',
    stop_service_done: '已请求关闭服务。GPU 释放完成后后端会退出，请查看监控页确认显存已释放。',
    stop_service_failed: '关闭请求未能送达：{e}',
    generating: '生成中', ready: '就绪', disconnected: '未连接', you: '你', copy: '复制消息', attached: '附加图片', thinking: '思考过程', remove: '移除图片',
    not_connected: '未连接：请双击“启动服务.exe”，或从配置页点击“打开聊天”。也可以在设置里粘贴密钥。',
    unavailable: '服务暂不可用（{s}）', after_done: '完成后仍收到响应', failed: '生成失败', incomplete: '响应不完整',
    bill_thinking: '思考 {s}s  ', bill_tool: '工具 {s}s  ', bill_ttft: '首字 {s}s  ', bill_tokens: '{n} tokens / {s}s',
    bad_tool: '不支持的工具调用', tool_limit: '已达到工具轮数上限', stopped: '已停止生成',
    bad_image: '仅支持 4 MiB 以内的 PNG 或 JPEG', image_limit: '最多 4 张图片', image_read: '图片读取失败' },
  en: {
    new_chat: 'New Chat', conversations: 'Conversations', local_gpu: 'Local GPU', experimental: 'Experimental', connecting: 'Connecting',
    export: 'Export Conversation', settings: 'Settings', conversation: 'Conversation', message: 'Message', placeholder: 'Message Augur',
    attach: 'Attach Images', tools: 'Tools', think: 'Think', image_detail: 'Image Detail', detail_auto: 'Auto', detail_high: 'High', detail_low: 'Low',
    stop: 'Stop Generation', send: 'Send Message', api_key: 'API Key', output_tokens: 'Output Tokens', thinking_language: 'Thinking Language',
    thinking_en: 'English (default)', thinking_zh: 'Chinese', ui_language: 'Interface Language', done: 'Done',
    stop_service: 'Stop Service', stop_service_confirm: 'Stopping interrupts the current generation, frees the GPU and exits the backend. This page can no longer send messages until you start the launcher again. Stop the service?',
    stop_service_done: 'Stop requested. The backend exits once the GPU is released; check the monitor page to confirm the memory is free.',
    stop_service_failed: 'The stop request could not be delivered: {e}',
    generating: 'Generating', ready: 'Ready', disconnected: 'Disconnected', you: 'You', copy: 'Copy Message', attached: 'Attached Image', thinking: 'Thinking', remove: 'Remove Image',
    not_connected: 'Not connected: double-click 启动服务.exe or open Chat from the setup page. You can also paste the key in Settings.',
    unavailable: 'Service unavailable ({s})', after_done: 'Response After Completion', failed: 'Generation Failed', incomplete: 'Incomplete Response',
    bill_thinking: 'Thinking {s}s  ', bill_tool: 'Tool {s}s  ', bill_ttft: 'TTFT {s}s  ', bill_tokens: '{n} tokens / {s}s',
    bad_tool: 'Unsupported Tool Call', tool_limit: 'Tool Round Limit Reached', stopped: 'Generation Stopped',
    bad_image: 'Bounded PNG Or JPEG Required', image_limit: 'Four Image Limit', image_read: 'Image Read Failed' }
};
const langStore = (() => { try { return window.localStorage; } catch { return null; } })();
let LANG = (() => { const hinted = new URLSearchParams(location.hash.slice(1)).get('lang'); const value = hinted || langStore?.getItem('augur-lang'); return value === 'en' ? 'en' : 'zh-CN'; })();
if (new URLSearchParams(location.hash.slice(1)).get('lang')) (() => { try { langStore?.setItem('augur-lang', LANG); } catch {} })();
function T(key, vars) { let text = (I18N[LANG] || I18N['zh-CN'])[key] ?? key; for (const [k, v] of Object.entries(vars || {})) text = text.replace('{' + k + '}', v); return text; }
function applyI18n() {
  document.documentElement.lang = LANG;
  for (const el of document.querySelectorAll('[data-i18n]')) el.textContent = T(el.dataset.i18n);
  for (const el of document.querySelectorAll('[data-i18n-title]')) { el.title = T(el.dataset.i18nTitle); el.setAttribute('aria-label', T(el.dataset.i18nTitle)); }
  for (const el of document.querySelectorAll('[data-i18n-ph]')) el.placeholder = T(el.dataset.i18nPh);
  for (const el of document.querySelectorAll('[data-i18n-label]')) el.setAttribute('aria-label', T(el.dataset.i18nLabel));
}
function setLang(value) { LANG = value === 'en' ? 'en' : 'zh-CN'; (() => { try { langStore?.setItem('augur-lang', LANG); } catch {} })(); applyI18n(); }
