'use strict';
const $ = (id) => document.getElementById(id);
// A key handed over by the launcher link is remembered for this origin only (127.0.0.1:<port>),
// so later visits to the bare address connect without typing it.
const store = (() => { try { return window.localStorage; } catch { return null; } })();
let key = new URLSearchParams(location.hash.slice(1)).get('key') || store?.getItem('augur-api-key') || '';
if (key) store?.setItem('augur-api-key', key);
history.replaceState(null, '', location.pathname);
let model = 'quasar-27b', active = null, request = null, visionAvailable = false;
let toolsAvailable = true, thinkingAvailable = true, thinkingParametersAvailable = true;
const chats = [], tools = [{type: 'function', function: {name: 'local_time', description: 'Read the local client date and time.', parameters: {type: 'object', properties: {}, additionalProperties: false}}}];
const icons = () => lucide.createIcons();
const headers = () => ({'Content-Type': 'application/json', Authorization: 'Bearer ' + key});
function localTime(now = new Date()) {
  const pad = value => String(value).padStart(2, '0'), offset = -now.getTimezoneOffset();
  const date = [now.getFullYear(), pad(now.getMonth() + 1), pad(now.getDate())].join('-');
  const time = [now.getHours(), now.getMinutes(), now.getSeconds()].map(pad).join(':');
  const zone = (offset < 0 ? '-' : '+') + pad(Math.floor(Math.abs(offset) / 60)) + ':' + pad(Math.abs(offset) % 60);
  return {local_time: date + 'T' + time + zone, time_zone: Intl.DateTimeFormat().resolvedOptions().timeZone};
}
function showError(error, notice = false) { $('error').textContent = error ? error.message || String(error) : ''; $('error').hidden = !error; $('error').classList.toggle('notice', !!error && notice); }
function setBusy(busy = !!request) {
  for (const id of ['new-chat', 'detail', 'budget', 'prompt', 'settings-button', 'export', 'files', 'thinking', 'thinking-language']) $(id).disabled = busy;
  $('send').disabled = busy || active.loadingImages;
  $('attach').disabled = busy || active.loadingImages || !visionAvailable;
  $('detail').disabled = busy || !visionAvailable;
  $('tools').disabled = busy || active.messages.length > 0 || !toolsAvailable;
  $('thinking').disabled = busy || !thinkingAvailable;
  $('thinking-language').disabled = busy || !thinkingAvailable;
  $('stop').hidden = !busy; $('send').hidden = busy; $('status').textContent = T(busy ? 'generating' : 'ready');
}
function addMessage(role, text = '', images = []) {
  const node = document.createElement('article'); node.className = 'message ' + role;
  const top = document.createElement('div'); top.className = 'message-top'; top.textContent = role === 'assistant' ? 'Augur' : T('you');
  const copy = document.createElement('button'); copy.className = 'icon'; copy.title = T('copy'); copy.setAttribute('aria-label', T('copy')); copy.innerHTML = '<i data-lucide="copy"></i>';
  const content = document.createElement('pre'); content.className = 'message-content'; content.textContent = text;
  copy.onclick = () => navigator.clipboard.writeText(content.textContent).catch(showError); top.append(copy); node.append(top);
  if (images.length) { const media = document.createElement('div'); media.className = 'message-images'; for (const image of images) { const img = document.createElement('img'); img.src = image; img.alt = T('attached'); media.append(img); } node.append(media); }
  node.append(content); $('messages').append(node); icons(); return {node, content};
}
function addReasoning(view, text) {
  if (!text) return;
  if (!view.reasoning) {
    const details = document.createElement('details'), summary = document.createElement('summary');
    summary.textContent = T('thinking'); details.append(summary);
    view.reasoning = document.createElement('pre'); view.reasoning.className = 'message-content';
    details.append(view.reasoning); view.node.insertBefore(details, view.content);
  }
  view.reasoning.textContent += text;
}
const chatTitle = chat => chat.title ?? T('new_chat');
function refreshChats() { $('chats').replaceChildren(); for (const chat of chats) { const button = document.createElement('button'); button.textContent = chatTitle(chat); button.setAttribute('aria-current', String(chat === active)); button.disabled = !!request; button.onclick = () => selectChat(chat); $('chats').append(button); } }
function saveDraft() {
  if (!active) return;
  active.draft.prompt = $('prompt').value; active.draft.detail = $('detail').value;
  active.thinking = $('thinking').checked; active.thinkingLanguage = $('thinking-language').value;
  if (!active.messages.length) active.tools = $('tools').checked;
}
function renderHistory() {
  $('messages').replaceChildren(); let view = null;
  for (const message of active.messages) {
    if (message.role === 'tool') { if (view) addToolResult(view, message.content); continue; }
    const parts = Array.isArray(message.content) ? message.content : null;
    view = addMessage(message.role, parts ? parts.filter(p => p.type === 'text').map(p => p.text).join('\n') : message.content || '', parts ? parts.filter(p => p.type === 'image_url').map(p => p.image_url.url) : []);
    if (message.reasoning_content) addReasoning(view, message.reasoning_content);
  }
}
function selectChat(chat) {
  if (request) return; saveDraft(); active = chat;
  $('chat-title').textContent = chatTitle(chat); $('bill').textContent = chat.bill; showError(chat.error);
  $('tools').checked = chat.tools && toolsAvailable; $('thinking').checked = chat.thinking && thinkingAvailable; $('thinking-language').value = chat.thinkingLanguage; $('prompt').value = chat.draft.prompt; $('detail').value = chat.draft.detail;
  renderHistory(); renderAttachments(); setBusy();
  document.body.classList.remove('sidebar-open'); refreshChats();
}
function newChat() {
  if (request) return;
  const chat = {title: null, messages: [], tools: $('tools').checked, thinking: $('thinking').checked, thinkingLanguage: $('thinking-language').value || 'en', bill: '', error: null, loadingImages: false, draft: {prompt: '', pending: [], detail: 'auto'}};
  chats.unshift(chat); selectChat(chat);
}
function renderAttachments() {
  const chat = active; $('attachments').replaceChildren();
  chat.draft.pending.forEach((item, index) => {
    const node = document.createElement('div'); node.className = 'attachment'; const image = document.createElement('img'); image.src = item.url; image.alt = item.name;
    const remove = document.createElement('button'); remove.className = 'icon'; remove.title = T('remove'); remove.setAttribute('aria-label', T('remove')); remove.innerHTML = '<i data-lucide="x"></i>'; remove.disabled = !!request || chat.loadingImages;
    remove.onclick = () => { if (request || chat.loadingImages) return; chat.draft.pending.splice(index, 1); renderAttachments(); };
    node.append(image, remove); $('attachments').append(node);
  }); icons();
}
function addToolResult(view, result) {
  const node = document.createElement('div'); node.className = 'tool-result'; node.textContent = 'local_time ' + result; view.node.append(node);
}
async function connect() {
  try { const response = await fetch('/v1/models', {headers: headers()}); if (response.status === 401 || response.status === 403) { store?.removeItem('augur-api-key'); throw new Error(T('not_connected')); } if (!response.ok) throw new Error(T('unavailable', {s: response.status})); const value = await response.json(); model = value.data[0].id; $('model').textContent = model; const caps = await fetch('/cq/capabilities', {headers: headers()}).then(r => r.json()); applyCapabilities(caps); applyOutputCeiling(caps.native_limits); setBusy(); showError(active.error); }
  catch (error) { $('status').textContent = T('disconnected'); showError(error); }
}
function applyCapabilities(caps) {
  visionAvailable = !!caps.vision_forward;
  // Older QUASAR servers retain their existing UI policy. An independent
  // profile may explicitly disable controls and omit unsupported parameters.
  toolsAvailable = caps.ui?.tools !== false;
  thinkingAvailable = caps.ui?.thinking !== false;
  thinkingParametersAvailable = caps.ui?.thinking_parameters !== false;
  if (!toolsAvailable) { $('tools').checked = false; for (const chat of chats) chat.tools = false; }
  if (!thinkingAvailable) { $('thinking').checked = false; for (const chat of chats) chat.thinking = false; }
  $('brand-model').textContent = caps.ui?.label || model;
  $('profile-notice').textContent = caps.ui?.notice || '';
  $('profile-notice').hidden = !caps.ui?.notice;
}
function cancellableDelay(signal, milliseconds) {
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const abort = () => { clearTimeout(timer); signal.removeEventListener('abort', abort); reject(signal.reason); };
    const timer = setTimeout(() => { signal.removeEventListener('abort', abort); resolve(); }, milliseconds);
    signal.addEventListener('abort', abort, {once: true});
  });
}
async function completionResponse(payload, signal) {
  const deadline = performance.now() + 2000, body = JSON.stringify(payload);
  while (true) {
    signal.throwIfAborted();
    const response = await fetch('/v1/chat/completions', {method: 'POST', headers: headers(), body, signal});
    if (response.ok) return response;
    const error = await response.json();
    // This exact response is returned before the single-owner admission lock.
    // Never replay an accepted stream, a transport failure or uncertain state.
    if (response.status !== 409 || error.error?.code !== 'single_user_busy' || performance.now() >= deadline)
      throw new Error(error.error?.message || 'HTTP ' + response.status);
    await cancellableDelay(signal, 10);
  }
}
// The output budget defaults to the server's ceiling for the KV capacity it was started with.
// A value the user typed is kept unless it exceeds that ceiling.
function applyOutputCeiling(limits) {
  const ceiling = limits && limits.initial_max_output_limit;
  if (!Number.isInteger(ceiling) || ceiling < 1) return;
  const input = $('budget'); input.max = ceiling;
  if (!input.dataset.edited || Number(input.value) > ceiling) input.value = ceiling;
}

async function stream(messages, view, signal, settings, interaction) {
  const started = performance.now(); let first = null, text = '', reasoning = '', finish = null, usage = null; const calls = new Map(); let done = false;
  const payload = {model, messages: messages.map(({reasoning_content, ...message}) => message), stream: true,
    stream_options: {include_usage: true}, max_completion_tokens: settings.budget};
  if (thinkingParametersAvailable) Object.assign(payload, {enable_thinking: settings.thinking, thinking_language: settings.thinkingLanguage});
  if (settings.tools && toolsAvailable) Object.assign(payload, {tools, parallel_tool_calls: false});
  const response = await completionResponse(payload, signal);
  const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = '';
  function line(raw) {
    if (!raw.startsWith('data: ')) return; const data = raw.slice(6).trim(); if (data === '[DONE]') { done = true; return; }
    if (done) throw new Error(T('after_done')); const event = JSON.parse(data); if (event.error) throw new Error(event.error.message || T('failed'));
    if (event.usage) usage = event.usage;
    for (const choice of event.choices || []) { const delta = choice.delta || {};
      if (delta.reasoning_content) {
        if (interaction.firstThinking === null) interaction.firstThinking = performance.now();
        reasoning += delta.reasoning_content; addReasoning(view, delta.reasoning_content);
      }
      if (delta.content) { if (first === null) first = performance.now(); if (interaction.firstContent === null) interaction.firstContent = first; text += delta.content; view.content.textContent = text; $('messages').scrollTop = $('messages').scrollHeight; }
      if (delta.tool_calls?.length && interaction.firstTool === null) interaction.firstTool = performance.now();
      for (const part of delta.tool_calls || []) { const call = calls.get(part.index) || {id: '', type: 'function', function: {name: '', arguments: ''}}; if (part.id) call.id = part.id; if (part.function?.name) call.function.name = part.function.name; call.function.arguments += part.function?.arguments || ''; calls.set(part.index, call); }
      if (choice.finish_reason) finish = choice.finish_reason;
    }
  }
  try { while (true) { const chunk = await reader.read(); buffer += decoder.decode(chunk.value || new Uint8Array(), {stream: !chunk.done}); let end; while ((end = buffer.indexOf('\n')) >= 0) { line(buffer.slice(0, end).replace(/\r$/, '')); buffer = buffer.slice(end + 1); } if (chunk.done) { if (buffer.trim()) line(buffer); break; } } }
  finally { reader.releaseLock(); }
  if (!done || !finish || !usage) throw new Error(T('incomplete'));
  const elapsed = (performance.now() - interaction.started) / 1000;
  signal.throwIfAborted();
  interaction.tokens += usage.completion_tokens;
  const since = at => ((at - interaction.started) / 1000).toFixed(2);
  active.bill = (interaction.firstThinking === null ? '' : T('bill_thinking', {s: since(interaction.firstThinking)}))
    + (interaction.firstTool === null ? '' : T('bill_tool', {s: since(interaction.firstTool)}))
    + (interaction.firstContent === null ? '' : T('bill_ttft', {s: since(interaction.firstContent)}))
    + T('bill_tokens', {n: interaction.tokens, s: elapsed.toFixed(2)});
  $('bill').textContent = active.bill;
  return {role: 'assistant', content: text, ...(reasoning ? {reasoning_content: reasoning} : {}), ...(calls.size ? {tool_calls: [...calls.values()]} : {})};
}
async function send(event) {
  event.preventDefault(); if (request || active.loadingImages) return; const prompt = $('prompt').value.trim(); if (!prompt) return;
  saveDraft(); const draft = {...active.draft, pending: active.draft.pending.slice()}, previousTitle = active.title, previousBill = active.bill;
  const settings = {tools: $('tools').checked, budget: Number($('budget').value),
    thinking: $('thinking').checked, thinkingLanguage: $('thinking-language').value || 'en'};
  const interaction = {started: performance.now(), firstContent: null, firstTool: null, firstThinking: null, tokens: 0};
  const input = draft.pending.length ? draft.pending.map(p => ({type: 'image_url', image_url: {url: p.url, detail: draft.detail}})).concat([{type: 'text', text: prompt}]) : prompt;
  const previousLength = active.messages.length;
  const user = {role: 'user', content: input}; active.tools = $('tools').checked; active.messages.push(user);
  if (active.title === null) { active.title = prompt.slice(0, 60); $('chat-title').textContent = active.title; }
  addMessage('user', prompt, draft.pending.map(p => p.url)); active.draft = {prompt: '', pending: [], detail: draft.detail}; renderAttachments(); $('prompt').value = ''; active.error = null; showError(null);
  request = new AbortController(); setBusy(true); refreshChats();
  try {
    let view = addMessage('assistant'); let answer = await stream(active.messages, view, request.signal, settings, interaction); active.messages.push(answer);
    if (answer.tool_calls) {
      for (const call of answer.tool_calls) { if (call.function.name !== 'local_time' || JSON.stringify(JSON.parse(call.function.arguments)) !== '{}') throw new Error(T('bad_tool')); const result = JSON.stringify(localTime()); active.messages.push({role: 'tool', tool_call_id: call.id, content: result}); addToolResult(view, result); }
      view = addMessage('assistant'); answer = await stream(active.messages, view, request.signal, settings, interaction); active.messages.push(answer); if (answer.tool_calls) throw new Error(T('tool_limit'));
    }
  } catch (error) {
    active.messages.splice(previousLength); active.title = previousTitle; active.bill = previousBill; active.draft = draft;
    active.error = error.name === 'AbortError' ? T('stopped') : error;
    $('chat-title').textContent = chatTitle(active); $('bill').textContent = active.bill; $('prompt').value = draft.prompt; $('detail').value = draft.detail;
    renderHistory(); renderAttachments(); showError(active.error);
  }
  finally { request = null; setBusy(false); renderAttachments(); refreshChats(); $('prompt').focus(); }
}
$('composer').onsubmit = send;
$('prompt').onkeydown = event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('composer').requestSubmit(); } };
$('attach').onclick = () => $('files').click();
$('files').onchange = async () => {
  const chat = active, files = Array.from($('files').files); $('files').value = '';
  if (request || chat.loadingImages || !files.length) return;
  try {
    if (files.some(file => !['image/png', 'image/jpeg'].includes(file.type) || file.size > 4 * 1024 * 1024)) throw new Error(T('bad_image'));
    const retained = chat.messages.flatMap(m => Array.isArray(m.content) ? m.content.filter(p => p.type === 'image_url') : []).length;
    if (chat.draft.pending.length + retained + files.length > 4) throw new Error(T('image_limit'));
    chat.loadingImages = true; setBusy(); renderAttachments();
    const additions = [];
    for (const file of files) {
      const url = await new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = () => reject(new Error(T('image_read'))); reader.readAsDataURL(file); });
      additions.push({name: file.name, url});
    }
    chat.draft.pending.push(...additions); chat.error = null;
  } catch (error) { chat.error = error; }
  finally { chat.loadingImages = false; if (chat === active) { renderAttachments(); showError(chat.error); setBusy(); } }
};
$('new-chat').onclick = newChat;
$('stop').onclick = () => request?.abort();
$('menu').onclick = () => document.body.classList.toggle('sidebar-open');
$('settings-button').onclick = () => { $('api-key').value = key; $('ui-language').value = LANG; $('settings').showModal(); };
$('ui-language').onchange = () => { setLang($('ui-language').value); $('chat-title').textContent = chatTitle(active); refreshChats(); setBusy(); };
$('budget').oninput = () => { $('budget').dataset.edited = '1'; };
$('settings').onclose = () => { const typed = $('api-key').value.trim(); if (typed !== key) { key = typed; if (key) store?.setItem('augur-api-key', key); } connect(); };
$('stop-service').onclick = async () => {
  // Answer before the drain: the backend replies 202 and then stops listening, so a waiting
  // page would lose the response it is waiting for.
  if (!confirm(T('stop_service_confirm'))) return;
  try {
    const response = await fetch('/augur/v1/stop', {method: 'POST', headers: headers(), body: ''});
    if (!response.ok) throw new Error('HTTP ' + response.status);
    $('settings').close(); request?.abort(); showError(T('stop_service_done'), true);
  } catch (error) { showError(T('stop_service_failed', {e: error.message || error})); }
};
$('export').onclick = () => { const blob = new Blob([JSON.stringify({model, messages: active.messages}, null, 2)], {type: 'application/json'}); const url = URL.createObjectURL(blob), a = document.createElement('a'); a.href = url; a.download = 'augur-conversation.json'; a.click(); URL.revokeObjectURL(url); };
applyI18n(); icons(); newChat(); connect();
