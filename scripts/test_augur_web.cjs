'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const test = require('node:test');

class Element {
  constructor() { this.children = []; this.value = ''; this.textContent = ''; this.disabled = false; this.hidden = false; this.checked = false; this.files = []; this.className = ''; this.classList = {toggle() {}, add() {}, remove() {}}; }
  append(...nodes) { this.children.push(...nodes); }
  insertBefore(node, before) { const index = this.children.indexOf(before); this.children.splice(index < 0 ? this.children.length : index, 0, node); }
  replaceChildren(...nodes) { this.children = nodes; }
  setAttribute(name, value) { this[name] = value; }
  focus() {}
  showModal() {}
}

async function app({hash = '#key=unit-test-only&lang=en', storage = new Map(), status = 200} = {}) {
  const elements = new Map(), readers = [];
  const localStorage = {getItem: k => storage.has(k) ? storage.get(k) : null, setItem: (k, v) => storage.set(k, String(v)), removeItem: k => storage.delete(k)};
  const get = id => { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); };
  get('detail').value = 'auto'; get('budget').value = '1024';
  const context = vm.createContext({
    document: {getElementById: get, createElement: () => new Element(), querySelectorAll: () => [], documentElement: {}, body: {classList: {remove() {}, toggle() {}}}},
    location: {hash, pathname: '/'}, history: {replaceState() {}}, window: {localStorage},
    lucide: {createIcons() {}}, navigator: {clipboard: {writeText: async () => {}}},
    URLSearchParams, AbortController, TextDecoder, Uint8Array, performance, setTimeout, clearTimeout,
    FileReader: class { readAsDataURL(file) { this.file = file; readers.push(this); } },
    fetch: async (url, options) => ({ok: status === 200, status, auth: options?.headers?.Authorization, json: async () => url === '/v1/models' ? {data: [{id: 'quasar-27b'}]} : {vision_forward: true}}),
  });
  const run = source => vm.runInContext(source, context);
  run(fs.readFileSync(path.join(__dirname, 'augur_web/i18n.js'), 'utf8'));
  run(fs.readFileSync(path.join(__dirname, 'augur_web/app.js'), 'utf8'));
  await run('connect()');
  return {context, get, run, readers, storage};
}

test('conversation drafts and attachments remain isolated', async () => {
  const a = await app();
  a.run("active.draft.pending.push({name:'one.png',url:'data:image/png;base64,A'}); $('prompt').value='draft one'; $('detail').value='high'; newChat();");
  assert.equal(a.get('prompt').value, '');
  assert.equal(a.get('detail').value, 'auto');
  assert.equal(a.get('attachments').children.length, 0);
  a.run("$('prompt').value='draft two'; selectChat(chats[1]);");
  assert.equal(a.get('prompt').value, 'draft one');
  assert.equal(a.get('detail').value, 'high');
  assert.equal(a.get('attachments').children.length, 1);
  a.run('selectChat(chats[0]);');
  assert.equal(a.get('prompt').value, 'draft two');
  assert.equal(a.get('attachments').children.length, 0);
});

test('local_time uses the client calendar and an explicit UTC offset', async () => {
  const a = await app();
  const value = a.run(`localTime({getFullYear:()=>2026,getMonth:()=>9,getDate:()=>10,
    getHours:()=>5,getMinutes:()=>7,getSeconds:()=>9,getTimezoneOffset:()=>-480})`);
  assert.equal(value.local_time, '2026-10-10T05:07:09+08:00');
  const negative = a.run(`localTime({getFullYear:()=>2026,getMonth:()=>9,getDate:()=>9,
    getHours:()=>14,getMinutes:()=>7,getSeconds:()=>9,getTimezoneOffset:()=>210})`);
  assert.equal(negative.local_time, '2026-10-09T14:07:09-03:30');
});

test('invalid file batch is atomic and retains existing attachments visibly', async () => {
  const a = await app();
  a.run("active.draft.pending.push({name:'existing.png',url:'data:image/png;base64,A'}); renderAttachments();");
  a.get('files').files = [{name: 'good.png', type: 'image/png', size: 1}, {name: 'bad.txt', type: 'text/plain', size: 1}];
  await a.get('files').onchange();
  assert.equal(a.readers.length, 0);
  assert.equal(a.run('active.draft.pending.length'), 1);
  assert.equal(a.get('attachments').children.length, 1);
  assert.match(a.get('error').textContent, /PNG Or JPEG/);
});

test('asynchronous file read binds to its originating conversation', async () => {
  const a = await app();
  a.get('files').files = [{name: 'one.png', type: 'image/png', size: 1}];
  const loading = a.get('files').onchange();
  assert.equal(a.get('send').disabled, true);
  a.run('newChat();');
  a.readers[0].result = 'data:image/png;base64,A'; a.readers[0].onload();
  await loading;
  assert.equal(a.run('active.draft.pending.length'), 0);
  assert.equal(a.get('attachments').children.length, 0);
  a.run('selectChat(chats[1]);');
  assert.equal(a.get('attachments').children.length, 1);
  assert.equal(a.get('send').disabled, false);
});

test('cancelled partial generation rolls back display and restores the retry draft', async () => {
  const a = await app();
  a.run("active.draft.pending.push({name:'one.png',url:'data:image/png;base64,A'}); $('prompt').value='  count slowly  '; renderAttachments();");
  let reads = 0;
  a.context.fetch = async (_, options) => ({ok: true, body: {getReader: () => ({
    read() { if (!reads++) return Promise.resolve({value: Buffer.from('data: {"choices":[{"delta":{"content":"1, 2"}}]}\n\n'), done: false});
      return new Promise((_, reject) => options.signal.addEventListener('abort', () => reject(Object.assign(new Error('abort'), {name: 'AbortError'})), {once: true})); },
    releaseLock() {},
  })}});
  const sending = a.run('send({preventDefault(){}})');
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(a.get('messages').children.length, 2);
  for (const id of ['prompt', 'settings-button', 'export', 'budget']) assert.equal(a.get(id).disabled, true, id);
  a.get('stop').onclick(); await sending;
  assert.equal(a.run('active.messages.length'), 0);
  assert.equal(a.get('messages').children.length, 0);
  assert.equal(a.get('prompt').value, '  count slowly  ');
  assert.equal(a.get('attachments').children.length, 1);
  assert.equal(a.get('chat-title').textContent, 'New Chat');
  assert.equal(a.get('attachments').children[0].children[1].disabled, false);
  assert.equal(a.get('error').textContent, 'Generation Stopped');
  assert.equal(a.get('prompt').disabled, false);
});

test('incomplete SSE rolls back to committed history and preserves error', async () => {
  const a = await app();
  a.run("active.messages.push({role:'user',content:'old'},{role:'assistant',content:'answer'}); renderHistory(); $('prompt').value='retry';");
  a.context.fetch = async () => ({ok: true, body: {getReader: () => ({read: async () => ({done: true}), releaseLock() {}})}});
  await a.run('send({preventDefault(){}})');
  assert.equal(a.run('active.messages.length'), 2);
  assert.equal(a.get('messages').children.length, 2);
  assert.equal(a.get('prompt').value, 'retry');
  assert.equal(a.get('error').textContent, 'Incomplete Response');
});

function sse(events) {
  let sent = false;
  const bytes = Buffer.from(events.map(event => 'data: ' + JSON.stringify(event) + '\n\n').join('') + 'data: [DONE]\n\n');
  return {ok: true, body: {getReader: () => ({
    read: async () => sent ? {done: true} : (sent = true, {done: false, value: bytes}),
    releaseLock() {},
  })}};
}

test('tool roundtrip commits actual execution and bills both HTTP rounds', async () => {
  const a = await app(), requests = [];
  a.run("$('tools').checked=true; $('prompt').value='get the date';");
  a.context.fetch = async (_, options) => {
    requests.push(JSON.parse(options.body));
    return requests.length === 1 ? sse([
      {choices: [{delta: {tool_calls: [{index: 0, id: 'call_one', function: {name: 'local_time', arguments: '{}'}}]}}]},
      {choices: [{finish_reason: 'tool_calls'}], usage: {completion_tokens: 10}},
    ]) : sse([
      {choices: [{delta: {content: 'actual date'}}]},
      {choices: [{finish_reason: 'stop'}], usage: {completion_tokens: 5}},
    ]);
  };
  await a.run('send({preventDefault(){}})');
  assert.equal(requests.length, 2);
  assert.equal(requests[1].messages[2].tool_call_id, 'call_one');
  assert.match(JSON.parse(requests[1].messages[2].content).local_time, /^\d{4}-\d{2}-\d{2}T.*[+-]\d{2}:\d{2}$/);
  assert.equal(a.run('active.messages.length'), 4);
  assert.match(a.get('bill').textContent, /Tool .*TTFT .*15 tokens/);
  a.run('newChat(); selectChat(chats[1]);');
  assert.equal(a.get('messages').children.length, 3);
  assert.equal(a.get('messages').children[1].children[2].className, 'tool-result');
});

test('unsupported tool is never executed and the interaction rolls back', async () => {
  const a = await app();
  a.run("$('tools').checked=true; $('prompt').value='tool prompt';");
  a.context.fetch = async () => sse([
    {choices: [{delta: {tool_calls: [{index: 0, id: 'bad', function: {name: 'shell', arguments: '{}'}}]}}]},
    {choices: [{finish_reason: 'tool_calls'}], usage: {completion_tokens: 7}},
  ]);
  await a.run('send({preventDefault(){}})');
  assert.equal(a.run('active.messages.length'), 0);
  assert.equal(a.get('messages').children.length, 0);
  assert.equal(a.get('prompt').value, 'tool prompt');
  assert.equal(a.get('error').textContent, 'Unsupported Tool Call');
});

test('exact pre-admission busy retry keeps one payload and includes waiting in the bill', async () => {
  const a = await app(), requests = [];
  a.run("$('prompt').value='recovered';");
  a.context.fetch = async (_, options) => {
    requests.push(options.body);
    if (requests.length <= 3) return {ok:false,status:409,json:async()=>({error:{code:'single_user_busy'}})};
    return sse([{choices:[{delta:{content:'recovered'}}]},
      {choices:[{finish_reason:'stop'}],usage:{completion_tokens:2}}]);
  };
  await a.run('send({preventDefault(){}})');
  assert.equal(requests.length, 4);
  assert.ok(requests.every(body => body === requests[0]));
  assert.equal(a.run('active.messages.length'), 2);
  assert.ok(Number(/TTFT ([\d.]+)s/.exec(a.get('bill').textContent)[1]) >= .02);
});

test('Stop during pre-admission wait cancels without submitting another generation', async () => {
  const a = await app(); let count = 0;
  a.run("$('prompt').value='keep this draft';");
  a.context.fetch = async () => { count++; return {ok:false,status:409,json:async()=>({error:{code:'single_user_busy'}})}; };
  const pending = a.run('send({preventDefault(){}})');
  await new Promise(resolve => setTimeout(resolve, 1));
  a.get('stop').onclick(); await pending;
  assert.equal(count, 1); assert.equal(a.get('error').textContent, 'Generation Stopped');
  assert.equal(a.get('prompt').value, 'keep this draft');
  assert.equal(a.run('active.messages.length'), 0);
});

test('uncertain state and other conflicts are never retried', async () => {
  for (const [status, code] of [[503,'native_state_uncertain'], [409,'other'], [500,'single_user_busy']]) {
    const a = await app(); let count = 0;
    a.context.fetch = async () => {count++; return {ok:false,status,json:async()=>({error:{code,message:code}})};};
    a.run("$('prompt').value='hello';"); await a.run('send({preventDefault(){}})');
    assert.equal(count, 1); assert.equal(a.get('error').textContent, code);
  }
});

test('busy retries expire and never swallow accepted stream failures', async () => {
  const a = await app(); let now = 0, count = 0;
  a.context.performance = {now: () => (now += 1000)};
  a.context.fetch = async () => {count++; return {ok:false,status:409,json:async()=>({error:{code:'single_user_busy',message:'busy'}})};};
  a.run("$('prompt').value='hello';"); await a.run('send({preventDefault(){}})');
  assert.ok(count <= 2); assert.equal(a.get('error').textContent, 'busy');
  count = 0;
  a.context.fetch = async () => {count++; return sse([{error:{code:'single_user_busy',message:'accepted failure'}}]);};
  await a.run('send({preventDefault(){}})');
  assert.equal(count, 1); assert.equal(a.get('error').textContent, 'accepted failure');
});


test('English is the default thinking language and chat settings stay isolated', async () => {
  const a = await app();
  assert.equal(a.run('active.thinkingLanguage'), 'en');
  a.run("$('thinking').checked = true; $('thinking-language').value = 'zh'; newChat(); $('thinking-language').value = 'en'; selectChat(chats[1]);");
  assert.equal(a.get('thinking').checked, true);
  assert.equal(a.get('thinking-language').value, 'zh');
});

test('thinking display is separate from the answer and copy target', async () => {
  const a = await app();
  a.run("globalThis.thinkingView = addMessage('assistant', '答案'); addReasoning(thinkingView, 'English reasoning');");
  assert.equal(a.run('thinkingView.content.textContent'), '答案');
  assert.equal(a.run('thinkingView.reasoning.textContent'), 'English reasoning');
});

test('launcher key is remembered so the bare address signs in without typing', async () => {
  const first = await app();
  assert.equal(first.storage.get('augur-api-key'), 'unit-test-only');
  const later = await app({hash: '', storage: first.storage});
  assert.equal(later.run('key'), 'unit-test-only');
  assert.equal(later.run('headers().Authorization'), 'Bearer unit-test-only');
});

test('rejected key is forgotten and points back to the launcher', async () => {
  const storage = new Map([['augur-api-key', 'stale-key']]);
  const a = await app({hash: '', storage, status: 401});
  assert.equal(storage.has('augur-api-key'), false);
  assert.match(a.get('error').textContent, /启动服务\.exe/);
});

test('interface defaults to Simplified Chinese and follows the launcher choice', async () => {
  const zh = await app({hash: '#key=k'});
  assert.equal(zh.run('LANG'), 'zh-CN');
  assert.equal(zh.get('chat-title').textContent, '新对话');
  assert.equal(zh.get('status').textContent, '就绪');
  const en = await app({hash: '#key=k&lang=en', storage: new Map()});
  assert.equal(en.storage.get('augur-lang'), 'en');
  const later = await app({hash: '', storage: en.storage});
  assert.equal(later.run('LANG'), 'en');
  assert.equal(later.get('chat-title').textContent, 'New Chat');
  later.run("$('ui-language').value = 'zh-CN'; $('ui-language').onchange();");
  assert.equal(later.get('chat-title').textContent, '新对话');
  assert.equal(later.storage.get('augur-lang'), 'zh-CN');
});

test('bounded Nerkyor capabilities disable controls and omit unsupported parameters', async () => {
  const a = await app(), requests = [];
  a.get('budget').dataset = {};
  a.run(`$('tools').checked=true; $('thinking').checked=true; active.tools=true; active.thinking=true;
    applyCapabilities({vision_forward:false,ui:{tools:false,thinking:false,thinking_parameters:false,label:'Nerkyor',notice:'512 tokens; cancellation retires owner'}});
    applyOutputCeiling({initial_max_output_limit:486}); setBusy();`);
  for (const id of ['tools', 'thinking', 'thinking-language', 'attach']) assert.equal(a.get(id).disabled, true, id);
  assert.equal(a.get('budget').value, 486);
  assert.equal(a.get('profile-notice').hidden, false);
  assert.equal(a.get('brand-model').textContent, 'Nerkyor');
  a.context.fetch = async (_, options) => { requests.push(JSON.parse(options.body)); return sse([
    {choices:[{delta:{content:'42'}}]}, {choices:[{finish_reason:'stop'}],usage:{completion_tokens:2}}
  ]); };
  a.run("$('prompt').value='test';");
  await a.run('send({preventDefault(){}})');
  assert.equal(requests.length,1);
  for (const key of ['enable_thinking','thinking_language','tools','parallel_tool_calls','seed']) assert.equal(key in requests[0],false,key);
  assert.equal(requests[0].max_completion_tokens,486);
  a.run('newChat(); selectChat(chats[1]);');
  assert.equal(a.get('thinking').checked,false); assert.equal(a.get('tools').checked,false);
});
