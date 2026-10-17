"""Thinking policy/channel/state contracts; model language quality is measured separately."""
import copy
import io
import http.client
import json
import threading
import unittest
import augur_api as Q
import augur_reasoning as R
import qwen35_api as A
import qwen35_chat as C
from test_augur_api import Backend, Template, Tokenizer


class ThinkingTokenizer(Tokenizer):
    def decode(self, ids, skip_special_tokens=False):
        return ''.join('</think>' if t == 1102 else super(ThinkingTokenizer, self).decode([t], skip_special_tokens) for t in ids)

    def token_to_id(self, text):
        return 1102 if text == '</think>' else None


class ThinkingTemplate(Template):
    tokenizer = ThinkingTokenizer()

    def render(self, messages, tools=None, add_generation_prompt=True):
        p = getattr(self, 'reasoning_policy', R.policy())
        if p['enabled']:
            messages = R.instruct(messages, p['language'])
        text = super().render(messages, tools, add_generation_prompt)
        return text + ('<think>\n' if p['enabled'] and add_generation_prompt else '')


class StreamTests(unittest.TestCase):
    def test_every_closing_delimiter_split_and_tool_isolation(self):
        text = 'Analysis <tool_call>not executable</tool_call></think>42'
        for split in range(len(text)+1):
            with self.subTest(split=split):
                content, reasoning = [], []
                output = R.Output(content.append, reasoning.append)
                output.push(text[:split]); output.push(text[split:])
                self.assertEqual(output.finish('eos'), '42')
                self.assertEqual(''.join(content), '42')
                self.assertEqual(''.join(x['reasoning_content'] for x in reasoning),
                                 'Analysis <tool_call>not executable</tool_call>')

    def test_length_is_reasoning_only_and_eos_without_close_is_error(self):
        for stop in ('length', 'eos'):
            output = R.Output(lambda x: self.fail('reasoning leaked'), lambda x: None)
            output.push('reasoning </thi')
            if stop == 'eos':
                with self.assertRaises(C.ProtocolError): output.finish(stop)
            else:
                self.assertEqual(output.finish(stop), '')
                self.assertEqual(output.reasoning, 'reasoning </thi')
                self.assertFalse(output.closed)

    def test_effort_reaches_text_and_vision_template(self):
        from unittest.mock import patch
        from jinja2 import StrictUndefined
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        template = object.__new__(Q.Template)
        template.template = ImmutableSandboxedEnvironment(undefined=StrictUndefined).from_string(
            '{{ enable_thinking }}:{{ reasoning_effort|default("xhigh") }}')
        messages = [dict(role='user', content='Make a file')]
        for vision in (False, True):
            template.vision = vision
            for requested, expected in ((None, 'xhigh'), ('low', 'low'), ('high', 'medium'), ('medium', 'medium'), ('xhigh', 'xhigh')):
                with self.subTest(vision=vision, effort=requested):
                    template.reasoning_policy = R.policy(True, 'en', requested)
                    self.assertEqual(template.render(messages), 'True:' + expected)
        self.assertEqual(messages, [dict(role='user', content='Make a file')])

    def test_policy_injection_is_stable_and_does_not_mutate_inputs(self):
        messages = [C.ChatTemplate.message('system', 'Answer in Chinese.'), C.ChatTemplate.message('user', 'Hi')]
        before = copy.deepcopy(messages)
        first = R.instruct(messages, 'en')
        self.assertEqual(messages, before)
        self.assertEqual(first, R.instruct(messages, 'en'))
        self.assertIn('Use English', first[0]['content'])
        self.assertIn('Answer in Chinese.', first[0]['content'])
        self.assertNotEqual(first, R.instruct(messages, 'zh'))


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.backend = Backend(); self.backend.template = ThinkingTemplate()
        self.controller = Q.Controller(self.backend.template, 'unused', 'augur-test', transport_factory=self.backend.factory)
        self.backend.outputs = list(b'English analysis') + [1102] + list(b'42')

    def payload(self, **kwargs):
        return dict(model='augur-test', messages=[dict(role='user', content='Answer 42')],
                    max_tokens=512, enable_thinking=True, **kwargs)

    def complete(self, payload=None):
        result, emitted = {}, []
        self.controller.complete(payload or self.payload(), lambda meta, r: result.update(meta=meta), emitted.append,
            lambda meta, r, text, finish, usage: result.update(request=r, text=text, finish=finish, usage=usage),
            lambda abort: lambda: None)
        return result, emitted

    def test_default_english_final_separate_and_actual_token_accounting(self):
        result, emitted = self.complete()
        self.assertEqual(result['text'], '42')
        self.assertEqual(result['request']['response_reasoning_content'], 'English analysis')
        self.assertEqual(result['meta']['reasoning_policy']['language'], 'en')
        self.assertEqual(''.join(x for x in emitted if isinstance(x, str)), '42')
        details = result['usage']['completion_tokens_details']
        self.assertEqual(details, dict(reasoning_tokens=16, reasoning_boundary_tokens=1, answer_tokens=2, eos_tokens=1))
        self.assertEqual(sum(details.values()), result['usage']['completion_tokens'])
        self.assertEqual(self.controller.current.last_text, 'English analysis</think>42')
        self.assertEqual(self.controller.current.messages[-1]['content'], '42')

    def test_policy_switch_reopens_and_same_policy_keeps_original_hidden_tokens(self):
        self.complete(); history = self.controller.current.history.copy()
        messages = [dict(role='user', content='Answer 42'), dict(role='assistant', content='42', reasoning_content='display metadata'),
                    dict(role='user', content='Continue')]
        payload = self.payload(); payload['messages'] = messages
        result, _ = self.complete(payload)
        self.assertEqual(result['meta']['session_mode'], 'exact_history_append')
        self.assertEqual(self.backend.requests[-1]['input']['exact_history'], history)
        payload['messages'] += [dict(role='assistant', content='42'), dict(role='user', content='Continue again')]
        payload['thinking_language'] = 'zh'
        result, _ = self.complete(payload)
        self.assertEqual(result['meta']['session_mode'], 'canonical_full_history_open')
        self.assertEqual(result['meta']['reasoning_policy']['language'], 'zh')

    def test_effort_mapping_and_default_are_visible_in_request_policy(self):
        for requested, expected in ((None, 'xhigh'), ('low', 'low'), ('medium', 'medium'), ('high', 'medium'), ('xhigh', 'xhigh')):
            with self.subTest(effort=requested):
                self.setUp()
                payload = self.payload(reasoning_effort=requested)
                before = copy.deepcopy(payload)
                result, _ = self.complete(payload)
                self.assertEqual(result['meta']['reasoning_policy']['effort'], expected)
                self.assertEqual(payload, before)
        payload = self.payload(reasoning_effort='low')
        payload.pop('enable_thinking')
        validated = self.controller.validate_request(payload)
        self.assertTrue(validated['reasoning_policy']['enabled'])
        self.assertEqual(validated['reasoning_policy']['effort'], 'low')
        for value in ('off', 'none'):
            payload['reasoning_effort'] = value
            validated = self.controller.validate_request(payload)
            self.assertFalse(validated['reasoning_policy']['enabled'])
            self.assertIsNone(validated['reasoning_policy']['effort'])

    def test_invalid_or_conflicting_effort_rejected_before_native(self):
        for value in ('ultra', '', 1, True, [], {}):
            with self.subTest(value=value), self.assertRaises(A.ApiError):
                self.complete(self.payload(reasoning_effort=value))
        for enabled, effort in ((False, 'low'), (False, 'xhigh'), (True, 'off'), (True, 'none')):
            payload = self.payload(reasoning_effort=effort)
            payload['enable_thinking'] = enabled
            with self.subTest(enabled=enabled, effort=effort), self.assertRaises(A.ApiError):
                self.complete(payload)
        self.assertEqual(self.backend.requests, [])

    def test_effort_change_reopens_but_same_effort_reuses_history(self):
        self.complete(self.payload(reasoning_effort='low'))
        payload = self.payload(reasoning_effort='low')
        payload['messages'] += [dict(role='assistant', content='42'), dict(role='user', content='Continue')]
        result, _ = self.complete(payload)
        self.assertEqual(result['meta']['session_mode'], 'exact_history_append')
        payload['messages'] += [dict(role='assistant', content='42'), dict(role='user', content='Continue again')]
        payload['reasoning_effort'] = 'xhigh'
        result, _ = self.complete(payload)
        self.assertEqual(result['meta']['session_mode'], 'canonical_full_history_open')
        self.assertEqual(result['meta']['reasoning_policy']['effort'], 'xhigh')

    def test_long_budget_is_explicit_and_capacity_stays_bounded(self):
        payload = self.payload(); payload['max_tokens'] = Q.MAX_OUTPUT
        result, _ = self.complete(payload)
        self.assertEqual(self.backend.requests[-1]['max_new_tokens'], Q.MAX_OUTPUT)
        self.assertEqual(result['usage']['completion_tokens'], 20)
        count = len(self.backend.requests)
        payload['max_tokens'] = Q.MAX_OUTPUT + 1
        with self.assertRaises(A.ApiError): self.complete(payload)
        self.assertEqual(len(self.backend.requests), count)

    def test_invalid_policy_rejected_before_native(self):
        for enabled, language in ((1, 'en'), (True, 'fr'), (True, None)):
            payload = self.payload(thinking_language=language); payload['enable_thinking'] = enabled
            with self.assertRaises(A.ApiError): self.complete(payload)
        self.assertEqual(self.backend.requests, [])

    def test_thought_xml_never_becomes_a_tool_call(self):
        from test_qwen35_tools import TOOLS, RAW, ToolTemplate
        self.backend.template.render = ToolTemplate.render.__get__(self.backend.template)
        thought = '<tool_call>not executable</tool_call>'
        self.backend.outputs = list(thought.encode()) + [1102] + self.backend.template.encode(RAW)
        result, emitted = self.complete(self.payload(tools=TOOLS))
        self.assertEqual(result['finish'], 'tool_calls')
        self.assertEqual(len(result['request']['response_tool_calls']), 1)
        self.assertEqual(result['request']['response_tool_calls'][0]['function']['name'], 'lookup')
        self.assertEqual(result['request']['response_reasoning_content'], thought)
        self.assertTrue(all(isinstance(x, dict) for x in emitted))

    def test_streamed_tool_preamble_round_trip_keeps_native_history(self):
        from test_qwen35_tools import TOOLS, RAW, ToolTemplate
        for preamble in ('\n\nI will check the record.\n\n', '\n\n'):
            with self.subTest(preamble=preamble):
                self.setUp()
                self.backend.template.render = ToolTemplate.render.__get__(self.backend.template)
                self.backend.outputs = list(b'English analysis') + [1102] + self.backend.template.encode(preamble + RAW)
                payload = self.payload(tools=TOOLS)
                result, emitted = self.complete(payload)
                streamed = ''.join(x for x in emitted if isinstance(x, str))
                self.assertEqual(streamed, preamble)
                calls = result['request']['response_tool_calls']
                history = self.controller.current.history.copy()
                continuation = self.payload(tools=TOOLS)
                continuation['messages'] += [
                    dict(role='assistant', content=streamed, tool_calls=calls),
                    dict(role='tool', tool_call_id=calls[0]['id'], content='Found'),
                ]
                before = copy.deepcopy(continuation)
                self.backend.outputs = list(b'Done') + [1102] + list(b'Found')
                second, _ = self.complete(continuation)
                self.assertEqual(continuation, before)
                self.assertEqual(second['meta']['session_mode'], 'exact_history_append')
                self.assertEqual(self.backend.requests[-1]['input']['exact_history'], history)

    def test_tool_preamble_normalization_keeps_real_edits_cold(self):
        from test_qwen35_tools import TOOLS, RAW, ToolTemplate
        for change in ('text', 'argument', 'leading_space'):
            with self.subTest(change=change):
                self.setUp()
                self.backend.template.render = ToolTemplate.render.__get__(self.backend.template)
                preamble = 'Checking now.\n\n'
                self.backend.outputs = list(b'English analysis') + [1102] + self.backend.template.encode(preamble + RAW)
                result, _ = self.complete(self.payload(tools=TOOLS))
                calls = copy.deepcopy(result['request']['response_tool_calls'])
                content = preamble
                if change == 'text': content = 'Checking a different record.\n\n'
                elif change == 'leading_space': content = ' ' + preamble
                else: calls[0]['function']['arguments'] = '{"key":"beta","count":2}'
                payload = self.payload(tools=TOOLS)
                payload['messages'] += [dict(role='assistant', content=content, tool_calls=calls),
                                        dict(role='tool', tool_call_id=calls[0]['id'], content='Found')]
                self.backend.outputs = list(b'Done') + [1102] + list(b'Found')
                second, _ = self.complete(payload)
                self.assertEqual(second['meta']['session_mode'], 'canonical_full_history_open')

    def test_truncated_native_eof_cannot_commit_answer(self):
        self.backend.mode = 'truncated_eof'
        with self.assertRaises(C.ProtocolError): self.complete()
        self.assertIsNone(self.controller.current)

    def test_thinking_length_does_not_become_final_answer(self):
        payload = self.payload(); payload['max_tokens'] = 4
        result, _ = self.complete(payload)
        self.assertEqual(result['text'], '')
        self.assertEqual(result['finish'], 'length')
        self.assertFalse(result['request']['thinking_complete'])
        self.assertEqual(result['usage']['completion_tokens_details']['reasoning_tokens'], 4)

    def test_buffered_and_sse_publish_separate_channels(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                key = 'thinking-test-key-' + 'x'*32
                server = A.Server(('127.0.0.1', 0), self.controller, key)
                thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
                thread.start()
                conn = http.client.HTTPConnection(*server.server_address, timeout=4)
                try:
                    payload = self.payload(stream=streaming)
                    conn.request('POST', '/v1/chat/completions', json.dumps(payload),
                                 {'Content-Type': 'application/json', 'Authorization': 'Bearer '+key})
                    response = conn.getresponse(); body = response.read().decode()
                    self.assertEqual(response.status, 200, body)
                    if streaming:
                        events = [json.loads(x[6:]) for x in body.splitlines() if x.startswith('data: ') and x != 'data: [DONE]']
                        deltas = [c['delta'] for event in events for c in event['choices']]
                        self.assertEqual(''.join(d.get('reasoning_content', '') for d in deltas), 'English analysis')
                        self.assertEqual(''.join(d.get('content', '') for d in deltas), '42')
                        self.assertIn('data: [DONE]', body)
                    else:
                        message = json.loads(body)['choices'][0]['message']
                        self.assertEqual(message['content'], '42')
                        self.assertEqual(message['reasoning_content'], 'English analysis')
                finally:
                    conn.close(); server.shutdown(); thread.join(); server.server_close()


class RepetitionTests(unittest.TestCase):
    paragraph = 'A distinctive reasoning paragraph considers an SVG, then returns to the same unresolved choice. ' * 3

    def test_long_cycle_rejected_independently_of_stream_chunking(self):
        counters=[]
        for width in (1,17,257,65536):
            guard=R.RepetitionGuard()
            text='Initial new reasoning.\n'+self.paragraph*100
            with self.assertRaises(C.GenerationFailure) as raised:
                for start in range(0,len(text),width):guard.push(text[start:start+width])
            self.assertEqual(raised.exception.code,'reasoning_repetition')
            counters.append(raised.exception.counters)
        self.assertTrue(all(c==counters[0] for c in counters))
        self.assertGreaterEqual(counters[0]['copies'],6)
        self.assertGreaterEqual(counters[0]['repeated_characters'],8192)

    def test_unique_long_reasoning_short_revisits_and_repetitive_final_are_allowed(self):
        guard=R.RepetitionGuard()
        for i in range(20000):guard.push(f'Distinct premise {i}: result {i*i}.\n')
        guard=R.RepetitionGuard();guard.push(self.paragraph*5)
        final=[];out=R.Output(final.append,lambda x:None)
        out.push('Useful analysis.</think>'+self.paragraph*100)
        self.assertEqual(out.finish('eos'),self.paragraph*100)
        self.assertLessEqual(len(guard.tail),guard.window)

    def test_multi_paragraph_cycle_above_old_4k_limit(self):
        # Synthetic distinct paragraphs, no user prompt or private reasoning.
        block = ''.join(f'Plan item {i:04d}: inspect shape, position and shading.\n'
                        for i in range(230))
        self.assertGreater(len(block), 4096)
        self.assertLess(len(block), R.RepetitionGuard.max_period)
        counters = []
        for width in (17, 257, 65536):
            guard = R.RepetitionGuard()
            text = 'A separate useful introduction.\n' + block * 7
            with self.assertRaises(C.GenerationFailure) as raised:
                for start in range(0, len(text), width):
                    guard.push(text[start:start+width])
            counters.append(raised.exception.counters)
            self.assertEqual(raised.exception.code, 'reasoning_repetition')
            self.assertEqual(counters[-1]['period_characters'], len(block))
            self.assertEqual(counters[-1]['copies'], 6)
            self.assertLessEqual(len(guard.tail), guard.window)
        self.assertTrue(all(value == counters[0] for value in counters))
        guard = R.RepetitionGuard()
        guard.push(block * 5)  # Five revisits do not meet the rejection threshold.
        guard = R.RepetitionGuard()
        for i in range(12):
            guard.push(block + f'New outcome {i}: proceed with different evidence.\n')

    def test_explicit_off_preserves_unrestricted_reasoning_for_diagnostics(self):
        out=R.Output(lambda x:None,lambda x:None,loop_guard=False)
        out.push(self.paragraph*100)
        self.assertEqual(out.finish('length'),'')
        self.assertEqual(out.reasoning,self.paragraph*100)

    def test_http_loop_error_never_commits_or_publishes_done(self):
        for stream in (False,True):
            backend=Backend();backend.template=ThinkingTemplate()
            paragraph=self.paragraph
            class LoopTokenizer(ThinkingTokenizer):
                def decode(self, ids, skip_special_tokens=False):
                    return ''.join(paragraph if t == 1100 else super(LoopTokenizer,self).decode([t],skip_special_tokens) for t in ids)
            backend.template.tokenizer=LoopTokenizer()
            backend.outputs=[1100]*100
            metrics=io.StringIO()
            controller=Q.Controller(backend.template,'unused','augur-test',transport_factory=backend.factory,metrics=metrics)
            server=A.Server(('127.0.0.1',0),controller,'guard-test-key-'+32*'x')
            thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True);thread.start()
            conn=http.client.HTTPConnection(*server.server_address,timeout=10)
            try:
                payload=dict(model='augur-test',messages=[dict(role='user',content='Think')],enable_thinking=True,max_tokens=40000,stream=stream)
                conn.request('POST','/v1/chat/completions',json.dumps(payload),{'Content-Type':'application/json','Authorization':'Bearer '+'guard-test-key-'+32*'x'})
                response=conn.getresponse();body=response.read().decode()
                self.assertEqual(response.status,200 if stream else 422)
                if stream:
                    self.assertNotIn('data: [DONE]',body)
                    value=json.loads([line[6:] for line in body.splitlines() if line.startswith('data: ')][-1])
                else:value=json.loads(body)
                self.assertEqual(value['error']['code'],'reasoning_repetition')
                self.assertNotIn('Native request failed',value['error']['message'])
                self.assertIsNone(controller.current)
                self.assertTrue(controller.faulted)
                events=[json.loads(line) for line in metrics.getvalue().splitlines()]
                self.assertEqual(events[-1]['error_code'],'reasoning_repetition')
                self.assertIn('repeated_characters',events[-1]['error_counters'])
            finally:
                conn.close();server.shutdown();thread.join();server.server_close()

    def test_guard_follows_decode_mode_unless_forced(self):
        backend=Backend();backend.template=ThinkingTemplate()
        controller=Q.Controller(backend.template,'unused','augur-test',transport_factory=backend.factory)
        sampled=dict(Q.SAMPLING_VALUES,preset=Q.SAMPLING_PRESET,greedy=False)
        for setting,health,expected in [('auto',{},True),('auto',dict(sampling=sampled),False),
                                        ('on',dict(sampling=sampled),True),('off',{},False)]:
            with self.subTest(setting=setting,health=health):
                controller.reasoning_loop_guard=setting;controller.health=health
                self.assertIs(controller.loop_guard_active(),expected)


if __name__ == '__main__': unittest.main()
