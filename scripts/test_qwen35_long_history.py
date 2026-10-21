"""CPU regressions for long agent histories; never load a model or drop messages."""
import copy
import unittest

import qwen35_api as A
import qwen35_chat as C
import qwen35_tools as T
from test_qwen35_api import MODEL, Backend, request


class LongHistoryTests(unittest.TestCase):
    def text_history(self, pairs, system=False):
        messages = [dict(role='system', content='Rules')] if system else []
        for i in range(pairs):
            messages.extend([dict(role='user', content=f'Q{i}'),
                             dict(role='assistant', content=f'A{i}')])
        return messages + [dict(role='user', content='Next')]

    def test_old_boundary_and_long_text_history_are_preserved(self):
        for pairs, system in ((64, False), (64, True), (65, False), (128, False)):
            with self.subTest(pairs=pairs, system=system):
                messages = self.text_history(pairs, system)
                original = copy.deepcopy(messages)
                result = A.validate_request(request(messages), MODEL)['messages']
                self.assertEqual([(m['role'], m['content']) for m in result],
                                 [(m['role'], m['content']) for m in messages])
                self.assertEqual(messages, original)

    def tool_history(self, count=80):
        messages = [dict(role='user', content='Continue the task')]
        for i in range(count):
            messages.extend([
                dict(role='assistant', content='', tool_calls=[dict(
                    id=f'call_{i}', type='function',
                    function=dict(name='historical_tool', arguments='{}'))]),
                dict(role='tool', tool_call_id=f'call_{i}', content=f'Result {i}')])
        return messages

    def test_long_tool_history_preserves_every_call_and_result(self):
        messages = self.tool_history()
        original = copy.deepcopy(messages)
        result = A.validate_request(request(messages), MODEL)['messages']
        self.assertEqual(len(result), 161)
        self.assertEqual([m['tool_call_id'] for m in result if m['role'] == 'tool'],
                         [f'call_{i}' for i in range(80)])
        self.assertEqual(result[-1]['content'], 'Result 79')
        self.assertEqual(messages, original)
        self.assertEqual(result, T.normalize_messages(messages, []))

    def test_steering_after_complete_results_needs_no_fake_assistant(self):
        messages = self.tool_history() + [dict(role='user', content='Use the new constraint; continue')]
        result = A.validate_request(request(messages), MODEL)['messages']
        self.assertEqual(len(result), len(messages))
        self.assertEqual([m['role'] for m in result[-2:]], ['tool', 'user'])
        self.assertEqual(result[-1]['content'], messages[-1]['content'])

    def test_steering_cannot_interrupt_a_partial_parallel_tool_group(self):
        messages = self.tool_history(1)
        messages[1]['tool_calls'].append(dict(id='call_extra', type='function',
            function=dict(name='historical_tool', arguments='{}')))
        messages.append(dict(role='user', content='New constraint'))
        with self.assertRaisesRegex(A.ApiError, 'outstanding tool calls'):
            A.validate_request(request(messages), MODEL)

    def test_long_completed_history_can_be_checkpointed(self):
        import qwen35_persistence as P
        from test_qwen35_persistence import live_ledger
        from test_qwen35_chat import fixture
        ledger = live_ledger()
        ledger.messages = A.validate_request(request(self.text_history(65)), MODEL)['messages'][:-1]
        ledger.history, ledger.terminal, ledger.prefix, ledger.last_text = [], None, '', ''
        plan = ledger.plan('Next')
        ledger.accept(plan, fixture(ledger, plan)[-1], 'OK')
        state = P.snapshot(ledger)
        self.assertEqual(len(state['messages']), 132)

    def test_long_tool_history_and_steering_work_over_real_loopback_http(self):
        import http.client
        import json
        import threading
        from test_qwen35_api import KEY
        from test_qwen35_tools import ToolTemplate
        backend = Backend()
        backend.template = ToolTemplate()
        controller = A.Controller(backend.template, 'http://127.0.0.1', MODEL,
                                  transport_factory=backend.factory)
        self.addCleanup(controller.close)
        server = A.Server(('127.0.0.1', 0), controller, KEY)
        thread = threading.Thread(target=server.serve_forever, kwargs=dict(poll_interval=.01), daemon=True)
        thread.start()
        try:
            for streaming in (False, True):
                messages = self.tool_history() + [dict(role='user', content='Continue, do not redo finished work')]
                conn = http.client.HTTPConnection(*server.server_address, timeout=5)
                try:
                    conn.request('POST', '/v1/chat/completions', json.dumps(request(messages, stream=streaming)),
                                 {'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
                    response = conn.getresponse()
                    raw = response.read().decode()
                    self.assertEqual(response.status, 200, raw)
                    if streaming:
                        self.assertIn('data: [DONE]', raw)
                    else:
                        self.assertEqual(json.loads(raw)['choices'][0]['message']['content'], 'OK')
                    self.assertEqual(len(controller.current.messages), len(messages) + 1)
                finally:
                    conn.close()
        finally:
            server.shutdown()
            thread.join(2)
            server.server_close()

    def test_long_history_still_rejects_unpaired_tool_results(self):
        messages = self.tool_history()
        messages[-1]['tool_call_id'] = 'missing_call'
        with self.assertRaisesRegex(A.ApiError, 'tool_call_id'):
            A.validate_request(request(messages), MODEL)

    def test_empty_non_array_and_object_count_bound_have_precise_errors(self):
        for value in (None, [], {}, 'messages'):
            with self.subTest(value=value), self.assertRaisesRegex(A.ApiError, 'non-empty array'):
                A.validate_request(dict(model=MODEL, messages=value), MODEL)
        messages = [dict(role='user', content='x')] * (C.MAX_MESSAGES + 1)
        for validate in (lambda: A.validate_request(request(messages), MODEL),
                         lambda: T.normalize_messages(messages, [])):
            with self.assertRaisesRegex(ValueError, f'{C.MAX_MESSAGES + 1} entries; maximum is {C.MAX_MESSAGES}'):
                validate()

    def test_long_text_history_reuses_native_ledger_on_next_turn(self):
        backend = Backend()
        controller = A.Controller(backend.template, 'http://127.0.0.1', MODEL,
                                  transport_factory=backend.factory)
        self.addCleanup(controller.close)
        messages = self.text_history(65)
        def complete(history):
            controller.complete(request(history), lambda *args: None, lambda *args: None,
                                lambda *args: None, lambda abort: lambda: None)
        complete(messages)
        messages += [dict(role='assistant', content='OK'), dict(role='user', content='More')]
        complete(messages)
        self.assertEqual(backend.loads, 1)
        self.assertEqual(backend.closes, 0)
        self.assertEqual(len(controller.current.messages), len(messages) + 1)


if __name__ == '__main__':
    unittest.main()
