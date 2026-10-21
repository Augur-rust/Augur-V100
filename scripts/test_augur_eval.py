"""CPU tests for answer extraction and grading in augur_eval.py."""
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import augur_eval as E


class Tests(unittest.TestCase):
    def test_boxed_is_last_and_balanced(self):
        self.assertEqual(E.boxed(r'first \boxed{1} then \boxed{\frac{a}{b}} end'), r'\frac{a}{b}')
        self.assertIsNone(E.boxed('no answer'))
        self.assertIsNone(E.boxed(r'\boxed{unclosed'))

    def test_grading(self):
        self.assertEqual(E.grade(dict(kind='integer', answer='70'), r'so \boxed{070}'), ('070', True))
        self.assertEqual(E.grade(dict(kind='integer', answer='70'), r'\boxed{71}'), ('71', False))
        self.assertEqual(E.grade(dict(kind='letter', answer='C'), 'Answer: B\nwait\nAnswer: C'), ('C', True))
        self.assertEqual(E.grade(dict(kind='letter', answer='C'), 'C is right'), (None, False))
        self.assertTrue(E.grade(dict(kind='math', answer=r'\frac{14}{3}'), r'\boxed{\dfrac{14}{3}}')[1])
        self.assertTrue(E.grade(dict(kind='math', answer=r'90^\circ'), r'\boxed{90}')[1])
        self.assertTrue(E.grade(dict(kind='math', answer=r'\text{(C)}'), r'\boxed{(C)}')[1])
        self.assertFalse(E.grade(dict(kind='math', answer='3'), r'\boxed{4}')[1])

    def test_aime_2026_and_legacy_schemas(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'aime2026.jsonl'
            path.write_text(json.dumps(dict(problem='A 2026 problem', answer='007', id=17)) + '\n\n', encoding='utf-8-sig')
            items = list(E.aime([path]))
            self.assertEqual(items[0]['id'], 'aime2026-17')
            self.assertEqual(items[0]['answer'], '7')
            self.assertIn('A 2026 problem', items[0]['prompt'])
            path.write_text(json.dumps(dict(question='A legacy problem', answer=0)) + '\n', encoding='utf-8')
            self.assertEqual(list(E.aime([path]))[0]['answer'], '0')
            self.assertEqual(list(E.aime([path]))[0]['id'], 'aime2026-1')

    def test_aime_rejects_bad_rows_and_duplicate_ids(self):
        rows = [dict(problem='', answer='3'), dict(problem='q', answer='1000'),
                dict(problem='q', answer='3.5')]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'aime.jsonl'
            for row in rows:
                path.write_text(json.dumps(row), encoding='utf-8')
                with self.assertRaises(ValueError):
                    list(E.aime([path]))
            row = dict(problem='q', answer='3', id='same')
            path.write_text(json.dumps(row) + '\n' + json.dumps(row), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                list(E.aime([path]))

    def test_integer_grading_does_not_concatenate_numbers(self):
        for value in ('1/2', '1, 2', '12 or 13', '-12', '0.12'):
            self.assertFalse(E.grade(dict(kind='integer', answer='12'), r'\boxed{' + value + '}')[1])
        self.assertTrue(E.grade(dict(kind='integer', answer='0'), r'\boxed{000}')[1])
        self.assertTrue(E.grade(dict(kind='integer', answer='12'), r'\boxed{012}')[1])

    def test_cli_2026_resume_and_dataset_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, key, out = root / 'aime2026.jsonl', root / 'key', root / 'out'
            data.write_text(json.dumps(dict(problem='Test question', answer='7', id=1)), encoding='utf-8')
            key.write_text('test-only', encoding='ascii')
            argv = ['augur_eval', '--api-key-file', str(key), '--aime2026', str(data), '--out', str(out)]
            response = dict(id='test-request', choices=[dict(message=dict(content=r'\boxed{007}'), finish_reason='stop')],
                            usage=dict(completion_tokens=100, prompt_tokens=20,
                                       completion_tokens_details=dict(reasoning_tokens=80)))
            with patch('sys.argv', argv), patch.object(E, 'ask', return_value=response) as ask, patch('builtins.print'):
                E.main()
                E.main()
            self.assertEqual(ask.call_count, 1)
            summary = json.loads((out / 'summary.json').read_text())
            self.assertTrue(summary['completed'])
            self.assertEqual(summary['correct'], 1)
            self.assertEqual(summary['independent_trials'], 1)
            data.write_text(json.dumps(dict(problem='Changed question', answer='7', id=1)), encoding='utf-8')
            with patch('sys.argv', argv), patch.object(E, 'ask') as ask:
                with self.assertRaisesRegex(ValueError, 'dataset changed'):
                    E.main()
            ask.assert_not_called()

    def test_seed_is_sent_recorded_and_bound_to_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'aime2026.jsonl';key=root/'key';out=root/'out'
            data.write_text(json.dumps(dict(problem='q',answer='7',id=10)),encoding='utf-8')
            key.write_text('test-only',encoding='ascii')
            argv=['augur_eval','--api-key-file',str(key),'--aime2026',str(data),'--out',str(out),'--seed','2']
            response=dict(id='r2',choices=[dict(message=dict(content=r'\boxed{7}'),finish_reason='stop')],
                          usage=dict(completion_tokens=10,prompt_tokens=5))
            with patch('sys.argv',argv),patch.object(E,'ask',return_value=response) as ask,patch('builtins.print'):
                E.main()
            self.assertEqual(ask.call_args.args[-1],2)
            self.assertEqual(json.loads((out/'run.json').read_text())['sampling']['seed'],2)
            self.assertEqual(json.loads((out/'summary.json').read_text())['seed'],2)
            self.assertEqual(json.loads((out/'results.jsonl').read_text())['seed'],2)
            argv[-1]='1'
            with patch('sys.argv',argv),patch.object(E,'ask') as ask:
                with self.assertRaisesRegex(ValueError,'configuration or dataset changed'):E.main()
            ask.assert_not_called()
        class Reply:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self):return b'{}'
        with patch.object(E.urllib.request,'urlopen',return_value=Reply()) as opened:
            E.ask('http://localhost','test-key','q',100,2)
        self.assertEqual(json.loads(opened.call_args.args[0].data)['seed'],2)

    def test_gpqa_shuffle_is_fixed_and_covers_all_letters(self):
        rows = [{'Record ID': f'r{i}', 'Question': 'q', 'Correct Answer': 'yes', 'Incorrect Answer 1': 'a',
                 'Incorrect Answer 2': 'b', 'Incorrect Answer 3': 'c'} for i in range(64)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'g.jsonl'
            path.write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
            first, second = list(E.gpqa(path)), list(E.gpqa(path))
        self.assertEqual(first, second)
        self.assertEqual({x['answer'] for x in first}, set('ABCD'))
        for x in first:
            self.assertIn(f"{x['answer']}) yes", x['prompt'])


if __name__ == '__main__':
    unittest.main()
