"""CPU tests for answer extraction and grading in augur_eval.py."""
import json
import tempfile
from pathlib import Path
import unittest
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
