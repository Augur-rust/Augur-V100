"""CPU tests for the benchmark harness: arm parsing, pooling, identity and the public suite."""
import json
from pathlib import Path
import tempfile
import unittest
import augur_bench as B


def row(arm, case, tps, rounds=100, trajectory='t'):
    tokens = 3072
    return dict(arm=arm, case=case, output_tokens=tokens, rounds=rounds, native_decode_s=tokens / tps,
                native_tps=tps, trajectory_sha256=trajectory)


class Tests(unittest.TestCase):
    def test_arm_parsing(self):
        self.assertEqual(B.parse_arm('base:'), ('base', {}))
        self.assertEqual(B.parse_arm('new:AUGUR_A=1,AUGUR_B=x'), ('new', dict(AUGUR_A='1', AUGUR_B='x')))
        for bad in (':AUGUR_A=1', 'x:PATH=evil', 'x:AUGUR_A', 'a b:'):
            with self.subTest(bad=bad), self.assertRaises(Exception):
                B.parse_arm(bad)

    def test_pooled_gain_identity_and_spread(self):
        rows = [row('a', 'c1', 50), row('b', 'c1', 55), row('b', 'c1', 55), row('a', 'c1', 50.5),
                row('a', 'c2', 80, trajectory='u'), row('b', 'c2', 84, trajectory='v'),
                row('b', 'c2', 84, trajectory='v'), row('a', 'c2', 80, trajectory='u')]
        s = B.summarize(rows, ['a', 'b', 'b', 'a'])
        c = s['comparisons']['b']
        self.assertEqual(c['vs'], 'a')
        self.assertEqual(c['identical_trajectory_cases'], 1)
        self.assertTrue(c['per_case']['c1']['identical_trajectory'])
        self.assertFalse(c['per_case']['c2']['identical_trajectory'])
        tokens = 4 * 3072
        expected = tokens / sum(3072 / t for t in (55, 55, 84, 84)) / (tokens / sum(3072 / t for t in (50, 50.5, 80, 80))) - 1
        self.assertAlmostEqual(c['gain_pct'], 100 * expected)
        self.assertAlmostEqual(s['max_repeat_spread_pct']['a'], 1.0)
        self.assertIn('identical trajectories 1/2', B.report(s))

    def test_contaminated_rows_are_marked_and_fail_a_cold_run(self):
        rows = [row('a', 'c1', 50), dict(row('a', 'c2', 80), cached_prompt_tokens=4096, contaminated=True)]
        with self.assertRaises(SystemExit):
            B.check_cold_prefix(rows, True)
        B.check_cold_prefix(rows, False)          # explicitly allowed: the caller knows why
        B.check_cold_prefix([row('a', 'c1', 50)], True)

    def test_missing_ready_frame_reads_as_no_prefix_store(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(B.serving_prefix_scope(d), dict(persistent=False))

    def test_public_suite(self):
        suite = json.loads(B.SUITE.read_bytes())
        self.assertEqual(suite['schema'], 'augur-bench-suite-v1')
        self.assertEqual(len(suite['cases']), 14)
        self.assertEqual(len({c['name'] for c in suite['cases']}), 14)
        r = suite['request']
        self.assertEqual((r['temperature'], r['top_p'], r['top_k'], r['seed']), (1.0, 0.95, 20, 0))
        self.assertTrue(r['enable_thinking'])
        for c in suite['cases']:
            self.assertEqual([m['role'] for m in c['messages']], ['user'])
            self.assertIn(c['max_completion_tokens'], (3072, 4096))


if __name__ == '__main__':
    unittest.main()
