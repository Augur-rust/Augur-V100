import unittest
from analyze_augur_sampling_rejections import nucleus,draw,uniform,classify,softmax,f32
class RejectionAuditTests(unittest.TestCase):
 def test_nucleus_includes_crossing_token(self):
  import math
  p=nucleus(list(range(20)),[math.log(.5),math.log(.3),math.log(.16),math.log(.04)]+[-100]*16)
  self.assertEqual(list(p),[0,1,2]);self.assertAlmostEqual(p[0],.5/.96)
  self.assertEqual(draw(p,.6),1)
 def test_tie_and_bad_rows(self):
  self.assertEqual(len(nucleus(list(range(20)),[0.]*20)),19)
  with self.assertRaises(ValueError):nucleus([0]*20,[0.]*20)
  with self.assertRaises(ValueError):nucleus(list(reversed(range(20))),[0.]*20)
 def test_json_f32_roundtrip(self):
  self.assertEqual(f32(0.1),0.10000000149011612)
 def test_position_rng(self):self.assertEqual(uniform(0),.8833108082136426)
 def test_disjoint_classification(self):
  self.assertEqual(classify(1,1,2,[1,2]),'accepted')
  self.assertEqual(classify(1,3,1,[1,2]),'sampled_target_missing_from_draft_top16')
  self.assertEqual(classify(1,2,1,[1,2]),'sampled_divergence_with_draft_at_target_top1')
  self.assertEqual(classify(1,2,3,[1,2,3]),'sampled_target_in_candidates_but_selector_chose_other_non_top1')
 def test_overlap_is_not_independent_draw_dot_product(self):
  p={0:.6,1:.4};q=softmax([0,1],[0,0],1)
  self.assertAlmostEqual(sum(min(p[k],q[k]) for k in p),.9)
  self.assertAlmostEqual(sum(p[k]*q[k] for k in p),.5)
if __name__=='__main__':unittest.main()
