"""CPU contract tests for the independent sparse probability trace oracle."""
import unittest
from unittest.mock import patch
import analyze_augur_probability_trace as oracle

class ProbabilityContractTests(unittest.TestCase):
    def test_calibration_temperature_only_affects_pos234(self):
        scores=[16.0-i for i in range(16)]
        default={'temperature':1.,'pair_scale':1.};cal={'temperature':.875,'pair_scale':1.5}
        for slot in [1,5,6,7]:self.assertEqual(oracle.proposal_distribution(scores,slot,default),oracle.proposal_distribution(scores,slot,cal))
        for slot in [2,3,4]:
            p=oracle.proposal_distribution(scores,slot,cal)
            self.assertAlmostEqual(sum(p),1.)
            self.assertGreater(p[0],oracle.proposal_distribution(scores,slot,default)[0])
        for t in [float('nan'),float('inf'),0,1.51]:
            with self.assertRaises(AssertionError):oracle.proposal_distribution(scores,2,{'temperature':t,'pair_scale':1.})

    def test_per_position_calibration_and_legacy_equivalence(self):
        scores=[16.0-i for i in range(16)]
        legacy={'temperature':.875,'pair_scale':1.5}
        per={'temperatures':[1.,.875,.875,.875,1.,1.,1.],'pair_scales':[1.,1.5,1.5,1.5,1.,1.,1.]}
        self.assertEqual(oracle.normalize_calibration(legacy),oracle.normalize_calibration(per))
        tiered={'temperatures':[.875,.875,.75,.625,.625,.5,.625],'pair_scales':[1.]+[1.5]*6}
        for slot in range(1,8):
            p=oracle.proposal_distribution(scores,slot,tiered);self.assertAlmostEqual(sum(p),1.)
            if tiered['temperatures'][slot-1]<1:self.assertGreater(p[0],oracle.proposal_distribution(scores,slot,None)[0])
        for bad in [{'temperatures':[1.]*6,'pair_scales':[1.]*7},{'temperatures':[1.]*7,'pair_scales':[1.5]*7},
                    {'temperatures':[.49]+[1.]*6,'pair_scales':[1.]*7},{'temperatures':[1.]*7,'pair_scales':[1.]*6+[float('nan')]}]:
            with self.assertRaises(AssertionError):oracle.normalize_calibration(bad)

    def test_block_verify_matches_exact_law(self):
        # Exact quadrature over coins and residual draws on a 2-step example (gamma=2, V=3).
        import itertools
        P={():{0:.2,1:.5,2:.3},(0,):{0:.6,1:.4},(1,):{1:.1,2:.9},(2,):{0:1.},(0,0):{1:1.},(0,1):{2:1.},(1,1):{0:1.},(1,2):{0:1.},(1,0):{0:1.},(0,2):{0:1.}}
        Q={():{0:.6,1:.3,2:.1},(0,):{0:.2,1:.8},(1,):{1:.5,2:.5},(2,):{0:1.}}
        law={};n=40
        for d1,q1 in Q[()].items():
            for d2,q2 in Q[(d1,)].items():
                Ps=[P[()],P[(d1,)],P.get((d1,d2),{0:1.})];Qs=[Q[()],Q[(d1,)]]
                for a1,a2,c in itertools.product(range(n),range(n),range(n)):
                    us={('accept',1):(a1+.5)/n,('accept',2):(a2+.5)/n}
                    with patch.object(oracle,'uniform',lambda purpose,pos:us.get((purpose,pos),(c+.5)/n)):
                        tau,x=oracle.block_verify(Ps,Qs,[d1,d2],0)
                    key=tuple([d1,d2][:tau]+[x])[:1];law[key]=law.get(key,0)+q1*q2/n**3
        for t,v in P[()].items():self.assertAlmostEqual(law.get((t,),0),v,places=2)

    def test_independent_domains(self):
        for position in (0,173,32760,262000):
            values=[oracle.uniform(p,position) for p in oracle.DOMAINS]
            self.assertEqual(len(set(values)),4)
            self.assertTrue(all(0<=x<1 for x in values))

    def test_exact_discrete_distribution_with_missing_support(self):
        p={0:.2,1:.5,2:.3};q={0:.6,1:.2,3:.2};counts=[0]*4;accepted=0
        for a in range(10):
            draft=0 if a<6 else 1 if a<8 else 3
            for b in range(30):
                for c in range(10):
                    with patch.object(oracle,'uniform',lambda purpose,pos:(b+.5)/30 if purpose=='accept' else (c+.5)/10):
                        token=oracle.verify(p,q,draft,100)
                    counts[token]+=1;accepted+=token==draft
        self.assertEqual(counts,[600,1500,900,0]);self.assertEqual(accepted,1200)

    def test_equal_q_always_accepts(self):
        p={4:.3,8:.7}
        for position in range(100):
            for draft in p:self.assertEqual(oracle.verify(p,p,draft,position),draft)

    def test_rejects_empty_q_and_zero_probability_proposal(self):
        for q,d in [({1:0.0},1),({1:.2,2:.2},1),({1:0.0,2:1.0},1)]:
            with self.assertRaises(AssertionError):oracle.verify({1:.5,2:.5},q,d,1)

if __name__=='__main__':unittest.main()
