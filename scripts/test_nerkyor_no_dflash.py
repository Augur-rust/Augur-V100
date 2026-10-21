import copy
import unittest
import nerkyor_protocol as P
from test_nerkyor_chat import template, ready, packet

class NoDflashProtocolTests(unittest.TestCase):
    def test_manifest_pins_policy_capacity_and_absent_draft(self):
        t = template(); t.manifest['execution_profile'] = P.NO_DFLASH_EXECUTION
        r = ready(t)
        r.update(profile=P.NO_DFLASH_PROFILE, capacity=4096, maximum_output_tokens=3891, dflash_enabled=False)
        r['target'].update(session_profile=P.NO_DFLASH_PROFILE, policy=P.NO_DFLASH_POLICY, kv_capacity=4096,
            nerkyor_target_execution=P.NO_DFLASH_EXECUTION, dflash_loaded=False, dflash_enabled=False,
            qpn_source_pair_exact_projection_count=400,
            qpn_source_pair_arithmetic='source_f32_pair_reassoc_k16_even_odd_f64_fold_v1')
        del r['target']['dflash_feature_capture']
        P.verify_ready(r,t)
        for key,value in [('policy',P.POLICY),('dflash_loaded',True),('dflash_feature_capture',{}),('kv_capacity',512)]:
            bad=copy.deepcopy(r);bad['target'][key]=value
            with self.assertRaises(P.C.ProtocolError):P.verify_ready(bad,t)
        old=template()
        with self.assertRaises(P.C.ProtocolError):P.verify_ready(r,old)

    def test_receipt_excludes_prefill_token_from_decode_tps(self):
        plan,ledger,tokens,frame=packet()
        ledger.template.manifest['execution_profile']=P.NO_DFLASH_EXECUTION
        r=frame['committed']['receipt'];r['dflash_enabled']=False;r['session']['profile']=P.NO_DFLASH_PROFILE
        r['decode_detail'].update(profile=P.NO_DFLASH_PROFILE,execution_profile=P.NO_DFLASH_EXECUTION,
            dflash_enabled=False,dflash_loaded=False,rounds=0,capacity=4096,selector='target_greedy',
            decode_steps=1,decode_wall_s=0.05,decode_tokens_per_second=20.0,first_prefill_selection_excluded=True)
        r['prefill_detail'].update(profile=P.NO_DFLASH_PROFILE,execution_profile=P.NO_DFLASH_EXECUTION,
            prefill='qpn_source_reassoc_scalar_v1',dflash_loaded=False)
        P.verify_terminal(frame,plan,ledger,tokens)
        for key,value in [('decode_steps',2),('decode_tokens_per_second',40.0),('decode_wall_s',float('nan'))]:
            bad=copy.deepcopy(frame);bad['committed']['receipt']['decode_detail'][key]=value
            with self.assertRaises(P.C.ProtocolError):P.verify_terminal(bad,plan,ledger,tokens)

if __name__ == '__main__':unittest.main()
