import copy,hashlib,struct,unittest
import audit_augur_sampling_trajectory as A

class TrajectoryTests(unittest.TestCase):
    def receipt(self,published):
        output=[10,11,88];pending=88 if published else 999
        selected=output if published else output+[pending]
        trace=dict(schema='quasar-sampled-selection-trace-v1',first_position=2,last_position=1+len(selected),
            selections=len(selected),sha256=hashlib.sha256(b''.join(struct.pack('<QI',2+i,t) for i,t in enumerate(selected))).hexdigest(),
            uncommitted_speculative_tail_excluded=True)
        return dict(schema='quasar-token-turn-receipt-v1',request_id='r',output_token_ids=output,
            session=dict(consumed_token_ids=[1,2]+(output[:-1] if published else output),pending_token=pending,pending_token_published=published),
            selected_tokens_including_eos=len(selected),stop_reason='length' if published else 'eos',
            decode_detail=dict(sampling=dict(preset='ninfer_thinking_v1',trajectory=trace)))
    def test_length_and_eos_pending_match_independent_hash(self):
        for published in (False,True):
            row=A.audit(self.receipt(published));self.assertTrue(row['native_selection_trace_verified']);self.assertEqual(row['prompt_tokens'],2)
    def test_tail_or_pending_drift_fails(self):
        for key,value in [('sha256','0'*64),('selections',9),('last_position',999),('first_position',0)]:
            r=self.receipt(False);r['decode_detail']['sampling']['trajectory'][key]=value
            with self.assertRaises(ValueError):A.audit(r)
        r=self.receipt(True);r['session']['pending_token']=12
        with self.assertRaises(ValueError):A.audit(r)
        r=self.receipt(False);r['session']['consumed_token_ids'][-1]=12
        with self.assertRaises(ValueError):A.audit(r)
    def test_unknown_is_not_inferred_greedy(self):
        r=self.receipt(False);r['decode_detail']={}
        self.assertEqual(A.audit(r)['decode_mode'],'unknown')
        r['decode_detail']['sampling']=dict(preset='greedy')
        self.assertEqual(A.audit(r)['decode_mode'],'greedy')

if __name__=='__main__':unittest.main()
