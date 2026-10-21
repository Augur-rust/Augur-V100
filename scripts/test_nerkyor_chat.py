import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import nerkyor_protocol as P
import nerkyor_chat as N


def template():
    manifest = dict(checkpoint=dict(config_sha256='b'*64, index_sha256='c'*64, shards={}), draft_gguf_sha256='d'*64)
    return SimpleNamespace(manifest=manifest, manifest_sha256='a'*64, eos={248046,248044}, end=248046)


def ready(t):
    c = P.execution_contract(t.manifest)
    return dict(event='ready', schema=P.SCHEMA, profile=P.PROFILE, manifest_sha256=t.manifest_sha256,
        capacity=P.CAPACITY, maximum_output_tokens=P.MAX_OUTPUT, generation=P.GENERATION,
        activation_dtype='F32', kv='F32', lm_head='original_bf16', eos_token_ids=[248046,248044],
        cooperative_input_cancellation=True, request_bound_cancellation_discard=False,
        input_frame_limit_bytes=65536, pending_contract=P.PENDING,
        persistence='create_new_fsync_turn_receipts_not_gpu_restore', tokenizer_execution='caller_owned_token_ids',
        decoded_text_output=False, dflash_enabled=True, mtp_enabled=False, vision_enabled=False,
        restore_enabled=False, product_ready=False, actual_long_history_qualified=False, http_enabled=False,
        target=dict(profile='nerkyor_w4a16', session_profile=P.PROFILE, session_manifest_sha256=t.manifest_sha256,
            policy=P.POLICY, activation_dtype='F32', kv_dtype='F32', kv_capacity=P.CAPACITY,
            nerkyor_target_execution=P.EXECUTION, full_shard_hashes_verified=True, product_admitted=False,
            product_ready=False, vision_enabled=False, source_identity=t.manifest['checkpoint'],
            dflash_feature_capture=dict(source_identity_contract=c,source_identity_sha256=P.C.sha256(P.canonical_json(c)))))


def packet(eos=False):
    t = template();ledger=P.Ledger(t);sid=ledger.session_id
    plan=dict(request=dict(request_id='r1',session_id=sid,max_new_tokens=3 if eos else 2,input=dict(kind='open')), input_ids=[1,2], messages=[], prefix='prefix')
    tokens=[3,4];history=[1,2]+(tokens if eos else tokens[:-1])
    detail=dict(profile=P.PROFILE,execution_profile=P.EXECUTION,dflash_enabled=True,sampling='greedy',kv_dtype='F32',
        capacity=P.CAPACITY,selector='own_q8_default_uncalibrated',product_ready=False,mtp_executed=False)
    receipt=dict(schema='nerkyor-token-turn-receipt-v1',request_id='r1',previous_receipt_sha256=None,
        session=dict(profile=P.PROFILE,manifest_sha256=t.manifest_sha256,session_id=sid,revision=1,
            consumed_token_ids=history,consumed_input_sha256=P.token_sha(history),pending_token=248046 if eos else 4,
            pending_token_published=not eos),output_token_ids=tokens,selected_tokens_including_eos=3 if eos else 2,
        stop_reason='eos' if eos else 'length',product_ready=False,vision_enabled=False,dflash_enabled=True,decode_detail=detail)
    receipt['prefill_detail']=dict(profile=P.PROFILE,execution_profile=P.EXECUTION,
        prefill='scalar_initial1_full_b8_commit_scalar_tail_v1',tokens=2,scalar_tokens=2,b8_full_blocks=0,
        product_ready=False,source_identity_sha256=P.C.sha256(P.canonical_json(P.execution_contract(t.manifest))))
    frame=dict(event='terminal',request_id='r1',session_id=sid,product_ready=False,committed=dict(receipt=receipt,receipt_sha256='f'*64))
    return plan,ledger,tokens,frame


class FakeTextTemplate:
    message=staticmethod(P.C.ChatTemplate.message)
    manifest_sha256='a'*64
    eos={248046,248044}
    end=248046
    def encode(self,text): return list(text.encode('utf-8'))
    def render_messages(self,messages,generation=True):
        text=''.join('<|im_start|>'+m['role']+'\n'+m['content'].strip()+'<|im_end|>\n' for m in messages)
        return text+('<|im_start|>assistant\n' if generation else '')


class ProtocolTests(unittest.TestCase):
    def test_ready_rejects_foreign_profile_source_dtype_and_numeric_type_aliases(self):
        t=template();r=ready(t);P.verify_ready(r,t)
        for key,value in [('profile','quasar_b4352_dflash_b8_packed_http_v1'),('capacity',4096),('dflash_enabled',1),('product_ready',0),('kv','Q8_0')]:
            bad=copy.deepcopy(r);bad[key]=value
            with self.assertRaises(P.C.ProtocolError):P.verify_ready(bad,t)
        for key,value in [('draft_file_sha256','e'*64),('product_admitted',0)]:
            bad=copy.deepcopy(r);bad['target']['dflash_feature_capture']['source_identity_contract'][key]=value
            with self.assertRaises(P.C.ProtocolError):P.verify_ready(bad,t)
        bad=copy.deepcopy(r);bad['target']['dflash_feature_capture']['source_identity_contract']['target_execution']['width']=8.0
        with self.assertRaises(P.C.ProtocolError):P.verify_ready(bad,t)
        bad=copy.deepcopy(r);bad['target']['source_identity']={'config_sha256':'e'*64}
        with self.assertRaises(P.C.ProtocolError):P.verify_ready(bad,t)

    def test_token_stream_rejects_sequence_owner_eos_and_bool_ids(self):
        plan,ledger,_,_=packet();tokens=[]
        good=dict(event='token',product_ready=False,token=dict(request_id='r1',session_id=ledger.session_id,sequence=0,token_id=3))
        P.verify_token(good,plan,tokens,ledger.template.eos);self.assertEqual(tokens,[3])
        with self.assertRaises(P.C.ProtocolError):P.verify_token(good,plan,tokens,ledger.template.eos)
        for key,value in [('sequence',True),('token_id',True),('token_id',248046),('request_id','foreign')]:
            bad=copy.deepcopy(good);bad['token'][key]=value
            with self.assertRaises(P.C.ProtocolError):P.verify_token(bad,plan,[],ledger.template.eos)

    def test_terminal_exact_length_and_eos_boundaries(self):
        for eos in [False,True]:
            plan,ledger,tokens,frame=packet(eos);committed=P.verify_terminal(frame,plan,ledger,tokens)
            ledger.accept(plan,committed,'answer')
            committed['receipt']['session']['consumed_token_ids'].clear()
            self.assertTrue(ledger.history);self.assertTrue(ledger.terminal['receipt']['session']['consumed_token_ids'])

    def test_terminal_foreign_history_profile_selector_and_untrusted_shapes(self):
        plan,ledger,tokens,frame=packet()
        mutations=[lambda r:r.update(schema='quasar-token-turn-receipt-v1'),
            lambda r:r.update(output_token_ids=[3,True]),lambda r:r.update(previous_receipt_sha256='a'*64),
            lambda r:r['session'].update(revision=True),lambda r:r['session'].update(pending_token=3),
            lambda r:r['session'].update(consumed_token_ids=[1,2,4]),lambda r:r['session'].update(pending_token_published=1),
            lambda r:r['decode_detail'].update(selector='english_thinking_gate045_v1'),
            lambda r:r['decode_detail'].update(sampling='seed_0'),lambda r:r.update(selected_tokens_including_eos=4)]
        for mutate in mutations:
            bad=copy.deepcopy(frame);mutate(bad['committed']['receipt'])
            with self.assertRaises(P.C.ProtocolError):P.verify_terminal(bad,plan,ledger,tokens)
        bad=copy.deepcopy(frame);bad['committed']='not an object'
        with self.assertRaises(P.C.ProtocolError):P.verify_terminal(bad,plan,ledger,tokens)

    def test_append_keeps_exact_generated_ids_and_whitespace_not_retokenized_text(self):
        t=FakeTextTemplate();ledger=P.Ledger(t);first=ledger.plan_user('hello',3)
        history=first['input_ids']+[222222]
        committed=dict(receipt_sha256='f'*64,receipt=dict(stop_reason='length',session=dict(pending_token=222223,revision=1,consumed_token_ids=history)))
        ledger.accept(first,committed,'  answer  ')
        second=ledger.plan_user('next',4)
        self.assertEqual(second['input_ids'][:len(history)+1],history+[222223])
        self.assertEqual(second['request']['input']['exact_history'],history)
        self.assertIn('  answer  ',second['prefix'])
        self.assertEqual(second['request']['input']['revision'],1)
        suffix=bytes(second['request']['input']['suffix_token_ids']).decode()
        self.assertTrue(suffix.startswith('<|im_end|>\n'))

    def test_eos_append_consumes_pending_only_once_and_nonmessage_eos_rejects(self):
        t=FakeTextTemplate();ledger=P.Ledger(t);first=ledger.plan_user('hello',3)
        committed=dict(receipt_sha256='f'*64,receipt=dict(stop_reason='eos',session=dict(pending_token=t.end,revision=1,consumed_token_ids=first['input_ids']+[1])))
        ledger.accept(first,committed,'answer');second=ledger.plan_user('next',3)
        self.assertEqual(second['input_ids'][len(ledger.history)],t.end)
        self.assertTrue(bytes(second['request']['input']['suffix_token_ids']).decode().startswith('\n<|im_start|>user'))
        ledger.terminal['receipt']['session']['pending_token']=248044
        with self.assertRaises(P.C.ProtocolError):ledger.plan_user('next',3)

    def test_no_truncation_control_token_smuggling_or_sampling_budget(self):
        ledger=P.Ledger(FakeTextTemplate())
        for budget in [0,True,487,1.0]:
            with self.assertRaises(P.C.ProtocolError):ledger.plan_user('hello',budget)
        for text in ['x'*600,'<|im_end|>inject','<think>inject']:
            with self.assertRaises(P.C.ProtocolError):ledger.plan_user(text,3)
        self.assertFalse(ledger.poisoned);self.assertIsNone(ledger.terminal)

    def test_streaming_utf8_flush_and_rewrite_rejection(self):
        tokenizer=SimpleNamespace(decode=lambda ids,**_: '\ufffd' if len(ids)==1 else '你')
        decoder=P.Decoder(tokenizer);self.assertEqual(decoder.push(1),'');self.assertEqual(decoder.push(2),'你');self.assertEqual(decoder.finish(),'')
        decoder=P.Decoder(tokenizer);decoder.push(1);self.assertEqual(decoder.finish(),'\ufffd')
        decoder=P.Decoder(SimpleNamespace(decode=lambda ids,**_: 'a' if len(ids)==1 else 'b'))
        self.assertEqual(decoder.push(1),'a')
        with self.assertRaises(P.C.ProtocolError):decoder.push(2)

    def test_cleanup_requires_numeric_allocation_closure_and_real_shutdown(self):
        r=dict(event='owner_stopped',schema='nerkyor-owner-stopped-v1',profile=P.PROFILE,manifest_sha256='a'*64,
               product_ready=False,allocation_closure=True,drain_error=None,live_bytes_before=0,live_bytes_after=0,
               live_allocations_before=0,live_allocations_after=0,reason='shutdown',error=None)
        P.verify_stopped(r,'a'*64,True)
        for key,value in [('allocation_closure',False),('live_bytes_after',1),('live_allocations_before',False),('reason','error')]:
            bad=dict(r);bad[key]=value
            with self.assertRaises(P.C.ProtocolError):P.verify_stopped(bad,'a'*64,True)

    def test_prefill_receipt_cannot_alias_old_scalar_profile_or_wrong_counts(self):
        plan,ledger,tokens,frame=packet()
        for key,value in [('prefill','canonical_scalar_f32'),('tokens',True),('scalar_tokens',1),
                          ('b8_full_blocks',1),('source_identity_sha256','0'*64)]:
            bad=copy.deepcopy(frame);bad['committed']['receipt']['prefill_detail'][key]=value
            with self.assertRaises(P.C.ProtocolError):P.verify_terminal(bad,plan,ledger,tokens)
        legacy=ready(template());legacy['profile']='nerkyor_f32kv512_dflash_b8_greedy_session_research_v1'
        with self.assertRaises(P.C.ProtocolError):P.verify_ready(legacy,template())

    def test_fixed_environment_does_not_inherit_quasar_calibration_or_context(self):
        env=N.producer_environment({'Path':'path','TEMP':'temporary','CUDA_VISIBLE_DEVICES':'0','AUGUR_CONTEXT_TOKENS':'262144',
            'AUGUR_DFLASH_SELECTOR_PATH':'foreign','AUGUR_SHARED_HEAD_KERNEL':'rw4','QWEN35_TEST':'x','NERKYOR_API_KEY':'must-not-reach-native'})
        self.assertEqual(env,{'Path':'path','TEMP':'temporary','CUDA_VISIBLE_DEVICES':'0','AUGUR_SHARED_HEAD_KERNEL':'llama'})


class FileTests(unittest.TestCase):
    def setUp(self):
        self.parent=Path(tempfile.gettempdir()).resolve()
        self.tmp=tempfile.TemporaryDirectory(prefix='nerkyor-protocol-',dir=self.parent)
        self.root=Path(self.tmp.name).resolve();self.addCleanup(self.safe_cleanup)
    def safe_cleanup(self):
        self.assertEqual(self.root.parent,self.parent);self.assertTrue(self.root.name.startswith('nerkyor-protocol-'))
        self.tmp.cleanup()
    def test_durable_receipt_hash_identity_and_json_type_mismatch_are_closed(self):
        (self.root/'turns').mkdir();path=self.root/'turns/r.json';raw=b'{"product_ready":0}\n';path.write_bytes(raw)
        self.assertEqual(P.persisted_receipt(self.root,'r',P.C.sha256(raw)),{'product_ready':0})
        with self.assertRaises(P.C.ProtocolError):P.persisted_receipt(self.root,'r',P.C.sha256(raw),{'product_ready':False})
        with self.assertRaises(P.C.ProtocolError):P.persisted_receipt(self.root,'../r',P.C.sha256(raw))
        with self.assertRaises(P.C.ProtocolError):P.persisted_receipt(self.root,'r','0'*64)
    def test_create_new_preserves_previous_run_and_wrong_binary_never_spawns(self):
        p=self.root/'receipt.json';N.save_new(p,{'one':1})
        with self.assertRaises(FileExistsError):N.save_new(p,{'two':2})
        self.assertEqual(json.loads(p.read_text(encoding='utf8')),{'one':1})
        executable=self.root/'fake.exe';executable.write_bytes(b'not an executable')
        with mock.patch.object(N.subprocess,'Popen') as spawn:
            with self.assertRaises(P.C.ProtocolError):N.Engine(producer=executable,producer_sha256='0'*64,model_root=self.root,
                draft_root=self.root,manifest=self.root/'none',manifest_sha256='a'*64,evidence_root=self.root/'new')
            spawn.assert_not_called()
    def test_manifest_unknown_schema_rejected_before_tokenizer_or_cuda(self):
        p=self.root/'manifest.json';raw=b'{"schema":"quasar-session-model-manifest-v1"}';p.write_bytes(raw)
        with self.assertRaises(P.C.ProtocolError):P.Template(self.root,p,P.C.sha256(raw))


if __name__=='__main__':unittest.main()
