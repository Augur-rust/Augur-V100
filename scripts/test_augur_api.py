"""CPU-only Augur API tests; native/model/long-history qualification is separate."""
import copy
import json
import threading
import unittest
import augur_api as Q
import qwen35_api as A
import qwen35_chat as C


class Tokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return ''.join('<|im_end|>' if t == 1000 else {1100:'a',1101:'b'}.get(t,chr(t)) for t in ids)


class Template:
    end, eos, vocab = 1000, {1000}, set(range(1200))
    manifest_sha256, manifest = 'a'*64, {'test':'augur-only'}
    tokenizer = Tokenizer()
    message = staticmethod(C.ChatTemplate.message)

    def encode(self, text):
        out = []
        while text:
            if text.startswith('<|im_end|>'):
                out.append(self.end); text = text[len('<|im_end|>'):]
            else:
                out.append(ord(text[0])); text = text[1:]
        return out

    def render(self, messages, tools=None, add_generation_prompt=True):
        return ''.join(f'<|im_start|>{m["role"]}\n{m["content"]}<|im_end|>\n' for m in messages) + (
            '<|im_start|>assistant\n' if add_generation_prompt else '')


def terminal(payload, prefix, tokens, prior, template, length=False):
    selected = len(tokens) + int(not length)
    history = prefix + (tokens[:-1] if length else tokens)
    s = dict(profile=getattr(template, 'profile', Q.PROFILE), manifest_sha256=template.manifest_sha256, session_id=payload['session_id'],
        revision=1 if prior is None else prior['receipt']['session']['revision']+1,
        consumed_token_ids=history, consumed_input_sha256=Q.token_sha(history),
        pending_token=tokens[-1] if length else template.end, pending_token_published=length)
    receipt = dict(schema='quasar-token-turn-receipt-v1',request_id=payload['request_id'], session=s,
        previous_receipt_sha256=prior['receipt_sha256'] if prior else None, output_token_ids=tokens,
        selected_tokens_including_eos=selected, stop_reason='length' if length else 'eos',
        product_ready=False, dflash_enabled=False, vision_enabled=False)
    return dict(event='terminal', request_id=payload['request_id'], session_id=payload['session_id'], product_ready=False,
        committed=dict(receipt=receipt, receipt_sha256='b'*64))


class Backend:
    def __init__(self):
        self.template, self.active, self.mode, self.outputs = Template(), None, 'ok', [79,75]
        self.closes = 0; self.requests = []; self.cache = None

    def health(self):
        return dict(profile=getattr(self.template, 'profile', Q.PROFILE), capacity=Q.CAPACITY, maximum_output_tokens=Q.MAX_OUTPUT,
            manifest_sha256=self.template.manifest_sha256, model_manifest=self.template.manifest,
            kv='Q8_0', lm_head='original_bf16', product_ready=False, qualified=False,
            actual_long_history_qualified=False, dflash_enabled=False, vision_enabled=False,
            restore_enabled=False, owner_resident=self.active is not None)

    def factory(self):
        backend = self
        class Transport:
            def __init__(self): self.cancelled = threading.Event()
            def abort(self): self.cancelled.set()
            def health(self): return backend.health()
            def close_session(self, ledger): backend.active=None; backend.closes+=1
            def frames(self, path, payload):
                backend.requests.append(copy.deepcopy(payload)); inp=payload['input']; prior=backend.active
                if inp['kind']=='open': prefix=inp['prompt_token_ids']; prior=None
                else:
                    assert inp['exact_history']==prior['receipt']['session']['consumed_token_ids']
                    prefix=inp['exact_history']+[prior['receipt']['session']['pending_token']]+inp['suffix_token_ids']
                budget=payload['max_new_tokens']; ids=backend.outputs[:budget]; length=len(backend.outputs)>=budget
                for index,t in enumerate(ids):
                    yield dict(event='token',product_ready=False, token=dict(request_id=payload['request_id'],session_id=payload['session_id'],sequence=index,token_id=t)),.01*(index+1)
                if backend.mode in ('native_error', 'foreign_native_error'):
                    yield dict(event='error', request_id=payload['request_id'],
                        session_id=payload['session_id'] if backend.mode == 'native_error' else 'foreign',
                        product_ready=False, reason='PRIVATE_NATIVE_ERROR'), .1
                    return
                frame=terminal(payload,prefix,ids,prior,backend.template,length)
                if backend.cache is not None:
                    frame['committed']['receipt']['prefill_detail'] = dict(input_prefix_cache=copy.deepcopy(backend.cache))
                backend.active=frame['committed']
                if backend.mode=='wrong_hash': frame['committed']['receipt']['session']['consumed_input_sha256']='c'*64
                if backend.mode=='wrong_profile':frame['committed']['receipt']['session']['profile']='resident'
                yield frame,.1
                if backend.mode=='truncated_eof': raise C.ProtocolError('missing zero HTTP chunk')
                if backend.mode=='post_terminal': yield frame,.11
        return Transport()


class Tests(unittest.TestCase):
    def setUp(self):
        self.backend=Backend(); self.controller=Q.Controller(self.backend.template,'unused','augur-test',transport_factory=self.backend.factory)

    def complete(self,messages=None,budget=24):
        result={}; emitted=[]
        request=dict(model='augur-test',messages=messages or [dict(role='user',content='Hi')],max_tokens=budget)
        self.controller.complete(request,lambda m,r:None,emitted.append,
            lambda meta,r,text,finish,usage:result.update(text=text,finish=finish,usage=usage,mode=meta['session_mode']),lambda abort:lambda:None)
        return result,emitted

    def test_error_frame_is_not_treated_as_terminal_or_committed(self):
        for mode, message in [('native_error', 'native engine failed'),
                              ('foreign_native_error', 'foreign native error frame')]:
            with self.subTest(mode=mode):
                self.setUp()
                self.backend.mode = mode
                with self.assertRaisesRegex(C.ProtocolError, message) as caught:
                    self.complete()
                self.assertIsNone(self.controller.current)
                self.assertNotIn('PRIVATE_NATIVE_ERROR', str(caught.exception))
                self.assertFalse(self.controller.busy.locked())

    def test_fixed_target_sampling_admission_and_capabilities(self):
        sampling = dict(Q.SAMPLING_VALUES, preset=Q.SAMPLING_PRESET, greedy=False,
                        verification='deterministic_draft_target_sample_equality_v1',
                        rng='splitmix64_absolute_position_v1')
        health = dict(self.backend.health(), sampling=sampling)
        self.controller.health = self.controller.check_health(health)
        self.assertFalse(self.controller.capabilities()['greedy'])
        request = dict(model='augur-test', messages=[dict(role='user', content='Hi')])
        for extra in ({}, Q.SAMPLING_VALUES):
            result = self.controller.validate_request(dict(request, **extra))
            self.assertEqual(result['sampling'], sampling)
        for key, wrong in [('temperature',0), ('top_p',1), ('top_k',16), ('min_p',0.1),
                           ('presence_penalty',1), ('frequency_penalty',1), ('seed',1),
                           ('seed',True), ('top_k',20.0), ('temperature',True), ('top_p',float('nan'))]:
            with self.subTest(key=key, wrong=wrong), self.assertRaises(A.ApiError):
                self.controller.validate_request(dict(request, **{key:wrong}))
        for field in ('rng', 'verification', 'temperature'):
            bad = dict(health, sampling=dict(sampling, **{field:'invalid'}))
            with self.assertRaises(C.ProtocolError):self.controller.check_health(bad)
        # The probability-draft contract is admitted only with its own RNG domains.
        speculative = dict(sampling, verification='sparse_p_over_q_positive_residual_v1',
                           rng='splitmix64_absolute_position_independent_purpose_domains_v1')
        self.controller.check_health(dict(health, sampling=speculative))
        with self.assertRaises(C.ProtocolError):
            self.controller.check_health(dict(health, sampling=dict(speculative, rng='splitmix64_absolute_position_v1')))

    def test_prefix_cache_usage_uses_committed_match_not_capture(self):
        count=len(self.backend.template.encode(self.backend.template.render([dict(role='user',content='Hi')])))
        for matched in (0, 7, count):
            with self.subTest(matched=matched):
                self.setUp()
                self.backend.cache=dict(matched_tokens=matched, captured_tokens=count,
                    hit=matched>0, exact_input_hit=matched==count, new_prefill_tokens=count-matched,
                    source='native_gdn_conv_kv_feature_logits_state')
                result,_=self.complete()
                self.assertEqual(result['mode'],'canonical_full_history_open')
                self.assertEqual(result['usage']['prompt_tokens_details']['cached_tokens'],matched)
                self.assertEqual(result['usage']['prompt_tokens'],count)

    def test_invalid_prefix_cache_receipt_is_not_published(self):
        valid=dict(matched_tokens=7,hit=True,exact_input_hit=False,new_prefill_tokens=42,
                   source='native_gdn_conv_kv_feature_logits_state')
        count=len(self.backend.template.encode(self.backend.template.render([dict(role='user',content='Hi')])))
        valid['new_prefill_tokens']=count-7
        for field,value in [('matched_tokens',-1),('matched_tokens',True),('matched_tokens',count+1),
                            ('hit',False),('exact_input_hit',True),('new_prefill_tokens',0),('source','unverified')]:
            with self.subTest(field=field,value=value):
                self.setUp();self.backend.cache=dict(valid,**{field:value})
                with self.assertRaises(C.ProtocolError):self.complete()
                self.assertIsNone(self.controller.current)

    def test_canonical_open_and_exact_append_both_pending_kinds(self):
        for budget in [1,24]:
            with self.subTest(budget=budget):
                self.setUp(); first,_=self.complete(budget=budget); prior=self.controller.current
                self.assertEqual(first['text'],'O' if budget==1 else 'OK')
                history=prior.history.copy();pending=prior.terminal['session']['pending_token']
                second,_=self.complete([dict(role='user',content='Hi'),dict(role='assistant',content=first['text']),dict(role='user',content='Next')])
                self.assertEqual(second['mode'],'exact_history_append')
                self.assertEqual(second['usage']['prompt_tokens_details']['cached_tokens'],len(history))
                inp=self.backend.requests[-1]['input'];self.assertEqual(inp['exact_history'],history)
                self.assertEqual(self.controller.current.history[:len(history)+1],history+[pending])
                self.assertEqual(self.backend.closes,0)

    def test_noncanonical_generated_bpe_is_not_reencoded_or_discarded(self):
        self.backend.outputs=[1100,1101]; first,_=self.complete();self.assertEqual(first['text'],'ab')
        self.complete([dict(role='user',content='Hi'),dict(role='assistant',content='ab'),dict(role='user',content='Next')])
        self.assertEqual(self.backend.requests[-1]['input']['exact_history'][-2:],[1100,1101])
        self.assertNotEqual(self.backend.template.encode('ab'),[1100,1101])

    def test_new_full_history_closes_prior_session_instead_of_false_reuse(self):
        self.complete();result,_=self.complete([dict(role='user',content='Other')])
        self.assertEqual(result['mode'],'canonical_full_history_open');self.assertEqual(self.backend.closes,1)

    def test_terminal_mismatch_incomplete_eof_and_extra_frames_do_not_commit(self):
        for mode in ['wrong_hash','wrong_profile','truncated_eof','post_terminal']:
            with self.subTest(mode=mode):
                self.setUp();self.backend.mode=mode
                with self.assertRaises(C.ProtocolError):self.complete()
                self.assertIsNone(self.controller.current);self.assertTrue(self.controller.faulted)

    def test_augur_health_must_not_inherit_legacy_identity(self):
        original=self.backend.health
        for key,value in [('capacity',196608),('lm_head','nvfp4'),('dflash_enabled',True),('manifest_sha256','c'*64)]:
            self.backend.health=lambda key=key,value=value:dict(original(),**{key:value})
            with self.assertRaises(C.ProtocolError):Q.Controller(self.backend.template,'unused','test',transport_factory=self.backend.factory)
        self.backend.health=original

    def test_recovery_rechecks_model_identity_before_any_new_generation(self):
        self.controller.faulted = True
        original = self.backend.health
        self.backend.health = lambda: dict(original(), manifest_sha256='c'*64, owner_resident=False)
        with self.assertRaises(C.ProtocolError): self.complete()
        self.assertEqual(self.backend.requests, [])
        self.assertTrue(self.controller.faulted)
        self.backend.health = original
        result, _ = self.complete()
        self.assertEqual(result['text'], 'OK')
        self.assertFalse(self.controller.faulted)

    def cancellation_health(self):
        original = self.backend.health()
        self.controller.health['producer_sha256'] = 'c'*64
        return dict(original, producer_sha256='c'*64, owner_resident=True, session_active=False,
            request_bound_cancellation_discard=True, last_cancel_discard=dict(receipt_sha256='d'*64,
            receipt=dict(schema='quasar-session-cancellation-discard-v1', request_id='discard-1',
                cancelled_request_id='failed-request', session_id='failed-session',
                manifest_sha256=self.backend.template.manifest_sha256,
                cancellation_boundary='before Augur selected-token consume',
                submitted_tokens_before_discard=12, submitted_tokens_after_discard=0,
                drain_completed=True, reset_completed=True, session_discarded=True,
                previous_history_reusable=False, product_ready=False, dflash_enabled=False)))

    def test_exact_discard_receipt_allows_new_session_without_claiming_history_reuse(self):
        health = self.cancellation_health()
        self.backend.health = lambda: copy.deepcopy(health)
        self.controller.faulted = True
        self.controller.failed_request = ('failed-request', 'failed-session')
        result, _ = self.complete()
        self.assertEqual(result['text'], 'OK')
        self.assertEqual(result['mode'], 'canonical_full_history_open')
        self.assertIsNone(self.controller.failed_request)
        self.assertFalse(self.controller.faulted)

    def test_discard_health_without_local_failed_request_is_not_authority(self):
        health = self.cancellation_health()
        self.backend.health = lambda: health
        self.controller.faulted = True
        with self.assertRaises(A.ApiError): self.complete()
        self.assertEqual(self.backend.requests, [])
        self.assertTrue(self.controller.faulted)

    def test_foreign_stale_or_incomplete_discard_does_not_clear_failure(self):
        good = self.cancellation_health()
        changes = [(['producer_sha256'], 'e'*64), (['session_active'], True),
            (['request_bound_cancellation_discard'], False),
            (['last_cancel_discard', 'receipt_sha256'], 'wrong'),
            *[(['last_cancel_discard', 'receipt', key], value) for key, value in [
                ('cancelled_request_id', 'stale'), ('session_id', 'foreign'),
                ('manifest_sha256', 'e'*64), ('reset_completed', False), ('drain_completed', False),
                ('session_discarded', False), ('previous_history_reusable', True),
                ('submitted_tokens_after_discard', 1), ('submitted_tokens_after_discard', False),
                ('submitted_tokens_before_discard', -1), ('submitted_tokens_before_discard', Q.CAPACITY+1),
                ('dflash_enabled', True), ('product_ready', True), ('cancellation_boundary', ''),
                ('request_id', None), ('schema', 'other')]]]
        for keys, value in changes:
            with self.subTest(keys=keys, value=value):
                h = copy.deepcopy(good); target = h
                for key in keys[:-1]: target = target[key]
                target[keys[-1]] = value
                self.backend.health = lambda: copy.deepcopy(h)
                self.controller.faulted = True
                self.controller.failed_request = ('failed-request', 'failed-session')
                with self.assertRaises(A.ApiError): self.complete()
                self.assertEqual(self.backend.requests, [])
                self.assertTrue(self.controller.faulted)

    def test_native_failure_remembers_exact_failed_turn_not_last_good_history(self):
        self.backend.mode = 'truncated_eof'
        with self.assertRaises(C.ProtocolError): self.complete()
        request = self.backend.requests[-1]
        self.assertEqual(self.controller.failed_request, (request['request_id'], request['session_id']))

    def test_augur_profile_is_explicit_and_legacy_mismatch_fails(self):
        self.backend.template.profile = Q.AUGUR_PROFILE
        self.controller = Q.Controller(self.backend.template, 'unused', 'augur-test', transport_factory=self.backend.factory)
        self.assertEqual(self.controller.capabilities()['profile'], Q.AUGUR_PROFILE)
        first, _ = self.complete()
        self.assertEqual(first['text'], 'OK')
        self.assertEqual(self.controller.current.terminal['session']['profile'], Q.AUGUR_PROFILE)
        health = self.backend.health()
        health['profile'] = Q.PROFILE
        with self.assertRaises(C.ProtocolError): self.controller.check_health(health)
        with self.assertRaises(C.ProtocolError): Q.Transport('http://127.0.0.1:1', 1, 'foreign')

    def test_verified_receipt_copy_has_no_nested_json_aliases(self):
        source = {
            "output_token_ids": [1, 2],
            "session": {"consumed_token_ids": [3, 4], "nested": {"values": [5]}},
            "decode_detail": {"dflash": {"by_position": [[1, 0, 0, 0]]}},
        }
        copied = Q.copy_verified_receipt(source)
        self.assertEqual(copied, source)
        copied["output_token_ids"].append(9)
        copied["session"]["consumed_token_ids"].append(8)
        copied["session"]["nested"]["values"].append(7)
        copied["decode_detail"]["dflash"]["by_position"][0].append(6)
        self.assertEqual(source["output_token_ids"], [1, 2])
        self.assertEqual(source["session"]["consumed_token_ids"], [3, 4])
        self.assertEqual(source["session"]["nested"]["values"], [5])
        self.assertEqual(source["decode_detail"]["dflash"]["by_position"], [[1, 0, 0, 0]])
    def test_native_bill_is_retained_as_an_independent_copy(self):
        self.complete()
        terminal = self.controller.current.terminal
        source = self.backend.active['receipt']
        self.assertEqual(terminal['native_receipt'], source)
        source['output_token_ids'].append(99)
        self.assertNotEqual(terminal['native_receipt']['output_token_ids'], source['output_token_ids'])

    def test_capacity_is_256ki_without_mutating_legacy_limits(self):
        template=self.backend.template
        # Avoid quadratic fake-character tokenizer on this large CPU fixture.
        template.encode=lambda text:[1]*220000
        plan=Q.Ledger(template).plan([dict(role='user',content='long')],4096)
        self.assertEqual(len(plan['input_ids']),220000)
        self.assertEqual(C.PROFILES['resident'][2],196608)
        template.encode=lambda text:[1]*Q.CAPACITY
        with self.assertRaises(C.ProtocolError):Q.Ledger(template).plan([dict(role='user',content='long')],1)
        with self.assertRaises(A.ApiError):self.complete(budget=4097)

    def test_claims_and_restore_stay_closed(self):
        caps=self.controller.capabilities()
        self.assertEqual(caps['full_kv_capacity'],262144)
        for key in ['product_ready','qualified','actual_long_history_qualified','dflash_enabled','vision_forward','startup_checkpoint_restore']:
            self.assertFalse(caps[key])
        with self.assertRaises(A.ApiError):self.controller.restore_saved({})

    def test_utf8_partial_is_delayed_and_non_eos_special_text_preserved(self):
        class T:
            def decode(self,ids,skip_special_tokens=False):
                assert not skip_special_tokens
                return '\ufffd' if ids==[1] else '中' if ids==[1,2] else '<tool_call>'
        d=Q.Decoder(T());self.assertEqual(d.push(1),'');self.assertEqual(d.push(2),'中');self.assertEqual(d.finish(),'')
        d=Q.Decoder(T());self.assertEqual(d.push(3),'<tool_call>')
        d=Q.Decoder(T());self.assertEqual(d.push(1),'');self.assertEqual(d.finish(),'\ufffd')

    def test_tools_validate_before_publication_and_keep_exact_xml_history(self):
        from test_qwen35_tools import ToolTemplate, TOOLS, RAW
        self.backend.template.render = ToolTemplate.render.__get__(self.backend.template)
        self.backend.outputs = self.backend.template.encode(RAW)
        emitted, result = [], {}
        payload = dict(model='augur-test', messages=[dict(role='user', content='Lookup alpha')],
                       tools=TOOLS, max_tokens=512)
        self.controller.complete(payload,lambda *a:None,emitted.append,
            lambda meta,r,text,finish,usage:result.update(calls=r.get('response_tool_calls'),finish=finish),lambda abort:lambda:None)
        self.assertEqual(result['finish'],'tool_calls')
        self.assertEqual(result['calls'][0]['function']['name'],'lookup')
        self.assertTrue(emitted and all(isinstance(x,dict) for x in emitted))
        history = self.controller.current.history.copy()
        # Completed EOF publishes tool calls; a truncated HTTP EOF must not.
        self.setUp();self.backend.template.render = ToolTemplate.render.__get__(self.backend.template)
        self.backend.outputs = self.backend.template.encode(RAW);self.backend.mode='truncated_eof';emitted=[]
        with self.assertRaises(C.ProtocolError):
            self.controller.complete(payload,lambda *a:None,emitted.append,lambda *a:self.fail('committed'),lambda abort:lambda:None)
        self.assertEqual(emitted,[])

    def test_token_digest_preserves_little_endian_native_ledger_contract(self):
        self.assertEqual(Q.token_sha([]), 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855')
        self.assertEqual(Q.token_sha([0, 1, 255, 256, 65535, 65536, 248319]),
                         '8a139974a4add2bf982b4ab7cbf8f059c184b95e895ce956d0aca28bd09a0b94')
        self.assertEqual(Q.token_sha([248319] * Q.CAPACITY),
                         'c680296a7869271b2b14d3773b9c8bfe5f888a847e8a73f880bf4b00cc6fe6c3')
        for invalid in ([True], [-1], [248320], [1.0], (1, 2)):
            with self.subTest(invalid=invalid), self.assertRaises(C.ProtocolError):
                Q.token_sha(invalid)

    def test_output_and_restore_errors_do_not_declare_success(self):
        with self.assertRaises(A.ApiError):self.complete(budget=0)
        self.assertEqual(self.backend.requests,[])
        with self.assertRaises(C.ProtocolError):Q.token_sha([True])

    def test_trimmed_assistant_append_preserves_exact_native_ids(self):
        template=self.backend.template
        def trim_render(messages,tools=None,add_generation_prompt=True):
            return ''.join(f'<|im_start|>{m["role"]}\n{m["content"].strip()}<|im_end|>\n' for m in messages)+(
                '<|im_start|>assistant\n' if add_generation_prompt else '')
        template.render=trim_render
        for length in (False,True):
            with self.subTest(length=length):
                ledger=Q.Ledger(template);messages=[dict(role='user',content='Name a color')]
                plan=ledger.plan(messages,24);tokens=[32,1100,1101,32]
                frame=terminal(plan['payload'],plan['input_ids'],tokens,None,template,length)
                committed=frame['committed']['receipt']
                term=dict(session=committed['session'],stop_reason=committed['stop_reason'],native_report_sha256='b'*64)
                ledger.accept(plan,term,' ab ')
                next_messages=ledger.messages+[dict(role='user',content='Again')]
                follow=ledger.plan(next_messages,24)
                self.assertEqual(follow['input_ids'][:len(ledger.history)],ledger.history)
                self.assertEqual(follow['input_ids'][len(ledger.history)],term['session']['pending_token'])
                suffix=''.join(template.tokenizer.decode([t]) for t in follow['payload']['input']['suffix_token_ids'])
                self.assertTrue(suffix.startswith('<|im_end|>\n' if length else '\n'))
                ledger.accept(follow,term,' ab ')
                changed=copy.deepcopy(ledger.messages);changed[0]['content']='different'
                with self.assertRaises(C.ProtocolError):ledger.plan(changed+[dict(role='user',content='Again')],24)


if __name__=='__main__':unittest.main()


class DflashProfile(unittest.TestCase):
    def receipt(self):
        return dict(dflash_enabled=True, decode_detail=dict(dflash=dict(profile=Q.DFLASH_PROFILE,
            active=True,block_width=8,target_projection='packed_b8',draft_projection='llama',
            draft_head='llama_b8',context='retained',B1_numerical_equivalence_qualified=False,
            rounds=2,proposed_draft_tokens=14,accepted_draft_tokens=8,checked_draft_tokens=10)))

    def test_opt_in_receipt_requires_both_policy_and_bounded_integer_counters(self):
        good=self.receipt();Q.validate_dflash_receipt(good,Q.DFLASH_PROFILE)
        for key,value in [('profile',Q.AUGUR_PROFILE),('active',False),('block_width',1),
            ('target_projection','expanded_f16'),('B1_numerical_equivalence_qualified',True),
            ('rounds',True),('rounds',-1),('proposed_draft_tokens',15),('accepted_draft_tokens',11),
            ('checked_draft_tokens',15),('draft_head','scalar')]:
            bad=copy.deepcopy(good);bad['decode_detail']['dflash'][key]=value
            with self.subTest(key=key),self.assertRaises(C.ProtocolError): Q.validate_dflash_receipt(bad,Q.DFLASH_PROFILE)

    def test_active_and_resident_profiles_never_silently_interchange(self):
        for profile in (Q.PROFILE,Q.AUGUR_PROFILE):
            with self.assertRaises(C.ProtocolError): Q.validate_dflash_receipt(self.receipt(),profile)
        for receipt in (dict(dflash_enabled=False),dict(dflash_enabled=True),dict(dflash_enabled=1)):
            with self.assertRaises(C.ProtocolError): Q.validate_dflash_receipt(receipt,Q.DFLASH_PROFILE)
        Q.validate_dflash_receipt(dict(dflash_enabled=False),Q.AUGUR_PROFILE)
        bad=self.receipt();bad['dflash_enabled']=False
        with self.assertRaises(C.ProtocolError): Q.validate_dflash_receipt(bad,Q.AUGUR_PROFILE)

    def test_empty_one_token_generation_has_zero_valid_acceptance_counters(self):
        r=self.receipt();d=r['decode_detail']['dflash']
        for key in ('rounds','proposed_draft_tokens','accepted_draft_tokens','checked_draft_tokens'): d[key]=0
        Q.validate_dflash_receipt(r,Q.DFLASH_PROFILE)

    def test_active_health_and_capabilities_are_bound_not_merely_permitted(self):
        b=Backend();b.template.profile=Q.DFLASH_PROFILE;original=b.health
        b.health=lambda:dict(original(),dflash_enabled=True)
        c=Q.Controller(b.template,'unused','test',transport_factory=b.factory)
        self.assertTrue(c.capabilities()['dflash_enabled'])
        for flag in (False,None,1):
            b.health=lambda flag=flag:dict(original(),dflash_enabled=flag)
            with self.assertRaises(C.ProtocolError): Q.Controller(b.template,'unused','test',transport_factory=b.factory)


class LongTransportTests(unittest.TestCase):
    @staticmethod
    def response(count, truncate=False):
        from types import SimpleNamespace
        def chunks():
            for start in range(0, count, 256):
                body = b''.join((json.dumps(dict(event='token', product_ready=False,
                    token=dict(request_id='r'*64, session_id='s'*64, sequence=i,
                               token_id=248319)), separators=(',', ':'))+'\n').encode()
                    for i in range(start, min(start+256, count)))
                yield f'{len(body):X}\r\n'.encode()+body+b'\r\n'
            if not truncate:
                body=b'{"event":"terminal"}\n'
                yield f'{len(body):X}\r\n'.encode()+body+b'\r\n0\r\n\r\n'
        class Reader:
            raw=SimpleNamespace(_sock=SimpleNamespace(settimeout=lambda n: None))
            def __init__(self): self.iterator=chunks(); self.pending=b''
            def read1(self,n):
                if not self.pending: self.pending=next(self.iterator,b'')
                value,self.pending=self.pending[:n],self.pending[n:]
                return value
        return SimpleNamespace(fp=Reader())

    def test_full_output_budget_exceeds_legacy_limit_but_completes_augur_stream(self):
        import time
        count=Q.MAX_OUTPUT
        seen=0
        for frame,_ in Q.Transport('http://127.0.0.1:1',60)._native_frames(self.response(count),time.perf_counter()):
            if frame['event']=='token':
                self.assertEqual(frame['token']['sequence'],seen)
                seen+=1
            else:
                self.assertEqual(frame['event'],'terminal')
        self.assertEqual(seen,249036)
        with self.assertRaises(C.GenerationFailure) as error:
            list(C.Transport('http://127.0.0.1:1',60)._native_frames(self.response(count),time.perf_counter()))
        self.assertEqual(error.exception.code,'native_response_limit')
        self.assertEqual(error.exception.counters['limit_bytes'],16*1024*1024)

    def test_larger_byte_budget_still_requires_eof_and_bounds_every_frame(self):
        import time
        with self.assertRaisesRegex(C.ProtocolError,'truncated HTTP chunk'):
            list(Q.Transport('http://127.0.0.1:1',60)._native_frames(self.response(128,True),time.perf_counter()))
        transport=Q.Transport('http://127.0.0.1:1',60)
        transport.max_frame=32
        with self.assertRaisesRegex(C.ProtocolError,'oversized NDJSON'):
            list(transport._native_frames(self.response(1),time.perf_counter()))
        transport=Q.Transport('http://127.0.0.1:1',60)
        transport.max_response=32
        with self.assertRaises(C.GenerationFailure):
            list(transport._native_frames(self.response(1),time.perf_counter()))
