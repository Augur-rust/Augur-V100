"""Monitor metric denominators and observer lifecycle; no GPU required."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import augur_monitor as M
import augur_live_telemetry as L

class MetricsTests(unittest.TestCase):
    def test_distinct_draft_and_input_denominators_and_no_private_payload(self):
        receipt=dict(request_id='turn-test',output_token_ids=[3]*112,prefill_wall_s=8.0,
            whole_wall_s_before_receipt=10.0,generation_complete_unix_ns=123000000000,
            prompt='PRIVATE',session={'consumed_token_ids':[12345]},
            decode_detail={'dflash':dict(active=True,rounds=24,accepted_draft_tokens=88,
                                        proposed_draft_tokens=168,checked_draft_tokens=105)})
        row=M.attach_http_metrics(M.summarize_receipt(receipt),dict(api_wall_s=11.0,
            messages=['PRIVATE'],usage=dict(prompt_tokens=1000,prompt_tokens_details=dict(cached_tokens=900))))
        self.assertAlmostEqual(row['accepted_over_proposed'],88/168)
        self.assertAlmostEqual(row['accepted_over_checked'],88/105)
        self.assertAlmostEqual(row['mean_accepted_drafts'],88/24)
        self.assertEqual(row['input_cache_coverage'],.9)
        self.assertEqual(row['native_generation_tps'],56)
        self.assertAlmostEqual(row['http_output_tps'],112/11)
        encoded=json.dumps(row)
        for forbidden in ('PRIVATE','12345','output_token_ids','consumed_token_ids'):
            self.assertNotIn(forbidden,encoded)

    def test_absent_counters_are_unknown_and_real_zero_stays_zero(self):
        row=M.attach_http_metrics(M.summarize_receipt({}),{})
        for key in ('accepted_over_proposed','accepted_over_checked','mean_accepted_drafts','input_cache_coverage'):
            self.assertIsNone(row[key])
        self.assertEqual(M.finite_ratio(0,7),0)
        for numerator,denominator in ((1,0),(None,7),(True,7),(1,float('nan')),(float('inf'),7),(-1,7)):
            self.assertIsNone(M.finite_ratio(numerator,denominator))

class DecodeModeTests(unittest.TestCase):
    def test_modes_are_explicit_and_native_evidence_wins(self):
        for preset, mode in [('ninfer_thinking_v1','sampling'), ('greedy','greedy'), ('unknown','unknown')]:
            receipt=dict(decode_detail=dict(sampling=dict(preset=preset, private_tokens=[12345])))
            row=M.summarize_receipt(receipt)
            self.assertEqual(row['decode_mode'],mode)
            self.assertNotIn('12345',json.dumps(row))
        row=M.summarize_receipt(dict(decode_detail=dict(sampling=dict(preset='ninfer_thinking_v1'))))
        M.attach_http_metrics(row,dict(sampling=dict(preset='greedy')))
        self.assertEqual(row['decode_mode'],'sampling')
        self.assertEqual(M.summarize_receipt({})['decode_mode'],'unknown')

class VisionTests(unittest.TestCase):
    def test_residency_is_receipt_state_not_inferred_from_gpu_usage(self):
        self.assertIsNone(M.summarize_receipt({})['vision_residency'])
        for resident in (True, False):
            value=dict(schema='quasar-vision-residency-v1', gpu_weights_resident=resident,
                       resident_bytes=928111554 if resident else 0, evictions=int(not resident))
            row=M.summarize_receipt(dict(decode_detail=dict(vision_residency=value)))
            self.assertEqual(row['vision_residency'], value)


class PrefillTests(unittest.TestCase):
    @staticmethod
    def receipt(detail=None, seconds=2.0):
        return dict(prefill_detail=detail, prefill_wall_s=seconds)

    @staticmethod
    def component(widths, seconds):
        return dict(plan=dict(widths=widths), execute_with_upload_head_capture_seconds=seconds)

    def test_native_execution_excludes_outer_request_overhead(self):
        row=M.summarize_receipt(self.receipt(self.component([4352,3824],1.5),seconds=9.0))
        self.assertEqual(row['prefill_tokens'],8176)
        self.assertAlmostEqual(row['prefill_tps'],8176/1.5)
        self.assertEqual(row['prefill_s'],1.5)
        self.assertEqual(row['prefill_wall_s'],9.0)
        self.assertEqual(row['prefill_clock'],'native_execute_with_upload_head_capture_seconds')

    def test_real_short_prompt_uses_execution_not_snapshot_or_setup(self):
        detail={'input_prefix_cache':{'new_prefill_tokens':176,'capture_s':.4713526},
                'suffix_prefill':dict(self.component([176],.377719),setup_seconds=.0293358,
                                      complete_wall_seconds=.4081455)}
        row=M.summarize_receipt(self.receipt(detail,seconds=.8797537))
        self.assertAlmostEqual(row['prefill_tps'],465.95485003401996)
        self.assertAlmostEqual(row['prefill_s'],.377719)
        self.assertEqual(row['prefill_rate_status'],'measured')

    def test_partial_cache_counts_only_executed_rows(self):
        detail={'input_prefix_cache':{'new_prefill_tokens':2296,'restore_s':8.0},
                'suffix_prefill':self.component([2296],2.0)}
        row=M.summarize_receipt(self.receipt(detail,seconds=10.0))
        row=M.attach_http_metrics(row,{'session_mode':'canonical_full_history_open','usage':{'prompt_tokens':32760,'prompt_tokens_details':{'cached_tokens':30464}}})
        self.assertEqual(row['prefill_tokens'],2296)
        self.assertEqual(row['prefill_tps'],1148)
        self.assertAlmostEqual(row['input_cache_coverage'],30464/32760)

    def test_vision_native_prefill_excludes_encoder_time(self):
        detail={'vision_encoder_wall_including_upload_release_s':1.2,
                'language_prefill':self.component([1024],.8)}
        row=M.summarize_receipt(self.receipt(detail,seconds=2.1))
        self.assertEqual(row['prefill_tokens'],1024)
        self.assertEqual(row['prefill_tps'],1280)

    def test_split_prefix_needs_matching_timers_for_all_rows(self):
        detail={'input_prefix_cache':{'new_prefill_tokens':8176,'recomputed_prefix_tokens':4352},
                'suffix_prefill':self.component([3824],3.0)}
        row=M.summarize_receipt(self.receipt(detail,seconds=8.0))
        self.assertIsNone(row['prefill_tps'])
        detail['prefix_prefill']=self.component([4352],3.5)
        row=M.summarize_receipt(self.receipt(detail,seconds=8.0))
        self.assertAlmostEqual(row['prefill_tps'],8176/6.5)

    def test_zero_rows_are_not_infinite_throughput(self):
        for detail in ({'input_prefix_cache':{'new_prefill_tokens':0,'exact_input_hit':True},'suffix_prefill':None}, {'plan':{'widths':[]}}):
            row=M.summarize_receipt(self.receipt(detail))
            self.assertIsNone(row['prefill_tps'])
            self.assertEqual(row['prefill_rate_status'],'not_executed')

    def test_invalid_or_conflicting_counts_not_overridden_by_http(self):
        details=[{'plan':{'widths':[True]}},{'plan':{'widths':[0]}},{'plan':{'widths':[-1]}},
            {'plan':{'widths':[1.5]}},{'plan':{'widths':'4352'}},
            {'input_prefix_cache':{'new_prefill_tokens':True}},
            {'input_prefix_cache':{'new_prefill_tokens':12,'exact_input_hit':True}},
            {'input_prefix_cache':{'new_prefill_tokens':12},'plan':{'widths':[13]}}]
        for detail in details:
            with self.subTest(detail=detail):
                row=M.attach_http_metrics(M.summarize_receipt(self.receipt(detail)),{'session_mode':'canonical_full_history_open','usage':{'prompt_tokens':99,'prompt_tokens_details':{'cached_tokens':0}}})
                self.assertIsNone(row['prefill_tokens'])
                self.assertIsNone(row['prefill_tps'])

    def test_http_usage_cannot_supply_a_missing_native_execution_clock(self):
        row=M.attach_http_metrics(M.summarize_receipt(self.receipt()),{'session_mode':'canonical_full_history_open','usage':{'prompt_tokens':100,'prompt_tokens_details':{'cached_tokens':90}}})
        self.assertEqual(row['prefill_tokens'],10)
        self.assertIsNone(row['prefill_tps'])
        self.assertEqual(row['prefill_rate_status'],'invalid_clock')

    def test_missing_invalid_or_unmatched_native_clock_never_uses_outer_wall(self):
        for seconds in (None,True,0,-1,float('nan'),float('inf')):
            row=M.summarize_receipt(self.receipt(self.component([16],seconds),seconds=2.0))
            self.assertIsNone(row['prefill_tps'])
            self.assertEqual(row['prefill_rate_status'],'invalid_clock')
        row=M.summarize_receipt(self.receipt({'plan':{'widths':[16]}}))
        self.assertIsNone(row['prefill_tps'])
        detail={'input_prefix_cache':{'new_prefill_tokens':17},'suffix_prefill':self.component([16],1.0)}
        self.assertIsNone(M.summarize_receipt(self.receipt(detail))['prefill_tps'])
        self.assertEqual(M.summarize_receipt({})['prefill_rate_status'],'unknown')


class GpuMemoryTests(unittest.TestCase):
    def test_free_memory_is_driver_reported_not_total_minus_used(self):
        with tempfile.TemporaryDirectory() as directory:
            sampler=M.Sampler(Path(directory), Path(directory)/'missing-live.json', 8294)
            with patch.object(M.subprocess, 'run') as run, patch.object(M.socket, 'create_connection', side_effect=OSError):
                run.return_value.stdout='0, GPU, 100, 32494, 32768, 60, 250, 133\n'
                gpu=sampler.collect()['gpu']
                self.assertEqual(gpu['free_mib'], 133)
                self.assertNotEqual(gpu['free_mib'], gpu['total_mib']-gpu['used_mib'])
                self.assertIn('memory.free', run.call_args.args[0][1])
                run.return_value.stdout='0, GPU, 0, 10, 32768, 40, 30, [N/A]\n'
                self.assertIsNone(sampler.collect()['gpu']['free_mib'])


class ContextTests(unittest.TestCase):
    def test_stream_counts_grow_reset_and_commit_pending_token(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(L.threading.Thread, 'start'):
            tracker=L.Tracker(Path(directory)/'live.json')
            tracker.start('a',249035,262144)
            self.assertEqual(tracker.snapshot()['context_used_tokens'],249035)
            tracker.token()
            live=tracker.snapshot()
            result=M.context_accounting(live,None,{},True)
            self.assertEqual(result['used_tokens'],249036)
            self.assertTrue(result['compression_due'])
            self.assertEqual(result['remaining_to_compression_tokens'],0)
            tracker.end('completed',{'session':{'consumed_token_ids':[0]*249036,'pending_token':2}})
            self.assertEqual(tracker.snapshot()['context_used_tokens'],249037)
            tracker.start('b',40,8192)
            self.assertEqual(tracker.snapshot()['context_used_tokens'],40)
            self.assertEqual(tracker.snapshot()['output_tokens'],0)
            tracker.token();tracker.end('failed_or_cancelled')
            self.assertIsNone(tracker.snapshot()['context_used_tokens'])
            self.assertIsNone(M.context_accounting(tracker.snapshot(),{'context_used_tokens':100},{},True)['used_tokens'])

    def test_length_stop_pending_is_counted_once(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(L.threading.Thread, 'start'):
            tracker=L.Tracker(Path(directory)/'live.json');tracker.start('a',20,8192)
            for _ in range(3):tracker.token()
            tracker.end('completed',{'session':{'consumed_token_ids':[0]*22,'pending_token':9,'pending_token_published':True}})
            self.assertEqual(tracker.snapshot()['context_used_tokens'],23)

    def test_capacity_threshold_allocation_and_offline_are_distinct(self):
        for n in (8192,32768,65536,131072,196608,262144):
            ready={'capacity':n,'target':{'kv_cache_bytes':n*34816}}
            row={'context_used_tokens':10}
            live=M.context_accounting(None,row,ready,True)
            offline=M.context_accounting(None,row,ready,False)
            self.assertEqual(live['compression_threshold_tokens'],n*95//100)
            self.assertEqual(live['kv_allocated_bytes'],n*34816)
            self.assertEqual(offline['used_tokens'],10)
            self.assertEqual(offline['kv_allocated_bytes'],0)
            self.assertFalse(live['compression_automatic_in_engine'])

    def test_untrusted_counts_do_not_look_like_available_context(self):
        for bad in (None,True,-1,9000,1.5):
            row=M.context_accounting(None,{'context_used_tokens':bad},{'capacity':8192},True)
            self.assertIsNone(row['used_tokens']);self.assertIsNone(row['compression_due'])
        self.assertIsNone(M.receipt_context({'session':{'consumed_token_ids':[1,2]}}))
        self.assertEqual(M.receipt_context({'session':{'consumed_token_ids':[1,2],'pending_token':3}}),3)


class ObserverTests(unittest.TestCase):
    def test_completed_elapsed_stops_and_new_request_resets(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(L.threading.Thread,'start'), patch.object(L.time,'monotonic') as clock:
            tracker=L.Tracker(Path(directory)/'live.json')
            clock.return_value=10;tracker.start('a')
            clock.return_value=11;tracker.token()
            clock.return_value=13;tracker.end('completed')
            clock.return_value=90;s=tracker.snapshot()
            self.assertEqual(s['elapsed_s'],3)
            self.assertEqual(s['ttft_s'],1)
            self.assertEqual(s['output_tokens'],1)
            self.assertIsNone(s['rolling_5s_tps'])
            tracker.start('b');clock.return_value=92;s=tracker.snapshot()
            self.assertEqual(s['elapsed_s'],2)
            self.assertEqual(s['output_tokens'],0)
            self.assertIsNone(s['ttft_s'])

    def test_rejected_or_cancelled_frames_are_not_counted(self):
        class Transport:
            def frames(self,path,payload):
                try:
                    yield {'event':'token'},0
                    yield {'event':'terminal'},1
                finally: self.closed=True
        class Counter:
            count=0
            def token(self):self.count+=1
        for reject in (True,False):
            t=Transport();c=Counter();stream=L.ObservedTransport(t,c).frames('/',{})
            next(stream)
            self.assertEqual(c.count,0)
            if not reject:
                next(stream);self.assertEqual(c.count,1)
            stream.close();self.assertTrue(t.closed)
            self.assertEqual(c.count,0 if reject else 1)



class MonitorHistoryUiTests(unittest.TestCase):
    def test_clear_history_browser_storage_and_new_requests(self):
        import shutil
        import subprocess
        node = shutil.which('node')
        if not node:
            self.skipTest('Node.js required for monitor DOM regression test')
        script = r"""
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(process.argv[2],'utf8')+';'+fs.readFileSync(process.argv[1],'utf8');
class Element{
 constructor(){this.children=[];this.events={};this.style={};this.classList={toggle(){}};this.textContent='';this.disabled=false}
 append(...nodes){this.children.push(...nodes)}
 replaceChildren(...nodes){this.children=nodes}
 setAttribute(){} removeAttribute(){}
 addEventListener(event,fn){this.events[event]=fn}
}
function page(store={},blocked=false){
 const elements={},events={};
 const element=id=>elements[id]||(elements[id]=new Element());
 const context={document:{getElementById:element,createElement:()=>new Element(),createTreeWalker:()=>({nextNode:()=>null}),querySelectorAll:()=>[],documentElement:{},body:{}},NodeFilter:{SHOW_TEXT:4},location:{hash:'',pathname:'/'},URLSearchParams,history:{replaceState(){}},window:{addEventListener:(name,fn)=>events[name]=fn},
 localStorage:{getItem:key=>{if(blocked)throw Error('blocked');return store[key]??null},setItem:(key,value)=>{if(blocked)throw Error('blocked');store[key]=value}},
 fetch:()=>new Promise(()=>{}),setTimeout:()=>{},console};
 vm.createContext(context);
 vm.runInContext(source+'\nglobalThis.testApi={render,visibleRequests,clearRequestHistory};',context);
 return {element,events,api:context.testApi};
}
const row=t=>({completed_unix_s:t,request_id:'turn-'+t,output_tokens:3,prefill_rate_status:'measured',prefill_tps:200,prefill_s:.1,prefill_tokens:20});
const data={updated_unix_s:102,recent:[row(101),row(100)],latest:row(101),context:{},history:[],status:'idle'};
const store={},a=page(store);a.api.render(data);
assert.equal(a.element('requests').children.length,2);assert.equal(a.element('clear-history').disabled,false);
a.element('clear-history').events.click();
assert.equal(a.api.visibleRequests(data).length,0);assert.equal(a.element('clear-history').disabled,true);
assert.match(a.element('requests').children[0].children[0].textContent,/历史记录已清空/);
assert.match(a.element('history-status').textContent,/原始账单/);
assert.equal(a.element('prefill-tps').textContent,'200.0');
assert.equal(data.recent.length,2);assert.equal(data.latest.request_id,'turn-101');
// Polling an old sample and reloading the page must not resurrect cleared rows.
a.api.render(data);assert.equal(a.api.visibleRequests(data).length,0);
const b=page(store);b.api.render(data);assert.equal(b.api.visibleRequests(data).length,0);
const newData={...data,updated_unix_s:104,recent:[row(103),...data.recent]};b.api.render(newData);
assert.equal(b.api.visibleRequests(newData).length,1);assert.equal(b.element('clear-history').disabled,false);
// Same-origin monitor tabs follow storage changes; no metrics or disk deletion.
b.api.clearRequestHistory();a.events.storage({key:Object.keys(store)[0]});
assert.equal(a.api.visibleRequests(newData).length,0);
// Browser privacy settings must not break rendering or misreport persistence.
const c=page({},true);c.api.render(data);c.api.clearRequestHistory();
assert.equal(c.api.visibleRequests(data).length,0);assert.match(c.element('history-status').textContent,/刷新后可能恢复/);
const d=page({[Object.keys(store)[0]]:'NaN'});d.api.render(data);assert.equal(d.api.visibleRequests(data).length,2);
console.log('PASS: clear, stale poll, refresh, new request, tab sync, preserved metrics, blocked storage');
"""
        result = subprocess.run([node, '-e', script, str(M.ASSETS/'app.js'), str(M.ASSETS/'i18n.js')],
                                capture_output=True, text=True, encoding='utf8', timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        html=(M.ASSETS/'index.html').read_text(encoding='utf8')
        self.assertIn('id="clear-history"', html)
        self.assertIn('aria-live="polite"', html)

if __name__=='__main__':unittest.main()
