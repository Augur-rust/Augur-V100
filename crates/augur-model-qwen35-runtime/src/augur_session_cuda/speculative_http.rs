//! HTTP/session owner adapter for explicit B8 numerical policy, not a new ledger.
use super::*;
use crate::checkpoint_target_cuda::dflash_bridge::AugurDflashBridge;
use serde_json::json;
use std::time::Instant;
mod cost_gate;
mod cost_runtime;
#[cfg(test)]
mod tests;
const ADAPTIVE_GATE_MIN_ROUNDS: usize = 8;
const ADAPTIVE_GATE_MIN_ACCEPTED_OVER_CHECKED: f64 = 0.80;

fn adaptive_gate_should_disable(rounds: usize, accepted: usize, checked: usize) -> bool {
    rounds >= ADAPTIVE_GATE_MIN_ROUNDS
        && checked > 0
        && (accepted as f64) / (checked as f64) < ADAPTIVE_GATE_MIN_ACCEPTED_OVER_CHECKED
}

#[derive(Default)]
struct Stats {
    plan_builds:usize,plan_resumes:usize,evicted_for_memory:bool,evicted_for_vision:bool,
    rounds:usize,accepted:usize,checked:usize,setup_s:f64,proposal_s:f64,verify_s:f64,
    billed_rounds:usize,verify_execute_s:f64,verify_decision_s:f64,verify_commit_s:f64,ids_d2h_s:f64,
    gpu_billed_rounds:usize,gpu_gather_s:f64,gpu_gdn_layers_s:f64,gpu_gdn_input_projection_s:f64,
    gpu_gdn_recurrence_s:f64,gpu_gdn_output_ffn_s:f64,
    gpu_gdn_output_projection_s:f64,gpu_gdn_post_attention_s:f64,
    gpu_gdn_ffn_gate_up_s:f64,gpu_gdn_ffn_activation_down_s:f64,gpu_gdn_final_add_s:f64,
    gpu_full_attention_layers_s:f64,
    gpu_target_norm_s:f64,gpu_output_head_s:f64,gpu_top1_s:f64,gpu_event_pairs:usize,
    gpu_gdn_layer_count:usize,gpu_full_layer_count:usize,
    committed_width_count:[usize;9],commit_wall_by_width_s:[f64;9],
    by_position:[[usize;4];7],speculation_disabled:bool,disable_reason:Option<&'static str>,
    gate_evaluations:usize,first_gate:Option<GateSnapshot>,disabled_gate:Option<GateSnapshot>,
}

/// Request-local observations of the existing gate, never a new gate policy.
/// Uses already measured component walls; no GPU synchronization or per-round
/// heap allocation is added. Reset/restore discard these snapshots with Stats.
#[derive(Clone, Copy)]
struct GateSnapshot {
    rounds:usize,accepted:usize,checked:usize,target_consumed:usize,
    setup_s:f64,proposal_s:f64,verify_s:f64,by_position:[[usize;4];7],
}
impl GateSnapshot {
    fn from_stats(stats:&Stats,target_consumed:usize)->Self {
        Self {rounds:stats.rounds,accepted:stats.accepted,checked:stats.checked,target_consumed,
            setup_s:stats.setup_s,proposal_s:stats.proposal_s,verify_s:stats.verify_s,
            by_position:stats.by_position}
    }
    fn detail(&self)->Value {
        let tokens=self.rounds+self.accepted;
        let components=self.setup_s+self.proposal_s+self.verify_s;
        json!({"after_completed_B8_rounds":self.rounds,"target_consumed_tokens":self.target_consumed,
            "proposed_draft_tokens":self.rounds*7,"accepted_draft_tokens":self.accepted,
            "checked_draft_tokens":self.checked,"completed_B8_block_tokens_including_anchors":tokens,
            "accepted_over_proposed":if self.rounds>0{Some(self.accepted as f64/(self.rounds*7) as f64)}else{None},
            "accepted_over_checked":if self.checked>0{Some(self.accepted as f64/self.checked as f64)}else{None},
            "by_position_accepted_rejected_unchecked_censored":self.by_position,
            "request_plan_setup_s":self.setup_s,"proposal_wall_s":self.proposal_s,
            "batch_verify_commit_wall_s":self.verify_s,"measured_component_sum_s":components,
            "component_s_per_completed_B8_block_token":if tokens>0{Some(components/tokens as f64)}else{None},
            "scope":"request cumulative completed B8 rounds at gate evaluation",
            "not_whole_decode_or_HTTP_wall":true,"native_B1_counterfactual_measured":false,
            "current_already_published_anchor_excluded":true})
    }
}
impl Stats {
    fn observe_gate(&mut self,target_consumed:usize)->bool {
        let disable=adaptive_gate_should_disable(self.rounds,self.accepted,self.checked);
        if self.rounds>=ADAPTIVE_GATE_MIN_ROUNDS && self.checked>0 {
            self.gate_evaluations+=1;
            if self.first_gate.is_none() || disable {
                let snapshot=GateSnapshot::from_stats(self,target_consumed);
                if self.first_gate.is_none() {self.first_gate=Some(snapshot);}
                if disable {self.disabled_gate=Some(snapshot);}
            }
        }
        disable
    }
    fn gate_detail(&self)->Value {
        json!({"policy":"legacy_cumulative_checked_acceptance",
            "eligible_evaluations":self.gate_evaluations,
            "first_evaluation":self.first_gate.as_ref().map(GateSnapshot::detail),
            "disable_evaluation":self.disabled_gate.as_ref().map(GateSnapshot::detail),
            "threshold_accepted_over_checked":ADAPTIVE_GATE_MIN_ACCEPTED_OVER_CHECKED,
            "minimum_completed_B8_rounds":ADAPTIVE_GATE_MIN_ROUNDS,
            "decision_policy_unchanged":true})
    }
}


#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub(super) enum CostPolicy {Legacy,Immediate,Deferred,FixedB8}
fn parse_cost_gate(value: Option<&str>, active: bool) -> Result<CostPolicy> {
    match value {
        None if active => Ok(CostPolicy::FixedB8),
        None | Some("legacy") => Ok(CostPolicy::Legacy),
        Some("fixed_b8") if active => Ok(CostPolicy::FixedB8),
        Some("cost_v1") if active => Ok(CostPolicy::Immediate),
        Some("cost_deferred_v2") if active => Ok(CostPolicy::Deferred),
        Some("cost_v1" | "cost_deferred_v2" | "fixed_b8") => Err("cost gate requires active batched8 DFlash".into()),
        _ => Err("AUGUR_DFLASH_GATE must be fixed_b8, legacy, cost_v1, or cost_deferred_v2".into()),
    }
}
pub(super) fn requested_cost_gate(active: bool) -> Result<CostPolicy> {
    match std::env::var("AUGUR_DFLASH_GATE") {
        Ok(value) => parse_cost_gate(Some(&value), active),
        Err(std::env::VarError::NotPresent) => parse_cost_gate(None, active),
        Err(error) => Err(error.into()),
    }
}
fn cost_lane(lane: cost_gate::Lane) -> &'static str {
    match lane { cost_gate::Lane::B1 => "B1", cost_gate::Lane::B8 => "B8" }
}
fn cost_sample(sample: cost_gate::Sample) -> Value {
    json!({"completed_transactions":sample.transactions,"consumed_tokens":sample.consumed_tokens,
        "wall_ns":sample.wall_ns,"wall_s":sample.wall_ns as f64/1e9,
        "ns_per_consumed_token":sample.ns_per_consumed_token()})
}
fn cost_detail(runtime: &cost_runtime::Runtime) -> Value {
    let total=runtime.totals();let tail=runtime.terminal();let fixed=runtime.fixed_b8();
    let decisions:Vec<Value>=runtime.control().map(|c|c.decisions().iter().flatten().map(|d|json!({
        "probe_start_consumed":d.probe_start_consumed,"decision_consumed":d.decision_consumed,
        "B1":cost_sample(d.b1),"B8":cost_sample(d.b8),
        "incumbent":cost_lane(d.incumbent),"selected":cost_lane(d.selected)})).collect()).unwrap_or_default();
    json!({"policy":if fixed{"fixed_b8_v1"}else if runtime.deferred_start(){"deferred_decode_cost_v2"}else{"measured_decode_cost_v1"},"decision_policy_unchanged":false,
        "initial_B8_before_cost_sampling":cost_sample(runtime.warmup()),
        "first_cost_probe_trigger":runtime.first_probe_reason(),
        "low_acceptance_trigger":runtime.low_acceptance_trigger().map(|(rounds,accepted,checked)|json!({"rounds":rounds,"accepted":accepted,"checked":checked})),
        "B1_sample_steps":if fixed{0}else{cost_gate::B1_SAMPLE_STEPS},"B8_sample_rounds":if fixed{0}else{cost_gate::B8_SAMPLE_ROUNDS},
        "recheck_after_consumed_tokens":if fixed{0}else{cost_gate::RECHECK_AFTER_CONSUMED},
        "maximum_decisions":if fixed{0}else{cost_gate::MAX_DECISIONS},"hysteresis_percent":if fixed{0}else{cost_gate::HYSTERESIS_PERCENT},
        "decisions":decisions,"all_completed_B1":cost_sample(total[0]),"all_completed_B8":cost_sample(total[1]),
        "next_lane_if_more_tokens":if runtime.reason().is_some(){Some("B1")}else{runtime.control().map(|c|cost_lane(c.next_lane())).or_else(||if fixed || runtime.deferred_start(){Some("B8")}else{None})},
        "start_consumed_tokens":runtime.start_cursor(),"terminal_consumed_tokens":runtime.terminal_cursor(),
        "forced_native_reason":runtime.reason(),"request_drained":runtime.finished(),
        "interval_open":runtime.has_open_interval(),
        "terminal_incomplete_intervals":{"lane":runtime.terminal_lane().map(cost_lane),"transactions":tail.0,"consumed_tokens":tail.1,
            "wall_ns":tail.2,"selection_without_consume_wall_ns":tail.3,"used_for_decisions":false},
        "clock":"after_synchronized_selection_before_publication_to_next_synchronized_selection",
        "includes_publication_setup_and_next_selection":true,"extra_GPU_synchronization":false,
        "publication_scope":"native Session emit callback; not HTTP network or browser rendering",
        "prefill_excluded":true,"same_prefix_B1_counterfactual_measured":false,
        "all_sample_output_is_published_and_charged":true,"not_whole_decode_or_HTTP_wall":true})
}

pub(crate) struct Target {
    // Drop all shared leases/plans before their target owner.
    draft:AugurDflashBridge,target:CheckpointCudaTarget,active:bool,park:bool,stats:Stats,
    cost:Option<cost_runtime::Runtime>,
    selector:session::selector::Phase,
    #[cfg(windows)] checkpoint:Option<super::checkpoint::Manager>,
}
impl Target {
    pub(super) fn new(target:CheckpointCudaTarget,draft:AugurDflashBridge,active:bool,park:bool,cost_policy:CostPolicy)->Self {Self{draft,target,active,park,stats:Stats::default(),selector:session::selector::Phase::default(),cost:match cost_policy {CostPolicy::Legacy=>None,CostPolicy::FixedB8=>Some(cost_runtime::Runtime::fixed()),CostPolicy::Immediate=>Some(cost_runtime::Runtime::default()),CostPolicy::Deferred=>Some(cost_runtime::Runtime::deferred())},#[cfg(windows)] checkpoint:None}}
    fn before_prefill(&mut self,rows:usize,images:bool)->Result<()> {
        self.stats=Stats::default();self.selector=Default::default();
        if let Some(cost)=&mut self.cost {cost.reset();}
        if self.park && !images {
            self.draft.park_batched_verifier()?;
            if self.draft.batched_cached_owned_bytes()>0 {
                let required=crate::checkpoint_target_cuda::augur_prefill::default_required_free_bytes_mode(rows,self.target.attention_mode())?;
                #[cfg(feature="qwen35-augur-vision")]
                self.target.reclaim_vision_for_budget(required,"prefill_preserve_b8_plan")?;
                let free=augur_backend::cuda_memory_info(0)?.free_bytes;
                if required>free {self.draft.release_batched_verifier()?;self.stats.evicted_for_memory=true;}
            }
        } else {
            self.stats.evicted_for_vision=images && self.draft.batched_cached_owned_bytes()>0;
            if images { self.stats.speculation_disabled=true; self.stats.disable_reason=Some("vision");
                if let Some(cost)=&mut self.cost {cost.force_native("vision");} }
            self.draft.release_batched_verifier()?;
        }
        Ok(())
    }
}
impl session::SessionTarget for Target {
    fn validate_selector_policy(&self,policy:Option<session::selector::Policy>,suffix:&[u32],images:bool)->Result<()> {
        let available=self.active && self.draft.http_english_gate_available() && self.cost.as_ref().is_some_and(|c|c.fixed_b8());
        session::selector::validate_request(policy,available,self.draft.http_global_gate_available(),suffix,images)
    }
    fn begin_decode(&mut self,policy:Option<session::selector::Policy>)->Result<()> {
        self.target.begin_sampling_trace();
        self.selector.begin(policy);self.draft.select_http_english_gate(self.selector.gate_active())
    }
    #[cfg(windows)] fn checkpoint_bindings(&self)->Option<session::checkpoint::Bindings>{self.checkpoint.as_ref().and_then(|m|m.bindings())}
    #[cfg(windows)] fn save_checkpoint(&mut self,d:&session::checkpoint::Descriptor,id:&str)->Result<Value>{
        self.draft.drain()?;self.checkpoint.as_mut().ok_or("checkpoint disabled")?.save(&mut self.target,d,id)
    }
    #[cfg(windows)] fn restore_checkpoint(&mut self,r:&session::checkpoint::RestoreRequest)->Result<(session::checkpoint::Descriptor,Value)>{
        self.draft.release_batched_verifier()?;self.target.reset_empty()?;self.stats=Stats::default();self.selector=Default::default();
        if let Some(cost)=&mut self.cost {cost.reset();}
        self.checkpoint.as_mut().ok_or("checkpoint disabled")?.restore(&mut self.target,r)
    }
    fn profile(&self)->&'static str {if self.active{DFLASH_B8_PROFILE}else{PROFILE}}
    fn dflash_enabled(&self)->bool {self.active}
    fn consumed_tokens(&self)->usize {self.target.consumed_tokens()}
    fn supports_images(&self)->bool {self.target.supports_images()}
    fn consume_sequence(&mut self,ids:&[u32])->Result<()> {self.target.consume_sequence(ids)}
    fn prefill(&mut self,ids:&[u32],cancel:&SessionCancellationToken)->Result<Option<Value>> {
        self.before_prefill(ids.len(),false)?;
        #[cfg(windows)] if let Some(m)=&mut self.checkpoint {if let Some(v)=m.prefill(&mut self.target,ids,cancel)? {return Ok(Some(v));}}
        session::SessionTarget::prefill(&mut self.target,ids,cancel)
    }
    fn prefill_image_history(&mut self,ids:&[u32],cancel:&SessionCancellationToken)->Result<Option<Value>> {
        // Retained image KV still uses native decoding. Text-only suffixes must
        // not re-enable B8 or attempt a Q8 gate absent from embedded FP32 weights.
        self.before_prefill(ids.len(),true)?;
        session::SessionTarget::prefill(&mut self.target,ids,cancel)
    }
    fn prefill_images(&mut self,ids:&[u32],images:&[augur_model_qwen35::Qwen35PreparedF16Image],cancel:&SessionCancellationToken)->Result<Option<Value>> {
        self.before_prefill(ids.len(),true)?;session::SessionTarget::prefill_images(&mut self.target,ids,images,cancel)
    }
    fn greedy_next_token(&mut self)->Result<u32> {
        let id=self.target.select_next_token()?;
        self.selector.selected(id);
        if let Some(cost)=&mut self.cost {cost.selected(self.target.consumed_tokens(),Instant::now())?;}
        Ok(id)
    }
    fn consume_selected(&mut self)->Result<()> {
        self.target.consume_selected_augur()?;
        if let Some(cost)=&mut self.cost {cost.enqueued(cost_gate::Lane::B1,self.target.consumed_tokens())?;}
        Ok(())
    }
    fn drain(&mut self)->Result<()> {
        self.draft.drain()?;self.target.drain_session_work()?;
        if let Some(cost)=&mut self.cost {cost.drained(self.target.consumed_tokens(),Instant::now())?;}
        Ok(())
    }
    fn reset_empty(&mut self)->Result<()> {if self.park{self.draft.park_batched_verifier()?;}else{self.draft.release_batched_verifier()?;}self.target.reset_empty()?;self.stats=Stats::default();self.selector=Default::default();if let Some(cost)=&mut self.cost {cost.reset();}Ok(())}
    fn speculative_step(&mut self,anchor:u32,remaining:usize,eos:&[u32],cancel:&SessionCancellationToken)->Result<Option<SpeculativeStep>> {
        if !self.active {return Ok(None);}
        if self.stats.speculation_disabled || self.target.consumed_tokens()+8>context_capacity()? {
            if let Some(cost)=&mut self.cost {cost.force_native(self.stats.disable_reason.unwrap_or("capacity"));}
            return Ok(None);
        }
        if let Some(cost)=&mut self.cost {
            cost.consider_legacy_fallback(self.stats.rounds,self.stats.accepted,self.stats.checked)?;
            if cost.next_lane()?==cost_gate::Lane::B1 {return Ok(None);}
        } else if self.stats.observe_gate(self.target.consumed_tokens()) {
            self.stats.speculation_disabled=true;
            self.stats.disable_reason=Some("low_acceptance");
            return Ok(None);
        }
        cancel.check("before Augur selected-token consume")?;
        let setup=Instant::now();let disposition=self.draft.prepare_http_batched_verifier(&self.target)?;self.stats.setup_s+=setup.elapsed().as_secs_f64();
        self.stats.plan_builds+=usize::from(disposition=="new");self.stats.plan_resumes+=usize::from(disposition=="resumed");
        self.draft.select_http_english_gate(self.selector.gate_active())?;
        let started=Instant::now();let proposal=self.draft.propose_resident(&mut self.target)?;
        self.stats.proposal_s+=started.elapsed().as_secs_f64();
        if proposal.path()[0]!=anchor {self.draft.discard_resident(proposal)?;return Err("HTTP draft anchor drift".into());}
        if cancel.is_requested() {self.draft.discard_resident(proposal)?;return match cancel.check("before Augur selected-token consume") {Err(e)=>Err(e.into()),Ok(())=>Err("cancellation unexpectedly cleared".into())};}
        // One bounded complete transaction. Cancellation never publishes a partially
        // verified block; the outer Session can durably discard a completed block.
        // The selector switches gates inside B8; the unchanged target verifier
        // commits only its strict accepted prefix, including an accepted close.
        let r=self.draft.verify_resident_batched(&mut self.target,proposal,remaining,eos)?;
        self.stats.verify_s+=r.batch_verify_commit_s;self.stats.rounds+=1;
        if let (Some(execute),Some(decision),Some(commit))=(r.verify_execute_s,r.verify_decision_s,r.verify_commit_s) {
            let consumed=r.end_consumed-r.start_consumed;
            self.stats.billed_rounds+=1;self.stats.verify_execute_s+=execute;
            self.stats.verify_decision_s+=decision;self.stats.verify_commit_s+=commit;
            if let Some(ids_d2h)=r.ids_d2h_s {self.stats.ids_d2h_s+=ids_d2h;}
            self.stats.committed_width_count[consumed]+=1;self.stats.commit_wall_by_width_s[consumed]+=commit;
        }
        if let Some(gpu)=r.verify_gpu_component_bill {
            self.stats.gpu_billed_rounds+=1;self.stats.gpu_gather_s+=gpu.gather_s;
            self.stats.gpu_gdn_layers_s+=gpu.gdn_layers_s;self.stats.gpu_gdn_input_projection_s+=gpu.gdn_input_projection_s;
            self.stats.gpu_gdn_recurrence_s+=gpu.gdn_recurrence_s;self.stats.gpu_gdn_output_ffn_s+=gpu.gdn_output_ffn_s;
            self.stats.gpu_gdn_output_projection_s+=gpu.gdn_output_projection_s;
            self.stats.gpu_gdn_post_attention_s+=gpu.gdn_post_attention_s;
            self.stats.gpu_gdn_ffn_gate_up_s+=gpu.gdn_ffn_gate_up_s;
            self.stats.gpu_gdn_ffn_activation_down_s+=gpu.gdn_ffn_activation_down_s;
            self.stats.gpu_gdn_final_add_s+=gpu.gdn_final_add_s;
            self.stats.gpu_full_attention_layers_s+=gpu.full_attention_layers_s;
            self.stats.gpu_target_norm_s+=gpu.target_norm_s;self.stats.gpu_output_head_s+=gpu.output_head_s;
            self.stats.gpu_top1_s+=gpu.top1_s;self.stats.gpu_event_pairs=gpu.event_pairs;
            self.stats.gpu_gdn_layer_count=gpu.gdn_layer_count;self.stats.gpu_full_layer_count=gpu.full_layer_count;
        }
        let accepted=r.output_tokens.len()-1;let rejected=usize::from(r.first_rejection.is_some());
        self.stats.accepted+=accepted;self.stats.checked+=accepted+rejected;
        self.selector.verified_round(&r.output_tokens,rejected!=0);
        let reject=r.first_rejection.as_ref().map(|r|r["offset"].as_u64().unwrap() as usize);
        for pos in 1..8 {let bucket=if pos<r.output_tokens.len(){0}else if reject==Some(pos){1}else if reject.is_some(){2}else{3};self.stats.by_position[pos-1][bucket]+=1;}
        if let Some(cost)=&mut self.cost {cost.enqueued(cost_gate::Lane::B8,r.end_consumed)?;}
        Ok(Some(SpeculativeStep{tokens:r.output_tokens,consumed:r.end_consumed-r.start_consumed}))
    }
    fn decode_detail(&self)->Option<Value> {
        let mut d=session::SessionTarget::decode_detail(&self.target).unwrap_or_else(||json!({}));
        if self.active {let s=&self.stats;d["dflash"]=json!({"profile":DFLASH_B8_PROFILE,"active":true,"block_width":8,
            "target_projection":"packed_b8","draft_projection":"llama","draft_head":"llama_b8","context":"retained",
            "draft_topk":self.draft.head_topk_policy(),"shared_head_kernel":{"draft":self.draft.shared_head_kernel(),"verify":self.draft.verify_head_kernel()},"draft_block_projection_kernel":self.draft.draft_block_projection_kernel(),"gdn_input_kernel":self.draft.gdn_input_kernel(),
            "english_selector": {"available":self.draft.http_english_gate_available(),"requested":self.selector.eligible,
                "policy":self.selector.policy_name(),
                "gate_sha256":self.selector.eligible.then_some(session::selector::GATE_SHA256),
                "calibrated_rounds":self.selector.calibrated_rounds,"q8_rounds":self.selector.q8_rounds,
                "calibrated_accepted_drafts":self.selector.calibrated_accepted,
                "calibrated_checked_drafts":self.selector.calibrated_checked,
                "within_block_phase_transition_rounds":self.selector.boundary_rounds,
                "phase_transition":if self.selector.global {"global_no_phase_switch"} else {"same_selector_walk_sticky_after_close"},
                "calibrated_rounds_scope":if self.selector.global {"all_text_output"} else {"rounds_starting_in_thinking"},
                "q8_accepted_drafts":self.selector.q8_accepted,"q8_checked_drafts":self.selector.q8_checked,"close_think_seen":self.selector.close_seen,
                "after_close_gate":if self.selector.global {"gate045_f32"} else {"q8_0"},"target_verification":self.target.sampling_detail()},
            "rounds":s.rounds,"proposed_draft_tokens":s.rounds*7,"accepted_draft_tokens":s.accepted,"checked_draft_tokens":s.checked,
            "by_position_accepted_rejected_unchecked_censored":s.by_position,
            "accepted_over_proposed":if s.rounds>0{Some(s.accepted as f64/(s.rounds*7) as f64)}else{None},
            "accepted_over_checked":if s.checked>0{Some(s.accepted as f64/s.checked as f64)}else{None},
            "plan_cache":if self.park{"park"}else{"recreate"},"plan_builds":s.plan_builds,"plan_resumes":s.plan_resumes,
            "plan_evicted_for_memory":s.evicted_for_memory,"plan_evicted_for_vision":s.evicted_for_vision,
            "cached_plan_owned_bytes":self.draft.batched_cached_owned_bytes(),
            "B8_GDN_graph_nodes":self.draft.batched_gdn_graph_nodes(),
            "request_plan_setup_s":s.setup_s,"proposal_wall_s":s.proposal_s,"batch_verify_commit_wall_s":s.verify_s,
            "proposal_component_bill":crate::checkpoint_target_cuda::dflash_bridge::proposal_bill_summary(),
            "verify_component_bill":if s.billed_rounds>0{Some(json!({"billed_rounds":s.billed_rounds,
                "execute_through_ids_D2H_wall_s":s.verify_execute_s,"cpu_decision_wall_s":s.verify_decision_s,
                "commit_through_existing_stream_sync_wall_s":s.verify_commit_s,
                "ids_d2h_wall_s":s.ids_d2h_s,
                "committed_width_count":s.committed_width_count,"commit_wall_by_width_s":s.commit_wall_by_width_s,
                "extra_GPU_synchronization":false,"diagnostic_timing":true,"HTTP_speedup_claim":false}))}else{None},
            "verify_gpu_component_bill":if s.gpu_billed_rounds>0{Some(json!({
                "billed_rounds":s.gpu_billed_rounds,"gather_s":s.gpu_gather_s,
                "gdn_layers_s":s.gpu_gdn_layers_s,"gdn_input_projection_s":s.gpu_gdn_input_projection_s,
                "gdn_recurrence_s":s.gpu_gdn_recurrence_s,"gdn_output_ffn_s":s.gpu_gdn_output_ffn_s,
                "gdn_output_projection_s":s.gpu_gdn_output_projection_s,
                "gdn_post_attention_s":s.gpu_gdn_post_attention_s,
                "gdn_ffn_gate_up_s":s.gpu_gdn_ffn_gate_up_s,
                "gdn_ffn_activation_down_s":s.gpu_gdn_ffn_activation_down_s,
                "gdn_final_add_s":s.gpu_gdn_final_add_s,
                "full_attention_layers_s":s.gpu_full_attention_layers_s,
                "target_norm_s":s.gpu_target_norm_s,"output_head_s":s.gpu_output_head_s,"top1_s":s.gpu_top1_s,
                "event_pairs":s.gpu_event_pairs,"gdn_stage_event_pairs":s.gpu_gdn_layer_count*8,
                "gdn_layer_count":s.gpu_gdn_layer_count,
                "full_layer_count":s.gpu_full_layer_count,"extra_GPU_synchronization":false,
                "diagnostic_timing":true,"HTTP_speedup_claim":false}))}else{None},
            "anchor_published_before_draft":true,"adaptive_gate_min_rounds":if self.cost.is_none(){Some(ADAPTIVE_GATE_MIN_ROUNDS)}else{None},
            "adaptive_gate_threshold_accepted_over_checked":if self.cost.is_none(){Some(ADAPTIVE_GATE_MIN_ACCEPTED_OVER_CHECKED)}else{None},
            "speculation_disabled":s.speculation_disabled,"speculation_disable_reason":s.disable_reason,
            "adaptive_gate_decisions":self.cost.as_ref().map(cost_detail).unwrap_or_else(||s.gate_detail()),
            "B1_numerical_equivalence_qualified":false});}Some(d)
    }
}

#[cfg(windows)]
impl Target {
    pub(crate) fn configure_checkpoint(&mut self,evidence:&std::path::Path,manifest:&str)->Result<Value>{
        if self.checkpoint.is_some() || self.target.consumed_tokens()!=0{return Err("checkpoint configuration is startup-only".into());}
        self.checkpoint=super::checkpoint::Manager::configure(evidence,manifest,self.target.attention_mode().label())?;
        Ok(self.checkpoint.as_ref().map(|m|m.ready()).unwrap_or_else(||json!({"save_restore_enabled":false,"input_prefix_tokens":null})))
    }
}
