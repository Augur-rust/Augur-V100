//! Isolated, single-owner image/text lane. Full target and Vision load once;
//! request-private KV/GDN retires before publication of the completion terminal.
//! Target-only first: DFlash and long-context admission remain explicitly off.
use super::*;
use crate::{Qwen35VisionCudaEncoder, Qwen35VisionCudaOutput};
use augur_backend::{CudaDeviceAllocationSnapshot, CudaQwen35VocabTop1ArgsV1};
use augur_model_qwen35::{
    Qwen35ImageMropePlan, Qwen35ImageMropeReceipt, Qwen35ImagePromptLayout, Qwen35PreparedF16Image,
};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::path::Path;

mod prefill;
mod profile;
mod session;

const MAX_CONTEXT: usize = 4096;
const MAX_IMAGES: usize = 4;
const CACHE_BYTES: usize = 20 * 1024 * 1024;
type Result<T> = std::result::Result<T, Qwen35MixedPrefixCudaError>;

pub struct Qwen35VisionTextCudaOwner {
    vision: Qwen35VisionCudaEncoder,
    target: target_weights::TargetWeights,
    model: Qwen35Config,
    tokenizer: Tokenizer,
    capacity: usize,
    eos: Vec<u32>,
    cached: VecDeque<Qwen35VisionCudaOutput>,
    poisoned: bool,
    completed_requests: u64,
    prefill_rows: usize,
    prefill_profile: bool,
    prefill_shared_attention: bool,
}
#[derive(Debug, Serialize)]
pub struct Qwen35VisionTextToken {
    pub index: usize,
    pub token_id: u32,
    pub text: String,
}
#[derive(Debug, Serialize)]
pub struct Qwen35VisionTextReport {
    pub schema: &'static str,
    pub request_id: String,
    pub prompt_tokens: usize,
    pub consumed_input_tokens: usize,
    pub image_input_rows_submitted: usize,
    pub image_encodings: Vec<Qwen35VisionCudaOutput>,
    pub image_embedding_cache_hits: usize,
    pub mrope: Qwen35ImageMropeReceipt,
    pub generated_ids: Vec<u32>,
    pub generated_text: String,
    pub stop_reason: &'static str,
    pub request_wall_s: f64,
    pub first_visible_text_s: Option<f64>,
    pub last_token_receipt_s: Option<f64>,
    pub language_prefill: Value,
    pub complete_model_sha256: String,
    pub allocation_before: Value,
    pub allocation_after: Value,
    pub request_allocations_balanced: bool,
    pub model_loaded_once: bool,
    pub request_private_kv_gdn: bool,
    pub dflash_state: &'static str,
    pub vision_forward_executed: bool,
    pub image_embeddings_consumed: bool,
    pub product_ready: bool,
    pub qualified: bool,
    pub visible_tps: bool,
}
struct Drain<'a> {
    stream: &'a OwnedCudaStream,
    pending: bool,
}
impl Drop for Drain<'_> {
    fn drop(&mut self) {
        if self.pending {
            let _ = self.stream.synchronize();
        }
    }
}
fn allocations(s: CudaDeviceAllocationSnapshot) -> Value {
    json!({"allocations":s.successful_allocations,"frees":s.successful_frees,
        "live_allocations":s.live_allocations,"live_bytes":s.live_bytes,"tracked_process_peak_bytes":s.peak_live_bytes})
}
fn balanced(a: CudaDeviceAllocationSnapshot, b: CudaDeviceAllocationSnapshot) -> bool {
    a.live_bytes == b.live_bytes
        && a.live_allocations == b.live_allocations
        && b.successful_allocations
            .checked_sub(a.successful_allocations)
            == b.successful_frees.checked_sub(a.successful_frees)
}
fn check_cancel(cancelled: &impl Fn() -> bool) -> Result<()> {
    if cancelled() {
        Err(invalid("Vision text request cancelled"))
    } else {
        Ok(())
    }
}
impl Qwen35VisionTextCudaOwner {
    pub fn open(
        config: &Qwen35AutoregressiveCudaConfig,
        vision_source: &Path,
        expected_vision_sha256: Option<&str>,
        generation_config: &Path,
    ) -> Result<Self> {
        if config.device_ordinal != 0
            || config.capacity == 0
            || config.capacity > MAX_CONTEXT
            || !augur_backend::cuda_q8_0::QWEN35_TARGET_SPLIT_ROPE
        {
            return Err(invalid(
                "Vision text v1 requires SM70 device0, split-half RoPE and context <=4096",
            ));
        }
        // Freeze the arithmetic/schedule policy once, before loading any CUDA
        // owner. A malformed switch must not silently enter another identity.
        let prefill_rows =
            prefill::parse_rows(std::env::var_os("AUGUR_QWEN35_VISION_PREFILL_ROWS").as_deref())?;
        let prefill_profile =
            profile::parse(std::env::var_os("AUGUR_QWEN35_VISION_PREFILL_PROFILE").as_deref())?;
        let prefill_shared_attention = prefill::parse_shared_attention(
            std::env::var_os("AUGUR_QWEN35_VISION_PREFILL_ATTENTION").as_deref(),
        )?;
        if prefill_profile && prefill_rows == 1 {
            return Err(invalid(
                "Vision operator profile requires a batched prefill policy",
            ));
        }
        let bytes = read_stable_regular_file(&config.config_path, "Vision text model config")?;
        let model = Qwen35Config::parse(&bytes)?;
        model.validate_exact_profile()?;
        let tokenizer = Tokenizer::from_bytes(&read_stable_regular_file(
            &config.tokenizer_path,
            "Vision text tokenizer",
        )?)
        .map_err(|e| invalid(e.to_string()))?;
        let generation: Value = serde_json::from_slice(&read_stable_regular_file(
            generation_config,
            "Vision text generation config",
        )?)
        .map_err(|e| invalid(e.to_string()))?;
        let eos = match &generation["eos_token_id"] {
            Value::Array(v) => v
                .iter()
                .map(|x| {
                    x.as_u64()
                        .filter(|&x| x < model.vocab_size as u64)
                        .map(|x| x as u32)
                        .ok_or_else(|| invalid("Vision invalid EOS"))
                })
                .collect::<Result<Vec<_>>>()?,
            Value::Number(n) => vec![n
                .as_u64()
                .filter(|&x| x < model.vocab_size as u64)
                .map(|x| x as u32)
                .ok_or_else(|| invalid("Vision invalid EOS"))?],
            _ => return Err(invalid("Vision EOS contract missing")),
        };
        if eos.is_empty() || eos.len() > 8 || !eos.contains(&model.eos_token_id) {
            return Err(invalid("Vision EOS contract mismatch"));
        }
        let vision =
            Qwen35VisionCudaEncoder::load(&bytes, vision_source, expected_vision_sha256, 0)
                .map_err(|e| invalid(e.to_string()))?;
        let target = target_weights::TargetWeights::open(config)?;
        Ok(Self {
            vision,
            target,
            model,
            tokenizer,
            capacity: config.capacity,
            eos,
            cached: VecDeque::new(),
            poisoned: false,
            completed_requests: 0,
            prefill_rows,
            prefill_profile,
            prefill_shared_attention,
        })
    }
    pub fn capabilities(&self) -> Value {
        json!({"schema":"qwen35-vision-native-capabilities-v1","model_loaded_once":true,
            "vision_forward":!self.poisoned,"image_embeddings_consumed":!self.poisoned,
            "max_context_tokens":self.capacity,"max_output_tokens":256,"max_images":MAX_IMAGES,
            "request_private_kv_gdn":true,"persistent_target_and_vision_weights":true,
            "embedding_cache_bytes":CACHE_BYTES,"dflash_enabled":false,
            "language_prefill_rows":self.prefill_rows,
            "language_prefill_policy":prefill::policy(self.prefill_rows),
            "instrumented_prefill_profile":self.prefill_profile,
            "language_prefill_attention":prefill::attention_policy(self.prefill_shared_attention),
            "supported_modalities":["text","image"],"product_ready":false,"qualified":false,"visible_tps":false})
    }
    pub fn respond(
        &mut self,
        request_id: &str,
        prompt_text: &str,
        images: &[Qwen35PreparedF16Image],
        max_new_tokens: usize,
        cancelled: impl Fn() -> bool,
        sink: impl FnMut(Qwen35VisionTextToken) -> Result<()>,
    ) -> Result<Qwen35VisionTextReport> {
        let before = cuda_device_allocation_snapshot();
        let result = self.respond_inner(
            request_id,
            prompt_text,
            images,
            max_new_tokens,
            cancelled,
            sink,
        );
        let after = cuda_device_allocation_snapshot();
        if !balanced(before, after) {
            self.poisoned = true;
            return Err(invalid("Vision request CUDA allocation did not retire"));
        }
        let mut report = result?;
        report.allocation_before = allocations(before);
        report.allocation_after = allocations(after);
        report.request_allocations_balanced = true;
        self.completed_requests += 1;
        Ok(report)
    }
    fn respond_inner(
        &mut self,
        request_id: &str,
        prompt_text: &str,
        images: &[Qwen35PreparedF16Image],
        max_new_tokens: usize,
        cancelled: impl Fn() -> bool,
        mut sink: impl FnMut(Qwen35VisionTextToken) -> Result<()>,
    ) -> Result<Qwen35VisionTextReport> {
        if self.poisoned {
            return Err(invalid("Vision owner is poisoned; retire process"));
        }
        if !(8..=64).contains(&request_id.len())
            || !request_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
            || !(1..=256).contains(&max_new_tokens)
            || images.len() > MAX_IMAGES
            || prompt_text.len() > 1024 * 1024
        {
            return Err(invalid("Vision request identity/budget"));
        }
        check_cancel(&cancelled)?;
        let started = Instant::now();
        let ids = self
            .tokenizer
            .encode(prompt_text, false)
            .map_err(|e| invalid(e.to_string()))?
            .get_ids()
            .to_vec();
        let kinds = ids
            .iter()
            .map(|&id| u8::from(id == self.model.image_token_id))
            .collect::<Vec<_>>();
        let grids = images.iter().map(|image| image.grid).collect::<Vec<_>>();
        let layout = Qwen35ImagePromptLayout::new(&ids, &kinds, &grids)?;
        let visual = layout.image_token_indices().len();
        let plan_end = ids
            .len()
            .checked_add(max_new_tokens)
            .ok_or_else(|| invalid("Vision context overflow"))?;
        if plan_end > self.capacity || visual > 1024 {
            return Err(invalid("Vision context/image token budget exceeded"));
        }
        let mrope = Qwen35ImageMropePlan::new(&layout, plan_end, 4 * 1024 * 1024)?;
        let mut encoded = Vec::with_capacity(images.len());
        let mut cache_hits = 0;
        for image in images {
            check_cancel(&cancelled)?;
            image.validate_payload_identity()?;
            let value = if let Some(index) = self
                .cached
                .iter()
                .position(|e| e.prepared_payload_sha256 == image.payload_sha256)
            {
                let cached = self
                    .cached
                    .remove(index)
                    .ok_or_else(|| invalid("Vision cache index"))?;
                self.cached.push_back(cached.clone());
                cache_hits += 1;
                let mut hit = cached;
                hit.vision_forward_executed = false;
                hit.encoder_wall_ms = 0.0;
                hit
            } else {
                let value = self
                    .vision
                    .encode(image, &cancelled)
                    .map_err(|e| invalid(e.to_string()))?;
                check_cancel(&cancelled)?;
                while self
                    .cached
                    .iter()
                    .map(|x| x.embeddings_f32.len() * 4)
                    .sum::<usize>()
                    + value.embeddings_f32.len() * 4
                    > CACHE_BYTES
                {
                    self.cached
                        .pop_front()
                        .ok_or_else(|| invalid("Vision embedding exceeds RAM cache budget"))?;
                }
                self.cached.push_back(value.clone());
                value
            };
            encoded.push(value);
        }
        let embeddings = encoded
            .iter()
            .flat_map(|value| value.embeddings_f32.iter().copied())
            .collect::<Vec<_>>();
        let mut hidden = Vec::with_capacity(ids.len() * self.model.hidden_size);
        for &id in &ids {
            hidden.extend(self.target.embedding_row(id as usize)?);
        }
        layout.scatter_image_embeddings(&mut hidden, &embeddings, self.model.hidden_size)?;
        self.generate(
            request_id,
            &ids,
            &kinds,
            &hidden,
            &mrope,
            max_new_tokens,
            &cancelled,
            &mut sink,
            started,
            encoded,
            cache_hits,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn generate(
        &mut self,
        request_id: &str,
        ids: &[u32],
        kinds: &[u8],
        hidden: &[f32],
        mrope: &Qwen35ImageMropePlan,
        max_new_tokens: usize,
        cancelled: &impl Fn() -> bool,
        sink: &mut impl FnMut(Qwen35VisionTextToken) -> Result<()>,
        started: Instant,
        encoded: Vec<Qwen35VisionCudaOutput>,
        cache_hits: usize,
    ) -> Result<Qwen35VisionTextReport> {
        let stream = OwnedCudaStream::create(0)?;
        let tables = Arc::new(FullRopeTables::new_image(&stream, mrope)?);
        let mut plans = Vec::<DevicePlan>::with_capacity(64);
        let first = &hidden[..self.model.hidden_size];
        for layer in &self.target.layers {
            check_cancel(cancelled)?;
            let source = plans
                .last()
                .map(DevicePlan::final_hidden_f32_view)
                .transpose()?;
            plans.push(match layer {
                target_weights::LayerWeights::Linear(w) => {
                    DevicePlan::Linear(bind_linear_plan(&stream, w, first, None, true, source, None)?)
                }
                target_weights::LayerWeights::Full(w) => {
                    DevicePlan::Full(bind_full_plan_with_position_schedule(
                        &stream,
                        w,
                        &self.model,
                        self.capacity,
                        mrope.capacity(),
                        source,
                        Some(tables.clone()),
                        Some(mrope),
                        None,
                    )?)
                }
            });
        }
        if plans.len() != 64 {
            return Err(invalid("Vision target must execute all 64 layers"));
        }
        let head = bind_device_output_head_plan(
            &stream,
            &self.target.head,
            plans
                .last()
                .ok_or_else(|| invalid("Vision plans missing"))?
                .final_hidden_view()?,
            self.model.rms_norm_eps as f32,
        )?;
        let token = OwnedCudaBuffer::allocate(0, 4)?;
        let selector = CudaQwen35VocabTop1ArgsV1::bind(
            &stream,
            plan_f32_view(&head.logits, self.model.vocab_size)?,
            token.view(0, 4)?,
            1,
        )?;
        let start_event = OwnedCudaEvent::create(0)?;
        let stop_event = OwnedCudaEvent::create(0)?;
        let terminal = OwnedCudaEvent::create(0)?;
        let prefill_started = Instant::now();
        let resident_before_prefill = cuda_device_allocation_snapshot();
        // This owner outlives the drain guard on all error paths. In particular,
        // cancellation midway through a layer cannot free in-flight workspace.
        let mut mixed_prefill = prefill::PreparedMixedRows::prepare(
            self.prefill_rows,
            self.prefill_profile,
            self.prefill_shared_attention,
            &stream,
            &plans,
            &self.model,
            mrope,
            ids.len(),
            hidden,
        )?;
        let prepare_ms = prefill_started.elapsed().as_secs_f64() * 1000.0;
        let resident_with_prefill = cuda_device_allocation_snapshot();
        // Every mutable owner is declared above: this guard always drains first.
        let mut drain = Drain {
            stream: &stream,
            pending: true,
        };
        let result = (|| -> Result<Qwen35VisionTextReport> {
            reset_plans(&mut plans, &stream)?;
            let prefix = if mixed_prefill.is_some() {
                ids.len() - 1
            } else {
                0
            };
            let (compute_ms, cuda_ms, chunks, operator_profile) =
                if let Some(prefill) = &mut mixed_prefill {
                    prefill.submit(
                        &stream,
                        &start_event,
                        &stop_event,
                        &terminal,
                        &hidden[..prefix * self.model.hidden_size],
                        cancelled,
                    )?
                } else {
                    (0.0, 0.0, 0, None)
                };
            // Every chunk has an owner+epoch-bound completion. Release the
            // prefill-only weight/activation scratch before first-token decode.
            drop(mixed_prefill.take());
            let resident_after_prefill = cuda_device_allocation_snapshot();
            if !balanced(resident_before_prefill, resident_after_prefill) {
                return Err(invalid("Vision mixed prefill workspace did not retire"));
            }
            let mut image_rows = kinds[..prefix].iter().filter(|&&kind| kind == 1).count();
            for position in prefix..ids.len() {
                check_cancel(cancelled)?;
                let row = &hidden
                    [position * self.model.hidden_size..(position + 1) * self.model.hidden_size];
                submit_autoregressive_token(
                    &plans,
                    if position + 1 == ids.len() {
                        Some(&head)
                    } else {
                        None
                    },
                    row,
                    &stream,
                    position,
                    &start_event,
                    &stop_event,
                    &terminal,
                )?;
                image_rows += usize::from(kinds[position] == 1);
            }
            let language_prefill = json!({
                "schema":"qwen35-vision-mixed-row-prefill-v1",
                "policy":prefill::policy(self.prefill_rows),
                "requested_chunk_rows":self.prefill_rows,"batched_prefix_rows":prefix,
                "scalar_prompt_rows":ids.len()-prefix,"chunk_count":chunks,
                "prepare_wall_ms":prepare_ms,"chunk_compute_wall_ms":compute_ms,
                "chunk_cuda_event_ms":cuda_ms,
                "operator_profile":operator_profile,
                "instrumented_only":self.prefill_profile,
                "attention_path":prefill::attention_policy(self.prefill_shared_attention),
                "prompt_prefill_wall_ms":prefill_started.elapsed().as_secs_f64()*1000.0,
                "workspace_live_bytes":resident_with_prefill.live_bytes
                    .saturating_sub(resident_before_prefill.live_bytes),
                "workspace_released":true,"workspace_allocations_balanced":true,
                "mixed_row_order_preserved":true,"padded_input_rows":0,
                "image_position_identity_sha256":mrope.identity_sha256(),
                "shared_request_private_kv_gdn":true,"decode_path_changed":false,
                "product_ready":false,"qualified":false,"visible_tps":false});
            let mut generated = Vec::with_capacity(max_new_tokens);
            let mut text = String::new();
            let mut decoder = self.tokenizer.decode_stream(true);
            let mut first_visible = None;
            let mut last_token = None;
            let mut consumed = ids.len();
            let mut reason = "length";
            for index in 0..max_new_tokens {
                check_cancel(cancelled)?;
                selector.submit(&stream)?;
                let mut raw = [0u8; 4];
                token.copy_to_host_sync(&stream, &mut raw)?;
                let id = u32::from_le_bytes(raw);
                if id >= self.model.vocab_size as u32 {
                    return Err(invalid(
                        "Vision target produced invalid/non-finite selection",
                    ));
                }
                if self.eos.contains(&id) {
                    reason = "stop";
                    break;
                }
                generated.push(id);
                last_token = Some(started.elapsed().as_secs_f64());
                let delta = decoder
                    .step(id)
                    .map_err(|e| invalid(e.to_string()))?
                    .unwrap_or_default();
                if !delta.is_empty() {
                    first_visible.get_or_insert(started.elapsed().as_secs_f64());
                    text.push_str(&delta);
                }
                check_cancel(cancelled)?;
                sink(Qwen35VisionTextToken {
                    index,
                    token_id: id,
                    text: delta,
                })?;
                if index + 1 < max_new_tokens {
                    let next = self.target.embedding_row(id as usize)?;
                    submit_autoregressive_token(
                        &plans,
                        Some(&head),
                        &next,
                        &stream,
                        ids.len() + index,
                        &start_event,
                        &stop_event,
                        &terminal,
                    )?;
                    consumed += 1;
                }
            }
            check_cancel(cancelled)?;
            // A partial UTF8 token must not silently become a completed reply.
            if self
                .tokenizer
                .decode(&generated, true)
                .map_err(|e| invalid(e.to_string()))?
                != text
            {
                return Err(invalid("Vision stream ended with incomplete decoder text"));
            }
            let visual = kinds.iter().filter(|&&kind| kind == 1).count();
            let executed = encoded.iter().any(|e| e.vision_forward_executed);
            Ok(Qwen35VisionTextReport {
                schema: "qwen35-vision-native-text-v1",
                request_id: request_id.into(),
                prompt_tokens: ids.len(),
                consumed_input_tokens: consumed,
                image_input_rows_submitted: image_rows,
                image_encodings: encoded,
                image_embedding_cache_hits: cache_hits,
                mrope: mrope.receipt(),
                generated_ids: generated,
                generated_text: text,
                stop_reason: reason,
                request_wall_s: started.elapsed().as_secs_f64(),
                first_visible_text_s: first_visible,
                last_token_receipt_s: last_token,
                language_prefill,
                complete_model_sha256: self
                    .target
                    .manifest
                    .file_sha256
                    .clone()
                    .ok_or_else(|| invalid("Vision model identity missing"))?,
                allocation_before: Value::Null,
                allocation_after: Value::Null,
                request_allocations_balanced: false,
                model_loaded_once: true,
                request_private_kv_gdn: true,
                dflash_state: "disabled_target_only_vision_baseline",
                vision_forward_executed: executed,
                image_embeddings_consumed: visual > 0 && image_rows == visual,
                product_ready: false,
                qualified: false,
                visible_tps: false,
            })
        })();
        if let Err(error) = stream.synchronize() {
            self.poisoned = true;
            return Err(error.into());
        }
        drain.pending = false;
        result
    }
    pub fn close(self) -> Result<()> {
        let Self { vision, target, .. } = self;
        drop(target);
        vision.close().map_err(|e| invalid(e.to_string()))
    }
}
