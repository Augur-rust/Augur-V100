//! Augur's real Vision -> language ingress. Uses the same64-layer target,
//! rotated physical KV and B4352 schedule; no legacy GGUF target substitution.
use super::*;
use augur_model_qwen35::{Qwen35PreparedF16Image,Qwen35ImagePromptLayout,Qwen35ImageMropePlan};
use serde_json::{json,Value};
pub(crate) mod residency;

impl CheckpointCudaTarget {
    pub(crate) fn vision_residency_info(&self) -> Value { self.vision_cache.info() }
    pub(crate) fn reclaim_vision_for_budget(&mut self, required: u64, reason: &str) -> Result<Value> {
        self.vision_cache.reclaim(self.stream.device_ordinal(), required, reason)
    }
    /// Readiness executes all 27 blocks. Only immutable encoder weights/plans
    /// remain in auto mode; image activations are drained and freed.
    pub(crate) fn initialize_vision(&mut self)->Result<Value> {
        let started=Instant::now();
        let host=self.prepare_vision_host()?;
        let grid=augur_model_qwen35::Qwen35ImageGrid::from_aligned_pixels(256,256)?;
        let image=augur_model_qwen35::prepare_qwen35_aligned_rgb8_f16(&vec![128;256*256*3],grid,32*1024*1024)?;
        let (mut outputs, cache)=self.vision_cache.encode(self.vision_host.as_ref().unwrap(),
            self.stream.device_ordinal(), &[image], ||false)?;
        let resident=self.vision_cache.resident();
        self.info["vision_host"]["gpu_weights_resident"]=resident.into();
        Ok(json!({"host":host,"encoder_smoke":outputs.remove(0),"allocation_closure":true,
            "allocation_closure_scope":"temporary_image_buffers", "residency":cache,
            "startup_wall_s":started.elapsed().as_secs_f64(),"gpu_weights_resident":resident,
            "vision_language_semantically_qualified":false,"product_ready":false}))
    }
    /// Authenticated CPU-only Vision retention. Repeat calls do not reload or
    /// rehash model files and do not allocate GPU weights.
    pub fn prepare_vision_host(&mut self)->Result<Value> {
        self.validate_cursor()?;
        if self.info["profile"]!="quasar_nvfp4" {return Err("QUASAR Vision source required".into());}
        let started=Instant::now();
        if self.vision_host.is_none() {
            let host=crate::Qwen35VisionHostWeights::from_checkpoint(&mut self._artifact)
                .map_err(|e|e.to_string())?;
            self.vision_host=Some(host);
        }
        let mut info=self.vision_host.as_ref().unwrap().info();
        info["gpu_weights_resident"]=self.vision_cache.resident().into();
        info["prepare_host_wall_s"]=started.elapsed().as_secs_f64().into();
        self.info["vision_host"]=info.clone();
        Ok(info)
    }
    /// First image prompt for an empty owner. Appended text consumes the same
    /// MRoPE continuation tables. Multimodal session/history authentication is
    /// a higher-level contract; this API does not silently reset an active turn.
    pub fn prefill_images(&mut self,tokens:&[u32],images:&[Qwen35PreparedF16Image],
        cancelled:impl Fn()->bool)->Result<Value> {
        self.validate_cursor()?;
        if self.consumed!=0 || self.image_rope_active || self.augur_decode.is_none()
            || images.is_empty() {return Err("image prefill needs an empty rotated-decoder owner and at least one image".into());}
        validate_sequence(0,self.capacity,tokens)?;
        for image in images {image.validate_payload_identity()?;}
        let kinds:Vec<u8>=tokens.iter().map(|&id|u8::from(id==248056)).collect();
        let grids:Vec<_>=images.iter().map(|x|x.grid).collect();
        let layout=Qwen35ImagePromptLayout::new(tokens,&kinds,&grids)?;
        let mrope=Qwen35ImageMropePlan::new(&layout,self.capacity,self.capacity*32*8)?;
        if layout.image_token_indices().len()>crate::augur_session::images::MAX_IMAGE_TOKENS {
            return Err("image history exceeds merged-token resource budget".into());
        }
        if cancelled() {return Err("image request cancelled before Vision load".into());}
        let started=Instant::now();
        let host_info=self.prepare_vision_host()?;
        let free_before=augur_backend::cuda_memory_info(self.stream.device_ordinal())?.free_bytes;
        let (mut outputs, encoder_cache)=self.vision_cache.encode(self.vision_host.as_ref().unwrap(),
            self.stream.device_ordinal(), images, &cancelled)?;
        if cancelled() {return Err("image request cancelled before target mutation".into());}
        let identity=json!({"prepared_images":outputs.iter().map(|x|&x.prepared_payload_sha256).collect::<Vec<_>>(),
            "weights":host_info,"positions":mrope.receipt()});
        let embeddings:Vec<f32>=outputs.iter().flat_map(|x|x.embeddings_f32.iter().copied()).collect();
        let device_rows=augur_prefill::image_rows::DeviceRows::upload(Rc::clone(&self.stream),tokens,
            layout.image_token_indices().to_vec(),&embeddings,identity)?;
        let widths=augur_backend::cuda_qwen35_prefill_cublas::augur_prefill_widths(tokens.len())?;
        let plan=self.plan_augur_sequence(tokens,widths)?;
        // After the first table write, any error requires reset; the same
        // allocations are used by both prefill and selected-token continuation.
        self.poisoned=true;self.image_rope_active=true;
        self.rope.install_image(&mrope)?;
        self.poisoned=false;
        let prefill=match self.consume_sequence_augur_image_rows(tokens,&plan,&device_rows,|_|cancelled()) {
            Ok(receipt)=>receipt,
            Err(error)=>{self.poisoned=true;return Err(error);}
        };
        drop(device_rows);
        for output in &mut outputs {output.target_image_consumed=true;}
        let receipt=json!({"schema":"quasar-vision-language-prefill-v1","image_encodings":outputs,
            "mrope":mrope.receipt(),"language_prefill":prefill,"vision_forward_executed":true,
            "image_embeddings_consumed":true,"physical_kv_capacity":self.capacity,
            "vision_gpu_weights_released_before_prefill":!self.vision_cache.resident(),
            "vision_encoder_resident_bytes":encoder_cache["encoder_resident_bytes"],
            "vision_encoder_cache_hit":encoder_cache["cache_hit"], "vision_weight_upload_s":encoder_cache["upload_s"],
            "vision_residency":self.vision_cache.info(),
            "vision_encoder_wall_including_upload_release_s":encoder_cache["wall_including_upload_release_s"],"gpu_free_before_encoder":free_before,
            "vision_phase_allocation_closure":true,"vision_allocation_closure_scope":"temporary_image_buffers","complete_vision_prefill_s":started.elapsed().as_secs_f64(),
            "product_ready":false,"numerical_qualified":false});
        self.info["active_image_input"]=receipt.clone();
        Ok(receipt)
    }
}

#[cfg(test)]
mod tests;
