//! Retain immutable Vision weights/plans, never image activations or KV state.
use super::*;
use crate::{Qwen35VisionCudaEncoder, Qwen35VisionCudaOutput, Qwen35VisionHostWeights};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Policy { Auto, OnDemand }
impl Policy {
    fn parse(value: Option<&str>) -> Result<Self> {
        match value {
            Some("auto") => Ok(Self::Auto),
            None | Some("on_demand") => Ok(Self::OnDemand),
            _ => Err("AUGUR_VISION_RESIDENCY must be auto or on_demand".into()),
        }
    }
    fn label(self) -> &'static str { match self { Self::Auto => "auto", Self::OnDemand => "on_demand" } }
}

pub(crate) struct Cache {
    policy: Policy,
    encoder: Option<Qwen35VisionCudaEncoder>,
    bytes: u64,
    uploads: u64,
    hits: u64,
    evictions: u64,
    last_eviction: Option<String>,
    upload_wall_s: f64,
}

fn should_evict(resident: bool, free: u64, required: u64) -> bool {
    resident && free < required
}

impl Cache {
    pub(crate) fn from_env() -> Result<Self> {
        let value = std::env::var("AUGUR_VISION_RESIDENCY")
            .map(Some).or_else(|e| match e {
                std::env::VarError::NotPresent => Ok(None),
                _ => Err(e),
            })?;
        Ok(Self { policy: Policy::parse(value.as_deref())?, encoder: None, bytes: 0,
            uploads: 0, hits: 0, evictions: 0, last_eviction: None, upload_wall_s: 0.0 })
    }
    pub(super) fn resident(&self) -> bool { self.encoder.is_some() }
    pub(super) fn info(&self) -> Value {
        json!({"schema":"quasar-vision-residency-v1", "policy":self.policy.label(),
            "gpu_weights_resident":self.resident(), "resident_bytes":self.bytes,
            "uploads":self.uploads, "cache_hits":self.hits, "evictions":self.evictions,
            "last_eviction_reason":self.last_eviction, "total_upload_wall_s":self.upload_wall_s,
            "image_activations_retained":false, "numerical_path_changed":false})
    }
    pub(super) fn release(&mut self, reason: &str) -> Result<u64> {
        let Some(encoder) = self.encoder.take() else { return Ok(0); };
        let bytes = self.bytes;
        self.bytes = 0;
        self.evictions += 1;
        self.last_eviction = Some(reason.to_owned());
        // close drains the independent encoder stream before releasing plans/weights.
        encoder.close().map_err(|e| e.to_string())?;
        Ok(bytes)
    }
    pub(super) fn reclaim(&mut self, ordinal: i32, required: u64, reason: &str) -> Result<Value> {
        let free_before = augur_backend::cuda_memory_info(ordinal)?.free_bytes;
        let resident_before = self.bytes;
        let released = if should_evict(self.resident(), free_before, required) {
            self.release(reason)?
        } else { 0 };
        let free_after = if released > 0 { augur_backend::cuda_memory_info(ordinal)?.free_bytes } else { free_before };
        Ok(json!({"required_free_bytes":required, "free_before":free_before, "free_after":free_after,
            "resident_bytes_before":resident_before, "reclaimed_bytes":released,
            "evicted":released>0, "reason":reason, "sufficient_after":free_after>=required}))
    }
    pub(super) fn encode(&mut self, host: &Qwen35VisionHostWeights, ordinal: i32,
        images: &[Qwen35PreparedF16Image], cancelled: impl Fn() -> bool)
        -> Result<(Vec<Qwen35VisionCudaOutput>, Value)> {
        if images.is_empty() || cancelled() { return Err("Vision request cancelled or empty before upload".into()); }
        let started = Instant::now();
        let hit = self.resident();
        let upload = Instant::now();
        if hit { self.hits += 1; } else {
            let before = cuda_device_allocation_snapshot();
            let encoder = host.upload(ordinal).map_err(|e| e.to_string())?;
            self.bytes = cuda_device_allocation_snapshot().live_bytes.checked_sub(before.live_bytes)
                .ok_or("Vision upload allocation accounting underflow")?;
            self.encoder = Some(encoder);
            self.uploads += 1;
        }
        let upload_s = if hit { 0.0 } else { upload.elapsed().as_secs_f64() };
        self.upload_wall_s += upload_s;
        let encoder_bytes = self.bytes;
        let before = cuda_device_allocation_snapshot();
        let result: Result<Vec<Qwen35VisionCudaOutput>> = (|| {
            let mut outputs = Vec::with_capacity(images.len());
            for image in images {
                outputs.push(self.encoder.as_mut().ok_or("missing Vision encoder")?
                    .encode(image, &cancelled).map_err(|e| e.to_string())?);
            }
            Ok(outputs)
        })();
        // encode() drains on success AND on cancellation/error. All per-image
        // activations must disappear even when immutable weights remain cached.
        let after = cuda_device_allocation_snapshot();
        let closure = before.live_bytes == after.live_bytes && before.live_allocations == after.live_allocations;
        // A cancelled first upload restores its incoming residency state. An
        // already-cached immutable encoder survives a cancelled request.
        if self.policy == Policy::OnDemand || (result.is_err() && !hit) || !closure {
            self.release(if result.is_err() || !closure { "encode_error" } else { "on_demand" })?;
        }
        if !closure { return Err("Vision temporary allocation closure failed".into()); }
        let outputs = result?;
        Ok((outputs, json!({"cache_hit":hit, "upload_s":upload_s,
            "encoder_resident_bytes":encoder_bytes, "wall_including_upload_release_s":started.elapsed().as_secs_f64(),
            "temporary_allocation_closure":true, "residency":self.info()})))
    }
}
impl Drop for Cache {
    fn drop(&mut self) { let _ = self.release("owner_shutdown"); }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn policy_is_explicit_and_defaults_to_on_demand() {
        assert_eq!(Policy::parse(None).unwrap(), Policy::OnDemand);
        assert_eq!(Policy::parse(Some("auto")).unwrap(), Policy::Auto);
        assert_eq!(Policy::parse(Some("on_demand")).unwrap(), Policy::OnDemand);
        for value in ["", "1", "forever", "AUTO"] { assert!(Policy::parse(Some(value)).is_err()); }
    }
    #[test]
    fn memory_pressure_releases_only_when_needed() {
        assert!(!should_evict(true, 1024, 1024));
        assert!(!should_evict(true, 1025, 1024));
        assert!(should_evict(true, 1023, 1024));
        assert!(!should_evict(false, 0, u64::MAX));
        assert!(!should_evict(true, u64::MAX, u64::MAX));
    }
}
