//! Opt-in CUDA-event attribution of the retained mixed-row arithmetic.
//! This pass is diagnostic, never a profiler-free latency/TPS sample. Events
//! are reused only after the chunk's stream/epoch completion has been checked.
use super::*;
use std::collections::BTreeMap;
use std::ffi::OsStr;

const MAX_STAGE_MARKS: usize = 40;

pub(super) fn parse(value: Option<&OsStr>) -> Result<bool> {
    match value.and_then(OsStr::to_str) {
        None if value.is_none() => Ok(false),
        Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => Err(invalid("Vision prefill profile must be exactly 0 or 1")),
    }
}

fn valid_parts(parts: &[f64], total: f64) -> bool {
    total.is_finite()
        && total > 0.0
        && !parts.is_empty()
        && parts.iter().all(|x| x.is_finite() && *x >= 0.0)
        && parts.iter().sum::<f64>() <= total + (total * 0.0001).max(0.5)
}

struct Layer {
    events: Vec<OwnedCudaEvent>,
    marks: Vec<(String, &'static str)>,
    operators: BTreeMap<String, f64>,
    categories: BTreeMap<&'static str, f64>,
    kind: &'static str,
}
impl Layer {
    fn new(device: i32) -> Result<Self> {
        Ok(Self {
            events: (0..=MAX_STAGE_MARKS)
                .map(|_| OwnedCudaEvent::create(device))
                .collect::<std::result::Result<Vec<_>, _>>()?,
            marks: Vec::with_capacity(MAX_STAGE_MARKS),
            operators: BTreeMap::new(),
            categories: BTreeMap::new(),
            kind: "not_submitted",
        })
    }
    fn mark(&mut self, stream: &OwnedCudaStream, name: &str, category: &'static str) -> Result<()> {
        if self.marks.len() >= MAX_STAGE_MARKS {
            return Err(invalid("Vision profile stage budget"));
        }
        self.events[self.marks.len() + 1].record(stream)?;
        self.marks.push((name.to_owned(), category));
        Ok(())
    }
    fn projection(
        &mut self,
        stream: &OwnedCudaStream,
        cublas: &CudaQwen35PrefillCublasPlanV1,
        projection: &CudaQwen35PrefillCublasProjectionV1,
        name: &str,
    ) -> Result<()> {
        let i = self.marks.len() + 1;
        if i + 2 > MAX_STAGE_MARKS {
            return Err(invalid("Vision profile projection stage budget"));
        }
        cublas.submit_profiled(
            stream,
            projection,
            [&self.events[i], &self.events[i + 1], &self.events[i + 2]],
        )?;
        for category in ["weight_decode", "input_decode", "gemm"] {
            self.marks.push((format!("{name}.{category}"), category));
        }
        Ok(())
    }
    fn collect(&mut self) -> Result<()> {
        if self.marks.is_empty() {
            return Err(invalid("Vision profile missing layer stages"));
        }
        for (i, (name, category)) in self.marks.iter().enumerate() {
            let ms = f64::from(self.events[i].elapsed_ms(&self.events[i + 1])?);
            if !ms.is_finite() || ms < 0.0 {
                return Err(invalid("Vision profile invalid stage duration"));
            }
            *self.operators.entry(name.clone()).or_default() += ms;
            *self.categories.entry(category).or_default() += ms;
        }
        Ok(())
    }
}

pub(super) struct Profile {
    layers: Vec<Layer>,
    chunks: usize,
    event_records: usize,
}
impl Profile {
    pub(super) fn new(stream: &OwnedCudaStream) -> Result<Self> {
        Ok(Self {
            layers: (0..FULL_TARGET_LAYER_COUNT)
                .map(|_| Layer::new(stream.device_ordinal()))
                .collect::<Result<Vec<_>>>()?,
            chunks: 0,
            event_records: 0,
        })
    }
    pub(super) fn submit(
        &mut self,
        layer: usize,
        op: &PrefillLayerOps,
        stream: &OwnedCudaStream,
        cublas: &CudaQwen35PrefillCublasPlanV1,
    ) -> Result<()> {
        let l = self
            .layers
            .get_mut(layer)
            .ok_or_else(|| invalid("Vision profile layer extent"))?;
        l.marks.clear();
        l.events[0].record(stream)?;
        macro_rules! ffn {
            ($p:ident) => {{
                $p.ffn_input_quantize.submit_for_plan(stream)?;
                l.mark(stream, "ffn.input_quantize", "non_matrix")?;
                l.projection(stream, cublas, &$p.ffn_gate, "ffn.gate")?;
                l.projection(stream, cublas, &$p.ffn_up, "ffn.up")?;
                #[cfg(feature = "qwen35-prefill-fused-silu-q8")]
                $p.silu_mul_q8.submit_for_plan(stream)?;
                #[cfg(not(feature = "qwen35-prefill-fused-silu-q8"))]
                {
                    $p.silu_mul.submit_for_plan(stream)?;
                    $p.ffn_down_quantize.submit_for_plan(stream)?;
                }
                l.mark(stream, "ffn.silu_down_quantize", "non_matrix")?;
                l.projection(stream, cublas, &$p.ffn_down, "ffn.down")?;
                $p.final_residual.submit_for_plan(stream)?;
                l.mark(stream, "ffn.final_residual", "non_matrix")?;
            }};
        }
        match op {
            PrefillLayerOps::Linear(p) => {
                // The Vision worker does not install canonical scalar overrides.
                // Reject one rather than profile a different computation.
                if p.canonical_gdn_scalars.is_some() || p.simt_shared_gdn_scalars {
                    return Err(invalid("Vision profile rejects GDN scalar override"));
                }
                l.kind = "linear_gdn";
                p.attn_norm.submit(stream)?;
                p.normalized_quantize.submit_for_plan(stream)?;
                l.mark(stream, "input.norm_quantize", "non_matrix")?;
                l.projection(stream, cublas, &p.qkv, "input.qkv")?;
                l.projection(stream, cublas, &p.z, "input.z")?;
                l.projection(stream, cublas, &p.alpha, "input.alpha")?;
                l.projection(stream, cublas, &p.beta, "input.beta")?;
                p.submit_recurrence(stream)?;
                l.mark(stream, "gdn.recurrence", "gdn")?;
                p.recurrent_quantize.submit_for_plan(stream)?;
                l.mark(stream, "output.input_quantize", "non_matrix")?;
                l.projection(stream, cublas, &p.output, "output.projection")?;
                p.attention_residual.submit_for_plan(stream)?;
                p.post_norm.submit(stream)?;
                l.mark(stream, "output.residual_post_norm", "non_matrix")?;
                ffn!(p);
            }
            PrefillLayerOps::Full(p) => {
                l.kind = "full_attention";
                p.attn_norm.submit(stream)?;
                p.normalized_quantize.submit_for_plan(stream)?;
                l.mark(stream, "input.norm_quantize", "non_matrix")?;
                l.projection(stream, cublas, &p.q, "input.q")?;
                l.projection(stream, cublas, &p.k, "input.k")?;
                l.projection(stream, cublas, &p.v, "input.v")?;
                #[cfg(feature = "qwen35-kv8-chunk-prefill")]
                if p.simt_shared_softmax {
                    p.submit_shared_simt_attention(stream, cublas)?;
                } else {
                    p.attention.submit(stream)?;
                }
                #[cfg(not(feature = "qwen35-kv8-chunk-prefill"))]
                p.attention.submit(stream)?;
                l.mark(stream, "attention.full", "attention")?;
                #[cfg(all(
                    feature = "qwen35-kv8-full-attention",
                    not(feature = "qwen35-kv8-chunk-prefill")
                ))]
                p.cache_quantize.submit(stream)?;
                p.gated_quantize.submit_for_plan(stream)?;
                l.mark(stream, "output.input_quantize", "non_matrix")?;
                l.projection(stream, cublas, &p.output, "output.projection")?;
                p.attention_residual.submit_for_plan(stream)?;
                p.post_norm.submit(stream)?;
                l.mark(stream, "output.residual_post_norm", "non_matrix")?;
                ffn!(p);
            }
            
        }
        self.event_records += l.marks.len() + 1;
        Ok(())
    }
    pub(super) fn collect_chunk(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            layer.collect()?;
        }
        self.chunks += 1;
        Ok(())
    }
    pub(super) fn summary(&self, cuda_ms: f64) -> Result<Value> {
        let mut categories = BTreeMap::<&str, f64>::new();
        let layers = self.layers.iter().enumerate().map(|(index, l)| {
            for (category, ms) in &l.categories {
                *categories.entry(category).or_default() += ms;
            }
            json!({"layer":index,"kind":l.kind,"operator_ms":l.operators,"category_ms":l.categories})
        }).collect::<Vec<_>>();
        let parts = categories.values().copied().collect::<Vec<_>>();
        let accounted = parts.iter().sum::<f64>();
        if self.chunks == 0 || !valid_parts(&parts, cuda_ms) {
            return Err(invalid("Vision operator profile accounting"));
        }
        Ok(json!({"schema":"qwen35-vision-prefill-operator-profile-v1",
            "instrumented_only":true,"included_in_profiler_free_latency":false,
            "chunk_count":self.chunks,"event_records":self.event_records,
            "category_ms":categories,"accounted_stage_ms":accounted,
            "chunk_cuda_event_ms":cuda_ms,"between_layer_gap_ms":(cuda_ms-accounted).max(0.0),
            "accounting_valid":true,"layers":layers,
            "product_ready":false,"qualified":false,"visible_tps":false}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn profile_switch_is_explicit_and_not_a_truthy_environment() {
        assert!(!parse(None).unwrap());
        assert!(!parse(Some(OsStr::new("0"))).unwrap());
        assert!(parse(Some(OsStr::new("1"))).unwrap());
        for value in ["", "true", "01", "1 ", "2"] {
            assert!(parse(Some(OsStr::new(value))).is_err());
        }
    }
    #[test]
    fn exclusive_stage_accounting_rejects_nonfinite_negative_and_overcount() {
        assert!(valid_parts(&[1.0, 2.0, 3.0], 6.0));
        assert!(!valid_parts(&[1.0, 2.0, 4.0], 6.0));
        assert!(!valid_parts(&[f64::NAN], 6.0));
        assert!(!valid_parts(&[-1.0], 6.0));
        assert!(!valid_parts(&[], 6.0));
        assert!(!valid_parts(&[1.0], 0.0));
    }
}
