//! QUASAR physical-layout adapter to the shared FP16/FP32 contraction.
//! No F64, hi/lo carry, limb workspace, or numerical acceptance claim.
use super::*;
use crate::cuda_qwen35_qpn::OwnedCudaQwen35QpnWeightV1;
use augur_cuda_sys::AugurCudaQwen35QpnPrefillArgsV1;
use std::rc::Rc;
const OP: &str = "augur_prefill";
pub const AUGUR_PREFILL_PROFILE: &str =
    "quasar_original_qpn_f16_f32_speed_first_v1";
pub const AUGUR_MAX_BATCH: usize =
    augur_cuda_sys::AUGUR_CUDA_QWEN35_PREFILL_CUBLAS_COMPILED_MAX_TOKENS_V1 as usize;
/// Measured choice for the opt-in large path; preserve B512 in ordinary builds.
pub const AUGUR_DEFAULT_BATCH: usize = if cfg!(feature = "qwen35-augur-large-prefill") {
    4352
} else {
    512
};
/// Literal input extent: no padding. Actual8236 tokens use4352+3884 in large builds.
pub fn augur_prefill_widths(rows: usize) -> Result<Vec<usize>, CudaRuntimeError> {
    augur_prefill_widths_for_batch(rows, AUGUR_DEFAULT_BATCH)
}
/// Explicit GEMM/GDN superbatch extent. Attention has its own512-row tile size.
pub fn augur_prefill_widths_for_batch(
    rows: usize,
    batch: usize,
) -> Result<Vec<usize>, CudaRuntimeError> {
    if rows == 0 || rows > 262144 || batch == 0 || batch > AUGUR_MAX_BATCH {
        return Err(CudaRuntimeError::invalid_argument(OP));
    }
    let mut result = vec![batch; rows / batch];
    if rows % batch != 0 {
        result.push(rows % batch);
    }
    Ok(result)
}
/// Retains all allocations and the exact serial stream owner through completion.
/// Caller shares one phase plan across all projections; never caches decoded weights.
pub struct CudaQwen35AugurProjectionV1 {
    plan: Rc<CudaQwen35PrefillCublasPlanV1>,
    _weight: Rc<OwnedCudaQwen35QpnWeightV1>,
    _input: Rc<OwnedCudaBuffer>,
    _output: Rc<OwnedCudaBuffer>,
    stream: Rc<OwnedCudaStream>,
    raw: AugurCudaQwen35QpnPrefillArgsV1,
    direct_b8: bool,
}
fn shared_input_rows(tokens:u32)->bool {(64..=AUGUR_MAX_BATCH as u32).contains(&tokens)}
fn same_input(a: &AugurCudaQwen35QpnPrefillArgsV1, b: &AugurCudaQwen35QpnPrefillArgsV1) -> bool {
    a.input_f32 == b.input_f32 && a.input_bytes == b.input_bytes
        && a.cols == b.cols && a.tokens == b.tokens
}
fn silu_matches_input(a: &augur_cuda_sys::AugurCudaSiluMulF32ArgsV1, down: &AugurCudaQwen35QpnPrefillArgsV1) -> bool {
    a.output == down.input_f32 && a.output_len_bytes == down.input_bytes
        && a.gate_len_bytes == down.input_bytes && a.up_len_bytes == down.input_bytes
        && u64::from(a.elements) == u64::from(down.tokens) * u64::from(down.cols)
}
impl CudaQwen35AugurProjectionV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn bind(
        plan: Rc<CudaQwen35PrefillCublasPlanV1>,
        stream: Rc<OwnedCudaStream>,
        weight: Rc<OwnedCudaQwen35QpnWeightV1>,
        input: Rc<OwnedCudaBuffer>,
        input_offset: usize,
        output: Rc<OwnedCudaBuffer>,
        output_offset: usize,
        tokens: usize,
    ) -> Result<Self, CudaRuntimeError> {
        if !(1..=AUGUR_MAX_BATCH).contains(&tokens) {
            return Err(CudaRuntimeError::invalid_argument(OP));
        }
        plan.require_stream(&stream)?;
        let direct_b8=plan.direct_b8_only;
        if direct_b8 && tokens!=8 {return Err(CudaRuntimeError::invalid_argument(OP));}
        let extent = |cols: usize| {
            tokens
                .checked_mul(cols)
                .and_then(|n| n.checked_mul(4))
                .ok_or_else(|| CudaRuntimeError::invalid_argument(OP))
        };
        let w = weight.physical_view()?;
        let x = input.view(input_offset, extent(weight.cols())?)?;
        let y = output.view(output_offset, extent(weight.rows())?)?;
        for view in [w, x, y] {
            if view.device_ordinal() != stream.device_ordinal() {
                return Err(CudaRuntimeError::invalid_argument(OP));
            }
        }
        let mut raw = AugurCudaQwen35QpnPrefillArgsV1::new();
        raw.weight = w.checked_address(OP)?;
        raw.weight_bytes = w.len_bytes() as u64;
        raw.input_f32 = x.checked_address(OP)?;
        raw.input_bytes = x.len_bytes() as u64;
        raw.output_f32 = y.checked_address(OP)?;
        raw.output_bytes = y.len_bytes() as u64;
        raw.rows = weight
            .rows()
            .try_into()
            .map_err(|_| CudaRuntimeError::invalid_argument(OP))?;
        raw.cols = weight
            .cols()
            .try_into()
            .map_err(|_| CudaRuntimeError::invalid_argument(OP))?;
        raw.tokens = tokens as u32;
        raw.tensor_scale = weight.tensor_scale();
        raw.split_k = 8;
        raw.numerical_profile = 2;
        // SAFETY: owners retain exact device views; native checks aliases and scratch.
        unsafe {
            CudaRuntimeError::from_result(
                OP,
                if direct_b8 {
                    augur_cuda_sys::augur_cuda_qwen35_qpn_prefill_preflight_v1(stream.device_ordinal(),stream.raw()?,plan.raw,&raw)
                } else {
                    augur_cuda_sys::augur_cuda_qwen35_augur_prefill_preflight_v1(stream.device_ordinal(),stream.raw()?,plan.raw,&raw)
                },
            )?;
        }
        Ok(Self {
            plan,
            _weight: weight,
            _input: input,
            _output: output,
            stream,
            raw,
            direct_b8,
        })
    }
    /// Reuse the established Rust QPN skinny-batch kernel, without expanding
    /// all weights to FP16 for every verification round. Profile2 is FP16 input,
    /// packed original weights and F32 MMA accumulation/postscale: no F64 carry.
    /// This is explicit B8-only admission; ordinary prefill remains unchanged.
    pub fn use_direct_b8(&mut self) -> Result<(), CudaRuntimeError> {
        if self.raw.tokens!=8 || self.raw.numerical_profile!=2 {return Err(CudaRuntimeError::invalid_argument(OP));}
        // SAFETY: the same immutable owners/ranges are checked by the existing ABI.
        unsafe {CudaRuntimeError::from_result(OP,augur_cuda_sys::augur_cuda_qwen35_qpn_prefill_preflight_v1(
            self.stream.device_ordinal(),self.stream.raw()?,self.plan.raw,&self.raw))?;}
        self.direct_b8=true;Ok(())
    }
    /// Submit a closed group sharing one immutable input on the serial plan.
    /// A stack array retains normal projection order without per-submit allocation.
    /// No cached-pointer heuristic: the first member always repacks, including
    /// on the next layer, chunk, invocation, or graph replay. B8/short paths stay
    /// on their original individual submit implementation.
    pub fn submit_same_input<const N: usize>(group: [&Self; N]) -> Result<(), CudaRuntimeError> {
        if !(2..=4).contains(&N) { return Err(CudaRuntimeError::invalid_argument(OP)); }
        let first = group[0];
        if group.iter().any(|p| !Rc::ptr_eq(&p.plan, &first.plan)
            || !Rc::ptr_eq(&p.stream, &first.stream) || !same_input(&p.raw, &first.raw)) {
            return Err(CudaRuntimeError::invalid_argument(OP));
        }
        // The legacy direct-B8 QPN entry point used to restage the same input
        // for every member. Use the established pair ABI so each pair stages
        // once, while retaining the individual fallback for an odd tail.
        if group.iter().any(|p| p.direct_b8) {
            if group.iter().any(|p| !p.direct_b8) {
                return Err(CudaRuntimeError::invalid_argument(OP));
            }
            let mut i = 0usize;
            while i + 1 < N {
                group[i].submit_shared_input_pair_direct(group[i + 1])?;
                i += 2;
            }
            if i < N { group[i].submit()?; }
            return Ok(());
        }
        // Match the existing Rust/QUASAR exact-input group admission at B64.
        if !shared_input_rows(first.raw.tokens) {
            for p in group { p.submit()?; }
            return Ok(());
        }
        first.plan.require_stream(&first.stream)?;
        let raw = group.map(|p| p.raw);
        // SAFETY: all immutable preflighted owners remain alive through this
        // synchronous host submission; the native call copies no host pointer.
        unsafe { CudaRuntimeError::from_result(OP,
            augur_cuda_sys::augur_cuda_qwen35_augur_prefill_group_submit_v1(
                first.stream.device_ordinal(), first.stream.raw()?, first.plan.raw,
                raw.as_ptr(), N as u32)) }
    }
    fn submit_shared_input_pair_direct(&self, other: &Self) -> Result<(), CudaRuntimeError> {
        if !self.direct_b8 || !other.direct_b8
            || !Rc::ptr_eq(&self.plan, &other.plan)
            || !Rc::ptr_eq(&self.stream, &other.stream)
            || !same_input(&self.raw, &other.raw)
        {
            return Err(CudaRuntimeError::invalid_argument(OP));
        }
        self.plan.require_stream(&self.stream)?;
        // SAFETY: both descriptors were preflighted and all owners remain alive
        // through the serial stream completion. The native pair entry point
        // stages the common F32 input exactly once.
        unsafe { CudaRuntimeError::from_result(OP,
            augur_cuda_sys::augur_cuda_qwen35_qpn_prefill_pair_prevalidated_submit_v1(
                self.stream.device_ordinal(), self.stream.raw()?, self.plan.raw,
                &self.raw, &other.raw)) }
    }

    /// Preserve the F32 activation and consume its FP16 input in one closed
    /// submission. Caller retains activation owners until completion, exactly
    /// as for submit_for_plan. Short/B8 and other shapes stay unchanged.
    pub fn submit_with_silu(&self, silu: &crate::CudaSiluMulF32ArgsV1) -> Result<(), CudaRuntimeError> {
        self.plan.require_stream(&self.stream)?;
        silu.validate_stream(&self.stream)?;
        if !silu_matches_input(silu.as_raw(), &self.raw) {
            return Err(CudaRuntimeError::invalid_argument(OP));
        }
        if self.direct_b8 {
            // Direct B8 has the same QPN projection math as before, but fuses
            // the already-required SiLU output write with F32->FP16 staging.
            return unsafe { CudaRuntimeError::from_result(OP,
                augur_cuda_sys::augur_cuda_qwen35_qpn_silu_down_direct_b8_prevalidated_submit_v1(
                    self.stream.device_ordinal(), self.stream.raw()?, self.plan.raw,
                    silu.as_raw(), &self.raw)) };
        }
        if !(512..=6400).contains(&self.raw.tokens)
            || self.raw.rows != 5120 || self.raw.cols != 17408 {
            silu.submit_for_plan(&self.stream)?;
            return self.submit();
        }
        // SAFETY: checked SiLU views and retained projection owners; native
        // validates all cross-operation ranges before any kernel launch.
        unsafe { CudaRuntimeError::from_result(OP,
            augur_cuda_sys::augur_cuda_qwen35_augur_silu_down_submit_v1(
                self.stream.device_ordinal(), self.stream.raw()?, self.plan.raw,
                silu.as_raw(), &self.raw)) }
    }
    pub fn submit(&self) -> Result<(), CudaRuntimeError> {
        self.plan.require_stream(&self.stream)?;
        if self.direct_b8 {
            // SAFETY: direct-B8 admission above retains the exact owner/stream.
            return unsafe {CudaRuntimeError::from_result(OP,augur_cuda_sys::augur_cuda_qwen35_qpn_prefill_prevalidated_submit_v1(
                self.stream.device_ordinal(),self.stream.raw()?,self.plan.raw,&self.raw))};
        }
        // SAFETY: immutable preflighted ranges, retained owners, same serial stream.
        unsafe {
            CudaRuntimeError::from_result(
                OP,
                augur_cuda_sys::augur_cuda_qwen35_augur_prefill_submit_v1(
                    self.stream.device_ordinal(),
                    self.stream.raw()?,
                    self.plan.raw,
                    &self.raw,
                ),
            )
        }
    }
}
/// One closed direct-B8 program: [residual add] -> rmsnorm -> FP16 stage -> 2..4
/// grouped profile2 projections of the normalized plane, with optional 64->48 column
/// trims and identical record stores. Native preflight checks every range and alias
/// once at bind; submit re-checks without driver queries. Bit-identical to the
/// separate norm/stage/projection/copy sequence it replaces.
pub struct CudaQwen35AugurNormGroupB8V1 {
    plan: Rc<CudaQwen35PrefillCublasPlanV1>,
    stream: Rc<OwnedCudaStream>,
    // The caller (one Program) owns the projections, norm, records and add operands
    // for this binding's whole lifetime, exactly as it does for the separate launches.
    raw: augur_cuda_sys::AugurCudaQwen35QpnNormGroupB8ArgsV1,
}
/// Per-member output: by default the projection's bound output; `output` redirects
/// it (e.g. 48 of a 64-row gate tile into a 48-wide plane). `record` receives the
/// identical words. Both extents must be exactly tokens*out_cols*4 bytes.
#[derive(Clone, Copy)]
pub struct NormGroupMemberV1 {
    pub out_cols: usize,
    pub output: Option<crate::CudaBufferView>,
    pub record: Option<crate::CudaBufferView>,
}
impl CudaQwen35AugurNormGroupB8V1 {
    pub fn bind(
        norm: &super::CudaQwen35PrefillRmsnormV1,
        members: &[(&CudaQwen35AugurProjectionV1, NormGroupMemberV1)],
        residual: Option<&crate::CudaAddF32ArgsV1>,
    ) -> Result<Self, CudaRuntimeError> {
        let Some((first, _)) = members.first() else { return Err(CudaRuntimeError::invalid_argument(OP)) };
        let (plan, stream) = (Rc::clone(&first.plan), Rc::clone(&first.stream));
        if !(2..=4).contains(&members.len()) {return Err(CudaRuntimeError::invalid_argument(OP));}
        if !norm.is_default_arithmetic() {return Err(CudaRuntimeError::invalid_argument(OP));}
        let mut raw = augur_cuda_sys::AugurCudaQwen35QpnNormGroupB8ArgsV1::new();
        raw.count = members.len() as u32;
        raw.norm = *norm.as_raw();
        for (i, (p, extra)) in members.iter().enumerate() {
            if !p.direct_b8 || !Rc::ptr_eq(&p.plan, &plan) || !Rc::ptr_eq(&p.stream, &stream) {
                return Err(CudaRuntimeError::invalid_argument(OP));
            }
            let mut m = p.raw;
            let out = u32::try_from(extra.out_cols).map_err(|_| CudaRuntimeError::invalid_argument(OP))?;
            m.output_bytes = u64::from(out) * u64::from(m.tokens) * 4;
            if let Some(output) = extra.output {
                m.output_f32 = output.checked_address(OP)?;
                if output.len_bytes() as u64 != m.output_bytes {return Err(CudaRuntimeError::invalid_argument(OP));}
            } else if out != m.rows {return Err(CudaRuntimeError::invalid_argument(OP));}
            raw.projections[i] = m;
            raw.out_cols[i] = out;
            if let Some(record) = extra.record {
                raw.records[i] = record.checked_address(OP)?;
                raw.record_bytes[i] = record.len_bytes() as u64;
            }
        }
        // The residual is exactly this already-bound add, whose output is the norm input.
        if let Some(add) = residual {
            let a = add.as_raw();
            if a.output != raw.norm.input_f32 || a.output_len_bytes != raw.norm.input_len_bytes
                || a.a_len_bytes != a.output_len_bytes || a.b_len_bytes != a.output_len_bytes
                || u64::from(a.elements) * 4 != a.output_len_bytes {
                return Err(CudaRuntimeError::invalid_argument(OP));
            }
            raw.residual = 1;
            raw.add_a = a.a;
            raw.add_b = a.b;
            raw.add_len_bytes = a.a_len_bytes;
        }
        plan.require_stream(&stream)?;
        // SAFETY: preflight reads only this descriptor and queries every range's
        // allocation; the caller keeps all owners alive for the binding's lifetime.
        unsafe { CudaRuntimeError::from_result(OP,
            augur_cuda_sys::augur_cuda_qwen35_qpn_norm_group_b8_preflight_v1(
                stream.device_ordinal(), stream.raw()?, plan.raw, &raw))?; }
        Ok(Self { plan, stream, raw })
    }
    pub fn submit(&self) -> Result<(), CudaRuntimeError> {
        self.plan.require_stream(&self.stream)?;
        // SAFETY: the same preflighted descriptor, retained owners and serial stream.
        unsafe { CudaRuntimeError::from_result(OP,
            augur_cuda_sys::augur_cuda_qwen35_qpn_norm_group_b8_prevalidated_submit_v1(
                self.stream.device_ordinal(), self.stream.raw()?, self.plan.raw, &self.raw)) }
    }
}
impl Drop for CudaQwen35AugurNormGroupB8V1 {
    fn drop(&mut self) { let _ = self.stream.synchronize(); }
}
impl Drop for CudaQwen35AugurProjectionV1 {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}
#[cfg(test)]
#[path = "norm_group_tests.rs"]
mod norm_group_tests;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn silu_down_requires_exact_extent_and_input_identity() {
        let mut down = AugurCudaQwen35QpnPrefillArgsV1::new();
        down.tokens=4352; down.cols=17408; down.input_f32=4096;
        down.input_bytes=u64::from(down.tokens)*u64::from(down.cols)*4;
        let mut a=augur_cuda_sys::AugurCudaSiluMulF32ArgsV1::new();
        a.output=down.input_f32; a.output_len_bytes=down.input_bytes;
        a.gate_len_bytes=down.input_bytes; a.up_len_bytes=down.input_bytes;
        a.elements=down.tokens*down.cols;
        assert!(silu_matches_input(&a,&down));
        let good=a;
        a.output+=4; assert!(!silu_matches_input(&a,&down)); a=good;
        a.output_len_bytes-=4; assert!(!silu_matches_input(&a,&down)); a=good;
        a.gate_len_bytes-=4; assert!(!silu_matches_input(&a,&down)); a=good;
        a.up_len_bytes-=4; assert!(!silu_matches_input(&a,&down)); a=good;
        a.elements-=1; assert!(!silu_matches_input(&a,&down));
    }
    #[test]
    fn null_silu_down_rejects_without_gpu_work() {
        unsafe {
            let r=augur_cuda_sys::augur_cuda_qwen35_augur_silu_down_submit_v1(
                0,std::ptr::null_mut(),std::ptr::null_mut(),std::ptr::null(),std::ptr::null());
            assert_ne!(r.status,0);
            let a=augur_cuda_sys::AugurCudaSiluMulF32ArgsV1::new();
            let r=augur_cuda_sys::augur_cuda_qwen35_augur_silu_down_submit_v1(
                0,std::ptr::null_mut(),std::ptr::null_mut(),&a,std::ptr::null());
            assert_ne!(r.status,0);
        }
    }
    #[test]
    fn short_input_group_matches_augur_b64_floor() {
        for n in [0,1,8,17,55,63] {assert!(!shared_input_rows(n));}
        for n in [64,65,73,79,327,511,512] {assert!(shared_input_rows(n));}
        assert!(!shared_input_rows(AUGUR_MAX_BATCH as u32+1));
    }
    #[test]
    fn real_rows_only() {
        for n in [1, 8, 44, 511, 512, 513, 8236, 8477, 262144] {
            let w = augur_prefill_widths(n).unwrap();
            assert_eq!(w.iter().sum::<usize>(), n);
            assert!(w
                .iter()
                .all(|x| (1..=AUGUR_DEFAULT_BATCH).contains(x)));
        }
        assert_eq!(
            augur_prefill_widths(8236).unwrap(),
            if AUGUR_DEFAULT_BATCH == 4352 {
                vec![4352, 3884]
            } else {
                [vec![512; 16], vec![44]].concat()
            }
        );
    }
    #[test]
    fn bounded_schedule() {
        for n in [0, 262145, usize::MAX] {
            assert!(augur_prefill_widths(n).is_err());
        }
    }
    #[test]
    fn same_input_requires_exact_view_and_shape_not_weight_shape() {
        let mut a = AugurCudaQwen35QpnPrefillArgsV1::new();
        a.input_f32=4096; a.input_bytes=4352*5120*4; a.cols=5120; a.tokens=4352;
        let mut b=a; b.rows=17408; b.weight=8192;
        assert!(same_input(&a,&b));
        b.input_f32+=8; assert!(!same_input(&a,&b)); b=a;
        b.input_bytes-=4; assert!(!same_input(&a,&b)); b=a;
        b.cols=6144; assert!(!same_input(&a,&b)); b=a;
        b.tokens=3884; assert!(!same_input(&a,&b));
    }
    #[test]
    fn null_group_rejects_before_gpu_and_before_second_descriptor() {
        let a=AugurCudaQwen35QpnPrefillArgsV1::new();
        for count in [0,1,2,4,5] {
            // SAFETY: invalid count or null owner rejects before a second read.
            let r=unsafe { augur_cuda_sys::augur_cuda_qwen35_augur_prefill_group_submit_v1(
                0,std::ptr::null_mut(),std::ptr::null_mut(),&a,count) };
            assert_ne!(r.status,0);
        }
    }
    #[test]
    fn null_native_owner_rejects_without_cuda() {
        let a = AugurCudaQwen35QpnPrefillArgsV1::new();
        // SAFETY: explicit null-owner rejection tests; no GPU operation is submitted.
        unsafe {
            let p = augur_cuda_sys::augur_cuda_qwen35_augur_prefill_preflight_v1(
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &a,
            );
            let s = augur_cuda_sys::augur_cuda_qwen35_augur_prefill_submit_v1(
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &a,
            );
            assert_ne!(p.status, 0);
            assert_ne!(s.status, 0);
        }
    }
}
