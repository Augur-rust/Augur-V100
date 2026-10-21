//! Q8_0 codebooks + resident F32 gates; no repeated hidden projection.
use crate::{CudaBufferView, CudaRuntimeError, CudaRuntimeErrorKind, OwnedCudaStream};

pub const QWEN35_DFLASH_Q8_SELECTOR_BYTES: [usize; 7] = [
    248320 * 8 * 34,
    248320 * 8 * 34,
    8 * 16 * 4,
    8 * 16 * 4,
    8 * 256 * 4,
    8 * 16 * 4,
    8 * 4,
];
/// Original BF16 codebooks with precomputed F32 gates. This entry only
/// exposes the original greedy scale=1 walk; calibration routes remain Q8.
#[derive(Clone,Copy,Debug)]
pub struct CudaQwen35DflashBf16CodebooksArgsV1(CudaQwen35DflashQ8SelectorArgsV1,augur_cuda_sys::AugurCudaStream);
impl CudaQwen35DflashBf16CodebooksArgsV1 {
    pub fn bind(stream:&OwnedCudaStream,views:CudaQwen35DflashQ8SelectorViewsV1,anchor:u32)->Result<Self,CudaRuntimeError> {
        let mut sizes=QWEN35_DFLASH_Q8_SELECTOR_BYTES;sizes[0]=248320*256*2;sizes[1]=sizes[0];
        let args=CudaQwen35DflashQ8SelectorArgsV1::bind_with_sizes(stream.device_ordinal(),views,anchor,sizes)?;
        Ok(Self(args,stream.raw()?))
    }
    pub fn submit(&self,stream:&OwnedCudaStream)->Result<(),CudaRuntimeError> {
        if stream.device_ordinal()!=self.0.ordinal{return Err(CudaRuntimeError::device_mismatch());}
        if stream.raw()?!=self.1{return Err(error(CudaRuntimeErrorKind::InvalidLayout));}
        // SAFETY: exact disjoint BF16/F32 views; the caller retains all owners.
        let result=unsafe{augur_cuda_sys::augur_cuda_qwen35_dflash_bf16_codebooks_submit_v1(self.0.ordinal,stream.raw()?,&self.0.raw)};
        CudaRuntimeError::from_result("cuda_qwen35_dflash_bf16_codebooks_submit_v1",result)
    }
}
/// Two resident gate tensors selected in one left-to-right walk. Closing the
/// thinking phase is sticky within the path, including an already closed anchor.
#[derive(Clone,Copy,Debug)]
pub struct CudaQwen35DflashQ8SelectorPhaseArgsV1 {
    base:CudaQwen35DflashQ8SelectorArgsV1, after:u64, close:u32, stream:augur_cuda_sys::AugurCudaStream,
}
impl CudaQwen35DflashQ8SelectorPhaseArgsV1 {
    pub fn bind(stream:&OwnedCudaStream,views:CudaQwen35DflashQ8SelectorViewsV1,anchor:u32,
        after:CudaBufferView,close:u32)->Result<Self,CudaRuntimeError> {
        let base=CudaQwen35DflashQ8SelectorArgsV1::bind(stream,views,anchor)?;
        let address=phase_gate_address(base.ordinal,views,after,close)?;
        Ok(Self{base,after:address,close,stream:stream.raw()?})
    }
    pub fn submit(&self,stream:&OwnedCudaStream)->Result<(),CudaRuntimeError> {
        if stream.device_ordinal()!=self.base.ordinal{return Err(CudaRuntimeError::device_mismatch());}
        if stream.raw()?!=self.stream{return Err(error(CudaRuntimeErrorKind::InvalidLayout));}
        // SAFETY: all eight exact-size nonoverlapping device views and this stream
        // were checked at bind; caller retains their owners through completion.
        let result=unsafe{augur_cuda_sys::augur_cuda_qwen35_dflash_q8_selector_phase_submit_v1(
            self.base.ordinal,stream.raw()?,&self.base.raw,self.after,(8*256*4) as u64,self.close)};
        CudaRuntimeError::from_result("cuda_qwen35_dflash_q8_selector_phase_submit_v1",result)
    }
}
fn phase_gate_address(ordinal:i32,views:CudaQwen35DflashQ8SelectorViewsV1,after:CudaBufferView,close:u32)->Result<u64,CudaRuntimeError> {
    if after.device_ordinal()!=ordinal{return Err(CudaRuntimeError::device_mismatch());}
    if close>=248320 || after.len_bytes()!=8*256*4{return Err(error(CudaRuntimeErrorKind::InvalidDimensions));}
    let start=after.device_ptr().raw_value().checked_add(after.offset_bytes() as u64).ok_or_else(CudaRuntimeError::range_overflow)?;
    let end=start.checked_add(after.len_bytes() as u64).ok_or_else(CudaRuntimeError::range_overflow)?;
    if start==0 || !start.is_multiple_of(4){return Err(error(CudaRuntimeErrorKind::InvalidLayout));}
    for b in views.buffers(){
        let other=b.device_ptr().raw_value().checked_add(b.offset_bytes() as u64).ok_or_else(CudaRuntimeError::range_overflow)?;
        let other_end=other.checked_add(b.len_bytes() as u64).ok_or_else(CudaRuntimeError::range_overflow)?;
        if start<other_end && other<end{return Err(error(CudaRuntimeErrorKind::InvalidLayout));}
    }
    Ok(start)
}

const OP: &str = "cuda_qwen35_dflash_q8_selector_submit_v1";

#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35DflashQ8SelectorViewsV1 {
    pub predecessor: CudaBufferView,
    pub successor: CudaBufferView,
    pub candidate_ids: CudaBufferView,
    pub unary_scores: CudaBufferView,
    pub gates: CudaBufferView,
    pub transition_scores: CudaBufferView,
    pub path_ids: CudaBufferView,
}
impl CudaQwen35DflashQ8SelectorViewsV1 {
    fn buffers(self) -> [CudaBufferView; 7] {
        [
            self.predecessor,
            self.successor,
            self.candidate_ids,
            self.unary_scores,
            self.gates,
            self.transition_scores,
            self.path_ids,
        ]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CudaQwen35DflashQ8SelectorArgsV1 {
    ordinal: i32,
    raw: augur_cuda_sys::AugurCudaQwen35DflashQ8SelectorArgsV1,
}
impl CudaQwen35DflashQ8SelectorArgsV1 {
    pub fn bind(
        stream: &OwnedCudaStream,
        views: CudaQwen35DflashQ8SelectorViewsV1,
        anchor: u32,
    ) -> Result<Self, CudaRuntimeError> {
        Self::bind_for_ordinal(stream.device_ordinal(), views, anchor)
    }
    fn bind_for_ordinal(
        ordinal: i32,
        views: CudaQwen35DflashQ8SelectorViewsV1,
        anchor: u32,
    ) -> Result<Self, CudaRuntimeError> {
        Self::bind_with_sizes(ordinal,views,anchor,QWEN35_DFLASH_Q8_SELECTOR_BYTES)
    }
    fn bind_with_sizes(ordinal:i32,views:CudaQwen35DflashQ8SelectorViewsV1,anchor:u32,sizes:[usize;7])->Result<Self,CudaRuntimeError> {
        if ordinal < 0 || anchor >= 248320 {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let buffers = views.buffers();
        let mut ranges = [(0u64, 0u64); 7];
        for (i, b) in buffers.iter().enumerate() {
            if b.device_ordinal() != ordinal {
                return Err(CudaRuntimeError::device_mismatch());
            }
            if b.len_bytes() != sizes[i] {
                return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
            }
            let start = b
                .device_ptr()
                .raw_value()
                .checked_add(b.offset_bytes() as u64)
                .ok_or_else(CudaRuntimeError::range_overflow)?;
            let end = start
                .checked_add(b.len_bytes() as u64)
                .ok_or_else(CudaRuntimeError::range_overflow)?;
            if start == 0
                || !start.is_multiple_of(if i < 2 { 2 } else { 4 })
                || ranges[..i].iter().any(|&(s, e)| start < e && s < end)
            {
                return Err(error(CudaRuntimeErrorKind::InvalidLayout));
            }
            ranges[i] = (start, end);
        }
        let p = |i: usize| ranges[i].0;
        let l = |i: usize| sizes[i] as u64;
        Ok(Self {
            ordinal,
            raw: augur_cuda_sys::AugurCudaQwen35DflashQ8SelectorArgsV1 {
                abi_version: 1,
                struct_size: 144,
                rows: 8,
                top_k: 16,
                rank: 256,
                vocab: 248320,
                anchor_token: anchor,
                reserved: 0,
                predecessor: p(0),
                predecessor_len_bytes: l(0),
                successor: p(1),
                successor_len_bytes: l(1),
                candidate_ids: p(2),
                candidate_ids_len_bytes: l(2),
                unary_scores: p(3),
                unary_scores_len_bytes: l(3),
                gates: p(4),
                gates_len_bytes: l(4),
                transition_scores: p(5),
                transition_scores_len_bytes: l(5),
                path_ids: p(6),
                path_ids_len_bytes: l(6),
            },
        })
    }
    /// Submission only. Retain all seven owners through stream completion.
    /// Output is a draft path, not target verification or token publication.
    pub fn submit(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if stream.device_ordinal() != self.ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        // SAFETY: exact Q8/F32 extents, ordinal, alignment and disjoint ranges
        // checked at bind; the caller retains owners until stream completion.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_dflash_q8_selector_submit_v1(
                self.ordinal,
                stream.raw()?,
                &self.raw,
            )
        };
        CudaRuntimeError::from_result(OP, result)
    }
    /// Opt-in selector calibration. Owners must remain alive through stream completion.
    /// Scale 1 selects the original kernel; invalid scales reject before enqueue.
    pub fn submit_with_transition_scale(
        &self,
        stream: &OwnedCudaStream,
        scale: f32,
    ) -> Result<(), CudaRuntimeError> {
        if !valid_transition_scale(scale) {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        if stream.device_ordinal() != self.ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        // SAFETY: bind checked exact extents and disjoint ranges; caller retains owners.
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_dflash_q8_selector_scaled_submit_v1(
                self.ordinal,
                stream.raw()?,
                &self.raw,
                scale,
            )
        };
        CudaRuntimeError::from_result(OP, result)
    }
    /// Experimental dynamic-programming path selector. It keeps the same
    /// top-16 candidate tensors and descriptor, but chooses the best causal
    /// path instead of greedily fixing one predecessor per row.
    pub fn submit_with_transition_dp(
        &self,
        stream: &OwnedCudaStream,
        scale: f32,
    ) -> Result<(), CudaRuntimeError> {
        if !valid_transition_scale(scale) {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        if stream.device_ordinal() != self.ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_dflash_q8_selector_dp_submit_v1(
                self.ordinal, stream.raw()?, &self.raw, scale,
            )
        };
        CudaRuntimeError::from_result(OP, result)
    }
    /// Experimental rank-compensated greedy path. A positive bonus softens
    /// the top-1 unary bias while keeping the historical transition scale.
    pub fn submit_with_rank_compensation(
        &self,
        stream: &OwnedCudaStream,
        rank_bonus: f32,
    ) -> Result<(), CudaRuntimeError> {
        if !valid_transition_scale(rank_bonus) {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        if stream.device_ordinal() != self.ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_dflash_q8_selector_rank_comp_submit_v1(
                self.ordinal, stream.raw()?, &self.raw, rank_bonus,
            )
        };
        CudaRuntimeError::from_result(OP, result)
    }
    /// Experimental tail rank-compensated path. Rows 1-2 keep both the
    /// original transition and unary ranking; later rows use the supplied
    /// transition scale plus rank compensation.
    pub fn submit_with_tail_rank_compensation(
        &self,
        stream: &OwnedCudaStream,
        transition_scale: f32,
        rank_bonus: f32,
    ) -> Result<(), CudaRuntimeError> {
        if !valid_transition_scale(transition_scale) || !valid_transition_scale(rank_bonus) {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        if stream.device_ordinal() != self.ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_dflash_q8_selector_tail_rank_comp_submit_v1(
                self.ordinal, stream.raw()?, &self.raw, transition_scale, rank_bonus,
            )
        };
        CudaRuntimeError::from_result(OP, result)
    }
    /// Opt-in scales for positions 1/2/3, fused into the existing selector walk.
    /// Later positions retain scale 1; no extra allocation or host transfer.
    pub fn submit_with_early_position_scales(
        &self,
        stream: &OwnedCudaStream,
        scales: [f32; 3],
    ) -> Result<(), CudaRuntimeError> {
        if !scales.iter().copied().all(valid_transition_scale) {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        if stream.device_ordinal() != self.ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_dflash_q8_selector_position_scaled_submit_v1(
                self.ordinal, stream.raw()?, &self.raw, scales[0], scales[1], scales[2],
            )
        };
        CudaRuntimeError::from_result(OP, result)
    }
    /// Experimental early-position calibration. Rows 1-3 use the requested
    /// transition scale; rows 4-7 retain the original scale.
    pub fn submit_with_early_transition_scale(
        &self,
        stream: &OwnedCudaStream,
        scale: f32,
    ) -> Result<(), CudaRuntimeError> {
        if !valid_transition_scale(scale) {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        if stream.device_ordinal() != self.ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_dflash_q8_selector_early_scaled_submit_v1(
                self.ordinal, stream.raw()?, &self.raw, scale,
            )
        };
        CudaRuntimeError::from_result(OP, result)
    }
    /// Experimental tail-calibrated greedy path. Rows 1-2 retain the original
    /// transition weight; later rows use the requested lower transition scale.
    pub fn submit_with_tail_transition_scale(
        &self,
        stream: &OwnedCudaStream,
        scale: f32,
    ) -> Result<(), CudaRuntimeError> {
        if !valid_transition_scale(scale) {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        if stream.device_ordinal() != self.ordinal {
            return Err(CudaRuntimeError::device_mismatch());
        }
        let result = unsafe {
            augur_cuda_sys::augur_cuda_qwen35_dflash_q8_selector_tail_scaled_submit_v1(
                self.ordinal, stream.raw()?, &self.raw, scale,
            )
        };
        CudaRuntimeError::from_result(OP, result)
    }
}
impl CudaQwen35DflashQ8SelectorArgsV1 {
    fn probability_address(&self,q:CudaBufferView,start:u64,positions:u32)->Result<u64,CudaRuntimeError> {
        if q.device_ordinal()!=self.ordinal {return Err(CudaRuntimeError::device_mismatch());}
        if q.len_bytes()!=8*16*8 || !(1..=7).contains(&positions) || start>u64::MAX-8 {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let address=q.device_ptr().raw_value().checked_add(q.offset_bytes() as u64).ok_or_else(CudaRuntimeError::range_overflow)?;
        let end=address.checked_add(q.len_bytes() as u64).ok_or_else(CudaRuntimeError::range_overflow)?;
        if address==0 || !address.is_multiple_of(8) {return Err(error(CudaRuntimeErrorKind::InvalidLayout));}
        let a=&self.raw;
        for (ptr,size) in [a.predecessor,a.successor,a.candidate_ids,a.unary_scores,a.gates,a.transition_scores,a.path_ids].into_iter().zip(QWEN35_DFLASH_Q8_SELECTOR_BYTES) {
            let other_end=ptr.checked_add(size as u64).ok_or_else(CudaRuntimeError::range_overflow)?;
            if address<other_end && ptr<end {return Err(error(CudaRuntimeErrorKind::InvalidLayout));}
        }
        Ok(address)
    }
    /// Calibrate only proposal pos2-4; the exact q is retained for rejection sampling.
    pub fn submit_probability_pos234(&self,stream:&OwnedCudaStream,q:CudaBufferView,start:u64,positions:u32,temperature:f32,pair_scale:f32)->Result<(),CudaRuntimeError> {
        if !valid_probability_calibration(positions,temperature,pair_scale) {return Err(error(CudaRuntimeErrorKind::InvalidLayout));}
        if stream.device_ordinal()!=self.ordinal {return Err(CudaRuntimeError::device_mismatch());}
        let address=self.probability_address(q,start,positions)?;
        // SAFETY: all eight views are exact, same-device, nonaliasing and retained
        // by the caller. Bounded finite scalar args are checked here and in v2.
        let result=unsafe{augur_cuda_sys::augur_cuda_qwen35_dflash_q8_selector_probability_submit_v2(
            self.ordinal,stream.raw()?,&self.raw,address,1024,start,positions,temperature,pair_scale)};
        CudaRuntimeError::from_result(OP,result)
    }
    /// Per-position proposal calibration for rows1-7; the exact q is retained for rejection sampling.
    /// Row1 has no draft predecessor, so its pair scale must be 1.
    pub fn submit_probability_positions(&self,stream:&OwnedCudaStream,q:CudaBufferView,start:u64,positions:u32,temperatures:[f32;7],pair_scales:[f32;7])->Result<(),CudaRuntimeError> {
        if !valid_probability_positions(positions,&temperatures,&pair_scales) {return Err(error(CudaRuntimeErrorKind::InvalidLayout));}
        if stream.device_ordinal()!=self.ordinal {return Err(CudaRuntimeError::device_mismatch());}
        let address=self.probability_address(q,start,positions)?;
        // SAFETY: as v2; the two seven-float host arrays are read synchronously by the
        // native entry before return (copied into kernel parameters), so stack lifetime suffices.
        let result=unsafe{augur_cuda_sys::augur_cuda_qwen35_dflash_q8_selector_probability_submit_v3(
            self.ordinal,stream.raw()?,&self.raw,address,1024,start,positions,temperatures.as_ptr(),pair_scales.as_ptr())};
        CudaRuntimeError::from_result(OP,result)
    }
    /// Same proposal law and arithmetic as v1/v3, with an explicit run seed.
    pub fn submit_probability_positions_seeded(&self,stream:&OwnedCudaStream,q:CudaBufferView,start:u64,positions:u32,
        temperatures:[f32;7],pair_scales:[f32;7],seed:u64)->Result<(),CudaRuntimeError> {
        let plain=temperatures==[1.0;7] && pair_scales==[1.0;7] && (1..=7).contains(&positions);
        if !plain && !valid_probability_positions(positions,&temperatures,&pair_scales) {return Err(error(CudaRuntimeErrorKind::InvalidLayout));}
        if stream.device_ordinal()!=self.ordinal {return Err(CudaRuntimeError::device_mismatch());}
        let address=self.probability_address(q,start,positions)?;
        // SAFETY: all retained disjoint buffers and seven-element host arrays are validated as v3.
        let result=unsafe{augur_cuda_sys::augur_cuda_qwen35_dflash_q8_selector_probability_submit_v4(
            self.ordinal,stream.raw()?,&self.raw,address,1024,start,positions,temperatures.as_ptr(),pair_scales.as_ptr(),seed)};
        CudaRuntimeError::from_result(OP,result)
    }
    /// Retained q belongs to this exact path. Caller retains owners until completion.
    pub fn submit_probability_prefix(&self,stream:&OwnedCudaStream,q:CudaBufferView,start:u64,positions:u32)->Result<(),CudaRuntimeError> {
        if stream.device_ordinal()!=self.ordinal {return Err(CudaRuntimeError::device_mismatch());}
        let address=self.probability_address(q,start,positions)?;
        // SAFETY: base binding checked seven exact disjoint Q8/F32 views. Above
        // checks the eighth FP64 view, bounds, ordinal and non-aliasing. The
        // native entry repeats validation; submission allocates/copies nothing.
        let result=unsafe{augur_cuda_sys::augur_cuda_qwen35_dflash_q8_selector_probability_submit_v1(
            self.ordinal,stream.raw()?,&self.raw,address,1024,start,positions)};
        CudaRuntimeError::from_result(OP,result)
    }
}
fn valid_probability_calibration(positions:u32,temperature:f32,pair_scale:f32)->bool {
    positions==7 && temperature.is_finite() && (0.5..=1.5).contains(&temperature)
        && pair_scale.is_finite() && (0.0..=2.0).contains(&pair_scale)
}
fn valid_probability_positions(positions:u32,temperatures:&[f32;7],pair_scales:&[f32;7])->bool {
    positions==7 && pair_scales[0]==1.0
        && temperatures.iter().zip(pair_scales).all(|(&t,&s)|valid_probability_calibration(7,t,s))
}
fn valid_transition_scale(scale: f32) -> bool {
    scale.is_finite() && (0.0..=2.0).contains(&scale)
}
fn error(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(OP, kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_runtime::test_buffer_view;
    #[test] fn probability_calibration_validation() {
        assert!(valid_probability_calibration(7,0.875,1.5));
        for n in [0,1,3,8] {assert!(!valid_probability_calibration(n,0.875,1.5));}
        for t in [f32::NAN,f32::INFINITY,0.0,0.499,1.501] {assert!(!valid_probability_calibration(7,t,1.5));}
        for s in [f32::NAN,f32::INFINITY,-0.001,2.001] {assert!(!valid_probability_calibration(7,0.875,s));}
        let t=[0.875,0.875,0.75,0.625,0.625,0.5,0.625];let s=[1.0,1.5,1.5,1.5,1.5,1.5,1.5];
        assert!(valid_probability_positions(7,&t,&s));
        assert!(!valid_probability_positions(3,&t,&s));
        let mut bad=s;bad[0]=1.5;assert!(!valid_probability_positions(7,&t,&bad));
        for i in 0..7 {
            let mut bt=t;bt[i]=f32::NAN;assert!(!valid_probability_positions(7,&bt,&s));
            let mut bs=s;bs[i]=2.01;assert!(!valid_probability_positions(7,&t,&bs));
        }
    }
    fn from(b: [CudaBufferView; 7]) -> CudaQwen35DflashQ8SelectorViewsV1 {
        CudaQwen35DflashQ8SelectorViewsV1 {
            predecessor: b[0],
            successor: b[1],
            candidate_ids: b[2],
            unary_scores: b[3],
            gates: b[4],
            transition_scores: b[5],
            path_ids: b[6],
        }
    }
    fn views() -> [CudaBufferView; 7] {
        core::array::from_fn(|i| {
            test_buffer_view(
                0x1000_0000 * (i as u64 + 1),
                0,
                0,
                QWEN35_DFLASH_Q8_SELECTOR_BYTES[i],
            )
        })
    }
    #[test]
    fn q8_selector_transition_scale_is_bounded_and_finite() {
        for scale in [0.0, 0.75, 1.0, 2.0] {
            assert!(valid_transition_scale(scale));
        }
        for scale in [-0.01, 2.01, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(!valid_transition_scale(scale));
        }
    }
    #[test]
    fn probability_extra_buffer_rejects_alias_and_invalid_ranges() {
        let a=CudaQwen35DflashQ8SelectorArgsV1::bind_for_ordinal(0,from(views()),1).unwrap();
        assert!(a.probability_address(test_buffer_view(0x90000000,0,0,1024),0,1).is_ok());
        for (address,size,start,n) in [(0,1024,0,1),(1,1024,0,1),(u64::MAX-7,1024,0,1),
            (0x90000000,1023,0,1),(0x90000000,1024,u64::MAX,1),(0x90000000,1024,0,0),(0x90000000,1024,0,8)] {
            assert!(a.probability_address(test_buffer_view(address,0,0,size),start,n).is_err());
        }
        for b in views(){assert!(a.probability_address(test_buffer_view(b.device_ptr().raw_value(),0,0,1024),0,1).is_err());}
        assert!(a.probability_address(test_buffer_view(0x90000000,1,0,1024),0,1).is_err());
    }
    #[test]
    fn q8_selector_abi_and_geometry() {
        type Raw = augur_cuda_sys::AugurCudaQwen35DflashQ8SelectorArgsV1;
        assert_eq!(core::mem::size_of::<Raw>(), 144);
        assert_eq!(core::mem::align_of::<Raw>(), 8);
        assert_eq!(core::mem::offset_of!(Raw, predecessor), 32);
        assert_eq!(core::mem::offset_of!(Raw, path_ids), 128);
        let b =
            CudaQwen35DflashQ8SelectorArgsV1::bind_for_ordinal(0, from(views()), 248319).unwrap();
        assert_eq!(b.raw.predecessor_len_bytes, 67543040);
        assert!(
            CudaQwen35DflashQ8SelectorArgsV1::bind_for_ordinal(0, from(views()), 248320).is_err()
        );
        assert!(CudaQwen35DflashQ8SelectorArgsV1::bind_for_ordinal(-1, from(views()), 1).is_err());
        assert!(CudaQwen35DflashQ8SelectorArgsV1::bind_for_ordinal(1, from(views()), 1).is_err());
    }
    #[test]
    fn q8_selector_rejects_extents_offsets_and_all_alias_pairs() {
        for i in 0..7 {
            for delta in [-2isize, 2] {
                let mut b = views();
                b[i] = test_buffer_view(
                    b[i].device_ptr().raw_value(),
                    0,
                    0,
                    (QWEN35_DFLASH_Q8_SELECTOR_BYTES[i] as isize + delta) as usize,
                );
                assert!(CudaQwen35DflashQ8SelectorArgsV1::bind_for_ordinal(0, from(b), 1).is_err());
            }
            for ptr in [0, 1, u64::MAX - 1] {
                let mut b = views();
                b[i] = test_buffer_view(ptr, 0, 0, QWEN35_DFLASH_Q8_SELECTOR_BYTES[i]);
                assert!(CudaQwen35DflashQ8SelectorArgsV1::bind_for_ordinal(0, from(b), 1).is_err());
            }
            for j in 0..i {
                let mut b = views();
                b[i] = test_buffer_view(
                    b[j].device_ptr().raw_value(),
                    0,
                    0,
                    QWEN35_DFLASH_Q8_SELECTOR_BYTES[i],
                );
                assert!(CudaQwen35DflashQ8SelectorArgsV1::bind_for_ordinal(0, from(b), 1).is_err());
            }
        }
        // A full F16 codebook must not be mistaken for packed Q8_0.
        let mut b = views();
        b[0] = test_buffer_view(0x1000_0000, 0, 0, 248320 * 256 * 2);
        assert!(CudaQwen35DflashQ8SelectorArgsV1::bind_for_ordinal(0, from(b), 1).is_err());
    }
    #[test]
    #[ignore = "requires SM70 CUDA; real Q8 packed scales, chain and poison"]
    fn q8_selector_gpu_chain_ties_and_fail_closed() -> Result<(), Box<dyn std::error::Error>> {
        use crate::{
            cuda_device_allocation_snapshot, cuda_host_boundary_snapshot, OwnedCudaBuffer,
        };
        let before = cuda_device_allocation_snapshot();
        let stream = OwnedCudaStream::create(0)?;
        let owners = QWEN35_DFLASH_Q8_SELECTOR_BYTES
            .iter()
            .map(|&n| OwnedCudaBuffer::allocate(0, n))
            .collect::<Result<Vec<_>, _>>()?;
        let v = owners
            .iter()
            .map(|o| o.view(0, o.len_bytes()))
            .collect::<Result<Vec<_>, _>>()?;
        let bound =
            CudaQwen35DflashQ8SelectorArgsV1::bind(&stream, from(v.try_into().unwrap()), 200000)?;
        let mut prev = vec![0u8; QWEN35_DFLASH_Q8_SELECTOR_BYTES[0]];
        let mut next = prev.clone();
        // Signed codes, positive and negative half scales, non-first quant blocks.
        for (id, scale, code) in [(200000, 0x3800u16, 2i8), (7, 0xbc00, -1), (8, 0x3c00, -1)] {
            let o = id * 272 + 4 * 34;
            prev[o..o + 2].copy_from_slice(&scale.to_le_bytes());
            prev[o + 2] = code as u8;
        }
        // prev(7)=+1, prev(8)=-1: first 8 then 7, alternating.
        for (id, code) in [(7, -32i8), (8, 32)] {
            let o = id * 272 + 4 * 34;
            next[o..o + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
            next[o + 2] = code as u8;
        }
        let ids = (0..8).flat_map(|_| 0..16u32).collect::<Vec<_>>();
        let unary = (0..8)
            .flat_map(|_| (0..16).map(|i| 16.0 - i as f32))
            .collect::<Vec<_>>();
        let mut gates = vec![0f32; 2048];
        for row in 0..8 {
            gates[row * 256 + 128] = 1.0;
        }
        let fb = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
        let ib = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
        for (i, bytes) in [
            (0, prev.clone()),
            (1, next.clone()),
            (2, ib(&ids)),
            (3, fb(&unary)),
            (4, fb(&gates)),
        ] {
            owners[i].copy_from_host_sync(&stream, &bytes)?;
        }
        let run = || -> Result<Vec<u32>, Box<dyn std::error::Error>> {
            let a = cuda_device_allocation_snapshot();
            let h = cuda_host_boundary_snapshot();
            bound.submit(&stream)?;
            let d = cuda_host_boundary_snapshot()
                .checked_delta_since(h)
                .ok_or("counter regression")?;
            assert_eq!(
                (
                    d.synchronous_h2d_bytes,
                    d.synchronous_d2h_bytes,
                    d.host_waits
                ),
                (0, 0, 0)
            );
            let mut bytes = [0u8; 32];
            owners[6].copy_to_host_sync(&stream, &mut bytes)?;
            assert_eq!(
                cuda_device_allocation_snapshot().successful_allocations,
                a.successful_allocations
            );
            Ok(bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect())
        };
        assert_eq!(run()?, [200000, 8, 7, 8, 7, 8, 7, 8]);
        // Independent CPU oracle for the conditional stochastic walk. Exercise
        // every prefix length, counter positions, and actual FP64 q readback.
        {
            let q=OwnedCudaBuffer::allocate(0,1024)?;
            let mut soft_gates=gates.clone();for row in 0..8 {soft_gates[row*256+128]=0.25;}
            owners[4].copy_from_host_sync(&stream,&fb(&soft_gates))?;
            // (row temperatures 1..7, row pair scales 1..7, ABI): 1=v1, 2=v2 rows2-4, 3=v3 per row.
            let mut arms:Vec<([f32;7],[f32;7],u8)>=vec![([1.0;7],[1.0;7],1)];
            for (t,s) in [(0.875f32,1.5f32),(0.5,0.0),(1.5,2.0)] {
                arms.push(([1.0,t,t,t,1.0,1.0,1.0],[1.0,s,s,s,1.0,1.0,1.0],2));
            }
            arms.push(([0.875,0.875,0.75,0.625,0.625,0.5,0.625],[1.0,1.5,1.5,1.5,1.5,1.5,1.5],3));
            arms.push(([1.5,0.5,1.0,0.75,1.25,0.5,1.5],[1.0,0.0,2.0,0.5,1.0,1.75,0.25],3));
            for (temps,scales,abi) in arms {
            for seed in [0u64,1,2] {
            for n in [1,3,7] {if n!=7 && abi!=1 {continue;}
            for position in [0u64,512,32760,262000] {
                let host=cuda_host_boundary_snapshot();let alloc=cuda_device_allocation_snapshot();
                if seed != 0 {
                    bound.submit_probability_positions_seeded(&stream,q.view(0,1024)?,position,n,temps,scales,seed)?;
                } else { match abi {
                    1=>bound.submit_probability_prefix(&stream,q.view(0,1024)?,position,n)?,
                    2=>bound.submit_probability_pos234(&stream,q.view(0,1024)?,position,n,temps[1],scales[1])?,
                    _=>bound.submit_probability_positions(&stream,q.view(0,1024)?,position,n,temps,scales)?,
                }}
                let delta=cuda_host_boundary_snapshot().checked_delta_since(host).unwrap();
                assert_eq!((delta.synchronous_h2d_bytes,delta.synchronous_d2h_bytes,delta.host_waits),(0,0,0));
                assert_eq!(cuda_device_allocation_snapshot().successful_allocations,alloc.successful_allocations);
                let mut path=[0u8;32];owners[6].copy_to_host_sync(&stream,&mut path)?;
                let path:Vec<_>=path.chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect();
                let mut raw=[0u8;1024];q.copy_to_host_sync(&stream,&mut raw)?;
                let actual:Vec<_>=raw.chunks_exact(8).map(|b|f64::from_le_bytes(b.try_into().unwrap())).collect();
                assert_eq!(path[0],200000);assert!(actual[..16].iter().all(|v|*v==0.0));
                let mut previous=200000;
                for row in 1..8 {
                    let pred=if previous==200000 || previous==7 {1.0}else if previous==8 {-1.0}else{0.0};
                    let scale=scales[row-1] as f64;let temp=temps[row-1] as f64;
                    let scores:Vec<f64>=(0..16).map(|i|16.0-i as f64+scale*pred*0.25*if i==7{-32.0}else if i==8{32.0}else{0.0}).collect();
                    let mut best=0;for i in 1..16 {if scores[i]>scores[best]{best=i;}}
                    let mut probs=vec![0.0;16];
                    if row<=n as usize {
                        let weights:Vec<_>=scores.iter().map(|x|((x-scores[best])/temp).exp()).collect();let sum: f64=weights.iter().sum();
                        for i in 0..16{probs[i]=weights[i]/sum;}
                        let mut x=(0x445241465450524fu64^seed).wrapping_add((position+row as u64+1).wrapping_mul(0x9e3779b97f4a7c15));
                        x=(x^(x>>30)).wrapping_mul(0xbf58476d1ce4e5b9);x=(x^(x>>27)).wrapping_mul(0x94d049bb133111eb);x^=x>>31;
                        let u=(x>>11) as f64/9007199254740992.0;let mut c=0.0;
                        for i in 0..16 {c+=probs[i];if u<c {best=i;break;}}
                    }else{probs[best]=1.0;}
                    assert_eq!(path[row],best as u32,"seed={seed} abi={abi} n={n} position={position} row={row}");
                    for i in 0..16 {assert!((actual[row*16+i]-probs[i]).abs()<2e-13);}
                    previous=best as u32;
                }
            }}}}
            // Native validation cannot be bypassed by calling the raw ABI.
            for (n,t,s) in [(3,0.875,1.5),(7,f32::NAN,1.0),(7,1.0,f32::INFINITY),(7,0.0,1.0),(7,1.0,2.01)] {
                assert!(bound.submit_probability_pos234(&stream,q.view(0,1024)?,0,n,t,s).is_err());
                let address=bound.probability_address(q.view(0,1024)?,0,7)?;
                // SAFETY: valid retained buffers/stream; deliberately invalid scalar values must fail before launch.
                let result=unsafe{augur_cuda_sys::augur_cuda_qwen35_dflash_q8_selector_probability_submit_v2(0,stream.raw()?,&bound.raw,address,1024,0,n,t,s)};
                assert!(CudaRuntimeError::from_result(OP,result).is_err());
            }
            let good_t=[0.875f32;7];let good_s=[1.0f32,1.5,1.5,1.5,1.5,1.5,1.5];
            for (n,t,s) in [(3,good_t,good_s),(7,good_t,[1.5;7]),(7,[f32::NAN;7],good_s),(7,good_t,[1.0,1.0,1.0,1.0,1.0,1.0,2.01]),(7,[0.875,0.875,0.875,0.875,0.875,0.875,0.49],good_s)] {
                assert!(bound.submit_probability_positions(&stream,q.view(0,1024)?,0,n,t,s).is_err());
                let address=bound.probability_address(q.view(0,1024)?,0,7)?;
                // SAFETY: valid retained buffers/stream and live seven-float arrays; invalid values must fail before launch.
                let result=unsafe{augur_cuda_sys::augur_cuda_qwen35_dflash_q8_selector_probability_submit_v3(0,stream.raw()?,&bound.raw,address,1024,0,n,t.as_ptr(),s.as_ptr())};
                assert!(CudaRuntimeError::from_result(OP,result).is_err());
            }
            let address=bound.probability_address(q.view(0,1024)?,0,7)?;
            // SAFETY: null host arrays must be rejected by the native entry.
            let result=unsafe{augur_cuda_sys::augur_cuda_qwen35_dflash_q8_selector_probability_submit_v3(0,stream.raw()?,&bound.raw,address,1024,0,7,std::ptr::null(),good_s.as_ptr())};
            assert!(CudaRuntimeError::from_result(OP,result).is_err());
            owners[4].copy_from_host_sync(&stream,&fb(&gates))?;
        }

        assert_eq!(run()?, [200000, 8, 7, 8, 7, 8, 7, 8]);
        let mut default_scores = vec![0u8; 128 * 4];
        owners[5].copy_to_host_sync(&stream, &mut default_scores)?;
        for scale in [0.0f32, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0] {
            let alloc = cuda_device_allocation_snapshot();
            let host = cuda_host_boundary_snapshot();
            bound.submit_with_transition_scale(&stream, scale)?;
            let delta = cuda_host_boundary_snapshot()
                .checked_delta_since(host)
                .unwrap();
            assert_eq!(
                (
                    delta.synchronous_h2d_bytes,
                    delta.synchronous_d2h_bytes,
                    delta.host_waits
                ),
                (0, 0, 0)
            );
            let mut path_bytes = [0u8; 32];
            let mut score_bytes = vec![0u8; 128 * 4];
            owners[6].copy_to_host_sync(&stream, &mut path_bytes)?;
            owners[5].copy_to_host_sync(&stream, &mut score_bytes)?;
            let actual_path: Vec<u32> = path_bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let scores: Vec<f32> = score_bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let mut expected_path = vec![200000];
            let mut expected_scores = vec![0.0f32; 128];
            for row in 1..8 {
                let previous = *expected_path.last().unwrap();
                let p = match previous {
                    200000 | 7 => 1.0,
                    8 => -1.0,
                    _ => 0.0,
                };
                let mut best = 0usize;
                for candidate in 0..16 {
                    let next = match candidate {
                        7 => -32.0,
                        8 => 32.0,
                        _ => 0.0,
                    };
                    let score = (p * next) * scale + unary[row * 16 + candidate];
                    expected_scores[row * 16 + candidate] = score;
                    if score > expected_scores[row * 16 + best] {
                        best = candidate;
                    }
                }
                expected_path.push(best as u32);
            }
            assert_eq!(actual_path, expected_path, "scaled path {scale}");
            assert_eq!(scores, expected_scores, "scaled score {scale}");
            if scale == 1.0 {
                assert_eq!(score_bytes, default_scores, "default API must be exact");
            }
            assert_eq!(
                cuda_device_allocation_snapshot().successful_allocations,
                alloc.successful_allocations
            );
        }
        // Independent rows must follow the newly selected predecessor, including
        // when scale changes between calls on the same bound workspace.
        for scales in [[1.0; 3], [0.5; 3], [0.25, 0.65, 0.6], [0.0, 2.0, 0.125]] {
            let alloc = cuda_device_allocation_snapshot();
            let host = cuda_host_boundary_snapshot();
            bound.submit_with_early_position_scales(&stream, scales)?;
            let delta = cuda_host_boundary_snapshot().checked_delta_since(host).ok_or("counter regression")?;
            assert_eq!((delta.synchronous_h2d_bytes, delta.synchronous_d2h_bytes, delta.host_waits), (0, 0, 0));
            let mut path_bytes = [0u8; 32];
            let mut score_bytes = vec![0u8; 128 * 4];
            owners[6].copy_to_host_sync(&stream, &mut path_bytes)?;
            owners[5].copy_to_host_sync(&stream, &mut score_bytes)?;
            let actual_path: Vec<u32> = path_bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
            let actual_scores: Vec<f32> = score_bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            let mut expected_path = vec![200000];
            let mut expected_scores = vec![0.0f32; 128];
            for row in 1..8 {
                let predecessor = match *expected_path.last().unwrap() { 200000 | 7 => 1.0, 8 => -1.0, _ => 0.0 };
                let scale = if row <= 3 { scales[row - 1] } else { 1.0 };
                let mut best = 0usize;
                for candidate in 0..16 {
                    let successor = match candidate { 7 => -32.0, 8 => 32.0, _ => 0.0 };
                    let score = (predecessor * successor) * scale + unary[row * 16 + candidate];
                    expected_scores[row * 16 + candidate] = score;
                    if score > expected_scores[row * 16 + best] { best = candidate; }
                }
                expected_path.push(best as u32);
            }
            assert_eq!(actual_path, expected_path, "position path {scales:?}");
            assert_eq!(actual_scores, expected_scores, "position scores {scales:?}");
            if scales == [1.0; 3] { assert_eq!(score_bytes, default_scores); }
            if scales == [0.5; 3] {
                bound.submit_with_early_transition_scale(&stream, 0.5)?;
                let mut uniform_path = [0u8; 32];
                let mut uniform_scores = vec![0u8; 128 * 4];
                owners[6].copy_to_host_sync(&stream, &mut uniform_path)?;
                owners[5].copy_to_host_sync(&stream, &mut uniform_scores)?;
                assert_eq!(uniform_path, path_bytes);
                assert_eq!(uniform_scores, score_bytes);
            }
            let after = cuda_device_allocation_snapshot();
            assert_eq!(after.successful_allocations, alloc.successful_allocations);
            assert_eq!(after.successful_frees, alloc.successful_frees);
        }
        for position in 0..3 {
            for invalid_scale in [-0.01, 2.01, f32::NAN, f32::INFINITY] {
                let mut scales = [1.0; 3];
                scales[position] = invalid_scale;
                assert!(bound.submit_with_early_position_scales(&stream, scales).is_err());
                // SAFETY: bad scales must be rejected before pointer access or enqueue.
                let invalid = unsafe {
                    augur_cuda_sys::augur_cuda_qwen35_dflash_q8_selector_position_scaled_submit_v1(
                        0, stream.raw()?, &bound.raw, scales[0], scales[1], scales[2],
                    )
                };
                assert_eq!(invalid.status, augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
            }
        }
        for scale in [-0.01, 2.01, f32::NAN, f32::INFINITY] {
            assert!(bound.submit_with_transition_scale(&stream, scale).is_err());
            // SAFETY: invalid scale must reject before pointer access/enqueue.
            let invalid = unsafe {
                augur_cuda_sys::augur_cuda_qwen35_dflash_q8_selector_scaled_submit_v1(
                    0,
                    stream.raw()?,
                    &bound.raw,
                    scale,
                )
            };
            assert_eq!(invalid.status, augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
        }

        for kind in 0..8 {
            let mut raw = bound.raw;
            match kind {
                0 => raw.rank = 255,
                1 => raw.anchor_token = 248320,
                2 => raw.path_ids = raw.gates,
                3 => raw.predecessor_len_bytes -= 2,
                4 => raw.predecessor = u64::MAX - 1,
                5 => raw.reserved = 1,
                6 => raw.struct_size = 176,
                _ => raw.rows = 7,
            }
            // SAFETY: intentionally malformed descriptors must be rejected before enqueue.
            let result = unsafe {
                augur_cuda_sys::augur_cuda_qwen35_dflash_q8_selector_submit_v1(0, stream.raw()?, &raw)
            };
            assert_eq!(result.status, augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
        }
        for kind in 0..7 {
            let mut bad_ids = ids.clone();
            let mut bad_unary = unary.clone();
            let mut bad_gates = gates.clone();
            let mut bad_prev = prev.clone();
            let mut bad_next = next.clone();
            match kind {
                0 => bad_ids[127] = u32::MAX,
                1 => bad_ids[127] = bad_ids[126],
                2 => bad_unary[127] = f32::NAN,
                3 => bad_gates[7 * 256] = f32::INFINITY,
                4 => bad_prev[200000 * 272..200000 * 272 + 2]
                    .copy_from_slice(&0x7c00u16.to_le_bytes()),
                5 => bad_next[15 * 272..15 * 272 + 2].copy_from_slice(&0x7e00u16.to_le_bytes()),
                _ => bad_unary[127] = 1000.0,
            }
            for (i, b) in [
                (0, bad_prev),
                (1, bad_next),
                (2, ib(&bad_ids)),
                (3, fb(&bad_unary)),
                (4, fb(&bad_gates)),
            ] {
                owners[i].copy_from_host_sync(&stream, &b)?;
            }
            assert_eq!(run()?, [u32::MAX; 8]);
        }
        owners[0].copy_from_host_sync(&stream, &prev)?;
        owners[1].copy_from_host_sync(&stream, &next)?;
        owners[2].copy_from_host_sync(&stream, &ib(&ids))?;
        owners[3].copy_from_host_sync(&stream, &fb(&vec![0.0; 128]))?;
        owners[4].copy_from_host_sync(&stream, &fb(&vec![0.0; 2048]))?;
        assert_eq!(run()?, [200000, 0, 0, 0, 0, 0, 0, 0]);
        drop(owners);
        let after = cuda_device_allocation_snapshot();
        assert_eq!(
            (after.live_bytes, after.live_allocations),
            (before.live_bytes, before.live_allocations)
        );
        Ok(())
    }
    #[test]
    fn bf16_selector_sizes_stay_separate_from_q8() {
        let mut sizes=QWEN35_DFLASH_Q8_SELECTOR_BYTES;sizes[0]=248320*256*2;sizes[1]=sizes[0];
        let views_bf=core::array::from_fn(|i|test_buffer_view(0x1000_0000*(i as u64+1),0,0,sizes[i]));
        assert!(CudaQwen35DflashQ8SelectorArgsV1::bind_with_sizes(0,from(views_bf),1,sizes).is_ok());
        assert!(CudaQwen35DflashQ8SelectorArgsV1::bind_for_ordinal(0,from(views_bf),1).is_err());
        assert!(CudaQwen35DflashQ8SelectorArgsV1::bind_with_sizes(0,from(views()),1,sizes).is_err());
    }
    #[test]
    #[ignore="exclusive SM70: original BF16 selector, chain, scores and guards"]
    fn bf16_selector_gpu_scores_chain_and_cleanup()->Result<(),Box<dyn std::error::Error>> {
        use crate::{OwnedCudaBuffer,cuda_device_allocation_snapshot};
        let before=cuda_device_allocation_snapshot();
        {
            let stream=OwnedCudaStream::create(0)?;let foreign=OwnedCudaStream::create(0)?;
            let mut sizes=QWEN35_DFLASH_Q8_SELECTOR_BYTES;sizes[0]=248320*256*2;sizes[1]=sizes[0];
            let owners=sizes.iter().map(|n|OwnedCudaBuffer::allocate(0,*n+512)).collect::<Result<Vec<_>,_>>()?;
            let mut payloads:Vec<Vec<u8>>=sizes.iter().map(|n|vec![0u8;*n]).collect();
            let pval=|id:usize,r:usize|(((id*3+r*7)%17) as i32-8) as f32/16.;
            let sval=|id:usize,r:usize|(((id*5+r*3)%19) as i32-9) as f32/8.;
            for id in 0..128 {for r in 0..256 {let at=(id*256+r)*2;
                payloads[0][at..at+2].copy_from_slice(&((pval(id,r).to_bits()>>16) as u16).to_le_bytes());
                payloads[1][at..at+2].copy_from_slice(&((sval(id,r).to_bits()>>16) as u16).to_le_bytes());
            }}
            for i in 0..128 {payloads[2][i*4..i*4+4].copy_from_slice(&(i as u32).to_le_bytes());
                payloads[3][i*4..i*4+4].copy_from_slice(&((16-i%16) as f32/128.).to_le_bytes());}
            for i in 0..2048 {payloads[4][i*4..i*4+4].copy_from_slice(&(((i%7) as i32-3) as f32/16.).to_le_bytes());}
            for (o,bytes) in owners.iter().zip(&payloads) {let mut guarded=vec![0xa5;o.len_bytes()];guarded[256..256+bytes.len()].copy_from_slice(bytes);o.copy_from_host_sync(&stream,&guarded)?;}
            let v:Vec<_>=owners.iter().zip(sizes).map(|(o,n)|o.view(256,n)).collect::<Result<_,_>>()?;
            let bound=CudaQwen35DflashBf16CodebooksArgsV1::bind(&stream,from(v.try_into().unwrap()),1)?;
            assert!(bound.submit(&foreign).is_err());let hot=cuda_device_allocation_snapshot();
            for _ in 0..3 {bound.submit(&stream)?;}stream.synchronize()?;assert_eq!(cuda_device_allocation_snapshot(),hot);
            let mut expected=vec![1u32];let mut expected_scores=vec![0f32;128];
            for row in 1..8 {let prev=*expected.last().unwrap() as usize;let mut best=0;
                for k in 0..16 {let id=row*16+k;let mut sums=[0f32;16];
                    for lane in 0..16 {for r in (lane..256).step_by(16) {let g=(((row*256+r)%7) as i32-3) as f32/16.;sums[lane]+=(pval(prev,r)*g)*sval(id,r);}}
                    for step in [8,4,2,1] {for lane in 0..step{sums[lane]+=sums[lane+step];}}
                    expected_scores[row*16+k]=sums[0]+(16-k) as f32/128.;
                    if expected_scores[row*16+k]>expected_scores[row*16+best]{best=k;}
                }expected.push((row*16+best) as u32);
            }
            for slot in [5,6] {let mut bytes=vec![0u8;owners[slot].len_bytes()];owners[slot].copy_to_host_sync(&stream,&mut bytes)?;
                assert!(bytes[..256].iter().chain(bytes[256+sizes[slot]..].iter()).all(|b|*b==0xa5));
                if slot==5 {let actual:Vec<f32>=bytes[256..256+sizes[slot]].chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();assert_eq!(actual,expected_scores);}
                else {let actual:Vec<u32>=bytes[256..256+sizes[slot]].chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect();assert_eq!(actual,expected);}
            }
        }
        let after=cuda_device_allocation_snapshot();assert_eq!(after.live_bytes,before.live_bytes);assert_eq!(after.live_allocations,before.live_allocations);
        Ok(())
    }

}

#[cfg(test)]
#[path="cuda_qwen35_dflash_q8_selector_phase_tests.rs"]
mod phase_tests;
