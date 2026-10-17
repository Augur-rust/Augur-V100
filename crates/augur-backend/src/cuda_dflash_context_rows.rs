//! Independent-row context/draft launches; retain owners through completion.
//! No precision change, submit-time allocation, transfer, or synchronization.
//! Optional history panels borrow explicitly bounded caller-owned scratch.
use crate::{
    CudaBufferView, CudaF32View, CudaQ8_0WeightViewV1, CudaRuntimeError, CudaRuntimeErrorKind,
    OwnedCudaStream,
};
use augur_cuda_sys::{AugurCudaDflashRopeF32ArgsV1, AugurCudaQ8_0F32MatvecArgsV1, AugurCudaRmsnormF32ArgsV1};

const OP: &str = "cuda_dflash_context_rows_v1";
fn error(kind: CudaRuntimeErrorKind) -> CudaRuntimeError {
    CudaRuntimeError::validation(OP, kind)
}
fn extent(rows: usize, width: usize, max_rows: usize) -> Result<usize, CudaRuntimeError> {
    if !(1..=max_rows).contains(&rows) || !(1..=65536).contains(&width) {
        return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
    }
    rows.checked_mul(width)
        .ok_or_else(|| error(CudaRuntimeErrorKind::RangeOverflow))
}
fn rope_extent(
    tokens: usize,
    heads: usize,
    position: usize,
    theta: f32,
) -> Result<usize, CudaRuntimeError> {
    if !(1..=2048).contains(&tokens)
        || !(1..=32).contains(&heads)
        || position.checked_add(tokens).is_none_or(|end| end > 262144)
        || !theta.is_finite()
        || theta <= 1.0
    {
        return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
    }
    Ok(tokens * heads * 128)
}
fn address(v: CudaBufferView) -> Result<u64, CudaRuntimeError> {
    v.device_ptr()
        .raw_value()
        .checked_add(v.offset_bytes() as u64)
        .ok_or_else(|| error(CudaRuntimeErrorKind::RangeOverflow))
}
fn exact_f32(view: CudaF32View) -> Result<(), CudaRuntimeError> {
    if view.buffer_view().len_bytes() != view.element_count() * 4 {
        return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
    }
    Ok(())
}
fn validate_views(
    ordinal: i32,
    inputs: &[CudaBufferView],
    output: CudaBufferView,
) -> Result<(), CudaRuntimeError> {
    if output.device_ordinal() != ordinal || inputs.iter().any(|v| v.device_ordinal() != ordinal) {
        return Err(error(CudaRuntimeErrorKind::DeviceMismatch));
    }
    for input in inputs {
        if input.overlaps(output, OP)? {
            return Err(error(CudaRuntimeErrorKind::OverlappingDeviceRanges));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Operation {
    Projection(AugurCudaQ8_0F32MatvecArgsV1, u32),
    ProjectionTile4(AugurCudaQ8_0F32MatvecArgsV1, u32),
    ProjectionB8Specialized(AugurCudaQ8_0F32MatvecArgsV1, u32, usize),
    ProjectionHistoryReuse(AugurCudaQ8_0F32MatvecArgsV1, u32),
    ProjectionHistoryPanel(AugurCudaQ8_0F32MatvecArgsV1, u32, u64, u64),
    ProjectionHistoryPanelToken8(AugurCudaQ8_0F32MatvecArgsV1, u32, u64, u64),
    ProjectionHistoryFcSlice(AugurCudaQ8_0F32MatvecArgsV1, u32, u64, u64),
    Norm(AugurCudaRmsnormF32ArgsV1, u32),
    Rope(AugurCudaDflashRopeF32ArgsV1, u32, u32),
    KeyNormRope(AugurCudaRmsnormF32ArgsV1, AugurCudaDflashRopeF32ArgsV1, u32, u32),
}

/// Contiguous token-major rows, sharing only immutable weights. Bound views do
/// not own their allocations; the caller must retain them through stream drain.
#[derive(Clone, Copy, Debug)]
pub struct CudaDflashContextRowsV1 {
    operation: Operation,
    ordinal: i32,
}
impl CudaDflashContextRowsV1 {
    pub fn specialized_b8_supported(tokens: usize, rows: usize, cols: usize) -> bool {
        tokens == 8 && matches!((rows, cols), (4096,5120)|(1024,5120)|(17408,5120)|(5120,17408))
    }

    /// Complete B8 only; same live views, exact stream and frozen FP64 order.
    pub fn projection_b8_specialized(
        stream: &OwnedCudaStream, weight: CudaQ8_0WeightViewV1,
        input: CudaF32View, output: CudaF32View, tokens: usize,
    ) -> Result<Self, CudaRuntimeError> {
        if !Self::specialized_b8_supported(tokens, weight.rows() as usize, weight.cols() as usize) {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let mut bound = Self::projection(stream, weight, input, output, tokens)?;
        let Operation::Projection(raw, n) = bound.operation else { unreachable!() };
        bound.operation = Operation::ProjectionB8Specialized(raw, n, stream.owner_instance_key()?);
        Ok(bound)
    }
    pub fn history_projection_supported(tokens: usize, rows: usize, cols: usize) -> bool {
        matches!(tokens, 512 | 2048) && matches!((rows, cols), (5120, 25600) | (1024, 5120))
    }

    pub fn history_panel_elements(tokens: usize, rows: usize, cols: usize) -> Option<usize> {
        Self::history_projection_supported(tokens, rows, cols)
            .then(|| (if cols == 25600 { 128 } else { 256 }) * cols)
    }

    /// Decode each bounded weight panel once; preserve FP64 dot/reduction order.
    /// Scratch is mutable and must remain private to this stream until drained.
    pub fn projection_history_panel(
        stream: &OwnedCudaStream,
        weight: CudaQ8_0WeightViewV1,
        input: CudaF32View,
        output: CudaF32View,
        scratch: CudaF32View,
        tokens: usize,
    ) -> Result<Self, CudaRuntimeError> {
        let elements = Self::history_panel_elements(tokens, weight.rows() as usize, weight.cols() as usize)
            .ok_or_else(|| error(CudaRuntimeErrorKind::InvalidDimensions))?;
        exact_f32(scratch)?;
        if scratch.element_count() != elements {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let mut bound = Self::projection(stream, weight, input, output, tokens)?;
        validate_views(stream.device_ordinal(),
            &[weight.buffer_view(), input.buffer_view(), output.buffer_view()], scratch.buffer_view())?;
        let Operation::Projection(raw, n) = bound.operation else { unreachable!() };
        bound.operation = Operation::ProjectionHistoryPanel(raw, n,
            address(scratch.buffer_view())?, scratch.buffer_view().len_bytes() as u64);
        Ok(bound)
    }

    /// Exact token8 retile of bounded decoded panels. Reuse the identical
    /// owner/alias/extent checks; no extra scratch or short/suffix dispatch.
    pub fn projection_history_panel_token8(
        stream: &OwnedCudaStream,
        weight: CudaQ8_0WeightViewV1,
        input: CudaF32View,
        output: CudaF32View,
        scratch: CudaF32View,
        tokens: usize,
    ) -> Result<Self, CudaRuntimeError> {
        let mut bound = Self::projection_history_panel(stream, weight, input, output, scratch, tokens)?;
        let Operation::ProjectionHistoryPanel(raw, n, ptr, bytes) = bound.operation else { unreachable!() };
        bound.operation = Operation::ProjectionHistoryPanelToken8(raw, n, ptr, bytes);
        Ok(bound)
    }

    /// Real capture intersection, including incomplete first/last token tiles.
    /// Only FC 5120x25600; caller retains private scratch and exact slice owners.
    pub fn projection_history_fc_slice(
        stream: &OwnedCudaStream, weight: CudaQ8_0WeightViewV1,
        input: CudaF32View, output: CudaF32View, scratch: CudaF32View, tokens: usize,
    ) -> Result<Self, CudaRuntimeError> {
        if !(1..=512).contains(&tokens) || weight.rows()!=5120 || weight.cols()!=25600 {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        exact_f32(scratch)?;
        if scratch.element_count()!=128*25600 {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let mut bound=Self::projection(stream,weight,input,output,tokens)?;
        validate_views(stream.device_ordinal(),
            &[weight.buffer_view(),input.buffer_view(),output.buffer_view()],scratch.buffer_view())?;
        let Operation::Projection(raw,n)=bound.operation else { unreachable!() };
        bound.operation=Operation::ProjectionHistoryFcSlice(raw,n,
            address(scratch.buffer_view())?,scratch.buffer_view().len_bytes() as u64);
        Ok(bound)
    }

    /// Independent opt-in bulk history path. Never changes short/tail projection().
    pub fn projection_history_reuse(
        stream: &OwnedCudaStream,
        weight: CudaQ8_0WeightViewV1,
        input: CudaF32View,
        output: CudaF32View,
        tokens: usize,
    ) -> Result<Self, CudaRuntimeError> {
        if !Self::history_projection_supported(tokens, weight.rows() as usize, weight.cols() as usize) {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let mut bound = Self::projection(stream, weight, input, output, tokens)?;
        let Operation::Projection(raw, n) = bound.operation else { unreachable!() };
        bound.operation = Operation::ProjectionHistoryReuse(raw, n);
        Ok(bound)
    }

    /// One launch, preserving both original normalized scratch and rotated keys.
    pub fn key_norm_rope(
        stream: &OwnedCudaStream,
        input: CudaF32View,
        weight: CudaF32View,
        normalized: CudaF32View,
        output: CudaF32View,
        tokens: usize,
        heads: usize,
        position: usize,
        theta: f32,
        epsilon: f32,
    ) -> Result<Self, CudaRuntimeError> {
        if heads != 8 || weight.element_count() != 128 {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        rope_extent(tokens, heads, position, theta)?;
        let n = Self::norm(stream, input, weight, normalized, tokens * heads, epsilon)?;
        let r = Self::rope(stream, normalized, output, tokens, heads, position, theta)?;
        validate_views(stream.device_ordinal(), &[input.buffer_view(), weight.buffer_view()], output.buffer_view())?;
        let (Operation::Norm(norm, _), Operation::Rope(rope, t, h)) = (n.operation, r.operation) else { unreachable!() };
        Ok(Self { operation: Operation::KeyNormRope(norm, rope, t, h), ordinal: stream.device_ordinal() })
    }

    /// Separate experimental weight-reuse path, not selected by projection().
    pub fn projection_tile4(
        stream: &OwnedCudaStream,
        weight: CudaQ8_0WeightViewV1,
        input: CudaF32View,
        output: CudaF32View,
        tokens: usize,
    ) -> Result<Self, CudaRuntimeError> {
        let mut bound = Self::projection(stream, weight, input, output, tokens)?;
        let Operation::Projection(raw, n) = bound.operation else {
            unreachable!()
        };
        bound.operation = Operation::ProjectionTile4(raw, n);
        Ok(bound)
    }

    pub fn projection(
        stream: &OwnedCudaStream,
        weight: CudaQ8_0WeightViewV1,
        input: CudaF32View,
        output: CudaF32View,
        tokens: usize,
    ) -> Result<Self, CudaRuntimeError> {
        exact_f32(input)?;
        exact_f32(output)?;
        if input.element_count() != extent(tokens, weight.cols() as usize, 2048)?
            || output.element_count() != extent(tokens, weight.rows() as usize, 2048)?
        {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let (w, i, o) = (
            weight.buffer_view(),
            input.buffer_view(),
            output.buffer_view(),
        );
        validate_views(stream.device_ordinal(), &[w, i], o)?;
        if w.overlaps(i, OP)? {
            return Err(error(CudaRuntimeErrorKind::OverlappingDeviceRanges));
        }
        let mut raw = AugurCudaQ8_0F32MatvecArgsV1::new();
        raw.weight = address(w)?;
        raw.weight_len_bytes = w.len_bytes() as u64;
        raw.input = address(i)?;
        raw.input_len_bytes = i.len_bytes() as u64;
        raw.output = address(o)?;
        raw.output_len_bytes = o.len_bytes() as u64;
        raw.rows = weight.rows();
        raw.cols = weight.cols();
        Ok(Self {
            operation: Operation::Projection(raw, tokens as u32),
            ordinal: stream.device_ordinal(),
        })
    }
    pub fn norm(
        stream: &OwnedCudaStream,
        input: CudaF32View,
        weight: CudaF32View,
        output: CudaF32View,
        rows: usize,
        epsilon: f32,
    ) -> Result<Self, CudaRuntimeError> {
        let width = weight.element_count();
        exact_f32(input)?;
        exact_f32(weight)?;
        exact_f32(output)?;
        let elements = extent(rows, width, 16384)?;
        if input.element_count() != elements
            || output.element_count() != elements
            || !epsilon.is_finite()
            || epsilon <= 0.0
        {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let (w, i, o) = (
            weight.buffer_view(),
            input.buffer_view(),
            output.buffer_view(),
        );
        validate_views(stream.device_ordinal(), &[w, i], o)?;
        let mut raw = AugurCudaRmsnormF32ArgsV1::new();
        raw.weight = address(w)?;
        raw.weight_len_bytes = w.len_bytes() as u64;
        raw.input = address(i)?;
        raw.input_len_bytes = i.len_bytes() as u64;
        raw.output = address(o)?;
        raw.output_len_bytes = o.len_bytes() as u64;
        raw.elements = width as u32;
        raw.epsilon_bits = epsilon.to_bits();
        Ok(Self {
            operation: Operation::Norm(raw, rows as u32),
            ordinal: stream.device_ordinal(),
        })
    }
    pub fn rope(
        stream: &OwnedCudaStream,
        input: CudaF32View,
        output: CudaF32View,
        tokens: usize,
        heads: usize,
        position: usize,
        theta: f32,
    ) -> Result<Self, CudaRuntimeError> {
        let elements = rope_extent(tokens, heads, position, theta)?;
        exact_f32(input)?;
        exact_f32(output)?;
        if input.element_count() != elements || output.element_count() != elements {
            return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
        }
        let (i, o) = (input.buffer_view(), output.buffer_view());
        validate_views(stream.device_ordinal(), &[i], o)?;
        let mut raw = AugurCudaDflashRopeF32ArgsV1::new();
        raw.input = address(i)?;
        raw.input_len_bytes = i.len_bytes() as u64;
        raw.output = address(o)?;
        raw.output_len_bytes = o.len_bytes() as u64;
        raw.elements = 128;
        raw.dimensions = 128;
        raw.rotary_dimensions = 128;
        raw.position = position as u64;
        raw.theta = theta;
        Ok(Self {
            operation: Operation::Rope(raw, tokens as u32, heads as u32),
            ordinal: stream.device_ordinal(),
        })
    }
    pub fn submit_for_plan(&self, stream: &OwnedCudaStream) -> Result<(), CudaRuntimeError> {
        if self.ordinal != stream.device_ordinal() {
            return Err(error(CudaRuntimeErrorKind::DeviceMismatch));
        }
        if let Operation::ProjectionB8Specialized(_, _, owner) = self.operation {
            if stream.owner_instance_key()? != owner {
                return Err(error(CudaRuntimeErrorKind::InvalidDimensions));
            }
        }
        let handle = stream.raw()?;
        // SAFETY: exact bounded geometry, matching device and disjoint writable
        // ranges were checked at bind. Owner lifetime is the caller's same
        // asynchronous-plan contract as scalar norm/Q8/RoPE. Native rechecks ABI.
        let result = unsafe {
            match &self.operation {
                Operation::Projection(raw, n) => {
                    augur_cuda_sys::augur_cuda_dflash_q8_context_rows_submit_v1(
                        self.ordinal,
                        handle,
                        raw,
                        *n,
                    )
                }
                Operation::ProjectionTile4(raw, n) => {
                    augur_cuda_sys::augur_cuda_dflash_q8_rows_tile4_submit_v1(
                        self.ordinal,
                        handle,
                        raw,
                        *n,
                    )
                }
                Operation::ProjectionB8Specialized(raw, n, _) => {
                    augur_cuda_sys::augur_cuda_dflash_q8_rows_b8_specialized_submit_v1(self.ordinal, handle, raw, *n)
                }
                Operation::ProjectionHistoryReuse(raw, n) => {
                    augur_cuda_sys::augur_cuda_dflash_q8_history_reuse_submit_v1(self.ordinal, handle, raw, *n)
                }
                Operation::ProjectionHistoryPanel(raw, n, scratch, bytes) => {
                    augur_cuda_sys::augur_cuda_dflash_q8_history_panel_submit_v1(self.ordinal, handle, raw, *n, *scratch, *bytes)
                }
                Operation::ProjectionHistoryPanelToken8(raw, n, scratch, bytes) => {
                    augur_cuda_sys::augur_cuda_dflash_q8_history_panel_token8_submit_v1(self.ordinal, handle, raw, *n, *scratch, *bytes)
                }
                Operation::ProjectionHistoryFcSlice(raw, n, scratch, bytes) => {
                    augur_cuda_sys::augur_cuda_dflash_q8_history_fc_slice_submit_v1(self.ordinal, handle, raw, *n, *scratch, *bytes)
                }
                Operation::KeyNormRope(norm, rope, t, h) => {
                    augur_cuda_sys::augur_cuda_dflash_key_norm_rope_context_submit_v1(self.ordinal, handle, norm, rope, *t, *h)
                }
                Operation::Norm(raw, n) => augur_cuda_sys::augur_cuda_dflash_norm_context_rows_submit_v1(
                    self.ordinal,
                    handle,
                    raw,
                    *n,
                ),
                Operation::Rope(raw, n, h) => {
                    augur_cuda_sys::augur_cuda_dflash_rope_context_rows_submit_v1(
                        self.ordinal,
                        handle,
                        raw,
                        *n,
                        *h,
                    )
                }
            }
        };
        CudaRuntimeError::from_result(OP, result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn specialized_b8_geometry_and_null_entry_are_closed() {
        for tokens in [0,1,7,8,9,512,2048] {
            for (rows,cols) in [(4096,5120),(1024,5120),(17408,5120),(5120,17408),(1280,5120),(5120,25600),(1023,5120),(1024,5088)] {
                let valid = tokens==8 && matches!((rows,cols),(4096,5120)|(1024,5120)|(17408,5120)|(5120,17408));
                assert_eq!(CudaDflashContextRowsV1::specialized_b8_supported(tokens,rows,cols),valid);
            }
        }
        // SAFETY: null metadata is rejected before device selection or launch.
        unsafe {
            assert_eq!(augur_cuda_sys::augur_cuda_dflash_q8_rows_b8_specialized_submit_v1(
                0,std::ptr::null_mut(),std::ptr::null(),8).status,augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
        }
    }

    #[test]
    #[ignore = "exclusive SM70; real draft weights and archived target input proxies"]
    fn specialized_b8_native_exact_guards_owner_and_rejections() {
        use crate::{OwnedCudaBuffer,cuda_device_allocation_snapshot};
        use augur_core::PhysicalLayoutId as Layout;
        struct Guard { owner: OwnedCudaBuffer, bytes: usize }
        impl Guard {
            fn new(s:&OwnedCudaStream, bytes:&[u8])->Self {
                let v=Self{owner:OwnedCudaBuffer::allocate(0,bytes.len()+512).unwrap(),bytes:bytes.len()};
                v.set(s,bytes);v
            }
            fn set(&self,s:&OwnedCudaStream,bytes:&[u8]) {
                assert_eq!(bytes.len(),self.bytes);let mut all=vec![0xa5;self.bytes+512];
                all[256..256+self.bytes].copy_from_slice(bytes);self.owner.copy_from_host_sync(s,&all).unwrap();
            }
            fn read(&self,s:&OwnedCudaStream)->Vec<u8> {
                let mut all=vec![0;self.bytes+512];self.owner.copy_to_host_sync(s,&mut all).unwrap();
                assert!(all[..256].iter().chain(all[256+self.bytes..].iter()).all(|b|*b==0xa5));
                all[256..256+self.bytes].to_vec()
            }
            fn f32(&self)->CudaF32View {
                CudaF32View::new(self.owner.view(256,self.bytes).unwrap(),Layout::DEVICE_CONTIGUOUS_F32_CUDA_V1,self.bytes/4).unwrap()
            }
        }
        let weights=std::path::PathBuf::from(std::env::var_os("AUGUR_QWEN35_DRAFT_B8_WEIGHT_ROOT").expect("draft weights"));
        let inputs=std::path::PathBuf::from(std::env::var_os("AUGUR_QWEN35_DRAFT_B8_INPUT_ROOT").expect("input proxies"));
        let before=cuda_device_allocation_snapshot();let mut cases=0;let mut rejections=0;
        {
            let stream=OwnedCudaStream::create(0).unwrap();let foreign=OwnedCudaStream::create(0).unwrap();
            for (name,rows,cols) in [("attn_q",4096,5120),("attn_k",1024,5120),("ffn_gate",17408,5120),("ffn_down",5120,17408)] {
                let weight=std::fs::read(weights.join(format!("blk.0.{name}.weight.bin"))).unwrap();
                assert_eq!(weight.len(),rows*cols/32*34);
                let input_name=if cols==5120 {"layer0-attn_norm-full.f32le.bin"}else{"layer0-ffn_activation-full.f32le.bin"};
                let proxy=std::fs::read(inputs.join(input_name)).unwrap();let original=proxy[..8*cols*4].to_vec();
                let w=Guard::new(&stream,&weight);let x=Guard::new(&stream,&original);
                let a=Guard::new(&stream,&vec![0xcd;8*rows*4]);let b=Guard::new(&stream,&vec![0xdc;8*rows*4]);
                let wv=CudaQ8_0WeightViewV1::new(w.owner.view(256,w.bytes).unwrap(),Layout::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,rows,cols).unwrap();
                let old=CudaDflashContextRowsV1::projection_tile4(&stream,wv,x.f32(),a.f32(),8).unwrap();
                let new=CudaDflashContextRowsV1::projection_b8_specialized(&stream,wv,x.f32(),b.f32(),8).unwrap();
                assert!(new.submit_for_plan(&foreign).is_err());rejections+=1;
                assert!(CudaDflashContextRowsV1::projection_b8_specialized(&stream,wv,x.f32(),x.f32(),8).is_err());rejections+=1;
                for generation in 0..5 {
                    let mut values=original.clone();
                    for (i,v) in values.chunks_exact_mut(4).enumerate() {
                        let x=f32::from_le_bytes(v.try_into().unwrap());
                        let y=match generation {1=> -x*0.8125+(i%17) as f32*0.001,2=>x*0.0009765625,_=>x};
                        v.copy_from_slice(&y.to_le_bytes());
                    }
                    if generation==3 {for (v,bits) in values.chunks_exact_mut(4).zip([0u32,0x80000000,1,0x80000001,0x7f7fffff,0xff7fffff]){v.copy_from_slice(&bits.to_le_bytes());}}
                    if generation==4 {for (v,bits) in values.chunks_exact_mut(4).zip([0x7fc12345u32,0x7f800000,0xff800000]){v.copy_from_slice(&bits.to_le_bytes());}}
                    x.set(&stream,&values);a.set(&stream,&vec![0xcd;a.bytes]);b.set(&stream,&vec![0xdc;b.bytes]);
                    let snapshot=cuda_device_allocation_snapshot();
                    old.submit_for_plan(&stream).unwrap();new.submit_for_plan(&stream).unwrap();stream.synchronize().unwrap();
                    assert_eq!(cuda_device_allocation_snapshot(),snapshot,"submit allocated");
                    assert_eq!(a.read(&stream),b.read(&stream),"{name} generation{generation}");
                    assert_eq!(x.read(&stream),values);assert_eq!(w.read(&stream),weight);cases+=1;
                }
                let Operation::ProjectionB8Specialized(raw,_,_)=new.operation else {panic!("wrong binding")};
                let poison=vec![0xab;b.bytes];b.set(&stream,&poison);
                for tokens in [0,1,7,9,512,2048] {
                    // SAFETY: valid owned pointers, invalid token/extent metadata.
                    let result=unsafe{augur_cuda_sys::augur_cuda_dflash_q8_rows_b8_specialized_submit_v1(0,stream.raw().unwrap(),&raw,tokens)};
                    assert_eq!(result.status,augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);rejections+=1;
                }
                let mut bads=Vec::new();
                let mut bad=raw;bad.abi_version=0;bads.push(bad);
                let mut bad=raw;bad.struct_size=0;bads.push(bad);
                let mut bad=raw;bad.weight_len_bytes-=1;bads.push(bad);
                let mut bad=raw;bad.input_len_bytes-=4;bads.push(bad);
                let mut bad=raw;bad.output_len_bytes-=4;bads.push(bad);
                let mut bad=raw;bad.output=bad.input;bads.push(bad);
                let mut bad=raw;bad.output+=1;bads.push(bad);
                for bad in bads {
                    let result=unsafe{augur_cuda_sys::augur_cuda_dflash_q8_rows_b8_specialized_submit_v1(0,stream.raw().unwrap(),&bad,8)};
                    assert_eq!(result.status,augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);rejections+=1;
                }
                stream.synchronize().unwrap();assert_eq!(b.read(&stream),poison,"rejected path wrote output");
            }
        }
        let after=cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes,before.live_bytes);
        assert_eq!(after.successful_allocations-before.successful_allocations,16);
        assert_eq!(after.successful_frees-before.successful_frees,16);
        println!("SPECIALIZED_B8_EXACT cases={cases} rejections={rejections} allocations=16 frees=16 live_delta=0");
    }

    #[test]
    fn history_prepare_geometry_and_null_native_entries_are_closed() {
        for tokens in [512, 2048] {
            for (rows, cols) in [(5120, 25600), (1024, 5120)] {
                assert!(CudaDflashContextRowsV1::history_projection_supported(tokens, rows, cols));
                assert_eq!(CudaDflashContextRowsV1::history_panel_elements(tokens, rows, cols),
                    Some(if cols == 25600 { 128 * cols } else { 256 * cols }));
            }
        }
        for (tokens, rows, cols) in [(0,5120,25600),(511,5120,25600),(513,5120,25600),
            (2049,5120,25600),(8,1024,5120),(512,5121,25600),(512,1024,5152)] {
            assert!(!CudaDflashContextRowsV1::history_projection_supported(tokens, rows, cols));
            assert_eq!(CudaDflashContextRowsV1::history_panel_elements(tokens, rows, cols), None);
        }
        // SAFETY: null arguments/stream reject before any CUDA action.
        unsafe {
            assert_eq!(augur_cuda_sys::augur_cuda_dflash_q8_history_fc_slice_submit_v1(
                0,std::ptr::null_mut(),std::ptr::null(),227,0,0).status,
                augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
            assert_eq!(augur_cuda_sys::augur_cuda_dflash_q8_history_panel_token8_submit_v1(
                0, std::ptr::null_mut(), std::ptr::null(), 512, 0, 0).status,
                augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
            assert_eq!(augur_cuda_sys::augur_cuda_dflash_q8_history_panel_submit_v1(
                0, std::ptr::null_mut(), std::ptr::null(), 512, 0, 0).status,
                augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
            assert_eq!(augur_cuda_sys::augur_cuda_dflash_q8_history_reuse_submit_v1(
                0, std::ptr::null_mut(), std::ptr::null(), 512).status,
                augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
            assert_eq!(augur_cuda_sys::augur_cuda_dflash_key_norm_rope_context_submit_v1(
                0, std::ptr::null_mut(), std::ptr::null(), std::ptr::null(), 8, 8).status,
                augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);
        }
    }

    #[test]
    #[ignore = "exclusive SM70; AUGUR_QWEN35_HISTORY_COMPONENT_INPUTS real Q8 weights/proxy activations"]
    fn history_prepare_native_bundle_bits_guards_and_rejections() {
        use crate::{OwnedCudaBuffer, cuda_device_allocation_snapshot};
        use augur_core::PhysicalLayoutId as Layout;
        struct Guard { owner: OwnedCudaBuffer, bytes: usize }
        impl Guard {
            fn new(stream: &OwnedCudaStream, value: &[u8]) -> Self {
                let owner = OwnedCudaBuffer::allocate(0, value.len()+512).unwrap();
                let result = Self { owner, bytes: value.len() };
                result.set(stream, value);
                result
            }
            fn set(&self, stream: &OwnedCudaStream, value: &[u8]) {
                assert_eq!(value.len(), self.bytes);
                let mut all=vec![0xa5;self.bytes+512];
                all[256..256+self.bytes].copy_from_slice(value);
                self.owner.copy_from_host_sync(stream, &all).unwrap();
            }
            fn read(&self, stream: &OwnedCudaStream) -> Vec<u8> {
                let mut all=vec![0;self.bytes+512];
                self.owner.copy_to_host_sync(stream, &mut all).unwrap();
                assert!(all[..256].iter().chain(all[256+self.bytes..].iter()).all(|b|*b==0xa5));
                all[256..256+self.bytes].to_vec()
            }
            fn f32(&self) -> CudaF32View {
                CudaF32View::new(self.owner.view(256,self.bytes).unwrap(),Layout::DEVICE_CONTIGUOUS_F32_CUDA_V1,self.bytes/4).unwrap()
            }
        }
        let root=std::path::PathBuf::from(std::env::var_os("AUGUR_QWEN35_HISTORY_COMPONENT_INPUTS").expect("history fixture root"));
        let before=cuda_device_allocation_snapshot();
        let mut projection_cases=0;let mut norm_cases=0;let mut rejections=0;
        let mut panel_cases=0;let mut panel_rejections=0;
        let mut panel8_cases=0;let mut panel8_rejections=0;
        {
            let stream=OwnedCudaStream::create(0).unwrap();
            for name in ["fc.weight","blk.0.attn_k.weight","blk.0.attn_v.weight","blk.4.attn_k.weight","blk.4.attn_v.weight"] {
                let (rows,cols)=if name=="fc.weight" {(5120,25600)} else {(1024,5120)};
                let weight=std::fs::read(root.join(format!("{name}.bin"))).unwrap();
                assert_eq!(weight.len(),rows*cols/32*34);
                let w=Guard::new(&stream,&weight);
                let wv=CudaQ8_0WeightViewV1::new(w.owner.view(256,w.bytes).unwrap(),Layout::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,rows,cols).unwrap();
                let proxy=std::fs::read(root.join(format!("proxy-history-{cols}.f32le"))).unwrap();
                for tokens in [512,2048] {
                    let mut input=proxy[..tokens*cols*4].to_vec();
                    let x=Guard::new(&stream,&input);
                    let a=Guard::new(&stream,&vec![0xcd;tokens*rows*4]);
                    let b=Guard::new(&stream,&vec![0xcd;tokens*rows*4]);
                    let c=Guard::new(&stream,&vec![0xcd;tokens*rows*4]);
                    let elements=CudaDflashContextRowsV1::history_panel_elements(tokens,rows,cols).unwrap();
                    let scratch=Guard::new(&stream,&vec![0xcd;elements*4]);
                    let old=CudaDflashContextRowsV1::projection_tile4(&stream,wv,x.f32(),a.f32(),tokens).unwrap();
                    let new=CudaDflashContextRowsV1::projection_history_reuse(&stream,wv,x.f32(),b.f32(),tokens).unwrap();
                    let panel=CudaDflashContextRowsV1::projection_history_panel(&stream,wv,x.f32(),c.f32(),scratch.f32(),tokens).unwrap();
                    let panel8=CudaDflashContextRowsV1::projection_history_panel_token8(&stream,wv,x.f32(),c.f32(),scratch.f32(),tokens).unwrap();
                    let alias=CudaF32View::new(x.owner.view(256,elements*4).unwrap(),Layout::DEVICE_CONTIGUOUS_F32_CUDA_V1,elements).unwrap();
                    assert!(CudaDflashContextRowsV1::projection_history_panel(&stream,wv,x.f32(),c.f32(),alias,tokens).is_err());
                    assert!(CudaDflashContextRowsV1::projection_history_panel_token8(&stream,wv,x.f32(),c.f32(),alias,tokens).is_err());
                    let short=CudaF32View::new(scratch.owner.view(256,elements*4-4).unwrap(),Layout::DEVICE_CONTIGUOUS_F32_CUDA_V1,elements-1).unwrap();
                    assert!(CudaDflashContextRowsV1::projection_history_panel(&stream,wv,x.f32(),c.f32(),short,tokens).is_err());
                    assert!(CudaDflashContextRowsV1::projection_history_panel_token8(&stream,wv,x.f32(),c.f32(),short,tokens).is_err());
                    for generation in 0..2 {
                        if generation==1 {for (i,v) in input.chunks_exact_mut(4).enumerate() {
                            let f=f32::from_le_bytes(v.try_into().unwrap());
                            v.copy_from_slice(&(-f*0.8125+((i%17) as f32-8.)*0.001).to_le_bytes());
                        } x.set(&stream,&input);}
                        let started=cuda_device_allocation_snapshot();
                        old.submit_for_plan(&stream).unwrap();new.submit_for_plan(&stream).unwrap();stream.synchronize().unwrap();
                        panel.submit_for_plan(&stream).unwrap();stream.synchronize().unwrap();
                        assert_eq!(a.read(&stream),c.read(&stream),"panel/{name}/{tokens}/{generation}");
                        panel8.submit_for_plan(&stream).unwrap();stream.synchronize().unwrap();
                        let ended=cuda_device_allocation_snapshot();
                        assert_eq!(started.successful_allocations,ended.successful_allocations);
                        assert_eq!(started.successful_frees,ended.successful_frees);
                        assert_eq!(a.read(&stream),b.read(&stream),"{name}/{tokens}/{generation}");
                        assert_eq!(a.read(&stream),c.read(&stream),"panel8/{name}/{tokens}/{generation}");
                        scratch.read(&stream);panel_cases+=1;panel8_cases+=1;
                        assert_eq!(x.read(&stream),input);projection_cases+=1;
                    }
                    let Operation::ProjectionHistoryReuse(raw,_) = new.operation else {unreachable!()};
                    for mutation in 0..12 {
                        let mut bad=raw;let mut count=tokens as u32;
                        match mutation {
                            0=>bad.abi_version+=1,1=>bad.struct_size-=1,2=>bad.weight_len_bytes-=1,
                            3=>bad.input_len_bytes-=4,4=>bad.output_len_bytes+=4,5=>bad.weight+=1,
                            6=>bad.input+=2,7=>bad.output=bad.input,8=>bad.output=bad.weight,
                            9=>bad.rows+=1,10=>count=511,_=>count=2049,
                        }
                        // SAFETY: deliberately malformed metadata; native must reject before launch.
                        let result=unsafe{augur_cuda_sys::augur_cuda_dflash_q8_history_reuse_submit_v1(0,stream.raw().unwrap(),&bad,count)};
                        assert_eq!(result.status,augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);rejections+=1;
                    }
                    stream.synchronize().unwrap();assert_eq!(a.read(&stream),b.read(&stream));
                    let Operation::ProjectionHistoryPanel(raw,_,ptr,len)=panel.operation else {unreachable!()};
                    for mutation in 0..25 {
                        let mut bad=raw;let mut count=tokens as u32;let mut p=ptr;let mut bytes=len;let mut ordinal=0;
                        match mutation {
                            0=>bad.abi_version+=1,1=>bad.struct_size-=1,2=>bad.weight_len_bytes-=1,
                            3=>bad.input_len_bytes-=4,4=>bad.output_len_bytes+=4,5=>bad.weight+=1,
                            6=>bad.input+=2,7=>bad.output=bad.input,8=>bad.output=bad.weight,
                            9=>bad.rows+=1,10=>count=511,11=>count=2049,
                            12=>p=0,13=>p+=2,14=>bytes-=4,15=>bytes+=4,16=>p=bad.input,
                            17=>p=bad.output,18=>p=bad.weight,19=>p=u64::MAX-3,
                            20=>ordinal=1,21=>count=31,22=>bad.cols+=32,
                            23=>bad.input=bad.weight,_=>count=0,
                        }
                        // SAFETY: all malformed descriptors must reject before either write.
                        let result=unsafe{augur_cuda_sys::augur_cuda_dflash_q8_history_panel_submit_v1(
                            ordinal,stream.raw().unwrap(),&bad,count,p,bytes)};
                        assert_eq!(result.status,augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT,"panel mutation {mutation}");
                        panel_rejections+=1;
                        // SAFETY: the token8 entry must reject the identical invalid set.
                        let result=unsafe{augur_cuda_sys::augur_cuda_dflash_q8_history_panel_token8_submit_v1(
                            ordinal,stream.raw().unwrap(),&bad,count,p,bytes)};
                        assert_eq!(result.status,augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT,"panel8 mutation {mutation}");
                        panel8_rejections+=1;
                    }
                    stream.synchronize().unwrap();assert_eq!(a.read(&stream),c.read(&stream));scratch.read(&stream);
                    assert_eq!(x.read(&stream),input);
                }
                assert_eq!(w.read(&stream),weight);
            }
            // Actual global512 capture intersections, not padded/history-relative
            // scheduling. Check every untouched destination suffix after each slice.
            {
                let weight=std::fs::read(root.join("fc.weight.bin")).unwrap();
                let proxy=std::fs::read(root.join("proxy-history-25600.f32le")).unwrap();
                let w=Guard::new(&stream,&weight);
                let wv=CudaQ8_0WeightViewV1::new(w.owner.view(256,w.bytes).unwrap(),Layout::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,5120,25600).unwrap();
                let mut input=proxy[..2048*25600*4].to_vec();
                let x=Guard::new(&stream,&input);
                let oracle=Guard::new(&stream,&vec![0xcd;2048*5120*4]);
                let sliced=Guard::new(&stream,&vec![0xcd;2048*5120*4]);
                let scratch=Guard::new(&stream,&vec![0xcd;128*25600*4]);
                let plans=[vec![512,512,512,512],vec![227,512,512,512,285],
                    vec![1,511,512,512,512],vec![7,505,509,3,512,512],vec![285,512,512,512,227]];
                let mut cases=0;let mut rejected=0;
                for generation in 0..3 {
                    if generation>0 {
                        for (index,bytes) in input.chunks_exact_mut(4).enumerate() {
                            let v=f32::from_le_bytes(bytes.try_into().unwrap());
                            let v=if generation==1 {-v*0.8125+(index%17) as f32*0.001} else {v*0.0009765625};
                            bytes.copy_from_slice(&v.to_le_bytes());
                        }
                        input[..16].copy_from_slice(&[0u32,0x80000000,1,0x80000001].into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>());
                        x.set(&stream,&input);
                    }
                    let full=CudaDflashContextRowsV1::projection_history_panel_token8(&stream,wv,x.f32(),oracle.f32(),scratch.f32(),2048).unwrap();
                    let start=std::time::Instant::now();full.submit_for_plan(&stream).unwrap();stream.synchronize().unwrap();
                    let full_ms=start.elapsed().as_secs_f64()*1000.0;
                    let expected=oracle.read(&stream);
                    for (plan_index,plan) in plans.iter().enumerate() {
                        assert_eq!(plan.iter().sum::<usize>(),2048);
                        sliced.set(&stream,&vec![0xcd;sliced.bytes]);
                        let mut first=0;let mut submit_ms=0.0;
                        for &tokens in plan {
                            let view=|g:&Guard,first:usize,tokens:usize,width:usize|
                                CudaF32View::new(g.owner.view(256+first*width*4,tokens*width*4).unwrap(),Layout::DEVICE_CONTIGUOUS_F32_CUDA_V1,tokens*width).unwrap();
                            let xv=view(&x,first,tokens,25600);let yv=view(&sliced,first,tokens,5120);
                            let op=CudaDflashContextRowsV1::projection_history_fc_slice(&stream,wv,xv,yv,scratch.f32(),tokens).unwrap();
                            let start=std::time::Instant::now();op.submit_for_plan(&stream).unwrap();stream.synchronize().unwrap();
                            submit_ms+=start.elapsed().as_secs_f64()*1000.0;
                            first+=tokens;
                            let actual=sliced.read(&stream);
                            assert_eq!(&actual[..first*5120*4],&expected[..first*5120*4],"slice {generation}/{plan_index}/{first}");
                            assert!(actual[first*5120*4..].iter().all(|&v|v==0xcd),"unwritten FC suffix");
                            if generation==0 && plan_index==1 && first==227 {
                                assert!(CudaDflashContextRowsV1::projection_history_fc_slice(&stream,wv,xv,yv,scratch.f32(),0).is_err());
                                assert!(CudaDflashContextRowsV1::projection_history_fc_slice(&stream,wv,xv,yv,scratch.f32(),513).is_err());
                                assert!(CudaDflashContextRowsV1::projection_history_fc_slice(&stream,wv,xv,xv,scratch.f32(),tokens).is_err());
                                let alias=view(&x,0,128,25600);
                                assert!(CudaDflashContextRowsV1::projection_history_fc_slice(&stream,wv,xv,yv,alias,tokens).is_err());
                                let Operation::ProjectionHistoryFcSlice(raw,n,p,bytes)=op.operation else {unreachable!()};
                                for mutation in 0..14 {
                                    let mut a=raw;let mut count=n;let mut ptr=p;let mut extent=bytes;
                                    match mutation {
                                        0=>count=0,1=>count=513,2=>a.rows=1024,3=>a.cols=5120,
                                        4=>a.abi_version+=1,5=>a.struct_size-=1,6=>a.input_len_bytes-=4,
                                        7=>a.output_len_bytes-=4,8=>a.weight_len_bytes-=2,9=>a.output=a.input,
                                        10=>ptr=a.output,11=>extent-=4,12=>a.input+=2,_=>ptr+=2,
                                    }
                                    // SAFETY: malformed metadata must reject before touching output.
                                    let result=unsafe{augur_cuda_sys::augur_cuda_dflash_q8_history_fc_slice_submit_v1(0,stream.raw().unwrap(),&a,count,ptr,extent)};
                                    assert_eq!(result.status,augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT,"slice mutation {mutation}");rejected+=1;
                                }
                            }
                        }
                        assert_eq!(x.read(&stream),input);scratch.read(&stream);
                        println!("FC_SLICE_TIMING generation={generation} plan={plan_index} full_ms={full_ms:.4} sliced_ms={submit_ms:.4}");
                        cases+=1;
                    }
                }
                assert_eq!(w.read(&stream),weight);assert_eq!((cases,rejected),(15,14));
                println!("HISTORY_FC_SLICE_NATIVE_PASS exact_cases={cases} metadata_rejections={rejected} bound_rejections=4");
            }
            for tokens in [1,7,8,31,512,2048] {
                for position in [0,8477,262144-tokens] {
                    for theta in [1_000_000.0, 10_000_000.0] {
                    for kind in 0..4 {
                        let count=tokens*8*128;
                        let mut input=(0..count).flat_map(|i|(((i*37%251) as f32-125.)/31.).to_le_bytes()).collect::<Vec<_>>();
                        if kind==1 {input.fill(0);input[4..8].copy_from_slice(&0x80000000_u32.to_le_bytes());input[8..12].copy_from_slice(&1_u32.to_le_bytes());}
                        if kind==2 {for v in input.chunks_exact_mut(4){let f=f32::from_le_bytes(v.try_into().unwrap())*1e18;v.copy_from_slice(&f.to_le_bytes());}}
                        if kind==3 {for (i,bits) in [0x7fc01234_u32,0xffc01234,0x7f800000,0xff800000].iter().enumerate(){input[i*4..i*4+4].copy_from_slice(&bits.to_le_bytes());}}
                        let weight=(0..128).flat_map(|i|(((i%19) as f32-9.)*0.03125+1.).to_le_bytes()).collect::<Vec<_>>();
                        let x=Guard::new(&stream,&input);let w=Guard::new(&stream,&weight);
                        let n0=Guard::new(&stream,&vec![0xcd;count*4]);let n1=Guard::new(&stream,&vec![0xcd;count*4]);
                        let y0=Guard::new(&stream,&vec![0xcd;count*4]);let y1=Guard::new(&stream,&vec![0xcd;count*4]);
                        let oldn=CudaDflashContextRowsV1::norm(&stream,x.f32(),w.f32(),n0.f32(),tokens*8,1e-6).unwrap();
                        let oldr=CudaDflashContextRowsV1::rope(&stream,n0.f32(),y0.f32(),tokens,8,position,theta).unwrap();
                        let new=CudaDflashContextRowsV1::key_norm_rope(&stream,x.f32(),w.f32(),n1.f32(),y1.f32(),tokens,8,position,theta,1e-6).unwrap();
                        let started=cuda_device_allocation_snapshot();
                        oldn.submit_for_plan(&stream).unwrap();oldr.submit_for_plan(&stream).unwrap();new.submit_for_plan(&stream).unwrap();stream.synchronize().unwrap();
                        let ended=cuda_device_allocation_snapshot();assert_eq!(started.successful_allocations,ended.successful_allocations);assert_eq!(started.successful_frees,ended.successful_frees);
                        assert_eq!(n0.read(&stream),n1.read(&stream),"norm {tokens}/{position}/{kind}");
                        assert_eq!(y0.read(&stream),y1.read(&stream),"rope {tokens}/{position}/{kind}");
                        assert_eq!(x.read(&stream),input);assert_eq!(w.read(&stream),weight);norm_cases+=1;
                        if tokens==8 && position==8477 && kind==0 && theta==10_000_000.0 {
                            assert!(CudaDflashContextRowsV1::key_norm_rope(&stream,x.f32(),w.f32(),n1.f32(),x.f32(),tokens,8,position,theta,1e-6).is_err());
                            let Operation::KeyNormRope(norm,rope,t,h)=new.operation else {unreachable!()};
                            for mutation in 0..16 {
                                let mut n=norm;let mut r=rope;let mut nt=t;let mut nh=h;
                                match mutation {
                                    0=>n.abi_version+=1,1=>r.struct_size-=1,2=>n.elements=256,
                                    3=>n.epsilon_bits=f32::NAN.to_bits(),4=>r.theta=f32::INFINITY,
                                    5=>r.position=u64::MAX,6=>n.weight_len_bytes-=4,7=>n.input_len_bytes-=4,
                                    8=>n.output_len_bytes-=4,9=>r.input+=4,10=>r.output=n.input,
                                    11=>r.output=n.weight,12=>r.output=n.output,13=>n.output+=2,
                                    14=>nt=2049,_=>nh=32,
                                }
                                // SAFETY: test rejection before either output can be touched.
                                let result=unsafe{augur_cuda_sys::augur_cuda_dflash_key_norm_rope_context_submit_v1(0,stream.raw().unwrap(),&n,&r,nt,nh)};
                                assert_eq!(result.status,augur_cuda_sys::AUGUR_CUDA_STATUS_INVALID_ARGUMENT);rejections+=1;
                            }
                            stream.synchronize().unwrap();assert_eq!(n0.read(&stream),n1.read(&stream));assert_eq!(y0.read(&stream),y1.read(&stream));
                            assert_eq!(x.read(&stream),input);assert_eq!(w.read(&stream),weight);
                        }
                    }
                    }
                }
            }
            stream.synchronize().unwrap();
        }
        let after=cuda_device_allocation_snapshot();
        assert_eq!(after.live_bytes,before.live_bytes);assert_eq!(after.live_allocations,before.live_allocations);
        assert_eq!(after.successful_allocations-before.successful_allocations,after.successful_frees-before.successful_frees);
        assert_eq!((projection_cases,norm_cases,rejections),(20,144,136));
        assert_eq!((panel_cases,panel_rejections),(20,250));
        assert_eq!((panel8_cases,panel8_rejections),(20,250));
        println!("HISTORY_PANEL_TOKEN8_NATIVE_PASS exact_cases={panel8_cases} metadata_rejections={panel8_rejections} bound_rejections=20");
        println!("HISTORY_PANEL_NATIVE_PASS exact_cases={panel_cases} metadata_rejections={panel_rejections} bound_rejections=20");
        println!("HISTORY_PREPARE_NATIVE_PASS projections={projection_cases} norm_rope={norm_cases} metadata_rejections={rejections} live_bytes={} allocs={} frees={}",after.live_bytes,after.successful_allocations-before.successful_allocations,after.successful_frees-before.successful_frees);
    }

    #[test]
    fn context_rows_geometry_is_bounded_and_exact() {
        for tokens in [1, 7, 8, 9, 2048] {
            assert_eq!(extent(tokens, 25600, 2048).unwrap(), tokens * 25600);
            assert_eq!(extent(tokens * 8, 128, 16384).unwrap(), tokens * 8 * 128);
            assert_eq!(
                rope_extent(tokens, 8, 262144 - tokens, 5000000.0).unwrap(),
                tokens * 8 * 128
            );
        }
        assert_eq!(rope_extent(8, 32, 196352, 5000000.0).unwrap(), 8 * 32 * 128);
        for (rows, width) in [
            (0, 128),
            (2049, 128),
            (1, 0),
            (1, 65537),
            (usize::MAX, 128),
            (1, usize::MAX),
        ] {
            assert!(extent(rows, width, 2048).is_err());
        }
    }
    #[test]
    fn context_rope_refuses_overflow_future_positions_and_invalid_head_geometry() {
        for (tokens, heads, pos, theta) in [
            (0, 8, 0, 10000.),
            (2049, 8, 0, 10000.),
            (8, 0, 0, 10000.),
            (8, 33, 0, 10000.),
            (8, 8, 262137, 10000.),
            (8, 8, usize::MAX, 10000.),
            (8, 8, 0, f32::NAN),
            (8, 8, 0, f32::INFINITY),
            (8, 8, 0, 1.),
        ] {
            assert!(rope_extent(tokens, heads, pos, theta).is_err());
        }
    }

    #[test]
    fn context_rows_require_same_device_and_disjoint_output() {
        use crate::cuda_runtime::test_buffer_view;
        let input = test_buffer_view(0x1000, 0, 0, 1024);
        let output = test_buffer_view(0x3000, 0, 0, 1024);
        validate_views(0, &[input], output).unwrap();
        assert!(validate_views(1, &[input], output).is_err());
        assert!(validate_views(0, &[input], test_buffer_view(0x1100, 0, 0, 1024)).is_err());
        let oversized = CudaF32View::new(
            input,
            augur_core::PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
            128,
        )
        .unwrap();
        assert!(exact_f32(oversized).is_err());
    }

    #[test]
    #[ignore = "requires exclusive SM70 GPU; independent-row bitwise component oracle"]
    fn context_rows_match_scalar_sm70_bits() {
        rows_oracle(false);
    }

    #[test]
    #[ignore = "requires exclusive SM70 GPU; four-token Q8 reuse bit oracle"]
    fn q8_tile4_matches_scalar_sm70_bits() {
        rows_oracle(true);
    }

    fn rows_oracle(tile4: bool) {
        use crate::{
            cuda_device_properties, CudaComputeCapability, CudaDflashRopeF32ArgsV1,
            CudaQ8_0F32MatvecArgsV1, CudaRmsnormF32ArgsV1, OwnedCudaBuffer, OwnedCudaEvent,
        };
        use augur_core::PhysicalLayoutId;
        assert_eq!(
            cuda_device_properties(0).unwrap().compute_capability,
            CudaComputeCapability::SM70
        );
        let stream = OwnedCudaStream::create(0).unwrap();
        fn view(b: &OwnedCudaBuffer, start: usize, n: usize) -> CudaF32View {
            CudaF32View::new(
                b.view(start * 4, n * 4).unwrap(),
                PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
                n,
            )
            .unwrap()
        }
        fn upload(stream: &OwnedCudaStream, bytes: &[u8]) -> OwnedCudaBuffer {
            let b = OwnedCudaBuffer::allocate(0, bytes.len()).unwrap();
            b.copy_from_host_sync(stream, bytes).unwrap();
            b
        }
        fn values(n: usize) -> Vec<u8> {
            (0..n)
                .flat_map(|i| (((i * 37 % 251) as f32 - 125.) / 31.).to_le_bytes())
                .collect()
        }
        fn equal(stream: &OwnedCudaStream, a: &OwnedCudaBuffer, b: &OwnedCudaBuffer, label: &str) {
            stream.synchronize().unwrap();
            let mut left = vec![0; a.len_bytes()];
            let mut right = vec![0; b.len_bytes()];
            a.copy_to_host_sync(stream, &mut left).unwrap();
            b.copy_to_host_sync(stream, &mut right).unwrap();
            let mismatches = left
                .chunks_exact(4)
                .zip(right.chunks_exact(4))
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(mismatches, 0, "{label}");
            assert!(left
                .chunks_exact(4)
                .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite()));
            println!(
                "DFLASH_CONTEXT_ROWS_EXACT {label} elements={}",
                left.len() / 4
            );
        }
        // Non-warp-multiple output rows, non-power-of-two reduction lengths,
        // full 2048-token row grid, and actual FC/K-projection dimensions.
        let mut projections = vec![
            (1, 37, 96),
            (7, 37, 96),
            (9, 37, 96),
            (2048, 37, 96),
            (8, 5120, 25600),
            (9, 1024, 5120),
            (8, 1280, 5120),
            (8, 4096, 5120),
            (8, 1024, 5120),
            (8, 5120, 4096),
            (8, 17408, 5120),
            (8, 5120, 17408),
            (8, 256, 5120),
            // Keep neighboring dimensions and token tails on the fallback.
            (8, 4095, 5120),
            (8, 1023, 5120),
            (8, 1024, 5088),
            (7, 1024, 5120),
            (9, 17408, 5120),
        ];
        if tile4 {
            projections.extend([
                (2, 37, 96),
                (3, 37, 96),
                (4, 37, 96),
                (5, 37, 96),
                (512, 5120, 25600),
                (2048, 5120, 25600),
                (512, 1024, 5120),
                (2048, 1024, 5120),
                // History admission is exact: adjacent token counts and
                // dimensions keep the original fallback contract.
                (511, 1024, 5120),
                (513, 1024, 5120),
                (512, 1023, 5120),
                (512, 1024, 5088),
            ]);
        }
        for (tokens, rows, cols) in projections {
            let mut bytes = Vec::with_capacity(rows * (cols / 32) * 34);
            for block in 0..rows * (cols / 32) {
                bytes.extend_from_slice(
                    &(if block % 2 == 0 { 0x3400u16 } else { 0x3c00u16 }).to_le_bytes(),
                );
                bytes.extend((0..32).map(|i| ((i * 17 + block * 7) % 256) as u8));
            }
            let weight = upload(&stream, &bytes);
            let input = upload(&stream, &values(tokens * cols + 1));
            let a = upload(&stream, &values(tokens * rows + 2));
            let b = upload(&stream, &values(tokens * rows + 2));
            let w = CudaQ8_0WeightViewV1::new(
                weight.view(0, bytes.len()).unwrap(),
                PhysicalLayoutId::DEVICE_CONTIGUOUS_Q8_0_CUDA_V1,
                rows,
                cols,
            )
            .unwrap();
            let projection = if tile4 {
                CudaDflashContextRowsV1::projection_tile4
            } else {
                CudaDflashContextRowsV1::projection
            };
            projection(
                &stream,
                w,
                view(&input, 1, tokens * cols),
                view(&a, 1, tokens * rows),
                tokens,
            )
            .unwrap()
            .submit_for_plan(&stream)
            .unwrap();
            for row in 0..tokens {
                CudaQ8_0F32MatvecArgsV1::bind(
                    &stream,
                    w,
                    view(&input, 1 + row * cols, cols),
                    view(&b, 1 + row * rows, rows),
                )
                .unwrap()
                .submit_for_plan(&stream)
                .unwrap();
            }
            equal(
                &stream,
                &a,
                &b,
                &format!("q8:tile4={tile4}:{tokens}x{rows}x{cols}"),
            );
            if tile4 && tokens >= 8 && rows >= 1024 {
                // Warm component timings inform the next experiment only.
                // These are not request wall measurements or promotion gates.
                let independent = CudaDflashContextRowsV1::projection(
                    &stream,
                    w,
                    view(&input, 1, tokens * cols),
                    view(&b, 1, tokens * rows),
                    tokens,
                )
                .unwrap();
                let reused = CudaDflashContextRowsV1::projection_tile4(
                    &stream,
                    w,
                    view(&input, 1, tokens * cols),
                    view(&a, 1, tokens * rows),
                    tokens,
                )
                .unwrap();
                let start = OwnedCudaEvent::create(0).unwrap();
                let stop = OwnedCudaEvent::create(0).unwrap();
                for trial in 0..3 {
                    let order = if trial % 2 == 0 {
                        [("independent", &independent), ("tile4", &reused)]
                    } else {
                        [("tile4", &reused), ("independent", &independent)]
                    };
                    for (policy, operation) in order {
                        operation.submit_for_plan(&stream).unwrap();
                        stream.synchronize().unwrap();
                        start.record(&stream).unwrap();
                        for _ in 0..3 {
                            operation.submit_for_plan(&stream).unwrap();
                        }
                        stop.record(&stream).unwrap();
                        stop.synchronize().unwrap();
                        let ms = start.elapsed_ms(&stop).unwrap() / 3.0;
                        assert!(ms.is_finite() && ms > 0.0);
                        println!("DFLASH_Q8_COMPONENT_TIMING_ONLY tokens={tokens} rows={rows} cols={cols} trial={trial} policy={policy} mean_cuda_ms={ms:.6}");
                    }
                }
                equal(&stream, &a, &b, "post_timing_bit_and_canary_check");
            }
            assert!(projection(
                &stream,
                w,
                view(&input, 1, tokens * cols),
                view(&a, 1, tokens * rows),
                tokens + 1
            )
            .is_err());
        }
        for (rows, width) in [
            (1, 128),
            (9, 128),
            (2048, 5120),
            (16384, 128),
            (8, 5120),
            (256, 128),
        ] {
            let input = upload(&stream, &values(rows * width + 1));
            let weight = upload(&stream, &values(width));
            let a = upload(&stream, &values(rows * width + 2));
            let b = upload(&stream, &values(rows * width + 2));
            CudaDflashContextRowsV1::norm(
                &stream,
                view(&input, 1, rows * width),
                view(&weight, 0, width),
                view(&a, 1, rows * width),
                rows,
                1e-6,
            )
            .unwrap()
            .submit_for_plan(&stream)
            .unwrap();
            for row in 0..rows {
                CudaRmsnormF32ArgsV1::bind(
                    &stream,
                    view(&input, 1 + row * width, width),
                    view(&weight, 0, width),
                    view(&b, 1 + row * width, width),
                    1e-6,
                )
                .unwrap()
                .submit_for_plan(&stream)
                .unwrap();
            }
            equal(&stream, &a, &b, &format!("norm:{rows}x{width}"));
        }
        for (tokens, heads, position) in [
            (1, 1, 0),
            (9, 8, 32765),
            (8, 8, 196352),
            (2048, 8, 260096),
            (8, 32, 196352),
            (8, 32, 262136),
        ] {
            let count = tokens * heads * 128;
            let input = upload(&stream, &values(count + 1));
            let a = upload(&stream, &values(count + 2));
            let b = upload(&stream, &values(count + 2));
            CudaDflashContextRowsV1::rope(
                &stream,
                view(&input, 1, count),
                view(&a, 1, count),
                tokens,
                heads,
                position,
                5000000.,
            )
            .unwrap()
            .submit_for_plan(&stream)
            .unwrap();
            for row in 0..tokens {
                for head in 0..heads {
                    let start = 1 + (row * heads + head) * 128;
                    CudaDflashRopeF32ArgsV1::bind(
                        &stream,
                        view(&input, start, 128),
                        view(&b, start, 128),
                        128,
                        128,
                        position + row,
                        5000000.,
                    )
                    .unwrap()
                    .submit_for_plan(&stream)
                    .unwrap();
                }
            }
            equal(
                &stream,
                &a,
                &b,
                &format!("rope:{tokens}x{heads}@{position}"),
            );
        }
        stream.synchronize().unwrap();
    }
}
