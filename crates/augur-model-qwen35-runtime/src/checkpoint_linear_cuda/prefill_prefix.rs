//! Optional prefill-only QKV/Z batch producer followed by canonical GDN rows.
//! No externally callable skip-prefix operation and no pointer readiness cache.
use super::*;
use augur_backend::cuda_qwen35_prefill_cublas::{
    CudaQwen35PrefillCublasPlanV1, CudaQwen35QpnRetainedProjectionV1,
};
const H: usize = 5120;
const Q: usize = 10240;
const Z: usize = 6144;
const B: usize = 8;
/// Controlled allocations only; CUDA/cuBLAS context costs need a separate margin.
pub(crate) const WORKSPACE_BYTES: u64 =
    (Q * B * 16 * 8 * 2 + H * 2 * B * 2 + B * (H + Q + Z) * 4) as u64;
pub(crate) struct Workspace {
    stream: Rc<OwnedCudaStream>,
    plan: Rc<CudaQwen35PrefillCublasPlanV1>,
    normalized: Buffer,
    qkv: Buffer,
    z: Buffer,
}
impl Workspace {
    pub(crate) fn new(stream: Rc<OwnedCudaStream>) -> Result<Self> {
        Ok(Self {
            plan: Rc::new(CudaQwen35PrefillCublasPlanV1::new(
                &stream,
                Q * B * 16 * 8,
                H * 2,
                B,
            )?),
            normalized: empty(&stream, B * H)?,
            qkv: empty(&stream, B * Q)?,
            z: empty(&stream, B * Z)?,
            stream,
        })
    }
}
pub(crate) struct Program {
    workspace: Rc<Workspace>,
    input: Buffer,
    _norm_weight: Buffer,
    norms: Vec<CudaRmsnormF32ArgsV1>,
    pairs: Vec<[CudaQwen35QpnRetainedProjectionV1; 2]>,
    scalar_input: Buffer,
    scalar_normalized: Buffer,
    scalar_qkv: Buffer,
    scalar_z: Buffer,
}
fn row(buffer: &OwnedCudaBuffer, at: usize, cols: usize) -> Result<CudaF32View> {
    Ok(CudaF32View::new(
        buffer.view(at * cols * 4, cols * 4)?,
        PhysicalLayoutId::DEVICE_CONTIGUOUS_F32_CUDA_V1,
        cols,
    )?)
}
fn copy(
    stream: &OwnedCudaStream,
    dst: &OwnedCudaBuffer,
    dst_row: usize,
    src: &OwnedCudaBuffer,
    src_row: usize,
    cols: usize,
) -> Result<()> {
    // SAFETY: Program and enclosing Batch retain every exact owner on this
    // one serial stream, and drain all exits before releasing these ranges.
    unsafe {
        dst.copy_from_device_async_unchecked(
            stream,
            dst_row * cols * 4,
            src.view(src_row * cols * 4, cols * 4)?,
        )?;
    }
    Ok(())
}
impl Program {
    pub(crate) fn bind(
        layer: &CheckpointLinearCudaLayer,
        workspace: Rc<Workspace>,
        input: Buffer,
        epsilon: f32,
    ) -> Result<Self> {
        if layer.poisoned
            || layer.input_norm_qpn.is_none()
            || !Rc::ptr_eq(&layer.stream, &workspace.stream)
        {
            return Err("batched GDN prefix requires the same fused profile8 owner".into());
        }
        let [Operation::Norm(_), Operation::Projection(qkv), Operation::Projection(z), ..] =
            layer.operations.as_slice()
        else {
            return Err("GDN prefix inventory changed".into());
        };
        let source = [qkv.retained_qpn()?, z.retained_qpn()?];
        let weight = Rc::clone(&layer.buffers["input_norm_weight"]);
        let mut norms = Vec::with_capacity(B);
        for i in 0..B {
            norms.push(
                CudaRmsnormF32ArgsV1::bind(
                    &workspace.stream,
                    row(&input, i, H)?,
                    row(&weight, 0, H)?,
                    row(&workspace.normalized, i, H)?,
                    epsilon,
                )?
                .with_exact_warp_tree5120()?,
            );
        }
        let mut pairs = Vec::with_capacity(B);
        for width in 1..=B {
            pairs.push([
                source[0].rebind_shared_weight(
                    Rc::clone(&workspace.plan),
                    Rc::clone(&workspace.stream),
                    Rc::clone(&workspace.normalized),
                    0,
                    Rc::clone(&workspace.qkv),
                    0,
                    width,
                )?,
                source[1].rebind_shared_weight(
                    Rc::clone(&workspace.plan),
                    Rc::clone(&workspace.stream),
                    Rc::clone(&workspace.normalized),
                    0,
                    Rc::clone(&workspace.z),
                    0,
                    width,
                )?,
            ]);
        }
        Ok(Self {
            workspace,
            input,
            _norm_weight: weight,
            norms,
            pairs,
            scalar_input: Rc::clone(&layer.buffers["input"]),
            scalar_normalized: Rc::clone(&layer.buffers["normalized"]),
            scalar_qkv: Rc::clone(&layer.buffers["qkv"]),
            scalar_z: Rc::clone(&layer.buffers["z"]),
        })
    }
    /// Produce and consume one entire layer prefix transaction. On any partial
    /// error the layer AND enclosing target remain poisoned until reset; no
    /// independently callable consumer can accept stale or foreign row data.
    pub(crate) fn submit_rows(
        &self,
        layer: &mut CheckpointLinearCudaLayer,
        width: usize,
        normalized: &OwnedCudaBuffer,
        residual: &OwnedCudaBuffer,
    ) -> Result<()> {
        if !(1..=B).contains(&width)
            || layer.poisoned
            || !Rc::ptr_eq(&layer.stream, &self.workspace.stream)
            || !Rc::ptr_eq(&layer.buffers["input"], &self.scalar_input)
            || !Rc::ptr_eq(&layer.buffers["normalized"], &self.scalar_normalized)
            || !Rc::ptr_eq(&layer.buffers["qkv"], &self.scalar_qkv)
            || !Rc::ptr_eq(&layer.buffers["z"], &self.scalar_z)
        {
            return Err("stale/foreign GDN prefill prefix or invalid width".into());
        }
        let next = layer
            .tokens
            .checked_add(width)
            .ok_or("token counter overflow")?;
        let tail = layer.ffn_batch_binding()?.0.tail_operations();
        layer.poisoned = true;
        for norm in &self.norms[..width] {
            norm.submit_for_plan(&layer.stream)?;
        }
        self.pairs[width - 1][0].submit_shared_input_pair(&self.pairs[width - 1][1])?;
        for at in 0..width {
            copy(&layer.stream, &self.scalar_input, 0, &self.input, at, H)?;
            copy(
                &layer.stream,
                &self.scalar_normalized,
                0,
                &self.workspace.normalized,
                at,
                H,
            )?;
            copy(
                &layer.stream,
                &self.scalar_qkv,
                0,
                &self.workspace.qkv,
                at,
                Q,
            )?;
            copy(&layer.stream, &self.scalar_z, 0, &self.workspace.z, at, Z)?;
            for op in &layer.operations[3..layer.operations.len() - tail] {
                op.submit(&layer.stream)?;
            }
            copy(
                &layer.stream,
                normalized,
                at,
                &layer.buffers["post_norm"],
                0,
                H,
            )?;
            copy(
                &layer.stream,
                residual,
                at,
                &layer.buffers["attention_residual"],
                0,
                H,
            )?;
        }
        layer.tokens = next;
        layer.poisoned = false;
        Ok(())
    }
}
impl Drop for Program {
    fn drop(&mut self) {
        let _ = self.workspace.stream.synchronize();
    }
}
