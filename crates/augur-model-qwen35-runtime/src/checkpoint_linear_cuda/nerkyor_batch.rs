//! Original-source B8 GDN: batched NVFP4 contractions and F32 recurrence.
//! Prefix rollback restores history/state, then replays recorded real operands.
use super::*;
use crate::checkpoint_projection_cuda::{
    nerkyor_gate::NerkyorBf16GatePair, CheckpointExactBatchProjection,
};
use crate::checkpoint_target_cuda::nerkyor_dflash_research::batched::{
    copy, norms, slice, Ffn, Workspace, H, WIDTH,
};
use augur_backend::cuda_qwen35_prefill_cublas::{
    CudaQwen35CheckpointPrefillLinearChunkV1, CudaQwen35PrefillLinearChunkV1,
    CudaQwen35PrefillLinearChunkViewsV1,
};
pub(crate) struct Program {
    w: Rc<Workspace>,
    _input: Buffer,
    _output: Buffer,
    _vectors: Vec<Buffer>,
    state: [Buffer; 2],
    backup: [Buffer; 2],
    records: [Buffer; 4],
    input_norm: Vec<CudaRmsnormF32ArgsV1>,
    post_norm: Vec<CudaRmsnormF32ArgsV1>,
    qkv: CheckpointExactBatchProjection,
    z: CheckpointExactBatchProjection,
    gates: Vec<NerkyorBf16GatePair>,
    recurrence: Vec<CudaQwen35CheckpointPrefillLinearChunkV1>,
    out: CheckpointExactBatchProjection,
    residual: CudaAddF32ArgsV1,
    ffn: Ffn,
    final_add: CudaAddF32ArgsV1,
    pending: Option<usize>,
    #[cfg(test)]
    pub(crate) fail_after_recurrence: bool,
}
impl Program {
    pub(crate) fn bind(
        layer: &CheckpointLinearCudaLayer,
        w: Rc<Workspace>,
        input: Buffer,
        output: Buffer,
        epsilon: f32,
    ) -> Result<Self> {
        let [Operation::Norm(_), Operation::Projection(qkv), Operation::Projection(z), Operation::NerkyorBf16GatePair(pair), Operation::Gates(_), Operation::Recurrent(_), Operation::Projection(out), Operation::Add(_), Operation::Norm(_), Operation::Projection(gate), Operation::Projection(up), Operation::Silu(_), Operation::Projection(down), Operation::Add(_)] =
            layer.operations.as_slice()
        else {
            return Err("B8 GDN requires exact Nerkyor gate-F32 original-source inventory".into());
        };
        if layer.poisoned || !Rc::ptr_eq(&w.stream, &layer.stream) {
            return Err("foreign or poisoned B8 GDN".into());
        }
        let s = &w.stream;
        let p = &w.planes;
        let state = [
            Rc::clone(&layer.buffers["history"]),
            Rc::clone(&layer.buffers["state"]),
        ];
        let backup = [
            empty(s, state[0].len_bytes() / 4)?,
            empty(s, state[1].len_bytes() / 4)?,
        ];
        let records = [
            empty(s, WIDTH * 10240)?,
            empty(s, WIDTH * 6144)?,
            empty(s, WIDTH * 48)?,
            empty(s, WIDTH * 48)?,
        ];
        let input_norm = norms(
            s,
            &input,
            &layer.buffers["input_norm_weight"],
            &p["normalized"],
            epsilon,
        )?;
        let post_norm = norms(
            s,
            &p["residual"],
            &layer.buffers["post_norm_weight"],
            &p["post_norm"],
            epsilon,
        )?;
        let qkv = qkv.rebind_exact_batch(
            s,
            Rc::clone(&p["normalized"]),
            Rc::clone(&records[0]),
            WIDTH,
        )?;
        let z = z.rebind_exact_batch(
            s,
            Rc::clone(&p["normalized"]),
            Rc::clone(&records[1]),
            WIDTH,
        )?;
        let gates = pair.rebind_rows(
            s,
            Rc::clone(&p["normalized"]),
            Rc::clone(&records[2]),
            Rc::clone(&records[3]),
            WIDTH,
        )?;
        let mut recurrence = Vec::with_capacity(WIDTH);
        for n in 1..=WIDTH {
            let scalar = |name| view(&layer.buffers[name]);
            recurrence.push(
                CudaQwen35PrefillLinearChunkV1::bind(
                    s,
                    CudaQwen35PrefillLinearChunkViewsV1 {
                        qkv: slice(&records[0], 0, n * 10240)?,
                        z: slice(&records[1], 0, n * 6144)?,
                        conv_weights: scalar("conv")?,
                        conv_history: view(&state[0])?,
                        alpha_raw: slice(&records[2], 0, n * 48)?,
                        beta_raw: slice(&records[3], 0, n * 48)?,
                        dt_bias: scalar("dt_bias")?,
                        ssm_a: scalar("ssm_a")?,
                        state: view(&state[1])?,
                        norm_weight: scalar("norm")?,
                        activated: slice(&p["activated"], 0, n * 10240)?,
                        recurrent: slice(&p["recurrent"], 0, n * 6144)?,
                        output: slice(&p["linear_output"], 0, n * 6144)?,
                    },
                    n,
                    epsilon,
                )?
                .for_checkpoint_source()?,
            );
        }
        let out = out.rebind_exact_batch(
            s,
            Rc::clone(&p["linear_output"]),
            Rc::clone(&p["attention"]),
            WIDTH,
        )?;
        let residual = CudaAddF32ArgsV1::bind(
            s,
            slice(&input, 0, WIDTH * H)?,
            slice(&p["attention"], 0, WIDTH * H)?,
            slice(&p["residual"], 0, WIDTH * H)?,
        )?;
        let ffn = Ffn::bind(Rc::clone(&w), [gate, up, down])?;
        let final_add = CudaAddF32ArgsV1::bind(
            s,
            slice(&p["residual"], 0, WIDTH * H)?,
            slice(&p["ffn"], 0, WIDTH * H)?,
            slice(&output, 0, WIDTH * H)?,
        )?;
        let vectors = [
            "input_norm_weight",
            "post_norm_weight",
            "conv",
            "dt_bias",
            "ssm_a",
            "norm",
        ]
        .map(|n| Rc::clone(&layer.buffers[n]))
        .to_vec();
        Ok(Self {
            w,
            _input: input,
            _output: output,
            _vectors: vectors,
            state,
            backup,
            records,
            input_norm,
            post_norm,
            qkv,
            z,
            gates,
            recurrence,
            out,
            residual,
            ffn,
            final_add,
            pending: None,
            #[cfg(test)]
            fail_after_recurrence: false,
        })
    }
    fn same_owner(&self, l: &CheckpointLinearCudaLayer) -> bool {
        Rc::ptr_eq(&self.w.stream, &l.stream)
            && Rc::ptr_eq(&self.state[0], &l.buffers["history"])
            && Rc::ptr_eq(&self.state[1], &l.buffers["state"])
    }
    pub(crate) fn validate(&self, l: &CheckpointLinearCudaLayer, cursor: usize) -> Result<()> {
        if !self.same_owner(l) || l.poisoned || l.tokens != cursor || self.pending.is_some() {
            return Err("stale/foreign B8 GDN transaction".into());
        }
        Ok(())
    }
    pub(crate) fn execute(
        &mut self,
        l: &mut CheckpointLinearCudaLayer,
        cursor: usize,
    ) -> Result<()> {
        self.validate(l, cursor)?;
        let s = &self.w.stream;
        l.poisoned = true;
        self.pending = Some(cursor);
        for (to, from) in self.backup.iter().zip(&self.state) {
            copy(s, to, 0, from.view(0, from.len_bytes())?)?;
        }
        for op in &self.input_norm {
            op.submit_for_plan(s)?;
        }
        self.qkv.submit()?;
        self.z.submit()?;
        for op in &self.gates {
            op.submit_for_plan()?;
        }
        self.recurrence[WIDTH - 1].submit(s)?;
        #[cfg(test)]
        if self.fail_after_recurrence {
            return Err("injected B8 GDN failure after recurrence".into());
        }
        self.out.submit()?;
        self.residual.submit_for_plan(s)?;
        for op in &self.post_norm {
            op.submit_for_plan(s)?;
        }
        self.ffn.submit()?;
        self.final_add.submit_for_plan(s)?;
        l.tokens = cursor + WIDTH;
        l.poisoned = false;
        Ok(())
    }
    pub(crate) fn validate_commit(
        &self,
        l: &CheckpointLinearCudaLayer,
        cursor: usize,
        n: usize,
    ) -> Result<()> {
        if n > WIDTH
            || !self.same_owner(l)
            || l.poisoned
            || self.pending != Some(cursor)
            || l.tokens != cursor + WIDTH
        {
            return Err("invalid B8 GDN prefix commit".into());
        }
        Ok(())
    }
    pub(crate) fn commit(
        &mut self,
        l: &mut CheckpointLinearCudaLayer,
        cursor: usize,
        n: usize,
    ) -> Result<()> {
        self.validate_commit(l, cursor, n)?;
        l.poisoned = true;
        if n < WIDTH {
            for (to, from) in self.state.iter().zip(&self.backup) {
                copy(&self.w.stream, to, 0, from.view(0, from.len_bytes())?)?;
            }
            if n > 0 {
                self.recurrence[n - 1].submit(&self.w.stream)?;
            }
        }
        l.tokens = cursor + n;
        l.poisoned = false;
        self.pending = None;
        Ok(())
    }
}
impl Drop for Program {
    fn drop(&mut self) {
        let _ = self.w.stream.synchronize();
    }
}

#[cfg(test)]
pub(crate) fn diagnostic_state(
    layer: &CheckpointLinearCudaLayer,
) -> Result<Vec<(&'static str, Vec<u8>)>> {
    if layer.poisoned {
        return Err("poisoned GDN diagnostic".into());
    }
    // The recurrent buffers say what state the GDN recurrence carries; the
    // hidden path says where the divergence entered it. `normalized` is the
    // layer's own RMSNorm output (the operand the QKV/Z/out projections read),
    // and the rest walk the token back out: `z`/`alpha`/`beta` are the
    // projection results, `linear_output` the pre-residual attention result and
    // `ffn_output` the pre-residual MLP result. As in the full layer, the
    // layer's `input`/`output` are NOT captured: they alias the target's global
    // hidden ping-pong, so they would all report the final token.
    [
        "normalized",
        "z",
        "alpha",
        "beta",
        "qkv",
        "activated",
        "recurrent",
        "decay",
        "linear_output",
        "attention_residual",
        "post_norm",
        "ffn_output",
        "history",
        "state",
    ]
        .into_iter()
        .map(|n| {
            let b = &layer.buffers[n];
            let mut raw = vec![0; b.len_bytes()];
            b.copy_to_host_sync(&layer.stream, &mut raw)?;
            Ok((n, raw))
        })
        .collect()
}
