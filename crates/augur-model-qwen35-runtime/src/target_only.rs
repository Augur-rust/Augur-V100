//! Target-only request orchestration for the isolated Qwen35 lane.
//!
//! This module owns the request-local token ledger and the per-layer state
//! lifecycle, but deliberately knows nothing about GGUF storage or CUDA.  A
//! caller supplies a layer executor and a logits head.  Keeping those two
//! boundaries explicit lets the CPU reference, a resident CUDA plan, and a
//! future speculative wrapper share the same transaction rules.

use augur_model_qwen35::{
    Qwen35Config, Qwen35Error, Qwen35LayerCheckpoint, Qwen35LayerSpec, Qwen35LayerState,
};
use std::fmt;

/// Lifecycle of one target-only request owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TargetOnlyPhase {
    Created,
    Prefilling,
    DecodeReady,
    Decoding,
    Finished,
    Poisoned,
}

/// Errors raised by the target-only orchestration boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetOnlyError {
    Model(String),
    Invalid { field: String, detail: String },
    Executor { layer: usize, detail: String },
    Logits(String),
    NotReady(TargetOnlyPhase),
    Checkpoint(String),
}

impl fmt::Display for TargetOnlyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(detail) => write!(formatter, "Qwen model error: {detail}"),
            Self::Invalid { field, detail } => write!(formatter, "{field}: {detail}"),
            Self::Executor { layer, detail } => {
                write!(formatter, "layer {layer} executor error: {detail}")
            }
            Self::Logits(detail) => write!(formatter, "logits head error: {detail}"),
            Self::NotReady(phase) => {
                write!(formatter, "target-only owner is not ready in {phase:?}")
            }
            Self::Checkpoint(detail) => write!(formatter, "checkpoint error: {detail}"),
        }
    }
}

impl std::error::Error for TargetOnlyError {}

impl From<Qwen35Error> for TargetOnlyError {
    fn from(error: Qwen35Error) -> Self {
        Self::Model(error.to_string())
    }
}

/// A layer implementation used by the target-only owner.
///
/// The executor must make its own device-side work transactional.  The
/// owner can restore the model layer states and token ledger, but it cannot
/// roll back hidden mutable state that an executor keeps outside the supplied
/// [`Qwen35LayerState`].  A resident CUDA executor should therefore bind all
/// mutable state to the supplied layer state/checkpoint or expose an
/// equivalent executor checkpoint around this call.
pub trait TargetOnlyLayerExecutor {
    fn forward(
        &mut self,
        layer: usize,
        spec: &Qwen35LayerSpec,
        state: &mut Qwen35LayerState,
        hidden: &[f32],
        position: usize,
    ) -> Result<Vec<f32>, TargetOnlyError>;
}

/// A final-norm/lm-head boundary supplied by the model adapter.
pub trait TargetOnlyLogitHead {
    fn logits(&mut self, hidden: &[f32]) -> Result<Vec<f32>, TargetOnlyError>;
}

/// Input and output for one prompt token in the prefill phase.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefillToken {
    pub token_id: u32,
    pub hidden: Vec<f32>,
}

/// Result of a prefill sequence.  The final hidden is intentionally returned
/// so the caller can run the final norm/lm-head without advancing the owner.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefillOutput {
    pub position: usize,
    pub final_hidden: Vec<f32>,
}

/// One committed target-only decode step.
///
/// `input_token_id` is the token whose embedding entered the transformer at
/// `position_before`; `next_token_id` is the greedy token selected from its
/// logits.  The selected token is recorded immediately, but its embedding is
/// consumed by the next call, which matches autoregressive decoder timing.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodeOutput {
    pub input_token_id: u32,
    pub next_token_id: u32,
    pub position_before: usize,
    pub position_after: usize,
    pub hidden: Vec<f32>,
    pub logits: Vec<f32>,
}

/// A complete checkpoint for the request owner.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetOnlyCheckpoint {
    phase: TargetOnlyPhase,
    position: usize,
    processed_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
    layer_checkpoints: Vec<Qwen35LayerCheckpoint>,
}

/// Request-local target-only state.  Model weights and execution resources
/// remain owned by the caller/executor; this object owns only mutable request
/// state and the publication ledger.
pub struct TargetOnlySession {
    vocab_size: usize,
    max_positions: usize,
    hidden_size: usize,
    specs: Vec<Qwen35LayerSpec>,
    states: Vec<Qwen35LayerState>,
    phase: TargetOnlyPhase,
    position: usize,
    processed_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
}

impl TargetOnlySession {
    /// Construct a session for the authenticated Qwen35 profile.
    pub fn for_config(
        config: &Qwen35Config,
        max_positions: usize,
    ) -> Result<Self, TargetOnlyError> {
        config.validate_exact_profile()?;
        let specs = (0..config.num_hidden_layers)
            .map(|layer| Qwen35LayerSpec::from_config(config, layer))
            .collect::<Result<Vec<_>, _>>()?;
        Self::from_specs(config.vocab_size, max_positions, specs)
    }

    /// Construct a session from explicit specs.  This is useful for a small
    /// CPU fixture and keeps the state-machine tests independent of a model
    /// file.  Production adapters should normally use [`Self::for_config`].
    pub fn from_specs(
        vocab_size: usize,
        max_positions: usize,
        specs: Vec<Qwen35LayerSpec>,
    ) -> Result<Self, TargetOnlyError> {
        if vocab_size == 0 {
            return Err(invalid("vocab_size", "must be nonzero"));
        }
        if max_positions == 0 {
            return Err(invalid("max_positions", "must be nonzero"));
        }
        if specs.is_empty() {
            return Err(invalid("layer_specs", "must contain at least one layer"));
        }
        let hidden_size = specs[0].hidden_size;
        if hidden_size == 0 {
            return Err(invalid("layer_specs[0].hidden_size", "must be nonzero"));
        }
        for (index, spec) in specs.iter().enumerate() {
            spec.validate().map_err(|error| TargetOnlyError::Invalid {
                field: format!("layer_specs[{index}]"),
                detail: error.to_string(),
            })?;
            if spec.hidden_size != hidden_size {
                return Err(invalid(
                    format!("layer_specs[{index}].hidden_size"),
                    format!("expected {hidden_size}, got {}", spec.hidden_size),
                ));
            }
        }
        let states = specs
            .iter()
            .map(|spec| Qwen35LayerState::for_spec(spec, max_positions))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            vocab_size,
            max_positions,
            hidden_size,
            specs,
            states,
            phase: TargetOnlyPhase::Created,
            position: 0,
            processed_token_ids: Vec::new(),
            generated_token_ids: Vec::new(),
        })
    }

    pub fn phase(&self) -> TargetOnlyPhase {
        self.phase
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn layer_count(&self) -> usize {
        self.specs.len()
    }

    pub fn processed_token_ids(&self) -> &[u32] {
        &self.processed_token_ids
    }

    pub fn generated_token_ids(&self) -> &[u32] {
        &self.generated_token_ids
    }

    pub fn checkpoint(&self) -> TargetOnlyCheckpoint {
        TargetOnlyCheckpoint {
            phase: self.phase,
            position: self.position,
            processed_token_ids: self.processed_token_ids.clone(),
            generated_token_ids: self.generated_token_ids.clone(),
            layer_checkpoints: self
                .states
                .iter()
                .map(Qwen35LayerState::checkpoint)
                .collect(),
        }
    }

    pub fn restore(&mut self, checkpoint: &TargetOnlyCheckpoint) -> Result<(), TargetOnlyError> {
        if checkpoint.layer_checkpoints.len() != self.states.len() {
            return Err(TargetOnlyError::Checkpoint(
                "layer checkpoint count does not match session".into(),
            ));
        }
        if checkpoint.position > self.max_positions
            || checkpoint.processed_token_ids.len() != checkpoint.position
        {
            return Err(TargetOnlyError::Checkpoint(
                "checkpoint position/token ledger is inconsistent".into(),
            ));
        }
        if checkpoint
            .processed_token_ids
            .iter()
            .chain(&checkpoint.generated_token_ids)
            .any(|token| (*token as usize) >= self.vocab_size)
        {
            return Err(TargetOnlyError::Checkpoint(
                "checkpoint contains a token outside the vocabulary".into(),
            ));
        }
        if checkpoint.processed_token_ids.is_empty()
            && !matches!(checkpoint.phase, TargetOnlyPhase::Created)
        {
            return Err(TargetOnlyError::Checkpoint(
                "an empty checkpoint must remain in Created phase".into(),
            ));
        }
        let current = self.checkpoint();
        for index in 0..self.states.len() {
            let result = self.states[index].restore(&checkpoint.layer_checkpoints[index]);
            if let Err(error) = result {
                // A public restore is transactional as well.  The current
                // checkpoint was captured before touching any layer state.
                for index in 0..self.states.len() {
                    let _ = self.states[index].restore(&current.layer_checkpoints[index]);
                }
                return Err(error.into());
            }
        }
        self.phase = checkpoint.phase;
        self.position = checkpoint.position;
        self.processed_token_ids = checkpoint.processed_token_ids.clone();
        self.generated_token_ids = checkpoint.generated_token_ids.clone();
        Ok(())
    }

    /// Process a prompt atomically.  The returned final hidden is not passed
    /// through a logits head; use [`Self::prefill_and_select`] when the first
    /// generated token is needed immediately.
    pub fn prefill<E: TargetOnlyLayerExecutor>(
        &mut self,
        tokens: &[PrefillToken],
        executor: &mut E,
    ) -> Result<PrefillOutput, TargetOnlyError> {
        if self.phase != TargetOnlyPhase::Created {
            return Err(TargetOnlyError::NotReady(self.phase));
        }
        if tokens.is_empty() {
            return Err(invalid("prefill", "prompt must contain at least one token"));
        }
        let checkpoint = self.checkpoint();
        self.phase = TargetOnlyPhase::Prefilling;
        let result = (|| {
            let mut final_hidden = vec![0.0_f32; self.hidden_size];
            for token in tokens {
                validate_token(token.token_id, self.vocab_size)?;
                final_hidden = self.forward_token(token.token_id, &token.hidden, executor)?;
            }
            self.phase = TargetOnlyPhase::DecodeReady;
            Ok(PrefillOutput {
                position: self.position,
                final_hidden,
            })
        })();
        if result.is_err() {
            self.restore(&checkpoint)?;
        }
        result
    }

    /// Process a prompt and greedily select its first output token.  Selecting
    /// a token records publication in the generation ledger, but does not
    /// advance the transformer state; the selected token is consumed by the
    /// next [`Self::decode_step`] call.
    pub fn prefill_and_select<E: TargetOnlyLayerExecutor, H: TargetOnlyLogitHead>(
        &mut self,
        tokens: &[PrefillToken],
        executor: &mut E,
        head: &mut H,
    ) -> Result<(PrefillOutput, u32), TargetOnlyError> {
        let checkpoint = self.checkpoint();
        let output = self.prefill(tokens, executor)?;
        let result = (|| {
            let logits = head.logits(&output.final_hidden)?;
            let next = greedy_argmax(&logits, self.vocab_size)?;
            self.generated_token_ids.push(next);
            Ok((output, next))
        })();
        if result.is_err() {
            self.restore(&checkpoint)?;
        }
        result
    }

    /// Process one input token, run the supplied logits head, and commit the
    /// greedy next token.  The whole operation is atomic, including the head:
    /// a failed logits readback cannot leave a partially advanced request.
    pub fn decode_step<E: TargetOnlyLayerExecutor, H: TargetOnlyLogitHead>(
        &mut self,
        input_token_id: u32,
        hidden: &[f32],
        executor: &mut E,
        head: &mut H,
    ) -> Result<DecodeOutput, TargetOnlyError> {
        if !matches!(
            self.phase,
            TargetOnlyPhase::DecodeReady | TargetOnlyPhase::Decoding
        ) {
            return Err(TargetOnlyError::NotReady(self.phase));
        }
        validate_token(input_token_id, self.vocab_size)?;
        let checkpoint = self.checkpoint();
        let position_before = self.position;
        self.phase = TargetOnlyPhase::Decoding;
        let result = (|| {
            let output_hidden = self.forward_token(input_token_id, hidden, executor)?;
            let logits = head.logits(&output_hidden)?;
            let next_token_id = greedy_argmax(&logits, self.vocab_size)?;
            self.generated_token_ids.push(next_token_id);
            Ok(DecodeOutput {
                input_token_id,
                next_token_id,
                position_before,
                position_after: self.position,
                hidden: output_hidden,
                logits,
            })
        })();
        if result.is_err() {
            self.restore(&checkpoint)?;
        }
        result
    }

    /// Mark the request terminal.  No further token execution is accepted.
    pub fn finish(&mut self) -> Result<(), TargetOnlyError> {
        if !matches!(
            self.phase,
            TargetOnlyPhase::DecodeReady | TargetOnlyPhase::Decoding
        ) {
            return Err(TargetOnlyError::NotReady(self.phase));
        }
        self.phase = TargetOnlyPhase::Finished;
        Ok(())
    }

    /// Permanently stop the owner after an unrecoverable device/runtime
    /// failure.  The caller should drain or destroy any external executor
    /// resources before starting a new session; a poisoned owner cannot be
    /// reused for token execution.
    pub fn poison(&mut self) {
        self.phase = TargetOnlyPhase::Poisoned;
    }

    /// Reset all request state to a fresh owner without rebuilding the specs.
    pub fn reset(&mut self) -> Result<(), TargetOnlyError> {
        let states = self
            .specs
            .iter()
            .map(|spec| Qwen35LayerState::for_spec(spec, self.max_positions))
            .collect::<Result<Vec<_>, _>>()?;
        self.states = states;
        self.phase = TargetOnlyPhase::Created;
        self.position = 0;
        self.processed_token_ids.clear();
        self.generated_token_ids.clear();
        Ok(())
    }

    fn forward_token<E: TargetOnlyLayerExecutor>(
        &mut self,
        token_id: u32,
        hidden: &[f32],
        executor: &mut E,
    ) -> Result<Vec<f32>, TargetOnlyError> {
        validate_token(token_id, self.vocab_size)?;
        if self.position >= self.max_positions {
            return Err(invalid(
                "position",
                format!(
                    "{} is outside max_positions {}",
                    self.position, self.max_positions
                ),
            ));
        }
        validate_hidden(hidden, self.hidden_size)?;
        let checkpoint = self.checkpoint();
        let result = (|| {
            let mut current = hidden.to_vec();
            for (layer, (spec, state)) in self.specs.iter().zip(&mut self.states).enumerate() {
                current = executor
                    .forward(layer, spec, state, &current, self.position)
                    .map_err(|error| match error {
                        TargetOnlyError::Executor { .. } => error,
                        other => TargetOnlyError::Executor {
                            layer,
                            detail: other.to_string(),
                        },
                    })?;
                validate_hidden(&current, self.hidden_size).map_err(|error| {
                    TargetOnlyError::Executor {
                        layer,
                        detail: error.to_string(),
                    }
                })?;
            }
            self.position = self
                .position
                .checked_add(1)
                .ok_or_else(|| invalid("position", "overflow while advancing target-only owner"))?;
            self.processed_token_ids.push(token_id);
            Ok(current)
        })();
        if result.is_err() {
            self.restore(&checkpoint)?;
        }
        result
    }
}

/// Select the lowest-index maximum logit.  This is deliberately the only
/// sampler policy in the target-only contract; temperature/top-p belong to a
/// separate, later boundary.
pub fn greedy_argmax(logits: &[f32], vocab_size: usize) -> Result<u32, TargetOnlyError> {
    if logits.len() != vocab_size || logits.is_empty() {
        return Err(TargetOnlyError::Logits(format!(
            "expected {vocab_size} logits, got {}",
            logits.len()
        )));
    }
    let mut best_index = 0usize;
    let mut best_value = logits[0];
    if !best_value.is_finite() {
        return Err(TargetOnlyError::Logits("logits must all be finite".into()));
    }
    for (index, &value) in logits.iter().enumerate().skip(1) {
        if !value.is_finite() {
            return Err(TargetOnlyError::Logits("logits must all be finite".into()));
        }
        if value > best_value {
            best_value = value;
            best_index = index;
        }
    }
    u32::try_from(best_index).map_err(|_| TargetOnlyError::Logits("token id overflow".into()))
}

fn validate_token(token_id: u32, vocab_size: usize) -> Result<(), TargetOnlyError> {
    if (token_id as usize) < vocab_size {
        Ok(())
    } else {
        Err(invalid(
            "token_id",
            format!("{token_id} is outside vocabulary size {vocab_size}"),
        ))
    }
}

fn validate_hidden(hidden: &[f32], hidden_size: usize) -> Result<(), TargetOnlyError> {
    if hidden.len() != hidden_size {
        return Err(invalid(
            "hidden",
            format!("expected {hidden_size} values, got {}", hidden.len()),
        ));
    }
    if hidden.iter().any(|value| !value.is_finite()) {
        return Err(invalid("hidden", "values must be finite"));
    }
    Ok(())
}

fn invalid(field: impl Into<String>, detail: impl Into<String>) -> TargetOnlyError {
    TargetOnlyError::Invalid {
        field: field.into(),
        detail: detail.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_spec() -> Qwen35LayerSpec {
        Qwen35LayerSpec {
            kind: augur_model_qwen35::Qwen35LayerKind::LinearAttention,
            hidden_size: 4,
            intermediate_size: 4,
            query_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            rotary_dim: 2,
            rope_theta: 10_000.0,
            rms_norm_eps: 1.0e-5,
            linear_key_heads: 1,
            linear_value_heads: 1,
            linear_head_dim: 2,
            linear_conv_kernel: 2,
        }
    }

    struct FixtureExecutor {
        fail_layer: Option<usize>,
        calls: Vec<(usize, usize)>,
    }

    impl TargetOnlyLayerExecutor for FixtureExecutor {
        fn forward(
            &mut self,
            layer: usize,
            _spec: &Qwen35LayerSpec,
            _state: &mut Qwen35LayerState,
            hidden: &[f32],
            position: usize,
        ) -> Result<Vec<f32>, TargetOnlyError> {
            self.calls.push((layer, position));
            if self.fail_layer == Some(layer) {
                return Err(TargetOnlyError::Executor {
                    layer,
                    detail: "fixture failure".into(),
                });
            }
            Ok(hidden
                .iter()
                .map(|value| value + layer as f32 + position as f32)
                .collect())
        }
    }

    struct FixtureHead {
        vocab_size: usize,
        fail: bool,
    }

    impl TargetOnlyLogitHead for FixtureHead {
        fn logits(&mut self, hidden: &[f32]) -> Result<Vec<f32>, TargetOnlyError> {
            if self.fail {
                return Err(TargetOnlyError::Logits("fixture readback failure".into()));
            }
            let mut logits = vec![0.0; self.vocab_size];
            let winner = hidden[0].round().max(0.0) as usize % self.vocab_size;
            logits[winner] = 10.0;
            Ok(logits)
        }
    }

    fn session() -> TargetOnlySession {
        TargetOnlySession::from_specs(8, 8, vec![tiny_spec(), tiny_spec()]).unwrap()
    }

    #[test]
    fn prefill_and_decode_commit_in_order() {
        let mut session = session();
        let mut executor = FixtureExecutor {
            fail_layer: None,
            calls: Vec::new(),
        };
        let mut head = FixtureHead {
            vocab_size: 8,
            fail: false,
        };
        let prompt = vec![
            PrefillToken {
                token_id: 1,
                hidden: vec![0.0, 0.0, 0.0, 0.0],
            },
            PrefillToken {
                token_id: 2,
                hidden: vec![1.0, 0.0, 0.0, 0.0],
            },
        ];
        let (prefill, first) = session
            .prefill_and_select(&prompt, &mut executor, &mut head)
            .unwrap();
        assert_eq!(prefill.position, 2);
        assert_eq!(first, 4);
        assert_eq!(session.processed_token_ids(), &[1, 2]);
        assert_eq!(session.generated_token_ids(), &[4]);
        assert_eq!(session.phase(), TargetOnlyPhase::DecodeReady);

        let step = session
            .decode_step(5, &[0.0, 0.0, 0.0, 0.0], &mut executor, &mut head)
            .unwrap();
        assert_eq!(step.position_before, 2);
        assert_eq!(step.position_after, 3);
        assert_eq!(step.input_token_id, 5);
        assert_eq!(step.next_token_id, 5);
        assert_eq!(session.processed_token_ids(), &[1, 2, 5]);
        assert_eq!(session.generated_token_ids(), &[4, 5]);
        assert_eq!(session.phase(), TargetOnlyPhase::Decoding);
        assert_eq!(executor.calls.len(), 6);
    }

    #[test]
    fn layer_failure_restores_position_and_ledger() {
        let mut session = session();
        let mut executor = FixtureExecutor {
            fail_layer: None,
            calls: Vec::new(),
        };
        let mut head = FixtureHead {
            vocab_size: 8,
            fail: false,
        };
        session
            .prefill(
                &[PrefillToken {
                    token_id: 1,
                    hidden: vec![0.0; 4],
                }],
                &mut executor,
            )
            .unwrap();
        let before = session.checkpoint();
        executor.fail_layer = Some(1);
        assert!(session
            .decode_step(2, &[0.0; 4], &mut executor, &mut head)
            .is_err());
        assert_eq!(session.checkpoint(), before);
        assert_eq!(session.phase(), TargetOnlyPhase::DecodeReady);
        executor.fail_layer = None;
        let output = session
            .decode_step(2, &[0.0; 4], &mut executor, &mut head)
            .unwrap();
        assert_eq!(output.position_after, 2);
    }

    #[test]
    fn logits_failure_is_transactional_and_argmax_ties_low() {
        assert_eq!(greedy_argmax(&[3.0, 3.0, 2.0], 3).unwrap(), 0);
        let mut session = session();
        let mut executor = FixtureExecutor {
            fail_layer: None,
            calls: Vec::new(),
        };
        let mut head = FixtureHead {
            vocab_size: 8,
            fail: false,
        };
        session
            .prefill(
                &[PrefillToken {
                    token_id: 1,
                    hidden: vec![0.0; 4],
                }],
                &mut executor,
            )
            .unwrap();
        let before = session.checkpoint();
        head.fail = true;
        assert!(session
            .decode_step(2, &[0.0; 4], &mut executor, &mut head)
            .is_err());
        assert_eq!(session.checkpoint(), before);
    }

    #[test]
    fn invalid_inputs_are_rejected_before_execution() {
        let mut session = session();
        let mut executor = FixtureExecutor {
            fail_layer: None,
            calls: Vec::new(),
        };
        assert!(session
            .prefill(
                &[PrefillToken {
                    token_id: 8,
                    hidden: vec![0.0; 4],
                }],
                &mut executor,
            )
            .is_err());
        assert!(executor.calls.is_empty());
        assert_eq!(session.phase(), TargetOnlyPhase::Created);
        assert!(greedy_argmax(&[f32::NAN, 0.0], 2).is_err());
    }

    #[test]
    fn empty_prefill_and_position_limit_leave_owner_unchanged() {
        let mut session = TargetOnlySession::from_specs(8, 1, vec![tiny_spec()]).unwrap();
        let mut executor = FixtureExecutor {
            fail_layer: None,
            calls: Vec::new(),
        };
        assert!(session.prefill(&[], &mut executor).is_err());
        assert_eq!(session.phase(), TargetOnlyPhase::Created);
        session
            .prefill(
                &[PrefillToken {
                    token_id: 1,
                    hidden: vec![0.0; 4],
                }],
                &mut executor,
            )
            .unwrap();
        let mut head = FixtureHead {
            vocab_size: 8,
            fail: false,
        };
        let before = session.checkpoint();
        assert!(session
            .decode_step(2, &[0.0; 4], &mut executor, &mut head)
            .is_err());
        assert_eq!(session.checkpoint(), before);
        session.finish().unwrap();
        assert!(session
            .decode_step(2, &[0.0; 4], &mut executor, &mut head)
            .is_err());
        session.reset().unwrap();
        assert_eq!(session.phase(), TargetOnlyPhase::Created);
        session.poison();
        assert!(session.finish().is_err());
    }
}
