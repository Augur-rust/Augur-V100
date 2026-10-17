//! Opt-in single exact input-prefix slot owned by one serial HTTP producer.
//! It never restores a generated ledger or carries CUDA handles across owners.
use super::*;
use augur_model_qwen35_runtime::session_checkpoint::{
    filesystem::http::PinnedHttpCheckpointProfile,
    input_prefix::{
        InputPrefixDescriptor, InputPrefixGenerationAdmission, InputPrefixRestoreAdmission,
    },
};

pub(super) struct InputCache {
    profile: PinnedHttpCheckpointProfile,
    directory: PathBuf,
    tokens: usize,
    saved: Option<(InputPrefixDescriptor, String)>,
}

pub(super) fn validate_tokens(tokens: usize) -> Result<(), Box<dyn Error>> {
    if !(2048..=16384).contains(&tokens) || tokens % 8 != 0 {
        return Err("input-prefix cache needs 2048..16384 tokens aligned to eight".into());
    }
    Ok(())
}

fn generation_reserve(max_new_tokens: usize) -> Result<usize, Box<dyn Error>> {
    use augur_model_qwen35_runtime::resident_session_profile::{ResidentSessionProfile, VERIFY_WIDTH};
    let maximum = ResidentSessionProfile::Resident.output_budget(Some(max_new_tokens), true)?;
    // Even an anchor-only public answer crosses the native suffix path, whose
    // physical admission requires one complete verifier block. This reserves
    // positions only; it must not increase the user's publication budget.
    Ok(maximum
        .checked_add(VERIFY_WIDTH - 2)
        .ok_or("input-prefix output reserve overflow")?
        .max(VERIFY_WIDTH))
}

impl InputCache {
    pub fn open(
        path: &std::path::Path,
        sha: &str,
        evidence: &std::path::Path,
        tokenizer_sha: &str,
        tokens: usize,
    ) -> Result<Self, Box<dyn Error>> {
        validate_tokens(tokens)?;
        let profile = PinnedHttpCheckpointProfile::open(path, sha)?;
        let build = augur_model_qwen35_runtime::build_identity::identity();
        if build["source_dirty"] != false
            || profile.profile.bindings.frozen_build_sha256 != digest(&serde_json::to_vec(&build)?)
            || profile.profile.bindings.tokenizer_sha256 != tokenizer_sha
            || !profile.profile.math.batched_suffix
            || !profile.profile.math.reuse_batched_plan
        {
            return Err(
                "input-prefix HTTP cache requires exact frozen canonical suffix profile".into(),
            );
        }
        Ok(Self {
            profile,
            directory: evidence.join("input-prefix-slot"),
            tokens,
            saved: None,
        })
    }

    pub fn ready(&self) -> Value {
        json!({"enabled":true,"prefix_tokens":self.tokens,"slots":1,
            "scope":"single_owner_exact_input_prefix","capture_on_first_eligible_open":true,
            "equal_prompt_is_miss":true,"cross_owner_reuse":false,"qualified":false})
    }

    pub fn open_turn(
        &mut self,
        owner: &mut Qwen35ResidentDraftCuda,
        config: &Qwen35AutoregressiveCudaConfig,
        generation: &std::path::Path,
        input: &[u32],
        session_id: &str,
        max_new_tokens: usize,
        sink: &Qwen35VerifiedCommitSink<'_>,
    ) -> Result<Option<Value>, Box<dyn Error>> {
        // A slot is an entire recurrent-state boundary, not a truncatable KV page.
        // Shorter/equal/changed prefixes use the ordinary cold request path.
        if input.len() <= self.tokens {
            return Ok(None);
        }
        let reserve = generation_reserve(max_new_tokens)?;
        let bindings = &self.profile.profile.bindings;
        let math = &self.profile.profile.math;
        if let Some((descriptor, _)) = &self.saved {
            if descriptor
                .lookup_suffix(bindings, math, input, reserve)?
                .is_none()
            {
                return Ok(None);
            }
        }
        let started = Instant::now();
        let hit = self.saved.is_some();
        let mut capture_ms = 0.0;
        let mut prepare_ms = 0.0;
        let mut restore_ms = 0.0;
        let boundary = if let Some((_, manifest)) = &self.saved {
            let restored = owner
                .restore_native_input_prefix(
                    config,
                    generation,
                    &self.directory,
                    manifest,
                    &InputPrefixRestoreAdmission {
                        bindings,
                        math,
                        new_request_id: session_id,
                        request_input: input,
                        output_reserve: reserve,
                    },
                )?
                .ok_or("admitted exact HTTP input-prefix failed native restore")?;
            restore_ms = started.elapsed().as_secs_f64() * 1000.0;
            restored["boundary"].clone()
        } else {
            let prepared = owner.prepare_native_input_prefix(
                config,
                &input[..self.tokens],
                bindings,
                options(generation)?,
            )?;
            prepare_ms = started.elapsed().as_secs_f64() * 1000.0;
            let descriptor = owner.describe_native_input_prefix(&prepared["boundary"], bindings)?;
            if descriptor.math != *math {
                return Err("HTTP input-prefix live/profile math mismatch".into());
            }
            let capture_started = Instant::now();
            let capture = owner.save_native_input_prefix(
                &prepared["boundary"],
                bindings.clone(),
                &self.directory,
            )?;
            capture_ms = capture_started.elapsed().as_secs_f64() * 1000.0;
            let manifest = capture["manifest_sha256"]
                .as_str()
                .ok_or("HTTP input-prefix capture manifest missing")?
                .to_owned();
            self.saved = Some((descriptor, manifest));
            prepared["boundary"].clone()
        };
        let suffix_started = Instant::now();
        let appended = owner
            .append_native_input_prefix(&boundary, bindings, input, reserve)?
            .ok_or("admitted HTTP input-prefix suffix failed")?;
        let suffix_ms = suffix_started.elapsed().as_secs_f64() * 1000.0;
        let mut report = owner.generate_native_input_prefix(
            &appended["boundary"],
            config,
            generation,
            &InputPrefixGenerationAdmission {
                bindings,
                session_id,
                max_new_tokens,
            },
            sink,
        )?;
        report["input_prefix_cache"] = json!({"hit":hit,"slot_created":!hit,
            "prefix_tokens":self.tokens,"prefix_tokens_reused":if hit {self.tokens} else {0},
            "input_tokens":input.len(),"suffix_tokens":input.len()-self.tokens,
            "output_reserve_positions":reserve,"max_new_tokens":max_new_tokens,
            "prefix_prepare_wall_ms":prepare_ms,"capture_wall_ms":capture_ms,
            "restore_wall_ms":restore_ms,"suffix_wall_ms":suffix_ms,
            "total_wall_ms":started.elapsed().as_secs_f64()*1000.0,
            "includes_capture_cost":true,"scope":"single_owner_exact_input_prefix",
            "qualified":false,"visible_tps":false});
        Ok(Some(report))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn one_token_answer_still_reserves_one_physical_verify_block() {
        for (budget, positions) in [(1, 8), (2, 8), (3, 9), (8, 14), (64, 70), (1024, 1030)] {
            assert_eq!(generation_reserve(budget).unwrap(), positions);
        }
        for budget in [0, 1025, usize::MAX] {
            assert!(generation_reserve(budget).is_err(), "{budget}");
        }
    }
    #[test]
    fn explicit_cache_extent_is_bounded_and_aligned() {
        for n in [2048, 8192, 16384] {
            assert!(validate_tokens(n).is_ok());
        }
        for n in [0, 1, 2047, 2049, 16385, 196608, usize::MAX] {
            assert!(validate_tokens(n).is_err(), "{n}");
        }
    }
}
