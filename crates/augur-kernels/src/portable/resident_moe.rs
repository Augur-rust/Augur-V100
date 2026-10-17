use augur_kernel_api::{
    KernelLaunchError, ResidentMoeF32HotCpu, ResidentMoeF32HotExpert, ResidentMoeF32Workspace,
    RESIDENT_MOE_MAX_EXPERTS, RESIDENT_MOE_MAX_TOP_K,
};

pub fn launch_resident_moe_f32_cpu_portable(
    weights: &ResidentMoeF32HotCpu,
    input: &[f32],
    top_k: usize,
    routed_scale: f32,
    workspace: &mut ResidentMoeF32Workspace,
    output: &mut [f32],
) -> Result<(), KernelLaunchError> {
    let tokens = workspace.tokens();
    let hidden = weights.hidden();
    let expert_count = weights.expert_count();
    let expected = tokens
        .checked_mul(hidden)
        .ok_or(KernelLaunchError::BadRuntimeConfig)?;
    if input.len() != expected {
        return Err(KernelLaunchError::BadActivationLength {
            expected,
            actual: input.len(),
        });
    }
    if output.len() != expected {
        return Err(KernelLaunchError::BadOutputLength {
            expected,
            actual: output.len(),
        });
    }
    if !matches!(top_k, 2 | 4)
        || top_k > expert_count
        || !routed_scale.is_finite()
        || routed_scale <= 0.0
    {
        return Err(KernelLaunchError::BadRuntimeConfig);
    }
    if workspace.hidden() != hidden
        || workspace.intermediate() != weights.intermediate()
        || workspace.expert_count() != expert_count
        || workspace.top_k() != top_k
    {
        return Err(KernelLaunchError::BadWorkspace);
    }
    if let Some(index) = input.iter().position(|value| !value.is_finite()) {
        return Err(KernelLaunchError::NonFiniteActivation { index });
    }

    workspace.reset_trace();
    for token in 0..tokens {
        let start = token * hidden;
        let end = start + hidden;
        let activation = &input[start..end];
        let destination = &mut output[start..end];
        destination.copy_from_slice(activation);

        let mut logits = [0.0_f32; RESIDENT_MOE_MAX_EXPERTS];
        for (expert, logit) in logits.iter_mut().enumerate().take(expert_count) {
            let row = &weights.router()[expert * hidden..(expert + 1) * hidden];
            *logit = dot(row, activation);
        }
        workspace
            .router_logits_token_mut(token)
            .ok_or(KernelLaunchError::BadWorkspace)?
            .copy_from_slice(&logits[..expert_count]);

        let mut selected_ids = [usize::MAX; RESIDENT_MOE_MAX_TOP_K];
        let mut selected_logits = [f32::NEG_INFINITY; RESIDENT_MOE_MAX_TOP_K];
        stable_top_k(
            &logits[..expert_count],
            top_k,
            &mut selected_ids,
            &mut selected_logits,
        );
        let mut route_weights = [0.0_f32; RESIDENT_MOE_MAX_TOP_K];
        selected_softmax(&selected_logits, top_k, &mut route_weights)?;

        {
            let trace_ids = workspace
                .selected_ids_token_mut(token)
                .ok_or(KernelLaunchError::BadWorkspace)?;
            for index in 0..top_k {
                trace_ids[index] = selected_ids[index] as u32;
            }
        }
        workspace
            .selected_weights_token_mut(token)
            .ok_or(KernelLaunchError::BadWorkspace)?
            .copy_from_slice(&route_weights[..top_k]);

        for route in 0..top_k {
            let expert_id = selected_ids[route];
            let expert = weights
                .expert(expert_id)
                .ok_or(KernelLaunchError::BadWeight)?;
            if !workspace.record_expert_access(expert_id) {
                return Err(KernelLaunchError::BadWorkspace);
            }
            accumulate_expert(
                expert,
                activation,
                route_weights[route] * routed_scale,
                hidden,
                weights.intermediate(),
                workspace,
                destination,
            );
        }

        workspace.record_shared_access();
        accumulate_expert(
            weights.shared_expert(),
            activation,
            1.0,
            hidden,
            weights.intermediate(),
            workspace,
            destination,
        );
    }
    Ok(())
}

fn stable_top_k(
    logits: &[f32],
    top_k: usize,
    selected_ids: &mut [usize; RESIDENT_MOE_MAX_TOP_K],
    selected_logits: &mut [f32; RESIDENT_MOE_MAX_TOP_K],
) {
    for (expert, &logit) in logits.iter().enumerate() {
        for slot in 0..top_k {
            let better = logit > selected_logits[slot]
                || (logit == selected_logits[slot] && expert < selected_ids[slot]);
            if better {
                for shift in (slot + 1..top_k).rev() {
                    selected_logits[shift] = selected_logits[shift - 1];
                    selected_ids[shift] = selected_ids[shift - 1];
                }
                selected_logits[slot] = logit;
                selected_ids[slot] = expert;
                break;
            }
        }
    }
}

fn selected_softmax(
    logits: &[f32; RESIDENT_MOE_MAX_TOP_K],
    top_k: usize,
    output: &mut [f32; RESIDENT_MOE_MAX_TOP_K],
) -> Result<(), KernelLaunchError> {
    let max = logits[0];
    let mut sum = 0.0;
    for index in 0..top_k {
        output[index] = (logits[index] - max).exp();
        sum += output[index];
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(KernelLaunchError::BadRuntimeConfig);
    }
    for value in &mut output[..top_k] {
        *value /= sum;
    }
    Ok(())
}

fn accumulate_expert(
    expert: &ResidentMoeF32HotExpert,
    activation: &[f32],
    factor: f32,
    hidden: usize,
    intermediate: usize,
    workspace: &mut ResidentMoeF32Workspace,
    output: &mut [f32],
) {
    let (gate, up, hidden_values) = workspace.scratch_mut();
    matvec(expert.gate(), intermediate, hidden, activation, gate);
    matvec(expert.up(), intermediate, hidden, activation, up);
    for index in 0..intermediate {
        hidden_values[index] = silu(gate[index]) * up[index];
    }
    for row in 0..hidden {
        let start = row * intermediate;
        output[row] += factor * dot(&expert.down()[start..start + intermediate], hidden_values);
    }
}

fn matvec(weight: &[f32], rows: usize, cols: usize, input: &[f32], output: &mut [f32]) {
    for row in 0..rows {
        let start = row * cols;
        output[row] = dot(&weight[start..start + cols], input);
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for index in 0..a.len() {
        sum += a[index] * b[index];
    }
    sum
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expert(hidden: usize, intermediate: usize, seed: usize) -> ResidentMoeF32HotExpert {
        let values = |count: usize, salt: usize| {
            (0..count)
                .map(|index| ((index * 13 + seed * 17 + salt) % 31) as f32 / 97.0 - 0.15)
                .collect()
        };
        ResidentMoeF32HotExpert::new(
            hidden,
            intermediate,
            values(hidden * intermediate, 1),
            values(hidden * intermediate, 7),
            values(hidden * intermediate, 11),
        )
        .unwrap()
    }

    #[test]
    fn stable_tie_break_selects_lower_expert_ids_and_skips_unselected() {
        let hidden = 4;
        let intermediate = 6;
        let router = vec![0.0; 4 * hidden];
        let weights = ResidentMoeF32HotCpu::new(
            hidden,
            intermediate,
            router,
            (0..4)
                .map(|index| expert(hidden, intermediate, index))
                .collect(),
            expert(hidden, intermediate, 9),
        )
        .unwrap();
        let mut workspace = ResidentMoeF32Workspace::new(1, hidden, intermediate, 4, 2).unwrap();
        let mut output = vec![0.0; hidden];
        launch_resident_moe_f32_cpu_portable(
            &weights,
            &[0.25, -0.5, 0.75, 1.0],
            2,
            1.0,
            &mut workspace,
            &mut output,
        )
        .unwrap();
        assert_eq!(workspace.selected_ids(), &[0, 1]);
        assert_eq!(workspace.expert_access_counts(), &[1, 1, 0, 0]);
        assert_eq!(workspace.shared_access_count(), 1);
    }
}
