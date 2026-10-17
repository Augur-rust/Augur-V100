#[derive(Clone, Copy, Debug)]
pub struct ResidentMoeReferenceExpert<'a> {
    pub gate: &'a [f32],
    pub up: &'a [f32],
    pub down: &'a [f32],
}

#[derive(Clone, Debug)]
pub struct ResidentMoeReferenceWeights<'a> {
    pub router: &'a [f32],
    pub experts: Vec<ResidentMoeReferenceExpert<'a>>,
    pub shared_expert: ResidentMoeReferenceExpert<'a>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ResidentMoeReferenceConfig {
    pub tokens: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub top_k: usize,
    pub routed_scale: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResidentMoeReferenceOutput {
    pub output: Vec<f32>,
    pub router_logits: Vec<f32>,
    pub selected_ids: Vec<u32>,
    pub selected_weights: Vec<f32>,
    pub expert_access_counts: Vec<u32>,
    pub shared_access_count: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentMoeReferenceError {
    InvalidConfig,
    InvalidWeight,
    InvalidInput,
}

pub fn resident_moe_f32_reference(
    config: ResidentMoeReferenceConfig,
    weights: &ResidentMoeReferenceWeights<'_>,
    input: &[f32],
) -> Result<ResidentMoeReferenceOutput, ResidentMoeReferenceError> {
    let expert_count = weights.experts.len();
    if config.tokens == 0
        || config.hidden == 0
        || config.intermediate == 0
        || !matches!(expert_count, 4 | 8)
        || !matches!(config.top_k, 2 | 4)
        || config.top_k > expert_count
        || !config.routed_scale.is_finite()
        || config.routed_scale <= 0.0
    {
        return Err(ResidentMoeReferenceError::InvalidConfig);
    }
    let activation_len = config
        .tokens
        .checked_mul(config.hidden)
        .ok_or(ResidentMoeReferenceError::InvalidConfig)?;
    if input.len() != activation_len || input.iter().any(|value| !value.is_finite()) {
        return Err(ResidentMoeReferenceError::InvalidInput);
    }
    if weights.router.len() != expert_count * config.hidden
        || weights
            .experts
            .iter()
            .chain(std::iter::once(&weights.shared_expert))
            .any(|expert| !expert_valid(*expert, config.hidden, config.intermediate))
    {
        return Err(ResidentMoeReferenceError::InvalidWeight);
    }

    let mut result = ResidentMoeReferenceOutput {
        output: input.to_vec(),
        router_logits: vec![0.0; config.tokens * expert_count],
        selected_ids: vec![u32::MAX; config.tokens * config.top_k],
        selected_weights: vec![0.0; config.tokens * config.top_k],
        expert_access_counts: vec![0; expert_count],
        shared_access_count: 0,
    };
    let mut gate = vec![0.0; config.intermediate];
    let mut up = vec![0.0; config.intermediate];
    let mut hidden_values = vec![0.0; config.intermediate];

    for token in 0..config.tokens {
        let activation_start = token * config.hidden;
        let activation = &input[activation_start..activation_start + config.hidden];
        let output = &mut result.output[activation_start..activation_start + config.hidden];
        let logits_start = token * expert_count;
        let logits = &mut result.router_logits[logits_start..logits_start + expert_count];
        for (expert, logit) in logits.iter_mut().enumerate() {
            *logit = dot(
                &weights.router[expert * config.hidden..(expert + 1) * config.hidden],
                activation,
            );
        }

        let mut routes: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        routes.sort_by(|(id_a, logit_a), (id_b, logit_b)| {
            logit_b.total_cmp(logit_a).then_with(|| id_a.cmp(id_b))
        });
        routes.truncate(config.top_k);
        let max = routes[0].1;
        let denominator: f32 = routes.iter().map(|(_, logit)| (*logit - max).exp()).sum();
        let route_start = token * config.top_k;
        for (route, (expert_id, logit)) in routes.into_iter().enumerate() {
            let weight = (logit - max).exp() / denominator;
            result.selected_ids[route_start + route] = expert_id as u32;
            result.selected_weights[route_start + route] = weight;
            result.expert_access_counts[expert_id] += 1;
            accumulate_expert(
                weights.experts[expert_id],
                activation,
                weight * config.routed_scale,
                config.hidden,
                config.intermediate,
                &mut gate,
                &mut up,
                &mut hidden_values,
                output,
            );
        }
        result.shared_access_count += 1;
        accumulate_expert(
            weights.shared_expert,
            activation,
            1.0,
            config.hidden,
            config.intermediate,
            &mut gate,
            &mut up,
            &mut hidden_values,
            output,
        );
    }
    Ok(result)
}

fn expert_valid(
    expert: ResidentMoeReferenceExpert<'_>,
    hidden: usize,
    intermediate: usize,
) -> bool {
    let matrix = hidden * intermediate;
    expert.gate.len() == matrix
        && expert.up.len() == matrix
        && expert.down.len() == matrix
        && expert
            .gate
            .iter()
            .chain(expert.up)
            .chain(expert.down)
            .all(|value| value.is_finite())
}

#[allow(clippy::too_many_arguments)]
fn accumulate_expert(
    expert: ResidentMoeReferenceExpert<'_>,
    activation: &[f32],
    factor: f32,
    hidden: usize,
    intermediate: usize,
    gate: &mut [f32],
    up: &mut [f32],
    hidden_values: &mut [f32],
    output: &mut [f32],
) {
    for row in 0..intermediate {
        gate[row] = dot(&expert.gate[row * hidden..(row + 1) * hidden], activation);
        up[row] = dot(&expert.up[row * hidden..(row + 1) * hidden], activation);
        hidden_values[row] = silu(gate[row]) * up[row];
    }
    for (row, value) in output.iter_mut().enumerate().take(hidden) {
        *value += factor
            * dot(
                &expert.down[row * intermediate..(row + 1) * intermediate],
                hidden_values,
            );
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(left, right)| left * right).sum()
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_tie_break_is_lower_expert_id() {
        let hidden = 2;
        let intermediate = 2;
        let zero = vec![0.0; hidden * intermediate];
        let expert = ResidentMoeReferenceExpert {
            gate: &zero,
            up: &zero,
            down: &zero,
        };
        let result = resident_moe_f32_reference(
            ResidentMoeReferenceConfig {
                tokens: 1,
                hidden,
                intermediate,
                top_k: 2,
                routed_scale: 1.0,
            },
            &ResidentMoeReferenceWeights {
                router: &[0.0; 8],
                experts: vec![expert; 4],
                shared_expert: expert,
            },
            &[1.0, -1.0],
        )
        .unwrap();
        assert_eq!(result.selected_ids, [0, 1]);
        assert_eq!(result.expert_access_counts, [1, 1, 0, 0]);
        assert_eq!(result.output, [1.0, -1.0]);
    }
}
