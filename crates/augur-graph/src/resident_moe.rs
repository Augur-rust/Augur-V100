use std::collections::HashSet;

use augur_core::{Shape, StorageId, StorageRef, TensorDesc};

#[derive(Clone, Debug, PartialEq)]
pub struct ResidentMoeExpertDesc {
    pub gate: TensorDesc,
    pub up: TensorDesc,
    pub down: TensorDesc,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResidentMoeGraph {
    pub input: TensorDesc,
    pub router: TensorDesc,
    pub experts: Box<[ResidentMoeExpertDesc]>,
    pub shared_expert: ResidentMoeExpertDesc,
    pub output: TensorDesc,
    pub top_k: usize,
    pub routed_scale: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidentMoeExpertStorages {
    pub gate: StorageId,
    pub up: StorageId,
    pub down: StorageId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResidentMoeWeightStorages {
    pub router: StorageId,
    pub experts: Box<[ResidentMoeExpertStorages]>,
    pub shared_expert: ResidentMoeExpertStorages,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResidentMoeAnalysis {
    pub tokens: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub expert_count: usize,
    pub top_k: usize,
    pub routed_scale: f32,
    pub weight_storages: ResidentMoeWeightStorages,
    pub logical_weight_count: usize,
    pub weight_payload_bytes: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResidentMoeTensorRole {
    Input,
    Output,
    Router,
    ExpertGate(usize),
    ExpertUp(usize),
    ExpertDown(usize),
    SharedGate,
    SharedUp,
    SharedDown,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResidentMoeValidationError {
    InvalidActivation { role: ResidentMoeTensorRole },
    InvalidActivationShape,
    UnsupportedExpertCount { actual: usize },
    UnsupportedTopK { top_k: usize, expert_count: usize },
    InvalidRoutedScale,
    InvalidWeight { role: ResidentMoeTensorRole },
    InvalidWeightShape { role: ResidentMoeTensorRole },
    DuplicateWeightStorage { storage: StorageId },
    ArithmeticOverflow,
}

impl ResidentMoeGraph {
    pub fn analyze(&self) -> Result<ResidentMoeAnalysis, ResidentMoeValidationError> {
        let (tokens, hidden) = activation_shape_for(&self.input, ResidentMoeTensorRole::Input)?;
        let (output_tokens, output_hidden) =
            activation_shape_for(&self.output, ResidentMoeTensorRole::Output)?;
        if tokens != output_tokens || hidden != output_hidden {
            return Err(ResidentMoeValidationError::InvalidActivationShape);
        }

        let expert_count = self.experts.len();
        if !matches!(expert_count, 4 | 8) {
            return Err(ResidentMoeValidationError::UnsupportedExpertCount {
                actual: expert_count,
            });
        }
        if !matches!(self.top_k, 2 | 4) || self.top_k > expert_count {
            return Err(ResidentMoeValidationError::UnsupportedTopK {
                top_k: self.top_k,
                expert_count,
            });
        }
        if !self.routed_scale.is_finite() || self.routed_scale <= 0.0 {
            return Err(ResidentMoeValidationError::InvalidRoutedScale);
        }

        let router = weight_storage(
            &self.router,
            &[expert_count, hidden],
            ResidentMoeTensorRole::Router,
        )?;
        let intermediate = expert_intermediate(&self.shared_expert, hidden)?;
        let mut experts = Vec::with_capacity(expert_count);
        for (index, expert) in self.experts.iter().enumerate() {
            experts.push(expert_storages(expert, index, hidden, intermediate)?);
        }
        let shared_expert = shared_storages(&self.shared_expert, hidden, intermediate)?;

        let mut unique = HashSet::with_capacity(1 + 3 * (expert_count + 1));
        for storage in std::iter::once(router)
            .chain(
                experts
                    .iter()
                    .flat_map(|expert| [expert.gate, expert.up, expert.down]),
            )
            .chain([shared_expert.gate, shared_expert.up, shared_expert.down])
        {
            if !unique.insert(storage) {
                return Err(ResidentMoeValidationError::DuplicateWeightStorage { storage });
            }
        }

        let logical_weight_count = expert_count
            .checked_add(1)
            .and_then(|count| count.checked_mul(3))
            .and_then(|count| count.checked_add(1))
            .ok_or(ResidentMoeValidationError::ArithmeticOverflow)?;
        let router_elements = expert_count
            .checked_mul(hidden)
            .ok_or(ResidentMoeValidationError::ArithmeticOverflow)?;
        let expert_elements = expert_count
            .checked_add(1)
            .and_then(|count| count.checked_mul(3))
            .and_then(|count| count.checked_mul(hidden))
            .and_then(|count| count.checked_mul(intermediate))
            .ok_or(ResidentMoeValidationError::ArithmeticOverflow)?;
        let weight_payload_bytes = router_elements
            .checked_add(expert_elements)
            .and_then(|elements| elements.checked_mul(size_of::<f32>()))
            .ok_or(ResidentMoeValidationError::ArithmeticOverflow)?;

        Ok(ResidentMoeAnalysis {
            tokens,
            hidden,
            intermediate,
            expert_count,
            top_k: self.top_k,
            routed_scale: self.routed_scale,
            weight_storages: ResidentMoeWeightStorages {
                router,
                experts: experts.into_boxed_slice(),
                shared_expert,
            },
            logical_weight_count,
            weight_payload_bytes,
        })
    }
}

fn activation_shape_for(
    desc: &TensorDesc,
    role: ResidentMoeTensorRole,
) -> Result<(usize, usize), ResidentMoeValidationError> {
    if !desc.is_plain_f32() || !desc.is_contiguous() || !desc.storage.is_none() {
        return Err(ResidentMoeValidationError::InvalidActivation { role });
    }
    match desc.shape.dims() {
        [hidden] if *hidden > 0 => Ok((1, *hidden)),
        [tokens, hidden] if *tokens > 0 && *hidden > 0 => Ok((*tokens, *hidden)),
        _ => Err(ResidentMoeValidationError::InvalidActivationShape),
    }
}

fn weight_storage(
    desc: &TensorDesc,
    shape: &[usize],
    role: ResidentMoeTensorRole,
) -> Result<StorageId, ResidentMoeValidationError> {
    if !desc.is_plain_f32() || !desc.is_contiguous() {
        return Err(ResidentMoeValidationError::InvalidWeight { role });
    }
    if desc.shape != Shape::from_dims(shape) {
        return Err(ResidentMoeValidationError::InvalidWeightShape { role });
    }
    match desc.storage {
        StorageRef::HostOwned { id } => Ok(id),
        _ => Err(ResidentMoeValidationError::InvalidWeight { role }),
    }
}

fn expert_intermediate(
    expert: &ResidentMoeExpertDesc,
    hidden: usize,
) -> Result<usize, ResidentMoeValidationError> {
    match expert.gate.shape.dims() {
        [intermediate, actual_hidden] if *intermediate > 0 && *actual_hidden == hidden => {
            Ok(*intermediate)
        }
        _ => Err(ResidentMoeValidationError::InvalidWeightShape {
            role: ResidentMoeTensorRole::SharedGate,
        }),
    }
}

fn expert_storages(
    expert: &ResidentMoeExpertDesc,
    index: usize,
    hidden: usize,
    intermediate: usize,
) -> Result<ResidentMoeExpertStorages, ResidentMoeValidationError> {
    Ok(ResidentMoeExpertStorages {
        gate: weight_storage(
            &expert.gate,
            &[intermediate, hidden],
            ResidentMoeTensorRole::ExpertGate(index),
        )?,
        up: weight_storage(
            &expert.up,
            &[intermediate, hidden],
            ResidentMoeTensorRole::ExpertUp(index),
        )?,
        down: weight_storage(
            &expert.down,
            &[hidden, intermediate],
            ResidentMoeTensorRole::ExpertDown(index),
        )?,
    })
}

fn shared_storages(
    expert: &ResidentMoeExpertDesc,
    hidden: usize,
    intermediate: usize,
) -> Result<ResidentMoeExpertStorages, ResidentMoeValidationError> {
    Ok(ResidentMoeExpertStorages {
        gate: weight_storage(
            &expert.gate,
            &[intermediate, hidden],
            ResidentMoeTensorRole::SharedGate,
        )?,
        up: weight_storage(
            &expert.up,
            &[intermediate, hidden],
            ResidentMoeTensorRole::SharedUp,
        )?,
        down: weight_storage(
            &expert.down,
            &[hidden, intermediate],
            ResidentMoeTensorRole::SharedDown,
        )?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use augur_core::{DType, LogicalLayout};

    fn weight(id: usize, shape: &[usize]) -> TensorDesc {
        TensorDesc {
            dtype: DType::F32,
            shape: Shape::from_dims(shape),
            layout: LogicalLayout::Contiguous,
            storage: StorageRef::HostOwned {
                id: StorageId::new(id, 0),
            },
            quant: None,
        }
    }

    fn graph() -> ResidentMoeGraph {
        let hidden = 8;
        let intermediate = 12;
        let mut next = 1;
        let router = weight(0, &[4, hidden]);
        let experts = (0..4)
            .map(|_| {
                let expert = ResidentMoeExpertDesc {
                    gate: weight(next, &[intermediate, hidden]),
                    up: weight(next + 1, &[intermediate, hidden]),
                    down: weight(next + 2, &[hidden, intermediate]),
                };
                next += 3;
                expert
            })
            .collect::<Vec<_>>();
        let shared_expert = ResidentMoeExpertDesc {
            gate: weight(next, &[intermediate, hidden]),
            up: weight(next + 1, &[intermediate, hidden]),
            down: weight(next + 2, &[hidden, intermediate]),
        };
        ResidentMoeGraph {
            input: TensorDesc::f32_contiguous(Shape::from_dims(&[hidden])),
            router,
            experts: experts.into_boxed_slice(),
            shared_expert,
            output: TensorDesc::f32_contiguous(Shape::from_dims(&[hidden])),
            top_k: 2,
            routed_scale: 1.25,
        }
    }

    #[test]
    fn validates_four_expert_top_two_decode() {
        let analysis = graph().analyze().unwrap();
        assert_eq!(analysis.tokens, 1);
        assert_eq!(analysis.hidden, 8);
        assert_eq!(analysis.intermediate, 12);
        assert_eq!(analysis.logical_weight_count, 16);
        assert_eq!(analysis.weight_payload_bytes, (4 * 8 + 5 * 3 * 8 * 12) * 4);
    }

    #[test]
    fn duplicate_weight_storage_is_rejected() {
        let mut graph = graph();
        graph.experts[1].gate.storage = graph.experts[0].gate.storage.clone();
        assert!(matches!(
            graph.analyze(),
            Err(ResidentMoeValidationError::DuplicateWeightStorage { .. })
        ));
    }

    #[test]
    fn unsupported_top_k_is_rejected() {
        let mut graph = graph();
        graph.top_k = 3;
        assert_eq!(
            graph.analyze().unwrap_err(),
            ResidentMoeValidationError::UnsupportedTopK {
                top_k: 3,
                expert_count: 4,
            }
        );
    }
}
