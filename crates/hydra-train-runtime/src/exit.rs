//! Minimal ExIt target collation helpers used by runtime data collation.

use burn::prelude::*;
use hydra_core::action::HYDRA_ACTION_SPACE;

type OptionalActionTargets = Option<([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE])>;

fn collate_optional_action_targets<B: Backend>(
    samples: &[OptionalActionTargets],
    device: &B::Device,
) -> (Option<Tensor<B, 2>>, Option<Tensor<B, 2>>) {
    if samples.is_empty() || samples.iter().all(|sample| sample.is_none()) {
        return (None, None);
    }

    let batch = samples.len();
    let mut target_data = vec![0.0f32; batch * HYDRA_ACTION_SPACE];
    let mut mask_data = vec![0.0f32; batch * HYDRA_ACTION_SPACE];
    for (index, sample) in samples.iter().enumerate() {
        if let Some((target, mask)) = sample {
            let offset = index * HYDRA_ACTION_SPACE;
            target_data[offset..offset + HYDRA_ACTION_SPACE].copy_from_slice(target);
            mask_data[offset..offset + HYDRA_ACTION_SPACE].copy_from_slice(mask);
        }
    }

    let target_tensor = Tensor::<B, 1>::from_floats(target_data.as_slice(), device)
        .reshape([batch, HYDRA_ACTION_SPACE]);
    let mask_tensor = Tensor::<B, 1>::from_floats(mask_data.as_slice(), device)
        .reshape([batch, HYDRA_ACTION_SPACE]);
    (Some(target_tensor), Some(mask_tensor))
}

pub fn collate_exit_targets<B: Backend>(
    samples: &[OptionalActionTargets],
    device: &B::Device,
) -> (Option<Tensor<B, 2>>, Option<Tensor<B, 2>>) {
    collate_optional_action_targets(samples, device)
}

pub fn collate_delta_q_targets<B: Backend>(
    samples: &[OptionalActionTargets],
    device: &B::Device,
) -> (Option<Tensor<B, 2>>, Option<Tensor<B, 2>>) {
    collate_optional_action_targets(samples, device)
}
