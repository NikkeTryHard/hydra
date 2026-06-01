use burn::prelude::*;
use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::{NUM_CHANNELS, NUM_TILES, OBS_SIZE};

use super::HydraModel;

impl<B: Backend> HydraModel<B> {
    /// Runs a single observation through the full model and returns policy
    /// logits and value scalar on the CPU.
    ///
    /// This is the adapter used by the live ExIt producer during self-play.
    /// It performs a single-sample forward pass, extracts the policy logits
    /// as a fixed-size array and the value head output as a scalar.
    ///
    /// # Panics
    ///
    /// Panics if the forward pass produces non-extractable tensor data.
    pub fn policy_value_cpu(
        &self,
        obs: &[f32; OBS_SIZE],
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> ([f32; HYDRA_ACTION_SPACE], f32) {
        let input = Tensor::<B, 1>::from_floats(obs.as_slice(), device).reshape([
            1,
            NUM_CHANNELS,
            NUM_TILES,
        ]);
        let (policy_logits, value) = self.forward_policy_value(input);
        let logits_data = policy_logits.to_data().convert::<f32>();
        let logits_slice = logits_data
            .as_slice::<f32>()
            .expect("policy logits extraction failed");
        let mut logits = [0.0f32; HYDRA_ACTION_SPACE];
        logits.copy_from_slice(&logits_slice[..HYDRA_ACTION_SPACE]);
        let value_scalar = value
            .to_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .expect("value extraction failed")[0];
        (logits, value_scalar)
    }

    pub fn policy_cpu(
        &self,
        obs: &[f32; OBS_SIZE],
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> [f32; HYDRA_ACTION_SPACE] {
        let input = Tensor::<B, 1>::from_floats(obs.as_slice(), device).reshape([
            1,
            NUM_CHANNELS,
            NUM_TILES,
        ]);
        let policy_logits = self.forward_policy(input);
        let logits_data = policy_logits.to_data().convert::<f32>();
        let logits_slice = logits_data
            .as_slice::<f32>()
            .expect("policy logits extraction failed");
        let mut logits = [0.0f32; HYDRA_ACTION_SPACE];
        logits.copy_from_slice(&logits_slice[..HYDRA_ACTION_SPACE]);
        logits
    }

    pub fn value_cpu(
        &self,
        obs: &[f32; OBS_SIZE],
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> f32 {
        let input = Tensor::<B, 1>::from_floats(obs.as_slice(), device).reshape([
            1,
            NUM_CHANNELS,
            NUM_TILES,
        ]);
        let value = self.forward_value(input);
        value
            .to_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .expect("value extraction failed")[0]
    }

    /// Batch inference using a caller-provided flat buffer to avoid
    /// per-call allocation. The buffer is cleared and reused each call.
    pub fn fill_batch_policy_value_cpu(
        &self,
        observations: &[[f32; OBS_SIZE]],
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
        flat_buf: &mut Vec<f32>,
        outputs: &mut Vec<([f32; HYDRA_ACTION_SPACE], f32)>,
    ) {
        if observations.is_empty() {
            outputs.clear();
            return;
        }
        let n = observations.len();
        flat_buf.clear();
        flat_buf.reserve(n * OBS_SIZE);
        for obs in observations {
            flat_buf.extend_from_slice(obs);
        }
        let input = Tensor::<B, 1>::from_floats(flat_buf.as_slice(), device).reshape([
            n as i32,
            NUM_CHANNELS as i32,
            NUM_TILES as i32,
        ]);
        let (policy_logits, value) = self.forward_policy_value(input);
        let logits_data = policy_logits.to_data().convert::<f32>();
        let logits_flat = logits_data
            .as_slice::<f32>()
            .expect("batch policy logits extraction failed");
        let values_data = value.to_data().convert::<f32>();
        let values_flat = values_data
            .as_slice::<f32>()
            .expect("batch value extraction failed");

        outputs.clear();
        outputs.reserve(n);
        for (i, &value) in values_flat.iter().enumerate().take(n) {
            let logits_start = i * HYDRA_ACTION_SPACE;
            let logits: [f32; HYDRA_ACTION_SPACE] = logits_flat
                [logits_start..logits_start + HYDRA_ACTION_SPACE]
                .try_into()
                .expect("logits slice length mismatch");
            outputs.push((logits, value));
        }
    }

    pub fn fill_batch_value_cpu(
        &self,
        observations: &[[f32; OBS_SIZE]],
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
        flat_buf: &mut Vec<f32>,
        values_out: &mut Vec<f32>,
    ) {
        if observations.is_empty() {
            values_out.clear();
            return;
        }
        let n = observations.len();
        flat_buf.clear();
        flat_buf.reserve(n * OBS_SIZE);
        for obs in observations {
            flat_buf.extend_from_slice(obs);
        }
        let input = Tensor::<B, 1>::from_floats(flat_buf.as_slice(), device).reshape([
            n as i32,
            NUM_CHANNELS as i32,
            NUM_TILES as i32,
        ]);
        let value = self.forward_value(input);
        let values_data = value.to_data().convert::<f32>();
        let values = values_data
            .as_slice::<f32>()
            .expect("batch value extraction failed");
        values_out.clear();
        values_out.extend_from_slice(values);
    }

    /// Runs a batch of observations through the full model and returns
    /// per-sample policy logits and value scalars on the CPU.
    ///
    /// This amortizes GPU kernel launch overhead across N samples. The
    /// input observations are concatenated into a single `[N, C, T]` tensor
    /// for one forward pass, then results are sliced per sample.
    pub fn batch_policy_value_cpu(
        &self,
        observations: &[[f32; OBS_SIZE]],
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> Vec<([f32; HYDRA_ACTION_SPACE], f32)> {
        if observations.is_empty() {
            return Vec::new();
        }
        let n = observations.len();
        let mut flat = Vec::with_capacity(n * OBS_SIZE);
        for obs in observations {
            flat.extend_from_slice(obs);
        }
        let input = Tensor::<B, 1>::from_floats(flat.as_slice(), device).reshape([
            n as i32,
            NUM_CHANNELS as i32,
            NUM_TILES as i32,
        ]);
        let (policy_logits, value) = self.forward_policy_value(input);
        let logits_data = policy_logits.to_data().convert::<f32>();
        let logits_flat = logits_data
            .as_slice::<f32>()
            .expect("batch policy logits extraction failed");
        let values_data = value.to_data().convert::<f32>();
        let values_flat = values_data
            .as_slice::<f32>()
            .expect("batch value extraction failed");

        (0..n)
            .map(|i| {
                let logits_start = i * HYDRA_ACTION_SPACE;
                let logits: [f32; HYDRA_ACTION_SPACE] = logits_flat
                    [logits_start..logits_start + HYDRA_ACTION_SPACE]
                    .try_into()
                    .expect("logits slice length mismatch");
                let value = values_flat[i];
                (logits, value)
            })
            .collect()
    }
}
