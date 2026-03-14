//! MjaiSample struct, GRP label construction, and batch collation.

use burn::prelude::*;
use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::{NUM_CHANNELS, OBS_SIZE};
use hydra_core::tile::{ALL_PERMUTATIONS, permute_tile_type};

use crate::training::exit::collate_exit_targets;
use crate::training::losses::HydraTargets;

use crate::data::augment::{
    augment_action_suit, augment_action_vector_suit, augment_belief_fields_suit, augment_mask_suit,
    augment_obs_suit,
};

fn permute_tile_vector_34(values: &[f32; 34], perm: &[u8; 3]) -> [f32; 34] {
    let mut out = [0.0f32; 34];
    for (tile, &value) in values.iter().enumerate() {
        let new_tile = permute_tile_type(tile as u8, perm) as usize;
        out[new_tile] = value;
    }
    out
}

fn permute_opp_next_targets(opp_next: [u8; 3], perm: &[u8; 3]) -> [u8; 3] {
    let mut out = opp_next;
    for tile in &mut out {
        if *tile < 34 {
            *tile = permute_tile_type(*tile, perm);
        }
    }
    out
}

fn permute_spatial_targets_3x34(values: [f32; 102], perm: &[u8; 3]) -> [f32; 102] {
    let mut out = [0.0f32; 102];
    for opp in 0..3usize {
        let start = opp * 34;
        let mut chunk = [0.0f32; 34];
        chunk.copy_from_slice(&values[start..start + 34]);
        let permuted = permute_tile_vector_34(&chunk, perm);
        out[start..start + 34].copy_from_slice(&permuted);
    }
    out
}

pub struct MjaiSample {
    pub obs: [f32; OBS_SIZE],
    pub action: u8,
    pub legal_mask: [f32; HYDRA_ACTION_SPACE],
    pub placement: u8,
    pub score_delta: i32,
    pub grp_label: u8,
    pub oracle_target: Option<[f32; 4]>,
    pub tenpai: [f32; 3],
    pub opp_next: [u8; 3],
    pub danger: [f32; 102],
    pub danger_mask: [f32; 102],
    pub safety_residual: Option<[f32; HYDRA_ACTION_SPACE]>,
    pub safety_residual_mask: Option<[f32; HYDRA_ACTION_SPACE]>,
    pub exit_target: Option<[f32; HYDRA_ACTION_SPACE]>,
    pub exit_mask: Option<[f32; HYDRA_ACTION_SPACE]>,
    pub belief_fields: Option<[f32; 16 * 34]>,
    pub mixture_weights: Option<[f32; 4]>,
    pub belief_fields_present: bool,
    pub mixture_weights_present: bool,
}

const SCORE_BIN_MIN: f32 = -50000.0;
const SCORE_BIN_MAX: f32 = 60000.0;
const SCORE_BINS: usize = 64;

pub fn scores_to_grp_index(scores: [i32; 4]) -> Result<u8, &'static str> {
    let mut indexed: [(i32, u8); 4] = [
        (scores[0], 0),
        (scores[1], 1),
        (scores[2], 2),
        (scores[3], 3),
    ];
    indexed.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let ranking = [indexed[0].1, indexed[1].1, indexed[2].1, indexed[3].1];
    GRP_PERM_TABLE
        .iter()
        .position(|p| *p == ranking)
        .map(|i| i as u8)
        .ok_or("invalid ranking permutation")
}

pub const GRP_PERM_TABLE: [[u8; 4]; 24] = generate_perm_table();

const fn generate_perm_table() -> [[u8; 4]; 24] {
    let mut table = [[0u8; 4]; 24];
    let mut idx = 0;
    let mut a = 0u8;
    while a < 4 {
        let mut b = 0u8;
        while b < 4 {
            if b != a {
                let mut c = 0u8;
                while c < 4 {
                    if c != a && c != b {
                        let d = 6 - a - b - c;
                        table[idx] = [a, b, c, d];
                        idx += 1;
                    }
                    c += 1;
                }
            }
            b += 1;
        }
        a += 1;
    }
    table
}

pub fn score_delta_to_bin(score_delta: i32) -> usize {
    let range = SCORE_BIN_MAX - SCORE_BIN_MIN;
    let normalized = (score_delta as f32 - SCORE_BIN_MIN) / range;
    let bin = (normalized * SCORE_BINS as f32) as usize;
    bin.min(SCORE_BINS - 1)
}

pub fn score_delta_to_value(score_delta: i32) -> f32 {
    (score_delta as f32 / 100_000.0).clamp(-1.0, 1.0)
}

pub fn score_delta_to_pdf(score_delta: i32) -> [f32; SCORE_BINS] {
    let mut pdf = [0.0f32; SCORE_BINS];
    pdf[score_delta_to_bin(score_delta)] = 1.0;
    pdf
}

pub fn score_delta_to_cdf(score_delta: i32) -> [f32; SCORE_BINS] {
    let bin = score_delta_to_bin(score_delta);
    let mut cdf = [0.0f32; SCORE_BINS];
    for v in &mut cdf[bin..] {
        *v = 1.0;
    }
    cdf
}

pub struct MjaiBatch<B: Backend> {
    pub obs: Tensor<B, 3>,
    pub actions: Tensor<B, 1, Int>,
    pub legal_mask: Tensor<B, 2>,
    pub value_target: Tensor<B, 1>,
    pub grp_target: Tensor<B, 2>,
    pub oracle_target: Option<Tensor<B, 2>>,
    pub oracle_target_mask: Tensor<B, 1>,
    pub tenpai_target: Tensor<B, 2>,
    pub danger_target: Tensor<B, 3>,
    pub danger_mask: Tensor<B, 3>,
    pub safety_residual_target: Option<Tensor<B, 2>>,
    pub safety_residual_mask: Option<Tensor<B, 2>>,
    pub exit_target: Option<Tensor<B, 2>>,
    pub exit_mask: Option<Tensor<B, 2>>,
    pub belief_fields_target: Option<Tensor<B, 3>>,
    pub mixture_weight_target: Option<Tensor<B, 2>>,
    pub belief_fields_mask: Option<Tensor<B, 1>>,
    pub mixture_weight_mask: Option<Tensor<B, 1>>,
    pub opp_next_target: Tensor<B, 3>,
    pub score_pdf_target: Tensor<B, 2>,
    pub score_cdf_target: Tensor<B, 2>,
}

struct CollateBuffers {
    obs_flat: Vec<f32>,
    actions: Vec<i64>,
    mask_flat: Vec<f32>,
    values: Vec<f32>,
    grp_flat: Vec<f32>,
    oracle_flat: Vec<f32>,
    oracle_mask: Vec<f32>,
    tenpai_flat: Vec<f32>,
    danger_flat: Vec<f32>,
    dmask_flat: Vec<f32>,
    safety_residual_flat: Vec<f32>,
    safety_residual_mask_flat: Vec<f32>,
    any_safety_residual: bool,
    exit_samples: Vec<Option<(Vec<f32>, Vec<f32>)>>,
    belief_fields_flat: Vec<f32>,
    mixture_weights_flat: Vec<f32>,
    any_belief_fields: bool,
    any_mixture_weights: bool,
    belief_fields_mask: Vec<f32>,
    mixture_weight_mask: Vec<f32>,
    opp_flat: Vec<f32>,
    pdf_flat: Vec<f32>,
    cdf_flat: Vec<f32>,
}

impl CollateBuffers {
    fn new(batch: usize) -> Self {
        Self {
            obs_flat: vec![0.0f32; batch * OBS_SIZE],
            actions: vec![0i64; batch],
            mask_flat: vec![0.0f32; batch * HYDRA_ACTION_SPACE],
            values: vec![0.0f32; batch],
            grp_flat: vec![0.0f32; batch * 24],
            oracle_flat: vec![0.0f32; batch * 4],
            oracle_mask: vec![0.0f32; batch],
            tenpai_flat: vec![0.0f32; batch * 3],
            danger_flat: vec![0.0f32; batch * 102],
            dmask_flat: vec![0.0f32; batch * 102],
            safety_residual_flat: vec![0.0f32; batch * HYDRA_ACTION_SPACE],
            safety_residual_mask_flat: vec![0.0f32; batch * HYDRA_ACTION_SPACE],
            any_safety_residual: false,
            exit_samples: vec![None; batch],
            belief_fields_flat: vec![0.0f32; batch * 16 * 34],
            mixture_weights_flat: vec![0.0f32; batch * 4],
            any_belief_fields: false,
            any_mixture_weights: false,
            belief_fields_mask: vec![0.0f32; batch],
            mixture_weight_mask: vec![0.0f32; batch],
            opp_flat: vec![0.0f32; batch * 102],
            pdf_flat: vec![0.0f32; batch * SCORE_BINS],
            cdf_flat: vec![0.0f32; batch * SCORE_BINS],
        }
    }

    fn write_sample(&mut self, index: usize, sample: &MjaiSample, perm: Option<&[u8; 3]>) {
        let obs = perm.map_or(sample.obs, |perm| augment_obs_suit(&sample.obs, perm));
        let action = perm.map_or(sample.action, |perm| {
            augment_action_suit(sample.action, perm)
        });
        let legal_mask = perm.map_or(sample.legal_mask, |perm| {
            augment_mask_suit(&sample.legal_mask, perm)
        });
        let opp_next = perm.map_or(sample.opp_next, |perm| {
            permute_opp_next_targets(sample.opp_next, perm)
        });
        let danger = perm.map_or(sample.danger, |perm| {
            permute_spatial_targets_3x34(sample.danger, perm)
        });
        let danger_mask = perm.map_or(sample.danger_mask, |perm| {
            permute_spatial_targets_3x34(sample.danger_mask, perm)
        });
        let safety_residual = match (sample.safety_residual, perm) {
            (Some(values), Some(perm)) => Some(augment_action_vector_suit(&values, perm)),
            (Some(values), None) => Some(values),
            (None, _) => None,
        };
        let safety_residual_mask = match (sample.safety_residual_mask, perm) {
            (Some(values), Some(perm)) => Some(augment_action_vector_suit(&values, perm)),
            (Some(values), None) => Some(values),
            (None, _) => None,
        };
        let belief_fields = match (sample.belief_fields, perm) {
            (Some(values), Some(perm)) => Some(augment_belief_fields_suit(&values, perm)),
            (Some(values), None) => Some(values),
            (None, _) => None,
        };
        let exit_target = match (sample.exit_target, perm) {
            (Some(values), Some(perm)) => Some(augment_action_vector_suit(&values, perm)),
            (Some(values), None) => Some(values),
            (None, _) => None,
        };
        let exit_mask = match (sample.exit_mask, perm) {
            (Some(values), Some(perm)) => Some(augment_action_vector_suit(&values, perm)),
            (Some(values), None) => Some(values),
            (None, _) => None,
        };

        self.obs_flat[index * OBS_SIZE..(index + 1) * OBS_SIZE].copy_from_slice(&obs);
        self.actions[index] = action as i64;
        self.mask_flat[index * HYDRA_ACTION_SPACE..(index + 1) * HYDRA_ACTION_SPACE]
            .copy_from_slice(&legal_mask);
        self.values[index] = score_delta_to_value(sample.score_delta);
        if (sample.grp_label as usize) < 24 {
            self.grp_flat[index * 24 + sample.grp_label as usize] = 1.0;
        }
        if let Some(oracle) = sample.oracle_target {
            self.oracle_flat[index * 4..(index + 1) * 4].copy_from_slice(&oracle);
            self.oracle_mask[index] = 1.0;
        }
        self.tenpai_flat[index * 3..(index + 1) * 3].copy_from_slice(&sample.tenpai);
        self.danger_flat[index * 102..(index + 1) * 102].copy_from_slice(&danger);
        self.dmask_flat[index * 102..(index + 1) * 102].copy_from_slice(&danger_mask);
        if let Some(values) = safety_residual {
            self.safety_residual_flat[index * HYDRA_ACTION_SPACE..(index + 1) * HYDRA_ACTION_SPACE]
                .copy_from_slice(&values);
            self.any_safety_residual = true;
        }
        if let Some(values) = safety_residual_mask {
            self.safety_residual_mask_flat
                [index * HYDRA_ACTION_SPACE..(index + 1) * HYDRA_ACTION_SPACE]
                .copy_from_slice(&values);
            self.any_safety_residual = true;
        }
        self.exit_samples[index] = match (exit_target, exit_mask) {
            (Some(target), Some(mask)) => Some((target.to_vec(), mask.to_vec())),
            _ => None,
        };
        if let Some(values) = belief_fields {
            self.belief_fields_flat[index * 16 * 34..(index + 1) * 16 * 34]
                .copy_from_slice(&values);
            self.any_belief_fields = true;
        }
        if sample.belief_fields_present {
            self.belief_fields_mask[index] = 1.0;
            self.any_belief_fields = true;
        }
        if let Some(values) = sample.mixture_weights {
            self.mixture_weights_flat[index * 4..(index + 1) * 4].copy_from_slice(&values);
            self.any_mixture_weights = true;
        }
        if sample.mixture_weights_present {
            self.mixture_weight_mask[index] = 1.0;
            self.any_mixture_weights = true;
        }
        for (opp, tile) in opp_next.iter().copied().enumerate() {
            if tile < 34 {
                self.opp_flat[index * 102 + opp * 34 + tile as usize] = 1.0;
            }
        }
        let pdf = score_delta_to_pdf(sample.score_delta);
        self.pdf_flat[index * SCORE_BINS..(index + 1) * SCORE_BINS].copy_from_slice(&pdf);
        let cdf = score_delta_to_cdf(sample.score_delta);
        self.cdf_flat[index * SCORE_BINS..(index + 1) * SCORE_BINS].copy_from_slice(&cdf);
    }

    fn into_batch<B: Backend>(self, batch: usize, device: &B::Device) -> MjaiBatch<B> {
        let (exit_target, exit_mask) = collate_exit_targets::<B>(&self.exit_samples, device);
        MjaiBatch {
            obs: Tensor::<B, 1>::from_floats(self.obs_flat.as_slice(), device).reshape([
                batch,
                NUM_CHANNELS,
                34,
            ]),
            actions: Tensor::<B, 1, Int>::from_ints(self.actions.as_slice(), device),
            legal_mask: Tensor::<B, 1>::from_floats(self.mask_flat.as_slice(), device)
                .reshape([batch, HYDRA_ACTION_SPACE]),
            value_target: Tensor::<B, 1>::from_floats(self.values.as_slice(), device),
            grp_target: Tensor::<B, 1>::from_floats(self.grp_flat.as_slice(), device)
                .reshape([batch, 24]),
            oracle_target: if self.oracle_mask.iter().any(|&v| v > 0.0) {
                Some(
                    Tensor::<B, 1>::from_floats(self.oracle_flat.as_slice(), device)
                        .reshape([batch, 4]),
                )
            } else {
                None
            },
            oracle_target_mask: Tensor::<B, 1>::from_floats(self.oracle_mask.as_slice(), device),
            tenpai_target: Tensor::<B, 1>::from_floats(self.tenpai_flat.as_slice(), device)
                .reshape([batch, 3]),
            danger_target: Tensor::<B, 1>::from_floats(self.danger_flat.as_slice(), device)
                .reshape([batch, 3, 34]),
            danger_mask: Tensor::<B, 1>::from_floats(self.dmask_flat.as_slice(), device)
                .reshape([batch, 3, 34]),
            safety_residual_target: if self.any_safety_residual {
                Some(
                    Tensor::<B, 1>::from_floats(self.safety_residual_flat.as_slice(), device)
                        .reshape([batch, HYDRA_ACTION_SPACE]),
                )
            } else {
                None
            },
            safety_residual_mask: if self.any_safety_residual {
                Some(
                    Tensor::<B, 1>::from_floats(self.safety_residual_mask_flat.as_slice(), device)
                        .reshape([batch, HYDRA_ACTION_SPACE]),
                )
            } else {
                None
            },
            exit_target,
            exit_mask,
            belief_fields_target: if self.any_belief_fields {
                Some(
                    Tensor::<B, 1>::from_floats(self.belief_fields_flat.as_slice(), device)
                        .reshape([batch, 16, 34]),
                )
            } else {
                None
            },
            mixture_weight_target: if self.any_mixture_weights {
                Some(
                    Tensor::<B, 1>::from_floats(self.mixture_weights_flat.as_slice(), device)
                        .reshape([batch, 4]),
                )
            } else {
                None
            },
            belief_fields_mask: if self.any_belief_fields {
                Some(Tensor::<B, 1>::from_floats(
                    self.belief_fields_mask.as_slice(),
                    device,
                ))
            } else {
                None
            },
            mixture_weight_mask: if self.any_mixture_weights {
                Some(Tensor::<B, 1>::from_floats(
                    self.mixture_weight_mask.as_slice(),
                    device,
                ))
            } else {
                None
            },
            opp_next_target: Tensor::<B, 1>::from_floats(self.opp_flat.as_slice(), device)
                .reshape([batch, 3, 34]),
            score_pdf_target: Tensor::<B, 1>::from_floats(self.pdf_flat.as_slice(), device)
                .reshape([batch, SCORE_BINS]),
            score_cdf_target: Tensor::<B, 1>::from_floats(self.cdf_flat.as_slice(), device)
                .reshape([batch, SCORE_BINS]),
        }
    }
}

impl<B: Backend> MjaiBatch<B> {
    pub fn into_hydra_targets(self) -> HydraTargets<B> {
        let batch = self.actions.dims()[0];
        let policy_target = self
            .actions
            .clone()
            .one_hot::<2>(46)
            .reshape([batch, 46])
            .float();
        HydraTargets {
            policy_target,
            legal_mask: self.legal_mask,
            value_target: self.value_target,
            grp_target: self.grp_target,
            tenpai_target: self.tenpai_target,
            danger_target: self.danger_target,
            danger_mask: self.danger_mask,
            safety_residual_target: self.safety_residual_target,
            opp_next_target: self.opp_next_target,
            score_pdf_target: self.score_pdf_target,
            score_cdf_target: self.score_cdf_target,
            oracle_target: self.oracle_target,
            belief_fields_target: self.belief_fields_target,
            mixture_weight_target: self.mixture_weight_target,
            opponent_hand_type_target: None,
            delta_q_target: None,
            safety_residual_mask: self.safety_residual_mask,
            belief_fields_mask: self.belief_fields_mask,
            mixture_weight_mask: self.mixture_weight_mask,
            oracle_guidance_mask: Some(self.oracle_target_mask),
        }
    }

    pub fn to_hydra_targets(&self) -> HydraTargets<B> {
        let batch = self.actions.dims()[0];
        let policy_target = self
            .actions
            .clone()
            .one_hot::<2>(46)
            .reshape([batch, 46])
            .float();
        HydraTargets {
            policy_target,
            legal_mask: self.legal_mask.clone(),
            value_target: self.value_target.clone(),
            grp_target: self.grp_target.clone(),
            tenpai_target: self.tenpai_target.clone(),
            danger_target: self.danger_target.clone(),
            danger_mask: self.danger_mask.clone(),
            safety_residual_target: self.safety_residual_target.clone(),
            opp_next_target: self.opp_next_target.clone(),
            score_pdf_target: self.score_pdf_target.clone(),
            score_cdf_target: self.score_cdf_target.clone(),
            oracle_target: self.oracle_target.clone(),
            belief_fields_target: self.belief_fields_target.clone(),
            mixture_weight_target: self.mixture_weight_target.clone(),
            opponent_hand_type_target: None,
            delta_q_target: None,
            safety_residual_mask: self.safety_residual_mask.clone(),
            belief_fields_mask: self.belief_fields_mask.clone(),
            mixture_weight_mask: self.mixture_weight_mask.clone(),
            oracle_guidance_mask: Some(self.oracle_target_mask.clone()),
        }
    }
}

pub fn collate_batch<B: Backend>(samples: &[MjaiSample], device: &B::Device) -> MjaiBatch<B> {
    let batch = samples.len();
    let mut buffers = CollateBuffers::new(batch);
    for (i, s) in samples.iter().enumerate() {
        buffers.write_sample(i, s, None);
    }
    buffers.into_batch(batch, device)
}

pub fn score_to_placement(scores: [i32; 4], player: u8) -> u8 {
    let mut indexed: Vec<(i32, u8)> = scores
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, i as u8))
        .collect();
    indexed.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    indexed.iter().position(|(_, p)| *p == player).unwrap_or(3) as u8
}

pub fn one_hot_action(action: u8, num_classes: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; num_classes];
    if (action as usize) < num_classes {
        v[action as usize] = 1.0;
    }
    v
}

pub fn collate_batch_augmented<B: Backend>(
    samples: &[MjaiSample],
    device: &B::Device,
) -> MjaiBatch<B> {
    let batch = samples.len() * ALL_PERMUTATIONS.len();
    let mut buffers = CollateBuffers::new(batch);
    let mut index = 0usize;
    for sample in samples {
        for perm in &ALL_PERMUTATIONS {
            buffers.write_sample(index, sample, Some(perm));
            index += 1;
        }
    }
    buffers.into_batch(batch, device)
}

pub fn collate_sample_refs<B: Backend>(
    samples: &[&MjaiSample],
    augment: bool,
    device: &B::Device,
) -> Option<(Tensor<B, 3>, HydraTargets<B>)> {
    let (obs, batch) = collate_sample_refs_with_batch::<B>(samples, augment, device)?;
    Some((obs, batch.into_hydra_targets()))
}

pub fn collate_sample_refs_with_batch<B: Backend>(
    samples: &[&MjaiSample],
    augment: bool,
    device: &B::Device,
) -> Option<(Tensor<B, 3>, MjaiBatch<B>)> {
    if samples.is_empty() {
        return None;
    }

    let batch = if augment {
        let batch = samples.len() * ALL_PERMUTATIONS.len();
        let mut buffers = CollateBuffers::new(batch);
        let mut index = 0usize;
        for sample in samples {
            for perm in &ALL_PERMUTATIONS {
                buffers.write_sample(index, sample, Some(perm));
                index += 1;
            }
        }
        buffers.into_batch(batch, device)
    } else {
        let batch = samples.len();
        let mut buffers = CollateBuffers::new(batch);
        for (index, sample) in samples.iter().enumerate() {
            buffers.write_sample(index, sample, None);
        }
        buffers.into_batch(batch, device)
    };
    let obs = batch.obs.clone();
    Some((obs, batch))
}

pub fn collate_samples<B: Backend>(
    samples: &[MjaiSample],
    augment: bool,
    device: &B::Device,
) -> Option<(Tensor<B, 3>, HydraTargets<B>)> {
    let (obs, batch) = collate_batch_samples::<B>(samples, augment, device)?;
    Some((obs, batch.into_hydra_targets()))
}

pub fn collate_batch_samples<B: Backend>(
    samples: &[MjaiSample],
    augment: bool,
    device: &B::Device,
) -> Option<(Tensor<B, 3>, MjaiBatch<B>)> {
    if samples.is_empty() {
        return None;
    }

    let batch = if augment {
        collate_batch_augmented(samples, device)
    } else {
        collate_batch(samples, device)
    };
    let obs = batch.obs.clone();
    Some((obs, batch))
}

pub fn augment_samples_6x(samples: &[MjaiSample]) -> Vec<MjaiSample> {
    use crate::data::augment::{
        augment_action_suit, augment_belief_fields_suit, augment_mask_suit, augment_obs_suit,
    };
    use hydra_core::tile::ALL_PERMUTATIONS;

    let mut augmented = Vec::with_capacity(samples.len() * 6);
    for sample in samples {
        for perm in &ALL_PERMUTATIONS {
            let obs = augment_obs_suit(&sample.obs, perm);
            let action = augment_action_suit(sample.action, perm);
            let legal_mask = augment_mask_suit(&sample.legal_mask, perm);
            augmented.push(MjaiSample {
                obs,
                action,
                legal_mask,
                placement: sample.placement,
                score_delta: sample.score_delta,
                grp_label: sample.grp_label,
                oracle_target: sample.oracle_target,
                tenpai: sample.tenpai,
                opp_next: permute_opp_next_targets(sample.opp_next, perm),
                danger: permute_spatial_targets_3x34(sample.danger, perm),
                danger_mask: permute_spatial_targets_3x34(sample.danger_mask, perm),
                safety_residual: sample
                    .safety_residual
                    .map(|values| crate::data::augment::augment_action_vector_suit(&values, perm)),
                safety_residual_mask: sample
                    .safety_residual_mask
                    .map(|values| crate::data::augment::augment_action_vector_suit(&values, perm)),
                exit_target: sample
                    .exit_target
                    .map(|values| crate::data::augment::augment_action_vector_suit(&values, perm)),
                exit_mask: sample
                    .exit_mask
                    .map(|values| crate::data::augment::augment_action_vector_suit(&values, perm)),
                belief_fields: sample
                    .belief_fields
                    .map(|values| augment_belief_fields_suit(&values, perm)),
                mixture_weights: sample.mixture_weights,
                belief_fields_present: sample.belief_fields_present,
                mixture_weights_present: sample.mixture_weights_present,
            });
        }
    }
    augmented
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray<f32>;

    fn dummy_sample(action: u8, score_delta: i32) -> MjaiSample {
        let mut legal_mask = [0.0f32; HYDRA_ACTION_SPACE];
        legal_mask[action as usize] = 1.0;
        legal_mask[45] = 1.0;
        MjaiSample {
            obs: [0.1f32; OBS_SIZE],
            action,
            legal_mask,
            placement: 0,
            score_delta,
            grp_label: 0,
            oracle_target: None,
            tenpai: [0.0; 3],
            opp_next: [0, 1, 255],
            danger: [0.0; 102],
            danger_mask: [1.0; 102],
            safety_residual: None,
            safety_residual_mask: None,
            exit_target: None,
            exit_mask: None,
            belief_fields: None,
            mixture_weights: None,
            belief_fields_present: false,
            mixture_weights_present: false,
        }
    }

    #[test]
    fn test_grp_index_sorted() {
        assert_eq!(
            scores_to_grp_index([40000, 30000, 20000, 10000]).unwrap(),
            0
        );
    }

    #[test]
    fn test_grp_index_reversed() {
        let idx = scores_to_grp_index([10000, 20000, 30000, 40000]).unwrap();
        assert_ne!(idx, 0);
        assert!(idx < 24);
    }

    #[test]
    fn test_grp_perm_table_has_24_unique() {
        let mut seen = std::collections::HashSet::new();
        for perm in &GRP_PERM_TABLE {
            assert!(seen.insert(*perm), "duplicate perm {perm:?}");
        }
        assert_eq!(seen.len(), 24);
    }

    #[test]
    fn test_grp_all_tie_scores() {
        let idx = scores_to_grp_index([25000, 25000, 25000, 25000]).unwrap();
        assert!(idx < 24);
    }

    #[test]
    fn test_score_bin_boundaries() {
        assert_eq!(score_delta_to_bin(-50000), 0);
        assert_eq!(score_delta_to_bin(60000), SCORE_BINS - 1);
        let mid = score_delta_to_bin(5000);
        assert!(mid > 0 && mid < SCORE_BINS - 1);
    }

    #[test]
    fn test_batch_shapes() {
        let device = Default::default();
        let samples: Vec<_> = (0..32)
            .map(|i| dummy_sample(i % 34, 1000 * i as i32))
            .collect();
        let batch = collate_batch::<B>(&samples, &device);
        assert_eq!(batch.obs.dims(), [32, NUM_CHANNELS, 34]);
        assert_eq!(batch.actions.dims(), [32]);
        assert_eq!(batch.legal_mask.dims(), [32, 46]);
        assert_eq!(batch.value_target.dims(), [32]);
        assert_eq!(batch.grp_target.dims(), [32, 24]);
        assert!(batch.oracle_target.is_none());
        assert_eq!(batch.oracle_target_mask.dims(), [32]);
        assert_eq!(batch.tenpai_target.dims(), [32, 3]);
        assert_eq!(batch.danger_target.dims(), [32, 3, 34]);
        assert_eq!(batch.danger_mask.dims(), [32, 3, 34]);
        assert!(batch.safety_residual_target.is_none());
        assert!(batch.safety_residual_mask.is_none());
        assert_eq!(batch.opp_next_target.dims(), [32, 3, 34]);
        assert_eq!(batch.score_pdf_target.dims(), [32, 64]);
        assert_eq!(batch.score_cdf_target.dims(), [32, 64]);
    }

    #[test]
    fn test_score_pdf_is_one_hot() {
        let pdf = score_delta_to_pdf(5000);
        let sum: f32 = pdf.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "pdf should sum to 1");
        let nonzero = pdf.iter().filter(|&&v| v > 0.0).count();
        assert_eq!(nonzero, 1, "pdf should be one-hot");
    }

    #[test]
    fn test_score_cdf_monotonic() {
        let cdf = score_delta_to_cdf(5000);
        for i in 1..64 {
            assert!(cdf[i] >= cdf[i - 1], "cdf not monotonic at {i}");
        }
        assert!((cdf[63] - 1.0).abs() < 1e-5, "cdf should end at 1");
    }

    #[test]
    fn test_value_target_range() {
        assert!((score_delta_to_value(0) - 0.0).abs() < 1e-5);
        assert!((score_delta_to_value(100_000) - 1.0).abs() < 1e-5);
        assert!((score_delta_to_value(-100_000) - (-1.0)).abs() < 1e-5);
        let mid = score_delta_to_value(50_000);
        assert!(mid > 0.0 && mid < 1.0);
    }

    #[test]
    fn test_legal_mask_valid() {
        let device = Default::default();
        let samples: Vec<_> = (0..4).map(|_| dummy_sample(0, 0)).collect();
        let batch = collate_batch::<B>(&samples, &device);
        let mask_data = batch.legal_mask.to_data();
        let mask_slice = mask_data.as_slice::<f32>().expect("f32");
        for row in mask_slice.chunks(46) {
            let sum: f32 = row.iter().sum();
            assert!(sum > 0.0, "all-zero mask found");
        }
    }

    #[test]
    fn test_opp_next_255_is_zero() {
        let device = Default::default();
        let samples = vec![dummy_sample(0, 0)];
        let batch = collate_batch::<B>(&samples, &device);
        let data = batch.opp_next_target.to_data();
        let slice = data.as_slice::<f32>().expect("f32");
        let opp2_start = 2 * 34;
        let opp2_sum: f32 = slice[opp2_start..opp2_start + 34].iter().sum();
        assert!(
            opp2_sum.abs() < 1e-5,
            "opp_next=255 should be all zero, sum={opp2_sum}"
        );
    }

    #[test]
    fn test_single_sample_batch() {
        let device = Default::default();
        let samples = vec![dummy_sample(5, 12000)];
        let batch = collate_batch::<B>(&samples, &device);
        assert_eq!(batch.obs.dims(), [1, NUM_CHANNELS, 34]);
        assert_eq!(batch.actions.dims(), [1]);
        let action_data = batch.actions.to_data();
        assert_eq!(action_data.as_slice::<i64>().expect("i64")[0], 5);
    }

    #[test]
    fn test_extreme_score_deltas() {
        let device = Default::default();
        let samples = vec![
            dummy_sample(0, -100_000),
            dummy_sample(1, 100_000),
            dummy_sample(2, 0),
        ];
        let batch = collate_batch::<B>(&samples, &device);
        let val_data = batch.value_target.to_data();
        let vals = val_data.as_slice::<f32>().expect("f32");
        assert!((vals[0] - (-1.0)).abs() < 1e-5);
        assert!((vals[1] - 1.0).abs() < 1e-5);
        assert!((vals[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_score_to_placement() {
        assert_eq!(score_to_placement([40000, 30000, 20000, 10000], 0), 0);
        assert_eq!(score_to_placement([40000, 30000, 20000, 10000], 3), 3);
        assert_eq!(score_to_placement([25000, 25000, 25000, 25000], 0), 0);
    }

    #[test]
    fn augment_samples_6x_permutes_aux_tile_targets() {
        use hydra_core::tile::ALL_PERMUTATIONS;

        let mut sample = dummy_sample(0, 0);
        sample.obs = [0.0; OBS_SIZE];
        sample.opp_next = [0, 9, 27];
        sample.danger[0] = 0.25;
        sample.danger[34 + 9] = 0.5;
        sample.danger_mask[18] = 1.0;
        let mut safety_residual = [0.0f32; HYDRA_ACTION_SPACE];
        let mut safety_residual_mask = [0.0f32; HYDRA_ACTION_SPACE];
        safety_residual[0] = -0.75;
        safety_residual[1] = 0.4;
        safety_residual_mask[0] = 1.0;
        safety_residual_mask[1] = 1.0;
        sample.safety_residual = Some(safety_residual);
        sample.safety_residual_mask = Some(safety_residual_mask);
        sample.obs[40 * 34] = 1.0;

        let augmented = augment_samples_6x(&[sample]);
        let swap_mp = &ALL_PERMUTATIONS[2];
        let swapped = augmented
            .iter()
            .find(|s| s.action == 9)
            .expect("swap man-pin permutation sample");

        assert_eq!(permute_tile_type(0, swap_mp), 9);
        assert_eq!(swapped.opp_next, [9, 0, 27]);
        assert!((swapped.danger[9] - 0.25).abs() < 1e-6);
        assert!((swapped.danger[34] - 0.5).abs() < 1e-6);
        assert_eq!(swapped.danger_mask[18], 1.0);
        let sr = swapped.safety_residual.expect("safety residual target");
        let srm = swapped.safety_residual_mask.expect("safety residual mask");
        assert!((sr[9] + 0.75).abs() < 1e-6);
        assert!((sr[10] - 0.4).abs() < 1e-6);
        assert!((srm[9] - 1.0).abs() < 1e-6);
        assert!((srm[10] - 1.0).abs() < 1e-6);
        assert_eq!(swapped.obs[41 * 34], 1.0);
        assert_eq!(swapped.obs[40 * 34], 0.0);
    }

    #[test]
    fn batch_to_hydra_targets_carries_oracle_target() {
        let device = Default::default();
        let mut sample = dummy_sample(5, 12000);
        sample.oracle_target = Some([0.1, -0.1, 0.2, -0.2]);
        let batch = collate_batch::<B>(&[sample], &device);
        let targets = batch.to_hydra_targets();
        assert_eq!(targets.policy_target.dims(), [1, 46]);
        let oracle = targets.oracle_target.expect("oracle target present");
        assert_eq!(oracle.dims(), [1, 4]);
        let data = oracle.to_data();
        let slice = data.as_slice::<f32>().expect("f32");
        assert!((slice[0] - 0.1).abs() < 1e-6);
        assert!((slice[1] + 0.1).abs() < 1e-6);
        assert!((slice[2] - 0.2).abs() < 1e-6);
        assert!((slice[3] + 0.2).abs() < 1e-6);
        let mask = targets.oracle_guidance_mask.expect("oracle mask present");
        let mask_slice = mask.to_data().as_slice::<f32>().expect("f32").to_vec();
        assert!((mask_slice[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn batch_to_hydra_targets_policy_matches_actions() {
        let device = Default::default();
        let samples = vec![dummy_sample(2, 0), dummy_sample(7, 0)];
        let batch = collate_batch::<B>(&samples, &device);
        let targets = batch.into_hydra_targets();
        assert_eq!(targets.policy_target.dims(), [2, 46]);
        let data = targets.policy_target.to_data();
        let slice = data.as_slice::<f32>().expect("f32");
        assert!((slice[2] - 1.0).abs() < 1e-6);
        assert!((slice[46 + 7] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn batch_to_hydra_targets_keeps_optional_advanced_targets_narrow() {
        let device = Default::default();
        let mut sample = dummy_sample(1, 0);
        sample.oracle_target = Some([0.25, 0.0, -0.25, 0.0]);
        let batch = collate_batch::<B>(&[sample], &device);
        let targets = batch.into_hydra_targets();
        assert!(targets.oracle_target.is_some());
        assert!(targets.belief_fields_target.is_none());
        assert!(targets.mixture_weight_target.is_none());
        assert!(targets.opponent_hand_type_target.is_none());
        assert!(targets.delta_q_target.is_none());
        assert!(targets.safety_residual_target.is_none());
    }

    #[test]
    fn batch_to_hydra_targets_keeps_oracle_absent_when_missing() {
        let device = Default::default();
        let batch = collate_batch::<B>(&[dummy_sample(3, 0)], &device);
        assert!(batch.oracle_target.is_none());
        let targets = batch.into_hydra_targets();
        assert!(targets.oracle_target.is_none());
        let mask = targets.oracle_guidance_mask.expect("oracle mask present");
        let mask_slice = mask.to_data().as_slice::<f32>().expect("f32").to_vec();
        assert!((mask_slice[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn batch_to_hydra_targets_carries_safety_residual() {
        let device = Default::default();
        let mut sample = dummy_sample(0, 0);
        let mut target = [0.0f32; HYDRA_ACTION_SPACE];
        let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
        target[0] = -0.4;
        target[34] = 0.7;
        mask[0] = 1.0;
        mask[34] = 1.0;
        sample.safety_residual = Some(target);
        sample.safety_residual_mask = Some(mask);
        let batch = collate_batch::<B>(&[sample], &device);
        let targets = batch.into_hydra_targets();
        let sr = targets
            .safety_residual_target
            .expect("safety residual target");
        let srm = targets.safety_residual_mask.expect("safety residual mask");
        assert_eq!(sr.dims(), [1, HYDRA_ACTION_SPACE]);
        assert_eq!(srm.dims(), [1, HYDRA_ACTION_SPACE]);
        let values = sr.to_data().as_slice::<f32>().expect("f32").to_vec();
        let mask_values = srm.to_data().as_slice::<f32>().expect("f32").to_vec();
        assert!((values[0] + 0.4).abs() < 1e-6);
        assert!((values[34] - 0.7).abs() < 1e-6);
        assert!((mask_values[0] - 1.0).abs() < 1e-6);
        assert!((mask_values[34] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn batch_to_hydra_targets_carries_projected_belief_targets() {
        let device = Default::default();
        let mut sample = dummy_sample(0, 0);
        let mut belief = [0.0f32; 16 * 34];
        let mut mix = [0.0f32; 4];
        belief[0] = 0.2;
        belief[33] = 0.8;
        mix[0] = 0.7;
        mix[1] = 0.3;
        sample.belief_fields = Some(belief);
        sample.mixture_weights = Some(mix);
        sample.belief_fields_present = true;
        sample.mixture_weights_present = true;
        let batch = collate_batch::<B>(&[sample], &device);
        let targets = batch.into_hydra_targets();
        let belief_target = targets
            .belief_fields_target
            .expect("belief field target should be present");
        let mix_target = targets
            .mixture_weight_target
            .expect("mixture weights should be present");
        assert_eq!(belief_target.dims(), [1, 16, 34]);
        assert_eq!(mix_target.dims(), [1, 4]);
        let belief_values = belief_target
            .to_data()
            .as_slice::<f32>()
            .expect("f32")
            .to_vec();
        let mix_values = mix_target
            .to_data()
            .as_slice::<f32>()
            .expect("f32")
            .to_vec();
        let belief_mask = targets.belief_fields_mask.expect("belief mask");
        let mixture_mask = targets.mixture_weight_mask.expect("mixture mask");
        let belief_mask_values = belief_mask
            .to_data()
            .as_slice::<f32>()
            .expect("f32")
            .to_vec();
        let mixture_mask_values = mixture_mask
            .to_data()
            .as_slice::<f32>()
            .expect("f32")
            .to_vec();
        assert!((belief_values[0] - 0.2).abs() < 1e-6);
        assert!((belief_values[33] - 0.8).abs() < 1e-6);
        assert!((mix_values[0] - 0.7).abs() < 1e-6);
        assert!((mix_values[1] - 0.3).abs() < 1e-6);
        assert!((belief_mask_values[0] - 1.0).abs() < 1e-6);
        assert!((mixture_mask_values[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn batch_to_hydra_targets_keeps_belief_targets_absent_when_missing() {
        let device = Default::default();
        let batch = collate_batch::<B>(&[dummy_sample(0, 0)], &device);
        let targets = batch.into_hydra_targets();
        assert!(targets.belief_fields_target.is_none());
        assert!(targets.mixture_weight_target.is_none());
        assert!(targets.belief_fields_mask.is_none());
        assert!(targets.mixture_weight_mask.is_none());
    }

    #[test]
    fn augment_samples_6x_permutes_belief_fields_and_preserves_mixture_weights() {
        use hydra_core::tile::ALL_PERMUTATIONS;

        let mut sample = dummy_sample(0, 0);
        let mut belief = [0.0f32; 16 * 34];
        belief[0] = 1.0;
        let mut mix = [0.0f32; 4];
        mix[0] = 0.8;
        sample.belief_fields = Some(belief);
        sample.belief_fields_present = true;
        sample.mixture_weights = Some(mix);
        sample.mixture_weights_present = true;

        let augmented = augment_samples_6x(&[sample]);
        let swap_mp = &ALL_PERMUTATIONS[2];
        let swapped = augmented
            .iter()
            .find(|s| s.action == 9)
            .expect("swap man-pin permutation sample");
        let swapped_belief = swapped.belief_fields.expect("belief fields");
        let swapped_mix = swapped.mixture_weights.expect("mixture weights");
        assert_eq!(permute_tile_type(0, swap_mp), 9);
        assert!((swapped_belief[9] - 1.0).abs() < 1e-6);
        assert!((swapped_mix[0] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn collate_sample_refs_matches_owned_collation_without_augmentation() {
        let device = Default::default();
        let samples = vec![dummy_sample(2, 100), dummy_sample(7, -500)];
        let refs: Vec<_> = samples.iter().collect();

        let (obs, targets) =
            collate_sample_refs::<B>(&refs, false, &device).expect("borrowed collate");
        let (owned_obs, owned_targets) =
            collate_samples::<B>(&samples, false, &device).expect("owned collate");

        assert_eq!(obs.dims(), owned_obs.dims());
        assert_eq!(
            targets.policy_target.dims(),
            owned_targets.policy_target.dims()
        );
        assert_eq!(targets.legal_mask.dims(), owned_targets.legal_mask.dims());
        assert_eq!(
            targets.danger_target.dims(),
            owned_targets.danger_target.dims()
        );
        assert_eq!(obs.to_data(), owned_obs.to_data());
        assert_eq!(
            targets.policy_target.to_data(),
            owned_targets.policy_target.to_data()
        );
        assert_eq!(
            targets.legal_mask.to_data(),
            owned_targets.legal_mask.to_data()
        );
        assert_eq!(
            targets.value_target.to_data(),
            owned_targets.value_target.to_data()
        );
        assert_eq!(
            targets.grp_target.to_data(),
            owned_targets.grp_target.to_data()
        );
        assert_eq!(
            targets.danger_target.to_data(),
            owned_targets.danger_target.to_data()
        );
        assert_eq!(
            targets.danger_mask.to_data(),
            owned_targets.danger_mask.to_data()
        );
        assert_eq!(
            targets.opp_next_target.to_data(),
            owned_targets.opp_next_target.to_data()
        );
        assert_eq!(
            targets.score_pdf_target.to_data(),
            owned_targets.score_pdf_target.to_data()
        );
        assert_eq!(
            targets.score_cdf_target.to_data(),
            owned_targets.score_cdf_target.to_data()
        );
    }

    #[test]
    fn collate_sample_refs_matches_owned_collation_with_augmentation() {
        let device = Default::default();
        let mut sample = dummy_sample(0, 0);
        sample.obs = [0.0; OBS_SIZE];
        sample.obs[40 * 34] = 1.0;
        sample.opp_next = [0, 9, 27];
        sample.danger[0] = 0.25;
        sample.danger[34 + 9] = 0.5;
        sample.danger_mask[18] = 1.0;
        let mut safety_residual = [0.0f32; HYDRA_ACTION_SPACE];
        let mut safety_residual_mask = [0.0f32; HYDRA_ACTION_SPACE];
        safety_residual[0] = -0.75;
        safety_residual[1] = 0.4;
        safety_residual_mask[0] = 1.0;
        safety_residual_mask[1] = 1.0;
        sample.safety_residual = Some(safety_residual);
        sample.safety_residual_mask = Some(safety_residual_mask);

        let refs = vec![&sample];
        let (obs, targets) =
            collate_sample_refs::<B>(&refs, true, &device).expect("borrowed collate");
        let (owned_obs, owned_targets) =
            collate_samples::<B>(&[sample], true, &device).expect("owned collate");

        assert_eq!(obs.dims(), owned_obs.dims());
        assert_eq!(
            targets.policy_target.to_data(),
            owned_targets.policy_target.to_data()
        );
        assert_eq!(
            targets.legal_mask.to_data(),
            owned_targets.legal_mask.to_data()
        );
        assert_eq!(
            targets.danger_target.to_data(),
            owned_targets.danger_target.to_data()
        );
        assert_eq!(
            targets.danger_mask.to_data(),
            owned_targets.danger_mask.to_data()
        );
        assert_eq!(
            targets.opp_next_target.to_data(),
            owned_targets.opp_next_target.to_data()
        );
        assert_eq!(
            targets.score_pdf_target.to_data(),
            owned_targets.score_pdf_target.to_data()
        );
        assert_eq!(
            targets.score_cdf_target.to_data(),
            owned_targets.score_cdf_target.to_data()
        );
        assert_eq!(
            targets
                .safety_residual_target
                .expect("borrowed safety residual")
                .to_data(),
            owned_targets
                .safety_residual_target
                .expect("owned safety residual")
                .to_data()
        );
        assert_eq!(
            targets
                .safety_residual_mask
                .expect("borrowed safety residual mask")
                .to_data(),
            owned_targets
                .safety_residual_mask
                .expect("owned safety residual mask")
                .to_data()
        );
    }
}
