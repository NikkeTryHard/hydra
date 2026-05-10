//! MjaiSample struct, GRP label construction, and batch collation.

use burn::prelude::*;
use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::{NUM_CHANNELS, OBS_SIZE};
use hydra_core::tile::ALL_PERMUTATIONS;
pub use hydra_data_core::sample::{
    GRP_PERM_TABLE, MjaiSample, SCORE_BINS, one_hot_action, score_delta_to_bin, score_delta_to_cdf,
    score_delta_to_pdf, score_delta_to_value, score_to_placement, score_to_placements,
    scores_to_grp_index,
};
use std::cell::RefCell;

use crate::data::sample_targets::collate_action_targets;
use crate::data::sample_targets::{
    cloned_hydra_targets, into_bc_batch_and_hydra_targets_inner, into_hydra_targets_inner,
};
use hydra_train_types::head_gates::{AdvancedHead, TargetPresence};
use hydra_train_types::losses::HydraTargets;

use crate::data::augment::{
    augment_action_suit, augment_action_vector_suit, augment_belief_fields_suit, augment_mask_suit,
    augment_obs_suit, permutation_index, permutation_tables,
};

fn permute_tile_vector_34(values: &[f32; 34], perm: &[u8; 3]) -> [f32; 34] {
    let mut out = [0.0f32; 34];
    let tile_perm = &permutation_tables().tile_34[permutation_index(perm)];
    for (tile, &value) in values.iter().enumerate() {
        let new_tile = tile_perm[tile];
        out[new_tile] = value;
    }
    out
}

fn permute_opp_next_targets(opp_next: [u8; 3], perm: &[u8; 3]) -> [u8; 3] {
    let mut out = opp_next;
    let tile_perm = &permutation_tables().tile_34[permutation_index(perm)];
    for tile in &mut out {
        if *tile < 34 {
            *tile = tile_perm[*tile as usize] as u8;
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

const PLAYER_COUNT: usize = 4;
const OPPONENT_COUNT: usize = 3;
const TILE_COUNT: usize = 34;
const GRP_CLASS_COUNT: usize = 24;
const BELIEF_FIELD_PLANES: usize = 16;
const BELIEF_FIELD_SIZE: usize = BELIEF_FIELD_PLANES * TILE_COUNT;
const SPATIAL_TARGET_SIZE: usize = OPPONENT_COUNT * TILE_COUNT;

thread_local! {
    static COLLATE_SCRATCH: RefCell<Option<CollateBuffers>> = const { RefCell::new(None) };
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
    pub delta_q_target: Option<Tensor<B, 2>>,
    pub delta_q_mask: Option<Tensor<B, 2>>,
    pub belief_fields_target: Option<Tensor<B, 3>>,
    pub mixture_weight_target: Option<Tensor<B, 2>>,
    pub belief_fields_mask: Option<Tensor<B, 1>>,
    pub mixture_weight_mask: Option<Tensor<B, 1>>,
    pub opp_next_target: Tensor<B, 3>,
    pub score_pdf_target: Tensor<B, 2>,
    pub score_cdf_target: Tensor<B, 2>,
    pub target_presence: Option<TargetPresence>,
}

pub struct MjaiBcBatch<B: Backend> {
    pub actions: Tensor<B, 1, Int>,
    pub exit_target: Option<Tensor<B, 2>>,
    pub exit_mask: Option<Tensor<B, 2>>,
}

pub type CollatedHydraBatch<B> = Result<Option<(Tensor<B, 3>, HydraTargets<B>)>, String>;
pub type CollatedSampleBatch<B> = Result<Option<(Tensor<B, 3>, MjaiBatch<B>)>, String>;
pub type CollatedOwnedBatch<B> =
    Result<Option<(Tensor<B, 3>, MjaiBatch<B>, HydraTargets<B>)>, String>;
pub type CollatedOwnedBcBatch<B> =
    Result<Option<(Tensor<B, 3>, MjaiBcBatch<B>, HydraTargets<B>)>, String>;
type OptionalActionTargets = Option<([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE])>;

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
    exit_samples: Vec<OptionalActionTargets>,
    delta_q_samples: Vec<OptionalActionTargets>,
    belief_fields_flat: Vec<f32>,
    mixture_weights_flat: Vec<f32>,
    any_belief_fields: bool,
    any_mixture_weights: bool,
    belief_fields_mask: Vec<f32>,
    mixture_weight_mask: Vec<f32>,
    opp_flat: Vec<f32>,
    pdf_flat: Vec<f32>,
    cdf_flat: Vec<f32>,
    target_presence: TargetPresence,
}

fn maybe_augment_action_vector(
    values: Option<[f32; HYDRA_ACTION_SPACE]>,
    perm: Option<&[u8; 3]>,
) -> Option<[f32; HYDRA_ACTION_SPACE]> {
    match (values, perm) {
        (Some(values), Some(perm)) => Some(augment_action_vector_suit(&values, perm)),
        (Some(values), None) => Some(values),
        (None, _) => None,
    }
}

fn maybe_augment_belief_fields(
    values: Option<[f32; BELIEF_FIELD_SIZE]>,
    perm: Option<&[u8; 3]>,
) -> Option<[f32; BELIEF_FIELD_SIZE]> {
    match (values, perm) {
        (Some(values), Some(perm)) => Some(augment_belief_fields_suit(&values, perm)),
        (Some(values), None) => Some(values),
        (None, _) => None,
    }
}

fn maybe_augment_spatial_target(
    values: [f32; SPATIAL_TARGET_SIZE],
    perm: Option<&[u8; 3]>,
) -> [f32; SPATIAL_TARGET_SIZE] {
    perm.map_or(values, |perm| permute_spatial_targets_3x34(values, perm))
}

fn collate_optional_target_pair(
    name: &str,
    target: Option<[f32; HYDRA_ACTION_SPACE]>,
    mask: Option<[f32; HYDRA_ACTION_SPACE]>,
) -> Result<OptionalActionTargets, String> {
    match (target, mask) {
        (Some(target), Some(mask)) => Ok(Some((target, mask))),
        (None, None) => Ok(None),
        _ => Err(format!("{name} target/mask mismatch for sample collation")),
    }
}

fn update_optional_presence(
    name: &str,
    values_present: bool,
    present_flag: bool,
    mask_slot: &mut f32,
    any_flag: &mut bool,
) -> Result<(), String> {
    match (values_present, present_flag) {
        (true, true) => {
            *mask_slot = 1.0;
            *any_flag = true;
            Ok(())
        }
        (false, false) => Ok(()),
        _ => Err(format!(
            "{name} target/presence mismatch for sample collation"
        )),
    }
}

fn optional_tensor_2d<B: Backend>(
    values: &[f32],
    any_present: bool,
    batch: usize,
    width: usize,
    device: &B::Device,
) -> Option<Tensor<B, 2>> {
    any_present.then(|| Tensor::<B, 1>::from_floats(values, device).reshape([batch, width]))
}

fn optional_mask_tensor_1d<B: Backend>(
    values: &[f32],
    any_present: bool,
    device: &B::Device,
) -> Option<Tensor<B, 1>> {
    any_present.then(|| Tensor::<B, 1>::from_floats(values, device))
}

fn optional_tensor_3d<B: Backend>(
    values: &[f32],
    any_present: bool,
    batch: usize,
    dim1: usize,
    dim2: usize,
    device: &B::Device,
) -> Option<Tensor<B, 3>> {
    any_present.then(|| Tensor::<B, 1>::from_floats(values, device).reshape([batch, dim1, dim2]))
}

fn collate_with_writer<'a, B: Backend, I>(
    samples: I,
    batch: usize,
    device: &B::Device,
) -> Result<MjaiBatch<B>, String>
where
    I: IntoIterator<Item = (&'a MjaiSample, Option<&'a [u8; 3]>)>,
{
    COLLATE_SCRATCH.with(|scratch| {
        let mut slot = scratch.borrow_mut();
        let mut buffers = match slot.take() {
            Some(mut buffers) if buffers.capacity_batch() >= batch => {
                buffers.reset_for_batch(batch);
                buffers
            }
            _ => CollateBuffers::new(batch),
        };
        for (index, (sample, perm)) in samples.into_iter().enumerate() {
            buffers.write_sample(index, sample, perm)?;
        }
        let batch_out = buffers.to_batch(batch, device);
        *slot = Some(buffers);
        Ok(batch_out)
    })
}

fn build_batch_from_samples<B: Backend>(
    samples: &[MjaiSample],
    augment: bool,
    device: &B::Device,
) -> Result<Option<MjaiBatch<B>>, String> {
    if samples.is_empty() {
        return Ok(None);
    }

    if augment {
        let batch = samples.len() * ALL_PERMUTATIONS.len();
        collate_with_writer::<B, _>(
            samples.iter().flat_map(|sample| {
                ALL_PERMUTATIONS
                    .iter()
                    .map(move |perm| (sample, Some(perm as &[u8; 3])))
            }),
            batch,
            device,
        )
        .map(Some)
    } else {
        collate_with_writer::<B, _>(
            samples.iter().map(|sample| (sample, None)),
            samples.len(),
            device,
        )
        .map(Some)
    }
}

fn build_batch_from_sample_refs<B: Backend>(
    samples: &[&MjaiSample],
    augment: bool,
    device: &B::Device,
) -> Result<Option<MjaiBatch<B>>, String> {
    if samples.is_empty() {
        return Ok(None);
    }

    if augment {
        let batch = samples.len() * ALL_PERMUTATIONS.len();
        collate_with_writer::<B, _>(
            samples.iter().flat_map(|sample| {
                ALL_PERMUTATIONS
                    .iter()
                    .map(move |perm| (*sample, Some(perm as &[u8; 3])))
            }),
            batch,
            device,
        )
        .map(Some)
    } else {
        collate_with_writer::<B, _>(
            samples.iter().map(|sample| (*sample, None)),
            samples.len(),
            device,
        )
        .map(Some)
    }
}

impl CollateBuffers {
    fn new(batch: usize) -> Self {
        Self {
            obs_flat: vec![0.0f32; batch * OBS_SIZE],
            actions: vec![0i64; batch],
            mask_flat: vec![0.0f32; batch * HYDRA_ACTION_SPACE],
            values: vec![0.0f32; batch],
            grp_flat: vec![0.0f32; batch * GRP_CLASS_COUNT],
            oracle_flat: vec![0.0f32; batch * PLAYER_COUNT],
            oracle_mask: vec![0.0f32; batch],
            tenpai_flat: vec![0.0f32; batch * OPPONENT_COUNT],
            danger_flat: vec![0.0f32; batch * SPATIAL_TARGET_SIZE],
            dmask_flat: vec![0.0f32; batch * SPATIAL_TARGET_SIZE],
            safety_residual_flat: vec![0.0f32; batch * HYDRA_ACTION_SPACE],
            safety_residual_mask_flat: vec![0.0f32; batch * HYDRA_ACTION_SPACE],
            any_safety_residual: false,
            exit_samples: vec![None; batch],
            delta_q_samples: vec![None; batch],
            belief_fields_flat: vec![0.0f32; batch * BELIEF_FIELD_SIZE],
            mixture_weights_flat: vec![0.0f32; batch * PLAYER_COUNT],
            any_belief_fields: false,
            any_mixture_weights: false,
            belief_fields_mask: vec![0.0f32; batch],
            mixture_weight_mask: vec![0.0f32; batch],
            opp_flat: vec![0.0f32; batch * SPATIAL_TARGET_SIZE],
            pdf_flat: vec![0.0f32; batch * SCORE_BINS],
            cdf_flat: vec![0.0f32; batch * SCORE_BINS],
            target_presence: TargetPresence::with_batch_size(batch),
        }
    }

    fn capacity_batch(&self) -> usize {
        self.actions.len()
    }

    fn reset_for_batch(&mut self, batch: usize) {
        debug_assert!(self.capacity_batch() >= batch);
        self.any_safety_residual = false;
        self.exit_samples[..batch].fill(None);
        self.delta_q_samples[..batch].fill(None);
        self.any_belief_fields = false;
        self.any_mixture_weights = false;
        self.belief_fields_mask[..batch].fill(0.0);
        self.mixture_weight_mask[..batch].fill(0.0);
        self.target_presence = TargetPresence::with_batch_size(batch);
    }

    fn write_sample(
        &mut self,
        index: usize,
        sample: &MjaiSample,
        perm: Option<&[u8; 3]>,
    ) -> Result<(), String> {
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
        let danger = maybe_augment_spatial_target(sample.danger, perm);
        let danger_mask = maybe_augment_spatial_target(sample.danger_mask, perm);
        let safety_residual = maybe_augment_action_vector(sample.safety_residual, perm);
        let safety_residual_mask = maybe_augment_action_vector(sample.safety_residual_mask, perm);
        let belief_fields = maybe_augment_belief_fields(sample.belief_fields, perm);
        let exit_target = maybe_augment_action_vector(sample.exit_target, perm);
        let exit_mask = maybe_augment_action_vector(sample.exit_mask, perm);
        let delta_q_target = maybe_augment_action_vector(sample.delta_q_target, perm);
        let delta_q_mask = maybe_augment_action_vector(sample.delta_q_mask, perm);
        let grp_row = &mut self.grp_flat[index * GRP_CLASS_COUNT..(index + 1) * GRP_CLASS_COUNT];
        let oracle_row = &mut self.oracle_flat[index * PLAYER_COUNT..(index + 1) * PLAYER_COUNT];
        let safety_residual_row = &mut self.safety_residual_flat
            [index * HYDRA_ACTION_SPACE..(index + 1) * HYDRA_ACTION_SPACE];
        let safety_residual_mask_row = &mut self.safety_residual_mask_flat
            [index * HYDRA_ACTION_SPACE..(index + 1) * HYDRA_ACTION_SPACE];
        let belief_fields_row = &mut self.belief_fields_flat
            [index * BELIEF_FIELD_SIZE..(index + 1) * BELIEF_FIELD_SIZE];
        let mixture_weights_row =
            &mut self.mixture_weights_flat[index * PLAYER_COUNT..(index + 1) * PLAYER_COUNT];
        let opp_row =
            &mut self.opp_flat[index * SPATIAL_TARGET_SIZE..(index + 1) * SPATIAL_TARGET_SIZE];

        self.obs_flat[index * OBS_SIZE..(index + 1) * OBS_SIZE].copy_from_slice(&obs);
        self.actions[index] = action as i64;
        self.mask_flat[index * HYDRA_ACTION_SPACE..(index + 1) * HYDRA_ACTION_SPACE]
            .copy_from_slice(&legal_mask);
        self.values[index] = score_delta_to_value(sample.score_delta);
        grp_row.fill(0.0);
        if (sample.grp_label as usize) < GRP_CLASS_COUNT {
            grp_row[sample.grp_label as usize] = 1.0;
        }
        oracle_row.fill(0.0);
        self.oracle_mask[index] = 0.0;
        if let Some(oracle) = sample.oracle_target {
            oracle_row.copy_from_slice(&oracle);
            self.oracle_mask[index] = 1.0;
            self.target_presence.counts[AdvancedHead::OracleCritic.index()] += 1;
        }
        self.tenpai_flat[index * OPPONENT_COUNT..(index + 1) * OPPONENT_COUNT]
            .copy_from_slice(&sample.tenpai);
        self.danger_flat[index * SPATIAL_TARGET_SIZE..(index + 1) * SPATIAL_TARGET_SIZE]
            .copy_from_slice(&danger);
        self.dmask_flat[index * SPATIAL_TARGET_SIZE..(index + 1) * SPATIAL_TARGET_SIZE]
            .copy_from_slice(&danger_mask);
        safety_residual_row.fill(0.0);
        safety_residual_mask_row.fill(0.0);
        if let Some(values) = safety_residual {
            safety_residual_row.copy_from_slice(&values);
            self.any_safety_residual = true;
        }
        if let Some(values) = safety_residual_mask {
            safety_residual_mask_row.copy_from_slice(&values);
            self.any_safety_residual = true;
            if values.iter().any(|&value| value > 0.0) {
                self.target_presence.counts[AdvancedHead::SafetyResidual.index()] += 1;
            }
        }
        self.exit_samples[index] = collate_optional_target_pair("exit", exit_target, exit_mask)?;
        self.delta_q_samples[index] =
            collate_optional_target_pair("delta_q", delta_q_target, delta_q_mask)?;
        if let Some((_, mask)) = self.delta_q_samples[index].as_ref() {
            let action_count = mask.iter().filter(|&&value| value > 0.0).count();
            if action_count > 0 {
                self.target_presence.counts[AdvancedHead::DeltaQ.index()] += 1;
                self.target_presence.delta_q_actions_present += action_count;
            }
        }
        belief_fields_row.fill(0.0);
        if let Some(values) = belief_fields {
            belief_fields_row.copy_from_slice(&values);
        }
        self.belief_fields_mask[index] = 0.0;
        update_optional_presence(
            "belief_fields",
            belief_fields.is_some(),
            sample.belief_fields_present,
            &mut self.belief_fields_mask[index],
            &mut self.any_belief_fields,
        )?;
        if self.belief_fields_mask[index] > 0.0 && self.oracle_mask[index] > 0.0 {
            self.target_presence.counts[AdvancedHead::BeliefFields.index()] += 1;
        }
        mixture_weights_row.fill(0.0);
        if let Some(values) = sample.mixture_weights {
            mixture_weights_row.copy_from_slice(&values);
        }
        self.mixture_weight_mask[index] = 0.0;
        update_optional_presence(
            "mixture_weight",
            sample.mixture_weights.is_some(),
            sample.mixture_weights_present,
            &mut self.mixture_weight_mask[index],
            &mut self.any_mixture_weights,
        )?;
        if self.mixture_weight_mask[index] > 0.0 && self.oracle_mask[index] > 0.0 {
            self.target_presence.counts[AdvancedHead::MixtureWeight.index()] += 1;
        }
        opp_row.fill(0.0);
        for (opp, tile) in opp_next.iter().copied().enumerate() {
            if tile < TILE_COUNT as u8 {
                opp_row[opp * TILE_COUNT + tile as usize] = 1.0;
            }
        }
        let pdf = score_delta_to_pdf(sample.score_delta);
        self.pdf_flat[index * SCORE_BINS..(index + 1) * SCORE_BINS].copy_from_slice(&pdf);
        let cdf = score_delta_to_cdf(sample.score_delta);
        self.cdf_flat[index * SCORE_BINS..(index + 1) * SCORE_BINS].copy_from_slice(&cdf);
        Ok(())
    }

    fn to_batch<B: Backend>(&self, batch: usize, device: &B::Device) -> MjaiBatch<B> {
        let (exit_target, exit_mask) =
            collate_action_targets::<B>(&self.exit_samples[..batch], device);
        let (delta_q_target, delta_q_mask) =
            collate_action_targets::<B>(&self.delta_q_samples[..batch], device);
        MjaiBatch {
            obs: Tensor::<B, 1>::from_floats(&self.obs_flat[..batch * OBS_SIZE], device).reshape([
                batch,
                NUM_CHANNELS,
                TILE_COUNT,
            ]),
            actions: Tensor::<B, 1, Int>::from_ints(&self.actions[..batch], device),
            legal_mask: Tensor::<B, 1>::from_floats(
                &self.mask_flat[..batch * HYDRA_ACTION_SPACE],
                device,
            )
            .reshape([batch, HYDRA_ACTION_SPACE]),
            value_target: Tensor::<B, 1>::from_floats(&self.values[..batch], device),
            grp_target: Tensor::<B, 1>::from_floats(
                &self.grp_flat[..batch * GRP_CLASS_COUNT],
                device,
            )
            .reshape([batch, GRP_CLASS_COUNT]),
            oracle_target: optional_tensor_2d::<B>(
                &self.oracle_flat[..batch * PLAYER_COUNT],
                self.oracle_mask[..batch].iter().any(|&v| v > 0.0),
                batch,
                PLAYER_COUNT,
                device,
            ),
            oracle_target_mask: Tensor::<B, 1>::from_floats(&self.oracle_mask[..batch], device),
            tenpai_target: Tensor::<B, 1>::from_floats(
                &self.tenpai_flat[..batch * OPPONENT_COUNT],
                device,
            )
            .reshape([batch, OPPONENT_COUNT]),
            danger_target: Tensor::<B, 1>::from_floats(
                &self.danger_flat[..batch * SPATIAL_TARGET_SIZE],
                device,
            )
            .reshape([batch, OPPONENT_COUNT, TILE_COUNT]),
            danger_mask: Tensor::<B, 1>::from_floats(
                &self.dmask_flat[..batch * SPATIAL_TARGET_SIZE],
                device,
            )
            .reshape([batch, OPPONENT_COUNT, TILE_COUNT]),
            safety_residual_target: optional_tensor_2d::<B>(
                &self.safety_residual_flat[..batch * HYDRA_ACTION_SPACE],
                self.any_safety_residual,
                batch,
                HYDRA_ACTION_SPACE,
                device,
            ),
            safety_residual_mask: optional_tensor_2d::<B>(
                &self.safety_residual_mask_flat[..batch * HYDRA_ACTION_SPACE],
                self.any_safety_residual,
                batch,
                HYDRA_ACTION_SPACE,
                device,
            ),
            exit_target,
            exit_mask,
            delta_q_target,
            delta_q_mask,
            belief_fields_target: optional_tensor_3d::<B>(
                &self.belief_fields_flat[..batch * BELIEF_FIELD_SIZE],
                self.any_belief_fields,
                batch,
                BELIEF_FIELD_PLANES,
                TILE_COUNT,
                device,
            ),
            mixture_weight_target: optional_tensor_2d::<B>(
                &self.mixture_weights_flat[..batch * PLAYER_COUNT],
                self.any_mixture_weights,
                batch,
                PLAYER_COUNT,
                device,
            ),
            belief_fields_mask: optional_mask_tensor_1d::<B>(
                &self.belief_fields_mask[..batch],
                self.any_belief_fields,
                device,
            ),
            mixture_weight_mask: optional_mask_tensor_1d::<B>(
                &self.mixture_weight_mask[..batch],
                self.any_mixture_weights,
                device,
            ),
            opp_next_target: Tensor::<B, 1>::from_floats(
                &self.opp_flat[..batch * SPATIAL_TARGET_SIZE],
                device,
            )
            .reshape([batch, OPPONENT_COUNT, TILE_COUNT]),
            score_pdf_target: Tensor::<B, 1>::from_floats(
                &self.pdf_flat[..batch * SCORE_BINS],
                device,
            )
            .reshape([batch, SCORE_BINS]),
            score_cdf_target: Tensor::<B, 1>::from_floats(
                &self.cdf_flat[..batch * SCORE_BINS],
                device,
            )
            .reshape([batch, SCORE_BINS]),
            target_presence: Some(self.target_presence),
        }
    }
}

impl<B: Backend> MjaiBatch<B> {
    pub fn into_hydra_targets(self) -> HydraTargets<B> {
        into_hydra_targets_inner(self)
    }

    pub fn to_hydra_targets(&self) -> HydraTargets<B> {
        cloned_hydra_targets(self)
    }
}

pub fn collate_batch<B: Backend>(samples: &[MjaiSample], device: &B::Device) -> MjaiBatch<B> {
    build_batch_from_samples::<B>(samples, false, device)
        .expect("valid sample collation")
        .expect("non-empty samples")
}

pub fn collate_batch_augmented<B: Backend>(
    samples: &[MjaiSample],
    device: &B::Device,
) -> MjaiBatch<B> {
    build_batch_from_samples::<B>(samples, true, device)
        .expect("valid sample collation")
        .expect("non-empty samples")
}

pub fn collate_sample_refs<B: Backend>(
    samples: &[&MjaiSample],
    augment: bool,
    device: &B::Device,
) -> CollatedHydraBatch<B> {
    let Some((obs, batch)) = collate_sample_refs_with_batch::<B>(samples, augment, device)? else {
        return Ok(None);
    };
    Ok(Some((obs, batch.into_hydra_targets())))
}

pub fn collate_sample_refs_owned<B: Backend>(
    samples: &[&MjaiSample],
    augment: bool,
    device: &B::Device,
) -> CollatedOwnedBatch<B> {
    let Some(batch) = build_batch_from_sample_refs::<B>(samples, augment, device)? else {
        return Ok(None);
    };
    let obs = batch.obs.clone();
    let targets = cloned_hydra_targets(&batch);
    Ok(Some((obs, batch, targets)))
}

pub fn collate_sample_refs_bc_owned<B: Backend>(
    samples: &[&MjaiSample],
    augment: bool,
    device: &B::Device,
) -> CollatedOwnedBcBatch<B> {
    let Some(batch) = build_batch_from_sample_refs::<B>(samples, augment, device)? else {
        return Ok(None);
    };
    Ok(Some(into_bc_batch_and_hydra_targets_inner(batch)))
}

pub fn collate_sample_refs_with_batch<B: Backend>(
    samples: &[&MjaiSample],
    augment: bool,
    device: &B::Device,
) -> CollatedSampleBatch<B> {
    let Some(batch) = build_batch_from_sample_refs::<B>(samples, augment, device)? else {
        return Ok(None);
    };
    let obs = batch.obs.clone();
    Ok(Some((obs, batch)))
}

pub fn collate_samples<B: Backend>(
    samples: &[MjaiSample],
    augment: bool,
    device: &B::Device,
) -> CollatedHydraBatch<B> {
    let Some((obs, batch)) = collate_batch_samples::<B>(samples, augment, device)? else {
        return Ok(None);
    };
    Ok(Some((obs, batch.into_hydra_targets())))
}

pub fn collate_samples_owned<B: Backend>(
    samples: &[MjaiSample],
    augment: bool,
    device: &B::Device,
) -> CollatedOwnedBatch<B> {
    let Some(batch) = build_batch_from_samples::<B>(samples, augment, device)? else {
        return Ok(None);
    };
    let obs = batch.obs.clone();
    let targets = cloned_hydra_targets(&batch);
    Ok(Some((obs, batch, targets)))
}

pub fn collate_samples_bc_owned<B: Backend>(
    samples: &[MjaiSample],
    augment: bool,
    device: &B::Device,
) -> CollatedOwnedBcBatch<B> {
    let Some(batch) = build_batch_from_samples::<B>(samples, augment, device)? else {
        return Ok(None);
    };
    Ok(Some(into_bc_batch_and_hydra_targets_inner(batch)))
}

pub fn collate_batch_samples<B: Backend>(
    samples: &[MjaiSample],
    augment: bool,
    device: &B::Device,
) -> CollatedSampleBatch<B> {
    let Some(batch) = build_batch_from_samples::<B>(samples, augment, device)? else {
        return Ok(None);
    };
    let obs = batch.obs.clone();
    Ok(Some((obs, batch)))
}

pub fn augment_samples_6x(samples: &[MjaiSample]) -> Vec<MjaiSample> {
    use crate::data::augment::{augment_action_suit, augment_mask_suit, augment_obs_suit};
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
                danger: maybe_augment_spatial_target(sample.danger, Some(perm)),
                danger_mask: maybe_augment_spatial_target(sample.danger_mask, Some(perm)),
                safety_residual: maybe_augment_action_vector(sample.safety_residual, Some(perm)),
                safety_residual_mask: maybe_augment_action_vector(
                    sample.safety_residual_mask,
                    Some(perm),
                ),
                exit_target: maybe_augment_action_vector(sample.exit_target, Some(perm)),
                exit_mask: maybe_augment_action_vector(sample.exit_mask, Some(perm)),
                delta_q_target: maybe_augment_action_vector(sample.delta_q_target, Some(perm)),
                delta_q_mask: maybe_augment_action_vector(sample.delta_q_mask, Some(perm)),
                belief_fields: maybe_augment_belief_fields(sample.belief_fields, Some(perm)),
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
    use hydra_core::tile::permute_tile_type;

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
            delta_q_target: None,
            delta_q_mask: None,
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
        assert_eq!(batch.legal_mask.dims(), [32, HYDRA_ACTION_SPACE]);
        assert_eq!(batch.value_target.dims(), [32]);
        assert_eq!(batch.grp_target.dims(), [32, GRP_CLASS_COUNT]);
        assert!(batch.oracle_target.is_none());
        assert_eq!(batch.oracle_target_mask.dims(), [32]);
        assert_eq!(batch.tenpai_target.dims(), [32, OPPONENT_COUNT]);
        assert_eq!(batch.danger_target.dims(), [32, OPPONENT_COUNT, TILE_COUNT]);
        assert_eq!(batch.danger_mask.dims(), [32, OPPONENT_COUNT, TILE_COUNT]);
        assert!(batch.safety_residual_target.is_none());
        assert!(batch.safety_residual_mask.is_none());
        assert_eq!(
            batch.opp_next_target.dims(),
            [32, OPPONENT_COUNT, TILE_COUNT]
        );
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
        for row in mask_slice.chunks(HYDRA_ACTION_SPACE) {
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
        let opp2_start = 2 * TILE_COUNT;
        let opp2_sum: f32 = slice[opp2_start..opp2_start + TILE_COUNT].iter().sum();
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
        assert_eq!(targets.policy_target.dims(), [1, HYDRA_ACTION_SPACE]);
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
        assert_eq!(targets.policy_target.dims(), [2, HYDRA_ACTION_SPACE]);
        let data = targets.policy_target.to_data();
        let slice = data.as_slice::<f32>().expect("f32");
        assert!((slice[2] - 1.0).abs() < 1e-6);
        assert!((slice[HYDRA_ACTION_SPACE + 7] - 1.0).abs() < 1e-6);
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
        assert!(targets.delta_q_mask.is_none());
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
    fn batch_to_hydra_targets_carries_delta_q_with_mask() {
        let device = Default::default();
        let mut sample = dummy_sample(0, 0);
        let mut target = [0.0f32; HYDRA_ACTION_SPACE];
        let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
        target[0] = 0.6;
        target[9] = -0.25;
        mask[0] = 1.0;
        mask[9] = 1.0;
        sample.delta_q_target = Some(target);
        sample.delta_q_mask = Some(mask);

        let batch = collate_batch::<B>(&[sample], &device);
        let targets = batch.into_hydra_targets();
        let dq = targets.delta_q_target.expect("delta_q target");
        let dqm = targets.delta_q_mask.expect("delta_q mask");
        assert_eq!(dq.dims(), [1, HYDRA_ACTION_SPACE]);
        assert_eq!(dqm.dims(), [1, HYDRA_ACTION_SPACE]);
        let values = dq.to_data().as_slice::<f32>().expect("f32").to_vec();
        let mask_values = dqm.to_data().as_slice::<f32>().expect("f32").to_vec();
        assert!((values[0] - 0.6).abs() < 1e-6);
        assert!((values[9] + 0.25).abs() < 1e-6);
        assert!((mask_values[0] - 1.0).abs() < 1e-6);
        assert!((mask_values[9] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn batch_to_hydra_targets_carries_target_presence_metadata() {
        let device = Default::default();
        let mut sample = dummy_sample(0, 0);
        sample.oracle_target = Some([0.1, -0.1, 0.2, -0.2]);
        sample.belief_fields = Some([0.0; 16 * 34]);
        sample.belief_fields_present = true;
        sample.mixture_weights = Some([0.7, 0.3, 0.0, 0.0]);
        sample.mixture_weights_present = true;
        let mut delta_q_target = [0.0f32; HYDRA_ACTION_SPACE];
        let mut delta_q_mask = [0.0f32; HYDRA_ACTION_SPACE];
        delta_q_target[0] = 0.6;
        delta_q_target[9] = -0.25;
        delta_q_mask[0] = 1.0;
        delta_q_mask[9] = 1.0;
        sample.delta_q_target = Some(delta_q_target);
        sample.delta_q_mask = Some(delta_q_mask);
        let mut safety_target = [0.0f32; HYDRA_ACTION_SPACE];
        let mut safety_mask = [0.0f32; HYDRA_ACTION_SPACE];
        safety_target[3] = 0.5;
        safety_mask[3] = 1.0;
        sample.safety_residual = Some(safety_target);
        sample.safety_residual_mask = Some(safety_mask);

        let batch = collate_batch::<B>(&[sample], &device);
        let presence = batch.target_presence.expect("batch target presence");
        assert_eq!(presence.batch_size, 1);
        assert_eq!(presence.counts[AdvancedHead::OracleCritic.index()], 1);
        assert_eq!(presence.counts[AdvancedHead::BeliefFields.index()], 1);
        assert_eq!(presence.counts[AdvancedHead::MixtureWeight.index()], 1);
        assert_eq!(presence.counts[AdvancedHead::DeltaQ.index()], 1);
        assert_eq!(presence.counts[AdvancedHead::SafetyResidual.index()], 1);
        assert_eq!(presence.delta_q_actions_present, 2);
        let targets = batch.into_hydra_targets();
        let target_presence = targets
            .target_presence
            .expect("targets should carry cached target presence");
        assert_eq!(target_presence.batch_size, 1);
        assert_eq!(target_presence.delta_q_actions_present, 2);
    }

    #[test]
    fn batch_to_hydra_targets_rejects_delta_q_when_pair_is_incomplete() {
        let device = Default::default();
        let mut target_only = dummy_sample(0, 0);
        let mut mask_only = dummy_sample(1, 0);
        let mut target = [0.0f32; HYDRA_ACTION_SPACE];
        let mut mask = [0.0f32; HYDRA_ACTION_SPACE];
        target[0] = 0.6;
        mask[1] = 1.0;
        target_only.delta_q_target = Some(target);
        mask_only.delta_q_mask = Some(mask);

        let err = collate_batch_samples::<B>(&[target_only, mask_only], false, &device)
            .err()
            .expect("incomplete delta_q pair should fail");
        assert!(err.contains("delta_q target/mask mismatch for sample collation"));
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
    fn batch_to_hydra_targets_rejects_belief_target_without_presence() {
        let device = Default::default();
        let mut sample = dummy_sample(0, 0);
        sample.belief_fields = Some([0.0; 16 * 34]);
        let err = collate_batch_samples::<B>(&[sample], false, &device)
            .err()
            .expect("belief target without presence should fail");
        assert!(err.contains("belief_fields target/presence mismatch for sample collation"));
    }

    #[test]
    fn batch_to_hydra_targets_rejects_belief_presence_without_target() {
        let device = Default::default();
        let mut sample = dummy_sample(0, 0);
        sample.belief_fields_present = true;
        let err = collate_batch_samples::<B>(&[sample], false, &device)
            .err()
            .expect("belief presence without target should fail");
        assert!(err.contains("belief_fields target/presence mismatch for sample collation"));
    }

    #[test]
    fn batch_to_hydra_targets_rejects_mixture_target_without_presence() {
        let device = Default::default();
        let mut sample = dummy_sample(0, 0);
        sample.mixture_weights = Some([0.0; 4]);
        let err = collate_batch_samples::<B>(&[sample], false, &device)
            .err()
            .expect("mixture target without presence should fail");
        assert!(err.contains("mixture_weight target/presence mismatch for sample collation"));
    }

    #[test]
    fn batch_to_hydra_targets_rejects_mixture_presence_without_target() {
        let device = Default::default();
        let mut sample = dummy_sample(0, 0);
        sample.mixture_weights_present = true;
        let err = collate_batch_samples::<B>(&[sample], false, &device)
            .err()
            .expect("mixture presence without target should fail");
        assert!(err.contains("mixture_weight target/presence mismatch for sample collation"));
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

        let (obs, targets) = collate_sample_refs::<B>(&refs, false, &device)
            .expect("borrowed collate")
            .expect("borrowed batch present");
        let (owned_obs, owned_targets) = collate_samples::<B>(&samples, false, &device)
            .expect("owned collate")
            .expect("owned batch present");

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
    fn collate_samples_owned_matches_split_batch_and_targets_without_augmentation() {
        let device = Default::default();
        let samples = vec![dummy_sample(2, 100), dummy_sample(7, -500)];

        let (owned_obs, owned_batch, owned_targets) =
            collate_samples_owned::<B>(&samples, false, &device)
                .expect("owned collate")
                .expect("owned batch present");
        let (obs, batch) = collate_batch_samples::<B>(&samples, false, &device)
            .expect("batch collate")
            .expect("batch present");
        let targets = batch.to_hydra_targets();

        assert_eq!(owned_obs.to_data(), obs.to_data());
        assert_eq!(owned_batch.obs.to_data(), batch.obs.to_data());
        assert_eq!(owned_batch.actions.to_data(), batch.actions.to_data());
        assert_eq!(
            owned_targets.policy_target.to_data(),
            targets.policy_target.to_data()
        );
        assert_eq!(
            owned_targets.legal_mask.to_data(),
            targets.legal_mask.to_data()
        );
        assert_eq!(
            owned_targets.value_target.to_data(),
            targets.value_target.to_data()
        );
        assert_eq!(
            owned_targets.grp_target.to_data(),
            targets.grp_target.to_data()
        );
        assert_eq!(
            owned_targets.danger_target.to_data(),
            targets.danger_target.to_data()
        );
        assert_eq!(
            owned_targets.danger_mask.to_data(),
            targets.danger_mask.to_data()
        );
        assert_eq!(
            owned_targets.opp_next_target.to_data(),
            targets.opp_next_target.to_data()
        );
        assert_eq!(
            owned_targets.score_pdf_target.to_data(),
            targets.score_pdf_target.to_data()
        );
        assert_eq!(
            owned_targets.score_cdf_target.to_data(),
            targets.score_cdf_target.to_data()
        );
        let owned_presence = owned_targets
            .target_presence
            .expect("owned target presence");
        let split_presence = targets.target_presence.expect("target presence");
        assert_eq!(owned_presence.batch_size, split_presence.batch_size);
        assert_eq!(owned_presence.counts, split_presence.counts);
        assert_eq!(
            owned_presence.delta_q_actions_present,
            split_presence.delta_q_actions_present
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
        let mut delta_q_target = [0.0f32; HYDRA_ACTION_SPACE];
        let mut delta_q_mask = [0.0f32; HYDRA_ACTION_SPACE];
        delta_q_target[0] = 0.6;
        delta_q_target[1] = -0.25;
        delta_q_mask[0] = 1.0;
        delta_q_mask[1] = 1.0;
        sample.delta_q_target = Some(delta_q_target);
        sample.delta_q_mask = Some(delta_q_mask);

        let refs = vec![&sample];
        let (obs, targets) = collate_sample_refs::<B>(&refs, true, &device)
            .expect("borrowed collate")
            .expect("borrowed batch present");
        let (owned_obs, owned_targets) = collate_samples::<B>(&[sample], true, &device)
            .expect("owned collate")
            .expect("owned batch present");

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
        assert_eq!(
            targets.delta_q_target.expect("borrowed delta_q").to_data(),
            owned_targets
                .delta_q_target
                .expect("owned delta_q")
                .to_data()
        );
        assert_eq!(
            targets
                .delta_q_mask
                .expect("borrowed delta_q mask")
                .to_data(),
            owned_targets
                .delta_q_mask
                .expect("owned delta_q mask")
                .to_data()
        );
    }

    #[test]
    fn collate_samples_owned_matches_split_batch_and_targets_with_augmentation() {
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
        let mut delta_q_target = [0.0f32; HYDRA_ACTION_SPACE];
        let mut delta_q_mask = [0.0f32; HYDRA_ACTION_SPACE];
        delta_q_target[0] = 0.6;
        delta_q_target[1] = -0.25;
        delta_q_mask[0] = 1.0;
        delta_q_mask[1] = 1.0;
        sample.delta_q_target = Some(delta_q_target);
        sample.delta_q_mask = Some(delta_q_mask);

        let owned_sample = sample;
        let mut split_sample = dummy_sample(0, 0);
        split_sample.obs = [0.0; OBS_SIZE];
        split_sample.obs[40 * 34] = 1.0;
        split_sample.opp_next = [0, 9, 27];
        split_sample.danger[0] = 0.25;
        split_sample.danger[34 + 9] = 0.5;
        split_sample.danger_mask[18] = 1.0;
        split_sample.safety_residual = Some(safety_residual);
        split_sample.safety_residual_mask = Some(safety_residual_mask);
        split_sample.delta_q_target = Some(delta_q_target);
        split_sample.delta_q_mask = Some(delta_q_mask);

        let (owned_obs, owned_batch, owned_targets) =
            collate_samples_owned::<B>(&[owned_sample], true, &device)
                .expect("owned collate")
                .expect("owned batch present");
        let (obs, batch) = collate_batch_samples::<B>(&[split_sample], true, &device)
            .expect("batch collate")
            .expect("batch present");
        let targets = batch.to_hydra_targets();

        assert_eq!(owned_obs.to_data(), obs.to_data());
        assert_eq!(owned_batch.obs.to_data(), batch.obs.to_data());
        assert_eq!(owned_batch.actions.to_data(), batch.actions.to_data());
        assert_eq!(
            owned_targets.policy_target.to_data(),
            targets.policy_target.to_data()
        );
        assert_eq!(
            owned_targets.danger_target.to_data(),
            targets.danger_target.to_data()
        );
        assert_eq!(
            owned_targets.danger_mask.to_data(),
            targets.danger_mask.to_data()
        );
        assert_eq!(
            owned_targets.opp_next_target.to_data(),
            targets.opp_next_target.to_data()
        );
        assert_eq!(
            owned_targets
                .safety_residual_target
                .expect("owned safety residual")
                .to_data(),
            targets
                .safety_residual_target
                .expect("safety residual")
                .to_data()
        );
        assert_eq!(
            owned_targets
                .delta_q_target
                .expect("owned delta_q")
                .to_data(),
            targets.delta_q_target.expect("delta_q").to_data()
        );
        let owned_presence = owned_targets
            .target_presence
            .expect("owned target presence");
        let split_presence = targets.target_presence.expect("target presence");
        assert_eq!(owned_presence.batch_size, split_presence.batch_size);
        assert_eq!(owned_presence.counts, split_presence.counts);
        assert_eq!(
            owned_presence.delta_q_actions_present,
            split_presence.delta_q_actions_present
        );
    }

    #[test]
    fn collate_samples_bc_owned_matches_split_batch_targets_and_exit_surface() {
        let device = Default::default();
        let mut sample = dummy_sample(2, 100);
        sample.oracle_target = Some([0.1, -0.1, 0.2, -0.2]);
        let mut exit_target = [0.0f32; HYDRA_ACTION_SPACE];
        let mut exit_mask = [0.0f32; HYDRA_ACTION_SPACE];
        exit_target[2] = 0.75;
        exit_target[45] = -0.25;
        exit_mask[2] = 1.0;
        exit_mask[45] = 1.0;
        sample.exit_target = Some(exit_target);
        sample.exit_mask = Some(exit_mask);
        let samples = [sample];

        let (obs, bc_batch, targets) = collate_samples_bc_owned::<B>(&samples, false, &device)
            .expect("bc owned collate")
            .expect("bc owned batch present");
        let (split_obs, split_batch) = collate_batch_samples::<B>(&samples, false, &device)
            .expect("split batch collate")
            .expect("split batch present");
        let split_targets = split_batch.to_hydra_targets();

        assert_eq!(obs.to_data(), split_obs.to_data());
        assert_eq!(bc_batch.actions.to_data(), split_batch.actions.to_data());
        assert_eq!(
            bc_batch
                .exit_target
                .as_ref()
                .expect("bc exit target")
                .to_data(),
            split_batch
                .exit_target
                .as_ref()
                .expect("split exit target")
                .to_data()
        );
        assert_eq!(
            bc_batch.exit_mask.as_ref().expect("bc exit mask").to_data(),
            split_batch
                .exit_mask
                .as_ref()
                .expect("split exit mask")
                .to_data()
        );
        assert_eq!(
            targets.policy_target.to_data(),
            split_targets.policy_target.to_data()
        );
        assert_eq!(
            targets.legal_mask.to_data(),
            split_targets.legal_mask.to_data()
        );
        assert_eq!(
            targets.value_target.to_data(),
            split_targets.value_target.to_data()
        );
        assert_eq!(
            targets.grp_target.to_data(),
            split_targets.grp_target.to_data()
        );
        assert_eq!(
            targets.oracle_target.expect("bc oracle target").to_data(),
            split_targets
                .oracle_target
                .expect("split oracle target")
                .to_data()
        );
        assert_eq!(
            targets
                .oracle_guidance_mask
                .expect("bc oracle mask")
                .to_data(),
            split_targets
                .oracle_guidance_mask
                .expect("split oracle mask")
                .to_data()
        );
    }

    #[test]
    fn collate_sample_refs_bc_owned_matches_split_batch_targets_and_exit_surface() {
        let device = Default::default();
        let mut sample = dummy_sample(2, 100);
        sample.oracle_target = Some([0.1, -0.1, 0.2, -0.2]);
        let mut exit_target = [0.0f32; HYDRA_ACTION_SPACE];
        let mut exit_mask = [0.0f32; HYDRA_ACTION_SPACE];
        exit_target[2] = 0.75;
        exit_target[45] = -0.25;
        exit_mask[2] = 1.0;
        exit_mask[45] = 1.0;
        sample.exit_target = Some(exit_target);
        sample.exit_mask = Some(exit_mask);
        let samples = [sample];
        let sample_refs: Vec<&MjaiSample> = samples.iter().collect();

        let (obs, bc_batch, targets) =
            collate_sample_refs_bc_owned::<B>(&sample_refs, false, &device)
                .expect("bc owned ref collate")
                .expect("bc owned ref batch present");
        let (split_obs, split_batch) =
            collate_sample_refs_with_batch::<B>(&sample_refs, false, &device)
                .expect("split ref batch collate")
                .expect("split ref batch present");
        let split_targets = split_batch.to_hydra_targets();

        assert_eq!(obs.to_data(), split_obs.to_data());
        assert_eq!(bc_batch.actions.to_data(), split_batch.actions.to_data());
        assert_eq!(
            bc_batch
                .exit_target
                .as_ref()
                .expect("bc ref exit target")
                .to_data(),
            split_batch
                .exit_target
                .as_ref()
                .expect("split ref exit target")
                .to_data()
        );
        assert_eq!(
            bc_batch
                .exit_mask
                .as_ref()
                .expect("bc ref exit mask")
                .to_data(),
            split_batch
                .exit_mask
                .as_ref()
                .expect("split ref exit mask")
                .to_data()
        );
        assert_eq!(
            targets.policy_target.to_data(),
            split_targets.policy_target.to_data()
        );
        assert_eq!(
            targets.legal_mask.to_data(),
            split_targets.legal_mask.to_data()
        );
        assert_eq!(
            targets.value_target.to_data(),
            split_targets.value_target.to_data()
        );
        assert_eq!(
            targets.grp_target.to_data(),
            split_targets.grp_target.to_data()
        );
        assert_eq!(
            targets
                .oracle_target
                .expect("bc ref oracle target")
                .to_data(),
            split_targets
                .oracle_target
                .expect("split ref oracle target")
                .to_data()
        );
        assert_eq!(
            targets
                .oracle_guidance_mask
                .expect("bc ref oracle mask")
                .to_data(),
            split_targets
                .oracle_guidance_mask
                .expect("split ref oracle mask")
                .to_data()
        );
    }
}
