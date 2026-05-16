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
pub type OptionalActionTargets = Option<([f32; HYDRA_ACTION_SPACE], [f32; HYDRA_ACTION_SPACE])>;
pub type TimedCollatedOwnedBcBatch<B> = Result<Option<TimedOwnedBcBatch<B>>, String>;

struct HostCollatedBatch {
    buffers: CollateBuffers,
    batch: usize,
}

pub struct TimedOwnedBcBatch<B: Backend> {
    pub obs: Tensor<B, 3>,
    pub batch: MjaiBcBatch<B>,
    pub targets: HydraTargets<B>,
    pub cpu_prep_seconds: f64,
    pub device_materialize_seconds: f64,
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
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Option<Tensor<B, 2>> {
    any_present.then(|| Tensor::<B, 1>::from_floats(values, device).reshape([batch, width]))
}

fn optional_mask_tensor_1d<B: Backend>(
    values: &[f32],
    any_present: bool,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Option<Tensor<B, 1>> {
    any_present.then(|| Tensor::<B, 1>::from_floats(values, device))
}

fn optional_tensor_3d<B: Backend>(
    values: &[f32],
    any_present: bool,
    batch: usize,
    dim1: usize,
    dim2: usize,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Option<Tensor<B, 3>> {
    any_present.then(|| Tensor::<B, 1>::from_floats(values, device).reshape([batch, dim1, dim2]))
}

fn collate_host_with_writer<'a, I>(samples: I, batch: usize) -> Result<HostCollatedBatch, String>
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
        Ok(HostCollatedBatch { buffers, batch })
    })
}

fn collate_with_writer<'a, B: Backend, I>(
    samples: I,
    batch: usize,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> Result<MjaiBatch<B>, String>
where
    I: IntoIterator<Item = (&'a MjaiSample, Option<&'a [u8; 3]>)>,
{
    Ok(collate_host_with_writer(samples, batch)?.into_batch(device))
}

fn build_batch_from_samples<B: Backend>(
    samples: &[MjaiSample],
    augment: bool,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
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
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
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

    fn to_batch<B: Backend>(
        &self,
        batch: usize,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> MjaiBatch<B> {
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
impl HostCollatedBatch {
    fn into_batch<B: Backend>(
        self,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> MjaiBatch<B> {
        let HostCollatedBatch { buffers, batch } = self;
        let batch_out = buffers.to_batch(batch, device);
        COLLATE_SCRATCH.with(|scratch| {
            *scratch.borrow_mut() = Some(buffers);
        });
        batch_out
    }

    fn into_bc_batch_and_targets<B: Backend>(
        self,
        device: &<B as burn::tensor::backend::BackendTypes>::Device,
    ) -> TimedOwnedBcBatch<B> {
        let started = std::time::Instant::now();
        let batch = self.into_batch(device);
        let device_materialize_seconds = started.elapsed().as_secs_f64();
        let (obs, batch, targets) = into_bc_batch_and_hydra_targets_inner(batch);
        TimedOwnedBcBatch {
            obs,
            batch,
            targets,
            cpu_prep_seconds: 0.0,
            device_materialize_seconds,
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

pub fn collate_batch<B: Backend>(
    samples: &[MjaiSample],
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> MjaiBatch<B> {
    build_batch_from_samples::<B>(samples, false, device)
        .expect("valid sample collation")
        .expect("non-empty samples")
}

pub fn collate_batch_augmented<B: Backend>(
    samples: &[MjaiSample],
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> MjaiBatch<B> {
    build_batch_from_samples::<B>(samples, true, device)
        .expect("valid sample collation")
        .expect("non-empty samples")
}

pub fn collate_sample_refs<B: Backend>(
    samples: &[&MjaiSample],
    augment: bool,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> CollatedHydraBatch<B> {
    let Some((obs, batch)) = collate_sample_refs_with_batch::<B>(samples, augment, device)? else {
        return Ok(None);
    };
    Ok(Some((obs, batch.into_hydra_targets())))
}

pub fn collate_sample_refs_owned<B: Backend>(
    samples: &[&MjaiSample],
    augment: bool,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
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
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> CollatedOwnedBcBatch<B> {
    let Some(batch) = build_batch_from_sample_refs::<B>(samples, augment, device)? else {
        return Ok(None);
    };
    Ok(Some(into_bc_batch_and_hydra_targets_inner(batch)))
}

pub fn collate_sample_refs_with_batch<B: Backend>(
    samples: &[&MjaiSample],
    augment: bool,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
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
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> CollatedHydraBatch<B> {
    let Some((obs, batch)) = collate_batch_samples::<B>(samples, augment, device)? else {
        return Ok(None);
    };
    Ok(Some((obs, batch.into_hydra_targets())))
}

pub fn collate_samples_owned<B: Backend>(
    samples: &[MjaiSample],
    augment: bool,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
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
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> CollatedOwnedBcBatch<B> {
    let Some(batch) = build_batch_from_samples::<B>(samples, augment, device)? else {
        return Ok(None);
    };
    Ok(Some(into_bc_batch_and_hydra_targets_inner(batch)))
}

pub fn collate_samples_bc_owned_timed<B: Backend>(
    samples: &[MjaiSample],
    augment: bool,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
) -> TimedCollatedOwnedBcBatch<B> {
    if samples.is_empty() {
        return Ok(None);
    }

    let cpu_started = std::time::Instant::now();
    let host_batch = if augment {
        let batch = samples.len() * ALL_PERMUTATIONS.len();
        collate_host_with_writer(
            samples.iter().flat_map(|sample| {
                ALL_PERMUTATIONS
                    .iter()
                    .map(move |perm| (sample, Some(perm as &[u8; 3])))
            }),
            batch,
        )?
    } else {
        collate_host_with_writer(samples.iter().map(|sample| (sample, None)), samples.len())?
    };
    let cpu_prep_seconds = cpu_started.elapsed().as_secs_f64();
    let mut timed = host_batch.into_bc_batch_and_targets(device);
    timed.cpu_prep_seconds = cpu_prep_seconds;
    Ok(Some(timed))
}

pub fn collate_batch_samples<B: Backend>(
    samples: &[MjaiSample],
    augment: bool,
    device: &<B as burn::tensor::backend::BackendTypes>::Device,
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
                compact_facts: sample.compact_facts.clone(),
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
mod tests;
