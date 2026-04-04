//! Pinned host memory + async H2D transfer for BC shard batches.
//!
//! When the `cuda-graph` feature is active, this module provides:
//!
//! - [`PinnedStagingArea`]: a reusable set of page-locked (pinned) host
//!   buffers sized for one full BC batch.  Pinned memory enables the CUDA
//!   DMA engine to copy data to the device without staging through an
//!   intermediate pageable bounce buffer.
//!
//! - [`AsyncH2DContext`]: owns a dedicated CUDA copy stream and an event
//!   used for synchronization.  The copy stream is distinct from the
//!   default compute stream, so H2D transfers overlap with ongoing GPU
//!   compute.
//!
//! - [`PreallocatedDeviceTensors`]: preallocated GPU-side tensors that
//!   eliminate per-batch `cudaMalloc` overhead.  Each iteration copies
//!   pinned host data directly into these buffers via `copy_()`, then
//!   hands out `shallow_clone()`-based views with fresh Burn ownership
//!   tracking.
//!
//! - [`materialize_staged`]: stages a [`BcShardHostBatch`] into pinned
//!   buffers, issues all `Tensor::from_floats` / `from_ints` calls on the
//!   copy stream, records an event, restores the compute stream, and
//!   makes the compute stream wait on the event before returning the
//!   device-side batch.

use std::ptr;

use burn::backend::libtorch::{LibTorchDevice, TchTensor};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::{NUM_CHANNELS, OBS_SIZE};
use hydra_train::data::bc_shards::{BcShardBatch, BcShardHostBatch};
use hydra_train::data::sample::MjaiBcBatch;
use hydra_train::training::losses::HydraTargets;

use super::cuda_graph::{CudaEvent, CudaStream, PinnedBuffer};

const OPPONENT_COUNT: usize = 3;
const PLAYER_COUNT: usize = 4;
const TILE_COUNT: usize = 34;
const SPATIAL_TARGET_SIZE: usize = OPPONENT_COUNT * TILE_COUNT;
const GRP_CLASS_COUNT: usize = 24;
const SCORE_BINS: usize = 64;

// ---------------------------------------------------------------------------
// Pinned staging area
// ---------------------------------------------------------------------------

/// Reusable page-locked host buffers for one BC shard batch.
///
/// All buffers are allocated once and reused across batches.  The total
/// pinned footprint is proportional to `batch_size`.
pub(crate) struct PinnedStagingArea {
    batch_size: usize,

    // f32 buffers (sizes in *elements*, not bytes)
    obs: PinnedBuffer,
    legal_mask: PinnedBuffer,
    value_target: PinnedBuffer,
    grp_target: PinnedBuffer,
    oracle_target: PinnedBuffer,
    oracle_target_mask: PinnedBuffer,
    tenpai: PinnedBuffer,
    danger: PinnedBuffer,
    danger_mask: PinnedBuffer,
    opp_next: PinnedBuffer,
    score_pdf: PinnedBuffer,
    score_cdf: PinnedBuffer,

    // i64 buffer
    actions: PinnedBuffer,

    // optional f32 buffers -- always allocated at max-batch size so we
    // don't need to reallocate when optional targets appear/disappear.
    safety_target: PinnedBuffer,
    safety_mask: PinnedBuffer,
    exit_target: PinnedBuffer,
    exit_mask: PinnedBuffer,
    delta_q_target: PinnedBuffer,
    delta_q_mask: PinnedBuffer,
}

impl PinnedStagingArea {
    /// Allocate pinned staging buffers for a given batch size.
    pub(crate) fn new(batch_size: usize) -> Self {
        let f32_bytes = std::mem::size_of::<f32>();
        let i64_bytes = std::mem::size_of::<i64>();
        let action_elems = batch_size * HYDRA_ACTION_SPACE;

        Self {
            batch_size,
            obs: PinnedBuffer::new(batch_size * OBS_SIZE * f32_bytes),
            legal_mask: PinnedBuffer::new(action_elems * f32_bytes),
            value_target: PinnedBuffer::new(batch_size * f32_bytes),
            grp_target: PinnedBuffer::new(batch_size * GRP_CLASS_COUNT * f32_bytes),
            oracle_target: PinnedBuffer::new(batch_size * PLAYER_COUNT * f32_bytes),
            oracle_target_mask: PinnedBuffer::new(batch_size * f32_bytes),
            tenpai: PinnedBuffer::new(batch_size * OPPONENT_COUNT * f32_bytes),
            danger: PinnedBuffer::new(batch_size * SPATIAL_TARGET_SIZE * f32_bytes),
            danger_mask: PinnedBuffer::new(batch_size * SPATIAL_TARGET_SIZE * f32_bytes),
            opp_next: PinnedBuffer::new(batch_size * SPATIAL_TARGET_SIZE * f32_bytes),
            score_pdf: PinnedBuffer::new(batch_size * SCORE_BINS * f32_bytes),
            score_cdf: PinnedBuffer::new(batch_size * SCORE_BINS * f32_bytes),
            actions: PinnedBuffer::new(batch_size * i64_bytes),
            safety_target: PinnedBuffer::new(action_elems * f32_bytes),
            safety_mask: PinnedBuffer::new(action_elems * f32_bytes),
            exit_target: PinnedBuffer::new(action_elems * f32_bytes),
            exit_mask: PinnedBuffer::new(action_elems * f32_bytes),
            delta_q_target: PinnedBuffer::new(action_elems * f32_bytes),
            delta_q_mask: PinnedBuffer::new(action_elems * f32_bytes),
        }
    }

    /// Stage a host batch into pinned memory.
    ///
    /// Copies every flat buffer from the pageable `Vec` backing into the
    /// corresponding pinned region.  The caller can then create Burn
    /// tensors from the pinned slices, which lets LibTorch issue truly
    /// async DMA copies.
    pub(crate) fn stage(&mut self, host: &BcShardHostBatch) {
        assert!(
            host.batch_size <= self.batch_size,
            "host batch {} exceeds staging area capacity {}",
            host.batch_size,
            self.batch_size,
        );
        copy_f32_to_pinned(&host.obs_flat, &mut self.obs);
        copy_f32_to_pinned(&host.legal_mask_flat, &mut self.legal_mask);
        copy_f32_to_pinned(&host.value_target, &mut self.value_target);
        copy_f32_to_pinned(&host.grp_target_flat, &mut self.grp_target);
        copy_f32_to_pinned(&host.oracle_target_flat, &mut self.oracle_target);
        copy_f32_to_pinned(&host.oracle_target_mask, &mut self.oracle_target_mask);
        copy_f32_to_pinned(&host.tenpai_flat, &mut self.tenpai);
        copy_f32_to_pinned(&host.danger_flat, &mut self.danger);
        copy_f32_to_pinned(&host.danger_mask_flat, &mut self.danger_mask);
        copy_f32_to_pinned(&host.opp_next_flat, &mut self.opp_next);
        copy_f32_to_pinned(&host.score_pdf_flat, &mut self.score_pdf);
        copy_f32_to_pinned(&host.score_cdf_flat, &mut self.score_cdf);
        copy_i64_to_pinned(&host.actions, &mut self.actions);

        if let Some(buf) = host.safety_target_flat.as_ref() {
            copy_f32_to_pinned(buf, &mut self.safety_target);
        }
        if let Some(buf) = host.safety_mask_flat.as_ref() {
            copy_f32_to_pinned(buf, &mut self.safety_mask);
        }
        if let Some(buf) = host.exit_target_flat.as_ref() {
            copy_f32_to_pinned(buf, &mut self.exit_target);
        }
        if let Some(buf) = host.exit_mask_flat.as_ref() {
            copy_f32_to_pinned(buf, &mut self.exit_mask);
        }
        if let Some(buf) = host.delta_q_target_flat.as_ref() {
            copy_f32_to_pinned(buf, &mut self.delta_q_target);
        }
        if let Some(buf) = host.delta_q_mask_flat.as_ref() {
            copy_f32_to_pinned(buf, &mut self.delta_q_mask);
        }
    }
}

// ---------------------------------------------------------------------------
// Async H2D context
// ---------------------------------------------------------------------------

/// Owns the dedicated CUDA copy stream and synchronization event for
/// async host-to-device transfers.
pub(crate) struct AsyncH2DContext {
    copy_stream: CudaStream,
    event: CudaEvent,
}

impl AsyncH2DContext {
    /// Create a new async H2D context on the given CUDA device.
    pub(crate) fn new(device_index: i64) -> Self {
        Self {
            copy_stream: CudaStream::from_pool(device_index),
            // Timing disabled -- we only need ordering guarantees.
            event: CudaEvent::new(false),
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level async materialize entry point
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Preallocated device tensors -- eliminates per-batch cudaMalloc
// ---------------------------------------------------------------------------

/// GPU-resident tensor buffers allocated once and reused across batches.
///
/// Each iteration, pinned host data is `copy_()`-d into these buffers on
/// the copy stream.  Callers receive `shallow_clone()`-based handles with
/// fresh Burn `Storage::Owned` arcs, so Burn's refcount-based in-place
/// mutation tracking works correctly without aliasing the persistent
/// storage arcs.
///
/// # Safety contract
///
/// The caller must guarantee that *all* tensors derived from a previous
/// batch are dropped before calling [`materialize_staged_reuse`] for the
/// next batch.  This is naturally satisfied in the shard training loop
/// because `train_logical_batch_from_host_batch` fully consumes the batch
/// (forward + backward + gradient extraction) before returning, and the
/// next iteration only starts after return.
pub(crate) struct PreallocatedDeviceTensors {
    // f32 buffers
    obs: tch::Tensor,
    legal_mask: tch::Tensor,
    value_target: tch::Tensor,
    oracle_target: tch::Tensor,
    oracle_target_mask: tch::Tensor,
    tenpai: tch::Tensor,
    danger: tch::Tensor,
    danger_mask: tch::Tensor,
    grp_target: tch::Tensor,
    opp_next: tch::Tensor,
    score_pdf: tch::Tensor,
    score_cdf: tch::Tensor,

    // i64 buffer
    actions: tch::Tensor,

    // optional f32 buffers (always allocated at max-batch size)
    safety_target: tch::Tensor,
    safety_mask: tch::Tensor,
    exit_target: tch::Tensor,
    exit_mask: tch::Tensor,
    delta_q_target: tch::Tensor,
    delta_q_mask: tch::Tensor,
}

impl PreallocatedDeviceTensors {
    /// Allocate GPU tensors for a given batch size on the specified device.
    pub(crate) fn new(batch_size: usize, device: &LibTorchDevice) -> Self {
        let dev: tch::Device = (*device).into();
        let opts_f32 = (tch::Kind::Float, dev);
        let opts_i64 = (tch::Kind::Int64, dev);
        let b = batch_size as i64;
        let action_space = HYDRA_ACTION_SPACE as i64;

        Self {
            obs: tch::Tensor::zeros([b * OBS_SIZE as i64], opts_f32),
            legal_mask: tch::Tensor::zeros([b * action_space], opts_f32),
            value_target: tch::Tensor::zeros([b], opts_f32),
            oracle_target: tch::Tensor::zeros([b * PLAYER_COUNT as i64], opts_f32),
            oracle_target_mask: tch::Tensor::zeros([b], opts_f32),
            tenpai: tch::Tensor::zeros([b * OPPONENT_COUNT as i64], opts_f32),
            danger: tch::Tensor::zeros([b * SPATIAL_TARGET_SIZE as i64], opts_f32),
            danger_mask: tch::Tensor::zeros([b * SPATIAL_TARGET_SIZE as i64], opts_f32),
            grp_target: tch::Tensor::zeros([b * GRP_CLASS_COUNT as i64], opts_f32),
            opp_next: tch::Tensor::zeros([b * SPATIAL_TARGET_SIZE as i64], opts_f32),
            score_pdf: tch::Tensor::zeros([b * SCORE_BINS as i64], opts_f32),
            score_cdf: tch::Tensor::zeros([b * SCORE_BINS as i64], opts_f32),
            actions: tch::Tensor::zeros([b], opts_i64),
            safety_target: tch::Tensor::zeros([b * action_space], opts_f32),
            safety_mask: tch::Tensor::zeros([b * action_space], opts_f32),
            exit_target: tch::Tensor::zeros([b * action_space], opts_f32),
            exit_mask: tch::Tensor::zeros([b * action_space], opts_f32),
            delta_q_target: tch::Tensor::zeros([b * action_space], opts_f32),
            delta_q_mask: tch::Tensor::zeros([b * action_space], opts_f32),
        }
    }
}

/// Like [`materialize_staged`] but copies into preallocated GPU tensors
/// instead of allocating new ones each batch.
///
/// The H2D copy goes: pinned host buffer -> `from_blob` CPU view (zero
/// alloc) -> `copy_()` into preallocated GPU tensor (async on copy
/// stream, zero GPU alloc) -> `shallow_clone()` + `TchTensor::new()` for
/// fresh Burn ownership tracking.
pub(crate) fn materialize_staged_reuse<B>(
    host: &BcShardHostBatch,
    staging: &mut PinnedStagingArea,
    h2d: &AsyncH2DContext,
    device: &LibTorchDevice,
    gpu_tensors: &mut PreallocatedDeviceTensors,
) -> BcShardBatch<B>
where
    B: AutodiffBackend<
        Device = LibTorchDevice,
        InnerBackend: Backend<FloatTensorPrimitive = TchTensor, IntTensorPrimitive = TchTensor>,
    >,
{
    let batch = host.batch_size;

    // 1. CPU memcpy: pageable Vec -> pinned staging
    staging.stage(host);

    // 2. Remember the current (compute) stream so we can restore it.
    let device_index = cuda_device_index(device);
    let compute_stream = CudaStream::current(device_index);

    // 3. Switch to dedicated copy stream.
    h2d.copy_stream.set_current();

    // 4. Copy pinned -> preallocated GPU tensors, produce Burn tensors.
    let shard_batch =
        unsafe { materialize_reuse_from_pinned::<B>(staging, host, batch, gpu_tensors) };

    // 5. Record event on copy stream.
    h2d.event.record(&h2d.copy_stream);

    // 6. Restore compute stream.
    compute_stream.set_current();

    // 7. Make compute stream wait on copy completion.
    compute_stream.wait_event(&h2d.event);

    shard_batch
}

pub(crate) fn materialize_staged_reuse_inner<B>(
    host: &BcShardHostBatch,
    staging: &mut PinnedStagingArea,
    h2d: &AsyncH2DContext,
    device: &LibTorchDevice,
    gpu_tensors: &mut PreallocatedDeviceTensors,
) -> BcShardBatch<B>
where
    B: Backend<
        Device = LibTorchDevice,
        FloatTensorPrimitive = TchTensor,
        IntTensorPrimitive = TchTensor,
    >,
{
    let batch = host.batch_size;

    staging.stage(host);

    let device_index = cuda_device_index(device);
    let compute_stream = CudaStream::current(device_index);

    h2d.copy_stream.set_current();

    let shard_batch =
        unsafe { materialize_reuse_from_pinned_inner::<B>(staging, host, batch, gpu_tensors) };

    h2d.event.record(&h2d.copy_stream);

    compute_stream.set_current();

    compute_stream.wait_event(&h2d.event);

    shard_batch
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Extract CUDA device index from LibTorchDevice, defaulting to 0.
fn cuda_device_index(device: &LibTorchDevice) -> i64 {
    match device {
        LibTorchDevice::Cuda(idx) => *idx as i64,
        _ => 0,
    }
}

/// # Safety
///
/// [`PinnedStagingArea::stage`] must have been called with a host batch
/// whose `batch_size <= staging.batch_size` before this function.
/// All tensors from a previous call must have been dropped.
unsafe fn materialize_reuse_from_pinned<B>(
    staging: &PinnedStagingArea,
    host: &BcShardHostBatch,
    batch: usize,
    gpu: &mut PreallocatedDeviceTensors,
) -> BcShardBatch<B>
where
    B: AutodiffBackend<
        Device = LibTorchDevice,
        InnerBackend: Backend<FloatTensorPrimitive = TchTensor, IntTensorPrimitive = TchTensor>,
    >,
{
    unsafe {
        let f = |pinned: &PinnedBuffer, count: usize, dst: &mut tch::Tensor| -> Tensor<B, 1> {
            burn_tensor_from_tch_f32::<B, 1>(copy_pinned_f32_to_gpu(pinned, count, dst))
        };

        let obs = f(&staging.obs, batch * OBS_SIZE, &mut gpu.obs).reshape([
            batch,
            NUM_CHANNELS,
            TILE_COUNT,
        ]);

        let actions_tensor = burn_int_tensor_from_tch::<B>(copy_pinned_i64_to_gpu(
            &staging.actions,
            batch,
            &mut gpu.actions,
        ));

        let legal_mask = burn_tensor_from_tch_f32::<B, 1>(copy_pinned_f32_to_gpu(
            &staging.legal_mask,
            batch * HYDRA_ACTION_SPACE,
            &mut gpu.legal_mask,
        ))
        .reshape([batch, HYDRA_ACTION_SPACE]);

        let value_target = f(&staging.value_target, batch, &mut gpu.value_target);

        let grp_target = burn_tensor_from_tch_f32::<B, 1>(copy_pinned_f32_to_gpu(
            &staging.grp_target,
            batch * GRP_CLASS_COUNT,
            &mut gpu.grp_target,
        ))
        .reshape([batch, GRP_CLASS_COUNT]);

        let oracle_target = f(
            &staging.oracle_target,
            batch * PLAYER_COUNT,
            &mut gpu.oracle_target,
        )
        .reshape([batch, PLAYER_COUNT]);

        let oracle_target_mask = f(
            &staging.oracle_target_mask,
            batch,
            &mut gpu.oracle_target_mask,
        );

        let tenpai_target = burn_tensor_from_tch_f32::<B, 1>(copy_pinned_f32_to_gpu(
            &staging.tenpai,
            batch * OPPONENT_COUNT,
            &mut gpu.tenpai,
        ))
        .reshape([batch, OPPONENT_COUNT]);

        let danger_target = burn_tensor_from_tch_f32::<B, 1>(copy_pinned_f32_to_gpu(
            &staging.danger,
            batch * SPATIAL_TARGET_SIZE,
            &mut gpu.danger,
        ))
        .reshape([batch, OPPONENT_COUNT, TILE_COUNT]);

        let danger_mask = burn_tensor_from_tch_f32::<B, 1>(copy_pinned_f32_to_gpu(
            &staging.danger_mask,
            batch * SPATIAL_TARGET_SIZE,
            &mut gpu.danger_mask,
        ))
        .reshape([batch, OPPONENT_COUNT, TILE_COUNT]);

        let opp_next_target = burn_tensor_from_tch_f32::<B, 1>(copy_pinned_f32_to_gpu(
            &staging.opp_next,
            batch * SPATIAL_TARGET_SIZE,
            &mut gpu.opp_next,
        ))
        .reshape([batch, OPPONENT_COUNT, TILE_COUNT]);

        let score_pdf_target = burn_tensor_from_tch_f32::<B, 1>(copy_pinned_f32_to_gpu(
            &staging.score_pdf,
            batch * SCORE_BINS,
            &mut gpu.score_pdf,
        ))
        .reshape([batch, SCORE_BINS]);
        let score_cdf_target = burn_tensor_from_tch_f32::<B, 1>(copy_pinned_f32_to_gpu(
            &staging.score_cdf,
            batch * SCORE_BINS,
            &mut gpu.score_cdf,
        ))
        .reshape([batch, SCORE_BINS]);

        let exit_target_tensor = host.exit_target_flat.as_ref().map(|_| {
            f(
                &staging.exit_target,
                batch * HYDRA_ACTION_SPACE,
                &mut gpu.exit_target,
            )
            .reshape([batch, HYDRA_ACTION_SPACE])
        });
        let exit_mask_tensor = host.exit_mask_flat.as_ref().map(|_| {
            burn_tensor_from_tch_f32::<B, 1>(copy_pinned_f32_to_gpu(
                &staging.exit_mask,
                batch * HYDRA_ACTION_SPACE,
                &mut gpu.exit_mask,
            ))
            .reshape([batch, HYDRA_ACTION_SPACE])
        });

        let batch_struct = MjaiBcBatch {
            actions: actions_tensor.clone(),
            exit_target: exit_target_tensor.clone(),
            exit_mask: exit_mask_tensor.clone(),
        };

        let targets = HydraTargets {
            policy_target: policy_target_from_actions::<B>(actions_tensor.clone(), batch),
            legal_mask,
            value_target,
            grp_target,
            tenpai_target,
            danger_target,
            danger_mask,
            opp_next_target,
            score_pdf_target,
            score_cdf_target,
            oracle_target: Some(oracle_target),
            belief_fields_target: None,
            belief_fields_mask: None,
            mixture_weight_target: None,
            mixture_weight_mask: None,
            opponent_hand_type_target: None,
            delta_q_target: host.delta_q_target_flat.as_ref().map(|_| {
                f(
                    &staging.delta_q_target,
                    batch * HYDRA_ACTION_SPACE,
                    &mut gpu.delta_q_target,
                )
                .reshape([batch, HYDRA_ACTION_SPACE])
            }),
            delta_q_mask: host.delta_q_mask_flat.as_ref().map(|_| {
                burn_tensor_from_tch_f32::<B, 1>(copy_pinned_f32_to_gpu(
                    &staging.delta_q_mask,
                    batch * HYDRA_ACTION_SPACE,
                    &mut gpu.delta_q_mask,
                ))
                .reshape([batch, HYDRA_ACTION_SPACE])
            }),
            safety_residual_target: host.safety_target_flat.as_ref().map(|_| {
                f(
                    &staging.safety_target,
                    batch * HYDRA_ACTION_SPACE,
                    &mut gpu.safety_target,
                )
                .reshape([batch, HYDRA_ACTION_SPACE])
            }),
            safety_residual_mask: host.safety_mask_flat.as_ref().map(|_| {
                burn_tensor_from_tch_f32::<B, 1>(copy_pinned_f32_to_gpu(
                    &staging.safety_mask,
                    batch * HYDRA_ACTION_SPACE,
                    &mut gpu.safety_mask,
                ))
                .reshape([batch, HYDRA_ACTION_SPACE])
            }),
            oracle_guidance_mask: Some(oracle_target_mask),
            target_presence: Some(host.target_presence),
        };

        BcShardBatch {
            obs,
            batch: batch_struct,
            targets,
        }
    }
}

/// # Safety
///
/// [`PinnedStagingArea::stage`] must have been called with a host batch
/// whose `batch_size <= staging.batch_size` before this function.
/// All tensors from a previous call must have been dropped.
pub(crate) unsafe fn materialize_reuse_from_pinned_inner<B>(
    staging: &PinnedStagingArea,
    host: &BcShardHostBatch,
    batch: usize,
    gpu: &mut PreallocatedDeviceTensors,
) -> BcShardBatch<B>
where
    B: Backend<
        Device = LibTorchDevice,
        FloatTensorPrimitive = TchTensor,
        IntTensorPrimitive = TchTensor,
    >,
{
    unsafe {
        let f = |pinned: &PinnedBuffer, count: usize, dst: &mut tch::Tensor| -> Tensor<B, 1> {
            burn_tensor_from_tch_f32_inner::<B, 1>(copy_pinned_f32_to_gpu(pinned, count, dst))
        };

        let obs = f(&staging.obs, batch * OBS_SIZE, &mut gpu.obs).reshape([
            batch,
            NUM_CHANNELS,
            TILE_COUNT,
        ]);

        let actions_tensor = burn_int_tensor_from_tch_inner::<B>(copy_pinned_i64_to_gpu(
            &staging.actions,
            batch,
            &mut gpu.actions,
        ));

        let legal_mask = burn_tensor_from_tch_f32_inner::<B, 1>(copy_pinned_f32_to_gpu(
            &staging.legal_mask,
            batch * HYDRA_ACTION_SPACE,
            &mut gpu.legal_mask,
        ))
        .reshape([batch, HYDRA_ACTION_SPACE]);

        let value_target = f(&staging.value_target, batch, &mut gpu.value_target);

        let grp_target = burn_tensor_from_tch_f32_inner::<B, 1>(copy_pinned_f32_to_gpu(
            &staging.grp_target,
            batch * GRP_CLASS_COUNT,
            &mut gpu.grp_target,
        ))
        .reshape([batch, GRP_CLASS_COUNT]);

        let oracle_target = f(
            &staging.oracle_target,
            batch * PLAYER_COUNT,
            &mut gpu.oracle_target,
        )
        .reshape([batch, PLAYER_COUNT]);

        let oracle_target_mask = f(
            &staging.oracle_target_mask,
            batch,
            &mut gpu.oracle_target_mask,
        );

        let tenpai_target = burn_tensor_from_tch_f32_inner::<B, 1>(copy_pinned_f32_to_gpu(
            &staging.tenpai,
            batch * OPPONENT_COUNT,
            &mut gpu.tenpai,
        ))
        .reshape([batch, OPPONENT_COUNT]);

        let danger_target = burn_tensor_from_tch_f32_inner::<B, 1>(copy_pinned_f32_to_gpu(
            &staging.danger,
            batch * SPATIAL_TARGET_SIZE,
            &mut gpu.danger,
        ))
        .reshape([batch, OPPONENT_COUNT, TILE_COUNT]);

        let danger_mask = burn_tensor_from_tch_f32_inner::<B, 1>(copy_pinned_f32_to_gpu(
            &staging.danger_mask,
            batch * SPATIAL_TARGET_SIZE,
            &mut gpu.danger_mask,
        ))
        .reshape([batch, OPPONENT_COUNT, TILE_COUNT]);

        let opp_next_target = burn_tensor_from_tch_f32_inner::<B, 1>(copy_pinned_f32_to_gpu(
            &staging.opp_next,
            batch * SPATIAL_TARGET_SIZE,
            &mut gpu.opp_next,
        ))
        .reshape([batch, OPPONENT_COUNT, TILE_COUNT]);

        let score_pdf_target = burn_tensor_from_tch_f32_inner::<B, 1>(copy_pinned_f32_to_gpu(
            &staging.score_pdf,
            batch * SCORE_BINS,
            &mut gpu.score_pdf,
        ))
        .reshape([batch, SCORE_BINS]);
        let score_cdf_target = burn_tensor_from_tch_f32_inner::<B, 1>(copy_pinned_f32_to_gpu(
            &staging.score_cdf,
            batch * SCORE_BINS,
            &mut gpu.score_cdf,
        ))
        .reshape([batch, SCORE_BINS]);

        let exit_target_tensor = host.exit_target_flat.as_ref().map(|_| {
            f(
                &staging.exit_target,
                batch * HYDRA_ACTION_SPACE,
                &mut gpu.exit_target,
            )
            .reshape([batch, HYDRA_ACTION_SPACE])
        });
        let exit_mask_tensor = host.exit_mask_flat.as_ref().map(|_| {
            burn_tensor_from_tch_f32_inner::<B, 1>(copy_pinned_f32_to_gpu(
                &staging.exit_mask,
                batch * HYDRA_ACTION_SPACE,
                &mut gpu.exit_mask,
            ))
            .reshape([batch, HYDRA_ACTION_SPACE])
        });

        let batch_struct = MjaiBcBatch {
            actions: actions_tensor.clone(),
            exit_target: exit_target_tensor.clone(),
            exit_mask: exit_mask_tensor.clone(),
        };

        let targets = HydraTargets {
            policy_target: policy_target_from_actions::<B>(actions_tensor.clone(), batch),
            legal_mask,
            value_target,
            grp_target,
            tenpai_target,
            danger_target,
            danger_mask,
            opp_next_target,
            score_pdf_target,
            score_cdf_target,
            oracle_target: Some(oracle_target),
            belief_fields_target: None,
            belief_fields_mask: None,
            mixture_weight_target: None,
            mixture_weight_mask: None,
            opponent_hand_type_target: None,
            delta_q_target: host.delta_q_target_flat.as_ref().map(|_| {
                f(
                    &staging.delta_q_target,
                    batch * HYDRA_ACTION_SPACE,
                    &mut gpu.delta_q_target,
                )
                .reshape([batch, HYDRA_ACTION_SPACE])
            }),
            delta_q_mask: host.delta_q_mask_flat.as_ref().map(|_| {
                burn_tensor_from_tch_f32_inner::<B, 1>(copy_pinned_f32_to_gpu(
                    &staging.delta_q_mask,
                    batch * HYDRA_ACTION_SPACE,
                    &mut gpu.delta_q_mask,
                ))
                .reshape([batch, HYDRA_ACTION_SPACE])
            }),
            safety_residual_target: host.safety_target_flat.as_ref().map(|_| {
                f(
                    &staging.safety_target,
                    batch * HYDRA_ACTION_SPACE,
                    &mut gpu.safety_target,
                )
                .reshape([batch, HYDRA_ACTION_SPACE])
            }),
            safety_residual_mask: host.safety_mask_flat.as_ref().map(|_| {
                burn_tensor_from_tch_f32_inner::<B, 1>(copy_pinned_f32_to_gpu(
                    &staging.safety_mask,
                    batch * HYDRA_ACTION_SPACE,
                    &mut gpu.safety_mask,
                ))
                .reshape([batch, HYDRA_ACTION_SPACE])
            }),
            oracle_guidance_mask: Some(oracle_target_mask),
            target_presence: Some(host.target_presence),
        };

        BcShardBatch {
            obs,
            batch: batch_struct,
            targets,
        }
    }
}

/// Copy `count` f32 elements from a pinned buffer into a preallocated
/// GPU tensor via `from_blob` (zero-copy CPU view) + `copy_()`.
///
/// Returns a `tch::Tensor` that is a `narrow` view of the GPU tensor
/// covering exactly `count` elements, safe to reshape downstream.
///
/// # Safety
///
/// The pinned buffer must contain at least `count` valid f32 elements.
unsafe fn copy_pinned_f32_to_gpu(
    pinned: &PinnedBuffer,
    count: usize,
    gpu_dst: &mut tch::Tensor,
) -> tch::Tensor {
    let dims = [count as i64];
    let cpu_view = unsafe {
        tch::Tensor::from_blob(
            pinned.as_ptr(),
            &dims,
            &[], // default strides
            tch::Kind::Float,
            tch::Device::Cpu,
        )
    };
    // narrow to the actual batch count (preallocated may be larger)
    let mut dst_slice = gpu_dst.narrow(0, 0, count as i64);
    dst_slice.copy_(&cpu_view);
    dst_slice
}

/// Like [`copy_pinned_f32_to_gpu`] but for i64 data.
///
/// # Safety
///
/// The pinned buffer must contain at least `count` valid i64 elements.
unsafe fn copy_pinned_i64_to_gpu(
    pinned: &PinnedBuffer,
    count: usize,
    gpu_dst: &mut tch::Tensor,
) -> tch::Tensor {
    let dims = [count as i64];
    let cpu_view = unsafe {
        tch::Tensor::from_blob(
            pinned.as_ptr(),
            &dims,
            &[],
            tch::Kind::Int64,
            tch::Device::Cpu,
        )
    };
    let mut dst_slice = gpu_dst.narrow(0, 0, count as i64);
    dst_slice.copy_(&cpu_view);
    dst_slice
}

fn burn_tensor_from_tch_f32<B, const D: usize>(t: tch::Tensor) -> Tensor<B, D>
where
    B: AutodiffBackend<
        Device = LibTorchDevice,
        InnerBackend: Backend<FloatTensorPrimitive = TchTensor>,
    >,
{
    let tch_tensor = TchTensor::new(t);
    Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(B::from_inner(
        tch_tensor,
    )))
}

fn burn_tensor_from_tch_f32_inner<B, const D: usize>(t: tch::Tensor) -> Tensor<B, D>
where
    B: Backend<Device = LibTorchDevice, FloatTensorPrimitive = TchTensor>,
{
    let tch_tensor = TchTensor::new(t);
    Tensor::from_primitive(burn::tensor::TensorPrimitive::Float(tch_tensor))
}

fn burn_int_tensor_from_tch<B>(t: tch::Tensor) -> Tensor<B, 1, Int>
where
    B: AutodiffBackend<
        Device = LibTorchDevice,
        InnerBackend: Backend<IntTensorPrimitive = TchTensor>,
    >,
{
    let tch_tensor = TchTensor::new(t);
    Tensor::from_primitive(B::int_from_inner(tch_tensor))
}

fn burn_int_tensor_from_tch_inner<B>(t: tch::Tensor) -> Tensor<B, 1, Int>
where
    B: Backend<Device = LibTorchDevice, IntTensorPrimitive = TchTensor>,
{
    let tch_tensor = TchTensor::new(t);
    Tensor::from_primitive(tch_tensor)
}

fn copy_f32_to_pinned(src: &[f32], dst: &mut PinnedBuffer) {
    let byte_count = std::mem::size_of_val(src);
    assert!(
        byte_count <= dst.len(),
        "f32 copy overflow: {} bytes into {} byte pinned buffer",
        byte_count,
        dst.len(),
    );
    unsafe {
        ptr::copy_nonoverlapping(src.as_ptr().cast::<u8>(), dst.as_mut_ptr(), byte_count);
    }
}

/// Copy `&[i64]` into a pinned buffer.
fn copy_i64_to_pinned(src: &[i64], dst: &mut PinnedBuffer) {
    let byte_count = std::mem::size_of_val(src);
    assert!(
        byte_count <= dst.len(),
        "i64 copy overflow: {} bytes into {} byte pinned buffer",
        byte_count,
        dst.len(),
    );
    unsafe {
        ptr::copy_nonoverlapping(src.as_ptr().cast::<u8>(), dst.as_mut_ptr(), byte_count);
    }
}

fn policy_target_from_actions<B: Backend>(
    actions: Tensor<B, 1, Int>,
    batch_size: usize,
) -> Tensor<B, 2> {
    actions
        .one_hot::<2>(HYDRA_ACTION_SPACE)
        .reshape([batch_size, HYDRA_ACTION_SPACE])
        .float()
}
