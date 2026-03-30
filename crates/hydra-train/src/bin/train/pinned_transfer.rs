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
//! - [`materialize_staged`]: stages a [`BcShardHostBatch`] into pinned
//!   buffers, issues all `Tensor::from_floats` / `from_ints` calls on the
//!   copy stream, records an event, restores the compute stream, and
//!   makes the compute stream wait on the event before returning the
//!   device-side batch.

#![cfg(feature = "cuda-graph")]

use std::ptr;

use burn::backend::libtorch::LibTorchDevice;
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
        copy_f32_to_pinned(&host.grp_target, &mut self.grp_target);
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

    /// Return a `&[f32]` slice covering exactly `count` elements from a
    /// pinned f32 buffer.
    ///
    /// # Safety
    ///
    /// The pinned buffer must have been written to with at least `count`
    /// elements via [`stage`](Self::stage) before this call.
    unsafe fn pinned_f32_slice(buf: &PinnedBuffer, count: usize) -> &[f32] {
        unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<f32>(), count) }
    }

    /// # Safety
    ///
    /// Same precondition as [`pinned_f32_slice`].
    unsafe fn pinned_i64_slice(buf: &PinnedBuffer, count: usize) -> &[i64] {
        unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<i64>(), count) }
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

/// Stage a host batch into pinned memory, issue H2D copies on the
/// dedicated copy stream, record a completion event, and make the
/// compute stream wait before returning the device batch.
///
/// The compute stream (whatever is current when this function is called)
/// will not proceed past the returned tensors until the copy stream has
/// finished writing them.  This is the single synchronization point that
/// guarantees correctness without blocking the CPU.
pub(crate) fn materialize_staged<B>(
    host: &BcShardHostBatch,
    staging: &mut PinnedStagingArea,
    h2d: &AsyncH2DContext,
    device: &LibTorchDevice,
) -> BcShardBatch<B>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
{
    let batch = host.batch_size;

    // 1. CPU memcpy: pageable Vec -> pinned staging (fast, cache-friendly)
    staging.stage(host);

    // 2. Remember the current (compute) stream so we can restore it.
    let device_index = cuda_device_index(device);
    let compute_stream = CudaStream::current(device_index);

    // 3. Switch current stream to the dedicated copy stream.
    //    All subsequent Tensor::from_floats / from_ints calls will issue
    //    cudaMemcpyAsync on this stream.
    h2d.copy_stream.set_current();

    // 4. Create device tensors from pinned slices (async DMA on copy stream).
    let shard_batch = unsafe { materialize_from_pinned::<B>(staging, host, batch, device) };

    // 5. Record an event on the copy stream marking the end of all copies.
    h2d.event.record(&h2d.copy_stream);

    // 6. Restore compute stream as the current CUDA stream.
    compute_stream.set_current();

    // 7. Make compute stream wait on the copy-complete event.
    //    This is a GPU-side dependency -- the CPU does not block.
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

/// Materialize device tensors from pinned staging buffers.
///
/// # Safety
///
/// [`PinnedStagingArea::stage`] must have been called with a host batch
/// whose `batch_size <= staging.batch_size` before this function.
unsafe fn materialize_from_pinned<B>(
    staging: &PinnedStagingArea,
    host: &BcShardHostBatch,
    batch: usize,
    device: &LibTorchDevice,
) -> BcShardBatch<B>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
{
    unsafe {
        let obs_slice = PinnedStagingArea::pinned_f32_slice(&staging.obs, batch * OBS_SIZE);
        let obs = Tensor::<B, 1>::from_floats(obs_slice, device).reshape([
            batch,
            NUM_CHANNELS,
            TILE_COUNT,
        ]);

        let actions_slice = PinnedStagingArea::pinned_i64_slice(&staging.actions, batch);
        let actions_tensor = Tensor::<B, 1, Int>::from_ints(actions_slice, device);

        let legal_mask_slice =
            PinnedStagingArea::pinned_f32_slice(&staging.legal_mask, batch * HYDRA_ACTION_SPACE);
        let legal_mask = Tensor::<B, 1>::from_floats(legal_mask_slice, device)
            .reshape([batch, HYDRA_ACTION_SPACE]);

        let value_slice = PinnedStagingArea::pinned_f32_slice(&staging.value_target, batch);
        let value_target = Tensor::<B, 1>::from_floats(value_slice, device);

        let grp_slice =
            PinnedStagingArea::pinned_f32_slice(&staging.grp_target, batch * GRP_CLASS_COUNT);
        let grp_target =
            Tensor::<B, 1>::from_floats(grp_slice, device).reshape([batch, GRP_CLASS_COUNT]);

        let oracle_slice =
            PinnedStagingArea::pinned_f32_slice(&staging.oracle_target, batch * PLAYER_COUNT);
        let oracle_target =
            Tensor::<B, 1>::from_floats(oracle_slice, device).reshape([batch, PLAYER_COUNT]);

        let oracle_mask_slice =
            PinnedStagingArea::pinned_f32_slice(&staging.oracle_target_mask, batch);
        let oracle_target_mask = Tensor::<B, 1>::from_floats(oracle_mask_slice, device);

        let tenpai_slice =
            PinnedStagingArea::pinned_f32_slice(&staging.tenpai, batch * OPPONENT_COUNT);
        let tenpai_target =
            Tensor::<B, 1>::from_floats(tenpai_slice, device).reshape([batch, OPPONENT_COUNT]);

        let danger_slice =
            PinnedStagingArea::pinned_f32_slice(&staging.danger, batch * SPATIAL_TARGET_SIZE);
        let danger_target = Tensor::<B, 1>::from_floats(danger_slice, device).reshape([
            batch,
            OPPONENT_COUNT,
            TILE_COUNT,
        ]);

        let danger_mask_slice =
            PinnedStagingArea::pinned_f32_slice(&staging.danger_mask, batch * SPATIAL_TARGET_SIZE);
        let danger_mask = Tensor::<B, 1>::from_floats(danger_mask_slice, device).reshape([
            batch,
            OPPONENT_COUNT,
            TILE_COUNT,
        ]);

        let opp_next_slice =
            PinnedStagingArea::pinned_f32_slice(&staging.opp_next, batch * SPATIAL_TARGET_SIZE);
        let opp_next_target = Tensor::<B, 1>::from_floats(opp_next_slice, device).reshape([
            batch,
            OPPONENT_COUNT,
            TILE_COUNT,
        ]);

        let score_pdf_slice =
            PinnedStagingArea::pinned_f32_slice(&staging.score_pdf, batch * SCORE_BINS);
        let score_pdf_target =
            Tensor::<B, 1>::from_floats(score_pdf_slice, device).reshape([batch, SCORE_BINS]);

        let score_cdf_slice =
            PinnedStagingArea::pinned_f32_slice(&staging.score_cdf, batch * SCORE_BINS);
        let score_cdf_target =
            Tensor::<B, 1>::from_floats(score_cdf_slice, device).reshape([batch, SCORE_BINS]);

        let exit_target_tensor = host.exit_target_flat.as_ref().map(|_| {
            let s = PinnedStagingArea::pinned_f32_slice(
                &staging.exit_target,
                batch * HYDRA_ACTION_SPACE,
            );
            Tensor::<B, 1>::from_floats(s, device).reshape([batch, HYDRA_ACTION_SPACE])
        });
        let exit_mask_tensor = host.exit_mask_flat.as_ref().map(|_| {
            let s =
                PinnedStagingArea::pinned_f32_slice(&staging.exit_mask, batch * HYDRA_ACTION_SPACE);
            Tensor::<B, 1>::from_floats(s, device).reshape([batch, HYDRA_ACTION_SPACE])
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
                let s = PinnedStagingArea::pinned_f32_slice(
                    &staging.delta_q_target,
                    batch * HYDRA_ACTION_SPACE,
                );
                Tensor::<B, 1>::from_floats(s, device).reshape([batch, HYDRA_ACTION_SPACE])
            }),
            delta_q_mask: host.delta_q_mask_flat.as_ref().map(|_| {
                let s = PinnedStagingArea::pinned_f32_slice(
                    &staging.delta_q_mask,
                    batch * HYDRA_ACTION_SPACE,
                );
                Tensor::<B, 1>::from_floats(s, device).reshape([batch, HYDRA_ACTION_SPACE])
            }),
            safety_residual_target: host.safety_target_flat.as_ref().map(|_| {
                let s = PinnedStagingArea::pinned_f32_slice(
                    &staging.safety_target,
                    batch * HYDRA_ACTION_SPACE,
                );
                Tensor::<B, 1>::from_floats(s, device).reshape([batch, HYDRA_ACTION_SPACE])
            }),
            safety_residual_mask: host.safety_mask_flat.as_ref().map(|_| {
                let s = PinnedStagingArea::pinned_f32_slice(
                    &staging.safety_mask,
                    batch * HYDRA_ACTION_SPACE,
                );
                Tensor::<B, 1>::from_floats(s, device).reshape([batch, HYDRA_ACTION_SPACE])
            }),
            oracle_guidance_mask: Some(oracle_target_mask),
            target_presence: Some(host.target_presence.clone()),
        };

        BcShardBatch {
            obs,
            batch: batch_struct,
            targets,
        }
    }
}

/// Copy `&[f32]` into a pinned buffer.
fn copy_f32_to_pinned(src: &[f32], dst: &mut PinnedBuffer) {
    let byte_count = src.len() * std::mem::size_of::<f32>();
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
    let byte_count = src.len() * std::mem::size_of::<i64>();
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
    let mut one_hot = vec![0.0f32; batch_size * HYDRA_ACTION_SPACE];
    let action_data = actions.clone().into_data().convert::<i64>();
    let action_values = action_data
        .as_slice::<i64>()
        .expect("action tensor should be readable as i64");
    for (row, &action) in action_values.iter().enumerate() {
        let action = action as usize;
        if action < HYDRA_ACTION_SPACE {
            one_hot[row * HYDRA_ACTION_SPACE + action] = 1.0;
        }
    }
    Tensor::<B, 1>::from_floats(one_hot.as_slice(), &actions.device())
        .reshape([batch_size, HYDRA_ACTION_SPACE])
}
