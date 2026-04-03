use burn::backend::libtorch::LibTorchDevice;
use burn::optim::{GradientsAccumulator, GradientsParams};
use burn::prelude::Tensor;
use burn::tensor::backend::{AutodiffBackend, Backend};

use hydra_train::amp::maybe_autocast;
use hydra_train::data::sample::{collate_samples, collate_samples_bc_owned, MjaiSample};
use hydra_train::model::HydraModel;
use hydra_train::preflight::{
    PROFILING_STAGE_BACKWARD, PROFILING_STAGE_COLLATION, PROFILING_STAGE_FORWARD,
    PROFILING_STAGE_LOSS,
};
use hydra_train::training::bc::{gated_bc_context, maybe_add_exit_loss, BcExitConfig};
use hydra_train::training::head_gates::HeadActivationController;
use hydra_train::training::losses::{HydraLoss, LossBreakdown};

use crate::progress::{batch_metric_sums_from_outputs, batch_stats_from_metric_sums, BatchStats};

use std::time::Instant;

use crate::epoch_runner::TrainSubStageTiming;

type ValidBackendOf<B> = <B as AutodiffBackend>::InnerBackend;

pub(super) struct FixedShapeTrainStepOutput {
    pub(super) grads: GradientsParams,
    pub(super) batch_stats: BatchStats,
    pub(super) sub_stage_timing: TrainSubStageTiming,
}

pub(super) struct FixedShapeBenchmarkStepOutput {
    pub(super) grads: GradientsParams,
    pub(super) batch_stats: Vec<BatchStats>,
}

pub(super) struct FixedShapeTrainConfig<'a, B>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<Device = LibTorchDevice>,
{
    pub(super) logical_batch: &'a [MjaiSample],
    pub(super) augment: bool,
    pub(super) microbatch_size: usize,
    pub(super) train_device: &'a LibTorchDevice,
    pub(super) loss_fn: &'a HydraLoss<B>,
    pub(super) bc_exit_cfg: &'a BcExitConfig,
    pub(super) head_controller: &'a mut HeadActivationController,
    pub(super) model: &'a HydraModel<B>,
    pub(super) use_amp: bool,
}

pub(super) struct FixedShapeProbeConfig<'a, B>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<Device = LibTorchDevice>,
{
    pub(super) logical_batch: &'a [MjaiSample],
    pub(super) augment: bool,
    pub(super) microbatch_size: usize,
    pub(super) train_device: &'a LibTorchDevice,
    pub(super) loss_fn: &'a HydraLoss<B>,
    pub(super) model: &'a HydraModel<B>,
    pub(super) use_amp: bool,
}

fn accumulate_metric_sums<B: AutodiffBackend<Device = LibTorchDevice>>(
    total_samples: usize,
    microbatch_count: usize,
    metric_sums: Tensor<B, 1>,
) -> BatchStats
where
    ValidBackendOf<B>: Backend<Device = LibTorchDevice>,
{
    batch_stats_from_metric_sums(total_samples, microbatch_count, metric_sums)
}

fn split_divisible_prefix(
    logical_batch: &[MjaiSample],
    microbatch_size: usize,
) -> (&[MjaiSample], &[MjaiSample]) {
    let fixed_shape_prefix_len = logical_batch.len() / microbatch_size * microbatch_size;
    logical_batch.split_at(fixed_shape_prefix_len)
}

fn merge_metric_sums<B: Backend>(
    metric_sums: &mut Option<Tensor<B, 1>>,
    chunk_metric_sums: Tensor<B, 1>,
) {
    *metric_sums = Some(match metric_sums.take() {
        Some(existing) => existing + chunk_metric_sums,
        None => chunk_metric_sums,
    });
}

pub(super) fn run_train_logical_batch_fixed_chunks<B>(
    config: FixedShapeTrainConfig<'_, B>,
) -> Result<Option<FixedShapeTrainStepOutput>, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<Device = LibTorchDevice>,
{
    let FixedShapeTrainConfig {
        logical_batch,
        augment,
        microbatch_size,
        train_device,
        loss_fn,
        bc_exit_cfg,
        head_controller,
        model,
        use_amp,
    } = config;
    if logical_batch.is_empty() {
        return Ok(None);
    }
    if microbatch_size == 0 {
        return Err("fixed-shape executor requires microbatch_size > 0".to_string());
    }

    let logical_batch_len = logical_batch.len().max(1) as f32;
    let (fixed_shape_prefix, tail_remainder) =
        split_divisible_prefix(logical_batch, microbatch_size);
    let mut accumulator: GradientsAccumulator<HydraModel<B>> = GradientsAccumulator::new();
    let mut metric_sums: Option<Tensor<B, 1>> = None;
    let mut total_samples = 0usize;
    let mut microbatch_count = 0usize;
    let mut sub_timing = TrainSubStageTiming::default();

    for chunk in fixed_shape_prefix.chunks_exact(microbatch_size) {
        let t = Instant::now();
        let collated = {
            let _collation_scope = super::nvtx::scope(PROFILING_STAGE_COLLATION);
            collate_samples_bc_owned::<B>(chunk, augment, train_device)
                .map_err(|err| format!("training collation failed: {err}"))?
        };
        sub_timing.collation_seconds += t.elapsed().as_secs_f64();
        let Some((obs, batch, targets)) = collated else {
            continue;
        };
        let (active_loss_fn, warmup_heads) =
            gated_bc_context(Some(head_controller), loss_fn, &targets);
        let t = Instant::now();
        let output = {
            let _forward_scope = super::nvtx::scope(PROFILING_STAGE_FORWARD);
            maybe_autocast(use_amp, || {
                model.forward_with_warmup(obs, &active_loss_fn.config, &warmup_heads)
            })
        };
        sub_timing.forward_seconds += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let (breakdown, total) = {
            let _loss_scope = super::nvtx::scope(PROFILING_STAGE_LOSS);
            let breakdown: LossBreakdown<B> = active_loss_fn.total_loss(&output, &targets);
            let total = maybe_add_exit_loss(
                breakdown.total.clone(),
                output.policy_logits.clone(),
                batch.exit_target.as_ref(),
                batch.exit_mask.as_ref(),
                bc_exit_cfg,
            );
            (breakdown, total)
        };
        sub_timing.loss_seconds += t.elapsed().as_secs_f64();
        let chunk_weight = chunk.len() as f32 / logical_batch_len;
        let weighted_total = total.clone() * chunk_weight;
        let chunk_metric_sums = batch_metric_sums_from_outputs(
            chunk.len(),
            output.policy_logits.clone(),
            targets.legal_mask.clone(),
            batch.actions.clone(),
            total,
            &breakdown,
        );
        merge_metric_sums(&mut metric_sums, chunk_metric_sums);
        total_samples += chunk.len();
        microbatch_count += 1;
        {
            let t = Instant::now();
            let _backward_scope = super::nvtx::scope(PROFILING_STAGE_BACKWARD);
            let grads = weighted_total.backward();
            let grads = GradientsParams::from_grads(grads, model);
            accumulator.accumulate(model, grads);
            sub_timing.backward_seconds += t.elapsed().as_secs_f64();
        }
    }

    if !tail_remainder.is_empty() {
        let t = Instant::now();
        let collated = {
            let _collation_scope = super::nvtx::scope(PROFILING_STAGE_COLLATION);
            collate_samples_bc_owned::<B>(tail_remainder, augment, train_device)
                .map_err(|err| format!("training collation failed: {err}"))?
        };
        sub_timing.collation_seconds += t.elapsed().as_secs_f64();
        let Some((obs, batch, targets)) = collated else {
            return Ok(None);
        };
        let (active_loss_fn, warmup_heads) =
            gated_bc_context(Some(head_controller), loss_fn, &targets);
        let t = Instant::now();
        let output = {
            let _forward_scope = super::nvtx::scope(PROFILING_STAGE_FORWARD);
            maybe_autocast(use_amp, || {
                model.forward_with_warmup(obs, &active_loss_fn.config, &warmup_heads)
            })
        };
        sub_timing.forward_seconds += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let (breakdown, total) = {
            let _loss_scope = super::nvtx::scope(PROFILING_STAGE_LOSS);
            let breakdown: LossBreakdown<B> = active_loss_fn.total_loss(&output, &targets);
            let total = maybe_add_exit_loss(
                breakdown.total.clone(),
                output.policy_logits.clone(),
                batch.exit_target.as_ref(),
                batch.exit_mask.as_ref(),
                bc_exit_cfg,
            );
            (breakdown, total)
        };
        sub_timing.loss_seconds += t.elapsed().as_secs_f64();
        let chunk_weight = tail_remainder.len() as f32 / logical_batch_len;
        let weighted_total = total.clone() * chunk_weight;
        let chunk_metric_sums = batch_metric_sums_from_outputs(
            tail_remainder.len(),
            output.policy_logits.clone(),
            targets.legal_mask.clone(),
            batch.actions.clone(),
            total.clone(),
            &breakdown,
        );
        merge_metric_sums(&mut metric_sums, chunk_metric_sums);
        total_samples += tail_remainder.len();
        microbatch_count += 1;
        {
            let t = Instant::now();
            let _backward_scope = super::nvtx::scope(PROFILING_STAGE_BACKWARD);
            let grads = weighted_total.backward();
            let grads = GradientsParams::from_grads(grads, model);
            accumulator.accumulate(model, grads);
            sub_timing.backward_seconds += t.elapsed().as_secs_f64();
        }
    }

    let Some(metric_sums) = metric_sums else {
        return Ok(None);
    };

    Ok(Some(FixedShapeTrainStepOutput {
        grads: accumulator.grads(),
        batch_stats: accumulate_metric_sums(total_samples, microbatch_count, metric_sums),
        sub_stage_timing: sub_timing,
    }))
}

pub(super) fn probe_train_fixed_chunks<B>(
    config: FixedShapeProbeConfig<'_, B>,
) -> Result<Option<GradientsParams>, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<Device = LibTorchDevice>,
{
    let FixedShapeProbeConfig {
        logical_batch,
        augment,
        microbatch_size,
        train_device,
        loss_fn,
        model,
        use_amp,
    } = config;
    if logical_batch.is_empty() {
        return Ok(None);
    }
    if microbatch_size == 0 {
        return Err("fixed-shape probe executor requires microbatch_size > 0".to_string());
    }

    let logical_batch_len = logical_batch.len().max(1) as f32;
    let (fixed_shape_prefix, tail_remainder) =
        split_divisible_prefix(logical_batch, microbatch_size);
    let mut accumulator: GradientsAccumulator<HydraModel<B>> = GradientsAccumulator::new();

    for chunk in fixed_shape_prefix.chunks_exact(microbatch_size) {
        let Some((obs, targets)) = collate_samples::<B>(chunk, augment, train_device)
            .map_err(|err| format!("fixed-shape probe collation failed: {err}"))?
        else {
            continue;
        };
        let output = maybe_autocast(use_amp, || model.forward(obs));
        let breakdown = loss_fn.total_loss(&output, &targets);
        let chunk_weight = chunk.len() as f32 / logical_batch_len;
        let grads = (breakdown.total * chunk_weight).backward();
        let grads = GradientsParams::from_grads(grads, model);
        accumulator.accumulate(model, grads);
    }

    if !tail_remainder.is_empty() {
        let Some((obs, targets)) = collate_samples::<B>(tail_remainder, augment, train_device)
            .map_err(|err| format!("fixed-shape probe collation failed: {err}"))?
        else {
            return Ok(Some(accumulator.grads()));
        };
        let output = maybe_autocast(use_amp, || model.forward(obs));
        let breakdown = loss_fn.total_loss(&output, &targets);
        let chunk_weight = tail_remainder.len() as f32 / logical_batch_len;
        let grads = (breakdown.total * chunk_weight).backward();
        let grads = GradientsParams::from_grads(grads, model);
        accumulator.accumulate(model, grads);
    }

    Ok(Some(accumulator.grads()))
}

pub(super) fn benchmark_train_fixed_chunks<B>(
    config: FixedShapeTrainConfig<'_, B>,
) -> Result<Option<FixedShapeBenchmarkStepOutput>, String>
where
    B: AutodiffBackend<Device = LibTorchDevice>,
    ValidBackendOf<B>: Backend<Device = LibTorchDevice>,
{
    let FixedShapeTrainConfig {
        logical_batch,
        augment,
        microbatch_size,
        train_device,
        loss_fn,
        bc_exit_cfg,
        head_controller,
        model,
        use_amp,
    } = config;
    if logical_batch.is_empty() {
        return Ok(None);
    }
    if microbatch_size == 0 {
        return Err("fixed-shape benchmark executor requires microbatch_size > 0".to_string());
    }

    let logical_batch_len = logical_batch.len().max(1) as f32;
    let (fixed_shape_prefix, tail_remainder) =
        split_divisible_prefix(logical_batch, microbatch_size);
    let mut accumulator: GradientsAccumulator<HydraModel<B>> = GradientsAccumulator::new();
    let mut step_batches = Vec::with_capacity(
        fixed_shape_prefix.len() / microbatch_size + usize::from(!tail_remainder.is_empty()),
    );

    for chunk in fixed_shape_prefix.chunks_exact(microbatch_size) {
        let Some((obs, batch, targets)) =
            collate_samples_bc_owned::<B>(chunk, augment, train_device)
                .map_err(|err| format!("benchmark train collation failed: {err}"))?
        else {
            continue;
        };
        let (active_loss_fn, warmup_heads) =
            gated_bc_context(Some(head_controller), loss_fn, &targets);
        let output = maybe_autocast(use_amp, || {
            model.forward_with_warmup(obs, &active_loss_fn.config, &warmup_heads)
        });
        let breakdown: LossBreakdown<B> = active_loss_fn.total_loss(&output, &targets);
        let total = maybe_add_exit_loss(
            breakdown.total.clone(),
            output.policy_logits.clone(),
            batch.exit_target.as_ref(),
            batch.exit_mask.as_ref(),
            bc_exit_cfg,
        );
        step_batches.push(BatchStats {
            sample_count: chunk.len(),
            batch_count: 1,
            ..accumulate_metric_sums(
                chunk.len(),
                1,
                batch_metric_sums_from_outputs(
                    chunk.len(),
                    output.policy_logits.clone(),
                    targets.legal_mask.clone(),
                    batch.actions.clone(),
                    total.clone(),
                    &breakdown,
                ),
            )
        });
        let chunk_weight = chunk.len() as f32 / logical_batch_len;
        let grads = (total * chunk_weight).backward();
        let grads = GradientsParams::from_grads(grads, model);
        accumulator.accumulate(model, grads);
    }

    if !tail_remainder.is_empty() {
        let Some((obs, batch, targets)) =
            collate_samples_bc_owned::<B>(tail_remainder, augment, train_device)
                .map_err(|err| format!("benchmark train collation failed: {err}"))?
        else {
            return Ok(Some(FixedShapeBenchmarkStepOutput {
                grads: accumulator.grads(),
                batch_stats: step_batches,
            }));
        };
        let (active_loss_fn, warmup_heads) =
            gated_bc_context(Some(head_controller), loss_fn, &targets);
        let output = maybe_autocast(use_amp, || {
            model.forward_with_warmup(obs, &active_loss_fn.config, &warmup_heads)
        });
        let breakdown = active_loss_fn.total_loss(&output, &targets);
        let total = maybe_add_exit_loss(
            breakdown.total.clone(),
            output.policy_logits.clone(),
            batch.exit_target.as_ref(),
            batch.exit_mask.as_ref(),
            bc_exit_cfg,
        );
        step_batches.push(BatchStats {
            sample_count: tail_remainder.len(),
            batch_count: 1,
            ..accumulate_metric_sums(
                tail_remainder.len(),
                1,
                batch_metric_sums_from_outputs(
                    tail_remainder.len(),
                    output.policy_logits.clone(),
                    targets.legal_mask.clone(),
                    batch.actions.clone(),
                    total.clone(),
                    &breakdown,
                ),
            )
        });
        let chunk_weight = tail_remainder.len() as f32 / logical_batch_len;
        let grads = (total * chunk_weight).backward();
        let grads = GradientsParams::from_grads(grads, model);
        accumulator.accumulate(model, grads);
    }

    Ok(Some(FixedShapeBenchmarkStepOutput {
        grads: accumulator.grads(),
        batch_stats: step_batches,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::validation_batch_stats;
    use burn::backend::{Autodiff, LibTorch};
    use burn::optim::{AdamConfig, GradientsAccumulator, GradientsParams, Optimizer};
    use burn::prelude::Tensor;
    use hydra_train::config::INPUT_CHANNELS;
    use hydra_train::data::sample::{
        collate_batch_samples, collate_samples, collate_samples_owned,
    };
    use hydra_train::training::bc::bc_total_with_exit_from_breakdown;
    use hydra_train::training::head_gates::HeadActivationConfig;
    use hydra_train::training::losses::HydraLossConfig;

    type TestTrainBackend = Autodiff<LibTorch<f32>>;

    fn tiny_dummy_model(device: &LibTorchDevice) -> HydraModel<TestTrainBackend> {
        hydra_train::model::HydraModelConfig::new(1)
            .with_input_channels(INPUT_CHANNELS)
            .with_hidden_channels(4)
            .with_num_groups(4)
            .with_se_bottleneck(1)
            .init::<TestTrainBackend>(device)
    }

    fn dummy_train_sample(action: u8) -> MjaiSample {
        let mut legal_mask = [0.0f32; hydra_core::action::HYDRA_ACTION_SPACE];
        legal_mask[action as usize] = 1.0;
        legal_mask[45] = 1.0;
        MjaiSample {
            obs: [0.1f32; hydra_core::encoder::OBS_SIZE],
            action,
            legal_mask,
            placement: 0,
            score_delta: 0,
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

    fn dummy_train_sample_with_exit(action: u8, exit_target: f32, exit_mask: f32) -> MjaiSample {
        let mut sample = dummy_train_sample(action);
        let mut exit_target_vec = [0.0f32; hydra_core::action::HYDRA_ACTION_SPACE];
        exit_target_vec[action as usize] = exit_target;
        let mut exit_mask_vec = [0.0f32; hydra_core::action::HYDRA_ACTION_SPACE];
        exit_mask_vec[action as usize] = exit_mask;
        sample.exit_target = Some(exit_target_vec);
        sample.exit_mask = Some(exit_mask_vec);
        sample
    }

    fn dummy_train_loss() -> HydraLoss<TestTrainBackend> {
        HydraLoss::<TestTrainBackend>::new(HydraLossConfig::new())
    }

    fn assert_close(actual: f64, expected: f64) {
        let diff = (actual - expected).abs();
        let scale = actual.abs().max(expected.abs()).max(1.0);
        assert!(
            diff < 1e-5 || diff / scale < 1e-5,
            "expected {expected}, got {actual} (abs diff {diff}, rel diff {})",
            diff / scale,
        );
    }

    fn assert_batch_stats_close(actual: BatchStats, expected: BatchStats) {
        assert_eq!(actual.sample_count, expected.sample_count);
        assert_eq!(actual.batch_count, expected.batch_count);
        assert_close(actual.total_loss, expected.total_loss);
        assert_close(actual.policy_agreement, expected.policy_agreement);
        assert_close(actual.loss_policy, expected.loss_policy);
        assert_close(actual.loss_value, expected.loss_value);
        assert_close(actual.loss_grp, expected.loss_grp);
        assert_close(actual.loss_tenpai, expected.loss_tenpai);
        assert_close(actual.loss_danger, expected.loss_danger);
        assert_close(actual.loss_opp_next, expected.loss_opp_next);
        assert_close(actual.loss_score_pdf, expected.loss_score_pdf);
        assert_close(actual.loss_score_cdf, expected.loss_score_cdf);
    }

    struct GenericTrainParityContext<'a> {
        augment: bool,
        microbatch_size: usize,
        train_device: &'a LibTorchDevice,
        loss_fn: &'a HydraLoss<TestTrainBackend>,
        bc_exit_cfg: &'a BcExitConfig,
        head_controller: &'a mut HeadActivationController,
        model: &'a HydraModel<TestTrainBackend>,
    }

    struct GenericProbeParityContext<'a> {
        augment: bool,
        microbatch_size: usize,
        train_device: &'a LibTorchDevice,
        loss_fn: &'a HydraLoss<TestTrainBackend>,
        model: &'a HydraModel<TestTrainBackend>,
    }

    fn generic_train_batch_stats(
        logical_batch: &[MjaiSample],
        context: GenericTrainParityContext<'_>,
    ) -> Option<BatchStats> {
        let GenericTrainParityContext {
            augment,
            microbatch_size,
            train_device,
            loss_fn,
            bc_exit_cfg,
            head_controller,
            model,
        } = context;
        if logical_batch.is_empty() {
            return None;
        }

        let mut metric_sums: Option<Tensor<TestTrainBackend, 1>> = None;
        let mut total_samples = 0usize;
        let mut microbatch_count = 0usize;

        for chunk in logical_batch.chunks(microbatch_size.max(1)) {
            let Some((obs, batch, targets)) =
                collate_samples_owned::<TestTrainBackend>(chunk, augment, train_device)
                    .expect("generic train collation should succeed")
            else {
                continue;
            };
            let (active_loss_fn, warmup_heads) =
                gated_bc_context(Some(head_controller), loss_fn, &targets);
            let output =
                model.forward_with_warmup(obs.clone(), &active_loss_fn.config, &warmup_heads);
            let breakdown = active_loss_fn.total_loss(&output, &targets);
            let total = bc_total_with_exit_from_breakdown(&output, &batch, &breakdown, bc_exit_cfg);
            let chunk_metric_sums = batch_metric_sums_from_outputs(
                chunk.len(),
                output.policy_logits.clone(),
                targets.legal_mask.clone(),
                batch.actions.clone(),
                total,
                &breakdown,
            );
            merge_metric_sums(&mut metric_sums, chunk_metric_sums);
            total_samples += chunk.len();
            microbatch_count += 1;
        }

        metric_sums
            .map(|metric_sums| accumulate_metric_sums(total_samples, microbatch_count, metric_sums))
    }

    fn generic_probe_grads(
        logical_batch: &[MjaiSample],
        context: GenericProbeParityContext<'_>,
    ) -> GradientsParams {
        let GenericProbeParityContext {
            augment,
            microbatch_size,
            train_device,
            loss_fn,
            model,
        } = context;
        let logical_batch_len = logical_batch.len().max(1) as f32;
        let mut accumulator: GradientsAccumulator<HydraModel<TestTrainBackend>> =
            GradientsAccumulator::new();

        for chunk in logical_batch.chunks(microbatch_size.max(1)) {
            let Some((obs, targets)) =
                collate_samples::<TestTrainBackend>(chunk, augment, train_device)
                    .expect("generic probe collation should succeed")
            else {
                continue;
            };
            let output = model.forward(obs);
            let breakdown = loss_fn.total_loss(&output, &targets);
            let chunk_weight = chunk.len() as f32 / logical_batch_len;
            let grads = (breakdown.total * chunk_weight).backward();
            let grads = GradientsParams::from_grads(grads, model);
            accumulator.accumulate(model, grads);
        }

        accumulator.grads()
    }

    fn generic_benchmark_step(
        logical_batch: &[MjaiSample],
        context: GenericTrainParityContext<'_>,
    ) -> FixedShapeBenchmarkStepOutput {
        let GenericTrainParityContext {
            augment,
            microbatch_size,
            train_device,
            loss_fn,
            bc_exit_cfg,
            head_controller,
            model,
        } = context;
        let logical_batch_len = logical_batch.len().max(1) as f32;
        let mut accumulator: GradientsAccumulator<HydraModel<TestTrainBackend>> =
            GradientsAccumulator::new();
        let mut step_batches = Vec::new();

        for chunk in logical_batch.chunks(microbatch_size.max(1)) {
            let Some((obs, batch)) =
                collate_batch_samples::<TestTrainBackend>(chunk, augment, train_device)
                    .expect("generic benchmark collation should succeed")
            else {
                continue;
            };
            let targets = batch.to_hydra_targets();
            let (active_loss_fn, warmup_heads) =
                gated_bc_context(Some(head_controller), loss_fn, &targets);
            let output =
                model.forward_with_warmup(obs.clone(), &active_loss_fn.config, &warmup_heads);
            let breakdown = active_loss_fn.total_loss(&output, &targets);
            let total = bc_total_with_exit_from_breakdown(&output, &batch, &breakdown, bc_exit_cfg);
            step_batches.push(validation_batch_stats(
                chunk.len(),
                &output,
                &batch,
                &targets,
                &breakdown,
                &total,
            ));
            let chunk_weight = chunk.len() as f32 / logical_batch_len;
            let grads = (total * chunk_weight).backward();
            let grads = GradientsParams::from_grads(grads, model);
            accumulator.accumulate(model, grads);
        }

        FixedShapeBenchmarkStepOutput {
            grads: accumulator.grads(),
            batch_stats: step_batches,
        }
    }

    fn step_model(
        model: HydraModel<TestTrainBackend>,
        grads: GradientsParams,
    ) -> HydraModel<TestTrainBackend> {
        let mut optimizer = AdamConfig::new().init();
        optimizer.step(1e-4, model, grads)
    }

    fn sample_policy_logits(
        model: &HydraModel<TestTrainBackend>,
        sample: &MjaiSample,
        train_device: &LibTorchDevice,
    ) -> Vec<f32> {
        let (obs, _) =
            collate_samples::<TestTrainBackend>(std::slice::from_ref(sample), false, train_device)
                .expect("single-sample probe collation should succeed")
                .expect("single-sample probe collation should produce tensors");
        model
            .forward(obs)
            .policy_logits
            .to_data()
            .convert::<f32>()
            .as_slice::<f32>()
            .expect("policy logits should be readable as f32")
            .to_vec()
    }

    #[test]
    fn fixed_shape_train_chunks_metrics_match_across_chunk_sizes() {
        let device = LibTorchDevice::Cpu;
        let base_model = tiny_dummy_model(&device);
        let mut head_controller_single =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));
        let mut head_controller_split =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));
        let train_loss_fn = dummy_train_loss();
        let logical_batch = vec![dummy_train_sample(0), dummy_train_sample(5)];

        let single = run_train_logical_batch_fixed_chunks(FixedShapeTrainConfig {
            logical_batch: &logical_batch,
            augment: false,
            microbatch_size: logical_batch.len(),
            train_device: &device,
            loss_fn: &train_loss_fn,
            bc_exit_cfg: &BcExitConfig::default(),
            head_controller: &mut head_controller_single,
            model: &base_model,
            use_amp: false,
        })
        .expect("single fixed-shape path should succeed")
        .expect("single fixed-shape path should return stats");

        let split = run_train_logical_batch_fixed_chunks(FixedShapeTrainConfig {
            logical_batch: &logical_batch,
            augment: false,
            microbatch_size: 1,
            train_device: &device,
            loss_fn: &train_loss_fn,
            bc_exit_cfg: &BcExitConfig::default(),
            head_controller: &mut head_controller_split,
            model: &base_model,
            use_amp: false,
        })
        .expect("split fixed-shape path should succeed")
        .expect("split fixed-shape path should return stats");

        assert_eq!(single.batch_stats.sample_count, logical_batch.len());
        assert_eq!(split.batch_stats.sample_count, logical_batch.len());
        assert_eq!(single.batch_stats.batch_count, 1);
        assert_eq!(split.batch_stats.batch_count, logical_batch.len());
        assert_close(split.batch_stats.total_loss, single.batch_stats.total_loss);
        assert_close(
            split.batch_stats.policy_agreement,
            single.batch_stats.policy_agreement,
        );
        assert_close(
            split.batch_stats.loss_policy,
            single.batch_stats.loss_policy,
        );
        assert_close(split.batch_stats.loss_value, single.batch_stats.loss_value);
        assert_close(split.batch_stats.loss_grp, single.batch_stats.loss_grp);
        assert_close(
            split.batch_stats.loss_tenpai,
            single.batch_stats.loss_tenpai,
        );
        assert_close(
            split.batch_stats.loss_danger,
            single.batch_stats.loss_danger,
        );
        assert_close(
            split.batch_stats.loss_opp_next,
            single.batch_stats.loss_opp_next,
        );
        assert_close(
            split.batch_stats.loss_score_pdf,
            single.batch_stats.loss_score_pdf,
        );
        assert_close(
            split.batch_stats.loss_score_cdf,
            single.batch_stats.loss_score_cdf,
        );
    }

    #[test]
    fn fixed_shape_train_chunks_match_generic_for_non_divisible_batches() {
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let mut head_controller_mixed =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));
        let mut head_controller_generic =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));
        let train_loss_fn = dummy_train_loss();
        let logical_batch = vec![
            dummy_train_sample(0),
            dummy_train_sample(5),
            dummy_train_sample(11),
        ];

        let mixed = run_train_logical_batch_fixed_chunks(FixedShapeTrainConfig {
            logical_batch: &logical_batch,
            augment: false,
            microbatch_size: 2,
            train_device: &device,
            loss_fn: &train_loss_fn,
            bc_exit_cfg: &BcExitConfig::default(),
            head_controller: &mut head_controller_mixed,
            model: &model,
            use_amp: false,
        })
        .expect("non-divisible logical batch should not error")
        .expect("mixed fixed-shape train path should return stats");

        let generic = generic_train_batch_stats(
            &logical_batch,
            GenericTrainParityContext {
                augment: false,
                microbatch_size: 2,
                train_device: &device,
                loss_fn: &train_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                head_controller: &mut head_controller_generic,
                model: &model,
            },
        )
        .expect("generic train path should return stats");

        assert_eq!(mixed.batch_stats.sample_count, logical_batch.len());
        assert_eq!(mixed.batch_stats.batch_count, 2);
        assert_batch_stats_close(mixed.batch_stats, generic);
    }

    #[test]
    fn fixed_shape_probe_chunks_match_generic_for_non_divisible_batches() {
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let train_loss_fn = dummy_train_loss();
        let logical_batch = vec![
            dummy_train_sample(0),
            dummy_train_sample(5),
            dummy_train_sample(11),
        ];

        let mixed_grads = probe_train_fixed_chunks(FixedShapeProbeConfig {
            logical_batch: &logical_batch,
            augment: false,
            microbatch_size: 2,
            train_device: &device,
            loss_fn: &train_loss_fn,
            model: &model,
            use_amp: false,
        })
        .expect("non-divisible probe batch should not error")
        .expect("mixed fixed-shape probe path should return gradients");

        let generic_grads = generic_probe_grads(
            &logical_batch,
            GenericProbeParityContext {
                augment: false,
                microbatch_size: 2,
                train_device: &device,
                loss_fn: &train_loss_fn,
                model: &model,
            },
        );

        let mixed_model = step_model(model.clone(), mixed_grads);
        let generic_model = step_model(model, generic_grads);
        let mixed_logits = sample_policy_logits(&mixed_model, &logical_batch[0], &device);
        let generic_logits = sample_policy_logits(&generic_model, &logical_batch[0], &device);

        assert_eq!(mixed_logits.len(), generic_logits.len());
        for (actual, expected) in mixed_logits.into_iter().zip(generic_logits) {
            assert_close(actual as f64, expected as f64);
        }
    }

    #[test]
    fn fixed_shape_benchmark_chunks_match_generic_for_non_divisible_batches() {
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let mut head_controller_mixed =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));
        let mut head_controller_generic =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));
        let train_loss_fn = dummy_train_loss();
        let logical_batch = vec![
            dummy_train_sample(0),
            dummy_train_sample(5),
            dummy_train_sample(11),
        ];

        let mixed = benchmark_train_fixed_chunks(FixedShapeTrainConfig {
            logical_batch: &logical_batch,
            augment: false,
            microbatch_size: 2,
            train_device: &device,
            loss_fn: &train_loss_fn,
            bc_exit_cfg: &BcExitConfig::default(),
            head_controller: &mut head_controller_mixed,
            model: &model,
            use_amp: false,
        })
        .expect("non-divisible benchmark batch should not error")
        .expect("mixed fixed-shape benchmark path should return step batches");

        let generic = generic_benchmark_step(
            &logical_batch,
            GenericTrainParityContext {
                augment: false,
                microbatch_size: 2,
                train_device: &device,
                loss_fn: &train_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                head_controller: &mut head_controller_generic,
                model: &model,
            },
        );

        assert_eq!(mixed.batch_stats.len(), 2);
        assert_eq!(mixed.batch_stats.len(), generic.batch_stats.len());
        for (actual, expected) in mixed.batch_stats.iter().zip(generic.batch_stats.iter()) {
            assert_batch_stats_close(*actual, *expected);
        }
    }

    #[test]
    fn fixed_shape_train_chunks_match_generic_for_non_divisible_batches_with_exit_loss() {
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let mut head_controller_mixed =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));
        let mut head_controller_generic =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));
        let train_loss_fn = dummy_train_loss();
        let logical_batch = vec![
            dummy_train_sample_with_exit(0, 1.0, 1.0),
            dummy_train_sample_with_exit(5, 0.0, 1.0),
            dummy_train_sample_with_exit(11, 1.0, 1.0),
        ];
        let exit_cfg = BcExitConfig { exit_weight: 0.25 };

        let mixed = run_train_logical_batch_fixed_chunks(FixedShapeTrainConfig {
            logical_batch: &logical_batch,
            augment: false,
            microbatch_size: 2,
            train_device: &device,
            loss_fn: &train_loss_fn,
            bc_exit_cfg: &exit_cfg,
            head_controller: &mut head_controller_mixed,
            model: &model,
            use_amp: false,
        })
        .expect("non-divisible logical batch with exit loss should not error")
        .expect("mixed fixed-shape train path should return stats");

        let generic = generic_train_batch_stats(
            &logical_batch,
            GenericTrainParityContext {
                augment: false,
                microbatch_size: 2,
                train_device: &device,
                loss_fn: &train_loss_fn,
                bc_exit_cfg: &exit_cfg,
                head_controller: &mut head_controller_generic,
                model: &model,
            },
        )
        .expect("generic train path should return stats");

        assert_eq!(mixed.batch_stats.sample_count, logical_batch.len());
        assert_eq!(mixed.batch_stats.batch_count, 2);
        assert_batch_stats_close(mixed.batch_stats, generic);
    }

    #[test]
    fn split_divisible_prefix_exact_divisible() {
        let samples: Vec<MjaiSample> = (0..4).map(dummy_train_sample).collect();
        let (prefix, tail) = split_divisible_prefix(&samples, 2);
        assert_eq!(prefix.len(), 4);
        assert_eq!(tail.len(), 0);
    }

    #[test]
    fn split_divisible_prefix_non_divisible() {
        let samples: Vec<MjaiSample> = (0..5).map(dummy_train_sample).collect();
        let (prefix, tail) = split_divisible_prefix(&samples, 2);
        assert_eq!(prefix.len(), 4);
        assert_eq!(tail.len(), 1);
    }

    #[test]
    fn split_divisible_prefix_microbatch_larger_than_batch() {
        let samples: Vec<MjaiSample> = (0..3).map(dummy_train_sample).collect();
        let (prefix, tail) = split_divisible_prefix(&samples, 10);
        assert_eq!(prefix.len(), 0);
        assert_eq!(tail.len(), 3);
    }

    #[test]
    fn split_divisible_prefix_single_sample() {
        let samples = vec![dummy_train_sample(0)];
        let (prefix, tail) = split_divisible_prefix(&samples, 1);
        assert_eq!(prefix.len(), 1);
        assert_eq!(tail.len(), 0);
    }

    #[test]
    fn split_divisible_prefix_empty() {
        let samples: Vec<MjaiSample> = vec![];
        let (prefix, tail) = split_divisible_prefix(&samples, 2);
        assert_eq!(prefix.len(), 0);
        assert_eq!(tail.len(), 0);
    }

    #[test]
    fn fixed_shape_train_returns_none_for_empty_batch() {
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let train_loss_fn = dummy_train_loss();
        let mut head_controller =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));

        let result = run_train_logical_batch_fixed_chunks(FixedShapeTrainConfig {
            logical_batch: &[],
            augment: false,
            microbatch_size: 1,
            train_device: &device,
            loss_fn: &train_loss_fn,
            bc_exit_cfg: &BcExitConfig::default(),
            head_controller: &mut head_controller,
            model: &model,
            use_amp: false,
        })
        .expect("empty batch should return Ok");

        assert!(result.is_none(), "empty batch should return None");
    }

    #[test]
    fn fixed_shape_train_rejects_zero_microbatch_size() {
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let train_loss_fn = dummy_train_loss();
        let mut head_controller =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));

        let result = run_train_logical_batch_fixed_chunks(FixedShapeTrainConfig {
            logical_batch: &[dummy_train_sample(0)],
            augment: false,
            microbatch_size: 0,
            train_device: &device,
            loss_fn: &train_loss_fn,
            bc_exit_cfg: &BcExitConfig::default(),
            head_controller: &mut head_controller,
            model: &model,
            use_amp: false,
        });

        assert!(result.is_err());
        let err_msg = result.err().expect("should be Err");
        assert!(
            err_msg.contains("microbatch_size > 0"),
            "error message should mention microbatch_size: {err_msg}"
        );
    }

    #[test]
    fn fixed_shape_train_handles_microbatch_larger_than_batch() {
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let train_loss_fn = dummy_train_loss();
        let mut head_controller =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));
        let logical_batch = vec![dummy_train_sample(0), dummy_train_sample(5)];

        let result = run_train_logical_batch_fixed_chunks(FixedShapeTrainConfig {
            logical_batch: &logical_batch,
            augment: false,
            microbatch_size: 10,
            train_device: &device,
            loss_fn: &train_loss_fn,
            bc_exit_cfg: &BcExitConfig::default(),
            head_controller: &mut head_controller,
            model: &model,
            use_amp: false,
        })
        .expect("microbatch > batch should succeed via tail remainder");

        let output = result.expect("should produce output from tail path");
        assert_eq!(output.batch_stats.sample_count, 2);
    }

    #[test]
    fn fixed_shape_nvtx_scopes_fire_for_both_prefix_and_tail_remainder() {
        let device = LibTorchDevice::Cpu;
        let model = tiny_dummy_model(&device);
        let train_loss_fn = dummy_train_loss();
        let mut head_controller =
            HeadActivationController::new(HeadActivationConfig::default_with_params(1));
        let logical_batch: Vec<MjaiSample> = (0..3).map(dummy_train_sample).collect();

        let (result, events) = crate::nvtx::with_test_recorder(|| {
            run_train_logical_batch_fixed_chunks(FixedShapeTrainConfig {
                logical_batch: &logical_batch,
                augment: false,
                microbatch_size: 2,
                train_device: &device,
                loss_fn: &train_loss_fn,
                bc_exit_cfg: &BcExitConfig::default(),
                head_controller: &mut head_controller,
                model: &model,
                use_amp: false,
            })
        });
        result.expect("non-divisible batch should succeed");

        let collation_pushes = events.iter().filter(|e| *e == "push:collation").count();
        assert_eq!(
            collation_pushes, 2,
            "should have 2 collation pushes: 1 for the fixed-shape prefix chunk + 1 for the tail remainder"
        );

        let forward_pushes = events.iter().filter(|e| *e == "push:forward").count();
        assert_eq!(forward_pushes, 2);
        let loss_pushes = events.iter().filter(|e| *e == "push:loss").count();
        assert_eq!(loss_pushes, 2);
        let backward_pushes = events.iter().filter(|e| *e == "push:backward").count();
        assert_eq!(backward_pushes, 2);

        for push_event in events.iter().filter(|e| e.starts_with("push:")) {
            let stage = push_event.strip_prefix("push:").unwrap();
            let pop = format!("pop:{stage}");
            assert!(
                events.contains(&pop),
                "every push should have a matching pop: {push_event}"
            );
        }
    }
}
