use super::*;
use crate::bc_runtime::bc_total_with_exit_from_breakdown;
#[cfg(feature = "bf16-autocast-proof")]
use crate::bf16_autocast_proof::{
    current_cuda_autocast_state, restore_cuda_autocast_state,
    with_cuda_bf16_autocast_dtype_proof_only,
};
use crate::data::sample::{
    MjaiSample, collate_batch_samples, collate_samples, collate_samples_owned,
};
use crate::model::HydraModelInit;
use burn::backend::{Autodiff, LibTorch};
use burn::module::{AutodiffModule, Module, ModuleVisitor, Param};
#[cfg(feature = "bf16-autocast-proof")]
use burn::optim::record::{AdaptorRecord, AdaptorRecordV1};
#[cfg(feature = "bf16-autocast-proof")]
use burn::optim::{Adam, AdamState};
use burn::optim::{AdamConfig, GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::Backend;
#[cfg(feature = "bf16-autocast-proof")]
use burn::record::{BinFileRecorder, Recorder};
use burn::record::{FullPrecisionSettings, Record};
use burn::tensor::{DType, Tensor, TensorData};
use hydra_core::encoder::NUM_CHANNELS;
#[cfg(feature = "libtorch")]
use hydra_train_runtime::config::{EffectivePrecision, PrecisionMode};
use hydra_train_runtime::head_gates::HeadActivationConfig;
use hydra_train_types::losses::HydraLossConfig;
#[cfg(feature = "bf16-autocast-proof")]
use std::panic::AssertUnwindSafe;
#[cfg(feature = "libtorch")]
use std::time::Instant;
#[cfg(any(feature = "bf16-autocast-proof", feature = "libtorch"))]
use tch::Cuda;
#[cfg(feature = "bf16-autocast-proof")]
use tch::{Device, Kind};

type TestTrainBackend = Autodiff<LibTorch<f32>>;

fn tiny_dummy_model(device: &LibTorchDevice) -> HydraModel<TestTrainBackend> {
    crate::model::HydraModelConfig::new(1)
        .with_input_channels(NUM_CHANNELS)
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
        compact_facts: None,
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

    let mut metric_sums: Option<BatchMetricSums<TestTrainBackend>> = None;
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
            model.forward_with_warmup_train(obs.clone(), &active_loss_fn.config, &warmup_heads);
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
            model.forward_with_warmup_train(obs.clone(), &active_loss_fn.config, &warmup_heads);
        let breakdown = active_loss_fn.total_loss(&output, &targets);
        let total = bc_total_with_exit_from_breakdown(&output, &batch, &breakdown, bc_exit_cfg);
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

    FixedShapeBenchmarkStepOutput {
        grads: accumulator.grads(),
        batch_stats: step_batches,
        sub_stage_timing: Default::default(),
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

#[derive(Default)]
struct FloatParamDtypeVisitor {
    saw_param: bool,
    all_fp32: bool,
}

impl ModuleVisitor<TestTrainBackend> for FloatParamDtypeVisitor {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<TestTrainBackend, D>>) {
        self.saw_param = true;
        let dtype = param.val().to_data().dtype;
        self.all_fp32 &= dtype == DType::F32;
    }
}

fn model_params_are_fp32(model: &HydraModel<TestTrainBackend>) -> bool {
    let mut visitor = FloatParamDtypeVisitor {
        saw_param: false,
        all_fp32: true,
    };
    model.visit(&mut visitor);
    visitor.saw_param && visitor.all_fp32
}

struct FiniteGradientVisitor<'a> {
    grads: &'a GradientsParams,
    saw_gradient: bool,
    nonfinite_gradients: usize,
}

impl ModuleVisitor<TestTrainBackend> for FiniteGradientVisitor<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<TestTrainBackend, D>>) {
        let Some(grad) = self.grads.get::<LibTorch<f32>, D>(param.id) else {
            return;
        };

        self.saw_gradient = true;
        if !grad.is_finite().all().into_scalar() {
            self.nonfinite_gradients += 1;
        }
    }
}

fn model_gradients_are_finite(
    model: &HydraModel<TestTrainBackend>,
    grads: &GradientsParams,
) -> bool {
    let mut visitor = FiniteGradientVisitor {
        grads,
        saw_gradient: false,
        nonfinite_gradients: 0,
    };
    model.visit(&mut visitor);
    visitor.saw_gradient && visitor.nonfinite_gradients == 0
}

#[cfg(feature = "libtorch")]
const PHASE4_RELATIVE_LOSS_TOLERANCE: f64 = 0.05;
#[cfg(feature = "libtorch")]
const PHASE4_MAX_ABS_LOGITS_TOLERANCE: f32 = 0.25;

#[cfg(feature = "libtorch")]
fn tensor2_to_vec(tensor: &Tensor<TestTrainBackend, 2>) -> Vec<f32> {
    tensor
        .to_data()
        .convert::<f32>()
        .as_slice::<f32>()
        .expect("tensor data should be readable as f32")
        .to_vec()
}

#[cfg(feature = "libtorch")]
fn max_abs_diff(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max)
}

#[cfg(feature = "libtorch")]
fn fixed_batch_forward_backward_probe(
    use_amp: bool,
) -> (
    f64,
    Vec<f32>,
    bool,
    bool,
    DType,
    HydraModel<TestTrainBackend>,
) {
    let device = LibTorchDevice::Cuda(0);
    <TestTrainBackend as Backend>::seed(&device, 7);
    let model = tiny_dummy_model(&device);
    let train_loss_fn = dummy_train_loss();
    let logical_batch = vec![
        dummy_train_sample(0),
        dummy_train_sample(5),
        dummy_train_sample(10),
    ];
    let (obs, targets) = collate_samples::<TestTrainBackend>(&logical_batch, false, &device)
        .expect("phase4 fixed batch collation should succeed")
        .expect("phase4 fixed batch collation should produce tensors");
    let output = hydra_model::amp::maybe_autocast(use_amp, || model.forward(obs));
    let logits_finite = output.is_finite();
    let logits = tensor2_to_vec(&output.policy_logits);
    let logits_dtype = output.policy_logits.dtype();
    let breakdown = train_loss_fn.total_loss(&output, &targets);
    let loss = breakdown.total.clone().into_scalar() as f64;
    let grads = GradientsParams::from_grads(breakdown.total.backward(), &model);
    let gradients_finite = model_gradients_are_finite(&model, &grads);
    (
        loss,
        logits,
        logits_finite,
        gradients_finite,
        logits_dtype,
        model,
    )
}

#[cfg(feature = "libtorch")]
fn tiny_phase4_replay() -> String {
    [
        r#"{"type":"start_game","names":["a","b","c","d"],"id":"phase4-game"}"#,
        r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["P","F","C","1m","1m","2m","2m","3m","3m","4m","4m","5m","5m"],["6p","6p","7p","7p","8p","8p","9p","9p","1s","1s","2s","2s","3s"]]}"#,
        r#"{"type":"dahai","actor":0,"pai":"4p","tsumogiri":false}"#,
        r#"{"type":"tsumo","actor":1,"pai":"P"}"#,
        r#"{"type":"dahai","actor":1,"pai":"P","tsumogiri":true}"#,
        r#"{"type":"tsumo","actor":2,"pai":"6m"}"#,
        r#"{"type":"dahai","actor":2,"pai":"6m","tsumogiri":true}"#,
        r#"{"type":"ryukyoku"}"#,
        r#"{"type":"end_kyoku"}"#,
    ]
    .join("\n")
}

#[cfg(feature = "libtorch")]
fn phase4_temp_dir(label: &str) -> std::path::PathBuf {
    let path = crate::test_support::unique_test_path("hydra-phase4", label);
    std::fs::create_dir_all(&path).expect("phase4 temp dir should be creatable");
    path
}

#[cfg(feature = "libtorch")]
fn build_phase4_shards(label: &str) -> std::path::PathBuf {
    let root = phase4_temp_dir(label);
    let input = root.join("input");
    let output = root.join("shards");
    std::fs::create_dir_all(&input).expect("phase4 input dir should be creatable");
    let replay = input.join("phase4.mjai");
    std::fs::write(&replay, tiny_phase4_replay()).expect("phase4 replay should be writable");
    let source_manifest = hydra_data_core::DataManifest {
        sources: vec![hydra_data_core::DataSource::LooseFile(replay)],
        total_games: 1,
        train_count: 1,
        val_count: 0,
        counts_exact: true,
    };
    let built =
        crate::bc_shard_builder::build_bc_shards(&crate::bc_shard_builder::BuildBcShardsConfig {
            input,
            output_dir: output,
            manifest_name: "manifest.json".to_string(),
            train_fraction: 1.0,
            shard_samples: 4,
            split_mode: hydra_bc_shards::BcShardSplitMode::Train,
            source_manifest: Some(source_manifest),
            num_threads: Some(1),
            queue_bound: 1,
            report_name: None,
            ..crate::bc_shard_builder::BuildBcShardsConfig::default()
        })
        .expect("phase4 compact shard build should pass");
    assert!(built.manifest.totals.sample_count > 0);
    built.manifest_path
}

#[cfg(feature = "libtorch")]
fn phase4_train_config(
    manifest_path: Option<std::path::PathBuf>,
) -> hydra_train_runtime::config::TrainConfig {
    let mut config = crate::test_support::dummy_train_config();
    config.device = "cuda:0".to_string();
    config.precision_mode = PrecisionMode::Bf16Autocast;
    config.batch_size = 2;
    config.microbatch_size = Some(2);
    config.validation_microbatch_size = Some(2);
    config.max_train_steps = Some(1);
    config.bc_shards_manifest_path = manifest_path;
    config.augment = false;
    config
}

#[cfg(feature = "libtorch")]
fn shard_train_step(manifest_path: &std::path::Path, use_amp: bool) -> (f64, bool, usize) {
    let device = LibTorchDevice::Cuda(0);
    let reader =
        hydra_bc_shards::load_bc_shard_reader(manifest_path, hydra_bc_shards::BcShardSplit::Train)
            .expect("phase4 train shard reader should load");
    let take = reader.sample_count().min(2);
    assert!(take > 0, "phase4 shard smoke requires at least one sample");
    let host_batch = reader
        .collate_host_batch_range(0, take, false)
        .expect("phase4 shard host batch should collate");
    let mut model_slot = Some(tiny_dummy_model(&device));
    let mut optimizer = AdamConfig::new().init();
    let train_loss_fn = dummy_train_loss();
    let mut head_controller =
        HeadActivationController::new(HeadActivationConfig::default_with_params(1));
    let (stats, _timing, _recycled) = crate::epoch_runner::train_logical_batch_from_host_batch(
        host_batch,
        crate::epoch_runner::TrainLogicalBatchConfig {
            microbatch_size: 2,
            use_amp,
            augment: false,
            train_device: &device,
            loss_fn: &train_loss_fn,
            bc_exit_cfg: &BcExitConfig::default(),
            lr: 1e-4,
        },
        &mut head_controller,
        &mut model_slot,
        &mut optimizer,
        #[cfg(feature = "cuda-graph")]
        None,
    )
    .expect("phase4 shard train step should pass");
    let model = model_slot
        .as_ref()
        .expect("phase4 shard model slot should remain populated");
    assert!(model_params_are_fp32(model));
    let stats = stats
        .first()
        .copied()
        .expect("phase4 shard train step should produce stats");
    (
        stats.total_loss,
        stats.total_loss.is_finite(),
        stats.sample_count,
    )
}

#[cfg(feature = "libtorch")]
fn synchronize_cuda() {
    tch::Cuda::synchronize(0);
}

#[cfg(feature = "bf16-autocast-proof")]
fn adam_state_is_fp32<const D: usize>(state: AdamState<LibTorch<f32>, D>) -> bool {
    let momentum = state.momentum;
    momentum.moment_1.dtype() == DType::F32
        && momentum.moment_2.dtype() == DType::F32
        && momentum
            .max_moment_2
            .as_ref()
            .is_none_or(|moment| moment.dtype() == DType::F32)
}

#[cfg(feature = "bf16-autocast-proof")]
fn adam_record_is_fp32(record: AdaptorRecord<Adam, TestTrainBackend>) -> bool {
    match record {
        AdaptorRecord::V1(record) => match record {
            AdaptorRecordV1::Rank0(state) => adam_state_is_fp32(state),
            AdaptorRecordV1::Rank1(state) => adam_state_is_fp32(state),
            AdaptorRecordV1::Rank2(state) => adam_state_is_fp32(state),
            AdaptorRecordV1::Rank3(state) => adam_state_is_fp32(state),
            AdaptorRecordV1::Rank4(state) => adam_state_is_fp32(state),
            AdaptorRecordV1::Rank5(state) => adam_state_is_fp32(state),
            AdaptorRecordV1::Rank6(state) => adam_state_is_fp32(state),
            AdaptorRecordV1::Rank7(state) => adam_state_is_fp32(state),
            AdaptorRecordV1::Rank8(state) => adam_state_is_fp32(state),
        },
    }
}

#[cfg(feature = "bf16-autocast-proof")]
fn adam_optimizer_record_is_fp32(
    record: &<burn::optim::adaptor::OptimizerAdaptor<
        Adam,
        HydraModel<TestTrainBackend>,
        TestTrainBackend,
    > as Optimizer<HydraModel<TestTrainBackend>, TestTrainBackend>>::Record,
) -> bool {
    !record.is_empty() && record.values().cloned().all(adam_record_is_fp32)
}

#[cfg(feature = "bf16-autocast-proof")]
fn assert_cuda_autocast_state_restored<F>(f: F)
where
    F: FnOnce(),
{
    let before = current_cuda_autocast_state().expect("CUDA autocast state should be readable");
    let result = std::panic::catch_unwind(AssertUnwindSafe(f));
    let after = current_cuda_autocast_state().expect("CUDA autocast state should be readable");
    assert_eq!(after, before, "CUDA autocast state was not restored");
    if let Err(payload) = result {
        std::panic::resume_unwind(payload);
    }
}

#[cfg(feature = "bf16-autocast-proof")]
#[test]
#[ignore = "manual CUDA BF16 AMP proof; run explicitly with --features bf16-autocast-proof"]
fn cuda_bf16_autocast_state_restores_after_return_panic_and_nested_scope() {
    if !Cuda::is_available() {
        panic!("CUDA is required for autocast restoration proof");
    }

    assert_cuda_autocast_state_restored(AssertUnwindSafe(|| {
        with_cuda_bf16_autocast_dtype_proof_only(|| {
            let active = current_cuda_autocast_state().expect("state readable in BF16 scope");
            assert_eq!(active.enabled, 1);
            assert_eq!(active.cache_enabled, 1);
        })
        .expect("BF16 scope should enter");
    }));

    let before_panic = current_cuda_autocast_state().expect("state readable before panic proof");
    let panic_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = with_cuda_bf16_autocast_dtype_proof_only(|| {
            panic!("intentional autocast restoration proof panic");
        });
    }));
    assert!(panic_result.is_err());
    let after_panic = current_cuda_autocast_state().expect("state readable after panic proof");
    assert_eq!(after_panic, before_panic);

    let outer = before_panic;
    restore_cuda_autocast_state(&outer).expect("outer state restore should work");
    with_cuda_bf16_autocast_dtype_proof_only(|| {
        let inner = current_cuda_autocast_state().expect("inner state readable");
        assert_eq!(inner.enabled, 1);
        assert_eq!(inner.cache_enabled, 1);
    })
    .expect("nested BF16 scope should enter");
    let after_nested = current_cuda_autocast_state().expect("state readable after nested proof");
    assert_eq!(after_nested, outer);
}

#[cfg(feature = "bf16-autocast-proof")]
#[test]
#[ignore = "manual CUDA BF16 AMP proof; run explicitly with --features bf16-autocast-proof"]
fn hydra_libtorch_cuda_forced_bf16_autocast_proof_probe() {
    if !Cuda::is_available() {
        println!("requested_precision=bf16_autocast");
        println!("effective_precision=unproven");
        println!("backend=Autodiff<LibTorch<f32>>");
        println!("device=unavailable_cuda");
        println!("forward_dtype_or_kernel_evidence=unproven_no_cuda");
        println!("loss_dtype=unproven_no_cuda");
        println!("params_fp32=not_observed_no_cuda");
        println!("gradients_finite=not_reached_no_cuda");
        println!("optimizer_state_fp32=not_reached_no_cuda");
        panic!("CUDA is required for BF16 autocast proof");
    }

    let tiny_output_kind = with_cuda_bf16_autocast_dtype_proof_only(|| {
        let lhs = tch::Tensor::ones([8, 8], (Kind::Float, Device::Cuda(0)));
        let rhs = tch::Tensor::ones([8, 8], (Kind::Float, Device::Cuda(0)));
        lhs.matmul(&rhs).kind()
    })
    .expect("BF16 autocast dtype shim should set CUDA autocast dtype");

    println!("requested_precision=bf16_autocast");
    println!("effective_precision=unproven");
    println!("backend=Autodiff<LibTorch<f32>>");
    println!("libtorch_version=2.9.0 (tch 0.22 / torch-sys 0.22 build contract)");
    println!("device=Cuda(0)");
    println!("forward_dtype_or_kernel_evidence=tiny_matmul_output_dtype={tiny_output_kind:?}");

    if tiny_output_kind != Kind::BFloat16 {
        println!("loss_dtype=unproven_tiny_gate_failed");
        println!("params_fp32=not_observed_tiny_gate_failed");
        println!("gradients_finite=not_reached_tiny_gate_failed");
        println!("optimizer_state_fp32=not_reached_tiny_gate_failed");
        panic!("forward_dtype_or_kernel_evidence is not bf16");
    }

    let device = LibTorchDevice::Cuda(0);
    let model = tiny_dummy_model(&device);
    assert!(
        model_params_are_fp32(&model),
        "params_fp32=false before AMP proof"
    );
    let train_loss_fn = dummy_train_loss();
    let logical_batch = vec![dummy_train_sample(0), dummy_train_sample(5)];
    let (obs, targets) = collate_samples::<TestTrainBackend>(&logical_batch, false, &device)
        .expect("Hydra proof collation should succeed")
        .expect("Hydra proof collation should produce tensors");

    let output = with_cuda_bf16_autocast_dtype_proof_only(AssertUnwindSafe(|| model.forward(obs)))
        .expect("BF16 autocast dtype shim should set CUDA autocast dtype for Hydra forward");
    let hydra_forward_dtype = output.policy_logits.dtype();
    let breakdown = train_loss_fn.total_loss(&output, &targets);
    let loss_dtype = breakdown.total.dtype();

    println!("forward_dtype_or_kernel_evidence=hydra_policy_logits_dtype={hydra_forward_dtype:?}");
    println!("loss_dtype={loss_dtype:?}");
    println!("params_fp32={}", model_params_are_fp32(&model));

    if hydra_forward_dtype != DType::BF16 {
        println!("gradients_finite=not_reached_forward_dtype_failed");
        println!("optimizer_state_fp32=not_reached_forward_dtype_failed");
        panic!("Hydra forward did not produce BF16 output evidence");
    }
    if loss_dtype != DType::F32 {
        println!("gradients_finite=not_reached_loss_dtype_failed");
        println!("optimizer_state_fp32=not_reached_loss_dtype_failed");
        panic!("loss_dtype is not fp32");
    }

    let backward_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        GradientsParams::from_grads(breakdown.total.backward(), &model)
    }));
    let grads = match backward_result {
        Ok(grads) => grads,
        Err(payload) => {
            println!("gradients_finite=not_reached_backward_dtype_mismatch");
            println!("optimizer_state_fp32=not_reached_backward_dtype_mismatch");
            std::panic::resume_unwind(payload);
        }
    };
    let gradients_finite = model_gradients_are_finite(&model, &grads);
    println!("gradients_finite={gradients_finite}");
    assert!(
        gradients_finite,
        "gradients_finite=false: no gradients produced or at least one gradient is non-finite"
    );

    let mut optimizer = AdamConfig::new().init();
    let stepped_model = optimizer.step(1e-4, model, grads);
    let params_fp32 = model_params_are_fp32(&stepped_model);
    println!("params_fp32={params_fp32}");
    assert!(params_fp32, "params_fp32=false after Adam step");

    let optimizer_record = optimizer.to_record();
    let optimizer_state_fp32 = adam_optimizer_record_is_fp32(&optimizer_record);
    assert!(
        optimizer_state_fp32,
        "optimizer_state_fp32=false: Adam moment state missing or not fp32"
    );

    let proof_dir =
        std::env::temp_dir().join(format!("hydra-bf16-adam-proof-{}", std::process::id()));
    std::fs::create_dir_all(&proof_dir)
        .expect("optimizer_state_proof=missing: temp proof dir should be creatable");
    let optimizer_base = proof_dir.join("optimizer");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    recorder
        .record(optimizer_record.clone(), optimizer_base.clone())
        .expect("optimizer_state_proof=missing: optimizer record should save");
    let reloaded_record = recorder
        .load(optimizer_base, &device)
        .expect("optimizer_state_proof=missing: optimizer record should load");
    let reloaded_optimizer = AdamConfig::new()
        .init::<TestTrainBackend, HydraModel<TestTrainBackend>>()
        .load_record(reloaded_record);
    let reloaded_record = reloaded_optimizer.to_record();
    let reloaded_optimizer_state_fp32 = adam_optimizer_record_is_fp32(&reloaded_record);
    let _ = std::fs::remove_dir_all(&proof_dir);

    println!("optimizer_state_fp32={reloaded_optimizer_state_fp32}");
    println!("optimizer_state_proof=direct_adam_state_dtype_and_checkpoint_roundtrip");
    assert!(
        reloaded_optimizer_state_fp32,
        "optimizer_state_fp32=false after checkpoint roundtrip"
    );
}

#[cfg(feature = "libtorch")]
#[test]
#[ignore = "manual CUDA BF16 AMP BC path proof; run explicitly with --features libtorch"]
fn bc_fixed_shape_cuda_bf16_amp_train_step_runs_real_amp() {
    if !Cuda::is_available() {
        println!("bc_amp_train_step=not_reached_no_cuda");
        panic!("CUDA is required for BC BF16 AMP train-step proof");
    }

    let device = LibTorchDevice::Cuda(0);
    let model = tiny_dummy_model(&device);
    let train_loss_fn = dummy_train_loss();
    let logical_batch = vec![
        dummy_train_sample(0),
        dummy_train_sample(5),
        dummy_train_sample(10),
    ];
    let mut head_controller =
        HeadActivationController::new(HeadActivationConfig::default_with_params(1));

    let output = run_train_logical_batch_fixed_chunks(FixedShapeTrainConfig {
        logical_batch: &logical_batch,
        augment: false,
        microbatch_size: 2,
        train_device: &device,
        loss_fn: &train_loss_fn,
        bc_exit_cfg: &BcExitConfig::default(),
        head_controller: &mut head_controller,
        model: &model,
        use_amp: true,
    })
    .expect("BC BF16 AMP train step should not error")
    .expect("BC BF16 AMP train step should produce gradients and stats");

    let gradients_finite = model_gradients_are_finite(&model, &output.grads);
    println!("bc_amp_train_step=ok");
    println!("gradients_finite={gradients_finite}");
    assert!(
        gradients_finite,
        "BC BF16 AMP train step gradients must be finite"
    );
    assert_eq!(output.batch_stats.sample_count, logical_batch.len());
}

#[test]
#[cfg(feature = "libtorch")]
#[ignore = "manual CUDA Phase 4 parity gate; run explicitly with --features libtorch"]
fn phase4_fixed_batch_fp32_vs_bf16_amp_parity_gate() {
    if !Cuda::is_available() {
        panic!("CUDA is required for Phase 4 parity gate");
    }
    let (
        fp32_loss,
        fp32_logits,
        fp32_logits_finite,
        fp32_gradients_finite,
        fp32_logits_dtype,
        fp32_model,
    ) = fixed_batch_forward_backward_probe(false);
    let (
        bf16_loss,
        bf16_logits,
        bf16_logits_finite,
        bf16_gradients_finite,
        bf16_logits_dtype,
        bf16_model,
    ) = fixed_batch_forward_backward_probe(true);
    let relative_loss_drift = (bf16_loss - fp32_loss).abs() / fp32_loss.abs().max(f64::EPSILON);
    let max_abs_logits_drift = max_abs_diff(&fp32_logits, &bf16_logits);
    println!("phase4_parity_fp32_loss={fp32_loss:.8}");
    println!("phase4_parity_bf16_loss={bf16_loss:.8}");
    println!("phase4_parity_relative_loss_drift={relative_loss_drift:.8}");
    println!("phase4_parity_max_abs_logits_drift={max_abs_logits_drift:.8}");
    println!("phase4_parity_fp32_logits_dtype={fp32_logits_dtype:?}");
    println!("phase4_parity_bf16_logits_dtype={bf16_logits_dtype:?}");
    assert!(fp32_loss.is_finite() && bf16_loss.is_finite());
    assert!(fp32_logits_finite && bf16_logits_finite);
    assert!(fp32_gradients_finite && bf16_gradients_finite);
    assert!(model_params_are_fp32(&fp32_model));
    assert!(model_params_are_fp32(&bf16_model));
    assert!(relative_loss_drift <= PHASE4_RELATIVE_LOSS_TOLERANCE);
    assert!(max_abs_logits_drift <= PHASE4_MAX_ABS_LOGITS_TOLERANCE);
}

#[test]
#[cfg(feature = "libtorch")]
#[ignore = "manual CUDA Phase 4 shard smoke; run explicitly with --features libtorch"]
fn phase4_tiny_compact_shard_bf16_smoke_gate() {
    if !Cuda::is_available() {
        panic!("CUDA is required for Phase 4 shard smoke");
    }
    let manifest_path = build_phase4_shards("shard-smoke");
    let config = phase4_train_config(Some(manifest_path.clone()));
    assert_eq!(config.effective_precision(), EffectivePrecision::Bf16Amp);
    let (loss, finite, sample_count) = shard_train_step(&manifest_path, config.use_amp());
    println!("phase4_shard_manifest={}", manifest_path.display());
    println!("effective_precision={}", config.effective_precision());
    println!("phase4_shard_loss={loss:.8}");
    println!("phase4_shard_samples={sample_count}");
    assert!(finite);
    assert!(sample_count > 0);
}

#[test]
#[cfg(feature = "libtorch")]
#[ignore = "manual CUDA Phase 4 validation policy gate; run explicitly with --features libtorch"]
fn phase4_validation_policy_train_bf16_validation_fp32_gate() {
    if !Cuda::is_available() {
        panic!("CUDA is required for Phase 4 validation policy gate");
    }
    let device = LibTorchDevice::Cuda(0);
    let model = tiny_dummy_model(&device);
    let train_loss_fn = dummy_train_loss();
    let logical_batch = vec![dummy_train_sample(0), dummy_train_sample(5)];
    let (obs, targets) = collate_samples::<TestTrainBackend>(&logical_batch, false, &device)
        .expect("phase4 validation gate collation should succeed")
        .expect("phase4 validation gate collation should produce tensors");
    let train_output = hydra_model::amp::maybe_autocast(true, || model.forward(obs));
    let train_dtype = train_output.policy_logits.dtype();
    let breakdown = train_loss_fn.total_loss(&train_output, &targets);
    let grads = GradientsParams::from_grads(breakdown.total.backward(), &model);
    assert!(model_gradients_are_finite(&model, &grads));

    let obs = Tensor::<LibTorch<f32>, 3>::from_data(
        TensorData::new(
            vec![0.1f32; logical_batch.len() * hydra_core::encoder::OBS_SIZE],
            [logical_batch.len(), NUM_CHANNELS, 34],
        ),
        &device,
    );
    let validation_output = model.valid().forward(obs);
    let validation_dtype = validation_output.policy_logits.dtype();
    println!("phase4_validation_train_forward_dtype={train_dtype:?}");
    println!("phase4_validation_forward_dtype={validation_dtype:?}");
    println!(
        "phase4_validation_params_fp32={}",
        model_params_are_fp32(&model)
    );
    assert_eq!(train_dtype, DType::BF16);
    assert_eq!(validation_dtype, DType::F32);
    assert!(validation_output.is_finite());
    assert!(model_params_are_fp32(&model));
}

#[test]
#[cfg(feature = "libtorch")]
#[ignore = "manual CUDA Phase 4 throughput/memory smoke; run explicitly with --features libtorch"]
fn phase4_throughput_memory_smoke_gate() {
    if !Cuda::is_available() {
        panic!("CUDA is required for Phase 4 throughput/memory smoke");
    }
    let manifest_path = build_phase4_shards("throughput");
    let measure = |use_amp: bool| -> (f64, f64, f64, usize) {
        synchronize_cuda();
        let started = Instant::now();
        let mut samples = 0usize;
        let mut last_loss = 0.0f64;
        for _ in 0..3 {
            let (loss, finite, sample_count) = shard_train_step(&manifest_path, use_amp);
            assert!(finite);
            last_loss = loss;
            samples += sample_count;
        }
        synchronize_cuda();
        let wall = started.elapsed().as_secs_f64();
        (
            samples as f64 / wall.max(f64::EPSILON),
            wall,
            last_loss,
            samples,
        )
    };
    let (fp32_sps, fp32_wall, fp32_loss, fp32_samples) = measure(false);
    let (bf16_sps, bf16_wall, bf16_loss, bf16_samples) = measure(true);
    let speed_ratio = bf16_sps / fp32_sps.max(f64::EPSILON);
    println!("phase4_perf_manifest={}", manifest_path.display());
    println!("phase4_perf_fp32_samples_per_sec={fp32_sps:.2}");
    println!("phase4_perf_bf16_samples_per_sec={bf16_sps:.2}");
    println!("phase4_perf_fp32_wall_sec={fp32_wall:.6}");
    println!("phase4_perf_bf16_wall_sec={bf16_wall:.6}");
    println!("phase4_perf_fp32_last_loss={fp32_loss:.8}");
    println!("phase4_perf_bf16_last_loss={bf16_loss:.8}");
    println!("phase4_perf_fp32_samples={fp32_samples}");
    println!("phase4_perf_bf16_samples={bf16_samples}");
    println!("phase4_perf_peak_cuda_memory=unavailable_tch_api_not_exposed");
    println!("phase4_perf_speed_ratio={speed_ratio:.6}");
    assert!(fp32_loss.is_finite() && bf16_loss.is_finite());
    assert!(
        speed_ratio >= 0.90,
        "BF16 AMP throughput slower than FP32 by more than 10%"
    );
}

#[cfg(feature = "bf16-autocast-proof")]
fn direct_tch_grad_is_finite(grad: &tch::Tensor) -> bool {
    grad.defined() && grad.isfinite().all().int64_value(&[]) != 0
}

#[cfg(feature = "bf16-autocast-proof")]
fn print_direct_tch_probe_result(
    op: &str,
    forward_dtype: Option<Kind>,
    loss_dtype: Option<Kind>,
    backward_ok: bool,
    backward_failure: Option<&str>,
    grads: &[(&str, Option<Kind>, Option<bool>)],
) {
    println!("op={op}");
    match forward_dtype {
        Some(dtype) => println!("forward_dtype={dtype:?}"),
        None => println!("forward_dtype=not_reached"),
    }
    match loss_dtype {
        Some(dtype) => println!("loss_dtype={dtype:?}"),
        None => println!("loss_dtype=not_reached"),
    }
    if backward_ok {
        println!("backward=ok");
    } else if let Some(failure) = backward_failure {
        println!("backward=fail; failure={failure}");
    } else {
        println!("backward=fail; failure=unknown_non_string_panic");
    }
    for (name, dtype, finite) in grads {
        match (dtype, finite) {
            (Some(dtype), Some(finite)) => {
                println!("grad.{name}.dtype={dtype:?}; grad.{name}.finite={finite}");
            }
            _ => println!("grad.{name}.dtype=not_reached; grad.{name}.finite=not_reached"),
        }
    }
}

#[cfg(feature = "bf16-autocast-proof")]
fn panic_message(payload: &(dyn std::any::Any + Send)) -> Option<&str> {
    payload
        .downcast_ref::<&'static str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
}

#[cfg(feature = "bf16-autocast-proof")]
fn assert_direct_tch_probe_succeeded(
    op: &str,
    forward_dtype: Kind,
    loss_dtype: Kind,
    grads: &[(&str, Kind, bool)],
) {
    if forward_dtype != Kind::BFloat16 {
        panic!("direct tch {op} forward dtype was {forward_dtype:?}, expected BFloat16");
    }
    if loss_dtype != Kind::Float {
        panic!("direct tch {op} loss dtype was {loss_dtype:?}, expected Float");
    }
    for (name, dtype, finite) in grads {
        if *dtype != Kind::Float {
            panic!("direct tch {op} grad {name} dtype was {dtype:?}, expected Float");
        }
        if !finite {
            panic!("direct tch {op} grad {name} was not finite");
        }
    }
}

#[cfg(feature = "bf16-autocast-proof")]
fn run_direct_tch_matmul_bf16_autocast_probe(device: Device) {
    let op = "matmul";
    let lhs = tch::Tensor::randn([16, 32], (Kind::Float, device)).set_requires_grad(true);
    let rhs = tch::Tensor::randn([32, 8], (Kind::Float, device)).set_requires_grad(true);

    let forward = with_cuda_bf16_autocast_dtype_proof_only(AssertUnwindSafe(|| lhs.matmul(&rhs)))
        .expect("BF16 autocast dtype shim should set CUDA autocast dtype for direct tch matmul");
    let forward_dtype = forward.kind();
    let loss = forward.to_kind(Kind::Float).square().mean(Kind::Float);
    let loss_dtype = loss.kind();

    let backward_result = std::panic::catch_unwind(AssertUnwindSafe(|| loss.backward()));
    let backward_failure = backward_result
        .as_ref()
        .err()
        .and_then(|payload| panic_message(payload.as_ref()));
    let backward_ok = backward_result.is_ok();
    let lhs_grad = backward_ok.then(|| lhs.grad());
    let rhs_grad = backward_ok.then(|| rhs.grad());
    let lhs_grad_dtype = lhs_grad.as_ref().map(tch::Tensor::kind);
    let rhs_grad_dtype = rhs_grad.as_ref().map(tch::Tensor::kind);
    let lhs_grad_finite = lhs_grad.as_ref().map(direct_tch_grad_is_finite);
    let rhs_grad_finite = rhs_grad.as_ref().map(direct_tch_grad_is_finite);

    print_direct_tch_probe_result(
        op,
        Some(forward_dtype),
        Some(loss_dtype),
        backward_ok,
        backward_failure,
        &[
            ("lhs", lhs_grad_dtype, lhs_grad_finite),
            ("rhs", rhs_grad_dtype, rhs_grad_finite),
        ],
    );

    if let Err(payload) = backward_result {
        std::panic::resume_unwind(payload);
    }
    assert_direct_tch_probe_succeeded(
        op,
        forward_dtype,
        loss_dtype,
        &[
            ("lhs", lhs_grad_dtype.unwrap(), lhs_grad_finite.unwrap()),
            ("rhs", rhs_grad_dtype.unwrap(), rhs_grad_finite.unwrap()),
        ],
    );
}

#[cfg(feature = "bf16-autocast-proof")]
fn run_direct_tch_linear_bf16_autocast_probe(device: Device) {
    let op = "linear";
    let input = tch::Tensor::randn([16, 32], (Kind::Float, device)).set_requires_grad(true);
    let weight = tch::Tensor::randn([8, 32], (Kind::Float, device)).set_requires_grad(true);
    let bias = tch::Tensor::randn([8], (Kind::Float, device)).set_requires_grad(true);

    let forward = with_cuda_bf16_autocast_dtype_proof_only(AssertUnwindSafe(|| {
        input.linear(&weight, Some(&bias))
    }))
    .expect("BF16 autocast dtype shim should set CUDA autocast dtype for direct tch linear");
    let forward_dtype = forward.kind();
    let loss = forward.to_kind(Kind::Float).square().mean(Kind::Float);
    let loss_dtype = loss.kind();

    let backward_result = std::panic::catch_unwind(AssertUnwindSafe(|| loss.backward()));
    let backward_failure = backward_result
        .as_ref()
        .err()
        .and_then(|payload| panic_message(payload.as_ref()));
    let backward_ok = backward_result.is_ok();
    let input_grad = backward_ok.then(|| input.grad());
    let weight_grad = backward_ok.then(|| weight.grad());
    let bias_grad = backward_ok.then(|| bias.grad());
    let input_grad_dtype = input_grad.as_ref().map(tch::Tensor::kind);
    let weight_grad_dtype = weight_grad.as_ref().map(tch::Tensor::kind);
    let bias_grad_dtype = bias_grad.as_ref().map(tch::Tensor::kind);
    let input_grad_finite = input_grad.as_ref().map(direct_tch_grad_is_finite);
    let weight_grad_finite = weight_grad.as_ref().map(direct_tch_grad_is_finite);
    let bias_grad_finite = bias_grad.as_ref().map(direct_tch_grad_is_finite);

    print_direct_tch_probe_result(
        op,
        Some(forward_dtype),
        Some(loss_dtype),
        backward_ok,
        backward_failure,
        &[
            ("input", input_grad_dtype, input_grad_finite),
            ("weight", weight_grad_dtype, weight_grad_finite),
            ("bias", bias_grad_dtype, bias_grad_finite),
        ],
    );

    if let Err(payload) = backward_result {
        std::panic::resume_unwind(payload);
    }
    assert_direct_tch_probe_succeeded(
        op,
        forward_dtype,
        loss_dtype,
        &[
            (
                "input",
                input_grad_dtype.unwrap(),
                input_grad_finite.unwrap(),
            ),
            (
                "weight",
                weight_grad_dtype.unwrap(),
                weight_grad_finite.unwrap(),
            ),
            ("bias", bias_grad_dtype.unwrap(), bias_grad_finite.unwrap()),
        ],
    );
}

#[cfg(feature = "bf16-autocast-proof")]
fn run_direct_tch_conv1d_bf16_autocast_probe(device: Device) {
    let op = "conv1d";
    let input = tch::Tensor::randn([4, 3, 17], (Kind::Float, device)).set_requires_grad(true);
    let weight = tch::Tensor::randn([5, 3, 3], (Kind::Float, device)).set_requires_grad(true);
    let bias = tch::Tensor::randn([5], (Kind::Float, device)).set_requires_grad(true);

    let forward = with_cuda_bf16_autocast_dtype_proof_only(AssertUnwindSafe(|| {
        input.conv1d(&weight, Some(&bias), [1], [1], [1], 1)
    }))
    .expect("BF16 autocast dtype shim should set CUDA autocast dtype for direct tch conv1d");
    let forward_dtype = forward.kind();
    let loss = forward.to_kind(Kind::Float).square().mean(Kind::Float);
    let loss_dtype = loss.kind();

    let backward_result = std::panic::catch_unwind(AssertUnwindSafe(|| loss.backward()));
    let backward_failure = backward_result
        .as_ref()
        .err()
        .and_then(|payload| panic_message(payload.as_ref()));
    let backward_ok = backward_result.is_ok();
    let input_grad = backward_ok.then(|| input.grad());
    let weight_grad = backward_ok.then(|| weight.grad());
    let bias_grad = backward_ok.then(|| bias.grad());
    let input_grad_dtype = input_grad.as_ref().map(tch::Tensor::kind);
    let weight_grad_dtype = weight_grad.as_ref().map(tch::Tensor::kind);
    let bias_grad_dtype = bias_grad.as_ref().map(tch::Tensor::kind);
    let input_grad_finite = input_grad.as_ref().map(direct_tch_grad_is_finite);
    let weight_grad_finite = weight_grad.as_ref().map(direct_tch_grad_is_finite);
    let bias_grad_finite = bias_grad.as_ref().map(direct_tch_grad_is_finite);

    print_direct_tch_probe_result(
        op,
        Some(forward_dtype),
        Some(loss_dtype),
        backward_ok,
        backward_failure,
        &[
            ("input", input_grad_dtype, input_grad_finite),
            ("weight", weight_grad_dtype, weight_grad_finite),
            ("bias", bias_grad_dtype, bias_grad_finite),
        ],
    );

    if let Err(payload) = backward_result {
        std::panic::resume_unwind(payload);
    }
    assert_direct_tch_probe_succeeded(
        op,
        forward_dtype,
        loss_dtype,
        &[
            (
                "input",
                input_grad_dtype.unwrap(),
                input_grad_finite.unwrap(),
            ),
            (
                "weight",
                weight_grad_dtype.unwrap(),
                weight_grad_finite.unwrap(),
            ),
            ("bias", bias_grad_dtype.unwrap(), bias_grad_finite.unwrap()),
        ],
    );
}

#[cfg(feature = "bf16-autocast-proof")]
#[test]
#[ignore = "manual direct tch CUDA BF16 AMP proof; run explicitly with --features bf16-autocast-proof"]
fn direct_tch_cuda_forced_bf16_autocast_backward_proof_probe() {
    if !Cuda::is_available() {
        println!("op=direct_tch_suite");
        println!("device=unavailable_cuda");
        println!("backward=not_reached_no_cuda");
        panic!("CUDA is required for direct tch BF16 autocast proof");
    }

    let device = Device::Cuda(0);
    println!("device=Cuda(0)");
    println!("requested_precision=bf16_autocast");
    run_direct_tch_matmul_bf16_autocast_probe(device);
    run_direct_tch_linear_bf16_autocast_probe(device);
    run_direct_tch_conv1d_bf16_autocast_probe(device);
}

#[test]
fn optimizer_step_probe_keeps_gradients_params_and_checkpoint_state_fp32() {
    let device = LibTorchDevice::Cpu;
    let model = tiny_dummy_model(&device);
    let train_loss_fn = dummy_train_loss();
    let logical_batch = vec![dummy_train_sample(0), dummy_train_sample(5)];
    let grads = probe_train_fixed_chunks(FixedShapeProbeConfig {
        logical_batch: &logical_batch,
        augment: false,
        microbatch_size: logical_batch.len(),
        train_device: &device,
        loss_fn: &train_loss_fn,
        model: &model,
        use_amp: false,
    })
    .expect("optimizer proof probe should not error")
    .expect("optimizer proof probe should produce gradients");

    assert!(
        !grads.is_empty(),
        "gradients_finite=false: no gradients produced"
    );
    let mut optimizer = AdamConfig::new().init();
    let stepped_model = optimizer.step(1e-4, model, grads);

    assert!(
        model_params_are_fp32(&stepped_model),
        "params_fp32=false after Adam step"
    );
    let optimizer_record = optimizer.to_record();
    assert!(
        !optimizer_record.is_empty(),
        "optimizer_state_fp32=false: Adam produced no state records"
    );

    let full_precision_item = optimizer_record
        .clone()
        .into_item::<FullPrecisionSettings>();
    let reloaded_record = <_ as Record<TestTrainBackend>>::from_item(full_precision_item, &device);
    let reloaded_optimizer = AdamConfig::new()
        .init::<TestTrainBackend, HydraModel<TestTrainBackend>>()
        .load_record(reloaded_record);
    assert_eq!(
        reloaded_optimizer.to_record().len(),
        optimizer_record.len(),
        "optimizer_state_fp32=false: FullPrecisionSettings optimizer state roundtrip lost records"
    );
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
