use criterion::{Criterion, black_box, criterion_group, criterion_main};
use std::time::Duration;

use burn::backend::libtorch::LibTorchDevice;
use burn::backend::{Autodiff, LibTorch};
use burn::module::AutodiffModule;
use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::arena::{Trajectory, TrajectoryStep};
use hydra_core::encoder::OBS_SIZE;
use hydra_train::data::bc_shards::{
    BcShardSplit, BcShardSplitMode, BuildBcShardsConfig, build_bc_shards, load_bc_shard_reader,
};
use hydra_train::data::mjai_loader::{
    ReplayTargetProfile, SidecarProvenance, load_game_from_reader,
    load_game_from_reader_with_sidecar,
};
use hydra_train::data::pipeline::{
    SourceFilterConfig, StreamingLoaderConfig, scan_data_sources_with_progress,
    stream_val_microbatches, stream_val_pass,
};
use hydra_train::data::sample::{MjaiSample, collate_samples_bc_owned, collate_samples_owned};
use hydra_train::model::{HydraModelConfig, HydraModelInit, HydraTrainModelExt};
use hydra_train::selfplay::{
    CooperativeSelfPlayCoordinator, generate_self_play_batch_source,
    generate_self_play_batch_source_cooperative, generate_self_play_batch_source_cooperative_reuse,
};
use hydra_train::selfplay_batch::{
    RlBatchScratch, default_gae_config, trajectories_to_rl_batch, trajectories_to_rl_batch_reuse,
};
use hydra_train::training::bc::BcExitConfig;
use hydra_train::training::bc::bc_total_with_optional_exit_from_breakdown;
use hydra_train::training::live_exit::LiveExitConfig;
use hydra_train::training::losses::LossBreakdown;
use hydra_train::training::losses::{HydraLoss, HydraLossConfig};

type TrainBackend = Autodiff<LibTorch<f32>>;
type ValidBackend = <TrainBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend;

fn bench_metric_sums_from_outputs<B: burn::tensor::backend::Backend>(
    sample_count: usize,
    policy_logits: burn::tensor::Tensor<B, 2>,
    legal_mask: burn::tensor::Tensor<B, 2>,
    actions: burn::tensor::Tensor<B, 1, burn::tensor::Int>,
    total_loss: burn::tensor::Tensor<B, 1>,
    breakdown: &LossBreakdown<B>,
) -> burn::tensor::Tensor<B, 1> {
    let masked = policy_logits + (legal_mask.ones_like() - legal_mask) * (-1e9f32);
    let predicted = masked.argmax(1).squeeze_dim::<1>(1).float();
    let actions = actions.float();
    let sample_weight = sample_count as f32;

    burn::tensor::Tensor::cat(
        vec![
            predicted.equal(actions).float().sum(),
            total_loss * sample_weight,
            breakdown.policy.clone() * sample_weight,
            breakdown.value.clone() * sample_weight,
            breakdown.grp.clone() * sample_weight,
            breakdown.tenpai.clone() * sample_weight,
            breakdown.danger.clone() * sample_weight,
            breakdown.opp_next.clone() * sample_weight,
            breakdown.score_pdf.clone() * sample_weight,
            breakdown.score_cdf.clone() * sample_weight,
        ],
        0,
    )
}

fn tiny_real_mjai_replay() -> String {
    [
        r#"{"type":"start_game","names":["a","b","c","d"],"id":"bench-game"}"#,
        r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["P","F","C","1m","1m","2m","2m","3m","3m","4m","4m","5m","5m"],["6p","6p","7p","7p","8p","8p","9p","9p","1s","1s","2s","2s","3s"]]}"#,
        r#"{"type":"dahai","actor":0,"pai":"4p","tsumogiri":false}"#,
        r#"{"type":"tsumo","actor":1,"pai":"P"}"#,
        r#"{"type":"dahai","actor":1,"pai":"P","tsumogiri":true}"#,
        r#"{"type":"ryukyoku"}"#,
        r#"{"type":"end_kyoku"}"#,
    ]
    .join("\n")
}

fn bench_loader(c: &mut Criterion) {
    let payload = tiny_real_mjai_replay();
    let mut group = c.benchmark_group("loader");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(6));
    group.bench_function("load_game_from_reader_minimal_bc", |b| {
        b.iter(|| {
            let cursor = std::io::Cursor::new(payload.as_bytes());
            let game = load_game_from_reader(cursor).expect("loader bench replay should parse");
            black_box(game.samples.len())
        });
    });
    group.bench_function("load_game_from_reader_oracle_profile", |b| {
        b.iter(|| {
            let cursor = std::io::Cursor::new(payload.as_bytes());
            let game = load_game_from_reader_with_sidecar(
                "bench-game",
                SidecarProvenance::default(),
                SidecarProvenance::default(),
                ReplayTargetProfile::with_optional_heads(true, false, false, false, false, false),
                cursor,
                None,
                None,
            )
            .expect("loader bench replay should parse with oracle profile");
            black_box(game.samples.len())
        });
    });
    group.finish();
}

fn bench_validation_batch_stats(c: &mut Criterion) {
    let device = LibTorchDevice::Cpu;
    let model = HydraModelConfig::actor().init::<TrainBackend>(&device);
    let valid = model.valid();
    let loss_fn = HydraLoss::<ValidBackend>::new(HydraLossConfig::new());
    let samples = vec![
        MjaiSample {
            obs: [0.0; OBS_SIZE],
            action: 0,
            legal_mask: [1.0; HYDRA_ACTION_SPACE],
            placement: 0,
            score_delta: 0,
            grp_label: 0,
            oracle_target: None,
            tenpai: [0.0; 3],
            opp_next: [255; 3],
            danger: [0.0; 102],
            danger_mask: [0.0; 102],
            safety_residual: None,
            safety_residual_mask: None,
            exit_target: None,
            exit_mask: None,
            delta_q_target: Some([0.0; HYDRA_ACTION_SPACE]),
            delta_q_mask: Some([1.0; HYDRA_ACTION_SPACE]),
            belief_fields: None,
            mixture_weights: None,
            belief_fields_present: false,
            mixture_weights_present: false,
        },
        MjaiSample {
            obs: [0.0; OBS_SIZE],
            action: 1,
            legal_mask: [1.0; HYDRA_ACTION_SPACE],
            placement: 1,
            score_delta: 0,
            grp_label: 1,
            oracle_target: None,
            tenpai: [0.0; 3],
            opp_next: [255; 3],
            danger: [0.0; 102],
            danger_mask: [0.0; 102],
            safety_residual: None,
            safety_residual_mask: None,
            exit_target: None,
            exit_mask: None,
            delta_q_target: Some([0.0; HYDRA_ACTION_SPACE]),
            delta_q_mask: Some([1.0; HYDRA_ACTION_SPACE]),
            belief_fields: None,
            mixture_weights: None,
            belief_fields_present: false,
            mixture_weights_present: false,
        },
    ];

    let mut group = c.benchmark_group("validation");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(8));
    group.bench_function("collate_forward_loss", |b| {
        b.iter(|| {
            let (obs, batch, targets) = hydra_train::data::sample::collate_samples_owned::<
                ValidBackend,
            >(&samples, false, &device)
            .expect("validation bench collate should succeed")
            .expect("validation bench batch should exist");
            let output = valid.forward(obs.clone());
            let breakdown = loss_fn.total_loss(&output, &targets);
            let total = hydra_train::training::bc::bc_total_with_optional_exit_from_breakdown(
                &output,
                Some(&batch),
                &breakdown,
                &BcExitConfig::default(),
            );
            black_box(total.to_data())
        });
    });

    group.bench_function("collate_only", |b| {
        b.iter(|| {
            let batch = hydra_train::data::sample::collate_samples_owned::<ValidBackend>(
                &samples, false, &device,
            )
            .expect("validation bench collate should succeed")
            .expect("validation bench batch should exist");
            black_box(batch.0.dims())
        });
    });

    group.bench_function("collate_bc_owned_only", |b| {
        b.iter(|| {
            let batch = collate_samples_bc_owned::<ValidBackend>(&samples, false, &device)
                .expect("validation bench bc-owned collate should succeed")
                .expect("validation bench bc-owned batch should exist");
            black_box((
                batch.0.dims(),
                batch.1.actions.dims(),
                batch.2.legal_mask.dims(),
            ))
        });
    });

    group.bench_function("collate_owned_only", |b| {
        b.iter(|| {
            let batch = collate_samples_owned::<ValidBackend>(&samples, false, &device)
                .expect("validation bench owned collate should succeed")
                .expect("validation bench owned batch should exist");
            black_box((
                batch.0.dims(),
                batch.1.actions.dims(),
                batch.2.legal_mask.dims(),
            ))
        });
    });

    let (bench_obs, bench_batch, bench_targets) =
        hydra_train::data::sample::collate_samples_owned::<ValidBackend>(&samples, false, &device)
            .expect("validation bench collate should succeed")
            .expect("validation bench batch should exist");
    group.bench_function("forward_loss_only", |b| {
        b.iter(|| {
            let output = valid.forward(bench_obs.clone());
            let breakdown = loss_fn.total_loss(&output, &bench_targets);
            let total = bc_total_with_optional_exit_from_breakdown(
                &output,
                Some(&bench_batch),
                &breakdown,
                &BcExitConfig::default(),
            );
            black_box(total.to_data())
        });
    });
    group.bench_function("candidate_forward_and_loss", |b| {
        b.iter(|| {
            let output = valid.forward_with_warmup_train(bench_obs.clone(), &loss_fn.config, &[]);
            let breakdown = loss_fn.total_loss(&output, &bench_targets);
            let total = bc_total_with_optional_exit_from_breakdown(
                &output,
                Some(&bench_batch),
                &breakdown,
                &BcExitConfig::default(),
            );
            black_box(total.to_data())
        });
    });
    group.bench_function("candidate_metrics_only", |b| {
        let output = valid.forward_with_warmup_train(bench_obs.clone(), &loss_fn.config, &[]);
        let breakdown = loss_fn.total_loss(&output, &bench_targets);
        let total = bc_total_with_optional_exit_from_breakdown(
            &output,
            Some(&bench_batch),
            &breakdown,
            &BcExitConfig::default(),
        );
        b.iter(|| {
            let metric_sums = bench_metric_sums_from_outputs(
                samples.len(),
                output.policy_logits.clone(),
                bench_targets.legal_mask.clone(),
                bench_batch.actions.clone(),
                total.clone(),
                &breakdown,
            );
            black_box(metric_sums.to_data())
        });
    });
    group.bench_function("baseline_policy_only", |b| {
        b.iter(|| {
            let policy = valid.forward_policy(bench_obs.clone());
            black_box(policy.to_data())
        });
    });
    group.finish();
}

fn bench_validation_stream_grouping(c: &mut Criterion) {
    let root = std::path::PathBuf::from("/home/nikketryhard/tmp/bench-validation-stream");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("bench temp dir");
    for idx in 0..4 {
        let replay_path = root.join(format!("game-{idx}.mjai.json"));
        std::fs::write(&replay_path, tiny_real_mjai_replay()).expect("fixture write");
    }

    let manifest =
        scan_data_sources_with_progress(&root, 0.0, &SourceFilterConfig::default(), None)
            .expect("manifest scan should succeed");
    let loader_config = StreamingLoaderConfig {
        buffer_games: 1,
        buffer_samples: 1,
        train_fraction: 0.0,
        seed: 0,
        archive_queue_bound: 1,
        max_skip_logs_per_source: 1,
        aggregate_skip_logs: true,
        source_filters: SourceFilterConfig::default(),
        replay_target_profile: ReplayTargetProfile::minimal_bc(),
        exit_sidecar: None,
        exit_sidecar_source_net_hash: None,
        exit_sidecar_source_version: None,
        delta_q_sidecar: None,
        delta_q_sidecar_source_net_hash: None,
        delta_q_sidecar_source_version: None,
        num_threads: None,
    };
    let microbatch_size = 4usize;

    let mut group = c.benchmark_group("validation_stream_grouping");
    group.sample_size(40);
    group.measurement_time(Duration::from_secs(8));

    group.bench_function("buffer_chunks", |b| {
        b.iter(|| {
            let mut batches = 0usize;
            let mut samples = 0usize;
            for buffer_result in stream_val_pass(&manifest, &loader_config, None) {
                let buffer = buffer_result.expect("validation stream should succeed");
                for chunk in buffer.chunks(microbatch_size) {
                    batches += 1;
                    samples += chunk.len();
                    black_box(chunk.len());
                }
            }
            black_box((batches, samples))
        });
    });

    group.bench_function("streamed_microbatches", |b| {
        b.iter(|| {
            let mut batches = 0usize;
            let mut samples = 0usize;
            for microbatch_result in
                stream_val_microbatches(&manifest, &loader_config, microbatch_size, None)
            {
                let microbatch = microbatch_result.expect("validation microbatch stream succeeds");
                batches += 1;
                samples += microbatch.len();
                black_box(microbatch.len());
            }
            black_box((batches, samples))
        });
    });

    group.finish();
    let _ = std::fs::remove_dir_all(&root);
}

fn test_step(player_id: u8, action: u8, reward: f32, done: bool, turn: u16) -> TrajectoryStep {
    let mut pi_old = [0.0f32; HYDRA_ACTION_SPACE];
    let mut legal_mask = [false; HYDRA_ACTION_SPACE];
    pi_old[action as usize] = 1.0;
    legal_mask[action as usize] = true;
    TrajectoryStep {
        obs: [0.0; OBS_SIZE],
        action,
        pi_old,
        legal_mask,
        exit_label: None,
        delta_q_label: None,
        reward,
        done,
        player_id,
        game_id: 0,
        turn,
        temperature: 1.0,
    }
}

fn bench_rl_batch_collation(c: &mut Criterion) {
    let device = LibTorchDevice::Cpu;
    let mut trajectory = Trajectory::new(1, 42);
    trajectory.final_scores = [32000, 24000, 22000, 22000];
    trajectory.steps.push(test_step(0, 5, 0.2, false, 0));
    trajectory.steps.push(test_step(1, 9, -0.1, true, 1));
    let values = vec![vec![0.3, -0.2]];
    let mut scratch = RlBatchScratch::default();

    let mut group = c.benchmark_group("selfplay_batch");
    group.sample_size(60);
    group.measurement_time(Duration::from_secs(6));
    group.bench_function("trajectories_to_rl_batch", |b| {
        b.iter(|| {
            let batch = trajectories_to_rl_batch::<TrainBackend>(
                std::slice::from_ref(&trajectory),
                &values,
                &default_gae_config(),
                &device,
            );
            black_box(batch.obs.dims())
        });
    });
    group.bench_function("trajectories_to_rl_batch_reuse", |b| {
        b.iter(|| {
            let batch = trajectories_to_rl_batch_reuse::<TrainBackend>(
                std::slice::from_ref(&trajectory),
                &values,
                &default_gae_config(),
                &device,
                &mut scratch,
            );
            black_box(batch.obs.dims())
        });
    });
    group.finish();
}

fn bench_model_cpu_bridge(c: &mut Criterion) {
    let device = LibTorchDevice::Cpu;
    let model = HydraModelConfig::actor().init::<TrainBackend>(&device);
    let obs = [0.0f32; OBS_SIZE];
    let batch_obs = vec![[0.0f32; OBS_SIZE]; 8];
    let mut flat_buf = Vec::new();
    let mut outputs_buf = Vec::new();
    let mut values_buf = Vec::new();

    let mut group = c.benchmark_group("model_cpu_bridge");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(6));
    group.bench_function("policy_value_cpu", |b| {
        b.iter(|| {
            let out = model.policy_value_cpu(&obs, &device);
            black_box(out)
        });
    });
    group.bench_function("policy_cpu", |b| {
        b.iter(|| {
            let out = model.policy_cpu(&obs, &device);
            black_box(out)
        });
    });
    group.bench_function("value_cpu", |b| {
        b.iter(|| {
            let out = model.value_cpu(&obs, &device);
            black_box(out)
        });
    });
    group.bench_function("policy_and_value_cpu", |b| {
        b.iter(|| {
            let out = model.policy_and_value_cpu(&obs, &device);
            black_box(out)
        });
    });
    group.bench_function("batch_policy_value_cpu_reuse", |b| {
        b.iter(|| {
            let out = model.batch_policy_value_cpu_reuse(
                &batch_obs,
                &device,
                &mut flat_buf,
                &mut outputs_buf,
            );
            black_box(out.len())
        });
    });
    group.bench_function("batch_value_cpu_reuse", |b| {
        b.iter(|| {
            let out =
                model.batch_value_cpu_reuse(&batch_obs, &device, &mut flat_buf, &mut values_buf);
            black_box(out.len())
        });
    });
    group.finish();
}

fn game_seeds_for_bench(base_seed: u64, games_per_batch: usize) -> Vec<u64> {
    (0..games_per_batch)
        .map(|offset| base_seed.wrapping_add(offset as u64))
        .collect()
}

fn black_box_selfplay_batch_source(source: hydra_train::selfplay::SelfPlayBatchSource) {
    let games = source.trajectories.len();
    let steps = source
        .trajectories
        .iter()
        .map(|trajectory| trajectory.steps.len())
        .sum::<usize>();
    let values = source.values.iter().map(Vec::len).sum::<usize>();
    black_box((games, steps, values));
}

fn bench_selfplay_source_generation(c: &mut Criterion) {
    let device = LibTorchDevice::Cpu;
    let model = tiny_model_config().init::<TrainBackend>(&device);
    let valid_model = model.valid();

    let mut group = c.benchmark_group("selfplay_source_generation");
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(10);

    for &games_per_batch in &[1usize, 2] {
        let seeds = game_seeds_for_bench(0x5EED_0000, games_per_batch);
        let temperature = 1.0f32;
        let rng_seed = 0xABCDEF01u64;
        let exit_cfg = LiveExitConfig {
            enabled: false,
            ..LiveExitConfig::default()
        };

        group.bench_function(format!("serial_exit_off/gpb_{games_per_batch}"), |b| {
            b.iter(|| {
                let source = generate_self_play_batch_source(
                    &seeds,
                    temperature,
                    rng_seed,
                    &valid_model,
                    &device,
                    exit_cfg.clone(),
                );
                black_box_selfplay_batch_source(source);
            });
        });

        group.bench_function(format!("coop_exit_off/gpb_{games_per_batch}"), |b| {
            b.iter(|| {
                let source = generate_self_play_batch_source_cooperative(
                    &seeds,
                    temperature,
                    rng_seed,
                    &valid_model,
                    &device,
                    exit_cfg.clone(),
                );
                black_box_selfplay_batch_source(source);
            });
        });

        group.bench_function(format!("coop_reuse_exit_off/gpb_{games_per_batch}"), |b| {
            let mut coordinator = CooperativeSelfPlayCoordinator::new();
            b.iter(|| {
                let source = generate_self_play_batch_source_cooperative_reuse(
                    &mut coordinator,
                    &seeds,
                    temperature,
                    rng_seed,
                    &valid_model,
                    &device,
                    exit_cfg.clone(),
                );
                black_box_selfplay_batch_source(source);
            });
        });
    }

    let exit_seeds = game_seeds_for_bench(0x5EED_2000, 1);
    let exit_temperature = 1.0f32;
    let exit_rng_seed = 0xABCDEF03u64;
    let exit_cfg = LiveExitConfig::default();

    group.bench_function("coop_exit_default/gpb_1", |b| {
        b.iter(|| {
            let source = generate_self_play_batch_source_cooperative(
                &exit_seeds,
                exit_temperature,
                exit_rng_seed,
                &valid_model,
                &device,
                exit_cfg.clone(),
            );
            black_box_selfplay_batch_source(source);
        });
    });

    group.finish();
}

fn tiny_model_config() -> HydraModelConfig {
    HydraModelConfig::new(1)
        .with_input_channels(hydra_core::encoder::NUM_CHANNELS)
        .with_hidden_channels(4)
        .with_num_groups(4)
        .with_se_bottleneck(1)
}

fn bench_model_cpu_bridge_tiny(c: &mut Criterion) {
    let device = LibTorchDevice::Cpu;
    let model = tiny_model_config().init::<TrainBackend>(&device);
    let obs = [0.0f32; OBS_SIZE];
    let batch_obs = vec![[0.0f32; OBS_SIZE]; 8];
    let mut flat_buf = Vec::new();
    let mut outputs_buf = Vec::new();
    let mut values_buf = Vec::new();

    let mut group = c.benchmark_group("model_cpu_bridge_tiny");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(4));
    group.bench_function("policy_value_cpu", |b| {
        b.iter(|| {
            let out = model.policy_value_cpu(&obs, &device);
            black_box(out)
        });
    });
    group.bench_function("policy_cpu", |b| {
        b.iter(|| {
            let out = model.policy_cpu(&obs, &device);
            black_box(out)
        });
    });
    group.bench_function("batch_policy_value_cpu_reuse", |b| {
        b.iter(|| {
            let out = model.batch_policy_value_cpu_reuse(
                &batch_obs,
                &device,
                &mut flat_buf,
                &mut outputs_buf,
            );
            black_box(out.len())
        });
    });
    group.bench_function("batch_value_cpu_reuse", |b| {
        b.iter(|| {
            let out =
                model.batch_value_cpu_reuse(&batch_obs, &device, &mut flat_buf, &mut values_buf);
            black_box(out.len())
        });
    });
    group.finish();
}

fn bench_shard_collation(c: &mut Criterion) {
    let root = std::path::PathBuf::from("/home/nikketryhard/tmp/bench-shards");
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("bench temp dir");
    let replay_path = root.join("game.mjai.json");
    std::fs::write(&replay_path, tiny_real_mjai_replay()).expect("fixture write");

    let shard_dir = root.join("shards");
    let build = build_bc_shards(&BuildBcShardsConfig {
        input: replay_path.clone(),
        output_dir: shard_dir.clone(),
        manifest_name: "manifest.json".into(),
        train_fraction: 1.0,
        shard_samples: 10_000,
        split_mode: BcShardSplitMode::Train,
        source_manifest: None,
        exit_sidecar: None,
        exit_sidecar_path: None,
        exit_provenance: SidecarProvenance::default(),
        delta_q_sidecar: None,
        delta_q_sidecar_path: None,
        delta_q_provenance: SidecarProvenance::default(),
    })
    .expect("shards should build");
    let reader = load_bc_shard_reader(&build.manifest_path, BcShardSplit::Train)
        .expect("reader should load");
    let device = Default::default();
    let sample_count = reader.sample_count();
    let mut scratch = reader.new_scratch(sample_count);

    let mut group = c.benchmark_group("shard_collation");
    group.sample_size(200);
    group.measurement_time(Duration::from_secs(8));

    group.bench_function("host_range_no_augment", |b| {
        b.iter(|| {
            reader
                .collate_host_batch_range_into(0, sample_count, false, &mut scratch)
                .expect("collate");
            black_box(scratch.batch_size)
        });
    });

    group.bench_function("host_range_augment", |b| {
        b.iter(|| {
            reader
                .collate_host_batch_range_into(0, sample_count, true, &mut scratch)
                .expect("collate");
            black_box(scratch.batch_size)
        });
    });

    group.bench_function("materialize_borrowed", |b| {
        b.iter(|| {
            reader
                .collate_host_batch_range_into(0, sample_count, false, &mut scratch)
                .expect("collate");
            let host_batch = scratch.take_batch();
            let batch = host_batch.materialize::<ValidBackend>(&device);
            black_box((
                batch.obs.dims(),
                batch.batch.actions.dims(),
                batch.targets.legal_mask.dims(),
            ))
        });
    });

    group.bench_function("materialize_owned", |b| {
        b.iter(|| {
            reader
                .collate_host_batch_range_into(0, sample_count, false, &mut scratch)
                .expect("collate");
            let host_batch = scratch.take_batch();
            let batch = host_batch.materialize_owned::<ValidBackend>(&device);
            black_box((
                batch.obs.dims(),
                batch.batch.actions.dims(),
                batch.targets.legal_mask.dims(),
            ))
        });
    });

    group.finish();
    let _ = std::fs::remove_dir_all(&root);
}

criterion_group!(
    benches,
    bench_loader,
    bench_shard_collation,
    bench_validation_batch_stats,
    bench_validation_stream_grouping,
    bench_rl_batch_collation,
    bench_model_cpu_bridge,
    bench_model_cpu_bridge_tiny,
    bench_selfplay_source_generation
);
criterion_main!(benches);
