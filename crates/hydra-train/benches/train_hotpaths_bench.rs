use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;

use burn::backend::libtorch::LibTorchDevice;
use burn::backend::{Autodiff, LibTorch};
use burn::module::AutodiffModule;
use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::arena::{Trajectory, TrajectoryStep};
use hydra_core::encoder::OBS_SIZE;
use hydra_train::data::mjai_loader::load_game_from_reader;
use hydra_train::data::sample::MjaiSample;
use hydra_train::model::HydraModelConfig;
use hydra_train::selfplay_batch::{
    default_gae_config, trajectories_to_rl_batch, trajectories_to_rl_batch_reuse, RlBatchScratch,
};
use hydra_train::training::bc::BcExitConfig;
use hydra_train::training::losses::{HydraLoss, HydraLossConfig};

type TrainBackend = Autodiff<LibTorch<f32>>;
type ValidBackend = <TrainBackend as burn::tensor::backend::AutodiffBackend>::InnerBackend;

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
    group.bench_function("load_game_from_reader", |b| {
        b.iter(|| {
            let cursor = std::io::Cursor::new(payload.as_bytes());
            let game = load_game_from_reader(cursor).expect("loader bench replay should parse");
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

    let (bench_obs, bench_batch, bench_targets) =
        hydra_train::data::sample::collate_samples_owned::<ValidBackend>(&samples, false, &device)
            .expect("validation bench collate should succeed")
            .expect("validation bench batch should exist");
    group.bench_function("forward_loss_only", |b| {
        b.iter(|| {
            let output = valid.forward(bench_obs.clone());
            let breakdown = loss_fn.total_loss(&output, &bench_targets);
            let total = hydra_train::training::bc::bc_total_with_optional_exit_from_breakdown(
                &output,
                Some(&bench_batch),
                &breakdown,
                &BcExitConfig::default(),
            );
            black_box(total.to_data())
        });
    });
    group.finish();
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

fn tiny_model_config() -> HydraModelConfig {
    HydraModelConfig::new(1)
        .with_input_channels(hydra_core::encoder::NUM_CHANNELS)
        .with_hidden_channels(32)
        .with_num_groups(4)
        .with_se_bottleneck(8)
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

criterion_group!(
    benches,
    bench_loader,
    bench_validation_batch_stats,
    bench_rl_batch_collation,
    bench_model_cpu_bridge,
    bench_model_cpu_bridge_tiny
);
criterion_main!(benches);
