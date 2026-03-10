use std::fs;
use std::io;
use std::path::Path;

use burn::prelude::*;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::data::mjai_loader::{load_game_from_path, MjaiDataset};
use crate::data::sample::{collate_batch, collate_batch_augmented, MjaiSample};
use crate::training::losses::HydraTargets;

const MJAI_LOAD_THREAD_STACK_SIZE: usize = 8 * 1024 * 1024;

fn next_seed(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    *seed
}

fn is_mjai_file(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name) if name.ends_with(".json") || name.ends_with(".json.gz")
    )
}

fn clone_sample(sample: &MjaiSample) -> MjaiSample {
    MjaiSample {
        obs: sample.obs,
        action: sample.action,
        legal_mask: sample.legal_mask,
        placement: sample.placement,
        score_delta: sample.score_delta,
        grp_label: sample.grp_label,
        oracle_target: sample.oracle_target,
        tenpai: sample.tenpai,
        opp_next: sample.opp_next,
        danger: sample.danger,
        danger_mask: sample.danger_mask,
        safety_residual: sample.safety_residual,
        safety_residual_mask: sample.safety_residual_mask,
        belief_fields: sample.belief_fields,
        mixture_weights: sample.mixture_weights,
        belief_fields_present: sample.belief_fields_present,
        mixture_weights_present: sample.mixture_weights_present,
    }
}

pub fn load_mjai_directory(dir: &Path, train_fraction: f32) -> io::Result<MjaiDataset> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        if file_type.is_file() && is_mjai_file(&path) {
            paths.push(path);
        }
    }
    paths.sort();

    let mut dataset = MjaiDataset::new(train_fraction);
    dataset.games.reserve(paths.len());
    let pool = ThreadPoolBuilder::new()
        .stack_size(MJAI_LOAD_THREAD_STACK_SIZE)
        .build()
        .map_err(|err| {
            io::Error::other(format!("failed to build MJAI loader thread pool: {err}"))
        })?;
    let results: Vec<_> = pool.install(|| {
        paths
            .par_iter()
            .map(|path| (path.clone(), load_game_from_path(path)))
            .collect()
    });

    let mut skipped = 0usize;
    for (path, result) in results {
        match result {
            Ok(game) => dataset.add_game(game),
            Err(err) => {
                eprintln!("Skipping {}: {err}", path.display());
                skipped += 1;
            }
        }
    }

    println!(
        "Loaded {} MJAI games ({} samples, {} skipped) from {}",
        dataset.num_games(),
        dataset.num_samples(),
        skipped,
        dir.display()
    );

    Ok(dataset)
}

pub fn collect_samples(dataset: &MjaiDataset) -> Vec<&MjaiSample> {
    dataset
        .games
        .iter()
        .flat_map(|game| game.samples.iter())
        .collect()
}

pub fn shuffle_samples(samples: &mut [&MjaiSample], seed: u64) {
    let mut state = seed;
    for idx in (1..samples.len()).rev() {
        let swap_idx = (next_seed(&mut state) % (idx as u64 + 1)) as usize;
        samples.swap(idx, swap_idx);
    }
}

pub fn build_batches<B: Backend>(
    samples: &[&MjaiSample],
    batch_size: usize,
    augment: bool,
    device: &B::Device,
) -> Vec<(Tensor<B, 3>, HydraTargets<B>)> {
    if samples.is_empty() || batch_size == 0 {
        return Vec::new();
    }

    samples
        .chunks(batch_size)
        .map(|chunk| {
            let owned: Vec<MjaiSample> = chunk.iter().map(|sample| clone_sample(sample)).collect();
            let batch = if augment {
                collate_batch_augmented(&owned, device)
            } else {
                collate_batch(&owned, device)
            };
            (batch.obs.clone(), batch.into_hydra_targets())
        })
        .collect()
}

pub fn collate_sample_chunk<B: Backend>(
    samples: &[&MjaiSample],
    augment: bool,
    device: &B::Device,
) -> Option<(Tensor<B, 3>, HydraTargets<B>)> {
    if samples.is_empty() {
        return None;
    }

    let owned: Vec<MjaiSample> = samples.iter().map(|sample| clone_sample(sample)).collect();
    let batch = if augment {
        collate_batch_augmented(&owned, device)
    } else {
        collate_batch(&owned, device)
    };
    Some((batch.obs.clone(), batch.into_hydra_targets()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use hydra_core::action::HYDRA_ACTION_SPACE;
    use hydra_core::encoder::OBS_SIZE;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::data::mjai_loader::{MjaiDataset, MjaiGame};

    type B = NdArray<f32>;

    fn dummy_sample(action: u8) -> MjaiSample {
        let mut legal_mask = [0.0f32; HYDRA_ACTION_SPACE];
        legal_mask[action as usize] = 1.0;

        MjaiSample {
            obs: [0.25; OBS_SIZE],
            action,
            legal_mask,
            placement: 0,
            score_delta: 0,
            grp_label: 0,
            oracle_target: None,
            tenpai: [0.0; 3],
            opp_next: [255; 3],
            danger: [0.0; 102],
            danger_mask: [1.0; 102],
            safety_residual: None,
            safety_residual_mask: None,
            belief_fields: None,
            mixture_weights: None,
            belief_fields_present: false,
            mixture_weights_present: false,
        }
    }

    fn dataset_with_samples(num_samples: usize) -> MjaiDataset {
        let mut dataset = MjaiDataset::new(0.9);
        dataset.add_game(MjaiGame {
            samples: (0..num_samples)
                .map(|idx| dummy_sample((idx % HYDRA_ACTION_SPACE) as u8))
                .collect(),
            final_scores: [25_000; 4],
        });
        dataset
    }

    #[test]
    fn test_collect_samples_empty() {
        let dataset = MjaiDataset::new(0.9);
        let samples = collect_samples(&dataset);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_build_batches_empty() {
        let device = Default::default();
        let batches = build_batches::<B>(&[], 4, false, &device);
        assert!(batches.is_empty());
    }

    #[test]
    fn test_build_batches_creates_correct_count() {
        let dataset = dataset_with_samples(10);
        let samples = collect_samples(&dataset);
        let device = Default::default();
        let batches = build_batches::<B>(&samples, 4, false, &device);
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].0.dims()[0], 4);
        assert_eq!(batches[1].0.dims()[0], 4);
        assert_eq!(batches[2].0.dims()[0], 2);
    }

    #[test]
    fn test_shuffle_samples_deterministic() {
        let dataset = dataset_with_samples(6);
        let mut a = collect_samples(&dataset);
        let mut b = collect_samples(&dataset);
        shuffle_samples(&mut a, 42);
        shuffle_samples(&mut b, 42);
        let actions_a: Vec<u8> = a.iter().map(|sample| sample.action).collect();
        let actions_b: Vec<u8> = b.iter().map(|sample| sample.action).collect();
        assert_eq!(actions_a, actions_b);
    }

    #[test]
    fn test_collate_sample_chunk_matches_requested_batch_size() {
        let dataset = dataset_with_samples(5);
        let samples = collect_samples(&dataset);
        let device = Default::default();
        let (obs, targets) =
            collate_sample_chunk::<B>(&samples[..3], false, &device).expect("chunk should collate");
        assert_eq!(obs.dims()[0], 3);
        assert_eq!(targets.policy_target.dims()[0], 3);
    }

    #[test]
    fn test_load_mjai_directory_parallel_keeps_sorted_successes_and_skip_count() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("hydra_pipeline_loader_{unique}"));
        fs::create_dir_all(&dir).expect("create temp mjai dir");

        let valid_game = [
            r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["P","F","C","1m","1m","2m","2m","3m","3m","4m","4m","5m","5m"],["6p","6p","7p","7p","8p","8p","9p","9p","1s","1s","2s","2s","3s"]]}"#,
            r#"{"type":"end_kyoku"}"#,
        ]
        .join("\n");

        let good_a = dir.join("a_valid.json");
        let good_b = dir.join("b_valid.json");
        let bad = dir.join("c_invalid.json");

        fs::write(&good_a, &valid_game).expect("write first valid game");
        fs::write(&good_b, &valid_game).expect("write second valid game");
        let mut file = fs::File::create(&bad).expect("create bad file");
        writeln!(file, "{{not valid json").expect("write invalid json");

        let dataset = load_mjai_directory(&dir, 0.5).expect("directory load should succeed");
        assert_eq!(dataset.num_games(), 2);
        assert_eq!(dataset.games.len(), 2);

        fs::remove_dir_all(&dir).ok();
    }
}
