use std::fs;
use std::io;
use std::path::Path;

use burn::prelude::*;

use crate::data::mjai_loader::{load_game_from_path, MjaiDataset};
use crate::data::sample::{collate_batch, collate_batch_augmented, MjaiSample};
use crate::training::losses::HydraTargets;

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
        let path = entry.path();
        if entry.file_type()?.is_file() && is_mjai_file(&path) {
            paths.push(path);
        }
    }
    paths.sort();

    let mut dataset = MjaiDataset::new(train_fraction);
    for path in &paths {
        dataset.add_game(load_game_from_path(path)?);
    }

    println!(
        "Loaded {} MJAI games ({} samples) from {}",
        dataset.num_games(),
        dataset.num_samples(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use hydra_core::action::HYDRA_ACTION_SPACE;
    use hydra_core::encoder::OBS_SIZE;

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
}
