use std::fs;
use std::io::{self, BufReader};
use std::path::Path;
use std::sync::mpsc;
use std::thread;

use burn::prelude::*;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use crate::data::mjai_loader::{load_game_from_path, load_game_from_stream, MjaiDataset, MjaiGame};
use crate::data::sample::{collate_batch, collate_batch_augmented, MjaiSample};
use crate::training::losses::HydraTargets;

const MJAI_LOAD_THREAD_STACK_SIZE: usize = 8 * 1024 * 1024;
const MJAI_ARCHIVE_QUEUE_BOUND: usize = 128;

struct ArchiveEntryJob {
    display_name: String,
    data: Vec<u8>,
}

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

fn is_tar_zst_file(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name) if name.ends_with(".tar.zst") || name.contains(".tar-") && name.ends_with(".zst")
    )
}

fn is_mjai_archive_entry(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some(name) if name.ends_with(".json") || name.ends_with(".json.gz") || name.ends_with(".mjai.json") || name.ends_with(".mjai.json.gz")
    )
}

fn load_mjai_archive(path: &Path, train_fraction: f32) -> io::Result<MjaiDataset> {
    let pool = ThreadPoolBuilder::new()
        .stack_size(MJAI_LOAD_THREAD_STACK_SIZE)
        .build()
        .map_err(|err| {
            io::Error::other(format!(
                "failed to build MJAI archive loader thread pool: {err}"
            ))
        })?;

    let path_buf = path.to_path_buf();
    let (job_tx, job_rx) = mpsc::sync_channel::<ArchiveEntryJob>(MJAI_ARCHIVE_QUEUE_BOUND);

    let producer = thread::Builder::new()
        .name("mjai-archive-reader".to_string())
        .stack_size(MJAI_LOAD_THREAD_STACK_SIZE)
        .spawn(move || -> io::Result<()> {
            let file = fs::File::open(&path_buf)?;
            let zstd = zstd::Decoder::new(file).map_err(|err| {
                io::Error::other(format!(
                    "failed to open zstd archive {}: {err}",
                    path_buf.display()
                ))
            })?;
            let mut archive = tar::Archive::new(zstd);

            for entry_result in archive.entries()? {
                let mut entry = entry_result?;
                let entry_path = entry.path()?.into_owned();
                if !is_mjai_archive_entry(&entry_path) {
                    continue;
                }

                let mut data = Vec::with_capacity(entry.size() as usize);
                std::io::Read::read_to_end(&mut entry, &mut data)?;
                let display_name = format!("{} in {}", entry_path.display(), path_buf.display());

                if job_tx.send(ArchiveEntryJob { display_name, data }).is_err() {
                    break;
                }
            }

            Ok(())
        })
        .map_err(|err| io::Error::other(format!("failed to spawn archive reader: {err}")))?;

    let results: Vec<(String, io::Result<MjaiGame>)> = pool.install(|| {
        job_rx
            .into_iter()
            .par_bridge()
            .map(|job| {
                let result = load_game_from_stream(BufReader::new(std::io::Cursor::new(job.data)));
                (job.display_name, result)
            })
            .collect()
    });

    producer.join().map_err(|_| {
        io::Error::other(format!(
            "archive reader thread panicked for {}",
            path.display()
        ))
    })??;

    let mut dataset = MjaiDataset::new(train_fraction);
    let mut skipped = 0usize;

    for (display_name, result) in results {
        match result {
            Ok(game) => dataset.add_game(game),
            Err(err) => {
                eprintln!("Skipping {display_name}: {err}");
                skipped += 1;
            }
        }
    }

    println!(
        "Loaded {} MJAI games ({} samples, {} skipped) from archive {}",
        dataset.num_games(),
        dataset.num_samples(),
        skipped,
        path.display()
    );

    Ok(dataset)
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
    if dir.is_file() {
        if is_tar_zst_file(dir) {
            return load_mjai_archive(dir, train_fraction);
        }
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "expected directory or .tar.zst archive, got {}",
                dir.display()
            ),
        ));
    }

    let mut paths = Vec::new();
    let mut archives = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        if file_type.is_file() {
            if is_mjai_file(&path) {
                paths.push(path);
            } else if is_tar_zst_file(&path) {
                archives.push(path);
            }
        }
    }
    paths.sort();
    archives.sort();

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

    for archive in archives {
        let archive_dataset = load_mjai_archive(&archive, train_fraction)?;
        for game in archive_dataset.games {
            dataset.add_game(game);
        }
    }

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
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use hydra_core::action::HYDRA_ACTION_SPACE;
    use hydra_core::encoder::OBS_SIZE;
    use std::fs::File;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};
    use tar::Builder;

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

    fn valid_game_json() -> String {
        [
            r#"{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","4p"],["1s","2s","3s","4s","5s","6s","7s","8s","9s","E","S","W","N"],["P","F","C","1m","1m","2m","2m","3m","3m","4m","4m","5m","5m"],["6p","6p","7p","7p","8p","8p","9p","9p","1s","1s","2s","2s","3s"]]}"#,
            r#"{"type":"end_kyoku"}"#,
        ]
        .join("\n")
    }

    fn write_tar_zst_with_entries(path: &Path, entries: &[(&str, Vec<u8>)]) {
        let file = File::create(path).expect("create archive");
        let encoder = zstd::Encoder::new(file, 19).expect("create zstd encoder");
        let mut builder = Builder::new(encoder.auto_finish());
        for (name, data) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_data(&mut header, *name, data.as_slice())
                .expect("append tar entry");
        }
        builder.finish().expect("finish tar builder");
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

    #[test]
    fn test_load_mjai_directory_reads_tar_zst_archive() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        let archive_path =
            std::env::temp_dir().join(format!("hydra_pipeline_archive_{unique}.tar.zst"));

        let raw = valid_game_json();
        let mut gz = GzEncoder::new(Vec::new(), Compression::default());
        gz.write_all(raw.as_bytes()).expect("write gz payload");
        let gz_bytes = gz.finish().expect("finish gz payload");

        write_tar_zst_with_entries(
            &archive_path,
            &[
                ("game_a.mjai.json", raw.clone().into_bytes()),
                ("game_b.mjai.json.gz", gz_bytes),
                ("ignore.txt", b"nope".to_vec()),
            ],
        );

        let dataset = load_mjai_directory(&archive_path, 0.5).expect("archive load should succeed");
        assert_eq!(dataset.num_games(), 2);
        assert_eq!(dataset.games.len(), 2);

        fs::remove_file(&archive_path).ok();
    }

    #[test]
    fn test_load_mjai_directory_reads_mixed_dir_and_archives() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("hydra_pipeline_mixed_{unique}"));
        fs::create_dir_all(&dir).expect("create temp dir");

        let raw = valid_game_json();
        fs::write(dir.join("loose.json"), &raw).expect("write loose game");
        write_tar_zst_with_entries(
            &dir.join("pack.tar.zst"),
            &[("packed.mjai.json", raw.into_bytes())],
        );

        let dataset = load_mjai_directory(&dir, 0.5).expect("mixed load should succeed");
        assert_eq!(dataset.num_games(), 2);

        fs::remove_dir_all(&dir).ok();
    }
}
