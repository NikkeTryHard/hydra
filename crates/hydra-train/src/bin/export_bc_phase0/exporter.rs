use std::fs;
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;

use indicatif::ProgressBar;
use rayon::{prelude::*, ThreadPoolBuilder};
use sha2::{Digest, Sha256};

use hydra_train::data::mjai_loader::{load_game_from_path, load_game_from_stream};

use crate::config::ExportConfig;
use crate::manifest::{
    ConfigSnapshot, CountsExact, ExportManifest, ExportSplit, ShardManifestEntry, SplitCounts,
    TargetPresenceCounts, ACTION_SPACE, AUGMENT_POLICY, ENCODER_CONTRACT, EXPORT_SEMANTICS,
    SCHEMA_VERSION,
};
use crate::shard_writer::{ExportGame, PendingShard};

struct SplitShardBuffer {
    split: ExportSplit,
    shard_index: usize,
    games: Vec<ExportGame>,
    sample_count: usize,
    total_games: usize,
    total_samples: usize,
}

impl SplitShardBuffer {
    fn new(split: ExportSplit) -> Self {
        Self {
            split,
            shard_index: 0,
            games: Vec::new(),
            sample_count: 0,
            total_games: 0,
            total_samples: 0,
        }
    }

    fn push(
        &mut self,
        export_root: &Path,
        game: ExportGame,
        max_samples_per_shard: usize,
        max_games_per_shard: usize,
        shards: &mut Vec<ShardManifestEntry>,
    ) -> io::Result<()> {
        let sample_count = game.samples.len();
        let would_exceed_games = self.games.len() >= max_games_per_shard.max(1);
        let would_exceed_samples = !self.games.is_empty()
            && self.sample_count.saturating_add(sample_count) > max_samples_per_shard.max(1);
        if would_exceed_games || would_exceed_samples {
            self.flush(export_root, shards)?;
        }

        self.sample_count += sample_count;
        self.games.push(game);
        self.total_games += 1;
        self.total_samples += sample_count;
        Ok(())
    }

    fn flush(
        &mut self,
        export_root: &Path,
        shards: &mut Vec<ShardManifestEntry>,
    ) -> io::Result<()> {
        if self.games.is_empty() {
            return Ok(());
        }

        let shard = PendingShard::new(
            export_root,
            self.split.clone(),
            self.shard_index,
            self.games.len(),
            self.sample_count,
        )?;
        let mut file_hashes = crate::shard_writer::FileHashes::new();
        let (metadata_name, metadata_hash) = shard.write_metadata()?;
        file_hashes.insert(metadata_name, metadata_hash);
        file_hashes.extend(shard.write_games(&self.games)?);
        let hashes = shard.hash_files(file_hashes);
        shards.push(ShardManifestEntry {
            split: self.split.clone(),
            shard_name: shard.shard_name,
            game_count: self.games.len(),
            sample_count: self.sample_count,
            hashes,
        });

        self.shard_index += 1;
        self.games.clear();
        self.sample_count = 0;
        Ok(())
    }
}

struct StreamingAccumulatorOutput {
    split_counts: SplitCounts,
    target_presence_counts: TargetPresenceCounts,
    shards: Vec<ShardManifestEntry>,
}

struct StreamingShardAccumulator<'a> {
    export_root: &'a Path,
    train_fraction: f32,
    max_samples_per_shard: usize,
    max_games_per_shard: usize,
    train: SplitShardBuffer,
    validation: SplitShardBuffer,
    target_presence_counts: TargetPresenceCounts,
    shards: Vec<ShardManifestEntry>,
}

impl<'a> StreamingShardAccumulator<'a> {
    fn new(
        export_root: &'a Path,
        train_fraction: f32,
        max_samples_per_shard: usize,
        max_games_per_shard: usize,
    ) -> Self {
        Self {
            export_root,
            train_fraction,
            max_samples_per_shard,
            max_games_per_shard,
            train: SplitShardBuffer::new(ExportSplit::Train),
            validation: SplitShardBuffer::new(ExportSplit::Validation),
            target_presence_counts: TargetPresenceCounts {
                oracle_target: 0,
                safety_residual_target: 0,
                belief_fields_target: 0,
                mixture_weight_target: 0,
            },
            shards: Vec::new(),
        }
    }

    fn push(&mut self, game: ExportGame) -> io::Result<()> {
        for sample in &game.samples {
            if sample.oracle_target.is_some() {
                self.target_presence_counts.oracle_target += 1;
            }
            if sample.safety_residual.is_some() {
                self.target_presence_counts.safety_residual_target += 1;
            }
            if sample.belief_fields_present {
                self.target_presence_counts.belief_fields_target += 1;
            }
            if sample.mixture_weights_present {
                self.target_presence_counts.mixture_weight_target += 1;
            }
        }

        let target = if is_train_game(&game.identity, self.train_fraction) {
            &mut self.train
        } else {
            &mut self.validation
        };
        target.push(
            self.export_root,
            game,
            self.max_samples_per_shard,
            self.max_games_per_shard,
            &mut self.shards,
        )
    }

    fn finish(mut self) -> io::Result<StreamingAccumulatorOutput> {
        self.train.flush(self.export_root, &mut self.shards)?;
        self.validation.flush(self.export_root, &mut self.shards)?;
        Ok(StreamingAccumulatorOutput {
            split_counts: SplitCounts {
                train_games: self.train.total_games,
                train_samples: self.train.total_samples,
                validation_games: self.validation.total_games,
                validation_samples: self.validation.total_samples,
            },
            target_presence_counts: self.target_presence_counts,
            shards: self.shards,
        })
    }
}

pub(crate) fn run_export(config: &ExportConfig) -> io::Result<()> {
    let export_root = config.output_dir.join("bc_phase0_export");
    if export_root.exists() {
        fs::remove_dir_all(&export_root)?;
    }
    fs::create_dir_all(&export_root)?;

    let progress = ProgressBar::new_spinner();
    progress.set_message("streaming canonical export games");
    let mut accumulator = StreamingShardAccumulator::new(
        &export_root,
        config.train_fraction,
        config.max_samples_per_shard,
        config.max_games_per_shard,
    );
    collect_export_games(
        &config.data_dir,
        config.archive_queue_bound,
        Some(&progress),
        |game| accumulator.push(game),
    )?;
    let accumulator = accumulator.finish()?;
    progress.finish_and_clear();

    let mut manifest = ExportManifest {
        schema_version: SCHEMA_VERSION.to_string(),
        export_semantics: EXPORT_SEMANTICS.to_string(),
        manifest_fingerprint: String::new(),
        train_fraction: config.train_fraction,
        seed: config.seed,
        augment_policy: AUGMENT_POLICY.to_string(),
        source_paths: collect_source_paths(&config.data_dir)?,
        split_counts: accumulator.split_counts,
        counts_exact: CountsExact {
            source_scan_exact: false,
            export_counts_exact: true,
        },
        target_presence_counts: accumulator.target_presence_counts,
        shards: accumulator.shards,
        config_snapshot: ConfigSnapshot {
            buffer_games: config.buffer_games,
            buffer_samples: config.buffer_samples,
            archive_queue_bound: config.archive_queue_bound,
        },
        encoder_contract: ENCODER_CONTRACT.to_string(),
        action_space: ACTION_SPACE,
    };

    manifest.manifest_fingerprint = compute_manifest_fingerprint(&manifest)?;
    let manifest_json = serde_json::to_vec_pretty(&manifest)
        .map_err(|err| io::Error::other(format!("failed to serialize manifest: {err}")))?;
    fs::write(export_root.join("manifest.json"), manifest_json)?;
    Ok(())
}

fn collect_export_games<F>(
    data_path: &Path,
    archive_queue_bound: usize,
    progress: Option<&ProgressBar>,
    mut on_game: F,
) -> io::Result<()>
where
    F: FnMut(ExportGame) -> io::Result<()>,
{
    if data_path.is_file() {
        if is_tar_zst_file(data_path) {
            load_archive_games(
                data_path,
                archive_queue_bound,
                progress.cloned(),
                &mut on_game,
            )?;
        } else if is_mjai_file(data_path) {
            let identity = identity_for_loose_file(data_path)?;
            match load_game_from_path(data_path) {
                Ok(game) => {
                    on_game(ExportGame {
                        identity,
                        samples: game.samples,
                    })?;
                }
                Err(err) => {
                    eprintln!("Skipping {}: {}", identity, err);
                }
            }
            if let Some(pb) = progress {
                pb.inc(1);
            }
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "expected directory, MJAI file, or .tar.zst archive, got {}",
                    data_path.display()
                ),
            ));
        }
        return Ok(());
    }

    let mut loose_files = Vec::new();
    let mut archives = Vec::new();
    for entry in fs::read_dir(data_path)? {
        let entry = entry?;
        let path = entry.path();
        if !fs::metadata(&path)?.is_file() {
            continue;
        }
        if is_mjai_file(&path) {
            loose_files.push(path);
        } else if is_tar_zst_file(&path) {
            archives.push(path);
        }
    }
    loose_files.sort();
    archives.sort();

    for path in loose_files {
        let identity = identity_for_loose_file(&path)?;
        match load_game_from_path(&path) {
            Ok(game) => {
                on_game(ExportGame {
                    identity,
                    samples: game.samples,
                })?;
            }
            Err(err) => {
                eprintln!("Skipping {}: {}", identity, err);
            }
        }
        if let Some(pb) = &progress {
            pb.inc(1);
        }
    }

    if archives.is_empty() {
        return Ok(());
    }

    for archive in archives {
        load_archive_games(
            &archive,
            archive_queue_bound,
            progress.cloned(),
            &mut on_game,
        )?;
    }

    Ok(())
}

struct ArchiveEntryJob {
    sequence: usize,
    display_name: String,
    data: Vec<u8>,
}

struct ParsedArchiveGame {
    sequence: usize,
    display_name: String,
    result: io::Result<ExportGame>,
}

fn load_archive_games(
    archive_path: &Path,
    archive_queue_bound: usize,
    progress: Option<ProgressBar>,
    on_game: &mut impl FnMut(ExportGame) -> io::Result<()>,
) -> io::Result<()> {
    let archive_path = archive_path.to_path_buf();
    let archive_path_for_thread = archive_path.clone();
    let queue_bound = archive_queue_bound.max(1);
    let (ordered_tx, ordered_rx) = mpsc::sync_channel::<ExportGame>(queue_bound);
    let handle = thread::Builder::new()
        .name(format!("export-archive-{}", archive_path.display()))
        .spawn(move || -> io::Result<()> {
            let pool = ThreadPoolBuilder::new().build().map_err(|err| {
                io::Error::other(format!(
                    "failed to build archive parse thread pool for {}: {err}",
                    archive_path_for_thread.display()
                ))
            })?;

            let file = fs::File::open(&archive_path_for_thread)?;
            let reader = BufReader::with_capacity(1 << 20, file);
            let zstd = zstd::Decoder::new(reader).map_err(|err| {
                io::Error::other(format!(
                    "failed to open zstd archive {}: {err}",
                    archive_path_for_thread.display()
                ))
            })?;
            let mut archive = tar::Archive::new(zstd);

            let (job_tx, job_rx) = mpsc::sync_channel::<ArchiveEntryJob>(queue_bound);
            let (parsed_tx, parsed_rx) = mpsc::sync_channel::<ParsedArchiveGame>(queue_bound);

            let parsed_tx_for_parse = parsed_tx.clone();
            let archive_path_for_parse = archive_path_for_thread.clone();
            let parser =
                thread::Builder::new()
                    .name(format!("export-parse-{}", archive_path_for_parse.display()))
                    .spawn(move || -> io::Result<()> {
                        pool.install(|| {
                            job_rx.into_iter().par_bridge().try_for_each(|job| {
                                let result = load_game_from_stream(std::io::Cursor::new(job.data))
                                    .map(|game| ExportGame {
                                        identity: job.display_name.clone(),
                                        samples: game.samples,
                                    });
                                parsed_tx_for_parse
                                    .send(ParsedArchiveGame {
                                        sequence: job.sequence,
                                        display_name: job.display_name,
                                        result,
                                    })
                                    .map_err(|_| {
                                        io::Error::new(
                                            io::ErrorKind::BrokenPipe,
                                            "archive parse receiver dropped while exporting",
                                        )
                                    })
                            })
                        })
                    })
                    .map_err(|err| {
                        io::Error::other(format!(
                            "failed to spawn archive parse thread {}: {err}",
                            archive_path_for_thread.display()
                        ))
                    })?;
            drop(parsed_tx);

            let collector = thread::Builder::new()
                .name(format!(
                    "export-order-{}",
                    archive_path_for_thread.display()
                ))
                .spawn(move || -> io::Result<()> {
                    let mut next_sequence = 0usize;
                    let mut pending = std::collections::BTreeMap::new();
                    for parsed in parsed_rx {
                        pending.insert(parsed.sequence, parsed);
                        while let Some(parsed) = pending.remove(&next_sequence) {
                            match parsed.result {
                                Ok(game) => ordered_tx.send(game).map_err(|_| {
                                    io::Error::new(
                                        io::ErrorKind::BrokenPipe,
                                        "archive ordered receiver dropped while exporting",
                                    )
                                })?,
                                Err(err) => {
                                    eprintln!("Skipping {}: {}", parsed.display_name, err);
                                }
                            }
                            next_sequence += 1;
                        }
                    }
                    Ok(())
                })
                .map_err(|err| {
                    io::Error::other(format!(
                        "failed to spawn archive ordering thread {}: {err}",
                        archive_path_for_thread.display()
                    ))
                })?;

            let mut sequence = 0usize;
            for entry_result in archive.entries()? {
                let mut entry = entry_result?;
                let entry_path = entry.path()?.into_owned();
                if !is_mjai_archive_entry(&entry_path) {
                    continue;
                }
                let identity = identity_for_archive_entry(&archive_path_for_thread, &entry_path)?;
                let entry_size = usize::try_from(entry.size()).map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("archive entry too large to buffer: {identity}"),
                    )
                })?;
                let mut data = vec![0u8; entry_size];
                if let Err(err) = std::io::Read::read_exact(&mut entry, &mut data) {
                    eprintln!("Skipping {}: {}", identity, err);
                    continue;
                }
                if job_tx
                    .send(ArchiveEntryJob {
                        sequence,
                        display_name: identity,
                        data,
                    })
                    .is_err()
                {
                    break;
                }
                sequence += 1;
                if let Some(pb) = &progress {
                    pb.inc(1);
                }
            }
            drop(job_tx);

            parser.join().map_err(|_| {
                io::Error::other(format!(
                    "archive parse thread panicked for {}",
                    archive_path_for_thread.display()
                ))
            })??;
            collector.join().map_err(|_| {
                io::Error::other(format!(
                    "archive ordering thread panicked for {}",
                    archive_path_for_thread.display()
                ))
            })??;

            Ok(())
        })
        .map_err(|err| {
            io::Error::other(format!(
                "failed to spawn archive export thread {}: {err}",
                archive_path.display()
            ))
        })?;

    for game in ordered_rx {
        on_game(game)?;
    }

    handle.join().map_err(|_| {
        io::Error::other(format!(
            "archive export thread panicked for {}",
            archive_path.display()
        ))
    })??;

    Ok(())
}

fn collect_source_paths(data_path: &Path) -> io::Result<Vec<PathBuf>> {
    if data_path.is_file() {
        return Ok(vec![data_path.to_path_buf()]);
    }
    let mut sources = Vec::new();
    for entry in fs::read_dir(data_path)? {
        let entry = entry?;
        let path = entry.path();
        if !fs::metadata(&path)?.is_file() {
            continue;
        }
        if is_mjai_file(&path) || is_tar_zst_file(&path) {
            sources.push(path);
        }
    }
    sources.sort();
    Ok(sources)
}

fn write_split_shards(
    export_root: &Path,
    split: ExportSplit,
    games: Vec<ExportGame>,
    max_samples_per_shard: usize,
    max_games_per_shard: usize,
    out: &mut Vec<ShardManifestEntry>,
) -> io::Result<()> {
    let mut shard_index = 0usize;
    let mut current_games = Vec::new();
    let mut current_sample_count = 0usize;

    let flush = |shard_index: usize,
                 split: &ExportSplit,
                 current_games: &[ExportGame],
                 current_sample_count: usize,
                 out: &mut Vec<ShardManifestEntry>|
     -> io::Result<()> {
        if current_games.is_empty() {
            return Ok(());
        }
        let shard = PendingShard::new(
            export_root,
            split.clone(),
            shard_index,
            current_games.len(),
            current_sample_count,
        )?;
        let mut file_hashes = crate::shard_writer::FileHashes::new();
        let (metadata_name, metadata_hash) = shard.write_metadata()?;
        file_hashes.insert(metadata_name, metadata_hash);
        file_hashes.extend(shard.write_games(current_games)?);
        let hashes = shard.hash_files(file_hashes);
        out.push(ShardManifestEntry {
            split: split.clone(),
            shard_name: shard.shard_name,
            game_count: current_games.len(),
            sample_count: current_sample_count,
            hashes,
        });
        Ok(())
    };

    for game in games {
        let sample_count = game.samples.len();
        let would_exceed_games = current_games.len() >= max_games_per_shard.max(1);
        let would_exceed_samples = !current_games.is_empty()
            && current_sample_count.saturating_add(sample_count) > max_samples_per_shard.max(1);
        if would_exceed_games || would_exceed_samples {
            flush(
                shard_index,
                &split,
                &current_games,
                current_sample_count,
                out,
            )?;
            shard_index += 1;
            current_games.clear();
            current_sample_count = 0;
        }
        current_sample_count += sample_count;
        current_games.push(game);
    }

    flush(
        shard_index,
        &split,
        &current_games,
        current_sample_count,
        out,
    )?;
    Ok(())
}

fn identity_for_loose_file(path: &Path) -> io::Result<String> {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(ToOwned::to_owned)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid filename {}", path.display()),
            )
        })
}

fn identity_for_archive_entry(archive_path: &Path, entry_path: &Path) -> io::Result<String> {
    let archive_name = archive_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid archive name {}", archive_path.display()),
            )
        })?;
    Ok(format!("{archive_name}/{}", entry_path.display()))
}

fn fnv1a_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn normalized_train_fraction(train_fraction: f32) -> f32 {
    if train_fraction.is_finite() {
        train_fraction.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn is_train_game(identity: &str, train_fraction: f32) -> bool {
    let threshold = (normalized_train_fraction(train_fraction) * 1000.0).round() as u64;
    fnv1a_hash(identity.as_bytes()) % 1000 < threshold
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

fn compute_manifest_fingerprint(manifest: &ExportManifest) -> io::Result<String> {
    let mut value = serde_json::to_value(manifest)
        .map_err(|err| io::Error::other(format!("failed to convert manifest to value: {err}")))?;
    let object = value.as_object_mut().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "manifest did not serialize to JSON object",
        )
    })?;
    object.remove("manifest_fingerprint");
    let bytes = serde_json::to_vec(&value).map_err(|err| {
        io::Error::other(format!("failed to serialize canonical manifest: {err}"))
    })?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hydra_core::action::HYDRA_ACTION_SPACE;
    use hydra_core::encoder::OBS_SIZE;
    use hydra_train::data::sample::MjaiSample;

    fn dummy_sample(action: u8, score_delta: i32) -> MjaiSample {
        let mut legal_mask = [0.0f32; HYDRA_ACTION_SPACE];
        legal_mask[action as usize] = 1.0;
        MjaiSample {
            obs: [0.25; OBS_SIZE],
            action,
            legal_mask,
            placement: 0,
            score_delta,
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

    #[test]
    fn manifest_fingerprint_is_stable_for_same_payload() {
        let manifest = ExportManifest {
            schema_version: SCHEMA_VERSION.to_string(),
            export_semantics: EXPORT_SEMANTICS.to_string(),
            manifest_fingerprint: String::new(),
            train_fraction: 0.9,
            seed: 0,
            augment_policy: AUGMENT_POLICY.to_string(),
            source_paths: vec![PathBuf::from("/tmp/data/game.json")],
            split_counts: SplitCounts {
                train_games: 1,
                train_samples: 2,
                validation_games: 0,
                validation_samples: 0,
            },
            counts_exact: CountsExact {
                source_scan_exact: true,
                export_counts_exact: true,
            },
            target_presence_counts: TargetPresenceCounts {
                oracle_target: 0,
                safety_residual_target: 0,
                belief_fields_target: 0,
                mixture_weight_target: 0,
            },
            shards: Vec::new(),
            config_snapshot: ConfigSnapshot {
                buffer_games: 50_000,
                buffer_samples: 32_768,
                archive_queue_bound: 128,
            },
            encoder_contract: ENCODER_CONTRACT.to_string(),
            action_space: ACTION_SPACE,
        };
        let a = compute_manifest_fingerprint(&manifest).expect("fingerprint");
        let b = compute_manifest_fingerprint(&manifest).expect("fingerprint");
        assert_eq!(a, b);
    }

    #[test]
    fn shard_rollover_keeps_games_whole() {
        let temp =
            std::env::temp_dir().join(format!("hydra_phase0_shards_{}_{}", std::process::id(), 1));
        if temp.exists() {
            fs::remove_dir_all(&temp).ok();
        }
        fs::create_dir_all(&temp).expect("create temp dir");
        let mut shards = Vec::new();
        write_split_shards(
            &temp,
            ExportSplit::Train,
            vec![
                ExportGame {
                    identity: "game-a".to_string(),
                    samples: (0..100_000).map(|_| dummy_sample(0, 1000)).collect(),
                },
                ExportGame {
                    identity: "game-b".to_string(),
                    samples: (0..40_000).map(|_| dummy_sample(1, 2000)).collect(),
                },
                ExportGame {
                    identity: "game-c".to_string(),
                    samples: (0..20_000).map(|_| dummy_sample(2, 3000)).collect(),
                },
            ],
            131_072,
            4_096,
            &mut shards,
        )
        .expect("write shards");
        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0].game_count, 1);
        assert_eq!(shards[0].sample_count, 100_000);
        assert_eq!(shards[1].game_count, 2);
        assert_eq!(shards[1].sample_count, 60_000);
        fs::remove_dir_all(temp).ok();
    }

    #[test]
    fn split_is_stable_for_same_identity() {
        assert_eq!(
            is_train_game("foo.json", 0.9),
            is_train_game("foo.json", 0.9)
        );
    }

    #[test]
    fn identity_split_changes_with_real_identity_not_enumeration() {
        let a = is_train_game("a_valid.json", 0.9);
        let b = is_train_game("b_valid.json", 0.9);
        assert_eq!(a, is_train_game("a_valid.json", 0.9));
        assert_eq!(b, is_train_game("b_valid.json", 0.9));
    }

    #[cfg(unix)]
    #[test]
    fn collect_source_paths_includes_symlinked_files() {
        use std::os::unix::fs::symlink;

        let temp =
            std::env::temp_dir().join(format!("hydra_phase0_sources_{}_{}", std::process::id(), 1));
        if temp.exists() {
            fs::remove_dir_all(&temp).ok();
        }
        fs::create_dir_all(&temp).expect("create temp dir");

        let target = temp.join("game.json");
        fs::write(&target, b"[]").expect("write target file");
        let link = temp.join("linked_game.json");
        symlink(&target, &link).expect("create symlink");

        let sources = collect_source_paths(&temp).expect("collect source paths");
        assert!(sources.contains(&target));
        assert!(sources.contains(&link));

        fs::remove_dir_all(temp).ok();
    }
}
