use std::collections::BTreeMap;
use std::fs;
use std::io::{self, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use sha2::{Digest, Sha256};

use hydra_core::action::HYDRA_ACTION_SPACE;
use hydra_core::encoder::OBS_SIZE;
use hydra_train::data::sample::{
    score_delta_to_cdf, score_delta_to_pdf, score_delta_to_value, MjaiSample,
};

use crate::manifest::{ExportSplit, ShardArtifactHashes, ShardMetadata, SCHEMA_VERSION};

const SCORE_BINS: usize = 64;
const BELIEF_FIELD_SIZE: usize = 16 * 34;

pub(crate) struct ExportGame {
    pub(crate) identity: String,
    pub(crate) samples: Vec<MjaiSample>,
}

pub(crate) struct PendingShard {
    pub(crate) split: ExportSplit,
    pub(crate) shard_name: String,
    pub(crate) root: PathBuf,
    pub(crate) game_count: usize,
    pub(crate) sample_count: usize,
}

impl PendingShard {
    pub(crate) fn new(
        output_root: &Path,
        split: ExportSplit,
        shard_index: usize,
        game_count: usize,
        sample_count: usize,
    ) -> io::Result<Self> {
        let split_dir = match split {
            ExportSplit::Train => output_root.join("train"),
            ExportSplit::Validation => output_root.join("validation"),
        };
        fs::create_dir_all(&split_dir)?;
        let shard_name = format!("shard-{shard_index:06}");
        let root = split_dir.join(&shard_name);
        fs::create_dir_all(&root)?;
        Ok(Self {
            split,
            shard_name,
            root,
            game_count,
            sample_count,
        })
    }

    pub(crate) fn write_json<T: serde::Serialize>(
        &self,
        file_name: &str,
        value: &T,
    ) -> io::Result<()> {
        let path = self.root.join(file_name);
        let json = serde_json::to_vec_pretty(value)
            .map_err(|err| io::Error::other(format!("failed to serialize {file_name}: {err}")))?;
        fs::write(path, json)
    }

    pub(crate) fn write_text(&self, file_name: &str, content: &str) -> io::Result<()> {
        fs::write(self.root.join(file_name), content)
    }

    pub(crate) fn write_metadata(&self) -> io::Result<()> {
        let shard = ShardMetadata {
            schema_version: SCHEMA_VERSION.to_string(),
            split: self.split.clone(),
            shard_name: self.shard_name.clone(),
            game_count: self.game_count,
            sample_count: self.sample_count,
        };
        self.write_json("shard.json", &shard)
    }

    pub(crate) fn write_games(&self, games: &[ExportGame]) -> io::Result<()> {
        let total_samples = games.iter().map(|game| game.samples.len()).sum::<usize>();
        if total_samples != self.sample_count {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "shard sample_count mismatch: expected {}, got {}",
                    self.sample_count, total_samples
                ),
            ));
        }

        let identities = games
            .iter()
            .map(|game| game.identity.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        self.write_text("game_identities.txt", &identities)?;

        let mut offsets = Vec::with_capacity(games.len() + 1);
        offsets.push(0i64);
        let mut running = 0i64;
        for game in games {
            running += game.samples.len() as i64;
            offsets.push(running);
        }
        write_npy_i64_1d(&self.root.join("game_sample_offsets.npy"), &offsets)?;

        let mut obs = Vec::with_capacity(self.sample_count * OBS_SIZE);
        let mut action = Vec::with_capacity(self.sample_count);
        let mut legal_mask = Vec::with_capacity(self.sample_count * HYDRA_ACTION_SPACE);
        let mut score_delta = Vec::with_capacity(self.sample_count);
        let mut value_target = Vec::with_capacity(self.sample_count);
        let mut grp_target = Vec::with_capacity(self.sample_count * 24);
        let mut tenpai_target = Vec::with_capacity(self.sample_count * 3);
        let mut danger_target = Vec::with_capacity(self.sample_count * 3 * 34);
        let mut danger_mask = Vec::with_capacity(self.sample_count * 3 * 34);
        let mut opp_next_target = Vec::with_capacity(self.sample_count * 3 * 34);
        let mut score_pdf_target = Vec::with_capacity(self.sample_count * SCORE_BINS);
        let mut score_cdf_target = Vec::with_capacity(self.sample_count * SCORE_BINS);
        let mut oracle_target = Vec::with_capacity(self.sample_count * 4);
        let mut oracle_present = Vec::with_capacity(self.sample_count);
        let mut safety_residual_target = Vec::with_capacity(self.sample_count * HYDRA_ACTION_SPACE);
        let mut safety_residual_present = Vec::with_capacity(self.sample_count);
        let mut safety_residual_mask = Vec::with_capacity(self.sample_count * HYDRA_ACTION_SPACE);
        let mut belief_fields_target = Vec::with_capacity(self.sample_count * BELIEF_FIELD_SIZE);
        let mut belief_fields_present = Vec::with_capacity(self.sample_count);
        let mut mixture_weight_target = Vec::with_capacity(self.sample_count * 4);
        let mut mixture_weight_present = Vec::with_capacity(self.sample_count);

        for game in games {
            for sample in &game.samples {
                obs.extend_from_slice(&sample.obs);
                action.push(sample.action);
                legal_mask.extend_from_slice(&sample.legal_mask);
                score_delta.push(sample.score_delta);
                value_target.push(score_delta_to_value(sample.score_delta));

                let mut grp = [0.0f32; 24];
                if (sample.grp_label as usize) < grp.len() {
                    grp[sample.grp_label as usize] = 1.0;
                }
                grp_target.extend_from_slice(&grp);

                tenpai_target.extend_from_slice(&sample.tenpai);
                danger_target.extend_from_slice(&sample.danger);
                danger_mask.extend_from_slice(&sample.danger_mask);

                let mut opp = [0.0f32; 3 * 34];
                for (idx, tile) in sample.opp_next.iter().copied().enumerate() {
                    if tile < 34 {
                        opp[idx * 34 + tile as usize] = 1.0;
                    }
                }
                opp_next_target.extend_from_slice(&opp);

                score_pdf_target.extend_from_slice(&score_delta_to_pdf(sample.score_delta));
                score_cdf_target.extend_from_slice(&score_delta_to_cdf(sample.score_delta));

                if let Some(target) = sample.oracle_target {
                    oracle_target.extend_from_slice(&target);
                    oracle_present.push(1);
                } else {
                    oracle_target.extend_from_slice(&[0.0; 4]);
                    oracle_present.push(0);
                }

                if let Some(target) = sample.safety_residual {
                    safety_residual_target.extend_from_slice(&target);
                    safety_residual_present.push(1);
                } else {
                    safety_residual_target.extend_from_slice(&[0.0; HYDRA_ACTION_SPACE]);
                    safety_residual_present.push(0);
                }

                if let Some(mask) = sample.safety_residual_mask {
                    safety_residual_mask.extend_from_slice(&mask);
                } else {
                    safety_residual_mask.extend_from_slice(&[0.0; HYDRA_ACTION_SPACE]);
                }

                if let Some(target) = sample.belief_fields {
                    belief_fields_target.extend_from_slice(&target);
                } else {
                    belief_fields_target.extend_from_slice(&[0.0; BELIEF_FIELD_SIZE]);
                }
                belief_fields_present.push(u8::from(sample.belief_fields_present));

                if let Some(target) = sample.mixture_weights {
                    mixture_weight_target.extend_from_slice(&target);
                } else {
                    mixture_weight_target.extend_from_slice(&[0.0; 4]);
                }
                mixture_weight_present.push(u8::from(sample.mixture_weights_present));
            }
        }

        write_npy_f32_3d(
            &self.root.join("obs.npy"),
            &obs,
            [self.sample_count, 192, 34],
        )?;
        write_npy_u8_1d(&self.root.join("action.npy"), &action)?;
        write_npy_f32_2d(
            &self.root.join("legal_mask.npy"),
            &legal_mask,
            [self.sample_count, HYDRA_ACTION_SPACE],
        )?;
        write_npy_i32_1d(&self.root.join("score_delta.npy"), &score_delta)?;
        write_npy_f32_1d(&self.root.join("value_target.npy"), &value_target)?;
        write_npy_f32_2d(
            &self.root.join("grp_target.npy"),
            &grp_target,
            [self.sample_count, 24],
        )?;
        write_npy_f32_2d(
            &self.root.join("tenpai_target.npy"),
            &tenpai_target,
            [self.sample_count, 3],
        )?;
        write_npy_f32_3d(
            &self.root.join("danger_target.npy"),
            &danger_target,
            [self.sample_count, 3, 34],
        )?;
        write_npy_f32_3d(
            &self.root.join("danger_mask.npy"),
            &danger_mask,
            [self.sample_count, 3, 34],
        )?;
        write_npy_f32_3d(
            &self.root.join("opp_next_target.npy"),
            &opp_next_target,
            [self.sample_count, 3, 34],
        )?;
        write_npy_f32_2d(
            &self.root.join("score_pdf_target.npy"),
            &score_pdf_target,
            [self.sample_count, SCORE_BINS],
        )?;
        write_npy_f32_2d(
            &self.root.join("score_cdf_target.npy"),
            &score_cdf_target,
            [self.sample_count, SCORE_BINS],
        )?;
        write_npy_f32_2d(
            &self.root.join("oracle_target.npy"),
            &oracle_target,
            [self.sample_count, 4],
        )?;
        write_npy_u8_1d(
            &self.root.join("oracle_target_present.npy"),
            &oracle_present,
        )?;
        write_npy_f32_2d(
            &self.root.join("safety_residual_target.npy"),
            &safety_residual_target,
            [self.sample_count, HYDRA_ACTION_SPACE],
        )?;
        write_npy_u8_1d(
            &self.root.join("safety_residual_present.npy"),
            &safety_residual_present,
        )?;
        write_npy_f32_2d(
            &self.root.join("safety_residual_mask.npy"),
            &safety_residual_mask,
            [self.sample_count, HYDRA_ACTION_SPACE],
        )?;
        write_npy_f32_3d(
            &self.root.join("belief_fields_target.npy"),
            &belief_fields_target,
            [self.sample_count, 16, 34],
        )?;
        write_npy_u8_1d(
            &self.root.join("belief_fields_present.npy"),
            &belief_fields_present,
        )?;
        write_npy_f32_2d(
            &self.root.join("mixture_weight_target.npy"),
            &mixture_weight_target,
            [self.sample_count, 4],
        )?;
        write_npy_u8_1d(
            &self.root.join("mixture_weight_present.npy"),
            &mixture_weight_present,
        )?;

        Ok(())
    }

    pub(crate) fn hash_files(&self) -> io::Result<ShardArtifactHashes> {
        let mut entries = fs::read_dir(&self.root)?.collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        let hashes = entries
            .par_iter()
            .filter_map(|entry| {
                let path = entry.path();
                path.is_file().then_some(path)
            })
            .map(|path| -> io::Result<(String, String)> {
                let file = fs::File::open(&path)?;
                let mut reader = BufReader::with_capacity(1 << 20, file);
                let mut hasher = Sha256::new();
                io::copy(&mut reader, &mut hasher)?;
                let digest = format!("{:x}", hasher.finalize());
                let name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "invalid shard file name")
                    })?
                    .to_string();
                Ok((name, digest))
            })
            .collect::<io::Result<BTreeMap<_, _>>>()?;
        Ok(ShardArtifactHashes { files: hashes })
    }
}

fn write_magic<W: Write>(writer: &mut W) -> io::Result<()> {
    writer.write_all(b"\x93NUMPY")?;
    writer.write_all(&[1, 0])
}

fn build_header(descr: &str, fortran_order: bool, shape: &[usize]) -> Vec<u8> {
    let shape_text = match shape {
        [single] => format!("({},)", single),
        _ => {
            let dims = shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("({dims},)").replace(",)", ")")
        }
    };
    let mut header = format!(
        "{{'descr': '{}', 'fortran_order': {}, 'shape': {}, }}",
        descr,
        if fortran_order { "True" } else { "False" },
        shape_text
    )
    .into_bytes();
    let preamble_len = 10usize;
    let total_without_padding = preamble_len + 2 + header.len() + 1;
    let padding = (16 - (total_without_padding % 16)) % 16;
    header.extend(std::iter::repeat_n(b' ', padding));
    header.push(b'\n');
    header
}

fn write_npy_bytes(path: &Path, descr: &str, shape: &[usize], data: &[u8]) -> io::Result<()> {
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    write_magic(&mut writer)?;
    let header = build_header(descr, false, shape);
    let header_len = u16::try_from(header.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "npy header too large"))?;
    writer.write_all(&header_len.to_le_bytes())?;
    writer.write_all(&header)?;
    writer.write_all(data)?;
    writer.flush()
}

#[cfg(target_endian = "little")]
fn f32_as_le_bytes(values: &[f32]) -> &[u8] {
    let ptr = values.as_ptr().cast::<u8>();
    let len = std::mem::size_of_val(values);
    // SAFETY: f32 has no padding, the source slice remains alive for the returned borrow,
    // and reading through u8 removes alignment concerns. On little-endian targets the
    // in-memory representation already matches the .npy <f4 byte order.
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

#[cfg(not(target_endian = "little"))]
fn f32_as_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

#[cfg(target_endian = "little")]
fn i32_as_le_bytes(values: &[i32]) -> &[u8] {
    let ptr = values.as_ptr().cast::<u8>();
    let len = std::mem::size_of_val(values);
    // SAFETY: i32 has no padding, the source slice outlives the returned borrow, and
    // u8 reads are alignment-agnostic. Little-endian memory already matches .npy <i4.
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

#[cfg(not(target_endian = "little"))]
fn i32_as_le_bytes(values: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

#[cfg(target_endian = "little")]
fn i64_as_le_bytes(values: &[i64]) -> &[u8] {
    let ptr = values.as_ptr().cast::<u8>();
    let len = std::mem::size_of_val(values);
    // SAFETY: i64 has no padding, the source slice outlives the returned borrow, and
    // u8 reads are alignment-agnostic. Little-endian memory already matches .npy <i8.
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

#[cfg(not(target_endian = "little"))]
fn i64_as_le_bytes(values: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

fn write_npy_f32_1d(path: &Path, values: &[f32]) -> io::Result<()> {
    write_npy_bytes(path, "<f4", &[values.len()], f32_as_le_bytes(values))
}

fn write_npy_f32_2d(path: &Path, values: &[f32], shape: [usize; 2]) -> io::Result<()> {
    write_npy_bytes(path, "<f4", &shape, f32_as_le_bytes(values))
}

fn write_npy_f32_3d(path: &Path, values: &[f32], shape: [usize; 3]) -> io::Result<()> {
    write_npy_bytes(path, "<f4", &shape, f32_as_le_bytes(values))
}

fn write_npy_u8_1d(path: &Path, values: &[u8]) -> io::Result<()> {
    write_npy_bytes(path, "|u1", &[values.len()], values)
}

fn write_npy_i32_1d(path: &Path, values: &[i32]) -> io::Result<()> {
    write_npy_bytes(path, "<i4", &[values.len()], i32_as_le_bytes(values))
}

fn write_npy_i64_1d(path: &Path, values: &[i64]) -> io::Result<()> {
    write_npy_bytes(path, "<i8", &[values.len()], i64_as_le_bytes(values))
}

#[cfg(test)]
mod tests {
    use super::*;

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
            oracle_target: Some([0.1, 0.2, 0.3, 0.4]),
            tenpai: [0.0, 1.0, 0.0],
            opp_next: [1, 255, 2],
            danger: [0.0; 102],
            danger_mask: [1.0; 102],
            safety_residual: Some([0.0; HYDRA_ACTION_SPACE]),
            safety_residual_mask: Some([1.0; HYDRA_ACTION_SPACE]),
            belief_fields: Some([0.0; BELIEF_FIELD_SIZE]),
            mixture_weights: Some([0.7, 0.3, 0.0, 0.0]),
            belief_fields_present: true,
            mixture_weights_present: true,
        }
    }

    #[test]
    fn writes_real_payload_files() {
        let temp =
            std::env::temp_dir().join(format!("hydra_phase0_writer_{}_{}", std::process::id(), 17));
        if temp.exists() {
            fs::remove_dir_all(&temp).ok();
        }
        fs::create_dir_all(&temp).expect("create temp");
        let shard = PendingShard::new(&temp, ExportSplit::Train, 0, 1, 2).expect("new shard");
        shard.write_metadata().expect("write metadata");
        shard
            .write_games(&[ExportGame {
                identity: "game-00000000".to_string(),
                samples: vec![dummy_sample(3, 1000), dummy_sample(4, -2000)],
            }])
            .expect("write games");

        let obs = fs::metadata(shard.root.join("obs.npy")).expect("obs.npy metadata");
        let offsets =
            fs::metadata(shard.root.join("game_sample_offsets.npy")).expect("offset metadata");
        assert!(obs.len() > 0);
        assert!(offsets.len() > 0);

        fs::remove_dir_all(temp).ok();
    }
}
