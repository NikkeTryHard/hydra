use std::collections::BTreeMap;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use hydra_core::action::HYDRA_ACTION_SPACE;
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

pub(crate) type FileHashes = BTreeMap<String, String>;

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
    ) -> io::Result<(String, String)> {
        let json = serde_json::to_vec_pretty(value)
            .map_err(|err| io::Error::other(format!("failed to serialize {file_name}: {err}")))?;
        write_tracked_file(&self.root.join(file_name), &json)
    }

    pub(crate) fn write_text(
        &self,
        file_name: &str,
        content: &str,
    ) -> io::Result<(String, String)> {
        write_tracked_file(&self.root.join(file_name), content.as_bytes())
    }

    pub(crate) fn write_metadata(&self) -> io::Result<(String, String)> {
        let shard = ShardMetadata {
            schema_version: SCHEMA_VERSION.to_string(),
            split: self.split.clone(),
            shard_name: self.shard_name.clone(),
            game_count: self.game_count,
            sample_count: self.sample_count,
        };
        self.write_json("shard.json", &shard)
    }

    pub(crate) fn write_games(&self, games: &[ExportGame]) -> io::Result<FileHashes> {
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
        let mut hashes = FileHashes::new();
        let (name, digest) = self.write_text("game_identities.txt", &identities)?;
        hashes.insert(name, digest);

        let mut offsets = Vec::with_capacity(games.len() + 1);
        offsets.push(0i64);
        let mut running = 0i64;
        for game in games {
            running += game.samples.len() as i64;
            offsets.push(running);
        }
        let (name, digest) =
            write_npy_i64_1d(&self.root.join("game_sample_offsets.npy"), &offsets)?;
        hashes.insert(name, digest);

        let mut writer = ShardTensorWriters::new(&self.root, self.sample_count)?;
        for game in games {
            for sample in &game.samples {
                writer.write_sample(sample)?;
            }
        }
        hashes.extend(writer.finish()?);

        Ok(hashes)
    }

    pub(crate) fn hash_files(&self, files: FileHashes) -> ShardArtifactHashes {
        ShardArtifactHashes { files }
    }
}

struct NpyWriter {
    writer: BufWriter<fs::File>,
    hasher: Sha256,
    file_name: String,
}

impl NpyWriter {
    fn new(path: &Path, descr: &str, shape: &[usize]) -> io::Result<Self> {
        let file = fs::File::create(path)?;
        let mut writer = BufWriter::new(file);
        write_magic(&mut writer)?;
        let header = build_header(descr, false, shape);
        let header_len = u16::try_from(header.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "npy header too large"))?;
        writer.write_all(&header_len.to_le_bytes())?;
        writer.write_all(&header)?;
        let mut hasher = Sha256::new();
        hasher.update(b"\x93NUMPY");
        hasher.update([1, 0]);
        hasher.update(header_len.to_le_bytes());
        hasher.update(&header);
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid npy file name"))?
            .to_string();
        Ok(Self {
            writer,
            hasher,
            file_name,
        })
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.hasher.update(bytes);
        self.writer.write_all(bytes)
    }

    fn finish(mut self) -> io::Result<(String, String)> {
        self.writer.flush()?;
        Ok((self.file_name, format!("{:x}", self.hasher.finalize())))
    }
}

struct ShardTensorWriters {
    obs: NpyWriter,
    action: NpyWriter,
    legal_mask: NpyWriter,
    score_delta: NpyWriter,
    value_target: NpyWriter,
    grp_target: NpyWriter,
    tenpai_target: NpyWriter,
    danger_target: NpyWriter,
    danger_mask: NpyWriter,
    opp_next_target: NpyWriter,
    score_pdf_target: NpyWriter,
    score_cdf_target: NpyWriter,
    oracle_target: NpyWriter,
    oracle_present: NpyWriter,
    safety_residual_target: NpyWriter,
    safety_residual_present: NpyWriter,
    safety_residual_mask: NpyWriter,
    belief_fields_target: NpyWriter,
    belief_fields_present: NpyWriter,
    mixture_weight_target: NpyWriter,
    mixture_weight_present: NpyWriter,
}

impl ShardTensorWriters {
    fn new(root: &Path, sample_count: usize) -> io::Result<Self> {
        Ok(Self {
            obs: NpyWriter::new(&root.join("obs.npy"), "<f4", &[sample_count, 192, 34])?,
            action: NpyWriter::new(&root.join("action.npy"), "|u1", &[sample_count])?,
            legal_mask: NpyWriter::new(
                &root.join("legal_mask.npy"),
                "<f4",
                &[sample_count, HYDRA_ACTION_SPACE],
            )?,
            score_delta: NpyWriter::new(&root.join("score_delta.npy"), "<i4", &[sample_count])?,
            value_target: NpyWriter::new(&root.join("value_target.npy"), "<f4", &[sample_count])?,
            grp_target: NpyWriter::new(&root.join("grp_target.npy"), "<f4", &[sample_count, 24])?,
            tenpai_target: NpyWriter::new(
                &root.join("tenpai_target.npy"),
                "<f4",
                &[sample_count, 3],
            )?,
            danger_target: NpyWriter::new(
                &root.join("danger_target.npy"),
                "<f4",
                &[sample_count, 3, 34],
            )?,
            danger_mask: NpyWriter::new(
                &root.join("danger_mask.npy"),
                "<f4",
                &[sample_count, 3, 34],
            )?,
            opp_next_target: NpyWriter::new(
                &root.join("opp_next_target.npy"),
                "<f4",
                &[sample_count, 3, 34],
            )?,
            score_pdf_target: NpyWriter::new(
                &root.join("score_pdf_target.npy"),
                "<f4",
                &[sample_count, SCORE_BINS],
            )?,
            score_cdf_target: NpyWriter::new(
                &root.join("score_cdf_target.npy"),
                "<f4",
                &[sample_count, SCORE_BINS],
            )?,
            oracle_target: NpyWriter::new(
                &root.join("oracle_target.npy"),
                "<f4",
                &[sample_count, 4],
            )?,
            oracle_present: NpyWriter::new(
                &root.join("oracle_target_present.npy"),
                "|u1",
                &[sample_count],
            )?,
            safety_residual_target: NpyWriter::new(
                &root.join("safety_residual_target.npy"),
                "<f4",
                &[sample_count, HYDRA_ACTION_SPACE],
            )?,
            safety_residual_present: NpyWriter::new(
                &root.join("safety_residual_present.npy"),
                "|u1",
                &[sample_count],
            )?,
            safety_residual_mask: NpyWriter::new(
                &root.join("safety_residual_mask.npy"),
                "<f4",
                &[sample_count, HYDRA_ACTION_SPACE],
            )?,
            belief_fields_target: NpyWriter::new(
                &root.join("belief_fields_target.npy"),
                "<f4",
                &[sample_count, 16, 34],
            )?,
            belief_fields_present: NpyWriter::new(
                &root.join("belief_fields_present.npy"),
                "|u1",
                &[sample_count],
            )?,
            mixture_weight_target: NpyWriter::new(
                &root.join("mixture_weight_target.npy"),
                "<f4",
                &[sample_count, 4],
            )?,
            mixture_weight_present: NpyWriter::new(
                &root.join("mixture_weight_present.npy"),
                "|u1",
                &[sample_count],
            )?,
        })
    }

    fn write_sample(&mut self, sample: &MjaiSample) -> io::Result<()> {
        self.obs.write_bytes(f32_as_le_bytes(&sample.obs))?;
        self.action.write_bytes(&[sample.action])?;
        self.legal_mask
            .write_bytes(f32_as_le_bytes(&sample.legal_mask))?;
        self.score_delta
            .write_bytes(i32_as_le_bytes(std::slice::from_ref(&sample.score_delta)))?;

        let value_target = [score_delta_to_value(sample.score_delta)];
        self.value_target
            .write_bytes(f32_as_le_bytes(&value_target))?;

        let mut grp = [0.0f32; 24];
        if (sample.grp_label as usize) < grp.len() {
            grp[sample.grp_label as usize] = 1.0;
        }
        self.grp_target.write_bytes(f32_as_le_bytes(&grp))?;

        self.tenpai_target
            .write_bytes(f32_as_le_bytes(&sample.tenpai))?;
        self.danger_target
            .write_bytes(f32_as_le_bytes(&sample.danger))?;
        self.danger_mask
            .write_bytes(f32_as_le_bytes(&sample.danger_mask))?;

        let mut opp = [0.0f32; 3 * 34];
        for (idx, tile) in sample.opp_next.iter().copied().enumerate() {
            if tile < 34 {
                opp[idx * 34 + tile as usize] = 1.0;
            }
        }
        self.opp_next_target.write_bytes(f32_as_le_bytes(&opp))?;

        let score_pdf = score_delta_to_pdf(sample.score_delta);
        self.score_pdf_target
            .write_bytes(f32_as_le_bytes(&score_pdf))?;
        let score_cdf = score_delta_to_cdf(sample.score_delta);
        self.score_cdf_target
            .write_bytes(f32_as_le_bytes(&score_cdf))?;

        if let Some(target) = sample.oracle_target {
            self.oracle_target.write_bytes(f32_as_le_bytes(&target))?;
            self.oracle_present.write_bytes(&[1])?;
        } else {
            self.oracle_target.write_bytes(f32_as_le_bytes(&[0.0; 4]))?;
            self.oracle_present.write_bytes(&[0])?;
        }

        if let Some(target) = sample.safety_residual {
            self.safety_residual_target
                .write_bytes(f32_as_le_bytes(&target))?;
            self.safety_residual_present.write_bytes(&[1])?;
        } else {
            self.safety_residual_target
                .write_bytes(f32_as_le_bytes(&[0.0; HYDRA_ACTION_SPACE]))?;
            self.safety_residual_present.write_bytes(&[0])?;
        }

        if let Some(mask) = sample.safety_residual_mask {
            self.safety_residual_mask
                .write_bytes(f32_as_le_bytes(&mask))?;
        } else {
            self.safety_residual_mask
                .write_bytes(f32_as_le_bytes(&[0.0; HYDRA_ACTION_SPACE]))?;
        }

        if let Some(target) = sample.belief_fields {
            self.belief_fields_target
                .write_bytes(f32_as_le_bytes(&target))?;
        } else {
            self.belief_fields_target
                .write_bytes(f32_as_le_bytes(&[0.0; BELIEF_FIELD_SIZE]))?;
        }
        self.belief_fields_present
            .write_bytes(&[u8::from(sample.belief_fields_present)])?;

        if let Some(target) = sample.mixture_weights {
            self.mixture_weight_target
                .write_bytes(f32_as_le_bytes(&target))?;
        } else {
            self.mixture_weight_target
                .write_bytes(f32_as_le_bytes(&[0.0; 4]))?;
        }
        self.mixture_weight_present
            .write_bytes(&[u8::from(sample.mixture_weights_present)])?;

        Ok(())
    }

    fn finish(self) -> io::Result<FileHashes> {
        let mut hashes = FileHashes::new();
        let mut insert = |entry: (String, String)| {
            hashes.insert(entry.0, entry.1);
        };
        insert(self.obs.finish()?);
        insert(self.action.finish()?);
        insert(self.legal_mask.finish()?);
        insert(self.score_delta.finish()?);
        insert(self.value_target.finish()?);
        insert(self.grp_target.finish()?);
        insert(self.tenpai_target.finish()?);
        insert(self.danger_target.finish()?);
        insert(self.danger_mask.finish()?);
        insert(self.opp_next_target.finish()?);
        insert(self.score_pdf_target.finish()?);
        insert(self.score_cdf_target.finish()?);
        insert(self.oracle_target.finish()?);
        insert(self.oracle_present.finish()?);
        insert(self.safety_residual_target.finish()?);
        insert(self.safety_residual_present.finish()?);
        insert(self.safety_residual_mask.finish()?);
        insert(self.belief_fields_target.finish()?);
        insert(self.belief_fields_present.finish()?);
        insert(self.mixture_weight_target.finish()?);
        insert(self.mixture_weight_present.finish()?);
        Ok(hashes)
    }
}

fn write_magic<W: Write>(writer: &mut W) -> io::Result<()> {
    writer.write_all(b"\x93NUMPY")?;
    writer.write_all(&[1, 0])
}

fn write_tracked_file(path: &Path, bytes: &[u8]) -> io::Result<(String, String)> {
    fs::write(path, bytes)?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid tracked file name"))?
        .to_string();
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok((name, format!("{:x}", hasher.finalize())))
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

fn write_npy_bytes(
    path: &Path,
    descr: &str,
    shape: &[usize],
    data: &[u8],
) -> io::Result<(String, String)> {
    let file = fs::File::create(path)?;
    let mut writer = BufWriter::new(file);
    write_magic(&mut writer)?;
    let header = build_header(descr, false, shape);
    let header_len = u16::try_from(header.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "npy header too large"))?;
    writer.write_all(&header_len.to_le_bytes())?;
    writer.write_all(&header)?;
    writer.write_all(data)?;
    writer.flush()?;

    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid npy file name"))?
        .to_string();
    let mut hasher = Sha256::new();
    hasher.update(b"\x93NUMPY");
    hasher.update([1, 0]);
    hasher.update(header_len.to_le_bytes());
    hasher.update(&header);
    hasher.update(data);
    Ok((name, format!("{:x}", hasher.finalize())))
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

fn write_npy_i64_1d(path: &Path, values: &[i64]) -> io::Result<(String, String)> {
    write_npy_bytes(path, "<i8", &[values.len()], i64_as_le_bytes(values))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hydra_core::encoder::OBS_SIZE;

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
