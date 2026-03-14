use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Seek};
use std::path::Path;

use burn::prelude::*;
use burn::tensor::{DType, TensorData};
use burn_store::{ModuleSnapshot, TensorSnapshot};
use serde::Deserialize;
use serde::Serialize;

use crate::model::{HydraModel, HydraModelConfig};

#[derive(Debug, Deserialize)]
struct ExportMetadata {
    schema_version: String,
    tensors: BTreeMap<String, TensorMetadata>,
}

#[derive(Debug, Deserialize)]
struct TensorMetadata {
    archive_key: String,
    shape: Vec<usize>,
    dtype: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExpectedTensorSpec {
    pub name: String,
    pub archive_key: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

pub fn expected_phase0_weight_shapes() -> BTreeMap<String, Vec<usize>> {
    let cfg = HydraModelConfig::learner();
    let h = cfg.hidden_channels;
    let b = cfg.se_bottleneck;
    let mut out = BTreeMap::new();

    out.insert(
        "SEResNet_0/Conv_0/kernel".to_string(),
        vec![3, cfg.input_channels, h],
    );
    out.insert("SEResNet_0/Conv_0/bias".to_string(), vec![h]);
    out.insert("SEResNet_0/GroupNorm_0/scale".to_string(), vec![h]);
    out.insert("SEResNet_0/GroupNorm_0/bias".to_string(), vec![h]);
    for i in 0..cfg.num_blocks {
        out.insert(
            format!("SEResNet_0/SEResBlock_{i}/GroupNorm_0/scale"),
            vec![h],
        );
        out.insert(
            format!("SEResNet_0/SEResBlock_{i}/GroupNorm_0/bias"),
            vec![h],
        );
        out.insert(
            format!("SEResNet_0/SEResBlock_{i}/Conv_0/kernel"),
            vec![3, h, h],
        );
        out.insert(format!("SEResNet_0/SEResBlock_{i}/Conv_0/bias"), vec![h]);
        out.insert(
            format!("SEResNet_0/SEResBlock_{i}/GroupNorm_1/scale"),
            vec![h],
        );
        out.insert(
            format!("SEResNet_0/SEResBlock_{i}/GroupNorm_1/bias"),
            vec![h],
        );
        out.insert(
            format!("SEResNet_0/SEResBlock_{i}/Conv_1/kernel"),
            vec![3, h, h],
        );
        out.insert(format!("SEResNet_0/SEResBlock_{i}/Conv_1/bias"), vec![h]);
        out.insert(
            format!("SEResNet_0/SEResBlock_{i}/SEBlock_0/Dense_0/kernel"),
            vec![h, b],
        );
        out.insert(
            format!("SEResNet_0/SEResBlock_{i}/SEBlock_0/Dense_0/bias"),
            vec![b],
        );
        out.insert(
            format!("SEResNet_0/SEResBlock_{i}/SEBlock_0/Dense_1/kernel"),
            vec![b, h],
        );
        out.insert(
            format!("SEResNet_0/SEResBlock_{i}/SEBlock_0/Dense_1/bias"),
            vec![h],
        );
    }
    out.insert("SEResNet_0/GroupNorm_1/scale".to_string(), vec![h]);
    out.insert("SEResNet_0/GroupNorm_1/bias".to_string(), vec![h]);

    out.insert("Dense_0/kernel".to_string(), vec![h, cfg.action_space]);
    out.insert("Dense_0/bias".to_string(), vec![cfg.action_space]);
    out.insert("Dense_1/kernel".to_string(), vec![h, 1]);
    out.insert("Dense_1/bias".to_string(), vec![1]);
    out.insert("Dense_2/kernel".to_string(), vec![h, cfg.score_bins]);
    out.insert("Dense_2/bias".to_string(), vec![cfg.score_bins]);
    out.insert("Dense_3/kernel".to_string(), vec![h, cfg.score_bins]);
    out.insert("Dense_3/bias".to_string(), vec![cfg.score_bins]);
    out.insert("Dense_4/kernel".to_string(), vec![h, cfg.num_opponents]);
    out.insert("Dense_4/bias".to_string(), vec![cfg.num_opponents]);
    out.insert("Dense_5/kernel".to_string(), vec![h, cfg.grp_classes]);
    out.insert("Dense_5/bias".to_string(), vec![cfg.grp_classes]);
    out.insert("Conv_0/kernel".to_string(), vec![1, h, cfg.num_opponents]);
    out.insert("Conv_0/bias".to_string(), vec![cfg.num_opponents]);
    out.insert("Conv_1/kernel".to_string(), vec![1, h, cfg.num_opponents]);
    out.insert("Conv_1/bias".to_string(), vec![cfg.num_opponents]);
    out.insert("Dense_6/kernel".to_string(), vec![h, 4]);
    out.insert("Dense_6/bias".to_string(), vec![4]);
    out.insert(
        "Conv_2/kernel".to_string(),
        vec![1, h, cfg.num_belief_components * 4],
    );
    out.insert(
        "Conv_2/bias".to_string(),
        vec![cfg.num_belief_components * 4],
    );
    out.insert(
        "Dense_7/kernel".to_string(),
        vec![h, cfg.num_belief_components],
    );
    out.insert("Dense_7/bias".to_string(), vec![cfg.num_belief_components]);
    out.insert(
        "Dense_8/kernel".to_string(),
        vec![h, cfg.num_opponents * cfg.opponent_hand_type_classes],
    );
    out.insert(
        "Dense_8/bias".to_string(),
        vec![cfg.num_opponents * cfg.opponent_hand_type_classes],
    );
    out.insert("Dense_9/kernel".to_string(), vec![h, cfg.action_space]);
    out.insert("Dense_9/bias".to_string(), vec![cfg.action_space]);
    out.insert("Dense_10/kernel".to_string(), vec![h, cfg.action_space]);
    out.insert("Dense_10/bias".to_string(), vec![cfg.action_space]);
    out
}

pub fn expected_phase0_weight_specs() -> Vec<ExpectedTensorSpec> {
    expected_phase0_weight_shapes()
        .into_iter()
        .map(|(name, shape)| ExpectedTensorSpec {
            archive_key: name.replace('/', "__"),
            name,
            shape,
            dtype: "float32".to_string(),
        })
        .collect()
}

fn read_export_metadata(metadata_path: &Path) -> Result<ExportMetadata, String> {
    let raw = fs::read_to_string(metadata_path).map_err(|err| {
        format!(
            "failed to read weight metadata {}: {err}",
            metadata_path.display()
        )
    })?;
    serde_json::from_str(&raw).map_err(|err| {
        format!(
            "failed to parse weight metadata {}: {err}",
            metadata_path.display()
        )
    })
}

fn export_tensor_to_burn_path(name: &str) -> Result<String, String> {
    match name {
        "SEResNet_0/Conv_0/kernel" => return Ok("backbone.input_conv.weight".to_string()),
        "SEResNet_0/Conv_0/bias" => return Ok("backbone.input_conv.bias".to_string()),
        "SEResNet_0/GroupNorm_0/scale" => return Ok("backbone.input_gn.gamma".to_string()),
        "SEResNet_0/GroupNorm_0/bias" => return Ok("backbone.input_gn.beta".to_string()),
        "SEResNet_0/GroupNorm_1/scale" => return Ok("backbone.final_gn.gamma".to_string()),
        "SEResNet_0/GroupNorm_1/bias" => return Ok("backbone.final_gn.beta".to_string()),
        "Dense_0/kernel" => return Ok("policy.linear.weight".to_string()),
        "Dense_0/bias" => return Ok("policy.linear.bias".to_string()),
        "Dense_1/kernel" => return Ok("value.linear.weight".to_string()),
        "Dense_1/bias" => return Ok("value.linear.bias".to_string()),
        "Dense_2/kernel" => return Ok("score_pdf.linear.weight".to_string()),
        "Dense_2/bias" => return Ok("score_pdf.linear.bias".to_string()),
        "Dense_3/kernel" => return Ok("score_cdf.linear.weight".to_string()),
        "Dense_3/bias" => return Ok("score_cdf.linear.bias".to_string()),
        "Dense_4/kernel" => return Ok("opp_tenpai.linear.weight".to_string()),
        "Dense_4/bias" => return Ok("opp_tenpai.linear.bias".to_string()),
        "Dense_5/kernel" => return Ok("grp.linear.weight".to_string()),
        "Dense_5/bias" => return Ok("grp.linear.bias".to_string()),
        "Conv_0/kernel" => return Ok("opp_next_discard.conv.weight".to_string()),
        "Conv_0/bias" => return Ok("opp_next_discard.conv.bias".to_string()),
        "Conv_1/kernel" => return Ok("danger.conv.weight".to_string()),
        "Conv_1/bias" => return Ok("danger.conv.bias".to_string()),
        "Dense_6/kernel" => return Ok("oracle_critic.linear.weight".to_string()),
        "Dense_6/bias" => return Ok("oracle_critic.linear.bias".to_string()),
        "Conv_2/kernel" => return Ok("belief_field.conv.weight".to_string()),
        "Conv_2/bias" => return Ok("belief_field.conv.bias".to_string()),
        "Dense_7/kernel" => return Ok("mixture_weight.linear.weight".to_string()),
        "Dense_7/bias" => return Ok("mixture_weight.linear.bias".to_string()),
        "Dense_8/kernel" => return Ok("opponent_hand_type.linear.weight".to_string()),
        "Dense_8/bias" => return Ok("opponent_hand_type.linear.bias".to_string()),
        "Dense_9/kernel" => return Ok("delta_q.linear.weight".to_string()),
        "Dense_9/bias" => return Ok("delta_q.linear.bias".to_string()),
        "Dense_10/kernel" => return Ok("safety_residual.linear.weight".to_string()),
        "Dense_10/bias" => return Ok("safety_residual.linear.bias".to_string()),
        _ => {}
    }

    if let Some(rest) = name.strip_prefix("SEResNet_0/SEResBlock_") {
        let (index, suffix) = rest
            .split_once('/')
            .ok_or_else(|| format!("unsupported Phase 0 export tensor name {name}"))?;
        let path = match suffix {
            "GroupNorm_0/scale" => format!("backbone.blocks.{index}.gn1.gamma"),
            "GroupNorm_0/bias" => format!("backbone.blocks.{index}.gn1.beta"),
            "Conv_0/kernel" => format!("backbone.blocks.{index}.conv1.weight"),
            "Conv_0/bias" => format!("backbone.blocks.{index}.conv1.bias"),
            "GroupNorm_1/scale" => format!("backbone.blocks.{index}.gn2.gamma"),
            "GroupNorm_1/bias" => format!("backbone.blocks.{index}.gn2.beta"),
            "Conv_1/kernel" => format!("backbone.blocks.{index}.conv2.weight"),
            "Conv_1/bias" => format!("backbone.blocks.{index}.conv2.bias"),
            "SEBlock_0/Dense_0/kernel" => format!("backbone.blocks.{index}.se.fc1.weight"),
            "SEBlock_0/Dense_0/bias" => format!("backbone.blocks.{index}.se.fc1.bias"),
            "SEBlock_0/Dense_1/kernel" => format!("backbone.blocks.{index}.se.fc2.weight"),
            "SEBlock_0/Dense_1/bias" => format!("backbone.blocks.{index}.se.fc2.bias"),
            _ => return Err(format!("unsupported Phase 0 export tensor name {name}")),
        };
        return Ok(path);
    }

    Err(format!("unsupported Phase 0 export tensor name {name}"))
}

fn parse_npy_header(
    bytes: &[u8],
    tensor_name: &str,
) -> Result<(String, bool, Vec<usize>, usize), String> {
    const MAGIC: &[u8] = b"\x93NUMPY";
    if bytes.len() < 10 || &bytes[..6] != MAGIC {
        return Err(format!("invalid npy header for {tensor_name}"));
    }

    let major = bytes[6];
    let header_len_offset = 8usize;
    let (header_len, data_offset) = match major {
        1 => {
            let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
            (len, 10usize)
        }
        2 | 3 => {
            if bytes.len() < 12 {
                return Err(format!("truncated npy header for {tensor_name}"));
            }
            let len = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
            (len, 12usize)
        }
        _ => return Err(format!("unsupported npy version {major} for {tensor_name}")),
    };
    let header_end = data_offset + header_len;
    if bytes.len() < header_end || bytes.len() <= header_len_offset {
        return Err(format!("truncated npy payload for {tensor_name}"));
    }
    let header = std::str::from_utf8(&bytes[data_offset..header_end])
        .map_err(|err| format!("invalid utf8 npy header for {tensor_name}: {err}"))?;

    let descr_key = "'descr':";
    let descr_start = header
        .find(descr_key)
        .ok_or_else(|| format!("missing descr in npy header for {tensor_name}"))?
        + descr_key.len();
    let descr_tail = &header[descr_start..];
    let quote_start = descr_tail
        .find('\'')
        .ok_or_else(|| format!("malformed descr in npy header for {tensor_name}"))?;
    let descr_tail = &descr_tail[quote_start + 1..];
    let quote_end = descr_tail
        .find('\'')
        .ok_or_else(|| format!("malformed descr in npy header for {tensor_name}"))?;
    let descr = descr_tail[..quote_end].to_string();

    let fortran = if header.contains("'fortran_order': True") {
        true
    } else if header.contains("'fortran_order': False") {
        false
    } else {
        return Err(format!(
            "missing fortran_order in npy header for {tensor_name}"
        ));
    };

    let shape_key = "'shape':";
    let shape_start = header
        .find(shape_key)
        .ok_or_else(|| format!("missing shape in npy header for {tensor_name}"))?
        + shape_key.len();
    let shape_tail = &header[shape_start..];
    let paren_start = shape_tail
        .find('(')
        .ok_or_else(|| format!("malformed shape in npy header for {tensor_name}"))?;
    let shape_tail = &shape_tail[paren_start + 1..];
    let paren_end = shape_tail
        .find(')')
        .ok_or_else(|| format!("malformed shape in npy header for {tensor_name}"))?;
    let shape_body = &shape_tail[..paren_end];
    let shape = if shape_body.trim().is_empty() {
        Vec::new()
    } else {
        shape_body
            .split(',')
            .filter_map(|segment| {
                let trimmed = segment.trim();
                (!trimmed.is_empty()).then_some(trimmed)
            })
            .map(|segment| {
                segment.parse::<usize>().map_err(|err| {
                    format!("invalid shape dimension '{segment}' for {tensor_name}: {err}")
                })
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    Ok((descr, fortran, shape, header_end))
}

fn read_npy_f32(bytes: &[u8], tensor_name: &str) -> Result<TensorData, String> {
    let (descr, fortran_order, shape, data_offset) = parse_npy_header(bytes, tensor_name)?;
    if fortran_order {
        return Err(format!(
            "fortran-order arrays are unsupported for {tensor_name}"
        ));
    }
    if !matches!(descr.as_str(), "<f4" | "=f4" | "|f4") {
        return Err(format!(
            "unsupported npy dtype {descr} for {tensor_name}; expected float32"
        ));
    }

    let raw = &bytes[data_offset..];
    let numel = shape.iter().product::<usize>();
    let expected_len = numel
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| format!("npy payload too large for {tensor_name}"))?;
    if raw.len() != expected_len {
        return Err(format!(
            "invalid npy payload length for {tensor_name}: expected {expected_len} bytes, got {}",
            raw.len()
        ));
    }
    let values = raw
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    Ok(TensorData::new(values, shape))
}

fn transpose_conv_kernel_1d(name: &str, data: TensorData) -> Result<TensorData, String> {
    if data.shape.len() != 3 {
        return Err(format!(
            "shape mismatch for {name}: expected rank-3 conv kernel, got {:?}",
            data.shape
        ));
    }
    let shape = data.shape.clone();
    let kernel = shape[0];
    let channels_in = shape[1];
    let channels_out = shape[2];
    let values = data
        .into_vec::<f32>()
        .map_err(|err| format!("failed to decode {name} as float32 tensor: {err}"))?;
    let mut transposed = vec![0.0f32; values.len()];
    for k in 0..kernel {
        for c_in in 0..channels_in {
            for c_out in 0..channels_out {
                let src = (k * channels_in + c_in) * channels_out + c_out;
                let dst = (c_out * channels_in + c_in) * kernel + k;
                transposed[dst] = values[src];
            }
        }
    }
    Ok(TensorData::new(
        transposed,
        [channels_out, channels_in, kernel],
    ))
}

fn transform_export_tensor(name: &str, data: TensorData) -> Result<TensorData, String> {
    let is_conv_kernel = matches!(
        name,
        "SEResNet_0/Conv_0/kernel" | "Conv_0/kernel" | "Conv_1/kernel" | "Conv_2/kernel"
    ) || (name.starts_with("SEResNet_0/SEResBlock_")
        && name.ends_with("/Conv_0/kernel"))
        || (name.starts_with("SEResNet_0/SEResBlock_") && name.ends_with("/Conv_1/kernel"));

    if is_conv_kernel {
        transpose_conv_kernel_1d(name, data)
    } else {
        Ok(data)
    }
}

fn read_tensor_data_from_archive<R: Read + Seek>(
    archive: &mut zip::ZipArchive<R>,
    archive_key: &str,
    tensor_name: &str,
) -> Result<TensorData, String> {
    let entry_name = format!("{}.npy", archive_key);
    let mut entry = archive.by_name(&entry_name).map_err(|err| {
        format!(
            "missing npz entry {} in weight archive for {tensor_name}: {err}",
            entry_name
        )
    })?;
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes).map_err(|err| {
        format!(
            "failed to read npz entry {} for {tensor_name}: {err}",
            entry_name
        )
    })?;
    read_npy_f32(&bytes, tensor_name)
}

pub fn apply_phase0_weight_export<B: Backend>(
    model: &mut HydraModel<B>,
    metadata_path: &Path,
    archive_path: &Path,
) -> Result<(), String> {
    validate_phase0_weight_archive(metadata_path, archive_path)?;
    let metadata = read_export_metadata(metadata_path)?;
    let expected_specs = expected_phase0_weight_specs();
    let target_snapshots = model
        .collect(None, None, false)
        .into_iter()
        .map(|snapshot| (snapshot.full_path(), snapshot))
        .collect::<BTreeMap<_, _>>();

    let file = fs::File::open(archive_path).map_err(|err| {
        format!(
            "failed to open weight archive {}: {err}",
            archive_path.display()
        )
    })?;
    let mut archive = zip::ZipArchive::new(file).map_err(|err| {
        format!(
            "failed to open npz archive {}: {err}",
            archive_path.display()
        )
    })?;

    let mut imported = Vec::with_capacity(expected_specs.len());
    for spec in expected_specs {
        let actual = metadata
            .tensors
            .get(&spec.name)
            .ok_or_else(|| format!("missing exported tensor metadata for {}", spec.name))?;
        let burn_path = export_tensor_to_burn_path(&spec.name)?;
        let target = target_snapshots.get(&burn_path).ok_or_else(|| {
            format!(
                "missing Burn tensor path {} for exported tensor {}",
                burn_path, spec.name
            )
        })?;
        let raw = read_tensor_data_from_archive(&mut archive, &actual.archive_key, &spec.name)?;
        if raw.shape != spec.shape {
            return Err(format!(
                "shape mismatch for {}: expected {:?}, got {:?}",
                spec.name, spec.shape, raw.shape
            ));
        }
        let transformed = transform_export_tensor(&spec.name, raw)?;
        if transformed.dtype != DType::F32 {
            return Err(format!(
                "dtype mismatch for {}: expected {:?}, got {:?}",
                spec.name,
                DType::F32,
                transformed.dtype
            ));
        }
        if transformed.shape != target.shape {
            return Err(format!(
                "shape mismatch for {}: expected {:?}, got {:?}",
                spec.name, target.shape, transformed.shape
            ));
        }
        if target.dtype != DType::F32 {
            return Err(format!(
                "dtype mismatch for {}: target expects {:?}, got {:?}",
                spec.name, target.dtype, transformed.dtype
            ));
        }
        let tensor_id = target
            .tensor_id
            .ok_or_else(|| format!("missing parameter id for Burn tensor path {burn_path}"))?;
        let path_stack = burn_path
            .split('.')
            .map(|segment| segment.to_string())
            .collect();
        imported.push(TensorSnapshot::from_data(
            transformed,
            path_stack,
            Vec::new(),
            tensor_id,
        ));
    }

    let result = model.apply(imported, None, None, false);
    if !result.errors.is_empty() || !result.missing.is_empty() || !result.unused.is_empty() {
        return Err(format!("failed to apply Phase 0 weights:\n{result}"));
    }
    Ok(())
}

pub fn load_phase0_weight_export<B: Backend>(
    metadata_path: &Path,
    archive_path: &Path,
    device: &B::Device,
) -> Result<HydraModel<B>, String> {
    let mut model = HydraModelConfig::learner().init::<B>(device);
    apply_phase0_weight_export(&mut model, metadata_path, archive_path)?;
    Ok(model)
}

pub fn validate_phase0_weight_export(metadata_path: &Path) -> Result<(), String> {
    let metadata = read_export_metadata(metadata_path)?;
    if metadata.schema_version != "hydra_phase0_weight_export_v1" {
        return Err(format!(
            "unsupported weight export schema_version {}",
            metadata.schema_version
        ));
    }
    let expected = expected_phase0_weight_shapes();
    let mut seen_archive_keys = BTreeMap::new();

    for (name, shape) in &expected {
        let actual = metadata
            .tensors
            .get(name)
            .ok_or_else(|| format!("missing exported tensor metadata for {name}"))?;
        if actual.shape != *shape {
            return Err(format!(
                "shape mismatch for {name}: expected {:?}, got {:?}",
                shape, actual.shape
            ));
        }
        if actual.dtype != "float32" {
            return Err(format!(
                "dtype mismatch for {name}: expected float32, got {}",
                actual.dtype
            ));
        }
        if actual.archive_key.is_empty() {
            return Err(format!("archive_key missing for {name}"));
        }
        if let Some(previous) = seen_archive_keys.insert(actual.archive_key.clone(), name.clone()) {
            return Err(format!(
                "duplicate archive_key {} for tensors {} and {}",
                actual.archive_key, previous, name
            ));
        }
    }
    Ok(())
}

pub fn validate_phase0_weight_archive(
    metadata_path: &Path,
    archive_path: &Path,
) -> Result<(), String> {
    validate_phase0_weight_export(metadata_path)?;
    let metadata = read_export_metadata(metadata_path)?;

    let file = fs::File::open(archive_path).map_err(|err| {
        format!(
            "failed to open weight archive {}: {err}",
            archive_path.display()
        )
    })?;
    let mut archive = zip::ZipArchive::new(file).map_err(|err| {
        format!(
            "failed to open npz archive {}: {err}",
            archive_path.display()
        )
    })?;
    let mut archive_entries = BTreeMap::new();
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|err| format!("failed to read npz entry {index}: {err}"))?;
        let name = entry.name().to_string();
        let mut header = [0u8; 8];
        let _ = entry.read(&mut header).ok();
        archive_entries.insert(name, ());
    }

    for tensor in metadata.tensors.values() {
        let expected_name = format!("{}.npy", tensor.archive_key);
        if !archive_entries.contains_key(&expected_name) {
            return Err(format!(
                "missing npz entry {} in {}",
                expected_name,
                archive_path.display()
            ));
        }
    }

    let expected_entries = metadata
        .tensors
        .values()
        .map(|tensor| format!("{}.npy", tensor.archive_key))
        .collect::<std::collections::BTreeSet<_>>();
    let unexpected_entries = archive_entries
        .keys()
        .filter(|name| !expected_entries.contains(*name))
        .cloned()
        .collect::<Vec<_>>();
    if !unexpected_entries.is_empty() {
        return Err(format!(
            "unexpected npz entries in {}: {:?}",
            archive_path.display(),
            unexpected_entries
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::module::Module;
    use std::io::Write;
    use std::path::PathBuf;
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};

    type TestBackend = NdArray<f32>;

    fn temp_metadata_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        std::env::temp_dir().join(format!("{name}_{nanos}.json"))
    }

    fn temp_npz_path(name: &str) -> PathBuf {
        temp_metadata_path(name).with_extension("npz")
    }

    fn temp_dir_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        std::env::temp_dir().join(format!("{name}_{nanos}"))
    }

    #[derive(serde::Deserialize)]
    struct ParityOutputs {
        policy_logits: Vec<f32>,
        value: Vec<f32>,
        score_pdf: Vec<f32>,
        score_cdf: Vec<f32>,
        opp_tenpai: Vec<f32>,
        grp: Vec<f32>,
        opp_next_discard: Vec<f32>,
        danger: Vec<f32>,
        oracle_critic: Vec<f32>,
        belief_fields: Vec<f32>,
        mixture_weight_logits: Vec<f32>,
        opponent_hand_type: Vec<f32>,
        delta_q: Vec<f32>,
        safety_residual: Vec<f32>,
    }

    fn tensor_values<const D: usize>(tensor: &Tensor<TestBackend, D>) -> Vec<f32> {
        tensor
            .to_data()
            .as_slice::<f32>()
            .expect("f32 tensor")
            .to_vec()
    }

    fn assert_close(name: &str, actual: &[f32], expected: &[f32], tol: f32) -> f32 {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch for {name}: {} vs {}",
            actual.len(),
            expected.len()
        );
        let mut max_delta = 0.0f32;
        for (index, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            let delta = (a - e).abs();
            if delta > max_delta {
                max_delta = delta;
            }
            assert!(
                delta <= tol,
                "{name}[{index}] mismatch: actual={a}, expected={e}, delta={delta}, tol={tol}"
            );
        }
        max_delta
    }

    fn generate_python_parity_fixture(output_dir: &Path) {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("workspace crates dir")
            .parent()
            .expect("workspace root");
        let output = Command::new("uv")
            .arg("run")
            .arg("--project")
            .arg("python/hydra_phase0_tpu")
            .arg("python")
            .arg("-m")
            .arg("hydra_phase0_tpu.parity_fixture")
            .arg(output_dir)
            .current_dir(repo_root)
            .output()
            .expect("spawn uv parity fixture generator");
        if !output.status.success() {
            panic!(
                "python parity fixture generation failed\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }

    fn load_obs_fixture(path: &Path) -> Vec<f32> {
        let bytes = fs::read(path).expect("read obs fixture bytes");
        let data = read_npy_f32(&bytes, "obs")
            .expect("parse obs fixture npy")
            .into_vec::<f32>()
            .expect("obs as f32 vec");
        assert_eq!(data.len(), 192 * 34, "unexpected obs fixture length");
        data
    }

    fn write_npy_f32(
        path: &str,
        values: &[f32],
        shape: &[usize],
        zip: &mut zip::ZipWriter<fs::File>,
    ) {
        let descr = "<f4";
        let shape_repr = match shape.len() {
            0 => "()".to_string(),
            1 => format!("({},)", shape[0]),
            _ => format!(
                "({})",
                shape
                    .iter()
                    .map(|dim| dim.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        let mut header =
            format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_repr}, }}");
        let preamble_len = 10usize;
        let padding = (16 - ((preamble_len + header.len() + 1) % 16)) % 16;
        header.push_str(&" ".repeat(padding));
        header.push('\n');

        zip.start_file(path, zip::write::SimpleFileOptions::default())
            .expect("start npy entry");
        zip.write_all(b"\x93NUMPY").expect("write magic");
        zip.write_all(&[1, 0]).expect("write version");
        let header_len = u16::try_from(header.len()).expect("header fits u16");
        zip.write_all(&header_len.to_le_bytes())
            .expect("write header len");
        zip.write_all(header.as_bytes()).expect("write header");
        for value in values {
            zip.write_all(&value.to_le_bytes())
                .expect("write f32 value");
        }
    }

    fn write_phase0_archive(
        metadata_path: &Path,
        npz_path: &Path,
        override_tensor: Option<(&str, Vec<f32>, Vec<usize>)>,
        skip_archive_key: Option<&str>,
        extra_entries: &[(&str, Vec<f32>, Vec<usize>)],
    ) {
        let override_lookup = override_tensor
            .as_ref()
            .map(|(name, values, shape)| (name.to_string(), (values.clone(), shape.clone())));

        let tensors = expected_phase0_weight_specs()
            .into_iter()
            .map(|spec| {
                (
                    spec.name.clone(),
                    serde_json::json!({
                        "archive_key": spec.archive_key,
                        "shape": spec.shape,
                        "dtype": "float32"
                    }),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        let payload = serde_json::json!({
            "schema_version": "hydra_phase0_weight_export_v1",
            "tensors": tensors
        });
        fs::write(
            metadata_path,
            serde_json::to_string(&payload).expect("serialize metadata"),
        )
        .expect("write metadata");

        let file = fs::File::create(npz_path).expect("create npz file");
        let mut zip = zip::ZipWriter::new(file);
        for spec in expected_phase0_weight_specs() {
            if skip_archive_key == Some(spec.archive_key.as_str()) {
                continue;
            }
            let (values, shape) = override_lookup
                .as_ref()
                .and_then(|(name, payload)| (name == &spec.name).then_some(payload.clone()))
                .unwrap_or_else(|| {
                    let count = spec.shape.iter().product::<usize>();
                    (vec![0.0f32; count], spec.shape.clone())
                });
            write_npy_f32(
                &format!("{}.npy", spec.archive_key),
                &values,
                &shape,
                &mut zip,
            );
        }
        for (entry_name, values, shape) in extra_entries {
            write_npy_f32(entry_name, values, shape, &mut zip);
        }
        zip.finish().expect("finish npz file");
    }

    #[test]
    fn expected_phase0_weight_shapes_contains_core_tensors() {
        let shapes = expected_phase0_weight_shapes();
        assert_eq!(shapes.get("Dense_0/kernel"), Some(&vec![256, 46]));
        assert_eq!(shapes.get("Dense_1/kernel"), Some(&vec![256, 1]));
        assert_eq!(shapes.get("Conv_0/kernel"), Some(&vec![1, 256, 3]));
        assert_eq!(
            shapes.get("SEResNet_0/Conv_0/kernel"),
            Some(&vec![3, 192, 256])
        );
    }

    #[test]
    fn expected_phase0_weight_specs_define_archive_keys() {
        let specs = expected_phase0_weight_specs();
        let policy = specs
            .iter()
            .find(|spec| spec.name == "Dense_0/kernel")
            .expect("policy spec");
        assert_eq!(policy.archive_key, "Dense_0__kernel");
        assert_eq!(policy.dtype, "float32");
    }

    #[test]
    fn validate_phase0_weight_export_accepts_matching_metadata() {
        let path = temp_metadata_path("phase0_weight_ok");
        let tensors = expected_phase0_weight_shapes()
            .into_iter()
            .map(|(name, shape)| {
                let archive_key = name.replace('/', "__");
                (
                    name,
                    serde_json::json!({
                        "archive_key": archive_key,
                        "shape": shape,
                        "dtype": "float32"
                    }),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        let payload = serde_json::json!({ "schema_version": "hydra_phase0_weight_export_v1", "tensors": tensors });
        fs::write(
            &path,
            serde_json::to_string(&payload).expect("serialize metadata"),
        )
        .expect("write metadata");
        let result = validate_phase0_weight_export(&path);
        fs::remove_file(path).ok();
        assert!(
            result.is_ok(),
            "expected metadata validation success, got {result:?}"
        );
    }

    #[test]
    fn validate_phase0_weight_export_rejects_shape_mismatch() {
        let path = temp_metadata_path("phase0_weight_bad");
        let mut tensors = expected_phase0_weight_shapes()
            .into_iter()
            .map(|(name, shape)| {
                let archive_key = name.replace('/', "__");
                (
                    name,
                    serde_json::json!({
                        "archive_key": archive_key,
                        "shape": shape,
                        "dtype": "float32"
                    }),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        tensors.insert(
            "Dense_0/kernel".to_string(),
            serde_json::json!({
                "archive_key": "dummy_key",
                "shape": [1, 2, 3],
                "dtype": "float32"
            }),
        );
        let payload = serde_json::json!({ "schema_version": "hydra_phase0_weight_export_v1", "tensors": tensors });
        fs::write(
            &path,
            serde_json::to_string(&payload).expect("serialize metadata"),
        )
        .expect("write metadata");
        let result = validate_phase0_weight_export(&path);
        fs::remove_file(path).ok();
        assert!(result.is_err(), "expected metadata validation failure");
    }

    #[test]
    fn validate_phase0_weight_archive_rejects_missing_entries() {
        let meta_path = temp_metadata_path("phase0_weight_archive_meta");
        let npz_path = temp_metadata_path("phase0_weight_archive").with_extension("npz");
        let tensors = expected_phase0_weight_shapes()
            .into_iter()
            .map(|(name, shape)| {
                (
                    name,
                    serde_json::json!({
                        "archive_key": "some_tensor",
                        "shape": shape,
                        "dtype": "float32"
                    }),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        let payload = serde_json::json!({ "schema_version": "hydra_phase0_weight_export_v1", "tensors": tensors });
        fs::write(
            &meta_path,
            serde_json::to_string(&payload).expect("serialize metadata"),
        )
        .expect("write metadata");

        {
            let file = fs::File::create(&npz_path).expect("create npz file");
            let mut zip = zip::ZipWriter::new(file);
            let options = zip::write::SimpleFileOptions::default();
            zip.start_file("other.npy", options)
                .expect("start npy file");
            zip.write_all(b"dummy").expect("write npy bytes");
            zip.finish().expect("finish zip");
        }

        let result = validate_phase0_weight_archive(&meta_path, &npz_path);
        fs::remove_file(meta_path).ok();
        fs::remove_file(npz_path).ok();
        assert!(result.is_err(), "expected archive validation failure");
    }

    #[test]
    fn burn_record_type_is_nameable_for_hydra_model() {
        type LearnerRecord = <crate::model::HydraModel<TestBackend> as Module<TestBackend>>::Record;
        let _size = core::mem::size_of::<LearnerRecord>();
        assert!(_size > 0);
    }

    #[test]
    fn learner_model_can_produce_record_value() {
        let device = Default::default();
        let model = crate::model::HydraModelConfig::learner().init::<TestBackend>(&device);
        let _record = model.into_record();
    }

    #[test]
    fn import_phase0_weights_loads_python_export_into_learner_record() {
        let metadata_path = temp_metadata_path("phase0_import_ok_meta");
        let npz_path = temp_npz_path("phase0_import_ok_weights");
        write_phase0_archive(&metadata_path, &npz_path, None, None, &[]);

        let device = Default::default();
        let model = load_phase0_weight_export::<TestBackend>(&metadata_path, &npz_path, &device)
            .expect("import should succeed");
        let input = Tensor::<TestBackend, 3>::zeros([1, 192, 34], &device);
        let out = model.forward(input);

        assert_eq!(out.policy_logits.dims(), [1, 46]);
        assert_eq!(out.value.dims(), [1, 1]);
        assert_eq!(out.score_pdf.dims(), [1, 64]);
        assert_eq!(out.score_cdf.dims(), [1, 64]);
        assert_eq!(out.opp_tenpai.dims(), [1, 3]);
        assert_eq!(out.grp.dims(), [1, 24]);
        assert_eq!(out.opp_next_discard.dims(), [1, 3, 34]);
        assert_eq!(out.danger.dims(), [1, 3, 34]);
        assert_eq!(out.oracle_critic.dims(), [1, 4]);
        assert_eq!(out.belief_fields.dims(), [1, 16, 34]);
        assert_eq!(out.mixture_weight_logits.dims(), [1, 4]);
        assert_eq!(out.opponent_hand_type.dims(), [1, 24]);
        assert_eq!(out.delta_q.dims(), [1, 46]);
        assert_eq!(out.safety_residual.dims(), [1, 46]);
        assert!(out.is_finite());

        fs::remove_file(metadata_path).ok();
        fs::remove_file(npz_path).ok();
    }

    #[test]
    fn import_phase0_weights_rejects_missing_npz_entry() {
        let metadata_path = temp_metadata_path("phase0_import_missing_meta");
        let npz_path = temp_npz_path("phase0_import_missing_weights");
        write_phase0_archive(
            &metadata_path,
            &npz_path,
            None,
            Some("Dense_0__kernel"),
            &[],
        );

        let device = Default::default();
        let err = load_phase0_weight_export::<TestBackend>(&metadata_path, &npz_path, &device)
            .expect_err("missing entry should fail");
        assert!(
            err.contains("missing npz entry Dense_0__kernel.npy"),
            "{err}"
        );

        fs::remove_file(metadata_path).ok();
        fs::remove_file(npz_path).ok();
    }

    #[test]
    fn import_phase0_weights_rejects_shape_mismatch() {
        let metadata_path = temp_metadata_path("phase0_import_shape_meta");
        let npz_path = temp_npz_path("phase0_import_shape_weights");
        write_phase0_archive(
            &metadata_path,
            &npz_path,
            Some(("Dense_0/kernel", vec![0.0f32; 3], vec![1, 3])),
            None,
            &[],
        );

        let device = Default::default();
        let err = load_phase0_weight_export::<TestBackend>(&metadata_path, &npz_path, &device)
            .expect_err("shape mismatch should fail");
        assert!(err.contains("shape mismatch for Dense_0/kernel"), "{err}");

        fs::remove_file(metadata_path).ok();
        fs::remove_file(npz_path).ok();
    }

    #[test]
    fn import_phase0_weights_matches_python_outputs_on_fixed_observation() {
        let fixture_dir = temp_dir_path("phase0_python_rust_parity");
        if fixture_dir.exists() {
            fs::remove_dir_all(&fixture_dir).ok();
        }
        generate_python_parity_fixture(&fixture_dir);

        let metadata_path = fixture_dir.join("weights_phase0_export.json");
        let npz_path = fixture_dir.join("weights_phase0_export.npz");
        let expected_path = fixture_dir.join("expected_outputs.json");
        let obs_path = fixture_dir.join("obs.npy");

        let device = Default::default();
        let model = load_phase0_weight_export::<TestBackend>(&metadata_path, &npz_path, &device)
            .expect("load exported parity weights");
        let obs_values = load_obs_fixture(&obs_path);
        let input = Tensor::<TestBackend, 1>::from_floats(obs_values.as_slice(), &device)
            .reshape([1, 192, 34]);
        let out = model.forward(input);
        let expected: ParityOutputs = serde_json::from_str(
            &fs::read_to_string(&expected_path).expect("read expected parity outputs"),
        )
        .expect("parse expected parity outputs");

        let tol = 5e-5f32;
        let mut max_deltas = Vec::new();
        max_deltas.push((
            "policy_logits",
            assert_close(
                "policy_logits",
                &tensor_values(&out.policy_logits),
                &expected.policy_logits,
                tol,
            ),
        ));
        max_deltas.push((
            "value",
            assert_close("value", &tensor_values(&out.value), &expected.value, tol),
        ));
        max_deltas.push((
            "score_pdf",
            assert_close(
                "score_pdf",
                &tensor_values(&out.score_pdf),
                &expected.score_pdf,
                tol,
            ),
        ));
        max_deltas.push((
            "score_cdf",
            assert_close(
                "score_cdf",
                &tensor_values(&out.score_cdf),
                &expected.score_cdf,
                tol,
            ),
        ));
        max_deltas.push((
            "opp_tenpai",
            assert_close(
                "opp_tenpai",
                &tensor_values(&out.opp_tenpai),
                &expected.opp_tenpai,
                tol,
            ),
        ));
        max_deltas.push((
            "grp",
            assert_close("grp", &tensor_values(&out.grp), &expected.grp, tol),
        ));
        max_deltas.push((
            "opp_next_discard",
            assert_close(
                "opp_next_discard",
                &tensor_values(&out.opp_next_discard),
                &expected.opp_next_discard,
                tol,
            ),
        ));
        max_deltas.push((
            "danger",
            assert_close("danger", &tensor_values(&out.danger), &expected.danger, tol),
        ));
        max_deltas.push((
            "oracle_critic",
            assert_close(
                "oracle_critic",
                &tensor_values(&out.oracle_critic),
                &expected.oracle_critic,
                tol,
            ),
        ));
        max_deltas.push((
            "belief_fields",
            assert_close(
                "belief_fields",
                &tensor_values(&out.belief_fields),
                &expected.belief_fields,
                tol,
            ),
        ));
        max_deltas.push((
            "mixture_weight_logits",
            assert_close(
                "mixture_weight_logits",
                &tensor_values(&out.mixture_weight_logits),
                &expected.mixture_weight_logits,
                tol,
            ),
        ));
        max_deltas.push((
            "opponent_hand_type",
            assert_close(
                "opponent_hand_type",
                &tensor_values(&out.opponent_hand_type),
                &expected.opponent_hand_type,
                tol,
            ),
        ));
        max_deltas.push((
            "delta_q",
            assert_close(
                "delta_q",
                &tensor_values(&out.delta_q),
                &expected.delta_q,
                tol,
            ),
        ));
        max_deltas.push((
            "safety_residual",
            assert_close(
                "safety_residual",
                &tensor_values(&out.safety_residual),
                &expected.safety_residual,
                tol,
            ),
        ));
        println!("phase0 parity max deltas: {:?}", max_deltas);

        fs::remove_dir_all(fixture_dir).ok();
    }

    #[test]
    fn validate_phase0_weight_archive_rejects_unexpected_entries() {
        let metadata_path = temp_metadata_path("phase0_import_extra_meta");
        let npz_path = temp_npz_path("phase0_import_extra_weights");
        write_phase0_archive(
            &metadata_path,
            &npz_path,
            None,
            None,
            &[("unexpected_extra.npy", vec![0.0f32], vec![1])],
        );

        let err = validate_phase0_weight_archive(&metadata_path, &npz_path)
            .expect_err("unexpected archive entry should fail");
        assert!(err.contains("unexpected npz entries"), "{err}");

        fs::remove_file(metadata_path).ok();
        fs::remove_file(npz_path).ok();
    }
}
