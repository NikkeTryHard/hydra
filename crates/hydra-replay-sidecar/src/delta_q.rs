//! Replay delta-q sidecar record and lookup index contracts.

use std::collections::HashMap;
use std::io;
use std::io::BufRead;

use hydra_core::action::{AKA_5M, AKA_5P, AKA_5S, DISCARD_END, HYDRA_ACTION_SPACE};
use hydra_core::arena::TrajectoryDeltaQLabel;
use serde::{Deserialize, Serialize};

use crate::ActionLabelPair;
use crate::error::{SidecarContractError, SidecarKind};
use crate::jsonl::read_jsonl_records;
use crate::key::ReplayDecisionKey;
use crate::label::{copy_label_arrays, legal_mask_digest_from_f32};
use crate::provenance::{REPLAY_DELTA_Q_PROVENANCE, REPLAY_DELTA_Q_SEMANTICS_V1};

/// Lookup key for replay delta-q labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReplayDeltaQLookupKey {
    /// Replay decision identity.
    pub replay: ReplayDecisionKey,
    /// Chosen action id at the replay decision.
    pub action: u8,
}

/// Version-1 replay delta-q JSONL sidecar record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayDeltaQRecordV1 {
    /// Schema version. Must be 1.
    pub version: u32,
    /// Semantics tag. Must match [`REPLAY_DELTA_Q_SEMANTICS_V1`].
    pub semantics: String,
    /// Provenance tag. Must match [`REPLAY_DELTA_Q_PROVENANCE`].
    pub provenance: String,
    /// Replay decision identity.
    pub key: ReplayDecisionKey,
    /// Chosen action id at the replay decision.
    pub action: u8,
    /// Digest of the legal action mask used to generate this label.
    pub legal_mask_digest: u64,
    /// Hash of the network/checkpoint identity used to generate this label.
    pub source_net_hash: u64,
    /// Version of the network/checkpoint identity contract.
    pub source_version: u32,
    /// Action-space delta-q targets.
    pub target: Vec<f32>,
    /// Action-space support mask.
    pub mask: Vec<f32>,
}

/// In-memory lookup index for replay delta-q sidecar records.
#[derive(Clone, Debug, Default)]
pub struct DeltaQSidecarIndex {
    records: HashMap<ReplayDeltaQLookupKey, ReplayDeltaQRecordV1>,
}

impl DeltaQSidecarIndex {
    /// Builds an index from records; duplicate keys keep the last record.
    pub fn from_records(records: Vec<ReplayDeltaQRecordV1>) -> Self {
        let records = records
            .into_iter()
            .map(|record| {
                (
                    ReplayDeltaQLookupKey {
                        replay: record.key,
                        action: record.action,
                    },
                    record,
                )
            })
            .collect();
        Self { records }
    }

    /// Returns the validated label for a matching key/action/source contract.
    ///
    /// Missing replay/action keys return `Ok(None)`. Present records with a
    /// mismatched contract return [`SidecarContractError`] so callers cannot
    /// silently treat incompatible sidecars as absent labels.
    pub fn lookup_label(
        &self,
        key: &ReplayDecisionKey,
        action: u8,
        legal_mask: &[f32; HYDRA_ACTION_SPACE],
        source_net_hash: u64,
        source_version: u32,
    ) -> Result<Option<ActionLabelPair>, SidecarContractError> {
        let Some(record) = self.records.get(&ReplayDeltaQLookupKey {
            replay: *key,
            action,
        }) else {
            return Ok(None);
        };
        validate_common_delta_q_record(record, legal_mask, source_net_hash, source_version)?;
        let (target, mask) = copy_label_arrays(&record.target, &record.mask)
            .ok_or_else(|| invalid_shape_error(SidecarKind::DeltaQ, record))?;
        let validated = validate_delta_q_contract(&target, &mask, legal_mask).ok_or(
            SidecarContractError::DeltaQContract {
                sidecar: SidecarKind::DeltaQ,
            },
        )?;
        Ok(Some((validated.target, validated.mask)))
    }

    /// Builds an index from a JSONL reader.
    pub fn from_jsonl_reader(reader: impl BufRead) -> io::Result<Self> {
        Ok(Self::from_records(read_jsonl_records(
            reader,
            "replay delta_q sidecar",
        )?))
    }

    /// Builds an index from a JSONL path.
    pub fn from_jsonl_path(path: &std::path::Path) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        Self::from_jsonl_reader(std::io::BufReader::new(file))
    }

    /// Returns number of indexed records.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns true when no records are indexed.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

fn validate_common_delta_q_record(
    record: &ReplayDeltaQRecordV1,
    legal_mask: &[f32; HYDRA_ACTION_SPACE],
    source_net_hash: u64,
    source_version: u32,
) -> Result<(), SidecarContractError> {
    if record.version != 1 {
        return Err(SidecarContractError::Version {
            sidecar: SidecarKind::DeltaQ,
            expected: 1,
            actual: record.version,
        });
    }
    if record.semantics != REPLAY_DELTA_Q_SEMANTICS_V1 {
        return Err(SidecarContractError::Semantics {
            sidecar: SidecarKind::DeltaQ,
            expected: REPLAY_DELTA_Q_SEMANTICS_V1,
        });
    }
    if record.provenance != REPLAY_DELTA_Q_PROVENANCE {
        return Err(SidecarContractError::Provenance {
            sidecar: SidecarKind::DeltaQ,
            expected: REPLAY_DELTA_Q_PROVENANCE,
        });
    }
    let expected_digest = legal_mask_digest_from_f32(legal_mask);
    if record.legal_mask_digest != expected_digest {
        return Err(SidecarContractError::LegalMaskDigest {
            sidecar: SidecarKind::DeltaQ,
            expected: expected_digest,
            actual: record.legal_mask_digest,
        });
    }
    if record.source_net_hash != source_net_hash {
        return Err(SidecarContractError::SourceNetHash {
            sidecar: SidecarKind::DeltaQ,
            expected: source_net_hash,
            actual: record.source_net_hash,
        });
    }
    if record.source_version != source_version {
        return Err(SidecarContractError::SourceVersion {
            sidecar: SidecarKind::DeltaQ,
            expected: source_version,
            actual: record.source_version,
        });
    }
    Ok(())
}

fn invalid_shape_error(
    sidecar: SidecarKind,
    record: &ReplayDeltaQRecordV1,
) -> SidecarContractError {
    if record.target.len() != HYDRA_ACTION_SPACE {
        SidecarContractError::Shape {
            sidecar,
            field: "target",
            expected: HYDRA_ACTION_SPACE,
            actual: record.target.len(),
        }
    } else {
        SidecarContractError::Shape {
            sidecar,
            field: "mask",
            expected: HYDRA_ACTION_SPACE,
            actual: record.mask.len(),
        }
    }
}

/// Validates delta-q sidecar array shape and discard-action contract.
pub fn validate_delta_q_contract(
    target: &[f32; HYDRA_ACTION_SPACE],
    mask: &[f32; HYDRA_ACTION_SPACE],
    legal_mask: &[f32; HYDRA_ACTION_SPACE],
) -> Option<TrajectoryDeltaQLabel> {
    let label = TrajectoryDeltaQLabel::from_slices(target, mask)?;
    let mut saw_masked = false;
    for (action_idx, &legal_value) in legal_mask.iter().enumerate().take(HYDRA_ACTION_SPACE) {
        let mask_value = label.mask[action_idx];
        if mask_value < -1e-6 || ((mask_value - 1.0).abs() > 1e-3 && mask_value > 1e-6) {
            return None;
        }
        let target_value = label.target[action_idx];
        if !target_value.is_finite() {
            return None;
        }
        if mask_value > 0.5 {
            saw_masked = true;
            if legal_value <= 0.0 {
                return None;
            }
            if action_idx > DISCARD_END as usize {
                return None;
            }
            if matches!(action_idx as u8, AKA_5M | AKA_5P | AKA_5S) {
                return None;
            }
        } else if target_value.abs() > 1e-5 {
            return None;
        }
    }
    saw_masked.then_some(label)
}
