//! Replay ExIt sidecar record and lookup index contracts.

use std::collections::HashMap;
use std::io;
use std::io::BufRead;

use hydra_core::action::HYDRA_ACTION_SPACE;
use serde::{Deserialize, Serialize};

use crate::ActionLabelPair;
use crate::error::{SidecarContractError, SidecarKind};
use crate::jsonl::read_jsonl_records;
use crate::key::ReplayDecisionKey;
use crate::label::{copy_label_arrays, legal_mask_digest_from_f32};
use crate::provenance::{REPLAY_EXIT_PROVENANCE, REPLAY_EXIT_SEMANTICS_V1};

/// Lookup key for replay ExIt labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReplayExitLookupKey {
    /// Replay decision identity.
    pub replay: ReplayDecisionKey,
    /// Chosen action id at the replay decision.
    pub action: u8,
}

/// Version-1 replay ExIt JSONL sidecar record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReplayExitRecordV1 {
    /// Schema version. Must be 1.
    pub version: u32,
    /// Semantics tag. Must match [`REPLAY_EXIT_SEMANTICS_V1`].
    pub semantics: String,
    /// Provenance tag. Must match [`REPLAY_EXIT_PROVENANCE`].
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
    /// Root search visit budget used for the record.
    pub root_visit_count: u32,
    /// Number of legal discard actions at the replay decision.
    pub legal_discard_count: u8,
    /// Number of target-supported actions in the record.
    pub supported_actions: u8,
    /// Supported-action coverage over legal discards.
    pub coverage: f32,
    /// KL divergence from the base policy.
    pub kl_to_base: f32,
    /// Action-space target probabilities.
    pub target: Vec<f32>,
    /// Action-space support mask.
    pub mask: Vec<f32>,
}

/// In-memory lookup index for replay ExIt sidecar records.
#[derive(Clone, Debug, Default)]
pub struct ExitSidecarIndex {
    records: HashMap<ReplayExitLookupKey, ReplayExitRecordV1>,
}

impl ExitSidecarIndex {
    /// Builds an index from records; duplicate keys keep the last record.
    pub fn from_records(records: Vec<ReplayExitRecordV1>) -> Self {
        let records = records
            .into_iter()
            .map(|record| {
                (
                    ReplayExitLookupKey {
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
        let Some(record) = self.records.get(&ReplayExitLookupKey {
            replay: *key,
            action,
        }) else {
            return Ok(None);
        };
        validate_common_exit_record(record, legal_mask, source_net_hash, source_version)?;
        copy_label_arrays(&record.target, &record.mask)
            .ok_or_else(|| invalid_shape_error(SidecarKind::Exit, record))
            .map(Some)
    }

    /// Builds an index from a JSONL reader.
    pub fn from_jsonl_reader(reader: impl BufRead) -> io::Result<Self> {
        Ok(Self::from_records(read_jsonl_records(
            reader,
            "replay ExIt sidecar",
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

fn validate_common_exit_record(
    record: &ReplayExitRecordV1,
    legal_mask: &[f32; HYDRA_ACTION_SPACE],
    source_net_hash: u64,
    source_version: u32,
) -> Result<(), SidecarContractError> {
    if record.version != 1 {
        return Err(SidecarContractError::Version {
            sidecar: SidecarKind::Exit,
            expected: 1,
            actual: record.version,
        });
    }
    if record.semantics != REPLAY_EXIT_SEMANTICS_V1 {
        return Err(SidecarContractError::Semantics {
            sidecar: SidecarKind::Exit,
            expected: REPLAY_EXIT_SEMANTICS_V1,
        });
    }
    if record.provenance != REPLAY_EXIT_PROVENANCE {
        return Err(SidecarContractError::Provenance {
            sidecar: SidecarKind::Exit,
            expected: REPLAY_EXIT_PROVENANCE,
        });
    }
    let expected_digest = legal_mask_digest_from_f32(legal_mask);
    if record.legal_mask_digest != expected_digest {
        return Err(SidecarContractError::LegalMaskDigest {
            sidecar: SidecarKind::Exit,
            expected: expected_digest,
            actual: record.legal_mask_digest,
        });
    }
    if record.source_net_hash != source_net_hash {
        return Err(SidecarContractError::SourceNetHash {
            sidecar: SidecarKind::Exit,
            expected: source_net_hash,
            actual: record.source_net_hash,
        });
    }
    if record.source_version != source_version {
        return Err(SidecarContractError::SourceVersion {
            sidecar: SidecarKind::Exit,
            expected: source_version,
            actual: record.source_version,
        });
    }
    Ok(())
}

fn invalid_shape_error(sidecar: SidecarKind, record: &ReplayExitRecordV1) -> SidecarContractError {
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
