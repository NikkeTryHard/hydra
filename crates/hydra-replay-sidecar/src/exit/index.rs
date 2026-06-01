use std::collections::HashMap;
use std::io;
use std::io::BufRead;

use hydra_core::action::HYDRA_ACTION_SPACE;

use crate::ActionLabelPair;
use crate::error::{SidecarContractError, SidecarKind};
use crate::jsonl::read_jsonl_records;
use crate::key::ReplayDecisionKey;
use crate::label::copy_label_arrays;

use super::record::{ReplayExitLookupKey, ReplayExitRecordV1};
use super::validate::{invalid_shape_error, validate_common_exit_record};

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
