use hydra_core::action::{AKA_5M, AKA_5P, AKA_5S, DISCARD_END, HYDRA_ACTION_SPACE};
use hydra_core::arena::TrajectoryDeltaQLabel;

use crate::error::{SidecarContractError, SidecarKind};
use crate::label::legal_mask_digest_from_f32;
use crate::provenance::{REPLAY_DELTA_Q_PROVENANCE, REPLAY_DELTA_Q_SEMANTICS_V1};

use super::record::ReplayDeltaQRecordV1;

pub(super) fn validate_common_delta_q_record(
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

pub(super) fn invalid_shape_error(
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
