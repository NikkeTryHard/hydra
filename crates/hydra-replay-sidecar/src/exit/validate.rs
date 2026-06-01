use hydra_core::action::HYDRA_ACTION_SPACE;

use crate::error::{SidecarContractError, SidecarKind};
use crate::label::legal_mask_digest_from_f32;
use crate::provenance::{REPLAY_EXIT_PROVENANCE, REPLAY_EXIT_SEMANTICS_V1};

use super::record::ReplayExitRecordV1;

pub(super) fn validate_common_exit_record(
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

pub(super) fn invalid_shape_error(
    sidecar: SidecarKind,
    record: &ReplayExitRecordV1,
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
