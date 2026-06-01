use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayTargetProfile {
    pub oracle: bool,
    pub safety_residual: bool,
    pub belief: bool,
    pub mixture: bool,
    pub exit: bool,
    pub delta_q: bool,
}

impl ReplayTargetProfile {
    pub const fn minimal_bc() -> Self {
        Self {
            oracle: false,
            safety_residual: false,
            belief: false,
            mixture: false,
            exit: false,
            delta_q: false,
        }
    }

    pub const fn with_optional_heads(
        oracle: bool,
        safety_residual: bool,
        belief: bool,
        mixture: bool,
        exit: bool,
        delta_q: bool,
    ) -> Self {
        Self {
            oracle,
            safety_residual,
            belief,
            mixture,
            exit,
            delta_q,
        }
    }
}

pub struct ReplayLoadPolicy<'a> {
    pub profile: ReplayTargetProfile,
    pub observation_profile: ReplayObservationProfile,
    pub exit_provenance: SidecarProvenance,
    pub delta_q_provenance: SidecarProvenance,
    pub exit_sidecar: Option<&'a ExitSidecarIndex>,
    pub delta_q_sidecar: Option<&'a DeltaQSidecarIndex>,
}

impl<'a> ReplayLoadPolicy<'a> {
    pub const fn new(
        profile: ReplayTargetProfile,
        observation_profile: ReplayObservationProfile,
        exit_provenance: SidecarProvenance,
        delta_q_provenance: SidecarProvenance,
        exit_sidecar: Option<&'a ExitSidecarIndex>,
        delta_q_sidecar: Option<&'a DeltaQSidecarIndex>,
    ) -> Self {
        Self {
            profile,
            observation_profile,
            exit_provenance,
            delta_q_provenance,
            exit_sidecar,
            delta_q_sidecar,
        }
    }

    pub(super) fn has_joined_sidecars(&self) -> bool {
        self.exit_sidecar.is_some() || self.delta_q_sidecar.is_some()
    }
}
#[derive(Clone, Copy, Debug, Default)]
pub struct SidecarProvenance {
    pub source_net_hash: Option<u64>,
    pub source_version: Option<u32>,
}

impl SidecarProvenance {
    pub const fn new(source_net_hash: Option<u64>, source_version: Option<u32>) -> Self {
        Self {
            source_net_hash,
            source_version,
        }
    }

    pub(super) fn complete(self) -> Option<(u64, u32)> {
        self.source_net_hash.zip(self.source_version)
    }
}
pub(super) fn lookup_joined_label<T, F>(
    sidecar: Option<&T>,
    replay_key: Option<ReplayDecisionKey>,
    action: u8,
    legal_mask: &[f32; HYDRA_ACTION_SPACE],
    provenance: SidecarProvenance,
    sidecar_kind: SidecarKind,
    lookup: F,
) -> Result<Option<ActionLabelPair>, SidecarContractError>
where
    F: FnOnce(
        &T,
        &ReplayDecisionKey,
        u8,
        &[f32; HYDRA_ACTION_SPACE],
        u64,
        u32,
    ) -> Result<Option<ActionLabelPair>, SidecarContractError>,
{
    let Some(replay_key) = replay_key else {
        return Ok(None);
    };
    let Some(sidecar) = sidecar else {
        return Ok(None);
    };
    let Some((source_net_hash, source_version)) = provenance.complete() else {
        return Err(SidecarContractError::Provenance {
            sidecar: sidecar_kind,
            expected: "complete source_net_hash and source_version",
        });
    };
    lookup(
        sidecar,
        &replay_key,
        action,
        legal_mask,
        source_net_hash,
        source_version,
    )
}
