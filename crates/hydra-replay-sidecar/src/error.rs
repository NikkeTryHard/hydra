//! Typed sidecar contract errors.

use std::error::Error;
use std::fmt;

/// Replay sidecar family whose contract was violated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SidecarKind {
    /// Replay ExIt sidecar.
    Exit,
    /// Replay delta-q sidecar.
    DeltaQ,
}

impl SidecarKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Exit => "replay ExIt sidecar",
            Self::DeltaQ => "replay delta_q sidecar",
        }
    }
}

/// Structured reason a present sidecar record cannot satisfy the lookup contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SidecarContractError {
    /// Record schema version does not match the lookup contract.
    Version {
        /// Sidecar family.
        sidecar: SidecarKind,
        /// Required schema version.
        expected: u32,
        /// Record schema version.
        actual: u32,
    },
    /// Record semantics tag does not match the lookup contract.
    Semantics {
        /// Sidecar family.
        sidecar: SidecarKind,
        /// Required semantics tag.
        expected: &'static str,
    },
    /// Record provenance tag does not match the lookup contract.
    Provenance {
        /// Sidecar family.
        sidecar: SidecarKind,
        /// Required provenance tag.
        expected: &'static str,
    },
    /// Record legal-mask digest does not match the lookup legal mask.
    LegalMaskDigest {
        /// Sidecar family.
        sidecar: SidecarKind,
        /// Digest computed from the lookup legal mask.
        expected: u64,
        /// Digest stored in the record.
        actual: u64,
    },
    /// Record source network hash does not match the lookup source.
    SourceNetHash {
        /// Sidecar family.
        sidecar: SidecarKind,
        /// Lookup source network hash.
        expected: u64,
        /// Record source network hash.
        actual: u64,
    },
    /// Record source version does not match the lookup source contract.
    SourceVersion {
        /// Sidecar family.
        sidecar: SidecarKind,
        /// Lookup source version.
        expected: u32,
        /// Record source version.
        actual: u32,
    },
    /// Record target or mask length is not the fixed action-space shape.
    Shape {
        /// Sidecar family.
        sidecar: SidecarKind,
        /// Field with invalid shape.
        field: &'static str,
        /// Required element count.
        expected: usize,
        /// Record element count.
        actual: usize,
    },
    /// Delta-q record violates supported action/mask/target invariants.
    DeltaQContract {
        /// Sidecar family.
        sidecar: SidecarKind,
    },
}

impl fmt::Display for SidecarContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Version {
                sidecar,
                expected,
                actual,
            } => write!(
                f,
                "{} version mismatch: expected {expected}, got {actual}",
                sidecar.as_str()
            ),
            Self::Semantics { sidecar, expected } => write!(
                f,
                "{} semantics mismatch: expected {expected}",
                sidecar.as_str()
            ),
            Self::Provenance { sidecar, expected } => write!(
                f,
                "{} provenance mismatch: expected {expected}",
                sidecar.as_str()
            ),
            Self::LegalMaskDigest {
                sidecar,
                expected,
                actual,
            } => write!(
                f,
                "{} legal-mask digest mismatch: expected {expected}, got {actual}",
                sidecar.as_str()
            ),
            Self::SourceNetHash {
                sidecar,
                expected,
                actual,
            } => write!(
                f,
                "{} source net hash mismatch: expected {expected}, got {actual}",
                sidecar.as_str()
            ),
            Self::SourceVersion {
                sidecar,
                expected,
                actual,
            } => write!(
                f,
                "{} source version mismatch: expected {expected}, got {actual}",
                sidecar.as_str()
            ),
            Self::Shape {
                sidecar,
                field,
                expected,
                actual,
            } => write!(
                f,
                "{} {field} shape mismatch: expected {expected}, got {actual}",
                sidecar.as_str()
            ),
            Self::DeltaQContract { sidecar } => {
                write!(f, "{} delta-q contract violation", sidecar.as_str())
            }
        }
    }
}

impl Error for SidecarContractError {}
