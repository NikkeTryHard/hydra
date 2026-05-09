//! Heavy training execution composition layer.
//!
//! This crate owns dependencies for train-command execution that are too broad
//! for `hydra-train-runtime`. It intentionally does not depend on `hydra-train`
//! so execution modules can move here without introducing a workspace cycle.

#![deny(missing_docs)]

/// Marker for the train execution boundary.
///
/// The type gives downstream migration code a concrete public surface to depend
/// on before train execution modules are moved into this crate. Keeping it
/// zero-sized avoids implying that execution delegation has already moved.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub struct TrainExec;

impl TrainExec {
    /// Creates a train execution boundary marker.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}
