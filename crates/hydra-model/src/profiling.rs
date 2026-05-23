//! Lightweight model-local profiling scopes.

/// Guard returned by [`scope`].
pub struct ModelScope;

/// Opens a no-op model profiling scope. Rust LibTorch/NVTX support has been removed.
#[inline]
pub fn scope(_stage: &'static str) -> ModelScope {
    ModelScope
}
