//! Native paired-arena evaluation for Python control.

mod batched;
mod metrics;
mod native;
pub(crate) mod sampling;
mod shared;

pub(crate) use batched::{run_paired_arena, run_paired_arena_batched};
pub(crate) use metrics::PyPairedArenaMetrics;
pub(crate) use native::run_paired_arena_rust_native;
