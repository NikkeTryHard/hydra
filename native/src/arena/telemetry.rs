//! Optional wall-clock counters at the native arena boundary.

use std::time::Instant;

#[derive(Default)]
pub(super) struct Telemetry {
    pub batch_calls: u64,
    pub rows: u64,
    pub prepare_ns: u64,
    pub pack_ns: u64,
    pub step_calls: u64,
    pub validate_ns: u64,
    pub apply_ns: u64,
}

impl Telemetry {
    pub(super) fn snapshot(&self) -> [(&'static str, u64); 7] {
        [
            ("batch_calls", self.batch_calls),
            ("rows", self.rows),
            ("prepare_ns", self.prepare_ns),
            ("pack_ns", self.pack_ns),
            ("step_calls", self.step_calls),
            ("validate_ns", self.validate_ns),
            ("apply_ns", self.apply_ns),
        ]
    }
}

pub(super) fn elapsed(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}
