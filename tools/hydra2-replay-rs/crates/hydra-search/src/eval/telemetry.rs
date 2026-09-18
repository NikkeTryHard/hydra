//! Telemetry row checks — never imputed (EvalControl-owned).
//!
//! Rust owner of `telemetry.py:31-219` logic half (dataclass shapes stay
//! Python-side; this module owns validation + the constructor guards).
//! B3 canon-wins: no digests here, only shape/flag checks.
//!
//! - `REQUIRED_CORE_FIELDS` (12, `:61-74`) + `MODE_REQUIRED_EXTRAS`
//!   (`:77-86`): the contract vocabulary. Row layout notes (Arrow-C
//!   batch: `mode: Utf8View`, digests `Utf8View`, counters `UInt64`,
//!   flags `Boolean`, optionals nullable) are caller-side; validation is
//!   a single null-bitmap + flag pass per batch. Widths arrive as plain
//!   args — this crate defines NO `BATCH` literal (M6, bridge owns it).
//! - `TelemetryTolerance::new` rejects required-core/mode-required at
//!   construction (`:187-197`) — the guard is ported, not just the check.
//! - `telemetry_invalid_reason` (`:200-207`) + `block_missing_telemetry_report`
//!   (`:210-219`): row gate + per-game report (empty = fully usable).
//! - `BlockTolerance` extends the row gate with the three predeclared
//!   admission flags (`blocks.py:68-78`, strict by default, frozen
//!   before results are seen).

use std::collections::{BTreeMap, HashSet};

use crate::SearchError;

use super::py_list_repr;

/// Fields every telemetry row MUST carry regardless of resource view
/// (`REQUIRED_CORE_FIELDS`, `:61-74` — 12, contract order).
pub const REQUIRED_CORE_FIELDS: [&str; 12] = [
    "mode",
    "candidate_spec_hash",
    "hardware_hash",
    "environment_hash",
    "cold_start",
    "synchronized_elapsed_ms",
    "model_calls",
    "exact_transitions",
    "particles",
    "fallback_used",
    "timeout",
    "illegal_action",
];

/// Extra fields a resource view turns from optional into required
/// (`MODE_REQUIRED_EXTRAS`, `:77-86`).
pub fn mode_required_extras(mode: &str) -> &'static [&'static str] {
    match mode {
        "cuda_eager" => &["cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"],
        "torch_compile" => &[
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
            "graph_breaks",
            "recompiles",
        ],
        "energy_metered" => &["energy_joules"],
        _ => &[],
    }
}

/// SPEC 18.2 telemetry row; field order matches the specification
/// (`ResourceTelemetry`, `:31-55`).
#[derive(Debug, Clone, PartialEq)]
pub struct TelemetryRow {
    /// Resource view (`gameplay_5s`, `cuda_eager`, `torch_compile`, ...).
    pub mode: String,
    /// Wall binding (nullable).
    pub wall_id: Option<String>,
    /// Case binding (nullable).
    pub case_id: Option<String>,
    /// Candidate spec digest (`sha256:<hex>`).
    pub candidate_spec_hash: String,
    /// Hardware digest.
    pub hardware_hash: String,
    /// Environment digest.
    pub environment_hash: String,
    /// Cold-start flag.
    pub cold_start: bool,
    /// Synchronized elapsed ms (finite, `>= 0`).
    pub synchronized_elapsed_ms: f64,
    /// Model calls (nonnegative).
    pub model_calls: u64,
    /// Exact transitions (nonnegative).
    pub exact_transitions: u64,
    /// Particles (nonnegative).
    pub particles: u64,
    /// Fallback used flag.
    pub fallback_used: bool,
    /// Timeout flag.
    pub timeout: bool,
    /// Illegal-action flag.
    pub illegal_action: bool,
    /// CUDA peak allocated (nullable; required in cuda views).
    pub cuda_peak_allocated_bytes: Option<u64>,
    /// CUDA peak reserved (nullable; required in cuda views).
    pub cuda_peak_reserved_bytes: Option<u64>,
    /// Host peak (nullable, always optional).
    pub host_peak_bytes: Option<u64>,
    /// Energy joules (nullable; required in `energy_metered`).
    pub energy_joules: Option<f64>,
    /// Graph breaks (nullable; required in `torch_compile`).
    pub graph_breaks: Option<u64>,
    /// Recompiles (nullable; required in `torch_compile`).
    pub recompiles: Option<u64>,
    /// Caller-marked invalidity (nullable, never excusable).
    pub invalid_reason: Option<String>,
}

impl TelemetryRow {
    /// Validate a row (`make_resource_telemetry`, `:118-173`): nonempty
    /// mode, digest shapes, nullable id shapes, nonneg numerics, optional
    /// shapes. Unknown-field rejection is caller-side (typed struct).
    pub fn check(&self) -> Result<(), SearchError> {
        if self.mode.is_empty() {
            return Err(SearchError::InvalidArg { detail: "mode must be a nonempty str" });
        }
        for (name, digest) in [
            ("candidate_spec_hash", &self.candidate_spec_hash),
            ("hardware_hash", &self.hardware_hash),
            ("environment_hash", &self.environment_hash),
        ] {
            if !super::is_digest_text(digest) {
                return Err(SearchError::InvalidArg { detail: name });
            }
        }
        for id in [&self.wall_id, &self.case_id] {
            if let Some(text) = id {
                if text.is_empty() {
                    return Err(SearchError::InvalidArg {
                        detail: "wall_id/case_id must be None or nonempty str",
                    });
                }
            }
        }
        if let Some(reason) = &self.invalid_reason {
            if reason.is_empty() {
                return Err(SearchError::InvalidArg {
                    detail: "invalid_reason must be None or a nonempty str",
                });
            }
        }
        if !self.synchronized_elapsed_ms.is_finite() || self.synchronized_elapsed_ms < 0.0 {
            return Err(SearchError::InvalidArg {
                detail: "synchronized_elapsed_ms must be finite and >= 0",
            });
        }
        if let Some(joules) = self.energy_joules {
            if !joules.is_finite() || joules < 0.0 {
                return Err(SearchError::InvalidArg {
                    detail: "energy_joules must be finite and >= 0",
                });
            }
        }
        Ok(())
    }

    /// `true` iff the named field is `None` (missing, never imputed).
    pub fn field_is_missing(&self, field: &str) -> bool {
        match field {
            "mode" | "candidate_spec_hash" | "hardware_hash" | "environment_hash"
            | "cold_start" | "synchronized_elapsed_ms" | "model_calls"
            | "exact_transitions" | "particles" | "fallback_used" | "timeout"
            | "illegal_action" => false,
            "cuda_peak_allocated_bytes" => self.cuda_peak_allocated_bytes.is_none(),
            "cuda_peak_reserved_bytes" => self.cuda_peak_reserved_bytes.is_none(),
            "host_peak_bytes" => self.host_peak_bytes.is_none(),
            "energy_joules" => self.energy_joules.is_none(),
            "graph_breaks" => self.graph_breaks.is_none(),
            "recompiles" => self.recompiles.is_none(),
            "wall_id" => self.wall_id.is_none(),
            "case_id" => self.case_id.is_none(),
            "invalid_reason" => self.invalid_reason.is_none(),
            _ => false,
        }
    }
}

/// Predeclared tolerance deciding which gaps invalidate a block
/// (`TelemetryTolerance`, `:176-197`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TelemetryTolerance {
    /// Optional fields this tolerance excuses (never required-core,
    /// never mode-required, never `invalid_reason`).
    pub allow_missing: HashSet<String>,
}

impl TelemetryTolerance {
    /// Strict default: excuse nothing.
    pub fn strict() -> Self {
        TelemetryTolerance { allow_missing: HashSet::new() }
    }

    /// Constructor with the `:187-197` guards: excusing a required-core
    /// field or `invalid_reason` fails here (not at check time); excusing
    /// a mode-required field fails in [`required_for`](Self::required_for).
    pub fn new(allow_missing: HashSet<String>) -> Result<Self, SearchError> {
        let core: HashSet<&str> = REQUIRED_CORE_FIELDS.iter().copied().collect();
        let mut forbidden: Vec<&String> =
            allow_missing.iter().filter(|f| core.contains(f.as_str()) || f.as_str() == "invalid_reason").collect();
        forbidden.sort();
        if !forbidden.is_empty() {
            return Err(SearchError::InvalidArg {
                detail: "tolerance cannot excuse required fields",
            });
        }
        Ok(TelemetryTolerance { allow_missing })
    }

    /// Fields required for `mode` (`:192-197`): core + view extras.
    /// Tolerating a mode-required extra fails closed here.
    pub fn required_for(&self, mode: &str) -> Result<Vec<&'static str>, SearchError> {
        let extras = mode_required_extras(mode);
        let tolerated: Vec<&&str> =
            extras.iter().filter(|e| self.allow_missing.contains(**e)).collect();
        if !tolerated.is_empty() {
            return Err(SearchError::InvalidArg {
                detail: "tolerance cannot excuse mode-required fields",
            });
        }
        let mut out: Vec<&'static str> = REQUIRED_CORE_FIELDS.to_vec();
        out.extend_from_slice(extras);
        Ok(out)
    }
}

/// Predeclared invalidity tolerances for block admission
/// (`BlockTolerance`, `blocks.py:68-78`). Boolean flags default strict;
/// they MUST be frozen before results are seen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockTolerance {
    /// Row-level missing-field tolerance.
    pub inner: TelemetryTolerance,
    /// Admit rows that used fallback.
    pub allow_fallback_used: bool,
    /// Admit rows that timed out.
    pub allow_timeout: bool,
    /// Admit rows with illegal actions.
    pub allow_illegal_action: bool,
}

impl BlockTolerance {
    /// Strict fail-closed default.
    pub fn strict() -> Self {
        BlockTolerance {
            inner: TelemetryTolerance::strict(),
            allow_fallback_used: false,
            allow_timeout: false,
            allow_illegal_action: false,
        }
    }
}

impl Default for BlockTolerance {
    fn default() -> Self {
        BlockTolerance::strict()
    }
}

/// Why this row invalidates its block, or `None` when usable
/// (`:200-207`). Strings are byte-identical to Python (the
/// `blocks.py:152` prefix mapping depends on them). A tolerated
/// mode-required extra fails closed here (never silently admitted).
pub fn telemetry_invalid_reason(
    row: &TelemetryRow,
    tolerance: &TelemetryTolerance,
) -> Result<Option<String>, SearchError> {
    if let Some(reason) = &row.invalid_reason {
        return Ok(Some(format!("row marked invalid: {reason}")));
    }
    let required = tolerance.required_for(&row.mode)?;
    let missing: Vec<&str> =
        required.into_iter().filter(|f| row.field_is_missing(f)).collect();
    if missing.is_empty() {
        Ok(None)
    } else {
        Ok(Some(format!(
            "missing required telemetry (never imputed): {}",
            py_list_repr(&missing)
        )))
    }
}

/// Per-game invalidity report; empty means fully usable rows
/// (`:210-219`). Iterates `game_id`-sorted for determinism.
pub fn block_missing_telemetry_report(
    rows_by_game: &BTreeMap<String, TelemetryRow>,
    tolerance: &TelemetryTolerance,
) -> Result<BTreeMap<String, String>, SearchError> {
    let mut report = BTreeMap::new();
    for (game_id, row) in rows_by_game {
        if let Some(reason) = telemetry_invalid_reason(row, tolerance)? {
            report.insert(game_id.clone(), reason);
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digests(ch: char) -> String {
        format!("sha256:{}", ch.to_string().repeat(64))
    }

    pub(crate) fn ok_row() -> TelemetryRow {
        TelemetryRow {
            mode: "gameplay_5s".to_string(),
            wall_id: Some("w-001".to_string()),
            case_id: None,
            candidate_spec_hash: digests('e'),
            hardware_hash: digests('d'),
            environment_hash: digests('d'),
            cold_start: false,
            synchronized_elapsed_ms: 12.5,
            model_calls: 32,
            exact_transitions: 128,
            particles: 0,
            fallback_used: false,
            timeout: false,
            illegal_action: false,
            cuda_peak_allocated_bytes: None,
            cuda_peak_reserved_bytes: None,
            host_peak_bytes: None,
            energy_joules: Some(1.5),
            graph_breaks: None,
            recompiles: None,
            invalid_reason: None,
        }
    }

    /// ROW goldens (`/tmp/eval_goldens4.py`): usable row, caller-marked
    /// row, cuda-view missing pair — byte-identical strings.
    #[test]
    fn row_gate_goldens() {
        let tol = TelemetryTolerance::strict();
        assert_eq!(telemetry_invalid_reason(&ok_row(), &tol).unwrap(), None);
        let mut marked = ok_row();
        marked.invalid_reason = Some("probe-motor-stall".to_string());
        assert_eq!(
            telemetry_invalid_reason(&marked, &tol).unwrap().as_deref(),
            Some("row marked invalid: probe-motor-stall")
        );
        let mut cuda = ok_row();
        cuda.mode = "cuda_eager".to_string();
        assert_eq!(
            telemetry_invalid_reason(&cuda, &tol).unwrap().as_deref(),
            Some(
                "missing required telemetry (never imputed): \
                 ['cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes']"
            )
        );
    }

    /// Constructor guards (`:187-197`): required-core + `invalid_reason`
    /// rejected at construction; mode-required rejected in `required_for`.
    #[test]
    fn tolerance_constructor_guards() {
        let mut bad = HashSet::new();
        bad.insert("model_calls".to_string());
        assert!(TelemetryTolerance::new(bad).is_err());
        let mut bad = HashSet::new();
        bad.insert("invalid_reason".to_string());
        assert!(TelemetryTolerance::new(bad).is_err());
        let mut bad = HashSet::new();
        bad.insert("energy_joules".to_string());
        let tol = TelemetryTolerance::new(bad).unwrap();
        assert!(tol.required_for("energy_metered").is_err());
        // Optional fields are excusable at construction (host_peak_bytes
        // is never required in any view).
        let mut ok = HashSet::new();
        ok.insert("host_peak_bytes".to_string());
        assert!(TelemetryTolerance::new(ok).is_ok());
    }

    /// Vocabulary: 12 core + view extras; `gameplay_5s` adds none,
    /// `cuda_eager` adds its pair.
    #[test]
    fn required_vocabulary() {
        let tol = TelemetryTolerance::strict();
        assert_eq!(tol.required_for("gameplay_5s").unwrap().len(), 12);
        let cuda = tol.required_for("cuda_eager").unwrap();
        assert!(cuda.contains(&"cuda_peak_allocated_bytes"));
        assert!(cuda.contains(&"cuda_peak_reserved_bytes"));
        let tc = tol.required_for("torch_compile").unwrap();
        assert!(tc.contains(&"graph_breaks"));
        assert!(tc.contains(&"recompiles"));
    }

    /// Report is `game_id`-sorted and empty for usable rows.
    #[test]
    fn missing_report_sorted_and_empty() {
        let tol = TelemetryTolerance::strict();
        let mut rows = BTreeMap::new();
        rows.insert("w-001:g1".to_string(), ok_row());
        rows.insert("w-001:g0".to_string(), ok_row());
        assert!(block_missing_telemetry_report(&rows, &tol).unwrap().is_empty());
    }

    /// Row constructor: empty mode, bad digest, empty ids, negative
    /// elapsed, non-finite energy all fail closed.
    #[test]
    fn row_constructor_guards() {
        let mut row = ok_row();
        row.mode.clear();
        assert!(row.check().is_err());
        let mut row = ok_row();
        row.candidate_spec_hash = "not-a-digest".to_string();
        assert!(row.check().is_err());
        let mut row = ok_row();
        row.wall_id = Some(String::new());
        assert!(row.check().is_err());
        let mut row = ok_row();
        row.synchronized_elapsed_ms = -1.0;
        assert!(row.check().is_err());
        let mut row = ok_row();
        row.energy_joules = Some(f64::NAN);
        assert!(row.check().is_err());
        assert!(ok_row().check().is_ok());
    }
}
