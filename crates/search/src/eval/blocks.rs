//! Block aggregate — wall-block atomicity is the determinism gate (EvalControl-owned).
//!
//! Rust owner of `blocks.py:37-178` logic half (dataclass shapes stay
//! Python-side; this module owns the math + admission chain).
//! B3 canon-wins: no digests here (identity lives in `wall`/`partition`),
//! so no canon calls — only float math + the telemetry checks via
//! `super::telemetry`.
//!
//! - `aggregate_wall_block` (`:61-65`): collapse ONE block via
//!   `Neumaier == math.fsum` (`super::block_mean`); empty raises.
//! - `aggregate_blocks` (`:108-126`): whole blocks, wall-id order stable,
//!   exclude-and-report invalid ones (never imputed, never averaged).
//! - `_first_disqualification` (`:129-178`): empty -> missing_telemetry ->
//!   row_invalid/missing -> fallback -> timeout -> illegal. Reason strings
//!   in `EXCLUSION_REASONS` (`:81-88`) are CONTRACT — byte-identical.
//!
//! T1/T6: games inside a wall NEVER separate — this module takes whole
//! blocks only; no game-level resample entry point exists.

use std::collections::HashMap;

use crate::SearchError;

use super::block_mean;
use super::telemetry::{BlockTolerance, TelemetryRow, telemetry_invalid_reason};

/// One complete wall block: atomic unit of confirmation (`:37-44`).
#[derive(Debug, Clone, PartialEq)]
pub struct WallBlock {
    /// Wall identity (nonempty).
    pub wall_id: String,
    /// Game ids (parallel to `contrasts`).
    pub game_ids: Vec<String>,
    /// Expected-final-placement contrasts (finite).
    pub contrasts: Vec<f64>,
}

impl WallBlock {
    /// Validate (`__post_init__`, `:45-58`): nonempty id, parallel
    /// lengths, nonempty game ids, finite contrasts.
    pub fn new(
        wall_id: String,
        game_ids: Vec<String>,
        contrasts: Vec<f64>,
    ) -> Result<Self, SearchError> {
        if wall_id.is_empty() {
            return Err(SearchError::InvalidArg { detail: "wall_id must be a nonempty str" });
        }
        if game_ids.len() != contrasts.len() {
            return Err(SearchError::InvalidArg {
                detail: "game_ids and contrasts must have equal length",
            });
        }
        for game_id in &game_ids {
            if game_id.is_empty() {
                return Err(SearchError::InvalidArg {
                    detail: "game_ids must be nonempty strings",
                });
            }
        }
        for v in &contrasts {
            if !v.is_finite() {
                return Err(SearchError::NonFinite { context: "block contrast" });
            }
        }
        Ok(WallBlock { wall_id, game_ids, contrasts })
    }
}

/// Collapse the block to ONE number; games inside are not independent
/// (`:61-65`). `math.fsum`-exact via [`block_mean`].
pub fn aggregate_wall_block(block: &WallBlock) -> Result<f64, SearchError> {
    block_mean(&block.contrasts)
}

/// Exclusion reasons — CONTRACT strings, byte-identical to
/// `EXCLUSION_REASONS` (`:81-88`).
pub const EXCLUSION_REASONS: [&str; 6] = [
    "missing_telemetry",
    "fallback_used",
    "timeout",
    "illegal_action",
    "row_invalid",
    "empty_block",
];

/// A reported exclusion: block identity, reason, human detail (`:91-97`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExcludedBlock {
    /// Excluded wall id.
    pub wall_id: String,
    /// One of [`EXCLUSION_REASONS`].
    pub reason: String,
    /// Human-readable detail.
    pub detail: String,
}

/// Validated block values plus the full exclusion report (`:100-105`).
#[derive(Debug, Clone, PartialEq)]
pub struct BlockAggregateResult {
    /// `(wall_id, block mean)` in wall-id order.
    pub valid: Vec<(String, f64)>,
    /// Exclusions in wall-id order.
    pub excluded: Vec<ExcludedBlock>,
}

/// Aggregate whole blocks; exclude-and-report invalid ones (`:108-126`).
///
/// Blocks visit in wall-id sorted order; the disqualification chain is
/// [`first_disqualification`]. Default tolerance is strict fail-closed.
pub fn aggregate_blocks(
    blocks: &[WallBlock],
    telemetry_by_game: &HashMap<String, TelemetryRow>,
    tolerance: &BlockTolerance,
) -> Result<BlockAggregateResult, SearchError> {
    let mut ordered: Vec<&WallBlock> = blocks.iter().collect();
    ordered.sort_by(|a, b| a.wall_id.cmp(&b.wall_id));
    let mut valid = Vec::new();
    let mut excluded = Vec::new();
    for block in ordered {
        if let Some(exc) = first_disqualification(block, telemetry_by_game, tolerance)? {
            excluded.push(exc);
            continue;
        }
        match aggregate_wall_block(block) {
            Ok(mean) => valid.push((block.wall_id.clone(), mean)),
            Err(_) => excluded.push(ExcludedBlock {
                wall_id: block.wall_id.clone(),
                reason: "empty_block".to_string(),
                detail: "block carries no games".to_string(),
            }),
        }
    }
    Ok(BlockAggregateResult { valid, excluded })
}

/// First disqualification in contract order (`:129-178`):
/// empty -> missing_telemetry (absent rows) -> row_invalid (caller-marked)
/// / missing_telemetry (gaps) -> fallback_used -> timeout -> illegal_action.
/// A tolerated mode-required extra fails closed via `Err` (never silently
/// admitted).
pub fn first_disqualification(
    block: &WallBlock,
    telemetry_by_game: &HashMap<String, TelemetryRow>,
    tolerance: &BlockTolerance,
) -> Result<Option<ExcludedBlock>, SearchError> {
    if block.contrasts.is_empty() {
        return Ok(Some(ExcludedBlock {
            wall_id: block.wall_id.clone(),
            reason: "empty_block".to_string(),
            detail: "block carries no games".to_string(),
        }));
    }
    let missing_rows: Vec<&str> = block
        .game_ids
        .iter()
        .filter(|g| !telemetry_by_game.contains_key(g.as_str()))
        .map(String::as_str)
        .collect();
    if !missing_rows.is_empty() {
        let mut detail = String::from("no telemetry rows for games [");
        for (i, game_id) in missing_rows.iter().enumerate() {
            if i > 0 {
                detail.push_str(", ");
            }
            detail.push('\'');
            detail.push_str(game_id);
            detail.push('\'');
        }
        detail.push(']');
        return Ok(Some(ExcludedBlock {
            wall_id: block.wall_id.clone(),
            reason: "missing_telemetry".to_string(),
            detail,
        }));
    }
    for game_id in &block.game_ids {
        let row = &telemetry_by_game[game_id.as_str()];
        if let Some(reason) = telemetry_invalid_reason(row, &tolerance.inner)? {
            // Prefix distinguishes caller-marked-invalid rows from
            // telemetry gaps; both exclude, reasons must not merge (`:150-152`).
            let mapped = if reason.starts_with("row marked") {
                "row_invalid"
            } else {
                "missing_telemetry"
            };
            return Ok(Some(ExcludedBlock {
                wall_id: block.wall_id.clone(),
                reason: mapped.to_string(),
                detail: format!("game {game_id}: {reason}"),
            }));
        }
    }
    for game_id in &block.game_ids {
        let row = &telemetry_by_game[game_id.as_str()];
        if row.fallback_used && !tolerance.allow_fallback_used {
            return Ok(Some(ExcludedBlock {
                wall_id: block.wall_id.clone(),
                reason: "fallback_used".to_string(),
                detail: format!("game {game_id} used fallback"),
            }));
        }
        if row.timeout && !tolerance.allow_timeout {
            return Ok(Some(ExcludedBlock {
                wall_id: block.wall_id.clone(),
                reason: "timeout".to_string(),
                detail: format!("game {game_id} timed out"),
            }));
        }
        if row.illegal_action && !tolerance.allow_illegal_action {
            return Ok(Some(ExcludedBlock {
                wall_id: block.wall_id.clone(),
                reason: "illegal_action".to_string(),
                detail: format!("game {game_id} produced an illegal action"),
            }));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::telemetry::TelemetryTolerance;

    fn digests(s: &str) -> String {
        format!("sha256:{}", s)
    }

    fn ok_row() -> TelemetryRow {
        TelemetryRow {
            mode: "gameplay_5s".to_string(),
            wall_id: Some("w-001".to_string()),
            case_id: None,
            candidate_spec_hash: digests(&"e".repeat(64)),
            hardware_hash: digests(&"d".repeat(64)),
            environment_hash: digests(&"d".repeat(64)),
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

    /// T1-part: wall order stable + Neumaier-exact collapse (AGG golden
    /// `0.4166666666666667`).
    #[test]
    fn aggregate_single_block_golden() {
        let block = WallBlock::new(
            "w-001".to_string(),
            vec!["w-001:g0".to_string(), "w-001:g1".to_string(), "w-001:g2".to_string()],
            vec![0.5, -0.25, 1.0],
        )
        .unwrap();
        assert_eq!(aggregate_wall_block(&block).unwrap(), 0.4166666666666667);
    }

    /// Empty block raises with the contract reason.
    #[test]
    fn empty_block_excluded() {
        let block =
            WallBlock::new("w-e".to_string(), Vec::new(), Vec::new()).unwrap();
        let rows = HashMap::new();
        let exc = first_disqualification(&block, &rows, &BlockTolerance::strict())
            .unwrap()
            .unwrap();
        assert_eq!(exc.reason, "empty_block");
    }

    /// T6: each `EXCLUSION_REASONS` value triggers once with a
    /// byte-identical reason string; wall order stable.
    #[test]
    fn exclusion_reasons_contract() {
        // missing_telemetry: no rows at all.
        let a = WallBlock::new("w-a".to_string(), vec!["w-a:g0".to_string()], vec![0.5]).unwrap();
        let rows = HashMap::new();
        let tol = BlockTolerance::strict();
        let r = aggregate_blocks(&[a], &rows, &tol).unwrap();
        assert!(r.valid.is_empty());
        assert_eq!(r.excluded[0].reason, "missing_telemetry");

        // row_invalid: caller-marked row.
        let b = WallBlock::new("w-b".to_string(), vec!["w-b:g0".to_string()], vec![0.5]).unwrap();
        let mut marked = ok_row();
        marked.invalid_reason = Some("probe-motor-stall".to_string());
        let rows: HashMap<String, TelemetryRow> =
            [("w-b:g0".to_string(), marked)].into_iter().collect();
        let r = aggregate_blocks(&[b], &rows, &tol).unwrap();
        assert_eq!(r.excluded[0].reason, "row_invalid");

        // fallback_used / timeout / illegal_action: flag rows.
        for (reason, mutator) in [
            ("fallback_used", true),
            ("timeout", true),
            ("illegal_action", true),
        ] {
            let w = WallBlock::new(
                format!("w-{reason}"),
                vec![format!("w-{reason}:g0")],
                vec![0.5],
            )
            .unwrap();
            let mut row = ok_row();
            match reason {
                "fallback_used" => row.fallback_used = mutator,
                "timeout" => row.timeout = mutator,
                _ => row.illegal_action = mutator,
            }
            let rows: HashMap<String, TelemetryRow> =
                [(format!("w-{reason}:g0"), row)].into_iter().collect();
            let r = aggregate_blocks(&[w], &rows, &tol).unwrap();
            assert_eq!(r.excluded[0].reason, reason);
        }

        // Reason vocabulary is exactly the contract set.
        let mut vocab = EXCLUSION_REASONS.to_vec();
        vocab.sort_unstable();
        assert_eq!(
            vocab,
            vec![
                "empty_block",
                "fallback_used",
                "illegal_action",
                "missing_telemetry",
                "row_invalid",
                "timeout"
            ]
        );
    }

    /// Tolerant flags admit flagged rows (admission semantics).
    #[test]
    fn tolerance_admits_flagged_rows() {
        let w =
            WallBlock::new("w-t".to_string(), vec!["w-t:g0".to_string()], vec![0.5]).unwrap();
        let mut row = ok_row();
        row.fallback_used = true;
        let rows: HashMap<String, TelemetryRow> =
            [("w-t:g0".to_string(), row)].into_iter().collect();
        let tol = BlockTolerance {
            inner: TelemetryTolerance::default(),
            allow_fallback_used: true,
            allow_timeout: false,
            allow_illegal_action: false,
        };
        let r = aggregate_blocks(&[w], &rows, &tol).unwrap();
        assert_eq!(r.valid.len(), 1);
    }

    /// Constructor guards: empty id, parallel lengths, empty game id,
    /// non-finite contrast all fail closed.
    #[test]
    fn wall_block_constructor_guards() {
        assert!(WallBlock::new(String::new(), vec!["g".to_string()], vec![0.5]).is_err());
        assert!(WallBlock::new("w".to_string(), vec!["g".to_string()], vec![]).is_err());
        assert!(WallBlock::new("w".to_string(), vec![String::new()], vec![0.5]).is_err());
        assert!(WallBlock::new("w".to_string(), vec!["g".to_string()], vec![f64::NAN]).is_err());
        assert!(WallBlock::new("w".to_string(), vec!["g".to_string()], vec![f64::INFINITY]).is_err());
    }
}
