//! rl/actions: native canonical action expansion over the frozen 6792 census.
//!
//! DAG: `riichienv_core` (engine slots) + `hydra_feed::census` (frozen
//! vocabulary/ids) + `hydra_shard::tile` (mjai-string fallback only).
//! FORBIDS: wall law, observation math, Python callbacks, `get_observation`.
//!
//! TABLE ported here (mirrors `python/hydra2/engines/riichienv/actions.py`):
//! - `canonical_from_engine` <- `:106-151`: one-to-one engine slot mapping;
//!   `riichi` kind is expanded by the caller, never mapped directly.
//! - `expand_engine_legals` <- `:154-241`: tsumogiri twin next to the drawn
//!   discard, one `riichi_discard` per in-hand candidate from
//!   `check_riichi_candidates`, kakan carries the mandatory prior-pon
//!   reference, coarse phase gate last.
//! - `engine_matches_canonical` <- `:244-275`: discard-fold, riichi
//!   candidacy, ankan/kakan/claim/win equalities.
//! - `legal_view` <- `:278-307`: census-id dedup, id-sorted order, full
//!   6792 mask.
//! - `_kind_of` <- `:62-75`: engine `ActionType` discriminants to kinds
//!   (KyushuKyuhai -> abort_nine_terminals; Kita unsupported, fail closed).
//!
//! Red composition (`plan 13`): identity matches on exact physical tids
//! (`16/52/88` aka vs `17/53/89` plain never collapse); `tile // 4` is used
//! ONLY for feature counts in `super::observation`, never for identity.

use hydra_feed::census::{ActionKind, ActionTemplate, census_index_of, frozen_census};
use riichienv_core::action::{Action as CoreAction, ActionType};
use std::collections::BTreeMap;

/// Frozen baseline width (`schema.BASELINE_ACTION_COUNT`, `encoder.rs`).
pub const BASELINE_ACTIONS: usize = 6792;
/// Max physical tile id (0..=135).
pub const TILE_ID_MAX: u8 = 135;
/// Red-five physical ids (aka copies, first of each five block).
pub const RED_TIDS: [u8; 3] = [16, 52, 88];

/// Native failure (fail-closed text, no Python mapping here).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionError(pub String);

impl std::fmt::Display for ActionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "rl actions: {}", self.0)
    }
}

impl std::error::Error for ActionError {}

/// Canonical decision in canonical-seat space (mirrors `CanonicalAction`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalAction {
    pub kind: ActionKind,
    pub actor: u8,
    pub tile: Option<u8>,
    pub called_tile: Option<u8>,
    pub consumed: Vec<u8>,
    pub source: Option<u8>,
    pub declares_riichi: bool,
    pub prior_pon_id: Option<String>,
}

/// Engine `ActionType` discriminant -> canonical kind (`actions.py:62-75`).
pub fn kind_of(action_type: ActionType) -> Result<ActionKind, ActionError> {
    match action_type {
        ActionType::Discard => Ok(ActionKind::Discard),
        ActionType::Chi => Ok(ActionKind::Chi),
        ActionType::Pon => Ok(ActionKind::Pon),
        ActionType::Daiminkan => Ok(ActionKind::Daiminkan),
        ActionType::Ron => Ok(ActionKind::Ron),
        ActionType::Riichi => Err(ActionError(
            "engine riichi is expanded by the caller (riichi_discard)".to_string(),
        )),
        ActionType::Tsumo => Ok(ActionKind::Tsumo),
        ActionType::Pass => Ok(ActionKind::Pass),
        ActionType::Ankan => Ok(ActionKind::Ankan),
        ActionType::Kakan => Ok(ActionKind::Kakan),
        ActionType::KyushuKyuhai => Ok(ActionKind::AbortNineTerminals),
        ActionType::Kita => Err(ActionError("kita is not valid in 4-player mode".to_string())),
    }
}

/// Relative source offset seen from actor (`(source - actor) % 4`, `3 -> -1`;
/// self-source fails closed; mirrors `offset_from_source`).
pub fn source_offset(source: Option<u8>, actor: u8) -> Result<Option<i8>, ActionError> {
    match source {
        None => Ok(None),
        Some(s) => {
            if s > 3 || actor > 3 {
                return Err(ActionError(format!(
                    "source offset seats out of range: source={s} actor={actor}"
                )));
            }
            if s == actor {
                return Err(ActionError("action may not source from its own seat".to_string()));
            }
            let delta = (s + 4 - actor) % 4;
            Ok(Some(if delta == 3 { -1 } else { delta as i8 }))
        }
    }
}

/// Canonical meld reference (`kind:sorted.tiles`; mirrors `meld_delta_value`
/// id composition used by `visible_meld_id`).
pub fn meld_id(kind: &str, tiles: &[u8]) -> String {
    let mut ordered = tiles.to_vec();
    ordered.sort_unstable();
    format!(
        "{kind}:{}",
        ordered
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(".")
    )
}

/// One-to-one engine slot mapping (`canonical_from_engine`).
pub fn canonical_from_engine(
    action: &CoreAction,
    actor: u8,
    offered_by: Option<u8>,
) -> Result<CanonicalAction, ActionError> {
    let kind = kind_of(action.action_type)?;
    let tile = action.tile;
    let mut consumed = action.consume_tiles.clone();
    consumed.sort_unstable();
    if let Some(t) = tile {
        if t > TILE_ID_MAX {
            return Err(ActionError(format!("engine tile {t} out of range")));
        }
    }
    for t in &consumed {
        if *t > TILE_ID_MAX {
            return Err(ActionError(format!("engine consumed tile {t} out of range")));
        }
    }
    match kind {
        ActionKind::Discard => Ok(CanonicalAction {
            kind,
            actor,
            tile,
            called_tile: None,
            consumed: vec![],
            source: None,
            declares_riichi: false,
            prior_pon_id: None,
        }),
        ActionKind::Chi | ActionKind::Pon | ActionKind::Daiminkan => Ok(CanonicalAction {
            kind,
            actor,
            tile: None,
            called_tile: tile,
            consumed,
            source: offered_by,
            declares_riichi: false,
            prior_pon_id: None,
        }),
        ActionKind::Ankan => Ok(CanonicalAction {
            kind,
            actor,
            tile: None,
            called_tile: None,
            consumed,
            source: None,
            declares_riichi: false,
            prior_pon_id: None,
        }),
        ActionKind::Kakan => {
            // Prior-pon gate needs meld context; the expanding caller fills
            // `prior_pon_id`. Direct mapping keeps the tile; caller upgrades.
            Ok(CanonicalAction {
                kind,
                actor,
                tile,
                called_tile: None,
                consumed: vec![],
                source: None,
                declares_riichi: false,
                prior_pon_id: None,
            })
        }
        ActionKind::Ron => Ok(CanonicalAction {
            kind,
            actor,
            tile,
            called_tile: None,
            consumed: vec![],
            source: offered_by,
            declares_riichi: false,
            prior_pon_id: None,
        }),
        ActionKind::Tsumo
        | ActionKind::Pass
        | ActionKind::AbortNineTerminals
        | ActionKind::AcceptAbortiveDraw => Ok(CanonicalAction {
            kind,
            actor,
            tile,
            called_tile: None,
            consumed: vec![],
            source: None,
            declares_riichi: false,
            prior_pon_id: None,
        }),
        _ => Err(ActionError(format!(
            "engine kind {kind:?} is expanded by the caller"
        ))),
    }
}

/// Full canonical expansion of one seat's engine legal set
/// (`expand_engine_legals`; `actor_melds` are `(kind, tiles)` pairs).
pub fn expand_engine_legals(
    engine_actions: &[CoreAction],
    actor: u8,
    phase: &str,
    drawn_tile: Option<u8>,
    own_hand: &[u8],
    offered_by: Option<u8>,
    actor_melds: &[(String, Vec<u8>)],
    riichi_candidates: &[u32],
) -> Result<Vec<CanonicalAction>, ActionError> {
    let mut out: Vec<CanonicalAction> = Vec::new();
    let mut has_riichi = false;
    for engine_action in engine_actions {
        if engine_action.action_type == ActionType::Riichi {
            has_riichi = true;
            continue;
        }
        if engine_action.action_type == ActionType::Kakan {
            let added = engine_action.tile.ok_or_else(|| {
                ActionError("engine offered a kakan without an added tile".to_string())
            })?;
            let added_type = added / 4;
            let prior = actor_melds
                .iter()
                .find(|(kind, tiles)| {
                    kind == "pon" && !tiles.is_empty() && tiles[0] / 4 == added_type
                })
                .ok_or_else(|| {
                    ActionError(format!(
                        "kakan of type {added_type}: no visible prior pon for seat {actor}"
                    ))
                })?;
            let mut action =
                canonical_from_engine(engine_action, actor, offered_by)?;
            action.prior_pon_id = Some(meld_id("pon", &prior.1));
            // Kakan tile must complete the prior pon's quad (mirrors
            // `_find_kakan_base` missing-set check).
            let mut missing: Vec<u8> = (0..4)
                .map(|c| added_type * 4 + c)
                .filter(|t| !prior.1.contains(t))
                .collect();
            missing.sort_unstable();
            if missing != vec![added] {
                return Err(ActionError(format!(
                    "kakan of tile {added}: prior pon leaves {missing:?} free"
                )));
            }
            out.push(action);
            continue;
        }
        let mut action = canonical_from_engine(engine_action, actor, offered_by)?;
        out.push(action.clone());
        // Tsumogiri twin next to the drawn discard.
        if engine_action.action_type == ActionType::Discard
            && let Some(drawn) = drawn_tile
            && engine_action.tile == Some(drawn)
        {
            action.kind = ActionKind::Tsumogiri;
            out.push(action);
        }
    }
    if has_riichi {
        let mut seen: Vec<u32> = riichi_candidates.to_vec();
        seen.sort_unstable();
        seen.dedup();
        for candidate in seen {
            if candidate <= u32::from(TILE_ID_MAX) && own_hand.contains(&(candidate as u8)) {
                out.push(CanonicalAction {
                    kind: ActionKind::RiichiDiscard,
                    actor,
                    tile: Some(candidate as u8),
                    called_tile: None,
                    consumed: vec![],
                    source: None,
                    declares_riichi: true,
                    prior_pon_id: None,
                });
            }
        }
    }
    Ok(out.into_iter().filter(|a| phase_allows(&a.kind, phase)).collect())
}

/// Coarse phase gate (`ACTION_PHASES`, `contracts.rs:1417-1434`).
pub fn phase_allows(kind: &ActionKind, phase: &str) -> bool {
    match kind {
        ActionKind::Pass => matches!(phase, "discard_response" | "kan_response"),
        ActionKind::Discard | ActionKind::Tsumogiri | ActionKind::RiichiDiscard => {
            phase == "draw_decision"
        }
        ActionKind::Chi | ActionKind::Pon | ActionKind::Daiminkan => phase == "discard_response",
        ActionKind::Ankan | ActionKind::Kakan => phase == "draw_decision",
        ActionKind::Ron => matches!(phase, "discard_response" | "kan_response"),
        ActionKind::Tsumo => phase == "draw_decision",
        ActionKind::AbortNineTerminals => phase == "draw_decision",
        ActionKind::AcceptAbortiveDraw => {
            matches!(phase, "discard_response" | "kan_response")
        }
    }
}

/// Exact-match test between a canonical action and an engine slot
/// (`engine_matches_canonical`).
pub fn engine_matches_canonical(
    engine_action: &CoreAction,
    wanted: &CanonicalAction,
    riichi_candidates: &[u32],
) -> bool {
    match engine_action.action_type {
        ActionType::Discard => {
            matches!(wanted.kind, ActionKind::Discard | ActionKind::Tsumogiri)
                && wanted.tile == engine_action.tile
        }
        ActionType::Riichi => {
            wanted.kind == ActionKind::RiichiDiscard
                && wanted.tile.is_some_and(|t| riichi_candidates.contains(&u32::from(t)))
        }
        _ => {
            if wanted.kind == ActionKind::Pass {
                return engine_action.action_type == ActionType::Pass;
            }
            if wanted.kind == ActionKind::AbortNineTerminals {
                return engine_action.action_type == ActionType::KyushuKyuhai;
            }
            let mut consumed = engine_action.consume_tiles.clone();
            consumed.sort_unstable();
            match wanted.kind {
                ActionKind::Ankan => {
                    engine_action.action_type == ActionType::Ankan && consumed == wanted.consumed
                }
                ActionKind::Kakan => {
                    engine_action.action_type == ActionType::Kakan
                        && engine_action.tile == wanted.tile
                }
                ActionKind::Chi | ActionKind::Pon | ActionKind::Daiminkan => {
                    kind_of(engine_action.action_type).ok() == Some(wanted.kind)
                        && engine_action.tile == wanted.called_tile
                        && consumed == wanted.consumed
                }
                ActionKind::Ron | ActionKind::Tsumo => {
                    kind_of(engine_action.action_type).ok() == Some(wanted.kind)
                        && engine_action.tile == wanted.tile
                }
                _ => false,
            }
        }
    }
}

/// Census probe for one canonical action in context.
pub fn canonical_id(action: &CanonicalAction) -> Result<u32, ActionError> {
    let offset = source_offset(action.source, action.actor)?;
    let mut consumed = [0u8; 4];
    if action.consumed.len() > 4 {
        return Err(ActionError("consumed group exceeds quad".to_string()));
    }
    let mut sorted = action.consumed.clone();
    sorted.sort_unstable();
    for (i, t) in sorted.iter().enumerate() {
        consumed[i] = *t;
    }
    let probe = ActionTemplate {
        kind: action.kind,
        tile: action.tile,
        called_tile: action.called_tile,
        consumed,
        consumed_len: sorted.len() as u8,
        source_offset: offset,
        declares_riichi: action.kind == ActionKind::RiichiDiscard,
        meld_ref_required: action.kind == ActionKind::Kakan,
    };
    census_index_of(frozen_census(), &probe)
        .map(|i| i as u32)
        .ok_or_else(|| ActionError(format!("action {action:?} absent from frozen census")))
}

/// Sorted canonical actions plus the full 6792 mask (`legal_view`).
pub fn legal_view(
    expanded: Vec<CanonicalAction>,
) -> Result<(Vec<(u32, CanonicalAction)>, Vec<bool>), ActionError> {
    let mut by_id: BTreeMap<u32, CanonicalAction> = BTreeMap::new();
    for action in expanded {
        by_id.insert(canonical_id(&action)?, action);
    }
    let mut mask = vec![false; BASELINE_ACTIONS];
    let mut ordered = Vec::with_capacity(by_id.len());
    for (id, action) in by_id {
        mask[id as usize] = true;
        ordered.push((id, action));
    }
    Ok((ordered, mask))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_gates_cover_all_kinds() {
        assert!(phase_allows(&ActionKind::Discard, "draw_decision"));
        assert!(!phase_allows(&ActionKind::Discard, "discard_response"));
        assert!(phase_allows(&ActionKind::Chi, "discard_response"));
        assert!(!phase_allows(&ActionKind::Chi, "draw_decision"));
        assert!(phase_allows(&ActionKind::Ron, "kan_response"));
        assert!(phase_allows(&ActionKind::Pass, "kan_response"));
        assert!(!phase_allows(&ActionKind::Tsumo, "discard_response"));
        assert!(phase_allows(&ActionKind::Ankan, "draw_decision"));
        assert!(phase_allows(&ActionKind::AbortNineTerminals, "draw_decision"));
    }

    #[test]
    fn source_offset_maps_three_to_minus_one() {
        assert_eq!(source_offset(None, 0).unwrap(), None);
        assert_eq!(source_offset(Some(1), 0).unwrap(), Some(1));
        assert_eq!(source_offset(Some(0), 1).unwrap(), Some(-1));
        assert!(source_offset(Some(2), 2).is_err());
    }

    #[test]
    fn red_and_plain_fives_hold_distinct_ids() {
        let mk = |kind, tile| CanonicalAction {
            kind,
            actor: 0,
            tile: Some(tile),
            called_tile: None,
            consumed: vec![],
            source: None,
            declares_riichi: false,
            prior_pon_id: None,
        };
        let aka = canonical_id(&mk(ActionKind::Discard, 16)).unwrap();
        let plain = canonical_id(&mk(ActionKind::Discard, 17)).unwrap();
        assert_ne!(aka, plain);
    }

    #[test]
    fn pass_templates_distinct_by_offset() {
        let mk = |source| CanonicalAction {
            kind: ActionKind::Pass,
            actor: 0,
            tile: None,
            called_tile: None,
            consumed: vec![],
            source,
            declares_riichi: false,
            prior_pon_id: None,
        };
        let a = canonical_id(&mk(None)).unwrap();
        let b = canonical_id(&mk(Some(1))).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn tsumogiri_twin_and_kakan_gate() {
        let discard = CoreAction::new(ActionType::Discard, Some(17), vec![], Some(0));
        let expanded = expand_engine_legals(
            std::slice::from_ref(&discard),
            0,
            "draw_decision",
            Some(17),
            &[17],
            None,
            &[],
            &[],
        )
        .unwrap();
        assert!(expanded.iter().any(|a| a.kind == ActionKind::Tsumogiri));
        let kakan = CoreAction::new(ActionType::Kakan, Some(20), vec![], Some(0));
        assert!(
            expand_engine_legals(
                std::slice::from_ref(&kakan),
                0,
                "draw_decision",
                None,
                &[20],
                None,
                &[],
                &[],
            )
            .is_err()
        );
    }
}
