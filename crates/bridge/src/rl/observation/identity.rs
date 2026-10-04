//! rl/observation/identity: CURRENT-field identity hash emitter.
//!
//! Sole production emitter behind `NativeObservation::bind_hash` and
//! `independent_validate`. The full-document `identity_json` stays the
//! external/oracle export path and shares its non-history value
//! constructions through `identity_base_map`, so the emitter cannot drift
//! from the oracle. Each call rebuilds every non-history value from the
//! live observation with the same printer/guards
//! (`hydra_feed::canon::canonical_bytes_value`, `rl:observation-identity`
//! record) that the oracle path uses, streams the fixed 35-key root in
//! byte-lexicographic order through a fresh `Sha256`, and references
//! history only through the arena's immutable per-event canonical bytes.
//!
//! Byte-equality argument: every non-history field value is canonicalized
//! with the same authority as the oracle path, so each embedded value byte
//! string is identical to its nested encoding. The root object emits fixed
//! ASCII keys as `"key":value` joined by `,` inside `{`/`}` in
//! byte-lexicographic order (== UTF-16BE order for ASCII, matching
//! `serde_jcs` key sorting); the history array emits cached envelope bytes
//! joined by `,` inside `[`/`]` with no whitespace, matching JCS array
//! encoding. No replacement text hacks, no second printer, no hash chain.
//!
//! Gate independence: each call rebuilds scalar/legal bytes from CURRENT
//! observation fields and rehashes the FULL canonical stream; immutable
//! event subbytes reuse is the only cross-gate sharing. A single reused
//! whole-observation hash for two gates never happens here: no
//! whole-observation digest, root byte string, or mask-hash cache is ever
//! stored or trusted.
//!
//! Legal-mask memo invariant: the memo holds exactly one entry per thread,
//! `(CURRENT mask copy, its canonical value bytes)`. A hit requires exact
//! length plus every `BASELINE_ACTIONS` bit equal to the CURRENT
//! `obs.legal_mask` (`cached_mask.as_slice() == mask`); length-only,
//! prefix-only, or digest comparisons are rejected. A miss rebuilds from
//! the CURRENT mask through the authority-derived atom fragments below and
//! replaces the entry. A legal mutation that keeps at least one `True`
//! therefore misses, rehashes from the CURRENT mask, and
//! `independent_validate` fails against the old bound hash. Memory is one
//! entry per thread (mask plus value bytes); worker-thread exit drops it
//! with the thread. `RefCell` borrows are scoped inside `try_with` and map
//! borrow/access failure to fail-closed errors; no references escape.

use sha2::{Digest, Sha256};

use super::{HISTORY_EVENT_CAP, NativeObservation, ObservationError};
use crate::rl::actions::BASELINE_ACTIONS;
use crate::rl::events::{EventArena, visible_to_actor};

thread_local! {
    /// Bounded one-entry legal-mask memo: `(mask copy, canonical value bytes)`.
    static LEGAL_MASK_MEMO: std::cell::RefCell<Option<(Vec<bool>, Vec<u8>)>> =
        const { std::cell::RefCell::new(None) };
}

// Boolean array fragments come from the same canonical authority; replacing
// false values with its true atom preserves every CURRENT legal bit. Width
// and framing fail closed so a serializer/layout change cannot corrupt IDs.
fn canonical_legal_atoms(mask: &[bool]) -> Result<Vec<u8>, ObservationError> {
    static FRAGMENTS: std::sync::LazyLock<Result<(Vec<u8>, Vec<u8>, usize), String>> =
        std::sync::LazyLock::new(|| {
            let canon = |v: &serde_json::Value| {
                hydra_feed::canon::canonical_bytes_value(v, "rl:observation-identity")
                    .map_err(|e| format!("legal fragment canonicalization failed: {e:?}"))
            };
            let zero = canon(&serde_json::Value::Bool(false))?;
            let one = canon(&serde_json::Value::Bool(true))?;
            let blank = canon(&serde_json::Value::Array(
                vec![serde_json::Value::Bool(false); BASELINE_ACTIONS],
            ))?;
            let stride = zero.len() + 1;
            if blank.len() != BASELINE_ACTIONS * stride + 1
                || blank.first() != Some(&b'[')
                || blank.last() != Some(&b']')
            {
                return Err("canonical legal array framing changed".to_string());
            }
            Ok((blank, one, stride))
        });
    let (blank, one, stride) = FRAGMENTS
        .as_ref()
        .map_err(|e| ObservationError(e.clone()))?;
    if mask.len() != BASELINE_ACTIONS {
        return Err(ObservationError(format!(
            "canonical legal array width must be {BASELINE_ACTIONS}"
        )));
    }
    let mut bytes = Vec::with_capacity(blank.len());
    let mut cursor = 0;
    for (index, &legal) in mask.iter().enumerate() {
        if legal {
            let start = 1 + index * stride;
            bytes.extend_from_slice(&blank[cursor..start]);
            bytes.extend_from_slice(one);
            cursor = start + stride - 1;
        }
    }
    bytes.extend_from_slice(&blank[cursor..]);
    Ok(bytes)
}

/// Feed the CURRENT `legal_mask` value bytes to `hasher` via the memo.
///
/// Hit path compares the full CURRENT mask contents before reusing the cached
/// bytes; miss path derives CURRENT bytes through the authority-derived atom
/// fragments (never an old observation or its digest) and replaces the entry.
/// Borrow scopes never escape; every failure is fail-closed.
fn feed_legal_mask_bytes(hasher: &mut Sha256, mask: &[bool]) -> Result<(), ObservationError> {
    let hit = LEGAL_MASK_MEMO
        .try_with(|cell| -> Result<bool, ObservationError> {
            let borrowed = cell
                .try_borrow()
                .map_err(|_| ObservationError("legal mask memo borrow failed".to_string()))?;
            match borrowed.as_ref() {
                Some((cached_mask, cached_bytes))
                    if cached_mask.len() == mask.len() && cached_mask.as_slice() == mask =>
                {
                    hasher.update(cached_bytes);
                    Ok(true)
                }
                _ => Ok(false),
            }
        })
        .map_err(|_| ObservationError("legal mask memo access failed".to_string()))??;
    if hit {
        return Ok(());
    }
    let bytes = canonical_legal_atoms(mask)?;
    hasher.update(&bytes);
    LEGAL_MASK_MEMO
        .try_with(|cell| {
            let mut borrowed = cell
                .try_borrow_mut()
                .map_err(|_| ObservationError("legal mask memo borrow failed".to_string()))?;
            *borrowed = Some((mask.to_vec(), bytes));
            Ok::<(), ObservationError>(())
        })
        .map_err(|_| ObservationError("legal mask memo access failed".to_string()))??;
    Ok(())
}

/// Sorted root keys in byte-lexicographic order (== UTF-16BE for fixed ASCII,
/// matching `serde_jcs` object key sorting). Includes `visible_history`.
const SORTED_ROOT_KEYS: [&str; 35] = [
    "action_table_hash",
    "actor",
    "actor_can_riichi",
    "actor_can_tsumo",
    "actor_furiten",
    "concealed_hand",
    "dealer",
    "decision_id",
    "dora_indicators",
    "event_schema_hash",
    "game_id",
    "hand_number",
    "honba",
    "ippatsu_active",
    "kan_count",
    "legal_mask",
    "live_wall_tiles_remaining",
    "observation_schema_hash",
    "own_drawn_tile",
    "packet_boundary_hash",
    "pending_declaration_discard",
    "phase",
    "riichi_states",
    "riichi_sticks",
    "round_index",
    "round_wind",
    "rules_hash",
    "rules_id",
    "scores",
    "seat_winds",
    "sequence",
    "turn_actor",
    "visible_discards",
    "visible_history",
    "visible_melds",
];

/// Hash CURRENT observation fields with cached history subbytes.
///
/// Rebuilds every non-history value from the live `obs` (same constructions
/// as `identity_json` via `identity_base_map`, plus memo-fed CURRENT legal
/// bytes so the full-width legal `Value` array is never allocated), streams the
/// sorted root plus the cached canonical event array to a fresh `Sha256`
/// with standard framing. Fails closed on over-cap history (no truncation),
/// missing arena index, missing cached subbytes, actor-invisible history, or
/// unsafe integers via the canonical guards. The full root is rehashed from
/// scratch on every call on both gates; only the `legal_mask` value byte
/// string may come from the content-verified memo.
pub(crate) fn hash_current_via_cache(
    obs: &NativeObservation,
    arena: &EventArena,
) -> Result<[u8; 32], ObservationError> {
    if obs.history_indices.len() > HISTORY_EVENT_CAP {
        return Err(ObservationError(format!(
            "visible history {} exceeds model bucket cap {HISTORY_EVENT_CAP}; rows are never truncated",
            obs.history_indices.len()
        )));
    }
    // CURRENT non-history fields; rebuilt on every gate call, never reused.
    let fields = obs.identity_base_map();
    let mut hasher = Sha256::new();
    hasher.update(b"{");
    for (pos, key) in SORTED_ROOT_KEYS.iter().enumerate() {
        if pos > 0 {
            hasher.update(b",");
        }
        // Fixed ASCII keys need no escaping; `"key":` matches JCS key bytes.
        hasher.update(b"\"");
        hasher.update(key.as_bytes());
        hasher.update(b"\":");
        if *key == "visible_history" {
            // Standard JCS array framing over immutable cached subbytes.
            hasher.update(b"[");
            for (elem, index) in obs.history_indices.iter().enumerate() {
                let event = arena.get(*index).ok_or_else(|| {
                    ObservationError(format!("history index {index} outside arena"))
                })?;
                if !visible_to_actor(event, obs.actor) {
                    return Err(ObservationError(format!(
                        "history holds a {} event this actor may not see",
                        event.visibility.as_str()
                    )));
                }
                let cached = arena.cached_canonical_bytes(*index).ok_or_else(|| {
                    ObservationError(format!(
                        "missing cached canonical bytes for history index {index}"
                    ))
                })?;
                if elem > 0 {
                    hasher.update(b",");
                }
                hasher.update(cached);
            }
            hasher.update(b"]");
        } else if *key == "legal_mask" {
            // CURRENT legal bytes via the content-verified memo; the root
            // hash itself is still recomputed from scratch on every call.
            feed_legal_mask_bytes(&mut hasher, &obs.legal_mask)?;
        } else {
            let value = fields
                .get(*key)
                .ok_or_else(|| ObservationError(format!("identity field {key} missing")))?;
            // Same serializer/guards as the oracle path; one small per-field
            // allocation, no history rebuild.
            let bytes =
                hydra_feed::canon::canonical_bytes_value(value, "rl:observation-identity")
                    .map_err(|e| ObservationError(format!("observation canon failed: {e:?}")))?;
            hasher.update(&bytes);
        }
    }
    hasher.update(b"}");
    Ok(hasher.finalize().into())
}
