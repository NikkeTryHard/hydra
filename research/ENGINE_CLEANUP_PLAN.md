# Engine Cleanup Plan

Post-optimization audit findings. 18 items across hydra-engine and hydra-core.
Ordered by priority. Each item has scope, effort, and dependencies.

---

## Critical (fix before training pipeline)

### C1: Refactor state_3p/mod.rs step() (911 lines)

**Problem**: 3p variant still has the monolithic `_execute_step` (HashMap-based)
and no extracted handler methods. 4p was refactored but 3p was skipped.

**Fix**: Mirror the 4p refactor:
1. Delete `_execute_step`, route `step()` through `_execute_step_array`
2. Extract `_handle_discard`, `_handle_riichi`, `_handle_ankan`, `_handle_kakan`,
   `_handle_tsumo`, `_handle_wait_response`, plus `_handle_kita` (sanma-specific)
3. Replace remaining Vec allocs with stack arrays (chankan_ronners, deltas, etc.)

**Files**: `hydra-engine/src/state_3p/mod.rs`
**Effort**: Large (1-2 sessions). Same pattern as 4p refactor.
**Depends on**: Nothing.
**Verify**: `cargo test --release`, `cargo clippy --all-targets -- -D warnings`

---

### C2: Replace HashMap in state_3p GameState

**Problem**: `win_results: HashMap<u8, WinResult>` and
`last_win_results: HashMap<u8, WinResult>` still use HashMap. 4p uses arrays.

**Fix**: Replace with `[Option<WinResult>; 3]` (NP=3 for sanma).
Update all `.get()`, `.insert()`, `.clear()` call sites.

**Files**: `hydra-engine/src/state_3p/mod.rs`
**Effort**: Small. Mechanical find-replace.
**Depends on**: Nothing (can parallelize with C1).
**Verify**: `cargo test --release`

---

### C3: Fill reserved encoder channels (9 zeros)

**Problem**: Ch 74-79 (suji context) and Ch 82-84 (tenpai hints) are
permanently zero. 10.6% of input tensor is wasted.

**Fix**: Two options:
- Option A: Fill Ch 74-79 with additional suji features (half-suji,
  no-chance-suji, matagi-suji). Fill Ch 82-84 with naive tenpai
  estimates (e.g., shanten==0 indicator per opponent).
- Option B: Remove the reserved channels, shrink to 76x34. Requires
  updating HYDRA_SPEC.md and all downstream model code.

**Decision needed**: Option A is better (no model architecture change).
**Files**: `hydra-core/src/encoder.rs`, `hydra-core/src/safety.rs`,
`research/HYDRA_SPEC.md`
**Effort**: Medium. Need to design the 9 features first.
**Depends on**: Architecture decision.
**Verify**: Golden encoder tests must be updated with new expected values.

---

### C4: Eliminate unwrap() in hydra-engine lib code (57 calls)

**Problem**: 57 unwrap()/expect() calls in non-test library code.
Crash risk in production. Mostly serde_json serialization in MJAI
logging and state assertions.

**Fix**: Replace each with:
- `serde_json::to_value(x).unwrap()` -> `serde_json::to_value(x)?`
  (requires changing return types to Result)
- `self.last_discard.unwrap()` -> checked access with error propagation
- For MJAI logging code (gated by skip_mjai_logging): acceptable to
  keep unwrap since it only runs in debug/replay mode. Add comment.

**Files**: `state/mod.rs` (17), `state_3p/mod.rs` (18),
`observation/python.rs` (8), `observation_3p/python.rs` (8),
`parser.rs` (4), `action.rs` (1), `replay/mjai_replay.rs` (4)
**Effort**: Medium. Some require signature changes for Result return.
**Depends on**: Nothing.
**Verify**: `cargo test --release`, `cargo clippy`


---

## High (code quality / maintainability)

### H1: Unify 4p/3p code duplication (~5,500 lines)

**Problem**: 9 file pairs with 85-95% identical code. state/mod.rs
(2273 lines) mirrors state_3p/mod.rs (2018 lines). Same for
legal_actions, event_handler, player, wall, observation, hand_evaluator.

**Fix**: Introduce `const NP: usize` generic on GameState, PlayerState,
WallState, and all methods. 4p = `GameState<4>`, 3p = `GameState<3>`.
Sanma-specific logic (kita, BaBei) behind `if NP == 3` branches.

**Files**: All 9 pairs listed in audit. ~18 files total.
**Effort**: Very large (multi-session). Highest-impact refactor.
**Depends on**: C1 (3p needs handler extraction first).
**Verify**: All 210 tests pass. Identical behavior verified by
deterministic replay comparison before/after.

---

### H2: Add rustdoc to all pub items in hydra-engine (306 functions)

**Problem**: Zero pub functions in the entire engine crate have `///` docs.
Violates AGENTS.md rule: "All pub items must have `///` docs."

**Fix**: Add `///` doc comments to all 306 pub functions.
Follow RFC 1574: imperative mood first line, then blank line,
then extended description. Group by module, delegate per-file.

**Files**: Every .rs file in hydra-engine/src/
**Effort**: Large but mechanical. Can delegate per-file.
**Depends on**: Nothing.
**Verify**: `cargo doc --no-deps` builds without warnings.

---

### H3: Add rustdoc to missing pub items in hydra-core (28 items) -- DONE

**Problem**: action.rs constants (16-30), HydraAction methods,
DirtyFlags consts, tile.rs AKA constants missing `///` docs.

**Fix**: Add `///` to each. Follow existing style in the file.

**Files**: `hydra-core/src/action.rs`, `hydra-core/src/tile.rs`,
`hydra-core/src/encoder.rs`
**Effort**: Small. ~30 minutes.
**Depends on**: Nothing.
**Verify**: `cargo doc --no-deps`

---

### H4: Break up remaining mega-functions (32 over 100 lines)

**Problem**: 32 functions exceed 100 lines. Worst offenders:
step() 911 lines (3p), apply_log_action 465 lines,
_handle_wait_response 417 lines, calculate_yaku 328 lines.

**Fix**: Extract sub-handlers from each. Examples:
- _handle_wait_response -> _handle_ron_wins + _handle_call_claim + _handle_all_pass
- calculate_yaku -> check_yakuhai + check_straights + check_special
- apply_log_action -> handle_discard_event + handle_call_event + etc.

**Files**: state/mod.rs, state_3p/mod.rs, yaku.rs, event_handler.rs
**Effort**: Large. Pure refactor, no logic changes.
**Depends on**: C1 (3p handler extraction).
**Verify**: All tests pass. No behavior change.

---

## Medium (known tech debt)

### M1: Remove dead_code allows (4 items) -- DONE

**Problem**: push_active_player (4p+3p), set_claims_from_vec (4p),
ID_NUKIDORA constant. Left from refactor.

**Fix**: Delete the functions/constants. They are unused.
If needed later, re-add.

**Files**: state/mod.rs, state_3p/mod.rs, yaku.rs
**Effort**: Tiny.
**Depends on**: Nothing.

---

### M2: Resolve TODO comments (2 items) -- DONE

**Problem**:
- mjsoul_replay.rs:213 -- "TODO: Parse header for rule"
- state/mod.rs:1816 -- "TODO: Delete 4, 5, 3"

**Fix**: Either implement or remove with explanation.

**Files**: replay/mjsoul_replay.rs, state/mod.rs
**Effort**: Tiny.

---

### M3: Document batch_encoder as future infrastructure -- DONE (already had docs)

**Problem**: batch_encoder module is never imported. Looks like dead code
but is actually infrastructure for hydra-train.

**Fix**: Add module-level doc comment:
`//! Batch observation encoding for training pipeline (hydra-train).`
`//! Not currently used -- will be integrated during Phase 3 training.`

**Files**: hydra-core/src/batch_encoder.rs
**Effort**: Tiny.

---

### M4: Fix suji channel spec discrepancy

**Problem**: HYDRA_SPEC.md diagram shows Ch 71-79 as "Suji" (9 channels)
but encoder only fills 3 (71-73). Channels 74-79 are zeros.

**Fix**: Update HYDRA_SPEC.md channel table to say:
"Ch 71-73: Suji (1 per opponent), Ch 74-79: Reserved suji context"

**Files**: research/HYDRA_SPEC.md
**Effort**: Tiny.
**Depends on**: C3 decision (if filling channels, update spec after).

---

### M5: Convert WinResult.yaku to fixed array

**Problem**: `yaku: Vec<u32>` is heap-allocated. Max ~15 yaku per hand.
Allocated on every win (once per round end).

**Fix**: `yaku: [u32; 16], yaku_count: u8`. Update all consumers:
yaku.rs, score.rs, hand_evaluator.rs, observation/ files.

**Files**: types.rs + ~8 consumer files
**Effort**: Medium. Cross-file type change.
**Depends on**: Nothing.
**Verify**: cargo test --release

---

### M6: Convert 4p PlayerState.kita_tiles to fixed array -- DONE (removed entirely)

**Problem**: `kita_tiles: Vec<u8>` in 4p PlayerState is dead code.
Never written to in 4-player games. 3p already uses `[u8; 4]`.

**Fix**: Change to `[u8; 4]` + `kita_count: u8` matching 3p variant.
Or remove entirely from 4p PlayerState (kita is sanma-only).

**Files**: state/player.rs, state/mod.rs (reset_round clear)
**Effort**: Tiny.
**Depends on**: Nothing.

---

## Low (cosmetic / polish)

### L1: Replace clippy::too_many_arguments with builder patterns

**Problem**: 8 `#[allow(clippy::too_many_arguments)]` across
Observation constructors, WinResult, GameRule.

**Fix**: Introduce builder structs or config objects.
E.g., `ObservationBuilder` instead of 12-arg constructor.

**Files**: observation/mod.rs, observation_3p/mod.rs, types.rs, rule.rs
**Effort**: Medium.
**Depends on**: Nothing.

---

### L2: Remove phantom round_end_scores field -- DONE

**Problem**: `round_end_scores: Option<Vec<i32>>` is never set to
`Some(...)` in 4p. Dead field.

**Fix**: Remove from GameState. Grep for all references, delete.

**Files**: state/mod.rs
**Effort**: Tiny.

---

### L3: Remove phantom last_error field from training path

**Problem**: `last_error: Option<String>` only set in validated
`step()`, never in `step_unchecked()`. Dead in training.

**Fix**: Gate behind a feature flag or move to a separate
ValidationState struct. Or just leave it (8 bytes, harmless).

**Files**: state/mod.rs
**Effort**: Tiny.

---

### L4: Stack-allocate Observation structs

**Problem**: Observation/Observation3P use Vecs for hands, melds,
discards, waits, dora_indicators, tsumogiri_flags. These are the
Python-facing API structs.

**Fix**: Not needed for training (hydra-core uses ObservationRef).
Only relevant if Observation is used in Rust-side replay/analysis.
Low priority -- the Python boundary needs Vecs anyway for PyO3.

**Files**: observation/mod.rs, observation_3p/mod.rs
**Effort**: Medium. Low value.
**Depends on**: Nothing.

---

## Execution Order

### Wave 1 (parallel, no dependencies)
- M1: Remove dead_code allows (tiny)
- M2: Resolve TODOs (tiny)
- M3: Document batch_encoder (tiny)
- M6: Convert 4p kita_tiles (tiny)
- H3: Add rustdoc to hydra-core (small)
- L2: Remove phantom round_end_scores (tiny)

### Wave 2 (parallel, no dependencies)
- C1: Refactor state_3p step() (large)
- C2: Replace HashMap in state_3p (small, can merge with C1)
- C4: Eliminate unwrap() in engine (medium)
- M4: Fix suji spec discrepancy (tiny)
- M5: Convert WinResult.yaku to array (medium)

### Wave 3 (depends on C1)
- H1: Unify 4p/3p duplication (very large)
- H4: Break up remaining mega-functions (large)

### Wave 4 (depends on architecture decisions)
- C3: Fill reserved encoder channels (medium)
- H2: Add rustdoc to all hydra-engine (large)

### Wave 5 (polish)
- L1: Builder patterns for clippy args (medium)
- L3: Gate last_error field (tiny)
- L4: Stack-allocate Observation structs (medium, low value)