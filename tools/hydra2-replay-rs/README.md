# hydra2-replay-rs — wall-less replay driver (slice 4)

Vendored MINIMAL wall-less replay core + dump CLI + DecisionRow-JSON
emission + frozen row hashes, with parity numbers vs the Python oracle
(`log_replay.replay_game`, wall-less v1).

## Provenance (read-only references, never copied wholesale)

| Item | Value |
|---|---|
| Repo HEAD | `9a0359e` (Docs: add root AGENTS.md workstation conventions) |
| `src/hydra2/engines/riichienv/log_replay.py` | sha256 `92aac4dbef77bbf51d22da91aee91e18aae58af473da38db0e7af60913729e68` (untracked working-tree file; ORDER reused) |
| `src/hydra2/engines/riichienv/tiles.py` | sha256 `7b442fb903051050ad78540e7365908c28bbd6391913d6b5f4a96ac06de77b4f` (codec rules) |
| `src/hydra2/data/parquet.py` | sha256 `faf5118c792c4cad0e2f4cb7845c8ddd76f753c6f3eff2f752a3d75ad18e4d5d5d` (`ACTOR_FIELDS` discipline) |
| `configs/contracts/action_table_v1.json` | file sha256 `4a66338f6e5d9bfd8432dbde75f7941acfcaa308358740c55e9ec23649e42214`, payload digest `sha256:7b55693428384713f6a6ab7292f57657259c2ca9f05139944d4c8c6197ae76e8` (6792 templates, loaded at runtime, never re-generated) |
| Pinned simulator | `riichienv 0.4.8` (python 3.12.14 pixi env) |
| Parity target | v1 log path, `SIM_DERIVATION_MARK=sim-replay-wall-less-v1` |
| Probe | 84-probe train split of `sample-tenhou` (seed 0, `smoke100.yaml`); oracle 78 ok / 6 double-ron quarantine |

## Oracle pin (fail closed: moving oracle voids parity)

`scripts/freeze_row_hashes.py` records sha256 of every Python oracle input
into the frozen-hash fixture (`tests/fixtures/frozen-row-hashes.json`,
`metadata.oracle_pins`) and `check` REFUSES with
`oracle moved - re-pin and re-run` (exit 3) when any pin differs. Slice 3
pins (working tree, verified unchanged since the probe materialized):

| Oracle input | sha256 |
|---|---|
| `src/hydra2/engines/riichienv/log_replay.py` | `92aac4dbef77bbf51d22da91aee91e18aae58af473da38db0e7af60913729e68` |
| `src/hydra2/engines/riichienv/adapter.py` | `9cbf0d35773626c906cb30f17e469bfafae86c86d1094296964027e5edad61d7` |
| `src/hydra2/engines/riichienv/actions.py` | `51a1c178c4e799786819897d9411323f000bcc620f8241e993d0420f1ce86f8c` |
| `src/hydra2/engines/riichienv/events.py` | `5f1460461a64fc3fe88d38c16f17554b23119a7430f6e9ceaa4ed80ec49015f6` |
| `src/hydra2/engines/riichienv/tiles.py` | `7b442fb903051050ad78540e7365908c28bbd6391913d6b5f4a96ac06de77b4f` |
| `src/hydra2/engines/riichienv/state.py` | `e7b37584fb3b7a2ad2251d9518d91b0b9182685ec0e53cb2e3a33d9490debf1e` |
| `src/hydra2/engines/riichienv/identity.py` | `da47b6a3ad29362d93617550988e4db49206ea6c9a57ba2a6645c2cf4d6fa79c` |
| `src/hydra2/data/parquet.py` | `faf5118c792c4cad0e2f4cb7845c8ddd76f753c6f3eff2f752a3d75ad18e4d5d` |
| `src/hydra2/models/encoder.py` | `eb3f8723117ca753096a77827455b572bf9588a13b933a5d448fbc9bfbc2c098` |
| `configs/contracts/action_table_v1.json` | `4a66338f6e5d9bfd8432dbde75f7941acfcaa308358740c55e9ec23649e42214` |

Reference order reused, implementations written fresh: MJAI event
vocabulary, Tenhou tile codec, start_kyoku tehais-install + countdown wall
(136-52 dealt, 14 dead, actor-masked views), stream/decisions/implicit-pass
handling, tile/scores shapes, sink-trait shape (rows sink + quarantine
sink), PyO3 bridge SKELETON only (no torch, no tensors — out of scope for
slice 1).

PROHIBITED and absent: hydra GameState wholesale, encoder math,
MjaiSample/46-action targets, parquet writes, privileged fields
(seat-filtered only; privileged unrepresentable in `ReplayRow`).

## Layout

- `src/tile.rs` — physical id (0..135) <-> MJAI string codec, aka
  16/52/88, red normalization, copy pools, terminal/honor classes.
- `src/mjai_event.rs` — framed event vocabulary + strict framing rules.
- `src/stream.rs` — game identity, wall-bearing rejection, take ledger
  (tehais seats 0..3, live draws, rinshan draws), tracked hands/rivers.
- `src/decisions.rs` — log-order walker, chosen-action construction,
  thin window predicate, mask approximation, win-shape proxy.
- `src/decision_ids.rs` — positional id formats `{game}:h{idx:02}` /
  `{game}:d{seq:04}` + strict parsers (5 unit tests).
- `src/hydra2_row.rs` — `ReplayRow` / `ChosenAction` / `Quarantine` schema
  + `to_decision_json()` (EXACTLY the 13 `ACTOR_FIELDS`, `actor_observation`
  as a JSON string, `wall_id` never bound, `derivation_hash` binding
  `sim-replay-wall-less-v1`) + RFC 8785 canonical bytes (5 unit tests).
- `src/py_stream.rs` — Slice-4 PyO3 JSON-rows handoff (5 unit tests).
- `tests/parity_84.rs` — env-gated decision-for-decision diff.
- `tests/fixtures/frozen-row-hashes.json` — 42553 frozen oracle hashes
  (compact mode, strict) + 10 oracle pins (see below).
- `scripts/freeze_row_hashes.py` (repo root) — `freeze` / `check` the
  per-decision hashes with the CLOSED allowlist
  (`wall_game_id,copy_fold,refit_number,wall_less_marker` only) and the
  oracle-pin fail-closed gate.

## CLI

```sh
cargo run -p hydra2-replay-rs --bin hydra2_replay_dump -- \
  --inputs /tmp/replay-s1/inputs \
  --out /tmp/replay-s1/rust-rows.jsonl \
  --action-table configs/contracts/action_table_v1.json \
  --manifest /tmp/replay-s1/manifest.json
```

Quarantines land in `<out>.quarantine.json` (override with
`--quarantine-out`). The CLI exits 0 with the quarantine sidecar always
written, even when every game replays clean. Inputs may be plain `.jsonl`
or compressed `.jsonl.zst` / `.jsonl.gz` (zstd+flate2 decode only).

## Row schema (one JSON object per decision row)

| Field | Meaning |
|---|---|
| `game_id` | Log game id (`game-<sha12(object_id)>` fallback) |
| `round_id` | `{game_id}:h{round_idx:02}` |
| `decision_id` | `{game_id}:d{seq:04}` (positional over every decision) |
| `seat` | Acting seat 0..3 |
| `phase` | `draw_decision` / `discard_response` (`kan_response` for chankan ron) |
| `turn_actor` | Draw rows: actor; claim/ron rows: discarder |
| `chosen` | `{kind, tile?, called?, consumed?, source?}` (physical ids) |
| `chosen_action_id` | Canonical table id, or `null` + `chosen_unresolved` reason |
| `legal_mask` | Sorted true action-table ids (APPROXIMATION, see below) |
| `history_kinds` | Per-seat visible envelope kinds in emission order |
| `concealed_hand` | MJAI strings, ascending physical-id order, drawn excluded |
| `drawn_tile` | Most recent drawn tile string (persists until next draw) |
| `dora_indicators` | Five slots of indicator strings, `null` tail (initial marker never revealed in v1) |
| `wall_remaining` | `max(0, 70 - draws_this_kyoku)` |

## DecisionRow-JSON envelope (Slice 3)

`ReplayRow::to_decision_json(provenance)` emits EXACTLY the 13
`ACTOR_FIELDS` (`src/hydra2/data/parquet.py` order): `game_id`, `round_id`,
`decision_id`, `seat`, `source_object_id`, `split`, `rules_hash`,
`adapter_hash`, `observation_hash`, `action_table_hash`, `derivation_hash`,
`actor_observation`, `chosen_action_id`. `actor_observation` rides as a JSON
string (the parquet column type); `derivation_hash` binds
`sim-replay-wall-less-v1` with `wall_digest: None` always (`wall_id` never
bound); `chosen_action_id` is `null` exactly when `chosen_unresolved` is
set. Provenance hashes pass through from the oracle materializer. Rust never
writes parquet — the envelope exists so whole-row identity is hashable per
`decision_id` (next section).
## Parity method

Oracle material (Step 0, `/tmp` scratch, repo code only):
`GameStream` + `build_manifest` + `_split_ratios` train split, then
`replay_game` rows dumped to `/tmp/replay-s1/oracle-compact.jsonl`.
`tests/parity_84.rs` joins Rust rows to oracle rows on `decision_id` and
compares seats, phases, chosen ids/kinds, concealed/drawn/dora strings,
wall, history kinds, and mask bits, plus the quarantine game sets. The test
first asserts scratch discipline (`HYDRA2_REPLAY_PROBE` under the platform
temp dir, `HYDRA2_ARTIFACT_ROOT` outside the raw corpus roots), which runs
even when the probe env is unset.

## Parity status (frozen after 3 fix iterations; Slice 3 re-verified)

Oracle: 78 ok / 6 double-ron quarantine, 42553 rows.
Rust: 78 ok / 6 double-ron quarantine (identical game sets), 42553 rows.
Same run: `cargo nextest run --manifest-path tools/hydra2-replay-rs/Cargo.toml`
(23 tests: 13 carried + 10 Slice 3 unit) plus the env-gated `parity_84` join
described above.

| Compared field | Exact | Rate |
|---|---|---|
| decision ids / seats / phases (join keys) | 42553/42553 | 100% |
| chosen action ids | 42553/42553 | 100% |
| chosen kinds | 42553/42553 | 100% |
| concealed-hand strings (id order) | 42553/42553 | 100% |
| drawn-tile strings | 42553/42553 | 100% |
| dora-indicator strings | 42553/42553 | 100% |
| wall remaining | 42553/42553 | 100% |
| history envelope kinds | 42440/42553 | 99.73% |
| legal masks fully exact | 40132/42553 | 94.33% |
| rows fully identical (all fields) | 40019/42553 | 94.06% (hash-verified, was ~40019) |
| per-game | 0/78 fully identical, 78/78 with diffs (all diffs are mask/history classes below), 6 quarantined |
| quarantine games | 6/6 identical (`double-ron` on the same games) | 100% |
| unresolved chosen ids | 0 | — |
| DecisionRow-JSON envelope | 13/13 `ACTOR_FIELDS`, `actor_observation` string, no privileged keys (5 unit tests) | — |
| frozen-hash gate | 40019/42553 identical, 2534/2534 mismatches in mask/history classes, 0 missing / 0 extra | — |
| oracle-pin gate | 10/10 pins match; `check` refuses on drift | — |

Frozen-hash verdict (Slice 3, strict allowlist-off):
`freeze --rows oracle-compact.jsonl` (42553 hashes) vs `freeze --rows
rust-rows.jsonl`, then `check`: identical 40019/42553, mismatch 2534
(= 2421 mask-inexact rows + 113 history-gap rows, union exact, zero exotic
dimensions: no seat/phase/chosen/hand/drawn/dora/wall diff on any mismatch
row), missing 0, extra 0. Fixture:
`tests/fixtures/frozen-row-hashes.json` (4.3 MB).

### Residual classes (reported with counts, never chased)

| Code | Count | Mechanism |
|---|---|---|
| `mask-missing/riichi-candidates` | 2966 bits | Engine tenpai/win evaluation offers `riichi_discard` declaration candidates the wall-less driver cannot derive; bits appear only when chosen. |
| `mask-extra/kuikae-discards` | 1675 bits, post-claim rows | Live engine withholds kuikae-illegal discards after melds (pinned known variation: kuikae discards are log-only). |
| `mask-extra/agari-kan-width` | 452 bits on meld-free rows (413 tsumo-row, 39 ankan-row) | Win/kan decisions narrow engine offers; the driver approximates the draw offer. |
| `mask-extra/chi-variants` | 280 bits, claim rows | String-distinct chi-variant enumeration vs replay-engine offers (pinned known variation: extra chi variants). |
| `mask-extra/closed-kan` | 10 bits | Closed-kan offer rule gaps (engine-owned). |
| `mask-extra/kyushu-proxy` | 4 bits | Thin nine-terminals proxy over-fires vs the engine offer (pinned known variation: kyushu offers are live-engine-only). |
| `history/call-window-gap` | 113 rows (0.27%), all oracle-longer-by-one | Engine-only windows from win evaluation (ron options incl. riichi shape-proxy gaps and chankan) beyond the log-faithful thin predicate, which is an exact port of the oracle's own `_thin_window_open`. Every logged claim still produced its row (100% join), so no claim window was missed; zero reverse diffs. |
| `quarantine/double-ron` | 6 games both sides | Single-winner pipeline quarantines double ron (pinned). |

Slice 3 re-verified: every count above reproduces exactly on the current
tree (mask missing 2966 all `riichi_discard`; extra 2421 = 1675 post-claim
+ 452 clean + 280 chi + 10 closed-kan + 4 kyushu; 113 oracle-longer-by-one
history rows, zero reverse). The frozen-hash mismatch set (2534 rows)
equals the mask-inexact (2421) union history-gap (113) row sets exactly.

Fix history: (1) test-side string-order comparison + copy-identity analysis;
(2) turn-scoped drawn rule, true-take post-reach tiles, collapsed
declaration tiles, offset-None window pass, ron mask contents;
(3) one-discard-per-string offers, forced-pair post-reach masks.
Copy-identity folding holds throughout (pool-first reporting, wall-less
marker `sim-replay-wall-less-v1`).

## Quarantine reason codes

`framing`, `wall-bearing`, `bare-dora`, `double-ron`, `unknown-event`,
`tile-conservation` (pool overuse), `turn-order`, `claim-no-offer`,
`draw-past-wall`, `kyushu-ambiguous`, `unmapped-ryukyoku-reason`,
`action-id-unresolved` (row-level, still emitted).

## PyO3 handoff (Slice 4, Phase A: JSON rows, no tensors/buffers)

`src/py_stream.rs` exposes `PyHydra2ReplayStream{inner:
Mutex<Option<ReplayStream>>}` (`import hydra2_replay_rs`; PyO3 0.28.3,
the only new dependency, matching the hydra workspace pin):
`open(data_dirs, batch, workers, queue, split, spec_hash)` (open-once,
sorted file discovery, `workers > 0` fails closed until the Slice-7 pool),
`next_into_json(buf_ptr, capacity)` (up to `batch` newline-delimited
decision-JSON rows into the caller buffer; exhaustion is `rows == 0 &&
games_consumed == 0`; a too-small buffer drains nothing),
`stats()` (`open_count == 1` on a live stream, plus `games_ok` /
`games_quarantined` / `rows_out`), `quarantines()` (Slice-3 reason codes,
load order), and idempotent `close()`. `py.detach` wraps every blocking
section; poison/closed errors mirror the hydra precedent. The payload is
`to_decision_json` output (exactly the 13 `ACTOR_FIELDS`), gated per row
by the `FORBIDDEN_IN_ACTOR` envelope check; the Python stub
(`src/hydra2/training/rust_stream.py`) re-checks the envelope,
`FORBIDDEN_REPLAY_KEYS`, and the split/spec-hash binding per batch, and
asserts `HYDRA2_ARTIFACT_ROOT` outside the raw roots. Provenance:
`source_object_id` is the file stem, `split` is the open split,
`rules_hash`/`adapter_hash` bind the operator `spec_hash` (Slice 5
replaces this with RunConfig-digest plumbing), `action_table_hash` is
sha256 of the loaded table (`HYDRA2_REPLAY_TABLE` when set, else the
baked v1 artifact). Proven serially by
`tests/unit/test_rust_stream_wp14.py` (5 tests: drain + envelope +
stats + quarantine taxonomy, byte-identical two-pass determinism,
buffer-too-small, open-args, artifact-root gates) and the 5 in-crate
`py_stream::stream_tests`.

## Slice 6 — engine-answer bridge + v1/v2 selectable backend

`src/engine.rs` is the frozen narrow engine-out protocol (seat-filtered
`SeatView` in, `DrawOffer` / `ClaimOffer` / `RonOffer` + window / furiten /
win / kyushu / dora answers out). The stock engine stays the sole rules
authority (queried spec-first, never patched); the native backend
implements its answers rule-by-rule with corpus pins on every rule.

Backends (`EngineVersion`, `--engine-version v1|v2` on the dump CLI,
`walk_game_with_version` / `replay_game_text_with_version`):
- **v1** (frozen default): pinned drained-oracle semantics, bug-compatible
  (stale takes offered, partial chi, no kyushu). Full parity_84 identity:
  42553/42553 rows, seat/phase/chosen/concealed/drawn/dora/wall/history/
  mask all 100%, 78/78 games identical, same 6 double-ron (strict asserts
  in `tests/parity_84.rs`).
- **v2** (single-pass live-engine semantics): one-shot immediate filter
  after chi/pon claims (just-claimed called take + non-melded chi
  neighborhood, never pon/kan), complete-chi enumeration (max-held takes,
  kamicha-gated, even on pon rows), standard kyushu (first draw, no
  calls, 9+ distinct terminals/honors). V2 changes masks only (row
  counts + chosen ids identical to v1, pinned on the s4 good fixtures).
  Against the single_pass oracle: 98.4% rows identical; the residual is
  take-copy identity (live takes vs canonical takes on multi-copy
  claims/discards — S7 live-take scope), never rule shape.

`engine-desync` (with `game + kyoku + step` detail) is the fail-closed
net for every engine disagreement (shapeless logged wins, unoffered
kans, forced-pair violations). Pinned by
`tests/fixtures/s6/q-engine-desync.jsonl` + `tests/s6_engine.rs`
(7 tests: chi positions, max-held takes, kuikae neighborhood, kyushu
count, v1/v2 row+chosen preservation, desync quarantine).
