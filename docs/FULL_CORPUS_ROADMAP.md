# Hydra2 full-corpus road — synthesized plan (review before execution)

Status inputs: Python wall-less replay proven (78/84 probe, smoke green, 977+55 lanes);
single-pass collapse mid-build; converter proven lossless (no reconversion, 2.5M census);
dbs immutable; torch transformer stays.

## Order (recommended)

1. LAND single-pass (in flight) — one engine walk + thin derivation, byte-parity gate.
2. SCALE-UP S1 single-year training — starts as soon as (1) lands.
3. RUST cutover (vendor `tools/hydra2-replay-rs`, replay-only, both-path parity) — last.

SKIPPED 2026-09-07: seed→wall exact walls. Reason: walls appear in neither BC inputs
(actor never sees them) nor labels (played moves); wall-less replay is the permanent
path and filename-identity + exact-hash dedup stays the disjoint mechanism. Design
retained in agent transcripts (SeedWallPlan) if a future belief/rollout package needs it.

DELETE POLICY 2026-09-08 (operator intent + workstation law): Python counterparts go
once proven worse than Rust — but each deletion still needs its exact enumerated path
list shown with parity evidence, approved by the operator (one word suffices),
then removed with a JSONL audit-log entry. No blanket auto-delete (XFS deletions are
unrecoverable); no silent removals; no deletions bundled inside other work.

STOCK-ENGINE LAW 2026-09-08 (operator, absolute): riichienv 0.4.8 stays unmodified —
no forks, no perf patches inside the engine (hydra1 paid correctness risk for that).
All performance work happens AROUND it: orchestration, batching, buffers, handoff,
validation placement. Engine interns are out of scope for every package.

## Package A — Seed→wall (spec highlights; full: SeedWallPlan report)

- Algorithm (sourced): strip scheme prefix → base64/hex to 624×u32 LE → MT19937
  init_by_array → per kyoku (INIT order, renchan consumes): 288 genrand → 9×SHA-512
  → 144 words → Fisher-Yates yama[136] → dice rnd[135]%6, rnd[136]%6. Three seed eras:
  A legacy per-INIT (default-reject, semantics unknown), B hex, C base64. Tile ids 1:1
  (aka 16/52/88 match).
- Mapping: yama → deal-order vec into existing `reset(wall=)` + `_reopen_hand` seam;
  anchors: INIT seed[5] == MJAI dora_marker == v[131]; tehais multiset match; ordered
  70-draw prefix; conservation sorted(v)==0..135.
- Verification: deal-match per kyoku BEFORE rows; mismatch → quarantine class
  (seed-mismatch/seedless/era-a-unverified) + wall-less fallback; HALT if mismatch >0.1%.
- Slices: S1 stdlib `seed_wall.py` + vendored goldens (b327da61, 17d39cdb) + mapping proof
  on ≥3 real games; S2 read-only DB extractor + content-hashed seed manifest (join key =
  corpus filename stem == logs.id) + per-game verify; S3 engine wiring via existing walled
  path (no forks); S4 full-corpus audit (~2.465M) + cutover seeded-first, wall-less remainder.
- Fallback: 2009 pre-07-22 (~994), all 2026, Era A, sanma → wall-less/quarantine as specified.

## Package B — Scale-up runbook (staged; calibration first)

- Math: minibatch 32 rows/update; model ~2.06M params; micro4 fits 12GB; bottleneck is
  Python expand, not GPU. S0 smoke DONE.
- FIRST ACTION: calibration run (S1 500 updates) publishing updates/sec + rows/sec +
  quarantine reconciliation; all time estimates hinge on it (assumption 2–5 upd/s).
- S1 single-year 2024 (147k train, 10k updates): gates overlap0, quarantine ≤10% warn/12% fail,
  loss-down/topk-up, one ckpt resume round-trip identical.
- S2 multi-year 2022–24 (416k train, 50k): + no val regression 3 evals, ECE flat, manual
  selection dry-run only, quarantine ±2pp.
- S3 full 2.01M train (200k, extend 500k on pass): + manifest pinned, keep_last pruning
  verified, promotion draft + ledger. Abort patterns checked every ckpt (fail-closed,
  collapse, generalisation, drift, infra). Arena gates §8 (frozen pool, N30 one peek,
  PromotionRecord).
- Configs: BC-only throughout; bf16 only as separate never-compared id; one id one spec.

## Package C — Rust cutover (last)

- Vendor minimal wall-less core to `tools/hydra2-replay-rs` (allowlist: event vocab,
  tile codec, tehais-install, stream/decisions/sink shapes, PyO3 skeleton). NO hydra GameState,
  NO encoder math, NO MjaiSample. Replay-only; encode stays Python.
- Two-phase FFI: rows-as-JSON first, pinned tensors deferred with new schema identity.
- Parity: DecisionRow-JSON level, pinned diff classes only (game id, copy fold, refit, marker).
- Slices: core CLI parity → PyO3 stream → parquet integration → (deferred) pinned encode
  with per-device qualification.

## Cross-cutting rules

Firewalls intact; fail-closed quarantine with named reasons; D-017 (no raw samples);
Pixi sole authority; one id one spec; bf16 never compared; no reconversion; dbs/corpus read-only.
Quarantine budget ~8% (7 multi-ron + 1 info-absent); multi-ron support is a separate ticket.
