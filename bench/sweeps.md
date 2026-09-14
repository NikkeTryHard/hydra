# Feed-rate sweeps (P0-A baseline grid + P6 A/B rotation)

Baseline first: publish the blank-threshold artifact
(`$HYDRA2_ARTIFACT_ROOT/reports/feed-rate/<manifest>/<config>.json` with
per-run arrays + medians, `thresholds: null`) BEFORE any sweep. Sweeps never
re-baseline: every sweep row references the pinned manifest digest.

All commands via pixi (sole authority); never bare `python`/`pytest`:

```sh
# Canonical defaults (f10 decision-bearing primary; S1 grid inherits these,
# each row varies one axis only):
# `--leg stream --corpus f10 --chunk 32 --zstd 1 --B 32 --T 64
# --depth 2 --pin` (threads = affinity count), seed 7. F10 is the
# decision-bearing primary (real dahai/hora rows; decisions/sec
# non-vacuous); F2/F8 are framing-only legs (turn_advance, wall-less —
# emitted_games units).
```

## 1. Sweep grid (one axis at a time; hold the rest at the canonical
defaults above)

| Axis    | Values              | Flag               | Notes                                  |
|---------|---------------------|--------------------|----------------------------------------|
| zstd    | 1, 2, 3             | `--zstd` + rebuild | rebuild corpus per level (hash pinned) |
| chunk   | 1, 8, 32, 128       | `--chunk`          | microbatch slice width                 |
| B       | 32, 256, 1024       | `--B`              | ring probe + geometry row              |
| T       | 64, 256             | `--T`              | ring probe + geometry row              |
| leg     | expand, stream      | `--leg`            | expand=F1 rows/s; stream=F2/F8 games/s |

Headline metric: **warm-median decisions/sec** (cold reported alongside,
never averaged with warm). Decisions = expanded rows (expand leg) or
emitted games (stream leg); the artifact `unit` field states which.

Saturated leg: F11 (`bench/corpus/f11-saturated/`, decision-primary like
F10) is the steady-state choice; F10 (`bench/corpus/f10-decisions/`) is
the baseline. Both emit real dahai/hora rows, so decisions/sec stays
non-vacuous either way (see `bench/corpus_entries.json` F10/F11 entries).

## 2. sonic-vs-simd A/B rotation (ingest codec candidates)

A/B rotation applies to the Rust ingest path (Wave P1+). Rotation rule
(plan-locked): switch the default codec **iff** the challenger is **>=15%**
faster on **BOTH** legs (cold fresh-process median AND warm 5-timed median)
**AND** G1-G7 stay green. One leg alone never flips the default; a failed
gate vetoes regardless of speed.

```sh
# A leg (incumbent) then B leg (challenger), same manifest + config slug.
# Canonical decision-bearing leg first; framing-only second:
pixi run python bench/feed_rate.py run --leg stream --corpus f10 --chunk 32 --pin
pixi run python bench/feed_rate.py run --leg expand --corpus f1 --chunk 32 --pin
```

Hardware counters (human-run only, post-delta triage) via the documented
wrapper — never from Python, never part of the gate:

```sh
bench/perf_wrap.sh "pixi run python bench/feed_rate.py run-once --leg stream --corpus f8 --pin"
```

## 3. Normative measurement rules (plan delta #EF78)

- **Overlap claims require the record leg.** H2D overlap on depth>=2 is valid
  only with per-slot events recorded on the transfer stream AFTER the `.to()`
  calls plus `wait_event` on the consumer stream. A bare `wait_event` without
  the record RACES — never assert overlap from it. The bench ring probe
  reports `h2d_ms_*`/`sync_wait_ms_*` from `PinnedRing.stats()`; the `feed`
  label is `ring` (probe ran) or `sync` (geometry only).
- **Hot-row ceiling is DERIVED, not a hand constant.** The Rust hot-13 row
  ceiling comes from the `ROW_BYTES` sum (~0.89KB@T64 / ~2.6KB@T256); the
  comeback bound is +10% over that derived sum. F9 pins the CURRENT-tree
  26-plane baseline instead (B=1024/T=256 slot ~9.8MB, ring = 2x slot);
  the two ceilings live on different plane sets — never compare them.
- **Caps are tensor BYTES.** Staging order is walk -> `t_len` bucket -> rows =
  min(caps/row_bytes) -> commit; plane:4 (T-variable) overflow fails the game,
  never truncates. `BufferTooSmall{plane, needed_bytes, capacity_bytes}` is in
  byte units.
- **G1 closed volatile set** (derivation_hash, privileged_label_ref + `--allow`
  wall_game_id/copy_fold/refit_number/wall_less_marker), else fail-closed.
  Wave3 pre-split baseline digest stands; post-split re-baselining is VOID.

## 4. Interpreting a row

- `warm.median` vs `cold.median`: cold includes import + first-touch;
  a warm/cold ratio >> 1 points at init cost, not steady-state feed.
- `phases`: scan / framing-or-expand / microbatch / ring walls
  (`time.perf_counter`); the max wall is the bottleneck to attack.
- `ring.sync_wait_ms_p99 >> h2d_ms_p99`: consumer starves the ring (raise
  depth 2 -> 3 per grid); inverse: transfer-bound.
- `stream_counters.quarantined > 0` on F2/F8 (clean synth): harness bug,
  stop and fix — never sweep over quarantine noise. This zero budget
  applies to clean synth only; real corpora carry a quota (~8%, see
  `docs/FULL_CORPUS_ROADMAP.md` cross-cutting rules). Multi-ron support
  is a separate ticket.
