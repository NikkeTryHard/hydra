# Hydra Testing Strategy

> **Status note:** File mixes active reqs + old baseline-prefix checks. Next impl priority: `research/design/HYDRA_RECONCILIATION.md`. Live runtime + compat truth: `docs/GAME_ENGINE.md`, `docs/COMPATIBILITY_SURFACE.md`, current code.
> >
> Live encoder/model contract = `192x34`. Old `85x34` view useful only as baseline prefix (`channels 0..84`); test it that way.

## Overview

Testing critical for mahjong AI. Engine bugs silently poison training data. One wrong legal mask, score, or tile encoding feeds garbage labels for huge training spans before detection. Unlike web apps, pipeline trains happily on bad data and yields model that plays confidently wrong. Every training-data-touching component needs independent ground-truth verification.

This doc defines testing strategy for all Hydra subsystems: Rust engine, observation encoder, MJAI parser, suit permutation augmentation, Burn training stack.

### Current Executable Inventory

- Workspace Rust tests: `cargo nextest run --release`.
- Narrow Rust cases: `cargo test --release -p <crate> <test-name>`.
- Integration tests: `crates/hydra-core/tests/{golden_encoder,mjai_replay,proptest_invariants,game_loop_integration}.rs`; `crates/hydra-train/tests/integration_pipeline.rs`.
- Benches: `cargo bench -p hydra-core`; `cargo bench -p hydra-engine`; `cargo bench -p hydra-train`.
- Core throughput example: `cargo run -p hydra-core --example bench_throughput`.
- Python script tests: `uv run python -m unittest discover -s scripts/tests`.

### Coverage Reporting

Hydra publishes workspace-wide Rust coverage via `cargo-llvm-cov`. Coverage = regression-review aid, not correctness proof.

- Coverage path: `./scripts/coverage.sh`, backed by `cargo llvm-cov nextest`.
- Default artifacts under `target/coverage/`: `summary.txt`, `summary.json`, `timings.txt`, `run.log`, optional sccache stats.
- HTML requires `HYDRA_COVERAGE_HTML=1`; LCOV requires `HYDRA_COVERAGE_LCOV=1`.
- `HYDRA_COVERAGE_FAST=1` skips HTML, LCOV, and summary generation.
Use coverage to confirm tests hit risky paths like encoder writes, replay roundtrips, legal-action generation, scoring, training-label gating. Do not treat one repo-wide percentage as semantic safety proof.

---

## Rust Engine Correctness

### Game State Machine Tests

Every Hydra live game-state transition needs dedicated test. Use `hydra-engine` / `hydra-core` runtime code and `docs/GAME_ENGINE.md` as primary truth; use `INFRASTRUCTURE.md` only as historical/supporting reference when still aligned with runtime reality. State machine controls round flow: dealing, drawing, discarding, call checks, kan processing, riichi declarations, win checks. One missed transition can create impossible states and silently corrupt downstream data.

**Required transition coverage:**

| Transition | Test Description |
|------------|-----------------|
| Dealing → Drawing | Deal 13 tiles to each player, verify hand sizes and wall count |
| Drawing → Discarding | Draw tile, verify hand size increments by 1 |
| Drawing → WinCheck (tsumo) | Draw winning tile, verify tsumo is in legal actions |
| Discarding → CallCheck | Discard tile, verify call check runs for all other players |
| CallCheck → Calling (chi/pon) | Call tile, verify meld is formed and caller must discard |
| CallCheck → KanProcess | Daiminkan, verify dead wall draw and dora flip |
| KanProcess → ChankanCheck | Kakan declared, verify other players can ron (chankan) |
| KanProcess → RinshanCheck | Dead wall draw after kan, verify rinshan tsumo detection |
| RiichiCheck → CallCheck | Riichi declared, verify 1000-point deposit and discard |
| WinCheck → GameEnd (ron) | Ron declared, verify scoring and payment |
| WinCheck → MultiRon | Two players can ron on same discard, verify both detected |

### Scoring Verification

Cross-validate Rust scoring engine against `mahjong` Python library (v1.4.0, used via one-time build.rs validation script). Verification corpus = full 11M+ Tenhou Houou hands.

**Methodology:**

1. Parse each Tenhou hand record for winning hand, melds, winning tile, game context (round wind, seat wind, dora indicators, riichi status, ippatsu, tsumo/ron)
2. Compute yaku, han, fu, final score in both Rust engine and Python `mahjong` library
3. Any disagreement = bug; log hand details and expected vs. actual
4. Target: zero disagreements across full corpus

**Edge cases requiring explicit test fixtures:**

- Pinfu tsumo (fu calculation differs from ron)
- Double yakuman (e.g., Daisangen + Tsuuiisou — additive stacking under Tenhou ruleset Hydra targets)
- Kazoe yakuman (13+ han from non-yakuman yaku — scored as yakuman per Tenhou rules)
- Paarenchan (8+ consecutive dealer wins — no special scoring per Tenhou rules; honba uncapped)
- Kiriage mangan (3 han 60 fu / 4 han 30 fu = 7700, NOT rounded to mangan per Tenhou ranked rules)

### Wall Shuffle Determinism

Verify `(seed, kyoku, honba) → wall` stays identical across runs, threads, platforms. This underpins evaluation protocol (see [SEEDING.md § Reproducibility and Seeding Strategy](SEEDING.md#reproducibility-and-seeding-strategy)).

**Tests:**

1. Fix seed, generate 1000 walls, compare byte-for-byte with golden file
2. Run same generation across 8 rayon threads, verify identical output regardless of scheduling
3. Cross-platform: generate walls on x86_64 and aarch64 (if available), verify identical output
4. Regression guard: pin `chacha20 = "=0.10.0"` and vendored Fisher-Yates shuffle — any change must fail CI until golden file updated

### Abortive Draw Handling

All five abortive draw types from INFRASTRUCTURE.md need tests:

| Condition | Test |
|-----------|------|
| Kyuushu Kyuuhai | Construct hand with 9+ unique terminals/honors, verify action 44 is legal; construct hand with 8, verify it is not |
| Suufon Renda | Force all 4 players to discard same wind on turn 1, verify round aborts |
| Suucha Riichi | Force all 4 players to declare riichi, verify round aborts |
| Suukaikan | Force 4 kans by different players, verify round aborts; force 4 kans by same player, verify round does NOT abort |
| Sanchahou | Force 3 players to declare ron on same discard, verify round aborts (triple ron is abortive in standard rules) |

### Nagashi Mangan Edge Cases

- Player's full discard pile = terminals/honors, none called by opponents → mangan payment
- Opponent calls one player terminal → nagashi mangan denied
- Player also tenpai at exhaustive draw → nagashi mangan overrides tenpai/noten payments
- Multiple players qualify for nagashi mangan simultaneously

---

## Observation Encoding Correctness

### Baseline-Prefix Verification (Channels 0-84)

Each first 85 channels must encode baseline public+safety prefix exactly as `docs/GAME_ENGINE.md` defines. Build harness that constructs known game states and verifies baseline prefix element by element, while keeping full live tensor shape at `192x34`.

**Channel-by-channel tests:**

| Channel Range | Verification |
|---------------|-------------|
| 0-3 (hand thermometer) | Set hand to [1m, 1m, 1m, 2m], verify ch0-2 at index 0 are 1.0, ch3 is 0.0 |
| 8 (drawn tile) | Draw 5p, verify only index 12 is 1.0, all others 0.0 |
| 9-10 (shanten masks) | Construct tenpai hand, verify keep-shanten and next-shanten masks match `xiangting` output |
| 11-22 (discards) | Discard 3 tiles with known tedashi/tsumogiri flags, verify encoding |
| 35-42 (dora/aka) | Set 2 dora indicators, verify thermometer encoding; check aka planes for red 5s |
| 42-45 (riichi status) | Declare riichi for player 2, verify only ch43 is all-1.0 |
| 46-49 (scores) | Set scores to [25000, 30000, 20000, 25000], verify normalization by 100000 |
| 62-70 (genbutsu) | Opponent declares riichi then player discards 7s → 7s is genbutsu for that opponent |
| 71-79 (suji) | Opponent discards 4m → verify 1m and 7m have suji safety > 0 |
| 80-81 (kabe/one-chance) | All 4 copies of 3p visible → verify kabe flag at index 11 |

### Known-State Golden Tests

Maintain 20+ hand-crafted game states with precomputed expected tensors, serialized as `.npz` files. These are regression tests — any encoder change that alters golden outputs must be reviewed and golden files explicitly regenerated.

### Roundtrip Tests

Construct game state programmatically → encode to live `192x34` tensor → verify expected values. Encoder is one-way (state → tensor), so "roundtrip" here means tensor faithfully represents state, not that state can be recovered from tensor.

---

## MJAI Parsing

### Log Reconstruction

Parse real Tenhou and Majsoul MJAI logs, replay events through engine, verify reconstructed state matches logged outcomes (final scores, winner, winning hand, yaku).

Current status note: live replay path now has explicit regression coverage for replay round-reset semantics and kan replay legality matching, and full Tenhou Houou 2025 audit (`178,897` MJAI files) completed with `0` skips after those fixes. Remaining replay failures should be treated as real file/data faults unless new regression reproducer says otherwise.

**Minimum test corpus:**

- 100 randomly sampled Tenhou Houou games
- 100 randomly sampled Majsoul Throne games
- 50 games containing special events (see edge cases below)

### Edge Cases

| Scenario | What to Verify |
|----------|---------------|
| Multiple ron (double/triple) | Both/all winners detected, correct payment split |
| Chankan | Ron on added kan, correct yaku assignment |
| Rinshan tsumo | Win from dead wall draw after kan, rinshan kaihou yaku applied |
| Double riichi | Riichi declared on first turn (no prior calls), double riichi yaku applied |
| Ippatsu with intervening call | Opponent calls between riichi and next draw, ippatsu denied |
| Haitei/Houtei | Win on last draw/discard, correct yaku applied |

### Event Roundtrip

Generate game programmatically → serialize to MJAI events → parse events back through engine → verify final state matches. This catches serialization/deserialization asymmetries.

---

## Suit Permutation Augmentation

### Validity

All 6 permutations of `[manzu, pinzu, souzu]` must yield valid game states. For each permutation:

1. Apply permutation to game MJAI event stream
2. Replay permuted events through engine
3. Verify: no illegal states, no assertion failures, game reaches same terminal condition

### Aka-Dora Roundtrip

`deaka → permute → re_akaize` chain must preserve aka-dora identity:

- Red 5m permuted to pinzu → becomes red 5p (not normal 5p)
- Red 5p permuted to souzu → becomes red 5s
- Identity permutation [m→m, p→p, s→s] produces bit-identical output

### Score Invariance

Same game under all 6 permutations must produce identical final scores. Suits are strategically interchangeable — no yaku depends on suit identity (unlike honor tiles).

### Identity Permutation

Permutation [0, 1, 2] (identity) must produce output identical to no permutation. Byte-for-byte comparison of encoded observations.

---

## Property-Based Testing

Use `proptest` crate for Rust engine invariants. Property-based tests generate many random inputs and check invariants hold.

### Core Invariants

| Property | Invariant |
|----------|-----------|
| Legal action mask | At least 1 legal action when game is not terminal |
| Score conservation | Sum of all 4 player scores equals 100,000 at all times (before riichi deposit adjustments, accounting for kyotaku) |
| Shanten bounds | Shanten is non-negative and at most 6 for any valid hand |
| Tile count bounds | No tile type appears more than 4 times across all visible locations |
| Total tile count | Exactly 136 tiles exist across wall, hands, discards, melds, and dead wall |
| State machine validity | No legal action sequence from valid state produces invalid state |
| Terminal detection | terminal state has empty legal action set |

### Strategy

1. Generate random valid initial game state (deal 13 tiles to each player from shuffled 136-tile wall)
2. At each step, choose random legal action from legal action mask
3. Apply action, check all invariants
4. Repeat until terminal or 500 actions (cap prevents infinite loops in degenerate cases)
5. Run 10,000+ such random games per CI run

---

## Cross-Validation

### Shanten

Compare Rust `xiangting` crate shanten calculation against independent impl on N=100,000 randomly generated hands.

**Methodology:**

1. Generate 100K random 13-tile hands (sampling without replacement from 136 tiles)
2. Compute shanten with `xiangting` (Rust)
3. Compute shanten with independent algorithm (e.g., lookup table or brute-force)
4. Any disagreement = bug; log hand tiles and both results
5. Include edge cases: complete hands (shanten = -1), kokushi tenpai, chiitoitsu tenpai

### Scoring

Cross-validate Rust scoring against `mahjong` Python library on 100K randomly constructed winning hands.

**Methodology:**

1. Generate random winning hands (tenpai hands + completing tile)
2. Assign random game context (round wind, seat wind, dora, riichi, tsumo/ron)
3. Compute yaku/han/fu/score in both Rust and Python
4. Diff results — any mismatch logged with full context for debugging
5. Focus extra on fu edge cases (open pinfu, closed tsumo, etc.)

---

## Burn Training Stack

Current tests are split by crate ownership:

| Crate / path | Test surface |
|---|---|
| `crates/hydra-model/src/{backbone,heads,inference,model,saf}.rs` | Model shape, head, inference, and SAF checks. |
| `crates/hydra-train-algo/src/{ach,bc,distill,drda,gae,losses}.rs` | Pure training loss/algo math: policy CE, distill, GAE, DRDA, composite losses. |
| `crates/hydra-train-exec/src/{data/sample,data/augment,bc_fixed_shape,bc_metrics,modes,artifacts,...}.rs` | Batch collation, augmentation, fixed-shape parity, metrics, modes, artifacts, preflight/probe/runtime seams. |
| `crates/hydra-selfplay/src/{lib,batch,validation}.rs` | Self-play generation, action selection, cooperative runner, RL batch collation, validation entry points. |
| `crates/hydra-train/tests/integration_pipeline.rs` | Integration smoke over model, losses, distill, GAE/DRDA, CT-SMC/AFBS, and replay sidecar compile paths. |

### Required Test Classes

- Model smoke: forward pass over `[N, 192, 34]`, actor/learner output shapes, legal-action mask behavior, backend inference tolerance.
- Losses: known logits/labels against hand-computed values; component-weighted composite totals; focal confidence behavior.
- Data pipeline: batch shape, shuffle/sampling behavior, suit permutation diversity, bad-metadata filtering, sidecar hydration gates.