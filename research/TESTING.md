# Hydra Testing Strategy

## Overview

Testing is critical for a mahjong AI because engine bugs silently corrupt training data. A single incorrect legal action mask, a mis-scored hand, or a wrong tile encoding feeds the neural network garbage labels for hundreds of thousands of training steps before anyone notices. Unlike a web app where users report bugs, a training pipeline happily trains on wrong data and produces a model that plays "confidently wrong" — the worst possible outcome. Every component that touches training data must be verified against independent ground truth.

This document specifies the testing strategy across all Hydra subsystems: the Rust game engine, observation encoder, MJAI parser, suit permutation augmentation, and the Python training stack.

---

## Rust Engine Correctness

### Game State Machine Tests

Every transition in INFRASTRUCTURE.md's state diagram must have a dedicated test. The state machine governs round flow — dealing, drawing, discarding, call checks, kan processing, riichi declarations, and win checks — and a single missed transition can produce impossible game states that silently corrupt downstream data.

**Required transition coverage:**

| Transition | Test Description |
|------------|-----------------|
| Dealing → Drawing | Deal 13 tiles to each player, verify hand sizes and wall count |
| Drawing → Discarding | Draw tile, verify hand size increments by 1 |
| Drawing → WinCheck (tsumo) | Draw a winning tile, verify tsumo is in legal actions |
| Discarding → CallCheck | Discard tile, verify call check runs for all other players |
| CallCheck → Calling (chi/pon) | Call a tile, verify meld is formed and caller must discard |
| CallCheck → KanProcess | Daiminkan, verify dead wall draw and dora flip |
| KanProcess → ChankanCheck | Kakan declared, verify other players can ron (chankan) |
| KanProcess → RinshanCheck | Dead wall draw after kan, verify rinshan tsumo detection |
| RiichiCheck → CallCheck | Riichi declared, verify 1000-point deposit and discard |
| WinCheck → GameEnd (ron) | Ron declared, verify scoring and payment |
| WinCheck → MultiRon | Two players can ron on same discard, verify both detected |

### Scoring Verification

Cross-validate the Rust scoring engine against the `mahjong` Python library (v1.4.0, already listed in INFRASTRUCTURE.md dependencies). The verification corpus is the full set of 11M+ Tenhou Houou hands.

**Methodology:**

1. Parse each Tenhou hand record to extract the winning hand, melds, winning tile, and game context (round wind, seat wind, dora indicators, riichi status, ippatsu, tsumo/ron)
2. Compute yaku, han, fu, and final score using both the Rust engine and the Python `mahjong` library
3. Any disagreement is a bug — log the hand details and expected vs. actual values
4. Target: zero disagreements across the full corpus

**Edge cases requiring explicit test fixtures:**

- Pinfu tsumo (fu calculation differs from ron)
- Double yakuman (e.g., Daisangen + Tsuuiisou)
- Kazoe yakuman (13+ han from non-yakuman yaku)
- Paarenchan (8+ consecutive dealer wins with honba stacking)
- Kiriage mangan (rounding up 3 han 60 fu or 4 han 30 fu — ruleset-dependent)

### Wall Shuffle Determinism

Verify that `(seed, kyoku, honba) → wall` produces identical results across runs, threads, and platforms. This is the foundation of the evaluation protocol (see [SEEDING.md § Reproducibility and Seeding Strategy](SEEDING.md#reproducibility-and-seeding-strategy)).

**Tests:**

1. Fix a seed, generate 1000 walls, compare byte-for-byte against a golden file
2. Run the same generation across 8 rayon threads, verify identical output regardless of thread scheduling
3. Cross-platform: generate walls on x86_64 and aarch64 (if available), verify identical output
4. Regression guard: pin `chacha20 = "=0.10.0"` and the vendored Fisher-Yates shuffle — any change to either must fail CI until the golden file is updated

### Abortive Draw Handling

All five abortive draw types from INFRASTRUCTURE.md must be tested:

| Condition | Test |
|-----------|------|
| Kyuushu Kyuuhai | Construct a hand with 9+ unique terminals/honors, verify action 44 is legal; construct a hand with 8, verify it is not |
| Suufon Renda | Force all 4 players to discard the same wind on turn 1, verify round aborts |
| Suucha Riichi | Force all 4 players to declare riichi, verify round aborts |
| Suukaikan | Force 4 kans by different players, verify round aborts; force 4 kans by same player, verify round does NOT abort |
| Sanchahou | Force 3 players to declare ron on the same discard, verify round aborts (triple ron is abortive in standard rules) |

### Nagashi Mangan Edge Cases

- Player's entire discard pile is terminals/honors, none called by opponents → mangan payment
- Opponent calls one of the player's terminals → nagashi mangan denied
- Player is also tenpai at exhaustive draw → nagashi mangan takes priority over tenpai/noten payments
- Multiple players qualify for nagashi mangan simultaneously

---

## Observation Encoding Correctness

### 84-Channel Verification

Each of the 84 channels must encode exactly what HYDRA_SPEC claims. Build a test harness that constructs known game states and verifies the output tensor element by element.

**Channel-by-channel tests:**

| Channel Range | Verification |
|---------------|-------------|
| 0-3 (hand thermometer) | Set hand to [1m, 1m, 1m, 2m], verify ch0-2 at index 0 are 1.0, ch3 is 0.0 |
| 8 (drawn tile) | Draw 5p, verify only index 12 is 1.0, all others 0.0 |
| 9-10 (shanten masks) | Construct a tenpai hand, verify keep-shanten and next-shanten masks match `xiangting` output |
| 11-22 (discards) | Discard 3 tiles with known tedashi/tsumogiri flags, verify encoding |
| 35-38 (dora) | Set 2 dora indicators, verify thermometer encoding |
| 42-45 (riichi status) | Declare riichi for player 2, verify only ch43 is all-1.0 |
| 46-49 (scores) | Set scores to [25000, 30000, 20000, 25000], verify normalization by 100000 |
| 61-69 (genbutsu) | Opponent declares riichi then player discards 7s → 7s is genbutsu for that opponent |
| 70-78 (suji) | Opponent discards 4m → verify 1m and 7m have suji safety > 0 |
| 79-80 (kabe/one-chance) | All 4 copies of 3p visible → verify kabe flag at index 11 |

### Known-State Golden Tests

Maintain a set of 20+ hand-crafted game states with pre-computed expected tensors, serialized as `.npz` files. These serve as regression tests — any encoder change that alters golden outputs must be reviewed and the golden files explicitly regenerated.

### Roundtrip Tests

Construct a game state programmatically → encode to 84x34 tensor → verify expected values. The encoder is one-way (state → tensor), so "roundtrip" means verifying that the tensor faithfully represents the state, not that the state can be recovered from the tensor.

---

## MJAI Parsing

### Log Reconstruction

Parse real Tenhou and Majsoul game logs in MJAI format, replay the events through the game engine, and verify that the reconstructed game state matches the log's recorded outcomes (final scores, winner, winning hand, yaku).

**Minimum test corpus:**

- 100 randomly sampled Tenhou Houou games
- 100 randomly sampled Majsoul Throne games
- 50 games containing special events (see edge cases below)

### Edge Cases

| Scenario | What to Verify |
|----------|---------------|
| Multiple ron (double/triple) | Both/all winners detected, correct payment split |
| Chankan | Ron on an added kan, correct yaku assignment |
| Rinshan tsumo | Win from dead wall draw after kan, rinshan kaihou yaku applied |
| Double riichi | Riichi declared on first turn (no prior calls), double riichi yaku applied |
| Ippatsu with intervening call | Opponent calls between riichi and next draw, ippatsu denied |
| Haitei/Houtei | Win on last draw/discard, correct yaku applied |

### Event Roundtrip

Generate a game programmatically → serialize to MJAI events → parse events back through the engine → verify final state matches. This catches serialization/deserialization asymmetries.

---

## Suit Permutation Augmentation

### Validity

All 6 permutations of `[manzu, pinzu, souzu]` must produce valid game states. For each permutation:

1. Apply permutation to a game's MJAI event stream
2. Replay permuted events through the engine
3. Verify: no illegal states, no assertion failures, game reaches the same terminal condition

### Aka-Dora Roundtrip

The `deaka → permute → re_akaize` chain must preserve aka-dora identity:

- Red 5m permuted to pinzu → becomes red 5p (not normal 5p)
- Red 5p permuted to souzu → becomes red 5s
- Identity permutation [m→m, p→p, s→s] produces bit-identical output

### Score Invariance

The same game played under all 6 permutations must produce identical final scores. Suits are strategically interchangeable — no yaku depends on suit identity (unlike honor tiles).

### Identity Permutation

Permutation [0, 1, 2] (identity) must produce output identical to no permutation. Byte-for-byte comparison of encoded observations.

---

## Property-Based Testing

Use the `proptest` crate for Rust engine invariants. Property-based tests generate thousands of random inputs and check that invariants hold for all of them.

### Core Invariants

| Property | Invariant |
|----------|-----------|
| Legal action mask | At least 1 legal action when game is not terminal |
| Score conservation | Sum of all 4 player scores equals 100,000 at all times (before riichi deposit adjustments, accounting for kyotaku) |
| Shanten bounds | Shanten is non-negative and at most 6 for any valid hand |
| Tile count bounds | No tile type appears more than 4 times across all visible locations |
| Total tile count | Exactly 136 tiles exist across wall, hands, discards, melds, and dead wall |
| State machine validity | No legal action sequence from a valid state produces an invalid state |
| Terminal detection | A terminal state has an empty legal action set |

### Strategy

1. Generate a random valid initial game state (deal 13 tiles to each player from a shuffled 136-tile wall)
2. At each step, choose a random legal action from the legal action mask
3. Apply the action, check all invariants
4. Repeat until terminal or 500 actions (capped to prevent infinite loops in degenerate cases)
5. Run 10,000+ such random games per CI run

---

## Cross-Validation

### Shanten

Compare the Rust `xiangting` crate's shanten calculation against an independent implementation on N=100,000 randomly generated hands.

**Methodology:**

1. Generate 100K random 13-tile hands (sampling without replacement from 136 tiles)
2. Compute shanten using `xiangting` (Rust)
3. Compute shanten using an independent algorithm (e.g., lookup table or brute-force)
4. Any disagreement is a bug — log the hand tiles and both results
5. Include edge cases: complete hands (shanten = -1), kokushi tenpai, chiitoitsu tenpai

### Scoring

Cross-validate Rust scoring against the `mahjong` Python library on 100K randomly constructed winning hands.

**Methodology:**

1. Generate random winning hands (tenpai hands + a completing tile)
2. Assign random game context (round wind, seat wind, dora, riichi, tsumo/ron)
3. Compute yaku/han/fu/score in both Rust and Python
4. Diff results — any mismatch is logged with full context for debugging
5. Special attention to fu calculation edge cases (open pinfu, closed tsumo, etc.)

---

## Python Training Stack

### Model Smoke Tests

- Forward pass with random input `[1, 84, 34]` produces correct output shapes: policy `[1, 46]`, value `[1, 1]`, GRP `[1, 24]`, tenpai `[1, 3]`, danger `[1, 3, 34]`
- Legal action masking: masked logits are negative infinity, softmax produces zero probability for illegal actions
- ONNX export: export model, run inference through ONNX Runtime, verify output matches PyTorch within tolerance (atol=1e-5)

### Loss Function Tests

- Policy CE loss with known logits and labels → verify against hand-computed value
- GRP 24-way CE loss sums to correct value for a known permutation distribution
- Focal BCE loss with γ=2.0 produces lower loss for high-confidence correct predictions than standard BCE
- Composite loss with known component values → verify weighted sum matches expected total

### Data Pipeline Tests

- DataLoader yields batches of correct shape `[2048, 84, 34]`
- 3-level shuffle produces different orderings across epochs (statistical test: correlation < 0.1)
- Suit permutation produces 6 distinct outputs for the same input game
- Filtering: a game with known bad metadata is excluded from the manifest
