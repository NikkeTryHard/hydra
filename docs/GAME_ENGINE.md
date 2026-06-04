# Hydra Game Engine

`hydra-core` is public runtime facade. Current code wins if this doc drifts.

Rules authority is `hydra-engine`, vendored Apache-2.0 RiichiEnv v0.4.8-derived engine. Hydra layers action mapping, encoding, safety, search features, simulator glue, and deterministic seeding on top.

Correctness is not assumed from upstream alone. Current Hydra baseline: default and strict `mjai_audit` over `/home/cachybtw/Downloads/dataset_bundle/tenhou-houou-mjai-2025` loaded `178,897` files, produced `95,118,458` samples, skipped `0`, and found `0` replay legality/load failures.

Mortal is black-box reference only. Do not copy/derive from Mortal code.

## Crate Ownership

| Surface | Owner |
|---|---|
| Rules engine | `hydra-engine` |
| Tile/action/runtime rails | `hydra-runtime-types` |
| Public facade/simulator/seeding/arena glue | `hydra-core` |
| Observation encoder | `hydra-encoder` |
| Safety features | `hydra-safety` |
| Belief/search/Hand-EV helpers | `hydra-belief-search` |

## Tile Contract

- Tile kind ids: `0..33`.
- Suits: `0..8` manzu, `9..17` pinzu, `18..26` souzu, `27..33` honors.
- Physical wall uses 136-format ids. Tile kind is `tile136 / 4`.
- Red fives are special physical tiles: 5m, 5p, 5s.
- Aka/red fives stay distinct on 136-format/action surfaces where required.
- Suit augmentation has exactly 6 numbered-suit permutations. Honors never change.

## Action Contract

Hydra model action space is fixed width `46`:

| Id | Meaning |
|---|---|
| `0..33` | discard tile kind |
| `34..36` | discard red 5m/5p/5s |
| `37` | declare riichi |
| `38..40` | chi left/mid/right |
| `41` | pon |
| `42` | kan bridge |
| `43` | win |
| `44` | abortive draw |
| `45` | pass |

Riichi is two-phase: declare riichi, then select legal discard.

Kan is compact: action `42` maps to `Ankan` in normal phase and `Daiminkan` in response phase. Inbound kan variants collapse to `42`.

compact action facade is 4-player. Sanma/Kita stays engine-level and is not represented in 46-action bridge.

Training and inference legal masks are `[bool; 46]`. Illegal actions must be masked before sampling/softmax.

## Observation Contract

Live model input is `192x34`.

Old `85x34` means historical baseline-prefix channels `0..84`, not live full encoder.

Baseline prefix contains public hand/discard/meld/dora/riichi/score/round/safety features. remaining channels hold fixed-shape search/belief/Hand-EV context with zero-fill and presence masks when dynamic features are unavailable.

Encoder buffers are fixed shape and reused. Hot paths should dirty/update changed channel groups instead of reallocating full tensors when avoidable.

## Safety Features

`hydra-safety` owns defensive tile features:

- genbutsu
- tedashi genbutsu
- riichi-era genbutsu
- suji
- half-suji
- matagi-suji danger
- kabe
- one-chance
- visible counts
- opponent riichi/tenpai hints

Safety data feeds encoder channels `62..84` and should update incrementally when discards/melds change.

## Replay / Event Order

Tenhou/Mahjong Soul/MJAI open-kan order is dora before `dahai`.

acting player still chooses rinshan discard before seeing dora; event stream exposes dora before response window. Legacy after-discard order exists only behind compatibility flag `open_kan_dora_after_discard = true`.

Replay, sidecar, checkpoint, shard, and action-contract mismatches hard-error. Do not silently clamp or skip contract violations.

## Determinism

Hydra uses explicit seeds and deterministic derivation for replay/eval/training surfaces.

Do not introduce unordered-map output dependence, hidden randomness, or host-time-dependent behavior into runtime/training contracts.

## Read Next

- Hard compatibility summary: [`COMPATIBILITY_SURFACE.md`](COMPATIBILITY_SURFACE.md)
- Current status: [`CURRENT_STATUS.md`](CURRENT_STATUS.md)
- Training/operator flow: [`TRAINING_RUNBOOK.md`](TRAINING_RUNBOOK.md)
