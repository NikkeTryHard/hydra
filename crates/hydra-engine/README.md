# hydra-engine

Crate-local landing page for vendored engine. Runtime contracts using this engine live in [`docs/GAME_ENGINE.md`](../../docs/GAME_ENGINE.md). Compatibility quick table lives in [`docs/COMPATIBILITY_SURFACE.md`](../../docs/COMPATIBILITY_SURFACE.md).

## Owns

- Vendored `riichienv-core` v0.4.7 engine behavior from [smly/RiichiEnv](https://github.com/smly/RiichiEnv).
- Mahjong state progression, rule enforcement, scoring, legal action generation.
- 4-player + 3-player/sanma engine paths, including Kita/BaBei support.
- Hydra throughput patches in engine layer: stack allocs, zero-copy observation, buffer-reuse legal actions, unchecked self-play step path.
- Workspace lib name remains `riichienv_core`; crate package is `hydra-engine`.

## Does not own

- Hydra encoder/channel semantics: `crates/hydra-core` owns.
- Hydra 46-action compact policy surface: `crates/hydra-core/src/action.rs` owns conversion/mask contract.
- Training/inference/model/runtime-selection: `crates/hydra-train` owns.
- Research doctrine: `research/design/*` owns.

## Critical invariants

| Surface | Contract |
|---|---|
| Origin | `riichienv-core` v0.4.7, vendored |
| Upstream license | Apache-2.0 preserved |
| Hydra additions | BSL-1.1 (`ObservationRef`, `MjaiEvent`, `step_unchecked`, etc.) |
| AGPL boundary | Mortal used only upstream as black-box MJAI correctness player; no Mortal/AGPL code in Hydra/RiichiEnv |
| Correctness basis | upstream reports 1M+ hanchan with Mortal black-box MJAI player, zero errors |
| Engine API name | `riichienv_core` for internal imports/backcompat |
| Publish status | workspace-internal; not crates.io surface |

## Hydra patches summary

| Area | Change | Why |
|---|---|---|
| `Action` / `Meld` | fixed arrays + counts | `Copy`, no heap alloc |
| `HandEvaluator` | borrowed hands/melds; stack melds; buffer-reuse waits | remove clones/allocs |
| `GameState` | `step_unchecked()` | skip redundant validation in trusted self-play loops |
| Step impl | `_execute_step_array` + extracted handlers | one maintained path |
| Observation | `ObservationRef` | zero-copy state access |
| Legal actions | `get_legal_actions_into()` | caller-owned buffer |
| Claim resolution | direct array write | zero-alloc claim path |
| Wall/player data | fixed arrays + cursor | stack/O(1) draw path |
| Safety | `u64` bitfields | compact genbutsu/kabe/one-chance tracking |
| MJAI logging | gated by `skip_mjai_logging` | zero-cost when disabled |
| Shanten tables | public | hydra-core batch shanten cache |

## Read next

- Runtime/channel/action contract: [`docs/GAME_ENGINE.md`](../../docs/GAME_ENGINE.md).
- Compat-sensitive surface: [`docs/COMPATIBILITY_SURFACE.md`](../../docs/COMPATIBILITY_SURFACE.md).
- Perf/history: [`research/infrastructure/ENGINE_BENCHMARKS.md`](../../research/infrastructure/ENGINE_BENCHMARKS.md).

## License

Original vendored engine code: Apache-2.0 (declared in [`Cargo.toml`](Cargo.toml)). Hydra-specific additions: BSL-1.1, aligned with root [`LICENSE`](../../LICENSE). No AGPL/Mortal code imported, linked, copied, or vendored.
