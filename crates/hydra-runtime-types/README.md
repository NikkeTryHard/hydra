# hydra-runtime-types

Crate-local map for shared runtime rails. Pure, low-level, no engine/training deps.

## Owns

- Runtime constants/newtypes used across Hydra runtime crates.
- Action rails in `src/action.rs`.
- Tile rails in `src/tile.rs`.
- Dependency-cycle breaker below engine-facing/runtime-facing crates.

## Does not own

- Mahjong rules/scoring/legal generation: `hydra-engine` owns.
- Observation encoding: `hydra-encoder` owns; public compat via `hydra-core`.
- Safety calculations: `hydra-safety` owns.
- Training/model/data contracts: train/data crates own.

## Critical invariants

| Surface | Contract |
|---|---|
| Dependency posture | no `hydra-engine`, no Burn, no training stack |
| Tile kinds | 34-format shared tile rail |
| Action rails | shared runtime action constants/types only |
| Role | sit below runtime crates without cycles |

## Read next

- Runtime contract: [`docs/GAME_ENGINE.md`](../../docs/GAME_ENGINE.md).
- Compatibility surface: [`docs/COMPATIBILITY_SURFACE.md`](../../docs/COMPATIBILITY_SURFACE.md).
- Public runtime bridge: [`crates/hydra-core/README.md`](../hydra-core/README.md).

## License

Business Source License 1.1 (BSL). See repo-root [`LICENSE`](../../LICENSE).
