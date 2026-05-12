# hydra-encoder

Crate-local map for observation encoder components. Public ABI remains via `hydra-core` compatibility re-exports.

## Owns

- Observation encoder impl modules.
- Batch encoder components.
- Runtime feature materialization into Hydra fixed-superset observation tensors.
- Internal encoder composition over runtime types, safety, belief/search pieces, and engine observations.

## Does not own

- Public runtime API/re-export compatibility: `hydra-core` owns.
- Mahjong state progression/rules/legal actions: `hydra-engine` owns.
- Safety primitive semantics: `hydra-safety` owns.
- Training/model/data pipelines: train/data/model crates own.

## Critical invariants

| Surface | Contract |
|---|---|
| Public ABI | compatibility surface exposed through `hydra-core` |
| Encoder shape | follows `docs/GAME_ENGINE.md` / `docs/COMPATIBILITY_SURFACE.md` |
| Engine input | consumes `riichienv_core` observations/ref views |
| Boundary | encoder impl here; simulator/action API not here |

## Read next

- Runtime/channel/action contract: [`docs/GAME_ENGINE.md`](../../docs/GAME_ENGINE.md).
- Shape-sensitive compat facts: [`docs/COMPATIBILITY_SURFACE.md`](../../docs/COMPATIBILITY_SURFACE.md).
- Public runtime bridge: [`crates/hydra-core/README.md`](../hydra-core/README.md).

## License

Business Source License 1.1 (BSL). See repo-root [`LICENSE`](../../LICENSE).
