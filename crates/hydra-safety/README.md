# hydra-safety

Crate-local map for safety rail primitives feeding runtime encoding.

## Owns

- Genbutsu, suji, half-suji, matagi, kabe, one-chance tracking.
- `SafetyInfo` state updated from discard/call/kan/riichi events.
- Tenpai hint activation threshold and cached opponent tenpai probs.
- Safety channels used by fixed-superset observation encoding.

## Does not own

- Observation tensor layout/encoding impl: `hydra-encoder` owns.
- Public runtime bridge/re-exports: `hydra-core` owns.
- Mahjong legality/scoring: `hydra-engine` owns.
- Policy/model/training execution: train/model crates own.

## Critical invariants

| Surface | Contract |
|---|---|
| Opponents | 3 relative opponents |
| Tile kinds | 34 tile types |
| Bitfields | `u64` tile bitfields; only bits `0..34` meaningful |
| Tenpai hint | active if riichi or cached prob `> 0.5` |
| Hot path | incremental updates; avoid per-turn heap work |

## Read next

- Runtime/channel contract: [`docs/GAME_ENGINE.md`](../../docs/GAME_ENGINE.md).
- Encoder owner: [`crates/hydra-encoder/README.md`](../hydra-encoder/README.md).
- Public runtime bridge: [`crates/hydra-core/README.md`](../hydra-core/README.md).

## License

Business Source License 1.1 (BSL). See repo-root [`LICENSE`](../../LICENSE).
