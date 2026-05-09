# hydra-core

Crate-local landing page. Runtime manual lives in [`docs/GAME_ENGINE.md`](../../docs/GAME_ENGINE.md). Shape/compat quick table lives in [`docs/COMPATIBILITY_SURFACE.md`](../../docs/COMPATIBILITY_SURFACE.md).

## Owns

- Runtime bridge from `hydra-engine` / `riichienv_core` observations into Hydra features.
- Fixed-superset observation encoder: `192 x 34`; baseline prefix = channels `0..84`.
- 46-action Hydra action space + legal mask semantics.
- Safety features: genbutsu, suji, kabe, one-chance, tenpai hints.
- Deterministic seeding: SHA-256 KDF, ChaCha8Rng, vendored Fisher-Yates wall shuffle.
- Batch sim + game loop glue for training/eval.
- Batch shanten cache + runtime feature plumbing for search/belief / Hand-EV planes.

## Does not own

- Mahjong rules, scoring, legal action generation internals: `crates/hydra-engine` owns vendored engine behavior.
- Model architecture, losses, runtime selection, BC/RL orchestration: `crates/hydra-train` owns.
- Research doctrine / promoted-vs-staged decisions: `research/design/HYDRA_RECONCILIATION.md` + `research/design/HYDRA_FINAL.md` own.

## Critical invariants

| Surface | Contract |
|---|---|
| Encoder shape | `192 x 34` floats, row-major |
| Baseline prefix | channels `0..84`; historical `85 x 34`, not full live encoder |
| Group C | channels `85..149`; search/belief context + masks + reserve |
| Group D | channels `150..191`; Hand-EV context + mask |
| Action space | 46 actions, Mortal-compatible indexing |
| Legal mask | `[bool; 46]`, same semantics for train + inference |
| Riichi / kan | two-phase: declare, then choose specific discard/kan when engine asks |
| Tiles | 34 tile kinds; 136-format preserves physical/aka identity where needed |
| Determinism | same seed/config -> deterministic per-game results across rayon schedules |

## Read next

- Need channel/action/seeding/runtime detail: [`docs/GAME_ENGINE.md`](../../docs/GAME_ENGINE.md).
- Need shape-sensitive compatibility facts: [`docs/COMPATIBILITY_SURFACE.md`](../../docs/COMPATIBILITY_SURFACE.md).
- Need engine ownership/license boundary: [`crates/hydra-engine/README.md`](../hydra-engine/README.md).
- Need train runtime authority: [`docs/TRAINING_RUNBOOK.md`](../../docs/TRAINING_RUNBOOK.md).
- Need perf numbers/history: [`research/infrastructure/ENGINE_BENCHMARKS.md`](../../research/infrastructure/ENGINE_BENCHMARKS.md).

## License

`hydra-core` = BSL-1.1. See [`LICENSE`](LICENSE). Personal/non-commercial/academic use allowed; commercial mahjong AI services require paid license from Licensor; converts to Apache-2.0 on 2031-03-02.
