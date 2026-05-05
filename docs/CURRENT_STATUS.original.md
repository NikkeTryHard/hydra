# Hydra Current Status

Current shipped/staged status for Hydra's already-built surfaces.

This file is Hydra's promoted current-status snapshot for things that already exist in code or are partially implemented in code. Use it to answer questions like "what is shipped today?", "what is implemented but still staged?", and "what is implemented but not default-on yet?"

This file reports shipped/staged status only.

- For the roadmap to Hydra v1, read `research/design/HYDRA_RECONCILIATION.md`.
- For runtime semantics and compatibility truth, read `docs/GAME_ENGINE.md`, `docs/COMPATIBILITY_SURFACE.md`, and current code.

When this file and current code disagree, current code wins. When this file and `HYDRA_RECONCILIATION.md` disagree on active vs reserve vs staged priority, refresh reconciliation and then refresh this file. When reconciliation or current status drift from the archive root, refresh the promoted docs rather than demoting the canonical archive source ledger.

## Status vocabulary

This file uses the status vocabulary defined in `research/design/HYDRA_RECONCILIATION.md`.

| Term | Meaning |
|---|---|
| `shipped baseline` | implemented and part of the current live baseline |
| `implemented but not default-on` | implemented and validated enough to exist in-code, but intentionally not the default runtime/training path |
| `implemented but staged` | core code path exists, but promotion/activation is still intentionally deferred |
| `reserve shelf` | documented later-work direction, not current mainline priority |
| `historical` | preserved context only; not current governing truth |

## Runtime and training snapshot

### Shipped baseline

- `hydra-core` is a real first-party runtime/encoder/simulator crate.
- The live encoder/model contract is `192x34`; the old `85x34` view is baseline-prefix only.
- The fixed runtime action space is 46 actions with two-phase riichi and kan handling.
- BC training now supports **epoch-boundary-only** reuse of matching preflight-selected runtime for the selected-runtime tuple (`train_microbatch_size`, `validation_microbatch_size`, derived `accum_steps`), while fresh runs remain config-derived, partial-epoch resumes still require identical runtime, and loader-runtime stays config-derived.
- The stronger public-teacher belief-semantics tranche is shipped as part of the current training baseline.
- The current Hand-EV realism upgrade is shipped as part of the live baseline surface.
- Replay-derived `safety_residual` is shipped as a narrow supervised lane.
- ExIt now has an end-to-end carrier across the live self-play lane and the replay/sample sidecar-first lane.

### Implemented but not default-on

- The narrow DeltaQ supervision lane is implemented in code and promotion-gated through an arena-confirmation path.
- DeltaQ promotion artifacts now persist explicit `arena_decision` plus `arena_report`, but the lane is still **not** default-on.

### Implemented but staged

- `mixture_weight` promotion remains staged.
- Richer opponent-target closure remains staged.
- Representative-world / per-particle CT-SMC Hand-EV remains staged.
- Selective AFBS / endgame deepening remains staged.

### Reserve shelf

- Broader public-belief search as project identity remains reserve-shelf, not active-path.
- Deeper robust-opponent search backups remain reserve-shelf.
- Larger latent-opponent / richer auxiliary-head expansion remains reserve-shelf until existing target closure improves.

## Area-by-area summary

| Area | Current status | Notes |
|---|---|---|
| Runtime encoder / action semantics | shipped baseline | See `docs/GAME_ENGINE.md` and `docs/COMPATIBILITY_SURFACE.md` |
| Hand-EV baseline surface | shipped baseline | Stronger local evaluator is live; representative-world CT-SMC Hand-EV remains staged |
| Belief semantics baseline | shipped baseline | Stronger public-teacher belief tranche is in the live baseline |
| BC runtime authority | shipped baseline | Fresh runs are config-derived; epoch-boundary resumes may reuse matching preflight-selected runtime for selected-runtime only; partial-epoch resumes still require identical runtime; loader-runtime remains config-derived |
| BF16/AMP precision | shipped baseline (BC); staged (RL, DeltaQ) | BC training, preflight, probe, autotune, and stage-2 benchmark all dispatch by precision. RL training and DeltaQ promotion are explicitly gated with hard errors. |
| Preflight cache system | shipped baseline | Fingerprint v4 key covers hardware, workload, preflight config, explicit microbatch overrides. Identical-run fast path skips probing on cache hit. BC and RL bootstrap read cache under documented authority rules. |
| NVTX profiling | shipped baseline | Orchestration-level fully instrumented (epoch, step, validation, checkpoint, logging, self-play, stage-2 benchmark). BC microbatch sub-stages (collation, forward, loss, backward, optimizer_step) instrumented. Library internals not yet instrumented. Gated by `HYDRA_NVTX` env var via dlopen. |
| `safety_residual` | shipped baseline | Narrow replay-derived supervised lane |
| ExIt carrier | shipped baseline | Live self-play lane + replay/sample sidecar-first lane |
| DeltaQ lane | implemented but not default-on | Arena-confirmation path implemented; promotion artifact now records pre-arena recommendation plus final `arena_decision`/`arena_report` |
| `mixture_weight` activation | implemented but staged | Surface exists, promotion remains deferred |
| `opponent_hand_type` activation | implemented but staged | Surface exists, target closure remains incomplete |
| AFBS broad default runtime | reserve shelf | Specialist / hard-state gated direction only |

## Where to read next

- Need the current runtime contract? Read `docs/GAME_ENGINE.md` and `docs/COMPATIBILITY_SURFACE.md`.
- Need the roadmap to Hydra v1 or the active-path / staged-vs-reserve decision? Read `research/design/HYDRA_RECONCILIATION.md`.
- Need the north-star architecture rather than current shipped status? Read `research/design/HYDRA_FINAL.md`.
