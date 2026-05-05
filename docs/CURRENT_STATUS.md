# Hydra Current Status

Current shipped/staged snapshot for Hydra built surfaces.

Use file for: what shipped today, what implemented but staged, what implemented but not default-on.

File reports shipped/staged status only.

- For roadmap to Hydra v1, read `research/design/HYDRA_RECONCILIATION.md`.
- For runtime semantics and compatibility truth, read `docs/GAME_ENGINE.md`, `docs/COMPATIBILITY_SURFACE.md`, and current code.

If file and current code disagree, current code wins. If file and `HYDRA_RECONCILIATION.md` disagree on active vs reserve vs staged priority, refresh reconciliation, then refresh file. If reconciliation or current status drift from archive root, refresh promoted docs, not demote canonical archive source ledger.

## Status vocabulary

File uses status vocabulary from `research/design/HYDRA_RECONCILIATION.md`.

| Term | Meaning |
|---|---|
| `shipped baseline` | implemented, part of current live baseline |
| `implemented but not default-on` | implemented, validated enough to exist in-code, intentionally not default runtime/training path |
| `implemented but staged` | core code path exists, promotion/activation intentionally deferred |
| `reserve shelf` | documented later-work direction, not current mainline priority |
| `historical` | preserved context only; not current governing truth |

## Runtime and training snapshot

### Shipped baseline

- `hydra-core` = real first-party runtime/encoder/simulator crate.
- Live encoder/model contract = `192x34`; old `85x34` view = baseline-prefix only.
- Fixed runtime action space = 46 actions with two-phase riichi and kan handling.
- BC training supports **epoch-boundary-only** reuse of matching preflight-selected runtime for selected-runtime tuple (`train_microbatch_size`, `validation_microbatch_size`, derived `accum_steps`); fresh runs stay config-derived, partial-epoch resumes still require identical runtime, loader-runtime stays config-derived.
- Stronger public-teacher belief-semantics tranche shipped in current training baseline.
- Current Hand-EV realism upgrade shipped in live baseline surface.
- Replay-derived `safety_residual` shipped as narrow supervised lane.
- ExIt has end-to-end carrier across live self-play lane and replay/sample sidecar-first lane.

### Implemented but not default-on

- Narrow DeltaQ supervision lane implemented in code, promotion-gated through arena-confirmation path.
- DeltaQ promotion artifacts now persist explicit `arena_decision` plus `arena_report`, but lane still **not** default-on.

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
| Hand-EV baseline surface | shipped baseline | Stronger local evaluator live; representative-world CT-SMC Hand-EV still staged |
| Belief semantics baseline | shipped baseline | Stronger public-teacher belief tranche in live baseline |
| BC runtime authority | shipped baseline | Fresh runs config-derived; epoch-boundary resumes may reuse matching preflight-selected runtime for selected-runtime only; partial-epoch resumes still require identical runtime; loader-runtime remains config-derived |
| BF16/AMP precision | shipped baseline (BC); staged (RL, DeltaQ) | BC training, preflight, probe, autotune, stage-2 benchmark all dispatch by precision. RL training and DeltaQ promotion explicitly gated with hard errors. |
| Preflight cache system | shipped baseline | Fingerprint v4 key covers hardware, workload, preflight config, explicit microbatch overrides. Identical-run fast path skips probing on cache hit. BC and RL bootstrap read cache under documented authority rules. |
| NVTX profiling | shipped baseline | Orchestration-level fully instrumented (epoch, step, validation, checkpoint, logging, self-play, stage-2 benchmark). BC microbatch sub-stages (collation, forward, loss, backward, optimizer_step) instrumented. Library internals not yet instrumented. Gated by `HYDRA_NVTX` env var via dlopen. |
| `safety_residual` | shipped baseline | Narrow replay-derived supervised lane |
| ExIt carrier | shipped baseline | Live self-play lane + replay/sample sidecar-first lane |
| DeltaQ lane | implemented but not default-on | Arena-confirmation path implemented; promotion artifact now records pre-arena recommendation plus final `arena_decision`/`arena_report` |
| `mixture_weight` activation | implemented but staged | Surface exists, promotion remains deferred |
| `opponent_hand_type` activation | implemented but staged | Surface exists, target closure remains incomplete |
| AFBS broad default runtime | reserve shelf | Specialist / hard-state gated direction only |

## Where to read next

- Need current runtime contract? Read `docs/GAME_ENGINE.md` and `docs/COMPATIBILITY_SURFACE.md`.
- Need roadmap to Hydra v1 or active-path / staged-vs-reserve decision? Read `research/design/HYDRA_RECONCILIATION.md`.
- Need north-star architecture, not current shipped status? Read `research/design/HYDRA_FINAL.md`.