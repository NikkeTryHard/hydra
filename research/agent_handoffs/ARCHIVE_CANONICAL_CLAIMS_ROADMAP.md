# Archive Canonical Claims Roadmap

> Non-SSOT handoff artifact.
>
> This file ranks archive-derived research lanes from `ARCHIVE_CANONICAL_CLAIMS.md`.
> It does **not** promote archive claims into current Hydra doctrine by itself.
>
> Authority order remains:
> 1. `research/design/HYDRA_FINAL.md`
> 2. `research/design/HYDRA_RECONCILIATION.md`
> 3. `docs/GAME_ENGINE.md`
>
> If this file conflicts with `HYDRA_RECONCILIATION.md` on sequencing, tranche priority,
> or active-vs-reserve status, `HYDRA_RECONCILIATION.md` wins.

## Scope

This roadmap exists to answer one narrow question:

- given the surviving archive claims,
- what should be treated as **do now**,
- what is **phase-next**,
- what belongs on the **reserve shelf**,
- and what should remain **blocked, rejected, or not active path**.

This file is an archive triage map, not a replacement build-order memo.
Use it after reading `README.md` authority routing and `HYDRA_RECONCILIATION.md`.

## Inputs and interpretation rules

Primary evidence base:

- `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.md`
- `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl`
- `research/design/HYDRA_RECONCILIATION.md`
- `research/design/HYDRA_ARCHIVE.md`
- strongest cited archive answers, especially:
  - `answer_18_combined.md`
  - `answer_15_combined.md`
  - `answer_14_combined.md`
  - `answer_13_combined.md`
  - `answer_16-1_combined.md`
  - `answer_20_combined.md`
  - `answer_19_combined.md`
  - `answer_21.md`

Interpretation rules:

1. Archive artifacts are evidence, not truth.
2. A surviving archive claim is **not** automatically doctrine.
3. A live seam is **not** the same thing as an active-path commitment.
4. Proposal-level rows must stay visibly proposal-level.
5. Reject/block rows must stay visibly rejected or blocked.

## Bucket meanings

| Bucket | Meaning |
|---|---|
| `Do now` | Fits current reconciliation doctrine and has the strongest current leverage or closure value |
| `Phase-next` | Promising and relevant, but still depends on explicit semantic, provenance, benchmark, or parity closure |
| `Reserve shelf` | Worth preserving, but should not steer the current mainline |
| `Blocked / reject / not active` | Practical grouping for lanes that are structurally blocked, explicitly demoted, rejected as currently scoped, or otherwise should not consume current implementation attention |

## Ranked roadmap

## Status update

- Completed (rank 1): doctrine truth-alignment pass in `research/design/HYDRA_RECONCILIATION.md`.
- Completed (rank 2): `safety_residual` semantic repair in code.
- Completed (rank 3): ExIt bridge helpers (`build_exit_from_afbs_tree`, `collate_exit_targets`) closing the AfbsTree-to-RlBatch gap.
- Completed (rank 4): advanced-head activation discipline gate pack in `hydra-train/src/training/head_gates.rs`.
- What we did (ranks 1-3):
  - corrected stale wording that still described `sample.rs` / `mjai_loader.rs` as baseline-only
  - recorded that `safety_residual` already has a live replay-builder -> sample/batch carrier -> masked-loss path
  - recorded that Stage A `belief_fields` / `mixture_weight` targets can already be emitted, while remaining semantically weak / default-off
  - kept `exit_target` and `delta_q_target` marked as real open producer-path gaps
  - patched `hydra-train/src/data/mjai_loader.rs` so `safety_residual` now uses signed replay-derived correction semantics: `exact_safety - public_score`
  - added loader, batch, augmentation, and loss-path tests that prove signed positive/negative residuals survive collation and contribute aux loss only when enabled and masked
  - tightened ExIt semantics in `hydra-train/src/training/exit.rs` to the stricter surviving archive form: compatible discard-only states, child-visit-count teacher, masked subset normalization, and visit / coverage / KL gating
  - added `exit_mask` to the RL consumer path and integration coverage proving masked ExIt loss can flow through `RlBatch`
  - added `build_exit_from_afbs_tree`: canonical bridge from AfbsTree search root to gated exit_target/exit_mask via child visit extraction
  - added `collate_exit_targets`: batch-level collation of per-sample `Option<(target, mask)>` into batch tensors with zero-masked rows for samples without exit targets
  - updated integration test to use bridge helpers instead of manual wiring
- What we did (rank 4):
  - created `hydra-train/src/training/head_gates.rs` implementing the archive gate pack from `answer_13_combined.md` sections 3.3, 5, and 6
  - density gate: per-head `rho_h` for dense heads (threshold 0.8) and `spp_h` for sparse search heads (threshold 5.0)
  - gradient conflict gate: per-head shared-trunk cosine tracking, blocks activation when negative fraction > 30%
  - `grad_cosine_from_flat`: proper gradient cosine on flattened shared-trunk vectors (replaces `grad_norm_approx` which is just a loss-magnitude proxy)
  - warmup protocol: Off -> Warmup (trunk frozen, head trains) -> Active (trunk unfrozen), with configurable countdown (default 10k steps)
  - `HeadActivationController` manages state machine for all 6 advanced heads
  - `extract_target_presence` extracts per-head sample counts from `HydraTargets` respecting per-sample masks
  - `approved_loss_config` zeros weights for Off heads, preserves Warmup/Active weights
  - 36 unit tests covering all components including full lifecycle integration with Burn NdArray backend
- Completed (rank 5): runtime ponder/cache provenance hardening in `hydra-core/src/afbs.rs` and `hydra-train/src/inference.rs`.
- What we did (rank 5):
  - added `TrustLevel` enum (LearnerOnly, Advisory, WarmStart, Authoritative) with `meets()` ordering method
  - added `CacheNamespace` enum (ObservedRoot, SpeculativeChildHint, LearnerTarget) for logical cache partitioning
  - added provenance fields to `PonderResult`: `source_net_hash: u64`, `source_version: u32`, `trust_level`, `cache_namespace`, `generation: u64`
  - updated `PonderResult::from_tree()` to accept `source_net_hash` and `source_version`; defaults trust to `LearnerOnly`, namespace to `ObservedRoot`
  - added `PonderResult::learner_only_stub()` constructor for tests and untracked producers
  - added generation tracking to `PonderCache` via `AtomicU64`; `insert()` stamps current generation, `get()` rejects stale entries
  - added `PonderCache::get_trusted()` for trust-level-gated lookups
  - added `PonderCache::invalidate()` (logical; bumps generation) and `flush()` (physical + generation bump)
  - `insert_predicted_child()` now auto-sets `CacheNamespace::SpeculativeChildHint`
  - replaced `PonderManager.cache` from raw `DashMap` to `PonderCache`; added `lookup_trusted()` and `invalidate_cache()`
  - replaced `InferenceServer.ponder_cache` from `Arc<DashMap<u64, PonderResult>>` to `Arc<PonderCache>`
  - added `InferenceServer::lookup_ponder_trusted()` and `invalidate_cache()`
  - gated `InferenceServer::infer_with_budget()` cache-hit early-return behind `TrustLevel::Authoritative` (nothing currently qualifies, keeping all ponder outputs learner-only per archive doctrine)
  - added 10 new provenance-specific tests (trust ordering, generation invalidation, flush, trust filtering, generation stamping, namespace enforcement, PonderManager provenance, from_tree provenance, learner-only runtime isolation, cache invalidation)
  - all 258 unit tests + 6 integration tests pass; clippy clean
- Not done yet:
  - no self-play loop or mainline batch construction code that constructs `RlBatch` yet (all bridge helpers are ready but the caller does not exist)
  - no `delta_q_target` closure yet
  - orchestrator integration for head gates (wiring controller into training step functions)
  - trunk detachment for warmup heads (requires model forward modification)
  - feature-ablation gate (Gate 5 from archive, requires evaluation arena)
  - full ProvenanceKey/PonderMeta/CacheDecisionAudit from answer_20 are aspirational and depend on infrastructure that does not yet exist (belief digest, policy assumption digest, CompressedAfbsTree, evaluation arena for G0-G3 re-entry gates)

### Do now

| Rank | Lane | Doctrine status | Repo status | Why now | Refs |
|---|---|---|---|---|---|
| 1 | Narrow advanced-target closure | Active-path doctrine | Completed | Reconciliation says the immediate need is supervision-loop closure, not broader search expansion. Doctrine truth-alignment pass done. | `HYDRA_RECONCILIATION.md` Recommendation 1; canonical rows 24, 34, 35, 55 |
| 2 | `safety_residual` semantic repair + narrow activation | Completed in code | Signed replay-derived residual live end-to-end | The builder, mask, batch carrier, head, and loss are now aligned on signed residual semantics; keep this lane narrow and replay-derived. | canonical rows 22, 23, 24; `answer_18_combined.md` |
| 3 | Real `exit_target` carrier and provenance closure | Completed (bridge/consumer); self-play loop is infrastructure | Bridge helpers done; remaining gap is the self-play loop that constructs RlBatch | ExIt now has helpers (`build_exit_from_afbs_tree`, `collate_exit_targets`), consumer mask support, and integration tests exercising the full AFBS-to-RL-loss path. The remaining blocker is that no non-test code constructs `RlBatch` yet. | canonical rows 34, 35; `answer_9_combined.md`, `answer_15_combined.md`, `answer_2-1_combined.md` |
| 4 | Advanced-head activation discipline | Completed in code | Gate pack implemented: density, conflict, warmup | `HeadActivationController` with density gates (`rho >= 0.8` dense, `spp >= 5` sparse), gradient conflict tracking (cosine < 30% negative), warmup protocol (Off->Warmup->Active), and `approved_loss_config` integration. 36 tests pass. | canonical row 55; `answer_13_combined.md`, `answer_3-1_combined.md` |
| 5 | Runtime ponder/cache provenance hardening | Completed in code | Provenance fields, generation tracking, trust gating implemented | `PonderResult` carries `source_net_hash`, `source_version`, `TrustLevel`, `CacheNamespace`, `generation`. `PonderCache` enforces generation freshness. `InferenceServer` gates runtime cache hits behind `Authoritative` trust (nothing qualifies, keeping everything learner-only). 10 new tests. | canonical rows 47, 48; `answer_20_combined.md`, `answer_16-1_combined.md` |

### Phase-next

| Rank | Lane | Doctrine status | Repo status | Promotion gate | Refs |
|---|---|---|---|---|---|
| 6 | H1a Hand-EV semantic repair on the current 42-plane surface | Fits current sequencing; still proposal-level | Strong live seam | Must beat current Hand-EV on exact-one-step oracle slices and stay within encode-cost limits. | canonical rows 36, 37, 38; `answer_14_combined.md`, `answer_17_combined.md` |
| 7 | Tile-aware spatial/global head routing correction | Strong archive architecture survivor; not current doctrine | Narrow problem is visible in repo now | Route tile-indexed outputs from spatial features first; do not auto-promote the whole history-sidecar package. | canonical row 56; `answer_21.md` |
| 8 | `delta_q` closure | Doctrinally interesting, not semantically closed | Mismatched surfaces, absent normal builder | Needs a single semantically aligned object, support mask, and credible training path before activation. | canonical row 29; `answer_15_combined.md`, `answer_18_combined.md`, `answer_7-1_combined.md` |
| 9 | Public-posterior belief teacher closure | Doctrine-compatible later lane | Current Stage-A path is semantically weak | Must replace Stage-A projection with a credible public teacher object before any belief/mix activation. | canonical rows 25, 26, 27, 28; `answer_15_combined.md`, `answer_18_combined.md`, `answer_3-1_combined.md` |
| 10 | World-aware CT-SMC Hand-EV (H1b) | Later selective lane only | Runtime seam exists; train/infer parity blocked | H1a must win first, CT-SMC quality gates must pass, and training-time matching search-context observations must exist. | canonical rows 39, 40, 41; `answer_14_combined.md`, `answer_1-1_combined.md` |
| 11 | Specialist endgame leaf exactification | Later/narrower than Hand-EV | Host shell exists, stronger leaf absent | Keep it as a specialist late-game lane, likely via pondering/search-side deployment, not fast-path identity. | canonical rows 42, 43; `answer_14_combined.md`, `answer_3-1_combined.md` |

### Reserve shelf

These survive as preserved options, but they should not steer the current active path.

| Rank | Lane | Why preserved | Why not mainline now | Refs |
|---|---|---|---|---|
| 12 | Tiny learned scorer for the next ponder quantum | Narrow compute-allocation seam exists and the current heuristic is replaceable | Should come after cache/provenance cleanup and should stay tiny, not grow into a runtime router | canonical rows 44, 45; `answer_19_combined.md`, `answer_10_combined.md`, `answer_5-1_combined.md` |
| 13 | Dedicated public-history path / asymmetric actor-learner package | Strongest surviving architecture challenger | Exact GRU-actor / transformer-learner package remains proposal-level; history path must earn promotion via collision/order tests | canonical row 56; `answer_21.md` |
| 14 | Action-sufficient CT-SMC world compression | Narrow benchmark-first survivor | Worth testing only as a runtime-only equal-budget falsification project | canonical row 51; `answer_11_combined.md`, `answer_8-2_combined.md`, `answer_6-1_combined.md` |
| 15 | Stronger endgame exactification | Reserve-shelf technique explicitly preserved in current archive doctrine | Narrower and later than supervision closure and Hand-EV realism | `HYDRA_ARCHIVE.md`; canonical rows 42, 43; `answer_14_combined.md` |
| 16 | Robust-opponent search backups / richer latent opponent posterior work | Preserved as last-mile strength ideas | Reconciliation keeps them reserve-only while target-generation and realism work remain open | `HYDRA_ARCHIVE.md`; canonical row 53; `HYDRA_RECONCILIATION.md` |
| 17 | DRDA/ACH as stronger optimizer/game-theory branch | Preserved as later optimizer direction | Should not compete with current target-pipeline closure | `HYDRA_ARCHIVE.md`; `HYDRA_RECONCILIATION.md` |

### Blocked / reject / not active path

| Lane | Current status | Why off | Refs |
|---|---|---|---|
| Current Stage-A `belief_fields` activation | Off | Current teacher is a public projection, not a credible public posterior teacher | canonical rows 25, 26; `answer_15_combined.md`, `answer_18_combined.md` |
| Current `mixture_weight` activation | Off | Inherits the same teacher weakness and lacks canonical component identity | canonical rows 27, 28; `answer_15_combined.md`, `answer_18_combined.md` |
| Current `opponent_hand_type` activation | Off | Typed hole: head exists, but ontology, mapping, and builder do not | canonical rows 30, 31; `answer_15_combined.md`, `answer_18_combined.md`, `answer_13_combined.md` |
| Treating `oracle_critic` as student-path closure | Off | Detached privileged branch is not evidence that public/student target closure is solved | canonical rows 32, 33; `answer_15_combined.md`, `answer_18_combined.md` |
| Broad multi-arm learned router | Blocked | Current runtime is too narrow to justify a real multi-arm routing regime | canonical row 46; `answer_10_combined.md` |
| Current rollout authority as live decisive truth | Off (provenance hardened) | Cache provenance now exists; runtime gated behind Authoritative trust (nothing qualifies) | canonical rows 47, 48; `answer_20_combined.md`, `answer_16-1_combined.md` |
| Exact rollout gates beyond `top2_policy_gap < 0.10` | Rejected | Earlier `risk_score` / `particle_ess` exact cutoffs were explicitly demoted | canonical rows 49, 50; `answer_16-1_combined.md`, `answer_16_combined.md` |
| Posterior-consensus ExIt as currently scoped | Reject | Missing public world-conditioned action-teacher object | canonical row 52; `answer_12_combined.md`, `answer_8-1_combined.md` |
| Regime-coupled opponent filtering as currently scoped | Reject | Missing emission model, regime state, and downstream consumer chain | canonical row 53; `answer_12_combined.md` |
| Broad “search everywhere” AFBS identity | Not active path | Reconciliation and archive doctrine both keep AFBS selective and specialist | `HYDRA_RECONCILIATION.md`; `HYDRA_ARCHIVE.md` |

## Dependency closure table

| Lane | Live seam now? | Main blocker | Earliest honest promotion |
|---|---|---|---|
| `safety_residual` repair | Yes | Closed for the current replay-derived lane; keep activation/provenance narrow | Completed |
| `exit_target` / conservative ExIt | Yes (bridge helpers complete) | Missing self-play loop / mainline RlBatch construction code | After self-play loop implementation |
| Advanced-head activation discipline | Yes (gate pack complete) | Orchestrator integration + trunk detachment for warmup heads | Completed (core); orchestrator wiring after self-play loop |
| `delta_q` | Partial | Dense `[46]` training surface and discard-only `[34]` runtime surface are not one closed object | After object + mask + builder closure |
| Belief supervision | Partial | Weak Stage-A teacher; no credible public posterior object in the current path | After public-teacher closure |
| H1a Hand-EV | Yes | Needs exact-one-step benchmark win | After benchmark pass |
| H1b world-aware Hand-EV | Runtime seam only | Train/infer parity + representative-world gates | After H1a + parity + CT-SMC gates |
| Endgame leaf exactification | Host shell only | Stronger leaf and explicit caller value | After offline late-game utility win |
| Tiny learned ponder scorer | Yes | Cache/provenance and label logging must be trustworthy first | After provenance hardening (done) + label logging |
| History path / asymmetric architecture | Narrow seam only | Must prove history adds value beyond current static tensor | After E0/E1-style experiment wins |

## Suggested execution order

1. Keep the active path aligned to `HYDRA_RECONCILIATION.md`. **(done)**
2. Patch `safety_residual` semantics and keep advanced activation narrow. **(done)**
3. Close real `exit_target` carrier/provenance plumbing. **(done, bridge/consumer; self-play loop is infrastructure)**
4. Add activation-density / transfer gates before broad advanced-head activation. **(done)**
5. Harden runtime ponder/cache provenance and admission boundaries. **(done)**
6. Run H1a Hand-EV exact-one-step benchmark; promote only if it clears the gate.
7. Evaluate tile-aware spatial/global head routing correction before broader architecture changes.
8. Revisit `delta_q`, belief teacher closure, and H1b only from the stronger base above.
9. Keep reserve-shelf items alive, but do not let them compete with the mainline before the earlier gates close.

## Source map

High-signal archive answers behind this roadmap:

- `answer_18_combined.md` — strongest target-semantic audit and narrow `safety_residual`-first reading
- `answer_15_combined.md` — target-provenance taxonomy, ExIt carrier gap, keep-off doctrine for belief/mix/hand-type
- `answer_14_combined.md` — H1a first, H1b later, endgame as later specialist lane
- `answer_13_combined.md` — density/interference risk and conservative staged-budget reading
- `answer_16-1_combined.md` — stricter rollout trust boundary and threshold cleanup
- `answer_20_combined.md` — cache/provenance authority critique and learner-only safe-policy recommendation
- `answer_19_combined.md` — narrow learned ponder-scorer seam
- `answer_21.md` — strongest surviving architecture lane: static compatibility path + head-routing correction + later history path

Canonical rows used most heavily:

- 22-24: `safety_residual`
- 25-31: belief / mixture / opponent-hand-type keep-off logic
- 34-35: ExIt doctrine versus missing carrier
- 36-41: Hand-EV realism, CT-SMC seam, parity blocker
- 42-45: endgame shell and tiny learned ponder scorer
- 46-50: router block, rollout provenance defect, threshold cleanup
- 51-56: compression benchmark lane, reject rows, conservative budget, activation discipline, architecture survivor

## Do not overread this file

- `Do now` here means “best archive-derived lane consistent with current doctrine,” not “new SSOT build order.”
- `Phase-next` means “promising but still gated.”
- `Reserve shelf` means “preserve, do not forget,” not “quietly promote later without re-validation.”
- `Blocked / reject / not active` mixes three different statuses on purpose only when the practical instruction is the same: do not let that lane consume current mainline attention.

When in doubt, re-read:

1. `README.md`
2. `research/design/HYDRA_RECONCILIATION.md`
3. `docs/GAME_ENGINE.md`

Only then use this file as the archive-derived prioritization layer.
