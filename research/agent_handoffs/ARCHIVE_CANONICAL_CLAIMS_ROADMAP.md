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
- Completed (rank 3): ExIt self-play loop (`generate_self_play_rl_batch`) and producer (`SelfPlayExitAdapter`) wired. Validation harness ran on 1759 states (5/6 infrastructure criteria passed; top-1 agreement requires trained weights). `LiveExitConfig::default().enabled` flipped to `true`. `SelfPlayExitAdapter` tile-lookup bug fixed.
- `delta_q_target` closure deferred (`keep-off-blocked` per answer_23 doctrine)
- orchestrator integration for head gates (wiring controller into training step functions)
- trunk detachment for warmup heads (requires model forward modification)
- feature-ablation gate (Gate 5 from archive, requires evaluation arena)
- full ProvenanceKey/PonderMeta/CacheDecisionAudit from answer_20 are aspirational and depend on infrastructure that does not yet exist (belief digest, policy assumption digest, CompressedAfbsTree, evaluation arena for G0-G3 re-entry gates)

### Do now

| Rank | Lane | Doctrine status | Repo status | Why now | Refs |
|---|---|---|---|---|---|
| 1 | Narrow advanced-target closure | Active-path doctrine | Completed | Reconciliation says the immediate need is supervision-loop closure, not broader search expansion. Doctrine truth-alignment pass done. | `HYDRA_RECONCILIATION.md` Recommendation 1; canonical rows 24, 34, 35, 55 |
| 2 | `safety_residual` semantic repair + narrow activation | Completed in code | Signed replay-derived residual live end-to-end | The builder, mask, batch carrier, head, and loss are now aligned on signed residual semantics; keep this lane narrow and replay-derived. | canonical rows 22, 23, 24; `answer_18_combined.md` |
| 3 | Real `exit_target` carrier and provenance closure | Completed | Self-play loop, producer wired, default-on | ExIt now has bridge helpers, consumer mask support, a live self-play loop (`generate_self_play_rl_batch`) with search-derived labels via `SelfPlayExitAdapter`, and the producer is default-on (`LiveExitConfig.enabled = true`) after infrastructure validation. | canonical rows 34, 35; `answer_9_combined.md`, `answer_15_combined.md`, `answer_2-1_combined.md` |

Additional narrowing from `answer_22.md`: if Hydra closes the live AFBS ExIt producer, the surviving archive verdict is now narrower than the older broad ExIt discussion. Teacher semantics should remain root child visits, `root_exit_policy()` / q-softmax should not be promoted into the teacher object, and the only narrow surviving evaluator source is the current public model value head used inside learner-only, root-only AFBS. The producer is now default-on after infrastructure validation cleared 5/6 criteria (top-1 agreement deferred to trained-model re-validation).
| 4 | Advanced-head activation discipline | Completed in code | Gate pack implemented: density, conflict, warmup | `HeadActivationController` with density gates (`rho >= 0.8` dense, `spp >= 5` sparse), gradient conflict tracking (cosine < 30% negative), warmup protocol (Off->Warmup->Active), and `approved_loss_config` integration. 36 tests pass. | canonical row 55; `answer_13_combined.md`, `answer_3-1_combined.md` |
| 5 | Runtime ponder/cache provenance hardening | Completed in code | Provenance fields, generation tracking, trust gating implemented | `PonderResult` carries `source_net_hash`, `source_version`, `TrustLevel`, `CacheNamespace`, `generation`. `PonderCache` enforces generation freshness. `InferenceServer` gates runtime cache hits behind `Authoritative` trust (nothing qualifies, keeping everything learner-only). 10 new tests. | canonical rows 47, 48; `answer_20_combined.md`, `answer_16-1_combined.md` |

### Phase-next

| Rank | Lane | Doctrine status | Repo status | Promotion gate | Refs |
|---|---|---|---|---|---|
| 6 | H1a Hand-EV semantic repair on the current 42-plane surface | Fits current sequencing; still proposal-level | Strong live seam | Must beat current Hand-EV on exact-one-step oracle slices and stay within encode-cost limits. | canonical rows 36, 37, 38; `answer_14_combined.md`, `answer_17_combined.md` |
| 7 | Tile-aware spatial/global head routing correction | Strong archive architecture survivor; not current doctrine | Narrow problem is visible in repo now | Route tile-indexed outputs from spatial features first; do not auto-promote the whole history-sidecar package. | canonical row 56; `answer_21.md` |
| 8 | `delta_q` closure | Doctrinally interesting, not semantically closed | Mismatched surfaces, absent normal builder | Needs a masked `[46]` root-child q-delta object, explicit `[46]` support mask, shared root-search producer, and validation-backed training path before activation. | canonical rows 29, 30; `answer_15_combined.md`, `answer_18_combined.md`, `answer_7-1_combined.md`, `answer_23_combined.md` |
| 9 | Public-posterior belief teacher closure | Doctrine-compatible later lane | Current Stage-A path is semantically weak | Must replace Stage-A projection with a credible public teacher object before any belief/mix activation. | canonical rows 25, 26, 27, 28; `answer_15_combined.md`, `answer_18_combined.md`, `answer_3-1_combined.md` |
| 10 | World-aware CT-SMC Hand-EV (H1b) | Later selective lane only | Runtime seam exists; train/infer parity blocked | H1a must win first, CT-SMC quality gates must pass, and training-time matching search-context observations must exist. | canonical rows 39, 40, 41; `answer_14_combined.md`, `answer_1-1_combined.md` |
| 11 | Specialist endgame leaf exactification | Later/narrower than Hand-EV | Host shell exists, stronger leaf absent | Keep it as a specialist late-game lane, likely via pondering/search-side deployment, not fast-path identity. | canonical rows 42, 43; `answer_14_combined.md`, `answer_3-1_combined.md` |

Additional narrowing from `answer_23_combined.md`: current doctrine still keeps `delta_q_target` off. The only surviving honest closure lane is learner-only, root-only, search-derived `Q(child) - Q(root)` supervision over discard-compatible actions, preserved in the existing `[46]` head space via an explicit `[46]` mask. Hydra should not zero-extend the bridge/runtime `[34]` plane into training supervision; it should first close one shared ExIt/`delta_q` producer, provenance contract, and validation envelope.

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
| `exit_target` / conservative ExIt | Yes | None (self-play loop wired, producer default-on) | Completed |
| Advanced-head activation discipline | Yes (gate pack complete) | Orchestrator integration + trunk detachment for warmup heads | Completed (core); orchestrator wiring after self-play loop |
| `delta_q` | Partial | Dense `[46]` training surface and discard-only `[34]` runtime surface are not one closed object; shared masked producer and validation closure are still missing | After masked object + shared producer + validation closure |
| Belief supervision | Partial | Weak Stage-A teacher; no credible public posterior object in the current path | After public-teacher closure |
| H1a Hand-EV | Yes | Needs exact-one-step benchmark win | After benchmark pass |
| H1b world-aware Hand-EV | Runtime seam only | Train/infer parity + representative-world gates | After H1a + parity + CT-SMC gates |
| Endgame leaf exactification | Host shell only | Stronger leaf and explicit caller value | After offline late-game utility win |
| Tiny learned ponder scorer | Yes | Cache/provenance and label logging must be trustworthy first | After provenance hardening (done) + label logging |
| History path / asymmetric architecture | Narrow seam only | Must prove history adds value beyond current static tensor | After E0/E1-style experiment wins |

## Suggested execution order

1. Keep the active path aligned to `HYDRA_RECONCILIATION.md`. **(done)**
2. Patch `safety_residual` semantics and keep advanced activation narrow. **(done)**
3. Close real `exit_target` carrier/provenance plumbing. **(done)**
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
- `answer_22.md` — narrow live-AFBS ExIt evaluator verdict: visits stay the teacher object; public value head is the only surviving narrow evaluator; producer now default-on after infrastructure validation
- `answer_23_combined.md` — delta_q closure blueprint: keep the lane off today; surviving future object is masked `[46]` root-child q-delta over discard-compatible actions with shared ExIt/search-label producer, provenance, and validation closure

Canonical rows used most heavily:

- 22-24: `safety_residual`
- 25-31: belief / mixture / opponent-hand-type keep-off logic
- 29-30: `delta_q` blocker plus masked future closure object
- 34-35: ExIt doctrine versus missing carrier
- 41: live-AFBS ExIt evaluator semantics (visit teacher, reject q-softmax teacher, value-head evaluator under learner-only root-only AFBS)
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
