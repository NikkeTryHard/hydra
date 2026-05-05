# Archive Canonical Claims Roadmap

> Derived archive triage view.
>
> File ranks archive-derived research lanes from `ARCHIVE_CANONICAL_CLAIMS.jsonl`, source ledger for Hydra canonical archive SSOT.
> File does **not** alone settle promoted doctrine or runtime truth.
>
> Layered routing:
> 1. `README.md`
> 2. `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl`
> 3. `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md` and this roadmap
> 4. `research/design/HYDRA_FINAL.md` and `research/design/HYDRA_RECONCILIATION.md`
> 5. `docs/GAME_ENGINE.md` and current code
>
> If file conflicts with JSONL source ledger, JSONL wins.
> If archive view or promoted doctrine drifts from current code/runtime,
> refresh lagging summary, not demote upstream source.
>
> Warning:
> archive roadmap intentionally preserves archive triage, may lag current repo truth.
> Known drift updated in-place when found, but future code/docs may still move faster than file.
> Read any ExIt, `delta_q`, or belief-teacher status here through current code/runtime and promoted docs before claiming live.

## Scope

Roadmap exists answer one narrow question:

- given surviving archive claims,
- what treat as **do now**,
- what is **phase-next**,
- what belongs on **reserve shelf**,
- what remains **blocked, rejected, or not active path**.

File = archive prioritization layer, not self-executing build-order memo.
Use after reading JSONL source ledger, before trusting or refreshing downstream doctrine summaries.

## Workflow note

When using any short-form next-step triage helper for repo, start from canonical archive layer, then check promoted doctrine plus runtime reality before implementing.

Helper should:
- read `ARCHIVE_CANONICAL_CLAIMS.jsonl` first
- check whether promoted doctrine and runtime docs refreshed after
- bias toward safest highest-impact next step
- report whether noisy archive surfaces actually checked

It does **not** turn derived archive view into self-executing doctrine, and does **not** outrank JSONL source ledger.

## Inputs and interpretation rules

Primary evidence base:

- `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl`
- `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_RENDERED.md` (generated human-readable render)
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

1. Canonical archive claims = upstream research truth, not automatic proof of code/runtime reality.
2. Raw combined answers are **not** same as canonical archive claims.
3. Surviving archive claim is **not** automatic already-implemented active-path commitment.
4. Proposal-level rows must stay visibly proposal-level.
5. Reject/block rows must stay visibly rejected or blocked.

## Bucket meanings

| Bucket | Meaning |
|---|---|
| `Do now` | Fits current reconciliation doctrine, has strongest current leverage or closure value |
| `Phase-next` | Promising, relevant, but still needs explicit semantic, provenance, benchmark, or parity closure |
| `Reserve shelf` | Worth preserving, should not steer current mainline |
| `Blocked / reject / not active` | Practical grouping for lanes structurally blocked, explicitly demoted, rejected as currently scoped, or otherwise should not consume current implementation attention |

## Ranked roadmap

## Status update

- Completed (rank 1): doctrine truth-alignment pass in `research/design/HYDRA_RECONCILIATION.md`.
- Completed (rank 2): `safety_residual` semantic repair in code.
- Completed (supporting tranche infrastructure): learner BC now supports hardware-agnostic microbatch accumulation, and `src/bin/train.rs` can stage narrow replay-derived `safety_residual` activation without opening weaker advanced-target lanes.
- Completed (supporting BC runtime contract): learner BC now keeps fresh runs config-derived, preserves strict partial-epoch runtime identity, and may reuse matching preflight-selected runtime for selected-runtime tuple at epoch boundaries only; loader-runtime remains config-derived.
- Completed (rank 3): ExIt self-play loop (`generate_self_play_rl_batch`) and producer (`SelfPlayExitAdapter`) wired. Validation harness ran on 1759 states (5/6 infrastructure criteria passed; top-1 agreement needs trained weights). `LiveExitConfig::default().enabled` flipped to `true`. `SelfPlayExitAdapter` tile-lookup bug fixed.
- Completed (rank 3 supporting lane): replay/sample ExIt sidecar-first producer/join path in code with provenance/version checks and BC-side optional ExIt loss support.
- Completed (rank 6): `delta_q` closure across replay/offline sidecar provenance, BC/train activation, and warmup detach.
- feature-ablation gate (Gate 5 from archive, needs evaluation arena)
- full ProvenanceKey/PonderMeta/CacheDecisionAudit from answer_20 are aspirational, depend on infrastructure not yet existing (belief digest, policy assumption digest, CompressedAfbsTree, evaluation arena for G0-G3 re-entry gates)

### Do now

| Rank | Lane | Doctrine status | Repo status | Why now | Refs |
|---|---|---|---|---|---|
| 1 | Narrow `safety_residual` / BC activation sublane | Active-path doctrine | Completed | Reconciliation says immediate need = supervision-loop closure, not broader search expansion. Completed part = replay-derived `safety_residual` BC lane plus hardware-agnostic BC microbatch accumulation, and current BC runtime authority now keeps fresh runs config-derived while allowing epoch-boundary-only reuse of matching preflight-selected runtime for selected-runtime tuple. Broader Recommendation 1 closure still open. | `HYDRA_RECONCILIATION.md` Recommendation 1; canonical rows 24, 34, 35, 55 |
| 2 | `safety_residual` semantic repair + narrow activation | Completed in code | Signed replay-derived residual live end-to-end, with BC-path staged activation and microbatch accumulation available for learner training | Builder, mask, batch carrier, head, loss now aligned on signed residual semantics; train binary can enable only narrow advanced loss while keeping weaker lanes blocked, and BC no longer assumes one machine's VRAM shape. Keep lane replay-derived and narrow. | canonical rows 22, 23, 24; `answer_18_combined.md` |
| 3 | ExIt carrier closure across live self-play and replay/sample sidecar lane | Completed in code | Live self-play loop, replay/sample sidecar producer/join, BC consumption, default-on live producer | ExIt now has bridge helpers, consumer mask support, live self-play loop (`generate_self_play_rl_batch`) with search-derived labels via `SelfPlayExitAdapter`, plus replay/sample sidecar-first lane that generates replay-indexed search-derived labels, joins back into replay samples with provenance/version checks, and feeds BC as separate optional ExIt loss. | canonical rows 34, 35, 51, 52; `answer_9_combined.md`, `answer_15_combined.md`, `answer_24_combined.md` |
| 4 | Advanced-head activation discipline | Completed in code | Gate pack implemented: density, conflict, warmup | `HeadActivationController` with density gates (`rho >= 0.8` dense, `spp >= 5` sparse), gradient conflict tracking (cosine < 30% negative), warmup protocol (Off->Warmup->Active), and `approved_loss_config` integration. 36 tests pass. | canonical row 55; `answer_13_combined.md`, `answer_3-1_combined.md` |
| 5 | Runtime ponder/cache provenance hardening | Completed in code | Provenance fields, generation tracking, trust gating implemented | `PonderResult` carries `source_net_hash`, `source_version`, `TrustLevel`, `CacheNamespace`, `generation`. `PonderCache` enforces generation freshness. `InferenceServer` gates runtime cache hits behind `Authoritative` trust (nothing qualifies, so all stays learner-only). 10 new tests. | canonical rows 47, 48; `answer_20_combined.md`, `answer_16-1_combined.md` |

### Phase-next

| Rank | Lane | Doctrine status | Repo status | Promotion gate | Refs |
|---|---|---|---|---|---|
| 6 | Public-posterior belief teacher closure | Stronger baseline tranche shipped; fuller closure still later | Current Stage-A path still semantically weaker than archive final public-posterior target object, but stronger baseline belief tranche now shipped in code | Keep belief-before-mixture discipline, preserve staged `mixture_weight`, promote fuller public-posterior / row-conditional closure only when narrower target object actually lands. | canonical rows 25, 26, 27, 28, 61, 62; `answer_15_combined.md`, `answer_18_combined.md`, `answer_27_combined.md`, `answer_28_combined.md`, `answer_3-1_combined.md` |
| 7 | H1a Hand-EV semantic repair on current 42-plane surface | Stronger realism baseline shipped; fuller H1a benchmark/promotion still later | Strong live seam with materially stronger shipped local evaluation already in repo | Use exact-one-step oracle benchmarking only for remaining H1a promotion question, not as evidence baseline tranche still missing. | canonical rows 36, 37, 38; `answer_14_combined.md`, `answer_17_combined.md` |
| 8 | Tile-aware spatial/global head routing correction | Strong archive architecture survivor; not current doctrine | Narrow problem visible in repo now | Route tile-indexed outputs from spatial features first; do not auto-promote whole history-sidecar package. | canonical row 56; `answer_21.md` |
| 9 | World-aware CT-SMC Hand-EV (H1b) | Later selective lane only | Runtime seam exists; train/infer parity blocked | H1a must win first, CT-SMC quality gates must pass, and training-time matching search-context observations must exist. | canonical rows 39, 40, 41; `answer_14_combined.md`, `answer_1-1_combined.md` |
| 10 | Specialist endgame leaf exactification | Later/narrower than Hand-EV | Host shell exists, stronger leaf absent | Keep as specialist late-game lane, likely via pondering/search-side deployment, not fast-path identity. | canonical rows 42, 43; `answer_14_combined.md`, `answer_3-1_combined.md` |

### Reserve shelf

These survive as preserved options, but should not steer current active path.

| Rank | Lane | Why preserved | Why not mainline now | Refs |
|---|---|---|---|---|
| 12 | Tiny learned scorer for the next ponder quantum | Narrow compute-allocation seam exists, current heuristic replaceable | Should come after cache/provenance cleanup and stay tiny, not grow into runtime router | canonical rows 44, 45; `answer_19_combined.md`, `answer_10_combined.md`, `answer_5-1_combined.md` |
| 13 | Dedicated public-history path / asymmetric actor-learner package | Strongest surviving architecture challenger | Exact GRU-actor / transformer-learner package remains proposal-level; history path must earn promotion via collision/order tests | canonical row 56; `answer_21.md` |
| 14 | Action-sufficient CT-SMC world compression | Narrow benchmark-first survivor | Worth testing only as runtime-only equal-budget falsification project | canonical row 51; `answer_11_combined.md`, `answer_8-2_combined.md`, `answer_6-1_combined.md` |
| 15 | Stronger endgame exactification | Reserve-shelf technique explicitly preserved in current archive doctrine | Narrower and later than supervision closure and Hand-EV realism | `HYDRA_ARCHIVE.md`; canonical rows 42, 43; `answer_14_combined.md` |
| 16 | Robust-opponent search backups / richer latent opponent posterior work | Preserved as last-mile strength ideas | Reconciliation keeps reserve-only while target-generation and realism work remain open | `HYDRA_ARCHIVE.md`; canonical row 53; `HYDRA_RECONCILIATION.md` |
| 17 | DRDA/ACH as stronger optimizer/game-theory branch | Preserved as later optimizer direction | Should not compete with current target-pipeline closure | `HYDRA_ARCHIVE.md`; `HYDRA_RECONCILIATION.md` |

### Blocked / reject / not active path

| Lane | Current status | Why off | Refs |
|---|---|---|---|
| Current Stage-A `belief_fields` activation | Off | Current teacher = public projection, not credible public posterior teacher; current component axis = repeated aggregate structure, and loss/object pairing still needs row-conditional repair | canonical rows 25, 26, 61, 62; `answer_15_combined.md`, `answer_18_combined.md`, `answer_27_combined.md`, `answer_28_combined.md` |
| Current `mixture_weight` activation | Off | Inherits same teacher weakness, and first honest closure keeps mixture off until canonical component identity exists | canonical rows 27, 28, 62; `answer_15_combined.md`, `answer_18_combined.md`, `answer_27_combined.md`, `answer_28_combined.md` |
| Current `opponent_hand_type` activation | Off | Typed hole: head exists, but ontology, mapping, builder do not | canonical rows 30, 31; `answer_15_combined.md`, `answer_18_combined.md`, `answer_13_combined.md` |
| Treating `oracle_critic` as student-path closure | Off | Detached privileged branch is not evidence public/student target closure solved | canonical rows 32, 33; `answer_15_combined.md`, `answer_18_combined.md` |
| Broad multi-arm learned router | Blocked | Current runtime too narrow to justify real multi-arm routing regime | canonical row 46; `answer_10_combined.md` |
| Current rollout authority as live decisive truth | Off (provenance hardened) | Cache provenance now exists; runtime gated behind Authoritative trust (nothing qualifies) | canonical rows 47, 48; `answer_20_combined.md`, `answer_16-1_combined.md` |
| Exact rollout gates beyond `top2_policy_gap < 0.10` | Rejected | Earlier `risk_score` / `particle_ess` exact cutoffs explicitly demoted | canonical rows 49, 50; `answer_16-1_combined.md`, `answer_16_combined.md` |
| Posterior-consensus ExIt as currently scoped | Reject | Missing public world-conditioned action-teacher object | canonical row 52; `answer_12_combined.md`, `answer_8-1_combined.md` |
| Regime-coupled opponent filtering as currently scoped | Reject | Missing emission model, regime state, downstream consumer chain | canonical row 53; `answer_12_combined.md` |
| Broad “search everywhere” AFBS identity | Not active path | Reconciliation and archive doctrine both keep AFBS selective and specialist | `HYDRA_RECONCILIATION.md`; `HYDRA_ARCHIVE.md` |

## Dependency closure table

Historical-note warning:
- table preserves archive-derived closure shapes, may still lag future repo updates.
- rows in table truth-aligned where known drift found, but current authority docs/code still win if table falls behind again.

| Lane | Live seam now? | Main blocker | Earliest honest promotion |
|---|---|---|---|
| `safety_residual` repair | Yes | Closed for current replay-derived lane; keep activation/provenance narrow | Completed |
| `exit_target` / conservative ExIt | Yes | Live self-play and replay/sample lanes both closed, but provenance must stay explicit and replay-pure boundary intact | Completed in code |
| Replay-indexed ExIt sidecar closure | Yes | Keep replay/sample ExIt search-derived and sidecar-first; do not collapse into ordinary replay truth | Completed in code |
| Advanced-head activation discipline | Yes (gate pack complete) | Orchestrator integration + trunk detachment for warmup heads | Completed (core); orchestrator wiring after self-play loop |
| `delta_q` | Yes | Closed for current learner-only replay/offline + BC/train lane; keep provenance explicit and do not broaden belief-style lanes by analogy | Completed in code |
| Belief supervision | Yes (stronger baseline shipped) | Full public-posterior teacher object and row-conditional loss repair remain later than shipped baseline tranche; `mixture_weight` stays staged | After fuller public-teacher closure |
| H1a Hand-EV | Yes (stronger baseline shipped) | Exact-one-step benchmark still decides whether archive fuller H1a target is promoted beyond shipped realism tranche | After benchmark pass |
| H1b world-aware Hand-EV | Runtime seam only | Train/infer parity + representative-world gates | After H1a + parity + CT-SMC gates |
| Endgame leaf exactification | Host shell only | Stronger leaf and explicit caller value | After offline late-game utility win |
| Tiny learned ponder scorer | Yes | Cache/provenance and label logging must be trustworthy first | After provenance hardening (done) + label logging |
| History path / asymmetric architecture | Narrow seam only | Must prove history adds value beyond current static tensor | After E0/E1-style experiment wins |

## Suggested execution order

1. Keep active path aligned to `HYDRA_RECONCILIATION.md`. **(done)**
2. Patch `safety_residual` semantics, keep advanced activation narrow. **(done)**
3. Close real `exit_target` carrier/provenance plumbing for live self-play lane. **(done)**
4. Add activation-density / transfer gates before broad advanced-head activation. **(done)**
5. Harden runtime ponder/cache provenance and admission boundaries. **(done)**
6. Treat shipped stronger belief baseline and Hand-EV realism baseline as new floor, not still-open next work.
7. If belief work resumes, target fuller public-posterior / row-conditional closure while keeping `mixture_weight` staged.
8. If Hand-EV work resumes, run H1a exact-one-step benchmark, promote only if gate clears.
9. Evaluate tile-aware spatial/global head routing correction before broader architecture changes.
10. Revisit H1b and other later selective lanes only from stronger base above, and keep reserve-shelf items alive without letting them compete with mainline prematurely.

## Source map

## Narrowing / status-update notes

These notes preserve high-signal archive narrowing and post-roadmap truth alignment without interrupting ranked roadmap tables above.

- Additional narrowing from `answer_22.md`: now that Hydra closed live AFBS ExIt producer lane, surviving archive verdict is narrower than older broad ExIt discussion. Teacher semantics stay root child visits, `root_exit_policy()` / q-softmax do not become teacher object, and only narrow surviving evaluator source is current public model value head used inside learner-only, root-only AFBS. Producer now default-on after infrastructure validation cleared 5/6 criteria (top-1 agreement still needs trained-model re-validation). Read as live self-play/producer status update, not claim that every training data path in repo now carries `exit_target`.

- Additional narrowing from `answer_23_combined.md`: surviving honest `delta_q` object was learner-only, root-only, search-derived `Q(child) - Q(root)` supervision over discard-compatible actions, preserved in existing `[46]` head space via explicit `[46]` mask. Closure now implemented in code across replay/offline sidecar provenance, sample/batch carriage, and BC/train activation, and archive warning against zero-extending bridge/runtime `[34]` plane remains right reading of what shipped.

- Additional narrowing from `answer_29_combined.md`: shipped narrow `delta_q` lane now best read as code-closed and promotion-gated through arena confirmation. Current code now has teacher-regret / policy-transfer reporting, dedicated promotion mode, persisted promotion artifacts, paired arena-confirmation executor, and explicit arena proof surfaces (`arena_decision`, `arena_report`). Remaining honest follow-up is no longer carrier plumbing or executor itself; it is downstream policy/doctrine question of whether and when DeltaQ should become default-on.

- Additional narrowing from `answer_25_combined.md`: RL-only validation follow-up survives as correctness requirement on now-implemented lane. Sparse-head activation accounting must count only rows whose `[46]` `delta_q_mask` has nonzero support, and invalid `(Some, None)` / `(None, Some)` target-mask pairs stay explicit plumbing failures rather than silently degrading into zero-loss behavior. Those pair-safety constraints now reflected in code and should be treated as completed narrowing, not future doctrine.

- Additional narrowing from `answer_26_combined.md`: replay/offline `delta_q` did correctly close through replay-indexed search-derived sidecar pattern rather than replay-built loader path, and separate BC/train activation-hook blocker now also closed in code through caller-side `HeadActivationController` integration plus warmup-time trunk detachment for `DeltaQ`. Keep archive split as historical explanation for implementation order, not current blocker.

- Additional narrowing from `answer_27_combined.md` and `answer_28_combined.md`: sharper surviving belief reading is narrower than older generic “Stage-A is weak” shorthand. Current Stage-A belief is one duplicated aggregate projection object cloned across four component blocks rather than real multimodal teacher, so first honest closure is belief-before-mixture: repair teacher object to public-posterior expected hidden allocation, repair loss/object pairing to row-conditional belief supervision, and keep `mixture_weight` off until canonical component identity exists.

- Additional narrowing from `answer_24_combined.md`: normal replay/sample loader should stay replay-pure for ExIt. Lane now closed in code through replay-indexed, search-derived offline sidecar producer plus supervised join path, and should now be treated as pattern for provenance-explicit search-derived labels rather than unfinished future work.

- Additional narrowing from `answer_32_combined.md`: shipped replay-derived `safety_residual` semantics are not missing piece anymore. Next honest follow-up is promotion-discipline closure: add validator/reporting visibility, close fail-fast target/mask parity, and judge behavior on recomposed safety (`public_score + residual`) while keeping lane discard-only, replay-derived, and narrower than `delta_q`.

High-signal archive answers behind roadmap:

- `answer_18_combined.md` — strongest target-semantic audit and narrow `safety_residual`-first reading
- `answer_15_combined.md` — target-provenance taxonomy, historical ExIt carrier gap, keep-off doctrine for belief/mix/hand-type
- `answer_14_combined.md` — H1a first, H1b later, endgame as later specialist lane
- `answer_13_combined.md` — density/interference risk and conservative staged-budget reading
- `answer_16-1_combined.md` — stricter rollout trust boundary and threshold cleanup
- `answer_20_combined.md` — cache/provenance authority critique and learner-only safe-policy recommendation
- `answer_19_combined.md` — narrow learned ponder-scorer seam
- `answer_21.md` — strongest surviving architecture lane: static compatibility path + head-routing correction + later history path
- `answer_22.md` — narrow live-AFBS ExIt evaluator verdict: visits stay teacher object; public value head is only surviving narrow evaluator; producer now default-on after infrastructure validation (older default-off wording is historical)
- `answer_23_combined.md` — delta_q closure blueprint: masked `[46]` root-child q-delta over discard-compatible actions with shared ExIt/search-label producer, provenance, and validation closure; keep-off wording now historical for completed lane
- `answer_24_combined.md` — replay/sample ExIt producer-path narrowing: keep ordinary loader replay-pure, classify replay-indexed offline ExIt as search-derived, and treat sidecar-first lane as completed in current code
- `answer_25_combined.md` — RL-only delta_q validation narrowing: mixed masked batches must use nonzero-mask row counting for sparse-head activation accounting, and invalid target/mask pair states stay keep-off plumbing failures
- `answer_26_combined.md` — replay/offline delta_q provenance narrowing: close replay/offline labels only through replay-indexed search-derived sidecar contract; later BC/train activation-hook blocker now historical for completed lane
- `answer_27_combined.md` — repo-grounded belief-closure narrowing: preserve current carrier surfaces and train-bin safety gates, allow belief before mixture only after stronger teacher plus row-conditional loss repair, and keep mixture off in v1
- `answer_28_combined.md` — semantic-object belief narrowing: current Stage-A is one duplicated aggregate object, and honest v1 teacher is public posterior expected hidden allocation rather than fake 4-component mixture
- `answer_29_combined.md` — DeltaQ promotion-gap narrowing: lane is code-closed but not promotion-closed, so current structural validation should not be overread as decision-quality proof
- `answer_30_combined.md` — routing-vs-DeltaQ head-to-head follow-up: largely reaffirmed existing delta_q-first bounded tranche without changing current routing rank or doctrine status
- `answer_31_combined.md` — runtime-authority follow-up: preserves exact observed-root proving shape, but keeps runtime authority as later challenger rather than new mainline lane
- `answer_32_combined.md` — SafetyResidual promotion-discipline narrowing: preserve replay-derived semantics, add validator/pair-discipline closure next, and judge behavior on recomposed safety rather than raw residuals

Canonical rows used most heavily:

- 22-24: `safety_residual`
- 25-31, 62-63: belief / mixture keep-off logic plus sharper Stage-A duplicated-object diagnosis and belief-before-mixture closure narrowing
- 40-41: `delta_q` blocker plus masked future closure object
- 60-61, 64: shipped `delta_q` lane closure plus later validation/provenance narrowing and still-open promotion-gap reading
- 65: shipped `safety_residual` semantics closed, but promotion discipline remains later follow-up
- 34-35: ExIt doctrine versus missing carrier
- 68-69: replay-pure loader boundary plus replay-indexed ExIt search-derived provenance / sidecar closure lane
- 41: live-AFBS ExIt evaluator semantics (visit teacher, reject q-softmax teacher, value-head evaluator under learner-only root-only AFBS)
- 36-41: Hand-EV realism, CT-SMC seam, parity blocker
- 42-45: endgame shell and tiny learned ponder scorer
- 46-50: router block, rollout provenance defect, threshold cleanup
- 51-56: compression benchmark lane, reject rows, conservative budget, activation discipline, architecture survivor

## Do not overread this file

- `Do now` here means “best archive-derived lane from canonical archive layer,” not “proof that downstream doctrine and runtime docs were already refreshed.”
- `Phase-next` means “promising but still gated.”
- `Reserve shelf` means “preserve, do not forget,” not “quietly promote later without re-validation.”
- `Blocked / reject / not active` mixes three different statuses on purpose only when practical instruction is same: do not let lane consume current mainline attention.

When in doubt, re-read:

1. `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl`
2. `research/design/HYDRA_FINAL.md`
3. `research/design/HYDRA_RECONCILIATION.md`
4. `docs/GAME_ENGINE.md`

Only then use file as archive-derived prioritization layer.