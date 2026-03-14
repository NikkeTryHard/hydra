# Hydra Reconciliation

> **Current execution doctrine.**
>
> If any implementation/reference doc conflicts with this file on sequencing, tranche priority, or active-vs-reserve status, this file wins.

This memo reconciles the strongest design inputs, the actual repository state, and the best immediate next move.

It is intentionally opinionated:
- keep the strongest old ideas in a reserve shelf so they are not lost
- remove weak or distracting ideas from the active path
- define the clearest version of Hydra to build right now

Scope:
- Target architecture authority: `research/design/HYDRA_FINAL.md`
- Verified code reality: `crates/hydra-core/`, `crates/hydra-train/`
- Deep-agent inputs: `ANSWER_1.md`, `ANSWER_2.md`, `ANSWER_3.md`
- High-impact drift only; not a full doc rewrite

## 1. Executive synthesis

The three answer files point in the same broad direction, but they are useful for different reasons.

- `ANSWER_1.md` is strongest on technical fill-ins:
  - unified belief stack
  - AFBS node semantics
  - factorized threat modeling
  - Hand-EV formulas
  - endgame trigger ideas
- `ANSWER_2.md` is strongest on repo-aware loop closure:
  - advanced heads exist but are not fully trained from real targets
  - AFBS exists as a shell, not a full information-state search runtime
  - Hand-EV and endgame are present but still too shallow
- `ANSWER_3.md` is strongest on strategic pruning:
  - do not let novelty outrank strength-per-effort
  - do not center the roadmap on broad expensive search
  - reduce architectural confusion before scaling implementation

Main consensus:
- Do not restart Hydra from zero.
- The biggest blocker is not missing files; it is partially closed loops plus doc drift.
- Stronger target generation is a better immediate lever than a giant search rewrite.
- AFBS should be selective and specialist, not the default path everywhere.
- Hand-EV is worth moving earlier than deeper AFBS expansion.

Main disagreements:
- How aggressively to keep DRDA/ACH on the critical path.
- How early opponent-latent and robust-opponent logic should move from helper math into the main runtime.
- How much to invest now in search semantics versus cheaper supervision and feature realism.

Best combined reading:
- Keep `HYDRA_FINAL.md` as the north star.
- Treat current code as a strong baseline with real advanced components already present.
- Fix guidance drift first.
- Make the first coding tranche close advanced target-generation and supervision loops before spending more engineering on deeper AFBS integration.

Working principle for this memo:
- **active path** = what the team should optimize for now
- **reserve shelf** = good ideas kept for later if the active path underdelivers
- **drop shelf** = ideas that should stop consuming mainline attention for now

## Doctrine routing

| Need | Primary file |
|---|---|
| Architecture north star | `HYDRA_FINAL.md` |
| Current execution doctrine | `HYDRA_RECONCILIATION.md` |
| Current runtime reality | `docs/GAME_ENGINE.md` |
| Historical / reserve-only planning surfaces | `HYDRA_ARCHIVE.md` |

Workflow helper note:
- Any short-form next-step triage helper must read the authority chain above, bias toward the safest highest-impact next task, and explicitly report which noisy research surfaces were checked.
- It is workflow tooling, not doctrine, and it never outranks this file.

## 2. Verified repo reality

What is confirmed in code today:

- Real advanced modules exist:
  - `hydra-core/src/ct_smc.rs`
  - `hydra-core/src/sinkhorn.rs`
  - `hydra-core/src/afbs.rs`
  - `hydra-core/src/robust_opponent.rs`
  - `hydra-core/src/hand_ev.rs`
  - `hydra-core/src/endgame.rs`
- The encoder already moved beyond the old baseline and now exposes a fixed-superset tensor:
  - `hydra-core/src/encoder.rs`
  - `NUM_CHANNELS = 192`
- The train model already includes advanced heads structurally:
  - `hydra-train/src/model.rs`
  - `hydra-train/src/heads.rs`
- The self-play/live ExIt lane is no longer just helper code:
  - `hydra-train/src/training/live_exit.rs`
  - `hydra-train/src/training/exit_validation.rs`
  - `hydra-train/src/selfplay.rs`
- Advanced-head activation discipline is implemented as an explicit controller:
  - `hydra-train/src/training/head_gates.rs`
- Runtime ponder/cache provenance hardening is implemented:
  - `hydra-core/src/afbs.rs`
  - `hydra-train/src/inference.rs`

What is only partially true:

- Advanced losses exist, but default advanced loss weights are zero:
  - `hydra-train/src/training/losses.rs`
- Advanced supervision hooks exist, but target closure is uneven rather than absent:
  - `hydra-train/src/data/sample.rs` / `hydra-train/src/data/mjai_loader.rs` already carry replay-derived `safety_residual` and can emit Stage A `belief_fields` / `mixture_weight` targets
  - `hydra-train/src/training/rl.rs` can already consume `exit_target`, and the self-play/live producer path now carries `exit_target` / `exit_mask` through `TrajectoryExitLabel` into `RlBatch`
- the normal replay/sample batch path now emits `exit_target` through a sidecar-first search-derived path rather than a direct replay-derived builder, so ExIt closure is now real in both the live self-play lane and the replay/sample sidecar lane while still remaining distinct by provenance
  - `delta_q_target` and `opponent_hand_type_target` still remain absent in the normal sample-to-target path
- AFBS exists as a search shell, but not as a fully integrated public-belief search runtime:
  - `hydra-core/src/afbs.rs`
- Hand-EV exists, but is still heuristic rather than a full offensive oracle:
  - `hydra-core/src/hand_ev.rs`
- Endgame exists, but as weighted particle/PIMC evaluation rather than true exactification:
  - `hydra-core/src/endgame.rs`

What is outdated, wrong, or overstated in docs:

- `README.md` authority routing has been fixed, but secondary docs still need cleanup discipline to avoid reintroducing stale guidance.
- `research/design/HYDRA_SPEC.md` is explicitly outdated but still heavily referenced.
- `research/infrastructure/INFRASTRUCTURE.md` still operationalizes old PPO-era assumptions.
- `docs/GAME_ENGINE.md` and `crates/hydra-core/README.md` now correctly describe `85x34` only as the baseline prefix, not the full live encoder.

Doc drift that materially affects decisions:

- target architecture says two-tier 12/24-block + ExIt-centered training
- secondary reference docs can still imply older 40-block / PPO / dead-TRAINING assumptions if they are read without authority routing
- implementation roadmap partially reflects the new world, but supporting docs often do not

## 3. Ranked next-step recommendations

### Recommendation 1
- recommendation: Close advanced target generation and supervision loops
- evidence basis:
  - strongest support from `ANSWER_2.md`
  - external evidence favors stronger teacher targets over immediate broad search expansion
- support from answers: `ANSWER_2.md`, `ANSWER_3.md`, with `ANSWER_1.md` providing useful target semantics
- repo verification status:
  - model heads exist in `hydra-train/src/model.rs`
  - advanced loss support exists in `hydra-train/src/training/losses.rs`
  - main data path still underpopulates advanced targets in `hydra-train/src/data/sample.rs`
- expected upside: high
- difficulty: medium
- risk: medium-low
- do now / later / drop: do now

### Recommendation 2
- recommendation: Rework Hand-EV realism before deeper AFBS expansion
- evidence basis:
  - `ANSWER_1.md` and `ANSWER_2.md` both rank this as a cheaper, higher-ROI upgrade than broader search
  - external evidence says auxiliary/offensive target generation is a good medium-cost multiplier
- support from answers: `ANSWER_1.md`, `ANSWER_2.md`, `ANSWER_3.md`
- repo verification status:
  - `hydra-core/src/hand_ev.rs` exists, but expected score and win modeling remain heuristic
  - `hydra-core/src/bridge.rs` already threads Hand-EV into encoder paths
- expected upside: medium-high
- difficulty: medium
- risk: low-medium
- do now / later / drop: do soon after recommendation 1

### Recommendation 3
- recommendation: Keep AFBS specialist and hard-state gated
- evidence basis:
  - all three answers warn against broad expensive search
  - external evidence supports selective exactification more than universal belief search
- support from answers: `ANSWER_1.md`, `ANSWER_2.md`, `ANSWER_3.md`
- repo verification status:
  - `hydra-core/src/afbs.rs` is a useful shell but not yet a fully integrated runtime
  - `hydra-train/src/inference.rs` already has a fast-path vs ponder-cache split
- expected upside: medium-high
- difficulty: high
- risk: medium-high
- do now / later / drop: later, after recommendation 1

### Recommendation 4
- recommendation: Integrate robust opponent logic at search backup/runtime level only after supervision and feature realism improve
- evidence basis:
  - useful, but downstream of better targets and better local evaluators
- support from answers: `ANSWER_1.md`, `ANSWER_2.md`, `ANSWER_3.md`
- repo verification status:
  - helper math exists in `hydra-core/src/robust_opponent.rs`
  - not yet deeply wired into `hydra-core/src/afbs.rs`
- expected upside: medium
- difficulty: medium-high
- risk: medium
- do now / later / drop: later

### Recommendation 5
- recommendation: Do not make full public-belief search the immediate mainline
- evidence basis:
  - external evidence says it is real but expensive
  - repo does not yet have the closed loops needed to justify that jump
- support from answers: mostly `ANSWER_3.md`, with `ANSWER_1.md` and `ANSWER_2.md` supporting narrower/search-grade use
- repo verification status: not ready for this as the first move
- expected upside: potentially high, but too delayed
- difficulty: very high
- risk: very high
- do now / later / drop: later research branch, not now

## 4. Active Hydra vs reserve shelf vs dropped shelf

### 4.1 Active Hydra mainline

This is the strongest version of Hydra to code right now.

#### Core identity
- two-tier / ExIt-centered target direction from `HYDRA_FINAL.md`
- current repo code treated as a partially built advanced baseline, not a restart point
- supervision-first before search-expansion-first
- AFBS used selectively on hard states, not as the universal default engine

#### What stays on the critical path
1. **Advanced target-generation / supervision loop closure**
   - because the heads and losses already exist in code, but are not fully fed or activated
2. **Hand-EV realism improvements**
   - because it is already wired into the bridge/encoder path and is cheaper than deeper search work
3. **Selective endgame / AFBS improvement only after supervision is alive end-to-end**
   - because deeper search without strong targets risks expensive confusion
4. **Unified belief story**
   - Mixture-SIB for amortized belief, CT-SMC for search-grade posterior

#### What this means in plain English
Hydra should become:
- a strong learned policy/value system
- with real advanced auxiliary targets
- with better public-belief-quality features
- and with selective search layered on top only where it clearly pays

Not:
- a giant search project first
- a giant theory project first
- or a giant “invent ten new heads” project first

### 4.2 Reserve shelf: good old ideas worth preserving

These ideas should stay documented so the team can come back to them if active Hydra tops out too early.

#### Keep in reserve
- **DRDA/ACH as stronger game-theoretic optimizer direction**
  - worth preserving because it may become important once the target pipeline is healthy
- **Robust-opponent search backups / safe exploitation layers**
  - worth preserving because multiplayer Mahjong is exploitable and these may matter at the last strength mile
- **Richer latent opponent posterior / more unified opponent modeling**
  - worth preserving because the current head surface may eventually want stronger coupling
- **Deeper AFBS semantics and hard-state expansion policies**
  - worth preserving because `ANSWER_1.md` contains good structure ideas here
- **Selective exactification and stronger endgame resolvers**
  - worth preserving because late-game precision is plausible high leverage once earlier loops are stable
- **Incremental / structured belief-network ideas**
  - worth preserving as a research branch, especially if current belief machinery is too slow or too blurry

#### Why these are reserve, not active
- they are not obviously wrong
- they may matter later
- but they add enough complexity that they should not steer the next coding tranche

### 4.3 Dropped shelf: stop letting these drive current planning

These are not “forbidden forever.” They are just bad uses of mainline attention right now.

For preserved historical context and reserve-only planning surfaces, see `HYDRA_ARCHIVE.md`.

#### Drop from the active path for now
- **full public-belief search as the immediate project identity**
- **broad “search everywhere” AFBS rollout**
- **duplicated belief stacks with overlapping responsibilities**
- **adding more output heads before existing advanced heads are properly trained**
- **big optimizer-theory detours before target-generation is closed**
- **speculative novelty that has weak evidence and no clear repo insertion point**

#### Why these are dropped for now
- too compute-heavy
- too architecturally confusing
- too weakly grounded in the repo's current bottlenecks
- likely to delay actual strength gains

## 5. Conflict resolutions

### Unified belief stack vs duplicated belief machinery
Decision:
- Use Mixture-SIB as the amortized belief representation and CT-SMC as the search-grade posterior.
- Do not create a separate competing belief stack as the next move.

Why:
- `HYDRA_FINAL.md` already supports this split.
- `hydra-core/src/sinkhorn.rs` and `hydra-core/src/ct_smc.rs` already exist.
- Another parallel belief system would increase drift and calibration cost.

### Hand-EV earlier vs deeper AFBS earlier
Decision:
- Move Hand-EV realism earlier than deeper AFBS work.

Why:
- It is cheaper.
- It already has plumbing in `hydra-core/src/bridge.rs` and `hydra-core/src/encoder.rs`.
- The current Hand-EV is clearly under-realistic, so there is a high-confidence improvement path.

### AFBS broad vs AFBS specialist
Decision:
- AFBS should be specialist / hard-state gated, not broad default runtime.

Why:
- Verified code already has fast-path inference and ponder hooks.
- External evidence supports selective exactification and planning more than universal expensive search.

### DRDA/ACH on critical path vs challenger status
Decision:
- Keep DRDA/ACH as the intended target architecture direction from `HYDRA_FINAL.md`, but do not make immediate implementation decisions depend on resolving every optimizer-level debate first.

Why:
- The first coding tranche is about target-generation/supervision closure, which is more robust to this uncertainty.

### Oracle guidance alignment
Decision:
- Oracle guidance should teach, not dominate.
- Immediate repo move: keep oracle-related supervision connected to the same representation learning plan, but do not expand privileged pathways before the public-target path is closed.

Why:
- `hydra-train/src/model.rs` currently detaches the oracle critic path from the shared pooled representation.
- That is a repo-verified issue to address deliberately in later coding, not by bolting on more teacher complexity first.

### Opponent modeling as unified latent posterior vs many disconnected heads
Decision:
- Long-term direction: more unified.
- Immediate move: do not expand head count further; first feed the existing advanced heads with better targets.

Why:
- The repo already has enough surface area. The bottleneck is not a lack of outputs.

### Must-have vs speculative
Must-have now:
- reconciliation of doc authority
- advanced target-generation / supervision loop closure
- Hand-EV realism improvements

Strong multipliers later:
- AFBS integration improvements
- robust-opponent search backups
- selective endgame exactification improvements

Speculative / not worth current complexity:
- broad public-belief search as immediate mainline
- major new latent machinery before existing heads are trained properly

### Old good parts to explicitly keep available
Keep these old ideas documented, but demote them from the current coding path:
- optimizer/game-theory upgrades that depend on a healthier training loop
- advanced search backup rules and exploitation layers
- deeper belief-model experiments
- richer ToM / latent-opponent modeling ideas

These should remain as fallback or phase-next material, not be deleted from project memory.

## 6. Best next action

Best immediate next move for Hydra:

1. Reconcile the repo around one truthful planning artifact.
2. Use that artifact to pin the first coding tranche as advanced target-generation / supervision loop closure.

Why this beats a direct broad implementation tranche:
- The repo already contains a lot of the advanced surfaces.
- The current highest-leverage missing piece is that those surfaces are not consistently fed real targets and active losses.
- Fixing that gives the project learning signal now, with less architecture risk than a large AFBS rewrite.

First concrete execution tranche:

- training target generation and activation, centered on:
  - `hydra-train/src/data/sample.rs`
  - `hydra-train/src/data/mjai_loader.rs`
  - `hydra-train/src/training/losses.rs`
  - `hydra-train/src/training/bc.rs`
  - `hydra-train/src/training/rl.rs`
  - `hydra-train/src/model.rs`
- supporting bridge/search context review in:
  - `hydra-core/src/bridge.rs`
  - `hydra-core/src/ct_smc.rs`
  - `hydra-core/src/afbs.rs`

Exact tranche intent:
- populate advanced targets where feasible from existing replay/context machinery
- turn on nonzero advanced loss weights in a controlled staged way
- keep AFBS deeper integration for the following tranche, not this one

### First tranche coding spec

The goal is not “make AFBS smarter.”
The goal is:
- make existing advanced model surfaces receive real targets
- make those targets participate in training with nonzero but staged weights
- verify the full path from data -> targets -> losses -> train step -> metrics

#### Concrete coding objectives
1. **Audit and populate advanced targets in sample construction**
   - confirm where `HydraTargets` fields are still `None`
   - populate fields that can already be built from existing replay/search/belief context
2. **Stage loss activation in one place**
   - make advanced loss weights move from zero to small nonzero defaults only when their targets exist and are numerically sane
3. **Keep the rollout narrow**
   - prefer ExIt target + delta-Q + safety-residual activation first
   - bring belief-field / mixture / hand-type targets online only where labels are credible
   - if belief supervision is activated, supervise projected/public-teacher belief objects or gauge-fixed marginals, not raw Sinkhorn fields and not realized hidden allocations as direct student targets
4. **Do not expand model surface in this tranche**
   - no new heads
   - no new broad search engine
   - no new optimizer family

#### Success criteria for the first tranche
- advanced targets are produced deterministically where expected
- losses are nonzero only when targets are present
- RL/BC steps consume those targets without NaN or silent skipping
- tests cover the new target plumbing explicitly

### File-by-file implementation checklist

Use this as the concrete coding handoff for the first tranche.

#### `hydra-train/src/data/sample.rs`
- **Current state**
  - `MjaiSample` already stores `safety_residual`, `belief_fields`, and `mixture_weights` in addition to the baseline targets
  - `MjaiBatch` already collates those advanced tensors and forwards them into `HydraTargets`, while `delta_q_target` and `opponent_hand_type_target` still remain `None`
- **Required changes**
  1. audit the existing advanced carriers before adding new ones
  2. extend batch collation only for still-missing tranche targets such as a future `delta_q_target` or explicit `exit_target` carrier
  3. keep augmentation behavior correct for any tile-indexed advanced targets
- **Do not do**
  - do not invent new model heads here
  - do not mix search-only targets into baseline batches unless provenance is explicit
  - do not treat existing belief/mixture carriers as evidence that their teacher semantics are already strong enough to activate

#### `hydra-train/src/data/mjai_loader.rs`
- **Current state**
  - already builds replay-derived `safety_residual_target` plus Stage A projected `belief_fields_target` / `mixture_weight_target`
  - now has a normal replay/sample `exit_target` production path via a separate search-derived sidecar producer plus replay-time join; `delta_q_target` and `opponent_hand_type_target` still remain absent
  - `safety_residual_target` now uses signed replay-derived correction semantics: `exact_safety - public_score` on legal discard actions only
  - `hydra-train/src/training/exit.rs` now carries the narrowed ExIt teacher semantics and gates: compatible discard-only state check, child-visit-count target construction, subset mask, coverage gate, and KL / visit safety valve
- **Required changes**
  1. separate targets that are already produced but semantically weak from targets that still have no producer path at all
  2. keep `safety_residual_target` replay-derived and narrow; do not drift it into search-derived semantics
  3. completed: add a real upstream producer path for `exit_target` / `exit_mask` via a separate search-derived replay sidecar, with replay-time join and provenance/version enforcement
  4. leave clearly unavailable targets as absent rather than fabricating weak labels
  5. document provenance inline: replay-derived, bridge-derived, or search-derived
- **Preferred order**
  - first completed: truth-align and patch `safety_residual_target` to signed replay-derived residual semantics, keeping activation narrow and conservative
  - next: close `delta_q_target`
  - keep `belief_fields_target` / `mixture_weight_target` default-off until the teacher object is stronger than the current Stage A projection
  - `opponent_hand_type_target` stays blocked on ontology / mapping / builder closure

#### `hydra-train/src/training/losses.rs`
- **Current state**
  - `HydraTargets` already exposes all advanced target slots
  - advanced weights default to `0.0`
- **Required changes**
  1. add a single, clear activation policy for advanced losses
  2. ensure each optional target contributes loss only when target data exists
  3. ensure breakdowns make missing-target behavior obvious during debugging
4. keep default behavior conservative: no accidental activation without valid labels
- **Key rule**
  - target presence should control whether an advanced loss exists at all; weight alone should not hide broken plumbing
  - future belief supervision must target projected belief objects, not raw field regression

#### `hydra-train/src/training/bc.rs`
- **Current state**
  - BC already routes through `HydraTargets`
  - it now consumes replay-derived `safety_residual` targets through the supervised path and supports hardware-agnostic microbatch accumulation for full-learner BC runs
- **Required changes**
  1. completed: add tranche-specific tests showing BC consumes replay-derived `safety_residual` targets when present
  2. completed: confirm policy-agreement remains sane with optional advanced targets present and that the BC path supports staged `advanced_loss.safety_residual` activation through `src/bin/train.rs`
  3. completed: add hardware-agnostic microbatch accumulation so full-learner BC keeps the same effective batch semantics without hardcoding machine-specific GPU assumptions
  4. later: keep extending the BC/RL surface only for semantically credible targets that are still actually missing (`exit_target` replay/sample production, `delta_q`, stronger belief teachers)

#### `hydra-train/src/training/rl.rs`
- **Current state**
  - `RlBatch` can already carry `targets: HydraTargets` and `exit_target: Option<Tensor<...>>`
  - RL only gets ExIt signal if upstream code produces it
- **Required changes**
  1. make upstream production of `exit_target` part of the tranche, not a future assumption
  2. add tests for mixed cases:
     - baseline targets only
     - baseline + exit
     - baseline + selected advanced auxiliary targets
  3. verify staged exit/aux weighting remains numerically stable

#### `hydra-train/src/model.rs`
- **Current state**
  - model already exposes advanced surfaces: `belief_fields`, `mixture_weight_logits`, `opponent_hand_type`, `delta_q`, `safety_residual`
  - oracle path is still detached from pooled representation
- **Required changes in this tranche**
  1. no new heads
  2. no architectural expansion
  3. use existing output surface exactly as-is for target/loss closure work
- **Deferred explicitly**
  - oracle-path detachment review is later, after supervision plumbing is healthier

#### `hydra-core/src/bridge.rs`
- **Current state**
  - already builds search/belief feature planes from `MixtureSib`, `CtSmc`, and `AfbsTree`
  - already threads CT-SMC-weighted Hand-EV into encoder-side features
- **Required changes**
  1. identify which bridge-side signals can be promoted into training labels without inventing new semantics
  2. define a clean mapping from bridge/search features to train-side targets where credible
  3. avoid coupling replay-only loading to runtime-only search context unless there is an explicit offline generation path

#### `hydra-core/src/afbs.rs`
- **Current state**
  - search shell exists with root exit policy, visit counts, priors, Q summaries
- **Required changes in this tranche**
  1. do not broaden AFBS itself yet
  2. only expose the minimum target-generation outputs needed for ExIt / delta-Q supervision
  3. keep hard-state-gated philosophy intact

#### `hydra-core/src/ct_smc.rs`
- **Current state**
  - exact contingency-table belief sampler already exists
- **Required changes in this tranche**
  1. treat CT-SMC as a source of credible belief-weighted features/targets, not as a new search project
  2. use it where it improves label quality for belief-related supervision
  3. avoid turning this tranche into a sampler redesign

### Suggested execution order inside the first tranche
1. `losses.rs`: make target-presence/activation behavior explicit
2. `sample.rs`: audit existing advanced carriers and extend only for still-missing tranche targets
3. `mjai_loader.rs`: keep signed replay-derived `safety_residual` semantics stable and add new target producers only where provenance and semantics are credible
4. `rl.rs` and `bc.rs`: prove train-step consumption with tests
5. `bridge.rs` / `afbs.rs` / `ct_smc.rs`: only add the minimum plumbing needed for ExIt / belief-grade labels

### Minimal tranche acceptance checklist
- `MjaiSample` / batch collation can carry the tranche-selected advanced targets
- `HydraTargets` fields used in the tranche are populated by real code paths, not left as always-`None`
- at least one train path produces nonzero advanced auxiliary loss contributions in tests
- `exit_target` is produced by a real upstream path, not just by unit-test fixtures
- no new heads, no broad AFBS rewrite, no duplicated belief stack

Current tranche-status note:
- completed for the narrow supervised BC lane: replay-derived `safety_residual` now reaches the train binary with explicit staged activation controls, and full-learner BC now has hardware-agnostic microbatch accumulation rather than assuming one machine's VRAM shape
- still open before Recommendation 1 is fully complete: `delta_q` closure and stronger public-teacher belief semantics

## 7. Final handoff / progress report

What I concluded:
- Hydra should not restart from zero.
- The best immediate next move is reconciliation plus a supervision-first first coding tranche.
- The repo is closer to “advanced baseline with partially inactive loops” than to “missing everything.”

What changed:
- Added this reconciliation memo.
- Next step also updates the repo root README so it no longer routes readers into stale or missing guidance first.

What was verified:
- doc drift across `HYDRA_FINAL.md`, `HYDRA_SPEC.md`, `INFRASTRUCTURE.md`, `README.md`, and runtime docs
- code reality in `hydra-core` and `hydra-train`
- advanced heads/losses present but partially inactive
- AFBS, Hand-EV, endgame, and robust-opponent modules present but not fully integrated end-to-end

What remains next:
- implement the first coding tranche described above
- then reassess Hand-EV realism and selective AFBS improvements against that stronger supervision base
