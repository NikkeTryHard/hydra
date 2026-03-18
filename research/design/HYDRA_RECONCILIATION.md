# Hydra Reconciliation

> **Promoted operational doctrine and roadmap to Hydra v1.**
>
> This file owns Hydra's active-path sequencing, roadmap to Hydra v1, and
> active-vs-staged-vs-reserve decisions after reconciling the canonical archive
> SSOT with current repository state.
>
> If a downstream implementation or reference doc conflicts with this file on
> sequencing, promotion order, or active-vs-staged-vs-reserve status, this file
> wins.
>
> If this file drifts from
> `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl` or current
> code/runtime, refresh this file instead of treating the drift as a demotion of
> the upstream source or of runtime truth.

This file is Hydra's promoted operational doctrine.

It has one job:

- keep **Max Hydra** as the long-term destination from `HYDRA_FINAL.md`
- define **Hydra v1** as the active path to ship and train first
- make Hydra v1 the most efficient path to start training soon without
  collapsing every advanced idea into the first training promise

In plain English:

- Hydra should not restart from zero
- Hydra should not wait for every north-star mechanism before training starts
- Hydra should train first on the strongest credible baseline already supported
  by the repo
- Hydra should promote harder lanes only when they clear real evidence gates

Relationship to adjacent surfaces:

- `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl` is the epistemic
  root / canonical archive source ledger that powers downstream promoted
  doctrine.
- `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md` is the derived
  archive prioritization view over that same root.
- `research/agent_handoffs/` preserves archive evidence, provenance, and claim
  trust; it does **not** replace this file as the owner of current active-path
  status.
- `research/design/HYDRA_FINAL.md` owns Hydra's architecture north star and max
  ceiling.
- `docs/CURRENT_STATUS.md` owns the promoted already-built shipped/staged repo
  snapshot derived from this file plus code/runtime validation.
- `docs/GAME_ENGINE.md` and `docs/COMPATIBILITY_SURFACE.md` own runtime
  semantics and compatibility-sensitive invariants.

Scope:

- Canonical archive SSOT:
  `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl`
- Target architecture summary: `research/design/HYDRA_FINAL.md`
- Current shipped/staged status snapshot: `docs/CURRENT_STATUS.md`
- Verified runtime reality: current code plus runtime docs
- Operational question answered here: what Hydra should train and promote next,
  in what order, and what explicitly remains later

## 1. Roadmap thesis

Hydra has two valid horizons, and they should not be confused.

1. **Max Hydra** is the destination.
   - Hydra's north star remains the maximum-ceiling system described in
     `HYDRA_FINAL.md`: ExIt-centered training, richer belief/search machinery,
     selective search amplification, stronger opponent modeling, and later
     endgame precision.
2. **Hydra v1** is the immediate target.
   - Hydra v1 is the strongest version Hydra can train soon with credible
     labels, closed enough loops, and controlled complexity.

This roadmap chooses Hydra v1 as the active path because it is the most
efficient route to long-run strength.

That means:

- **trainable beats theoretically fuller** when the fuller system still depends
  on weak labels, open promotion questions, or broad compute-heavy integration
- **close loops before expanding architecture** because the repo already has a
  lot of advanced surface area
- **promote by evidence, not by excitement** because code existence alone is not
  proof that a lane should become default-on

Hydra v1 is not a retreat from Max Hydra. It is the shortest honest path
from today's repo state to sustained training and later ceiling-raising
promotion.

## 2. Status and roadmap vocabulary

### 2.1 Status vocabulary

These terms are shared with `docs/CURRENT_STATUS.md`.

| Term | Meaning |
|---|---|
| `active path` | current mainline direction to optimize/build now |
| `shipped baseline` | implemented and part of the current live Hydra baseline |
| `implemented but not default-on` | implemented and intentionally not the default runtime/training path |
| `implemented but staged` | implemented in core form, but activation/promotion remains intentionally deferred |
| `reserve shelf` | preserved later-work direction; not current mainline |
| `blocked` | not ready because a real dependency, semantic gap, or promotion requirement remains |
| `rejected` | not part of the current plan |
| `historical` | preserved context only; not current governing truth |

### 2.2 Roadmap vocabulary

| Term | Meaning |
|---|---|
| `Hydra v1` | the immediate, training-first active path |
| `Max Hydra` | the long-term destination owned by `HYDRA_FINAL.md` |
| `baseline first` | start training from the shipped baseline before broadening the active surface |
| `staged lane` | a capability kept off the default path until evidence supports promotion |
| `promotion gate` | an explicit pass/fail condition required before a staged lane moves upward |
| `training start condition` | the minimum baseline health required before the next training cycle should begin |
| `anti-chaos principle` | a rule that prevents Hydra from broadening too many uncertain fronts at once |

## 3. Starting baseline for the next version

Hydra v1 does not start from a blank slate. It starts from the strongest
repo surface already promoted as baseline or near-baseline truth.

### 3.1 Shipped baseline

The current shipped baseline includes:

- `hydra-core` as a real first-party runtime/encoder/simulator crate
- the live encoder/model contract at `192x34`, with the old `85x34` view treated
  as baseline-prefix only
- the fixed 46-action runtime with two-phase riichi and kan handling
- the stronger public-teacher belief-semantics tranche as part of the current
  training baseline
- the current Hand-EV realism upgrade as part of the live baseline surface
- replay-derived `safety_residual` as a narrow supervised lane
- an end-to-end ExIt carrier across the live self-play lane and the
  replay/sample sidecar-first lane

### 3.2 Implemented but not default-on

The current challenger lane is:

- the narrow DeltaQ supervision lane, which is implemented in code and remains
  promotion-gated through an arena-confirmation path

### 3.3 Implemented but staged

The current staged lanes are:

- `mixture_weight` promotion
- richer opponent-target closure
- representative-world / per-particle CT-SMC Hand-EV
- selective AFBS / endgame deepening

### 3.4 Reserve shelf

The current reserve shelf includes:

- broader public-belief search as project identity
- deeper robust-opponent search backups
- larger latent-opponent / richer auxiliary-head expansion until existing target
  closure improves

This baseline is already strong enough to justify a Hydra v1 training plan.
The roadmap should not keep talking about shipped baseline work as if it were
still hypothetical future work.

## 4. Hydra v1, the active path

### 4.1 What Hydra v1 is

Hydra v1 is the strongest trainable Hydra that can be pursued now without
turning the first version into a chaos pile of half-closed lanes.

Hydra v1 is:

- a **strong learned policy/value baseline first**
- a **supervision-first, search-second** training path
- an **ExIt-aware** training path that keeps the live carrier in scope
- a baseline that already includes the shipped belief-semantics tranche and the
  shipped Hand-EV realism tranche
- a path that keeps selective search where it clearly pays instead of making it
  the project identity too early
- a promotion-based roadmap where staged lanes move only when they clear proof

### 4.2 What Hydra v1 is not

Hydra v1 is not:

- a broad “search everywhere” AFBS project
- a freeze-until-Max-Hydra project
- a fork away from the architecture in `HYDRA_FINAL.md`
- a head-count expansion phase
- a promise that every advanced surface in the repo becomes default-on in the
  first training cycle

### 4.3 Why this is the active path now

Hydra v1 is the active path now because:

- the repo already contains a partially built advanced baseline
- the strongest near-term leverage is better training loop closure, not broader
  search identity
- the shipped belief and Hand-EV tranches already raise the baseline without
  demanding that every harder lane be promoted first
- broad search-first Hydra remains more compute-heavy, more integration-heavy,
  and less likely to accelerate the first honest training campaign

## 5. Staged lanes and promotion order

Hydra v1 should grow through a narrow promotion order.

### Lane A. Baseline training launch

This is the immediate roadmap target.

What is in:

- the shipped baseline surface from Section 3.1
- training on the live `192x34` / 46-action contract
- the shipped belief baseline
- the shipped Hand-EV realism baseline
- the replay-derived `safety_residual` lane
- the live ExIt carrier as part of the training story

What stays out of the first baseline promise:

- default-on DeltaQ
- promoted `mixture_weight`
- richer opponent-target closure
- representative-world CT-SMC Hand-EV
- selective AFBS / endgame deepening as a required launch condition

Immediate objective:

- start the next honest Hydra training cycle from the strongest already-promoted
  baseline instead of delaying for full Max-Hydra closure

### Lane B. Controlled promotion lanes

These lanes are the first candidates for measured promotion after baseline
training is healthy.

Current priority order:

1. **DeltaQ as a challenger lane**
   - implemented, measurable, and explicitly promotion-gated
   - remains non-default until its promotion evidence clears
2. **Belief-adjacent staged semantics**
   - preserve `mixture_weight` as staged until the teacher object is stronger
     than the current staged reading
3. **Richer opponent-target closure**
   - keep staged until labels and ontology are more credible

Principle:

- promotion lanes should be narrow, measurable, and one-fight-at-a-time

### Lane C. Selective search-strength lanes

These are real strength multipliers, but they are not the first training start
condition.

They include:

- representative-world / per-particle CT-SMC Hand-EV
- selective AFBS / endgame deepening
- later search-grade integration improvements that build on a healthier training
  loop

Principle:

- search should stay selective and specialist until the baseline training path
  is alive and promotion evidence says the broader cost is worth paying

### Lane D. Destination-facing Max Hydra lanes

These remain aligned with `HYDRA_FINAL.md`, but they are not Hydra v1
blockers.

They include:

- deeper robust-opponent search backups / safe exploitation layers
- broader public-belief-search identity
- richer latent-opponent / more unified opponent modeling
- deeper endgame exactification and later hard-state expansion policies
- optimizer/game-theory escalations that depend on a healthier training loop

Principle:

- preserve these lanes, but do not let them outrank a working Hydra v1
  training loop

## 6. Training start conditions

Hydra v1 is ready to begin the next training cycle when the following are
true.

### 6.1 Required to start

- the shipped baseline is the declared default training surface
- this roadmap and `docs/CURRENT_STATUS.md` agree on what is baseline versus
  staged versus reserve
- shipped belief semantics and shipped Hand-EV realism are treated as current
  baseline truth, not as future work
- ExIt remains part of the baseline training story through its live carrier
- staged lanes that are not part of the baseline remain explicitly off by
  default

### 6.2 Not required to start

The next training cycle does **not** require:

- broad public-belief search as the main runtime identity
- default-on AFBS everywhere
- default-on DeltaQ
- promoted `mixture_weight`
- representative-world / per-particle CT-SMC Hand-EV
- selective AFBS / endgame deepening
- deeper robust-opponent search backups
- richer opponent-target closure
- full Max-Hydra search stack closure

This section exists to stop Hydra from delaying training in the name of features
that are explicitly later.

## 7. Promotion gates

Implemented code is not enough to earn default-on status. Promotion follows
gates.

### 7.1 Baseline gate

Baseline work is ready when:

- the capability is already promoted as shipped baseline truth
- its semantics are honest in docs and in runtime/training surfaces
- it does not depend on still-staged lanes to justify training start

### 7.2 Challenger lane gates

An implemented-but-not-default-on or implemented-but-staged lane moves upward
only when:

- its labels or targets are semantically credible
- its activation behavior is explicit rather than accidental
- its contribution is measurable in training/eval rather than inferred from
  theory alone
- promoting it does not blur the distinction between baseline and experiment

Explicit example:

- DeltaQ remains implemented but not default-on because its promotion is tied to
  an arena-confirmation path rather than mere structural existence

### 7.3 Search-strength gates

A search-strength lane moves upward only when:

- the baseline training loop is already alive
- the lane has a clear insertion point and a narrow scope
- the lane improves real strength-per-complexity instead of reopening project
  identity debates

### 7.4 Max-Hydra-only gates

Destination-facing lanes should only become active-path work when:

- Hydra v1 has already proved too weak or too capped
- the simpler promotion lanes have been fairly tested first
- the extra complexity is justified by evidence instead of north-star gravity

## 8. Anti-chaos principles

These principles are mandatory for Hydra v1.

1. **Baseline before breadth**
   - do not broaden multiple uncertain lanes before the baseline training path is
     live
2. **One promotion fight at a time**
   - do not try to promote several staged lanes at once
3. **No architecture identity flip midstream**
   - the next version is training-first, not search-first by surprise later
4. **Shipped means baseline, staged means staged**
   - do not keep talking about shipped baseline work as if it were still future
     work
5. **North star is destination, not checklist**
   - `HYDRA_FINAL.md` remains the target architecture, but it does not force all
     destination-facing machinery into the first training promise
6. **Preserve reserve ideas without letting them steer**
   - reserve shelf exists to keep good ideas alive, not to dominate current
     sequencing

## 9. Destination-facing reserve shelf

These lanes remain consistent with Max Hydra and should stay documented.

### 9.1 Preserve for later

- deeper robust-opponent search backups / safe exploitation layers
- broader public-belief-search identity
- richer latent-opponent / more unified opponent modeling
- deeper AFBS semantics and hard-state expansion policies
- selective exactification and stronger endgame resolvers
- deeper belief-network experiments
- optimizer/game-theory escalations that depend on a healthier training loop

### 9.2 Not active for the next version

These are not rejected forever. They are simply not allowed to steer Hydra v1.

- broad “search everywhere” AFBS rollout
- full public-belief search as immediate project identity
- adding more heads before existing advanced surfaces are properly promoted
- large optimizer-theory detours ahead of the first honest training campaign
- speculative novelty that lacks a strong repo insertion point

## 10. Hydra v1 roadmap summary

Hydra's roadmap to v1 is straightforward.

### Immediate objective

Start training soon on Hydra v1: the strongest credible baseline already
supported by promoted doctrine and current shipped surfaces.

### First version scope

Hydra v1 means:

- fixed live runtime/encoder compatibility surface
- shipped belief baseline
- shipped Hand-EV realism baseline
- narrow replay-derived `safety_residual`
- live ExIt carrier
- staged lanes kept staged unless they clear promotion gates

### First promotions after launch

After baseline training is healthy, Hydra should evaluate narrow challenger lanes
in order, starting with DeltaQ and only then considering later staged belief,
opponent-target, and search-strength promotions.

### Long-term destination

Max Hydra from `HYDRA_FINAL.md` remains the long-term destination. Hydra v1
exists to reach that destination efficiently, not to replace it.

### Final doctrine sentence

Hydra should begin with the strongest trainable baseline it can honestly defend,
then grow toward its full ceiling through narrow, evidence-gated promotion. That
is the active path most likely to produce a strong Hydra over time.
