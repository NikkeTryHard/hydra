# Hydra Reconciliation

> **Promoted operational doctrine and roadmap to Hydra v1.**
>
> This file owns Hydra active-path sequencing, Hydra v1 roadmap, active-vs-staged-vs-reserve calls after reconciling canonical archive SSOT with current repo state.
>
> If downstream impl or ref doc conflicts with this file on sequencing, promotion order, or active-vs-staged-vs-reserve status, this file wins.
>
> If this file drifts from
> `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl` or current
> code/runtime, refresh this file; do not treat drift as demotion of upstream source or runtime truth.

This file = Hydra promoted operational doctrine.

One job:

- keep **Max Hydra** = long-term destination from `HYDRA_FINAL.md`
- define **Hydra v1** = active path to ship/train first
- make Hydra v1 fastest honest path to start training soon without forcing every advanced idea into first training promise

Plain English:

- Hydra should not restart from zero
- Hydra should not wait for every north-star mechanism before training starts
- Hydra should train first on strongest credible baseline repo already supports
- Hydra should promote harder lanes only after real evidence gates clear

Relationship to adjacent surfaces:

- `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl` = epistemic root / canonical archive source ledger powering downstream promoted doctrine
- `research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS_ROADMAP.md` = derived archive prioritization view over same root
- `research/agent_handoffs/` preserves archive evidence, provenance, claim trust; does **not** replace this file as owner of current active-path status
- `research/design/HYDRA_FINAL.md` owns Hydra architecture north star and max ceiling
- `docs/CURRENT_STATUS.md` owns promoted already-built shipped/staged repo snapshot derived from this file plus code/runtime validation
- `docs/GAME_ENGINE.md` and `docs/COMPATIBILITY_SURFACE.md` own runtime semantics and compatibility-sensitive invariants

Scope:

- Canonical archive SSOT:
`research/agent_handoffs/ARCHIVE_CANONICAL_CLAIMS.jsonl`
- Target architecture summary: `research/design/HYDRA_FINAL.md`
- Current shipped/staged status snapshot: `docs/CURRENT_STATUS.md`
- Verified runtime reality: current code plus runtime docs
- Operational question answered here: what Hydra should train/promote next, in what order, what stays later

## 1. Roadmap thesis

Hydra has two valid horizons. Do not confuse them.

1. **Max Hydra** = destination.
   - Hydra north star stays maximum-ceiling system from
`HYDRA_FINAL.md`: ExIt-centered training, richer belief/search machinery,
selective search amplification, stronger opponent modeling, later
endgame precision.
2. **Hydra v1** = immediate target.
   - Hydra v1 = strongest Hydra trainable soon with credible labels,
closed-enough loops, controlled complexity.

This roadmap picks Hydra v1 as active path because it is fastest route to long-run strength.

That means:

- **trainable beats theoretically fuller** when fuller system still depends on weak labels, open promotion questions, or broad compute-heavy integration
- **close loops before expanding architecture** because repo already has much advanced surface area
- **promote by evidence, not excitement** because code existence alone does not prove lane should become default-on

Hydra v1 not retreat from Max Hydra. It is shortest honest path from current repo state to sustained training, then later ceiling-raising promotion.

## 2. Status and roadmap vocabulary

### 2.1 Status vocabulary

These terms are shared with `docs/CURRENT_STATUS.md`.

| Term | Meaning |
|---|---|
| `active path` | current mainline direction to optimize/build now |
| `shipped baseline` | implemented and part of current live Hydra baseline |
| `implemented but not default-on` | implemented and intentionally not default runtime/training path |
| `implemented but staged` | implemented in core form, but activation/promotion intentionally deferred |
| `reserve shelf` | preserved later-work direction; not current mainline |
| `blocked` | not ready because real dependency, semantic gap, or promotion requirement remains |
| `rejected` | not part of current plan |
| `historical` | preserved context only; not current governing truth |

### 2.2 Roadmap vocabulary

| Term | Meaning |
|---|---|
| `Hydra v1` | immediate, training-first active path |
| `Max Hydra` | long-term destination owned by `HYDRA_FINAL.md` |
| `baseline first` | start training from shipped baseline before broadening active surface |
| `staged lane` | capability kept off default path until evidence supports promotion |
| `promotion gate` | explicit pass/fail condition before staged lane moves up |
| `training start condition` | minimum baseline health before next training cycle starts |
| `anti-chaos principle` | rule preventing Hydra from broadening too many uncertain fronts at once |

## 3. Starting baseline for the next version

Hydra v1 does not start blank. It starts from strongest repo surface already promoted as baseline or near-baseline truth.

### 3.1 Shipped baseline

Current shipped baseline includes:

- `hydra-core` as real first-party runtime/encoder/simulator crate
- live encoder/model contract at `192x34`, with old `85x34` view treated as baseline-prefix only
- fixed 46-action runtime with two-phase riichi and kan handling
- stronger public-teacher belief-semantics tranche as part of current training baseline
- current Hand-EV realism upgrade as part of live baseline surface
- replay-derived `safety_residual` as narrow supervised lane
- end-to-end ExIt carrier across live self-play lane and replay/sample sidecar-first lane

### 3.2 Implemented but not default-on

Current challenger lane:

- narrow DeltaQ supervision lane, implemented in code and still promotion-gated through arena-confirmation path

### 3.3 Implemented but staged

Current staged lanes:

- `mixture_weight` promotion
- richer opponent-target closure
- representative-world / per-particle CT-SMC Hand-EV
- selective AFBS / endgame deepening

### 3.4 Reserve shelf

Current reserve shelf includes:

- broader public-belief search as project identity
- deeper robust-opponent search backups
- larger latent-opponent / richer auxiliary-head expansion until existing target closure improves

This baseline already strong enough to justify Hydra v1 training plan. Roadmap should not keep describing shipped baseline work as hypothetical future work.

## 4. Hydra v1, the active path

### 4.1 What Hydra v1 is

Hydra v1 = strongest trainable Hydra pursuable now without turning first version into chaos pile of half-closed lanes.

Hydra v1 is:

- **strong learned policy/value baseline first**
- **supervision-first, search-second** training path
- **ExIt-aware** training path keeping live carrier in scope
- baseline already including shipped belief-semantics tranche and shipped Hand-EV realism tranche
- path keeping selective search where it clearly pays instead of making it project identity too early
- promotion-based roadmap where staged lanes move only after proof clears

### 4.2 What Hydra v1 is not

Hydra v1 is not:

- broad “search everywhere” AFBS project
- freeze-until-Max-Hydra project
- fork away from architecture in `HYDRA_FINAL.md`
- head-count expansion phase
- promise that every advanced repo surface becomes default-on in first training cycle

### 4.3 Why this is the active path now

Hydra v1 active now because:

- repo already contains partially built advanced baseline
- strongest near-term leverage = better training loop closure, not broader search identity
- shipped belief and Hand-EV tranches already raise baseline without requiring every harder lane first
- broad search-first Hydra remains more compute-heavy, more integration-heavy, less likely to accelerate first honest training campaign

## 5. Staged lanes and promotion order

Hydra v1 should grow through narrow promotion order.

### Lane A. Baseline training launch

Immediate roadmap target.

What is in:

- shipped baseline surface from Section 3.1
- training on live `192x34` / 46-action contract
- shipped belief baseline
- shipped Hand-EV realism baseline
- replay-derived `safety_residual` lane
- live ExIt carrier as part of training story

What stays out of first baseline promise:

- default-on DeltaQ
- promoted `mixture_weight`
- richer opponent-target closure
- representative-world CT-SMC Hand-EV
- selective AFBS / endgame deepening as required launch condition

Immediate objective:

- start next honest Hydra training cycle from strongest already-promoted baseline instead of delaying for full Max-Hydra closure

### Lane B. Controlled promotion lanes

These lanes = first candidates for measured promotion after baseline training is healthy.

Current priority order:

1. **DeltaQ as challenger lane**
   - implemented, measurable, explicitly promotion-gated
   - stays non-default until promotion evidence clears
2. **Belief-adjacent staged semantics**
   - preserve `mixture_weight` as staged until teacher object is stronger than current staged reading
3. **Richer opponent-target closure**
   - keep staged until labels and ontology are more credible

Principle:

- promotion lanes should be narrow, measurable, one-fight-at-a-time

### Lane C. Selective search-strength lanes

These are real strength multipliers, but not first training start condition.

They include:

- representative-world / per-particle CT-SMC Hand-EV
- selective AFBS / endgame deepening
- later search-grade integration improvements building on healthier training loop

Principle:

- search should stay selective and specialist until baseline training path is alive and promotion evidence says broader cost is worth paying

### Lane D. Destination-facing Max Hydra lanes

These remain aligned with `HYDRA_FINAL.md`, but are not Hydra v1 blockers.

They include:

- deeper robust-opponent search backups / safe exploitation layers
- broader public-belief-search identity
- richer latent-opponent / more unified opponent modeling
- deeper endgame exactification and later hard-state expansion policies
- optimizer/game-theory escalations depending on healthier training loop

Principle:

- preserve these lanes, but do not let them outrank working Hydra v1 training loop

## 6. Training start conditions

Hydra v1 ready to begin next training cycle when following are true.

### 6.1 Required to start

- shipped baseline is declared default training surface
- this roadmap and `docs/CURRENT_STATUS.md` agree on what is baseline versus staged versus reserve
- shipped belief semantics and shipped Hand-EV realism are treated as current baseline truth, not future work
- ExIt remains part of baseline training story through live carrier
- staged lanes not part of baseline remain explicitly off by default

### 6.2 Not required to start

Next training cycle does **not** require:

- broad public-belief search as main runtime identity
- default-on AFBS everywhere
- default-on DeltaQ
- promoted `mixture_weight`
- representative-world / per-particle CT-SMC Hand-EV
- selective AFBS / endgame deepening
- deeper robust-opponent search backups
- richer opponent-target closure
- full Max-Hydra search stack closure

This section exists to stop Hydra from delaying training for features explicitly marked later.

## 7. Promotion gates

Implemented code not enough for default-on status. Promotion follows gates.

### 7.1 Baseline gate

Baseline work ready when:

- capability already promoted as shipped baseline truth
- semantics are honest in docs and runtime/training surfaces
- it does not depend on still-staged lanes to justify training start

### 7.2 Challenger lane gates

Implemented-but-not-default-on or implemented-but-staged lane moves upward only when:

- labels or targets are semantically credible
- activation behavior is explicit, not accidental
- contribution is measurable in training/eval, not inferred from theory alone
- promoting it does not blur baseline vs experiment distinction

Explicit example:

- DeltaQ remains implemented but not default-on because promotion ties to arena-confirmation path, not mere structural existence

### 7.3 Search-strength gates

Search-strength lane moves upward only when:

- baseline training loop already alive
- lane has clear insertion point and narrow scope
- lane improves real strength-per-complexity instead of reopening project identity debates

### 7.4 Max-Hydra-only gates

Destination-facing lanes should become active-path work only when:

- Hydra v1 already proved too weak or too capped
- simpler promotion lanes were fairly tested first
- extra complexity is justified by evidence, not north-star gravity

## 8. Anti-chaos principles

These principles are mandatory for Hydra v1.

1. **Baseline before breadth**
   - do not broaden multiple uncertain lanes before baseline training path is live
2. **One promotion fight at time**
   - do not try to promote several staged lanes at once
3. **No architecture identity flip midstream**
   - next version is training-first, not search-first by surprise later
4. **Shipped means baseline, staged means staged**
   - do not keep describing shipped baseline work as future work
5. **North star is destination, not checklist**
   - `HYDRA_FINAL.md` remains target architecture, but does not force all destination-facing machinery into first training promise
6. **Preserve reserve ideas without letting them steer**
   - reserve shelf exists to keep good ideas alive, not dominate current sequencing

## 9. Destination-facing reserve shelf

These lanes remain consistent with Max Hydra and should stay documented.

### 9.1 Preserve for later

- deeper robust-opponent search backups / safe exploitation layers
- broader public-belief-search identity
- richer latent-opponent / more unified opponent modeling
- deeper AFBS semantics and hard-state expansion policies
- selective exactification and stronger endgame resolvers
- deeper belief-network experiments
- optimizer/game-theory escalations depending on healthier training loop

### 9.2 Not active for the next version

These are not rejected forever. They must not steer Hydra v1.

- broad “search everywhere” AFBS rollout
- full public-belief search as immediate project identity
- adding more heads before existing advanced surfaces are properly promoted
- large optimizer-theory detours ahead of first honest training campaign
- speculative novelty lacking strong repo insertion point

## 10. Hydra v1 roadmap summary

Hydra roadmap to v1 is straightforward.

### Immediate objective

Start training soon on Hydra v1: strongest credible baseline already supported by promoted doctrine and current shipped surfaces.

### First version scope

Hydra v1 means:

- fixed live runtime/encoder compatibility surface
- shipped belief baseline
- shipped Hand-EV realism baseline
- narrow replay-derived `safety_residual`
- live ExIt carrier
- staged lanes kept staged unless they clear promotion gates

### First promotions after launch

After baseline training is healthy, Hydra should evaluate narrow challenger lanes in order, starting with DeltaQ and only then considering later staged belief, opponent-target, and search-strength promotions.

### Long-term destination

Max Hydra from `HYDRA_FINAL.md` remains long-term destination. Hydra v1 exists to reach that destination efficiently, not replace it.

### Final doctrine sentence

Hydra should begin with strongest trainable baseline it can honestly defend, then grow toward full ceiling through narrow, evidence-gated promotion. That is active path most likely to produce strong Hydra over time.