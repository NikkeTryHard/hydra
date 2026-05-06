# Hydra hidden-world pass-two context packet — handcrafted verbose artifact file

Not thin instruction sheet. Goal = give genie **large, explicit context packet** for pass two. Job no longer ask whether hidden-world lane matters. Job now: decide exact winning design stack, first build order, what kill, where Hydra must accept being wrong.

Pass one already did broad diagnosis and target-object narrowing. This file therefore biases toward:

- rich paper context,
- exact mechanisms,
- strong transfer arguments,
- assumptions and breakpoints,
- candidate algorithm families,
- training and evaluation implications,
- explicit reminder: current Hydra support not outer boundary.

If strongest design exceeds current Hydra support, fine. That is point.

## What pass two is actually for

Pass two not “what hidden-world object exists?”

Pass two is:

- what full design stack should win,
- what shortest honest tranche preserves winner semantics,
- what teacher hierarchy and training recipe should exist,
- what evaluation gates and kill criteria should govern promotion,
- what current Hydra assumptions should be overturned if they block stronger design,
- what evidence would prove chosen path is fake strength, not real strength.

So file built like research notebook / artifact packet, not short prompt.

## Keep these high-level answer obligations in mind

Answer from this packet should still cover these top-level outputs:

1. Executive Verdict
2. Best Full Design
3. Shortest Honest Tranche
4. Algorithm Family Decision Table
5. Teacher Hierarchy
6. Training Recipe
7. Evaluation Gates
8. Kill Criteria
9. Where Hydra Is Wrong
10. What Stays Open
11. Minimal Experiment Matrix

Do not waste budget re-listing rules. Use context below.

---

# Context block A — pass-two design pressure

Pass-two winner must be judged on actual Hydra strength, not elegance alone.

Winning design should be evaluated against at least these tensions:

1. **Semantic honesty vs cheap approximations**
   - preserve real hidden-world object, or silently collapse into weaker surrogate?

2. **Teacher quality vs deployable legality**
   - can strongest teacher use privileged signals while student remains public-information legal?

3. **Search-grade fidelity vs amortized speed**
   - maintain real search-grade hidden-world object then export distilled structure, or over-amortize too early?

4. **Calibration vs mere sharpness**
   - are confidence outputs trustworthy, or only decisive-looking?

5. **Offline teacher strength vs online improvement stability**
   - does training recipe explain how strong teacher objects become stronger policies, not brittle offline artifacts?

6. **Strength evidence vs internal metric theater**
   - does improved hidden-world modeling cash out into better actions and arena outcomes?

7. **Hydra compatibility vs Hydra correctness**
   - if current carrier, gating support, or sequencing blocks strongest design, change those; do not treat them as sacred.

---

# Context block B — local design surfaces that matter for pass two

These local surfaces matter more in pass two because they shape what can be trained, gated, promoted, or rejected.

## B1. CT-SMC as a real local teacher/search surface

Hydra already has concrete CT-SMC surface with:

- particle count,
- ESS threshold,
- explicit `update(...)`,
- weighted per-tile means,
- unweighted `mean_allocation()`,
- systematic resampling.

This matters because pass two should not treat CT-SMC as doctrine dream only. It should decide whether CT-SMC is:

- merely existence proof,
- diagnostics-only object,
- actual search-grade teacher,
- bridge object to something stronger,
- lane that loses to better factorized or distillation-heavy design.

Main pass-two question here not “does CT-SMC exist?” but:

**Is current CT-SMC surface strong enough to anchor winning teacher stack, or should it be subordinated to stronger hidden-world family?**

## B2. Train-bin support blocks are real design pressure

Current train entrypoints explicitly reject several advanced losses in safe BC/RL entrypaths. That matters because if strongest design requires belief or mixture-side objectives that train-bin blocks, genie should say one of two things clearly:

- either strongest honest tranche must change those contracts,
- or stronger design is real but Hydra too weak to train it honestly.

Wrong answer = weaken design to fit current train-bin support.

## B3. Head activation discipline already exists

Hydra already has serious gate stack around advanced heads:

- density gates,
- sparse-search samples-per-parameter logic,
- warmup states,
- negative-transfer / gradient-conflict checks,
- per-head state transitions.

Pass two should not reinvent promotion logic from nothing. It should decide whether this gate stack is:

- already correct template for hidden-world promotion,
- missing crucial calibration and action-usefulness gates,
- structurally wrong for winning design.

## B4. DeltaQ promotion is an existence proof for “real gates”

Hydra already has more serious promotion machinery than many repos:

- comparison reports,
- policy-transfer reports,
- arena reports,
- explicit threshold structs,
- rejection / require-arena / strong-promotion logic.

Pass two should ask:

**what would hidden-world lane need, beyond this, to earn promotion honestly?**

## B5. Validation and paired arena surfaces already exist

Hydra’s validation and evaluation code already carries:

- policy agreement metrics,
- DeltaQ promotion metric collection,
- paired arena evaluation,
- non-regression vs strong-promotion decisions,
- benchmark gates like CT-SMC latency and self-play throughput,
- stable-dan / mean-placement style evaluation.

So pass two should output belief-lane evaluation gates that can plug into real validation culture, not detached benchmark religion.

## B6. ExIt validation harness is a template for teacher validation

Hydra already has clean producer-validation harness pattern:

- emission rate,
- coverage,
- supported actions,
- root visits,
- KL,
- top-1 agreement,
- explicit rejection counts,
- threshold objects.

Hidden-world lane should almost certainly reuse this pattern for teacher validation, even if exact metrics differ.

## B7. Testing and falsification should be first-class

Current testing strategy reinforces: do not trust coverage alone, use goldens and cross-validation, treat silently wrong labels as catastrophic. For pass two, hidden-world lane should probably define:

- known-state golden beliefs,
- teacher/student roundtrip or replay validations,
- cross-validation against independent hidden-world samplers or baselines,
- explicit regression tests around gating and calibration behavior.

---

# Context block C — core external paper bank for pass two

This section should add real value. Goal not list papers. Goal = give genie enough dense external context to design winning pass-two stack, not merely wave at precedent.

## C1. Guo et al. (2017) — On Calibration of Modern Neural Networks

### Core idea

Guo et al. show modern deep nets can become **more accurate and less trustworthy at same time**. Core distinction = classification accuracy vs probability quality. Model can predict right class more often yet become dangerously overconfident. For hidden-world lane, this matters because downstream use of belief outputs is not only ranking; it is risk gating, search trust, and deployment-time confidence use.

Calibration condition simple: among predictions made with confidence `p`, empirical correctness rate should also be `p`. Modern nets violate this badly enough that paper treats calibration repair as its own object, not side effect of better classification.

### Exact mechanisms worth stealing

- Expected Calibration Error:
`ECE = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)|`
- Reliability diagrams as visual evidence, not only scalar evidence.
- Temperature scaling as rank-preserving repair of logit sharpness.
- NLL-vs-accuracy disconnect as warning sign that confidence quality is decaying while top-1 still improves.

### Why this matters for pass two

Pass two must design **evaluation gates**, not merely say “calibration matters.” Guo gives first layer, but not enough alone. Pass-two answer should probably say something like:

- calibration gates must exist,
- naive scalar ECE alone is insufficient,
- confidence should gate search or trust only if structurally and probabilistically honest.

### Candidate family this supports

- calibrated structural posterior
- confidence-gated search / trust systems
- temperature-corrected trust heads

### Where it breaks

- not structured combinatorial hidden-world paper,
- does not solve target semantics,
- calibration over marginals does not imply calibration over whole hidden worlds.

### Keep these terms alive

- Expected Calibration Error
- reliability diagram
- temperature scaling
- negative log-likelihood
- overconfidence gap
- post-hoc calibration

---

## C2. Posocco & Bonnefoy (2021) — Estimating Expected Calibration Errors

### Core idea

This paper matters because pass two should not inherit ECE lazily. Central point: even calibration **evaluation itself** is estimator-sensitive, and naive ECE habits can mislead. Exactly kind of thing pass-two genie should know if designing promotion gates instead of checklist theater.

Paper argues calibration assessment deserves deeper analysis and not all ECE estimators work equally across settings. That matters for Hydra because hidden-world lane may look “well calibrated” under loose estimator and fail under stronger evaluation.

### Exact mechanisms worth stealing

- estimator-aware calibration evaluation,
- deeper analysis of binning and estimator bias,
- scenario-dependent choice of ECE estimator rather than one universal default.

### Why this matters for pass two

If winning design relies on belief confidence, search trust, abstention-to-search, or risk gating, evaluation gates should probably reference stronger calibration-estimation practice rather than single naive ECE number.

### Candidate family this supports

- calibration gate design
- confidence-based search deferral or trust gating
- stronger kill criteria for fake-calibrated belief heads

### Keep these terms alive

- ECE estimator quality
- calibration evaluation procedure
- bin sensitivity
- calibration metric bias

---

## C3. Montemerlo et al. (2002) — FastSLAM

### Core idea

FastSLAM’s gift to Hydra not robotics. Gift = giant horrible posterior may become tractable if factorized around right latent backbone. Condition on correct path/backbone, remaining pieces become cheaper conditional objects. Massive signal for pass two: right answer to combinatorial pain may be **structural factorization**, not surrender.

### Exact mechanisms worth stealing

- factored posterior decomposition,
- Rao-Blackwellized particle filters,
- per-particle structured state,
- tree/path memory sharing,
- per-particle data association.

### Why this matters for pass two

Pass one already used FastSLAM as inspiration. Pass two should do more: decide whether Rao-Blackwellized tile-world family is winner, reserve family, or overcomplicated trap.

Paper gives strong case that if Hydra can find right backbone, true hidden-world family may still be feasible. But warning too: if backbone is wrong, whole factorization story collapses.

### Candidate family this supports

- Rao-Blackwellized tile-world particle filtering
- backbone-plus-conditional-allocation teacher families
- multi-hypothesis hidden-world reasoning with shared structure

### Keep these terms alive

- Rao-Blackwellization
- factored posterior
- conditional independence given path
- per-particle structured state
- data association
- shared-tree memory reuse

---

## C4. Arulampalam et al. (2002) — Particle Filters tutorial

### Core idea

This is anti-hand-waving paper. If Hydra wants any sample-based hidden-world family, this tutorial says you do not get to say sample worlds” and stop there. Need proposal quality, weight updates, ESS, resampling strategy, plan for degeneracy and sample impoverishment.

### Exact mechanisms worth stealing

- SIS weight update:
`w_k^i ∝ w_(k-1)^i * p(z_k | x_k^i) * p(x_k^i | x_(k-1)^i) / q(x_k^i | x_(k-1)^i, z_k)`
- `N_eff = 1 / sum_i (w_k^i)^2`
- optimal importance density idea,
- systematic resampling,
- explicit degeneracy monitoring.

### Why this matters for pass two

If genie chooses particle-like winner, it must also specify:

- how particles are proposed,
- what public-event likelihood is,
- how ESS triggers work,
- what resampling discipline prevents collapse.

If it cannot, particle family should probably lose.

### Candidate family this supports

- event-conditioned CT-SMC
- particle-based replay teacher generation
- Rao-Blackwellized hidden-world families

### Keep these terms alive

- Sequential Importance Sampling
- effective sample size
- particle degeneracy
- sample impoverishment
- proposal density
- systematic resampling

---

## C5. Grudzien & Bocquet (2023) — Bayesian Data Assimilation

### Core idea

Data assimilation brings forecast-assimilate-smooth worldview. Strong fit for pass two because one sharp question is whether Hydra’s teacher should be pure filter, short-window smoother, or hierarchy where smoothing is diagnostics-only. This paper says right framing is not one-shot inference but sequence:

- forecast,
- absorb evidence,
- revise,
- smooth,
- maybe revise model.

### Exact mechanisms worth stealing

- forecast-analysis cycle,
- filtering vs smoothing distinction,
- fixed-lag smoothing,
- variational trajectory correction,
- posterior repair under later evidence.

### Why this matters for pass two

This paper should pressure genie to answer:

- is teacher generation filter-only,
- filter plus short smoothing,
- or filter for student, smoother for diagnostics?

It also sharpens difference between legal student target and stronger replay-side teacher object.

### Candidate family this supports

- fixed-lag public smoother
- replay-side teacher repair
- trajectory-window teacher generation

### Keep these terms alive

- forecast-analysis cycle
- filtering vs smoothing
- data assimilation window
- covariance inflation
- variational update
- retrospective correction

---

## C6. Mimori et al. (2021) — Diagnostic Uncertainty Calibration

### Core idea

Paper treats situations where “label itself” is not single clean object, because experts can disagree. Valuable here because public Mahjong states often support multiple plausible hidden worlds. Paper says uncertainty is not one low-confidence point estimate; sometimes uncertainty exists **over probability object itself**.

### Exact mechanisms worth stealing

- label histograms,
- disagreement-aware calibration,
- Dirichlet-style uncertainty over class-probability estimates,
- posterior updates of uncertainty objects.

### Why this matters for pass two

If genie wants student or search stack to carry not only belief object but also “confidence in that object,” this paper is better source than Guo. Especially relevant for decisions about:

- trust gating,
- abstention to search,
- uncertainty-driven deferral,
- diagnostics proving belief object is honest about ambiguity.

### Candidate family this supports

- concentration/confidence heads over structured posterior
- disagreement-aware teacher diagnostics
- ambiguity-aware trust gating

### Keep these terms alive

- label histogram
- disagreement-aware calibration
- Dirichlet concentration
- posterior uncertainty object
- epistemic uncertainty

---

## C7. Evensen (2003) — Ensemble Kalman Filter

### Core idea

EnKF matters because it gives mature alternative for giant hidden-state tracking: maintain ensemble, update by innovation, keep uncertainty explicit, use smoothing if needed. Likely not literal winner for Mahjong tile worlds, but strong pressure against pretending only options are “flat amortized head” or “full particle monster.”

### Exact mechanisms worth stealing

- ensemble representation,
- innovation-based update,
- state augmentation,
- ensemble smoother,
- perturbed observations.

### Why this matters for pass two

If genie wants compressed uncertainty-over-summary tracker or opponent-latent tracker instead of full explicit tile-allocation posterior everywhere, EnKF-style thinking is useful.

### Candidate family this supports

- projected ensemble tracker
- latent uncertainty tracker over opponent style or world summaries

### Keep these terms alive

- innovation
- ensemble smoother
- state augmentation
- inbreeding
- perturbed observations

---

## C8. Karl et al. (2017) — Deep Variational Bayes Filters

### Core idea

DVBF says hidden representation should be judged by whether it supports correct dynamics, not only whether it reconstructs current snapshot. Crucial if genie wants compact amortized hidden-world state rather than fully explicit allocation object.

### Exact mechanisms worth stealing

- transition-aware latent state,
- reparameterized stochastic transitions,
- dynamics-consistent hidden representation,
- training pressure from future evidence.

### Why this matters for pass two

This paper pushes genie to answer whether compact latent world model is:

- actual winner,
- only reserve fast path,
- or too opaque / too hard to calibrate to beat explicit structured teachers.

### Candidate family this supports

- latent roll-forward world model
- compact belief-state fast path with explicit teacher calibration

### Keep these terms alive

- dynamics-consistent latent state
- latent state-space model
- reparameterized transition
- reconstruction vs dynamics tradeoff

---

## C9. Friston et al. (2017) — Active Inference

### Core idea

This paper matters less as literal algorithm spec and more as language for information-sensitive action choice. If genie wants to justify some actions as good because they reduce hidden-world ambiguity, active inference gives principled way to talk about epistemic value instead of hand-wavy “exploration.”

### Exact mechanisms worth stealing

- expected free energy,
- epistemic value vs pragmatic value,
- precision control,
- smoothing / posterior repair.

### Why this matters for pass two

Especially relevant if winning stack includes:

- trust-gated search,
- confidence-sensitive action choice,
- explicit “information value” term in planner.

But this paper should probably remain **reserve / fusion** artifact, not core winner, unless genie can cash it out into something testable.

### Candidate family this supports

- epistemic-assist heuristic
- confidence-weighted search or planner precision control

### Keep these terms alive

- expected free energy
- epistemic value
- pragmatic value
- precision
- variational free energy

---

## C10. Brown & Sandholm (2019) — Pluribus

### Core idea

Pluribus is best pass-two reality check that multiplayer ugly theory does not kill practical strength. Blueprint plus local search plus continuation strategies was enough for six-player superhuman success.

### Exact mechanisms worth stealing

- blueprint strategy,
- depth-limited search,
- continuation strategies,
- discounted early-iteration regret via Linear CFR,
- variance-reduced evaluation like AIVAT.

### Why this matters for pass two

If genie wants belief-conditioned blueprint-plus-search stack, Pluribus is strongest direct precedent for saying:

- yes, multiplayer theory is ugly,
- but robust empirical strength can still come from right hybrid system.

### Candidate family this supports

- oracle-guided blueprint plus selective belief search
- continuation-aware leaf evaluation in search

### Keep these terms alive

- blueprint strategy
- depth-limited search
- continuation strategies
- linear CFR
- empirical robustness

---

## C11. Li et al. (2020) — Suphx

### Core idea

Suphx is same-game anchor. Reminder: strong Mahjong systems do not only need nice hidden-state object; they need stack that turns that object into better action quality under long-horizon, high-variance, multiplayer conditions. Suphx’s signature trio — global reward prediction, oracle guiding, and runtime policy adaptation — should stay in pass-two context because they speak directly to training and action quality, not only semantics.

### Exact mechanisms worth stealing

- global reward prediction across round prefixes,
- oracle guiding with feature dropout,
- pMCPA/runtime adaptation,
- look-ahead features,
- distributed self-play / training architecture.

### Why this matters for pass two

This is one of strongest pass-two anchors because winning hidden-world design must explain:

- how teacher-side privileged information becomes legal student,
- how long-horizon reward repair interacts with hidden-world lane,
- whether runtime adaptation should exist beyond static policy inference.

### Candidate family this supports

- oracle-guided blueprint plus selective belief search,
- privileged teacher + public student training stacks,
- reward-smoothed hidden-world policy training

### Keep these terms alive

- oracle guiding
- global reward prediction
- parametric Monte-Carlo policy adaptation
- deal-in rate
- stable rank
- same-game precedent

---

## C12. Privileged Information Distillation for Language Models (2026)

### Core idea

This paper is extremely relevant for pass two because it sharpens privileged-teacher problem beyond Suphx alone. Main point: training-time privileged information can enable success on hard tasks, but real difficulty is transferring that advantage to student that must act **without** privileged information at inference time. Paper’s answer is joint teacher-student objective where privileged-information-conditioned teacher and unconditioned student are trained together, not isolated stages.

### Exact mechanisms worth stealing

- joint `pi-Distill` objective,
- explicit teacher objective and student objective,
- KL-coupled distillation,
- OPSD / on-policy self-distillation,
- privileged-information utility analysis,
- alpha scheduling between teacher and student pressure.

### Why this matters for pass two

This should strongly influence hidden-world lane if genie wants:

- privileged search-grade or oracle-grade teacher,
- public-information-legal student,
- concrete recipe for transfer instead of hand-wave.

This paper gives stronger pass-two pressure on exact question:

**if best teacher is stronger because it uses hidden or privileged information, how does student inherit benefit without inheriting illegality?**

### Candidate family this supports

- privileged teacher + public student distillation stack
- joint teacher-student hidden-world training
- staged or coupled teacher/student optimization with explicit information fencing

### Keep these terms alive

- privileged information
- `pi-Distill`
- OPSD
- teacher objective
- student objective
- KL collapse
- action-only PI
- reward-tilted posterior

---

## C13. Continual Policy Revitalization (offline-to-online RL)

### Core idea

This paper matters because hidden-world pass two probably needs better answer to offline-to-online transition than “pretrain and hope.” CPR argues offline-pretrained policies can become brittle or overtrained, and online fine-tuning can be unstable if policy is initialized from offline model and pushed forward naively. Proposed answer = periodic policy revitalization plus adaptive policy constraints so policy can recover learning capacity without catastrophic collapse.

### Exact mechanisms worth stealing

- continual policy revitalization,
- adaptive policy constraint during online optimization,
- explicit framing of offline-to-online instability,
- stable fine-tuning after offline pretraining.

### Why this matters for pass two

If hidden-world winner uses strong replay-side teacher or offline supervised warm-start, genie must explain how that evolves into real online strength. CPR is useful because it reminds answer not to treat offline teacher quality as end of story.

### Candidate family this supports

- offline teacher + online policy revitalization pipeline
- staged hidden-world pretraining with controlled online refinement

### Keep these terms alive

- offline-to-online RL
- continual policy revitalization
- adaptive policy constraint
- stable fine-tuning

---

# Context block D — how these papers should actually pressure the pass-two answer

This section exists so genie does not list papers and smile.

## D1. If the answer chooses a particle-based winner

Then it should explicitly answer:

- what backbone is,
- what proposal is,
- how event likelihood enters,
- how ESS is used,
- how resampling is done,
- how weighted means are exported,
- how teacher-side smoothing or replay windows interact with it,
- what compute or collapse evidence would kill family.

Relevant papers:

- FastSLAM
- Particle Filters tutorial
- Bayesian Data Assimilation

## D2. If the answer chooses a strongly amortized structured posterior

Then it should explicitly answer:

- what student object is,
- how structural legality is enforced,
- how confidence is calibrated,
- whether confidence gates search or trust,
- how teacher is generated,
- how teacher/student mismatch is audited,
- what failure would prove amortized path too lossy.

Relevant papers:

- Guo calibration
- ECE-estimation paper
- Mimori uncertainty calibration
- Privileged Information Distillation

## D3. If the answer chooses a latent world-model fast path

Then it should explicitly answer:

- how dynamics-consistency is enforced,
- how future public evidence pressures hidden state,
- how latent state remains interpretable enough to trust,
- what would show compact latent summary is too opaque or too hard to calibrate.

Relevant papers:

- DVBF
- EnKF
- Data Assimilation

## D4. If the answer chooses a blueprint-plus-search winner

Then it should explicitly answer:

- what blueprint policy is,
- when search is invoked,
- what continuation or leaf logic exists,
- what hidden-world object search consumes,
- how confidence gates search,
- what runtime latency is acceptable,
- what evidence would prove search lane not paying cost.

Relevant papers:

- Pluribus
- Suphx
- calibration papers
- active-inference-style information value only if it cashes out into real planner term

## D5. If the answer relies on privileged teachers

Then it must explicitly explain:

- how teacher is stronger,
- why student can still inherit benefit,
- how privileged information is fenced,
- how it is removed, masked, or distilled,
- what would count as evidence student only learned to cheat.

Relevant papers:

- Suphx
- Privileged Information Distillation

## D6. If the answer defines evaluation gates lazily

That should be treated as failure.

Relevant papers already show that:

- calibration metrics can mislead,
- empirical multiplayer strength can exist without clean guarantees,
- offline-to-online transitions can collapse,
- reward-side improvements can mask other problems.

Therefore pass-two answer should define:

- belief metrics,
- calibration metrics,
- action-quality metrics,
- paired arena or non-regression metrics,
- benchmark/latency thresholds.

---

# Context block E — minimal steering, not another rules memo

Context above = main payload. Steering below intentionally short.

1. Do not rerun pass-one diagnosis.
2. Choose primary winning design.
3. Name shortest honest tranche.
4. State where Hydra must be willing to be wrong.
5. Give measurable gates.
6. Give kill criteria.
7. If stronger design exceeds current Hydra support, say so instead of weakening it.

---

# Final ask

Given this context, answer pass-two question directly:

**What exact hidden-world design stack should Hydra build to gain real strength, what is shortest honest path to it, what current Hydra assumptions must be overturned if necessary, and what evidence would kill that rec if it is wrong?**