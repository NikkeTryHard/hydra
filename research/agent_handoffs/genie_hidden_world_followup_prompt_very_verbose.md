# Hydra hidden-world pass-two context packet — handcrafted verbose artifact file

This file is not meant to read like a thin instruction sheet. The point is to give the genie a **large, explicit context packet** for pass two, where the job is to stop asking whether the hidden-world lane matters and instead decide what exact design stack should win, what should be built first, what should be killed, and where Hydra must be willing to be wrong.

The first pass already did the broad diagnosis and target-object narrowing. This file is therefore intentionally biased toward:

- rich paper context,
- exact mechanisms,
- strong transfer arguments,
- assumptions and breakpoints,
- candidate algorithm families,
- training and evaluation implications,
- and explicit reminders that current Hydra support is not the outer boundary of the answer.

It is okay if the strongest design exceeds current Hydra support. That is part of the point.

## What pass two is actually for

Pass two is not “what hidden-world object exists?”

Pass two is:

- what full design stack should actually win,
- what shortest honest tranche preserves the winner’s semantics,
- what teacher hierarchy and training recipe should exist,
- what evaluation gates and kill criteria should govern promotion,
- what current Hydra assumptions should be overturned if they block the stronger design,
- and what evidence would prove the chosen path is fake strength rather than real strength.

So this file is built like a research notebook / artifact packet rather than a short prompt.

## Keep these high-level answer obligations in mind

The answer that comes out of this packet should still cover these top-level outputs:

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

But do not spend your budget re-listing rules. Spend it using the context below.

---

# Context block A — pass-two design pressure

The pass-two design winner must be judged on actual Hydra strength, not elegance alone.

That means the winning design should be evaluated against at least these tensions:

1. **Semantic honesty vs cheap approximations**
   - does the design preserve the real hidden-world object, or does it silently collapse it into a weaker surrogate?

2. **Teacher quality vs deployable legality**
   - can the strongest teacher use privileged signals while the student remains public-information legal?

3. **Search-grade fidelity vs amortized speed**
   - does the design maintain a real search-grade hidden-world object and then export distilled structure, or does it over-amortize too early?

4. **Calibration vs mere sharpness**
   - do confidence outputs deserve to be trusted, or do they only look decisive?

5. **Offline teacher strength vs online improvement stability**
   - does the training recipe explain how good teacher objects become actual stronger policies rather than brittle offline artifacts?

6. **Strength evidence vs internal metric theater**
   - does the design specify how improved hidden-world modeling cashes out into stronger action choices and stronger arena outcomes?

7. **Hydra compatibility vs Hydra correctness**
   - if the current carrier, gating support, or sequencing order blocks the strongest design, those should be changed rather than treated as sacred.

---

# Context block B — local design surfaces that matter for pass two

The following local surfaces matter for pass two more than they did for pass one, because they shape what can actually be trained, gated, promoted, or rejected.

## B1. CT-SMC as a real local teacher/search surface

Hydra already has a concrete CT-SMC implementation surface with:

- particle count,
- ESS threshold,
- explicit `update(...)`,
- weighted per-tile means,
- unweighted `mean_allocation()`,
- and systematic resampling.

That matters because pass two should not talk about CT-SMC only as a doctrine dream. It should decide whether CT-SMC is:

- merely an existence proof,
- a diagnostics-only object,
- the actual search-grade teacher,
- a bridge object to something stronger,
- or a lane that should lose to a better factorized or distillation-heavy design.

The most important pass-two question here is not “does CT-SMC exist?” but:

**Is the current CT-SMC surface strong enough to anchor the winning teacher stack, or should it be subordinated to a stronger hidden-world family?**

## B2. Train-bin support blocks are real design pressure

The current train entrypoints explicitly reject several advanced losses in the safe BC/RL entrypaths. That matters for pass two because if the strongest design requires belief or mixture-side objectives that train-bin currently blocks, the genie should say one of two things clearly:

- either the strongest honest tranche must change those contracts,
- or the stronger design is real but Hydra is currently too weak to train it honestly.

The wrong answer would be to weaken the design just to fit current train-bin support.

## B3. Head activation discipline already exists

Hydra already has a serious gate stack around advanced heads:

- density gates,
- sparse-search samples-per-parameter logic,
- warmup states,
- negative-transfer / gradient-conflict checks,
- and per-head state transitions.

Pass two should not reinvent promotion logic from nothing. It should decide whether this gate stack is:

- already the correct template for hidden-world promotion,
- missing crucial calibration and action-usefulness gates,
- or structurally wrong for the winning design.

## B4. DeltaQ promotion is an existence proof for “real gates”

Hydra already has a more serious promotion machinery than a lot of repos:

- comparison reports,
- policy-transfer reports,
- arena reports,
- explicit threshold structs,
- and rejection / require-arena / strong-promotion logic.

Pass two should look at this and ask:

**what would the hidden-world lane need, over and above this, to earn promotion honestly?**

## B5. Validation and paired arena surfaces already exist

Hydra’s validation and evaluation code already carries:

- policy agreement metrics,
- DeltaQ promotion metric collection,
- paired arena evaluation,
- non-regression vs strong-promotion decisions,
- benchmark gates like CT-SMC latency and self-play throughput,
- and stable-dan / mean-placement style evaluation.

This means pass two should output belief-lane evaluation gates that can actually plug into a real validation culture instead of inventing a detached benchmark religion.

## B6. ExIt validation harness is a template for teacher validation

Hydra already has a clean producer-validation harness pattern:

- emission rate,
- coverage,
- supported actions,
- root visits,
- KL,
- top-1 agreement,
- explicit rejection counts,
- threshold objects.

The hidden-world lane should almost certainly steal this pattern for teacher validation, even if the exact metrics differ.

## B7. Testing and falsification should be first-class

The current testing strategy reinforces that Hydra should not trust coverage alone, should use goldens and cross-validation, and should treat silently wrong labels as catastrophic. For pass two, that means the hidden-world lane should probably define:

- known-state golden beliefs,
- teacher/student roundtrip or replay validations,
- cross-validation against independent hidden-world samplers or baselines,
- and explicit regression tests around gating and calibration behavior.

---

# Context block C — core external paper bank for pass two

This section is where the real extra value should come from. The goal is not to list papers. The goal is to give the genie enough dense external context that it can design the winning pass-two stack rather than just wave at precedent.

## C1. Guo et al. (2017) — On Calibration of Modern Neural Networks

### Core idea

Guo et al. show that modern deep nets can become **more accurate and less trustworthy at the same time**. The paper’s core distinction is between classification accuracy and probability quality. A model that predicts the right class more often can still become dangerously overconfident. For a hidden-world lane, that matters because the downstream use of belief outputs is not just ranking — it is risk gating, search trust, and deployment-time confidence use.

The paper’s calibration condition is conceptually simple: among predictions made with confidence `p`, the empirical correctness rate should also be `p`. Modern nets violate this badly enough that the paper treats calibration repair as its own object, not a side effect of better classification.

### Exact mechanisms worth stealing

- Expected Calibration Error:
  `ECE = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)|`
- Reliability diagrams as visual evidence, not just scalar evidence.
- Temperature scaling as a rank-preserving repair of logit sharpness.
- NLL-vs-accuracy disconnect as a warning sign that confidence quality is decaying while top-1 still improves.

### Why this matters for pass two

Pass two needs to design **evaluation gates**, not just say “calibration matters.” Guo gives the first layer of that, but it is not enough on its own. The pass-two answer should probably say something like:

- calibration gates must exist,
- but naive scalar ECE alone is insufficient,
- and confidence should only gate search or trust if it is structurally and probabilistically honest.

### Candidate family this supports

- calibrated structural posterior
- confidence-gated search / trust systems
- temperature-corrected trust heads

### Where it breaks

- it is not a structured combinatorial hidden-world paper,
- it does not solve target semantics,
- and calibration over marginals does not imply calibration over whole hidden worlds.

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

This paper matters because pass two should not just inherit ECE lazily. The paper’s central point is that even calibration **evaluation itself** is estimator-sensitive, and naive habits around ECE can mislead. That is exactly the kind of thing a pass-two genie should know if it wants to design promotion gates instead of checklist theater.

The paper argues that calibration assessment deserves deeper analysis and that not all ECE estimators are equally good across settings. That matters for Hydra because a hidden-world lane may look “well calibrated” under one loose estimator and fail under stronger evaluation.

### Exact mechanisms worth stealing

- estimator-aware calibration evaluation,
- deeper analysis of binning and estimator bias,
- scenario-dependent choice of ECE estimator rather than one universal default.

### Why this matters for pass two

If the winning design relies on belief confidence, search trust, abstention-to-search, or risk gating, then the evaluation gates should probably reference stronger calibration-estimation practices rather than a single naive ECE number.

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

FastSLAM’s real gift to Hydra is not robotics. It is the idea that a horrible giant posterior may become tractable if you factor it around the right latent backbone. Once the correct path/backbone is conditioned on, the remaining pieces become cheaper conditional objects. That is a massively important pass-two signal, because it says the right response to combinatorial pain may be **structural factorization**, not giving up.

### Exact mechanisms worth stealing

- factored posterior decomposition,
- Rao-Blackwellized particle filters,
- per-particle structured state,
- tree/path memory sharing,
- per-particle data association.

### Why this matters for pass two

Pass one already used FastSLAM as inspiration. Pass two should do more: it should decide whether a Rao-Blackwellized tile-world family is actually the winner, a reserve family, or an overcomplicated trap.

The paper gives a strong case that if Hydra can find the right backbone, then a true hidden-world family may still be feasible. But it also gives a warning: if the backbone is wrong, the whole factorization story collapses.

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

This is the anti-hand-waving paper. If Hydra wants any sample-based hidden-world family, this tutorial says you do not get to say “just sample worlds” and stop there. You need proposal quality, weight updates, ESS, resampling strategy, and a plan for degeneracy and sample impoverishment.

### Exact mechanisms worth stealing

- SIS weight update:
  `w_k^i ∝ w_(k-1)^i * p(z_k | x_k^i) * p(x_k^i | x_(k-1)^i) / q(x_k^i | x_(k-1)^i, z_k)`
- `N_eff = 1 / sum_i (w_k^i)^2`
- optimal importance density idea,
- systematic resampling,
- explicit degeneracy monitoring.

### Why this matters for pass two

If the genie chooses a particle-like winner, it must also specify:

- how particles are proposed,
- what the public-event likelihood is,
- how ESS triggers work,
- and what resampling discipline prevents collapse.

If it cannot, the particle family should probably lose.

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

Data assimilation brings the forecast-assimilate-smooth worldview. That is a really strong fit for pass two because one of the sharpest questions is whether Hydra’s teacher should be a pure filter, a short-window smoother, or a hierarchy where smoothing is diagnostics-only. This paper says the right way to think is not one-shot inference but a sequence:

- forecast,
- absorb evidence,
- revise,
- smooth,
- maybe revise the model.

### Exact mechanisms worth stealing

- forecast-analysis cycle,
- filtering vs smoothing distinction,
- fixed-lag smoothing,
- variational trajectory correction,
- posterior repair under later evidence.

### Why this matters for pass two

This paper should pressure the genie to answer:

- is teacher generation filter-only,
- filter plus short smoothing,
- or filter for student, smoother for diagnostics?

It also helps sharpen the difference between a legal student target and a stronger replay-side teacher object.

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

The paper treats situations where “the label itself” is not a single clean object, because experts can disagree. That is valuable here because public Mahjong states often support multiple plausible hidden worlds. The paper says uncertainty is not just one low-confidence point estimate; sometimes uncertainty exists **over the probability object itself**.

### Exact mechanisms worth stealing

- label histograms,
- disagreement-aware calibration,
- Dirichlet-style uncertainty over class-probability estimates,
- posterior updates of uncertainty objects.

### Why this matters for pass two

If the genie wants the student or search stack to carry not only a belief object but also a “confidence in that object,” this paper is a better source than Guo. It is especially relevant for decisions about:

- trust gating,
- abstention to search,
- uncertainty-driven deferral,
- and what diagnostics prove a belief object is honest about ambiguity.

### Candidate family this supports

- concentration/confidence heads over a structured posterior
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

The EnKF matters because it provides a mature alternative way to think about giant hidden-state tracking: maintain an ensemble, update by innovation, keep uncertainty explicit, and use smoothing if needed. It is not the likely literal winner for Mahjong tile worlds, but it is strong pressure against pretending the only options are “flat amortized head” or “full particle monster.”

### Exact mechanisms worth stealing

- ensemble representation,
- innovation-based update,
- state augmentation,
- ensemble smoother,
- perturbed observations.

### Why this matters for pass two

If the genie wants a compressed uncertainty-over-summary tracker or an opponent-latent tracker instead of a full explicit tile-allocation posterior everywhere, EnKF-style thinking is useful.

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

DVBF says a hidden representation should be judged by whether it supports the correct dynamics, not just by whether it reconstructs the current snapshot. That is crucial if the genie wants to recommend a compact amortized hidden-world state rather than a fully explicit allocation object.

### Exact mechanisms worth stealing

- transition-aware latent state,
- reparameterized stochastic transitions,
- dynamics-consistent hidden representation,
- training pressure from future evidence.

### Why this matters for pass two

This paper pushes the genie to answer whether a compact latent world model is:

- the actual winner,
- only a reserve fast path,
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

This paper matters less as a literal algorithm spec and more as a language for information-sensitive action choice. If the genie wants to justify that some actions are good because they reduce hidden-world ambiguity, active inference gives a principled way to talk about epistemic value instead of hand-wavy “exploration.”

### Exact mechanisms worth stealing

- expected free energy,
- epistemic value vs pragmatic value,
- precision control,
- smoothing / posterior repair.

### Why this matters for pass two

It is especially relevant if the winning stack includes:

- trust-gated search,
- confidence-sensitive action choice,
- or an explicit “information value” term in a planner.

But this paper should probably remain a **reserve / fusion** artifact, not the core winner, unless the genie can cash it out into something testable.

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

Pluribus is the best pass-two reality check that multiplayer ugly theory does not kill practical strength. Blueprint plus local search plus continuation strategies was enough for six-player superhuman success.

### Exact mechanisms worth stealing

- blueprint strategy,
- depth-limited search,
- continuation strategies,
- discounted early-iteration regret via Linear CFR,
- variance-reduced evaluation like AIVAT.

### Why this matters for pass two

If the genie wants to recommend a belief-conditioned blueprint-plus-search stack, Pluribus is the strongest direct precedent for saying:

- yes, multiplayer theory is ugly,
- but robust empirical strength can still come from the right hybrid system.

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

Suphx is the same-game anchor. It is the reminder that strong Mahjong systems do not just need a nice hidden-state object; they need a stack that turns that object into better action quality under long-horizon, high-variance, multiplayer conditions. Suphx’s signature trio — global reward prediction, oracle guiding, and runtime policy adaptation — should stay in the pass-two context because they speak directly to training and action quality rather than just semantics.

### Exact mechanisms worth stealing

- global reward prediction across round prefixes,
- oracle guiding with feature dropout,
- pMCPA/runtime adaptation,
- look-ahead features,
- distributed self-play / training architecture.

### Why this matters for pass two

This is one of the strongest pass-two anchors because the winning hidden-world design must explain:

- how teacher-side privileged information becomes a legal student,
- how long-horizon reward repair interacts with the hidden-world lane,
- and whether runtime adaptation should exist beyond static policy inference.

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

This paper is extremely relevant for pass two because it sharpens the privileged-teacher problem beyond what Suphx alone gives. Its main point is that training-time privileged information can enable success on hard tasks, but the real difficulty is transferring that advantage to a student that must act **without** privileged information at inference time. The paper’s answer is a joint teacher-student objective in which a privileged-information-conditioned teacher and an unconditioned student are trained together rather than in isolated stages.

### Exact mechanisms worth stealing

- joint `pi-Distill` objective,
- explicit teacher objective and student objective,
- KL-coupled distillation,
- OPSD / on-policy self-distillation,
- privileged-information utility analysis,
- alpha scheduling between teacher and student pressure.

### Why this matters for pass two

This should strongly influence the hidden-world lane if the genie wants:

- a privileged search-grade or oracle-grade teacher,
- a public-information-legal student,
- and a concrete recipe for how the teacher advantage transfers instead of being hand-waved.

This paper gives much stronger pass-two pressure on the exact question:

**if the best teacher is stronger because it uses hidden or privileged information, how does the student inherit the benefit without inheriting the illegality?**

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

This paper matters because hidden-world pass two probably needs a better answer to the offline-to-online transition than “pretrain and hope.” CPR argues that offline-pretrained policies can become brittle or overtrained, and that online fine-tuning can be unstable if the policy is simply initialized from the offline model and pushed forward naïvely. The proposed answer is periodic policy revitalization plus adaptive policy constraints so the policy can recover learning capacity without catastrophic collapse.

### Exact mechanisms worth stealing

- continual policy revitalization,
- adaptive policy constraint during online optimization,
- explicit framing of offline-to-online instability,
- stable fine-tuning after offline pretraining.

### Why this matters for pass two

If the hidden-world winner uses a strong replay-side teacher or offline supervised warm-start, the genie needs to explain how that evolves into real online strength. CPR is a useful pass-two paper because it reminds the answer not to treat offline teacher quality as the end of the story.

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

This section exists so the genie does not just list papers and smile.

## D1. If the answer chooses a particle-based winner

Then it should explicitly answer:

- what the backbone is,
- what the proposal is,
- how event likelihood enters,
- how ESS is used,
- how resampling is done,
- how weighted means are exported,
- how teacher-side smoothing or replay windows interact with it,
- and what compute or collapse evidence would kill the family.

Relevant papers:

- FastSLAM
- Particle Filters tutorial
- Bayesian Data Assimilation

## D2. If the answer chooses a strongly amortized structured posterior

Then it should explicitly answer:

- what the student object is,
- how structural legality is enforced,
- how confidence is calibrated,
- whether confidence gates search or trust,
- how the teacher is generated,
- how teacher/student mismatch is audited,
- and what failure would prove the amortized path is too lossy.

Relevant papers:

- Guo calibration
- ECE-estimation paper
- Mimori uncertainty calibration
- Privileged Information Distillation

## D3. If the answer chooses a latent world-model fast path

Then it should explicitly answer:

- how dynamics-consistency is enforced,
- how future public evidence pressures the hidden state,
- how the latent state remains interpretable enough to be trusted,
- and what would show that the compact latent summary is too opaque or too hard to calibrate.

Relevant papers:

- DVBF
- EnKF
- Data Assimilation

## D4. If the answer chooses a blueprint-plus-search winner

Then it should explicitly answer:

- what the blueprint policy is,
- when search is invoked,
- what continuation or leaf logic exists,
- what hidden-world object search actually consumes,
- how confidence gates search,
- what runtime latency is acceptable,
- and what evidence would prove the search lane is not paying its cost.

Relevant papers:

- Pluribus
- Suphx
- calibration papers
- active-inference-style information value only if it cashes out into a real planner term

## D5. If the answer relies on privileged teachers

Then it must explicitly explain:

- how the teacher is stronger,
- why the student can still inherit the benefit,
- how privileged information is fenced,
- how it is removed, masked, or distilled,
- and what would count as evidence that the student only learned to cheat.

Relevant papers:

- Suphx
- Privileged Information Distillation

## D6. If the answer defines evaluation gates lazily

That should be treated as a failure.

Relevant papers already show that:

- calibration metrics can be misleading,
- empirical multiplayer strength can exist without clean guarantees,
- offline-to-online transitions can collapse,
- and reward-side improvements can mask other problems.

Therefore the pass-two answer should define:

- belief metrics,
- calibration metrics,
- action-quality metrics,
- paired arena or non-regression metrics,
- and benchmark/latency thresholds.

---

# Context block E — minimal steering, not another rules memo

The context above is the main payload. The steering below is intentionally short.

1. Do not rerun pass-one diagnosis.
2. Choose a primary winning design.
3. Name the shortest honest tranche.
4. State where Hydra must be willing to be wrong.
5. Give measurable gates.
6. Give kill criteria.
7. If a stronger design exceeds current Hydra support, say so instead of weakening it.

---

# Final ask

Given this context, answer the pass-two question directly:

**What exact hidden-world design stack should Hydra build to gain real strength, what is the shortest honest path to it, what current Hydra assumptions must be overturned if necessary, and what evidence would kill that recommendation if it is wrong?**
