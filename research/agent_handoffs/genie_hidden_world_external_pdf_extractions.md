# Hydra hidden-world genie packet — PDF extraction notes

This file preserves detailed Hydra-facing extraction notes from downloaded PDFs used to seed the external artifact bank. The aim is not to restate abstracts. The aim is to preserve the parts of each paper that are most likely to matter for Hydra’s hidden-world modeling lane: target objects, update equations, latent-structure assumptions, calibration discipline, failure modes, and experiment ideas that a genie could fuse into a stronger formulation.

Each extraction below tries to answer seven questions:

1. What is the paper actually claiming?
2. What exact mechanisms or equations are worth carrying over?
3. What assumptions make the method work in its native domain?
4. What transfers cleanly to Hydra’s hidden-world lane?
5. What breaks under Mahjong’s public-information, finite-tile, multiplayer setting?
6. What experiments or diagnostics does the paper suggest, even indirectly?
7. Which phrases or named mechanisms are worth preserving verbatim in a genie packet?

---

## PDF X01 — Guo et al. (2017), On Calibration of Modern Neural Networks

- Source page: https://proceedings.mlr.press/v70/guo17a.html
- PDF: http://proceedings.mlr.press/v70/guo17a/guo17a.pdf

### Dense thesis

Guo et al. make a very specific and still underrated claim: modern neural networks can become more accurate while simultaneously becoming worse probability estimators. The paper shows that modern architectural choices like deeper nets, wider nets, batch normalization, and weaker weight decay can push models toward a regime where their confidence values are too sharp relative to reality. That matters because downstream systems often behave as if those probabilities were meaningful. In Hydra terms, this is the difference between “the model often guesses the right tile-danger ordering” and “the model’s 0.82 danger probability or 0.68 tenpai probability can actually be trusted as a number.” The key lesson is that decision systems need a probability-quality audit, not just an accuracy audit.

### Technical mechanisms and equations

- **Perfect calibration:** `P(Y_hat = Y | P_hat = p) = p` for all confidence levels `p`.
- **Expected Calibration Error (ECE):** bin predictions by confidence, compare average confidence to average accuracy, and take the weighted average gap.
- **Reliability diagrams:** visualize how confidence deviates from empirical success frequency.
- **Negative log-likelihood (NLL):** used as a calibration-sensitive objective and as the fitting objective for temperature scaling.
- **Temperature scaling:** divide logits by a learned scalar temperature `T` before softmax; `T > 1` softens confidence without changing class ranking.

More exact notebook-grade anchors worth preserving:

- In the paper’s notation, `ECE = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)|`, where the partition is over confidence bins rather than semantic state classes.
- Temperature scaling is intentionally rank-preserving. That sounds small, but it is exactly why it can be used on action-sensitive heads without changing the model’s favorite class while still changing how much trust downstream components assign to the probability.
- The paper’s actual empirical point is not just “calibration exists.” It is that deep modern nets can keep improving 0/1 error while overfitting NLL, meaning the confidence surface gets worse even while the argmax decisions improve.

The useful conceptual point is that calibration is about the relationship between score magnitude and correctness frequency, not about top-1 accuracy itself.

### Assumptions and scope limits

- The calibration fit assumes a held-out validation slice representative of deployment conditions.
- The correction is post-hoc; it does not repair a semantically wrong model.
- ECE is a summary metric, not a complete structural test.
- The paper studies standard classification outputs, not structured combinatorial latent objects.

### Clean Hydra transfer

Hydra can borrow this paper almost directly for any probability-like head:

- tenpai probability,
- deal-in probability,
- tile occupancy marginals,
- opponent wait plausibility,
- search-confidence or trust scores,
- safety residual or risk estimates.

The paper also gives Hydra a language for separating “confidence sharpness” from “semantic correctness.” That is huge because a belief target can be public-information-legal and still be badly miscalibrated. Conversely, it can also be calibrated in a marginal sense while still violating structural constraints. Hydra needs both tests.

### Failure modes and non-transfer

- Temperature scaling will not fix a wrong teacher object.
- Calibration over marginals does not guarantee calibration over joint hidden worlds.
- Multiplayer strategic feedback means the deployment distribution may drift as the policy changes.
- Public-information legality and tile conservation remain independent constraints.

### Hydra experiments inspired by the paper

- Reliability diagrams for opponent-tenpai predictions.
- ECE for tile-danger predictions by phase of hand and seat position.
- Calibration-before-and-after temperature scaling on belief or danger logits.
- NLL-vs-action-quality curves: does better calibration actually improve discard choice?
- State-conditional calibration: easy public states vs messy ambiguous states.

### Candidate Hydra algorithm or head

- `ThermalLogitCalibrationHead`: a tiny post-hoc calibration layer for belief-adjacent logits such as tenpai, danger, or trust-in-search heads. The point is not to change Hydra’s hidden-world semantics. The point is to stop downstream policy logic from reading raw modern-net confidence as if it were honest probability. This head should be applied only after structural legality is already enforced elsewhere, because calibrated marginals over an illegal hidden-world object are still junk. The most believable use is on scalar or low-dimensional trust-sensitive heads, not on the entire structured posterior object.

### Vocabulary to preserve

- Expected Calibration Error
- reliability diagram
- temperature scaling
- overconfidence gap
- negative log-likelihood
- post-hoc calibration

### Short Hydra verdict

This paper does not tell Hydra what the hidden-world object should be. It tells Hydra that once that object exists, confidence over it must be audited or the downstream policy will trust numbers that are too sharp to deserve trust.

---

## PDF X02 — Montemerlo et al. (2002), FastSLAM

- Source page: https://aaai.org/papers/00593-AAAI02-089-fastslam-a-factored-solution-to-the-simultaneous-localization-and-mapping-problem/
- PDF: http://robots.stanford.edu/papers/montemerlo.fastslam-tr.pdf

### Dense thesis

FastSLAM’s real breakthrough is not “use particles in robotics.” The breakthrough is recognizing that a huge joint posterior can sometimes be factored into a sampled backbone plus many smaller conditional objects. In SLAM, once the robot path is conditioned on, landmark estimates become conditionally independent. That lets the algorithm replace a global covariance monster with a particle set where each particle maintains structured conditional state. In Hydra terms, this is exactly the style of move we want: do not attack the full Mahjong hidden-world joint distribution head-on if there is a factored posterior object hiding inside the problem.

### Technical mechanisms and equations

- **Factored posterior:** `p(path, landmarks | observations) = p(path | observations) * product_k p(landmark_k | path, observations)`.
- **Rao-Blackwellized particle filter:** sample the hard latent object, integrate or maintain the easier conditional pieces analytically.
- **Importance weights:** update particle plausibility using new observation likelihood.
- **Per-particle state:** each hypothesis carries its own conditional substate.
- **Tree/path memory sharing:** only changed parts of the structured belief need to be copied on update.
- **Per-particle data association:** multiple interpretations of the same observation can be maintained simultaneously.

Source-specific details worth carrying into a maximal packet:

- The source paper’s complexity claim is not hand-wavy. It explicitly contrasts the new scaling against EKF-style `O(K^2)` handling of landmarks, and it only wins by pairing Rao-Blackwellization with tree-based structure sharing.
- The per-particle data-association step matters a lot for Hydra translation because it legitimizes a world where different hidden-state hypotheses explain the same public event differently instead of collapsing to one interpretation.
- The source setting also relies on each conditional landmark estimate being small and Gaussian after conditioning, which is a sharp warning that Hydra must find its own conditional object rather than importing the robotics conditional verbatim.

### Assumptions and scope limits

- The native domain is continuous geometry with static landmarks.
- Conditional sub-updates rely on continuous Gaussian machinery.
- The factorization depends on the path being the right conditioning object.
- Particle quality still depends on proposal quality and coverage.

### Clean Hydra transfer

This paper is extremely valuable for Hydra because it suggests that the hidden Mahjong world should maybe be broken into:

- a sampled or enumerated coarse hidden-world backbone,
- count-constrained conditional allocations,
- opponent-local conditional summaries,
- shared structural state across nearby hidden-world hypotheses.

It also legitimizes the idea that Hydra may need to maintain multiple plausible interpretations of the same discard or timing clue instead of averaging them into an impossible middle state.

### Failure modes and non-transfer

- Mahjong is discrete and combinatorial, not continuous and geometric.
- Opponents are strategic agents, not passive landmarks.
- The “condition on path, then solve everything else” factorization may not map 1:1 to Mahjong; Hydra still has to discover the right conditioning object.
- Any direct Gaussian update must be replaced or constrained.

### Hydra experiments inspired by the paper

- Compare a flat joint posterior approximation to a factored backbone-plus-conditional-allocation scheme.
- Test structural memory sharing for hidden-world hypotheses.
- Evaluate whether per-hypothesis interpretation of public events preserves useful rare branches.
- Benchmark equal compute: more particles with weaker conditionals vs fewer particles with stronger structured conditionals.

### Candidate Hydra algorithm

- `RB-TilePF`: a Rao-Blackwellized tile-world particle filter where each particle samples a coarse hidden-world backbone and then fills in opponent-zone or wall-allocation structure conditionally. The win condition is not “more particles.” The win condition is finding a backbone that actually induces useful conditional simplifications under finite-tile constraints. The algorithm only survives if shared-state representations and projection rules keep the runtime from exploding; otherwise it is just an expensive sampling story with no structural payoff.

### Vocabulary to preserve

- factored posterior
- Rao-Blackwellized particle filter
- conditional independence given path
- per-particle structured state
- data association
- shared-tree memory reuse

### Short Hydra verdict

FastSLAM is one of the strongest cross-field precedents for “a giant hidden posterior can become manageable if you factor the right latent backbone.” It does not hand Hydra the exact factorization, but it strongly suggests that the right answer is structural rather than monolithic.

---

## PDF X03 — Friston et al. (2017), Active Inference: A Process Theory

- Source page: https://direct.mit.edu/neco/article/29/1/1/8207/Active-Inference-A-Process-Theory
- PDF: https://activeinference.github.io/papers/process_theory.pdf

### Dense thesis

This paper proposes that perception, action, and learning can all be framed as minimizing variational free energy or maximizing model evidence. The Hydra-relevant part is not the neuroscience dress. The Hydra-relevant part is that planning can be decomposed into a pressure to reach preferred outcomes and a pressure to reduce uncertainty about hidden state. That gives a principled language for moves that are valuable because they reveal information or reduce ambiguity. In a hidden-information multiplayer game, that is immediately interesting because some moves are not just offensive or defensive; they are epistemically clarifying.

### Technical mechanisms and equations

- **Variational free energy `F`:** roughly complexity minus accuracy; used for current-state inference.
- **Expected free energy `G`:** used for policy selection over future states.
- **Epistemic value:** expected information gain / uncertainty reduction.
- **Extrinsic or pragmatic value:** preference satisfaction / expected utility.
- **Precision parameter:** effectively an inverse-temperature-like confidence term controlling policy concentration.
- **Bayesian smoothing:** later evidence can update beliefs about past and future states.

### Assumptions and scope limits

- The core presentation uses MDP-like generative models and often discrete state spaces.
- The framework is very general and can become more metaphor than mechanism if not grounded.
- Standard presentations are mostly single-agent and do not natively solve adversarial reasoning.

### Clean Hydra transfer

Hydra can borrow active-inference language in a very selective way:

- use epistemic value to reason about information-gathering actions,
- think of uncertainty-reduction as a real planning resource,
- treat confidence or trust in belief-conditioned planning as a controllable precision term,
- allow posterior repair and smoothing when later public evidence changes the story.

This is especially useful if the genie proposes a hidden-world planner that explicitly values “clarifying the opponent’s likely wait structure” or “collapsing uncertainty over the wall.”

### Failure modes and non-transfer

- The framework is easy to romanticize and hard to operationalize.
- Mahjong’s multiplayer common-knowledge recursion and strategic deception are not solved by generic active inference alone.
- A free-energy formulation can sound elegant without providing a concrete teacher object, runtime seam, or evaluation gate.

### Hydra experiments inspired by the paper

- Compare reward-only move ranking to reward-plus-information-gain move ranking on replay.
- Track whether uncertainty-reducing actions actually reduce later regret.
- Explore whether belief-conditioned search should have state-dependent precision or trust.
- Test whether some public actions deserve positive value because they clarify hidden-world structure.

### Vocabulary to preserve

- variational free energy
- expected free energy
- epistemic value
- pragmatic value
- precision / inverse temperature
- smoothing / posterior repair

### Short Hydra verdict

This paper is best treated as a conceptual fusion partner, not as a direct algorithm spec. It can sharpen the genie’s language about information-seeking value, but it must be grounded in Hydra’s exact target object and runtime constraints.

---

## PDF X04 — Karl et al. (2016), Deep Variational Bayes Filters

- Source page: https://arxiv.org/abs/1605.06432
- PDF: https://arxiv.org/pdf/1605.06432.pdf

### Dense thesis

DVBF argues that a latent state is only meaningful if it supports the correct dynamics. Standard reconstruction-driven latent models often learn representations that explain the current observation but do not preserve the information needed for future prediction. DVBF forces the latent space to respect the transition structure by making future observations push gradients back through the transition dynamics. In Hydra language, that means a hidden-world representation should not merely explain the current public state; it should stay coherent as future public events arrive.

### Technical mechanisms and equations

- Deterministic transition conditioned on stochastic reparameterized noise: `z_{t+1} = f(z_t, u_t, beta_t)`.
- Lower-bound objective combining reconstruction/prediction quality with KL regularization over transition noise.
- Transition priors used to prevent the recognition model from cheating.
- Locally linear transition variants as practical parameterizations.
- Annealing schedules to avoid getting stuck in “good reconstruction, bad dynamics” regimes.

### Assumptions and scope limits

- Assumes a Markovian latent state can exist in the chosen dimensionality.
- Often developed for continuous latent dynamics rather than sharp discrete combinatorial state jumps.
- Needs known or modeled controls and transitions.
- Can drift toward opaque latent objects that are hard to audit semantically.

### Clean Hydra transfer

Hydra could use DVBF as support for at least three claims:

1. a compact hidden-world summary should be evaluated by temporal coherence, not only snapshot fidelity,
2. future public evidence should pressure the latent hidden-world summary to encode what actually matters,
3. “belief quality” could include roll-forward compatibility, not just current-state agreement.

This is especially attractive if Hydra wants a fast amortized world model living alongside a more explicit search-grade teacher.

### Failure modes and non-transfer

- Multiplayer strategic opponents violate the stationary-transition vibe of many latent SSMs.
- Mahjong has discrete branching, interrupts, and hard constraints that do not look like smooth local dynamics.
- A purely latent representation can hide whether public-information legality and tile conservation are being respected.

### Hydra experiments inspired by the paper

- Train a hidden-world summary and test whether it predicts future public reveals better than a snapshot-only baseline.
- Compare reconstruction-heavy belief losses vs dynamics-aware objectives.
- Audit latent roll-forward compatibility with actual future public events.
- Test whether a compact latent state helps action quality without sacrificing interpretability.

### Vocabulary to preserve

- dynamics-consistent latent state
- transition-aware representation
- reparameterized transition
- system identification
- latent state-space model
- reconstruction vs dynamics tradeoff

### Short Hydra verdict

DVBF does not solve public-posterior semantics by itself, but it is a strong external argument that any compact hidden-world model should be judged by temporal coherence, not just one-step plausibility.

---

## PDF X05 — Li et al. (2020), Suphx: Mastering Mahjong with Deep Reinforcement Learning

- Source page: https://arxiv.org/abs/2003.13590
- PDF: https://arxiv.org/pdf/2003.13590.pdf

### Dense thesis

Suphx argues that strong Mahjong AI requires a hybrid response to three pain points: hidden information, ugly long-horizon credit assignment, and irregular search structure. The system’s signature ideas — global reward prediction, oracle guiding, and runtime policy adaptation — all exist to make learning signal cleaner or to make action-time behavior more context-sensitive. For Hydra, Suphx is crucial because it proves that strong Mahjong systems already rely on privileged-training ideas and selective runtime refinement, not on a naive “just do pure RL” story.

### Technical mechanisms and equations

- **Policy-gradient / actor style training** with entropy control.
- **Global Reward Prediction (GRP):** learn a recurrent predictor of final outcome from intermediate round states to smooth reward assignment.
- **Oracle Guiding:** feed privileged hidden information during training and gradually remove it with dropout-like scheduling.
- **pMCPA:** parametric runtime adaptation by rolling out trajectories and updating policy parameters for the current hand slice.
- Strong feature engineering and look-ahead features to support action quality.

Higher-value exact anchors from the source:

- The policy-gradient system explicitly uses importance weighting to handle stale trajectories in asynchronous training, not just plain actor-critic updates.
- The entropy coefficient is dynamically adjusted toward a target entropy, which is relevant because the paper treats exploration stability as an engineering control problem rather than a static hyperparameter.
- The GRP objective predicts final game reward from partial round prefixes, which matters because it is not merely predicting the next-step reward; it is doing long-horizon rank-aware credit assignment.
- Oracle Guiding is not a vague teacher/student metaphor. The source uses a decaying privileged-feature mask so the student is forced to survive as privileged access approaches zero.

### Assumptions and scope limits

- Requires large-scale training infrastructure.
- Runtime adaptation is expensive and not always deployable under strict latency constraints.
- Oracle-guided transition from hidden-state access to public-only play must be handled carefully.
- Same-game precedent does not automatically tell Hydra which exact teacher object is best.

### Clean Hydra transfer

- privileged teacher objects can be useful as long as deployment obeys public-information legality,
- long-horizon reward smoothing is real, not optional,
- same-game systems already mix amortized policy learning with runtime adaptation,
- actual Mahjong strength requires the hidden-world lane to cash out into concrete action-value improvements.

### Failure modes and non-transfer

- Standard MCTS assumptions do not hold cleanly in Mahjong because of interrupts and irregular turn order.
- Simple imitation of an oracle policy is not enough if the student lacks a credible public-information representation.
- Reward shaping shortcuts can hide belief-side weaknesses rather than solving them.

### Hydra experiments inspired by the paper

- Compare oracle-assisted vs public-only belief-side training.
- Test oracle critic vs oracle actor style guidance.
- Measure whether belief improvements help final discard choices or only internal metrics.
- Explore runtime adaptation only on the narrow states where belief-conditioned search is strongest.

### Candidate Hydra algorithm

- `POGPA-Hydra`: Parametric Oracle-Guided Policy Adaptation for Hydra. Train with privileged hidden-state teachers, fade privileged information using a staged masking schedule, pair the student with a global outcome predictor for long-horizon credit assignment, and reserve runtime adaptation for states where belief quality is already high enough to justify local refinement. The big reason this candidate matters is same-game evidence: the source shows that hidden-information relief, reward repair, and runtime adaptation are not separate gimmicks but a coherent stack. The kill criterion is brutal and simple: if the student’s gains disappear once privileged information is fully removed, the method is only learning to cheat.

### Vocabulary to preserve

- oracle guiding
- global reward prediction
- privileged training / public deployment
- runtime policy adaptation
- irregular game tree
- same-game precedent

### Short Hydra verdict

Suphx is the strongest same-domain external anchor in the packet. It does not define Hydra’s hidden-world object, but it strongly supports selective oracle-assisted training and non-naive long-horizon reward repair.

---

## PDF X06 — Evensen (2003), The Ensemble Kalman Filter

- Source page: https://www.ecmwf.int/sites/default/files/elibrary/2003/9321-ensemble-kalman-filter-theoretical-formulation-and-practical-implementation.pdf
- PDF: https://www.ecmwf.int/sites/default/files/elibrary/2003/9321-ensemble-kalman-filter-theoretical-formulation-and-practical-implementation.pdf

### Dense thesis

Evensen’s EnKF paper gives a mature recipe for belief maintenance in enormous hidden systems: carry an ensemble of plausible worlds, propagate them forward, compare them with new observations, and update uncertainty without explicitly storing a giant covariance matrix. The key Hydra-facing insight is not “Mahjong should literally use a Kalman filter.” The insight is that large-scale uncertain state tracking can work by evolving a set of plausible world realizations and using observation mismatch to correct them.

### Technical mechanisms and equations

- Ensemble covariance estimated from perturbations around the ensemble mean.
- Analysis update applied to each ensemble member.
- Innovation / forecast-observation mismatch as the correction driver.
- State augmentation for uncertain latent parameters or model bias.
- Ensemble smoothing that revises past states using later evidence.

More exact notebook-grade anchors worth preserving:

- The classic member-wise update is `psi_j^a = psi_j^f + P_e^f H^T (H P_e^f H^T + R_e)^(-1) (d_j - H psi_j^f)`.
- The paper explicitly motivates ensemble-space methods as a way to avoid the tangent-linear and adjoint burden of EKF-like covariance propagation in huge systems.
- The source also emphasizes that the analysis ensemble remains a weakly nonlinear combination of dynamically consistent forecast states, which is why naive covariance filtering can damage the method’s internal logic.

### Assumptions and scope limits

- Gaussian-ish analysis assumptions.
- Continuous-state mentality.
- Independence assumptions between model and observation error processes.
- Small ensembles can create spurious confidence or collapse.

### Clean Hydra transfer

- maintain an ensemble of hidden-world hypotheses,
- track latent opponent tendencies as slowly varying hidden parameters,
- use innovation-style diagnostics to detect when the belief system is lying,
- support smoothing or retrospective belief repair.

### Failure modes and non-transfer

- Direct continuous Gaussian updates are a bad literal fit for tile counts.
- “Soft” updates can violate hard finite-tile constraints.
- Public strategic actions are not passive sensor readings.

### Hydra experiments inspired by the paper

- Hidden-world ensemble plus innovation diagnostics on replay.
- Opponent-style parameter tracking as state augmentation.
- Fixed-lag smoothing using later public evidence.
- Compare unconstrained ensemble updates vs ensemble-plus-constraint-projection.

### Candidate Hydra algorithm

- `ProjectedEnsembleTracker`: carry an ensemble over hidden-world summaries or opponent-style latent parameters, assimilate new public actions through innovation-style corrections, then project the result back to a legal discrete support. This is not a direct tile-allocation solver; it is a cheaper uncertainty-over-summary tracker that may be useful around belief confidence, opponent tendencies, or search trust. The kill criterion is whether projection wipes out the gain from the ensemble update or creates mode-averaged nonsense.

### Vocabulary to preserve

- innovation
- ensemble perturbation
- forecast-analysis cycle
- state augmentation
- ensemble smoother
- covariance collapse / inbreeding

### Short Hydra verdict

EnKF is more valuable as ensemble-state discipline than as a literal final Hydra algorithm. It gives the genie mature beliefs-about-beliefs language and a strong case for smoothing and innovation diagnostics.

---

## PDF X07 — Arulampalam et al. (2002), Particle Filter Tutorial

- Source page: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/optreadings/Arulampalam_etal_2002.pdf
- PDF: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/optreadings/Arulampalam_etal_2002.pdf

### Dense thesis

This paper is the canonical practical guide to sequential Monte Carlo in nonlinear, non-Gaussian settings. Its real Hydra value is that it turns “we’ll sample hidden worlds” from a slogan into an engineering discipline. It explains how to represent a posterior as weighted particles, why naive importance sampling degenerates, how resampling saves you, and how resampling can also destroy you through sample impoverishment. For Hydra, it is the best outside artifact for taking sequential hidden-world inference seriously instead of treating it as just another neural head.

### Technical mechanisms and equations

- State-space model with prediction and update stages.
- Bayesian filtering recursion.
- Sequential importance weights.
- Effective sample size `N_eff`.
- Resampling and regularized variants.
- Auxiliary proposal ideas that focus particles where the likelihood will matter most.

More exact notebook-grade anchors worth preserving:

- The generic SIS update is `w_k^i ∝ w_(k-1)^i * p(z_k | x_k^i) * p(x_k^i | x_(k-1)^i) / q(x_k^i | x_(k-1)^i, z_k)`.
- The source makes a sharp point that the optimal proposal minimizes weight variance and is the one-step posterior conditioned on both the previous particle and the new observation.
- The degeneracy result is not a maybe; the tutorial treats variance growth of importance weights as the default failure mode unless proposal quality and resampling logic are handled carefully.
- Systematic resampling gets special love because it stays linear-time while reducing Monte Carlo noise.

### Assumptions and scope limits

- Markovian state transition assumption.
- Need for a usable likelihood model.
- Severe sensitivity to proposal quality.
- Still expensive in large state spaces without domain structure.

### Clean Hydra transfer

- CT-SMC diagnostics and proposal engineering.
- Sequential maintenance of multiple hidden worlds under new public evidence.
- Rare but important branch preservation.
- Explicit collapse detectors instead of hand-wavy “belief low quality” complaints.

### Failure modes and non-transfer

- Naive PFs do not know about finite tile conservation.
- Deterministic public actions induced by strategic policies can cause particle collapse if the likelihood is too sharp.
- Regularization designed for continuous spaces does not directly fit tile combinatorics.

### Hydra experiments inspired by the paper

- Monitor `N_eff` during replay-side teacher generation.
- Compare proposal families with and without public-history likelihood conditioning.
- Measure action-quality degradation after particle impoverishment.
- Study whether auxiliary proposals help preserve high-value rare opponent states.

### Candidate Hydra algorithm

- `EventConditionedCTSMC`: a constrained particle system for hidden Mahjong worlds that proposes count-consistent worlds, reweights them using public-event compatibility, tracks `N_eff`, and resamples only when the posterior has effectively collapsed. The source contribution here is not only the equations; it is the warning that proposal choice is half the algorithm. For Hydra, the algorithm lives first in replay-side teacher generation or diagnostics, because that is where you can afford better proposals, smoothing tricks, and richer legality checks. If action quality gains do not survive after enforcing finite-shared-pool legality, the method should not graduate.

### Vocabulary to preserve

- sequential importance sampling
- effective sample size
- particle degeneracy
- sample impoverishment
- proposal density
- auxiliary resampling

### Short Hydra verdict

If the genie proposes any sample-based hidden-world teacher or runtime hypothesis system, this paper should be part of the evaluation contract. It provides the failure modes and metrics that keep particle talk honest.

---

## PDF X08 — Mimori et al. (2021), Diagnostic Uncertainty Calibration

- Source page: https://proceedings.mlr.press/v130/mimori21a.html
- PDF: https://proceedings.mlr.press/v130/mimori21a/mimori21a.pdf

### Dense thesis

Mimori et al. extend calibration into a setting where the target itself is uncertain because multiple experts may disagree. Instead of treating uncertainty as merely “low confidence in one class,” the paper models a distribution over class-probability estimates using a Dirichlet-style object and builds estimators that separate calibration and epistemic components even when ground truth is distributional. This matters to Hydra because many public Mahjong states do not warrant a fake single hidden-world label. The paper gives the genie an external precedent for uncertainty objects richer than one-point targets.

### Technical mechanisms and equations

- Label histograms built from multiple annotations.
- Decomposition of loss into calibration-related and epistemic pieces.
- **Alpha-calibration:** learn a concentration parameter over a base predictive distribution.
- Dirichlet uncertainty over class-probability estimates.
- Posterior update of the uncertainty object when new evidence arrives.

More exact notebook-grade anchors worth preserving:

- The source’s alpha-calibration gives closed-form disagreement and posterior update expressions, which is why it is more than a hand-wavy “uncertainty head.”
- The unbiased epistemic-loss correction depends explicitly on multi-rater counts, which is important because it tells Hydra where the method’s clean evaluation story will break.
- The paper’s empirical win is not just on class accuracy; it is on disagreement-probability quality and posterior refinement quality after new evidence arrives.

### Assumptions and scope limits

- The natural domain is multiclass label distributions, not combinatorial structured worlds.
- Dirichlet families impose their own shape assumptions.
- Multiple labels per item are used for evaluation and interpretation.
- Does not natively handle hard structural constraints like shared tile counts.

### Clean Hydra transfer

- use richer uncertainty objects for hidden-world ambiguity,
- separate “belief is wrong” from “belief is appropriately uncertain,”
- treat new public reveals as posterior updates over uncertainty objects,
- create disagreement-aware diagnostics between teacher families or hidden-world generators.

### Failure modes and non-transfer

- A Dirichlet over a simple class simplex is far easier than a constrained hidden Mahjong world.
- Expert disagreement is not identical to hidden-world multimodality.
- The method says little about finite-shared-pool constraints or strategic adaptation.

### Hydra experiments inspired by the paper

- Compare single-target belief supervision to distribution-aware supervision.
- Evaluate whether confidence over belief objects predicts downstream regret.
- Use replay reveals as pseudo-expert evidence to update uncertainty objects.
- Study whether disagreement-aware metrics help choose better teacher families.

### Candidate Hydra algorithm or head

- `DirichletConcentrationHead`: emit a base predictive belief plus a concentration parameter representing how much trust Hydra should place in that predictive object. The same-game mapping is not exact, but the structural lesson is valuable: uncertainty should sometimes be modeled as uncertainty over probability objects, not just as one low-confidence point estimate. This candidate is only worth keeping if the concentration signal predicts reveal surprise, action regret, or teacher disagreement better than a plain scalar confidence head.

### Vocabulary to preserve

- label histogram
- alpha-calibration
- Dirichlet concentration
- disagreement-aware uncertainty
- posterior update of CPE / uncertainty object
- epistemic loss

### Short Hydra verdict

This paper gives the genie a principled excuse not to force a single fake teacher label when the public state actually supports many plausible hidden worlds.

---

## PDF X09 — Grudzien & Bocquet (2023), A Tutorial on Bayesian Data Assimilation

- Source page: https://cw3e.ucsd.edu/wp-content/uploads/2023/08/Grudzien_and_Bocquet_2023_Ch3.pdf
- PDF: https://cw3e.ucsd.edu/wp-content/uploads/2023/08/Grudzien_and_Bocquet_2023_Ch3.pdf

### Dense thesis

This tutorial presents data assimilation as the principled Bayesian fusion of a forecast model with incoming observations, often over a finite assimilation window that permits retrospective belief repair. The Hydra-facing power of the tutorial is not any one filter or variational method. It is the mature operational stance that belief tracking is a loop: predict, ingest evidence, revise, smooth, and optionally adapt the model itself. That stance is highly relevant to hidden-world modeling in Mahjong, where later public actions often reinterpret earlier hidden-world possibilities.

### Technical mechanisms and equations

- Bayes-law forecast/analysis cycle.
- Observation operator plus noise model.
- 3D-VAR and 4D-VAR style cost functions using prior-vs-observation tradeoffs.
- Tangent-linear / adjoint thinking for sensitivity to future evidence.
- Fixed-lag smoothing and data-assimilation windows.
- Covariance inflation and regularization ideas.

More exact notebook-grade anchors worth preserving:

- The source treats filtering and smoothing as genuinely different inference regimes rather than minor implementation tweaks.
- Variational DA works by minimizing a cost that trades off prior consistency against observation mismatch, not by merely nudging a point estimate.
- The tutorial also makes it clear that some of the strongest methods are trajectory-level estimators, which matters for Hydra because replay teacher generation is naturally a trajectory-inference problem.

### Assumptions and scope limits

- Often assumes Gaussian or locally linear approximations.
- Original domain is passive physical dynamics rather than adversarial strategic systems.
- Observation noise stories in science differ from strategic policy-conditioned public actions.

### Clean Hydra transfer

- forecast hidden worlds,
- assimilate new public actions,
- smooth earlier posterior mass with later evidence,
- inflate uncertainty when the model overcommits,
- treat teacher generation as a full trajectory inference problem, not just local inference.

### Failure modes and non-transfer

- Strategic opponents violate the passive-world assumption.
- Common-knowledge recursion is not handled in basic DA derivations.
- Some variational machinery may be too heavy for runtime and only belong in teacher generation.

### Hydra experiments inspired by the paper

- fixed-lag smoothing for replay teacher generation,
- trajectory-window belief repair,
- compare filter-only vs smoother-based teacher objects,
- use posterior revision magnitude as a diagnostic for hidden-world model mismatch.

### Candidate Hydra algorithm

- `FixedLagPublicSmoother`: run a public-information filter forward through a replay, then allow a short lag window of backward repair when later public evidence changes what earlier hidden-world explanations were plausible. The attraction here is that it improves teacher coherence without polluting runtime observability. This candidate belongs squarely in replay teacher generation, and it should be judged by whether smoothed teachers improve downstream action quality more than they inflate hindsight bias.

### Vocabulary to preserve

- forecast-analysis cycle
- data assimilation window
- filtering vs smoothing
- covariance inflation
- variational cost function
- adjoint sensitivity

### Short Hydra verdict

This tutorial gives the genie a strong mature language for why later evidence should be allowed to repair earlier beliefs. That is very relevant to Hydra’s public-posterior closure story.

---

## PDF X10 — Brown & Sandholm (2019), Superhuman AI for Multiplayer Poker

- Source page: https://www.science.org/doi/10.1126/science.aay2400
- PDF: https://noambrown.github.io/papers/19-Science-Superhuman.pdf

### Dense thesis

Pluribus shows that multiplayer imperfect-information superhuman play does not require a beautiful general equilibrium guarantee. The system wins by combining an offline blueprint strategy with real-time depth-limited search and a small continuation-strategy set at the search frontier. The Hydra-relevant lesson is not “copy poker abstraction.” The real lesson is that in multiplayer general-sum games, practical strength can come from robust empirical policy + selective live refinement, even when theoretical exploitability-style targets get ugly.

### Technical mechanisms and equations

- Blueprint strategy from self-play CFR-style training.
- Linear CFR weighting that emphasizes later iterations.
- Action and information abstraction.
- Depth-limited real-time search.
- Continuation-strategy sets instead of fixed leaf policies.
- AIVAT-style variance-reduced evaluation.

More exact notebook-grade anchors worth preserving:

- The paper’s linear-CFR weighting explicitly downweights early noisy regret contributions, which is useful as a general “late iterations matter more” design lesson.
- The continuation-strategy trick exists because imperfect-information leaf nodes cannot honestly be assigned one frozen continuation policy.
- The source’s hardware and cost claims matter because they prove the architecture is a practical strategy system, not an academic impossibility argument.

### Assumptions and scope limits

- Poker range structure is not Mahjong tile-allocation structure.
- Uses abstraction aggressively.
- Does not adapt online to specific opponents in the same way a rich opponent model might.
- Operates in a different public-information density regime than Mahjong.

### Clean Hydra transfer

- strength-first multiplayer realism,
- selective search where it matters,
- continuation-style robustness at search leaves,
- robust empirical evaluation instead of over-fixating on a perfect exploitability scalar.

### Failure modes and non-transfer

- Mahjong interrupts and richer public evidence change the shape of the search problem.
- CFR infrastructure may be too expensive to treat as an immediate Hydra default lane.
- Pluribus does not tell Hydra what the public-posterior teacher object is.

### Hydra experiments inspired by the paper

- blueprint-only vs blueprint-plus-live-search ablations,
- continuation-style leaf alternatives for robust-opponent response,
- variance-reduced promotion tests for search lanes,
- population or robustness tests rather than single-match conclusions.

### Candidate Hydra algorithm

- `OracleBlueprintBeliefSearch`: maintain a strong amortized blueprint policy, then use selective belief-conditioned depth-limited search only where the hidden-world uncertainty is tight enough that continuation strategies mean something. The source contribution that matters most is not poker-specific abstraction; it is the idea that multiplayer strength can come from a robust blueprint plus smart local thinking. The kill criterion is straightforward: if leaf continuation sets become meaningless under wide Mahjong hidden-world ambiguity, the local search should abstain instead of hallucinating certainty.

### Vocabulary to preserve

- blueprint strategy
- depth-limited search
- continuation strategies
- multiplayer empirical robustness
- no-guarantees but works
- variance-reduced evaluation

### Short Hydra verdict

Pluribus is the anti-paralysis paper. It tells the genie not to worship theory beauty so hard that Hydra misses the real strength path.

---

## Extraction-level synthesis

If these ten papers are read together, the most useful meta-lessons for Hydra are:

1. **Posterior structure matters** more than raw posterior ambition. FastSLAM and particle-filter papers say factorization, proposal quality, and collapse diagnostics matter.
2. **Uncertainty honesty matters** as much as point accuracy. Guo and Mimori say a model can be sharp and still untrustworthy.
3. **Belief is a process, not a snapshot.** EnKF and data assimilation say forecast, assimilate, smooth, revise.
4. **Dynamic coherence matters.** DVBF says a hidden-state summary is not real if it fails under future evidence.
5. **Information value is real.** Active inference says uncertainty reduction can itself be action-relevant.
6. **Multiplayer theory ugliness is not a stop sign.** Pluribus and Suphx say strength can come from selective search, oracle-assisted training, and robust empirical validation even when the math is not pretty.

The genie should therefore not merely ask “which external paper looks coolest?” The right question is: which fused formulation gives Hydra the strongest public-information-legitimate hidden-world target object, teacher hierarchy, and promotion plan?

## Second-pass candidate Hydra algorithms distilled from the extraction file

The extraction notebook now supports a more explicit algorithm-first reading of the external bank. The strongest candidate families are:

1. `RB-TilePF` — factor the hidden Mahjong world around a sampled backbone, then maintain structured conditional allocations under finite-tile constraints.
2. `EventConditionedCTSMC` — sequential Monte Carlo with event-aware proposals and ESS-gated resampling for replay teacher generation.
3. `FixedLagPublicSmoother` — filter forward on public information, then smooth backward across a short lag window for better teacher quality.
4. `CalibratedStructuralPosterior` — structured belief outputs plus explicit calibration diagnostics and optional temperature-style repair on trust-sensitive heads.
5. `DirichletConcentrationHead` — ambiguity-aware concentration or disagreement head that measures confidence over probability objects.
6. `ProjectedEnsembleTracker` — ensemble-style uncertainty tracker over compressed hidden-world summaries or opponent-style latent variables.
7. `OracleBlueprintBeliefSearch` — strong blueprint policy plus selective belief-conditioned depth-limited search with continuation-style leaf handling.
8. `POGPA-Hydra` — same-game oracle-guided training and local adaptation stack with long-horizon reward repair.
9. `LatentRollForwardBeliefState` — reserve compact world-model candidate that must prove temporal coherence and legality compatibility.
10. `EpistemicAssistHeuristic` — reserve information-value bonus layered on top of a grounded planner, not a replacement for one.

The notebook supports a practical ranking too. If the goal is near-term buildable value, the best external-backed candidates are `FixedLagPublicSmoother`, `CalibratedStructuralPosterior`, `OracleBlueprintBeliefSearch`, and the reward-side lessons in `POGPA-Hydra`. If the goal is maximum genie-grade breakthrough, the highest-upside hidden-world candidates are `RB-TilePF` and `EventConditionedCTSMC`, with `ProjectedEnsembleTracker` and `LatentRollForwardBeliefState` as alternative compressed-world routes.
