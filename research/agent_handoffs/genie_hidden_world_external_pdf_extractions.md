# Hydra hidden-world genie packet — PDF extraction notes

Hydra-facing PDF extraction notes for external artifact bank. Goal: not abstract summary. Goal: preserve paper parts most useful for Hydra hidden-world lane: target objs, update eqs, latent-structure assumptions, calibration discipline, failure modes, experiment ideas, fusion hooks.

Each extraction tries answer seven qs:

1. What paper claims?
2. What exact mechanisms or eqs worth carry?
3. What assumptions make method work in native domain?
4. What transfers cleanly to Hydra hidden-world lane?
5. What breaks under Mahjong public-info, finite-tile, multiplayer setting?
6. What experiments or diagnostics paper suggests, even indirectly?
7. Which phrases or named mechanisms worth preserve verbatim in genie packet?

---

## PDF X01 — Guo et al. (2017), On Calibration of Modern Neural Networks

- Source page: https://proceedings.mlr.press/v70/guo17a.html
- PDF: http://proceedings.mlr.press/v70/guo17a/guo17a.pdf

### Dense thesis

Guo et al. claim modern nets can gain accuracy yet worsen as probability estimators. Deeper/wider nets, batch norm, weaker weight decay can oversharpen confidence vs reality. Downstream systems trust those numbers. Hydra translation: model may rank tile danger well yet its 0.82 danger or 0.68 tenpai number may still lie. Core lesson: audit probability quality, not only accuracy.

### Technical mechanisms and equations

- **Perfect calibration:** `P(Y_hat = Y | P_hat = p) = p` for all confidence levels `p`.
- **Expected Calibration Error (ECE):** bin preds by confidence, compare avg confidence vs avg accuracy, take weighted avg gap.
- **Reliability diagrams:** confidence vs empirical success freq.
- **Negative log-likelihood (NLL):** calibration-sensitive objective; also fitting objective for temperature scaling.
- **Temperature scaling:** divide logits by learned scalar temperature `T` before softmax; `T > 1` softens confidence without changing class ranking.

More exact notebook-grade anchors worth preserving:

- In paper notation, `ECE = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)|`, with partition over confidence bins, not semantic state classes.
- Temperature scaling preserves rank. Small detail, big use: works on action-sensitive heads without changing favorite class while changing downstream trust.
- Empirical point not mere “calibration exists.” Deep modern nets can improve 0/1 error while overfitting NLL, so confidence surface worsens while argmax improves.

Useful concept: calibration = relation between score magnitude and correctness freq, not top-1 accuracy.

### Assumptions and scope limits

- Calibration fit assumes held-out validation slice represents deployment.
- Correction post-hoc; does not fix semantically wrong model.
- ECE summary metric, not full structural test.
- Paper studies standard classification outputs, not structured combinatorial latent objs.

### Clean Hydra transfer

Hydra can borrow paper near-directly for probability-like heads:

- tenpai probability,
- deal-in probability,
- tile occupancy marginals,
- opponent wait plausibility,
- search-confidence or trust scores,
- safety residual or risk estimates.

Paper also gives Hydra language to split “confidence sharpness” from “semantic correctness.” Big because belief target may be public-info-legal yet badly miscalibrated. Also may be marginally calibrated yet structurally invalid. Hydra needs both tests.

### Failure modes and non-transfer

- Temperature scaling will not fix wrong teacher obj.
- Calibration over marginals does not guarantee calibration over joint hidden worlds.
- Multiplayer strategic feedback means deployment distro may drift as policy changes.
- Public-info legality and tile conservation stay separate constraints.

### Hydra experiments inspired by the paper

- Reliability diagrams for opponent-tenpai preds.
- ECE for tile-danger preds by hand phase and seat position.
- Calibration-before-and-after temperature scaling on belief or danger logits.
- NLL-vs-action-quality curves: does better calibration improve discard choice?
- State-conditional calibration: easy public states vs messy ambiguous states.

### Candidate Hydra algorithm or head

- `ThermalLogitCalibrationHead`: tiny post-hoc calibration layer for belief-adjacent logits like tenpai, danger, or trust-in-search heads. Goal: not change hidden-world semantics. Goal: stop downstream policy logic from reading raw modern-net confidence as honest probability. Apply only after structural legality enforced elsewhere, because calibrated marginals over illegal hidden-world obj still junk. Best use: scalar or low-dim trust-sensitive heads, not full structured posterior obj.

### Vocabulary to preserve

- Expected Calibration Error
- reliability diagram
- temperature scaling
- overconfidence gap
- negative log-likelihood
- post-hoc calibration

### Short Hydra verdict

Paper does not tell Hydra what hidden-world obj should be. It tells Hydra: once obj exists, confidence over it must be audited or downstream policy will trust numbers too sharp to deserve trust.

---

## PDF X02 — Montemerlo et al. (2002), FastSLAM

- Source page: https://aaai.org/papers/00593-AAAI02-089-fastslam-a-factored-solution-to-the-simultaneous-localization-and-mapping-problem/
- PDF: http://robots.stanford.edu/papers/montemerlo.fastslam-tr.pdf

### Dense thesis

FastSLAM breakthrough not “use particles in robotics.” Breakthrough: huge joint posterior can sometimes factor into sampled backbone plus many smaller conditional objs. In SLAM, condition on robot path → landmark estimates become conditionally independent. Then global covariance monster becomes particle set with structured conditional state per particle. Hydra translation: do not attack full Mahjong hidden-world joint head-on if problem hides factored posterior obj.

### Technical mechanisms and equations

- **Factored posterior:** `p(path, landmarks | observations) = p(path | observations) * product_k p(landmark_k | path, observations)`.
- **Rao-Blackwellized particle filter:** sample hard latent obj, integrate or maintain easier conditional pieces analytically.
- **Importance weights:** update particle plausibility with new observation likelihood.
- **Per-particle state:** each hypothesis carries its own conditional substate.
- **Tree/path memory sharing:** only changed parts of structured belief need copying on update.
- **Per-particle data association:** multiple interpretations of same observation can be maintained simultaneously.

Source-specific details worth carrying into maximal packet:

- Complexity claim explicit, not vibe. Paper contrasts new scaling with EKF-style `O(K^2)` landmark handling; win comes only from Rao-Blackwellization plus tree-based structure sharing.
- Per-particle data-association step matters for Hydra because it legitimizes world where different hidden-state hypotheses explain same public event differently instead of collapsing to one interpretation.
- Source setting also relies on each conditional landmark estimate being small and Gaussian after conditioning. Sharp warning: Hydra must find its own conditional obj, not import robotics conditional verbatim.

### Assumptions and scope limits

- Native domain = continuous geometry with static landmarks.
- Conditional sub-updates rely on continuous Gaussian machinery.
- Factorization depends on path being right conditioning obj.
- Particle quality still depends on proposal quality and coverage.

### Clean Hydra transfer

Paper highly valuable for Hydra because it suggests hidden Mahjong world maybe should split into:

- sampled or enumerated coarse hidden-world backbone,
- count-constrained conditional allocations,
- opponent-local conditional summaries,
- shared structural state across nearby hidden-world hypotheses.

It also legitimizes maintaining multiple plausible interpretations of same discard or timing clue instead of averaging into impossible middle state.

### Failure modes and non-transfer

- Mahjong is discrete/combinatorial, not continuous/geometric.
- Opponents are strategic agents, not passive landmarks.
- “Condition on path, then solve everything else” factorization may not map 1:1; Hydra still must discover right conditioning obj.
- Any direct Gaussian update must be replaced or constrained.

### Hydra experiments inspired by the paper

- Compare flat joint posterior approx vs factored backbone-plus-conditional-allocation scheme.
- Test structural memory sharing for hidden-world hypotheses.
- Evaluate whether per-hypothesis interpretation of public events preserves useful rare branches.
- Benchmark equal compute: more particles with weaker conditionals vs fewer particles with stronger structured conditionals.

### Candidate Hydra algorithm

- `RB-TilePF`: Rao-Blackwellized tile-world particle filter where each particle samples coarse hidden-world backbone then fills opponent-zone or wall-allocation structure conditionally. Win condition not “more particles.” Win condition = backbone that induces useful conditional simplifications under finite-tile constraints. Algorithm survives only if shared-state reps and projection rules keep runtime from exploding; else expensive sampling story with no structural payoff.

### Vocabulary to preserve

- factored posterior
- Rao-Blackwellized particle filter
- conditional independence given path
- per-particle structured state
- data association
- shared-tree memory reuse

### Short Hydra verdict

FastSLAM is one of strongest cross-field precedents for “giant hidden posterior can become manageable if right latent backbone is factored.” It does not hand Hydra exact factorization, but strongly suggests answer is structural, not monolithic.

---

## PDF X03 — Friston et al. (2017), Active Inference: A Process Theory

- Source page: https://direct.mit.edu/neco/article/29/1/1/8207/Active-Inference-A-Process-Theory
- PDF: https://activeinference.github.io/papers/process_theory.pdf

### Dense thesis

Paper proposes perception, action, learning all can be framed as minimizing variational free energy or maximizing model evidence. Hydra-relevant part not neuroscience skin. Hydra-relevant part: planning can decompose into pressure toward preferred outcomes plus pressure to reduce hidden-state uncertainty. That gives principled language for moves valuable because they reveal info or reduce ambiguity. In hidden-info multiplayer game, some moves are offensive/defensive and also epistemically clarifying.

### Technical mechanisms and equations

- **Variational free energy `F`:** roughly complexity minus accuracy; used for current-state inference.
- **Expected free energy `G`:** used for policy selection over future states.
- **Epistemic value:** expected info gain / uncertainty reduction.
- **Extrinsic or pragmatic value:** preference satisfaction / expected utility.
- **Precision parameter:** inverse-temperature-like confidence term controlling policy concentration.
- **Bayesian smoothing:** later evidence can update beliefs about past and future states.

### Assumptions and scope limits

- Core presentation uses MDP-like generative models and often discrete state spaces.
- Framework general and can become metaphor instead of mechanism if ungrounded.
- Standard presentations mostly single-agent and do not natively solve adversarial reasoning.

### Clean Hydra transfer

Hydra can borrow active-inference language selectively:

- use epistemic value for info-gathering actions,
- treat uncertainty reduction as real planning resource,
- treat confidence/trust in belief-conditioned planning as controllable precision term,
- allow posterior repair and smoothing when later public evidence changes story.

Especially useful if genie proposes hidden-world planner that explicitly values “clarifying opponent likely wait structure” or “collapsing wall uncertainty.”

### Failure modes and non-transfer

- Framework easy to romanticize, hard to operationalize.
- Mahjong multiplayer common-knowledge recursion and strategic deception not solved by generic active inference alone.
- Free-energy formulation can sound elegant while giving no concrete teacher obj, runtime seam, or eval gate.

### Hydra experiments inspired by the paper

- Compare reward-only move ranking vs reward-plus-information-gain move ranking on replay.
- Track whether uncertainty-reducing actions reduce later regret.
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

Best treated as conceptual fusion partner, not direct algorithm spec. Can sharpen genie language about information-seeking value, but must be grounded in Hydra’s exact target obj and runtime constraints.

---

## PDF X04 — Karl et al. (2016), Deep Variational Bayes Filters

- Source page: https://arxiv.org/abs/1605.06432
- PDF: https://arxiv.org/pdf/1605.06432.pdf

### Dense thesis

DVBF argues latent state matters only if it supports correct dynamics. Standard reconstruction-driven latent models often learn reps that explain current observation but fail to preserve info needed for future prediction. DVBF forces latent space to respect transition structure by making future observations push gradients back through transition dynamics. Hydra translation: hidden-world rep should not merely explain current public state; it should stay coherent as future public events arrive.

### Technical mechanisms and equations

- Deterministic transition conditioned on stochastic reparameterized noise: `z_{t+1} = f(z_t, u_t, beta_t)`.
- Lower-bound objective combining reconstruction/prediction quality with KL regularization over transition noise.
- Transition priors used to stop recognition model cheating.
- Locally linear transition variants as practical parameterizations.
- Annealing schedules to avoid “good reconstruction, bad dynamics” regimes.

### Assumptions and scope limits

- Assumes Markovian latent state can exist in chosen dimensionality.
- Often built for continuous latent dynamics, not sharp discrete combinatorial jumps.
- Needs known or modeled controls and transitions.
- Can drift into opaque latent objs hard to audit semantically.

### Clean Hydra transfer

Hydra could use DVBF to support three claims:

1. compact hidden-world summary should be judged by temporal coherence, not only snapshot fidelity,
2. future public evidence should pressure latent hidden-world summary to encode what matters,
3. “belief quality” could include roll-forward compatibility, not only current-state agreement.

Especially attractive if Hydra wants fast amortized world model beside more explicit search-grade teacher.

### Failure modes and non-transfer

- Multiplayer strategic opponents violate stationary-transition vibe of many latent SSMs.
- Mahjong has discrete branching, interrupts, hard constraints unlike smooth local dynamics.
- Purely latent rep can hide whether public-info legality and tile conservation are respected.

### Hydra experiments inspired by the paper

- Train hidden-world summary and test whether it predicts future public reveals better than snapshot-only baseline.
- Compare reconstruction-heavy belief losses vs dynamics-aware objectives.
- Audit latent roll-forward compatibility with actual future public events.
- Test whether compact latent state helps action quality without sacrificing interpretability.

### Vocabulary to preserve

- dynamics-consistent latent state
- transition-aware representation
- reparameterized transition
- system identification
- latent state-space model
- reconstruction vs dynamics tradeoff

### Short Hydra verdict

DVBF does not solve public-posterior semantics by itself, but gives strong external argument that any compact hidden-world model should be judged by temporal coherence, not only one-step plausibility.

---

## PDF X05 — Li et al. (2020), Suphx: Mastering Mahjong with Deep Reinforcement Learning

- Source page: https://arxiv.org/abs/2003.13590
- PDF: https://arxiv.org/pdf/2003.13590.pdf

### Dense thesis

Suphx argues strong Mahjong AI needs hybrid response to three pains: hidden info, ugly long-horizon credit assignment, irregular search structure. Signature ideas — global reward prediction, oracle guiding, runtime policy adaptation — all exist to clean learning signal or make action-time behavior more context-sensitive. For Hydra, Suphx matters because it proves strong Mahjong systems already use privileged-training ideas and selective runtime refinement, not naive pure RL.”

### Technical mechanisms and equations

- **Policy-gradient / actor style training** with entropy control.
- **Global Reward Prediction (GRP):** learn recurrent predictor of final outcome from intermediate round states to smooth reward assignment.
- **Oracle Guiding:** feed privileged hidden info during training and gradually remove it with dropout-like scheduling.
- **pMCPA:** parametric runtime adaptation by rolling out trajectories and updating policy params for current hand slice.
- Strong feature engineering and look-ahead features to support action quality.

Higher-value exact anchors from source:

- Policy-gradient system explicitly uses importance weighting to handle stale trajectories in async training, not plain actor-critic updates.
- Entropy coefficient dynamically adjusted toward target entropy, so exploration stability treated as engineering control problem, not static hyperparameter.
- GRP objective predicts final game reward from partial round prefixes, not merely next-step reward; it does long-horizon rank-aware credit assignment.
- Oracle Guiding is concrete, not vague teacher/student talk. Source uses decaying privileged-feature mask so student must survive as privileged access goes to zero.

### Assumptions and scope limits

- Requires large-scale training infra.
- Runtime adaptation expensive and not deployable under strict latency constraints.
- Oracle-guided transition from hidden-state access to public-only play must be handled carefully.
- Same-game precedent does not automatically tell Hydra which exact teacher obj is best.

### Clean Hydra transfer

- privileged teacher objs can help if deployment obeys public-info legality,
- long-horizon reward smoothing is real, not optional,
- same-game systems already mix amortized policy learning with runtime adaptation,
- actual Mahjong strength requires hidden-world lane to cash out into concrete action-value gains.

### Failure modes and non-transfer

- Standard MCTS assumptions do not hold cleanly in Mahjong because of interrupts and irregular turn order.
- Simple imitation of oracle policy is not enough if student lacks credible public-info rep.
- Reward shaping shortcuts can hide belief-side weakness instead of fixing it.

### Hydra experiments inspired by the paper

- Compare oracle-assisted vs public-only belief-side training.
- Test oracle critic vs oracle actor style guidance.
- Measure whether belief improvements help final discard choices or only internal metrics.
- Explore runtime adaptation only on narrow states where belief-conditioned search is strongest.

### Candidate Hydra algorithm

- `POGPA-Hydra`: Parametric Oracle-Guided Policy Adaptation for Hydra. Train with privileged hidden-state teachers, fade privileged info with staged masking schedule, pair student with global outcome predictor for long-horizon credit assignment, reserve runtime adaptation for states where belief quality is already high enough to justify local refinement. Big reason candidate matters: same-game evidence shows hidden-info relief, reward repair, runtime adaptation are coherent stack, not separate gimmicks. Kill criterion simple: if student gains vanish once privileged info is fully removed, method only learned to cheat.

### Vocabulary to preserve

- oracle guiding
- global reward prediction
- privileged training / public deployment
- runtime policy adaptation
- irregular game tree
- same-game precedent

### Short Hydra verdict

Suphx is strongest same-domain external anchor in packet. It does not define Hydra’s hidden-world obj, but strongly supports selective oracle-assisted training and non-naive long-horizon reward repair.

---

## PDF X06 — Evensen (2003), The Ensemble Kalman Filter

- Source page: https://www.ecmwf.int/sites/default/files/elibrary/2003/9321-ensemble-kalman-filter-theoretical-formulation-and-practical-implementation.pdf
- PDF: https://www.ecmwf.int/sites/default/files/elibrary/2003/9321-ensemble-kalman-filter-theoretical-formulation-and-practical-implementation.pdf

### Dense thesis

Evensen’s EnKF paper gives mature recipe for belief maintenance in huge hidden systems: carry ensemble of plausible worlds, propagate forward, compare with new observations, update uncertainty without explicitly storing giant covariance matrix. Hydra-facing insight not “Mahjong should literally use Kalman filter.” Insight: large uncertain state tracking can work by evolving plausible world realizations and using observation mismatch to correct them.

### Technical mechanisms and equations

- Ensemble covariance estimated from perturbations around ensemble mean.
- Analysis update applied to each ensemble member.
- Innovation / forecast-observation mismatch as correction driver.
- State augmentation for uncertain latent params or model bias.
- Ensemble smoothing that revises past states using later evidence.

More exact notebook-grade anchors worth preserving:

- Classic member-wise update is `psi_j^a = psi_j^f + P_e^f H^T (H P_e^f H^T + R_e)^(-1) (d_j - H psi_j^f)`.
- Paper explicitly motivates ensemble-space methods as way to avoid tangent-linear and adjoint burden of EKF-like covariance propagation in huge systems.
- Source also stresses analysis ensemble remains weakly nonlinear combination of dynamically consistent forecast states; naive covariance filtering can damage internal logic.

### Assumptions and scope limits

- Gaussian-ish analysis assumptions.
- Continuous-state mentality.
- Independence assumptions between model and observation error processes.
- Small ensembles can create spurious confidence or collapse.

### Clean Hydra transfer

- maintain ensemble of hidden-world hypotheses,
- track latent opponent tendencies as slowly varying hidden params,
- use innovation-style diagnostics to detect when belief system is lying,
- support smoothing or retrospective belief repair.

### Failure modes and non-transfer

- Direct continuous Gaussian updates are bad literal fit for tile counts.
- “Soft” updates can violate hard finite-tile constraints.
- Public strategic actions are not passive sensor readings.

### Hydra experiments inspired by the paper

- Hidden-world ensemble plus innovation diagnostics on replay.
- Opponent-style parameter tracking as state augmentation.
- Fixed-lag smoothing using later public evidence.
- Compare unconstrained ensemble updates vs ensemble-plus-constraint-projection.

### Candidate Hydra algorithm

- `ProjectedEnsembleTracker`: carry ensemble over hidden-world summaries or opponent-style latent params, assimilate new public actions through innovation-style corrections, then project result back to legal discrete support. Not direct tile-allocation solver; cheaper uncertainty-over-summary tracker potentially useful around belief confidence, opponent tendencies, or search trust. Kill criterion: projection wipes out gain from ensemble update or creates mode-averaged nonsense.

### Vocabulary to preserve

- innovation
- ensemble perturbation
- forecast-analysis cycle
- state augmentation
- ensemble smoother
- covariance collapse / inbreeding

### Short Hydra verdict

EnKF more valuable as ensemble-state discipline than as literal final Hydra algorithm. Gives genie mature beliefs-about-beliefs language and strong case for smoothing and innovation diagnostics.

---

## PDF X07 — Arulampalam et al. (2002), Particle Filter Tutorial

- Source page: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/optreadings/Arulampalam_etal_2002.pdf
- PDF: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/optreadings/Arulampalam_etal_2002.pdf

### Dense thesis

Paper is canonical practical guide to sequential Monte Carlo in nonlinear, non-Gaussian settings. Real Hydra value: turns “we’ll sample hidden worlds” from slogan into engineering discipline. It explains posterior as weighted particles, why naive importance sampling degenerates, how resampling saves you, and how resampling can also destroy you through sample impoverishment. For Hydra, best outside artifact for taking sequential hidden-world inference seriously instead of treating it as merely another neural head.

### Technical mechanisms and equations

- State-space model with prediction and update stages.
- Bayesian filtering recursion.
- Sequential importance weights.
- Effective sample size `N_eff`.
- Resampling and regularized variants.
- Auxiliary proposal ideas that focus particles where likelihood matters most.

More exact notebook-grade anchors worth preserving:

- Generic SIS update is `w_k^i ∝ w_(k-1)^i * p(z_k | x_k^i) * p(x_k^i | x_(k-1)^i) / q(x_k^i | x_(k-1)^i, z_k)`.
- Source makes sharp point: optimal proposal minimizes weight variance and is one-step posterior conditioned on previous particle and new observation.
- Degeneracy not maybe; tutorial treats weight-variance growth as default failure mode unless proposal quality and resampling logic are handled carefully.
- Systematic resampling gets special love because it stays linear-time while reducing Monte Carlo noise.

### Assumptions and scope limits

- Markovian state transition assumption.
- Need usable likelihood model.
- Severe sensitivity to proposal quality.
- Still expensive in large state spaces without domain structure.

### Clean Hydra transfer

- CT-SMC diagnostics and proposal engineering.
- Sequential maintenance of multiple hidden worlds under new public evidence.
- Rare but important branch preservation.
- Explicit collapse detectors instead of hand-wavy “belief low quality” complaints.

### Failure modes and non-transfer

- Naive PFs do not know finite tile conservation.
- Deterministic public actions induced by strategic policies can cause particle collapse if likelihood too sharp.
- Regularization designed for continuous spaces does not directly fit tile combinatorics.

### Hydra experiments inspired by the paper

- Monitor `N_eff` during replay-side teacher generation.
- Compare proposal families with and without public-history likelihood conditioning.
- Measure action-quality degradation after particle impoverishment.
- Study whether auxiliary proposals help preserve high-value rare opponent states.

### Candidate Hydra algorithm

- `EventConditionedCTSMC`: constrained particle system for hidden Mahjong worlds that proposes count-consistent worlds, reweights with public-event compatibility, tracks `N_eff`, resamples only when posterior has effectively collapsed. Source contribution not only eqs; also warning that proposal choice is half algorithm. For Hydra, algorithm lives first in replay-side teacher generation or diagnostics, because there you can afford better proposals, smoothing tricks, richer legality checks. If action-quality gains do not survive after enforcing finite-shared-pool legality, method should not graduate.

### Vocabulary to preserve

- sequential importance sampling
- effective sample size
- particle degeneracy
- sample impoverishment
- proposal density
- auxiliary resampling

### Short Hydra verdict

If genie proposes any sample-based hidden-world teacher or runtime hypothesis system, this paper should be part of evaluation contract. It provides failure modes and metrics that keep particle talk honest.

---

## PDF X08 — Mimori et al. (2021), Diagnostic Uncertainty Calibration

- Source page: https://proceedings.mlr.press/v130/mimori21a.html
- PDF: https://proceedings.mlr.press/v130/mimori21a/mimori21a.pdf

### Dense thesis

Mimori et al. extend calibration into setting where target itself is uncertain because multiple experts may disagree. Instead of treating uncertainty as merely “low confidence in one class,” paper models distro over class-probability estimates with Dirichlet-style obj and builds estimators that separate calibration and epistemic components even when ground truth is distributional. This matters to Hydra because many public Mahjong states do not justify fake single hidden-world label. Paper gives genie external precedent for uncertainty objs richer than one-point targets.

### Technical mechanisms and equations

- Label histograms built from multiple annotations.
- Decomposition of loss into calibration-related and epistemic pieces.
- **Alpha-calibration:** learn concentration parameter over base predictive distro.
- Dirichlet uncertainty over class-probability estimates.
- Posterior update of uncertainty obj when new evidence arrives.

More exact notebook-grade anchors worth preserving:

- Source’s alpha-calibration gives closed-form disagreement and posterior update expressions, so more than hand-wavy “uncertainty head.”
- Unbiased epistemic-loss correction depends explicitly on multi-rater counts, important because it tells Hydra where clean eval story breaks.
- Empirical win not merely class accuracy; also disagreement-probability quality and posterior refinement quality after new evidence arrives.

### Assumptions and scope limits

- Natural domain = multiclass label distros, not combinatorial structured worlds.
- Dirichlet families impose own shape assumptions.
- Multiple labels per item used for eval and interpretation.
- Does not natively handle hard structural constraints like shared tile counts.

### Clean Hydra transfer

- use richer uncertainty objs for hidden-world ambiguity,
- separate “belief is wrong” from “belief is appropriately uncertain,”
- treat new public reveals as posterior updates over uncertainty objs,
- create disagreement-aware diagnostics between teacher families or hidden-world generators.

### Failure modes and non-transfer

- Dirichlet over simple class simplex is far easier than constrained hidden Mahjong world.
- Expert disagreement is not identical to hidden-world multimodality.
- Method says little about finite-shared-pool constraints or strategic adaptation.

### Hydra experiments inspired by the paper

- Compare single-target belief supervision vs distribution-aware supervision.
- Evaluate whether confidence over belief objs predicts downstream regret.
- Use replay reveals as pseudo-expert evidence to update uncertainty objs.
- Study whether disagreement-aware metrics help choose better teacher families.

### Candidate Hydra algorithm or head

- `DirichletConcentrationHead`: emit base predictive belief plus concentration parameter representing how much trust Hydra should place in predictive obj. Same-game mapping not exact, but structural lesson valuable: uncertainty should sometimes be modeled as uncertainty over probability objs, not one low-confidence point estimate. Candidate worth keeping only if concentration signal predicts reveal surprise, action regret, or teacher disagreement better than plain scalar confidence head.

### Vocabulary to preserve

- label histogram
- alpha-calibration
- Dirichlet concentration
- disagreement-aware uncertainty
- posterior update of CPE / uncertainty object
- epistemic loss

### Short Hydra verdict

Paper gives genie principled excuse not to force single fake teacher label when public state supports many plausible hidden worlds.

---

## PDF X09 — Grudzien & Bocquet (2023), A Tutorial on Bayesian Data Assimilation

- Source page: https://cw3e.ucsd.edu/wp-content/uploads/2023/08/Grudzien_and_Bocquet_2023_Ch3.pdf
- PDF: https://cw3e.ucsd.edu/wp-content/uploads/2023/08/Grudzien_and_Bocquet_2023_Ch3.pdf

### Dense thesis

Tutorial presents data assimilation as principled Bayesian fusion of forecast model with incoming observations, often over finite assimilation window that permits retrospective belief repair. Hydra-facing power not any one filter or variational method. It is mature operational stance that belief tracking is loop: predict, ingest evidence, revise, smooth, optionally adapt model itself. Highly relevant to hidden-world modeling in Mahjong, where later public actions often reinterpret earlier hidden-world possibilities.

### Technical mechanisms and equations

- Bayes-law forecast/analysis cycle.
- Observation operator plus noise model.
- 3D-VAR and 4D-VAR style cost functions using prior-vs-observation tradeoffs.
- Tangent-linear / adjoint thinking for sensitivity to future evidence.
- Fixed-lag smoothing and data-assimilation windows.
- Covariance inflation and regularization ideas.

More exact notebook-grade anchors worth preserving:

- Source treats filtering and smoothing as genuinely different inference regimes, not minor impl tweaks.
- Variational DA works by minimizing cost trading prior consistency against observation mismatch, not merely nudging point estimate.
- Tutorial also makes clear some strongest methods are trajectory-level estimators, which matters because replay teacher generation is naturally trajectory-inference problem.

### Assumptions and scope limits

- Often assumes Gaussian or locally linear approximations.
- Original domain is passive physical dynamics, not adversarial strategic systems.
- Observation noise stories in science differ from strategic policy-conditioned public actions.

### Clean Hydra transfer

- forecast hidden worlds,
- assimilate new public actions,
- smooth earlier posterior mass with later evidence,
- inflate uncertainty when model overcommits,
- treat teacher generation as full trajectory inference problem, not only local inference.

### Failure modes and non-transfer

- Strategic opponents violate passive-world assumption.
- Common-knowledge recursion not handled in basic DA derivations.
- Some variational machinery may be too heavy for runtime and belong only in teacher generation.

### Hydra experiments inspired by the paper

- fixed-lag smoothing for replay teacher generation,
- trajectory-window belief repair,
- compare filter-only vs smoother-based teacher objs,
- use posterior revision magnitude as diagnostic for hidden-world model mismatch.

### Candidate Hydra algorithm

- `FixedLagPublicSmoother`: run public-info filter forward through replay, then allow short lag window of backward repair when later public evidence changes what earlier hidden-world explanations were plausible. Attraction: improves teacher coherence without polluting runtime observability. Candidate belongs squarely in replay teacher generation and should be judged by whether smoothed teachers improve downstream action quality more than they inflate hindsight bias.

### Vocabulary to preserve

- forecast-analysis cycle
- data assimilation window
- filtering vs smoothing
- covariance inflation
- variational cost function
- adjoint sensitivity

### Short Hydra verdict

Tutorial gives genie strong mature language for why later evidence should be allowed to repair earlier beliefs. relevant to Hydra’s public-posterior closure story.

---

## PDF X10 — Brown & Sandholm (2019), Superhuman AI for Multiplayer Poker

- Source page: https://www.science.org/doi/10.1126/science.aay2400
- PDF: https://noambrown.github.io/papers/19-Science-Superhuman.pdf

### Dense thesis

Pluribus shows multiplayer imperfect-info superhuman play does not require beautiful general equilibrium guarantee. System wins by combining offline blueprint strategy with real-time depth-limited search and small continuation-strategy set at search frontier. Hydra-relevant lesson not “copy poker abstraction.” Real lesson: in multiplayer general-sum games, practical strength can come from robust empirical policy plus selective live refinement even when exploitability-style theory targets get ugly.

### Technical mechanisms and equations

- Blueprint strategy from self-play CFR-style training.
- Linear CFR weighting that emphasizes later iterations.
- Action and info abstraction.
- Depth-limited real-time search.
- Continuation-strategy sets instead of fixed leaf policies.
- AIVAT-style variance-reduced evaluation.

More exact notebook-grade anchors worth preserving:

- Paper’s linear-CFR weighting explicitly downweights early noisy regret contributions; useful general lesson: late iterations matter more.
- Continuation-strategy trick exists because imperfect-info leaf nodes cannot honestly get one frozen continuation policy.
- Source hardware and cost claims matter because they prove architecture is practical strategy system, not academic impossibility argument.

### Assumptions and scope limits

- Poker range structure is not Mahjong tile-allocation structure.
- Uses abstraction aggressively.
- Does not adapt online to specific opponents in same way rich opponent model might.
- Operates in different public-info density regime than Mahjong.

### Clean Hydra transfer

- strength-first multiplayer realism,
- selective search where it matters,
- continuation-style robustness at search leaves,
- robust empirical eval instead of over-fixating on perfect exploitability scalar.

### Failure modes and non-transfer

- Mahjong interrupts and richer public evidence change shape of search problem.
- CFR infra may be too expensive to treat as immediate Hydra default lane.
- Pluribus does not tell Hydra what public-posterior teacher obj is.

### Hydra experiments inspired by the paper

- blueprint-only vs blueprint-plus-live-search ablations,
- continuation-style leaf alternatives for robust-opponent response,
- variance-reduced promotion tests for search lanes,
- population or robustness tests rather than single-match conclusions.

### Candidate Hydra algorithm

- `OracleBlueprintBeliefSearch`: maintain strong amortized blueprint policy, then use selective belief-conditioned depth-limited search only where hidden-world uncertainty is tight enough that continuation strategies mean something. Source contribution that matters most not poker-specific abstraction; it is idea that multiplayer strength can come from robust blueprint plus smart local thinking. Kill criterion straightforward: if leaf continuation sets become meaningless under wide Mahjong hidden-world ambiguity, local search should abstain instead of hallucinating certainty.

### Vocabulary to preserve

- blueprint strategy
- depth-limited search
- continuation strategies
- multiplayer empirical robustness
- no-guarantees but works
- variance-reduced evaluation

### Short Hydra verdict

Pluribus is anti-paralysis paper. It tells genie not to worship theory beauty so hard that Hydra misses real strength path.

---

## Extraction-level synthesis

If these ten papers are read together, most useful meta-lessons for Hydra are:

1. **Posterior structure matters** more than raw posterior ambition. FastSLAM and particle-filter papers say factorization, proposal quality, collapse diagnostics matter.
2. **Uncertainty honesty matters** as much as point accuracy. Guo and Mimori say model can be sharp yet untrustworthy.
3. **Belief is process, not snapshot.** EnKF and data assimilation say forecast, assimilate, smooth, revise.
4. **Dynamic coherence matters.** DVBF says hidden-state summary is not real if it fails under future evidence.
5. **Information value is real.** Active inference says uncertainty reduction can itself be action-relevant.
6. **Multiplayer theory ugliness is not stop sign.** Pluribus and Suphx say strength can come from selective search, oracle-assisted training, and robust empirical validation even when math is ugly.

Genie therefore should not merely ask “which external paper looks coolest?” Right q: which fused formulation gives Hydra strongest public-info-legit hidden-world target obj, teacher hierarchy, and promotion plan?

## Second-pass candidate Hydra algorithms distilled from the extraction file

Extraction notebook now supports more explicit algorithm-first reading of external bank. Strongest candidate families are:

1. `RB-TilePF` — factor hidden Mahjong world around sampled backbone, then maintain structured conditional allocations under finite-tile constraints.
2. `EventConditionedCTSMC` — sequential Monte Carlo with event-aware proposals and ESS-gated resampling for replay teacher generation.
3. `FixedLagPublicSmoother` — filter forward on public info, then smooth backward across short lag window for better teacher quality.
4. `CalibratedStructuralPosterior` — structured belief outputs plus explicit calibration diagnostics and optional temperature-style repair on trust-sensitive heads.
5. `DirichletConcentrationHead` — ambiguity-aware concentration or disagreement head that measures confidence over probability objs.
6. `ProjectedEnsembleTracker` — ensemble-style uncertainty tracker over compressed hidden-world summaries or opponent-style latent vars.
7. `OracleBlueprintBeliefSearch` — strong blueprint policy plus selective belief-conditioned depth-limited search with continuation-style leaf handling.
8. `POGPA-Hydra` — same-game oracle-guided training and local adaptation stack with long-horizon reward repair.
9. `LatentRollForwardBeliefState` — reserve compact world-model candidate that must prove temporal coherence and legality compatibility.
10. `EpistemicAssistHeuristic` — reserve info-value bonus layered on top of grounded planner, not replacement for one.

Notebook also supports practical ranking. If goal is near-term buildable value, best external-backed candidates are `FixedLagPublicSmoother`, `CalibratedStructuralPosterior`, `OracleBlueprintBeliefSearch`, and reward-side lessons in `POGPA-Hydra`. If goal is maximum genie-grade breakthrough, highest-upside hidden-world candidates are `RB-TilePF` and `EventConditionedCTSMC`, with `ProjectedEnsembleTracker` and `LatentRollForwardBeliefState` as alternative compressed-world routes.