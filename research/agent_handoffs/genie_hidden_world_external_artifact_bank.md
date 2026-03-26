# Hydra hidden-world genie packet — external artifact bank

This file is the deliberately overpacked outside-world artifact bank for Hydra’s hidden-world modeling lane. The target lane is not generic “belief” in the abstract. The actual target is a strong, deployable, public-information-legitimate model of the hidden world in four-player Riichi Mahjong: concealed tile allocation, wall uncertainty, opponent latent state, likely waits, danger, and downstream search/value implications under finite-tile conservation.

The point of this file is not to prove that Mahjong is secretly robotics or meteorology or medicine. The point is to give the genie a dense transfer bank from fields that already solved structurally similar subproblems: multi-hypothesis inference, posterior maintenance under partial observation, calibration of uncertainty, smoothing with delayed evidence, and strength-first decision-making in ugly imperfect-information settings.

The guiding rule for this bank is simple:

- keep the exact transferable mechanism,
- state why it matters to Hydra,
- state where the analogy breaks,
- do not pretend outside ideas transfer cleanly if they do not survive Hydra’s public-information and multiplayer constraints.

## How to use this bank inside a genie packet

Use this file in three passes:

1. **Mechanism discovery** — identify exact ideas that look isomorphic to Hydra’s hidden-world problem: Rao-Blackwellization, particle reweighting, calibration diagnostics, smoothing, continuation strategies, oracle-to-observable training, and disagreement-aware uncertainty.
2. **Constraint filtering** — kill anything that violates Mahjong’s finite-tile conservation, public-information legality, discrete combinatorics, or multiplayer strategic feedback.
3. **Fusion** — only keep the cross-field pieces that improve Hydra’s real target object: a public posterior / expected hidden allocation / search-grade world model that actually helps action quality.

The external artifacts below are intentionally heterogeneous. Some are very close to Hydra’s core problem, like particle filters, Pluribus, and Suphx. Others are weird on purpose, like medical disagreement calibration and weather data assimilation, because they bring mature language for uncertainty honesty and sequential posterior repair that game AI often hand-waves.

---

## Artifact E01 — On Calibration of Modern Neural Networks

- Domain: calibration / probabilistic reliability
- Artifact type: paper + PDF
- Canonical page: https://proceedings.mlr.press/v70/guo17a.html
- Direct PDF: http://proceedings.mlr.press/v70/guo17a/guo17a.pdf
- Suggested packaging label: `ext_guo17_calibration`

### Why this artifact belongs in the packet

Hydra’s hidden-world lane does not merely need accurate top-1 guesses about the hidden world. It needs **usable probabilities**. A posterior over waits, tile occupancy, tenpai risk, or danger is only valuable if its confidence means what it says. Guo et al. is the canonical external reminder that modern neural nets can improve accuracy while getting worse at probability quality. That is exactly the failure mode Hydra must avoid if belief outputs are going to influence search depth, risk gating, push/fold choices, or opponent-specific exploitation.

This paper is also useful because it is simple and brutal. It does not ask whether the model is philosophically Bayesian enough. It asks whether confidence tracks reality. That is the right pressure to apply to Hydra’s belief heads, danger heads, and any mixture or uncertainty outputs that later feed downstream policy logic.

### Technical ideas worth stealing

- **Perfect calibration condition:** a prediction with confidence `p` should be correct with frequency `p`.
- **Expected Calibration Error (ECE):** a compact scalar for “how wrong are the probabilities themselves?”
- **Reliability diagrams:** a visual audit of underconfidence vs overconfidence.
- **Temperature scaling:** a single-parameter post-hoc repair for overconfident logits that preserves ranking/argmax while changing confidence sharpness.
- **NLL-vs-accuracy disconnect:** a warning sign that the model is becoming more certain without becoming more trustworthy.

Useful formula anchors for the genie packet:

- Perfect calibration can be written as `P(Y_hat = Y | P_hat = p) = p`.
- ECE is the weighted bin-wise confidence/accuracy mismatch: `ECE = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)|`.
- Temperature scaling uses a learned scalar `T > 0` and replaces `softmax(z)` with `softmax(z / T)` while leaving the argmax unchanged.

Source-owned conditions worth preserving:

- the validation slice used to fit `T` must match the deployment distribution reasonably well,
- temperature scaling is a post-hoc confidence repair rather than a semantic repair,
- the paper’s strongest empirical warning is that high-capacity, batch-normalized modern nets can become more accurate while becoming more miscalibrated.

### Clean transfer path into Hydra

Hydra can use this artifact in several very concrete ways:

1. **Belief-head auditing:** if Hydra predicts the probability that a tile type lives in a zone, or the probability that an opponent is tenpai, ECE-style diagnostics can directly test whether those probabilities are honest.
2. **Danger calibration:** discard-danger estimates should not merely rank tiles; the actual predicted danger bands should mean something when aggregated over many states.
3. **Search confidence gating:** if search or Hand-EV logic relies on “belief confidence” or “posterior trust,” temperature-style calibration or state-conditional calibration checks can be a guardrail.
4. **Mixture sanity:** if Hydra ever emits multiple components or uncertainty estimates, this paper is a reminder that sharper uncertainty objects are not automatically better uncertainty objects.

### What does not transfer cleanly

- The paper is mostly about standard classification-style outputs, not structured combinatorial belief objects with hard constraints.
- Temperature scaling fixes **confidence sharpness**, not semantic target mistakes. If the teacher object is wrong, calibration will not magically make it correct.
- ECE alone is not enough for Hydra because a hidden-world model can be marginally calibrated but structurally illegal with respect to tile conservation or zone-size constraints.

### Hydra-facing diagnostics and experiments

- Plot reliability diagrams for predicted tenpai probabilities, deal-in probabilities, and tile occupancy marginals.
- Compare raw logits vs temperature-scaled logits on held-out replay slices.
- Audit calibration separately by game phase, seat position, and “messy” vs “clean” information states.
- Compute calibration not only on per-tile marginals but also on downstream action-triggered events like “this discard deals in” or “this opponent was actually in tenpai.”
- Pair calibration audits with constraint residual audits so Hydra does not celebrate calibrated nonsense.

Candidate Hydra fusion algorithm:

- `CalibratedStructuralPosterior`: keep Hydra’s structured belief object unchanged, but attach calibration audits and optionally a lightweight thermal-logit calibration layer to heads like tenpai, danger, and trust gates. The point is not to change the hidden-world semantics; the point is to stop overconfident logits from poisoning downstream risk and search decisions. This candidate is only valid if calibration is measured jointly with structural legality checks such as tile-conservation residuals.

### Named vocabulary worth preserving

- Expected Calibration Error (ECE)
- reliability diagram
- temperature scaling
- overconfidence gap
- negative log-likelihood / accuracy disconnect
- post-hoc calibration

### Pairing notes inside the genie packet

This artifact pairs especially well with:

- Mimori et al. for uncertainty over uncertainty rather than just point confidence,
- Hydra’s own opponent-modeling / Sinkhorn notes for structural calibration,
- any proposal that wants belief outputs to gate runtime behavior.

---

## Artifact E02 — FastSLAM: A Factored Solution to the Simultaneous Localization and Mapping Problem

- Domain: SLAM / localization / Rao-Blackwellized filtering
- Artifact type: paper + PDF
- Canonical page: https://aaai.org/papers/00593-AAAI02-089-fastslam-a-factored-solution-to-the-simultaneous-localization-and-mapping-problem/
- Direct PDF: http://robots.stanford.edu/papers/montemerlo.fastslam-tr.pdf
- Suggested packaging label: `ext_fastslam_factored_posterior`

### Why this artifact belongs in the packet

FastSLAM is probably the cleanest cross-field artifact for the exact sentence Hydra wants to be true: “a monstrous hidden-state posterior can become manageable if you factor it correctly.” The intellectual move is not robotics-specific. The deep lesson is that once you condition on the right latent backbone, the rest of the posterior can break into smaller conditional objects that are much cheaper to maintain.

That is gold for Hydra because the raw hidden Mahjong world is enormous. Opponent concealed hands, wall composition, tile-count constraints, and public-history conditioning create a combinatorial nightmare. FastSLAM says you should not attack the full joint directly if there is a factored posterior hiding inside it.

### Technical ideas worth stealing

- **Factored posterior:** conditional independence emerges after conditioning on the right latent trajectory/backbone.
- **Rao-Blackwellized particle filter:** sample the expensive latent object; maintain the rest analytically or semi-analytically.
- **Per-hypothesis structured state:** each particle carries its own conditional substate.
- **Shared-tree / path-copy data structures:** nearby hypotheses can share most of their memory and only fork where evidence forces divergence.
- **Per-particle data association:** different particles can maintain different interpretations of the same observation.

Useful formula anchors for the genie packet:

- the key factorization is `p(path, map | observations) = p(path | observations) * product_k p(map_k | path, observations)`;
- particle weights correct the mismatch between target posterior and proposal dynamics;
- per-particle data association means the same public event can be interpreted differently by different hidden-world hypotheses.

Source-owned conditions worth preserving:

- the tractability win depends on choosing the right conditioning backbone,
- the landmark conditionals in the source setting are Gaussian and low-dimensional,
- the claimed scaling win depends heavily on copy-on-write or shared-tree map storage rather than naive full-state copying.

### Clean transfer path into Hydra

Hydra can translate the FastSLAM mindset into several concrete design ideas:

1. **Backbone + conditional allocation split:** sample a coarser hidden-world summary or tile-count world, then derive zone-specific or opponent-specific beliefs conditionally.
2. **Structured posterior objects:** instead of a flat impossible belief tensor, maintain a set of weighted hidden-world hypotheses with shared structure.
3. **Partial recomputation:** when a new public event happens, update only the parts of the hidden-world representation that event actually touches.
4. **Multiple interpretation maintenance:** if a discard or timing signal is ambiguous, Hydra can keep several plausible latent explanations alive instead of averaging them into mush.

### What does not transfer cleanly

- Mahjong is discrete and adversarial, not continuous geometry.
- The “landmarks” in SLAM are static objects; Mahjong hidden objects move between wall, hands, melds, and visible zones.
- Kalman-style conditional updates are not a direct drop-in for discrete tile allocation.
- Opponents are strategic agents, not passive environmental features.

### Hydra-facing diagnostics and experiments

- Compare a monolithic posterior teacher to a factored backbone-plus-conditional-allocation teacher on calibration and runtime cost.
- Test memory-sharing representations for particle-like hidden-world hypotheses to avoid full state copies.
- Audit whether multiple hypothesis maintenance preserves rare but action-relevant branches better than a single collapsed expected allocation.
- Measure whether factorization improves posterior quality at fixed compute budget.

Candidate Hydra fusion algorithm:

- `RB-TilePF`: maintain particles over a coarse hidden-world backbone, then derive opponent-zone or wall-allocation structure conditionally, followed by a structural projection step to preserve finite tile counts. In Hydra terms, this is a research candidate for replay-side teacher generation or search-side hypothesis maintenance, not necessarily the first amortized runtime head. The anti-transfer gate is simple: if the chosen backbone does not actually induce useful conditional simplifications, this whole transfer collapses into an expensive monolith with particle overhead and no structural win.

### Named vocabulary worth preserving

- Rao-Blackwellization
- factored posterior
- conditional independence given path
- per-particle structured belief
- shared-tree memory reuse
- data association

### Pairing notes inside the genie packet

This artifact pairs best with:

- Arulampalam’s particle filter tutorial for disciplined SMC mechanics,
- Hydra’s own CT-SMC / Mixture-SIB doctrine,
- Sinkhorn-style structural projection to keep discrete count constraints honest.

---

## Artifact E03 — A Tutorial on Particle Filters for Online Nonlinear/Non-Gaussian Bayesian Tracking

- Domain: Bayesian tracking / sequential Monte Carlo
- Artifact type: tutorial paper + PDF
- Canonical PDF: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa12/optreadings/Arulampalam_etal_2002.pdf
- Suggested packaging label: `ext_arulampalam_particle_filters`

### Why this artifact belongs in the packet

If FastSLAM is the elegant factorization story, Arulampalam et al. is the practical discipline manual for sequential Monte Carlo. It is the best compact outside artifact for the nuts and bolts of particle filtering: weight updates, proposal choices, degeneracy, resampling, impoverishment, and diagnostic metrics. That matters because game-AI conversations love saying “just sample hidden states” without paying the real statistical cost of bad proposals or particle collapse.

Hydra needs this paper because its hidden-world lane lives or dies on **sequential evidence**. Every discard, meld, riichi, hesitation, or timing cue is a new observation. A real hidden-world tracker needs a mature way to reweight hypotheses over time rather than just produce one-shot guesses.

### Technical ideas worth stealing

- Recursive Bayesian filtering as **predict then update**.
- Sequential Importance Sampling (SIS) and the generic weight update rule.
- **Effective sample size** as a concrete degeneracy diagnostic.
- **Sampling Importance Resampling (SIR)** and **ASIR** as practical variants.
- **Sample impoverishment** as a real failure mode, not an implementation footnote.
- The idea that proposal quality matters as much as model architecture.

Useful formula anchors for the genie packet:

- the generic update is `w_k^i ∝ w_(k-1)^i * p(z_k | x_k^i) * p(x_k^i | x_(k-1)^i) / q(x_k^i | x_(k-1)^i, z_k)`;
- the effective sample size proxy is `N_eff = 1 / sum_i (w_k^i)^2`;
- the optimal importance density is the one-step posterior `q_opt(x_k | x_(k-1)^i, z_k) = p(x_k | x_(k-1)^i, z_k)`.

Source-owned conditions worth preserving:

- the filter assumes first-order Markov state evolution and conditionally independent observations given the current state,
- degeneracy is not a corner case; the source explicitly treats it as inevitable without better proposals or resampling discipline,
- resampling itself creates a second failure mode: sample impoverishment.

### Clean transfer path into Hydra

1. **CT-SMC discipline:** Hydra’s hidden-world search teacher or runtime seam can use ESS-style metrics to detect when the posterior has effectively collapsed.
2. **Proposal engineering:** public-state-aware proposals can bias sampling toward hidden worlds consistent with event likelihoods rather than blind count-consistent worlds.
3. **Rare-branch preservation:** better proposals can keep low-probability but high-consequence opponent states alive longer.
4. **Posterior maintenance language:** the paper gives Hydra a mature vocabulary for discussing particle collapse instead of vague “belief drift” complaints.

### What does not transfer cleanly

- Standard particle filters assume a measurement likelihood that is usually noisy and smooth. Mahjong observations can be strategic and sharply selective.
- Vanilla regularized particle filters blur continuous states; hidden tile allocations are discrete and count-constrained.
- The raw Mahjong hidden-state space is too large for naive particle methods without domain-specific structure.

### Hydra-facing diagnostics and experiments

- Track ESS over the course of a hand and correlate low ESS with catastrophic downstream decisions.
- Compare blind proposals vs event-aware proposals vs count-aware + event-aware proposals.
- Evaluate resampling schedules on diversity preservation and action quality.
- Measure how often particle collapse precedes belief overconfidence or search failure.

Candidate Hydra fusion algorithm:

- `EventConditionedCTSMC`: a constrained sequential importance sampler for hidden Mahjong worlds where proposals are conditioned on public events, reweighted by event compatibility, and resampled only when `N_eff` drops below a threshold. The important Hydra twist is that proposals must preserve finite-shared-pool legality, so generic SIR is not enough; proposal design and structural projection become part of the algorithm definition. This candidate belongs first in replay-side teacher generation and diagnostics, then maybe in selective runtime use if it survives the compute budget.

### Named vocabulary worth preserving

- Sequential Importance Sampling
- importance density
- effective sample size
- particle degeneracy
- sample impoverishment
- auxiliary proposals
- resampling threshold

### Pairing notes inside the genie packet

This artifact is the process-control partner for FastSLAM. Use it whenever the genie proposes sampling-based hidden-world teachers or runtime hypothesis tracking.

---

## Artifact E04 — The Ensemble Kalman Filter: Theoretical Formulation and Practical Implementation

- Domain: weather / geoscience / ensemble data assimilation
- Artifact type: reference paper + PDF
- Canonical PDF: https://www.ecmwf.int/sites/default/files/elibrary/2003/9321-ensemble-kalman-filter-theoretical-formulation-and-practical-implementation.pdf
- Suggested packaging label: `ext_evensen_enkf`

### Why this artifact belongs in the packet

The EnKF matters less as a literal algorithm candidate for Mahjong and more as a mature worldview for belief maintenance in massive hidden systems. Its central message is powerful: you can track a huge uncertain state by carrying an ensemble of plausible worlds and updating them with new observations, without storing or propagating a full exact posterior object.

That is useful for Hydra because a full exact combinatorial posterior is probably too expensive to be the practical target. Even if EnKF itself is not the right discrete update rule, its ensemble logic, update discipline, and smoothing mentality are all relevant.

### Technical ideas worth stealing

- Ensemble representation instead of explicit dense covariance.
- Update each ensemble member using information from a forecast-observation mismatch.
- State augmentation for persistent biases or latent parameters.
- Smoother variants that revise past states using future evidence.
- Explicit innovation diagnostics as a check for under- or over-confidence.

Useful formula anchors for the genie packet:

- the classic EnKF-style member update is `psi_j^a = psi_j^f + K * (d_j - H psi_j^f)` with the gain induced by forecast covariance and observation noise,
- innovation is the mismatch between predicted observation and actual observation,
- smoothing variants propagate later evidence backward into earlier ensemble states.

Source-owned conditions worth preserving:

- the source method is explicitly suboptimal outside Gaussian-ish regimes,
- small ensembles create spurious confidence and sampling artifacts,
- the analysis is designed as a combination of dynamically consistent forecast states, not arbitrary post-hoc fixes.

### Clean transfer path into Hydra

1. **Ensemble hidden-world reasoning:** Hydra can maintain a manageable ensemble of plausible hidden worlds rather than chasing a single point estimate.
2. **Opponent-style latent parameters:** persistent opponent tendencies can be treated like slowly varying latent parameters that get updated over time.
3. **Smoother mentality:** when later public evidence lands, Hydra can revise what earlier hidden-world explanations were plausible.
4. **Innovation checks:** if observed public actions consistently look impossible under the forecasted hidden-world ensemble, the model is missing something.

### What does not transfer cleanly

- EnKF’s Gaussian update assumptions are a bad literal fit for tile counts and discrete hidden allocations.
- Soft continuous updates can easily generate illegal fractional or over-counted tile beliefs unless heavily constrained.
- The observation operator in Mahjong is strategic and agent-dependent, not a passive physical sensor.

### Hydra-facing diagnostics and experiments

- Use ensemble innovation-style diagnostics for hidden-world mismatch.
- Maintain explicit opponent-style latent variables and measure whether they improve belief calibration.
- Test fixed-lag smoothing on replay to see how much later evidence should revise earlier beliefs.
- Explore “ensemble over hidden worlds + structural projection” as a hybrid rather than raw EnKF.

Candidate Hydra fusion algorithm:

- `ProjectedEnsembleTracker`: maintain an ensemble over hidden-world summaries or opponent-style latent parameters, assimilate new public actions using innovation-style updates, then project the result back onto a count-consistent discrete support. This is not meant to replace explicit posterior teachers everywhere; it is a candidate for a cheap uncertainty-over-summary tracker that carries smoother-like revision behavior. If projection destroys the information gained by the ensemble update, the candidate should be cut.

### Named vocabulary worth preserving

- ensemble representation
- innovation
- state augmentation
- ensemble smoother
- forecast-analysis cycle
- covariance collapse / inbreeding

### Pairing notes inside the genie packet

This artifact pairs naturally with the Bayesian data assimilation tutorial, because together they give the genie the forecast-assimilate-smooth mindset even if the final Hydra algorithm is discrete rather than Gaussian.

---

## Artifact E05 — A Tutorial on Bayesian Data Assimilation

- Domain: weather / sequential Bayesian estimation / forecast repair
- Artifact type: tutorial chapter + PDF
- Canonical PDF: https://cw3e.ucsd.edu/wp-content/uploads/2023/08/Grudzien_and_Bocquet_2023_Ch3.pdf
- Suggested packaging label: `ext_bayesian_data_assimilation`

### Why this artifact belongs in the packet

This is one of the best “weird but right” artifacts in the bank. The reason is simple: data assimilation is the mature scientific version of the exact problem Hydra faces at a high level. You have a hidden evolving world, partial noisy observations, a forward model that may be incomplete, and a repeated need to repair beliefs as new evidence arrives. That is unreasonably close to “public Mahjong history plus hidden tiles plus opponent behavior.”

The strongest transfer is not any one Kalman or variational formula. The strongest transfer is the operational loop: **forecast -> assimilate -> smooth -> revise model/uncertainty**.

### Technical ideas worth stealing

- Bayes-law framing of the forecast-analysis cycle.
- Distinction between **filtering** and **smoothing**.
- Data assimilation windows and fixed-lag retrospective updates.
- Variational objectives for reconciling a prior trajectory and new evidence.
- Covariance inflation / regularization as protection against false certainty.

Useful formula anchors for the genie packet:

- assimilation is forecast plus observation correction under Bayes law;
- variational updates can be framed as minimizing a prior-vs-observation cost over a trajectory window;
- smoothing means later evidence can update earlier latent-state beliefs rather than only the current state.

Source-owned conditions worth preserving:

- much of the classical DA stack assumes passive dynamics and observation noise models that are not strategic,
- some of the strongest methods use linearization or Gaussian approximations that can break in highly nonlinear regimes,
- the shift from filtering to smoothing is powerful but dangerous if future information leaks into runtime policy inputs.

### Clean transfer path into Hydra

1. **Forecast:** simulate or summarize plausible hidden-world evolution under public-history constraints.
2. **Assimilate:** incorporate the new discard/meld/riichi/timing event.
3. **Smooth:** revise earlier beliefs when later public evidence strongly reinterprets the past.
4. **Model repair:** if repeated public actions systematically violate the assumed hidden-world transition model, adapt the model class or uncertainty inflation.

This is especially compelling for Hydra because Mahjong often contains delayed evidence. A later reveal can make an earlier discard suddenly look much more meaningful. Data assimilation gives the genie a mature language for that posterior repair instead of leaving it as an informal intuition.

### What does not transfer cleanly

- Real-world DA often assumes passive environment dynamics rather than adversarial strategic agents.
- Some DA methods assume local linearity or Gaussian errors, which are weak fits for Mahjong’s discrete and strategic state space.
- Public-information games create common-knowledge recursion that standard DA papers do not usually model explicitly.

### Hydra-facing diagnostics and experiments

- Fixed-lag smoothing on replay: how much does belief quality improve if later evidence is allowed to revise the last `k` turns?
- Compare a pure filtering teacher to a smoothing-enhanced teacher.
- Use posterior repair magnitude as a diagnostic: if later evidence often forces huge rewrites, the forecast model is too weak.
- Explore lightweight “assimilation windows” for replay-side teacher generation.

Candidate Hydra fusion algorithm:

- `FixedLagPublicSmoother`: build replay-side hidden-world teachers by running a public-information filter forward, then performing a short-window backward smoothing pass when later public evidence strongly reinterprets the earlier state. This candidate is highly plausible for teacher generation because it improves belief coherence without demanding runtime future knowledge. Its hard caveat is non-negotiable: smoothed beliefs must never leak into student inputs as if they were live-time observables.

### Named vocabulary worth preserving

- forecast-analysis cycle
- filtering vs smoothing
- data assimilation window
- covariance inflation
- Mahalanobis distance
- adjoint / variational update

### Pairing notes inside the genie packet

This artifact is especially good to pair with Hydra’s own archive language about public posterior closure, because it legitimizes the idea that a teacher can and should revise earlier belief mass as later public evidence arrives.

---

## Artifact E06 — Diagnostic Uncertainty Calibration: Towards Reliable Machine Predictions in Medical Domain

- Domain: medical diagnosis / uncertainty calibration / disagreement modeling
- Artifact type: paper + PDF
- Canonical page: https://proceedings.mlr.press/v130/mimori21a.html
- Direct PDF: https://proceedings.mlr.press/v130/mimori21a/mimori21a.pdf
- Suggested packaging label: `ext_mimori_diagnostic_calibration`

### Why this artifact belongs in the packet

This is the bank’s best artifact for a subtle but extremely important point: sometimes the correct target is not a single label but a distribution over plausible judgments. That maps beautifully onto Hydra’s hidden-world problem because many public Mahjong states support multiple plausible hidden worlds. If the packet only thinks in terms of one “true” teacher label everywhere, it risks overcommitting to brittle targets.

The medical diagnosis angle is useful precisely because it is foreign. It forces the genie to take ambiguity seriously instead of assuming every uncertainty problem is just softmax confidence over one correct class.

### Technical ideas worth stealing

- Label histograms instead of single labels.
- Epistemic-vs-calibration-aware loss decomposition.
- **Dirichlet-style uncertainty objects** over class probability estimates.
- Post-hoc or learned concentration parameters controlling confidence in a distribution over probabilities.
- Efficient posterior updating when new evidence appears.

Useful formula anchors for the genie packet:

- the concentration-controlled disagreement object can be written as a Dirichlet over a base predictive distribution,
- posterior updates can be expressed in closed form when new evidence arrives,
- disagreement-aware metrics can distinguish model ignorance from inherent target ambiguity.

Source-owned conditions worth preserving:

- the source setting assumes conditionally i.i.d. annotations under roughly homogeneous expertise,
- the debiased estimators rely on multiple annotations or a multi-label signal, not a single hard label,
- Dirichlet shape assumptions may be too narrow for combinatorial hidden-world uncertainty.

### Clean transfer path into Hydra

1. **Ambiguity-aware targets:** represent hidden-world uncertainty as a structured distribution rather than a single teacher snapshot.
2. **Disagreement-aware evaluation:** compare multiple plausible teachers or multiple posterior generators rather than pretending there is one obvious truth.
3. **Confidence over belief objects:** not only “what is the predicted hidden state marginal?” but “how concentrated is the model’s belief over that marginal object?”
4. **Posterior update framing:** when a public reveal lands, update the uncertainty object rather than merely overwriting it.

### What does not transfer cleanly

- The paper’s natural object is a low-dimensional class simplex, not a giant combinatorial structured hidden world.
- Inter-rater disagreement is not identical to multi-world uncertainty under hard tile constraints.
- Dirichlet forms may be too restrictive or too lossy for Hydra’s actual hidden-world object.

### Hydra-facing diagnostics and experiments

- Compare single-point belief targets vs distribution-aware or disagreement-aware targets.
- Measure whether concentration/confidence estimates correlate with downstream action regret.
- Use replay reveals to test whether high-confidence hidden-world beliefs are actually deserved.
- Explore whether “uncertainty over probabilities” helps search know when not to overtrust the amortized fast path.

Candidate Hydra fusion algorithm:

- `DirichletConcentrationHead`: attach a concentration output to a structured or semi-structured belief head so Hydra can distinguish “high confidence because the state is clean” from “low confidence because the public evidence admits multiple worlds.” This candidate is most useful when coupled to teacher families or search trust gates, not as a replacement for the hidden-world semantics themselves. If the concentration signal fails to correlate with downstream action regret or reveal surprise, it should be dropped.

### Named vocabulary worth preserving

- label histogram
- disagreement-aware calibration
- Dirichlet concentration
- posterior update of uncertainty object
- epistemic loss
- probability-of-probabilities

### Pairing notes inside the genie packet

This artifact pairs best with Guo et al. Use Guo for “are the probabilities honest?” and Mimori for “is the uncertainty object itself rich enough to admit ambiguity?”

---

## Artifact E07 — Deep Variational Bayes Filters: Unsupervised Learning of State Space Models from Raw Data

- Domain: latent state-space modeling / world models
- Artifact type: paper + PDF
- Canonical page: https://arxiv.org/abs/1605.06432
- Direct PDF: https://arxiv.org/pdf/1605.06432.pdf
- Suggested packaging label: `ext_dvbf_world_model`

### Why this artifact belongs in the packet

DVBF belongs in the bank because it pushes a strong idea that Hydra needs to hear: a hidden-state representation is only meaningful if it supports the correct dynamics, not merely reconstruction of the current observation. In Hydra language, that means a hidden-world belief is not good because it looks plausible in the current snapshot; it is good because it evolves coherently as future public evidence arrives.

This is a powerful corrective against static belief targets. A world model for hidden Mahjong state should not only summarize the current concealed world; it should remain stable, updateable, and predictive over time.

### Technical ideas worth stealing

- Transition-aware latent representations.
- Reparameterized stochastic transitions with gradients flowing through time.
- Training pressure that forces the latent space to support prediction, not only reconstruction.
- Annealing schedules and priors that stabilize learning of dynamic latent structure.

Useful formula anchors for the genie packet:

- transition-aware latent models can be framed as `z_(t+1) = f(z_t, u_t, beta_t)` with reparameterized stochasticity,
- the training objective balances observation fit against regularized latent transition structure,
- future prediction error is allowed to shape the current latent state rather than only current reconstruction.

### Clean transfer path into Hydra

1. **Temporal consistency audits:** if Hydra’s belief representation is good, it should support future public-history prediction or future reveal compatibility.
2. **Transition-shaped latent belief spaces:** hidden-world summaries can be trained to preserve what matters for later evidence.
3. **Teacher smoothing:** future evidence can provide gradient signal about whether current belief summaries were dynamically coherent.
4. **World-model vocabulary:** useful if the genie proposes a compact latent hidden-world state rather than a fully explicit allocation tensor.

### What does not transfer cleanly

- DVBF assumes latent state transitions of a single evolving process, not four-player strategic adaptation.
- Mahjong has discrete combinatorial state jumps and interrupt mechanics that do not look like smooth continuous dynamics.
- A pure latent world model can become opaque and hard to calibrate against public-posterior semantics.

### Hydra-facing diagnostics and experiments

- Measure whether a learned latent hidden-world summary supports better prediction of future public reveals or future opponent actions.
- Compare static reconstruction-based belief training against dynamics-aware training.
- Audit whether hidden-state embeddings remain interpretable enough to preserve public-information legality and tile conservation.
- Use replay roll-forward compatibility as a diagnostic for whether the latent state actually encodes the right hidden variables.

Candidate Hydra fusion algorithm:

- `LatentRollForwardBeliefState`: maintain a compact amortized hidden-world summary that is explicitly trained to remain predictive under future public events, then pair it with an explicit structural teacher for calibration audits. This is a reserve candidate for Hydra’s fast path, not the first target object. If the latent state improves prediction but destroys calibration, legality, or interpretability, it belongs in reserve only.

### Named vocabulary worth preserving

- dynamics-consistent latent state
- transition-aware representation
- reparameterized transition noise
- system identification
- latent state-space model
- reconstruction vs dynamics tension

### Pairing notes inside the genie packet

This is a good fusion partner for active inference and data assimilation when the genie wants to propose a compact fast-path world model rather than only explicit allocation tensors.

---

## Artifact E08 — Active Inference: A Process Theory

- Domain: neuroscience / active inference / information-seeking control
- Artifact type: paper + PDF
- Canonical page: https://direct.mit.edu/neco/article/29/1/1/8207/Active-Inference-A-Process-Theory
- Direct PDF: https://activeinference.github.io/papers/process_theory.pdf
- Suggested packaging label: `ext_friston_active_inference`

### Why this artifact belongs in the packet

This is one of the most speculative artifacts in the bank, but it is here for a reason. Active inference gives the genie a principled language for moves that are valuable because they **reduce uncertainty**, not merely because they increase short-term reward. That matters in Mahjong because some actions change your hand value, some reduce danger, and some reveal information about the hidden world. Hydra needs a language for that triad.

Even if Hydra never adopts full active inference, expected free energy is useful as a framing device for “epistemic value plus pragmatic value” rather than hacky exploration bonuses.

### Technical ideas worth stealing

- Variational free energy for current-state inference.
- Expected free energy for policy selection.
- Epistemic value vs pragmatic value decomposition.
- Precision / confidence as part of policy selection.
- Bayesian smoothing and retrospective belief repair.

Useful formula anchors for the genie packet:

- planning pressure can be described as a combination of pragmatic value and epistemic value,
- precision acts like a trust or inverse-temperature control over policy concentration,
- expected free energy is attractive when uncertainty reduction itself has downstream value.

### Clean transfer path into Hydra

1. **Information-seeking interpretation of moves:** some discards can be valuable because they reduce uncertainty about opponents.
2. **Belief-aware planning objectives:** downstream search can explicitly value uncertainty reduction when it improves later safety or score quality.
3. **Confidence / precision control:** policy temperature or trust in belief-conditioned branches can be framed as precision allocation.
4. **Unified language:** useful for the genie when trying to explain a hidden-world planner that balances offense, defense, and information gathering.

### What does not transfer cleanly

- The framework is abstract and easy to oversell.
- Standard active-inference treatments are not built around public-information legality in multiplayer games.
- The math can become more inspirational than operational unless translated into Hydra’s exact objects and contracts.

### Hydra-facing diagnostics and experiments

- Estimate whether uncertainty-reducing actions correlate with later regret reduction.
- Compare fixed reward-only action ranking against reward-plus-epistemic-value ranking in replay analysis.
- Audit whether epistemic heuristics actually help or merely rationalize noise.
- Test whether policy “precision” should vary with belief certainty or state ambiguity.

Candidate Hydra fusion algorithm:

- `EpistemicAssistHeuristic`: a reserve-only auxiliary score that gives a small bonus to actions expected to collapse harmful hidden-world ambiguity, but only inside a grounded planner that already has a public-information-legitimate belief object. This is not a first-line target object or a replacement for search/value. If it cannot beat a simpler value-of-information heuristic in replay analysis, it should be rejected.

### Named vocabulary worth preserving

- variational free energy
- expected free energy
- epistemic value
- pragmatic value
- precision / inverse temperature
- active smoothing / posterior repair

### Pairing notes inside the genie packet

Use this artifact sparingly and concretely. It is best as a conceptual fusion layer on top of a more grounded hidden-world representation, not as a substitute for one.

---

## Artifact E09 — Superhuman AI for Multiplayer Poker (Pluribus)

- Domain: multiplayer imperfect-information game solving
- Artifact type: paper + PDF
- Canonical page: https://www.science.org/doi/10.1126/science.aay2400
- Direct PDF: https://noambrown.github.io/papers/19-Science-Superhuman.pdf
- Suggested packaging label: `ext_pluribus_multiplayer_search`

### Why this artifact belongs in the packet

Pluribus is the strongest reality check in the bank for the multiplayer theory problem. It proves that a multiplayer general-sum imperfect-information game can be crushed in practice without a clean exploitability story. For Hydra, this matters because it stops the genie from wasting all its effort trying to invent a beautiful 2p0s-style target that may not be the highest-leverage route to actual strength.

Pluribus also matters because it gives a concrete shape to a strength-first system: offline blueprint plus online depth-limited refinement, continuation strategies at the fringe, and empirical robustness instead of theoretical perfection.

### Technical ideas worth stealing

- Blueprint strategy computed offline.
- Real-time depth-limited search around the current subgame.
- Continuation-strategy sets at leaf nodes rather than pretending the future is fixed.
- Linear CFR weighting and multiplayer search pragmatism.
- Variance-reduced evaluation like AIVAT.

Useful formula anchors for the genie packet:

- the source uses weighted later-iteration regret contributions in linear CFR,
- the live search value of a leaf is approximated by a continuation-strategy set rather than a single frozen continuation,
- the key empirical lesson is that a static blueprint plus local online refinement can be enough for superhuman multiplayer performance.

Source-owned conditions worth preserving:

- the source system intentionally avoids direct online opponent adaptation,
- abstraction is used carefully and not uniformly across all decision slices,
- the result is empirical superiority, not an “unbeatable” guarantee.

### Clean transfer path into Hydra

1. **Strength-first evaluation stance:** real performance can outrank clean exploitability-style elegance.
2. **Selective search refinement:** use live search where it pays, not as a dogmatic everywhere-stack.
3. **Continuation-style leaf sets:** useful inspiration for Hydra’s robust-opponent or safe-exploitation future lanes.
4. **Multiplayer realism:** a reminder that Hydra may need strong empirical evaluation, population tests, and robustness metrics rather than one perfect scalar.

### What does not transfer cleanly

- Poker hand ranges and Mahjong tile-world posteriors are not identical.
- Mahjong has richer public sequential evidence and interrupt mechanics.
- CFR-style infrastructure scales differently in four-player Riichi than in poker abstractions.
- Pluribus does not solve Hydra’s public-posterior teacher object by itself.

### Hydra-facing diagnostics and experiments

- Compare blueprint-only vs selective-search variants in replay or arena tests.
- Test continuation-style leaf alternatives for opponent-response robustness.
- Use variance-reduced evaluation for search-lane promotion rather than trusting noisy match outcomes.
- Audit whether search improves action quality specifically in the states where belief uncertainty is still manageable.

Candidate Hydra fusion algorithm:

- `OracleBlueprintBeliefSearch`: pair a strong amortized blueprint policy with selective belief-conditioned depth-limited search, and evaluate leaf regions using a small continuation set rather than pretending the far future is fixed. This is appealing for Hydra because it naturally respects the reality that multiplayer theory is ugly while still giving runtime thinking room to matter. If belief uncertainty is too wide for the leaf continuation set to stay meaningful, the search should back off rather than hallucinate precision.

### Named vocabulary worth preserving

- blueprint strategy
- depth-limited search
- continuation strategies
- multiplayer no-guarantees but works
- empirical robustness
- variance-reduced evaluation

### Pairing notes inside the genie packet

This artifact should sit next to Hydra’s own multiplayer-search research notes. The combo keeps the genie honest about what theory does and does not buy in four-player general-sum settings.

---

## Artifact E10 — Suphx: Mastering Mahjong with Deep Reinforcement Learning

- Domain: frontier Mahjong AI
- Artifact type: paper + PDF
- Canonical page: https://arxiv.org/abs/2003.13590
- Direct PDF: https://arxiv.org/pdf/2003.13590.pdf
- Suggested packaging label: `ext_suphx_oracle_guiding`

### Why this artifact belongs in the packet

Suphx is the anchor that keeps the whole cross-field bank from floating away into clever nonsense. It is same-game evidence that strong Mahjong systems already use privileged training signals, long-horizon reward repair, and runtime adaptation. If the genie proposes a brilliant cross-field fusion that cannot cash out into Mahjong-strength mechanisms at least as concrete as Suphx, that proposal should be treated with suspicion.

Suphx is especially important because it reinforces the idea that the hardest part is not only hidden-state inference. The hardest part is turning hidden-state understanding into better action quality under long-horizon, high-variance, multiplayer conditions.

### Technical ideas worth stealing

- Oracle guiding / oracle-to-observable transfer.
- Global reward prediction for long-horizon credit assignment.
- Parametric runtime policy adaptation.
- Heuristic / look-ahead feature pipelines that support policy quality.
- Same-domain precedent for hybrid systems rather than pure end-to-end dogma.

Useful formula anchors for the genie packet:

- global reward prediction learns a final-outcome proxy over intermediate round slices,
- oracle guiding gradually removes privileged features instead of dropping them all at once,
- runtime adaptation uses rollout-conditioned policy adjustment rather than a full explicit search tree.

Source-owned conditions worth preserving:

- Mahjong’s irregular game tree is the reason pMCPA exists in the source system,
- the source result depends on large-scale infrastructure and substantial feature engineering,
- same-game success does not imply Hydra should import every component unchanged.

### Clean transfer path into Hydra

1. **Oracle teacher logic:** use privileged hidden information during training to shape a student that must eventually obey public-information deployment.
2. **Reward variance reduction synergy:** Suphx strongly supports Hydra’s interest in oracle critics and expected-reward nets.
3. **Runtime adaptation:** reinforces that action-time refinement can matter even if the final system is not a pure search machine.
4. **Mahjong reality check:** any hidden-world proposal should explain how it improves actual discard/push/fold/search decisions, not only belief metrics.

### What does not transfer cleanly

- Suphx’s exact policy stack is not automatically Hydra’s best stack.
- Oracle guiding is precedent, not proof that any particular Hydra teacher object is the right one.
- Runtime adaptation in Suphx was compute-sensitive and not always fully deployable in live settings.

### Hydra-facing diagnostics and experiments

- Compare public-only training vs oracle-assisted teacher generation.
- Evaluate whether hidden-world improvements actually reduce downstream decision regret rather than just improving belief scores.
- Test reward-prediction or oracle-critic variants specifically in belief-heavy states.
- Audit whether same-game adaptation ideas help without violating Hydra’s deployment constraints.

Candidate Hydra fusion algorithm:

- `POGPA-Hydra`: use privileged hidden-state teachers during training, distill them into a public-information student via a staged dropout or masking schedule, pair that with a long-horizon outcome predictor, and reserve runtime adaptation for the narrow hand slices where hidden-world quality is already strong enough to make adaptation meaningful. This is the strongest same-game bridge between belief quality and action quality in the whole external bank. Its kill criterion is also clear: if oracle-assisted gains fail to survive the public-information transition, the system is only learning to cheat.

### Named vocabulary worth preserving

- oracle guiding
- global reward prediction
- parametric runtime adaptation
- privileged training / public deployment
- same-game precedent
- hidden-information credit assignment

### Pairing notes inside the genie packet

This artifact should always be kept near Hydra’s own reward-design notes. Those two together give the genie a strong same-domain lane for reward-side improvements while the belief-side target object is being repaired.

---

## Cross-artifact fusion map

The bank is strongest when read as a fusion space, not as ten isolated citations. The most interesting cross-field fusions are:

### Fusion F01 — Factored posterior + sequential maintenance + structural projection

- FastSLAM for factorization.
- Arulampalam for SMC discipline.
- Hydra Sinkhorn notes for hard count constraints.

Hydra meaning: sample or maintain a manageable hidden-world backbone, update sequentially with public evidence, then project or constrain outputs so the final object respects finite-tile conservation and zone sizes.

### Fusion F02 — Ensemble belief + smoothing + delayed-evidence repair

- EnKF for ensemble maintenance.
- Bayesian DA for forecast/assimilate/smooth.
- Hydra belief-closure docs for public-posterior semantics.

Hydra meaning: stop treating belief as a one-shot framewise output and allow future public evidence to revise earlier posterior mass in a disciplined teacher-generation pipeline.

### Fusion F03 — Honest uncertainty instead of fake certainty

- Guo for calibration of confidence.
- Mimori for uncertainty over probability objects and disagreement-aware evaluation.
- Hydra opponent-modeling for structural consistency.

Hydra meaning: distinguish “wrong but honest uncertainty” from “sharp but bogus belief,” and refuse to activate heads whose uncertainty stories are not credible.

### Fusion F04 — Dynamics-consistent hidden-world summary

- DVBF for transition-aware latent summaries.
- Active inference for information-seeking value language.
- Hydra runtime/search docs for where a fast hidden-world summary might pay off.

Hydra meaning: if Hydra wants a compact amortized world model, it should be judged by whether it stays coherent under future public evidence and whether it helps action quality, not just whether it reconstructs the current board.

### Fusion F05 — Strength-first multiplayer realism

- Pluribus for multiplayer no-guarantees-but-works search thinking.
- Suphx for same-game privileged-training and reward-side realism.
- Hydra reconciliation/current-status docs for staged-vs-live constraints.

Hydra meaning: do not stall on the lack of a perfect exploitability scalar. Build what improves actual action quality and validate it with strong evaluation.

## Candidate Hydra algorithm slate distilled from the bank

These are the most explicit algorithm candidates that survive the current external bank after constraint filtering:

1. `RB-TilePF` — Rao-Blackwellized factored particle filtering over a coarse hidden-world backbone with conditional tile allocation and structural projection.
2. `EventConditionedCTSMC` — event-aware constrained SMC for replay-side hidden-world teacher generation and diagnostics.
3. `FixedLagPublicSmoother` — public-information filtering plus short-window backward smoothing for better teacher quality without runtime leakage.
4. `CalibratedStructuralPosterior` — structured belief outputs plus calibration audits and optional thermal-logit repair for trust-sensitive heads.
5. `DirichletConcentrationHead` — ambiguity-aware concentration or disagreement head layered on top of a structural belief object.
6. `ProjectedEnsembleTracker` — ensemble-style uncertainty tracker over compressed hidden-world summaries or opponent-style latents.
7. `OracleBlueprintBeliefSearch` — blueprint policy plus selective belief-conditioned depth-limited search with continuation-style leaf handling.
8. `POGPA-Hydra` — same-game oracle-guided policy adaptation stack with reward-side smoothing and strict public-deployment transition.
9. `LatentRollForwardBeliefState` — reserve-only compact latent world model trained for future public-event coherence.
10. `EpistemicAssistHeuristic` — reserve-only information-value bonus attached to a grounded planner.

For a buildability-first shortlist, the top external candidates are `FixedLagPublicSmoother`, `CalibratedStructuralPosterior`, `OracleBlueprintBeliefSearch`, and `POGPA-Hydra`. For a bigger genie-only research lane, the more ambitious posterior-family candidates are `RB-TilePF` and `EventConditionedCTSMC`.

## Short curation verdict

If the genie packet must keep only a compact outside spine, the highest-value external cluster is still:

1. `ext_fastslam_factored_posterior`
2. `ext_arulampalam_particle_filters`
3. `ext_guo17_calibration`
4. `ext_pluribus_multiplayer_search`
5. `ext_suphx_oracle_guiding`

That five-artifact spine gives Hydra:

- factored hidden-world inference,
- sequential posterior maintenance discipline,
- probability honesty,
- multiplayer search realism,
- and same-game strength precedent.

If the packet has room for a second layer, add:

6. `ext_bayesian_data_assimilation`
7. `ext_mimori_diagnostic_calibration`
8. `ext_evensen_enkf`
9. `ext_dvbf_world_model`
10. `ext_friston_active_inference`

That second layer gives the genie richer language for posterior repair, uncertainty-over-uncertainty, ensemble maintenance, dynamic latent structure, and epistemic action value — but those ideas should only survive if they cash out into a stronger Hydra target object and a cleaner buildable tranche.
