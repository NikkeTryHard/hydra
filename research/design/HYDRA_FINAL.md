# HYDRA-OMEGA: A Maximum-Ceiling 4-Player Riichi Mahjong AI (Complexity-Free)

**Single-source-of-truth (SSOT).** This document supersedes the two prior internal variants: the throughput-first "compute-constrained elegance" plan and the "information-geometric / all-out" plan. HYDRA-OMEGA keeps their best ideas, removes their ceilings, and adds a rigorously-grounded robustness layer.

---

## 0. Abstract

4-player Riichi Mahjong is a large, general-sum, imperfect-information game with a **finite shared hidden pool** (multivariate hypergeometric), **hard conservation constraints**, and **decision-critical correlations** that strengthen late game.

HYDRA-OMEGA is built around one central engine:

> **ExIt + Pondering + Search-as-Feature (SaF)**
> Deep anytime belief-search generates training targets continuously during self-play, amplified by opponent-turn idle time; those targets are amortized back into the policy/value networks so inference remains fast.

The system couples this engine with:

1. **Belief correctness with constraints**: SIB / Mixture-SIB (Sinkhorn KL projection onto conservation constraints) + particle posterior (SMC + rejuvenation) for correlation-critical regimes.
2. **Anytime Factored-Belief Search (AFBS)**: top-k pruning, heavy caching, incremental reuse, and predictive pondering.
3. **Robust opponent modeling inside search**: opponent nodes solved as distributionally robust soft-min within a KL uncertainty set around the learned opponent policy.
4. **Conservative safety math that is tight enough to matter**: Negative dependence / Strongly Rayleigh + Hunter/Kounias union tightening + bounded-error Monte Carlo intersections.
5. **Stable multiagent training**: DRDA (multiplayer POSGs) as the stability backbone, with PPO as a pragmatic high-throughput option.

Goal: **maximize expected Tenhou stable rank**; LuckyJ's 10.68 stable dan is the current public benchmark.

---

## 1. Design principles

### P1. Ceiling first, then amortize
If a mechanism raises ceiling but is too slow at inference, it belongs in pondering, teacher search, offline solvers, or distillation targets -- not in the critical inference loop.

### P2. The "teacher" must optimize the information state, not the hidden state
Any training target used to update the deployable policy must be a function of the public/information state, not privileged knowledge. We allow perfect-information networks for variance reduction and diagnostics, but the improvement operator must respect the information constraints.

### P3. Every "guarantee-like" claim must be either a theorem (with conditions), a bound (with explicit constants), or an empirical gate with a measurable pass/fail threshold.

### P4. Robustness is not optional in 4-player general-sum
Instead of equilibrium-style guarantees (which do not cleanly extend to 4p), we use distributional robustness: robust to belief error, robust to opponent policy misspecification, robust to population shifts.

---

## 2. Game model and notation

- Tile types: $k \in \{1,\dots,34\}$, multiplicity 4, total 136 tiles.
- Hidden locations: $z \in \{1,2,3,W\}$: three opponent concealed hands + wall remainder.
- Public information state at time $t$: $I_t$ (our hand, discards/melds, riichi, dora, scores, round meta).
- Remaining tile counts: $r_t(k) = 4 - \mathrm{visible}_t(k)$.
- Hidden location sizes: $s_t(z) \in \mathbb{Z}_{\ge 0}$, $\sum_z s_t(z) = \sum_k r_t(k)$.
- Hidden allocation matrix: $X_t \in \mathbb{Z}_{\ge 0}^{34\times 4}$, $\sum_z X_t(k,z) = r_t(k)$, $\sum_k X_t(k,z)=s_t(z)$.

Under purely random dealing, $X_t$ is multivariate hypergeometric; under strategic play, $p(X_t\mid I_t)$ is shaped by action likelihoods.

---

## 3. System overview -- four interacting loops

**Loop A: Belief loop** -- Mixture-SIB for fast marginal updates under constraints, particle SMC for joint correlation capture.

**Loop B: Search loop** -- AFBS on $I_t$ with belief $q_t$: on-turn (shallow, feature-producing), off-turn/pondering (deep, cached, predictive).

**Loop C: Distillation loop** -- Train policy/value to predict $\pi^{\text{ExIt}}$, $V^{\text{ExIt}}$, and calibrated safety features.

**Loop D: Population loop** -- League with self-play variants, human-style anchors, adversarial exploiters.

---

## 4. Neural architecture

### 4.1 Input tensor

**Group A -- Public encoding (~80-120 planes):** Hand, ordered discards (recency), open melds, riichi state, dora, round/scoring context, shanten/uke-ire summaries.

**Group B -- Safety planes (~23 planes):** Tenpai hints, furiten, genbutsu/suji/kabe safe-tile masks.

**Group C -- Search and belief features (dynamic, ~60-200 planes):** Belief marginals $B_t(k,z)$, mixture weights/entropy/ESS, AFBS action deltas $\Delta Q(a)$, risk estimates, robust opponent stress indicators. Zeroed with presence mask when unavailable.

### 4.2 Backbone

40-block SE-ResNet with GroupNorm(32) and Mish activation. ~16.5M parameters. bf16 precision. Inference: single forward pass ~0.5ms on RTX 5000.

### 4.3 Heads (multi-task)

**Core decision heads:** (1) Policy $\pi_\theta(a\mid I_t)$, 46 actions. (2) Value $V_\theta(I_t)$, scalar. (3) Score distribution: pdf + cdf (64 bins, KataGo-style).

**Opponent and safety heads:** (4) Opponent tenpai (3 sigmoids). (5) Opponent next discard (3x34). (6) Danger: per-tile deal-in probability (3x34).

**Belief heads:** (7) Mixture-SIB external fields $F_\theta^{(\ell)}(k,z)$ and mixture weight logits. (8) Opponent hand-type latent predictor.

**Search distillation heads:** (9) $\Delta Q$ regression (predict search advantage over baseline). (10) Safety bound residual (predict conservatism gap).

---

## 5. Belief inference: SIB, Mixture-SIB, and particle posterior

### 5.1 SIB as KL projection

Let $K_\theta(k,z)=\exp(F_\theta(k,z))>0$. The transportation polytope: $\mathcal{U}(r_t,s_t)=\{B\ge0: B\mathbf{1}=r_t, B^\top \mathbf{1}=s_t\}$.

**SIB operator:** $\mathrm{SIB}(K_\theta;r_t,s_t) := \arg\min_{B\in\mathcal{U}} D_{\mathrm{KL}}(B\|K_\theta)$. Solution: $B^*=\mathrm{diag}(u)\cdot K_\theta\cdot\mathrm{diag}(v)$ via Sinkhorn-Knopp.

### 5.2 Mixture-SIB for multimodality

$L$ components: $q_t(X)=\sum_{\ell=1}^L w_t^{(\ell)} q_t^{(\ell)}(X)$, each $B_t^{(\ell)}=\mathrm{SIB}(\exp(F_\theta^{(\ell)});r_t,s_t)$.

Weight update (Bayes): $w_{t+1}^{(\ell)}\propto w_t^{(\ell)} \cdot p_\phi(e_t\mid I_t, B_t^{(\ell)}, \ell)$. Anti-collapse via entropy regularizer, split-merge on low ESS, diversity penalty between components.

### 5.3 Particle posterior (SMC) for joint structure

Particles $\{X_t^{(p)},\alpha_t^{(p)}\}_{p=1}^P$ targeting $p(X_t\mid I_t)$. Proposal via constrained sequential fill guided by mixture component. Resample when $\mathrm{ESS}<0.4P$. Rejuvenation via Metropolis-Hastings swap moves preserving row/col sums.

### 5.4 Correlation scale diagnostic

$|\rho_{ij}|=K_i K_j / ((H-K_i)(H-K_j))^{1/2}$. At $H=50$: $|\rho|=0.087$; at $H=25$: $|\rho|=0.191$. Late-game correlations motivate Mixture-SIB + particles over first-moment alone.

---

## 6. Conservative safety estimates without over-folding

### 6.1 Strongly Rayleigh / negative dependence foundations

The remaining-tile distribution under "draw without replacement" is Strongly Rayleigh (BBL 2009), implying strong negative dependence. Used only for bounding monotone danger events.

### 6.2 Hunter bound (spanning tree correction)

For threat events $\{A_j\}_{j=1}^J$ and any spanning tree $\mathcal{T}$: $P(\bigcup_j A_j) \le \sum_j P(A_j) - \sum_{(u,v)\in\mathcal{T}} P(A_u\cap A_v)$. Maximum-weight spanning tree gives the tightest bound. Kounias (1968) bound is a member; we take the minimum computable bound.

### 6.3 Computing intersections reliably

Analytic formulas for simple events; particle estimates with Hoeffding CIs otherwise. Never use an intersection estimate unless CI half-width $<\delta_\cap$ (e.g., 0.01). Fall back to conservative Boole if CI not met.

---

## 7. Anytime Factored Belief Search (AFBS)

### 7.0 AFBS vs LuckyJ's OLSS

| | LuckyJ OLSS | HYDRA AFBS |
|---|---|---|
| When | Runtime only | Training (pondering) + Runtime |
| Sims/decision | 1000 | 100-1000 (adaptive playout cap) |
| Belief model | Simple hand sampling | Mixture-SIB + particles (constraint-correct) |
| Opponent model | Fixed N=1-4 strategies | Robust KL soft-min (worst-case aware) |
| Results fed back to network? | No | **Yes (SaF)** -- novel |
| Used during training? | No | **Yes (oracle pondering ExIt)** -- novel |
| Network size | 3 ResBlocks (tiny) | 40 SE-ResBlocks (13x larger) |
| Oracle info during training? | No | Yes (CTDE: perfect-info leaf eval) |

### 7.1 Tree structure

Node state: $(I, \mathcal{B}, \mathcal{P})$ -- info state, Mixture-SIB summary, particle set handle.

### 7.2 Beam parameters

| Mode | Beam W | Depth D | Particles P | Mixture L |
|------|-------:|--------:|------------:|----------:|
| On-turn | 64-128 | 4-6 | 128-256 | 4-8 |
| Ponder | 256-1024 | 10-14 | 1024-4096 | 8-32 |

### 7.3 Caches

Transposition table (public hash + belief signature), neural eval cache (batched GPU, LRU), Sinkhorn warm-start cache (u,v scalings), predictive ponder cache (subtrees for top-M predicted opponent actions).

### 7.4 Incremental reuse across turns

On event: lookup predicted child key; if match, shift root and keep statistics; else reuse TT/NN cache and rebuild shallow frontier.

---

## 8. Robust opponent modeling inside search

### 8.1 Opponent uncertainty set

Learned opponent policy $p(a)$. True policy $q(a)$ lies in KL ball: $\mathcal{Q}_\varepsilon(p)=\{q: D_{\mathrm{KL}}(q\|p)\le \varepsilon\}$. $\varepsilon$ calibrated from data as empirical upper quantile of observed KL, bucketed by context.

### 8.2 Robust value at opponent nodes

$V_{\text{rob}}=\min_{q\in \mathcal{Q}_\varepsilon(p)} \sum_a q(a) Q(a)$. Solution: $q_\tau(a)\propto p(a)\exp(-Q(a)/\tau)$ for $\tau$ chosen so $D_{\mathrm{KL}}(q_\tau\|p)=\varepsilon$.

**Contract.** For any opponent policy $q$ in the KL ball, AFBS's robust backup gives a lower bound on expected value against $q$.

---

## 9. Search-as-Feature (SaF)

For each legal action $a$, AFBS returns: $\Delta Q(a)$, deal-in risk estimates (Boole/Hunter/robust), epistemic terms (entropy drop), robust stress ($\tau$), uncertainty (variance, ESS).

**Logit-residual policy:** $\ell_{\text{final}}(a)=\ell_\theta(a) + \alpha_{\text{SaF}}\cdot g_\psi(f(a))\cdot m(a)$ where $m(a)\in\{0,1\}$ indicates features present.

---

## 10. ExIt + Pondering as the central training engine

### 10.1 ExIt targets

From AFBS: $\pi^{\text{ExIt}}(\cdot\mid I)=\mathrm{Softmax}(Q(I,\cdot)/\tau_{\text{ExIt}})$.

### 10.2 Pondering = label amplification

75% idle time used for: deepening current root search + precomputing searches for predicted near-future states. Every completed search yields additional labeled training examples.

### 10.3 Playout cap randomization

More compute when top-2 policy gap is small, in high-risk defense contexts, or when particle ESS is low.

---

## 11. Training pipeline

### Phase 0: BC warm start
BC on large expert corpora with suit permutation x seat rotation augmentation. Train all heads.

### Phase 1: Oracle-visible supervision (CTDE)
Self-play with full hidden state access. Supervised labels for belief likelihood, danger calibration, opponent action model. Inspired by Suphx oracle guiding (Li et al. 2020).

### Phase 2: Stable multiplayer self-play (DRDA)
DRDA (ICLR 2025) as base multiplayer learning dynamic for stability in POSGs.

### Phase 3: ExIt + AFBS + Pondering (main run)
Actors run AFBS continuously. Store visited and pondered state labels. Distill $\pi^{\text{ExIt}}$ and $V^{\text{ExIt}}$ + RL fine-tuning.

### Phase 3b: PPO accelerator (optional)
Tuned PPO competitive in IIGs per Rudolph et al. 2025. Support DRDA-primary or DRDA-then-PPO modes. Entropy coeff 0.05-0.1 (critical for IIGs).

### Population training
League: latest policy, trailing checkpoints, human-style anchors, adversarial exploiters.

---

## 12. Risk, information, and placement

### 12.1 Distributional value and CVaR
Score pdf/cdf heads. CVaR for "avoid 4th" objectives.

### 12.2 Information-Value Decomposition (IVD)
$Q^{\text{total}}(I,a)=Q^{\text{inst}}(I,a)+\eta Q^{\text{epi}}(I,a)+\xi Q^{\text{str}}(I,a)$ where instrumental = score utility, epistemic = posterior entropy decrease, strategic = concealment/leakage penalty.

### 12.3 Primal-dual risk constraints
Constraints: deal-in risk below $\kappa_{\text{deal}}$, info leakage below $\kappa_{\text{leak}}$. Dual updates: $\lambda \leftarrow [\lambda+\alpha(\hat{C}-\kappa)]_+$.

---

## 13. Validation gates

**G0:** Does Mixture-SIB + particles + AFBS produce positive decision improvement? 200K stratified states, mean $\Delta>0$, <40% negative.

**G1:** Robustness calibration. KL deviations between opponent model and held-out opponents at 95th percentile.

**G2:** Safety bound usefulness. Hunter reduces over-folding without underestimating risk beyond CI.

**G3:** SaF amortization. Shallow search + SaF must dominate shallow search alone.

---

## 14. Deployment profile

**Fast path:** Network forward + SaF adaptor. **Slow path:** Reuse pondered AFBS subtree. On-turn: 80-150ms. Call reactions: 20-50ms. Pondering: use all idle time. Agari guard always active.

---

## 15. Heritage from prior Hydra variants

**From the throughput-first plan:** Asynchronous pondering as "free" label compute, distributional value heads, oracle guiding/critic, PPO hyperparameters (entropy coeff 0.05+), double-buffered weight sync, ExIt safety valves.

**From the all-out plan:** Mixture-SIB, anytime FBS, SaF, Hunter/Kounias tightening, ExIt+Pondering centrality, SR concentration.

**OMEGA additions:** Particle posterior as first-class object, robust opponent nodes (KL-uncertainty soft-min), explicit calibration gates for every high-uncertainty link.

---

## 16. Limitations

1. **4-player general-sum has no clean exploitability target.** We use robustness + population training instead.
2. **Belief model misspecification** remains the core risk; G0 detects it early.
3. **Compute allocation**: deep AFBS is expensive; depends on caching, pondering hit rate, distillation efficiency.
4. **Strategy fusion / determinization pitfalls**: particles + robust opponent nodes mitigate but do not eliminate all pathologies.

---

## 17. References

1. Sinkhorn, Knopp. "Doubly Stochastic Matrices." *Pacific J. Math*, 1967.
2. Hunter. "Upper Bound for Union." *J. Applied Probability*, 1976.
3. Kounias. "Bounds for Union." *Annals Math Stat*, 1968.
4. Borcea, Branden, Liggett. "SR and Geometry of Polynomials." *JAMS*, 2009.
5. Bardenet, Maillard. "Concentration for Sampling Without Replacement." *Bernoulli*, 2015.
6. Anthony, Tian, Barber. "Expert Iteration." *NeurIPS*, 2017.
7. Silver et al. "Mastering Go Without Human Knowledge." *Nature* 550, 2017.
8. Wu. "Accelerating Self-Play Learning in Go (KataGo)." *arXiv 1902.10565*, 2020.
9. Li et al. "Suphx: Mastering Mahjong with Deep RL." *arXiv 2003.13590*, 2020.
10. Li et al. "Speedup Training via Reward Variance Reduction." *IEEE CoG*, 2022.
11. Farina et al. "DRDA for Multiplayer POSGs." *ICLR*, 2025.
12. Rudolph et al. "Reevaluating PG Methods in IIGs." *arXiv 2502.08938*, 2025.
13. Kalogiannis, Farina. "PG Converge in IIEFGs." *NeurIPS*, 2024.
14. Schulman et al. "Proximal Policy Optimization." *arXiv 1707.06347*, 2017.
15. Perolat et al. "Mastering Stratego (DeepNash)." *Science*, 2022.
16. Boney et al. "Learning to Play IIGs by Imitating an Oracle Planner." *IEEE Trans. Games*, 2021.
17. Abbasi-Yadkori et al. "POLITEX." *ICML*, 2019.
18. Cuturi. "Sinkhorn Distances." *NeurIPS*, 2013.
