# HYDRA: Information-Geometric Play for 4-Player Riichi Mahjong

**A proposal for a high-ceiling Mahjong AI built around an ExIt + Pondering virtuous cycle, with structured belief inference (SIB / Mixture-SIB), anytime factored-belief search (FBS), search-as-feature amortization, and primal-dual information-value decomposition (IVD).**

---

## 0. Abstract

4-player Riichi Mahjong is a large, general-sum, imperfect-information game with strong structural constraints (tile conservation) and decision-critical hidden correlations (hands/wall share a finite pool; draws without replacement). Existing top systems either (i) lean heavily on inference-time subgame solving/search, or (ii) lean on pure evaluation with massive training and little/no search. HYDRA proposes a third route:

**Central engine: Expert Iteration (ExIt) + Pondering.**
During training, deep Factored Belief Search (FBS) continuously produces improved policy targets and value targets. Pondering turns the 75% "idle time" of each self-play agent (the other 3 players' turns) into search compute and, crucially, into extra ExIt-labeled training data (not just inference-time strength).

To make this rigorous and practical for Mahjong, HYDRA combines:
- **SIB / Mixture-SIB**: differentiable, constraint-correct belief inference via Sinkhorn (optimal-transport style projection), extended to a mixture for multimodal/correlated hidden states.
- **Anytime FBS**: top-k pruned, cache-heavy, interruptible search that reuses work across turns and can exploit opponent think time under Tenhou 5+10.
- **Search-as-Feature**: FBS outputs feed into the policy network as additional inputs; deep search is amortized into fast inference.
- **Conservative safety math**: negative dependence (NA / Strongly Rayleigh) yields valid tail and conjunction bounds; Hunter/Kounias-style pairwise corrections reduce over-folding from naive union bounds.
- **Primal-dual IVD**: decomposes action-value into instrumental + epistemic + strategic (concealment) components with explicit constraints and a convergent multiplier update.

---

## 1. The 16-Technique "All-Out" Plan (Integrated)

1. **40-block SE-ResNet** backbone + **nested bottleneck blocks**
2. **85x34 encoding** with **23 explicit safety channels**
3. **7 auxiliary heads** (policy, value, GRP, tenpai, danger, opp-next, score)
4. **24x augmentation** (6 suit permutations x 4 seat rotations)
5. **Agari guard** (never miss legal win)
6. Phase 1: **behavioral cloning** warm-start
7. Phase 2: **oracle distillation**
8. **ExIt training signal** (search-improved targets)
9. **SIB + Mixture-SIB** belief inference
10. **FBS** (factored-belief search) as the policy-improvement operator
11. **Pondering** (idle time -> search + extra ExIt labels)
12. **DRDA-M** (divergence-regularized multiplayer dynamics) or **ACH fallback**
13. **POT** (population/human-style opponents mixed into training; 30% HPM mix)
14. **IVD (primal-dual)** (instrumental + epistemic + concealment)
15. **Hard position mining** (focus training on decision-critical states)
16. **Progressive model scaling** (small->large curriculum)

---

## 2. Game Model and Notation

- Tile types $k \in \{1,\dots,34\}$, each with multiplicity 4.
- Hidden locations $z \in \{1,2,3,W\}$: three opponent concealed hands + wall.
- $r_t(k) = 4 - \text{visible}_t(k)$: remaining count of tile type $k$.
- $s_t(z)$: size of hidden location $z$.
- Hidden allocation: $X_t \in \mathbb{Z}_{\ge 0}^{34 \times 4}$ with constraints $\sum_z X_t(k,z) = r_t(k)$ and $\sum_k X_t(k,z) = s_t(z)$.

---

## 3. Network: Encoding, Backbone, Heads

### 3.1 Input: 85 x 34
- **Base public encoding (62 channels)**: hand, discards, melds, riichi, dora, round meta, shanten.
- **Safety planes (23 channels)**: genbutsu (9), suji (9), kabe (2), tenpai hints (3).

### 3.2 Backbone: 40-block SE-ResNet with nested bottlenecks
- Stem: Conv1d(85->256, k=3, pad=1). 40 residual blocks with nested bottleneck (256->128->256) + SE gate + GroupNorm. Output: $\mathbf{h}_t \in \mathbb{R}^{256 \times 34}$ + pooled $\bar{\mathbf{h}}_t \in \mathbb{R}^{256}$.
- Latency target: <15ms on consumer GPU.

### 3.3 Heads: 7 auxiliary + belief outputs
1. **Policy** (46 actions) 2. **Value** (placement points) 3. **GRP** (24-way rank distribution) 4. **Tenpai** (3 sigmoids) 5. **Danger** (3x34) 6. **Opp-next** (3x34 softmax) 7. **Score** (expected swing)
- Plus: Sinkhorn belief output $B_t(k,z)$ and Mixture-SIB components $\{B_t^{(\ell)}, w_t^{(\ell)}\}_{\ell=1}^L$.

---

## 4. Structured Beliefs: SIB and Mixture-SIB

### 4.1 SIB as KL projection
Let $F_\theta(I_t) \in \mathbb{R}^{34 \times 4}$ be a learned external field. Define $K_\theta(k,z) = \exp(F_\theta(k,z))$.

$$\mathrm{SIB}(K_\theta; r_t, s_t) := \arg\min_{B \in \mathcal{U}(r_t, s_t)} D_{\mathrm{KL}}(B \| K_\theta)$$

where $\mathcal{U}(r,s) = \{B \ge 0 : B\mathbf{1}=r, B^\top\mathbf{1}=s\}$.

**Proposition 4.1.** Unique minimizer exists for $K_\theta > 0$.
**Proposition 4.2.** Minimizer has form $B^* = \mathrm{diag}(u) K_\theta \mathrm{diag}(v)$ (Sinkhorn-Knopp).

Algorithm: $u^{(\ell+1)} = r_t \oslash (K_\theta v^{(\ell)})$, $v^{(\ell+1)} = s_t \oslash (K_\theta^\top u^{(\ell+1)})$. Log-domain stabilized. 10-30 iterations. Residuals $\varepsilon_{\text{row}}, \varepsilon_{\text{col}}$ are explicit measurables.

### 4.2 Mixture-SIB: multimodality for correlations
$L$ belief modes with mixture:

$$q_t(X) = \sum_{\ell=1}^L w_t^{(\ell)} q_t^{(\ell)}(X), \qquad B_t^{(\ell)} = \mathrm{SIB}(\exp(F_\theta^{(\ell)}); r_t, s_t)$$

Weights update by Bayes: $w_{t+1}^{(\ell)} \propto w_t^{(\ell)} \cdot p_{\theta,\ell}(e_t \mid B_t^{(\ell)})$.

Each mode = distinct hand hypothesis cluster (honitsu, toitoi, damaten, etc.). Cost: $O(LKM)$ per update. $L \in [4,16]$.

Anti-collapse: entropy regularization on $w_t$ + diversity penalty + load balancing.

### 4.3 Sampling: Constrained Marginal-Proportional Sampling (CMPS)
For each opponent $i$, fill $s_t(i)$ tiles sequentially: $\pi(k) \propto B^{(\ell)}(k,i) \cdot \mathbf{1}[r_{\text{rem}}(k)>0]$. Guaranteed valid by dynamic masking. Fallback: importance sampling or MCMC with contract $|\mathbb{E}_{\tilde q}[V] - \mathbb{E}_q[V]| \le 2\epsilon V_{\max}$.

---

## 5. Conservative Probability Foundations

### 5.1 NA of multivariate hypergeometric
**Theorem 5.1.** Counts from multivariate hypergeometric are negatively associated (NA). For disjoint index sets $A,B$ and nondecreasing $f,g$: $\mathrm{Cov}(f(X_A), g(X_B)) \le 0$.

### 5.2 Orthant/conjunction bound
**Theorem 5.2.** For NA variables and thresholds $t$: $\Pr[\bigcap_j \{X_j \ge t_j\}] \le \prod_j \Pr[X_j \ge t_j]$.

*Proof:* Induction on NA indicator functions.

### 5.3 Exact second-order statistics
$\mathrm{Cov}(X_i, X_j) = -\frac{n(H-n)}{H-1}\frac{K_i K_j}{H^2}$. Correlation $\rho = -\sqrt{K_i K_j / ((H-K_i)(H-K_j))}$.

| Hidden tiles $H$ | $\|\rho\|$ (worst case $K_i=K_j=4$) |
|---|---|
| 70 | 0.061 |
| 50 | 0.087 |
| 25 | 0.191 |
| 20 | 0.250 |

Late-game correlations grow but remain bounded; Mixture-SIB targets exactly these cases.

---

## 6. Factored Belief Search (FBS)

### 6.1 What FBS computes
Given $I_t$, belief $q_t$, and network $(\pi_\theta, V_\theta)$, FBS returns: action-values $Q^{\mathrm{FBS}}(a)$, uncertainty estimates, safety estimates (deal-in probability), epistemic values, strategic values. These drive ExIt targets, search-as-feature inputs, and inference action selection.

### 6.2 Safety: Hunter/Kounias union bounds
Threat events $A_1, \dots, A_J$ (opponent wait patterns). Naive union: $\Pr[\bigcup A_j] \le \sum \Pr[A_j]$ (too pessimistic).

**Theorem 6.1 (Hunter).** For any spanning tree $\mathcal{T}$ on $\{1,\dots,J\}$:

$$\Pr\Big[\bigcup_j A_j\Big] \le \sum_j \Pr[A_j] - \sum_{(u,v) \in \mathcal{T}} \Pr[A_u \cap A_v]$$

Tightest with maximum spanning tree (edge weights = $\Pr[A_u \cap A_v]$, MST via Kruskal in $O(J^2 \log J)$). Intersection probabilities computed analytically for orthant-type threats or via bounded-error MC with Hoeffding CI.

### 6.3 Anytime FBS: top-k pruning + caches + incremental updates

**Pruning**: self top-$k_{\text{self}}=8$ actions, opponent top-$k_{\text{opp}}=6$, chance top-$k_{\text{draw}}=6$.

**Priority (EVoC)**: $\mathrm{prio}(I) = P_{\text{reach}}(I) \cdot \widehat{\mathrm{Var}}[V(I)] \cdot \Delta_{\max}(I)$. High reach + high uncertainty + ambiguous decision = expand first.

**4 caches**: (1) Transposition table keyed by public hash. (2) NN eval cache (LRU). (3) Belief cache keyed by (hash, mode id), warm-started Sinkhorn. (4) Ponder cache keyed by predicted next-turn hashes, validity-tagged.

**Incremental tree reuse**: if opponent action matches a predicted child, set root = child (keep stats). Otherwise rebuild root, keep TT + NN cache.

**Runtime budgets (Tenhou 5+10, pessimistic 1-2s opponent think)**:
- On-turn: 120ms hard budget. $W=64$ beam, $D=4$, $S=64$ belief samples.
- Pondering: 3-6s idle per opponent cycle. $W=256$, $D=8$-$12$, $S=512$ samples.

### 6.4 Search-as-Feature: amortize search into the policy

For each action $a$, FBS produces features: $Q_{\text{inst}}(a)$, $p_{\text{deal}}(a)$, $Q_{\text{epi}}(a)$, $Q_{\text{str}}(a)$, $\widehat{\mathrm{Var}}(Q(a))$, $\Delta Q(a)$, Hunter-bound danger, CVaR danger.

**Logit-residual architecture**: final logits = $\ell_\theta(a) + \alpha_{\text{sf}} \cdot g_\psi(f(a))$ where $g_\psi$ is a small MLP on search features. Preserves base behavior when features absent. Even shallow on-turn FBS produces useful features.

---

## 7. ExIt + Pondering Virtuous Cycle (Central Engine)

### 7.1 ExIt formalized
Let $\mathcal{S}_\beta$ be FBS with compute budget $\beta$. ExIt target: $\pi^{\mathrm{FBS}}_\beta(\cdot \mid I) = \mathrm{Softmax}(Q^{\mathrm{FBS}}_\beta(I, \cdot) / \tau)$.

Iterate: (1) policy evaluation via self-play, (2) policy improvement via FBS targets, (3) distillation: train $\pi_\theta \approx \pi^{\mathrm{FBS}}_\beta$.

### 7.2 Approximate policy iteration contract
If each iteration produces $\varepsilon$-approximate greedy policy: $Q^\pi(I, \pi'(I)) \ge \max_a Q^\pi(I,a) - \varepsilon$, then performance loss bounded: $V^{\pi^*}(I_0) - V^{\pi'}(I_0) \le O(H\varepsilon)$. HYDRA's measurable objective: reduce $\varepsilon$ by increasing search budget $\beta$ and improving value accuracy.

### 7.3 Pondering: two uses
**Inference-time**: expand likely next states during opponent turns. Reuse deep subtree on prediction hit.

**Training-time (label amplification)**: every pondered state with computed $\pi^{\mathrm{FBS}}_\beta$ is a valid ExIt training example. Pondering increases labeled states per environment step: $N_{\text{lab}} / N_{\text{env}} \gg 1$. This is the central mechanism by which pondering accelerates training.

### 7.4 ExIt loss
$$\mathcal{L} = \lambda_\pi \mathrm{CE}(\pi_\theta, \pi^{\mathrm{ExIt}}) + \lambda_V \|V_\theta - V^{\mathrm{ExIt}}\|_2^2 + \sum_{h \in \text{aux}} \lambda_h \mathcal{L}_h + \lambda_{\text{reg}}\|\theta\|^2$$

---

## 8. IVD: Primal-Dual Information-Value Decomposition

### 8.1 Three-component value
$$Q^{\text{total}}(I,a) = Q^{\text{inst}}(I,a) + \eta Q^{\text{epi}}(I,a) + \xi Q^{\text{str}}(I,a)$$

### 8.2 Epistemic = mutual information
$\mathrm{IG}(I,a) = H(b_t) - \mathbb{E}_{o_{t+1}}[H(b_{t+1})]$. Approximated via Mixture-SIB entropy + belief updates under sampled transitions.

### 8.3 Strategic = concealment
$Q^{\text{str}}(I,a) = -\mathbb{E}[C_{\text{leak}} \mid I,a]$ where $C_{\text{leak}} = I(\text{our hand}; \text{public signals} \mid I,a)$, approximated via RSA-style listener model or opp-next predictability.

### 8.4 Primal-dual constrained optimization
Constraints: $\mathbb{E}[C_{\text{deal}}] \le \kappa_{\text{deal}}$, $\mathbb{E}[C_{\text{leak}}] \le \kappa_{\text{leak}}$.

Lagrangian: $\mathcal{J}(\pi; \lambda) = \mathbb{E}[R^{\text{inst}}] - \lambda_{\text{deal}}(\mathbb{E}[C_{\text{deal}}] - \kappa_{\text{deal}}) - \lambda_{\text{leak}}(\mathbb{E}[C_{\text{leak}}] - \kappa_{\text{leak}}) + \eta\mathbb{E}[R^{\text{epi}}]$

Multiplier update: $\lambda_{\text{deal}} \leftarrow [\lambda_{\text{deal}} + \alpha(\hat{C}_{\text{deal}} - \kappa_{\text{deal}})]_+$. PID controller for stability in nonconvex RL.

---

## 9. Training Plan

### 9.1 Phase 1: BC warm-start
5-6M expert games, 24x augmentation, all 7 heads + SIB supervision.

### 9.2 Phase 2: Oracle distillation
Perfect-info teacher -> blind student. $\mathcal{L} = \mathcal{L}_{\text{RL}} + \lambda_{\text{KL}} D_{\text{KL}}^\tau(\pi_S \| \pi_T) + \lambda_{\text{anchor}} D_{\text{KL}}(\pi_S \| \pi_{\text{BC}})$. Provides dense supervision for SIB/Mixture-SIB calibration.

### 9.3 Phase 3: League self-play with ExIt + Pondering
Each actor runs anytime FBS + pondering. Stores $(o_t, f_t, \pi^{\mathrm{ExIt}}_t, V^{\mathrm{ExIt}}_t)$ for visited AND pondered states. Train by ExIt distillation + RL fine-tuning.

### 9.4 DRDA-M (or ACH fallback)
DRDA-M for convergent multiplayer dynamics. Fallback to ACH if stability metrics degrade.

### 9.5 POT: 70% self-play / 30% human-like opponents
Population exploitation as first-class objective.

### 9.6 Hard position mining
Oversample: high policy-FBS disagreement, high regret, riichi defense, furiten edge cases.

### 9.7 Progressive scaling
Train 20-block first, expand to 40-block with weight transfer + lower LR.

### 9.8 Agari guard
Rule-based: always take legal ron/tsumo. Zero-cost catastrophic error elimination.

---

## 10. Empirical Validation Gates

### Gate G0 (highest uncertainty): Does FBS + Mixture-SIB improve actions?
1. Train student through Phase 2, freeze as baseline $\pi_0$
2. Collect 200K stratified states (early/mid/late, offense/defense, single/multi-threat)
3. For each: run deep FBS to produce $\pi^{\mathrm{FBS}}$, compare against $\pi_0$
4. Use oracle teacher as evaluator: $\Delta(I) = \mathbb{E}_{a \sim \pi^{\mathrm{FBS}}}[Q_{\text{oracle}}(I,a)] - \mathbb{E}_{a \sim \pi_0}[Q_{\text{oracle}}(I,a)]$

**Go/no-go**: Mean $\Delta > 0$ (meaningful threshold), holds in defense buckets, <40% states with negative $\Delta$.

**If G0 fails**: fix beliefs/values before proceeding to ExIt. Do NOT amplify errors.

### Secondary gates
- G1: Sinkhorn residuals small and stable
- G2: Hunter-bound danger vs MC UCB: no systematic over/under estimation
- G3: Pondering label amplification $N_{\text{lab}}/N_{\text{env}}$ exceeds target

---

## 11. Limitations

1. **Belief misspecification**: Mixture-SIB may not capture real opponent strategy correlations. Gate G0 catches this early.
2. **Strategy fusion**: determinization-like methods risk fusion. Mitigated by belief sampling + search-as-feature (policy learns to hedge) + oracle diagnostics.
3. **General-sum non-convergence**: DRDA-M/ACH are stability methods, not equilibrium proofs. HYDRA optimizes strength, not exploitability bounds.
4. **Compute allocation**: deep ExIt targets are expensive. Pondering + batching + caching + progressive scaling keep it feasible.

---

## 12. References

1. Joag-Dev, Proschan. "Negative Association." Annals of Statistics, 1983.
2. Borcea, Branden, Liggett. "Negative Dependence and Geometry of Polynomials." JAMS, 2009.
3. Cuturi. "Sinkhorn Distances." NeurIPS, 2013.
4. Sinkhorn, Knopp. "Doubly Stochastic Matrices." Pacific J. Math, 1967.
5. Hunter. "An Upper Bound for Probability of a Union." JAP, 1976.
6. Kounias. "Bounds for Probability of a Union." Annals Math Stat, 1968.
7. Farquhar et al. "Factorised Active Inference for Strategic Multi-Agent." arXiv 2411.07362, 2024.
8. Foerster et al. "Bayesian Action Decoder." ICML, 2019.
9. Loftin et al. "Strategic Exploration in Competitive MARL." UAI, 2021.
10. Farina et al. "Divergence-Regularized Dynamics." ICLR, 2025.
11. Perolat et al. "Mastering Stratego." Science, 2022.
12. Ge et al. "Securing Equal Share." arXiv, 2024.
13. Nasu. "NNUE." 2018.
14. Anthony et al. "Expert Iteration." NeurIPS, 2017.
15. Hsu, Griffiths, Schreiber. "Absence of Evidence." Cognitive Science, 2017.
16. Dubhashi, Panconesi. "Concentration of Measure." Cambridge, 2009.

---

## Appendix A: Proof of Theorem 5.2 (orthant bound under NA)
Let $E_j = \{Y_j \ge t_j\}$ be increasing events. For NA variables: $\Pr(\bigcap_j E_j) \le \Pr(\bigcap_{j<d} E_j) \Pr(E_d) \le \dots \le \prod_j \Pr(E_j)$ by induction on NA indicator functions.

## Appendix B: Proof of Theorem 6.1 (Hunter bound)
Inclusion-exclusion gives exact union probability. A spanning tree $\mathcal{T}$ selects $J-1$ pairwise intersections such that higher-order terms are bounded. Maximum spanning tree maximizes the subtraction, minimizing the bound.

## Appendix C: Simulation lemma for belief sampling
If FBS uses sampler $\tilde{q}$ with $d_{\mathrm{TV}}(\tilde{q}, q) \le \epsilon$, then for bounded rollout return $|G| \le G_{\max}$: $|\mathbb{E}_{\tilde{q}}[G] - \mathbb{E}_q[G]| \le 2\epsilon G_{\max}$.

</content>
