# Information-Geometric Play: Structured Beliefs, Factored Search, and Value Decomposition for 4-Player Tile Games

---

## 1. Introduction

We propose a framework for decision-making in multiplayer imperfect-information tile games that exploits the mathematical structure of tile conservation to achieve polynomial-time belief tracking and search. The framework introduces three novel mechanisms:

- **Structured Incremental Beliefs (SIB)**: A doubly-constrained coupling table maintained via Sinkhorn-projected neural updates, providing conservation-consistent, incrementally-updatable probability distributions over hidden tiles. Optionally extended to a small mixture of belief tables (Mixture-SIB) to capture cross-type correlations.
- **Factored Belief Search (FBS)**: A polynomial-time search algorithm that operates in the marginal belief space, exploiting the negative association property of the multivariate hypergeometric distribution to guarantee conservative bounds for conjunction-type safety queries.
- **Information-Value Decomposition (IVD)**: A three-component action-value function that explicitly separates instrumental reward, epistemic information gain (mutual information), and strategic information concealment, formulated as a constrained optimization with primal-dual convergence guarantees.

These are supported by DRDA-M (multiplayer-convergent training with population exploitation) and Predictive Pondering (opponent-turn pre-computation guided by epistemic value).

The central insight: tile games possess a conservation structure (fixed row and column marginals on the hidden tile distribution) that enables exact constraint enforcement via Sinkhorn projection, polynomial-time search via negative association, and information-theoretic action evaluation via belief entropy. No prior work exploits this structure.

---

## 2. Structured Incremental Beliefs (SIB)

### 2.1 The Belief Matrix

Hydra's belief state is not a set of particles and not an explicit posterior over joint hands. It is a single **doubly-constrained coupling table**:

$$B_t \in \mathbb{R}_{\ge 0}^{K \times M}, \qquad B_t[k,m] \approx \mathbb{E}[T_{k,m} \mid \mathcal{I}_t]$$

where $T \in \mathbb{Z}_{\ge 0}^{K \times M}$ is the (integer) hidden allocation table of unseen tiles across hidden locations and $\mathcal{I}_t$ is the public information at time $t$. Here $K=34$ tile types and $M=4$ hidden locations: the 3 opponent concealed hands and the wall.

This table satisfies hard **first-moment** conservation constraints:

$$\sum_m B_t[k,m] = r_t[k] \quad \text{(remaining copies of tile type } k\text{)}$$
$$\sum_k B_t[k,m] = s_t[m] \quad \text{(hidden tiles in location } m\text{)}$$

These are the row/column marginals of a transportation polytope (not "doubly-stochastic" unless normalized).

**Probabilistic semantics.** $B_t$ is not itself a full posterior. Hydra uses $B_t$ as a sufficient statistic for a tractable **approximate posterior family** used by search. The base instantiation is a row-factorized mean-field family:

$$q_{B_t}(T) \equiv \prod_{k=1}^{K} \mathrm{Multinomial}(T_{k,\cdot};\, r_t[k],\, p_{k,\cdot}), \qquad p_{k,m} \equiv \frac{B_t[k,m]}{r_t[k]}$$

This family enforces $\sum_m T_{k,m} = r_t[k]$ exactly and matches location sizes in expectation. It captures all first moments and exact per-type conservation, but does not fully represent cross-type correlations induced by fixed location sizes or by opponent strategy. (Hydra optionally addresses this via Mixture-SIB; see Section 2.4.)

### 2.2 Incremental Update via Neural Delta + Sinkhorn Projection

Each game event $e_t$ triggers:

$$B_{t+1} = \mathrm{Sinkhorn}(B_t \odot \exp(f_\theta(B_t, e_t, c_t)), \; r_{t+1}, \; s_{t+1})$$

where $f_\theta \in \mathbb{R}^{K \times M}$ outputs additive log-factors ("external fields") per tile-location entry, and Sinkhorn performs the KL (I-)projection back onto the constraint polytope. Cost: ~0.02ms per event.

**Proposition (Sinkhorn as KL projection).** Each SIB update is the unique solution of a convex optimization:

$$B_{t+1} = \arg\min_{X \in \mathcal{U}(r_{t+1}, s_{t+1})} D_{\mathrm{KL}}(X \| B_t \odot \exp(f_\theta(e_t)))$$

where $\mathcal{U}(r,s) = \{X \ge 0: X\mathbf{1}=r, X^\top\mathbf{1}=s\}$ is the transportation polytope. Thus SIB is an information projection: among all tables with the correct marginals, it picks the closest (in KL) to the multiplicatively-updated logits. This is the exact update rule of iterative proportional fitting / entropic optimal transport (Sinkhorn & Knopp 1967, Cuturi 2013).

### 2.3 Absent-Evidence Tracking

When player j has the opportunity to call chi on tile k but passes, SIB treats this as an explicit observation event. The neural update $f_\theta$ learns **negative likelihood factors** (a "no-call" external field): if an opponent could have called a tile but didn't, $f_\theta$ reduces logits for states where that call would have been likely. This formalizes absent-evidence inference (Hsu, Griffiths & Schreiber, Cognitive Science 2017) as a first-class Bayesian operation.

### 2.4 Mixture-SIB: Capturing Correlations Without Particles

A single first-moment table $B_t$ cannot express multimodality or "if A then B" correlations. Hydra's low-overhead mitigation is a **small mixture of SIB beliefs**.

Let $Z \in \{1,\dots,L\}$ be a latent "plan/mode" variable (e.g., push vs fold; honitsu vs tanyao). Conditioned on $Z=\ell$, Hydra maintains a belief table $B_t^{(\ell)}$ and corresponding factorized distribution $q_{B_t^{(\ell)}}$. The overall belief is the mixture:

$$q_t(T) = \sum_{\ell=1}^{L} w_t^{(\ell)} q_{B_t^{(\ell)}}(T), \qquad \sum_\ell w_t^{(\ell)} = 1$$

**Update.** For each observation $e_t$:

$$B_{t+1}^{(\ell)} = \mathrm{Sinkhorn}(B_t^{(\ell)} \odot \exp(f_{\theta,\ell}(e_t)))$$

and the mixture weights update by Bayes' rule:

$$w_{t+1}^{(\ell)} \propto w_t^{(\ell)} \cdot p_{\theta,\ell}(e_t \mid B_t^{(\ell)})$$

where $p_{\theta,\ell}$ is a learned scalar likelihood head trained from self-play trajectories.

**Why this helps.** Mixtures of simple beliefs induce correlations in the overall posterior (via latent modes) while keeping per-component update and search cost $O(LKM)$. In practice, $L \in [4,16]$ is enough to represent the dominant Mahjong multimodalities. Cost: $L \times 0.02$ms = 0.08-0.32ms per event. Negligible.

### 2.5 The NNUE Analogy

NNUE (Nasu 2018) revolutionized chess evaluation by replacing O(N) full neural forward passes with O(1) incremental feature updates. SIB applies the same principle to probabilistic beliefs: instead of recomputing the full belief distribution at each decision, maintain a running accumulator that updates per event. Over a typical game (~70 events), SIB accumulates a posterior ~100x cheaper than repeated full-network inference.

---

## 3. Factored Belief Search (FBS)

### 3.1 Key Insight: Negative Association from Tile Conservation

Let $T \in \mathbb{Z}_{\ge 0}^{K \times M}$ be the hidden allocation table. Under the "random assignment" model (uniformly random permutation of unseen tiles into hands/wall given their sizes), $T$ follows a multivariate hypergeometric distribution on contingency tables.

A random vector $X = (X_1, \dots, X_d)$ is **negatively associated (NA)** if for every pair of disjoint index sets $A, B$ and every pair of coordinatewise nondecreasing functions $f, g$:

$$\mathrm{Cov}(f(X_A), g(X_B)) \le 0$$

Sampling without replacement -- including the multivariate hypergeometric -- is NA (Joag-Dev & Proschan 1983).

Hydra uses NA for one specific purpose: **certified upper bounds for conjunction-type ("AND") minimum-count requirements** of the form "the opponent has at least these tiles." For these events, mean-field factorization is conservative.

**Important nuance**: for union-type ("OR") events, naive independence is generally **not** conservative under NA (it tends to underestimate unions). Safety-critical queries like "opponent can win on this discard" are treated as unions of conjunction patterns and bounded via union bounds (Section 3.3).

### 3.2 The Search Algorithm

FBS builds a depth-limited tree in marginal belief space:

```
FBS(B_t, depth D):
  For each action a in legal_actions:
    opponent_responses = policy_net(public_state, B_t)
    B' = SIB_update(B_t, a, opponent_responses)
    if D == 1:
      V(a) = V_phi(B', public_state')    // neural leaf evaluation
    else:
      V(a) = FBS(B', D-1)                // recurse
  return argmax_a V(a)  // or CVaR for defensive play
```

Complexity: O(A * D * K * M * S) where A=13 actions, D=4 depth, K=34 tiles, M=4 locations, S=20 Sinkhorn iterations. Total: ~1-5ms on GPU per search.

### 3.3 Theoretical Properties (proof-grade statements)

**Definition (Upper-orthant event).** For thresholds $t \in \mathbb{Z}_{\ge 0}^d$, define $\mathcal{O}(t) \equiv \{x : x_i \ge t_i \; \forall i\}$. These are conjunctions ("AND") of per-coordinate tail events.

**Theorem 1 (Conservative orthant bound under NA).** Let $X = (X_1, \dots, X_d)$ be negatively associated. Then for any $t$:

$$\Pr[X \in \mathcal{O}(t)] = \Pr\Big[\bigcap_{i=1}^d \{X_i \ge t_i\}\Big] \le \prod_{i=1}^d \Pr[X_i \ge t_i]$$

*Proof.* Let $f_i(X_i) = \mathbf{1}[X_i \ge t_i]$, which is nondecreasing. For NA vectors, the defining covariance inequality implies $\mathbb{E}[\prod_{i=1}^d f_i] \le \prod_{i=1}^d \mathbb{E}[f_i]$ by induction (apply NA with $f = \prod_{i<d} f_i$ and $g = f_d$). Hence $\Pr(\cap_i A_i) = \mathbb{E}[\prod_i f_i] \le \prod_i \Pr(A_i)$. $\square$

**Corollary (Conservative bounds for pattern-unions).** Mahjong safety queries decompose as finite unions of conjunction patterns: $U = \bigcup_{j=1}^J \mathcal{O}(t^{(j)})$. Then:

$$\Pr[U] \le \sum_{j=1}^J \Pr[\mathcal{O}(t^{(j)})] \le \sum_{j=1}^J \prod_{i=1}^d \Pr[X_i \ge t_i^{(j)}]$$

Hydra uses this decomposition for safety-critical estimation; it does **not** rely on naive independence for union events.

**Proposition 2 (Exact second-order statistics).** For a multivariate hypergeometric draw of size $n$ from a population of size $N$ with category counts $K_i$:

$$\mathrm{Cov}(X_i, X_j) = -\frac{n(N-n)}{N-1} \frac{K_i K_j}{N^2} \quad (i \ne j)$$

$$\rho(X_i, X_j) = -\sqrt{\frac{K_i K_j}{(N-K_i)(N-K_j)}}$$

In early-game Mahjong (uniform $K_i=4$, $N=136$): $\rho = -4/132 = -0.0303$.
In late-game (~50 hidden tiles, $K_i \le 4$): $|\rho| \le 4/46 \approx 0.087$.

**Computed Mahjong constants.** For a random 13-tile hand from the full 136-tile set, the exact probability of containing at least one copy of each of $m$ specific tile types:

$$\Pr[\forall k \in S, X_k \ge 1] = \sum_{j=0}^m (-1)^j \binom{m}{j} \frac{\binom{136-4j}{13}}{\binom{136}{13}}$$

For $m=2$: exact $0.10584$ vs independent product $0.11163$ (**5.5% relative overestimate**).
For $m=4$: exact $0.00882$ vs product $0.01246$ (**1.41x**, absolute difference $0.00364$).

The overestimation is small-to-moderate and always in the safe direction.

**Proposition 3 (Search correctness via simulation lemma).** Let $Q^*(a)$ be the optimal action-value in the true belief-MDP, and $\widehat{Q}_D(a)$ be FBS's depth-$D$ estimate. If $|\widehat{Q}_D(a) - Q^*(a)| \le \varepsilon$ for all $a$, then choosing $\hat{a} = \arg\max_a \widehat{Q}_D(a)$ yields a $2\varepsilon$-optimal action:

$$Q^*(\hat{a}) \ge \max_a Q^*(a) - 2\varepsilon$$

For a finite-horizon model with horizon $H$, bounded rewards $|r| \le R_{\max}$, per-step reward error $\varepsilon_r$, and per-step transition error $\varepsilon_p$ in total variation:

$$|V^\pi - \widehat{V}^\pi| \le H\varepsilon_r + H(H-1) R_{\max} \varepsilon_p$$

This is Hydra's search correctness contract: improve leaf value accuracy ($\varepsilon_r$) and belief-transition accuracy ($\varepsilon_p$), and deeper search monotonically improves decision quality.

---

## 4. Information-Value Decomposition (IVD)

### 4.1 The Decomposition

For each candidate action $a$ at belief $B_t$:

$$V(a) = V_{\text{inst}}(a) + \lambda_e \cdot V_{\text{epist}}(a) + \lambda_s \cdot V_{\text{strat}}(a)$$

**V_instrumental**: Standard RL action-value. Expected placement score from taking action $a$.

**V_epistemic**: Expected information gain about opponents. With approximate Bayesian belief update, this satisfies:

$$V_{\text{epist}}(a) = \mathbb{E}_{o \sim P(o|a)}\left[D_{\text{KL}}(B_{t+1}(a,o) \| B_t)\right] = I(S; O \mid \mathcal{I}_t, a)$$

This IS mutual information: exploration that provably reduces uncertainty about hidden state.

**V_strategic**: Information concealment. Let $W$ be an opponent-relevant hidden variable about us (e.g., our wait set) and $P_t$ be public information. The information leaked by our action is $I(W; A \mid P_t) = H(W \mid P_t) - H(W \mid P_t, A)$. If $L_0(\cdot \mid P_t) \approx P(W \mid P_t)$, then:

$$V_{\text{strat}}(a) = H(L_0(\cdot \mid P_t, a)) - H(L_0(\cdot \mid P_t))$$

is an estimator of $-I(W; A \mid P_t)$: it rewards actions that keep opponents uncertain.

### 4.2 IVD as Constrained Optimization

The decomposition is not justified by "weak dominance." It is justified because it is the Lagrangian form of a well-posed information-constrained control problem:

$$\max_\pi \mathbb{E}_\pi[\text{score}] \quad \text{s.t.} \quad I_\pi(W; A \mid P_t) \le c_s$$

and/or an exploration budget based on $I(S; O)$. The $\lambda$ weights are the dual variables of these constraints.

### 4.3 Learning Dynamics: Primal-Dual Convergence

To avoid relying on SGD to "discover" a good decomposition, Hydra uses two-timescale primal-dual learning:

- **Primal step (policy/value):** optimize $\theta$ to maximize $\mathbb{E}[V_{\text{inst}} + \lambda_e V_{\text{epist}} + \lambda_s V_{\text{strat}}]$
- **Dual step (weights):** adapt $\lambda_e, \lambda_s \ge 0$ to meet target budgets:

$$\lambda_e \leftarrow [\lambda_e + \eta(V_{\text{epist}} - c_e)]_+, \qquad \lambda_s \leftarrow [\lambda_s + \eta(c_s - V_{\text{strat}})]_+$$

Under standard stochastic approximation conditions (bounded gradients; separated step sizes), primal-dual updates converge to stationary points of the constrained objective. This is the missing "learning will reliably find it" guarantee.

### 4.4 Prior Art

Three direct precedents: (1) Factorised AIF (arXiv 2411.07362, 2024): 3-component EFE in multi-player games. (2) BAD (Foerster et al., ICML 2019): instrumental + informational value in Hanabi. (3) Strategic ULCB (Loftin et al., UAI 2021): +18.8% sample efficiency from strategic exploration.

### 4.5 SIB-IVD Synergy

IVD requires explicit beliefs: $V_{\text{epist}}$ is mutual information computed from expected belief updates; $V_{\text{strat}}$ is computed from observer entropy changes. Without SIB, IVD is impossible.

---

## 5. Training: DRDA-M

### 5.1 DRDA Self-Play (what is actually guaranteed)

Hydra is trained via self-play using DRDA-M, an adaptation of DRDA (Divergence-Regularized Dynamics; ICLR 2025) to 4-player Mahjong with belief-conditioned networks.

DRDA provides a clean theoretical contract in multiplayer games: in a single round it converges (in last iterate) to a **rest point** that induces a Nash-like distribution with bounded deviation under a hypomonotonicity condition; in a multi-round variant it can converge to an exact Nash equilibrium under stronger conditions. Hydra's contribution is not the DRDA theorem itself, but making this style of training practical for 4-player Mahjong.

R-NaD (the predecessor) was proven at neural scale via DeepNash: 1024 TPUs, Stratego ($10^{535}$ states), 7.21M training steps. DRDA extends R-NaD with last-iterate convergence -- the neural infrastructure is identical.

### 5.2 Population-Optimal Training (POT)

Initialize $\pi_{\text{base}_0}$ as a behavioral clone from 5-6M expert games. Phase 3 mix: 50% self-play, 30% frozen Human Population Model, 20% frozen anchors.

Mahjong is multiplayer and non-zero-sum, and "equilibrium" is not the only sensible notion of robustness. Recent work (Ge et al., "Securing Equal Share," 2024) highlights that in multiplayer settings, equilibrium behavior can fail to guarantee desirable robustness for each player. Hydra therefore targets robust population performance via DRDA-M self-play plus IVD: exploit when advantageous, explore when information has value, conceal when opponents can infer our plan.

---

## 6. Predictive Pondering

During opponent turns (5-15s idle), pre-compute FBS for the top-10 most likely opponent discards:

```
for k in top_10_predicted_discards(policy_net):
    B_hyp = SIB_update(B_current, "opponent discards k")
    V_pondered[k] = FBS(B_hyp, D=8)  // deeper search with idle time
```

**Compute math**: ~260s idle per round / 5ms per FBS = 52,000 searches available. Enables D=8-12 with hundreds of samples per decision. Total compute per decision: ~1,500-4,500 GPU-equivalent CPU-seconds -- same ballpark as OLSS on thousands of CPUs.

Ponder hit rate: top-10 predicted discards cover ~70-80% of actual opponent moves. Measured pondering gains in chess: +20-66 Elo (TalkChess; TCEC disables pondering as too advantageous).

IVD guides pondering: pre-search high-epistemic-value scenarios (where search matters most).

---

## 7. System Properties

### 7.1 Validation Gates

| Component | Gate | Metric | Fallback |
|-----------|------|--------|----------|
| SIB | MAE < 0.3 on held-out hands | End Phase 1 | One-shot Sinkhorn head |
| Mixture-SIB | Mixture NLL < single-table NLL | End Phase 1 | Single-table SIB |
| DRDA-M | No divergence | Mid Phase 3 | ACH training |
| FBS | +1 Naga point vs policy-only | End Phase 3 | Policy-only inference |
| IVD | $\lambda_e, \lambda_s$ converge to nonzero | End Phase 2 | Monolithic value |
| Pondering | Ponder hit rate > 40% | End Phase 3 | On-turn compute only |

### 7.2 Incremental Build

| Month | Addition | New Code | Go/No-Go |
|-------|----------|----------|----------|
| 1-2 | Base SE-ResNet + BC | 0 (existing) | BC loss converges |
| 3 | + SIB + Mixture-SIB | ~200 lines | MAE gate |
| 4 | + DRDA-M | ~200 lines | Stability gate |
| 5 | + FBS | ~200 lines | Naga gate |
| 6 | + IVD (primal-dual) | ~150 lines | Lambda gate |
| 7 | + Pondering | ~100 lines | Hit rate gate |

Total: ~850 lines novel code.

---

## 8. Limitations

1. **Belief sufficiency (first moment vs full posterior).** A single table $B_t$ captures only first moments; some decision-critical correlations can be lost. Mitigation: (i) safety-critical estimates use NA-certified conjunction bounds + union bounds (Section 3.3), and (ii) Mixture-SIB represents multimodality/correlation via $L$ belief modes (Section 2.4). This directly targets "if A then B" correlation failure cases without reverting to particle filtering.
2. **DRDA at neural scale.** Unproven for DRDA specifically; R-NaD (predecessor) scaled to 1024 TPUs via DeepNash. DRDA is a lightweight theoretical extension.
3. **FBS mean-field approximation.** Conjunction bounds are provably conservative; action-value accuracy depends on leaf-value quality and transition accuracy per the simulation lemma (Proposition 3).
4. **IVD training signal.** Primal-dual convergence is guaranteed under standard SA conditions, but whether the information-constrained optimum is meaningfully better than unconstrained is an empirical question gated by validation.
5. **Integration risk.** 6 novel mechanisms (SIB, Mixture-SIB, FBS, IVD, POT, Pondering). Mitigated by incremental build and independent fallbacks.

---

## References

1. Sinkhorn, R., Knopp, P. "Concerning nonnegative matrices and doubly stochastic matrices." Pacific J. Math. 1967.
2. Cuturi, M. "Sinkhorn Distances: Lightspeed Computation of Optimal Transport." NeurIPS 2013.
3. Joag-Dev, K., Proschan, F. "Negative Association of Random Variables with Applications." Annals of Statistics 1983.
4. Dubhashi, D., Panconesi, A. "Concentration of Measure for the Analysis of Randomized Algorithms." Cambridge, 2009.
5. Borcea, J., Branden, P., Liggett, T. "Negative Dependence and the Geometry of Polynomials." JAMS 2009.
6. Hsu, A., Griffiths, T., Schreiber, E. "When absence of evidence is evidence of absence." Cognitive Science 2017.
7. Farquhar, G., et al. "Factorised Active Inference for Strategic Multi-Agent Interactions." arXiv 2411.07362, 2024.
8. Foerster, J., et al. "Bayesian Action Decoder for Deep Multi-Agent RL." ICML 2019.
9. Loftin, R., et al. "Strategically Efficient Exploration in Competitive Multi-Agent RL." UAI 2021.
10. Farina, G., et al. "Divergence-Regularized Dynamics in Multinomial Games." ICLR 2025.
11. Perolat, J., et al. "Mastering the Game of Stratego." Science 2022.
12. Ge, X., et al. "Securing Equal Share." arXiv 2024.
13. Nasu, Y. "Efficiently Updatable Neural Network-based Evaluation Functions." 2018.
14. Frank, I., Basin, D. "Search in Games with Incomplete Information." AAAI 1998.
15. Russo, D., Van Roy, B. "Learning to Optimize via Information-Directed Sampling." NeurIPS 2018.

</content>
