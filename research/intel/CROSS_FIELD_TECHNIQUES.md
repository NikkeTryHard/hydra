# Cross-Field Mathematical Techniques for Game AI

**Generated**: 2026-03-03
**Context**: Math from other fields whose structure transfers direct to 4-player imperfect-information Mahjong (34 tile types, 4 copies each, ~50 hidden tiles, sequential draws without replacement).

**Ranked by transfer strength** (strongest first).

---

## 1. Glosten-Milgrom Sequential Trade Model (Market Microstructure)

**Source**: Glosten & Milgrom, "Bid, ask and transaction prices in specialist market with heterogeneously informed traders", *J. Financial Economics* (1985). [Scholar](https://www.sciencedirect.com/science/article/pii/0304405X85900443)

### The Setup

Market maker sets prices for asset with unknown true value V. Traders arrive sequentially. Some **informed** (know V), some **noise traders** (random). Market maker sees only action (buy/sell), not trader type.

### Key Math

After observing action a_t at time t, market maker updates beliefs:

```
P(V = v | a_1, ..., a_t) = P(a_t | V=v) * P(V=v | a_1,...,a_{t-1}) / P(a_t | a_1,...,a_{t-1})
```

**Adverse selection** component: bid-ask spread equals:

```
spread = E[V | buy] - E[V | sell]
       = Sum_v v * [P(buy|V=v)P(V=v) / P(buy)] - Sum_v v * [P(sell|V=v)P(V=v) / P(sell)]
```

Spread widens when information asymmetry high (more informed traders).

### Why This Is an EXACT Structural Match to Mahjong

| Market Concept | Mahjong Equivalent |
|---|---|
| Asset value V | Opponent's hand composition |
| Informed trader's action (buy/sell) | Opponent's discard choice |
| Noise trader | Random/defensive discards |
| Market maker | Our AI, inferring from observations |
| Bid-ask spread | Confidence interval on opponent hand |
| Adverse selection | "They DIDN'T discard X, so they need it" |

Mapping tight:
- Each opponent discard = **signal** about hidden hand, same as trade signaling hidden asset value.
- Probability player discards tile X given hand H is P(discard=X|H), analogous to P(buy|V=v).
- **Absence of action is informative**: not discarding drawn tile reveals need for it, same as absence of selling revealing bullish private info.
- Sequential Bayesian update identical in structure.

### Concrete Application

Define for each opponent i and tile type j:

```
mu_t(j) = P(opponent_i holds tile j | discard_history_1:t)
```

Update rule after opponent discards tile d at time t:

```
mu_t(j) = P(discard=d | holds_j) * mu_{t-1}(j) / P(discard=d)
```

"Glosten-Milgrom lambda" (probability of informed trading) maps to estimate of how strategic vs random opponent plays -- directly usable as **player modeling parameter**.

### Verdict: STRONGEST TRANSFER

Math literally same Bayesian sequential update under information asymmetry. No analogy needed -- same problem in different clothes.

---

## 2. Rao-Blackwellized Particle Filters (Robotics/SLAM)

**Source**: Doucet, de Freitas, Murphy, Russell, "Rao-Blackwellised Particle Filtering for Dynamic Bayesian Networks", *UAI* (2000). Also: Montemerlo et al., "FastSLAM", *AAAI* (2002). [Scholar](https://arxiv.org/abs/2312.09860)

### The Key Theorem

**Rao-Blackwell Theorem**: If state decomposes into (x, y) where p(y|x, observations) analytically tractable, then:

```
Var[E[f(x,y) | x]] <= Var[f(x,y)]
```

Translation: integrate out y analytically instead of sampling it -> variance drops. Better estimates with fewer particles.

### The Technique (FastSLAM style)

Decompose state space into:
1. **Sampled component** x: use particles (Monte Carlo)
2. **Analytical component** y|x: use closed-form (e.g., Kalman filter per particle)

Each particle carries own analytical posterior over y. Total cost: O(N * cost_of_analytical_update) instead of O(N * dim(y)) for full particle filtering.

### Transfer to Mahjong

Hidden Mahjong state decomposes cleanly:

```
Full hidden state = (wall_composition, opponent_1_hand, opponent_2_hand, opponent_3_hand)
```

**Decomposition**:
- **x = tile_type_counts_remaining[34]**: how many of each tile type still unseen? 34-dimensional integer vector with known constraints (each entry in {0,...,4}, sum = tiles_remaining).
- **y = assignment of remaining tiles to {wall, opp1, opp2, opp3}**: given x, this is multinomial/multivariate hypergeometric.

Key insight: **given x, distribution over y is analytically tractable**.

```
P(opp_i has k copies of tile j | x_j copies remain, hand_size_i) = Hypergeometric(k; x_j, hand_size_i, total_remaining)
```

So we can:
1. Use particles to sample plausible x vectors (tile count profiles)
2. For each particle, analytically compute opponent hand probabilities using hypergeometric distributions
3. No need to sample individual tile assignments -- huge variance reduction

### Variance Reduction Estimate

Without RB: sample space size C(~50, 13) * C(~37, 13) * C(~24, 13) ~ 10^30.
With RB: sample 34-dimensional count vectors (much smaller effective space), then integrate assignments analytically. Expected variance reduction: **orders of magnitude**.

### Verdict: VERY STRONG TRANSFER

Tile combinatorics (4 copies of 34 types) are exactly kind of structure Rao-Blackwellization exploits. Hypergeometric distribution gives analytical component. Likely single highest-impact algorithmic technique for Mahjong belief tracking.

---

## 3. Active Inference / Expected Free Energy (Neuroscience)

**Source**: Friston & Kiebel, "Predictive coding under free-energy principle", *Phil. Trans. Royal Society B* (2009). [Scholar](https://royalsocietypublishing.org/rstb/article/364/1521/1211/45615). Also: Maisto et al., "Active inference tree search in large POMDPs" (2021). [Scholar](https://arxiv.org/abs/2103.13860)

### The Core Framework

Brain minimizes **variational free energy**:

```
F = E_q[log q(s) - log p(o, s)]
  = KL[q(s) || p(s)] - E_q[log p(o|s)]
  = Complexity - Accuracy
```

where q(s) = approximate posterior over hidden states s, o = observations.

This is standard ELBO from variational inference. New piece = **Expected Free Energy (EFE)** for action selection:

```
G(pi) = E_q[ H[p(o_tau | s_tau)] ] - E_q[ D_KL[q(s_tau | o_tau, pi) || q(s_tau | pi)] ]
       = Expected Ambiguity    -    Information Gain (epistemic value)
       + E_q[ D_KL[q(o_tau | pi) || p(o_tau)] ]
       = Pragmatic Value (reward-seeking)
```

### What's Novel vs Standard RL

Standard RL maximizes expected reward. EFE minimizes expected surprise, which automatically trades off:
1. **Epistemic value**: choose actions that reduce uncertainty (information-seeking)
2. **Pragmatic value**: choose actions leading to preferred outcomes (reward-seeking)

No exploration bonus needed -- emerges from math.

### Transfer to Mahjong

Mahjong has core exploration-exploitation dilemma:
- **Exploitation**: play tiles advancing hand toward win
- **Exploration**: play tiles revealing opponent-hand info (for defense)

EFE gives principled handling:

```
G(discard_action) = Expected_ambiguity_about_opponents
                  - Information_gain_from_opponent_reactions
                  + Pragmatic_value(closer_to_winning - danger_of_deal_in)
```

Concrete example: discarding tile nobody called gives low information gain but may be safe. Discarding near opponent melds gives high information gain (they react) but is risky.

### The Tighter Bound Question

Standard VAE/ELBO uses KL[q||p]. Free energy principle literature also explores:
- **Bethe free energy**: tighter than mean-field for structured graphical models
- **Generalized free energy**: accounts for model uncertainty (epistemic + aleatory)
- **Renyi divergence bounds**: F_alpha = (1/(alpha-1)) log E_p[(q/p)^(alpha-1)], which interpolates between KL and other divergences

For Mahjong, Bethe free energy interesting because opponent hands have local structure (melds, sequences) that factored approximation can exploit.

### Verdict: STRONG TRANSFER

EFE elegantly solves exploration-exploitation tradeoff in Mahjong AI. Variational bounds directly applicable. Main risk: compute cost for EFE over large action/observation space.

---

## 4. Compressed Sensing for Sparse Hand Recovery

**Source**: Candes & Tao, "Decoding by Linear Programming", *IEEE Trans. Info. Theory* (2005). Gross et al., "Quantum state tomography via compressed sensing", *PRL* (2010). [Scholar](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.105.150401)

### The Key Theorem (Candes-Tao)

If x in R^n is s-sparse (at most s nonzero entries), and is m x n measurement matrix satisfying **Restricted Isometry Property** (RIP):

```
(1 - delta_s) ||x||^2 <= ||Ax||^2 <= (1 + delta_s) ||x||^2  for all s-sparse x
```

then from m = O(s * log(n/s)) measurements y = Ax + noise, recover x via L1 minimization:

```
minimize ||x||_1  subject to  ||Ax - y||_2 <= epsilon
```

### Transfer to Mahjong

Opponent hand vector h in {0,1,2,3,4}^34 has at most 13 nonzero entries (13 tiles from 34 types). This is **sparse**: s=13, n=34, sparsity ratio ~38%.

Each observation (discard, call, pass) gives constraint on h:
- Discard of tile j: h_j was >= 1 before discard (now reduced by 1)
- Chi/Pon call: specific tiles were in hand
- Pass on call opportunity: certain tiles were NOT in hand (or player chose not to call)

### The Problem: Measurement Model Doesn't Fit Clean CS

Measurements are not linear in standard CS sense. Observations are:
- Binary (did/didn't discard)
- Conditional on strategy (not hand contents)
- Sequential and dependent

Still, **spirit** of CS transfers: reconstruct sparse vector from fewer observations than dimensions. Practical form:

```
Instead of L1 minimization, use:
  maximize  P(h | observations)  subject to  h sparse, h consistent with game rules
```

This is **sparse Bayesian learning** (Tipping, 2001), not classical CS. Tile-game constraints (sum = hand_size, each entry <= remaining copies) add structure beyond pure sparsity.

### Where It Actually Helps

Sparsity insight helps most as **regularization** for neural belief heads:
- Add L1 penalty to opponent hand prediction heads
- Network learns sparse hand distributions
- Prevents "diffuse belief" failure mode where network assigns small probability to everything

### Verdict: MODERATE TRANSFER

Sparsity insight real, useful, but measurement model does not fit classical CS cleanly. Best use = regularization principle, not direct algorithm. Subsumes quantum tomography idea CS on density matrices).

---

## 5. Quantum State Tomography (Quantum Information) -- WEAK, INCLUDED FOR COMPLETENESS

**Source**: Gross, Liu, Flammia, Becker, Eisert, "Quantum state tomography via compressed sensing", *PRL* (2010).

### The Technique

Reconstruct density matrix rho (positive semidefinite, trace 1) from Pauli measurements:

```
minimize  ||rho||_tr  (trace norm / nuclear norm)
subject to  |Tr(P_i * rho) - y_i| <= epsilon  for all measurements i
            rho >= 0, Tr(rho) = 1
```

For rank-r states in d dimensions, need O(r * d * log^2(d)) measurements instead of d^2.

### Transfer Assessment

Structural analogy:
- Density matrix rho ~ joint probability distribution over opponent hands
- Pauli measurements ~ observations (discards, calls)
- Low rank ~ opponent hands are "structured" (going for specific yakus)

But this is mostly compressed sensing plus matrix-structure constraint. Quantum-specific parts (Pauli basis, density-matrix positivity, entanglement structure) do not map to Mahjong.

One useful insight: **nuclear norm minimization** for low-rank matrix recovery could apply if opponent strategies modeled as low-rank matrix (few latent strategy types). Still stretch.

### Verdict: WEAK TRANSFER

Useful parts reduce to compressed sensing (#4). Skip unless specifically needing low-rank matrix recovery for player modeling.

---

## Summary Table

| Rank | Technique | Source Field | Key Math | Transfer Strength | impl Effort |
|------|-----------|-------------|----------|-------------------|----------------------|
| 1 | Glosten-Milgrom | Finance | Sequential Bayesian update under info asymmetry | EXACT MATCH | Low -- direct impl |
| 2 | Rao-Blackwellized PF | Robotics/SLAM | Variance reduction via analytical marginalization | STRONG | Medium -- needs hypergeometric computation |
| 3 | Active Inference / EFE | Neuroscience | Expected free energy = epistemic + pragmatic value | STRONG | High -- requires variational inference pipeline |
| 4 | Compressed Sensing | Signal Processing | Sparse recovery from limited observations | MODERATE | Low -- mainly regularization trick |
| 5 | Quantum Tomography | Quantum Info | Nuclear norm minimization | WEAK | N/A -- subsumes into #4 |

## Recommended Priority

**Phase 1** (immediate): Implement Glosten-Milgrom style belief tracking. Same math you would write anyway, with 40 years of theory behind it.

**Phase 2** (training pipeline): Use Rao-Blackwellization in simulator belief module. Variance reduction too large to ignore for Monte Carlo methods.

**Phase 3** (research): Explore EFE as auxiliary training objective. Train value head to predict expected free energy instead of, or alongside, expected reward.

**Phase 4** (regularization): Add L1 sparsity penalties to opponent hand prediction heads.