# Cross-Field Mathematical Techniques for Game AI

**Generated**: 2026-03-03
**Context**: Techniques from unrelated fields whose math transfers directly to 4-player imperfect-information Mahjong (34 tile types, 4 copies each, ~50 hidden tiles, sequential draws without replacement).

**Ranked by transfer strength** (strongest first).

---

## 1. Glosten-Milgrom Sequential Trade Model (Market Microstructure)

**Source**: Glosten & Milgrom, "Bid, ask and transaction prices in a specialist market with heterogeneously informed traders", *J. Financial Economics* (1985). [Scholar](https://www.sciencedirect.com/science/article/pii/0304405X85900443)

### The Setup

A market maker must set prices for an asset whose true value V is unknown. Traders arrive sequentially. Some are **informed** (know V) and some are **noise traders** (random). The market maker observes only the *action* (buy/sell), not the trader's type.

### Key Math

After observing action a_t at time t, the market maker updates beliefs:

```
P(V = v | a_1, ..., a_t) = P(a_t | V=v) * P(V=v | a_1,...,a_{t-1}) / P(a_t | a_1,...,a_{t-1})
```

The **adverse selection** component: the bid-ask spread equals:

```
spread = E[V | buy] - E[V | sell]
       = Sum_v v * [P(buy|V=v)P(V=v) / P(buy)] - Sum_v v * [P(sell|V=v)P(V=v) / P(sell)]
```

The spread widens when information asymmetry is high (more informed traders).

### Why This Is an EXACT Structural Match to Mahjong

| Market Concept | Mahjong Equivalent |
|---|---|
| Asset value V | Opponent's hand composition |
| Informed trader's action (buy/sell) | Opponent's discard choice |
| Noise trader | Random/defensive discards |
| Market maker | Our AI, inferring from observations |
| Bid-ask spread | Confidence interval on opponent hand |
| Adverse selection | "They DIDN'T discard X, so they need it" |

The mapping is remarkably tight:
- Each opponent discard is a **signal** about their hidden hand, exactly like a trade is a signal about hidden asset value.
- The probability a player discards tile X given hand H is P(discard=X|H), analogous to P(buy|V=v).
- **Absence of action is informative**: a player NOT discarding a tile type they drew reveals they need it, exactly like a market where the absence of selling reveals bullish private info.
- The sequential Bayesian update is identical in structure.

### Concrete Application

Define for each opponent i and tile type j:

```
mu_t(j) = P(opponent_i holds tile j | discard_history_1:t)
```

Update rule after opponent discards tile d at time t:

```
mu_t(j) = P(discard=d | holds_j) * mu_{t-1}(j) / P(discard=d)
```

The "Glosten-Milgrom lambda" (probability of informed trading) maps to our estimate of how strategically (vs randomly) the opponent is playing -- directly usable as a **player modeling parameter**.

### Verdict: STRONGEST TRANSFER

The math is literally the same Bayesian sequential update under information asymmetry. No analogies needed -- it IS the same problem in different clothing.

---

## 2. Rao-Blackwellized Particle Filters (Robotics/SLAM)

**Source**: Doucet, de Freitas, Murphy, Russell, "Rao-Blackwellised Particle Filtering for Dynamic Bayesian Networks", *UAI* (2000). Also: Montemerlo et al., "FastSLAM", *AAAI* (2002). [Scholar](https://arxiv.org/abs/2312.09860)

### The Key Theorem

**Rao-Blackwell Theorem**: If you can decompose state into (x, y) where p(y|x, observations) is analytically tractable, then:

```
Var[E[f(x,y) | x]] <= Var[f(x,y)]
```

Translation: analytically integrating out y (instead of sampling it) ALWAYS reduces variance. You get better estimates with fewer particles.

### The Technique (FastSLAM style)

Decompose the state space into:
1. **Sampled component** x: use particles (Monte Carlo)
2. **Analytical component** y|x: use closed-form (e.g., Kalman filter per particle)

Each particle carries its own analytical posterior over y. Total cost: O(N * cost_of_analytical_update) instead of O(N * dim(y)) for full particle filtering.

### Transfer to Mahjong

The hidden state in Mahjong decomposes beautifully:

```
Full hidden state = (wall_composition, opponent_1_hand, opponent_2_hand, opponent_3_hand)
```

**Decomposition**:
- **x = tile_type_counts_remaining[34]**: How many of each tile type are still unseen? This is a 34-dimensional integer vector with known constraints (each entry in {0,...,4}, sum = tiles_remaining).
- **y = assignment of remaining tiles to {wall, opp1, opp2, opp3}**: Given x, this is a multinomial/multivariate hypergeometric distribution.

The key insight: **given x, the distribution over y is analytically tractable**.

```
P(opp_i has k copies of tile j | x_j copies remain, hand_size_i) = Hypergeometric(k; x_j, hand_size_i, total_remaining)
```

So we can:
1. Use particles to sample plausible x vectors (tile count profiles)
2. For each particle, analytically compute opponent hand probabilities using hypergeometric distributions
3. No need to sample individual tile assignments -- massive variance reduction

### Variance Reduction Estimate

Without RB: need to sample from a space of size C(~50, 13) * C(~37, 13) * C(~24, 13) ~ 10^30.
With RB: sample 34-dimensional count vectors (much smaller effective space), then integrate out assignments analytically. Expected variance reduction: **orders of magnitude**.

### Verdict: VERY STRONG TRANSFER

The combinatorial structure of tiles (4 copies of 34 types) is EXACTLY the kind of structure Rao-Blackwellization was designed to exploit. The hypergeometric distribution provides the analytical component. This is probably the single most impactful algorithmic technique for belief tracking in Mahjong.

---

## 3. Active Inference / Expected Free Energy (Neuroscience)

**Source**: Friston & Kiebel, "Predictive coding under the free-energy principle", *Phil. Trans. Royal Society B* (2009). [Scholar](https://royalsocietypublishing.org/rstb/article/364/1521/1211/45615). Also: Maisto et al., "Active inference tree search in large POMDPs" (2021). [Scholar](https://arxiv.org/abs/2103.13860)

### The Core Framework

The brain minimizes **variational free energy**:

```
F = E_q[log q(s) - log p(o, s)]
  = KL[q(s) || p(s)] - E_q[log p(o|s)]
  = Complexity - Accuracy
```

where q(s) is the approximate posterior over hidden states s, and o is observations.

This is the standard ELBO from variational inference. What's NEW is the **Expected Free Energy (EFE)** for action selection:

```
G(pi) = E_q[ H[p(o_tau | s_tau)] ] - E_q[ D_KL[q(s_tau | o_tau, pi) || q(s_tau | pi)] ]
       = Expected Ambiguity    -    Information Gain (epistemic value)
       + E_q[ D_KL[q(o_tau | pi) || p(o_tau)] ]
       = Pragmatic Value (reward-seeking)
```

### What's Novel vs Standard RL

Standard RL maximizes expected reward. EFE minimizes expected surprise, which **automatically** trades off:
1. **Epistemic value**: Choose actions that resolve uncertainty (information-seeking)
2. **Pragmatic value**: Choose actions that lead to preferred outcomes (reward-seeking)

No exploration bonus needed -- it emerges naturally from the math.

### Transfer to Mahjong

Mahjong has a fundamental exploration-exploitation dilemma:
- **Exploitation**: Play tiles that advance your hand toward winning
- **Exploration**: Play tiles that reveal information about opponent hands (for defense)

The EFE framework provides a principled way to handle this:

```
G(discard_action) = Expected_ambiguity_about_opponents
                  - Information_gain_from_opponent_reactions
                  + Pragmatic_value(closer_to_winning - danger_of_deal_in)
```

Concrete example: discarding a tile nobody has called gives low information gain but might be safe. Discarding near an opponent's melds gives high information gain (they'll react) but is risky.

### The Tighter Bound Question

Standard VAE/ELBO uses KL[q||p]. The free energy principle literature explores:
- **Bethe free energy**: tighter than mean-field for structured graphical models
- **Generalized free energy**: accounts for model uncertainty (epistemic + aleatory)
- **Renyi divergence bounds**: F_alpha = (1/(alpha-1)) log E_p[(q/p)^(alpha-1)], which interpolates between KL and other divergences

For Mahjong, the Bethe free energy is interesting because opponent hands have local structure (melds, sequences) that a factored approximation can exploit.

### Verdict: STRONG TRANSFER

The EFE framework elegantly solves the exploration-exploitation tradeoff that plagues Mahjong AI. The variational bounds are directly applicable. Main risk: computational cost of computing EFE over the large action/observation space.

---

## 4. Compressed Sensing for Sparse Hand Recovery

**Source**: Candes & Tao, "Decoding by Linear Programming", *IEEE Trans. Info. Theory* (2005). Gross et al., "Quantum state tomography via compressed sensing", *PRL* (2010). [Scholar](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.105.150401)

### The Key Theorem (Candes-Tao)

If x in R^n is s-sparse (at most s nonzero entries), and A is an m x n measurement matrix satisfying the **Restricted Isometry Property** (RIP):

```
(1 - delta_s) ||x||^2 <= ||Ax||^2 <= (1 + delta_s) ||x||^2  for all s-sparse x
```

then from m = O(s * log(n/s)) measurements y = Ax + noise, we can recover x via L1 minimization:

```
minimize ||x||_1  subject to  ||Ax - y||_2 <= epsilon
```

### Transfer to Mahjong

An opponent's hand vector h in {0,1,2,3,4}^34 has at most 13 nonzero entries (13 tiles from 34 types). This is **sparse**: s=13, n=34, sparsity ratio ~38%.

Each observation (discard, call, pass) provides a constraint on h:
- Discard of tile j: h_j was >= 1 before discard (now reduced by 1)
- Chi/Pon call: specific tiles were in hand
- Pass on a call opportunity: certain tiles were NOT in hand (or player chose not to call)

### The Problem: Measurement Model Doesn't Fit Clean CS

The measurements aren't linear in the standard CS sense. Observations are:
- Binary (did/didn't discard)
- Conditional on strategy (not just hand contents)
- Sequential and dependent

However, the **spirit** of CS transfers: we're reconstructing a sparse vector from fewer observations than dimensions. The practical approach is:

```
Instead of L1 minimization, use:
  maximize  P(h | observations)  subject to  h sparse, h consistent with game rules
```

This is really **sparse Bayesian learning** (Tipping, 2001) rather than classical CS. The tile game constraints (sum = hand_size, each entry <= remaining copies) provide additional structure beyond pure sparsity.

### Where It Actually Helps

The sparsity insight is most useful for **regularization** of neural network belief heads:
- Add L1 penalty to opponent hand prediction heads
- The network learns to predict sparse hand distributions
- Prevents the "diffuse belief" failure mode where the network assigns small probability to everything

### Verdict: MODERATE TRANSFER

The sparsity insight is real and useful, but the measurement model doesn't fit classical CS cleanly. Best applied as a regularization principle rather than a direct algorithm. Subsumes the quantum tomography idea (which is just CS applied to density matrices).

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

The structural analogy is:
- Density matrix rho ~ joint probability distribution over opponent hands
- Pauli measurements ~ observations (discards, calls)
- Low rank ~ opponent hands are "structured" (going for specific yakus)

But this is really just compressed sensing with a matrix structure constraint. The quantum-specific parts (Pauli basis, positivity of density matrices, entanglement structure) don't map to Mahjong.

The one useful insight: **nuclear norm minimization** for low-rank matrix recovery could apply if we model opponent strategies as a low-rank matrix (few latent strategy types). But this is a stretch.

### Verdict: WEAK TRANSFER

The useful parts reduce to compressed sensing (#4). Skip unless you specifically need low-rank matrix recovery for player modeling.

---

## Summary Table

| Rank | Technique | Source Field | Key Math | Transfer Strength | Implementation Effort |
|------|-----------|-------------|----------|-------------------|----------------------|
| 1 | Glosten-Milgrom | Finance | Sequential Bayesian update under info asymmetry | EXACT MATCH | Low -- direct implementation |
| 2 | Rao-Blackwellized PF | Robotics/SLAM | Variance reduction via analytical marginalization | VERY STRONG | Medium -- needs hypergeometric computation |
| 3 | Active Inference / EFE | Neuroscience | Expected free energy = epistemic + pragmatic value | STRONG | High -- requires variational inference pipeline |
| 4 | Compressed Sensing | Signal Processing | Sparse recovery from limited observations | MODERATE | Low -- mainly a regularization trick |
| 5 | Quantum Tomography | Quantum Info | Nuclear norm minimization | WEAK | N/A -- subsumes into #4 |

## Recommended Priority

**Phase 1** (immediate): Implement Glosten-Milgrom style belief tracking. It's the same math you'd write anyway but with 40 years of theoretical analysis behind it.

**Phase 2** (training pipeline): Use Rao-Blackwellization in the simulator's belief module. The variance reduction is too large to ignore for Monte Carlo approaches.

**Phase 3** (research): Explore EFE as an auxiliary training objective. Train the value head to predict expected free energy instead of (or in addition to) expected reward.

**Phase 4** (regularization): Add L1 sparsity penalties to opponent hand prediction heads.
