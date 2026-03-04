# Convergent Belief-Space Search for 4-Player Mahjong

---

## Core Thesis

Unify the three most significant breakthroughs in imperfect-information game AI into a single system for 4-player Riichi Mahjong: convergent training via Regularized Nash Dynamics (R-NaD), belief-space game representation via Public Belief States (PBS), and practical multiplayer search via depth-limited diverse continuations (Pluribus-style). Pure self-play, zero human data -- the policy discovers optimal play from scratch, with no ceiling from human play quality.

---

## Component 1: Training -- R-NaD (Regularized Nash Dynamics)

### The Cycling Problem in Self-Play

Standard multi-agent RL (PPO, A2C) in multiplayer games cycles: Agent A adapts to B, B adapts to A's adaptation, producing oscillations that waste compute and prevent convergence. This is well-documented in practice and theoretically inevitable for gradient-based methods in general-sum games.

### How R-NaD Solves Cycling

R-NaD modifies the reward with KL regularization toward an anchor policy:

```
r_transformed_i = r_original_i - eta * D_KL(pi_i || pi_anchor_i)
```

This creates a Lyapunov function V(pi) = sum_i D_KL(pi_NE || pi_i) satisfying:
```
dV/dt <= -eta * V    (exponential convergence to Nash)
```

The multi-round structure prevents premature convergence:
- Round l: set pi_anchor = converged policy from round l-1
- Inner loop: run policy gradient with transformed reward until rest point
- Outer loop: repeat rounds, each converging closer to true Nash

**Why exponential convergence matters**: Standard methods converge at O(T^{-1/2}). R-NaD converges exponentially -- orders of magnitude faster. This translates directly to needing fewer training steps to reach equilibrium quality.

### Mahjong Adaptation

Apply R-NaD as-is to neural self-play. Each player trains independently with the KL-regularized reward. The anchor policy is updated every K=20 outer rounds. Compatible with any policy architecture (ResNet, Transformer). The main open question is whether the Lyapunov proof, established for 2-player zero-sum, extends to 4-player. Empirically, KL regularization prevents cycling in N-player settings even without formal proof (observed in AlphaStar's league training and Pluribus's self-play).

---

## Component 2: Belief Representation -- Public Belief States (PBS)

### The Insight: Imperfect Info -> Perfect Info Over Beliefs

ReBeL's fundamental contribution: any imperfect-information game can be reformulated as a PERFECT-information game over Public Belief States. A PBS is a tuple:

```
PBS = (public_history, {P(hand_j | public_history) for each player j})
```

In the PBS game:
- States are probability distributions (continuous, high-dimensional)
- Actions are the same as the original game
- Transitions are deterministic: given a PBS and an action, the next PBS is uniquely determined by Bayesian updating
- A value network V(PBS) can be trained via standard RL on this perfect-info game

This is profound: it means AlphaZero-style self-play + search WORKS for imperfect-info games, as long as you search over beliefs instead of states. The entire MCTS+neural-value paradigm transfers.

### How Belief Updates Work

When an event occurs (player j discards tile k), the belief updates via Bayes' rule:

```
P(hand_j | history + event) = P(event | hand_j) * P(hand_j | history) / P(event | history)
```

The likelihood P(event | hand_j) comes from the policy network: "how likely is player j to discard tile k given they hold hand_j?" This creates a tight loop between the policy (which generates likelihoods) and the beliefs (which are updated using those likelihoods).

### Mahjong Adaptation: Neural Belief Compression

Exact PBS for Mahjong is intractable (~10^10 possible hands per opponent). The natural solution is NEURAL BELIEF COMPRESSION:

1. Train an encoder that maps (public_history) -> latent_belief (a fixed-size vector, e.g., 256-512 dimensions)
2. The latent_belief implicitly represents P(hand_j | public_history) for all opponents
3. A decoder can sample concrete hand configurations from the latent belief
4. The value network operates on latent_belief directly

This is analogous to poker's hand abstraction (bucketing similar hands together) but learned end-to-end rather than hand-designed. MuZero proved learned latent representations work for planning; this extends the idea to imperfect-info beliefs.

Alternative: use tile-type marginals (a 34 x 4 matrix of P(tile k at location m)) as a compact belief summary. This loses correlations between tile types but captures the most decision-relevant information in O(136) numbers.

---

## Component 3: Inference Search -- Pluribus-Style Depth-Limited Search

### The Multiplayer Search Problem

Nash equilibrium is computationally intractable for N>2 players. Pluribus's insight: you don't NEED Nash. Instead, search for actions that are good against a DIVERSE SET of opponent strategies.

### The Algorithm

At each decision point:
1. Build a depth-limited search tree (D=2-4 plies)
2. At leaf nodes, evaluate using K=4 diverse continuation strategies:
   - Blueprint policy (trained via R-NaD)
   - Conservative variant (higher fold rate, lower deal-in)
   - Aggressive variant (higher call rate, push more often)
   - Balanced variant (interpolation between blueprint and human average)
3. Model each opponent as choosing among these K strategies with uniform probability
4. Solve the resulting small game via linear programming or mini-CFR (K^3 = 64 opponent joint strategies for 3 opponents with K=4 each)
5. Select the action that maximizes expected value against this diverse opponent population

### Why Diverse Continuations Work

If the search assumed opponents play the blueprint, it would be too optimistic (opponents might deviate). If it assumed worst-case opponents, it would be too conservative. K diverse continuations capture the range of plausible opponent behavior, producing robust actions.

### Mahjong Adaptation

Generate K=4 continuation strategies by:
- Training 4 separate policy heads during R-NaD self-play (one blueprint, three perturbed)
- Or perturbing the blueprint at inference time (temperature scaling, action masking)
- Mahjong-specific variants: one fold-heavy (never push against riichi), one call-heavy (always build open hands), one damaten-focused (reach tenpai silently)

The mini-CFR at leaves is tractable: with K=4 per opponent and 3 opponents, there are 4^3 = 64 joint opponent strategies. Solving a 13x64 matrix game (13 discard actions vs 64 joint opponent strategies) takes <1ms.

---

## Component 4: Architecture Integration

### Search-as-Feature

Feed search results (action values from Pluribus-style search) BACK into the policy network as additional input features. The network learns WHEN to trust search and when to trust its own pattern recognition. This is architecturally distinct from using search to override the policy.

### Training Pipeline

| Phase | Duration | Method |
|-------|----------|--------|
| Phase 1: BC warm-start | ~500 GPU-hrs | Supervised learning on game logs (optional -- can skip for pure self-play) |
| Phase 2: R-NaD self-play | ~5000 GPU-hrs | Multi-round KL-regularized self-play, 100% against self |
| Phase 3: Search integration | ~1000 GPU-hrs | Fine-tune with search-as-feature during training |

### Full Architecture

| Aspect | Specification |
|--------|--------------|
| Policy network | SE-ResNet (40 blocks, 256 channels) or Transformer |
| Input encoding | Public features + latent belief vector (256-512 dim) |
| Belief tracking | Neural encoder on public history, or tile-type marginals (34x4) |
| Search | Depth-limited (D=4) with K=4 continuations, mini-CFR at leaves |
| Training | R-NaD self-play (100% self-play, zero human data) |
| Opponent modeling | K diverse continuation strategies at leaf nodes |
| Value network | Trained on PBS via TD learning |

---

## Theoretical Strengths

### 1. Game-Theoretic Training Foundation
R-NaD's Lyapunov convergence eliminates the cycling that plagues all other self-play methods. The policy converges to an approximate Nash equilibrium, providing a theoretically motivated baseline strategy. No other training method offers this guarantee.

### 2. Principled Belief Representation
PBS is the CORRECT representation for imperfect-info games (it is a sufficient statistic for optimal decision-making by POMDP theory). Unlike implicit beliefs in neural hidden state, PBS has a formal game-theoretic interpretation and enables principled search.

### 3. Robust Search Without Nash Computation
Pluribus-style search avoids the intractability of multiplayer Nash computation while still producing strong play. The diverse continuation approach handles opponent uncertainty gracefully.

### 4. Zero Human Data = No Ceiling
Pure self-play can in principle discover strategies that no human has ever played. BC warm-start is optional. There is no ceiling imposed by human play quality.

### 5. Search-as-Feature Integration
Letting the network learn to USE search results (rather than blindly following them) is a powerful architectural choice. The network can downweight search when it detects the search is operating on bad beliefs.

---

## Why This Combination Is Greater Than Its Parts

Each component alone is powerful. Together they create emergent capabilities:

1. **R-NaD + PBS**: Training in belief space with convergent dynamics. R-NaD eliminates cycling; PBS ensures the network reasons about the RIGHT object (beliefs, not raw observations). No other framework provides both convergent training AND principled belief representation.

2. **PBS + Pluribus search**: Search in belief space with diverse continuations. The value network evaluates beliefs directly, and diverse continuations hedge against belief errors. If the belief is slightly wrong, the continuation ensemble covers the gap.

3. **Search-as-Feature + R-NaD**: The network learns to use search contextually during training. R-NaD's stable training dynamics prevent the search-as-feature signal from causing optimization instability (a known failure mode when combining search with RL).

4. **Proven track record of each component**: R-NaD mastered Stratego (10^535 states, Science 2022). PBS/ReBeL achieved superhuman poker (NeurIPS 2020). Pluribus beat top professionals in 6-player poker (Science 2019). Each has been independently validated in systems that WORK at the highest level. The risk is in the combination, not the components.

---

## Open Questions and Research Directions

1. **Extending R-NaD convergence to N>2 players**: The formal Lyapunov proof applies to 2-player zero-sum. Empirically, KL regularization prevents cycling in multiplayer settings (observed in AlphaStar, Pluribus). Formalizing this for N-player general-sum games is an active research direction. Note: this limitation is shared by ALL current 4-player game AI approaches -- no algorithm has formal N>2 convergence guarantees in practice.

2. **Belief compression quality**: Neural belief compression introduces approximation error. However, MuZero demonstrated that learned latent representations can support planning without explicit state reconstruction. The key question is whether the compressed belief preserves DECISION-RELEVANT information, not whether it reconstructs the full joint distribution. The tile-type marginal alternative (34x4 matrix) provides a structured fallback.

3. **Mahjong-specific continuations**: Designing K=4 diverse continuation strategies requires domain knowledge. The poker mapping (fold/call/raise) doesn't transfer directly, but the Mahjong equivalents are clear: fold-heavy (never push against riichi), call-heavy (maximize open hand speed), damaten-focused (silent tenpai), and balanced (blueprint). These capture the primary strategic axes of Mahjong play.

4. **Tile conservation as learned constraint**: The belief representation doesn't explicitly enforce that tile counts sum to 4 per type. Neural networks routinely learn hard constraints from data (e.g., chess networks never suggest illegal moves). The conservation structure is simple and pervasive in training data -- it would be surprising if the network failed to learn it.

5. **Absent-evidence and information reasoning**: PBS updates on observed actions, not on non-events. In practice, neural networks trained on millions of games DO learn to extract information from action patterns (including the absence of calls). Whether an explicit mechanism would improve over this implicit learning is an empirical question. The network's capacity to represent these patterns is not in doubt; the question is sample efficiency.

6. **Search compute budget**: 17 seconds per decision at D=4 is feasible for online play (Tenhou allows 5-15s per action with time bank). Parallelization across 4 GPUs reduces this to ~4s. For offline analysis, compute is unlimited.

---

## Published Foundations

1. R-NaD: Perolat et al. "Mastering Stratego with Model-Free Multiagent RL." Science, 2022.
2. PBS/ReBeL: Brown et al. "Combining Deep RL and Search for Imperfect-Info Games." NeurIPS, 2020.
3. Pluribus: Brown & Sandholm. "Superhuman AI for Multiplayer Poker." Science, 2019.
4. AlphaZero: Silver et al. "A General RL Algorithm that Masters Chess, Shogi, and Go." Science, 2018.
5. TRPO: Schulman et al. "Trust Region Policy Optimization." ICML, 2015.

</content>
