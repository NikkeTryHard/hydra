# Belief State Tracking & Opponent Hand Inference in Tile/Card Games
## Literature Survey for Hydra Mahjong AI

**Date**: 2026-03-02
**Purpose**: Find best way sample plausible opponent hands (100+ samples/sec) from Sinkhorn marginal probs while keeping tile-count constraints.

---

## Table of Contents

1. [Determinization: GIB (Bridge)](#1-determinization-gib-bridge)
2. [CFR-Based: Pluribus / Libratus / DeepStack (Poker)](#2-cfr-based-pluribus--libratus--deepstack-poker)
3. [ISMCTS: Information Set Monte Carlo Tree Search](#3-ismcts-information-set-monte-carlo-tree-search)
4. [Bayesian Hand Inference in Mahjong: Suphx & Mortal](#4-bayesian-hand-inference-in-mahjong-suphx--mortal)
5. [Constraint-Based Belief States with Belief Propagation](#5-constraint-based-belief-states-with-belief-propagation)
6. [MCMC / Gibbs Sampling for History Generation](#6-mcmc--gibbs-sampling-for-history-generation)
7. [Sinkhorn Operator for Constrained Sampling](#7-sinkhorn-operator-for-constrained-sampling)
8. [Synthesis: Recommended Approach for Hydra](#8-synthesis-recommended-approach-for-hydra)

---

## 1. Determinization: GIB (Bridge)

**Paper**: Ginsberg, "GIB: Imperfect Information in Computationally Challenging Game" (JAIR 2001)
**Link**: https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume14/ginsberg01a.pdf
**arXiv**: https://arxiv.org/abs/1106.0669

### Core Idea
GIB handles bridge (52-card imperfect-info game) via **Monte Carlo determinization**: sample N deals consistent with observations, solve each as perfect-info game, aggregate.

### Algorithm (Section 3, Algorithm 3.0.1)
1. Build set D of deals consistent with bidding + play history
2. For each move m and deal d, evaluate double-dummy result s(m, d)
3. Return move m maximizing SUM_d s(m, d)

### Sample Counts
| Context | Samples | Time |
|---------|---------|------|
| Production play | **50 deals** | ~1-2 sec total |
| Extended resource | **100 deals** | N/A |
| World Championship (1998) | **500 deals** | N/A |

### Bayesian Weighting (Section 3, p.323)
Deals not equal weight. GIB uses Bayes on play history:
- If player fails play King, GIB updates P(player holds King) via Bayes rule
- Weighted eval: SUM_d w_d * s(m, d) where w_d = deal weight

### The Strategy Fusion Problem (Section 3, p.319)
Core flaw of determinization: solver assumes different choices per sampled world, though worlds indistinguishable to player. Example: if Queen maybe with West or East, determinization says "play line when West has it, line B when East has it" but player cannot know world.

### Fix: Achievable Sets (Section 7.1, Definition 7.1.1)
Instead of maximizing average tricks, find **single plan** winning on maximal subset of worlds. Forces one committed line.

### Relevance to Hydra
- **Direct fit**: Mahjong structurally like bridge (hidden hands, known total tile counts)
- **50 samples enough** for competitive bridge; Mahjong may need more because 3 opponents not 1
- **Bayesian weighting from play history** maps to weighting from discard patterns
- **Strategy fusion real risk** if using determinized search

---

## 2. CFR-Based: Pluribus / Libratus / DeepStack (Poker)

### 2a. Pluribus (6-player No-Limit Hold'em)

**Paper**: Brown & Sandholm, "Superhuman AI for Multiplayer Poker" (Science, 2019)
**Link**: https://www.science.org/doi/10.1126/science.aay2400

#### Blueprint Strategy
- Computed **offline via self-play** with Linear CFR (MCCFR variant)
- 8 days, 12,400 CPU core-hours, ~$144 compute
- Uses card abstraction (bucket similar hands) and action abstraction (limited bet sizes)

#### Real-Time Search
Pluribus does NOT sample opponent hands naively. Instead:
1. After round 1, does **depth-limited search** with CFR on current subgame
2. Tracks **reach probabilities** over all possible opponent hands (distribution, not samples)
3. At depth limit, each remaining player picks among **k=4 continuation strategies**:
   - Blueprint strategy
   - Fold-biased blueprint
   - Call-biased blueprint
   - Raise-biased blueprint
4. Leaf values computed by **rolling out** under chosen continuation strategies

#### Key Insight
Pluribus does not "sample hands." It works with full info-set distribution. This avoids strategy fusion, but needs CFR infra.

### 2b. Libratus (Heads-Up No-Limit Hold'em)

**Paper**: Brown & Sandholm, "Superhuman AI for Heads-Up No-Limit Poker" (Science, 2018)
**Link**: https://www.science.org/doi/10.1126/science.aao1733

Three modules:
1. **Blueprint**: Coarse strategy via CFR on abstracted game
2. **Nested Subgame Solving**: Repeated real-time refinement with safety guarantees
3. **Self-Improver**: Fills missing branches overnight

#### Private Information Handling
Libratus keeps **range** (distribution over opponent possible hands) and updates from observed actions. Nested subgame solving keeps refined strategy **safe** (never worse than blueprint).

### 2c. DeepStack (Heads-Up No-Limit Hold'em)

**Paper**: Moravcik et al., "DeepStack: Expert-Level AI in Heads-Up No-Limit Poker" (Science, 2017)

Key innovations:
- **Continual Resolving**: Re-solves from scratch each decision point, keeping consistency via counterfactual values
- **Neural Network Leaf Evaluation**: At depth limit, neural net estimates values instead of full play
- **No pre-computed blueprint** needed (unlike Libratus/Pluribus)

### Relevance to Hydra
- **CFR theoretically better** but needs full CFR infra
- **4-player Mahjong harder** than 2-player poker for CFR (info sets explode)
- **Pluribus's k=4 continuation strategies** clever; maybe adapt to Mahjong
- **DeepStack's neural leaf evaluation** closest to Hydra now (value head)
- **Key takeaway**: If possible, work with distributions not samples. If must sample, weight carefully.

---

## 3. ISMCTS: Information Set Monte Carlo Tree Search

**Paper**: Cowling, Powley, Whitehouse, "Information Set Monte Carlo Tree Search" (IEEE TCIAIG, 2012)
**Link**: https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf

### How It Works
Instead of searching many determinized trees, ISMCTS builds **single tree where nodes = information sets** (not states).

1. Each iteration, sample determinization d from root info set
2. Descend info-set tree, visiting only nodes compatible with d
3. Expand, simulate (rollout), backprop as normal MCTS
4. Stats accumulate across determinizations at each info-set node

### Avoiding Strategy Fusion
Because stats collected at **information set nodes** (not state nodes), algorithm finds moves good across many hidden states, not only one determinization.

### Three Variants
| Variant | Description | Use Case |
|---------|-------------|----------|
| **SO-ISMCTS** | Single observer, root player's info sets | Simplest, good for card games |
| **SO-ISMCTS+POM** | Handles partially observable opponent moves | Games where opponent actions not fully seen |
| **MO-ISMCTS** | Separate tree per player | Most accurate opponent modeling |

### Performance Numbers
- **10,000 iterations per decision** (~1 second on 2010 hardware)
- Tested on Lord of Rings: Confrontation, Phantom games, Dou Di Zhu (Chinese card game)
- Dou Di Zhu has **avg 88 legal moves per state**; comparable to Mahjong complexity

### Relevance to Hydra
- **ISMCTS natural fit** for determinization-based Mahjong search
- Solves strategy fusion without full CFR
- **10K iterations in 1 second (2010 hardware)**; modern hardware likely 100K+
- **MO-ISMCTS with per-player trees** allows opponent modeling
- Can **combine with neural rollout policy** (replace random rollouts with Hydra policy head)

---

## 4. Bayesian Hand Inference in Mahjong: Suphx & Mortal

### 4a. Suphx (Microsoft, 2020)

**Paper**: Li et al., "Suphx: Mastering Mahjong with Deep Reinforcement Learning" (arXiv:2003.13590)
**Link**: https://arxiv.org/abs/2003.13590

#### Imperfect Information Handling
Suphx does NOT explicitly predict opponent hands. Instead uses:

1. **Oracle Guiding** (Section 3.3): Train "oracle agent" with perfect info (sees all tiles), then gradually transition to normal agent via **perfect feature dropout** (probability gamma_t decays from 1 to 0). Normal agent implicitly learns hidden-state inference.

2. **Run-time Policy Adaptation (pMCPA)** (Section 3.4): At round start:
   - Randomly sample opponent tiles and wall tiles from remaining pool
   - Run K rollouts with offline policy
   - Fine-tune policy from these K trajectories
   - Play with adapted policy

#### Feature Encoding
- Input: 34 x 838 (discard model) or 34 x 958 (other models)
- Includes: discard sequences of all players, open melds, accumulated scores, dealer info, riichi bets
- Over 10^48 hidden states per information set

#### Key Insight
Suphx's pMCPA = **determinization + policy adaptation**. It samples random opponent hands and adapts, instead of inferring exact distribution.

> **Deprecated (2026-03-03):** pMCPA removed from inference plans. Needs ~100K trajectories per round, infeasible real-time even with 90s idle. See RESEARCH_LOG.md entry 4.

### 4b. Mortal (Equim-chan, Open Source)

**Repo**: https://github.com/Equim-chan/Mortal
**Docs**: https://mortal.ekyu.moe/

Mortal uses similar approach to Suphx but open-source:
- SE-ResNet architecture (basis for Hydra architecture)
- No explicit opponent hand inference
- Implicit learning through self-play RL
- 40K hanchans/hour simulation speed

### Key Mahjong-Specific Insight
Neither Suphx nor Mortal do explicit Bayesian hand inference. Both rely on:
1. Neural nets learning implicit belief representations
2. Safety features (which tiles are "safe" from discards)
3. Oracle training to transfer perfect-info reasoning to imperfect-info play

**This gap Hydra's Sinkhorn head fills**: explicit probabilistic tile allocation.

---

## 5. Constraint-Based Belief States with Belief Propagation

**Paper**: "Modeling Uncertainty: Constraint-Based Belief States in Imperfect Information Games" (2025)
**Link**: https://arxiv.org/abs/2507.19263

### THIS IS THE MOST DIRECTLY RELEVANT PAPER FOR HYDRA.

### Core Framework
Model hidden tile/piece identities as **Constraint Satisfaction Problem (CSP)**:
- **Variables**: Each unknown piece/tile
- **Domains**: Possible identities for each variable
- **Global Cardinality Constraints (GCC)**: "Number of occurrences of each identity must remain within allowed limits" i.e. exactly 4 copies of each tile type in Mahjong

### Belief Propagation for Marginals
1. Reinterpret CSP as **factor graph** (variables = variable nodes, constraints = factor nodes)
2. Decompose each GCC into simpler count constraints (one per identity)
3. Run iterative BP message passing to estimate **marginal probabilities**
4. Result: P(tile_i has identity_j) for every unknown tile

### Sampling from Marginals
Two tested approaches:
- **Constraint-based**: Sample uniformly random, with constraint propagation enforcing consistency
- **Probabilistic**: Sample guided by BP marginals, choose variables by confidence order

### Performance
- 10 determinizations per move, 1000 simulations per action
- Tested on Mini-Stratego (5x5, hidden identities) and Goofspiel (13-card bidding)
- Key finding: "Added cost of probabilistic inference may be unjustified when constraint filtering already approximates state well"

### Direct Mapping to Hydra
| Paper Concept | Hydra Equivalent |
|--------------|-----------------|
| Variables (unknown pieces) | Unknown tiles (opponent hands + wall) |
| GCC (piece count limits) | Exactly 4 of each tile type, minus known |
| BP marginal estimates | **Sinkhorn head output** |
| Constraint-based sampling | Sequential tile allocation with GCC pruning |
| Factor graph | Tile-to-player allocation graph |

---

## 6. MCMC / Gibbs Sampling for History Generation

**Paper**: "History Filtering in Imperfect Information Games: Algorithms and Complexity" (2023)
**Link**: https://arxiv.org/abs/2311.14651

### Core Algorithm: Gibbs Sampler with RingSwap
For trick-taking card games (bridge, hearts, Oh Hell), paper gives MCMC sampler:

1. Start with any valid deal (found via max-flow in polynomial time)
2. **RingSwap**: Swap cards between suits/players while keeping correct row/column sums
3. Accept/reject via Metropolis-Hastings with reach probability ratio
4. Repeat to generate diverse samples from correct distribution

### Theoretical Guarantees
- **Aperiodic and irreducible** (Theorem 3): can reach any valid history
- **Correct stationary distribution** (Theorem 4): converges to P^pi (policy-weighted)
- **Polynomial time per transition** w.r.t. history length

### Performance on Oh Hell
| PBS Size | Histories | Samples Needed | State Transitions |
|----------|-----------|----------------|-------------------|
| Small | 192 | ~100 | ~2,000 |
| Medium | 12,960 | ~200 | ~4,000 |
| Large | 544,320 | **~400** | **~8,000** |

- **Burn-in of 20 steps** enough for accurate value estimates
- Strongly beat importance sampling for large state spaces
- Memory-efficient: no full history enumeration, only local transitions

### Adaptation to Mahjong
RingSwap maps to Mahjong:
- **"Suits" = tile types** (man, pin, sou, honors)
- **"Players" = 3 opponents + wall**
- **Constraints**: Void info (player discarded all of type X => cannot hold more)
- **Initial valid deal**: Build via max-flow on tile-to-player bipartite graph
- **Swap**: Move tile from opponent to opponent B (or wall), keeping hand sizes

**This likely most efficient sampling algorithm for Hydra use case.**

---

## 7. Sinkhorn Operator for Constrained Sampling

### 7a. Gumbel-Sinkhorn Networks (Mena et al., 2018)

**Tutorial**: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/sampling/permutations.html

Sinkhorn operator iteratively normalizes rows and columns of matrix to produce **doubly-stochastic matrix** (all rows and columns sum to 1). Continuous relaxation of permutation matrix.

### 7b. Sinkhorn Policy Gradient (Emami & Ranka, 2018)

**Paper**: https://arxiv.org/abs/1805.07010

Key details:
- Sinkhorn operator S^L(X/tau): L iterations of alternating row/column normalization
- Temperature tau controls sharpness (tau->0 gives hard permutation)
- **10 Sinkhorn iterations** optimal tradeoff (Section D, p.15)
- Rounding via **Hungarian algorithm O(n^3)** to get discrete permutation
- Exploration via k=2 row exchanges (epsilon-greedy)

### Relevance to Hydra's Sinkhorn Head
Hydra's Sinkhorn tile allocation head outputs matrix where:
- Rows = tile types (34 types)
- Columns = locations (3 opponents + wall)
- Entries = P(tile_type_i is at location_j)
- **Row sums** = number of copies of each tile type remaining (known exactly)
- **Column sums** = number of tiles each opponent/wall holds (known exactly)

Sinkhorn operator **directly enforces these constraints** during training. Question: how to **sample** from this output.

### Sampling Approaches

**Approach Hungarian Rounding**
- Round doubly-stochastic matrix to nearest integer allocation via Hungarian algorithm
- Gives one "most likely" allocation, not diverse samples
- O(n^3) per sample

**Approach B: Sequential Allocation with Gumbel Noise**
- Add Gumbel noise to log-probabilities, then apply Sinkhorn
- Each noise sample gives different valid allocation
- Differentiable if needed for training
- Fast: only Sinkhorn iterations + argmax

**Approach C: Categorical Sampling with Constraint Repair**
- Sample each tile independently from marginals
- "Repair" violations (too many tiles assigned to one player) via redistribution
- Fast but may distort distribution

**Approach D: Gibbs Sampling from Sinkhorn Marginals** (RECOMMENDED)
- Start from any valid allocation
- Use Sinkhorn marginals as proposal distribution
- RingSwap-style moves (swap tile between two locations)
- Accept/reject based on marginal probability ratio
- Guaranteed converge to correct distribution
- fast per-step, good mixing because proposals informed

---

## 8. Synthesis: Recommended Approach for Hydra

### The Problem Statement
Given:
- Sinkhorn head outputs: M[tile_type][location] = P(tile at location) for all unknown tiles
- Row constraints: sum_j M[i][j] = remaining_count[tile_type_i] (known exactly)
- Column constraints: sum_i M[i][j] = hand_size[player_j] (known exactly)
- Need: 100+ valid samples/second of complete opponent hands

### Recommended: Gibbs Sampler with Sinkhorn-Informed Proposals

```
Algorithm: Sinkhorn-Gibbs Tile Sampler
========================================
Input: M (Sinkhorn marginals), constraints (tile counts, hand sizes)
Output: Stream of valid tile allocations

1. INITIALIZE: Construct initial valid allocation via greedy sequential
   assignment (assign tiles to locations in order of decreasing marginal
   probability, respecting constraints)

2. SAMPLE LOOP (for each desired sample):
   a. Pick random tile t from unknown tiles
   b. Pick random swap partner: another tile t' at a different location
   c. Propose swap: move t to t's location, t' to t's location
   d. If swap maintains constraints (hand sizes, tile counts):
      - Accept with probability min(1, M[t'][new_loc] * M[t][new_loc] / 
                                        M[t][old_loc] * M[t'][old_loc])
   e. Repeat steps a-d for B burn-in steps (B ~ 20-50)
   f. Record current allocation as a sample

3. Return collected samples
```

### Why This Approach

| Criterion | Score | Reason |
|-----------|-------|--------|
| Speed | Excellent | O(1) per swap step, no matrix ops |
| Correctness | Proven | Gibbs sampler converges to target distribution |
| Constraint satisfaction | Guaranteed | Only valid swaps proposed |
| Uses Sinkhorn output | Yes | Marginals guide acceptance probability |
| Diversity | Good | MCMC explores space, not only MAP |
| Burn-in | Fast | ~20 steps enough (per history filtering paper) |
| impl complexity | Low | Only swaps + probability ratios |

### Performance Estimate
- ~50 unknown tiles (typical mid-game Mahjong)
- ~20 swap steps per sample
- Each swap: 2 random indices + 1 probability ratio + 1 comparison
- **Estimate: 10,000+ samples/second on single CPU core**
- With 100 samples for search: **100+ search evaluations per second**

### Alternative: Vectorized Gumbel-Sinkhorn Sampling
For GPU acceleration during training:
1. Generate batch of B Gumbel noise matrices
2. Add to log(M) element-wise
3. Apply Sinkhorn operator (10 iterations each)
4. Round to integer allocation via argmax per tile
5. Entire batch computed in parallel on GPU

This gives B valid samples in one forward pass. Good for training-time sampling.

### Comparison to Existing Approaches

| Approach | Speed | Accuracy | Complexity | Used By |
|----------|-------|----------|------------|---------|
| Uniform sampling + rejection | Poor | Poor | Low | Naive baseline |
| Bayesian weighting (GIB) | Good | Good | Medium | GIB (50-500 samples) |
| Full CFR on info sets | N/A | Best | High | Pluribus/Libratus |
| pMCPA random sampling (Suphx) | Good | Moderate | Low | Suphx |
| CSP + BP (constraint paper) | Good | Good | Medium | Mini-Stratego |
| **Gibbs + Sinkhorn marginals** | **Best** | **Good** | **Low** | **Proposed for Hydra** |
| ISMCTS (per-iteration sampling) | Good | Good | Medium | Card games |

### Implementation Priorities for Hydra
1. **Phase 1 (Now)**: Implement Sinkhorn head for marginals (already planned)
2. **Phase 2**: Implement Gibbs sampler for CPU-side sampling during search
3. **Phase 3**: Implement Gumbel-Sinkhorn batch sampling for GPU training
4. **Phase 4**: Integrate with ISMCTS-style search (or Pluribus-style depth-limited search)

---

## Paper Reference Table

| # | Paper | Year | Topic | Key Contribution | Link |
|---|-------|------|-------|-----------------|------|
| 1 | Ginsberg, "GIB" | 2001 | Bridge AI | Monte Carlo determinization, achievable sets, Bayesian weighting. 50-500 samples. | [JAIR](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume14/ginsberg01a.pdf) |
| 2 | Brown & Sandholm, "Pluribus" | 2019 | 6-player Poker | Blueprint + depth-limited search, k=4 continuation strategies, reach probabilities | [Science](https://www.science.org/doi/10.1126/science.aay2400) |
| 3 | Brown & Sandholm, "Libratus" | 2018 | Heads-up Poker | Nested safe subgame solving, blueprint + self-improver | [Science](https://www.science.org/doi/10.1126/science.aao1733) |
| 4 | Moravcik et al., "DeepStack" | 2017 | Heads-up Poker | Continual resolving, neural leaf evaluation, no pre-computed blueprint | [Science](https://doi.org/10.1126/science.aam6960) |
| 5 | Li et al., "Suphx" | 2020 | Riichi Mahjong | Oracle guiding, pMCPA run-time adaptation, random tile sampling | [arXiv:2003.13590](https://arxiv.org/abs/2003.13590) |
| 6 | Cowling et al., "ISMCTS" | 2012 | Card Games | Info set MCTS, avoids strategy fusion, 3 variants | [IEEE](https://ieeexplore.ieee.org/document/6203567) |
| 7 | (Anonymous), "Constraint-Based Belief States" | 2025 | Stratego/Goofspiel | CSP + GCC + Belief Propagation for marginals, constrained sampling | [arXiv:2507.19263](https://arxiv.org/abs/2507.19263) |
| 8 | (Authors), "History Filtering" | 2023 | Trick-taking Games | MCMC Gibbs sampler, RingSwap, 400 samples for 544K histories, 20-step burn-in | [arXiv:2311.14651](https://arxiv.org/abs/2311.14651) |
| 9 | Emami & Ranka, "Sinkhorn Policy Gradient" | 2018 | Combinatorial Opt | Sinkhorn layer, Hungarian rounding, epsilon-greedy exploration | [arXiv:1805.07010](https://arxiv.org/abs/1805.07010) |
| 10 | Billings et al., "Selective Sampling in Poker" | 1999 | Poker (Loki) | Weighted sampling from opponent hand distributions, 500 trials | [UAlberta](https://poker.cs.ualberta.ca/publications/AAAISS99.pdf) |

---

## Glossary

- **Determinization**: Sample specific hidden state, solve as perfect-info
- **Strategy Fusion**: Determinization bug; assumes different choices per world
- **Information Set**: Set of states indistinguishable to player
- **CFR**: Counterfactual Regret Minimization; iterative algorithm converging to Nash equilibrium
- **GCC**: Global Cardinality Constraint; enforces count limits per value
- **BP**: Belief Propagation; message passing on factor graphs for marginals
- **Sinkhorn Operator**: Iterative row/column normalization producing doubly-stochastic matrix
- **RingSwap**: Move cards/tiles between players while keeping count constraints
- **pMCPA**: Parametric Monte-Carlo Policy Adaptation (Suphx run-time search)
- **ISMCTS**: Information Set Monte Carlo Tree Search