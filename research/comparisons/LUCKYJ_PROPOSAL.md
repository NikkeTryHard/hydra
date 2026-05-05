# Proposal A: Game-Theoretic Self-Play with Subgame Search for 4-Player Mahjong

**Team**: Tencent AI Platform Dept. (5+ researchers)
**Compute**: Est. 10,000-50,000 GPU-hours

---

## Core Thesis

Train policy net via pure self-play with game-theoretic RL having Nash-style convergence, then add imperfect-information subgame solving at inference. Zero human data.

---

## Component 1: Training -- ACH (Actor-Critic Hedge)

**Paper**: ICLR 2022 -- "Actor-Critic Policy Optimization in a Large-Scale Imperfect-Information Game"

Merges deep RL (actor-critic) + Hedge (weighted CFR) for policy optimization in imperfect-information games.

### Algorithm
- Keeps regret-weighted policy mixture (Hedge/multiplicative weights)
- Actor-critic gives advantage estimates from self-play trajectories
- Policy update blends RL gradient + regret-minimization update
- Pure self-play: train from scratch, zero human data

### Theoretical Properties
- Nash convergence at `O(T^{-1/2})` in 2-player zero-sum games
- Lower variance than prior sampled regret methods (Monte Carlo CFR)
- No convergence guarantee in 4-player; empirical only

### Training Paradigm
- 100% self-play (no behavioral cloning, no human data, no oracle)
- League-style training with frozen opponent pool
- RVR (Reward Variance Reduction, IEEE CoG 2022) for faster training

---

## Component 2: Inference Search -- OLSS (Opponent-Limited Subgame Solving)

**Paper**: ICML 2023 -- "Opponent-Limited Online Search for Imperfect Information Games"

### Algorithm
At each decision point:
1. Build subgame tree rooted at current information set
2. Limit opponent strategy space (key idea: prune unlikely opponent strategies)
3. Solve subgame via CFR for approximate Nash equilibrium
4. Pick action from subgame solution

### Theoretical Properties
- Bounded exploitability: subgame solution is epsilon-Nash in restricted game
- Orders faster than common-knowledge subgame solving (Burch et al.)
- Formally tested on 2-player Mahjong

### Computational Requirements
- Must build + solve explicit game trees
- Est.: ~2400 CPUs + 8 V100 GPUs for real-time play
- Subgame solving is game-theoretically sound (minimax/Nash, not heuristic)

---

## Component 3: Search-as-Feature Integration (Unpublished)

Search results (OLSS subgame solution values) fed BACK into policy net as input features. Architecturally unlike AlphaGo-style MCTS, where search directly overrides policy.

### Mechanism
- OLSS yields action values for current decision
- Values encoded as extra input channels to policy net
- Net learns to combine search info with learned representations
- Enables learned arbitration when search and policy disagree

### Theoretical Motivation
- Policy net can learn WHEN to trust search vs own features
- Search-as-feature lets net contextualize search results
- Avoids "search override" problem where search worse than policy in some states

---

## Component 4: Training Acceleration -- RVR

**Paper**: IEEE CoG 2022 -- "Speedup Training AI for Mahjong via Reward Variance Reduction"

Reduces variance in RL reward signal for Mahjong, which has high stochastic variance from tile draws + scoring structure. Standard variance-reduction method applied to Mahjong domain.

---

## Architecture (Reconstructed, Partially Unknown)

| Aspect | Known | Unknown |
|--------|-------|---------|
| Policy network | Neural net (type unspecified) | Exact architecture, layer count, dims |
| Input encoding | Unspecified | Channel layout, tile representation |
| Output | Policy (action distribution) | Head count, auxiliary objectives |
| Value network | Assumed separate or shared | Architecture details |
| Opponent modeling | None explicit (implicit in self-play) | Any latent opponent representation? |
| Belief tracking | None explicit (implicit in net state) | Any structured belief maintained? |
| Safety/defense | Strong defense observed empirically | How defense encoded/trained |

---

## Design Choices and Their Implications

### Strengths of This Proposal
1. **Game-theoretic training**: ACH gives regret-minimization properties, reducing strategy cycling
2. **Game-theoretic search**: OLSS gives formal safety guarantees on subgame solutions
3. **Zero human data**: No ceiling from human play quality; can exceed human strategies in principle
4. **Search-as-feature**: Novel integration lets net learn contextual search use

### Theoretical Limitations
1. **No multiplayer convergence guarantee**: ACH converges to Nash only in 2-player. In 4-player, no formal guarantee. Training depends on empirical stability.
2. **No explicit belief tracking**: Beliefs about opponent hands stay implicit in hidden state. Not verifiable, not incrementally updated, not constraint-consistent.
3. **No exploitation of opponent tendencies**: Pure self-play trends toward Nash-like strategies. Does not directly target human biases (over-folding, suji overreliance, damaten blindness).
4. **Massive compute requirement**: OLSS needs thousands of CPUs for real-time play. Inaccessible for most teams.
5. **No absent-evidence reasoning**: Does not explicitly model "dog that didn't bark" (non-call evidence). Must learn implicitly from self-play.
6. **No information-theoretic action selection**: Does not explicitly reason about information gain or concealment. Must learn implicitly.
7. **Subgame solving assumes 2-player**: OLSS formally tested on 2-player Mahjong. 4-player adaptation unpublished; theoretical properties unknown.

---

## Published Papers

1. ACH: Fu et al. "Actor-Critic Policy Optimization in a Large-Scale Imperfect-Information Game." ICLR 2022.
2. OLSS: Liu, Fu, Fu, Wei. "Opponent-Limited Online Search for Imperfect Information Games." ICML 2023.
3. RVR: Li, Wu, Fu, Fu, Zhao, Xing. "Speedup Training AI for Mahjong via Reward Variance Reduction." IEEE CoG 2022.
4. DDCFR: Xu, Li, Fu et al. "Dynamic Discounted CFR." ICLR 2024 (Spotlight). (Same team, meta-learned CFR discounting.)