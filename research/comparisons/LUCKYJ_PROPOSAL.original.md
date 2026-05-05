# Proposal A: Game-Theoretic Self-Play with Subgame Search for 4-Player Mahjong

**Team**: Tencent AI Platform Department (5+ researchers)
**Compute**: Estimated 10,000-50,000 GPU-hours

---

## Core Thesis

Train a policy network entirely via self-play using a game-theoretic RL algorithm with Nash convergence properties, then augment at inference time with imperfect-information subgame solving. Use zero human data.

---

## Component 1: Training -- ACH (Actor-Critic Hedge)

**Paper**: ICLR 2022 -- "Actor-Critic Policy Optimization in a Large-Scale Imperfect-Information Game"

Merges deep RL (actor-critic) with Hedge algorithm (weighted CFR) for policy optimization in imperfect-information games.

### Algorithm
- Maintains regret-weighted policy mixture (from Hedge/multiplicative weights)
- Actor-critic provides advantage estimates from self-play trajectories
- Policy update blends RL gradient with regret-minimization update
- Pure self-play: trains entirely from scratch, zero human data

### Theoretical Properties
- Nash convergence at O(T^{-1/2}) rate in 2-player zero-sum games
- Lower variance than previous sampled regret methods (Monte Carlo CFR)
- No convergence guarantee in 4-player (empirical only)

### Training Paradigm
- 100% self-play (no behavioral cloning, no human data, no oracle)
- League-style training with frozen opponent pool
- RVR (Reward Variance Reduction, IEEE CoG 2022) for training acceleration

---

## Component 2: Inference Search -- OLSS (Opponent-Limited Subgame Solving)

**Paper**: ICML 2023 -- "Opponent-Limited Online Search for Imperfect Information Games"

### Algorithm
At each decision point:
1. Construct a subgame tree rooted at the current information set
2. Limit the opponent strategy space (key innovation: prune unlikely opponent strategies)
3. Solve the subgame via CFR to find an approximate Nash equilibrium
4. Select the action prescribed by the subgame solution

### Theoretical Properties
- Bounded exploitability: the subgame solution is epsilon-Nash in the restricted game
- Orders of magnitude faster than common-knowledge subgame solving (Burch et al.)
- Formally tested on 2-player Mahjong

### Computational Requirements
- Requires building and solving explicit game trees
- Estimated: ~2400 CPUs + 8 V100 GPUs for real-time play
- Subgame solving is game-theoretically sound (minimax/Nash, not heuristic)

---

## Component 3: Search-as-Feature Integration (Unpublished)

Search results (OLSS subgame solution values) are fed BACK into the policy neural network as input features. This is architecturally distinct from AlphaGo-style MCTS where search directly overrides the policy.

### Mechanism
- OLSS produces action values for the current decision
- These values are encoded as additional input channels to the policy network
- The network learns to integrate search information with its own trained representations
- Enables learned arbitration between search and policy when they disagree

### Theoretical Motivation
- The policy network can learn WHEN to trust search and when to trust its own features
- Search-as-feature allows the network to contextualize search results
- Avoids the "search override" problem where search can be worse than policy in some states

---

## Component 4: Training Acceleration -- RVR

**Paper**: IEEE CoG 2022 -- "Speedup Training AI for Mahjong via Reward Variance Reduction"

Reduces variance in RL reward signal for Mahjong (which has inherently high stochastic variance due to tile draws and scoring structure). Standard technique from variance reduction literature, applied to the Mahjong domain.

---

## Architecture (Reconstructed, Partially Unknown)

| Aspect | Known | Unknown |
|--------|-------|---------|
| Policy network | Neural network (type unspecified) | Exact architecture, layer count, dimensions |
| Input encoding | Unspecified | Channel layout, tile representation |
| Output | Policy (action distribution) | Number of heads, auxiliary objectives |
| Value network | Assumed separate or shared | Architecture details |
| Opponent modeling | None explicit (implicit in self-play) | Whether any latent opponent representation exists |
| Belief tracking | None explicit (implicit in network state) | Whether any structured belief is maintained |
| Safety/defense | Observed strong defense empirically | How defense is encoded/trained |

---

## Design Choices and Their Implications

### Strengths of This Proposal
1. **Game-theoretic training**: ACH provides regret-minimization properties, preventing strategy cycling
2. **Game-theoretic search**: OLSS provides formal safety guarantees on subgame solutions
3. **Zero human data**: No ceiling from human play quality; can in principle exceed human strategies
4. **Search-as-feature**: Novel integration that lets the network learn to use search contextually

### Theoretical Limitations
1. **No multiplayer convergence guarantee**: ACH converges to Nash only in 2-player. In 4-player, no formal guarantee exists. Training relies on empirical stability.
2. **No explicit belief tracking**: Beliefs about opponent hands are implicit in network hidden state. Not verifiable, not incrementally updated, not constraint-consistent.
3. **No exploitation of opponent tendencies**: Pure self-play converges toward Nash-like strategies. Does not specifically target human biases (over-folding, suji overreliance, damaten blindness).
4. **Massive compute requirement**: OLSS requires thousands of CPUs for real-time play. Not accessible to most research teams.
5. **No absent-evidence reasoning**: Does not explicitly model "the dog that didn't bark" (non-call evidence). Must learn this implicitly from self-play data.
6. **No information-theoretic action selection**: Does not reason about information gain or concealment. Must learn these strategies implicitly.
7. **Subgame solving assumes 2-player**: OLSS was formally tested on 2-player Mahjong. The 4-player adaptation is unpublished and its theoretical properties are unknown.

---

## Published Papers

1. ACH: Fu et al. "Actor-Critic Policy Optimization in a Large-Scale Imperfect-Information Game." ICLR 2022.
2. OLSS: Liu, Fu, Fu, Wei. "Opponent-Limited Online Search for Imperfect Information Games." ICML 2023.
3. RVR: Li, Wu, Fu, Fu, Zhao, Xing. "Speedup Training AI for Mahjong via Reward Variance Reduction." IEEE CoG 2022.
4. DDCFR: Xu, Li, Fu et al. "Dynamic Discounted CFR." ICLR 2024 (Spotlight). (Same team, meta-learned CFR discounting.)
