# Paradigm-Breaking Game AI Approaches: Research for Hydra Mahjong AI

> Research from primary sources, papers, official docs.
> Focus: tech details for novel Mahjong AI design.

---

## Table of Contents

1. [NNUE: Efficiently Updatable Neural Networks](#1-nnue-efficiently-updatable-neural-networks)
2. [Novel NN + Classical Algorithm Hybrids](#2-novel-nn-classical-algorithm-hybrids)
3. [Poker AI: Libratus & Pluribus](#3-poker-ai-beyond-standard-cfr)
4. [Recent Imperfect Information Game AI (2020-2025)](#4-recent-imperfect-information-game-ai-2020-2025)
5. [Cross-Domain Approaches](#5-cross-domain-approaches)
6. [Synthesis: Ideas for Mahjong AI](#6-synthesis-paradigm-breaking-ideas-for-hydra-mahjong-ai)

---

## 1. NNUE: Efficiently Updatable Neural Networks

**Source**: [Chessprogramming Wiki](https://www.chessprogramming.org/NNUE) | [Stockfish NNUE PyTorch](https://deepwiki.com/official-stockfish/nnue-pytorch/9-nnue-architecture-reference)

### 1.1 The Paradigm Break

Before NNUE (2018-2020), chess engines split into 2 camps:
- **Handcrafted eval** (Stockfish classic): fast, millions nodes/sec, human-limited
- **Large NN eval** (Leela/AlphaZero): superhuman knowledge, ~1000x slower, GPU-heavy

**NNUE insight**: get NN-grade eval at handcrafted-eval speed by exploiting *incremental state change structure*.

### 1.2 Architecture

```
Input Layer (HalfKP): 40,960 binary features (per perspective)
    |
Feature Transformer: 40960 -> 256 (with ACCUMULATOR)
    |
[White Accumulator (256)] || [Black Accumulator (256)] = 512
    |
Hidden Layer 1: 512 -> 32
    |
Hidden Layer 2: 32 -> 32
    |
Output: 32 -> 1 (centipawn score)
```

Modern Stockfish uses **8 LayerStack buckets** (material-count-indexed mixture-of-experts).

### 1.3 HalfKP Feature Encoding

"Half-King-Piece" encodes piece-king spatial relations as binary features:

```
index = piece_square + (piece_type * 2 + piece_color + king_square * 10) * 64
```

- 64 king squares x 10 piece types (Q,R,B,N,P for each side) x 64 piece squares = **40,960 features**
- Both perspectives kept = 81,920 total binary inputs
- sparse: ~30 active features per position (~0.07%)

### 1.4 The Key Innovation: Incremental Accumulator Updates

Core insight. First-layer output (accumulator):

**Full computation:**
```
accumulator = bias + SUM(W[:, i] for i in active_features)
```

**Incremental update (after move):**
```
acc_new = acc_old - SUM(W[:, r] for r in removed_features)
                  + SUM(W[:, a] for a in added_features)
```

Typical non-king move changes only **2-4 features** of 40,960. So instead of
recomputing 40,960 x 256 multiplications, do 2-4 vector adds/subtracts
of size 256. Result: **~10-15x speedup** vs full recompute.

**Why works**: input is *binary* (0/1), so "multiplication" becomes conditional
addition. Moves change few pieces, so delta tiny.

**King moves special**: they invalidate ALL features for that perspective (every
feature depends on king position). Need full refresh. Stockfish uses "Finny Tables"
(cached accumulator per king bucket) to amortize cost.

### 1.5 Quantization for CPU SIMD

NNUE runs fully in integer arithmetic on CPU SIMD:

| Component | Type | Scale | Range |
|-----------|------|-------|-------|
| FT weights/biases | int16 | 127 | [-127, 127] |
| FT activations | int16 (ClippedReLU) | - | [0, 127] |
| Hidden weights | int8 | 64 | [-127, 127] |
| Hidden activations | int8 (ClippedReLU) | - | [0, 127] |
| Output accumulation | int32 | - | full range |

**Feature Transformer quantization:**
```
W_ij = round(127 * w_ij)     // float -> int16
B_j  = round(127 * b_j)
Y_j  = SUM(x_i * W_ij) + B_j  // x_i is 0 or 1, so this is conditional add
output_j = clamp(Y_j, 0, 127)  // ClippedReLU
```

**Hidden layer quantization:**
```
W_jk = round(64 * w_jk)      // float -> int8
B_k  = round(b_k * 127 * 64) // int32
Y_k  = (SUM(X_j * W_jk) + B_k) / 64
output_k = clamp(Y_k, 0, 127)
```

### 1.6 Advanced Techniques

- **SCReLU** (Squared ClippedReLU): `clamp(x,0,1)^2` -- stronger than CReLU, harder to vectorize
- **Pairwise Multiplication**: split accumulator, multiply pairs, reduce width
- **King Input Buckets**: multiple weight sets per king region (mixture-of-experts)
- **LayerStacks**: switch post-accumulator parameters by material count
- **Lizard SCReLU trick**: compute `(v*w)*v` not `(v*v)*w` to stay in int16 range

---

## 2. Novel NN + Classical Algorithm Hybrids

### 2.1 Student of Games (DeepMind, 2023) -- GT-CFR

**Source**: [Science Advances 2023](https://www.science.org/doi/pdf/10.1126/sciadv.adg3256) | [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10651118/)

**Paradigm break**: first single algorithm for BOTH perfect and imperfect
information games. AlphaZero works only for perfect info; CFR-family mostly for
imperfect info. SoG does both.

**How it works -- Growing-Tree CFR (GT-CFR)**:
1. Like MCTS, grows search tree non-uniformly toward promising states
2. Like CFR, uses regret minimization for game-theoretic soundness
3. Uses **Counterfactual Value-and-Policy Network (CVPN)** at frontier leaves

**CVPN (neural net)**:
- Input: Public Belief State beta = (s_pub, r) where r = range distributions
- Output: counterfactual values + policy targets
- Training losses: Huber loss (values) + cross-entropy (policy)
- Targets from both game outcomes and bootstrapped GT-CFR solves

**Key insight -- Sound Self-Play**:
Searches at different public states must stay *globally consistent*.
Trivial in perfect-info (subtrees independent), critical in imperfect-info
where beliefs propagate through tree.

**Complexity**: GT-CFR re-solving with T iterations, k expanded children gives
O(kT^2) public state visits. In perfect-info, simplifies to O(T) network calls.

### 2.2 ReBeL (Meta/CMU, 2020) -- RL+Search for Imperfect Info

**Source**: [NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/c61f571dbd2fb949d3fe5ae1608dd48b-Paper.pdf) | [Meta AI Blog](https://ai.meta.com/blog/rebel-a-general-game-playing-ai-bot-that-excels-at-poker-and-more/)

**Paradigm break**: generalizes AlphaZero's "RL + Search" to imperfect
information games. Before ReBeL, RL+Search was viewed as fundamentally
incompatible with imperfect info.

**Core concept -- Public Belief States (PBS)**:
- In imperfect-info games, "state" = probability distribution over possible
hidden states, not single state
- PBS = (action_history, belief_distributions_per_player)
- Converts imperfect-info game into continuous-state perfect-info game
- Then AlphaZero-style RL+Search can run on converted game

**Why naive conversion fails**: PBS space extremely high-dimensional.
In toy poker, action space alone is 156-dimensional.

**How ReBeL handles it**: uses CFR as "gradient descent for games" -- efficient
search exploiting convex optimization structure of two-player zero-sum games.
Proven to converge to Nash equilibrium.

### 2.3 Deep Synoptic Monte Carlo Planning (DSMCP, 2021)

**Source**: [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/215a71a12769b056c3c32e7299f1c5ed-Paper.pdf)

**Paradigm break**: uses particle filters + stochastic abstractions for
imperfect-info planning.

**How it works**:
1. Maintains belief state via **unweighted particle filter** (set of possible worlds)
2. Plans by sampling from belief state and running playouts
3. Instead of exact information states, uses **"synopses"** --
stochastic abstractions summarizing information states
4. Neural nets evaluate synopsis-conditioned states

**Key insight**: no need enumerate all hidden states. Sample them, abstract into
"synopses" (statistical summaries), plan over those.

---

## 3. Poker AI: Beyond Standard CFR

### 3.1 Libratus (CMU, 2017) -- Three-Module Architecture

**Source**: [IJCAI 2017](https://noambrown.github.io/papers/17-IJCAI-Libratus.pdf) | [NSF](https://par.nsf.gov/servlets/purl/10077470)

First AI to beat top human professionals in heads-up no-limit Texas Hold'em.

**Module 1: Blueprint Strategy (offline)**
- Game has ~10^161 decision points; abstracted to ~10^12
- Solved with MCCFR + **Regret-Based Pruning (RBP)**
  - RBP: skip branches with strongly negative cumulative regret
  - Speeds convergence and mitigates imperfect-recall abstraction errors
- No card abstraction on preflop/flop; coarser on later rounds:
  - Round 3: 55M hands -> 2.5M buckets
  - Round 4: 2.4B possibilities -> 1.25M buckets

**Module 2: Nested Safe Subgame Solving (online)**
Main novelty:
- When opponent makes any bet, construct and solve NEW subgame in real time
- Each subgame uses: NO card abstraction + DENSE action abstraction
- **Safety guarantee**: new strategy provably no worse than blueprint
- Uses opponent's cumulative mistakes to EXPAND safe optimization polytope
- **Dynamic action abstractions**: each subgame uses different bet sizes,
forcing opponent to keep adapting

Prior approaches used "action translation" (round off-tree bets to nearest
in-tree action). Libratus removes this in later rounds.

**Module 3: Self-Improver (background)**
- Monitors off-tree actions opponent uses most
- Adds those actions to abstraction
- Selection criterion: frequency * distance_from_nearest_abstract_action
- Computes new strategy for each added action via subgame solving

### 3.2 Pluribus (Meta/CMU, 2019) -- Multiplayer Depth-Limited Search

**Source**: [Science 2019](https://www.science.org/doi/10.1126/science.aay2400)

First superhuman AI for 6-player no-limit Texas Hold'em.

**Key novelty -- Depth-Limited Imperfect-Info Search**:

Libratus could search only when close enough to solve endgame. With 6 players,
game tree explodes exponentially. Pluribus solved this with **depth-limited
search + continuation policies**.

**Blueprint**: computed with Monte Carlo Linear CFR
- **Linear CFR**: iteration T weighted by T (not 1), so early random iterations
decay as 1/SUM(t=1..T, t) instead of 1/T. Faster convergence.

**Real-time search with continuation policies**:
At depth limit (where full solve to endgame impossible), each player chooses from
k=4 **continuation strategies**:
1. Blueprint strategy (balanced play)
2. Blueprint biased toward folding
3. Blueprint biased toward calling
4. Blueprint biased toward raising

Terminal values estimated by **rolling out** rest of game with chosen
continuation profile. This avoids core imperfect-info problem:
leaf values depend on strategy chosen in subgame.

**During search**: Pluribus tracks its **range** (distribution over
private hands) and computes strategy balanced across ALL possible hands,
then samples action for actual hand held.

### 3.3 What Made Poker AI Paradigm-Breaking (Summary)

| Innovation | Why It Matters |
|-----------|---------------|
| Safe subgame solving | Proves new strategy >= old, enables online refinement |
| Depth-limited search + continuation policies | Makes online search tractable for multiplayer |
| Linear CFR weighting | Faster convergence by discounting early noise |
| Blueprint + online refinement | Two-phase architecture: coarse offline, fine online |
| Regret-based pruning | Skip losing branches, focus compute on promising ones |
| Range-balanced play | Strategy coherent across all possible private states |

---

## 4. Recent Imperfect Information Game AI (2020-2025)

### 4.1 DeepNash / R-NaD (DeepMind, 2022) -- Model-Free Nash Convergence

**Source**: [Science 2022](https://www.science.org/doi/10.1126/science.add4679) | [arXiv:2206.15378](https://arxiv.org/abs/2206.15378)

Mastered Stratego (10^535 game tree, 10^175x larger than Go) with NO search.

**Paradigm break**: model-free RL with provable Nash convergence
in imperfect-info games. No CFR, no search, no explicit belief modeling.

**R-NaD Algorithm (Regularized Nash Dynamics)**:

Core problem: standard policy gradient in multi-agent games CYCLES around Nash
equilibrium (like Rock-Paper-Scissors -- keep adjusting, overshoot).

R-NaD fixes this with **reward transformation**:

```
r_transformed(pi_i, pi_-i, a_i, a_-i) =
    r(a_i, a_-i)
    - eta * log(pi_i(a_i) / pi_reg_i(a_i))     // penalize deviation from reg
    + eta * log(pi_-i(a_-i) / pi_reg_-i(a_-i))  // reward opponent staying close
```

Where:
- eta > 0 is regularization strength
- pi_reg is regularization policy (anchor point)
- log-ratio terms are KL-divergence gradients

**Why this prevents cycling -- Lyapunov function**:

Transformed game has UNIQUE fixed point pi_fix. Distance to this fixed
point decreases exponentially:

```
d/dt H(pi_fix, pi_t) <= -eta * H(pi_fix, pi_t)
```

where H is KL divergence from pi_fix to current policy pi_t.

**Nested loop structure**:
1. OUTER LOOP: set regularization policy pi_reg
2. INNER LOOP: run replicator dynamics (policy gradient) on transformed game
until convergence to fixed point pi_fix
3. UPDATE: set pi_reg = pi_fix for next outer iteration
4. REPEAT: sequence of fixed points converges to Nash equilibrium of ORIGINAL game

**NeuRD (Neural Replicator Dynamics) for deep learning**:

In practice, R-NaD uses NeuRD -- neural network parameterization of
replicator dynamics:

- Fast parameters (theta_n): updated every step via Adam on NeuRD loss
- Slow target parameters: theta_{n+1,target} = gamma * theta_{n+1} + (1-gamma) * theta_{n,target}
- After Delta_m steps: extract pi_fix from slow params, set as new pi_reg

NeuRD update operates on LOGITS (not probabilities), with clipping to stop
logit explosion:

```
Lambda_n = -[lr * grad(L_critic) + (1/T) * SUM_t SUM_a grad(logit(a) * Clip(Q(a), c))]
```

**For Mahjong relevance**: R-NaD shows Nash convergence can happen
WITHOUT search, WITHOUT explicit belief tracking, via pure model-free RL
with right reward transformation. Huge for 4-player mahjong, where
search is often intractable.

### 4.2 Suphx (Microsoft Research, 2020) -- State of Mahjong AI

**Source**: [arXiv:2003.13590](https://arxiv.org/abs/2003.13590)

Current strongest Mahjong AI (10-dan on Tenhou, top 0.01% of humans).

**Three key techniques**:

1. **Global Reward Prediction**: predict tournament-level
outcomes, not per-hand reward. Better aligns signal with real objective.

2. **Oracle Guiding**: train with oracle (perfect info) as teacher, then
distill to imperfect-info student. Oracle sees all tiles; student learns
oracle-like decisions from partial information.

3. **Run-time Policy Adaptation**: adjust policy during play from observed
opponent patterns. Strategy not fixed; adapts online.

**What Suphx does NOT do**: no search, no CFR, no explicit belief modeling over
opponent hands. Pure policy network with strong training tricks.

### 4.3 Bayesian Opponent Modeling with Belief Updates (2024-2025)

**Source**: [arXiv:2405.14122](https://arxiv.org/abs/2405.14122) | [HORSE-CFR (2024)](https://www.sciencedirect.com/science/article/pii/S0957417424025648)

Recent work combines Bayesian belief tracking with game-theoretic solving:

**Key concepts**:
- Maintain posterior over opponent types/strategies
- Update beliefs via Bayes' theorem after each observed action
- Use updated beliefs to choose exploitation strategy
- Balance Nash (safe) play with exploitative (Bayesian) play

**HORSE-CFR**: Hierarchical Opponent Reasoning for Safe Exploitation
- Neural net infers missing information to improve Bayesian posterior accuracy
- Accounts for UNCERTAINTY in belief update, not only point estimates
- Hierarchical: reasons about opponent's model of YOUR strategy

### 4.4 Preference-CFR: Beyond Nash Equilibrium (2024)

**Source**: [Semantic Scholar](https://www.semanticscholar.org/paper/Preference-CFR%3A-Beyond-Nash-Equilibrium-for-Better-Ju-Tellier/548481122339a162bf1bba36f878536380003061)

**Key insight**: Nash equilibrium is DEFENSIVE -- unexploitable strategy.
Against weak opponents, better exploit mistakes than only play safe.
Preference-CFR computes beyond-Nash strategies by incorporating preferences
about opponent tendencies.

---

## 5. Cross-Domain Approaches

### 5.1 Decision Transformers (2021) -- RL as Sequence Modeling

**Source**: [NeurIPS 2021](https://openreview.net/forum?id=gaCGNwsWITG) | [Berkeley](https://sites.google.com/berkeley.edu/decision-transformer)

**Paradigm break**: recasts reinforcement learning as SEQUENCE MODELING.
No value functions, no policy gradients, no Bellman equations. transformer
predicting next action from history.

**Architecture**:
- Input sequence: (R_1, s_1, a_1, R_2, s_2, a_2,..., R_t, s_t)
- R_t = return-to-go (desired future cumulative reward)
- Output: predicted action a_t
- Trained on offline trajectories via standard cross-entropy/MSE loss
- At inference: set R_1 = desired_return, autoregressive generation

**Why matters**: no temporal difference learning, no bootstrapping,
no exploration-exploitation tradeoff. Transformer learns mapping from
(desired outcome + history) -> action. Want stronger play? Set higher return-to-go.

**For Mahjong**: mahjong game is naturally sequence of observations/actions.
Decision Transformer could learn from expert replays without reward shaping.

### 5.2 DreamerV3 (2023) -- World Models

**Source**: [Nature 2025](https://www.nature.com/articles/s41586-025-08744-2) | [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)

**Paradigm break**: learn world model in LATENT space, then train policy
entirely inside "dreams" (imagined trajectories). No real env interaction
needed during policy optimization.

**Architecture**:
1. **Encoder**: observation -> latent state z_t
2. **Dynamics model**: (z_t, a_t) -> z_{t+1} (predicts next latent state)
3. **Reward predictor**: z_t -> r_t
4. **Decoder**: z_t -> reconstructed observation (for training signal)
5. **Actor-Critic**: trained on imagined trajectories in latent space

**Key innovation -- Symlog predictions**: handles rewards across many orders of
magnitude without manual normalization. Uses symlog(x) = sign(x) * ln(|x| + 1).

**For Mahjong**: world model could learn "physics" of mahjong tile dynamics --
likely draws from visible info, how opponent strategies evolve.
Train policy in imagined games instead of expensive self-play.

### 5.3 MPC + RL Unification (Bertsekas, 2024) -- Newton's Method Bridge

**Source**: [arXiv:2406.00592](https://arxiv.org/abs/2406.00592) | [MIT](https://web.mit.edu/dimitrib/www/IFAC_Overview_Paper_2024.pdf)

**Paradigm break**: shows Model Predictive Control (MPC) and RL are
same algorithm viewed from different lenses, linked by
Newton's method for solving Bellman equations.

**Two-phase architecture**:
1. **Offline training**: learn approximate value function via RL/self-play
2. **Online planning**: use value function as terminal cost in MPC-style
lookahead planning (Newton step refinement)

Offline phase gives "landscape"; online phase performs local
optimization on that landscape. Each improves other.

**For Mahjong**: Libratus-style (blueprint + online
subgame solving), but mathematically formalized. Good template for
principled offline/online Hydra architecture.

### 5.4 Particle Filter Belief Tracking for Games (DSMCP, 2021)

Covered in Section 2.3, but cross-domain view: particle filters
are standard in robotics (SLAM, object tracking) but novel in game AI.

**Key cross-pollination idea**: instead of exact beliefs over opponent
hands (combinatorial explosion), maintain PARTICLE SET -- collection of
plausible opponent hand assignments + wall states -- and update via sequential Monte
Carlo as new evidence (discards, calls, etc.) arrives.

---

## 6. Synthesis: Paradigm-Breaking Ideas for Hydra Mahjong AI

### 6.1 The Mahjong Problem Space

Mahjong combines challenges from many domains:
- **Imperfect info** (like poker): 3 opponents' hands + wall hidden
- **Sequential decisions** (like chess): ~70 decision points per hand
- **4 players** (like Pluribus): no Nash equilibrium guarantee in general
- **Stochastic** (like backgammon): tile draws random
- **Rich action space**: Discard (34), Chi/Pon/Kan, Riichi, Tsumo, Ron

### 6.2 Idea 1: NNUE-Style Incremental Encoding for Mahjong

**Inspiration**: NNUE's accumulator update

**Insight**: in mahjong, each action (draw, discard, call) changes few
features in game state. Current Hydra 85x34 encoding could benefit from
NNUE-style incremental updates:

- Tile draw: 1 tile moves wall -> hand (+1 feature change)
- Discard: 1 tile moves hand -> discard pond (+2 feature changes)
- Call (chi/pon): 2-3 tiles move hand -> melds (+3-4 feature changes)

**Proposed architecture**:
```
Feature Transformer: sparse_features -> 256 (with accumulator)
    |-- Incremental update on each action
    |
Residual Blocks: 256-channel SE-ResNet (current Hydra architecture)
    |
Output Heads: Policy(46) + Value(1) + GRP(24) + Tenpai(3) + Danger(3x34)
```

Feature Transformer keeps accumulator, updated incrementally.
Residual blocks still run fully each time "thinking" part),
but expensive input encoding gets amortized.

**Potential speedup**: during search/simulation, if Hydra adds lookahead,
accumulator updates make eval much cheaper.

**Quantization angle**: NNUE int8/int16 quantization could let Hydra run
inference on CPU, no GPU dependency -- important for deployment.

### 6.3 Idea 2: R-NaD-Style Training for 4-Player Convergence

**Inspiration**: DeepNash's R-NaD

**Problem**: standard self-play RL in 4-player games often does not converge to Nash.
Policies cycle. This is why many mahjong AIs (Suphx, Mortal) rely heavily on
supervised learning from human data.

**Proposed approach**: apply R-NaD reward transformation to Hydra PPO training:

```
r_transformed_i = r_original_i
    - eta * KL(pi_i || pi_reg_i)      // stay near anchor
    + eta * SUM_j!=i KL(pi_j || pi_reg_j)  // adapted for 4-player
```

Nested loop:
1. Train with current pi_reg as anchor (inner loop, many PPO steps)
2. Extract converged policy, set as new pi_reg (outer loop)
3. Repeat until equilibrium

**Challenge**: R-NaD proven for 2-player zero-sum. Mahjong is 4-player,
not strictly zero-sum (one wins, three lose, scores vary).
Lyapunov proof may fail. Still, R-NaD-style regularization may
empirically stabilize training.

### 6.4 Idea 3: Pluribus-Style Depth-Limited Search with Continuation Policies

**Inspiration**: Pluribus

**Approach**: Hydra trains policy net (like current plan). At inference,
before critical decisions, run lightweight depth-limited search:

1. For current state, enumerate K plausible opponent hand configs
(using danger/tenpai heads as belief model)
2. For each config, simulate D turns ahead with policy net
3. At depth limit, evaluate using k=4 continuation variants:
   - Balanced (base policy)
   - Defensive (bias toward safe discards)
   - Aggressive (bias toward riichi/calls)
   - Opportunistic (bias toward value hands)
4. Average values across opponent configs, weighted by belief probability
5. Choose action with highest expected value

**Key advantage**: policy net gives fast eval. Search runs only
at critical decisions (riichi, dangerous discards, calling choices).
Most turns use policy net directly.

### 6.5 Idea 4: Particle Filter Belief Tracking

**Inspiration**: DSMCP + robotics SLAM

**Approach**: instead of fixed belief features about opponent hands,
maintain N=1000 "particles" -- each particle = full assignment of
hidden tiles to opponents + wall.

After each observed action (discard, call, skip):
1. Weight each particle by P(action | particle's hidden state, opponent_policy)
2. Resample particles proportional to weights
3. Add noise (jitter) to stop particle depletion

This gives continuously updated Bayesian belief over hidden game state.
Can be:
- Summarized as features for policy net
- Used directly for search (sample particles, plan per world)
- Used for danger estimates (what fraction of particles have opponent tenpai?)

### 6.6 Idea 5: Decision Transformer for Mahjong

**Inspiration**: Decision Transformers + Suphx's oracle guiding

**Approach**: frame mahjong as sequence modeling:

Input tokens:
```
[R_target, obs_1, action_1, obs_2, action_2, ..., obs_t]
-> predict action_t
```

Where R_target is desired placement (1st/2nd/3rd/4th) or score.

**Training**: on human expert replays from Tenhou. No reward shaping needed.
Transformer learns mapping: (desired outcome + game history) -> action.

**Key advantage**: at inference, set R_target = "1st place" and model
generates actions conditioned on that goal. Want safer play?
Set R_target = "2nd place". Gives natural risk-reward control.

**Combined with oracle guiding**: train 2 models:
1. Oracle DT: sees all hands, conditioned on outcome
2. Student DT: sees only own hand, trained to match oracle's actions

This is Suphx approach with transformer architecture
instead of CNN, potentially capturing longer-range dependencies.

### 6.7 Idea 6: World Model for Mahjong (DreamerV3-inspired)

**Inspiration**: DreamerV3

**Approach**: learn latent dynamics model of mahjong:

1. **Encoder**: game_observation -> z_t (latent state, ~256 dims)
2. **Dynamics**: (z_t, action_t) -> z_{t+1}
But also: (z_t, opponent_action_t) -> z_{t+1}
And: (z_t) -> draw_distribution (what tile am I likely to draw?)
3. **Reward**: z_t -> expected_final_score
4. **Policy**: trained in "dreams" -- imagined game trajectories in latent space

**Key advantage for mahjong**: dynamics model implicitly learns:
- What tiles likely remain in wall
- How opponent discards correlate with their hands
- "Flow" of mahjong game (early exploration -> mid-game direction -> endgame defense)

**Critical challenge**: mahjong has high stochasticity. Each draw is random
from depleting wall. World model must capture uncertainty well.
DreamerV3 symlog predictions may help with variable reward scales.

### 6.8 Idea 7: Two-Phase Architecture (Most Promising Synthesis)

**Inspiration**: Libratus/Pluribus + NNUE + R-NaD

Most promising approach combines many paradigms:

**Phase 1: Offline (Training)**
- Train strong policy net with R-NaD-style regularized self-play
- Use NNUE-style architecture: incremental feature transformer + residual blocks
- Multi-head output: policy + value + danger + tenpai + game result prediction
- Quantize to int8/int16 for CPU inference

**Phase 2: Online (Inference)**
- Most turns: use policy net directly (fast, ~1ms per decision)
- Critical decisions (riichi, dangerous discards, late game):
  - Run particle filter to estimate opponent hands
  - Do depth-limited search (4-8 turns ahead) with policy net as evaluator
  - Use Pluribus-style continuation policies at search leaves
  - Choose action balancing expected value across belief particles

**Why this could be paradigm-breaking for mahjong**:
- Current SOTA (Mortal, Suphx) uses pure policy nets with NO online search
- Even lightweight search at critical moments could be big jump
- NNUE-style incremental updates make CPU search feasible
- R-NaD-style training may converge better than standard self-play
- Particle filter beliefs give principled opponent modeling

---

## Key References

1. **NNUE**: Yu Nasu (2018). Chessprogramming Wiki. Stockfish NNUE-PyTorch.
2. **Student of Games**: Schmid et al. (2023). Science Advances.
3. **ReBeL**: Brown et al. (2020). NeurIPS 2020.
4. **Libratus**: Brown & Sandholm (2017). IJCAI 2017.
5. **Pluribus**: Brown & Sandholm (2019). Science.
6. **DeepNash / R-NaD**: Perolat et al. (2022). Science.
7. **DSMCP**: Markowitz et al. (2021). NeurIPS 2021.
8. **Decision Transformer**: Chen et al. (2021). NeurIPS 2021.
9. **DreamerV3**: Hafner et al. (2023). Nature 2025.
10. **Suphx**: Li et al. (2020). arXiv:2003.13590.
11. **MPC+RL**: Bertsekas (2024). arXiv:2406.00592.
12. **Preference-CFR**: Ju & Tellier (2024).
13. **HORSE-CFR**: (2024). Expert Systems with Applications.