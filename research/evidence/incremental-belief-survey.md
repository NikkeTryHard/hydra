# Prior Art Survey: Incremental Belief Networks for Imperfect-Information Games

**Date**: 2026-03-03
**Context**: Hydra Mahjong AI -- exploring NNUE-inspired incremental belief tracking

---

## Executive Summary

**Idea of NNUE-style incrementally-updatable belief nets for imperfect-info games looks genuinely novel.** No prior work found that directly combines NNUE accumulator-style incremental feature updates with belief-state tracking for hidden-info games. Closest prior work falls into 7 distinct threads below.

---

## 1. NNUE: The Incremental Update Foundation (Perfect Information Only)

**Key insight**: NNUE gets 10-15x speedup by incrementally updating only changed features.

### Architecture
- Overparameterized input layer (e.g., 40,960 inputs for HalfKP in chess)
- Only ~30 features active per position
- Accumulator stores first-layer output as persistent state

### Core Equations

**Full refresh** (from scratch):
```
accumulator = bias + SUM(weight_column[i] for i in active_features)
```

**Incremental update** (on move, removing set R, adding set S):
```
accumulator_new = accumulator_old - SUM(W[:, r] for r in R) + SUM(W[:, a] for a in S)
```

Between consecutive positions, usually only 2-4 features change, so cost becomes O(changed_features * hidden_dim) not O(total_features * hidden_dim).

**Sources**:
- [Stockfish NNUE docs](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)
- [NNUE Architecture Reference (DeepWiki)](https://deepwiki.com/official-stockfish/nnue-pytorch/9-nnue-architecture-reference)
- [HalfKP/NNUE GitHub](https://github.com/HalfKP/NNUE)
- [NNUE Wikipedia](https://en.wikipedia.org/wiki/Efficiently_updatable_neural_network)

**Gap**: NNUE has NEVER been applied to imperfect-information games. It works on fully observable board states only (chess, shogi). Proposed extension to belief states looks novel.

---

## 2. ReBeL: Recursive Belief-Based Learning (Meta AI, 2020)

**Foundational paper for neural belief-state game AI.**

### Paper
Brown, Noam et al. "Combining Deep Reinforcement Learning and Search for Imperfect-Information Games." NeurIPS 2020.
- [arXiv](https://arxiv.org/abs/2007.13544)
- [Meta AI Blog](https://ai.meta.com/blog/rebel-a-general-game-playing-ai-bot-that-excels-at-poker-and-more/)
- [GitHub (Liar's Dice)](https://github.com/facebookresearch/rebel)

### Key Concept: Public Belief State (PBS)

PBS = probability distribution over hidden information sets, conditioned on public history:

```
PBS_t = b_t = Pr(hidden_infosets | public_history, common_knowledge_policy)
```

This turns imperfect-information game into continuous-state perfect-information game over beliefs.

### Belief Update (Bayes Rule)

Given weights w(s) over hidden states s, and observed public action

```
P(a) = SUM_s P(a|s) * w(s) / SUM_s w(s)

w'(s) = w(s) * P(a|s) / SUM_s' w(s') * P(a|s')
```

This is standard Bayesian filtering. Key innovation: `P(a|s)` comes from CURRENT policy, which changes during training, so target moves too.

### Search: CFR in Belief Space

- AlphaZero-style MCTS is intractable in belief space (actions become continuous distributions)
- ReBeL uses Counterfactual Regret Minimization (CFR) as search in depth-limited subgames
- At each decision: build subgame at current PBS, run K iterations of CFR, sample action, update PBS

### Value Network
- Trained on terminal game values from self-play trajectories
- Input: PBS (belief distribution)
- Output: expected value for each player

### Convergence

Provably converges to epsilon-Nash equilibrium in two-player zero-sum games.

**Relevance to idea**: ReBeL Bayesian belief update gives "what" to update, but computes it externally, not learned. NNUE-style net could LEARN belief update as incremental weight adjustment.

---

## 3. Student of Games (DeepMind/Amii, 2023)

**Unifies perfect- and imperfect-information game AI.**

### Paper
Schmid et al. "Student of Games: unified learning algorithm for both perfect and imperfect information games." Science Advances, 2023.
- [Science](https://www.science.org/doi/10.1126/sciadv.adg3256)
- [arXiv](https://arxiv.org/abs/2112.03178)

### Key Innovation: GT-CFR + CVPN

- **Growing-Tree CFR (GT-CFR)**: incrementally grows search tree instead of fixed-depth subgames like ReBeL
- **Counterfactual Value-and-Policy Network (CVPN)**: neural net takes PBS input, outputs both counterfactual values and action policies for each information state
- Self-play generates two training-data types:
  1. Search queries (PBS nodes queried during GT-CFR regret updates)
  2. Full-game trajectories

### Architecture
- CVPN: belief -> (values_per_infostate, policy_per_infostate)
- Sound for both perfect-info (chess, Go) and imperfect-info (poker, Scotland Yard)

**Relevance**: SoG CVPN is closest existing architecture to proposal; it processes belief states through neural net. But it does NOT use incremental updates; it recomputes from scratch each time.

---

## 4. DreamerV3 RSSM: World Models for Partial Observability

**State of art for learned latent dynamics under partial observability.**

### Paper
Hafner et al. "Mastering Diverse Domains through World Models." Nature, 2025.
- [Nature](https://www.nature.com/articles/s41586-025-08744-2)
- [DreamerV3 RSSM (DeepWiki)](https://deepwiki.com/danijar/dreamerv3/4.1-world-model-(rssm))

### Recurrent State-Space Model (RSSM)

Latent state is hybrid: s_t = (d_t, z_t) where:
- d_t in R^8192: deterministic recurrent component (GRU-like)
- z_t in {0,1}^(32x64): stochastic discrete component (categorical)

**Observe mode** (with real observations):
```
x_t = Encoder(o_t)
(d_t, z_t), carry_t = RSSM.observe(carry_{t-1}, x_t, a_{t-1}, reset_t)
```

**Imagine mode** (no observations, planning):
```
a_t ~ pi(concat(d_t, flatten(z_t)))
(d_{t+1}, z_{t+1}), carry_{t+1} = RSSM.imagine((d_t, z_t), a_t)
```

**Feature vector for all heads**:
```
f_t = concat(d_t, flatten(z_t)) in R^10240
```

### How It Handles Partial Observability

- Deterministic recurrent state d_t summarizes ALL past observations/actions
- This IS learned belief state, sufficient statistic of history
- Stochastic z_t captures remaining uncertainty
- Training uses KL divergence between prior (dynamics-predicted) and posterior (observation-informed) z_t

**Relevance**: RSSM is probably closest existing architecture to "incremental belief update net." Each step updates d_t incrementally through GRU. But:
1. Built for single-agent POMDPs, not multi-agent games
2. Update is implicit (learned GRU weights), not sparse/efficient like NNUE
3. Does not exploit card-game structure (sparse, discrete changes)

---

## 5. Deep CFR and DREAM: Neural CFR for Large Games

### Deep Counterfactual Regret Minimization
Brown et al. "Deep Counterfactual Regret Minimization." ICML 2019.
- [arXiv](https://arxiv.org/abs/1811.00164)
- [Meta AI](https://ai.meta.com/research/publications/deep-counterfactual-regret-minimization/)

Replaces tabular CFR with neural nets that approximate cumulative regrets.
- At each CFR iteration, neural net V_theta predicts counterfactual values
- Advantage net stores regrets as training targets in reservoir buffer
- Strategy net averages over time

### DREAM (Deep Regret minimization with Advantage baselines and Model-free learning)
- [GitHub](https://github.com/EricSteinberger/DREAM)
- Scalable impl of Deep CFR variants
- Includes SD-CFR, Deep CFR, and NFSP implementations in PokerRL framework

**Relevance**: Deep CFR advantage net could benefit from NNUE-style incremental updates; regrets change incrementally between CFR iterations.

---

## 6. Efficient Incremental Belief Updates (Non-Game, Bayesian)

### Paper
"Efficient Incremental Belief Updates" (arXiv 2402.06940, 2024)
- [arXiv](https://arxiv.org/html/2402.06940)

### Core Idea: Weighted Virtual Observations

Compress past posterior into small set of weighted "virtual observations" that ~ preserve posterior:

```
min_w KL(p(x|y*) || p(x|y_hat, w))
```

Where y* is original data, y_hat are virtual observations, w are learned weights.

**Weighted virtual observation likelihood**:
```
p(w|x) = exp(log h(w) + SUM_i w_i * log p(y_hat_i | x))
```

**Incremental update**: Instead of rerunning inference on ALL historical observations, condition on:

```
new_posterior = inference(compressed_past_belief + new_observations)
```

This is strongest mathematical incremental-belief-update framework found, but for general probabilistic programming, not game AI.

**Relevance**: Direct mathematical base for approach. Could combine with NNUE-style accumulator: "accumulator" stores compressed belief state, new observations trigger incremental weight adjustments.

---

## 7. BetaZero: Belief-State Planning for POMDPs

### Paper
Moss et al. "BetaZero: Belief-State Planning for Long-Horizon POMDPs using Learned Approximations." RLC 2024.
- [arXiv](https://arxiv.org/abs/2306.00249)
- [GitHub](https://github.com/sisl/BetaZero.jl)

### Architecture
- AlphaZero-style self-play + MCTS, but over BELIEF states not world states
- Neural net approximates value and policy given BELIEF REPRESENTATION
- User defines `input_representation(belief)` -- e.g., mean and std of particle filter
- Uses PUCT search in belief space

### Key Design
```julia
function BetaZero.input_representation(b::ParticleCollection)
    mu, sigma = mean_and_std(s.y for s in particles(b))
    return Float32[mu, sigma]
end
```

Belief is compressed into sufficient statistics (mean, variance) before feed into neural net.

**Relevance**: BetaZero shows MCTS can work over belief states if sufficient statistics are good. NNUE-style net could LEARN these sufficient statistics incrementally.

---

## 8. Deep Belief Markov Models (DBMM) for POMDP Inference

### Paper
"Deep Belief Markov Models for POMDP Inference" (arXiv 2503.13438, 2025)
- [arXiv](https://arxiv.org/abs/2503.13438)
- [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0893608025012675)

### Key Claims
- Learns network belief representations that converge to ground-truth belief representations in discrete POMDPs
- Gives better information than observations + EnKF in continuous POMDPs
- Model-formulation agnostic (works without knowing POMDP dynamics)

**Relevance**: If DBMM can learn belief representations that converge to true beliefs, that supports claim neural nets CAN learn compact belief states. NNUE extension would add incremental-update efficiency on top.

---

## 9. Mahjong-Specific: Mortal and Suphx

### Mortal (Equim-chan, open-source)
- [GitHub](https://github.com/Equim-chan/Mortal)
- ResNet-based feature extractor ("Brain") + DQN for action selection
- 46 discrete actions (Hydra-compatible action space)
- Brain outputs: latent distribution (mu, log_sigma) in v1, direct features (phi) in v2-v4
- Has `is_oracle` mode for training with vs. without hidden information
- **Does NOT use incremental updates** -- full forward pass every decision

### Suphx (Microsoft, 2020)
- [arXiv](https://arxiv.org/abs/2003.13590)
- Global reward prediction for long-horizon credit assignment
- Oracle guiding: train with perfect info, then transfer to imperfect info
- Run-time policy adaptation (parametric Monte Carlo)
- **Does NOT use belief tracking** in NNUE sense -- uses observation history directly

---

## 10. Cicero/Diplomacy: Multiplayer Imperfect-Info with Language

### Paper
Meta FAIR. "Human-level play in game of Diplomacy." Science, 2022.
- [Science](https://www.science.org/doi/10.1126/science.ade9097)
- [GitHub](https://github.com/facebookresearch/diplomacy_cicero)

### Belief Tracking
- Uses bilateral search (`bqre1p_agent.py`) for opponent modeling
- piKL (policy-anchored KL): regularizes search policy toward learned "anchor" policy
- Searches over opponent action distributions, not explicit belief states
- Combines language-model predictions with strategic planning

**Relevance**: Cicero shows multiplayer games need opponent modeling beyond Bayes updates. Mahjong (4-player) faces same challenge.

---

## 11. EfficientZero V2 and Model-Based RL

### EfficientZero V2
- [arXiv](https://arxiv.org/abs/2403.00564)
- General framework for sample-efficient model-based RL
- Handles discrete/continuous actions, visual/low-dimensional inputs
- Still MDP-focused at core, not built for hidden-information games

### Model-Based RL for Imperfect Info (Earlier Work)
- [IEEE 2014](https://ieeexplore.ieee.org/document/6797023): Model-based RL for Hearts (card game)
- POMDP formulation with learned transition model
- Limited to single-agent approximation of multi-agent game

---

## Synthesis: The Novelty Gap

| Approach | Incremental Update | Belief Tracking | Game AI | Neural |
|---|---|---|---|---|
| NNUE (Stockfish) | YES | NO | YES (perfect info) | YES |
| ReBeL | NO | YES (Bayesian) | YES (imperfect info) | YES (value net) |
| Student of Games | NO | YES (CVPN) | YES (both) | YES |
| DreamerV3 RSSM | PARTIAL (GRU) | YES (implicit) | NO (single agent) | YES |
| BetaZero | NO | YES (particle) | YES (POMDP) | YES |
| Incremental Belief Updates | YES | YES | NO | NO |
| Deep CFR | NO | PARTIAL | YES | YES |
| **Your Proposal** | **YES** | **YES** | **YES** | **YES** |

**No existing system combines all four properties.** Proposed NNUE-inspired incremental belief net for mahjong would be first to:
1. Use sparse, incrementally-updatable feature representations (like NNUE)
2. Track belief states over hidden information (like ReBeL/BetaZero)
3. Target imperfect-information game AI (like ReBeL/SoG/Deep CFR)
4. Learn belief update function neurally (like DreamerV3 RSSM)

---

## Proposed Architecture Sketch (Based on Prior Art)

Drawing from above, hypothetical "Incremental Belief NNUE" for mahjong could work like:

### Accumulator = Belief State

```
belief_accumulator in R^d    (analogous to NNUE's accumulator)
```

### On each game event (discard, draw, call, reveal):

```
# Identify changed features (like NNUE)
removed_features = features_invalidated_by_event
added_features = features_activated_by_event

# Incremental belief update (NNUE-style)
belief_accumulator -= SUM(W_belief[:, r] for r in removed_features)
belief_accumulator += SUM(W_belief[:, a] for a in added_features)

# Bayesian correction (ReBeL-style, but learned)
belief_accumulator = belief_accumulator + delta_bayes(event, belief_accumulator)
```

### Why this could work for Mahjong specifically:

1. **Sparse changes**: Each event changes few tiles (1 discard = 1 tile revealed out of ~136)
2. **Structured hidden info**: Unknown tiles form finite, trackable set
3. **Incremental Bayes**: When player X discards tile Y, belief about their hand updates sparsely
4. **Speed**: Mahjong AI needs fast inference for real-time play; incremental updates avoid redundant compute

### Open Questions:

1. Can Bayesian correction term delta_bayes be learned end-to-end?
2. How handle "king moves" (equivalent: large state changes like kan declaration)?
3. Training: self-play with incremental updates vs. full-recompute baseline?
4. Does accumulator retain enough information for multi-step lookahead?

---

## Key References (Ranked by Relevance)

1. **ReBeL** - Brown et al., NeurIPS 2020 - [arXiv:2007.13544](https://arxiv.org/abs/2007.13544)
2. **NNUE** - Stockfish - [Docs](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)
3. **Student of Games** - Schmid et al., Science Advances 2023 - [DOI](https://www.science.org/doi/10.1126/sciadv.adg3256)
4. **DreamerV3** - Hafner et al., Nature 2025 - [DOI](https://www.nature.com/articles/s41586-025-08744-2)
5. **BetaZero** - Moss et al., RLC 2024 - [arXiv:2306.00249](https://arxiv.org/abs/2306.00249)
6. **Efficient Incremental Belief Updates** - arXiv 2024 - [arXiv:2402.06940](https://arxiv.org/abs/2402.06940)
7. **Deep CFR** - Brown et al., ICML 2019 - [arXiv:1811.00164](https://arxiv.org/abs/1811.00164)
8. **DBMM** - arXiv 2025 - [arXiv:2503.13438](https://arxiv.org/abs/2503.13438)
9. **Mortal** - Equim-chan - [GitHub](https://github.com/Equim-chan/Mortal)
10. **Suphx** - Li et al., 2020 - [arXiv:2003.13590](https://arxiv.org/abs/2003.13590)
11. **Cicero** - Meta FAIR, Science 2022 - [DOI](https://www.science.org/doi/10.1126/sciadv.adg3256)
12. **DREAM** - Steinberger et al. - [GitHub](https://github.com/EricSteinberger/DREAM)