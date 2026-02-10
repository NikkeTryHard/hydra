# Reward Design in Imperfect Information Games

A comprehensive survey of reward functions, variance reduction, and training signals across landmark AI systems. Written specifically to inform Hydra's reward design choices.

---

## Table of Contents

1. [Pluribus (Poker)](#1-pluribus-poker)
2. [ReBeL (Poker / General)](#2-rebel-poker--general)
3. [AlphaStar (StarCraft II)](#3-alphastar-starcraft-ii)
4. [OpenAI Five (Dota 2)](#4-openai-five-dota-2)
5. [Reward Variance Reduction for Mahjong (IEEE CoG 2022)](#5-reward-variance-reduction-for-mahjong-ieee-cog-2022)
6. [Cross-System Comparison](#6-cross-system-comparison)
7. [Implications for Hydra](#7-implications-for-hydra)

---

## 1. Pluribus (Poker)

**Paper:** "Superhuman AI for multiplayer poker" — Brown & Sandholm, Science 2019
**Game:** 6-player No-Limit Texas Hold'em (NLHE)

### Reward Signal

- **Terminal, zero-sum.** The reward is purely the chip gain/loss at the end of each hand. No intermediate shaping.
- **Unit:** milli-big-blinds per game (mbb/game). 1 mbb = 1/1000 of a big blind.
- **Formula:** In a multiplayer game, the human's win rate = negative of the average of the five AIs' win rates (enforcing zero-sum).

### How Luck/Variance Is Handled

Pluribus does **not** reduce variance during *training* — the CFR algorithm inherently averages over many iterations. The variance problem arises in **evaluation**:

- **AIVAT (Action-Informed Value Assessment Tool):** The key variance reduction technique for evaluation, published at AAAI 2018 by Burch et al. (U. of Alberta).
- **AIVAT Formula:**

```
AIVAT(z) = Σ_{z' ∈ W} π_Pa(z') v_p(z') / Σ_{z' ∈ W} π_Pa(z') + Σ_{H ∈ H} k_H(z)
```

Where the correction term k_H removes variance from:
1. **Chance nodes** (card deals) — subtracts "luck of the draw"
2. **Known player actions** (agent's own randomized strategy) — removes variance from mixed strategies
3. **Information exploitation** — walks backward through the game, shifting observed outcomes toward expected values

- **Result:** Reduces standard deviation by **85%**, requiring **44× fewer games** for statistical significance.
- **Multi-player adaptation:** In Pluribus experiments, AIVAT is applied to chance nodes and Pluribus's decision points only (not human decisions, since human strategies are unknown). A "Control" copy replays in the human's seat to provide a baseline.

### Per-Step vs Per-Episode

**Strictly per-episode (per-hand).** No intermediate rewards. This is possible because poker hands are short (typically <100 decision points).

### Baseline Subtraction

In CFR, regret is inherently a form of baseline subtraction:
- **Counterfactual regret** = value of action `a` − expected value under current strategy
- **Linear CFR** weights iteration T's regret by T, decaying influence of early suboptimal iterations
- **Negative-regret pruning:** After 200 minutes of training, actions with regret below −300M are skipped in 95% of iterations

### Reward Normalization

No explicit normalization. The zero-sum structure naturally constrains rewards. During evaluation, the mbb/game metric normalizes by the big blind size.

### Key Takeaway for Hydra

Pluribus shows that in short-episode zero-sum games, **pure terminal reward + CFR converges fine**. The variance problem is an *evaluation* problem, not a training problem. For Mahjong (also zero-sum, ~1 hanchan = 8-12 hands), this suggests terminal placement reward per hanchan is viable, but individual hand luck may require explicit variance reduction during training (unlike poker, Mahjong hands have far more hidden info and stochasticity).

---

## 2. ReBeL (Poker / General)

**Paper:** "Combining Deep Reinforcement Learning and Search for Imperfect-Information Games" — Brown et al., NeurIPS 2020
**Game:** Heads-up NLHE (also demonstrated on Liar's Dice)

### Reward Signal

- **Terminal, zero-sum.** Win/loss in chips per hand.
- **Unit:** tenths of a chip per hand (for evaluation).
- The game result (showdown or fold) produces the terminal payoff.

### How Luck/Variance Is Handled

ReBeL addresses variance through **architectural design** rather than explicit variance reduction:

- **Public Belief States (PBS):** Instead of reasoning about specific hidden cards, ReBeL operates on a *probability distribution* over all possible hidden states. This marginalizes out the luck of individual card deals.
- **PBS update:** After each observed action, both players' belief distributions are updated via **Bayes' theorem** based on the action and the current policy profile.
- **Value network prediction:** The value network predicts expected payoffs for the PBS (averaged over all possible hidden states), not for specific card holdings. This inherently reduces variance.

### Per-Step vs Per-Episode

**Per-episode with bootstrapping at subgame leaves:**
- Within a subgame (partial game tree explored during search), leaf node values are **predicted by the value network** rather than rolled out to completion.
- At game end (fold/showdown), realized winnings are collected and used as training labels for the value network at the root PBS.

### Baseline Subtraction

- **CFR within subgames:** Regret is computed relative to the current average strategy (inherent baseline).
- **Value network as baseline:** The value network V(PBS) acts as a soft baseline — predicted leaf values anchor the search.
- **Huber loss** for value network training (MSE produced unsatisfactory results, likely due to outlier chip swings).

### Reward Normalization

No explicit normalization beyond the zero-sum constraint. The value network learns to predict expected chips directly.

### Architecture Details

- **Value network:** 6-layer MLP, 1536 units each, GELU activation, normalization layers.
- **Inputs:** PBS, player positions, stack-to-pot ratio, board cards (rank/suit/card embeddings), action history.
- **Training buffer:** Circular buffer of 12M examples.
- **Policy initialization:** CFR iterations start from a **policy network** prediction (not uniform), accelerating convergence.

### Key Takeaway for Hydra

ReBeL's core insight is that **operating on belief distributions marginalizes away luck variance**. For Mahjong, this suggests that if we ever add search, the search should operate on "what tiles could opponents hold?" distributions, not specific tile assignments. The value network predicting over belief states is analogous to Suphx's oracle guiding — both try to inform the agent about hidden information.

---

## 3. AlphaStar (StarCraft II)

**Paper:** "Grandmaster level in StarCraft II using multi-agent reinforcement learning" — Vinyals et al., Nature 2019
**Game:** StarCraft II (full game, all 3 races)

### Reward Signal

**Sparse terminal + pseudo-rewards:**

```
r_T = +1 (win), 0 (draw), −1 (loss)     [terminal, no discount]
```

Plus **pseudo-rewards** (active with **25% probability** each during training):
1. **Build Order pseudo-reward:** edit distance between sampled human build orders and executed build orders
2. **Cumulative Statistics pseudo-reward:** Hamming distance between sampled and executed cumulative statistics (units, buildings, effects, upgrades present in a game)

### How Luck/Variance Is Handled

AlphaStar handles the enormous strategy space (which creates variance) through several mechanisms:

1. **z-statistic conditioning:** Policy π_θ(a_t | s_t, z) is conditioned on a latent variable z sampled from human replays. z encodes:
   - First 20 constructed buildings/units (build order)
   - All units/buildings/effects/upgrades present in the game (cumulative stats)
   - During SL: z = 0 (unconditional) 10% of the time
   - During RL: some agents conditioned on z, others trained unconditionally

2. **League training:** A diverse population of agents with different reward functions and matchmaking strategies. This reduces variance from strategy-cycling (A beats B, B beats C, C beats A).

3. **KL divergence penalty:** Prevents the policy from straying too far from human-like behavior, keeping it in a well-explored region of strategy space.

### Per-Step vs Per-Episode

**Per-episode (terminal reward) + per-step pseudo-rewards.**
- Terminal win/loss: undiscounted
- Pseudo-rewards: delivered per-step as the edit/Hamming distance changes

### Baseline Subtraction

- **Advantage Actor-Critic:** V_θ(s_t, z) serves as the baseline.
- **UPGO (Upgoing Policy Update):** Policy updated toward [G_t^U − V_θ(s_t, z)] where G_t^U bootstraps when the behavior policy takes a worse-than-average action. Only propagates "good" outcomes upward.
- **Centralized value function:** During training, the value function sees observations from **both player and opponent perspectives** (privileged information, like Suphx's oracle).
- **V-trace:** Off-policy correction for the mismatch between current policy and experience from older policies.

### Reward Normalization

No explicit reward normalization described. The pseudo-rewards are implicitly bounded (edit distance / Hamming distance are naturally bounded by sequence length).

### Key Takeaway for Hydra

AlphaStar's approach of **sparse terminal + stochastic pseudo-rewards** is directly relevant. For Mahjong:
- **z-conditioning ≈ strategy conditioning.** Mahjong has fewer strategies than SC2, but conditioning on "intended yaku" or "play style" could help exploration.
- **UPGO** is interesting: only update toward outcomes better than the value baseline. This would prevent a lucky tsumo on a bad hand from reinforcing bad play.
- **Centralized value (oracle) at training time** — already planned for Hydra via oracle distillation.

---

## 4. OpenAI Five (Dota 2)

**Paper:** "Dota 2 with Large Scale Deep Reinforcement Learning" — OpenAI, 2019
**Game:** Dota 2 (5v5, restricted hero pool initially)

### Reward Signal

**Dense per-tick rewards** from a linear combination of 28 hand-crafted signals (selected from 20,000 API features):

#### Individual Hero Signals

| Signal | Weight | Notes |
|--------|--------|-------|
| Hero Health | 2.0 | Quartic interpolation 0→1 |
| Mana | 0.75 | Fraction of total |
| Deny | 0.2 | |
| Last Hit | 0.16 | |
| Gold | 0.006 | Per unit gained |
| Experience | 0.002 | Per unit |
| Kill (supplemental) | −0.6 | On top of gold/XP |
| Death | −1.0 | |

#### Building Signals

Score = Weight × (1 + 2 × Health Fraction)

| Structure | Weight |
|-----------|--------|
| Ancient | 2.5 | Total win reward ≈ 10.0 |
| Barracks | 2.0 |
| Tower T3 | 1.5 |
| Tower T2 | 1.0 |
| Shrine / T1 / T4 | 0.75 |

#### Team/Meta Signals

| Signal | Weight |
|--------|--------|
| Megas (last barracks) | 4.0 |
| Win | 2.5 |
| Lane Assignment penalty | −0.02 |

The reward per tick = **change in score** from one tick to the next.

### Reward Processing Pipeline

Three transformations applied **in order**:

1. **Zero-sum enforcement:**
   ```
   hero_rewards[i] -= mean(enemy_rewards)
   ```

2. **Team Spirit (τ):** Annealed from 0.2 → 0.97 during training:
   ```
   hero_rewards[i] = τ * mean(hero_rewards) + (1 − τ) * hero_rewards[i]
   ```
   - Low τ early: individual skill learning (mechanical play, last-hitting)
   - High τ late: team coordination (sacrifice for team objectives)

3. **Time decay:** Prevents late-game domination of gradient:
   ```
   hero_rewards[i] *= 0.6^(T / 10 min)
   ```

### How Luck/Variance Is Handled

- **Zero-sum subtraction** acts as a control variate (my reward minus opponent's reward reduces common-mode variance)
- **Team Spirit annealing** reduces gradient variance in early training by giving clearer individual signals
- **Time decay** normalizes reward magnitude across game phases

### Per-Step vs Per-Episode

**Heavily per-step.** The win/loss reward (2.5) is small relative to the sum of all per-tick signals over a full game. Building health is delivered as: 2/3 linearly as health decreases + 1/3 lump sum on destruction.

### Baseline Subtraction

- **GAE (Generalized Advantage Estimation):** λ = 0.95, smoothing rewards over ~20 timesteps
- **PPO** with clipped objective for policy updates
- Data sent to optimizers every 256 timesteps

### Reward Normalization

- Zero-sum enforcement (sum of all 10 heroes' rewards = 0)
- Time decay (exponential damping)
- No explicit per-feature normalization described (weights are hand-tuned)

### Key Takeaway for Hydra

OpenAI Five demonstrates that **dense per-step shaping works at scale** but requires careful design:
- **Team Spirit** is brilliant: start selfish, end cooperative. For Mahjong (1v3), this isn't directly applicable, but a similar curriculum of "learn hand value first, then learn defensive play" could work.
- **Time decay** is relevant: in Mahjong, late-game decisions (riichi declaration with 1 tile left) shouldn't dominate gradients just because point values are higher.
- **Hand-crafted weights** — all 28 signals and weights were designed by human Dota experts and **never updated during training**. This is a critical design choice: reward shaping is fixed, only the policy learns.

---

## 5. Reward Variance Reduction for Mahjong (IEEE CoG 2022)

**Paper:** "Speedup Training Artificial Intelligence for Mahjong via Reward Variance Reduction" — Li, Wu, Fu, Fu, Zhao, Xing (Tencent AI Lab + CAS + Tsinghua), IEEE CoG 2022
**Game:** 4-player Mahjong (Chinese rules)

### The Core Problem

Mahjong reward has **extremely high variance** from two sources:
1. **Invisibility:** 3/4 of tiles are hidden (vs. ~50% in poker), making value estimation noisy
2. **Stochasticity:** The last tile drawn determines win/loss outcome, and *how* you win (tsumo vs. ron, specific tile) dramatically changes the point value

### RVR Technique

Two neural networks work together:

#### Component 1: Relative Value Network

- **Purpose:** Reduce variance from hidden information (invisibility)
- **Input:** Oracle view (all 4 players' hands — privileged information)
- **Output:** Simultaneous value estimates for all 4 players: V_θ = (V₁, V₂, V₃, V₄)
- **Zero-sum constraint:** Loss function enforces Σ V_i = 0

This is exactly **Suphx's oracle guiding** / AlphaStar's centralized value function applied to Mahjong. By seeing all hands during training, the value estimate has much lower variance than one estimated from the acting player's partial observation alone.

#### Component 2: Expected Reward Network

- **Purpose:** Reduce variance from end-of-hand stochasticity (luck)
- **Input:** Game state at round T−1 (the penultimate state before the game ends)
- **Output:** Predicted expected reward f_θ(g^{T-1})
- **Key insight:** The *last* tile draw introduces massive variance. A hand might be worth 0 or 12000 points depending on the final draw. By predicting the *expected* reward from the state just before the final draw, this filters out last-tile luck.

#### Combined Training

- During training, the raw game reward r_i is replaced with f_θ(g^{T-1}) for the RL update
- The Relative Value Network provides the baseline V(s) for advantage computation
- Together, they reduce both sources of variance simultaneously

### Exact Reward Formula

```
RL reward = f_θ(g^{T-1})    [Expected Reward Network output]
Advantage = f_θ(g^{T-1}) − V_oracle(s)   [relative to oracle value baseline]
```

### Per-Step vs Per-Episode

**Per-episode** (per-hand). The reward is the final placement/point change from one hand, but filtered through the Expected Reward Network.

### Baseline Subtraction

- **Relative Value Network** serves as the value baseline
- Zero-sum constraint ensures the four players' advantages sum to zero
- Oracle information (all tiles visible) dramatically tightens the baseline

### Reward Normalization

Not explicitly described. The zero-sum constraint naturally bounds the rewards.

### Results

- **3.7× speedup** in training convergence compared to vanilla PPO
- Achieves the same final policy quality with significantly less compute

### Key Takeaway for Hydra

This is **the most directly relevant work.** For Hydra:
1. **Oracle value baseline** (Relative Value Network) = already planned via oracle distillation
2. **Expected Reward Network** at T−1 is novel and high-value: it directly addresses Mahjong's biggest variance source (last-tile luck)
3. **Zero-sum constraint** on value estimates is cheap to implement and provably correct
4. The 3.7× speedup matters enormously for Hydra's single-GPU training constraint

---

## 6. Cross-System Comparison

| Feature | Pluribus | ReBeL | AlphaStar | OpenAI Five | RVR (Mahjong) |
|---------|----------|-------|-----------|-------------|---------------|
| **Reward type** | Terminal (chips) | Terminal (chips) | Terminal (+1/−1) + pseudo | Dense per-tick | Terminal (points) |
| **Per-step signals** | None | None | Build order, stats (25% prob) | 28 hand-crafted signals | None (but filtered) |
| **Algorithm** | CFR (MCCFR) | CFR + RL + Search | PPO (IMPALA) + V-trace | PPO + GAE | PPO |
| **Variance reduction** | AIVAT (eval only) | PBS marginalization | z-conditioning, UPGO, league | Zero-sum, team spirit, time decay | Oracle value + Expected Reward Net |
| **Baseline** | CFR regret | Value network + CFR | V(s,z) + UPGO | GAE (λ=0.95) | Oracle Relative Value Network |
| **Oracle/privileged info** | No | PBS (averaged) | Centralized value (both sides) | No | Full oracle (all hands visible) |
| **Zero-sum enforcement** | Natural (chips) | Natural (chips) | Natural (win/loss) | Explicit subtraction | Explicit constraint (Σ=0) |
| **Training scale** | 1 node, 64 cores | Small cluster | ~3000 TPUs | 256 GPUs (P100) | Single GPU viable |
| **Episode length** | ~50-100 actions | ~50-100 actions | ~50,000 steps | ~45,000 steps | ~70-140 actions/hand |

---

## 7. Hydra's Reward Function — Final Decision

Based on this cross-domain survey, Mortal source code analysis, Mortal community insights (30+ GitHub discussions), Mortal-Policy PPO fork analysis, Suphx paper extraction, RVR paper analysis, PPO best practices from CleanRL/SB3, and scoring system comparison across all major platforms:

### The Formula

```
Episode = one kyoku (round), NOT one hanchan (game)

Per-kyoku reward:
  r_k = E[pts]_after_kyoku_k - E[pts]_before_kyoku_k

Where:
  E[pts] = rank_prob · pts_vector
  rank_prob = marginalized from GRP's 24-class permutation softmax
  pts_vector = [3, 1, -1, -3]  (symmetric, zero-sum, configurable)

Per-action Q-target (within kyoku k):
  All actions in kyoku k share reward r_k
  Credit assignment handled by GAE + value function, not reward

Advantage computation:
  A(o_t, a_t) = f_θ(g^{T-1}) - V_oracle(s_full_t)

Where:
  f_θ(g^{T-1}) = Expected Reward Network output (replaces raw r for last kyoku)
  V_oracle(s_full_t) = Oracle critic seeing all 136 tiles (training only)
  Zero-sum constraint: Σ_i V_i(s) = 0

Normalization:
  1. Running std normalization on rewards (Welford algorithm, do NOT subtract mean)
  2. Per-minibatch advantage normalization: (A - mean) / (std + 1e-8)
  3. Clip normalized rewards to [-5, 5]
```

### Why This Design

| Decision | Choice | Evidence |
|----------|--------|----------|
| **Episode boundary** | Per-kyoku | Both Mortal and Suphx use this. ~100× lower variance than per-game. |
| **Reward signal** | GRP ΔE[pts] | Mortal's proven approach. Equivalent to potential-based reward shaping (Ng 1999) — policy-invariant. |
| **Placement points** | [3, 1, -1, -3] | Mortal's training default. Symmetric, zero-sum. Each rank step = 2 pts. Platform-specific via config swap. |
| **GRP design** | 24-class permutation softmax | Captures inter-player rank correlations. 4-class loses this. Mortal proved it works. |
| **Discount γ** | 1.0 | Mortal uses γ=1. Kyoku is short enough (~15 steps). No need for temporal discounting. |
| **Variance reduction** | Oracle critic + ERN | RVR paper: 3.7× speedup. Attacks both variance sources (hidden info + last-tile luck). |
| **GRP lifecycle** | Pretrained, frozen during RL | Stable reward signal. Mortal does this. Avoids moving-target problem. |
| **Reward normalization** | Running std (Welford) | Mortal-Policy's exact approach. Essential for PPO in high-variance games. |
| **No reward shaping** | Skip (GRP delta IS PBRS already) | Double-shaping adds risk. Shanten-based shaping creates offensive bias — worst possible for Mahjong. |
| **No intrinsic motivation** | Skip | SL warm-start solves exploration. RND/ICM would add noise from tile draw stochasticity. |
| **Same reward all phases** | Mandatory | Changing reward invalidates value function. Cal-QL (NeurIPS 2023) showed this causes "unlearning." |

### Implementation Priority

1. **Phase 1 (SL warm-start):** No reward needed — cross-entropy on expert actions
2. **Phase 2 (Oracle RL):** GRP ΔE[pts] reward + oracle critic with zero-sum constraint
3. **Phase 3 (League):** Same reward + Expected Reward Network for last-kyoku variance
4. **Phase 3+ (Optional):** HCA (Hindsight Credit Assignment) to reweigh returns — research-grade

### Confirmed Anti-Patterns (From Mortal Community)

| What | Why Not |
|------|---------|
| TD bootstrapping | Mortal tried it → no improvement, adds instability (Discussion #81) |
| PPO without reward normalization | Mortal creator spent months on PPO, failed — likely due to unnormalized reward variance (Discussion #102) |
| Oracle guiding of policy | "Optimal play with full info doesn't transfer to blind play" (Discussion #102, zl0v7hzr) |
| Deeper networks | Mortal tried b75c256 → no improvement |
| Sample reuse in online training | Made things worse (Discussion #64) |
| Dense per-step shaping | Mahjong has ~15 decisions/kyoku — not 45,000 like Dota. Dense shaping adds complexity for no benefit. |
| Elo as reward signal | Non-stationary, noisy, redundant with placement points |

### Platform-Specific Fine-Tuning (Via pts_vector Swap)

| Target Platform | pts_vector | Strategy Bias |
|----------------|------------|---------------|
| General training | [3, 1, -1, -3] | Balanced (default) |
| Tenhou Houou | [3, 1.5, 0, -4.5] | Avoid 4th (normalized Tenhou net pts) |
| Mahjong Soul Throne | [3, 1, -1, -3] | Balanced (Majsoul uma is already nearly symmetric) |
| WRC / EMA tournament | [3, 1, -1, -3] | Balanced (identical incentive structure) |
| M-League style | [5, 1, -1, -3] | Push for 1st |

---

## References

| Ref | Paper | Year | Venue |
|-----|-------|------|-------|
| [1] | Brown & Sandholm, "Superhuman AI for multiplayer poker" | 2019 | Science |
| [2] | Burch et al., "AIVAT: A New Variance Reduction Technique for Agent Evaluation in Imperfect Information Games" | 2018 | AAAI |
| [3] | Brown et al., "Combining Deep Reinforcement Learning and Search for Imperfect-Information Games" (ReBeL) | 2020 | NeurIPS |
| [4] | Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement learning" (AlphaStar) | 2019 | Nature |
| [5] | OpenAI, "Dota 2 with Large Scale Deep Reinforcement Learning" (OpenAI Five) | 2019 | arXiv |
| [6] | Li et al., "Speedup Training Artificial Intelligence for Mahjong via Reward Variance Reduction" | 2022 | IEEE CoG |
| [7] | Mathieu et al., "AlphaStar Unplugged: Large-Scale Offline Reinforcement Learning" | 2023 | arXiv |
| [8] | Farhi, "Dota Reward Function" (appendix gist) | 2019 | OpenAI internal |
| [9] | Ng et al., "Policy invariance under reward transformations" | 1999 | ICML |
| [10] | Harutyunyan et al., "Hindsight Credit Assignment" | 2019 | NeurIPS |
| [11] | Li et al., "Suphx: Mastering Mahjong with Deep Reinforcement Learning" | 2020 | arXiv |
| [12] | Nakamoto et al., "Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning" | 2023 | NeurIPS |
| [13] | Ball et al., "Efficient Online Reinforcement Learning with Offline Data" (RLPD) | 2023 | ICML |
| [14] | Engstrom et al., "Implementation Matters in Deep Policy Gradients" | 2020 | ICLR |
| [15] | Huang, "The 37 Implementation Details of Proximal Policy Optimization" | 2022 | Blog/ICLR |
| [16] | Peng et al., "Advantage-Weighted Regression" (AWR) | 2019 | arXiv |
