# Reward Design Survey: Cross-Domain AI Systems

Archived literature survey of reward functions, variance reduction, and training signals across landmark AI systems. Originally part of [REWARD_DESIGN.md](../REWARD_DESIGN.md) — moved here to keep the main document focused on Hydra's actual reward design decisions.

> **For Hydra's reward function design, see [REWARD_DESIGN.md](../REWARD_DESIGN.md).**

---

## Table of Contents

1. [Pluribus (Poker)](#1-pluribus-poker)
2. [ReBeL (Poker / General)](#2-rebel-poker--general)
3. [AlphaStar (StarCraft II)](#3-alphastar-starcraft-ii)
4. [OpenAI Five (Dota 2)](#4-openai-five-dota-2)
5. [Cross-System Comparison](#5-cross-system-comparison)

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

## 5. Cross-System Comparison

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

## References

| Ref | Paper | Year | Venue |
|-----|-------|------|-------|
| [1] | Brown & Sandholm, "Superhuman AI for multiplayer poker" | 2019 | Science |
| [2] | Burch et al., "AIVAT: A New Variance Reduction Technique for Agent Evaluation in Imperfect Information Games" | 2018 | AAAI |
| [3] | Brown et al., "Combining Deep Reinforcement Learning and Search for Imperfect-Information Games" (ReBeL) | 2020 | NeurIPS |
| [4] | Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement learning" (AlphaStar) | 2019 | Nature |
| [5] | OpenAI, "Dota 2 with Large Scale Deep Reinforcement Learning" (OpenAI Five) | 2019 | arXiv |
| [7] | Mathieu et al., "AlphaStar Unplugged: Large-Scale Offline Reinforcement Learning" | 2023 | arXiv |
| [8] | Farhi, "Dota Reward Function" (appendix gist) | 2019 | OpenAI internal |
