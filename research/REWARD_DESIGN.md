# Hydra Reward Design

Hydra's reward function design, informed by cross-domain analysis of reward systems in Pluribus, ReBeL, AlphaStar, OpenAI Five, and RVR Mahjong.

> **Background reading:** The full literature survey of reward functions across landmark AI systems is archived in [archive/REWARD_SURVEY.md](archive/REWARD_SURVEY.md).

---

## Table of Contents

1. [Reward Variance Reduction for Mahjong (IEEE CoG 2022)](#1-reward-variance-reduction-for-mahjong-ieee-cog-2022)
2. [Hydra's Reward Function — Final Decision](#2-hydras-reward-function--final-decision)
3. [References](#references)

---

## 1. Reward Variance Reduction for Mahjong (IEEE CoG 2022)

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

## 2. Hydra's Reward Function — Final Decision

Based on cross-domain survey ([archived](archive/REWARD_SURVEY.md)), Mortal source code analysis, Mortal community insights (30+ GitHub discussions), Mortal-Policy PPO fork analysis, Suphx paper extraction, RVR paper analysis, PPO best practices from CleanRL/SB3, and scoring system comparison across all major platforms:

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
| [6] | Li et al., "Speedup Training Artificial Intelligence for Mahjong via Reward Variance Reduction" | 2022 | IEEE CoG |
| [9] | Ng et al., "Policy invariance under reward transformations" | 1999 | ICML |
| [10] | Harutyunyan et al., "Hindsight Credit Assignment" | 2019 | NeurIPS |
| [11] | Li et al., "Suphx: Mastering Mahjong with Deep Reinforcement Learning" | 2020 | arXiv |
| [12] | Nakamoto et al., "Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning" | 2023 | NeurIPS |
| [14] | Engstrom et al., "Implementation Matters in Deep Policy Gradients" | 2020 | ICLR |
| [15] | Huang, "The 37 Implementation Details of Proximal Policy Optimization" | 2022 | Blog/ICLR |

> For references [1]-[5], [7]-[8], [13], [16] (Pluribus, ReBeL, AlphaStar, OpenAI Five, and related works), see [archive/REWARD_SURVEY.md](archive/REWARD_SURVEY.md#references).
