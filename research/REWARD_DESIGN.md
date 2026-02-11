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

- Reported faster training convergence compared to vanilla PPO (the paper describes "speedup" qualitatively but does not state a specific speedup multiplier)
- Achieves the same final policy quality with significantly less compute

### Key Takeaway for Hydra

This is **the most directly relevant work.** For Hydra:
1. **Oracle value baseline** (Relative Value Network) = already planned via oracle distillation
2. **Expected Reward Network** at T−1 is novel and high-value: it directly addresses Mahjong's biggest variance source (last-tile luck)
3. **Zero-sum constraint** on value estimates is cheap to implement and provably correct
4. The convergence speedup matters enormously for Hydra's single-GPU training constraint

---

## 2. Hydra's Reward Function — Final Decision

Based on cross-domain survey ([archived](archive/REWARD_SURVEY.md)), Mortal source code analysis, Mortal community insights (30+ GitHub discussions), Mortal-Policy PPO fork analysis, Suphx paper extraction, RVR paper analysis, PPO best practices from CleanRL/SB3, and scoring system comparison across all major platforms:

### The Formula

> See [TRAINING.md § Reward Function](TRAINING.md#reward-function) for the authoritative formula, normalization, and implementation priority.

### Why This Design

| Decision | Choice | Evidence |
|----------|--------|----------|
| **Episode boundary** | Per-kyoku | Both Mortal and Suphx use this. ~100× lower variance than per-game. |
| **Reward signal** | GRP ΔE[pts] | Mortal's proven approach. Equivalent to potential-based reward shaping (Ng 1999) — policy-invariant. |
| **Placement points** | [3, 1, -1, -3] | Mortal's training default. Symmetric, zero-sum. Each rank step = 2 pts. Platform-specific via config swap. |
| **GRP design** | 24-class permutation softmax | Captures inter-player rank correlations. 4-class loses this. Mortal proved it works. |
| **Discount γ** | 1.0 | Mortal uses γ=1. Kyoku is short enough (~15 steps). No need for temporal discounting. |
| **Variance reduction** | Oracle critic + ERN | RVR paper: significant speedup. Attacks both variance sources (hidden info + last-tile luck). |
| **GRP lifecycle** | Pretrained, frozen during RL | Stable reward signal. Mortal does this. Avoids moving-target problem. |
| **Reward normalization** | Running std (Welford) | Mortal-Policy's exact approach. Essential for PPO in high-variance games. |
| **No reward shaping** | Skip (GRP delta IS PBRS already) | Double-shaping adds risk. Shanten-based shaping creates offensive bias — worst possible for Mahjong. |
| **No intrinsic motivation** | Skip | SL warm-start solves exploration. RND/ICM would add noise from tile draw stochasticity. |
| **Same reward all phases** | Mandatory | Changing reward invalidates value function. Cal-QL (NeurIPS 2023) showed this causes "unlearning." |

### Confirmed Anti-Patterns (From Mortal Community)

> See [TRAINING.md § What NOT to Do](TRAINING.md#what-not-to-do-confirmed-failures) for the full list.

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

> For references [1]-[5], [7]-[8] (Pluribus, ReBeL, AlphaStar, OpenAI Five, and related works), see [archive/REWARD_SURVEY.md](archive/REWARD_SURVEY.md#references).
