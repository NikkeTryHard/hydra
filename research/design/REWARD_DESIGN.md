# Hydra Reward Design

> **Status note:** mixed design/reference doc. Keep reward-analysis evidence + reserve ideas here. Active-path doctrine: `research/design/HYDRA_RECONCILIATION.md`. Runtime truth: `docs/GAME_ENGINE.md`, `docs/COMPATIBILITY_SURFACE.md`, current code.
>
> Do not treat older `TRAINING.md` refs as current governing doctrine.

Hydra reward fn design, informed by cross-domain analysis of reward systems in Pluribus, ReBeL, AlphaStar, OpenAI Five, RVR Mahjong.

> **Background reading:** full literature survey of reward fns across landmark AI systems not preserved as standalone archive file in this repo; treat refs + analysis below as surviving summary surface.

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

Mahjong reward has **extremely high variance** from 2 sources:
1. **Invisibility:** 3/4 tiles hidden (vs. ~50% in poker), making value estimation noisy
2. **Stochasticity:** last tile drawn determines win/loss outcome, and *how* win happens (tsumo vs. ron, specific tile) drastically changes point value

### RVR Technique

2 neural nets work together:

#### Component 1: Relative Value Network

- **Purpose:** reduce variance from hidden info (invisibility)
- **Input:** oracle view (all 4 players' hands — privileged info)
- **Output:** simultaneous value estimates for all 4 players: V_θ = (V₁, V₂, V₃, V₄)
- **Zero-sum constraint:** loss fn enforces Σ V_i = 0

This is **Suphx's oracle guiding** / AlphaStar's centralized value fn applied to Mahjong. By seeing all hands during training, value estimate has far lower variance than one estimated from acting player's partial observation alone.

#### Component 2: Expected Reward Network

- **Purpose:** reduce variance from end-of-hand stochasticity (luck)
- **Input:** game state at round T−1 (penultimate state before game ends)
- **Output:** predicted expected reward f_θ(g^{T-1})
- **Key insight:** *last* tile draw introduces massive variance. Hand might be worth 0 or 12000 points depending on final draw. Predicting *expected* reward from state before final draw filters out last-tile luck.

#### Combined Training

- During training, raw game reward r_i replaced with f_θ(g^{T-1}) for RL update
- Relative Value Network provides baseline V(s) for advantage computation
- Together, they reduce both variance sources at once

### Exact Reward Formula

```
RL reward = f_θ(g^{T-1})    [Expected Reward Network output]
Advantage = f_θ(g^{T-1}) − V_oracle(s)   [relative to oracle value baseline]
```

### Per-Step vs Per-Episode

**Per-episode** (per-hand). Reward is final placement/point change from one hand, but filtered through Expected Reward Network.

### Baseline Subtraction

- **Relative Value Network** serves as value baseline
- Zero-sum constraint ensures 4 players' advantages sum to zero
- Oracle info (all tiles visible) sharply tightens baseline

### Reward Normalization

Not explicitly described. Zero-sum constraint naturally bounds rewards.

### Results

- Reported faster training convergence vs vanilla PPO (paper describes "speedup" qualitatively but gives no specific multiplier)
- Reaches same final policy quality with significantly less compute

### Key Takeaway for Hydra

This is **most directly relevant work.** For Hydra:
1. **Oracle value baseline** (Relative Value Network) = already planned via oracle distillation
2. **Expected Reward Network** at T−1 is novel + high-value: directly attacks Mahjong's biggest variance source (last-tile luck)
3. **Zero-sum constraint** on value estimates is cheap to implement + provably correct
4. Convergence speedup matters greatly for Hydra's single-GPU training constraint

---

## 2. Hydra's Reward Function — Final Decision

Based on earlier cross-domain survey work (no longer preserved here as standalone `archive/REWARD_SURVEY.md` file), Mortal source code analysis, Mortal community insights (30+ GitHub discussions), Mortal-Policy PPO fork analysis, Suphx paper extraction, RVR paper analysis, PPO best practices from CleanRL/SB3, and scoring system comparison across all major platforms:

### The Formula

Exact reward formula + impl priority should be treated as active only when promoted by reconciled doctrine. Keep analysis below as reference/evidence, not hidden source of authority.

### Why This Design

| Decision | Choice | Evidence |
|----------|--------|----------|
| **Episode boundary** | Per-kyoku | Both Mortal and Suphx use this. ~100× lower variance than per-game. |
| **Reward signal** | GRP ΔE[pts] | Mortal's proven approach. Equivalent to potential-based reward shaping (Ng 1999) — policy-invariant. |
| **Placement points** | [3, 1, -1, -3] | Mortal's training default. Symmetric, zero-sum. Each rank step = 2 pts. Platform-specific via config swap. |
| **GRP design** | 24-class permutation softmax | Captures inter-player rank correlations. 4-class loses this. Mortal proved it works. |
| **Discount γ** | 1.0 | Mortal uses γ=1. Kyoku is short enough (~15 steps). No temporal discount needed. |
| **Variance reduction** | Oracle critic + ERN | RVR paper: significant speedup. Attacks both variance sources (hidden info + last-tile luck). |
| **GRP lifecycle** | Pretrained, frozen during RL | Stable reward signal. Mortal does this. Avoids moving-target problem. |
| **Reward normalization** | Running std (Welford) | Mortal-Policy's exact approach. Essential for PPO in high-variance games. |
| **No reward shaping** | Skip (GRP delta IS PBRS already) | Double-shaping adds risk. Shanten-based shaping creates offensive bias — worst possible for Mahjong. |
| **No intrinsic motivation** | Skip | SL warm-start solves exploration. RND/ICM would add noise from tile draw stochasticity. |
| **Same reward all phases** | Mandatory | Changing reward invalidates value fn. Cal-QL (NeurIPS 2023) showed this causes "unlearning." |

### Confirmed Anti-Patterns (From Mortal Community)

Anti-pattern list below retained as reference guidance; do not treat dead `TRAINING.md` links as live authority.

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
| [14] | Engstrom et al., "impl Matters in Deep Policy Gradients" | 2020 | ICLR |
| [15] | Huang, 37 impl Details of Proximal Policy Optimization" | 2022 | Blog/ICLR |

> Refs [1]-[5] and [7]-[8] come from earlier cross-domain reward survey work, but standalone archive file not present in this repo.