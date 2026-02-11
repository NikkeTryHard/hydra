# Hydra Ablation Plan

## Methodology

### Evaluation Protocol

All ablation experiments use the 1v3 duplicate format defined in [INFRASTRUCTURE.md § Rating and Evaluation](INFRASTRUCTURE.md#rating-and-evaluation). The challenger (treatment model) plays against 3 copies of the champion (control model). Each game seed is played 4 times with the challenger rotating through East/South/West/North seats, controlling for positional advantage and tile draw variance.

Rank point distribution: [90, 45, 0, -135] (Tenhou Houou-style uma).

### Sample Sizes

> Sample size tiers and statistical significance requirements are defined in [INFRASTRUCTURE.md § Rating and Evaluation](INFRASTRUCTURE.md#rating-and-evaluation). The tiers used for ablations:

| Tier | Sets | Games (x4 rotations) | Purpose |
|------|------|----------------------|---------|
| Quick | 1,000 | 4,000 | Trend detection during development |
| Full | 50,000 | 200,000 | Publication-quality comparison |
| Ablation | 250,000 | 1,000,000 | Detecting effects < 1 rank-pt/game |

All ablation experiments default to the **full** tier (200K games) unless the expected effect size is small (< 1 rank-pt/game), in which case the **ablation** tier (1M games) is used.

### Statistical Framework

- **Primary test:** Welch's t-test on per-game rank points (p < 0.05).
- **Effect size:** Mahjong has high per-game variance (~sigma = 80 rank pts). Detecting a 1 rank-pt/game improvement at 95% confidence requires ~100K games in 1v3 duplicate format.
- **Reporting:** IQM with bootstrap confidence intervals (Agarwal et al. 2021), as specified in [SEEDING.md § Known Limitations](SEEDING.md#known-limitations).
- **Multiple comparisons:** Bonferroni correction when running > 3 experiments simultaneously (adjusted alpha = 0.05 / k).

### Baseline

All experiments use the Phase 1 BC model (`bc_best.pt`) as the control baseline unless otherwise specified. This is the first checkpoint that passes the Phase 1 readiness gate (discard accuracy >= 65%, placement <= 2.55, deal-in <= 15%).

---

## Priority-Ordered Ablation Queue

| ID | Hypothesis | Control | Treatment | Primary Metric | Sample Size | Gates | Priority |
|----|-----------|---------|-----------|---------------|-------------|-------|----------|
| A1 | Safety planes (ch 61-83) reduce deal-in rate by >= 1pp | 61ch input (no safety planes) | 84ch input (full safety planes) | Deal-in rate | Full (200K) | A2, A3 | P0 |
| A2 | Tenpai aux head improves damaten deal-in avoidance | No tenpai head | Full tenpai head (sigmoid(3)) | Deal-in rate vs non-riichi opponents | Full (200K) | None | P0 |
| A3 | Danger head improves tile-level deal-in avoidance | No danger head | Full danger head with focal loss | Deal-in rate, win/deal-in ratio | Full (200K) | None | P0 |
| A4 | Oracle distillation (KD) improves implicit opponent reading | Skip Phase 2 (BC -> League) | Full 3-stage pipeline (BC -> Oracle -> League) | Avg placement | Ablation (1M) | None | P1 |
| A5 | Uncapped scores + overtake thresholds improve orasu pushing | Mortal-style dual-scale (100K/30K) | Hydra uncapped + gaps + thresholds | 1st place rate in South 4 | Full (200K) | None | P1 |
| A6 | Joint 24-class GRP captures placement correlations better | Scalar GRP (MSE regression) | 24-class GRP (CE over permutations) | Placement-aware decision quality (orasu rank-pt delta) | Full (200K) | None | P1 |
| A7 | Focal loss (gamma=2.0) outperforms weighted BCE for rare events | BCEWithLogitsLoss(pos_weight=10) | Focal BCE (alpha=0.25, gamma=2.0) | Danger head AUC | Quick (4K) | None | P2 |
| A8 | ERN reduces last-tile variance and accelerates convergence | Raw terminal reward | ERN-smoothed reward | Value loss variance, convergence speed | Full (200K) | A4 | P2 |

**Execution order:** A1-A3 run in parallel (independent, no gates). A4-A6 run after A1-A3 results are known. A7 is a quick standalone experiment. A8 requires A4 results (it only matters if oracle distillation is used).

---

## Per-Experiment Protocol

### Template

Each experiment follows this protocol:

1. **Training duration:** Match the baseline training budget exactly. For BC experiments (A1-A3, A5-A7), train for 3 epochs on the filtered dataset (~480K steps at batch 2048). For pipeline experiments (A4, A8), train through all relevant phases with the same step budgets specified in INFRASTRUCTURE.md.

2. **Controlled variables (hold constant):**
   - Master seed (use 5 seeds per experiment for confidence intervals)
   - Training data (same filtered manifest, same chronological split)
   - Batch size (2048 for BC, 4096 for PPO minibatches)
   - Hardware (RTX PRO 6000 Blackwell)
   - All hyperparameters not under test
   - Evaluation seed bank (`data/eval_seeds.json`)

3. **Measurement:**
   - Quick eval (4K games) every 50K training steps during training for trend monitoring
   - Full eval (200K games) at the end of training for the final comparison
   - Log all metrics to W&B with experiment tags for cross-run comparison

4. **Reporting format:**

   | Metric | Control (mean +/- CI) | Treatment (mean +/- CI) | Delta | p-value | Significant? |
   |--------|----------------------|------------------------|-------|---------|-------------|
   | Avg placement | X.XX +/- 0.XX | X.XX +/- 0.XX | +/- X.XX | 0.XXX | Yes/No |
   | Deal-in rate | XX.X% +/- X.X% | XX.X% +/- X.X% | +/- X.X% | 0.XXX | Yes/No |
   | Win rate | XX.X% +/- X.X% | XX.X% +/- X.X% | +/- X.X% | 0.XXX | Yes/No |
   | 1st place rate | XX.X% +/- X.X% | XX.X% +/- X.X% | +/- X.X% | 0.XXX | Yes/No |

   Use IQM with bootstrap CI (10,000 resamples) for all rank-point metrics.

---

## Experiment Details

### A1: Safety Planes

**Hypothesis:** Explicit safety planes (genbutsu, suji, kabe — channels 61-83) reduce deal-in rate by at least 1 percentage point compared to a model that must learn these patterns implicitly from raw tile counts.

**Control model:** Modify the stem to `Conv1d(61, 256, 3)`. Channels 61-83 are zeroed out during encoding. All other architecture and training identical.

**Treatment model:** Standard 84-channel Hydra architecture with full safety plane encoding.

**Secondary metrics:** Win rate (safety planes should not reduce aggression excessively), danger head AUC (with vs. without precomputed safety features as input), tenpai head AUC.

**Success criterion:** Deal-in rate drops by >= 1pp without win rate dropping by more than 0.5pp. If safety planes reduce deal-in but also reduce win rate significantly, they are a defensive crutch, not a genuine improvement.

### A2: Tenpai Head

**Hypothesis:** An explicit tenpai prediction head improves the model's ability to detect damaten (hidden tenpai) opponents and avoid dealing into them.

**Control model:** Remove the tenpai head entirely. Remove tenpai BCE from the composite loss.

**Treatment model:** Full tenpai head (sigmoid(3)) with BCE loss, coefficient 0.1 (2× the training default of 0.05 — testing whether a stronger tenpai signal improves damaten detection).

**Measurement focus:** Deal-in rate specifically against non-riichi opponents (riichi is trivially detectable). Use the evaluation log to partition deal-ins by whether the winning opponent had declared riichi.

### A3: Danger Head

**Hypothesis:** An explicit per-tile, per-opponent danger head provides stronger learning signal for defensive play than relying solely on policy gradients.

**Control model:** Remove the danger head. Remove danger focal loss from the composite loss.

**Treatment model:** Full danger head (sigmoid(3x34)) with focal BCE (alpha=0.25, gamma=2.0), coefficient 0.2 (4× the training default of 0.05 — testing whether a louder initial danger signal helps PID-Lagrangian λ converge faster).

**Measurement focus:** Win/deal-in ratio (should increase), deal-in rate (should decrease), and qualitative analysis of whether the model avoids dangerous tiles when danger head probability is high.

### A4: Oracle Distillation

**Hypothesis:** The full 3-stage pipeline (BC -> Oracle Distillation -> League) produces a stronger final model than skipping oracle distillation (BC -> League directly).

**Control model:** Train BC model, skip Phase 2, go directly to Phase 3 league self-play.

**Treatment model:** Full 3-stage pipeline as specified in [HYDRA_SPEC.md](HYDRA_SPEC.md) (architecture) and [TRAINING.md](TRAINING.md) (training pipeline).

**This is the most expensive ablation** — it requires training through all phases twice. Use the ablation tier (1M games) for evaluation because the expected effect may be subtle (oracle distillation primarily improves implicit opponent reading, which manifests as small improvements across many game situations rather than large improvements in specific scenarios).

### A5: Score Context Vector

**Hypothesis:** Hydra's uncapped score encoding with pairwise gaps and overtake thresholds produces better orasu (final round) decisions than Mortal's dual-scale encoding.

**Control model:** Replace channels 46-53 with Mortal-style dual-scale encoding: `score / 100000` and `score / 30000` (capped at 1.0). Remove overtake threshold channels.

**Treatment model:** Standard Hydra encoding: uncapped raw scores, relative gaps, overtake thresholds (16-dim score context vector for GRP head).

**Measurement focus:** 1st place rate specifically in South 4 (the round where placement decisions are most critical). Also measure frequency of riichi declarations in South 4 when the player is in 1st place with a narrow lead (should be lower with better score awareness).

### A6: GRP 24-way vs Scalar

**Hypothesis:** Mortal's 24-class joint rank distribution captures inter-player placement correlations that a scalar GRP predictor (Suphx-style) cannot represent.

**Control model:** Replace the 24-class GRP head with a scalar regression head: `MLP(272 -> 256 -> 128 -> 1)`, trained with MSE on expected final game reward.

**Treatment model:** Standard 24-class GRP head with CE loss over the 4! = 24 rank permutations.

**Measurement focus:** Placement-aware decision quality, measured as the rank-point delta in orasu situations where the model's placement is contested (within 8000 points of an adjacent rank).

### A7: Focal Loss vs Weighted BCE

**Hypothesis:** Focal loss (gamma=2.0) produces a better danger head than standard weighted BCE, because deal-in events are rare (~12-15% of rounds) and focal loss down-weights easy negatives.

**Control model:** `BCEWithLogitsLoss(pos_weight=10)` for the danger head.

**Treatment model:** Focal BCE with alpha=0.25, gamma=2.0.

**This is a quick experiment** — only the danger head loss function changes. Use the quick eval tier (4K games) for initial screening, then full eval if the result is promising. Primary metric is danger head AUC on a held-out evaluation set, not downstream play strength.

### A8: Expected Reward Network

**Hypothesis:** Replacing the raw terminal reward in the final kyoku with a predicted expected reward (ERN) reduces value loss variance and accelerates Phase 3 convergence.

**Control model:** Standard terminal reward: placement points [+3, +1, -1, -3] per kyoku.

**Treatment model:** ERN predicts E[placement_points | state, all_hands] for the last kyoku. The ERN prediction replaces the raw terminal reward. The ERN is trained separately on completed games with oracle information, frozen during RL training.

**Measurement focus:** Value loss variance (should decrease by >= 2x), convergence speed (how many PPO updates to reach the same eval strength), and final model quality (should be at least equal to control).

**Gate:** Depends on A4 results. If oracle distillation shows no benefit, the ERN (which also uses oracle information) may not be worth the complexity.

---

## Gating Structure

```mermaid
graph LR
    subgraph "P0: Foundation (Parallel)"
        A1[A1: Safety Planes]
        A2[A2: Tenpai Head]
        A3[A3: Danger Head]
    end

    subgraph "P1: Architecture (After P0)"
        A4[A4: Oracle Distillation]
        A5[A5: Score Context]
        A6[A6: GRP 24-way]
    end

    subgraph "P2: Refinement"
        A7[A7: Focal Loss]
        A8[A8: ERN]
    end

    A1 --> A4
    A1 --> A5
    A1 --> A6
    A2 --> A4
    A3 --> A4
    A4 --> A8
```

**Reading the diagram:** Arrows indicate "results from X inform the design of Y." A1-A3 run first because their results determine what the baseline model looks like for all subsequent experiments. A4-A6 run after P0 results are incorporated. A8 depends on A4 because the ERN only makes sense if oracle distillation is part of the pipeline.

---

## Open Questions from HYDRA_SPEC

[TRAINING.md § Open Questions](TRAINING.md#open-questions) identifies five unresolved design questions. Each maps to a specific ablation experiment:

| Open Question | Ablation | How It Resolves the Question |
|--------------|----------|------------------------------|
| 1. GRP Horizon: final game rank vs next round rank? | A6 | Compares 24-class (final game) vs scalar (can be configured for either). If scalar performs equally, the simpler approach wins. |
| 2. Safety Plane Utility: do explicit suji/kabe planes help? | A1 | Directly tests 61ch (no safety) vs 84ch (full safety). |
| 3. Tedashi Encoding: channel-only vs GRU head? | Not in current queue | Deferred — requires architectural change to the backbone. Add as A9 if resources permit. |
| 4. Distillation Duration: when does teacher knowledge saturate? | A4 | If oracle distillation shows no benefit (BC -> League equals BC -> Oracle -> League), duration is moot. If it helps, monitor KL divergence curve during Phase 2 training to find the saturation point. |
| 5. Aggression Balance: how to prevent oracle-guided passivity? | A4 + monitoring | Track win/deal-in ratio during Phase 2 training. Healthy range is 2:1 to 2.5:1 per [TRAINING.md](TRAINING.md). If the ratio exceeds 3:1, the agent is too passive. |
