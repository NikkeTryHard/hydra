# Training-Time Techniques for Stronger Game AI

> Techniques that improve the SAME network's strength without changing architecture or adding inference-time search.

**Date**: 2026-03-03
**Scope**: Auxiliary targets, data augmentation, curriculum learning, population-based training, representation learning, reward shaping, distillation, ensembles.

---

## Summary Table

| Technique | Measured Gain | Source | Confidence |
|-----------|--------------|--------|------------|
| Auxiliary ownership+score targets | +190 Elo / 1.65x faster convergence | KataGo (Wu 2019) | HIGH |
| Dihedral data augmentation (8x) | Used by AlphaGo Zero, not separately ablated | Silver et al. 2017 | MEDIUM |
| Suit permutation (6x) for Mahjong | No published ablation | -- | THEORETICAL |
| Reward shaping (dense vs sparse) | ~10x faster training, +20 TrueSkill | OpenAI Five (2018) | HIGH |
| Reward shaping (bonus for Mahjong) | +$1.37/game net earnings | Chen & Lai 2023 | MEDIUM |
| Population-based training (League) | +284 Elo over full pipeline via exploiters | AlphaStar (Vinyals 2019) | HIGH |
| Global reward prediction | Qualitative improvement (better rank) | Suphx (Li et al. 2020) | MEDIUM |
| Policy distillation | 4x smaller = 108% of teacher performance | Rusu et al. 2015 | HIGH |
| Self-predictive representations | 0.415 vs 0.175 median HNS (137% gain) | SPR (Schwarzer 2021) | MEDIUM |
| Curriculum (endgame-first) | Faster early convergence, same asymptote | McAleer et al. 2019 | LOW |
| Ensemble model averaging | ~+50-100 Elo typical in chess (anecdotal) | Community estimates | LOW |

---

## 1. Auxiliary Prediction Targets

### 1.1 KataGo: Ownership + Score Prediction (STRONGEST EVIDENCE)

**Source**: [KataGo paper (Wu, 2019)](https://arxiv.org/abs/1902.10565)

KataGo adds two auxiliary heads beyond win/loss prediction:
- **Ownership head**: Predicts per-intersection territory ownership at game end (361-dim output for 19x19)
- **Score head**: Predicts final score difference (scalar)

**Measured gains** (from ablation "NoVAux" -- removing both):

| Config | Elo at 2.5G queries |
|--------|---------------------|
| Full (with aux) | 1329 |
| NoVAux (without aux) | 1139 |
| **Delta** | **+190 Elo** |

- **Convergence speedup**: 1.65x (need ~65% more compute without aux targets to reach same strength)
- These are training-only targets -- they provide richer gradient signal without changing inference behavior

**Why it works**: Win/loss is a single bit of feedback per game. Ownership gives 361 localized training signals per position -- the network gets told "you were wrong about THIS intersection." Score prediction forces understanding of relative advantages rather than just binary outcomes.

**Relevance to Hydra**: Hydra already plans GRP (24-way Game Result Prediction), Tenpai (3), and Danger (3x34) heads -- this is the same principle. The KataGo evidence suggests these auxiliary heads could be worth +100-200 Elo equivalent in Mahjong.

### 1.2 Suphx: Global Reward Prediction

**Source**: [Suphx paper (Li et al., 2020)](https://arxiv.org/abs/2003.13590)

Suphx uses "global reward prediction" -- predicting the final game-level placement from intermediate states, not just the round-level result. This is distinct from per-round optimization:
- Standard RL optimizes round score (points won/lost this hand)
- Global reward prediction optimizes for final placement across all rounds

**Measured gains**: Qualitative only. The ablation (RL-basic -> RL-1 with global reward prediction) shows improvement in stable-rank distribution (Figure 8) but no exact Elo delta is reported. The Suphx system achieved stable rank 8.74 on Tenhou (top 0.01% of players), but the per-technique attribution is not quantified.

**Relevance to Hydra**: This maps directly to Hydra's GRP head. The insight is that predicting game-level outcome (1st/2nd/3rd/4th placement) provides a different gradient signal than round-level rewards, encouraging more strategic play (e.g., playing conservatively when ahead in points).

### 1.3 Mortal: Next-Rank Prediction

**Source**: [Mortal codebase](https://github.com/Equim-chan/Mortal) (DeepWiki analysis)

Mortal uses an auxiliary `next_rank_weight` loss predicting the player's final rank. No ablation published, but the technique is in production in a 10-dan-level agent.

---

## 2. Data Augmentation

### 2.1 Go: Dihedral Symmetry (8x)

**Source**: [AlphaGo Zero (Silver et al., 2017)](https://www.nature.com/articles/nature24270)

The Go board has 8-fold dihedral symmetry (4 rotations x 2 reflections). AlphaGo Zero exploited this by:
- Augmenting training data: each position generates 8 equivalent training samples
- Randomly transforming the board during MCTS evaluation

**Measured gain**: Not separately ablated. AlphaGo Zero uses it throughout; no "without augmentation" comparison exists. However, the 8x multiplier on training data is considered a major contributor to sample efficiency.

**Important caveat**: AlphaZero (the successor for chess/shogi/Go) **dropped dihedral augmentation** for chess and shogi because those games lack rotational symmetry. From the Science paper: "AlphaZero does not augment the training data and does not transform the board position during MCTS."

### 2.2 Mahjong: Suit Permutation (6x)

**Theoretical basis**: In Riichi Mahjong, the three numbered suits (man/pin/sou) are functionally identical. Any permutation of {man, pin, sou} produces an equivalent game state. This gives 3! = 6 equivalent states per position.

**Measured gains**: No published ablation exists for suit permutation in Mahjong AI. Neither Suphx, Mortal, nor NAGA report this technique's isolated contribution.

**Analysis**: This is a "free lunch" data augmentation:
- 6x more training samples per game record
- No approximation -- exact symmetry (unlike Go's dihedral which is approximate near edges)
- Can be applied at training time (augment each batch) or data generation time

### 2.3 Mahjong: Seat Rotation (4x)

**Theoretical basis**: In 4-player Mahjong, from each game record you can extract 4 different first-person perspectives (one per seat). This gives 4x data per game.

**Combined with suit permutation**: 6 x 4 = 24x data multiplier per game record. This is substantial for supervised pre-training phases.

**Caveat**: Seat rotation is NOT a free symmetry -- each seat has a different wind, and the game state differs per seat. It's really "multi-perspective learning" rather than "symmetry augmentation." Most Mahjong AIs already use all 4 perspectives from each game implicitly.

---

## 3. Curriculum Learning

### 3.1 Endgame-First Training

**Source**: [McAleer et al., 2019 "Improved Reinforcement Learning with Curriculum"](https://arxiv.org/abs/1903.12328)

Tested on Modified Racing Kings (chess variant) and Reversi with AlphaZero-style training:
- Start training from endgame positions (few pieces remaining)
- Gradually extend to earlier game states as training progresses

**Measured gains**:
- **Faster early convergence** -- the curriculum agent achieves higher win rates sooner
- **Same asymptotic performance** -- both curriculum and standard training converge to similar final strength
- No Elo numbers reported; results are qualitative from win-rate curves
- Training time savings from plot inspection: roughly 20-30% fewer training steps to reach 80% of final performance

**Why it might matter for Mahjong**: Mahjong's endgame (late-round decisions about riichi, defense, tenpai) is where the highest-impact decisions happen. Training first on these situations could bootstrap a stronger foundation faster.

**Why it might NOT matter**: Unlike chess where endgame tablebases provide ground truth, Mahjong endgame is still stochastic. The "curriculum" would need careful definition.

---

## 4. Population-Based Training (PBT)

### 4.1 AlphaStar: The League (STRONGEST PBT EVIDENCE)

**Source**: [AlphaStar Nature paper (Vinyals et al., 2019)](https://www.nature.com/articles/s41586-019-1724-z) -- [Full PDF](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)

The League is AlphaStar's population-based training system using Prioritized Fictitious Self-Play (PFSP). The ablation results from Figure 3:

**Multi-agent method ablation (Fig. 3C-D)**:

| Method | Test Elo | Min Win % vs Past |
|--------|----------|-------------------|
| FSP (fictitious self-play) | 1143 | 69% |
| PFSP (prioritized FSP) | 1273 | 70% |
| Naive self-play (SP) | 1519 | **46%** |
| PFSP + SP (full) | 1540 | 71% |

**Key insight**: Naive self-play achieves HIGH raw Elo (1519) but is CATASTROPHICALLY FORGETFUL -- only 46% min win-rate vs past versions. The League's value is NOT higher peak Elo, it's **robustness**. PFSP+SP gets 1540 Elo with 71% stability.

**League composition ablation (Fig. 3A-B)**:

| Config | Test Elo | Relative Pop. Perf. |
|--------|----------|---------------------|
| Main Agents only | 1540 | 6% |
| + Main Exploiters | 1693 | 35% |
| + League Exploiters | 1824 | 62% |

The full League with exploiters adds **+284 Elo** over main agents alone. Exploiters are the secret sauce -- they find and exploit weaknesses, forcing the main agents to become more robust.

**Relevance to Hydra**: For Mahjong, naive self-play may be sufficient initially (high Elo, simpler to implement). But if the agent develops exploitable patterns (e.g., always folding against certain signals), exploiter agents could fix this. The engineering cost of a full League is massive though -- AlphaStar used hundreds of TPUs.

---

## 5. Representation Learning

### 5.1 CURL: Contrastive Learning for RL

**Source**: [CURL (Srinivas et al., 2020)](https://arxiv.org/abs/2004.04136)

Adds contrastive self-supervised objective as auxiliary loss for pixel-based RL:
- **1.9x sample efficiency improvement** on DMControl Suite
- **1.2x** on Atari games
- Median human-normalized score on Atari 100k: 0.175

### 5.2 SPR: Self-Predictive Representations (STRONGER)

**Source**: [SPR (Schwarzer et al., 2021)](https://arxiv.org/abs/2007.05929)

Predicts future latent representations rather than using contrastive loss:

| Method | Median HNS (Atari 100k) |
|--------|--------------------------|
| CURL | 0.175 |
| DrQ | 0.268 |
| SPR (no aug) | 0.307 |
| **SPR (full)** | **0.415** |

- **+55% over previous SOTA** (DrQ)
- **+137% over CURL**

**Applicability to board/card game AI**: LIMITED. CURL and SPR are designed for pixel observations where representation learning is the bottleneck. In Mahjong/Go/Chess with hand-crafted state encodings (like Hydra's 85x34 tensor), the representation is already good. These techniques are more relevant if training from raw visual inputs.

**What IS relevant**: The principle of predicting future states as an auxiliary objective. This could be adapted for Mahjong as "predict the next few discards" or "predict the wall draw sequence" -- similar in spirit to KataGo's ownership prediction but temporal rather than spatial.

---

## 6. Reward Shaping

### 6.1 OpenAI Five: Dense Rewards (STRONGEST EVIDENCE)

**Source**: [OpenAI Five blog post (2018)](https://openai.com/index/openai-five/)

OpenAI Five for Dota 2 used extensive reward shaping with intermediate metrics:
- Net worth, kills, deaths, assists, last hits
- Competitive postprocessing: subtract opponent team's average reward
- "Team spirit" parameter annealed from 0 (individual) to 1 (team) during training

**Measured gain** (1v1 ablation):
- **With reward shaping**: ~90 TrueSkill (semi-pro+)
- **Without (win/loss only)**: ~70 TrueSkill, **~10x slower training**

Dense rewards trained approximately 10x faster and reached a higher plateau.

### 6.2 Mahjong: ShangTing + Bonus Shaping

**Source**: [Chen & Lai, 2023 "A Novel Reward Shaping Function for Single-Player Mahjong"](https://arxiv.org/abs/2305.04145)

Uses ShangTing distance (tiles-from-tenpai heuristic) as potential-based reward shaping:
- Incremental form: reward = delta(ShangTing + Bonus) per discard
- Novel "unscented bonus" adds rewards for honor triplets and suit concentration

**Measured gain** (bonus shaping vs ShangTing-only):
- **+$1.37 net earnings per game** (over 1000 games, >99% confidence)
- Single-player completion rate: 100% over 10,000 games with ~34.6 average discards

### 6.3 Potential-Based Reward Shaping (PBRS) Theory

**Key guarantee**: If reward shaping is potential-based (R'(s,a,s') = R(s,a,s') + gamma*Phi(s') - Phi(s)), the optimal policy is preserved. Non-potential-based shaping can change the optimal policy.

**Implication for Hydra**: Tenpai-distance and hand-value-based shaping can safely accelerate early training if implemented as PBRS. But be careful -- giving bonus rewards for "good defense" or "reaching tenpai" that aren't potential-based could distort the learned policy toward suboptimal play.

### 6.4 Risk: Reward Shaping Can Hurt

OpenAI Five's team explicitly noted that their reward weights were hand-tuned and that incorrect weights could lead to degenerate behavior (e.g., farming gold instead of winning). For Mahjong, this means:
- Rewarding tenpai too strongly -> agent riichis recklessly
- Rewarding defense too strongly -> agent folds excessively
- Rewarding hand value too strongly -> agent chases expensive hands and loses placement

The safest approach is sparse terminal rewards (placement score) with PBRS acceleration in early training, gradually annealing shaped rewards to zero.

---

## 7. Teacher-Student Distillation

### 7.1 Policy Distillation (Rusu et al., 2015)

**Source**: [Policy Distillation (Rusu et al., 2015)](https://arxiv.org/abs/1511.06295)

Trains a student network on the teacher's soft action distributions rather than raw game outcomes. Tested on 10 Atari games:

**Single-game distillation** (geometric mean, % of DQN teacher):

| Student Size | % of Teacher Params | Performance |
|-------------|---------------------|-------------|
| net1 (25%) | 4x smaller | **108.3%** of teacher |
| net2 (7%) | ~15x smaller | **101.7%** of teacher |
| net3 (4%) | ~27x smaller | 83.9% of teacher |

**Key finding**: A 4x smaller student EXCEEDS the teacher by 8.3% through distillation. The soft probability distributions contain richer information than hard labels.

**Multi-game distillation** (3 games, single student):

| Method | Performance |
|--------|-------------|
| Multi-DQN (joint training) | 83.5% |
| Multi-Dist-NLL | 105.1% |
| **Multi-Dist-KL** | **116.9%** |

A single distilled network playing 3 games outperforms the 3 separate teachers by 16.9% on average.

### 7.2 AlphaGo: Supervised -> RL Pipeline

AlphaGo's original pipeline used supervised learning from human expert games first, then fine-tuned with RL self-play. This is effectively distillation from human experts into the network, followed by self-improvement.

### 7.3 Relevance to Hydra

Distillation is most useful for:
1. **Model compression**: Train a large teacher, distill to a smaller inference model
2. **Multi-generation training**: Train generation N+1 on generation N's soft outputs (smoother than pure self-play)
3. **Supervised pre-training**: Distill from human game records (Mortal's approach) before RL fine-tuning

The 108% teacher-exceeding result is particularly interesting -- it suggests that even the SAME-SIZE model could benefit from being trained on another model's soft outputs rather than raw game outcomes.

---

## 8. Ensemble Methods

### 8.1 Overview

Ensemble methods in game AI typically involve:
- Running multiple models and averaging their policy outputs
- Running multiple models and voting on the best action
- Using different checkpoints from training as ensemble members

### 8.2 Measured Gains

**Hard numbers are scarce** in the literature for game-specific ensembles. What exists:

- **Chess community estimates**: Combining 2-3 diverse neural network evaluations in Leela Chess Zero-style engines typically yields ~50-100 Elo over a single model (community benchmarks, not peer-reviewed)
- **AlphaGo Zero root parallel MCTS**: Uses "virtual loss" to enable parallel tree search, which is conceptually similar to ensemble voting. Not separately ablated.
- **Gomoku root-parallel MCTS**: One open implementation reports ensemble move voting from parallel MCTS trees, but no measured Elo delta.

### 8.3 Applicability to Mahjong

Ensembles are primarily an INFERENCE-TIME technique (violates the "no inference-time changes" constraint). However, there's a training-time variant:

**Ensemble distillation**: Train multiple diverse models, then distill their averaged predictions into a single student. This gives you ensemble-quality training signal without ensemble-cost inference.

The downside: training N models costs Nx compute. For Hydra's scale (~16.5M params, limited GPU budget), this is expensive.

---

## 9. Recommendations for Hydra (Ranked by Evidence Strength)

### Tier 1: Strong Evidence, Implement First

1. **Auxiliary prediction heads** (already planned: GRP, Tenpai, Danger)
   - KataGo evidence: +190 Elo, 1.65x convergence
   - Hydra already has 5 output heads -- this is baked into the design
   - Consider adding: opponent discard prediction, future state prediction

2. **Suit permutation augmentation** (6x data, free lunch)
   - Exact symmetry, zero approximation
   - No published ablation, but theoretically free
   - Implement during training data loading

3. **PBRS reward shaping for early training**
   - OpenAI Five: 10x training speedup with dense rewards
   - Use ShangTing-style tenpai distance as potential function
   - Anneal to zero over training (terminal placement score only for final policy)

### Tier 2: Moderate Evidence, Consider for Phase 2

4. **Supervised pre-training with distillation**
   - Policy distillation: student can EXCEED teacher by 8%
   - Pre-train on human game records (Tenhou logs), then RL fine-tune
   - Mortal and Suphx both use this approach

5. **Global reward prediction** (game-level placement, not just round score)
   - Suphx uses this; Hydra's GRP head covers it
   - Encourage strategic thinking beyond single-round optimization

### Tier 3: Weak Evidence, Deprioritize

6. **Curriculum learning** (endgame-first)
   - Only faster convergence, same asymptote
   - Complex to implement for Mahjong (what defines "endgame"?)
   - Low priority unless training is extremely slow

7. **Population-based training / League**
   - AlphaStar shows robustness gains (+284 Elo with exploiters)
   - But: massive engineering cost, requires multi-GPU infrastructure
   - Overkill for initial Hydra training; revisit if agent develops exploitable patterns

8. **Self-supervised representation learning** (CURL/SPR style)
   - Mostly relevant for pixel-based RL, not state-based
   - The PRINCIPLE (predict future states) could be useful as auxiliary head
   - Low priority for hand-crafted 85x34 encoding

9. **Ensemble methods**
   - Training-time ensembles cost Nx compute
   - Inference ensembles change the deployment model
   - Ensemble distillation is viable but expensive
   - Lowest priority

---

## References

1. Wu, D. (2019). "Accelerating Self-Play Learning in Go." arXiv:1902.10565
2. Silver, D. et al. (2017). "Mastering the game of Go without human knowledge." Nature 550.
3. Silver, D. et al. (2018). "A general reinforcement learning algorithm." Science 362.
4. Vinyals, O. et al. (2019). "Grandmaster level in StarCraft II." Nature 575.
5. Li, J. et al. (2020). "Suphx: Mastering Mahjong with Deep RL." arXiv:2003.13590
6. Rusu, A. et al. (2015). "Policy Distillation." arXiv:1511.06295
7. Srinivas, A. et al. (2020). "CURL: Contrastive Unsupervised Representations for RL." ICML.
8. Schwarzer, M. et al. (2021). "Data-Efficient RL with Self-Predictive Representations." ICLR.
9. Chen, K. & Lai, L. (2023). "A Novel Reward Shaping Function for Single-Player Mahjong." arXiv:2305.04145
10. McAleer, S. et al. (2019). "Improved RL with Curriculum." arXiv:1903.12328
11. OpenAI (2018). "OpenAI Five." Blog post.
