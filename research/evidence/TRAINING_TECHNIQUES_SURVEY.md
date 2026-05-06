# Training-Time Techniques for Stronger Game AI

> Techniques improving SAME network's strength without architecture change or inference-time search.

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
- **Ownership head**: Predict per-intersection end ownership (361-dim output for 19x19)
- **Score head**: Predict final score difference (scalar)

**Measured gains** (from ablation "NoVAux" -- removing both):

| Config | Elo at 2.5G queries |
|--------|---------------------|
| Full (with aux) | 1329 |
| NoVAux (without aux) | 1139 |
| **Delta** | **+190 Elo** |

- **Convergence speedup**: 1.65x (need ~65% more compute without aux targets for same strength)
- Training-only targets -- richer gradient signal, no inference behavior change

**Why it works**: Win/loss = one bit feedback per game. Ownership gives 361 local signals per position -- network learns "wrong HERE." Score prediction forces understanding relative advantage, not only binary outcomes.

**Relevance to Hydra**: Hydra already plans GRP (24-way Game Result Prediction), Tenpai (3), Danger (3x34) heads -- same principle. KataGo evidence suggests these aux heads may be worth +100-200 Elo equivalent in Mahjong.

### 1.2 Suphx: Global Reward Prediction

**Source**: [Suphx paper (Li et al., 2020)](https://arxiv.org/abs/2003.13590)

Suphx uses "global reward prediction" -- predict final game-level placement from intermediate states, not only round-level result. Distinct from per-round optimization:
- Standard RL optimizes round score (points won/lost this hand)
- Global reward prediction optimizes final placement across all rounds

**Measured gains**: Qualitative only. Ablation (RL-basic -> RL-1 with global reward prediction) shows improved stable-rank distribution (Figure 8) but no exact Elo delta. Suphx reached stable rank 8.74 on Tenhou (top 0.01% of players), but per-technique attribution not quantified.

**Relevance to Hydra**: Direct map to Hydra's GRP head. Core insight: predicting game-level outcome (1st/2nd/3rd/4th placement) gives different gradient than round-level rewards, pushing more strategic play (e.g., conservative play when ahead on points).

### 1.3 Mortal: Next-Rank Prediction

**Source**: [Mortal codebase](https://github.com/Equim-chan/Mortal) (DeepWiki analysis)

Mortal uses auxiliary `next_rank_weight` loss predicting player's final rank. No published ablation, but technique exists in production 10-dan-level agent.

---

## 2. Data Augmentation

### 2.1 Go: Dihedral Symmetry (8x)

**Source**: [AlphaGo Zero (Silver et al., 2017)](https://www.nature.com/articles/nature24270)

Go board has 8-fold dihedral symmetry (4 rotations x 2 reflections). AlphaGo Zero used this by:
- Augmenting training data: each position yields 8 equivalent training samples
- Randomly transforming board during MCTS evaluation

**Measured gain**: Not separately ablated. AlphaGo Zero uses it throughout; no "without augmentation" comparison. Still, 8x data multiplier widely viewed as major sample-efficiency contributor.

**Important caveat**: AlphaZero (successor for chess/shogi/Go) **dropped dihedral augmentation** for chess and shogi because those games lack rotational symmetry. From Science paper: "AlphaZero does not augment training data and does not transform board position during MCTS."

### 2.2 Mahjong: Suit Permutation (6x)

**Theoretical basis**: In Riichi Mahjong, three numbered suits (man/pin/sou) are functionally identical. Any permutation of {man, pin, sou} yields equivalent game state. Gives 3! = 6 equivalent states per position.

**Measured gains**: No published ablation for suit permutation in Mahjong AI. Suphx, Mortal, NAGA do not report isolated contribution.

**Analysis**: "Free lunch" data augmentation:
- 6x more training samples per game record
- No approximation -- exact symmetry (unlike Go dihedral, approximate near edges)
- Usable at training time (augment each batch) or data generation time

### 2.3 Mahjong: Seat Rotation (4x)

**Theoretical basis**: In 4-player Mahjong, each game record can yield 4 different first-person perspectives (one per seat). Gives 4x data per game.

**Combined with suit permutation**: 6 x 4 = 24x data multiplier per game record. Big for supervised pre-training.

**Caveat**: Seat rotation is NOT free symmetry -- each seat has different wind, game state differs per seat. Better framed as "multi-perspective learning" than "symmetry augmentation." Most Mahjong AIs already use all 4 perspectives from each game implicitly.

---

## 3. Curriculum Learning

### 3.1 Endgame-First Training

**Source**: [McAleer et al., 2019 "Improved Reinforcement Learning with Curriculum"](https://arxiv.org/abs/1903.12328)

Tested on Modified Racing Kings (chess variant) and Reversi with AlphaZero-style training:
- Start from endgame positions (few pieces remaining)
- Gradually extend toward earlier game states as training progresses

**Measured gains**:
- **Faster early convergence** -- curriculum agent reaches higher win rates sooner
- **Same asymptotic performance** -- curriculum and standard training converge to similar final strength
- No Elo numbers reported; results qualitative from win-rate curves
- Training-time savings from plot inspection: roughly 20-30% fewer steps to reach 80% of final performance

**Why it might matter for Mahjong**: Mahjong endgame (late-round riichi/defense/tenpai decisions) contains highest-impact decisions. Training there first may bootstrap stronger foundation faster.

**Why it might NOT matter**: Unlike chess, no endgame tablebase ground truth. Mahjong endgame still stochastic. "Curriculum" needs careful definition.

---

## 4. Population-Based Training (PBT)

### 4.1 AlphaStar: The League (STRONGEST PBT EVIDENCE)

**Source**: [AlphaStar Nature paper (Vinyals et al., 2019)](https://www.nature.com/articles/s41586-019-1724-z) -- [Full PDF](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)

League = AlphaStar's population-based training system using Prioritized Fictitious Self-Play (PFSP). Ablation results from Figure 3:

**Multi-agent method ablation (Fig. 3C-D)**:

| Method | Test Elo | Min Win % vs Past |
|--------|----------|-------------------|
| FSP (fictitious self-play) | 1143 | 69% |
| PFSP (prioritized FSP) | 1273 | 70% |
| Naive self-play (SP) | 1519 | **46%** |
| PFSP + SP (full) | 1540 | 71% |

**Key insight**: Naive self-play reaches HIGH raw Elo (1519) but becomes CATASTROPHICALLY FORGETFUL -- only 46% min win-rate vs past versions. League value is NOT peak Elo gain; it is **robustness**. PFSP+SP gets 1540 Elo with 71% stability.

**League composition ablation (Fig. 3A-B)**:

| Config | Test Elo | Relative Pop. Perf. |
|--------|----------|---------------------|
| Main Agents only | 1540 | 6% |
| + Main Exploiters | 1693 | 35% |
| + League Exploiters | 1824 | 62% |

Full League with exploiters adds **+284 Elo** over main agents alone. Exploiters are secret sauce -- they find/exploit weaknesses, forcing main agents to become more robust.

**Relevance to Hydra**: For Mahjong, naive self-play may be enough at first (high Elo, simpler impl). But if agent learns exploitable patterns (e.g., folding vs certain signals), exploiter agents could fix this. Full League engineering cost massive though -- AlphaStar used hundreds of TPUs.

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

Predict future latent representations instead of using contrastive loss:

| Method | Median HNS (Atari 100k) |
|--------|--------------------------|
| CURL | 0.175 |
| DrQ | 0.268 |
| SPR (no aug) | 0.307 |
| **SPR (full)** | **0.415** |

- **+55% over previous SOTA** (DrQ)
- **+137% over CURL**

**Applicability to board/card game AI**: LIMITED. CURL and SPR target pixel observations where representation learning is bottleneck. In Mahjong/Go/Chess with hand-crafted state encodings (like Hydra's 85x34 tensor), representation already decent. More relevant if training from raw visual input.

**What IS relevant**: Principle of predicting future states as auxiliary objective. Could adapt to Mahjong as "predict next few discards" or "predict wall draw sequence" -- same spirit as KataGo ownership prediction, but temporal not spatial.

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

Dense rewards trained roughly 10x faster and reached higher plateau.

### 6.2 Mahjong: ShangTing + Bonus Shaping

**Source**: [Chen & Lai, 2023 "A Novel Reward Shaping Function for Single-Player Mahjong"](https://arxiv.org/abs/2305.04145)

Uses ShangTing distance (tiles-from-tenpai heuristic) as potential-based reward shaping:
- Incremental form: reward = delta(ShangTing + Bonus) per discard
- Novel "unscented bonus" adds rewards for honor triplets and suit concentration

**Measured gain** (bonus shaping vs ShangTing-only):
- **+$1.37 net earnings per game** (over 1000 games, >99% confidence)
- Single-player completion rate: 100% over 10,000 games with ~34.6 average discards

### 6.3 Potential-Based Reward Shaping (PBRS) Theory

**Key guarantee**: If reward shaping is potential-based (R'(s,a,s') = R(s,a,s') + gamma*Phi(s') - Phi(s)), optimal policy is preserved. Non-potential-based shaping can change optimal policy.

**Implication for Hydra**: Tenpai-distance and hand-value-based shaping can safely speed early training if implemented as PBRS. But caution -- bonus rewards for "good defense" or "reaching tenpai" that are not potential-based could distort learned policy toward suboptimal play.

### 6.4 Risk: Reward Shaping Can Hurt

OpenAI Five team explicitly noted reward weights were hand-tuned; wrong weights could create degenerate behavior (e.g., farming gold instead of winning). For Mahjong, this means:
- Reward tenpai too strongly -> agent riichis recklessly
- Reward defense too strongly -> agent folds excessively
- Reward hand value too strongly -> agent chases expensive hands and loses placement

Safest path: sparse terminal rewards (placement score) with PBRS acceleration in early training, then anneal shaped rewards to zero.

---

## 7. Teacher-Student Distillation

### 7.1 Policy Distillation (Rusu et al., 2015)

**Source**: [Policy Distillation (Rusu et al., 2015)](https://arxiv.org/abs/1511.06295)

Train student network on teacher's soft action distributions, not raw game outcomes. Tested on 10 Atari games:

**Single-game distillation** (geometric mean, % of DQN teacher):

| Student Size | % of Teacher Params | Performance |
|-------------|---------------------|-------------|
| net1 (25%) | 4x smaller | **108.3%** of teacher |
| net2 (7%) | ~15x smaller | **101.7%** of teacher |
| net3 (4%) | ~27x smaller | 83.9% of teacher |

**Key finding**: 4x smaller student EXCEEDS teacher by 8.3% through distillation. Soft probability distributions carry richer information than hard labels.

**Multi-game distillation** (3 games, single student):

| Method | Performance |
|--------|-------------|
| Multi-DQN (joint training) | 83.5% |
| Multi-Dist-NLL | 105.1% |
| **Multi-Dist-KL** | **116.9%** |

Single distilled network playing 3 games outperforms 3 separate teachers by 16.9% on average.

### 7.2 AlphaGo: Supervised -> RL Pipeline

AlphaGo's original pipeline used supervised learning from human expert games first, then RL self-play fine-tuning. Effectively distillation from human experts into network, then self-improvement.

### 7.3 Relevance to Hydra

Distillation most useful for:
1. **Model compression**: Train large teacher, distill to smaller inference model
2. **Multi-generation training**: Train generation N+1 on generation N's soft outputs (smoother than pure self-play)
3. **Supervised pre-training**: Distill from human game records (Mortal's approach) before RL fine-tuning

108% teacher-exceeding result especially interesting -- suggests even SAME-SIZE model may benefit from training on another model's soft outputs instead of raw game outcomes.

---

## 8. Ensemble Methods

### 8.1 Overview

Ensemble methods in game AI usually involve:
- Running multiple models and averaging policy outputs
- Running multiple models and voting best action
- Using different training checkpoints as ensemble members

### 8.2 Measured Gains

**Hard numbers are scarce** in literature for game-specific ensembles. Existing data:
- **Chess community estimates**: Combining 2-3 diverse neural network evaluations in Leela Chess Zero-style engines typically yields ~50-100 Elo over one model (community benchmarks, not peer-reviewed)
- **AlphaGo Zero root parallel MCTS**: Uses "virtual loss" to enable parallel tree search, conceptually similar to ensemble voting. Not separately ablated.
- **Gomoku root-parallel MCTS**: One open impl reports ensemble move voting from parallel MCTS trees, but no measured Elo delta.

### 8.3 Applicability to Mahjong

Ensembles are mainly INFERENCE-TIME technique (violates "no inference-time changes" constraint). Training-time variant exists:

**Ensemble distillation**: Train multiple diverse models, then distill averaged predictions into one student. Gives ensemble-quality training signal without ensemble-cost inference.

Downside: training N models costs Nx compute. For Hydra's scale (~16.5M params, limited GPU budget), expensive.

---

## 9. Recommendations for Hydra (Ranked by Evidence Strength)

### Tier 1: Strong Evidence, Implement First

1. **Auxiliary prediction heads** (already planned: GRP, Tenpai, Danger)
   - KataGo evidence: +190 Elo, 1.65x convergence
   - Hydra already has 5 output heads -- baked into design
   - Consider adding: opponent discard prediction, future state prediction

2. **Suit permutation augmentation** (6x data, free lunch)
   - Exact symmetry, zero approximation
   - No published ablation, but theoretically free
   - Implement in training data loader

3. **PBRS reward shaping for early training**
   - OpenAI Five: 10x training speedup with dense rewards
   - Use ShangTing-style tenpai distance as potential function
   - Anneal to zero over training (terminal placement score only for final policy)

### Tier 2: Moderate Evidence, Consider for Phase 2

4. **Supervised pre-training with distillation**
   - Policy distillation: student can EXCEED teacher by 8%
   - Pre-train on human game records (Tenhou logs), then RL fine-tune
   - Mortal and Suphx both use this path

5. **Global reward prediction** (game-level placement, not only round score)
   - Suphx uses this; Hydra's GRP head covers it
   - Encourage strategic thinking beyond single-round optimization

### Tier 3: Weak Evidence, Deprioritize

6. **Curriculum learning** (endgame-first)
   - Only faster convergence, same asymptote
   - Complex to implement for Mahjong (what defines "endgame"?)
   - Low priority unless training slow

7. **Population-based training / League**
   - AlphaStar shows robustness gains (+284 Elo with exploiters)
   - But: massive engineering cost, needs multi-GPU infrastructure
   - Overkill for initial Hydra training; revisit if agent develops exploitable patterns

8. **Self-supervised representation learning** (CURL/SPR style)
   - Mostly relevant for pixel-based RL, not state-based
   - PRINCIPLE (predict future states) may still help as auxiliary head
   - Low priority for hand-crafted 85x34 encoding

9. **Ensemble methods**
   - Training-time ensembles cost Nx compute
   - Inference ensembles change deployment model
   - Ensemble distillation viable but expensive
   - Lowest priority

---

## References

1. Wu, D. (2019). "Accelerating Self-Play Learning in Go." arXiv:1902.10565
2. Silver, D. et al. (2017). "Mastering game of Go without human knowledge." Nature 550.
3. Silver, D. et al. (2018). general reinforcement learning algorithm." Science 362.
4. Vinyals, O. et al. (2019). "Grandmaster level in StarCraft II." Nature 575.
5. Li, J. et al. (2020). "Suphx: Mastering Mahjong with Deep RL." arXiv:2003.13590
6. Rusu, et al. (2015). "Policy Distillation." arXiv:1511.06295
7. Srinivas, et al. (2020). "CURL: Contrastive Unsupervised Representations for RL." ICML.
8. Schwarzer, M. et al. (2021). "Data-Efficient RL with Self-Predictive Representations." ICLR.
9. Chen, K. & Lai, L. (2023). Novel Reward Shaping Function for Single-Player Mahjong." arXiv:2305.04145
10. McAleer, S. et al. (2019). "Improved RL with Curriculum." arXiv:1903.12328
11. OpenAI (2018). "OpenAI Five." Blog post.