# Compute Budget Feasibility Analysis: Hydra on 2668 GPU-hours

## Executive Summary

**Bottom line: 2668 GPU-hours on Quadro RTX 5000 is sufficient for reaching strong amateur/low-dan play (Phase 1-2 of training), but fundamentally insufficient for LuckyJ-level (10+ dan) play without radical efficiency innovations. The budget is comparable to what Suphx used per RL agent on much weaker hardware, putting us in "one good shot" territory.**

**Recommendation: Option (b) -- pursue radically more efficient approach. BC-heavy pipeline with targeted RL fine-tuning is the only viable path at this budget.**

---

## 1. Hard Compute Data from Mahjong AI Systems

### 1.1 Suphx (Microsoft Research, 2020)

**Source**: [arXiv:2003.13590](https://arxiv.org/abs/2003.13590), Section 4.2

| Metric | Value |
|--------|-------|
| Training hardware | 4 Titan XP (param server) + 40 Tesla K80 (self-play) |
| Training time | **2 days per RL agent** |
| Total GPU-hours | **2,112 GPU-hours per agent** (44 GPUs x 48h) |
| Self-play games | **1.5M games per agent** |
| Evaluation cost | 20 K80s for 2 additional days |
| SL training data | 15M discard + 5M riichi + 10M chow + 10M pong + 4M kong samples |
| Model | 50 residual blocks, 256 channels, 5 separate networks |
| Result | **10 dan** (Tenhou, somewhat unstable) |

**FLOPS-normalized to RTX 5000:**
- K80 die: ~2.9 TFLOPS FP32; Titan XP: ~12.1 TFLOPS; Quadro RTX 5000: **11.2 TFLOPS**
- Suphx effective compute: (40 x 2.9 + 4 x 12.1) x 48h = ~7,900 TFLOPS-hours
- Our budget: 2668 x 11.2 = **~29,900 TFLOPS-hours**
- **We have ~3.8x one Suphx agent's compute budget in raw FLOPS**

### 1.2 LuckyJ / JueJong (Tencent AI Lab, 2022-2023)

**No public training compute data exists for LuckyJ itself.** However, we have data from the two published papers underlying its techniques:

**ACH / JueJong (ICLR 2022)** -- predecessor, 1-on-1 Mahjong only:

| Metric | Value |
|--------|-------|
| Hardware | 800 CPUs, 3200 GB RAM, 8 NVIDIA M40 GPUs |
| Training steps | 1,000,000 |
| Batch size | 8,192 |
| Model | 3 stages x 3 res blocks = 9 blocks, channels 64->128->32 |
| Eval games | 7,700 games vs 157 humans + 1,000 vs champion |
| Result | Beat 2014 World Mahjong champion in 1v1 |

**OLSS (ICML 2023)** -- online search for 4-player Mahjong:

| Metric | Value |
|--------|-------|
| Blueprint training | 8 V100 GPUs + 1,200 CPUs for 2 days = **384 V100-hours** |
| Environmental model | 8 V100 GPUs + 2,400 CPUs |
| Key innovation | 100x faster than common-knowledge subgame solving |
| Search at inference | pUCT (much cheaper than CFR) |

**Estimated LuckyJ total**: Given Tencent AI Lab's resources, likely **10,000-50,000+ GPU-hours** (V100/A100 class) across multiple training phases, hyperparameter sweeps, and iterative self-play leagues. The OLSS paper alone needed ~768 V100-hours for two model components, and LuckyJ would have required many iterations plus the league training described in their NeurIPS 2023 StarCraft work.

### 1.3 Mortal (Individual Developer, Open Source)

**No published total compute budget.** Key data points:

| Metric | Value |
|--------|-------|
| Self-play throughput | 40K hanchans/hour on RTX 4090 + Ryzen 9 7950X |
| Default config | 192 channels, 40 res blocks (~our Hydra spec) |
| Batch size | 512 |
| Developer note | "cost me far more time and money than I anticipated for a hobby" |

**Estimated compute**: Based on architecture similarity (192ch/40 blocks is close to our 256ch/40 blocks) and the developer running on consumer hardware for months, likely **500-2,000 GPU-hours equivalent** on a 4090 over the full training run. Mortal achieves strong play but not 10-dan level.

### 1.4 LsAc*-MJ (Low-Resource Mahjong, 2024)

**Source**: [Wiley 10.1155/2024/4558614](https://onlinelibrary.wiley.com/doi/full/10.1155/2024/4558614)

| Metric | Value |
|--------|-------|
| Training time | **51.4 hours** (vs DQN 277h, NFSP 355h, Dueling 1105h) |
| Parameters | 308K (much smaller model) |
| Key technique | Knowledge-guided pretraining + A2C self-play |
| Result | Beats DQN/NFSP/Dueling baselines (not competitive with Suphx/Mortal) |

This is the only paper explicitly targeting **low-resource Mahjong training**. Their two-stage approach (knowledge-guided + self-play) is conceptually similar to our BC + RL pipeline.

---

## 2. Comparison Game AI Compute Budgets

| System | Game | Hardware | Training Time | Effective Compute | Cost | Result |
|--------|------|----------|--------------|-------------------|------|--------|
| **Suphx** | Mahjong | 44 GPUs (K80+TitanXP) | 2 days/agent | ~2,100 GPU-hr/agent | Unknown | 10 dan |
| **LuckyJ** | Mahjong | V100s + thousands CPUs | Unknown | Est. 10K-50K+ GPU-hr | Unknown | 10.68 dan (stable) |
| **Mortal** | Mahjong | Consumer GPU (4090) | Months | Est. 500-2,000 GPU-hr | Personal funds | Strong amateur |
| **Pluribus** | Poker | 64-core CPU server | 8 days | 12,400 CPU-hr | **$144** | Superhuman |
| **AlphaStar** | StarCraft II | 16 TPUs/agent x 600 agents | 14-44 days | ~Millions TPU-hr | ~$Millions | Grandmaster |
| **OpenAI Five** | Dota 2 | 256 P100 + 128K CPUs | Months | ~Millions GPU-hr | ~$Millions | Beat pros |
| **Hydra (ours)** | Mahjong | Quadro RTX 5000 | TBD | **2,668 GPU-hr** | Grant-funded | Target: 80%+ |

## 3. Scaling Laws for Game AI

**Source**: [arXiv:2301.13442](https://arxiv.org/abs/2301.13442) -- "Scaling Laws for Single-Agent RL"

Key findings relevant to us:

1. **RL performance follows power laws** in both model size (N) and environment interactions (E): `I^(-beta) = (Nc/N)^alpha_N + (Ec/E)^alpha_E`

2. **Optimal model size scales with compute budget**: The exponent ranges from 0.40-0.80 depending on domain. For Dota 2: ~0.76. This means as compute doubles, optimal model size should increase by ~70%.

3. **Environment cost dominates**: "It is usually inefficient to use a model that is much cheaper to run than the environment." For Mahjong, our engine is fast (Rust), so we can afford a bigger model.

4. **Dota 2 vs MNIST**: Same model needs ~2000x more training on Dota 2 than MNIST. Game complexity matters enormously.

5. **Sample efficiency vs humans**: RL needs 100-10,000x more interactions than humans to reach the same level.

**No Mahjong-specific scaling laws exist.** But Mahjong complexity is between poker (simpler) and StarCraft/Dota (much more complex). The hidden information aspect adds sample complexity beyond what game-tree size alone would suggest.

---

## 4. Sample Efficiency Techniques (Dan-per-GPU-hour)

Ranked by expected impact:

### 4.1 Oracle Guiding (Suphx, highest impact)
Train with perfect information first, then distill to imperfect-information agent. Suphx showed this dramatically accelerates early RL training. **Estimated 3-10x speedup** in early phases.

### 4.2 Behavioral Cloning Pretraining (our Phase 1)
BC from expert games gives a strong initialization. Instead of learning from scratch via RL, start from a competent policy. **Estimated 5-20x speedup** vs pure RL from random.

### 4.3 Global Reward Prediction (Suphx)
Predict final game outcome from intermediate states. Reduces credit assignment problem in long-horizon games. **Estimated 2-5x improvement** in value estimation quality.

### 4.4 CQL (Conservative Q-Learning) -- Mortal's approach
Mortal uses offline RL (CQL) which is dramatically more sample-efficient than online RL because it reuses logged data. **This is our biggest efficiency lever** -- no wasted self-play games, every sample is reused.

### 4.5 Knowledge Distillation
Train a small "fast" model for self-play generation, distill into the large model. Reduces self-play GPU cost by 4-8x.

### 4.6 Prioritized Experience Replay
Re-weight training samples by TD error or novelty. Standard 1.5-2x improvement.

---

## 5. BC Data Scaling Saturation

No Mahjong-specific study exists, but from the general imitation learning literature and the Suphx data:

- **Suphx SL data**: 15M discard samples, 5M riichi, 10M each for chow/pong, 4M kong = ~44M total samples
- **Mortal**: Trained on years of Tenhou log data (millions of games available)
- **General pattern**: BC performance follows a log-linear curve -- each 10x increase in data gives a roughly constant improvement. Saturation typically occurs when the policy captures ~95% of expert behavior variance.

**Estimated saturation point for Mahjong BC**: ~500K-2M expert games (10-40M decision samples). Beyond this, additional data yields diminishing returns on action prediction accuracy. The remaining gap must be closed by RL.

**Key insight**: BC gets you to ~dan level cheaply. The jump from dan to 10-dan requires RL, which is where compute really matters.

## 6. Optimal BC/RL Compute Split (the "Chinchilla" for Game AI)

No formal equivalent exists, but we can derive estimates from what worked:

**Suphx approach**: ~20% SL, ~80% RL (SL was a pretraining step, bulk of compute in RL self-play)

**Mortal approach**: Primarily offline RL on logged data (CQL). Essentially 100% data-reuse, minimal fresh self-play. This is massively more compute-efficient.

**AlphaStar**: ~5% imitation learning (from replays), ~95% RL (league self-play)

**Recommended split for Hydra at 2668 GPU-hours**:

| Phase | Budget | GPU-hours | What it buys |
|-------|--------|-----------|-------------|
| Phase 1: BC Pretraining | 10-15% | 250-400h | Train on expert Tenhou logs to ~dan prediction accuracy |
| Phase 2: Offline RL (CQL) | 40-50% | 1,000-1,300h | Fine-tune with conservative Q-learning on same data |
| Phase 3: Online RL Self-play | 30-40% | 800-1,000h | Self-play with PPO to go beyond expert data |
| Evaluation + Tuning | 5-10% | 130-270h | 1v3 testing, hyperparameter sweeps |

---

## 7. Minimum Viable Compute for Superhuman Mahjong

Nobody has studied this directly. But we can triangulate:

- **Pluribus** (poker, simpler): $144 on CPUs. Poker is *much* simpler than Mahjong.
- **Suphx** (Mahjong, 2020): ~2,100 GPU-hours on old hardware per agent, reached 10 dan (unstable). Multiple agents trained iteratively.
- **Mortal** (Mahjong, ongoing): Hundreds to low-thousands of GPU-hours on modern consumer hardware, reached strong amateur.
- **LuckyJ** (Mahjong, 2023): Unknown but almost certainly >>10K GPU-hours, reached stable 10.68 dan.

**Estimated minimum for 10+ dan Mahjong AI**:
- With state-of-the-art techniques (OLSS, oracle guiding, CQL): **5,000-15,000 RTX 5000-equivalent GPU-hours**
- Without advanced techniques (basic PPO self-play): **50,000-100,000+ GPU-hours**
- For 80% agreement (strong dan, not necessarily 10 dan): **1,500-5,000 GPU-hours**

---

## 8. Feasibility Assessment

### What 2668 GPU-hours CAN achieve:
- Full BC pretraining to expert-level prediction accuracy
- Substantial offline RL (CQL) fine-tuning
- Limited online self-play (~10-20M games at our engine speed)
- A model that plays at strong amateur / low-dan level
- Probably 70-80% agreement with expert play

### What 2668 GPU-hours CANNOT achieve:
- LuckyJ-level (10+ dan) performance
- Extensive hyperparameter search
- League-based training with multiple agents
- Many iterations of the train->evaluate->iterate cycle

### Hardware context:
- Quadro RTX 5000: 11.2 TFLOPS FP32, 16GB GDDR6, 384 Tensor Cores
- 4 per Frontera node, likely running 1 training job across all 4
- bf16 training supported via Tensor Cores

### The verdict:

**Option (b) is the answer: pursue radically more efficient approach.**

Specifically:
1. **Maximize BC phase**: Use ALL available Tenhou expert data. This is the cheapest path to competence.
2. **CQL over PPO**: Mortal's offline RL approach is dramatically more sample-efficient than online self-play. Reuse logged data.
3. **Oracle guiding**: Train with perfect-information oracle first, distill to imperfect-information policy.
4. **Single focused RL run**: No hyperparameter sweeps. Pick proven hyperparameters from Mortal/Suphx.
5. **Accept 80% as ceiling**: 80% agreement with expert play is achievable. 90%+ (10 dan) is not at this budget.
6. **Future compute**: If initial results are promising, apply for additional allocation (TACC, NSF, etc.)

### Risk assessment:
- **High confidence** (>90%): BC phase succeeds, model predicts expert moves at ~65-70% top-1 accuracy
- **Medium confidence** (50-70%): RL phase lifts performance to ~75-80% agreement  
- **Low confidence** (<20%): Reaching 10+ dan with 2668 GPU-hours alone
- **Negligible confidence** (<5%): Matching LuckyJ's stable 10.68 dan at this budget

---

## Sources

1. Li et al. "Suphx: Mastering Mahjong with Deep Reinforcement Learning." arXiv:2003.13590, 2020.
2. Fu et al. "Actor-Critic Policy Optimization in a Large-Scale Imperfect-Information Game." ICLR 2022.
3. Liu et al. "Opponent-Limited Online Search for Imperfect Information Games." ICML 2023.
4. Brown & Sandholm. "Superhuman AI for Multiplayer Poker." Science, 2019.
5. Vinyals et al. "Grandmaster Level in StarCraft II Using Multi-Agent RL." Nature, 2019.
6. Hilton et al. "Scaling Laws for Single-Agent Reinforcement Learning." arXiv:2301.13442, 2023.
7. Li et al. "LsAc*-MJ: A Low-Resource Consumption RL Model for Mahjong." IJIS, 2024.
8. Mortal Documentation: mortal.ekyu.moe
9. Haobo Fu personal page: haobofu.github.io
10. TACC Frontera Documentation: docs.tacc.utexas.edu/hpc/frontera/
