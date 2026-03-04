# Mahjong AI Systems -- Ranked by Novelty of Approach

**Criteria**: Pure theoretical/design novelty. Performance results are IRRELEVANT.
"Novel" = introduced a technique or framing that did not previously exist in Mahjong AI or game AI more broadly.

---

## Tier 1: Genuinely Novel Theoretical Contributions

### 1. JueJong / ACH (Tencent, ICLR 2022) -- MOST NOVEL
**Paper**: "Actor-Critic Policy Optimization in a Large-Scale Imperfect-Information Game"
**Authors**: Haobo Fu et al. (Tencent AI Lab)

**What's genuinely new**:
- **Actor-Critic Hedge (ACH)** algorithm: A completely new actor-critic objective for imperfect-information games. Instead of maximizing discounted return (standard policy gradient), it minimizes weighted cumulative counterfactual regret using Hedge-based policy updates.
- **Neural-Weighted CFR (NW-CFR)**: New CFR variant that uses only trajectory samples (no full game-tree traversal) with neural function approximation for regret targets. Has convergence guarantee with O(T^{-1/2}) exploitability decrease.
- **Theoretical bridge**: Unifies actor-critic RL with CFR theory -- two previously separate paradigms in game AI. This is the key insight: they proved ACH is a practical implementation of NW-CFR.

**Why it's #1**: This doesn't just apply existing RL to Mahjong. It creates a fundamentally new algorithm class that bridges two theoretical traditions (RL actor-critic + CFR regret minimization). The convergence proof for the neural approximation case is genuinely new theory.

**Note**: LuckyJ (the 10-dan Tenhou bot) is the *productionized version* of this research line. The paper itself doesn't name it LuckyJ, but Haobo Fu's page confirms the connection.

---

### 2. Suphx (Microsoft Research Asia, 2020)
**Paper**: "Suphx: Mastering Mahjong with Deep Reinforcement Learning" (arXiv:2003.13590)
**Authors**: Junjie Li, Sotetsu Koyamada, Qiwei Ye, et al.

**What's genuinely new**:
- **Oracle Guiding**: Training with perfect information (oracle that sees all hidden tiles), then distilling knowledge to the imperfect-information agent. Novel teacher-student paradigm for imperfect info games -- the agent learns WHAT to do from a god-mode oracle, then learns to approximate it from partial observations.
- **Global Reward Prediction (GRP)**: Predicting final game ranking from current state as an auxiliary training signal. Addresses the sparse reward problem in Mahjong (you only learn your placement at game end).
- **Run-time Policy Adaptation**: Dynamically adjusting policy at inference time based on the current game's trajectory. The policy shifts based on how well/poorly the current game is going -- playing more aggressively when behind, more defensively when ahead.

**Why it's #2**: Oracle guiding is a genuinely novel training paradigm for imperfect-information games. GRP as an auxiliary head for sparse-reward games was new. Run-time policy adaptation was a fresh idea for adapting play style within a single game.

---

### 3. Generative Sequence Modeling for Mahjong (Jiang et al., 2024)
**Paper**: "Generative sequence modeling for action prediction in Chinese Standard Mahjong" (SPIE 2024)

**What's genuinely new**:
- **Treats Mahjong as a language modeling problem**: Converts entire games into discrete token sequences (tile faces + actions), then trains a Transformer decoder with autoregressive next-token prediction -- exactly like GPT.
- **Cross-paradigm transfer**: Directly applies the NLP generative pre-training paradigm (GPT-style) to a game domain, without any RL at all. Pure sequence prediction.

**Why it's #3**: This is a genuine paradigm shift in framing. Every other Mahjong AI uses RL or supervised classification. This one says "a Mahjong game IS a language" and uses autoregressive generation. The novelty is in the FRAMING, not the architecture (which is standard Transformer decoder).

---

## Tier 2: Novel Application of Existing Techniques to Mahjong

### 4. Kanachan (Cryolite, open-source, ongoing)
**Repo**: github.com/Cryolite/kanachan

**What's genuinely new**:
- **BERT/Transformer encoder for Mahjong state representation**: Uses a BERT-style encoder to convert game states into latent features, with separate heads for Q-values, state-values, and action policy.
- **Offline RL for Mahjong**: Applies offline RL algorithms (IQL, ILQL, CQL) instead of the standard online self-play paradigm. Trains purely from logged game data without environment interaction.

**Why it's #4**: Transformers and offline RL both exist, but applying them together to Mahjong is novel. The offline RL angle is especially interesting -- it means you can train competitive agents from Tenhou logs alone, without a simulator.

---

### 5. Tjong (NJIT, 2024)
**Paper**: "Tjong: A transformer-based Mahjong AI via hierarchical decision-making and fan backward"

**What's genuinely new**:
- **Hierarchical action decomposition**: Decouples Mahjong decisions into two stages -- first decide WHAT to do (action type), then decide WHICH tile. Reduces decision complexity from huge flat action space to two smaller sequential choices.
- **Fan backward**: Goal-conditioned planning where the agent reasons backward from target scoring patterns (fan/yaku) to current actions. Instead of "what should I discard?", it's "I want this winning hand, what discard gets me closer?"

**Why it's #5**: The hierarchical decomposition is elegant and new for Mahjong. Fan backward is genuinely novel -- it's goal-conditioned RL where the goals are specific Mahjong scoring patterns. No other Mahjong AI explicitly does backward planning from target yaku.

---

### 6. MPPO / Styled Mahjong Agents (Peking University, 2025)
**Paper**: "Elevating Styled Mahjong Agents with Learning from Demonstration" (arXiv:2506.16995)

**What's genuinely new**:
- **Mixed Proximal Policy Optimization (MPPO)**: Unified pipeline that mixes offline demonstration data with online self-play in standard PPO training. Not two separate losses (BC + RL) -- one objective over both data sources.
- **Style-preserving skill improvement**: The explicit goal of making agents BETTER while keeping their PLAY STYLE, not just maximizing win rate. Filters demo trajectories to keep only positive-return ones to preserve teacher style.

**Why it's #6**: The problem framing is novel (style + skill simultaneously), and MPPO's unified mixing approach is cleaner than typical LfD methods. But the components (PPO, demos, filtering) individually exist.

---

## Tier 3: Solid Engineering, Limited Theoretical Novelty

### 7. Mortal (Equim-chan, open-source)
**Repo**: github.com/Equim-chan/Mortal

**What's genuinely new**:
- Not much theoretically. It's a well-engineered ResNet + PPO self-play pipeline with GRP (borrowed from Suphx).
- **Contribution is practical**: First open-source Mahjong AI that's actually competitive (10-dan class). Rust engine + Python training = high throughput (40K hanchans/hour).

**Why it's lower**: Mortal is an excellent engineering achievement but doesn't introduce new algorithms or theoretical ideas. It competently applies known techniques (ResNet, PPO, self-play, GRP auxiliary head) in a fast, open-source package.

---

### 8. LsAc*-MJ (2024)
**Paper**: "LsAc*-MJ: A Low-Resource Consumption Reinforcement Learning Model for Mahjong"

**What's genuinely new**:
- **LSTM + optimized A2C with experience replay** for low-resource Mahjong training. Focuses on making RL work without massive compute.
- Novelty is in the CONSTRAINT: can you train a decent Mahjong AI without Tencent/Microsoft-scale resources?

**Why it's #8**: The low-resource angle is practical and useful, but LSTM + A2C + experience replay are well-established. The novelty is in showing these can work for Mahjong at small scale.

---

### 9. Meowjong (2022)
**Paper**: "Building a 3-Player Mahjong AI using Deep Reinforcement Learning" (arXiv:2202.12847)

**What's genuinely new**:
- **First AI for Sanma (3-player Mahjong)**: Compact 2D encoding for Sanma's unique features, 5 separate CNN heads for different action types, MC policy gradient for discard RL.
- Domain novelty only -- applying standard deep RL to an unstudied Mahjong variant.

---

### 10. Evo-Sparrow (2025, AAAI AIIDE)
**Paper**: "Evolutionary Optimization of Deep Learning Agents for Sparrow Mahjong"

**What's genuinely new**:
- **CMA-ES (evolutionary strategy) for training an LSTM Mahjong agent** instead of gradient-based RL. Derivative-free optimization for a game agent.
- Shows comparable performance to PPO with lower training time.
- Domain novelty: first study of Sparrow Mahjong variant.

**Why it's lower**: CMA-ES + LSTM is well-known in neuroevolution. Applying it to a simplified Mahjong variant is a modest contribution.

---

## Tier 4: Historical / Foundational (Pre-Deep-RL Era)

### 11. Bakuuchi (University of Tokyo, 2014)
- **Supervised learning from expert data** (first major SL approach to Japanese Mahjong).
- Standard ML at the time, but pioneered the SL-from-human-logs paradigm that everyone later uses for warm-starting.

### 12. Mizukami & Tsuruoka (University of Tokyo, 2015)
**Paper**: "Building a Computer Mahjong Player Based on Monte Carlo Simulation and Opponent Modeling"
- **MC simulation with explicit opponent modeling**: Decomposed opponent play into waiting/winning-tiles/winning-scores predictions, then ran MC forward simulation.
- One of the first principled opponent modeling approaches for Mahjong.

### 13. NAGA (Dwango/Niconico, commercial)
- Commercial Mahjong analysis tool. Achieved 10-dan on Tenhou.
- **No published papers** -- architecture is proprietary. Likely SL + RL based on observable behavior.
- Can't rank its novelty without knowing the approach.

---

## Tier 5: Competition Methods (IJCAI Botzone)

From the IJCAI 2024 competition survey paper, the notable approaches used by competition teams include:

- **Shanten-value search + heuristic rules** (traditional, no novelty)
- **ResNet/ResNeXt supervised learning with tile-symmetry data augmentation** (modest novelty in the augmentation trick)
- **PPO + IMPALA distributed self-play + SL warm start + reward clipping** (standard modern pipeline)
- **DQN variants** (surprisingly competitive under limited compute)
- **Shanten as explicit input feature** (surprisingly effective trick)

---

## Notable Non-Mahjong Work That's Relevant

### Poker AI (CMU/Meta FAIR)
- **Libratus** (CMU, 2017): Nested subgame solving + blueprint abstraction. Not Mahjong, but the theoretical foundation for imperfect-info game solving.
- **Pluribus** (CMU+Meta, 2019): First superhuman 6-player poker AI. Depth-limited search + modified blueprint strategy.
- **Neither lab has published Mahjong-specific work** as of this search.

### DeepMind
- **No published Mahjong work found.** Their imperfect-info work focuses on Hanabi, Stratego (DeepNash), and general MARL.

### Baidu / Alibaba
- **No published Mahjong AI work found.** Haobo Fu was previously at Baidu before joining Tencent, but his Mahjong work is all under Tencent.

---

## Summary Ranking (Novelty Only)

| Rank | System | Novel Contribution | Novelty Type |
|------|--------|-------------------|--------------|
| 1 | JueJong/ACH (Tencent) | New algorithm class bridging AC + CFR with convergence proof | New theory |
| 2 | Suphx (Microsoft) | Oracle guiding, GRP, run-time policy adaptation | New training paradigms |
| 3 | Generative Seq. Model | Mahjong-as-language, GPT-style autoregressive | Paradigm reframe |
| 4 | Kanachan | BERT encoder + offline RL (IQL/CQL) for Mahjong | Novel combination |
| 5 | Tjong | Hierarchical action decomp + fan backward planning | Novel decomposition |
| 6 | MPPO/Styled Agents | Style-preserving LfD with unified PPO mixing | Novel problem + method |
| 7 | Mortal | (Engineering excellence, no new theory) | Engineering |
| 8 | LsAc*-MJ | Low-resource LSTM+A2C approach | Constraint novelty |
| 9 | Meowjong | First 3-player Sanma AI | Domain novelty |
| 10 | Evo-Sparrow | CMA-ES neuroevolution for Mahjong | Method transfer |

---

## Bottom Line

**LuckyJ/JueJong (Tencent) IS the most novel approach** -- not just for Mahjong, but arguably a genuine contribution to game theory broadly. The ACH algorithm creates a new theoretical bridge between two previously separate paradigms (actor-critic RL and counterfactual regret minimization).

**Suphx is a close second** -- oracle guiding was a genuinely clever idea for imperfect-info games.

**The dark horse for novelty is the Generative Sequence Modeling work** -- treating Mahjong as a language and using GPT-style autoregressive prediction is a radical reframing that nobody else has attempted seriously.

**Everything else** applies known techniques (RL, SL, transformers) to Mahjong with varying degrees of engineering quality.
