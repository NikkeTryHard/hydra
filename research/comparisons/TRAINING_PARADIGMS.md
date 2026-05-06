# Alternative Training Paradigms: Beyond Standard Self-Play

**Date**: 2026-03-03
**Purpose**: Survey alternatives to standard self-play RL (PPO/ACH/R-NaD) for stronger policies with equal or less compute.
**Relevance**: Hydra's Phase 2 (oracle distillation) and Phase 3 (league self-play) may gain from these approaches.

---

## Executive Summary

| Paradigm | Beats Self-Play? | Measured Gains | Compute Cost | Hydra Relevance |
|---|---|---|---|---|
| Offline RL (CQL/IQL/DT) | No (ceiling = dataset) | Bootstrapping only | Lower | Phase 1 warm-start |
| Expert Iteration (ExIt) | Yes | Beats pure RL baselines | Higher (search) | Phase 3 upgrade |
| Counterfactual (CFR) | Yes (IIGs) | Foundation of poker AI | Variable | Complementary to RL |
| Imagination (LAMIR) | Yes | Up to 80% WR vs R-NaD | Higher | Promising but immature |
| Inverse RL | Uncertain | No game AI evidence | High | Low priority |
| Multi-task/Auxiliary | Better sample efficiency | 2-5x faster convergence | Neutral | Already in Hydra spec |
| Asymmetric (Oracle) | Yes | Suphx: top 0.01% Tenhou | Moderate | Phase 2 is this |
| Student of Games | Yes | Beats SOTA in poker/Scotland Yard | Higher | Future consideration |

**Bottom line**: ExIt (search-guided training) and asymmetric oracle training most actionable for Hydra. Imagination-augmented (LAMIR) most exciting recent path but needs maturation for 4-player mahjong scale.

---

## 1. Offline RL on Expert Data (CQL, IQL, Decision Transformer)

### What It Is

Train policy entirely from static expert-game dataset (no environment interaction).
Three main approaches:

- **CQL** (Conservative Q-Learning): Learns Q-values but penalizes Q-values for out-of-distribution actions via `logsumexp(Q) - mean(Q)` regularization. Prevents overestimation.
- **IQL** (Implicit Q-Learning): Avoids querying Q-values on unseen actions entirely. Uses expectile regression on dataset's Q-distribution.
- **Decision Transformer (DT)**: Reformulates RL as sequence modeling. Conditions on desired return and autoregressively predicts actions. No Q-values.

### Measured Comparisons (Caunhye & Jeewa 2025, arXiv:2511.16475)

D4RL Ant continuous-control benchmarks (normalized score, 4 seeds):

| Dataset | Reward | CQL | IQL | DT |
|---|---|---|---|---|
| medium | sparse | **91.55** | 84.49 | 87.9 |
| medium-replay | sparse | **71.99** | 42.14 | 66.3 |
| medium-expert | sparse | 103.38 | 85.95 | **120.6** |
| medium | dense | **99.49** | 95.5 | 88.0 |
| medium-replay | dense | 92.99 | **97.5** | 88.07 |
| medium-expert | dense | 107.0 | **124.2** | 90.24 |

**Compute**: DT 7.5h, CQL 5.0h, IQL 2.0h (100k steps, 4 seeds).

**Takeaway**: No universal winner. CQL best on lower-quality sparse data. IQL best on dense-reward high-quality data. DT most stable/low-variance across settings.

### Mortal's Use of CQL

Mortal uses CQL specifically in its **offline training mode** (DeepWiki: Mortal Training Pipeline):
- Combined loss = DQN loss (MSE to MC Q-targets) + CQL loss * `min_q_weight` + next-rank loss
- CQL active during offline training from historical Tenhou logs
- CQL **disabled** during online self-play (where `min_q_weight = 0`)

### CQL Limitations (Critical for Hydra)

1. **Dataset ceiling**: CQL cannot exceed quality of expert data. Conservative penalty actively blocks exploration beyond dataset distribution.
2. **Conservative bias**: By design, CQL underestimates Q-values. Safe but suboptimal. Policy becomes overly cautious.
3. **No self-improvement**: Unlike online RL, CQL cannot discover novel strategies. It only compresses and generalizes existing expert behavior.
4. **Distribution mismatch**: If dataset has systematic biases (e.g. all players from one rank tier), CQL inherits those biases.
5. **Hyperparameter sensitivity**: `min_q_weight` balance between DQN loss and CQL regularization needs careful tuning. Too high = too conservative, too low = overestimation.

**Verdict for Hydra**: CQL useful **only for Phase 1 warm-start** from expert logs.
It cannot replace online self-play for Phase 3. Mortal architecture confirms this --
they use CQL offline then switch to pure online RL.

**Sources**: [CQL Paper (NeurIPS 2020)](https://arxiv.org/abs/2006.04779) |
[CQL vs IQL vs DT Comparison](https://arxiv.org/abs/2511.16475) |
[Mortal Training Pipeline](https://deepwiki.com/Equim-chan/Mortal/3.3-training-pipeline)

---

## 2. Expert Iteration (ExIt)

### What It Is

ExIt ("Thinking Fast and Slow with Deep Learning and Tree Search", Anthony et al. 2017)
splits learning into two interacting systems:

1. **Expert (slow)**: Tree search (MCTS or CFR) that produces strong but expensive policies
2. **Apprentice (fast)**: Neural network that learns to imitate search output

Loop:
```
Repeat:
  1. Expert uses search (guided by current apprentice) to produce improved action targets
  2. Apprentice trains on these search-generated targets via supervised learning
  3. Apprentice's improved policy guides the expert's search in the next iteration
```

This is AlphaGo/AlphaZero: MCTS generates training targets, policy network learns to predict them. AlphaZero IS Expert Iteration.

### Why ExIt Beats Pure RL

Key insight: **search produces higher-quality training signal than raw RL returns**.

In pure RL (e.g. PPO), policy gradient uses noisy game outcomes as training signal.
In ExIt, search looks ahead many moves and produces more informed action distribution. Neural network then learns from better signal.

### Measured Results

- **Hex**: ExIt beats REINFORCE for training neural Hex players. Final ExIt agent (trained tabula rasa) defeats **MoHex 1.0** (strongest publicly available Olympiad champion at publication time).
- **Go (AlphaZero)**: AlphaZero (ExIt with MCTS) defeats Stockfish, Elmo, and original AlphaGo without human data.
- **Quality delta**: Search "expert" consistently gives better training targets than network alone, and gap persists even as network improves (because search depth keeps amplifying network improvements).

### Applicability to Mahjong / Hydra

**Challenge**: ExIt needs search procedure. For imperfect-info games like mahjong, standard MCTS fails. Need:
- CFR-based search (like Student of Games)
- Information-set MCTS (IS-MCTS)
- Learned-model search (like LAMIR)

**Opportunity**: If Hydra implements inference-time search (already planned per spec), ExIt is natural training paradigm. Instead of pure PPO self-play, use search at training time to generate stronger training targets, then distill into policy network.

**Estimated compute cost**: Higher than pure RL per sample (search expensive), but likely much better sample efficiency -- fewer total environment steps needed.

**Sources**: [ExIt Paper (NeurIPS 2017)](https://arxiv.org/abs/1705.08439) |
[AlphaZero Paper](https://arxiv.org/abs/1712.01815)

---

## 3. Hindsight Learning / Counterfactual Training

### What It Is

Two distinct concepts here:

Hindsight Experience Replay (HER)** -- Andrychowicz et al. 2017
- Originally for goal-conditioned robotics with sparse rewards
- After failed trajectory, relabel goal to what was achieved
- Turns every failure into successful training example for some goal
- **Not directly applicable to competitive games** (no goal-relabel analog)

**B. Counterfactual Regret Minimization (CFR)** -- Zinkevich et al. 2007
- Main method for imperfect-information games (poker, etc.)
- Asks: "What regret would I have for not playing action X, across all possible hidden states?"
- Iteratively minimizes total counterfactual regret, converging to Nash equilibrium
- **Pluribus** (superhuman 6-player poker) and **Libratus** both use CFR variants

### Game AI Applications

**CFR for Mahjong (CFR-p, arXiv:2307.12087)**:
- Applies CFR to two-player mahjong with hierarchical abstraction
- Game-theoretic analysis + winning-policy-based abstraction
- Shows CFR feasibility for mahjong variants, though 4-player Riichi far larger

**ReBeL (Brown et al. 2020, Facebook AI)**:
- Combines CFR with learned value networks
- Self-play generates data, CFR resolves subgames at test time
- Achieves strong performance in poker and Liar's Dice
- Key innovation: treats belief states as "public states" and learns values over them

**Counterfactual value networks (DeepStack)**:
- Learns "what-if" value function: given hidden state, what would each action be worth?
- This is inherently counterfactual -- evaluating unchosen actions across unobserved states

### Applicability to Hydra

Counterfactual view already embedded in CFR-based approaches. For Hydra:
- **Phase 3 could incorporate CFR-style reasoning** instead of/alongside PPO
- Danger-head and tenpai-head in Hydra architecture already form counterfactual reasoning ("what would happen if opponent is tenpai?")
- Full CFR likely too expensive for 4-player Riichi game tree, but **depth-limited CFR with learned values** (as in ReBeL/Student of Games) is feasible

**Sources**: [HER Paper](https://arxiv.org/abs/1707.01495) |
[CFR-p for Mahjong](https://arxiv.org/abs/2307.12087) |
[ReBeL Paper](https://arxiv.org/abs/2007.13544)

---

## 4. Imagination-Augmented Training (Learned World Models)

### What It Is

MuZero (Schrittwieser et al. 2020) learns world model in latent space:
- **Representation**: encodes observations into latent states
- **Dynamics**: predicts next latent state given action
- **Prediction**: outputs policy, value, and reward from latent state

Training generates "imagined" trajectories in latent space, giving extra training data beyond real experience. Like dreaming -- model practices in imagination.

### LAMIR: Extending to Imperfect-Information Games (Oct 2024, arXiv:2510.05048)

**LAMIR** (Learned Abstract Model for Imperfect-information Reasoning) most relevant recent work. Key innovations:

1. **Information-set representations**: Learns latent representations of players' belief states, not only world states, capturing what each player knows
2. **Abstract subgame construction**: Learns domain-independent abstraction of information sets, capped at size L, making subgames tractable
3. **CFR+ resolving at test time**: Instead of MCTS (unsound for IIGs), uses CFR+ with continual resolving over learned model
4. **Depth-limited search**: Learned value functions at horizon boundary

### Measured Results (Beating R-NaD)

Head-to-head win rates vs RNaD (3M training episodes):

| Game | LAMIR Win Rate |
|---|---|
| II Goofspiel 10 | **54.5% +/- 0.25** |
| II Goofspiel 13 | **60.7% +/- 0.34** |
| II Goofspiel 15 | **80.5% +/- 0.26** |

These are huge wins. Advantage grows with game complexity, suggesting learned models become more valuable as games get larger/harder.

### Limitations (from the paper)

- Does **not explicitly model chance nodes** (relevant for mahjong tile draws)
- CFR guarantees may weaken with imperfect-recall abstractions
- Action-space size not abstracted (mahjong has 46 actions, manageable)
- Only tested on Goofspiel variants so far, not on games at mahjong scale

### Applicability to Hydra

**High potential but high risk.** LAMIR approach is exactly what Hydra would need for search-augmented training in IIG. However:
- 4-player Riichi Mahjong far larger than Goofspiel
- Chance-node limitation is real problem (tile draws central to mahjong)
- impl complexity significant (learned model + CFR resolving + value networks)

**rec**: Monitor LAMIR closely. If approach scales to larger games in future work, it could be Hydra's Phase 4 upgrade. Not ready for Phase 3 today.

**Sources**: [MuZero Paper](https://arxiv.org/abs/1911.08265) |
[LAMIR Paper (2024)](https://arxiv.org/abs/2510.05048) |
[Demystifying MuZero](https://arxiv.org/abs/2411.04580)

---

## 5. Inverse RL from Expert Play

### What It Is

Instead of defining reward function and optimizing it, IRL:
1. Observes expert behavior (human pro mahjong games)
2. Infers what reward function expert must optimize
3. Uses learned reward to train RL agent

Idea: human experts may optimize subtle objectives that hand-crafted reward functions miss (e.g. "this discard is safe AND develops hand flexibility AND signals to opponents I'm not dangerous").

### State of the Art (2024-2025)

Recent survey (Springer 2025): IRL advancing but mostly in robotics and autonomous driving, not competitive games.

- **AIRL + reward shaping** (arXiv:2410.03847): Model-based reward shaping for adversarial IRL. Improves performance in stochastic environments. No game applications.
- **Potential-based reward shaping for IRL** (ICLR 2025): Reduces computational burden of IRL sub-problems. Theoretical contribution, not game-specific.
- **Gamer behavior decoding** (Yale 2024): Uses IRL to understand player motivations in gaming. Analytical, not for training stronger agents.

### Could This Capture Nuances That Placement Score Misses?

**In theory, yes.** With large dataset of 10-dan games and IRL, you might discover reward-shaping terms that placement-based rewards miss. e.g.:
- Implicit risk preferences (not only expected value but variance aversion)
- Tempo/pace-of-play preferences
- Meta-game signaling rewards

**In practice, doubtful.** Problems:
1. IRL is computationally expensive (requires solving many forward RL problems)
2. Recovered reward often degenerate (multiple rewards explain same behavior)
3. No demonstrated improvement over hand-crafted rewards in competitive game AI
4. Mahjong stochasticity makes reward inference noisy

**Verdict for Hydra**: Low priority. Reward design in REWARD_DESIGN.md (placement-based with RVR variance reduction) likely sufficient. If anything, multi-head architecture (value + GRP + tenpai + danger) already captures nuances IRL would discover.

**Sources**: [IRL Survey (Springer 2025)](https://link.springer.com/article/10.1007/s00521-025-11100-0) |
[Model-Based Reward Shaping for AIRL](https://arxiv.org/abs/2410.03847)

---

## 6. Multi-Task Learning / Auxiliary Objectives

### What It Is

Train model on multiple related tasks simultaneously. Shared representation learns features useful across all tasks, improving generalization and sample efficiency.

### Evidence Base

**UNREAL (Jaderberg et al. 2017, DeepMind)**:
- Added auxiliary tasks (reward prediction, pixel control, feature control) to A3C
- **10x median improvement** across 57 Atari games
- Auxiliary tasks act as "free" extra gradient signal

**Comparing Auxiliary Tasks for RL (arXiv:2310.04241, ICLR venue)**:
Most helpful auxiliary tasks ranked:
1. **Forward state prediction (fsp)**: predict next observation given current obs + action
2. **Forward state-difference prediction (fsdp)**: predict delta between observations
3. **Reward prediction (rwp)**: least helpful of three

Key finding: **auxiliary tasks help more as task complexity increases.** Simple environments get little benefit; complex environments (like mahjong) get big gains.

### What Hydra Already Has

Hydra spec already includes multi-task heads:
- **Value head**: scalar expected placement score
- **GRP head (24-way)**: global reward prediction (placement distribution)
- **Tenpai head (3-way)**: opponent tenpai probability
- **Danger head (3x34)**: per-tile danger probabilities per opponent

Mortal uses: **next-rank prediction** as auxiliary task.

### What Could Be Added

Additional auxiliary objectives that could help:
1. **Opponent action prediction**: predict what each opponent will discard next
2. **Tile draw prediction**: predict distribution over next drawn tile (given visible info)
3. **Hand reconstruction**: predict opponents' hidden hands from visible information
4. **Shanten prediction**: predict own/opponents' shanten count
5. **Forward state prediction**: predict next game-state features after your action

### Measured Improvement Expectations

Based on auxiliary-task literature:
- Sample efficiency improvement: **2-5x** for complex tasks (UNREAL benchmarks)
- Maximum performance improvement: **moderate** (learns faster, eventual ceiling similar)
- Most benefit during **early/mid training**, diminishing returns at convergence
- Tenpai and danger heads in Hydra already capture most important auxiliary signals

**Verdict for Hydra**: Current design already strong. Adding opponent-action prediction as 6th head highest-value addition. Low impl cost, moderate training-signal gain.

**Sources**: [UNREAL Paper](https://arxiv.org/abs/1611.05397) |
[Auxiliary Task Comparison](https://arxiv.org/abs/2310.04241) |
[Hydra Final](../design/HYDRA_FINAL.md)

---

## 7. Asymmetric Self-Play (Oracle-Student Training)

### What It Is

During training, one agent ("oracle") sees hidden information that other agent ("student") does not. Oracle's stronger play gives stronger training signal.

Two main approaches:
1. **Oracle as opponent**: Oracle plays against student, student learns from harder games
2. **Oracle as teacher**: Oracle's value estimates guide student's learning (distillation)

### Suphx's Oracle Guiding (Li et al. 2020, Microsoft Research)

Suphx pioneered this for mahjong:

1. **Train oracle agent** that sees all players' tiles (perfect information)
2. **Oracle produces value estimates** for each game state
3. **Student agent learns from oracle's value function** via distillation, but plays with only its own visible information at test time
4. **Global reward prediction** provides reward signal

**Results**: Suphx reached top **0.01%** of all officially ranked human players on Tenhou, achieving stable rating above **10-dan** level. First AI to outperform most top human mahjong players.

### Why It Works

Oracle sees ground truth (all tiles), so its value estimates are far more accurate than values learned from partial information. When student distills from these estimates:
- It learns better hidden-state representations
- It gets lower-variance training signal
- It converges faster because teacher already "knows answer"

Like having answer key while studying -- learn more efficiently even though answer key absent at test time.

### Latest Research on Asymmetric Training

**Student of Games (SoG, Schmid et al. 2023, Science Advances)**:
- Unifies search + self-play + game-theoretic reasoning
- Uses **growing-tree CFR (GT-CFR)** for sound search in both perfect and imperfect info games
- Beats strongest openly available agent in heads-up no-limit Texas hold'em
- Defeats SOTA agent in Scotland Yard
- Achieves strong performance in chess and Go

SoG's "sound self-play" ensures search-generated training data does not introduce exploitable biases, known risk in naive asymmetric training.

**DeepNash (Perolat et al. 2022, Science)**:
- R-NaD (Regularized Nash Dynamics) for Stratego
- Model-free, search-free, pure self-play
- Achieves human-expert level, top-3 all-time on Gravon platform
- Key insight: R-NaD converges TO Nash equilibrium instead of cycling around it
- Not asymmetric, but relevant as baseline that LAMIR beats

### Applicability to Hydra

**This IS Hydra's Phase 2.** Training pipeline already specifies oracle distillation:
- Phase 1: Supervised warm-start from expert logs
- Phase 2: Oracle distillation (oracle sees all tiles, student learns from oracle values)
- Phase 3: League self-play (student plays against itself and past versions)

Suphx evidence strongly supports this pipeline. Question: enhance Phase 2 with search (making it ExIt-style oracle distillation) or keep pure value distillation.

**rec**: Phase 2 as designed has strong evidence support. Consider search-guided training in Phase 3 (ExIt-style) for more improvement.

**Sources**: [Suphx Paper](https://arxiv.org/abs/2003.13590) |
[Student of Games (Science Advances 2023)](https://www.science.org/doi/10.1126/sciadv.adg3256) |
[DeepNash / R-NaD (Science 2022)](https://www.science.org/doi/10.1126/science.add4679)

---

## 8. Recent Papers Beating Standard Self-Play (2024-2025)

### LAMIR (Oct 2024) -- Learned World Model + CFR for IIGs

Already covered in Section 4. Up to **80% win rate** vs R-NaD in Goofspiel variants.
Most impressive recent result for alternatives to standard self-play in IIGs.

### Student of Games (2023, published Science Advances)

Already covered in Section 7. First algorithm to achieve strong performance across both perfect AND imperfect information games with one unified approach.

### SPIRAL (2025) -- Self-Play for LLM Reasoning

- Uses self-play on zero-sum games to improve LLM reasoning
- Not directly applicable to game AI, but shows self-play principles extending to new domains
- Source: [github.com/spiral-rl/spiral](https://github.com/spiral-rl/spiral)

### Dynamic Discounted CFR (DDCFR, 2024-2025)

- Automatically adjusts discounting weights in CFR variants
- Improves convergence rate over vanilla CFR, CFR+, DCFR
- Relevant for any approach using CFR-based search/training

### Auto-designing CFR Algorithms (AIJ 2024)

- Sciencedirect paper on automatically designing CFR algorithms for IIGs
- Meta-learning approach: learn which CFR variant works best for given game
- Future direction for automating search component

### Self-Play Survey (Aug 2024, arXiv:2408.01072)

Comprehensive survey classifying all self-play methods in RL:
- Categorizes by: opponent selection, learning dynamics, convergence properties
- Identifies open challenges: non-stationarity, catastrophic forgetting, scalability
- Covers: fictitious play, PSRO, R-NaD, population-based training, league training

---

## Hydra-Specific Recommendations

### Tier 1: Already in Pipeline (High Confidence)

1. **Offline RL warm-start (Phase 1)**: CQL or simple behavioral cloning on expert logs.
Use this to get reasonable starting policy before expensive self-play.
2. **Oracle distillation (Phase 2)**: Suphx-style asymmetric training. Strong evidence
from Suphx's 10-dan results. Already in Hydra's training spec.
3. **Multi-task auxiliary heads**: Tenpai, danger, GRP heads already specified. These
give free gradient signal during training.

### Tier 2: Strong Evidence, Worth Implementing (Medium Effort)

4. **Expert Iteration for Phase 3**: Instead of pure PPO self-play, use search at training
time to generate stronger training targets. This is what makes AlphaZero work. Requires
search procedure for 4-player mahjong (significant effort).
5. **Opponent action prediction head**: Add as 6th auxiliary head. Predicts opponent discards.
Low impl cost, moderate training-signal benefit.

### Tier 3: Promising but Premature (Watch List)

6. **LAMIR-style learned world model**: Most exciting recent development. 80% WR vs R-NaD.
But not tested at mahjong scale, and chance-node modeling unsolved. Monitor closely.
7. **CFR-based training (ReBeL-style)**: Counterfactual reasoning during training could
produce more robust policies. Requires significant infrastructure. Consider for Phase 4.
8. **Student of Games**: Most general approach. If Hydra later wants to support multiple
game types or integrate search+game-theory, SoG is template.

### Not Recommended

9. **Inverse RL**: No evidence it beats hand-crafted rewards for games. High compute
cost. Multi-head architecture already captures what IRL would discover.
10. **Decision Transformer**: Poor fit for competitive games. Requires conditioning on
desired return at inference time, awkward for multi-player competitive settings.

---

## Key Insight for Hydra

Biggest delta between "standard self-play" and "state-of-the-art training" is
**search-guided training signal quality**. Every major advance (AlphaZero, ExIt, Student
of Games, LAMIR) gets gains by using search to generate better training targets
than raw RL returns.

For Hydra, this means: **planned inference-time search (from spec) is not only
inference-time upgrade -- it is training paradigm upgrade.** Once search works, it
should enter training loop (ExIt-style) for Phase 3, not only test time.

Compute tradeoff: search at training time is expensive per sample, but sample
efficiency gains usually more than repay cost. AlphaZero uses ~100x fewer environment
interactions than pure PPO to reach same strength, because each interaction produces
much higher-quality training signal.