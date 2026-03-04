# Alternative Training Paradigms: Beyond Standard Self-Play

**Date**: 2026-03-03
**Purpose**: Survey alternatives to standard self-play RL (PPO/ACH/R-NaD) for stronger policies with equal or less compute.
**Relevance**: Hydra's Phase 2 (oracle distillation) and Phase 3 (league self-play) could benefit from these approaches.

---

## Executive Summary

| Paradigm | Beats Self-Play? | Measured Gains | Compute Cost | Hydra Relevance |
|---|---|---|---|---|
| Offline RL (CQL/IQL/DT) | No (ceiling = dataset) | Bootstrapping only | Lower | Phase 1 warm-start |
| Expert Iteration (ExIt) | Yes | Defeats pure RL baselines | Higher (search) | Phase 3 upgrade |
| Counterfactual (CFR) | Yes (IIGs) | Foundation of poker AI | Variable | Complementary to RL |
| Imagination (LAMIR) | Yes | Up to 80% WR vs R-NaD | Higher | Promising but immature |
| Inverse RL | Uncertain | No game AI evidence | High | Low priority |
| Multi-task/Auxiliary | Improves sample efficiency | 2-5x faster convergence | Neutral | Already in Hydra spec |
| Asymmetric (Oracle) | Yes | Suphx: top 0.01% Tenhou | Moderate | Phase 2 is this |
| Student of Games | Yes | Beats SOTA in poker/Scotland Yard | Higher | Future consideration |

**Bottom line**: ExIt (search-guided training) and asymmetric oracle training are the two most
actionable paradigms for Hydra. Imagination-augmented (LAMIR) is the most exciting recent
development but needs maturation for 4-player mahjong scale.

---

## 1. Offline RL on Expert Data (CQL, IQL, Decision Transformer)

### What It Is

Train a policy entirely from a static dataset of expert games (no environment interaction).
Three main approaches:

- **CQL** (Conservative Q-Learning): Learns Q-values but penalizes Q-values for out-of-distribution
  actions via `logsumexp(Q) - mean(Q)` regularization. Prevents overestimation.
- **IQL** (Implicit Q-Learning): Avoids querying Q-values on unseen actions entirely. Uses
  expectile regression on the dataset's Q-distribution.
- **Decision Transformer (DT)**: Reformulates RL as sequence modeling. Conditions on desired
  return and autoregressively predicts actions. No Q-values at all.

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

**Takeaway**: No universal winner. CQL excels on lower-quality sparse data. IQL excels on
dense-reward high-quality data. DT is most stable/low-variance across settings.

### Mortal's Use of CQL

Mortal uses CQL specifically in its **offline training mode** (DeepWiki: Mortal Training Pipeline):
- Combined loss = DQN loss (MSE to MC Q-targets) + CQL loss * `min_q_weight` + next-rank loss
- CQL is active during offline training from historical Tenhou logs
- CQL is **disabled** during online self-play (where `min_q_weight = 0`)

### CQL Limitations (Critical for Hydra)

1. **Dataset ceiling**: CQL can never exceed the quality of the expert data it trains on.
   The conservative penalty actively prevents exploration beyond the dataset distribution.
2. **Conservative bias**: By design, CQL underestimates Q-values. This makes it safe but
   suboptimal -- the policy becomes overly cautious.
3. **No self-improvement**: Unlike online RL, CQL cannot discover novel strategies. It can
   only compress and generalize existing expert behavior.
4. **Distribution mismatch**: If the dataset has systematic biases (e.g., all players from
   one rank tier), CQL will inherit those biases.
5. **Hyperparameter sensitivity**: The `min_q_weight` balance between DQN loss and CQL
   regularization requires careful tuning. Too high = too conservative, too low = overestimation.

**Verdict for Hydra**: CQL is useful **only for Phase 1 warm-start** from expert logs.
It cannot replace online self-play for Phase 3. Mortal's architecture confirms this --
they use CQL offline then switch to pure online RL.

**Sources**: [CQL Paper (NeurIPS 2020)](https://arxiv.org/abs/2006.04779) |
[CQL vs IQL vs DT Comparison](https://arxiv.org/abs/2511.16475) |
[Mortal Training Pipeline](https://deepwiki.com/Equim-chan/Mortal/3.3-training-pipeline)

---

## 2. Expert Iteration (ExIt)

### What It Is

ExIt ("Thinking Fast and Slow with Deep Learning and Tree Search", Anthony et al. 2017)
decomposes learning into two interacting systems:

1. **Expert (slow)**: Tree search (MCTS or CFR) that produces strong but expensive policies
2. **Apprentice (fast)**: Neural network that learns to imitate the search output

The loop:
```
Repeat:
  1. Expert uses search (guided by current apprentice) to produce improved action targets
  2. Apprentice trains on these search-generated targets via supervised learning
  3. Apprentice's improved policy guides the expert's search in the next iteration
```

This is essentially what AlphaGo/AlphaZero does: MCTS generates training targets, and the
policy network learns to predict those targets. AlphaZero IS Expert Iteration.

### Why ExIt Beats Pure RL

The key insight: **search produces higher-quality training signal than raw RL returns**.

In pure RL (e.g., PPO), the policy gradient uses noisy game outcomes as the training signal.
In ExIt, the search process looks ahead many moves and produces a more informed action
distribution. The neural network then learns from this better signal.

### Measured Results

- **Hex**: ExIt outperforms REINFORCE for training neural Hex players. The final ExIt agent
  (trained tabula rasa) defeats **MoHex 1.0** (strongest publicly available Olympiad champion
  at time of publication).
- **Go (AlphaZero)**: AlphaZero (which is ExIt with MCTS) defeats Stockfish, Elmo, and the
  original AlphaGo without any human data.
- **Quality delta**: The search "expert" consistently provides better training targets than
  the network alone, and this gap persists even as the network improves (because search
  depth keeps amplifying the network's improvements).

### Applicability to Mahjong / Hydra

**Challenge**: ExIt requires a search procedure. For imperfect-info games like mahjong,
standard MCTS doesn't work. You need:
- CFR-based search (like Student of Games uses), or
- Information-set MCTS (IS-MCTS), or
- Learned-model search (like LAMIR)

**Opportunity**: If Hydra implements inference-time search (which is already planned per the
spec), ExIt is the natural training paradigm. Instead of pure PPO self-play, use search at
training time to generate stronger training targets, then distill into the policy network.

**Estimated compute cost**: Higher than pure RL per sample (search is expensive), but
potentially much better sample efficiency -- fewer total environment steps needed.

**Sources**: [ExIt Paper (NeurIPS 2017)](https://arxiv.org/abs/1705.08439) |
[AlphaZero Paper](https://arxiv.org/abs/1712.01815)

---

## 3. Hindsight Learning / Counterfactual Training

### What It Is

Two distinct concepts here:

**A. Hindsight Experience Replay (HER)** -- Andrychowicz et al. 2017
- Originally for goal-conditioned robotics with sparse rewards
- After a failed trajectory, relabel the goal to be what was actually achieved
- Turns every failure into a successful training example for some goal
- **Not directly applicable to competitive games** (no goal relabeling analog)

**B. Counterfactual Regret Minimization (CFR)** -- Zinkevich et al. 2007
- THE method for imperfect-information games (poker, etc.)
- Asks: "What regret would I have for not playing action X, across all possible hidden states?"
- Iteratively minimizes total counterfactual regret, converging to Nash equilibrium
- **Pluribus** (superhuman 6-player poker) and **Libratus** both use CFR variants

### Game AI Applications

**CFR for Mahjong (CFR-p, arXiv:2307.12087)**:
- Applies CFR to two-player mahjong with hierarchical abstraction
- Game-theoretic analysis + winning-policy-based abstraction
- Demonstrates CFR feasibility for mahjong variants, though 4-player Riichi is much larger

**ReBeL (Brown et al. 2020, Facebook AI)**:
- Combines CFR with learned value networks
- Self-play generates data, CFR resolves subgames at test time
- Achieves strong performance in poker and Liar's Dice
- Key innovation: treats belief states as "public states" and learns values over them

**Counterfactual value networks (DeepStack)**:
- Learns a "what-if" value function: given this hidden state, what would each action be worth?
- This is inherently counterfactual -- evaluating unchosen actions across unobserved states

### Applicability to Hydra

The counterfactual perspective is already embedded in CFR-based approaches. For Hydra:
- **Phase 3 could incorporate CFR-style reasoning** instead of/alongside PPO
- The danger-head and tenpai-head in Hydra's architecture are already a form of counterfactual
  reasoning ("what would happen if opponent is tenpai?")
- Full CFR is likely too expensive for 4-player Riichi's game tree, but **depth-limited CFR
  with learned values** (as in ReBeL/Student of Games) is feasible

**Sources**: [HER Paper](https://arxiv.org/abs/1707.01495) |
[CFR-p for Mahjong](https://arxiv.org/abs/2307.12087) |
[ReBeL Paper](https://arxiv.org/abs/2007.13544)

---

## 4. Imagination-Augmented Training (Learned World Models)

### What It Is

MuZero (Schrittwieser et al. 2020) learns a world model in latent space:
- **Representation**: encodes observations into latent states
- **Dynamics**: predicts next latent state given action
- **Prediction**: outputs policy, value, and reward from latent state

Training generates "imagined" trajectories in latent space, providing extra training data
beyond real experience. This is like dreaming -- the model practices in its imagination.

### LAMIR: Extending to Imperfect-Information Games (Oct 2024, arXiv:2510.05048)

**LAMIR** (Learned Abstract Model for Imperfect-information Reasoning) is the most relevant
recent work. Key innovations:

1. **Information-set representations**: Learns latent representations of players' belief states
   (not just world states), capturing what each player knows
2. **Abstract subgame construction**: Learns a domain-independent abstraction of information
   sets, capped at size L, making subgames tractable
3. **CFR+ resolving at test time**: Instead of MCTS (unsound for IIGs), uses CFR+ with
   continual resolving over the learned model
4. **Depth-limited search**: Learned value functions at the horizon boundary

### Measured Results (Beating R-NaD)

Head-to-head win rates vs RNaD (3M training episodes):

| Game | LAMIR Win Rate |
|---|---|
| II Goofspiel 10 | **54.5% +/- 0.25** |
| II Goofspiel 13 | **60.7% +/- 0.34** |
| II Goofspiel 15 | **80.5% +/- 0.26** |

These are massive wins. The advantage grows with game complexity, suggesting learned models
become more valuable as games get larger/harder.

### Limitations (from the paper)

- Does **not explicitly model chance nodes** (relevant for mahjong's tile draws)
- CFR guarantees may weaken with imperfect-recall abstractions
- Action-space size is not abstracted (mahjong has 46 actions, manageable)
- Only tested on Goofspiel variants so far, not on games at mahjong's scale

### Applicability to Hydra

**High potential but high risk.** LAMIR's approach is exactly what Hydra would need for
search-augmented training in a IIG. However:
- 4-player Riichi Mahjong is vastly larger than Goofspiel
- The chance-node limitation is a real problem (tile draws are central to mahjong)
- Implementation complexity is significant (learned model + CFR resolving + value networks)

**Recommendation**: Monitor LAMIR closely. If the approach scales to larger games in future
work, it could be Hydra's Phase 4 upgrade. Not ready for Phase 3 today.

**Sources**: [MuZero Paper](https://arxiv.org/abs/1911.08265) |
[LAMIR Paper (2024)](https://arxiv.org/abs/2510.05048) |
[Demystifying MuZero](https://arxiv.org/abs/2411.04580)

---

## 5. Inverse RL from Expert Play

### What It Is

Instead of defining a reward function and optimizing it, IRL:
1. Observes expert behavior (human pro mahjong games)
2. Infers what reward function the expert must be optimizing
3. Uses that learned reward to train an RL agent

The idea: human experts might be optimizing for subtle objectives that hand-crafted reward
functions miss (e.g., "this discard is safe AND develops hand flexibility AND signals to
opponents I'm not dangerous").

### State of the Art (2024-2025)

Recent survey (Springer 2025): IRL is advancing but primarily in robotics and autonomous
driving, not competitive games.

- **AIRL + reward shaping** (arXiv:2410.03847): Model-based reward shaping for adversarial
  IRL. Improves performance in stochastic environments. No game applications.
- **Potential-based reward shaping for IRL** (ICLR 2025): Reduces computational burden of
  IRL sub-problems. Theoretical contribution, not game-specific.
- **Gamer behavior decoding** (Yale 2024): Uses IRL to understand player motivations in
  gaming. Analytical, not for training stronger agents.

### Could This Capture Nuances That Placement Score Misses?

**In theory, yes.** If you had a large dataset of 10-dan games and ran IRL on them, you
might discover reward shaping terms that placement-based rewards miss. For example:
- Implicit risk preferences (not just expected value but variance aversion)
- Tempo/pace-of-play preferences
- Meta-game signaling rewards

**In practice, doubtful.** Problems:
1. IRL is computationally expensive (requires solving many forward RL problems)
2. The recovered reward is often degenerate (multiple rewards explain the same behavior)
3. No demonstrated improvement over hand-crafted rewards in competitive game AI
4. Mahjong's stochasticity makes reward inference very noisy

**Verdict for Hydra**: Low priority. The reward design in REWARD_DESIGN.md (placement-based
with RVR variance reduction) is likely sufficient. If anything, the multi-head architecture
(value + GRP + tenpai + danger) already captures the nuances that IRL would discover.

**Sources**: [IRL Survey (Springer 2025)](https://link.springer.com/article/10.1007/s00521-025-11100-0) |
[Model-Based Reward Shaping for AIRL](https://arxiv.org/abs/2410.03847)

---

## 6. Multi-Task Learning / Auxiliary Objectives

### What It Is

Train the model on multiple related tasks simultaneously. The shared representation learns
features useful across all tasks, improving generalization and sample efficiency.

### Evidence Base

**UNREAL (Jaderberg et al. 2017, DeepMind)**:
- Added auxiliary tasks (reward prediction, pixel control, feature control) to A3C
- **10x median improvement** across 57 Atari games
- Auxiliary tasks act as "free" additional gradient signal

**Comparing Auxiliary Tasks for RL (arXiv:2310.04241, ICLR venue)**:
Most helpful auxiliary tasks ranked:
1. **Forward state prediction (fsp)**: predict next observation given current obs + action
2. **Forward state-difference prediction (fsdp)**: predict delta between observations
3. **Reward prediction (rwp)**: least helpful of the three

Key finding: **auxiliary tasks help more as task complexity increases.** Simple environments
see minimal benefit; complex environments (like mahjong!) see large gains.

### What Hydra Already Has

Hydra's spec already includes multi-task heads:
- **Value head**: scalar expected placement score
- **GRP head (24-way)**: global reward prediction (placement distribution)
- **Tenpai head (3-way)**: opponent tenpai probability
- **Danger head (3x34)**: per-tile danger probabilities per opponent

Mortal uses: **next-rank prediction** as its auxiliary task.

### What Could Be Added

Additional auxiliary objectives that could help:
1. **Opponent action prediction**: predict what each opponent will discard next
2. **Tile draw prediction**: predict distribution over next drawn tile (given visible info)
3. **Hand reconstruction**: predict opponents' hidden hands from visible information
4. **Shanten prediction**: predict own/opponents' shanten count
5. **Forward state prediction**: predict next game state features after your action

### Measured Improvement Expectations

Based on the auxiliary task literature:
- Sample efficiency improvement: **2-5x** for complex tasks (UNREAL benchmarks)
- Maximum performance improvement: **moderate** (helps learn faster, eventual ceiling similar)
- Most benefit during **early/mid training**, diminishing returns at convergence
- The tenpai and danger heads in Hydra already capture the most important auxiliary signals

**Verdict for Hydra**: Already well-served by current design. Adding opponent-action prediction
as a 6th head would be the highest-value addition. Low-hanging fruit since the encoder
already processes all visible game state.

**Sources**: [UNREAL Paper](https://arxiv.org/abs/1611.05397) |
[Auxiliary Task Comparison](https://arxiv.org/abs/2310.04241) |
[Hydra Spec](research/design/HYDRA_SPEC.md)

---

## 7. Asymmetric Self-Play (Oracle-Student Training)

### What It Is

During training, one agent (the "oracle") sees hidden information that the other agent
(the "student") doesn't. The oracle's superior play provides a stronger training signal.

Two main approaches:
1. **Oracle as opponent**: Oracle plays against student, student learns from harder games
2. **Oracle as teacher**: Oracle's value estimates guide the student's learning (distillation)

### Suphx's Oracle Guiding (Li et al. 2020, Microsoft Research)

Suphx pioneered this for mahjong:

1. **Train an oracle agent** that sees all players' tiles (perfect information)
2. **Oracle produces value estimates** for each game state
3. **Student agent learns from oracle's value function** via distillation, but plays with
   only its own visible information at test time
4. **Global reward prediction** provides the reward signal

**Results**: Suphx reached the top **0.01%** of all officially ranked human players on Tenhou,
achieving a stable rating above **10-dan** level. This was the first AI to outperform most
top human players in Mahjong.

### Why It Works

The oracle sees ground truth (all tiles), so its value estimates are much more accurate
than values learned from partial information. When the student distills from these estimates:
- It learns better representations of hidden state
- It gets a lower-variance training signal
- It converges faster because the teacher already "knows the answer"

Think of it like having the answer key while studying -- you learn more efficiently even
though you won't have the answer key during the test.

### Latest Research on Asymmetric Training

**Student of Games (SoG, Schmid et al. 2023, Science Advances)**:
- Unifies search + self-play + game-theoretic reasoning
- Uses **growing-tree CFR (GT-CFR)** for sound search in both perfect and imperfect info games
- Beats strongest openly available agent in heads-up no-limit Texas hold'em
- Defeats SOTA agent in Scotland Yard
- Achieves strong performance in chess and Go

SoG's "sound self-play" ensures the search-generated training data doesn't introduce
exploitable biases, which is a known risk of naive asymmetric training.

**DeepNash (Perolat et al. 2022, Science)**:
- R-NaD (Regularized Nash Dynamics) for Stratego
- Model-free, search-free, pure self-play
- Achieves human-expert level, top-3 all-time on Gravon platform
- Key insight: R-NaD converges TO Nash equilibrium instead of cycling around it
- Not asymmetric, but relevant as the baseline that LAMIR beats

### Applicability to Hydra

**This IS Hydra's Phase 2.** The training pipeline already specifies oracle distillation:
- Phase 1: Supervised warm-start from expert logs
- Phase 2: Oracle distillation (oracle sees all tiles, student learns from oracle values)
- Phase 3: League self-play (student plays against itself and past versions)

The Suphx evidence strongly supports this pipeline. The question is whether to enhance
Phase 2 with search (making it ExIt-style oracle distillation) or keep it pure value
distillation.

**Recommendation**: Phase 2 as designed is well-supported by evidence. Consider adding
search-guided training in Phase 3 (ExIt-style) for additional improvement.

**Sources**: [Suphx Paper](https://arxiv.org/abs/2003.13590) |
[Student of Games (Science Advances 2023)](https://www.science.org/doi/10.1126/sciadv.adg3256) |
[DeepNash / R-NaD (Science 2022)](https://www.science.org/doi/10.1126/science.add4679)

---

## 8. Recent Papers Beating Standard Self-Play (2024-2025)

### LAMIR (Oct 2024) -- Learned World Model + CFR for IIGs

Already covered in Section 4. Up to **80% win rate** vs R-NaD in Goofspiel variants.
The most impressive recent result for alternatives to standard self-play in IIGs.

### Student of Games (2023, published Science Advances)

Already covered in Section 7. First algorithm to achieve strong performance across both
perfect AND imperfect information games with a single unified approach.

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
- Meta-learning approach: learn which CFR variant works best for a given game
- Future direction for automating the search component

### Self-Play Survey (Aug 2024, arXiv:2408.01072)

Comprehensive survey classifying all self-play methods in RL:
- Categorizes by: opponent selection, learning dynamics, convergence properties
- Identifies open challenges: non-stationarity, catastrophic forgetting, scalability
- Covers: fictitious play, PSRO, R-NaD, population-based training, league training

---

## Hydra-Specific Recommendations

### Tier 1: Already in Pipeline (High Confidence)

1. **Offline RL warm-start (Phase 1)**: CQL or simple behavioral cloning on expert logs.
   Use this to get a reasonable starting policy before expensive self-play.
2. **Oracle distillation (Phase 2)**: Suphx-style asymmetric training. Strong evidence
   from Suphx's 10-dan results. Already in Hydra's training spec.
3. **Multi-task auxiliary heads**: Tenpai, danger, GRP heads already specified. These
   provide free gradient signal during training.

### Tier 2: Strong Evidence, Worth Implementing (Medium Effort)

4. **Expert Iteration for Phase 3**: Instead of pure PPO self-play, use search at training
   time to generate stronger training targets. This is what makes AlphaZero work. Requires
   implementing a search procedure for 4-player mahjong (significant effort).
5. **Opponent action prediction head**: Add as 6th auxiliary head. Predicts opponent discards.
   Low implementation cost, moderate training signal benefit.

### Tier 3: Promising but Premature (Watch List)

6. **LAMIR-style learned world model**: Most exciting recent development. 80% WR vs R-NaD.
   But not tested at mahjong scale, and chance-node modeling is unsolved. Monitor closely.
7. **CFR-based training (ReBeL-style)**: Counterfactual reasoning during training could
   produce more robust policies. Requires significant infrastructure. Consider for Phase 4.
8. **Student of Games**: Most general approach. If Hydra later wants to support multiple
   game types or integrate search+game-theory, SoG is the template.

### Not Recommended

9. **Inverse RL**: No evidence it improves on hand-crafted rewards for games. High compute
   cost. The multi-head architecture already captures what IRL would discover.
10. **Decision Transformer**: Poor fit for competitive games. Requires conditioning on
    desired return at inference time, which is awkward for multi-player competitive settings.

---

## Key Insight for Hydra

The biggest delta between "standard self-play" and "state-of-the-art training" is
**search-guided training signal quality**. Every major advance (AlphaZero, ExIt, Student
of Games, LAMIR) achieves its gains by using search to generate better training targets
than raw RL returns provide.

For Hydra, this means: **the planned inference-time search (from the spec) isn't just an
inference-time upgrade -- it's a training paradigm upgrade.** Once search is working, it
should be integrated into the training loop (ExIt-style) for Phase 3, not just used at
test time.

The compute tradeoff: search at training time is expensive per sample, but the sample
efficiency gains typically more than compensate. AlphaZero uses ~100x fewer environment
interactions than pure PPO to reach the same strength, because each interaction produces
a much higher-quality training signal.
