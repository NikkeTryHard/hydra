# Mahjong AI Research Landscape

Comprehensive survey of academic papers, AI systems, repositories, and methodologies for building stronger mahjong AI.

---

## 1. Academic Papers

### 1.1 Suphx: Mastering Mahjong with Deep Reinforcement Learning

- **Authors**: Junjie Li, Sotetsu Koyamada, Qiwei Ye, Guoqing Liu, Chao Wang, Ruihan Yang, Li Zhao, Tao Qin, Tie-Yan Liu, Hsiao-Wuen Hon
- **Year**: 2020
- **Venue**: arXiv:2003.13590 [cs.LG]
- **URL**: https://arxiv.org/abs/2003.13590

**Key Contributions:**

1. **Oracle Guiding**: Train an "Oracle" agent with perfect information (sees wall + opponents' hands), then distill knowledge to a "Normal" agent that only sees observable information. Solves the exploration problem in imperfect-information games.
2. **Global Reward Prediction (GRP)**: Neural network mapping current state → expected final rank probability distribution. Enables rank-aware play (defensive when leading, aggressive when behind).
3. **Run-time Policy Adaptation**: Lightweight MCTS combined with policy network for dynamic mid-game adjustments based on GRP and shanten.

**Performance**: First AI to reach Tenhou 10-dan (only ~180 humans ever achieved this).

**Architecture**: 50+ layer deep CNN processing 16×34 plane state representation.

---

### 1.2 Tjong: Transformer-based Mahjong AI

- **Authors**: Xiali Li, Bo Liu, Zhi Wei, Zhaoqi Wang, Licheng Wu
- **Year**: 2024
- **Venue**: CAAI Transactions on Intelligence Technology
- **URL**: https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cit2.12298

**Key Contributions:**

1. **Hierarchical Decision-Making**: Decouples action selection (Discard/Chi/Pon/Riichi) from tile selection. Reduces action space explosion.
2. **Transformer Architecture**: Treats game state as token sequence. Multi-head self-attention captures long-range dependencies between early discards and current decisions.
3. **Fan Backward**: Reward shaping that propagates Yaku (scoring pattern) potential backward through decision chain. Prevents greedy speed optimization, encourages high-value hands.

**Code**: No public implementation available.

---

### 1.3 Information Set Monte Carlo Tree Search (ISMCTS)

- **Authors**: P. I. Cowling, E. J. Powley, D. Whitehouse
- **Year**: 2012
- **Venue**: IEEE Transactions on Computational Intelligence and AI in Games
- **URL**: https://ieeexplore.ieee.org/document/6203567

**Key Contribution**: Foundation for handling imperfect information in games like Mahjong. Samples possible "worlds" for hidden tiles during tree search.

---

### 1.4 NAGA (Neural Architectural Game Agent)

- **Developer**: Dwango Media Village (Yuri Odagiri)
- **Year**: 2018 (initial), ongoing
- **URL**: https://dmv.nico/en/articles/mahjong_ai_naga/

**Key Features:**

- Deep CNN trained on hundreds of millions of Tenhou Houou-room games
- Achieved Tenhou 10-dan
- Famous "Confidence Bar" showing agreement with human moves
- Commercial service (not open-source)

---

### 1.5 Phoenix: Open-Source Reproducible Mahjong Agent

- **Year**: 2023
- **URL**: https://csci527-phoenix.github.io/documents/Paper.pdf

**Key Contribution**: Transparent baseline for Mahjong research with interpretable decision-making.

---

### 1.6 Bakuuchi (Historical)

- **Authors**: Mizukami et al.
- **Year**: 2014
- **Title**: "Real-time Mahjong AI based on Monte Carlo Tree Search"

**Significance**: Pre-deep learning SOTA using ISMCTS + rule-based heuristics.

---

## 2. Alternative AI Systems

### 2.1 Comparison Table

| System | Developer | Architecture | Training | Code |
|--------|-----------|--------------|----------|------|
| **Mortal** | Equim-chan | ResNet + Channel Attention | DQN + CQL | [Open Source](https://github.com/Equim-chan/Mortal) |
| **Kanachan** | Cryolite | Transformer (BERT-style) | Offline RL / SL | [Open Source](https://github.com/Cryolite/kanachan) |
| **Akochan** | Critter | C++ Heuristics + Search | EV Calculation | [Open Source](https://github.com/critter-mj/akochan) |
| **NAGA** | Dwango | Deep CNN | Supervised (Human) | [Commercial](https://dmv.nico/en/articles/mahjong_ai_naga/) |
| **Suphx** | Microsoft | Deep CNN + Oracle | Self-play RL | Proprietary |
| **AlphaJong** | Jimboom7 | Heuristic Simulation | Look-ahead Search | [Open Source](https://github.com/Jimboom7/AlphaJong) |

---

### 2.2 Kanachan (Cryolite)

- **Repository**: https://github.com/Cryolite/kanachan
- **Stars**: 326

**Architecture Philosophy**: "No Human-crafted Features"

- Everything is tokenized (tiles, events, positions)
- BERT-style Transformer encoder learns relationships from data
- Separate decoders for different objectives (Imitation → Score → Rank)

**Components:**

- `src/annotation/`: C++ log converter for Mahjong Soul WebSocket data
- `src/simulation/`: High-fidelity C++ Mahjong Soul rule simulator
- `src/xiangting/`: LOUDS-based TRIE shanten calculator
- `kanachan/training/`: PyTorch Transformer training

**Data**: Trained on 100M+ Mahjong Soul rounds (10x larger than Tenhou datasets)

**Key Difference from Mortal**: Learns shanten, suji, defense purely from data rather than encoded features.

---

### 2.3 Akochan

- **Repository**: https://github.com/critter-mj/akochan

**Architecture**: C++ heuristic engine with EV (Expected Value) search

- Calculates tile efficiency using shanten tables
- Defensive metrics: Suji, Kabe, Genbutsu analysis
- Interpretable decisions (shows exact EV for each discard)

**Why Still Relevant**:

- Backend for many analysis tools
- Faster than neural networks for bulk analysis
- Transparent reasoning vs black-box NNs

---

### 2.4 AlphaJong

- **Repository**: https://github.com/Jimboom7/AlphaJong
- **Type**: Heuristic-based (NOT AlphaZero/neural)

> "The AI does not use machine learning, but conventional algorithms. Simply said it's simulating some turns and looking for the best move." — README

**Architecture**: Browser userscript for Mahjong Soul with tunable constants for defense/offense balance.

**Note**: No true AlphaZero (MCTS+NN self-play) mahjong implementation exists publicly. Mahjong's imperfect information makes traditional MCTS computationally impractical without heavy modifications like ISMCTS.

---

## 3. GitHub Repositories

### 3.1 AI Engines

| Repository | Stars | Language | Description |
|------------|-------|----------|-------------|
| [Equim-chan/Mortal](https://github.com/Equim-chan/Mortal) | 1,334 | Rust/Python | State-of-the-art open-source Riichi AI |
| [Cryolite/kanachan](https://github.com/Cryolite/kanachan) | 326 | C++/Python | Transformer-based framework |
| [erreurt/MahjongAI](https://github.com/erreurt/MahjongAI) | 446 | Python | Extensible general-purpose agent |
| [gimite/mjai-manue](https://github.com/gimite/mjai-manue) | 37 | Ruby | Original MJAI client |

### 3.2 Analysis & Review Tools

| Repository | Stars | Description |
|------------|-------|-------------|
| [Equim-chan/mjai-reviewer](https://github.com/Equim-chan/mjai-reviewer) | 1,168 | Review game logs with MJAI engines |
| [Xerxes-2/mjai-batch-review](https://github.com/Xerxes-2/mjai-batch-review) | 9 | Batch analyze Mahjong Soul logs |

### 3.3 Protocol & Infrastructure

| Repository | Description |
|------------|-------------|
| [gimite/mjai](https://github.com/gimite/mjai) | Original MJAI protocol server |
| [tomohxx/mjai-gateway](https://github.com/tomohxx/mjai-gateway) | MJAI ↔ Tenhou translator |
| [shinkuan/Akagi](https://github.com/shinkuan/Akagi) | Real-time MITM assistant for Majsoul/Tenhou |

### 3.4 Simulators & Platforms

| Repository | Description |
|------------|-------------|
| [smly/mjai.app](https://github.com/smly/mjai.app) | Web-based mahjong simulator |

### 3.5 Utility Libraries

| Search Term | Purpose |
|-------------|---------|
| `shanten-calculator` | Fast shanten computation |
| `mahjong-utils` | Tile efficiency, ukeire |
| `tenhou-log` | Log parsing and conversion |
| `mjlog-parser` | MJLOG format handling |

---

## 4. Training Methodologies

### 4.1 Reinforcement Learning Approaches

**Dueling DQN** (used by Mortal, Suphx)

Decomposes the Q-function into a state-value component V(s) and an action-advantage component A(s,a), combining them as Q(s,a) = V(s) + A(s,a) − mean(A). This separation allows the network to learn which states are inherently valuable independently of the action taken, which is especially useful when many actions have similar value. Preferred for discrete action spaces like tile discard selection.

**Conservative Q-Learning (CQL)**

Designed for offline training from human game logs. CQL penalizes Q-values for out-of-distribution actions—those the agent might consider but that never appeared in the training data. The conservative penalty is computed as the log-sum-exp of Q-values across all actions minus the mean Q-value for dataset actions. This prevents the dangerous overestimation of unseen actions that plague standard offline RL.

**PPO (Mortal-Policy fork)**

Policy gradient method with a clipped surrogate objective that bounds policy updates, preventing destructively large steps. More stable than DQN for online self-play training. Implementations targeting mahjong use GroupNorm instead of BatchNorm for compatibility with varying batch sizes during self-play data collection.

### 4.2 Reward Shaping

**Global Reward Prediction (Suphx)**

Predicts the final rank probability distribution at every decision point during a game. The reward signal at each turn is defined as the change in expected final points between successive states. This approach solves the credit assignment problem over mahjong's long decision horizons—individual discard choices are evaluated by their impact on the predicted game outcome rather than waiting for the final result.

**Delta-EV Reward**

Computes the reward as the difference in expected points between the current state and the previous state. Mortal applies a point vector of [3, 1, −1, −3] that heavily penalizes finishing in 4th place, reflecting ranked lobby scoring where avoiding last place is paramount.

**Fan Backward (Tjong)**

Propagates Yaku (scoring pattern) potential backward through the decision chain. Actions that maintain paths toward high-value hands receive positive shaping rewards, while actions that collapse hand value receive penalties. This prevents the common failure mode where agents optimize purely for speed (tenpai as fast as possible) at the expense of hand value.

### 4.3 Dataset Sources

| Source | Volume | Quality | Access |
|--------|--------|---------|--------|
| Tenhou Phoenix | ~17M rounds | Very High (Dan players) | Log download tools |
| Mahjong Soul | 100M+ rounds | Mixed (all ranks) | WebSocket capture |
| Self-play | Unlimited | Depends on policy | Generated |

### 4.4 Training Challenges

**Catastrophic Forgetting**

Performance typically peaks at approximately 3M online training steps then degrades. Q-values converge and the policy becomes increasingly deterministic, reducing exploration of novel situations. Mitigation strategies include auxiliary prediction tasks that maintain representational diversity, and learning rate scheduling that gradually reduces update magnitudes.

**Sample Efficiency**

Mahjong has an estimated 10^60 possible game states, making exhaustive coverage impossible. Efficient feature encoding is critical—Mortal compresses the observable game state into a 1012-plane observation tensor over 34 tile positions, balancing expressiveness against tractability.

**Exploration vs Exploitation**

Boltzmann exploration with epsilon-based temperature tuning creates a convergence dilemma: too much exploration prevents policy refinement, too little locks in suboptimal strategies. CQL provides a natural regularizer by penalizing Q-values for risky unexplored actions, biasing the agent toward well-understood play patterns during offline training.

---

## 5. Architectural Innovations

### 5.1 Network Architectures

**ResNet + Channel Attention (Mortal)**

Deep residual blocks with skip connections enable training very deep networks without vanishing gradients. Squeeze-Excitation style channel attention layers learn to weight the importance of different observation planes—for example, learning that the "discarded by kamicha" plane matters more for defense decisions than the "round wind" plane.

**Transformer (Kanachan, Tjong)**

Treats the game state as a sequence of tokens (tiles, events, player positions) and applies multi-head self-attention to capture long-range dependencies. A discard made in the first few turns can influence tenpai patterns many turns later—attention mechanisms handle these temporal relationships more naturally than CNN or RNN architectures.

### 5.2 Observation Encoding

**Multi-Plane One-Hot (Mortal, Suphx)**

Encodes the game state as a stack of binary planes over a 4×9 grid (one position per tile type, 34 total). Each plane represents a different aspect of the state: tile counts in hand, tiles discarded by each player, dora indicators, and positional information. Mortal v4 uses 1012 planes × 34 tiles, creating a rich but high-dimensional input tensor.

**Learned Embeddings (Kanachan)**

Treats tiles as vocabulary tokens in a language-model style embedding. The model learns tile relationships entirely from data—discovering suji patterns, yaku compatibility, and defensive safety without any hand-crafted encoding. This approach sacrifices interpretability for generality.

### 5.3 Novel Techniques

**Oracle Guiding (Suphx)**

A three-phase training process: first, train an Oracle agent that sees all hidden information (the wall and opponents' hands); second, distill the Oracle's knowledge into a Normal agent that operates with only public information; third, fine-tune the Normal agent through self-play. The Oracle transfers "intuition" about likely hidden states, bootstrapping the Normal agent past the cold-start problem.

**Hierarchical Decision (Tjong)**

Decomposes the action space into two levels. The high-level network first selects an action type (discard, chi, pon, kan, riichi, tsumo, ron, or pass). The low-level network then selects the specific tile for that action. This hierarchical structure reduces the combinatorial action space and allows specialized reasoning at each level.

**Global Reward Prediction**

Maps the current game state to a probability distribution over final rank positions (1st through 4th). This prediction enables rank-aware play: an agent leading in points can adopt defensive strategies to protect placement, while an agent trailing can take calculated risks. The GRP network serves both as a reward shaping mechanism during training and as a strategic signal during inference.

### 5.4 Output Heads

**Dueling Structure**

Separates the network's output into two streams: V(s), the state value head estimating how good the current situation is regardless of action, and A(s,a), the action advantage head estimating how much better a specific action is compared to the average. This decomposition allows the network to learn state values even for actions rarely taken, improving sample efficiency.

**Action Masking**

A binary mask identifying all legal actions in the current game state. Applied before the softmax activation, illegal actions receive large negative values that drive their probabilities to zero. This technique is universal across all mahjong AI implementations, as the legality constraints in Riichi mahjong are complex and context-dependent.

---

## 6. Research Gaps & Opportunities

### 6.1 Opponent Modeling

Most AIs (including Mortal v4) treat mahjong as "Single Player Tables"—they optimize their own play without explicit modeling of opponent behavior. There is no tenpai prediction based on discard patterns, no behavioral profiling during a game, and no adjustment to individual opponent tendencies. Cross-attention mechanisms that attend to opponent discard sequences could enable real-time threat assessment and adaptive defense.

### 6.2 Situational Adaptation

Current systems struggle with Orras (final round) point-based decisions, where optimal play depends heavily on the current score differential rather than hand quality alone. Agents tend toward conservative play when aggressive pushes are needed to avoid 4th place, and vice versa. Better integration of Global Reward Prediction with score-aware training could address this gap.

### 6.3 Yaku Awareness

Neural agents tend to optimize for tile efficiency (reaching tenpai quickly) over hand value (building expensive scoring patterns). They frequently miss dama (hidden tenpai) opportunities where declaring riichi would be suboptimal. Fan Backward style reward shaping, as proposed by Tjong, could train agents to maintain high-value hand paths rather than collapsing to the fastest tenpai.

### 6.4 Pure Rust Implementation

The current state-of-the-art (Mortal) splits its stack between Rust for game logic and Python for neural network training and inference. A full Rust implementation using emerging ML frameworks like Burn or Candle could enable unified zero-copy inference, eliminating the FFI boundary overhead and enabling tighter integration between game simulation and model evaluation.

---

*Last Updated: 2026-02-04 | Verified via spam-investigation (14 agents, 3 rounds)*
