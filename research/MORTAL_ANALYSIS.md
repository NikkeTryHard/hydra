# Mortal Analysis

Consolidated reference for the Mortal Mahjong AI — architecture, limitations, ecosystem, and community findings.

## Mortal Architecture Summary

### Neural Network

Mortal's backbone is a ResNet with Channel Attention (Squeeze-Excitation style). The v4 observation shape is **(1012, 34)**, representing 1012 channels across 34 tile types (source: `libriichi/src/consts.rs:15` — `obs_shape(4)` returns `[1012, 34]`; the 1012 channels are constructed in `libriichi/src/state/obs_repr.rs`). The action space consists of **46 discrete actions**: indices 0–36 map to discard or kan for each tile, 37 is riichi declaration, 38–40 are the three chi variants, 41 is pon, 42 is open kan, 43 is agari (win), 44 is ryuukyoku (draw), and 45 is pass.

All versions use the **Dueling DQN** decomposition `Q = V + A − mean(A)`, but the head architecture differs by version:

| Version | V/A Head Implementation | Feature dim |
|---------|------------------------|-------------|
| v1 | Separate `nn.Linear(512, 1)` / `nn.Linear(512, ACTION_SPACE)` from VAE latent (μ, log σ) | 512 |
| v2 | Separate 2-layer MLPs (1024 → 512 → 1 and 1024 → 512 → 46), Mish activation | 1024 |
| v3 | Separate 2-layer MLPs (1024 → 256 → 1 and 1024 → 256 → 46), Mish activation | 1024 |
| v4 | Single `nn.Linear(1024, 1 + ACTION_SPACE)`, split into V(1) and A(46) post-hoc | 1024 |

In v4, V and A share parameters in a single linear layer — the decomposition formula is still applied, but there are no separate streams. A **GRP** (GRU-based Rank Predictor) head predicts final placement probabilities.

Source: `mortal/model.py` (DQN class, Brain class), `libriichi/src/consts.rs`

### Training Algorithm

Training uses **DQN + Conservative Q-Learning (CQL)** for offline training. The CQL loss is computed as the difference between the logsumexp of Q-outputs (across the action dimension) and the mean Q-value: `logsumexp(q_out, dim=-1).mean() − q.mean()`. CQL is disabled during online training mode.

**Reward shaping** relies on the GRP head, which predicts rank probabilities. The reward signal is **delta expected value**: `E[pts]_t − E[pts]_{t−1}`, where the pts vector is **[3, 1, −1, −3]** corresponding to 1st through 4th place finishes.

Source: `mortal/train.py:237`, `mortal/reward_calculator.py:10` (pts), `:36-37` (delta)

### Training Pipeline Details

Key architectural decisions discovered from source analysis:

- **Q-targets use Monte Carlo returns, not TD bootstrapping.** `q_target = gamma^steps_to_done * kyoku_reward` — no bootstrap from next-state Q-values. This explains why GRP predicts game-level reward.
- **GRP is pretrained separately** (`train_grp.py`) with cross-entropy on 24-class placement permutations, then frozen during DQN training. It's not jointly trained.
- **No target network.** Vanilla DQN without double-DQN or EMA target. Known to cause training instability — Hydra's PPO approach avoids this entirely.
- **v4 DQN head uses a single shared linear layer** (`nn.Linear(1024, 1 + ACTION_SPACE)`) — V and A are split post-hoc from the same output, unlike v1–v3 which had separate V/A head networks. The dueling decomposition formula `Q = V + A − mean(A)` is still applied.
- **SP calculator assumes tsumo-only agari** (`is_ron: false` hardcoded in `calc.rs`). All expected values ignore ron possibilities, undervaluing hands with good ron waits.
- **Training data uses suit augmentation** — tile suits (manzu/pinzu/souzu) are permuted during training for 6× data multiplier. Hydra should do the same.

### Shanten Tables

Mortal uses the **tomohxx** table-based shanten lookup algorithm. Two precomputed tables provide instant shanten calculation:

| Table | Entries | Size (raw) | Compressed |
|-------|---------|-----------|------------|
| Suhai (suited) | 1,940,777 × 10 bytes | ~19.4 MB | 191 KB (.bin.gz) |
| Jihai (honor) | 78,032 × 10 bytes | ~0.78 MB | 5.6 KB (.bin.gz) |

Indexing uses **base-5 encoding**: each tile count (0–4) is folded into a single integer via `acc * 5 + tile_count` across all tiles in the suit or honor group. This produces a unique index into the corresponding table.

Source: `libriichi/src/algo/shanten.rs:82-84`

## libriichi Python API

Mortal exposes its Rust mahjong engine to Python via PyO3 bindings as the `libriichi` package.

### Exported Modules

| Module | Contents | Purpose |
|--------|----------|---------|
| `state` | `PlayerState`, `ActionCandidate` | Game state tracking, legal action enumeration |
| `dataset` | `Gameplay`, `GameplayLoader`, `Grp` | Training data loading and replay parsing |
| `consts` | `ACTION_SPACE`, `obs_shape`, `oracle_obs_shape` | Architecture constants |
| `arena` | `OneVsThree` | Evaluation simulation (one agent vs three copies) |
| `mjai` | `Bot` | Bot interface for MJAI protocol communication |
| `stat` | Statistical counters | Performance tracking |

### Constants

- **ACTION_SPACE** = 46 (37 discard/kan + riichi + 3 chi + pon + kan + agari + ryuukyoku + pass)
- **obs_shape(version=4)** = (1012, 34)
- **oracle_obs_shape(version=4)** = (217, 34) — 51ch opponent state (3×17: hand/aka/shanten/waits/furiten) + 166ch wall (138 yama draw order + 8 rinshan + 10 dora + 10 ura). Each tile uses 2ch (one-hot identity + aka flag). The oracle observation is concatenated with the public observation along the channel dimension before the stem Conv1d, making the oracle model's input (1229, 34). Source: `invisible.rs:152-245`, `model.py:109-155`. Note: Mortal's published training pipeline never activates oracle mode — the infrastructure exists but `is_oracle=True` is never set in `train.py`.

### Usage Pattern

The typical workflow involves creating a `PlayerState` for a given seat (0–3), feeding it line-delimited JSON events via an update method, and then rendering observations at a specified version. The `GameplayLoader` handles batch loading of recorded games for training, while `OneVsThree` runs evaluation matches where one agent plays against three copies of a baseline.

Source: `libriichi/src/lib.rs`, `libriichi/src/consts.rs`

## MJAI Protocol

> See [INFRASTRUCTURE.md § MJAI Protocol](INFRASTRUCTURE.md#mjai-protocol) for the full protocol specification (message types, tile encoding, Mortal meta extensions). Hydra's canonical MJAI definition lives there.

Source: `libriichi/src/mjai/event.rs`

## Confirmed Limitations

### No Opponent Modeling

Mortal uses `SinglePlayerTables` for expected value calculation, assuming no opponent interaction. The v4 observation encoder (lines 564–624 of `obs_repr.rs`) provides no pre-computed safety features such as suji, kabe, or genbutsu analysis. There is no opponent tenpai estimation, no aggression or tendency profiling, and no tracking of opponent discard patterns for intent reading.

Source: `libriichi/src/state/obs_repr.rs` (v4 uses SinglePlayerTables at lines 564–624)

### Score Encoding Issues

The v4 observation encoding uses **dual-scale score channels**: one normalized by 100,000 (preserving coarse information for high scores) and another normalized by 30,000 (providing higher resolution for the strategically common range). Scores above 30K are degraded in the fine-grained channel but still captured by the coarse channel. There is no explicit overtake threshold encoding — the network has no direct representation of how many points are needed to change placement. This leads to miscalculated hand-building near placement thresholds (Source: Issue #111). The dual-scale encoding may contribute to unreliable decisions in high-score late-game situations where precise placement awareness matters most.

Source: GitHub Discussion #108 (about max player score in observations), Issue #111

### Training Infrastructure Bugs

Online training hangs for unknown reasons — there is an explicit bug comment in the training code acknowledging this. The workaround is subprocess spawning with a watchdog that restarts the training process when it stalls. Additionally, there are Windows compatibility issues with GRP initialization.

Source: `mortal/train.py:382-386`

### Oracle Guiding Removal

Oracle guiding (training with perfect information, then distilling to imperfect) existed in Mortal v1 and v2 but was **removed in v3**. According to Equim-chan, the reason was: "It didn't bring improvements in practice" — the removal was not motivated by throughput concerns. Oracle guiding was replaced with a **next-rank prediction** auxiliary task (implemented as `AuxNet` in `mortal/model.py`, referred to as "NextRankPredictor" in GitHub Discussion #52 where Equim-chan explains its rationale; removal of oracle guiding discussed in #102).

Source: GitHub Discussion #102 (Equim-chan)

## Community Observations

### Playstyle Statistics

Compared to akochan (another mahjong AI), Mortal plays a noticeably more conservative and efficient style:

| Metric | Mortal | Akochan | Delta |
|--------|--------|---------|-------|
| Riichi rate | 18.9% | 21.5% | −2.6pp |
| Call rate | 29.3% | 33.0% | −3.7pp |
| Deal-in rate | 11.3% | 13.0% | −1.7pp |

Despite lower action rates across the board, Mortal achieves a **higher win rate** than akochan — winning more often while declaring riichi less, calling less, and dealing in less.

Source: `docs/src/perf/strength.md`

### Known Weaknesses

**1. Orras Over-Defensive** — Mortal plays too safely in the final round (orras/south 4) when in 2nd or 3rd place, missing opportunities to overtake 1st. This is compounded by the overtake score miscalculation documented in Issue #111. The model does not adequately distinguish between "safe 2nd" and "aggressive push for 1st" situations. (Source: GitHub Issue #111, Reddit r/Mahjong)

**2. Early Riichi Push Errors** — Mortal underestimates the threat level of early riichi (turns 1–6). It pushes forward with sub-optimal hands against unknown waits, failing to recognize that early riichi declarations correlate with stronger hands and more dangerous waits. (Source: Reddit r/Mahjong)

**3. Damaten Detection Failures** — Mortal has no intent reading for silent tenpai (damaten). It relies exclusively on explicit signals like riichi declarations and visible melds. As a result, it deals into high-value silent hands that an experienced human player would recognize from discard patterns and timing tells. (Source: Reddit r/Mahjong)

**4. Efficiency Over Yaku** — Mortal prioritizes shanten reduction speed (tile efficiency) over hand value construction. It discards dora or yaku-building tiles in favor of raw efficiency improvements. This is particularly problematic in comeback situations where a high-value hand is needed — the model builds fast, cheap hands when it should be pursuing expensive ones. (Source: Reddit r/Mahjong)

**5. Coarse Placement Sensitivity** — Mortal maintains essentially the same playstyle regardless of the point spread. It does not adjust aggression levels based on specific overtake thresholds (e.g., how many points separate 2nd from 1st). The dual-scale score encoding (100K/30K channels) degrades fine-grained score information above 30K (see Score Encoding Issues above), further limiting its ability to make placement-aware decisions. (Source: GitHub Issue #111, Reddit r/Mahjong)

### General Japanese Community Feedback

Japanese mahjong community members note that Mortal struggles with **場況 (bakyou)** — reading the overall field status and table flow. It prioritizes raw efficiency over situational tactics. The rating system is also criticized for penalizing alternative playstyles that may be strategically valid. (Source: Note.com mahjong blogs, Reddit r/Mahjong)

## Rating System

### Formula

Mortal's review rating uses a **Boltzmann softmax** over Q-values to compute action probabilities:

P(action_i) = exp(Q(action_i) / τ) / Σ_j exp(Q(action_j) / τ)

The overall rating is then: **Rating = 100 × mean(P(human_action))** across all decision points in a game. A higher rating indicates the human's moves more closely match Mortal's top-rated actions.

### Criticisms

- **Near-equal penalty**: When multiple actions have similar Q-values (i.e., are nearly equivalent), the softmax distributes probability mass among them, harshly penalizing the human for picking any single one — even if all options are essentially equal in expected value.
- **Hindsight bias**: Moves are labeled as wrong based on outcome-influenced evaluation rather than decision quality given the information available at the time.
- **EV vs placement**: The rating optimizes for expected value rather than placement security, potentially marking defensive plays as suboptimal when they secure a safe finish.
- **Score capping**: The v4 model's dual-scale encoding degrades score resolution above 30K, making ratings less reliable in high-score late-game scenarios where placement decisions matter most.

Source: `mjai-reviewer` codebase, community discussions

## Policy vs Value Architecture

Comparison between original Mortal (DQN) and the Mortal-Policy fork (PPO), maintained by Nitasurin:

| Aspect | DQN (Mortal) | PPO (Mortal-Policy) |
|--------|-------------|---------------------|
| Exploration strategy | ε-greedy / Boltzmann | Entropy weight in policy loss |
| Stability | Prone to catastrophic forgetting | Clipping prevents large policy updates |
| Normalization | BatchNorm (must freeze during eval) | GroupNorm (batch-size agnostic) |
| Network output | Single Q-value per action | Separate Actor (policy distribution) + Critic (state value) |
| Strength profile | Stronger at tile efficiency | Better at tactical decisions and defense |

Source: `Nitasurin/Mortal-Policy`, GitHub Discussion #91

## Training Best Practices

Compiled from GitHub Discussions #64, #27, #70.

### Hyperparameters

- **batch_size**: Higher is better, limited only by available VRAM. Larger batches provide more stable gradient estimates.
- **sample_reuse**: Should be disabled — use fresh samples only for each training step.
- **boltzmann_epsilon**: Should be annealed over time; starting high for exploration and decreasing for stability as training progresses.
- **learning_rate**: Very low rates (e.g., 1e-10) prevent catastrophic forgetting but effectively stall learning; finding the right balance is critical.

### Architecture Tips

- Adding **dropout** to ResBlocks improves stability during the online training phase.
- **Full network fine-tuning** is required — freezing layers does not work well for this task.
- Using `torch.compile()` provides meaningful inference speed improvements (Discussion #43). Mortal uses default compilation mode.

### Training Phases

1. **Offline phase**: Behavior cloning from Tenhou and Majsoul game logs. The model learns basic play patterns from human expert data.
2. **Online phase**: Self-play reinforcement learning. Performance typically peaks around ~3 million steps, then begins to degrade due to forgetting.
3. **Bootstrap loop**: Train for a short period, use the resulting model as the `test_play` opponent, then repeat. This iterative approach helps maintain training stability.

### Known Issues

- **Catastrophic forgetting**: Performance peaks during online training then drops sharply. The model "forgets" patterns learned during offline training as self-play shifts the distribution.
- **Q-value collapse**: After degradation sets in, all action Q-values converge to similar numbers, making the policy effectively random. This is the terminal failure mode of extended online training.

Source: GitHub Discussions #64, #27, #70

## 3-Player (Sanma) Adaptations

> Sanma is a stretch goal for Hydra. See [ECOSYSTEM.md](ECOSYSTEM.md) for the Meowjong reference implementation and [REFERENCES.md](REFERENCES.md) for the `mjai-reviewer3p` fork. Key differences from 4-player: remove 2-8m tiles (108 total), disable chi, add nukidora (N wind as bonus tile, action 40), shrink player arrays to 3. Requires `libriichi3p` fork.

## Ecosystem

### mjai-reviewer Tools

| Tool | Author | Purpose |
|------|--------|---------|
| mjai-reviewer | Equim-chan | CLI tool generating HTML review reports from game logs |
| mjai.ekyu.moe | Equim-chan | Web interface for instant game reviews without local setup |
| mjai-reviewer3p | hidacow | Fork adding 3-player (sanma) review support |
| killer_mortal_gui | killerducky | Enhanced statistics display with deal-in heuristics |
| crx-mortal | announce | Chrome extension for in-browser analysis on Majsoul/Tenhou |

### Mortal Forks

| Fork | Author | Key Difference |
|------|--------|---------------|
| Mortal-Policy | Nitasurin | Replaces DQN with PPO, uses GroupNorm instead of BatchNorm, adds entropy weight for exploration |

Source: `Nitasurin/Mortal-Policy`, GitHub Discussion #91

### Integration Projects

| Project | Architecture | Notes |
|---------|-------------|-------|
| Akagi | MITM bridge | Real-time assistant for Majsoul and Tenhou, intercepts game traffic and queries Mortal for recommendations |
| Riki | Client helper | Integration with Riichi City client |
| kanachan | Transformer-based | Independent mahjong AI (not a Mortal fork), trained on 65M+ rounds, uses a C++ simulator |

Source: `shinkuan/Akagi`, `Cryolite/kanachan`

## Rust ML Ecosystem

> See [ECOSYSTEM.md § Rust Inference Engines](ECOSYSTEM.md#4-inference--deployment) and [REFERENCES.md § Components](REFERENCES.md#components) for the full Rust ML tooling survey (Burn, tch-rs, ort, tract, candle) and distributed training patterns.
