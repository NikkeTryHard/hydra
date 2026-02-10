# Mortal Analysis

Consolidated reference for the Mortal Mahjong AI — architecture, limitations, ecosystem, and community findings.

## Mortal Architecture Summary

### Neural Network

Mortal's backbone is a ResNet with Channel Attention (Squeeze-Excitation style). The v4 observation shape is **(1012, 34)**, representing 1012 channels across 34 tile types. The action space consists of **46 discrete actions**: indices 0–36 map to discard or kan for each tile, 37 is riichi declaration, 38–40 are the three chi variants, 41 is pon, 42 is open kan, 43 is agari (win), 44 is ryuukyoku (draw), and 45 is pass.

The network uses a **Dueling DQN** structure with separate value (V) and advantage (A) streams, combined with a **GRP** (GRU-based Rank Predictor) head that predicts final placement probabilities.

Source: `mortal/model.py`, `libriichi/src/consts.rs`

### Training Algorithm

Training uses **DQN + Conservative Q-Learning (CQL)** for offline training. The CQL loss is computed as the difference between the logsumexp of Q-outputs (across the action dimension) and the mean Q-value: `logsumexp(q_out, dim=-1).mean() − q.mean()`. CQL is disabled during online training mode.

**Reward shaping** relies on the GRP head, which predicts rank probabilities. The reward signal is **delta expected value**: `E[pts]_t − E[pts]_{t−1}`, where the pts vector is **[3, 1, −1, −3]** corresponding to 1st through 4th place finishes.

Source: `mortal/train.py:237`, `mortal/reward_calculator.py:10` (pts), `:36-37` (delta)

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
- **oracle_obs_shape(version=4)** = (217, 34)

### Usage Pattern

The typical workflow involves creating a `PlayerState` for a given seat (0–3), feeding it line-delimited JSON events via an update method, and then rendering observations at a specified version. The `GameplayLoader` handles batch loading of recorded games for training, while `OneVsThree` runs evaluation matches where one agent plays against three copies of a baseline.

Source: `libriichi/src/lib.rs`, `libriichi/src/consts.rs`

## MJAI Protocol

MJAI is a line-delimited JSON protocol for mahjong AI communication.

### Message Types

| Type | Key Fields | Description |
|------|-----------|-------------|
| `start_game` | `names: [String; 4]` | Match start, player names |
| `start_kyoku` | `bakaze, dora_marker, kyoku, honba, kyotaku, oya, scores, tehais` | Round start with full state |
| `tsumo` | `actor, pai` | Tile draw |
| `dahai` | `actor, pai, tsumogiri` | Tile discard (tsumogiri = drew and immediately discarded) |
| `chi` / `pon` | `actor, target, pai, consumed` | Sequence or triplet call |
| `daiminkan` / `kakan` / `ankan` | `actor, [target], pai, consumed` | Open kan, added kan, concealed kan |
| `reach` | `actor` | Riichi declaration |
| `hora` | `actor, target, [deltas, ura_markers]` | Win declaration |
| `ryukyoku` | `[deltas]` | Exhaustive draw |

### Tile Encoding

- **Suited tiles**: `1m`–`9m` (manzu), `1p`–`9p` (pinzu), `1s`–`9s` (souzu)
- **Red fives**: `5mr`, `5pr`, `5sr`
- **Wind honors**: `E` (East), `S` (South), `W` (West), `N` (North)
- **Dragon honors**: `P` (Haku/White), `F` (Hatsu/Green), `C` (Chun/Red)
- **Actor IDs**: 0–3 via `BoundedU8`

Source: `libriichi/src/mjai/event.rs`

### Mortal Meta Extensions

Mortal extends the MJAI protocol with a metadata structure attached to bot responses, containing the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `q_values` | `Vec<f32>` (optional) | Q-value estimate for each of the 46 possible actions |
| `mask_bits` | `u64` (optional) | Bitmask indicating which actions are legal in the current state |
| `shanten` | `i8` (optional) | Current shanten number (distance to tenpai; 0 = tenpai, −1 = complete) |
| `is_greedy` | `bool` (optional) | Whether the bot chose the action with the maximum Q-value |
| `eval_time_ns` | `u64` (optional) | Wall-clock inference time in nanoseconds |
| `at_furiten` | `bool` (optional) | Whether the player is currently in furiten (cannot ron) |
| `kan_select` | `Box<Metadata>` (optional) | Nested metadata for kan-specific decisions when a kan action triggers further choices |

Source: `libriichi/src/mjai/event.rs:141-150`

## Confirmed Limitations

### No Opponent Modeling

Mortal uses `SinglePlayerTables` for expected value calculation, assuming no opponent interaction. The v4 observation encoder (lines 564–624 of `obs_repr.rs`) provides no pre-computed safety features such as suji, kabe, or genbutsu analysis. There is no opponent tenpai estimation, no aggression or tendency profiling, and no tracking of opponent discard patterns for intent reading.

Source: `libriichi/src/state/obs_repr.rs` (v4 uses SinglePlayerTables at lines 564–624)

### Score Encoding Issues

The v4 observation encoding **caps scores at 30,000 points**, which loses information about large point spreads that occur in real games. There is no explicit overtake threshold encoding — the network has no direct representation of how many points are needed to change placement. This leads to miscalculated hand-building near placement thresholds (Source: Issue #111). The score capping also makes the model unreliable in high-score late-game situations.

Source: GitHub Discussion #108, Issue #111

### Training Infrastructure Bugs

Online training hangs for unknown reasons — there is an explicit bug comment in the training code acknowledging this. The workaround is subprocess spawning with a watchdog that restarts the training process when it stalls. Additionally, there are Windows compatibility issues with GRP initialization.

Source: `mortal/train.py:382-386`

### Oracle Guiding Removal

Oracle guiding (training with perfect information, then distilling to imperfect) existed in Mortal v1 and v2 but was **removed in v3**. According to Equim-chan, the reason was: "It didn't bring improvements in practice" — the removal was not motivated by throughput concerns. Oracle guiding was replaced with a **NextRankPredictor** auxiliary task.

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

**5. Coarse Placement Sensitivity** — Mortal maintains essentially the same playstyle regardless of the point spread. It does not adjust aggression levels based on specific overtake thresholds (e.g., how many points separate 2nd from 1st). The score encoding caps at 30k (see Score Encoding Issues above), further limiting its ability to make placement-aware decisions. (Source: GitHub Issue #111, Reddit r/Mahjong)

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
- **Score capping**: The v4 model caps at 30k, making ratings unreliable in high-score late-game scenarios where placement decisions matter most.

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
- Using `torch.compile(mode="reduce-overhead")` provides meaningful inference speed improvements.

### Training Phases

1. **Offline phase**: Behavior cloning from Tenhou and Majsoul game logs. The model learns basic play patterns from human expert data.
2. **Online phase**: Self-play reinforcement learning. Performance typically peaks around ~3 million steps, then begins to degrade due to forgetting.
3. **Bootstrap loop**: Train for a short period, use the resulting model as the `test_play` opponent, then repeat. This iterative approach helps maintain training stability.

### Known Issues

- **Catastrophic forgetting**: Performance peaks during online training then drops sharply. The model "forgets" patterns learned during offline training as self-play shifts the distribution.
- **Q-value collapse**: After degradation sets in, all action Q-values converge to similar numbers, making the policy effectively random. This is the terminal failure mode of extended online training.

Source: GitHub Discussions #64, #27, #70

## 3-Player (Sanma) Adaptations

Adapting Mortal for 3-player mahjong (sanma) requires several structural changes:

1. **Tile removal**: Manzu tiles 2m through 8m are removed, reducing the live tile count to 108 total tiles.
2. **Nukidora**: The North wind tile (N) is extracted as a bonus tile rather than played normally. This maps to action label 40 in the action space.
3. **Chii disabled**: Sequence calls (chi) are not permitted in sanma — only pon and kan calls are legal.
4. **Array sizes**: All player-indexed arrays shrink from 4 elements to 3 elements (3 seats instead of 4).
5. **Library**: Sanma requires the `libriichi3p` fork rather than the standard `libriichi`.

Source: `hidacow/mjai-reviewer3p`

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

### Burn Framework

Burn is the most promising native Rust option for ML training. Community DQN implementations already exist and serve as reference points:

| Repository | Author |
|-----------|--------|
| `Reinforcement_Learning_DQN_in_Rust_using_Burn` | rootware |
| `burn-rl-examples` | yunjhongwu |
| `sb3-burn` | will-maclean |

Burn supports multiple backends: **WGPU** (cross-platform GPU compute), **LibTorch** (PyTorch interop), and **CUDA** (direct NVIDIA GPU). There is no official `burn-rl` crate yet — reinforcement learning pipelines must be implemented manually using the framework primitives.

### tch-rs

tch-rs provides Rust bindings for LibTorch (the C++ backend of PyTorch). It is more mature and feature-complete than Burn for ML workloads but carries a heavy C++ dependency chain, making builds and deployment more complex.

### Distributed Training Patterns

- **IMPALA pattern**: Decoupled actors generate experience while a central learner consumes batches and updates weights. This avoids Mortal's subprocess-with-watchdog hack.
- **crossbeam-channel**: High-throughput, lock-free channels for same-machine actor-to-learner communication.
- **tokio**: Async runtime for coordinating distributed actors across machines.
- **rayon**: Data-parallel library for running game simulations across CPU cores.
- **Zero-copy transfer**: The `shared_memory` crate and Apache Arrow enable zero-copy data sharing between actor and learner processes.
