# Mortal Analysis

Consolidated ref for Mortal Mahjong AI — arch, limits, ecosystem, community findings.

## Mortal Architecture Summary

### Neural Network

Mortal backbone = ResNet + Channel Attention (Squeeze-Excitation style). v4 obs shape = **(1012, 34)**: 1012 channels across 34 tile types (src: `libriichi/src/consts.rs:25` at commit `0cff2b5` — `obs_shape(4)` returns `[1012, 34]`; 1012 channels built in `libriichi/src/state/obs_repr.rs`). Action space = **46 discrete actions**: indices 0–36 map to discard or kan-tile-select for 37 tiles (34 base + 3 aka), 37 = riichi, 38–40 = 3 chi variants (low/mid/high), 41 = pon, 42 = kan (all types: daiminkan, ankan, kakan — 2-phase system where 2nd forward pass with `at_kan_select=true` reuses indices 0–36 to pick *which* tile to kan), 43 = agari (win), 44 = ryuukyoku (draw), 45 = pass.

> **Note:** Hydra uses Mortal exact 46-action map (see [../../docs/GAME_ENGINE.md § 46-Action Space](../../docs/GAME_ENGINE.md#46-action-space)). Both use 46 total actions, same indices: 0–36 discards (incl aka), 37 riichi, 38–40 chi, 41 pon, 42 kan, 43 agari, 44 ryuukyoku, 45 pass.

All versions use **Dueling DQN** decomposition `Q = V + A − mean(A)`, but head arch differs by version:

| Version | V/A Head impl | Feature dim |
|---------|------------------------|-------------|
| v1 | Separate `nn.Linear(512, 1)` / `nn.Linear(512, ACTION_SPACE)` from VAE latent (μ, log σ) | 512 |
| v2 | Separate 2-layer MLPs (1024 → 512 → 1 and 1024 → 512 → 46), Mish activation | 1024 |
| v3 | Separate 2-layer MLPs (1024 → 256 → 1 and 1024 → 256 → 46), Mish activation | 1024 |
| v4 | Single `nn.Linear(1024, 1 + ACTION_SPACE)`, split into V(1) and A(46) post-hoc | 1024 |

In v4, V/A share params in single linear layer — decomposition formula still applies, but no separate streams. **GRP** (GRU-based Rank Predictor) head predicts final placement probs.

Source: `mortal/model.py` (DQN class, Brain class), `libriichi/src/consts.rs`

### Training Algorithm

Training uses **DQN + Conservative Q-Learning (CQL)** for offline train. CQL loss = diff between logsumexp of Q-outputs (across action dim) and mean Q-value: `logsumexp(q_out, dim=-1).mean() − q.mean()`. CQL off during online train mode.

**Reward shaping** uses GRP head, which predicts rank probs. Reward signal = **delta expected value**: `E[pts]_t − E[pts]_{t−1}`, where pts vector = **[3, 1, −1, −3]** for 1st through 4th.

Source: `mortal/train.py:236-238` (at commit `0cff2b5`), `mortal/reward_calculator.py:10` (pts), `:36-37` (delta) (at commit `0cff2b5`)

### Training Pipeline Details

Key arch decisions from source analysis:

- **Q-targets use Monte Carlo returns, not TD bootstrapping.** `q_target = gamma^steps_to_done * kyoku_reward` — no bootstrap from next-state Q-values. Explains why GRP predicts game-level reward.
- **GRP pretrained separately** (`train_grp.py`) with cross-entropy on 24-class placement perms, then frozen during DQN train. Not jointly trained.
- **No target network.** Vanilla DQN, no double-DQN or EMA target. Known source of train instability — Hydra PPO avoids this.
- **v4 DQN head uses single shared linear layer** (`nn.Linear(1024, 1 + ACTION_SPACE)`) — V/A split post-hoc from same output, unlike v1–v3 which had separate V/A head nets. Dueling decomposition `Q = V + A − mean(A)` still applied.
- **SP calculator assumes tsumo-only agari** (`is_ron: false` hardcoded in `calc.rs`). All EV ignores ron, undervaluing hands with good ron waits.
- **Training data uses suit augmentation** — tile suits (manzu/pinzu/souzu) permuted during train for 6× data multiplier. Hydra should do same.

### Shanten Tables

Mortal uses **tomohxx** table-based shanten lookup algo. Two precomputed tables give instant shanten calc:

| Table | Entries | Size (raw) | Compressed |
|-------|---------|-----------|------------|
| Suhai (suited) | 1,940,777 × 10 bytes | ~19.4 MB | 191 KB (.bin.gz) |
| Jihai (honor) | 78,032 × 10 bytes | ~0.78 MB | 5.6 KB (.bin.gz) |

Indexing uses **base-5 encoding**: each tile count (0–4) folded into single int via `acc * 5 + tile_count` across all tiles in suit or honor group. Produces unique index into corresponding table.

Source: `libriichi/src/algo/shanten.rs:82-84` (at commit `0cff2b5`)

## libriichi Python API

Mortal exposes Rust mahjong engine to Python via PyO3 bindings as `libriichi` package.

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
- **oracle_obs_shape(version=4)** = (217, 34) — 51ch opponent state (3×17: hand/aka/shanten/waits/furiten) + 166ch wall (138 yama draw order + 8 rinshan + 10 dora + 10 ura). Each tile uses 2ch (one-hot identity + aka flag). Oracle obs concatenated with public obs along channel dim before stem Conv1d, making oracle model input (1229, 34). Source: `invisible.rs:152-245` (at commit `0cff2b5`), `model.py:109-155` (at commit `0cff2b5`). Note: Mortal published train pipeline never enables oracle mode — infra exists but `is_oracle=True` never set in `train.py`.

### Usage Pattern

Typical flow: create `PlayerState` for seat (0–3), feed line-delimited JSON events via update method, then render obs at given version. `GameplayLoader` handles batch load of recorded games for train, while `OneVsThree` runs eval matches where one agent plays against three copies of baseline.

Source: `libriichi/src/lib.rs`, `libriichi/src/consts.rs`

## MJAI Protocol

> See [../infrastructure/INFRASTRUCTURE.md#mjai-protocol](../infrastructure/INFRASTRUCTURE.md#mjai-protocol) for full protocol spec (message types, tile encoding, Mortal meta extensions). Hydra canonical MJAI def lives there.

Source: `libriichi/src/mjai/event.rs`

## Confirmed Limitations

### No Opponent Modeling

Mortal uses `SinglePlayerTables` for EV calc, assuming no opponent interaction. v4 obs encoder (lines 564–624 of `obs_repr.rs`, at commit `0cff2b5`) gives no precomputed safety features such as suji, kabe, or genbutsu analysis. No opponent tenpai estimation, no aggression/tendency profiling, no tracking of opponent discard patterns for intent reading.

Source: `libriichi/src/state/obs_repr.rs` (v4 uses SinglePlayerTables at lines 564–624, at commit `0cff2b5`)

### Score Encoding Issues

v4 obs encoding uses **dual-scale score channels**: one normalized by 100,000 (keeps coarse info for high scores) and one normalized by 30,000 (higher resolution for common strategic range). Scores above 30K degrade in fine-grained channel but still exist in coarse channel. No explicit overtake-threshold encoding — net has no direct rep of points needed to change placement. This leads to bad hand-building near placement thresholds (Source: Issue #111). Dual-scale encoding may worsen reliability in high-score late-game spots where precise placement awareness matters most.

Source: GitHub Discussion #108 (about max player score in observations), Issue #111

### Training Infrastructure Bugs

Online train hangs for unknown reasons — explicit bug comment in train code says so. Workaround = subprocess spawning with watchdog that restarts train process when stalled. Also Windows compat issues with GRP init.

Source: `mortal/train.py:382-386` (at commit `0cff2b5`)

### Oracle Guiding Removal

Oracle guiding (train with perfect info, then distill to imperfect) existed in Mortal v1/v2 but was **removed in v3**. Per Equim-chan: "It didn't bring improvements in practice" — removal not driven by throughput concerns. Oracle guiding replaced with **next-rank prediction** aux task (implemented as `AuxNet` in `mortal/model.py`, called "NextRankPredictor" in GitHub Discussion #52 where Equim-chan explains rationale; oracle guiding removal discussed in #102).

Source: GitHub Discussion #102 (Equim-chan)

## Community Observations

> **Ownership note:** This file = authoritative list of Mortal strengths/limits with evidence context. `COMMUNITY_INSIGHTS.md` stores broader community signals; current Hydra arch-level deltas now live across `README.md`, `HYDRA_FINAL.md`, `HYDRA_RECONCILIATION.md`, and focused design docs rather than legacy spec.

### Playstyle Statistics

Compared to akochan (another mahjong AI), Mortal plays noticeably more conservative, efficient style:

| Metric | Mortal | Akochan | Delta |
|--------|--------|---------|-------|
| Riichi rate | 18.9% | 21.5% | −2.6pp |
| Call rate | 29.3% | 33.0% | −3.7pp |
| Deal-in rate | 11.3% | 13.0% | −1.7pp |

Despite lower action rates across board, Mortal gets **higher win rate** than akochan — wins more while declaring riichi less, calling less, dealing in less.

Source: `docs/src/perf/strength.md`

Public Mortal performance tables also publish **Deal-in rate after riichi** and **Deal-in rate after call**. No dedicated **damaten-specific deal-in** breakdown published in same source.

### Known Weaknesses

**1. Orras Over-Defensive** — Mortal plays too safely in final round (orras/south 4) when in 2nd or 3rd, missing overtake chances on 1st. Compounded by overtake score miscalc in Issue #111. Model does not distinguish enough between "safe 2nd" and "aggressive push for 1st" spots. (Source: GitHub Issue #111, Reddit r/Mahjong)

**2. Early Riichi Push Errors** — Mortal underestimates threat of early riichi (turns 1–6). It pushes with suboptimal hands against unknown waits, failing to recognize early riichi correlates with stronger hands and more dangerous waits. (Source: Reddit r/Mahjong)

**3. Damaten Detection Failures** — Mortal has no intent reading for silent tenpai (damaten). It relies only on explicit signals like riichi and visible melds. Result: deals into high-value silent hands that experienced humans would read from discard patterns and timing tells. (Source: Reddit r/Mahjong)

**4. Efficiency Over Yaku** — Mortal prioritizes shanten reduction speed (tile efficiency) over hand value construction. It discards dora or yaku-building tiles for raw efficiency gains. Especially bad in comeback spots where high-value hand needed — model builds fast cheap hands when it should chase expensive ones. (Source: Reddit r/Mahjong)

**5. Coarse Placement Sensitivity** — Mortal keeps almost same playstyle regardless of point spread. It does not tune aggression to specific overtake thresholds (e.g., exact points between 2nd and 1st). Dual-scale score encoding (100K/30K channels) degrades fine-grained score info above 30K (see Score Encoding Issues above), further limiting placement-aware play. (Source: GitHub Issue #111, Reddit r/Mahjong)

### General Japanese Community Feedback

Japanese mahjong community notes Mortal struggles with **場況 (bakyou)** — reading overall field status and table flow. It prioritizes raw efficiency over situational tactics. Rating system also criticized for penalizing alternative playstyles that may still be strategically valid. (Source: Note.com mahjong blogs, Reddit r/Mahjong)

## Rating System

### Formula

Mortal review rating uses **Boltzmann softmax** over Q-values to compute action probs:

P(action_i) = exp(Q(action_i) / τ) / Σ_j exp(Q(action_j) / τ)

Overall rating then = **Rating = 100 × mean(P(human_action))** across all decision points in game. Higher rating = human moves align more with Mortal top-rated actions.

### Criticisms

- **Near-equal penalty**: When multiple actions have similar Q-values (near-equivalent), softmax spreads probability mass among them, harshly penalizing human for choosing any single one — even if all options are equal in EV.
- **Hindsight bias**: Moves labeled wrong based on outcome-influenced eval rather than decision quality under info available at time.
- **EV vs placement**: Rating optimizes EV, not placement security, potentially marking defensive plays suboptimal when they secure safe finish.
- **Score capping**: v4 model dual-scale encoding degrades score resolution above 30K, making ratings less reliable in high-score late-game spots where placement decisions matter.

Source: `mjai-reviewer` codebase, community discussions

## Policy vs Value Architecture

Comparison between original Mortal (DQN) and Mortal-Policy fork (PPO), maintained by Nitasurin:

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

- **batch_size**: Higher better, limited only by available VRAM. Larger batches give more stable gradient estimates.
- **sample_reuse**: Disable — use fresh samples only for each train step.
- **boltzmann_epsilon**: Should anneal over time; start high for exploration, decrease for stability as train progresses.
- **learning_rate**: low rates (e.g., 1e-10) prevent catastrophic forgetting but effectively stall learning; right balance critical.

### Architecture Tips

- Adding **dropout** to ResBlocks improves stability during online train phase.
- **Full network fine-tuning** required — freezing layers does not work well here.
- Using `torch.compile()` gives meaningful inference speed gains (Discussion #43). Mortal uses default compilation mode.

### Training Phases

1. **Offline phase**: Behavior cloning from Tenhou and Majsoul game logs. Model learns basic play patterns from human expert data.
2. **Online phase**: Self-play RL. Performance usually peaks around ~3 million steps, then degrades due to forgetting.
3. **Bootstrap loop**: Train short period, use resulting model as `test_play` opponent, then repeat. This iterative approach helps keep train stable.

### Known Issues

- **Catastrophic forgetting**: Performance peaks during online train then drops sharply. Model "forgets" patterns learned during offline train as self-play shifts distribution.
- **Q-value collapse**: After degradation starts, all action Q-values converge to similar numbers, making policy effectively random. This is terminal failure mode of long online train.

Source: GitHub Discussions #64, #27, #70

## Rust ML Ecosystem

> See [ECOSYSTEM.md](ECOSYSTEM.md) and [REFERENCES.md](REFERENCES.md) for Rust ML tooling.