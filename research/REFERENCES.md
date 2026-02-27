# Hydra References

Single source of truth for all citations in the Hydra project.

---

## Academic Papers

### Mahjong AI

| Paper | Authors | Year | Venue / URL | Key Contribution | Relevance to Hydra |
|-------|---------|------|-------------|------------------|---------------------|
| Suphx: Mastering Mahjong with Deep Reinforcement Learning | Junjie Li, Sotetsu Koyamada, Qiwei Ye, Guoqing Liu, Chao Wang, Ruihan Yang, Li Zhao, Tao Qin, Tie-Yan Liu, Hsiao-Wuen Hon | 2020 | [arXiv:2003.13590](https://arxiv.org/abs/2003.13590) | Oracle guiding, Global Reward Prediction (GRP), run-time policy adaptation, 10-dan achievement on Tenhou. Architecture: 50 residual blocks, 256 filters, separate models per action type with 838 input channels (discard/riichi) and 958 input channels (chow/pong/kong) (Table 2, Figures 4-5). | Core inspiration for oracle distillation and GRP head design |
| Tjong: A Transformer-based Mahjong AI via Hierarchical Decision-Making and Fan Backward | Xiali Li, Bo Liu, Zhi Wei, Zhaoqi Wang, Licheng Wu | 2024 | [CAAI Trans. Intel. Tech.](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cit2.12298) DOI: 10.1049/cit2.12298 | Hierarchical decision-making (action type → tile selection), transformer architecture for game sequences, fan backward reward shaping | Alternative architecture reference; fan backward considered for yaku awareness |
| Information Set Monte Carlo Tree Search | P. I. Cowling, E. J. Powley, D. Whitehouse | 2012 | [IEEE TCIAIG](https://ieeexplore.ieee.org/document/6203567) | Foundation for handling imperfect information via determinization and information-set sampling | Theoretical basis for imperfect-info game approaches |
| Real-time Mahjong AI based on Monte Carlo Tree Search (Bakuuchi) | Mizukami et al. | 2014 | IEEE | Pre-deep-learning SOTA using ISMCTS + rule-based heuristics | Historical baseline for MCTS approaches |
| An Open-Source Interpretable and Reproducible Mahjong Agent (Phoenix) | — | 2021 | [USC CSCI 527 Course Project](https://csci527-phoenix.github.io/documents/Paper.pdf) | Transparent baseline with interpretable decision-making | Open-source baseline reference |
| Building a Computer Mahjong Player via Deep Convolutional Neural Networks | — | 2018 | IEEE | CNN for Mahjong, baseline methods | Early CNN approach for mahjong |
| Speedup Training Artificial Intelligence for Mahjong via Reward Variance Reduction | Li, Wu, Fu, Fu, Zhao, Xing | 2022 | [IEEE CoG](https://ieee-cog.org/2022/assets/papers/paper_103.pdf) | RVR technique for reducing gradient noise from luck variance, oracle critic + expected reward network | Enables training on limited hardware; hand-luck baseline subtraction |
| Actor-Critic Policy Optimization in a Large-Scale Imperfect-Information Game | Fu, Liu, Wu, Wang, Yang, Li, Xing, Li, Ma, Fu, Yang | 2022 | [ICLR 2022](https://openreview.net/forum?id=DTXZqTNV5nW) | ACH (Actor-Critic Hedge): merges deep RL with Weighted CFR for Nash Equilibrium convergence in imperfect-info games. Core offline training algorithm for Tencent's LuckyJ. | Game-theoretic RL alternative to PPO/DQN; LuckyJ's ACH + OLSS reached 10.68 stable dan on Tenhou |
| Opponent-Limited Online Search for Imperfect Information Games | Liu, Fu, Fu, Yang | 2023 | [ICML 2023](https://proceedings.mlr.press/v202/liu23k.html) | OLSS: imperfect-info subgame solving with opponent-limited tree pruning, orders of magnitude faster than common-knowledge methods. Tested on 2-player mahjong. | Core search component for LuckyJ; search-as-feature integration enables real-time strategy adjustment |

### General Game AI

| Paper | Authors | Year | Venue / URL | Key Contribution | Relevance to Hydra |
|-------|---------|------|-------------|------------------|---------------------|
| Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (AlphaZero) | Silver et al. | 2017 | [arXiv](https://arxiv.org/abs/1712.01815) | MCTS + neural network self-play, general game learning | Baseline game AI paradigm |
| Superhuman AI for Multiplayer Poker (Pluribus) | Brown, Sandholm | 2019 | Science | Imperfect-information game solving at scale | Opponent modeling in imperfect-info games |
| OpenAI Five | OpenAI | 2019 | [OpenAI](https://openai.com/five/) | Large-scale PPO for complex games | Training stability and PPO scaling |
| AlphaStar: Mastering the Real-Time Strategy Game StarCraft II | Vinyals et al. | 2019 | Nature | League training for multi-agent robustness | League training methodology for Phase 3 |
| Mastering the Game of Stratego with Model-Free Multiagent Reinforcement Learning (DeepNash) | Perolat et al. | 2022 | Science | R-NaD for Nash equilibrium approximation | Considered and rejected; Nash approach less suitable for 4-player ranking |

### Architecture Components

| Paper | Authors | Year | Venue / URL | Key Contribution | Relevance to Hydra |
|-------|---------|------|-------------|------------------|---------------------|
| Squeeze-and-Excitation Networks | Hu et al. | 2018 | CVPR | SE attention blocks for channel recalibration | Backbone design: dual-pool SE attention in every ResBlock |
| CBAM: Convolutional Block Attention Module | Woo et al. | 2018 | ECCV | Channel + spatial attention via dual-pool (avg+max) shared MLP | Hydra's SE module uses CBAM's channel attention component (dual-pool shared MLP) |
| Group Normalization | Wu & He | 2018 | ECCV | Batch-independent normalization | Training stability: GroupNorm(32) replaces BatchNorm |
| Proximal Policy Optimization Algorithms | Schulman et al. | 2017 | [arXiv](https://arxiv.org/abs/1707.06347) | PPO clipped surrogate objective | Core RL algorithm for Phases 2-3 |
| Attention Is All You Need | Vaswani et al. | 2017 | NeurIPS | Transformer architecture | Considered for backbone; used by Kanachan and Tjong |
| Learning Confidence for Out-of-Distribution Detection | DeVries, Taylor | 2018 | [arXiv:1802.04865](https://arxiv.org/abs/1802.04865) | Confidence estimation as training regularization | Used by NAGA for calibrated action distributions |

---

## Open Source Projects

### Mahjong AI

| Project | URL | Language | Stars | License | Notes |
|---------|-----|----------|-------|---------|-------|
| Mortal | https://github.com/Equim-chan/Mortal | Rust/Python | 1.3K+ | AGPL-3.0-or-later | Primary competitor. ResNet(40 blocks, 192ch) + Channel Attention → DQN(Dueling) + CQL. Reference only — AGPL, cannot derive code. Study: obs encoding (1012×34), action masking (46 actions), GRP head, 1v3 duplicate evaluation. Weights have additional distribution restrictions beyond AGPL. |
| Kanachan | https://github.com/Cryolite/kanachan | C++/Python | 300+ | Unlicensed | **Transformer encoder (BERT-style)** — two configs: base (~90M params, 12L/768d) and large (~310M params, 24L/1024d). Trained on 65M+ Majsoul rounds (Gold+), zero hand-crafted features. 184 tokens: 33 sparse + 6 numeric + 113 progression + 32 candidates. Pipeline: BC → curriculum fine-tuning → offline RL (IQL/ILQL/CQL). **No published benchmarks** despite multi-year development (public repo created 2021-08-05). Parameter count makes online RL infeasible. ⚠️ No LICENSE file in repo — do not depend on code. |
| Akochan | https://github.com/critter-mj/akochan | C++ | ~280 | Custom (restrictive, Japanese) | EV-based heuristic engine with explicit suji/kabe/genbutsu analysis. Not ML-based. Matters: its hand-crafted defense logic is a useful sanity check — if Hydra's neural network disagrees with Akochan's defense in obvious spots, something is wrong. Also used as the backend for the original mjai-reviewer. |
| MahjongAI | https://github.com/erreurt/MahjongAI | Python | ~450 | — | Extensible agent framework with pluggable strategies. Matters less for architecture, more for its Tenhou client implementation — shows how to connect an AI to Tenhou's protocol if we ever need that. |
| AlphaJong | https://github.com/Jimboom7/AlphaJong | JavaScript | — | — | Browser-based heuristic engine (NOT AlphaZero despite the name). Tunable offense/defense balance via sliders. Matters only as a weak baseline — useful for sanity-checking that Hydra beats simple heuristics by a wide margin. |
| mjai-manue | https://github.com/gimite/mjai-manue | Ruby | 37 | — | Original MJAI protocol client. Matters as protocol reference — defines the canonical MJAI message format that Hydra must be compatible with. |
| NAGA | https://dmv.nico/en/articles/mahjong_ai_naga/ | — | — | Commercial | **Pure supervised learning** — 4 independent CNNs (discard, call, riichi, kan) trained on Tenhou Houou game logs via imitation learning. No self-play, no RL. Uses confidence estimation (DeVries & Taylor 2018) as training regularization and Guided Backpropagation (Springenberg et al. 2014) for interpretability. 5 playstyle variants (Omega, Gamma, Nishiki, Hibakari, Kagashi) differentiated by training on different players' game records, not architecture changes. CNN architecture details (layers, filters, input shape) never publicly disclosed — the [DMV article](https://dmv.nico/en/articles/mahjong_ai_naga/) is the sole official technical document. Achieved 10-dan on Tenhou (26,598 games — source unverified; number does not appear in the DMV article or any locatable public source), current models estimated ~9-dan stable. Not open-source. NAGA's "match%" metric is a common (but imperfect) benchmark. |
| LuckyJ | https://haobofu.github.io/ | — | — | Commercial | Tencent AI Lab's mahjong AI (绝艺/JueYi brand). 10-dan on Tenhou in 1,321 games, 10.68 stable dan — strongest known AI. ACH + OLSS architecture, pure self-play. See [COMMUNITY_INSIGHTS § LuckyJ](COMMUNITY_INSIGHTS.md#4-luckyj-tencent-ai-lab) for detailed architecture analysis. |

### Analysis & Review Tools

| Project | URL | Stars | Description |
|---------|-----|-------|-------------|
| mjai-reviewer | https://github.com/Equim-chan/mjai-reviewer | 1.1K+ | CLI that generates HTML review reports showing Q-value differences per discard. Primary tool for evaluating Hydra's play quality. Apache-2.0 — can use directly. |
| mjai-reviewer3p | https://github.com/hidacow/mjai-reviewer3p | — | 3-player (sanma) fork of mjai-reviewer. Matters only if Hydra targets sanma. |
| killer_mortal_gui | https://github.com/killerducky/killer_mortal_gui | — | Enhanced Mortal review with deal-in heuristic multipliers (ryanmen 3.5×, kanchan suji-trap 2.6×, honor tanki/shanpon 1.7×, etc). Matters: these empirically-tuned danger multipliers are the best public reference for tile danger calibration — useful for validating Hydra's learned defense signals. |
| crx-mortal | https://github.com/announce/crx-mortal | — | Chrome extension for in-browser Mortal analysis. Low relevance for training. |
| mjai-batch-review | https://github.com/Xerxes-2/mjai-batch-review | 9 | Batch analyze multiple game logs at once. Matters for large-scale evaluation — when testing Hydra across thousands of games, batch review is faster than one-by-one. |

### Mortal Forks

| Fork | URL | Key Difference |
|------|-----|----------------|
| Mortal-Policy | https://github.com/Nitasurin/Mortal-Policy | PPO instead of DQN, GroupNorm instead of BatchNorm, entropy weight tuning. AGPL-3.0, reference only. Matters: closest public reference to Hydra's own architecture choice (PPO + GroupNorm). Study their AWR→PPO transition code path and how they handle the policy gradient with mahjong's 46-action space. |

### Components

| Project | URL | Language | License | Purpose |
|---------|-----|----------|---------|---------|
| xiangting | https://github.com/Apricot-S/xiangting | Rust | MIT | Primary shanten library. Compile-time embedded tables (~200KB), `no_std` compatible, 3-player support, returns both shanten number and necessary/unnecessary tile sets. 34× faster than brute-force for replacement tile calculation. Hydra uses this for obs encoding channels (shanten features) and action masking. |
| xiangting-py | — | Python | MIT | Python bindings for xiangting via PyO3. Useful for training-side shanten calculation if needed. |
| tomohxx/shanten-number | — | C++ | LGPL-3.0 | Original table-based shanten algorithm that xiangting is derived from. Algorithm reference only — LGPL prevents static linking. Tables: suhai (1.9M entries, ~19.4MB), jihai (78K entries, ~0.78MB). Base-5 encoding for tile state indexing. |
| PyO3 | https://pyo3.rs/ | Rust | Apache-2.0 | Rust↔Python FFI for exposing game engine bindings to the training loop. |
| rayon | https://docs.rs/rayon/ | Rust | Apache-2.0 | Work-stealing data parallelism for batch game simulation. |
| serde / serde_json | https://serde.rs/ | Rust | Apache-2.0 | JSON serialization/deserialization for MJAI protocol parsing. |
| ndarray | https://docs.rs/ndarray/ | Rust | Apache-2.0 | N-dimensional array operations for constructing observation tensors. |
| ort | https://docs.rs/ort/ | Rust | Apache-2.0 | ONNX Runtime Rust bindings. Primary inference engine for self-play: loads exported PyTorch model as ONNX, runs forward passes with CUDA EP, CUDA graphs, and I/O binding for <5ms latency. This is the hot path during self-play — inference speed directly limits training throughput. |
| tract | https://docs.rs/tract/ | Rust | MIT OR Apache-2.0 | Pure Rust ML inference engine (no C++ deps). CPU-only fallback for environments without CUDA. Useful for CI testing and CPU-only deployment. |
| candle | https://github.com/huggingface/candle | Rust | Apache-2.0 | HuggingFace's Rust ML framework with CUDA and Metal support. Alternative to ONNX path — write inference directly in Rust, avoiding the PyTorch→ONNX export step. Worth evaluating if ONNX export causes accuracy loss or operator compatibility issues. |
| Burn | https://github.com/tracel-ai/burn | Rust | MIT OR Apache-2.0 | Native Rust training + inference framework with WGPU, CUDA, and LibTorch backends. Long-term option for moving the entire training loop to Rust (eliminating Python entirely). Growing ONNX import support. |
| tch-rs | — | Rust | MIT OR Apache-2.0 | Rust bindings for LibTorch. Alternative to PyO3 approach — call LibTorch directly from Rust instead of going through Python. Trades Python flexibility for lower FFI overhead. |
| mahjong (Python) | https://github.com/MahjongRepository/mahjong | Python | MIT | Hand scoring oracle — yaku detection, han/fu/score calculation, validated against 11M+ Tenhou hands. Pin to v1.4.0. Dev dependency for Rust engine verification and test case extraction. |
| agari | https://github.com/rysb-dev/agari | Rust | MIT (no LICENSE file) | Complete scoring engine (35 yaku, fu, payment, hand decomposition, ~100 unit tests). Most architecturally clean Rust mahjong scorer — study its `HandDecomposition` trait and `Fu` calculation for Hydra's own scoring module. `Cargo.toml` declares MIT but repo lacks a LICENSE file — safe to use as reference. |
| mahc | https://github.com/DrCheeseFace/mahc | Rust | BSD-3 | Scoring library with explicit `Fu` enum (each fu source is a named variant, not magic numbers). 38 yaku, 30K crates.io downloads. Study the `Fu` enum pattern — makes fu calculation self-documenting and testable vs Mortal's opaque approach. |
| mahjax | https://github.com/nissymori/mahjax | Python/JAX | Apache-2.0 | JAX-vectorized riichi environment reaching ~1.6M steps/sec on 8×A100 via JIT compilation. Matters for self-play: JAX vectorization can run thousands of games simultaneously on GPU, potentially 10-100x faster than sequential Rust simulator for generating training data. Study their state representation and vectorized game logic. |
| RiichiEnv | https://github.com/smly/RiichiEnv | Rust/Python | Apache-2.0 | Gym-style RL environment with Rust core + Python bindings, Mortal-compatible MJAI output. Verified correct over 1M+ games. Matters because it provides a ready-made OpenAI Gym interface — if Hydra's training loop uses standard Gym APIs (reset/step/reward), this slots in directly. Also useful as correctness oracle for our own Rust game engine. |
| Meowjong | https://github.com/VictorZXY/Meowjong | Python | MIT | Only open-source 3-player (sanma) mahjong AI. IEEE CoG 2022. Includes 5 CNN model variants and a Tenhou sanma log downloader. Matters because sanma is a stretch goal — if Hydra ever targets 3-player, this is the only reference implementation with published results. Also validates that CNN architectures work for reduced-player mahjong. |
| CleanRL | https://github.com/vwxyzjn/cleanrl | Python | MIT | Single-file PPO implementation (~250 lines) with wandb integration. Accompanied by the "37 Implementation Details of PPO" blog post that documents every hyperparameter and trick that matters. Hydra's PPO should be validated against CleanRL's implementation — same clipping, advantage normalization, value loss clipping, entropy coefficient schedule. The blog post is required reading before writing our PPO. |
| OpenSpiel | https://github.com/google-deepmind/open_spiel | C++/Python | Apache-2.0 | DeepMind's game RL framework with 70+ games, including AlphaZero, MCTS, CFR, and self-play training loops. Matters for Hydra's Phase 3 (league training): study their self-play loop architecture — how they manage opponent pools, ELO tracking, and policy selection. Also has imperfect-info game solvers that inform belief-state approaches. |
| Microsoft Olive | https://github.com/microsoft/Olive | Python | MIT | End-to-end model optimization: PyTorch → ONNX with quantization, pruning, operator fusion, shape inference via YAML config. Matters for inference speed during self-play: training generates millions of forward passes, so even 2x speedup from INT8 quantization directly halves self-play wall time. Use after model architecture stabilizes. |
| rlcard | https://github.com/datamllab/rlcard | Python | MIT | RL toolkit with a mahjong environment and pre-built DQN/NFSP agents. Lower fidelity than mahjax/RiichiEnv (simplified rules), but useful for rapid prototyping of reward shaping and training loop mechanics before running on the full environment. |
| mjai.app | https://github.com/smly/mjai.app | — | AGPL-3.0 | RiichiLab competition platform using MJAI protocol with Docker-based evaluation. Matters because this is a target venue — Hydra must produce MJAI-compatible output to enter competitions and benchmark against other AIs. Study their Docker submission format and evaluation harness. |

### Protocol & Infrastructure

| Project | URL | Description |
|---------|-----|-------------|
| mjai | https://github.com/gimite/mjai | Original MJAI protocol server |
| mjai-gateway | https://github.com/tomohxx/mjai-gateway | MJAI ↔ Tenhou translator |

---

## Community Resources

### Documentation

| Resource | URL | Content |
|----------|-----|---------|
| Mortal Documentation | https://mortal.ekyu.moe | Architecture insights, performance data, playstyle statistics |
| MJAI Protocol Wiki | https://gimite.net/pukiwiki/index.php?MJAI | Standard protocol specification (⚠️ may require login) |
| MJAI Web Reviewer | https://mjai.ekyu.moe/ | Web interface for instant game reviews |
| Tenhou Documentation | https://tenhou.net/man/ | Tenhou log format specification (old `/doc/` path returns 404) |
| Majsoul API | Various GitHub repos | Log extraction methods via WebSocket capture |
| NAGA Documentation | https://dmv.nico/en/articles/mahjong_ai_naga/ | Commercial AI architecture overview |
| Riichi Wiki — NAGA | https://riichi.wiki/Mahjong_AI_%E3%80%8CNAGA%E3%80%8D | Community wiki page on NAGA |
| Phoenix Paper | https://csci527-phoenix.github.io/documents/Paper.pdf | Open-source reproducible mahjong agent |
| ONNX Runtime | https://onnxruntime.ai/ | Production inference runtime |

### Discussion Sources

| Source | Topics |
|--------|--------|
| Mortal GitHub Issues & Discussions | Known weaknesses, training problems, oracle guiding removal |
| r/Mahjong (Reddit) | Player perspective on AI behavior, known weaknesses |
| Discord (Riichi Mahjong) | Community testing, strategy discussion |
| Tenhou forums | High-level play analysis |
| Note.com mahjong blogs (Japanese) | 場況 (bakyou) struggles, efficiency vs situational tactics |

---

## Training Data Sources

> See [ECOSYSTEM.md § Data Sources & Datasets](ECOSYSTEM.md#3-data-sources--datasets) for the training data summary and [archive/DATA_SOURCES.md](archive/DATA_SOURCES.md) for full details on sources, converters, and alternative datasets.

---

## Algorithm References

### Shanten Calculation

| Resource | Description |
|----------|-------------|
| tomohxx Algorithm | Set-based recurrence, O(n) complexity; table-based lookup |
| tomohxx Tables | Suhai table: 1,940,777 entries × 10 bytes (~19.4 MB); Jihai table: 78,032 entries × 10 bytes (~0.78 MB) |
| tomohxx Indexing | Base-5 encoding: `tiles.iter().fold(0, |acc, &x| acc * 5 + x as usize)` |
| tomohxx Compressed | shanten_suhai.bin.gz (191 KB), shanten_jihai.bin.gz (5.6 KB) |
| xiangting Implementation | Rust port with 3-player support |
| Kanachan xiangting | LOUDS-based TRIE shanten calculator |
| Mahjong Algorithm Book | Japanese reference, theoretical background |
| Cryolite (2023) | "A Fast and Space-Efficient Algorithm for Calculating Deficient Numbers" |

### Suji / Kabe / Genbutsu

| Resource | Description |
|----------|-------------|
| Japanese Mahjong Strategy Books | Traditional defense theory |
| Daina Chiba's Defense | Quantitative suji analysis |
| Tenhou Player Guides | Statistical safety percentages |
| Suji Safety Note | Suji is approximately 60-70% safe (not 100%); protects only against ryanmen waits |
| Genbutsu Definition | 100% safe — tiles discarded by or after opponent's riichi |
| Kabe Definition | All 4 copies visible → no-chance wait; 3 copies = one-chance |
| Half-suji / Full-suji | One side visible vs both sides visible |
| killer_mortal_gui Heuristics | Ryanmen 3.5×, Kanchan 0.21×, Kanchan suji-trap 2.6×, Penchan 1.0×, Honor tanki/shanpon 1.7×; modifiers: Dora 1.2×, Ura-suji 1.3×, Matagi early 0.6×, Matagi riichi 1.2×, Red 5 discard 0.14× |

### Scoring

| Resource | Description |
|----------|-------------|
| Tenhou Scoring Tables | Standard yaku/fu calculation |
| World Riichi Championship Rules | International standard |
| EMA Rules | European standard |

---

## Benchmark References

### Tenhou Ranking

| Rank | Dan | Approx. Strength |
|------|-----|-------------------|
| R2000+ | 7-dan+ | Expert |
| R1800-2000 | 5-6 dan | Strong |
| R1600-1800 | 3-4 dan | Intermediate |

### AI Achievements

| AI | Platform | Achievement | Year | Notes |
|----|----------|-------------|------|-------|
| NAGA | Tenhou | 10-dan (26,598 games — unverified) | 2018+ | Pure imitation learning; current models ~9-dan stable |
| Suphx | Tenhou | 10-dan (5,373 games), 8.74 stable | 2020 | SL + RL + oracle guiding; paper states 100+ humans have achieved 10-dan |
| LuckyJ | Tenhou | **10-dan (1,321 games), 10.68 stable** | 2023 | ACH + OLSS; statistically stronger than both NAGA and Suphx |
| Mortal | — | **No ranked play** | — | Tenhou rejected Mortal's AI account request ([FAQ](https://github.com/Equim-chan/mjai-reviewer/blob/master/faq.md): "Tenhou rejected my AI account request for Mortal because Mortal was developed by an individual rather than a company"). Community-estimated ~7-dan play strength from mjai-reviewer analysis. |
| NAGA | Majsoul | Celestial | 2022 | — |

---

## License Compatibility

> License policy: See [INFRASTRUCTURE.md § License Compatibility](INFRASTRUCTURE.md#license-compatibility)

---

## GitHub Discussions

Mortal repository discussions relevant to Hydra design decisions:

| Discussion # | Topic | Key Insight |
|-------------|-------|-------------|
| (source code) | MC returns vs TD | Mortal uses MC returns (not TD) for Q-targets — confirmed from source code (`train.py` Q-target computation). `q_target = gamma^steps_to_done * kyoku_reward` with no bootstrap from next-state Q-values. Hydra follows the same approach. |
| #27 | Batch size recommendations | Practical guidance on training batch sizes for mahjong RL. |
| #43 | torch.compile speedup | torch.compile gives 15-20% training speedup on Mortal. Hydra should enable this from day one. |
| #52 | NextRankPredictor rationale | Auxiliary task that predicts next placement — stabilizes feature learning by giving the backbone a secondary objective beyond Q-values. |
| #64 | Catastrophic forgetting in online RL | When transitioning from offline (behavioral cloning) to online (self-play), the model forgets offline knowledge. Equim-chan confirms this is a real problem. Hydra must plan for gradual transition with replay buffer mixing. |
| #70 | DeepCFR for GRP replacement | Community explored using DeepCFR instead of GRP. Conclusion: not practical for 4-player mahjong due to game tree size. |
| #91 | Mortal-Policy (PPO fork) | Nitasurin's PPO fork open-sourced. Confirms PPO works for mahjong, validates Hydra's algorithm choice. |
| #102 | Oracle guiding removed | Equim-chan: "didn't bring improvements in practice." Critical for Hydra — Suphx's oracle guiding (our Phase 1 inspiration) was tried and abandoned by Mortal's author. Hydra's oracle approach must differ from Suphx's naive implementation. |
| #108 | Maximum player score in observations | Discussion about score capping at 30K in observation encoding. Relevant to Hydra's uncapped score encoding decision. |

---

## GitHub Issues

Mortal repository issues relevant to Hydra improvements:

| Issue # | Description |
|---------|-------------|
| #111 | Overtake score miscalculation — Mortal miscalculates hand-building near placement thresholds; motivates Hydra's uncapped score encoding |
| #113 | Rating system closure discussion — community debate on whether to shut down Mortal's rating feature |

---

## Citation Format

For academic reference to Hydra:

```
Hydra: A Practical Mahjong AI Architecture
Combining Oracle Distillation with Explicit Opponent Modeling
2026
```

Key techniques to cite:
- Oracle Distillation: Li et al. (2020) "Suphx"
- SE-ResNet Backbone: Hu et al. (2018) "Squeeze-and-Excitation Networks"
- PPO Training: Schulman et al. (2017) "Proximal Policy Optimization"
- GroupNorm: Wu & He (2018) "Group Normalization"
- League Training: Vinyals et al. (2019) "AlphaStar"
