# Hydra References

Single source truth for Hydra citations.

---

## Academic Papers

### Mahjong AI

| Paper | Authors | Year | Venue / URL | Key Contribution | Relevance to Hydra |
|-------|---------|------|-------------|------------------|---------------------|
| Suphx: Mastering Mahjong with Deep Reinforcement Learning | Junjie Li, Sotetsu Koyamada, Qiwei Ye, Guoqing Liu, Chao Wang, Ruihan Yang, Li Zhao, Tao Qin, Tie-Yan Liu, Hsiao-Wuen Hon | 2020 | [arXiv:2003.13590](https://arxiv.org/abs/2003.13590) | Oracle guiding, Global Reward Prediction (GRP), run-time policy adaptation, 10-dan on Tenhou. Arch: 50 res blocks, 256 filters, separate models per action type, 838 input ch (discard/riichi), 958 input ch (chow/pong/kong) (Table 2, Figures 4-5). | Core inspiration: oracle distill, GRP head |
| Tjong: Transformer-based Mahjong AI via Hierarchical Decision-Making and Fan Backward | Xiali Li, Bo Liu, Zhi Wei, Zhaoqi Wang, Licheng Wu | 2024 | [CAAI Trans. Intel. Tech.](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cit2.12298) DOI: 10.1049/cit2.12298 | Hierarchical decision-making (action type -> tile choice), transformer for game seq, fan backward reward shaping | Alt arch ref; fan backward maybe for yaku awareness |
| Information Set Monte Carlo Tree Search | P. I. Cowling, E. J. Powley, D. Whitehouse | 2012 | [IEEE TCIAIG](https://ieeexplore.ieee.org/document/6203567) | Base for imperfect-info via determinization, information-set sampling | Theory base for imperfect-info game methods |
| Real-time Mahjong AI based on Monte Carlo Tree Search (Bakuuchi) | Mizukami et al. | 2014 | IEEE | Pre-deep-learning SOTA: ISMCTS + rule heuristics | Historical MCTS baseline |
| Open-Source Interpretable and Reproducible Mahjong Agent (Phoenix) | — | 2021 | [USC CSCI 527 Course Project](https://csci527-phoenix.github.io/documents/Paper.pdf) | Transparent baseline, interpretable decisions | Open-source baseline ref |
| Building Computer Mahjong Player via Deep Convolutional Neural Networks | — | 2018 | IEEE | CNN for Mahjong, baseline methods | Early CNN mahjong approach |
| Speedup Training Artificial Intelligence for Mahjong via Reward Variance Reduction | Li, Wu, Fu, Fu, Zhao, Xing | 2022 | [IEEE CoG](https://ieee-cog.org/2022/assets/papers/paper_103.pdf) | RVR cuts gradient noise from luck variance, oracle critic + expected reward net | Helps training on limited HW; hand-luck baseline subtraction |
| Actor-Critic Policy Optimization in Large-Scale Imperfect-Information Game | Fu, Liu, Wu, Wang, Yang, Li, Xing, Li, Ma, Fu, Yang | 2022 | [ICLR 2022](https://openreview.net/forum?id=DTXZqTNV5nW) | ACH: deep RL + Weighted CFR for NE convergence in imperfect-info games. LuckyJ offline training core. | Game-theoretic RL alt to PPO/DQN; LuckyJ ACH + OLSS hit 10.68 stable dan on Tenhou |
| Opponent-Limited Online Search for Imperfect Information Games | Liu, Fu, Fu, Yang | 2023 | [ICML 2023](https://proceedings.mlr.press/v202/liu23k.html) | OLSS: imperfect-info subgame solving with opponent-limited pruning, much faster than common-knowledge methods. Tested on 2p mahjong. | Core LuckyJ search part; search-as-feature for real-time adaptation |
| Look-ahead Reasoning with Learned Model in Imperfect Information Games (LAMIR) | Kubicek, Lisy | 2026 | [ICLR 2026](https://openreview.net/forum?id=NnBbr4hI8a) | Learns abstract game models from interaction, enables CFR-based depth-limited look-ahead in imperfect-info games. Tested on 2p games. [arXiv:2510.05048](https://arxiv.org/abs/2510.05048), [Code](https://github.com/aicenter/lamir) | Inspiration for Hydra inference-time search (historical `SEARCH_PGOI.md` planning surface; no standalone doc in current repo). Referenced in TACC allocation proposal as "LAS" framing. |
| Hierarchical CFR with Policy Abstraction in Mahjong | (CFR-p authors) | 2023 | [arXiv:2307.12087](https://arxiv.org/abs/2307.12087) | Vanilla CFR on simplified 2p 68-tile Mahjong with hierarchical policy abstraction. Even reduced game had ~10^43 leaf nodes pre-abstraction. Only known CFR use on Mahjong variant. | Confirms 4p Mahjong intractable for tabular CFR. Supports Hydra RL over game-theoretic solving. |

### General Game AI

| Paper | Authors | Year | Venue / URL | Key Contribution | Relevance to Hydra |
|-------|---------|------|-------------|------------------|---------------------|
| Mastering Chess and Shogi by Self-Play with General Reinforcement Learning Algorithm (AlphaZero) | Silver et al. | 2017 | [arXiv](https://arxiv.org/abs/1712.01815) | MCTS + neural net self-play, general game learning | Baseline game AI paradigm |
| Superhuman AI for Multiplayer Poker (Pluribus) | Brown, Sandholm | 2019 | Science | Imperfect-info game solving at scale | Opponent modeling in imperfect-info games |
| OpenAI Five | OpenAI | 2019 | [OpenAI](https://openai.com/five/) | Large-scale PPO for complex games | Training stability, PPO scaling |
| AlphaStar: Mastering Real-Time Strategy Game StarCraft II | Vinyals et al. | 2019 | Nature | League training for multi-agent robustness | League training method for Phase 3 |
| Mastering Game of Stratego with Model-Free Multiagent Reinforcement Learning (DeepNash) | Perolat et al. | 2022 | Science | R-NaD for Nash approximation | Considered, rejected; Nash less fit for 4p ranking |

### Architecture Components

| Paper | Authors | Year | Venue / URL | Key Contribution | Relevance to Hydra |
|-------|---------|------|-------------|------------------|---------------------|
| Squeeze-and-Excitation Networks | Hu et al. | 2018 | CVPR | SE attention blocks for channel recalibration | Backbone: dual-pool SE attention each ResBlock |
| CBAM: Convolutional Block Attention Module | Woo et al. | 2018 | ECCV | Channel + spatial attention via dual-pool (avg+max) shared MLP | Hydra SE uses CBAM channel-attn part (dual-pool shared MLP) |
| Group Normalization | Wu & He | 2018 | ECCV | Batch-independent normalization | Training stability: GroupNorm(32) over BatchNorm |
| Proximal Policy Optimization Algorithms | Schulman et al. | 2017 | [arXiv](https://arxiv.org/abs/1707.06347) | PPO clipped surrogate objective | Core RL algo for Phases 2-3 |
| Attention Is All You Need | Vaswani et al. | 2017 | NeurIPS | Transformer architecture | Considered for backbone; used by Kanachan, Tjong |
| Learning Confidence for Out-of-Distribution Detection | DeVries, Taylor | 2018 | [arXiv:1802.04865](https://arxiv.org/abs/1802.04865) | Confidence estimation as training regularization | Used by NAGA for calibrated action dists |

---

## Open Source Projects

### Mahjong AI

| Project | URL | Language | Stars | License | Notes |
|---------|-----|----------|-------|---------|-------|
| Mortal | https://github.com/Equim-chan/Mortal | Rust/Python | 1.3K+ | AGPL-3.0-or-later | Primary competitor. ResNet(40 blocks, 192ch) + Channel Attention -> DQN(Dueling) + CQL. Ref only; AGPL, no code derivation. Study: obs encoding (1012x34), action masking (46 actions), GRP head, 1v3 duplicate eval. Weights have extra distribution restrictions beyond AGPL. |
| Kanachan | https://github.com/Cryolite/kanachan | C++/Python | 300+ | Unlicensed | **Transformer encoder (BERT-style)**. Two configs: base (~90M params, 12L/768d), large (~310M params, 24L/1024d). Trained on 65M+ Majsoul rounds (Gold+), zero hand-crafted features. 184 tokens: 33 sparse + 6 numeric + 113 progression + 32 candidates. Pipeline: BC -> curriculum fine-tune -> offline RL (IQL/ILQL/CQL). **No published benchmarks** despite multi-year dev (public repo 2021-08-05). Param count makes online RL infeasible. WARNING: no LICENSE file; do not depend on code. |
| Akochan | https://github.com/critter-mj/akochan | C++ | ~280 | Custom (restrictive, Japanese) | EV heuristic engine with explicit suji/kabe/genbutsu analysis. Not ML. Matters: good defense sanity check. If Hydra NN disagrees in obvious defense spots, bug likely. Also backend for original mjai-reviewer. |
| MahjongAI | https://github.com/erreurt/MahjongAI | Python | ~450 | — | Extensible agent framework with pluggable strategies. Less about arch, more about Tenhou client impl if Hydra ever connects to Tenhou protocol. |
| AlphaJong | https://github.com/Jimboom7/AlphaJong | JavaScript | — | — | Browser heuristic engine (NOT AlphaZero despite name). Tunable offense/defense sliders. Only weak baseline; useful to sanity-check Hydra beats simple heuristics by lot. |
| mjai-manue | https://github.com/gimite/mjai-manue | Ruby | 37 | — | Original MJAI protocol client. Protocol ref; defines canonical MJAI message format Hydra must match. |
| NAGA | https://dmv.nico/en/articles/mahjong_ai_naga/ | — | — | Commercial | **Pure supervised learning**. 4 independent CNNs (discard, call, riichi, kan) trained on Tenhou Houou logs via imitation learning. No self-play, no RL. Uses confidence estimation (DeVries & Taylor 2018) as training regularization and Guided Backpropagation (Springenberg et al. 2014) for interpretability. 5 playstyle variants (Omega, Gamma, Nishiki, Hibakari, Kagashi) differ by training data, not arch. CNN details (layers, filters, input shape) not public; [DMV article](https://dmv.nico/en/articles/mahjong_ai_naga/) only official technical doc. Achieved 10-dan on Tenhou (26,598 games — unverified; number absent from DMV article and no public source found), current models estimated ~9-dan stable. Not open-source. NAGA "match%" common but imperfect benchmark. |
| LuckyJ | https://haobofu.github.io/ | — | — | Commercial | Tencent mahjong AI (绝艺/JueYi brand). 10-dan on Tenhou in 1,321 games, 10.68 stable dan, strongest known AI. ACH + OLSS, pure self-play. See [COMMUNITY_INSIGHTS § LuckyJ](COMMUNITY_INSIGHTS.md#4-luckyj-tencent) for arch analysis. |

### Analysis & Review Tools

| Project | URL | Stars | Description |
|---------|-----|-------|-------------|
| mjai-reviewer | https://github.com/Equim-chan/mjai-reviewer | 1.1K+ | CLI makes HTML review reports with Q-value deltas per discard. Primary tool for Hydra play-quality eval. Apache-2.0, safe to use directly. |
| mjai-reviewer3p | https://github.com/hidacow/mjai-reviewer3p | — | 3-player (sanma) fork of mjai-reviewer. Matters only if Hydra targets sanma. |
| killer_mortal_gui | https://github.com/killerducky/killer_mortal_gui | — | Enhanced Mortal review with deal-in heuristic multipliers (ryanmen 3.5x, kanchan suji-trap 2.6x, honor tanki/shanpon 1.7x, etc). Best public ref for tile-danger calibration; useful to validate Hydra defense signals. |
| crx-mortal | https://github.com/announce/crx-mortal | — | Chrome extension for in-browser Mortal analysis. Low training relevance. |
| mjai-batch-review | https://github.com/Xerxes-2/mjai-batch-review | 9 | Batch-analyze many logs at once. Useful for large-scale eval; faster than one-by-one review. |

### Mortal Forks

| Fork | URL | Key Difference |
|------|-----|----------------|
| Mortal-Policy | https://github.com/Nitasurin/Mortal-Policy | PPO over DQN, GroupNorm over BatchNorm, entropy weight tuning. AGPL-3.0, ref only. Matters: closest public ref to Hydra arch choice (PPO + GroupNorm). Study AWR -> PPO transition path and policy-gradient handling for 46-action mahjong space. |

### Components

| Project | URL | Language | License | Purpose |
|---------|-----|----------|---------|---------|
| xiangting | https://github.com/Apricot-S/xiangting | Rust | MIT | Primary shanten lib. Compile-time embedded tables (~200KB), `no_std`, 3p support, returns shanten num + necessary/unnecessary tile sets. 34x faster than brute-force replacement-tile calc. Hydra uses for obs channels (shanten features) and action masking. |
| xiangting-py | — | Python | MIT | Python bindings for xiangting via PyO3. Useful for training-side shanten calc if needed. |
| tomohxx/shanten-number | — | C++ | LGPL-3.0 | Original table-based shanten algo xiangting derives from. Algo ref only; LGPL blocks static linking. Tables: suhai (1.9M entries, ~19.4MB), jihai (78K entries, ~0.78MB). Base-5 encoding for tile-state indexing. |
| PyO3 | https://pyo3.rs/ | Rust | Apache-2.0 | Rust<->Python FFI for game engine bindings into training loop. |
| rayon | https://docs.rs/rayon/ | Rust | Apache-2.0 | Work-stealing data parallelism for batch game simulation. |
| serde / serde_json | https://serde.rs/ | Rust | Apache-2.0 | JSON ser/de for MJAI protocol parsing. |
| ndarray | https://docs.rs/ndarray/ | Rust | Apache-2.0 | N-D array ops for observation tensors. |
| ort | https://docs.rs/ort/ | Rust | Apache-2.0 | ONNX Runtime Rust bindings. Primary self-play inference engine: loads exported PyTorch model as ONNX, runs forward passes with CUDA EP, CUDA graphs, I/O binding for <5ms latency. Hot path in self-play; inference speed caps training throughput. |
| tract | https://docs.rs/tract/ | Rust | MIT OR Apache-2.0 | Pure Rust ML inference engine (no C++ deps). CPU-only fallback for no-CUDA envs. Good for CI and CPU-only deploy. |
| candle | https://github.com/huggingface/candle | Rust | Apache-2.0 | HuggingFace Rust ML framework with CUDA and Metal. Alt to ONNX path: direct Rust inference, no PyTorch -> ONNX export. Worth testing if ONNX export hurts accuracy or operator support. |
| Burn | https://github.com/tracel-ai/burn | Rust | MIT OR Apache-2.0 | Native Rust training + inference with WGPU, CUDA, LibTorch backends. Long-term option to move full training loop to Rust, remove Python. ONNX import support growing. |
| tch-rs | — | Rust | MIT OR Apache-2.0 | Rust bindings for LibTorch. Alt to PyO3: call LibTorch from Rust directly. Less FFI overhead, less Python flexibility. |
| mahjong (Python) | https://github.com/MahjongRepository/mahjong | Python | MIT | Hand-scoring oracle: yaku, han/fu, score calc; validated on 11M+ Tenhou hands. Pin v1.4.0. Dev dep for Rust engine verification and test-case extraction. |
| agari | https://github.com/rysb-dev/agari | Rust | MIT (no LICENSE file) | Complete scoring engine (35 yaku, fu, payment, hand decomposition, ~100 unit tests). Cleanest Rust mahjong scorer arch; study `HandDecomposition` trait and `Fu` calc for Hydra scoring module. `Cargo.toml` says MIT but repo lacks LICENSE; safe as reference. |
| mahc | https://github.com/DrCheeseFace/mahc | Rust | BSD-3 | Scoring lib with explicit `Fu` enum (each fu source named, not magic numbers). 38 yaku, 30K crates.io downloads. Study `Fu` enum pattern; makes fu calc self-documenting, testable vs Mortal opaque style. |
| mahjax | https://github.com/nissymori/mahjax | Python/JAX | Apache-2.0 | JAX-vectorized riichi env hitting ~1.6M steps/sec on 8xA100 via JIT. Matters for self-play: GPU vectorization can run thousands of games at once, maybe 10-100x faster than sequential Rust sim for training data. Study state representation and vectorized game logic. |
| RiichiEnv | https://github.com/smly/RiichiEnv | Rust/Python | Apache-2.0 | Gym-style RL env with Rust core + Python bindings, Mortal-compatible MJAI output. Verified over 1M+ games. Useful because ready Gym interface slots into standard training loops. Also correctness oracle for Hydra Rust engine. |
| Meowjong | https://github.com/VictorZXY/Meowjong | Python | MIT | Only open-source 3p (sanma) mahjong AI. IEEE CoG 2022. Includes 5 CNN variants and Tenhou sanma log downloader. Matters if Hydra targets sanma; only ref impl with published results. Also shows CNNs work for reduced-player mahjong. |
| CleanRL | https://github.com/vwxyzjn/cleanrl | Python | MIT | Single-file PPO impl (~250 lines) with wandb integration. Comes with "37 impl Details of PPO" blog post covering crucial hyperparams/tricks. Hydra PPO should validate against CleanRL: clipping, advantage norm, value-loss clipping, entropy schedule. Blog required reading before writing PPO. |
| OpenSpiel | https://github.com/google-deepmind/open_spiel | C++/Python | Apache-2.0 | DeepMind game RL framework with 70+ games, incl AlphaZero, MCTS, CFR, self-play loops. Matters for Hydra Phase 3 (league training): study self-play loop arch, opponent pools, ELO tracking, policy selection. Also has imperfect-info solvers for belief-state ideas. |
| Microsoft Olive | https://github.com/microsoft/Olive | Python | MIT | End-to-end model optimization: PyTorch -> ONNX with quantization, pruning, op fusion, shape inference via YAML. Matters for self-play inference speed: millions of forward passes; even 2x from INT8 halves wall time. Use after model arch stabilizes. |
| rlcard | https://github.com/datamllab/rlcard | Python | MIT | RL toolkit with mahjong env and prebuilt DQN/NFSP agents. Lower fidelity than mahjax/RiichiEnv (simplified rules), but useful for fast prototyping reward shaping and loop mechanics before full env. |
| mjai.app | https://github.com/smly/mjai.app | — | AGPL-3.0 | RiichiLab competition platform using MJAI protocol with Docker eval. Matters because target venue; Hydra must emit MJAI-compatible output to compete and benchmark. Study Docker submission format and eval harness. |

### Protocol & Infrastructure

| Project | URL | Description |
|---------|-----|-------------|
| mjai | https://github.com/gimite/mjai | Original MJAI protocol server |
| mjai-gateway | https://github.com/tomohxx/mjai-gateway | MJAI <-> Tenhou translator |

---

## Community Resources

### Documentation

| Resource | URL | Content |
|----------|-----|---------|
| Mortal docs | https://mortal.ekyu.moe | Arch insights, perf data, playstyle stats |
| MJAI Protocol Wiki | https://gimite.net/pukiwiki/index.php?MJAI | Standard protocol spec (WARNING: may need login) |
| MJAI Web Reviewer | https://mjai.ekyu.moe/ | Web UI for instant game reviews |
| Tenhou docs | https://tenhou.net/man/ | Tenhou log format spec (old `/doc/` path = 404) |
| Majsoul API | Various GitHub repos | Log extraction via WebSocket capture |
| NAGA docs | https://dmv.nico/en/articles/mahjong_ai_naga/ | Commercial AI arch overview |
| Riichi Wiki — NAGA | https://riichi.wiki/Mahjong_AI_%E3%80%8CNAGA%E3%80%8D | Community wiki page on NAGA |
| Phoenix Paper | https://csci527-phoenix.github.io/documents/Paper.pdf | Open-source reproducible mahjong agent |
| ONNX Runtime | https://onnxruntime.ai/ | Production inference runtime |

### Discussion Sources

| Source | Topics |
|--------|--------|
| Mortal GitHub Issues & Discussions | Known weaknesses, training problems, oracle guiding removal |
| r/Mahjong (Reddit) | Player view on AI behavior, known weaknesses |
| Discord (Riichi Mahjong) | Community testing, strategy discussion |
| Tenhou forums | High-level play analysis |
| Note.com mahjong blogs (Japanese) | 場況 (bakyou) struggles, efficiency vs situational tactics |

---

## Training Data Sources

> See [ECOSYSTEM.md § Data Sources & Datasets](ECOSYSTEM.md#3-data-sources--datasets) for current training-data summary. Separate `archive/DATA_SOURCES.md` not present in current repo.

---

## Algorithm References

### Shanten Calculation

| Resource | Description |
|----------|-------------|
| tomohxx Algorithm | Set-based recurrence, O(n); table lookup |
| tomohxx Tables | Suhai: 1,940,777 entries x 10 bytes (~19.4 MB); Jihai: 78,032 entries x 10 bytes (~0.78 MB) |
| tomohxx Indexing | Base-5 encoding: `tiles.iter().fold(0, |acc, &x| acc * 5 + x as usize)` |
| tomohxx Compressed | shanten_suhai.bin.gz (191 KB), shanten_jihai.bin.gz (5.6 KB) |
| xiangting impl | Rust port with 3p support |
| Kanachan xiangting | LOUDS-based TRIE shanten calculator |
| Mahjong Algorithm Book | Japanese ref, theory background |
| Cryolite (2023) | Fast and Space-Efficient Algorithm for Calculating Deficient Numbers" |

### Suji / Kabe / Genbutsu

| Resource | Description |
|----------|-------------|
| Japanese Mahjong Strategy Books | Traditional defense theory |
| Daina Chiba's Defense | Quantitative suji analysis |
| Tenhou Player Guides | Statistical safety percentages |
| Suji Safety Note | Suji ~60-70% safe, not 100%; only protects vs ryanmen waits |
| Genbutsu Definition | 100% safe: tiles discarded by or after opponent riichi |
| Kabe Definition | All 4 visible -> no-chance wait; 3 visible = one-chance |
| Half-suji / Full-suji | One side visible vs both sides visible |
| killer_mortal_gui Heuristics | Ryanmen 3.5x, Kanchan 0.21x, Kanchan suji-trap 2.6x, Penchan 1.0x, Honor tanki/shanpon 1.7x; modifiers: Dora 1.2x, Ura-suji 1.3x, Matagi early 0.6x, Matagi riichi 1.2x, Red 5 discard 0.14x |

### Scoring

| Resource | Description |
|----------|-------------|
| Tenhou Scoring Tables | Standard yaku/fu calc |
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
| Suphx | Tenhou | 10-dan (5,373 games), 8.74 stable | 2020 | SL + RL + oracle guiding; paper says 100+ humans achieved 10-dan |
| LuckyJ | Tenhou | **10-dan (1,321 games), 10.68 stable** | 2023 | ACH + OLSS; statistically stronger than NAGA, Suphx |
| Mortal | — | **No ranked play** | — | Tenhou rejected Mortal AI account request ([FAQ](https://github.com/Equim-chan/mjai-reviewer/blob/master/faq.md): "Tenhou rejected my AI account request for Mortal because Mortal was developed by individual rather than company"). Community estimate ~7-dan via mjai-reviewer analysis. |
| NAGA | Majsoul | Celestial | 2022 | — |

---

## License Compatibility

> License policy: See [../infrastructure/INFRASTRUCTURE.md#license-compatibility](../infrastructure/INFRASTRUCTURE.md#license-compatibility)

---

## GitHub Discussions

Mortal repo discussions relevant to Hydra design choices:

| Discussion # | Topic | Key Insight |
|-------------|-------|-------------|
| (source code) | MC returns vs TD | Mortal uses MC returns, not TD, for Q-targets. Confirmed from source (`train.py` Q-target calc). `q_target = gamma^steps_to_done * kyoku_reward`, no bootstrap from next-state Q-values. Hydra follows same approach. |
| #27 | Batch size recommendations | Practical mahjong RL batch-size guidance |
| #43 | torch.compile speedup | `torch.compile` gives 15-20% training speedup on Mortal. Hydra should enable day one. |
| #52 | NextRankPredictor rationale | Aux task predicts next placement; stabilizes feature learning by giving backbone secondary target beyond Q-values |
| #64 | Catastrophic forgetting in online RL | Offline (BC) -> online (self-play) transition can forget offline knowledge. Equim-chan confirms real. Hydra must do gradual transition with replay-buffer mixing. |
| #70 | DeepCFR for GRP replacement | Community explored DeepCFR over GRP. Conclusion: impractical for 4p mahjong due to game-tree size. |
| #91 | Mortal-Policy (PPO fork) | Nitasurin PPO fork open-sourced. Confirms PPO works for mahjong, validates Hydra algo choice. |
| #102 | Oracle guiding removed | Equim-chan: "didn't bring improvements in practice." Critical for Hydra: Suphx oracle guiding tried, abandoned by Mortal author. Hydra oracle approach must differ from naive Suphx impl. |
| #108 | Maximum player score in observations | Discussion on score cap at 30K in obs encoding. Relevant to Hydra uncapped-score encoding choice. |

---

## GitHub Issues

Mortal repo issues relevant to Hydra improvements:

| Issue # | Description |
|---------|-------------|
| #111 | Overtake score miscalc. Mortal miscalculates hand-building near placement thresholds; motivates Hydra uncapped-score encoding |
| #113 | Rating system closure discussion; community debate on shutting down Mortal rating feature |

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