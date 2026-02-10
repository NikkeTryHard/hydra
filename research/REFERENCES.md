# Hydra References

Single source of truth for all citations in the Hydra project.

---

## Academic Papers

### Mahjong AI

| Paper | Authors | Year | Venue / URL | Key Contribution | Relevance to Hydra |
|-------|---------|------|-------------|------------------|---------------------|
| Suphx: Mastering Mahjong with Deep Reinforcement Learning | Junjie Li, Sotetsu Koyamada, Qiwei Ye, Guoqing Liu, Chao Wang, Ruihan Yang, Li Zhao, Tao Qin, Tie-Yan Liu, Hsiao-Wuen Hon | 2020 | [arXiv:2003.13590](https://arxiv.org/abs/2003.13590) | Oracle guiding, Global Reward Prediction (GRP), run-time policy adaptation, 10-dan achievement on Tenhou | Core inspiration for oracle distillation and GRP head design |
| Tjong: A Transformer-based Mahjong AI via Hierarchical Decision-Making and Fan Backward | Xiali Li, Bo Liu, Zhi Wei, Zhaoqi Wang, Licheng Wu | 2024 | [CAAI Trans. Intel. Tech.](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cit2.12298) DOI: 10.1049/cit2.12298 | Hierarchical decision-making (action type → tile selection), transformer architecture for game sequences, fan backward reward shaping | Alternative architecture reference; fan backward considered for yaku awareness |
| Information Set Monte Carlo Tree Search | P. I. Cowling, E. J. Powley, D. Whitehouse | 2012 | [IEEE TCIAIG](https://ieeexplore.ieee.org/document/6203567) | Foundation for handling imperfect information via determinization and information-set sampling | Theoretical basis for imperfect-info game approaches |
| Real-time Mahjong AI based on Monte Carlo Tree Search (Bakuuchi) | Mizukami et al. | 2014 | IEEE | Pre-deep-learning SOTA using ISMCTS + rule-based heuristics | Historical baseline for MCTS approaches |
| Phoenix: Open-Source Reproducible Mahjong Agent | — | 2023 | [Paper](https://csci527-phoenix.github.io/documents/Paper.pdf) | Transparent baseline with interpretable decision-making | Open-source baseline reference |
| Building a Computer Mahjong Player via Deep Convolutional Neural Networks | — | 2018 | IEEE | CNN for Mahjong, baseline methods | Early CNN approach for mahjong |
| Reward Variance Reduction for Limited-Compute RL | — | 2022 | IEEE CoG | RVR technique for reducing gradient noise from luck variance, single-GPU feasibility | Enables training on limited hardware; hand-luck baseline subtraction |

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
| Group Normalization | Wu & He | 2018 | ECCV | Batch-independent normalization | Training stability: GroupNorm(32) replaces BatchNorm |
| Proximal Policy Optimization Algorithms | Schulman et al. | 2017 | [arXiv](https://arxiv.org/abs/1707.06347) | PPO clipped surrogate objective | Core RL algorithm for Phases 2-3 |
| Attention Is All You Need | Vaswani et al. | 2017 | NeurIPS | Transformer architecture | Considered for backbone; used by Kanachan and Tjong |

---

## Open Source Projects

### Mahjong AI

| Project | URL | Language | Stars | License | Notes |
|---------|-----|----------|-------|---------|-------|
| Mortal | https://github.com/Equim-chan/Mortal | Rust/Python | 1,334 | AGPL-3.0-or-later | Reference only — cannot derive code; weights have separate distribution restrictions. ResNet + Channel Attention, DQN + CQL |
| Kanachan | https://github.com/Cryolite/kanachan | C++/Python | 326 | MIT | Transformer-based (BERT-style), trained on 100M+ Mahjong Soul rounds, no hand-crafted features |
| Akochan | https://github.com/critter-mj/akochan | C++ | 281 | Custom (restrictive, Japanese) | C++ EV-based heuristic engine with suji/kabe/genbutsu analysis |
| Mjx | — | Python/JAX | — | Apache-2.0 | GPU-accelerated mahjong simulator |
| riichi-rs | — | Rust | — | MIT | Basic Rust riichi implementation |
| MahjongAI | https://github.com/erreurt/MahjongAI | Python | 446 | — | Extensible general-purpose mahjong agent |
| AlphaJong | https://github.com/Jimboom7/AlphaJong | JavaScript | — | — | Browser heuristic simulation (NOT AlphaZero); tunable defense/offense balance |
| mjai-manue | https://github.com/gimite/mjai-manue | Ruby | 37 | — | Original MJAI client |
| NAGA | https://dmv.nico/en/articles/mahjong_ai_naga/ | — | — | Commercial | Deep CNN trained on hundreds of millions of Tenhou Houou-room games; Tenhou 10-dan; not open-source |

### Analysis & Review Tools

| Project | URL | Stars | Description |
|---------|-----|-------|-------------|
| mjai-reviewer | https://github.com/Equim-chan/mjai-reviewer | 1,168 | CLI for HTML review reports with Q-value display |
| mjai-reviewer3p | https://github.com/hidacow/mjai-reviewer3p | — | 3-player (sanma) review support |
| killer_mortal_gui | https://github.com/killerducky/killer_mortal_gui | — | Enhanced stats + deal-in heuristics (suji trap multipliers) |
| crx-mortal | https://github.com/announce/crx-mortal | — | Chrome extension for in-browser Mortal analysis |
| mjai-batch-review | https://github.com/Xerxes-2/mjai-batch-review | 9 | Batch analyze Mahjong Soul logs |

### Mortal Forks

| Fork | URL | Key Difference |
|------|-----|----------------|
| Mortal-Policy | https://github.com/Nitasurin/Mortal-Policy | PPO instead of DQN, GroupNorm, entropy weight |

### Integration Projects

| Project | URL | Description |
|---------|-----|-------------|
| Akagi | https://github.com/shinkuan/Akagi | Real-time MITM assistant for Majsoul/Tenhou/RiichiCity (714 stars) |
| Riki | — | Riichi City integration client helper |

### Components

| Project | URL | Language | License | Purpose |
|---------|-----|----------|---------|---------|
| xiangting | https://github.com/Apricot-S/xiangting | Rust | MIT | Shanten calculation (3-player support) |
| xiangting-py | — | Python | MIT | Python bindings for xiangting |
| tomohxx/shanten-number | — | C++ | LGPL-3.0 | Reference shanten algorithm (table-based lookup) — algorithm reference only, Hydra uses xiangting (MIT) instead |
| PyO3 | https://pyo3.rs/ | Rust | Apache-2.0 | Rust-Python FFI bindings |
| rayon | https://docs.rs/rayon/ | Rust | Apache-2.0 | Data parallelism for batch simulation |
| serde / serde_json | https://serde.rs/ | Rust | Apache-2.0 | JSON serialization / MJAI parsing |
| ndarray | https://docs.rs/ndarray/ | Rust | Apache-2.0 | Tensor operations |
| ort | https://docs.rs/ort/ | Rust | — | ONNX Runtime bindings for Rust inference |
| tract | https://docs.rs/tract/ | Rust | — | Pure Rust ML inference |
| candle | https://github.com/huggingface/candle | Rust | — | HuggingFace Rust ML framework |
| Burn | https://github.com/tracel-ai/burn | Rust | — | Native Rust ML training framework (WGPU, LibTorch, CUDA backends) |
| tch-rs | — | Rust | — | Rust bindings for LibTorch |
| mahjong (Python) | https://github.com/MahjongRepository/mahjong | Python | — | Python shanten/yaku/scoring library (448 stars) |
| rlcard | https://github.com/datamllab/rlcard | Python | — | RL toolkit with mahjong environment |
| mjai.app | https://github.com/smly/mjai.app | — | — | Web-based mahjong simulator |

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
| MJAI Protocol Wiki | https://gimite.net/pukiwiki/index.php?MJAI | Standard protocol specification |
| MJAI Web Reviewer | https://mjai.ekyu.moe/ | Web interface for instant game reviews |
| Tenhou Documentation | https://tenhou.net/doc/ | Tenhou log format specification |
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

| Source | Volume | Quality | Access |
|--------|--------|---------|--------|
| Tenhou Phoenix Logs | 2M+ games (~17M rounds) | Very high — R>2000 / Dan players | Archive/scraping, log download tools |
| Majsoul Throne Logs | 1M+ games | Very high — Saint3+ players | API extraction (WebSocket capture) |
| Majsoul Jade Logs | 2M+ games | High — Master+ players | API extraction (lower training weight) |
| Mahjong Soul (all ranks) | 100M+ rounds | Mixed (all ranks) | WebSocket capture |
| Self-play | Unlimited | Depends on policy quality | Generated during RL training |

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

## Technical References

### PyTorch

| Resource | URL |
|----------|-----|
| PyTorch Documentation | https://pytorch.org/docs/ |
| torch.compile Guide | https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html |
| Mixed Precision Training | https://pytorch.org/docs/stable/amp.html |

### Rust

| Resource | URL |
|----------|-----|
| Rust Book | https://doc.rust-lang.org/book/ |
| PyO3 User Guide | https://pyo3.rs/ |
| Rayon Parallelism | https://docs.rs/rayon/ |

### Experiment Tracking

| Resource | URL |
|----------|-----|
| Weights & Biases | https://wandb.ai/site |
| MLflow | https://mlflow.org/ |

---

## Benchmark References

### Tenhou Ranking

| Rank | Dan | Approx. Strength |
|------|-----|-------------------|
| R2000+ | 7-dan+ | Expert |
| R1800-2000 | 5-6 dan | Strong |
| R1600-1800 | 3-4 dan | Intermediate |

### AI Achievements

| AI | Platform | Achievement | Year |
|----|----------|-------------|------|
| Suphx | Tenhou | 10-dan (~180 humans ever achieved this) | 2020 |
| Mortal | Tenhou | 10-dan | 2023 |
| NAGA | Tenhou | 10-dan | 2018+ |
| NAGA | Majsoul | Celestial | 2022 |

---

## License Compatibility

### Safe to Use

| License | Commercial | Derivatives | Notes |
|---------|------------|-------------|-------|
| MIT | ✓ | ✓ | Preferred for Hydra components |
| Apache-2.0 | ✓ | ✓ | Patent grant included |
| BSD | ✓ | ✓ | Various versions acceptable |

### Cannot Use for Hydra

| License | Issue |
|---------|-------|
| AGPL | Copyleft, requires source disclosure for network use |
| GPL | Copyleft, restricts derivative works |
| LGPL | Weak copyleft, requires relinking capability for static linking |
| Mortal's Custom Restrictions | Additional restrictions on model weights beyond AGPL |

---

## GitHub Discussions

Mortal repository discussions relevant to Hydra design decisions:

| Discussion # | Topic |
|-------------|-------|
| #27 | Batch size recommendations for training |
| #64 | Online training forgetting problem (catastrophic forgetting) |
| #70 | DeepCFR feasibility for replacing GRP |
| #91 | Mortal-Policy (PPO fork) open sourcing |
| #102 | Oracle guiding removal rationale (Equim-chan: "didn't bring improvements in practice") |
| #108 | Score encoding / capping discussion |

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
