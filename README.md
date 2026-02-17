# Hydra

Open-source Riichi Mahjong AI. The goal is to build an AI stronger than [Mortal](https://github.com/Equim-chan/Mortal) with open weights — no restrictive licensing, no closed models.

## Goal

Train a mahjong AI that:
- Beats Mortal (community-estimated ~7-dan play strength) in head-to-head evaluation
- Releases weights under a permissive license (MIT)
- Adds opponent modeling — something no existing mahjong AI does

## Architecture

SE-ResNet backbone (40 blocks, 256 channels) with PPO training. Three-phase pipeline: supervised warm start on ~6.6M games (2M Tenhou Houou + 1M Majsoul Throne + 3M Majsoul Jade) → oracle distillation → league self-play. Hybrid Rust (game engine) + Python (training) stack.

See [research/HYDRA_SPEC.md](research/HYDRA_SPEC.md) for the architecture specification and [research/TRAINING.md](research/TRAINING.md) for the training pipeline.

## Research

| File | What's In It |
|------|-------------|
| [HYDRA_SPEC.md](research/HYDRA_SPEC.md) | Architecture, input encoding, output heads, inference |
| [TRAINING.md](research/TRAINING.md) | Training pipeline (phases 1-3), loss functions, hyperparameters, roadmap |
| [MORTAL_ANALYSIS.md](research/MORTAL_ANALYSIS.md) | Mortal's architecture, training details, confirmed weaknesses |
| [OPPONENT_MODELING.md](research/OPPONENT_MODELING.md) | Safety planes, tenpai/danger heads, oracle distillation |
| [INFRASTRUCTURE.md](research/INFRASTRUCTURE.md) | Rust/Python stack, data pipeline, training infra, hardware, deployment |
| [SEEDING.md](research/SEEDING.md) | RNG hierarchy, reproducibility, evaluation seed bank |
| [CHECKPOINTING.md](research/CHECKPOINTING.md) | Checkpoint format, save protocol, retention policy |
| [ECOSYSTEM.md](research/ECOSYSTEM.md) | Every useful repo, tool, and framework in the mahjong AI space |
| [REWARD_DESIGN.md](research/REWARD_DESIGN.md) | Hydra's reward function design and RVR variance reduction |
| [COMMUNITY_INSIGHTS.md](research/COMMUNITY_INSIGHTS.md) | Reddit, blogs, Mortal GitHub discussions — community knowledge |
| [REFERENCES.md](research/REFERENCES.md) | All citations with why each matters for Hydra |
| [TESTING.md](research/TESTING.md) | Testing strategy, correctness verification, property-based tests |
| [ABLATION_PLAN.md](research/ABLATION_PLAN.md) | Structured experiment queue with hypotheses and protocols |

## Status

Research phase — documentation and architecture specification complete. Implementation not yet started.

## License

MIT
