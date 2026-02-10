# Hydra

Open-source Riichi Mahjong AI. The goal is to build an AI stronger than [Mortal](https://github.com/Equim-chan/Mortal) with open weights — no restrictive licensing, no closed models.

## Goal

Train a mahjong AI that:
- Beats Mortal (currently Tenhou 10-dan) in head-to-head evaluation
- Releases weights under a permissive license (MIT)
- Adds opponent modeling — something no existing mahjong AI does

## Architecture

SE-ResNet backbone (40 blocks, 256 channels) with PPO training. Three-phase pipeline: supervised warm start on 3M+ Tenhou Phoenix games → oracle distillation → league self-play. Hybrid Rust (game engine) + Python (training) stack.

See [research/HYDRA_SPEC.md](research/HYDRA_SPEC.md) for the full specification.

## Research

| File | What's In It |
|------|-------------|
| [HYDRA_SPEC.md](research/HYDRA_SPEC.md) | Architecture, training pipeline, loss functions, hyperparameters |
| [MORTAL_ANALYSIS.md](research/MORTAL_ANALYSIS.md) | Mortal's architecture, training details, confirmed weaknesses |
| [OPPONENT_MODELING.md](research/OPPONENT_MODELING.md) | Safety planes, tenpai/danger heads, oracle distillation |
| [INFRASTRUCTURE.md](research/INFRASTRUCTURE.md) | Rust/Python stack, module layout, hardware requirements |
| [ECOSYSTEM.md](research/ECOSYSTEM.md) | Every useful repo, tool, and framework in the mahjong AI space |
| [COMMUNITY_INSIGHTS.md](research/COMMUNITY_INSIGHTS.md) | Reddit, blogs, Mortal GitHub discussions — community knowledge |
| [REFERENCES.md](research/REFERENCES.md) | All citations with why each matters for Hydra |

## Status

Research phase. Open design issues: [#7](../../issues/7)–[#14](../../issues/14).

## License

MIT
