# Hydra

Open-source Riichi Mahjong AI. The goal is to build an AI that rivals [LuckyJ](https://haobofu.github.io/) (Tencent AI Lab, 10.68 stable dan on Tenhou) with open weights.

## Goal

Train a mahjong AI that:
- Surpasses [Mortal](https://github.com/Equim-chan/Mortal) (~7-dan) and approaches LuckyJ-level play (10+ dan) in head-to-head evaluation
- Releases weights under a permissive license
- Adds opponent modeling and inference-time search — the two capabilities that separate LuckyJ from all other mahjong AIs

## Architecture

SE-ResNet backbone (40 blocks, 256 channels) with PPO training. Three-phase pipeline: supervised warm start on ~6.6M games (2M Tenhou Houou + 1M Majsoul Throne + 3M Majsoul Jade) -> oracle distillation -> league self-play. 100% Rust stack using [Burn](https://github.com/tracel-ai/burn) framework with burn-tch (libtorch/cuDNN) backend.

See [research/HYDRA_SPEC.md](research/HYDRA_SPEC.md) for the architecture specification and [research/TRAINING.md](research/TRAINING.md) for the training pipeline.

## Research

| File | What's In It |
|------|-------------|
| [HYDRA_SPEC.md](research/HYDRA_SPEC.md) | Architecture, input encoding, output heads, inference |
| [TRAINING.md](research/TRAINING.md) | Training pipeline (phases 1-3), loss functions, hyperparameters, roadmap |
| [MORTAL_ANALYSIS.md](research/MORTAL_ANALYSIS.md) | Mortal's architecture, training details, confirmed weaknesses |
| [OPPONENT_MODELING.md](research/OPPONENT_MODELING.md) | Safety planes, 9-head specs (tenpai/danger/FiLM/call-intent/Sinkhorn), RSA deception, CVaR, L0 observer |
| [INFRASTRUCTURE.md](research/INFRASTRUCTURE.md) | Rust stack, data pipeline, training infra, hardware, deployment |
| [SEEDING.md](research/SEEDING.md) | RNG hierarchy, reproducibility, evaluation seed bank |
| [CHECKPOINTING.md](research/CHECKPOINTING.md) | Checkpoint format, save protocol, retention policy |
| [ECOSYSTEM.md](research/ECOSYSTEM.md) | Every useful repo, tool, and framework in the mahjong AI space |
| [REWARD_DESIGN.md](research/REWARD_DESIGN.md) | Hydra's reward function design and RVR variance reduction |
| [COMMUNITY_INSIGHTS.md](research/COMMUNITY_INSIGHTS.md) | Reddit, blogs, Mortal GitHub discussions — community knowledge |
| [REFERENCES.md](research/REFERENCES.md) | All citations with why each matters for Hydra |
| [TESTING.md](research/TESTING.md) | Testing strategy, correctness verification, property-based tests |
| [ABLATION_PLAN.md](research/ABLATION_PLAN.md) | Structured experiment queue with hypotheses and protocols |
| [AUDIT_LOG.md](research/AUDIT_LOG.md) | Change history tracking SSOT fixes across all research docs |
| [RUST_STACK.md](research/RUST_STACK.md) | 100% Rust decision, Burn framework, Python migration, all concerns resolved |

### Documentation Ownership (SSOT)

- `MORTAL_ANALYSIS.md`: authoritative Mortal strengths/limitations and architecture evidence
- `OPPONENT_MODELING.md`: authoritative safety-channel, tenpai-head, and danger-head design rationale
- `HYDRA_SPEC.md`: high-level architecture/inference summary with cross-references to detail docs
- `COMMUNITY_INSIGHTS.md`: community observations and hypotheses (non-canonical for hard architectural claims)

## Status

Research phase — documentation and architecture specification complete. Implementation not yet started.

## License

- **hydra-core** (encoder, training pipeline): [BSL 1.1](hydra-core/LICENSE) -- free for non-commercial use, converts to Apache-2.0 on 2031-03-02
- **hydra-engine** (game rules): Apache-2.0 (vendored from riichienv-core)
