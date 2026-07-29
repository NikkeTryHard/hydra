<p align="center">
<img src="assets/hydra.webp" alt="Hydra2 banner" width="720">
</p>

<h1 align="center">Hydra2</h1>

Hydra2 is a Riichi Mahjong AI research project focused on correct, reproducible training, belief modeling, GPU search, and evaluation under Tenhou four-player hanchan rules. The long-term goal remains stronger play than LuckyJ.

## Project update

Hydra1 did not reach its initially anticipated strength relative to Mortal. Hydra2 is a clean research reset: it keeps only validated infrastructure lessons and explores new model and search architectures instead of treating Hydra1 experiments as proven improvements.

Current Hydra2 work is specification-first. No Hydra2 model, planner, belief system, or simulator adapter is implemented yet. The preserved Rust MJAI dataset packager is the only existing code component.

Primary research directions:

- actor-visible policy/value models with exact legal masking;
- public-belief search using natural particles, ISMCTS, and DESPOT baselines;
- public-belief reusable event forests for safe opponent-time pondering;
- exact-rules Gumbel search, local resolving, and search distillation;
- optional self-play RL and independently qualified architecture ablations.

No current result establishes superiority over LuckyJ or Mortal.

## Repository map

- [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md): durable direction, milestones, dependencies, and current status.
- [`docs/BUILD_EXECUTION_PLAN.md`](docs/BUILD_EXECUTION_PLAN.md): normative work-package order and completion gates.
- [`docs/IMPLEMENTATION_SPEC.md`](docs/IMPLEMENTATION_SPEC.md): canonical schemas, APIs, algorithms, and fixtures.
- [`docs/ALGORITHM_EXPERIMENT_BLUEPRINT.md`](docs/ALGORITHM_EXPERIMENT_BLUEPRINT.md): research candidates, equations, comparisons, and promotion rules.
- [`tools/mjai-dataset-packager/`](tools/mjai-dataset-packager/): preserved restart-safe Rust MJAI dataset packager, including its tracked [`src/main.rs`](tools/mjai-dataset-packager/src/main.rs).

## Current build boundary

Do not use Hydra1 setup or training commands here. Hydra2 has not yet implemented its Python/Pixi training environment. Execute only unblocked packages from the build execution plan; missing contracts or artifacts mean blocked, not permission to invent substitutes.

## Compute support

<p align="center">
<img src="assets/delta.webp" alt="Delta GPU node" width="720">
</p>

Hydra2 remains sponsored with compute support from Delta advanced computing and data resources, supported by the National Science Foundation (award OAC 2005572) and the State of Illinois. Delta is a joint effort of the University of Illinois Urbana-Champaign and its National Center for Supercomputing Applications.
