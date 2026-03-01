# Hydra Game Simulator Selection Plan

**Goal**: Find or build the fastest Riichi Mahjong game simulator for RL self-play training.

**Date**: 2026-02-28

---

## 1. Executive Summary

After surveying every notable mahjong simulator in existence, the recommendation is:

**Fork and extend RiichiEnv (Rust core) for correctness, and study mahjax (JAX) for GPU vectorization patterns.**

The "fastest simulator the world has ever seen" will likely be a **hybrid**: Rust for the authoritative game engine (correctness, MJAI compatibility, Mortal-verified) with a JAX-vectorized fast path for pure self-play data generation where you trade some rule edge-case coverage for 100x throughput.

But first, here is every option ranked.

---

## 2. Candidate Comparison Table

| Repo | Lang | License | Stars | Throughput | Rule Coverage | Batch Sim | Activity | Verdict |
|------|------|---------|-------|------------|---------------|-----------|----------|---------|
| **RiichiEnv** (smly) | Rust/Python | Apache-2.0 | ~50 | Unknown (Rust-fast) | Full 4p+3p, red dora, all kan, furiten, Tenhou+MJSoul rules | Via Python VectorEnv | Very active (2025-2026) | **TOP PICK (Rust)** |
| **mahjax** (nissymori) | Python/JAX | Apache-2.0 | ~23 | ~1.6M steps/sec (8xA100) | 4p, NO red dora yet, most rules | Via jax.vmap | Active (2025) | **TOP PICK (GPU)** |
| **Mortal/libriichi** (Equim-chan) | Rust | AGPL-3.0 | 1.3K+ | ~100K games/hr (est.) | Full 4p, battle-tested | No native batch | Maintained | Reference only (AGPL) |
| **Mjx** (mjx-project) | C++/Python | MIT | ~150 | 100x faster than Mjai | Full Tenhou rules | gRPC distributed | Build broken, stale | Study only |
| **PGX** (sotetsuk) | Python/JAX | Apache-2.0 | 1K+ | 10-100x vs PettingZoo | Only "Sparrow Mahjong" (simplified) | jax.vmap + pmap | Active | Architecture reference only |
| **mahjax-fixed** (hnsqdtt) | Python/JAX | Apache-2.0 | ~5 | Same as mahjax | Bug fixes over mahjax | Via jax.vmap | Low activity | Watch as mahjax alt |
| **lizhisim** (Apricot-S) | Rust | MIT | ~2 | N/A | WIP, not usable | N/A | Early dev | Watch (same author as xiangting) |
| **riichi-rs** (summivox) | Rust | LGPL-2.1 | ~5 | N/A | Partial | No | Stale | Skip (LGPL) |
| **rlcard** (datamllab) | Python | MIT | 3.4K+ | Slow (pure Python) | Simplified rules | No | Active | Skip (too simplified) |

---

## 3. Deep Dive: Top Candidates

### 3.1 RiichiEnv (smly/RiichiEnv) -- RECOMMENDED STARTING POINT

**What it is**: A Rust-core mahjong engine with Python bindings via maturin/PyO3. Made by smly, who also built mjai.app (the RiichiLab competition platform). This person knows mahjong engine correctness better than almost anyone.

**Why it is the best starting point**:

1. **Correctness is proven** -- verified against 1M+ hanchan with Mortal. That is not a typo. One million complete games where every single state transition was validated against Mortal's output. If your engine disagrees with Mortal on a game state, your engine is wrong, not Mortal.

2. **Full rule coverage** -- 4-player and 3-player (sanma). Red dora. All kan types (ankan, daiminkan, kakan). Chankan for kokushi. Furiten. Tenhou and MJSoul rule presets. Abortive draws. This is not some toy implementation.

3. **License is clean** -- Apache-2.0. You can fork it, modify it, ship it, commercialize it, whatever. No AGPL landmines.

4. **Already has the Gym API Hydra needs** -- `reset() -> obs`, `step(actions) -> obs`. Plus `VectorEnv` for batched operation. The API is literally what INFRASTRUCTURE.md specifies.

5. **Shanten calculation built-in** -- Uses Cryolite/nyanten lookup tables, both 4p and 3p.

6. **Hand evaluation built-in** -- Full yaku detection, han/fu scoring, payment calculation.

7. **MJAI protocol native** -- Events in/out as MJAI JSON. Mortal-compatible.

8. **Active development** -- The author is actively pushing commits in 2025-2026.

**What is missing (what Hydra would need to add)**:

- **rayon batch simulation** -- RiichiEnv does not have a pure Rust batch simulator that runs N games in parallel on a thread pool. Its VectorEnv is Python-side. Hydra needs a `Simulator::run_batch(n)` in Rust that uses rayon.
- **Observation encoder** -- RiichiEnv does not produce the 85x34 tensor Hydra needs. You get MJAI events and game state, but the CNN-ready observation encoding is Hydra-specific.
- **No GPU path** -- Pure CPU. For Phase 3 self-play at 512 concurrent games, this is fine (Mortal does the same). For pure data generation without NN inference, mahjax on GPU would be faster.

**Effort to adapt**: Medium. The engine is correct and complete. What you are adding is a Rust-level batch simulation layer and your own observation encoder on top of a proven foundation.

### 3.2 mahjax (nissymori/mahjax) -- FASTEST RAW THROUGHPUT

**What it is**: A full Riichi Mahjong environment written entirely in JAX. Every game operation -- tile shuffling, draw, discard, calls, scoring -- is a JAX operation that can be JIT-compiled and vectorized across thousands of games on GPU.

**Why it matters**:

1. **1.6M steps/sec on 8xA100** -- This is the highest published throughput number for any Riichi Mahjong simulator. For context, a "step" is one game action (draw, discard, call, etc.), and a typical hanchan has roughly 200-400 steps. That translates to roughly 4,000-8,000 complete games per second on 8xA100.

2. **JAX vmap vectorization** -- You write the game logic for ONE game, then `jax.vmap` automatically vectorizes it across a batch dimension. `jax.jit` compiles it. `jax.pmap` distributes across multiple GPUs. The entire game state lives on GPU memory -- no CPU-GPU transfers per step.

3. **PGX-style API** -- `init`, `step`, `observe` functions that are all JIT-compilable. Clean functional API.

4. **Inspired by PGX** -- Built by collaborators of the PGX project (NeurIPS 2023 paper). They know what they are doing with JAX game vectorization.

**What is missing / limitations**:

- **No red dora (aka-dora)** -- Currently supports standard 4-player rules WITHOUT red tiles. Red 5 tiles are on their roadmap but not implemented. This is a significant gap for Tenhou/MJSoul-style play where red dora are standard.
- **Rule edge cases unclear** -- The README does not specify whether all abortive draws, complex kan interactions, and multi-ron scenarios are handled. JAX requires all control flow to be expressible as `jax.lax.cond` and `jax.lax.switch` (no Python if/else in JIT), which makes complex branching game logic harder to get right.
- **No MJAI protocol** -- Pure numeric actions/states. No compatibility with Mortal or mjai.app evaluation infrastructure.
- **Young project** -- ~23 stars, no paper yet ("Paper coming soon"). The correctness is unproven at scale compared to RiichiEnv's 1M+ game verification.
- **JAX ecosystem lock-in** -- Your training stack must be JAX-based (not PyTorch) to get the GPU-resident benefit. If you use PyTorch for training and JAX for simulation, you are doing CPU-GPU-CPU-GPU transfers that kill the throughput advantage.

**Effort to adapt**: High if you want PyTorch training. The speed advantage comes from keeping everything on GPU in JAX. Switching to PyTorch training means the GPU vectorization benefit is mostly lost. To use mahjax properly, you would need to rewrite Hydra's training stack in JAX -- which is a massive change from the current PyTorch plan.

### 3.3 Mortal/libriichi (Equim-chan/Mortal) -- REFERENCE ONLY

**What it is**: The most battle-tested Riichi Mahjong engine in the open-source world. Powers Mortal (~7-dan play), mjai-reviewer, and the entire Mortal ecosystem.

**Why you cannot use it**: AGPL-3.0. If you link against libriichi or derive code from it, your entire project becomes AGPL. Hydra is MIT-licensed. AGPL is a hard no.

**Why you should study it**: libriichi is the gold standard for correctness. Its game state machine, scoring logic, and MJAI protocol handling are proven across millions of games. When building Hydra's engine (whether on RiichiEnv or from scratch), libriichi is the correctness oracle you validate against.

**Key architectural lessons from libriichi**:
- State machine is a flat enum, not nested
- Tile representation: 0-33 (34 unique tiles), with separate tracking of which physical copy (136 total)
- Action space: 46 actions (34 discards + riichi + tsumo + ron + 9 call types)
- Observation: 1012x34 tensor (much wider than Hydra's planned 85x34 because Mortal encodes differently)

### 3.4 Mjx (mjx-project/mjx) -- STUDY ONLY

**What it is**: C++ mahjong engine with Python bindings, gRPC distributed computing support. Made by the same research group as PGX (Koyamada et al., IEEE CoG 2022).

**Why it is interesting**: Claims 100x faster than the original Mjai Ruby server. Exact Tenhou compatibility validated against real game logs. Has a proper Gym API. MIT licensed.

**Why you should skip it**: The build is currently broken (their own README warns about this). The project appears semi-abandoned -- the team moved on to PGX and mahjax. Also, C++ adds complexity vs Rust for memory safety.

**Worth studying**: Their gRPC distributed architecture for running agents on separate machines. If Hydra ever needs distributed self-play across multiple nodes, Mjx's approach is a reference.

### 3.5 PGX (sotetsuk/pgx) -- ARCHITECTURE REFERENCE

**What it is**: JAX-native game simulator library covering 20+ games. NeurIPS 2023 paper. 10-100x faster than PettingZoo/OpenSpiel.

**Why it does not work for Hydra**: PGX only has "Sparrow Mahjong" -- a simplified, 3-player, 11-tile-type children's variant. This is NOT Riichi Mahjong. The full mahjong module exists in PGX's codebase but is not registered as an available environment (it appears to be a development branch that mahjax spun off from).

**Why you should study it**: PGX's architecture for JAX vectorization is the template that mahjax follows. The NeurIPS paper (arxiv:2303.17503) documents exactly how to vectorize game state, handle variable-length game trees, and scale across GPUs. If you ever build a JAX path for Hydra, start with this paper.

---

## 4. Speed Lessons from Other Domains

Before deciding an approach, here is what the fastest game simulators in other domains teach us:

### 4.1 The GPU Vectorization Pattern (JAX ecosystem)

**How it works**: You write your game logic as pure functions operating on JAX arrays. `jax.vmap` takes a function that works on a single game state and automatically creates a version that operates on a BATCH of game states in parallel on GPU. `jax.jit` compiles it to XLA, which optimizes the GPU kernel. `jax.pmap` distributes across multiple GPUs.

Think of it like this -- imagine you wrote a function `step(state, action) -> new_state` for one mahjong game. `vmap(step)` gives you `batch_step(states, actions) -> new_states` that runs 10,000 games simultaneously on a single GPU, with ZERO extra code from you. The GPU just sees it as one big matrix operation.

**Who does this**:
- **Brax** (Google): Physics simulation, millions of steps/sec for robot training
- **PGX**: Board games (Go, Chess, Shogi), 10-100x vs CPU baselines
- **Gymnax**: Classic RL envs (CartPole etc.), 1000x vs Gym on CPU
- **mahjax**: Riichi Mahjong, 1.6M steps/sec on 8xA100
- **EnvPool** (Sail-SG): C++ envs with async batch, 1M+ FPS for Atari

**The catch for mahjong**: JAX requires all control flow to be static (known at compile time) or expressed through `jax.lax.cond`/`jax.lax.switch`. Mahjong has complex branching -- after a discard, you need to check 3 opponents for ron, then check for chi/pon/kan, then resolve priority. This branching is expressible in JAX but painful to write and debug. Every `if/else` becomes a `jax.lax.cond` that evaluates BOTH branches and selects one, which wastes compute on the branch not taken.

### 4.2 The Rust + rayon Pattern (CPU parallelism)

**How it works**: You write your game engine in Rust. rayon gives you work-stealing parallelism -- you throw N games into a parallel iterator, and rayon distributes them across CPU cores. Each game runs independently, sequentially, on whatever core grabs it.

Think of it like a restaurant kitchen. You have 16 chefs (cores). 512 orders (games) come in. Each chef grabs an order, cooks it start to finish, then grabs the next one. If one chef finishes early, they steal work from a chef who is behind. No coordination needed -- each order is independent.

**Who does this**:
- **Mortal/libriichi**: Single-game sequential, no batch (but fast per game)
- **RiichiEnv**: Rust core, Python VectorEnv wrapper
- **Hydra's INFRASTRUCTURE.md plan**: rayon batch simulation targeting 100K+ games/hr/core

**The advantage for mahjong**: Normal Rust code. Normal if/else. Normal pattern matching. The complex branching logic of mahjong is natural to write. Debugging is straightforward. Correctness verification is easy. And Rust is still extremely fast -- 100K+ games/hour/core means a 16-core machine does 1.6M games/hour.

### 4.3 Throughput Math: GPU vs CPU for Hydra's Use Case

Hydra's Phase 3 self-play bottleneck is NOT the game simulation -- it is the neural network inference. Each game step requires a forward pass through a 40-block SE-ResNet to select an action. At ~0.5-1ms per batch-512 inference, the GPU is the bottleneck.

So the question is: does GPU-vectorized game simulation help when the GPU is already busy doing inference?

**Scenario A: Rust CPU sim + GPU inference (Hydra's current plan)**
- CPU runs 512 games via rayon, generates observations
- Observations batched and sent to GPU for inference
- GPU returns actions, CPU advances games
- Target: 10,000+ games/hour end-to-end (INFRASTRUCTURE.md)
- GPU utilization: ~60-80% (inference), CPU handles game logic

**Scenario B: JAX GPU sim + JAX GPU inference (mahjax path)**
- Everything on GPU: game sim AND inference
- Zero CPU-GPU transfers
- But: game sim and inference compete for GPU compute
- Risk: complex mahjong branching wastes GPU cycles on dead branches
- Benefit: if your model is small, the GPU has spare capacity for sim

**Bottom line**: For Hydra's 40-block SE-ResNet (large model), Scenario A is likely better. The GPU is already saturated with inference. Making it also do game simulation means contention. The CPU is sitting there with spare cycles -- let it do what CPUs are good at (branching game logic) while the GPU does what GPUs are good at (matrix multiplication for inference).

If Hydra ever moves to a smaller model or wants to do pure data generation WITHOUT inference (like pre-generating random games for curriculum), then the JAX path becomes compelling.

---

## 5. Recommended Approach: The Build Plan

### Phase 0: Fork RiichiEnv (Week 1)

**What**: Fork smly/RiichiEnv into Hydra's repository as the foundation.

**Why RiichiEnv over building from scratch**:
- Saves 2-4 months of engine development
- 1M+ game correctness verification already done
- Full rule coverage already implemented
- Apache-2.0 license is clean
- Author (smly) is the RiichiLab platform maintainer -- the person who runs the competition that Hydra wants to enter

**What to fork**:
- The `riichienv-core/` Rust crate (game engine, state machine, scoring, shanten)
- The PyO3 bindings layer
- The hand evaluator and tile conversion utilities

**What NOT to fork**:
- The UI/visualization (not needed for training)
- The Jupyter notebook viewer
- The demo notebooks

### Phase 1: Add Hydra-Specific Layers (Weeks 2-4)

These are the pieces that RiichiEnv does not have that INFRASTRUCTURE.md requires:

**1. Observation Encoder (encoder.rs)**
- Build the 85x34 tensor encoder on top of RiichiEnv's game state
- This is Hydra-specific -- the 23 safety planes, temporal weights, etc.
- RiichiEnv gives you the raw game state; you transform it into the neural net input
- Pre-allocated buffers, contiguous memory layout per INFRASTRUCTURE.md spec

**2. Batch Simulator (simulator.rs)**
- Wrap RiichiEnv's single-game engine in a rayon parallel iterator
- `Simulator::run_batch(n, seed)` -> runs N complete hanchans in parallel
- Each game gets a deterministic seed per SEEDING.md's hierarchy
- Target: 100,000+ games/hour per core for pure simulation (no NN)

**3. PyO3 Bridge Updates (python.rs)**
- Expose `simulate_batch(n)` to Python
- Expose the observation encoder to Python
- Ensure the GIL is released during Rust computation
- Wire up to Hydra's `MahjongEnv` and `VectorEnv` interfaces

**4. MJAI Protocol Adapter**
- RiichiEnv already speaks MJAI natively
- Wire Hydra's action space (46 actions) to RiichiEnv's action types
- Ensure Mortal compatibility for evaluation

### Phase 2: Performance Optimization (Weeks 5-6)

Once the foundation works correctly:

1. **Benchmark baseline throughput** -- How many games/hour/core does the naive rayon wrapper achieve?
2. **Profile hotspots** -- flamegraph the Rust code. Where is time spent? Scoring? Shanten? State transitions?
3. **Incremental observation updates** -- Instead of recomputing the full 85x34 tensor every step, track deltas (hand +/-1 tile, discards +1). INFRASTRUCTURE.md notes this as a planned optimization.
4. **Memory layout optimization** -- Ensure game state structs are cache-line aligned. Use `#[repr(C)]` where needed for predictable layout.
5. **Pre-allocated game pools** -- Instead of allocating/deallocating games, maintain a pool of pre-allocated game states that get recycled.

### Phase 3: Validation (Week 7)

1. **Replay verification** -- Parse real Tenhou and Majsoul game logs through the engine, verify final states match
2. **Golden tensor tests** -- 20+ hand-crafted game states with pre-computed expected observation tensors (per TESTING.md)
3. **Cross-reference with Mortal** -- Run identical seeds through both engines (via RiichiEnv's Mortal compatibility), verify state agreement
4. **Throughput benchmarks** -- Measure and document games/hour/core, compare against targets

---

## 6. Stretch Goal: JAX Fast Path (Future)

If Hydra ever needs 10x more throughput than Rust CPU can deliver:

1. Study mahjax's JAX implementation for vectorization patterns
2. Build a JAX-native "lite" simulator that handles the common case (no exotic kan, no abortive draws)
3. Use it ONLY for self-play data generation where the occasional rule edge case is acceptable
4. Keep the Rust engine as the authoritative reference for evaluation and correctness

This is explicitly NOT recommended for Phase 1. Get the Rust engine working and correct first. Speed without correctness is worse than useless in RL training -- corrupted game states silently corrupt your training data.

---

## 7. Links and Resources

| Resource | URL | Why |
|----------|-----|-----|
| RiichiEnv | https://github.com/smly/RiichiEnv | Fork target |
| mahjax | https://github.com/nissymori/mahjax | GPU vectorization reference |
| mahjax-fixed | https://github.com/hnsqdtt/mahjax-fixed | Bug-fixed mahjax fork |
| PGX | https://github.com/sotetsuk/pgx | JAX game sim architecture reference |
| PGX NeurIPS paper | https://arxiv.org/abs/2303.17503 | How to vectorize game state on GPU |
| Mortal | https://github.com/Equim-chan/Mortal | Correctness oracle (AGPL, reference only) |
| Mjx | https://github.com/mjx-project/mjx | Distributed sim architecture reference |
| Mjx IEEE CoG paper | https://ieee-cog.org/2022/assets/papers/paper_162.pdf | Fast mahjong sim design |
| lizhisim | https://github.com/Apricot-S/lizhisim | Watch (MIT Rust sim, same author as xiangting) |
| xiangting | https://github.com/Apricot-S/xiangting | Shanten library (already selected by Hydra) |
| agari | https://github.com/rysb-dev/agari | Rust scoring reference |
| mahc | https://github.com/DrCheeseFace/mahc | Rust scoring reference (Fu enum pattern) |
| Hydra INFRASTRUCTURE.md | research/INFRASTRUCTURE.md | Hydra's own infra spec |
| Hydra TESTING.md | research/TESTING.md | Hydra's testing strategy |
| Hydra SEEDING.md | research/SEEDING.md | RNG reproducibility spec |
