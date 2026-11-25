# Self-play performance contract

The training metric is **decisions per wall-clock second from collection through checkpointing**, not actor forwards per second. Count every decision in complete four-player hanchan, keep the same policy capacity, expose the same legal actions and perform the same PPO training dose. A faster run that learns a worse policy is not a replacement for the default.

## Reference configuration

On the RTX 5070 (12 GB), the qualified path uses 1,024 games per generation, a 1,024-decision minibatch, one epoch, one Torch CPU thread, automatic attention selection, fused AdamW, and a compiled learner. The actor uses eager bfloat16 scoring with CUDA graph replay, **not** the experimental compiled scorer. Replaying an eager graph changes how its kernels are launched; compiling the scorer can change its arithmetic. Pixi's `agent` task disables JAX device preallocation so that JAX and Torch can share the card.

From the project root, `pixi run --locked agent ppo @ppo.local.args INITIAL.pt OUTPUT.pt` selects this workstation's ignored profile. It contains one CLI argument per line. There is no automatic profile discovery: omit `@ppo.local.args` for portable defaults, or place an option after it to override it. Checkpoint paths, the run seed and the number of generations are run inputs, not machine settings.

## What to preserve in the hot path

- **Tile lookup:** gather from each encoded state directly. Expanding a state across every candidate looks cheap in a forward-only benchmark because the expansion is a view. Its backward pass can allocate and reduce a candidate-sized gradient buffer. Measure complete PPO updates and check gradient accumulation for repeated tiles and masked slots.
- **Collection:** keep the arena on the device, exchange batched tensors, and write each turn directly into the bfloat16-token/int32-descriptor archive. Buffer growth must not drop decisions. A host spill is a supported recovery path, not an equivalent GPU performance result: inspect `rollout_archive_on_device`.
- **Actor replay:** capture only a frozen policy, not sampling or rules. Do not reuse graphs across weight updates. Archive values must survive the next replay. Execution buckets may pad computation, never remove a legal action or truncate a match. Check wide legal sets as well as common shapes.

## Requalifying a change

1. On an otherwise idle GPU, use the same device, driver, power limit, locked environment, initial checkpoint, seed and settings for both arms. Create a checkpoint once with `pixi run --locked agent init runs/perf/initial.pt --seed 1` if needed. Before and after the change, run the following with distinct output and log names. The example is the control; use `candidate.pt` and `candidate.jsonl` for the second run. On another checkout without a local profile, omit the `@` argument and specify the reference settings above.

   ```sh
   pixi run --locked agent ppo @ppo.local.args runs/perf/initial.pt runs/perf/control.pt \
     --generations 9 --seed 1 --telemetry > runs/perf/control.jsonl
   ```

2. Exclude generation 0 (warmup), then include all eight consecutive generations, their training updates, inter-generation gaps and checkpoint writes. Divide **total actual decisions** by the time from the first included `collect_started_ns` through the final `learn_started_ns + learn_ns + checkpoint_ns`; do not average the per-generation rates. Report games, decisions, collection and learner rates, archive residency and the end-to-end rate. The [RTX 5070 scorer record](../../benchmarks/mahjax/rtx-5070-move-scorer.json) measured 41,928 decisions/s on the default path. That is a reference for a paired comparison, not a hardware-independent CI threshold.

3. Run `pixi run --locked lint`, `format-check`, `typecheck` and `test` through Pixi. `test` runs the CPU lane in four parallel workers and then the `gpu`-marked lane in one serial process, because the whole-match and compiled-kernel tests need the device alone; `test-cpu` and `test-gpu` run either lane by itself. Compiled kernels are cached under the ignored `.pixi/` directory, so repeated runs skip recompilation. If changing the engine, also run `pixi run --locked python tests/engine_parity.py`; do not regenerate its oracle to make a performance patch pass. Compare real actor-visible decisions for identical legal IDs and masks, finite scores on legal moves and zero probability on padding. Numerical or sampling differences do not require bit-for-bit trajectories, but they do require a playing-strength check.

4. For changes to scoring or learning, train both policies and evaluate final placement on the **same held-out walls**, each focal policy in all four seats against the same frozen opponent. Pair differences by wall, average its four seats, and bootstrap whole walls rather than individual games. Separate a same-weights scorer comparison from the trained-policy comparison. The recorded gate used 4,096 walls and required the upper bound of the two-sided 95% interval for the increase in mean rank to be below **+0.03** (lower rank is better). The compiled scorer gained 10.6% end to end but missed that gate: **+0.197 places** (95% interval **+0.174 to +0.221**). Keep it opt-in unless new evidence supports promotion; one training seed and opponent do not establish general playing strength.
