# Hydra v3 notes

Hydra v3 trains a PyTorch tile-edit Transformer with four-player red-five hanchan self-play. The rollout and paired-evaluation rules run in MahJax. The native Rust arena remains a separate reference implementation. No trained weights or model-strength claim are included.

Hydra3 pins MahJax 0.3.3 to revision `d4ca79a49676232f48d275d746d7b26dc50a534d` and uses its `red_4p` half-game preset without placement bonuses. This preset is not certified as interchangeable with RiichiEnv's Tenhou preset. Paired evaluation repeats each wall stream with the candidate in all four seats; walls are keyed by deal number rather than action count.

Use Pixi 0.84.2 or newer, Git 2.49 or newer, and stable Rust. On x86-64 Linux, install Clang, mold, and sccache on `PATH`. The first build fetches the pinned canonical action vocabulary; later builds can run offline.

```sh
pixi run build
pixi run test
pixi run typecheck && pixi run lint
pixi run agent --help
```

`pixi run build` builds the native reference extension; `pixi run test` rebuilds it for the test suite. `pixi run agent` does not build or load the native arena.

PPO and paired evaluation default to CUDA; `--device cpu` is available for diagnostics. MahJax actor wire 3 requires new checkpoints and corpora.

PPO's `--telemetry` separates host wall time spent on arena packet retrieval, actor submission, arena step submission, and rollout archiving. These are not GPU kernel timings: asynchronous work can be charged to the call that waits for it.

The vendored rules engine checks tenpai (one tile short of a winning hand) by looking up each suit once instead of constructing a hand for every possible draw after every discard. The lookup is derived from the pinned engine's completion table; seven pairs and thirteen orphans are checked separately. The actor encoder indexes public discards and calls by their history positions and selects up to four visible copy records per tile face. Neither change alters the checkpoint or corpus format.

The collector writes bfloat16 tokens and legal action descriptors directly into a device-resident archive. If a growth allocation fails, it spills the archive to host memory and reports `rollout_archive_on_device=0`. With `--record`, decisions are grouped by game and tokens are exported as float32.

On the reference RTX 5070, the default policy sustained **90.2k self-play decisions/s during collection and 37.8k decisions/s including learning and checkpoints**. Eight consecutive generations after one warmup completed 8,192 hanchan and 8,474,414 decisions in 224.2 seconds. Comparing the first four warm generations on both builds, collection rose from 68.4k to 92.9k decisions/s and end-to-end throughput from 33.9k to 38.2k. A separate fixed-policy replay matched the original observations, legal actions, sampled actions, behaviour probabilities, values and final scores across 1,024 complete hanchan. See the [benchmark record](benchmarks/mahjax/rtx-5070-self-play.json) for timing boundaries and hashes.

PPO defaults to 1,024 games, a 1,024-decision minibatch, one epoch, fused AdamW, a compiled learner loss and CUDA graph replay for frequent actor shapes. The model and its precision are unchanged. In earlier capacity tests on this card, 1,536 games exhausted memory in the policy forward and 2,048 failed during XLA autotuning; neither is a verified faster configuration.

The rules engine is vendored under `vendor/mahjax` so it can be worked on in-tree. `python/hydra3/engine.py` puts that copy first on `sys.path` unless `HYDRA3_ENGINE=installed` asks for the pinned distribution instead. `tests/engine_parity.py` replays fixed seeds and actions against `tests/fixtures/engine_trace.json`, a whole-state digest trace recorded from the pinned distribution, so an engine change either reproduces the pin exactly or fails. Re-record it only when engine behaviour is meant to change.
