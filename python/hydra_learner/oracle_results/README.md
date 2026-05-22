# Oracle results

Preserved JSON results from `/home/cachybtw/dev/hydra-pytorch-oracle` probe and `/home/cachybtw/tmp` artifacts before deleting temporary worktree.

Original CUDA 13 nightly probe:

| Variant | Samples/s | Step ms | Compile s | Max alloc |
|---|---:|---:|---:|---:|
| eager FP32 | 12,592 | 162.64 | 0 | 3.405 GB |
| eager BF16 | 17,460 | 117.30 | 0 | 2.774 GB |
| compile default | 28,712 | 71.33 | 1.38 | 1.435 GB |
| compile reduce-overhead | 28,414 | 72.08 | 1.40 | 1.401 GB |
| compile backend=cudagraphs | 17,572 | 116.55 | 1.80 | 2.718 GB |
| manual CUDA graph | failed | - | - | - |

Main repo CUDA 12.8 smoke uses current Pixi torch `2.9.0+cu128`; see `/home/cachybtw/tmp/hydra_py_migration_smoke.json` and `/home/cachybtw/tmp/hydra_py_migration_compile_smoke.json`.
