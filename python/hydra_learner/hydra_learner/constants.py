from __future__ import annotations

VARIANTS = ("eager_fp32", "eager_bf16", "compile_default", "compile_reduce_overhead", "compile_max_autotune")
PYTHON_VARIANT_DEFAULT = "compile_max_autotune"
LOSS_MODES = ("policy_only", "full_base")
COMPILED_LOSS_MODES = ("policy_only", "full_base")
ADAMW_FLAG_MODES = ("auto", "on", "off")
LR_SCHEDULES = ("constant", "cosine")
VALIDATION_SOURCE_MODES = ("fixed", "streaming")


WARMUP_MODE = "non_mutating_replay_first_batch"
COMPILE_DRY_RUN_MODE = "snapshot_restore_first_batch"
