"""Run-config section records: frozen YAML sections plus resume state.

Owns the fourteen validated section dataclasses bound into the frozen
:class:`RunConfig` surface, the vocabulary constants every parser below
validates against, and the resumable stream-reader state records
(:class:`StreamCursor` plus shuffle/accum/worker/resume plans) that the
digest, layout, and resume paths consume.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hydra2.contracts.common import ContractError

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "CONFIG_SECTIONS",
    "INTERPOLATION_ALLOWLIST",
    "RUN_KINDS",
    "_ADAPTER_IDS",
    "_COMPILE_MODES",
    "_CUDA_DEVICE_RE",
    "_INTERPOLATION_RE",
    "_OPTIMIZER_IDS",
    "_RUNTIME_PRECISIONS",
    "_RUN_ID_RE",
    "_SCHEDULER_IDS",
    "_SELECTION_DESIGNS",
    "_TENHOU_SCOPE",
    "AccumState",
    "DataConfig",
    "EvalConfig",
    "LoopConfig",
    "MirrorConfig",
    "ModelConfig",
    "OptimizerConfig",
    "OutputConfig",
    "ResumePlan",
    "RunConfig",
    "RunMeta",
    "RuntimeConfig",
    "SchedulerConfig",
    "SeedsConfig",
    "SelectionConfig",
    "ShuffleState",
    "StreamCursor",
    "TelemetryConfig",
    "WeightsConfig",
    "WorkerPlan",
]

#: The fourteen RunSpec YAML sections, in canonical order. Unknown top-level
#: keys are rejected (strict loader).
CONFIG_SECTIONS: tuple[str, ...] = (
    "run",
    "data",
    "model",
    "weights",
    "optimizer",
    "scheduler",
    "runtime",
    "loop",
    "seeds",
    "selection",
    "mirror",
    "telemetry",
    "eval",
    "output",
)

#: Run kinds accepted by the streaming-first v1 surface. ``distill`` / ``rl``
#: are deferred to later waves (recorded, not silent-picked): requesting
#: them raises :class:`ContractError`.
RUN_KINDS: tuple[str, ...] = ("supervised",)

#: Environment variables allowed inside ``${VAR}`` interpolation. Mounts and
#: roots arrive via config/env only; anything outside this list (secrets,
#: tokens, ad-hoc paths) raises instead of interpolating.
INTERPOLATION_ALLOWLIST: frozenset[str] = frozenset(
    {
        "HYDRA2_ARTIFACT_ROOT",
        "HYDRA2_DATA_ROOT",
        "HOME",
        "XDG_CACHE_HOME",
    }
)

_INTERPOLATION_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_CUDA_DEVICE_RE = re.compile(r"^cuda(:[0-9]+)?$")
_TENHOU_SCOPE = "tenhou-4p-hanchan"

_OPTIMIZER_IDS: tuple[str, ...] = ("adamw", "adam", "sgd")
_SCHEDULER_IDS: tuple[str, ...] = ("cosine", "constant", "linear")
_ADAPTER_IDS: tuple[str, ...] = ("plain_pytorch",)
_RUNTIME_PRECISIONS: tuple[str, ...] = ("fp32", "bf16_mixed")
_COMPILE_MODES: tuple[str, ...] = (
    "eager",
    "default",
    "max-autotune-no-cudagraphs",
    "max-autotune",
)
_SELECTION_DESIGNS: tuple[str, ...] = ("fixed_n", "time_uniform_cs")


# ---------------------------------------------------------------------------
# Section dataclasses (all frozen; RunConfig is the single validated surface)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunMeta:
    """``run`` section: identity and kind."""

    run_id: str = ""
    kind: str = "supervised"
    description: str = ""


@dataclass(frozen=True, slots=True)
class DataConfig:
    """``data`` section: streaming source root plus contract flags.

    ``root`` MUST interpolate to an absolute path (via ``${HYDRA2_DATA_ROOT}``;
    never a literal mount). ``scope`` is Tenhou-only in v1. The contract
    flags mirror the ephemeral-row obligations: wall-disjoint manifests,
    leakage checks, ``(5,)`` dora, and the actor/privileged split. Any flag
    that weakens them fails closed.
    ``dataset_manifest_hash`` is an optional provenance pin: when set, the
    stream builder MUST verify the stream manifest matches it before the
    first game is consumed (v1 ``manifest_input`` provenance, mechanism
    replaced by in-memory manifest hashes plus the validation hash — no
    shard files). v1's manifest-vs-raw-dirs exclusivity is parked: the
    supervised stream root is the single v1 source; the rollout side
    arrives with ``run.kind=rl`` (deferred, top-level ``rl`` must be null).
    Stream mechanics (ResumeExactPlan handoff): ``shuffle_buffer_size`` pins
    the shuffle buffer capacity (buffer persists verbatim as row KEYS, never
    refilled HF-style); 0 disables shuffling (stream order). ``drop_last``
    pins tail behavior; ``num_workers`` / ``world_size`` pin the worker plan
    hard error). Shuffle order derives from ``seeds.data_seed`` with
    ``epoch_seed = data_seed + epoch``; the stream builder owns the draw,
    this config owns the pins.
    """

    root: str = ""
    scope: str = _TENHOU_SCOPE
    train_split: str = "train"
    val_split: str = "validation"
    wall_disjoint: bool = True
    leakage_check: bool = True
    dora_width: int = 5
    actor_privileged_split: bool = True
    dataset_manifest_hash: str | None = None
    shuffle_buffer_size: int = 10000
    drop_last: bool = True
    num_workers: int = 0
    world_size: int = 1
    #: Replay backend: ``"rust"`` (Rust per-game walk + tensor-native batch assembly over the
    #: identical stream: same order/shuffle/resume/sidecar, same rows, no
    #: per-row Python) or ``"python"`` (oracle shim, byte-identical rows, parity only).
    #: The value is part of the run digest, so a
    #: changed backend refuses resume fail-closed (digest mismatch, as does
    #: the dataset buffer backend tag).
    replay_backend: str = "rust"
    #: Decode lookahead: games held in the background decode pool
    #: (``PrefetchGameStream`` in ``src/hydra2/data/stream.py``). Must cover
    #: one full expansion batch; larger smooths variable-rows-per-game burst.
    decode_prefetch: int = 64
    #: Expansion batch: games pulled per fill round and sharded over the
    #: spawn workers (``_fill_parallel`` in ``src/hydra2/training/stream_train.py``,
    #: one bridge FFI per chunk via ``expand_game_batch``).
    expand_batch_games: int = 64
    #: Stable bucket-grouped takes: when true, ``_StreamDataset`` stably
    #: partitions the unconsumed window by history bucket before each
    #: contiguous-prefix take (one bucket per microbatch, less pad waste).
    #: Default False is byte-identical legacy order. Part of the run digest:
    #: flipping it churns the digest, so a stale checkpoint fails closed on
    #: drift instead of resuming on reordered rows (same precedent as
    #: ``fetch_prefetch_depth``).
    homogeneous_buckets: bool = False


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """``model`` section: architecture identity plus frozen action width."""

    architecture_id: str = "hydra2_baseline_transformer_v1"
    action_count: int = 6792
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class WeightsConfig:
    """``weights`` section: objective weights (explicit, zero may be absent).

    Zero or absent disables a head. Any positive ``w_event`` / ``w_belief``
    entry REQUIRES ``privileged_source_hash`` (positive-requires-binding):
    privileged labels MUST come from the pinned privileged manifest, joined
    by opaque decision id only (v1 sidecar provenance, mechanism replaced by
    :class:`PrivilegedLabelStore` guards — no sidecar paths). Unknown weight
    keys (v1 ``belief_fields`` / mixture / opponent-hand patterns) are
    rejected as unknown keys, never read as weights.
    """

    w_policy: float = 1.0
    w_placement: float = 0.0
    w_value: float = 0.0
    w_event: dict[str, float] | None = None
    w_belief: dict[str, float] | None = None
    privileged_source_hash: str | None = None
    #: Legal-only label-smoothing mass ``eps`` in ``[0, 1)`` (default
    #: ``0.03``, the SOTA-quoted constant; ``0.0`` disables to byte-identical
    #: plain masked CE).  The ``eps`` mass spreads over the LEGAL set only
    #: (illegal mass stays exactly zero).
    label_smoothing: float = 0.03


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """``optimizer`` section: registered optimizer id plus parameters."""

    name: str = "adamw"
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.01
    #: Policy-head LR multiplier (SOTA dominant-head precedent; the policy
    #: head trains at ``lr * head_lr_mult`` with ``weight_decay`` forced to
    #: ``0.0``).  Only ``policy_head.`` parameters take the head LR; the
    #: placement/value/event/belief heads stay in the trunk groups at the
    #: base LR by design (policy-only head LR: the policy head dominates the
    #: update and benefits from the larger step, while auxiliary heads train
    #: alongside the trunk they read out from).  Trunk parameters split into
    #: decay (``weight_decay``) vs no-decay (``0.0``) groups in
    #: ``stream_train._optimizer_param_groups``.
    head_lr_mult: float = 3.0


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """``scheduler`` section: registered scheduler id plus parameters."""

    name: str = "cosine"
    warmup_updates: int = 100
    parameters: dict[str, Any] = field(default_factory=dict)
    #: Final LR as a fraction of peak (Mortal
    #: ``LinearWarmUpCosineAnnealingLR`` ``final`` mirror): ``0.0``
    #: (default) decays to zero, ``1.0`` holds peak.  Must lie in ``[0, 1]``.
    final_factor: float = 0.0
    #: Warmup init LR as a fraction of peak (current ``LinearLR``
    #: ``start_factor`` ``0.01`` preserved as the default).  Must lie in
    #: ``(0, 1]``.
    warmup_start_factor: float = 0.01


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """``runtime`` section: mirrors :class:`RuntimeSpec` plus loop coherence.

    ``fp16_mixed`` is rejected by design (the loop owns bf16-only autocast;
    see :class:`TrainingLoopConfig`); the runtime gate
    :func:`validate_runtime_spec` remains authoritative at bind time.
    """

    adapter_id: str = "plain_pytorch"
    device: str = "cuda"
    precision: str = "fp32"
    compile_mode: str = "eager"


@dataclass(frozen=True, slots=True)
class LoopConfig:
    """``loop`` section: microbatching, clipping, horizon, checkpoint cadence.

    ``optimizer_minibatch_size`` derives as ``microbatch_size`` times
    ``accumulation_steps`` (same derivation as :class:`TrainingLoopConfig`;
    v1 ``batch`` maps to it, v1 ``micro`` maps to ``microbatch_size``).
    Checkpoints land on update boundaries only; a partial accumulation
    window is re-run after resume, never half-applied. ``keep_last_checkpoints``
    bounds retained ``ckpt-<update>.pt`` files (null keeps all).
    """

    microbatch_size: int = 4
    accumulation_steps: int = 8
    gradient_clip_norm: float | None = 1.0
    max_updates: int = 10000
    checkpoint_frequency_updates: int = 500
    precision: str = "fp32"
    keep_last_checkpoints: int | None = None
    #: Stratified rare-action sampling (SOTA R5): when ``True`` the sampler
    #: oversamples rare kinds (kan/ron families) per ``sampling_ratios`` so
    #: every optimizer minibatch sees them.  ``False`` (default) preserves
    #: the canonical cursor order bit-identically.
    stratified_sampling: bool = False
    #: Per-kind oversample multipliers, e.g.
    #: ``{daiminkan: 3.0, ankan: 3.0, kakan: 3.0, ron: 3.0}`` (starter
    #: ``2``-``4``x).  ``None`` (default) means uniform (no oversampling).
    #: Values MUST be positive and finite; keys MUST be non-empty strings.
    sampling_ratios: dict[str, float] | None = None
    #: Fetch lookahead: microbatches held in the loop prefetch queue
    #: (``SupervisedLoop.train`` in ``src/hydra2/training/loop.py``). Must
    #: cover a fill burst behind compute windows (fetch/compute ratio sized).
    fetch_prefetch_depth: int = 3
    #: Log flattened per-type ``per_type/<kind>/{n,nll,top1,top3,ece,recall,low_support}``
    #: scorecards in train history and eval reports (default ``True``).
    log_per_type_metrics: bool = True
    #: Fit post-hoc temperature on eval reports (default ``True``).
    #: Validation-only, never touches training.
    fit_temperature: bool = True

    @property
    def optimizer_minibatch_size(self) -> int:
        return self.microbatch_size * self.accumulation_steps


@dataclass(frozen=True, slots=True)
class SeedsConfig:
    """``seeds`` section: counter-based semantic seeds (never wall-clock).

    Same seeds plus the same :class:`StreamCursor` give the same sequence;
    the stream builder derives counter streams from these roots.
    """

    data_seed: int = 0
    train_seed: int = 0
    selection_seed: int = 0


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    """``selection`` section: frozen pre-result selection parameters.

    Mirrors :class:`eval.statistics.SelectionConfig` field-for-field; that
    class remains the authoritative gate at selection time. Frozen before
    any result exists: training NEVER calls selection, and selection
    outcomes affect only best-checkpoint promotion (the ``best-ckpt.pt``
    copy), never resume writes or the optimizer trajectory.
    """

    N: int = 30
    pilot_s: float = 1.5
    delta: float = 0.5
    alpha: float = 0.05
    beta: float = 0.2
    design: str = "fixed_n"
    declared_peeks: tuple[int, ...] = (30,)
    margin: float = 0.0
    resamples: int = 2000
    seed: int = 0


@dataclass(frozen=True, slots=True)
class MirrorConfig:
    """``mirror`` section: observer-only ClearML mirror flags.

    Local artifacts stay authoritative; the mirror never feeds values back.
    ``HYDRA2_CLEARML_DISABLED=1`` is the kill-switch and wins over
    ``enabled: true`` (see :mod:`hydra2.tracking.clearml_mirror`).
    """

    enabled: bool = False
    project: str = "hydra2-tenhou-4p"
    task_name: str | None = None
    offline_dir: str | None = None


@dataclass(frozen=True, slots=True)
class TelemetryConfig:
    """``telemetry`` section: observer-only MLflow + verbose sampler flags.

    Local artifacts stay authoritative; telemetry never feeds values back.
    ``mlflow_enabled`` defaults on (quiet per-update scalars into the
    per-artifact-root SQLite store); ``HYDRA2_MLFLOW_DISABLED=1`` is the
    kill-switch and wins over ``true``. ``verbose_enabled`` defaults off
    (20-50ms NVML/psutil rows into ``logs/verbose-telemetry.jsonl``).
    ``profiler_captures`` arms that many single-update torch.profiler
    captures per run (0 disables; CUDA-only, traces under ``profiler/``).
    """

    mlflow_enabled: bool = True
    verbose_enabled: bool = False
    verbose_interval_ms: int = 50
    profiler_captures: int = 0


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """``eval`` section: duplicate-wall evaluation cadence.

    Evaluation walls never enter replay; the eval builder enforces that.
    ``microbatch_size`` null means the loop microbatch (v1 ``val-micro``
    gap default); when set it must be positive.
    """

    frequency_updates: int = 500
    num_batches: int = 10
    walls_dir: str | None = None
    microbatch_size: int | None = None


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """``output`` section: artifact-root override and run-id override.

    Both default to ``None`` (honor ``HYDRA2_ARTIFACT_ROOT`` via
    :func:`hydra2.config.artifact_root` and ``run.run_id`` respectively).
    Everything this module writes stays under the resolved artifact root.
    """

    artifact_root: str | None = None
    run_id: str | None = None


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Frozen, fully-validated training run configuration (YAML authority)."""

    run: RunMeta = field(default_factory=RunMeta)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    weights: WeightsConfig = field(default_factory=WeightsConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    seeds: SeedsConfig = field(default_factory=SeedsConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    mirror: MirrorConfig = field(default_factory=MirrorConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


@dataclass(frozen=True, slots=True)
class StreamCursor:
    """Resumable stream position (shared interface contract, all builders).

    ``file_index`` orders files deterministically (sha256 hex order —
    same FNV1a-equivalent property as v1: a total order independent of
    filesystem readdir order). ``byte_offset`` resumes inside the current
    file; ``games_seen`` counts fully-consumed games for catch-up;
    ``seed`` / ``epoch`` bind the deterministic order for this pass.
    """

    file_index: int = 0
    byte_offset: int = 0
    games_seen: int = 0
    seed: int = 0
    epoch: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "file_index": self.file_index,
            "byte_offset": self.byte_offset,
            "games_seen": self.games_seen,
            "seed": self.seed,
            "epoch": self.epoch,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> StreamCursor:
        if not isinstance(raw, dict):
            raise ContractError(f"stream cursor must be a mapping, got {type(raw).__name__}")
        expected = ("file_index", "byte_offset", "games_seen", "seed", "epoch")
        unknown = sorted(k for k in raw if k not in expected)
        missing = [k for k in expected if k not in raw]
        if len(unknown) > 0 or len(missing) > 0:
            raise ContractError(
                f"stream cursor envelope mismatch; missing={missing} unknown={unknown}"
            )
        values: dict[str, int] = {}
        for key in expected:
            value = raw[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractError(f"stream cursor {key!r} must be a non-negative int")
            values[key] = value
        return cls(
            file_index=values["file_index"],
            byte_offset=values["byte_offset"],
            games_seen=values["games_seen"],
            seed=values["seed"],
            epoch=values["epoch"],
        )


@dataclass(frozen=True, slots=True)
class ShuffleState:
    """Persisted shuffle-buffer state (ResumeExactPlan P1; KEYS, not rows).

    The buffer is re-materialized verbatim at resume (HF-style refill is
    REJECTED): ``buffer_keys`` holds the ordered buffered row keys —
    ``O(buffer)`` keys, never rows. ``epoch_seed`` is ``data_seed + epoch``;
    ``buffer_rng_state`` is the opaque buffer-RNG JSON value the stream
    builder restores. ``prefix_hashes`` is the ``{file, count, sha256}``
    record for the sidecar-adjacent dedup-prefix file (``{}`` when the
    checkpoint predates prefix persistence).
    """

    buffer_keys: tuple[str, ...] = ()
    epoch_seed: int = 0
    buffer_size: int = 0
    buffer_rng_state: dict[str, Any] = field(default_factory=dict)
    prefix_hashes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AccumState:
    """Persisted accumulation-window state (ResumeExactPlan P8).

    Checkpoints land on update boundaries; a partial window (``micro_in_update``
    nonzero) is re-run after resume, never half-applied. ``yielded_batches``
    counts fully-yielded optimizer batches in this run.
    """

    yielded_batches: int = 0
    micro_in_update: int = 0


@dataclass(frozen=True, slots=True)
class WorkerPlan:
    """Persisted loader worker plan (ResumeExactPlan P7).

    ``num_workers`` / ``world_size`` MUST match the run config at resume;
    ``rank`` MUST match the launching rank. Any mismatch is a hard error
    before any state is restored.
    """

    num_workers: int = 0
    world_size: int = 1
    rank: int = 0


@dataclass(frozen=True, slots=True)
class ResumePlan:
    """Resolved resume point: latest valid checkpoint + cursor + RNG.

    The resolver only reads (verify-before-mutate): checkpoint payload
    identity is verified by :mod:`hydra2.runtime.checkpoint` at bind time,
    before any runtime object is touched. ``rng_state`` is the opaque RNG
    bundle from the sidecar (torch CPU + CUDA + python + numpy + semantic
    stream states). ``run_digest`` MUST equal the digest of ``run.yaml`` in
    ``run_dir`` (resume compat digest). A mid-game ``byte_offset`` is
    corrupt: offsets MUST sit on game boundaries (the stream reader owns
    the boundary check at ``skip_to`` time); partial-epoch resume requires
    the identical contract, only epoch-boundary loader shape may relax.
    """

    run_dir: Path
    checkpoint: Path
    cursor: StreamCursor
    global_update: int
    rng_state: dict[str, Any]
    run_digest: str = ""
    shuffle: ShuffleState = field(default_factory=ShuffleState)
    accum: AccumState = field(default_factory=AccumState)
    worker_plan: WorkerPlan = field(default_factory=WorkerPlan)
    manifests: dict[str, Any] = field(default_factory=dict)
