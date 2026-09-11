"""Streaming-first training run configuration: YAML authority, output layout, resume.

Greenfield YAML surface (no materialized parquet shards required): the YAML
file is the single authority for a training run. It binds twelve sections
(``run`` / ``data`` / ``model`` / ``weights`` / ``optimizer`` / ``scheduler``
/ ``runtime`` / ``loop`` / ``seeds`` / ``selection`` / ``mirror`` / ``eval`` /
``output``) into the frozen :class:`RunConfig` dataclass. Real training
execution (stream reader, loop binding) lands in the next wave; this module
owns parsing, validation, output layout, and resume-plan resolution only.

Contract lock (streaming work, 2026-09-06; recorded here, not silent-picked):

- No materialized shards does NOT mean no validation. Ephemeral rows MUST
  carry the identical contracts as the parquet path: ingest
  validate-then-quarantine, actor/privileged split with leakage rejection,
  wall-disjoint train/val manifests, the actor-privileged firewall (no
  privileged key ever reaches the actor batch), counter-based semantic RNG
  for determinism (never wall-clock seeds), and ``(5,)`` dora indicators.
  The resumed stream position is the :class:`StreamCursor`. Enforcement
  lives in the stream reader / loop builders; this module carries the
  cursor shape and the scope flags (``wall_disjoint``, ``leakage_check``,
  ``dora_width``, ``actor_privileged_split``) so a config that disables
  them fails closed here.
- YAML and output stay under the artifact root. Mount paths NEVER appear
  literally in code or example configs; the dataset root is spelled
  ``${HYDRA2_DATA_ROOT}/...`` and resolved through the allowlisted
  ``${VAR}`` interpolation below. No secrets are read or written.
- Dependency: ``pyyaml`` is a pinned Pixi dependency (see
  ``[tool.pixi.pypi-dependencies]``); this module imports it at top level.

Deterministic resume: the same seed plus the same :class:`StreamCursor`
yields the same sequence. Resume restores the latest valid checkpoint plus
the cursor plus RNG state, verifying identity BEFORE any runtime object is
mutated (verification itself lives in
:mod:`hydra2.runtime.checkpoint`; this module only resolves the plan).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hydra2._canon import canonical_json_bytes
from hydra2.artifacts.atomic import atomic_replace_bytes
from hydra2.contracts.common import ContractError

__all__ = [
    "CONFIG_SECTIONS",
    "INTERPOLATION_ALLOWLIST",
    "RUN_KINDS",
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
    "create_run_layout",
    "deep_merge",
    "find_latest_checkpoint",
    "format_plan",
    "load_run_config",
    "read_latest_run",
    "resolve_resume_plan",
    "run_config_digest",
    "run_config_to_dict",
    "run_dir_for",
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
_ADAPTER_IDS: tuple[str, ...] = ("plain_pytorch", "fabric_2.6.5")
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


# ---------------------------------------------------------------------------
# Strict YAML loading: interpolation, base+override merge, unknown-key reject
# ---------------------------------------------------------------------------


def _interpolate_string(value: str, *, environ: Any, where: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in INTERPOLATION_ALLOWLIST:
            raise ContractError(
                f"{where}: ${{{name}}} is not allowlisted for interpolation; "
                f"allowed={sorted(INTERPOLATION_ALLOWLIST)}"
            )
        resolved = environ.get(name)
        if resolved is None or str(resolved) == "":
            raise ContractError(f"{where}: environment variable {name!r} is unset or empty")
        return str(resolved)

    return _INTERPOLATION_RE.sub(_replace, value)


def _interpolate_tree(node: Any, *, environ: Any, where: str) -> Any:
    if isinstance(node, str):
        return _interpolate_string(node, environ=environ, where=where)
    if isinstance(node, dict):
        return {
            key: _interpolate_tree(item, environ=environ, where=f"{where}.{key}")
            for key, item in node.items()
        }
    if isinstance(node, list):
        return [
            _interpolate_tree(item, environ=environ, where=f"{where}[{i}]")
            for i, item in enumerate(node)
        ]
    return node


def deep_merge(base: Any, override: Any) -> Any:
    """Deep-merge ``override`` onto ``base`` (base+override YAML composition).

    Mappings merge key-wise; every other type (including lists) is replaced
    wholesale by the override. Neither input is mutated.
    """
    if isinstance(base, dict) and isinstance(override, dict):
        merged: dict[Any, Any] = dict(base)
        for key, value in override.items():
            if key in merged:
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
    return override


def _reject_unknown(raw: Any, allowed: tuple[str, ...], *, where: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ContractError(f"{where} must be a mapping, got {type(raw).__name__}")
    unknown = sorted(k for k in raw if k not in allowed)
    if len(unknown) > 0:
        raise ContractError(f"{where} has unknown keys {unknown}; allowed={sorted(allowed)}")
    return dict(raw)


def _require_nonempty_str(raw: dict[str, Any], key: str, *, where: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or value.strip() == "":
        raise ContractError(f"{where}.{key} must be a non-empty string")
    return value


def _require_positive_int(raw: dict[str, Any], key: str, *, where: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{where}.{key} must be a positive int, got {value!r}")
    return value


def _require_bounded_int(raw: dict[str, Any], key: str, *, where: str, lo: int, hi: int) -> int:
    """Tuning knob in ``[lo, hi]`` (fail-closed; bounds are memory sanity, not tuning)."""
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not (lo <= value <= hi):
        raise ContractError(f"{where}.{key} must be an int in [{lo}, {hi}], got {value!r}")
    return value


def _require_nonnegative_int(raw: dict[str, Any], key: str, *, where: str) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{where}.{key} must be a non-negative int, got {value!r}")
    return value


def _require_nonnegative_float(raw: dict[str, Any], key: str, *, where: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where}.{key} must be a non-negative number, got {value!r}")
    number = float(value)
    if not (number == number and number not in (float("inf"), float("-inf"))) or number < 0.0:
        raise ContractError(f"{where}.{key} must be finite and non-negative, got {value!r}")
    return number


def _weight_map(raw: dict[str, Any], key: str, *, where: str) -> dict[str, float] | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ContractError(f"{where}.{key} must be a mapping or null, got {type(value).__name__}")
    out: dict[str, float] = {}
    for head, weight in value.items():
        if not isinstance(head, str) or head == "":
            raise ContractError(f"{where}.{key} keys must be non-empty strings")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not (float(weight) == float(weight))
            or float(weight) in (float("inf"), float("-inf"))
            or float(weight) < 0.0
        ):
            raise ContractError(f"{where}.{key}[{head!r}] must be finite non-negative")
        out[head] = float(weight)
    return out


def _require_bounded_float(
    raw: dict[str, Any],
    key: str,
    *,
    where: str,
    lo: float,
    hi: float,
    lo_open: bool = False,
    hi_open: bool = False,
) -> float:
    """Strict ``[lo, hi]`` float (``lo_open``/``hi_open`` select open ends)."""
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where}.{key} must be a number, got {value!r}")
    number = float(value)
    if not (number == number and number not in (float("inf"), float("-inf"))):
        raise ContractError(f"{where}.{key} must be finite, got {value!r}")
    lo_ok = lo < number if lo_open else lo <= number
    hi_ok = number < hi if hi_open else number <= hi
    if not (lo_ok and hi_ok):
        bound = f"{'(' if lo_open else '['}{lo}, {hi}{')' if hi_open else ']'}"
        raise ContractError(f"{where}.{key} must lie in {bound}, got {value!r}")
    return number


def _positive_float_map(raw: dict[str, Any], key: str, *, where: str) -> dict[str, float] | None:
    """Optional ``{name: positive-finite-mult}`` map (null when absent)."""
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ContractError(f"{where}.{key} must be a mapping or null, got {type(value).__name__}")
    out: dict[str, float] = {}
    for head, mult in value.items():
        if not isinstance(head, str) or head == "":
            raise ContractError(f"{where}.{key} keys must be non-empty strings")
        if (
            isinstance(mult, bool)
            or not isinstance(mult, (int, float))
            or not (float(mult) == float(mult))
            or float(mult) in (float("inf"), float("-inf"))
            or float(mult) <= 0.0
        ):
            raise ContractError(f"{where}.{key}[{head!r}] must be positive and finite")
        out[head] = float(mult)
    return out


def _require_loop_bool(raw: dict[str, Any], key: str, *, default: bool) -> bool:
    """Strict bool knob (absent/null → ``default``; non-bool raises)."""
    value = raw.get(key, default)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ContractError(f"loop.{key} must be a bool, got {value!r}")
    return value


_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _digest_pin_or_none(raw: dict[str, Any], key: str, *, where: str) -> str | None:
    """Optional ``sha256:<64 hex>`` provenance pin (null when absent)."""
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ContractError(f"{where}.{key} must be null or sha256:<64 hex>, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Section parsers (each rejects unknown keys; cross-checks at the end)
# ---------------------------------------------------------------------------


def _parse_run(raw: Any) -> RunMeta:
    fields = _reject_unknown(raw, ("id", "kind", "description"), where="run")
    run_id = fields.get("id", "")
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise ContractError(f"run.id must match [A-Za-z0-9][A-Za-z0-9_-]*, got {run_id!r}")
    kind = fields.get("kind", "supervised")
    if kind not in RUN_KINDS:
        raise ContractError(f"run.kind must be one of {list(RUN_KINDS)}, got {kind!r}")
    description = fields.get("description", "")
    if not isinstance(description, str):
        raise ContractError("run.description must be a string")
    return RunMeta(run_id=run_id, kind=kind, description=description)


def _parse_data(raw: Any) -> DataConfig:
    fields = _reject_unknown(
        raw,
        (
            "root",
            "scope",
            "train_split",
            "val_split",
            "wall_disjoint",
            "leakage_check",
            "dora_width",
            "actor_privileged_split",
            "dataset_manifest_hash",
            "shuffle_buffer_size",
            "drop_last",
            "num_workers",
            "world_size",
            "replay_backend",
            "decode_prefetch",
            "expand_batch_games",
        ),
        where="data",
    )
    root = _require_nonempty_str(fields, "root", where="data")
    if not Path(root).is_absolute():
        raise ContractError(f"data.root must be an absolute path after interpolation, got {root!r}")
    scope = fields.get("scope", _TENHOU_SCOPE)
    if scope != _TENHOU_SCOPE:
        raise ContractError(
            f"data.scope is Tenhou-only in v1 (must be {_TENHOU_SCOPE!r}), got {scope!r}"
        )
    train_split = fields.get("train_split", "train")
    val_split = fields.get("val_split", "validation")
    if not isinstance(train_split, str) or train_split == "":
        raise ContractError("data.train_split must be a non-empty string")
    if not isinstance(val_split, str) or val_split == "":
        raise ContractError("data.val_split must be a non-empty string")
    for flag in ("wall_disjoint", "leakage_check", "actor_privileged_split"):
        if fields.get(flag, True) is not True:
            raise ContractError(
                f"data.{flag} must stay true (ephemeral rows carry the full "
                "validate/quarantine/split/leakage contract); refusing to weaken it"
            )
    dora_width = fields.get("dora_width", 5)
    if dora_width != 5:
        raise ContractError(f"data.dora_width must be 5 (frozen (5,) dora), got {dora_width!r}")
    drop_last = fields.get("drop_last", True)
    if not isinstance(drop_last, bool):
        raise ContractError(f"data.drop_last must be a bool, got {drop_last!r}")
    replay_backend = fields.get("replay_backend", "rust")
    if replay_backend not in ("python", "rust"):
        raise ContractError(
            f"data.replay_backend must be 'python' or 'rust', got {replay_backend!r}"
        )
    return DataConfig(
        root=root,
        scope=str(scope),
        train_split=train_split,
        val_split=val_split,
        wall_disjoint=True,
        leakage_check=True,
        dora_width=5,
        actor_privileged_split=True,
        dataset_manifest_hash=_digest_pin_or_none(fields, "dataset_manifest_hash", where="data"),
        shuffle_buffer_size=_require_nonnegative_int(fields, "shuffle_buffer_size", where="data")
        if "shuffle_buffer_size" in fields
        else 10000,
        drop_last=drop_last,
        num_workers=_require_nonnegative_int(fields, "num_workers", where="data")
        if "num_workers" in fields
        else 0,
        world_size=_require_positive_int(fields, "world_size", where="data")
        if "world_size" in fields
        else 1,
        replay_backend=str(replay_backend),
        decode_prefetch=_require_bounded_int(fields, "decode_prefetch", where="data", lo=1, hi=1024)
        if "decode_prefetch" in fields
        else 64,
        expand_batch_games=_require_bounded_int(
            fields, "expand_batch_games", where="data", lo=1, hi=1024
        )
        if "expand_batch_games" in fields
        else 64,
    )


def _parse_model(raw: Any) -> ModelConfig:
    fields = _reject_unknown(raw, ("architecture_id", "action_count", "parameters"), where="model")
    architecture_id = fields.get("architecture_id", "hydra2_baseline_transformer_v1")
    if not isinstance(architecture_id, str) or architecture_id.strip() == "":
        raise ContractError("model.architecture_id must be a non-empty string")
    try:
        from hydra2.models.schema import KNOWN_ARCHITECTURES

        if architecture_id not in KNOWN_ARCHITECTURES:
            raise ContractError(f"model.architecture_id unknown: {architecture_id!r}")
    except ImportError:
        pass
    action_count = fields.get("action_count", 6792)
    if isinstance(action_count, bool) or not isinstance(action_count, int) or action_count <= 0:
        raise ContractError(f"model.action_count must be a positive int, got {action_count!r}")
    from hydra2.models.schema import BASELINE_ACTION_COUNT

    if action_count != BASELINE_ACTION_COUNT:
        raise ContractError(
            f"model.action_count {action_count} != baseline {BASELINE_ACTION_COUNT}"
        )
    parameters = fields.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ContractError("model.parameters must be a mapping")
    return ModelConfig(
        architecture_id=architecture_id,
        action_count=action_count,
        parameters=dict(parameters),
    )


def _parse_weights(raw: Any) -> WeightsConfig:
    fields = _reject_unknown(
        raw,
        (
            "w_policy",
            "w_placement",
            "w_value",
            "w_event",
            "w_belief",
            "privileged_source_hash",
            "label_smoothing",
        ),
        where="weights",
    )
    return WeightsConfig(
        w_policy=_require_nonnegative_float(fields, "w_policy", where="weights")
        if "w_policy" in fields
        else 1.0,
        w_placement=_require_nonnegative_float(fields, "w_placement", where="weights")
        if "w_placement" in fields
        else 0.0,
        w_value=_require_nonnegative_float(fields, "w_value", where="weights")
        if "w_value" in fields
        else 0.0,
        w_event=_weight_map(fields, "w_event", where="weights") if "w_event" in fields else None,
        w_belief=_weight_map(fields, "w_belief", where="weights") if "w_belief" in fields else None,
        privileged_source_hash=_digest_pin_or_none(
            fields, "privileged_source_hash", where="weights"
        ),
        label_smoothing=_require_bounded_float(
            fields, "label_smoothing", where="weights", lo=0.0, hi=1.0, hi_open=True
        )
        if "label_smoothing" in fields and fields.get("label_smoothing") is not None
        else 0.03,
    )


def _parse_optimizer(raw: Any) -> OptimizerConfig:
    fields = _reject_unknown(
        raw, ("id", "lr", "betas", "weight_decay", "head_lr_mult"), where="optimizer"
    )
    name = fields.get("id", "adamw")
    if name not in _OPTIMIZER_IDS:
        raise ContractError(f"optimizer.id must be one of {list(_OPTIMIZER_IDS)}, got {name!r}")
    lr = fields.get("lr", 3e-4)
    if (
        isinstance(lr, bool)
        or not isinstance(lr, (int, float))
        or not (float(lr) == float(lr))
        or float(lr) in (float("inf"), float("-inf"))
        or float(lr) <= 0.0
    ):
        raise ContractError(f"optimizer.lr must be positive and finite, got {lr!r}")
    betas_raw = fields.get("betas", [0.9, 0.999])
    if not isinstance(betas_raw, (list, tuple)) or len(betas_raw) != 2:
        raise ContractError(f"optimizer.betas must be a 2-element list, got {betas_raw!r}")
    betas = (float(betas_raw[0]), float(betas_raw[1]))
    for beta in betas:
        if not 0.0 <= beta < 1.0 or beta != beta:
            raise ContractError(f"optimizer.betas entries must lie in [0, 1), got {betas!r}")
    decay = fields.get("weight_decay", 0.01)
    if (
        isinstance(decay, bool)
        or not isinstance(decay, (int, float))
        or not (float(decay) == float(decay))
        or float(decay) in (float("inf"), float("-inf"))
        or float(decay) < 0.0
    ):
        raise ContractError(f"optimizer.weight_decay must be finite non-negative, got {decay!r}")
    head_mult = fields.get("head_lr_mult", 3.0)
    if (
        isinstance(head_mult, bool)
        or not isinstance(head_mult, (int, float))
        or not (float(head_mult) == float(head_mult))
        or float(head_mult) in (float("inf"), float("-inf"))
        or float(head_mult) <= 0.0
    ):
        raise ContractError(
            f"optimizer.head_lr_mult must be positive and finite, got {head_mult!r}"
        )
    return OptimizerConfig(
        name=name,
        lr=float(lr),
        betas=betas,
        weight_decay=float(decay),
        head_lr_mult=float(head_mult),
    )


def _parse_scheduler(raw: Any) -> SchedulerConfig:
    fields = _reject_unknown(
        raw,
        ("id", "warmup_updates", "parameters", "final_factor", "warmup_start_factor"),
        where="scheduler",
    )
    name = fields.get("id", "cosine")
    if name not in _SCHEDULER_IDS:
        raise ContractError(f"scheduler.id must be one of {list(_SCHEDULER_IDS)}, got {name!r}")
    warmup = fields.get("warmup_updates", 100)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ContractError(f"scheduler.warmup_updates must be a non-negative int, got {warmup!r}")
    parameters = fields.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ContractError("scheduler.parameters must be a mapping")
    return SchedulerConfig(
        name=name,
        warmup_updates=warmup,
        parameters=dict(parameters),
        final_factor=_require_bounded_float(
            fields, "final_factor", where="scheduler", lo=0.0, hi=1.0
        )
        if "final_factor" in fields and fields.get("final_factor") is not None
        else 0.0,
        warmup_start_factor=_require_bounded_float(
            fields, "warmup_start_factor", where="scheduler", lo=0.0, hi=1.0, lo_open=True
        )
        if "warmup_start_factor" in fields and fields.get("warmup_start_factor") is not None
        else 0.01,
    )


def _parse_runtime(raw: Any) -> RuntimeConfig:
    fields = _reject_unknown(
        raw, ("adapter_id", "device", "precision", "compile_mode"), where="runtime"
    )
    adapter_id = fields.get("adapter_id", "plain_pytorch")
    if adapter_id not in _ADAPTER_IDS:
        raise ContractError(
            f"runtime.adapter_id must be one of {list(_ADAPTER_IDS)}, got {adapter_id!r}"
        )
    device = fields.get("device", "cuda")
    if not isinstance(device, str) or (
        device != "cpu" and _CUDA_DEVICE_RE.fullmatch(device) is None
    ):
        raise ContractError(f"runtime.device must be 'cpu' or cuda[:N], got {device!r}")
    precision = fields.get("precision", "fp32")
    if precision not in _RUNTIME_PRECISIONS:
        raise ContractError(
            f"runtime.precision must be one of {list(_RUNTIME_PRECISIONS)} "
            f"(fp16 excluded by design), got {precision!r}"
        )
    compile_mode = fields.get("compile_mode", "eager")
    if compile_mode not in _COMPILE_MODES:
        raise ContractError(
            f"runtime.compile_mode must be one of {list(_COMPILE_MODES)}, got {compile_mode!r}"
        )
    try:
        from hydra2.runtime.protocol import RuntimeSpec, validate_runtime_spec

        validate_runtime_spec(
            RuntimeSpec(
                adapter_id=adapter_id,  # type: ignore[arg-type]
                device=device,
                precision=precision,  # type: ignore[arg-type]
                compile_mode=compile_mode,  # type: ignore[arg-type]
            )
        )
    except ImportError:
        pass
    return RuntimeConfig(
        adapter_id=adapter_id,
        device=device,
        precision=precision,
        compile_mode=compile_mode,
    )


def _parse_loop(raw: Any) -> LoopConfig:
    fields = _reject_unknown(
        raw,
        (
            "microbatch_size",
            "accumulation_steps",
            "gradient_clip_norm",
            "max_updates",
            "checkpoint_frequency_updates",
            "precision",
            "keep_last_checkpoints",
            "stratified_sampling",
            "sampling_ratios",
            "log_per_type_metrics",
            "fit_temperature",
            "fetch_prefetch_depth",
        ),
        where="loop",
    )
    clip = fields.get("gradient_clip_norm", 1.0)
    if clip is not None and (
        isinstance(clip, bool)
        or not isinstance(clip, (int, float))
        or not (float(clip) == float(clip))
        or float(clip) in (float("inf"), float("-inf"))
        or float(clip) <= 0.0
    ):
        raise ContractError(
            f"loop.gradient_clip_norm must be null or positive finite, got {clip!r}"
        )
    precision = fields.get("precision", "fp32")
    if precision not in ("fp32", "bf16_mixed"):
        raise ContractError(f"loop.precision must be 'fp32' or 'bf16_mixed', got {precision!r}")
    return LoopConfig(
        microbatch_size=_require_positive_int(fields, "microbatch_size", where="loop")
        if "microbatch_size" in fields
        else 4,
        accumulation_steps=_require_positive_int(fields, "accumulation_steps", where="loop")
        if "accumulation_steps" in fields
        else 8,
        gradient_clip_norm=None if clip is None else float(clip),
        max_updates=_require_positive_int(fields, "max_updates", where="loop")
        if "max_updates" in fields
        else 10000,
        checkpoint_frequency_updates=_require_positive_int(
            fields, "checkpoint_frequency_updates", where="loop"
        )
        if "checkpoint_frequency_updates" in fields
        else 500,
        precision=str(precision),
        keep_last_checkpoints=None
        if fields.get("keep_last_checkpoints") is None
        else _require_positive_int(fields, "keep_last_checkpoints", where="loop"),
        stratified_sampling=_require_loop_bool(fields, "stratified_sampling", default=False),
        sampling_ratios=_positive_float_map(fields, "sampling_ratios", where="loop"),
        log_per_type_metrics=_require_loop_bool(fields, "log_per_type_metrics", default=True),
        fit_temperature=_require_loop_bool(fields, "fit_temperature", default=True),
        fetch_prefetch_depth=_require_bounded_int(
            fields, "fetch_prefetch_depth", where="loop", lo=1, hi=16
        )
        if "fetch_prefetch_depth" in fields
        else 3,
    )


def _parse_seeds(raw: Any) -> SeedsConfig:
    fields = _reject_unknown(raw, ("data_seed", "train_seed", "selection_seed"), where="seeds")
    return SeedsConfig(
        data_seed=_require_nonnegative_int(fields, "data_seed", where="seeds")
        if "data_seed" in fields
        else 0,
        train_seed=_require_nonnegative_int(fields, "train_seed", where="seeds")
        if "train_seed" in fields
        else 0,
        selection_seed=_require_nonnegative_int(fields, "selection_seed", where="seeds")
        if "selection_seed" in fields
        else 0,
    )


def _parse_selection(raw: Any) -> SelectionConfig:
    fields = _reject_unknown(
        raw,
        (
            "N",
            "pilot_s",
            "delta",
            "alpha",
            "beta",
            "design",
            "declared_peeks",
            "margin",
            "resamples",
            "seed",
        ),
        where="selection",
    )
    design = fields.get("design", "fixed_n")
    if design not in _SELECTION_DESIGNS:
        raise ContractError(
            f"selection.design must be one of {list(_SELECTION_DESIGNS)}, got {design!r}"
        )
    peeks_raw = fields.get("declared_peeks", [30])
    if not isinstance(peeks_raw, (list, tuple)) or len(peeks_raw) == 0:
        raise ContractError("selection.declared_peeks must be a non-empty list of positive ints")
    peeks = tuple(peeks_raw)
    for peek in peeks:
        if isinstance(peek, bool) or not isinstance(peek, int) or peek < 1:
            raise ContractError(
                f"selection.declared_peeks entries must be positive ints, got {peeks!r}"
            )
    for key in ("pilot_s", "delta"):
        value = fields.get(key, 1.5 if key == "pilot_s" else 0.5)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not (float(value) == float(value))
            or float(value) in (float("inf"), float("-inf"))
            or float(value) <= 0.0
        ):
            raise ContractError(f"selection.{key} must be positive and finite, got {value!r}")
    for key in ("alpha", "beta"):
        value = fields.get(key, 0.05 if key == "alpha" else 0.2)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not (0.0 < float(value) < 1.0)
        ):
            raise ContractError(f"selection.{key} must lie in (0, 1), got {value!r}")
    margin = fields.get("margin", 0.0)
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not (float(margin) == float(margin))
        or float(margin) in (float("inf"), float("-inf"))
    ):
        raise ContractError(f"selection.margin must be finite, got {margin!r}")
    selection = SelectionConfig(
        N=_require_positive_int(fields, "N", where="selection") if "N" in fields else 30,
        pilot_s=float(fields.get("pilot_s", 1.5)),
        delta=float(fields.get("delta", 0.5)),
        alpha=float(fields.get("alpha", 0.05)),
        beta=float(fields.get("beta", 0.2)),
        design=design,
        declared_peeks=peeks,
        margin=float(margin),
        resamples=_require_positive_int(fields, "resamples", where="selection")
        if "resamples" in fields
        else 2000,
        seed=_require_nonnegative_int(fields, "seed", where="selection") if "seed" in fields else 0,
    )
    try:
        from hydra2.eval.statistics import SelectionConfig as _Authoritative

        _ = _Authoritative(  # intentionally discarded: construction validates cross-module contract
            N=selection.N,
            pilot_s=selection.pilot_s,
            delta=selection.delta,
            alpha=selection.alpha,
            beta=selection.beta,
            design=selection.design,  # type: ignore[arg-type]
            declared_peeks=selection.declared_peeks,
            margin=selection.margin,
            resamples=selection.resamples,
            seed=selection.seed,
        )
    except ImportError:
        pass
    return selection


def _parse_telemetry(raw: Any) -> TelemetryConfig:
    fields = _reject_unknown(
        raw,
        ("mlflow_enabled", "verbose_enabled", "verbose_interval_ms", "profiler_captures"),
        where="telemetry",
    )
    mlflow_enabled = fields.get("mlflow_enabled", True)
    if not isinstance(mlflow_enabled, bool):
        raise ContractError(f"telemetry.mlflow_enabled must be a bool, got {mlflow_enabled!r}")
    verbose_enabled = fields.get("verbose_enabled", False)
    if not isinstance(verbose_enabled, bool):
        raise ContractError(f"telemetry.verbose_enabled must be a bool, got {verbose_enabled!r}")
    verbose_interval_ms = fields.get("verbose_interval_ms", 50)
    if (
        not isinstance(verbose_interval_ms, bool)
        and isinstance(verbose_interval_ms, int)
        and verbose_interval_ms in (20, 50)
    ):
        pass
    else:
        raise ContractError(
            f"telemetry.verbose_interval_ms must be 20 or 50, got {verbose_interval_ms!r}"
        )
    profiler_captures = fields.get("profiler_captures", 0)
    if (
        not isinstance(profiler_captures, bool)
        and isinstance(profiler_captures, int)
        and 0 <= profiler_captures <= 16
    ):
        pass
    else:
        raise ContractError(
            f"telemetry.profiler_captures must be an int in [0, 16], got {profiler_captures!r}"
        )
    return TelemetryConfig(
        mlflow_enabled=mlflow_enabled,
        verbose_enabled=verbose_enabled,
        verbose_interval_ms=verbose_interval_ms,
        profiler_captures=profiler_captures,
    )


def _parse_mirror(raw: Any) -> MirrorConfig:
    fields = _reject_unknown(
        raw, ("enabled", "project", "task_name", "offline_dir"), where="mirror"
    )
    enabled = fields.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ContractError(f"mirror.enabled must be a bool, got {enabled!r}")
    project = fields.get("project", "hydra2-tenhou-4p")
    if not isinstance(project, str) or project == "":
        raise ContractError("mirror.project must be a non-empty string")
    task_name = fields.get("task_name")
    if task_name is not None and (not isinstance(task_name, str) or task_name == ""):
        raise ContractError("mirror.task_name must be null or a non-empty string")
    offline_dir = fields.get("offline_dir")
    if offline_dir is not None and (not isinstance(offline_dir, str) or offline_dir == ""):
        raise ContractError("mirror.offline_dir must be null or a non-empty string")
    return MirrorConfig(
        enabled=enabled, project=project, task_name=task_name, offline_dir=offline_dir
    )


def _parse_eval(raw: Any) -> EvalConfig:
    fields = _reject_unknown(
        raw, ("frequency_updates", "num_batches", "walls_dir", "microbatch_size"), where="eval"
    )
    walls_dir = fields.get("walls_dir")
    if walls_dir is not None and (not isinstance(walls_dir, str) or walls_dir == ""):
        raise ContractError("eval.walls_dir must be null or a non-empty string")
    return EvalConfig(
        frequency_updates=_require_positive_int(fields, "frequency_updates", where="eval")
        if "frequency_updates" in fields
        else 500,
        num_batches=_require_positive_int(fields, "num_batches", where="eval")
        if "num_batches" in fields
        else 10,
        walls_dir=walls_dir,
        microbatch_size=None
        if fields.get("microbatch_size") is None
        else _require_positive_int(fields, "microbatch_size", where="eval"),
    )


def _parse_output(raw: Any) -> OutputConfig:
    fields = _reject_unknown(raw, ("artifact_root", "run_id"), where="output")
    artifact_root = fields.get("artifact_root")
    if artifact_root is not None and (
        not isinstance(artifact_root, str) or artifact_root.strip() == ""
    ):
        raise ContractError("output.artifact_root must be null or a non-empty string")
    run_id = fields.get("run_id")
    if run_id is not None and (not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None):
        raise ContractError(f"output.run_id must match [A-Za-z0-9][A-Za-z0-9_-]*, got {run_id!r}")
    return OutputConfig(artifact_root=artifact_root, run_id=run_id)


_SECTION_PARSERS: dict[str, Any] = {
    "run": _parse_run,
    "data": _parse_data,
    "model": _parse_model,
    "weights": _parse_weights,
    "optimizer": _parse_optimizer,
    "scheduler": _parse_scheduler,
    "runtime": _parse_runtime,
    "loop": _parse_loop,
    "seeds": _parse_seeds,
    "selection": _parse_selection,
    "mirror": _parse_mirror,
    "telemetry": _parse_telemetry,
    "eval": _parse_eval,
    "output": _parse_output,
}


def _parse_root(mapping: Any) -> RunConfig:
    if not isinstance(mapping, dict):
        raise ContractError(f"run-config top level must be a mapping, got {type(mapping).__name__}")
    parked_rl = mapping.get("rl")
    if parked_rl is not None:
        raise ContractError(
            "top-level 'rl' is parked in streaming-first v1 (must be null); "
            "RL rollout sources arrive with run.kind=rl in a later wave"
        )
    fields = _reject_unknown(
        {key: value for key, value in mapping.items() if key != "rl"},
        CONFIG_SECTIONS,
        where="run-config",
    )
    parsed: dict[str, Any] = {}
    for section in CONFIG_SECTIONS:
        raw_section: Any = fields.get(section, {})
        if not isinstance(raw_section, dict):
            raise ContractError(f"{section} must be a mapping, got {type(raw_section).__name__}")
        parsed[section] = _SECTION_PARSERS[section](raw_section)
    config = RunConfig(
        run=parsed["run"],
        data=parsed["data"],
        model=parsed["model"],
        weights=parsed["weights"],
        optimizer=parsed["optimizer"],
        scheduler=parsed["scheduler"],
        runtime=parsed["runtime"],
        loop=parsed["loop"],
        seeds=parsed["seeds"],
        selection=parsed["selection"],
        mirror=parsed["mirror"],
        telemetry=parsed["telemetry"],
        eval=parsed["eval"],
        output=parsed["output"],
    )
    _cross_validate(config)
    return config


def _cross_validate(config: RunConfig) -> None:
    """Cross-section coherence checks (fail closed)."""
    runtime_precision = config.runtime.precision
    loop_precision = config.loop.precision
    expected_loop = "fp32" if runtime_precision == "fp32" else "bf16_mixed"
    if loop_precision != expected_loop:
        raise ContractError(
            f"loop.precision {loop_precision!r} disagrees with runtime.precision "
            f"{runtime_precision!r} (expected {expected_loop!r}); autocast scope must match"
        )
    output_id = config.output.run_id
    effective_run_id = (
        output_id if output_id is not None and len(output_id) > 0 else config.run.run_id
    )
    if _RUN_ID_RE.fullmatch(effective_run_id) is None:
        raise ContractError(f"effective run id invalid: {effective_run_id!r}")
    if runtime_precision == "bf16_mixed" and config.runtime.device == "cpu":
        raise ContractError(
            "runtime.precision 'bf16_mixed' requires a CUDA device "
            "(never silent CPU fallback; device availability itself is "
            "checked at bind time via require_device_available)"
        )
    auxiliary_positive = any(
        weight > 0.0
        for weight in [
            config.weights.w_placement,
            config.weights.w_value,
            *((config.weights.w_event if config.weights.w_event is not None else {}).values()),
            *((config.weights.w_belief if config.weights.w_belief is not None else {}).values()),
        ]
    )
    if auxiliary_positive and config.weights.privileged_source_hash is None:
        raise ContractError(
            "positive placement/value/event/belief weight REQUIRES "
            "weights.privileged_source_hash (positive-requires-binding): "
            "privileged labels must come from the pinned manifest"
        )


def _read_yaml_mapping(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"run config unreadable: {path} ({exc})") from exc
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ContractError(f"run config YAML malformed: {path} ({exc})") from exc
    if document is None:
        return {}
    if not isinstance(document, dict):
        raise ContractError(
            f"run config top level must be a mapping, got {type(document).__name__}"
        )
    return document


def load_run_config(
    path: str | Path,
    *,
    override_path: str | Path | None = None,
    environ: Any | None = None,
) -> RunConfig:
    """Load, merge, interpolate, and strictly validate a run config.

    ``override_path`` (optional) is deep-merged over ``path`` (base+override
    composition: operator overrides win key-wise; lists replace wholesale).
    ``${VAR}`` interpolation uses ``environ`` (default ``os.environ``) and
    the :data:`INTERPOLATION_ALLOWLIST`. Unknown keys raise at every level.
    """
    base_path = Path(path)
    document = _read_yaml_mapping(base_path)
    if override_path is not None:
        override = _read_yaml_mapping(Path(override_path))
        if not isinstance(override, dict):
            raise ContractError("override top level must be a mapping")
        if not isinstance(document, dict):
            raise ContractError("base top level must be a mapping")
        document = deep_merge(document, override)
    env: Any = os.environ if environ is None else environ
    resolved = _interpolate_tree(document, environ=env, where="run-config")
    return _parse_root(resolved)


def run_config_to_dict(config: RunConfig) -> dict[str, Any]:
    """Resolved config as a plain YAML/JSON-serializable mapping (run.yaml)."""
    selection = config.selection
    return {
        "run": {
            "id": config.run.run_id,
            "kind": config.run.kind,
            "description": config.run.description,
        },
        "data": {
            "root": config.data.root,
            "scope": config.data.scope,
            "train_split": config.data.train_split,
            "val_split": config.data.val_split,
            "wall_disjoint": config.data.wall_disjoint,
            "leakage_check": config.data.leakage_check,
            "dora_width": config.data.dora_width,
            "actor_privileged_split": config.data.actor_privileged_split,
            "dataset_manifest_hash": config.data.dataset_manifest_hash,
            "shuffle_buffer_size": config.data.shuffle_buffer_size,
            "drop_last": config.data.drop_last,
            "num_workers": config.data.num_workers,
            "world_size": config.data.world_size,
            "replay_backend": config.data.replay_backend,
            "decode_prefetch": config.data.decode_prefetch,
            "expand_batch_games": config.data.expand_batch_games,
        },
        "model": {
            "architecture_id": config.model.architecture_id,
            "action_count": config.model.action_count,
            "parameters": dict(config.model.parameters),
        },
        "weights": {
            "w_policy": config.weights.w_policy,
            "w_placement": config.weights.w_placement,
            "w_value": config.weights.w_value,
            "w_event": None if config.weights.w_event is None else dict(config.weights.w_event),
            "w_belief": None if config.weights.w_belief is None else dict(config.weights.w_belief),
            "privileged_source_hash": config.weights.privileged_source_hash,
            "label_smoothing": config.weights.label_smoothing,
        },
        "optimizer": {
            "id": config.optimizer.name,
            "lr": config.optimizer.lr,
            "betas": [config.optimizer.betas[0], config.optimizer.betas[1]],
            "weight_decay": config.optimizer.weight_decay,
            "head_lr_mult": config.optimizer.head_lr_mult,
        },
        "scheduler": {
            "id": config.scheduler.name,
            "warmup_updates": config.scheduler.warmup_updates,
            "parameters": dict(config.scheduler.parameters),
            "final_factor": config.scheduler.final_factor,
            "warmup_start_factor": config.scheduler.warmup_start_factor,
        },
        "runtime": {
            "adapter_id": config.runtime.adapter_id,
            "device": config.runtime.device,
            "precision": config.runtime.precision,
            "compile_mode": config.runtime.compile_mode,
        },
        "loop": {
            "microbatch_size": config.loop.microbatch_size,
            "accumulation_steps": config.loop.accumulation_steps,
            "gradient_clip_norm": config.loop.gradient_clip_norm,
            "max_updates": config.loop.max_updates,
            "checkpoint_frequency_updates": config.loop.checkpoint_frequency_updates,
            "precision": config.loop.precision,
            "keep_last_checkpoints": config.loop.keep_last_checkpoints,
            "stratified_sampling": config.loop.stratified_sampling,
            "sampling_ratios": None
            if config.loop.sampling_ratios is None
            else dict(config.loop.sampling_ratios),
            "log_per_type_metrics": config.loop.log_per_type_metrics,
            "fit_temperature": config.loop.fit_temperature,
            "fetch_prefetch_depth": config.loop.fetch_prefetch_depth,
        },
        "seeds": {
            "data_seed": config.seeds.data_seed,
            "train_seed": config.seeds.train_seed,
            "selection_seed": config.seeds.selection_seed,
        },
        "selection": {
            "N": selection.N,
            "pilot_s": selection.pilot_s,
            "delta": selection.delta,
            "alpha": selection.alpha,
            "beta": selection.beta,
            "design": selection.design,
            "declared_peeks": list(selection.declared_peeks),
            "margin": selection.margin,
            "resamples": selection.resamples,
            "seed": selection.seed,
        },
        "mirror": {
            "enabled": config.mirror.enabled,
            "project": config.mirror.project,
            "task_name": config.mirror.task_name,
            "offline_dir": config.mirror.offline_dir,
        },
        "eval": {
            "frequency_updates": config.eval.frequency_updates,
            "num_batches": config.eval.num_batches,
            "walls_dir": config.eval.walls_dir,
            "microbatch_size": config.eval.microbatch_size,
        },
        "output": {
            "artifact_root": config.output.artifact_root,
            "run_id": config.output.run_id,
        },
    }


def run_config_digest(config: RunConfig) -> str:
    """Stable ``sha256:<hex>`` identity of the resolved config (RFC 8785)."""
    return "sha256:" + hashlib.sha256(canonical_json_bytes(run_config_to_dict(config))).hexdigest()


# ---------------------------------------------------------------------------
# Output layout under <artifact_root>/runs/<id>/
# ---------------------------------------------------------------------------


def resolve_artifact_root(config: RunConfig) -> Path:
    """Resolved artifact root: ``output.artifact_root`` wins, else the lazy default."""
    if config.output.artifact_root is not None:
        return Path(config.output.artifact_root).resolve()
    from hydra2.config import artifact_root as _artifact_root

    return _artifact_root()


def effective_run_id(config: RunConfig) -> str:
    """``output.run_id`` override wins, else ``run.id``."""
    override = config.output.run_id
    return override if override is not None and len(override) > 0 else config.run.run_id


def run_dir_for(config: RunConfig, *, artifact_root: Path | str | None = None) -> Path:
    """``<artifact_root>/runs/<effective run id>`` (pure; creates nothing)."""
    run_id = effective_run_id(config)
    if artifact_root is not None:
        return (Path(artifact_root).resolve() / "runs" / run_id).resolve()
    return (resolve_artifact_root(config) / "runs" / run_id).resolve()


def _dump_run_yaml(config: RunConfig) -> bytes:
    text: str = yaml.safe_dump(
        run_config_to_dict(config), sort_keys=True, default_flow_style=False, allow_unicode=True
    )
    return text.encode("utf-8")


def create_run_layout(config: RunConfig, *, artifact_root: Path | str | None = None) -> Path:
    """Create the authoritative run directory tree (idempotent, fail closed).

    Layout under ``<artifact_root>/runs/<id>/``::

        run.yaml               resolved config (this module's output)
        manifests/             dataset / model / runtime / environment manifests
        checkpoints/           ckpt-<update>.pt + sidecars; resume source
        best-ckpt.pt           published on promotion only (absent until then)
        logs/train.log         line-oriented training log (created empty)
        logs/metrics.jsonl     one JSON object per line (created empty)
        mirror/                observer-mirror staging (ClearML offline sessions)
        eval/                  duplicate-wall evaluation reports

    Plus the ``<artifact_root>/runs/latest-run`` marker (text file holding
    the latest created run id). Re-creating with a byte-identical
    ``run.yaml`` is a no-op; a differing ``run.yaml`` raises instead of
    overwriting (never silently adopt a new spec under an old id).
    """
    run_dir = run_dir_for(config, artifact_root=artifact_root)
    for subdir in ("manifests", "checkpoints", "logs", "mirror", "eval"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    payload = _dump_run_yaml(config)
    target = run_dir / "run.yaml"
    if target.is_file():
        if target.read_bytes() != payload:
            raise ContractError(
                f"run.yaml mismatch in {run_dir}: refusing to overwrite a different "
                "resolved spec under the same run id (pick a new run id)"
            )
    else:
        atomic_replace_bytes(target, payload)
    for log_name in ("train.log", "metrics.jsonl"):
        log_path = run_dir / "logs" / log_name
        if not log_path.exists():
            log_path.touch(exist_ok=True)
    marker = run_dir.parent / "latest-run"
    atomic_replace_bytes(marker, (effective_run_id(config) + "\n").encode("utf-8"))
    return run_dir


def read_latest_run(artifact_root: Path | str | None = None) -> str | None:
    """Latest created run id from the ``runs/latest-run`` marker, if present."""
    if artifact_root is None:
        from hydra2.config import artifact_root as _artifact_root

        root = _artifact_root()
    else:
        root = Path(artifact_root).resolve()
    marker = root / "runs" / "latest-run"
    if not marker.is_file():
        return None
    text = marker.read_text(encoding="utf-8").strip()
    return text if text != "" else None


# ---------------------------------------------------------------------------
# Resume-plan resolution (read-only; verify-before-mutate at bind time)
# ---------------------------------------------------------------------------


_CKPT_STEM_RE = re.compile(r"ckpt-([0-9]+)\Z")


@dataclass(frozen=True, slots=True)
class _Sidecar:
    """Parsed checkpoint sidecar (ResumeExactPlan P1-P10 envelope)."""

    cursor: StreamCursor
    global_update: int
    run_digest: str
    rng_state: dict[str, Any]
    shuffle: ShuffleState
    accum: AccumState
    worker_plan: WorkerPlan
    manifests: dict[str, Any]


def _parse_shuffle(raw: Any, *, sidecar: Path) -> ShuffleState:
    if not isinstance(raw, dict):
        raise ContractError(f"checkpoint sidecar shuffle must be a mapping: {sidecar}")
    allowed = (
        "buffer_keys",
        "epoch_seed",
        "buffer_size",
        "buffer_rng_state",
        "buffer_entries",
        "prefix_hashes",
    )
    unknown = sorted(k for k in raw if k not in allowed)
    if len(unknown) > 0:
        raise ContractError(f"checkpoint sidecar shuffle unknown keys {unknown}: {sidecar}")
    keys = raw.get("buffer_keys", [])
    if not isinstance(keys, list) or any(not isinstance(k, str) or k == "" for k in keys):
        raise ContractError(f"checkpoint sidecar shuffle.buffer_keys must be str list: {sidecar}")
    epoch_seed = raw.get("epoch_seed", 0)
    if isinstance(epoch_seed, bool) or not isinstance(epoch_seed, int) or epoch_seed < 0:
        raise ContractError(f"checkpoint sidecar shuffle.epoch_seed invalid: {sidecar}")
    buffer_size = raw.get("buffer_size", 0)
    if isinstance(buffer_size, bool) or not isinstance(buffer_size, int) or buffer_size < 0:
        raise ContractError(f"checkpoint sidecar shuffle.buffer_size invalid: {sidecar}")
    if len(keys) > buffer_size:
        raise ContractError(f"checkpoint sidecar shuffle overflows buffer_size: {sidecar}")
    buffer_rng = raw.get("buffer_rng_state", {})
    if not isinstance(buffer_rng, dict):
        raise ContractError(f"checkpoint sidecar shuffle.buffer_rng_state invalid: {sidecar}")
    prefix_raw = raw.get("prefix_hashes", {})
    if not isinstance(prefix_raw, dict):
        raise ContractError(f"checkpoint sidecar shuffle.prefix_hashes invalid: {sidecar}")
    if len(prefix_raw) > 0:
        unknown_prefix = sorted(k for k in prefix_raw if k not in ("file", "count", "sha256"))
        if len(unknown_prefix) > 0:
            raise ContractError(
                f"checkpoint sidecar shuffle.prefix_hashes unknown keys {unknown_prefix}: {sidecar}"
            )
        prefix_file = prefix_raw.get("file")
        prefix_count = prefix_raw.get("count")
        prefix_digest = prefix_raw.get("sha256")
        if not isinstance(prefix_file, str) or prefix_file == "":
            raise ContractError(f"checkpoint sidecar shuffle.prefix_hashes.file invalid: {sidecar}")
        if isinstance(prefix_count, bool) or not isinstance(prefix_count, int) or prefix_count < 0:
            raise ContractError(
                f"checkpoint sidecar shuffle.prefix_hashes.count invalid: {sidecar}"
            )
        if not isinstance(prefix_digest, str) or not prefix_digest.startswith("sha256:"):
            raise ContractError(
                f"checkpoint sidecar shuffle.prefix_hashes.sha256 invalid: {sidecar}"
            )
    return ShuffleState(
        buffer_keys=tuple(keys),
        epoch_seed=epoch_seed,
        buffer_size=buffer_size,
        buffer_rng_state=dict(buffer_rng),
        prefix_hashes=dict(prefix_raw),
    )


def _parse_accum(raw: Any, *, sidecar: Path) -> AccumState:
    if not isinstance(raw, dict):
        raise ContractError(f"checkpoint sidecar accum must be a mapping: {sidecar}")
    unknown = sorted(k for k in raw if k not in ("yielded_batches", "micro_in_update"))
    if len(unknown) > 0:
        raise ContractError(f"checkpoint sidecar accum unknown keys {unknown}: {sidecar}")
    yielded = raw.get("yielded_batches", 0)
    micro = raw.get("micro_in_update", 0)
    for name, value in (("yielded_batches", yielded), ("micro_in_update", micro)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractError(f"checkpoint sidecar accum.{name} invalid: {sidecar}")
    return AccumState(yielded_batches=yielded, micro_in_update=micro)


def _parse_worker_plan(raw: Any, *, sidecar: Path) -> WorkerPlan:
    if not isinstance(raw, dict):
        raise ContractError(f"checkpoint sidecar worker_plan must be a mapping: {sidecar}")
    unknown = sorted(k for k in raw if k not in ("num_workers", "world_size", "rank"))
    if len(unknown) > 0:
        raise ContractError(f"checkpoint sidecar worker_plan unknown keys {unknown}: {sidecar}")
    workers = raw.get("num_workers", 0)
    world = raw.get("world_size", 1)
    rank = raw.get("rank", 0)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ContractError(f"checkpoint sidecar worker_plan.num_workers invalid: {sidecar}")
    if isinstance(world, bool) or not isinstance(world, int) or world < 1:
        raise ContractError(f"checkpoint sidecar worker_plan.world_size invalid: {sidecar}")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world:
        raise ContractError(f"checkpoint sidecar worker_plan.rank invalid: {sidecar}")
    return WorkerPlan(num_workers=workers, world_size=world, rank=rank)


def _parse_sidecar_manifests(raw: Any, *, sidecar: Path) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ContractError(f"checkpoint sidecar manifests must be a mapping: {sidecar}")
    unknown = sorted(k for k in raw if k not in ("stream_manifest_hash", "dataset_manifest_hash"))
    if len(unknown) > 0:
        raise ContractError(f"checkpoint sidecar manifests unknown keys {unknown}: {sidecar}")
    out: dict[str, Any] = {}
    for key in ("stream_manifest_hash", "dataset_manifest_hash"):
        value = raw.get(key)
        if value is not None and (
            not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None
        ):
            raise ContractError(f"checkpoint sidecar manifests.{key} invalid: {sidecar}")
        out[key] = value
    return out


def _read_sidecar(ckpt_path: Path) -> _Sidecar:
    sidecar = ckpt_path.with_suffix(".json")
    if not sidecar.is_file():
        raise ContractError(f"checkpoint sidecar missing: {sidecar}")
    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ContractError(f"checkpoint sidecar unreadable: {sidecar} ({exc})") from exc
    if not isinstance(raw, dict):
        raise ContractError(f"checkpoint sidecar must be a mapping: {sidecar}")
    global_update = raw.get("global_update")
    if isinstance(global_update, bool) or not isinstance(global_update, int) or global_update < 0:
        raise ContractError(f"checkpoint sidecar global_update invalid: {sidecar}")
    run_digest = raw.get("run_digest")
    if not isinstance(run_digest, str) or _DIGEST_RE.fullmatch(run_digest) is None:
        raise ContractError(f"checkpoint sidecar run_digest invalid: {sidecar}")
    cursor = StreamCursor.from_dict(raw.get("stream_cursor"))
    rng_state = raw.get("rng_state")
    if not isinstance(rng_state, dict):
        raise ContractError(f"checkpoint sidecar rng_state must be a mapping: {sidecar}")
    return _Sidecar(
        cursor=cursor,
        global_update=global_update,
        run_digest=run_digest,
        rng_state=dict(rng_state),
        shuffle=_parse_shuffle(raw.get("shuffle", {}), sidecar=sidecar),
        accum=_parse_accum(raw.get("accum", {}), sidecar=sidecar),
        worker_plan=_parse_worker_plan(raw.get("worker_plan", {}), sidecar=sidecar),
        manifests=_parse_sidecar_manifests(raw.get("manifests", {}), sidecar=sidecar),
    )


def _iter_structural_candidates(run_dir: str | Path) -> list[tuple[int, Path]]:
    """Checkpoints with a structurally valid sidecar, newest first (read-only)."""
    checkpoint_dir = Path(run_dir) / "checkpoints"
    if not checkpoint_dir.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for candidate in sorted(checkpoint_dir.glob("ckpt-*.pt")):
        match = _CKPT_STEM_RE.fullmatch(candidate.stem)
        if match is None or not candidate.is_file():
            continue
        try:
            parsed = _read_sidecar(candidate)
        except ContractError:
            continue
        file_update = int(match.group(1))
        if parsed.global_update != file_update:
            continue
        found.append((parsed.global_update, candidate))
    found.sort(key=lambda item: item[0], reverse=True)
    return found


def find_latest_checkpoint(run_dir: str | Path) -> Path | None:
    """Latest *valid* checkpoint in ``<run_dir>/checkpoints/`` (read-only).

    Candidates are ``ckpt-<update>.pt`` files with a readable sidecar
    (``ckpt-<update>.json`` carrying ``global_update`` + ``run_digest`` +
    ``stream_cursor`` + RNG bundle + shuffle/accum/worker/manifest extras).
    Entries with missing/corrupt sidecars are skipped so a torn write from
    an interrupted run can never become the resume point. Returns ``None``
    when no valid checkpoint exists. Compatibility against the run config
    (digest, worker plan, seed) is checked by :func:`resolve_resume_plan`,
    not here.
    """
    candidates = _iter_structural_candidates(run_dir)
    return candidates[0][1] if len(candidates) > 0 else None


def _check_compat(config: RunConfig, expected_digest: str, parsed: _Sidecar, ckpt: Path) -> None:
    """Compatibility gates: digest, worker plan, batch, seed (all hard errors)."""
    if parsed.run_digest != expected_digest:
        raise ContractError(
            f"resume checkpoint {ckpt} records run_digest {parsed.run_digest} "
            f"but run.yaml resolves to {expected_digest} (config drift)"
        )
    plan = parsed.worker_plan
    if plan.num_workers != config.data.num_workers:
        raise ContractError(
            f"resume worker_plan.num_workers={plan.num_workers} != "
            f"config data.num_workers={config.data.num_workers}: {ckpt}"
        )
    if plan.world_size != config.data.world_size:
        raise ContractError(
            f"resume worker_plan.world_size={plan.world_size} != "
            f"config data.world_size={config.data.world_size}: {ckpt}"
        )
    if parsed.cursor.seed != config.seeds.data_seed:
        raise ContractError(
            f"resume cursor seed={parsed.cursor.seed} != "
            f"config seeds.data_seed={config.seeds.data_seed}: {ckpt}"
        )
    if parsed.accum.micro_in_update >= config.loop.accumulation_steps:
        raise ContractError(
            f"resume accum.micro_in_update={parsed.accum.micro_in_update} outside "
            f"[0, {config.loop.accumulation_steps}): {ckpt}"
        )
    if len(parsed.shuffle.buffer_keys) > config.data.shuffle_buffer_size:
        raise ContractError(
            f"resume shuffle buffer {len(parsed.shuffle.buffer_keys)} keys exceeds "
            f"config data.shuffle_buffer_size={config.data.shuffle_buffer_size}: {ckpt}"
        )
    pinned = config.data.dataset_manifest_hash
    recorded = parsed.manifests.get("dataset_manifest_hash")
    if pinned is not None and recorded is not None and recorded != pinned:
        raise ContractError(
            f"resume dataset_manifest_hash {recorded} != config pin {pinned}: {ckpt}"
        )


def resolve_resume_plan(run_dir: str | Path, *, which: str | Path | None = None) -> ResumePlan:
    """Resolve the resume point for ``run_dir`` (read-only).

    ``which`` is ``None``/``"latest"`` (latest compatible checkpoint) or a
    path to an explicit ``ckpt-*.pt`` file. Compatibility (run digest,
    worker plan, batch horizon, seed, manifest pin) is verified against
    ``run.yaml`` BEFORE anything is returned; any mismatch raises
    :class:`ContractError`. Payload identity is verified by
    :mod:`hydra2.runtime.checkpoint` at bind time, before any runtime
    object is mutated — this plan only names the point.
    """
    directory = Path(run_dir)
    run_yaml = directory / "run.yaml"
    if not run_yaml.is_file():
        raise ContractError(f"run layout absent (no run.yaml): {directory}")
    config = load_run_config(run_yaml)
    expected_digest = run_config_digest(config)
    if which is None or (isinstance(which, str) and which == "latest"):
        candidates = _iter_structural_candidates(directory)
        if len(candidates) == 0:
            raise ContractError(f"no valid checkpoint in {directory / 'checkpoints'}")
        failures: list[str] = []
        for _, candidate in candidates:
            parsed = _read_sidecar(candidate)
            try:
                _check_compat(config, expected_digest, parsed, candidate)
            except ContractError as exc:
                failures.append(f"{candidate.name}: {exc}")
                continue
            return ResumePlan(
                run_dir=directory.resolve(),
                checkpoint=candidate.resolve(),
                cursor=parsed.cursor,
                global_update=parsed.global_update,
                rng_state=parsed.rng_state,
                run_digest=parsed.run_digest,
                shuffle=parsed.shuffle,
                accum=parsed.accum,
                worker_plan=parsed.worker_plan,
                manifests=parsed.manifests,
            )
        joined = "; ".join(failures)
        raise ContractError(f"no compatible checkpoint in {directory}: {joined}")
    checkpoint = Path(which)
    if not checkpoint.is_file():
        raise ContractError(f"resume checkpoint does not exist: {checkpoint}")
    parsed = _read_sidecar(checkpoint)
    _check_compat(config, expected_digest, parsed, checkpoint)
    return ResumePlan(
        run_dir=directory.resolve(),
        checkpoint=checkpoint.resolve(),
        cursor=parsed.cursor,
        global_update=parsed.global_update,
        rng_state=parsed.rng_state,
        run_digest=parsed.run_digest,
        shuffle=parsed.shuffle,
        accum=parsed.accum,
        worker_plan=parsed.worker_plan,
        manifests=parsed.manifests,
    )


# ---------------------------------------------------------------------------
# Human-readable plan (what `hydra2 train --dry-run` prints)
# ---------------------------------------------------------------------------


def format_plan(
    config: RunConfig,
    *,
    artifact_root: Path | str | None = None,
    resume: ResumePlan | None = None,
) -> str:
    """Resolved plan summary: ids, paths, sizes, seeds, and resume point."""
    run_dir = run_dir_for(config, artifact_root=artifact_root)
    digest = run_config_digest(config)
    minibatch = config.loop.optimizer_minibatch_size
    eval_micro_raw = config.eval.microbatch_size
    if eval_micro_raw is not None and eval_micro_raw != 0:
        eval_micro = eval_micro_raw
    else:
        eval_micro = config.loop.microbatch_size
    lines = [
        f"run: {effective_run_id(config)} (kind={config.run.kind})",
        f"digest: {digest}",
        f"run_dir: {run_dir}",
        f"data: root={config.data.root} scope={config.data.scope} "
        f"train={config.data.train_split} val={config.data.val_split} "
        f"wall_disjoint={config.data.wall_disjoint} buffer={config.data.shuffle_buffer_size} "
        f"workers={config.data.num_workers}x{config.data.world_size}",
        f"model: {config.model.architecture_id} actions={config.model.action_count}",
        f"weights: policy={config.weights.w_policy} placement={config.weights.w_placement} "
        f"value={config.weights.w_value} smoothing={config.weights.label_smoothing}",
        f"optimizer: {config.optimizer.name} lr={config.optimizer.lr} "
        f"betas={list(config.optimizer.betas)} decay={config.optimizer.weight_decay} "
        f"head_lr_mult={config.optimizer.head_lr_mult}",
        f"scheduler: {config.scheduler.name} warmup={config.scheduler.warmup_updates} "
        f"final-x{config.scheduler.final_factor} "
        f"warmup-start-x{config.scheduler.warmup_start_factor}",
        f"runtime: {config.runtime.adapter_id} {config.runtime.device} "
        f"{config.runtime.precision} {config.runtime.compile_mode}",
        f"loop: microbatch={config.loop.microbatch_size} accum={config.loop.accumulation_steps} "
        f"minibatch={minibatch} clip={config.loop.gradient_clip_norm} "
        f"max_updates={config.loop.max_updates}",
        f"ckpt_every={config.loop.checkpoint_frequency_updates}",
        f"keep={config.loop.keep_last_checkpoints} "
        f"stratified={config.loop.stratified_sampling} ratios={config.loop.sampling_ratios} "
        f"per_type={config.loop.log_per_type_metrics} fit_temp={config.loop.fit_temperature}",
        f"mirror: enabled={config.mirror.enabled} project={config.mirror.project}",
        f"seeds: data={config.seeds.data_seed} train={config.seeds.train_seed} "
        f"selection={config.seeds.selection_seed}",
        f"telemetry: mlflow={config.telemetry.mlflow_enabled} "
        f"verbose={config.telemetry.verbose_enabled}@{config.telemetry.verbose_interval_ms}ms "
        f"profiler={config.telemetry.profiler_captures}",
        f"selection: N={config.selection.N} design={config.selection.design} "
        f"delta={config.selection.delta} alpha={config.selection.alpha}",
        f"eval: every={config.eval.frequency_updates} batches={config.eval.num_batches} "
        f"micro={eval_micro}",
    ]
    if resume is not None:
        cursor = resume.cursor
        lines.append(
            f"resume: ckpt={resume.checkpoint} update={resume.global_update} "
            f"cursor=(file={cursor.file_index} offset={cursor.byte_offset} "
            f"games={cursor.games_seen} seed={cursor.seed} epoch={cursor.epoch})"
        )
        lines.append(
            f"resume-state: digest_ok={resume.run_digest == digest} "
            f"buffered_keys={len(resume.shuffle.buffer_keys)} "
            f"yielded={resume.accum.yielded_batches} "
            f"micro_in_update={resume.accum.micro_in_update} "
            f"workers={resume.worker_plan.num_workers}x{resume.worker_plan.world_size} "
            f"rank={resume.worker_plan.rank}"
        )
    else:
        lines.append("resume: fresh start (no --resume)")
    return "\n".join(lines) + "\n"
