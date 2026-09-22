"""Run-config resume plans: checkpoint sidecars to read-only plans.

Owns the checkpoint sidecar envelope parse, the structural candidate
scan, the compatibility gates, the read-only resume resolver, and the
human-readable plan summary. Payload identity stays verified by
:mod:`hydra2.runtime.checkpoint` at bind time, before any runtime
object is mutated.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra2.contracts.common import ContractError
from hydra2.training._rc_digest import effective_run_id as effective_run_id
from hydra2.training._rc_digest import run_config_digest as run_config_digest
from hydra2.training._rc_digest import run_dir_for as run_dir_for
from hydra2.training._rc_require import _DIGEST_RE as _DIGEST_RE
from hydra2.training._rc_root import load_run_config as load_run_config
from hydra2.training._rc_sections import AccumState as AccumState
from hydra2.training._rc_sections import ResumePlan as ResumePlan
from hydra2.training._rc_sections import RunConfig as RunConfig
from hydra2.training._rc_sections import ShuffleState as ShuffleState
from hydra2.training._rc_sections import StreamCursor as StreamCursor
from hydra2.training._rc_sections import WorkerPlan as WorkerPlan

try:
    from hydra2._native import contracts as _rc_bridge  # pyrefly: ignore[missing-import]
except ImportError:  # pragma: no cover - import-time signal, same text as call-site
    _rc_bridge = None  # type: ignore[assignment]


def _require_rc_bridge() -> Any:
    """Resolve the bridge, fail closed when the extension is not built."""
    # All four sidecar gates ship atomically; probe each so partial wiring
    # fails closed here instead of surfacing AttributeError at call time.
    if (
        _rc_bridge is None
        or not hasattr(_rc_bridge, "rc_check_compat")
        or not hasattr(_rc_bridge, "rc_resume_shuffle")
        or not hasattr(_rc_bridge, "rc_resume_accum")
        or not hasattr(_rc_bridge, "rc_resume_worker_plan")
        or not hasattr(_rc_bridge, "rc_resume_manifests")
    ):
        raise ImportError(
            "hydra2._native extension with contracts not importable; "
            "run `pixi run build-ext` to build the extension before use"
        )
    return _rc_bridge


__all__ = [
    "_CKPT_STEM_RE",
    "_Sidecar",
    "_check_compat",
    "_iter_structural_candidates",
    "_parse_accum",
    "_parse_shuffle",
    "_parse_sidecar_manifests",
    "_parse_worker_plan",
    "_read_sidecar",
    "find_latest_checkpoint",
    "format_plan",
    "resolve_resume_plan",
]

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
    """Thin bridge delegate: ``hydra2._native.contracts.rc_resume_shuffle`` decides."""
    try:
        resolved: dict[str, Any] = _require_rc_bridge().rc_resume_shuffle(raw, str(sidecar))
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    buffer_keys_raw: list[str] = resolved["buffer_keys"]
    buffer_keys: tuple[str, ...] = tuple(buffer_keys_raw)
    epoch_seed: int = resolved["epoch_seed"]
    buffer_size: int = resolved["buffer_size"]
    buffer_rng_state: dict[str, Any] = resolved["buffer_rng_state"]
    prefix_hashes: dict[str, Any] = resolved["prefix_hashes"]
    return ShuffleState(
        buffer_keys=buffer_keys,
        epoch_seed=epoch_seed,
        buffer_size=buffer_size,
        buffer_rng_state=dict(buffer_rng_state),
        prefix_hashes=dict(prefix_hashes),
    )


def _parse_accum(raw: Any, *, sidecar: Path) -> AccumState:
    """Thin bridge delegate: ``hydra2._native.contracts.rc_resume_accum`` decides."""
    try:
        resolved: dict[str, Any] = _require_rc_bridge().rc_resume_accum(raw, str(sidecar))
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    yielded_batches: int = resolved["yielded_batches"]
    micro_in_update: int = resolved["micro_in_update"]
    return AccumState(
        yielded_batches=yielded_batches,
        micro_in_update=micro_in_update,
    )


def _parse_worker_plan(raw: Any, *, sidecar: Path) -> WorkerPlan:
    """Thin bridge delegate: ``hydra2._native.contracts.rc_resume_worker_plan`` decides."""
    try:
        resolved: dict[str, Any] = _require_rc_bridge().rc_resume_worker_plan(raw, str(sidecar))
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    num_workers: int = resolved["num_workers"]
    world_size: int = resolved["world_size"]
    rank: int = resolved["rank"]
    return WorkerPlan(
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
    )


def _parse_sidecar_manifests(raw: Any, *, sidecar: Path) -> dict[str, Any]:
    """Thin bridge delegate: ``hydra2._native.contracts.rc_resume_manifests`` decides."""
    try:
        resolved: dict[str, Any] = _require_rc_bridge().rc_resume_manifests(raw, str(sidecar))
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    return dict(resolved)


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
    shuffle_raw: dict[str, Any] = raw.get("shuffle", {})
    accum_raw: dict[str, Any] = raw.get("accum", {})
    worker_plan_raw: dict[str, Any] = raw.get("worker_plan", {})
    manifests_raw: dict[str, Any] = raw.get("manifests", {})
    return _Sidecar(
        cursor=cursor,
        global_update=global_update,
        run_digest=run_digest,
        rng_state=dict(rng_state),
        shuffle=_parse_shuffle(shuffle_raw, sidecar=sidecar),
        accum=_parse_accum(accum_raw, sidecar=sidecar),
        worker_plan=_parse_worker_plan(worker_plan_raw, sidecar=sidecar),
        manifests=_parse_sidecar_manifests(manifests_raw, sidecar=sidecar),
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
    try:
        _require_rc_bridge().rc_check_compat(
            expected_digest,
            parsed.run_digest,
            parsed.worker_plan.num_workers,
            config.data.num_workers,
            parsed.worker_plan.world_size,
            config.data.world_size,
            parsed.cursor.seed,
            config.seeds.data_seed,
            parsed.accum.micro_in_update,
            config.loop.accumulation_steps,
            len(parsed.shuffle.buffer_keys),
            config.data.shuffle_buffer_size,
            config.data.dataset_manifest_hash,
            parsed.manifests.get("dataset_manifest_hash"),
            str(ckpt),
        )
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc


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
