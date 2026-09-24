"""Run-config auxiliary section parsers: seeds/selection/mirror/telemetry/eval/output.

Owns the strict per-section constructors for the six observer-side
sections. Selection parsing mirrors
:class:`eval.statistics.SelectionConfig` field-for-field at construction
time; the eval class stays authoritative at selection time.
"""

from __future__ import annotations

from typing import Any

from hydra2.contracts.common import ContractError
from hydra2.training._rc_sections import EvalConfig as EvalConfig
from hydra2.training._rc_sections import MirrorConfig as MirrorConfig
from hydra2.training._rc_sections import OutputConfig as OutputConfig
from hydra2.training._rc_sections import SeedsConfig as SeedsConfig
from hydra2.training._rc_sections import SelectionConfig as SelectionConfig
from hydra2.training._rc_sections import TelemetryConfig as TelemetryConfig

try:
    from hydra2._native import contracts as _aux_bridge  # pyrefly: ignore[missing-import]
except ImportError:  # pragma: no cover - import-time signal, same text as call-site
    _aux_bridge = None  # type: ignore[assignment]


def _require_aux_bridge() -> Any:
    """Resolve the bridge, fail closed when the extension is not built."""
    if _aux_bridge is None or not hasattr(_aux_bridge, "rc_parse_seeds"):
        raise ImportError(
            "hydra2._native extension with contracts not importable; "
            "run `pixi run build-ext` to build the extension before use"
        )
    return _aux_bridge


__all__ = [
    "_parse_eval",
    "_parse_mirror",
    "_parse_output",
    "_parse_seeds",
    "_parse_selection",
    "_parse_telemetry",
]


def _parse_seeds(raw: Any) -> SeedsConfig:
    """Thin bridge delegate: ``hydra2._native.contracts.rc_parse_seeds`` decides."""
    try:
        resolved: dict[str, Any] = _require_aux_bridge().rc_parse_seeds(raw)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    data_seed: int = resolved["data_seed"]
    train_seed: int = resolved["train_seed"]
    selection_seed: int = resolved["selection_seed"]
    return SeedsConfig(data_seed=data_seed, train_seed=train_seed, selection_seed=selection_seed)


def _parse_selection(raw: Any) -> SelectionConfig:
    """Thin bridge delegate: ``hydra2._native.contracts.rc_parse_selection`` decides."""
    try:
        resolved: dict[str, Any] = _require_aux_bridge().rc_parse_selection(raw)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    select_n: int = resolved["N"]
    pilot_s: float = resolved["pilot_s"]
    delta: float = resolved["delta"]
    alpha: float = resolved["alpha"]
    beta: float = resolved["beta"]
    design: str = resolved["design"]
    peeks_raw: list[int] = resolved["declared_peeks"]
    declared_peeks: tuple[int, ...] = tuple(peeks_raw)
    margin: float = resolved["margin"]
    resamples: int = resolved["resamples"]
    seed: int = resolved["seed"]
    selection = SelectionConfig(
        N=select_n,
        pilot_s=pilot_s,
        delta=delta,
        alpha=alpha,
        beta=beta,
        design=design,
        declared_peeks=declared_peeks,
        margin=margin,
        resamples=resamples,
        seed=seed,
    )
    try:
        from hydra2.eval.selection import SelectionConfig as _Authoritative

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
    """Thin bridge delegate: ``hydra2._native.contracts.rc_parse_telemetry`` decides."""
    try:
        resolved: dict[str, Any] = _require_aux_bridge().rc_parse_telemetry(raw)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    mlflow_enabled: bool = resolved["mlflow_enabled"]
    verbose_enabled: bool = resolved["verbose_enabled"]
    verbose_interval_ms: int = resolved["verbose_interval_ms"]
    profiler_captures: int = resolved["profiler_captures"]
    return TelemetryConfig(
        mlflow_enabled=mlflow_enabled,
        verbose_enabled=verbose_enabled,
        verbose_interval_ms=verbose_interval_ms,
        profiler_captures=profiler_captures,
    )


def _parse_mirror(raw: Any) -> MirrorConfig:
    """Thin bridge delegate: ``hydra2._native.contracts.rc_parse_mirror`` decides."""
    try:
        resolved: dict[str, Any] = _require_aux_bridge().rc_parse_mirror(raw)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    enabled: bool = resolved["enabled"]
    project: str = resolved["project"]
    task_name: str | None = resolved["task_name"]
    offline_dir: str | None = resolved["offline_dir"]
    return MirrorConfig(
        enabled=enabled, project=project, task_name=task_name, offline_dir=offline_dir
    )


def _parse_eval(raw: Any) -> EvalConfig:
    """Thin bridge delegate: ``hydra2._native.contracts.rc_parse_eval`` decides."""
    try:
        resolved: dict[str, Any] = _require_aux_bridge().rc_parse_eval(raw)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    frequency_updates: int = resolved["frequency_updates"]
    num_batches: int = resolved["num_batches"]
    walls_dir: str | None = resolved["walls_dir"]
    eval_microbatch_size: int | None = resolved["microbatch_size"]
    return EvalConfig(
        frequency_updates=frequency_updates,
        num_batches=num_batches,
        walls_dir=walls_dir,
        microbatch_size=eval_microbatch_size,
    )


def _parse_output(raw: Any) -> OutputConfig:
    """Thin bridge delegate: ``hydra2._native.contracts.rc_parse_output`` decides."""
    try:
        resolved: dict[str, Any] = _require_aux_bridge().rc_parse_output(raw)
    except (ValueError, TypeError) as exc:
        raise ContractError(str(exc)) from exc
    artifact_root: str | None = resolved["artifact_root"]
    run_id: str | None = resolved["run_id"]
    return OutputConfig(artifact_root=artifact_root, run_id=run_id)
