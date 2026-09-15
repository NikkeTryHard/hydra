"""Run-config auxiliary section parsers: seeds/selection/mirror/telemetry/eval/output.

Owns the strict per-section constructors for the six observer-side
sections. Selection parsing mirrors
:class:`eval.statistics.SelectionConfig` field-for-field at construction
time; the eval class stays authoritative at selection time.
"""

from __future__ import annotations

from typing import Any

from hydra2.contracts.common import ContractError
from hydra2.training._rc_require import _reject_unknown as _reject_unknown
from hydra2.training._rc_require import _require_nonnegative_int as _require_nonnegative_int
from hydra2.training._rc_require import _require_positive_int as _require_positive_int
from hydra2.training._rc_sections import _RUN_ID_RE as _RUN_ID_RE
from hydra2.training._rc_sections import _SELECTION_DESIGNS as _SELECTION_DESIGNS
from hydra2.training._rc_sections import EvalConfig as EvalConfig
from hydra2.training._rc_sections import MirrorConfig as MirrorConfig
from hydra2.training._rc_sections import OutputConfig as OutputConfig
from hydra2.training._rc_sections import SeedsConfig as SeedsConfig
from hydra2.training._rc_sections import SelectionConfig as SelectionConfig
from hydra2.training._rc_sections import TelemetryConfig as TelemetryConfig

__all__ = [
    "_parse_eval",
    "_parse_mirror",
    "_parse_output",
    "_parse_seeds",
    "_parse_selection",
    "_parse_telemetry",
]


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
