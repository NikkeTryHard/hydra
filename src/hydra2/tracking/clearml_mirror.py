"""WP-05B/WP-11 ClearML observer mirror over local authoritative artifacts.

Local checkpoints/manifests stay authoritative (D-007): this module only
copies allowlisted scalars, digests, and JSON snapshots into ClearML for
visualization. It NEVER feeds values back into training, RNG, or sampler
state, and every method degrades to a warn-only no-op instead of raising.

Sole ``import clearml`` owner: ``clearml`` is imported lazily inside
:meth:`ClearmlMirror.start_run` and :func:`is_enabled` only. The rest of
the codebase (including :mod:`hydra2.training.loop` and
:mod:`hydra2.training.replay`) references this module via
``TYPE_CHECKING`` imports plus a lazy :func:`make_mirror` call, so
importing :mod:`hydra2.tracking` never requires the ClearML SDK.

Enablement (disabled by default):

- ``HYDRA2_CLEARML_ENABLED=1`` opts in; without it the factory returns
  :class:`NullMirror`.
- ``HYDRA2_CLEARML_DISABLED=1`` is the kill-switch and wins over
  everything, including an explicit ``enabled=True``.
- :func:`is_enabled` additionally requires a clean ``import clearml``.

Offline mode: when ``CLEARML_OFFLINE_MODE=1`` is set, :meth:`start_run`
calls ``Task.set_offline(True)`` before ``Task.init`` (required ordering)
and points the SDK cache at :func:`default_offline_dir` — ``<artifact_root>/
clearml_offline`` unless ``HYDRA2_CLEARML_OFFLINE_DIR`` overrides it — so
sessions buffer to a hermetic zip replayable later via
``Task.import_offline_session``. Model weights are excluded from the
offline contract: checkpoints stay in local artifacts, never in the model
registry. Autolog is forbidden: ``Task.init`` always passes
``auto_connect_frameworks=False, auto_connect_arg_parser=False``; only the
explicit ``report_scalar`` / ``upload_artifact`` / ``connect`` calls below
emit data.

Call sites: ``SupervisedLoop`` / ``ActorLearnerReplay`` construct via
``make_mirror(manifest_hashes=..., loop_config={...})`` + ``start_run()``
and log ``log_update`` + ``log_checkpoint`` after each published
checkpoint. ``log_eval_report`` / ``log_promotion`` /
``log_duplicate_audit`` are reserved builders (covered by mirror unit
tests; no production callers yet).
"""

from __future__ import annotations

import contextlib
import math
import os
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "EXPERIMENT_DEFAULT",
    "METRIC_ALLOWLIST",
    "METRIC_ALLOWLIST_PREFIXES",
    "TASK_DEFAULT",
    "ClearmlMirror",
    "NullMirror",
    "default_offline_dir",
    "is_enabled",
    "make_mirror",
]

#: ClearML project name (auto-created server-side on first use).
EXPERIMENT_DEFAULT = "hydra2-tenhou-4p"

#: Fallback ClearML task name when the caller supplies neither task nor run name.
TASK_DEFAULT = "hydra2-training"

#: Scalar keys allowed into ClearML. ``global_update`` travels as the
#: report iteration, never as a series. Same key set as the MLflow mirror,
#: which shares :func:`_filter_metrics` (placement/value/event/belief heads).
METRIC_ALLOWLIST = frozenset(
    {
        "total",
        "policy",
        "placement",
        "value",
        "event",
        "belief",
        "masked_nll",
        "top1",
        "top3",
        "top5",
        "calibration_ece",
        "legal_uniform_nll",
        "legal_uniform_gap",
        "legal_uniform_comparison",
        "support_min",
        "support_max",
        "confusion",
        "strata",
        "observed_estimate",
        "ci_lo",
        "ci_hi",
        "num_eval_batches",
        "temperature",
        "calibrated_nll",
        "calibrated_ece",
    }
)

#: Per-head series (``event_<head>`` / ``belief_<head>``), flattened
#: per-type scorecards (``per_type/<kind>/<metric>``), and post-hoc
#: calibration scalars (``temperature`` / ``calibrated_*``) pass the
#: allowlist.  Exact keys above cover the known scalars; the prefixes
#: future-proof eval-report additions under the same families.
METRIC_ALLOWLIST_PREFIXES = ("event_", "belief_", "per_type/", "calibrated_", "temperature")

#: Promotion record keys mirrored as scalars alongside the record artifact.
_PROMOTION_METRIC_KEYS = ("observed_estimate", "ci_lo", "ci_hi")

_TRUTHY = frozenset({"1", "true", "yes", "on"})


class _TaskLogger(Protocol):
    """Structural type for the ClearML task logger (SDK is untyped/optional)."""

    def report_scalar(self, *args: object, **kwargs: object) -> None: ...
    def report_text(self, *args: object, **kwargs: object) -> None: ...


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def default_offline_dir(*, artifact_root: Path | str | None = None) -> Path:
    """Hermetic default for ClearML offline sessions (no mkdir side effect).

    ``HYDRA2_CLEARML_OFFLINE_DIR`` wins when set and non-empty; otherwise
    ``<artifact_root>/clearml_offline`` where the root honors
    ``HYDRA2_ARTIFACT_ROOT`` via :func:`hydra2.config.artifact_root`.
    """
    override = os.environ.get("HYDRA2_CLEARML_OFFLINE_DIR")
    if override is not None and override.strip() != "":
        return Path(override.strip())
    if artifact_root is None:
        from hydra2.config import artifact_root as _artifact_root

        base = _artifact_root()
    else:
        base = Path(artifact_root)
    return base / "clearml_offline"


def is_enabled(*, explicit: bool | None = None) -> bool:
    """Disabled by default; opt-in plus a clean ``import clearml``.

    ``explicit=False`` (or ``HYDRA2_CLEARML_DISABLED`` truthy) forces off and
    wins over everything. Otherwise requires ``explicit=True`` (or
    ``HYDRA2_CLEARML_ENABLED`` truthy) AND an importable ``clearml``
    package. Never raises.
    """
    try:
        if explicit is False or _env_truthy("HYDRA2_CLEARML_DISABLED"):
            return False
        if explicit is not True and not _env_truthy("HYDRA2_CLEARML_ENABLED"):
            return False
        from clearml import Task  # noqa: F401

        return True
    except Exception:
        return False


def _filter_metrics(entry: Mapping[str, Any] | Any) -> dict[str, float]:
    """Keep allowlisted numeric series; drop everything else (never raises)."""
    if not isinstance(entry, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, value in entry.items():
        if not isinstance(key, str):
            continue
        if key not in METRIC_ALLOWLIST and not key.startswith(METRIC_ALLOWLIST_PREFIXES):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        number = float(value)
        if not math.isfinite(number):
            continue
        out[key] = number
    return out


def _flatten_params(params: Mapping[str, Any] | Any, *, _prefix: str = "") -> dict[str, str]:
    """Flatten nested mappings to ``parent.child`` string params (never raises)."""
    if not isinstance(params, Mapping):
        return {}
    flat: dict[str, str] = {}
    for key, value in params.items():
        name = str(key) if _prefix == "" else f"{_prefix}.{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten_params(value, _prefix=name))
        else:
            # Non-mapping leaf: static type degrades to Unknown here; object is honest.
            leaf: object = value
            flat[name] = str(leaf)
    return flat


def _manifest_tags(
    manifest_hashes: Mapping[str, Any] | Any, environment_digest: str | None
) -> dict[str, str]:
    """Manifest digests as ``manifest.<key>`` tags plus the environment digest."""
    tags: dict[str, str] = {}
    if isinstance(manifest_hashes, Mapping):
        for key, value in manifest_hashes.items():
            if isinstance(key, str) and isinstance(value, str) and value != "":
                tags[f"manifest.{key}"] = value
    if isinstance(environment_digest, str) and environment_digest != "":
        tags["environment.digest"] = environment_digest
    return tags


def _ensure_store_dir(path: Path | str) -> None:
    """Best-effort ``mkdir -p`` for the offline dir; swallows all errors."""
    with contextlib.suppress(Exception):
        Path(path).mkdir(parents=True, exist_ok=True)


class ClearmlMirror:
    """Observer-only ClearML sink; every backend call is warn-only on failure."""

    def __init__(
        self,
        *,
        enabled: bool,
        offline_dir: Path | str | None = None,
        project: str = EXPERIMENT_DEFAULT,
        task_name: str | None = None,
        run_name: str | None = None,
        manifest_hashes: Mapping[str, Any] | None = None,
        loop_config: Mapping[str, Any] | None = None,
        environment_digest: str | None = None,
    ) -> None:
        try:
            self._enabled = enabled
        except Exception:
            self._enabled = False
        try:
            self._offline_dir = (
                Path(offline_dir) if offline_dir is not None else default_offline_dir()
            )
        except Exception:
            self._offline_dir = default_offline_dir()
        try:
            self._project = project if project != "" else EXPERIMENT_DEFAULT
        except Exception:
            self._project = EXPERIMENT_DEFAULT
        try:
            name = task_name if task_name is not None and task_name != "" else run_name
            self._task_name = name if name is not None and name != "" else TASK_DEFAULT
        except Exception:
            self._task_name = TASK_DEFAULT
        try:
            self._manifest_hashes = (
                {k: str(v) for k, v in dict(manifest_hashes).items()}
                if isinstance(manifest_hashes, Mapping)
                else {}
            )
        except Exception:
            self._manifest_hashes = {}
        try:
            self._loop_config = dict(loop_config) if isinstance(loop_config, Mapping) else {}
        except Exception:
            self._loop_config = {}
        try:
            self._environment_digest = (
                environment_digest
                if environment_digest is not None and environment_digest != ""
                else None
            )
        except Exception:
            self._environment_digest = None
        self._task: Any = None
        self._task_id: str | None = None

    def _degraded(self, op: str, exc: BaseException) -> None:
        self._enabled = False
        warnings.warn(
            f"clearml mirror degraded ({op}): {exc.__class__.__name__}: {exc}", stacklevel=2
        )

    def start_run(self) -> str | None:
        """Create (once) the ClearML task; idempotent, never raises."""
        if not self._enabled:
            return None
        if self._task_id is not None:
            return self._task_id
        try:
            from clearml import Task

            _ensure_store_dir(self._offline_dir)
            if _env_truthy("CLEARML_OFFLINE_MODE") and hasattr(Task, "set_offline"):
                Task.set_offline(True)
                if "CLEARML_CACHE_DIR" not in os.environ:
                    os.environ["CLEARML_CACHE_DIR"] = str(self._offline_dir)
            tags = [
                f"{key}={value}"
                for key, value in _manifest_tags(
                    self._manifest_hashes, self._environment_digest
                ).items()
            ]
            task: Any = Task.init(
                project_name=self._project,
                task_name=self._task_name,
                tags=tags,
                reuse_last_task_id=False,
                auto_connect_arg_parser=False,
                auto_connect_frameworks=False,
            )
            params = _flatten_params(self._loop_config)
            if len(params) > 0:
                task.connect(params)
            raw_id: object = task.id
            self._task = task
            self._task_id = str(raw_id) if raw_id is not None else None
            return self._task_id
        except Exception as exc:
            self._degraded("start_run", exc)
            return None

    def log_update(self, entry: Mapping[str, Any], *, step: int) -> None:
        """Report one allowlisted scalar series per key at ``iteration=step``."""
        if not self._enabled or self._task is None:
            return
        try:
            metrics = _filter_metrics(entry)
            if len(metrics) == 0:
                return
            logger: _TaskLogger = self._task.get_logger()
            for key, value in metrics.items():
                logger.report_scalar(title=key, series=key, value=value, iteration=step)
        except Exception as exc:
            self._degraded("log_update", exc)

    def log_checkpoint(
        self,
        *,
        checkpoint_path: Path | str,
        manifest_json: Mapping[str, Any] | None = None,
    ) -> None:
        """Upload the checkpoint file plus its manifest snapshot as artifacts."""
        if not self._enabled or self._task is None:
            return
        try:
            path = Path(checkpoint_path)
            self._task.upload_artifact(f"checkpoints/{path.name}", path.as_posix())
            if manifest_json is not None and isinstance(manifest_json, Mapping):
                self._task.upload_artifact(f"manifests/{path.stem}.json", dict(manifest_json))
        except Exception as exc:
            self._degraded("log_checkpoint", exc)

    def log_eval_report(
        self, name: str, report: Mapping[str, Any], *, digest: str | None = None
    ) -> None:
        """Upload the eval JSON, mirror allowlisted scalars, tag the digest."""
        if not self._enabled or self._task is None:
            return
        try:
            label = name
            payload = dict(report) if isinstance(report, Mapping) else {}
            self._task.upload_artifact(f"eval/{label}.json", payload)
            logger: _TaskLogger = self._task.get_logger()
            for key, value in _filter_metrics(payload).items():
                logger.report_scalar(title=f"eval/{label}", series=key, value=value, iteration=0)
            if digest is not None and digest != "":
                self._task.add_tags([f"eval.{label}.digest={digest}"])
                logger.report_text(f"eval {label} digest {digest}")
        except Exception as exc:
            self._degraded("log_eval_report", exc)

    def log_promotion(self, record_json: Mapping[str, Any], *, digest: str | None = None) -> None:
        """Upload the promotion record plus the CI triple as scalars."""
        if not self._enabled or self._task is None:
            return
        try:
            record = dict(record_json) if isinstance(record_json, Mapping) else {}
            self._task.upload_artifact("promotion/record.json", record)
            logger: _TaskLogger = self._task.get_logger()
            for key in _PROMOTION_METRIC_KEYS:
                value = record.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                number = float(value)
                if not math.isfinite(number):
                    continue
                logger.report_scalar(title="promotion", series=key, value=number, iteration=0)
            if digest is not None and digest != "":
                self._task.add_tags([f"promotion.digest={digest}"])
                logger.report_text(f"promotion digest {digest}")
        except Exception as exc:
            self._degraded("log_promotion", exc)

    def log_duplicate_audit(
        self,
        *,
        manifest_digest: str | None = None,
        telemetry_digest: str | None = None,
        sidecar: Mapping[str, Any] | None = None,
    ) -> None:
        """Tag duplicate-wall digests and upload the confirmation sidecar.

        Duplicate wall: the dedup gate proving no training game repeats
        (manifest + telemetry digests); the sidecar is its evidence JSON.
        """
        if not self._enabled or self._task is None:
            return
        try:
            tags: list[str] = []
            if manifest_digest is not None and manifest_digest != "":
                tags.append(f"duplicate.manifest_digest={manifest_digest}")
            if telemetry_digest is not None and telemetry_digest != "":
                tags.append(f"duplicate.telemetry_digest={telemetry_digest}")
            if len(tags) > 0:
                self._task.add_tags(tags)
            if sidecar is not None and isinstance(sidecar, Mapping):
                self._task.upload_artifact("duplicate/confirmation_sidecar.json", dict(sidecar))
            self._task.get_logger().report_text(
                f"duplicate audit manifest={manifest_digest} telemetry={telemetry_digest}"
            )
        except Exception as exc:
            self._degraded("log_duplicate_audit", exc)

    def close(self) -> None:
        """Close the task; idempotent, warn-only, never raises."""
        task, self._task = self._task, None
        self._task_id = None
        if task is None:
            return
        try:
            task.close()
        except Exception as exc:
            warnings.warn(
                f"clearml mirror degraded (close): {exc.__class__.__name__}: {exc}", stacklevel=2
            )


class NullMirror(ClearmlMirror):
    """Disabled mirror: every method is a cheap no-op that never imports clearml."""

    def __init__(self) -> None:
        super().__init__(enabled=False)


def make_mirror(**kwargs: Any) -> ClearmlMirror:
    """Build an enabled mirror when opted in, else a :class:`NullMirror`.

    Never raises: any misconfiguration (bad kwargs, missing SDK, offline
    failure) falls back to a disabled mirror.
    """
    try:
        enabled = kwargs.pop("enabled", None)
        if not is_enabled(explicit=enabled):
            return NullMirror()
        return ClearmlMirror(enabled=True, **kwargs)
    except Exception:
        return NullMirror()
