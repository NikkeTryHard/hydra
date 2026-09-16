"""WP-14 MLflow observer mirror over local authoritative artifacts.

Local checkpoints/manifests stay authoritative (D-007): this module only
copies allowlisted scalars, digests, and JSON snapshots for
visualization. It NEVER feeds values back into training, RNG, or sampler
state, and every method degrades to a warn-only no-op instead of raising.

Transport end-state (M2 declare-or-defer): the MLflow SDK transport is
DEAD — ``start_run`` / ``log_update`` / ``log_checkpoint`` /
``log_eval_report`` / ``log_promotion`` / ``log_duplicate_audit`` perform
NO SDK calls. The REST sender (Rust ``reqwest``, blocking client,
observer-only) is DECLARED but DEFERRED: no ``reqwest`` dependency is
added by this slice (minimal-goal stop conditions halt on dep adds), so
transport stays stubbed behind a TODO-gate — filtered payloads are built
exactly as before (shared ``_filter_metrics`` / ``_flatten_params`` /
``_manifest_tags`` with the ClearML mirror, same allowlist set), then
sunk to the warn-only file fallback (``mirror/mlflow/mirror.jsonl`` +
``manifests/<stem>.json``). What MUST stay byte-identical and visible:

- ``METRIC_ALLOWLIST`` lives in :mod:`hydra2.tracking.clearml_mirror` and
  is shared here via ``_filter_metrics`` (placement/value/event/belief
  heads + ``event_/belief_/per_type/calibrated_/temperature`` prefixes) —
  frozen on both sides; the byte-identical filter gate is the
  resume/ckpt/ring verify step.
- ``NullMlflowMirror`` no-op + warn-only ``_degraded`` + file fallback.

Enablement (enabled by default): ``HYDRA2_MLFLOW_DISABLED=1`` is the
kill-switch and wins over everything, including an explicit
``enabled=True``. ``is_enabled`` keeps its historical shape (kill-switch
+ explicit flag, never raises) but no longer gates on an importable SDK
— there is no SDK path left to gate on.

Call sites: ``run_stream_training`` constructs via
``make_mirror(manifest_hashes=..., loop_config={...})`` + ``start_run()``
and logs ``log_update`` + ``log_checkpoint`` after each published
checkpoint. ``log_eval_report`` / ``log_promotion`` /
``log_duplicate_audit`` are reserved builders (covered by mirror unit
tests; no production callers yet).

m4 key-stability note: this kill is transport-only — allowlist keys,
payload shapes, and file-fallback layout are frozen; tests stay hermetic
(no localhost binds, no live server; hermetic dirs + ``mirror.jsonl``
only).
"""

import contextlib
import json
import os
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hydra2.tracking.clearml_mirror import (
    _append_jsonl,
    _ensure_store_dir,
    _env_truthy,
    _filter_metrics,
    _flatten_params,
    _manifest_tags,
)

__all__ = [
    "EXPERIMENT_DEFAULT",
    "RUN_NAME_DEFAULT",
    "MlflowMirror",
    "NullMlflowMirror",
    "default_tracking_dir",
    "is_enabled",
    "make_mirror",
]

#: MLflow experiment name (auto-created in the file store on first use).
EXPERIMENT_DEFAULT = "hydra2-tenhou-4p"

#: Fallback run name when the caller supplies none.
RUN_NAME_DEFAULT = "hydra2-training"

#: Promotion record keys mirrored as metrics alongside the record artifact.
_PROMOTION_METRIC_KEYS = ("observed_estimate", "ci_lo", "ci_hi")


def default_tracking_dir(
    *, artifact_root: Path | str | None = None, run_id: str | None = None
) -> Path:
    """Hermetic default for the MLflow file store (no mkdir side effect).

    ``HYDRA2_MLFLOW_TRACKING_DIR`` wins when set and non-empty; otherwise
    ``<artifact_root>/mirror/mlflow`` where the root honors
    ``HYDRA2_ARTIFACT_ROOT`` via :func:`hydra2.config.artifact_root`.
    ``run_id`` is accepted for call-site symmetry and ignored: one store
    holds every run of the artifact root.
    """
    override = os.environ.get("HYDRA2_MLFLOW_TRACKING_DIR")
    if override is not None and override.strip() != "":
        return Path(override.strip())
    if artifact_root is None:
        from hydra2.config import artifact_root as _artifact_root

        base = _artifact_root()
    else:
        base = Path(artifact_root)
    _ = run_id
    return base / "mirror" / "mlflow"


def is_enabled(*, explicit: bool | None = None) -> bool:
    """Enabled by default; kill-switch plus explicit flag (no SDK gate remains).

    ``explicit=False`` (or ``HYDRA2_MLFLOW_DISABLED`` truthy) forces off and
    wins over everything. The historical ``import mlflow`` probe is gone
    with the SDK transport (M2): nothing here imports the SDK anymore.
    Never raises.
    """
    try:
        if explicit is False or _env_truthy("HYDRA2_MLFLOW_DISABLED"):
            return False
        return True
    except Exception:
        return False


class MlflowMirror:
    """Observer-only MLflow sink; every backend call is warn-only on failure.

    Transport is REST-note only (M2 declare-or-defer): payloads are filtered
    exactly as the SDK path filtered them, then sunk to the file fallback.
    The ``reqwest`` REST sender is declared here and deferred —

    TODO(reqwest): replace the ``_append_jsonl`` file fallback in
    :meth:`log_update` / :meth:`log_checkpoint` with
    ``POST {base}/runs/{run}/scalars`` (+ checkpoints endpoint), blocking
    client, rustls, timeout 5s, retries 0 (observer must not stall train),
    keeping this same warn-only + file-fallback shape on every error.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        tracking_dir: Path | str | None = None,
        experiment: str = EXPERIMENT_DEFAULT,
        run_name: str | None = None,
        manifest_hashes: Mapping[str, Any] | None = None,
        loop_config: Mapping[str, Any] | None = None,
        environment_digest: str | None = None,
        system_metrics_interval: float | None = 10.0,
    ) -> None:
        try:
            self._enabled = enabled
        except Exception:
            self._enabled = False
        try:
            self._tracking_dir = (
                Path(tracking_dir) if tracking_dir is not None else default_tracking_dir()
            )
        except Exception:
            self._tracking_dir = default_tracking_dir()
        try:
            self._experiment = experiment if experiment != "" else EXPERIMENT_DEFAULT
        except Exception:
            self._experiment = EXPERIMENT_DEFAULT
        try:
            name = run_name
            self._run_name = name if name is not None and name != "" else RUN_NAME_DEFAULT
        except Exception:
            self._run_name = RUN_NAME_DEFAULT
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
        try:
            raw_interval = system_metrics_interval
            interval = float(raw_interval) if raw_interval is not None else None
            self._system_metrics_interval = (
                interval if interval is not None and interval > 0 else None
            )
        except Exception:
            self._system_metrics_interval = None
        self._run_id: str | None = None
        self._previous_tracking_uri: str | None = None
        self._client: Any = None

    def _degraded(self, op: str, exc: BaseException) -> None:
        self._enabled = False
        warnings.warn(
            f"mlflow mirror degraded ({op}): {exc.__class__.__name__}: {exc}", stacklevel=2
        )

    def _store_uri(self) -> str:
        """Hermetic file-store path inside the mirror dir (offline, no server)."""
        return f"{self._tracking_dir}/mlruns.db"

    def _fallback_path(self) -> Path:
        return Path(self._tracking_dir) / "mirror.jsonl"

    def start_run(self, *, run_id: str | None = None) -> str | None:
        """Create (once) or resume the observer run; idempotent, never raises.

        REST-note: allocates a local run id (no server round-trip while the
        ``reqwest`` sender is deferred) and records the run header
        (experiment/run/tags/params) to the file fallback.
        """
        if not self._enabled:
            return None
        if self._run_id is not None:
            return self._run_id
        try:
            _ensure_store_dir(self._tracking_dir)
            tags = dict(_manifest_tags(self._manifest_hashes, self._environment_digest))
            resume = run_id if run_id is not None and run_id != "" else None
            params = _flatten_params(self._loop_config)
            self._run_id = resume or f"offline-{abs(hash((self._experiment, self._run_name))) % 10**12:012d}"
            _append_jsonl(
                self._fallback_path(),
                {
                    "op": "start_run",
                    "run_id": self._run_id,
                    "experiment": self._experiment,
                    "run_name": self._run_name,
                    "tags": tags,
                    "params": params,
                },
            )
            return self._run_id
        except Exception as exc:
            self._degraded("start_run", exc)
            return None

    def log_update(self, entry: Mapping[str, Any], *, step: int) -> None:
        """Log one allowlisted scalar series per key at ``step`` (file fallback)."""
        if not self._enabled or self._run_id is None:
            return
        try:
            metrics = _filter_metrics(entry)
            if len(metrics) == 0:
                return
            _ = time.time()
            _append_jsonl(
                self._fallback_path(),
                {
                    "op": "log_update",
                    "run_id": self._run_id,
                    "step": step,
                    "metrics": metrics,
                },
            )
        except Exception as exc:
            self._degraded("log_update", exc)

    def log_checkpoint(
        self,
        *,
        checkpoint_path: Path | str,
        manifest_json: Mapping[str, Any] | None = None,
    ) -> None:
        """Record the checkpoint file plus its manifest snapshot (file fallback)."""
        if not self._enabled or self._run_id is None:
            return
        try:
            path = Path(checkpoint_path)
            _append_jsonl(
                self._fallback_path(),
                {
                    "op": "log_checkpoint",
                    "run_id": self._run_id,
                    "checkpoint": path.name,
                },
            )
            if manifest_json is not None and isinstance(manifest_json, Mapping):
                dest = Path(self._tracking_dir) / "manifests" / f"{path.stem}.json"
                try:
                    _ensure_store_dir(dest.parent)
                    dest.write_text(
                        json.dumps(dict(manifest_json), indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                except Exception as exc:
                    self._degraded("log_checkpoint", exc)
        except Exception as exc:
            self._degraded("log_checkpoint", exc)

    def log_eval_report(
        self, name: str, report: Mapping[str, Any], *, digest: str | None = None
    ) -> None:
        """Record the eval JSON, mirror allowlisted scalars, tag the digest."""
        if not self._enabled or self._run_id is None:
            return
        try:
            label = name
            payload = dict(report) if isinstance(report, Mapping) else {}
            _append_jsonl(
                self._fallback_path(),
                {
                    "op": "log_eval_report",
                    "run_id": self._run_id,
                    "label": label,
                    "digest": digest if digest not in (None, "") else None,
                    "metrics": _filter_metrics(payload),
                    "report": payload,
                },
            )
        except Exception as exc:
            self._degraded("log_eval_report", exc)

    def log_promotion(self, record_json: Mapping[str, Any], *, digest: str | None = None) -> None:
        """Record the promotion record plus the CI triple as metrics."""
        if not self._enabled or self._run_id is None:
            return
        try:
            record = dict(record_json) if isinstance(record_json, Mapping) else {}
            scalars: dict[str, float] = {}
            for key in _PROMOTION_METRIC_KEYS:
                value = record.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                import math

                number = float(value)
                if not math.isfinite(number):
                    continue
                scalars[key] = number
            _append_jsonl(
                self._fallback_path(),
                {
                    "op": "log_promotion",
                    "run_id": self._run_id,
                    "digest": digest if digest not in (None, "") else None,
                    "metrics": scalars,
                    "record": record,
                },
            )
        except Exception as exc:
            self._degraded("log_promotion", exc)

    def log_duplicate_audit(
        self,
        *,
        manifest_digest: str | None = None,
        telemetry_digest: str | None = None,
        sidecar: Mapping[str, Any] | None = None,
    ) -> None:
        """Tag duplicate-wall digests and record the confirmation sidecar.

        Duplicate wall: the dedup gate proving no training game repeats
        (manifest + telemetry digests); the sidecar is its evidence JSON.
        """
        if not self._enabled or self._run_id is None:
            return
        try:
            _append_jsonl(
                self._fallback_path(),
                {
                    "op": "log_duplicate_audit",
                    "run_id": self._run_id,
                    "manifest_digest": manifest_digest
                    if manifest_digest not in (None, "")
                    else None,
                    "telemetry_digest": telemetry_digest
                    if telemetry_digest not in (None, "")
                    else None,
                    "sidecar": dict(sidecar)
                    if sidecar is not None and isinstance(sidecar, Mapping)
                    else None,
                },
            )
        except Exception as exc:
            self._degraded("log_duplicate_audit", exc)

    def close(self) -> None:
        """End the run; idempotent, warn-only, never raises."""
        run_id, self._run_id = self._run_id, None
        if run_id is None:
            return
        try:
            with contextlib.suppress(Exception):
                _append_jsonl(self._fallback_path(), {"op": "close", "run_id": run_id})
            previous, self._previous_tracking_uri = self._previous_tracking_uri, None
            _ = previous
        except Exception as exc:
            warnings.warn(
                f"mlflow mirror degraded (close): {exc.__class__.__name__}: {exc}",
                stacklevel=2,
            )


class NullMlflowMirror(MlflowMirror):
    """Disabled mirror: every method is a cheap no-op that never touches transport."""

    def __init__(self) -> None:
        super().__init__(enabled=False)


def make_mirror(**kwargs: Any) -> MlflowMirror:
    """Build an enabled mirror by default, else a :class:`NullMlflowMirror`.

    Never raises: any misconfiguration (bad kwargs, store
    failure) falls back to a disabled mirror.
    """
    try:
        enabled = kwargs.pop("enabled", None)
        if not is_enabled(explicit=enabled):
            return NullMlflowMirror()
        return MlflowMirror(enabled=True, **kwargs)
    except Exception:
        return NullMlflowMirror()
