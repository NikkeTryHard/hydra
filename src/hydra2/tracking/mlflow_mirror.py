"""WP-14 MLflow observer mirror over local authoritative artifacts.

Local checkpoints/manifests stay authoritative (D-007): this module only
copies allowlisted scalars, digests, and JSON snapshots into a per-artifact-
root SQLite MLflow store for visualization. It NEVER feeds values back into
training, RNG, or sampler state, and every method degrades to a warn-only
no-op instead of raising.

Sole ``import mlflow`` owner: ``mlflow`` is imported lazily inside
:meth:`MlflowMirror.start_run` and :func:`is_enabled` only. The rest of
the codebase (including :mod:`hydra2.training.stream_train`) references
this module via a lazy :func:`make_mirror` call, so importing
:mod:`hydra2.tracking` never requires the MLflow SDK.

Enablement (enabled by default once the SDK is importable):

- ``HYDRA2_MLFLOW_DISABLED=1`` is the kill-switch and wins over
  everything, including an explicit ``enabled=True``. The test suite sets
  it (see ``tests/conftest.py``) so unit tests never touch the store.
- Otherwise the mirror is on when ``mlflow`` imports cleanly; explicit
  ``enabled=False`` (YAML ``telemetry.mlflow_enabled: false``) also
  forces off.

Store: SQLite at ``<artifact_root>/mirror/mlflow/mlruns.db`` (one file per
artifact root, offline, no server). The MLflow file-store backend is in
maintenance mode upstream and refuses new runs, so SQLite is the quiet-tier
store; artifacts land under ``<artifact_root>/mirror/mlflow/mlartifacts``.
Dependency stack (all latest-stable floats): ``mlflow-skinny`` (ships
``MlflowClient`` + the system-metrics monitor; full ``mlflow`` was rejected
— its pypi solve conflicts with the conda pyarrow pin) plus ``sqlalchemy`` /
``alembic``, which the skinny wheel omits but the SQLite store needs.
Fluent API note: the mirror uses the fluent ``mlflow.start_run`` /
``log_metric`` surface (not ``MlflowClient``) because the system-metrics
monitor only attaches to a fluent active run (``log_system_metrics=True``).
Training runs one run per process, so the process-global fluent state is
safe here; :meth:`close` restores the previous tracking URI best-effort.
System metrics (10s cadence, machine-level CPU/GPU/memory) ride the
monitor; allowlisted training scalars go through :meth:`log_update`.

Call sites: ``run_stream_training`` constructs via
``make_mirror(manifest_hashes=..., loop_config={...})`` + ``start_run()``
and logs ``log_update`` + ``log_checkpoint`` after each published
checkpoint. ``log_eval_report`` / ``log_promotion`` /
``log_duplicate_audit`` have no production callers yet (future builders).
"""

import contextlib
import os
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hydra2.tracking.clearml_mirror import (
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
    """Enabled by default; kill-switch plus a clean ``import mlflow``.

    ``explicit=False`` (or ``HYDRA2_MLFLOW_DISABLED`` truthy) forces off and
    wins over everything. Otherwise requires an importable ``mlflow``
    package with a working ``mlflow.tracking`` client import. Never raises.
    """
    try:
        if explicit is False or _env_truthy("HYDRA2_MLFLOW_DISABLED"):
            return False
        from mlflow.tracking import MlflowClient  # noqa: F401

        return True
    except Exception:
        return False


class MlflowMirror:
    """Observer-only MLflow sink; every backend call is warn-only on failure."""

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
        """SQLite tracking URI inside the mirror dir (offline, no server)."""
        return f"sqlite:///{self._tracking_dir}/mlruns.db"

    def start_run(self, *, run_id: str | None = None) -> str | None:
        """Create (once) or resume the MLflow run; idempotent, never raises."""
        if not self._enabled:
            return None
        if self._run_id is not None:
            return self._run_id
        try:
            import mlflow

            _ensure_store_dir(self._tracking_dir)
            _ensure_store_dir(self._tracking_dir / "mlartifacts")
            try:
                self._previous_tracking_uri = mlflow.get_tracking_uri()
            except Exception:
                self._previous_tracking_uri = None
            uri = self._store_uri()
            registry = f"file://{self._tracking_dir}/mlregistry"
            client = mlflow.tracking.MlflowClient(tracking_uri=uri, registry_uri=registry)
            experiment = client.get_experiment_by_name(self._experiment)
            if experiment is None:
                client.create_experiment(
                    self._experiment,
                    artifact_location=f"file://{self._tracking_dir}/mlartifacts",
                )
            self._client = client
            mlflow.set_tracking_uri(uri)
            # Registry is unused (models never registered) but the client
            # validates its URI on tracking calls; hermetic file fallback.
            mlflow.set_registry_uri(f"file://{self._tracking_dir}/mlregistry")
            mlflow.set_experiment(self._experiment)
            tags = dict(_manifest_tags(self._manifest_hashes, self._environment_digest))
            resume = run_id if run_id is not None and run_id != "" else None
            if self._system_metrics_interval is not None:
                mlflow.system_metrics.set_system_metrics_sampling_interval(
                    self._system_metrics_interval
                )
            run = mlflow.start_run(
                run_id=resume,
                run_name=self._run_name,
                tags=tags,
                log_system_metrics=self._system_metrics_interval is not None,
            )
            params = _flatten_params(self._loop_config)
            if len(params) > 0:
                mlflow.log_params(params)
            raw_id: object = run.info.run_id
            self._run_id = str(raw_id) if raw_id is not None else None
            return self._run_id
        except Exception as exc:
            self._degraded("start_run", exc)
            return None

    def log_update(self, entry: Mapping[str, Any], *, step: int) -> None:
        """Log one allowlisted scalar series per key at ``step``."""
        if not self._enabled or self._run_id is None:
            return
        try:
            import mlflow

            metrics = _filter_metrics(entry)
            if len(metrics) == 0:
                return
            client = self._client
            if client is None:
                client = mlflow.tracking.MlflowClient(tracking_uri=self._store_uri())
                self._client = client
            stamp_ms = int(time.time() * 1000)
            batch = [
                mlflow.entities.Metric(key, value, stamp_ms, step) for key, value in metrics.items()
            ]
            client.log_batch(self._run_id, metrics=batch)
        except Exception as exc:
            self._degraded("log_update", exc)

    def log_checkpoint(
        self,
        *,
        checkpoint_path: Path | str,
        manifest_json: Mapping[str, Any] | None = None,
    ) -> None:
        """Attach the checkpoint file plus its manifest snapshot as artifacts."""
        if not self._enabled or self._run_id is None:
            return
        try:
            import mlflow

            path = Path(checkpoint_path)
            mlflow.log_artifact(path.as_posix(), artifact_path="checkpoints")
            if manifest_json is not None and isinstance(manifest_json, Mapping):
                mlflow.log_dict(dict(manifest_json), f"manifests/{path.stem}.json")
        except Exception as exc:
            self._degraded("log_checkpoint", exc)

    def log_eval_report(
        self, name: str, report: Mapping[str, Any], *, digest: str | None = None
    ) -> None:
        """Attach the eval JSON, mirror allowlisted scalars, tag the digest."""
        if not self._enabled or self._run_id is None:
            return
        try:
            import mlflow

            label = name
            payload = dict(report) if isinstance(report, Mapping) else {}
            mlflow.log_dict(payload, f"eval/{label}.json")
            for key, value in _filter_metrics(payload).items():
                mlflow.log_metric(f"eval_{label}_{key}", value, step=0)
            if digest is not None and digest != "":
                mlflow.set_tag(f"eval.{label}.digest", digest)
        except Exception as exc:
            self._degraded("log_eval_report", exc)

    def log_promotion(self, record_json: Mapping[str, Any], *, digest: str | None = None) -> None:
        """Attach the promotion record plus the CI triple as metrics."""
        if not self._enabled or self._run_id is None:
            return
        try:
            import mlflow

            record = dict(record_json) if isinstance(record_json, Mapping) else {}
            mlflow.log_dict(record, "promotion/record.json")
            for key in _PROMOTION_METRIC_KEYS:
                value = record.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                import math

                number = float(value)
                if not math.isfinite(number):
                    continue
                mlflow.log_metric(f"promotion_{key}", number, step=0)
            if digest is not None and digest != "":
                mlflow.set_tag("promotion.digest", digest)
        except Exception as exc:
            self._degraded("log_promotion", exc)

    def log_duplicate_audit(
        self,
        *,
        manifest_digest: str | None = None,
        telemetry_digest: str | None = None,
        sidecar: Mapping[str, Any] | None = None,
    ) -> None:
        """Tag duplicate-wall digests and attach the confirmation sidecar."""
        if not self._enabled or self._run_id is None:
            return
        try:
            import mlflow

            if manifest_digest is not None and manifest_digest != "":
                mlflow.set_tag("duplicate.manifest_digest", manifest_digest)
            if telemetry_digest is not None and telemetry_digest != "":
                mlflow.set_tag("duplicate.telemetry_digest", telemetry_digest)
            if sidecar is not None and isinstance(sidecar, Mapping):
                mlflow.log_dict(dict(sidecar), "duplicate/confirmation_sidecar.json")
        except Exception as exc:
            self._degraded("log_duplicate_audit", exc)

    def close(self) -> None:
        """End the run; idempotent, warn-only, never raises."""
        run_id, self._run_id = self._run_id, None
        if run_id is None:
            return
        try:
            import mlflow

            with contextlib.suppress(Exception):
                mlflow.end_run(status="FINISHED")
            previous, self._previous_tracking_uri = self._previous_tracking_uri, None
            if previous is not None:
                with contextlib.suppress(Exception):
                    mlflow.set_tracking_uri(previous)
        except Exception as exc:
            warnings.warn(
                f"mlflow mirror degraded (close): {exc.__class__.__name__}: {exc}",
                stacklevel=2,
            )


class NullMlflowMirror(MlflowMirror):
    """Disabled mirror: every method is a cheap no-op that never imports mlflow."""

    def __init__(self) -> None:
        super().__init__(enabled=False)


def make_mirror(**kwargs: Any) -> MlflowMirror:
    """Build an enabled mirror by default, else a :class:`NullMlflowMirror`.

    Never raises: any misconfiguration (bad kwargs, missing SDK, store
    failure) falls back to a disabled mirror.
    """
    try:
        enabled = kwargs.pop("enabled", None)
        if not is_enabled(explicit=enabled):
            return NullMlflowMirror()
        return MlflowMirror(enabled=True, **kwargs)
    except Exception:
        return NullMlflowMirror()
