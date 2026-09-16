"""WP-05B/WP-11 ClearML observer mirror over local authoritative artifacts.

Local checkpoints/manifests stay authoritative (D-007): this module only
copies allowlisted scalars, digests, and JSON snapshots for
visualization. It NEVER feeds values back into training, RNG, or sampler
state, and every method degrades to a warn-only no-op instead of raising.

Transport end-state (M2 declare-or-defer): the ClearML SDK transport is
DEAD — ``start_run`` / ``log_update`` / ``log_checkpoint`` /
``log_eval_report`` / ``log_promotion`` / ``log_duplicate_audit`` perform
NO SDK calls. The REST sender (Rust ``reqwest``, blocking client,
observer-only) is DECLARED but DEFERRED: no ``reqwest`` dependency is
added by this slice (minimal-goal stop conditions halt on dep adds), so
transport stays stubbed behind a TODO-gate — filtered payloads are built
exactly as before, then sunk to the warn-only file fallback
(``<offline_dir>/mirror.jsonl`` + ``manifests/<stem>.json`` beside the
checkpoint path convention). What MUST stay byte-identical and visible:

- ``METRIC_ALLOWLIST`` + ``METRIC_ALLOWLIST_PREFIXES`` + ``_filter_metrics``
  (placement/value/event/belief heads) — frozen on both sides; the
  byte-identical filter gate is the resume/ckpt/ring verify step.
- ``NullMirror`` no-op + warn-only ``_degraded`` + file fallback.

Enablement (disabled by default): ``HYDRA2_CLEARML_ENABLED=1`` opts in;
``HYDRA2_CLEARML_DISABLED=1`` is the kill-switch and wins over
everything, including an explicit ``enabled=True``. ``is_enabled`` keeps
its historical shape (opt-in flag check, never raises) but no longer
gates on an importable SDK — there is no SDK path left to gate on.

Call sites: ``SupervisedLoop`` / ``ActorLearnerReplay`` construct via
``make_mirror(manifest_hashes=..., loop_config={...})`` + ``start_run()``
and log ``log_update`` + ``log_checkpoint`` after each published
checkpoint. ``log_eval_report`` / ``log_promotion`` /
``log_duplicate_audit`` are reserved builders (covered by mirror unit
tests; no production callers yet).

m4 key-stability note: this kill is transport-only — allowlist keys,
payload shapes, and file-fallback layout are frozen; tests stay hermetic
(no localhost binds, no live server; hermetic dirs + ``mirror.jsonl``
only).
"""

from __future__ import annotations

import contextlib
import json
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
    "_append_jsonl",
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
    """Structural type for the (removed) SDK task logger (kept for shape)."""

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
    """Disabled by default; opt-in flag only (no SDK gate remains).

    ``explicit=False`` (or ``HYDRA2_CLEARML_DISABLED`` truthy) forces off and
    wins over everything. Otherwise requires ``explicit=True`` (or
    ``HYDRA2_CLEARML_ENABLED`` truthy). The historical ``import clearml``
    probe is gone with the SDK transport (M2): nothing here imports the
    SDK anymore. Never raises.
    """
    try:
        if explicit is False or _env_truthy("HYDRA2_CLEARML_DISABLED"):
            return False
        if explicit is not True and not _env_truthy("HYDRA2_CLEARML_ENABLED"):
            return False
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


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    """Best-effort JSONL append for the REST-deferred file fallback."""
    try:
        _ensure_store_dir(path.parent)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")
    except Exception:
        pass


class ClearmlMirror:
    """Observer-only ClearML sink; every backend call is warn-only on failure.

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
        """Create (once) the observer run record; idempotent, never raises.

        REST-note: allocates a local run id (no server round-trip while the
        ``reqwest`` sender is deferred) and records the run header
        (project/task/tags/params) to the file fallback.
        """
        if not self._enabled:
            return None
        if self._task_id is not None:
            return self._task_id
        try:
            _ensure_store_dir(self._offline_dir)
            tags = [
                f"{key}={value}"
                for key, value in _manifest_tags(
                    self._manifest_hashes, self._environment_digest
                ).items()
            ]
            params = _flatten_params(self._loop_config)
            self._task = {
                "project": self._project,
                "task_name": self._task_name,
                "tags": tags,
                "params": params,
            }
            self._task_id = f"offline-{abs(hash((self._project, self._task_name))) % 10**12:012d}"
            _append_jsonl(
                Path(self._offline_dir) / "mirror.jsonl",
                {"op": "start_run", "run_id": self._task_id, "task": self._task},
            )
            return self._task_id
        except Exception as exc:
            self._degraded("start_run", exc)
            return None

    def log_update(self, entry: Mapping[str, Any], *, step: int) -> None:
        """Sink one allowlisted scalar series per key at ``iteration=step``.

        REST-note: the allowlisted ``metrics`` payload is built exactly as
        the SDK path built it, then appended to the file fallback while the
        ``reqwest`` sender is deferred (see class TODO).
        """
        if not self._enabled or self._task is None:
            return
        try:
            metrics = _filter_metrics(entry)
            if len(metrics) == 0:
                return
            _append_jsonl(
                Path(self._offline_dir) / "mirror.jsonl",
                {
                    "op": "log_update",
                    "run_id": self._task_id,
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
        """Record the checkpoint file plus its manifest snapshot (file fallback).

        REST-note: while the ``reqwest`` sender is deferred this copies the
        manifest JSON beside the offline dir (never the tensor bytes) instead
        of ``upload_artifact``.
        """
        if not self._enabled or self._task is None:
            return
        try:
            path = Path(checkpoint_path)
            _append_jsonl(
                Path(self._offline_dir) / "mirror.jsonl",
                {
                    "op": "log_checkpoint",
                    "run_id": self._task_id,
                    "checkpoint": path.name,
                },
            )
            if manifest_json is not None and isinstance(manifest_json, Mapping):
                dest = Path(self._offline_dir) / "manifests" / f"{path.stem}.json"
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
        if not self._enabled or self._task is None:
            return
        try:
            label = name
            payload = dict(report) if isinstance(report, Mapping) else {}
            _append_jsonl(
                Path(self._offline_dir) / "mirror.jsonl",
                {
                    "op": "log_eval_report",
                    "run_id": self._task_id,
                    "label": label,
                    "digest": digest if digest not in (None, "") else None,
                    "metrics": _filter_metrics(payload),
                    "report": payload,
                },
            )
        except Exception as exc:
            self._degraded("log_eval_report", exc)

    def log_promotion(self, record_json: Mapping[str, Any], *, digest: str | None = None) -> None:
        """Record the promotion record plus the CI triple as scalars."""
        if not self._enabled or self._task is None:
            return
        try:
            record = dict(record_json) if isinstance(record_json, Mapping) else {}
            scalars: dict[str, float] = {}
            for key in _PROMOTION_METRIC_KEYS:
                value = record.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                number = float(value)
                if not math.isfinite(number):
                    continue
                scalars[key] = number
            _append_jsonl(
                Path(self._offline_dir) / "mirror.jsonl",
                {
                    "op": "log_promotion",
                    "run_id": self._task_id,
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
        if not self._enabled or self._task is None:
            return
        try:
            tags: list[str] = []
            if manifest_digest is not None and manifest_digest != "":
                tags.append(f"duplicate.manifest_digest={manifest_digest}")
            if telemetry_digest is not None and telemetry_digest != "":
                tags.append(f"duplicate.telemetry_digest={telemetry_digest}")
            _append_jsonl(
                Path(self._offline_dir) / "mirror.jsonl",
                {
                    "op": "log_duplicate_audit",
                    "run_id": self._task_id,
                    "tags": tags,
                    "sidecar": dict(sidecar)
                    if sidecar is not None and isinstance(sidecar, Mapping)
                    else None,
                },
            )
        except Exception as exc:
            self._degraded("log_duplicate_audit", exc)

    def close(self) -> None:
        """Close the run; idempotent, warn-only, never raises."""
        task, self._task = self._task, None
        self._task_id = None
        if task is None:
            return
        try:
            _append_jsonl(
                Path(self._offline_dir) / "mirror.jsonl",
                {"op": "close"},
            )
        except Exception as exc:
            warnings.warn(
                f"clearml mirror degraded (close): {exc.__class__.__name__}: {exc}", stacklevel=2
            )


class NullMirror(ClearmlMirror):
    """Disabled mirror: every method is a cheap no-op that never touches transport."""

    def __init__(self) -> None:
        super().__init__(enabled=False)


def make_mirror(**kwargs: Any) -> ClearmlMirror:
    """Build an enabled mirror when opted in, else a :class:`NullMirror`.

    Never raises: any misconfiguration (bad kwargs, offline
    failure) falls back to a disabled mirror.
    """
    try:
        enabled = kwargs.pop("enabled", None)
        if not is_enabled(explicit=enabled):
            return NullMirror()
        return ClearmlMirror(enabled=True, **kwargs)
    except Exception:
        return NullMirror()
