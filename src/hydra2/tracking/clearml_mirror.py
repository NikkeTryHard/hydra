"""WP-05B/WP-11 ClearML observer mirror over local authoritative artifacts.

Local checkpoints/manifests stay authoritative (D-007): this module only
copies allowlisted scalars, digests, and JSON snapshots for
visualization. It NEVER feeds values back into training, RNG, or sampler
state, and every method degrades to a warn-only no-op instead of raising.

Transport (M2 declared+landed): the ClearML SDK is DEAD — this module
performs NO ``import clearml`` anywhere. :meth:`ClearmlMirror.log_update`
/ :meth:`log_checkpoint` (plus the eval/promotion scalar paths) POST
already-filtered payloads through the Rust bridge REST sender
(``hydra2_replay_rs.mirror``: blocking ``reqwest`` client, rustls,
timeout 5s, retries 0 — the observer must not stall train), then ALWAYS
append the same payload to the warn-only file fallback
(``<offline_dir>/mirror.jsonl`` + ``manifests/<stem>.json``; tensor bytes
never cross — checkpoints travel as file name + manifest snapshot only).
Every transport failure degrades to ``warn`` + file record, never a raise
and never a disable (only unexpected local errors use ``_degraded``).
What stays frozen and visible:

- ``METRIC_ALLOWLIST`` + ``METRIC_ALLOWLIST_PREFIXES`` + ``_filter_metrics``
  (placement/value/event/belief heads) — frozen on both sides; the
  byte-identical filter gate is the resume/ckpt/ring verify step.
- ``NullMirror`` no-op + warn-only ``_degraded`` + file fallback layout.

Enablement (disabled by default): ``HYDRA2_CLEARML_ENABLED=1`` opts in;
``HYDRA2_CLEARML_DISABLED=1`` is the kill-switch and wins over
everything, including an explicit ``enabled=True``. ``is_enabled`` keeps
its historical shape (opt-in flag check, never raises) but no longer
gates on an importable SDK — there is no SDK path left to gate on.

Endpoint: ``HYDRA2_CLEARML_MIRROR_URL`` (or the ``base_url`` kwarg) wins;
otherwise a closed loopback default (``http://127.0.0.1:9``) fails fast
to the file fallback, so hermetic runs never stall and never bind.

Call sites: ``SupervisedLoop`` / ``ActorLearnerReplay`` construct via
``make_mirror(manifest_hashes=..., loop_config={...})`` + ``start_run()``
and log ``log_update`` + ``log_checkpoint`` after each published
checkpoint. ``log_eval_report`` / ``log_promotion`` /
``log_duplicate_audit`` are reserved builders (covered by mirror unit
tests; no production callers yet).

m4 key-stability note: this kill is transport-only — allowlist keys,
payload shapes, and file-fallback layout are frozen; tests stay hermetic
(no localhost binds, no live server; hermetic dirs + ``mirror.jsonl``
only, with a recording fake bridge for the REST expects).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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

#: Closed-loopback default: connection-refused fast, hermetic, never binds.
_DEFAULT_BASE_URL = "http://127.0.0.1:9"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


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
        return explicit is True or _env_truthy("HYDRA2_CLEARML_ENABLED")
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
    """Best-effort JSONL append for the warn-only file fallback (never raises)."""
    try:
        _ensure_store_dir(path.parent)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")
    except Exception:
        pass


def _resolve_base_url(explicit: Any | None, env_name: str) -> str:
    """Explicit kwarg, then ``env_name``, else the closed-loopback default (never raises)."""
    try:
        if explicit is not None and str(explicit).strip() != "":
            return str(explicit).strip()
    except Exception:
        pass
    try:
        override = os.environ.get(env_name)
        if override is not None and override.strip() != "":
            return override.strip()
    except Exception:
        pass
    return _DEFAULT_BASE_URL


def _bridge_mirror() -> Any | None:
    """Import the Rust REST sender lazily; ``None`` when unbuilt (fallback path)."""
    try:
        from hydra2_replay_rs import mirror as bridge_mirror  # pyrefly: ignore[missing-import]

        return bridge_mirror
    except Exception:
        return None


class ClearmlMirror:
    """Observer-only ClearML sink; every backend call is warn-only on failure.

    Transport is the Rust bridge REST sender (blocking ``reqwest``, rustls,
    timeout 5s, retries 0). Every log call POSTs the already-filtered
    payload, then ALWAYS appends the same payload to the file fallback —
    offline durability first, server best-effort. Transport failure warns
    and keeps the file record; it never raises and never disables the
    mirror (only unexpected local errors degrade via :meth:`_degraded`).
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
        base_url: str | None = None,
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
        try:
            self._base_url = _resolve_base_url(base_url, "HYDRA2_CLEARML_MIRROR_URL")
        except Exception:
            self._base_url = _DEFAULT_BASE_URL
        self._task: Any = None
        self._task_id: str | None = None

    def _degraded(self, op: str, exc: BaseException) -> None:
        self._enabled = False
        warnings.warn(
            f"clearml mirror degraded ({op}): {exc.__class__.__name__}: {exc}", stacklevel=2
        )

    def _fallback_path(self) -> Path:
        return Path(self._offline_dir) / "mirror.jsonl"

    def _post_scalars(self, metrics: Mapping[str, float], step: int) -> bool:
        """POST filtered scalars; ``False`` on every transport failure (never raises)."""
        try:
            bridge = _bridge_mirror()
            if bridge is None or self._task_id is None:
                return False
            body = json.dumps(dict(metrics), sort_keys=True)
            return bool(bridge.post_scalars(self._base_url, self._task_id, int(step), body))
        except Exception:
            return False

    def _post_checkpoint(self, name: str, manifest: Mapping[str, Any]) -> bool:
        """POST the checkpoint manifest snapshot; ``False`` on failure (never raises)."""
        try:
            bridge = _bridge_mirror()
            if bridge is None or self._task_id is None:
                return False
            body = json.dumps(dict(manifest), sort_keys=True)
            return bool(bridge.post_checkpoint(self._base_url, self._task_id, str(name), body))
        except Exception:
            return False

    def _note_transport(self, op: str, ok: bool) -> None:
        """Warn-only transport note; the file record below always lands."""
        if not ok:
            warnings.warn(f"clearml mirror transport fallback ({op})", stacklevel=3)

    def start_run(self) -> str | None:
        """Allocate (once) the observer run record; idempotent, never raises.

        Local-only (no server round-trip): records the run header
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
                self._fallback_path(),
                {"op": "start_run", "run_id": self._task_id, "task": self._task},
            )
            return self._task_id
        except Exception as exc:
            self._degraded("start_run", exc)
            return None

    def log_update(self, entry: Mapping[str, Any], *, step: int) -> None:
        """POST one allowlisted scalar series per key at ``step`` + file record."""
        if not self._enabled or self._task is None:
            return
        try:
            metrics = _filter_metrics(entry)
            if len(metrics) == 0:
                return
            self._note_transport("log_update", self._post_scalars(metrics, step))
            _append_jsonl(
                self._fallback_path(),
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
        """POST the checkpoint manifest snapshot + file record (never tensor bytes)."""
        if not self._enabled or self._task is None:
            return
        try:
            path = Path(checkpoint_path)
            manifest = dict(manifest_json) if isinstance(manifest_json, Mapping) else {}
            self._note_transport("log_checkpoint", self._post_checkpoint(path.name, manifest))
            _append_jsonl(
                self._fallback_path(),
                {
                    "op": "log_checkpoint",
                    "run_id": self._task_id,
                    "checkpoint": path.name,
                },
            )
            if isinstance(manifest_json, Mapping):
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
        """POST allowlisted eval scalars (``eval_<label>_<key>``) + file record."""
        if not self._enabled or self._task is None:
            return
        try:
            label = name
            payload = dict(report) if isinstance(report, Mapping) else {}
            metrics = {
                f"eval_{label}_{key}": value for key, value in _filter_metrics(payload).items()
            }
            if len(metrics) > 0:
                self._note_transport("log_eval_report", self._post_scalars(metrics, 0))
            _append_jsonl(
                self._fallback_path(),
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
        """POST the promotion CI triple as scalars + file record."""
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
                scalars[f"promotion_{key}"] = number
            if len(scalars) > 0:
                self._note_transport("log_promotion", self._post_scalars(scalars, 0))
            _append_jsonl(
                self._fallback_path(),
                {
                    "op": "log_promotion",
                    "run_id": self._task_id,
                    "digest": digest if digest not in (None, "") else None,
                    "metrics": {
                        key: scalars[f"promotion_{key}"]
                        for key in _PROMOTION_METRIC_KEYS
                        if f"promotion_{key}" in scalars
                    },
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
        """Record duplicate-wall digests + confirmation sidecar to the file fallback.

        Duplicate wall: the dedup gate proving no training game repeats
        (manifest + telemetry digests); the sidecar is its evidence JSON.
        Tag-shaped digests have no REST mapping, so the fallback is the record.
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
                self._fallback_path(),
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

    Never raises: any misconfiguration (bad kwargs, missing bridge,
    offline failure) falls back to a disabled mirror.
    """
    try:
        enabled = kwargs.pop("enabled", None)
        if not is_enabled(explicit=enabled):
            return NullMirror()
        return ClearmlMirror(enabled=True, **kwargs)
    except Exception:
        return NullMirror()
