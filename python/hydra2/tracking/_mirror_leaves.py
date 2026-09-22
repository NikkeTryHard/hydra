"""Tracking mirror record-shape leaves (shared pure helpers).

Single home for allowlists, env gates, metric filters, param flattening,
manifest tags, base-URL resolution, promotion and eval scalars, and duplicate
tags. Bridge record-shape leaves decide first when built; bodies below are
byte-identical stale-`.so` fallbacks. Transport, file fallback, clocks, and
run registries stay in the mirrors. Failure mode is never-raise (empty
containers on bad input), never silent training feedback.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
except ImportError:  # pragma: no cover
    _bridge_contracts = None  # type: ignore[assignment]


#: ClearML project name (auto-created server-side on first use). Single
#: source: ``hydra2._native.contracts.TRACKING_EXPERIMENT_DEFAULT``; the
#: literal is the stale-``.so`` fallback.
_EXPERIMENT_RAW: str | None = getattr(_bridge_contracts, "TRACKING_EXPERIMENT_DEFAULT", None)
EXPERIMENT_DEFAULT: str = _EXPERIMENT_RAW if _EXPERIMENT_RAW is not None else "hydra2-tenhou-4p"

#: Fallback ClearML task name when the caller supplies neither task nor run name.
_TASK_RAW: str | None = getattr(_bridge_contracts, "TRACKING_TASK_DEFAULT", None)
TASK_DEFAULT: str = _TASK_RAW if _TASK_RAW is not None else "hydra2-training"

#: Scalar keys allowed into ClearML. ``global_update`` travels as the
#: report iteration, never as a series. Same key set as the MLflow mirror,
#: which shares :func:`_filter_metrics` (placement/value/event/belief heads).
_ALLOWLIST_RAW: frozenset[str] | None = getattr(
    _bridge_contracts, "TRACKING_METRIC_ALLOWLIST", None
)
METRIC_ALLOWLIST: frozenset[str] = (
    _ALLOWLIST_RAW
    if _ALLOWLIST_RAW is not None
    else frozenset(
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
)

#: Per-head series (``event_<head>`` / ``belief_<head>``), flattened
#: per-type scorecards (``per_type/<kind>/<metric>``), and post-hoc
#: calibration scalars (``temperature`` / ``calibrated_*``) pass the
#: allowlist.  Exact keys above cover the known scalars; the prefixes
#: future-proof eval-report additions under the same families.
_PREFIXES_RAW: tuple[str, ...] | None = getattr(
    _bridge_contracts, "TRACKING_METRIC_ALLOWLIST_PREFIXES", None
)
METRIC_ALLOWLIST_PREFIXES: tuple[str, ...] = (
    _PREFIXES_RAW
    if _PREFIXES_RAW is not None
    else ("event_", "belief_", "per_type/", "calibrated_", "temperature")
)

#: Promotion record keys mirrored as scalars alongside the record artifact.
_PROMOTION_RAW: tuple[str, ...] | None = getattr(
    _bridge_contracts, "TRACKING_PROMOTION_METRIC_KEYS", None
)
_PROMOTION_METRIC_KEYS: tuple[str, ...] = (
    _PROMOTION_RAW if _PROMOTION_RAW is not None else ("observed_estimate", "ci_lo", "ci_hi")
)

#: Closed-loopback default: connection-refused fast, hermetic, never binds.
_BASE_URL_RAW: str | None = getattr(_bridge_contracts, "TRACKING_DEFAULT_BASE_URL", None)
_DEFAULT_BASE_URL: str = _BASE_URL_RAW if _BASE_URL_RAW is not None else "http://127.0.0.1:9"
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
    gate = getattr(_bridge_contracts, "tracking_filter_metrics", None)
    if gate is not None:
        with contextlib.suppress(TypeError, ValueError):
            filtered: dict[str, float] = gate(entry)
            return filtered
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
    gate = getattr(_bridge_contracts, "tracking_flatten_params", None)
    if gate is not None and _prefix == "":
        with contextlib.suppress(TypeError, ValueError):
            flattened: dict[str, str] = gate(params)
            return flattened
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
    gate = getattr(_bridge_contracts, "tracking_manifest_tags", None)
    if gate is not None:
        with contextlib.suppress(TypeError, ValueError):
            tagged: dict[str, str] = gate(manifest_hashes, environment_digest)
            return tagged
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
            _written: int = handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")
    except Exception:
        pass


def _resolve_base_url(explicit: Any | None, env_name: str) -> str:
    """Explicit kwarg, then ``env_name``, else the closed-loopback default (never raises)."""
    gate = getattr(_bridge_contracts, "tracking_resolve_base_url", None)
    if gate is not None:
        with contextlib.suppress(TypeError, ValueError):
            resolved: str = gate(explicit, os.environ.get(env_name))
            return resolved
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


def _promotion_scalars(record_json: Mapping[str, Any] | Any) -> dict[str, float]:
    """Promotion CI triple as ``promotion_`` scalars (never raises)."""
    gate = getattr(_bridge_contracts, "tracking_promotion_scalars", None)
    if gate is not None:
        with contextlib.suppress(TypeError, ValueError):
            promoted: dict[str, float] = gate(record_json)
            return promoted
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
    return scalars


def _eval_metrics(label: str, report: Mapping[str, Any] | Any) -> dict[str, float]:
    """Allowlisted eval scalars rendered as ``eval_<label>_<key>`` (never raises)."""
    gate = getattr(_bridge_contracts, "tracking_eval_metrics_for", None)
    if gate is not None:
        with contextlib.suppress(TypeError, ValueError):
            evaluated: dict[str, float] = gate(label, report)
            return evaluated
    payload = dict(report) if isinstance(report, Mapping) else {}
    return {f"eval_{label}_{key}": value for key, value in _filter_metrics(payload).items()}


def _duplicate_audit_tags(manifest_digest: str | None, telemetry_digest: str | None) -> list[str]:
    """Duplicate-wall digests as ``duplicate.<kind>=<digest>`` tags (never raises)."""
    gate = getattr(_bridge_contracts, "tracking_duplicate_audit_tags", None)
    if gate is not None:
        with contextlib.suppress(TypeError, ValueError):
            audited: list[str] = gate(manifest_digest, telemetry_digest)
            return audited
    tags: list[str] = []
    if manifest_digest is not None and manifest_digest != "":
        tags.append(f"duplicate.manifest_digest={manifest_digest}")
    if telemetry_digest is not None and telemetry_digest != "":
        tags.append(f"duplicate.telemetry_digest={telemetry_digest}")
    return tags
