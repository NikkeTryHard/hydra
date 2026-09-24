"""WP-05B/WP-11 ClearML observer mirror over local authoritative artifacts.

Local checkpoints/manifests stay authoritative: this module only
copies allowlisted scalars, digests, and JSON snapshots for
visualization. It NEVER feeds values back into training, RNG, or sampler
state, and every method degrades to a warn-only no-op instead of raising.

Transport (M2 declared+landed): the ClearML SDK is DEAD — this module
performs NO ``import clearml`` anywhere. :meth:`ClearmlMirror.log_update`
/ :meth:`log_checkpoint` (plus the eval/promotion scalar paths) POST
already-filtered payloads through the Rust bridge REST sender
(``hydra2._native.mirror``: blocking ``reqwest`` client, rustls,
timeout 5s, retries 0 — the observer must not stall train), then ALWAYS
append the same payload to the warn-only file fallback
(``<offline_dir>/mirror.jsonl`` + ``manifests/<stem>.json``; tensor bytes
never cross — checkpoints travel as file name + manifest snapshot only).
A failed POST against a configured endpoint degrades to ``warn`` + file
record, never a raise and never a disable; the hermetic default endpoint
(closed loopback, no server configured) records silently — the file
fallback IS the mode there (only unexpected local errors use ``_degraded``).
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

Bridge: the pure record-shape leaves (``_filter_metrics`` /
``_flatten_params`` / ``_manifest_tags`` / ``_resolve_base_url`` /
``_promotion_scalars`` / ``_eval_metrics`` / ``_duplicate_audit_tags`` plus
the ``TRACKING_*`` tables on ``hydra2._native.contracts``) decide first when
the extension is built; the bodies below are the byte-identical stale-``.so``
fallback. Transport, file fallback, clocks, and run registries stay Python.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hydra2.tracking._mirror_leaves import _DEFAULT_BASE_URL as _DEFAULT_BASE_URL
from hydra2.tracking._mirror_leaves import _PROMOTION_METRIC_KEYS as _PROMOTION_METRIC_KEYS
from hydra2.tracking._mirror_leaves import _TRUTHY as _TRUTHY
from hydra2.tracking._mirror_leaves import EXPERIMENT_DEFAULT as EXPERIMENT_DEFAULT
from hydra2.tracking._mirror_leaves import METRIC_ALLOWLIST as METRIC_ALLOWLIST
from hydra2.tracking._mirror_leaves import METRIC_ALLOWLIST_PREFIXES as METRIC_ALLOWLIST_PREFIXES
from hydra2.tracking._mirror_leaves import TASK_DEFAULT as TASK_DEFAULT
from hydra2.tracking._mirror_leaves import _append_jsonl as _append_jsonl
from hydra2.tracking._mirror_leaves import _duplicate_audit_tags as _duplicate_audit_tags
from hydra2.tracking._mirror_leaves import _ensure_store_dir as _ensure_store_dir
from hydra2.tracking._mirror_leaves import _env_truthy as _env_truthy
from hydra2.tracking._mirror_leaves import _eval_metrics as _eval_metrics
from hydra2.tracking._mirror_leaves import _filter_metrics as _filter_metrics
from hydra2.tracking._mirror_leaves import _flatten_params as _flatten_params
from hydra2.tracking._mirror_leaves import _manifest_tags as _manifest_tags
from hydra2.tracking._mirror_leaves import _promotion_scalars as _promotion_scalars
from hydra2.tracking._mirror_leaves import _resolve_base_url as _resolve_base_url
from hydra2.tracking._mirror_leaves import default_offline_dir as default_offline_dir
from hydra2.tracking._mirror_leaves import is_enabled as is_enabled

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


def _bridge_mirror() -> Any | None:
    """Import the Rust REST sender lazily; ``None`` when unbuilt (fallback path)."""
    try:
        from hydra2._native import mirror as bridge_mirror  # pyrefly: ignore[missing-import]

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
            posted: bool = bridge.post_scalars(self._base_url, self._task_id, step, body)
            return posted
        except Exception:
            return False

    def _post_checkpoint(self, name: str, manifest: Mapping[str, Any]) -> bool:
        """POST the checkpoint manifest snapshot; ``False`` on failure (never raises)."""
        try:
            bridge = _bridge_mirror()
            if bridge is None or self._task_id is None:
                return False
            body = json.dumps(dict(manifest), sort_keys=True)
            saved: bool = bridge.post_checkpoint(self._base_url, self._task_id, name, body)
            return saved
        except Exception:
            return False

    def _note_transport(self, op: str, ok: bool) -> None:
        # Warn-only transport note; the file record below always lands.
        # Hermetic default endpoint (closed loopback = no server configured):
        # the file fallback IS the mode there, so a failed POST warns
        # nothing. A configured endpoint that fails warns — something is
        # actually wrong. Never raises, never disables.
        if not ok and self._base_url != _DEFAULT_BASE_URL:
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
                    _manifest_chars: int = dest.write_text(
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
            metrics = _eval_metrics(label, payload)
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
            scalars = _promotion_scalars(record)
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
            tags = _duplicate_audit_tags(manifest_digest, telemetry_digest)
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
