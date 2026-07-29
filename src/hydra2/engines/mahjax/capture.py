"""MahJax environment-tuple capture (BUILD WP-03C checklist item 2).

Captures the complete runtime tuple that a WP-04C qualification token binds
against: installed mahjax origin (Git commit id), pixi lock hash, JAX/jaxlib
versions, relevant XLA flags, accelerator/CPU device fingerprint, and the
Python implementation. The tuple is hashed into an ``environment fragment``
that is persisted alongside the WP-01 environment manifest.

This module never mutates :mod:`hydra2.runtime.environment`; it reuses its
helpers read-only per the WP-03C brief.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from hydra2._canon import atomic_write_bytes, canonical_json_bytes, sha256_digest_of_json
from hydra2.config import MAHJAX_GIT_URL, MAHJAX_PIN_SHA
from hydra2.contracts.common import (
    DigestText,
    QualificationRequiredError,
    make_digest_text,
)

_HEX40_RE = re.compile(r"[0-9a-f]{40}")

#: Artifact filename stored next to ``environment-manifest.json``.
FRAGMENT_ARTIFACT_NAME = "mahjax-environment-fragment.json"


def installed_origin_commit_id(*, expected_url: str = MAHJAX_GIT_URL) -> str:
    """Return the installed mahjax Git origin commit id.

    Reads ``direct_url.json`` from the installed ``mahjax`` distribution
    (setuptools/pip records the VCS origin there for Git installs). Raises
    :class:`QualificationRequiredError` when the origin is absent, not a Git
    install, points at a foreign URL, or carries a malformed commit id —
    the origin is then unverifiable and consumption stays closed.
    """
    import importlib.metadata as md

    try:
        dist = md.distribution("mahjax")
    except md.PackageNotFoundError as exc:
        raise QualificationRequiredError(
            "mahjax distribution is not installed; origin unverifiable"
        ) from exc

    direct_url_path: Path | None = None
    files: Any = dist.files
    for entry in (files if files is not None else ()):
        if str(entry).endswith("direct_url.json"):
            direct_url_path = Path(str(dist.locate_file(entry)))
            break
    if direct_url_path is None or not direct_url_path.is_file():
        raise QualificationRequiredError(
            "installed mahjax distribution lacks direct_url.json; install origin unverifiable"
        )

    try:
        payload = json.loads(direct_url_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise QualificationRequiredError(
            "installed mahjax direct_url.json is unreadable; origin unverifiable"
        ) from exc
    vcs_info: Any = payload.get("vcs_info") or {}
    url: Any = payload.get("url")
    commit_id: Any = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
    if url != expected_url:
        raise QualificationRequiredError(
            f"installed mahjax origin url {url!r} is not the pinned {expected_url!r}"
        )
    if not isinstance(commit_id, str) or _HEX40_RE.fullmatch(commit_id) is None:
        raise QualificationRequiredError("installed mahjax origin carries no valid git commit_id")
    return commit_id


def verify_installed_origin(*, pin_sha: str = MAHJAX_PIN_SHA) -> str:
    """Verify the installed mahjax origin commit id against ``pin_sha``.

    Returns the verified commit id; raises :class:`QualificationRequiredError`
    on any mismatch (BUILD WP-03C checklist item 1: runtime SHA verification).
    """
    commit_id = installed_origin_commit_id()
    if commit_id != pin_sha:
        raise QualificationRequiredError(
            f"installed mahjax origin commit {commit_id} != pinned {pin_sha}; construction refused"
        )
    return commit_id


@dataclass(frozen=True, slots=True)
class DeviceFingerprint:
    """One JAX-visible device reduced to safe identity fields."""

    kind: str  # "cpu" | "gpu" | other jax platform, lowercased
    name: str | None
    compute_capability: str | None


def _device_fingerprints(devices: tuple[Any, ...]) -> tuple[DeviceFingerprint, ...]:
    fingerprints: list[DeviceFingerprint] = []
    for device in devices:
        capability: Any = getattr(device, "compute_capability", None)
        if isinstance(capability, tuple) and len(capability) == 2:
            cap_tuple: tuple[Any, Any] = cast("tuple[Any, Any]", capability)
            capability_text: str | None = ".".join(str(cast("Any", part)) for part in cap_tuple)  # pyrefly: ignore[explicit-any]
        elif capability is None:
            capability_text: str | None = None
        else:
            capability_text: str | None = str(cast("Any", capability))  # pyrefly: ignore[explicit-any]
        fingerprints.append(
            DeviceFingerprint(
                kind=str(getattr(device, "platform", "unknown")).lower(),
                name=getattr(device, "device_kind", None),
                compute_capability=capability_text,
            )
        )
    return tuple(fingerprints)


def relevant_xla_flags() -> tuple[tuple[tuple[str, str], ...], dict[str, str]]:
    """Collect every ``XLA_*`` environment flag (sorted).

    Returns ``(pairs, mapping)``: the sorted pairs feed frozen dataclasses and
    token comparisons, the mapping feeds the JSON fragment. Includes flags
    such as ``XLA_PYTHON_CLIENT_PREALLOCATE`` and memory-fraction controls.
    """
    pairs = tuple(
        sorted((key, value) for key, value in os.environ.items() if key.startswith("XLA_"))
    )
    return pairs, dict(pairs)


def _lock_sha256() -> DigestText:
    # Read-only reuse of the WP-01 helper (directive: runtime.environment
    # helpers are consumed, never mutated, by this shell).
    from hydra2.runtime.environment import _pixi_lock_hash

    return make_digest_text(_pixi_lock_hash())


@dataclass(frozen=True, slots=True)
class MahJaxEnvironmentTuple:
    """The captured runtime tuple hashed into qualification bindings."""

    mahjax_commit_id: str
    mahjax_version: str
    mahjax_git_url: str
    pixi_lock_sha256: DigestText
    jax_version: str
    jaxlib_version: str
    backend_platform: str
    xla_flags: tuple[tuple[str, str], ...]
    devices: tuple[DeviceFingerprint, ...]
    python_implementation: str
    python_version: str

    @classmethod
    def capture(cls) -> MahJaxEnvironmentTuple:
        """Capture the live tuple now (imports jax/mahjax lazily)."""
        import importlib.metadata as md

        import jax

        xla_pairs, _ = relevant_xla_flags()
        versions = md.version
        return cls(
            mahjax_commit_id=verify_installed_origin(),
            mahjax_version=versions("mahjax"),
            mahjax_git_url=MAHJAX_GIT_URL,
            pixi_lock_sha256=_lock_sha256(),
            jax_version=versions("jax"),
            jaxlib_version=versions("jaxlib"),
            backend_platform=jax.default_backend(),
            xla_flags=xla_pairs,
            devices=_device_fingerprints(tuple(jax.devices())),
            python_implementation=sys.implementation.name,
            python_version=".".join(str(part) for part in sys.version_info[:3]),
        )

    def qualification_fragment(self) -> dict[str, Any]:
        """Environment half bound by a qualification token.

        Deliberately excludes descriptive-only fields (``mahjax_version``,
        ``mahjax_git_url``): the commit id is the strong origin identity.
        Shape MUST stay aligned with
        :meth:`hydra2.engines.mahjax.quarantine.QualificationToken.environment_fragment`.
        """
        return {
            "backend_platform": self.backend_platform,
            "devices": [
                {
                    "compute_capability": device.compute_capability,
                    "kind": device.kind,
                    "name": device.name,
                }
                for device in self.devices
            ],
            "jax_version": self.jax_version,
            "jaxlib_version": self.jaxlib_version,
            "mahjax_commit_id": self.mahjax_commit_id,
            "pixi_lock_sha256": str(self.pixi_lock_sha256),
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
            "xla_flags": dict(self.xla_flags),
        }

    def to_fragment(self) -> dict[str, Any]:
        """Complete JSON-safe fragment persisted beside the env manifest."""
        return {
            **self.qualification_fragment(),
            "mahjax_git_url": self.mahjax_git_url,
            "mahjax_version": self.mahjax_version,
        }

    @property
    def digest(self) -> DigestText:
        """sha256 over the canonical bytes of :meth:`to_fragment`."""
        return sha256_digest_of_json(self.to_fragment())


def capture_mahjax_tuple() -> MahJaxEnvironmentTuple:
    """Capture the current mahjax runtime tuple."""
    return MahJaxEnvironmentTuple.capture()


def write_mahjax_environment_fragment(
    destination_dir: Path | None = None,
) -> tuple[Path, DigestText]:
    """Atomically publish the tuple fragment beside the WP-01 env manifest.

    Defaults to ``<artifact_root>/environment/`` — i.e. stored *alongside*
    ``environment-manifest.json``. Returns ``(path, fragment_digest)``.
    """
    if destination_dir is None:
        from hydra2.config import artifact_root

        destination_dir = artifact_root() / "environment"
    tuple_ = capture_mahjax_tuple()
    destination = destination_dir / FRAGMENT_ARTIFACT_NAME
    atomic_write_bytes(destination, canonical_json_bytes(tuple_.to_fragment()))
    return destination, tuple_.digest
