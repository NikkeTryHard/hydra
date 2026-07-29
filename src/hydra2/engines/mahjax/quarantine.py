"""MahJax adapter qualification state machine (BUILD WP-03C checklist item 3).

The adapter boots :attr:`AdapterState.QUARANTINED` and may transition to
:data:`AdapterState.QUALIFIED` ONLY via a
:class:`QualificationToken` bound to the complete environment tuple:
mahjax origin SHA, pixi lock hash, JAX/jaxlib versions, XLA flags, device
fingerprints, observation mode, rules id, and adapter version.

Token *issuance* is NOT implemented here — WP-04C issues real tokens after a
differential pass against the RiichiEnv reference. The only fabrication path
is :func:`fabricate_test_only_token`, an internal API marked test-only so the
gate logic itself stays provable before WP-04C exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from hydra2._canon import sha256_digest_of_json
from hydra2.contracts.common import DigestText, SchemaVersion

#: Adapter identity bound into every token.
ADAPTER_VERSION = SchemaVersion("1.0.0")

#: Observation mode served by this engine family (SPEC §8 canonical boundary).
OBSERVATION_MODE = "canonical_v1"


class AdapterState(Enum):
    """Lifecycle of the MahJax shell."""

    QUARANTINED = "QUARANTINED"
    QUALIFIED = "QUALIFIED"


@dataclass(frozen=True, slots=True)
class DeviceFingerprintToken:
    """Token-side mirror of a captured device fingerprint."""

    kind: str
    name: str | None
    compute_capability: str | None


def _environment_fragment(
    *,
    mahjax_commit_id: str,
    pixi_lock_sha256: str,
    jax_version: str,
    jaxlib_version: str,
    backend_platform: str,
    xla_flags: tuple[tuple[str, str], ...],
    devices: tuple[DeviceFingerprintToken, ...],
    python_implementation: str,
    python_version: str,
) -> dict[str, Any]:
    return {
        "backend_platform": backend_platform,
        "devices": [
            {
                "compute_capability": device.compute_capability,
                "kind": device.kind,
                "name": device.name,
            }
            for device in devices
        ],
        "jax_version": jax_version,
        "jaxlib_version": jaxlib_version,
        "mahjax_commit_id": mahjax_commit_id,
        "pixi_lock_sha256": pixi_lock_sha256,
        "python_implementation": python_implementation,
        "python_version": python_version,
        "xla_flags": dict(xla_flags),
    }


@dataclass(frozen=True, slots=True)
class QualificationToken:
    """Qualification token bound to a complete MahJax environment tuple.

    A token is an inert value object: it authorizes nothing by existing. The
    shell accepts it only when every bound dimension equals the live capture
    at qualification time (and re-checks at each consumption attempt).
    """

    adapter_version: SchemaVersion
    rules_id: DigestText
    observation_mode: str
    mahjax_commit_id: str
    pixi_lock_sha256: str
    jax_version: str
    jaxlib_version: str
    backend_platform: str
    xla_flags: tuple[tuple[str, str], ...]
    devices: tuple[DeviceFingerprintToken, ...]
    python_implementation: str
    python_version: str

    def environment_fragment(self) -> dict[str, Any]:
        """Environment half of the binding (compared field-by-field)."""
        return _environment_fragment(
            mahjax_commit_id=self.mahjax_commit_id,
            pixi_lock_sha256=self.pixi_lock_sha256,
            jax_version=self.jax_version,
            jaxlib_version=self.jaxlib_version,
            backend_platform=self.backend_platform,
            xla_flags=self.xla_flags,
            devices=self.devices,
            python_implementation=self.python_implementation,
            python_version=self.python_version,
        )

    def to_fragment(self) -> dict[str, Any]:
        """Complete token identity including engine-binding dimensions."""
        return {
            **self.environment_fragment(),
            "adapter_version": str(self.adapter_version),
            "observation_mode": self.observation_mode,
            "rules_id": str(self.rules_id),
        }

    @property
    def identity_digest(self) -> DigestText:
        """sha256 over canonical bytes of the complete token fragment."""
        return sha256_digest_of_json(self.to_fragment())


def fabricate_test_only_token(
    capture: Any,
    *,
    rules_id: DigestText,
    observation_mode: str = OBSERVATION_MODE,
    adapter_version: SchemaVersion = ADAPTER_VERSION,
) -> QualificationToken:
    """Build a token from a live capture. **TEST-ONLY internal API.**

    Real issuance lives in WP-04C behind the differential pass; this helper
    exists solely so unit tests can prove gate logic (accept/reject paths)
    before issuance exists. NEVER use it to qualify production consumption.
    """
    from hydra2.engines.mahjax.capture import DeviceFingerprint

    devices: tuple[DeviceFingerprintToken, ...] = tuple(
        DeviceFingerprintToken(
            kind=device.kind,
            name=device.name,
            compute_capability=device.compute_capability,
        )
        if isinstance(device, DeviceFingerprint)
        else cast("DeviceFingerprintToken", device)
        for device in cast("Any", capture.devices)  # pyrefly: ignore[explicit-any]
    )
    mahjax_commit_id: str = cast("str", capture.mahjax_commit_id)  # pyrefly: ignore[explicit-any]
    pixi_lock_sha256: str = str(cast("Any", capture.pixi_lock_sha256))  # pyrefly: ignore[explicit-any]
    jax_version: str = cast("str", capture.jax_version)  # pyrefly: ignore[explicit-any]
    jaxlib_version: str = cast("str", capture.jaxlib_version)  # pyrefly: ignore[explicit-any]
    backend_platform: str = cast("str", capture.backend_platform)  # pyrefly: ignore[explicit-any]
    xla_flags: tuple[tuple[str, str], ...] = cast(
        "tuple[tuple[str, str], ...]", tuple(cast("Any", capture.xla_flags))
    )  # pyrefly: ignore[explicit-any]
    python_implementation: str = cast("str", capture.python_implementation)  # pyrefly: ignore[explicit-any]
    python_version: str = cast("str", capture.python_version)  # pyrefly: ignore[explicit-any]
    return QualificationToken(
        adapter_version=adapter_version,
        rules_id=rules_id,
        observation_mode=observation_mode,
        mahjax_commit_id=mahjax_commit_id,
        pixi_lock_sha256=pixi_lock_sha256,
        jax_version=jax_version,
        jaxlib_version=jaxlib_version,
        backend_platform=backend_platform,
        xla_flags=xla_flags,
        devices=devices,
        python_implementation=python_implementation,
        python_version=python_version,
    )
