"""MahJax quarantine shell (BUILD WP-03C).

Construction verifies the installed mahjax origin SHA and boots
:attr:`AdapterState.QUARANTINED`. Every trajectory/data/evaluation
consumption route fails closed with :class:`QualificationRequiredError`
until :meth:`MahJaxQuarantineShell.qualify` accepts a token bound to the
complete live environment tuple. Only the import/compile probes are legal
before qualification.
"""

from __future__ import annotations

import hashlib
import importlib
import struct
from typing import Any

from hydra2.artifacts.digest import of_bytes
from hydra2.config import MAHJAX_PIN_SHA
from hydra2.contracts.common import DigestText, QualificationRequiredError
from hydra2.engines.mahjax.capture import (
    MahJaxEnvironmentTuple,
    capture_mahjax_tuple,
    verify_installed_origin,
)
from hydra2.engines.mahjax.quarantine import (
    ADAPTER_VERSION,
    OBSERVATION_MODE,
    AdapterState,
    QualificationToken,
)


class MahJaxQuarantineShell:
    """Fail-closed shell around the pinned mahjax distribution.

    The shell never exposes engine outputs while quarantined; the only
    pre-qualification surface is the import/compile probe pair. Post
    qualification, :attr:`module` yields the imported mahjax module — the
    single root through which WP-04C's qualified adapter will operate.
    """

    def __init__(self, *, pin_sha: str = MAHJAX_PIN_SHA) -> None:
        self._pin_sha = pin_sha
        # Checklist item 1: runtime SHA verification at construction.
        self._verified_commit_id = verify_installed_origin(pin_sha=pin_sha)
        # Checklist item 3: QUARANTINED is the default state, always.
        self._state = AdapterState.QUARANTINED
        self._token: QualificationToken | None = None
        self._rules_id: DigestText | None = None

    # -- introspection ------------------------------------------------------

    @property
    def pin_sha(self) -> str:
        return self._pin_sha

    @property
    def verified_commit_id(self) -> str:
        return self._verified_commit_id

    @property
    def state(self) -> AdapterState:
        return self._state

    @property
    def qualified(self) -> bool:
        return self._state is AdapterState.QUALIFIED

    @property
    def rules_id(self) -> DigestText | None:
        return self._rules_id

    @property
    def token_identity_digest(self) -> DigestText | None:
        return self._token.identity_digest if self._token is not None else None

    # -- qualification gate -------------------------------------------------

    def qualify(self, token: QualificationToken, *, rules_id: DigestText) -> DigestText:
        """Transition QUARANTINED -> QUALIFIED via a complete-tuple token.

        Re-captures the live tuple and rejects the token unless every bound
        dimension matches now (not at some earlier moment). ``rules_id``
        binds the rules manifest identity this qualification covers.
        Returns the accepted token identity digest for records.
        """
        if not isinstance(token, QualificationToken):
            raise QualificationRequiredError(
                "qualification requires a hydra2.engines.mahjax QualificationToken"
            )
        if token.adapter_version != ADAPTER_VERSION:
            raise QualificationRequiredError(
                f"token adapter_version {token.adapter_version!r} != "
                f"shell adapter_version {ADAPTER_VERSION!r}"
            )
        if token.observation_mode != OBSERVATION_MODE:
            raise QualificationRequiredError(
                f"token observation_mode {token.observation_mode!r} != {OBSERVATION_MODE!r}"
            )
        if token.rules_id != rules_id:
            raise QualificationRequiredError(
                "token rules_id does not match the rules identity supplied at qualification"
            )
        fresh = capture_mahjax_tuple()
        if fresh.mahjax_commit_id != self._pin_sha:
            raise QualificationRequiredError(
                "installed mahjax origin drifted from the pin during qualification"
            )
        if token.environment_fragment() != fresh.qualification_fragment():
            raise QualificationRequiredError(
                "qualification token does not bind the live environment tuple"
            )
        self._token = token
        self._rules_id = rules_id
        self._state = AdapterState.QUALIFIED
        return token.identity_digest

    def require_qualified(self) -> None:
        """Fail closed unless currently qualified against the live tuple.

        Every trajectory/data/evaluation consumption route MUST call this
        first. Environment drift after qualification demotes the shell back
        to QUARANTINED and denies consumption.
        """
        if self._state is not AdapterState.QUALIFIED or self._token is None:
            raise QualificationRequiredError(
                "mahjax trajectory/data/evaluation output requires a "
                f"qualification token; adapter state={self._state.value}"
            )
        fresh = capture_mahjax_tuple()
        if self._token.environment_fragment() != fresh.qualification_fragment():
            self._state = AdapterState.QUARANTINED
            self._token = None
            self._rules_id = None
            raise QualificationRequiredError(
                "live environment tuple drifted from the bound qualification "
                "token; adapter re-quarantined and consumption denied"
            )

    @property
    def module(self):
        """The imported mahjax module; the gated root of all consumption."""
        self.require_qualified()
        return importlib.import_module("mahjax")

    # -- probes (legal before qualification) --------------------------------

    def import_probe(self) -> dict[str, Any]:
        """Import probe: may mahjax be imported from the pinned origin?

        Checklist item 4: probes are allowed in any state. Returns a
        structured verdict instead of consuming anything; a failed import is
        reported honestly as ``importable=false``.
        """
        try:
            module = importlib.import_module("mahjax")
        except ImportError as exc:
            return {"importable": False, "error_class": type(exc).__name__}
        commit_id = verify_installed_origin(pin_sha=self._pin_sha)
        del module  # presence proven; no attribute surface leaked pre-gate
        return {
            "importable": True,
            "commit_id": commit_id,
            "matches_pin": commit_id == self._pin_sha,
        }

    def compile_probe(self) -> dict[str, Any]:
        """Compile probe: jit-compile and evaluate a trivial function.

        Proves the XLA toolchain works on the live backend without touching
        any trajectory/data/evaluation surface. Allowed while quarantined.
        """
        import jax
        import jax.numpy as jnp

        def trivial(value: Any) -> Any:
            return value * 2.0 + 1.0

        compiled = jax.jit(trivial)
        result = compiled(jnp.asarray(3.0, dtype=jnp.float32))
        actual = float(result)
        digest = of_bytes(hashlib.sha256(struct.pack("<d", actual)).digest())
        devices: tuple[Any, ...] = tuple(jax.devices())  # pyrefly: ignore[explicit-any]
        device_kind: Any = getattr(devices[0], "device_kind", None) if len(devices) != 0 else None
        return {
            "ok": actual == 7.0,
            "expected": 7.0,
            "actual": actual,
            "output_digest": digest,
            "backend_platform": jax.default_backend(),
            "device_kind": device_kind,
        }

    # -- evidence -----------------------------------------------------------

    def captured_tuple(self) -> MahJaxEnvironmentTuple:
        """Capture the live tuple (checklist item 2 evidence helper)."""
        return capture_mahjax_tuple()


__all__ = [
    "ADAPTER_VERSION",
    "OBSERVATION_MODE",
    "AdapterState",
    "MahJaxQuarantineShell",
    "QualificationToken",
]
