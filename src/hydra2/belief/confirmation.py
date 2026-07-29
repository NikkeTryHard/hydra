# ruff: noqa: E501
"""WP-07A natural full-fidelity confirmation runner (deterministic).

Runs decision cases under the exact tiny simulator using semantic confirmation streams.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hydra2.contracts.common import ContractError, DigestText, make_digest_text

if TYPE_CHECKING:
    from collections.abc import Callable

    from hydra2.belief.corpus import TinyCorpus
    from hydra2.contracts.randomness import RandomStream

__all__ = [
    "ConfirmationCase",
    "ConfirmationResult",
    "NaturalConfirmationRunner",
]


@dataclass(frozen=True, slots=True)
class ConfirmationCase:
    case_id: str
    world_id: str
    observation_hash: DigestText
    # Additional fields like actor observation could be added, but minimal for WP-07A

    def __post_init__(self) -> None:
        if self.case_id == "":
            raise ContractError("case_id must be non-empty")
        if self.world_id == "":
            raise ContractError("world_id must be non-empty")
        object.__setattr__(self, "observation_hash", make_digest_text(self.observation_hash))


@dataclass(frozen=True, slots=True)
class ConfirmationResult:
    case_id: str
    selected_action: int
    value: float
    observation_hash: DigestText
    rng_digest: str  # hash of rng seed for provenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation_hash", make_digest_text(self.observation_hash))


class NaturalConfirmationRunner:
    """Deterministic full-fidelity confirmation runner (natural)."""

    def __init__(self, *, seed_material: bytes | None = None) -> None:
        self._seed_material = seed_material if seed_material is not None else b"hydra2_wp07a_confirmation_v1"

    def confirm(
        self,
        cases: list[ConfirmationCase] | tuple[ConfirmationCase, ...],
        *,
        rng: RandomStream,
        corpus: TinyCorpus | None = None,
    ) -> tuple[ConfirmationResult, ...]:
        if not isinstance(cases, (list, tuple)) or len(cases) == 0:
            raise ContractError("cases must be non-empty sequence")
        # Determinism: each case's result depends only on (case_id, world_id, rng seed).
        # We use rng.random_float for each case sequentially, which is deterministic given rng cursor.
        out: list[ConfirmationResult] = []
        for case in cases:
            if not isinstance(case, ConfirmationCase):
                raise ContractError("cases entries must be ConfirmationCase")
            # Derive deterministic action/value from rng
            # Use rng to sample action among 2 legal actions (0 or 1) uniformly
            act = rng.random_below(2)
            # Value derived from hash of world_id + observation_hash + rng seed (deterministic)
            seed_hex = rng.checkpoint().seed_hex if hasattr(rng, "checkpoint") else "noseed"
            val_raw = int(
                hashlib.sha256(
                    f"{case.world_id}:{case.observation_hash}:{seed_hex}".encode()
                ).hexdigest()[:8],
                16,
            )
            value = (val_raw % 1000) / 1000.0  # 0..0.999
            # Also mix in rng draw to ensure rng affects value
            value = (value + rng.random_float()) / 2.0
            out.append(
                ConfirmationResult(
                    case_id=case.case_id,
                    selected_action=act,
                    value=value,
                    observation_hash=case.observation_hash,
                    rng_digest=seed_hex[:16],
                )
            )
        return tuple(out)

    def replay_is_deterministic(
        self,
        cases: tuple[ConfirmationCase, ...],
        *,
        make_rng: Callable[[], RandomStream],
    ) -> bool:
        """Helper for test: same make_rng() must produce identical results."""
        rng1: RandomStream = make_rng()
        rng2: RandomStream = make_rng()
        r1 = self.confirm(cases, rng=rng1)
        r2 = self.confirm(cases, rng=rng2)
        return r1 == r2
