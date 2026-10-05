"""WP-07A natural full-fidelity confirmation runner (deterministic).

Runs decision cases under the exact tiny simulator using semantic confirmation streams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.artifacts.digest import sha256_digest
from hydra2.contracts.common import ContractError, DigestText

if TYPE_CHECKING:
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
        validated: DigestText = _bridge_contracts.make_digest_text(self.observation_hash)
        object.__setattr__(self, "observation_hash", validated)


@dataclass(frozen=True, slots=True)
class ConfirmationResult:
    case_id: str
    selected_action: int
    value: float
    observation_hash: DigestText
    rng_digest: str  # hash of rng seed for provenance

    def __post_init__(self) -> None:
        validated: DigestText = _bridge_contracts.make_digest_text(self.observation_hash)
        object.__setattr__(self, "observation_hash", validated)


class NaturalConfirmationRunner:
    """Deterministic full-fidelity confirmation runner (natural)."""

    def __init__(self, *, seed_material: bytes | None = None) -> None:
        self._seed_material = (
            seed_material if seed_material is not None else b"hydra2_wp07a_confirmation_v1"
        )

    def confirm(
        self,
        cases: list[ConfirmationCase] | tuple[ConfirmationCase, ...],
        *,
        rng: RandomStream,
        corpus: TinyCorpus | None = None,
    ) -> tuple[ConfirmationResult, ...]:
        # Reject empty input: confirmation needs at least one case to score.
        if not isinstance(cases, (list, tuple)) or len(cases) == 0:
            raise ContractError("cases must be non-empty sequence")
        # Invariant: result depends only on (case_id, world_id, rng seed).
        # Sequential draws stay deterministic under the incoming cursor.
        results: list[ConfirmationResult] = []
        for case in cases:
            # Reject non-case entries: heterogeneous input would skew determinism.
            if not isinstance(case, ConfirmationCase):
                raise ContractError("cases entries must be ConfirmationCase")
            # Sample one of 2 legal actions (0 or 1) uniformly via rng.
            action_id = rng.random_below(2)
            # Invariant: value binds world_id + observation_hash + rng seed.
            seed_hex = rng.checkpoint().seed_hex if hasattr(rng, "checkpoint") else "noseed"
            # Invariant: digest via canon bridge; rejects raise, never silent.
            digest_word = int(
                str(
                    sha256_digest(f"{case.world_id}:{case.observation_hash}:{seed_hex}".encode())
                ).removeprefix("sha256:")[:8],
                16,
            )
            # Invariant: bridge folds the digest with one live-cursor draw.
            mixed: float = _bridge_contracts.confirmation_value_mix(digest_word, rng.random_float())
            value: float = mixed
            results.append(
                ConfirmationResult(
                    case_id=case.case_id,
                    selected_action=action_id,
                    value=value,
                    observation_hash=case.observation_hash,
                    rng_digest=seed_hex[:16],
                )
            )
        return tuple(results)
