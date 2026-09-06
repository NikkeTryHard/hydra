"""Sampled kernel mode (SPEC 14.3.1, PR2): non-exhaustive versioned sibling.

Beside (never replacing) :meth:`NaturalPacketKernel.enumerate_next`: per
(parent, action), a frozen count L of draws from the SAME frame law (zero new
policy semantics). Each draw carries its packet, successor refs, raw weight
``P(packet)/L``, and provenance tagged with the mode string.

NO finite-sample mass-one claim: batch raw masses fluctuate around one
(renormalizing does not recover an exact partition). Unsampled support is
unobserved, never zero-probability. Exhaustive enumeration remains the only
WP-09A certificate path; sampled batches carry the mode string in every
downstream key so the modes never mix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hydra2.belief.kernel import NaturalPacketKernel
from hydra2.contracts.common import ContractError, StaleBeliefError

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "SAMPLED_KERNEL_MODE",
    "SampledKernelConfig",
    "SampledSuccessor",
    "enumerate_sampled",
]

#: Frozen mode identity. Every sampled batch and downstream key binds this string.
SAMPLED_KERNEL_MODE = "natural_trace_sample_v1"


@dataclass(frozen=True, slots=True)
class SampledKernelConfig:
    """Frozen sampled-mode hyper-parameters."""

    samples_per_parent_action: int
    kernel_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        if (
            not isinstance(self.samples_per_parent_action, int)
            or isinstance(self.samples_per_parent_action, bool)
            or self.samples_per_parent_action <= 0
        ):
            raise ContractError("samples_per_parent_action must be a positive int")
        if (
            not isinstance(self.kernel_tolerance, float)
            or not math.isfinite(self.kernel_tolerance)
            or not 0 < self.kernel_tolerance < 0.01
        ):
            raise ContractError("kernel_tolerance must be a float in (0, 0.01)")


@dataclass(frozen=True, slots=True)
class SampledSuccessor:
    """One sampled trace: packet + successor refs + raw weight + provenance."""

    packet: Any
    successor_world_ref: str
    successor_delta: str
    raw_weight: float
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.successor_world_ref, str) or self.successor_world_ref == "":
            raise ContractError("successor_world_ref must be a non-empty str")
        if not isinstance(self.successor_delta, str) or self.successor_delta == "":
            raise ContractError("successor_delta must be a non-empty str")
        if (
            not isinstance(self.raw_weight, float)
            or not math.isfinite(self.raw_weight)
            or self.raw_weight < 0
        ):
            raise ContractError("raw_weight must be a finite nonnegative float")
        mode = self.provenance.get("mode", None)
        if mode != SAMPLED_KERNEL_MODE:
            raise ContractError(f"provenance mode must be {SAMPLED_KERNEL_MODE!r}")


def _frame_probability(successor: Any) -> float:
    prob = float(getattr(successor, "probability", 0.0))
    if not math.isfinite(prob) or prob < 0:
        raise ContractError("frame successor probability must be finite nonnegative")
    return prob


def enumerate_sampled(
    *,
    epoch: Any,
    particle: Any,
    action: Any,
    policy_set: Any | None = None,
    config: SampledKernelConfig | None = None,
    kernel: NaturalPacketKernel | None = None,
    rng: Any,
) -> tuple[SampledSuccessor, ...]:
    """Draw L traces from the exhaustive frame law (SPEC 14.3.1).

    Stale checks mirror :meth:`NaturalPacketKernel.enumerate_next`. Draws use
    ``rng.random_float()`` over the frame categorical; draw ``i`` of packet
    ``e`` carries ``raw_weight = P(e) / L``. Total mass fluctuates by design.
    Deterministic in (epoch, particle, action, L, rng stream).
    """
    cfg: SampledKernelConfig = (
        config if config is not None else SampledKernelConfig(samples_per_parent_action=1)
    )
    if not isinstance(cfg, SampledKernelConfig):
        raise ContractError("config must be SampledKernelConfig")
    if int(getattr(particle, "epoch", -1)) != int(getattr(epoch, "epoch", -2)):
        raise StaleBeliefError("particle epoch stale for sampled kernel")
    if getattr(particle, "target_id", None) != getattr(epoch, "target_id", None):
        raise StaleBeliefError("particle target stale for sampled kernel")
    if getattr(particle, "world_ref", None) is None:
        raise ContractError("particle world_ref missing")
    if rng is None or not hasattr(rng, "random_float"):
        raise ContractError("rng with random_float is required")
    out: list[SampledSuccessor] = []
    frame_kernel = kernel if kernel is not None else NaturalPacketKernel()
    frame = frame_kernel.enumerate_next(
        epoch=epoch, particle=particle, action=action, policy_set=policy_set
    )
    if len(frame) == 0:
        raise ContractError("frame kernel returned no successors")
    probs = [_frame_probability(s) for s in frame]
    total = math.fsum(probs)
    if not math.isfinite(total) or total <= 0:
        raise ContractError("frame kernel total mass must be positive finite")
    draws = cfg.samples_per_parent_action
    for _ in range(draws):
        u = float(rng.random_float())
        if not 0.0 <= u < 1.0:
            raise ContractError("rng.random_float must lie in [0, 1)")
        cumulative = 0.0
        chosen = frame[-1]
        for successor, prob in zip(frame, probs, strict=True):
            cumulative += prob / total
            if u < cumulative:
                chosen = successor
                break
        packet = getattr(chosen, "packet", None)
        if packet is None:
            raise ContractError("frame successor must carry a packet")
        out.append(
            SampledSuccessor(
                packet=packet,
                successor_world_ref=str(getattr(chosen, "successor_world_ref", "")),
                successor_delta=str(
                    getattr(chosen, "delta_ref", getattr(chosen, "successor_delta", ""))
                ),
                raw_weight=_frame_probability(chosen) / draws,
                provenance={
                    "mode": SAMPLED_KERNEL_MODE,
                    "samples_per_parent_action": draws,
                    "frame_mass": total,
                },
            )
        )
    return tuple(out)
