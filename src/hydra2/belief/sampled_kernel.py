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


def _require_search_bridge() -> Any:
    """Import the built ``search`` bridge surface (fail closed)."""
    try:
        import hydra2_replay_rs as _ext  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2_replay_rs extension with search not importable; "
            "build the bridge with `pixi run build-ext` before sampling traces"
        ) from exc
    try:
        mod = _ext.search
    except AttributeError as exc:
        raise ImportError(
            "hydra2_replay_rs.search submodule missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc
    if not hasattr(mod, "sampled_draws"):
        raise ImportError(
            "hydra2_replay_rs.search.sampled_draws missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        )
    return mod


def _ctr_seed_cursor(rng: Any) -> tuple[bytes, int]:
    """Extract CTR (seed, cursor) for bridge replay (fail closed, no fallback)."""
    try:
        cp = rng.checkpoint()
        seed_hex = cp.seed_hex
        cursor = int(cp.cursor)
        seed = bytes.fromhex(str(seed_hex))
    except AttributeError as exc:
        raise ContractError(
            "rng must expose checkpoint() with seed_hex/cursor for bridge replay"
        ) from exc
    except (ValueError, TypeError) as exc:
        raise ContractError(f"rng checkpoint malformed: {exc}") from exc
    if len(seed) == 0:
        raise ContractError("rng seed must be non-empty bytes")
    if cursor < 0 or cursor > 0xFFFF_FFFF_FFFF_FFFF:
        raise ContractError(f"rng cursor out of u64 range: {cursor!r}")
    return seed, cursor


if TYPE_CHECKING:
    from collections.abc import Mapping

    from hydra2.contracts.randomness import RandomStream

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
    rng: RandomStream,
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
    # Draws via the search bridge (CTR-exact categorical over the frame law).
    # Cursor replay: seed/cursor in, end_cursor out, rng jumped so the stream
    # continues exactly. raw_weight = P(chosen)/draws comes from the bridge
    # (proven equal below). ImportError with build-ext hint, no fallback.
    seed, cursor = _ctr_seed_cursor(rng)
    search_mod = _require_search_bridge()
    try:
        chosen_idx, raw_weights, end_cursor = search_mod.sampled_draws(probs, draws, seed, cursor)
    except ImportError:
        raise
    except (ValueError, OverflowError, TypeError) as exc:
        raise ContractError(f"sampled_draws bridge rejected input: {exc}") from exc
    try:
        rng.jump_to(int(end_cursor))
    except (AttributeError, ValueError, TypeError) as exc:
        raise ContractError(f"rng jump_to failed for bridge replay: {exc}") from exc
    try:
        picks = list(chosen_idx)
        weights = list(raw_weights)
    except TypeError as exc:
        raise ContractError(f"sampled_draws bridge returned non-sequence: {exc}") from exc
    if len(picks) != draws or len(weights) != draws:
        raise ContractError(
            f"sampled_draws bridge returned {len(picks)}/{len(weights)} draws, expected {draws}"
        )
    out: list[SampledSuccessor] = []
    for pick_raw, w_raw in zip(picks, weights, strict=True):
        pick = int(pick_raw)
        if pick < 0 or pick >= len(frame):
            raise ContractError(f"bridge pick {pick!r} out of frame range")
        chosen = frame[pick]
        # Byte-identity: bridge weight must equal P(chosen)/draws exactly.
        # Bridge computes same division, so exact equality holds; mismatch
        # fail-closes (1e-15 admits float noise only, never drift).
        exp_w = _frame_probability(chosen) / draws
        if float(w_raw) != exp_w and abs(float(w_raw) - exp_w) > 1e-15:
            raise ContractError(
                f"bridge raw_weight {float(w_raw)!r} != P/draws {exp_w!r} (bridge!=oracle)"
            )
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
                raw_weight=float(w_raw),
                provenance={
                    "mode": SAMPLED_KERNEL_MODE,
                    "samples_per_parent_action": draws,
                    "frame_mass": total,
                },
            )
        )
    return tuple(out)
