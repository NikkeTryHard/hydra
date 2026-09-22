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

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.belief.kernel import NaturalPacketKernel
from hydra2.belief.natural import (
    _ctr_seed_cursor as _ctr_seed_cursor,
)
from hydra2.belief.natural import (
    _require_search_bridge as _require_search_bridge,
)
from hydra2.contracts.common import ContractError, StaleBeliefError

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
#: Pinned by the `SAMPLED_KERNEL_MODE` const on `hydra2._native.contracts`
#: (the literal stays here so module import never touches a not-yet-built
#: attr; the translators below read the bridge lazily).
SAMPLED_KERNEL_MODE = "natural_trace_sample_v1"


def _check_samples_per_parent_action(value: int) -> int:
    """Validate `samples_per_parent_action` via the bridge (oracle-identical text)."""
    try:
        checked: int = _bridge_contracts.sampled_check_samples(value)
        return checked
    except (ValueError, OverflowError, TypeError) as exc:
        raise ContractError("samples_per_parent_action must be a positive int") from exc


def _check_sampled_tolerance(value: float) -> float:
    """Validate `kernel_tolerance` via the bridge (oracle-identical text)."""
    try:
        tol: float = _bridge_contracts.sampled_check_tolerance(value)
        return tol
    except (ValueError, OverflowError, TypeError) as exc:
        raise ContractError("kernel_tolerance must be a float in (0, 0.01)") from exc


def _check_raw_weight(value: float) -> float:
    """Validate `raw_weight` via the bridge (oracle-identical text)."""
    try:
        weight: float = _bridge_contracts.sampled_check_raw_weight(value)
        return weight
    except (ValueError, OverflowError, TypeError) as exc:
        raise ContractError("raw_weight must be a finite nonnegative float") from exc


def _check_provenance_mode(mode: Any) -> str:
    """Validate provenance `mode` against the frozen mode via the bridge."""
    try:
        mode_text: str = _bridge_contracts.sampled_check_provenance_mode(mode)
        return mode_text
    except (ValueError, OverflowError, TypeError) as exc:
        raise ContractError(f"provenance mode must be {SAMPLED_KERNEL_MODE!r}") from exc


@dataclass(frozen=True, slots=True)
class SampledKernelConfig:
    """Frozen sampled-mode hyper-parameters."""

    samples_per_parent_action: int
    kernel_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        _samples: int = _check_samples_per_parent_action(self.samples_per_parent_action)
        _tol: float = _check_sampled_tolerance(self.kernel_tolerance)


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
        _weight: float = _check_raw_weight(self.raw_weight)
        _mode: str = _check_provenance_mode(self.provenance.get("mode", None))


def _frame_probability(successor: Any) -> float:
    # The `float(...)` conversion stays Python-side so garbage inputs raise
    # the oracle's own exception types; the finite-nonnegative gate below
    # runs on the bridge with oracle-identical text.
    prob = float(getattr(successor, "probability", 0.0))
    try:
        framed: float = _bridge_contracts.sampled_frame_probability(prob)
        return framed
    except (ValueError, OverflowError, TypeError) as exc:
        raise ContractError("frame successor probability must be finite nonnegative") from exc


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
    particle_epoch: int = getattr(particle, "epoch", -1)
    epoch_epoch: int = getattr(epoch, "epoch", -2)
    if particle_epoch != epoch_epoch:
        raise StaleBeliefError("particle epoch stale for sampled kernel")
    particle_target: object = getattr(particle, "target_id", None)
    epoch_target: object = getattr(epoch, "target_id", None)
    if particle_target != epoch_target:
        raise StaleBeliefError("particle target stale for sampled kernel")
    particle_world: object = getattr(particle, "world_ref", None)
    if particle_world is None:
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
    search_mod = _require_search_bridge(need="sampled_draws", purpose="sampling traces")
    try:
        draw_out: tuple[list[int], list[float], int] = search_mod.sampled_draws(
            probs, draws, seed, cursor
        )
    except ImportError:
        raise
    except (ValueError, OverflowError, TypeError) as exc:
        raise ContractError(f"sampled_draws bridge rejected input: {exc}") from exc
    chosen_idx, raw_weights, end_cursor = draw_out
    try:
        rng.jump_to(end_cursor)
    except (AttributeError, ValueError, TypeError) as exc:
        raise ContractError(f"rng jump_to failed for bridge replay: {exc}") from exc
    try:
        picks: list[int] = list(chosen_idx)
        weights: list[float] = list(raw_weights)
    except TypeError as exc:
        raise ContractError(f"sampled_draws bridge returned non-sequence: {exc}") from exc
    if len(picks) != draws or len(weights) != draws:
        raise ContractError(
            f"sampled_draws bridge returned {len(picks)}/{len(weights)} draws, expected {draws}"
        )
    out: list[SampledSuccessor] = []
    for pick_raw, w_raw in zip(picks, weights, strict=True):
        pick: int = pick_raw
        if pick < 0 or pick >= len(frame):
            raise ContractError(f"bridge pick {pick!r} out of frame range")
        chosen = frame[pick]
        # Byte-identity: bridge weight must equal P(chosen)/draws exactly.
        # The fold and the 1e-15 predicate run on the bridge
        # (`sampled_expected_weight` / `sampled_weights_match`); the message
        # text stays oracle-identical (Rust never renders Python reprs, so
        # the translator formats it). `float(w_raw)` stays outside the guard
        # so garbage inputs raise the oracle's own exception types.
        frame_p = _frame_probability(chosen)
        try:
            exp_w: float = _bridge_contracts.sampled_expected_weight(frame_p, draws)
        except (ValueError, OverflowError, TypeError) as exc:
            raise ContractError(f"sampled expected weight bridge rejected input: {exc}") from exc
        w: float = w_raw
        try:
            matched: bool = _bridge_contracts.sampled_weights_match(w, exp_w)
        except (ValueError, OverflowError, TypeError) as exc:
            raise ContractError(
                f"bridge raw_weight {w!r} != P/draws {exp_w!r} (bridge!=oracle)"
            ) from exc
        if not matched:
            raise ContractError(f"bridge raw_weight {w!r} != P/draws {exp_w!r} (bridge!=oracle)")
        packet = getattr(chosen, "packet", None)
        if packet is None:
            raise ContractError("frame successor must carry a packet")
        ref_text: str = getattr(chosen, "successor_world_ref", "")
        delta_text: str = getattr(chosen, "delta_ref", getattr(chosen, "successor_delta", ""))
        out.append(
            SampledSuccessor(
                packet=packet,
                successor_world_ref=ref_text,
                successor_delta=delta_text,
                raw_weight=w_raw,
                provenance={
                    "mode": SAMPLED_KERNEL_MODE,
                    "samples_per_parent_action": draws,
                    "frame_mass": total,
                },
            )
        )
    return tuple(out)
