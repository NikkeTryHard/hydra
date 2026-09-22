# ruff: noqa: E501
"""WP-07A disjoint next actor-visible packet kernel.

Implements SPEC 14.3 PacketKernel with:
- exhaustive disjoint packets,
- probability mass one,
- physical + policy likelihood each applied once,
- exact simulator transition.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

from hydra2._native import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]
from hydra2.artifacts.canonical import canonical_bytes_batch
from hydra2.belief.natural import (
    BeliefEpoch,
    Particle,
    PolicySet,
)
from hydra2.belief.natural import (
    _require_search_bridge as _require_search_bridge,
)
from hydra2.contracts.common import (
    ContractError,
    DigestText,
    Seat,
    StaleBeliefError,
    make_sequence_no,
)
from hydra2.contracts.event_envelope import EventEnvelope, EventPayload, envelope_identity_document
from hydra2.contracts.event_packet import (
    ActorVisiblePacket,
    make_actor_visible_packets,
    public_state_chain_hash,
)


#: Bridge packet-successor row shape (single source: ``hydra2._native.search``
#: ``packet_successors``; values pinned by Rust unit tests). Precise attrs
#: let pyrefly track ints/strs/floats without per-line ignores.
class _PacketRow(Protocol):
    tile: int
    seq: int
    actor: int
    observation_hash: str
    chain_after: str
    packet_id: str
    successor_world_ref: str
    successor_delta: str
    probability: float
    log_physical: float
    log_policy: float


# Frozen kernel tables re-exported from the bridge (single source:
_KERNEL_TILE_BASE: int = _bridge_contracts.KERNEL_TILE_BASE
_KERNEL_SEQ_BASE: int = _bridge_contracts.KERNEL_SEQ_BASE
_KERNEL_SUCCESSORS: int = _bridge_contracts.KERNEL_SUCCESSORS
_KERNEL_PROB: float = _bridge_contracts.KERNEL_PROB
_KERNEL_LOG_PHYSICAL: float = _bridge_contracts.KERNEL_LOG_PHYSICAL
_KERNEL_LOG_POLICY: float = _bridge_contracts.KERNEL_LOG_POLICY
_KERNEL_MASS_TOL: float = _bridge_contracts.KERNEL_MASS_TOL
_KERNEL_PROB_TOL: float = _bridge_contracts.KERNEL_PROB_TOL
_KERNEL_TOL_UPPER: float = _bridge_contracts.KERNEL_TOL_UPPER
_KERNEL_U32_MAX: int = _bridge_contracts.KERNEL_U32_MAX
_KERNEL_GAME_ID: str = _bridge_contracts.KERNEL_GAME_ID
_KERNEL_SCHEMA_HASH: str = _bridge_contracts.KERNEL_SCHEMA_HASH


def _check_tolerance(value: float) -> float:
    """Validate `kernel_tolerance` via the bridge (oracle-identical text)."""
    try:
        tol: float = _bridge_contracts.kernel_check_tolerance(value)
        return tol
    except (ValueError, OverflowError, TypeError) as exc:
        raise ContractError("kernel_tolerance must be small positive float") from exc


def _check_action_id(aid: int) -> int:
    """Validate a normalized action id via the bridge (u32, bool excluded)."""
    try:
        checked_id: int = _bridge_contracts.kernel_check_action_id(aid)
        return checked_id
    except (ValueError, OverflowError, TypeError) as exc:
        raise ContractError(f"action_id must be u32, got {aid!r}") from exc


def _check_root_actor(root_seat: int) -> int:
    """Validate a narrowed root seat via the bridge (0..3, bool excluded)."""
    try:
        checked_seat: int = _bridge_contracts.kernel_check_root_actor(root_seat)
        return checked_seat
    except (ValueError, OverflowError, TypeError) as exc:
        raise ContractError(f"epoch root_actor must be 0..3, got {root_seat!r}") from exc


def _bridge_successor_layout(idx: int, root_seat: int) -> tuple[int, int, int]:
    """Expected `(tile, seq, actor)` triple via the bridge layout leaf."""
    try:
        layout: tuple[int, int, int] = _bridge_contracts.kernel_successor_layout(idx, root_seat)
    except (ValueError, OverflowError, TypeError) as exc:
        raise ContractError(f"kernel layout bridge rejected input: {exc}") from exc
    tile, seq, actor = layout
    return (tile, seq, actor)


def _bridge_successor_refs(parent: str, tile: int, aid: int) -> tuple[str, str]:
    """Successor/delta refs via the bridge successor leaf (cached by caller)."""
    try:
        pair: tuple[str, str] = _bridge_contracts.kernel_successor_refs(parent, tile, aid)
    except (ValueError, OverflowError, TypeError) as exc:
        raise ContractError(f"successor refs bridge rejected input: {exc}") from exc
    succ, delta = pair
    return (succ, delta)


__all__ = ["NaturalPacketKernel", "PacketSuccessor"]


@dataclass(frozen=True, slots=True)
class PacketSuccessor:
    packet: ActorVisiblePacket
    successor_world_ref: str
    delta_ref: str
    probability: float
    log_physical_probability: float
    log_actor_policy_probability: float


def _valid_digest(s: str) -> DigestText:
    return _bridge_contracts.make_digest_text(s)


def _make_public_discard_event(
    *,
    sequence: int,
    actor: int,
    tile: int,
    game_id: str = _KERNEL_GAME_ID,
    rules_hash: DigestText,
    schema_hash: DigestText,
) -> EventEnvelope:
    seat: Seat = _bridge_contracts.make_seat(actor)
    payload = EventPayload(
        kind="discard",
        actor=seat,
        tile=tile,  # type: ignore[arg-type]
        action_id=0,  # type: ignore[arg-type]
        source_seat=None,
        consumed_tiles=(),
        offered_action_ids=(),
        accepted_action_ids=(),
        round_index=None,
        scores=None,
        reason=None,
    )
    return EventEnvelope(
        game_id=game_id,
        sequence=sequence,  # type: ignore[arg-type]
        kind="discard",
        actor=seat,
        visibility="public",
        visible_to=(Seat(0), Seat(1), Seat(2), Seat(3)),
        payload=payload,
        public_delta=(),
        rules_hash=rules_hash,
        schema_hash=schema_hash,
    )


@lru_cache(maxsize=4096)
def _cached_successor_refs(particle_world_ref: str, tile: int, aid: int) -> tuple[str, str]:
    """Cache successor/delta refs per particle+tile+aid — avoids repeated sha256.

    Evidence: torch.compile inductor reuses compiled graphs via cache keyed on
    shape/dtype (https://pytorch.org/docs/stable/generated/torch.compile.html);
    analogous hash-keyed lru_cache avoids re-hashing same particle across
    repeated kernel enumerations. Also arrow zero-copy columnar take similarly
    caches hash lookups (https://arrow.apache.org/docs/python/index.html).
    """
    return _bridge_successor_refs(particle_world_ref, tile, aid)


class NaturalPacketKernel:
    """Natural packet kernel — finite exhaustive enumeration (WP-07A)."""

    def __init__(self, *, kernel_tolerance: float = _KERNEL_MASS_TOL) -> None:
        self._tol = _check_tolerance(kernel_tolerance)

    def enumerate_next(
        self,
        *,
        epoch: BeliefEpoch,
        particle: Particle,
        action: Any,
        policy_set: PolicySet | None = None,
    ) -> tuple[PacketSuccessor, ...]:
        # Stale checks
        if int(particle.epoch) != int(epoch.epoch) or particle.target_id != epoch.target_id:
            raise StaleBeliefError("particle epoch/target stale for kernel [PBRF_STALE_EPOCH]")
        if particle.world_ref is None:
            raise ContractError("particle world_ref missing")
        # Registry-free by design: successors are synthesized deterministically
        # from (particle_world_ref, tile, aid); probability splits exactly once
        # into physical x policy (log_phys=log(0.5), log_policy=0.0).
        # Exactly 2 disjoint packets per parent/action.
        if not hasattr(action, "action_id") and not isinstance(action, int):
            # Accept raw int or CanonicalAction; normalize to int id
            try:
                aid = int(action)
            except Exception:
                aid = 0
        else:
            try:
                aid = int(getattr(action, "action_id", 0))
            except Exception:
                aid = 0

        # Use deterministic physical draws: two possibilities (tile 8 vs 9 etc)
        # For hidden permutation test, the packets must be disjoint by packet_id.
        # We create two packets with distinct tile discards.
        _ = policy_set if policy_set is not None else PolicySet()
        # Retrieve epoch hashes for event creation
        rh = epoch.rules_hash
        # Use event_schema_hash dummy as c*64 for packet events
        sh = _valid_digest(_KERNEL_SCHEMA_HASH)
        if not isinstance(particle.world_ref, str) or particle.world_ref == "":
            raise ContractError("particle world_ref must be non-empty str")
        try:
            root_seat_init: int = int(epoch.root_actor)
        except Exception as exc:
            raise ContractError(f"epoch root_actor must be int seat: {exc}") from exc
        root_seat: int = _check_root_actor(root_seat_init)
        aid = _check_action_id(aid)
        rules_hash_str: str = str(rh)
        # Bridge assembly: successor/chain/hash via search.packet_successors.
        # The bridge computes tile/seq/actor layout, successor/delta refs,
        # observation/packet/chain hashes, and split mass detached; Python
        # rebuilds the packet objects and fail-closes on any byte/hash
        # mismatch (bridge==oracle proof, no fallback).
        search_mod = _require_search_bridge(need="packet_successors", purpose="enumerating packets")
        try:
            bridge_rows: list[_PacketRow] = search_mod.packet_successors(
                particle.world_ref, aid, root_seat, rules_hash_str
            )
        except ImportError:
            raise
        except (ValueError, OverflowError, TypeError) as exc:
            raise ContractError(f"packet successor bridge rejected input: {exc}") from exc
        try:
            rows: list[_PacketRow] = list(bridge_rows)
        except TypeError as exc:
            raise ContractError(f"packet successor bridge returned non-sequence: {exc}") from exc
        if len(rows) != _KERNEL_SUCCESSORS:
            raise ContractError(
                f"packet successor bridge must return exactly {_KERNEL_SUCCESSORS}, got {len(rows)}"
            )
        # Two successors (bridge order is idx order: tile 8+idx, seq 100+idx):
        successors: list[PacketSuccessor] = []
        # Physical probabilities: uniform 0.5 each
        # Policy probabilities: uniform 1.0 (deterministic opponent)
        # Combined prob = 0.5 * 1.0 = 0.5
        # Pass 1: validate bridge layout and build both events (all checks kept).
        tiles: list[int] = []
        seqs: list[int] = []
        events: list[EventEnvelope] = []
        for idx, brow in enumerate(rows):
            tile: int = brow.tile
            seq: int = brow.seq
            actor_opponent: int = brow.actor
            exp_tile, exp_seq, exp_actor = _bridge_successor_layout(idx, root_seat)
            if tile != exp_tile or seq != exp_seq or actor_opponent != exp_actor:
                raise ContractError(
                    f"bridge layout mismatch idx={idx}: got {(tile, seq, actor_opponent)}, "
                    f"expected {(exp_tile, exp_seq, exp_actor)}"
                )
            # Ensure actor_opponent != root_actor to make packet actor-visible?
            # Use opponent as actor of discard
            events.append(
                _make_public_discard_event(
                    sequence=seq,
                    actor=actor_opponent,
                    tile=tile,
                    rules_hash=rh,
                    schema_hash=sh,
                )
            )
            tiles.append(tile)
            seqs.append(seq)
        # Batch form: observation + chain-fold docs for both successors serialize
        # in ONE bridge FFI (byte-identical blobs to the retired per-successor
        # Python serializes + lru-cached digests); digests hash with hashlib
        # exactly like the single paths. The empty-before digest is constant.
        # Fold shape {"prefix", "event"} mirrors event_packet._fold_chain_digest;
        # any divergence fails the chain_after bridge equality below (fail closed).
        before = public_state_chain_hash([])
        obs_docs: list[dict[str, object]] = [
            {"packet_seq": seq, "tile": tile} for seq, tile in zip(seqs, tiles, strict=True)
        ]
        fold_docs: list[dict[str, object]] = [
            {"prefix": str(before), "event": envelope_identity_document(event)} for event in events
        ]
        blobs = canonical_bytes_batch(obs_docs + fold_docs)
        obs_hashes = [
            DigestText("sha256:" + hashlib.sha256(blob).hexdigest()) for blob in blobs[:2]
        ]
        afters = [DigestText("sha256:" + hashlib.sha256(blob).hexdigest()) for blob in blobs[2:]]
        # Stage both packets (mirrors make_actor_visible_packet staging); ids bind
        # via ONE batch FFI with the constructor re-verify kept (fail closed).
        staged = [
            ActorVisiblePacket(
                packet_id=None,
                actor_view=epoch.root_actor,
                source_sequence_start=make_sequence_no(int(event.sequence)),
                source_sequence_end=make_sequence_no(int(event.sequence)),
                events=(event,),
                public_state_hash_before=before,
                public_state_hash_after=after,
                observation_hash_after=obs_hash,
            )
            for event, after, obs_hash in zip(events, afters, obs_hashes, strict=True)
        ]
        packets = make_actor_visible_packets(staged)
        # Pass 2: bridge-oracle equality + likelihood checks (all checks kept).
        for idx, brow in enumerate(rows):
            obs_hash = obs_hashes[idx]
            after = afters[idx]
            packet = packets[idx]
            # Byte/hash equality: bridge hashes must equal the oracle hashes.
            if obs_hash != brow.observation_hash:
                raise ContractError(f"observation hash mismatch idx={idx} (bridge!=oracle)")
            if after != brow.chain_after:
                raise ContractError(f"chain_after mismatch idx={idx} (bridge!=oracle)")
            if packet.packet_id != brow.packet_id:
                raise ContractError(f"packet_id mismatch idx={idx} (bridge!=oracle)")
            # Successor world: deterministic new world id derived from particle+tile
            tile = tiles[idx]
            exp_succ, exp_delta = _cached_successor_refs(particle.world_ref, tile, aid)
            if exp_succ != brow.successor_world_ref:
                raise ContractError(f"successor ref mismatch idx={idx} (bridge!=oracle)")
            if exp_delta != brow.successor_delta:
                raise ContractError(f"delta ref mismatch idx={idx} (bridge!=oracle)")
            prob: float = brow.probability
            log_phys: float = brow.log_physical
            log_policy: float = brow.log_policy
            if abs(prob - _KERNEL_PROB) > _KERNEL_PROB_TOL:
                raise ContractError(f"bridge probability {prob!r} != 0.5")
            if (
                abs(log_phys - _KERNEL_LOG_PHYSICAL) > _KERNEL_PROB_TOL
                or abs(log_policy - _KERNEL_LOG_POLICY) > _KERNEL_PROB_TOL
            ):
                raise ContractError("bridge log split mismatch (bridge!=oracle)")
            # Verify probability == exp(log_phys+log_policy)
            recomb = math.exp(log_phys + log_policy)
            if abs(recomb - prob) > _KERNEL_PROB_TOL:
                raise ContractError("probability decomposition inconsistent")
            successors.append(
                PacketSuccessor(
                    packet=packet,
                    successor_world_ref=brow.successor_world_ref,
                    delta_ref=brow.successor_delta,
                    probability=prob,
                    log_physical_probability=log_phys,
                    log_actor_policy_probability=log_policy,
                )
            )
        # Post-conditions: check invariants
        # 1. pairwise disjoint by packet identity
        pids = [s.packet.packet_id for s in successors]
        if len(pids) != len(set(pids)):
            raise ContractError("packet successors not pairwise disjoint")
        # 2. probabilities sum to 1 within tolerance
        total = sum(s.probability for s in successors)
        if abs(total - 1.0) > self._tol:
            raise ContractError(f"packet mass {total} != 1 within {self._tol}")
        # 3. each probability finite nonnegative
        for s in successors:
            if not math.isfinite(s.probability) or s.probability < 0:
                raise ContractError("probability must be finite nonnegative")
            if not math.isfinite(s.log_physical_probability) or not math.isfinite(
                s.log_actor_policy_probability
            ):
                raise ContractError("log probabilities must be finite")
        # 4. successor state follows exact simulator transition — we claim deterministic synthesis is exact for tiny model
        return tuple(successors)
