# ruff: noqa: E501, N806
"""WP-07A Natural Belief Harness — immutable epoch, natural law, scoreable proposals.

Implements SPEC 14.2 Target and proposal, and Belief protocol for natural worlds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from hydra2_replay_rs import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]

from hydra2.artifacts.digest import of_canonical
from hydra2.contracts.common import (
    BeliefEpochId,
    ContractError,
    DigestText,
    ParentId,
    ProposalSupportError,
    Seat,
    StaleBeliefError,
    make_belief_epoch_id,
    make_parent_id,
)
from hydra2.contracts.event_packet import ActorVisiblePacket
from hydra2.contracts.observation_actor import ActorObservation


def _require_search_bridge(*, need: str, purpose: str) -> Any:
    """Import the built ``search`` bridge surface with the ``need`` pyfn (fail closed).

    Single home for the belief search-bridge importer: the packet kernel and
    the sampled mode import this; per-mode ``need``/``purpose`` keep every
    fail-closed message byte-identical to the retired per-module copies.
    """
    try:
        import hydra2_replay_rs as _ext  # pyrefly: ignore[missing-import]
    except ImportError as exc:
        raise ImportError(
            "hydra2_replay_rs extension with search not importable; "
            f"build the bridge with `pixi run build-ext` before {purpose}"
        ) from exc
    try:
        mod = _ext.search
    except AttributeError as exc:
        raise ImportError(
            "hydra2_replay_rs.search submodule missing (stale .so); "
            "rebuild the bridge with `pixi run build-ext`"
        ) from exc
    if not hasattr(mod, need):
        raise ImportError(
            f"hydra2_replay_rs.search.{need} missing (stale .so); "
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
    from hydra2.belief.world import FullWorld
    from hydra2.contracts.randomness import RandomStream

__all__ = [
    "BeliefEpoch",
    "NaturalBelief",
    "Particle",
    "PolicySet",
    "ProposalSpec",
]


# ---------------------------------------------------------------------------
# SPEC 14.2 dataclasses — exact field order and types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BeliefEpoch:
    epoch: BeliefEpochId
    target_id: DigestText
    root_actor: Seat
    observation_hash: DigestText
    rules_hash: DigestText
    belief_model_hash: DigestText
    event_model_hash: DigestText
    proposal_spec_hash: DigestText


@dataclass(frozen=True, slots=True)
class Particle:
    parent_id: ParentId
    world_ref: str
    epoch: BeliefEpochId
    target_id: DigestText
    source: Literal["natural", "proposal"]
    log_target_density: float
    log_proposal_density: float
    proposal_id: DigestText


@dataclass(frozen=True, slots=True)
class ProposalSpec:
    """Minimal proposal spec for WP-07A natural harness."""

    proposal_id: DigestText
    digest: DigestText

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "proposal_id", _bridge_contracts.make_digest_text(self.proposal_id)
        )
        object.__setattr__(self, "digest", _bridge_contracts.make_digest_text(self.digest))


@dataclass(frozen=True, slots=True)
class PolicySet:
    """Provenance-only policy set consumed by the packet kernel."""

    # Policies carry seat->policy_id provenance only; the kernel supplies
    # the deterministic likelihood (log_policy=0.0).
    policies: tuple[tuple[int, str], ...] = ()

    def log_prob(self, actor: int, action_id: int) -> float:
        # Returns log(1.0): the kernel applies the exact 0.5/0.5 split once
        # per successor, so the policy factor here is unity.
        return 0.0


# ---------------------------------------------------------------------------
# Helpers — target identity and world corpus
# ---------------------------------------------------------------------------


def _target_id_for(
    *,
    observation_hash: DigestText,
    rules_hash: DigestText,
    belief_model_hash: DigestText,
    event_model_hash: DigestText,
    proposal_spec_hash: DigestText,
) -> DigestText:
    doc = {
        "observation_hash": observation_hash,
        "rules_hash": rules_hash,
        "belief_model_hash": belief_model_hash,
        "event_model_hash": event_model_hash,
        "proposal_spec_hash": proposal_spec_hash,
    }
    # Digest line via the canon bridge (byte-identical to the retired
    # hashlib-over-canonical_bytes loop; ImportError with build-ext hint,
    # no oracle fallback). Math/doc shape untouched.
    return of_canonical(doc)


def _validate_finite(value: float, *, name: str) -> float:
    if not isinstance(value, float) or not math.isfinite(value):
        raise ContractError(f"{name} must be finite float, got {value!r}")
    return value


# ---------------------------------------------------------------------------
# Tiny corpus generation — deterministic consistent worlds
# ---------------------------------------------------------------------------


def _build_tiny_corpus_for_epoch(
    epoch: BeliefEpoch,
    *,
    registry: dict[str, FullWorld],
) -> list[FullWorld]:
    """Return worlds consistent with epoch.observation_hash from registry.

    Registry is the belief's world store. If epoch has no stored corpus,
    lazily generate a deterministic tiny corpus of 4 worlds sharing the same
    root observation. This is used for uniform natural law.
    """
    # Filter registry by observation_hash and rules_hash
    consistent = [
        w
        for w in registry.values()
        if w.observation_hash == epoch.observation_hash and w.rules_hash == epoch.rules_hash
    ]
    if len(consistent) > 0:
        # Deterministic order by world_id
        consistent.sort(key=lambda w: w.world_id)
        return consistent
    # No stored corpus: synthesize 4 worlds consistent by construction with
    # the same root hand, then bind their observation_hash to the epoch so
    # the generated worlds match the epoch observation.
    # Tiny-domain tile pool 0..11: seats hold variants over 0..7, the wall
    # holds 8..11. Root hand stays fixed for hidden-permutation invariance.
    base_hands_options = [
        ((0, 1), (2, 3), (4, 5), (6, 7)),
        ((0, 1), (2, 4), (3, 5), (6, 7)),
        ((0, 1), (2, 5), (3, 4), (6, 7)),
        ((0, 1), (2, 6), (3, 4), (5, 7)),
    ]
    wall = (8, 9, 10, 11)
    worlds: list[FullWorld] = []
    for idx, hands in enumerate(base_hands_options):
        # Deterministically perturb latent_state with idx to keep world_id unique even if hands repeat
        from hydra2.belief.world import make_full_world

        w = make_full_world(
            concealed_hands=hands,
            live_wall=wall,
            dead_wall=(),
            latent_state={"corpus_idx": idx},
            rules_hash=epoch.rules_hash,
            observation_hash=epoch.observation_hash,
            simulator_snapshot=f"tiny:{epoch.target_id}:{idx}",
        )
        worlds.append(w)
    # Register them into provided dict (caller will extend)
    for w in worlds:
        registry[w.world_id] = w
    worlds.sort(key=lambda w: w.world_id)
    return worlds


# ---------------------------------------------------------------------------
# NaturalBelief implementation
# ---------------------------------------------------------------------------


class NaturalBelief:
    """Natural world law consistent with actor observation (WP-07A).

    - Target law is uniform over tiny corpus consistent with observation.
    - Natural samples have log_target == log_proposal (ratio 1).
    - Proposal samples are skewed (0.5 vs 0.5/(K-1)) with differing logs but same support.
    - Stale epoch/target/provenance rejected with typed errors.
    - Epoch increments after committed transition via pushforward_condition.
    """

    def __init__(
        self,
        *,
        rules_hash: DigestText | None = None,
        belief_model_hash: DigestText | None = None,
        event_model_hash: DigestText | None = None,
        proposal_spec_hash: DigestText | None = None,
    ) -> None:
        self._rules_hash: DigestText = _bridge_contracts.make_digest_text(
            rules_hash if rules_hash is not None else ("sha256:" + "a" * 64)
        )
        self._belief_model_hash: DigestText = _bridge_contracts.make_digest_text(
            belief_model_hash if belief_model_hash is not None else ("sha256:" + "b" * 64)
        )
        self._event_model_hash: DigestText = _bridge_contracts.make_digest_text(
            event_model_hash if event_model_hash is not None else ("sha256:" + "c" * 64)
        )
        self._proposal_spec_hash: DigestText = _bridge_contracts.make_digest_text(
            proposal_spec_hash if proposal_spec_hash is not None else ("sha256:" + "d" * 64)
        )
        self._next_epoch: int = 0
        self._epochs: dict[int, BeliefEpoch] = {}
        self._worlds: dict[str, FullWorld] = {}
        self._current_epoch_id: int | None = None

    # -- epoch management -----------------------------------------------------

    def _store_epoch(self, epoch: BeliefEpoch) -> None:
        eid = int(epoch.epoch)
        self._epochs[eid] = epoch
        self._current_epoch_id = eid
        # Materialize the epoch corpus now so later sampling never branches.
        _ = _build_tiny_corpus_for_epoch(epoch, registry=self._worlds)

    def _require_epoch(self, epoch: BeliefEpoch) -> BeliefEpoch:
        eid = int(epoch.epoch)
        stored = self._epochs.get(eid)
        if stored is None or stored != epoch:
            raise StaleBeliefError(f"stale epoch {eid}: not found or mismatched [PBRF_STALE_EPOCH]")
        # Target check is part of epoch equality, but also ensure target_id matches
        if stored.target_id != epoch.target_id:
            raise StaleBeliefError("target_id mismatch for epoch [PBRF_STALE_TARGET]")
        return stored

    def _require_particle_epoch(self, particle: Particle) -> None:
        # Particle's epoch and target must match a stored epoch
        eid = int(particle.epoch)
        stored = self._epochs.get(eid)
        if stored is None or stored.target_id != particle.target_id:
            raise StaleBeliefError(
                f"stale particle provenance epoch={eid} target mismatch [PBRF_STALE_PROVENANCE]"
            )
        # Also check that world_ref exists
        if particle.world_ref not in self._worlds:
            raise StaleBeliefError(
                f"unknown world_ref {particle.world_ref!r} [PBRF_STALE_WORLDREF]"
            )

    # -- public API ------------------------------------------------------------

    def begin(
        self, observation: ActorObservation, *, model_id: DigestText | None = None
    ) -> BeliefEpoch:
        """Bind a new epoch to an actor observation; raises ContractError."""
        if not isinstance(observation, ActorObservation):
            raise ContractError("begin requires ActorObservation")
        assert observation.observation_hash is not None
        assert observation.rules_hash is not None
        # Use supplied model_id as belief_model_hash if given, else default
        bh = (
            _bridge_contracts.make_digest_text(model_id)
            if model_id is not None
            else self._belief_model_hash
        )
        # Compute target identity
        target_id = _target_id_for(
            observation_hash=_bridge_contracts.make_digest_text(observation.observation_hash),
            rules_hash=_bridge_contracts.make_digest_text(observation.rules_hash),
            belief_model_hash=bh,
            event_model_hash=self._event_model_hash,
            proposal_spec_hash=self._proposal_spec_hash,
        )
        epoch = BeliefEpoch(
            epoch=make_belief_epoch_id(self._next_epoch),
            target_id=target_id,
            root_actor=_bridge_contracts.make_seat(int(observation.actor)),
            observation_hash=_bridge_contracts.make_digest_text(observation.observation_hash),
            rules_hash=_bridge_contracts.make_digest_text(observation.rules_hash),
            belief_model_hash=bh,
            event_model_hash=self._event_model_hash,
            proposal_spec_hash=self._proposal_spec_hash,
        )
        self._next_epoch += 1
        self._store_epoch(epoch)
        return epoch

    def sample_natural(
        self, epoch: BeliefEpoch, *, count: int, rng: RandomStream
    ) -> tuple[Particle, ...]:
        """Sample uniform natural particles (log_target == log_proposal)."""
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ContractError("count must be positive int")
        _ = self._require_epoch(epoch)
        corpus = _build_tiny_corpus_for_epoch(epoch, registry=self._worlds)
        K = len(corpus)
        if K == 0:
            raise ContractError("empty corpus for epoch")
        log_prob = -math.log(K)
        _ = _validate_finite(log_prob, name="log_target_density")
        # Draws via the search bridge (CTR-exact, including the K==1
        # no-consume rule). Cursor replay: seed/cursor in, end_cursor out,
        # rng jumped so the stream continues exactly (checkpoint/jump_to
        # shape). ImportError with build-ext hint, no oracle fallback.
        seed, cursor = _ctr_seed_cursor(rng)
        search_mod = _require_search_bridge(need="natural_indices", purpose="sampling belief")
        try:
            indices, end_cursor = search_mod.natural_indices(K, count, seed, cursor)
        except ImportError:
            raise
        except (ValueError, OverflowError, TypeError) as exc:
            raise ContractError(f"natural_indices bridge rejected input: {exc}") from exc
        try:
            rng.jump_to(int(end_cursor))
        except (AttributeError, ValueError, TypeError) as exc:
            raise ContractError(f"rng jump_to failed for bridge replay: {exc}") from exc
        try:
            idx_list = list(indices)
        except TypeError as exc:
            raise ContractError(f"natural_indices bridge returned non-sequence: {exc}") from exc
        if len(idx_list) != count:
            raise ContractError(
                f"natural_indices bridge returned {len(idx_list)} indices, expected {count}"
            )
        out: list[Particle] = []
        for raw_idx in idx_list:
            idx = int(raw_idx)
            if idx < 0 or idx >= K:
                raise ContractError(f"bridge index {idx!r} out of range for K={K}")
            world = corpus[idx]
            # For natural, log_target == log_proposal, ratio 1
            pid = make_parent_id(
                world.world_id.split(":")[1][:16] if ":" in world.world_id else world.world_id[:16]
            )
            # Use world_id as world_ref opaque
            particle = Particle(
                parent_id=pid,
                world_ref=world.world_id,
                epoch=epoch.epoch,
                target_id=epoch.target_id,
                source="natural",
                log_target_density=log_prob,
                log_proposal_density=log_prob,
                proposal_id=epoch.proposal_spec_hash,
            )
            # Extra validation: densities finite
            _ = _validate_finite(particle.log_target_density, name="log_target_density")
            _ = _validate_finite(particle.log_proposal_density, name="log_proposal_density")
            if particle.log_target_density != particle.log_proposal_density:
                raise ContractError("natural sample requires log_target == log_proposal")
            out.append(particle)
        return tuple(out)

    def sample_proposal(
        self, epoch: BeliefEpoch, *, proposal: ProposalSpec, count: int, rng: RandomStream
    ) -> tuple[Particle, ...]:
        """Sample skewed proposal particles over the same corpus support."""
        if not isinstance(proposal, ProposalSpec):
            raise ContractError("proposal must be ProposalSpec")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ContractError("count must be positive int")
        _ = self._require_epoch(epoch)
        corpus = _build_tiny_corpus_for_epoch(epoch, registry=self._worlds)
        K = len(corpus)
        if K == 0:
            raise ContractError("empty corpus")
        # Proposal distribution: skewed
        if K == 1:
            proposal_probs = [1.0]
        else:
            # first world 0.5, rest share 0.5/(K-1)
            rest = 0.5 / (K - 1)
            proposal_probs = [0.5] + [rest] * (K - 1)
        # Validate support: every target-positive (all K) must have proposal>0
        for p in proposal_probs:
            if p <= 0.0 or not math.isfinite(p):
                raise ProposalSupportError(
                    "proposal lacks support for target-positive region [PBRF_SUPPORT_REGION]"
                )
        out: list[Particle] = []
        for _ in range(count):
            # Sample according to proposal_probs
            r = rng.random_float()
            cum = 0.0
            idx = K - 1
            for i, p in enumerate(proposal_probs):
                cum += p
                if r < cum:
                    idx = i
                    break
            world = corpus[idx]
            log_target = -math.log(K)
            log_proposal = math.log(proposal_probs[idx])
            _ = _validate_finite(log_target, name="log_target_density")
            _ = _validate_finite(log_proposal, name="log_proposal_density")
            # Support check: target-positive must have proposal>0 (already)
            if math.exp(log_target) > 0 and math.exp(log_proposal) == 0:
                raise ProposalSupportError(
                    "proposal density zero for target-positive world [PBRF_SUPPORT_POINT]"
                )
            pid = make_parent_id(
                world.world_id.split(":")[1][:16] if ":" in world.world_id else world.world_id[:16]
            )
            particle = Particle(
                parent_id=pid,
                world_ref=world.world_id,
                epoch=epoch.epoch,
                target_id=epoch.target_id,
                source="proposal",
                log_target_density=log_target,
                log_proposal_density=log_proposal,
                proposal_id=proposal.proposal_id,
            )
            out.append(particle)
        return tuple(out)

    def condition_for_actor(
        self,
        epoch: BeliefEpoch,
        *,
        actor_observation: ActorObservation,
        count: int,
        rng: RandomStream,
    ) -> tuple[Particle, ...]:
        if not isinstance(actor_observation, ActorObservation):
            raise ContractError("actor_observation must be ActorObservation")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ContractError("count must be positive int")
        _ = self._require_epoch(epoch)
        # Immutable-constraint filter: keep worlds whose concealed hand for
        # the queried seat exactly equals the observation's concealed hand.
        corpus = _build_tiny_corpus_for_epoch(epoch, registry=self._worlds)
        actor_seat = int(actor_observation.actor)
        filtered = [
            w
            for w in corpus
            if tuple(w.concealed_hands[actor_seat]) == tuple(actor_observation.concealed_hand)
        ]
        if len(filtered) == 0:
            # Empty result is a hard ContractError: the observation violates
            # the epoch's immutable constraints, so no world is consistent.
            raise ContractError(
                "no worlds consistent with actor_observation (immutable constraints violated)"
            )
        K = len(filtered)
        log_prob = -math.log(K) if K > 0 else float("-inf")
        # Draws via the search bridge (CTR-exact, K==1 no-consume). Cursor
        # replay keeps the stream exact; ImportError with build-ext hint.
        seed, cursor = _ctr_seed_cursor(rng)
        search_mod = _require_search_bridge(need="natural_indices", purpose="sampling belief")
        try:
            indices, end_cursor = search_mod.natural_indices(K, count, seed, cursor)
        except ImportError:
            raise
        except (ValueError, OverflowError, TypeError) as exc:
            raise ContractError(f"natural_indices bridge rejected input: {exc}") from exc
        try:
            rng.jump_to(int(end_cursor))
        except (AttributeError, ValueError, TypeError) as exc:
            raise ContractError(f"rng jump_to failed for bridge replay: {exc}") from exc
        try:
            idx_list = list(indices)
        except TypeError as exc:
            raise ContractError(f"natural_indices bridge returned non-sequence: {exc}") from exc
        if len(idx_list) != count:
            raise ContractError(
                f"natural_indices bridge returned {len(idx_list)} indices, expected {count}"
            )
        out: list[Particle] = []
        for raw_idx in idx_list:
            idx = int(raw_idx)
            if idx < 0 or idx >= K:
                raise ContractError(f"bridge index {idx!r} out of range for K={K}")
            world = filtered[idx]
            pid = make_parent_id(
                world.world_id.split(":")[1][:16] if ":" in world.world_id else world.world_id[:16]
            )
            particle = Particle(
                parent_id=pid,
                world_ref=world.world_id,
                epoch=epoch.epoch,
                target_id=epoch.target_id,
                source="natural",
                log_target_density=log_prob,
                log_proposal_density=log_prob,
                proposal_id=epoch.proposal_spec_hash,
            )
            out.append(particle)
        return tuple(out)

    def pushforward_condition(
        self, epoch: BeliefEpoch, *, action: Any, packet: ActorVisiblePacket
    ) -> BeliefEpoch:
        _ = self._require_epoch(epoch)
        if not isinstance(packet, ActorVisiblePacket):
            raise ContractError("packet must be ActorVisiblePacket")
        # Packet actor_view must equal the epoch root actor.
        if int(packet.actor_view) != int(epoch.root_actor):
            raise ContractError("packet actor_view must equal epoch root_actor")
        # The packet's post-state observation is authoritative for the epoch.
        new_obs_hash = packet.observation_hash_after
        new_epoch_id = int(epoch.epoch) + 1
        # Target identity is recomputed from the post-packet observation via
        # _target_id_for, so pushforward equals a fresh begin() on that
        # observation; the epoch id increments monotonically.
        new_target = _target_id_for(
            observation_hash=new_obs_hash,
            rules_hash=epoch.rules_hash,
            belief_model_hash=epoch.belief_model_hash,
            event_model_hash=epoch.event_model_hash,
            proposal_spec_hash=epoch.proposal_spec_hash,
        )
        new_epoch = BeliefEpoch(
            epoch=make_belief_epoch_id(new_epoch_id),
            target_id=new_target,
            root_actor=epoch.root_actor,
            observation_hash=new_obs_hash,
            rules_hash=epoch.rules_hash,
            belief_model_hash=epoch.belief_model_hash,
            event_model_hash=epoch.event_model_hash,
            proposal_spec_hash=epoch.proposal_spec_hash,
        )
        # Advance counter if needed
        if new_epoch_id >= self._next_epoch:
            self._next_epoch = new_epoch_id + 1
        self._store_epoch(new_epoch)
        return new_epoch

    def log_density(self, epoch: BeliefEpoch, world_ref: str) -> float:
        _ = self._require_epoch(epoch)
        if not isinstance(world_ref, str) or world_ref == "":
            raise ContractError("world_ref must be non-empty str")
        world = self._worlds.get(world_ref)
        if world is None:
            raise StaleBeliefError(f"unknown world_ref {world_ref!r} [PBRF_STALE_WORLDREF]")
        corpus = _build_tiny_corpus_for_epoch(epoch, registry=self._worlds)
        K = len(corpus)
        # Support contract: worlds outside the epoch corpus have density zero
        # (log -inf); valid worlds return finite -log(K) below. Callers
        # needing a hard failure raise ProposalSupportError at sampling time.
        if world.world_id not in {w.world_id for w in corpus}:
            return float("-inf")
        logp = -math.log(K)
        _ = _validate_finite(logp, name="log_density")
        return logp
