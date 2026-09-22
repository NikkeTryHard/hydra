# reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (SIM105 fallback-chain try/except-pass idiom; B007/F841 intentional scratch loop locals; B904 ContractError preconditions; N814 upstream casing; F401 cross-module names re-exported for the shim path). Evidence: https://docs.astral.sh/ruff/rules/
"""Candidate 3 PBRF commit — miss rebuild, rekey and verify, authoritative commit.

Owns the miss-path :func:`_fresh_rebuild`, the epoch-rekeying
:func:`rekey_and_verify` with its delta-reconstruction check, and the
:func:`commit` entry that promotes the authoritative realized child or
falls back to a depleted forest. The partition vocabulary and guarded
dependency flags live in :mod:`hydra2.search.pbrf_partition`, the
forest and core builder in :mod:`hydra2.search.pbrf_forest`, the
CandidateSpec factory in :mod:`hydra2.search.pbrf_spec`, and the
Planner runner in :mod:`hydra2.search.pbrf_search` plus
:mod:`hydra2.search.pbrf_act` so each file stays inside the
review-size ceiling.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, cast

from hydra2.contracts.common import ContractError, StaleBeliefError
from hydra2.search.pbrf_forest import ImmutableForest as ImmutableForest
from hydra2.search.pbrf_forest import _conditional_carry_logps as _conditional_carry_logps
from hydra2.search.pbrf_forest import _is_target_compatible as _is_target_compatible
from hydra2.search.pbrf_forest import _verify_delta_reconstruction as _verify_delta_reconstruction
from hydra2.search.pbrf_partition import ChildEntry as ChildEntry
from hydra2.search.pbrf_partition import CommitDisposition as CommitDisposition
from hydra2.search.pbrf_partition import PbrfConfig as PbrfConfig
from hydra2.search.pbrf_partition import RandomStream as RandomStream
from hydra2.search.pbrf_partition import _action_id as _action_id
from hydra2.search.pbrf_partition import _require_random_stream as _require_random_stream
from hydra2.search.pbrf_partition import fixed_allocate as fixed_allocate

__all__ = [
    "_fresh_rebuild",
    "commit",
    "rekey_and_verify",
]


def _fresh_rebuild(
    authoritative_epoch: Any,
    belief: Any,
    *,
    config: PbrfConfig | None = None,
    rng: Any | None = None,
    frozen_candidates: tuple[Any, ...] | None = None,
) -> ImmutableForest:
    """Recover from miss: sample fresh parents, return the depleted forest.

    The depleted forest carries naturally sampled parents from the
    authoritative epoch, the committing forest's frozen candidates, and empty
    children/allocations — no packet children are enumerated here (that needs
    ``candidates_fn`` and belongs to the next ``act()``). No synthetic
    particles, actions, or children are fabricated: a miss looks like a miss.
    """
    cfg: PbrfConfig = config if config is not None else PbrfConfig()
    if (
        frozen_candidates is None
        or not isinstance(frozen_candidates, tuple)
        or len(frozen_candidates) == 0
    ):
        raise ContractError("fresh rebuild requires the committing forest's frozen_candidates")
    # derive deterministic rng if not supplied (fail closed, no silent None).
    if rng is None:
        _require_random_stream()
        try:
            seed = hashlib.sha256(
                f"fresh:{authoritative_epoch.target_id}:{authoritative_epoch.epoch}".encode()
            ).digest()
            rng = RandomStream(seed)  # type: ignore[call-arg]
        except ImportError:
            raise
        except Exception as exc:
            raise ContractError(f"fresh rebuild requires deterministic RNG: {exc}") from exc
    # Wave 2 bridge audit: kept Python — miss-rebuild sampling needs Particle objects
    # from the belief corpus (bridge natural_indices returns indices only).
    new_parents: Any = ()
    if belief is not None and rng is not None:
        try:
            _new_parents_raw: Any = belief.sample_natural(
                authoritative_epoch, count=cfg.parent_count, rng=rng
            )
            new_parents = tuple(_new_parents_raw)
        except Exception as exc:
            raise ContractError(f"fresh rebuild requires belief sampling: {exc}") from exc
    else:
        raise ContractError("fresh rebuild requires belief sampling")
    if len(new_parents) == 0:
        raise ContractError("fresh rebuild sampled no parents")
    try:
        return ImmutableForest(
            epoch=authoritative_epoch,
            parents=tuple(new_parents),
            frozen_candidates=tuple(frozen_candidates),
            children={},
            config=cfg,
            allocations={},
        )
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"fresh_rebuild failed: {exc}") from exc


def rekey_and_verify(
    matching: tuple[ChildEntry, ...],
    authoritative_epoch: Any,
    *,
    forest: ImmutableForest | None = None,
    action_id: int | None = None,
) -> tuple[ChildEntry, ...]:
    """Rekey matching child entries to new epoch and verify delta reconstruction.

    For each entry, verify that ``successor_world_ref`` digest-equals reconstruction
    from ``parent_id / delta``. Uses brute-force tile search as in ``_verify_delta_reconstruction``
    when parent world_ref is available via forest.

    Raises ``DigestMismatchError`` or ``StaleBeliefError`` on failure.
    """
    # Wave 2 bridge audit: kept Python — delta-reconstruction needs parent/delta packet
    # refs (bridge PacketSuccessor carries digests only, not the parent world_ref lookup).
    if len(matching) == 0:
        raise ContractError("matching child must be non-empty")
    # Verify provenance target and epoch
    for e in matching:
        # Target must have been forest's target (already checked), but now authoritative epoch is incremented
        # We verify epoch increment
        try:
            if int(e.epoch) + 1 != int(authoritative_epoch.epoch):  # type: ignore[attr-defined,arg-type]
                raise StaleBeliefError(
                    f"child epoch {e.epoch} stale for authoritative {authoritative_epoch.epoch} [PBRF_STALE_EPOCH]"
                )
        except StaleBeliefError:
            raise
        except Exception:
            # if epochs not int-like, skip
            pass

    # Verify delta reconstruction if forest supplied (to map parent_id -> world_ref)
    if forest is not None:
        parent_map = {str(p.parent_id): str(p.world_ref) for p in forest.parents}  # type: ignore[attr-defined]
        for e in matching:
            parent_ref = parent_map.get(e.parent_id)
            if parent_ref is None:
                raise StaleBeliefError(f"parent_id {e.parent_id} not in forest [PBRF_STALE_PARENT]")
            # The committing action is known to commit(): it passes its id so
            # verification tries the exact aid first. The kernel derives aids
            # by its own rule (getattr(action, "action_id", 0)), which can
            # disagree with the planner-side _action_id hash fallback, so the
            # legacy candidate sweep plus the 0..4 fallback stays as fallback.
            # TODO(codec-aid): give CanonicalAction one codec-assigned id so
            # both fallbacks die; kernel/contracts side owned by Forge track.
            # Direct callers without an action keep the full sweep.
            aids: list[int] = []
            if (
                action_id is not None
                and not isinstance(action_id, bool)
                and isinstance(action_id, int)
            ):
                aids.append(action_id)
            aids.extend(_action_id(cand) for cand in forest.frozen_candidates)
            aids.extend(a for a in range(5) if a not in aids)
            verified = False
            for aid in aids:
                if _verify_delta_reconstruction(
                    parent_world_ref=parent_ref,
                    successor_world_ref=e.successor_world_ref,
                    successor_delta=e.successor_delta,
                    action_id=aid,
                    tile=e.tile,
                ):
                    verified = True
                    break
            if not verified:
                from hydra2.contracts.common import DigestMismatchError  # local

                raise DigestMismatchError(
                    f"delta reconstruction failed for parent {e.parent_id[:12]} [PBRF_DIGEST_DELTA]"
                )

    # Return rekeyed entries with authoritative epoch (but keep same target? authoritative target may differ)
    # For PBRF, rekey means entries now belong to new epoch: we update epoch field to authoritative epoch
    rekeyed: list[ChildEntry] = []
    for e in matching:
        rekeyed.append(
            ChildEntry(
                parent_id=e.parent_id,
                successor_world_ref=e.successor_world_ref,
                successor_delta=e.successor_delta,
                raw_weight=e.raw_weight,  # weight stays? Normalized later will recompute
                target_id=authoritative_epoch.target_id,  # type: ignore[attr-defined]
                epoch=authoritative_epoch.epoch,  # type: ignore[attr-defined]
                ancestors=e.ancestors,
                tile=e.tile,
            )
        )
    return tuple(rekeyed)


def commit(
    forest: ImmutableForest,
    action: Any,
    actual_packet: Any,
    belief: Any,
) -> tuple[ImmutableForest, CommitDisposition]:
    """Commit to authoritative realized child (action must have been emitted from the forest; the authoritative epoch comes from exact pushforward-condition on the realized packet; absent or target-incompatible matches take the miss-rebuild path; hits rekey, carry normalized conditional weights log(raw_i/Z), and squash all sibling values/visits/posteriors).

    Steps:
      require action was emitted from forest
      authoritative_epoch = belief.pushforward_condition(forest.epoch, action=action, packet=actual_packet)
      matching = forest.child(action, actual_packet.packet_id)
      if matching is absent or not target-compatible(authoritative_epoch):
          return fresh_rebuild(authoritative_epoch), miss_rebuild
      promoted = rekey_and_verify(matching, authoritative_epoch)
      squash_all_sibling_values_visits_posteriors_pairings(forest)
      return promoted_forest, hit_commit
    """
    if forest is None or not isinstance(forest, ImmutableForest):
        raise ContractError("forest must be ImmutableForest")
    # require action was emitted
    found = False
    for cand in forest.frozen_candidates:
        if _action_id(cand) == _action_id(action):
            found = True
            break
    if not found:
        raise ContractError(f"commit action {_action_id(action)} not in forest candidates")
    if actual_packet is None or not hasattr(actual_packet, "packet_id"):
        raise ContractError("actual_packet must have packet_id")
    _pid_raw: Any = getattr(actual_packet, "packet_id", None)
    pid: str = str(_pid_raw)

    # authoritative epoch via belief pushforward
    try:
        authoritative_epoch: Any = belief.pushforward_condition(  # type: ignore[union-attr]
            forest.epoch, action=action, packet=actual_packet
        )
    except Exception as exc:
        raise ContractError(f"pushforward_condition failed: {exc}") from exc

    matching = forest.child(action, pid)

    # Check target compatibility and presence
    if matching is None:
        # miss rebuild
        fresh = _fresh_rebuild(
            authoritative_epoch,
            belief,
            config=forest.config,
            frozen_candidates=forest.frozen_candidates,
        )
        return fresh, CommitDisposition("miss_rebuild")
    # verify target compatibility: if not compatible, miss
    if not _is_target_compatible(matching, authoritative_epoch, packet=actual_packet):
        fresh = _fresh_rebuild(
            authoritative_epoch,
            belief,
            config=forest.config,
            frozen_candidates=forest.frozen_candidates,
        )
        return fresh, CommitDisposition("miss_rebuild")

    # Promote: rekey and verify delta reconstruction
    try:
        rekeyed = rekey_and_verify(
            matching, authoritative_epoch, forest=forest, action_id=_action_id(action)
        )
    except (StaleBeliefError, ContractError):
        # verification failure -> miss rebuild (hard failure path but contract says rebuild)
        fresh = _fresh_rebuild(
            authoritative_epoch,
            belief,
            config=forest.config,
            frozen_candidates=forest.frozen_candidates,
        )
        return fresh, CommitDisposition("miss_rebuild")
    except Exception as exc:
        raise ContractError(f"rekey_and_verify failed: {exc}") from exc

    # squash siblings — we model by returning a new forest that contains only the matching child
    # All sibling-specific values/visits/posteriors are discarded
    # Build promoted forest: its parents are successor worlds of matching entries (one per entry)
    # For stub, we convert each ChildEntry's successor_world_ref into a synthetic parent particle
    # This promoted forest's epoch is authoritative_epoch, candidates remain same (or filtered to action?), children empty?
    # Per spec, promoted,CommitDisposition("hit_commit") returns promoted forest that will be used for next decision
    # Its parents are the successor worlds; its children are initially empty (will be rebuilt on next build)
    # We construct promoted parents as synthetic particles with same target as authoritative epoch

    # Synthesize promoted parents from rekeyed entries.
    # LAW: these are CARRIED conditionals sampling b_{eta,a,e}^+, never fresh
    # naturals — the emitted action and realized packet selected them, and
    # selection conditions the population. Densities are the normalized
    # conditional weights (see _conditional_carry_logps), not uniform -log(N).
    try:
        carry_logps = _conditional_carry_logps(tuple(rekeyed))
    except ContractError:
        # Zero-mass conditioning supports no population: miss, don't fabricate.
        fresh = _fresh_rebuild(
            authoritative_epoch,
            belief,
            config=forest.config,
            frozen_candidates=forest.frozen_candidates,
        )
        return fresh, CommitDisposition("miss_rebuild")
    promoted_parents: list[Any] = []
    for e, logp in zip(rekeyed, carry_logps, strict=True):
        # Create synthetic particle representing successor world
        @dataclass(frozen=True, slots=True)
        class _PromotedParticle:
            parent_id: str = e.parent_id
            world_ref: str = e.successor_world_ref
            epoch: Any = authoritative_epoch.epoch  # type: ignore[attr-defined]
            target_id: Any = authoritative_epoch.target_id  # type: ignore[attr-defined]
            source: str = "carried"
            log_target_density: float = logp
            log_proposal_density: float = logp
            proposal_id: str = "sha256:" + "0" * 64
            ancestors: tuple[str, ...] = (*e.ancestors, e.parent_id)

        obj = _PromotedParticle(
            parent_id=e.parent_id,
            world_ref=e.successor_world_ref,
            epoch=authoritative_epoch.epoch,  # type: ignore[attr-defined]
            target_id=authoritative_epoch.target_id,  # type: ignore[attr-defined]
            source="carried",
            log_target_density=logp,
            log_proposal_density=logp,
            proposal_id="sha256:" + "0" * 64,
            ancestors=(*e.ancestors, e.parent_id),
        )
        promoted_parents.append(obj)

    # For promoted forest, we keep same candidates and config, but children are subset: only matching child remains as historical?
    # Spec says squash_all_sibling_values_visits_posteriors_pairings(forest) — so promoted forest's children should be cleared (no siblings)
    # We'll set children to contain only the matching key with rekeyed entries, and allocations accordingly
    # But promoted forest's epoch is incremented, so we need to adjust ChildEntry epoch to authoritative epoch (already done)
    promoted_key = (_action_id(action), pid)
    promoted_children: dict[tuple[int, str], tuple[ChildEntry, ...]] = {
        promoted_key: tuple(rekeyed)
    }
    # Promoted search batches are consumed, so reallocate the fixed budget
    # across the single surviving child for determinism.
    try:
        promoted_alloc = fixed_allocate(
            cast(
                "dict[tuple[int, str], tuple[ChildEntry, ...] | list[ChildEntry]]",
                promoted_children,
            ),
            total_batches=forest.config.max_search_batches,
        )
    except Exception:
        promoted_alloc = {promoted_key: forest.config.max_search_batches}

    promoted_forest = ImmutableForest(
        epoch=authoritative_epoch,
        parents=tuple(promoted_parents),
        frozen_candidates=forest.frozen_candidates,
        children=promoted_children,
        config=forest.config,
        allocations=promoted_alloc,
    )
    # Ensure sibling squash invariant: no way to access sibling via promoted forest
    # (they are not in promoted_children)
    return promoted_forest, CommitDisposition("hit_commit")
