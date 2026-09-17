# reason: legacy blanket kept, not narrowed — narrowing surfaces unrelated mid-flight noise outside the owned error set (SIM105 fallback-chain try/except-pass idiom; B007/F841 intentional scratch loop locals; B904 ContractError preconditions; N814 upstream casing; F401 re-exported split symbols). Evidence: https://docs.astral.sh/ruff/rules/
"""Candidate 3 PBRF core — natural immutable parent population, packet forest.

Re-export facade over the split modules: :mod:`hydra2.search.pbrf_partition`
(frozen config, child records, allocation guards plus the guarded
verification, immutable forest, core builder),
:mod:`hydra2.search.pbrf_commit` (miss rebuild, rekey and verify,
authoritative commit), :mod:`hydra2.search.pbrf_spec` (CandidateSpec
factory), and :mod:`hydra2.search.pbrf_act` (Planner construction plus
budget/telemetry/value driver, Planner act plus :class:`PbrfPlanner`).
Import from this path; it preserves every public name and ``__all__``.

Implements SPEC 16.4 + Blueprint §10 PBRF core:

- Natural immutable parent population via ``NaturalBelief.sample_natural`` (ratio 1).
- Frozen root candidate generator before any packet enumeration evidence.
- Exhaustive disjoint packet kernel per parent/action via ``NaturalPacketKernel``.
- Child entries store ``parent_id, successor_world_ref, successor_delta, raw_weight, target_id, epoch, tile``.
- Child normalizers partition one within ``kernel_tolerance`` (mass 1).
- Fixed search batches allocated deterministically (``fixed_allocate``).
- Candidates frozen before natural confirmation (confirmation always fresh natural).
- Commit only authoritative realized child, increment belief epoch, squash siblings.
- Deterministic semantic seeds, actor-visible only keys, privileged world_ref isolation.
- ``successor_world_ref`` mandatory, ``successor_delta`` verified via reconstruction.

Ownership: this module owns the PBRF core contracts; peers extend without redefinition.
"""

from __future__ import annotations

from hydra2.search.pbrf_act import (
    PbrfPlanner as PbrfPlanner,
)
from hydra2.search.pbrf_act import (
    PbrfPlannerActMixin as PbrfPlannerActMixin,
)
from hydra2.search.pbrf_act import PbrfPlannerSearchMixin as PbrfPlannerSearchMixin
from hydra2.search.pbrf_commit import (
    _fresh_rebuild as _fresh_rebuild,
)
from hydra2.search.pbrf_commit import (
    commit as commit,
)
from hydra2.search.pbrf_commit import (
    rekey_and_verify as rekey_and_verify,
)
from hydra2.search.pbrf_forest import (
    ImmutableForest as ImmutableForest,
)
from hydra2.search.pbrf_forest import (
    _conditional_carry_logps as _conditional_carry_logps,
)
from hydra2.search.pbrf_forest import (
    _is_target_compatible as _is_target_compatible,
)
from hydra2.search.pbrf_forest import (
    _tile_for_successor as _tile_for_successor,
)
from hydra2.search.pbrf_forest import (
    _verify_delta_reconstruction as _verify_delta_reconstruction,
)
from hydra2.search.pbrf_forest import (
    build_pbrf as build_pbrf,
)
from hydra2.search.pbrf_partition import (
    ChildEntry as ChildEntry,
)
from hydra2.search.pbrf_partition import (
    CommitDisposition as CommitDisposition,
)
from hydra2.search.pbrf_partition import (
    PbrfConfig as PbrfConfig,
)
from hydra2.search.pbrf_partition import (
    fixed_allocate as fixed_allocate,
)
from hydra2.search.pbrf_partition import (
    validate_packet_partition as validate_packet_partition,
)
from hydra2.search.pbrf_spec import (
    _canonical_hashes as _canonical_hashes,
)
from hydra2.search.pbrf_spec import (
    _derive_utility_manifest_hash as _derive_utility_manifest_hash,
)
from hydra2.search.pbrf_spec import (
    make_pbrf_candidate_spec as make_pbrf_candidate_spec,
)

__all__ = [
    "ChildEntry",
    "CommitDisposition",
    "ImmutableForest",
    "PbrfConfig",
    "PbrfPlanner",
    "build_pbrf",
    "commit",
    "fixed_allocate",
    "make_pbrf_candidate_spec",
    "validate_packet_partition",
]

# Names importable from this path before the split that live in the
# submodules now (kept so the search package, lazy candidate factories,
# and type-checking imports resolve without touching the new paths).
from hydra2.search.pbrf_partition import (
    _BELIEF_IMPORT_ERROR as _BELIEF_IMPORT_ERROR,
)
from hydra2.search.pbrf_partition import (
    _KERNEL_IMPORT_ERROR as _KERNEL_IMPORT_ERROR,
)
