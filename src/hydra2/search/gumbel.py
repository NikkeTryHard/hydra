"""Candidate 6 Gumbel search — deterministic root Gumbels, sequential halving, exact rules.

Re-export facade over the split modules: :mod:`hydra2.search.gumbel_core`
(firewall vocabulary, deterministic roots, vectors, simulator),
:mod:`hydra2.search.gumbel_config` (frozen configs, per-action stats),
:mod:`hydra2.search.gumbel_search` (continuation policy plus the
:class:`GumbelSearchPlanner` construction/search driver),
:mod:`hydra2.search.gumbel_act` (:class:`GumbelSearchPlanner` Planner
protocol adapter), :mod:`hydra2.search.gumbel_puct` (matched PUCT
comparator), and :mod:`hydra2.search.gumbel_spec` (CandidateSpec
factories). Import from this path; public name and ``__all__``.
"""

from __future__ import annotations

from hydra2.search.gumbel_act import (
    GumbelSearchPlanner as GumbelSearchPlanner,
)
from hydra2.search.gumbel_act import (
    GumbelSearchPlannerActMixin as GumbelSearchPlannerActMixin,
)
from hydra2.search.gumbel_config import (
    GumbelSearchConfig as GumbelSearchConfig,
)
from hydra2.search.gumbel_config import (
    PuctConfig as PuctConfig,
)
from hydra2.search.gumbel_core import (
    FORBIDDEN_IN_TREE_KEY as FORBIDDEN_IN_TREE_KEY,
)
from hydra2.search.gumbel_core import (
    cached_full_history_agreement as cached_full_history_agreement,
)
from hydra2.search.gumbel_core import (
    deterministic_gumbel as deterministic_gumbel,
)
from hydra2.search.gumbel_core import (
    deterministic_root_gumbels as deterministic_root_gumbels,
)
from hydra2.search.gumbel_core import (
    exact_transition as exact_transition,
)
from hydra2.search.gumbel_core import (
    info_key_for_observation as info_key_for_observation,
)
from hydra2.search.gumbel_core import (
    learned_rules_transition_rejected as learned_rules_transition_rejected,
)
from hydra2.search.gumbel_core import (
    model_vector_for_world as model_vector_for_world,
)
from hydra2.search.gumbel_core import (
    scalarize_vector as scalarize_vector,
)
from hydra2.search.gumbel_core import (
    terminal_vector_for_world as terminal_vector_for_world,
)
from hydra2.search.gumbel_core import (
    validate_hidden_permutation_invariance as validate_hidden_permutation_invariance,
)
from hydra2.search.gumbel_core import (
    validate_packet_partition as validate_packet_partition,
)
from hydra2.search.gumbel_puct import PuctBaselinePlanner as PuctBaselinePlanner
from hydra2.search.gumbel_search import (
    GumbelSearchPlannerSearchMixin as GumbelSearchPlannerSearchMixin,
)
from hydra2.search.gumbel_spec import (
    make_gumbel_candidate_spec as make_gumbel_candidate_spec,
)
from hydra2.search.gumbel_spec import (
    make_puct_candidate_spec as make_puct_candidate_spec,
)

__all__ = [
    "FORBIDDEN_IN_TREE_KEY",
    "GumbelSearchConfig",
    "GumbelSearchPlanner",
    "PuctBaselinePlanner",
    "PuctConfig",
    "cached_full_history_agreement",
    "deterministic_gumbel",
    "deterministic_root_gumbels",
    "exact_transition",
    "info_key_for_observation",
    "learned_rules_transition_rejected",
    "make_gumbel_candidate_spec",
    "make_puct_candidate_spec",
    "model_vector_for_world",
    "scalarize_vector",
    "terminal_vector_for_world",
    "validate_hidden_permutation_invariance",
    "validate_packet_partition",
]

# Names importable from this path before the split that live in the
# submodules now (kept so the search package, lazy candidate factories,
# and type-checking imports resolve without touching the new paths).
from hydra2.search.gumbel_config import _ActionStats as _ActionStats
from hydra2.search.gumbel_core import (
    _BELIEF_IMPORT_ERROR as _BELIEF_IMPORT_ERROR,
)
from hydra2.search.gumbel_core import (
    _GUMBEL_SEED_DOMAIN as _GUMBEL_SEED_DOMAIN,
)
from hydra2.search.gumbel_spec import (
    _canonical_hashes as _canonical_hashes,
)
from hydra2.search.gumbel_spec import (
    _derive_utility_manifest_hash as _derive_utility_manifest_hash,
)
