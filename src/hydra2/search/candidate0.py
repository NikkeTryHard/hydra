"""Candidate 0 frozen policy — SPEC 16.1, Blueprint §7.

Re-export facade over the split modules: :mod:`hydra2.search.candidate0_frozen`
(frozen choice, digest loaders, spec factory) and
:mod:`hydra2.search.candidate0_act` (single-evaluation act path plus
:class:`FrozenCandidate0`). Import from this path; public name and ``__all__``
are unchanged. One model evaluation only, no particles/search/pondering/
learning. Greedy, frozen-temperature and value tie-break arms. Deadline
fallback is Candidate 0 itself.
"""

from __future__ import annotations

from hydra2.search.candidate0_act import FrozenCandidate0 as FrozenCandidate0
from hydra2.search.candidate0_act import candidate0 as candidate0
from hydra2.search.candidate0_frozen import frozen_choice as frozen_choice
from hydra2.search.candidate0_frozen import make_candidate0_spec as make_candidate0_spec

__all__ = [
    "FrozenCandidate0",
    "candidate0",
    "frozen_choice",
    "make_candidate0_spec",
]

# Names importable from this path before the split that live in the
# submodules now (kept so the search package, lazy candidate factories,
# and type-checking imports resolve without touching the new paths).
from hydra2.search.candidate0_act import _action_context_from_obs as _action_context_from_obs
from hydra2.search.candidate0_frozen import _file_sha256 as _file_sha256
from hydra2.search.candidate0_frozen import _load_default_hashes as _load_default_hashes
from hydra2.search.candidate0_frozen import _model_hash_from_identity as _model_hash_from_identity
