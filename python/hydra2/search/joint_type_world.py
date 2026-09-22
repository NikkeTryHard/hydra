"""Candidate 8 joint type/world model — observation-only opponent types, joint posterior, robust set.

Re-export facade over the split modules: :mod:`hydra2.search.joint_types`
(info keys, type policy, joint particles), :mod:`hydra2.search.joint_uncertainty`
(exact oracle, coherent trajectory, uncertainty set, spec factory), and
:mod:`hydra2.search.joint_planner` (:class:`JointTypeWorldPlanner`). Import from
this path; public name and ``__all__``.
"""

from __future__ import annotations

import math
from typing import Any

from hydra2.contracts.common import ContractError
from hydra2.search.joint_planner import JointTypeWorldPlanner as JointTypeWorldPlanner
from hydra2.search.joint_types import (
    FORBIDDEN_IN_TREE_KEY as FORBIDDEN_IN_TREE_KEY,
)
from hydra2.search.joint_types import (
    THETA_IDS as THETA_IDS,
)
from hydra2.search.joint_types import (
    JointParticle as JointParticle,
)
from hydra2.search.joint_types import (
    JointPosterior as JointPosterior,
)
from hydra2.search.joint_types import (
    OpponentTypePolicy as OpponentTypePolicy,
)
from hydra2.search.joint_types import (
    deterministic_joint_gumbel as deterministic_joint_gumbel,
)
from hydra2.search.joint_types import (
    info_key_for_observation as info_key_for_observation,
)
from hydra2.search.joint_types import (
    validate_hidden_permutation_invariance as validate_hidden_permutation_invariance,
)
from hydra2.search.joint_types import (
    validate_same_information_equality as validate_same_information_equality,
)
from hydra2.search.joint_uncertainty import (
    JointTypeWorldConfig as JointTypeWorldConfig,
)
from hydra2.search.joint_uncertainty import (
    UncertaintySet as UncertaintySet,
)
from hydra2.search.joint_uncertainty import (
    coherent_trajectory as coherent_trajectory,
)
from hydra2.search.joint_uncertainty import (
    exact_joint_posterior_oracle as exact_joint_posterior_oracle,
)
from hydra2.search.joint_uncertainty import (
    hidden_marginalization as hidden_marginalization,
)
from hydra2.search.joint_uncertainty import (
    make_joint_type_world_candidate_spec as make_joint_type_world_candidate_spec,
)
from hydra2.search.joint_uncertainty import (
    preserve_correlation_check as preserve_correlation_check,
)
from hydra2.search.joint_uncertainty import (
    sequential_joint_update as sequential_joint_update,
)

__all__ = [
    "FORBIDDEN_IN_TREE_KEY",
    "JOINT_WORLD_FIRST_EPOCH",
    "JointParticle",
    "JointPosterior",
    "JointTypeWorldConfig",
    "JointTypeWorldPlanner",
    "OpponentTypePolicy",
    "UncertaintySet",
    "coherent_trajectory",
    "deterministic_joint_gumbel",
    "exact_joint_posterior_oracle",
    "hidden_marginalization",
    "info_key_for_observation",
    "joint_world_check_config",
    "joint_world_is_next_epoch",
    "joint_world_next_epoch",
    "make_joint_type_world_candidate_spec",
    "preserve_correlation_check",
    "sequential_joint_update",
    "validate_hidden_permutation_invariance",
    "validate_same_information_equality",
]

# Names importable from this path before the split that live in the
# submodules now (kept so the search package, lazy candidate factories,
# and type-checking imports resolve without touching the new paths).
from hydra2.search.joint_types import (
    _BELIEF_IMPORT_ERROR as _BELIEF_IMPORT_ERROR,
)
from hydra2.search.joint_types import (
    _COMMON_AVAILABLE as _COMMON_AVAILABLE,
)

try:
    from hydra2._native import search as _joint_world_bridge  # pyrefly: ignore[missing-import]
except ImportError:  # pragma: no cover — bridge-less env keeps pure-Python oracles
    _joint_world_bridge = None  # type: ignore[assignment]

_U64_MAX: int = (1 << 64) - 1


def _joint_world_bridge_attr(name: str, default: Any) -> Any:
    """Read one ``hydra2._native.search`` joint-world leaf with a HEAD-literal fallback."""
    try:
        bridge: Any = _joint_world_bridge
        if bridge is None:
            return default
        return getattr(bridge, name, default)
    except Exception:
        return default


JOINT_WORLD_FIRST_EPOCH: int = int(_joint_world_bridge_attr("JOINT_WORLD_FIRST_EPOCH", 0))


def joint_world_next_epoch(prior_epoch: int) -> int:
    """Next epoch for a packet commit — ``prior_epoch + 1`` (caller-increments rule).

    Bridge-first over ``hydra2._native.search.joint_world_next_epoch`` (checked
    ``u64`` ``+1``); ``u64::MAX`` and exotic domains keep the unbounded-Python
    fallback below so no new ``ContractError`` is introduced. Stale-``.so``
    falls through to the same value.
    """
    if isinstance(prior_epoch, bool) or not isinstance(prior_epoch, int) or prior_epoch < 0:
        raise ContractError(f"prior_epoch must be non-negative int, got {prior_epoch!r}")
    if prior_epoch >= _U64_MAX:
        return prior_epoch + 1
    if _joint_world_bridge is not None:
        try:
            fn: Any = getattr(_joint_world_bridge, "joint_world_next_epoch", None)
            if fn is not None:
                return fn(prior_epoch)  # pyrefly: ignore[unknown-argument-type] # untyped bridge epoch fn
        except (ImportError, AttributeError):
            pass  # stale .so: fall through to the oracle below (same value)
        except (ValueError, TypeError) as exc:
            raise ContractError(str(exc)) from exc
        except Exception as exc:
            raise ContractError(f"joint bridge next epoch failed: {exc}") from exc
    return prior_epoch + 1


def joint_world_is_next_epoch(prior_epoch: int, next_epoch: int) -> bool:
    """Stale-reject predicate — ``next_epoch == prior_epoch + 1`` (``False`` on overflow).

    Bridge-first over ``hydra2._native.search.joint_world_is_next_epoch``;
    exotic domains (``bool``/non-``int``/negative) read ``False`` via the
    fallback so no new ``ContractError`` is introduced. Stale-``.so`` falls
    through to the same verdict.
    """
    for _value in (prior_epoch, next_epoch):
        if isinstance(_value, bool) or not isinstance(_value, int) or _value < 0:
            return False
    if prior_epoch > _U64_MAX or next_epoch > _U64_MAX:
        return next_epoch == prior_epoch + 1
    if _joint_world_bridge is not None:
        try:
            fn: Any = getattr(_joint_world_bridge, "joint_world_is_next_epoch", None)
            if fn is not None:
                return fn(prior_epoch, next_epoch)  # pyrefly: ignore[unknown-argument-type] # untyped bridge epoch fn
        except (ImportError, AttributeError):
            pass  # stale .so: fall through to the oracle below (same verdict)
        except (ValueError, TypeError):
            pass  # bool gate never raises: fall through to the comparison
        except Exception as exc:
            raise ContractError(f"joint bridge is-next-epoch failed: {exc}") from exc
    return next_epoch == prior_epoch + 1


def joint_world_check_config(
    *,
    rho: float,
    epsilon: float,
    max_particles: int,
    calibration_threshold: float,
) -> bool:
    """Numeric config gate — ``JointTypeWorldConfig``/``UncertaintySet`` pure lanes.

    Bridge-first over ``hydra2._native.search.joint_world_check_config``
    (``rho`` finite ``>= 0``, ``epsilon`` finite in ``[0, 1]``,
    ``calibration_threshold`` finite, ``max_particles >= 1``); exotic domains
    (``int`` floats, ``bool`` counts) keep the oracle below so the
    ``isinstance(v, float)`` shape is preserved. Stale-``.so`` falls through
    to the same verdict.
    """

    def _oracle() -> bool:
        for _v in (rho, epsilon, calibration_threshold):
            if not isinstance(_v, float) or not math.isfinite(_v):
                return False
        if rho < 0.0:
            return False
        if not 0.0 <= epsilon <= 1.0:
            return False
        if isinstance(max_particles, bool) or not isinstance(max_particles, int):
            return False
        return max_particles > 0

    _plain = (
        isinstance(rho, float)
        and isinstance(epsilon, float)
        and isinstance(calibration_threshold, float)
        and isinstance(max_particles, int)
        and not isinstance(max_particles, bool)
        and 0 <= max_particles <= _U64_MAX
    )
    if _plain and _joint_world_bridge is not None:
        try:
            fn: Any = getattr(_joint_world_bridge, "joint_world_check_config", None)
            if fn is not None:
                return fn(  # pyrefly: ignore[unknown-argument-type] # untyped bridge config fn
                    rho, epsilon, max_particles, calibration_threshold
                )
        except (ImportError, AttributeError):
            pass  # stale .so: fall through to the oracle below (same verdict)
        except Exception as exc:
            raise ContractError(f"joint bridge check_config failed: {exc}") from exc
    return _oracle()
