"""WP-04C MahJax differential runner (BUILD checklist items 1-3, 5).

Re-export facade over the split modules:
:mod:`hydra2.engines.mahjax.differential_projection` (wall translation,
state surgery, scenario/action records, script mapping, projection
readers), :mod:`hydra2.engines.mahjax.differential_compare` (checkpoint
comparator, action lookup, persistence, token publication),
:mod:`hydra2.engines.mahjax.differential_cases` (scenario registry),
:mod:`hydra2.engines.mahjax.differential_modes` (execution-mode sweep and
soak probes), and :mod:`hydra2.engines.mahjax.differential_runner`
(declared intersection, exclusions, suite runner). Import from this path;
it preserves every public name and ``__all__``.
"""

from __future__ import annotations

from hydra2.engines.mahjax.differential_cases import SCENARIO_REGISTRY as SCENARIO_REGISTRY
from hydra2.engines.mahjax.differential_modes import cpu_soak as cpu_soak
from hydra2.engines.mahjax.differential_modes import execution_mode_sweep as execution_mode_sweep
from hydra2.engines.mahjax.differential_modes import gpu_soak_probe as gpu_soak_probe
from hydra2.engines.mahjax.differential_projection import DEAD_WALL_ROLE_MAP as DEAD_WALL_ROLE_MAP
from hydra2.engines.mahjax.differential_projection import (
    MAHJAX_LIVE_DRAW_COUNT as MAHJAX_LIVE_DRAW_COUNT,
)
from hydra2.engines.mahjax.differential_projection import CheckpointFailure as CheckpointFailure
from hydra2.engines.mahjax.differential_projection import DifferentialResult as DifferentialResult
from hydra2.engines.mahjax.differential_projection import RoundProjection as RoundProjection
from hydra2.engines.mahjax.differential_projection import Scenario as Scenario
from hydra2.engines.mahjax.differential_projection import (
    build_seeded_round_state as build_seeded_round_state,
)
from hydra2.engines.mahjax.differential_projection import jnp_nonzero as jnp_nonzero
from hydra2.engines.mahjax.differential_projection import (
    make_single_round_env as make_single_round_env,
)
from hydra2.engines.mahjax.differential_projection import (
    map_script_step_to_mahjax as map_script_step_to_mahjax,
)
from hydra2.engines.mahjax.differential_projection import wall_to_mahjax_deck as wall_to_mahjax_deck
from hydra2.engines.mahjax.differential_runner import (
    CONVERGENT_DORA_INDICATOR_TYPES as CONVERGENT_DORA_INDICATOR_TYPES,
)
from hydra2.engines.mahjax.differential_runner import DECLARED_INTERSECTION as DECLARED_INTERSECTION
from hydra2.engines.mahjax.differential_runner import EXCLUDED_DIMENSIONS as EXCLUDED_DIMENSIONS
from hydra2.engines.mahjax.differential_runner import run_differential as run_differential

__all__ = [
    "CONVERGENT_DORA_INDICATOR_TYPES",
    "DEAD_WALL_ROLE_MAP",
    "DECLARED_INTERSECTION",
    "EXCLUDED_DIMENSIONS",
    "MAHJAX_LIVE_DRAW_COUNT",
    "SCENARIO_REGISTRY",
    "DifferentialResult",
    "Scenario",
    "build_seeded_round_state",
    "cpu_soak",
    "execution_mode_sweep",
    "gpu_soak_probe",
    "run_differential",
    "wall_to_mahjax_deck",
]
