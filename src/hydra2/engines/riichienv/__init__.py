"""RiichiEnv 0.4.8 reference adapter (WP-03A, decision D-003).

Public surface: :class:`RiichiEnvExactSimulator` (SPEC 9 ExactSimulator) and
the import-verified :data:`ENGINE_IDENTITY`.
"""

from __future__ import annotations

from hydra2.engines.riichienv.adapter import RiichiEnvExactSimulator
from hydra2.engines.riichienv.identity import (
    ADAPTER_VERSION,
    ENGINE_IDENTITY,
    ENGINE_NAME,
    RIICHENV_VERSION_PIN,
)

__all__ = [
    "ADAPTER_VERSION",
    "ENGINE_IDENTITY",
    "ENGINE_NAME",
    "RIICHENV_VERSION_PIN",
    "RiichiEnvExactSimulator",
]
