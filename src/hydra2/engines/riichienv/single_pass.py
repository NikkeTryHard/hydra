"""Single-engine wall-less replay: framed MJAI games -> actor ``DecisionRow`` rows (WP-14).

Re-export facade over the split modules:
:mod:`hydra2.engines.riichienv._sp_records`
:mod:`hydra2.engines.riichienv._sp_capture`
:mod:`hydra2.engines.riichienv._sp_walk`
:mod:`hydra2.engines.riichienv._sp_windows`
:mod:`hydra2.engines.riichienv._sp_reach`
:mod:`hydra2.engines.riichienv._sp_kans`
:mod:`hydra2.engines.riichienv._sp_wins`
:mod:`hydra2.engines.riichienv._sp_game`
:mod:`hydra2.engines.riichienv._sp_oracle`
Import from this path; it preserves every public name and ``__all__``.
"""

from __future__ import annotations

from hydra2.engines.riichienv._sp_game import replay_game as replay_game
from hydra2.engines.riichienv._sp_records import SIM_DERIVATION_MARK as SIM_DERIVATION_MARK

__all__ = ["SIM_DERIVATION_MARK", "replay_game"]
