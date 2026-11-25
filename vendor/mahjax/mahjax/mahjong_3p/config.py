"""The rule presets of the 3-player game."""

from __future__ import annotations

from mahjax.common.config import GameConfig

# Tenhou's 三人麻雀: red 5p and 5s, double ron, pao and the special abortive draws a
# 3-player table has (nine terminals, four kans).
RED = GameConfig()
NO_RED = GameConfig(
    use_red_fives=False,
    allow_double_ron=False,
    enable_special_abortive_draw=False,
    enable_pao=False,
)
