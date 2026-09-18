"""WP-04C MahJax differential — wall translation and state projections.

Pure-move part of :mod:`hydra2.engines.mahjax.differential`; import from
that path. Covers wall-to-deck translation, deterministic round-state
surgery, scenario/action records, the script mapper, auto policies, and the
reference/mahjax projection readers compared at checkpoints.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, TypedDict, cast

import numpy as np

from hydra2.conformance.runner import ScriptedDecision, TraceRunnerError
from hydra2.conformance.walls import build_wall, haipai_index, type_id

__all__ = [
    "DEAD_WALL_ROLE_MAP",
    "MAHJAX_LIVE_DRAW_COUNT",
    "CheckpointFailure",
    "DifferentialResult",
    "RoundProjection",
    "Scenario",
    "build_seeded_round_state",
    "jnp_nonzero",
    "make_single_round_env",
    "map_script_step_to_mahjax",
    "wall_to_mahjax_deck",
]


MAHJAX_LIVE_DRAW_COUNT = 70  # deck[83] .. deck[14]
_MAHJAX_FIRST_DRAW_IDX = 83

#: hydra2 dead-wall index -> mahjax deck index for the 14 semantic slots.
DEAD_WALL_ROLE_MAP: dict[int, int] = {
    **{131 - 2 * k: 9 - 2 * k for k in range(5)},  # indicators
    **{130 - 2 * k: 8 - 2 * k for k in range(5)},  # ura indicators
    **{135 - j: 10 + j for j in range(4)},  # rinshan stack
}


def _haipai_role_map() -> dict[int, int]:
    """hydra2 haipai index -> mahjax deck index (84+13p+j)."""
    mapping: dict[int, int] = {}
    for seat in range(4):
        for position in range(13):
            mapping[haipai_index(seat, position)] = 84 + 13 * seat + position
    return mapping


def wall_to_mahjax_deck(wall: tuple[int, ...]) -> tuple[int, ...]:
    """Translate a 136-tile hydra2 wall into a mahjax type deck.

    Every consumed index is placed by semantic role; unconsumed indices fall
    back to identity so both engines' unpinned regions agree trivially.
    """
    if len(wall) != 136:
        raise ValueError(f"wall must hold 136 tiles, got {len(wall)}")
    role = _haipai_role_map()
    role.update(DEAD_WALL_ROLE_MAP)
    for k in range(MAHJAX_LIVE_DRAW_COUNT):
        role[52 + k] = _MAHJAX_FIRST_DRAW_IDX - k
    # identity fallback (corrected: values are tile ids,
    # not indices; previously wall[index] re-indexed)
    deck: list[int] = [type_id(t) for t in wall]
    for hydra_index, deck_index in role.items():
        deck[deck_index] = type_id(wall[hydra_index])
    counts = np.bincount(np.asarray(deck, dtype=np.int64), minlength=34)
    if int(counts.max()) > 4:
        raise ValueError("translated deck exceeds four copies of a tile type")
    return tuple(deck)


# ---------------------------------------------------------------------------
# mahjax side: deterministic round-state surgery.
# ---------------------------------------------------------------------------

_JAX_STATE: dict[str, Any] = {}


def _mahjax_modules() -> Any:
    """Import and cache the mahjax internals used for surgery."""
    if len(_JAX_STATE) == 0:
        import jax
        import jax.numpy as jnp
        from mahjax.no_red_mahjong import env as menv
        from mahjax.no_red_mahjong.action import Action
        from mahjax.no_red_mahjong.env import NoRedMahjong
        from mahjax.no_red_mahjong.hand import Hand
        from mahjax.no_red_mahjong.shanten import Shanten
        from mahjax.no_red_mahjong.tile import Tile

        _JAX_STATE.update(
            jax=jax,
            jnp=jnp,
            menv=menv,
            Action=Action,
            NoRedMahjong=NoRedMahjong,
            Hand=Hand,
            Shanten=Shanten,
            Tile=Tile,
        )
    return _JAX_STATE


def make_single_round_env() -> Any:
    """One-round mahjax environment (auto round advance never fires)."""
    modules = _mahjax_modules()
    return modules["NoRedMahjong"](round_mode="single", next_round_style="auto")


def build_seeded_round_state(env: Any, deck_types: tuple[int, ...], *, dealer: int = 0) -> Any:
    """Deterministically rebuild what ``env.init`` randomises.

    Surgery steps (documented for the record):
      1. take a fresh ``env.init(PRNGKey(0))`` state as the structural template;
      2. replace ``round_state.deck`` with the translated type deck;
      3. point the initial dora/ura indicator slots at ``deck[9]``/``deck[8]``
         (the same slots ``_init`` reads);
      4. deal haipai with ``Hand.make_init_hand(deck)`` (consumes deck[-52:]
         exactly like ``_init``);
      5. consume the dealer's opening draw from ``deck[83]``, setting
         ``next_deck_ix=82``/``last_draw`` like ``_init``;
      6. rebuild ``can_win`` via the engine's own ``v_can_win`` and the opening
         legal mask via ``_make_legal_action_mask_after_draw`` - all engine
         functions, no hand-baking (upstream #74/cff90d1 removed the stored
         ``shanten_current_player`` field; shanten is computed on demand via
         ``Shanten.number`` in ``_mahjax_shanten``);
      7. pin dealer/winds/target to the identity-seat convention.
    """
    modules: Any = _mahjax_modules()
    jnp: Any = modules["jnp"]  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    menv: Any = modules["menv"]  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    hand_cls: Any = modules["Hand"]  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    template: Any = env.init(modules["jax"].random.PRNGKey(0))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    deck: Any = jnp.asarray(list(deck_types), dtype=jnp.int8)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    hands: Any = hand_cls.make_init_hand(deck)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    first: Any = deck[_MAHJAX_FIRST_DRAW_IDX]  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    hand_dealer: Any = hand_cls.add(hands[dealer], first)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    can_win: Any = menv.v_can_win(hands, menv.TILE_RANGE)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    players: Any = dataclasses.replace(  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        cast("Any", template.players),
        hand=cast("Any", hands.at[dealer].set(hand_dealer)),
        can_win=can_win,  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    )
    neg: Any = jnp.int8(-1)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    round_state: Any = dataclasses.replace(  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        cast("Any", template.round_state),  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        deck=deck,  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        dora_indicators=cast("Any", jnp.array([deck[9], neg, neg, neg, neg], dtype=jnp.int8)),  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        ura_dora_indicators=cast("Any", jnp.array([deck[8], neg, neg, neg, neg], dtype=jnp.int8)),  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        next_deck_ix=cast("Any", jnp.int32(_MAHJAX_FIRST_DRAW_IDX - 1)),  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        last_draw=cast("Any", jnp.int8(first)),  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        draw_next=cast("Any", jnp.bool_(False)),  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        dealer=cast("Any", jnp.int8(dealer)),  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        init_wind=cast("Any", menv._calc_wind(jnp.int8(dealer))),  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        seat_wind=cast("Any", menv._calc_wind(jnp.int8(dealer))),  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        last_player=cast("Any", jnp.int8(-1)),  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    )
    base: Any = _replace(  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        template,  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        round_state=round_state,  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        players=players,  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        current_player=cast("Any", jnp.int8(dealer)),  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        target=cast("Any", jnp.int8(-1)),  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    )
    mask_current: Any = menv._make_legal_action_mask_after_draw(  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        base,
        players.hand.at[dealer].set(hand_dealer),
        jnp.int8(dealer),
        first,  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    )
    legal_mask: Any = jnp.zeros((4, modules["Action"].NUM_ACTION), dtype=jnp.bool_)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    legal_mask = legal_mask.at[dealer, :].set(mask_current)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    # NOTE (mahjax cff90d1/#74): RoundState.shanten_current_player no longer
    # exists; shanten is computed on demand via Shanten.number (see
    # _mahjax_shanten and _vmap_batch_shanten). No shanten value is stored here.
    return _replace(
        base,
        target=cast("Any", jnp.int8(-1)),
        legal_action_mask=cast("Any", legal_mask),
    )


def _replace(state: Any, **updates: Any) -> Any:
    """Engine-native field replacement (mirrors env._replace_state)."""
    return _mahjax_modules()["menv"]._replace_state(state, **updates)


# ---------------------------------------------------------------------------
# Scenario definition.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Scenario:
    """One differential case: identical inputs for both engines."""

    case_id: str
    title: str
    rule_fields: tuple[str, ...]
    evidence: tuple[str, ...]
    hands: dict[int, dict[int, int]]
    live_draws: dict[int, int]
    script: tuple[ScriptedDecision, ...]
    dead_wall: dict[int, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reference-side driver (interleaved checkpoints).
# ---------------------------------------------------------------------------

_ACTION_PON = 72
_ACTION_OPEN_KAN = 73
_ACTION_CHI_L, _ACTION_CHI_R = 74, 76
_ACTION_PASS = 77
_ACTION_RIICHI = 69
_ACTION_TSUMOGIRI = 68
_ACTION_RON = 71
_ACTION_TSUMO = 70


@dataclass(slots=True)
class CheckpointFailure:
    """First divergence observed at one checkpoint."""

    case_id: str
    step_index: int
    dimension: str
    detail: str


def map_script_step_to_mahjax(decision: ScriptedDecision, action: Any) -> list[int]:
    """Translate one canonical action into primitive mahjax actions.

    ``riichi_discard`` is canonical-declaration-plus-discard on hydra2 and two
    sequential mahjax steps (RIICHI then the discard).
    """
    del decision
    kind: Any = action.kind  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    if kind == "pass":
        return [_ACTION_PASS]
    if kind == "tsumogiri":
        return [_ACTION_TSUMOGIRI]
    if kind == "discard":
        # action.tile is physical id; map to type
        if action.tile is None:
            # fallback via consumed_tiles for robustness
            if getattr(action, "consumed_tiles", None) is not None:
                return [type_id(int(cast("Any", action.consumed_tiles[0])))]  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            raise TraceRunnerError(f"discard action missing tile: {action}")
        return [type_id(int(cast("Any", action.tile)))]  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    if kind == "riichi_discard":
        if action.tile is None:
            raise TraceRunnerError(f"riichi_discard missing tile: {action}")
        return [_ACTION_RIICHI, type_id(int(cast("Any", action.tile)))]  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    if kind == "pon":
        return [_ACTION_PON]
    if kind == "daiminkan":
        return [_ACTION_OPEN_KAN]
    if kind == "ankan":
        # reference ankan has tile=None but consumed_tiles holds the quad
        tile: Any = action.tile  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        if tile is None:
            if getattr(action, "consumed_tiles", None) is None:
                raise TraceRunnerError(f"ankan missing tile info: {action}")
            tile = int(cast("Any", action.consumed_tiles[0]))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        return [34 + type_id(int(tile))]
    if kind == "kakan":
        tile: Any = action.tile  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        if tile is None:
            # engine 0.4.10 stores added tile via _kakan_added capture;
            # consumed_tiles or tile may be None; fallback to consumed
            if getattr(action, "consumed_tiles", None) is not None:
                tile = int(cast("Any", action.consumed_tiles[0]))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            elif hasattr(action, "tile") and action.tile is not None:
                tile = int(cast("Any", action.tile))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            else:
                # last resort: try to infer from action metadata
                raise TraceRunnerError(f"kakan missing tile info: {action}")
        return [34 + type_id(int(tile))]
    if kind == "chi":
        types = sorted(
            type_id(int(cast("Any", t))) for t in (action.called_tile, *action.consumed_tiles)
        )  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        called = type_id(int(cast("Any", action.called_tile)))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        low, mid, high = types
        if called == low:
            return [_ACTION_CHI_L]
        if called == high:
            return [_ACTION_CHI_R]
        assert called == mid
        return [_ACTION_CHI_L + 1]
    if kind == "ron":
        return [_ACTION_RON]
    if kind == "tsumo":
        return [_ACTION_TSUMO]
    raise TraceRunnerError(f"no mahjax mapping for canonical kind {kind!r}")


def _mahjax_auto_policy(state: Any) -> int:
    """Mirror conformance.runner._auto_action semantics on the mahjax side."""
    modules = _mahjax_modules()
    mask: Any = state.legal_action_mask  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    if bool(cast("Any", mask[_ACTION_PASS])):  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        return _ACTION_PASS
    if bool(cast("Any", mask[_ACTION_TSUMOGIRI])):  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        return _ACTION_TSUMOGIRI
    discard_actions = [
        a for a in range(cast("Any", modules["Tile"].NUM_TILE_TYPE)) if bool(cast("Any", mask[a]))
    ]  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    if len(discard_actions) != 0:
        return discard_actions[0]
    offered = jnp_nonzero(mask)
    if len(offered) != 0:
        return offered[0]
    raise TraceRunnerError("mahjax auto policy found no neutral action")


def jnp_nonzero(mask: Any) -> list[int]:
    return [int(cast("Any", i)) for i in _mahjax_modules()["jnp"].nonzero(mask)[0].tolist()]  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX


def _reference_discard_types(sim: Any, actor: int) -> set[int]:
    """Legal discard TYPES for the actor on the reference side."""
    from hydra2.contracts.common import Seat

    types: set[int] = set()
    for act in sim.legal_actions(Seat(actor)):
        if cast("Any", act.kind) in ("discard", "tsumogiri") and act.tile is not None:  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            types.add(type_id(int(cast("Any", act.tile))))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        elif act.kind == "discard" and act.tile is None and act.consumed_tiles:
            # fallback for malformed but keep
            types.add(type_id(int(cast("Any", act.consumed_tiles[0]))))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    return types


def _mahjax_discard_types(state: Any) -> set[int]:
    mask: Any = state.legal_action_mask  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    types = {t for t in range(34) if bool(cast("Any", mask[t]))}  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    # tsumogiri is a distinct action (68) but discards the drawn tile's type
    if bool(cast("Any", mask[_ACTION_TSUMOGIRI])):  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        try:
            ld = int(cast("Any", state.round_state.last_draw))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            if 0 <= ld < 34:
                types.add(ld)
        except Exception:  # why-broad: best-effort last-draw read; failure keeps probed set
            pass
    return types


class RoundProjection(TypedDict, total=False):
    current_player: int
    dora_indicators: tuple[int, ...]
    ura_indicators: tuple[int, ...]
    discard_types: set[int]
    shanten: int
    can_win_types: set[int]
    score: tuple[int, ...]


def _reference_shanten(sim: Any, actor: int) -> int:
    """Compute reference shanten via mahjax evaluator on reference hand."""
    modules = _mahjax_modules()
    shanten_cls: Any = modules["Shanten"]  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    # hand is list of physical ids from engine
    engine_pid = int(cast("Any", sim._perm[actor])) if hasattr(sim, "_perm") else actor  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    try:
        hand_list = [int(t) for t in sim._engine.hands[engine_pid]]  # type: ignore[attr-defined]  # reason: no stubs; runtime
    except Exception:  # why-broad: _engine/_env probe; any shape tries _env
        hand_list = [
            int(cast("Any", t)) for t in sim._env.hands[engine_pid]
        ]  # fallback  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    # drawn tile? The hand list may include drawn tile already; we just count
    counts = [0] * 34
    for pid in hand_list:
        counts[pid // 4] += 1
    # If actor just drew, hand includes 14 tiles; else 13.
    # Shanten expects 13? Use current hand as is
    # Trim to 14? Shanten handles any; we pass counts as jnp array
    import jax.numpy as jnp

    arr = jnp.asarray(counts, dtype=jnp.int8)
    return int(cast("Any", shanten_cls.number(arr)))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX


def _mahjax_shanten(state: Any) -> int:
    """Shanten of the current player via the engine's own ``Shanten.number``.

    Upstream #74 (cff90d1) removed ``RoundState.shanten_current_player``; the
    engine no longer stores it. Compute over the actor hand - the same function
    ``_reference_shanten`` already uses. Fail-closed: callers report exceptions
    as checkpoint failures, never a default value.
    """
    modules = _mahjax_modules()
    shanten_cls: Any = modules["Shanten"]  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    current = int(cast("Any", state.current_player))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    hand = cast("Any", state.players.hand[current])  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    return int(cast("Any", shanten_cls.number(hand)))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX


def _reference_dora_types(sim: Any) -> tuple[int, ...]:
    """Reference dora indicator types (physical//4)."""
    try:
        indicators = list(sim._engine.dora_indicators)  # type: ignore[attr-defined]  # reason: no stubs; runtime
    except Exception:  # why-broad: attr probe; any shape falls back to _env, then ()
        try:
            indicators = list(sim._env.dora_indicators)  # type: ignore[attr-defined]  # reason: no stubs; runtime
        except Exception:  # why-broad: attr probe; unknown shape reports ()
            return ()
    types = tuple(
        type_id(int(cast("Any", t)))
        for t in indicators
        if int(cast("Any", t)) != -1 and t is not None
    )  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    return types


def _mahjax_dora_types(state: Any) -> tuple[int, ...]:
    arr: Any = state.round_state.dora_indicators  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    return tuple(int(cast("Any", t)) for t in arr.tolist() if int(cast("Any", t)) != -1)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX


def _reference_win_offer(sim: Any, actor: int) -> bool:
    from hydra2.contracts.common import Seat

    for act in cast("Any", sim).legal_actions(Seat(actor)):
        if cast("Any", cast("Any", act).kind) in ("ron", "tsumo"):
            return True
    return False


def _mahjax_win_offer(state: Any) -> bool:
    mask: Any = cast("Any", state).legal_action_mask
    return bool(cast("Any", cast("Any", mask)[_ACTION_RON] or cast("Any", mask)[_ACTION_TSUMO]))


# ---------------------------------------------------------------------------
# DifferentialResult and helpers for persistence / token.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DifferentialResult:
    """Result of running the full differential suite."""

    verdict: str  # "passed" (zero mismatch) or "blocked"/"mismatch"
    total_cases: int
    passed_cases: int
    failed_cases: int
    mismatches: tuple[CheckpointFailure, ...]
    first_counterexample_path: str | None
    token_path: str | None
    token_digest: str | None
    env_tuple_digest: str
    execution_mode_deterministic: bool
    gpu_probe: dict[str, Any]
    cpu_soak: dict[str, Any]


def _wall_for_scenario(scenario: Scenario) -> tuple[int, ...]:
    return build_wall(
        hands=scenario.hands, live_draws=scenario.live_draws, dead_wall=scenario.dead_wall
    )
