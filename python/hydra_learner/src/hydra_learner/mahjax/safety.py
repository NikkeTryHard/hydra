from __future__ import annotations

import importlib
from typing import Any, Final, NamedTuple

HYDRA_TILE_WIDTH: Final = 34
HYDRA_SAFETY_CHANNEL_START: Final = 62
MAHJAX_EXACT_SAFETY_CHANNELS: Final = tuple(range(62, 85))
TENPAI_HINT_THRESHOLD: Final = 0.5
MAHJAX_TSUMOGIRI: Final = 71
MAHJAX_RIICHI: Final = 72


class MahjaxSafetyState(NamedTuple):
    genbutsu_all: Any
    genbutsu_tedashi: Any
    genbutsu_riichi_era: Any
    suji: Any
    half_suji: Any
    matagi: Any
    visible_counts: Any
    kabe: Any
    one_chance: Any
    opponent_riichi: Any
    cached_tenpai_prob: Any


def _jnp() -> Any:
    return importlib.import_module("jax.numpy")


def _jax() -> Any:
    return importlib.import_module("jax")


def empty_safety_state_jax() -> MahjaxSafetyState:
    jnp = _jnp()
    return MahjaxSafetyState(
        genbutsu_all=jnp.zeros((3, HYDRA_TILE_WIDTH), dtype=jnp.bool_),
        genbutsu_tedashi=jnp.zeros((3, HYDRA_TILE_WIDTH), dtype=jnp.bool_),
        genbutsu_riichi_era=jnp.zeros((3, HYDRA_TILE_WIDTH), dtype=jnp.bool_),
        suji=jnp.zeros((3, HYDRA_TILE_WIDTH), dtype=jnp.float32),
        half_suji=jnp.zeros((3, HYDRA_TILE_WIDTH), dtype=jnp.bool_),
        matagi=jnp.zeros((3, HYDRA_TILE_WIDTH), dtype=jnp.float32),
        visible_counts=jnp.zeros((HYDRA_TILE_WIDTH,), dtype=jnp.uint8),
        kabe=jnp.zeros((HYDRA_TILE_WIDTH,), dtype=jnp.bool_),
        one_chance=jnp.zeros((HYDRA_TILE_WIDTH,), dtype=jnp.bool_),
        opponent_riichi=jnp.zeros((3,), dtype=jnp.bool_),
        cached_tenpai_prob=jnp.zeros((3,), dtype=jnp.float32),
    )


def empty_safety_state_batch_jax(batch_size: int) -> MahjaxSafetyState:
    jax = _jax()
    return jax.vmap(lambda _: empty_safety_state_jax())(_jnp().arange(batch_size, dtype=_jnp().int32))


def empty_safety_bank_jax() -> MahjaxSafetyState:
    jax = _jax()
    return jax.vmap(lambda _: empty_safety_state_jax())(_jnp().arange(4, dtype=_jnp().int32))


def empty_safety_bank_batch_jax(batch_size: int) -> MahjaxSafetyState:
    jax = _jax()
    return jax.vmap(lambda _: empty_safety_bank_jax())(_jnp().arange(batch_size, dtype=_jnp().int32))


def select_observer_safety_jax(bank: MahjaxSafetyState, observer: Any) -> MahjaxSafetyState:
    jnp = _jnp()
    observer_i = jnp.asarray(observer, dtype=jnp.int32)
    return MahjaxSafetyState(
        genbutsu_all=bank.genbutsu_all[observer_i],
        genbutsu_tedashi=bank.genbutsu_tedashi[observer_i],
        genbutsu_riichi_era=bank.genbutsu_riichi_era[observer_i],
        suji=bank.suji[observer_i],
        half_suji=bank.half_suji[observer_i],
        matagi=bank.matagi[observer_i],
        visible_counts=bank.visible_counts[observer_i],
        kabe=bank.kabe[observer_i],
        one_chance=bank.one_chance[observer_i],
        opponent_riichi=bank.opponent_riichi[observer_i],
        cached_tenpai_prob=bank.cached_tenpai_prob[observer_i],
    )


def set_observer_safety_jax(bank: MahjaxSafetyState, observer: Any, safety: MahjaxSafetyState) -> MahjaxSafetyState:
    jnp = _jnp()
    observer_i = jnp.asarray(observer, dtype=jnp.int32)
    return MahjaxSafetyState(
        genbutsu_all=bank.genbutsu_all.at[observer_i].set(safety.genbutsu_all),
        genbutsu_tedashi=bank.genbutsu_tedashi.at[observer_i].set(safety.genbutsu_tedashi),
        genbutsu_riichi_era=bank.genbutsu_riichi_era.at[observer_i].set(safety.genbutsu_riichi_era),
        suji=bank.suji.at[observer_i].set(safety.suji),
        half_suji=bank.half_suji.at[observer_i].set(safety.half_suji),
        matagi=bank.matagi.at[observer_i].set(safety.matagi),
        visible_counts=bank.visible_counts.at[observer_i].set(safety.visible_counts),
        kabe=bank.kabe.at[observer_i].set(safety.kabe),
        one_chance=bank.one_chance.at[observer_i].set(safety.one_chance),
        opponent_riichi=bank.opponent_riichi.at[observer_i].set(safety.opponent_riichi),
        cached_tenpai_prob=bank.cached_tenpai_prob.at[observer_i].set(safety.cached_tenpai_prob),
    )


def _base_tile_jax(tile: Any) -> Any:
    jnp = _jnp()
    tile_i = jnp.asarray(tile, dtype=jnp.int32)
    return jnp.select(
        [tile_i == 34, tile_i == 35, tile_i == 36],
        [jnp.asarray(4, dtype=jnp.int32), jnp.asarray(13, dtype=jnp.int32), jnp.asarray(22, dtype=jnp.int32)],
        default=tile_i,
    )


def _recompute_center_suji(genbutsu: Any, suji: Any, half_suji: Any, opp: Any) -> tuple[Any, Any]:
    jnp = _jnp()
    centers = jnp.asarray([3, 4, 5, 12, 13, 14, 21, 22, 23], dtype=jnp.int32)
    low = centers - 3
    high = centers + 3
    p_low = genbutsu[opp, low]
    p_high = genbutsu[opp, high]
    both = p_low & p_high
    one = p_low ^ p_high
    values = jnp.where(both, 1.0, jnp.where(one, 0.5, 0.0)).astype(jnp.float32)
    return suji.at[opp, centers].set(values), half_suji.at[opp, centers].set(one)


def _update_suji_for_tile(state: MahjaxSafetyState, opp: Any, tile: Any) -> tuple[Any, Any]:
    jnp = _jnp()
    suit_offset = (tile // 9) * 9
    number = tile - suit_offset
    suji = state.suji
    low_tile = suit_offset + number - 3
    high_tile = suit_offset + number + 3
    suji = suji.at[opp, low_tile].set(jnp.where(number >= 3, 1.0, suji[opp, low_tile]))
    suji = suji.at[opp, high_tile].set(jnp.where(number + 3 < 9, 1.0, suji[opp, high_tile]))
    return _recompute_center_suji(state.genbutsu_all, suji, state.half_suji, opp)


def _record_visible_tile(state: MahjaxSafetyState, tile: Any) -> tuple[Any, Any, Any]:
    jnp = _jnp()
    counts = state.visible_counts.at[tile].set(jnp.minimum(state.visible_counts[tile] + jnp.uint8(1), jnp.uint8(255)))
    return counts, state.kabe.at[tile].set(counts[tile] >= 4), state.one_chance.at[tile].set(counts[tile] == 3)


def on_discard_jax(state: MahjaxSafetyState, tile: Any, opponent_idx: Any, is_tedashi: Any) -> MahjaxSafetyState:
    jnp = _jnp()
    jax = _jax()
    tile_i = jnp.asarray(tile, dtype=jnp.int32)
    opp = jnp.asarray(opponent_idx, dtype=jnp.int32)
    valid = (tile_i >= 0) & (tile_i < HYDRA_TILE_WIDTH) & (opp >= 0) & (opp < 3)

    def apply() -> MahjaxSafetyState:
        genbutsu_all = state.genbutsu_all.at[opp, tile_i].set(True)
        genbutsu_tedashi = state.genbutsu_tedashi.at[opp, tile_i].set(
            jnp.asarray(is_tedashi, dtype=jnp.bool_) | state.genbutsu_tedashi[opp, tile_i]
        )
        genbutsu_riichi_era = state.genbutsu_riichi_era.at[opp, tile_i].set(
            state.opponent_riichi[opp] | state.genbutsu_riichi_era[opp, tile_i]
        )
        with_genbutsu = state._replace(
            genbutsu_all=genbutsu_all,
            genbutsu_tedashi=genbutsu_tedashi,
            genbutsu_riichi_era=genbutsu_riichi_era,
        )
        visible_counts, kabe, one_chance = _record_visible_tile(with_genbutsu, tile_i)
        suji, half_suji = jax.lax.cond(
            tile_i < 27,
            lambda: _update_suji_for_tile(with_genbutsu, opp, tile_i),
            lambda: (with_genbutsu.suji, with_genbutsu.half_suji),
        )
        suit_pos = tile_i % 9
        left = tile_i - 1
        right = tile_i + 1
        is_tedashi_bool = jnp.asarray(is_tedashi, dtype=jnp.bool_)
        matagi = with_genbutsu.matagi
        matagi = matagi.at[opp, left].set(
            jnp.where(is_tedashi_bool & (tile_i < 27) & (suit_pos > 0), 1.0, matagi[opp, left])
        )
        matagi = matagi.at[opp, right].set(
            jnp.where(is_tedashi_bool & (tile_i < 27) & (suit_pos < 8), 1.0, matagi[opp, right])
        )
        return with_genbutsu._replace(
            suji=suji,
            half_suji=half_suji,
            matagi=matagi,
            visible_counts=visible_counts,
            kabe=kabe,
            one_chance=one_chance,
        )

    return jax.lax.cond(valid, apply, lambda: state)


def on_riichi_jax(state: MahjaxSafetyState, opponent_idx: Any) -> MahjaxSafetyState:
    jnp = _jnp()
    jax = _jax()
    opp = jnp.asarray(opponent_idx, dtype=jnp.int32)
    valid = (opp >= 0) & (opp < 3)
    return jax.lax.cond(
        valid, lambda: state._replace(opponent_riichi=state.opponent_riichi.at[opp].set(True)), lambda: state
    )


def on_call_jax(state: MahjaxSafetyState, tiles34: Any) -> MahjaxSafetyState:
    jax = _jax()
    jnp = _jnp()
    tiles = jnp.asarray(tiles34, dtype=jnp.int32)

    def body(idx: Any, carry: MahjaxSafetyState) -> MahjaxSafetyState:
        tile = tiles[idx]
        valid = (tile >= 0) & (tile < HYDRA_TILE_WIDTH)

        def apply() -> MahjaxSafetyState:
            visible_counts, kabe, one_chance = _record_visible_tile(carry, tile)
            return carry._replace(visible_counts=visible_counts, kabe=kabe, one_chance=one_chance)

        return jax.lax.cond(valid, apply, lambda: carry)

    return jax.lax.fori_loop(0, tiles.shape[0], body, state)


def update_safety_bank_for_action_jax(
    bank: MahjaxSafetyState, actor: Any, action87: Any, last_draw: Any
) -> MahjaxSafetyState:
    jnp = _jnp()
    jax = _jax()
    actor_i = jnp.asarray(actor, dtype=jnp.int32)
    action_i = jnp.asarray(action87, dtype=jnp.int32)
    tile = jnp.where(action_i == MAHJAX_TSUMOGIRI, _base_tile_jax(last_draw), _base_tile_jax(action_i))
    is_discard = ((action_i >= 0) & (action_i <= 36)) | (action_i == MAHJAX_TSUMOGIRI)
    is_tedashi = action_i != MAHJAX_TSUMOGIRI
    is_riichi = action_i == MAHJAX_RIICHI

    def update_observer(observer: Any, carry: MahjaxSafetyState) -> MahjaxSafetyState:
        observer_i = jnp.asarray(observer, dtype=jnp.int32)
        rel = (actor_i - observer_i) % 4
        opp_idx = rel - 1
        observer_state = select_observer_safety_jax(carry, observer_i)

        def apply_discard() -> MahjaxSafetyState:
            updated = on_discard_jax(observer_state, tile, opp_idx, is_tedashi)
            return set_observer_safety_jax(carry, observer_i, updated)

        def apply_riichi() -> MahjaxSafetyState:
            updated = on_riichi_jax(observer_state, opp_idx)
            return set_observer_safety_jax(carry, observer_i, updated)

        carry = jax.lax.cond((observer_i != actor_i) & is_discard, apply_discard, lambda: carry)
        return jax.lax.cond((observer_i != actor_i) & is_riichi, apply_riichi, lambda: carry)

    return jax.lax.fori_loop(0, 4, update_observer, bank)


def update_safety_bank_batch_for_action_jax(
    bank: MahjaxSafetyState, actor: Any, action87: Any, last_draw: Any
) -> MahjaxSafetyState:
    jax = _jax()
    return jax.vmap(update_safety_bank_for_action_jax)(bank, actor, action87, last_draw)


def set_tenpai_prediction_jax(state: MahjaxSafetyState, opponent_idx: Any, probability: Any) -> MahjaxSafetyState:
    jnp = _jnp()
    opp = jnp.asarray(opponent_idx, dtype=jnp.int32)
    prob = jnp.clip(jnp.asarray(probability, dtype=jnp.float32), 0.0, 1.0)
    return state._replace(cached_tenpai_prob=state.cached_tenpai_prob.at[opp].set(prob))


def encode_tenpai_hint_channels_jax(state: MahjaxSafetyState) -> Any:
    jnp = _jnp()
    active = state.opponent_riichi | (state.cached_tenpai_prob > TENPAI_HINT_THRESHOLD)
    return active[:, None].repeat(HYDRA_TILE_WIDTH, axis=1).astype(jnp.float32)


def encode_safety_channels_jax(state: MahjaxSafetyState) -> Any:
    jnp = _jnp()
    return jnp.concatenate(
        [
            state.genbutsu_all.astype(jnp.float32),
            state.genbutsu_tedashi.astype(jnp.float32),
            state.genbutsu_riichi_era.astype(jnp.float32),
            state.suji.astype(jnp.float32),
            state.half_suji.astype(jnp.float32),
            state.matagi.astype(jnp.float32),
            state.kabe[None, :].astype(jnp.float32),
            state.one_chance[None, :].astype(jnp.float32),
            encode_tenpai_hint_channels_jax(state),
        ],
        axis=0,
    )
