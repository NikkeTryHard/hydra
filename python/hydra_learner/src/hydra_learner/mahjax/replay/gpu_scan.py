from __future__ import annotations

import argparse
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra_learner.mahjax import contract
from hydra_learner.mahjax.compat import (
    HYDRA_PASS,
    MAHJAX_CHI_LEFT,
    MAHJAX_CHI_LEFT_RED,
    MAHJAX_CHI_MID,
    MAHJAX_CHI_MID_RED,
    MAHJAX_CHI_RIGHT,
    MAHJAX_CHI_RIGHT_RED,
    MAHJAX_OPEN_KAN,
    MAHJAX_PASS,
    MAHJAX_PON,
    MAHJAX_PON_RED,
    MAHJAX_RIICHI,
    MAHJAX_RON,
    MAHJAX_SELF_KAN_START,
    MAHJAX_TSUMO,
    MAHJAX_TSUMOGIRI,
)
from hydra_learner.mahjax.constructor import _tile_type, mahjax_state_from_start_kyoku, mjai_tile_to_mahjax_id
from hydra_learner.mahjax.replay.parity import (
    _draw_tiles_for_kyoku,
    _hydra_authority_rows,
    _load_replay_events,
    _next_kyoku_boundary,
    _patch_draw_deck,
)

CLEAR_RESPONSES = -2
NOOP = -1


@dataclass(frozen=True)
class GpuReplayScanResult:
    matched_rows: int
    authority_rows: int
    first_failure: int | None
    elapsed_s: float
    rows_per_s: float


def _contains_red_five(tiles: list[str]) -> bool:
    return any(tile in {"5mr", "5pr", "5sr"} for tile in tiles)


def _action87_from_event(event: dict[str, Any]) -> int | None:
    event_type = event.get("type")
    if event_type == "dahai":
        if bool(event["tsumogiri"]):
            return MAHJAX_TSUMOGIRI
        return mjai_tile_to_mahjax_id(str(event["pai"]))
    if event_type == "reach":
        return MAHJAX_RIICHI
    if event_type == "none":
        return MAHJAX_PASS
    if event_type == "pon":
        return MAHJAX_PON_RED if _contains_red_five(list(event.get("consumed", ()))) else MAHJAX_PON
    if event_type == "chi":
        target = _tile_type(mjai_tile_to_mahjax_id(str(event["pai"])))
        consumed = sorted(_tile_type(mjai_tile_to_mahjax_id(str(tile))) for tile in event.get("consumed", ()))
        sequence = sorted([target, *consumed])
        chi_index = sequence.index(target)
        red = _contains_red_five(list(event.get("consumed", ())))
        if chi_index == 0:
            return MAHJAX_CHI_LEFT_RED if red else MAHJAX_CHI_LEFT
        if chi_index == 1:
            return MAHJAX_CHI_MID_RED if red else MAHJAX_CHI_MID
        return MAHJAX_CHI_RIGHT_RED if red else MAHJAX_CHI_RIGHT
    if event_type in {"kan", "daiminkan"}:
        return MAHJAX_OPEN_KAN
    if event_type == "ankan":
        consumed = list(event.get("consumed", ()))
        if not consumed:
            raise ValueError("ankan command requires consumed tiles")
        return MAHJAX_SELF_KAN_START + _tile_type(mjai_tile_to_mahjax_id(str(consumed[0])))
    if event_type == "kakan":
        return MAHJAX_SELF_KAN_START + _tile_type(mjai_tile_to_mahjax_id(str(event["pai"])))
    if event_type == "hora":
        return MAHJAX_TSUMO if int(event["actor"]) == int(event["target"]) else MAHJAX_RON
    return None


def _state_from_start_event(events: list[dict[str, Any]], start: int, end: int) -> Any:
    start_event = events[start]
    first_tsumo = events[start + 1]
    state = mahjax_state_from_start_kyoku(
        tehais=start_event["tehais"],
        scores=start_event["scores"],
        dora_marker=start_event["dora_marker"],
        oya=int(start_event["oya"]),
        kyoku=int(start_event["kyoku"]),
        honba=int(start_event["honba"]),
        kyotaku=int(start_event["kyotaku"]),
        first_draw=str(first_tsumo["pai"]),
    )
    live_draws, kan_draws = _draw_tiles_for_kyoku(events[start + 2 : end])
    return _patch_draw_deck(state, live_draws, kan_draws)


def _compile_scan(env: Any) -> Any:
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    step_fn = jax.jit(env.step)

    def scan_segment(state: Any, actors: Any, actions: Any, checks: Any, expected_masks: Any, action46s: Any) -> Any:
        def body(carry: tuple[Any, Any, Any], row: tuple[Any, Any, Any, Any, Any]) -> tuple[tuple[Any, Any, Any], Any]:
            current_state, first_failure, index = carry
            actor, action, check, expected, action46 = row
            normal_mask = contract.project_mask_jax(
                current_state.legal_action_mask, current_state.round_state.last_draw
            ).astype(jnp.uint8)
            response_mask = contract.project_mask_jax(
                current_state.players.legal_action_mask[actor], current_state.round_state.last_draw
            ).astype(jnp.uint8)
            response_mask = response_mask.at[HYDRA_PASS].set(
                jnp.where(jnp.any(response_mask), 1, response_mask[HYDRA_PASS])
            )
            actual = jnp.where(current_state.round_state.target >= 0, response_mask, normal_mask)
            mask_ok = jnp.all(actual == expected)
            action_ok = (action46 < 0) | expected[action46]
            row_ok = (check == 0) | (mask_ok & action_ok)
            first_failure = jnp.where((first_failure < 0) & (~row_ok), index, first_failure)

            def do_step(s: Any) -> Any:
                return step_fn(s, action)

            def clear_responses(s: Any) -> Any:
                def cond(loop_state: Any) -> Any:
                    return loop_state.legal_action_mask[MAHJAX_PASS]

                def body_pass(loop_state: Any) -> Any:
                    return step_fn(loop_state, jnp.asarray(MAHJAX_PASS, dtype=jnp.int32))

                return jax.lax.while_loop(cond, body_pass, s)

            def dispatch_step(s: Any) -> Any:
                return jax.lax.cond(
                    action == CLEAR_RESPONSES,
                    clear_responses,
                    lambda inner: jax.lax.cond(action >= 0, do_step, lambda noop: noop, inner),
                    s,
                )

            current_state = dispatch_step(current_state)
            return (current_state, first_failure, index + 1), None

        first_failure = jnp.asarray(-1, dtype=jnp.int32)
        index = jnp.asarray(0, dtype=jnp.int32)
        (_state, first_failure, _index), _ = jax.lax.scan(
            body, (state, first_failure, index), (actors, actions, checks, expected_masks, action46s)
        )
        return first_failure

    return jax.jit(scan_segment)


def _compile_batch_scan(env: Any) -> Any:
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    step_batch = jax.vmap(lambda state, action: env.step(state, action))
    project_mask = jax.vmap(contract.project_mask_jax)

    def scan_batch(
        batch_state: Any, actors: Any, actions: Any, checks: Any, expected_masks: Any, action46s: Any
    ) -> Any:
        batch_size = actions.shape[1]
        batch_index = jnp.arange(batch_size, dtype=jnp.int32)

        def body(carry: tuple[Any, Any, Any], row: tuple[Any, Any, Any, Any, Any]) -> tuple[tuple[Any, Any, Any], Any]:
            current_state, first_failure, index = carry
            actor, action, check, expected, action46 = row
            normal_mask = project_mask(current_state.legal_action_mask, current_state.round_state.last_draw).astype(
                jnp.uint8
            )
            response_mask = project_mask(
                current_state.players.legal_action_mask[batch_index, actor], current_state.round_state.last_draw
            ).astype(jnp.uint8)
            response_any = jnp.any(response_mask, axis=1)
            response_mask = response_mask.at[:, HYDRA_PASS].set(
                jnp.where(response_any, jnp.asarray(1, dtype=jnp.uint8), response_mask[:, HYDRA_PASS])
            )
            actual = jnp.where((current_state.round_state.target >= 0)[:, None], response_mask, normal_mask)
            mask_ok = jnp.all(actual == expected, axis=1)
            action_ok = (action46 < 0) | expected[batch_index, action46]
            row_ok = (check == 0) | (mask_ok & action_ok)
            failing = (first_failure < 0) & (~row_ok)
            first_failure = jnp.where(failing, index, first_failure)
            safe_action = jnp.where(action >= 0, action, jnp.asarray(MAHJAX_PASS, dtype=jnp.int32))
            stepped = step_batch(current_state, safe_action)
            current_state = jax.tree_util.tree_map(
                lambda old, new: jnp.where((action >= 0), new, old), current_state, stepped
            )
            return (current_state, first_failure, index + 1), None

        first_failure = jnp.full((actions.shape[1],), -1, dtype=jnp.int32)
        index = jnp.asarray(0, dtype=jnp.int32)
        (_state, first_failure, _index), _ = jax.lax.scan(
            body, (batch_state, first_failure, index), (actors, actions, checks, expected_masks, action46s)
        )
        return first_failure

    return jax.jit(scan_batch)


def _first_segment_commands(
    events: list[dict[str, Any]], authority: list[dict[str, Any]]
) -> tuple[Any, list[int], list[int], list[int], list[list[int]], list[int]] | None:
    rows_by_event: dict[int, list[dict[str, Any]]] = {}
    for row in authority:
        rows_by_event.setdefault(int(row["event_index"]), []).append(row)
    for start, end in _segments(events):
        state = _state_from_start_event(events, start, end)
        actors: list[int] = []
        actions: list[int] = []
        checks: list[int] = []
        expected_masks: list[list[int]] = []
        action46s: list[int] = []
        for event_index in range(start + 2, end):
            event = events[event_index]
            rows = rows_by_event.get(event_index, [])
            for row in rows:
                actors.append(int(row["actor"]))
                checks.append(1)
                expected_masks.append(list(row["legal_mask"]))
                action46s.append(int(row["action_id"]))
                if row["kind"] == "implicit_pass":
                    actions.append(MAHJAX_PASS)
                else:
                    action = _action87_from_event(event)
                    if action is None:
                        raise ValueError(f"sampled event has no MahJAX action: {event.get('type')}")
                    actions.append(action)
            if not rows:
                action = _action87_from_event(event)
                if action is not None:
                    actors.append(int(event.get("actor", 0)))
                    checks.append(0)
                    expected_masks.append([0] * 46)
                    action46s.append(-1)
                    actions.append(action)
        if actions:
            return state, actors, actions, checks, expected_masks, action46s
    return None


def validate_replays_gpu_batch(paths: list[Path], *, row_limit: int | None = None) -> dict[str, Any]:
    jax = importlib.import_module("jax")
    jnp = importlib.import_module("jax.numpy")
    mahjax = importlib.import_module("mahjax")
    env = mahjax.make("red_mahjong", observe_type="dict")
    scan_batch = _compile_batch_scan(env)
    prepared: list[tuple[Any, list[int], list[int], list[int], list[list[int]], list[int], int]] = []
    for path in paths:
        events = _load_replay_events(path, None)
        authority = _hydra_authority_rows(path, row_limit)
        segment = _first_segment_commands(events, authority)
        if segment is None:
            continue
        state, actors, actions, checks, expected_masks, action46s = segment
        prepared.append((state, actors, actions, checks, expected_masks, action46s, len(authority)))
    if not prepared:
        raise ValueError("no replay segment commands built")
    max_len = max(len(item[2]) for item in prepared)
    states = [item[0] for item in prepared]
    batch_state = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)

    def pad(values: list[int], fill: int) -> list[int]:
        return [*values, *([fill] * (max_len - len(values)))]

    actors = jnp.asarray([pad(item[1], 0) for item in prepared], dtype=jnp.int32).T
    actions = jnp.asarray([pad(item[2], NOOP) for item in prepared], dtype=jnp.int32).T
    checks = jnp.asarray([pad(item[3], 0) for item in prepared], dtype=jnp.uint8).T
    expected = jnp.asarray(
        [item[4] + [[0] * 46] * (max_len - len(item[4])) for item in prepared], dtype=jnp.uint8
    ).transpose((1, 0, 2))
    action46s = jnp.asarray([pad(item[5], -1) for item in prepared], dtype=jnp.int32).T
    started = time.perf_counter()
    failures = scan_batch(batch_state, actors, actions, checks, expected, action46s)
    failures.block_until_ready()
    elapsed = time.perf_counter() - started
    check_counts = [sum(item[3]) for item in prepared]
    matched = sum(
        count if int(failures[index]) < 0 else int(failures[index]) for index, count in enumerate(check_counts)
    )
    return {
        "segments": len(prepared),
        "max_commands": max_len,
        "matched_rows": matched,
        "authority_rows": sum(item[6] for item in prepared),
        "first_failures": [int(value) for value in list(failures)],
        "elapsed_s": elapsed,
        "rows_per_s": 0.0 if elapsed <= 0.0 else matched / elapsed,
    }


def _segments(events: list[dict[str, Any]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for index, event in enumerate(events):
        if event.get("type") == "start_kyoku" and index + 1 < len(events) and events[index + 1].get("type") == "tsumo":
            out.append((index, _next_kyoku_boundary(events, index + 2)))
    return out


def validate_replay_gpu_scan(path: Path, *, row_limit: int | None = None) -> GpuReplayScanResult:
    jnp = importlib.import_module("jax.numpy")
    mahjax = importlib.import_module("mahjax")
    env = mahjax.make("red_mahjong", observe_type="dict")
    scan_segment = _compile_scan(env)
    events = _load_replay_events(path, None)
    authority = _hydra_authority_rows(path, row_limit)
    rows_by_event: dict[int, list[dict[str, Any]]] = {}
    for row in authority:
        rows_by_event.setdefault(int(row["event_index"]), []).append(row)

    started = time.perf_counter()
    matched = 0
    first_failure: int | None = None
    for start, end in _segments(events):
        if matched >= len(authority):
            break
        state = _state_from_start_event(events, start, end)
        actors: list[int] = []
        actions: list[int] = []
        checks: list[int] = []
        expected_masks: list[list[int]] = []
        action46s: list[int] = []
        row_base = matched
        for event_index in range(start + 2, end):
            event = events[event_index]
            rows = rows_by_event.get(event_index, [])
            for row in rows:
                actors.append(int(row["actor"]))
                checks.append(1)
                expected_masks.append(list(row["legal_mask"]))
                action46s.append(int(row["action_id"]))
                if row["kind"] == "implicit_pass":
                    actions.append(MAHJAX_PASS)
                else:
                    action = _action87_from_event(event)
                    if action is None:
                        raise ValueError(f"sampled event has no MahJAX action: {event.get('type')}")
                    actions.append(action)
            if not rows:
                if event.get("type") == "tsumo":
                    actors.append(int(event.get("actor", 0)))
                    checks.append(0)
                    expected_masks.append([0] * 46)
                    action46s.append(-1)
                    actions.append(CLEAR_RESPONSES)
                action = _action87_from_event(event)
                if action is not None:
                    actors.append(int(event.get("actor", 0)))
                    checks.append(0)
                    expected_masks.append([0] * 46)
                    action46s.append(-1)
                    actions.append(action)
        if not actions:
            continue
        failure = int(
            scan_segment(
                state,
                jnp.asarray(actors, dtype=jnp.int32),
                jnp.asarray(actions, dtype=jnp.int32),
                jnp.asarray(checks, dtype=jnp.uint8),
                jnp.asarray(expected_masks, dtype=jnp.uint8),
                jnp.asarray(action46s, dtype=jnp.int32),
            )
        )
        matched += sum(checks)
        if failure >= 0:
            first_failure = row_base + int(sum(checks[:failure]))
            break
    elapsed = time.perf_counter() - started
    return GpuReplayScanResult(
        matched_rows=matched if first_failure is None else first_failure,
        authority_rows=len(authority),
        first_failure=first_failure,
        elapsed_s=elapsed,
        rows_per_s=0.0 if elapsed <= 0.0 else matched / elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental trace-command MahJAX GPU replay validator.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--row-limit", type=int)
    args = parser.parse_args()
    result = validate_replay_gpu_scan(args.path, row_limit=args.row_limit)
    print(json.dumps(result.__dict__, sort_keys=True))
    if result.first_failure is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
