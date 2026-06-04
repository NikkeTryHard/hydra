from __future__ import annotations

import importlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

UInt8Array = NDArray[np.uint8]

from hydra_learner.mahjax import contract
from hydra_learner.mahjax.compat import HYDRA_PASS
from hydra_learner.mahjax.constructor import mahjax_state_from_start_kyoku, mjai_tile_to_mahjax_id

DATASET_ROOT = Path("/home/cachybtw/Downloads/dataset_bundle/tenhou-houou-mjai-2025")
_REPO_ROOT = Path(__file__).resolve().parents[5]


def _authority_trace_command(path: Path) -> list[str]:
    binary = _REPO_ROOT / "target" / "debug" / "mjai_authority_fixture"
    if binary.exists():
        return [str(binary), "--strict", "--trace", str(path)]
    return [
        "pixi",
        "run",
        "cargo",
        "run",
        "-p",
        "hydra-train",
        "--bin",
        "mjai_authority_fixture",
        "--features",
        "training",
        "--quiet",
        "--",
        "--strict",
        "--trace",
        str(path),
    ]


def _authority_trace_batch_command(list_path: Path) -> list[str]:
    binary = _REPO_ROOT / "target" / "debug" / "mjai_authority_fixture"
    if binary.exists():
        return [str(binary), "--strict", "--trace", "--batch", str(list_path)]
    return [
        "pixi",
        "run",
        "cargo",
        "run",
        "-p",
        "hydra-train",
        "--bin",
        "mjai_authority_fixture",
        "--features",
        "training",
        "--quiet",
        "--",
        "--strict",
        "--trace",
        "--batch",
        str(list_path),
    ]


@dataclass(frozen=True)
class ReplayParityResult:
    path: Path
    matched_rows: int
    stopped_reason: str
    authority_rows: int = 0
    first_failure: dict[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return self.stopped_reason == "authority_exhausted" and self.matched_rows == self.authority_rows


def _load_replay_events(path: Path, event_limit: int | None) -> list[dict[str, Any]]:
    proc = subprocess.Popen(
        ["zstd", "-dc", str(path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    if proc.stdout is None:
        raise RuntimeError("zstd stdout pipe unavailable")
    events: list[dict[str, Any]] = []
    try:
        for line in proc.stdout:
            if line:
                events.append(json.loads(line))
            if event_limit is not None and len(events) >= event_limit:
                break
    finally:
        proc.kill()
        proc.wait()
    return events


def _hydra_authority_rows(path: Path, row_limit: int | None) -> list[dict[str, Any]]:
    raw = subprocess.check_output(_authority_trace_command(path), text=True)
    rows = json.loads(raw)["rows"]
    return rows if row_limit is None else rows[:row_limit]


def hydra_authority_rows_for_paths(paths: list[Path], row_limit: int | None) -> dict[Path, list[dict[str, Any]]]:
    with tempfile.NamedTemporaryFile(
        "w", dir=Path.home() / "tmp", prefix="mahjax-trace-", suffix=".txt", delete=False
    ) as handle:
        list_path = Path(handle.name)
        handle.write("".join(f"{path}\n" for path in paths))
    try:
        raw = subprocess.check_output(_authority_trace_batch_command(list_path), text=True)
    finally:
        list_path.unlink(missing_ok=True)
    grouped: dict[Path, list[dict[str, Any]]] = {path: [] for path in paths}
    for row in json.loads(raw)["rows"]:
        replay_path = row.get("replay_path")
        if replay_path is None:
            raise ValueError("batch trace row missing replay_path")
        path = Path(replay_path)
        rows = grouped.get(path)
        if rows is not None and (row_limit is None or len(rows) < row_limit):
            rows.append(row)
    return grouped


def _projected_mask(state: Any) -> UInt8Array:
    return contract.projected_hydra_mask(state)


def _projected_player_response_mask(state: Any, player: int) -> UInt8Array:
    return contract.projected_response_hydra_mask(state, player)


def _trace_row_mask(state: Any, row: dict[str, Any]) -> UInt8Array:
    return contract.hydra_mask_for_actor(state, int(row["actor"]))


def _assert_trace_row(state: Any, row: dict[str, Any], matched: int) -> None:
    if row["phase"] not in {"normal", "riichi_select", "kan_select"}:
        raise ValueError(f"unsupported trace phase: {row['phase']}")
    mask = _trace_row_mask(state, row)
    np.testing.assert_array_equal(mask, np.asarray(row["legal_mask"], dtype=np.uint8), err_msg=f"trace row {matched}")
    action_id = int(row["action_id"])
    if action_id < 0 or action_id >= mask.shape[0] or not bool(mask[action_id]):
        raise ValueError(f"trace action {action_id} is not legal at row {matched}")


def _failure_detail(
    path: Path,
    authority: list[dict[str, Any]],
    matched: int,
    stopped_reason: str,
    event_index: int | None,
) -> ReplayParityResult:
    row = authority[matched] if matched < len(authority) else None
    detail: dict[str, Any] = {"reason": stopped_reason}
    if event_index is not None:
        detail["event_index"] = event_index
    if row is not None:
        detail.update(
            {
                "authority_row_index": matched,
                "authority_event_index": row.get("event_index"),
                "actor": row.get("actor"),
                "phase": row.get("phase"),
                "kind": row.get("kind"),
                "source_event_type": row.get("source_event_type"),
                "authority_action_id": row.get("action_id"),
                "authority_legal_mask": row.get("legal_mask"),
            }
        )
    return ReplayParityResult(
        path=path,
        matched_rows=matched,
        stopped_reason=stopped_reason,
        authority_rows=len(authority),
        first_failure=detail,
    )


def _state_for_trace_row(events: list[dict[str, Any]], search_start: int) -> tuple[Any, int, int]:
    state, cursor, end = _start_state_from_events(events, search_start)
    kyoku_events = events[cursor:end]
    live_draws, kan_draws = _draw_tiles_for_kyoku(kyoku_events)
    return _patch_draw_deck(state, live_draws, kan_draws), cursor, end


def _event_type_matches_trace(event_type: object, source_event_type: object) -> bool:
    return event_type == source_event_type or (event_type == "daiminkan" and source_event_type == "kan")


def _apply_untraced_event(step_fn: Any, state: Any, event: dict[str, Any]) -> Any:
    event_type = event.get("type")
    if event_type in {"dora", "reach_accepted", "end_kyoku", "start_kyoku"}:
        return state
    if contract.response_phase(state):
        state = contract.apply_all_response_passes(step_fn, state)
    return contract.apply_mjai_event_action(step_fn, state, event)


def _advance_to_trace_actor(step_fn: Any, state: Any, actor: int) -> Any:
    return contract.advance_response_to_actor(step_fn, state, actor)


def _apply_traced_event_rows(
    step_fn: Any,
    state: Any,
    event: dict[str, Any],
    rows: list[dict[str, Any]],
    matched: int,
) -> Any:
    event_type = event.get("type")
    sampled = False
    for offset, row in enumerate(rows):
        if not _event_type_matches_trace(event_type, row["source_event_type"]):
            raise ValueError(
                "trace event type "
                f"{row['source_event_type']} does not match replay event {event_type} at row {matched + offset}"
            )
        if row["kind"] == "implicit_pass":
            _assert_trace_row(state, row, matched + offset)
            if int(row["action_id"]) != HYDRA_PASS:
                raise ValueError(f"implicit pass trace row has action {row['action_id']}")
            continue
        if row["kind"] != "sampled_event":
            raise ValueError(f"unsupported trace row kind: {row['kind']}")
        state = _advance_to_trace_actor(step_fn, state, int(row["actor"]))
        _assert_trace_row(state, row, matched + offset)
        projected = contract.projected_hydra_action_for_mjai_event(state, event)
        if projected is None:
            raise ValueError(f"trace sampled row has no MahJAX action for event {event_type}")
        if projected != int(row["action_id"]):
            raise ValueError(f"trace action {row['action_id']} does not match replay event projection {projected}")
        state = contract.apply_mjai_event_action(step_fn, state, event)
        sampled = True
    if not sampled and event_type == "hora" and "actor" in event:
        state = _advance_to_trace_actor(step_fn, state, int(event["actor"]))
        return contract.apply_mjai_event_action(step_fn, state, event)
    if not sampled:
        state = _apply_untraced_event(step_fn, state, event)
    return state


def _apply_action(step_fn: Any, state: Any, action: int) -> Any:
    return contract.apply_mahjax_action(step_fn, state, action)


def _apply_mjai_event_fast(step_fn: Any, state: Any, event: dict[str, Any]) -> Any:
    return contract.apply_mjai_event_action(step_fn, state, event)


def _apply_implicit_passes_fast(step_fn: Any, state: Any) -> Any:
    return contract.apply_all_response_passes(step_fn, state)


def _patch_draw_deck(state: Any, live_draws: list[str], kan_draws: list[str]) -> Any:
    deck = state.round_state.deck
    for index, tile in zip(range(82, 82 - len(live_draws), -1), live_draws, strict=True):
        deck = deck.at[index].set(mjai_tile_to_mahjax_id(tile))
    for index, tile in zip(range(10, 10 + len(kan_draws)), kan_draws, strict=True):
        deck = deck.at[index].set(mjai_tile_to_mahjax_id(tile))
    return state.replace(round_state=state.round_state.replace(deck=deck))


def _draw_tiles_for_kyoku(events: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    live_draws: list[str] = []
    kan_draws: list[str] = []
    previous_action: object = None
    for event in events:
        event_type = event.get("type")
        if event_type == "tsumo":
            if previous_action in {"kan", "daiminkan", "ankan", "kakan"}:
                kan_draws.append(str(event["pai"]))
            else:
                live_draws.append(str(event["pai"]))
            continue
        if event_type != "dora":
            previous_action = event_type
    return live_draws, kan_draws


def _first_action_after_initial_draw(events: list[dict[str, Any]], start: int) -> object:
    for event in events[start:]:
        event_type = event.get("type")
        if event_type not in {"tsumo", "reach_accepted", "end_kyoku"}:
            return event_type
    return None


def _start_state_from_events(events: list[dict[str, Any]], start: int = 0) -> tuple[Any, int, int]:
    for index in range(start, len(events)):
        event = events[index]
        if event.get("type") != "start_kyoku":
            continue
        next_event = events[index + 1]
        if next_event.get("type") != "tsumo":
            raise ValueError("start_kyoku must be followed by first tsumo")
        end = _next_kyoku_boundary(events, index + 2)
        first_action = _first_action_after_initial_draw(events[index:end], 2)
        if first_action in {"ryukyoku", "start_kyoku", None}:
            continue
        state = mahjax_state_from_start_kyoku(
            tehais=event["tehais"],
            scores=event["scores"],
            dora_marker=event["dora_marker"],
            oya=int(event["oya"]),
            kyoku=int(event["kyoku"]),
            honba=int(event["honba"]),
            kyotaku=int(event["kyotaku"]),
            first_draw=str(next_event["pai"]),
        )
        return state, index + 2, end
    raise ValueError("replay prefix lacks start_kyoku")


def _next_kyoku_boundary(events: list[dict[str, Any]], start: int) -> int:
    for index in range(start, len(events)):
        if events[index].get("type") in {"end_kyoku", "start_kyoku"}:
            return index
    return len(events)


def compare_replay_prefix_to_hydra_authority(
    path: Path,
    *,
    row_limit: int | None = 32,
    event_limit: int | None = 96,
    authority_rows: list[dict[str, Any]] | None = None,
) -> ReplayParityResult:
    events = _load_replay_events(path, event_limit)
    authority = authority_rows if authority_rows is not None else _hydra_authority_rows(path, row_limit)
    if row_limit is not None:
        authority = authority[:row_limit]
    mahjax = importlib.import_module("mahjax")
    env = mahjax.make("red_mahjong", observe_type="dict")
    jax = importlib.import_module("jax")
    step_fn = jax.jit(env.step)

    matched = 0
    search_start = 0
    state: Any | None = None
    cursor = 0
    end = 0
    current_event_index = 0

    while matched < len(authority):
        row_event_index = int(authority[matched]["event_index"])
        if row_event_index >= len(events):
            return _failure_detail(path, authority, matched, "event_limit", row_event_index)

        while state is None or row_event_index >= end:
            try:
                state, cursor, end = _state_for_trace_row(events, search_start)
            except ValueError:
                return _failure_detail(path, authority, matched, "event_limit", row_event_index)
            current_event_index = cursor
            search_start = end + 1
            if row_event_index < cursor:
                return _failure_detail(path, authority, matched, "trace_before_selected_kyoku", row_event_index)

        if row_event_index < current_event_index:
            return _failure_detail(path, authority, matched, "trace_event_regressed", row_event_index)

        try:
            while current_event_index < row_event_index:
                state = _apply_untraced_event(step_fn, state, events[current_event_index])
                current_event_index += 1

            rows_for_event: list[dict[str, Any]] = []
            while matched + len(rows_for_event) < len(authority):
                row = authority[matched + len(rows_for_event)]
                if int(row["event_index"]) != row_event_index:
                    break
                rows_for_event.append(row)
            state = _apply_traced_event_rows(step_fn, state, events[row_event_index], rows_for_event, matched)
            matched += len(rows_for_event)
            current_event_index = row_event_index + 1
        except (AssertionError, ValueError) as exc:
            return _failure_detail(path, authority, matched, str(exc), row_event_index)

    stopped_reason = "row_limit" if row_limit is not None and matched >= row_limit else "authority_exhausted"
    return ReplayParityResult(
        path=path,
        matched_rows=matched,
        stopped_reason=stopped_reason,
        authority_rows=len(authority),
    )
