# Copyright 2025 The Mahjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Saving games as mjai logs, and replaying them.

A record stores no board states -- only the seed and the events. Replaying runs
the env again from that seed, which is why the log has to be checked against the
env as it goes: a different JAX build can deal a different wall from the same
key, and a silently different game would be worse than a refused one.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from . import mjai, view
from .rules import ENV_IDS, Rules, rules_for, shared_env, shared_step_fn
from .view import SeatInfo

RECORD_SUFFIX = ".mjson"


class RecordError(RuntimeError):
    """A record could not be read or replayed."""


def records_dir() -> Path:
    path = Path(os.environ.get("MAHJAX_RECORD_DIR", "records"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _meta_of(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not events or events[0].get("type") != "start_game":
        raise RecordError("Record does not begin with a start_game event")
    meta = events[0].get("mahjax")
    if not isinstance(meta, dict):
        raise RecordError("Record is missing its mahjax metadata")
    return meta


def save_events(
    events: Sequence[Dict[str, Any]], *, record_id: Optional[str] = None
) -> str:
    meta = _meta_of(events)
    if record_id is None:
        created = meta.get("created_at")
        try:
            stamp = _dt.datetime.fromisoformat(str(created)).strftime("%Y%m%d-%H%M%S")
        except (TypeError, ValueError):
            stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        record_id = f"{stamp}-{uuid.uuid4().hex[:8]}"
    path = records_dir() / f"{record_id}{RECORD_SUFFIX}"
    with open(path, "w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return record_id


def save_match(match: Any) -> str:
    """``on_game_over`` hook for :class:`~mahjax.ui.match.Match`."""
    return save_events(match.events)


def read_events(record_id: str) -> List[Dict[str, Any]]:
    path = records_dir() / f"{record_id}{RECORD_SUFFIX}"
    if not path.exists():
        raise RecordError(f"No record {record_id}")
    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as err:
                raise RecordError(f"{record_id}: bad JSON on line {lineno}") from err
    return events


def _summary(record_id: str, first_line: Dict[str, Any], mtime: float) -> Dict[str, Any]:
    meta = first_line.get("mahjax") or {}
    names = first_line.get("names") or []
    players = meta.get("players") or []
    env_id = meta.get("env_id")
    seats = rules_for(env_id).num_players if env_id in ENV_IDS else 4
    return {
        "id": record_id,
        "createdAt": meta.get("created_at")
        or _dt.datetime.fromtimestamp(mtime).astimezone().isoformat(timespec="seconds"),
        "env": env_id,
        "roundMode": meta.get("round_mode"),
        "humanSeat": meta.get("human_seat"),
        "complete": bool(meta.get("complete")),
        "players": [
            {
                "seat": i,
                "name": names[i] if i < len(names) else f"Player {i + 1}",
                "kind": (players[i] or {}).get("kind", "agent") if i < len(players) else "agent",
            }
            for i in range(seats)
        ],
    }


def list_records() -> List[Dict[str, Any]]:
    """Newest first. Only the first line of each file is read."""
    out: List[Dict[str, Any]] = []
    for path in records_dir().glob(f"*{RECORD_SUFFIX}"):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                first = json.loads(handle.readline())
        except (OSError, json.JSONDecodeError):
            continue
        if first.get("type") != "start_game":
            continue
        out.append(_summary(path.stem, first, path.stat().st_mtime))
    out.sort(key=lambda r: r["createdAt"], reverse=True)
    return out


def delete_record(record_id: str) -> None:
    path = records_dir() / f"{record_id}{RECORD_SUFFIX}"
    if path.exists():
        path.unlink()


@dataclass
class RoundSpan:
    index: int
    label: str
    start: int
    end: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "start": self.start,
            "end": self.end,
        }


_WIND_JA = ("東", "南", "西", "北")


def round_label(state: Any) -> str:
    rs = state.round_state
    index = int(rs.round)
    honba = int(rs.honba)
    seats = int(state.constants.NUM_PLAYERS)  # a wind lasts one round per seat
    label = f"{_WIND_JA[min(index // seats, 3)]}{index % seats + 1}局"
    if honba:
        label += f" {honba}本場"
    return label


class Replay:
    """A finished game, re-run from its seed and held as a list of frames.

    A frame is either the opening board of a round or the board after one env
    action. The four DUMMY actions that share the result between rounds are not
    frames: they change no board a viewer would recognise.
    """

    def __init__(self, events: Sequence[Dict[str, Any]], *, verify: bool = True) -> None:
        self.id = uuid.uuid4().hex
        self.events = list(events)
        self.meta = _meta_of(self.events)
        env_id = self.meta.get("env_id")
        if env_id not in ENV_IDS:
            raise RecordError(f"Record has an unknown env id: {env_id!r}")
        self.rules: Rules = rules_for(env_id)
        self.round_mode = self.meta.get("round_mode", "half")
        self.env = shared_env(env_id, self.round_mode)
        self.seats = self._seats()
        self.human_seat = self.meta.get("human_seat")

        self._states: List[Any] = []
        self._steps: List[Dict[str, Any]] = []
        self._results: Dict[int, Dict[str, Any]] = {}
        self.rounds: List[RoundSpan] = []
        try:
            self._replay(verify=verify)
        except (RecordError, mjai.MjaiError) as err:
            import mahjax

            # The replay re-runs the env from the seed, so a log from another
            # version fails wherever the rules changed. Say that, not where.
            recorded = self.meta.get("mahjax_version")
            if recorded and recorded != mahjax.__version__:
                raise RecordError(
                    f"This record was written by mahjax {recorded}, and mahjax "
                    f"{mahjax.__version__} plays the game differently, so it cannot be replayed."
                ) from err
            if isinstance(err, RecordError):
                raise
            raise RecordError(str(err)) from err

    def _seats(self) -> List[SeatInfo]:
        names = self.events[0].get("names") or []
        players = self.meta.get("players") or []
        out: List[SeatInfo] = []
        for i in range(self.rules.num_players):
            kind = (players[i] or {}).get("kind", "agent") if i < len(players) else "agent"
            name = names[i] if i < len(names) else f"Player {i + 1}"
            out.append(SeatInfo(name=name, kind=kind))
        return out

    # ------------------------------------------------------------- replaying

    def _replay(self, *, verify: bool) -> None:
        rules = self.rules
        actions = mjai.decode_events(rules, self.events)
        root = jax.random.PRNGKey(int(self.meta["seed"]))
        step_fn = shared_step_fn(self.rules.env_id, self.round_mode)
        state = jax.device_get(self.env.init(jax.random.fold_in(root, 0)))

        logged = [e for e in self.events if e.get("type") in ("start_kyoku", "tsumo")]
        checks = iter(logged) if verify else iter(())

        def check(produced: Iterable[Dict[str, Any]]) -> None:
            for event in produced:
                if event.get("type") not in ("start_kyoku", "tsumo"):
                    continue
                expected = next(checks, None)
                if expected is None:
                    return
                if not _same_deal(expected, event):
                    raise RecordError(
                        "Replay diverged from the record: expected "
                        f"{json.dumps(expected, ensure_ascii=False)} but the env produced "
                        f"{json.dumps(event, ensure_ascii=False)}. The record was written "
                        "with a different JAX build or mahjax version."
                    )

        check([mjai.start_kyoku_event(rules, state)])
        # A round's result cannot be labelled until the DUMMY steps after it have
        # run, because that is where the env decides the game is over. So the
        # result is held here and attached once those steps are behind us.
        pending: Optional[Any] = None
        round_index = 0
        round_start_frame = 0
        score_start = rules.scores(state)
        wins: List[Dict[str, Any]] = []
        self._push(state, None, round_index)

        def finalize(current: Any) -> bool:
            """Attach the held result. Returns True when nothing follows it: the
            game is over, or the log stops on this result."""
            nonlocal pending, round_index, round_start_frame, score_start, wins
            if pending is None:
                return False
            frame_idx, end_state, held_wins, held_scores, abortive = pending
            pending = None
            game_over = bool(current.terminated)
            self._results[frame_idx] = self._round_result(
                end_state, held_wins, held_scores, game_over, abortive
            )
            self.rounds.append(
                RoundSpan(
                    index=round_index,
                    label=round_label(end_state),
                    start=round_start_frame,
                    end=frame_idx,
                )
            )
            if game_over:
                return True
            if rules.is_round_over(current):
                # Saved while this result was on screen: the next round was never dealt.
                return True
            round_index += 1
            wins = []
            score_start = rules.scores(current)
            round_start_frame = len(self._states)
            self._push(current, None, round_index)
            # The log has a start_kyoku here too; checking it keeps the two
            # streams aligned and catches a wall that was dealt differently.
            check([mjai.start_kyoku_event(rules, current)])
            return False

        for i, action in enumerate(actions):
            key = jax.random.fold_in(root, i + 1)
            if action == rules.DUMMY:
                state = jax.device_get(step_fn(state, jnp.int32(action), key))
                continue
            if pending is not None and finalize(state):
                break
            pre = state
            state = jax.device_get(step_fn(pre, jnp.int32(action), key))
            win = None
            if action in (rules.RON, rules.TSUMO):
                win = view.build_win(
                    rules, pre, state, int(pre.current_player), action == rules.RON
                )
                wins.append(win)
            check(mjai.encode_step(rules, pre, int(action), state, win=win))

            # A nine-terminals abortion deals the next round inside the same
            # step, so the board worth showing is the one from before it.
            is_abortive = action == rules.KYUUSHU
            # Why it was abortive has to be read before the step is applied,
            # exactly as the live game reads it, or a replay would label every
            # abortive draw with the same generic title.
            abortive = view.abortive_cause(rules, pre) if is_abortive else None
            self._push(pre if is_abortive else state, (pre, int(action)), round_index)
            if is_abortive or rules.is_round_over(state):
                pending = (
                    len(self._states) - 1,
                    pre if is_abortive else state,
                    list(wins),
                    list(score_start),
                    abortive,
                )
        finalize(state)

        # Frames after the last finished round -- a round still in play when the
        # record was saved, or the whole game when none finished -- need a span of
        # their own, or the replay bar files them under the first round.
        last_end = self.rounds[-1].end if self.rounds else -1
        if last_end < len(self._states) - 1:
            self.rounds.append(
                RoundSpan(
                    index=len(self.rounds),
                    label=round_label(self._states[-1]),
                    start=last_end + 1,
                    end=len(self._states) - 1,
                )
            )

    def _push(
        self, state: Any, produced_by: Optional[Any], round_index: int
    ) -> None:
        self._states.append(state)
        if produced_by is None:
            self._steps.append({"roundIndex": round_index, "event": None})
        else:
            pre, action = produced_by
            self._steps.append(
                {
                    "roundIndex": round_index,
                    "event": view.describe_action(self.rules, pre, action),
                }
            )

    def _round_result(
        self,
        end_state: Any,
        wins: List[Dict[str, Any]],
        score_start: Sequence[int],
        game_over: bool,
        abortive: Optional[Tuple[str, Optional[int]]],
    ) -> Dict[str, Any]:
        abort_seat = None
        if abortive is not None:
            result_type, (reason, abort_seat) = "abortive", abortive
            # A triple ron is three declared wins that the round then voids, so
            # the log carries three. The live board lists none for an abortive
            # draw and the replay has to say the same thing.
            wins = []
        elif wins:
            result_type = "ron" if any(w["from"] is not None for w in wins) else "tsumo"
            reason = None
        else:
            result_type, reason = "draw", None
        return view.build_round_result(
            self.rules,
            type=result_type,
            reason=reason,
            state=end_state,
            winners=wins,
            score_start=list(score_start),
            game_over=game_over,
            abort_seat=abort_seat,
        )

    # ----------------------------------------------------------------- output

    @property
    def total(self) -> int:
        return len(self._states)

    def frame(self, index: int, *, viewpoint: Optional[int], show_all: bool) -> Dict[str, Any]:
        if not 0 <= index < len(self._states):
            raise IndexError(index)
        seats = self.rules.num_players
        if show_all:
            reveal = [True] * seats
        else:
            seat = 0 if viewpoint is None else int(viewpoint)
            reveal = [i == seat for i in range(seats)]
            if index in self._results:
                reveal = [True] * seats
        meta = dict(self._steps[index])
        step = {
            "index": index,
            "total": len(self._states),
            "roundIndex": meta["roundIndex"],
            "event": meta["event"],
        }
        return view.build_view(
            self.rules,
            self._states[index],
            self.seats,
            reveal=reveal,
            result=self._results.get(index),
            step=step,
        )

    def frames(
        self,
        start: int,
        stop: int,
        *,
        viewpoint: Optional[int] = None,
        show_all: bool = True,
    ) -> List[Dict[str, Any]]:
        start = max(0, start)
        stop = min(len(self._states), stop)
        return [
            self.frame(i, viewpoint=viewpoint, show_all=show_all)
            for i in range(start, stop)
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "replayId": self.id,
            "env": self.rules.env_id,
            "roundMode": self.meta.get("round_mode"),
            "humanSeat": self.human_seat,
            "seats": [{"name": s.name, "kind": s.kind} for s in self.seats],
            "rounds": [r.to_dict() for r in self.rounds],
            "total": self.total,
        }


def unstack_states(states: Any) -> List[Any]:
    """One state, a list of states, or a state with a leading axis -> unbatched states on the host."""
    if isinstance(states, (list, tuple)):
        out = [jax.device_get(state) for state in states]
    else:
        host = jax.device_get(states)
        if np.ndim(host.current_player) == 0:
            out = [host]
        else:
            n = int(np.shape(host.current_player)[0])
            out = [jax.tree_util.tree_map(lambda x, i=i: x[i], host) for i in range(n)]
    if not out:
        raise ValueError("No states to show")
    for state in out:
        if np.ndim(state.current_player) != 0:
            raise ValueError("Pass unbatched states: take one env out of a batch first")
    return out


def env_id_of(state: Any) -> str:
    """The UI rules for a state. Its preset is not stored, so a wall with red fives means a red one."""
    has_red = bool((np.asarray(state.round_state.deck) >= 34).any())
    return f"{'red' if has_red else 'no_red'}_{int(state.constants.NUM_PLAYERS)}p"


def _recorded_action(pre: Any, post: Any) -> Optional[int]:
    """The action that turned ``pre`` into ``post``, when ``post`` is one recorded action later."""
    before, after = pre.round_state, post.round_state
    if int(before.round) != int(after.round) or int(before.honba) != int(after.honba):
        return None
    n = int(after.round_step)
    if n != int(before.round_step) + 1:
        return None
    history = np.asarray(after.action_history)
    if int(history[0, n - 1]) != int(pre.current_player):
        return None
    return int(history[1, n - 1])


class StateReplay(Replay):
    """States handed in from Python, shown frame by frame like a finished game.

    Every state is a frame. Where two neighbouring states are one recorded action
    apart, the frame names that action, and a round that ends in view gets its
    result, as in a replayed record. States from an ``round_end="auto"`` env skip
    the end of each round, so their rounds just follow one another.
    """

    def __init__(self, states: Any) -> None:
        # The serving half of ``Replay`` (frames, spans, results) is shared; only
        # the frames come from the caller instead of a re-run.
        self.id = uuid.uuid4().hex
        self.events = []
        states = unstack_states(states)
        env_id = env_id_of(states[0])
        self.meta = {"env_id": env_id, "round_mode": None}
        self.rules = rules_for(env_id)
        self.round_mode = None
        self.env = None
        self.seats = [SeatInfo(name=f"Player {i}", kind="agent") for i in range(self.rules.num_players)]
        self.human_seat = None
        self._states = []
        self._steps = []
        self._results = {}
        self.rounds = []
        self._collect(states)

    def _collect(self, states: List[Any]) -> None:
        rules = self.rules
        round_index, round_start = 0, 0
        wins: List[Dict[str, Any]] = []
        score_start = rules.scores(states[0])
        self._push(states[0], None, round_index)
        for i in range(1, len(states)):
            pre, state = states[i - 1], states[i]
            over_before, over_now = rules.is_round_over(pre), rules.is_round_over(state)
            new_deal = (int(pre.round_state.round), int(pre.round_state.honba)) != (
                int(state.round_state.round),
                int(state.round_state.honba),
            ) or (over_before and not over_now)
            if not new_deal and over_before and over_now:
                continue  # the DUMMY steps that share a result change no board
            if new_deal:
                self._close(round_index, round_start, len(self._states) - 1, pre)
                round_index, round_start, wins = round_index + 1, len(self._states), []
                score_start = rules.scores(state)
            action = _recorded_action(pre, state)
            self._push(state, None if action is None else (pre, action), round_index)
            if action in (rules.RON, rules.TSUMO):
                wins.append(view.build_win(rules, pre, state, int(pre.current_player), action == rules.RON))
            if over_now and not over_before:
                game_over = _ends_the_game(rules, states[i:])
                frame = len(self._states) - 1
                self._results[frame] = self._round_result(state, wins, score_start, game_over, None)
                self._close(round_index, round_start, frame, state)
        self._close(round_index, round_start, len(self._states) - 1, states[-1])

    def _close(self, round_index: int, start: int, end: int, state: Any) -> None:
        if self.rounds and self.rounds[-1].index == round_index:
            return
        self.rounds.append(RoundSpan(index=round_index, label=round_label(state), start=start, end=end))


def _ends_the_game(rules: Rules, states: Sequence[Any]) -> bool:
    """Whether the round that ends at ``states[0]`` ends the game: the env flips
    ``terminated`` in the sharing steps after it, before any next deal."""
    for state in states:
        if bool(state.terminated):
            return True
        if not rules.is_round_over(state):
            return False  # the next round was dealt
    return False


def _same_deal(expected: Dict[str, Any], produced: Dict[str, Any]) -> bool:
    if expected.get("type") != produced.get("type"):
        return False
    if expected["type"] == "tsumo":
        return expected.get("pai") == produced.get("pai") and expected.get(
            "actor"
        ) == produced.get("actor")
    return (
        expected.get("tehais") == produced.get("tehais")
        and expected.get("dora_marker") == produced.get("dora_marker")
        and expected.get("oya") == produced.get("oya")
    )






def load_replay(record_id: str, *, verify: bool = True) -> Replay:
    return Replay(read_events(record_id), verify=verify)


__all__ = [
    "Replay",
    "RecordError",
    "load_replay",
    "list_records",
    "read_events",
    "save_events",
    "save_match",
    "delete_record",
    "records_dir",
    "round_label",
]
