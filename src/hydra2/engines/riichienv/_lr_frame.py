"""Wall-less sim replay: framing records and digests."""

from __future__ import annotations

import json
from dataclasses import dataclass as dataclass
from pathlib import Path as Path
from typing import (
    TYPE_CHECKING as TYPE_CHECKING,
)
from typing import (
    cast as cast,
)

import riichienv
from hydra2_replay_rs import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]

from hydra2.artifacts.digest import of_canonical as of_canonical
from hydra2.contracts.action_artifact import load_action_table as load_action_table
from hydra2.contracts.common import ContractError as ContractError
from hydra2.data.decode import (
    GameRecord as GameRecord,
)
from hydra2.data.decode import (
    decode_game_object as decode_game_object,
)
from hydra2.engines.riichienv.identity import ENGINE_IDENTITY as ENGINE_IDENTITY

if TYPE_CHECKING:
    from typing import Any as Any

    from hydra2.contracts.rules_manifest import RulesManifest as RulesManifest


#: MJAI framing vocabulary: the single shared source is replay_expand's
#: closed sets (do not fork a second vocabulary here). Bound once at import
#: (module constants, UPPER by convention).
def _vocabulary() -> tuple[
    frozenset[str], frozenset[str], frozenset[str], frozenset[str], frozenset[str]
]:
    from hydra2.data import replay_expand as _re

    return (_re._START_TYPES, _re._END_TYPES, _re._SKIP_TYPES, _re._ROW_TYPES, _re._CLAIM_TYPES)


_START, _END, _SKIP, _ROW, _CLAIM = _vocabulary()


def _rules() -> RulesManifest:
    from hydra2.data import replay_expand as _re

    return _re._load_rules()


_TABLE_CACHE: dict[str, Any] = {}
_EVENT_SCHEMA_HASH_CACHE: dict[str, str] = {}
_PACKET_BOUNDARY_HASH_CACHE: dict[str, str] = {}
_RULES_HASH_CACHE: dict[str, str] = {}


def _table() -> Any:
    from hydra2.config import repo_root
    from hydra2.contracts.action_artifact import ACTION_TABLE_RELPATH

    root = str(repo_root())
    if root not in _TABLE_CACHE:
        _TABLE_CACHE[root] = load_action_table(Path(root) / ACTION_TABLE_RELPATH)
    return _TABLE_CACHE[root]


def _event_schema_hash() -> str:
    from hydra2.config import repo_root
    from hydra2.contracts.event_schema import EVENT_SCHEMA_RELPATH, parse_event_schema

    root = str(repo_root())
    if root not in _EVENT_SCHEMA_HASH_CACHE:
        document: dict[str, Any] = cast(
            "dict[str, Any]",
            parse_event_schema((Path(root) / EVENT_SCHEMA_RELPATH).read_bytes()),
        )
        payload: Any = document["payload"]
        if not isinstance(payload, dict) or "digest" not in payload:
            raise ContractError("event schema artifact lacks a digest")
        _EVENT_SCHEMA_HASH_CACHE[root] = str(cast("Any", payload["digest"]))
    return _EVENT_SCHEMA_HASH_CACHE[root]


def _packet_boundary_hash() -> str:
    from hydra2.config import repo_root
    from hydra2.contracts.event_packet import (
        build_packet_boundary_payload,
        compute_event_schema_digest,
    )

    # Perf-C P1b: the boundary payload is process-constant; hashing it per
    # builder (per kyoku) re-ran canonicalization for an identical digest.
    root = str(repo_root())
    if root not in _PACKET_BOUNDARY_HASH_CACHE:
        _PACKET_BOUNDARY_HASH_CACHE[root] = str(
            compute_event_schema_digest(build_packet_boundary_payload())
        )
    return _PACKET_BOUNDARY_HASH_CACHE[root]


def _rules_hash(manifest: RulesManifest, recomputed: str) -> str:
    """Published rules bytes win when present (same authority as the adapter)."""
    import hashlib

    from hydra2.config import repo_root

    # Perf-C P1b: published rules bytes are immutable mid-run; hash once per
    # (root, rules_id). The no-published-file branch stays uncached and
    # returns ``recomputed`` verbatim, exactly like before.
    root = str(repo_root())
    key = f"{root}\x00{manifest.rules_id}"
    if key not in _RULES_HASH_CACHE:
        published = Path(root) / "configs" / "rules" / f"{manifest.rules_id}.json"
        if published.is_file():
            _RULES_HASH_CACHE[key] = "sha256:" + hashlib.sha256(published.read_bytes()).hexdigest()
    cached = _RULES_HASH_CACHE.get(key)
    return cached if cached is not None else recomputed


def _sim_game_id(game: GameRecord, *, rules_hash: str) -> str:
    """Builder game id with the engine path's exact scheme when walled.

    A walled game replays row-identical to :func:`expand_game`, so the
    builder identity (wall-derived seed material) is reproduced exactly from
    the real wall. A wall-less game has no wall to bind -- the log's game id
    is carried instead (never a placeholder digest).
    """
    if game.wall_tiles is None:
        if game.game_id == "":
            raise ContractError("wall-less game has no game_id for sim replay")
        return game.game_id
    if len(game.wall_tiles) != 136:
        raise ContractError(f"wall_tiles must carry 136 tiles, got {len(game.wall_tiles)}")
    from hydra2.engines.protocol import wall_schedule_digest

    physical = tuple(_bridge_contracts.make_tile_id(t) for t in game.wall_tiles)
    schedule_id = f"replay-{game.game_id}"
    wall_digest = str(wall_schedule_digest(schedule_id, physical))
    seed_material = of_canonical(
        {
            "rules_hash": rules_hash,
            "wall_digest": wall_digest,
            "seat_permutation": [0, 1, 2, 3],
            "adapter_version": str(ENGINE_IDENTITY.adapter_version),
        }
    )
    return f"hydra2-riichienv-{str(seed_material).removeprefix('sha256:')[:16]}"


# ---------------------------------------------------------------------------
# Oracle records (frozen primitives extracted eagerly per yielded step).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _SimStep:
    """One yielded ``(Observation, Action)`` pair as frozen primitives."""

    seat: int
    mjai_type: str
    tile: int | None
    consume: tuple[int, ...]
    mjai: dict[str, object]
    raw_action: Any
    raw_legals: tuple[Any, ...]
    hand: tuple[int, ...]
    drawn: int | None
    dora: tuple[int, ...]
    discards: tuple[tuple[int, ...], ...]
    hands_lens: tuple[int, int, int, int]
    scores: tuple[int, int, int, int]
    sticks: int
    oya: int
    honba: int
    round_wind: int
    player_id: int
    riichi: tuple[bool, bool, bool, bool]


@dataclass(frozen=True, slots=True)
class _TablePosition:
    """Frozen throwaway-engine snapshot for oracle-less forced rows."""

    hand: tuple[int, ...]
    drawn: int | None
    dora: tuple[int, ...]
    rivers: tuple[tuple[int, ...], ...]
    lens: tuple[int, int, int, int]
    sticks: int
    declared: tuple[bool, bool, bool, bool]
    legals: tuple[Any, ...]


def _coerce_game(game: GameRecord | bytes | str | Path) -> GameRecord:
    if isinstance(game, GameRecord):
        return game
    if isinstance(game, (str, Path)):
        raw = Path(game).read_bytes()
        object_id = f"simreplay-path-{Path(game).name}"
    else:
        raw = game
        import hashlib

        object_id = "simreplay-bytes-" + hashlib.sha256(raw).hexdigest()[:16]
    return decode_game_object(object_id=object_id, packaged_object_id=object_id, decoded_bytes=raw)


def _frame_bytes(game: GameRecord) -> bytes:
    return ("\n".join(json.dumps(event) for event in game.events) + "\n").encode("utf-8")


def _drain_game_steps(staged_path: str, *, game_id: str) -> list[list[list[_SimStep]]]:
    """Drain ``steps(seat)`` for seats 0..3 of every kyoku (fail closed).

    One ``from_jsonl``/``take_kyokus`` pass per game; each ``(kyoku, seat)``
    step iterator replays independently (verified: sequential drains of one
    kyoku object yield identical per-seat queues to fresh replays).
    """

    def _fail(kyoku_index: int, why: str) -> ContractError:
        return ContractError(f"sim replay desync game {game_id!r} kyoku {kyoku_index}: {why}")

    try:
        replay = riichienv.MjaiReplay.from_jsonl(staged_path)
    except ContractError:
        raise
    except Exception as exc:
        raise _fail(-1, f"MjaiReplay.from_jsonl failed: {exc}") from exc
    try:
        kyokus = list(replay.take_kyokus())
    except ContractError:
        raise
    except Exception as exc:
        raise _fail(-1, f"take_kyokus failed: {exc}") from exc
    drained: list[list[list[_SimStep]]] = []
    for kyoku_index, kyoku in enumerate(kyokus):
        queues: list[list[_SimStep]] = [[], [], [], []]
        for seat in range(4):
            try:
                pairs = list(kyoku.steps(seat=seat))
            except ContractError:
                raise
            except Exception as exc:
                raise _fail(kyoku_index, f"seat {seat} steps failed: {exc}") from exc
            for position, (obs, act) in enumerate(pairs):
                try:
                    queues[seat].append(_extract_step(seat, obs, act, game_id=game_id))
                except ContractError:
                    raise
                except Exception as exc:
                    raise _fail(kyoku_index, f"seat {seat} step {position}: {exc}") from exc
        drained.append(queues)
    return drained


def _extract_step(seat: int, obs: Any, act: Any, *, game_id: str) -> _SimStep:
    """Freeze one yielded pair; the engine-masked firewall is asserted here."""
    where = f"game {game_id!r} seat {seat}"
    hands: Any = obs.hands
    lens = tuple(len(h) for h in hands)
    if len(lens) != 4:
        raise ContractError(f"{where}: seat-filtered hands must cover 4 seats")
    for other in range(4):
        if other != seat and lens[other] != 0:
            raise ContractError(
                f"{where}: firewall breach -- seat {other} hand leaks "
                f"{lens[other]} tiles into seat {seat}'s observation"
            )
    if lens[seat] == 0:
        raise ContractError(f"{where}: acting-seat hand is empty (desync)")
    try:
        mjai_raw: Any = act.to_mjai()
    except Exception as exc:
        raise ContractError(f"{where}: yielded action has no MJAI mapping: {exc}") from exc
    if isinstance(mjai_raw, str):
        try:
            mjai_raw = json.loads(mjai_raw)
        except ValueError as exc:
            raise ContractError(f"{where}: yielded action MJAI unparseable: {exc}") from exc
    mjai: Any = mjai_raw
    if not isinstance(mjai, dict):
        raise ContractError(f"{where}: yielded action has no MJAI mapping")
    kind = mjai.get("type")
    if not isinstance(kind, str) or kind == "":
        raise ContractError(f"{where}: yielded action without a type: {mjai!r}")
    tile_raw: Any = act.tile
    tile = None if tile_raw is None else int(tile_raw)
    consume = tuple(sorted(int(t) for t in act.consume_tiles))
    from hydra2.engines.riichienv._lr_rows import (
        _distinct_copies as _distinct_copies,
    )  # deferred: rows imports frame hashes

    hand = _distinct_copies(tuple(int(t) for t in obs.hand))
    drawn_raw: Any = obs.drawn_tile
    drawn = None if drawn_raw is None else int(drawn_raw)
    dora = tuple(int(t) for t in obs.dora_indicators)
    discards = tuple(tuple(int(t) for t in river) for river in obs.discards)
    scores_raw: Any = obs.scores
    scores = tuple(int(s) for s in scores_raw)
    if len(scores) != 4:
        raise ContractError(f"{where}: scores must cover 4 seats")
    riichi = tuple(bool(v) for v in obs.riichi_declared)
    if len(riichi) != 4:
        raise ContractError(f"{where}: riichi_declared must cover 4 seats")
    legals: Any = obs.legal_actions()
    return _SimStep(
        seat=seat,
        mjai_type=kind,
        tile=tile,
        consume=consume,
        mjai=dict(mjai),
        raw_action=act,
        raw_legals=tuple(legals),
        hand=hand,
        drawn=drawn,
        dora=dora,
        discards=discards,
        hands_lens=lens,
        scores=scores,
        sticks=int(obs.riichi_sticks),
        oya=int(obs.oya),
        honba=int(obs.honba),
        round_wind=int(obs.round_wind),
        player_id=int(obs.player_id),
        riichi=riichi,
    )
