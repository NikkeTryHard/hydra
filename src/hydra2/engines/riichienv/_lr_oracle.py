"""Wall-less sim replay: window/furiten oracle."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING as TYPE_CHECKING
from typing import cast as cast

import riichienv

from hydra2.contracts.common import ContractError as ContractError
from hydra2.engines.riichienv._lr_frame import _TablePosition as _TablePosition
from hydra2.engines.riichienv._oracle_base import _TRANSPARENT_KINDS as _TRANSPARENT_KINDS
from hydra2.engines.riichienv.tiles import mjai_string_of as mjai_string_of
from hydra2.engines.riichienv.tiles import physical_of as physical_of

if TYPE_CHECKING:
    from collections.abc import Sequence as Sequence
    from typing import Any as Any

    from hydra2.contracts.observation import VisibleMeld as VisibleMeld
    from hydra2.engines.riichienv._lr_frame import _SimStep as _SimStep


# ---------------------------------------------------------------------------
# Raw-engine window/furiten oracle.
# ---------------------------------------------------------------------------

#: Log event kinds skipped when looking behind/ahead through a kyoku.


def _peek_window_claim(
    events: Sequence[dict[str, object]], start: int, discarder: int
) -> dict[str, object] | None:
    """Next window claim on ``discarder``'s tile, if the log shows one.

    Scans forward past transparent events; a draw decision, another offer, a
    terminal, or any boundary ends the window with no claim.
    """
    idx = start
    total = len(events)
    while idx < total:
        event = events[idx]
        if not isinstance(event, dict):
            raise ContractError(f"mjai event [{idx}] must be an object")
        kind = str(event.get("type", ""))
        if kind in _TRANSPARENT_KINDS:
            idx += 1
            continue
        if kind in ("chi", "pon", "daiminkan"):
            target = event.get("target")
            if target == discarder:
                return event
            return None
        if kind == "hora" and not bool(event.get("tsumo", False)):
            # Tenhou-real tsumo wins carry no flag; target==actor marks them
            # and they never open a discard window.
            if event.get("target") == event.get("actor"):
                return None
            if event.get("target") == discarder:
                return event
            return None
        return None
    return None


class _WindowOracle:
    """Throwaway raw-engine oracle answering windows and furiten exactly.

    One pinned ``riichienv.RiichiEnv`` per kyoku, reset with an ephemeral
    countdown wall built from the log itself (tehais in deal order, logged
    tsumo in draw order with rinshan tiles at the dead-wall tail, filler
    elsewhere). The wall is purely mechanical and its digest is never bound
    anywhere. The log's events are translated to raw steps with string-level
    matching; any mismatch fails closed (quarantine upstream), never
    approximated.

    Answered queries (the ONLY surfaces consumed):

    - ``window_open()``: the engine's own claim-window predicate (phase plus
      responder legals) after a discard or kakan step;
    - ``furiten(pid)``: :func:`furiten_of` on the live engine at row time;
    - ``check_row(...)``: string-level hand/river/meld agreement between the
      seat-filtered oracle observation and the throwaway state.

    Wins, draws, dora reveals, and reach acceptances are never translated:
    the kyoku is decided by then and no further query can occur before the
    next reset.
    """

    def __init__(self, *, game_id: str) -> None:
        self._game_id = game_id
        self._env: Any = None
        self._kyoku = -1
        self._missed: set[int] = set()

    def _fail(self, why: str) -> ContractError:
        return ContractError(f"sim replay desync game {self._game_id!r} kyoku {self._kyoku}: {why}")

    def reset_kyoku(
        self,
        *,
        ordinal: int,
        oya: int,
        scores: tuple[int, int, int, int],
        honba: int,
        kyotaku: int,
        bakaze: str,
        tehais: tuple[Any, ...],
        live_draws: list[str],
        rinshan_draws: list[str],
    ) -> None:
        self._kyoku = ordinal
        self._missed = set()
        if len(tehais) != 4:
            raise self._fail(f"tehais cover {len(tehais)} seats, not 4")
        # Distinct physical copies per occurrence (the engine's multiset
        # accounting assumes unique tiles; collapsed ids corrupt it). Tile
        # conservation bounds every string to its four copies -- exhausting a
        # pool means an impossible log and fails closed.
        pools: dict[str, list[int]] = {}
        taken: dict[str, int] = {}

        def take(pai: str) -> int:
            if pai not in pools:
                first = int(physical_of(pai))
                if pai in ("5mr", "0m"):
                    pools[pai] = [16]
                elif pai in ("5pr", "0p"):
                    pools[pai] = [52]
                elif pai in ("5sr", "0s"):
                    pools[pai] = [88]
                else:
                    base = (first // 4) * 4
                    ids = [base, base + 1, base + 2, base + 3]
                    if len(pai) == 2 and pai[0] == "5":
                        ids = [i for i in ids if i != base]
                    pools[pai] = ids
            cursor = taken.get(pai, 0)
            if cursor >= len(pools[pai]):
                raise self._fail(f"tile string {pai!r} overused (tile conservation)")
            taken[pai] = cursor + 1
            return pools[pai][cursor]

        wall: list[int] = [-1] * 136
        # Deal order rotates with the dealer: block b feeds seat
        # (oya + b) mod 4, so seat s takes blocks congruent to (s - oya).
        for seat in range(4):
            hand = list(tehais[seat])
            if len(hand) != 13:
                raise self._fail(f"seat {seat} tehais hold {len(hand)} tiles, not 13")
            rel = (seat - oya) % 4
            for j in range(12):
                k, m = divmod(j, 4)
                wall[4 * (rel + 4 * k) + m] = take(str(hand[j]))
            wall[48 + rel] = take(str(hand[12]))
        for i, pai in enumerate(live_draws):
            wall[52 + i] = take(pai)
        for i, pai in enumerate(rinshan_draws):
            wall[135 - i] = take(pai)
        used = {t for t in wall if t != -1}
        filler = (t for t in range(136) if t not in used)
        wall = [t if t != -1 else next(filler) for t in wall]
        wind = {"E": 0, "S": 1, "W": 2, "N": 3}[bakaze]
        env = riichienv.RiichiEnv(
            game_mode=riichienv.GameType.YON_HANCHAN,
            rule=riichienv.GameRule.default_tenhou(),
            seed=ordinal,
        )
        try:
            _ = env.reset(
                oya=oya,
                wall=wall,
                scores=list(scores),
                honba=honba,
                kyotaku=kyotaku,
                round_wind=wind,
            )  # discard initial obs; reset side-effect installs the position
        except Exception as exc:
            raise self._fail(f"engine reset failed: {exc}") from exc
        self._env = env

    @property
    def _engine(self) -> Any:
        if self._env is None:
            raise self._fail("oracle engine used before reset")
        return self._env

    def _legals(self, pid: int) -> list[Any]:
        try:
            return list(self._engine.get_observation(pid).legal_actions())
        except Exception as exc:
            raise self._fail(f"seat {pid} legal query failed: {exc}") from exc

    @staticmethod
    def _mjai(raw: Any) -> dict[str, Any]:
        mjai: Any = raw.to_mjai()
        if isinstance(mjai, str):
            mjai = json.loads(mjai)
        if not isinstance(mjai, dict):
            raise ContractError(f"engine action without an MJAI mapping: {mjai!r}")
        return dict(mjai)

    def _find(
        self,
        pid: int,
        *,
        mjai_type: str,
        tile_str: str | None = None,
        consumed_strs: Sequence[str] | None = None,
    ) -> Any:
        matches: list[Any] = []
        for raw in self._legals(pid):
            try:
                mjai = self._mjai(raw)
            except (ContractError, ValueError):
                continue
            if str(mjai.get("type", "")) != mjai_type:
                continue
            if tile_str is not None:
                tile_raw: Any = raw.tile
                if tile_raw is None or mjai_string_of(int(tile_raw)) != tile_str:
                    continue
            if consumed_strs is not None:
                got = sorted(mjai_string_of(int(t)) for t in raw.consume_tiles)
                if got != sorted(consumed_strs):
                    continue
            matches.append(raw)
        if len(matches) == 0:
            raise self._fail(f"seat {pid} has no {mjai_type} offer for {tile_str!r}")
        return matches[0]

    def _step(self, moves: dict[int, Any], *, where: str) -> None:
        try:
            self._engine.step(moves)
        except Exception as exc:
            raise self._fail(f"engine step failed at {where}: {exc}") from exc

    def note_tsumo(self, actor: int, pai: str) -> None:
        """Assert-or-draw the logged tsumo (rinshan draws included)."""
        self._missed.discard(actor)
        drawn_raw: Any = self._engine.drawn_tile
        if drawn_raw is not None and mjai_string_of(int(drawn_raw)) == pai:
            self._confirm_draw_holder(actor, pai, drawn_raw)
            return
        if drawn_raw is not None:
            raise self._fail(f"seat {actor} drew a different tile than logged")
        self._step({}, where=f"tsumo seat {actor}")
        drawn_raw = self._engine.drawn_tile
        if drawn_raw is None or mjai_string_of(int(drawn_raw)) != pai:
            raise self._fail(f"seat {actor} drew a different tile than logged")
        self._confirm_draw_holder(actor, pai, drawn_raw)

    def _confirm_draw_holder(self, actor: int, pai: str, drawn_raw: Any) -> None:
        """Fail closed when the drawn tile sits in another seat's hand.

        The engine auto-draws the next seat on discard steps while draws
        match by string, so an out-of-turn log tsumo can string-match a tile
        the engine dealt elsewhere; physical-id membership names the true
        drawer at the tsumo instead of desyncing a later row.
        """
        try:
            hands: Any = self._engine.hands
            owned = int(drawn_raw) in {int(t) for t in hands[actor]}
        except Exception as exc:
            raise self._fail(f"seat {actor} drawer query failed: {exc}") from exc
        if not owned:
            raise self._fail(
                f"seat {actor} tsumo {pai!r} sits in another seat's hand (out-of-turn draw)"
            )

    def do_dahai(self, actor: int, pai: str) -> None:
        """Step one logged discard (tsumogiri and tedashi share a slot)."""
        action = self._find(actor, mjai_type="dahai", tile_str=pai)
        self._step({actor: action}, where=f"dahai seat {actor}")

    def do_reach(self, actor: int) -> None:
        """Step the riichi declaration (the discard follows separately)."""
        action = self._find(actor, mjai_type="reach")
        self._step({actor: action}, where=f"reach seat {actor}")

    def do_ankan(self, actor: int, consumed: Sequence[str]) -> None:
        """Step one logged closed kan."""
        action = self._find(actor, mjai_type="ankan", consumed_strs=list(consumed))
        self._step({actor: action}, where=f"ankan seat {actor}")

    def do_kakan(self, actor: int, pai: str) -> None:
        """Step one logged added kan."""
        action = self._find(actor, mjai_type="kakan", tile_str=pai)
        self._step({actor: action}, where=f"kakan seat {actor}")

    def do_tsumo_win(self, actor: int, pai: str) -> None:
        """Step one logged self-draw win."""
        for mjai_type in ("hora", "tsumo"):
            try:
                action = self._find(actor, mjai_type=mjai_type, tile_str=pai)
            except ContractError:
                continue
            self._step({actor: action}, where=f"tsumo seat {actor}")
            return
        raise self._fail(f"seat {actor} has no tsumo win offer for {pai!r}")

    def window_open(self, discarder: int) -> bool:
        """The engine's own claim-window predicate after a discard/kan step."""
        env = self._engine
        try:
            waiting = int(env.phase) == int(riichienv.Phase.WaitResponse)
        except Exception as exc:
            raise self._fail(f"phase query failed: {exc}") from exc
        if not waiting:
            return False
        for pid in range(4):
            if pid == discarder:
                continue
            if len(self._legals(pid)) > 0:
                return True
        return False

    def resolve_window(self, discarder: int, claim: dict[str, object] | None) -> None:
        """Step one combined window resolution (claim or all-pass).

        Ron windows are never stepped: winning terminates the engine into an
        auto-dealt next hand (all subsequent queries would read garbage), and
        the kyoku is decided by the logged hora anyway. Responders passing a
        genuine ron offer are recorded as temporarily furiten manually; the
        winner is excluded (they won, not passed).
        """
        if claim is not None and str(claim.get("type", "")) == "hora":
            actor_raw = claim.get("actor")
            winner = None if isinstance(actor_raw, bool) else actor_raw
            for pid in range(4):
                if pid in (discarder, winner):
                    continue
                try:
                    legals = self._legals(pid)
                except ContractError:
                    continue
                for raw in legals:
                    try:
                        mjai = self._mjai(raw)
                    except (ContractError, ValueError):
                        continue
                    if str(mjai.get("type", "")) in ("hora", "ron"):
                        self._missed.add(pid)
                        break
            return
        moves: dict[int, Any] = {}
        if claim is not None:
            kind = str(claim.get("type", ""))
            actor_raw = claim.get("actor")
            if isinstance(actor_raw, bool) or not isinstance(actor_raw, int):
                raise self._fail("window claim without an actor")
            actor = actor_raw
            if kind not in ("chi", "pon", "daiminkan"):
                raise self._fail(f"window claim of unexpected kind {kind!r}")
            pai = str(claim.get("pai", ""))
            consumed = [str(t) for t in cast("Any", claim.get("consumed", []))]
            moves[actor] = self._find(actor, mjai_type=kind, tile_str=pai, consumed_strs=consumed)
        for pid in range(4):
            if pid == discarder or pid in moves:
                continue
            if len(self._legals(pid)) == 0:
                continue
            moves[pid] = self._find(pid, mjai_type="none")
        if len(moves) == 0:
            return
        self._step(moves, where="window resolution")

    def furiten(self, pid: int) -> str:
        """Exact furiten state from the live engine flags plus ron passes.

        Ron windows are never stepped (winning would terminate the engine),
        so responders passing a genuine ron offer are recorded manually; the
        engine's own doujun/riichi flags cover everything else.
        """
        from hydra2.engines.riichienv.state import furiten_of

        try:
            state = furiten_of(self._engine, pid)
        except Exception as exc:
            raise self._fail(f"seat {pid} furiten query failed: {exc}") from exc
        if state == "none" and pid in self._missed:
            return "temporary"
        return state

    def offers_kyushu(self, pid: int) -> bool:
        """Whether the live engine offers the nine-terminals abort to ``pid``."""
        for raw in self._legals(pid):
            try:
                action_type = int(cast("Any", raw.action_type))
            except (TypeError, ValueError):
                continue
            if action_type == int(riichienv.ActionType.KYUSHU_KYUHAI):
                return True
        return False

    def position(self, pid: int) -> _TablePosition:
        """Frozen live-table snapshot backing oracle-less forced rows."""
        env = self._engine
        try:
            hands_raw: Any = env.hands
            rivers_raw: Any = env.discards
            drawn_raw: Any = env.drawn_tile
            dora_raw: Any = env.dora_indicators
            sticks_raw: Any = env.riichi_sticks
            declared_raw: Any = env.riichi_declared
            legals = list(self._legals(pid))
            hand = tuple(int(t) for t in hands_raw[pid])
            rivers = tuple(tuple(int(t) for t in river) for river in rivers_raw)
            lens = tuple(len(cast("Any", item)) for item in hands_raw)
            dora = tuple(int(t) for t in dora_raw)
            sticks = int(sticks_raw)
            declared = tuple(bool(v) for v in declared_raw)
            drawn = None if drawn_raw is None else int(drawn_raw)
        except ContractError:
            raise
        except Exception as exc:
            raise self._fail(f"seat {pid} position query failed: {exc}") from exc
        if len(declared) != 4 or len(lens) != 4 or len(rivers) != 4:
            raise self._fail(f"seat {pid} position must cover 4 seats")
        return _TablePosition(
            hand=hand,
            drawn=drawn,
            dora=dora,
            rivers=rivers,
            lens=lens,
            sticks=sticks,
            declared=declared,
            legals=tuple(legals),
        )

    def check_row(
        self,
        seat: int,
        step: _SimStep,
        *,
        melds: Sequence[Sequence[VisibleMeld]],
        hand_check: bool,
    ) -> None:
        """String-level agreement between oracle observation and engine state.

        Draw rows compare hands (with drawn tile), rivers, and meld counts;
        claim rows compare rivers and meld counts only (hands move through
        the meld at different points of the two pipelines).
        """
        env = self._engine
        try:
            hands: Any = env.hands
            rivers: Any = env.discards
            meld_rows: Any = env.melds
        except Exception as exc:
            raise self._fail(f"seat {seat} state query failed: {exc}") from exc
        if hand_check:
            hand_strings = sorted(mjai_string_of(t) for t in step.hand)
            engine_strings = sorted(mjai_string_of(int(t)) for t in hands[seat])
            if hand_strings != engine_strings:
                raise self._fail(f"seat {seat} hand differs from the engine state")
            if step.drawn is not None:
                drawn_raw: Any = env.drawn_tile
                if drawn_raw is None or mjai_string_of(int(drawn_raw)) != mjai_string_of(
                    step.drawn
                ):
                    raise self._fail(f"seat {seat} drawn tile differs from engine state")
        for other in range(4):
            river_strings = [mjai_string_of(int(t)) for t in rivers[other]]
            oracle_strings = [mjai_string_of(t) for t in step.discards[other]]
            if river_strings != oracle_strings:
                raise self._fail(f"seat {other} river differs from engine state")
        for owner in range(4):
            if len(meld_rows[owner]) != len(melds[owner]):
                raise self._fail(f"seat {owner} meld count differs from engine state")
