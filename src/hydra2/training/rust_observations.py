"""Full-observation assembly over the Rust replay stream (Slice 8 cutover).

The Rust JSON handoff emits a 13-field projection per row (strings, sparse
mask, kinds, hashes) — enough for parity, not for the encoder, which needs
full :class:`ActorObservation` content (take ids, melds, rivers, scores,
histories, furiten). This module rebuilds full observations WITHOUT a
second engine:

- takes are string-derivable everywhere they affect features: concealed
  and rivers render pool-base takes (PG-S8 probe: 42553/42553 concealed,
  rivers base-rendered pre-accept), drawn renders the live-string base,
  dora renders pool-first takes with repeat-marker reuse (42553/42553),
  meld takes follow the logged consumed strings with called-take swaps
  (meld-sane on the probe); all other takes use valid distinct copies of
  the right strings (feature-identical: features read counts/kinds, and
  the observation hash is recomputed over the rebuilt content);
- scalars (scores, sticks, honba, oya, riichi states, kan count) track the
  framed log mechanically (hora deltas, headers, reach markers);
- history envelopes translate the framed log in adapter order through the
  kept envelope constructors; kinds and order are verified against the
  Rust ``history_kinds`` per row (fail closed on divergence);
  rows assemble through the kept :class:`ObservationBuilder` (exact ids,
  sequences, hashes, validity), with masks from Rust (densified) and
  chosen ids from Rust.

Feature-identity (not hash-identity) is the contract: encoding assembled
rows yields byte-identical feature tensors to encoding oracle rows.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from hydra2.contracts.common import ContractError
from hydra2.data.decode import GameRecord
from hydra2.data.parquet import DecisionRow
from hydra2.engines.riichienv.log_replay import SIM_DERIVATION_MARK
from hydra2.engines.riichienv.tiles import mjai_string_of, physical_of

#: Full-doc envelope version assembled here (mirrors
#: `hydra2_row::BRIDGE_FULL_DOC_ENVELOPE_VERSION` and the
#: `py_stream::DECISION_ENVELOPE_VERSION` v1 projection input). The marker
#: is documentary at the row level (the `DecisionRow` shape is frozen):
#: ingress distinguishes expanded full docs (35-field engine `to_json`
#: documents) from unexpanded projections (13-key docs carrying the
#: `projection` tag) by shape, rejecting the latter fail-closed
#: (`unexpanded-projection-row`). Bump only with a matching ingress update
#: plus an oracle re-freeze.
BRIDGE_FULL_DOC_ENVELOPE_VERSION = "full-doc-v1"


def _pool_of(pai: str) -> list[int]:
    if pai in ("5mr", "0m"):
        return [16]
    if pai in ("5pr", "0p"):
        return [52]
    if pai in ("5sr", "0s"):
        return [88]
    first = int(physical_of(pai))
    if len(pai) == 2 and pai[0] == "5":
        return [first, first + 1, first + 2]
    base = (first // 4) * 4
    return [base, base + 1, base + 2, base + 3]


def _norm(s: str) -> str:
    return {"0m": "5mr", "0p": "5pr", "0s": "5sr"}.get(s, s)


def _is_aka(s: str) -> bool:
    return _norm(s) in ("5mr", "5pr", "5sr")


def _base_take(pai: str) -> int:
    return _pool_of(_norm(pai))[0]


def _recompute_takes(strings: list[str]) -> list[int]:
    """Pool-base takes over a held string multiset (concealed rendering)."""
    out: list[int] = []
    for text, count in sorted(Counter(strings).items()):
        if _is_aka(text):
            out.append(_pool_of(_norm(text))[0])
            continue
        pool = _pool_of(_norm(text))
        out.extend(pool[n] for n in range(count))
    return sorted(out)


# ---------------------------------------------------------------------------
# Tracked string hands + valid take assignment (features read counts/kinds;
# dora/drawn takes are exact rules, all other takes are valid distinct
# copies of the right strings).
# ---------------------------------------------------------------------------


class _Hands:
    """Per-seat string multisets with on-demand valid take assignment."""

    def __init__(self) -> None:
        self.strings: list[Counter] = [Counter(), Counter(), Counter(), Counter()]

    def install(self, seat: int, text: str) -> None:
        self.strings[seat][_norm(text)] += 1

    def draw(self, seat: int, text: str) -> None:
        self.strings[seat][_norm(text)] += 1

    def discard(self, seat: int, text: str) -> None:
        key = _norm(text)
        if self.strings[seat][key] <= 0:
            raise ContractError(f"no held copy of discard {text!r} for seat {seat}")
        self.strings[seat][key] -= 1
        if self.strings[seat][key] == 0:
            del self.strings[seat][key]

    def takes(self, seat: int) -> list[int]:
        return _recompute_takes(sorted(self.strings[seat].elements()))

    def consume(self, seat: int, strings: list[str], called: int | None = None) -> list[int]:
        """Valid distinct takes for logged consumed strings (meld-sane)."""
        held = self.takes(seat)
        picked: list[int] = []
        for text in sorted(strings):
            key = _norm(text)
            choice = None
            for take in held:
                if take in picked:
                    continue
                if mjai_string_of(int(take)) == key:
                    choice = take
                    break
            if choice is None:
                raise ContractError(f"no held copy left for meld tile {text!r}")
            picked.append(choice)
        if called is not None:
            used = set(picked) | {called}
            for pos, take in enumerate(picked):
                if int(take) == int(called):
                    key = _norm(mjai_string_of(int(take)))
                    for candidate in _pool_of(key):
                        if candidate not in used:
                            picked[pos] = candidate
                            used.add(candidate)
                            break
                    else:
                        raise ContractError(f"no distinct copy left for meld tile {key!r}")
        for text in strings:
            self.discard(seat, text)
        return sorted(picked)


def _kakan_added(prior_tiles: list[int], pai: str) -> int:
    """The one pool copy of ``pai`` absent from the prior pon triple."""
    pool = _pool_of(_norm(pai))
    if len(pai) == 2 and pai[0] == "5":
        first = int(physical_of(pai))
        base = (first // 4) * 4
        pool = [base, base + 1, base + 2, base + 3]
    have = set(prior_tiles)
    missing = [c for c in pool if c not in have]
    if len(missing) != 1:
        raise ContractError(f"prior pon leaves no single added copy for kakan {pai!r}")
    return missing[0]


_BAKAZE_WIND = {"E": 27, "S": 28, "W": 29, "N": 30}
_ROUND_WIND = {"E": 0, "S": 1, "W": 2, "N": 3}


class _GameWalk:
    """Per-game tracked state: strings, scalars, dora, envelopes-in-waiting."""

    def __init__(self, game_id: str) -> None:
        self.game_id = game_id
        self.hands = _Hands()
        self.scores: list[int] = [25000, 25000, 25000, 25000]
        self.sticks = 0
        self.honba = 0
        self.oya = 0
        self.round_index = -1
        self.hand_number = 0
        self.kyoku_no = 0
        self.event_count = 0
        self.seq = 0
        self.dora_used: dict[str, int] = {}
        self.dora_list: list[int] = []
        self.riichi: list[str] = ["none", "none", "none", "none"]
        self.missed: list[set[str]] = [set(), set(), set(), set()]
        self.rivers: list[list[int]] = [[], [], [], []]
        self.melds: list[list[dict[str, Any]]] = [[], [], [], []]
        self.live: dict[int, str] = {}
        self.window: tuple[int, str, set[int]] | None = None
        self.pending_reach: set[int] = set()
        self.pending_accepts: list[int] = []
        # Ippatsu window flags, one per seat (D-WP03A-9 intended semantics,
        # mirrored from the adapter event derivation, NOT the engine stub:
        # set at reach_accepted, cleared by any meld interrupt (chi/pon/kan)
        # and by the declarer's own next discard, reset every hand. The
        # adapter never exposes a runtime ippatsu property (engine 0.4.8),
        # so these flags derive deterministically from the framed log. Any
        # malformed transition raises ContractError with an
        # "ippatsu-active-unrecoverable" reason (fail closed, never stubbed).
        self.ippatsu: list[bool] = [False, False, False, False]

    def install(self, ev: dict[str, Any]) -> None:
        tehais = ev.get("tehais", [])
        self.oya = int(ev.get("oya", 0))
        self.bakaze = str(ev.get("bakaze", "E"))
        self.honba = int(ev.get("honba", 0))
        scores = ev.get("scores")
        if isinstance(scores, list) and len(scores) == 4:
            self.scores = [int(v) for v in scores]
        kyotaku = ev.get("kyotaku")
        if isinstance(kyotaku, int) and not isinstance(kyotaku, bool):
            self.sticks = kyotaku
        kyoku = ev.get("kyoku")
        if isinstance(kyoku, int):
            self.kyoku_no = kyoku
        self.round_index += 1
        wind = {"E": 0, "S": 1, "W": 2, "N": 3}.get(str(ev.get("bakaze", "E")), 0)
        self.hand_number = wind * 4 + max(1, self.kyoku_no)
        self.hands = _Hands()
        for seat in range(4):
            hand = tehais[seat] if seat < len(tehais) else []
            for text in hand:
                self.hands.install(seat, str(text))
        self.riichi = ["none", "none", "none", "none"]
        self.missed = [set(), set(), set(), set()]
        self.rivers = [[], [], [], []]
        self.melds = [[], [], [], []]
        self.live = {}
        self.window = None
        self.pending_reach = set()
        self.pending_accepts = []
        # New hand: every ippatsu chance resets (mirrors the adapter's
        # per-hand window; the declarer must re-declare to reopen it).
        self.ippatsu = [False, False, False, False]
        self.dora_used = {}
        self.dora_list = []

    def reveal_dora(self, marker: str) -> int:
        key = _norm(marker)
        if key in self.dora_used:
            take = self.dora_used[key]
        else:
            take = _pool_of(key)[0]
            self.dora_used[key] = take
        self.dora_list.append(take)
        return take

    def apply_scores(self, deltas: Any) -> None:
        if isinstance(deltas, list) and len(deltas) == 4:
            self.scores = [a + b for a, b in zip(self.scores, deltas)]

    def _require_ippatsu_seat(self, actor: Any, *, where: str) -> int:
        """Validate a seat for an ippatsu transition (fail closed, named reason)."""
        if isinstance(actor, bool) or not isinstance(actor, int) or not 0 <= actor <= 3:
            raise ContractError(
                f"ippatsu-active-unrecoverable: {where} carries unusable actor {actor!r}"
            )
        return actor

    def open_ippatsu(self, actor: Any, *, where: str) -> None:
        """Open one seat's ippatsu window at reach_accepted (adapter order)."""
        seat = self._require_ippatsu_seat(actor, where=where)
        self.ippatsu[seat] = True

    def close_ippatsu_seat(self, actor: Any, *, where: str) -> None:
        """Close one seat's window at its own next discard (adapter order)."""
        seat = self._require_ippatsu_seat(actor, where=where)
        self.ippatsu[seat] = False

    def close_ippatsu_all(self) -> None:
        """Close every window at any meld interrupt (chi/pon/kan)."""
        self.ippatsu = [False, False, False, False]

    def ippatsu_snapshot(self) -> tuple[bool, bool, bool, bool]:
        """Current window flags for the public snapshot (fail closed on drift)."""
        flags = self.ippatsu
        if (
            not isinstance(flags, list)
            or len(flags) != 4
            or any(not isinstance(flag, bool) for flag in flags)
        ):
            raise ContractError(
                f"ippatsu-active-unrecoverable: window flags drifted to {flags!r}"
            )
        return (flags[0], flags[1], flags[2], flags[3])


class _Emitter:
    """Envelope sequence builder (adapter order, Rust kinds verify)."""

    def __init__(self, walk: _GameWalk, rules_hash: str, schema_hash: str) -> None:
        from hydra2.engines.riichienv.events import make_delta, make_envelope

        self.walk = walk
        self.rules_hash = rules_hash
        self.schema_hash = schema_hash
        self.make_envelope = make_envelope
        self.make_delta = make_delta
        self.envelopes: list[Any] = []
        self.builder: Any = None
    def emit(
        self,
        kind: str,
        visibility: str,
        *,
        actor: int | None = None,
        tile: int | None = None,
        action_id: int | None = None,
        source_seat: int | None = None,
        consumed: list[int] | None = None,
        offered: list[int] | None = None,
        accepted: list[int] | None = None,
        round_index: int | None = None,
        scores: list[int] | None = None,
        reason: str | None = None,
        deltas: list[Any] | None = None,
    ) -> Any:
        env = self.make_envelope(
            game_id=self.walk.game_id,
            sequence=self.walk.event_count,
            kind=kind,  # type: ignore[arg-type]
            visibility=visibility,  # type: ignore[arg-type]
            rules_hash=self.rules_hash,
            schema_hash=self.schema_hash,
            actor=actor,
            tile=tile,
            action_id=action_id,
            source_seat=source_seat,
            consumed_tiles=tuple(consumed or ()),
            offered_action_ids=tuple(offered or ()),
            accepted_action_ids=tuple(accepted or ()),
            round_index=round_index,
            scores=None if scores is None else tuple(scores),
            reason=reason,
            public_delta=tuple(deltas or ()),
        )
        self.walk.event_count += 1
        self.envelopes.append(env)
        if self.builder is not None:
            self.builder.append_visible(env)
        return env

    def meld_delta(self, kind: str, owner: int, tiles: list[int], called: int | None, source: int | None) -> Any:
        ordered = sorted(tiles)
        value: dict[str, Any] = {
            "meld_id": f"{kind}:" + ".".join(str(t) for t in ordered),
            "kind": kind,
            "owner": owner,
            "source_seat": source,
            "called_tile": called,
            "tiles": ordered,
        }
        return self.make_delta(("melds", owner), "append", value)

    def dora_delta(self, take: int) -> Any:
        return self.make_delta(("dora_indicators",), "append", take)


_TABLE_INDEX: dict[tuple, int] | None = None


def _table_index() -> dict[tuple, int]:
    global _TABLE_INDEX
    if _TABLE_INDEX is None:
        from hydra2.config import repo_root
        from hydra2.contracts.action import ACTION_TABLE_RELPATH, load_action_table

        table = load_action_table(str(repo_root() / ACTION_TABLE_RELPATH))
        index: dict[tuple, int] = {}
        for pos, template in enumerate(table.actions):
            key = (
                str(template.kind),
                None if template.tile is None else int(template.tile),
                None if template.called_tile is None else int(template.called_tile),
                tuple(int(t) for t in template.consumed_tiles),
                template.source_offset,
                bool(template.declares_riichi),
                bool(template.meld_ref_required),
            )
            index.setdefault(key, pos)
        _TABLE_INDEX = index
    return _TABLE_INDEX


def _action_id(kind: str, tile: int | None, called: int | None, consumed: list[int]) -> int:
    index = _table_index()
    for offset in (None, 0, 1, 2, -1, -2):
        for riichi in (False, True):
            for meldref in (False, True):
                key = (kind, tile, called, tuple(consumed), offset, riichi, meldref)
                if key in index:
                    return index[key]
    raise ContractError(f"no action-table template for {kind} tile={tile} called={called} consumed={consumed}")


_ID_TO_KIND: dict[int, str] | None = None


def _id_to_kind() -> dict[int, str]:
    global _ID_TO_KIND
    if _ID_TO_KIND is None:
        inv: dict[int, str] = {}
        for key, pos in _table_index().items():
            inv[pos] = str(key[0])
        _ID_TO_KIND = inv
    return _ID_TO_KIND


def _waits(concealed: list[int], meld_kinds: list[tuple[str, list[int]]]) -> set[str]:
    try:
        import riichienv
    except Exception:
        return set()
    codes = {
        "chi": (riichienv.MeldType.Chi, True),
        "pon": (riichienv.MeldType.Pon, True),
        "daiminkan": (riichienv.MeldType.Daiminkan, True),
        "ankan": (riichienv.MeldType.Ankan, False),
        "kakan": (riichienv.MeldType.Kakan, True),
    }
    meld_objs = []
    for kind, tiles in meld_kinds:
        try:
            code, opened = codes[kind]
            meld_objs.append(riichienv.Meld(code, [int(t) for t in tiles], opened))
        except Exception:
            return set()
    try:
        raw = riichienv.HandEvaluator(sorted(int(t) for t in concealed), meld_objs).get_waits()
    except Exception:
        return set()
    out: set[str] = set()
    for item in raw or []:
        try:
            out.add(mjai_string_of(int(item)))
        except Exception:
            continue
    return out


def _builder_for(game_id: str) -> Any:
    import hashlib

    from hydra2.config import repo_root
    from hydra2.contracts.action import ACTION_TABLE_RELPATH, load_action_table
    from hydra2.contracts.event import (
        EVENT_SCHEMA_RELPATH,
        build_packet_boundary_payload,
        compute_event_schema_digest,
        parse_event_schema,
    )
    from hydra2.contracts.observation import ObservationBuilder
    from hydra2.data.replay_expand import _load_rules

    manifest = _load_rules()
    table = load_action_table(str(repo_root() / ACTION_TABLE_RELPATH))
    schema_payload = parse_event_schema((repo_root() / EVENT_SCHEMA_RELPATH).read_bytes())
    payload = schema_payload.get("payload")
    if not isinstance(payload, dict) or "digest" not in payload:
        raise ContractError("event schema artifact lacks a digest")
    event_schema_hash = str(payload["digest"])
    packet_hash = str(compute_event_schema_digest(build_packet_boundary_payload()))
    published = repo_root() / "configs" / "rules" / f"{manifest.rules_id}.json"
    if published.is_file():
        rules_hash = "sha256:" + hashlib.sha256(published.read_bytes()).hexdigest()
    else:
        from hydra2.engines.riichienv.state import rules_identity_hash

        rules_hash = str(rules_identity_hash(manifest))
    return ObservationBuilder(
        game_id=game_id,
        rules_id=manifest.rules_id,
        rules_hash=rules_hash,
        action_table_hash=table.digest,
        expected_legal_mask_length=6792,
        event_schema_hash=event_schema_hash,
        packet_boundary_hash=packet_hash,
    )

class _Assembler:
    """One-game log walk feeding the kept builder from Rust decisions."""

    def __init__(self, game: GameRecord, split: str, rust_rows: list[dict[str, Any]]) -> None:
        from hydra2.engines.riichienv.state import seat_winds_for_dealer

        self.game = game
        self.split = split
        self.rust = rust_rows
        self.walk = _GameWalk(str(game.game_id))
        self.seat_winds_for_dealer = seat_winds_for_dealer
        self.builder: Any = None
        self.emitter: Any = None
        self.pos = 0
        self.out: list[DecisionRow] = []
        self.kyoku_env0 = 0

    def _new_builder(self) -> None:
        self.builder = _builder_for(str(self.game.game_id))
        first = self.rust[0] if self.rust else {}
        self.emitter = _Emitter(self.walk, str(first.get("rules_hash", "")), "")
        schema = self.builder._event_schema_hash  # type: ignore[attr-defined]
        self.emitter.schema_hash = str(schema)
        self.emitter.builder = self.builder

    def _snapshot(self, seat: int, phase: str, turn_actor: int, wall: int) -> None:
        walk = self.walk
        self.builder.update_public_state(
            decision_id=f"{walk.game_id}:{walk.event_count}",
            round_index=max(0, walk.round_index),
            round_wind=_BAKAZE_WIND.get(walk.bakaze, 27),
            hand_number=max(1, walk.hand_number),
            seat_winds=self.seat_winds_for_dealer(walk.oya),
            honba=walk.honba,
            riichi_sticks=walk.sticks,
            dealer=walk.oya,
            scores=tuple(walk.scores),
            turn_actor=turn_actor,
            phase=phase,  # type: ignore[arg-type]
            live_wall_tiles_remaining=max(0, int(wall)),
            ippatsu_active=walk.ippatsu_snapshot(),
        )

    def _furiten(self, seat: int) -> str:
        walk = self.walk
        if walk.riichi[seat] == "accepted":
            return "riichi"
        if not walk.missed[seat]:
            return "none"
        concealed = _recompute_takes(sorted(walk.hands.strings[seat].elements()))
        kinds: list[tuple[str, list[int]]] = []
        for owner_melds in walk.melds:
            for meld in owner_melds:
                if int(meld["owner"]) == seat:
                    kinds.append((str(meld["kind"]), [int(t) for t in meld["tiles"]]))
        waits = _waits(concealed, kinds)
        if waits and (waits & walk.missed[seat]):
            return "temporary"
        return "none"

    def _can_flags(self, mask: list[bool]) -> tuple[bool, bool]:
        inv = _id_to_kind()
        can_tsumo = any(inv.get(i) == "tsumo" for i, flag in enumerate(mask) if flag)
        can_riichi = any("riichi" in (inv.get(i) or "") for i, flag in enumerate(mask) if flag)
        return can_tsumo, can_riichi

    def _close_window(self, claimer: int | None, accepted: list[int] | None) -> None:
        walk = self.walk
        if walk.window is None:
            return
        discarder, text, pending = walk.window
        for seat in pending:
            if seat == discarder:
                continue
            if claimer is not None and seat == claimer:
                walk.missed[seat].discard(text)
                continue
            walk.missed[seat].add(text)
        walk.window = None
        aids = [int(a) for a in (accepted or [])]
        self.emitter.emit(
            "call_resolved", "server_private", offered=aids, accepted=aids
        )

    def _flush_accepts(self) -> None:
        walk = self.walk
        for actor in walk.pending_accepts:
            walk.sticks += 1
            walk.scores[actor] -= 1000
        walk.pending_accepts = []
    def _emit_row(self, seat: int) -> None:
        walk = self.walk
        if self.pos >= len(self.rust):
            raise ContractError(f"rust rows exhausted at decision {walk.seq}")
        row = self.rust[self.pos]
        expected_did = f"{walk.game_id}:d{walk.seq:04d}"
        if str(row.get("decision_id")) != expected_did:
            raise ContractError(
                f"rust row order diverged: got {row.get('decision_id')} want {expected_did}"
            )
        if int(row.get("seat", -1)) != seat:
            raise ContractError(f"rust row seat {row.get('seat')} != actor {seat} at {expected_did}")
        actor_doc = row.get("actor_observation", {})
        kinds = list(actor_doc.get("history_kinds", []))
        mine = self._kinds_since_kyoku(seat)
        if mine != kinds:
            first = next((i for i, (a, b) in enumerate(zip(mine, kinds)) if a != b), min(len(mine), len(kinds)))
            raise ContractError(
                f"history kinds diverged at {expected_did} "
                f"(len mine={len(mine)} rust={len(kinds)} first-diff@{first}): "
                f"mine={mine} rust={kinds}"
            )
        phase = str(actor_doc.get("phase", "draw_decision"))
        turn_actor = int(actor_doc.get("turn_actor", seat))
        wall = int(actor_doc.get("live_wall_tiles_remaining", 0))
        self._snapshot(seat, phase, turn_actor, wall)
        strings = sorted(walk.hands.strings[seat].elements())
        concealed = _recompute_takes(strings)
        if seat in walk.live:
            base = _base_take(walk.live[seat])
            if base in concealed:
                concealed.remove(base)
        self.builder.set_concealed_hand(seat, concealed)
        mask_raw = list(actor_doc.get("legal_mask", []))
        mask = [False] * 6792
        for value in mask_raw:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ContractError(f"rust legal_mask malformed at {expected_did}")
            if value < 0 or value >= 6792:
                raise ContractError(f"rust legal_mask id out of range at {expected_did}")
            mask[int(value)] = True
        can_tsumo, can_riichi = self._can_flags(mask)
        self.builder.set_actor_state(
            seat, furiten=self._furiten(seat), can_tsumo=can_tsumo, can_riichi=can_riichi
        )
        observation = self.builder.build(actor=seat, legal_mask=mask)
        doc = observation.to_json()
        from hydra2.artifacts.digest import of_canonical

        derivation = str(
            of_canonical(
                {
                    "game_id": str(row.get("game_id")),
                    "decision_id": str(row.get("decision_id")),
                    "observation_hash": str(observation.observation_hash),
                    "chosen_action_id": int(row.get("chosen_action_id")),
                    "wall_digest": None,
                    "adapter_hash": str(row.get("adapter_hash")),
                    "derivation": SIM_DERIVATION_MARK,
                }
            )
        )
        self.out.append(
            DecisionRow(
                game_id=str(row.get("game_id")),
                round_id=str(row.get("round_id")),
                decision_id=str(row.get("decision_id")),
                seat=int(row.get("seat")),
                source_object_id=str(row.get("source_object_id")),
                split=str(row.get("split")),
                rules_hash=str(row.get("rules_hash")),
                adapter_hash=str(row.get("adapter_hash")),
                observation_hash=str(observation.observation_hash),
                action_table_hash=str(row.get("action_table_hash")),
                derivation_hash=derivation,
                actor_observation=dict(doc),
                chosen_action_id=int(row.get("chosen_action_id")),
                privileged_label_ref=str(row.get("decision_id")),
            )
        )
        walk.seq += 1
        self.pos += 1

    def _expect_window(self, seat: int) -> bool:
        if self.pos >= len(self.rust):
            return False
        kinds = list(self.rust[self.pos].get("actor_observation", {}).get("history_kinds", []))
        mine = self._kinds_since_kyoku(seat)
        return len(kinds) > len(mine) and kinds[len(mine)] == "call_window"

    def _kinds_since_kyoku(self, seat: int) -> list[str]:
        out: list[str] = []
        for env in self.emitter.envelopes[self.kyoku_env0:]:
            vis = str(getattr(env, "visibility", "public"))
            if vis == "public":
                out.append(str(env.kind))
            elif vis == "actor_private":
                to = [int(s) for s in getattr(env, "visible_to", ())]
                if to == [seat]:
                    out.append(str(env.kind))
        return out

    def _peek_next(self, events: list[dict[str, Any]], idx: int) -> dict[str, Any] | None:
        for nxt in events[idx + 1:]:
            kind = nxt.get("type")
            if kind in ("dora", "reach_accepted", "reach"):
                continue
            return nxt if isinstance(nxt, dict) else None
        return None

    def run(self) -> list[DecisionRow]:
        walk = self.walk
        events = [dict(e) for e in self.game.events]
        self.kyoku_env0 = 0
        for idx, ev in enumerate(events):
            kind = ev.get("type")
            if kind == "start_kyoku":
                walk.install(ev if isinstance(ev, dict) else {})
                self._new_builder()
                self.kyoku_env0 = len(self.emitter.envelopes)
                if walk.round_index == 0:
                    self.emitter.emit("game_start", "public", round_index=0, scores=list(walk.scores))
                self.emitter.emit(
                    "round_start", "public", actor=int(ev.get("oya", 0) or 0),
                    round_index=max(0, walk.round_index), scores=list(walk.scores),
                    deltas=[
                        self.emitter.make_delta(("round_index",), "set", max(0, walk.round_index)),
                        self.emitter.make_delta(("honba",), "set", walk.honba),
                        self.emitter.make_delta(("riichi_sticks",), "set", walk.sticks),
                        self.emitter.make_delta(("scores",), "set", list(walk.scores)),
                    ],
                )
            elif kind == "tsumo":
                self._flush_accepts()
                actor = int(ev.get("actor", 0))
                walk.missed[actor].clear()
                self._close_window(None, None)
                pai = _norm(str(ev.get("pai", "")))
                walk.hands.draw(actor, pai)
                walk.live[actor] = pai
                self.emitter.emit("turn_advance", "public", actor=actor)
                self.emitter.emit("draw_tile", "actor_private", actor=actor, tile=_base_take(pai))
                nxt = self._peek_next(events, idx)
                nxt_kind = nxt.get("type") if nxt else None
                if nxt_kind in ("ankan", "kakan", "hora"):
                    self._emit_row(actor)
                    if actor in walk.live:
                        del walk.live[actor]
            elif kind == "dahai":
                self._flush_accepts()
                actor = int(ev.get("actor", 0))
                pai = _norm(str(ev.get("pai", "")))
                take = _base_take(pai)
                tsumogiri = bool(ev.get("tsumogiri", False))
                if actor in walk.pending_reach:
                    dkind = "riichi_discard"
                    walk.pending_reach.discard(actor)
                elif tsumogiri:
                    dkind = "tsumogiri"
                else:
                    dkind = "discard"
                self._emit_row(actor)
                if actor in walk.live:
                    del walk.live[actor]
                aid = _action_id(dkind, take, None, [])
                self.emitter.emit("discard", "public", actor=actor, tile=take, action_id=aid)
                walk.hands.discard(actor, pai)
                walk.rivers[actor].append(take)
                # Adapter order (_on_dahai): the declarer's own next discard
                # closes only their window (other seats' discards leave it open).
                walk.close_ippatsu_seat(actor, where=f"dahai@{walk.seq}")
                if actor in walk.live:
                    del walk.live[actor]
                next_seat = int(self.rust[self.pos].get("seat", actor)) if self.pos < len(self.rust) else actor
                if self._expect_window(next_seat):
                    self.emitter.emit("call_window", "public")
                    pending: set[int] = set()
                    for other in range(4):
                        if other == actor:
                            continue
                        concealed = _recompute_takes(sorted(walk.hands.strings[other].elements()))
                        kinds: list[tuple[str, list[int]]] = []
                        for owner_melds in walk.melds:
                            for meld in owner_melds:
                                if int(meld["owner"]) == other:
                                    kinds.append((str(meld["kind"]), [int(t) for t in meld["tiles"]]))
                        if pai in _waits(concealed, kinds):
                            pending.add(other)
                    walk.window = (actor, pai, pending)
            elif kind in ("chi", "pon", "daiminkan"):
                actor = int(ev.get("actor", 0))
                called_str = _norm(str(ev.get("pai", "")))
                called = _base_take(called_str)
                consumed_strs = [_norm(str(s)) for s in (ev.get("consumed", []) or [])]
                source = ev.get("target")
                source_seat = int(source) if isinstance(source, int) else None
                self._emit_row(actor)
                if actor in walk.live:
                    del walk.live[actor]
                chosen = int(self.rust[self.pos - 1].get("chosen_action_id", 0))
                self._close_window(actor, [chosen])
                consumed = walk.hands.consume(actor, consumed_strs, called)
                aid = _action_id(str(kind), None, called, consumed)
                self.emitter.emit(
                    str(kind), "public", actor=actor, tile=called, action_id=aid,
                    source_seat=source_seat, consumed=consumed,
                    deltas=[self.emitter.meld_delta(str(kind), actor, consumed + [called], called, source_seat)],
                )
                walk.melds[actor].append(
                    {"kind": str(kind), "owner": actor, "tiles": sorted(consumed + [called]), "called": called, "source": source_seat}
                )
                # Adapter order (_on_call): any open call interrupts every
                # ippatsu chance on the table.
                walk.close_ippatsu_all()
            elif kind == "ankan":
                actor = int(ev.get("actor", 0))
                consumed_strs = [_norm(str(s)) for s in (ev.get("consumed", []) or [])]
                consumed = walk.hands.consume(actor, consumed_strs)
                aid = _action_id("ankan", None, None, consumed)
                self.emitter.emit(
                    "ankan", "public", actor=actor, action_id=aid, consumed=consumed,
                    deltas=[self.emitter.meld_delta("ankan", actor, consumed, None, None)],
                )
                walk.melds[actor].append(
                    {"kind": "ankan", "owner": actor, "tiles": sorted(consumed), "called": None, "source": None}
                )
                # Adapter order (_on_ankan): closed kan interrupts every chance.
                walk.close_ippatsu_all()
            elif kind == "kakan":
                actor = int(ev.get("actor", 0))
                pai = _norm(str(ev.get("pai", "")))
                prior = None
                for meld in walk.melds[actor]:
                    if meld["kind"] == "pon" and (min(meld["tiles"]) // 4) == (_base_take(pai) // 4):
                        prior = meld
                        break
                if prior is None:
                    raise ContractError(f"kakan of {pai!r}: no prior pon owned by seat {actor}")
                added = _kakan_added(list(prior["tiles"]), pai)
                aid = _action_id("kakan", added, None, [])
                self.emitter.emit(
                    "kakan", "public", actor=actor, tile=added, action_id=aid,
                    deltas=[self.emitter.meld_delta("kakan", actor, sorted(prior["tiles"]) + [added], None, None)],
                )
                prior["kind"] = "kakan"
                prior["tiles"] = sorted(prior["tiles"] + [added])
                walk.hands.discard(actor, pai)
                if actor in walk.live:
                    del walk.live[actor]
                # Adapter order (_on_kakan): added kan interrupts every chance.
                walk.close_ippatsu_all()
            elif kind == "hora":
                actor = int(ev.get("actor", 0))
                target = ev.get("target")
                # Scores/sticks move on headers (authoritative per kyoku):
                # hora rows show pre-movement values, so apply nothing here.
                if isinstance(target, int) and target != actor:
                    self._emit_row(actor)
                    if actor in walk.live:
                        del walk.live[actor]
                    chosen = int(self.rust[self.pos - 1].get("chosen_action_id", 0))
                    self._close_window(actor, [chosen])
            elif kind == "reach":
                actor = ev.get("actor")
                if isinstance(actor, int) and not isinstance(actor, bool):
                    walk.pending_reach.add(actor)
            elif kind == "reach_accepted":
                raw_actor = ev.get("actor", 0)
                actor = walk._require_ippatsu_seat(raw_actor, where="reach_accepted")
                walk.riichi[actor] = "accepted"
                walk.pending_accepts.append(actor)
                # Adapter order (_on_reach_accepted): the declarer's window
                # opens here; it closes on their next discard or any meld.
                walk.open_ippatsu(actor, where="reach_accepted")
                self.emitter.emit("riichi_accepted", "public", actor=actor)
            elif kind == "dora":
                marker = ev.get("dora_marker")
                if not isinstance(marker, str) or marker == "":
                    raise ContractError("kan-dora indicator unrecoverable: dora event without a dora_marker")
                take = walk.reveal_dora(marker)
                self.emitter.emit(
                    "dora_revealed", "public", tile=take,
                    deltas=[self.emitter.dora_delta(take)],
                )
            elif kind == "reach":
                actor = ev.get("actor")
                if isinstance(actor, int) and not isinstance(actor, bool):
                    walk.pending_reach.add(actor)
            elif kind == "reach_accepted":
                # Duplicate arm below the first (unreachable while the first
                # stands; kept in step so a future merge cannot silently drop
                # the ippatsu transition — MAIN owns deletion).
                raw_actor = ev.get("actor", 0)
                actor = walk._require_ippatsu_seat(raw_actor, where="reach_accepted")
                walk.riichi[actor] = "accepted"
                walk.pending_accepts.append(actor)
                walk.open_ippatsu(actor, where="reach_accepted")
                self.emitter.emit("riichi_accepted", "public", actor=actor)
            elif kind == "hora":
                actor = int(ev.get("actor", 0))
                target = ev.get("target")
                # Scores/sticks move on headers (authoritative per kyoku):
                # hora rows show pre-movement values, so apply nothing here.
                if isinstance(target, int) and target != actor:
                    self._emit_row(actor)
                    if actor in walk.live:
                        del walk.live[actor]
                    chosen = int(self.rust[self.pos - 1].get("chosen_action_id", 0))
                    self._close_window(actor, [chosen])
            elif kind == "ryukyoku":
                pass
            elif kind == "end_kyoku":
                self._close_window(None, None)
            elif kind in ("end_game", "start_game"):
                if kind == "start_game":
                    scores = [25000, 25000, 25000, 25000]
                    for probe in events:
                        if probe.get("type") == "start_kyoku" and isinstance(probe.get("scores"), list):
                            scores = [int(v) for v in probe["scores"][:4]]
                            break
                    walk.scores = scores
            if kind not in ("dora", "reach_accepted"):
                walk.prev_kind = kind
        if self.pos != len(self.rust):
            raise ContractError(f"assembled {self.pos} rows, rust produced {len(self.rust)}")
        return self.out


def assemble_game_rows(game: GameRecord, split: str, rust_rows: list[dict[str, Any]]) -> list[DecisionRow]:
    """Assemble full training rows over Rust replay decisions (no engine)."""
    assembler = _Assembler(game, split, rust_rows)
    return assembler.run()