"""Engine-state extraction: public snapshots, settlement facts, state digests.

All values here come straight off verified RiichiEnv 0.4.8 runtime
properties; nothing is inferred from mjai strings except where noted.

Owner decisions recorded:
* D-WP03A-3 furiten mapping: ``riichi_declared`` -> "riichi";
  ``missed_agari_doujun`` -> "temporary"; otherwise "none". The engine
  exposes no separate permanent discard-furiten flag, so the literal
  "discard" is reserved and currently unreachable (conformance watchpoint).
* D-WP03A-4 rules identity: ``sha256`` over the RFC 8785 canonical bytes of
  the SPEC 2.2 envelope document rebuilt from the manifest payload - byte
  equal to hashing the published configs/rules artifact file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from hydra2.artifacts.digest import of_canonical
from hydra2.contracts.common import DigestText, make_seat
from hydra2.contracts.rules import manifest_to_payload
from hydra2.contracts.utility import RawOutcome, SettlementFact

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hydra2.contracts.rules import RulesManifest


__all__ = [
    "ENGINE_WIND_TO_TILE_TYPE",
    "TILE_TYPE_TO_ENGINE_WIND",
    "build_state_digest_document",
    "engine_scores_canonical_order",
    "furiten_of",
    "live_wall_remaining",
    "raw_outcome_from_final",
    "rules_identity_hash",
    "seat_winds_for_dealer",
    "settlement_facts_from_deltas",
    "state_digest",
]

#: Engine round/seat winds are ints 0..3 (E,S,W,N); canonical TileType 27..30.
ENGINE_WIND_TO_TILE_TYPE = {0: 27, 1: 28, 2: 29, 3: 30}
TILE_TYPE_TO_ENGINE_WIND = {v: k for k, v in ENGINE_WIND_TO_TILE_TYPE.items()}


def rules_identity_hash(manifest: RulesManifest) -> DigestText:
    """D-WP03A-4: envelope-canonical sha256 of the manifest payload."""
    envelope = {
        "artifact_type": "hydra2.rules_manifest",
        "compatibility": "exact",
        "schema_version": "1.0.0",
        "payload": manifest_to_payload(manifest),
    }
    return of_canonical(envelope)


def seat_winds_for_dealer(dealer: int) -> tuple[int, int, int, int]:
    """Seat winds permute East/South/West/North aligned by seat from dealer."""
    return cast(
        "tuple[int, int, int, int]",
        tuple(27 + ((dealer + offset) % 4) for offset in range(4)),
    )


def engine_scores_canonical_order(engine: Any, permutation: Sequence[int]) -> list[int]:
    """Engine scores reordered into canonical-seat order."""
    raw: list[int] = [int(cast("Any", s)) for s in cast("Any", engine.scores())]
    return [raw[permutation[seat]] for seat in range(4)]


def live_wall_remaining(engine: Any) -> int:
    """Drawable live-wall tiles left; the engine keeps a 14-tile dead wall."""
    wall: Any = engine.wall
    return max(0, len(cast("Any", wall)) - 14)


def furiten_of(engine: Any, engine_pid: int) -> str:
    """D-WP03A-3 literal mapping (see module docstring)."""
    riichi_declared: Any = engine.riichi_declared
    if bool(cast("Any", riichi_declared[engine_pid])):
        return "riichi"
    missed: Any = engine.missed_agari_doujun
    if bool(cast("Any", missed[engine_pid])):
        return "temporary"
    return "none"


def raw_outcome_from_final(
    *,
    final_scores: Sequence[int],
    starting_scores: Sequence[int],
    settlements: Sequence[SettlementFact],
    rules_id: str,
    rules_hash: DigestText,
) -> RawOutcome:
    """Terminal RawOutcome with ranks resolved through the rules contract.

    ``point_deltas`` is the net movement from the game's starting points;
    per-hand movements live in ``settlements`` (SPEC 5.2).
    """
    from hydra2.contracts.rules import resolve_final_ranks

    scores = cast("tuple[int, int, int, int]", tuple(s for s in final_scores))
    start = tuple(s for s in starting_scores)
    return RawOutcome(
        final_scores=scores,
        ranks=resolve_final_ranks(scores),
        point_deltas=cast(
            "tuple[int, int, int, int]", tuple(scores[i] - start[i] for i in range(4))
        ),
        settlements=tuple(settlements),
        rules_id=rules_id,
        rules_hash=rules_hash,
    )


def settlement_facts_from_deltas(
    *,
    kind: str,
    deltas: Sequence[int],
    payer_seat: int | None,
    winner_seats: Sequence[int],
) -> tuple[SettlementFact, ...]:
    """Atomic settlement movement of one hora/ryukyoku event.

    ``deltas`` are the engine's authoritative four-seat movements (they
    already include riichi-stick transfers and honba). One fact per terminal
    event keeps identities minimal; winners/payer name the point flow.
    """
    quad = cast("tuple[int, int, int, int]", tuple(d for d in deltas))
    recipients = tuple(make_seat(w) for w in winner_seats)
    if len(recipients) == 0:
        # Nobody gained points (e.g. all-payers exhaustive draw): the
        # settlement contract requires at least one recipient, and a fact
        # with none would be unrepresentable.
        return ()
    return (
        SettlementFact(
            kind=kind,
            from_seat=None if payer_seat is None else make_seat(payer_seat),
            to_seats=recipients,
            point_deltas=quad,
            detail={"source": "riichienv.mjai_deltas"},
        ),
    )


def build_state_digest_document(
    engine: Any, *, hand_index: int, permutation: Sequence[int]
) -> dict[str, Any]:
    """Complete deterministic state document feeding ``state_digest``."""
    melds: list[dict[str, Any]] = []
    for pid in range(4):
        melds_pid: Any = cast("Any", engine.melds[pid])
        for meld_any in cast("Any", melds_pid):
            meld: Any = meld_any
            meld_type_val: Any = meld.meld_type
            tiles_val: Any = meld.tiles
            opened_val: Any = meld.opened
            from_who_val: Any = meld.from_who
            tile_list: list[int] = sorted(int(cast("Any", t)) for t in cast("Any", tiles_val))
            from_who_idx: int = int(cast("Any", from_who_val))
            from_who_mapped: int | None = (
                None if from_who_idx < 0 else permutation[from_who_idx]
            )
            melds.append(
                {
                    "owner": permutation[pid],
                    "meld_type": int(cast("Any", meld_type_val)),
                    "tiles": tile_list,
                    "opened": bool(cast("Any", opened_val)),
                    "from_who": from_who_mapped,
                }
            )
    hands: dict[str, list[int]] = {
        str(permutation[pid]): sorted(int(cast("Any", t)) for t in cast("Any", engine.hands[pid]))
        for pid in range(4)
    }
    discards: dict[str, list[int]] = {
        str(permutation[pid]): [int(cast("Any", t)) for t in cast("Any", engine.discards[pid])]
        for pid in range(4)
    }
    current: Any = engine.current_player
    drawn: Any = engine.drawn_tile
    kyoku_idx_val: Any = engine.kyoku_idx
    phase_val: Any = engine.phase
    wall_val: Any = engine.wall
    dora_val: Any = engine.dora_indicators
    honba_val: Any = engine.honba
    riichi_sticks_val: Any = engine.riichi_sticks
    riichi_declared_val: Any = engine.riichi_declared
    turn_count_val: Any = engine.turn_count
    rinshan_val: Any = engine.rinshan_draw_count
    is_done_val: Any = engine.is_done
    current_idx: int = int(cast("Any", current))
    current_player_field: int | None = (
        None if current_idx < 0 else permutation[current_idx]
    )
    drawn_tile_field: int | None = None if drawn is None else int(cast("Any", drawn))
    return {
        "hand_index": hand_index,
        "kyoku_idx": int(cast("Any", kyoku_idx_val)),
        "phase": int(cast("Any", phase_val)),
        "current_player": current_player_field,
        "hands": hands,
        "wall": [int(cast("Any", t)) for t in cast("Any", wall_val)],
        "discards": discards,
        "melds": sorted(melds, key=lambda m: (m["owner"], m["meld_type"], m["tiles"])),
        "dora_indicators": [int(cast("Any", t)) for t in cast("Any", dora_val)],
        "scores": engine_scores_canonical_order(engine, permutation),
        "honba": int(cast("Any", honba_val)),
        "riichi_sticks": int(cast("Any", riichi_sticks_val)),
        "riichi_declared": [bool(cast("Any", v)) for v in cast("Any", riichi_declared_val)],
        "drawn_tile": drawn_tile_field,
        "turn_count": int(cast("Any", turn_count_val)),
        "rinshan_draw_count": int(cast("Any", rinshan_val)),
        "is_done": bool(cast("Any", is_done_val)),
    }


def state_digest(engine: Any, *, hand_index: int, permutation: Sequence[int]) -> DigestText:
    """sha256 over canonical bytes of the full state document."""
    document: dict[str, Any] = build_state_digest_document(
        engine, hand_index=hand_index, permutation=permutation
    )
    return of_canonical(document)
