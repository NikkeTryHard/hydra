"""Bridge W3-A: full-doc assembly, ippatsu derivation, projection rejection.

Zero-synthesis gate: the Rust JSON handoff emits 13-key string projections
(bridge INPUT); :mod:`hydra2.training.rust_observations` expands them into
35-field engine full docs. Ippatsu windows derive from the framed log exactly
like furiten derives from S6 answers (adapter D-WP03A-9 intended semantics:
set at reach_accepted, cleared by any meld interrupt and by the declarer's
own next discard, reset every hand) — never stubbed. Unexpanded projections
reaching the encoder ingress fail closed with a named reason.
"""

from __future__ import annotations

import pytest

from hydra2.contracts.common import ContractError
from hydra2.models.schema import BASELINE_ACTION_COUNT
from hydra2.training import rust_observations as ro
from hydra2.training.dataset import _actor_observation_from_json_dict

pytestmark = [pytest.mark.contract_package("WP-14")]


def test_full_doc_envelope_version_pinned() -> None:
    """The full-doc marker mirrors the Rust bridge constant (bump together)."""
    assert ro.BRIDGE_FULL_DOC_ENVELOPE_VERSION == "full-doc-v1"


class TestIppatsuWalk:
    def test_open_and_declarer_discard_close(self) -> None:
        walk = ro._GameWalk("g-ippatsu")
        assert walk.ippatsu_snapshot() == (False, False, False, False)
        walk.open_ippatsu(1, where="reach_accepted")
        assert walk.ippatsu_snapshot() == (False, True, False, False)
        # Other seats' discards leave the window open; only the declarer's
        # own next discard closes it (adapter _on_dahai order).
        walk.close_ippatsu_seat(1, where="dahai@3")
        assert walk.ippatsu_snapshot() == (False, False, False, False)

    def test_meld_interrupt_clears_every_window(self) -> None:
        walk = ro._GameWalk("g-ippatsu")
        walk.open_ippatsu(0, where="reach_accepted")
        walk.open_ippatsu(2, where="reach_accepted")
        assert walk.ippatsu_snapshot() == (True, False, True, False)
        walk.close_ippatsu_all()
        assert walk.ippatsu_snapshot() == (False, False, False, False)

    def test_new_hand_resets_windows(self) -> None:
        walk = ro._GameWalk("g-ippatsu")
        walk.open_ippatsu(3, where="reach_accepted")
        walk.install(
            {
                "tehais": [[], [], [], []],
                "oya": 0,
                "bakaze": "E",
                "honba": 0,
                "scores": [25000, 25000, 25000, 25000],
                "kyotaku": 0,
                "kyoku": 2,
            }
        )
        assert walk.ippatsu_snapshot() == (False, False, False, False)

    @pytest.mark.parametrize("bad", [7, -1, True, "0", None, 2.0])
    def test_malformed_transition_fails_closed(self, bad: object) -> None:
        walk = ro._GameWalk("g-ippatsu")
        with pytest.raises(ContractError, match="ippatsu-active-unrecoverable"):
            walk.open_ippatsu(bad, where="reach_accepted")
        with pytest.raises(ContractError, match="ippatsu-active-unrecoverable"):
            walk.close_ippatsu_seat(bad, where="dahai@0")


def _snapshot_observation(*, window_seat: int | None) -> object:
    """Drive the real ``_Assembler._snapshot`` then build one observation."""
    from hydra2.engines.riichienv.state import seat_winds_for_dealer

    asm = ro._Assembler.__new__(ro._Assembler)
    asm.walk = ro._GameWalk("g-snap")
    asm.walk.install(
        {
            "tehais": [[], [], [], []],
            "oya": 0,
            "bakaze": "E",
            "honba": 0,
            "scores": [25000, 25000, 25000, 25000],
            "kyotaku": 0,
            "kyoku": 1,
        }
    )
    asm.builder = ro._builder_for("g-snap")
    asm.seat_winds_for_dealer = seat_winds_for_dealer
    if window_seat is not None:
        asm.walk.open_ippatsu(window_seat, where="reach_accepted")
    asm._snapshot(seat=0, phase="draw_decision", turn_actor=0, wall=70)
    asm.builder.set_concealed_hand(0, ())
    asm.builder.set_actor_state(0, furiten="none", can_tsumo=True, can_riichi=False)
    return asm.builder.build(actor=0, legal_mask=(True,) + (False,) * (BASELINE_ACTION_COUNT - 1))


def test_snapshot_carries_open_window() -> None:
    obs: object = _snapshot_observation(window_seat=0)
    assert obs.ippatsu_active == (True, False, False, False)  # type: ignore[attr-defined]


def test_snapshot_defaults_to_closed_windows() -> None:
    obs: object = _snapshot_observation(window_seat=None)
    assert obs.ippatsu_active == (False, False, False, False)  # type: ignore[attr-defined]


def _projection_doc(tag: str) -> dict[str, object]:
    return {
        "projection": tag,
        "game_id": "g-proj",
        "decision_id": "g-proj:d0000",
        "seat": 0,
        "phase": "draw_decision",
        "turn_actor": 0,
        "concealed_hand": [],
        "drawn_tile": None,
        "dora_indicators": [None, None, None, None, None],
        "live_wall_tiles_remaining": 70,
        "history_kinds": [],
        "legal_mask": [],
        "chosen": None,
    }


@pytest.mark.parametrize("tag", ["wall-less-v1", "walled-v1"])
def test_unexpanded_projection_rejected_fail_closed(tag: str) -> None:
    """Projections are bridge input, never encoder input (named reason)."""
    with pytest.raises(ContractError, match="unexpanded-projection-row"):
        _actor_observation_from_json_dict(_projection_doc(tag), decision_id="g-proj:d0000")


def test_full_doc_without_projection_tag_accepted() -> None:
    """A 35-field engine document parses (versioned digests ride along)."""
    from hydra2.contracts.observation import make_actor_observation

    obs = make_actor_observation(
        game_id="g-full",
        decision_id="g-full:d0000",
        sequence=1,
        actor=0,
        rules_id="tenhou_4p_hanchan_v1",
        rules_hash="sha256:" + "ab" * 32,
        action_table_hash="sha256:" + "ac" * 32,
        event_schema_hash="sha256:" + "ad" * 32,
        observation_schema_hash="sha256:" + "ae" * 32,
        packet_boundary_hash="sha256:" + "af" * 32,
        round_index=0,
        round_wind=27,
        hand_number=1,
        seat_winds=(27, 28, 29, 30),
        honba=0,
        riichi_sticks=0,
        dealer=0,
        scores=(25000, 25000, 25000, 25000),
        turn_actor=0,
        phase="draw_decision",
        live_wall_tiles_remaining=70,
        kan_count=0,
        ippatsu_active=(True, False, False, False),
        actor_furiten="none",
        actor_can_tsumo=True,
        actor_can_riichi=False,
        pending_declaration_discard=None,
        concealed_hand=(),
        own_drawn_tile=None,
        visible_discards=((), (), (), ()),
        visible_melds=((), (), (), ()),
        riichi_states=("none", "none", "none", "none"),
        dora_indicators=(-1, -1, -1, -1, -1),
        visible_history=(),
        legal_mask=(True,) + (False,) * (BASELINE_ACTION_COUNT - 1),
    )
    rebuilt = _actor_observation_from_json_dict(obs.to_json(), decision_id="g-full:d0000")
    assert rebuilt.ippatsu_active == (True, False, False, False)
    assert str(rebuilt.observation_hash) == str(obs.observation_hash)
