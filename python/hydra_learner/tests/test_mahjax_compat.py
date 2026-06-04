from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from hydra_learner.mahjax import compat
from hydra_learner.mahjax import contract as mahjax_contract


def test_action_space_constants_match_contract() -> None:
    assert compat.HYDRA_ACTION_SPACE == 46
    assert compat.MAHJAX_RED_ACTION_SPACE == 87
    assert compat.HYDRA_AKA_5M == 34
    assert compat.HYDRA_AKA_5P == 35
    assert compat.HYDRA_AKA_5S == 36
    assert compat.HYDRA_RIICHI == 37
    assert compat.HYDRA_CHI_LEFT == 38
    assert compat.HYDRA_CHI_MID == 39
    assert compat.HYDRA_CHI_RIGHT == 40
    assert compat.HYDRA_PON == 41
    assert compat.HYDRA_KAN == 42
    assert compat.HYDRA_AGARI == 43
    assert compat.HYDRA_RYUUKYOKU == 44
    assert compat.HYDRA_PASS == 45


def test_mahjax_gpu_lane_stays_opt_in_until_default_readiness_blockers_clear() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text())
    pixi = pyproject["tool"]["pixi"]
    environments = pixi["environments"]

    default_features = set(environments["default"]["features"])
    assert "mahjax" not in default_features
    assert "mahjax-train" not in default_features
    assert environments["mahjax"]["no-default-feature"] is True
    assert environments["mahjax-train"]["no-default-feature"] is True

    default_deps = pixi.get("pypi-dependencies", {})
    assert "mahjax" not in default_deps
    assert "jax" not in default_deps

    mahjax_feature_deps = pixi["feature"]["mahjax"]["pypi-dependencies"]
    assert mahjax_feature_deps["mahjax"] == "==0.1.2"
    assert "jax" in mahjax_feature_deps

    mahjax_train_feature_deps = pixi["feature"]["mahjax-train"]["pypi-dependencies"]
    assert mahjax_train_feature_deps["mahjax"] == "==0.1.2"
    assert "jax" in mahjax_train_feature_deps


def test_mahjax_discard_actions_project_to_same_hydra_ids() -> None:
    for action in range(37):
        assert compat.mahjax_action_to_hydra(action) == action

    assert compat.hydra_discard_base_tile(compat.HYDRA_AKA_5M) == 4
    assert compat.hydra_discard_base_tile(compat.HYDRA_AKA_5P) == 13
    assert compat.hydra_discard_base_tile(compat.HYDRA_AKA_5S) == 22


def test_mahjax_collapsed_actions_project_to_hydra_ids() -> None:
    for action in range(compat.MAHJAX_SELF_KAN_START, compat.MAHJAX_SELF_KAN_END + 1):
        assert compat.mahjax_action_to_hydra(action) == compat.HYDRA_KAN

    assert compat.mahjax_action_to_hydra(compat.MAHJAX_RIICHI) == compat.HYDRA_RIICHI
    assert compat.mahjax_action_to_hydra(compat.MAHJAX_TSUMO) == compat.HYDRA_AGARI
    assert compat.mahjax_action_to_hydra(compat.MAHJAX_RON) == compat.HYDRA_AGARI
    assert compat.mahjax_action_to_hydra(compat.MAHJAX_PON) == compat.HYDRA_PON
    assert compat.mahjax_action_to_hydra(compat.MAHJAX_PON_RED) == compat.HYDRA_PON
    assert compat.mahjax_action_to_hydra(compat.MAHJAX_OPEN_KAN) == compat.HYDRA_KAN
    assert compat.mahjax_action_to_hydra(compat.MAHJAX_CHI_LEFT) == compat.HYDRA_CHI_LEFT
    assert compat.mahjax_action_to_hydra(compat.MAHJAX_CHI_LEFT_RED) == compat.HYDRA_CHI_LEFT
    assert compat.mahjax_action_to_hydra(compat.MAHJAX_CHI_MID) == compat.HYDRA_CHI_MID
    assert compat.mahjax_action_to_hydra(compat.MAHJAX_CHI_MID_RED) == compat.HYDRA_CHI_MID
    assert compat.mahjax_action_to_hydra(compat.MAHJAX_CHI_RIGHT) == compat.HYDRA_CHI_RIGHT
    assert compat.mahjax_action_to_hydra(compat.MAHJAX_CHI_RIGHT_RED) == compat.HYDRA_CHI_RIGHT
    assert compat.mahjax_action_to_hydra(compat.MAHJAX_PASS) == compat.HYDRA_PASS
    assert compat.mahjax_action_to_hydra(compat.MAHJAX_KYUUSHU) == compat.HYDRA_RYUUKYOKU
    assert compat.mahjax_action_to_hydra(compat.MAHJAX_DUMMY) is None


def test_tsumogiri_requires_last_draw_and_projects_red_aware() -> None:
    with pytest.raises(ValueError, match="TSUMOGIRI requires last_draw"):
        compat.mahjax_action_to_hydra(compat.MAHJAX_TSUMOGIRI)

    assert compat.mahjax_action_to_hydra(compat.MAHJAX_TSUMOGIRI, last_draw=4) == 4
    assert compat.mahjax_action_to_hydra(compat.MAHJAX_TSUMOGIRI, last_draw=34) == compat.HYDRA_AKA_5M


def test_mask_projection_or_reduces_collapsed_groups() -> None:
    mask = [False] * compat.MAHJAX_RED_ACTION_SPACE
    mask[0] = True
    mask[compat.MAHJAX_TSUMOGIRI] = True
    mask[compat.MAHJAX_TSUMO] = True
    mask[compat.MAHJAX_RON] = True
    mask[compat.MAHJAX_PON_RED] = True
    mask[compat.MAHJAX_CHI_RIGHT_RED] = True
    mask[compat.MAHJAX_OPEN_KAN] = True
    mask[compat.MAHJAX_DUMMY] = True

    hydra = compat.mahjax_mask_to_hydra(mask, last_draw=compat.HYDRA_AKA_5P)

    assert len(hydra) == compat.HYDRA_ACTION_SPACE
    assert all(isinstance(value, bool) for value in hydra)
    assert hydra[0]
    assert hydra[compat.HYDRA_AKA_5P]
    assert hydra[compat.HYDRA_AGARI]
    assert hydra[compat.HYDRA_PON]
    assert hydra[compat.HYDRA_CHI_RIGHT]
    assert hydra[compat.HYDRA_KAN]
    assert not any(hydra[action] for action in (compat.HYDRA_RIICHI, compat.HYDRA_PASS, compat.HYDRA_RYUUKYOKU))


def test_contract_jax_mask_projection_matches_cpu_and_ignores_dummy() -> None:
    jax = pytest.importorskip("jax")
    mask = [False] * compat.MAHJAX_RED_ACTION_SPACE
    expected_actions = {
        0,
        compat.HYDRA_AKA_5P,
        compat.HYDRA_AGARI,
        compat.HYDRA_PON,
        compat.HYDRA_CHI_RIGHT,
        compat.HYDRA_KAN,
        compat.HYDRA_PASS,
        compat.HYDRA_RYUUKYOKU,
    }
    for action in (
        0,
        compat.MAHJAX_TSUMOGIRI,
        compat.MAHJAX_TSUMO,
        compat.MAHJAX_RON,
        compat.MAHJAX_PON_RED,
        compat.MAHJAX_CHI_RIGHT_RED,
        compat.MAHJAX_OPEN_KAN,
        compat.MAHJAX_PASS,
        compat.MAHJAX_KYUUSHU,
        compat.MAHJAX_DUMMY,
    ):
        mask[action] = True

    actual = mahjax_contract.project_mask_jax(jax.numpy.asarray(mask), compat.HYDRA_AKA_5P).tolist()

    assert actual == compat.mahjax_mask_to_hydra(mask, last_draw=compat.HYDRA_AKA_5P)
    assert {index for index, legal in enumerate(actual) if legal} == expected_actions
    assert len(actual) == compat.HYDRA_ACTION_SPACE


def test_contract_round_trips_projected_legal_actions_to_mahjax() -> None:
    mask = [False] * compat.MAHJAX_RED_ACTION_SPACE
    for action in (
        0,
        compat.MAHJAX_TSUMOGIRI,
        compat.MAHJAX_RIICHI,
        compat.MAHJAX_PON,
        compat.MAHJAX_PON_RED,
        compat.MAHJAX_OPEN_KAN,
        compat.MAHJAX_TSUMO,
        compat.MAHJAX_RON,
        compat.MAHJAX_PASS,
        compat.MAHJAX_KYUUSHU,
    ):
        mask[action] = True

    hydra_mask = compat.mahjax_mask_to_hydra(mask, last_draw=0)
    for hydra_action, legal in enumerate(hydra_mask):
        if not legal:
            continue
        mahjax_action = compat.hydra_action_to_mahjax(
            hydra_action,
            legal_mask=mask,
            last_draw=0,
            response_phase=hydra_action == compat.HYDRA_AGARI,
        )
        assert mask[mahjax_action]
        assert compat.mahjax_action_to_hydra(mahjax_action, last_draw=0) == hydra_action


def test_contract_inactive_action_is_dummy_control_action() -> None:
    jax = pytest.importorskip("jax")

    assert mahjax_contract.inactive_action() == compat.MAHJAX_DUMMY
    assert int(mahjax_contract.inactive_action_jax(jax.numpy).tolist()) == compat.MAHJAX_DUMMY


def test_reverse_discard_prefers_tsumogiri_when_it_matches_last_draw() -> None:
    mask = [False] * compat.MAHJAX_RED_ACTION_SPACE
    mask[compat.MAHJAX_TSUMOGIRI] = True
    mask[compat.HYDRA_AKA_5S] = True

    assert (
        compat.hydra_action_to_mahjax(compat.HYDRA_AKA_5S, legal_mask=mask, last_draw=compat.HYDRA_AKA_5S)
        == compat.MAHJAX_TSUMOGIRI
    )

    mask[compat.MAHJAX_TSUMOGIRI] = False
    assert (
        compat.hydra_action_to_mahjax(compat.HYDRA_AKA_5S, legal_mask=mask, last_draw=compat.HYDRA_AKA_5S)
        == compat.HYDRA_AKA_5S
    )


def test_reverse_call_variants_use_legal_mask_and_red_preference() -> None:
    mask = [False] * compat.MAHJAX_RED_ACTION_SPACE
    mask[compat.MAHJAX_PON] = True
    mask[compat.MAHJAX_PON_RED] = True
    mask[compat.MAHJAX_CHI_LEFT] = True
    mask[compat.MAHJAX_CHI_LEFT_RED] = True

    assert compat.hydra_action_to_mahjax(compat.HYDRA_PON, legal_mask=mask) == compat.MAHJAX_PON
    assert (
        compat.hydra_action_to_mahjax(compat.HYDRA_PON, legal_mask=mask, prefer_red_call=True) == compat.MAHJAX_PON_RED
    )
    assert compat.hydra_action_to_mahjax(compat.HYDRA_CHI_LEFT, legal_mask=mask) == compat.MAHJAX_CHI_LEFT
    assert (
        compat.hydra_action_to_mahjax(compat.HYDRA_CHI_LEFT, legal_mask=mask, prefer_red_call=True)
        == compat.MAHJAX_CHI_LEFT_RED
    )


def test_reverse_kan_agari_and_control_actions() -> None:
    mask = [False] * compat.MAHJAX_RED_ACTION_SPACE
    mask[compat.MAHJAX_SELF_KAN_START + 13] = True
    mask[compat.MAHJAX_RON] = True
    mask[compat.MAHJAX_KYUUSHU] = True
    mask[compat.MAHJAX_PASS] = True

    assert (
        compat.hydra_action_to_mahjax(compat.HYDRA_KAN, legal_mask=mask, kan_tile_type=13)
        == compat.MAHJAX_SELF_KAN_START + 13
    )
    assert compat.hydra_action_to_mahjax(compat.HYDRA_AGARI, legal_mask=mask, response_phase=True) == compat.MAHJAX_RON
    assert compat.hydra_action_to_mahjax(compat.HYDRA_RYUUKYOKU, legal_mask=mask) == compat.MAHJAX_KYUUSHU
    assert compat.hydra_action_to_mahjax(compat.HYDRA_PASS, legal_mask=mask) == compat.MAHJAX_PASS

    mask[compat.MAHJAX_OPEN_KAN] = True
    assert compat.hydra_action_to_mahjax(compat.HYDRA_KAN, legal_mask=mask, kan_tile_type=13) == compat.MAHJAX_OPEN_KAN


def test_invalid_ids_and_masks_fail_closed() -> None:
    with pytest.raises(ValueError, match="MahJAX action id out of range"):
        compat.mahjax_action_to_hydra(compat.MAHJAX_RED_ACTION_SPACE)
    with pytest.raises(ValueError, match="Hydra action id out of range"):
        compat.hydra_action_to_mahjax(compat.HYDRA_ACTION_SPACE)
    with pytest.raises(ValueError, match="legal mask width"):
        compat.mahjax_mask_to_hydra([False])
    with pytest.raises(ValueError, match="not legal"):
        compat.hydra_action_to_mahjax(compat.HYDRA_RIICHI, legal_mask=[False] * compat.MAHJAX_RED_ACTION_SPACE)
