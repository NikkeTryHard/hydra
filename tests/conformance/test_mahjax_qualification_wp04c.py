"""WP-04C MahJax qualification — differential over declared rule intersection.

BUILD checklist 452-457:
1. Differential over DECLARED_INTERSECTION (enumerated explicitly)
2. Eager/JIT/vmap determinism (CPU documented)
3. Cases: fifth dora (WP04A-01), chankan (02), kuikae (03), shanten parity
4. GPU soak probe else CPU soak + GPU blocked-xfail
5. First-counterexample persistence ($ROOT/counterexamples/WP-04C/)
6. Token bound to full env tuple ONLY after zero mismatch ($ROOT/tokens/ + round-trip)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hydra2.config import artifact_root
from hydra2.engines.mahjax.capture import capture_mahjax_tuple
from hydra2.engines.mahjax.differential import (
    CONVERGENT_DORA_INDICATOR_TYPES,
    DECLARED_INTERSECTION,
    EXCLUDED_DIMENSIONS,
    SCENARIO_REGISTRY,
    DifferentialResult,
    build_seeded_round_state,
    cpu_soak,
    execution_mode_sweep,
    gpu_soak_probe,
    make_single_round_env,
    run_differential,
    wall_to_mahjax_deck,
)
from hydra2.engines.mahjax.quarantine import ADAPTER_VERSION, OBSERVATION_MODE
from hydra2.engines.mahjax.shell import MahJaxQuarantineShell

pytestmark = pytest.mark.contract_package("WP-04C")


def test_declared_intersection_enumerated() -> None:
    assert len(DECLARED_INTERSECTION) == 14
    assert "deal_order" in DECLARED_INTERSECTION
    assert "kan_dora_reveal_policy" in DECLARED_INTERSECTION
    assert "chankan_window" in DECLARED_INTERSECTION
    assert "kuikae_policy_forbidden" in DECLARED_INTERSECTION
    assert "shanten_parity" in DECLARED_INTERSECTION
    assert len(EXCLUDED_DIMENSIONS) == 6
    excluded_names = {name for name, _ in EXCLUDED_DIMENSIONS}
    assert "red_dora" in excluded_names
    assert "dora_successor_divergent_types" in excluded_names
    assert 17 not in CONVERGENT_DORA_INDICATOR_TYPES
    assert 18 not in CONVERGENT_DORA_INDICATOR_TYPES
    assert all(
        t not in {17, 18, 27, 28, 29, 30, 31, 32, 33} for t in CONVERGENT_DORA_INDICATOR_TYPES
    )
    for scenario in SCENARIO_REGISTRY:
        for field_name in scenario.rule_fields:
            assert field_name in DECLARED_INTERSECTION, (
                f"{scenario.case_id} field {field_name} not in DECLARED_INTERSECTION"
            )


def test_scenario_registry_has_four_cases() -> None:
    assert len(SCENARIO_REGISTRY) == 4
    ids = {s.case_id for s in SCENARIO_REGISTRY}
    assert "WP04C-01-fifth-dora" in ids
    assert "WP04C-02-chankan" in ids
    assert "WP04C-03-kuikae" in ids
    assert "WP04C-04-shanten-parity" in ids
    for scenario in SCENARIO_REGISTRY:
        from hydra2.conformance.walls import build_wall

        wall = build_wall(
            hands=scenario.hands, live_draws=scenario.live_draws, dead_wall=scenario.dead_wall
        )
        assert len(wall) == 136
        deck = wall_to_mahjax_deck(wall)
        assert len(deck) == 136
        assert max(deck.count(t) for t in range(34)) <= 4
        for idx in (131, 129, 127, 125, 123):
            if idx in scenario.dead_wall:
                t = scenario.dead_wall[idx] // 4
                assert t in CONVERGENT_DORA_INDICATOR_TYPES, (
                    f"{scenario.case_id} indicator {idx} type {t} not convergent"
                )


def test_wall_translation_and_surgery_deterministic() -> None:
    scenario = SCENARIO_REGISTRY[0]
    from hydra2.conformance.walls import build_wall

    wall = build_wall(
        hands=scenario.hands, live_draws=scenario.live_draws, dead_wall=scenario.dead_wall
    )
    deck = wall_to_mahjax_deck(wall)
    env = make_single_round_env()
    s1 = build_seeded_round_state(env, deck, dealer=0)
    s2 = build_seeded_round_state(env, deck, dealer=0)
    assert list(s1.round_state.deck.tolist()) == list(s2.round_state.deck.tolist())
    assert list(s1.round_state.dora_indicators.tolist()) == list(
        s2.round_state.dora_indicators.tolist()
    )
    assert int(s1.current_player) == int(s2.current_player) == 0
    assert int(s1.round_state.dealer) == 0
    assert deck[83] == wall[52] // 4
    assert deck[82] == wall[53] // 4
    assert deck[9] == wall[131] // 4


def test_execution_mode_sweep_deterministic_cpu_documented() -> None:
    result = execution_mode_sweep()
    assert "deterministic" in result
    assert "eager_digest" in result
    assert "jit_digest" in result
    assert result["backend"] in ("cpu", "gpu")
    assert result["deterministic"] is True
    assert result["eager_digest"] == result["jit_digest"]


def test_gpu_soak_probe_blocked_on_cpu() -> None:
    result = gpu_soak_probe()
    if result["gpu_available"]:
        assert result["status"] == "passed"
        assert result["backend"] == "gpu"
    else:
        assert result["backend"] == "cpu"
        assert result["gpu_available"] is False
        assert result["status"] == "blocked"
        assert "CPU-only" in result["reason"]


@pytest.mark.xfail(
    strict=False,
    reason="GPU jaxlib not installed at pin 0.11.1 - GPU soak blocked with evidence (CPU-only)",
)
def test_gpu_soak_requires_cuda_device() -> None:
    result = gpu_soak_probe()
    assert result["gpu_available"] is True
    assert result["status"] == "passed"


def test_cpu_soak_bounded_and_deterministic() -> None:
    result = cpu_soak(steps=20)
    assert result["status"] == "passed"
    assert result["steps"] > 0
    assert result["backend"] == "cpu"
    assert result["steps"] == 20 * len(SCENARIO_REGISTRY)


def test_differential_zero_mismatch_and_token_issued() -> None:
    root = artifact_root()
    result: DifferentialResult = run_differential(artifact_root=root)
    assert result.verdict == "passed"
    assert result.total_cases == 4
    assert result.passed_cases == 4
    assert result.failed_cases == 0
    assert len(result.mismatches) == 0
    assert result.first_counterexample_path is None
    assert result.token_path is not None
    assert result.token_digest is not None
    assert result.token_digest.startswith("sha256:")
    token_path = Path(result.token_path)
    assert token_path.is_file()
    payload = json.loads(token_path.read_text())
    assert payload["artifact_type"] == "hydra2.wp04c_qualification_token"
    assert payload["identity_digest"] == result.token_digest
    assert result.env_tuple_digest.startswith("sha256:")
    capture = capture_mahjax_tuple()
    assert (
        str(capture.digest) == result.env_tuple_digest
        or payload["environment_fragment"] == capture.to_fragment()
    )
    from hydra2.contracts.common import make_digest_text
    from hydra2.engines.mahjax.quarantine import fabricate_test_only_token

    shell = MahJaxQuarantineShell()
    fresh_token = fabricate_test_only_token(
        capture, rules_id=make_digest_text(payload["token"]["rules_id"])
    )
    digest = shell.qualify(fresh_token, rules_id=make_digest_text(payload["token"]["rules_id"]))
    assert str(digest) == str(fresh_token.identity_digest)
    tampered = fabricate_test_only_token(
        capture, rules_id=make_digest_text(payload["token"]["rules_id"])
    )
    object.__setattr__(tampered, "jax_version", "0.0.0-tampered")
    with pytest.raises(Exception):  # noqa: B017
        shell2 = MahJaxQuarantineShell()
        shell2.qualify(tampered, rules_id=make_digest_text(payload["token"]["rules_id"]))


def test_token_not_issued_without_full_env_binding() -> None:
    root = artifact_root()
    token_path = root / "tokens" / "WP-04C" / "mahjax-qualification-token.json"
    assert token_path.is_file(), "token should exist after passed differential"
    payload = json.loads(token_path.read_text())
    for key in (
        "backend_platform",
        "devices",
        "jax_version",
        "jaxlib_version",
        "mahjax_commit_id",
        "pixi_lock_sha256",
        "python_implementation",
        "python_version",
        "xla_flags",
    ):
        assert key in json.dumps(payload), f"missing {key} in token binding"


def test_first_counterexample_persistence_path() -> None:
    root = artifact_root()
    counter_dir = root / "counterexamples" / "WP-04C"
    if counter_dir.exists():
        for p in counter_dir.glob("*.json"):
            data = json.loads(p.read_text())
            assert data["artifact_type"] == "hydra2.wp04c_counterexample"
            assert "case_id" in data
            assert "failure" in data


def test_observation_mode_and_adapter_version_bound() -> None:
    root = artifact_root()
    token_path = root / "tokens" / "WP-04C" / "mahjax-qualification-token.json"
    payload = json.loads(token_path.read_text())
    token = payload["token"]
    assert token["adapter_version"] == str(ADAPTER_VERSION)
    assert token["observation_mode"] == OBSERVATION_MODE


def test_execution_mode_sweep_covers_all_scenarios() -> None:
    for scenario in SCENARIO_REGISTRY:
        result = execution_mode_sweep(scenario)
        assert result["deterministic"] is True
        assert result["scenario"] == scenario.case_id


def test_shanten_parity_and_dora_within_scenarios() -> None:
    by_id = {s.case_id: s for s in SCENARIO_REGISTRY}
    assert "kan_dora_reveal_policy" in by_id["WP04C-01-fifth-dora"].rule_fields
    assert "chankan_window" in by_id["WP04C-02-chankan"].rule_fields
    assert "kuikae_policy_forbidden" in by_id["WP04C-03-kuikae"].rule_fields
    assert "shanten_parity" in by_id["WP04C-04-shanten-parity"].rule_fields
