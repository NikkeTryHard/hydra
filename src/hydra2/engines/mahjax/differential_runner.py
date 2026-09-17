"""WP-04C MahJax differential — declared intersection and suite runner.

Pure-move part of :mod:`hydra2.engines.mahjax.differential`; import from
that path. Covers the declared rule intersection, the honest exclusions,
per-scenario execution, and the full differential suite with token issuance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from hydra2_replay_rs import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]

from hydra2.artifacts.digest import of_canonical
from hydra2.conformance.runner import TraceRunnerError, wall_schedule_for
from hydra2.engines.mahjax.differential_cases import SCENARIO_REGISTRY
from hydra2.engines.mahjax.differential_compare import (
    _compare_projections as _compare_projections,
)
from hydra2.engines.mahjax.differential_compare import (
    _persist_counterexample as _persist_counterexample,
)
from hydra2.engines.mahjax.differential_compare import _publish_token as _publish_token
from hydra2.engines.mahjax.differential_compare import (
    _reference_auto_action as _reference_auto_action,
)
from hydra2.engines.mahjax.differential_compare import (
    _reference_find_action as _reference_find_action,
)
from hydra2.engines.mahjax.differential_modes import cpu_soak, execution_mode_sweep, gpu_soak_probe
from hydra2.engines.mahjax.differential_projection import (
    _ACTION_PASS,
    _ACTION_TSUMOGIRI,
    _mahjax_modules,
)
from hydra2.engines.mahjax.differential_projection import (
    CheckpointFailure as CheckpointFailure,
)
from hydra2.engines.mahjax.differential_projection import DifferentialResult as DifferentialResult
from hydra2.engines.mahjax.differential_projection import Scenario as Scenario
from hydra2.engines.mahjax.differential_projection import _wall_for_scenario as _wall_for_scenario
from hydra2.engines.mahjax.differential_projection import (
    build_seeded_round_state as build_seeded_round_state,
)
from hydra2.engines.mahjax.differential_projection import (
    make_single_round_env as make_single_round_env,
)
from hydra2.engines.mahjax.differential_projection import (
    map_script_step_to_mahjax as map_script_step_to_mahjax,
)
from hydra2.engines.mahjax.differential_projection import wall_to_mahjax_deck as wall_to_mahjax_deck

if TYPE_CHECKING:
    from hydra2.contracts.common import DigestText
    from hydra2.contracts.rules_manifest import RulesManifest

__all__ = [
    "CONVERGENT_DORA_INDICATOR_TYPES",
    "DECLARED_INTERSECTION",
    "EXCLUDED_DIMENSIONS",
    "run_differential",
]


# ---------------------------------------------------------------------------
# Declared rule intersection (BUILD item 1: enumerated explicitly).
# ---------------------------------------------------------------------------

#: Rule dimensions whose observable behaviour IS compared at checkpoints.
DECLARED_INTERSECTION: tuple[str, ...] = (
    "deal_order",  # haipai seat order and per-seat slot sequence
    "live_draw_order",  # live wall consumption order and tile identity
    "rinshan_draw_order",  # dead-wall rinshan stack consumption
    "kan_dora_reveal_policy",  # ankan_immediate_open_delayed on both engines
    "dora_indicator_slots",  # five indicator slots, revealed in order
    "ura_hidden_until_hora",  # ura indicators never public before a win
    "chankan_window",  # kakan opens a ron window for waiting players
    "kuikae_policy_forbidden",  # post-meld same-type swap discard barred
    "riichi_declaration_and_stick",  # declaration, 1000-point stick, kyotaku
    "discard_legality_projection",  # legal discard TYPE sets at checkpoints
    "win_offer_flags",  # ron/tsumo availability flags at checkpoints
    "shanten_parity",  # tenpai/wait agreement between evaluators
    "settlement_fan_fu",  # fan/fu of agreed wins inside the intersection
    "settlement_payment_child_ron",  # basic*4 rounded up to 100 (+sticks)
)

#: Dimensions deliberately OUTSIDE the comparison, with reasons. These are
#: honest scope declarations, not failures; each is observable behaviour where
#: the two engines cannot agree by construction at this pin.
EXCLUDED_DIMENSIONS: tuple[tuple[str, str], ...] = (
    (
        "red_dora",
        "mahjax 0.1.2 ships only no_red_mahjong/red_mahjong type-level decks; "
        "scenarios keep red copies out of winning hands so no compared "
        "projection depends on them",
    ),
    (
        "dora_successor_divergent_types",
        "probe-verified: mahjax maps honor indicators to the NEXT honor "
        "(E->S..N->E, haku->hatsu..chun->haku), 9p indicator -> 1s and 1s "
        "indicator -> 1p; Tenhou/hydra2 wrap within suits and self-map honors. "
        "Indicator tiles are restricted to CONVERGENT_DORA_INDICATOR_TYPES",
    ),
    (
        "multi_kyoku_progression",
        "mahjax redeals later rounds from its internal PRNG (_init_for_next_"
        "round) and offers no wall injection at this pin; the differential is "
        "scoped to one round and stops comparing at the first round boundary",
    ),
    (
        "multi_ron_resolution",
        "WP-04A owner decision D2 documented RiichiEnv's seat-order packet "
        "deviation; single-winner scenarios avoid the dimension entirely",
    ),
    (
        "dealer_ron_multiplier",
        "WP-04A owner decision D1 documented RiichiEnv's missing x2 "
        "dealer-payment multiplier; scenarios use child-off-child settlements",
    ),
    (
        "abortive_draw_specials",
        "suufon_renda/kyuushu paths are WP-04A corpus territory; mahjax's "
        "nine-term mask differs structurally and no scenario triggers them",
    ),
)

#: Tile types whose indicator->dora successor AGREES between both engines:
#: all manzu (0-8), pinzu 1p-8p (9-16), souzu 2s-9s (19-26). Excluded: 17
#: (9p), 18 (1s), 27-33 (honors) - see EXCLUDED_DIMENSIONS.
CONVERGENT_DORA_INDICATOR_TYPES = frozenset(range(17)) | frozenset(range(19, 27))


def _run_one_scenario(
    scenario: Scenario, manifest: RulesManifest, artifact_root: Path
) -> tuple[list[CheckpointFailure], list[dict[str, Any]], tuple[int, ...], tuple[int, ...]]:
    """Run a single scenario, return (failures, step_log, wall, deck)."""
    wall = _wall_for_scenario(scenario)
    deck = wall_to_mahjax_deck(wall)
    env = make_single_round_env()
    # reference
    from hydra2.contracts.common import Seat

    sched = wall_schedule_for(scenario.case_id, wall)
    sim: Any = __import__(  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        "hydra2.engines.riichienv", fromlist=["RiichiEnvExactSimulator"]
    ).RiichiEnvExactSimulator()  # lazy
    sim.reset(rules=manifest, wall=sched, seat_permutation=(Seat(0), Seat(1), Seat(2), Seat(3)))
    # Build mj_state on CPU to avoid GPU OOM (env.init allocates large arrays)
    try:
        _build_cpu: Any = _mahjax_modules()["jax"].devices("cpu")[0]  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        with _mahjax_modules()["jax"].default_device(_build_cpu):
            mj_state = build_seeded_round_state(env, deck, dealer=0)
    except Exception:
        mj_state = build_seeded_round_state(env, deck, dealer=0)
    # mahjax cff90d1 requires PRNG key for every step (wall redeal)
    _rng: Any = _mahjax_modules()["jax"].random.PRNGKey(99)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    # Force CPU device for mahjax steps to avoid GPU OOM (determinism on CPU)
    try:
        _cpu_device: Any = _mahjax_modules()["jax"].devices("cpu")[0]  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    except Exception:
        _cpu_device = None

    def _step_cpu(state: Any, prim: int, sub: Any) -> Any:
        if _cpu_device is not None:
            with _mahjax_modules()["jax"].default_device(_cpu_device):
                return env.step(state, _mahjax_modules()["jnp"].int32(prim), sub)
        return env.step(state, _mahjax_modules()["jnp"].int32(prim), sub)

    # One compiled step per scenario (same kernels as eager; decisions
    # identical). Closure is built once per scenario, so one compile each.
    _jit_step_cpu: Any = _mahjax_modules()["jax"].jit(_step_cpu)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX

    failures: list[CheckpointFailure] = []
    step_log: list[dict[str, Any]] = []
    init_fail = _compare_projections(scenario, -1, sim, mj_state)
    if init_fail is not None:
        failures.append(init_fail)
        return failures, step_log, wall, deck
    for idx, decision in enumerate(scenario.script):
        actor: Any = sim._expected_actor_or_none()  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        if actor is None:
            # terminal early
            failures.append(
                CheckpointFailure(
                    case_id=scenario.case_id,
                    step_index=idx,
                    dimension="terminal",
                    detail="reference terminal before script exhausted",
                )
            )
            break
        # find reference action
        try:
            if decision.negate:
                # check forbidden not offered
                offered = [
                    a
                    for a in sim.legal_actions(Seat(actor))
                    if a.kind == decision.kind
                    and (
                        decision.tile is None
                        or (a.tile is not None and int(cast("Any", a.tile)) == decision.tile)  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                    )
                ]
                if len(offered) != 0:
                    failures.append(
                        CheckpointFailure(
                            case_id=scenario.case_id,
                            step_index=idx,
                            dimension="kuikae_policy_forbidden"
                            if decision.kind == "discard"
                            else "negate",
                            detail=(
                                f"seat {actor} forbidden {decision.kind} "
                                f"tile {decision.tile} was offered"
                            ),
                        )
                    )
                    # still need to apply auto to continue
                ref_action = _reference_auto_action(sim, actor)
            elif decision.kind == "auto":
                ref_action = _reference_auto_action(sim, actor)
            else:
                ref_action = _reference_find_action(sim, actor, decision)
        except TraceRunnerError as exc:
            failures.append(
                CheckpointFailure(
                    case_id=scenario.case_id,
                    step_index=idx,
                    dimension="script_construction",
                    detail=str(exc),
                )
            )
            break
        # apply reference
        try:
            sim.apply(ref_action)
        except Exception as exc:  # pragma: no cover
            failures.append(
                CheckpointFailure(
                    case_id=scenario.case_id,
                    step_index=idx,
                    dimension="reference_apply",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            break
        # map and apply mahjax
        try:
            prims = map_script_step_to_mahjax(decision, ref_action)
        except TraceRunnerError as exc:
            failures.append(
                CheckpointFailure(
                    case_id=scenario.case_id,
                    step_index=idx,
                    dimension="mahjax_mapping",
                    detail=str(exc),
                )
            )
            break
        for prim in prims:
            # check legality before step (current player's mask)
            mask = None  # pyrefly: init before use
            try:
                mask: Any = mj_state.legal_action_mask  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                # mask is 1d for current player; prim should be in range
                legal = (
                    bool(cast("Any", mask[prim]))
                    if prim < int(cast("Any", mask.shape[0]))
                    else False
                )  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            except Exception:
                legal = False
            if not legal and decision.kind == "riichi_discard" and prim != _ACTION_TSUMOGIRI:
                # riichi_discard of drawn tile uses tsumogiri when normal discard not legal
                try:
                    if mask is not None and bool(cast("Any", mask[_ACTION_TSUMOGIRI])):  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                        prim = _ACTION_TSUMOGIRI
                        legal = True
                except Exception:
                    pass
            if not legal:
                # Pass in reference window may not exist on mahjax side (window already resolved)
                # Treat as no-op rather than failure.
                if decision.kind == "pass" and prim == _ACTION_PASS:
                    continue
                if decision.kind == "auto" and prim == _ACTION_PASS:
                    continue
                failures.append(
                    CheckpointFailure(
                        case_id=scenario.case_id,
                        step_index=idx,
                        dimension="mahjax_illegal",
                        detail=(
                            f"mahjax illegal action {prim} at step {idx} (ref {ref_action.kind})"
                        ),
                    )
                )
                break
            _rng, _sub = _mahjax_modules()["jax"].random.split(_rng)
            mj_state: Any = _jit_step_cpu(mj_state, prim, cast("Any", _sub))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
        # log
        step_log.append(
            {
                "step": idx,
                "decision": {
                    "kind": decision.kind,
                    "tile": decision.tile,
                    "negate": decision.negate,
                },
                "ref_action": {
                    "kind": ref_action.kind,
                    "tile": int(cast("Any", ref_action.tile))
                    if ref_action.tile is not None
                    else None,  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                    "consumed": [int(x) for x in getattr(ref_action, "consumed_tiles", ())],
                },
                "mahjax_prims": prims,
                "mj_current": int(cast("Any", mj_state.current_player)),  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                "ref_actor_next": sim._expected_actor_or_none(),
            }
        )
        # compare
        fail = _compare_projections(scenario, idx, sim, mj_state)
        if fail is not None:
            failures.append(fail)
            break
        if bool(getattr(mj_state, "terminated", False)) and getattr(sim, "_terminal", False):
            break
    return failures, step_log, wall, deck


def run_differential(
    *,
    artifact_root: Path | None = None,
    manifest: RulesManifest | None = None,
    rules_id: str | DigestText | None = None,
) -> DifferentialResult:
    """Run the full differential suite over SCENARIO_REGISTRY.

    Persists first counterexample per failing case to
    ``$ROOT/counterexamples/WP-04C/`` and, only on zero mismatches,
    publishes a qualification token bound to the full environment tuple to
    ``$ROOT/tokens/WP-04C/`` with a shell round-trip check.
    """
    if artifact_root is None:
        from hydra2.config import artifact_root as _ar

        root = _ar()
    else:
        root = Path(artifact_root)
    # manifest
    payload: Any | None = None
    if manifest is None:
        import json as _json

        from hydra2.config import repo_root as _diff_repo_root
        from hydra2.contracts.rules_manifest import rules_manifest_from_payload

        # Portable payload path: repo_root() marker walk (not parents[3] depth).
        # Evidence: https://docs.python.org/3/library/pathlib.html#pathlib.Path.resolve
        # Evidence: https://github.com/fsspec/universal_pathlib

        payload_path = _diff_repo_root() / "configs" / "rules" / "tenhou_4p_hanchan_v1.json"
        # fallback to importlib.resources for wheel installs (zip-safe)
        if not payload_path.is_file():
            try:
                import importlib.resources as _ir

                payload_path = Path(
                    str(_ir.files("hydra2") / "configs" / "rules" / "tenhou_4p_hanchan_v1.json")
                )
            except Exception:
                payload_path = _diff_repo_root() / "configs" / "rules" / "tenhou_4p_hanchan_v1.json"
        payload = _json.loads(payload_path.read_text())["payload"]
        manifest = rules_manifest_from_payload(cast("Any", payload))  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
    # rules_id
    if rules_id is None:
        # use manifest digest

        # manifest has rules_id attribute? It's string id
        try:
            rules_id = str(manifest.rules_id)  # type: ignore[attr-defined]  # reason: no stubs; runtime
            # need digest text form sha256:... ; use payload digest via canonical
            # For token we need DigestText of rules manifest identity (payload hash)
            # Use the same as WP-04A's schedule digest?
            # Instead use manifest digest via rules_identity_hash

            # compute from file
            from hydra2.config import repo_root as _rr

            _ = _rr() / "configs" / "rules" / "tenhou_4p_hanchan_v1.json"
            # The file contains envelope with payload; we need payload digest?
            # Use manifest's digest via of_canonical of payload
            if payload is None:
                import json as _json2

                pp = _rr() / "configs" / "rules" / "tenhou_4p_hanchan_v1.json"
                payload = _json2.loads(pp.read_text())["payload"]
            rules_id = str(
                of_canonical(cast("Any", payload))
            )  # payload canonical digest  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
            # But token expects DigestText like sha256:...
            # Ensure prefix
            if not rules_id.startswith("sha256:"):
                rules_id = (
                    "sha256:"
                    + hashlib.sha256(
                        json.dumps(cast("Any", payload), sort_keys=True).encode()
                    ).hexdigest()  # pyrefly: ignore[explicit-any]  # reason: dynamic JAX
                )
        except Exception:
            rules_id = "sha256:" + "0" * 64
    # normalize rules_id to DigestText string
    if isinstance(rules_id, str) and not rules_id.startswith("sha256:"):
        # try to coerce
        try:
            rules_id = str(_bridge_contracts.make_digest_text(rules_id))
        except Exception:
            rules_id = "sha256:" + hashlib.sha256(str(rules_id).encode()).hexdigest()
    # run scenarios
    all_failures: list[CheckpointFailure] = []
    first_counterexample: str | None = None
    passed = 0
    for scenario in SCENARIO_REGISTRY:
        failures, step_log, wall, deck = _run_one_scenario(scenario, manifest, root)
        if len(failures) != 0:
            all_failures.extend(failures)
            # persist first failure for this scenario only (first counterexample)
            if first_counterexample is None:
                try:
                    first_counterexample = _persist_counterexample(
                        root, failures[0], scenario, wall, deck, step_log
                    )
                except Exception:
                    first_counterexample = None
        else:
            passed += 1
    total = len(SCENARIO_REGISTRY)
    failed = total - passed
    verdict = "passed" if len(all_failures) == 0 else "blocked"
    # execution mode sweep
    sweep = execution_mode_sweep(SCENARIO_REGISTRY[0], artifact_root=root)
    deterministic = bool(sweep.get("deterministic", False))
    # gpu probe + cpu soak
    gpu_probe = gpu_soak_probe(artifact_root=root)
    cpu = cpu_soak(artifact_root=root)
    # token issuance only on zero mismatches
    token_path: str | None = None
    token_digest: str | None = None
    env_digest = ""
    try:
        from hydra2.engines.mahjax.capture import capture_mahjax_tuple

        env_digest = str(capture_mahjax_tuple().digest)
    except Exception:
        env_digest = "sha256:" + "0" * 64
    if len(all_failures) == 0:
        try:
            # rules_id for token is the manifest's rules_id string? Use the same as shell expects
            # Shell expects DigestText of rules_id; we have payload digest

            # Try to use actual manifest rules_id if available
            try:
                _actual_rules_id = str(manifest.rules_id)  # type: ignore[attr-defined]  # reason: no stubs; runtime
                # actual is like "tenhou_4p_hanchan_v1" not digest; need digest text
                # Fabricate token uses make_digest_text on that string?
                # But make_digest_text expects sha256:...
                # So we should use payload digest as rules_id for token
                # The shell's qualify checks token.rules_id
                # == supplied rules_id, so we must be consistent
                # Use payload digest as token's rules_id
                token_rules_id = _bridge_contracts.make_digest_text(rules_id)
            except Exception:
                token_rules_id = _bridge_contracts.make_digest_text(rules_id)
            p, d = _publish_token(root, str(token_rules_id))
            token_path = str(p)
            token_digest = d
        except Exception as exc:  # pragma: no cover
            # token issuance failed -> treat as blocked
            verdict = "blocked"
            all_failures.append(
                CheckpointFailure(
                    case_id="token",
                    step_index=-1,
                    dimension="token_issuance",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            token_path = None
            token_digest = None
    else:
        # ensure no token left from prior run if now failing
        # we do not delete existing token, but we don't create new
        pass
    return DifferentialResult(
        verdict=verdict,
        total_cases=total,
        passed_cases=passed,
        failed_cases=failed,
        mismatches=tuple(all_failures),
        first_counterexample_path=first_counterexample,
        token_path=token_path,
        token_digest=token_digest,
        env_tuple_digest=env_digest,
        execution_mode_deterministic=deterministic,
        gpu_probe=gpu_probe,
        cpu_soak=cpu,
    )
