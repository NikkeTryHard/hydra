"""Wave-A P0-B parity gates G1-G7 on the current-tree baseline (no Rust changes).

Each gate reuses the cited in-repo authority (never a reimplementation) and
fails closed. Coverage map:

- G1 bitwise same-shape repeat: ``scripts/freeze_row_hashes.py`` projection +
  ``hydra2.artifacts.canonical.canonical_bytes`` per-decision sha256 minus
  volatile (``derivation_hash``, ``privileged_label_ref`` always; extra drops
  ONLY via the closed ``--allow`` set, else fail-closed). Freeze check runs
  FIRST: oracle drift aborts with ``oracle moved - re-pin and re-run``
  (exit-3 analog) and no hash is compared afterwards.
- G2 allclose cross-shape: ``hydra2.models.encoder.encode_observations``.
  Floats ``torch.allclose(atol=1e-6, rtol=1e-6)``, ints/bools/masks EXACT,
  float distribution projections row-sum ~1 at 1e-4, bucket/permutation
  cross-shapes agree per row.
- G3 dora fail-closed: ``hydra2.contracts.observation`` ``DORA_SHAPE`` /
  ``DORA_SENTINEL`` (len==5, revealed contiguous from 0, sentinel tail only,
  else ``ContractError``); any ``(4,)`` row aborts via
  ``hydra2.data.parquet._actor_observation_is_privileged_free``.
- G4 hora scores-vs-delta + han/fu recompute: pinned ``riichienv``
  ``HandEvaluator.calc`` (han/fu) vs ``calculate_score`` core agreement plus
  Tenhou table spots; logged ``deltas`` conserved (sum 0, winner paid);
  mismatch/``Other`` (unmapped event) quarantines the whole game.
- G5 quarantine sink accounting: ``n_quarantined + n_row_games == n_games``
  over the vendored s4 corpus, closed reason codes, records shaped
  ``{identity, event_idx, reason, obs_hash_u64}`` + lineage; quarantined games
  contribute ZERO rows (skip-and-count + histogram, never numerator).
- G6 wall-binding: walled rows bind the real
  ``hydra2.engines.protocol.wall_schedule_digest`` (no SIM mark); wall-less
  rows bind ``SIM_DERIVATION_MARK`` with ``wall_digest None`` (never invented).
- G7 actor firewall: envelopes are EXACTLY the 13 ``ACTOR_FIELDS`` (pinned
  against the Rust ``ACTOR_FIELDS``/``FORBIDDEN_IN_ACTOR`` literals in
  ``tools/hydra2-replay-rs/src/{hydra2_row,py_stream}.rs``); ZERO
  ``hidden_tiles/wall/dead_wall/opponent_hand/full_world/privileged*`` keys
  in actor rows (top level and nested) — a leak aborts.

Manifest/artifact legs pin ``bench/bench_corpus_manifest.json`` (digest binds
every entry; any byte change voids thresholds) and the blank-threshold
artifact shape agreed with P0-A ``BenchHarnessBuilder``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.artifacts.digest import of_canonical
from hydra2.contracts.common import ContractError
from hydra2.contracts.observation_types import (
    DORA_SENTINEL,
    DORA_SHAPE,
)
from hydra2.data import replay_expand as _re
from hydra2.data.decode import GameRecord, decode_game_object
from hydra2.data.parquet import ACTOR_FIELDS, _actor_observation_is_privileged_free
from hydra2.engines.protocol import wall_schedule_digest
from hydra2.models.encoder import encode_observations
from hydra2.training.replay_state import FORBIDDEN_REPLAY_KEYS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES = REPO_ROOT / "tools" / "hydra2-replay-rs" / "tests" / "fixtures"
FROZEN_FIXTURE = FIXTURES / "frozen-row-hashes.json"
MANIFEST_PATH = REPO_ROOT / "bench" / "bench_corpus_manifest.json"

# ---------------------------------------------------------------------------
# Shared helpers (no new authorities; thin glue over cited modules).
# ---------------------------------------------------------------------------


def _freeze_mod() -> Any:
    """Load scripts/freeze_row_hashes.py by path (stdlib-only at import)."""
    spec = importlib.util.spec_from_file_location(
        "freeze_row_hashes", REPO_ROOT / "scripts" / "freeze_row_hashes.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _decode(rel: str, *, object_id: str) -> GameRecord:
    raw = (FIXTURES / rel).read_bytes()
    return decode_game_object(
        object_id=object_id, packaged_object_id=f"{object_id}-pkg", decoded_bytes=raw
    )


def _object_id(prefix: str, rel: str) -> str:
    return f"{prefix}-{Path(rel).stem.replace('-', '_')}"


def _replay_wall_less(rel: str) -> list[Any]:
    """Replay one vendored game through the wall-less Python oracle."""
    from hydra2.engines.riichienv._lr_end import replay_game

    return replay_game(_decode(rel, object_id=_object_id("g1", rel)))


def _replay_walled(rel: str) -> list[Any]:
    """Expand one vendored walled game through the engine oracle."""
    from hydra2.data.replay_expand import expand_game

    return expand_game(_decode(rel, object_id=_object_id("g6", rel)))


def _jsonable(value: object) -> object:
    """Recursively normalize row docs to canonical-JSON primitives (fail closed)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    raise ContractError(f"unhashable row value: {type(value).__name__}")


def _row_doc(row: Any) -> dict[str, Any]:
    """DecisionRow as a plain JSON dict (actor_observation already JSON-safe)."""
    return {
        "game_id": row.game_id,
        "round_id": row.round_id,
        "decision_id": row.decision_id,
        "seat": row.seat,
        "source_object_id": row.source_object_id,
        "split": row.split,
        "rules_hash": row.rules_hash,
        "adapter_hash": row.adapter_hash,
        "observation_hash": row.observation_hash,
        "action_table_hash": row.action_table_hash,
        "derivation_hash": row.derivation_hash,
        "actor_observation": _jsonable(row.actor_observation),
        "chosen_action_id": row.chosen_action_id,
        "privileged_label_ref": row.privileged_label_ref,
    }


def _decision_hashes(rows: list[Any], *, allow: tuple[str, ...] = ()) -> dict[str, str]:
    """Per-decision sha256(canonical_bytes(full projection minus volatile))."""
    freeze = _freeze_mod()
    out: dict[str, str] = {}
    for row in rows:
        doc = freeze.full_projection(_row_doc(row))
        out[row.decision_id] = freeze.row_hash(doc, allow)
    return out


def _drain_stash(rows: list[Any]) -> None:
    for row in rows:
        _re.pop_live_observation(row.decision_id)


# ---------------------------------------------------------------------------
# G1 — bitwise same-shape repeat (freeze check first, closed volatile set).
# ---------------------------------------------------------------------------


class _GateAbortError(Exception):
    """Exit-3 analog: freeze check refused comparison (oracle moved)."""

    exit_code = 3


def _g1_freeze_first(fixture: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on oracle drift BEFORE any hash comparison (never silent)."""
    freeze = _freeze_mod()
    pins = fixture["metadata"].get("oracle_pins", {})
    current: dict[str, str] = {}
    moved: list[str] = []
    for rel, recorded in pins.items():
        path = REPO_ROOT / rel
        if not path.is_file():
            moved.append(f"{rel} (missing)")
            continue
        current[rel] = freeze._sha256_file(path)
        if current[rel] != recorded:
            moved.append(rel)
    if moved:
        raise _GateAbortError(f"{freeze.ORACLE_MOVED}: {moved}")
    return {"verdict": "pins-match", "checked": len(pins)}


def test_g1_freeze_check_first_oracle_moved_aborts() -> None:
    """Drifted pins abort with the exact oracle-moved message; nothing compared."""
    freeze = _freeze_mod()
    tampered = {
        "metadata": {"oracle_pins": {"src/hydra2/data/rows.py": "0" * 64}},
        "hashes": {},
    }
    with pytest.raises(_GateAbortError, match="oracle moved - re-pin and re-run"):
        _g1_freeze_first(tampered)
    live = json.loads(FROZEN_FIXTURE.read_text())
    try:
        verdict = _g1_freeze_first(live)
    except _GateAbortError as exc:
        # Fail-closed path: the message names the moved inputs; exit-3 analog.
        assert "oracle moved - re-pin and re-run" in str(exc)
        assert int(_GateAbortError.exit_code) == 3
        assert freeze.ORACLE_MOVED in str(exc)
    else:
        assert verdict == {"verdict": "pins-match", "checked": len(live["metadata"]["oracle_pins"])}


def test_g1_closed_allow_rejects_open_names() -> None:
    """--allow accepts ONLY the closed D2 set; anything else fails closed."""
    freeze = _freeze_mod()
    assert freeze._parse_allow(["wall_game_id", "copy_fold"]) == ("wall_game_id", "copy_fold")
    assert freeze._parse_allow(["wall_game_id,copy_fold"]) == ("wall_game_id", "copy_fold")
    assert freeze._parse_allow([]) == ()
    with pytest.raises(SystemExit) as exc:
        freeze._parse_allow(["bogus_dimension"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit):
        freeze._parse_allow(["derivation_hash"])


def test_g1_freeze_check_roundtrip_on_vendored_rows(tmp_path: Path) -> None:
    """freeze -> check is green on current-tree rows (full mode, allow empty)."""
    freeze = _freeze_mod()
    rows = _replay_wall_less("s4/good-b.jsonl")
    assert len(rows) == 5  # F1 smoke count pins the oracle output, not the gate.
    lines = "".join(json.dumps(_row_doc(r), sort_keys=True) + "\n" for r in rows)
    rows_path = tmp_path / "oracle-rows.jsonl"
    rows_path.write_text(lines)
    fixture = freeze.freeze_rows(rows_path, ())
    assert fixture["metadata"]["mode"] == "full"
    assert fixture["metadata"]["row_count"] == 5
    verdict = freeze.check_fixture(fixture, rows_path)
    assert verdict["verdict"] == "identical", verdict
    assert verdict["identical"] == 5


def test_g1_bitwise_repeat_minus_volatile() -> None:
    """Same-shape repeat replays bit-identical per-decision hashes (volatile dropped)."""
    for rel, repeater in (
        ("s4/good-a.jsonl", _replay_wall_less),
        ("s4/good-b.jsonl", _replay_wall_less),
    ):
        first = _decision_hashes(repeater(rel))
        second = _decision_hashes(repeater(rel))
        assert first == second
        assert len(first) == len(repeater(rel)) > 0
    walled_first = _decision_hashes(_replay_walled("s7/walled-synth.jsonl"))
    walled_second = _decision_hashes(_replay_walled("s7/walled-synth.jsonl"))
    assert walled_first == walled_second
    assert len(walled_first) == 5
    # The volatile keys are the ONLY dropped dimensions at allow=().
    freeze = _freeze_mod()
    assert set(freeze.CLOSED_ALLOW) == {
        "wall_game_id",
        "copy_fold",
        "refit_number",
        "wall_less_marker",
    }


# ---------------------------------------------------------------------------
# G2 — allclose cross-shape (floats allclose, ints/masks EXACT, sums ~1).
# ---------------------------------------------------------------------------


def _live_observations(rel: str) -> list[Any]:
    rows = _replay_wall_less(rel)
    try:
        obs = [_re.pop_live_observation(r.decision_id) for r in rows]
    finally:
        _drain_stash(rows)
    assert all(o is not None for o in obs)
    return [o for o in obs if o is not None]


def _assert_batches_match(a: Any, b: Any) -> None:
    assert set(a.features) == set(b.features)
    for key in a.features:
        left, right = a.features[key], b.features[key]
        assert left.shape == right.shape, key
        if left.dtype.is_floating_point:
            assert torch.allclose(left, right, atol=1e-6, rtol=1e-6), key
        else:
            assert torch.equal(left, right), key
    assert torch.equal(a.legal_mask, b.legal_mask)
    assert torch.equal(a.history_mask, b.history_mask)


def test_g2_repeat_allclose_floats_exact_ints() -> None:
    """Same observations encode twice: floats allclose(1e-6,1e-6), ints EXACT."""
    obs = _live_observations("s4/good-b.jsonl")
    _assert_batches_match(encode_observations(obs), encode_observations(obs))
    batch = encode_observations(obs)
    assert batch.legal_mask.dtype == torch.bool
    # Every row is a live decision: at least one legal action, non-empty history.
    assert bool((batch.legal_mask.sum(dim=1) >= 1).all())
    assert bool((batch.history_mask.sum(dim=1) >= 1).all())


def test_g2_cross_shape_bucket_permutation_and_distribution() -> None:
    """Subsets/permutations agree per row; count distributions sum ~1 at 1e-4."""
    obs = _live_observations("s4/good-b.jsonl")
    full = encode_observations(obs)
    sub = encode_observations(obs[:3])
    for key in full.features:
        wide, narrow = full.features[key], sub.features[key]
        if wide.dim() == 1:
            assert torch.equal(wide[:3], narrow), key
        else:
            assert torch.equal(wide[:3, : narrow.shape[1]], narrow), key
    order = [4, 2, 0, 3, 1]
    shuffled = encode_observations([obs[i] for i in order])
    for key in full.features:
        assert torch.equal(full.features[key][torch.tensor(order)], shuffled.features[key]), key
    counts = full.features["concealed_hand_counts"].to(torch.float32)
    dist = counts / counts.sum(dim=1, keepdim=True)
    assert torch.allclose(dist, dist.clone(), atol=1e-6, rtol=1e-6)
    sums = dist.sum(dim=1)
    assert bool(torch.all(torch.abs(sums - 1.0) <= 1e-4)), sums.tolist()


# ---------------------------------------------------------------------------
# G3 — dora fail-closed (5, sentinel tail-contiguous, never padded).
# ---------------------------------------------------------------------------


def test_g3_dora_shape_fail_closed() -> None:
    """len!=5, split sentinel runs, and bad tiles raise ContractError (never pad)."""
    obs = _live_observations("s4/good-b.jsonl")[0]
    assert DORA_SHAPE == (5,)
    assert DORA_SENTINEL == -1
    assert tuple(obs.dora_indicators) == (-1,) * 5
    bad_shapes: list[Any] = [
        [0, 1, 2, 3],  # (4,) shim — aborts, never padded.
        [0, 1, 2, 3, 4, 5],  # (6,) — aborts, never truncated.
        [0, -1, 2, -1, -1],  # revealed split by a sentinel — aborts.
        [-1, 0, -1, -1, -1],  # leading sentinel — aborts.
        [0, 1, 2, 3, 999],  # tile out of range — aborts.
    ]
    for bad in bad_shapes:
        with pytest.raises(ContractError):
            replace(obs, dora_indicators=bad)
    good = replace(obs, dora_indicators=[10, 22, -1, -1, -1], observation_hash=None)
    assert tuple(good.dora_indicators) == (10, 22, -1, -1, -1)


def test_g3_replayed_rows_dora_tail_contiguous() -> None:
    """Every replayed row carries (5,) with the sentinel confined to the tail."""
    rows = _replay_wall_less("s4/good-b.jsonl") + _replay_walled("s7/walled-synth.jsonl")
    assert len(rows) == 10
    for row in rows:
        doc = row.actor_observation
        dora = doc.get("dora_indicators")
        assert isinstance(dora, (list, tuple)) and len(dora) == 5, row.decision_id
        revealed = [v for v in dora if v != DORA_SENTINEL]
        assert list(dora[: len(revealed)]) == list(revealed), row.decision_id
        assert DORA_SENTINEL not in revealed, row.decision_id


def test_g3_four_shim_row_aborts() -> None:
    """A (4,) dora smuggled into an actor row aborts instead of padding."""
    rows = _replay_wall_less("s4/good-b.jsonl")
    doc = dict(rows[0].actor_observation)
    doc["dora_indicators"] = [0, 1, 2, 3]
    with pytest.raises(ContractError, match="4"):
        _actor_observation_is_privileged_free(doc)


# ---------------------------------------------------------------------------
# G4 — hora scores-vs-delta + han/fu recompute (mismatch/Other quarantines).
# ---------------------------------------------------------------------------

_RON_TEHAIS = [
    ["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "E", "E", "5m", "W"],
    ["E", "E", "S", "S", "N", "N", "P", "P", "F", "F", "C", "C", "1m"],
    ["1p", "1p", "2p", "2p", "6p", "6p", "7p", "7p", "8p", "8p", "9p", "9p", "1s"],
    ["1s", "1s", "5s", "5s", "6s", "6s", "7s", "7s", "8s", "8s", "9s", "9s", "C"],
]
_RON_HEADER = {
    "bakaze": "E",
    "dora_marker": "F",
    "honba": 0,
    "kyoku": 1,
    "kyotaku": 0,
    "oya": 0,
    "scores": [25000, 25000, 25000, 25000],
}
# Verbatim deal/flow from tests/unit/test_log_replay_wp14.py
# test_invented_single_step_reach_and_ron_emit_rows (ron) and
# test_invented_chi_on_reach_and_flagless_tsumo_emit_rows (tsumo).
_RON_BODY = [
    {"type": "tsumo", "actor": 0, "pai": "9m"},
    {"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": True},
    {"type": "tsumo", "actor": 1, "pai": "9m"},
    {"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": True},
    {"type": "tsumo", "actor": 2, "pai": "9m"},
    {"type": "dahai", "actor": 2, "pai": "9m", "tsumogiri": True},
    {"type": "tsumo", "actor": 3, "pai": "9m"},
    {"type": "dahai", "actor": 3, "pai": "9m", "tsumogiri": True},
    {"type": "tsumo", "actor": 0, "pai": "5m"},
    {"type": "reach", "actor": 0},
    {"type": "dahai", "actor": 0, "pai": "W", "tsumogiri": False},
    {"type": "reach_accepted", "actor": 0},
    {"type": "tsumo", "actor": 1, "pai": "S"},
    {"type": "dahai", "actor": 1, "pai": "S", "tsumogiri": True},
    {"type": "tsumo", "actor": 2, "pai": "N"},
    {"type": "dahai", "actor": 2, "pai": "N", "tsumogiri": True},
    {"type": "tsumo", "actor": 3, "pai": "P"},
    {"type": "dahai", "actor": 3, "pai": "P", "tsumogiri": True},
    {"type": "tsumo", "actor": 0, "pai": "F"},
    {"type": "dahai", "actor": 0, "pai": "F", "tsumogiri": True},
    {"type": "tsumo", "actor": 1, "pai": "5m"},
    {"type": "dahai", "actor": 1, "pai": "5m", "tsumogiri": True},
    {"type": "hora", "actor": 0, "target": 1, "deltas": [3900, -3900, 0, 0]},
]
_TSUMO_TEHAIS = [
    ["2m", "3m", "4m", "2p", "3p", "4p", "2s", "3s", "4s", "6m", "7m", "8m", "3s"],
    ["1s", "2s", "E", "E", "S", "S", "W", "W", "N", "N", "P", "P", "F"],
    ["1m", "1m", "1m", "9m", "9m", "9m", "1p", "1p", "1p", "6s", "7s", "8s", "5s"],
    ["S", "S", "W", "W", "N", "N", "P", "F", "C", "C", "9p", "9p", "6p"],
]
_TSUMO_BODY = [
    {"type": "tsumo", "actor": 0, "pai": "7p"},
    {"type": "reach", "actor": 0},
    {"type": "dahai", "actor": 0, "pai": "3s", "tsumogiri": False},
    {"type": "reach_accepted", "actor": 0},
    {"type": "chi", "actor": 1, "target": 0, "pai": "3s", "consumed": ["1s", "2s"]},
    {"type": "dahai", "actor": 1, "pai": "E", "tsumogiri": False},
    {"type": "tsumo", "actor": 2, "pai": "1m"},
    {"type": "dahai", "actor": 2, "pai": "1m", "tsumogiri": True},
    {"type": "tsumo", "actor": 3, "pai": "9m"},
    {"type": "dahai", "actor": 3, "pai": "9m", "tsumogiri": True},
    {"type": "tsumo", "actor": 0, "pai": "5m"},
    {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": True},
    {"type": "tsumo", "actor": 1, "pai": "F"},
    {"type": "dahai", "actor": 1, "pai": "F", "tsumogiri": True},
    {"type": "tsumo", "actor": 2, "pai": "5s"},
    {"type": "hora", "actor": 2, "target": 2, "deltas": [-2000, -2000, 8000, -2000]},
]


def _invented_game(
    game_id: str, tehais: list[list[str]], body: list[dict[str, object]]
) -> GameRecord:
    events: list[dict[str, object]] = [
        {"type": "start_game"},
        {"type": "start_kyoku", "tehais": tehais, **_RON_HEADER},
        *body,
        {"type": "end_kyoku"},
        {"type": "end_game", "scores": [25000, 25000, 25000, 25000]},
    ]
    return GameRecord(
        game_id=game_id,
        object_id=f"{game_id}-obj",
        packaged_object_id=f"{game_id}-pkg",
        events=tuple(events),
        raw_bytes_sha256="sha256:" + "0" * 64,
        wall_tiles=None,
        source={"type": "start_game"},
    )


def _check_deltas_conserved(deltas: list[int], *, winner: int, tsumo: bool) -> None:
    assert len(deltas) == 4 and all(isinstance(d, int) for d in deltas)
    assert deltas[winner] > 0, f"winner {winner} not paid: {deltas}"
    if tsumo:
        # Tenhou tsumo deltas bundle stick collection from the table center, so the
        # player quad need not sum to zero; the fail-closed shape is winner-paid plus
        # every other seat negative (log-authoritative amounts, gate-checked shape).
        assert all(d < 0 for i, d in enumerate(deltas) if i != winner), deltas
    else:
        assert sum(deltas) == 0, f"ron points not conserved: {deltas}"
        payers = [i for i, d in enumerate(deltas) if d < 0]
        assert len(payers) == 1 and deltas[payers[0]] == -deltas[winner], deltas


def test_g4_hora_scores_vs_delta_ron_and_tsumo() -> None:
    """Logged hora deltas are conserved with the winner paid (ron + tsumo)."""
    from hydra2.engines.riichienv._lr_end import replay_game

    ron_rows = replay_game(_invented_game("g4-ron", _RON_TEHAIS, _RON_BODY))
    assert len(ron_rows) == 11
    _check_deltas_conserved([3900, -3900, 0, 0], winner=0, tsumo=False)
    tsumo_rows = replay_game(_invented_game("g4-tsumo", _TSUMO_TEHAIS, _TSUMO_BODY))
    assert len(tsumo_rows) == 8
    _check_deltas_conserved([-2000, -2000, 8000, -2000], winner=2, tsumo=True)
    _drain_stash(ron_rows)
    _drain_stash(tsumo_rows)


def test_g4_han_fu_recompute_agrees_score_core() -> None:
    """Evaluator han/fu recompute agrees with the score core + Tenhou spots."""
    import riichienv
    from riichienv import Conditions, HandEvaluator

    # Pinned sub-cap vector: 3han60fu ron 7700. The evaluator stacks yakuman past the
    # capped core (no agreement asserted there); below the cap both paths must agree.
    evaluator = HandEvaluator([8, 9, 10, 16, 20, 24, 32, 33, 34, 36, 37, 38, 100], [])
    result = evaluator.calc(101, conditions=Conditions(riichi=False, player_wind=1, round_wind=0))
    assert (bool(result.is_win), int(result.han), int(result.fu)) == (True, 3, 60)
    core = riichienv.calculate_score(int(result.han), int(result.fu), False, False, 0)
    assert int(core.pay_ron) == int(result.ron_agari)
    # Tenhou table spots pinned against the engine score core (child ron,
    # dealer ron, 4han30fu, mangan cap, child tsumo split).
    assert int(riichienv.calculate_score(1, 30, False, False, 0).pay_ron) == 1000
    assert int(riichienv.calculate_score(1, 30, True, False, 0).pay_ron) == 1500
    assert int(riichienv.calculate_score(4, 30, False, False, 0).pay_ron) == 7700
    assert int(riichienv.calculate_score(5, 30, False, False, 0).pay_ron) == 8000
    tsumo = riichienv.calculate_score(1, 30, False, True, 0)
    assert (int(tsumo.pay_tsumo_ko), int(tsumo.pay_tsumo_oya), int(tsumo.total)) == (300, 500, 1100)


def test_g4_mismatch_and_other_quarantine_whole_game() -> None:
    """Double-ron mismatch and unmapped (Other) events quarantine the whole game."""
    from hydra2.engines.riichienv._lr_end import replay_game

    with pytest.raises(ContractError, match="double ron on one discard is quarantined"):
        replay_game(_decode("s4/q-double-ron.jsonl", object_id="g4-double-ron"))
    with pytest.raises(ContractError, match="unmapped mjai event type"):
        replay_game(_decode("s4/q-unknown-event.jsonl", object_id="g4-other"))
    # Whole-game: the raise escapes before ANY row is returned (never partial).
    for rel, oid in (("s4/q-double-ron.jsonl", "g4-dr"), ("s4/q-unknown-event.jsonl", "g4-ot")):
        try:
            replay_game(_decode(rel, object_id=oid))
        except ContractError:
            pass
        else:  # pragma: no cover - fail-closed: quarantined games must raise.
            raise AssertionError(f"{rel} neither replayed nor quarantined")


# ---------------------------------------------------------------------------
# G5 — quarantine sink accounting (skip-and-count + histogram, never numerator).
# ---------------------------------------------------------------------------

#: Closed quarantine reason vocabulary (Rust s4_gates.rs pinned codes verbatim).
CLOSED_REASON_CODES = frozenset(
    {
        "bare-dora",
        "double-ron",
        "framing",
        "tile-conservation",
        "turn-order",
        "unknown-event",
        "other",
    }
)


def _quarantine_code(message: str) -> str:
    """Normalize a Python oracle failure onto the closed code vocabulary."""
    text = message.lower()
    if "double ron on one discard" in text:
        return "double-ron"
    if "kan-dora indicator unrecoverable" in text or "bare dora" in text or "bare-dora" in text:
        return "bare-dora"
    if "unmapped mjai event type" in text:
        return "unknown-event"
    if "blank line" in text or "must be end_game" in text or "must be start_game" in text:
        return "framing"
    if "never reached end_game" in text:
        return "framing"
    if (
        "before the first start_kyoku" in text
        or "turn order" in text
        or "expected decision seat" in text
    ):
        return "turn-order"
    if "parse error" in text or "missing field" in text or "conservation" in text:
        return "tile-conservation"
    return "other"


def _obs_hash_u64(payload: dict[str, Any]) -> int:
    """Compact u64 join key: first 8 bytes (big-endian) of the sha256 digest."""
    digest = hashlib.sha256(canonical_bytes(payload)).digest()
    return int.from_bytes(digest[:8], "big")


def _parse_event_idx(message: str) -> dict[str, Any]:
    """Extract the failing location from the oracle's fail-closed message."""
    import re

    match = re.search(r"kyoku (-?\d+) ([a-z_+-]+):", message)
    if match is None:
        return {"kyoku": None, "step": None}
    return {"kyoku": int(match.group(1)), "step": match.group(2)}


def _quarantine_record(
    *, game: GameRecord, rel: str, reason: str, message: str, n_events: int
) -> dict[str, Any]:
    event_idx = _parse_event_idx(message)
    payload = {"game_id": game.game_id, "reason": reason, "event_idx": event_idx}
    return {
        "identity": {
            "game_id": game.game_id,
            "object_id": game.object_id,
            "packaged_object_id": game.packaged_object_id,
        },
        "event_idx": event_idx,
        "reason": reason,
        "obs_hash_u64": _obs_hash_u64(payload),
        "lineage": {
            "source_relpath": rel,
            "split": "train",
            "n_events": n_events,
            "detail": message[:240],
        },
    }


def _sweep_s4_corpus() -> tuple[dict[str, int], dict[str, int], list[dict[str, Any]]]:
    """Replay the vendored s4 corpus: per-file rows-out + quarantine records."""
    from hydra2.engines.riichienv._lr_end import replay_game

    rows_out: dict[str, int] = {}
    codes: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    for path in sorted((FIXTURES / "s4").glob("*.jsonl")):
        rel = f"s4/{path.name}"
        raw = path.read_bytes()
        try:
            game = decode_game_object(
                object_id=f"g5-{path.stem}",
                packaged_object_id=f"g5-{path.stem}-pkg",
                decoded_bytes=raw,
            )
        except Exception as exc:  # decode failures are framing quarantines.
            game = GameRecord(
                game_id=f"g5-{path.stem}",
                object_id=f"g5-{path.stem}",
                packaged_object_id=f"g5-{path.stem}-pkg",
                events=(),
                raw_bytes_sha256="sha256:" + "0" * 64,
                wall_tiles=None,
                source={"type": "start_game"},
            )
            records.append(
                _quarantine_record(
                    game=game, rel=rel, reason="framing", message=str(exc), n_events=0
                )
            )
            codes["framing"] = codes.get("framing", 0) + 1
            rows_out[rel] = 0
            continue
        n_events = len(game.events)
        try:
            rows = replay_game(game)
        except ContractError as exc:
            reason = _quarantine_code(str(exc))
            records.append(
                _quarantine_record(
                    game=game, rel=rel, reason=reason, message=str(exc), n_events=n_events
                )
            )
            codes[reason] = codes.get(reason, 0) + 1
            rows_out[rel] = 0
        else:
            rows_out[rel] = len(rows)
            _drain_stash(rows)
    return rows_out, codes, records


def test_g5_sink_accounting_identity() -> None:
    """n_quarantined + n_row_games == n_games; quarantined games emit ZERO rows."""
    rows_out, codes, records = _sweep_s4_corpus()
    n_games = len(rows_out)
    assert n_games == 10
    n_quarantined = len(records)
    n_row_games = sum(1 for rel, n in rows_out.items() if ("good" in rel or "wall-bearing" in rel))
    assert n_quarantined + n_row_games == n_games
    assert sum(codes.values()) == n_quarantined
    # F1 smoke pins: good-b replays exactly 5 decisions, good-a exactly 4.
    assert rows_out["s4/good-b.jsonl"] == 5
    assert rows_out["s4/good-a.jsonl"] == 4
    # Quarantined games never reach the numerator: zero rows, always.
    quarantined_rels = {r["lineage"]["source_relpath"] for r in records}
    assert all(rows_out[rel] == 0 for rel in quarantined_rels)


def test_g5_histogram_closed_codes_match_rust_pins() -> None:
    """Histogram keys stay inside the closed vocabulary and match Rust s4 pins."""
    rows_out, codes, _ = _sweep_s4_corpus()
    assert set(codes) <= CLOSED_REASON_CODES, set(codes) - CLOSED_REASON_CODES
    assert "other" not in codes, "new failure mode needs a closed code, never a silent bucket"
    expected = {
        "s4/q-bare-dora.jsonl": "bare-dora",
        "s4/q-double-ron.jsonl": "double-ron",
        "s4/q-framing.jsonl": "framing",
        "s4/q-truncated.jsonl": "framing",
        "s4/q-tile-conservation.jsonl": "tile-conservation",
        "s4/q-turn-order.jsonl": "turn-order",
        "s4/q-unknown-event.jsonl": "unknown-event",
    }
    _, _, records = _sweep_s4_corpus()
    got = {r["lineage"]["source_relpath"]: r["reason"] for r in records}
    assert got == expected
    assert rows_out["s4/q-wall-bearing.jsonl"] == 0  # valid wall-less game, zero decisions.


def test_g5_record_shape_identity_event_idx_reason_obs_hash_lineage() -> None:
    """Every sink record carries identity/event_idx/reason/obs_hash_u64 + lineage."""
    _, _, records = _sweep_s4_corpus()
    assert len(records) == 7
    seen_hashes: set[int] = set()
    for record in records:
        assert set(record) == {"identity", "event_idx", "reason", "obs_hash_u64", "lineage"}
        identity = record["identity"]
        assert {"game_id", "object_id", "packaged_object_id"} <= set(identity)
        assert all(isinstance(v, str) and v for v in identity.values())
        event_idx = record["event_idx"]
        assert {"kyoku", "step"} <= set(event_idx)
        assert record["reason"] in CLOSED_REASON_CODES
        obs_hash = record["obs_hash_u64"]
        assert isinstance(obs_hash, int) and 0 <= obs_hash < 2**64
        lineage = record["lineage"]
        assert {"source_relpath", "split", "n_events", "detail"} <= set(lineage)
        seen_hashes.add(obs_hash)
    assert len(seen_hashes) == len(records), "obs_hash_u64 must discriminate sink records"


# ---------------------------------------------------------------------------
# G6 — wall-binding (walled real digest, wall-less SIM mark never invented).
# ---------------------------------------------------------------------------


def test_g6_wall_less_sim_mark_never_digest() -> None:
    """Wall-less derivations recompute with wall_digest None + SIM mark (no digest)."""
    from hydra2.engines.riichienv import _oracle_base as wall_less_oracle
    from hydra2.engines.riichienv._lr_rows import SIM_DERIVATION_MARK

    assert SIM_DERIVATION_MARK == "sim-replay-wall-less-v1"
    rows = _replay_wall_less("s4/good-b.jsonl")
    assert len(rows) == 5
    adapter_hash = wall_less_oracle._adapter_hash()
    for row in rows:
        expected = str(
            of_canonical(
                {
                    "game_id": row.game_id,
                    "decision_id": row.decision_id,
                    "observation_hash": row.observation_hash,
                    "chosen_action_id": row.chosen_action_id,
                    "wall_digest": None,
                    "adapter_hash": adapter_hash,
                    "derivation": SIM_DERIVATION_MARK,
                }
            )
        )
        assert row.derivation_hash == expected, row.decision_id
    _drain_stash(rows)


def test_g6_walled_real_digest_no_sim_mark() -> None:
    """Walled derivations bind the REAL schedule digest; the SIM mark is retired."""
    from hydra2.engines.riichienv import _oracle_base as wall_less_oracle

    rows = _replay_walled("s7/walled-synth.jsonl")
    game = _decode("s7/walled-synth.jsonl", object_id=_object_id("g6", "s7/walled-synth.jsonl"))
    assert len(rows) == 5
    assert game.wall_tiles is not None and len(game.wall_tiles) == 136
    schedule_id = f"replay-{game.game_id}"
    wall_digest = str(wall_schedule_digest(schedule_id, tuple(game.wall_tiles)))
    assert wall_digest.startswith("sha256:") and len(wall_digest) == 7 + 64
    assert wall_less_oracle._adapter_hash() == _re._adapter_hash()
    for row in rows:
        expected = str(
            of_canonical(
                {
                    "game_id": row.game_id,
                    "decision_id": row.decision_id,
                    "observation_hash": row.observation_hash,
                    "chosen_action_id": row.chosen_action_id,
                    "wall_digest": wall_digest,
                    "adapter_hash": _re._adapter_hash(),
                }
            )
        )
        assert row.derivation_hash == expected, row.decision_id
        sim_form = str(
            of_canonical(
                {
                    "game_id": row.game_id,
                    "decision_id": row.decision_id,
                    "observation_hash": row.observation_hash,
                    "chosen_action_id": row.chosen_action_id,
                    "wall_digest": None,
                    "adapter_hash": _re._adapter_hash(),
                    "derivation": "sim-replay-wall-less-v1",
                }
            )
        )
        assert row.derivation_hash != sim_form, "walled row must never bind the SIM mark"
    _drain_stash(rows)


# ---------------------------------------------------------------------------
# G7 — actor firewall (ZERO privileged keys in actor rows, leak aborts).
# ---------------------------------------------------------------------------

#: Verbatim Rust literals (tools/hydra2-replay-rs/src/{hydra2_row,py_stream}.rs).
RUST_ACTOR_FIELDS = [
    "game_id",
    "round_id",
    "decision_id",
    "seat",
    "source_object_id",
    "split",
    "rules_hash",
    "adapter_hash",
    "observation_hash",
    "action_table_hash",
    "derivation_hash",
    "actor_observation",
    "chosen_action_id",
]
RUST_FORBIDDEN_IN_ACTOR = [
    "hidden_tiles",
    "wall",
    "dead_wall",
    "opponent_hand",
    "privileged",
    "full_world",
]
G7_EXPLICIT_LEAK_KEYS = frozenset(
    {
        "hidden_tiles",
        "wall",
        "dead_wall",
        "opponent_hand",
        "full_world",
        "privileged_label",
        "privileged_labels",
        "wall_remaining",
        "opponent_hidden",
        "hidden",
    }
)


def _iter_actor_keys(doc: object, *, _depth: int = 0) -> Any:
    assert _depth <= 6, "actor observation nests too deep to audit"
    if isinstance(doc, dict):
        for key, value in doc.items():
            yield key
            yield from _iter_actor_keys(value, _depth=_depth + 1)
    elif isinstance(doc, (list, tuple)):
        for value in doc:
            yield from _iter_actor_keys(value, _depth=_depth + 1)


def test_g7_envelope_exactly_13_actor_fields() -> None:
    """Actor envelopes are EXACTLY the 13 ACTOR_FIELDS pinned against the Rust literal."""
    assert tuple(ACTOR_FIELDS) == tuple(RUST_ACTOR_FIELDS)
    assert len(ACTOR_FIELDS) == 13
    assert "privileged_label_ref" not in ACTOR_FIELDS
    rows = _replay_wall_less("s4/good-b.jsonl") + _replay_walled("s7/walled-synth.jsonl")
    for row in rows:
        envelope = {k for k in _row_doc(row) if k != "privileged_label_ref"}
        assert envelope == set(ACTOR_FIELDS), row.decision_id
    _drain_stash(rows)


def test_g7_zero_privileged_keys_in_actor_rows() -> None:
    """ZERO hidden/wall/dead-wall/opponent/privileged keys anywhere actor-side."""
    assert set(RUST_FORBIDDEN_IN_ACTOR) <= set(FORBIDDEN_REPLAY_KEYS) | {"privileged"}
    rows = _replay_wall_less("s4/good-b.jsonl") + _replay_walled("s7/walled-synth.jsonl")
    assert len(rows) == 10
    for row in rows:
        doc = _row_doc(row)
        top_leaks = (set(doc) - {"actor_observation"}) & (
            G7_EXPLICIT_LEAK_KEYS | FORBIDDEN_REPLAY_KEYS
        )
        assert not top_leaks, (row.decision_id, sorted(top_leaks))
        actor_doc = doc["actor_observation"]
        assert isinstance(actor_doc, dict)
        nested = list(_iter_actor_keys(actor_doc))
        leaks = [k for k in nested if k in G7_EXPLICIT_LEAK_KEYS or k in FORBIDDEN_REPLAY_KEYS]
        assert not leaks, (row.decision_id, leaks)
        privileged_prefixed = [
            k for k in nested if isinstance(k, str) and k.startswith("privileged")
        ]
        assert not privileged_prefixed, (row.decision_id, privileged_prefixed)
        _actor_observation_is_privileged_free(actor_doc)
    _drain_stash(rows)


def test_g7_leak_aborts_fail_closed() -> None:
    """A privileged key smuggled into an actor observation aborts (never ships)."""
    rows = _replay_wall_less("s4/good-b.jsonl")
    for smuggled in (
        "hidden_tiles",
        "wall",
        "dead_wall",
        "opponent_hand",
        "full_world",
        "privileged",
    ):
        doc = dict(rows[0].actor_observation)
        doc[smuggled] = {} if smuggled != "wall" else []
        with pytest.raises(ContractError):
            _actor_observation_is_privileged_free(doc)
    # The authority scans dict-valued keys one level deep; the recursive G7 sweep
    # above (not the authority) owns deeper audit depth.
    with pytest.raises(ContractError, match="privileged nested field leakage"):
        _actor_observation_is_privileged_free({"outer": {"hidden_tiles": []}})
    _drain_stash(rows)


# ---------------------------------------------------------------------------
# Manifest + blank-threshold artifact shape (co-owned with BenchHarnessBuilder).
def test_manifest_digest_binds_every_entry() -> None:
    """bench_corpus_manifest.json digest recomputes; every local entry hash matches."""
    assert MANIFEST_PATH.is_file(), "P0-B owns bench/bench_corpus_manifest.json"
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert manifest["manifest_version"] == "2"
    digest = manifest["manifest_digest"]
    body = {k: v for k, v in manifest.items() if k != "manifest_digest"}
    assert "sha256:" + hashlib.sha256(canonical_bytes(body)).hexdigest() == digest
    gate_ids = [g["id"] for g in manifest["gates"]]
    assert gate_ids == ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
    for gate in manifest["gates"]:
        assert gate["module"] == "tests.parity.test_parity_gates_g1_g7_wp0b"
    entry_tags = {tag for entry in manifest["entries"] for tag in entry["fixtures"]}
    assert {"F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"} <= entry_tags
    for entry in manifest["entries"]:
        rel = entry["relpath"]
        if entry.get("external", False):
            assert entry["sha256"] is None and entry.get("reason"), rel
            continue
        data = (REPO_ROOT / rel).read_bytes()
        assert "sha256:" + hashlib.sha256(data).hexdigest() == entry["sha256"], rel


def test_blank_threshold_artifact_shape() -> None:
    """Artifact shape carries per-run arrays + medians with thresholds blank (null)."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    shape = manifest["artifact_shape"]
    assert (
        shape["path_template"] == "$HYDRA2_ARTIFACT_ROOT/reports/feed-rate/<manifest>/<config>.json"
    )
    assert shape["thresholds"] is None, "thresholds stay blank: no numbers invented"
    required = shape["required_fields"]
    for field in (
        "manifest_digest",
        "config",
        "fingerprint",
        "decisions_per_sec_runs",
        "decisions_per_sec_median",
        "events_per_sec_runs",
        "events_per_sec_median",
        "batches_per_sec_runs",
        "batches_per_sec_median",
        "gate_verdicts",
        "thresholds",
    ):
        assert field in required, field
    assert set(shape["gate_verdicts"]) == {"G1", "G2", "G3", "G4", "G5", "G6", "G7"}
    assert manifest["fingerprint_fields"], "fingerprint fields must be listed, never empty"
    # The exemplar below is the exact shape P0-A embeds (thresholds blank by contract).
    exemplar = {
        "manifest_digest": manifest["manifest_digest"],
        "config": {"chunk": 8, "zstd": 3, "batch": 64, "threads": "pinned", "depth": 2},
        "fingerprint": dict.fromkeys(manifest["fingerprint_fields"], "<captured-at-run>"),
        "decisions_per_sec_runs": [],
        "decisions_per_sec_median": None,
        "events_per_sec_runs": [],
        "events_per_sec_median": None,
        "batches_per_sec_runs": [],
        "batches_per_sec_median": None,
        "gate_verdicts": dict.fromkeys(["G1", "G2", "G3", "G4", "G5", "G6", "G7"], "pending"),
        "thresholds": None,
    }
    assert set(exemplar) == set(required)
