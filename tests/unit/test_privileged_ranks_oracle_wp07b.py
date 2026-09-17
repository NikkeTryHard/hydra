"""WP-07B privileged ranks carrier — 1..4 convention + writer/loader round trip."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from hydra2.belief.oracle_join import join_oracle_targets
from hydra2.belief.oracle_store import PrivilegedOracleLoader
from hydra2.contracts.common import ContractError
from hydra2.data.parquet import (
    validate_privileged_ranks,
    write_privileged_ranks,
    write_privileged_shards,
)

pytestmark = pytest.mark.contract_package("WP-07B")


def test_ranks_convention_strict_1_to_4_permutation() -> None:
    assert validate_privileged_ranks([1, 2, 3, 4]) == (1, 2, 3, 4)
    assert validate_privileged_ranks((4, 3, 2, 1)) == (4, 3, 2, 1)


@pytest.mark.parametrize(
    "bad",
    [
        [1, 2, 3],  # short
        [1, 2, 3, 4, 1],  # long
        [1, 1, 2, 3],  # duplicate
        [0, 1, 2, 3],  # seat convention: read-bridge only, never written
        [1, 2, 3, 5],  # out of range
        [1, 2, 3, "4"],  # non-int
        [True, 2, 3, 4],  # bool is not int
        "1234",  # not a list
        None,
    ],
)
def test_bad_ranks_raise(bad: object) -> None:
    with pytest.raises(ContractError):
        validate_privileged_ranks(bad, "dec-0001")
    with pytest.raises(ContractError):
        write_privileged_ranks(
            destination=Path("unused"),
            ranks_by_id={"dec-0001": bad},  # type: ignore[dict-item]
        )


def test_writer_shard_name_matches_loader_glob(tmp_path: Path) -> None:
    dest = tmp_path / "priv"
    write_privileged_ranks(
        destination=dest,
        ranks_by_id={"dec-0001": [1, 2, 3, 4], "dec-0002": [4, 3, 2, 1]},
        wall_ids={"dec-0001": "wall-a", "dec-0002": "wall-b"},
    )
    shards = sorted(dest.glob("privileged-*.parquet"))
    assert [p.name for p in shards] == ["privileged-000.parquet"]
    # Ranks-only labels carry no belief signal: synthetic-belief opt-in at
    # construction (join itself stays strict — see round-trip test below).
    loader = PrivilegedOracleLoader(dest, split="train", verify=True, allow_synthetic=True)
    assert len(loader) == 2


def _expected_utility_values(ranks: tuple[int, int, int, int]) -> tuple[float, ...]:
    from hydra2.contracts.rules_canonical import RULES_ID
    from hydra2.contracts.utility import (
        UTILITY_OBJECTIVE,
        UTILITY_TIE_POLICY,
        RawOutcome,
        make_utility_manifest,
        utility,
    )

    manifest = make_utility_manifest(
        utility_id="expected_final_placement_tenhou_4p_hanchan_v1",
        schema_version="1.0.0",
        rules_id=RULES_ID,
        rules_hash="sha256:3042a493280224f533d831f371275b1c96585cf1db5a2e5fb86ec259f403286b",
        objective=UTILITY_OBJECTIVE,
        rank_values=(20.0, 10.0, -10.0, -20.0),
        tie_policy=UTILITY_TIE_POLICY,
        value_min=-100.0,
        value_max=100.0,
        zero_sum=True,
    )
    score_for_rank = {1: 40000, 2: 30000, 3: 20000, 4: 10000}
    final_scores = tuple(score_for_rank[r] for r in ranks)
    outcome = RawOutcome(
        final_scores=final_scores,
        ranks=ranks,
        point_deltas=tuple(s - 25000 for s in final_scores),
        settlements=(),
        rules_id=manifest.rules_id,
        rules_hash=manifest.rules_hash,
    )
    return tuple(utility(outcome, manifest).values)


def test_writer_loader_join_round_trip_matches_utility(tmp_path: Path) -> None:
    dest = tmp_path / "priv"
    ranks = {"dec-0001": [2, 1, 4, 3], "dec-0002": [1, 2, 3, 4]}
    write_privileged_ranks(destination=dest, ranks_by_id=ranks)
    loader = PrivilegedOracleLoader(dest, split="train", verify=True, allow_synthetic=True)
    ids = sorted(ranks)
    joined = join_oracle_targets(ids, loader)
    placement = joined["placement_target"].tolist()
    values = joined["value_target"].tolist()
    assert tuple(joined["placement_target"].shape) == (2, 4)
    assert tuple(joined["value_target"].shape) == (2, 4)
    for row, did in zip(placement, ids, strict=True):
        assert row == [r - 1 for r in ranks[did]]
    for row, did in zip(values, ids, strict=True):
        expected = _expected_utility_values(
            (ranks[did][0], ranks[did][1], ranks[did][2], ranks[did][3])
        )
        assert row == pytest.approx(list(expected))


def test_wall_split_passthrough_stays_in_opaque_dict(tmp_path: Path) -> None:
    dest = tmp_path / "priv"
    write_privileged_ranks(
        destination=dest,
        ranks_by_id={"dec-0001": [1, 2, 3, 4]},
        wall_ids={"dec-0001": "wall-a"},
    )
    table = pq.read_table(dest / "privileged-000.parquet")
    assert table.schema.names == ["decision_id", "privileged_label", "full_world"]
    label = json.loads(table.column("privileged_label").to_pylist()[0])
    assert label["ranks"] == [1, 2, 3, 4]
    assert label["wall_id"] == "wall-a"
    assert label["split"] == "train"


def test_legacy_write_privileged_shards_also_glob_visible(tmp_path: Path) -> None:
    from hydra2.data.parquet import PrivilegedRow

    dest = tmp_path / "priv"
    write_privileged_shards(
        destination=dest,
        rows=[PrivilegedRow(decision_id="dec-0001", privileged_label={"ranks": [1, 2, 3, 4]})],
    )
    assert (dest / "privileged-000.parquet").is_file()
    assert len(sorted(dest.glob("privileged-*.parquet"))) == 1


def test_ranks_from_final_scores_golden_matches_utility(tmp_path: Path) -> None:
    from hydra2.belief.oracle_join import ranks_from_final_scores
    from hydra2.belief.oracle_targets import _value_from_ranks_via_utility
    from hydra2.contracts.rules_manifest import resolve_final_ranks

    scores = [35000, 25000, 15000, 30000]
    ranks = ranks_from_final_scores(scores)
    assert ranks == (1, 3, 4, 2)
    # Strict-order input agrees with Tenhou rules resolution.
    assert ranks == resolve_final_ranks(scores)
    # scores -> ranks -> values hits the utility() fixed point.
    values = _value_from_ranks_via_utility(list(ranks))
    assert values is not None
    assert tuple(values) == pytest.approx((20.0, -10.0, -20.0, 10.0))
    assert tuple(values) == pytest.approx(list(_expected_utility_values(ranks)))

    # Round trip through writer/loader/join with the exported ranks.
    dest = tmp_path / "priv_export"
    write_privileged_ranks(destination=dest, ranks_by_id={"dec-exp-001": list(ranks)})
    loader = PrivilegedOracleLoader(dest, split="train", verify=True, allow_synthetic=True)
    joined = join_oracle_targets(["dec-exp-001"], loader)
    assert joined["placement_target"].tolist() == [[0, 2, 3, 1]]
    assert joined["value_target"].tolist()[0] == pytest.approx([20.0, -10.0, -20.0, 10.0])


@pytest.mark.parametrize(
    "bad_scores",
    [
        [35000, 25000, 15000],  # short
        [35000, 25000, 15000, 30000, 10000],  # long
        [35000, 35000, 15000, 30000],  # tie
        [25000, 25000, 25000, 25000],  # all tied
        [35000, float("inf"), 15000, 30000],  # non-finite
        [35000, float("nan"), 15000, 30000],  # non-finite
        [35000, True, 15000, 30000],  # bool is not a score
        [35000, "25000", 15000, 30000],  # non-numeric
        "35000",  # not a list
        None,
    ],
)
def test_ranks_from_final_scores_strict(bad_scores: object) -> None:
    from hydra2.belief.oracle_join import ranks_from_final_scores

    with pytest.raises(ContractError):
        ranks_from_final_scores(bad_scores)


def test_join_missing_label_fail_closed_and_synthetic_opt_in(tmp_path: Path) -> None:
    dest = tmp_path / "priv_join_flag"
    write_privileged_ranks(destination=dest, ranks_by_id={"dec-present": [1, 2, 3, 4]})
    loader = PrivilegedOracleLoader(dest, split="train", verify=True, allow_synthetic=True)

    # Default: missing id raises (fail closed) for both source kinds.
    with pytest.raises(ContractError, match="allow_synthetic"):
        join_oracle_targets(["dec-present", "dec-absent"], loader)
    store = {"dec-present": {"ranks": [1, 2, 3, 4]}}
    with pytest.raises(ContractError, match="allow_synthetic"):
        join_oracle_targets(["dec-present", "dec-absent"], store)

    # Opt-in: deterministic synthetic join — utility-scale, zero-sum permutation.
    j1 = join_oracle_targets(["dec-absent"], store, allow_synthetic=True)
    j2 = join_oracle_targets(["dec-absent"], store, allow_synthetic=True)
    assert j1["placement_target"].tolist() == j2["placement_target"].tolist()
    assert j1["value_target"].tolist() == j2["value_target"].tolist()
    assert sorted(j1["placement_target"].tolist()[0]) == [0, 1, 2, 3]
    assert sorted(j1["value_target"].tolist()[0]) == pytest.approx(
        sorted([20.0, 10.0, -10.0, -20.0])
    )
    assert sum(j1["value_target"].tolist()[0]) == pytest.approx(0.0)
    # Same derivation through the loader source kind.
    jl = join_oracle_targets(["dec-absent"], loader, allow_synthetic=True)
    assert jl["placement_target"].tolist() == j1["placement_target"].tolist()
    assert jl["value_target"].tolist() == j1["value_target"].tolist()

    # Present-but-invalid labels always raise, even opted in (never mask corruption).
    bad_store = {"dec-bad": {"no_ranks": [0.0]}}
    with pytest.raises(ContractError):
        join_oracle_targets(["dec-bad"], bad_store, allow_synthetic=True)


def test_join_ranks_written_shard_wall_overlap_raises(tmp_path: Path) -> None:
    """Ranks-written shards keep wall provenance inside the label dict; the
    loader-path join must fall back to it so evaluation overlap raises."""
    dest = tmp_path / "priv_wall_overlap"
    write_privileged_ranks(
        destination=dest,
        ranks_by_id={"dec-w1": [1, 2, 3, 4], "dec-w2": [4, 3, 2, 1]},
        wall_ids={"dec-w1": "wall-eval-1", "dec-w2": "wall-train-9"},
    )
    # Ranks-only labels carry no belief signal: synthetic-belief opt-in at
    # construction (join itself stays strict).
    loader = PrivilegedOracleLoader(dest, split="train", verify=True, allow_synthetic=True)
    assert {"wall-eval-1", "wall-train-9"} <= set(loader.wall_ids)
    assert loader.get_oracle_target("dec-w1").wall_id == "wall-eval-1"
    with pytest.raises(ContractError, match="wall leakage"):
        join_oracle_targets(["dec-w1", "dec-w2"], loader, evaluation_wall_ids={"wall-eval-1"})
    joined = join_oracle_targets(["dec-w1", "dec-w2"], loader, evaluation_wall_ids={"wall-other"})
    assert joined["placement_target"].tolist() == [[0, 1, 2, 3], [3, 2, 1, 0]]


def test_loop_maybe_join_forwards_evaluation_walls() -> None:
    """Training loop threads evaluation_wall_ids into join_oracle_targets so
    wall overlap fails closed on the scoring path; default stays unchecked."""
    import inspect

    from hydra2.training.loop_state import TrainingLoopConfig
    from hydra2.training.loop_train import SupervisedLoop

    sig = inspect.signature(SupervisedLoop._maybe_join_oracle_targets)
    assert "evaluation_wall_ids" in sig.parameters
    assert sig.parameters["evaluation_wall_ids"].default is None
    # Minimal instance: the method only reads privileged_source + config.
    loop = object.__new__(SupervisedLoop)
    loop.privileged_source = {
        "dec-loop-1": {
            "ranks": [1, 2, 3, 4],
            "wall_id": "wall-loop-eval",
            "split": "train",
        },
    }
    loop.config = TrainingLoopConfig(w_placement=1.0, w_value=1.0)
    batch = {"_decision_ids": ["dec-loop-1"]}
    with pytest.raises(ContractError, match="wall leakage"):
        loop._maybe_join_oracle_targets(batch, evaluation_wall_ids={"wall-loop-eval"})
    ok = loop._maybe_join_oracle_targets(batch, evaluation_wall_ids={"wall-other"})
    assert ok["placement_target"].tolist() == [[0, 1, 2, 3]]
    default = loop._maybe_join_oracle_targets(dict(batch))
    assert default["placement_target"].tolist() == [[0, 1, 2, 3]]
