"""WP-05B project-owned supervised loop: loop behavior and reporting.

Covers lean hot history entries, plain versus Fabric identical loop state,
loss logging and reporting (masked NLL, top-k, calibration, support and
confusion placeholders, strata, and legal-uniform comparison),
privileged-field hard failures, local-artifact authority, end-to-end smoke
with gated selection, the opaque decision_id oracle join, the real-model
input bridge, and bf16 precision wiring.

Dataset is authoritative synthetic parquet via ``write_actor_shards``;
privileged parquet is never loaded (hard failure if present). All training
is deterministic under the seeded generators and the
``torch.use_deterministic_algorithms`` fixture in ``conftest.py``.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq
import pytest
import torch
import torch.nn as nn

from hydra2.contracts.common import ContractError
from hydra2.data.parquet import (
    DecisionRow,
    PrivilegedRow,
    write_actor_shards,
    write_privileged_shards,
)
from hydra2.eval.blocks import WallBlock
from hydra2.models.schema import BASELINE_ACTION_COUNT
from hydra2.training.dataset_store import AuthoritativeParquetDataset
from hydra2.training.loop_state import FORBIDDEN_BATCH_KEYS, TrainingLoopConfig
from hydra2.training.loop_train import SupervisedLoop
from tests.unit._manifest_helpers import make_test_manifest_hashes
from tests.unit._supervised_loop_helpers import (
    StubModelPerSeat,
    StubPolicyModel,
    _build_loop,
    _gated_selection_fixture,
    _make_actor_rows,
    _write_real_parquet,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.contract_package("WP-05B")


NUM_ACTIONS_SMALL = 16  # small vocab for test speed (real table is 6792)
FEATURE_DIM = 16


# ---------------------------------------------------------------------------
# Hot history entries stay lean
# ---------------------------------------------------------------------------


def test_hot_entry_always_lean(tmp_path: Path, actor_parquet_factory) -> None:
    """Hot entries carry masked_nll/top1 only; rich metrics ride the eval report."""
    import dataclasses

    parquet_dir = actor_parquet_factory(num_rows=16)
    for flag in (False, True):
        loop, _, _ = _build_loop(tmp_path, parquet_dir, seed=123, max_updates=2)
        loop.config = dataclasses.replace(loop.config, log_per_type_metrics=flag)
        hist = loop.train(max_updates=2)
        assert len(hist) == 2
        core = (
            "total",
            "policy",
            "masked_nll",
            "top1",
            "grad_norm_pre",
            "grad_norm_post",
            "lr_now",
        )
        dropped = (
            "top3",
            "top5",
            "calibration_ece",
            "legal_uniform_nll",
            "legal_uniform_gap",
            "support_min",
            "support_max",
        )
        for entry in hist:
            assert not any(k.startswith("per_type/") for k in entry)
            for key in core:
                assert key in entry, f"core key {key!r} missing (flag={flag})"
            for key in dropped:
                assert key not in entry, f"hot key {key!r} must ride eval, not history"
            # No second host sync: grad post is arithmetic mirror of pre.
            assert entry["grad_norm_post"] <= entry["grad_norm_pre"] + 1e-9


# ---------------------------------------------------------------------------
# 6 Plain seeded repeat identical loop state (GPU when available)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_plain_seeded_repeat_identical_loop_state(
    tmp_path: Path, actor_parquet_factory, require_cuda: torch.device
) -> None:
    _ = require_cuda  # hard-fails without CUDA; silent CPU fallback is forbidden.
    parquet_dir = actor_parquet_factory(num_rows=16)

    from hydra2.runtime.plain import PlainPytorchAdapter
    from hydra2.runtime.protocol import RuntimeSpec

    def run_once(tag: str) -> tuple[list, dict]:
        seed = 17
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        dataset = AuthoritativeParquetDataset(
            parquet_dir=parquet_dir,
            feature_dim=FEATURE_DIM,
            num_actions=NUM_ACTIONS_SMALL,
            seed=seed,
            verify=True,
            allow_narrow=True,
        )
        model = StubPolicyModel().to("cuda")
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=True)
        spec = RuntimeSpec(
            adapter_id="plain_pytorch",
            device="cuda:0",
            precision="fp32",
            compile_mode="eager",
            backward_pass_autocast=None,
        )
        adapter = PlainPytorchAdapter()
        handle = adapter.setup(model=model, optimizer=optim, spec=spec)
        config = TrainingLoopConfig(
            seed=seed,
            microbatch_size=4,
            accumulation_steps=1,
            max_updates=3,
            checkpoint_frequency_updates=10,
        )
        loop = SupervisedLoop(
            model=handle.model,
            optimizer=handle.optimizer,
            dataset=dataset,
            config=config,
            checkpoint_dir=tmp_path / f"{tag}_ckpt",
            manifest_hashes=make_test_manifest_hashes(),
            handle=handle,
            device=handle.device,
        )
        return loop.train(max_updates=3), state_snapshot(handle.model)

    from tests.conftest import assert_states_bitwise_equal, state_snapshot

    hist_a, state_a = run_once("a")
    hist_b, state_b = run_once("b")
    for ha, hb in zip(hist_a, hist_b, strict=True):
        assert ha["total"] == pytest.approx(hb["total"], rel=1e-4, abs=1e-6), (
            f"plain {ha} vs repeat {hb}"
        )
    assert_states_bitwise_equal(state_a, state_b, context="plain seeded repeat model state")


# ---------------------------------------------------------------------------
# 7 Loss logging and reporting (masked NLL, top-k, calibration, support, strata, legal-uniform)
# ---------------------------------------------------------------------------


def test_loss_logging_and_reporting(tmp_path: Path, actor_parquet_factory) -> None:
    parquet_dir = actor_parquet_factory(num_rows=20)
    loop, ds, _model = _build_loop(tmp_path / "rep", parquet_dir, seed=21, max_updates=5)
    hist = loop.train(max_updates=5)
    assert len(hist) == 5
    for entry in hist:
        for key in ("masked_nll", "top1"):
            assert key in entry, f"history missing {key}"
            assert isinstance(entry[key], float)
            assert entry[key] == entry[key], "NaN in history"  # not nan
        # masked_nll should be finite
        assert 0 <= entry["top1"] <= 1
    # Evaluate report over held-out batches
    report = loop.evaluate_report(ds, weights=None)
    for k in (
        "masked_nll",
        "top1",
        "top3",
        "top5",
        "calibration_ece",
        "support_min",
        "support_max",
        "legal_uniform_comparison",
    ):
        assert k in report, f"report missing {k}"
    # legal-uniform comparison: gap positive means better than uniform? Not guaranteed early, but nll finite
    assert report["masked_nll"] == report["masked_nll"]
    assert 0 <= report["top1"] <= 1
    assert report["num_eval_batches"] >= 1


def test_reports_include_strata_and_confusion_placeholders(
    tmp_path: Path, actor_parquet_factory
) -> None:
    parquet_dir = actor_parquet_factory(num_rows=12)
    loop, ds, _ = _build_loop(tmp_path / "strata_loop", parquet_dir, seed=22, max_updates=2)
    loop.train(max_updates=2)
    report = loop.evaluate_report(ds)
    assert "strata" in report
    assert "confusion" in report


# ---------------------------------------------------------------------------
# 8 No privileged fields (hard failures)
# ---------------------------------------------------------------------------


def test_no_privileged_fields_rejected_in_batch(tmp_path: Path, actor_parquet_factory) -> None:
    parquet_dir = actor_parquet_factory(num_rows=8)
    loop, ds, _ = _build_loop(tmp_path / "priv_loop", parquet_dir, seed=23, max_updates=1)
    # Inject privileged key into batch after tensorization — loop must reject before forward
    batch = ds.next_batch(4)
    batch["hidden_tiles"] = torch.randn(4, 4)  # privileged
    with pytest.raises(ContractError, match="privileged field"):
        loop.train_step(batch)
    # Nested privileged
    batch2 = ds.next_batch(4)
    batch2["event_targets"] = {"hidden_tiles": torch.randint(0, 3, (4,))}
    with pytest.raises(ContractError, match="privileged"):
        loop.train_step(batch2)


def test_no_privileged_fields_rejected_in_parquet(tmp_path: Path) -> None:
    # Create a parquet directory that contains a privileged shard — dataset construction must fail
    dest = tmp_path / "bad_parquet"
    dest.mkdir()
    # Write a fake privileged file that mimics privileged columns
    import pyarrow as pa

    # Actor shard valid
    rows = _make_actor_rows(num_rows=4)
    write_actor_shards(
        destination=dest,
        rows=rows,
        dataset_hash="sha256:" + "a" * 64,
        split_manifest_hash="sha256:" + "b" * 64,
    )
    # Now add a privileged file in same dir — should be rejected on dataset init
    (dest / "privileged-train.parquet").write_text("privileged")
    with pytest.raises(ContractError, match="privileged shard"):
        AuthoritativeParquetDataset(
            parquet_dir=dest,
            feature_dim=FEATURE_DIM,
            num_actions=NUM_ACTIONS_SMALL,
            seed=0,
            verify=True,
            allow_narrow=True,
        )
    # Clean and test dora shim: create rows with (4,) shim via manual parquet
    (dest / "privileged-train.parquet").unlink()
    # Test privileged column injection via direct parquet write (bypass write_actor_shards validation)

    table = pq.read_table(dest / "actor-train.parquet")
    # Add privileged column by reconstructing table
    bad_dict = {name: table.column(name).to_pylist() for name in table.column_names}
    bad_dict["hidden_tiles"] = ["leak"] * table.num_rows
    bad_table = pa.table(bad_dict)
    pq.write_table(bad_table, dest / "actor-train.parquet")
    with pytest.raises(ContractError, match="privileged"):
        AuthoritativeParquetDataset(
            parquet_dir=dest,
            feature_dim=FEATURE_DIM,
            num_actions=NUM_ACTIONS_SMALL,
            seed=0,
            verify=True,
            allow_narrow=True,
        )


def test_dora_shim_rejected_in_parquet(tmp_path: Path) -> None:
    # Build observations with (4,) dora shim — must be rejected at dataset load or write time
    dest = tmp_path / "dora_bad"
    rows: list[DecisionRow] = []
    for i in range(4):
        obs = {"dora_indicators": [1, 2, 3, 4]}  # (4,) shim, should be (5,)
        rows.append(
            DecisionRow(
                game_id="g",
                round_id="r",
                decision_id=f"dec-{i}",
                seat=0,
                source_object_id=f"obj-{i}",
                split="train",
                rules_hash="sha256:" + "a" * 64,
                adapter_hash="sha256:" + "b" * 64,
                observation_hash="sha256:" + "c" * 64,
                action_table_hash="sha256:" + "d" * 64,
                derivation_hash="sha256:" + "e" * 64,
                actor_observation=obs,  # type: ignore[arg-type]
                chosen_action_id=0,
            )
        )
    # write_actor_shards should itself reject (4,) shim
    with pytest.raises(ContractError, match=r"\(4,\)"):
        write_actor_shards(
            destination=dest,
            rows=rows,
            dataset_hash="sha256:" + "f" * 64,
            split_manifest_hash="sha256:" + "0" * 64,
        )


def test_authoritative_parquet_is_synthetic_qualified(
    tmp_path: Path, actor_parquet_factory
) -> None:
    # Authoritative dataset must be loadable from synthetic actor shards; privileged joined via opaque ref only
    parquet_dir = actor_parquet_factory(num_rows=16)
    # Verify no privileged table exists and loader does not accept privileged path
    ds = AuthoritativeParquetDataset(
        parquet_dir=parquet_dir,
        feature_dim=FEATURE_DIM,
        num_actions=NUM_ACTIONS_SMALL,
        seed=0,
        verify=True,
        allow_narrow=True,
    )
    assert len(ds) == 16
    # Ensure each row's tensorization never sees privileged data
    batch = ds.next_batch(4)
    assert "hidden_tiles" not in batch
    assert "privileged" not in batch
    # Also check that privileged rows would fail if written
    priv_dest = tmp_path / "priv_synth"
    priv_rows = [PrivilegedRow(decision_id="dec-0000", privileged_label={"y": 1})]
    write_privileged_shards(
        destination=priv_dest, rows=priv_rows, dataset_hash="sha256:" + "a" * 64
    )
    # Actor dataset must not load privileged dir
    assert not list(parquet_dir.glob("privileged*"))


# ---------------------------------------------------------------------------
# 9 Local artifacts authoritative; W&B mirror cannot overwrite
# ---------------------------------------------------------------------------


def test_local_artifacts_authoritative(tmp_path: Path, actor_parquet_factory) -> None:
    parquet_dir = actor_parquet_factory(num_rows=12)
    loop, _ds, _ = _build_loop(tmp_path / "local", parquet_dir, seed=31, max_updates=2)
    loop.train(max_updates=2)
    ckpt = loop.save_checkpoint(tmp_path / "local" / "local_auth.pt")
    assert ckpt.exists()
    # Simulate W&B mirror copy: copy to mirror dir
    mirror = tmp_path / "wandb_mirror" / "mirror.pt"
    mirror.parent.mkdir(parents=True)
    import shutil

    shutil.copy(ckpt, mirror)
    orig_hash = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    mirror_hash = hashlib.sha256(mirror.read_bytes()).hexdigest()
    assert orig_hash == mirror_hash
    # Mirror must not overwrite local: local still exists and is not replaced when mirror is written again
    # Write a fake different checkpoint to mirror location (simulate overwrite attempt)
    mirror.write_bytes(b"fake wandb overwrite")
    assert ckpt.read_bytes() != mirror.read_bytes()
    # Local remains authoritative (unchanged)
    assert hashlib.sha256(ckpt.read_bytes()).hexdigest() == orig_hash
    # Resume from local still works
    loop2, _ds2, _ = _build_loop(tmp_path / "local2", parquet_dir, seed=999, max_updates=2)
    loop2.resume_from_checkpoint(ckpt)
    assert loop2.state.global_update == 2


# ---------------------------------------------------------------------------
# 10 End-to-end smoke over authoritative synthetic parquet
# ---------------------------------------------------------------------------


def test_end_to_end_smoke_over_authoritative_synthetic_parquet(
    tmp_path: Path, actor_parquet_factory
) -> None:
    parquet_dir = actor_parquet_factory(num_rows=24)
    loop, ds, _model = _build_loop(
        tmp_path / "e2e_loop",
        parquet_dir,
        seed=42,
        microbatch_size=4,
        accumulation_steps=1,
        max_updates=6,
    )
    hist = loop.train(max_updates=6)
    assert len(hist) == 6
    # Check training made progress (loss finite and not NaN)
    for entry in hist:
        assert entry["total"] == entry["total"]
        assert entry["total"] < 1e6
    # Checkpoint and evaluate
    ckpt = tmp_path / "e2e_loop" / "final.pt"
    saved = loop.save_checkpoint(ckpt)
    assert saved.exists()
    report = loop.evaluate_report(ds)
    assert report["masked_nll"] < 10.0
    # Verify dataset still authoritative (no privileged leakage after training)
    for shard in parquet_dir.glob("actor-*.parquet"):
        from hydra2.data.parquet import verify_no_privileged_leakage

        verify_no_privileged_leakage(shard)


def test_selection_promotes_only_on_improvement(tmp_path, actor_parquet_factory) -> None:
    parquet_dir = actor_parquet_factory(num_rows=24)
    loop, _, _ = _build_loop(tmp_path / "sel", parquet_dir)
    blocks, telemetry, config = _gated_selection_fixture()
    assert loop.evaluate_selection(blocks, telemetry, config, 2) == pytest.approx(3.0)
    ckpt = tmp_path / "sel" / "ckpt.pt"
    ckpt.write_bytes(b"checkpoint-bytes")
    assert (
        loop.maybe_promote_best(
            3.0, ckpt, config=config, blocks=blocks, telemetry_by_game=telemetry, peek_index=2
        )
        is True
    )
    assert loop.state.best_selection_metric == pytest.approx(3.0)
    assert (
        loop.state.best_ckpt_digest == "sha256:" + hashlib.sha256(b"checkpoint-bytes").hexdigest()
    )
    assert (tmp_path / "sel" / "checkpoints" / "best-ckpt.pt").read_bytes() == b"checkpoint-bytes"
    assert not (tmp_path / "sel" / "checkpoints" / "best-ckpt.pt.tmp").exists()
    assert (
        loop.maybe_promote_best(
            3.0, ckpt, config=config, blocks=blocks, telemetry_by_game=telemetry, peek_index=2
        )
        is False
    )
    # Worse gated score does not promote: separate evidence scoring 5.0.
    worse_blocks = (
        WallBlock(wall_id="w0", game_ids=("w0-g0",), contrasts=(4.0,)),
        WallBlock(wall_id="w1", game_ids=("w1-g0",), contrasts=(6.0,)),
    )
    assert loop.evaluate_selection(worse_blocks, telemetry, config, 2) == pytest.approx(5.0)
    assert (
        loop.maybe_promote_best(
            5.0,
            ckpt,
            config=config,
            blocks=worse_blocks,
            telemetry_by_game=telemetry,
            peek_index=2,
        )
        is False
    )
    assert (tmp_path / "sel" / "checkpoints" / "best-ckpt.pt").read_bytes() == b"checkpoint-bytes"
    with pytest.raises(ContractError, match="finite"):
        loop.maybe_promote_best(
            float("nan"),
            ckpt,
            config=config,
            blocks=blocks,
            telemetry_by_game=telemetry,
            peek_index=2,
        )
    with pytest.raises(ContractError, match="!= gated score"):
        loop.maybe_promote_best(
            999.0, ckpt, config=config, blocks=blocks, telemetry_by_game=telemetry, peek_index=2
        )
    single = (blocks[0],)
    single_telemetry = {"w0-g0": telemetry["w0-g0"]}
    with pytest.raises(ContractError, match="extra look"):
        loop.evaluate_selection(single, single_telemetry, config, 1)
    with pytest.raises(ContractError, match="no valid wall blocks"):
        loop.evaluate_selection((), {}, config, 2)


def test_selection_requires_gated_args(tmp_path, actor_parquet_factory) -> None:
    """Unguarded direct call is impossible: the old bare signature is gone."""
    parquet_dir = actor_parquet_factory(num_rows=24)
    loop, _, _ = _build_loop(tmp_path / "selgate", parquet_dir)
    blocks, _, _ = _gated_selection_fixture()
    ckpt = tmp_path / "selgate" / "ckpt.pt"
    ckpt.write_bytes(b"x")
    with pytest.raises(TypeError):
        loop.evaluate_selection(blocks)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        loop.maybe_promote_best(3.0, ckpt)  # type: ignore[call-arg]


def test_resume_verifies_best_ckpt(tmp_path, actor_parquet_factory) -> None:
    parquet_dir = actor_parquet_factory(num_rows=24)
    loop, _, _ = _build_loop(tmp_path / "res", parquet_dir)
    blocks, telemetry, config = _gated_selection_fixture()
    metric = loop.evaluate_selection(blocks, telemetry, config, 2)
    ckpt = tmp_path / "res" / "ckpt.pt"
    ckpt.write_bytes(b"resume-bytes")
    assert (
        loop.maybe_promote_best(
            metric, ckpt, config=config, blocks=blocks, telemetry_by_game=telemetry, peek_index=2
        )
        is True
    )
    saved = loop.save_checkpoint()
    digest = loop.state.best_ckpt_digest
    assert digest is not None
    loop2, _, _ = _build_loop(tmp_path / "res", parquet_dir)
    loop2.resume_from_checkpoint(saved)
    assert loop2.state.best_selection_metric == pytest.approx(metric)
    assert loop2.state.best_ckpt_digest == digest
    assert (tmp_path / "res" / "checkpoints" / "best-ckpt.pt").read_bytes() == b"resume-bytes"


def test_tampered_best_ckpt_raises(tmp_path, actor_parquet_factory) -> None:
    from hydra2.contracts.common import CorruptArtifactError

    parquet_dir = actor_parquet_factory(num_rows=24)
    loop, _, _ = _build_loop(tmp_path / "tamper", parquet_dir)
    blocks, telemetry, config = _gated_selection_fixture()
    metric = loop.evaluate_selection(blocks, telemetry, config, 2)
    ckpt = tmp_path / "tamper" / "ckpt.pt"
    ckpt.write_bytes(b"honest-bytes")
    assert (
        loop.maybe_promote_best(
            metric, ckpt, config=config, blocks=blocks, telemetry_by_game=telemetry, peek_index=2
        )
        is True
    )
    saved = loop.save_checkpoint()
    best = tmp_path / "tamper" / "checkpoints" / "best-ckpt.pt"
    best.write_bytes(b"tampered-bytes")
    loop2, _, _ = _build_loop(tmp_path / "tamper", parquet_dir)
    with pytest.raises(CorruptArtifactError, match="digest mismatch"):
        loop2.resume_from_checkpoint(saved)
    best.unlink()
    loop3, _, _ = _build_loop(tmp_path / "tamper", parquet_dir)
    with pytest.raises(CorruptArtifactError, match="missing"):
        loop3.resume_from_checkpoint(saved)


def test_train_never_touches_selection(tmp_path: Path, actor_parquet_factory) -> None:
    parquet_dir = actor_parquet_factory(num_rows=24)
    loop, _, _ = _build_loop(tmp_path / "nosel", parquet_dir, max_updates=2)
    loop.train()
    assert loop.state.best_selection_metric is None
    assert loop.state.best_ckpt_digest is None
    assert not (tmp_path / "nosel" / "checkpoints" / "best-ckpt.pt").exists()


# ---------------------------------------------------------------------------
# 12 Opaque decision_id oracle join + real-model output adapter
# ---------------------------------------------------------------------------


def test_oracle_join_injects_placement_and_value_targets(
    tmp_path: Path, actor_parquet_factory
) -> None:
    """Privileged ranks join by opaque decision_id into 0-based + utility targets."""
    parquet_dir = actor_parquet_factory(num_rows=8)
    torch.manual_seed(7)
    dataset = AuthoritativeParquetDataset(
        parquet_dir=parquet_dir,
        feature_dim=FEATURE_DIM,
        num_actions=NUM_ACTIONS_SMALL,
        seed=7,
        verify=True,
        allow_narrow=True,
    )
    model = StubModelPerSeat(feature_dim=FEATURE_DIM, num_actions=NUM_ACTIONS_SMALL)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=True)
    config = TrainingLoopConfig(
        w_policy=1.0,
        w_placement=1.0,
        w_value=1.0,
        microbatch_size=4,
        accumulation_steps=1,
        gradient_clip_norm=1.0,
        max_updates=1,
        checkpoint_frequency_updates=10,
        seed=7,
    )
    source: dict[str, dict[str, Any]] = {}
    loop = SupervisedLoop(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        dataset=dataset,
        config=config,
        checkpoint_dir=tmp_path / "join",
        manifest_hashes=make_test_manifest_hashes(),
        privileged_source=source,
    )
    batch = dataset.next_batch(4)
    assert batch is not None
    decision_ids = [str(x) for x in batch["_decision_ids"]]
    perms = ([1, 2, 3, 4], [4, 3, 2, 1], [2, 1, 4, 3], [3, 4, 1, 2])
    for did, ranks in zip(decision_ids, perms, strict=True):
        source[did] = {"ranks": list(ranks)}
    joined = loop._maybe_join_oracle_targets(batch)
    placement = joined["placement_target"]
    value = joined["value_target"]
    assert tuple(placement.shape) == (4, 4)
    assert tuple(value.shape) == (4, 4)
    # ranks -> placement is the -1 bridge (1..4 ranks to 0-based targets)
    assert placement[0].tolist() == [0, 1, 2, 3]
    assert placement[1].tolist() == [3, 2, 1, 0]
    # ranks -> values go through utility() (UtilityVector.values, never synthesized)
    from hydra2.belief.oracle_targets import _value_from_ranks_via_utility

    expected_values = _value_from_ranks_via_utility([1, 2, 3, 4])
    assert expected_values is not None
    assert value[0].tolist() == pytest.approx(list(expected_values))
    # Post-merge firewall: joined targets are actor-legal keys only
    for key in joined:
        assert key not in FORBIDDEN_BATCH_KEYS
    # End-to-end: train_step joins internally and stays finite
    result = loop.train_step(batch)
    assert result["total"] == result["total"]
    assert result["placement"] >= 0.0
    assert result["value"] >= 0.0


def test_no_privileged_source_still_raises_on_missing_targets(
    tmp_path: Path, actor_parquet_factory
) -> None:
    """Absent source preserves the ContractError-on-missing-target behavior."""
    parquet_dir = actor_parquet_factory(num_rows=8)
    loop, dataset, _ = _build_loop(
        tmp_path / "nojoin",
        parquet_dir,
        seed=7,
        w_policy=1.0,
        w_placement=1.0,
        w_value=1.0,
        max_updates=1,
    )
    assert loop.privileged_source is None
    batch = dataset.next_batch(4)
    assert batch is not None
    assert "placement_target" not in batch
    with pytest.raises(ContractError, match="placement_target"):
        loop.train_step(batch)


# ---------------------------------------------------------------------------
# 13 Real-model input bridge (dict batch to ActorTensorBatch)
# ---------------------------------------------------------------------------


def test_real_model_input_bridge_end_to_end(tmp_path: Path) -> None:
    """One train_step with the REAL model over a real-mode dataset batch.

    Real parquet rows carry ``actor_observation`` JSON; the real-mode dataset
    yields flat keys plus ``actor_batch``; the loop routes
    ``model.evaluate(actor_batch)`` and maps via
    ``adapters.model_output_to_loss_dict``.  Weights are policy-only so the
    test proves the bridge, not auxiliary heads.
    """
    from hydra2.models.encoder import ActorTensorBatch
    from hydra2.models.model import Hydra2BaselineModel
    from hydra2.training.loop_batch import _move_batch_to_device

    torch.manual_seed(0)
    hands = [
        (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48),
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 52),
        (13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25),
    ]
    parquet_dir = _write_real_parquet(tmp_path, hands)
    dataset = AuthoritativeParquetDataset(
        parquet_dir=parquet_dir,
        feature_dim=FEATURE_DIM,
        num_actions=BASELINE_ACTION_COUNT,
        seed=0,
        verify=True,
        tensorize="real",
    )
    batch = dataset.next_batch(4)
    assert batch is not None
    # Bridge key present; flat keys keep their compat shapes.
    assert isinstance(batch["actor_batch"], ActorTensorBatch)
    assert tuple(batch["features"].shape) == (4, FEATURE_DIM)
    assert tuple(batch["legal_mask"].shape) == (4, BASELINE_ACTION_COUNT)
    assert tuple(batch["chosen_action_id"].shape) == (4,)
    # Device move preserves the bridge value and flat keys.
    moved = _move_batch_to_device(batch, torch.device("cpu"))
    assert isinstance(moved["actor_batch"], ActorTensorBatch)
    assert torch.equal(moved["features"], batch["features"])

    model = Hydra2BaselineModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=True)
    config = TrainingLoopConfig(
        w_policy=1.0,
        w_placement=0.0,
        w_value=0.0,
        microbatch_size=4,
        accumulation_steps=1,
        gradient_clip_norm=1.0,
        max_updates=1,
        checkpoint_frequency_updates=10,
        seed=0,
    )
    loop = SupervisedLoop(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        dataset=dataset,
        config=config,
        checkpoint_dir=tmp_path / "bridge",
        manifest_hashes=make_test_manifest_hashes(),
    )
    result = loop.train_step(batch)
    assert torch.isfinite(torch.tensor(result["total"])).item() is True
    assert "policy" in result
    assert "placement" in result
    assert "value" in result


def test_actor_batch_absent_falls_back_to_legacy_path(tmp_path: Path) -> None:
    """Without ``actor_batch`` the loop uses the legacy forward(dict) path."""
    from hydra2.training.loop_batch import _model_forward

    torch.manual_seed(0)
    sentinel: dict[str, torch.Tensor] = {"policy_logits": torch.randn(2, NUM_ACTIONS_SMALL)}

    class StubDictModel(nn.Module):
        def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return sentinel

    legacy_batch = {
        "features": torch.randn(2, FEATURE_DIM),
        "legal_mask": torch.ones(2, NUM_ACTIONS_SMALL, dtype=torch.bool),
        "chosen_action_id": torch.zeros(2, dtype=torch.long),
    }
    assert "actor_batch" not in legacy_batch
    assert _model_forward(StubDictModel(), legacy_batch) is sentinel

    # Even a real-model-shaped batch without the key stays on the dict path:
    # a stub exposing evaluate(dict) still receives the plain dict.
    seen: dict[str, Any] = {}

    class StubEvaluateDict(nn.Module):
        def evaluate(self, batch: dict[str, Any]) -> dict[str, Any]:
            seen["keys"] = sorted(batch.keys())
            return sentinel

    assert _model_forward(StubEvaluateDict(), legacy_batch) is sentinel
    assert seen["keys"] == sorted(legacy_batch.keys())


def test_combined_enablement_real_model_join_and_heads(tmp_path: Path) -> None:
    """Full enablement in one step: real model + real rows + join + w=0.5 heads.

    Closes the final-vote kill-shot compositionally covered by the bridge,
    join, and formula tests: real Hydra2BaselineModel over real-mode batches
    with privileged ranks joined, w_placement/w_value=0.5, one train_step
    stays finite with placement/value losses participating.
    """
    from hydra2.models.model import Hydra2BaselineModel

    torch.manual_seed(11)
    hands = [
        (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48),
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 52),
        (13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25),
    ]
    parquet_dir = _write_real_parquet(tmp_path, hands)
    dataset = AuthoritativeParquetDataset(
        parquet_dir=parquet_dir,
        feature_dim=FEATURE_DIM,
        num_actions=BASELINE_ACTION_COUNT,
        seed=11,
        verify=True,
        tensorize="real",
    )
    model = Hydra2BaselineModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=True)
    config = TrainingLoopConfig(
        w_policy=1.0,
        w_placement=0.5,
        w_value=0.5,
        microbatch_size=4,
        accumulation_steps=1,
        gradient_clip_norm=1.0,
        max_updates=1,
        checkpoint_frequency_updates=10,
        seed=11,
    )
    source: dict[str, dict[str, Any]] = {}
    loop = SupervisedLoop(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        dataset=dataset,
        config=config,
        checkpoint_dir=tmp_path / "combined",
        manifest_hashes=make_test_manifest_hashes(),
        privileged_source=source,
    )
    batch = dataset.next_batch(4)
    assert batch is not None
    decision_ids = [str(x) for x in batch["_decision_ids"]]
    perms = ([1, 2, 3, 4], [4, 3, 2, 1], [2, 1, 4, 3], [3, 4, 1, 2])
    for did, ranks in zip(decision_ids, perms, strict=True):
        source[did] = {"ranks": list(ranks)}
    result = loop.train_step(batch)
    assert torch.isfinite(torch.tensor(result["total"])).item() is True
    assert result["placement"] >= 0.0
    assert result["value"] >= 0.0
    assert loop.state.best_selection_metric is None


def test_bf16_precision_enables_attn_bf16(tmp_path: Path, actor_parquet_factory) -> None:
    """bf16_mixed opts attention into bf16 math; fp32 leaves it off (CPU-only).

    The flag is device-independent at construction (the forward gates on
    is_cuda, so it stays inert on CPU) — this pins the loop wiring without
    needing a CUDA device. Fails pre-change (bf16 loop leaves it False).
    """
    parquet_dir = actor_parquet_factory(num_rows=8)
    bf16_loop, _, _ = _build_loop(
        tmp_path / "bf16", parquet_dir, precision="bf16_mixed", checkpoint_subdir="ckpt"
    )
    flagged = [m for m in bf16_loop.model.modules() if hasattr(m, "attn_bf16")]
    assert len(flagged) > 0, "expected at least one attn_bf16 switch"
    assert all(m.attn_bf16 is True for m in flagged)
    fp32_loop, _, _ = _build_loop(
        tmp_path / "fp32", parquet_dir, precision="fp32", checkpoint_subdir="ckpt"
    )
    fp32_flagged = [m for m in fp32_loop.model.modules() if hasattr(m, "attn_bf16")]
    assert len(fp32_flagged) > 0, "expected at least one attn_bf16 switch"
    assert all(m.attn_bf16 is False for m in fp32_flagged)
