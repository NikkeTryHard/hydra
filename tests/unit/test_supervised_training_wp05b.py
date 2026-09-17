"""WP-05B supervised training: deterministic parquet training and resume.

Covers fused-kernel agreement with the eager path, value-head training over
authoritative synthetic parquet, deterministic training and shuffle order,
bitwise checkpoint resume (model, optimizer, scheduler, RNG, sampler, and
manifest), project-owned optimizer, scheduler, and accumulation,
real-observation tensorization alongside the synthetic stand-in, oracle
wall-ledger overlap, and narrow-vocab fail-closed.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pytest
import torch

from hydra2.contracts.common import ContractError
from hydra2.models.schema import BASELINE_ACTION_COUNT
from hydra2.runtime.checkpoint import hash_state_tree
from hydra2.training.dataset_encode import (
    encode_observation_rows,
    tensorize_actor_row,
)
from hydra2.training.dataset_store import (
    AuthoritativeParquetDataset,
)
from hydra2.training.loop_state import TrainingLoopConfig
from hydra2.training.loop_train import SupervisedLoop
from hydra2.training.objectives_loss import compute_supervised_loss, masked_cross_entropy
from tests.unit._manifest_helpers import make_test_manifest_hashes
from tests.unit._supervised_loop_helpers import (
    StubModelPerSeat,
    StubPolicyModel,
    _build_loop,
    _make_real_observation,
    _real_row_dict,
    _write_real_parquet,
    _write_synthetic_parquet,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.contract_package("WP-05B")


NUM_ACTIONS_SMALL = 16  # small vocab for test speed (real table is 6792)
FEATURE_DIM = 16


# ---------------------------------------------------------------------------
# Fused kernels agree with the eager path
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_fused_ce_matches_eager_forward_backward(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fused Triton CE agrees with the eager path (production fast path).

    Guards the masked_cross_entropy CUDA branch: identical loss (tight),
    close grads (reduction-order noise only), exact-zero illegal grads.
    Fails without the fused kernel (no reference to agree with) and on any
    formula drift in either path.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    fused_ce = pytest.importorskip("hydra2.training.fused_ce")
    if not fused_ce.TRITON_AVAILABLE:
        pytest.skip("needs triton")
    gen = torch.Generator(device="cpu").manual_seed(20260912)
    b, a = 64, 1536
    logits = (torch.randn(b, a, generator=gen) * 3).to(torch.bfloat16).cuda()
    legal = (torch.rand(b, a, generator=gen) < 0.02).cuda()
    legal[torch.arange(b), torch.randint(a, (b,), generator=gen)] = True
    tgt_list = []
    for i in range(b):
        li = torch.nonzero(legal[i], as_tuple=False).squeeze(1)
        tgt_list.append(int(li[int(torch.randint(len(li), (1,), generator=gen).item())]))
    targets = torch.tensor(tgt_list, dtype=torch.long).cuda()
    eps = 0.03
    # Eager reference: force the fallback path via the availability flag
    # (same function, untaken branch — otherwise this test is a tautology).
    monkeypatch.setattr(fused_ce, "TRITON_AVAILABLE", False)
    le = logits.detach().clone().requires_grad_(True)
    loss_e = masked_cross_entropy(le, targets, legal, label_smoothing=eps)
    loss_e.backward()
    monkeypatch.setattr(fused_ce, "TRITON_AVAILABLE", True)
    lf = logits.detach().clone().requires_grad_(True)
    row = fused_ce.fused_masked_ce_row_losses(lf, legal, targets, eps)
    loss_f = row.mean()
    loss_f.backward()
    d = (le.grad.detach().float() - lf.grad.detach().float()).abs()
    assert float(d.max()) < 1e-3, float(d.max())
    assert float(lf.grad.detach()[~legal].abs().max()) == 0.0
    # Production wiring: masked_cross_entropy's fused branch must return the
    # same mean (pins branch-taken + reduction, not just the op).
    loss_w = masked_cross_entropy(logits.detach(), targets, legal, label_smoothing=eps)
    assert abs(float(loss_w) - float(loss_f)) < 1e-6, (float(loss_w), float(loss_f))


@pytest.mark.gpu
def test_fused_hot_scalars_matches_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fused reporting NLL/top1 agrees with the eager path.

    Guards the compute_hot_scalars CUDA branch: identical NLL (tight),
    exactly equal top1. Fails without the fused kernel and on formula
    drift; ties are excluded (argmax/first-max order documented).
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    fused_ce = pytest.importorskip("hydra2.training.fused_ce")
    if not fused_ce.TRITON_AVAILABLE:
        pytest.skip("needs triton")
    from hydra2.training.objectives_metrics import compute_hot_scalars

    gen = torch.Generator(device="cpu").manual_seed(20260913)
    b, a = 64, 1536
    logits = (torch.randn(b, a, generator=gen) * 3).to(torch.bfloat16).cuda()
    legal = (torch.rand(b, a, generator=gen) < 0.02).cuda()
    legal[torch.arange(b), torch.randint(a, (b,), generator=gen)] = True
    tgt_list = []
    for i in range(b):
        li = torch.nonzero(legal[i], as_tuple=False).squeeze(1)
        tgt_list.append(int(li[int(torch.randint(len(li), (1,), generator=gen).item())]))
    targets = torch.tensor(tgt_list, dtype=torch.long).cuda()
    monkeypatch.setattr(fused_ce, "TRITON_AVAILABLE", False)
    eager = compute_hot_scalars(logits.detach(), targets, legal)
    monkeypatch.setattr(fused_ce, "TRITON_AVAILABLE", True)
    fused = compute_hot_scalars(logits.detach(), targets, legal)
    assert abs(eager["masked_nll"] - fused["masked_nll"]) < 1e-4, (eager, fused)
    assert eager["top1"] == fused["top1"]


# ---------------------------------------------------------------------------
# Value-head training logs exact auxiliary losses
# ---------------------------------------------------------------------------


def test_train_with_value_head_logs_value(tmp_path: Path, actor_parquet_factory) -> None:
    parquet_dir = actor_parquet_factory(num_rows=16)
    loop, dataset, model = _build_loop(
        tmp_path, parquet_dir, seed=123, w_policy=1.0, w_value=0.5, max_updates=2
    )
    assert loop.config.objective_weights()["w_value"] == 0.5
    orig_next_batch = dataset.next_batch

    def _next_with_value(batch_size: int):
        batch = orig_next_batch(batch_size)
        assert batch is not None
        n = int(batch["chosen_action_id"].shape[0])
        batch["value_target"] = torch.zeros(n, 4)
        return batch

    dataset.next_batch = _next_with_value  # type: ignore[method-assign]
    hist = loop.train(max_updates=2)
    assert len(hist) == 2
    for entry in hist:
        assert "value" in entry, f"loss_history entry missing value: {entry}"
        assert isinstance(entry["value"], float)
        assert entry["value"] == entry["value"], "NaN value in history"
        assert entry["value"] >= 0.0
        assert entry["total"] == entry["total"], "NaN total in history"
        # total = policy_nll + w_value * value with policy_nll >= 0, so
        # total must cover the value contribution (means over the window).
        assert entry["total"] + 1e-6 >= 0.5 * entry["value"]
    # Total includes the value term: exact formula parity on a fresh batch.
    batch = dataset.next_batch(4)
    assert batch is not None and "value_target" in batch
    batch = {k: (v.to(loop.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    model_out = model(batch)
    weights = loop.config.objective_weights()
    losses = compute_supervised_loss(model_out, batch, weights)
    expected_total = masked_cross_entropy(
        model_out["policy_logits"],
        batch["chosen_action_id"],
        batch["legal_mask"],
        label_smoothing=weights["label_smoothing"],
    ) + 0.5 * torch.nn.functional.mse_loss(model_out["value_vector"], batch["value_target"])
    assert torch.allclose(losses["total"], expected_total, atol=1e-6)
    assert torch.allclose(
        losses["value"],
        torch.nn.functional.mse_loss(model_out["value_vector"], batch["value_target"]),
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# 3 Deterministic training over authoritative synthetic parquet
# ---------------------------------------------------------------------------


def test_deterministic_training_over_authoritative_synthetic_parquet(
    tmp_path: Path, actor_parquet_factory
) -> None:
    parquet_dir = actor_parquet_factory(num_rows=24)

    def run_once(seed: int) -> list[dict[str, float]]:
        loop, _, _ = _build_loop(tmp_path / f"run{seed}", parquet_dir, seed=42, max_updates=4)
        # need fresh parquet copy per run? Reuse same dir but loop clones dataset ordering via seed
        # To keep parquet identical, pass same dir; dataset permutation seeded.
        # But we built loop with tmp_path/run{seed} ckpt dir differ, dataset same dir.
        # For identical comparison we want two loops both seeded 42 but reading same parquet.
        # So override dataset to use same parquet_dir but separate loop instance
        # Actually _build_loop above created dataset from parquet_dir; but checkpoint_dir differs.
        # We need to recreate loop with same seed but isolated ckpt dir; the function already uses seed=42 for both.
        hist = loop.train(max_updates=4)
        return hist

    # First run
    torch.use_deterministic_algorithms(True)
    hist1 = run_once(1)
    # Need to recreate parquet dataset for second run (same file) but fresh loop
    # Use same helper but different ckpt dir — deterministic should hold even though tmp_path differ
    # Instead do second run with same seed 42 but new tmp_path
    loop2, _, _ = _build_loop(tmp_path / "run_second", parquet_dir, seed=42, max_updates=4)
    hist2 = loop2.train(max_updates=4)
    assert len(hist1) == len(hist2) == 4
    for h1, h2 in zip(hist1, hist2, strict=True):
        assert h1["total"] == pytest.approx(h2["total"], abs=1e-6), (
            f"nondeterministic loss {h1} vs {h2}"
        )
        assert h1["masked_nll"] == pytest.approx(h2["masked_nll"], abs=1e-6)
    # Also verify parquet was authoritative (actor shards only)
    assert (parquet_dir / "actor-train.parquet").exists()
    assert not list(parquet_dir.glob("privileged*")), (
        "privileged shards must not exist in actor dataset"
    )


def test_deterministic_requires_same_shuffle_order(tmp_path: Path, actor_parquet_factory) -> None:
    # Ensure dataset ordering is deterministic: two datasets with same seed have same order
    parquet_dir = actor_parquet_factory(num_rows=12)
    ds1 = AuthoritativeParquetDataset(
        parquet_dir=parquet_dir,
        feature_dim=FEATURE_DIM,
        num_actions=NUM_ACTIONS_SMALL,
        seed=999,
        verify=True,
        allow_narrow=True,
    )
    ds2 = AuthoritativeParquetDataset(
        parquet_dir=parquet_dir,
        feature_dim=FEATURE_DIM,
        num_actions=NUM_ACTIONS_SMALL,
        seed=999,
        verify=True,
        allow_narrow=True,
    )
    b1 = ds1.next_batch(4)
    b2 = ds2.next_batch(4)
    assert torch.equal(b1["features"], b2["features"])
    assert torch.equal(b1["legal_mask"], b2["legal_mask"])


# ---------------------------------------------------------------------------
# 4 Checkpoint resume restores bitwise (model, optimizer, scheduler, RNG, sampler, manifest)
# ---------------------------------------------------------------------------


def test_checkpoint_resume_bitwise(tmp_path: Path, actor_parquet_factory) -> None:
    parquet_dir = actor_parquet_factory(num_rows=24)
    # Train uninterrupted 8 updates
    loop_full, _ds_full, model_full = _build_loop(
        tmp_path / "full", parquet_dir, seed=7, max_updates=8
    )
    # Save initial state for comparison
    hash_state_tree(model_full.state_dict())
    hist_full = loop_full.train(max_updates=8)
    final_state_full = hash_state_tree(model_full.state_dict())

    # Train 4, checkpoint, resume 4
    loop_part, ds_part, model_part = _build_loop(
        tmp_path / "part", parquet_dir, seed=7, max_updates=8
    )
    hist_first4 = loop_part.train(max_updates=4)
    ckpt_path = loop_part.save_checkpoint(tmp_path / "part" / "ckpts" / "mid.pt")
    assert ckpt_path.exists()
    # Capture sampler cursor at checkpoint
    cursor_at_ckpt = loop_part.state.sampler_cursor
    # Create new loop instance to simulate fresh process resume (same seed, but will be overwritten by resume)
    loop_resumed, ds_resumed, model_resumed = _build_loop(
        tmp_path / "resumed", parquet_dir, seed=999, max_updates=8
    )
    # Before resume, model states differ (seed 999 vs 7)
    assert hash_state_tree(model_resumed.state_dict()) != hash_state_tree(model_part.state_dict())
    loop_resumed.resume_from_checkpoint(ckpt_path)
    # After resume, states must be bitwise identical
    assert hash_state_tree(model_resumed.state_dict()) == hash_state_tree(model_part.state_dict())
    # Also sampler cursor must have been restored
    assert loop_resumed.state.sampler_cursor == cursor_at_ckpt
    assert ds_resumed.get_sampler_state()["offset"] == ds_part.get_sampler_state()["offset"]
    # Continue training 4 more
    hist_resumed_second4 = loop_resumed.train(max_updates=4)
    # Histories: first 4 from part + second 4 from resumed should equal full's 8
    hist_combined = hist_first4 + hist_resumed_second4
    assert len(hist_combined) == len(hist_full) == 8
    for i, (hc, hf) in enumerate(zip(hist_combined, hist_full, strict=True)):
        assert hc["total"] == pytest.approx(hf["total"], abs=1e-6), (
            f"mismatch at update {i}: {hc} vs {hf}"
        )
    # Final model state bitwise identical
    assert hash_state_tree(model_resumed.state_dict()) == final_state_full


def test_checkpoint_manifest_verified_before_mutation(
    tmp_path: Path, actor_parquet_factory
) -> None:
    parquet_dir = actor_parquet_factory(num_rows=12)
    loop, _, _ = _build_loop(tmp_path / "mloop", parquet_dir, seed=11, max_updates=2)
    loop.train(max_updates=2)
    ckpt = loop.save_checkpoint(tmp_path / "mloop" / "ckpt.pt")
    # Try to resume with mismatched dataset hash — should fail before mutation
    bad_hashes = dict(loop.manifest_hashes)
    bad_hashes["dataset_manifest_hash"] = "sha256:" + "f" * 64
    loop2, _, model2 = _build_loop(tmp_path / "bad", parquet_dir, seed=11, max_updates=2)
    loop2.manifest_hashes = bad_hashes  # inject wrong expected hash
    before_state = hash_state_tree(model2.state_dict())
    with pytest.raises(ContractError):
        loop2.resume_from_checkpoint(ckpt)
    # Model must be unchanged (no partial mutation)
    assert hash_state_tree(model2.state_dict()) == before_state


def test_manifest_hashes_required_and_validated(tmp_path: Path, actor_parquet_factory) -> None:
    parquet_dir = actor_parquet_factory(num_rows=8)
    dataset = AuthoritativeParquetDataset(
        parquet_dir=parquet_dir,
        feature_dim=FEATURE_DIM,
        num_actions=NUM_ACTIONS_SMALL,
        seed=0,
        verify=True,
        allow_narrow=True,
    )
    model = StubPolicyModel(feature_dim=FEATURE_DIM, num_actions=NUM_ACTIONS_SMALL)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=True)
    config = TrainingLoopConfig(
        microbatch_size=4,
        accumulation_steps=1,
        max_updates=1,
        checkpoint_frequency_updates=10,
        seed=0,
    )
    with pytest.raises(ContractError):
        SupervisedLoop(
            model=model,
            optimizer=optimizer,
            dataset=dataset,
            config=config,
            checkpoint_dir=tmp_path / "ckpt-none",
            manifest_hashes=None,
        )
    incomplete = make_test_manifest_hashes()
    del incomplete["dataset_manifest_hash"]
    with pytest.raises(ContractError):
        SupervisedLoop(
            model=model,
            optimizer=optimizer,
            dataset=dataset,
            config=config,
            checkpoint_dir=tmp_path / "ckpt-missing",
            manifest_hashes=incomplete,
        )
    malformed = make_test_manifest_hashes()
    malformed["run_spec_hash"] = "not-a-digest"
    with pytest.raises(ContractError):
        SupervisedLoop(
            model=model,
            optimizer=optimizer,
            dataset=dataset,
            config=config,
            checkpoint_dir=tmp_path / "ckpt-malformed",
            manifest_hashes=malformed,
        )


def test_sampler_cursor_tracked_for_resume(tmp_path: Path, actor_parquet_factory) -> None:
    parquet_dir = actor_parquet_factory(num_rows=16)
    loop, ds, _ = _build_loop(
        tmp_path / "curloop",
        parquet_dir,
        seed=5,
        microbatch_size=4,
        accumulation_steps=1,
        max_updates=3,
    )
    loop.train(max_updates=3)
    # After 3 updates with microbatch 4, we consumed 12 rows (wrapping not yet because total 16)
    state = loop.state
    sampler = ds.get_sampler_state()
    assert sampler["offset"] == 12
    assert state.sampler_cursor["offset"] == 12
    # Checkpoint and resume must preserve cursor
    ckpt = loop.save_checkpoint(tmp_path / "curloop" / "ckpt.pt")
    loop2, ds2, _ = _build_loop(
        tmp_path / "curloop2", parquet_dir, seed=999, microbatch_size=4, max_updates=3
    )
    loop2.resume_from_checkpoint(ckpt)
    assert ds2.get_sampler_state()["offset"] == 12
    assert loop2.state.sampler_cursor["offset"] == 12


# ---------------------------------------------------------------------------
# 5 Project-owned optimizer/scheduler/accumulation/checkpoint
# ---------------------------------------------------------------------------


def test_project_owned_optimizer_scheduler_accumulation_checkpoint(
    tmp_path: Path, actor_parquet_factory
) -> None:
    parquet_dir = actor_parquet_factory(num_rows=16)
    # Use accumulation_steps=2, microbatch=2 => optimizer_minibatch 4
    loop, _ds, model = _build_loop(
        tmp_path / "accum",
        parquet_dir,
        seed=13,
        microbatch_size=2,
        accumulation_steps=2,
        max_updates=4,
        gradient_clip_norm=0.5,
    )
    # Verify optimizer_minibatch_size derived correctly
    assert loop.config.optimizer_minibatch_size == 4
    init_params = {k: v.clone() for k, v in model.named_parameters()}
    hist = loop.train(max_updates=4)
    assert len(hist) == 4
    # Check that parameters did change (optimizer stepped)
    changed = any(not torch.equal(p, init_params[k]) for k, p in model.named_parameters())
    assert changed, "optimizer did not update parameters"
    # Scheduler stepped (lr changed)
    if loop.scheduler is not None:
        # After 4 updates, scheduler's last_epoch should be 3 or 4
        assert loop.scheduler.last_epoch >= 3
    # Checkpoint written at least once (frequency default 10 but final always writes)
    ckpts = list((tmp_path / "accum").rglob("checkpoint-*.pt"))
    assert len(ckpts) >= 1
    # Verify checkpoint payload contains required 6 sections
    from hydra2.runtime.checkpoint import load_checkpoint

    _m, payload = load_checkpoint(
        source=ckpts[0],
        expected_run_spec_hash=loop.manifest_hashes["run_spec_hash"],
        expected_source_hash=loop.manifest_hashes["dataset_manifest_hash"],
    )
    for key in (
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "training_state",
        "sampler_state",
        "rng_state",
    ):
        assert key in payload, f"checkpoint missing {key}"


def test_accumulation_exactness(tmp_path: Path) -> None:
    # Loss scaling over accumulation window must equal mean over full minibatch
    torch.manual_seed(0)
    B, A = 4, 8
    logits = torch.randn(B, A)
    legal_mask = torch.ones(B, A, dtype=torch.bool)
    targets = torch.randint(0, A, (B,))
    # Full batch loss
    full_loss = masked_cross_entropy(logits, targets, legal_mask)
    # Split into 2 microbatches of 2
    loss1 = masked_cross_entropy(logits[:2], targets[:2], legal_mask[:2])
    loss2 = masked_cross_entropy(logits[2:], targets[2:], legal_mask[2:])
    # Accumulation mean: average of micro means equals full mean when batches are equal size
    # For non-equal or general, scaled sum / accumulation_steps
    micro_avg = (loss1 + loss2) / 2
    assert torch.allclose(full_loss, micro_avg, atol=1e-6)


# ---------------------------------------------------------------------------
# 11 Real-observation tensorization alongside the synthetic stand-in
# ---------------------------------------------------------------------------


def test_real_encode_features_differ_for_same_decision_id() -> None:
    # Kills hash-memorization: identical decision_id, distinct observations
    # must yield distinct real features (the synthetic stand-in hashes only
    # the decision_id, so it cannot distinguish these rows).
    hand_a = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48)
    hand_b = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    obs_a = _make_real_observation(decision_id="same-dec", concealed_hand=hand_a)
    obs_b = _make_real_observation(decision_id="same-dec", concealed_hand=hand_b)
    batch = encode_observation_rows(
        [_real_row_dict("same-dec", obs_a), _real_row_dict("same-dec", obs_b)],
        num_actions=BASELINE_ACTION_COUNT,
        feature_dim=FEATURE_DIM,
    )
    assert tuple(batch["features"].shape) == (2, FEATURE_DIM)
    assert tuple(batch["legal_mask"].shape) == (2, BASELINE_ACTION_COUNT)
    assert batch["chosen_action_id"].tolist() == [0, 0]
    assert not torch.equal(batch["features"][0], batch["features"][1])
    # Legal mask carries the observation's own content, not a hash draw.
    assert bool(batch["legal_mask"][0, 0].item()) is True
    assert bool(batch["legal_mask"][0, 10].item()) is True
    assert bool(batch["legal_mask"][0, 5].item()) is False
    assert torch.equal(batch["legal_mask"][0], batch["legal_mask"][1])
    # Synthetic stand-in cannot distinguish these rows (documents the kill).
    synth_a = tensorize_actor_row(
        {"decision_id": "same-dec", "chosen_action_id": 0},
        num_actions=BASELINE_ACTION_COUNT,
        feature_dim=FEATURE_DIM,
        seed=0,
    )
    synth_b = tensorize_actor_row(
        {"decision_id": "same-dec", "chosen_action_id": 0},
        num_actions=BASELINE_ACTION_COUNT,
        feature_dim=FEATURE_DIM,
        seed=0,
    )
    assert torch.equal(synth_a["features"], synth_b["features"])


def test_encode_pin_flag_byte_identical_skips_page_lock(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """pin_memory=False skips all page-locking with byte-identical tensors.

    The pinned-ring feed stages H2D from its own pinned slots, so
    encode-side pin_memory() calls are pure overhead there. Both arms are
    hermetic (no CUDA needed): the True arm forces the oracle fallback path
    (bridge bulk stage bypasses Tensor.pin_memory via torch.empty +
    ring_fill_batch, so the ring lookup is nulled first) by making
    pin_memory() raise, proving the warning fires; the False arm makes
    pin_memory() boom-if-called, proving the gate never attempts it.
    """
    hand = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48)
    obs = _make_real_observation(decision_id="pin-dec", concealed_hand=hand)
    rows = [_real_row_dict("pin-dec", obs), _real_row_dict("pin-dec", obs)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _boom(self: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("boom: page lock must not be attempted")

    # True arm: forced fallback — warning fires, tensors stay pageable.
    # Null the ring lookup first: with the bridge installed the bulk stage
    # pins via torch.empty(pin_memory=True) + ring_fill_batch and never
    # calls Tensor.pin_memory, so the boom below would never fire. The
    # warning owns the ImportError-only oracle path exercised here.
    monkeypatch.setattr("hydra2.models.encoder._ring_native", lambda: None)
    monkeypatch.setattr(torch.Tensor, "pin_memory", _boom)
    with caplog.at_level(logging.WARNING):
        pinned = encode_observation_rows(
            rows, num_actions=BASELINE_ACTION_COUNT, feature_dim=FEATURE_DIM, pin_memory=True
        )
    assert any("pin_memory failed" in r.message for r in caplog.records)
    caplog.clear()
    # False arm: gate never calls pin_memory — boom would fail the test.
    with caplog.at_level(logging.WARNING):
        plain = encode_observation_rows(
            rows, num_actions=BASELINE_ACTION_COUNT, feature_dim=FEATURE_DIM, pin_memory=False
        )
    assert caplog.records == []
    for key in ("features", "legal_mask", "chosen_action_id"):
        assert torch.equal(pinned[key], plain[key])
    assert torch.equal(
        pinned["actor_batch"].features["concealed_hand_counts"],
        plain["actor_batch"].features["concealed_hand_counts"],
    )
    assert torch.equal(pinned["actor_batch"].history_mask, plain["actor_batch"].history_mask)


def test_real_dataset_mode_batch_over_real_parquet(tmp_path: Path) -> None:
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
    assert tuple(batch["features"].shape) == (4, FEATURE_DIM)
    assert tuple(batch["legal_mask"].shape) == (4, BASELINE_ACTION_COUNT)
    assert tuple(batch["chosen_action_id"].shape) == (4,)
    assert bool(torch.isfinite(batch["features"]).all().item()) is True
    # Chosen actions stay legal under the real masks.
    for i in range(4):
        chosen = int(batch["chosen_action_id"][i].item())
        assert bool(batch["legal_mask"][i, chosen].item()) is True


def test_synthetic_default_unchanged(tmp_path: Path, actor_parquet_factory) -> None:
    parquet_dir = actor_parquet_factory(num_rows=8)
    dataset = AuthoritativeParquetDataset(
        parquet_dir=parquet_dir,
        feature_dim=FEATURE_DIM,
        num_actions=NUM_ACTIONS_SMALL,
        seed=0,
        verify=True,
        allow_narrow=True,
    )
    assert dataset.tensorize == "synthetic"
    batch = dataset.next_batch(4)
    assert batch is not None
    assert tuple(batch["features"].shape) == (4, FEATURE_DIM)
    assert tuple(batch["legal_mask"].shape) == (4, NUM_ACTIONS_SMALL)
    # Synthetic features remain decision_id-hash deterministic.
    probe = tensorize_actor_row(
        {"decision_id": batch["_decision_ids"][0], "chosen_action_id": 0},
        num_actions=NUM_ACTIONS_SMALL,
        feature_dim=FEATURE_DIM,
        seed=0,
        allow_narrow=True,
    )
    assert torch.equal(batch["features"][0], probe["features"])


def test_real_mode_rejects_bad_tensorize_flag(tmp_path: Path, actor_parquet_factory) -> None:
    parquet_dir = actor_parquet_factory(num_rows=4)
    with pytest.raises(ContractError):
        AuthoritativeParquetDataset(
            parquet_dir=parquet_dir,
            feature_dim=FEATURE_DIM,
            num_actions=NUM_ACTIONS_SMALL,
            seed=0,
            verify=True,
            allow_narrow=True,
            tensorize="hashed",
        )  # type: ignore[arg-type]


def test_real_mode_bad_row_raises_contract_error(tmp_path: Path) -> None:
    with pytest.raises(ContractError):
        encode_observation_rows(
            [{"decision_id": "bad", "chosen_action_id": 0, "actor_observation": "not-json"}],
            num_actions=BASELINE_ACTION_COUNT,
            feature_dim=FEATURE_DIM,
        )
    # Well-formed JSON that is not an ActorObservation never hash-falls-back.
    with pytest.raises(ContractError):
        encode_observation_rows(
            [
                {
                    "decision_id": "bad2",
                    "chosen_action_id": 0,
                    "actor_observation": json.dumps({"dora_indicators": [1, 2, -1, -1, -1]}),
                }
            ],
            num_actions=BASELINE_ACTION_COUNT,
            feature_dim=FEATURE_DIM,
        )
    # Real mode over synthetic-stand-in parquet (valid JSON, not observations)
    # fails closed at batch time instead of tensorising by hash.
    dest = _write_synthetic_parquet(tmp_path, num_rows=4)
    dataset = AuthoritativeParquetDataset(
        parquet_dir=dest,
        feature_dim=FEATURE_DIM,
        num_actions=BASELINE_ACTION_COUNT,
        seed=0,
        verify=True,
        tensorize="real",
    )
    with pytest.raises(ContractError):
        dataset.next_batch(4)


def test_supervised_loop_wall_ledger_overlap_raises(tmp_path: Path, actor_parquet_factory) -> None:
    """Loop ledger mirrors replay: ranks-written corpus overlapping eval walls raises from the join."""
    from hydra2.belief.oracle_loader import PrivilegedOracleLoader
    from hydra2.data.parquet import write_privileged_ranks

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
    batch = dataset.next_batch(4)
    assert batch is not None
    decision_ids = [str(x) for x in batch["_decision_ids"]]
    assert len(decision_ids) == 4
    dest = tmp_path / "priv_wall_loop"
    write_privileged_ranks(
        destination=dest,
        ranks_by_id={did: [1, 2, 3, 4] for did in decision_ids},
        wall_ids={
            did: ("wall-eval-1" if i == 0 else "wall-train-9") for i, did in enumerate(decision_ids)
        },
    )
    loader = PrivilegedOracleLoader(dest, split="train", verify=True, allow_synthetic=True)
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
    loop = SupervisedLoop(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        dataset=dataset,
        config=config,
        checkpoint_dir=tmp_path / "wall_ledger",
        manifest_hashes=make_test_manifest_hashes(),
        privileged_source=loader,
        evaluation_wall_ids={"wall-eval-1"},
    )
    assert loop.evaluation_wall_ids == frozenset({"wall-eval-1"})
    with pytest.raises(ContractError, match="wall leakage"):
        loop.train_step(batch)


def test_narrow_vocab_requires_explicit_flag(tmp_path: Path, actor_parquet_factory) -> None:
    """Narrow vocabs fail closed without allow_narrow (no silent aliasing)."""
    parquet_dir = actor_parquet_factory(num_rows=4)
    with pytest.raises(ContractError, match="allow_narrow"):
        AuthoritativeParquetDataset(
            parquet_dir=parquet_dir,
            feature_dim=FEATURE_DIM,
            num_actions=NUM_ACTIONS_SMALL,
            seed=0,
            verify=True,
        )
    with pytest.raises(ContractError, match="allow_narrow"):
        tensorize_actor_row(
            {"decision_id": "dec-0000", "chosen_action_id": 0},
            num_actions=NUM_ACTIONS_SMALL,
            feature_dim=FEATURE_DIM,
            seed=0,
        )
