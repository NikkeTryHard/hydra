"""Fail-closed precision correctness (Job 1).

Covers only genuinely uncertain edges: plain pin, replay pin, loop/runtime
agreement matrix, digest distinctness, resume rejection, finite-skip.
Deterministic (fixed seeds, tmp_path); no CUDA required except where noted.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest
import torch
import torch.nn as nn

from hydra2.contracts.common import ContractError, CorruptArtifactError
from hydra2.runtime.protocol import RuntimeSpec, runtime_identity
from tests.unit._manifest_helpers import make_test_manifest_hashes


def _fp32_cpu_spec(**overrides: Any) -> RuntimeSpec:
    fields: dict[str, Any] = {
        "adapter_id": "plain_pytorch",
        "device": "cpu",
        "precision": "fp32",
        "compile_mode": "eager",
        "fullgraph": False,
        "dynamic": None,
        "backward_pass_autocast": None,
    }
    fields.update(overrides)
    return RuntimeSpec(**fields)  # type: ignore[arg-type]


class _StubModel(nn.Module):
    def __init__(self, feature_dim: int = 8, num_actions: int = 8) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_actions, bias=True)
        torch.manual_seed(0)
        nn.init.normal_(self.linear.weight, std=0.1)
        nn.init.zeros_(self.linear.bias)

    def forward(self, batch: dict) -> dict:  # type: ignore[override]
        feats = batch["features"].float()
        logits = self.linear(feats)
        mask = batch["legal_mask"]
        logits = logits.masked_fill(~mask, -1e9)
        return {"policy_logits": logits}


class _StubDataset:
    """Minimal loop dataset (in-memory, deterministic)."""

    def __init__(self, num_actions: int = 8, feature_dim: int = 8, seed: int = 0) -> None:
        self.num_actions = num_actions
        self.feature_dim = feature_dim
        self._offset = 0
        self._rng = torch.Generator().manual_seed(seed)

    def next_batch(self, n: int) -> dict[str, Any]:
        feats = torch.randn(n, self.feature_dim, generator=self._rng)
        legal_mask = torch.ones(n, self.num_actions, dtype=torch.bool)
        chosen = torch.randint(0, self.num_actions, (n,), generator=self._rng)
        self._offset += n
        return {
            "features": feats,
            "legal_mask": legal_mask,
            "chosen_action_id": chosen,
        }

    def get_sampler_state(self) -> dict[str, Any]:
        return {"offset": self._offset, "seed": 0, "total": 1000, "epoch": 0}

    def set_sampler_state(self, state: Any) -> None:
        if isinstance(state, dict) and "offset" in state:
            self._offset = int(state["offset"])

    def __len__(self) -> int:
        return 1000


def _build_loop(
    tmp_path: Path,
    *,
    loop_precision: str,
    rt_precision: str | None = None,
    adapter_id: str = "plain_pytorch",
    device: str | None = None,
):
    from hydra2.training.loop_state import TrainingLoopConfig
    from hydra2.training.loop_train import SupervisedLoop

    torch.manual_seed(0)
    model = _StubModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    dataset = _StubDataset()
    config = TrainingLoopConfig(
        microbatch_size=4,
        accumulation_steps=1,
        max_updates=2,
        checkpoint_frequency_updates=10,
        seed=0,
        precision=loop_precision,  # type: ignore[arg-type]
    )
    rt_spec = None
    if rt_precision is not None:
        rt_spec = _fp32_cpu_spec(adapter_id=adapter_id, precision=rt_precision)
    loop = SupervisedLoop(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        config=config,
        checkpoint_dir=tmp_path / f"ckpt-{loop_precision}-{rt_precision}-{adapter_id}",
        manifest_hashes=make_test_manifest_hashes(),
        runtime_spec=rt_spec,
        device=device,
    )
    return loop, model


class TestPlainPin:
    def test_plain_rejects_bf16_without_cuda(self) -> None:
        from hydra2.runtime.plain import PlainPytorchAdapter

        adapter = PlainPytorchAdapter()
        model = _StubModel()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        spec = _fp32_cpu_spec(precision="bf16_mixed")
        with pytest.raises(ContractError, match="PlainPytorchAdapter"):
            adapter.setup(model=model, optimizer=opt, spec=spec)

    def test_plain_rejects_fp16_without_cuda(self) -> None:
        from hydra2.runtime.plain import PlainPytorchAdapter

        adapter = PlainPytorchAdapter()
        model = _StubModel()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        spec = _fp32_cpu_spec(precision="fp16_mixed")
        with pytest.raises(ContractError, match="PlainPytorchAdapter"):
            adapter.setup(model=model, optimizer=opt, spec=spec)

    def test_plain_fp32_preserves_identity(self) -> None:
        from hydra2.runtime.plain import PlainPytorchAdapter

        adapter = PlainPytorchAdapter()
        torch.manual_seed(0)
        model = _StubModel()
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        spec = _fp32_cpu_spec()
        handle = adapter.setup(model=model, optimizer=opt, spec=spec)
        assert handle.model is model
        assert handle.optimizer is opt
        for k, v in model.state_dict().items():
            assert torch.equal(v, before[k])


class TestPlainSetup:
    def test_plain_setup_returns_same_objects(self) -> None:
        from hydra2.runtime.plain import PlainPytorchAdapter

        adapter = PlainPytorchAdapter()
        model = _StubModel()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        spec = _fp32_cpu_spec(adapter_id="plain_pytorch", precision="fp32")
        handle = adapter.setup(model=model, optimizer=opt, spec=spec)
        assert handle.model is model
        assert handle.optimizer is opt


class TestReplayPin:
    def test_replay_config_rejects_bf16(self) -> None:
        from hydra2.training.replay_state import ReplayConfig

        with pytest.raises(ContractError, match="fp32-only"):
            ReplayConfig(
                microbatch_size=2,
                accumulation_steps=1,
                max_updates=1,
                seed=0,
                precision="bf16_mixed",
            ).validate()  # type: ignore[arg-type]

    def test_replay_runtime_spec_bf16_fails_closed(self, tmp_path: Path) -> None:
        from hydra2.data.parquet import DecisionRow, write_actor_shards
        from hydra2.training.dataset_store import AuthoritativeParquetDataset
        from hydra2.training.replay_engine import ActorLearnerReplay
        from hydra2.training.replay_state import ReplayConfig

        dest = tmp_path / "actor_parquet"
        rows = [
            DecisionRow(
                game_id=f"game-{i // 4:04d}",
                round_id=f"round-{i:04d}",
                decision_id=f"dec-{i:04d}",
                seat=i % 4,
                source_object_id=f"src-{i:04d}",
                split="train",
                rules_hash="sha256:" + "a" * 64,
                adapter_hash="sha256:" + "b" * 64,
                observation_hash="sha256:" + hashlib.sha256(f"dec-{i:04d}".encode()).hexdigest(),
                action_table_hash="sha256:" + "c" * 64,
                derivation_hash="sha256:" + "d" * 64,
                actor_observation={  # type: ignore[arg-type]
                    "game_id": f"game-{i // 4:04d}",
                    "decision_id": f"dec-{i:04d}",
                    "actor": i % 4,
                    "hand": [[1, 2, 3]],
                    "dora_indicators": [0, 1, 2, 3, 4],
                    "legal_mask": [True] * 8,
                },
                chosen_action_id=i % 8,
            )
            for i in range(8)
        ]
        write_actor_shards(
            destination=dest,
            rows=rows,
            dataset_hash="sha256:" + "e" * 64,
            split_manifest_hash="sha256:" + "f" * 64,
        )
        dataset = AuthoritativeParquetDataset(
            parquet_dir=dest, feature_dim=16, num_actions=8, seed=0, allow_narrow=True
        )
        # NOTE: feature_dim mismatch (16 vs stub 8) is fine — construction
        # fails on precision before any forward.
        model = _StubModel(feature_dim=8, num_actions=8)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        config = ReplayConfig(microbatch_size=2, accumulation_steps=1, max_updates=1, seed=0)
        rt_bf16 = _fp32_cpu_spec(adapter_id="plain_pytorch", precision="bf16_mixed")
        with pytest.raises(ContractError, match="fp32-only"):
            ActorLearnerReplay(
                model=model,
                optimizer=opt,
                dataset=dataset,
                config=config,
                checkpoint_dir=tmp_path / "ckpt",
                manifest_hashes=make_test_manifest_hashes(),
                runtime_spec=rt_bf16,
            )


@pytest.mark.serial
class TestLoopRuntimeAgreement:
    @pytest.mark.parametrize(
        ("adapter_id", "rt_precision", "loop_precision"),
        [
            ("plain_pytorch", "fp32", "bf16_mixed"),
            ("plain_pytorch", "bf16_mixed", "fp32"),
        ],
    )
    def test_mismatch_matrix_raises(
        self, tmp_path: Path, adapter_id: str, rt_precision: str, loop_precision: str
    ) -> None:
        with pytest.raises(ContractError, match=r"[Pp]recision|plain"):
            _build_loop(
                tmp_path,
                loop_precision=loop_precision,
                rt_precision=rt_precision,
                adapter_id=adapter_id,
            )

    @pytest.mark.parametrize(
        ("adapter_id", "precision"),
        [("plain_pytorch", "fp32"), ("plain_pytorch", "bf16_mixed")],
    )
    def test_agreeing_constructs(self, tmp_path: Path, adapter_id: str, precision: str) -> None:
        loop, _ = _build_loop(
            tmp_path, loop_precision=precision, rt_precision=precision, adapter_id=adapter_id
        )

        assert loop.config.precision == precision

    def test_plain_bf16_cuda_agrees(self, tmp_path: Path) -> None:
        """plain+bf16_mixed constructs on CUDA (loop-owned autocast is CUDA-only)."""
        loop, _ = _build_loop(
            tmp_path,
            loop_precision="bf16_mixed",
            rt_precision="bf16_mixed",
            adapter_id="plain_pytorch",
            device="cuda",
        )
        assert loop.config.precision == "bf16_mixed"
        assert loop.device.type == "cuda"

    def test_plain_bf16_cpu_rejected(self, tmp_path: Path) -> None:
        """plain+bf16_mixed on CPU fails closed (would silently compute fp32)."""
        with pytest.raises(ContractError, match="CUDA"):
            _build_loop(
                tmp_path,
                loop_precision="bf16_mixed",
                rt_precision="bf16_mixed",
                adapter_id="plain_pytorch",
                device="cpu",
            )


class TestDigestDistinctness:
    def test_fp32_vs_bf16_runtime_identity_distinct(self) -> None:
        fp32 = runtime_identity(_fp32_cpu_spec(precision="fp32"))
        bf16 = runtime_identity(_fp32_cpu_spec(precision="bf16_mixed"))
        assert fp32 != bf16

    def test_training_state_hash_binds_precision(self) -> None:
        from hydra2.runtime.checkpoint import hash_state_tree
        from hydra2.training.loop_state import TrainingState

        fp32 = TrainingState(precision="fp32").to_dict()
        bf16 = TrainingState(precision="bf16_mixed").to_dict()
        assert hash_state_tree(fp32) != hash_state_tree(bf16)


@pytest.mark.serial
class TestResumePrecisionMismatch:
    def test_resume_rejects_cross_regime(self, tmp_path: Path) -> None:
        from hydra2.training.loop_state import TrainingLoopConfig
        from hydra2.training.loop_train import SupervisedLoop

        torch.manual_seed(0)
        loop, _ = _build_loop(tmp_path / "a", loop_precision="fp32")
        loop.train(max_updates=1)
        ckpt = loop.save_checkpoint()
        # New loop under bf16 with the SAME manifest hashes (lying digests):
        # precision gate must still fire.
        torch.manual_seed(0)
        model2 = _StubModel()
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        dataset2 = _StubDataset()
        config_bf16 = TrainingLoopConfig(
            microbatch_size=4,
            accumulation_steps=1,
            max_updates=2,
            checkpoint_frequency_updates=10,
            seed=0,
            precision="bf16_mixed",
        )
        loop2 = SupervisedLoop(
            model=model2,
            optimizer=opt2,
            dataset=dataset2,
            config=config_bf16,
            checkpoint_dir=tmp_path / "b",
            manifest_hashes=make_test_manifest_hashes(),
        )
        with pytest.raises((ContractError, CorruptArtifactError)):
            loop2.resume_from_checkpoint(ckpt)


@pytest.mark.serial
class TestFiniteSkip:
    def test_inf_grad_skips_without_poisoning(self, tmp_path: Path) -> None:
        loop, model = _build_loop(tmp_path, loop_precision="fp32")
        init_params = [p.detach().clone() for p in model.parameters()]
        orig_backward = loop._backward

        def poisoned(loss: Any) -> None:
            orig_backward(loss)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.fill_(float("inf"))
                    break

        loop._backward = poisoned  # type: ignore[method-assign]
        history = loop.train(max_updates=2)
        assert loop.state.skipped_updates == 2
        assert all(h.get("skipped_this_update") == 1.0 for h in history)
        assert history[-1].get("skipped_updates") == 2.0
        for p0, p1 in zip(init_params, model.parameters(), strict=True):
            assert torch.equal(p0, p1), "skipped step must not move weights"
        # Skip entries omit non-finite grad keys (absence, not value) but keep lr.
        for h in history:
            assert "grad_norm_pre" not in h
            assert "grad_norm_post" not in h
            assert "lr_now" in h

    def test_helper_flags_nonfinite(self) -> None:
        from hydra2.training.objectives_loss import global_grad_norm_is_finite

        model = _StubModel()
        for p in model.parameters():
            p.grad = torch.ones_like(p) * float("inf")
            break
        finite, norm = global_grad_norm_is_finite(model)
        assert finite is False
        assert norm == float("inf") or norm > 0

    def test_helper_finite_single_sync_exact(self) -> None:
        """Fused probe: finite grads report exact norm with one host sync."""
        from hydra2.training.objectives_loss import global_grad_norm_is_finite

        model = _StubModel()
        for p in model.parameters():
            p.grad = torch.ones_like(p) * 3.0
        finite, norm = global_grad_norm_is_finite(model)
        assert finite is True
        expected = sum(p.grad.numel() * 9.0 for p in model.parameters()) ** 0.5
        assert norm == pytest.approx(expected, rel=1e-5)

    def test_helper_no_grads_is_finite_zero(self) -> None:
        """Fused probe: no grads anywhere is finite with zero norm."""
        from hydra2.training.objectives_loss import global_grad_norm_is_finite

        model = _StubModel()
        finite, norm = global_grad_norm_is_finite(model)
        assert finite is True
        assert norm == 0.0
