"""GPU probe: checkpoint round trip resumes an IDENTICAL next update
(bitwise state equality across model/optimizer/training/RNG state).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest
import torch

from hydra2.runtime.checkpoint import (
    build_manifest,
    capture_rng_state,
    hash_state_tree,
    load_checkpoint,
    resume_checkpoint,
    save_checkpoint,
)
from hydra2.runtime.plain import PlainPytorchAdapter
from hydra2.runtime.protocol import RuntimeSpec, build_runtime, runtime_identity
from tests.conftest import (
    assert_states_bitwise_equal,
    make_batch,
    make_model_and_optimizer,
    run_supervised_steps,
    state_snapshot,
    unwrap_model,
)

if TYPE_CHECKING:
    from pathlib import Path

SEED = 424242


def spec() -> RuntimeSpec:
    return RuntimeSpec(
        adapter_id="plain_pytorch",
        device="cuda",
        precision="fp32",
        compile_mode="eager",
        fullgraph=False,
        dynamic=None,
        backward_pass_autocast=None,
    )


def build_handle(model, optimizer):
    return build_runtime(
        adapter=PlainPytorchAdapter(), model=model, optimizer=optimizer, spec=spec()
    )


def plain_model(handle):
    return unwrap_model(handle.model)


def optimizer_state_cpu(optimizer):
    def detach(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {k: detach(v) for k, v in value.items()}
        if isinstance(value, list):
            return [detach(v) for v in value]
        return value

    raw = optimizer.state_dict()
    return {key: detach(value) for key, value in raw.items()}


def save_manifest(*, handle, destination: Path, run_spec_hash: str, source_hash: str, parent):
    payload = {
        "model_state": plain_model(handle).state_dict(),
        "optimizer_state": handle.optimizer.state_dict(),
        "scheduler_state": {},
        "training_state": {
            "global_update": 6,
            "microstep": 12,
            "epoch": 0,
            "examples_seen": 48,
        },
        "sampler_state": {"cursor": 5},
        "rng_state": capture_rng_state(),
    }
    manifest = build_manifest(
        run_spec_hash=run_spec_hash,
        model_spec_hash="sha256:" + "a" * 64,
        optimizer_spec_hash="sha256:" + "b" * 64,
        scheduler_spec_hash="sha256:" + "c" * 64,
        environment_hash="sha256:" + "d" * 64,
        rules_hash="sha256:" + "e" * 64,
        utility_manifest_hash="sha256:" + "f" * 64,
        action_schema_hash="sha256:" + "0" * 64,
        observation_schema_hash="sha256:" + "1" * 64,
        dataset_manifest_hash=source_hash,
        rollout_artifact_hash=None,
        parent_checkpoint_hash=parent,
        payload=payload,
    )
    save_checkpoint(destination=destination, manifest=manifest, payload=payload)
    return manifest


@pytest.mark.gpu
class TestCheckpointResumeIdentity:
    def test_resume_reproduces_identical_next_update(self, require_cuda, tmp_path):
        x, y = make_batch(SEED + 1)
        run_spec_hash = runtime_identity(spec())
        source_hash = "sha256:" + "7" * 64  # dataset manifest identity stand-in

        # Trajectory A: train N steps, save, continue M steps.
        model_a, optimizer_a = make_model_and_optimizer(SEED)
        handle_a = build_handle(model_a, optimizer_a)
        run_supervised_steps(handle_a, x, y, steps=3)
        destination = tmp_path / "wp01-resume.pt"
        manifest = save_manifest(
            handle=handle_a,
            destination=destination,
            run_spec_hash=run_spec_hash,
            source_hash=source_hash,
            parent=None,
        )
        losses_after_save = run_supervised_steps(handle_a, x, y, steps=3)

        # Trajectory B: identical rebuild, verified resume, run M steps.
        model_b, optimizer_b = make_model_and_optimizer(SEED)
        handle_b = build_handle(model_b, optimizer_b)
        returned_manifest = resume_checkpoint(
            source=destination,
            run_spec_hash=run_spec_hash,
            source_hash=source_hash,
            model=plain_model(handle_b),
            optimizer=handle_b.optimizer,
        )
        assert returned_manifest == manifest

        # Restored RNG stream is bitwise the stream saved with the checkpoint.
        _, payload = load_checkpoint(
            source=destination,
            expected_run_spec_hash=run_spec_hash,
            expected_source_hash=source_hash,
        )
        assert torch.equal(payload["rng_state"]["cpu"].cpu(), torch.get_rng_state().cpu())

        losses_b = run_supervised_steps(handle_b, x, y, steps=3)

        # The resumed next updates must be IDENTICAL, not merely close.
        assert losses_b == losses_after_save
        assert_states_bitwise_equal(
            state_snapshot(plain_model(handle_a)),
            state_snapshot(plain_model(handle_b)),
            context="model-after-resume",
        )
        assert_states_bitwise_equal(
            optimizer_state_cpu(optimizer_a),
            optimizer_state_cpu(optimizer_b),
            context="optimizer-after-resume",
        )

    def test_parent_chain_second_generation_restores(self, require_cuda, tmp_path):
        """Save v1, step, save v2 with parent=v1 file hash; both verify."""
        x, y = make_batch(SEED + 2)
        run_spec_hash = runtime_identity(spec())
        source_hash = "sha256:" + "8" * 64

        model, optimizer = make_model_and_optimizer(SEED)
        handle = build_handle(model, optimizer)
        run_supervised_steps(handle, x, y, steps=1)

        v1_path = tmp_path / "gen1.pt"
        v1 = save_manifest(
            handle=handle,
            destination=v1_path,
            run_spec_hash=run_spec_hash,
            source_hash=source_hash,
            parent=None,
        )
        v1_file_hash = "sha256:" + hashlib.sha256(v1_path.read_bytes()).hexdigest()

        run_supervised_steps(handle, x, y, steps=1)
        v2_path = tmp_path / "gen2.pt"
        v2 = save_manifest(
            handle=handle,
            destination=v2_path,
            run_spec_hash=run_spec_hash,
            source_hash=source_hash,
            parent=v1_file_hash,
        )
        assert v2.parent_checkpoint_hash == v1_file_hash
        assert v2.model_state_hash != v1.model_state_hash

        m1, p1 = load_checkpoint(
            source=v1_path,
            expected_run_spec_hash=run_spec_hash,
            expected_source_hash=source_hash,
        )
        m2, p2 = load_checkpoint(
            source=v2_path,
            expected_run_spec_hash=run_spec_hash,
            expected_source_hash=source_hash,
        )
        assert m1 == v1 and m2 == v2
        assert hash_state_tree(p1["model_state"]) == v1.model_state_hash
        assert hash_state_tree(p2["model_state"]) == v2.model_state_hash

        fresh_model, fresh_optimizer = make_model_and_optimizer(SEED)
        fresh_handle = build_handle(fresh_model, fresh_optimizer)
        resume_checkpoint(
            source=v2_path,
            run_spec_hash=run_spec_hash,
            source_hash=source_hash,
            model=plain_model(fresh_handle),
            optimizer=fresh_handle.optimizer,
        )
        assert_states_bitwise_equal(
            state_snapshot(plain_model(handle)),
            state_snapshot(plain_model(fresh_handle)),
            context="generation-restore",
        )
