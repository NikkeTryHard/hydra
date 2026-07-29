"""Checkpoint manifest invariants, deterministic hashing, save/load
verification-before-mutation (CPU scope; GPU resume identity lives in
integration/).
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from hydra2.contracts.common import (
    ContractError,
    CorruptArtifactError,
    DigestMismatchError,
)
from hydra2.runtime.checkpoint import (
    CheckpointManifest,
    build_manifest,
    hash_state_tree,
    load_checkpoint,
    manifest_from_json,
    manifest_to_json,
    resume_checkpoint,
    save_checkpoint,
    state_tree,
)

D = "sha256:" + "b" * 64


def sample_payload(seed: int = 7):
    generator = torch.Generator().manual_seed(seed)
    model_state = {
        "net.0.weight": torch.randn(4, 3, generator=generator),
        "net.0.bias": torch.randn(4, generator=generator),
    }
    optimizer_state = {
        "state": {0: {"exp_avg": torch.rand(4, 3, generator=generator), "step": torch.tensor(3)}},
        "param_groups": [{"lr": 0.01, "betas": [0.9, 0.999]}],
    }
    return {
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": {},
        "training_state": {
            "global_update": 3,
            "microstep": 6,
            "epoch": 0,
            "examples_seen": 24,
        },
        "sampler_state": {"cursor": 17, "split_ids": ["a", "b"]},
        "rng_state": {"cpu": torch.randint(0, 255, (16,), generator=generator, dtype=torch.uint8)},
    }


def make_manifest(payload, **overrides) -> CheckpointManifest:
    kwargs = {
        "run_spec_hash": D,
        "model_spec_hash": D,
        "optimizer_spec_hash": D,
        "scheduler_spec_hash": D,
        "environment_hash": D,
        "rules_hash": D,
        "utility_manifest_hash": D,
        "action_schema_hash": D,
        "observation_schema_hash": D,
        "dataset_manifest_hash": "sha256:" + "c" * 64,
        "rollout_artifact_hash": None,
        "payload": payload,
    }
    kwargs.update(overrides)
    return build_manifest(**kwargs)


class TestStateTreeHashing:
    def test_identical_values_identical_digest(self):
        a = {"w": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
        b = {"w": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
        assert hash_state_tree(a) == hash_state_tree(b)

    def test_value_flip_changes_digest(self):
        a = {"w": torch.zeros(3)}
        b = {"w": torch.ones(3)}
        assert hash_state_tree(a) != hash_state_tree(b)

    def test_dtype_and_shape_are_identity(self):
        f32 = {"w": torch.ones(3, dtype=torch.float32)}
        f64 = {"w": torch.ones(3, dtype=torch.float64)}
        assert hash_state_tree(f32) != hash_state_tree(f64)

    def test_bfloat16_tensors_hashable(self):
        tree = state_tree({"w": torch.ones(2, dtype=torch.bfloat16)})
        assert tree["items"]["w"]["dtype"] == "torch.bfloat16"

    def test_tensor_bytes_domain_is_sha256_hex(self):
        tree = state_tree({"w": torch.tensor([1], dtype=torch.uint8)})
        leaf = tree["items"]["w"]
        assert leaf["bytes_sha256"].startswith("sha256:")
        assert len(leaf["bytes_sha256"]) == len("sha256:") + 64

    def test_map_order_irrelevant(self):
        assert hash_state_tree({"x": 1, "y": 2}) == hash_state_tree({"y": 2, "x": 1})

    def test_int_map_keys_encode_deterministically(self):
        tree = state_tree({1: "a", "b": 2})
        assert tree["items"] == {
            "#int:1": {"kind": "scalar", "value": "a"},
            "b": {"kind": "scalar", "value": 2},
        }

    def test_bool_map_key_rejected(self):
        with pytest.raises(ContractError):
            state_tree({True: "a"})

    def test_unsupported_leaf_rejected(self):
        class Opaque:
            pass

        with pytest.raises(CorruptArtifactError):
            state_tree({"x": Opaque()})

    def test_raw_bytes_helper_matches_hashlib(self):
        raw = b"hydra2"
        tree = state_tree(raw)
        assert tree == {"kind": "bytes_hex", "value": raw.hex()}
        assert hashlib.sha256(bytes.fromhex(tree["value"])).digest() == hashlib.sha256(raw).digest()


class TestSourceIdentityInvariant:
    @pytest.mark.parametrize(
        "dataset,rollout",
        [
            ("sha256:" + "c" * 64, "sha256:" + "d" * 64),
            (None, None),
        ],
    )
    def test_exactly_one_source_required(self, dataset, rollout):
        with pytest.raises(ContractError, match="exactly one"):
            make_manifest(
                sample_payload(), dataset_manifest_hash=dataset, rollout_artifact_hash=rollout
            )

    def test_rl_variant_accepted(self):
        manifest = make_manifest(
            sample_payload(),
            dataset_manifest_hash=None,
            rollout_artifact_hash="sha256:" + "d" * 64,
        )
        assert manifest.rollout_artifact_hash == "sha256:" + "d" * 64
        assert manifest.dataset_manifest_hash is None

    def test_parent_checkpoint_chain_recorded(self):
        parent = "sha256:" + "1" * 64
        manifest = make_manifest(sample_payload(), parent_checkpoint_hash=parent)
        assert manifest.parent_checkpoint_hash == parent


class TestSaveLoadVerification:
    def test_round_trip_preserves_payload(self, tmp_path):
        payload = sample_payload()
        manifest = make_manifest(payload)
        destination = tmp_path / "ck.pt"
        save_checkpoint(destination=destination, manifest=manifest, payload=payload)

        loaded_manifest, loaded_payload = load_checkpoint(
            source=destination,
            expected_run_spec_hash=D,
            expected_source_hash="sha256:" + "c" * 64,
        )
        assert loaded_manifest == manifest
        assert set(loaded_payload) == set(payload)

    def test_wrong_run_spec_hash_rejected(self, tmp_path):
        payload = sample_payload()
        destination = tmp_path / "ck.pt"
        save_checkpoint(destination=destination, manifest=make_manifest(payload), payload=payload)
        with pytest.raises(DigestMismatchError, match="run_spec_hash"):
            load_checkpoint(
                source=destination,
                expected_run_spec_hash="sha256:" + "e" * 64,
                expected_source_hash="sha256:" + "c" * 64,
            )

    def test_wrong_source_hash_rejected(self, tmp_path):
        payload = sample_payload()
        destination = tmp_path / "ck.pt"
        save_checkpoint(destination=destination, manifest=make_manifest(payload), payload=payload)
        with pytest.raises(DigestMismatchError, match="source identity"):
            load_checkpoint(
                source=destination,
                expected_run_spec_hash=D,
                expected_source_hash="sha256:" + "f" * 64,
            )

    def test_tampered_section_detected(self, tmp_path):
        payload = sample_payload()
        destination = tmp_path / "ck.pt"
        save_checkpoint(destination=destination, manifest=make_manifest(payload), payload=payload)

        container = torch.load(destination, weights_only=True)
        container["payload"]["model_state"]["net.0.bias"][0] += 1.0
        torch.save(container, destination)

        with pytest.raises(CorruptArtifactError, match="model_state"):
            load_checkpoint(
                source=destination,
                expected_run_spec_hash=D,
                expected_source_hash="sha256:" + "c" * 64,
            )

    def test_missing_container_keys_rejected(self, tmp_path):
        torch.save({"unrelated": True}, tmp_path / "bad.pt")
        with pytest.raises(CorruptArtifactError, match="malformed"):
            load_checkpoint(
                source=tmp_path / "bad.pt",
                expected_run_spec_hash=D,
                expected_source_hash="sha256:" + "c" * 64,
            )

    def test_verify_before_mutate_leaves_model_untouched(self, tmp_path):
        model = torch.nn.Linear(3, 4)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        payload = sample_payload()
        destination = tmp_path / "ck.pt"
        save_checkpoint(destination=destination, manifest=make_manifest(payload), payload=payload)

        with pytest.raises(DigestMismatchError):
            resume_checkpoint(
                source=destination,
                run_spec_hash="sha256:" + "e" * 64,
                source_hash="sha256:" + "c" * 64,
                model=model,
            )
        after = model.state_dict()
        for key in before:
            assert torch.equal(before[key], after[key]), f"{key} mutated before verification"

    def test_resume_applies_states_after_verification(self, tmp_path):
        torch.manual_seed(11)
        model_a = torch.nn.Linear(3, 4)
        optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=0.01)
        x = torch.randn(5, 3)
        y = torch.randn(5, 4)
        for _ in range(2):
            optimizer_a.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model_a(x), y)
            loss.backward()
            optimizer_a.step()

        payload = {
            "model_state": model_a.state_dict(),
            "optimizer_state": optimizer_a.state_dict(),
            "scheduler_state": {},
            "training_state": {
                "global_update": 2,
                "microstep": 2,
                "epoch": 0,
                "examples_seen": 10,
            },
            "sampler_state": {"cursor": 0},
            "rng_state": {"cpu": torch.get_rng_state()},
        }
        manifest = make_manifest(payload)
        destination = tmp_path / "ck.pt"
        save_checkpoint(destination=destination, manifest=manifest, payload=payload)

        torch.manual_seed(99)  # diverge RNG so restore proves application happened
        model_b = torch.nn.Linear(3, 4)
        model_b.load_state_dict(model_a.state_dict())
        optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=0.01)
        returned = resume_checkpoint(
            source=destination,
            run_spec_hash=D,
            source_hash="sha256:" + "c" * 64,
            model=model_b,
            optimizer=optimizer_b,
        )
        assert returned == manifest
        for key in model_a.state_dict():
            assert torch.equal(model_a.state_dict()[key], model_b.state_dict()[key])
        # AdamW moments and step counters restored bitwise, per parameter.
        for ref_state, new_state in zip(
            optimizer_a.state.values(), optimizer_b.state.values(), strict=True
        ):
            assert torch.equal(ref_state["exp_avg"].cpu(), new_state["exp_avg"].cpu())
            assert torch.equal(ref_state["exp_avg_sq"].cpu(), new_state["exp_avg_sq"].cpu())
            assert torch.equal(ref_state["step"].cpu(), new_state["step"].cpu())


class TestManifestEnvelope:
    def test_json_round_trip_validates(self):
        manifest = make_manifest(sample_payload())
        parsed = manifest_from_json(manifest_to_json(manifest))
        assert parsed == manifest

    def test_unknown_field_rejected(self):
        raw = manifest_to_json(make_manifest(sample_payload()))
        raw["surprise"] = 1
        with pytest.raises(ContractError, match="unknown"):
            manifest_from_json(raw)

    def test_noncanonical_digest_in_envelope_rejected(self):
        raw = manifest_to_json(make_manifest(sample_payload()))
        raw["run_spec_hash"] = "SHA256:" + "b" * 64
        with pytest.raises(ContractError):
            manifest_from_json(raw)

    def test_missing_payload_sections_rejected(self):
        payload = sample_payload()
        del payload["rng_state"]
        with pytest.raises(ContractError, match="missing sections"):
            make_manifest(payload)
