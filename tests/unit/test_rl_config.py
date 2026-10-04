"""WP-11 closed RL config: discriminated source union plus frozen profile.

Covers the genuine RL config surface only (no training execution): the
native ``rl`` section parser (closed keys, frozen recipe, identities), the
source union (supervised/distill keep dataset roots; RL forbids them and
requires ``rl``), the RL cross gate (caps vs exact counts, recipe mirrors,
tails trained), schema-registry digests, and dry-plan rendering. All
fixtures are worker-local ``tmp_path`` trees with fixed seeds; no sockets,
no sleeps, no wall-clock randomness.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import pytest
import yaml

if TYPE_CHECKING:
    from pathlib import Path

from hydra2.contracts.common import ContractError
from hydra2.training._rc_digest import run_config_digest, run_config_to_dict
from hydra2.training._rc_resume import _check_compat, format_plan
from hydra2.training._rc_root import load_run_config
from hydra2.training.rl_config import describe_rl_plan, parse_rl_section

pytestmark = pytest.mark.contract_package("WP-11")


def _digest(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _valid_rl(**overrides: Any) -> dict[str, Any]:
    section: dict[str, Any] = {
        "schema_version": "1.0.0",
        "initial_checkpoint_id": _digest("initial-checkpoint"),
        "initial_model_spec_hash": _digest("initial-model-spec"),
        "master_seed_hex": "ab" * 32,
        "experiment_id": "rl-pilot-001",
        "split_scope": "selection_training",
        "generations": 8,
        "games_per_generation": 128,
        "native_workers": 16,
        "env_slots": 128,
        "max_lookahead": 1,
        "inference_cap": 128,
        "microbatch_cap": 128,
        "accumulation_cap": 32,
        "optimizer_minibatch_cap": 4096,
        "epochs": 1,
        "row_cap": 262144,
        "decision_cap_per_game": 9000,
        "max_updates_per_generation": 67,
        "compressed_byte_cap": 8589934592,
        "min_free_disk_bytes": 21474836480,
        "host_byte_budget": 12884901888,
        "gpu_cache_byte_budget": 1073741824,
        "clip_eps": 0.2,
        "eps_std": 1e-8,
        "w_value": 0.005,
        "w_bc": 0.0,
        "ppo_alpha": 0.01,
        "gamma": 1.0,
        "lam": 1.0,
        "optimizer_lr": 1e-5,
        "optimizer_beta1": 0.9,
        "optimizer_beta2": 0.999,
        "optimizer_weight_decay": 0.01,
        "head_lr_mult": 1.0,
        "gradient_clip_norm": 1.0,
        "warmup_updates": 0,
        "rules_hash": _digest("rules"),
        "utility_hash": _digest("utility"),
        "action_table_hash": _digest("action-table"),
        "event_schema_hash": _digest("event-schema"),
        "observation_schema_hash": _digest("observation-schema"),
        "packet_boundary_hash": _digest("packet-boundary"),
        "input_schema_hash": _digest("input-schema"),
        "wall_ledger_hash": _digest("wall-ledger"),
        "opponent_catalog_hash": _digest("opponent-catalog"),
        "wall_recipe_hash": _digest("wall-recipe"),
        "sampling_recipe_hash": _digest("sampling-recipe"),
        "return_recipe_hash": _digest("return-recipe"),
        "shuffle_recipe_hash": _digest("shuffle-recipe"),
        "core_revision": "479c1faeb33d082965eef8198f63261a79c0fce3",
    }
    section.update(overrides)
    return section


def _rl_mapping() -> dict[str, Any]:
    return {
        "run": {"id": "rl-pilot-001", "kind": "rl"},
        "weights": {"w_policy": 1.0, "w_value": 0.005},
        "optimizer": {
            "id": "adamw",
            "lr": 1e-5,
            "betas": [0.9, 0.999],
            "weight_decay": 0.01,
            "head_lr_mult": 1.0,
        },
        "scheduler": {"id": "constant", "warmup_updates": 0},
        "runtime": {
            "adapter_id": "plain_pytorch",
            "device": "cuda",
            "precision": "bf16_mixed",
            "compile_mode": "eager",
        },
        "loop": {
            "microbatch_size": 128,
            "accumulation_steps": 32,
            "gradient_clip_norm": 1.0,
            "max_updates": 67,
            "checkpoint_frequency_updates": 1,
            "precision": "bf16_mixed",
            "stratified_sampling": False,
        },
        "rl": _valid_rl(),
    }


def _write_yaml(directory: Path, name: str, mapping: dict[str, Any]) -> Path:
    path = directory / name
    path.write_text(yaml.safe_dump(mapping, sort_keys=True), encoding="utf-8")
    return path


class TestValidProfile:
    def test_full_rl_config_loads(self, tmp_path: Path) -> None:
        config = load_run_config(_write_yaml(tmp_path, "rl.yaml", _rl_mapping()), environ={})
        assert config.run.kind == "rl"
        assert config.data is None
        assert config.rl is not None
        assert config.rl.generations == 8
        assert config.rl.games_per_generation == 128
        assert config.rl.eps_std == 1e-8
        assert config.rl.core_revision == "479c1faeb33d082965eef8198f63261a79c0fce3"
        assert config.rl.replicate_id == 0
        assert config.rl.attempt_id == 0

    def test_explicit_integer_scopes_accept(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        assert isinstance(mapping["rl"], dict)
        mapping["rl"].update({"replicate_id": 2, "attempt_id": 3})
        config = load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})
        assert config.rl is not None
        assert (config.rl.replicate_id, config.rl.attempt_id) == (2, 3)

    def test_string_scope_rejects(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        assert isinstance(mapping["rl"], dict)
        mapping["rl"]["replicate_id"] = "replicate-0"
        with pytest.raises(ContractError, match="non-negative int"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})

    def test_bool_scope_rejects(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        assert isinstance(mapping["rl"], dict)
        mapping["rl"]["attempt_id"] = True
        with pytest.raises(ContractError, match="non-negative int"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})

    def test_negative_scope_rejects(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        assert isinstance(mapping["rl"], dict)
        mapping["rl"]["attempt_id"] = -1
        with pytest.raises(ContractError, match="non-negative int"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})
        mapping = _rl_mapping()
        assert isinstance(mapping["rl"], dict)
        mapping["rl"].update(
            {"native_workers": 8, "env_slots": 64, "microbatch_cap": 64, "accumulation_cap": 16}
        )
        assert isinstance(mapping["loop"], dict)
        mapping["loop"].update({"microbatch_size": 64, "accumulation_steps": 16})
        config = load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})
        assert config.rl is not None and config.rl.native_workers == 8
        assert config.loop.microbatch_size == 64

    def test_dict_round_trip_carries_rl_and_null_data(self, tmp_path: Path) -> None:
        config = load_run_config(_write_yaml(tmp_path, "rl.yaml", _rl_mapping()), environ={})
        framed = run_config_to_dict(config)
        assert framed["data"] is None
        assert framed["rl"]["experiment_id"] == "rl-pilot-001"
        assert "rl" in framed
        assert run_config_digest(config).startswith("sha256:")

    def test_parse_section_entry_point(self) -> None:
        parsed = parse_rl_section(_valid_rl())
        assert parsed.experiment_id == "rl-pilot-001"
        assert parsed.w_bc == 0.0


class TestClosedKeys:
    def test_unknown_rl_key_raises(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        assert isinstance(mapping["rl"], dict)
        mapping["rl"]["bogus_knob"] = 1
        with pytest.raises(ContractError, match="unknown keys"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})

    def test_unknown_top_level_key_still_raises(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        mapping["bogus_section"] = {"x": 1}
        with pytest.raises(ContractError, match="unknown keys"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})

    def test_non_mapping_rl_raises(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        mapping["rl"] = ["not", "a", "mapping"]
        with pytest.raises(ContractError, match="rl must be a mapping"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})


class TestSourceUnion:
    def test_rl_on_supervised_is_parked(self, tmp_path: Path) -> None:
        mapping = {
            "run": {"id": "sup-001", "kind": "supervised"},
            "data": {"root": str(tmp_path / "corpus")},
            "rl": _valid_rl(),
        }
        with pytest.raises(ContractError, match="parked"):
            load_run_config(_write_yaml(tmp_path, "run.yaml", mapping), environ={})

    def test_rl_on_distill_requires_rl_kind(self, tmp_path: Path) -> None:
        mapping = {
            "run": {"id": "dist-001", "kind": "distill"},
            "data": {"root": str(tmp_path / "corpus")},
            "rl": _valid_rl(),
        }
        with pytest.raises(ContractError, match=r"requires run.kind 'rl'"):
            load_run_config(_write_yaml(tmp_path, "run.yaml", mapping), environ={})

    def test_distill_keeps_dataset_requirements(self, tmp_path: Path) -> None:
        mapping = {
            "run": {"id": "dist-001", "kind": "distill"},
            "data": {"root": str(tmp_path / "corpus")},
        }
        config = load_run_config(_write_yaml(tmp_path, "run.yaml", mapping), environ={})
        assert config.run.kind == "distill"
        assert config.data is not None and len(config.data.roots) == 1
        assert config.rl is None

    def test_missing_rl_section_raises(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        del mapping["rl"]
        with pytest.raises(ContractError, match="requires an explicit 'rl' section"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})

    def test_dataset_roots_forbidden_on_rl(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        mapping["data"] = {"root": str(tmp_path / "corpus")}
        with pytest.raises(ContractError, match="forbids dataset roots"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})

    def test_nonempty_data_section_rejected_on_rl(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        mapping["data"] = {"scope": "tenhou-4p-hanchan"}
        with pytest.raises(ContractError, match="must omit the data section"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})

    def test_empty_data_section_tolerated_on_rl(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        mapping["data"] = {}
        config = load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})
        assert config.data is None and config.rl is not None


class TestFrozenRecipe:
    def test_wrong_clip_eps_raises(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        assert isinstance(mapping["rl"], dict)
        mapping["rl"]["clip_eps"] = 0.3
        with pytest.raises(ContractError, match=r"rl.clip_eps must be exactly 0.2"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})

    def test_wrong_eps_std_raises(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        assert isinstance(mapping["rl"], dict)
        mapping["rl"]["eps_std"] = 1e-6
        with pytest.raises(ContractError, match=r"rl.eps_std must be exactly"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})

    def test_bool_rejected_for_int_slot(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        assert isinstance(mapping["rl"], dict)
        mapping["rl"]["generations"] = True
        with pytest.raises(ContractError, match=r"rl.generations must be exactly 8"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})

    def test_over_cap_workers_raise(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        assert isinstance(mapping["rl"], dict)
        mapping["rl"]["native_workers"] = 21
        with pytest.raises(ContractError, match="positive int <= 20"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})

    def test_over_cap_microbatch_raises(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        assert isinstance(mapping["rl"], dict)
        mapping["rl"]["microbatch_cap"] = 256
        with pytest.raises(ContractError, match="positive int <= 128"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})

    def test_wrong_optimizer_lr_raises(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        assert isinstance(mapping["optimizer"], dict)
        mapping["optimizer"]["lr"] = 3e-4
        with pytest.raises(ContractError, match=r"rl optimizer.lr must be exactly"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})

    def test_cosine_scheduler_raises(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        assert isinstance(mapping["scheduler"], dict)
        mapping["scheduler"]["id"] = "cosine"
        mapping["scheduler"]["warmup_updates"] = 100
        with pytest.raises(ContractError, match="rl scheduler must be 'constant'"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})

    def test_stratified_loop_raises(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        assert isinstance(mapping["loop"], dict)
        mapping["loop"]["stratified_sampling"] = True
        with pytest.raises(ContractError, match="must stay false"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})

    def test_over_cap_max_updates_raises(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        assert isinstance(mapping["loop"], dict)
        mapping["loop"]["max_updates"] = 68
        with pytest.raises(ContractError, match="positive int <= 67"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})

    def test_weights_value_mirror_raises(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        assert isinstance(mapping["weights"], dict)
        mapping["weights"]["w_value"] = 0.0
        with pytest.raises(ContractError, match=r"rl weights.w_value must be exactly 0.005"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})

    def test_minibatch_product_over_cap_raises(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        assert isinstance(mapping["loop"], dict)
        mapping["loop"].update({"microbatch_size": 128, "accumulation_steps": 33})
        mapping["rl"] = _valid_rl(accumulation_cap=33)
        with pytest.raises(ContractError, match="positive int <= 32"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})


class TestIdentities:
    def test_malformed_digest_raises(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        assert isinstance(mapping["rl"], dict)
        mapping["rl"]["wall_ledger_hash"] = "not-a-digest"
        with pytest.raises(ContractError, match="sha256"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})

    def test_missing_identity_raises(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        assert isinstance(mapping["rl"], dict)
        del mapping["rl"]["return_recipe_hash"]
        with pytest.raises(ContractError, match=r"rl.return_recipe_hash is required"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})

    def test_bad_master_seed_raises(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        assert isinstance(mapping["rl"], dict)
        mapping["rl"]["master_seed_hex"] = "xyz"
        with pytest.raises(ContractError, match="64 lowercase hex"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})

    def test_bad_core_revision_raises(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        assert isinstance(mapping["rl"], dict)
        mapping["rl"]["core_revision"] = "main"
        with pytest.raises(ContractError, match="40 lowercase hex"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})

    def test_wrong_schema_version_raises(self, tmp_path: Path) -> None:
        mapping = _rl_mapping()
        assert isinstance(mapping["rl"], dict)
        mapping["rl"]["schema_version"] = "2.0.0"
        with pytest.raises(ContractError, match=r"rl.schema_version must be '1.0.0'"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})


class TestSupervisedStable:
    def test_supervised_dict_has_no_rl_key(self, tmp_path: Path) -> None:
        mapping = {
            "run": {"id": "sup-001", "kind": "supervised"},
            "data": {"root": str(tmp_path / "corpus")},
        }
        config = load_run_config(_write_yaml(tmp_path, "run.yaml", mapping), environ={})
        assert config.rl is None
        assert config.data is not None
        framed = run_config_to_dict(config)
        assert "rl" not in framed
        assert framed["data"]["roots"] == [["corpus", str(tmp_path / "corpus")]]


class TestDryPlan:
    def test_describe_rl_plan(self, tmp_path: Path) -> None:
        config = load_run_config(_write_yaml(tmp_path, "rl.yaml", _rl_mapping()), environ={})
        text = describe_rl_plan(config)
        assert "experiment=rl-pilot-001" in text
        assert "INSIDE sqrt" in text
        assert "cohort generations=8 games_per_generation=128" in text
        assert "weight-only load; fresh optimizer" in text
        assert "first full optimizer update" in text

    def test_describe_without_rl_raises(self, tmp_path: Path) -> None:
        mapping = {
            "run": {"id": "sup-001", "kind": "supervised"},
            "data": {"root": str(tmp_path / "corpus")},
        }
        config = load_run_config(_write_yaml(tmp_path, "run.yaml", mapping), environ={})
        with pytest.raises(ContractError, match=r"requires run.kind 'rl'"):
            describe_rl_plan(config)

    def test_format_plan_carries_rl_line(self, tmp_path: Path) -> None:
        config = load_run_config(_write_yaml(tmp_path, "rl.yaml", _rl_mapping()), environ={})
        plan = format_plan(config, artifact_root=tmp_path / "artifacts")
        assert "(kind=rl)" in plan
        assert "immutable artifact ledger" in plan
        assert "rl: experiment=rl-pilot-001" in plan

    def test_resume_compat_owned_by_control_path(self, tmp_path: Path) -> None:
        from hydra2.training._rc_sections import AccumState, ShuffleState, StreamCursor, WorkerPlan

        config = load_run_config(_write_yaml(tmp_path, "rl.yaml", _rl_mapping()), environ={})
        from hydra2.training._rc_resume import _Sidecar

        sidecar = _Sidecar(
            global_update=1,
            run_digest=run_config_digest(config),
            cursor=StreamCursor(file_index=0, byte_offset=0, games_seen=0, seed=0, epoch=0),
            rng_state={},
            shuffle=ShuffleState(buffer_keys=(), epoch_seed=0, buffer_size=8),
            accum=AccumState(yielded_batches=1, micro_in_update=0),
            worker_plan=WorkerPlan(num_workers=0, world_size=1, rank=0),
            manifests={},
        )
        with pytest.raises(ContractError, match="control-snapshot"):
            _check_compat(config, run_config_digest(config), sidecar, tmp_path / "ckpt.pt")


class TestRlCompileIntake:
    def test_replay_arm_parses_on_cuda_bf16(self, tmp_path: Path) -> None:
        # Valid owned RL replay config: exact arm plus configured CUDA and
        # current bf16 numerics parses and carries through to identity.
        mapping = _rl_mapping()
        mapping["runtime"] = {
            "adapter_id": "plain_pytorch",
            "device": "cuda",
            "precision": "bf16_mixed",
            "compile_mode": "cuda_graph_replay",
        }
        config = load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})
        assert config.runtime.compile_mode == "cuda_graph_replay"
        framed = run_config_to_dict(config)
        assert framed["runtime"]["compile_mode"] == "cuda_graph_replay"

    def test_replay_arm_changes_run_digest(self, tmp_path: Path) -> None:
        # Runtime choice binds identity: replay vs eager digests differ.
        eager = load_run_config(_write_yaml(tmp_path, "eager.yaml", _rl_mapping()), environ={})
        mapping = _rl_mapping()
        mapping["runtime"] = dict(mapping["runtime"], compile_mode="cuda_graph_replay")
        replay = load_run_config(_write_yaml(tmp_path, "replay.yaml", mapping), environ={})
        assert run_config_digest(replay) != run_config_digest(eager)

    def test_torch_compile_mode_rejected_on_rl(self, tmp_path: Path) -> None:
        # torch.compile modes never apply to the owned RL route.
        mapping = _rl_mapping()
        mapping["runtime"] = dict(mapping["runtime"], compile_mode="default")
        with pytest.raises(ContractError, match="unsupported for rl"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})

    def test_cpu_rejected_for_all_rl(self, tmp_path: Path) -> None:
        # Owned RL requires CUDA for every arm (rl_gpu requires it; replay
        # never runs on CPU either): never silent CPU fallback.
        for mode in ("eager", "cuda_graph_replay"):
            mapping = _rl_mapping()
            mapping["runtime"] = dict(mapping["runtime"], device="cpu", compile_mode=mode)
            with pytest.raises(ContractError, match="CUDA device"):
                load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})

    def test_fp32_rejected_for_owned_rl(self, tmp_path: Path) -> None:
        # No FP32 arm exists (services hardcode the BF16 path): fp32 fails
        # closed for owned RL, eager and replay alike. SFT fp32 is untouched.
        for mode in ("eager", "cuda_graph_replay"):
            mapping = _rl_mapping()
            mapping["runtime"] = dict(mapping["runtime"], precision="fp32", compile_mode=mode)
            mapping["loop"] = dict(mapping["loop"], precision="fp32")
            with pytest.raises(ContractError, match="bf16-only"):
                load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})

    def test_unsupported_precision_rejected_not_relabeled(self, tmp_path: Path) -> None:
        # Unknown precisions fail closed instead of relabeling into bf16.
        mapping = _rl_mapping()
        mapping["runtime"] = dict(mapping["runtime"], precision="fp16_mixed")
        with pytest.raises(ContractError, match="precision"):
            load_run_config(_write_yaml(tmp_path, "rl.yaml", mapping), environ={})
