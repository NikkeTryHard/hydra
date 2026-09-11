"""WP-14 streaming run-config: strict load, interpolation, layout, resume.

Covers the RunConfig surface only (no training execution): unknown keys
raise, ``${VAR}`` interpolation honors the allowlist, the output layout is
created idempotently under the artifact root, and resume-plan resolution
picks the latest compatible checkpoint (digest + worker plan + seed gates).
All fixtures are worker-local ``tmp_path`` trees with fixed seeds; no
sockets, no sleeps, no wall-clock randomness.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
import yaml

if TYPE_CHECKING:
    from pathlib import Path

from hydra2.contracts.common import ContractError
from hydra2.training.run_config import (
    StreamCursor,
    create_run_layout,
    deep_merge,
    find_latest_checkpoint,
    format_plan,
    load_run_config,
    read_latest_run,
    resolve_resume_plan,
    run_config_digest,
    run_config_to_dict,
)


def _write_yaml(directory: Path, name: str, mapping: dict[str, Any]) -> Path:
    path = directory / name
    path.write_text(yaml.safe_dump(mapping, sort_keys=True), encoding="utf-8")
    return path


def _minimal_mapping(root: Path, run_id: str = "wp14-run-001") -> dict[str, Any]:
    return {
        "run": {"id": run_id, "kind": "supervised", "description": "wp14 fixture"},
        "data": {"root": str(root / "corpus")},
    }


def _sidecar(
    update: int,
    digest: str,
    *,
    seed: int = 0,
    workers: int = 0,
    world: int = 1,
    micro_in_update: int = 0,
    buffer_keys: list[str] | None = None,
) -> dict[str, Any]:
    keys = buffer_keys if buffer_keys is not None else []
    return {
        "global_update": update,
        "run_digest": digest,
        "stream_cursor": {
            "file_index": 0,
            "byte_offset": 0,
            "games_seen": update * 2,
            "seed": seed,
            "epoch": 0,
        },
        "rng_state": {"torch_cpu": [1, 2, 3], "semantic": {"seed_hex": "ab" * 32}},
        "shuffle": {
            "buffer_keys": keys,
            "epoch_seed": seed,
            "buffer_size": max(len(keys), 8),
            "buffer_rng_state": {},
        },
        "accum": {"yielded_batches": update, "micro_in_update": micro_in_update},
        "worker_plan": {"num_workers": workers, "world_size": world, "rank": 0},
        "manifests": {"stream_manifest_hash": None, "dataset_manifest_hash": None},
    }


def _write_ckpt(run_dir: Path, update: int, digest: str, **kwargs: Any) -> Path:
    ckpt = run_dir / "checkpoints" / f"ckpt-{update:06d}.pt"
    ckpt.write_bytes(b"fake-ckpt-%06d" % update)
    ckpt.with_suffix(".json").write_text(json.dumps(_sidecar(update, digest, **kwargs)))
    return ckpt


class TestStrictKeys:
    def test_unknown_top_level_key_raises(self, tmp_path: Path) -> None:
        mapping = _minimal_mapping(tmp_path)
        mapping["bogus_section"] = {"x": 1}
        path = _write_yaml(tmp_path, "run.yaml", mapping)
        with pytest.raises(ContractError, match="unknown keys"):
            load_run_config(path, environ={})

    def test_unknown_nested_key_raises(self, tmp_path: Path) -> None:
        mapping = _minimal_mapping(tmp_path)
        mapping["loop"] = {"microbatch_size": 2, "bogus_key": 1}
        path = _write_yaml(tmp_path, "run.yaml", mapping)
        with pytest.raises(ContractError, match="unknown keys"):
            load_run_config(path, environ={})

    def test_unknown_weight_pattern_raises(self, tmp_path: Path) -> None:
        mapping = _minimal_mapping(tmp_path)
        mapping["weights"] = {"w_policy": 1.0, "belief_fields": ["x"]}
        path = _write_yaml(tmp_path, "run.yaml", mapping)
        with pytest.raises(ContractError, match="unknown keys"):
            load_run_config(path, environ={})

    def test_parked_rl_null_loads_but_set_raises(self, tmp_path: Path) -> None:
        mapping = _minimal_mapping(tmp_path)
        mapping["rl"] = None
        assert load_run_config(_write_yaml(tmp_path, "a.yaml", mapping), environ={}).run.run_id
        mapping["rl"] = {"objective": "ppo"}
        with pytest.raises(ContractError, match="parked"):
            load_run_config(_write_yaml(tmp_path, "b.yaml", mapping), environ={})


class TestInterpolation:
    def test_allowlisted_var_interpolates(self, tmp_path: Path) -> None:
        mapping = _minimal_mapping(tmp_path)
        mapping["data"] = {"root": "${HOME}/corpus"}
        config = load_run_config(
            _write_yaml(tmp_path, "run.yaml", mapping), environ={"HOME": str(tmp_path)}
        )
        assert config.data.root == str(tmp_path / "corpus")

    def test_non_allowlisted_var_raises(self, tmp_path: Path) -> None:
        mapping = _minimal_mapping(tmp_path)
        mapping["data"] = {"root": "${SUPER_SECRET_TOKEN}/corpus"}
        with pytest.raises(ContractError, match="not allowlisted"):
            load_run_config(
                _write_yaml(tmp_path, "run.yaml", mapping),
                environ={"SUPER_SECRET_TOKEN": "x"},
            )

    def test_unset_allowlisted_var_raises(self, tmp_path: Path) -> None:
        mapping = _minimal_mapping(tmp_path)
        mapping["data"] = {"root": "${HYDRA2_DATA_ROOT}/corpus"}
        with pytest.raises(ContractError, match="unset or empty"):
            load_run_config(_write_yaml(tmp_path, "run.yaml", mapping), environ={})


class TestMerge:
    def test_override_wins_keywise_and_base_survives(self, tmp_path: Path) -> None:
        base = _minimal_mapping(tmp_path)
        base["loop"] = {"microbatch_size": 4, "max_updates": 100}
        override = {"loop": {"max_updates": 10}, "run": {"id": "wp14-run-002"}}
        merged = deep_merge(base, override)
        assert merged["loop"] == {"microbatch_size": 4, "max_updates": 10}
        assert merged["run"]["id"] == "wp14-run-002"
        assert merged["data"]["root"] == base["data"]["root"]

    def test_load_with_override_file(self, tmp_path: Path) -> None:
        base_path = _write_yaml(tmp_path, "base.yaml", _minimal_mapping(tmp_path))
        override_path = _write_yaml(tmp_path, "over.yaml", {"loop": {"max_updates": 7}})
        config = load_run_config(base_path, override_path=override_path, environ={})
        assert config.loop.max_updates == 7
        assert config.loop.microbatch_size == 4


class TestBindingGates:
    def test_positive_aux_weight_requires_privileged_binding(self, tmp_path: Path) -> None:
        mapping = _minimal_mapping(tmp_path)
        mapping["weights"] = {"w_policy": 1.0, "w_event": {"goal": 0.5}}
        with pytest.raises(ContractError, match="positive-requires-binding"):
            load_run_config(_write_yaml(tmp_path, "run.yaml", mapping), environ={})

    def test_bound_aux_weight_loads(self, tmp_path: Path) -> None:
        mapping = _minimal_mapping(tmp_path)
        digest = "sha256:" + "c" * 64
        mapping["weights"] = {
            "w_policy": 1.0,
            "w_event": {"goal": 0.5},
            "privileged_source_hash": digest,
        }
        config = load_run_config(_write_yaml(tmp_path, "run.yaml", mapping), environ={})
        assert config.weights.privileged_source_hash == digest

    def test_bf16_on_cpu_rejected(self, tmp_path: Path) -> None:
        mapping = _minimal_mapping(tmp_path)
        mapping["runtime"] = {"device": "cpu", "precision": "bf16_mixed"}
        mapping["loop"] = {"precision": "bf16_mixed"}
        with pytest.raises(ContractError, match="CUDA device"):
            load_run_config(_write_yaml(tmp_path, "run.yaml", mapping), environ={})


class TestLayout:
    def test_layout_creation_and_idempotent_recreate(self, tmp_path: Path) -> None:
        config = load_run_config(
            _write_yaml(tmp_path, "run.yaml", _minimal_mapping(tmp_path)), environ={}
        )
        root = tmp_path / "artifacts"
        run_dir = create_run_layout(config, artifact_root=root)
        assert run_dir == root / "runs" / "wp14-run-001"
        for sub in ("manifests", "checkpoints", "logs", "mirror", "eval"):
            assert (run_dir / sub).is_dir()
        assert (run_dir / "logs" / "train.log").is_file()
        assert (run_dir / "logs" / "metrics.jsonl").is_file()
        assert read_latest_run(root) == "wp14-run-001"
        reloaded = load_run_config(run_dir / "run.yaml", environ={})
        assert run_config_digest(reloaded) == run_config_digest(config)
        # Byte-identical recreate is a no-op.
        assert create_run_layout(config, artifact_root=root) == run_dir

    def test_conflicting_spec_same_id_raises(self, tmp_path: Path) -> None:
        mapping = _minimal_mapping(tmp_path)
        config = load_run_config(_write_yaml(tmp_path, "a.yaml", mapping), environ={})
        root = tmp_path / "artifacts"
        create_run_layout(config, artifact_root=root)
        mapping["loop"] = {"max_updates": 11}
        other = load_run_config(_write_yaml(tmp_path, "b.yaml", mapping), environ={})
        with pytest.raises(ContractError, match="refusing to overwrite"):
            create_run_layout(other, artifact_root=root)


class TestResume:
    def _run_dir_with_ckpts(self, tmp_path: Path) -> tuple[Path, str]:
        config = load_run_config(
            _write_yaml(tmp_path, "run.yaml", _minimal_mapping(tmp_path)), environ={}
        )
        run_dir = create_run_layout(config, artifact_root=tmp_path / "artifacts")
        digest = run_config_digest(config)
        _write_ckpt(run_dir, 100, digest)
        _write_ckpt(run_dir, 200, digest, buffer_keys=["k1", "k2"])
        # Torn write: corrupt sidecar is skipped, never a resume point.
        torn = run_dir / "checkpoints" / "ckpt-000300.pt"
        torn.write_bytes(b"fake-ckpt-000300")
        torn.with_suffix(".json").write_text("{not json", encoding="utf-8")
        # Filename/sidecar update mismatch is skipped as well.
        stray = run_dir / "checkpoints" / "ckpt-000150.pt"
        stray.write_bytes(b"fake-ckpt-000150")
        stray.with_suffix(".json").write_text(json.dumps(_sidecar(151, digest)))
        return run_dir, digest

    def test_latest_valid_checkpoint_skips_torn_writes(self, tmp_path: Path) -> None:
        run_dir, _ = self._run_dir_with_ckpts(tmp_path)
        assert find_latest_checkpoint(run_dir) is not None
        found = find_latest_checkpoint(run_dir)
        assert found is not None and found.name == "ckpt-000200.pt"

    def test_resolve_latest_returns_compat_plan(self, tmp_path: Path) -> None:
        run_dir, digest = self._run_dir_with_ckpts(tmp_path)
        plan = resolve_resume_plan(run_dir)
        assert plan.global_update == 200
        assert plan.run_digest == digest
        assert plan.cursor.games_seen == 400
        assert plan.shuffle.buffer_keys == ("k1", "k2")
        assert plan.accum.micro_in_update == 0
        assert (plan.worker_plan.num_workers, plan.worker_plan.world_size) == (0, 1)

    def test_resolve_explicit_checkpoint(self, tmp_path: Path) -> None:
        run_dir, _ = self._run_dir_with_ckpts(tmp_path)
        target = run_dir / "checkpoints" / "ckpt-000100.pt"
        assert resolve_resume_plan(run_dir, which=target).global_update == 100

    def test_config_drift_rejected(self, tmp_path: Path) -> None:
        run_dir, _ = self._run_dir_with_ckpts(tmp_path)
        tampered = _minimal_mapping(tmp_path)
        tampered["loop"] = {"max_updates": 999}
        other = load_run_config(_write_yaml(tmp_path, "other.yaml", tampered), environ={})
        (run_dir / "run.yaml").write_text(
            yaml.safe_dump(run_config_to_dict(other), sort_keys=True),
            encoding="utf-8",
        )
        with pytest.raises(ContractError, match="run_digest"):
            resolve_resume_plan(run_dir)

    def test_worker_plan_mismatch_rejected(self, tmp_path: Path) -> None:
        run_dir, digest = self._run_dir_with_ckpts(tmp_path)
        _write_ckpt(run_dir, 400, digest, workers=4)
        with pytest.raises(ContractError, match="num_workers"):
            resolve_resume_plan(run_dir, which=run_dir / "checkpoints" / "ckpt-000400.pt")

    def test_empty_run_dir_raises(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "runs" / "ghost"
        run_dir.mkdir(parents=True)
        with pytest.raises(ContractError, match=r"no run\.yaml"):
            resolve_resume_plan(run_dir)


class TestCursorContract:
    def test_cursor_round_trip(self) -> None:
        cursor = StreamCursor(file_index=3, byte_offset=128, games_seen=10, seed=7, epoch=2)
        assert StreamCursor.from_dict(cursor.to_dict()) == cursor

    def test_cursor_envelope_mismatch_raises(self) -> None:
        with pytest.raises(ContractError, match="envelope mismatch"):
            StreamCursor.from_dict({"file_index": 0})
        with pytest.raises(ContractError, match="non-negative int"):
            StreamCursor.from_dict(
                {"file_index": -1, "byte_offset": 0, "games_seen": 0, "seed": 0, "epoch": 0}
            )


class TestPlanFormat:
    def test_plan_prints_ids_and_digest(self, tmp_path: Path) -> None:
        config = load_run_config(
            _write_yaml(tmp_path, "run.yaml", _minimal_mapping(tmp_path)), environ={}
        )
        text = format_plan(config, artifact_root=tmp_path / "artifacts")
        assert "wp14-run-001" in text
        assert run_config_digest(config) in text
        assert "resume: fresh start" in text


class TestTelemetrySection:
    def test_absent_defaults(self, tmp_path: Path) -> None:
        """No telemetry block: mlflow on, verbose off, no captures."""
        path = _write_yaml(tmp_path, "run.yaml", _minimal_mapping(tmp_path))
        config = load_run_config(path)
        assert config.telemetry.mlflow_enabled is True
        assert config.telemetry.verbose_enabled is False
        assert config.telemetry.verbose_interval_ms == 50
        assert config.telemetry.profiler_captures == 0

    def test_explicit_values(self, tmp_path: Path) -> None:
        mapping = _minimal_mapping(tmp_path)
        mapping["telemetry"] = {
            "mlflow_enabled": False,
            "verbose_enabled": True,
            "verbose_interval_ms": 20,
            "profiler_captures": 3,
        }
        config = load_run_config(_write_yaml(tmp_path, "run.yaml", mapping))
        assert config.telemetry.mlflow_enabled is False
        assert config.telemetry.verbose_enabled is True
        assert config.telemetry.verbose_interval_ms == 20
        assert config.telemetry.profiler_captures == 3

    def test_strict_rejects(self, tmp_path: Path) -> None:
        bad_interval = _minimal_mapping(tmp_path)
        bad_interval["telemetry"] = {"verbose_interval_ms": 100}
        with pytest.raises(ContractError):
            load_run_config(_write_yaml(tmp_path, "bad-interval.yaml", bad_interval))
        bad_captures = _minimal_mapping(tmp_path)
        bad_captures["telemetry"] = {"profiler_captures": -1}
        with pytest.raises(ContractError):
            load_run_config(_write_yaml(tmp_path, "bad-captures.yaml", bad_captures))
        bad_key = _minimal_mapping(tmp_path)
        bad_key["telemetry"] = {"nope": True}
        with pytest.raises(ContractError):
            load_run_config(_write_yaml(tmp_path, "bad-key.yaml", bad_key))

    def test_plan_reports_telemetry(self, tmp_path: Path) -> None:
        path = _write_yaml(tmp_path, "run.yaml", _minimal_mapping(tmp_path))
        plan = format_plan(load_run_config(path), artifact_root=tmp_path)
        assert "telemetry: mlflow=True verbose=False@50ms profiler=0" in plan


class TestFeedKnobs:
    def test_defaults(self, tmp_path: Path) -> None:
        """Absent knobs: decode 64, batch 64, loop depth 3."""
        config = load_run_config(_write_yaml(tmp_path, "run.yaml", _minimal_mapping(tmp_path)))
        assert config.data.decode_prefetch == 64
        assert config.data.expand_batch_games == 64
        assert config.loop.fetch_prefetch_depth == 3

    def test_explicit_values_round_trip(self, tmp_path: Path) -> None:
        mapping = _minimal_mapping(tmp_path)
        mapping["data"]["decode_prefetch"] = 32
        mapping["data"]["expand_batch_games"] = 16
        mapping["loop"] = {"fetch_prefetch_depth": 2}
        config = load_run_config(_write_yaml(tmp_path, "run.yaml", mapping))
        assert config.data.decode_prefetch == 32
        assert config.data.expand_batch_games == 16
        assert config.loop.fetch_prefetch_depth == 2
        again = run_config_to_dict(config)
        assert again["data"]["decode_prefetch"] == 32
        assert again["data"]["expand_batch_games"] == 16
        assert again["loop"]["fetch_prefetch_depth"] == 2

    def test_strict_rejects(self, tmp_path: Path) -> None:
        for section, key, bad in (
            ("data", "decode_prefetch", 0),
            ("data", "decode_prefetch", 2048),
            ("data", "expand_batch_games", -1),
            ("loop", "fetch_prefetch_depth", 0),
            ("loop", "fetch_prefetch_depth", 17),
            ("loop", "fetch_prefetch_depth", True),
        ):
            mapping = _minimal_mapping(tmp_path)
            mapping.setdefault(section, {})[key] = bad
            with pytest.raises(ContractError):
                load_run_config(_write_yaml(tmp_path, "bad.yaml", mapping))
