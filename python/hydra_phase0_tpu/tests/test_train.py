from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import hashlib
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import jax_utils

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hydra_phase0_tpu.export_weights import export_weights
from hydra_phase0_tpu.dataset import ExportDataset
from hydra_phase0_tpu.model import HydraModel, HydraOutput
from hydra_phase0_tpu.config import ExperimentConfig
from hydra_phase0_tpu.train import (
    _build_model,
    _can_shard_batch,
    _effective_run_config,
    _runtime_info,
    _validate,
    _shard_batch,
)
from hydra_phase0_tpu.train_state import (
    create_learning_rate_schedule,
    create_train_state,
)
from hydra_phase0_tpu.train_step import (
    train_logical_batch,
    train_logical_batch_data_parallel,
    train_step,
)
from hydra_phase0_tpu.eval_step import eval_step
from hydra_phase0_tpu.losses import total_loss


def write_npy(path: Path, array: np.ndarray) -> None:
    np.save(path, array)


def shard_file_hashes(root: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.iterdir())
        if path.is_file()
    }


def python_command() -> list[str]:
    return ["uv", "run", "--project", str(ROOT), "python"]


class TrainSmokeTests(unittest.TestCase):
    def _write_fixture_export(
        self, tmp_root: Path, train_samples: int, validation_samples: int
    ) -> tuple[Path, Path]:
        export_root = tmp_root / "bc_phase0_export"
        train_root = export_root / "train" / "shard-000000"
        validation_root = export_root / "validation" / "shard-000000"
        train_root.mkdir(parents=True, exist_ok=True)
        validation_root.mkdir(parents=True, exist_ok=True)

        manifest = {
            "manifest_fingerprint": "abc123",
            "schema_version": "hydra_bc_phase0_v1",
            "export_semantics": "hydra_bc_phase0_v1",
            "encoder_contract": "192x34",
            "action_space": 46,
            "shards": [
                {
                    "split": "train",
                    "shard_name": "shard-000000",
                    "game_count": 1,
                    "sample_count": train_samples,
                    "hashes": {"files": {}},
                },
                {
                    "split": "validation",
                    "shard_name": "shard-000000",
                    "game_count": 1,
                    "sample_count": validation_samples,
                    "hashes": {"files": {}},
                },
            ],
        }

        def write_split(dirpath: Path, n: int) -> None:
            write_npy(
                dirpath / "game_sample_offsets.npy", np.array([0, n], dtype=np.int64)
            )
            obs = np.zeros((n, 192, 34), dtype=np.float32)
            for i in range(n):
                obs[i, 40, i % 34] = 1.0
                obs[i, 41, (i + 9) % 34] = 0.5
                obs[i, 42, (i + 18) % 34] = 0.25
            write_npy(dirpath / "obs.npy", obs)
            write_npy(dirpath / "action.npy", np.array([1] * n, dtype=np.uint8))
            legal_mask = np.zeros((n, 46), dtype=np.float32)
            legal_mask[:, 1] = 1.0
            legal_mask[:, 45] = 1.0
            write_npy(dirpath / "legal_mask.npy", legal_mask)
            write_npy(dirpath / "score_delta.npy", np.array([100] * n, dtype=np.int32))
            write_npy(
                dirpath / "value_target.npy", np.array([0.001] * n, dtype=np.float32)
            )
            grp_target = np.zeros((n, 24), dtype=np.float32)
            grp_target[:, 0] = 1.0
            write_npy(dirpath / "grp_target.npy", grp_target)
            write_npy(dirpath / "tenpai_target.npy", np.zeros((n, 3), dtype=np.float32))
            write_npy(
                dirpath / "danger_target.npy", np.zeros((n, 3, 34), dtype=np.float32)
            )
            write_npy(
                dirpath / "danger_mask.npy", np.zeros((n, 3, 34), dtype=np.float32)
            )
            write_npy(
                dirpath / "opp_next_target.npy", np.zeros((n, 3, 34), dtype=np.float32)
            )
            score_pdf_target = np.zeros((n, 64), dtype=np.float32)
            score_pdf_target[:, 29] = 1.0
            write_npy(dirpath / "score_pdf_target.npy", score_pdf_target)
            score_cdf_target = np.zeros((n, 64), dtype=np.float32)
            score_cdf_target[:, 29:] = 1.0
            write_npy(dirpath / "score_cdf_target.npy", score_cdf_target)
            write_npy(dirpath / "oracle_target.npy", np.zeros((n, 4), dtype=np.float32))
            write_npy(
                dirpath / "oracle_target_present.npy", np.zeros((n,), dtype=np.uint8)
            )
            write_npy(
                dirpath / "safety_residual_target.npy",
                np.zeros((n, 46), dtype=np.float32),
            )
            write_npy(
                dirpath / "safety_residual_present.npy", np.zeros((n,), dtype=np.uint8)
            )
            write_npy(
                dirpath / "safety_residual_mask.npy",
                np.zeros((n, 46), dtype=np.float32),
            )
            write_npy(
                dirpath / "belief_fields_target.npy",
                np.zeros((n, 16, 34), dtype=np.float32),
            )
            write_npy(
                dirpath / "belief_fields_present.npy", np.zeros((n,), dtype=np.uint8)
            )
            write_npy(
                dirpath / "mixture_weight_target.npy",
                np.zeros((n, 4), dtype=np.float32),
            )
            write_npy(
                dirpath / "mixture_weight_present.npy", np.zeros((n,), dtype=np.uint8)
            )

        write_split(train_root, train_samples)
        write_split(validation_root, validation_samples)
        manifest["shards"][0]["hashes"]["files"] = shard_file_hashes(train_root)
        manifest["shards"][1]["hashes"]["files"] = shard_file_hashes(validation_root)
        (export_root / "manifest.json").write_text(json.dumps(manifest))
        return export_root, tmp_root / "config.yaml"

    def test_train_smoke_prints_manifest_summary(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_train_test"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        export_root, config_path = self._write_fixture_export(
            tmp_root, train_samples=1, validation_samples=1
        )

        config_path.write_text(
            f"""
data:
  export_root: {export_root}
run:
  output_dir: {tmp_root / "out"}
  seed: 0
  num_epochs: 1
  smoke_test: true
""".strip()
        )

        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(ROOT) if not existing else f"{ROOT}:{existing}"
        result = subprocess.run(
            [
                *python_command(),
                "-m",
                "hydra_phase0_tpu.train",
                "--config",
                str(config_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertIn("manifest_fingerprint", result.stdout)
        self.assertIn("train_loss", result.stdout)

    def test_effective_run_config_enables_smoke_limits(self) -> None:
        config = ExperimentConfig.model_validate(
            {
                "data": {"export_root": "/tmp/export"},
                "run": {
                    "output_dir": "/tmp/out",
                    "seed": 0,
                    "num_epochs": 4,
                    "batch_size": 128,
                    "microbatch_size": 32,
                    "validation_microbatch_size": 64,
                    "max_validation_samples": 4096,
                    "validate_every_n_steps": 50,
                    "checkpoint_every_n_steps": 50,
                },
            }
        )
        run = _effective_run_config(config, cli_smoke_test=True)
        self.assertTrue(run.smoke_test)
        self.assertEqual(run.num_epochs, 1)
        self.assertEqual(run.max_validation_samples, 256)
        self.assertEqual(run.validate_every_n_steps, 1)
        self.assertEqual(run.checkpoint_every_n_steps, 1)

    def test_validate_honors_microbatch_size_and_sample_cap(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_validate_limits_test"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        export_root, _ = self._write_fixture_export(
            tmp_root, train_samples=1, validation_samples=5
        )
        dataset = ExportDataset(export_root)
        model = HydraModel(
            num_blocks=1, hidden_channels=32, num_groups=4, se_bottleneck=8
        )
        obs = jnp.zeros((1, 192, 34), dtype=jnp.float32)
        params = model.init(jax.random.PRNGKey(0), obs)["params"]
        state = create_train_state(
            model.apply,
            params,
            learning_rate=2.5e-4,
            min_learning_rate=1e-6,
            weight_decay=1e-5,
            grad_clip_norm=1.0,
            warmup_steps=1000,
        )
        validation = _validate(
            state,
            model,
            dataset.shards("validation"),
            validation_microbatch_size=2,
            max_samples=3,
        )
        self.assertEqual(validation["samples"], 3.0)

    def test_validate_rejects_missing_validation_samples(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_validate_empty_test"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        export_root = tmp_root / "bc_phase0_export"
        train_root = export_root / "train" / "shard-000000"
        validation_root = export_root / "validation" / "shard-000000"
        train_root.mkdir(parents=True, exist_ok=True)
        validation_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "manifest_fingerprint": "abc123",
            "schema_version": "hydra_bc_phase0_v1",
            "export_semantics": "hydra_bc_phase0_v1",
            "encoder_contract": "192x34",
            "action_space": 46,
            "shards": [
                {
                    "split": "train",
                    "shard_name": "shard-000000",
                    "game_count": 1,
                    "sample_count": 1,
                    "hashes": {"files": {}},
                },
                {
                    "split": "validation",
                    "shard_name": "shard-000000",
                    "game_count": 0,
                    "sample_count": 0,
                    "hashes": {"files": {}},
                },
            ],
        }
        (export_root / "manifest.json").write_text(json.dumps(manifest))
        write_npy(
            train_root / "game_sample_offsets.npy", np.array([0, 1], dtype=np.int64)
        )
        write_npy(train_root / "obs.npy", np.zeros((1, 192, 34), dtype=np.float32))
        write_npy(train_root / "action.npy", np.array([1], dtype=np.uint8))
        legal_mask = np.zeros((1, 46), dtype=np.float32)
        legal_mask[:, 1] = 1.0
        write_npy(train_root / "legal_mask.npy", legal_mask)
        write_npy(train_root / "score_delta.npy", np.zeros((1,), dtype=np.int32))
        write_npy(train_root / "value_target.npy", np.zeros((1,), dtype=np.float32))
        write_npy(train_root / "grp_target.npy", np.zeros((1, 24), dtype=np.float32))
        write_npy(train_root / "tenpai_target.npy", np.zeros((1, 3), dtype=np.float32))
        write_npy(
            train_root / "danger_target.npy", np.zeros((1, 3, 34), dtype=np.float32)
        )
        write_npy(
            train_root / "danger_mask.npy", np.zeros((1, 3, 34), dtype=np.float32)
        )
        write_npy(
            train_root / "opp_next_target.npy", np.zeros((1, 3, 34), dtype=np.float32)
        )
        write_npy(
            train_root / "score_pdf_target.npy", np.zeros((1, 64), dtype=np.float32)
        )
        write_npy(
            train_root / "score_cdf_target.npy", np.zeros((1, 64), dtype=np.float32)
        )
        write_npy(train_root / "oracle_target.npy", np.zeros((1, 4), dtype=np.float32))
        write_npy(
            train_root / "oracle_target_present.npy", np.zeros((1,), dtype=np.uint8)
        )
        write_npy(
            train_root / "safety_residual_target.npy",
            np.zeros((1, 46), dtype=np.float32),
        )
        write_npy(
            train_root / "safety_residual_present.npy", np.zeros((1,), dtype=np.uint8)
        )
        write_npy(
            train_root / "safety_residual_mask.npy", np.zeros((1, 46), dtype=np.float32)
        )
        write_npy(
            train_root / "belief_fields_target.npy",
            np.zeros((1, 16, 34), dtype=np.float32),
        )
        write_npy(
            train_root / "belief_fields_present.npy", np.zeros((1,), dtype=np.uint8)
        )
        write_npy(
            train_root / "mixture_weight_target.npy", np.zeros((1, 4), dtype=np.float32)
        )
        write_npy(
            train_root / "mixture_weight_present.npy", np.zeros((1,), dtype=np.uint8)
        )
        write_npy(
            validation_root / "game_sample_offsets.npy", np.array([0], dtype=np.int64)
        )
        write_npy(validation_root / "obs.npy", np.zeros((0, 192, 34), dtype=np.float32))
        write_npy(validation_root / "action.npy", np.zeros((0,), dtype=np.uint8))
        write_npy(
            validation_root / "legal_mask.npy", np.zeros((0, 46), dtype=np.float32)
        )
        write_npy(validation_root / "score_delta.npy", np.zeros((0,), dtype=np.int32))
        write_npy(
            validation_root / "value_target.npy", np.zeros((0,), dtype=np.float32)
        )
        write_npy(
            validation_root / "grp_target.npy", np.zeros((0, 24), dtype=np.float32)
        )
        write_npy(
            validation_root / "tenpai_target.npy", np.zeros((0, 3), dtype=np.float32)
        )
        write_npy(
            validation_root / "danger_target.npy",
            np.zeros((0, 3, 34), dtype=np.float32),
        )
        write_npy(
            validation_root / "danger_mask.npy", np.zeros((0, 3, 34), dtype=np.float32)
        )
        write_npy(
            validation_root / "opp_next_target.npy",
            np.zeros((0, 3, 34), dtype=np.float32),
        )
        write_npy(
            validation_root / "score_pdf_target.npy",
            np.zeros((0, 64), dtype=np.float32),
        )
        write_npy(
            validation_root / "score_cdf_target.npy",
            np.zeros((0, 64), dtype=np.float32),
        )
        write_npy(
            validation_root / "oracle_target.npy", np.zeros((0, 4), dtype=np.float32)
        )
        write_npy(
            validation_root / "oracle_target_present.npy",
            np.zeros((0,), dtype=np.uint8),
        )
        write_npy(
            validation_root / "safety_residual_target.npy",
            np.zeros((0, 46), dtype=np.float32),
        )
        write_npy(
            validation_root / "safety_residual_present.npy",
            np.zeros((0,), dtype=np.uint8),
        )
        write_npy(
            validation_root / "safety_residual_mask.npy",
            np.zeros((0, 46), dtype=np.float32),
        )
        write_npy(
            validation_root / "belief_fields_target.npy",
            np.zeros((0, 16, 34), dtype=np.float32),
        )
        write_npy(
            validation_root / "belief_fields_present.npy",
            np.zeros((0,), dtype=np.uint8),
        )
        write_npy(
            validation_root / "mixture_weight_target.npy",
            np.zeros((0, 4), dtype=np.float32),
        )
        write_npy(
            validation_root / "mixture_weight_present.npy",
            np.zeros((0,), dtype=np.uint8),
        )
        write_npy(
            validation_root / "game_sample_offsets.npy",
            np.array([0, 0], dtype=np.int64),
        )
        manifest["shards"][0]["hashes"]["files"] = shard_file_hashes(train_root)
        manifest["shards"][1]["hashes"]["files"] = shard_file_hashes(validation_root)
        (export_root / "manifest.json").write_text(json.dumps(manifest))

        dataset = ExportDataset(export_root)
        model = HydraModel(
            num_blocks=1, hidden_channels=32, num_groups=4, se_bottleneck=8
        )
        obs = jnp.zeros((1, 192, 34), dtype=jnp.float32)
        params = model.init(jax.random.PRNGKey(0), obs)["params"]
        state = create_train_state(
            model.apply,
            params,
            learning_rate=2.5e-4,
            min_learning_rate=1e-6,
            weight_decay=1e-5,
            grad_clip_norm=1.0,
            warmup_steps=1000,
        )
        with self.assertRaisesRegex(ValueError, "validation produced zero samples"):
            _validate(
                state,
                model,
                dataset.shards("validation"),
                validation_microbatch_size=1,
                max_samples=1,
            )

    def test_train_full_loop_writes_artifacts(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_full_train_test"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        export_root, config_path = self._write_fixture_export(
            tmp_root, train_samples=2, validation_samples=1
        )

        config_path.write_text(
            f"""
data:
  export_root: {export_root}
run:
  output_dir: {tmp_root / "out"}
  seed: 0
  num_epochs: 1
  smoke_test: false
  validate_every_n_steps: 1
  checkpoint_every_n_steps: 1
""".strip()
        )

        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(ROOT) if not existing else f"{ROOT}:{existing}"
        result = subprocess.run(
            [
                *python_command(),
                "-m",
                "hydra_phase0_tpu.train",
                "--config",
                str(config_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertIn("validation_total_loss", result.stdout)
        self.assertIn("backend", result.stdout)
        self.assertIn("device_kind", result.stdout)
        output_root = tmp_root / "out" / "bc_tpu_phase0"
        self.assertTrue((output_root / "latest_state.json").exists())
        self.assertTrue((output_root / "latest_orbax" / "metadata.json").exists())
        self.assertTrue((output_root / "training_log.jsonl").exists())
        self.assertFalse((output_root / "step_log.jsonl").exists())
        self.assertTrue((output_root / "best" / "latest_state.json").exists())
        latest_state = json.loads((output_root / "latest_state.json").read_text())
        self.assertIn(latest_state["backend"], {"cpu", "gpu", "tpu"})
        self.assertTrue(latest_state["device_kind"])

    def test_train_rejects_empty_first_train_shard(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_empty_train_shard_test"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        export_root = tmp_root / "bc_phase0_export"
        train_root = export_root / "train" / "shard-000000"
        validation_root = export_root / "validation" / "shard-000000"
        train_root.mkdir(parents=True, exist_ok=True)
        validation_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "manifest_fingerprint": "abc123",
            "schema_version": "hydra_bc_phase0_v1",
            "export_semantics": "hydra_bc_phase0_v1",
            "encoder_contract": "192x34",
            "action_space": 46,
            "shards": [
                {
                    "split": "train",
                    "shard_name": "shard-000000",
                    "game_count": 0,
                    "sample_count": 0,
                    "hashes": {"files": {}},
                },
                {
                    "split": "validation",
                    "shard_name": "shard-000000",
                    "game_count": 1,
                    "sample_count": 1,
                    "hashes": {"files": {}},
                },
            ],
        }
        (export_root / "manifest.json").write_text(json.dumps(manifest))
        write_npy(
            train_root / "game_sample_offsets.npy", np.array([0, 0], dtype=np.int64)
        )
        write_npy(train_root / "obs.npy", np.zeros((0, 192, 34), dtype=np.float32))
        write_npy(train_root / "action.npy", np.zeros((0,), dtype=np.uint8))
        write_npy(train_root / "legal_mask.npy", np.zeros((0, 46), dtype=np.float32))
        write_npy(train_root / "score_delta.npy", np.zeros((0,), dtype=np.int32))
        write_npy(train_root / "value_target.npy", np.zeros((0,), dtype=np.float32))
        write_npy(train_root / "grp_target.npy", np.zeros((0, 24), dtype=np.float32))
        write_npy(train_root / "tenpai_target.npy", np.zeros((0, 3), dtype=np.float32))
        write_npy(
            train_root / "danger_target.npy", np.zeros((0, 3, 34), dtype=np.float32)
        )
        write_npy(
            train_root / "danger_mask.npy", np.zeros((0, 3, 34), dtype=np.float32)
        )
        write_npy(
            train_root / "opp_next_target.npy", np.zeros((0, 3, 34), dtype=np.float32)
        )
        write_npy(
            train_root / "score_pdf_target.npy", np.zeros((0, 64), dtype=np.float32)
        )
        write_npy(
            train_root / "score_cdf_target.npy", np.zeros((0, 64), dtype=np.float32)
        )
        write_npy(train_root / "oracle_target.npy", np.zeros((0, 4), dtype=np.float32))
        write_npy(
            train_root / "oracle_target_present.npy", np.zeros((0,), dtype=np.uint8)
        )
        write_npy(
            train_root / "safety_residual_target.npy",
            np.zeros((0, 46), dtype=np.float32),
        )
        write_npy(
            train_root / "safety_residual_present.npy", np.zeros((0,), dtype=np.uint8)
        )
        write_npy(
            train_root / "safety_residual_mask.npy", np.zeros((0, 46), dtype=np.float32)
        )
        write_npy(
            train_root / "belief_fields_target.npy",
            np.zeros((0, 16, 34), dtype=np.float32),
        )
        write_npy(
            train_root / "belief_fields_present.npy", np.zeros((0,), dtype=np.uint8)
        )
        write_npy(
            train_root / "mixture_weight_target.npy", np.zeros((0, 4), dtype=np.float32)
        )
        write_npy(
            train_root / "mixture_weight_present.npy", np.zeros((0,), dtype=np.uint8)
        )

        write_npy(
            validation_root / "game_sample_offsets.npy",
            np.array([0, 1], dtype=np.int64),
        )
        write_npy(validation_root / "obs.npy", np.zeros((1, 192, 34), dtype=np.float32))
        write_npy(validation_root / "action.npy", np.array([1], dtype=np.uint8))
        legal_mask = np.zeros((1, 46), dtype=np.float32)
        legal_mask[0, 1] = 1.0
        write_npy(validation_root / "legal_mask.npy", legal_mask)
        write_npy(validation_root / "score_delta.npy", np.zeros((1,), dtype=np.int32))
        write_npy(
            validation_root / "value_target.npy", np.zeros((1,), dtype=np.float32)
        )
        write_npy(
            validation_root / "grp_target.npy", np.zeros((1, 24), dtype=np.float32)
        )
        write_npy(
            validation_root / "tenpai_target.npy", np.zeros((1, 3), dtype=np.float32)
        )
        write_npy(
            validation_root / "danger_target.npy",
            np.zeros((1, 3, 34), dtype=np.float32),
        )
        write_npy(
            validation_root / "danger_mask.npy", np.zeros((1, 3, 34), dtype=np.float32)
        )
        write_npy(
            validation_root / "opp_next_target.npy",
            np.zeros((1, 3, 34), dtype=np.float32),
        )
        write_npy(
            validation_root / "score_pdf_target.npy",
            np.zeros((1, 64), dtype=np.float32),
        )
        write_npy(
            validation_root / "score_cdf_target.npy",
            np.zeros((1, 64), dtype=np.float32),
        )
        write_npy(
            validation_root / "oracle_target.npy", np.zeros((1, 4), dtype=np.float32)
        )
        write_npy(
            validation_root / "oracle_target_present.npy",
            np.zeros((1,), dtype=np.uint8),
        )
        write_npy(
            validation_root / "safety_residual_target.npy",
            np.zeros((1, 46), dtype=np.float32),
        )
        write_npy(
            validation_root / "safety_residual_present.npy",
            np.zeros((1,), dtype=np.uint8),
        )
        write_npy(
            validation_root / "safety_residual_mask.npy",
            np.zeros((1, 46), dtype=np.float32),
        )
        write_npy(
            validation_root / "belief_fields_target.npy",
            np.zeros((1, 16, 34), dtype=np.float32),
        )
        write_npy(
            validation_root / "belief_fields_present.npy",
            np.zeros((1,), dtype=np.uint8),
        )
        write_npy(
            validation_root / "mixture_weight_target.npy",
            np.zeros((1, 4), dtype=np.float32),
        )
        write_npy(
            validation_root / "mixture_weight_present.npy",
            np.zeros((1,), dtype=np.uint8),
        )
        manifest["shards"][0]["hashes"]["files"] = shard_file_hashes(train_root)
        manifest["shards"][1]["hashes"]["files"] = shard_file_hashes(validation_root)
        (export_root / "manifest.json").write_text(json.dumps(manifest))

        config_path = tmp_root / "config.yaml"
        config_path.write_text(
            f"""
data:
  export_root: {export_root}
run:
  output_dir: {tmp_root / "out"}
  seed: 0
  num_epochs: 1
""".strip()
        )
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(ROOT) if not existing else f"{ROOT}:{existing}"
        result = subprocess.run(
            [
                *python_command(),
                "-m",
                "hydra_phase0_tpu.train",
                "--config",
                str(config_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("first train shard contains zero samples", result.stderr)

    def test_train_full_loop_with_microbatch_accumulation(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_accum_train_test"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        export_root, config_path = self._write_fixture_export(
            tmp_root, train_samples=2, validation_samples=1
        )
        config_path.write_text(
            f"""
data:
  export_root: {export_root}
run:
  output_dir: {tmp_root / "out"}
  seed: 0
  num_epochs: 1
  batch_size: 2
  microbatch_size: 1
  smoke_test: false
  validate_every_n_steps: 1
  checkpoint_every_n_steps: 1
""".strip()
        )
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(ROOT) if not existing else f"{ROOT}:{existing}"
        result = subprocess.run(
            [
                *python_command(),
                "-m",
                "hydra_phase0_tpu.train",
                "--config",
                str(config_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertIn("global_step", result.stdout)

    def test_train_resume_from_latest_checkpoint(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_resume_train_test"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        export_root, config_path = self._write_fixture_export(
            tmp_root, train_samples=2, validation_samples=1
        )

        first_output = tmp_root / "out_first"
        config_path.write_text(
            f"""
data:
  export_root: {export_root}
run:
  output_dir: {first_output}
  seed: 0
  num_epochs: 1
  smoke_test: false
  validate_every_n_steps: 1
  checkpoint_every_n_steps: 1
""".strip()
        )

        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(ROOT) if not existing else f"{ROOT}:{existing}"
        subprocess.run(
            [
                *python_command(),
                "-m",
                "hydra_phase0_tpu.train",
                "--config",
                str(config_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

        second_output = tmp_root / "out_second"
        resume_from = first_output / "bc_tpu_phase0" / "latest_orbax"
        config_path.write_text(
            f"""
data:
  export_root: {export_root}
run:
  output_dir: {second_output}
  seed: 0
  num_epochs: 2
  smoke_test: false
  validate_every_n_steps: 1
  checkpoint_every_n_steps: 1
  resume_from: {resume_from}
""".strip()
        )
        result = subprocess.run(
            [
                *python_command(),
                "-m",
                "hydra_phase0_tpu.train",
                "--config",
                str(config_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertIn("global_step", result.stdout)
        self.assertTrue(
            (
                second_output / "bc_tpu_phase0" / "latest_orbax" / "resume_state.json"
            ).exists()
        )
        resume_state = json.loads(
            (
                second_output / "bc_tpu_phase0" / "latest_orbax" / "resume_state.json"
            ).read_text()
        )
        self.assertIn("next_batch_index", resume_state)
        self.assertNotIn("next_shard_index", resume_state)

    def test_train_can_initialize_from_exported_weights(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_init_weights_test"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        export_root, config_path = self._write_fixture_export(
            tmp_root, train_samples=1, validation_samples=1
        )

        model = HydraModel()
        obs = jnp.zeros((1, 192, 34), dtype=jnp.float32)
        params = model.init(jax.random.PRNGKey(0), obs)["params"]
        weights_prefix = tmp_root / "weights_phase0_export"
        export_weights(params, weights_prefix)

        config_path.write_text(
            f"""
data:
  export_root: {export_root}
run:
  output_dir: {tmp_root / "out"}
  seed: 0
  num_epochs: 1
  smoke_test: true
  init_weights_from: {weights_prefix}
""".strip()
        )

        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(ROOT) if not existing else f"{ROOT}:{existing}"
        result = subprocess.run(
            [
                *python_command(),
                "-m",
                "hydra_phase0_tpu.train",
                "--config",
                str(config_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertIn("manifest_fingerprint", result.stdout)

    def test_train_rejects_mismatched_exported_weights(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_bad_init_weights_test"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        export_root, config_path = self._write_fixture_export(
            tmp_root, train_samples=1, validation_samples=1
        )

        model = HydraModel(
            num_blocks=1, hidden_channels=32, num_groups=4, se_bottleneck=8
        )
        obs = jnp.zeros((1, 192, 34), dtype=jnp.float32)
        params = model.init(jax.random.PRNGKey(0), obs)["params"]
        weights_prefix = tmp_root / "weights_phase0_export"
        export_weights(params, weights_prefix)

        config_path.write_text(
            f"""
data:
  export_root: {export_root}
run:
  output_dir: {tmp_root / "out"}
  seed: 0
  num_epochs: 1
  smoke_test: true
  init_weights_from: {weights_prefix}
""".strip()
        )

        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(ROOT) if not existing else f"{ROOT}:{existing}"
        result = subprocess.run(
            [
                *python_command(),
                "-m",
                "hydra_phase0_tpu.train",
                "--config",
                str(config_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("shape", result.stderr.lower())

    def test_train_rejects_required_backend_mismatch(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_backend_mismatch_test"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        export_root, config_path = self._write_fixture_export(
            tmp_root, train_samples=1, validation_samples=1
        )
        config_path.write_text(
            f"""
data:
  export_root: {export_root}
run:
  output_dir: {tmp_root / "out"}
  seed: 0
  num_epochs: 1
  required_backend: tpu
""".strip()
        )
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(ROOT) if not existing else f"{ROOT}:{existing}"
        result = subprocess.run(
            [
                *python_command(),
                "-m",
                "hydra_phase0_tpu.train",
                "--config",
                str(config_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required backend tpu not available", result.stderr)

    def test_train_rejects_resume_runtime_mismatch(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_resume_runtime_mismatch_test"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        export_root, config_path = self._write_fixture_export(
            tmp_root, train_samples=2, validation_samples=1
        )

        first_output = tmp_root / "out_first"
        config_path.write_text(
            f"""
data:
  export_root: {export_root}
run:
  output_dir: {first_output}
  seed: 0
  num_epochs: 1
  validate_every_n_steps: 1
  checkpoint_every_n_steps: 1
""".strip()
        )
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(ROOT) if not existing else f"{ROOT}:{existing}"
        subprocess.run(
            [
                *python_command(),
                "-m",
                "hydra_phase0_tpu.train",
                "--config",
                str(config_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

        second_output = tmp_root / "out_second"
        resume_from = first_output / "bc_tpu_phase0" / "latest_orbax"
        config_path.write_text(
            f"""
data:
  export_root: {export_root}
run:
  output_dir: {second_output}
  seed: 0
  num_epochs: 2
  batch_size: 1
  validate_every_n_steps: 1
  checkpoint_every_n_steps: 1
  resume_from: {resume_from}
""".strip()
        )
        result = subprocess.run(
            [
                *python_command(),
                "-m",
                "hydra_phase0_tpu.train",
                "--config",
                str(config_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("resume runtime mismatch", result.stderr)

    def test_train_rejects_nondivisible_data_parallel_microbatch(self) -> None:
        if jax.local_device_count() < 2:
            self.skipTest(
                "requires multiple local devices to hit data-parallel divisibility guard"
            )
        tmp_root = Path(__file__).resolve().parent / ".tmp_microbatch_divisibility_test"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        export_root, config_path = self._write_fixture_export(
            tmp_root,
            train_samples=jax.local_device_count(),
            validation_samples=1,
        )
        bad_microbatch = jax.local_device_count() + 1
        config_path.write_text(
            f"""
data:
  export_root: {export_root}
run:
  output_dir: {tmp_root / "out"}
  seed: 0
  num_epochs: 1
  batch_size: {jax.local_device_count()}
  microbatch_size: {bad_microbatch}
  validate_every_n_steps: 1
  checkpoint_every_n_steps: 1
""".strip()
        )
        env = dict(os.environ)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(ROOT) if not existing else f"{ROOT}:{existing}"
        result = subprocess.run(
            [
                *python_command(),
                "-m",
                "hydra_phase0_tpu.train",
                "--config",
                str(config_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("microbatch_size", result.stderr)
        self.assertIn("divisible by local device count", result.stderr)

    def test_restore_train_state_preserves_opt_state_and_step(self) -> None:
        from hydra_phase0_tpu.train_state import create_train_state, restore_train_state

        model = HydraModel()
        obs = jnp.zeros((1, 192, 34), dtype=jnp.float32)
        params = model.init(jax.random.PRNGKey(0), obs)["params"]
        state = create_train_state(
            model.apply,
            params,
            learning_rate=2.5e-4,
            min_learning_rate=1e-6,
            weight_decay=1e-5,
            grad_clip_norm=1.0,
            warmup_steps=1000,
        )
        restored = restore_train_state(
            model.apply,
            state.params,
            state.opt_state,
            7,
            learning_rate=2.5e-4,
            min_learning_rate=1e-6,
            weight_decay=1e-5,
            grad_clip_norm=1.0,
            warmup_steps=1000,
        )
        self.assertEqual(int(restored.step), 7)
        self.assertEqual(restored.opt_state, state.opt_state)

    def test_learning_rate_schedule_warms_up_to_peak(self) -> None:
        schedule = create_learning_rate_schedule(2.5e-4, 1e-6, 4)
        self.assertAlmostEqual(float(schedule(0)), 1e-6, places=10)
        self.assertGreater(float(schedule(1)), float(schedule(0)))
        self.assertAlmostEqual(float(schedule(4)), 2.5e-4, places=10)
        self.assertAlmostEqual(float(schedule(10)), 2.5e-4, places=10)

    def test_build_model_honors_compute_dtype(self) -> None:
        config = ExperimentConfig.model_validate(
            {
                "data": {"export_root": "/tmp/export"},
                "run": {"output_dir": "/tmp/out", "seed": 0, "num_epochs": 1},
                "model": {"compute_dtype": "bfloat16"},
            }
        )
        model = _build_model(config)
        self.assertEqual(model.dtype, jnp.bfloat16)

    def test_shard_batch_splits_first_axis_by_device(self) -> None:
        batch = {
            "obs": jnp.zeros((4, 192, 34), dtype=jnp.float32),
            "legal_mask": jnp.ones((4, 46), dtype=jnp.float32),
        }
        self.assertTrue(_can_shard_batch(batch, 2))
        sharded = _shard_batch(batch, 2)
        self.assertEqual(sharded["obs"].shape, (2, 2, 192, 34))
        self.assertEqual(sharded["legal_mask"].shape, (2, 2, 46))

    def test_runtime_info_reports_execution_mode(self) -> None:
        config = ExperimentConfig.model_validate(
            {
                "data": {"export_root": "/tmp/export"},
                "run": {"output_dir": "/tmp/out", "seed": 0, "num_epochs": 1},
            }
        )
        info = _runtime_info(config)
        self.assertIn(info["execution_mode"], {"single_device", "data_parallel"})
        self.assertGreaterEqual(int(info["local_device_count"]), 1)
        self.assertEqual(int(info["process_count"]), 1)

    def test_compiled_train_and_eval_steps_run(self) -> None:
        model = HydraModel()
        obs = jnp.zeros((1, 192, 34), dtype=jnp.float32)
        params = model.init(jax.random.PRNGKey(0), obs)["params"]
        state = create_train_state(
            model.apply,
            params,
            learning_rate=2.5e-4,
            min_learning_rate=1e-6,
            weight_decay=1e-5,
            grad_clip_norm=1.0,
            warmup_steps=1000,
        )
        batch = {
            "obs": obs,
            "policy_target": jax.nn.one_hot(jnp.array([1]), 46),
            "legal_mask": jnp.ones((1, 46), dtype=jnp.float32),
            "value_target": jnp.zeros((1,), dtype=jnp.float32),
            "grp_target": jax.nn.one_hot(jnp.array([0]), 24, dtype=jnp.float32),
            "tenpai_target": jnp.zeros((1, 3), dtype=jnp.float32),
            "danger_target": jnp.zeros((1, 3, 34), dtype=jnp.float32),
            "danger_mask": jnp.zeros((1, 3, 34), dtype=jnp.float32),
            "opp_next_target": jnp.zeros((1, 3, 34), dtype=jnp.float32),
            "score_pdf_target": jax.nn.one_hot(jnp.array([0]), 64, dtype=jnp.float32),
            "score_cdf_target": jnp.zeros((1, 64), dtype=jnp.float32),
            "safety_residual_target": jnp.zeros((1, 46), dtype=jnp.float32),
            "safety_residual_mask": jnp.zeros((1, 46), dtype=jnp.float32),
            "safety_residual_present": jnp.zeros((1,), dtype=jnp.float32),
        }

        next_state, loss, breakdown = train_step(state, batch, model)
        self.assertTrue(np.isfinite(float(loss)))
        self.assertEqual(int(next_state.step), 1)
        self.assertTrue(np.isfinite(float(breakdown.total)))

        logical_state, logical_loss, logical_breakdown = train_logical_batch(
            state, batch, model, 1
        )
        self.assertTrue(np.isfinite(float(logical_loss)))
        self.assertEqual(int(logical_state.step), 1)
        self.assertTrue(np.isfinite(float(logical_breakdown.total)))

        eval_breakdown = eval_step(state.params, batch, model)
        self.assertTrue(np.isfinite(float(eval_breakdown.total)))

    def test_train_logical_batch_aggregates_breakdown_across_microbatches(self) -> None:
        model = HydraModel()
        obs = jnp.zeros((2, 192, 34), dtype=jnp.float32)
        params = model.init(jax.random.PRNGKey(0), obs[:1])["params"]
        state = create_train_state(
            model.apply,
            params,
            learning_rate=2.5e-4,
            min_learning_rate=1e-6,
            weight_decay=1e-5,
            grad_clip_norm=1.0,
            warmup_steps=1000,
        )
        batch = {
            "obs": obs,
            "policy_target": jnp.stack(
                [
                    jax.nn.one_hot(jnp.array(0), 46, dtype=jnp.float32),
                    jax.nn.one_hot(jnp.array(1), 46, dtype=jnp.float32),
                ]
            ),
            "legal_mask": jnp.ones((2, 46), dtype=jnp.float32),
            "value_target": jnp.array([0.0, 1.0], dtype=jnp.float32),
            "grp_target": jnp.stack(
                [
                    jax.nn.one_hot(jnp.array(0), 24, dtype=jnp.float32),
                    jax.nn.one_hot(jnp.array(1), 24, dtype=jnp.float32),
                ]
            ),
            "tenpai_target": jnp.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32
            ),
            "danger_target": jnp.zeros((2, 3, 34), dtype=jnp.float32),
            "danger_mask": jnp.zeros((2, 3, 34), dtype=jnp.float32),
            "opp_next_target": jnp.zeros((2, 3, 34), dtype=jnp.float32),
            "score_pdf_target": jnp.stack(
                [
                    jax.nn.one_hot(jnp.array(0), 64, dtype=jnp.float32),
                    jax.nn.one_hot(jnp.array(1), 64, dtype=jnp.float32),
                ]
            ),
            "score_cdf_target": jnp.array(
                [
                    [1.0] + [0.0] * 63,
                    [1.0, 1.0] + [0.0] * 62,
                ],
                dtype=jnp.float32,
            ),
            "safety_residual_target": jnp.array(
                [
                    jnp.zeros((46,), dtype=jnp.float32),
                    jnp.ones((46,), dtype=jnp.float32),
                ]
            ),
            "safety_residual_mask": jnp.ones((2, 46), dtype=jnp.float32),
            "safety_residual_present": jnp.ones((2,), dtype=jnp.float32),
        }

        _, logical_loss, logical_breakdown = train_logical_batch(state, batch, model, 1)
        breakdown0 = total_loss(
            model.apply({"params": state.params}, batch["obs"][:1]),
            {key: value[:1] for key, value in batch.items()},
        )
        breakdown1 = total_loss(
            model.apply({"params": state.params}, batch["obs"][1:2]),
            {key: value[1:2] for key, value in batch.items()},
        )
        expected_policy = (float(breakdown0.policy) + float(breakdown1.policy)) / 2.0
        expected_total = (float(breakdown0.total) + float(breakdown1.total)) / 2.0

        self.assertAlmostEqual(float(logical_loss), expected_total, places=6)
        self.assertAlmostEqual(
            float(logical_breakdown.policy), expected_policy, places=6
        )
        self.assertAlmostEqual(float(logical_breakdown.total), expected_total, places=6)

    def test_data_parallel_train_logical_batch_runs(self) -> None:
        model = HydraModel()
        device_count = jax.local_device_count()
        obs = jnp.zeros((device_count, 192, 34), dtype=jnp.float32)
        params = model.init(jax.random.PRNGKey(0), obs[:1])["params"]
        state = create_train_state(
            model.apply,
            params,
            learning_rate=2.5e-4,
            min_learning_rate=1e-6,
            weight_decay=1e-5,
            grad_clip_norm=1.0,
            warmup_steps=1000,
        )
        replicated_state = jax_utils.replicate(state)
        batch = {
            "obs": jnp.zeros((device_count, 1, 192, 34), dtype=jnp.float32),
            "policy_target": jax.nn.one_hot(
                jnp.ones((device_count, 1), dtype=jnp.int32), 46
            ),
            "legal_mask": jnp.ones((device_count, 1, 46), dtype=jnp.float32),
            "value_target": jnp.zeros((device_count, 1), dtype=jnp.float32),
            "grp_target": jax.nn.one_hot(
                jnp.zeros((device_count, 1), dtype=jnp.int32), 24, dtype=jnp.float32
            ),
            "tenpai_target": jnp.zeros((device_count, 1, 3), dtype=jnp.float32),
            "danger_target": jnp.zeros((device_count, 1, 3, 34), dtype=jnp.float32),
            "danger_mask": jnp.zeros((device_count, 1, 3, 34), dtype=jnp.float32),
            "opp_next_target": jnp.zeros((device_count, 1, 3, 34), dtype=jnp.float32),
            "score_pdf_target": jax.nn.one_hot(
                jnp.zeros((device_count, 1), dtype=jnp.int32), 64, dtype=jnp.float32
            ),
            "score_cdf_target": jnp.zeros((device_count, 1, 64), dtype=jnp.float32),
            "safety_residual_target": jnp.zeros(
                (device_count, 1, 46), dtype=jnp.float32
            ),
            "safety_residual_mask": jnp.zeros((device_count, 1, 46), dtype=jnp.float32),
            "safety_residual_present": jnp.zeros((device_count, 1), dtype=jnp.float32),
        }

        next_state, loss, breakdown = train_logical_batch_data_parallel(
            replicated_state, batch, model, 1
        )
        unreplicated = jax_utils.unreplicate(next_state)
        self.assertEqual(int(unreplicated.step), 1)
        self.assertTrue(np.isfinite(float(loss)))
        self.assertTrue(np.isfinite(float(jax_utils.unreplicate(breakdown).total)))

    def test_safety_residual_loss_contributes_when_present(self) -> None:
        output = HydraOutput(
            policy_logits=jnp.zeros((1, 46), dtype=jnp.float32),
            value=jnp.zeros((1, 1), dtype=jnp.float32),
            score_pdf=jnp.zeros((1, 64), dtype=jnp.float32),
            score_cdf=jnp.zeros((1, 64), dtype=jnp.float32),
            opp_tenpai=jnp.zeros((1, 3), dtype=jnp.float32),
            grp=jnp.zeros((1, 24), dtype=jnp.float32),
            opp_next_discard=jnp.zeros((1, 3, 34), dtype=jnp.float32),
            danger=jnp.zeros((1, 3, 34), dtype=jnp.float32),
            oracle_critic=jnp.zeros((1, 4), dtype=jnp.float32),
            belief_fields=jnp.zeros((1, 16, 34), dtype=jnp.float32),
            mixture_weight_logits=jnp.zeros((1, 4), dtype=jnp.float32),
            opponent_hand_type=jnp.zeros((1, 24), dtype=jnp.float32),
            delta_q=jnp.zeros((1, 46), dtype=jnp.float32),
            safety_residual=jnp.ones((1, 46), dtype=jnp.float32),
        )
        batch = {
            "policy_target": jax.nn.one_hot(jnp.array([0]), 46, dtype=jnp.float32),
            "legal_mask": jnp.ones((1, 46), dtype=jnp.float32),
            "value_target": jnp.zeros((1,), dtype=jnp.float32),
            "grp_target": jax.nn.one_hot(jnp.array([0]), 24, dtype=jnp.float32),
            "tenpai_target": jnp.zeros((1, 3), dtype=jnp.float32),
            "danger_target": jnp.zeros((1, 3, 34), dtype=jnp.float32),
            "danger_mask": jnp.zeros((1, 3, 34), dtype=jnp.float32),
            "opp_next_target": jnp.zeros((1, 3, 34), dtype=jnp.float32),
            "score_pdf_target": jax.nn.one_hot(jnp.array([0]), 64, dtype=jnp.float32),
            "score_cdf_target": jnp.zeros((1, 64), dtype=jnp.float32),
            "safety_residual_target": jnp.zeros((1, 46), dtype=jnp.float32),
            "safety_residual_mask": jnp.ones((1, 46), dtype=jnp.float32),
            "safety_residual_present": jnp.ones((1,), dtype=jnp.float32),
        }
        breakdown = total_loss(output, batch)
        self.assertGreater(float(breakdown.safety_residual), 0.0)
        self.assertGreater(float(breakdown.total), float(breakdown.policy))


if __name__ == "__main__":
    unittest.main()
