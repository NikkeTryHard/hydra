from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hydra_phase0_tpu.dataset import (
    ExportDataset,
    iter_train_epoch_batches,
    load_shard_arrays,
)


def write_npy(path: Path, array: np.ndarray) -> None:
    np.save(path, array)


def shard_file_hashes(root: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.iterdir())
        if path.is_file()
    }


class DatasetTests(unittest.TestCase):
    def test_export_dataset_reads_manifest_and_shard(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_dataset_test"
        export_root = tmp_root / "bc_phase0_export"
        shard_root = export_root / "train" / "shard-000000"
        shard_root.mkdir(parents=True, exist_ok=True)

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
                    "sample_count": 2,
                    "hashes": {"files": {}},
                }
            ],
        }

        write_npy(
            shard_root / "game_sample_offsets.npy", np.array([0, 2], dtype=np.int64)
        )
        write_npy(shard_root / "obs.npy", np.zeros((2, 192, 34), dtype=np.float32))
        write_npy(shard_root / "action.npy", np.array([1, 2], dtype=np.uint8))
        legal_mask = np.zeros((2, 46), dtype=np.float32)
        legal_mask[0, 1] = 1.0
        legal_mask[1, 2] = 1.0
        write_npy(shard_root / "legal_mask.npy", legal_mask)
        write_npy(shard_root / "score_delta.npy", np.array([100, -100], dtype=np.int32))
        write_npy(shard_root / "value_target.npy", np.zeros((2,), dtype=np.float32))
        write_npy(shard_root / "grp_target.npy", np.zeros((2, 24), dtype=np.float32))
        write_npy(shard_root / "tenpai_target.npy", np.zeros((2, 3), dtype=np.float32))
        write_npy(
            shard_root / "danger_target.npy", np.zeros((2, 3, 34), dtype=np.float32)
        )
        write_npy(
            shard_root / "danger_mask.npy", np.zeros((2, 3, 34), dtype=np.float32)
        )
        write_npy(
            shard_root / "opp_next_target.npy", np.zeros((2, 3, 34), dtype=np.float32)
        )
        write_npy(
            shard_root / "score_pdf_target.npy", np.zeros((2, 64), dtype=np.float32)
        )
        write_npy(
            shard_root / "score_cdf_target.npy", np.zeros((2, 64), dtype=np.float32)
        )
        write_npy(shard_root / "oracle_target.npy", np.zeros((2, 4), dtype=np.float32))
        write_npy(
            shard_root / "oracle_target_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        write_npy(
            shard_root / "safety_residual_target.npy",
            np.zeros((2, 46), dtype=np.float32),
        )
        write_npy(
            shard_root / "safety_residual_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        write_npy(
            shard_root / "safety_residual_mask.npy", np.zeros((2, 46), dtype=np.float32)
        )
        write_npy(
            shard_root / "belief_fields_target.npy",
            np.zeros((2, 16, 34), dtype=np.float32),
        )
        write_npy(
            shard_root / "belief_fields_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        write_npy(
            shard_root / "mixture_weight_target.npy", np.zeros((2, 4), dtype=np.float32)
        )
        write_npy(
            shard_root / "mixture_weight_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        manifest["shards"][0]["hashes"]["files"] = shard_file_hashes(shard_root)
        (export_root / "manifest.json").write_text(json.dumps(manifest))

        dataset = ExportDataset(export_root)
        shards = dataset.shards("train")
        self.assertEqual(dataset.manifest_fingerprint, "abc123")
        self.assertEqual(len(shards), 1)
        arrays = load_shard_arrays(shards[0])
        self.assertEqual(arrays["obs"].shape, (2, 192, 34))
        self.assertEqual(arrays["action"].dtype, np.uint8)

    def test_manifest_contract_rejects_wrong_schema(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_dataset_bad_schema"
        export_root = tmp_root / "bc_phase0_export"
        export_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "manifest_fingerprint": "abc123",
            "schema_version": "wrong",
            "export_semantics": "hydra_bc_phase0_v1",
            "encoder_contract": "192x34",
            "action_space": 46,
            "shards": [],
        }
        (export_root / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            ExportDataset(export_root)

    def test_manifest_contract_requires_schema_version(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_dataset_missing_schema"
        export_root = tmp_root / "bc_phase0_export"
        export_root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "manifest_fingerprint": "abc123",
            "export_semantics": "hydra_bc_phase0_v1",
            "encoder_contract": "192x34",
            "action_space": 46,
            "shards": [],
        }
        (export_root / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(
            ValueError, "manifest missing required field schema_version"
        ):
            ExportDataset(export_root)

    def test_load_shard_arrays_rejects_tampered_file_hash(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_dataset_bad_hash"
        export_root = tmp_root / "bc_phase0_export"
        shard_root = export_root / "train" / "shard-000000"
        shard_root.mkdir(parents=True, exist_ok=True)

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
                    "sample_count": 2,
                    "hashes": {"files": {}},
                }
            ],
        }
        write_npy(
            shard_root / "game_sample_offsets.npy", np.array([0, 2], dtype=np.int64)
        )
        write_npy(shard_root / "obs.npy", np.zeros((2, 192, 34), dtype=np.float32))
        write_npy(shard_root / "action.npy", np.array([1, 2], dtype=np.uint8))
        legal_mask = np.zeros((2, 46), dtype=np.float32)
        legal_mask[0, 1] = 1.0
        legal_mask[1, 2] = 1.0
        write_npy(shard_root / "legal_mask.npy", legal_mask)
        write_npy(shard_root / "score_delta.npy", np.array([100, -100], dtype=np.int32))
        write_npy(shard_root / "value_target.npy", np.zeros((2,), dtype=np.float32))
        write_npy(shard_root / "grp_target.npy", np.zeros((2, 24), dtype=np.float32))
        write_npy(shard_root / "tenpai_target.npy", np.zeros((2, 3), dtype=np.float32))
        write_npy(
            shard_root / "danger_target.npy", np.zeros((2, 3, 34), dtype=np.float32)
        )
        write_npy(
            shard_root / "danger_mask.npy", np.zeros((2, 3, 34), dtype=np.float32)
        )
        write_npy(
            shard_root / "opp_next_target.npy", np.zeros((2, 3, 34), dtype=np.float32)
        )
        write_npy(
            shard_root / "score_pdf_target.npy", np.zeros((2, 64), dtype=np.float32)
        )
        write_npy(
            shard_root / "score_cdf_target.npy", np.zeros((2, 64), dtype=np.float32)
        )
        write_npy(shard_root / "oracle_target.npy", np.zeros((2, 4), dtype=np.float32))
        write_npy(
            shard_root / "oracle_target_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        write_npy(
            shard_root / "safety_residual_target.npy",
            np.zeros((2, 46), dtype=np.float32),
        )
        write_npy(
            shard_root / "safety_residual_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        write_npy(
            shard_root / "safety_residual_mask.npy", np.zeros((2, 46), dtype=np.float32)
        )
        write_npy(
            shard_root / "belief_fields_target.npy",
            np.zeros((2, 16, 34), dtype=np.float32),
        )
        write_npy(
            shard_root / "belief_fields_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        write_npy(
            shard_root / "mixture_weight_target.npy", np.zeros((2, 4), dtype=np.float32)
        )
        write_npy(
            shard_root / "mixture_weight_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        manifest["shards"][0]["hashes"]["files"] = shard_file_hashes(shard_root)
        (export_root / "manifest.json").write_text(json.dumps(manifest))

        write_npy(shard_root / "obs.npy", np.ones((2, 192, 34), dtype=np.float32))

        dataset = ExportDataset(export_root)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_shard_arrays(dataset.shards("train")[0])

    def test_shards_require_nonempty_file_hashes(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_dataset_missing_hashes"
        export_root = tmp_root / "bc_phase0_export"
        export_root.mkdir(parents=True, exist_ok=True)
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
                    "sample_count": 2,
                    "hashes": {"files": {}},
                }
            ],
        }
        (export_root / "manifest.json").write_text(json.dumps(manifest))
        dataset = ExportDataset(export_root)
        with self.assertRaisesRegex(ValueError, "missing required file hashes"):
            dataset.shards("train")

    def test_iter_train_epoch_batches_is_seed_stable(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_dataset_epoch_iter"
        export_root = tmp_root / "bc_phase0_export"
        train_root = export_root / "train" / "shard-000000"
        train_root.mkdir(parents=True, exist_ok=True)
        (export_root / "validation").mkdir(parents=True, exist_ok=True)
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
                    "game_count": 2,
                    "sample_count": 2,
                    "hashes": {"files": {}},
                }
            ],
        }
        (export_root / "manifest.json").write_text(json.dumps(manifest))
        write_npy(
            train_root / "game_sample_offsets.npy", np.array([0, 1, 2], dtype=np.int64)
        )
        obs = np.zeros((2, 192, 34), dtype=np.float32)
        obs[0, 40, 0] = 1.0
        obs[1, 41, 1] = 1.0
        write_npy(train_root / "obs.npy", obs)
        write_npy(train_root / "action.npy", np.array([1, 2], dtype=np.uint8))
        write_npy(train_root / "legal_mask.npy", np.ones((2, 46), dtype=np.float32))
        write_npy(train_root / "score_delta.npy", np.array([100, 100], dtype=np.int32))
        write_npy(
            train_root / "value_target.npy", np.array([0.001, 0.001], dtype=np.float32)
        )
        write_npy(train_root / "grp_target.npy", np.zeros((2, 24), dtype=np.float32))
        write_npy(train_root / "tenpai_target.npy", np.zeros((2, 3), dtype=np.float32))
        write_npy(
            train_root / "danger_target.npy", np.zeros((2, 3, 34), dtype=np.float32)
        )
        write_npy(
            train_root / "danger_mask.npy", np.zeros((2, 3, 34), dtype=np.float32)
        )
        write_npy(
            train_root / "opp_next_target.npy", np.zeros((2, 3, 34), dtype=np.float32)
        )
        write_npy(
            train_root / "score_pdf_target.npy", np.zeros((2, 64), dtype=np.float32)
        )
        write_npy(
            train_root / "score_cdf_target.npy", np.zeros((2, 64), dtype=np.float32)
        )
        write_npy(train_root / "oracle_target.npy", np.zeros((2, 4), dtype=np.float32))
        write_npy(
            train_root / "oracle_target_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        write_npy(
            train_root / "safety_residual_target.npy",
            np.zeros((2, 46), dtype=np.float32),
        )
        write_npy(
            train_root / "safety_residual_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        write_npy(
            train_root / "safety_residual_mask.npy", np.zeros((2, 46), dtype=np.float32)
        )
        write_npy(
            train_root / "belief_fields_target.npy",
            np.zeros((2, 16, 34), dtype=np.float32),
        )
        write_npy(
            train_root / "belief_fields_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        write_npy(
            train_root / "mixture_weight_target.npy", np.zeros((2, 4), dtype=np.float32)
        )
        write_npy(
            train_root / "mixture_weight_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        manifest["shards"][0]["hashes"]["files"] = shard_file_hashes(train_root)
        (export_root / "manifest.json").write_text(json.dumps(manifest))
        dataset = ExportDataset(export_root)
        seq1 = [
            batch["action"].tolist()
            for batch in iter_train_epoch_batches(
                dataset,
                seed=7,
                epoch=0,
                buffer_games=2,
                buffer_samples=2,
                logical_batch_size=1,
            )
        ]
        seq2 = [
            batch["action"].tolist()
            for batch in iter_train_epoch_batches(
                dataset,
                seed=7,
                epoch=0,
                buffer_games=2,
                buffer_samples=2,
                logical_batch_size=1,
            )
        ]
        self.assertEqual(seq1, seq2)

    def test_load_shard_arrays_rejects_bad_optional_target_shape(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_dataset_bad_shape"
        export_root = tmp_root / "bc_phase0_export"
        shard_root = export_root / "train" / "shard-000000"
        shard_root.mkdir(parents=True, exist_ok=True)
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
                    "sample_count": 2,
                    "hashes": {"files": {}},
                }
            ],
        }
        write_npy(
            shard_root / "game_sample_offsets.npy", np.array([0, 2], dtype=np.int64)
        )
        write_npy(shard_root / "obs.npy", np.zeros((2, 192, 34), dtype=np.float32))
        write_npy(shard_root / "action.npy", np.array([1, 2], dtype=np.uint8))
        write_npy(shard_root / "legal_mask.npy", np.ones((2, 46), dtype=np.float32))
        write_npy(shard_root / "score_delta.npy", np.zeros((2,), dtype=np.int32))
        write_npy(shard_root / "value_target.npy", np.zeros((2,), dtype=np.float32))
        write_npy(shard_root / "grp_target.npy", np.zeros((2, 24), dtype=np.float32))
        write_npy(shard_root / "tenpai_target.npy", np.zeros((2, 3), dtype=np.float32))
        write_npy(shard_root / "danger_target.npy", np.zeros((2, 34), dtype=np.float32))
        write_npy(
            shard_root / "danger_mask.npy", np.zeros((2, 3, 34), dtype=np.float32)
        )
        write_npy(
            shard_root / "opp_next_target.npy", np.zeros((2, 3, 34), dtype=np.float32)
        )
        write_npy(
            shard_root / "score_pdf_target.npy", np.zeros((2, 64), dtype=np.float32)
        )
        write_npy(
            shard_root / "score_cdf_target.npy", np.zeros((2, 64), dtype=np.float32)
        )
        write_npy(shard_root / "oracle_target.npy", np.zeros((2, 4), dtype=np.float32))
        write_npy(
            shard_root / "oracle_target_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        write_npy(
            shard_root / "safety_residual_target.npy",
            np.zeros((2, 46), dtype=np.float32),
        )
        write_npy(
            shard_root / "safety_residual_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        write_npy(
            shard_root / "safety_residual_mask.npy", np.zeros((2, 46), dtype=np.float32)
        )
        write_npy(
            shard_root / "belief_fields_target.npy",
            np.zeros((2, 16, 34), dtype=np.float32),
        )
        write_npy(
            shard_root / "belief_fields_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        write_npy(
            shard_root / "mixture_weight_target.npy", np.zeros((2, 4), dtype=np.float32)
        )
        write_npy(
            shard_root / "mixture_weight_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        manifest["shards"][0]["hashes"]["files"] = shard_file_hashes(shard_root)
        (export_root / "manifest.json").write_text(json.dumps(manifest))

        dataset = ExportDataset(export_root)
        with self.assertRaisesRegex(ValueError, "danger_target shape mismatch"):
            load_shard_arrays(dataset.shards("train")[0])

    def test_load_shard_arrays_rejects_bad_offsets(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_dataset_bad_offsets"
        export_root = tmp_root / "bc_phase0_export"
        shard_root = export_root / "train" / "shard-000000"
        shard_root.mkdir(parents=True, exist_ok=True)
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
                    "sample_count": 2,
                    "hashes": {"files": {}},
                }
            ],
        }
        write_npy(
            shard_root / "game_sample_offsets.npy", np.array([1, 2], dtype=np.int64)
        )
        write_npy(shard_root / "obs.npy", np.zeros((2, 192, 34), dtype=np.float32))
        write_npy(shard_root / "action.npy", np.array([1, 2], dtype=np.uint8))
        write_npy(shard_root / "legal_mask.npy", np.ones((2, 46), dtype=np.float32))
        write_npy(shard_root / "score_delta.npy", np.zeros((2,), dtype=np.int32))
        write_npy(shard_root / "value_target.npy", np.zeros((2,), dtype=np.float32))
        write_npy(shard_root / "grp_target.npy", np.zeros((2, 24), dtype=np.float32))
        write_npy(shard_root / "tenpai_target.npy", np.zeros((2, 3), dtype=np.float32))
        write_npy(
            shard_root / "danger_target.npy", np.zeros((2, 3, 34), dtype=np.float32)
        )
        write_npy(
            shard_root / "danger_mask.npy", np.zeros((2, 3, 34), dtype=np.float32)
        )
        write_npy(
            shard_root / "opp_next_target.npy", np.zeros((2, 3, 34), dtype=np.float32)
        )
        write_npy(
            shard_root / "score_pdf_target.npy", np.zeros((2, 64), dtype=np.float32)
        )
        write_npy(
            shard_root / "score_cdf_target.npy", np.zeros((2, 64), dtype=np.float32)
        )
        write_npy(shard_root / "oracle_target.npy", np.zeros((2, 4), dtype=np.float32))
        write_npy(
            shard_root / "oracle_target_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        write_npy(
            shard_root / "safety_residual_target.npy",
            np.zeros((2, 46), dtype=np.float32),
        )
        write_npy(
            shard_root / "safety_residual_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        write_npy(
            shard_root / "safety_residual_mask.npy", np.zeros((2, 46), dtype=np.float32)
        )
        write_npy(
            shard_root / "belief_fields_target.npy",
            np.zeros((2, 16, 34), dtype=np.float32),
        )
        write_npy(
            shard_root / "belief_fields_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        write_npy(
            shard_root / "mixture_weight_target.npy", np.zeros((2, 4), dtype=np.float32)
        )
        write_npy(
            shard_root / "mixture_weight_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        manifest["shards"][0]["hashes"]["files"] = shard_file_hashes(shard_root)
        (export_root / "manifest.json").write_text(json.dumps(manifest))

        dataset = ExportDataset(export_root)
        with self.assertRaisesRegex(ValueError, "game_sample_offsets must start at 0"):
            load_shard_arrays(dataset.shards("train")[0])

    def test_load_shard_arrays_rejects_illegal_action(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_dataset_bad_action"
        export_root = tmp_root / "bc_phase0_export"
        shard_root = export_root / "train" / "shard-000000"
        shard_root.mkdir(parents=True, exist_ok=True)
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
                    "sample_count": 2,
                    "hashes": {"files": {}},
                }
            ],
        }
        write_npy(
            shard_root / "game_sample_offsets.npy", np.array([0, 2], dtype=np.int64)
        )
        write_npy(shard_root / "obs.npy", np.zeros((2, 192, 34), dtype=np.float32))
        write_npy(shard_root / "action.npy", np.array([1, 2], dtype=np.uint8))
        legal_mask = np.zeros((2, 46), dtype=np.float32)
        legal_mask[:, 0] = 1.0
        write_npy(shard_root / "legal_mask.npy", legal_mask)
        write_npy(shard_root / "score_delta.npy", np.zeros((2,), dtype=np.int32))
        write_npy(shard_root / "value_target.npy", np.zeros((2,), dtype=np.float32))
        write_npy(shard_root / "grp_target.npy", np.zeros((2, 24), dtype=np.float32))
        write_npy(shard_root / "tenpai_target.npy", np.zeros((2, 3), dtype=np.float32))
        write_npy(
            shard_root / "danger_target.npy", np.zeros((2, 3, 34), dtype=np.float32)
        )
        write_npy(
            shard_root / "danger_mask.npy", np.zeros((2, 3, 34), dtype=np.float32)
        )
        write_npy(
            shard_root / "opp_next_target.npy", np.zeros((2, 3, 34), dtype=np.float32)
        )
        write_npy(
            shard_root / "score_pdf_target.npy", np.zeros((2, 64), dtype=np.float32)
        )
        write_npy(
            shard_root / "score_cdf_target.npy", np.zeros((2, 64), dtype=np.float32)
        )
        write_npy(shard_root / "oracle_target.npy", np.zeros((2, 4), dtype=np.float32))
        write_npy(
            shard_root / "oracle_target_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        write_npy(
            shard_root / "safety_residual_target.npy",
            np.zeros((2, 46), dtype=np.float32),
        )
        write_npy(
            shard_root / "safety_residual_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        write_npy(
            shard_root / "safety_residual_mask.npy", np.zeros((2, 46), dtype=np.float32)
        )
        write_npy(
            shard_root / "belief_fields_target.npy",
            np.zeros((2, 16, 34), dtype=np.float32),
        )
        write_npy(
            shard_root / "belief_fields_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        write_npy(
            shard_root / "mixture_weight_target.npy", np.zeros((2, 4), dtype=np.float32)
        )
        write_npy(
            shard_root / "mixture_weight_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        manifest["shards"][0]["hashes"]["files"] = shard_file_hashes(shard_root)
        (export_root / "manifest.json").write_text(json.dumps(manifest))

        dataset = ExportDataset(export_root)
        with self.assertRaisesRegex(
            ValueError, "action must be legal under legal_mask"
        ):
            load_shard_arrays(dataset.shards("train")[0])

    def test_load_shard_arrays_rejects_nonzero_absent_optional_targets(self) -> None:
        tmp_root = (
            Path(__file__).resolve().parent / ".tmp_dataset_bad_optional_presence"
        )
        export_root = tmp_root / "bc_phase0_export"
        shard_root = export_root / "train" / "shard-000000"
        shard_root.mkdir(parents=True, exist_ok=True)
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
                    "sample_count": 2,
                    "hashes": {"files": {}},
                }
            ],
        }
        write_npy(
            shard_root / "game_sample_offsets.npy", np.array([0, 2], dtype=np.int64)
        )
        write_npy(shard_root / "obs.npy", np.zeros((2, 192, 34), dtype=np.float32))
        write_npy(shard_root / "action.npy", np.array([1, 2], dtype=np.uint8))
        legal_mask = np.zeros((2, 46), dtype=np.float32)
        legal_mask[0, 1] = 1.0
        legal_mask[1, 2] = 1.0
        write_npy(shard_root / "legal_mask.npy", legal_mask)
        write_npy(shard_root / "score_delta.npy", np.zeros((2,), dtype=np.int32))
        write_npy(shard_root / "value_target.npy", np.zeros((2,), dtype=np.float32))
        write_npy(shard_root / "grp_target.npy", np.zeros((2, 24), dtype=np.float32))
        write_npy(shard_root / "tenpai_target.npy", np.zeros((2, 3), dtype=np.float32))
        write_npy(
            shard_root / "danger_target.npy", np.zeros((2, 3, 34), dtype=np.float32)
        )
        write_npy(
            shard_root / "danger_mask.npy", np.zeros((2, 3, 34), dtype=np.float32)
        )
        write_npy(
            shard_root / "opp_next_target.npy", np.zeros((2, 3, 34), dtype=np.float32)
        )
        write_npy(
            shard_root / "score_pdf_target.npy", np.zeros((2, 64), dtype=np.float32)
        )
        write_npy(
            shard_root / "score_cdf_target.npy", np.zeros((2, 64), dtype=np.float32)
        )
        oracle_target = np.zeros((2, 4), dtype=np.float32)
        oracle_target[1, 0] = 1.0
        write_npy(shard_root / "oracle_target.npy", oracle_target)
        write_npy(
            shard_root / "oracle_target_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        write_npy(
            shard_root / "safety_residual_target.npy",
            np.zeros((2, 46), dtype=np.float32),
        )
        write_npy(
            shard_root / "safety_residual_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        write_npy(
            shard_root / "safety_residual_mask.npy", np.zeros((2, 46), dtype=np.float32)
        )
        write_npy(
            shard_root / "belief_fields_target.npy",
            np.zeros((2, 16, 34), dtype=np.float32),
        )
        write_npy(
            shard_root / "belief_fields_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        write_npy(
            shard_root / "mixture_weight_target.npy", np.zeros((2, 4), dtype=np.float32)
        )
        write_npy(
            shard_root / "mixture_weight_present.npy", np.zeros((2,), dtype=np.uint8)
        )
        manifest["shards"][0]["hashes"]["files"] = shard_file_hashes(shard_root)
        (export_root / "manifest.json").write_text(json.dumps(manifest))

        dataset = ExportDataset(export_root)
        with self.assertRaisesRegex(
            ValueError, "oracle_target must be zero-filled when absent"
        ):
            load_shard_arrays(dataset.shards("train")[0])


if __name__ == "__main__":
    unittest.main()
