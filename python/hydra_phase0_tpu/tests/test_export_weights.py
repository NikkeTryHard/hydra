from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import unittest

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hydra_phase0_tpu.export_weights import export_weights, load_exported_weights
from hydra_phase0_tpu.model import HydraModel


class ExportWeightsTests(unittest.TestCase):
    def test_export_writes_npz_and_metadata(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_export_weights"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True, exist_ok=True)
        params = {"dense": {"kernel": jnp.ones((2, 3), dtype=jnp.float32)}}
        npz_path, json_path = export_weights(params, tmp_root / "weights_phase0_export")
        self.assertTrue(npz_path.exists())
        self.assertTrue(json_path.exists())
        with np.load(npz_path) as archive:
            self.assertIn("dense__kernel", archive.files)
        metadata = json.loads(json_path.read_text())
        self.assertEqual(metadata["tensors"]["dense/kernel"]["shape"], [2, 3])

    def test_export_then_load_roundtrip_for_model_params(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_export_roundtrip"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True, exist_ok=True)
        model = HydraModel(
            num_blocks=1, hidden_channels=32, num_groups=4, se_bottleneck=8
        )
        obs = jnp.zeros((1, 192, 34), dtype=jnp.float32)
        params = model.init(jax.random.PRNGKey(0), obs)["params"]
        npz_path, json_path = export_weights(params, tmp_root / "weights_phase0_export")
        restored = load_exported_weights(json_path, npz_path)
        self.assertEqual(restored.keys(), params.keys())

    def test_load_exported_weights_rejects_dtype_mismatch(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_export_bad_dtype"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True, exist_ok=True)
        prefix = tmp_root / "weights_phase0_export"
        archive_arrays = {"dense__kernel": np.ones((2, 3), dtype=np.float64)}
        np.savez(prefix.with_suffix(".npz"), **archive_arrays)
        prefix.with_suffix(".json").write_text(
            json.dumps(
                {
                    "schema_version": "hydra_phase0_weight_export_v1",
                    "tensors": {
                        "dense/kernel": {
                            "archive_key": "dense__kernel",
                            "shape": [2, 3],
                            "dtype": "float32",
                        }
                    },
                }
            )
        )

        with self.assertRaisesRegex(ValueError, "dtype mismatch"):
            load_exported_weights(
                prefix.with_suffix(".json"), prefix.with_suffix(".npz")
            )

    def test_load_exported_weights_rejects_unexpected_archive_entry(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_export_extra_entry"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True, exist_ok=True)
        prefix = tmp_root / "weights_phase0_export"
        archive_arrays = {
            "dense__kernel": np.ones((2, 3), dtype=np.float32),
            "extra": np.ones((1,), dtype=np.float32),
        }
        np.savez(prefix.with_suffix(".npz"), **archive_arrays)
        prefix.with_suffix(".json").write_text(
            json.dumps(
                {
                    "schema_version": "hydra_phase0_weight_export_v1",
                    "tensors": {
                        "dense/kernel": {
                            "archive_key": "dense__kernel",
                            "shape": [2, 3],
                            "dtype": "float32",
                        }
                    },
                }
            )
        )

        with self.assertRaisesRegex(ValueError, "unexpected archive entries"):
            load_exported_weights(
                prefix.with_suffix(".json"), prefix.with_suffix(".npz")
            )


if __name__ == "__main__":
    unittest.main()
