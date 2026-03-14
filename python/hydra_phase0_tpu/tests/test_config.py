from __future__ import annotations

from pathlib import Path
import sys
import unittest

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hydra_phase0_tpu.config import load_config


class ConfigTests(unittest.TestCase):
    def test_load_config_applies_defaults(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_config_test"
        tmp_root.mkdir(exist_ok=True)
        config_path = tmp_root / "config.yaml"
        config_path.write_text(
            """
data:
  export_root: /tmp/export
run:
  output_dir: /tmp/out
  seed: 0
  num_epochs: 1
""".strip()
        )

        config = load_config(config_path)
        self.assertEqual(config.data.buffer_games, 50_000)
        self.assertEqual(config.data.buffer_samples, 32_768)
        self.assertEqual(config.run.batch_size, 2048)
        self.assertEqual(config.run.microbatch_size, 2048)
        self.assertEqual(config.run.validation_microbatch_size, 2048)

    def test_load_config_rejects_zero_runtime_intervals(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_config_invalid_runtime"
        tmp_root.mkdir(exist_ok=True)
        config_path = tmp_root / "config.yaml"
        config_path.write_text(
            """
data:
  export_root: /tmp/export
run:
  output_dir: /tmp/out
  seed: 0
  num_epochs: 1
  batch_size: 0
""".strip()
        )

        with self.assertRaises(ValidationError):
            load_config(config_path)

    def test_load_config_rejects_invalid_optimizer_bounds(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_config_invalid_optimizer"
        tmp_root.mkdir(exist_ok=True)
        config_path = tmp_root / "config.yaml"
        config_path.write_text(
            """
data:
  export_root: /tmp/export
run:
  output_dir: /tmp/out
  seed: 0
  num_epochs: 1
optimizer:
  learning_rate: 0.0001
  min_learning_rate: 0.001
""".strip()
        )

        with self.assertRaises(ValidationError):
            load_config(config_path)


if __name__ == "__main__":
    unittest.main()
