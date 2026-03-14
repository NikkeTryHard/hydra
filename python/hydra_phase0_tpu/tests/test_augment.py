from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hydra_phase0_tpu.augment import apply_train_augmentation


class AugmentTests(unittest.TestCase):
    def test_apply_train_augmentation_expands_6x(self) -> None:
        batch = {
            "obs": np.zeros((1, 192, 34), dtype=np.float32),
            "action": np.array([1], dtype=np.uint8),
            "legal_mask": np.zeros((1, 46), dtype=np.float32),
            "opp_next_target": np.zeros((1, 3, 34), dtype=np.float32),
            "danger_target": np.zeros((1, 3, 34), dtype=np.float32),
            "danger_mask": np.zeros((1, 3, 34), dtype=np.float32),
            "safety_residual_target": np.zeros((1, 46), dtype=np.float32),
            "safety_residual_mask": np.zeros((1, 46), dtype=np.float32),
            "belief_fields_target": np.zeros((1, 16, 34), dtype=np.float32),
        }
        batch["obs"][0, 40, 0] = 1.0
        batch["legal_mask"][0, 1] = 1.0
        out = apply_train_augmentation(batch)
        self.assertEqual(out["obs"].shape[0], 6)
        self.assertEqual(out["action"].shape[0], 6)
        self.assertEqual(out["legal_mask"].shape[0], 6)
        self.assertEqual(out["belief_fields_target"].shape[0], 6)


if __name__ == "__main__":
    unittest.main()
