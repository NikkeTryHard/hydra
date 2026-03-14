from __future__ import annotations

from pathlib import Path
import shutil
import sys
import unittest

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hydra_phase0_tpu.checkpoints import restore_checkpoint, save_checkpoint
from hydra_phase0_tpu.checkpoints import (
    ResumeState,
    RuntimeResumeContract,
    read_resume_state,
    write_resume_state,
)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_roundtrip(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_checkpoint_test"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        state = {
            "params": {"w": jnp.ones((2, 2), dtype=jnp.float32)},
            "opt_state": {"step": jnp.asarray(3, dtype=jnp.int32)},
        }
        save_checkpoint(tmp_root, state, {"global_step": 3})
        restored = restore_checkpoint(tmp_root)
        self.assertTrue(jnp.allclose(restored["params"]["w"], state["params"]["w"]))
        self.assertEqual(int(restored["opt_state"]["step"]), 3)

    def test_resume_state_roundtrip(self) -> None:
        tmp_root = Path(__file__).resolve().parent / ".tmp_resume_state_test"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        state = ResumeState(
            epoch=2,
            global_step=7,
            next_batch_index=3,
            runtime=RuntimeResumeContract(
                manifest_fingerprint="abc123",
                batch_size=2048,
                train_microbatch_size=512,
                validation_microbatch_size=512,
                model_fingerprint='{"compute_dtype": "bfloat16"}',
                optimizer_fingerprint='{"learning_rate": 0.00025}',
            ),
        )
        write_resume_state(tmp_root, state)
        restored = read_resume_state(tmp_root)
        self.assertEqual(restored, state)


if __name__ == "__main__":
    unittest.main()
