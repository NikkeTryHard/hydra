from __future__ import annotations

from pathlib import Path
import sys
import unittest

import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hydra_phase0_tpu.model import HydraModel


class ModelTests(unittest.TestCase):
    def test_learner_output_shapes(self) -> None:
        model = HydraModel()
        obs = jnp.zeros((2, 192, 34), dtype=jnp.float32)
        params = model.init(jax.random.PRNGKey(0), obs)["params"]
        out = model.apply({"params": params}, obs)
        self.assertEqual(out.policy_logits.shape, (2, 46))
        self.assertEqual(out.value.shape, (2, 1))
        self.assertEqual(out.score_pdf.shape, (2, 64))
        self.assertEqual(out.score_cdf.shape, (2, 64))
        self.assertEqual(out.opp_tenpai.shape, (2, 3))
        self.assertEqual(out.grp.shape, (2, 24))
        self.assertEqual(out.opp_next_discard.shape, (2, 3, 34))
        self.assertEqual(out.danger.shape, (2, 3, 34))
        self.assertEqual(out.oracle_critic.shape, (2, 4))
        self.assertEqual(out.belief_fields.shape, (2, 16, 34))
        self.assertEqual(out.mixture_weight_logits.shape, (2, 4))
        self.assertEqual(out.opponent_hand_type.shape, (2, 24))
        self.assertEqual(out.delta_q.shape, (2, 46))
        self.assertEqual(out.safety_residual.shape, (2, 46))


if __name__ == "__main__":
    unittest.main()
