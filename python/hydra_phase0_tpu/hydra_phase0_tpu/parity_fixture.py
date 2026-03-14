from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .export_weights import export_weights
from .model import HydraModel


def _to_list(array: jnp.ndarray) -> list[float]:
    return np.asarray(array, dtype=np.float32).reshape(-1).tolist()


def generate_phase0_parity_fixture(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model = HydraModel()
    obs = jnp.linspace(-1.0, 1.0, num=192 * 34, dtype=jnp.float32).reshape((1, 192, 34))
    params = model.init(jax.random.PRNGKey(0), obs)["params"]
    out = model.apply({"params": params}, obs)

    export_weights(params, output_dir / "weights_phase0_export")
    np.save(output_dir / "obs.npy", np.asarray(obs, dtype=np.float32))

    payload = {
        "policy_logits": _to_list(out.policy_logits),
        "value": _to_list(out.value),
        "score_pdf": _to_list(out.score_pdf),
        "score_cdf": _to_list(out.score_cdf),
        "opp_tenpai": _to_list(out.opp_tenpai),
        "grp": _to_list(out.grp),
        "opp_next_discard": _to_list(out.opp_next_discard),
        "danger": _to_list(out.danger),
        "oracle_critic": _to_list(out.oracle_critic),
        "belief_fields": _to_list(out.belief_fields),
        "mixture_weight_logits": _to_list(out.mixture_weight_logits),
        "opponent_hand_type": _to_list(out.opponent_hand_type),
        "delta_q": _to_list(out.delta_q),
        "safety_residual": _to_list(out.safety_residual),
    }
    (output_dir / "expected_outputs.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python -m hydra_phase0_tpu.parity_fixture <output_dir>"
        )
    generate_phase0_parity_fixture(Path(sys.argv[1]))
