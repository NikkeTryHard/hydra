from __future__ import annotations

import sys
from unittest.mock import patch

from hydra_learner.train_bc import PYTHON_VARIANT_DEFAULT, parse_args


def test_parse_args_defaults_to_compile_max_autotune() -> None:
    with patch.object(sys, "argv", ["python-bc-train"]):
        args = parse_args()

    assert PYTHON_VARIANT_DEFAULT == "compile_max_autotune"
    assert args.variant == "compile_max_autotune"
