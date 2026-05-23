#!/usr/bin/env python3
"""Run the experimental Hydra PyTorch policy oracle from the source tree."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "python" / "hydra_learner"
sys.path.insert(0, str(PACKAGE))

from hydra_learner.cli import main

if __name__ == "__main__":
    sys.exit(main())
