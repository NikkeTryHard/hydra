"""Publish complete tensor artifacts without exposing partial files to concurrent readers."""

import os
import tempfile
from pathlib import Path

import torch


def atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as f:
        temporary = Path(f.name)
        try:
            torch.save(payload, f)
            f.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
