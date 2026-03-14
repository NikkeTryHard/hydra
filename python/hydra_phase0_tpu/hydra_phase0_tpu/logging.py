from __future__ import annotations

import json
from pathlib import Path


def ensure_logging_dirs(output_dir: Path) -> None:
    (output_dir / "tb").mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
