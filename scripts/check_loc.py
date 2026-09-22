"""LOC gate: enforced ceilings per tree (stdlib only, zero new deps).

Ceilings: python Python 600 | tests 1000 | Lean 800 | crates Rust 2000.
600 keeps new modules inside the last high-detection review band;
800 matches the long-standing split-past threshold (AGENTS.md ~500/800).
Grandfather entries are exact repo-relpaths; splits remove entries, never add.
Rust 2000 covers only files above it (smaller modules below it split only
if the ceiling drops). Lean lists Yaku + Turn as-is; larger splits need
their own approval.
"""

from __future__ import annotations

import pathlib
import sys

try:
    from hydra2._native import contracts as _loc_bridge
except ImportError:  # pragma: no cover - stale .so falls back to the oracle below
    _loc_bridge = None  # type: ignore[assignment]

_bridge_loc_exceeds = getattr(_loc_bridge, "script_loc_exceeds", None)


def _loc_exceeds(n: int, lim: int) -> bool:
    """Bridge-first `n > lim` (byte-identical fallback)."""
    if _bridge_loc_exceeds is not None:
        return bool(_bridge_loc_exceeds(n, lim))
    return n > lim


LIMITS = {"python": 600, "tests": 1000, "lean": 800, "crates": 2000}
GRANDFATHERED = {
    # python (leaves first; god-driver last)
    "python/hydra2/data/replay_expand.py",
    "python/hydra2/data/shard_build.py",
    "python/hydra2/training/stream_train.py",
    "python/hydra2/training/shard_reader.py",
    "python/hydra2/models/encoder.py",
    "python/hydra2/models/model.py",
    "python/hydra2/models/schema.py",
    "python/hydra2/distillation/teacher.py",
    "python/hydra2/search/candidate0.py",
    "python/hydra2/search/gumbel.py",
    "python/hydra2/search/local_resolving.py",
    "python/hydra2/search/joint_type_world.py",
    # registry file; split with its package if ever
    "python/hydra2/search/modules/__init__.py",
    # tests
    "tests/unit/test_supervised_loop_wp05b.py",
    "tests/conformance/test_reference_corpus_wp04a.py",
    "tests/integration/test_data_lineage_wp04b.py",
    "tests/unit/test_pbrf_wp09a.py",
    "tests/unit/test_log_replay_wp14.py",
    "tests/unit/test_stream_train_wp14.py",
    "tests/unit/test_belief_natural_wp07a.py",
    # lean (grandfathered as-is; Yaku-split past-1200 never approved)
    "lean/Formal/Mahjong/Yaku.lean",
    "lean/Formal/Mahjong/Turn.lean",
    # crates (2000 covers walk+decisions+ingest only)
    "crates/feed/src/walk.rs",
    "crates/shard/src/decisions.rs",
    "crates/feed/src/ingest.rs",
}


def main() -> int:
    bad = []
    for root, lim in LIMITS.items():
        ext = ".lean" if root == "lean" else (".rs" if root == "crates" else ".py")
        for p in sorted(pathlib.Path(root).rglob(f"*{ext}")):
            if ".lake" in p.parts or "target" in p.parts:
                continue
            try:
                n = len(p.read_text(encoding="utf-8").splitlines())
            except OSError:
                continue
            if _loc_exceeds(n, lim) and str(p) not in GRANDFATHERED:
                bad.append(f"{p}:{n}")
    if bad:
        print("LOC-GATE FAIL " + ", ".join(bad))
        return 1
    print("LOC-GATE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
