"""LOC gate: enforced ceilings per tree (stdlib only, zero new deps).

Ceilings: src Python 600 | tests 1000 | Lean 800 | tools Rust 2000.
600 keeps new modules inside the last high-detection review band;
800 matches the long-standing split-past threshold (AGENTS.md ~500/800).
Grandfather entries are exact repo-relpaths; splits remove entries, never add.
Rust 2000 covers only files above it (smaller modules below it split only
if the ceiling drops). Lean lists Yaku + Turn as-is; larger splits need
their own approval.
"""

import pathlib
import sys

LIMITS = {"src": 600, "tests": 1000, "lean": 800, "tools": 2000}
GRANDFATHERED = {
    # src (leaves first; god-driver last)
    "src/hydra2/data/replay_expand.py",
    "src/hydra2/data/shard_build.py",
    "src/hydra2/training/run_config.py",
    "src/hydra2/training/stream_train.py",
    "src/hydra2/training/shard_reader.py",
    "src/hydra2/models/encoder.py",
    "src/hydra2/models/model.py",
    "src/hydra2/models/schema.py",
    "src/hydra2/belief/oracle_loader.py",
    "src/hydra2/belief/oracle_distillation.py",
    "src/hydra2/distillation/teacher.py",
    "src/hydra2/engines/riichienv/adapter.py",
    "src/hydra2/engines/riichienv/log_replay.py",
    "src/hydra2/engines/riichienv/single_pass.py",
    "src/hydra2/engines/mahjax/differential.py",
    "src/hydra2/search/candidate0.py",
    "src/hydra2/search/gumbel.py",
    "src/hydra2/search/local_resolving.py",
    "src/hydra2/search/joint_type_world.py",
    # registry file; split with its package if ever
    "src/hydra2/search/modules/__init__.py",
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
    # tools (2000 covers walk+decisions+ingest only)
    "tools/hydra2-replay-rs/crates/hydra-feed/src/walk.rs",
    "tools/hydra2-replay-rs/crates/hydra-shard/src/decisions.rs",
    "tools/hydra2-replay-rs/crates/hydra-feed/src/ingest.rs",
}


def main() -> int:
    bad = []
    for root, lim in LIMITS.items():
        ext = ".lean" if root == "lean" else (".rs" if root == "tools" else ".py")
        for p in sorted(pathlib.Path(root).rglob(f"*{ext}")):
            if ".lake" in p.parts or "target" in p.parts:
                continue
            try:
                n = len(p.read_text(encoding="utf-8").splitlines())
            except OSError:
                continue
            if n > lim and str(p) not in GRANDFATHERED:
                bad.append(f"{p}:{n}")
    if bad:
        print("LOC-GATE FAIL " + ", ".join(bad))
        return 1
    print("LOC-GATE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
