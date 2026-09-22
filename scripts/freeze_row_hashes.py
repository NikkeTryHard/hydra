#!/usr/bin/env python3
"""Freeze / check per-decision row hashes for the wall-less replay cutover.

Slice 3 gate: ``sha256(canonical_bytes(row minus volatile keys))`` per
``decision_id``. Two projection modes (recorded in the fixture, never mixed):

- ``compact``: the parity-harness projection shared by
  ``oracle-compact.jsonl`` (``d/s/ph/a/hand/drawn/dora/wall/hk/mask``) and
  the Rust dump rows (``decision_id/seat/phase/chosen_action_id/`` ...).
  Key aliases and tile-value encodings (physical id vs MJAI string,
  ``-1`` vs ``null`` dora tail) are normalized so cross-schema comparison is
  meaningful; ``turn_actor``/chosen-detail/provenance stay covered by the
  ``parity_84`` join, not the hash.
- ``full``: whole ``DecisionRow`` JSON (``oracle-rows.jsonl``);
  ``derivation_hash`` is always dropped (spec) as is
  ``privileged_label_ref`` (privileged-join pointer, not an actor field).

Closed hash allowlist (operator decision D2): ``--allow`` accepts ONLY
``wall_game_id,copy_fold,refit_number,wall_less_marker``; anything else
fails closed. Each name drops a documented dimension symmetrically on both
sides (no-op where the mode has no such key):

- ``wall_game_id`` -> ``source_object_id`` (+ ``game_id``/``round_id`` in
  compact mode: the identity binding form, never the join key itself);
- ``copy_fold`` -> ``observation_hash`` (+ ``concealed``/``drawn``/``dora``
  in compact mode: pool-first copy-folded string content);
- ``refit_number`` -> ``wall``/``live_wall_tiles_remaining`` countdown;
- ``wall_less_marker`` -> ``derivation_hash``/``derivation``/``wall_digest``
  (the SIM-mark rev; always dropped in full mode regardless).

Oracle pinning (fail closed): ``freeze`` records sha256 of every Python
oracle input into the fixture metadata; ``check`` re-hashes the working
tree first and REFUSES with ``oracle moved - re-pin and re-run`` (exit 3)
when any pin differs. Moving oracle = void parity.

Usage (from the repo root, ``pixi run python`` so ``hydra2`` imports resolve):
  freeze:  scripts/freeze_row_hashes.py freeze --rows /tmp/replay-s1/oracle-compact.jsonl \\
               --out crates/tests/fixtures/frozen-row-hashes.json [--allow ...]
  check:   scripts/freeze_row_hashes.py check --frozen <fixture> --rows <rust-rows.jsonl>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    from hydra2._native import contracts as _freeze_bridge
except ImportError:  # pragma: no cover - stale .so falls back to the oracle below
    _freeze_bridge = None  # type: ignore[assignment]

_bridge_allow_is_valid = getattr(_freeze_bridge, "script_allow_is_valid", None)
_bridge_allow_drop_keys = getattr(_freeze_bridge, "script_allow_drop_keys", None)


def _allow_is_valid(name: str) -> bool:
    """Bridge-first closed-allowlist membership (byte-identical fallback)."""
    if _bridge_allow_is_valid is not None:
        return bool(_bridge_allow_is_valid(name))
    return name in CLOSED_ALLOW


def _allow_drop_keys(name: str) -> tuple[str, ...]:
    """Bridge-first ALLOW_DROP lookup (byte-identical fallback)."""
    if _bridge_allow_drop_keys is not None:
        keys = _bridge_allow_drop_keys(name)
        if keys is not None:
            return tuple(str(k) for k in keys)
    return ALLOW_DROP[name]


REPO_ROOT = Path(__file__).resolve().parent.parent

# Every oracle input the wall-less rows depend on (relpath -> pinned).
# The tile codec moved to Rust in Wave 3 (tiles.py deleted); its owner
# files are pinned here like any other oracle input.
ORACLE_INPUTS = (
    "python/hydra2/engines/riichienv/actions.py",
    "python/hydra2/engines/riichienv/events.py",
    "crates/shard/src/tile.rs",
    "crates/bridge/src/tiles.rs",
    "python/hydra2/engines/riichienv/state.py",
    "python/hydra2/engines/riichienv/identity.py",
    "python/hydra2/engines/riichienv/_oracle_base.py",
    "python/hydra2/engines/riichienv/_lr_act.py",
    "python/hydra2/engines/riichienv/_lr_claim.py",
    "python/hydra2/engines/riichienv/_lr_end.py",
    "python/hydra2/engines/riichienv/_lr_frame.py",
    "python/hydra2/engines/riichienv/_lr_oracle.py",
    "python/hydra2/engines/riichienv/_lr_rows.py",
    "python/hydra2/engines/riichienv/_lr_walk.py",
    "python/hydra2/engines/riichienv/_sp_capture.py",
    "python/hydra2/engines/riichienv/_sp_game.py",
    "python/hydra2/engines/riichienv/_sp_kans.py",
    "python/hydra2/engines/riichienv/_sp_oracle.py",
    "python/hydra2/engines/riichienv/_sp_reach.py",
    "python/hydra2/engines/riichienv/_sp_records.py",
    "python/hydra2/engines/riichienv/_sp_walk.py",
    "python/hydra2/engines/riichienv/_sp_windows.py",
    "python/hydra2/engines/riichienv/_sp_wins.py",
    "python/hydra2/engines/riichienv/adapter_identity.py",
    "python/hydra2/engines/riichienv/adapter_core.py",
    "python/hydra2/engines/riichienv/adapter_step.py",
    "python/hydra2/engines/riichienv/adapter_events_a.py",
    "python/hydra2/engines/riichienv/adapter_events_b.py",
    "python/hydra2/data/parquet.py",
    "python/hydra2/models/encoder.py",
    "configs/contracts/action_table_v1.json",
)

# Closed allowlist (D2): name -> top-level projection keys dropped before hashing.
CLOSED_ALLOW = ("wall_game_id", "copy_fold", "refit_number", "wall_less_marker")
ALLOW_DROP: dict[str, tuple[str, ...]] = {
    "wall_game_id": ("source_object_id", "game_id", "round_id"),
    "copy_fold": ("observation_hash", "concealed", "drawn", "dora"),
    "refit_number": ("wall", "live_wall_tiles_remaining"),
    "wall_less_marker": ("derivation_hash", "derivation", "wall_digest"),
}

ORACLE_MOVED = "oracle moved - re-pin and re-run"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def oracle_pins(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    """sha256 (bare hex) of every oracle input; raises if any is missing."""
    pins: dict[str, str] = {}
    for rel in ORACLE_INPUTS:
        path = repo_root / rel
        if not path.is_file():
            raise FileNotFoundError(f"oracle input missing: {rel}")
        pins[rel] = _sha256_file(path)
    return pins


def _mjai_string_of(tile_id: int):  # lazy import: needs the pixi env
    from hydra2._native import tiles

    return tiles.mjai_string_of(tile_id)


def _norm_tile_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0:  # dora sentinel tail
            return None
        return _mjai_string_of(value)
    raise ValueError(f"unhashable tile value: {value!r}")


def row_mode(row: dict) -> str:
    if "actor_observation" in row and "decision_id" in row:
        return "full"
    if "d" in row and "hk" in row:
        return "compact"
    if "decision_id" in row and "history_kinds" in row:
        return "compact"
    raise ValueError(f"unknown row schema, keys={sorted(row.keys())[:8]}")


def compact_projection(row: dict) -> dict:
    """Alias + value normalization shared by oracle-compact and Rust rows."""
    if "d" in row:  # oracle-compact
        hand = [str(_norm_tile_str(t)) for t in row.get("hand", [])]
        dora = [_norm_tile_str(t) for t in row.get("dora", [])]
        return {
            "decision_id": row["d"],
            "game_id": row.get("g"),
            "round_id": row.get("r"),
            "seat": row.get("s"),
            "phase": row.get("ph"),
            "chosen_action_id": row.get("a"),
            "concealed": hand,
            "drawn": _norm_tile_str(row.get("drawn")),
            "dora": dora,
            "wall": row.get("wall"),
            "history": list(row.get("hk", [])),
            "mask": sorted(row.get("mask", [])),
        }
    # Rust ReplayRow
    dora = [_norm_tile_str(t) for t in row.get("dora_indicators", [])]
    return {
        "decision_id": row["decision_id"],
        "game_id": row.get("game_id"),
        "round_id": row.get("round_id"),
        "seat": row.get("seat"),
        "phase": row.get("phase"),
        "chosen_action_id": row.get("chosen_action_id"),
        "concealed": list(row.get("concealed_hand", [])),
        "drawn": _norm_tile_str(row.get("drawn_tile")),
        "dora": dora,
        "wall": row.get("wall_remaining"),
        "history": list(row.get("history_kinds", [])),
        "mask": sorted(row.get("legal_mask", [])),
    }


def full_projection(row: dict) -> dict:
    """Whole DecisionRow minus derivation/privilege volatile keys."""
    doc = {k: v for k, v in row.items() if k not in ("derivation_hash", "privileged_label_ref")}
    obs = doc.get("actor_observation")
    if isinstance(obs, str):  # parquet column shape: JSON text
        try:
            doc["actor_observation"] = json.loads(obs)
        except json.JSONDecodeError:
            pass
    return doc


def _drop_paths(doc: dict, names: tuple[str, ...]) -> dict:
    doc = dict(doc)
    for key in names:
        doc.pop(key, None)
    obs = doc.get("actor_observation")
    if isinstance(obs, dict):
        doc["actor_observation"] = {k: v for k, v in obs.items() if k not in names}
    return doc


def row_hash(doc: dict, allow: tuple[str, ...], repo_root: Path = REPO_ROOT) -> str:
    """sha256 digest-text over RFC 8785 canonical bytes of the projection."""
    from hydra2.artifacts.digest import of_canonical

    for name in allow:
        doc = _drop_paths(doc, _allow_drop_keys(name))
    return str(of_canonical(doc))


def _parse_allow(values: list[str]) -> tuple[str, ...]:
    # Accept `--allow a b` and `--allow a,b` spellings alike.
    flat = [part for value in values for part in value.split(",") if part]
    unknown = [v for v in flat if not _allow_is_valid(v)]
    if unknown:
        print(
            f"error: --allow accepts only the closed set {list(CLOSED_ALLOW)}; "
            f"rejected {unknown} (open-ended waivers hide regressions)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return tuple(flat)


def freeze_rows(rows_path: Path, allow: tuple[str, ...], repo_root: Path = REPO_ROOT) -> dict:
    hashes: dict[str, str] = {}
    mode: str | None = None
    with open(rows_path) as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row_mode(row)
            mode = kind if mode is None else mode
            if kind != mode:
                raise ValueError(f"mixed row schemas at line {lineno}: {kind} vs {mode}")
            doc = full_projection(row) if kind == "full" else compact_projection(row)
            decision_id = row.get("decision_id") or row.get("d")
            hashes[decision_id] = row_hash(doc, allow, repo_root)
    return {
        "metadata": {
            "tool": "scripts/freeze_row_hashes.py",
            "mode": mode,
            "allow": list(allow),
            "row_count": len(hashes),
            "source_rows": str(rows_path),
            "source_rows_sha256": _sha256_file(rows_path),
            "oracle_pins": oracle_pins(repo_root),
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "hashes": hashes,
    }


def check_fixture(fixture: dict, rows_path: Path, repo_root: Path = REPO_ROOT) -> dict:
    """Fail closed on oracle drift, then join hashes; returns the verdict."""
    pins = fixture["metadata"].get("oracle_pins", {})
    current: dict[str, str] = {}
    moved = []
    for rel, recorded in pins.items():
        path = repo_root / rel
        if not path.is_file():
            moved.append(f"{rel} (missing)")
            continue
        current[rel] = _sha256_file(path)
        if current[rel] != recorded:
            moved.append(rel)
    if moved:
        return {"verdict": "oracle-moved", "moved": moved}
    allow = tuple(fixture["metadata"].get("allow", []))
    mode = fixture["metadata"].get("mode")
    expected: dict[str, str] = fixture["hashes"]
    actual: dict[str, str] = {}
    with open(rows_path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row_mode(row) != mode:
                return {"verdict": "mode-mismatch", "expected_mode": mode}
            doc = full_projection(row) if mode == "full" else compact_projection(row)
            decision_id = row.get("decision_id") or row.get("d")
            actual[decision_id] = row_hash(doc, allow, repo_root)
    identical = sum(1 for k, v in actual.items() if expected.get(k) == v)
    mismatch = sorted(k for k, v in actual.items() if k in expected and expected[k] != v)
    missing = sorted(k for k in expected if k not in actual)
    extra = sorted(k for k in actual if k not in expected)
    return {
        "verdict": "identical" if not mismatch and not missing and not extra else "mismatch",
        "identical": identical,
        "total_expected": len(expected),
        "total_actual": len(actual),
        "mismatch": mismatch,
        "missing": missing,
        "extra": extra,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="hash rows JSONL -> fixture JSON")
    freeze.add_argument("--rows", required=True, type=Path)
    freeze.add_argument("--out", required=True, type=Path)
    freeze.add_argument("--allow", nargs="*", default=[])
    freeze.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    check = sub.add_parser("check", help="compare rows JSONL against a fixture")
    check.add_argument("--frozen", required=True, type=Path)
    check.add_argument("--rows", required=True, type=Path)
    check.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    args = parser.parse_args(argv)

    if args.command == "freeze":
        allow = _parse_allow(args.allow)
        fixture = freeze_rows(args.rows, allow, args.repo_root)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(fixture, indent=1, sort_keys=True) + "\n")
        print(
            f"froze {fixture['metadata']['row_count']} hashes ({fixture['metadata']['mode']}) -> {args.out}"
        )
        return 0

    with open(args.frozen) as handle:
        fixture = json.load(handle)
    verdict = check_fixture(fixture, args.rows, args.repo_root)
    if verdict["verdict"] == "oracle-moved":
        print(f"{ORACLE_MOVED}: {verdict['moved']}", file=sys.stderr)
        return 3
    if verdict["verdict"] == "mode-mismatch":
        print(f"mode mismatch: fixture is {verdict['expected_mode']}", file=sys.stderr)
        return 2
    print(
        f"identical {verdict['identical']}/{verdict['total_expected']} "
        f"(actual {verdict['total_actual']}); mismatch {len(verdict['mismatch'])} "
        f"missing {len(verdict['missing'])} extra {len(verdict['extra'])}"
    )
    for sample in verdict["mismatch"][:10]:
        print(f"  mismatch: {sample}")
    return 0 if verdict["verdict"] == "identical" else 1


if __name__ == "__main__":
    raise SystemExit(main())
