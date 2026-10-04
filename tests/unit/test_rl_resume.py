"""WP-11 gate: ordinary receipts, control publication, tip authority."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.contract_package("WP-11")

_MANIFEST_KEYS = (
    "checkpoint_version",
    "run_spec_hash",
    "model_spec_hash",
    "model_state_hash",
    "optimizer_spec_hash",
    "optimizer_state_hash",
    "scheduler_spec_hash",
    "scheduler_state_hash",
    "training_state_hash",
    "sampler_state_hash",
    "rng_state_hash",
    "environment_hash",
    "rules_hash",
    "utility_manifest_hash",
    "action_schema_hash",
    "observation_schema_hash",
    "dataset_manifest_hash",
    "rollout_artifact_hash",
    "parent_checkpoint_hash",
)


def _jcs_minimal(document: object) -> bytes:
    """Independent JCS for ASCII-only fixtures (no serde_jcs on this path)."""
    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _manifest(version: str = "1.0.0", *, dataset: bool = False) -> dict:
    doc = {}
    for key in _MANIFEST_KEYS:
        if key == "checkpoint_version":
            doc[key] = version
        elif key == "dataset_manifest_hash":
            doc[key] = ("sha256:" + "d" * 64) if dataset else None
        elif key == "rollout_artifact_hash":
            doc[key] = None if dataset else ("sha256:" + "a" * 64)
        elif key == "parent_checkpoint_hash":
            doc[key] = "sha256:" + "c" * 64
        else:
            doc[key] = "sha256:" + "e" * 64
    return doc


def _bridge():
    from hydra2.training import rl_bridge

    return rl_bridge


def _write(root: Path, locator: str, payload: bytes) -> None:
    path = root / locator
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _receipts(bridge, root: Path) -> tuple[dict, dict, list[dict]]:
    _write(root, "ckpt/learner.pt", b"learner-payload")
    _write(root, "ckpt/actor.pt", b"actor-payload")
    learner = bridge.checkpoint_receipt(str(root), "ckpt/learner.pt", _manifest())
    actor = bridge.checkpoint_receipt(str(root), "ckpt/actor.pt", _manifest())
    assert learner["manifest_id"] == actor["manifest_id"]
    assert learner["manifest_id"].startswith("sha256:")
    # Independent FULL-JCS manifest id (no digest exclusion anywhere here).
    expect = "sha256:" + hashlib.sha256(_jcs_minimal(_manifest())).hexdigest()
    assert learner["manifest_id"] == expect
    return learner, actor, [learner, actor]


def _control(
    learner: dict,
    actor: dict,
    parent: str,
    phase: str,
    generation: int,
    **overrides: object,
) -> dict:
    doc: dict = {
        "schema_version": "1.0.0",
        "pipeline_spec_id": "sha256:" + "0" * 64,
        "learner_manifest_id": learner["manifest_id"],
        "learner_transport_sha": learner["transport_sha256"],
        "learner_locator": learner["locator"],
        "actor_manifest_id": actor["manifest_id"],
        "actor_transport_sha": actor["transport_sha256"],
        "actor_locator": actor["locator"],
        "generation": generation,
        "generation_statuses": {},
        "source_refs": {},
        "order_ref": "sha256:" + "1" * 64,
        "ledger_refs": {},
        "native_snapshot_digest": "sha256:" + "2" * 64,
        "tape_blob_digest": "sha256:" + "2" * 64,
        "event_blob_digest": "sha256:" + "2" * 64,
        "cohort_membership": {"generations": 8, "games_per_generation": 128},
        "frontier_current": {"generation": generation, "round": 0, "epoch": 0},
        "frontier_next": {"generation": generation, "round": 0, "epoch": 0},
        "terminal_set": [],
        "tickets": [
            {
                "ticket_descriptor": "slot-0:epoch-1",
                "phase": "ready",
                "slot_epoch": 1,
                "stream_cursor": {"before": 0, "after": 8},
            }
        ],
        "committed_row_offset": 0,
        "committed_update_offset": 0,
        "publication_phase": phase,
        "parent_checkpoint_hash": parent,
    }
    doc.update(overrides)
    body = {key: value for key, value in doc.items() if key != "digest"}
    doc["digest"] = "sha256:" + hashlib.sha256(_jcs_minimal(body)).hexdigest()
    return doc


def test_receipt_roundtrip_rejects_tamper_shapes(rust_extension: object, tmp_path: Path) -> None:
    _ = rust_extension
    bridge = _bridge()
    _write(tmp_path, "ckpt/l0.pt", b"fake-torch-bytes")
    receipt = bridge.checkpoint_receipt(str(tmp_path), "ckpt/l0.pt", _manifest())
    assert receipt["byte_len"] == len(b"fake-torch-bytes")
    # Tampered payload breaks the bound receipt at publication time.
    _write(tmp_path, "ckpt/l0.pt", b"tampered")
    with pytest.raises(ValueError):
        bridge.publish_control(
            str(tmp_path),
            _control(
                {
                    "manifest_id": receipt["manifest_id"],
                    "transport_sha256": receipt["transport_sha256"],
                    "locator": "ckpt/l0.pt",
                },
                {
                    "manifest_id": receipt["manifest_id"],
                    "transport_sha256": receipt["transport_sha256"],
                    "locator": "ckpt/l0.pt",
                },
                "sha256:" + "c" * 64,
                "not_captured",
                0,
            ),
            [receipt],
            "sha256:" + "c" * 64,
            [],
        )
    # Shape rejections: version, dual source, absolute/escape locators.
    _write(tmp_path, "ckpt/l0.pt", b"fake-torch-bytes")
    with pytest.raises(ValueError):
        bridge.checkpoint_receipt(str(tmp_path), "ckpt/l0.pt", _manifest(version="2.0.0"))
    with pytest.raises(ValueError):
        bridge.checkpoint_receipt(
            str(tmp_path),
            "ckpt/l0.pt",
            _manifest(dataset=True) | {"rollout_artifact_hash": "sha256:" + "a" * 64},
        )
    with pytest.raises(ValueError):
        bridge.checkpoint_receipt(str(tmp_path), "/abs/evil.pt", _manifest())
    with pytest.raises(ValueError):
        bridge.checkpoint_receipt(str(tmp_path), "../evil.pt", _manifest())
    with pytest.raises(ValueError):
        bridge.checkpoint_receipt(str(tmp_path), "ckpt/missing.pt", _manifest())


def test_publish_phases_tip_authority_fallback(rust_extension: object, tmp_path: Path) -> None:
    _ = rust_extension
    bridge = _bridge()
    parent = "sha256:" + "c" * 64
    learner, actor, receipts = _receipts(bridge, tmp_path)
    first = bridge.publish_control(
        str(tmp_path),
        _control(learner, actor, parent, "captured_at_update1", 2),
        receipts,
        parent,
        [],
    )
    assert first["phase"] == "captured_at_update1"
    assert bridge.read_tip(str(tmp_path))["publication_phase"] == "captured_at_update1"
    # Same-generation regression is rejected; forward motion commits.
    with pytest.raises(ValueError):
        bridge.publish_control(
            str(tmp_path), _control(learner, actor, parent, "not_captured", 2), receipts, parent, []
        )
    bridge.publish_control(
        str(tmp_path),
        _control(learner, actor, parent, "checkpoint_durable", 2),
        receipts,
        parent,
        [],
    )
    assert bridge.read_tip(str(tmp_path))["publication_phase"] == "checkpoint_durable"
    # Wrong fixed parent is rejected even for an otherwise valid envelope.
    with pytest.raises(ValueError):
        bridge.publish_control(
            str(tmp_path),
            _control(learner, actor, parent, "checkpoint_durable", 2),
            receipts,
            "sha256:" + "f" * 64,
            [],
        )
    # Corrupt the live tip: the one-backup tip (first phase) is authority.
    (tmp_path / "control" / "tip.json").write_bytes(b"{broken")
    assert bridge.read_tip(str(tmp_path))["publication_phase"] == "captured_at_update1"


def test_stale_tip_and_closed_envelope(rust_extension: object, tmp_path: Path) -> None:
    _ = rust_extension
    bridge = _bridge()
    parent = "sha256:" + "c" * 64
    learner, actor, receipts = _receipts(bridge, tmp_path)
    # No tip yet: reads fail closed, never default.
    with pytest.raises(ValueError):
        bridge.read_tip(str(tmp_path))
    bridge.publish_control(
        str(tmp_path), _control(learner, actor, parent, "not_captured", 1), receipts, parent, []
    )
    # Delete the committed control file: the tip is stale and rejected.
    for path in (tmp_path / "control").glob("control_*.json"):
        path.unlink()
    with pytest.raises(ValueError):
        bridge.read_tip(str(tmp_path))
    # Closed envelope: unknown keys never publish.
    evil = _control(learner, actor, parent, "not_captured", 1, extra_key="nope")
    with pytest.raises(ValueError):
        bridge.publish_control(str(tmp_path), evil, receipts, parent, [])
    # Missing receipt coverage never publishes.
    with pytest.raises(ValueError):
        bridge.publish_control(
            str(tmp_path), _control(learner, actor, parent, "not_captured", 1), [], parent, []
        )


def _native_game() -> dict:
    return {
        "schema_version": "1.0.0",
        "wall_id": "wall-1",
        "game_id": "game-a",
        "seat_policy_ids": ["ckpt-c0"] * 4,
        "ordinal": 3,
        "core_carry_hash": "sha256:" + "7" * 64,
        "action_prefix_digest": "sha256:" + "7" * 64,
        "state_digest": "sha256:" + "7" * 64,
        "decision_count": 41,
        "pending_responses": [],
        "stream_cursor": {"seed_hex": "00" * 32, "cursor": 88},
    }


def test_native_snapshot_stage_read_mismatch(rust_extension: object, tmp_path: Path) -> None:
    _ = rust_extension
    bridge = _bridge()
    receipt = bridge.stage_native_snapshot(str(tmp_path), _native_game())
    assert receipt["game_id"] == "game-a"
    digest = receipt["digest"]
    assert digest.startswith("sha256:")
    back = bridge.read_native_snapshot(str(tmp_path), "game-a", digest)
    assert back["ordinal"] == 3
    with pytest.raises(ValueError):
        bridge.read_native_snapshot(str(tmp_path), "game-B", digest)
    with pytest.raises(ValueError):
        bridge.read_native_snapshot(str(tmp_path), "game-a", "sha256:" + "0" * 64)
