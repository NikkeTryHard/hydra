"""Drive-batch parity for Persistence — frozen pins vs the Rust driver."""

from __future__ import annotations

import json

import pytest

RULES = "sha256:" + "a" * 64
OBS = "sha256:" + "b" * 64


def _world(wid: str) -> dict:
    return {
        "world_id": wid,
        "hands": [[0, 1], [2, 3], [4, 5], [6, 7]],
        "live": [8, 9],
        "dead": [],
        "step": 0,
        "turn": 0,
        "corpus_idx": 0,
        "snapshot": "snap:test",
    }


def _ticket(**over) -> bytes:
    doc = {
        "worlds": [_world("sha256:" + "1" * 64)],
        "rules_hash": RULES,
        "observation_hash": OBS,
        "root_legal": [5, 6],
        "candidate_id": "candidateP",
        "case_id": "case_0",
        "arm_id": "P",
        "epoch": "epoch:test",
        "forest_blob": [],
        "ponder_quota": None,
        "max_sims": 1,
        "max_depth": 4,
        "max_transitions": 128,
        "max_model_calls": 32,
        "deadline_ms": 5000,
        "fallback_margin_ms": 200,
        "tie_break": "lowest_action_id",
        "seed_hex": "00",
        "cursor": 0,
    }
    doc.update(over)
    return json.dumps(doc).encode()


def test_persistence_batch_retains_blob() -> None:
    from hydra2._native import search as s

    out = s.persistence_search_batch(_ticket())
    assert out.completed is True
    assert len(out.next_state_blob) > 0
    assert out.decision_digest.startswith("sha256:")


def test_persistence_batch_replay_identical() -> None:
    from hydra2._native import search as s

    a = s.persistence_search_batch(_ticket())
    b = s.persistence_search_batch(_ticket())
    assert a.selected_id == b.selected_id
    assert a.decision_digest == b.decision_digest


def test_persistence_batch_empty_legal_fails_closed() -> None:
    from hydra2._native import search as s

    with pytest.raises(ValueError):
        s.persistence_search_batch(_ticket(root_legal=[]))
    with pytest.raises(ValueError):
        s.persistence_search_batch(b"not json")
