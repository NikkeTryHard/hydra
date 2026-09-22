"""Drive-batch parity for PBRF — frozen pins vs the Rust driver."""

from __future__ import annotations

import json

import pytest

RULES = "sha256:" + "a" * 64
OBS = "sha256:" + "b" * 64
TARGET = "sha256:" + "c" * 64


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


def _ticket(n: int = 2, **over) -> bytes:
    worlds = [_world("sha256:" + ("1" * 64 if i == 0 else "2" * 64)) for i in range(n)]
    doc = {
        "worlds": worlds,
        "rules_hash": RULES,
        "observation_hash": OBS,
        "root_legal": [1, 2],
        "root_seat": 0,
        "candidate_id": "candidate3",
        "case_id": "case_0",
        "parent_count": 16,
        "policy_set": [],
        "epoch": "epoch:test",
        "target_id": TARGET,
        "max_sims": n,
        "max_depth": 4,
        "max_transitions": 256,
        "max_model_calls": 64,
        "deadline_ms": 5000,
        "fallback_margin_ms": 200,
        "tie_break": "lowest_action_id",
        "seed_hex": "00",
        "cursor": 0,
    }
    doc.update(over)
    return json.dumps(doc).encode()


def test_pbrf_batch_selects_and_blobs() -> None:
    from hydra2._native import search as s

    out = s.pbrf_search_batch(_ticket(2))
    assert out.candidate_ids == [1, 2]
    assert out.completed is True
    assert sum(out.visits) == out.sims_run
    assert out.decision_digest.startswith("sha256:")
    assert len(out.next_state_blob) > 0


def test_pbrf_batch_replay_identical() -> None:
    from hydra2._native import search as s

    a = s.pbrf_search_batch(_ticket(2))
    b = s.pbrf_search_batch(_ticket(2))
    assert a.selected_id == b.selected_id
    assert list(a.value_vectors) == list(b.value_vectors)
    assert a.decision_digest == b.decision_digest


def test_pbrf_batch_empty_fails_closed() -> None:
    from hydra2._native import search as s

    with pytest.raises(ValueError):
        s.pbrf_search_batch(b"")
    with pytest.raises(ValueError):
        s.pbrf_search_batch(b"not json")
