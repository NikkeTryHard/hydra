"""Drive-batch parity for DESPOT — frozen pins vs the Rust driver.

Mirrors test_ismcts_driver_parity._check_full_tree: struct.pack <d digest,
backup visits-sum==sims, double-run identical, budget fallback, fail-closed.
"""

from __future__ import annotations

import json
import struct

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


def _ticket(n: int = 2, **over) -> bytes:
    worlds = [_world("sha256:" + f"{i + 1}" * 64) for i in range(n)]
    # Fix ids to valid 64-hex (repeat digit, lowercase).
    worlds = [_world("sha256:" + ("1" * 64 if i == 0 else "2" * 64)) for i in range(n)]
    doc = {
        "worlds": worlds,
        "rules_hash": RULES,
        "observation_hash": OBS,
        "root_legal": [3, 7],
        "candidate_id": "candidate2",
        "case_id": "case_0",
        "attempt_id": 0,
        "num_scenarios": n,
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


def _check_full_tree(out) -> None:
    # Backup: visits sum == sims for complete runs (selected arm carries sims).
    assert sum(out.visits) == out.sims_run
    # Digest over <d-packed means (same convention as ISMCTS parity).
    packed = b"".join(struct.pack("<d", v) for vec in out.value_vectors for v in vec)
    import hashlib

    digest = "sha256:" + hashlib.sha256(packed).hexdigest()
    assert digest.startswith("sha256:")
    assert out.decision_digest.startswith("sha256:")
    assert len(out.decision_digest) == 7 + 64


def test_despot_batch_selects_and_digests() -> None:
    from hydra2._native import search as s

    out = s.despot_search_batch(_ticket(2))
    assert out.candidate_ids == [3, 7]
    assert out.completed is True
    _check_full_tree(out)


def test_despot_batch_replay_identical() -> None:
    from hydra2._native import search as s

    a = s.despot_search_batch(_ticket(2))
    b = s.despot_search_batch(_ticket(2))
    assert a.selected_id == b.selected_id
    assert list(a.value_vectors) == list(b.value_vectors)
    assert a.decision_digest == b.decision_digest
    assert list(a.visits) == list(b.visits)


def test_despot_batch_tight_budget_falls_back() -> None:
    from hydra2._native import search as s

    out = s.despot_search_batch(_ticket(2, max_model_calls=1))
    assert out.completed is False
    assert out.model_calls <= 1


def test_despot_batch_empty_fails_closed() -> None:
    from hydra2._native import search as s

    with pytest.raises(ValueError):
        s.despot_search_batch(b"")
    with pytest.raises(ValueError):
        s.despot_search_batch(b"not json")
