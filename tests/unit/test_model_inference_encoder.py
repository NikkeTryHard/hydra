"""WP-05A actor-visible encoder and bucketed-history batching."""

from __future__ import annotations

import pytest
import torch

from hydra2.contracts.common import ContractError
from hydra2.models.encoder import (
    bucket_for_length,
    encode_observations,
    validate_batch_against_schema,
)
from hydra2.models.schema import _BASELINE_FIELDS, HISTORY_BUCKET_LENGTHS
from tests.unit._model_inference_helpers import _history_of_length, _make_observation

pytestmark = pytest.mark.contract_package("WP-05A")

# ---------------------------------------------------------------------------
# 1 actor-visible tensor encoder (no privileged fields)
# ---------------------------------------------------------------------------


def test_actor_visible_tensor_encoder_no_privileged_fields() -> None:
    obs = _make_observation()
    batch = encode_observations([obs])
    validate_batch_against_schema(batch)

    # Every feature name must be in the frozen baseline field table.
    expected_names = {f.name for f in _BASELINE_FIELDS}
    assert set(batch.features.keys()) == expected_names
    # Encoder isolation: the live import-graph proof passes (behavioral —
    # observes the loaded module graph, not source text). The batch asserts
    # below pin the observable side (actor-typed, schema-valid construction).
    from hydra2.belief.oracle_join import (
        assert_privileged_loader_isolated_from_encoder as _assert_encoder_isolated,
    )

    assert _assert_encoder_isolated() is None
    assert batch.legal_mask.dtype == torch.bool
    assert bool(batch.legal_mask.any(dim=1).all().item())

    # Rejects non-ActorObservation and missing observation_hash.
    with pytest.raises(ContractError):
        encode_observations([object()])  # type: ignore[arg-type]
    # Privileged leakage via observation hash absent should be rejected

    bad = _make_observation()
    object.__setattr__(bad, "observation_hash", None)
    with pytest.raises(ContractError):
        encode_observations([bad])


# ---------------------------------------------------------------------------
# 2 padded/bucketed histories with explicit masks initially
# ---------------------------------------------------------------------------


def test_padded_bucketed_histories_with_explicit_masks() -> None:
    # Bucket boundaries
    assert bucket_for_length(0) == 32
    assert bucket_for_length(1) == 32
    assert bucket_for_length(32) == 32
    assert bucket_for_length(33) == 64
    assert bucket_for_length(64) == 64
    assert bucket_for_length(65) == 128
    assert bucket_for_length(200) == 256
    assert bucket_for_length(999) == 256

    # Empty history -> bucket 32, all padding
    obs_empty = _make_observation(history=())
    batch_empty = encode_observations([obs_empty])
    assert batch_empty.history_mask.shape == (1, 32)
    assert not batch_empty.history_mask.any()
    assert (batch_empty.features["history_event_kind"] == 0).all()

    # Length 2 -> bucket 32, first 2 true, rest false/padding 0
    obs2 = _make_observation(history=_history_of_length(2))
    batch2 = encode_observations([obs2])
    assert batch2.history_mask.shape == (1, 32)
    assert batch2.history_mask[0, :2].all()
    assert not batch2.history_mask[0, 2:].any()
    # Padded positions carry padding_value 0 without participation.
    assert (batch2.features["history_event_kind"][0, 2:] == 0).all()

    # Batch with mixed lengths buckets to max length's bucket.
    obs_a = _make_observation(history=_history_of_length(1))
    obs_b = _make_observation(history=_history_of_length(33))
    batch_mixed = encode_observations([obs_a, obs_b])
    assert batch_mixed.history_mask.shape == (2, 64)
    assert batch_mixed.history_mask[0, :1].all()
    assert not batch_mixed.history_mask[0, 1:].any()
    assert batch_mixed.history_mask[1, :33].all()
    assert not batch_mixed.history_mask[1, 33:].any()

    # History mask True means participate; masked validation.
    validate_batch_against_schema(batch_mixed)

    # Non-bucket sizes are not emitted: 33 correctly lands in 64, not 33.
    assert batch_mixed.history_mask.shape[1] in HISTORY_BUCKET_LENGTHS
