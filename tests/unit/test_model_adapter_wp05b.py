"""WP-05B real-model adapter — unit + real train-step integration.

Covers ``hydra2.training.adapters``: the ``ModelOutput`` -> loss-dict bridge
(keys/shapes matching ``compute_supervised_loss``, ``[B,4,4]`` / ``[B,4]``
asserted at return) and ``encode_actor_rows`` over the existing encoder path.
The integration test runs a real ``Hydra2BaselineModel`` against a real
``AuthoritativeParquetDataset`` (synthetic actor parquet) for one train step
through ``compute_supervised_loss`` with ``w_policy=1.0``.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest
import torch
from hydra2_replay_rs import contracts as _bridge_contracts  # pyrefly: ignore[missing-import]

from hydra2.contracts.common import ContractError
from hydra2.contracts.event_envelope import (
    EventEnvelope,
    EventPayload,
)
from hydra2.contracts.observation_actor import make_actor_observation
from hydra2.data.parquet import DecisionRow, write_actor_shards
from hydra2.models.model import Hydra2BaselineModel, ModelOutput
from hydra2.models.schema import BASELINE_ACTION_COUNT
from hydra2.training.adapters import encode_actor_rows, model_output_to_loss_dict
from hydra2.training.dataset_store import AuthoritativeParquetDataset
from hydra2.training.objectives_loss import compute_supervised_loss

pytestmark = pytest.mark.contract_package("WP-05B")

_BATCH = 2


def _make_turn_advance(sequence: int, actor: int = 0) -> EventEnvelope:
    payload = EventPayload(
        kind="turn_advance",
        actor=actor,
        tile=None,
        action_id=None,
        source_seat=None,
        consumed_tiles=(),
        offered_action_ids=(),
        accepted_action_ids=(),
        round_index=None,
        scores=None,
        reason=None,
    )
    return EventEnvelope(
        game_id="g-wp05b-adapter",
        sequence=sequence,
        kind="turn_advance",
        actor=actor,
        visibility="public",
        visible_to=(0, 1, 2, 3),
        payload=payload,
        public_delta=(),
        rules_hash="sha256:" + "ab" * 32,
        schema_hash="sha256:" + "ac" * 32,
    )


def _legal_mask(*legal: int) -> tuple[bool, ...]:
    mask = [False] * BASELINE_ACTION_COUNT
    for action in legal:
        mask[action] = True
    return tuple(mask)


def _make_observation(
    *,
    actor: int = 0,
    decision_id: str = "d-0-1",
    sequence: int = 1,
    history: tuple[EventEnvelope, ...] = (),
    legal_mask: tuple[bool, ...] | None = None,
) -> Any:
    return make_actor_observation(
        game_id="g-wp05b-adapter",
        decision_id=decision_id,
        sequence=sequence,
        actor=actor,
        rules_id="tenhou_4p_hanchan_v1",
        rules_hash="sha256:" + "ab" * 32,
        action_table_hash="sha256:" + "ac" * 32,
        event_schema_hash="sha256:" + "ad" * 32,
        observation_schema_hash="sha256:" + "ae" * 32,
        packet_boundary_hash="sha256:" + "af" * 32,
        round_index=0,
        round_wind=27,
        hand_number=0,
        seat_winds=(27, 28, 29, 30),
        honba=0,
        riichi_sticks=0,
        dealer=0,
        scores=(25000, 25000, 25000, 25000),
        turn_actor=0,
        phase="draw_decision",
        live_wall_tiles_remaining=70,
        kan_count=0,
        ippatsu_active=(False, False, False, False),
        actor_furiten="none",
        actor_can_tsumo=True,
        actor_can_riichi=False,
        pending_declaration_discard=None,
        concealed_hand=(0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48),
        own_drawn_tile=None,
        visible_discards=((), (), (), ()),
        visible_melds=((), (), (), ()),
        riichi_states=("none", "none", "none", "none"),
        dora_indicators=(-1, -1, -1, -1, -1),
        visible_history=tuple(history),
        legal_mask=legal_mask if legal_mask is not None else _legal_mask(0, 10),
    )


def _digest(byte: str) -> Any:
    return _bridge_contracts.make_digest_text("sha256:" + (byte * 64)[:64])


def _make_model_output(
    *,
    batch: int = _BATCH,
    policy: torch.Tensor | None = None,
    placement: torch.Tensor | None = None,
    value: torch.Tensor | None = None,
    event: dict[str, torch.Tensor] | None = None,
    belief: dict[str, torch.Tensor] | Any | None = None,
) -> ModelOutput:
    return ModelOutput(
        policy_logits=policy if policy is not None else torch.randn(batch, BASELINE_ACTION_COUNT),
        placement_logits=placement if placement is not None else torch.randn(batch, 4, 4),
        value_vector=value if value is not None else torch.randn(batch, 4),
        event_logits=event if event is not None else {"next_event": torch.randn(batch, 5)},
        belief_logits=belief if belief is not None else {"next_event": torch.randn(batch, 5)},
        diagnostics={},
        utility_id="expected_final_placement_tenhou_4p_hanchan_v1",
        utility_manifest_hash=_digest("ab"),
        model_identity=_digest("ac"),
    )


# ---------------------------------------------------------------------------
# model_output_to_loss_dict
# ---------------------------------------------------------------------------


def test_loss_dict_keys_and_shapes_from_real_model() -> None:
    torch.manual_seed(0)
    observations = [
        _make_observation(decision_id=f"d-real-{i}", sequence=i + 1) for i in range(_BATCH)
    ]
    model = Hydra2BaselineModel()
    model.eval()
    output = model.evaluate(encode_actor_rows(observations))
    loss_dict = model_output_to_loss_dict(output)
    assert set(loss_dict) == {
        "policy_logits",
        "placement_logits",
        "value_vector",
        "event_logits",
        "belief_logits",
    }
    assert tuple(loss_dict["policy_logits"].shape) == (_BATCH, BASELINE_ACTION_COUNT)
    assert tuple(loss_dict["placement_logits"].shape) == (_BATCH, 4, 4)
    assert tuple(loss_dict["value_vector"].shape) == (_BATCH, 4)
    assert set(loss_dict["event_logits"]) == {"next_event"}
    assert set(loss_dict["belief_logits"]) == {"next_event"}


def test_loss_dict_rejects_non_model_output() -> None:
    with pytest.raises(ContractError):
        model_output_to_loss_dict({"policy_logits": torch.zeros(1, 2)})  # type: ignore[arg-type]


def test_loss_dict_rejects_bad_policy_shape() -> None:
    with pytest.raises(ContractError):
        model_output_to_loss_dict(_make_model_output(policy=torch.randn(_BATCH, 16)))


def test_loss_dict_rejects_bad_placement_shape() -> None:
    with pytest.raises(ContractError):
        model_output_to_loss_dict(_make_model_output(placement=torch.randn(_BATCH, 4)))
    with pytest.raises(ContractError):
        model_output_to_loss_dict(_make_model_output(placement=torch.randn(_BATCH, 3, 4)))


def test_loss_dict_rejects_bad_value_shape() -> None:
    with pytest.raises(ContractError):
        model_output_to_loss_dict(_make_model_output(value=torch.randn(_BATCH)))
    with pytest.raises(ContractError):
        model_output_to_loss_dict(_make_model_output(value=torch.randn(_BATCH + 1, 4)))


def test_loss_dict_rejects_head_batch_mismatch_and_non_mapping() -> None:
    with pytest.raises(ContractError):
        model_output_to_loss_dict(
            _make_model_output(event={"next_event": torch.randn(_BATCH + 1, 5)})
        )
    with pytest.raises(ContractError):
        model_output_to_loss_dict(_make_model_output(belief=torch.randn(_BATCH, 5)))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# encode_actor_rows
# ---------------------------------------------------------------------------


def test_encode_actor_rows_shapes() -> None:
    history = (_make_turn_advance(1), _make_turn_advance(2))
    observations = [
        _make_observation(decision_id="d-e0", sequence=1),
        _make_observation(decision_id="d-e1", sequence=2, history=history),
        _make_observation(decision_id="d-e2", sequence=3, actor=1),
    ]
    batch = encode_actor_rows(observations)
    assert tuple(batch.history_mask.shape) == (3, 32)
    assert tuple(batch.legal_mask.shape) == (3, BASELINE_ACTION_COUNT)
    assert tuple(batch.actor_seats.shape) == (3,)
    assert len(batch.observation_hashes) == 3
    # Tuples are accepted as well (Sequence contract).
    batch_tuple = encode_actor_rows(tuple(observations))
    assert tuple(batch_tuple.legal_mask.shape) == (3, BASELINE_ACTION_COUNT)


def test_encode_actor_rows_rejects_empty_and_non_observations() -> None:
    with pytest.raises(ContractError):
        encode_actor_rows([])
    with pytest.raises(ContractError):
        encode_actor_rows([object()])  # type: ignore[list-item]


def test_encode_actor_rows_rejects_unbound_hash() -> None:
    bad = _make_observation(decision_id="d-unbound", sequence=9)
    object.__setattr__(bad, "observation_hash", None)
    with pytest.raises(ContractError):
        encode_actor_rows([bad])


# ---------------------------------------------------------------------------
# Integration: real model + real dataset, one train step
# ---------------------------------------------------------------------------


def _write_adapter_parquet(tmp_path: Path, num_rows: int = 8) -> Path:
    rows: list[DecisionRow] = []
    for i in range(num_rows):
        obs = {
            "dora_indicators": [10 + (i % 5), 11 + (i % 5), -1, -1, -1],
            "hand_counts": [4] * 34,
            "history_mask": [1] * 8 + [0] * 8,
            "legal_mask_bits": [1] * BASELINE_ACTION_COUNT,
            "phase": "draw_decision",
        }
        rows.append(
            DecisionRow(
                game_id=f"game-{i // 4:03d}",
                round_id=f"round-{i // 4}-0",
                decision_id=f"dec-{i:04d}",
                seat=i % 4,
                source_object_id=f"obj-{i:04d}",
                split="train",
                rules_hash="sha256:" + "a" * 64,
                adapter_hash="sha256:" + "b" * 64,
                observation_hash="sha256:" + hashlib.sha256(f"obs-{i}".encode()).hexdigest(),
                action_table_hash="sha256:" + "c" * 64,
                derivation_hash="sha256:" + "d" * 64,
                actor_observation=obs,  # type: ignore[arg-type]
                chosen_action_id=i % BASELINE_ACTION_COUNT,
            )
        )
    dest = tmp_path / "actor_parquet"
    write_actor_shards(
        destination=dest,
        rows=rows,
        dataset_hash="sha256:" + "e" * 64,
        split_manifest_hash="sha256:" + "f" * 64,
    )
    return dest


def test_real_model_real_dataset_train_step(tmp_path: Path) -> None:
    torch.manual_seed(0)
    parquet_dir = _write_adapter_parquet(tmp_path)
    dataset = AuthoritativeParquetDataset(
        parquet_dir=parquet_dir,
        feature_dim=16,
        num_actions=BASELINE_ACTION_COUNT,
        seed=0,
        verify=True,
    )
    ds_batch_any: Any = dataset.next_batch(4)
    assert ds_batch_any is not None
    ds_legal: torch.Tensor = ds_batch_any["legal_mask"]
    ds_chosen: torch.Tensor = ds_batch_any["chosen_action_id"]
    assert tuple(ds_legal.shape) == (4, BASELINE_ACTION_COUNT)

    # The encoder batch reuses the dataset's own legal masks so the chosen
    # actions stay legal under both views of the batch.
    observations = [
        _make_observation(
            decision_id=str(decision_id),
            sequence=index + 1,
            legal_mask=tuple(bool(flag) for flag in ds_legal[index].tolist()),
        )
        for index, decision_id in enumerate(ds_batch_any["_decision_ids"])
    ]
    actor_batch = encode_actor_rows(observations)
    assert bool(torch.equal(actor_batch.legal_mask, ds_legal)) is True

    model = Hydra2BaselineModel()
    model.eval()
    loss_dict = model_output_to_loss_dict(model.evaluate(actor_batch))
    # Placement/value keys are present (real heads) while their weights stay 0.0.
    assert tuple(loss_dict["placement_logits"].shape) == (4, 4, 4)
    assert tuple(loss_dict["value_vector"].shape) == (4, 4)

    losses = compute_supervised_loss(
        loss_dict,
        {"chosen_action_id": ds_chosen, "legal_mask": ds_legal},
        {"w_policy": 1.0, "w_placement": 0.0, "w_value": 0.0, "w_event": {}, "w_belief": {}},
    )
    for key in ("total", "policy", "placement", "value", "event", "belief"):
        assert key in losses, f"loss missing {key}"
    total: torch.Tensor = losses["total"]
    assert bool(torch.isfinite(total).item()) is True
    assert float(losses["placement"].detach().item()) == 0.0
    assert float(losses["value"].detach().item()) == 0.0

    total.backward()
    grad = model.policy_head.weight.grad
    assert grad is not None
    assert bool(torch.isfinite(grad).all().item()) is True
