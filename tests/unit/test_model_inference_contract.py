"""WP-05A model/inference contract: determinism, shapes, masks, SDPA dropout."""

from __future__ import annotations

import pytest
import torch

from hydra2.contracts.common import ContractError
from hydra2.contracts.event_vocab import EVENT_KINDS
from hydra2.models.encoder import encode_observations
from hydra2.models.model import Hydra2BaselineModel, masked_policy, select_actions
from hydra2.models.schema import BASELINE_ACTION_COUNT
from tests.unit._model_inference_helpers import _history_of_length, _make_observation

pytestmark = pytest.mark.contract_package("WP-05A")

# ---------------------------------------------------------------------------
# 3 model contract + inference contract (deterministic, shapes, masks)
# ---------------------------------------------------------------------------


def test_model_contract_inference_contract_deterministic_shapes_masks() -> None:
    torch.manual_seed(0)
    model = Hydra2BaselineModel()
    model.eval()

    obs = _make_observation(history=_history_of_length(5))
    batch = encode_observations([obs, _make_observation(actor=1, history=_history_of_length(3))])

    out1 = model.evaluate(batch)
    out2 = model.evaluate(batch)
    # Deterministic: bitwise identical logits in eval mode.
    assert torch.equal(out1.policy_logits, out2.policy_logits)
    assert torch.equal(out1.placement_logits, out2.placement_logits)
    assert torch.equal(out1.value_vector, out2.value_vector)
    assert torch.equal(out1.event_logits["next_event"], out2.event_logits["next_event"])
    assert torch.equal(out1.belief_logits["next_event"], out2.belief_logits["next_event"])

    # Shapes (B = 2, A = 6792)
    assert out1.policy_logits.shape == (2, BASELINE_ACTION_COUNT)
    assert out1.placement_logits.shape == (2, 4, 4)
    assert out1.value_vector.shape == (2, 4)
    assert out1.event_logits["next_event"].shape == (2, len(EVENT_KINDS))
    assert out1.belief_logits["next_event"].shape == (2, len(EVENT_KINDS))

    # Legal mask respected through masked_policy helper (illegal exact zero).
    probs = masked_policy(out1.policy_logits, batch.legal_mask)
    assert probs.shape == (2, BASELINE_ACTION_COUNT)
    # Illegal entries are exactly zero.
    assert torch.all(probs[~batch.legal_mask] == 0)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)

    # select_actions never picks illegal.
    picked = select_actions(out1.policy_logits, batch.legal_mask)
    for row, idx in enumerate(picked.tolist()):
        assert batch.legal_mask[row, idx].item() is True

    # Tie is deterministic first-legal-max (candidate spec).
    logits_tie = torch.zeros(1, BASELINE_ACTION_COUNT)
    legal_tie = torch.zeros(1, BASELINE_ACTION_COUNT, dtype=torch.bool)
    legal_tie[0, [5, 9, 100]] = True
    # all zeros -> first legal (5) should be chosen.
    assert int(select_actions(logits_tie, legal_tie).item()) == 5

    # Contract rejects nonterminal all-false legal row.
    bad_mask = tuple(False for _ in range(BASELINE_ACTION_COUNT))
    with pytest.raises(ContractError):
        _make_observation(legal_mask=bad_mask)


# ---------------------------------------------------------------------------
# 4 SDPA dense attention; eval dropout exactly zero
# ---------------------------------------------------------------------------


def test_sdpa_dense_attention_eval_dropout_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    # Dense (non-causal) attention with eval dropout exactly 0, pinned
    # behaviorally: spy the SDPA entry point during evaluate and capture
    # the dispatched kwargs (is_causal / dropout_p per layer call).
    torch.manual_seed(123)
    model = Hydra2BaselineModel(dropout=0.1)
    model.train()
    assert model.dropout_p == pytest.approx(0.1)
    batch = encode_observations([_make_observation(history=_history_of_length(4))])
    model.eval()
    calls: list[dict[str, object]] = []
    orig_sdpa = torch.nn.functional.scaled_dot_product_attention

    def _spy(query: object, key: object, value: object, *args: object, **kwargs: object) -> object:
        if "is_causal" in kwargs:
            is_causal: object = kwargs["is_causal"]
        elif len(args) >= 3:
            is_causal = args[2]
        else:
            is_causal = False
        if "dropout_p" in kwargs:
            dropout_p: object = kwargs["dropout_p"]
        elif len(args) >= 2:
            dropout_p = args[1]
        else:
            dropout_p = 0.0
        calls.append({"is_causal": is_causal, "dropout_p": dropout_p})
        return orig_sdpa(query, key, value, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", _spy)
    with torch.no_grad():
        o_eval_1 = model.evaluate(batch)
        o_eval_2 = model.evaluate(batch)
    # Eval is exactly deterministic (dropout 0): repeated eval forwards agree
    # bitwise, and train-mode stochasticity cannot leak into the eval path.
    assert torch.equal(o_eval_1.policy_logits, o_eval_2.policy_logits)
    assert torch.equal(o_eval_1.value_vector, o_eval_2.value_vector)
    # Every layer dispatched dense attention with dropout gated off in eval.
    assert len(calls) == 2 * len(model.layers) and len(calls) > 0
    assert all(call["is_causal"] is False for call in calls)
    assert all(call["dropout_p"] == 0.0 for call in calls)
    for layer in model.layers:
        assert layer.dropout == 0.1  # configured train rate, gated off in eval
