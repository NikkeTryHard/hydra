"""WP-05A fused-head compat, big-trunk arch variants, and model builder."""

from __future__ import annotations

from typing import Any

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
# 12 fused QKV + small-heads compat (Slice A)
# ---------------------------------------------------------------------------


def test_legacy_qkv_heads_state_dict_migrates() -> None:
    """Legacy unfused checkpoints remap to fused keys with bitwise parity."""
    torch.manual_seed(7)
    template = Hydra2BaselineModel()
    template.eval()
    fused_keys = template.state_dict()
    d_model: int = int(template.d_model)
    n_layers: int = int(template.n_layers)
    n_small_in: int = int(template.small_heads.in_features)
    n_events: int = len(EVENT_KINDS)
    torch.manual_seed(1234)
    legacy: dict[str, torch.Tensor] = {}
    for key, value in fused_keys.items():
        if key.endswith("qkv_proj.weight") or key in ("small_heads.weight", "small_heads.bias"):
            continue
        legacy[key] = value.clone()
    for layer in range(n_layers):
        for proj in ("q", "k", "v"):
            legacy[f"layers.{layer}.{proj}_proj.weight"] = torch.randn(d_model, d_model)
    legacy["placement_head.weight"] = torch.randn(16, n_small_in)
    legacy["placement_head.bias"] = torch.randn(16)
    legacy["value_head.weight"] = torch.randn(4, n_small_in)
    legacy["value_head.bias"] = torch.randn(4)
    legacy["event_head.weight"] = torch.randn(n_events, n_small_in)
    legacy["event_head.bias"] = torch.randn(n_events)
    legacy["belief_head.weight"] = torch.randn(n_events, n_small_in)
    legacy["belief_head.bias"] = torch.randn(n_events)
    assert "small_heads.weight" not in legacy
    assert not any(key.endswith("qkv_proj.weight") for key in legacy)
    migrated_model = Hydra2BaselineModel()
    migrated_model.eval()
    result = migrated_model.load_state_dict(legacy, strict=True)
    assert list(result.missing_keys) == []
    assert list(result.unexpected_keys) == []
    # Second model gets the fused stacks assigned directly (independent cat in
    # test code); forward must then agree bitwise with the migrated model.
    expected: dict[str, torch.Tensor] = dict(legacy)
    for layer in range(n_layers):
        expected[f"layers.{layer}.qkv_proj.weight"] = torch.cat(
            [
                legacy[f"layers.{layer}.q_proj.weight"],
                legacy[f"layers.{layer}.k_proj.weight"],
                legacy[f"layers.{layer}.v_proj.weight"],
            ],
            dim=0,
        )
        del expected[f"layers.{layer}.q_proj.weight"]
        del expected[f"layers.{layer}.k_proj.weight"]
        del expected[f"layers.{layer}.v_proj.weight"]
    expected["small_heads.weight"] = torch.cat(
        [
            legacy["placement_head.weight"],
            legacy["value_head.weight"],
            legacy["event_head.weight"],
            legacy["belief_head.weight"],
        ],
        dim=0,
    )
    expected["small_heads.bias"] = torch.cat(
        [
            legacy["placement_head.bias"],
            legacy["value_head.bias"],
            legacy["event_head.bias"],
            legacy["belief_head.bias"],
        ],
        dim=0,
    )
    for key in (
        "placement_head.weight",
        "placement_head.bias",
        "value_head.weight",
        "value_head.bias",
        "event_head.weight",
        "event_head.bias",
        "belief_head.weight",
        "belief_head.bias",
    ):
        del expected[key]
    direct_model = Hydra2BaselineModel()
    direct_model.eval()
    direct_result = direct_model.load_state_dict(expected, strict=True)
    assert list(direct_result.missing_keys) == []
    assert list(direct_result.unexpected_keys) == []
    batch = encode_observations([_make_observation(history=_history_of_length(4))])
    with torch.no_grad():
        out_migrated = migrated_model.evaluate(batch)
        out_direct = direct_model.evaluate(batch)
    assert torch.isfinite(out_migrated.policy_logits).all()
    assert torch.equal(out_migrated.policy_logits, out_direct.policy_logits)
    assert torch.equal(out_migrated.placement_logits, out_direct.placement_logits)
    assert torch.equal(out_migrated.value_vector, out_direct.value_vector)
    assert torch.equal(
        out_migrated.event_logits["next_event"], out_direct.event_logits["next_event"]
    )
    assert torch.equal(
        out_migrated.belief_logits["next_event"], out_direct.belief_logits["next_event"]
    )


def test_fused_forward_matches_contract_shapes() -> None:
    """Fused heads keep contract shapes incl. the all-padding row."""
    torch.manual_seed(0)
    model = Hydra2BaselineModel()
    model.eval()
    n_events: int = len(EVENT_KINDS)
    assert model.small_heads.in_features == model.d_model * 2
    assert model.small_heads.out_features == 20 + 2 * n_events
    assert model.policy_head.out_features == BASELINE_ACTION_COUNT
    assert model.policy_head.in_features == model.d_model * 2
    batch = encode_observations(
        [_make_observation(history=_history_of_length(5)), _make_observation(history=())]
    )
    assert batch.history_mask.shape[0] == 2
    assert not batch.history_mask[1].any()
    out = model.evaluate(batch)
    assert out.policy_logits.shape == (2, BASELINE_ACTION_COUNT)
    assert out.placement_logits.shape == (2, 4, 4)
    assert out.value_vector.shape == (2, 4)
    assert out.event_logits["next_event"].shape == (2, n_events)
    assert out.belief_logits["next_event"].shape == (2, n_events)
    assert out.policy_logits.dtype == torch.float32
    assert out.placement_logits.dtype == torch.float32
    assert out.value_vector.dtype == torch.float32
    assert torch.isfinite(out.policy_logits).all()
    assert torch.isfinite(out.placement_logits).all()
    assert torch.isfinite(out.value_vector).all()
    assert torch.isfinite(out.event_logits["next_event"]).all()
    assert torch.isfinite(out.belief_logits["next_event"]).all()
    # Event/belief widths pin the off-by-E slice bounds.
    assert out.event_logits["next_event"].shape[1] == n_events
    assert out.belief_logits["next_event"].shape[1] == n_events
    # Legal gating still holds on the fused path (illegal exact zero).
    probs = masked_policy(out.policy_logits, batch.legal_mask)
    assert torch.all(probs[~batch.legal_mask] == 0)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-6)
    picked = select_actions(out.policy_logits, batch.legal_mask)
    for row, idx in enumerate(picked.tolist()):
        assert batch.legal_mask[row, idx].item() is True
    assert int(out.diagnostics["history_length"][0].item()) == 5
    assert int(out.diagnostics["history_length"][1].item()) == 0


def test_big_trunk_param_count_and_dims() -> None:
    """Mortal-class trunk: 8x256x8/d_ff1024 is exactly 9914166 params."""
    from hydra2.models.model import _ARCH_DEFAULTS

    model = Hydra2BaselineModel(
        architecture_id="hydra2_big_transformer_v1",
        d_model=256,
        n_heads=8,
        n_layers=8,
        d_ff=1024,
        dropout=0.0,
    )
    assert model.d_model == 256
    assert model.n_heads == 8
    assert model.n_layers == 8
    assert model.d_ff == 1024
    assert len(model.layers) == 8
    assert sum(p.numel() for p in model.parameters()) == 9914166
    assert _ARCH_DEFAULTS["hydra2_big_transformer_v1"]["d_model"] == 256
    # Unknown arch ids fail closed (never a silent baseline fallback).
    with pytest.raises(ContractError):
        Hydra2BaselineModel(architecture_id="hydra2_huge_transformer_v99")


def test_big_trunk_identity_binds_arch_and_dff() -> None:
    """Identity pins the arch id and the true d_ff (never the default)."""
    base = Hydra2BaselineModel()
    big = Hydra2BaselineModel(
        architecture_id="hydra2_big_transformer_v1",
        d_model=256,
        n_heads=8,
        n_layers=8,
        d_ff=1024,
        dropout=0.0,
    )
    assert base.model_identity != big.model_identity
    assert big.model_spec()["architecture_id"] == "hydra2_big_transformer_v1"
    assert big.model_spec()["architecture_parameters"]["d_ff"] == 1024
    assert base.model_spec()["architecture_parameters"]["d_ff"] == 256


def test_big_trunk_forward_contract_shapes() -> None:
    """Big trunk forward keeps every contract shape (heads scale with D)."""
    torch.manual_seed(0)
    model = Hydra2BaselineModel(
        architecture_id="hydra2_big_transformer_v1",
        d_model=256,
        n_heads=8,
        n_layers=8,
        d_ff=1024,
        dropout=0.0,
    )
    model.eval()
    assert model.policy_head.in_features == 2 * 256
    batch = encode_observations([_make_observation(history=_history_of_length(5))])
    out = model.evaluate(batch)
    assert out.policy_logits.shape == (1, BASELINE_ACTION_COUNT)
    assert out.placement_logits.shape == (1, 4, 4)
    assert out.value_vector.shape == (1, 4)
    assert torch.isfinite(out.policy_logits).all()


def test_build_model_selects_arch_defaults() -> None:
    """_build_model: big id gets big widths; explicit params win; unknown fails."""
    from types import SimpleNamespace

    from hydra2.training.stream_build import _build_model

    def _config(arch_id: str, parameters: dict) -> Any:
        return SimpleNamespace(
            model=SimpleNamespace(
                architecture_id=arch_id,
                action_count=BASELINE_ACTION_COUNT,
                parameters=parameters,
            )
        )

    big = _build_model(_config("hydra2_big_transformer_v1", {}))
    assert (big.d_model, big.n_heads, big.n_layers, big.d_ff) == (256, 8, 8, 1024)
    assert sum(p.numel() for p in big.parameters()) == 9914166
    narrow = _build_model(_config("hydra2_big_transformer_v1", {"d_model": 128, "n_heads": 4}))
    assert (narrow.d_model, narrow.n_heads, narrow.n_layers) == (128, 4, 8)
    baseline = _build_model(_config("hydra2_baseline_transformer_v1", {}))
    assert (baseline.d_model, baseline.n_layers) == (128, 2)
    with pytest.raises(ContractError):
        _build_model(_config("hydra2_huge_transformer_v99", {}))
