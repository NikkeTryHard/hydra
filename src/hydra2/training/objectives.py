"""WP-05B supervised objectives: masked behavior cloning + auxiliary heads.

Implements masked cross-entropy over legal actions and auxiliary losses
with explicit weights. All weights and masking are caller-supplied; zero
weight heads may be absent (no implicit defaults).

Deterministic: no nondeterministic ops; illegal logits are masked to -inf
(float("-inf")) before softmax so illegal probability is exactly zero
(exp(-inf)=0) and gradients for illegal actions are exactly zero.

Portability: no hardcoded paths (``/home/cachybtw/tmp``), GPU (``sm_120``/
``RTX 5070``), or CTA device strings.  All ops stay on ``logits.device`` /
``targets.device`` via ``torch.arange(..., device=targets.device)`` and
``logits.new_zeros(())``; contract ``.item()`` syncs below are eager-only
and guarded by ``torch.compiler.is_compiling()`` + ``torch._check_tensor_all``
for compile compatibility (see perf-A §4.5/§6; ``pyrefly.toml``
promotes ``pytorch-efficiency-lint-item-call`` to ``warn`` and
``ruff`` ``PERF`` is enabled per-file for this module).  Tmpfs vs xfs
(``/tmp`` tmpfs evicted, ``/home/cachybtw/tmp`` xfs durable per
perf-A §8.1) does not affect these pure-tensor ops; only ``config.py:25``
``DEFAULT_ARTIFACT_ROOT`` selects the durable root.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812 -- conventional alias per PyTorch docs; lowercase diverges from docs. Evidence: https://docs.pytorch.org/docs/stable/nn.functional.html

from hydra2.contracts.common import ContractError, IllegalActionError

__all__ = [
    "compute_metrics",
    "compute_supervised_loss",
    "masked_cross_entropy",
    "masked_topk_accuracy",
]

# Masked logit value — ``-inf`` gives exact zero illegal prob (exp(-inf)=0)
# and exact zero gradient; ``-1e9`` also underflows to 0 in fp32/fp64 but
# ``-inf`` is mathematically exact and matches ``eval/baseline.py`` and
# ``models/model.py`` (masked_policy).  When all logits are masked (all-false
# legal row) softmax would be NaN for *both* ``-inf`` and ``-1e9`` (sum 0);
# that case is a hard ``ContractError`` per SPEC 11.1 before masking, so
# finiteness is preserved.  Single source avoids magic ``-1.0e9`` repeats.
_MASKED_LOGIT_NEG: float = float("-inf")
# ECE bins — 10 equal-width bins over ``[0,1]`` confidence; frozen for
# metric comparability across runs (not a tuned hyperparam).
_ECE_NUM_BINS: int = 10

def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked cross-entropy over legal actions only.

    Args:
        logits: ``[B,A]`` unmasked logits.
        targets: ``[B]`` chosen action indices.
        legal_mask: ``[B,A]`` bool, ``True`` = legal.  Each row MUST contain
            at least one ``True``; ``targets`` MUST be legal.

    Returns:
        Scalar loss (mean over batch).  Gradients for illegal logits are
        exactly zero (masked to ``-inf`` before softmax).
    """
    if logits.dim() != 2:
        raise ContractError(f"logits must be [B,A], got shape {tuple(logits.shape)}")
    if legal_mask.shape != logits.shape:
        raise ContractError(
            f"legal_mask shape {tuple(legal_mask.shape)} != logits shape {tuple(logits.shape)}"
        )
    if legal_mask.dtype != torch.bool:
        raise ContractError(f"legal_mask dtype must be bool, got {legal_mask.dtype}")
    if targets.dim() != 1 or targets.shape[0] != logits.shape[0]:
        raise ContractError(
            f"targets shape {tuple(targets.shape)} incompatible with logits {tuple(logits.shape)}"
        )
    if targets.dtype not in (torch.int64, torch.long, torch.int32):
        # Allow int64/int32 but coerce to long for indexing
        targets = targets.long()
    # Each row must have at least one legal action (nonterminal check)
    if torch.compiler.is_compiling():
        torch._check_tensor_all(
            legal_mask.any(dim=1),
            lambda: "nonterminal all-false legal row is hard error (SPEC 11.1)",
        )
    elif bool(torch.all(legal_mask.any(dim=1)).item()) is False:  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for contract validation; alternative loses validation. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        raise ContractError("nonterminal all-false legal row is hard error (SPEC 11.1)")
    # Targets must be in range and legal
    num_actions = logits.shape[1]
    if torch.compiler.is_compiling():
        torch._check_tensor_all(
            targets >= 0,
            lambda: f"target action_id out of range [0,{num_actions})",
        )
        torch._check_tensor_all(
            targets < num_actions,
            lambda: f"target action_id out of range [0,{num_actions})",
        )
    elif bool((targets < 0).any().item()) is True or bool(  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for range check; alternative loses validation. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        (targets >= num_actions).any().item()  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for range check; alternative loses validation. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
    ) is True:
        raise ContractError(f"target action_id out of range [0,{num_actions})")
    batch_idx = torch.arange(targets.shape[0], device=targets.device)
    if torch.compiler.is_compiling():
        torch._check_tensor_all(
            legal_mask[batch_idx, targets],
            lambda: "selected action is illegal per legal_mask",
        )
    elif bool(torch.all(legal_mask[batch_idx, targets]).item()) is False:  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for legality check; alternative loses validation. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        raise IllegalActionError("selected action is illegal per legal_mask")
    # Mask illegal to -inf so illegal probability exactly zero (exp(-inf)=0, gradient zero)
    masked_logits = logits.masked_fill(~legal_mask, _MASKED_LOGIT_NEG)
    loss = F.cross_entropy(masked_logits, targets.long(), reduction="mean")
    if torch.compiler.is_compiling():
        torch._check_tensor_all(
            torch.isfinite(loss), lambda: "masked CE produced non-finite loss"
        )
    elif bool(torch.isfinite(loss).item()) is False:  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for finite check; alternative loses validation. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        raise ContractError(
            f"masked CE produced non-finite loss: {loss.item()}"  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for error message; alternative loses diagnostics. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        )
    return loss


def masked_topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal_mask: torch.Tensor,
    k: int = 1,
) -> float:
    """Top-k accuracy restricted to legal actions.

    Returns float in [0,1].  Deterministic.
    """
    if k <= 0:
        raise ContractError(f"k must be positive, got {k}")
    masked_logits = logits.masked_fill(~legal_mask, _MASKED_LOGIT_NEG)
    # Get top-k among all (illegals are -inf so never top)
    topk = torch.topk(masked_logits, k=min(k, masked_logits.shape[1]), dim=1).indices
    # Check if target in topk for each row
    correct = (topk == targets.unsqueeze(1)).any(dim=1).float().mean()
    return float(correct.item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for metric reporting; alternative loses metric. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
def _generic_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """Unmasked cross-entropy helper for auxiliary heads."""
    if logits.shape[0] != targets.shape[0]:
        raise ContractError(f"{name}: logits/targets batch mismatch")
    return F.cross_entropy(logits, targets.long(), reduction="mean")


def _generic_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ContractError(f"{name}: shape mismatch {tuple(pred.shape)} vs {tuple(target.shape)}")
    return F.mse_loss(pred, target.float(), reduction="mean")


def compute_supervised_loss(
    model_output: dict[str, Any],
    batch: dict[str, Any],
    weights: dict[str, Any],
) -> dict[str, Any]:
    """Compute SPEC 20 supervised loss.

    Formula:
        L = w_policy * masked_cross_entropy
          + w_placement * placement_loss
          + sum_h w_event[h] * event_loss[h]
          + sum_h w_belief[h] * belief_loss[h]

    Args:
        model_output: must contain ``policy_logits`` ``[B,A]``; optional
            ``placement_logits`` ``[B,4]`` or ``[B,num_classes]``,
            ``value_vector`` ``[B,4]``, ``event_logits`` dict.
        batch: must contain ``chosen_action_id`` ``[B]``, ``legal_mask``
            ``[B,A]``; optional auxiliary targets:
            ``placement_target`` ``[B]``, ``value_target`` ``[B,4]``,
            ``event_targets`` dict ``{head_id: [B]}``.
        weights: dict with keys ``w_policy``, ``w_placement``,
            ``w_event`` (dict), ``w_belief`` (dict).  Missing keys
            default to ``0``; zero-weight heads MAY be absent.

    Returns:
        dict with ``total`` scalar tensor and ``policy``, ``placement``,
        ``event``, ``belief`` components (tensors) for logging.

    Raises:
        ContractError if a nonzero-weight head is absent or shapes mismatch.
    """
    if "legal_mask" not in batch or "chosen_action_id" not in batch:
        raise ContractError("batch must contain 'legal_mask' and 'chosen_action_id'")
    if "policy_logits" not in model_output:
        raise ContractError("model_output missing 'policy_logits'")
    legal_mask: torch.Tensor = batch["legal_mask"]
    targets: torch.Tensor = batch["chosen_action_id"]
    logits: torch.Tensor = model_output["policy_logits"]

    w_policy = float(weights.get("w_policy", 0.0))
    w_placement = float(weights.get("w_placement", 0.0))
    w_event: dict[str, float] = dict(weights.get("w_event", {}) or {})
    w_belief: dict[str, float] = dict(weights.get("w_belief", {}) or {})

    losses: dict[str, Any] = {}
    total = logits.new_zeros(())

    # Policy (masked BC) — the only required head when w_policy>0
    if w_policy != 0.0:
        ploss = masked_cross_entropy(logits, targets, legal_mask)
        losses["policy"] = ploss
        total = total + w_policy * ploss
    else:
        # Even when weight zero, we still compute for reporting if logits present,
        # but do not require it.  For determinism we put zero.
        losses["policy"] = logits.new_zeros(())

    # Placement auxiliary
    if w_placement != 0.0:
        if "placement_logits" not in model_output:
            raise ContractError("w_placement>0 but model_output missing 'placement_logits'")
        if "placement_target" not in batch:
            raise ContractError("w_placement>0 but batch missing 'placement_target'")
        pl_logits: torch.Tensor = model_output["placement_logits"]
        pl_target: torch.Tensor = batch["placement_target"]
        # Support both class indices [B] and one-hot/distribution [B,C]
        if pl_target.dim() == 1:
            ploss = _generic_ce_loss(pl_logits, pl_target, name="placement")
        else:
            # Distribution: KL or MSE? Use MSE for simplicity if distribution
            ploss = _generic_mse_loss(
                F.softmax(pl_logits, dim=-1), pl_target.float(), name="placement_dist"
            )
        losses["placement"] = ploss
        total = total + w_placement * ploss
    else:
        losses["placement"] = logits.new_zeros(())

    # Event auxiliary heads
    event_losses: dict[str, torch.Tensor] = {}
    for head_id, w in w_event.items():
        if w == 0.0:
            continue
        logits_key = f"event_logits_{head_id}"
        # Also accept dict form: model_output["event_logits"][head_id]
        ev_logits: torch.Tensor | None = None
        if "event_logits" in model_output and isinstance(model_output["event_logits"], dict):
            ev_logits = model_output["event_logits"].get(head_id)
        if ev_logits is None:
            ev_logits = model_output.get(logits_key)
        if ev_logits is None:
            raise ContractError(f"w_event[{head_id!r}]>0 but missing logits for that head")
        # Targets: batch["event_targets"][head_id] or batch[f"event_target_{head_id}"]
        ev_target: torch.Tensor | None = None
        if "event_targets" in batch and isinstance(batch["event_targets"], dict):
            ev_target = batch["event_targets"].get(head_id)
        if ev_target is None:
            ev_target = batch.get(f"event_target_{head_id}")
        if ev_target is None:
            ev_target = batch.get(f"event_targets_{head_id}")
        if ev_target is None:
            raise ContractError(f"w_event[{head_id!r}]>0 but missing target for that head")
        eloss = _generic_ce_loss(ev_logits, ev_target, name=f"event[{head_id}]")
        event_losses[head_id] = eloss
        total = total + w * eloss
    losses["event"] = (
        sum(event_losses.values(), start=logits.new_zeros(()))
        if len(event_losses) > 0
        else logits.new_zeros(())
    )
    # Store per-head for logging (detached later)
    losses["_event_per_head"] = event_losses

    # Belief auxiliary heads (treated identically to event for WP-05B)
    belief_losses: dict[str, torch.Tensor] = {}
    for head_id, w in w_belief.items():
        if w == 0.0:
            continue
        b_logits: torch.Tensor | None = None
        if "belief_logits" in model_output and isinstance(model_output["belief_logits"], dict):
            b_logits = model_output["belief_logits"].get(head_id)
        if b_logits is None:
            b_logits = model_output.get(f"belief_logits_{head_id}")
        if b_logits is None:
            raise ContractError(f"w_belief[{head_id!r}]>0 but missing logits")
        b_target: torch.Tensor | None = None
        if "belief_targets" in batch and isinstance(batch["belief_targets"], dict):
            b_target = batch["belief_targets"].get(head_id)
        if b_target is None:
            b_target = batch.get(f"belief_target_{head_id}")
        if b_target is None:
            raise ContractError(f"w_belief[{head_id!r}]>0 but missing target")
        bloss = _generic_ce_loss(b_logits, b_target, name=f"belief[{head_id}]")
        belief_losses[head_id] = bloss
        total = total + w * bloss
    losses["belief"] = (
        sum(belief_losses.values(), start=logits.new_zeros(()))
        if len(belief_losses) > 0
        else logits.new_zeros(())
    )
    losses["_belief_per_head"] = belief_losses

    losses["total"] = total
    # Finite check on total
    if torch.compiler.is_compiling():
        torch._check_tensor_all(torch.isfinite(total), lambda: "total loss non-finite")
    elif bool(torch.isfinite(total).item()) is False:  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for finite check; alternative loses validation. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        raise ContractError(f"total loss non-finite: {total.item()}")  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for error message; alternative loses diagnostics. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
    return losses

def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal_mask: torch.Tensor,
) -> dict[str, float]:
    """Report masked NLL, top-k, calibration, support/confusion, strata, legal-uniform.

    All metrics are deterministic and computed on the legal subspace only.

    Returns dict with keys:
      masked_nll, top1, top3, top5, calibration_ece, legal_uniform_nll,
      support_min, support_max, confusion (placeholder), strata (placeholder)
    """
    # ---- Contract validation (same guarantees as masked_cross_entropy) ----
    if logits.dim() != 2:
        raise ContractError(f"logits must be [B,A], got shape {tuple(logits.shape)}")
    if legal_mask.shape != logits.shape:
        raise ContractError(
            f"legal_mask shape {tuple(legal_mask.shape)} != logits shape {tuple(logits.shape)}"
        )
    if legal_mask.dtype != torch.bool:
        raise ContractError(f"legal_mask dtype must be bool, got {legal_mask.dtype}")
    if targets.dim() != 1 or targets.shape[0] != logits.shape[0]:
        raise ContractError(
            f"targets shape {tuple(targets.shape)} incompatible with logits {tuple(logits.shape)}"
        )
    if targets.dtype not in (torch.int64, torch.long, torch.int32):
        targets = targets.long()
    num_actions = logits.shape[1]
    if torch.compiler.is_compiling():
        torch._check_tensor_all(
            legal_mask.any(dim=1),
            lambda: "nonterminal all-false legal row is hard error (SPEC 11.1)",
        )
        torch._check_tensor_all(
            targets >= 0,
            lambda: f"target action_id out of range [0,{num_actions})",
        )
        torch._check_tensor_all(
            targets < num_actions,
            lambda: f"target action_id out of range [0,{num_actions})",
        )
        torch._check_tensor_all(torch.isfinite(logits), lambda: "logits must be finite")
    elif bool(torch.all(legal_mask.any(dim=1)).item()) is False:  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for contract; alternative loses validation. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        raise ContractError("nonterminal all-false legal row is hard error (SPEC 11.1)")
    elif bool((targets < 0).any().item()) is True or bool(  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for range check; alternative loses validation. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        (targets >= num_actions).any().item()  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for range check; alternative loses validation. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
    ) is True:
        raise ContractError(f"target action_id out of range [0,{num_actions})")
    # Targets must be legal per mask (also required for legal_uniform_nll correctness)
    _batch_idx = torch.arange(targets.shape[0], device=targets.device)
    if torch.compiler.is_compiling():
        torch._check_tensor_all(
            legal_mask[_batch_idx, targets.long()],
            lambda: "selected action is illegal per legal_mask",
        )
    elif bool(torch.all(legal_mask[_batch_idx, targets.long()]).item()) is False:  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for legality; alternative loses validation. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        raise IllegalActionError("selected action is illegal per legal_mask")
    if not torch.compiler.is_compiling() and bool(torch.isfinite(logits).all().item()) is False:  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for finite check; alternative loses validation. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        raise ContractError("logits must be finite (no inf/nan)")
    # ---- End validation ----
    masked_logits = logits.masked_fill(~legal_mask, _MASKED_LOGIT_NEG)
    log_prob = F.log_softmax(masked_logits, dim=-1)
    batch_idx = torch.arange(targets.shape[0], device=targets.device)
    nll = -log_prob[batch_idx, targets.long()].mean().item()  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for metric; alternative loses metric. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html

    # Legal-uniform baseline: uniform over legal actions (requires target legality, validated above)
    legal_counts = legal_mask.sum(dim=1).float()
    uniform_nll = torch.log(legal_counts).mean().item()  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for metric; alternative loses metric. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
    if not math.isfinite(float(uniform_nll)):
        raise ContractError(f"legal_uniform_nll must be finite, got {uniform_nll!r}")

    # Top-k
    top1 = masked_topk_accuracy(logits, targets, legal_mask, k=1)
    k3 = masked_topk_accuracy(logits, targets, legal_mask, k=min(3, logits.shape[1]))
    k5 = masked_topk_accuracy(logits, targets, legal_mask, k=min(5, logits.shape[1]))

    # Calibration (ECE) with frozen _ECE_NUM_BINS bins over confidence (max softmax prob)
    probs = F.softmax(masked_logits, dim=-1)
    confidences, predictions = probs.max(dim=1)
    accuracies = (predictions == targets.long()).float()
    ece = 0.0
    num_bins = _ECE_NUM_BINS
    for b in range(num_bins):
        lo = b / num_bins
        hi = (b + 1) / num_bins
        mask = (
            (confidences >= lo) & (confidences < hi)
            if b < num_bins - 1
            else (confidences >= lo) & (confidences <= hi)
        )
        if bool(mask.any().item()) is True:  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for bin check; alternative loses metric. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
            bin_acc = accuracies[mask].mean().item()  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for metric; alternative loses metric. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
            bin_conf = confidences[mask].mean().item()  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for metric; alternative loses metric. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
            ece += (mask.float().mean().item()) * abs(bin_acc - bin_conf)  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for ECE; alternative loses metric. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html

    # Support / confusion placeholders (per-class counts)
    # Support: min/max count of predicted class among legal
    pred_counts: dict[int, int] = {}
    # Note: predictions already legal due to masking
    pred_list: Any = predictions.tolist()
    for p_any in pred_list:
        p: int = int(p_any)
        pred_counts[p] = pred_counts.get(p, 0) + 1
    if len(pred_counts) > 0:
        sup_min = min(pred_counts.values())
        sup_max = max(pred_counts.values())
    else:
        sup_min = 0
        sup_max = 0

    return {
        "masked_nll": float(nll),
        "top1": top1,
        "top3": k3,
        "top5": k5,
        "calibration_ece": ece,
        "legal_uniform_nll": float(uniform_nll),
        "legal_uniform_gap": float(uniform_nll - nll),  # positive means better than uniform
        "support_min": float(sup_min),
        "support_max": float(sup_max),
        # Strata placeholder: would be per-seat/split breakdown in real loop
        "strata": 0.0,
        "confusion": 0.0,
    }
