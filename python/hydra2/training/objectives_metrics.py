"""Supervised report metrics: NLL, top-k, calibration, temperature.

Masked NLL, top-k, calibration, support/confusion, strata, and legal-uniform
comparison reported.

Owns the eval-time report math over the legal subspace: shared metric-input
validation with fp32 upcast, frozen-bin ECE, support extremes, per-type
scorecards, single-temperature scaling fit on validation logits only, top-k
accuracy, hot scalars, and the aggregate report dict. Loss math lives in
:mod:`hydra2.training.objectives_loss`; this module imports only the mask
constant from it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

import torch
import torch.nn.functional as F  # noqa: N812 -- conventional alias per PyTorch docs; lowercase diverges from docs. Evidence: https://docs.pytorch.org/docs/stable/nn.functional.html

from hydra2.contracts.common import ContractError, IllegalActionError
from hydra2.training.objectives_loss import _MASKED_LOGIT_NEG as _MASKED_LOGIT_NEG

__all__ = [
    "compute_hot_scalars",
    "compute_metrics",
    "compute_per_type_metrics",
    "fit_temperature_scaling",
    "masked_topk_accuracy",
    "row_eval_primitives",
]

# ECE bins — 10 equal-width bins over ``[0,1]`` confidence; frozen for
# metric comparability across runs (not a tuned hyperparam).
_ECE_NUM_BINS: int = 10
# Minimum per-kind rows for a fully-supported scorecard slice.  Kinds with
# ``n < 30`` carry ``low_support = 1.0`` (informational only, report path;
# kinds never enter the loss, so support never affects training).
_PER_TYPE_MIN_N: int = 30
# Post-hoc temperature bounds (Guo et al. 2017: single-T Platt scaling fit
# on validation NLL only, accuracy-preserving, never fit on train).
_TEMPERATURE_MIN: float = 0.05
_TEMPERATURE_MAX: float = 20.0


def _validate_metric_inputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared contract validation for the metric helpers.

    Same guarantees as :func:`masked_cross_entropy` (all-false rows and
    illegal targets are hard errors; logits must be finite).  Returns
    ``(logits_fp32, masked_logits, log_prob, targets_long)`` where the
    fp32 upcast keeps bf16-autocast forwards from lying about metrics.
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
    elif (
        bool((targets < 0).any().item()) is True  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for range check; alternative loses validation. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        or bool(  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for range check; alternative loses validation. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
            (targets >= num_actions).any().item()  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for range check; alternative loses validation. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        )
        is True
    ):
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
    # Precision: upcast logits to fp32 before NLL/top-k/ECE so bf16 autocast
    # forward does not lie about metrics. Same math, comparable regimes.
    logits_fp32 = logits.to(torch.float32)
    masked_logits = logits_fp32.masked_fill(~legal_mask, _MASKED_LOGIT_NEG)
    log_prob = F.log_softmax(masked_logits, dim=-1)
    return logits_fp32, masked_logits, log_prob, targets.long()


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


def compute_hot_scalars(
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal_mask: torch.Tensor,
) -> dict[str, float]:
    """Per-update hot metrics: masked NLL + top-1/3/5 (4 host syncs).

    No validation (the batch was pre-validated this microbatch) and no
    uniform/ECE/support/per-type work — those ride the eval report
    (:func:`compute_metrics`). Keys mirror the :func:`compute_metrics`
    subset so hot entries keep ``masked_nll``/``top1`` continuity; top-3/5
    show ranking quality moving before top-1 does. The fused path still
    mints exact NLL + top1 in one Triton op; top-3/5 always come from the
    eager top-k on fp32 logits (same math both paths, parity-tested).
    """
    # Fused reporting fast path (exact NLL + top1, one Triton op; see
    # fused_ce.py): CUDA + triton + bf16/fp32 only. The eager fallback below
    # is unchanged for all other cases; top-3/5 ride eager top-k either way.
    fused: dict[str, float] | None = None
    if logits.is_cuda and logits.dtype in (torch.bfloat16, torch.float32):
        from hydra2.training.fused_ce import TRITON_AVAILABLE, fused_hot_scalars

        if TRITON_AVAILABLE and legal_mask.is_cuda and targets.is_cuda:
            targets_long = targets.long() if targets.dtype != torch.long else targets
            nll_vec: torch.Tensor
            ok_vec: torch.Tensor
            nll_vec, ok_vec = fused_hot_scalars(logits, legal_mask, targets_long)
            nll_mean: float = nll_vec.mean().item()  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # reporting fast path; alternative loses metric. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
            top1_mean: float = ok_vec.float().mean().item()  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # reporting fast path; alternative loses metric. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
            fused = {"masked_nll": nll_mean, "top1": top1_mean}
    logits_fp32 = logits.to(torch.float32)
    masked_logits = logits_fp32.masked_fill(~legal_mask, _MASKED_LOGIT_NEG)
    log_prob = F.log_softmax(masked_logits, dim=-1)
    targets_long = targets.long() if targets.dtype != torch.long else targets
    batch_idx = torch.arange(targets_long.shape[0], device=targets_long.device)
    if fused is not None:
        out = dict(fused)
    else:
        nll = -log_prob[batch_idx, targets_long].mean().item()  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for metric; alternative loses metric. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        out = {
            "masked_nll": float(nll),
            "top1": masked_topk_accuracy(logits_fp32, targets_long, legal_mask, k=1),
        }
    out["top3"] = masked_topk_accuracy(
        logits_fp32, targets_long, legal_mask, k=min(3, logits_fp32.shape[1])
    )
    out["top5"] = masked_topk_accuracy(
        logits_fp32, targets_long, legal_mask, k=min(5, logits_fp32.shape[1])
    )
    return out


def row_eval_primitives(
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Per-row eval primitives on CPU (streaming-observer support).

    Same validation + fp32 upcast as :func:`compute_metrics`, but returns
    per-row CPU tensors (``row_nll``, ``hit1``/``hit3``/``hit5``, ``conf``)
    instead of pooled means, so an observer can accumulate exact pooled
    means, standard errors, and pooled ECE over shards far larger than one
    batch without ever pooling full logits (rows x 6792 fp32 per batch
    would OOM a CPU observer). Row means of these primitives reproduce the
    :func:`compute_metrics` headline values exactly (pinned by test); pooled
    ECE still needs the frozen-bin pass over pooled ``conf``/``hit1``.
    """
    _, masked_logits, log_prob, targets_long = _validate_metric_inputs(logits, targets, legal_mask)
    batch_idx = torch.arange(targets_long.shape[0], device=targets_long.device)
    row_nll = -log_prob[batch_idx, targets_long]
    probs = F.softmax(masked_logits, dim=-1)
    conf, _ = probs.max(dim=1)
    top5 = torch.topk(masked_logits, k=min(5, masked_logits.shape[1]), dim=1).indices
    expanded = targets_long.unsqueeze(1)
    hit1 = (top5[:, :1] == expanded).any(dim=1).float()
    hit3 = (top5[:, : min(3, top5.shape[1])] == expanded).any(dim=1).float()
    hit5 = (top5 == expanded).any(dim=1).float()
    return {
        "row_nll": row_nll.detach().to("cpu"),
        "hit1": hit1.detach().to("cpu"),
        "hit3": hit3.detach().to("cpu"),
        "hit5": hit5.detach().to("cpu"),
        "conf": conf.detach().to("cpu"),
    }


def _ece_from_confidence(confidences: torch.Tensor, accuracies: torch.Tensor) -> float:
    """Frozen 10-bin ECE over confidence (max softmax prob)."""
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
    return ece


def _support_min_max(predictions: torch.Tensor) -> tuple[int, int]:
    """Min/max count of predicted class among legal (support)."""
    pred_counts: dict[int, int] = {}
    # Note: predictions already legal due to masking
    pred_list: Any = predictions.tolist()
    for p_any in pred_list:
        p: int = int(p_any)
        pred_counts[p] = pred_counts.get(p, 0) + 1
    if len(pred_counts) > 0:
        return min(pred_counts.values()), max(pred_counts.values())
    return 0, 0


def _per_type_breakdown(
    logits_fp32: torch.Tensor,
    masked_logits: torch.Tensor,
    log_prob: torch.Tensor,
    targets_long: torch.Tensor,
    legal_mask: torch.Tensor,
    action_kinds: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Per-action-type ``{n, nll, top1, top3, ece, recall, low_support}``.

    ``recall`` is the within-kind exact-match rate: kinds label the target
    action, so per-kind true-positive rate coincides with per-kind top-1
    accuracy by construction (same value, both keys emitted for eval
    consumers).  ``low_support`` is ``1.0`` when ``n < _PER_TYPE_MIN_N``
    (informational only, report path; never affects loss).
    """
    kinds = list(action_kinds)
    if len(kinds) != logits_fp32.shape[0]:
        raise ContractError(f"action_kinds length {len(kinds)} != batch {logits_fp32.shape[0]}")
    for kind in kinds:
        if not isinstance(kind, str) or kind == "":
            raise ContractError(f"action_kinds entries must be non-empty strings, got {kind!r}")
    probs = F.softmax(masked_logits, dim=-1)
    confidences, predictions = probs.max(dim=1)
    accuracies = (predictions == targets_long).float()
    out: dict[str, dict[str, float]] = {}
    for kind in sorted(set(kinds)):
        idx = [i for i, k in enumerate(kinds) if k == kind]
        sel = torch.tensor(idx, dtype=torch.long, device=logits_fp32.device)
        n = len(idx)
        nll = float(-log_prob[sel, targets_long[sel]].mean().item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for metric; alternative loses metric. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        top1 = masked_topk_accuracy(logits_fp32[sel], targets_long[sel], legal_mask[sel], k=1)
        k3 = masked_topk_accuracy(
            logits_fp32[sel], targets_long[sel], legal_mask[sel], k=min(3, logits_fp32.shape[1])
        )
        ece = _ece_from_confidence(confidences[sel], accuracies[sel])
        recall = top1  # by construction: kinds label the target, so kind recall == kind top1
        low_support = 1.0 if n < _PER_TYPE_MIN_N else 0.0
        out[kind] = {
            "n": float(n),
            "nll": nll,
            "top1": top1,
            "top3": k3,
            "ece": ece,
            "recall": recall,
            "low_support": low_support,
        }
    return out


def compute_per_type_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal_mask: torch.Tensor,
    action_kinds: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Per-action-type NLL/top-k/ECE/recall over the legal subspace.

    Args:
        logits/targets/legal_mask: same contract as :func:`compute_metrics`.
        action_kinds: one non-empty kind string per row (e.g. ``"discard"``,
            ``"ron"``); length MUST equal the batch size.

    Returns:
        ``{kind: {"n", "nll", "top1", "top3", "ece", "recall",
        "low_support"}}`` with kinds in sorted order.  ECE uses the frozen
        10-bin grid.  ``recall`` equals ``top1`` by construction (kinds label
        the target); ``low_support`` is ``1.0`` when ``n < 30``
        (informational, report-only).  Deterministic.
    """
    logits_fp32, masked_logits, log_prob, targets_long = _validate_metric_inputs(
        logits, targets, legal_mask
    )
    return _per_type_breakdown(
        logits_fp32, masked_logits, log_prob, targets_long, legal_mask, action_kinds
    )


def fit_temperature_scaling(
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal_mask: torch.Tensor,
) -> dict[str, float]:
    """Fit post-hoc single-temperature scaling on the GIVEN logits only.

    Guo et al. 2017 (single-T Platt scaling): fit ``T > 0`` to minimize
    masked NLL of ``logits / T`` via LBFGS on ``log T`` (init ``T = 1``),
    then report pre/post NLL + frozen-10-bin ECE.  Scaling by a positive
    temperature preserves argmax (accuracy-preserving up to ties).

    Callers MUST fit on validation logits only — never on train.  Training
    is untouched (zero train-time change).  Deterministic; a monotone
    guard falls back to ``T = 1`` when the fit cannot improve NLL.
    """
    logits_fp32, masked_logits, log_prob, targets_long = _validate_metric_inputs(
        logits, targets, legal_mask
    )
    fit_idx = torch.arange(targets_long.shape[0], device=targets_long.device)
    nll_before = float(-log_prob[fit_idx, targets_long].mean().item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for metric; alternative loses metric. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
    probs = F.softmax(masked_logits, dim=-1)
    confidences, predictions = probs.max(dim=1)
    accuracies = (predictions == targets_long).float()
    ece_before = _ece_from_confidence(confidences, accuracies)
    base = logits_fp32.detach()
    tgt = targets_long.detach()
    lmask = legal_mask.detach().clone() if legal_mask.requires_grad else legal_mask
    log_temp = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temp], lr=0.25, max_iter=50, tolerance_grad=1e-7, tolerance_change=1e-9
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scaled = base / torch.exp(log_temp)
        masked = scaled.masked_fill(~lmask, _MASKED_LOGIT_NEG)
        lp = F.log_softmax(masked, dim=-1)
        loss = -lp[fit_idx, tgt].mean()
        _ = loss.backward()  # intentionally discarded: backward populates grads in place
        return loss

    try:
        # intentionally discarded: LBFGS step returns loss
        _: torch.Tensor | None = optimizer.step(closure)
        fitted = float(
            torch.exp(log_temp.detach()).clamp(min=_TEMPERATURE_MIN, max=_TEMPERATURE_MAX).item()  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for calibration scalar; alternative loses metric. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        )
    except Exception:
        fitted = 1.0
    cal_logits = logits_fp32 / fitted
    cal_masked = cal_logits.masked_fill(~legal_mask, _MASKED_LOGIT_NEG)
    cal_log_prob = F.log_softmax(cal_masked, dim=-1)
    nll_after = float(-cal_log_prob[fit_idx, targets_long].mean().item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for metric; alternative loses metric. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
    cal_probs = F.softmax(cal_masked, dim=-1)
    cal_conf, cal_pred = cal_probs.max(dim=1)
    cal_acc = (cal_pred == targets_long).float()
    ece_after = _ece_from_confidence(cal_conf, cal_acc)
    if not math.isfinite(fitted) or nll_after > nll_before + 1e-6:
        fitted, nll_after, ece_after = 1.0, nll_before, ece_before
    return {
        "temperature": fitted,
        "nll_before": nll_before,
        "nll_after": nll_after,
        "ece_before": ece_before,
        "ece_after": ece_after,
    }


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal_mask: torch.Tensor,
    action_kinds: Sequence[str] | None = None,
) -> dict[str, float]:
    """Report masked NLL, top-k, calibration, support/confusion, strata, legal-uniform.

    All metrics are deterministic and computed on the legal subspace only.

    Args:
        action_kinds: optional one kind string per row.  When given, the
            result also carries flattened per-type keys
            ``per_type/<kind>/{n,nll,top1,top3,ece,recall,low_support}``
            and ``strata`` counts the kinds present; when ``None``
            (default) the checklist-compat constants (``strata``/``confusion``
            ``0.0``, never bound downstream) are kept.  ``recall`` equals
            ``top1`` by construction
            (kinds label the target); ``low_support`` flags ``n < 30``

    Returns dict with keys:
      masked_nll, top1, top3, top5, calibration_ece, legal_uniform_nll,
      legal_uniform_gap, support_min, support_max, confusion, strata
      (+ ``per_type/...`` when ``action_kinds`` is given)
    """
    logits_fp32, masked_logits, log_prob, targets_long = _validate_metric_inputs(
        logits, targets, legal_mask
    )
    batch_idx = torch.arange(targets_long.shape[0], device=targets_long.device)
    nll = -log_prob[batch_idx, targets_long].mean().item()  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for metric; alternative loses metric. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html

    # Legal-uniform baseline: uniform over legal actions (requires target legality, validated above)
    legal_counts = legal_mask.sum(dim=1).float()
    uniform_nll = torch.log(legal_counts).mean().item()  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for metric; alternative loses metric. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
    if not math.isfinite(float(uniform_nll)):
        raise ContractError(f"legal_uniform_nll must be finite, got {uniform_nll!r}")

    # Top-k (fp32, same upcast logits)
    top1 = masked_topk_accuracy(logits_fp32, targets_long, legal_mask, k=1)
    k3 = masked_topk_accuracy(logits_fp32, targets_long, legal_mask, k=min(3, logits_fp32.shape[1]))
    k5 = masked_topk_accuracy(logits_fp32, targets_long, legal_mask, k=min(5, logits_fp32.shape[1]))

    # Calibration (ECE) with frozen _ECE_NUM_BINS bins over confidence (max softmax prob)
    probs = F.softmax(masked_logits, dim=-1)
    confidences, predictions = probs.max(dim=1)
    accuracies = (predictions == targets_long).float()
    ece = _ece_from_confidence(confidences, accuracies)

    # Support: min/max count of predicted class among legal
    sup_min, sup_max = _support_min_max(predictions)

    result: dict[str, float] = {
        "masked_nll": float(nll),
        "top1": top1,
        "top3": k3,
        "top5": k5,
        "calibration_ece": ece,
        "legal_uniform_nll": float(uniform_nll),
        "legal_uniform_gap": float(uniform_nll - nll),  # positive means better than uniform
        "support_min": float(sup_min),
        "support_max": float(sup_max),
        # Checklist-compat constants: populated below when action_kinds is given
        # (strata = kinds present); confusion stays 0.0, never bound downstream.
        "strata": 0.0,
        "confusion": 0.0,
    }
    if action_kinds is not None:
        per_type = _per_type_breakdown(
            logits_fp32, masked_logits, log_prob, targets_long, legal_mask, action_kinds
        )
        for kind in sorted(per_type):
            metrics = per_type[kind]
            result[f"per_type/{kind}/n"] = metrics["n"]
            result[f"per_type/{kind}/nll"] = metrics["nll"]
            result[f"per_type/{kind}/top1"] = metrics["top1"]
            result[f"per_type/{kind}/top3"] = metrics["top3"]
            result[f"per_type/{kind}/ece"] = metrics["ece"]
            result[f"per_type/{kind}/recall"] = metrics["recall"]
            result[f"per_type/{kind}/low_support"] = metrics["low_support"]
        result["strata"] = float(len(per_type))
    return result
