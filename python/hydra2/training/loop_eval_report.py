"""Supervised-loop eval report: frozen-model metrics (torch only).

Split from ``loop_checkpoint`` (LOC gate): the frozen-model
``evaluate_report`` (mean-of-batch-means + standard errors + pooled
per-type scorecards + post-hoc temperature) lives here so the
checkpoint-persistence module keeps sampler, publish, resume, and
selection. Re-exported by ``loop_checkpoint`` so the mixin body and all
import sites keep working unchanged. Logging-only: never touches train
state, gradients, or checkpoints.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from hydra2.contracts.common import ContractError as ContractError
from hydra2.training.loop_batch import (
    _batch_action_kinds as _batch_action_kinds,
)
from hydra2.training.loop_batch import (
    _model_forward as _model_forward,
)
from hydra2.training.loop_batch import (
    _move_batch_to_device as _move_batch_to_device,
)
from hydra2.training.loop_batch import (
    _validate_batch_no_privileged as _validate_batch_no_privileged,
)
from hydra2.training.objectives_metrics import (
    compute_metrics as compute_metrics,
)
from hydra2.training.objectives_metrics import (
    compute_per_type_metrics as compute_per_type_metrics,
)
from hydra2.training.objectives_metrics import (
    fit_temperature_scaling as fit_temperature_scaling,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

__all__ = ["evaluate_report"]


def _mean_se(values: list[float]) -> tuple[float, float]:
    """Mean + standard error (sample std, ddof=1; 0.0 for a single batch)."""
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var / len(values))


def evaluate_report(
    self: Any,
    eval_batches: list[dict[str, Any]] | Any,
    *,
    weights: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute report over eval batches (no grad) with the frozen model.

    Returns dict with keys: ``masked_nll`` (+``masked_nll_se`` batch-mean
    standard error), ``top1``/``top3``/``top5`` (+``top1_se``),
    ``num_eval_batches``/``num_eval_rows``,
    ``calibration_ece``, ``support_min``/``max``,
    ``legal_uniform_nll``/``legal_uniform_gap``,
    ``legal_uniform_comparison`` (checklist alias),
    ``strata`` (kinds present when batches carry ``"_action_kinds"``)
    and ``confusion`` (legacy ``0.0``), plus flattened
    ``per_type/<kind>/{n,nll,top1,top3,ece,recall,low_support}`` when kinds
    are complete (``recall`` equals ``top1`` by construction;
    ``low_support`` flags ``n<30``, report-only), first-class
    ``discard_nll``/``discard_n``/``discard_top1`` (the real game: most
    rows, hardest kind; absent when kinds are incomplete),
    and post-hoc ``temperature``/``calibrated_nll``/``calibrated_ece``
    fit on the pooled eval rows (validation-only, never training).
    """
    self.model.eval()
    total_nll = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
    total_top5 = 0.0
    total_ece = 0.0
    total_uniform_nll = 0.0
    total_uniform_gap = 0.0
    n = 0
    # Pooled eval tensors for per-type scorecards + temperature (Wave 3-B
    # logging): overall means below stay mean-of-batch-means for
    # checklist comparability; per-type/temperature pool rows.
    pooled_logits: list[torch.Tensor] = []
    pooled_targets: list[torch.Tensor] = []
    pooled_masks: list[torch.Tensor] = []
    pooled_kinds: list[str] = []
    kinds_complete = True
    # Batch-mean lists for headline standard errors (mean-of-batch-means
    # SE: batch means are iid draws under the fixed eval set, so
    # std/sqrt(n) is the honest error bar on the reported mean).
    batch_nlls: list[float] = []
    batch_top1s: list[float] = []
    total_rows = 0

    # eval_batches may be iterable of batch dicts or a dataset with iter_batches
    if hasattr(eval_batches, "iter_batches"):
        _iter_batches: Callable[..., Iterable[dict[str, Any]]] = eval_batches.iter_batches
        batches = list(_iter_batches(4, max_batches=5))
    elif isinstance(eval_batches, list):
        batches = eval_batches
    else:
        batches = list(eval_batches)

    with torch.no_grad():
        for raw in batches:
            _validate_batch_no_privileged(raw)
            batch = _move_batch_to_device(raw, self.device)
            # AMP: forward runs under bf16 autocast when enabled; metrics stay fp32.
            with self._forward_autocast():
                out = _model_forward(self.model, batch)
            eval_logits: torch.Tensor = out["policy_logits"]
            eval_targets: torch.Tensor = batch["chosen_action_id"]
            eval_mask: torch.Tensor = batch["legal_mask"]
            metrics = compute_metrics(eval_logits, eval_targets, eval_mask)
            total_nll += metrics["masked_nll"]
            total_top1 += metrics["top1"]
            total_top3 += metrics["top3"]
            total_top5 += metrics["top5"]
            total_ece += metrics["calibration_ece"]
            total_uniform_nll += metrics["legal_uniform_nll"]
            total_uniform_gap += metrics["legal_uniform_gap"]
            batch_nlls.append(metrics["masked_nll"])
            batch_top1s.append(metrics["top1"])
            total_rows += int(eval_targets.shape[0])
            n += 1
            pooled_logits.append(eval_logits.detach().to("cpu"))
            pooled_targets.append(eval_targets.detach().to("cpu"))
            pooled_masks.append(eval_mask.detach().to("cpu"))
            batch_kinds = _batch_action_kinds(batch, eval_targets)
            if batch_kinds is None:
                kinds_complete = False
            else:
                pooled_kinds.extend(batch_kinds)

    if n == 0:
        raise ContractError("evaluate_report requires at least one eval batch")

    _, nll_se = _mean_se(batch_nlls)
    _, top1_se = _mean_se(batch_top1s)
    report: dict[str, Any] = {
        "masked_nll": total_nll / n,
        "top1": total_top1 / n,
        "top3": total_top3 / n,
        "top5": total_top5 / n,
        "calibration_ece": total_ece / n,
        "legal_uniform_nll": total_uniform_nll / n,
        "legal_uniform_gap": total_uniform_gap / n,
        "support_min": 0.0,
        "support_max": 0.0,
        "confusion": 0.0,
        "strata": 0.0,
        "legal_uniform_comparison": total_nll / n,  # alias for checklist
        "num_eval_batches": float(n),
        "num_eval_rows": float(total_rows),
        "masked_nll_se": float(nll_se),
        "top1_se": float(top1_se),
    }
    # Pooled per-type scorecards + post-hoc temperature (logging only;
    # failures degrade to legacy keys, never fail the report).
    try:
        flat_logits = torch.cat(pooled_logits, dim=0)
        flat_targets = torch.cat(pooled_targets, dim=0)
        flat_masks = torch.cat(pooled_masks, dim=0)
        flat_masked = flat_logits.to(torch.float32).masked_fill(~flat_masks, float("-inf"))
        flat_probs = torch.softmax(flat_masked, dim=-1)
        _, flat_pred = flat_probs.max(dim=1)
        counts: dict[int, int] = {}
        pred_list: list[int] = flat_pred.tolist()
        for _p in pred_list:
            _pi = _p
            counts[_pi] = counts.get(_pi, 0) + 1
        if len(counts) > 0:
            report["support_min"] = float(min(counts.values()))
            report["support_max"] = float(max(counts.values()))
        _kinds_ok = kinds_complete and len(pooled_kinds) == flat_targets.shape[0]
        if _kinds_ok and self.config.log_per_type_metrics:
            per_type = compute_per_type_metrics(flat_logits, flat_targets, flat_masks, pooled_kinds)
            for _kind in sorted(per_type):
                _km = per_type[_kind]
                report[f"per_type/{_kind}/n"] = _km["n"]
                report[f"per_type/{_kind}/nll"] = _km["nll"]
                report[f"per_type/{_kind}/top1"] = _km["top1"]
                report[f"per_type/{_kind}/top3"] = _km["top3"]
                report[f"per_type/{_kind}/ece"] = _km["ece"]
                report[f"per_type/{_kind}/recall"] = _km["recall"]
                report[f"per_type/{_kind}/low_support"] = _km["low_support"]
            report["strata"] = float(len(per_type))
            # Discard-primary surface: discards are most decisions and the
            # hardest kind, so the headline mean (flattered by trivial
            # chi/pon/ron calls) gets a first-class counterpart here.
            # Absent when kinds are incomplete: sidecar no-ops on absence.
            _disc = per_type.get("discard")
            if isinstance(_disc, dict):
                report["discard_nll"] = float(_disc.get("nll", float("nan")))
                report["discard_n"] = float(_disc.get("n", 0))
                report["discard_top1"] = float(_disc.get("top1", float("nan")))
        if self.config.fit_temperature:
            temp = fit_temperature_scaling(flat_logits, flat_targets, flat_masks)
            report["temperature"] = temp["temperature"]
            report["calibrated_nll"] = temp["nll_after"]
            report["calibrated_ece"] = temp["ece_after"]
        else:
            report["temperature"] = 1.0
            report["calibrated_nll"] = report["masked_nll"]
            report["calibrated_ece"] = report["calibration_ece"]
    except Exception:
        _ = report.setdefault("temperature", 1.0)
        _ = report.setdefault("calibrated_nll", report["masked_nll"])
        _ = report.setdefault("calibrated_ece", report["calibration_ece"])
    self.model.train()
    return report
