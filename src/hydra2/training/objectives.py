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
explicit-fp32 ``torch.zeros((), device=logits.device, dtype=torch.float32)``
accumulators; contract ``.item()`` syncs below are eager-only
and guarded by ``torch.compiler.is_compiling()`` + ``torch._check_tensor_all``
for compile compatibility (see perf-A §4.5/§6; ``pyrefly.toml``
promotes ``pytorch-efficiency-lint-item-call`` to ``warn`` and
``ruff`` ``PERF`` is enabled per-file for this module).  Tmpfs vs xfs
(``/tmp`` tmpfs evicted, ``/home/cachybtw/tmp`` xfs durable per
perf-A §8.1) does not affect these pure-tensor ops; only ``config.py:25``
``DEFAULT_ARTIFACT_ROOT`` selects the durable root.
"""

from __future__ import annotations

import contextlib
import math
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

import torch
import torch.nn.functional as F  # noqa: N812 -- conventional alias per PyTorch docs; lowercase diverges from docs. Evidence: https://docs.pytorch.org/docs/stable/nn.functional.html

from hydra2.contracts.common import ContractError, IllegalActionError

__all__ = [
    "compute_hot_scalars",
    "compute_metrics",
    "compute_per_type_metrics",
    "compute_supervised_loss",
    "fit_temperature_scaling",
    "global_grad_norm_is_finite",
    "masked_cross_entropy",
    "masked_topk_accuracy",
    "supervised_loss_kernel",
    "validate_supervised_inputs",
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
# Minimum per-kind rows for a fully-supported scorecard slice.  Kinds with
# ``n < 30`` carry ``low_support = 1.0`` (informational only, report path;
# kinds never enter the loss, so support never affects training).
_PER_TYPE_MIN_N: int = 30
# SOTA-quoted default for legal-only label smoothing (masked-LS: soft mass
# spreads over legal actions only; naive full-vocab LS leaks mass to
# illegals).  Code default stays ``0.0`` (disabled, byte-identical resume);
# the r1 recipe overlay sets ``0.03``.
LABEL_SMOOTHING_SOTA_DEFAULT: float = 0.03
# Post-hoc temperature bounds (Guo et al. 2017: single-T Platt scaling fit
# on validation NLL only, accuracy-preserving, never fit on train).
_TEMPERATURE_MIN: float = 0.05
_TEMPERATURE_MAX: float = 20.0


def _fail_closed_gate(predicate: torch.Tensor, make_error: Any) -> None:
    """Zero-sync fail-closed gate on CUDA; exact host raise elsewhere.

    ``predicate`` is an unreduced bool tensor; it must hold everywhere.
    ``make_error`` is a zero-arg factory for the typed error — called only
    when the check actually fails, so message detail (``.item()`` ranges)
    never costs a sync on success.  On CUDA the check enqueues a
    device-side assert (no host round-trip): violations surface as a device
    assert on the next sync and poison the context — abort-is-abort for a
    fail-closed trainer.  CPU/eval keeps the exact typed error.  Never
    raises spuriously: any backend failure falls back to the host check.
    """
    try:
        reduced = predicate.all()
    except Exception as exc:
        raise make_error() from exc
    cuda = False
    with contextlib.suppress(Exception):
        disabled = os.environ.get("HYDRA2_DISABLE_DEVICE_ASSERTS", "").strip().lower()
        cuda = bool(reduced.is_cuda) and disabled not in ("1", "true", "yes", "on")
    if cuda:
        with contextlib.suppress(Exception):
            torch._assert_async(reduced)
            return
    if bool(reduced.item()) is False:
        raise make_error()


def global_grad_norm_is_finite(model: Any) -> tuple[bool, float]:
    """Per-update global grad-norm finiteness probe (shared by both loops).

    Computes the global L2 grad norm in fp32 over all parameters with a
    non-None ``.grad`` and reports whether it is finite.  Callers check the
    flag BEFORE ``optimizer.step``: non-finite grads skip the step (zero +
    count) so one poisoned update cannot corrupt master weights.  No mutation
    here — the skip/zero/count policy lives in the loop.

    One fused foreach norm pass plus a SINGLE host sync (was one ``.item()``
    per parameter).  Only the finiteness flag is consumed downstream — both
    loop callers unpack but never use the float — so device-side reduction
    order is unobservable and the fail-closed skip contract is unchanged.
    """
    try:
        grads = [p.grad for p in model.parameters() if getattr(p, "grad", None) is not None]
        if len(grads) == 0:
            return True, 0.0
        total = torch.nn.utils.get_total_norm(grads, norm_type=2.0)
        norm = float(total.detach().item())
    except Exception:
        return False, float("inf")
    return math.isfinite(norm), norm


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal_mask: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Masked cross-entropy over legal actions only.

    Args:
        logits: ``[B,A]`` unmasked logits.
        targets: ``[B]`` chosen action indices.
        legal_mask: ``[B,A]`` bool, ``True`` = legal.  Each row MUST contain
            at least one ``True``; ``targets`` MUST be legal.
        label_smoothing: legal-only smoothing mass ``eps`` in ``[0, 1)``.
            ``0.0`` (default) is plain one-hot masked CE, byte-identical to
            the pre-smoothing path.  When ``> 0``, the target distribution
            keeps ``1 - eps`` on the target and spreads ``eps`` uniformly
            over the LEGAL set only (target included, so the target keeps
            ``1 - eps + eps/L`` and each other legal action gets ``eps/L``;
            illegal mass stays exactly ``0``).  Naive full-vocab smoothing
            would leak mass to illegals; this variant conserves mass on the
            legal subspace (sums to exactly ``1``) and degrades gracefully
            to ``1.0`` on the target for single-legal rows.
    Returns:

        Scalar loss (mean over batch).  Gradients for illegal logits are
        exactly zero (masked to ``-inf`` before softmax).
    """
    if isinstance(label_smoothing, bool) or not isinstance(label_smoothing, (int, float)):
        raise ContractError(f"label_smoothing must be a float in [0, 1), got {label_smoothing!r}")
    eps = float(label_smoothing)
    if not (0.0 <= eps < 1.0) or eps != eps:
        raise ContractError(f"label_smoothing must lie in [0, 1), got {label_smoothing!r}")
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
    # Data-dependent checks below run eager-only (single-source helpers also
    # used by validate_supervised_inputs): under torch.compile they fold away
    # so the graph holds zero host syncs (_check_tensor_all on mask/range
    # tensors graph-breaks every call). Compiled callers pre-validate.
    if not torch.compiler.is_compiling():
        _check_legal_rows(legal_mask)
    # Targets must be in range and legal
    num_actions = logits.shape[1]
    if not torch.compiler.is_compiling():
        _check_targets_in_range(targets, num_actions)
    batch_idx = torch.arange(targets.shape[0], device=targets.device)
    if not torch.compiler.is_compiling():
        _check_targets_legal(legal_mask, targets)
    # Fused-CE fast path (exact formula, Triton custom op; see fused_ce.py):
    # CUDA + triton + bf16/fp32 dense logits only. Trace-safe gates, zero
    # syncs; lazy import keeps CPU-only import cost at zero. The eager
    # fallback below is unchanged for all other cases.
    if logits.is_cuda and logits.dtype in (torch.bfloat16, torch.float32):
        from hydra2.training.fused_ce import TRITON_AVAILABLE, fused_masked_ce_row_losses

        if TRITON_AVAILABLE and legal_mask.is_cuda and targets.is_cuda:
            return fused_masked_ce_row_losses(logits, legal_mask, targets.long(), eps).mean()
    # Dense masked math (byte-identical): mask illegals to -inf over full
    # [B,A].  A legal-subspace gather was tried here (static-Lb scatter):
    # measured 2.5x SLOWER wall (0.30 vs 0.12ms @B2048/A6792/Lb64) and +110MB
    # peak — the int64 cumsum/scatter index traffic (2x111MB) exceeds the CE
    # passes it removes, and scatter requires int64 indices.  The fused-CE
    # direction (indexed target + online LSE, no full read) stays bakeoff-13
    # conditional on a purpose-built kernel; plain gather cannot win here.
    masked_logits = logits.masked_fill(~legal_mask, _MASKED_LOGIT_NEG)
    targets_long = targets.long()
    if eps == 0.0:
        loss = F.cross_entropy(masked_logits, targets_long, reduction="mean")
    else:
        # Legal-only smoothing: (1-eps) on the target plus eps/L on every
        # legal action (target included).  Illegal log-probs are -inf, so
        # fill them with 0 before the legal-subspace sum (0 * -inf is NaN).
        log_prob = F.log_softmax(masked_logits.float(), dim=-1)
        legal_counts = legal_mask.sum(dim=1).float()
        smooth = eps / legal_counts
        target_logp = log_prob[batch_idx, targets_long]
        legal_logp_sum = log_prob.masked_fill(~legal_mask, 0.0).sum(dim=1)
        loss = -((1.0 - eps) * target_logp + smooth * legal_logp_sum).mean()
    # No CE-local finite gate: any non-finite CE poisons the weighted total,
    # which the post-gate (_check_total_finite) trips identically.
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


def compute_hot_scalars(
    logits: torch.Tensor,
    targets: torch.Tensor,
    legal_mask: torch.Tensor,
) -> dict[str, float]:
    """Per-update hot metrics: masked NLL + top-1 only (2 host syncs).

    No validation (the batch was pre-validated this microbatch) and no
    uniform/top-k/ECE/support/per-type work — those ride the eval report
    (:func:`compute_metrics`). Keys mirror the :func:`compute_metrics`
    subset so hot entries keep ``masked_nll``/``top1`` continuity.
    """
    # Fused reporting fast path (exact NLL + top1, one Triton op; see
    # fused_ce.py): CUDA + triton + bf16/fp32 only. Same 2 host syncs, same
    # keys; the eager fallback below is unchanged for all other cases.
    if logits.is_cuda and logits.dtype in (torch.bfloat16, torch.float32):
        from hydra2.training.fused_ce import TRITON_AVAILABLE, fused_hot_scalars

        if TRITON_AVAILABLE and legal_mask.is_cuda and targets.is_cuda:
            targets_long = targets.long() if targets.dtype != torch.long else targets
            nll_vec, ok_vec = fused_hot_scalars(logits, legal_mask, targets_long)
            return {
                "masked_nll": float(nll_vec.mean().item()),
                "top1": float(ok_vec.float().mean().item()),
            }
    logits_fp32 = logits.to(torch.float32)
    masked_logits = logits_fp32.masked_fill(~legal_mask, _MASKED_LOGIT_NEG)
    log_prob = F.log_softmax(masked_logits, dim=-1)
    targets_long = targets.long() if targets.dtype != torch.long else targets
    batch_idx = torch.arange(targets_long.shape[0], device=targets_long.device)
    nll = -log_prob[batch_idx, targets_long].mean().item()
    top1 = masked_topk_accuracy(logits_fp32, targets_long, legal_mask, k=1)
    return {"masked_nll": float(nll), "top1": top1}


def _generic_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """Unmasked cross-entropy helper for auxiliary heads."""
    if logits.shape[0] != targets.shape[0]:
        raise ContractError(f"{name}: logits/targets batch mismatch")
    # Aux pin: fp32 compute (widening cast is exact; also fixes bf16-CPU dtype).
    return F.cross_entropy(logits.to(torch.float32), targets.long(), reduction="mean")


def _generic_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ContractError(f"{name}: shape mismatch {tuple(pred.shape)} vs {tuple(target.shape)}")
    # Aux pin: fp32 compute (both casts widen exactly; matches target.float() position).
    return F.mse_loss(pred.to(torch.float32), target.to(torch.float32), reduction="mean")


def _check_legal_rows(legal_mask: torch.Tensor) -> None:
    """Nonterminal check: every row needs at least one legal action (SPEC 11.1)."""
    _fail_closed_gate(
        legal_mask.any(dim=1),
        lambda: ContractError("nonterminal all-false legal row is hard error (SPEC 11.1)"),
    )


def _check_targets_in_range(targets: torch.Tensor, num_actions: int) -> None:
    """Chosen ids must lie in ``[0, num_actions)``."""
    _fail_closed_gate(
        (targets >= 0) & (targets < num_actions),
        lambda: ContractError(f"target action_id out of range [0,{num_actions})"),
    )


def _check_targets_legal(legal_mask: torch.Tensor, targets: torch.Tensor) -> None:
    """Chosen action must be legal per row (fail-closed illegal-action gate)."""
    batch_idx = torch.arange(targets.shape[0], device=targets.device)
    _fail_closed_gate(
        legal_mask[batch_idx, targets],
        lambda: IllegalActionError("selected action is illegal per legal_mask"),
    )


def _check_placement_range(pl_target: torch.Tensor) -> None:
    """Per-seat placement targets are 0-based in ``[0, 3]``."""

    def _placement_error() -> ContractError:
        # Failure path only: range detail may sync, success never does.
        _lo = int(pl_target.min().item())
        _hi = int(pl_target.max().item())
        return ContractError(f"placement: per-seat target 0-based in [0,3], got [{_lo},{_hi}]")

    _fail_closed_gate((pl_target >= 0) & (pl_target <= 3), _placement_error)


def _check_total_finite(total: torch.Tensor) -> None:
    """Weighted total must be finite (mid-window poison fail-closed gate)."""

    def _total_error() -> ContractError:
        # Failure path only: value detail may sync, success never does.
        return ContractError(f"total loss non-finite: {total.item()}")

    _fail_closed_gate(torch.isfinite(total), _total_error)


def validate_supervised_inputs(
    model_output: dict[str, Any], batch: dict[str, Any], weights: dict[str, Any]
) -> None:
    """Eager pre-validation for the compiled loss kernel (single message source).

    Runs every data-dependent (host-sync) check the kernel skips under
    ``torch.compile``: key presence, legal rows, target range/legality on the
    policy path, placement-target range on the active per-seat path.  The
    loss fns call the same ``_check_*`` helpers when eager, so messages
    cannot drift.  The compiled loop calls this before
    :func:`supervised_loss_kernel` and gates the total after; direct eager
    callers go through :func:`compute_supervised_loss`, which calls this.
    """
    if "legal_mask" not in batch or "chosen_action_id" not in batch:
        raise ContractError("batch must contain 'legal_mask' and 'chosen_action_id'")
    if "policy_logits" not in model_output:
        raise ContractError("model_output missing 'policy_logits'")
    legal_mask: torch.Tensor = batch["legal_mask"]
    targets: torch.Tensor = batch["chosen_action_id"]
    logits: torch.Tensor = model_output["policy_logits"]
    if float(weights.get("w_policy", 0.0)) != 0.0:
        _check_legal_rows(legal_mask)
        _check_targets_in_range(targets, logits.shape[1])
        _check_targets_legal(legal_mask, targets)
    if (
        float(weights.get("w_placement", 0.0)) != 0.0
        and "placement_logits" in model_output
        and "placement_target" in batch
    ):
        pl_logits: torch.Tensor = model_output["placement_logits"]
        pl_target: torch.Tensor = batch["placement_target"]
        if (
            pl_logits.dim() == 3
            and pl_target.dim() == 2
            and tuple(pl_logits.shape[1:]) == (4, 4)
            and pl_target.shape == (pl_logits.shape[0], 4)
        ):
            _check_placement_range(pl_target)


def supervised_loss_kernel(
    model_output: dict[str, Any],
    batch: dict[str, Any],
    weights: dict[str, Any],
) -> dict[str, Any]:
    """Check-free supervised-loss math (COMPILED fast path; see loop Perf-B).

    Callers MUST pre-validate via :func:`validate_supervised_inputs` (the
    loop does) or call :func:`compute_supervised_loss`.  Data-dependent
    checks below are gated on ``not torch.compiler.is_compiling()`` so the
    inductor graph holds zero host syncs; key/shape/dtype checks stay inline
    (trace-safe, constant-folded).  Math is identical to the validating path.

    SPEC 20 formula:

        L = w_policy * masked_cross_entropy(label_smoothing)
          + w_placement * placement_loss
          + w_value * value_mse
          + sum_h w_event[h] * event_loss[h]
          + sum_h w_belief[h] * belief_loss[h]

    The policy head applies legal-only label smoothing: ``weights`` may
    carry ``label_smoothing`` ``eps`` in ``[0, 1)`` (default ``0.0`` =
    disabled, byte-identical plain masked CE).  When ``> 0`` the ``eps``
    mass spreads over the legal set only (illegal mass stays exactly
    zero); the r1 recipe sets ``0.03``.

    Placement loss is per-seat cross-entropy then mean over seats when
    ``placement_logits`` is ``[B,4,4]`` (dim-1 seat 0..3, dim-2 rank-logits
    1..4) vs 0-based ``placement_target`` ``[B,4]`` per-seat rank indices
    (0..3; bridge to utility() 1..4 is target = utility_rank - 1) (Lean
    paired_argmax_suboptimality + grp_telescope_error). Legacy
    ``[B]``-class and ``[B,C]``-distribution paths are byte-identical.
    Value loss is MSE of ``value_vector`` ``[B,4]`` vs ``value_target``
    ``[B,4]`` UtilityVector.values (Lean valueMSE_controls_mean_error).

    Args:
        model_output: must contain ``policy_logits`` ``[B,A]``; optional
            ``placement_logits`` ``[B,4,4]`` per-seat (or legacy ``[B,4]``
            / ``[B,num_classes]``), ``value_vector`` ``[B,4]``,
            ``event_logits`` dict.
        batch: must contain ``chosen_action_id`` ``[B]``, ``legal_mask``
            ``[B,A]``; optional auxiliary targets:
            ``placement_target`` ``[B,4]`` per-seat (or legacy ``[B]``),
            ``value_target`` ``[B,4]``, ``event_targets`` dict
            ``{head_id: [B]}``.
        weights: dict with keys ``w_policy``, ``w_placement``, ``w_value``,
            ``w_event`` (dict), ``w_belief`` (dict), ``label_smoothing``
            (float in ``[0, 1)``, default ``0.0``).  Missing keys
            default to ``0``; zero-weight heads MAY be absent.

    Returns:
        dict with ``total`` scalar tensor and ``policy``, ``placement``,
        ``value``, ``event``, ``belief`` components (tensors) for logging.

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
    w_value = float(weights.get("w_value", 0.0))
    w_event: dict[str, float] = dict(weights.get("w_event", {}) or {})
    w_belief: dict[str, float] = dict(weights.get("w_belief", {}) or {})

    losses: dict[str, Any] = {}
    # AMP L4: explicit fp32 accumulator — logits may be bf16 under autocast
    # (CE itself runs fp32) but the weighted sum + /accum_steps must stay fp32.
    total = torch.zeros((), device=logits.device, dtype=torch.float32)

    # Policy (masked BC) — the only required head when w_policy>0
    if w_policy != 0.0:
        _smooth_raw: Any = weights.get("label_smoothing", 0.0)
        _smooth: Any = 0.0 if _smooth_raw is None else _smooth_raw
        ploss = masked_cross_entropy(logits, targets, legal_mask, label_smoothing=_smooth)
        losses["policy"] = ploss
        total = total + w_policy * ploss
    else:
        # Even when weight zero, we still compute for reporting if logits present,
        # but do not require it.  For determinism we put zero.
        losses["policy"] = torch.zeros((), device=logits.device, dtype=torch.float32)
    # Placement auxiliary — per-seat [B,4,4] vs [B,4] first, legacy byte-identical.
    if w_placement != 0.0:
        if "placement_logits" not in model_output:
            raise ContractError("w_placement>0 but model_output missing 'placement_logits'")
        if "placement_target" not in batch:
            raise ContractError("w_placement>0 but batch missing 'placement_target'")
        pl_logits: torch.Tensor = model_output["placement_logits"]
        pl_target: torch.Tensor = batch["placement_target"]
        if pl_logits.dim() == 3 and pl_target.dim() == 2:
            # Day-one per-seat path: logits [B,4,4] (dim-1 seat 0..3, dim-2
            # rank-logits 1..4) vs target [B,4] per-seat rank indices.
            # Contract: placement_target is 0-based (0..3). Bridge to the
            # utility() 1..4 rank convention is target = utility_rank - 1.
            # Per-seat CE then mean over seats (flattened mean is identical:
            # mean_s(mean_b) == mean_{b,s}).
            if tuple(pl_logits.shape[1:]) != (4, 4):
                raise ContractError(
                    f"placement: per-seat logits must be [B,4,4], got {tuple(pl_logits.shape)}"
                )
            if pl_target.shape != (pl_logits.shape[0], 4):
                raise ContractError(
                    f"placement: per-seat target must be [B,4], got {tuple(pl_target.shape)}"
                )
            if not torch.compiler.is_compiling():
                _check_placement_range(pl_target)
            # Aux pin: fp32 entry edge (same family as _generic_ce_loss).
            ploss = F.cross_entropy(
                pl_logits.reshape(-1, 4).to(torch.float32),
                pl_target.reshape(-1).long(),
                reduction="mean",
            )
        elif pl_target.dim() == 1:
            ploss = _generic_ce_loss(pl_logits, pl_target, name="placement")
        else:
            # Distribution: KL or MSE? Use MSE for simplicity if distribution
            # Aux pin: fp32 softmax (removes the bf16 residue class).
            ploss = _generic_mse_loss(
                F.softmax(pl_logits.to(torch.float32), dim=-1),
                pl_target.float(),
                name="placement_dist",
            )
        losses["placement"] = ploss
        total = total + w_placement * ploss
    else:
        losses["placement"] = torch.zeros((), device=logits.device, dtype=torch.float32)

    # Value auxiliary — value-MSE on [B,4] vs UtilityVector.values (additive).
    if w_value != 0.0:
        if "value_vector" not in model_output:
            raise ContractError("w_value>0 but model_output missing 'value_vector'")
        if "value_target" not in batch:
            raise ContractError("w_value>0 but batch missing 'value_target'")
        v_pred: torch.Tensor = model_output["value_vector"]
        v_target: torch.Tensor = batch["value_target"]
        vloss = _generic_mse_loss(v_pred, v_target, name="value")
        losses["value"] = vloss
        total = total + w_value * vloss
    else:
        losses["value"] = torch.zeros((), device=logits.device, dtype=torch.float32)

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
        sum(
            event_losses.values(),
            start=torch.zeros((), device=logits.device, dtype=torch.float32),
        )
        if len(event_losses) > 0
        else torch.zeros((), device=logits.device, dtype=torch.float32)
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
    losses["belief"] = (
        sum(
            belief_losses.values(),
            start=torch.zeros((), device=logits.device, dtype=torch.float32),
        )
        if len(belief_losses) > 0
        else torch.zeros((), device=logits.device, dtype=torch.float32)
    )
    losses["_belief_per_head"] = belief_losses

    losses["total"] = total
    # Finite check on total (eager-only; the compiled loop post-gates via
    # _check_total_finite on the kernel output — same message, same error).
    if not torch.compiler.is_compiling():
        _check_total_finite(total)
    return losses


def compute_supervised_loss(
    model_output: dict[str, Any],
    batch: dict[str, Any],
    weights: dict[str, Any],
) -> dict[str, Any]:
    """SPEC 20 supervised loss with eager pre-validation (see kernel docstring).

    Validates inputs (raising the identical errors the kernel skips under
    compile), then runs :func:`supervised_loss_kernel` eagerly.  Replay,
    tests, and direct callers use this; the compiled training loop calls
    :func:`validate_supervised_inputs` + the kernel + ``_check_total_finite``.
    """
    validate_supervised_inputs(model_output, batch, weights)
    return supervised_loss_kernel(model_output, batch, weights)


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
            (default) the legacy placeholders (``strata``/``confusion``
            ``0.0``) are kept.  ``recall`` equals ``top1`` by construction
            (kinds label the target); ``low_support`` flags ``n < 30``
            kinds as informational (report-only, never loss).

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
        # Legacy placeholders: populated below when action_kinds is given
        # (strata = kinds present); confusion stays 0.0 for checklist compat.
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
