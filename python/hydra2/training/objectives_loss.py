"""Supervised loss: masked behavior cloning + auxiliary heads.

Owns the training-time loss math: fail-closed device gate, grad-norm probe,
masked cross-entropy over legal actions (formula mirrored by
:mod:`hydra2.training.fused_ce`), auxiliary helpers, eager contract checks,
and the supervised loss kernel (L = w_policy * masked_cross_entropy +
w_placement * placement_loss + w_value * value_mse + event/belief head terms,
all weights model-spec supplied) with its eager entry point.
"""

from __future__ import annotations

import contextlib
import math
import os
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812 -- conventional alias per PyTorch docs; lowercase diverges from docs. Evidence: https://docs.pytorch.org/docs/stable/nn.functional.html

from hydra2.contracts.common import ContractError, IllegalActionError

__all__ = [
    "compute_supervised_loss",
    "global_grad_norm_is_finite",
    "masked_cross_entropy",
    "supervised_loss_kernel",
    "validate_supervised_inputs",
]

# Masked logit value — ``-inf`` gives exact zero illegal prob (exp(-inf)=0)
# and exact zero gradient; ``-1e9`` also underflows to 0 in fp32/fp64 but
# ``-inf`` is mathematically exact and matches ``eval/baseline.py`` and
# ``models/model.py`` (masked_policy).  When all logits are masked (all-false
# legal row) softmax would be NaN for *both* ``-inf`` and ``-1e9`` (sum 0);
# that case is a hard ``ContractError`` (nonterminal all-false legal row is
# a hard error: mask before softmax/loss/argmax with illegal probability
# exactly zero) before masking, so
# finiteness is preserved.  Single source avoids magic ``-1.0e9`` repeats.
_MASKED_LOGIT_NEG: float = float("-inf")
# Pinned default for legal-only label smoothing (masked-LS: soft mass
# spreads over legal actions only; naive full-vocab LS leaks mass to
# illegals).  Code default stays ``0.0`` (disabled, byte-identical resume);
# the r1 recipe overlay sets ``0.03``.
LABEL_SMOOTHING_SOTA_DEFAULT: float = 0.03


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
        cuda = reduced.is_cuda and disabled not in ("1", "true", "yes", "on")
    if cuda:
        with contextlib.suppress(Exception):
            torch._assert_async(reduced)
            return
    if bool(reduced.item()) is False:  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for fail-closed gate; alternative loses validation. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
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
        grads: list[torch.Tensor] = [p.grad for p in model.parameters() if p.grad is not None]
        if len(grads) == 0:
            return True, 0.0
        total: torch.Tensor = torch.nn.utils.get_total_norm(grads, norm_type=2.0)
        norm = float(total.item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # intentional host sync for grad-norm flag; alternative loses probe. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
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
    # Fused-CE fast path (exact masked-CE formula via the lazily-imported
    # Triton custom op; CUDA + triton + bf16/fp32 dense-logits only,
    # trace-safe gates, zero syncs; CPU-only import stays zero-cost):
    # The eager fallback below is unchanged for all other cases.
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
    """Nonterminal check: every row needs at least one legal action (mask
    before softmax/loss/argmax; illegal probability exactly zero after
    normalization)."""
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
        _lo = int(pl_target.min().item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # failure-path detail only; alternative loses error parity. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        _hi = int(pl_target.max().item())  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # failure-path detail only; alternative loses error parity. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html
        return ContractError(f"placement: per-seat target 0-based in [0,3], got [{_lo},{_hi}]")

    _fail_closed_gate((pl_target >= 0) & (pl_target <= 3), _placement_error)


def _check_total_finite(total: torch.Tensor) -> None:
    """Weighted total must be finite (mid-window poison fail-closed gate)."""

    def _total_error() -> ContractError:
        # Failure path only: value detail may sync, success never does.
        return ContractError(f"total loss non-finite: {total.item()}")  # pyrefly: ignore[pytorch-efficiency-lint-item-call] # failure-path detail only; alternative loses error parity. Evidence: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.item.html

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
    """Check-free supervised-loss math (COMPILED fast path; the compiled loop
    calls validate-then-kernel so the inductor graph holds zero host syncs).

    Callers MUST pre-validate via :func:`validate_supervised_inputs` (the
    loop does) or call :func:`compute_supervised_loss`.  Data-dependent
    checks below are gated on ``not torch.compiler.is_compiling()`` so the
    inductor graph holds zero host syncs; key/shape/dtype checks stay inline
    (trace-safe, constant-folded).  Math is identical to the validating path.
    Supervised objective (every weight/head/target/masking/reduction
    model-spec supplied; zero-weight heads MAY be absent; implicit defaults
    prohibited):

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
    paired_argmax_suboptimality + grp_telescope_error). Pre-per-seat
    ``[B]``-class and ``[B,C]``-distribution paths are byte-identical.
    Value loss is MSE of ``value_vector`` ``[B,4]`` vs ``value_target``
    ``[B,4]`` UtilityVector.values (Lean valueMSE_controls_mean_error).

    Args:
        model_output: must contain ``policy_logits`` ``[B,A]``; optional
            ``placement_logits`` ``[B,4,4]`` per-seat (or pre-per-seat
            ``[B,4]`` / ``[B,num_classes]``), ``value_vector`` ``[B,4]``,
            ``event_logits`` dict.
        batch: must contain ``chosen_action_id`` ``[B]``, ``legal_mask``
            ``[B,A]``; optional auxiliary targets:
            ``placement_target`` ``[B,4]`` per-seat (or pre-per-seat ``[B]``),
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
    # Placement auxiliary — per-seat [B,4,4] vs [B,4] first, pre-per-seat byte-identical.
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

    # Belief auxiliary heads (explicit weights; zero-weight heads absent)
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
    """Supervised loss with eager pre-validation (see kernel docstring;
    L = w_policy * masked_cross_entropy + w_placement * placement_loss +
    w_value * value_mse + event/belief head terms).

    Validates inputs (raising the identical errors the kernel skips under
    compile), then runs :func:`supervised_loss_kernel` eagerly.  Replay,
    tests, and direct callers use this; the compiled training loop calls
    :func:`validate_supervised_inputs` + the kernel + ``_check_total_finite``.
    """
    validate_supervised_inputs(model_output, batch, weights)
    return supervised_loss_kernel(model_output, batch, weights)
