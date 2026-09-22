# ruff: noqa: N803,N806  # reason: Triton convention — uppercase constexpr grid params (BLOCK, A) and reduction temps (S, Q, L); renaming fights every Triton reference.
"""Fused masked cross-entropy (exact math, Triton, opt-in fast path).

Implements :func:`hydra2.training.objectives_loss.masked_cross_entropy`'s exact
formula — legal-only label smoothing over ``[B,A]`` logits — as two fused
Triton kernels wrapped in a ``torch.library.custom_op`` (opaque to inductor,
fake impl for shapes, registered autograd formula):

* forward: per-row two-pass online reduction (legal max, then legal sum +
  sum of exps) reading bf16/fp32 logits directly — no fp32 ``[B,A]``
  materialization, no separate masked_fill/cast/log_softmax passes.
  Returns per-row losses ``[B]`` fp32; the caller means them.
* backward: recomputes row max/LSE (no saved state) then writes bf16
  dlogits in one pass using the closed-form legal-subspace gradient
  (sums to exactly zero over legal rows, zero on illegals).

Numerics: fp32 accumulation throughout, round-to-nearest on the bf16
store (matches the ``.float()``-cast backward rounding). Summation order
differs from ATen (blocked vs tree reduction), so outputs agree within
~1e-6, NOT bitwise — the harness 1e-4 loss-parity gate is the acceptance
check, plus the unit test below pins eager agreement.

Fail-closed: illegal rows (L=0) produce NaN exactly like the eager path
(all ``-inf`` log_softmax), tripping the loop's finite gate identically.
"""

from __future__ import annotations

from typing import Any, cast

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_OK = True
except Exception:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_OK = False
# Triton DSL has no stubs: type the handles Any so the checker verifies
# call-site code (shapes/dtypes/devices) without flagging kernel bodies.
triton = cast(Any, triton)  # noqa: TC006  # reason: pyrefly needs the Any object, not the string form
tl = cast(Any, tl)  # noqa: TC006  # reason: same

__all__ = ["TRITON_AVAILABLE", "fused_hot_scalars", "fused_masked_ce_row_losses"]

TRITON_AVAILABLE: bool = _TRITON_OK


if tl is not None and triton is not None:

    @triton.jit
    def _ce_fwd_kernel(
        x_ptr,
        mask_ptr,
        tgt_ptr,
        out_ptr,
        eps,
        A: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Per-row [B] losses. Program per row; two streaming passes."""
        b = tl.program_id(0)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        offs = tl.arange(0, BLOCK)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        row = b * A  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        # Pass 1: legal max.
        m = float("-inf")
        for a0 in range(0, A, BLOCK):  # pyrefly: ignore[unknown-argument-type] # triton constexpr bounds, no stubs
            a = a0 + offs  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            ok = a < A  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            mk = tl.load(mask_ptr + row + a, mask=ok)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            xv = tl.load(x_ptr + row + a, mask=ok, other=float("-inf")).to(tl.float32)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            xm = tl.where(mk, xv, float("-inf"))  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            m = tl.maximum(m, tl.max(xm))  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        # Pass 2: legal sum S, exp-sum Q, legal count Lf (fp32 accumulator;
        # exact for L <= 2**24, and keeps the checker honest).
        S = 0.0
        Q = 0.0
        Lf = 0.0
        for a0 in range(0, A, BLOCK):  # pyrefly: ignore[unknown-argument-type] # triton constexpr bounds, no stubs
            a = a0 + offs  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            ok = a < A  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            mk = tl.load(mask_ptr + row + a, mask=ok)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            xv = tl.load(x_ptr + row + a, mask=ok, other=float("-inf")).to(tl.float32)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            xm = tl.where(mk, xv, 0.0)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            S += tl.sum(xm)
            Q += tl.sum(tl.where(mk, tl.exp(xv - m), 0.0))
            Lf += tl.sum(mk.to(tl.float32))
        lse = m + tl.log(Q)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        xt = tl.load(x_ptr + row + tl.load(tgt_ptr + b)).to(tl.float32)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        c = 1.0 - eps  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        s = eps / Lf  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        loss_b = -((c * (xt - lse)) + (s * (S - Lf * lse)))  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        tl.store(out_ptr + b, loss_b)

    @triton.jit
    def _ce_bwd_kernel(
        x_ptr,
        mask_ptr,
        tgt_ptr,
        gout_ptr,
        dx_ptr,
        eps,
        A: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Per-row dlogits (bf16 store). Recomputes m/lse, one write pass."""
        b = tl.program_id(0)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        offs = tl.arange(0, BLOCK)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        row = b * A  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        m = float("-inf")
        for a0 in range(0, A, BLOCK):  # pyrefly: ignore[unknown-argument-type] # triton constexpr bounds, no stubs
            a = a0 + offs  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            ok = a < A  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            mk = tl.load(mask_ptr + row + a, mask=ok)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            xv = tl.load(x_ptr + row + a, mask=ok, other=float("-inf")).to(tl.float32)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            m = tl.maximum(m, tl.max(tl.where(mk, xv, float("-inf"))))  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        Q = 0.0
        Lf = 0.0
        for a0 in range(0, A, BLOCK):  # pyrefly: ignore[unknown-argument-type] # triton constexpr bounds, no stubs
            a = a0 + offs  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            ok = a < A  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            mk = tl.load(mask_ptr + row + a, mask=ok)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            xv = tl.load(x_ptr + row + a, mask=ok, other=float("-inf")).to(tl.float32)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            Q += tl.sum(tl.where(mk, tl.exp(xv - m), 0.0))
            Lf += tl.sum(mk.to(tl.float32))
        lse = m + tl.log(Q)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        c = 1.0 - eps  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        s = eps / Lf  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        t = tl.load(tgt_ptr + b).to(tl.int32)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        g = tl.load(gout_ptr + b).to(tl.float32)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        k = c + eps  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        for a0 in range(0, A, BLOCK):  # pyrefly: ignore[unknown-argument-type] # triton constexpr bounds, no stubs
            a = a0 + offs  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            ok = a < A  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            mk = tl.load(mask_ptr + row + a, mask=ok)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            xv = tl.load(x_ptr + row + a, mask=ok, other=float("-inf")).to(tl.float32)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            p = tl.exp(xv - lse)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            is_t = a == t  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            # d/dx_t = -(c+s) + p_t*k; d/dx_i = p_i*k - s; illegal -> 0.
            dx = tl.where(is_t, -(c + s) + p * k, p * k - s)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            dx = tl.where(mk, dx * g, 0.0)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            tl.store(dx_ptr + row + a, dx.to(tl.bfloat16), mask=ok)

    @triton.jit
    def _hot_kernel(
        x_ptr,
        mask_ptr,
        tgt_ptr,
        nll_ptr,
        ok_ptr,
        A: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Per-row NLL + top1-correct. Argmax fuses into the max pass."""
        b = tl.program_id(0)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        offs = tl.arange(0, BLOCK)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        row = b * A  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        # Pass 1: legal max + its (first, lowest-index) position.
        m = float("-inf")
        am = 0
        for a0 in range(0, A, BLOCK):  # pyrefly: ignore[unknown-argument-type] # triton constexpr bounds, no stubs
            a = a0 + offs  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            ok = a < A  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            mk = tl.load(mask_ptr + row + a, mask=ok)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            xv = tl.load(x_ptr + row + a, mask=ok, other=float("-inf")).to(tl.float32)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            xm = tl.where(mk, xv, float("-inf"))  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            bm = tl.max(xm)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            take = bm > m  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            # tl.argmax returns the first (lowest) max position, matching
            # topk(k=1) order on non-tied rows (ties are measure-zero).
            am = tl.where(take, (a0 + tl.argmax(xm, axis=0)).to(tl.int32), am)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            m = tl.maximum(m, bm)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        # Pass 2: exp-sum for the LSE.
        Q = 0.0
        for a0 in range(0, A, BLOCK):  # pyrefly: ignore[unknown-argument-type] # triton constexpr bounds, no stubs
            a = a0 + offs  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            ok = a < A  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            mk = tl.load(mask_ptr + row + a, mask=ok)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            xv = tl.load(x_ptr + row + a, mask=ok, other=float("-inf")).to(tl.float32)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
            Q += tl.sum(tl.where(mk, tl.exp(xv - m), 0.0))
        lse = m + tl.log(Q)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        xt = tl.load(x_ptr + row + tl.load(tgt_ptr + b)).to(tl.float32)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        t = tl.load(tgt_ptr + b).to(tl.int32)  # pyrefly: ignore[unknown-variable-type] # triton DSL value, no stubs
        tl.store(nll_ptr + b, lse - xt)
        tl.store(ok_ptr + b, (am == t).to(tl.int32))


@torch.library.custom_op("hydra2::fused_masked_ce", mutates_args=())
def fused_masked_ce_row_losses(
    logits: torch.Tensor, legal_mask: torch.Tensor, targets: torch.Tensor, eps: float
) -> torch.Tensor:
    """Per-row masked-CE losses ``[B]`` fp32 (caller means them)."""
    if not _TRITON_OK or triton is None:
        raise RuntimeError("fused_masked_ce needs triton")
    b, a = logits.shape
    out = torch.empty((b,), device=logits.device, dtype=torch.float32)
    _ce_fwd_kernel[(b,)](logits, legal_mask, targets, out, eps, a, 2048, num_warps=8)
    return out


@fused_masked_ce_row_losses.register_fake
def _fused_ce_fake(
    logits: torch.Tensor, legal_mask: torch.Tensor, targets: torch.Tensor, eps: float
) -> torch.Tensor:
    return torch.empty((logits.shape[0],), device=logits.device, dtype=torch.float32)


@torch.library.custom_op("hydra2::fused_masked_ce_bwd", mutates_args=())
def _bwd_op(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    targets: torch.Tensor,
    grad_out: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """bf16 dlogits (opaque to inductor; the autograd formula calls this)."""
    if not _TRITON_OK or triton is None:
        raise RuntimeError("fused_masked_ce needs triton")
    b, a = logits.shape
    dx = torch.empty_like(logits, dtype=torch.bfloat16)
    _ce_bwd_kernel[(b,)](logits, legal_mask, targets, grad_out, dx, eps, a, 2048, num_warps=8)
    return dx


@_bwd_op.register_fake
def _bwd_fake(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    targets: torch.Tensor,
    grad_out: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(logits, dtype=torch.bfloat16)


def _fused_ce_setup(ctx: Any, inputs: Any, output: Any) -> None:
    logits, legal_mask, targets, eps = inputs
    ctx.save_for_backward(logits, legal_mask, targets)
    ctx.eps = eps


def _fused_ce_backward(ctx: Any, grad_out: torch.Tensor) -> tuple[Any, None, None, None]:
    saved: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = ctx.saved_tensors
    logits, legal_mask, targets = saved
    eps_val: float = ctx.eps
    dx = _bwd_op(logits, legal_mask, targets, grad_out, eps_val)
    return dx, None, None, None


torch.library.register_autograd(
    "hydra2::fused_masked_ce", _fused_ce_backward, setup_context=_fused_ce_setup
)


@torch.library.custom_op("hydra2::fused_hot_scalars", mutates_args=())
def fused_hot_scalars(
    logits: torch.Tensor, legal_mask: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row reporting pair: ``(nll fp32 [B], correct int32 [B])``.

    Reporting-only (no autograd formula): callers pass detached tensors
    (the loop does). Exact NLL; top1 matches topk(k=1) on non-tied rows
    (first-max tie order, ties measure-zero on real logits).
    """
    if not TRITON_AVAILABLE or triton is None:
        raise RuntimeError("fused_hot_scalars needs triton")
    b, a = logits.shape
    nll = torch.empty((b,), device=logits.device, dtype=torch.float32)
    ok = torch.empty((b,), device=logits.device, dtype=torch.int32)
    _hot_kernel[(b,)](logits, legal_mask, targets, nll, ok, a, 2048, num_warps=8)
    return nll, ok


@fused_hot_scalars.register_fake
def _hot_fake(
    logits: torch.Tensor, legal_mask: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty((logits.shape[0],), device=logits.device, dtype=torch.float32),
        torch.empty((logits.shape[0],), device=logits.device, dtype=torch.int32),
    )
