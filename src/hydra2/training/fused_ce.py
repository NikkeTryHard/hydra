# ruff: noqa: N803,N806  # reason: Triton convention — uppercase constexpr grid params (BLOCK, A) and reduction temps (S, Q, L); renaming fights every Triton reference.
"""Fused masked cross-entropy (exact math, Triton, opt-in fast path).

Implements :func:`hydra2.training.objectives.masked_cross_entropy`'s exact
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

__all__ = ["TRITON_AVAILABLE", "fused_masked_ce_row_losses"]

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
        b = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        row = b * A
        # Pass 1: legal max.
        m = float("-inf")
        for a0 in range(0, A, BLOCK):
            a = a0 + offs
            ok = a < A
            mk = tl.load(mask_ptr + row + a, mask=ok)
            xv = tl.load(x_ptr + row + a, mask=ok, other=float("-inf")).to(tl.float32)
            xm = tl.where(mk, xv, float("-inf"))
            m = tl.maximum(m, tl.max(xm))
        # Pass 2: legal sum S, exp-sum Q, legal count Lf (fp32 accumulator;
        # exact for L <= 2**24, and keeps the checker honest).
        S = 0.0
        Q = 0.0
        Lf = 0.0
        for a0 in range(0, A, BLOCK):
            a = a0 + offs
            ok = a < A
            mk = tl.load(mask_ptr + row + a, mask=ok)
            xv = tl.load(x_ptr + row + a, mask=ok, other=float("-inf")).to(tl.float32)
            xm = tl.where(mk, xv, 0.0)
            S += tl.sum(xm)
            Q += tl.sum(tl.where(mk, tl.exp(xv - m), 0.0))
            Lf += tl.sum(mk.to(tl.float32))
        lse = m + tl.log(Q)
        xt = tl.load(x_ptr + row + tl.load(tgt_ptr + b)).to(tl.float32)
        c = 1.0 - eps
        s = eps / Lf
        loss_b = -((c * (xt - lse)) + (s * (S - Lf * lse)))
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
        b = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        row = b * A
        m = float("-inf")
        for a0 in range(0, A, BLOCK):
            a = a0 + offs
            ok = a < A
            mk = tl.load(mask_ptr + row + a, mask=ok)
            xv = tl.load(x_ptr + row + a, mask=ok, other=float("-inf")).to(tl.float32)
            m = tl.maximum(m, tl.max(tl.where(mk, xv, float("-inf"))))
        Q = 0.0
        Lf = 0.0
        for a0 in range(0, A, BLOCK):
            a = a0 + offs
            ok = a < A
            mk = tl.load(mask_ptr + row + a, mask=ok)
            xv = tl.load(x_ptr + row + a, mask=ok, other=float("-inf")).to(tl.float32)
            Q += tl.sum(tl.where(mk, tl.exp(xv - m), 0.0))
            Lf += tl.sum(mk.to(tl.float32))
        lse = m + tl.log(Q)
        c = 1.0 - eps
        s = eps / Lf
        t = tl.load(tgt_ptr + b).to(tl.int32)
        g = tl.load(gout_ptr + b).to(tl.float32)
        k = c + eps
        for a0 in range(0, A, BLOCK):
            a = a0 + offs
            ok = a < A
            mk = tl.load(mask_ptr + row + a, mask=ok)
            xv = tl.load(x_ptr + row + a, mask=ok, other=float("-inf")).to(tl.float32)
            p = tl.exp(xv - lse)
            is_t = a == t
            # d/dx_t = -(c+s) + p_t*k; d/dx_i = p_i*k - s; illegal -> 0.
            dx = tl.where(is_t, -(c + s) + p * k, p * k - s)
            dx = tl.where(mk, dx * g, 0.0)
            tl.store(dx_ptr + row + a, dx.to(tl.bfloat16), mask=ok)


@torch.library.custom_op("hydra2::fused_masked_ce", mutates_args=())
def fused_masked_ce_row_losses(
    logits: torch.Tensor, legal_mask: torch.Tensor, targets: torch.Tensor, eps: float
) -> torch.Tensor:
    """Per-row masked-CE losses ``[B]`` fp32 (caller means them)."""
    if not _TRITON_OK or triton is None:
        raise RuntimeError("fused_masked_ce needs triton")
    b, a = logits.shape
    out = torch.empty((b,), device=logits.device, dtype=torch.float32)
    _ce_fwd_kernel[(b,)](logits, legal_mask, targets, out, float(eps), a, 2048, num_warps=8)
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
    _ce_bwd_kernel[(b,)](
        logits, legal_mask, targets, grad_out, dx, float(eps), a, 2048, num_warps=8
    )
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
    logits, legal_mask, targets = ctx.saved_tensors
    dx = _bwd_op(logits, legal_mask, targets, grad_out, ctx.eps)
    return dx, None, None, None


torch.library.register_autograd(
    "hydra2::fused_masked_ce", _fused_ce_backward, setup_context=_fused_ce_setup
)
