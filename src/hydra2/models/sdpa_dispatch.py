"""SDPA dispatch pinning — perf arm (a), dispatch-only, no math change.

Pins flash -> efficient -> math with explicit priority and restores on
exit (``sdpa_kernel`` guarantee). cuDNN is excluded: nondeterministic per
the SDPA docs and already unavailable under deterministic algorithms.
MATH stays listed so the pin degrades to the same fallback the default
path would take — never an error, never a silent new kernel. No SM
version is hardcoded anywhere: health is probed at runtime with
:func:`backend_health` on real-shaped inputs. Full backend table:
tests/integration/test_attention_dispatch_wp13.py.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = [
    "backend_health",
    "describe_sdpa_runtime",
    "documented_a100_expectation",
    "pinned_sdpa_kernel",
]

#: Explicit dispatch priority: flash for a future mask-None arm, efficient
#: for today's bool-mask call, math as the universal fallback. cuDNN is
#: deliberately absent (see module docstring).
_PINNED_BACKENDS: tuple[SDPBackend, SDPBackend, SDPBackend] = (
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
)


@contextlib.contextmanager
def pinned_sdpa_kernel() -> Generator[None, None, None]:
    """Pin SDPA backend selection to flash -> efficient -> math.

    Exit restores the previous backend flags (``sdpa_kernel`` guarantee).
    Dispatch-only: identical math to the default path whenever the default
    path would have picked one of the three pinned backends.
    """
    with sdpa_kernel(list(_PINNED_BACKENDS), set_priority=True):
        yield


def backend_health(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    *,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> dict[str, bool]:
    """Probe per-backend eligibility for exact-shaped SDPA inputs.

    No warnings (``debug=False``); use the harness table for the record.
    Keys are ``flash`` / ``efficient`` / ``cudnn`` / ``math``.
    """
    params = torch.backends.cuda.SDPAParams(
        query, key, value, attn_mask, dropout_p, is_causal, False
    )
    return {
        "flash": torch.backends.cuda.can_use_flash_attention(params),
        "efficient": torch.backends.cuda.can_use_efficient_attention(params),
        "cudnn": torch.backends.cuda.can_use_cudnn_attention(params),
        "math": True,
    }


def describe_sdpa_runtime() -> dict[str, str | bool | None]:
    """Identify this run's SDPA runtime (arch labels derived, never hardcoded)."""
    cuda = torch.cuda.is_available()
    capability: str | None = None
    device_name: str | None = None
    if cuda:
        major, minor = torch.cuda.get_device_capability()
        capability = f"sm_{major}{minor}"
        device_name = torch.cuda.get_device_name(0)
    return {
        "torch": torch.__version__,
        "cuda_available": cuda,
        "device_name": device_name,
        "capability": capability,
        "flash_enabled": torch.backends.cuda.flash_sdp_enabled() if cuda else None,
        "efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled() if cuda else None,
        "math_enabled": torch.backends.cuda.math_sdp_enabled() if cuda else None,
        "cudnn_enabled": torch.backends.cuda.cudnn_sdp_enabled() if cuda else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def documented_a100_expectation() -> dict[str, str]:
    """Doc-derived A100 (sm80) guesses — UNMEASURED, kept for the table only.

    Unmeasured on A100 hardware: flash healthy on sm80 for fp16/bf16
    mask-None, mem-efficient healthy on sm80, cuDNN opt-in. Replace with
    measured numbers when an A100 run lands; never treat these as evidence.
    """
    return {
        "status": "UNMEASURED doc guess — not evidence",
        "flash": "expected healthy (fp16/bf16, mask-None only)",
        "efficient": "expected healthy (accepts bool mask)",
        "cudnn": "opt-in, nondeterministic per SDPA docs",
        "math": "fallback everywhere",
    }
