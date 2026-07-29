"""Plain-PyTorch runtime adapter (SPEC 10).

Moves the model and optimizer state to the target device and returns the
exact same objects for subsequent operations. Owns no loop, checkpoint,
optimizer policy, or compilation.
"""

from __future__ import annotations

from typing import Any, cast

from hydra2.runtime.protocol import (
    RuntimeHandle,
    RuntimeSpec,
    require_device_available,
    runtime_identity,
)


def _is_accelerator_available() -> bool:
    """Unified accelerator availability — torch.accelerator (2.14) with cuda fallback.

    Evidence:
    - https://docs.pytorch.org/docs/2.14/accelerator/index.html
      unified torch.accelerator entrypoint
    - https://pytorch.org/docs/main/accelerator.html
      is_available() delegates without runtime bring-up (same cost)
    - https://pytorch.org/blog/pytorch-2-14-release-blog/
      2.14 GA announcement (accelerator, CuTeDSL)
    Keep cuda-specific torch.cuda.get_arch_list / set_device /
    get_rng_state_all as-is (no generic accelerator equiv).
    """
    import torch

    if hasattr(torch, "accelerator"):
        try:
            # 2.14-only API; hasattr-guarded above, cuda fallback below.
            return bool(torch.accelerator.is_available())  # type: ignore[attr-defined]
        except Exception:
            pass
    return bool(torch.cuda.is_available())


def _synchronize_accelerator() -> None:
    """Synchronize current accelerator — torch.accelerator (2.14) with cuda fallback."""
    import torch

    if hasattr(torch, "accelerator"):
        try:
            # 2.14-only API; hasattr-guarded above, cuda fallback below.
            torch.accelerator.synchronize()  # type: ignore[attr-defined]
            return
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class PlainPytorchAdapter:
    """Single-process eager PyTorch adapter."""

    def setup(self, *, model: Any, optimizer: Any, spec: RuntimeSpec) -> RuntimeHandle:
        import torch

        require_device_available(spec.device)
        device: Any = cast("Any", torch.device(spec.device))
        model_any: Any = cast("Any", model.to(cast("Any", device)))
        _move_optimizer_state(cast("Any", optimizer), cast("Any", device))

        def backward(loss: Any) -> None:
            cast("Any", loss).backward()

        return RuntimeHandle(
            model=cast("Any", model_any),
            optimizer=cast("Any", optimizer),
            backward=backward,
            device=cast("Any", device),
            runtime_identity=runtime_identity(spec),
        )

    def barrier(self) -> None:
        # Single process: the barrier degenerates to a full synchronization.
        self.synchronize()

    def synchronize(self) -> None:
        if _is_accelerator_available():
            _synchronize_accelerator()


def _move_optimizer_state(optimizer: Any, device: Any) -> None:
    # Portable H2D: use non_blocking when device is CUDA to overlap copy with compute.
    # Evidence: https://docs.pytorch.org/docs/2.13/generated/torch.Tensor.to.html
    #  (`non_blocking` flag) and https://docs.pytorch.org/docs/2.13/notes/cuda.html#pinned-memory
    #  and tutorial https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    #  SDPA/compile idioms keep eager plain adapter simple; non_blocking is safe for pinned.
    state_values: Any = cast("Any", optimizer.state.values())
    # Detect CUDA device portably (handles torch.device("cuda") and str "cuda:0").
    is_cuda = getattr(device, "type", "") == "cuda" or str(device).startswith("cuda")
    for state in state_values:
        state_dict: dict[Any, Any] = cast("dict[Any, Any]", state)
        for key, value in list(cast("list[tuple[Any, Any]]", state_dict.items())):
            value_any: Any = cast("Any", value)
            if hasattr(value_any, "to"):
                moved = cast("Any", value_any).to(
                    cast("Any", device), non_blocking=is_cuda
                )
                state_dict[key] = cast("Any", moved)
