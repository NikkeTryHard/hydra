"""Standalone Lightning-Fabric runtime adapter (SPEC 10).

Uses ``lightning_fabric.Fabric`` with ``accelerator=cuda, devices=1`` and
project-owned loop semantics: Fabric provides device placement and the
backward hook only. Fabric's precision setting is mapped from RuntimeSpec;
the training loop itself stays outside this adapter.
"""

from __future__ import annotations

from typing import Any, cast

from hydra2.contracts.common import ContractError
from hydra2.runtime.protocol import (
    PrecisionId,
    RuntimeHandle,
    RuntimeSpec,
    require_device_available,
    runtime_identity,
)

_FabricPrecision = Any  # Lightning's precision literal union; mapped values verified above


_FABRIC_PRECISION: dict[PrecisionId, str] = {
    "fp32": "32-true",
    "fp16_mixed": "16-mixed",
    "bf16_mixed": "bf16-mixed",
}


def fabric_precision(precision: PrecisionId) -> str:
    try:
        return _FABRIC_PRECISION[precision]
    except KeyError:
        raise ContractError(f"unknown precision {precision!r}") from None


class FabricRuntimeAdapter:
    """Standalone fabric 2.6.5 adapter; one process, one CUDA device."""

    def __init__(self) -> None:
        # Lightning must be imported BEFORE torch.compile runs anywhere in
        # the process: Fabric introspects compiled modules during setup and
        # reconstructs their compile arguments. Our SPEC-mandated order is
        # compile-then-setup, so the adapter binds the import at
        # construction, strictly earlier than build_runtime compiles.
        import lightning_fabric  # noqa: F401 - imported for compile-order side effect

        self._fabric: Any = None
        self._fabric_precision: str | None = None

    def _ensure_fabric(self, spec: RuntimeSpec) -> Any:
        import torch
        from lightning_fabric import Fabric

        want = fabric_precision(spec.precision)
        if self._fabric is not None:
            if self._fabric_precision != want:
                raise ContractError(
                    f"adapter already bound to precision {self._fabric_precision!r}, "
                    f"cannot rebind to {want!r}"
                )
            return self._fabric
        accelerator = "cpu" if spec.device == "cpu" else "cuda"
        self._fabric = Fabric(
            accelerator=accelerator,
            devices=1,
            precision=cast("_FabricPrecision", want),
            strategy="auto",
        )
        if accelerator == "cuda":
            # Materialize the device now so failures surface at setup time.
            torch.cuda.set_device(self._fabric.device)
        return self._fabric

    def setup(self, *, model: Any, optimizer: Any, spec: RuntimeSpec) -> RuntimeHandle:
        require_device_available(spec.device)
        fabric: Any = self._ensure_fabric(spec)
        setup_result: tuple[Any, Any] = cast("tuple[Any, Any]", fabric.setup(model, optimizer))
        model_any: Any = cast("Any", setup_result[0])
        optimizer_any: Any = cast("Any", setup_result[1])

        def backward(loss: object) -> None:
            fabric.backward(cast("Any", loss))

        return RuntimeHandle(
            model=cast("Any", model_any),
            optimizer=cast("Any", optimizer_any),
            backward=backward,
            device=cast("Any", fabric.device),
            runtime_identity=runtime_identity(spec),
        )
    def barrier(self) -> None:
        assert self._fabric is not None, "barrier before setup"
        self._fabric.barrier()

    def synchronize(self) -> None:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
