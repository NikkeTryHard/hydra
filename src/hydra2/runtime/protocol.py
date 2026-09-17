"""SPEC 10 runtime protocol: RuntimeSpec, RuntimeHandle, RuntimeAdapter,
supported-value validation, and the build order (compile-before-setup).

Neither adapter owns the training loop, checkpoint schema, optimizer policy,
or compilation decisions; :func:`build_runtime` owns the exact call order.
"""

from __future__ import annotations

import contextlib
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

from hydra2.artifacts.digest import of_canonical
from hydra2.contracts.common import ContractError, DigestText

if TYPE_CHECKING:
    from collections.abc import Callable

PrecisionId = Literal["fp32", "fp16_mixed", "bf16_mixed"]
CompileMode = Literal["eager", "default", "max-autotune-no-cudagraphs", "max-autotune"]

PRECISION_IDS: tuple[PrecisionId, ...] = ("fp32", "fp16_mixed", "bf16_mixed")
COMPILE_MODES: tuple[CompileMode, ...] = (
    "eager",
    "default",
    "max-autotune-no-cudagraphs",
    "max-autotune",
)
SUPPORTED_ADAPTER_IDS: tuple[str, ...] = ("plain_pytorch",)

_CUDA_DEVICE_RE = re.compile(r"^cuda(:([0-9]+))?$")


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    adapter_id: Literal["plain_pytorch"]
    device: str
    precision: PrecisionId
    compile_mode: CompileMode
    fullgraph: bool = False
    dynamic: bool | None = None
    backward_pass_autocast: Literal["off"] | None = None
    # Compile tuning: dynamic=True compiles one kernel for varying
    # shapes instead of one per shape; cudagraph modes require static
    # shapes, so dynamic input pairs with max-autotune-no-cudagraphs.
    # Evidence:
    # https://docs.pytorch.org/docs/2.14/generated/torch.compile.html
    # isolate_recompiles scopes each compile to its own cache bucket,
    # so factory-built models stop colliding in the shared cache.
    # TypeError fallback covers torch without the kwarg; the floor is
    # an open question (lockfile pins 2.14, older notes name 2.13), so
    # the fallback stays until the floor is decided.
    isolate_recompiles: bool = True
    recompile_limit: int | None = None


@dataclass(frozen=True, slots=True)
class RuntimeHandle:
    model: object
    optimizer: object
    backward: Callable[[object], None]
    device: object
    runtime_identity: DigestText


@runtime_checkable
class RuntimeAdapter(Protocol):
    def setup(self, *, model: object, optimizer: object, spec: RuntimeSpec) -> RuntimeHandle: ...

    def barrier(self) -> None: ...

    def synchronize(self) -> None: ...


def validate_runtime_spec(spec: RuntimeSpec) -> None:
    """Reject unknown adapter_id, precision, compile_mode, or device.

    Pure format/enum validation; device *availability* is enforced where the
    adapter binds to hardware (see :func:`build_runtime`).
    """
    if not isinstance(spec, RuntimeSpec):
        raise ContractError(f"runtime spec must be RuntimeSpec, got {type(spec).__name__}")
    if spec.adapter_id not in SUPPORTED_ADAPTER_IDS:
        raise ContractError(
            f"unknown runtime adapter_id {spec.adapter_id!r}; "
            f"supported: {list(SUPPORTED_ADAPTER_IDS)}"
        )
    if spec.precision not in PRECISION_IDS:
        raise ContractError(
            f"unknown precision {spec.precision!r}; supported: {list(PRECISION_IDS)}"
        )
    if spec.compile_mode not in COMPILE_MODES:
        raise ContractError(
            f"unknown compile_mode {spec.compile_mode!r}; supported: {list(COMPILE_MODES)}"
        )
    _validate_device_string(spec.device)
    for flag_name in ("fullgraph",):
        value = getattr(spec, flag_name)
        if not isinstance(value, bool):
            raise ContractError(f"{flag_name} must be bool, got {type(value).__name__}")
    if spec.dynamic is not None and not isinstance(spec.dynamic, bool):
        raise ContractError(f"dynamic must be bool or None, got {type(spec.dynamic).__name__}")
    if spec.backward_pass_autocast is not None and spec.backward_pass_autocast != "off":
        raise ContractError(
            f"backward_pass_autocast must be None or 'off', got {spec.backward_pass_autocast!r}"
        )


def _validate_device_string(device: str) -> None:
    if not isinstance(device, str):
        raise ContractError(f"device must be a str, got {type(device).__name__}")
    if device == "cpu":
        return
    if _CUDA_DEVICE_RE.match(device) is not None:
        index_text = device.split(":", 1)[1] if ":" in device else None
        if index_text is not None and int(index_text) > 63:
            raise ContractError(f"cuda device index out of range: {device!r}")
        return
    raise ContractError(
        f"unknown device {device!r}; supported forms: 'cpu', 'cuda', 'cuda:<index>'"
    )


def require_device_available(device: str) -> None:
    """Typed rejection when a CUDA device string cannot be bound."""
    import torch

    if device != "cpu" and not torch.cuda.is_available():
        raise ContractError(
            f"device {device!r} requested but torch.cuda.is_available() is False; "
            "GPU probes must never silently fall back to CPU"
        )


def runtime_identity(spec: RuntimeSpec) -> DigestText:
    """Stable identity digest of the exact runtime configuration."""
    payload = {
        "artifact_type": "hydra2.runtime_spec",
        "schema_version": "1.0.0",
        **asdict(spec),
    }
    return of_canonical(payload)


def build_runtime(
    *, adapter: RuntimeAdapter, model: object, optimizer: object, spec: RuntimeSpec
) -> RuntimeHandle:
    """SPEC 10 build order: validate, compile once, then adapter.setup.

    For non-fp32 precision with a compiled path, ``backward_pass_autocast``
    MUST be 'off' and the functorch patch stays active around BOTH the
    compile call and ``adapter.setup`` (Fabric may unwrap/reapply compile
    inside setup).
    """
    validate_runtime_spec(spec)
    require_device_available(spec.device)

    def compile_once(m: object) -> object:
        if spec.compile_mode == "eager":
            return m
        import torch

        # Varying shapes compile one kernel under dynamic=True; see the
        # RuntimeSpec field comment for the cudagraph pairing. Torch
        # itself refuses cudagraph capture under deterministic
        # algorithms — this code only passes the mode through, it
        # disables nothing.
        compile_kwargs: dict[str, object] = {
            "backend": "inductor",
            "mode": spec.compile_mode,
            "fullgraph": spec.fullgraph,
            "dynamic": spec.dynamic,
        }
        if spec.recompile_limit is not None:
            compile_kwargs["recompile_limit"] = spec.recompile_limit
        # isolate_recompiles buckets each compile separately; the
        # TypeError fallback covers torch without the kwarg. Floor is
        # an open question (see RuntimeSpec field comment); keep the
        # fallback — removing it is out of scope.
        # Route the calls through Any: pyrefly's torch stubs lag the
        # 2.14 runtime (no isolate_recompiles/recompile_limit), so a
        # typed torch.compile call cannot verify. Runtime behavior is
        # unchanged; the TypeError fallback still covers 2.13.
        torch_compile: Any = torch.compile
        try:
            return torch_compile(
                cast("Any", m),
                **compile_kwargs,
                isolate_recompiles=spec.isolate_recompiles,
            )
        except TypeError:
            # Fallback without isolate_recompiles; recompile_limit
            # bounds the shared cache so factory-built models collide
            # less often without per-compile buckets.
            return torch_compile(
                cast("Any", m),
                **compile_kwargs,
            )

    if spec.precision != "fp32" and spec.compile_mode != "eager":
        if spec.backward_pass_autocast != "off":
            raise ContractError(
                "compiled non-fp32 precision requires backward_pass_autocast == 'off'; "
                f"got {spec.backward_pass_autocast!r}"
            )
        # Functorch shim: backward_pass_autocast is missing from the
        # public torch.compiler.config in 2.14, so the private import
        # stays with a nullcontext fallback; no extra dep, behavior
        # unchanged.
        # Evidence:
        # https://docs.pytorch.org/docs/2.14/generated/torch.compiler.config.html
        try:  # runtime import inside branch, not top
            # Private import stays: backward_pass_autocast is missing
            # from the public config (Evidence above), so the shim
            # falls back to a no-op when this surface is unavailable.
            # torch.compiler verifies the 2.14 public surface
            # (compile-order side effect).
            # Evidence:
            # https://docs.pytorch.org/docs/2.14/generated/torch.func.html
            # https://docs.pytorch.org/docs/2.14/generated/torch.compiler.html
            import torch._functorch.config as functorch_config  # type: ignore[import-not-found, no-redef]
            import torch.compiler  # noqa: F401  # pyrefly: ignore[missing-import]
        except ImportError:
            try:
                # Public fallback lacks backward_pass_autocast, so the
                # nullcontext below absorbs it (no-op path).
                import torch.compiler.config as functorch_config  # type: ignore[import-not-found, no-redef]
            except ImportError:
                # No functorch surface; None selects nullcontext below.
                # Ignore covers assigning None to the module alias.
                functorch_config = None  # type: ignore[assignment]
        # Private patch() lacks stubs; hasattr-guarded with nullcontext
        # fallback, so the ignore is safe (see shim comment above).
        _patch_ctx = (
            functorch_config.patch(backward_pass_autocast="off")  # type: ignore[attr-defined]
            if functorch_config is not None and hasattr(functorch_config, "patch")
            else contextlib.nullcontext()
        )
        with _patch_ctx:
            model = compile_once(model)
            return adapter.setup(model=model, optimizer=optimizer, spec=spec)
    model = compile_once(model)
    return adapter.setup(model=model, optimizer=optimizer, spec=spec)
