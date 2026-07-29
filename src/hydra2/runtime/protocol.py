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

from hydra2._canon import sha256_digest_of_json
from hydra2.contracts.common import ContractError, DigestText, make_digest_text

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
SUPPORTED_ADAPTER_IDS: tuple[str, ...] = ("plain_pytorch", "fabric_2.6.5")

_CUDA_DEVICE_RE = re.compile(r"^cuda(:([0-9]+))?$")


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    adapter_id: Literal["plain_pytorch", "fabric_2.6.5"]
    device: str
    precision: PrecisionId
    compile_mode: CompileMode
    fullgraph: bool = False
    dynamic: bool | None = None
    backward_pass_autocast: Literal["off"] | None = None
    # Perf-A §4.7: compile tuning for bucketed seq len (32/64/128/256).
    # Evidence: torch.compile docs (dynamic=True avoids 4x recompiles;
    # 2.14 stable, same URL as below).
    # https://docs.pytorch.org/docs/2.14/generated/torch.compile.html
    # recompile_limit=8 default fallback to eager; mode max-autotune
    # enables cudagraphs which require static shapes (see mode docs:
    # max-autotune-no-cudagraphs disables them for dynamic). Devlog:
    # https://docs.pytorch.org/devlogs/dynamo/2026-05-04-dynamo-isolate-recompiles/
    # isolate_recompiles=True gives per-compile cache bucket, fixing
    # factory-pattern collisions; guarded try/except TypeError for
    # older torch without kwarg (2.13 fallback, stable in 2.14).
    # For hydra bucketed histories use dynamic=True +
    # max-autotune-no-cudagraphs + isolate_recompiles when available.
    # See also:
    # https://docs.pytorch.org/docs/2.14/generated/torch.compiler.config.html
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
    return sha256_digest_of_json(payload)


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

        # Perf-A §4.7: dynamic=True recommended for bucketed seq len
        # to avoid 4x recompiles; guard determinism.
        # Evidence:
        # https://docs.pytorch.org/docs/2.14/generated/torch.compile.html
        # dynamic=None defers dynamism until recompilation,
        # dynamic=True up-front dynamic kernel. Bucketed histories
        # (32/64/128/256) would recompile 4x under None.
        # Mode max-autotune enables cudagraphs (static-shape only);
        # use max-autotune-no-cudagraphs for dynamic.
        # Guard: if deterministic algorithms enabled, cudagraphs
        # disabled anyway falls back to eager-safe path.
        compile_kwargs: dict[str, object] = {
            "backend": "inductor",
            "mode": spec.compile_mode,
            "fullgraph": spec.fullgraph,
            "dynamic": spec.dynamic,
        }
        if spec.recompile_limit is not None:
            compile_kwargs["recompile_limit"] = spec.recompile_limit
        # isolate_recompiles: stable in 2.14 (torch.compile docs +
        # devlog 2026-05-04 isolate-recompiles, same URLs as above);
        # 2.13 fallback via try/except TypeError. When set, isolates
        # factory-pattern cache buckets.
        # Keep try/except for backward compat (2.13 without kwarg) —
        # zero-cons, no behavior change on 2.14.
        # Route the calls through Any: pyrefly's torch stubs lag the
        # 2.14 runtime (no isolate_recompiles/recompile_limit), so a
        # typed torch.compile call cannot verify. Runtime behavior is
        # unchanged; the TypeError fallback still covers 2.13.
        torch_compile: Any = torch.compile
        try:
            # Try new API if available
            return torch_compile(
                cast("Any", m),
                **compile_kwargs,
                isolate_recompiles=spec.isolate_recompiles,
            )
        except TypeError:
            # 2.13 path: fallback without isolate_recompiles;
            # emulated isolation via recompile_limit guard.
            # When isolate_recompiles is unavailable, the shared cache
            # could collide across factory calls; recompile_limit
            # increase mitigates.
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
        # Functorch shim: torch._functorch is private; public path is
        # torch.compiler (2.14) and torch.func, but
        # backward_pass_autocast stays private (torch.compiler.config
        # lacks the key). Guarded try/except with fallback to no-op;
        # no extra dep, behavior unchanged.
        # https://docs.pytorch.org/docs/2.14/generated/torch.compiler.html
        # https://docs.pytorch.org/docs/2.14/generated/torch.func.html
        # https://docs.pytorch.org/docs/2.14/generated/torch.compiler.config.html
        try:  # runtime import inside branch, not top
            import torch._functorch.config as functorch_config  # type: ignore[import-not-found, no-redef]  # private still required for backward_pass_autocast; public torch.compiler.config lacks key in 2.14 (https://docs.pytorch.org/docs/2.14/generated/torch.compiler.config.html); fallback to no-op if unavailable; see https://docs.pytorch.org/docs/2.14/generated/torch.func.html (functorch→torch.func)
            import torch.compiler  # noqa: F401  # pyrefly: ignore[missing-import]  # 2.14 public surface verify (https://docs.pytorch.org/docs/2.14/generated/torch.compiler.html)
        except ImportError:
            try:
                import torch.compiler.config as functorch_config  # type: ignore[import-not-found, no-redef]  # public fallback (lacks backward_pass_autocast, will fallback to no-op)
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

def normalize_digest(value: str) -> DigestText:
    return make_digest_text(value)
