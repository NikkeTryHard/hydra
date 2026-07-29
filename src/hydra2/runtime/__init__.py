"""Hydra2 runtime adapters, checkpoint protocol, and environment capture.

Public surface (SPEC 10): :class:`RuntimeSpec`, :class:`RuntimeHandle`,
:class:`RuntimeAdapter`, :func:`build_runtime`, the plain-PyTorch and
standalone-Fabric adapters, project-owned checkpointing, and the
environment manifest.
"""

from hydra2.runtime.checkpoint import (
    CheckpointManifest,
    apply_checkpoint,
    build_manifest,
    load_checkpoint,
    resume_checkpoint,
    save_checkpoint,
)
from hydra2.runtime.environment import capture_environment_manifest
from hydra2.runtime.fabric import FabricRuntimeAdapter
from hydra2.runtime.plain import PlainPytorchAdapter
from hydra2.runtime.protocol import (
    COMPILE_MODES,
    PRECISION_IDS,
    SUPPORTED_ADAPTER_IDS,
    CompileMode,
    PrecisionId,
    RuntimeAdapter,
    RuntimeHandle,
    RuntimeSpec,
    build_runtime,
    validate_runtime_spec,
)

__all__ = [
    "COMPILE_MODES",
    "PRECISION_IDS",
    "SUPPORTED_ADAPTER_IDS",
    "CheckpointManifest",
    "CompileMode",
    "FabricRuntimeAdapter",
    "PlainPytorchAdapter",
    "PrecisionId",
    "RuntimeAdapter",
    "RuntimeHandle",
    "RuntimeSpec",
    "apply_checkpoint",
    "build_manifest",
    "build_runtime",
    "capture_environment_manifest",
    "load_checkpoint",
    "resume_checkpoint",
    "save_checkpoint",
    "validate_runtime_spec",
]
