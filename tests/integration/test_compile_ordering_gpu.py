"""GPU probes: real compile-before-Fabric ordering, functorch patch window
around compile AND setup on the live Fabric path, and eager fallback after
the compiled path is disabled.
"""

from __future__ import annotations

import pytest
import torch

from hydra2.config import PARITY_ABS_TOL, PARITY_REL_TOL
from hydra2.runtime.fabric import FabricRuntimeAdapter
from hydra2.runtime.protocol import RuntimeSpec, build_runtime
from tests.conftest import make_batch, make_model_and_optimizer, run_supervised_steps

# Frozen shapes for every compile probe below: identical (rows, features,
# hidden) keeps one Dynamo specialization per compile_mode so the three tests
# share inductor artifacts via TORCHINDUCTOR_CACHE_DIR. Seeds may vary but
# MUST only seed init/data — never tensor values or control flow inside the
# compiled region (that would add guards and force recompiles).
_BATCH_ROWS = 8
_BATCH_FEATURES = 16
SEED = 777


class OrderingRecordingFabric(FabricRuntimeAdapter):
    """Delegating adapter recording what build_runtime handed to setup."""

    def __init__(self):
        super().__init__()
        self.events: list[dict] = []

    def setup(self, *, model, optimizer, spec) -> object:
        import torch._functorch.config as fcfg

        self.events.append(
            {
                "model_compiled": isinstance(model, torch._dynamo.OptimizedModule),
                "patch_value": fcfg.backward_pass_autocast,
            }
        )
        return super().setup(model=model, optimizer=optimizer, spec=spec)


def spec(**overrides) -> RuntimeSpec:
    fields = {
        "adapter_id": "fabric_2.6.5",
        "device": "cuda",
        "precision": "fp32",
        "compile_mode": "default",
        "fullgraph": False,
        "dynamic": False,
        "backward_pass_autocast": None,
    }
    fields.update(overrides)
    return RuntimeSpec(**fields)


@pytest.fixture(scope="module")
def inductor_available():
    try:
        import triton  # noqa: F401
    except ImportError as exc:
        pytest.fail(f"BLOCKER: triton/inductor unavailable for compiled probes: {exc}")
    return True


@pytest.mark.gpu
class TestCompileBeforeFabric:
    def test_compiled_fp32_build_orders_compile_before_setup(
        self, require_cuda, inductor_available
    ):
        adapter = OrderingRecordingFabric()
        model, optimizer = make_model_and_optimizer(SEED)
        handle = build_runtime(
            adapter=adapter,
            model=model,
            optimizer=optimizer,
            spec=spec(precision="fp32"),
        )
        # The model was ALREADY a torch.compile product when Fabric.setup ran.
        assert adapter.events == [{"model_compiled": True, "patch_value": "same_as_forward"}]
        # Fabric rebinds the compiled product as its forward module; the
        # OptimizedModule must survive inside the returned handle.
        assert type(handle.model).__name__ == "_FabricModule"
        assert isinstance(handle.model._forward_module, torch._dynamo.OptimizedModule)

        x, y = make_batch(SEED + 1, rows=_BATCH_ROWS, features=_BATCH_FEATURES)
        losses = run_supervised_steps(handle, x, y, steps=2)
        assert len(losses) == 2 and all(loss > 0 for loss in losses)

    def test_patch_wraps_compile_and_setup_on_real_fabric_amp(
        self, require_cuda, inductor_available
    ):
        import torch._functorch.config as fcfg

        default_value = fcfg.backward_pass_autocast
        adapter = OrderingRecordingFabric()
        model, optimizer = make_model_and_optimizer(SEED + 10)
        handle = build_runtime(
            adapter=adapter,
            model=model,
            optimizer=optimizer,
            spec=spec(precision="bf16_mixed", backward_pass_autocast="off"),
        )
        assert adapter.events == [{"model_compiled": True, "patch_value": "off"}], (
            "patch must be active around BOTH compile and Fabric.setup"
        )
        # Patch restored once the build leaves its context.
        assert fcfg.backward_pass_autocast == default_value

        x, y = make_batch(SEED + 11, rows=_BATCH_ROWS, features=_BATCH_FEATURES)
        losses = run_supervised_steps(handle, x, y, steps=1)
        assert len(losses) == 1 and losses[0] > 0

    def test_eager_fallback_runs_after_compiled_path_disabled(
        self, require_cuda, inductor_available
    ):
        """Compiled build first; then the identical seeded trajectory through
        the same builder with the compiled path disabled."""
        x, y = make_batch(SEED + 21, rows=_BATCH_ROWS, features=_BATCH_FEATURES)

        def build(compiled: bool):
            adapter = OrderingRecordingFabric()
            model, optimizer = make_model_and_optimizer(SEED + 22)
            overrides = {"compile_mode": "default"} if compiled else {"compile_mode": "eager"}
            handle = build_runtime(
                adapter=adapter,
                model=model,
                optimizer=optimizer,
                spec=spec(precision="fp32", **overrides),
            )
            assert adapter.events[0]["model_compiled"] is compiled
            return handle, run_supervised_steps(handle, x, y, steps=3)

        _, losses_compiled = build(compiled=True)
        _, losses_eager = build(compiled=False)
        assert len(losses_compiled) == 3
        for la, lb in zip(losses_compiled, losses_eager, strict=True):
            assert abs(la - lb) <= PARITY_ABS_TOL + PARITY_REL_TOL * abs(la), (
                f"eager fallback diverged from compiled path: {la} vs {lb}"
            )
