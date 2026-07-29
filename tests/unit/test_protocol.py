"""RuntimeSpec validation, typed rejections, and build_runtime call order.

The compile-before-setup ordering and the functorch patch window are proven
with an instrumented adapter plus a stubbed torch.compile (no real
compilation in unit scope; real compiled probes live under integration/).
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from hydra2.contracts.common import ContractError, make_digest_text
from hydra2.runtime.protocol import (
    RuntimeHandle,
    RuntimeSpec,
    build_runtime,
    runtime_identity,
    validate_runtime_spec,
)


def base_spec(**overrides) -> RuntimeSpec:
    fields = {
        "adapter_id": "plain_pytorch",
        "device": "cpu",
        "precision": "fp32",
        "compile_mode": "eager",
        "fullgraph": False,
        "dynamic": None,
        "backward_pass_autocast": None,
    }
    fields.update(overrides)
    return RuntimeSpec(**fields)


class RecordingAdapter:
    """Records event order and observed functorch patch state."""

    def __init__(self):
        self.events: list[tuple[str, object]] = []
        self.handle = None

    def setup(self, *, model, optimizer, spec) -> RuntimeHandle:
        import torch._functorch.config as fcfg

        compiled = isinstance(getattr(model, "_compiled_marker", None), str)
        self.events.append(
            ("setup", {"patch_value": fcfg.backward_pass_autocast, "model_compiled": compiled})
        )
        assert self.handle is None, "setup called twice"
        self.handle = RuntimeHandle(
            model=model,
            optimizer=optimizer,
            backward=lambda loss: loss.backward(),
            device=torch.device(spec.device),
            runtime_identity=runtime_identity(spec),
        )
        return self.handle

    def barrier(self) -> None:
        pass

    def synchronize(self) -> None:
        pass


class CompiledStub:
    """Stand-in for torch.compile output; carries an observable marker."""

    def __init__(self, inner):
        self._orig_mod = inner
        self._compiled_marker = "stub-compiled"

    def __call__(self, *args, **kwargs):
        return self._orig_mod(*args, **kwargs)


@pytest.fixture
def stub_compile(monkeypatch):
    calls: list[dict] = []

    def fake_compile(m, **kwargs):
        import torch._functorch.config as fcfg

        calls.append({"patch_value": fcfg.backward_pass_autocast, **kwargs})
        return CompiledStub(m)

    monkeypatch.setattr(torch, "compile", fake_compile)
    return calls


class TestValidationRejections:
    @pytest.mark.parametrize(
        "bad",
        [
            {"adapter_id": "pytorch_lightning"},
            {"adapter_id": "fabric"},
            {"precision": "fp64"},
            {"precision": "amp"},
            {"compile_mode": "reduce-overhead"},
            {"compile_mode": "none"},
            {"device": "tpu"},
            {"device": "CUDA"},
            {"device": "cuda:"},
            {"backward_pass_autocast": "on"},
        ],
    )
    def test_unknown_values_rejected_with_typed_error(self, bad):
        with pytest.raises(ContractError):
            validate_runtime_spec(base_spec(**bad))

    def test_non_bool_fullgraph_rejected(self):
        with pytest.raises(ContractError):
            validate_runtime_spec(base_spec(fullgraph="yes"))

    def test_cuda_unavailable_is_typed(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(ContractError, match="cuda"):
            build_runtime(
                adapter=RecordingAdapter(),
                model=torch.nn.Linear(2, 2),
                optimizer=None,
                spec=base_spec(device="cuda"),
            )


class TestRuntimeIdentity:
    def test_equal_specs_equal_digest(self):
        assert runtime_identity(base_spec()) == runtime_identity(base_spec())

    def test_field_change_changes_digest(self):
        other = runtime_identity(base_spec(precision="bf16_mixed"))
        assert runtime_identity(base_spec()) != other

    def test_identity_is_canonical_digest_form(self):
        value = runtime_identity(base_spec())
        assert value == make_digest_text(value)


class TestBuildOrder:
    def test_eager_fp32_compiles_not_and_patches_not(self, stub_compile):
        adapter = RecordingAdapter()
        model = torch.nn.Linear(4, 4)
        handle = build_runtime(adapter=adapter, model=model, optimizer=None, spec=base_spec())
        assert [name for name, _ in adapter.events] == ["setup"]
        assert stub_compile == []
        assert adapter.events[0][1]["patch_value"] != "off"
        assert adapter.events[0][1]["model_compiled"] is False
        assert handle.model is model  # plain setup returns the exact object

    def test_mixed_precision_eager_requires_no_autocast_flag(self, stub_compile):
        adapter = RecordingAdapter()
        build_runtime(
            adapter=adapter,
            model=torch.nn.Linear(4, 4),
            optimizer=None,
            spec=base_spec(precision="bf16_mixed", compile_mode="eager"),
        )
        assert stub_compile == []  # eager path never compiles

    def test_compiled_amp_without_off_rejected_before_setup(self, stub_compile):
        adapter = RecordingAdapter()
        with pytest.raises(ContractError, match="backward_pass_autocast"):
            build_runtime(
                adapter=adapter,
                model=torch.nn.Linear(4, 4),
                optimizer=None,
                spec=base_spec(
                    precision="bf16_mixed",
                    compile_mode="default",
                    backward_pass_autocast=None,
                ),
            )
        assert adapter.events == []
        assert stub_compile == []

    @pytest.mark.parametrize("precision", ["fp16_mixed", "bf16_mixed"])
    def test_patch_active_around_compile_and_setup_then_restored(self, stub_compile, precision):
        import torch._functorch.config as fcfg

        default_value = fcfg.backward_pass_autocast
        adapter = RecordingAdapter()
        build_runtime(
            adapter=adapter,
            model=torch.nn.Linear(4, 4),
            optimizer=None,
            spec=base_spec(
                precision=precision,
                compile_mode="default",
                backward_pass_autocast="off",
            ),
        )
        # Order: compile happened before setup; both observed patch == 'off'.
        assert len(stub_compile) == 1
        assert stub_compile[0]["patch_value"] == "off"
        setup_event = adapter.events[0][1]
        assert setup_event["patch_value"] == "off"
        assert setup_event["model_compiled"] is True
        # Patch restored after the build completes.
        assert fcfg.backward_pass_autocast == default_value

    def test_fp32_compiled_does_not_enter_patch_window(self, stub_compile):

        adapter = RecordingAdapter()
        build_runtime(
            adapter=adapter,
            model=torch.nn.Linear(4, 4),
            optimizer=None,
            spec=base_spec(precision="fp32", compile_mode="default"),
        )
        assert len(stub_compile) == 1
        assert adapter.events[0][1]["patch_value"] != "off"

    def test_handle_is_frozen_slots(self):
        handle_fields = {f.name for f in dataclasses.fields(RuntimeHandle)}
        assert handle_fields == {
            "model",
            "optimizer",
            "backward",
            "device",
            "runtime_identity",
        }
        assert RuntimeHandle.__dataclass_params__.frozen is True
