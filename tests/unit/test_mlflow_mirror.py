"""MLflow observer mirror: default-on gating, warn-only sinks, REST + file fallback.

REST-transport contract (M2 declared+landed): the MLflow SDK is dead, so
this file injects a recording ``_FakeBridgeMirror`` double for
``hydra2._native.mirror`` (no server, no extension build required) and
asserts: default-on gating, kill-switch precedence, Null never touching
transport, allowlisted scalars over the REST envelope + file fallback,
resume reusing the run id, and warn-only fallback on unreachable server /
raising bridge / missing extension.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from hydra2.tracking.mlflow_mirror import (
    NullMlflowMirror,
    is_enabled,
    make_mirror,
)

pytestmark = pytest.mark.contract_package("WP-14")


class _FakeBridgeMirror:
    """Recording stand-in for ``hydra2._native.mirror`` (no server)."""

    def __init__(self, *, ok: bool = True, fail: BaseException | None = None) -> None:
        self.ok = ok
        self.fail = fail
        self.scalars_calls: list[dict[str, Any]] = []
        self.checkpoint_calls: list[dict[str, Any]] = []

    def post_scalars(self, base_url: str, run_id: str, step: int, metrics_json: str) -> bool:
        if self.fail is not None:
            raise self.fail
        self.scalars_calls.append(
            {
                "base_url": base_url,
                "run_id": run_id,
                "step": step,
                "metrics_json": metrics_json,
            }
        )
        return self.ok

    def post_checkpoint(self, base_url: str, run_id: str, name: str, manifest_json: str) -> bool:
        if self.fail is not None:
            raise self.fail
        self.checkpoint_calls.append(
            {
                "base_url": base_url,
                "run_id": run_id,
                "name": name,
                "manifest_json": manifest_json,
            }
        )
        return self.ok


def _install_bridge(monkeypatch: pytest.MonkeyPatch, bridge: _FakeBridgeMirror | None) -> None:
    from types import ModuleType

    if bridge is None:
        monkeypatch.setitem(sys.modules, "hydra2._native", None)
        return
    top = ModuleType("hydra2._native")
    top.mirror = bridge  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hydra2._native", top)


@pytest.fixture
def _store_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Re-enable the mirror against a hermetic tracking dir (conftest disables)."""
    monkeypatch.delenv("HYDRA2_MLFLOW_DISABLED", raising=False)
    for key in [
        key for key in sys.modules if key == "hydra2._native" or key.startswith("hydra2._native.")
    ]:
        monkeypatch.delitem(sys.modules, key, raising=False)
    store = tmp_path / "mlruns"
    monkeypatch.setenv("HYDRA2_MLFLOW_TRACKING_DIR", str(store))
    return store


@pytest.fixture
def _bridge(monkeypatch: pytest.MonkeyPatch) -> _FakeBridgeMirror:
    bridge = _FakeBridgeMirror()
    _install_bridge(monkeypatch, bridge)
    return bridge


def _read_ops(store: Path) -> list[dict[str, Any]]:
    path = store / "mirror.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_default_on_without_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """No env, no explicit flag: mirror is on; the SDK is dead so its absence is fine."""
    monkeypatch.delenv("HYDRA2_MLFLOW_DISABLED", raising=False)
    monkeypatch.delenv("HYDRA2_MLFLOW_ENABLED", raising=False)
    monkeypatch.setitem(sys.modules, "mlflow", None)
    assert is_enabled() is True
    assert is_enabled(explicit=True) is True


def test_kill_switch_wins_over_explicit_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYDRA2_MLFLOW_DISABLED", "1")
    assert is_enabled(explicit=True) is False
    assert isinstance(make_mirror(enabled=True), NullMlflowMirror)


def test_explicit_false_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HYDRA2_MLFLOW_DISABLED", raising=False)
    assert is_enabled(explicit=False) is False


def test_null_mirror_never_touches_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabled mirror: every method no-ops without the bridge or the SDK."""
    for key in [
        key for key in sys.modules if key == "hydra2._native" or key.startswith("hydra2._native.")
    ]:
        monkeypatch.delitem(sys.modules, key, raising=False)
    store = tmp_path / "mlruns"
    mirror = make_mirror(enabled=False, tracking_dir=store)
    assert isinstance(mirror, NullMlflowMirror)
    assert mirror.start_run() is None
    mirror.log_update({"total": 1.0}, step=0)
    mirror.log_checkpoint(checkpoint_path=tmp_path / "missing.pt")
    mirror.log_eval_report("heldout", {"top1": 0.5})
    mirror.log_promotion({"observed_estimate": 2.5})
    mirror.log_duplicate_audit(manifest_digest="sha256:0")
    mirror.close()  # never raises, never imports
    assert "hydra2._native" not in sys.modules
    assert "mlflow" not in sys.modules
    assert not store.exists()


def test_file_fallback_roundtrip(_store_env: Path, _bridge: _FakeBridgeMirror) -> None:
    """Enabled mirror POSTs allowlisted scalars and records the file fallback."""
    mirror = make_mirror(
        enabled=True,
        tracking_dir=_store_env,
        run_name="mirror-test",
        loop_config={"microbatch_size": 4},
        base_url="http://mirror.local",
    )
    run_id = mirror.start_run()
    assert run_id
    assert mirror.start_run() == run_id  # idempotent
    mirror.log_update({"total": 1.25, "top1": 0.5, "not_a_metric": 9.0}, step=7)
    mirror.log_update("not-a-mapping", step=8)  # type: ignore[arg-type]
    mirror.close()

    assert len(_bridge.scalars_calls) == 1
    call = _bridge.scalars_calls[0]
    assert call["base_url"] == "http://mirror.local"
    assert call["run_id"] == run_id
    assert call["step"] == 7
    assert json.loads(call["metrics_json"]) == {"total": 1.25, "top1": 0.5}

    ops = _read_ops(_store_env)
    assert [op["op"] for op in ops] == ["start_run", "log_update", "close"]
    assert ops[0]["run_id"] == run_id
    assert ops[0]["run_name"] == "mirror-test"
    assert ops[0]["params"] == {"microbatch_size": "4"}
    assert ops[1]["metrics"] == {"total": 1.25, "top1": 0.5}
    assert ops[1]["step"] == 7
    assert ops[2]["run_id"] == run_id


def test_checkpoint_manifest_roundtrip(
    _store_env: Path, _bridge: _FakeBridgeMirror, tmp_path: Path
) -> None:
    """Checkpoint manifests POST plus land beside the fallback (never tensor bytes)."""
    mirror = make_mirror(enabled=True, tracking_dir=_store_env)
    run_id = mirror.start_run()
    assert run_id
    ckpt = tmp_path / "ckpt-000007.pt"
    ckpt.write_bytes(b"fake-checkpoint")
    mirror.log_checkpoint(
        checkpoint_path=ckpt, manifest_json={"checkpoint_file": ckpt.name, "global_update": 7}
    )
    assert len(_bridge.checkpoint_calls) == 1
    posted = json.loads(_bridge.checkpoint_calls[0]["manifest_json"])
    assert posted == {"checkpoint_file": "ckpt-000007.pt", "global_update": 7}
    stored = json.loads((_store_env / "manifests" / "ckpt-000007.json").read_text())
    assert stored == posted
    ckpt_op = _read_ops(_store_env)[-1]
    assert ckpt_op["op"] == "log_checkpoint"
    assert ckpt_op["checkpoint"] == "ckpt-000007.pt"


def test_resume_reuses_run_id(_store_env: Path, _bridge: _FakeBridgeMirror) -> None:
    """Resume passes the stored run id back; updates share one run id in the fallback."""
    first = make_mirror(enabled=True, tracking_dir=_store_env)
    run_id = first.start_run()
    assert run_id
    first.log_update({"total": 1.0}, step=0)
    first.close()

    second = make_mirror(enabled=True, tracking_dir=_store_env)
    assert second.start_run(run_id=run_id) == run_id
    second.log_update({"total": 0.5}, step=1)
    second.close()

    updates = [
        op for op in _read_ops(_store_env) if op["op"] == "log_update" and op["run_id"] == run_id
    ]
    assert [(op["step"], op["metrics"]) for op in updates] == [
        (0, {"total": 1.0}),
        (1, {"total": 0.5}),
    ]


def test_unreachable_server_falls_back_without_raise(
    _store_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Transport ``False`` (unreachable server) warns + records, never raises."""
    _install_bridge(monkeypatch, _FakeBridgeMirror(ok=False))
    mirror = make_mirror(enabled=True, tracking_dir=_store_env)
    assert mirror.start_run() is not None
    with pytest.warns(UserWarning, match="transport fallback"):
        mirror.log_update({"total": 1.0}, step=0)
    updates = [op for op in _read_ops(_store_env) if op["op"] == "log_update"]
    assert [(op["step"], op["metrics"]) for op in updates] == [(0, {"total": 1.0})]
    # The mirror stays enabled: the next update still attempts transport.
    with pytest.warns(UserWarning, match="transport fallback"):
        mirror.log_update({"total": 0.5}, step=1)
    assert len([op for op in _read_ops(_store_env) if op["op"] == "log_update"]) == 2


def test_bridge_raise_falls_back_without_raise(
    _store_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raising bridge degrades to warn + file record, never raises."""
    _install_bridge(monkeypatch, _FakeBridgeMirror(fail=TimeoutError("timed out")))
    mirror = make_mirror(enabled=True, tracking_dir=_store_env)
    assert mirror.start_run() is not None
    with pytest.warns(UserWarning, match="transport fallback"):
        mirror.log_update({"total": 1.0}, step=0)
    assert _read_ops(_store_env)[-1]["metrics"] == {"total": 1.0}


def test_missing_bridge_extension_falls_back(
    _store_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unbuilt extension: file fallback only, never raises, never imports the SDK."""
    _install_bridge(monkeypatch, None)
    mirror = make_mirror(enabled=True, tracking_dir=_store_env)
    assert mirror.start_run() is not None
    with pytest.warns(UserWarning, match="transport fallback"):
        mirror.log_update({"total": 1.0, "not_a_metric": 2.0}, step=3)
    assert _read_ops(_store_env)[-1]["metrics"] == {"total": 1.0}
    assert "mlflow" not in sys.modules


def test_eval_promotion_duplicate_records(_store_env: Path, _bridge: _FakeBridgeMirror) -> None:
    """Eval/promotion scalars POST namespaced; full payloads land in the fallback."""
    mirror = make_mirror(enabled=True, tracking_dir=_store_env)
    assert mirror.start_run() is not None
    mirror.log_eval_report(
        "heldout",
        {"top1": 0.5, "mystery": 9.0},
        digest="sha256:ee",
    )
    mirror.log_promotion(
        {"observed_estimate": 2.5, "ci_lo": 2.0, "ci_hi": 3.0, "extra": 1.0},
        digest="sha256:pp",
    )
    mirror.log_duplicate_audit(
        manifest_digest="sha256:aa",
        telemetry_digest="sha256:bb",
        sidecar={"verdict": "pass"},
    )
    posted = [json.loads(call["metrics_json"]) for call in _bridge.scalars_calls]
    assert {"eval_heldout_top1": 0.5} in posted
    assert {"promotion_observed_estimate": 2.5, "promotion_ci_lo": 2.0, "promotion_ci_hi": 3.0} in (
        posted
    )
    ops = _read_ops(_store_env)
    eval_op = next(op for op in ops if op["op"] == "log_eval_report")
    assert eval_op["metrics"] == {"top1": 0.5}
    assert eval_op["report"]["mystery"] == 9.0
    assert eval_op["digest"] == "sha256:ee"
    promo_op = next(op for op in ops if op["op"] == "log_promotion")
    assert promo_op["metrics"] == {
        "observed_estimate": 2.5,
        "ci_lo": 2.0,
        "ci_hi": 3.0,
    }
    audit_op = next(op for op in ops if op["op"] == "log_duplicate_audit")
    assert audit_op["sidecar"] == {"verdict": "pass"}
    assert audit_op["manifest_digest"] == "sha256:aa"


def test_factory_never_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Unimportable SDK never disables the mirror; bad kwargs still fall back to Null."""
    monkeypatch.delenv("HYDRA2_MLFLOW_DISABLED", raising=False)
    for key in [key for key in sys.modules if key == "mlflow" or key.startswith("mlflow.")]:
        monkeypatch.setitem(sys.modules, key, None)
    monkeypatch.setitem(sys.modules, "mlflow", None)
    assert isinstance(make_mirror(tracking_dir=tmp_path), NullMlflowMirror) is False
    assert isinstance(make_mirror(bad_kwarg="nope"), NullMlflowMirror) is True
