"""MLflow observer mirror: default-on gating, warn-only sinks, file-store roundtrip."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from hydra2.tracking.mlflow_mirror import (
    NullMlflowMirror,
    is_enabled,
    make_mirror,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.contract_package("WP-14")


@pytest.fixture
def _store_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Re-enable the mirror against a hermetic file store (conftest disables)."""
    monkeypatch.delenv("HYDRA2_MLFLOW_DISABLED", raising=False)
    store = tmp_path / "mlruns"
    monkeypatch.setenv("HYDRA2_MLFLOW_TRACKING_DIR", str(store))
    return store


def test_default_on_when_sdk_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """No env, no explicit flag: mirror is on (user: normal metrics always)."""
    monkeypatch.delenv("HYDRA2_MLFLOW_DISABLED", raising=False)
    monkeypatch.delenv("HYDRA2_MLFLOW_ENABLED", raising=False)
    assert is_enabled() is True


def test_kill_switch_wins_over_explicit_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYDRA2_MLFLOW_DISABLED", "1")
    assert is_enabled(explicit=True) is False
    assert isinstance(make_mirror(enabled=True), NullMlflowMirror)


def test_explicit_false_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HYDRA2_MLFLOW_DISABLED", raising=False)
    assert is_enabled(explicit=False) is False


def test_null_mirror_never_touches_sdk(tmp_path: Path) -> None:
    """Disabled mirror: every method no-ops without importing mlflow."""
    mirror = make_mirror(enabled=False)
    assert isinstance(mirror, NullMlflowMirror)
    assert mirror.start_run() is None
    mirror.log_update({"total": 1.0}, step=0)
    mirror.log_checkpoint(checkpoint_path=tmp_path / "missing.pt")
    mirror.log_eval_report("heldout", {"top1": 0.5})
    mirror.log_promotion({"observed_estimate": 2.5})
    mirror.log_duplicate_audit(manifest_digest="sha256:0")
    mirror.close()  # never raises, never imports


def test_system_metrics_monitor_attaches(_store_env: Path) -> None:
    """Non-null interval starts the stock monitor thread into the run."""
    import threading

    mlflow = pytest.importorskip("mlflow")
    _ = mlflow
    mirror = make_mirror(
        enabled=True,
        tracking_dir=_store_env,
        system_metrics_interval=1.0,
    )
    assert mirror.start_run() is not None
    try:
        assert any(t.name == "SystemMetricsMonitor" for t in threading.enumerate())
    finally:
        mirror.close()


def test_sqlite_store_roundtrip(_store_env: Path) -> None:
    """Enabled mirror writes allowlisted scalars to the hermetic store."""
    mlflow = pytest.importorskip("mlflow")
    _ = mlflow
    mirror = make_mirror(
        enabled=True,
        tracking_dir=_store_env,
        run_name="mirror-test",
        loop_config={"microbatch_size": 4},
        system_metrics_interval=None,  # no monitor thread in tests
    )
    run_id = mirror.start_run()
    assert run_id
    assert mirror.start_run() == run_id  # idempotent
    mirror.log_update({"total": 1.25, "top1": 0.5, "not_a_metric": 9.0}, step=7)
    mirror.log_update("not-a-mapping", step=8)  # type: ignore[arg-type]
    mirror.close()
    client = mlflow.tracking.MlflowClient(tracking_uri=f"sqlite:///{_store_env}/mlruns.db")

    history = client.get_metric_history(run_id, "total")
    assert [point.value for point in history] == [1.25]
    assert [point.step for point in history] == [7]
    assert client.get_metric_history(run_id, "not_a_metric") == []
    assert client.get_run(run_id).data.params["microbatch_size"] == "4"


def test_resume_reuses_run_id(_store_env: Path) -> None:
    """Resume passes the stored run id back; metrics append to one run."""
    mlflow = pytest.importorskip("mlflow")
    _ = mlflow
    first = make_mirror(enabled=True, tracking_dir=_store_env, system_metrics_interval=None)
    run_id = first.start_run()
    assert run_id
    first.log_update({"total": 1.0}, step=0)
    first.close()

    second = make_mirror(enabled=True, tracking_dir=_store_env, system_metrics_interval=None)
    assert second.start_run(run_id=run_id) == run_id
    second.log_update({"total": 0.5}, step=1)
    second.close()

    client = mlflow.tracking.MlflowClient(tracking_uri=f"sqlite:///{_store_env}/mlruns.db")
    assert [p.value for p in client.get_metric_history(run_id, "total")] == [1.0, 0.5]


def test_sdk_failure_degrades_to_null(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unimportable SDK (or store failure) never raises out of the factory."""
    monkeypatch.delenv("HYDRA2_MLFLOW_DISABLED", raising=False)
    for key in [key for key in sys.modules if key == "mlflow" or key.startswith("mlflow.")]:
        monkeypatch.setitem(sys.modules, key, None)
    monkeypatch.setitem(sys.modules, "mlflow", None)
    assert isinstance(make_mirror(), NullMlflowMirror)
