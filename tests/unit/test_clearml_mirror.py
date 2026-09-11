"""WP-05B/WP-11 ClearML observer mirror — behavior contract.

Ports the six removed-MLflow-mirror behaviors onto
:mod:`hydra2.tracking.clearml_mirror` using a recording ``_FakeClearml``
double injected via ``sys.modules`` (no server, no SDK install required):

1. disabled by default (every method a no-op, ``clearml`` never imported)
2. explicit disable wins over the opt-in flag
3. missing dependency degrades to :class:`NullMirror`
4. metric allowlist drops unknown keys across all log paths
5. offline dir defaults under the artifact root (hermetic) + offline init
6. :func:`make_mirror` never raises, even on malformed kwargs
"""

from __future__ import annotations

import hashlib
import os
import sys
from types import ModuleType
from typing import Any, ClassVar

import pytest

from hydra2.tracking.clearml_mirror import (
    EXPERIMENT_DEFAULT,
    ClearmlMirror,
    NullMirror,
    default_offline_dir,
    is_enabled,
    make_mirror,
)

pytestmark = pytest.mark.contract_package("WP-05B")

_ENV_VARS = (
    "HYDRA2_CLEARML_ENABLED",
    "HYDRA2_CLEARML_DISABLED",
    "HYDRA2_CLEARML_OFFLINE_DIR",
    "CLEARML_OFFLINE_MODE",
    "CLEARML_CACHE_DIR",
)

_DIGEST = "sha256:" + "ab" * 32
_TELEMETRY_DIGEST = "sha256:" + "cd" * 32


class _FakeLogger:
    """Recording stand-in for ``clearml.Logger``."""

    def __init__(self) -> None:
        self.scalars: list[dict[str, Any]] = []
        self.texts: list[str] = []

    def report_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
        self.scalars.append(
            {"title": title, "series": series, "value": value, "iteration": iteration}
        )

    def report_text(self, msg: str, level: int = 20, print_console: bool = True) -> None:
        self.texts.append(str(msg))


class _FakeTask:
    """Recording stand-in for ``clearml.Task`` (class-level call registry)."""

    init_calls: ClassVar[list[dict[str, Any]]] = []
    set_offline_calls: ClassVar[list[bool]] = []
    instances: ClassVar[list[_FakeTask]] = []

    @classmethod
    def reset(cls) -> None:
        cls.init_calls = []
        cls.set_offline_calls = []
        cls.instances = []

    @classmethod
    def init(cls, **kwargs: Any) -> _FakeTask:
        cls.init_calls.append(dict(kwargs))
        task = cls(**kwargs)
        cls.instances.append(task)
        return task

    @classmethod
    def set_offline(cls, offline_mode: bool = False) -> None:
        cls.set_offline_calls.append(bool(offline_mode))

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = dict(kwargs)
        self.logger = _FakeLogger()
        self.artifacts: dict[str, Any] = {}
        self.connected: list[Any] = []
        self.tags: list[str] = []
        self.closed = 0
        self.id = f"offline-{len(_FakeTask.instances):04d}"

    def get_logger(self) -> _FakeLogger:
        return self.logger

    def upload_artifact(self, name: str, artifact_object: Any, **kwargs: Any) -> bool:
        self.artifacts[str(name)] = artifact_object
        return True

    def connect(self, mutable: Any, name: str | None = None, **kwargs: Any) -> Any:
        self.connected.append(mutable)
        return mutable

    def add_tags(self, tags: list[str]) -> None:
        self.tags.extend(list(tags))

    def close(self) -> None:
        self.closed += 1

    def get_offline_mode_folder(self) -> None:
        return None


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delitem(sys.modules, "clearml", raising=False)


@pytest.fixture
def fake_clearml(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    _FakeTask.reset()
    module = ModuleType("clearml")
    module.Task = _FakeTask  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "clearml", module)
    return module


def _manifest_hashes() -> dict[str, str]:
    return {
        key: "sha256:" + hashlib.sha256(f"test:{key}".encode()).hexdigest()
        for key in ("run_spec_hash", "dataset_manifest_hash")
    }


def _loop_config() -> dict[str, Any]:
    return {
        "microbatch_size": 4,
        "accumulation_steps": 2,
        "max_updates": 10,
        "checkpoint_frequency_updates": 5,
        "seed": 7,
        "w_policy": 1.0,
        "w_placement": 0.5,
        "w_value": 0.0,
        "w_event": {"seat": 0.5},
    }


def test_disabled_by_default_is_noop(clean_env: None, tmp_path: Any) -> None:
    assert is_enabled() is False
    mirror = make_mirror()
    assert isinstance(mirror, NullMirror)
    assert mirror.start_run() is None
    mirror.log_update({"total": 1.0}, step=0)
    mirror.log_checkpoint(
        checkpoint_path=tmp_path / "missing.pt", manifest_json={"global_update": 0}
    )
    mirror.log_eval_report("wp06", {"total": 1.0}, digest=_DIGEST)
    mirror.log_promotion({"observed_estimate": 1.0}, digest=_DIGEST)
    mirror.log_duplicate_audit(manifest_digest=_DIGEST, sidecar={"verdict": "pass"})
    mirror.close()
    mirror.close()
    assert "clearml" not in sys.modules


def test_explicit_disable_wins_over_enable(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HYDRA2_CLEARML_ENABLED", "1")
    assert is_enabled(explicit=False) is False
    assert isinstance(make_mirror(enabled=False), NullMirror)


def test_missing_dependency_degrades_to_null(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HYDRA2_CLEARML_ENABLED", "1")
    monkeypatch.setitem(sys.modules, "clearml", None)
    assert is_enabled() is False
    assert is_enabled(explicit=True) is False
    assert isinstance(make_mirror(), NullMirror)


def test_metric_allowlist_drops_unknown_keys(
    fake_clearml: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setenv("HYDRA2_CLEARML_ENABLED", "1")
    manifest_hashes = _manifest_hashes()
    mirror = make_mirror(
        manifest_hashes=manifest_hashes, loop_config=_loop_config(), run_name="test-run"
    )
    assert isinstance(mirror, ClearmlMirror)
    assert not isinstance(mirror, NullMirror)
    first = mirror.start_run()
    assert first is not None
    assert mirror.start_run() == first
    assert len(_FakeTask.init_calls) == 1
    init_kwargs = _FakeTask.init_calls[0]
    assert init_kwargs["project_name"] == EXPERIMENT_DEFAULT
    assert init_kwargs["task_name"] == "test-run"
    assert init_kwargs["auto_connect_frameworks"] is False
    assert init_kwargs["auto_connect_arg_parser"] is False
    task = _FakeTask.instances[0]
    assert f"manifest.run_spec_hash={manifest_hashes['run_spec_hash']}" in init_kwargs["tags"]
    assert len(task.connected) == 1
    assert task.connected[0]["seed"] == "7"
    assert task.connected[0]["w_event.seat"] == "0.5"

    mirror.log_update(
        {
            "total": 0.5,
            "top1": 0.25,
            "wall": 1.0,
            "hidden_tiles": 2.0,
            "sneaky_future_scalar": 3.0,
            "global_update": 7.0,
        },
        step=3,
    )
    series = {
        (row["title"], row["series"]): (row["value"], row["iteration"])
        for row in task.logger.scalars
    }
    assert series[("total", "total")] == (0.5, 3)
    assert series[("top1", "top1")] == (0.25, 3)
    assert all(
        key not in ("wall", "hidden_tiles", "sneaky_future_scalar", "global_update")
        for _, key in series
    )

    ckpt = tmp_path / "ckpt-000003.pt"
    ckpt.write_bytes(b"fake-checkpoint")
    mirror.log_checkpoint(
        checkpoint_path=ckpt,
        manifest_json={
            "checkpoint_file": ckpt.name,
            "global_update": 3,
            "manifest_hashes": manifest_hashes,
        },
    )
    assert task.artifacts["checkpoints/ckpt-000003.pt"] == ckpt.as_posix()
    assert task.artifacts["manifests/ckpt-000003.json"]["global_update"] == 3

    mirror.log_eval_report(
        "wp06",
        {"observed_estimate": 0.1, "ci_lo": 0.05, "ci_hi": 0.15, "mystery": 99.0},
        digest=_DIGEST,
    )
    assert task.artifacts["eval/wp06.json"]["mystery"] == 99.0
    eval_series = {row["series"] for row in task.logger.scalars if row["title"] == "eval/wp06"}
    assert {"observed_estimate", "ci_lo", "ci_hi"} <= eval_series
    assert "mystery" not in eval_series
    assert f"eval.wp06.digest={_DIGEST}" in task.tags

    mirror.log_promotion(
        {"observed_estimate": 0.1, "ci_lo": 0.05, "ci_hi": 0.15, "extra": 1.0},
        digest=_DIGEST,
    )
    assert "promotion/record.json" in task.artifacts
    promo = {
        row["series"]: row["value"] for row in task.logger.scalars if row["title"] == "promotion"
    }
    assert set(promo) == {"observed_estimate", "ci_lo", "ci_hi"}

    mirror.log_duplicate_audit(
        manifest_digest=_DIGEST,
        telemetry_digest=_TELEMETRY_DIGEST,
        sidecar={"verdict": "pass"},
    )
    assert task.artifacts["duplicate/confirmation_sidecar.json"] == {"verdict": "pass"}
    assert f"duplicate.manifest_digest={_DIGEST}" in task.tags
    assert f"duplicate.telemetry_digest={_TELEMETRY_DIGEST}" in task.tags

    mirror.close()
    mirror.close()
    assert task.closed == 1


def test_offline_dir_defaults_under_artifact_root(
    fake_clearml: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from hydra2.config import artifact_root

    monkeypatch.setenv("HYDRA2_ARTIFACT_ROOT", str(tmp_path))
    assert default_offline_dir() == artifact_root() / "clearml_offline"
    custom = tmp_path / "custom-offline"
    monkeypatch.setenv("HYDRA2_CLEARML_OFFLINE_DIR", str(custom))
    assert default_offline_dir() == custom
    assert not custom.exists()

    monkeypatch.setenv("HYDRA2_CLEARML_ENABLED", "1")
    monkeypatch.setenv("CLEARML_OFFLINE_MODE", "1")
    previous_cache = os.environ.get("CLEARML_CACHE_DIR")
    try:
        mirror = make_mirror()
        assert mirror.start_run() is not None
        assert _FakeTask.set_offline_calls == [True]
        assert os.environ.get("CLEARML_CACHE_DIR") == str(custom)
        assert custom.is_dir()
    finally:
        if previous_cache is None:
            os.environ.pop("CLEARML_CACHE_DIR", None)
        else:
            os.environ["CLEARML_CACHE_DIR"] = previous_cache


def test_make_mirror_never_raises(clean_env: None) -> None:
    assert isinstance(make_mirror(), NullMirror)
    assert isinstance(
        make_mirror(manifest_hashes="garbage", loop_config=["not", "a", "dict"], enabled=True),
        ClearmlMirror,
    )


def test_metric_allowlist_passes_per_type_and_calibration(
    fake_clearml: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-type scorecards + temperature/calibrated keys ride the mirror; unknown still dropped."""
    monkeypatch.setenv("HYDRA2_CLEARML_ENABLED", "1")
    mirror = make_mirror(
        manifest_hashes=_manifest_hashes(), loop_config=_loop_config(), run_name="kinds-run"
    )
    assert mirror.start_run() is not None
    task = _FakeTask.instances[0]
    mirror.log_update(
        {
            "total": 0.5,
            "temperature": 1.25,
            "calibrated_nll": 2.1,
            "calibrated_ece": 0.04,
            "per_type/ron/recall": 0.2,
            "per_type/ron/low_support": 1.0,
            "per_type/discard/n": 35.0,
            "wall": 1.0,
            "sneaky_future_scalar": 3.0,
        },
        step=5,
    )
    series = {
        (row["title"], row["series"]): (row["value"], row["iteration"])
        for row in task.logger.scalars
    }
    assert series[("temperature", "temperature")] == (1.25, 5)
    assert series[("calibrated_nll", "calibrated_nll")] == (2.1, 5)
    assert series[("calibrated_ece", "calibrated_ece")] == (0.04, 5)
    assert series[("per_type/ron/recall", "per_type/ron/recall")] == (0.2, 5)
    assert series[("per_type/ron/low_support", "per_type/ron/low_support")] == (1.0, 5)
    assert series[("per_type/discard/n", "per_type/discard/n")] == (35.0, 5)
    assert all(key not in ("wall", "sneaky_future_scalar") for _, key in series)
