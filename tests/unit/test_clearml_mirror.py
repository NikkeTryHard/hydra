"""WP-05B/WP-11 ClearML observer mirror — behavior contract.

REST-transport contract (M2 declared+landed): the ClearML SDK is dead, so
this file injects a recording ``_FakeBridgeMirror`` double for
``hydra2_replay_rs.mirror`` (no server, no extension build required) and
asserts:

1. disabled by default (every method a no-op, neither ``clearml`` nor the
   bridge is ever imported)
2. explicit disable wins over the opt-in flag
3. missing SDK still enables (the SDK is dead; its absence is fine)
4. metric allowlist drops unknown keys across all log paths, byte-identical
   over the REST envelope and the file fallback
5. offline dir defaults under the artifact root (hermetic) + run-header record
6. :func:`make_mirror` never raises, even on malformed kwargs
7. per-type scorecards + temperature/calibrated keys ride the mirror
8. unreachable server (``False``) falls back to the file record, never raises,
   and never disables the mirror
9. raising bridge falls back the same way
10. missing bridge extension falls back the same way
11. :class:`NullMirror` never touches transport
"""

from __future__ import annotations

import hashlib
import json
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

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
    "HYDRA2_CLEARML_MIRROR_URL",
)

_DIGEST = "sha256:" + "ab" * 32
_TELEMETRY_DIGEST = "sha256:" + "cd" * 32


class _FakeBridgeMirror:
    """Recording stand-in for ``hydra2_replay_rs.mirror`` (no server)."""

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


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delitem(sys.modules, "clearml", raising=False)
    for key in [key for key in sys.modules if key.split(".")[0] == "hydra2_replay_rs"]:
        monkeypatch.delitem(sys.modules, key, raising=False)


@pytest.fixture
def fake_bridge(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> _FakeBridgeMirror:
    from types import ModuleType

    bridge = _FakeBridgeMirror()
    top = ModuleType("hydra2_replay_rs")
    top.mirror = bridge  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hydra2_replay_rs", top)
    return bridge


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


def _read_ops(offline_dir: Path) -> list[dict[str, Any]]:
    path = offline_dir / "mirror.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _make_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> tuple[ClearmlMirror, Path]:
    monkeypatch.setenv("HYDRA2_CLEARML_ENABLED", "1")
    offline = tmp_path / "offline"
    kwargs.setdefault("offline_dir", offline)
    kwargs.setdefault("base_url", "http://mirror.local")
    mirror = make_mirror(manifest_hashes=_manifest_hashes(), loop_config=_loop_config(), **kwargs)
    assert isinstance(mirror, ClearmlMirror)
    assert not isinstance(mirror, NullMirror)
    return mirror, offline


def test_disabled_by_default_is_noop(clean_env: None, tmp_path: Path) -> None:
    assert is_enabled() is False
    offline = tmp_path / "offline"
    mirror = make_mirror(offline_dir=offline)
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
    assert "hydra2_replay_rs" not in sys.modules
    assert not offline.exists()


def test_explicit_disable_wins_over_enable(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HYDRA2_CLEARML_ENABLED", "1")
    assert is_enabled(explicit=False) is False
    assert isinstance(make_mirror(enabled=False), NullMirror)


def test_missing_sdk_still_enables(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The SDK is dead: its absence must not disable the mirror."""
    monkeypatch.setenv("HYDRA2_CLEARML_ENABLED", "1")
    monkeypatch.setitem(sys.modules, "clearml", None)
    assert is_enabled() is True
    assert is_enabled(explicit=True) is True
    mirror, _ = _make_enabled(tmp_path, monkeypatch, run_name="no-sdk-run")
    assert mirror.start_run() is not None


def test_metric_allowlist_drops_unknown_keys(
    fake_bridge: _FakeBridgeMirror, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_hashes = _manifest_hashes()
    mirror, offline = _make_enabled(tmp_path, monkeypatch, run_name="test-run")
    first = mirror.start_run()
    assert first is not None
    assert mirror.start_run() == first

    ops = _read_ops(offline)
    assert [op["op"] for op in ops] == ["start_run"]
    header = ops[0]["task"]
    assert header["project"] == EXPERIMENT_DEFAULT
    assert header["task_name"] == "test-run"
    assert f"manifest.run_spec_hash={manifest_hashes['run_spec_hash']}" in header["tags"]
    assert header["params"]["seed"] == "7"
    assert header["params"]["w_event.seat"] == "0.5"

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
    assert len(fake_bridge.scalars_calls) == 1
    call = fake_bridge.scalars_calls[0]
    assert call["base_url"] == "http://mirror.local"
    assert call["run_id"] == first
    assert call["step"] == 3
    assert json.loads(call["metrics_json"]) == {"total": 0.5, "top1": 0.25}

    ops = _read_ops(offline)
    update = ops[-1]
    assert update["op"] == "log_update"
    assert update["run_id"] == first
    assert update["step"] == 3
    assert update["metrics"] == {"total": 0.5, "top1": 0.25}

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
    assert len(fake_bridge.checkpoint_calls) == 1
    ckpt_call = fake_bridge.checkpoint_calls[0]
    assert ckpt_call["name"] == "ckpt-000003.pt"
    posted_manifest = json.loads(ckpt_call["manifest_json"])
    assert posted_manifest["global_update"] == 3
    assert posted_manifest["manifest_hashes"] == manifest_hashes
    stored = json.loads((offline / "manifests" / "ckpt-000003.json").read_text())
    assert stored["global_update"] == 3

    mirror.log_eval_report(
        "wp06",
        {"observed_estimate": 0.1, "ci_lo": 0.05, "ci_hi": 0.15, "mystery": 99.0},
        digest=_DIGEST,
    )
    eval_posted = json.loads(fake_bridge.scalars_calls[-1]["metrics_json"])
    assert set(eval_posted) == {
        "eval_wp06_observed_estimate",
        "eval_wp06_ci_lo",
        "eval_wp06_ci_hi",
    }
    eval_op = _read_ops(offline)[-1]
    assert eval_op["op"] == "log_eval_report"
    assert eval_op["report"]["mystery"] == 99.0
    assert set(eval_op["metrics"]) == {"observed_estimate", "ci_lo", "ci_hi"}
    assert "mystery" not in eval_op["metrics"]
    assert eval_op["digest"] == _DIGEST

    mirror.log_promotion(
        {"observed_estimate": 0.1, "ci_lo": 0.05, "ci_hi": 0.15, "extra": 1.0},
        digest=_DIGEST,
    )
    promo_posted = json.loads(fake_bridge.scalars_calls[-1]["metrics_json"])
    assert set(promo_posted) == {
        "promotion_observed_estimate",
        "promotion_ci_lo",
        "promotion_ci_hi",
    }
    promo_op = _read_ops(offline)[-1]
    assert promo_op["op"] == "log_promotion"
    assert set(promo_op["metrics"]) == {"observed_estimate", "ci_lo", "ci_hi"}

    mirror.log_duplicate_audit(
        manifest_digest=_DIGEST,
        telemetry_digest=_TELEMETRY_DIGEST,
        sidecar={"verdict": "pass"},
    )
    audit_op = _read_ops(offline)[-1]
    assert audit_op["op"] == "log_duplicate_audit"
    assert audit_op["sidecar"] == {"verdict": "pass"}
    assert f"duplicate.manifest_digest={_DIGEST}" in audit_op["tags"]
    assert f"duplicate.telemetry_digest={_TELEMETRY_DIGEST}" in audit_op["tags"]

    mirror.close()
    mirror.close()
    closes = [op for op in _read_ops(offline) if op["op"] == "close"]
    assert len(closes) == 1


def test_offline_dir_defaults_under_artifact_root(
    fake_bridge: _FakeBridgeMirror, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from hydra2.config import artifact_root

    monkeypatch.setenv("HYDRA2_ARTIFACT_ROOT", str(tmp_path))
    assert default_offline_dir() == artifact_root() / "clearml_offline"
    custom = tmp_path / "custom-offline"
    monkeypatch.setenv("HYDRA2_CLEARML_OFFLINE_DIR", str(custom))
    assert default_offline_dir() == custom
    assert not custom.exists()

    monkeypatch.setenv("HYDRA2_CLEARML_ENABLED", "1")
    mirror = make_mirror()
    assert mirror.start_run() is not None
    assert custom.is_dir()
    header = _read_ops(custom)[0]
    assert header["op"] == "start_run"
    assert header["run_id"] is not None


def test_make_mirror_never_raises(clean_env: None) -> None:
    assert isinstance(make_mirror(), NullMirror)
    assert isinstance(
        make_mirror(manifest_hashes="garbage", loop_config=["not", "a", "dict"], enabled=True),
        ClearmlMirror,
    )


def test_metric_allowlist_passes_per_type_and_calibration(
    fake_bridge: _FakeBridgeMirror, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Per-type scorecards + temperature/calibrated keys ride the mirror; unknown still dropped."""
    mirror, _ = _make_enabled(tmp_path, monkeypatch, run_name="kinds-run")
    assert mirror.start_run() is not None
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
    posted = json.loads(fake_bridge.scalars_calls[0]["metrics_json"])
    assert posted == {
        "total": 0.5,
        "temperature": 1.25,
        "calibrated_nll": 2.1,
        "calibrated_ece": 0.04,
        "per_type/ron/recall": 0.2,
        "per_type/ron/low_support": 1.0,
        "per_type/discard/n": 35.0,
    }


def test_unreachable_server_falls_back_without_raise(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Transport ``False`` (unreachable server) warns + records, never raises, stays enabled."""
    from types import ModuleType

    bridge = _FakeBridgeMirror(ok=False)
    top = ModuleType("hydra2_replay_rs")
    top.mirror = bridge  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hydra2_replay_rs", top)
    mirror, offline = _make_enabled(tmp_path, monkeypatch, run_name="fallback-run")
    assert mirror.start_run() is not None
    with pytest.warns(UserWarning, match="transport fallback"):
        mirror.log_update({"total": 0.5}, step=1)
    with pytest.warns(UserWarning, match="transport fallback"):
        mirror.log_update({"total": 0.75}, step=2)
    assert len(bridge.scalars_calls) == 2
    updates = [op for op in _read_ops(offline) if op["op"] == "log_update"]
    assert [(op["step"], op["metrics"]) for op in updates] == [
        (1, {"total": 0.5}),
        (2, {"total": 0.75}),
    ]


def test_bridge_raise_falls_back_without_raise(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A raising bridge degrades to warn + file record, never raises."""
    from types import ModuleType

    bridge = _FakeBridgeMirror(fail=ConnectionError("refused"))
    top = ModuleType("hydra2_replay_rs")
    top.mirror = bridge  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hydra2_replay_rs", top)
    mirror, offline = _make_enabled(tmp_path, monkeypatch, run_name="raise-run")
    assert mirror.start_run() is not None
    with pytest.warns(UserWarning, match="transport fallback"):
        mirror.log_update({"total": 0.5}, step=1)
    assert _read_ops(offline)[-1]["metrics"] == {"total": 0.5}


def test_missing_bridge_extension_falls_back(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unbuilt extension: file fallback only, never raises, never imports the SDK."""
    monkeypatch.setitem(sys.modules, "hydra2_replay_rs", None)
    mirror, offline = _make_enabled(tmp_path, monkeypatch, run_name="nobridge-run")
    assert mirror.start_run() is not None
    with pytest.warns(UserWarning, match="transport fallback"):
        mirror.log_update({"total": 0.5, "wall": 1.0}, step=4)
    assert _read_ops(offline)[-1]["metrics"] == {"total": 0.5}
    assert "clearml" not in sys.modules


def test_null_mirror_never_touches_transport(clean_env: None, tmp_path: Path) -> None:
    """Disabled mirror: every method no-ops without the bridge or the SDK."""
    offline = tmp_path / "offline"
    mirror = make_mirror(enabled=False, offline_dir=offline)
    assert isinstance(mirror, NullMirror)
    assert mirror.start_run() is None
    mirror.log_update({"total": 1.0}, step=0)
    mirror.log_checkpoint(checkpoint_path=tmp_path / "missing.pt")
    mirror.log_eval_report("heldout", {"top1": 0.5})
    mirror.log_promotion({"observed_estimate": 2.5})
    mirror.log_duplicate_audit(manifest_digest=_DIGEST)
    mirror.close()
    assert "hydra2_replay_rs" not in sys.modules
    assert "clearml" not in sys.modules
    assert not offline.exists()


def test_base_url_env_override(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Explicit kwarg wins over ``HYDRA2_CLEARML_MIRROR_URL``; both beat the default."""
    from types import ModuleType

    bridge = _FakeBridgeMirror()
    top = ModuleType("hydra2_replay_rs")
    top.mirror = bridge  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hydra2_replay_rs", top)
    monkeypatch.setenv("HYDRA2_CLEARML_ENABLED", "1")
    monkeypatch.setenv("HYDRA2_CLEARML_MIRROR_URL", "http://from-env:8080")
    mirror = make_mirror(offline_dir=tmp_path / "a", base_url="http://explicit:9000")
    assert mirror.start_run() is not None
    mirror.log_update({"total": 1.0}, step=0)
    assert bridge.scalars_calls[0]["base_url"] == "http://explicit:9000"

    mirror2 = make_mirror(offline_dir=tmp_path / "b")
    assert mirror2.start_run() is not None
    mirror2.log_update({"total": 1.0}, step=0)
    assert bridge.scalars_calls[1]["base_url"] == "http://from-env:8080"

    monkeypatch.delenv("HYDRA2_CLEARML_MIRROR_URL")
    mirror3 = make_mirror(offline_dir=tmp_path / "c")
    assert mirror3._base_url == "http://127.0.0.1:9"
