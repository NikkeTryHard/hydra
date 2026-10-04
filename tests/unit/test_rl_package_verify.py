"""WP-11 gate: native work-package completion verifier behavior."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from hydra2.artifacts.canonical import canonical_bytes
from hydra2.artifacts.digest import sha256_digest
from hydra2.artifacts.package_verify import verify

if TYPE_CHECKING:
    from pathlib import Path
pytestmark = pytest.mark.contract_package("WP-11")

_TS = ("2026-09-30T00:00:00Z", "2026-09-30T00:01:00Z")


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _payload(wp: str, **over: object) -> dict:
    doc: dict[str, object] = {
        "work_package": wp,
        "status": "passed",
        "inputs": [{"id": "pixi-lock", "sha256": "sha256:" + "f" * 64}],
        "outputs": [],
        "commands": [],
        "tests": [],
        "environment_manifest_sha256": "sha256:" + "e" * 64,
        "started_at_utc": _TS[0],
        "finished_at_utc": _TS[1],
        "blockers": [],
        "deviations": [],
    }
    doc.update(over)
    return doc


def _publish(root: Path, wp: str, payload: dict) -> str:
    envelope = {
        "artifact_type": "hydra2.work_package_completion",
        "schema_version": "1.0.0",
        "compatibility": "exact",
        "payload": payload,
    }
    raw = canonical_bytes(envelope)
    digest = sha256_digest(raw)
    _write(root / "work_packages" / wp / (digest.split(":")[1] + ".json"), raw)
    return digest


def _dep_payload(wp: str, status: str) -> dict:
    return {
        "work_package": wp,
        "status": status,
        "inputs": [],
        "outputs": [],
        "commands": [],
        "tests": [],
        "environment_manifest_sha256": "sha256:" + "e" * 64,
        "started_at_utc": _TS[0],
        "finished_at_utc": _TS[1],
        "blockers": [] if status == "passed" else ["entry gate open"],
        "deviations": [],
    }


def _write_index(root: Path, currents: dict[str, str]) -> None:
    doc = {
        "schema_version": "1.0.0",
        "packages": {wp: {"current": digest, "superseded": []} for wp, digest in currents.items()},
    }
    _write(root / "work_packages" / "index.json", canonical_bytes(doc))


def _report(wp: str) -> dict:
    entry = {
        "module": "test_rl_x.py",
        "fields": ["gate-smoke"],
        "status": "passed",
        "passed": 1,
        "failed": 0,
        "skipped": 0,
    }
    return {
        "work_package": wp,
        "kind": "hydra2.contract_test_report",
        "status": "passed",
        "finished_at_utc": _TS[1],
        "fixtures": {"gate-smoke": entry},
        "totals": {"modules": 1, "passed": 1, "failed": 0, "skipped": 0},
    }


def _valid_target(
    root: Path,
    wp: str,
    *,
    command_argv: list[str],
    deps: tuple[str, ...],
    dep_statuses: dict[str, str] | None = None,
    with_index: bool = True,
) -> str:
    statuses = dep_statuses or {}
    log = f"{wp} gate log\n".encode()
    log_digest = sha256_digest(log)
    _write(root / "logs" / f"{wp}-gate.log", log)
    _write(root / "reports" / wp / "run-1" / "report.json", canonical_bytes(_report(wp)))
    manifest = {"artifact_type": "hydra2.environment", "schema_version": "1.0.0", "note": "fixture"}
    manifest_bytes = canonical_bytes(manifest)
    manifest_digest = sha256_digest(manifest_bytes)
    _write(root / "environment" / "environment-manifest.json", manifest_bytes)
    payload = _payload(
        wp,
        outputs=[{"path": f"logs/{wp}-gate.log", "sha256": log_digest}],
        commands=[{"argv": command_argv, "exit_code": 0, "log_sha256": log_digest}],
        tests=[
            {"id": "gate-smoke", "result": "passed", "evidence": f"reports/{wp}/run-1/report.json"}
        ],
        environment_manifest_sha256=manifest_digest,
    )
    digest = _publish(root, wp, payload)
    currents = {wp: digest}
    for dep in deps:
        currents[dep] = _publish(root, dep, _dep_payload(dep, statuses.get(dep, "passed")))
    if with_index:
        _write_index(root, currents)
    return digest


_WP02D_ARGV = ["pixi", "run", "test-contracts", "--package", "WP-02D"]
_WP02D_DEPS = ("WP-02A", "WP-02B", "WP-02C")
_WP11_ARGV = ["pixi", "run", "test-training", "--package", "WP-11"]
_WP11_DEPS = ("WP-05C", "WP-06")


def test_valid_record_resolves(rust_extension: object, tmp_path: Path) -> None:
    _ = rust_extension
    digest = _valid_target(tmp_path, "WP-02D", command_argv=_WP02D_ARGV, deps=_WP02D_DEPS)
    receipt = verify("WP-02D", str(tmp_path))
    assert receipt["verdict"] == "passed"
    assert receipt["record_digest"] == digest
    assert receipt["record_status"] == "passed"
    assert receipt["index_current"] == digest


def test_wp11_blocked_on_unsupported_entry(rust_extension: object, tmp_path: Path) -> None:
    _ = rust_extension
    _valid_target(tmp_path, "WP-11", command_argv=_WP11_ARGV, deps=_WP11_DEPS)
    receipt = verify("WP-11", str(tmp_path))
    assert receipt["verdict"] == "blocked"
    assert any(
        check["name"] == "dep:WP-05C" and not check["ok"] and "Entry authority" in check["detail"]
        for check in receipt["checks"]
    )


def test_digest_tamper_fails(rust_extension: object, tmp_path: Path) -> None:
    _ = rust_extension
    digest = _valid_target(tmp_path, "WP-02D", command_argv=_WP02D_ARGV, deps=_WP02D_DEPS)
    record = tmp_path / "work_packages" / "WP-02D" / (digest.split(":")[1] + ".json")
    raw = bytearray(record.read_bytes())
    raw[-20] ^= 0x01
    record.write_bytes(bytes(raw))
    receipt = verify("WP-02D", str(tmp_path))
    assert receipt["verdict"] == "failed"


def test_dep_missing_blocked(rust_extension: object, tmp_path: Path) -> None:
    _ = rust_extension
    _valid_target(tmp_path, "WP-02D", command_argv=_WP02D_ARGV, deps=_WP02D_DEPS, with_index=False)
    for stale in (tmp_path / "work_packages" / "WP-02A").glob("*.json"):
        stale.unlink()
    receipt = verify("WP-02D", str(tmp_path))
    assert receipt["verdict"] == "blocked"
    assert any(check["name"] == "dep:WP-02A" and not check["ok"] for check in receipt["checks"])


def test_dep_failed_blocked(rust_extension: object, tmp_path: Path) -> None:
    _ = rust_extension
    _valid_target(
        tmp_path,
        "WP-02D",
        command_argv=_WP02D_ARGV,
        deps=_WP02D_DEPS,
        dep_statuses={"WP-02B": "failed"},
    )
    receipt = verify("WP-02D", str(tmp_path))
    assert receipt["verdict"] == "blocked"
    assert any(
        check["name"] == "dep:WP-02B" and not check["ok"] and "failed" in check["detail"]
        for check in receipt["checks"]
    )


def test_path_escape_fails(rust_extension: object, tmp_path: Path) -> None:
    _ = rust_extension
    _valid_target(tmp_path, "WP-02D", command_argv=_WP02D_ARGV, deps=_WP02D_DEPS, with_index=False)
    log_digest = sha256_digest((tmp_path / "logs" / "WP-02D-gate.log").read_bytes())
    manifest_digest = sha256_digest(
        (tmp_path / "environment" / "environment-manifest.json").read_bytes()
    )
    payload = _payload(
        "WP-02D",
        outputs=[{"path": "../escape.log", "sha256": log_digest}],
        commands=[{"argv": _WP02D_ARGV, "exit_code": 0, "log_sha256": log_digest}],
        tests=[
            {"id": "gate-smoke", "result": "passed", "evidence": "reports/WP-02D/run-1/report.json"}
        ],
        environment_manifest_sha256=manifest_digest,
    )
    digest = _publish(tmp_path, "WP-02D", payload)
    for stale in (tmp_path / "work_packages" / "WP-02D").glob("*.json"):
        if stale.stem != digest.split(":")[1]:
            stale.unlink()
    receipt = verify("WP-02D", str(tmp_path))
    assert receipt["verdict"] == "failed"
    assert any(check["name"] == "outputs" and not check["ok"] for check in receipt["checks"])


def test_unknown_work_package_blocked(rust_extension: object, tmp_path: Path) -> None:
    _ = rust_extension
    receipt = verify("WP-99", str(tmp_path))
    assert receipt["verdict"] == "blocked"
