"""Completion-record validation, hash verification, index semantics, and
the verify command contract (including the real WP-00A record fixture).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from hydra2.cli import main as cli_main
from hydra2.completion import (
    DEPENDENCIES,
    check_dependency_records,
    find_record_path,
    load_index,
    record_hash_of_file,
    update_index,
    validate_record,
    verify_work_package,
)
from hydra2.config import artifact_root, repo_root
from hydra2.contracts.common import ContractError


def _real_wp00a_path() -> Path:
    """Portable WP-00A record path via :func:`hydra2.config.artifact_root`."""
    return artifact_root() / "work_packages" / "WP-00A" / "wp00a-record.json"


# Kept for backward compat; prefer _real_wp00a_path() for lazy env-aware resolution.
REAL_WP00A_RECORD = _real_wp00a_path()


def valid_record(**overrides) -> dict:
    record = {
        "artifact_type": "hydra2.work_package_completion",
        "schema_version": "1.0.0",
        "work_package": "WP-TST",
        "status": "passed",
        "inputs": [{"id": "src/input.txt", "sha256": "sha256:" + "a" * 64}],
        "outputs": [{"path": "out/result.txt", "sha256": "sha256:" + "b" * 64}],
        "commands": [
            {"argv": ["pixi", "run", "test"], "exit_code": 0, "log_sha256": "sha256:" + "c" * 64}
        ],
        "tests": [{"id": "case::one", "result": "passed", "evidence": "assertion held"}],
        "environment_manifest_sha256": "sha256:" + "d" * 64,
        "started_at_utc": "2026-08-22T10:00:00Z",
        "finished_at_utc": "2026-08-22T11:00:00Z",
        "blockers": [],
        "deviations": [],
    }
    record.update(overrides)
    return record


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "repo" / "src").mkdir(parents=True)
    (tmp_path / "root" / "work_packages" / "WP-TST").mkdir(parents=True)
    content = b"deterministic input bytes"
    import hashlib

    input_hash = "sha256:" + hashlib.sha256(content).hexdigest()
    (tmp_path / "repo" / "src" / "input.txt").write_bytes(content)
    record = valid_record()
    record["inputs"][0]["sha256"] = input_hash
    # outputs path absent -> skipped; log absent -> skipped
    record["commands"][0].pop("log_sha256")
    record["environment_manifest_sha256"] = None
    path = tmp_path / "root" / "work_packages" / "WP-TST" / "wp-tst-record.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return tmp_path


class TestValidateRecord:
    def test_valid_record_passes(self):
        assert validate_record(valid_record())["work_package"] == "WP-TST"

    @pytest.mark.parametrize(
        "mutation,match",
        [
            ({"artifact_type": "other.type"}, "artifact_type"),
            ({"schema_version": "2.0.0"}, "schema_version"),
            ({"status": "skipped"}, "status"),
            ({"work_package": "01"}, "work_package"),
            (
                {"inputs": [{"id": "x", "sha256": "nope"}]},
                "digest",
            ),
            ({"inputs": "not-a-list"}, "array"),
            ({"commands": []}, "non-empty"),
            (
                {"commands": [{"argv": [], "exit_code": 0}]},
                "argv",
            ),
            (
                {"commands": [{"argv": ["a"], "exit_code": True}]},
                "exit_code",
            ),
            ({"tests": []}, "non-empty"),
            (
                {"tests": [{"id": "t", "result": "green", "evidence": "e"}]},
                "result",
            ),
            ({"started_at_utc": "yesterday"}, "timestamp"),
            ({"blockers": [1]}, "strings"),
        ],
    )
    def test_invalid_mutations_rejected(self, mutation, match):
        from hydra2.contracts.common import Hydra2Error

        with pytest.raises(Hydra2Error, match=match):
            validate_record(valid_record(**mutation))

    def test_finished_before_started_rejected(self):
        record = valid_record(
            started_at_utc="2026-08-22T12:00:00Z",
            finished_at_utc="2026-08-22T11:00:00Z",
        )
        with pytest.raises(ContractError, match="after"):
            validate_record(record)

    def test_minor_version_newer_than_supported_rejected(self):
        with pytest.raises(Exception, match="newer"):
            validate_record(valid_record(schema_version="1.1.0"))


class TestIndexSemantics:
    def test_update_creates_then_supersedes(self, tmp_path):
        first = "sha256:" + "1" * 64
        second = "sha256:" + "2" * 64
        _, changed = update_index(tmp_path, "WP-X", first)
        assert changed is True
        _, changed = update_index(tmp_path, "WP-X", first)
        assert changed is False  # idempotent
        _, changed = update_index(tmp_path, "WP-X", second)
        assert changed is True
        index = load_index(tmp_path)
        assert index["current"]["WP-X"] == second
        assert index["superseded"]["WP-X"] == [first]

    def test_index_write_is_atomic_canonical(self, tmp_path):
        update_index(tmp_path, "WP-X", "sha256:" + "3" * 64)
        raw = (tmp_path / "work_packages" / "index.json").read_text(encoding="utf-8")
        assert raw == json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":"))

    def test_corrupt_index_rejected(self, tmp_path):
        target = tmp_path / "work_packages" / "index.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError):
            load_index(tmp_path)


def test_dependency_table_covers_registry_entries():
    assert DEPENDENCIES["WP-01"] == ("WP-00A",)
    assert (
        check_dependency_records(
            Path("/unused"), "WP-01", {"current": {"WP-00A": "sha256:" + "0" * 64}}
        )
        == []
    )
    missing = check_dependency_records(Path("/unused"), "WP-01", {"current": {}})
    assert missing == ["WP-00A"]


class TestVerifyWorkPackage:
    def test_missing_package_reports_invalid(self, workspace):
        outcome = verify_work_package(
            "WP-404",
            artifact_root=workspace / "root",
            repo_root=workspace / "repo",
        )
        assert outcome.disposition == "invalid"
        assert "no completion record" in outcome.errors[0]

    def test_happy_path_updates_index(self, workspace):
        outcome = verify_work_package(
            "WP-TST", artifact_root=workspace / "root", repo_root=workspace / "repo"
        )
        assert outcome.disposition == "pass"
        assert outcome.status == "passed"
        assert outcome.checked_hashes == ["input:src/input.txt@live"]
        index = load_index(workspace / "root")
        expected = record_hash_of_file(
            workspace / "root" / "work_packages" / "WP-TST" / "wp-tst-record.json"
        )
        assert index["current"]["WP-TST"] == expected

    def test_input_hash_mismatch_fails_verification(self, workspace):
        record_path = workspace / "root" / "work_packages" / "WP-TST" / "wp-tst-record.json"
        record = json.loads(record_path.read_text())
        record["inputs"][0]["sha256"] = "sha256:" + "9" * 64
        record_path.write_text(json.dumps(record))
        outcome = verify_work_package(
            "WP-TST", artifact_root=workspace / "root", repo_root=workspace / "repo"
        )
        assert outcome.disposition == "invalid"
        assert any("DigestMismatchError" in e for e in outcome.errors)

    def test_failed_status_disposition_fail_without_index_update(self, workspace):
        record_path = workspace / "root" / "work_packages" / "WP-TST" / "wp-tst-record.json"
        record = json.loads(record_path.read_text())
        record["status"] = "failed"
        record_path.write_text(json.dumps(record))
        before = load_index(workspace / "root")
        outcome = verify_work_package(
            "WP-TST", artifact_root=workspace / "root", repo_root=workspace / "repo"
        )
        assert outcome.disposition == "fail"
        after = load_index(workspace / "root")
        assert before.get("current") == after.get("current")

    def test_blocked_status_surfaces_blockers(self, workspace):
        record_path = workspace / "root" / "work_packages" / "WP-TST" / "wp-tst-record.json"
        record = json.loads(record_path.read_text())
        record["status"] = "blocked"
        record["blockers"] = ["waiting on attestation"]
        record_path.write_text(json.dumps(record))
        outcome = verify_work_package(
            "WP-TST", artifact_root=workspace / "root", repo_root=workspace / "repo"
        )
        assert outcome.disposition == "blocked"
        assert outcome.blockers == ["waiting on attestation"]

    def test_find_record_matches_envelope_only(self, workspace):
        stray = workspace / "root" / "work_packages" / "WP-TST" / "stray.json"
        stray.write_text(json.dumps({"work_package": "WP-TST"}), encoding="utf-8")  # no envelope
        found = find_record_path(workspace / "root", "WP-TST")
        assert found.name == "wp-tst-record.json"


class TestRealWp00aRecord:
    @pytest.mark.skipif(not _real_wp00a_path().is_file(), reason="WP-00A record not present")
    def test_real_record_verifies_and_registers(self, tmp_path):
        package_dir = tmp_path / "work_packages" / "WP-00A"
        package_dir.mkdir(parents=True)
        real_record = _real_wp00a_path()
        shutil.copy(real_record, package_dir / "wp00a-record.json")
        # Logs referenced by the real record live beside it; copy them too.
        for name in ("wp00a-cargo-clean.log", "wp00a-nextest.log"):
            source = real_record.parent / name
            if source.is_file():
                shutil.copy(source, package_dir / name)

        outcome = verify_work_package(
            "WP-00A",
            artifact_root=tmp_path,
            repo_root=repo_root(),
        )
        assert outcome.disposition == "pass"
        assert outcome.index_updated is True
        index = load_index(tmp_path)
        assert set(index["current"]) == {"WP-00A"}
        # Verified hashes include every existing repo-relative output fixture.
        verified_outputs = [c for c in outcome.checked_hashes if c.startswith("output:")]
        assert len(verified_outputs) >= 20


class TestCliContract:
    def _write_record(self, root: Path, status: str) -> None:
        package_dir = root / "work_packages" / "WP-TST"
        package_dir.mkdir(parents=True, exist_ok=True)
        record = valid_record(status=status)
        (package_dir / "wp-tst-record.json").write_text(json.dumps(record))

    def test_exit_code_pass(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HYDRA2_ARTIFACT_ROOT", str(tmp_path))
        self._write_record(tmp_path, "passed")
        code = cli_main(
            [
                "work-package",
                "verify",
                "WP-TST",
                "--artifact-root",
                str(tmp_path),
                "--repo-root",
                str(tmp_path),
            ]
        )
        assert code == 0

    def test_exit_code_blocked_is_distinct(self, tmp_path):
        self._write_record(tmp_path, "blocked")
        code = cli_main(
            [
                "work-package",
                "verify",
                "WP-TST",
                "--artifact-root",
                str(tmp_path),
                "--repo-root",
                str(tmp_path),
            ]
        )
        assert code == 3

    def test_exit_code_missing_is_two(self, tmp_path):
        code = cli_main(["work-package", "verify", "WP-GONE", "--artifact-root", str(tmp_path)])
        assert code == 2
