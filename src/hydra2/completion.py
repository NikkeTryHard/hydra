"""WP-01 bootstrap: universal work-package completion records and registry.

Implements the completion-record envelope from BUILD_EXECUTION_PLAN §1:
validation of ``hydra2.work_package_completion`` records (schema 1.0.0),
verification of recorded hashes where referenced paths exist, dependency
record presence, exit disposition, and the atomic mutable index at
``$ARTIFACT_ROOT/work_packages/index.json``.

Canonical serialization is stdlib deterministic JSON (see hydra2._canon);
WP-02A upgrades to full RFC 8785 without changing this command contract.

Record hash definition: sha256 over the canonical bytes of the parsed
record document (whitespace-independent).
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from hydra2._canon import (
    atomic_write_bytes,
    canonical_json_bytes,
    sha256_digest_of_json,
    sha256_file,
)
from hydra2.contracts.common import (
    ContractError,
    DigestMismatchError,
    IncompatibleSchemaError,
    make_digest_text,
    make_utc_timestamp,
)

RECORD_ARTIFACT_TYPE = "hydra2.work_package_completion"
RECORD_SCHEMA_VERSION = "1.0.0"
INDEX_SCHEMA_VERSION = "1.0.0"

VALID_STATUSES = ("passed", "failed", "blocked")
VALID_TEST_RESULTS = ("passed", "failed", "skipped")

# Wave-graph dependency edges relevant to registry verification. A package's
# dependencies MUST already be registered before its own record verifies.
DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "WP-00B": ("WP-00A",),
    "WP-01": ("WP-00A",),
    "WP-02A": ("WP-01",),
    # BUILD §Wave2: WP-02B (rules/utility) and WP-02C (action contract) both
    # enter after WP-02A; added under the authorized cutover treatment used
    # for the WP-02A edge (see wp01-record deviations).
    "WP-02B": ("WP-02A",),
    "WP-02C": ("WP-02A",),
    # BUILD §Wave2: WP-02D (events/observation) consumes the WP-02B rules
    # identity and the WP-02C action table; added under the authorized cutover
    # treatment used for the earlier wave-graph edges.
    "WP-02D": ("WP-02B", "WP-02C"),
    # BUILD §6: WP-03C (MahJax quarantine shell) enters after WP-01 and
    # WP-02D; added under the authorized cutover treatment used for the
    # earlier wave-graph edges.
    "WP-03C": ("WP-01", "WP-02D"),
    # BUILD §6: WP-03A (RiichiEnv reference engine) enters after WP-01 and
    # WP-02D; added under the authorized cutover treatment used for the
    # earlier wave-graph edges.
    "WP-03A": ("WP-01", "WP-02D"),
    # BUILD §6: WP-03B (evaluation schemas and synthetic statistics) consumes
    # the rules, action, and event/observation contracts; added under the
    # same authorized cutover treatment.
    "WP-03B": ("WP-02B", "WP-02C", "WP-02D"),
    # BUILD §6: WP-04A (reference conformance corpus) replays cases through
    # the WP-03A reference adapter against the WP-02B/D rules and event
    # contracts; added under the same authorized cutover treatment.
    "WP-04A": ("WP-01", "WP-02D", "WP-03A"),
    # BUILD §9: WP-06 duplicate-block match qualification reuses WP-03B
    # schedules/blocks/telemetry, the WP-04A exact-game corpus lineage, and
    # the WP-05C frozen baseline; added under the same authorized cutover
    # treatment.
    "WP-06": ("WP-03B", "WP-04A", "WP-05C"),
    # BUILD §8 Wave 5 supervised baseline; WP-05A model/inference, WP-05B loop,
    # WP-05C qualification — all consume WP-04A + WP-04B lineage where applicable.
    "WP-05A": ("WP-04A", "WP-04B"),
    "WP-05B": ("WP-05A",),
    "WP-05C": ("WP-05A", "WP-05B"),
    # BUILD §10 belief foundations; natural harness needs evaluation + reference.
    "WP-07A": ("WP-03B", "WP-04A", "WP-05C"),
    "WP-07B": ("WP-05C", "WP-06"),
    # BUILD §11 Wave 8 frozen policy baseline; entry per spec: WP-02B-D, WP-04A,
    # WP-05C, WP-07A. Added under the same authorized cutover.
    "WP-08A": ("WP-02B", "WP-02C", "WP-02D", "WP-04A", "WP-05C", "WP-07A"),
    "WP-08B": ("WP-02B", "WP-02C", "WP-02D", "WP-04A", "WP-05C", "WP-07A"),
    "WP-08C": ("WP-02B", "WP-02C", "WP-02D", "WP-04A", "WP-05C", "WP-07A"),
    # BUILD §12 Wave 9 PBRF core; entry: natural belief + reference.
    "WP-09A": ("WP-03B", "WP-04A", "WP-05C", "WP-07A", "WP-08A", "WP-08B", "WP-08C"),
    # BUILD §12 Wave 9M Candidate 4 modules — one at a time; entry is PBRF core.
    "WP-09B": ("WP-09A",),
    # BUILD §12 Wave 9 persistence factorial; entry: core + WP-09B9 forest.
    # Core alone does not authorize R/P reuse; WP-09C requires promoted forest.
    # For registry gate, depend on WP-09A; B9 gate enforced inside tests/report.
    "WP-09C": ("WP-09A",),
    # BUILD §12 Wave 9 Candidate 5 local resolving; entry: WP-07A, WP-08, WP-09A.
    # Depends on natural belief (WP-07A), fresh baselines (WP-08A/B/C), PBRF (WP-09A).
    "WP-09D": ("WP-07A", "WP-08A", "WP-08B", "WP-08C", "WP-09A"),
    "WP-09E": ("WP-07A", "WP-08A", "WP-08B", "WP-08C"),
    # BUILD §13 Wave 10 Candidate 7 Teacher Distillation; entry: WP-09C/D/E persistence & resolving & gumbel  # noqa: E501
    # plus 5-gate teacher-eligibility (contract/exact/search/match/analysis). For registry gate, depend on  # noqa: E501
    # WP-09C/D/E plus WP-12 analysis gates; analysis compute_only enforced inside teacher selection logic  # noqa: E501
    # (missing/ineligible gate raises ContractError — WP-10 blocked, never synthetic fallback).  # noqa: E501
    "WP-10": ("WP-09C", "WP-09D", "WP-09E", "WP-12"),
    # BUILD §14 Wave 11 Optional custom self-play RL; entry: WP-05C, WP-06.
    "WP-11": ("WP-05C", "WP-06"),
    # BUILD §15 Wave 12 Offline analysis qualification; entry: completed Candidate 0-6
    # outcome registry and every contract/exact/search/match gate for teacher-eligible
    # outcomes. Depends on WP-08A/B/C (Candidates 0-2) and WP-09A/B/D/E (3-6) plus
    # persistence factorial WP-09C for whole-block resource view.
    "WP-12": ("WP-08A", "WP-08B", "WP-08C", "WP-09A", "WP-09B", "WP-09C", "WP-09D", "WP-09E"),
    # BUILD §16 Wave 13 Candidate 8 joint type/world; entry: WP-04B, WP-07A, WP-10
    # plus sufficient held-out logs (data condition). Registry gate on earlier waves.
    # Gate uses WP-04A (conformance) as proxy for WP-04B lineage, plus belief/search.
    "WP-13": ("WP-04A", "WP-07A", "WP-08A", "WP-08B", "WP-08C", "WP-09A"),
}
_BARE_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def coerce_sha256(value: Any) -> str:
    """Accept canonical 'sha256:<hex>' or the bare 64-hex form used by
    pre-WP-01 records (e.g. WP-00A); returns canonical DigestText."""
    try:
        return make_digest_text(value)
    except ContractError:
        if isinstance(value, str) and _BARE_SHA256_RE.fullmatch(value) is not None:
            return make_digest_text("sha256:" + value)
        raise


@dataclass
class VerifyOutcome:
    work_package: str
    status: str
    disposition: str  # "pass" | "blocked" | "fail" | "invalid"
    record_hash: str | None
    record_path: str | None
    index_updated: bool
    checked_hashes: list[str] = field(default_factory=list)
    skipped_paths: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def validate_record(raw: Any) -> dict[str, Any]:
    """Validate a completion-record document; returns it unchanged on success."""
    if not isinstance(raw, dict):
        raise ContractError("completion record must be a JSON object")
    if raw.get("artifact_type") != RECORD_ARTIFACT_TYPE:
        raise ContractError(
            f"artifact_type must be {RECORD_ARTIFACT_TYPE!r}, got {raw.get('artifact_type')!r}"
        )
    version = raw.get("schema_version")
    if not isinstance(version, str) or version.split(".")[0] != RECORD_SCHEMA_VERSION.split(".")[0]:
        raise IncompatibleSchemaError(
            f"unsupported record schema_version {version!r}; major must be 1"
        )
    if version != RECORD_SCHEMA_VERSION:
        raise IncompatibleSchemaError(
            f"record schema_version {version!r} is newer than supported {RECORD_SCHEMA_VERSION!r}"
        )
    wp = raw.get("work_package")
    if not isinstance(wp, str) or not wp.startswith("WP-"):
        raise ContractError(f"work_package must be 'WP-...', got {wp!r}")
    status = raw.get("status")
    if cast("Any", status) not in VALID_STATUSES:  # pyrefly: ignore[explicit-any]
        raise ContractError(f"status must be one of {list(VALID_STATUSES)}, got {status!r}")

    _validate_inputs(raw)
    _validate_outputs(raw)
    _validate_commands(raw)
    _validate_tests(raw)
    for key in ("started_at_utc", "finished_at_utc"):
        value = raw.get(key)
        if not isinstance(value, str):
            raise ContractError(f"{key} must be a UTC timestamp string")
        _ = make_utc_timestamp(value)
    if _parse_utc(cast("Any", raw["started_at_utc"])) > _parse_utc(cast("Any", raw["finished_at_utc"])):  # pyrefly: ignore[explicit-any]  # noqa: E501
        raise ContractError("started_at_utc must not be after finished_at_utc")
    manifest_hash = raw.get("environment_manifest_sha256")
    if manifest_hash is not None:
        _ = coerce_sha256(cast("Any", manifest_hash))  # pyrefly: ignore[explicit-any]
    for key in ("blockers", "deviations"):
        value: Any = raw.get(key, [])  # pyrefly: ignore[explicit-any]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ContractError(f"{key} must be an array of strings")
    return raw


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _validate_inputs(raw: dict[str, Any]) -> None:
    inputs = raw.get("inputs")
    if not isinstance(inputs, list):
        raise ContractError("inputs must be an array")
    for entry in inputs:
        if not isinstance(entry, dict):
            raise ContractError("each input must be an object")
        ident = entry.get("id")
        if not isinstance(ident, str) or ident == "":
            raise ContractError(f"input id invalid: {ident!r}")
        _ = coerce_sha256(entry.get("sha256"))


def _validate_outputs(raw: dict[str, Any]) -> None:
    outputs = raw.get("outputs")
    if not isinstance(outputs, list):
        raise ContractError("outputs must be an array")
    for entry in outputs:
        if not isinstance(entry, dict):
            raise ContractError("each output must be an object")
        path = entry.get("path")
        if not isinstance(path, str) or path == "":
            raise ContractError(f"output path invalid: {path!r}")
        _ = coerce_sha256(entry.get("sha256"))


def _validate_commands(raw: dict[str, Any]) -> None:
    commands = raw.get("commands")
    if not isinstance(commands, list) or len(commands) == 0:
        raise ContractError("commands must be a non-empty array")
    for entry in commands:
        if not isinstance(entry, dict):
            raise ContractError("each command must be an object")
        argv = entry.get("argv")
        if not isinstance(argv, list) or len(argv) == 0 or not all(isinstance(a, str) for a in argv):  # noqa: E501
            raise ContractError(f"command argv invalid: {argv!r}")
        code = entry.get("exit_code")
        if isinstance(code, bool) or not isinstance(code, int):
            raise ContractError(f"exit_code must be int, got {code!r}")
        log_hash = entry.get("log_sha256")
        if log_hash is not None:
            _ = coerce_sha256(cast("Any", log_hash))  # pyrefly: ignore[explicit-any]


def _validate_tests(raw: dict[str, Any]) -> None:
    tests = raw.get("tests")
    if not isinstance(tests, list) or len(tests) == 0:
        raise ContractError("tests must be a non-empty array")
    for entry in tests:
        if not isinstance(entry, dict):
            raise ContractError("each test must be an object")
        if not isinstance(entry.get("id"), str) or entry.get("id") == "":
            raise ContractError(f"test id invalid: {entry.get('id')!r}")
        if cast("Any", entry.get("result")) not in VALID_TEST_RESULTS:  # pyrefly: ignore[explicit-any]
            raise ContractError(
                f"test result must be one of {list(VALID_TEST_RESULTS)}, "
                f"got {entry.get('result')!r}"
            )
        if not isinstance(entry.get("evidence"), str):
            raise ContractError(f"test evidence missing for {entry.get('id')!r}")


def find_record_path(artifact_root: Path, work_package: str) -> Path:
    """Locate the current record file for a package.

    Preference order: canonical immutable name ``<artifact_id>.json`` when a
    matching envelope exists, else any single *.json in the package directory
    whose envelope declares that work package.
    """
    package_dir = artifact_root / "work_packages" / work_package
    candidates = sorted(package_dir.glob("*.json")) if package_dir.is_dir() else []
    matches: list[Path] = []
    for candidate in candidates:
        try:
            raw = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            isinstance(raw, dict)
            and raw.get("work_package") == work_package
            and raw.get("artifact_type") == RECORD_ARTIFACT_TYPE
        ):
            matches.append(candidate)
    if len(matches) == 0:
        raise FileNotFoundError(
            f"no completion record found for {work_package} under {package_dir}"
        )
    if len(matches) > 1:
        # Prefer the canonical artifact_id-named file when present.
        for candidate in matches:
            digest = record_hash_of_file(candidate)
            if candidate.name == digest.split(":", 1)[1] + ".json":
                return candidate
        raise ContractError(
            f"multiple record files declare {work_package}: {[str(m) for m in matches]}"
        )
    return matches[0]


def record_hash_of_file(path: Path) -> str:
    """Record identity: sha256 over canonical bytes of the parsed document."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return sha256_digest_of_json(raw)


def _input_matches(
    path: Path,
    recorded: str,
    *,
    repo_root: Path,
    git_baseline: str | None,
) -> tuple[bool, str]:
    """Live-hash check with git-baseline fallback for tracked sources.

    Later packages legitimately modify shared files; a record whose input
    matches its declared ``git_baseline`` blob is verified as of that
    baseline. Neither live nor baseline matching is a hard failure.
    """
    import hashlib
    import subprocess

    recomputed = sha256_file(path)
    if recomputed == recorded:
        return True, "live"
    if git_baseline is not None:
        rel = path.relative_to(repo_root).as_posix()
        try:
            proc = subprocess.run(
                ["git", "-C", str(repo_root), "cat-file", "blob", f"{git_baseline}:{rel}"],
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            proc = None
        if proc is not None and proc.returncode == 0:
            blob_hash = "sha256:" + hashlib.sha256(proc.stdout).hexdigest()
            if blob_hash == recorded:
                return True, f"git-baseline:{git_baseline}"
    return False, recomputed


def verify_record_content(
    *,
    record_path: Path,
    repo_root: Path,
    artifact_root: Path,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Validate schema and recompute recorded hashes where paths exist.

    Returns (record, checked_descriptions, skipped_descriptions); raises
    DigestMismatchError/ContractError on hard failures.
    """
    raw = json.loads(Path(record_path).read_text(encoding="utf-8"))
    record = validate_record(raw)
    checked: list[str] = []
    skipped: list[str] = []

    package_dir = record_path.parent
    git_baseline = record.get("git_baseline")
    for entry in record.get("inputs", []):
        path: Any = repo_root / entry["id"]  # pyrefly: ignore[explicit-any]
        if path.is_file():
            matched, evidence = _input_matches(
                cast("Any", path),  # pyrefly: ignore[explicit-any]
                coerce_sha256(cast("Any", entry["sha256"])),  # pyrefly: ignore[explicit-any]
                repo_root=repo_root,
                git_baseline=git_baseline if isinstance(git_baseline, str) else None,
            )
            if not matched:
                raise DigestMismatchError(
                    f"input {entry['id']}: recorded {entry['sha256']} != "
                    f"live/baseline recomputed {evidence}"
                )
            checked.append(f"input:{entry['id']}@{evidence}")
        else:
            skipped.append(f"input:{entry['id']} (path absent)")

    for entry in record.get("outputs", []):
        path: Any = repo_root / entry["path"]  # pyrefly: ignore[explicit-any]
        if path.is_file():
            recomputed = sha256_file(cast("Any", path))  # pyrefly: ignore[explicit-any]
            if recomputed != coerce_sha256(cast("Any", entry["sha256"])):  # pyrefly: ignore[explicit-any]
                raise DigestMismatchError(
                    f"output {entry['path']}: recorded {entry['sha256']} != recomputed {recomputed}"
                )
            checked.append(f"output:{entry['path']}")
        else:
            skipped.append(f"output:{entry['path']} (path absent)")

    for entry in record.get("commands", []):
        log_name: Any = entry.get("log")  # pyrefly: ignore[explicit-any]
        log_hash: Any = entry.get("log_sha256")  # pyrefly: ignore[explicit-any]
        if log_name and log_hash:
            log_path: Any = package_dir / log_name  # pyrefly: ignore[explicit-any]
            if log_path.is_file():
                recomputed = sha256_file(cast("Any", log_path))  # pyrefly: ignore[explicit-any]
                if recomputed != coerce_sha256(cast("Any", log_hash)):  # pyrefly: ignore[explicit-any]
                    raise DigestMismatchError(
                        f"command log {log_name}: recorded {log_hash} != recomputed {recomputed}"
                    )
                checked.append(f"log:{log_name}")
            else:
                skipped.append(f"log:{log_name} (file absent)")

    env_hash = record.get("environment_manifest_sha256")
    if env_hash is not None:
        env_path = _resolve_environment_manifest(
            artifact_root, package_dir, coerce_sha256(env_hash)
        )
        if env_path is not None:
            checked.append(f"environment_manifest:{env_path.name}")
        else:
            skipped.append("environment_manifest (no artifact matches recorded hash)")
    return record, checked, skipped


def _resolve_environment_manifest(
    artifact_root: Path,
    package_dir: Path,
    expected: str,
) -> Path | None:
    """Find the environment manifest matching this record's identity.

    Candidates in order: package-local ``*environment*.json`` artifacts
    (per-package captures), then the shared root manifest. Returns the file
    whose sha256 equals the recorded hash; None when nothing matches.
    """
    candidates: list[Path] = sorted(package_dir.glob("*environment*.json"))
    candidates.append(artifact_root / "environment" / "environment-manifest.json")
    for candidate in candidates:
        if candidate.is_file() and sha256_file(candidate) == expected:
            return candidate
    return None


def load_index(artifact_root: Path) -> dict[str, Any]:
    index_path = artifact_root / "work_packages" / "index.json"
    if not index_path.is_file():
        return {"schema_version": INDEX_SCHEMA_VERSION, "current": {}, "superseded": {}}
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise IncompatibleSchemaError(
            f"index schema_version {raw.get('schema_version')!r} unsupported"
        )
    for key in ("current", "superseded"):
        if not isinstance(raw.get(key), dict):
            raise ContractError(f"index.{key} must be an object")
    return raw


def update_index(
    artifact_root: Path, work_package: str, record_hash: str
) -> tuple[dict[str, Any], bool]:
    """Point ``current[work_package]`` at ``record_hash`` atomically.

    Superseding keeps the previous hash in ``superseded``. Returns
    (index, changed). Idempotent: same hash -> unchanged file.
    """
    _ = make_digest_text(record_hash)
    index_path = artifact_root / "work_packages" / "index.json"
    index = load_index(artifact_root)
    current = dict(index["current"])
    superseded = {k: list(cast("Any", v)) for k, v in index["superseded"].items()}  # pyrefly: ignore[explicit-any]
    previous = current.get(work_package)
    if previous == record_hash:
        return index, False
    if previous is not None:
        history = superseded.setdefault(work_package, [])
        if previous not in history:
            history.append(previous)
    current[work_package] = record_hash
    new_index = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "current": dict(sorted(current.items())),
        "superseded": {k: sorted(v) for k, v in sorted(superseded.items())},
    }
    atomic_write_bytes(index_path, canonical_json_bytes(new_index))
    return new_index, True


def check_dependency_records(
    artifact_root: Path, work_package: str, index: dict[str, Any]
) -> list[str]:
    """Return missing dependency packages (empty list = satisfied)."""
    required = DEPENDENCIES.get(work_package, ())
    current = index.get("current", {})
    return [dep for dep in required if dep not in current]


def verify_work_package(
    work_package: str,
    *,
    artifact_root: Path,
    repo_root: Path,
) -> VerifyOutcome:
    """Full verification pipeline used by ``hydra2 work-package verify``."""
    artifact_root = Path(artifact_root)
    repo_root = Path(repo_root)
    outcome = VerifyOutcome(
        work_package=work_package,
        status="unknown",
        disposition="invalid",
        record_hash=None,
        record_path=None,
        index_updated=False,
    )

    try:
        record_path = find_record_path(artifact_root, work_package)
    except FileNotFoundError as exc:
        outcome.errors.append(str(exc))
        return outcome
    outcome.record_path = str(record_path)

    try:
        record, checked, skipped = verify_record_content(
            record_path=record_path,
            repo_root=repo_root,
            artifact_root=artifact_root,
        )
    except (ContractError, DigestMismatchError, OSError, ValueError) as exc:
        outcome.errors.append(f"{type(exc).__name__}: {exc}")
        return outcome
    outcome.checked_hashes = checked
    outcome.skipped_paths = skipped
    outcome.status = record["status"]
    outcome.blockers = list(record.get("blockers", []))

    record_hash = record_hash_of_file(record_path)
    outcome.record_hash = record_hash

    try:
        index = load_index(artifact_root)
    except (ContractError, IncompatibleSchemaError, OSError, ValueError) as exc:
        outcome.errors.append(f"{type(exc).__name__}: {exc}")
        return outcome

    missing_deps = check_dependency_records(artifact_root, work_package, index)
    if len(missing_deps) != 0:
        outcome.errors.append(f"dependency records missing from registry: {missing_deps}")

    if outcome.status == "passed" and len(missing_deps) == 0:
        outcome.disposition = "pass"
    elif outcome.status == "blocked":
        outcome.disposition = "blocked"
    elif outcome.status == "failed":
        outcome.disposition = "fail"
    else:
        outcome.disposition = "invalid"

    if outcome.disposition == "pass":
        _, updated = update_index(artifact_root, work_package, record_hash)
        outcome.index_updated = updated
    return outcome


def format_outcome(outcome: VerifyOutcome) -> str:
    lines = [
        f"verify {outcome.work_package}: disposition={outcome.disposition}",
        f"  status={outcome.status}",
        f"  record={outcome.record_path}",
        f"  record_hash={outcome.record_hash}",
        f"  index_updated={outcome.index_updated}",
        f"  verified_hashes={len(outcome.checked_hashes)}",
        f"  skipped_absent={len(outcome.skipped_paths)}",
    ]
    lines.extend(f"  skip: {item}" for item in outcome.skipped_paths)
    lines.extend(f"  blocker: {blocker}" for blocker in outcome.blockers)
    lines.extend(f"  error: {error}" for error in outcome.errors)
    return "\n".join(lines)
