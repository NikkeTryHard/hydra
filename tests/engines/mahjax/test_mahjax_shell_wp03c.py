"""WP-03C MahJax quarantine shell gates (BUILD lines 381-393).

Proves, in THIS environment: the pinned-origin import probe passes, the jit
compile probe passes, the captured environment tuple matches live ground
truth, consumption fails closed while quarantined, and the token gate logic
accepts/rejects fabricated tokens via the TEST-ONLY internal API (real
issuance arrives with WP-04C).
"""

from __future__ import annotations

import dataclasses
import importlib.metadata as md
import json
import os
import sys
from typing import Any

import pytest

from hydra2._canon import canonical_json_bytes, sha256_digest_of_json
from hydra2.artifacts.digest import sha256_file
from hydra2.config import MAHJAX_PIN_SHA, repo_root
from hydra2.contracts.common import (
    DigestText,
    QualificationRequiredError,
    make_digest_text,
)
from hydra2.engines.mahjax import (
    FRAGMENT_ARTIFACT_NAME,
    AdapterState,
    MahJaxEnvironmentTuple,
    MahJaxQuarantineShell,
    QualificationToken,
    capture_mahjax_tuple,
    fabricate_test_only_token,
    write_mahjax_environment_fragment,
)

pytestmark = pytest.mark.contract_package("WP-03C")

#: Stand-in rules-manifest identity (real binding is issued by WP-04C).
RULES_ID = make_digest_text("sha256:" + "3a" * 32)


def _fresh_shell() -> MahJaxQuarantineShell:
    return MahJaxQuarantineShell()


# ---------------------------------------------------------------------------
# Checklist item 1: runtime SHA verification.
# ---------------------------------------------------------------------------


def test_construction_verifies_pinned_origin_and_boots_quarantined() -> None:
    shell = _fresh_shell()
    assert shell.verified_commit_id == MAHJAX_PIN_SHA
    assert shell.state is AdapterState.QUARANTINED
    assert shell.qualified is False


def test_construction_refuses_unpinned_origin() -> None:
    foreign_pin = "b" * 40
    with pytest.raises(QualificationRequiredError):
        MahJaxQuarantineShell(pin_sha=foreign_pin)


def test_origin_reader_rejects_missing_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(name: str, **_: Any) -> None:
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "distribution", explode)
    with pytest.raises(QualificationRequiredError):
        capture_mahjax_tuple()


# ---------------------------------------------------------------------------
# Checklist items 4+5: probes pass in this env while quarantined.
# ---------------------------------------------------------------------------


def test_import_probe_passes_on_pinned_install() -> None:
    shell = _fresh_shell()
    verdict = shell.import_probe()
    assert verdict["importable"] is True
    assert verdict["commit_id"] == MAHJAX_PIN_SHA
    assert verdict["matches_pin"] is True
    # Probes never flip the lifecycle state.
    assert shell.state is AdapterState.QUARANTINED


def test_compile_probe_jits_trivial_function() -> None:
    shell = _fresh_shell()
    verdict = shell.compile_probe()
    assert verdict["ok"] is True
    assert verdict["actual"] == verdict["expected"] == 7.0
    digest = verdict["output_digest"]
    assert isinstance(digest, str)
    assert digest.startswith("sha256:") and len(digest) == len("sha256:") + 64
    assert shell.state is AdapterState.QUARANTINED


# ---------------------------------------------------------------------------
# Checklist item 2: captured tuple matches actual environment.
# ---------------------------------------------------------------------------


def test_captured_tuple_matches_live_ground_truth() -> None:
    import jax

    captured = capture_mahjax_tuple()
    assert captured.mahjax_commit_id == MAHJAX_PIN_SHA
    assert captured.mahjax_version == md.version("mahjax")
    assert captured.jax_version == md.version("jax")
    assert captured.jaxlib_version == md.version("jaxlib")
    assert captured.backend_platform == jax.default_backend()
    assert len(captured.devices) == len(jax.devices())
    expected_xla = {key: value for key, value in os.environ.items() if key.startswith("XLA_")}
    assert dict(captured.xla_flags) == expected_xla
    assert captured.python_implementation == sys.implementation.name
    assert captured.python_version == ".".join(str(part) for part in sys.version_info[:3])
    lock_hash = sha256_file(repo_root() / "pixi.lock")
    assert str(captured.pixi_lock_sha256) == str(lock_hash)


def test_tuple_fragment_digest_is_canonical_and_stable() -> None:
    first = capture_mahjax_tuple()
    second = capture_mahjax_tuple()
    assert first.to_fragment() == second.to_fragment()
    assert str(first.digest) == str(second.digest)
    recomputed = sha256_digest_of_json(first.to_fragment())
    assert first.digest == recomputed


def test_fragment_persists_beside_environment_manifest(
    tmp_path: Any,
) -> None:
    destination, digest = write_mahjax_environment_fragment(tmp_path)
    assert destination.name == FRAGMENT_ARTIFACT_NAME
    raw = destination.read_bytes()
    parsed = json.loads(raw.decode("utf-8"))
    assert parsed == MahJaxEnvironmentTuple.capture().to_fragment()
    assert raw == canonical_json_bytes(parsed)
    assert digest == sha256_digest_of_json(parsed)


# ---------------------------------------------------------------------------
# Checklist item 4: fail-closed consumption while quarantined.
# ---------------------------------------------------------------------------


def _planner_caller(shell: MahJaxQuarantineShell) -> Any:
    return shell.module  # simulated search/planner entry into engine outputs


def _data_caller(shell: MahJaxQuarantineShell) -> None:
    shell.require_qualified()  # simulated dataset builder gate


def _evaluation_caller(shell: MahJaxQuarantineShell) -> None:
    shell.require_qualified()  # simulated evaluation harness gate


def test_consumption_fails_closed_for_every_caller_kind() -> None:
    shell = _fresh_shell()
    for caller in (_planner_caller, _data_caller, _evaluation_caller):
        with pytest.raises(QualificationRequiredError):
            caller(shell)
    assert shell.state is AdapterState.QUARANTINED
    assert "QUARANTINED" in str(_planner_caller_error(shell))


def _planner_caller_error(shell: MahJaxQuarantineShell) -> QualificationRequiredError:
    try:
        _planner_caller(shell)
    except QualificationRequiredError as exc:
        return exc
    raise AssertionError("planner consumption did not fail closed")


# ---------------------------------------------------------------------------
# Checklist items 3+5: token gate proven with the TEST-ONLY fabrication API.
# ---------------------------------------------------------------------------


def _live_token() -> QualificationToken:
    return fabricate_test_only_token(capture_mahjax_tuple(), rules_id=RULES_ID)


def test_fabricated_token_qualifies_gate_accept_path() -> None:
    shell = _fresh_shell()
    token = _live_token()
    accepted = shell.qualify(token, rules_id=RULES_ID)
    assert accepted == token.identity_digest
    assert shell.state is AdapterState.QUALIFIED
    assert shell.qualified is True
    assert shell.rules_id == RULES_ID
    assert shell.token_identity_digest == token.identity_digest
    # Qualified consumption reaches the genuine installed module.
    module = shell.module
    assert module.__name__ == "mahjax"


def test_fabricated_token_rejects_every_tampered_dimension() -> None:
    base_token = _live_token()

    def rejects(token: QualificationToken, *, rules_id: DigestText = RULES_ID) -> None:
        shell = _fresh_shell()
        with pytest.raises(QualificationRequiredError):
            shell.qualify(token, rules_id=rules_id)
        assert shell.state is AdapterState.QUARANTINED

    rejects(dataclasses.replace(base_token, mahjax_commit_id="c" * 40))
    rejects(dataclasses.replace(base_token, pixi_lock_sha256="sha256:" + "0" * 64))
    rejects(dataclasses.replace(base_token, jax_version="0.0.0"))
    rejects(dataclasses.replace(base_token, jaxlib_version="0.0.0"))
    rejects(dataclasses.replace(base_token, backend_platform="tpu"))
    rejects(dataclasses.replace(base_token, xla_flags=(("XLA_TAMPERED", "1"),)))
    rejects(
        dataclasses.replace(
            base_token,
            devices=((dataclasses.replace(base_token.devices[0], kind="tpu")),),
        )
    )
    rejects(dataclasses.replace(base_token, python_version="3.0.0"))
    rejects(dataclasses.replace(base_token, observation_mode="legacy_v0"))
    rejects(dataclasses.replace(base_token, adapter_version="9.9.9"))
    rejects(base_token, rules_id=make_digest_text("sha256:" + "bb" * 32))
    with pytest.raises(QualificationRequiredError):
        _fresh_shell().qualify("not-a-token", rules_id=RULES_ID)  # type: ignore[arg-type]


def test_post_qualification_drift_demotes_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shell = _fresh_shell()
    shell.qualify(_live_token(), rules_id=RULES_ID)
    assert shell.qualified is True

    # A changed relevant XLA flag alters the live tuple behind the token.
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    with pytest.raises(QualificationRequiredError, match="drifted"):
        shell.require_qualified()
    assert shell.state is AdapterState.QUARANTINED
    assert shell.token_identity_digest is None
    assert shell.rules_id is None

    # Demotion persists after the drift is removed: re-qualification required.
    monkeypatch.undo()
    with pytest.raises(QualificationRequiredError):
        _planner_caller(shell)
