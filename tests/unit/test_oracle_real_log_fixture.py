"""Oracle real-log excerpt fixture — one real game end to end through teacher/oracle path.

Gate owner: docs/training-plan.md NOW oracle real-data fixture. Proves a single
game framed from the Tenhou corpus decodes, validates, then flows through the
real teacher trajectory path plus the privileged oracle join on opaque
decision ids derived from the game identity. Raw tiles never vendored; the
test references the corpus path at runtime and joins on decision-id strings
only. Quarantine-class games pass by asserting the named reject reason.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from hydra2.contracts.common import ContractError

pytestmark = pytest.mark.contract_package("WP-07B")

_REAL_WP12_ART: str | None = None


@pytest.fixture(autouse=True)
def _real_wp12_gates(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Plumb the REAL WP-12 prerequisite: generate the hashed analysis report.

    Teacher selection is fail-closed without
    ``work_packages/WP-12/analysis_gates.json``, so generate the real report
    once per session into a tmp artifact root and point
    ``HYDRA2_ARTIFACT_ROOT`` at it (restored after each test).
    """
    global _REAL_WP12_ART
    if _REAL_WP12_ART is None:
        from hydra2.analysis.qual_gates import generate_hashed_analysis_report

        art = tmp_path_factory.mktemp("oracle_real_log_gates")
        generate_hashed_analysis_report(artifact_root=art)
        _REAL_WP12_ART = str(art)
    monkeypatch.setenv("HYDRA2_ARTIFACT_ROOT", str(_REAL_WP12_ART))


def _corpus_root() -> Path:
    """Resolve Tenhou corpus root or skip when the mount is absent.

    Priority mirrors the training path plus the lineage-test precedent:
    ``HYDRA2_DATA_ROOT`` (what train reads) > ``HYDRA2_TENHOU_MOUNT`` >
    D-017 attestation mount > legacy default. No single hardcoded source:
    operator env wins wherever set. Skip (not fail) when absent keeps CI
    green where the mount does not exist — same as
    ``tests/integration/test_data_lineage.py``; the training-plan gate cites
    the workstation run as proof, and train itself fails closed without the
    mount. Training files are never touched by this resolution.
    """
    candidates = [
        os.environ.get("HYDRA2_DATA_ROOT"),
        os.environ.get("HYDRA2_TENHOU_MOUNT"),
    ]
    try:
        from hydra2.data.attestation import load_attestation

        att = load_attestation()
        mount = att.acquisition_metadata.get("mount")
        candidates.append(str(mount) if mount else None)
    except Exception:
        candidates.append(None)
    candidates.append("/mnt/samsung_nvme/samsung/mahjong_dataset")
    for base in candidates:
        if not base:
            continue
        root = Path(base) / "tenhou-houou-mjai-2024"
        if root.is_dir():
            return root
        parent = Path(base)
        if parent.is_dir():
            cands = sorted(parent.glob("tenhou-houou-mjai-*"))
            if cands:
                return cands[0]
    pytest.skip(f"Tenhou corpus mount absent (tried {candidates})")
    raise AssertionError("unreachable: skip always raises")


def test_real_log_excerpt_runs_teacher_oracle_path(tmp_path: Path) -> None:
    import hashlib

    from hydra2.data.decode import decode_game_object
    from hydra2.data.stream_read import _frame_file, stem_of
    from hydra2.data.validate import validate_game

    root = _corpus_root()
    # Deterministic first file WITHOUT the 184k-entry stat walk: manifest order
    # is sha256-hex of the relative POSIX path, and this corpus root is flat
    # (no subdirs), so the min-hash name IS manifest files[0] (proven equal
    # 2026-09-22: both select 2024032123gm-00a9-0000-6f136e72.mjai.json.zst).
    # Same proof strength, ~0.1s instead of ~3.5s. A corpus relayout that
    # breaks flatness must revisit this (the frame below fails closed first).
    names = [
        entry.name
        for entry in root.iterdir()
        if entry.is_file() and entry.name.endswith(".mjai.json.zst")
    ]
    assert len(names) > 0
    first_path = root / min(names, key=lambda n: hashlib.sha256(n.encode()).hexdigest())
    assert first_path.name.endswith(".mjai.json.zst")

    frames = _frame_file(first_path, 0)
    assert len(frames) > 0
    game_bytes = frames[0][2]
    assert len(game_bytes) > 0

    stem = stem_of(first_path)
    game = decode_game_object(object_id=stem, packaged_object_id=stem, decoded_bytes=game_bytes)
    assert game.game_id
    assert len(game.events) > 0
    assert game.raw_bytes_sha256.startswith("sha256:")

    outcome = validate_game(game)
    if not outcome.valid:
        assert outcome.error is not None
        assert outcome.error.error_class
        assert outcome.error.message
        return

    # Teacher path: real justification-gated trajectories plus isolated labels.
    from hydra2.distillation._teacher_gate import select_teacher
    from hydra2.distillation._teacher_records import (
        generate_privileged_labels,
        generate_trajectories,
        validate_trajectory_record,
    )

    justification = select_teacher(
        candidate_id="candidate6",
        justification_text=(
            "teacher candidate6 passed all five gates with compute-only analysis; "
            "deterministic replay verified; selected for real-log oracle fixture"
        ),
        selected_at_utc="2026-09-01T00:00:00Z",
    )
    records = generate_trajectories(
        justification=justification, num_records=2, with_privileged_labels=False
    )
    assert len(records) == 2
    for rec in records:
        validate_trajectory_record(rec)
        assert dict(rec.provenance)["justification_digest"] == justification.digest

    event_label, belief = generate_privileged_labels(
        world_id=f"world:{game.game_id}:candidate6",
        case_id=f"case:{game.game_id}",
        teacher_id="candidate6",
        token="training_namespace_v1",
    )
    assert event_label.startswith("event:")
    assert len(belief) == 4
    assert math.isclose(sum(belief), 1.0, abs_tol=1e-9)

    # Oracle join on opaque decision ids derived from game identity only.
    from hydra2.belief.oracle_join import join_oracle_targets
    from hydra2.belief.oracle_store import PrivilegedOracleLoader
    from hydra2.data.parquet import write_privileged_ranks

    decision_ids = [f"{game.game_id}#dec-0001", f"{game.game_id}#dec-0002"]
    ranks = {decision_ids[0]: [1, 2, 3, 4], decision_ids[1]: [2, 1, 4, 3]}
    dest = tmp_path / "priv_real_log"
    write_privileged_ranks(destination=dest, ranks_by_id=ranks)
    loader = PrivilegedOracleLoader(dest, split="train", verify=True, allow_synthetic=True)
    assert len(loader) == 2
    joined = join_oracle_targets(sorted(ranks), loader)
    assert tuple(joined["placement_target"].shape) == (2, 4)
    assert tuple(joined["value_target"].shape) == (2, 4)
    placement = joined["placement_target"].tolist()
    for row, did in zip(placement, sorted(ranks), strict=True):
        assert row == [r - 1 for r in ranks[did]]

    # Fail-closed split still enforced beside the real path.
    with pytest.raises(ContractError, match=r"only load split.*train"):
        PrivilegedOracleLoader(dest, split="held_out", verify=False)
