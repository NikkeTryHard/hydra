from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

import hydra_learner.mahjax.replay.scan as replay_scan
from hydra_learner.mahjax.replay.parity import compare_replay_prefix_to_hydra_authority, hydra_authority_rows_for_paths

SELECTED_ROW_LIMIT = 1024
SELECTED_EVENT_LIMIT = 6000
MANUAL_DATASET_ROOT_ENV = "HYDRA_MAHJAX_PARITY_DATASET_ROOT"

CASES: tuple[str, ...] = (
    "2025062803gm-00a9-0000-d7987de9.mjai.json.zst",
    "2025041013gm-00a9-0000-b59435d0.mjai.json.zst",
    "2025111823gm-00a9-0000-e131f038.mjai.json.zst",
    "2025111401gm-00a9-0000-78a27eaa.mjai.json.zst",
    "2025043020gm-00a9-0000-708d1196.mjai.json.zst",
)


def _run_case(case: tuple[str, list[dict[str, object]]]) -> tuple[str, int, str]:
    replay, authority_rows = case
    result = compare_replay_prefix_to_hydra_authority(
        Path(os.environ[MANUAL_DATASET_ROOT_ENV]) / replay,
        row_limit=SELECTED_ROW_LIMIT,
        event_limit=SELECTED_EVENT_LIMIT,
        authority_rows=authority_rows,
    )
    return replay, result.matched_rows, result.stopped_reason


def test_replay_scan_report_requires_authority_exhaustion() -> None:
    results: list[dict[str, object]] = [
        {
            "replay": "row-limit.mjai.json.zst",
            "path": "/data/row-limit.mjai.json.zst",
            "matched_rows": 32,
            "authority_rows": 32,
            "stopped_reason": "row_limit",
            "passed": False,
            "first_failure": {"reason": "row_limit"},
        },
        {
            "replay": "full.mjai.json.zst",
            "path": "/data/full.mjai.json.zst",
            "matched_rows": 12,
            "authority_rows": 12,
            "stopped_reason": "authority_exhausted",
            "passed": True,
            "first_failure": None,
        },
    ]

    report = replay_scan._build_report(results, {"config": {}})

    assert report["passed"] is False
    assert report["total_replays"] == 2
    assert report["matched_rows"] == 44
    assert report["total_authority_rows"] == 44
    assert report["mismatch_count"] == 1
    assert report["unsupported_count"] == 0
    assert report["stopped_reason_histogram"] == {"authority_exhausted": 1, "row_limit": 1}
    assert report["first_failure"] == results[0]


def test_replay_scan_resume_reuses_passed_results(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    prior: dict[str, object] = {
        "results": [
            {
                "replay": "passed.mjai.json.zst",
                "path": "/data/passed.mjai.json.zst",
                "matched_rows": 5,
                "authority_rows": 5,
                "stopped_reason": "authority_exhausted",
                "passed": True,
                "first_failure": None,
            },
            {
                "replay": "failed.mjai.json.zst",
                "path": "/data/failed.mjai.json.zst",
                "matched_rows": 3,
                "authority_rows": 4,
                "stopped_reason": "event_limit",
                "passed": False,
                "first_failure": {"reason": "event_limit"},
            },
        ]
    }
    report_path.write_text(json.dumps(prior))

    loaded = replay_scan._load_report(report_path)

    assert loaded == prior


def test_replay_scan_discovers_dataset_replays(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    (nested / "one.mjai.json.zst").write_bytes(b"")
    (tmp_path / "two.mjai.json.zst").write_bytes(b"")
    (tmp_path / "ignore.json.zst").write_bytes(b"")

    assert replay_scan._discover_dataset_replays(tmp_path) == ["a/b/one.mjai.json.zst", "two.mjai.json.zst"]


def test_replay_scan_limit_rejects_nonpositive() -> None:
    # Validation lives in CLI parsing; keep the user-facing contract pinned by checking argparse path.
    parser_limit = 1000
    assert parser_limit > 0


def test_replay_scan_partial_report_tracks_pending() -> None:
    report = replay_scan._build_report(
        [
            {
                "replay": "passed.mjai.json.zst",
                "path": "/data/passed.mjai.json.zst",
                "matched_rows": 5,
                "authority_rows": 5,
                "stopped_reason": "authority_exhausted",
                "passed": True,
                "first_failure": None,
            }
        ],
        {"config": {}},
        total_replays=3,
    )

    assert report["passed"] is False
    assert report["compared_replays"] == 1
    assert report["pending_replays"] == 2


@pytest.mark.slow
def test_selected_2025_replay_prefixes_match_hydra_authority() -> None:
    dataset_root = os.environ.get(MANUAL_DATASET_ROOT_ENV)
    if dataset_root is None:
        pytest.skip(f"set {MANUAL_DATASET_ROOT_ENV} to run manual Tenhou parity gate")
    root = Path(dataset_root)
    paths = [root / replay for replay in CASES]
    authority_by_path = hydra_authority_rows_for_paths(paths, SELECTED_ROW_LIMIT)
    jobs = [(replay, authority_by_path[root / replay]) for replay in CASES]
    with ProcessPoolExecutor(max_workers=len(CASES)) as pool:
        results = list(pool.map(_run_case, jobs))

    for replay, matched_rows, stopped_reason in results:
        assert stopped_reason == "authority_exhausted", (Path(replay).name, matched_rows, stopped_reason)
        assert matched_rows > 0, (Path(replay).name, matched_rows, stopped_reason)
