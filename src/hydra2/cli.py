"""``hydra2`` console entry point.

Subcommands:
  work-package verify <WP-ID> --artifact-root ROOT [--repo-root PATH]

Maps typed Hydra2Error subclasses to nonzero exit with a stable error class.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hydra2.completion import format_outcome, verify_work_package
from hydra2.contracts.common import Hydra2Error

# Disposition: pass (0), invalid-or-error (2), blocked (3), fail (4).
DISPOSITION_EXIT_CODES = {
    "pass": 0,
    "blocked": 3,
    "fail": 4,
    "invalid": 2,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hydra2",
        description="Hydra2 research stack control plane (work-package registry and "
        "training runs).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    work_package = subparsers.add_parser(
        "work-package", help="work-package completion record registry"
    )
    wp_sub = work_package.add_subparsers(dest="work_package_command", required=True)
    verify = wp_sub.add_parser("verify", help="verify a completion record and sync the index")
    _ = verify.add_argument("wp_id", help="work package identifier, e.g. WP-01")
    _ = verify.add_argument(
        "--artifact-root",
        required=False,
        default=None,
        help="artifact root (defaults to $HYDRA2_ARTIFACT_ROOT)",
    )
    _ = verify.add_argument(
        "--repo-root",
        required=False,
        default=None,
        help="repository root for repo-relative input/output hashes (default: repo root auto via marker walk)",  # noqa: E501
    )
    train = subparsers.add_parser(
        "train", help="streaming-first training run (plan, resume, dry-run)"
    )
    _ = train.add_argument(
        "config",
        help="path to training run YAML (see configs/training/example.yaml)",
    )
    _ = train.add_argument(
        "--dry-run",
        action="store_true",
        help="parse, validate, and print the resolved plan without mutating artifacts",
    )
    _ = train.add_argument(
        "--resume",
        required=False,
        default=None,
        help="'latest' for the latest compatible checkpoint in the run dir, or a path to ckpt-*.pt",
    )
    _ = train.add_argument(
        "--mlflow",
        dest="mlflow",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override YAML telemetry.mlflow_enabled (default: honor YAML)",
    )
    _ = train.add_argument(
        "--verbose-telemetry",
        dest="verbose_telemetry",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override YAML telemetry.verbose_enabled (default: honor YAML)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from hydra2.config import artifact_root
    from hydra2.config import repo_root as _repo_root

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "work-package" and args.work_package_command == "verify":
        # Precedence: --artifact-root flag > $HYDRA2_ARTIFACT_ROOT >
        # _default_artifact_root() chain; env re-read per call so late
        # mutations (conftest, pixi wrappers) apply — see
        # hydra2.config.artifact_root.
        root = Path(args.artifact_root).resolve() if args.artifact_root else artifact_root()
        # Portable repo_root default: marker walk (pyproject.toml/.git), not invocation-dir cwd.
        # Evidence: https://docs.python.org/3/library/pathlib.html#pathlib.Path.cwd (cwd is fragile)
        # Evidence: hydra2.config.repo_root/_find_repo_root marker walk (pyproject.toml/.git)
        repo_root = Path(args.repo_root).resolve() if args.repo_root else _repo_root()
        try:
            outcome = verify_work_package(args.wp_id, artifact_root=root, repo_root=repo_root)
        except Hydra2Error as exc:
            print(f"error[{type(exc).__name__}]: {exc}", file=sys.stderr)
            return 2
        print(format_outcome(outcome))
        code = DISPOSITION_EXIT_CODES.get(outcome.disposition, 2)
        if outcome.disposition == "invalid":
            for error in outcome.errors:
                print(f"error: {error}", file=sys.stderr)
        elif outcome.disposition == "blocked" and len(outcome.errors) == 0:
            # Blocked is a recorded, legitimate disposition; keep exit distinct
            # from pass but surface the blockers on stderr for visibility.
            blockers = (
                outcome.blockers
                if outcome.blockers is not None and len(outcome.blockers) != 0
                else ["record declares blocked status"]
            )
            for blocker in blockers:
                print(f"blocked: {blocker}", file=sys.stderr)
        return code
    if args.command == "train":
        from hydra2.training._rc_digest import create_run_layout, run_dir_for
        from hydra2.training._rc_resume import format_plan, resolve_resume_plan
        from hydra2.training._rc_root import load_run_config

        try:
            config = load_run_config(Path(args.config))
        except Hydra2Error as exc:
            print(f"error[{type(exc).__name__}]: {exc}", file=sys.stderr)
            return 2
        # Tri-state CLI flags override YAML telemetry knobs (None honors YAML).
        if args.mlflow is not None or args.verbose_telemetry is not None:
            from dataclasses import replace as _dc_replace

            telemetry = config.telemetry
            if args.mlflow is not None:
                telemetry = _dc_replace(telemetry, mlflow_enabled=args.mlflow)
            if args.verbose_telemetry is not None:
                telemetry = _dc_replace(telemetry, verbose_enabled=args.verbose_telemetry)
            config = _dc_replace(config, telemetry=telemetry)
        # Dry-run reads no data and mutates nothing: the plan is pure
        # config resolution (operator edits YAML by hand when it is wrong).
        if args.dry_run:
            resume = None
            if args.resume is not None:
                try:
                    run_dir = run_dir_for(config)
                    resume = resolve_resume_plan(
                        run_dir, which=args.resume if args.resume != "latest" else "latest"
                    )
                except Hydra2Error as exc:
                    print(f"error[{type(exc).__name__}]: {exc}", file=sys.stderr)
                    return 2
            print(format_plan(config, resume=resume), end="")
            return 0
        try:
            run_dir = create_run_layout(config)
            resume = None
            if args.resume is not None:
                resume = resolve_resume_plan(
                    run_dir, which=args.resume if args.resume != "latest" else "latest"
                )
        except Hydra2Error as exc:
            print(f"error[{type(exc).__name__}]: {exc}", file=sys.stderr)
            return 2
        print(format_plan(config, resume=resume), end="")
        from hydra2.training.stream_train import run_stream_training

        try:
            summary = run_stream_training(config, resume)
        except Hydra2Error as exc:
            print(f"error[{type(exc).__name__}]: {exc}", file=sys.stderr)
            return 2
        print(
            f"train: updates {summary['start_update']}->{summary['end_update']} "
            f"train_games={summary['train_games']} val_games={summary['val_games']} "
            f"quarantined={summary['quarantined']} "
            f"replayed={summary.get('replayed', 0)} "
            f"sim_replayed={summary.get('sim_replayed', 0)} "
            f"expand_quarantined={summary.get('expand_quarantined', 0)} "
            f"checkpoints={','.join(summary['checkpoints'])}"
        )
        return 0
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
