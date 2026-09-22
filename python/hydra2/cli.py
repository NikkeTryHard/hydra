"""``hydra2`` console entry point: streaming training runs.

Maps typed Hydra2Error subclasses to nonzero exit with a stable error class.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hydra2.contracts.common import Hydra2Error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hydra2",
        description="Hydra2 research stack control plane (training runs).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

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
    parser = build_parser()
    args = parser.parse_args(argv)

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
