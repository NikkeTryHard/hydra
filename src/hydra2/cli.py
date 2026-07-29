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

DISPOSITION_EXIT_CODES = {
    "pass": 0,
    "blocked": 3,
    "fail": 4,
    "invalid": 2,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hydra2",
        description="Hydra2 research stack control plane (WP-01 bootstrap).",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    from hydra2.config import artifact_root
    from hydra2.config import repo_root as _repo_root

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "work-package" and args.work_package_command == "verify":
        root = Path(args.artifact_root).resolve() if args.artifact_root else artifact_root()
        # Portable repo_root default: marker walk (pyproject.toml/.git), not invocation-dir cwd.
        # Evidence: https://docs.python.org/3/library/pathlib.html#pathlib.Path.cwd (cwd is fragile)
        # Evidence: src/hydra2/config.py:90-108 repo_root() marker walk pattern
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
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
