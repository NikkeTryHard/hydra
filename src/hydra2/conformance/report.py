"""WP-04A supported-rule intersection report.

Builds the ``rules_id x case -> supported | unsupported | mismatch`` matrix
from executed :class:`~hydra2.conformance.runner.CaseResult` rows and
publishes it atomically. Declared support of the WHOLE rules manifest demands
zero unresolved mismatches; any mismatch blocks the package.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from hydra2.artifacts.atomic import atomic_replace_bytes

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from hydra2.conformance.runner import CaseResult

__all__ = ["build_intersection_report", "write_intersection_report"]


def build_intersection_report(
    *,
    rules_id: str,
    rules_manifest_sha256: str,
    results: list[CaseResult],
    documented_unsupported: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Intersection matrix plus the package-level support verdict.

    ``documented_unsupported`` names rules ({rule_field: reason}) the reference
    engine provably does not implement. A mismatch case whose rule_fields are
    ALL documented-unsupported resolves to ``resolved-unsupported`` with its
    evidence chain retained; it no longer blocks declared support, but stays
    recorded for WP-04B admission decisions.
    """
    documented = dict(documented_unsupported) if documented_unsupported is not None else {}
    cases: dict[str, str] = {}
    resolutions: dict[str, dict[str, Any]] = {}
    for result in results:
        if result.case_id in cases and cases[result.case_id] != result.status:
            raise ValueError(f"conflicting statuses for case {result.case_id}")
        if (
            result.status == "mismatch"
            and len(result.rule_fields) != 0
            and all(field in documented for field in result.rule_fields)
        ):
            cases[result.case_id] = "resolved-unsupported"
            resolutions[result.case_id] = {
                "rule_fields": list(result.rule_fields),
                "reasons": {field: documented[field] for field in result.rule_fields},
                "evidence": list(result.evidence),
                "counterexample_path": result.counterexample_path,
                "error_detail": result.error_detail,
            }
    mismatches = sorted(cid for cid, status in cases.items() if status == "mismatch")
    resolved = sorted(cid for cid, status in cases.items() if status == "resolved-unsupported")
    blocked = sorted(cid for cid, status in cases.items() if status == "blocked")
    supported = sorted(
        cid for cid, status in cases.items() if status in ("supported", "resolved-unsupported")
    )
    rule_fields: dict[str, set[str]] = {}
    for result in results:
        for field_name in result.rule_fields:
            rule_fields.setdefault(field_name, set()).add(result.case_id)

    def field_status(field: str) -> str:
        statuses = {cases[cid] for cid in rule_fields[field]}
        if "mismatch" in statuses:
            return "mismatch"
        if statuses <= {"supported", "resolved-unsupported"} and (
            len(statuses & {"resolved-unsupported"}) != 0 or field in documented
        ):
            return "unsupported-for-reference-engine"
        if "blocked" in statuses:
            return "unsupported"
        return "supported"

    document: dict[str, Any] = {
        "artifact_type": "hydra2.wp04a_supported_rule_report",
        "schema_version": "1.0.0",
        "rules_id": rules_id,
        "rules_manifest_sha256": rules_manifest_sha256,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "adapter_compatibility_notes": {
            "reference_engine": "riichienv@0.4.8",
            "documented_unsupported_rules": dict(sorted(documented.items())),
            "resolution_policy": (
                "A mismatch case whose every rule field is documented-unsupported "
                "resolves to resolved-unsupported; evidence is retained and the "
                "package disposition may pass only when no unresolved mismatch "
                "remains. Data-admission consequences are deferred to WP-04B."
            ),
        },
        "matrix": {
            "by_case": {cid: cases[cid] for cid in sorted(cases)},
            "case_resolutions": {cid: resolutions[cid] for cid in sorted(resolutions)},
            "rule_field_intersections": {
                field: {
                    "cases": sorted(rule_fields[field]),
                    "status": field_status(field),
                }
                for field in sorted(rule_fields)
            },
        },
        "tally": {
            "supported": len(supported) - len([c for c in supported if c in resolutions]),
            "resolved_unsupported": len(resolved),
            "mismatch": len(mismatches),
            "blocked": len(blocked),
        },
        "unresolved_mismatch_cases": mismatches,
        "declared_support": {
            "whole_rules_manifest": len(mismatches) == 0,
            "verdict": "supported" if len(mismatches) == 0 else "blocked",
        },
    }
    return document


def write_intersection_report(document: dict[str, Any], destination: Path) -> str:
    """Atomically publish the report; returns its sha256 digest text."""
    import json

    from hydra2.artifacts.digest import sha256_digest  # noqa: F401

    payload = json.dumps(document, sort_keys=True, indent=1).encode()
    atomic_replace_bytes(destination, payload)
    from hydra2.artifacts.digest import of_bytes

    return str(of_bytes(payload))
