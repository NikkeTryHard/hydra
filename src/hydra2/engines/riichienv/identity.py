"""Import-time engine identity pinning (BUILD WP-03A checklist item 1).

``import hydra2.engines.riichienv`` fails with :class:`UnsupportedRuleError`
unless the installed ``riichienv`` distribution is exactly the pinned 0.4.8
reference build (PROJECT_PLAN decision D-003).
"""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from typing import Any, cast

from hydra2.artifacts.digest import of_canonical
from hydra2.contracts.common import SchemaVersion, UnsupportedRuleError, make_schema_version
from hydra2.engines.protocol import EngineIdentity

__all__ = [
    "ADAPTER_VERSION",
    "ENGINE_IDENTITY",
    "ENGINE_NAME",
    "environment_document",
]

ENGINE_NAME = "riichienv"
#: PROJECT_PLAN D-003 pin; recon confirmed 0.4.8 is the latest release.
RIICHENV_VERSION_PIN = "0.4.8"
ADAPTER_VERSION = make_schema_version(SchemaVersion("1.0.0"))


def _source_revision() -> str | None:
    """VCS revision of the installed wheel when provenance metadata exists."""
    dist = importlib.metadata.distribution(ENGINE_NAME)
    direct_url: str | None = dist.read_text("direct_url.json")
    if direct_url is not None:
        try:
            import json

            document: dict[str, Any] = cast("dict[str, Any]", json.loads(direct_url))
        except ValueError:  # pragma: no cover - malformed metadata
            return None
        url: str = str(document.get("url", ""))
        vcs_info: Any = document.get("vcs_info", {})
        revision: Any = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
        if isinstance(revision, str) and revision != "":
            return revision
        if url != "":
            return url
    return None


def environment_document() -> dict[str, str]:
    """Engine-relevant environment facts (owner decision D-WP03A-2).

    Deliberately machine-stable: Python implementation/version and platform
    class plus the pinned engine version. GPU/driver facts are excluded on
    purpose - the reference adapter is CPU-only simulation.
    """
    return {
        "python_implementation": sys.implementation.name,
        "python_version": ".".join(str(p) for p in sys.version_info[:3]),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "engine_name": ENGINE_NAME,
        "engine_version": RIICHENV_VERSION_PIN,
    }


def _verify_pinned_version() -> str:
    version = importlib.metadata.version(ENGINE_NAME)
    if version != RIICHENV_VERSION_PIN:
        raise UnsupportedRuleError(
            f"hydra2.engines.riichienv pins RiichiEnv=={RIICHENV_VERSION_PIN} "
            f"(decision D-003); installed distribution reports {version!r}"
        )
    return version


_version = _verify_pinned_version()

ENGINE_IDENTITY = EngineIdentity(
    name=ENGINE_NAME,
    version=_version,
    adapter_version=ADAPTER_VERSION,
    source_revision=_source_revision(),
    environment_hash=of_canonical(environment_document()),
)
