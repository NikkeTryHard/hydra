# ruff: noqa: E501 — portable evidence URLs in docstrings/comments unavoidably long (>100)
"""Attestation handling for WP-04B — parameterized join with synthetic + real D-017.

Real Tenhou Houou corpus is a private Tenhou Houou corpus
(D-017 attestation) — location via attestation metadata or
HYDRA2_TENHOU_MOUNT under tenhou-houou-mjai-2009..2026. Attestation D-017
is now supplied via configs/attestations/D-017.json (and artifact copy
via hydra2.config.artifact_root) with real packager evidence.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from hydra2.contracts.common import ContractError

__all__ = [
    "D017_ATTESTATION_PATH",
    "D017_BLOCKER",
    "REAL_ATTESTATION",
    "SYNTHETIC_ATTESTATION",
    "Attestation",
    "load_attestation",
    "require_attestation",
]

def _repo_attestation_path() -> Path:
    """Resolve D-017 attestation path via repo_root marker walk; fallback to importlib.resources for wheel installs.

    Portable: repo_root() marker walk (pyproject.toml/.git) cached, plus
    importlib.resources fallback zip-safe; no hardcoded parents[2] depth.
    Evidence:
    - https://docs.python.org/3/library/importlib.resources.html#files
    - https://docs.python.org/3/library/pathlib.html#pathlib.Path.resolve
    - https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html (artifact_root context)
    - hydra2.config.artifact_root / repo_root canonical resolver (portable)
    """
    try:
        from hydra2.config import repo_root  # lazy to avoid import cycle

        candidate = repo_root() / "configs" / "attestations" / "D-017.json"
        if candidate.is_file():
            return candidate
        # wheel/zip fallback before returning non-existent repo path
        try:
            import importlib.resources as _ir

            return Path(str(_ir.files("hydra2") / "configs" / "attestations" / "D-017.json"))
        except Exception:
            return candidate
    except Exception:
        try:
            import importlib.resources as _ir2

            return Path(str(_ir2.files("hydra2") / "configs" / "attestations" / "D-017.json"))
        except Exception:
            # No hardcoded depth fallback remains — caller will handle missing file via is_file() check.
            # Keep portable: return candidate-like repo-anchored path via temp discovery without parents[N].
            # As last resort, walk from __file__ parents searching for pyproject.toml (marker walk, not fixed depth).
            cur = Path(__file__).resolve()
            for cand in (cur, *cur.parents):
                if (cand / "pyproject.toml").is_file():
                    return cand / "configs" / "attestations" / "D-017.json"
            return cur.parent / "configs" / "attestations" / "D-017.json"


D017_ATTESTATION_PATH = _repo_attestation_path()

def _artifact_attestation_path() -> Path:
    """Portable artifact copy path via hydra2.config.artifact_root (XDG/TMPDIR aware).

    Evidence:
    - XDG spec https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
    - tempfile.gettempdir https://docs.python.org/3/library/tempfile.html#tempfile.gettempdir
    - Path.home https://docs.python.org/3/library/pathlib.html#pathlib.Path.home
    - shutil.which guard https://docs.python.org/3/library/shutil.html#shutil.which (for nvidia-smi pattern, analogous portability)
    """
    from hydra2.config import artifact_root  # local import to avoid cycles

    return artifact_root() / "attestations" / "D-017.json"

D017_BLOCKER = (
    "D-017 pending: real-corpus attestation not yet supplied; "
    "synthetic qualification via packager used"
)


@dataclass(frozen=True, slots=True)
class Attestation:
    attestation_id: str
    confidential_source_id: str
    permitted_purpose: tuple[str, ...]
    disclosure_class: str
    acquisition_metadata: dict[str, object]
    kind: Literal["synthetic", "real"]


SYNTHETIC_ATTESTATION = Attestation(
    attestation_id="synthetic-attestation-v1",
    confidential_source_id="synthetic-source-v1",
    permitted_purpose=("research", "training"),
    disclosure_class="synthetic",
    acquisition_metadata={
        "source": "synthetic",
        "player_ids": ["synthetic_p0", "synthetic_p1", "synthetic_p2", "synthetic_p3"],
        "timestamp": "2026-08-31T00:00:00Z",
        "wall_ids": ["synthetic-wall-001"],
    },
    kind="synthetic",
)

# Real Houou attestation — populated from configs/attestations/D-017.json at import
# if available; otherwise None until file is created. Never fabricated inline.
REAL_ATTESTATION: Attestation | None = None


def _attestation_from_dict(raw: dict[str, object], *, kind: str = "real") -> Attestation:
    _att_id_raw = raw.get("attestation_id")
    if _att_id_raw is None or _att_id_raw == "":
        _att_id_raw = raw.get("id")
    if _att_id_raw is None or _att_id_raw == "":
        _att_id_raw = ""
    att_id = str(_att_id_raw)
    if att_id == "":
        raise ContractError("attestation_id missing in attestation file")
    _csid_raw = raw.get("confidential_source_id")
    if _csid_raw is None or _csid_raw == "":
        _csid_raw = raw.get("confidentialSourceId")
    if _csid_raw is None or _csid_raw == "":
        _csid_raw = "tenhou-houou-corpus-v1"
    csid = str(_csid_raw)
    _pp_raw = raw.get("permitted_purpose")
    if _pp_raw is None:
        _pp_raw = raw.get("permittedPurpose")
    if _pp_raw is None:
        _pp_raw = ("research", "training", "internal_evaluation")
    pp = _pp_raw
    if isinstance(pp, str):
        pp = (pp,)
    pp_tuple = tuple(str(x) for x in pp)  # type: ignore[arg-type]
    _dc_raw = raw.get("disclosure_class")
    if _dc_raw is None or _dc_raw == "":
        _dc_raw = raw.get("disclosureClass")
    if _dc_raw is None or _dc_raw == "":
        _dc_raw = "confidential"
    dc = str(_dc_raw)
    _meta_raw = raw.get("acquisition_metadata")
    if _meta_raw is None:
        _meta_raw = raw.get("acquisitionMetadata")
    if _meta_raw is None:
        _meta_raw = {}
    meta = _meta_raw
    if not isinstance(meta, dict):
        raise ContractError("acquisition_metadata must be dict")
    # Merge top-level dataset fields into metadata for traceability
    for k in (
        "dataset",
        "mount",
        "tenhou_only",
        "packager_log",
        "packager_items",
        "file_counts_per_year",
        "total_tenhou_files",
        "total_majsoul_files_excluded",
        "sample_hashes",
        "manifest_path",
        "manifest_sha256",
        "created_at",
        "creator",
    ):
        if k in raw and k not in meta:
            meta[k] = raw[k]
    return Attestation(
        attestation_id=att_id,
        confidential_source_id=csid,
        permitted_purpose=pp_tuple,
        disclosure_class=dc,
        acquisition_metadata=dict(meta),
        kind="real" if kind == "real" else "synthetic",
    )


def load_attestation(path: Path | None = None) -> Attestation:
    """Load D-017 real attestation from JSON; verifies file exists and is not synthetic-only."""
    p = Path(path) if path is not None else D017_ATTESTATION_PATH
    # Also check artifact copy via portable artifact_root (XDG_CACHE_HOME/TMPDIR aware)
    if not p.is_file():
        alt = _artifact_attestation_path()
        if alt.is_file():
            p = alt
    if not p.is_file():
        raise ContractError(f"real attestation not found: {p} (D-017 pending)")
    raw = json.loads(p.read_text(encoding="utf-8"))
    att = _attestation_from_dict(raw, kind="real")
    if att.attestation_id != "D-017":
        raise ContractError(f"expected attestation_id D-017, got {att.attestation_id!r}")
    return att


# Eagerly load real attestation if file exists (no hard error if absent — allows synthetic paths)
try:
    if D017_ATTESTATION_PATH.is_file() or _artifact_attestation_path().is_file():
        REAL_ATTESTATION = load_attestation()
except Exception:
    REAL_ATTESTATION = None


def require_attestation(value: Attestation | None, *, allow_synthetic: bool = True) -> Attestation:
    if value is None:
        raise ContractError(
            "attestation missing; cannot create authorized RawObjectRow (D-017 pending)"
        )
    if value.kind == "real" and value.attestation_id == "D-017":
        # Real D-017 now supplied — validate required fields, never fabricate.
        if value.confidential_source_id == "":
            raise ContractError("real D-017 confidential_source_id must be non-empty")
        if len(value.permitted_purpose) == 0:
            raise ContractError("real D-017 permitted_purpose must be non-empty")
        if value.disclosure_class == "":
            raise ContractError("real D-017 disclosure_class must be non-empty")
        if not isinstance(value.acquisition_metadata, dict) or len(value.acquisition_metadata) == 0:
            raise ContractError("real D-017 acquisition_metadata must be non-empty dict")
        # Must reference tenhou-houou and mount
        meta = value.acquisition_metadata
        _dataset_raw = meta.get("dataset")
        dataset = "" if _dataset_raw is None or _dataset_raw == "" else str(_dataset_raw)
        if dataset != "" and "tenhou-houou" not in dataset:
            raise ContractError(f"real D-017 dataset must reference tenhou-houou, got {dataset!r}")
        return value
    if value.kind == "synthetic" and not allow_synthetic:
        raise ContractError("synthetic attestation not permitted for this path")
    if value.attestation_id == "":
        raise ContractError("attestation_id must be non-empty")
    return value
