from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from flax.traverse_util import flatten_dict
from flax.traverse_util import unflatten_dict


def export_weights(params, output_prefix: Path) -> tuple[Path, Path]:
    flat = flatten_dict(params, sep="/")
    archive_arrays = {
        name.replace("/", "__"): np.asarray(value) for name, value in flat.items()
    }
    npz_path = output_prefix.with_suffix(".npz")
    json_path = output_prefix.with_suffix(".json")
    np.savez(npz_path, **dict(archive_arrays))
    metadata = {
        "schema_version": "hydra_phase0_weight_export_v1",
        "tensors": {
            name: {
                "archive_key": name.replace("/", "__"),
                "shape": list(np.asarray(value).shape),
                "dtype": str(np.asarray(value).dtype),
            }
            for name, value in flat.items()
        },
    }
    json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return npz_path, json_path


def load_exported_weights(metadata_path: Path, archive_path: Path):
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("schema_version") != "hydra_phase0_weight_export_v1":
        raise ValueError(
            f"unsupported weight export schema_version {metadata.get('schema_version')}"
        )
    archive = np.load(archive_path)
    expected_archive_keys: set[str] = set()
    flat: dict[tuple[str, ...], np.ndarray] = {}
    for name, spec in metadata["tensors"].items():
        archive_key = spec["archive_key"]
        if archive_key in expected_archive_keys:
            raise ValueError(f"duplicate archive key {archive_key} in {metadata_path}")
        expected_archive_keys.add(archive_key)
        if archive_key not in archive.files:
            raise ValueError(f"missing archive entry {archive_key} in {archive_path}")
        value = archive[archive_key]
        if list(value.shape) != list(spec["shape"]):
            raise ValueError(
                f"shape mismatch for {name}: expected {spec['shape']}, got {list(value.shape)}"
            )
        actual_dtype = str(value.dtype)
        if actual_dtype != spec["dtype"]:
            raise ValueError(
                f"dtype mismatch for {name}: expected {spec['dtype']}, got {actual_dtype}"
            )
        flat[tuple(name.split("/"))] = value
    archive_files = set(archive.files)
    unexpected = archive_files - expected_archive_keys
    if unexpected:
        raise ValueError(
            f"unexpected archive entries in {archive_path}: {sorted(unexpected)}"
        )
    return unflatten_dict(flat)
