"""GPU probe: environment manifest capture, canonical round trip, and
atomic publication.
"""

from __future__ import annotations

import json

import pytest

from hydra2._canon import canonical_json_bytes, sha256_digest_of_json
from hydra2.config import MAHJAX_PIN_SHA, PARITY_ABS_TOL, PARITY_REL_TOL
from hydra2.runtime.environment import (
    ENV_MANIFEST_ARTIFACT_TYPE,
    capture_environment_manifest,
    write_environment_manifest,
)


@pytest.mark.gpu
class TestEnvironmentManifest:
    def test_capture_fields_present(self, require_cuda):
        manifest, digest = capture_environment_manifest()
        assert manifest["artifact_type"] == ENV_MANIFEST_ARTIFACT_TYPE
        assert digest.startswith("sha256:")
        assert manifest["pixi_lock_sha256"].startswith("sha256:")
        assert manifest["python"]["implementation"] == "cpython"
        assert manifest["python"]["version"].startswith("3.12.")
        torch_block = manifest["torch"]
        assert torch_block["version"].startswith("2.13.0")
        assert torch_block["cuda"] is not None
        assert "sm_120" in torch_block["arch_list"], (
            f"Blackwell kernels missing from wheel: {torch_block['arch_list']}"
        )
        assert manifest["driver"]["nvidia_smi_available"] is True
        gpus = manifest["driver"]["gpus"]
        assert gpus and gpus[0]["name"].startswith("NVIDIA")
        assert gpus[0]["compute_capability"].startswith("12.")
        extensions = manifest["extensions"]
        assert extensions["lightning-fabric"] == "2.6.5"
        assert extensions["riichienv"] == "0.4.8"
        assert extensions["mahjax_pin_sha"] == MAHJAX_PIN_SHA

    def test_capture_is_deterministic(self, require_cuda):
        manifest_a, digest_a = capture_environment_manifest()
        manifest_b, digest_b = capture_environment_manifest()
        assert digest_a == digest_b
        assert canonical_json_bytes(manifest_a) == canonical_json_bytes(manifest_b)

    def test_canonical_round_trip(self, require_cuda):
        manifest, digest = capture_environment_manifest()
        reparsed = json.loads(canonical_json_bytes(manifest).decode("utf-8"))
        assert sha256_digest_of_json(reparsed) == digest
        # Key order in the original dict must not affect identity.
        shuffled = {k: manifest[k] for k in reversed(list(manifest))}
        assert sha256_digest_of_json(shuffled) == digest

    def test_atomic_publication(self, require_cuda, tmp_path):
        destination = tmp_path / "environment" / "environment-manifest.json"
        path, digest = write_environment_manifest(destination)
        raw = path.read_bytes()
        assert raw == canonical_json_bytes(json.loads(raw.decode("utf-8")))
        assert sha256_digest_of_json(json.loads(raw.decode("utf-8"))) == digest
        leftovers = [p.name for p in path.parent.iterdir() if ".tmp-" in p.name]
        assert leftovers == []

    def test_frozen_tolerance_constants_are_declared(self):
        assert 0 < PARITY_REL_TOL < 1e-3
        assert 0 <= PARITY_ABS_TOL <= PARITY_REL_TOL
