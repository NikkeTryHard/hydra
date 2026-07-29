"""ART-ATOMIC-001: atomic publication and interrupted-write safety (SPEC 2.3)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

import hydra2.artifacts.atomic as atomic_module
from hydra2.artifacts.atomic import atomic_replace_bytes, publish_atomic
from hydra2.artifacts.digest import of_bytes, sha256_digest, sha256_file
from hydra2.contracts.common import ContractError, DigestMismatchError

pytestmark = pytest.mark.contract_package("WP-02A")


class TestPublishAtomicHappyPath:
    def test_publishes_exact_bytes_and_cleans_temp(self, tmp_path):
        destination = tmp_path / "artifact.json"
        data = b'{"a":1}'
        publish_atomic(destination=destination, data=data, expected=sha256_digest(data))
        assert destination.read_bytes() == data
        assert [p.name for p in tmp_path.iterdir()] == ["artifact.json"]

    def test_identical_republish_keeps_original_inode(self, tmp_path):
        destination = tmp_path / "artifact.bin"
        data = b"stable-bytes"
        publish_atomic(destination=destination, data=data, expected=of_bytes(data))
        first = destination.stat().st_ino
        publish_atomic(destination=destination, data=data, expected=of_bytes(data))
        assert destination.stat().st_ino == first  # original bytes never replaced
        assert destination.read_bytes() == data

    def test_data_digest_mismatch_rejected_before_touching_filesystem(self, tmp_path):
        destination = tmp_path / "x.bin"
        with pytest.raises(DigestMismatchError):
            publish_atomic(destination=destination, data=b"abc", expected=of_bytes(b"zzz"))
        assert not destination.exists()

    def test_missing_parent_rejected(self, tmp_path):
        with pytest.raises(ContractError):
            publish_atomic(
                destination=tmp_path / "nope" / "x.bin",
                data=b"a",
                expected=of_bytes(b"a"),
            )

    def test_symlinked_parent_rejected(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        os.symlink(real, link)
        with pytest.raises(ContractError):
            publish_atomic(destination=link / "x.bin", data=b"a", expected=of_bytes(b"a"))

    def test_symlinked_destination_rejected(self, tmp_path):
        victim = tmp_path / "victim.txt"
        victim.write_bytes(b"victim")
        link = tmp_path / "alias.bin"
        os.symlink(victim, link)
        with pytest.raises(ContractError):
            publish_atomic(destination=link, data=b"a", expected=of_bytes(b"a"))
        assert victim.read_bytes() == b"victim"


class TestOverwriteProtection:
    def test_existing_different_content_is_hard_failure(self, tmp_path):
        destination = tmp_path / "immutable.json"
        publish_atomic(destination=destination, data=b"original", expected=of_bytes(b"original"))
        with pytest.raises(DigestMismatchError):
            publish_atomic(destination=destination, data=b"attack", expected=of_bytes(b"attack"))
        assert destination.read_bytes() == b"original"

    def test_concurrent_appearance_of_foreign_content_loses(self, tmp_path, monkeypatch):
        # Simulate a racing writer creating the destination between the temp
        # write and the no-clobber link: link then raises FileExistsError.
        destination = tmp_path / "race.json"
        data = b"winner"
        real_link = os.link

        def racing_link(src, dst, *args, **kwargs):
            dst_path = Path(dst)
            if not dst_path.exists():
                dst_path.write_bytes(b"interloper")
            return real_link(src, dst, *args, **kwargs)

        monkeypatch.setattr(os, "link", racing_link)
        with pytest.raises(DigestMismatchError):
            publish_atomic(destination=destination, data=data, expected=of_bytes(data))
        leftovers = [p for p in tmp_path.iterdir() if ".tmp-" in p.name]
        assert leftovers == []


class TestInterruptedPublication:
    """ART-ATOMIC-001: failure between write and rename publishes nothing."""

    def test_link_failure_leaves_no_destination_and_no_temp(self, tmp_path, monkeypatch):
        destination = tmp_path / "interrupted.json"
        data = b"payload"

        def broken_link(*args, **kwargs):
            raise OSError("injected interruption before rename")

        monkeypatch.setattr(os, "link", broken_link)
        with pytest.raises(OSError, match="injected interruption"):
            publish_atomic(destination=destination, data=data, expected=of_bytes(data))
        assert not destination.exists()
        assert list(tmp_path.iterdir()) == []

    def test_fsync_failure_during_write_leaves_no_destination(self, tmp_path, monkeypatch):
        destination = tmp_path / "interrupted2.json"
        data = b"payload"

        def broken_fsync(fd):
            raise OSError("injected fsync failure")

        monkeypatch.setattr(os, "fsync", broken_fsync)
        with pytest.raises(OSError, match="injected fsync"):
            publish_atomic(destination=destination, data=data, expected=of_bytes(data))
        assert not destination.exists()
        assert list(tmp_path.iterdir()) == []

    def test_late_directory_fsync_failure_never_removes_destination(self, tmp_path, monkeypatch):
        destination = tmp_path / "late.json"
        data = b"durable"
        monkeypatch.setattr(
            atomic_module, "_fsync_dir", lambda directory: (_ for _ in ()).throw(OSError("late"))
        )
        with pytest.raises(OSError, match="late"):
            publish_atomic(destination=destination, data=data, expected=of_bytes(data))
        # Publication itself completed; the destination MUST survive any error.
        assert destination.read_bytes() == data
        assert [p.name for p in tmp_path.iterdir()] == ["late.json"]


class TestIndependentDigestRecomputation:
    def test_chunked_file_hash_matches_in_memory_hash_across_chunk_boundaries(self, tmp_path):
        payload = bytes(range(256)) * ((1 << 20) * 2 // 256 + 1) + b"\x13tail"
        path = tmp_path / "big.bin"
        atomic_replace_bytes(path, payload)
        assert sha256_file(path) == sha256_digest(payload)
        assert of_bytes(payload) == "sha256:" + hashlib.sha256(payload).hexdigest()

    def test_empty_input_digest_is_known_vector(self):
        assert of_bytes(b"") == (
            "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )


class TestAtomicReplaceMutable:
    def test_replace_overwrites_and_cleans_temp(self, tmp_path):
        target = tmp_path / "index.json"
        atomic_replace_bytes(target, b"v1")
        atomic_replace_bytes(target, b"v2-longer")
        assert target.read_bytes() == b"v2-longer"
        assert [p.name for p in tmp_path.iterdir()] == ["index.json"]
