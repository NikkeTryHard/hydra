#!/usr/bin/env python3
"""
Encrypt .tar.zst archives with AES-256-GCM.

Format: [12-byte nonce][ciphertext + 16-byte tag]

Key: 32-byte hex string from VOCNET_KEY env var.
Generates a new key if --genkey is passed.

Usage:
    # Generate a key
    python encrypt_archives.py --genkey

    # Encrypt all archives in a directory
    VOCNET_KEY=<hex> python encrypt_archives.py /path/to/archives/ /path/to/output/

    # Decrypt and verify one file (test round-trip)
    VOCNET_KEY=<hex> python encrypt_archives.py --verify /path/to/file.tar.zst.enc
"""

import os
import sys
import struct
from pathlib import Path
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def genkey():
    key = os.urandom(32)
    print(key.hex())


def encrypt_file(src: Path, dst: Path, key: bytes):
    data = src.read_bytes()
    nonce = os.urandom(12)
    aes = AESGCM(key)
    ct = aes.encrypt(nonce, data, None)
    dst.write_bytes(nonce + ct)
    return len(data), len(nonce + ct)


def decrypt_file(src: Path, key: bytes) -> bytes:
    data = src.read_bytes()
    nonce = data[:12]
    ct = data[12:]
    aes = AESGCM(key)
    return aes.decrypt(nonce, ct, None)


def get_key() -> bytes:
    key_hex = os.environ.get("VOCNET_KEY", "").strip()
    if not key_hex or len(key_hex) != 64:
        print(
            "ERROR: VOCNET_KEY env var must be 64 hex chars (32 bytes)", file=sys.stderr
        )
        sys.exit(1)
    return bytes.fromhex(key_hex)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == "--genkey":
        genkey()
        return

    if sys.argv[1] == "--verify":
        key = get_key()
        path = Path(sys.argv[2])
        pt = decrypt_file(path, key)
        print(f"OK: {path.name} -> {len(pt):,} bytes plaintext")
        if path.name.endswith(".tar.zst.enc"):
            import zstandard

            dctx = zstandard.ZstdDecompressor()
            decompressed = dctx.decompress(pt, max_output_size=len(pt) * 10)
            print(f"  zstd decompressed: {len(decompressed):,} bytes")
        return

    key = get_key()
    src_dir = Path(sys.argv[1])
    dst_dir = Path(sys.argv[2])
    dst_dir.mkdir(parents=True, exist_ok=True)

    archives = sorted(src_dir.glob("*.tar.zst"))
    if not archives:
        print(f"No .tar.zst files found in {src_dir}")
        sys.exit(1)

    print(f"Encrypting {len(archives)} archives...")
    total_plain = 0
    total_enc = 0

    for i, src in enumerate(archives, 1):
        dst = dst_dir / (src.name + ".enc")
        plain_sz, enc_sz = encrypt_file(src, dst, key)
        total_plain += plain_sz
        total_enc += enc_sz
        print(
            f"  [{i}/{len(archives)}] {src.name} ({plain_sz / 1e6:.1f}MB -> {enc_sz / 1e6:.1f}MB)"
        )

    print(f"\nDone. {total_plain / 1e9:.2f}GB -> {total_enc / 1e9:.2f}GB")
    print(f"Overhead: {(total_enc - total_plain) / 1e3:.1f}KB (nonces + tags)")


if __name__ == "__main__":
    main()
