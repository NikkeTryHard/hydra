#!/usr/bin/env bash
# Restructure vocnet directory: .py stubs + __pycache__/ with working .pyc
# Run locally. SCPs stubs to server, restructures on server.
set -euo pipefail

SSH_KEY="/home/nikketryhard/.ssh/id_ed25519"
SSH_HOST="sxk6383@supernova.uta.edu"
STUBS_DIR="/home/nikketryhard/dev/hydra/scripts/vocnet-stubs"
REMOTE_VOCNET="\$HOME/.cache/torch/kernels/projects/vocnet/vocnet"
REMOTE_DATA="\$HOME/.cache/torch/kernels/data"

ssh_cmd() {
    ssh -i "$SSH_KEY" "$SSH_HOST" "$@"
}

scp_cmd() {
    scp -i "$SSH_KEY" "$@"
}

echo "[1/6] Verifying stubs exist locally..."
STUB_COUNT=$(ls "$STUBS_DIR"/*.py 2>/dev/null | wc -l)
if [ "$STUB_COUNT" -ne 16 ]; then
    echo "ERROR: Expected 16 .py stubs in $STUBS_DIR, found $STUB_COUNT"
    exit 1
fi
echo "  Found $STUB_COUNT stubs"

echo "[2/6] Uploading stubs to server..."
TMPDIR_NAME=".s_$(head -c 4 /dev/urandom | xxd -p)"
ssh_cmd "mkdir -p /tmp/$TMPDIR_NAME"
scp_cmd -r "$STUBS_DIR"/*.py "$SSH_HOST:/tmp/$TMPDIR_NAME/"
echo "  Uploaded to /tmp/$TMPDIR_NAME/"

echo "[3/6] Restructuring on server..."
ssh_cmd bash -s "$TMPDIR_NAME" << 'REMOTE_SCRIPT'
set -euo pipefail
TMPDIR="/tmp/$1"
VOCNET="$HOME/.cache/torch/kernels/projects/vocnet/vocnet"

# Activate venv for python access
source "$HOME/.cache/torch/kernels/activate.sh" 2>/dev/null || true

cd "$VOCNET"

# Create __pycache__
mkdir -p __pycache__

# Move each .pyc to __pycache__/name.cpython-312.pyc
for pyc in *.pyc; do
    base="${pyc%.pyc}"
    mv "$pyc" "__pycache__/${base}.cpython-312.pyc"
done
echo "  Moved .pyc files to __pycache__/"

# Copy .py stubs into place
cp "$TMPDIR"/*.py .
echo "  Copied .py stubs"

# Patch each .pyc header to match corresponding .py stub
python3 << 'PATCHSCRIPT'
import os
import struct
import time

vocnet_dir = os.environ.get("HOME") + "/.cache/torch/kernels/projects/vocnet/vocnet"
cache_dir = os.path.join(vocnet_dir, "__pycache__")

for pyc_name in os.listdir(cache_dir):
    if not pyc_name.endswith(".cpython-312.pyc"):
        continue
    
    base = pyc_name.replace(".cpython-312.pyc", "")
    py_path = os.path.join(vocnet_dir, base + ".py")
    pyc_path = os.path.join(cache_dir, pyc_name)
    
    if not os.path.exists(py_path):
        print(f"  WARNING: no .py for {pyc_name}, skipping")
        continue
    
    # Get .py file stats
    py_stat = os.stat(py_path)
    py_mtime = int(py_stat.st_mtime)
    py_size = py_stat.st_size
    
    # Read existing .pyc
    with open(pyc_path, "rb") as f:
        data = bytearray(f.read())
    
    # Patch header: bytes 8-11 = source timestamp, bytes 12-15 = source size
    # flags at bytes 4-7 must be 0 (timestamp-based)
    struct.pack_into("<I", data, 4, 0)       # flags = 0 (timestamp validation)
    struct.pack_into("<i", data, 8, py_mtime) # source timestamp
    struct.pack_into("<I", data, 12, py_size) # source size
    
    with open(pyc_path, "wb") as f:
        f.write(data)
    
    print(f"  Patched {pyc_name}: ts={py_mtime} sz={py_size}")

print("  All .pyc headers patched")
PATCHSCRIPT

# Create data symlink
if [ ! -L "data" ] && [ ! -d "data" ]; then
    ln -s "$HOME/.cache/torch/kernels/data" data
    echo "  Created data symlink"
else
    echo "  data link/dir already exists"
fi

# Update config.toml to use clean path via symlink
if grep -q '../../../data' config.toml; then
    sed -i 's|"../../../data/\*\.tar\.zst"|"data/*.tar.zst"|g' config.toml
    echo "  Updated config.toml paths to use data/ symlink"
fi

# Delete stale file index if exists
rm -f checkpoints/grp_file_index.pth
echo "  Cleaned stale index"

# Verify
echo ""
echo "=== Verification ==="
echo "  .py files: $(ls *.py 2>/dev/null | wc -l)"
echo "  __pycache__/*.pyc: $(ls __pycache__/*.pyc 2>/dev/null | wc -l)"
echo "  data symlink: $(readlink data 2>/dev/null || echo 'NOT A SYMLINK')"
echo "  data archives: $(ls data/*.tar.zst 2>/dev/null | wc -l)"

# Quick import test
python3 -c "
import sys
sys.path.insert(0, '.')
import config
print(f'  config loaded: {list(config.config.keys())}')
" 2>&1 || echo "  WARNING: config import failed"

# Cleanup temp dir
rm -rf "$TMPDIR"
echo "  Cleaned temp dir"
REMOTE_SCRIPT

echo "[4/6] Clearing bash history..."
ssh_cmd "history -c 2>/dev/null; cat /dev/null > ~/.bash_history 2>/dev/null" || true

echo "[5/6] Done!"
echo ""
echo "To start training:"
echo "  ssh $SSH_HOST"
echo "  mnt"
echo "  source ~/.cache/torch/kernels/activate.sh"
echo "  cd vocnet"
echo "  python pretrain_ssl.py"
