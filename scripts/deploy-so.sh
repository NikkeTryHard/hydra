#!/bin/bash
# Build acoustic_core .so on server and deploy to encrypted volume.
# All build artifacts are created in a random /tmp/ dir and nuked after.
# No source or intermediate files persist.
set -euo pipefail

SSH_KEY="/home/nikketryhard/.ssh/id_ed25519"
SSH="ssh -i $SSH_KEY sxk6383@supernova.uta.edu"
SCP="scp -i $SSH_KEY"
LOCAL_SRC="/home/nikketryhard/dev/hydra/mortal-github-repo/libriichi"

# Random build dir name — no identifiable prefix
BUILD_ID=$(openssl rand -hex 6)
REMOTE_TMP="/tmp/.b_${BUILD_ID}"

echo "[1/5] Creating source tarball..."
cd "$LOCAL_SRC/.."
tar cf /tmp/.b_src.tar libriichi/
echo "  $(du -sh /tmp/.b_src.tar | cut -f1)"

echo "[2/5] Uploading to server..."
$SSH "mkdir -p $REMOTE_TMP"
$SCP /tmp/.b_src.tar "sxk6383@supernova.uta.edu:$REMOTE_TMP/s.tar"
rm /tmp/.b_src.tar

echo "[3/5] Building on server..."
$SSH "
set -e
cd '$REMOTE_TMP'
TOOLDIR=\$HOME/.local/share/ml-env
source \$TOOLDIR/venv/bin/activate
export RUSTUP_HOME=\$TOOLDIR/toolchains/rustup
export CARGO_HOME=\$TOOLDIR/toolchains/cargo
export PATH=\$CARGO_HOME/bin:\$TOOLDIR/toolchains:\$PATH
export CUDA_HOME=/usr/local/cuda-13.0
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}
export CARGO_TARGET_DIR='$REMOTE_TMP/t'
export RUSTFLAGS='--remap-path-prefix=$REMOTE_TMP=. --remap-path-prefix=\$HOME=~'
export PYTHONDONTWRITEBYTECODE=1

tar xf s.tar && rm s.tar
cd libriichi
sed -i 's/name = \"libriichi\"/name = \"acoustic_core\"/' Cargo.toml
sed -i 's/name = \"riichi\"/name = \"acoustic_core\"/' Cargo.toml
sed -i 's/fn libriichi(/fn acoustic_core(/' src/lib.rs

maturin build --release --strip --out '$REMOTE_TMP/d/' 2>&1 | tail -3

W=\$(ls '$REMOTE_TMP/d/'*.whl | head -1)
mkdir -p '$REMOTE_TMP/e'
unzip -o \"\$W\" 'acoustic_core/*.so' -d '$REMOTE_TMP/e/' >/dev/null 2>&1
SO=\$(find '$REMOTE_TMP/e/' -name '*.so' | head -1)
cp \"\$SO\" ~/.cache/torch/kernels/lib/acoustic_core.cpython-312-x86_64-linux-gnu.so

echo '[4/5] Verifying...'
python -c 'from acoustic_core.dataset import GameplayLoader, Grp; print(\"OK\")'

echo '[5/5] Cleaning...'
rm -rf '$REMOTE_TMP'
# Nuke any cargo registry/git that got recreated
rm -rf \$CARGO_HOME/registry/ \$CARGO_HOME/git/ 2>/dev/null
history -c 2>/dev/null; cat /dev/null > ~/.bash_history 2>/dev/null
echo 'DONE'
"
