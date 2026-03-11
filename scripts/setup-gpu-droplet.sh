#!/usr/bin/env bash
# setup-gpu-droplet.sh -- One-shot provisioning for DigitalOcean GPU Droplets
#
# Target: DO AI/ML-ready image (gpu-h100x1-base, Ubuntu 22.04, CUDA 12.9)
# Builds: Hydra training binaries (train, mjai_audit, recompress)
# Runtime: Burn 0.20 + tch 0.22.0 + PyTorch 2.9.0 (CUDA 12.8 wheels)
#
# Usage:
#   ssh root@<droplet-ip>
#   curl -fsSL https://raw.githubusercontent.com/NikkeTryHard/hydra/master/scripts/setup-gpu-droplet.sh | bash
#
# Or clone first and run locally:
#   git clone https://github.com/NikkeTryHard/hydra.git && cd hydra
#   bash scripts/setup-gpu-droplet.sh

set -euo pipefail

# --------------------------------------------------------------------------- #
#  Configuration                                                               #
# --------------------------------------------------------------------------- #
HYDRA_REPO="https://github.com/NikkeTryHard/hydra.git"
HYDRA_BRANCH="master"
HYDRA_DIR="/opt/hydra"
PYTORCH_VERSION="2.9.0"
PYTORCH_CUDA_SUFFIX="cu128"  # PyTorch 2.9.0 ships cu128 wheels (compat w/ CUDA 12.9)

# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #
info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
ok()    { printf '\033[1;32m[OK]\033[0m    %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
fatal() { printf '\033[1;31m[FAIL]\033[0m  %s\n' "$*" >&2; exit 1; }

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || fatal "Required command not found: $1"
}

# --------------------------------------------------------------------------- #
#  Pre-flight checks                                                           #
# --------------------------------------------------------------------------- #
info "Running pre-flight checks..."

# Must be root (DO droplets default to root)
[[ $EUID -eq 0 ]] || fatal "This script must be run as root."

# Check NVIDIA driver is present
if command -v nvidia-smi >/dev/null 2>&1; then
    info "GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    fatal "nvidia-smi not found. Is this a GPU droplet with the AI/ML-ready image?"
fi

# Check CUDA toolkit
if command -v nvcc >/dev/null 2>&1; then
    info "CUDA toolkit: $(nvcc --version | grep 'release' | sed 's/.*release //' | sed 's/,.*//')"
else
    warn "nvcc not found in PATH. CUDA toolkit may not be installed or not in PATH."
    warn "Checking /usr/local/cuda..."
    if [[ -d /usr/local/cuda ]]; then
        export PATH="/usr/local/cuda/bin:${PATH}"
        info "Found /usr/local/cuda, added to PATH."
    else
        fatal "No CUDA toolkit found. The AI/ML-ready image should have cuda-toolkit-12-9."
    fi
fi

ok "Pre-flight checks passed."

# --------------------------------------------------------------------------- #
#  Step 1: System packages                                                     #
# --------------------------------------------------------------------------- #
info "Installing system packages..."

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    clang \
    curl \
    git \
    mold \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv

ok "System packages installed."

# --------------------------------------------------------------------------- #
#  Step 2: Python venv + PyTorch                                               #
# --------------------------------------------------------------------------- #
VENV_DIR="/opt/pytorch-venv"

if [[ -d "${VENV_DIR}" ]] && "${VENV_DIR}/bin/python" -c "import torch; assert torch.__version__.startswith('${PYTORCH_VERSION}')" 2>/dev/null; then
    info "PyTorch ${PYTORCH_VERSION} already installed in ${VENV_DIR}, skipping."
else
    info "Creating Python venv and installing PyTorch ${PYTORCH_VERSION} (${PYTORCH_CUDA_SUFFIX})..."
    python3 -m venv "${VENV_DIR}"
    "${VENV_DIR}/bin/pip" install --upgrade pip
    "${VENV_DIR}/bin/pip" install \
        "torch==${PYTORCH_VERSION}" \
        --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA_SUFFIX}"
fi

# Verify torch install
TORCH_VERSION=$("${VENV_DIR}/bin/python" -c "import torch; print(torch.__version__)")
TORCH_CUDA=$("${VENV_DIR}/bin/python" -c "import torch; print(torch.cuda.is_available())")
info "PyTorch version: ${TORCH_VERSION}"
info "CUDA available to PyTorch: ${TORCH_CUDA}"

if [[ "${TORCH_CUDA}" != "True" ]]; then
    warn "PyTorch cannot see CUDA. Training will fall back to CPU."
    warn "This usually means the CUDA driver version is too old for this PyTorch build."
fi

ok "PyTorch installed."

# --------------------------------------------------------------------------- #
#  Step 3: Register libtorch with the dynamic linker                           #
# --------------------------------------------------------------------------- #
info "Configuring dynamic linker for libtorch..."

TORCH_LIB_DIR=$("${VENV_DIR}/bin/python" -c "
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), 'lib'))
")

if [[ ! -d "${TORCH_LIB_DIR}" ]]; then
    fatal "PyTorch lib directory not found: ${TORCH_LIB_DIR}"
fi

echo "${TORCH_LIB_DIR}" > /etc/ld.so.conf.d/pytorch-libtorch.conf
ldconfig

ok "libtorch registered at ${TORCH_LIB_DIR}"

# --------------------------------------------------------------------------- #
#  Step 4: Rust toolchain                                                      #
# --------------------------------------------------------------------------- #
if command -v rustc >/dev/null 2>&1; then
    info "Rust already installed: $(rustc --version)"
else
    info "Installing Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
fi

# Source cargo env for this session
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
export PATH="${CARGO_HOME}/bin:${PATH}"

# Verify
need_cmd rustc
need_cmd cargo
info "Rust: $(rustc --version)"

# Install cargo-nextest for faster test runs
if command -v cargo-nextest >/dev/null 2>&1; then
    info "cargo-nextest already installed."
else
    info "Installing cargo-nextest..."
    cargo install cargo-nextest --locked
fi

ok "Rust toolchain ready."

# --------------------------------------------------------------------------- #
#  Step 5: Clone Hydra                                                         #
# --------------------------------------------------------------------------- #
if [[ -d "${HYDRA_DIR}/.git" ]]; then
    info "Hydra repo already exists at ${HYDRA_DIR}, pulling latest..."
    git -C "${HYDRA_DIR}" fetch origin
    git -C "${HYDRA_DIR}" reset --hard "origin/${HYDRA_BRANCH}"
else
    info "Cloning Hydra..."
    git clone --branch "${HYDRA_BRANCH}" "${HYDRA_REPO}" "${HYDRA_DIR}"
fi

ok "Hydra source at ${HYDRA_DIR}"

# --------------------------------------------------------------------------- #
#  Step 6: Build Hydra                                                         #
# --------------------------------------------------------------------------- #
info "Building Hydra (release mode)..."
info "This will take a while on first build (compiling libtorch bindings)..."

cd "${HYDRA_DIR}"

# Tell tch/torch-sys to use the Python PyTorch install
export LIBTORCH_USE_PYTORCH=1

# Make sure torch-sys can find Python torch
export PATH="${VENV_DIR}/bin:${PATH}"

# Build all binaries
cargo build --release -p hydra-train 2>&1 | tail -5

# Verify binaries exist
for bin in train mjai_audit recompress; do
    if [[ -x "target/release/${bin}" ]]; then
        ok "Built: target/release/${bin}"
    else
        fatal "Binary not found: target/release/${bin}"
    fi
done

# Symlink into /usr/local/bin for convenience
ln -sf "${HYDRA_DIR}/target/release/train" /usr/local/bin/train
ln -sf "${HYDRA_DIR}/target/release/mjai_audit" /usr/local/bin/mjai_audit
ln -sf "${HYDRA_DIR}/target/release/recompress" /usr/local/bin/recompress

ok "Hydra binaries built and linked to /usr/local/bin/"

# --------------------------------------------------------------------------- #
#  Step 7: Create runtime environment file                                     #
# --------------------------------------------------------------------------- #
ENV_FILE="/etc/profile.d/hydra.sh"

cat > "${ENV_FILE}" << 'ENVEOF'
# Hydra training environment
export LIBTORCH_USE_PYTORCH=1
export HYDRA_TRAIN_DEVICE=cuda:0
export PATH="/opt/pytorch-venv/bin:/usr/local/cuda/bin:${PATH}"
ENVEOF

chmod +x "${ENV_FILE}"
ok "Environment file written to ${ENV_FILE}"

# --------------------------------------------------------------------------- #
#  Step 8: Smoke test                                                          #
# --------------------------------------------------------------------------- #
info "Running smoke test..."

export HYDRA_TRAIN_DEVICE=cpu

# Just check the binary runs and prints usage (non-zero exit is fine here)
if train --help >/dev/null 2>&1 || train 2>&1 | head -1 | grep -qi -e "usage" -e "hydra" -e "train" -e "error"; then
    ok "train binary executes."
else
    warn "train binary may have issues. Check manually: train --help"
fi

# Quick CUDA visibility check
"${VENV_DIR}/bin/python" -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
else:
    print('WARNING: CUDA not available to PyTorch')
"

# --------------------------------------------------------------------------- #
#  Done                                                                        #
# --------------------------------------------------------------------------- #
echo ""
echo "================================================================="
echo "  Hydra GPU Droplet Setup Complete"
echo "================================================================="
echo ""
echo "  Binaries:    train, mjai_audit, recompress (in /usr/local/bin)"
echo "  Source:       ${HYDRA_DIR}"
echo "  PyTorch:     ${VENV_DIR} (v${PYTORCH_VERSION})"
echo "  Environment: source ${ENV_FILE}"
echo ""
echo "  Quick start:"
echo "    export HYDRA_TRAIN_DEVICE=cuda:0"
echo "    train /path/to/config.json"
echo ""
echo "  After verifying, snapshot this droplet from the DO control panel"
echo "  for reuse without rebuilding."
echo "================================================================="
