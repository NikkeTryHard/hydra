#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'USAGE'
Usage: scripts/lint-check.sh [--fast|--exhaustive|--cuda-graph] [--install-hook] [--anti-game-only] [--rust-only]

Hydra quality gates:
  --fast            default; caveman-compress hook + anti-game scan + Python gate + rustfmt + clippy no-default-features
  --exhaustive      caveman-compress hook + anti-game scan + Python gate + CUDA/libtorch prep + rustfmt + clippy all-features
  --cuda-graph      caveman-compress hook + anti-game scan + Python gate + CUDA/libtorch prep + focused cuda-graph clippy
  --anti-game-only  run cheap anti-game scan only; no fmt, clippy, CUDA/libtorch prep, Python gate, or hook install side effects
  --rust-only       skip Python Ruff/Pyrefly/pytest; for direct Rust-only Pixi subtask only
  --install-hook    install pre-commit hook; hook runs fast mode by default
  --help            show this help

Env compatibility:
  HYDRA_LINT_ANTI_GAME_ONLY=1  same as --anti-game-only
  HYDRA_LINT_CUDA_GRAPH=1     same as --cuda-graph unless --exhaustive is passed
  HYDRA_LINT_EXHAUSTIVE=1     same as --exhaustive
  HYDRA_LINT_SKIP_INSTALL=1    avoid hook install prompt/side effects
  HYDRA_LINT_VERBOSE=1         print lint step names and successful CUDA prerequisite paths
USAGE
}

INSTALL_HOOK=0
ANTI_GAME_ONLY="${HYDRA_LINT_ANTI_GAME_ONLY:-0}"
MODE=fast
RUST_ONLY=0
VERBOSE="${HYDRA_LINT_VERBOSE:-0}"
if [[ "${HYDRA_LINT_EXHAUSTIVE:-0}" == "1" ]]; then
  MODE=exhaustive
elif [[ "${HYDRA_LINT_CUDA_GRAPH:-0}" == "1" ]]; then
  MODE=cuda-graph
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fast) MODE=fast ;;
    --exhaustive) MODE=exhaustive ;;
    --cuda-graph) MODE=cuda-graph ;;
    --install-hook) INSTALL_HOOK=1 ;;
    --anti-game-only) ANTI_GAME_ONLY=1 ;;
    --rust-only) RUST_ONLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'error: unknown arg: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

fail() {
  printf '\n[hydra lint] error: %s\n' "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}
run_rustfmt_check() {
  if [[ "$VERBOSE" == "1" ]]; then
    run_step 'checking rustfmt' cargo fmt --all -- --check
    return
  fi

  local output
  if ! output="$(cargo fmt --all -- --check 2>&1)"; then
    printf '[hydra lint] rustfmt check failed; run `pixi run fmt` and re-run lint.\n' >&2
    return 1
  fi
}

run_step() {
  local label="$1"
  shift
  if [[ "$VERBOSE" == "1" ]]; then
    printf '\n[hydra lint] %s\n' "$label"
    "$@"
    return
  fi

  local output
  if ! output="$("$@" 2>&1)"; then
    printf '[hydra lint] %s failed\n' "$label" >&2
    printf '%s' "$output" >&2
    return 1
  fi
}


libtorch_include_dir() {
  local root="$1"
  if [[ -d "$root/include" ]]; then
    printf '%s/include\n' "$root"
  elif [[ -d "$root/libtorch/include" ]]; then
    printf '%s/libtorch/include\n' "$root"
  else
    return 1
  fi
}

first_existing_file() {
  local base="$1"
  shift
  local rel
  for rel in "$@"; do
    if [[ -f "$base/$rel" ]]; then
      printf '%s/%s\n' "$base" "$rel"
      return 0
    fi
  done
  return 1
}

python_torch_root() {
  python3 - <<'PY'
import pathlib
import sys
try:
    import torch
except Exception as exc:
    raise SystemExit(f"python torch import failed: {exc}")
root = pathlib.Path(torch.__file__).resolve().parent
header = root / "include" / "c10" / "cuda" / "impl" / "cuda_cmake_macros.h"
libs = [root / "lib" / "libc10_cuda.so", root / "lib" / "libtorch_cuda.so"]
missing = [str(path) for path in [header, *libs] if not path.is_file()]
if missing:
    raise SystemExit("python torch is not a CUDA libtorch install; missing: " + ", ".join(missing))
print(root)
PY
}

resolve_libtorch_root() {
  if [[ -n "${LIBTORCH:-}" ]]; then
    printf '%s\n' "$LIBTORCH"
    return 0
  fi

  if [[ "${LIBTORCH_USE_PYTORCH:-}" == "1" ]]; then
    python_torch_root
    return 0
  fi

  local torch_root
  if torch_root="$(python_torch_root 2>/dev/null)"; then
    export LIBTORCH_USE_PYTORCH=1
    export LIBTORCH="$torch_root"
    printf '%s\n' "$torch_root"
    return 0
  fi

  fail "CUDA graph lint requires CUDA-enabled libtorch. Set LIBTORCH to a CUDA libtorch root, or install/select a Python torch package containing include/c10/cuda/impl/cuda_cmake_macros.h, lib/libc10_cuda.so, and lib/libtorch_cuda.so."
}

cuda_home_has_runtime() {
  local root="$1"
  [[ -n "$root" ]] || return 1
  first_existing_file "$root" include/cuda_runtime_api.h include/cuda_runtime.h >/dev/null || return 1
  first_existing_file "$root" lib64/libcudart.so lib/libcudart.so targets/x86_64-linux/lib/libcudart.so >/dev/null || return 1
}


prepare_cuda_libtorch() {
  if ! cuda_home_has_runtime "${CUDA_HOME:-}" && [[ -d /opt/cuda ]] && cuda_home_has_runtime /opt/cuda; then
    export CUDA_HOME=/opt/cuda
  fi

  local libtorch_root
  libtorch_root="$(resolve_libtorch_root)"
  export LIBTORCH="$libtorch_root"

  local missing=0
  local libtorch_header libtorch_c10_cuda libtorch_torch_cuda cuda_runtime_header cudart_lib

  local libtorch_include=""
  if ! libtorch_include="$(libtorch_include_dir "$libtorch_root")"; then
    printf '[hydra lint] missing CUDA libtorch include directory under LIBTORCH=%s\n' "$libtorch_root" >&2
    missing=1
  elif ! libtorch_header="$(first_existing_file "$libtorch_include" c10/cuda/impl/cuda_cmake_macros.h)"; then
    printf '[hydra lint] missing CUDA libtorch header under LIBTORCH=%s: include/c10/cuda/impl/cuda_cmake_macros.h\n' "$libtorch_root" >&2
    missing=1
  fi
  if ! libtorch_c10_cuda="$(first_existing_file "$libtorch_root" lib/libc10_cuda.so)"; then
    printf '[hydra lint] missing CUDA libtorch library: %s/lib/libc10_cuda.so\n' "$libtorch_root" >&2
    missing=1
  fi
  if ! libtorch_torch_cuda="$(first_existing_file "$libtorch_root" lib/libtorch_cuda.so)"; then
    printf '[hydra lint] missing CUDA libtorch library: %s/lib/libtorch_cuda.so\n' "$libtorch_root" >&2
    missing=1
  fi

  if [[ -z "${CUDA_HOME:-}" ]]; then
    printf '[hydra lint] missing CUDA_HOME; set CUDA_HOME to a CUDA toolkit root containing include/cuda_runtime_api.h and lib64/libcudart.so\n' >&2
    missing=1
  else
    if ! cuda_runtime_header="$(first_existing_file "$CUDA_HOME" include/cuda_runtime_api.h include/cuda_runtime.h)"; then
      printf '[hydra lint] missing CUDA runtime header under CUDA_HOME=%s: include/cuda_runtime_api.h or include/cuda_runtime.h\n' "$CUDA_HOME" >&2
      missing=1
    fi
    if ! cudart_lib="$(first_existing_file "$CUDA_HOME" lib64/libcudart.so lib/libcudart.so targets/x86_64-linux/lib/libcudart.so)"; then
      printf '[hydra lint] missing CUDA runtime library under CUDA_HOME=%s: lib64/libcudart.so, lib/libcudart.so, or targets/x86_64-linux/lib/libcudart.so\n' "$CUDA_HOME" >&2
      missing=1
    fi
  fi

  if [[ "$missing" != "0" ]]; then
    fail "CUDA graph/all-features lint prerequisites are incomplete; install CUDA toolkit and select a CUDA-enabled libtorch/PyTorch instead of skipping cuda-graph."
  fi

  if [[ "$VERBOSE" == "1" ]]; then
    printf '[hydra lint] CUDA libtorch: %s\n' "$libtorch_root"
    printf '[hydra lint] CUDA toolkit: %s\n' "$CUDA_HOME"
    printf '[hydra lint] verified %s\n' "$libtorch_header"
    printf '[hydra lint] verified %s\n' "$libtorch_c10_cuda"
    printf '[hydra lint] verified %s\n' "$libtorch_torch_cuda"
    printf '[hydra lint] verified %s\n' "$cuda_runtime_header"
    printf '[hydra lint] verified %s\n' "$cudart_lib"
  fi
}
anti_game_scan() {
  need_cmd rg
  local failed=0


  if rg --line-number --hidden --glob '!target/**' --glob '!.git/**' --glob '!research/agent_handoffs/**' \
    '(^|[^[:alnum:]_])RUSTFLAGS=.*(^|[[:space:]])(-A|--allow)([[:space:]]|=)|cargo[[:space:]]+clippy[^\n]*--[^\n]*(^|[[:space:]])(-A|--allow)([[:space:]]|=)|CLIPPY_ARGS=.*(^|[[:space:]])(-A|--allow)([[:space:]]|=)' \
    .; then
    printf '\n[hydra lint] found command-line lint suppression\n' >&2
    failed=1
  fi

  if rg --line-number --hidden --glob '!target/**' --glob '!.git/**' --glob '!research/agent_handoffs/**' \
    '#!?\[allow\((warnings|clippy::all|clippy::correctness|clippy::suspicious|clippy::complexity|clippy::perf|clippy::style|clippy::pedantic|clippy::restriction|clippy::nursery|clippy::cargo)\)\]' \
    .; then
    printf '\n[hydra lint] found blanket lint allow attribute\n' >&2
    failed=1
  fi

  if rg --line-number --hidden --glob '!target/**' --glob '!.git/**' --glob '!research/agent_handoffs/**' \
    'clippy[[:space:]]*::[[:space:]]*allow_attributes_without_reason[[:space:]]*=[[:space:]]*"allow"|allow_attributes_without_reason[[:space:]]*=[[:space:]]*"allow"' \
    .; then
    printf '\n[hydra lint] found attempt to disable allow-reason enforcement\n' >&2
    failed=1
  fi

  if [[ "$VERBOSE" == "1" ]]; then
    rg --line-number --hidden --glob '!target/**' --glob '!.git/**' --glob '!research/agent_handoffs/**' \
      '#!?\[allow\([^\]]*\)\]' \
      crates scripts || true
  fi

  [[ "$failed" == "0" ]] || fail 'anti-game scan failed'
}

install_hook() {
  local hook='.git/hooks/pre-commit'
  mkdir -p .git/hooks
  cat > "$hook" <<'HOOK'
#!/usr/bin/env sh
set -eu
ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
exec "$ROOT/scripts/lint-check.sh" --fast
HOOK
  chmod +x "$hook"
  printf '[hydra lint] installed %s\n' "$hook"
}

run_python_gate() {
  run_step 'checking Python format' pixi run -e py-train scripts/ruff-quiet.sh format --check python scripts/hydra_pytorch_oracle.py
  run_step 'checking Python lint' pixi run -e py-train scripts/ruff-quiet.sh check python scripts/hydra_pytorch_oracle.py
  run_step 'checking Python types' pixi run -e py-train scripts/pyrefly-quiet.sh
  run_step 'checking Python tests' pixi run -e py-train scripts/pytest-quiet.sh python/hydra_learner
}

if [[ "$INSTALL_HOOK" == "1" && "$ANTI_GAME_ONLY" != "1" ]]; then
  install_hook
fi

if [[ "$ANTI_GAME_ONLY" != "1" ]] && git rev-parse --git-dir >/dev/null 2>&1; then
  run_step 'caveman-compress staged markdown' scripts/caveman-compress-hook.sh
fi

anti_game_scan

if [[ "$ANTI_GAME_ONLY" == "1" ]]; then
  exit 0
fi

need_cmd cargo
if [[ "$RUST_ONLY" != "1" ]]; then
  run_python_gate
fi
run_rustfmt_check

case "$MODE" in
  fast)
    run_step 'checking clippy (no default features)' cargo clippy --workspace --all-targets --no-default-features --quiet -- -D warnings
    ;;
  exhaustive)
    prepare_cuda_libtorch
    run_step 'checking clippy (all features)' cargo clippy --workspace --all-targets --all-features --quiet -- -D warnings
    ;;
  cuda-graph)
    prepare_cuda_libtorch
    run_step 'checking clippy (hydra-train + hydra-train-exec cuda-graph)' cargo clippy -p hydra-train -p hydra-train-exec --all-targets --no-default-features --features cuda-graph --quiet -- -D warnings
    ;;
  *)
    fail "internal error: unknown lint mode: $MODE"
    ;;
esac
