#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'USAGE'
Usage: scripts/lint-check.sh [--install-hook] [--anti-game-only]

Strict Hydra quality gate:
  - caveman-compress staged Markdown
  - anti-game pattern scan
  - cargo fmt --all -- --check
  - cargo clippy --workspace --all-targets --all-features -- -D warnings

Env:
  HYDRA_LINT_ANTI_GAME_ONLY=1  skip fmt/clippy
  HYDRA_LINT_SKIP_INSTALL=1    avoid hook install prompt/side effects
USAGE
}

INSTALL_HOOK=0
ANTI_GAME_ONLY="${HYDRA_LINT_ANTI_GAME_ONLY:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-hook) INSTALL_HOOK=1 ;;
    --anti-game-only) ANTI_GAME_ONLY=1 ;;
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

run_step() {
  printf '\n[hydra lint] %s\n' "$1"
  shift
  "$@"
}

anti_game_scan() {
  need_cmd rg
  local failed=0

  printf '\n[hydra lint] anti-game scan\n'

  if rg --line-number --hidden --glob '!target/**' --glob '!.git/**' --glob '!graphify-out/**' --glob '!research/agent_handoffs/**' \
    '(^|[^[:alnum:]_])RUSTFLAGS=.*(^|[[:space:]])(-A|--allow)([[:space:]]|=)|cargo[[:space:]]+clippy[^\n]*--[^\n]*(^|[[:space:]])(-A|--allow)([[:space:]]|=)|CLIPPY_ARGS=.*(^|[[:space:]])(-A|--allow)([[:space:]]|=)' \
    .; then
    printf '\n[hydra lint] found command-line lint suppression\n' >&2
    failed=1
  fi

  if rg --line-number --hidden --glob '!target/**' --glob '!.git/**' --glob '!graphify-out/**' --glob '!research/agent_handoffs/**' \
    '#!?\[allow\((warnings|clippy::all|clippy::correctness|clippy::suspicious|clippy::complexity|clippy::perf|clippy::style|clippy::pedantic|clippy::restriction|clippy::nursery|clippy::cargo)\)\]' \
    .; then
    printf '\n[hydra lint] found blanket lint allow attribute\n' >&2
    failed=1
  fi

  if rg --line-number --hidden --glob '!target/**' --glob '!.git/**' --glob '!graphify-out/**' --glob '!research/agent_handoffs/**' \
    'clippy[[:space:]]*::[[:space:]]*allow_attributes_without_reason[[:space:]]*=[[:space:]]*"allow"|allow_attributes_without_reason[[:space:]]*=[[:space:]]*"allow"' \
    .; then
    printf '\n[hydra lint] found attempt to disable allow-reason enforcement\n' >&2
    failed=1
  fi

  if rg --line-number --hidden --glob '!target/**' --glob '!.git/**' --glob '!graphify-out/**' --glob '!research/agent_handoffs/**' \
    '#!?\[allow\([^\]]*\)\]' \
    crates scripts; then
    printf '\n[hydra lint] allow attributes above are permitted only when narrow and justified; clippy enforces reason text\n'
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
sh "$ROOT/scripts/caveman-compress-hook.sh"
exec "$ROOT/scripts/lint-check.sh"
HOOK
  chmod +x "$hook"
  printf '[hydra lint] installed %s\n' "$hook"
}

if [[ "$INSTALL_HOOK" == "1" ]]; then
  install_hook
fi

if git rev-parse --git-dir >/dev/null 2>&1; then
  run_step 'caveman-compress staged markdown' scripts/caveman-compress-hook.sh
fi

anti_game_scan

if [[ "$ANTI_GAME_ONLY" == "1" ]]; then
  exit 0
fi

need_cmd cargo
run_step 'checking rustfmt' cargo fmt --all -- --check
run_step 'checking clippy' cargo clippy --workspace --all-targets --all-features -- -D warnings
