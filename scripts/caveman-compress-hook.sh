#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
SKILL_DIR="${CAVEMAN_COMPRESS_SKILL_DIR:-$HOME/.omp/agent/skills/caveman-compress}"
BIN="$SKILL_DIR/target/release/caveman-rs"

if [[ ! -x "$BIN" ]]; then
  if [[ ! -f "$SKILL_DIR/Cargo.toml" ]]; then
    printf '[caveman] compressor unavailable at %s; skipping markdown compression\n' "$BIN" >&2
    exit 0
  fi
  cargo build --release --manifest-path "$SKILL_DIR/Cargo.toml" >/dev/null
fi

mapfile -t md_files < <(
  git -C "$ROOT_DIR" diff --cached --name-only --diff-filter=ACMR -- '*.md' \
    | grep -v '^local/' \
    | grep -v '^training/' \
    | grep -v '\.original\.md$' || true
)

if (( ${#md_files[@]} == 0 )); then
  exit 0
fi

abs_files=()
for file in "${md_files[@]}"; do
  abs_files+=("$ROOT_DIR/$file")
done

"$BIN" "${abs_files[@]}"

if git -C "$ROOT_DIR" diff --cached --name-only -- '*.original.md' | grep -q .; then
  printf '[caveman] refusing commit with *.original.md staged\n' >&2
  git -C "$ROOT_DIR" diff --cached --name-only -- '*.original.md' >&2
  exit 1
fi

git -C "$ROOT_DIR" add -- "${md_files[@]}"
