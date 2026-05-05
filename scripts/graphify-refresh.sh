#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
GRAPHIFY_DIR="$ROOT_DIR/graphify-out"
PYTHON_FILE="$GRAPHIFY_DIR/.graphify_python"
ROOT_FILE="$GRAPHIFY_DIR/.graphify_root"
LOG_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/hydra"
LOG_FILE="$LOG_DIR/graphify-refresh.log"
LOCK_DIR="$GRAPHIFY_DIR/.refresh.lock"

usage() {
  cat <<'USAGE'
Usage: scripts/graphify-refresh.sh [--install-hook] [--check] [--full] [--force]

Refresh Hydra graphify artifacts using the uv-managed graphifyy tool.

Modes:
  default        rebuild code graph, regenerate wiki, verify required artifacts
  --full         run graphify extract when API keys are available; otherwise fall back to code rebuild
  --check        verify graphify version, hooks, graph report, graph JSON, manifest, wiki
  --install-hook install graphify post-commit/post-checkout hooks, then verify
  --force        pass GRAPHIFY_FORCE=1 to graphify code rebuild

Env:
  HYDRA_GRAPHIFY_FULL=1       same as --full
  HYDRA_GRAPHIFY_FORCE=1      same as --force
  HYDRA_GRAPHIFY_NO_HOOK=1    skip hook status check in --check
USAGE
}

INSTALL_HOOK=0
CHECK_ONLY=0
FULL=0
FORCE="${HYDRA_GRAPHIFY_FORCE:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install-hook) INSTALL_HOOK=1 ;;
    --check) CHECK_ONLY=1 ;;
    --full) FULL=1 ;;
    --force) FORCE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'error: unknown arg: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ "${HYDRA_GRAPHIFY_FULL:-0}" == "1" ]]; then
  FULL=1
fi

mkdir -p "$GRAPHIFY_DIR" "$LOG_DIR"
cd "$ROOT_DIR"

log() {
  printf '[graphify-refresh] %s\n' "$*" | tee -a "$LOG_FILE"
}

fail() {
  printf '[graphify-refresh] error: %s\n' "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

acquire_lock() {
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    trap 'rm -rf "$LOCK_DIR"' EXIT
    return
  fi
  fail "another graphify refresh is running: $LOCK_DIR"
}

resolve_python() {
  need_cmd uv

  local uv_python="$HOME/.local/share/uv/tools/graphifyy/bin/python"
  if [[ ! -x "$uv_python" ]]; then
    log 'installing graphifyy with uv tool'
    uv tool install graphifyy >>"$LOG_FILE" 2>&1
  else
    if ! "$uv_python" -c 'import graphify' >/dev/null 2>&1; then
      log 'repairing graphifyy uv tool install'
      uv tool install --force graphifyy >>"$LOG_FILE" 2>&1
    fi
  fi

  [[ -x "$uv_python" ]] || fail "uv graphifyy interpreter not found: $uv_python"
  "$uv_python" -c 'import graphify' >/dev/null || fail 'graphify import failed from uv tool interpreter'

  printf '%s\n' "$uv_python" > "$PYTHON_FILE"
  printf '%s\n' "$ROOT_DIR" > "$ROOT_FILE"
  PYTHON="$uv_python"
}

version_check() {
  local version
  version="$($PYTHON -c 'import importlib.metadata as m; print(m.version("graphifyy"))')"
  log "graphifyy=$version python=$PYTHON"
}

install_hooks() {
  need_cmd graphify
  log 'installing graphify git hooks'
  graphify hook install | tee -a "$LOG_FILE"
}

check_hooks() {
  if [[ "${HYDRA_GRAPHIFY_NO_HOOK:-0}" == "1" ]]; then
    return
  fi
  local post_commit='.git/hooks/post-commit'
  local post_checkout='.git/hooks/post-checkout'
  [[ -x "$post_commit" ]] || fail 'post-commit hook is not executable'
  [[ -x "$post_checkout" ]] || fail 'post-checkout hook is not executable'
  grep -q 'scripts/graphify-refresh.sh' "$post_commit" || fail 'post-commit hook does not call scripts/graphify-refresh.sh'
  grep -q 'scripts/graphify-refresh.sh' "$post_checkout" || fail 'post-checkout hook does not call scripts/graphify-refresh.sh'
  log 'git hooks installed: post-commit and post-checkout call scripts/graphify-refresh.sh'
}
rebuild_code_graph() {
  log 'rebuilding graphify code graph'
  if [[ "$FORCE" == "1" ]]; then
    GRAPHIFY_FORCE=1 "$PYTHON" -c 'from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path("."), force=True)' 2>&1 | tee -a "$LOG_FILE"
  else
    "$PYTHON" -c 'from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path("."))' 2>&1 | tee -a "$LOG_FILE"
  fi
}

full_extract_if_possible() {
  if [[ "$FULL" != "1" ]]; then
    rebuild_code_graph
    return
  fi

  if [[ -z "${MOONSHOT_API_KEY:-}" && -z "${ANTHROPIC_API_KEY:-}" ]]; then
    log 'full refresh requested but no MOONSHOT_API_KEY or ANTHROPIC_API_KEY set; falling back to code graph rebuild'
    rebuild_code_graph
    return
  fi

  log 'running graphify headless full extract'
  "$PYTHON" -m graphify extract "$ROOT_DIR" --out "$ROOT_DIR" 2>&1 | tee -a "$LOG_FILE"
}

generate_wiki() {
  log 'generating graphify wiki'
  "$PYTHON" - <<'PY'
import json
from pathlib import Path
from graphify.build import build_from_json
from graphify.wiki import to_wiki

root = Path('.')
out = root / 'graphify-out'
extract_path = out / '.graphify_extract.json'
analysis_path = out / '.graphify_analysis.json'
labels_path = out / '.graphify_labels.json'
if not extract_path.exists():
    extract_path = root / '.graphify_extract.json'
if not analysis_path.exists():
    analysis_path = root / '.graphify_analysis.json'
if not labels_path.exists():
    labels_path = root / '.graphify_labels.json'
if not extract_path.exists() or not analysis_path.exists():
    raise SystemExit('missing .graphify_extract.json or .graphify_analysis.json; run full /graphify once before wiki generation')
extraction = json.loads(extract_path.read_text())
analysis = json.loads(analysis_path.read_text())
labels_raw = json.loads(labels_path.read_text()) if labels_path.exists() else {}
G = build_from_json(extraction)
communities = {int(k): v for k, v in analysis['communities'].items()}
cohesion = {int(k): v for k, v in analysis.get('cohesion', {}).items()}
labels = {int(k): v for k, v in labels_raw.items()}
gods = []
for item in analysis.get('gods', []):
    if isinstance(item, dict):
        label = item.get('label') or item.get('id') or item.get('node')
        fallback_degree = item.get('degree', 0)
    else:
        label = str(item)
        fallback_degree = 0
    degree = G.degree(label) if label in G else fallback_degree
    gods.append({'label': label, 'degree': degree})
n = to_wiki(G, communities, out / 'wiki', community_labels=labels or None, cohesion=cohesion or None, god_nodes_data=gods)
print(f'wiki files: {n}')
PY
}

verify_artifacts() {
  log 'verifying graphify artifacts'
  [[ -s "$GRAPHIFY_DIR/graph.json" ]] || fail 'missing graphify-out/graph.json'
  [[ -s "$GRAPHIFY_DIR/GRAPH_REPORT.md" ]] || fail 'missing graphify-out/GRAPH_REPORT.md'
  [[ -s "$GRAPHIFY_DIR/manifest.json" ]] || fail 'missing graphify-out/manifest.json'
  [[ -s "$GRAPHIFY_DIR/wiki/index.md" ]] || fail 'missing graphify-out/wiki/index.md'

  "$PYTHON" - <<'PY'
import json
from pathlib import Path
from networkx.readwrite import json_graph

out = Path('graphify-out')
with (out / 'graph.json').open() as fh:
    raw = json.load(fh)
try:
    G = json_graph.node_link_graph(raw, edges='links')
except TypeError:
    G = json_graph.node_link_graph(raw)
if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
    raise SystemExit('graph is empty')
report = (out / 'GRAPH_REPORT.md').read_text()
if '## Summary' not in report or '## God Nodes' not in report:
    raise SystemExit('GRAPH_REPORT.md missing expected sections')
wiki_files = list((out / 'wiki').glob('*.md'))
if len(wiki_files) < 2:
    raise SystemExit('wiki has too few markdown files')
manifest = json.loads((out / 'manifest.json').read_text())
if not manifest:
    raise SystemExit('manifest is empty')
print(f'graph ok: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(wiki_files)} wiki files')
PY
}
main() {
  : > "$LOG_FILE"
  acquire_lock
  resolve_python
  version_check

  if [[ "$INSTALL_HOOK" == "1" ]]; then
    install_hooks
  fi

  check_hooks

  if [[ "$CHECK_ONLY" == "0" ]]; then
    full_extract_if_possible
    generate_wiki
  fi

  verify_artifacts
  log 'done'
}

main
