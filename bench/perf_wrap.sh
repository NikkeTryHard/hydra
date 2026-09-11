#!/usr/bin/env bash
# bench/perf_wrap.sh — MANUAL perf wrapper for the feed-rate bench (P0-A).
#
# This file DOCUMENTS the canonical `perf stat` counter set. It is NEVER
# called from bench/feed_rate.py (or any Python): the bench harness stays
# stdlib+torch only and records wall-clock rates; hardware counters are a
# human-run adjunct for A/B triage (Wave P6) AFTER a >=15% warm-median delta.
#
# Canonical counter set (fixed; compare like-for-like across A/B legs):
#   perf stat -e cycles,instructions,cache-misses,branch-misses
#
# Manual usage (from the repo root; single pinned run, no sweeping here):
#   bench/perf_wrap.sh "pixi run python bench/feed_rate.py run-once \
#       --leg stream --corpus f8 --chunk 32 --pin --ring"
#
# Notes:
#   - Pin first (taskset) or pass a --pin bench config; unpinned perf numbers
#     are not comparable and MUST be discarded, never averaged with pinned.
#   - Record the perfed config slug + manifest digest alongside the output.
#   - `perf` needs perf_event_paranoid <= 1 for unprivileged counters; see your
#     distro docs. No sudo is embedded here by design.
set -u
if [ "$#" -eq 0 ]; then
  echo "usage: bench/perf_wrap.sh \"<bench command>\"" >&2
  exit 2
fi
exec perf stat -e cycles,instructions,cache-misses,branch-misses "$@"
