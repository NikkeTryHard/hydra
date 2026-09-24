#!/usr/bin/env bash
# Train + live ClearML sidecar. Training stays SDK-free; this wrapper owns upload.
# usage: train_and_report.sh CONFIG [--tags t1 t2 ...] [-- TRAIN_ARGS...]
set -euo pipefail

CONFIG="${1:?usage: train_and_report.sh CONFIG [--tags ...] [-- TRAIN_ARGS...]}"
shift
TAGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --tags) shift; while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do TAGS+=("$1"); shift; done ;;
        --) shift; break ;;
        *) break ;;
    esac
done

export CONFIG_PATH="$CONFIG"
RUN_DIR="$(pixi run python -c "import os; from hydra2.training._rc_root import load_run_config; from hydra2.training._rc_digest import run_dir_for; print(run_dir_for(load_run_config(os.environ['CONFIG_PATH'])))" )"
echo "train_and_report: run_dir=$RUN_DIR"

SIDECAR_ARGS=(--run-dir "$RUN_DIR" --online --follow --upload-artifacts)
if [ "${#TAGS[@]}" -gt 0 ]; then
    SIDECAR_ARGS+=(--tags "${TAGS[@]}")
fi
pixi run python scripts/clearml_sidecar_report.py "${SIDECAR_ARGS[@]}" &
SIDECAR_PID="$!"

TRAIN_CODE=0
pixi run hydra2 train "$CONFIG" "$@" || TRAIN_CODE="$?"

mkdir -p "$RUN_DIR/logs"
touch "$RUN_DIR/logs/.sidecar-done"
wait "$SIDECAR_PID"
echo "train_and_report: train_exit=$TRAIN_CODE"
exit "$TRAIN_CODE"
