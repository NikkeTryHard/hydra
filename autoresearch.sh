#!/usr/bin/env bash
# Autoresearch harness SEGMENT 6: production-feed training rows/s (feed+compute).
# Workload: F11 saturated corpus (64x .mjai.json.zst, 768 games / 3456 decisions)
# through the PRODUCTION path: _StreamDataset (rust backend, BC, seeded) ->
# next_batch(micro=288, exact divisor of 3456 so drop_last keeps all rows) ->
# compiled Hydra2BaselineModel fwd+bwd (AdamW, masked CE), sequential.
# Warmup 1 pass (untimed, absorbs inductor compile), 3 timed passes,
# primary = median rows/s.
# Anti-gaming (ALL must pass or NO metric is emitted -> run crashes):
#  1. corpus sha256 pinned against bench/bench_corpus_manifest.json
#  2. every pass: staged==3456, consumed==3456, games==768, loss finite
#  3. CUDA required (fail closed; CPU fallback would fake the number)
#  4. dual-clock: script wall-sum <= external elapsed <= script + 15s
set -u
set -o pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORPUS="${ROOT}/bench/corpus/f11-saturated"
MANIFEST="${ROOT}/bench/bench_corpus_manifest.json"
REPLAY_RS="${ROOT}/tools/hydra2-replay-rs"
MICRO=288
PASSES=3
EXP_ROWS=3456
EXP_GAMES=768
EXP_FILES=64
SCRATCH="$(mktemp -d)"
trap 'rm -rf "${SCRATCH}"' EXIT
fail() { echo "HARNESS FAIL: $1" >&2; exit 1; }

cd "${ROOT}" || exit 1
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi missing (GPU harness)"

# --- 1. corpus pin: 64 files, every sha present in the pinned manifest ---
[ -d "${CORPUS}" ] || fail "corpus dir missing"
[ "$(ls "${CORPUS}"/*.zst | wc -l)" -eq "${EXP_FILES}" ] || fail "corpus file count != ${EXP_FILES}"
while read -r h f; do
    grep -q "${h}" "${MANIFEST}" || fail "corpus sha not in manifest: ${f}"
done < <(sha256sum "${CORPUS}"/*.zst)

# --- 2. build release extension (not timed) ---
PIXI_PY="${ROOT}/.pixi/envs/default/bin/python"
[ -x "${PIXI_PY}" ] || fail "pixi python missing"
export PYO3_PYTHON="${PIXI_PY}"
cargo build --release --quiet -p hydra2-replay-rs --manifest-path "${REPLAY_RS}/Cargo.toml" \
    2>"${SCRATCH}/build.log" || { tail -5 "${SCRATCH}/build.log" >&2; fail "build failed"; }
BUILT="${REPLAY_RS}/target/release/libhydra2_replay_rs.so"
[ -f "${BUILT}" ] || fail "cdylib missing after build"
SUF="$("${PIXI_PY}" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")"
mkdir -p "${SCRATCH}/hydra_ext"
cp "${BUILT}" "${SCRATCH}/hydra_ext/hydra2_replay_rs${SUF}" || fail "ext stage failed"

# --- 3. timed production-path training passes (external dual-clock) ---
cat >"${SCRATCH}/train_burst.py" <<'PYEOF'
import os, sys, time
sys.path.insert(0, sys.argv[1])
os.environ["HYDRA2_ARTIFACT_ROOT"] = sys.argv[2]
corpus, micro, passes = sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
import torch
assert torch.cuda.is_available(), "GPU harness requires CUDA"
torch.manual_seed(0)
from hydra2.data.stream import GameStream, build_manifest
from hydra2.models.schema import BASELINE_ACTION_COUNT
from hydra2.training import stream_train as driver
from hydra2.models.model import Hydra2BaselineModel
import torch.nn.functional as F
manifest = build_manifest(corpus)
model = Hydra2BaselineModel().cuda().train()
for _layer in model.layers:
    _layer.attn_bf16 = True
model = torch.compile(model)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

def run_pass():
    staged = consumed = games = 0
    feed_s = step_s = 0.0
    loss_last = float("nan")
    t0 = time.perf_counter()
    ds = driver._StreamDataset(
        stream_factory=lambda: GameStream(manifest, seed=7, ratios={"train": 1.0}, split=None),
        num_actions=BASELINE_ACTION_COUNT, feature_dim=64, seed=7,
        drop_last=True, need_privileged=False, replay_backend="rust",
    )
    from hydra2.models.encoder import ActorTensorBatch
    try:
        f0 = time.perf_counter()
        while ds._pull_game():
            pass
        feed_s += time.perf_counter() - f0
        staged = len(ds._rows)
        games = ds.replayed + ds.sim_replayed
        while consumed < staged:
            f0 = time.perf_counter()
            batch = ds.next_batch(micro)
            ab0 = batch["actor_batch"]
            feats = {k: v.cuda(non_blocking=True) for k, v in ab0.features.items()}
            ab = ActorTensorBatch(features=feats, history_mask=feats["history_mask"],
                                  legal_mask=feats["legal_mask"], observation_hashes=(),
                                  actor_seats=feats["actor"])
            ch = batch["chosen_action_id"].cuda(non_blocking=True)
            torch.cuda.current_stream().synchronize()
            feed_s += time.perf_counter() - f0
            s0 = time.perf_counter()
            out = model(ab)
            loss = F.cross_entropy(
                out.policy_logits.masked_fill(~ab.legal_mask, float("-inf")), ch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step_s += time.perf_counter() - s0
            consumed += int(ch.shape[0])
            loss_last = float(loss.detach().item())
    finally:
        ds.close()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return staged, consumed, games, wall, feed_s, step_s, loss_last

wb = run_pass()
assert (wb[0], wb[1], wb[2]) == (3456, 3456, 768), f"warmup counts {wb[:3]}"
import math
assert math.isfinite(wb[6]), "warmup loss non-finite"
rates = []
for _ in range(passes):
    r = run_pass()
    assert (r[0], r[1], r[2]) == (3456, 3456, 768), f"counts {r[:3]}"
    assert math.isfinite(r[6]), "loss non-finite"
    rates.append((3456 / r[3], r[3], r[4], r[5], r[6]))
rates.sort()
med = rates[len(rates) // 2]
print(f"TRAIN rows_s={med[0]:.1f} wall_s={med[1]:.3f} feed_s={med[2]:.3f} "
      f"step_s={med[3]:.3f} loss={med[4]:.4f} staged=3456 consumed=3456 games=768 warmup_s={wb[3]:.3f}")
PYEOF

ART="${SCRATCH}/artifacts"
mkdir -p "${ART}"
EXT_START=$(date +%s%N)
CUBLAS_WORKSPACE_CONFIG=:4096:8 pixi run python "${SCRATCH}/train_burst.py" \
    "${SCRATCH}/hydra_ext" "${ART}" "${CORPUS}" "${MICRO}" "${PASSES}" \
    >"${SCRATCH}/train.log" 2>&1 || { tail -20 "${SCRATCH}/train.log" >&2; fail "train burst failed"; }
EXT_END=$(date +%s%N)
grep -q "TRAIN " "${SCRATCH}/train.log" || fail "TRAIN line missing"
grep -q "staged=3456 consumed=3456 games=768" "${SCRATCH}/train.log" \
    || fail "staged/consumed/games != 3456/3456/768"
RATE="$(grep -oP "rows_s=\K[0-9.]+" "${SCRATCH}/train.log" | head -1)"
WALL="$(grep -oP "wall_s=\K[0-9.]+" "${SCRATCH}/train.log" | head -1)"
FEED="$(grep -oP "feed_s=\K[0-9.]+" "${SCRATCH}/train.log" | head -1)"
STEP="$(grep -oP "step_s=\K[0-9.]+" "${SCRATCH}/train.log" | head -1)"
WARM="$(grep -oP "warmup_s=\K[0-9.]+" "${SCRATCH}/train.log" | head -1)"
[ -n "${RATE}" ] && [ -n "${WALL}" ] || fail "metric parse failed"
EXT_MS="$(( (EXT_END - EXT_START) / 1000000 ))"
BUSY="$(awk -v s="$STEP" -v w="$WALL" 'BEGIN {printf "%d", 100*s/w}')"
SCRIPT_MS="$(awk -v w="$WALL" -v p="$PASSES" -v u="$WARM" 'BEGIN {printf "%d", (w*p+u)*1000}')"
awk -v s="$SCRIPT_MS" -v e="$EXT_MS" 'BEGIN { if (!(s <= e && e <= s + 15000)) exit 1 }' \
    || fail "wall anomaly script_sum=${SCRIPT_MS}ms external=${EXT_MS}ms"
STEPMS="$(awk -v s="$STEP" -v r="$EXP_ROWS" -v m="$MICRO" 'BEGIN {printf "%.2f", s/(r/m)*1000}')"
FEEDRATE="$(awk -v r="$EXP_ROWS" -v f="$FEED" 'BEGIN {printf "%.0f", r/f}')"
echo "METRIC train_rows_per_sec=${RATE}"
echo "METRIC wall_seconds=$(awk -v e="$EXT_MS" 'BEGIN {printf "%.3f", e/1000}')"
echo "METRIC rows_staged=${EXP_ROWS}"
echo "METRIC rejects=0"
echo "METRIC gpu_util_pct=${BUSY}"
echo "METRIC step_ms=${STEPMS}"
echo "METRIC feed_rows_per_sec=${FEEDRATE}"
