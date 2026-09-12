#!/usr/bin/env bash
# Canonical throughput benchmark at the overnight-4 training shape (THROUGHPUT.md).
# Runs train.py for a fixed number of updates on the Rust sim, in its own
# systemd scope (never inside the bb/peace cgroup), serialized across rifts
# with a flock so two benches never share the GPU, and prints one JSON line:
# median sps / learner_sps and per-buffer-step ms for rollout, eval_gpu, eval_env
# over the steady-state updates (first 20 dropped).
#
# Usage: bench_shape.sh <label> [updates=300] [-- extra train.py args...]
#   BOPL_BENCH_WORKERS=128 BOPL_BENCH_BUFFERS=4 BOPL_BENCH_PASSES=4 override the shape.
#   OUT=<dir> (default native/artifacts/rl/bench-shape in this checkout).
# Results: $OUT/<label>/ (run dir, log) and $OUT/results.jsonl (appended).
set -euo pipefail
LABEL=${1:?label}; shift
UPDATES=${1:-300}; [ $# -gt 0 ] && shift
[ "${1:-}" = "--" ] && shift
R=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$R"
OUT=${OUT:-$R/native/artifacts/rl/bench-shape}
mkdir -p "$OUT"
WORKERS=${BOPL_BENCH_WORKERS:-128}
BUFFERS=${BOPL_BENCH_BUFFERS:-4}
PASSES=${BOPL_BENCH_PASSES:-4}
HORIZON=128
STEPS=$((UPDATES * WORKERS * 3 / 2 * HORIZON))  # learner rows x horizon per update
DIR="$OUT/$LABEL"
rm -rf "$DIR"
PUF=$(ls -d "$R"/native/artifacts/rl/dependencies/PufferLib-* | sort | tail -1)
[ -f native/target/release/libbopl_sim.so ] || { echo "libbopl_sim.so missing: cargo build --release in native/" >&2; exit 1; }
ulimit -c 0
UNIT="bopl-bench-$(echo "$LABEL" | tr -c 'a-zA-Z0-9\n' '-')-$$"
echo "$(date -u '+%F %T UTC') $LABEL: waiting for GPU bench lock" >&2
exec 9>/tmp/bopl-gpu-bench.lock
flock 9
echo "$(date -u '+%F %T UTC') $LABEL: running $UPDATES updates (w$WORKERS b$BUFFERS p$PASSES), load $(cut -d' ' -f1-3 /proc/loadavg)" >&2
set +e
BOPL_RL_BACKEND=rust PYTHONPATH="$PUF" \
BOPL_OMP_THREADS=${BOPL_OMP_THREADS:-12} OMP_WAIT_POLICY=${OMP_WAIT_POLICY:-passive} \
systemd-run --user --scope --collect --quiet -p MemoryMax=20G --unit "$UNIT" -- \
  nice -n 5 native/artifacts/rl/venv/bin/python native/rl/puffer/train.py \
  --output "$DIR" --banks 4 --buffers "$BUFFERS" --checkpoint-every 1000000000 --cudagraphs 1 \
  --encoder set --entropy 0.008 --entropy-floor 0.001 --entropy-free-rows 1 \
  --entropy-half-life 200000000 --entropy-head-scale 0.15,1.5,1 --episode-ticks 7200 \
  --no-eval --hidden 128 --horizon "$HORIZON" --layers 1 --learner-passes "$PASSES" \
  --learning-rate 0.00125 --reward-kill 0.27 --reward-timeout -1 \
  --scripted-bots evasive-v2 --scripted-decay 120000000 --scripted-level hard \
  --scripted-opponents mixed-hard,mixed-v2-hard,archer-v2-hard,pursuer-v2-hard,evasive-v2-hard \
  --seed 73 --steps "$STEPS" --training-maps 0,1,2 --workers "$WORKERS" "$@" > "$DIR.log" 2>&1
rc=$?
set -e
if [ $rc -ne 0 ]; then
  echo "{\"label\": \"$LABEL\", \"failed\": true, \"rc\": $rc}" | tee -a "$OUT/results.jsonl"
  tail -n 30 "$DIR.log" >&2
  exit $rc
fi
python3 - "$DIR" "$LABEL" "$WORKERS" "$BUFFERS" "$PASSES" "$HORIZON" <<'PY' | tee -a "$OUT/results.jsonl"
import json, statistics as st, sys, os
d, label, workers, buffers, passes, horizon = sys.argv[1:]
rows = [json.loads(l) for l in open(os.path.join(d, 'metrics.jsonl'))]
rows = [r for r in rows if 'native_perf' in r][20:]
H = int(horizon)
per = {k: round(st.median(r['native_perf'][k] for r in rows) * 1000 / H, 4)
       for k in rows[0]['native_perf'] if k in ('rollout', 'eval_gpu', 'eval_env')}
# With BOPL_ASYNC (default) rollout_s and train_s overlap; wall_s is the update's
# wall time (sps = physical decisions / wall_s), so rollout_s + train_s != wall_s.
out = dict(label=label, workers=int(workers), buffers=int(buffers), passes=int(passes), updates=len(rows),
           sps=round(st.median(r['sps'] for r in rows)), learner_sps=round(st.median(r['learner_sps'] for r in rows)),
           rollout_s=round(st.median(r['rollout_seconds'] for r in rows), 4),
           train_s=round(st.median(r['train_seconds'] for r in rows), 4),
           wall_s=round(st.median(r.get('wall_seconds', r['rollout_seconds'] + r['train_seconds']) for r in rows), 4),
           overlap=rows[-1].get('overlap'),
           ms_per_step=per, cuda_mb=rows[-1].get('cuda_memory_mb'), load1=round(os.getloadavg()[0], 1))
for k in ('explained_variance', 'entropy', 'kl', 'gradient_norm'):
    if k in rows[-1]:
        out[k] = round(rows[-1][k], 4)
print(json.dumps(out))
PY
