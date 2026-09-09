#!/usr/bin/env bash
# SIM_OVERHAUL.md section 4: how far does the lockstep scale with games now
# that many worlds fit in one runtime? Sweeps --workers at --buffers 2 on the
# in-process .so (all worlds in one runtime) and, where memory allows, the
# socket Host for reference. One JSON line per run in $OUT/results.jsonl.
#
# Usage: bench_scale.sh [updates] [-- "<workers> <omp-threads> <wait-policy> <mode> [worlds-per-runtime]" ...]
# Run as its own unit (never inside the bb service cgroup):
#   systemd-run --user --unit bopl-scale-bench -p MemoryMax=16G \
#     -p WorkingDirectory=$PWD /usr/bin/bash $PWD/native/rl/puffer/bench_scale.sh
set -euo pipefail
UPDATES=${1:-300}
R=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$R"
export BOPL_SIM_EXPORT=$R
OUT=${OUT:-$R/native/artifacts/rl/scale}
mkdir -p "$OUT"
GAME=$R/sim/Host/bin/Release/net8.0/linux-x64/publish/Host
LIB=$R/sim/Native/bin/Release/net8.0/linux-x64/publish/BoplSimNative.so
ulimit -c 0
run() { # workers omp-threads wait-policy mode(so|socket) [worlds-per-runtime, default all]
  local workers=$1 omp=$2 wait=$3 mode=$4 per=${5:-0}
  local tag="w${workers}-omp${omp}-${wait}-${mode}"
  [ "$per" != 0 ] && tag="$tag-per$per"
  local dir="$OUT/$tag" game="$LIB"
  [ "$mode" = socket ] && game="$GAME"
  local steps=$((UPDATES * workers * 3 / 2 * 128))
  rm -rf "$dir"
  if ! BOPL_BANK_BRANCHES=1 BOPL_OMP_THREADS=$omp OMP_WAIT_POLICY=$wait BOPL_RL_WORLDS_PER_RUNTIME=$per \
    native/artifacts/rl/venv/bin/python native/rl/puffer/train.py \
    --output "$dir" --game "$game" --encoder set --hidden 128 --layers 1 \
    --steps "$steps" --seed 73 --workers "$workers" --buffers 2 --banks 4 --horizon 128 \
    --learner-passes 1 --learning-rate 0.0003 --training-maps 0,1,2 --episode-ticks 7200 --stocks 0 \
    --entropy 0.01 --entropy-floor 0.004 --entropy-half-life 150000000 --reward-timeout -1 --reward-kill 0.1 \
    --checkpoint-every 1000000000 --no-eval > "$dir.log" 2>&1; then
    echo "{\"tag\": \"$tag\", \"failed\": true}" >> "$OUT/results.jsonl"; return 0
  fi
  python3 - "$dir" "$tag" "$workers" "$omp" "$wait" "$mode" "$per" >> "$OUT/results.jsonl" <<'PY'
import json, statistics as st, sys, os
d, tag, workers, omp, wait, mode, wpr = sys.argv[1:]
rows = [json.loads(l) for l in open(os.path.join(d, 'metrics.jsonl'))][20:]
per = {k: st.median(r['native_perf'][k] for r in rows) * 1000 / 128 for k in ('rollout', 'eval_gpu', 'eval_env')}
print(json.dumps(dict(tag=tag, workers=int(workers), omp=int(omp), wait=wait, mode=mode, worlds_per_runtime=int(wpr), updates=len(rows),
                      load1=os.getloadavg()[0], learner_sps_median=st.median(r['learner_sps'] for r in rows),
                      sps_median=st.median(r['sps'] for r in rows), ms_per_step=per)))
PY
}
if [ "${2:-}" = "--" ]; then shift 2; for spec in "$@"; do run $spec; done
else
  run 16 16 active so
  run 32 32 active so
  run 64 64 active so
  run 64 16 passive so
  run 128 128 active so
  run 128 16 passive so
  run 32 32 active socket
fi
python3 - "$OUT" <<'EOF2'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1] + '/results.jsonl')]
lines = []
for r in rows:
    if r.get('failed'): lines.append(f"{r['tag']}: FAILED"); continue
    m = r['ms_per_step']
    lines.append(f"{r['tag']:28s} learner/s {r['learner_sps_median']:7.0f}  rows/s {r['sps_median']:7.0f}  "
                 f"ms/step rollout {m['rollout']:.3f} gpu {m['eval_gpu']:.3f} env {m['eval_env']:.3f}  load {r['load1']:.0f}")
open(sys.argv[1] + '/summary.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
EOF2
