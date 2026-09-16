#!/usr/bin/env bash
# Policy v4 throughput smoke (POLICY_V4_PLAN.md section 6): the 024 recipe's
# rollout shape (128 workers x 4 buffers, 4 banks, 3 scripted, horizon 128,
# learner passes 4) for three policies, back to back, no evaluation:
#   set-024   the v3 set encoder at 024's shape (hidden 128, 1 layer): control
#   v4-full   policy v4 defaults (hidden 256, set 128/128, FFN 512, critic 512 privileged)
#   v4-lite   set inner 64, FFN off, critic 384
# Writes one JSON line per run to $OUT/results.jsonl with the medians over the
# updates after warm-up: learner_sps, sps, the trainer's per-step rollout /
# eval_gpu / eval_env times (ms per rollout step) and train_seconds per update,
# plus the v4 set overflow. A run that exits non-zero is recorded with its exit
# code and the tail of its log, and the script exits 1 at the end.
#
# Memory: at this shape v3 takes ~2.6 GB of device memory and v4 ~6 GB
# (BOPL_V4_MEMLOG=1 prints the allocations); the runs wait for $NEED_MB free
# (default 7000) up to $WAIT_S seconds (default 600) so a co-tenant does not
# turn the smoke into an out-of-memory crash.
# Meant to run from the GPU queue (one job at a time):
#   bash native/rl/puffer/bench_v4_smoke.sh [updates]
#   WORKERS=16 BUFFERS=1 NEED_MB=1500 bash ... 10     scaled-down end-to-end check
set -uo pipefail
UPDATES=${1:-600}
WORKERS=${WORKERS:-128}
BUFFERS=${BUFFERS:-4}
NEED_MB=${NEED_MB:-7000}
WAIT_S=${WAIT_S:-600}
R=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$R"
OUT=${OUT:-$R/native/artifacts/rl/v4-smoke}
mkdir -p "$OUT"
export BOPL_RL_BACKEND=rust BOPL_V4_MEMLOG=1
export PYTHONPATH=$R/native/artifacts/rl/dependencies/PufferLib-42f70d6932c30ac977736f861006809c50168ba9
STEPS=$((UPDATES * WORKERS * 3 / 2 * 128))   # learner rows per update = workers * 3/2 * horizon
COMMON=(--aim-fourier 4 --banks 4 --buffers "$BUFFERS" --checkpoint-every 1000000000 --cudagraphs 1
        --entropy 0.01532588085540654 --entropy-floor 0.001 --entropy-free-rows 1 --entropy-half-life 718611646
        --entropy-head-scale 0.028821467398601395,1.5,1 --episode-ticks 7200 --horizon 128 --learner-passes 4
        --learning-rate 0.0003431012662975937 --reward-kill 0.2883693237829292 --reward-timeout -1
        --scripted-banks 3 --scripted-opponents mixed-hard,mixed-v2-hard,archer-v2-hard,pursuer-v2-hard,evasive-v2-hard
        --scripted-roster default --scripted-share-max 1 --scripted-share-min 0.11048204896277423
        --seed 73 --training-maps 0,1,2 --workers "$WORKERS" --no-eval --steps "$STEPS")
failed=0
free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }
wait_for_memory() {
  local waited=0 free
  while :; do
    free=$(free_mb)
    [ "${free:-0}" -ge "$NEED_MB" ] && return 0
    if [ "$waited" -ge "$WAIT_S" ]; then
      echo "$(date -Is) only $free MB of device memory free after ${WAIT_S}s (need $NEED_MB); co-tenants:"
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
      return 1
    fi
    sleep 15; waited=$((waited + 15))
  done
}
run() {
  local name=$1; shift
  local dir="$OUT/$name" rc=0
  rm -rf "$dir" "$dir.log"
  if ! wait_for_memory; then
    echo "{\"name\": \"$name\", \"updates\": 0, \"error\": \"device memory\", \"free_mb\": $(free_mb)}" >> "$OUT/results.jsonl"
    failed=1; return
  fi
  echo "$(date -Is) $name ($(free_mb) MB free)"
  native/artifacts/rl/venv/bin/python native/rl/puffer/train.py --output "$dir" "${COMMON[@]}" "$@" > "$dir.log" 2>&1 || rc=$?
  [ "$rc" -ne 0 ] && { echo "$(date -Is) $name exited $rc"; grep -v '^\s*$' "$dir.log" | tail -4 | cut -c1-300; failed=1; }
  python3 - "$dir" "$name" "$rc" >> "$OUT/results.jsonl" <<'EOF'
import json, statistics as st, sys, os
d, name, rc = sys.argv[1:]
rc = int(rc)
def tail():
    try:
        return [l.rstrip()[:300] for l in open(d + '.log') if l.strip()][-4:]
    except FileNotFoundError:
        return []
try:
    rows = [json.loads(l) for l in open(os.path.join(d, 'metrics.jsonl'))]
except FileNotFoundError:
    print(json.dumps(dict(name=name, updates=0, exit=rc, error='no metrics', log_tail=tail()))); raise SystemExit
rows = [r for r in rows[10:] if r['native_perf'].get('rollout')]
med = lambda key: st.median(r[key] for r in rows) if rows else None
per = {k: st.median(r['native_perf'][k] for r in rows) * 1000 / 128 for k in ('rollout', 'eval_gpu', 'eval_env')} if rows else {}
overflow = [max(r.get('set_overflow', [0, 0, 0, 0])[i] for r in rows) for i in range(4)] if rows and rows[0].get('set_overflow') else None
print(json.dumps(dict(name=name, updates=len(rows), exit=rc, load1=os.getloadavg()[0], learner_sps=med('learner_sps'), sps=med('sps'),
                      ms_per_rollout_step=per, train_seconds=med('train_seconds'), wall_seconds=med('wall_seconds'),
                      cuda_memory_mb=med('cuda_memory_mb'), set_overflow_max=overflow, **({} if rc == 0 else dict(log_tail=tail())))))
EOF
}
run set-024 --encoder set --hidden 128 --layers 1
run v4-full --encoder v4 --hidden 256 --layers 1
run v4-lite --encoder v4 --hidden 256 --layers 1 --v4-set-inner 64 --v4-ffn-dim 0 --v4-critic-dim 384
tail -3 "$OUT/results.jsonl"
exit $failed
