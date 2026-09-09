#!/usr/bin/env bash
# A/B the rollout-step changes on branch trainer-step (see sim/RL.md,
# "Trainer step latency", and native/rl/SIM_OVERHAUL.md section 2), interleaved,
# at the 500M run's shape (16 workers x 2 buffers, set encoder, 4 banks):
#   m0  BOPL_BANK_BRANCHES=0, socket Host   (upstream serial bank chain)
#   m1  BOPL_BANK_BRANCHES=1, socket Host   (parallel graph branches)
#   m2  BOPL_BANK_BRANCHES=1, in-process BoplSimNative.so
# Writes one JSON line per run to $OUT/results.jsonl and a summary to
# $OUT/summary.txt.
#
# Usage: bench_trainer_step.sh [rounds] [updates]
#   WAIT_FOR=<systemd user unit>  wait until that unit is inactive first
#   (default: bopl-long-run-500m-dotnet, the live 003 run).
# Run it from its own unit so it is not in the peace.service cgroup (systemd-run
# does not inherit cwd: absolute paths):
#   systemd-run --user --unit bopl-trainer-step-bench -p MemoryMax=14G \
#     -p WorkingDirectory=$PWD /usr/bin/bash $PWD/native/rl/puffer/bench_trainer_step.sh
set -euo pipefail
ROUNDS=${1:-2}
UPDATES=${2:-300}
R=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$R"
OUT=${OUT:-$R/native/artifacts/rl/trainer-step}
mkdir -p "$OUT"
GAME=${BOPL_RL_GAME:-$R/sim/Host/bin/Release/net8.0/linux-x64/publish/Host}
LIB=${BOPL_RL_LIB:-$R/sim/Native/bin/Release/net8.0/linux-x64/publish/BoplSimNative.so}
WAIT_FOR=${WAIT_FOR:-bopl-long-run-500m-dotnet}
if [ -n "$WAIT_FOR" ]; then
  while systemctl --user is-active --quiet "$WAIT_FOR"; do sleep 30; done
  sleep 60  # let its game processes exit and the page cache settle
fi
ulimit -c 0
STEPS=$((UPDATES * 24 * 128))  # learner rows (16 workers * 3/2) * horizon
run() {
  local mode=$1 tag=$2 dir="$OUT/w16b2-m$mode-$tag" branches=1 game="$GAME"
  [ "$mode" = 0 ] && branches=0
  [ "$mode" = 2 ] && game="$LIB"
  rm -rf "$dir"
  BOPL_BANK_BRANCHES=$branches native/artifacts/rl/venv/bin/python native/rl/puffer/train.py \
    --output "$dir" --game "$game" --encoder set --hidden 128 --layers 1 \
    --steps "$STEPS" --seed 73 --workers 16 --buffers 2 --banks 4 --horizon 128 \
    --learner-passes 1 --learning-rate 0.0003 --training-maps 0,1,2 --episode-ticks 7200 --stocks 0 \
    --entropy 0.01 --entropy-floor 0.004 --entropy-half-life 150000000 --reward-timeout -1 --reward-kill 0.1 \
    --checkpoint-every 1000000000 --no-eval > "$dir.log" 2>&1
  python3 - "$dir" "$mode" "$tag" >> "$OUT/results.jsonl" <<'EOF'
import json, statistics as st, sys, os
d, mode, tag = sys.argv[1:]
rows = [json.loads(l) for l in open(os.path.join(d, 'metrics.jsonl'))][20:]
per = {k: st.median(r['native_perf'][k] for r in rows) * 1000 / 128 for k in ('rollout', 'eval_gpu', 'eval_env')}
print(json.dumps(dict(mode=int(mode), tag=tag, updates=len(rows), load1=os.getloadavg()[0],
                      learner_sps_median=st.median(r['learner_sps'] for r in rows),
                      sps_median=st.median(r['sps'] for r in rows), ms_per_step=per)))
EOF
}
for round in $(seq 1 "$ROUNDS"); do
  run 0 "$round"
  run 1 "$round"
  [ -f "$LIB" ] && run 2 "$round"
done
python3 - "$OUT" <<'EOF'
import json, sys, statistics as st
out = sys.argv[1]
rows = [json.loads(l) for l in open(out + '/results.jsonl')]
lines = []
names = {0: 'm0 serial banks, socket', 1: 'm1 parallel banks, socket', 2: 'm2 parallel banks, in-process'}
for mode in (0, 1, 2):
    rs = [r for r in rows if r['mode'] == mode]
    if not rs: continue
    sps = st.median(r['learner_sps_median'] for r in rs)
    ms = {k: st.median(r['ms_per_step'][k] for r in rs) for k in ('rollout', 'eval_gpu', 'eval_env')}
    lines.append(f"{names[mode]}: learner/s median {sps:.0f} over {len(rs)} runs; "
                 f"per step ms rollout {ms['rollout']:.3f} gpu {ms['eval_gpu']:.3f} env {ms['eval_env']:.3f}")
open(out + '/summary.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
EOF
