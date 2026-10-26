#!/bin/bash
# Drill specialist round 2 decision eval (train/research/arms/drill-r2.toml; DRILL-R2-2026-10.md Section 2), drill.sh with OUT = research/drill-r2:
# the drill_look protocol (native/rl/research/drill-spec/drill_look.py) at the arm's annealed endpoint, the
# specialist on Drill/Dash/Dash with Drill in slot pair % 3 (its training rotation):
#   drill-eval-main  vs l2-main-136002m on iid random kits, 600 seed pairs, seed base 4100001 (P1 primary; P2, P4)
#   drill-eval-me31  vs l2-me31-drill-grenade-2000m on its declared kits, 600 seed pairs, seed base 4300001 (P3)
# Each is one single-threaded CPU unit (CPUWeight 30); the two run side by side. A complete eval (1200 games) is
# kept; anything else is deleted and rerun. train/research/analyze.py drill applies the rules.
#   train/research/arms/drill-r2.sh eval <arm>   (arm: K, R2-s7211, R2-s7212, R2x2, R2v42)
set -euo pipefail
T=$(cd "$(dirname "$0")/../../.." && pwd)
RL=$T/native/artifacts/rl
PY=$RL/venv/bin/python
OUT=$RL/research/drill-r2
[ "${1:-}" = eval ] && [ -n "${2:-}" ] || { echo "usage: drill-r2.sh eval <arm>"; exit 2; }
arm=$2
RUN=$OUT/$arm/anneal
[ "$(python3 -c "import json; print(json.load(open('$RUN/status.json'))['status'])" 2>/dev/null)" = complete ] \
  || { echo "$RUN not complete"; exit 1; }
CK=$RUN/checkpoints/$(ls $RUN/checkpoints | grep -E '^policy-[0-9]+\.bin$' | sort -t- -k2 -n | tail -1)
look() {
  local name=$1; shift
  local d=$OUT/$arm/$name
  if [ -f $d/games.jsonl ] && [ "$(wc -l < $d/games.jsonl)" = 1200 ]; then echo "$d complete"; return 0; fi
  rm -rf $d; mkdir -p $d
  systemd-run --user --wait --collect --unit drill-r2-eval-$arm-$name -p MemoryHigh=8G -p CPUWeight=30 \
    --setenv=BOPL_RL_BACKEND=rust --setenv=OMP_NUM_THREADS=1 \
    --setenv=PYTHONPATH=$RL/dependencies/PufferLib-42f70d6932c30ac977736f861006809c50168ba9 --working-directory=$d \
    /bin/bash -c "$PY $T/native/rl/research/drill-spec/drill_look.py --a $CK --a-kit ddd $* --games 1200 --out $d > $d/log.txt 2>&1"
  [ "$(wc -l < $d/games.jsonl)" = 1200 ] || { echo "$d incomplete"; return 1; }
}
look drill-eval-main --b l2-main-136002m --b-kit random --seed-base 4100001 &
m=$!
look drill-eval-me31 --b l2-me31-drill-grenade-2000m --b-kit "'2,26,3;26,3,2'" --seed-base 4300001 &
e=$!
wait $m && wait $e
