#!/usr/bin/env bash
# Interleaved A/B of two sim builds on eval/env_bench.c: the same games on the
# same core, the order alternating per round so drifting host load hits both
# alike. Prints each run (game-set hash, sim and env us per decision, load1),
# then whether the hashes agree (the exact gate) and the median sim ratio.
#   eval/perf/env_ab.sh <base.so> <cand.so> [games=100] [rounds=3] [core=11]
# The corpus defaults to the held-out half, native/artifacts/rl/env-corpus/test
# (eval/perf/record_corpus.py; games 200-399, PGO trains on 0-199).
set -euo pipefail
cd "$(dirname "$0")/../.."
B=$(realpath "$1"); C=$(realpath "$2"); G=${3:-100}; R=${4:-3}; CORE=${5:-11}
BENCH=native/artifacts/rl/env_bench; CORPUS=${AB_CORPUS:-native/artifacts/rl/env-corpus/test}
make -s env-bench
run() { taskset -c "$CORE" "$BENCH" "$1" "$CORPUS" "$G" 2>/dev/null | tail -1; }
for r in $(seq 1 "$R"); do
  if (( r % 2 )); then order="B C"; else order="C B"; fi
  for w in $order; do
    so=$B; [[ $w == C ]] && so=$C
    run "$so" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print('$w', d['hash'], f\"sim {d['sim_us']:.2f} lite {d['env_lite_us']:.2f} full {d['env_full_us']:.2f} load {d['load1']}\")"
  done
done | tee /dev/stderr | python3 -c "
import sys, statistics as st
rows=[l.split() for l in sys.stdin]
h={w:{r[1] for r in rows if r[0]==w} for w in 'BC'}
s={w:[float(r[3]) for r in rows if r[0]==w] for w in 'BC'}
print('exact' if h['B']==h['C'] and len(h['B'])==1 else f'HASH DIFF {h}')
print(f\"sim median B {st.median(s['B']):.2f} C {st.median(s['C']):.2f} ratio {st.median(s['C'])/st.median(s['B']):.3f}\")"
