#!/usr/bin/env bash
# Interleaved A/B of two sim builds under the trainer's load shape: per build,
# one eval/env_interleave.c copy per hardware thread (K worlds each, stepped
# round-robin), so the sim pays both the world-cycling cache misses and the
# SMT sharing it pays in the pool. The builds alternate per round. Prints each
# round's median sim us per decision over the copies, then whether the hashes
# agree (exact) and the ratio of the medians.
#   eval/perf/env_ilv_ab.sh <base.so> <cand.so> [games=100] [rounds=3] [worlds=23]
set -euo pipefail
cd "$(dirname "$0")/../.."
B=$(realpath "$1"); C=$(realpath "$2"); G=${3:-100}; R=${4:-3}; K=${5:-23}
OUT=${OUT:-/tmp/env-ilv}; make -s env-interleave OUT="$OUT"
CORPUS=$(realpath "${AB_CORPUS:-native/artifacts/rl/env-corpus/test}")
CPUS=$(nproc)
run() {
  for c in $(seq 0 $((CPUS - 1))); do taskset -c "$c" "$OUT/env_interleave" "$1" "$CORPUS" "$G" "$K" 2>/dev/null & done | python3 -c "
import json, statistics as st, sys
d = [json.loads(l) for l in sys.stdin]
print('$2', ','.join(sorted({x['hash'] for x in d})), f\"{st.median(x['sim_us'] for x in d):.3f}\", len(d))"
}
for r in $(seq 1 "$R"); do
  if (( r % 2 )); then order="B C"; else order="C B"; fi
  for w in $order; do so=$B; [[ $w == C ]] && so=$C; run "$so" "$w"; done
done | tee /dev/stderr | python3 -c "
import sys, statistics as st
rows = [l.split() for l in sys.stdin]
h = {w: {r[1] for r in rows if r[0] == w} for w in 'BC'}
s = {w: [float(r[2]) for r in rows if r[0] == w] for w in 'BC'}
print('exact' if h['B'] == h['C'] and len(h['B']) == 1 else f'HASH DIFF {h}')
print(f\"sim median B {st.median(s['B']):.3f} C {st.median(s['C']):.3f} ratio {st.median(s['C'])/st.median(s['B']):.4f}\")"
