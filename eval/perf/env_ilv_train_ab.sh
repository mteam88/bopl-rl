#!/usr/bin/env bash
# Paired A/B of two writer builds of eval/env_ilv_train.c (make env-ilv-train
# OUT=... from each tree): both run at once, one per SMT sibling of a core
# (CPUS, default 5,11), sides swapped each round. Prints the median ratio of
# sample+finish TSC cycles per decision (cand/base) over the rounds, with the
# finish and sample ratios apart and the sim's as a control (expect 1.00).
#   eval/perf/env_ilv_train_ab.sh <base_bin> <cand_bin> [games=60] [rounds=10] [worlds=23]
# Exactness is a separate run of each with BOPL_ENV_ILV_HASH=1 (same "hash").
set -euo pipefail
cd "$(dirname "$0")/../.."
B=$(realpath "$1"); C=$(realpath "$2"); G=${3:-60}; N=${4:-10}; K=${5:-23}
IFS=, read -r P Q <<<"${CPUS:-5,11}"
SO=$(realpath native/target/release/libbopl_sim.so)
CORPUS=$(realpath "${AB_CORPUS:-native/artifacts/rl/env-corpus/test}")
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
one() { taskset -c "$2" "$1" "$SO" "$CORPUS" "$G" "$K" 2>/dev/null | tail -1 >"$3"; }
for r in $(seq "$N"); do
  if (( r % 2 )); then one "$B" "$P" "$T/b" & one "$C" "$Q" "$T/c" & else one "$B" "$Q" "$T/b" & one "$C" "$P" "$T/c" & fi
  wait; cat "$T/b" "$T/c"
done | python3 -c "
import sys, json, statistics as st
L=[json.loads(l) for l in sys.stdin if l.strip()]
r=[];rf=[];rs=[];b=[];c=[];sim=[]
for x,y in zip(L[0::2],L[1::2]):
    tb=x['sample_cyc']+x['finish_cyc']; tc=y['sample_cyc']+y['finish_cyc']
    r.append(tc/tb); b.append(tb); c.append(tc); sim.append(y['sim_cyc']/x['sim_cyc'])
    rf.append(y['finish_cyc']/x['finish_cyc']); rs.append(y['sample_cyc']/x['sample_cyc'])
print('sample+finish TSC/decision: base %.0f cand %.0f; ratio median %.4f mean %.4f sd %.4f n=%d (finish %.4f sample %.4f) sim %.4f' % (st.median(b), st.median(c), st.median(r), st.mean(r), st.stdev(r) if len(r)>1 else 0, len(r), st.median(rf), st.median(rs), st.median(sim)))"
