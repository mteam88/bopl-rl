#!/usr/bin/env bash
# Exact gate between two f64 sim builds: divergence_rate random play over all
# 38 abilities on all 124 maps (the ability sweep's stratification), 2p and 6p,
# DR_TOL=0, reference = base, both candidates = cand. Any divergence, crash or
# ended-apart game fails. usage: exact_gate.sh <base.so> <cand.so> <out> [shards=6]
set -uo pipefail
cd "$(dirname "$0")/../.."
B=$(realpath "$1"); C=$(realpath "$2"); OUT=$3; S=${4:-6}
make -s native/artifacts/rl/divergence_rate
mkdir -p "$OUT"; BIN=$(realpath native/artifacts/rl/divergence_rate)
# cell|games|players  (the ability sweep's 80/map 2p and 20/map 6p; DR_HOLD so held casts play)
pids=()
for cell in "2p|${GATE_2P:-9920}|2" "6p|${GATE_6P:-2480}|6"; do
  IFS='|' read -r name games players <<<"$cell"
  per=$(( (games + S/2 - 1) / (S/2) ))
  for ((k=0; k<S/2; k++)); do
    first=$((k*per)); n=$(( first+per > games ? games-first : per ))
    DR_CELL=random DR_GAMES=$n DR_FIRST=$first DR_POOL=all DR_MAPS=all DR_PLAYERS=$players DR_STOCKS=0 \
    DR_TICKS=3600 DR_TOL=0 DR_FRESH=0 DR_HOLD=1 DR_SEED=${GATE_SEED:-24601} \
      nice -n 10 "$BIN" "$B" "$C" "$C" > "$OUT/$name-$k.jsonl" 2> "$OUT/$name-$k.err" &
    pids+=($!)
  done
done
wait "${pids[@]}"
games=$(grep -h -c '"record":"game"' "$OUT"/*.jsonl | paste -sd+ | bc)
fails=$(cat "$OUT"/*.jsonl | grep -c '"record":"fail"')
div=$(cat "$OUT"/*.jsonl | grep -c '_any":{')
apart=$(cat "$OUT"/*.jsonl | grep -c '"end":2')
if [[ $fails == 0 && $div == 0 && $apart == 0 && $games -gt 0 ]]; then echo "EXACT_GATE_OK games=$games"; else echo "EXACT_GATE_FAIL games=$games fails=$fails divergent=$div apart=$apart"; exit 1; fi
