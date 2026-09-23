#!/usr/bin/env bash
# The two controls the headline numbers stand or fall on.
#
#   research/divergence/run_divergence_controls.sh <output-dir> [workers]
#
# selfcheck-*   The oracle played against itself, at bit-exact tolerance, over
#               the same maps and the same random loadouts as the study. Any
#               divergence here is the measuring instrument, not the port, so
#               this has to come out at exactly zero for the study to mean
#               anything. Three throwaway comparators have already produced
#               confidently wrong answers on this codebase; this is the check
#               that would have caught them.
#
# random-fixed  The study's random actions on the Dash/Bow/Platform loadout -
#               the narrow loadout the existing gates use. If the divergence
#               rate collapses here relative to random-all, then the gates pass
#               because of what they hold fixed rather than because the port
#               agrees, and the difference between the two cells is the size of
#               that selection bias.
#
# *-noorient    The same cells with the ability-body ORIENTATION fields on the
#               ignore list (FIXTURES.md #1: a live gun-family body re-aims in
#               the port and is latched in the oracle). That one class is the
#               first divergence in about half of all divergent games, so the
#               headline curve mostly measures it; this cell answers the
#               question that actually decides whether to keep auditing - how
#               much later does the SECOND layer of divergence arrive. The
#               unignored `*_any` fields in the same rows still carry the
#               with-orientation curve, so one run gives both.
set -uo pipefail
OUT=${1:?output directory}
WORKERS=${2:-4}
mkdir -p "$OUT"
BIN=${DR_BIN:?set DR_BIN to the divergence_rate binary}
: "${REFERENCE:?}" "${RUST_ROLLOUT_Q32:?}" "${RUST_ROLLOUT:?}"

ORIENT='entity[].hierarchy,entity[].right_x,entity[].right_y,entity[].up_x,entity[].up_y,entity[].rotation'

# name|cell|games|pool|players|stocks|ticks|A|B|C|tol|ignore|fresh
CELLS=$(cat <<LIST
selfcheck-oracle|random|1240|all|2|0|3600|$REFERENCE|$REFERENCE|$REFERENCE|0|$ORIENT|1
selfcheck-q32|random|1240|all|2|0|3600|$RUST_ROLLOUT_Q32|$RUST_ROLLOUT_Q32|$RUST_ROLLOUT_Q32|0|$ORIENT|1
random-fixed|random|4960|fixed|2|0|3600|$REFERENCE|$RUST_ROLLOUT_Q32|$RUST_ROLLOUT|1|entity[].hierarchy|1
random-all-noorient|random|4960|all|2|0|3600|$REFERENCE|$RUST_ROLLOUT_Q32|$RUST_ROLLOUT|1|$ORIENT|1
bot-mixed-v2-noorient|bot:9:2|2480|fixed|2|0|3600|$REFERENCE|$RUST_ROLLOUT_Q32|$RUST_ROLLOUT|1|$ORIENT|1
leak-batched|random|4960|all|2|0|3600|$REFERENCE|$RUST_ROLLOUT_Q32|$RUST_ROLLOUT|1|entity[].hierarchy|0
LIST
)

run_cell() {
    IFS='|' read -r name cell games pool players stocks ticks a b c tol ignore fresh <<<"$1"
    DR_CELL=$cell DR_GAMES=$games DR_POOL=$pool DR_PLAYERS=$players DR_STOCKS=$stocks \
    DR_TICKS=$ticks DR_TOL=$tol DR_IGNORE="$ignore" DR_FRESH=$fresh DR_SEED=${DR_SEED:-24601} DR_PROGRESS=500 \
      nice -n 19 "$BIN" "$a" "$b" "$c" > "$OUT/$name.jsonl" 2> "$OUT/$name.err"
    echo "$name rc=$? $(date +%H:%M:%S)"
}
export -f run_cell
export OUT BIN

echo "$CELLS" | xargs -d '\n' -P "$WORKERS" -I{} bash -c 'run_cell "$@"' _ {}
echo "all controls finished"
