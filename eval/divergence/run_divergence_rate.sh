#!/usr/bin/env bash
# The whole divergence-rate study: every cell, sharded across a few workers.
#
#   eval/divergence/run_divergence_rate.sh <output-dir> [workers]
#
# Reads REFERENCE / RUST_ROLLOUT / RUST_ROLLOUT_Q32 from the environment (the
# Makefile target supplies them). Writes one .jsonl per cell; nothing per-tick
# ever touches the disk, so the whole study is a few tens of megabytes.
#
# Cells, and why each exists:
#
#   random-all      random actions, loadouts drawn from all 38 usable abilities.
#                   The broad unbiased sample. Reaches state no played game does.
#   random-train    the same with the training pool (less Time Stop and
#                   Revival, both unfinished ports): the number that describes training today.
#   random-6p       six players, for the interaction density two never reach.
#   random-stocks5  five stocks, so games last long enough to measure the
#                   1000- and 3600-tick tail instead of censoring it away.
#   bot-*           the scripted opponents on their native Dash/Bow/Platform
#                   loadout (bopl_bots.h:143 hardcodes those slots): realistic
#                   play, which concentrates in a different part of the state
#                   space than random flailing.
#   botpool-mixedv2 mixed-v2-hard with random loadouts. The bots aim and move
#                   as if slot 1 were a bow, so this is NOT realistic play; it
#                   is competent movement crossed with the full ability space,
#                   and it is reported as its own cell for that reason.
set -uo pipefail
OUT=${1:?output directory}
WORKERS=${2:-4}
mkdir -p "$OUT"
BIN=${DR_BIN:?set DR_BIN to the divergence_rate binary}
: "${REFERENCE:?}" "${RUST_ROLLOUT_Q32:?}" "${RUST_ROLLOUT:?}"

# name|cell|games|pool|players|stocks|ticks
CELLS=$(cat <<'LIST'
random-all|random|9920|all|2|0|3600
random-train|random|4960|train|2|0|3600
random-6p|random|2480|all|6|0|3600
random-stocks5|random|3720|all|2|5|3600
bot-mixed-hard|bot:4:2|2480|fixed|2|0|3600
bot-mixed-v2-hard|bot:9:2|2480|fixed|2|0|3600
bot-archer-v2-hard|bot:6:2|1240|fixed|2|0|3600
bot-pursuer-v2-hard|bot:7:2|1240|fixed|2|0|3600
bot-evasive-v2-hard|bot:8:2|1240|fixed|2|0|3600
bot-search-medium|bot:5:1|620|fixed|2|0|3600
bot-search-hard|bot:5:2|620|fixed|2|0|3600
bot-mixed-v2-stocks5|bot:9:2|1240|fixed|2|5|3600
botpool-mixed-v2-hard|bot:9:2|2480|all|2|0|3600
LIST
)

run_cell() {
    local line=$1
    IFS='|' read -r name cell games pool players stocks ticks <<<"$line"
    DR_CELL=$cell DR_GAMES=$games DR_POOL=$pool DR_PLAYERS=$players \
    DR_STOCKS=$stocks DR_TICKS=$ticks DR_TOL=${DR_TOL:-1} DR_SEED=${DR_SEED:-24601} \
    DR_PROGRESS=500 \
      nice -n 19 "$BIN" "$REFERENCE" "$RUST_ROLLOUT_Q32" "$RUST_ROLLOUT" \
      > "$OUT/$name.jsonl" 2> "$OUT/$name.err"
    echo "$name rc=$? games=$games $(date +%H:%M:%S)"
}
export -f run_cell
export OUT BIN REFERENCE RUST_ROLLOUT_Q32 RUST_ROLLOUT

echo "$CELLS" | xargs -d '\n' -P "$WORKERS" -I{} bash -c 'run_cell "$@"' _ {}
echo "all cells finished"
