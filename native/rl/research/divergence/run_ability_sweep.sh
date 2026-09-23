#!/usr/bin/env bash
# The randomised-loadout gate: every ability, on every map, with no oracle.
#
#   research/divergence/run_ability_sweep.sh <output-dir> [workers]
#
# Why this exists. Every other gate in the suite plays the hardcoded
# Dash/Bow/Platform kit (puffer/bopl_bots.h:143-149, DR_POOL=fixed), so the
# whole suite only ever exercises 3 of the 38 abilities. Three distinct real
# defects were found in that blind spot and none of them was reachable from the
# gate loadout: the cross-episode rope-pool leak (FIXTURES.md #2, 15.0% of
# full-pool episodes and 0 of 1240 fixed-kit ones), the replay_ab
# order-sensitivity it caused (#3), and a hard panic on vector-field maps
# triggered by Black hole (#4, 0 crashes in 13,000+ fixed-kit bot episodes).
# This gate draws loadouts uniformly from all 38 abilities instead.
#
# Stratification. divergence_rate assigns maps by `index % map_count`, not by
# drawing, so a game count that is a multiple of 124 covers every authored map
# exactly equally - which matters because backend disagreement is map-localised
# (the checkpoint-transfer study found one specialist's entire strength deficit
# on two held-out maps). Abilities are then drawn iid per slot within each map,
# and the cell sizes below are chosen so that the product grid closes: every
# one of the 37 resolvable abilities is equipped on every one of the 124 maps,
# 4588/4588 cells, which the gate asserts rather than assumes. 2p needs 80
# games/map for that (it saturates at 60; 10/map covers only 81%), 6p needs 20
# because it draws 18 slots a game instead of 6.
#
# No oracle, on purpose, for three reasons: it is 40x faster, so the gate can
# afford ten thousand episodes instead of a few hundred; the port-vs-oracle
# full-pool comparison cannot be zero today (FIXTURES.md #1/#1b are open), so it
# could not be a pass/fail gate without a baseline to drift against; and the
# oracle has a crash of its own on the full pool (FIXTURES.md #4, map 27
# ShootDuplicator NullReferenceException), so putting it in a crash gate would
# make the port's gate fail on a .NET defect.
#
# The cells, and what each one can catch:
#
#   leak-*        The library against two more copies of itself with the two
#                 candidate handles reopened per episode (DR_FRESH=1) and the
#                 reference handle reused. Two copies of the same build, given
#                 the same seed and the same actions, can only disagree because
#                 one handle had played earlier episodes. That is exactly the
#                 cross-episode leak, with the port/oracle comparison removed:
#                 338/1240 before the rope-pool fix, 0 after.
#   selfcheck-*   The same games with all three handles reused equally
#                 (DR_FRESH=0). Every handle now sees the same history, so this
#                 must be zero by construction. It is the control that makes a
#                 leak-* verdict mean anything - if it fires, the simulator is
#                 nondeterministic and the leak probe is measuring noise - and
#                 it is the only check in the suite that would notice that.
#
# Both cells also count crashed episodes, which is how the vector-field panic
# is caught: divergence_rate reports a step failure as {"record":"fail"} and
# attributes it to a map, seed and loadout.
#
# Everything is pinned: DR_SEED, DR_FIRST=0, and map assignment by index. Two
# runs of the same commit produce identical .jsonl apart from the `seconds`
# field of each cell's closing {"record":"done"} line, which is wall clock:
#   diff <(sed 's/,"seconds":[0-9.]*//' a.jsonl) <(sed 's/,"seconds":[0-9.]*//' b.jsonl)
set -uo pipefail
OUT=${1:?output directory}
WORKERS=${2:-4}
mkdir -p "$OUT"
BIN=${DR_BIN:?set DR_BIN to the divergence_rate binary}
: "${RUST_ROLLOUT_Q32:?}"
SEED=${ABILITY_SWEEP_SEED:-24601}

# name|games|players|fresh
CELLS=$(cat <<LIST
leak-2p|9920|2|1
selfcheck-2p|9920|2|0
leak-6p|2480|6|1
selfcheck-6p|2480|6|0
LIST
)

run_cell() {
    IFS='|' read -r name games players fresh <<<"$1"
    DR_CELL=random DR_GAMES=$games DR_FIRST=0 DR_POOL=all DR_MAPS=all \
    DR_PLAYERS=$players DR_STOCKS=0 DR_TICKS=3600 DR_TOL=0 DR_FRESH=$fresh \
    DR_SEED=$SEED \
      nice -n 19 "$BIN" "$RUST_ROLLOUT_Q32" "$RUST_ROLLOUT_Q32" "$RUST_ROLLOUT_Q32" \
      > "$OUT/$name.jsonl" 2> "$OUT/$name.err"
}
export -f run_cell
export OUT BIN RUST_ROLLOUT_Q32 SEED

echo "$CELLS" | xargs -d '\n' -P "$WORKERS" -I{} bash -c 'run_cell "$@"' _ {}

# ---- verdict -------------------------------------------------------------
# `*_any` is the ignore-list-INDEPENDENT hit (divergence_compare.h:142 only
# filters the `*_anyx` copy), so this compares every field the ABI publishes,
# orientation and hierarchy included. For a library against itself there is no
# field that is allowed to differ.
status=0
total_episodes=0
while IFS='|' read -r name games players fresh; do
    file=$OUT/$name.jsonl
    read -r abilities cells < <(awk '
        /"record":"game"/ {
            match($0, /"map":[0-9]+/); m = substr($0, RSTART+6, RLENGTH-6)
            s = index($0, "\"resolved\":["); e = index($0, "\"activations\":[")
            r = substr($0, s+11, e-s-13)
            gsub(/[^0-9]/, " ", r)
            n = split(r, p, " ")
            for (i = 1; i <= n; i++) { ab[p[i]] = 1; cell[m " " p[i]] = 1 }
        }
        END { na = 0; for (k in ab) na++; nc = 0; for (k in cell) nc++; print na, nc }' "$file")
    played=$(sed -n 's/.*"record":"done".*"games":\([0-9]*\).*/\1/p' "$file")
    crashes=$(grep -c '"record":"fail"' "$file"); crashes=${crashes:-0}
    divergent=$(grep -c '_any":{' "$file"); divergent=${divergent:-0}
    apart=$(grep -c '"end":2' "$file"); apart=${apart:-0}
    maps=$(sed -n 's/.*"record":"run".*,"maps":\([0-9]*\),.*/\1/p' "$file")
    expect_cells=$((maps * abilities))
    bad=
    [ "$((played + crashes))" = "$games" ] || bad="$bad episodes-lost"
    [ "$crashes" = 0 ] || bad="$bad crashes"
    [ "$divergent" = 0 ] || bad="$bad divergent"
    [ "$apart" = 0 ] || bad="$bad ended-apart"
    [ "$maps" = 124 ] || bad="$bad maps"
    [ "$abilities" = 37 ] || bad="$bad abilities"
    [ "$cells" = "$expect_cells" ] || bad="$bad coverage"
    total_episodes=$((total_episodes + played + crashes))
    if [ -n "$bad" ]; then
        status=1
        echo "ABILITY_SWEEP_FAIL cell=$name players=$players fresh=$fresh episodes=$played crashes=$crashes divergent=$divergent ended_apart=$apart maps=$maps abilities=$abilities cells=$cells/$expect_cells ($bad )"
        grep '"record":"fail"' "$file" | head -5
        grep -o '"i":[0-9]*,"map":[0-9]*,"seed":[0-9]*.*"q32_all_any":{[^}]*}' "$file" | head -5
    else
        echo "  cell=$name players=$players fresh=$fresh episodes=$played crashes=0 divergent=0 cells=$cells/$expect_cells ok"
    fi
done <<<"$CELLS"

if [ $status = 0 ]; then
    echo "ABILITY_SWEEP_OK episodes=$total_episodes maps=124 abilities=37 ability_map_cells=4588 crashes=0 divergent=0 seed=$SEED"
else
    echo "ABILITY_SWEEP_FAILED seed=$SEED"
fi
exit $status
