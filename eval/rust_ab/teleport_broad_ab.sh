#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <BoplSimNative.so> <libbopl_sim.so>" >&2
    exit 2
fi

reference=$1
rust=$2
project=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM

# One broad Teleport gate over marker creation/recast, stationary and moving
# casts, both captured query sets, mixed-team overlap, effect deactivation,
# strict reactivation timing, the cast tick's frozen slime state under every
# aim, and subsequent ordinary airborne movement.
for scenario in teleport_ability_ab teleport_multi_ab teleport_cast_state_ab
do
    cc -O2 -I"$project/sim/abi" -o "$tmp/$scenario" \
        "$project/eval/rust_ab/$scenario.c" "$project/sim/abi/bopl_rl.c" -ldl -lm
    (cd "$project" && "$tmp/$scenario" "$reference" "$rust")
done

echo "TELEPORT_BROAD_GOLDEN_OK scenarios=3"
