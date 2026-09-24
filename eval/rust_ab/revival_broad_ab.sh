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

# One Revival gate spans ordinary cast/recast lifetime, a complete
# marker-driven death and replacement spawn, and pickup replacement followed
# by an immediate airborne cast of the newly equipped prefab.
for scenario in revival_ability_ab revival_death_ab ability_pickup_cast_ab
do
    cc -O2 -Wall -Wextra -Werror -I"$project/sim/abi" -o "$tmp/$scenario" \
        "$project/eval/rust_ab/$scenario.c" "$project/sim/abi/bopl_rl.c" -ldl -lm
    if [ "$scenario" = ability_pickup_cast_ab ]; then
        (cd "$project" && BOPL_RL_ABILITY_SPAWNS=1 \
            "$tmp/$scenario" "$reference" "$rust")
    else
        (cd "$project" && "$tmp/$scenario" "$reference" "$rust")
    fi
done

echo "REVIVAL_BROAD_GOLDEN_OK programs=3 ticks=3835"
