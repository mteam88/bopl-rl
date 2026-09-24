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

# One Poke Ball gate spans airborne throw/fizzle, live capture/ejection,
# duplicate callbacks, Growth/Shrink, Arrow, Gust, Magnet and Time Stop.
for scenario in poke_ball_ability_ab poke_ball_capture_ab \
    duplicator_poke_capture_ab scale_poke_ab poke_ball_arrow_ab \
    gust_poke_ball_ab magnet_poke_ball_ab time_stop_poke_capture_ab
do
    cc -O2 -I"$project/sim/abi" -o "$tmp/$scenario" \
        "$project/eval/rust_ab/$scenario.c" "$project/sim/abi/bopl_rl.c" -ldl -lm
    (cd "$project" && "$tmp/$scenario" "$reference" "$rust")
done

echo "POKE_BALL_BROAD_GOLDEN_OK programs=8 ticks=4668"
