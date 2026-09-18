#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <BoplSimNative.so> <libbopl_sim.so>" >&2
    exit 2
fi

reference=$1
rust=$2
here=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
project=$(CDPATH= cd -- "$here/../.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM

# One Drill gate spans entry/exit, air and terrain motion, lethal hitboxes,
# bounds/stock reconstruction, Time Stop, a live Web Swing endpoint and the
# pushable PlayerBody callback from Gust's source Shockwave.
for scenario in drill_ability_ab drill_rope_ab gust_drill_ab
do
    cc -O2 -Wall -Wextra -Werror -I"$here" -o "$tmp/$scenario" \
        "$here/research/rust_ab/$scenario.c" "$here/bopl_rl.c" -ldl -lm
    (cd "$project" && "$tmp/$scenario" "$reference" "$rust")
done

echo "DRILL_BROAD_GOLDEN_OK programs=3 ticks=3229"
