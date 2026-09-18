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

# One broad Roll gate over grounded charge/release and stock reconstruction,
# airborne short-jump exit, lethal Dash contact, Gust force and a complete
# charging-form Time Stop freeze/restoration.
for scenario in roll_ability_ab roll_air_release_ab dash_roll_ab gust_roll_ab time_stop_roll_ab
do
    cc -O2 -I"$here" -o "$tmp/$scenario" \
        "$here/research/rust_ab/$scenario.c" "$here/bopl_rl.c" -ldl -lm
    (cd "$project" && "$tmp/$scenario" "$reference" "$rust")
done

echo "ROLL_BROAD_GOLDEN_OK scenarios=5 ticks=3000"
