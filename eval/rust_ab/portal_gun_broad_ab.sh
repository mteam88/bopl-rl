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

# One broad Portal gun gate over shot flight/wall landing, linked-mouth player
# transit, Arrow and active-Rock rigid bodies, hierarchy-ordered pickup motion,
# recasts, lifetime expiry, and Time Stop across a live pair.
for scenario in \
    portal_gun_ability_ab \
    portal_arrow_ab \
    portal_pickup_ab \
    portal_rock_ab \
    time_stop_portal_ab
do
    cc -O2 -I"$here" -o "$tmp/$scenario" \
        "$here/research/rust_ab/$scenario.c" "$here/bopl_rl.c" -ldl -lm
    (cd "$project" && "$tmp/$scenario" "$reference" "$rust")
done

echo "PORTAL_GUN_BROAD_GOLDEN_OK scenarios=5"
