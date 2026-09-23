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

# One Tesla gate spans placement and ring eviction, parent/child connection
# and arming, player/item/Smoke contacts, composite motion, the plant-exit
# stop/idle branch, and an active lightning pair across global Time Stop and
# restoration.
for scenario in tesla_ability_ab tesla_contact_ab tesla_exit_state_ab time_stop_tesla_ab
do
    cc -O2 -I"$here" -o "$tmp/$scenario" \
        "$here/research/rust_ab/$scenario.c" "$here/bopl_rl.c" -ldl -lm
    (cd "$project" && "$tmp/$scenario" "$reference" "$rust")
done

echo "TESLA_BROAD_GOLDEN_OK scenarios=4 ticks=4300"
