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

# One Tack gate spans aerial throw, settling, arming, owner-clear/re-entry,
# strict expiry, duplicate traps, Growth/Shrink, Arrow, Gust, Magnet and Time Stop.
for scenario in tack_ability_ab tack_kill_ab duplicator_tack_ab scale_tack_ab \
    tack_arrow_ab gust_tack_ab magnet_tack_ab time_stop_released_items_ab
do
    cc -O2 -I"$here" -o "$tmp/$scenario" \
        "$here/research/rust_ab/$scenario.c" "$here/bopl_rl.c" -ldl -lm
    (cd "$project" && "$tmp/$scenario" "$reference" "$rust")
done

echo "TACK_BROAD_GOLDEN_OK programs=8 ticks=5670"
