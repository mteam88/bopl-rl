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

# One broad Rock gate over grounded rigid motion, lethal Player contact,
# Gust, Time Stop, PortalShot/portal transit and Dash.
for scenario in \
    rock_ability_ab \
    rock_contact_ab \
    gust_rock_ab \
    time_stop_rock_ab \
    portal_rock_ab \
    dash_rock_ab
do
    cc -O2 -I"$here" -o "$tmp/$scenario" \
        "$here/research/rust_ab/$scenario.c" "$here/bopl_rl.c" -ldl -lm
    (cd "$project" && "$tmp/$scenario" "$reference" "$rust")
done

echo "ROCK_BROAD_GOLDEN_OK scenarios=6"
