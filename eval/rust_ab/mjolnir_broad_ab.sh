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

# One broad Mjolnir gate over airborne flight/recall, platform clang and push,
# Gust, Time Stop, duplication and the shared ThrowItem form callback.
for scenario in \
    mjolnir_ability_ab \
    mjolnir_wall_ab \
    gust_mjolnir_ab \
    time_stop_mjolnir_ab \
    duplicator_mjolnir_ab \
    dash_throw_forms_ab
do
    cc -O2 -I"$here" -o "$tmp/$scenario" \
        "$here/research/rust_ab/$scenario.c" "$here/bopl_rl.c" -ldl -lm
    (cd "$project" && "$tmp/$scenario" "$reference" "$rust")
done

echo "MJOLNIR_BROAD_GOLDEN_OK scenarios=6"
