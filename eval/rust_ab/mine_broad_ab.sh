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

# One broad Mine gate over grounded and pickup casts, autonomous state
# transitions, owner scan immunity and owner ignition, active forms, Gust,
# Time Stop, duplication and shared ThrowItem entry/exit behavior.
for scenario in \
    mine_ability_ab \
    mine_owner_contact_ab \
    gust_mine_ab \
    gust_mine_form_ab \
    time_stop_mine_ab \
    time_stop_mine_anticipation_ab \
    time_stop_mine_form_ab \
    duplicator_mine_ab \
    dash_throw_forms_ab
do
    cc -O2 -I"$here" -o "$tmp/$scenario" \
        "$here/research/rust_ab/$scenario.c" "$here/bopl_rl.c" -ldl -lm
    (cd "$project" && "$tmp/$scenario" "$reference" "$rust")
done

echo "MINE_BROAD_GOLDEN_OK scenarios=9"
