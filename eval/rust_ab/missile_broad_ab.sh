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

# One broad Missile gate over SparkShot, Missile and the represented contact
# matrix: entry/exit, steering and instability, forms, rigid contacts, water,
# scale, duplication, Magnet, Gust, Time Stop and Dash.
for scenario in \
    missile_ability_ab \
    missile_delayed_ground_entry_ab \
    missile_airborne_entry_ab \
    missile_player_ab \
    missile_bow_ab \
    missile_arrow_ab \
    missile_grenade_ab \
    missile_water_ab \
    gust_missile_ab \
    gust_missile_form_ab \
    time_stop_missile_ab \
    duplicator_missile_ab \
    scale_missile_ab \
    magnet_missile_ab \
    magnet_missile_landing_ab \
    dash_gun_forms_ab
do
    cc -O2 -I"$here" -o "$tmp/$scenario" \
        "$here/research/rust_ab/$scenario.c" "$here/bopl_rl.c" -ldl -lm
    (cd "$project" && "$tmp/$scenario" "$reference" "$rust")
done

echo "MISSILE_BROAD_GOLDEN_OK scenarios=16"
