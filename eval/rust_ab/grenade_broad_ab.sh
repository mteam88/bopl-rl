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

# One broad Grenade gate over the complete ThrowItem2/Grenade/GrenadeExplode
# source surface: grounded and airborne casts, charge and forced release,
# contacts, fuse, bounces, owner gates, multi-kill, pickup replacement and the
# major cross-ability paths.
for scenario in \
    grenade_ability_ab \
    grenade_air_ab \
    grenade_bounce_ab \
    grenade_fuse_ab \
    grenade_non_owner_ab \
    grenade_multi_kill_ab \
    grenade_forced_exit_ab \
    grenade_pickup_ab \
    grenade_arrow_ab \
    gust_grenade_form_ab \
    gust_grenade_ab \
    magnet_grenade_ab \
    time_stop_throw_forms_ab \
    time_stop_released_items_ab \
    duplicator_item_ab \
    dash_throw_forms_ab
do
    cc -O2 -I"$here" -o "$tmp/$scenario" \
        "$here/research/rust_ab/$scenario.c" "$here/bopl_rl.c" -ldl -lm
    if [ "$scenario" = grenade_pickup_ab ]; then
        (cd "$project" && BOPL_RL_ABILITY_SPAWNS=1 \
            "$tmp/$scenario" "$reference" "$rust")
    else
        (cd "$project" && "$tmp/$scenario" "$reference" "$rust")
    fi
done

echo "GRENADE_BROAD_GOLDEN_OK scenarios=16"
