#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
    echo "usage: $0 <BoplSimNative.so> <libbopl_sim.so>" >&2
    exit 2
fi

reference=$1
rust=$2
project=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT HUP INT TERM

# One broad Duplicator differential command. Together these scenarios cover
# grounded/airborne form use, platform duplication, every modeled Item fan,
# guided Missile, pickup-driven replacement, Poke Ball capture, Gust, Time
# Stop and Dash interaction. Newly source-ported target classes stay
# provisional until their broad public capture is added.
for scenario in \
    duplicator_ability_ab \
    duplicator_platform_lifetime_ab \
    duplicator_item_ab \
    duplicator_smoke_ab \
    duplicator_tack_ab \
    duplicator_mjolnir_ab \
    duplicator_poke_ab \
    duplicator_poke_capture_ab \
    duplicator_mine_ab \
    duplicator_missile_ab \
    duplicator_pickup_ab \
    gust_gun_forms_ab \
    time_stop_gun_forms_ab \
    dash_gun_forms_ab
do
    cc -O2 -I"$project/sim/abi" -o "$tmp/$scenario" \
        "$project/eval/rust_ab/$scenario.c" "$project/sim/abi/bopl_rl.c" -ldl -lm
    if [ "$scenario" = duplicator_pickup_ab ]; then
        (cd "$project" && BOPL_RL_ABILITY_SPAWNS=1 \
            "$tmp/$scenario" "$reference" "$rust")
    else
        (cd "$project" && "$tmp/$scenario" "$reference" "$rust")
    fi
done

echo "DUPLICATOR_BROAD_GOLDEN_OK scenarios=14"
