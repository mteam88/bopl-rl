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

# One broad Magnet differential command. These scenarios jointly cover the
# form, wall/player pull, parry, all ordinary ThrowItem2 bodies, Projectile and
# item-layer projectiles, Time Stop and the held-airborne-landing updater path.
for scenario in \
    magnet_ability_ab \
    magnet_parry_ab \
    magnet_item_ab \
    magnet_grenade_ab \
    magnet_smoke_ab \
    magnet_poke_ball_ab \
    magnet_tack_ab \
    magnet_arrow_ab \
    magnet_missile_ab \
    magnet_missile_landing_ab \
    magnet_time_stop_ab \
    gust_magnet_form_ab
do
    cc -O2 -I"$project/sim/abi" -o "$tmp/$scenario" \
        "$project/eval/rust_ab/$scenario.c" "$project/sim/abi/bopl_rl.c" -ldl -lm
    "$tmp/$scenario" "$reference" "$rust"
done

echo "MAGNET_BROAD_GOLDEN_OK scenarios=12"
