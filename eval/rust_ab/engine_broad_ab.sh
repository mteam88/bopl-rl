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

# One Engine gate spans ordinary placement, startup, the complete thrust
# curve, composite collision response, form exit, and live thrust across a
# full global Time Stop followed by restoration.
for scenario in engine_ability_ab time_stop_engine_ab
do
    cc -O2 -I"$project/sim/abi" -o "$tmp/$scenario" \
        "$project/eval/rust_ab/$scenario.c" "$project/sim/abi/bopl_rl.c" -ldl -lm
    (cd "$project" && "$tmp/$scenario" "$reference" "$rust")
done

echo "ENGINE_BROAD_GOLDEN_OK scenarios=2 ticks=2040"
