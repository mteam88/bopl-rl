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

# The ordinary gate covers quick and held pickup paths, ground consumption,
# launch, collision, death and stock recovery. The composition gate adds a
# live boulder alongside Portal Shot and Black Hole under Gust shockwaves.
for scenario in macho_throw_ability_ab gust_broad_ab
do
    cc -O2 -I"$project/sim/abi" -o "$tmp/$scenario" \
        "$project/eval/rust_ab/$scenario.c" "$project/sim/abi/bopl_rl.c" -ldl -lm
    (cd "$project" && "$tmp/$scenario" "$reference" "$rust")
done

echo "THROW_BROAD_GOLDEN_OK programs=2 ticks=1430"
