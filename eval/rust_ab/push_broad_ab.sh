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

cc -O2 -I"$project/sim/abi" -o "$tmp/push_ability_ab" \
    "$project/eval/rust_ab/push_ability_ab.c" "$project/sim/abi/bopl_rl.c" -ldl -lm
(cd "$project" && "$tmp/push_ability_ab" "$reference" "$rust")

echo "PUSH_BROAD_GOLDEN_OK programs=1 ticks=420"
