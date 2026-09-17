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

# One Spike gate covers ordinary placement, composite attachment, down-thrust,
# platform impulse, form recovery, persistent lifetime, live Growth/Shrink
# repositioning, and the active form's nonlethal Dash callback.
for scenario in spike_ability_ab scale_spike_ab dash_remaining_forms_ab
do
    cc -O2 -I"$here" -o "$tmp/$scenario" \
        "$here/research/rust_ab/$scenario.c" "$here/bopl_rl.c" -ldl -lm
    (cd "$project" && "$tmp/$scenario" "$reference" "$rust")
done

echo "SPIKE_BROAD_GOLDEN_OK programs=3 ticks=4740"
