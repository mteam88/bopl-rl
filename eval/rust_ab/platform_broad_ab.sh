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

for scenario in platform_ab platform_phase_ab platform_boundary_ab platform_rotation_ab
do
    cc -O2 -I"$project/sim/abi" -o "$tmp/$scenario" \
        "$project/eval/rust_ab/$scenario.c" "$project/sim/abi/bopl_rl.c" -ldl -lm
done

(cd "$project" && "$tmp/platform_ab" "$reference" "$rust") >"$tmp/platform.csv"
(cd "$project" && "$tmp/platform_phase_ab" "$reference" "$rust") >"$tmp/phase.csv"
(cd "$project" && "$tmp/platform_boundary_ab" "$reference" "$rust") >"$tmp/boundary.txt"
(cd "$project" && "$tmp/platform_rotation_ab" 0 65 333 700 \
    "net=$reference" "rust=$rust") >"$tmp/rotation.csv"

python3 - "$tmp/platform.csv" "$tmp/phase.csv" "$tmp/boundary.txt" "$tmp/rotation.csv" <<'PY'
import csv
import math
import re
import sys

platform_path, phase_path, boundary_path, rotation_path = sys.argv[1:]

def number(value):
    return float(value)

def assert_pair(row, name, tolerance):
    expected = number(row["net_" + name])
    actual = number(row["rust_" + name])
    if math.isnan(expected) and math.isnan(actual):
        return
    if abs(expected - actual) > tolerance:
        raise SystemExit(
            f"{name} differs by {abs(expected - actual)} at tick {row.get('tick')}"
        )

with open(platform_path, newline="") as handle:
    for row in csv.DictReader(handle):
        scenario = row["scenario"]
        if scenario == "overlap":
            for field in ("px", "py", "map_x", "map_y", "spawn_x", "spawn_y"):
                assert_pair(row, field, 0.05)
            for field in ("map_vx", "map_vy", "spawn_vx", "spawn_vy"):
                assert_pair(row, field, 0.01)
            assert_pair(row, "spawn_inverse_mass", 0.000002)
            assert_pair(row, "map_inverse_mass", 0.0)
        elif scenario == "landing":
            assert_pair(row, "px", 0.0001)
            assert_pair(row, "py", 0.0001)
        elif scenario == "orbiter":
            for field in (
                "px", "py", "map_x", "map_y", "map_vx", "map_vy",
                "map_inverse_mass",
            ):
                assert_pair(row, field, 0.000000001)
        else:
            raise SystemExit(f"unknown platform scenario {scenario}")

with open(phase_path, newline="") as handle:
    for row in csv.DictReader(handle):
        for field in (
            "player_flags", "slime_flags", "ability_flags", "circle_flags", "spawned",
        ):
            assert_pair(row, field, 0.0)
        for field in ("px", "py", "slime_x", "slime_y", "sx", "sy"):
            assert_pair(row, field, 0.002)
        assert_pair(row, "radius", 0.0)
        assert_pair(row, "width", 0.00002)
        assert_pair(row, "height", 0.0004)
        assert_pair(row, "inverse_mass", 0.000002)

boundary = open(boundary_path).read().strip()
match = re.fullmatch(
    r"net_rr=(\d+)/(\d+) net_death=(\d+) y=([-0-9.]+)/([-0-9.]+) "
    r"rust_rr=(\d+)/(\d+) rust_death=(\d+) y=([-0-9.]+)/([-0-9.]+)",
    boundary,
)
if not match:
    raise SystemExit(f"unrecognized boundary result: {boundary}")
values = match.groups()
if values[:5] != values[5:]:
    raise SystemExit(f"platform boundary differs: {boundary}")

with open(rotation_path, newline="") as handle:
    rows = list(csv.DictReader(handle))
by_tick = {}
for row in rows:
    by_tick.setdefault(row["tick"], {})[row["backend"]] = row
for tick, pair in by_tick.items():
    if set(pair) != {"net", "rust"}:
        raise SystemExit(f"missing rotation backend at tick {tick}")
    tolerances = {
        "orientation": 0.012,
        "angular_velocity": 0.005,
        "x": 0.002,
        "y": 0.002,
        "width": 0.00002,
        "height": 0.0004,
    }
    for field, tolerance in tolerances.items():
        difference = abs(number(pair["net"][field]) - number(pair["rust"][field]))
        if difference > tolerance:
            raise SystemExit(
                f"rotation {field} differs by {difference} at tick {tick}"
            )
PY

echo "PLATFORM_BROAD_GOLDEN_OK scenarios=6 ticks=4140"
