#!/usr/bin/env bash
# Trainer-scale Rust vs .NET A/B at the run-007
# training shape (set encoder, hidden 128, 4 banks, horizon 128, buffers 2),
# interleaved over two rounds so drift in background load hits both sides.
# Rust hosts every world in one runtime and needs no spare (train.py sets that
# from BOPL_RL_BACKEND=rust); .NET runs at its train.py default of four worlds
# per runtime. Waits for the live training unit
# (and any other bopl-*bench* unit) to go inactive first.
#
# Run as its own unit, never inside the bb service cgroup:
#   systemd-run --user --unit bopl-rust-headline-bench -p MemoryMax=16G \
#     -p WorkingDirectory=$PWD /usr/bin/bash $PWD/train/bench_rust_headline.sh
# Results: native/artifacts/rl/scale-quiet-{rust,dotnet}/summary.txt
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
cd "$R"
UPDATES=${UPDATES:-300}
ROUNDS=${ROUNDS:-2}
WAIT_FOR=${WAIT_FOR:-bopl-long-run-1b-007}
ME=${ME:-bopl-rust-headline-bench}
BENCH=$R/train/bench_scale.sh
[ -f native/target/release/libbopl_sim.so ] || { echo "libbopl_sim.so missing: cargo build --release in native/" >&2; exit 1; }
[ -f sim/Native/bin/Release/net8.0/linux-x64/publish/BoplSimNative.so ] || { echo "BoplSimNative.so missing" >&2; exit 1; }
busy() {
  [ -n "$WAIT_FOR" ] && systemctl --user is-active --quiet "$WAIT_FOR" && return 0
  systemctl --user list-units --plain --no-legend 'bopl-*bench*' 2>/dev/null | awk '{print $1}' | grep -v "^$ME" | grep -q . && return 0
  return 1
}
echo "$(date -u '+%F %T UTC') waiting for $WAIT_FOR and other bopl-*bench* units"
while busy; do sleep 30; done
sleep 60  # let the trainer's worlds exit and the page cache settle
echo "$(date -u '+%F %T UTC') box free: load $(cut -d' ' -f1-3 /proc/loadavg); free -g | sed -n 2p"
for round in $(seq 1 "$ROUNDS"); do
  echo "$(date -u '+%F %T UTC') round $round rust"
  OUT=$R/native/artifacts/rl/scale-quiet-rust BOPL_RL_BACKEND=rust bash "$BENCH" "$UPDATES" -- \
    "16 16 active so" "32 32 active so" "64 16 passive so" "128 16 passive so" "256 16 passive so"
  echo "$(date -u '+%F %T UTC') round $round dotnet"
  OUT=$R/native/artifacts/rl/scale-quiet-dotnet bash "$BENCH" "$UPDATES" -- \
    "16 16 active so 4" "32 32 active so 4" "64 16 passive so 4"
done
echo "$(date -u '+%F %T UTC') done: load $(cut -d' ' -f1-3 /proc/loadavg)"
