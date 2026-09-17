#!/bin/bash
# Interleaved A/B of OLD vs NEW libbopl_sim.so under the default loadout.
# Usage: run_ab.sh <ticks> <reps>
B=/home/mte/Documents/Coding/bopl-re/ripped/.rifts/ExportedProject/ability-port
BENCH=$B/native/target/perf/benchmark
NEW=$B/native/target/release/libbopl_sim.so
OLD=$B/native/target/oldsrc/target/release/libbopl_sim.so
LOG=$B/native/target/perf/ab-benchmark.log
TICKS=${1:-120000}
REPS=${2:-5}
CPU=${CPU:-11}
for players in 6 2; do
  for r in $(seq 1 $REPS); do
    # Alternate which library runs first so a changing shared-host load does
    # not systematically favor OLD or NEW across the whole sample.
    order="OLD NEW"
    [ $((r % 2)) -eq 0 ] && order="NEW OLD"
    for which in $order; do
      lib=$OLD; [ $which = NEW ] && lib=$NEW
      out=$(taskset -c $CPU $BENCH $lib /dev/null $LOG $TICKS $players 1)
      tps=$(printf '%s' "$out" | grep -o '"simulation_ticks_per_step_second":[0-9.]*' | cut -d: -f2)
      echo "$players $which $r $tps"
    done
  done
done
