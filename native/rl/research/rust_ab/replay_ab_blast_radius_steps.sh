#!/usr/bin/env bash
# Cumulative policy-visible divergence of each library on the command line against
# a fixed baseline, one column per kit. Used to attribute the blast radius of a
# merge sequence to the individual merges. Drive it with libraries built from the
# commits of interest; see replay_ab_blast_radius.py for the per-game detail.
RIFT=/home/mte/Documents/Coding/bopl-re/ripped/.rifts/ExportedProject
N=$RIFT/blast-radius/native
BIN=$N/artifacts/rl/replay_ab
BASE=/tmp/blast/libs/baseline-82cccb98.so
cd $RIFT/blast-radius
printf "%-12s" "library"
for k in dashbow rollgren telemine webmjol dashmiss grapple duplic drill; do printf " %8s" $k; done; printf " %8s\n" "ALL%"
for so in "$@"; do
  printf "%-12s" "$(basename $so .so | sed 's/step-//;s/baseline-//;s/main-//')"
  tot=0; dif=0
  for k in dash-bow-platform roll-grenade-smoke teleport-mine-tesla web-mjolnir-freeze dash-missile-platform grapple-beam-rock duplicator-blink-blackhole drill-throw-magnet; do
    S=$(REPLAY_AB_VISIBLE=1 REPLAY_AB_TOL=1 REPLAY_AB_QUIET=1 $BIN $BASE $so $N/artifacts/rl/sweeps/$k-long/000/evaluations 2>/dev/null | grep '^games ')
    g=$(echo "$S" | sed 's/games \([0-9]*\).*/\1/'); c=$(echo "$S" | sed 's/.*clean \([0-9]*\),.*/\1/')
    d=$((g-c)); tot=$((tot+g)); dif=$((dif+d))
    printf " %7.1f%%" $(python3 -c "print(100*$d/$g)")
  done
  printf " %7.1f%%\n" $(python3 -c "print(100*$dif/$tot)")
done
