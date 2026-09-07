#!/usr/bin/env bash
set -euo pipefail
root=/home/mte/Documents/Coding/bopl-re/ripped/ExportedProject
report=$(python3 "$root/native/rl/autoresearch/watch.py" \
  --config "$root/native/artifacts/rl/autoresearch-monitor/config.json" --check)
bb thread tell thr_6bmdfjijez --mode queue --json \
  "Hourly Bopl research check. Read this report, inspect the Fable session and recent logs, verify forward progress and model/effort/full access, and address stalls or failures. Fable owns optimization; do not run competing benchmarks. Resume only if it stopped unintentionally, never after Matthew stopped it. Review evidence behind retained gains and summarize briefly with the graph link. Research workspace and session are in the monitor config; supervision instructions are in native/rl/autoresearch/WATCH.md in ExportedProject. Report: $report"
