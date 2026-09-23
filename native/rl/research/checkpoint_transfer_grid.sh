#!/usr/bin/env bash
# Play the 8 kits x 2 libraries A/B grid (checkpoint_transfer.py run), six cells at a
# time on two workers each. Each cell is independent and resumable: evaluate.py skips a
# cell whose evaluations/<step>/result.json already exists.
#   checkpoint_transfer_grid.sh OUT_DIR BASE_SO MAIN_SO [GAMES] [HELDOUT_GAMES] [PARALLEL]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$1; BASE=$2; MAIN=$3; G=${4:-192}; H=${5:-288}; P=${6:-6}
PY=$(cd "$HERE/../../.." && pwd)/native/artifacts/rl/venv/bin/python
KITS="dash-bow-platform roll-grenade-smoke teleport-mine-tesla web-mjolnir-freeze
      dash-missile-platform grapple-beam-rock duplicator-blink-blackhole drill-throw-magnet"
mkdir -p "$OUT/logs"
for kit in $KITS; do for arm in base main; do echo "$kit $arm"; done; done |
  xargs -P "$P" -n 2 bash -c '
    so=$([ "$1" = base ] && echo "'"$BASE"'" || echo "'"$MAIN"'")
    nice -n 19 "'"$PY"'" "'"$HERE"'/checkpoint_transfer.py" run --out "'"$OUT"'" \
      --libs "$1=$so" --kits "$0" --games '"$G"' --heldout-games '"$H"' --workers 2 \
      >> "'"$OUT"'/logs/grid-$0-$1.log" 2>&1 && echo "cell done $0 $1"
  '
