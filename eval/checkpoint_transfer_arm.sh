#!/usr/bin/env bash
# Play one arm of the checkpoint-transfer grid: the same fixed checkpoint against the
# same frozen scripted pool, under ONE named library, for a chosen list of kits.
#
# checkpoint_transfer_grid.sh plays the two-library base/main A/B; this plays a single
# arm, which is what an extra backend needs. The C# oracle (sim/Native's
# BoplSimNative.so) is the motivating case: it exports the bopl_rl.h ABI but not
# bopl_rl_fork, so the two `search` bots (which decide by rollouts on a fork) abort on
# it and the arm must be scheduled with the five training-era bots only. Because
# evaluation_protocol.schedule derives each seed from (seed_base, map, game index) and
# never from the opponent's position in the list, that arm still pairs game-for-game
# with a 7-opponent arm on the games they share -- `checkpoint_transfer.py crosscheck`
# is the check that proves it on real data.
#
#   checkpoint_transfer_arm.sh OUT_DIR ARM_NAME LIB.so [GAMES] [HELDOUT] [PARALLEL] [SCRIPTED] [KITS...]
#
# Each cell is independent and resumable: evaluate.py skips a cell whose
# evaluations/<step>/result.json already exists, so re-running is safe and cheap.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$1; ARM=$2; LIB=$3; G=${4:-192}; H=${5:-288}; P=${6:-2}
SCRIPTED=${7:-mixed-hard,mixed-v2-hard,archer-v2-hard,pursuer-v2-hard,evasive-v2-hard}
shift 7 || shift $#
KITS=${*:-"dash-bow-platform roll-grenade-smoke teleport-mine-tesla web-mjolnir-freeze
           dash-missile-platform grapple-beam-rock duplicator-blink-blackhole drill-throw-magnet"}
PY=$(cd "$HERE/.." && pwd)/native/artifacts/rl/venv/bin/python
mkdir -p "$OUT/logs"
printf '%s\n' $KITS |
  xargs -P "$P" -n 1 bash -c '
    nice -n 19 "'"$PY"'" "'"$HERE"'/checkpoint_transfer.py" run --out "'"$OUT"'" \
      --libs "'"$ARM"'='"$LIB"'" --kits "$0" --games '"$G"' --heldout-games '"$H"' \
      --scripted '"${SCRIPTED//,/ }"' --workers 2 \
      >> "'"$OUT"'/logs/grid-$0-'"$ARM"'.log" 2>&1 && echo "cell done $0 '"$ARM"'"
  '
