#!/bin/bash
# Play against a trained policy or a scripted bot in the native macOS build.
#
#   Tools/play-vs-agent-mac.sh <policy-*.bin | mixed-v2-hard | ... | <name>.roster> [--seed N] [--maps training|held-out|0,1,2]
#   Tools/play-vs-agent-mac.sh            # default agent (native/artifacts/rl/human/default-agent, else mixed-v2-hard)
#
# A .roster (Tools/rating_roster.py) draws a different member per round; the
# shim logs the draws and play_vs_agent_results.py attributes each round.
#
# The macOS counterpart of play-vs-agent.sh: same BOPL_AGENT / BOPL_AGENT_SEED /
# BOPL_AGENT_RESULTS contract with RlAgent.cs, same result archive
# (Tools/play_vs_agent_results.py -> native/artifacts/rl/human). Differences:
# the game is Builds/macOS/BoplBattle.app (just sync-mac + just build-mac on
# the PowerSpec), the shim is libbopl_agent.dylib (`make agent-mac`,
# built here with clang) placed in the bundle's Contents/Plugins, and the
# game is launched with `open -n --env` so it runs in the GUI session (controller
# input) with the environment intact.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
APP=${BOPL_APP:-$ROOT/Builds/macOS/BoplBattle.app}
GAME=$APP/Contents/MacOS/BoplBattle
LIB=$ROOT/native/artifacts/rl/libbopl_agent.dylib
HUMAN=$ROOT/native/artifacts/rl/human
PYTHON=${PYTHON:-python3}
PLAYER_LOG="$HOME/Library/Logs/Johan Gronvall/BoplBattle/Player.log"

AGENT="" SEED="" MAPS=${BOPL_LEVEL_SET:-}
while [ $# -gt 0 ]; do
  case "$1" in
    --seed) SEED=$2; shift 2 ;;
    --maps) MAPS=$2; shift 2 ;;   # RL map indices the rotation may draw (0,1,2,3,4), or training | held-out | all (native/data/map-set.json)
    -h|--help) sed -n 2,15p "$0"; exit 0 ;;
    *) AGENT=$1; shift ;;
  esac
done
if [ -z "$AGENT" ]; then
  AGENT=${BOPL_AGENT_DEFAULT:-}
  [ -z "$AGENT" ] && [ -f "$HUMAN/default-agent" ] && AGENT=$(head -1 "$HUMAN/default-agent")
  [ -z "$AGENT" ] && AGENT=mixed-v2-hard
fi
case "$AGENT" in
  *.bin) AGENT=$(cd "$(dirname "$AGENT")" && pwd)/$(basename "$AGENT")
         [ -f "$AGENT" ] && [ -f "$AGENT.json" ] || { echo "checkpoint or its .json sidecar missing: $AGENT" >&2; exit 2; } ;;
  *.roster) AGENT=$(cd "$(dirname "$AGENT")" && pwd)/$(basename "$AGENT")
         [ -f "$AGENT" ] || { echo "roster missing: $AGENT" >&2; exit 2; } ;;
esac
[ -x "$GAME" ] || { echo "no build at $GAME (just sync-mac && just build-mac on the PowerSpec)" >&2; exit 2; }
# `open` would only focus a running copy; a second exec'd instance would fight over the save dir.
if pgrep -x BoplBattle >/dev/null; then echo "BoplBattle is already running; quit it first" >&2; exit 3; fi

[ -f "$LIB" ] || make -C "$ROOT" agent-mac >/dev/null || { echo "could not build libbopl_agent.dylib" >&2; exit 2; }
mkdir -p "$APP/Contents/Plugins"
cp -f "$LIB" "$APP/Contents/Plugins/libbopl_agent.dylib"
codesign -s - -f "$APP/Contents/Plugins/libbopl_agent.dylib" 2>/dev/null
export DYLD_LIBRARY_PATH="$APP/Contents/Plugins${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

mkdir -p "$HUMAN"
SESSION=$(date +%Y%m%d-%H%M%S)
RESULTS=$HUMAN/.session-$SESSION.jsonl
[ -n "$SEED" ] || SEED=$(( $(date +%s) % 1000000000 ))
ROSTER_LOG=$HUMAN/.session-$SESSION.roster.jsonl
export BOPL_AGENT=$AGENT BOPL_AGENT_SEED=$SEED BOPL_AGENT_RESULTS=$RESULTS BOPL_AGENT_ROSTER_LOG=$ROSTER_LOG
# The game's BOPL_LEVEL_SET takes Unity levels; --maps takes canonical RL map
# ids, a named map set, or a campaign role. map_set.py validates/translates the
# selection; the identity-124 catalogue maps every RL id 0-123 to the same level.
if [ -n "$MAPS" ]; then
  LEVELS=$("$PYTHON" "$ROOT/eval/map_set.py" levels "$MAPS") || { echo "bad --maps $MAPS" >&2; exit 2; }
  export BOPL_LEVEL_SET=$LEVELS
fi
echo "agent $AGENT  seed $SEED  maps ${MAPS:-all}${LEVELS:+ (levels $LEVELS)}  results $RESULTS"
echo "$AGENT" > "$HUMAN/last-agent"

# Launch through `open` so the process joins the user's GUI session: a game
# exec'd from an ssh shell lives in the ssh audit session, where the Input
# System sees no HID controllers. `open --env` (macOS 14+) carries the
# variables; Mono finds the shim in Contents/Plugins without DYLD_LIBRARY_PATH.
ENVARGS=(--env "BOPL_AGENT=$BOPL_AGENT" --env "BOPL_AGENT_SEED=$BOPL_AGENT_SEED"
         --env "BOPL_AGENT_RESULTS=$BOPL_AGENT_RESULTS" --env "BOPL_AGENT_ROSTER_LOG=$BOPL_AGENT_ROSTER_LOG")
[ -n "${BOPL_LEVEL_SET:-}" ] && ENVARGS+=(--env "BOPL_LEVEL_SET=$BOPL_LEVEL_SET")
open -W -n -a "$APP" "${ENVARGS[@]}"
STATUS=$?
"$PYTHON" "$ROOT/Tools/play_vs_agent_results.py" "$RESULTS" --agent "$AGENT" --game "$GAME" --library "$LIB" --out "$HUMAN" --player-log "$PLAYER_LOG" --roster-log "$ROSTER_LOG"
exit $STATUS
