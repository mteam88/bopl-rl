#!/bin/bash
# Play against a trained policy or a scripted bot in the native macOS build.
#
#   Tools/play-vs-agent-mac.sh <policy-*.bin | mixed-v2-hard | ...> [--seed N]
#   Tools/play-vs-agent-mac.sh            # default agent (native/artifacts/rl/human/default-agent, else mixed-v2-hard)
#
# The macOS counterpart of play-vs-agent.sh: same BOPL_AGENT / BOPL_AGENT_SEED /
# BOPL_AGENT_RESULTS contract with RlAgent.cs, same result archive
# (Tools/play_vs_agent_results.py -> native/artifacts/rl/human). Differences:
# the game is Builds/macOS/BoplBattle.app (just sync-mac + just build-mac on
# the PowerSpec), the shim is libbopl_agent.dylib (`make -C native/rl agent-mac`,
# built here with clang) placed in the bundle's Contents/Plugins, and the
# binary is exec'd directly so the environment reaches it (`open` would only
# focus an already running copy and drops the variables).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
APP=${BOPL_APP:-$ROOT/Builds/macOS/BoplBattle.app}
GAME=$APP/Contents/MacOS/BoplBattle
LIB=$ROOT/native/artifacts/rl/libbopl_agent.dylib
HUMAN=$ROOT/native/artifacts/rl/human
PYTHON=${PYTHON:-python3}
PLAYER_LOG="$HOME/Library/Logs/Johan Gronvall/BoplBattle/Player.log"

AGENT="" SEED=""
while [ $# -gt 0 ]; do
  case "$1" in
    --seed) SEED=$2; shift 2 ;;
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
esac
[ -x "$GAME" ] || { echo "no build at $GAME (just sync-mac && just build-mac on the PowerSpec)" >&2; exit 2; }
# `open` would only focus a running copy; a second exec'd instance would fight over the save dir.
if pgrep -x BoplBattle >/dev/null; then echo "BoplBattle is already running; quit it first" >&2; exit 3; fi

[ -f "$LIB" ] || make -C "$ROOT/native/rl" agent-mac >/dev/null || { echo "could not build libbopl_agent.dylib" >&2; exit 2; }
mkdir -p "$APP/Contents/Plugins"
cp -f "$LIB" "$APP/Contents/Plugins/libbopl_agent.dylib"
codesign -s - -f "$APP/Contents/Plugins/libbopl_agent.dylib" 2>/dev/null
export DYLD_LIBRARY_PATH="$APP/Contents/Plugins${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

mkdir -p "$HUMAN"
SESSION=$(date +%Y%m%d-%H%M%S)
RESULTS=$HUMAN/.session-$SESSION.jsonl
[ -n "$SEED" ] || SEED=$(( $(date +%s) % 1000000000 ))
export BOPL_AGENT=$AGENT BOPL_AGENT_SEED=$SEED BOPL_AGENT_RESULTS=$RESULTS
echo "agent $AGENT  seed $SEED  results $RESULTS"
echo "$AGENT" > "$HUMAN/last-agent"

"$GAME"
STATUS=$?
"$PYTHON" "$ROOT/Tools/play_vs_agent_results.py" "$RESULTS" --agent "$AGENT" --game "$GAME" --library "$LIB" --out "$HUMAN" --player-log "$PLAYER_LOG"
exit $STATUS
