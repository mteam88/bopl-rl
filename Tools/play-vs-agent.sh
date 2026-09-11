#!/bin/bash
# Play the real Unity game against a trained policy or a scripted bot
# (native/rl/PLAY_UNITY.md).
#
#   Tools/play-vs-agent.sh <policy-*.bin | mixed-v2-hard | ...> [--seed N] [--headless [:DISPLAY]]
#   Tools/play-vs-agent.sh                      # default agent (see below)
#   Tools/play-vs-agent.sh --install-desktop    # desktop entry next to bopl-battle.desktop
#
# Launches this checkout's Linux build (Builds/Linux) the Sunshine/Moonlight-safe
# way (play.sh / launch-desktop.sh: -force-glcore, compositor-managed maximized
# window, never exclusive fullscreen) with BOPL_AGENT set, so the character
# select adds the AI player (RlAgent.cs) — the human joins with the gamepad as
# usual and presses start. When the game exits, every round's auto-saved replay
# is copied to native/artifacts/rl/human/ and a result row is appended to
# native/artifacts/rl/human/games.jsonl (Tools/play_vs_agent_results.py).
#
# Default agent when none is given: $BOPL_AGENT_DEFAULT, else the first line of
# native/artifacts/rl/human/default-agent, else mixed-v2-hard.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD=${BOPL_BUILD:-$ROOT/Builds/Linux}
GAME=$BUILD/BoplBattle.x86_64
LIB=$ROOT/native/artifacts/rl/libbopl_agent.so
HUMAN=$ROOT/native/artifacts/rl/human
PYTHON=$ROOT/native/artifacts/rl/venv/bin/python
[ -x "$PYTHON" ] || PYTHON=python3
REPLAYS="$HOME/.config/unity3d/Johan Gronvall/BoplBattle/replays"
PLAYER_LOG="$HOME/.config/unity3d/Johan Gronvall/BoplBattle/Player.log"

AGENT=""; SEED=""; HEADLESS=""; INSTALL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --seed) SEED=$2; shift 2 ;;
    --headless) HEADLESS=":94"; if [ $# -gt 1 ] && [[ "$2" == :* ]]; then HEADLESS=$2; shift; fi; shift ;;
    --install-desktop) INSTALL=1; shift ;;
    -h|--help) sed -n 2,20p "$0"; exit 0 ;;
    *) AGENT=$1; shift ;;
  esac
done

if [ $INSTALL = 1 ]; then
  mkdir -p "$HOME/.local/share/applications"
  cat > "$HOME/.local/share/applications/bopl-battle-vs-agent.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Bopl Battle vs AI (play-unity build)
Comment=Local game with one RL policy / scripted bot player; results in native/artifacts/rl/human
Exec=$ROOT/Tools/play-vs-agent.sh
Icon=applications-games
Terminal=false
Categories=Game;
Keywords=game;bopl;ai;rl;
EOF
  echo "installed $HOME/.local/share/applications/bopl-battle-vs-agent.desktop"
  exit 0
fi

if [ -z "$AGENT" ]; then
  AGENT=${BOPL_AGENT_DEFAULT:-}
  [ -z "$AGENT" ] && [ -f "$HUMAN/default-agent" ] && AGENT=$(head -1 "$HUMAN/default-agent")
  [ -z "$AGENT" ] && AGENT=mixed-v2-hard
fi
case "$AGENT" in
  *.bin) AGENT=$(readlink -f "$AGENT"); [ -f "$AGENT" ] && [ -f "$AGENT.json" ] || { echo "checkpoint or its .json sidecar missing: $AGENT" >&2; exit 2; } ;;
esac
[ -x "$GAME" ] || { echo "no build at $GAME (build the play-unity project first)" >&2; exit 2; }
if pgrep -f 'BoplBattle\.x86_64' >/dev/null; then echo "BoplBattle is already running; not launching a second instance" >&2; exit 3; fi

# The agent shim: build if missing, then place it where Mono's DllImport("bopl_agent") looks.
[ -f "$LIB" ] || make -C "$ROOT/native/rl" agent >/dev/null || { echo "could not build libbopl_agent.so" >&2; exit 2; }
mkdir -p "$BUILD/BoplBattle_Data/Plugins/x86_64"
cp -f "$LIB" "$BUILD/BoplBattle_Data/Plugins/x86_64/libbopl_agent.so"
export LD_LIBRARY_PATH="$BUILD/BoplBattle_Data/Plugins/x86_64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$HUMAN"
SESSION=$(date +%Y%m%d-%H%M%S)
RESULTS=$HUMAN/.session-$SESSION.jsonl
[ -n "$SEED" ] || SEED=$(( $(date +%s) % 1000000000 ))
export BOPL_AGENT=$AGENT BOPL_AGENT_SEED=$SEED BOPL_AGENT_RESULTS=$RESULTS
echo "agent $AGENT  seed $SEED  results $RESULTS"
echo "$AGENT" > "$HUMAN/last-agent"

if [ -n "$HEADLESS" ]; then
  # Xvfb (tests): the game only takes input after xdotool windowfocus.
  pkill -f "Xvfb $HEADLESS" 2>/dev/null; sleep 0.5
  Xvfb "$HEADLESS" -screen 0 1280x720x24 >/dev/null 2>&1 &
  XVFB=$!
  sleep 2
  export DISPLAY=$HEADLESS
  ( cd "$BUILD" && __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./BoplBattle.x86_64 -screen-fullscreen 0 -screen-width 1280 -screen-height 720 ) &
  GPID=$!
else
  export DISPLAY=:0
  export XAUTHORITY=$(ls -t /run/user/1000/.mutter-Xwaylandauth.* 2>/dev/null | head -1)
  export XDG_RUNTIME_DIR=/run/user/1000
  ( cd "$BUILD" && ./BoplBattle.x86_64 -force-glcore -screen-fullscreen 0 -screen-width 2560 -screen-height 1440 ) &
  GPID=$!
  WID=""
  for _ in $(seq 1 30); do
    WID=$(xdotool search --name "BoplBattle" 2>/dev/null | tail -1)
    [ -n "$WID" ] && break
    sleep 0.5
  done
  if [ -n "$WID" ]; then
    wmctrl -i -r "$WID" -b remove,fullscreen 2>/dev/null
    wmctrl -i -r "$WID" -b add,maximized_vert,maximized_horz 2>/dev/null
    xdotool windowactivate "$WID" 2>/dev/null
  fi
fi
wait "$GPID"
STATUS=$?
[ -n "$HEADLESS" ] && { kill "${XVFB:-0}" 2>/dev/null; pkill -f "Xvfb $HEADLESS" 2>/dev/null; }
"$PYTHON" "$ROOT/Tools/play_vs_agent_results.py" "$RESULTS" --agent "$AGENT" --game "$GAME" --library "$LIB" --out "$HUMAN" --player-log "$PLAYER_LOG"
exit $STATUS
