#!/bin/bash
# Headless live session of this checkout's Linux build for the shadow test
# (docs/SHADOW.md). Everything runs in capped user units; the game's config
# dir (Player.log, prefs, saved replays) is isolated under $DIR/config.
#
#   session.sh start [pads] [extra env...]   Xvfb + N virtual pads + the game
#   session.sh play                          main menu -> character select
#   session.sh join <n...>                   pads join, ready, and start the round
#   session.sh pad <n> <command...>          send vpad commands to pad n
#   session.sh key <keys...>                 xdotool key to the game window
#   session.sh keydown|keyup <key>           hold/release a key
#   session.sh click <x> <y>                 mouse click in screen pixels
#   session.sh shot [file]                   screenshot
#   session.sh stop
# Then drive play with eval/shadow_live/human.py and grep BOPL_AGENT_SHADOW in
# $DIR/player.log.
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DIR=${SHADOW_LIVE_DIR:-/tmp/shadow-live}
DISP=${SHADOW_LIVE_DISPLAY:-:97}
SIZE=${SHADOW_LIVE_SIZE:-1280x720}
BUILD=$ROOT/Builds/Linux
mkdir -p "$DIR"
win() { DISPLAY=$DISP xdotool search --name "BoplBattle" 2>/dev/null | tail -1; }
case "${1:-}" in
  start)
    shift
    PADS=${1:-1}; shift || true
    "$0" stop >/dev/null 2>&1
    cc -O2 -o "$DIR/vpad" "$ROOT/eval/shadow_live/vpad.c" || exit 2
    P=$BUILD/BoplBattle_Data/Plugins/x86_64
    mkdir -p "$P"
    cp -f "$ROOT/native/artifacts/rl/libbopl_agent.so" "$ROOT/native/artifacts/rl/libbopl_sim_q32.so" "$P/"
    systemd-run --user --unit=shim-shadow-xvfb --collect -p MemoryHigh=2G Xvfb "$DISP" -screen 0 "${SIZE}x24" >/dev/null
    for n in $(seq 1 "$PADS"); do
      rm -f "$DIR/pad$n.fifo"; mkfifo "$DIR/pad$n.fifo"
      systemd-run --user --unit=shim-shadow-pad$n --collect sh -c "exec \"$DIR/vpad\" 'Microsoft X-Box 360 pad' < \"$DIR/pad$n.fifo\" > \"$DIR/pad$n.log\" 2>&1" >/dev/null
      systemd-run --user --unit=shim-shadow-padhold$n --collect sh -c "exec sleep infinity > \"$DIR/pad$n.fifo\"" >/dev/null
    done
    sleep 2
    mkdir -p "$DIR/config"
    ENVS=(--setenv=DISPLAY="$DISP" --setenv=XDG_CONFIG_HOME="$DIR/config" --setenv=LD_LIBRARY_PATH="$P"
          --setenv=__NV_PRIME_RENDER_OFFLOAD=1 --setenv=__GLX_VENDOR_LIBRARY_NAME=nvidia)
    for kv in "$@"; do ENVS+=(--setenv="$kv"); done
    systemd-run --user --unit=shim-shadow-game --collect -p MemoryHigh=8G -p WorkingDirectory="$BUILD" "${ENVS[@]}" \
      nice -n 5 "$BUILD/BoplBattle.x86_64" -screen-fullscreen 0 -screen-width "${SIZE%x*}" -screen-height "${SIZE#*x}" \
      -logFile "$DIR/player.log" >/dev/null
    for _ in $(seq 1 60); do [ -n "$(win)" ] && break; sleep 0.5; done
    W=$(win); [ -n "$W" ] && DISPLAY=$DISP xdotool windowfocus "$W" 2>/dev/null
    echo "window ${W:-none}"
    ;;
  play) # mouse clicks do not register under Xvfb, but hovering PLAY then Return does
    DISPLAY=$DISP xdotool mousemove 640 100; sleep 0.5; "$0" key Return ;;
  join) # A joins a pad; start readies it; start again (first pad) begins the round
    shift
    for n in "$@"; do "$0" pad "$n" "a 1"; sleep 0.3; "$0" pad "$n" "a 0"; sleep 1.5; done
    for n in "$@"; do "$0" pad "$n" "start 1"; sleep 0.3; "$0" pad "$n" "start 0"; sleep 1.2; done
    "$0" pad "$1" "start 1"; sleep 0.3; "$0" pad "$1" "start 0" ;;
  pad)
    n=$2; shift 2
    for c in "$@"; do echo "$c" > "$DIR/pad$n.fifo"; done
    ;;
  key) # held 0.25 s each: a bare `xdotool key` tap is often missed by the game
    shift; W=$(win); DISPLAY=$DISP xdotool windowfocus "$W" 2>/dev/null
    for k in "$@"; do DISPLAY=$DISP xdotool keydown "$k"; sleep 0.25; DISPLAY=$DISP xdotool keyup "$k"; sleep 0.1; done ;;
  keydown|keyup) W=$(win); DISPLAY=$DISP xdotool "$1" "$2" ;;
  click) DISPLAY=$DISP xdotool mousemove "$2" "$3" click 1 ;;
  shot) DISPLAY=$DISP import -window root "${2:-$DIR/shot.png}" 2>/dev/null; echo "${2:-$DIR/shot.png}" ;;
  stop)
    for u in shim-shadow-game shim-shadow-xvfb $(systemctl --user list-units --plain --no-legend 'shim-shadow-pad*' | awk '{print $1}'); do
      systemctl --user stop "$u" 2>/dev/null
    done
    echo stopped
    ;;
  *) sed -n 2,16p "$0" ;;
esac
