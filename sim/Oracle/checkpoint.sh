#!/usr/bin/env bash
# Day-1 checkpoint scorecard for the .NET port.
#
#   sim/Oracle/checkpoint.sh            # markdown to stdout + sim/Oracle/CHECKPOINT.md
#   sim/Oracle/checkpoint.sh --full     # every committed fixture with a Unity
#                                       # trace present in /tmp/oracle, not just
#                                       # the l{0..4}-p2 level sweep
#
# Steps: build the solution, replay each l{0..4}-p2 fixture through sim/Host
# with tracing on, diff against the Unity oracle trace, and time an untraced
# reset run. Every step degrades to a "not built yet" / "missing" row instead
# of failing, so the scorecard is always complete.
#
# Host CLI contract (per coordinator brief to the shim thread):
#   dotnet run --project sim/Host -- replay <episode.rep> --ticks N
#   dotnet run --project sim/Host -- reset --level N --players 2 --ticks N
# If the Host's actual CLI differs when it lands, adapt the two run_host
# call sites below.
#
# Unity oracle traces are expected in /tmp/oracle/l{N}-p2/replayed.jsonl.gz
# (override with BOPL_ORACLE_TMP). A run under RL training: everything is
# nice -n 19.

set -u
FULL=no
[ "${1:-}" = "--full" ] && FULL=yes
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORACLE_TMP="${BOPL_ORACLE_TMP:-/tmp/oracle}"
FIXTURES="$ROOT/sim/Oracle/fixtures"
OUT="$ROOT/sim/Oracle/CHECKPOINT.md"
SLN="$ROOT/sim/BoplSim.sln"
HOST_PROJ="$ROOT/sim/Host/Host.csproj"
TIMING_TICKS=2400
WORK="$(mktemp -d /tmp/bopl-checkpoint.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

md="$WORK/scorecard.md"
say() { printf '%s\n' "$*" >> "$md"; }

say "# Day-1 checkpoint — $(date -u '+%Y-%m-%d %H:%M UTC')"
say ""
say "Commit: \`$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)\`"
say ""

# ---- 1. build ---------------------------------------------------------------

built=no
say "## Build"
say ""
if ! command -v dotnet >/dev/null 2>&1; then
    say "- **dotnet not installed** — all rows below skipped"
elif [ ! -f "$SLN" ]; then
    say "- **sim/BoplSim.sln not present** (shim rift not merged yet) — all rows below skipped"
else
    if nice -n 19 dotnet build -c Release "$SLN" > "$WORK/build.log" 2>&1; then
        built=yes
        warnings=$(grep -Ec "warning [A-Z]+[0-9]+" "$WORK/build.log" || true)
        say "- **pass** ($warnings warning lines)"
    else
        warnings=$(grep -Ec "warning [A-Z]+[0-9]+" "$WORK/build.log" || true)
        errors=$(grep -Ec "error [A-Z]+[0-9]+" "$WORK/build.log" || true)
        say "- **FAIL** ($errors error lines, $warnings warning lines). First errors:"
        say ""
        say '```'
        grep -E "error [A-Z]+[0-9]+" "$WORK/build.log" | head -5 >> "$md"
        say '```'
    fi
fi
say ""

# ---- 2. fixture replays vs the Unity oracle ---------------------------------

# run_host <log> <args...> — nice, no-build, 15 min cap. Returns dotnet's exit code.
run_host() {
    local log="$1"; shift
    nice -n 19 timeout 900 dotnet run -c Release --no-build --project "$ROOT/sim/Host" -- "$@" \
        > "$log" 2>&1
}

say "## Oracle replay (Unity replayed.jsonl.gz vs port trace, presentation + known export gaps excluded)"
say ""
say "| fixture | ticks | result | detail |"
say "|---|---|---|---|"

# Fixture set: the l{0..4}-p2 level sweep by default; with --full, every
# committed fixture directory (sim/Oracle/fixtures/*/episode.rep).
if [ "$FULL" = yes ]; then
    names=$(ls -d "$FIXTURES"/*/ 2>/dev/null | xargs -rn1 basename | sort)
else
    names="l0-p2 l1-p2 l2-p2 l3-p2 l4-p2"
fi

for name in $names; do
    fixture="$FIXTURES/$name"
    unity="$ORACLE_TMP/$name/replayed.jsonl.gz"
    port="$ORACLE_TMP/$name/dotnet.jsonl.gz"
    if [ ! -f "$fixture/episode.rep" ]; then
        say "| $name | - | skipped | fixture missing ($fixture) |"; continue
    fi
    ticks=$(python3 -c "import json;print(json.load(open('$fixture/report.json'))['requested_ticks'])" 2>/dev/null || echo 480)
    if [ ! -f "$unity" ]; then
        say "| $name | $ticks | skipped | Unity trace missing ($unity — rerun verify_replay) |"; continue
    fi
    if [ "$built" != yes ] || [ ! -f "$HOST_PROJ" ]; then
        say "| $name | $ticks | not built yet | sim/Host absent or build failed |"; continue
    fi
    rm -f "$port"
    # BOPL_TRACE_TICKS must exceed the run (default 600): when the capture's
    # tick limit fires mid-run it Finish()es and Application.Quit()s, but the
    # shim's Quit does not stop the Host loop, and the next capture call
    # re-creates (truncates) the finished trace file. Same envs verify_replay
    # uses.
    if ! BOPL_TRACE="$port" BOPL_TRACE_TICKS=100000 BOPL_TRACE_BATCH=16 \
        run_host "$WORK/replay-$name.log" replay "$fixture/episode.rep" --ticks "$ticks"; then
        say "| $name | $ticks | **replay crashed** | $(tail -1 "$WORK/replay-$name.log" | tr '|' '/' | cut -c1-120) |"
        continue
    fi
    if [ ! -f "$port" ]; then
        say "| $name | $ticks | **no trace written** | Host ran but produced no $port |"; continue
    fi
    # Known gaps, excluded so the scorecard shows real regressions instead of
    # a constant tick-0 diff:
    # - Player.playerDied/playerWon: subscribers come from the unexported menu
    #   flow (export gap; see sim/Oracle/README.md).
    # - Keyframe.m_InWeight/m_OutWeight: loader predates the v4 export's curve
    #   weights; drop these two excludes once the shim loader consumes them.
    python3 "$ROOT/sim/Oracle/diff.py" "$unity" "$port" --ignore-presentation --summary \
        --exclude Player.playerDied --exclude Player.playerWon \
        --exclude Keyframe.m_InWeight --exclude Keyframe.m_OutWeight \
        > "$WORK/diff-$name.log" 2>&1
    code=$?
    case $code in
        0) say "| $name | $ticks | **match** | $(grep -m1 '^match:' "$WORK/diff-$name.log") |";;
        1) matched=$(grep -m1 '^matched ticks:' "$WORK/diff-$name.log" | cut -d: -f2 | tr -d ' ')
           path=$(grep -m1 '^first difference at:' "$WORK/diff-$name.log" | cut -d' ' -f4- | tr '|' '/')
           sections=$(grep -m1 '^differing sections:' "$WORK/diff-$name.log" | cut -d' ' -f3- | tr '|' '/')
           say "| $name | $ticks | diverged at ${matched:-?}/$ticks | \`${path:-?}\` (${sections:-}) |";;
        2) say "| $name | $ticks | **header mismatch** | $(head -1 "$WORK/diff-$name.log" | tr '|' '/') |";;
        *) say "| $name | $ticks | diff error ($code) | $(tail -1 "$WORK/diff-$name.log" | tr '|' '/' | cut -c1-120) |";;
    esac
done
say ""

# ---- 3. untraced timing ------------------------------------------------------

say "## Throughput (untraced)"
say ""
if [ "$built" = yes ] && [ -f "$HOST_PROJ" ]; then
    start=$(date +%s.%N)
    if run_host "$WORK/timing.log" reset --level 0 --players 2 --ticks "$TIMING_TICKS"; then
        end=$(date +%s.%N)
        elapsed=$(python3 -c "print(f'{$end - $start:.2f}')")
        rate=$(python3 -c "print(f'{$TIMING_TICKS / ($end - $start):.0f}')")
        say "- Port: **$rate ticks/s** wall ($TIMING_TICKS ticks in ${elapsed}s, includes dotnet run + world load)"
        reported=$(grep -iE "ticks/s|ticks per second" "$WORK/timing.log" | head -1)
        [ -n "$reported" ] && say "- Port self-reported: $reported"
    else
        say "- Port timing run **failed**: $(tail -1 "$WORK/timing.log" | cut -c1-160)"
    fi
else
    say "- not built yet"
fi
say "- Unity baseline (sim/Oracle/README.md): ~5,000 ticks/s marginal per process"
say "  (0.2 ms/tick mono), 40-50 ticks/s with tracing on; boot ~9 s per process."
say ""

cp "$md" "$OUT"
cat "$md"
