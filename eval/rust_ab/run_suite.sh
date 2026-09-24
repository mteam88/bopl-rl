#!/bin/bash
# Build and run every rust_ab differential against both backends, then apply
# the trace comparator so that the trace-only probes -- which always exit 0 --
# can actually fail. See README.md "Trace-only probes".
#
#   eval/rust_ab/run_suite.sh [name-filter]
#
# Environment:
#   REFERENCE  .NET oracle .so   (default: the publish/ tree)
#   RUST       Rust backend .so  (default: target-q32, the bit-exact build)
#   OUTDIR     where captures land (default: native/target/rust_ab_suite)
#
# Exits non-zero if any probe fails, which is the whole point: a suite that
# only checks exit status reports the trace-only probes as passing forever.
set -u
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../.." && pwd)
REFERENCE=${REFERENCE:-$root/sim/Native/bin/Release/net8.0/linux-x64/publish/BoplSimNative.so}
RUST=${RUST:-$root/native/target-q32/release/libbopl_sim.so}
OUTDIR=${OUTDIR:-$root/native/target/rust_ab_suite}
PYTHON=${PYTHON:-python3}
compare="$here/trace_compare.py"
bin="$OUTDIR/bin"
mkdir -p "$bin" || exit 1

for library in "$REFERENCE" "$RUST"; do
  [ -f "$library" ] || { echo "missing backend: $library" >&2; exit 2; }
done
# A stale .so silently invalidates the whole run. Say what is being measured.
echo "reference: $REFERENCE ($(date -r "$REFERENCE" '+%Y-%m-%d %H:%M'))"
echo "rust:      $RUST ($(date -r "$RUST" '+%Y-%m-%d %H:%M'))"

# Six pickup probes only exercise their subject with ability spawns enabled.
export BOPL_RL_ABILITY_SPAWNS=1

# Probes that print a per-backend trace and always return 0. The comparator is
# their only failure path, so it runs in strict mode: if one stops emitting
# comparable rows, that is an error rather than a silent pass. Derived
# mechanically -- any probe whose output trace_compare.py can pair belongs
# here. Add to it when you add a trace-only probe.
trace_only="
  ability_push_ab arrow_bounds_ab bow_ab bow_ground_exit_ab bow_phase_ab
  dash_phase_ab entity_velocity_ab keyboard_ab move_ab platform_ab
  platform_boundary_ab platform_phase_ab platform_rotation_ab recast_ab
  state_ab stock_respawn_ab sudden_death_ab
"

# Not two-backend differentials: single-backend tools, or a Rust-vs-Rust
# representation check that takes two Rust builds rather than the oracle.
# `replay_ab` is a differential, but it takes replay-specific arguments
# rather than two library paths; drive it with `make replay-ab`.
# `bounds_case_trace` does take two library paths, but it is a diagnostic that
# reports a first divergence and exits 0 whether or not it found one, so
# counting it as a pass would inflate the suite with a probe that cannot fail.
not_differential=" aim_map bounds_case_trace bow_test platform_rotation_trace repr_ab replay_ab "

pass=0; fail=0; skipped=0
failures=()
filter="${1:-}"

run_probe() {
  local name=$1 capture=$2
  case " $not_differential " in *" $name "*) return 99;; esac
  if [ "$name" = platform_rotation_ab ]; then
    # Spawned-platform dynamics are phase-sensitive: sweep maps and aim angles
    # rather than trusting one combination.
    : > "$capture"; local rc=0 m a
    for m in 0 1 2 3 4; do for a in 0 32 64 100 160 200; do
      ( cd "$root" && "$bin/$name" "$m" "$a" 400 900 net="$REFERENCE" rust="$RUST" ) \
        >> "$capture" 2>&1 || rc=1
    done; done
    return $rc
  fi
  ( cd "$root" && "$bin/$name" "$REFERENCE" "$RUST" ) > "$capture" 2>&1
}

cd "$root" || exit 1
for source in eval/rust_ab/*.c eval/rust_ab/*.sh; do
  name=$(basename "$source"); name=${name%.*}
  [ "$name" = run_suite ] && continue
  if [ -n "$filter" ] && [[ "$name" != *"$filter"* ]]; then continue; fi
  case " $not_differential " in
    *" $name "*) echo "SKIP $name (not a two-backend differential)"; skipped=$((skipped+1)); continue;;
  esac
  capture="$OUTDIR/$name.out"

  if [[ "$source" == *.sh ]]; then
    ( cd "$root" && REFERENCE="$REFERENCE" RUST_ROLLOUT="$RUST" \
      bash "$source" "$REFERENCE" "$RUST" ) > "$capture" 2>&1
    rc=$?
  else
    if [ ! -x "$bin/$name" ] || [ "$source" -nt "$bin/$name" ] || [ sim/abi/bopl_rl.c -nt "$bin/$name" ]; then
      if ! cc -O2 -I sim/abi -o "$bin/$name" "$source" sim/abi/bopl_rl.c -ldl -lm > "$OUTDIR/$name.build" 2>&1; then
        echo "BUILD $name (see $OUTDIR/$name.build)"; fail=$((fail+1)); failures+=("$name (build)"); continue
      fi
    fi
    run_probe "$name" "$capture"; rc=$?
  fi

  if [ $rc -ne 0 ]; then
    echo "FAIL  $name (exit $rc)"; fail=$((fail+1)); failures+=("$name (exit $rc)"); continue
  fi

  # Exit status was clean. That is not yet an answer for a trace-only probe.
  case "$trace_only" in
    *" $name "*) verdict=$("$PYTHON" "$compare" --quiet "$capture" 2>&1); crc=$?;;
    *) verdict=$("$PYTHON" "$compare" --quiet --self-asserting "$capture" 2>&1); crc=$?;;
  esac
  if [ $crc -ne 0 ]; then
    echo "TRACE $name"; printf '%s\n' "$verdict" | sed 's/^/      /'
    fail=$((fail+1)); failures+=("$name (trace)"); continue
  fi
  pass=$((pass+1))
done

echo
echo "pass=$pass fail=$fail skipped=$skipped"
if [ ${#failures[@]} -gt 0 ]; then
  printf 'failed: %s\n' "${failures[@]}"
  exit 1
fi
