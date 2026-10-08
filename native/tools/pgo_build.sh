#!/usr/bin/env bash
# The release library, profile-guided: build the sim instrumented, re-step
# the trained-policy games in native/pgo/corpus.tar.xz through it (pgo-train,
# the trainer's step_sampled decisions), merge the profile and build
# native/target/release with it. Exact: PGO moves code, not arithmetic (the
# env_bench state hash of the PGO and plain builds matches). 22% fewer
# sim microseconds per decision on held-out games (2026-10-06,
# native/pgo/README.md). Needs llvm-profdata: rustup component add llvm-tools.
#
#   native/tools/pgo_build.sh [extra cargo args]
set -euo pipefail
cd "$(dirname "$0")/.."
PROFDATA=$(dirname "$(rustc --print target-libdir)")/bin/llvm-profdata
[[ -x $PROFDATA ]] || { echo "pgo_build: no $PROFDATA; run: rustup component add llvm-tools" >&2; exit 1; }
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/corpus" "$WORK/raw"
tar xJf pgo/corpus.tar.xz -C "$WORK/corpus"
# RUSTFLAGS replaces .cargo/config.toml's rustflags, so it repeats target-cpu.
CARGO_TARGET_DIR=target/pgo-gen RUSTFLAGS="-C target-cpu=native -Cprofile-generate=$WORK/raw" \
  cargo build --release --bin pgo-train "$@"
LLVM_PROFILE_FILE="$WORK/raw/pgo-%p-%m.profraw" target/pgo-gen/release/pgo-train "$WORK/corpus"
"$PROFDATA" merge -o "$WORK/merged.profdata" "$WORK"/raw/pgo-*.profraw
# The profile-use build places blocks by LLVM's ext-TSP model (fall-throughs
# and short jumps weighted by the profile's edge counts) instead of the
# default chain placement: -0.27% sim instructions, -0.3% cycles. The sim's
# code misses are capacity (a tick touches ~84 KB of code lines, all from
# L2), so layout buys little; machine function splitting and hot/cold
# splitting were neutral to +0.4%.
RUSTFLAGS="-C target-cpu=native -Cprofile-use=$WORK/merged.profdata -C llvm-args=-enable-ext-tsp-block-placement" \
  cargo build --release "$@"
