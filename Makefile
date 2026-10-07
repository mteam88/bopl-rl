CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Werror
OUT ?= native/artifacts/rl
# Probes under eval/rust_ab resolve their fixture and log through
# /proc/self/cwd/native/..., so they have to run from the project root. This
# Makefile lives at the root, so make's working directory already is that -
# which is why no recipe below changes directory. It was not always so: the
# Makefile used to sit in native/rl and the standalone targets silently failed
# with a bare `open:` and an empty message until someone added a cd.
ABI := sim/abi
INCLUDES := -I$(ABI) -Itrain

all: $(OUT)/libbopl_rl.so $(OUT)/benchmark $(OUT)/coverage $(OUT)/record $(OUT)/reset_check $(OUT)/sampled_check $(OUT)/reset_profile $(OUT)/frame_profile

# Standalone evaluation/extraction binding; does not require rebuilding CUDA.
research: $(OUT)/libbopl_research.so

map-ab: $(OUT)/map_ab

# `dotnet publish` refreshes publish/; the sibling native/ directory keeps whatever
# an earlier build left there, so pointing at it silently compares against a stale
# oracle - it held an ABI 2 library for nine days after ABI 3 landed.
REFERENCE ?= sim/Native/bin/Release/net8.0/linux-x64/publish/BoplSimNative.so
RUST_ROLLOUT ?= native/target/release/libbopl_sim.so
# The default `cargo build --release` stores `Fix` as an f64 (native/src/fixed.rs)
# and is what training uses. It is NOT bit-comparable with the .NET oracle: its
# multiplications and divisions round to the nearest double instead of
# truncating to 2^-32, so every differential probe below sees an arithmetic
# delta of a few raw units at tick 0 and reports that instead of its own
# subject. Every A/B gate must therefore run the `q32` library, which is the
# reference's exact Q32.32 arithmetic. It is built into its own target
# directory so it does not fight the f64 build for `native/target`.
RUST_ROLLOUT_Q32 ?= native/target-q32/release/libbopl_sim.so

rust-q32:
	cargo build --release --features q32 --manifest-path native/Cargo.toml --target-dir native/target-q32

# Exhaustive authored-map gate: all 124 maps, both 2- and 6-player layouts,
# 3000 ticks per live case, with discrete player/slime state required exact.
map-sweep-3000: $(OUT)/map_ab rust-q32
	MAP_AB_STRICT_ACTORS=1 $(abspath $(OUT)/map_ab) $(abspath $(REFERENCE)) $(abspath $(RUST_ROLLOUT_Q32)) 3000

$(OUT)/map_ab: eval/rust_ab/map_ab.c $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) $(INCLUDES) eval/rust_ab/map_ab.c $(ABI)/bopl_rl.c -ldl -lm -o $@

# DestroyIfOutsideSceneBounds gate: walks a player off the edge of map 0 and
# engages each ability form at three depths relative to the kill plane, then
# requires the reference and the port to agree on the death tick, the death
# position, and the cause of death. Catches ability bodies that hover below
# the plane (GunTransform's slow-mo family) and abilities entered on the tick
# the slime is already out of bounds.
bounds-ability-ab: $(OUT)/bounds_ability_ab rust-q32
	$(abspath $(OUT)/bounds_ability_ab) $(abspath $(REFERENCE)) $(abspath $(RUST_ROLLOUT_Q32))

$(OUT)/bounds_ability_ab: eval/rust_ab/bounds_ability_ab.c $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) $(INCLUDES) eval/rust_ab/bounds_ability_ab.c $(ABI)/bopl_rl.c -ldl -lm -o $@

# Diagnostic for a single bounds-ability-ab case: the same script, stepped in
# lockstep, reporting the first tick at which any exported field differs rather
# than only the boundary kill. Not a gate - it takes its case from the
# environment (BCT_ABILITIES, BCT_SLOT, BCT_AIM, BCT_ENGAGE_Y, BCT_DUMP) and is
# run by hand when bounds-ability-ab reports a kill-tick gap too large to be
# rounding. See the header of eval/rust_ab/bounds_case_trace.c.
bounds-case-trace: $(OUT)/bounds_case_trace rust-q32
	$(abspath $(OUT)/bounds_case_trace) $(abspath $(REFERENCE)) $(abspath $(RUST_ROLLOUT_Q32))

$(OUT)/bounds_case_trace: eval/rust_ab/bounds_case_trace.c eval/divergence/divergence_compare.h $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -Wno-unused-function $(INCLUDES) -Ieval/divergence eval/rust_ab/bounds_case_trace.c $(ABI)/bopl_rl.c -ldl -lm -o $@

# Collider-rescale vs entry-pass boundary gate: a grounded player pressing a
# rescaling ability slot (Dash) while sudden death has carried it under the
# kill plane. The entry-pass check runs Move on a COPY of the body, and the
# copy does not re-sync the collider, so clone and real Move can disagree.
# Currently FAILS on map 2: the reference lets Dash claim the body and kills
# it as an ability body (cause Other, ungrounded 0.05 off the surface) while
# the port pre-empts and kills the ordinary slime (cause Accident).
scale-boundary-ab: $(OUT)/scale_boundary_ab rust-q32
	$(abspath $(OUT)/scale_boundary_ab) $(abspath $(REFERENCE)) $(abspath $(RUST_ROLLOUT_Q32))

$(OUT)/scale_boundary_ab: eval/rust_ab/scale_boundary_ab.c $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) $(INCLUDES) eval/rust_ab/scale_boundary_ab.c $(ABI)/bopl_rl.c -ldl -lm -o $@

# Sudden-death publication gate: the tick each backend first PUBLISHES
# header.sudden_death, on maps 0..4. GameSessionHandler raises the flag from
# inside the updater pass against a clock the tick has not advanced yet, so the
# port's post-increment session update is a pass ahead of it unless the
# published bit lags by one.
sudden-death-publication-ab: $(OUT)/sudden_death_publication_ab rust-q32
	$(abspath $(OUT)/sudden_death_publication_ab) $(abspath $(REFERENCE)) $(abspath $(RUST_ROLLOUT_Q32))

$(OUT)/sudden_death_publication_ab: eval/rust_ab/sudden_death_publication_ab.c $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) $(INCLUDES) eval/rust_ab/sudden_death_publication_ab.c $(ABI)/bopl_rl.c -ldl -lm -o $@

# Sudden-death endgame gate: maps 0-4 played out with no input at all, every
# entity, slime and player field compared exactly from reset through the tick
# that resolves the round (level 7302-8218 depending on the map). This is the
# only gate that sees the game after sudden death begins at level tick 7200;
# map-sweep-3000 stops 4,200 ticks short of it. Found the round-resolution
# sudden-death flag bug: WinGame lowers SuddenDeathInProgress before the
# authored level updaters run, so map 2's AntiLockPlatform orbiter takes one
# more waypoint kick on the deciding tick.
sudden-death-chunk-ab: $(OUT)/sudden_death_chunk_ab rust-q32
	BOPL_RL_ABILITY_SPAWNS=1 $(abspath $(OUT)/sudden_death_chunk_ab) $(abspath $(REFERENCE)) $(abspath $(RUST_ROLLOUT_Q32))

$(OUT)/sudden_death_chunk_ab: eval/rust_ab/sudden_death_chunk_ab.c $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) $(INCLUDES) eval/rust_ab/sudden_death_chunk_ab.c $(ABI)/bopl_rl.c -ldl -lm -o $@

# The same authored-map gate as map-sweep-3000, run long enough that every
# map reaches sudden death and the round resolves. Strictly wider coverage at
# ~3x the runtime, so both targets exist: keep 3000 as the quick gate.
# Nothing is excluded here: the header's sudden_death flag is compared like
# every other field, which is what makes this the gate for the observation
# writer's one-pass publication lag.
map-sweep-endgame: $(OUT)/map_ab rust-q32
	MAP_AB_STRICT_ACTORS=1 $(abspath $(OUT)/map_ab) $(abspath $(REFERENCE)) $(abspath $(RUST_ROLLOUT_Q32)) 8200

# Randomised-loadout gate: all 38 abilities, stratified over all 124 maps.
# Every other gate here plays the hardcoded Dash/Bow/Platform kit
# (train/bopl_bots.h:143-149), so the suite only ever exercises 3 of the 38
# abilities - and all three defects in eval/divergence/FIXTURES.md were
# found in that blind spot, none of them reachable from the gate loadout. This
# runs the port against two more copies of itself, so it needs no oracle and
# has no known-divergence baseline to drift against: crashes and self-
# disagreement must both be exactly zero. 24,800 episodes, ~1 min wall on four
# workers. Subsumes and supersedes the leakprobe-q32/selfcheck-q32 control
# cells, which stay in run_divergence_controls.sh at the study's 1240 games
# because FIXTURES.md quotes those numbers.
ABILITY_SWEEP_OUT ?= native/artifacts/rl/ability-sweep
ABILITY_SWEEP_WORKERS ?= 4
ability-sweep: $(OUT)/divergence_rate rust-q32
	DR_BIN=$(abspath $(OUT)/divergence_rate) RUST_ROLLOUT_Q32=$(abspath $(RUST_ROLLOUT_Q32)) \
	  eval/divergence/run_ability_sweep.sh $(abspath $(ABILITY_SWEEP_OUT)) $(ABILITY_SWEEP_WORKERS)

# Whole rust_ab differential suite, comparator included. Seventeen probes in
# eval/rust_ab only *print* a per-backend trace and always exit 0, so
# checking exit status alone reports them as passing unconditionally; this
# target runs trace_compare.py over their output and fails on any difference.
# See eval/rust_ab/README.md "Trace-only probes".
rust-ab-suite: rust-q32
	REFERENCE=$(abspath $(REFERENCE)) RUST=$(abspath $(RUST_ROLLOUT_Q32)) \
	  eval/rust_ab/run_suite.sh $(PROBE)

# The comparator on its own, for a capture you already have:
#   make rust-ab-trace CAPTURE=/tmp/out.csv
rust-ab-trace:
	$(PYTHON) eval/rust_ab/trace_compare.py $(CAPTURE)

# Real-game replay differential: re-steps saved evaluation games (the .rep
# recordings beside every evaluations/*.json) in both backends with the
# recorded per-tick inputs and reports the first divergence per game.
replay-ab: $(OUT)/replay_ab

$(OUT)/replay_ab: eval/rust_ab/replay_ab.c $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) $(INCLUDES) eval/rust_ab/replay_ab.c $(ABI)/bopl_rl.c -ldl -lm -o $@

# Whole-directory sweep over one study or sweep evaluation directory:
#   make replay-sweep GAMES=native/artifacts/rl/study-grenade/evaluations
GAMES ?= native/artifacts/rl/study-grenade/evaluations
# `REPLAY_RUST=$(RUST_ROLLOUT)` compares the shipping f64 build instead, which
# only makes sense with a loose REPLAY_AB_TOL. The default 1 raw unit (2.3e-10)
# absorbs the one-ulp residue q32 leaves on statically authored extents at tick
# 0; 1 and 4 report the same divergences, so it is not hiding drift.
REPLAY_RUST ?= $(RUST_ROLLOUT_Q32)
replay-sweep: $(OUT)/replay_ab rust-q32
	REPLAY_AB_TOL=$${REPLAY_AB_TOL-1} $(abspath $(OUT)/replay_ab) $(abspath $(REFERENCE)) $(abspath $(REPLAY_RUST)) $(abspath $(GAMES))

# The same sweep restricted to what a trained policy can actually read
# (bopl_puffer.h:316): the ranking that answers whether a divergence could
# have changed a policy's behaviour.
replay-visible: $(OUT)/replay_ab rust-q32
	REPLAY_AB_VISIBLE=1 REPLAY_AB_TOL=$${REPLAY_AB_TOL-1} $(abspath $(OUT)/replay_ab) $(abspath $(REFERENCE)) $(abspath $(REPLAY_RUST)) $(abspath $(GAMES))

# Self check: the reference replayed against itself must be bit-identical.
replay-selfcheck: $(OUT)/replay_ab rust-q32
	REPLAY_AB_TOL=0 $(abspath $(OUT)/replay_ab) $(abspath $(REFERENCE)) $(abspath $(REFERENCE)) $(abspath $(GAMES))
	REPLAY_AB_TOL=0 $(abspath $(OUT)/replay_ab) $(abspath $(REPLAY_RUST)) $(abspath $(REPLAY_RUST)) $(abspath $(GAMES))

# The stronger half of the self check: the recorded inputs, replayed in the
# backend that produced the recording, must reproduce the saved <game>.json.
# LIBRARY must be a build of the commit the corpus was recorded with - the
# prebuilt `native/target/release` in a working checkout is not reliably that, and
# replaying a corpus in a later build fails this by design.
PYTHON ?= python3
LIBRARY ?= $(RUST_ROLLOUT)
replay-roundtrip: $(OUT)/replay_ab
	native/artifacts/rl/venv/bin/python eval/rust_ab/replay_ab_selfcheck.py \
	  --binary $(abspath $(OUT)/replay_ab) --library $(abspath $(LIBRARY)) $(GAMES)/*/*.rep

# How long does the port agree with the oracle, and about what?
#
# Every other differential here checks something somebody thought to check, so
# what they can report is "these bugs are fixed", not how much is left. This
# one plays whole games from randomly drawn seeds, maps and loadouts, drives
# the oracle and BOTH Rust builds off one action stream, and measures the
# survival curve of ticks-to-first-divergence. The product is a rate, and a
# regression metric to re-measure after any change.
#
# It deliberately lives in eval/divergence and not eval/rust_ab:
# run_suite.sh globs that directory, so a probe added there changes the gate
# result for everybody. This target is not a gate and never fails a build.
#
#   make divergence-rate            # the full study (~1 h, 4 workers)
#   make divergence-report          # curves + attribution + coverage
#   DIVERGENCE_WORKERS=2 make divergence-rate   # on a busy machine
DIVERGENCE_OUT ?= native/artifacts/rl/divergence-rate
DIVERGENCE_WORKERS ?= 4

divergence-rate: $(OUT)/divergence_rate rust-q32 $(RUST_ROLLOUT)
	DR_BIN=$(abspath $(OUT)/divergence_rate) REFERENCE=$(abspath $(REFERENCE)) \
	  RUST_ROLLOUT_Q32=$(abspath $(RUST_ROLLOUT_Q32)) RUST_ROLLOUT=$(abspath $(RUST_ROLLOUT)) \
	  eval/divergence/run_divergence_rate.sh $(abspath $(DIVERGENCE_OUT)) $(DIVERGENCE_WORKERS)

# The controls the study's numbers depend on: the instrument measured against
# itself, and the narrow loadout the existing gates use.
divergence-controls: $(OUT)/divergence_rate rust-q32 $(RUST_ROLLOUT)
	DR_BIN=$(abspath $(OUT)/divergence_rate) REFERENCE=$(abspath $(REFERENCE)) \
	  RUST_ROLLOUT_Q32=$(abspath $(RUST_ROLLOUT_Q32)) RUST_ROLLOUT=$(abspath $(RUST_ROLLOUT)) \
	  eval/divergence/run_divergence_controls.sh $(abspath $(DIVERGENCE_OUT))-controls $(DIVERGENCE_WORKERS)

divergence-report:
	$(PYTHON) eval/divergence/divergence_summary.py $(abspath $(DIVERGENCE_OUT))

# The f64 build the library actually trains in, profile-guided; the q32 one
# is `rust-q32`.
$(RUST_ROLLOUT):
	native/tools/pgo_build.sh

$(OUT)/divergence_rate: eval/divergence/divergence_rate.c eval/divergence/divergence_compare.h $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h train/bopl_bots.h train/bopl_bots_v2.h train/bopl_bots_v2_play.h train/bopl_layout.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -std=gnu11 -Wno-unused-function $(INCLUDES) -Ieval/divergence \
	  eval/divergence/divergence_rate.c $(ABI)/bopl_rl.c -ldl -lm -lpthread -o $@

# Observation-writer microbenchmark + sim ticks/s on one core.
writer-bench: $(OUT)/writer_bench

$(OUT)/writer_bench: eval/writer_bench.c train/bopl_puffer.h train/bopl_obs_v4.h train/bopl_rays_v4.h train/bopl_obs_pack.h train/bopl_layout_v4.h train/bopl_bots.h train/bopl_layout.h train/bopl_aim.h $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -std=gnu11 -Wno-unused-function $(INCLUDES) eval/writer_bench.c $(ABI)/bopl_rl.c -lm -lpthread -o $@

# The trainer's c_step phases timed over a replay corpus (eval/env_bench.c),
# with the writer built as train/build.py builds the trainer's.
env-bench: $(OUT)/env_bench

$(OUT)/env_bench: eval/env_bench.c train/bopl_puffer.h train/bopl_obs_v4.h train/bopl_rays_v4.h train/bopl_obs_pack.h train/bopl_layout_v4.h train/bopl_bots.h train/bopl_layout.h train/bopl_aim.h $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	clang $(CFLAGS) -march=znver4 -ffp-contract=off -std=gnu11 -Wno-unused-function $(INCLUDES) eval/env_bench.c $(ABI)/bopl_rl.c -lm -lpthread -o $@

# The sim's cost with the trainer's cache behaviour: K worlds stepped
# round-robin (eval/env_interleave.c); eval/perf/env_ilv_ab.sh runs one copy
# per hardware thread.
env-interleave: $(OUT)/env_interleave

$(OUT)/env_interleave: eval/env_interleave.c eval/env_bench.c train/bopl_puffer.h train/bopl_obs_v4.h train/bopl_rays_v4.h train/bopl_obs_pack.h train/bopl_layout_v4.h train/bopl_bots.h train/bopl_layout.h train/bopl_aim.h $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	clang $(CFLAGS) -march=znver4 -ffp-contract=off -std=gnu11 -Wno-unused-function $(INCLUDES) eval/env_interleave.c $(ABI)/bopl_rl.c -lm -lpthread -o $@

# The GPU v4 row writer (pack mode 4, train/bopl_rays_v4.cu) against the CPU
# writer bit for bit over a replay corpus (eval/encode_check.c). The writer
# objects are built as train/build.py builds the trainer's: clang with
# -march=znver4 -ffp-contract=off for the host, nvcc -fmad=false for the kernel.
CUDA_HOME ?= native/artifacts/rl/venv/lib/python3.11/site-packages/nvidia/cu13
encode-check: $(OUT)/encode_check

$(OUT)/encode_check: eval/encode_check.c eval/env_bench.c train/bopl_rays_v4.cu train/bopl_rows_v4.h train/bopl_puffer.h train/bopl_obs_v4.h train/bopl_rays_v4.h train/bopl_obs_pack.h train/bopl_layout_v4.h train/bopl_layout.h $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	$(CUDA_HOME)/bin/nvcc -O3 -std=c++17 -arch=sm_120 -fmad=false -ccbin g++-15 $(INCLUDES) -c train/bopl_rays_v4.cu -o $(OUT)/encode_check_rays.o
	clang $(CFLAGS) -march=znver4 -ffp-contract=off -std=gnu11 -Wno-unused-function $(INCLUDES) -I$(CUDA_HOME)/include -c eval/encode_check.c -o $(OUT)/encode_check.o
	clang $(CFLAGS) -march=znver4 -ffp-contract=off -std=gnu11 $(INCLUDES) -c $(ABI)/bopl_rl.c -o $(OUT)/encode_check_rl.o
	g++-15 $(OUT)/encode_check.o $(OUT)/encode_check_rl.o $(OUT)/encode_check_rays.o \
	  $(abspath $(CUDA_HOME))/lib/libcudart.so.13 -Wl,-rpath,$(abspath $(CUDA_HOME))/lib -lm -lpthread -ldl -o $@

$(OUT)/libbopl_research.so: train/eval_binding.c train/bopl_puffer.h train/bopl_obs_v4.h train/bopl_rays_v4.h train/bopl_obs_pack.h train/bopl_layout_v4.h train/bopl_bots.h train/bopl_bots_v2.h train/bopl_bots_v2_play.h train/bopl_layout.h train/bopl_aim.h $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -Wno-unused-function -fPIC -shared $(INCLUDES) train/eval_binding.c $(ABI)/bopl_rl.c -lm -o $@

# In-game agent shim (train/bopl_agent.h): the training decision loop fed by
# a running game, plus the CPU policy evaluator. Loaded by the Unity build
# (Assets/Scripts/Assembly-CSharp/RlAgent.cs) and eval/agent_shim.py.
PUFFERLIB ?= $(OUT)/dependencies/PufferLib-42f70d6932c30ac977736f861006809c50168ba9/src
agent: $(OUT)/libbopl_agent.so

$(OUT)/libbopl_agent.so: train/bopl_agent.c train/bopl_agent.h train/policy_eval.c train/bopl_set_encoder.h train/bopl_v4_eval.h train/bopl_obs_v4.h train/bopl_puffer.h train/bopl_bots.h train/bopl_bots_v2.h train/bopl_bots_v2_play.h train/bopl_layout.h train/bopl_layout_v4.h train/bopl_aim.h $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -std=gnu11 -Wno-unused-function -fPIC -c $(INCLUDES) train/bopl_agent.c -o $(OUT)/bopl_agent.o
	$(CC) $(CFLAGS) -fPIC -c $(ABI)/bopl_rl.c -o $(OUT)/bopl_agent_rl.o
	$(CC) -O3 -std=gnu11 -fPIC -c -I$(PUFFERLIB) -Itrain train/policy_eval.c -o $(OUT)/bopl_agent_policy.o
	$(CC) -shared $(OUT)/bopl_agent.o $(OUT)/bopl_agent_rl.o $(OUT)/bopl_agent_policy.o -lm -lpthread -o $@
	rm -f $(OUT)/bopl_agent.o $(OUT)/bopl_agent_rl.o $(OUT)/bopl_agent_policy.o

# The same shim for the macOS player (arm64, run on the Mac: `make agent-mac`);
# Mono resolves DllImport("bopl_agent") to BoplBattle.app/Contents/Plugins/libbopl_agent.dylib.
agent-mac: $(OUT)/libbopl_agent.dylib

$(OUT)/libbopl_agent.dylib: train/bopl_agent.c train/bopl_agent.h train/policy_eval.c train/bopl_set_encoder.h train/bopl_v4_eval.h train/bopl_obs_v4.h train/bopl_puffer.h train/bopl_bots.h train/bopl_bots_v2.h train/bopl_bots_v2_play.h train/bopl_layout.h train/bopl_layout_v4.h train/bopl_aim.h $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -std=gnu11 -Wno-unused-function -fPIC -c $(INCLUDES) train/bopl_agent.c -o $(OUT)/bopl_agent.o
	$(CC) $(CFLAGS) -fPIC -c $(ABI)/bopl_rl.c -o $(OUT)/bopl_agent_rl.o
	$(CC) -O3 -std=gnu11 -fPIC -c -I$(PUFFERLIB) -Itrain train/policy_eval.c -o $(OUT)/bopl_agent_policy.o
	$(CC) -dynamiclib $(OUT)/bopl_agent.o $(OUT)/bopl_agent_rl.o $(OUT)/bopl_agent_policy.o -lm -lpthread -install_name @rpath/libbopl_agent.dylib -o $@
	rm -f $(OUT)/bopl_agent.o $(OUT)/bopl_agent_rl.o $(OUT)/bopl_agent_policy.o

$(OUT)/libbopl_rl.so: $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -fPIC -shared $(ABI)/bopl_rl.c -o $@

$(OUT)/benchmark: eval/benchmark.c $(ABI)/bopl_rl.h $(OUT)/libbopl_rl.so
	$(CC) $(CFLAGS) -I$(ABI) eval/benchmark.c -L$(OUT) -lbopl_rl -Wl,-rpath,'$$ORIGIN' -o $@

$(OUT)/coverage: eval/coverage.c $(ABI)/bopl_rl.h $(OUT)/libbopl_rl.so
	$(CC) $(CFLAGS) -I$(ABI) eval/coverage.c -L$(OUT) -lbopl_rl -Wl,-rpath,'$$ORIGIN' -o $@

$(OUT)/record: eval/record.c $(ABI)/bopl_rl.h $(OUT)/libbopl_rl.so
	$(CC) $(CFLAGS) -I$(ABI) eval/record.c -L$(OUT) -lbopl_rl -Wl,-rpath,'$$ORIGIN' -o $@

$(OUT)/reset_check: eval/reset_check.c $(ABI)/bopl_rl.h $(OUT)/libbopl_rl.so
	$(CC) $(CFLAGS) -I$(ABI) eval/reset_check.c -L$(OUT) -lbopl_rl -Wl,-rpath,'$$ORIGIN' -o $@

$(OUT)/sampled_check: eval/sampled_check.c $(ABI)/bopl_rl.h $(OUT)/libbopl_rl.so
	$(CC) $(CFLAGS) -I$(ABI) eval/sampled_check.c -L$(OUT) -lbopl_rl -Wl,-rpath,'$$ORIGIN' -o $@

$(OUT)/reset_profile: eval/reset_profile.c $(ABI)/bopl_rl.h $(OUT)/libbopl_rl.so
	$(CC) $(CFLAGS) -std=gnu11 -I$(ABI) eval/reset_profile.c -L$(OUT) -lbopl_rl -Wl,-rpath,'$$ORIGIN' -o $@

$(OUT)/frame_profile: eval/frame_profile.c $(ABI)/bopl_rl.h $(OUT)/libbopl_rl.so
	$(CC) $(CFLAGS) -std=gnu11 -I$(ABI) eval/frame_profile.c -L$(OUT) -lbopl_rl -Wl,-rpath,'$$ORIGIN' -o $@

.PHONY: divergence-rate divergence-controls divergence-report ability-sweep all research map-ab map-sweep-3000 map-sweep-endgame bounds-ability-ab bounds-case-trace scale-boundary-ab sudden-death-chunk-ab sudden-death-publication-ab rust-ab-suite rust-ab-trace replay-ab replay-sweep replay-visible replay-selfcheck replay-roundtrip rust-q32 writer-bench env-bench env-interleave encode-check agent agent-mac check-reward check-routing

# Reward semantics on hand-built terminal states (eval/check_reward.c):
# the five outcomes a match can have, and the incentive ordering between them.
# No game process and no oracle - both backends already agree on the counters
# this reads, so what is under test is only what the reward makes of them.
check-reward: $(OUT)/check_reward
	$(OUT)/check_reward

# Per-player routing: PufferLib's agent permutation, the frozen bank slices, and
# the tag / count_aligned alignment protocol that decides which matches count
# towards per-bank pool statistics. No game process and no GPU - it loads a
# backend library and steps it on the host path.
#
# Both backends on purpose. They must agree, and *how* they disagree is the
# diagnostic: if the oracle and the port fail the same checks the bug is in the
# check, because two independent implementations do not share one; if only the
# port fails, it is a port divergence. Diagnosing the 2026-09-24 failures took
# exactly that comparison (FIXTURES.md, "Open items").
#
# --output must be absolute: it becomes the native log path, and bopl_rl_open
# rejects a relative executable/replay/log outright (and aborts doing it).
check-routing: $(RUST_ROLLOUT)
	native/artifacts/rl/venv/bin/python train/check_routing.py \
	  --game $(abspath $(REFERENCE)) --output $(abspath $(OUT))/check-routing-oracle
	native/artifacts/rl/venv/bin/python train/check_routing.py \
	  --game $(abspath $(RUST_ROLLOUT)) --output $(abspath $(OUT))/check-routing-port

$(OUT)/check_reward: eval/check_reward.c train/bopl_puffer.h train/bopl_obs_v4.h train/bopl_rays_v4.h train/bopl_obs_pack.h train/bopl_layout_v4.h train/bopl_bots.h train/bopl_layout.h train/bopl_aim.h $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -std=gnu11 -Wno-unused-function $(INCLUDES) eval/check_reward.c $(ABI)/bopl_rl.c -ldl -lm -lpthread -o $@
