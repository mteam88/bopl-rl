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

$(OUT)/replay_ab: eval/rust_ab/replay_ab.c $(ABI)/bopl_rl_compare.h $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
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
# BOPL_ENV_BENCH_STYLE=<style.bin> (train/style_compile.py -o) adds the style
# phase: style_net_inst / style_net_pct against env_train (the trainer's c_step).
env-bench: $(OUT)/env_bench

$(OUT)/env_bench: eval/env_bench.c train/bopl_puffer.h train/bopl_style.h train/bopl_obs_v4.h train/bopl_rays_v4.h train/bopl_obs_pack.h train/bopl_layout_v4.h train/bopl_bots.h train/bopl_layout.h train/bopl_aim.h $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	clang $(CFLAGS) -march=znver4 -ffp-contract=off -std=gnu11 -Wno-unused-function $(INCLUDES) eval/env_bench.c $(ABI)/bopl_rl.c -lm -lpthread -o $@

# The sim's cost with the trainer's cache behaviour: K worlds stepped
# round-robin (eval/env_interleave.c); eval/perf/env_ilv_ab.sh runs one copy
# per hardware thread.
env-interleave: $(OUT)/env_interleave

$(OUT)/env_interleave: eval/env_interleave.c eval/env_bench.c train/bopl_puffer.h train/bopl_obs_v4.h train/bopl_rays_v4.h train/bopl_obs_pack.h train/bopl_layout_v4.h train/bopl_bots.h train/bopl_layout.h train/bopl_aim.h $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	clang $(CFLAGS) -march=znver4 -ffp-contract=off -std=gnu11 -Wno-unused-function $(INCLUDES) eval/env_interleave.c $(ABI)/bopl_rl.c -lm -lpthread -o $@

# The trainer's sample + finish (the delayed v4 rows, tail, masks, privileged
# block, pack) with its cache behaviour: K worlds stepped round-robin, each its
# own BoplPuffer (eval/env_ilv_train.c); eval/perf/env_ilv_train_ab.sh pairs
# two builds of it.
env-ilv-train: $(OUT)/env_ilv_train

$(OUT)/env_ilv_train: eval/env_ilv_train.c eval/env_bench.c train/bopl_puffer.h train/bopl_obs_v4.h train/bopl_rays_v4.h train/bopl_obs_pack.h train/bopl_layout_v4.h train/bopl_bots.h train/bopl_layout.h train/bopl_aim.h $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	clang $(CFLAGS) -march=znver4 -ffp-contract=off -std=gnu11 -Wno-unused-function $(INCLUDES) eval/env_ilv_train.c $(ABI)/bopl_rl.c -lm -lpthread -o $@

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

# Its negative controls for layout v4.4's line to the enemy (bopl_rays_v4.cu
# BR4_LINE_NEG): encode_check_neg1 casts the line without the exit march,
# encode_check_neg2 samples it half a step early. Each must mismatch at writer
# bits 112, in the line's slot alone (native/rl/research/drill-sense/sense-gate.sh).
encode-check-neg: $(OUT)/encode_check_neg1 $(OUT)/encode_check_neg2

$(OUT)/encode_check_neg%: $(OUT)/encode_check
	$(CUDA_HOME)/bin/nvcc -O3 -std=c++17 -arch=sm_120 -fmad=false -ccbin g++-15 -DBR4_LINE_NEG=$* $(INCLUDES) -c train/bopl_rays_v4.cu -o $(OUT)/encode_check_rays_neg$*.o
	g++-15 $(OUT)/encode_check.o $(OUT)/encode_check_rl.o $(OUT)/encode_check_rays_neg$*.o \
	  $(abspath $(CUDA_HOME))/lib/libcudart.so.13 -Wl,-rpath,$(abspath $(CUDA_HOME))/lib -lm -lpthread -ldl -o $@

# The policy v4 rollout forward of one buffer step on dumped trainer steps
# (train/fwd_bench.cu): bit-exactness against the trainer and between builds,
# and the step's latency on 1 and 4 streams. Built against the trainer's
# overlay (run train/build.py --device cuda first); the row writer object as
# build.py builds it (-fmad=false).
PUFFER_OVERLAY ?= native/artifacts/rl/puffer-overlay
PUFFER_SRC ?= native/artifacts/rl/dependencies/PufferLib-42f70d6932c30ac977736f861006809c50168ba9/src
fwd-bench: $(OUT)/fwd_bench

$(OUT)/fwd_bench: train/fwd_bench.cu train/bopl_rollout.cu train/bopl_rollout_v4.cu train/bopl_v4.cu train/bopl_rays_v4.cu train/bopl_rows_v4.h train/bopl_rays_v4.h train/bopl_obs_pack.h train/bopl_layout_v4.h train/bopl_layout.h train/bopl_aim.h
	mkdir -p $(OUT)
	$(CUDA_HOME)/bin/nvcc -O3 -std=c++17 -arch=sm_120 -fmad=false -ccbin g++-15 $(INCLUDES) -c train/bopl_rays_v4.cu -o $(OUT)/fwd_bench_rays.o
	$(CUDA_HOME)/bin/nvcc -O3 -std=c++17 -arch=sm_120 -ccbin g++-15 -DPRECISION_FLOAT -I$(PUFFER_OVERLAY) -I$(PUFFER_SRC) $(INCLUDES) -c train/fwd_bench.cu -o $(OUT)/fwd_bench.o
	g++-15 $(OUT)/fwd_bench.o $(OUT)/fwd_bench_rays.o $(abspath $(CUDA_HOME))/lib/libcudart.so.13 $(abspath $(CUDA_HOME))/lib/libcublas.so.13 \
	  $(abspath $(CUDA_HOME))/lib/libcublasLt.so.13 $(abspath $(CUDA_HOME))/lib/libcurand.so.10 -lcuda -Wl,-rpath,$(abspath $(CUDA_HOME))/lib -lm -lpthread -ldl -o $@

# The research library also carries the CPU policy evaluator (train/policy_eval.c,
# as in the agent shim below) for decision-time search (train/bopl_search.h).
# -march=native -ffast-math vectorises its scalar loops (9x: 800 -> 90 us per
# row at hidden 256; logits and values agree to 5 decimals), so the library is
# built for the machine that runs it.
$(OUT)/libbopl_research.so: train/eval_binding.c train/bopl_search.h train/policy_eval.c train/bopl_set_encoder.h train/bopl_v4_eval.h train/bopl_puffer.h train/bopl_style.h train/bopl_obs_v4.h train/bopl_rays_v4.h train/bopl_obs_pack.h train/bopl_layout_v4.h train/bopl_bots.h train/bopl_bots_v2.h train/bopl_bots_v2_play.h train/bopl_layout.h train/bopl_aim.h $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -Wno-unused-function -fPIC -c $(INCLUDES) train/eval_binding.c -o $(OUT)/bopl_research.o
	$(CC) $(CFLAGS) -fPIC -c $(ABI)/bopl_rl.c -o $(OUT)/bopl_research_rl.o
	$(CC) -O3 -std=gnu11 -march=native -ffast-math -fPIC -c -I$(PUFFERLIB) -Itrain train/policy_eval.c -o $(OUT)/bopl_research_policy.o
	$(CC) -shared $(OUT)/bopl_research.o $(OUT)/bopl_research_rl.o $(OUT)/bopl_research_policy.o -lm -lpthread -o $@
	rm -f $(OUT)/bopl_research.o $(OUT)/bopl_research_rl.o $(OUT)/bopl_research_policy.o

# In-game agent shim (train/bopl_agent.h): the training decision loop fed by
# a running game, plus the CPU policy evaluator. Loaded by the Unity build
# (bridge/RlAgent.cs, compiled into the game through game/Assets/Bridge) and eval/agent_shim.py.
PUFFERLIB ?= $(OUT)/dependencies/PufferLib-42f70d6932c30ac977736f861006809c50168ba9/src
# The policy evaluator is built for the machine that runs the game, like the
# research library's: -ffast-math plus the host's vector ISA turns ~800 us
# per row into ~90 us, which the in-game search (BOPL_AGENT_SEARCH,
# docs/INGAME_SEARCH.md) needs to fit 8,1,1 into a decision. Override for a
# portable build (AGENT_EVAL_CPU=-march=x86-64-v3, AGENT_MAC_EVAL_CPU=-mcpu=apple-m1).
# Fast-math reorders float sums, so the shim's logits are no longer bitwise
# equal to the torch reference; eval/check_agent_fidelity.py needs the exact
# build: make agent AGENT_EVAL_CPU= AGENT_EVAL_FAST=
AGENT_EVAL_FAST ?= -ffast-math
AGENT_EVAL_CPU ?= -march=native
AGENT_MAC_EVAL_CPU ?= -mcpu=native
agent: $(OUT)/libbopl_agent.so

$(OUT)/libbopl_agent.so: train/bopl_agent.c train/bopl_agent.h train/bopl_search.h train/policy_eval.c train/bopl_set_encoder.h train/bopl_v4_eval.h train/bopl_obs_v4.h train/bopl_puffer.h train/bopl_bots.h train/bopl_bots_v2.h train/bopl_bots_v2_play.h train/bopl_layout.h train/bopl_layout_v4.h train/bopl_aim.h train/bopl_shadow.c train/bopl_shadow.h $(ABI)/bopl_rl_compare.h $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -std=gnu11 -Wno-unused-function -fPIC -c $(INCLUDES) train/bopl_agent.c -o $(OUT)/bopl_agent.o
	$(CC) $(CFLAGS) -fPIC -c $(ABI)/bopl_rl.c -o $(OUT)/bopl_agent_rl.o
	$(CC) -O3 -std=gnu11 $(AGENT_EVAL_CPU) $(AGENT_EVAL_FAST) -fPIC -c -I$(PUFFERLIB) -Itrain train/policy_eval.c -o $(OUT)/bopl_agent_policy.o
	$(CC) $(CFLAGS) -std=gnu11 -fPIC -c $(INCLUDES) train/bopl_shadow.c -o $(OUT)/bopl_agent_shadow.o
	$(CC) -shared $(OUT)/bopl_agent.o $(OUT)/bopl_agent_rl.o $(OUT)/bopl_agent_policy.o $(OUT)/bopl_agent_shadow.o -ldl -lm -lpthread -o $@
	rm -f $(OUT)/bopl_agent.o $(OUT)/bopl_agent_rl.o $(OUT)/bopl_agent_policy.o $(OUT)/bopl_agent_shadow.o

# The same shim for the macOS player (arm64, run on the Mac: `make agent-mac`);
# Mono resolves DllImport("bopl_agent") to BoplBattle.app/Contents/Plugins/libbopl_agent.dylib.
agent-mac: $(OUT)/libbopl_agent.dylib

$(OUT)/libbopl_agent.dylib: train/bopl_agent.c train/bopl_agent.h train/bopl_search.h train/policy_eval.c train/bopl_set_encoder.h train/bopl_v4_eval.h train/bopl_obs_v4.h train/bopl_puffer.h train/bopl_bots.h train/bopl_bots_v2.h train/bopl_bots_v2_play.h train/bopl_layout.h train/bopl_layout_v4.h train/bopl_aim.h train/bopl_shadow.c train/bopl_shadow.h $(ABI)/bopl_rl_compare.h $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -std=gnu11 -Wno-unused-function -fPIC -c $(INCLUDES) train/bopl_agent.c -o $(OUT)/bopl_agent.o
	$(CC) $(CFLAGS) -fPIC -c $(ABI)/bopl_rl.c -o $(OUT)/bopl_agent_rl.o
	$(CC) -O3 -std=gnu11 $(AGENT_MAC_EVAL_CPU) $(AGENT_EVAL_FAST) -fPIC -c -I$(PUFFERLIB) -Itrain train/policy_eval.c -o $(OUT)/bopl_agent_policy.o
	$(CC) $(CFLAGS) -std=gnu11 -fPIC -c $(INCLUDES) train/bopl_shadow.c -o $(OUT)/bopl_agent_shadow.o
	$(CC) -dynamiclib $(OUT)/bopl_agent.o $(OUT)/bopl_agent_rl.o $(OUT)/bopl_agent_policy.o $(OUT)/bopl_agent_shadow.o -lm -lpthread -install_name @rpath/libbopl_agent.dylib -o $@
	rm -f $(OUT)/bopl_agent.o $(OUT)/bopl_agent_rl.o $(OUT)/bopl_agent_policy.o $(OUT)/bopl_agent_shadow.o

# The shadow sim the shim loads with BOPL_AGENT_SHADOW=1 (train/bopl_shadow.h,
# docs/SHADOW.md): the bit-exact q32 Rust sim, found beside libbopl_agent.
# Tools/play-vs-agent.sh copies it into the player's plugin directory.
agent-shadow-lib: rust-q32
	mkdir -p $(OUT)
	cp $(RUST_ROLLOUT_Q32) $(OUT)/libbopl_sim_q32.so

# The same on the Mac (Tools/play-vs-agent-mac.sh), from native/ synced there
# (docs/INGAME_SEARCH.md): cargo build --release --features q32.
agent-shadow-lib-mac:
	cargo build --release --features q32 --manifest-path native/Cargo.toml --target-dir native/target-q32
	mkdir -p $(OUT)
	cp native/target-q32/release/libbopl_sim.dylib $(OUT)/libbopl_sim_q32.dylib

# Offline proof of the shim's shadow path (eval/shadow_replay.c): real game
# recordings, the .NET reference standing in for the running game.
shadow-replay: $(OUT)/shadow_replay rust-q32
	$(abspath $(OUT)/shadow_replay) $(abspath $(REFERENCE)) $(abspath $(RUST_ROLLOUT_Q32)) $(abspath $(GAMES))

$(OUT)/shadow_replay: eval/shadow_replay.c train/bopl_shadow.c train/bopl_shadow.h $(ABI)/bopl_rl_compare.h $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -std=gnu11 $(INCLUDES) eval/shadow_replay.c train/bopl_shadow.c $(ABI)/bopl_rl.c -ldl -lm -lpthread -o $@

# Offline proof that the in-game search (bopl_agent_search) plays exactly as
# the research library's (eval/shim_search_check.c, docs/INGAME_SEARCH.md):
# make shim-search-check SEARCH_CHECKPOINT=<policy-*.bin> [SEARCH_CHECK_ARGS="--players 3"]
SEARCH_CHECK_ARGS ?=
shim-search-check: $(OUT)/shim_search_check rust-q32
	$(abspath $(OUT)/shim_search_check) $(abspath $(RUST_ROLLOUT_Q32)) $(abspath $(SEARCH_CHECKPOINT)) $(SEARCH_CHECK_ARGS)

$(OUT)/shim_search_check: eval/shim_search_check.c train/bopl_agent.c train/bopl_agent.h train/bopl_search.h train/bopl_puffer.h train/policy_eval.c train/bopl_shadow.c train/bopl_shadow.h $(ABI)/bopl_rl_compare.h $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	$(CC) -O3 -std=gnu11 -Wall -Wextra -Werror -Wno-unused-function $(INCLUDES) -c eval/shim_search_check.c -o $(OUT)/shim_search_check.o
	$(CC) -O3 -std=gnu11 $(AGENT_EVAL_CPU) $(AGENT_EVAL_FAST) -c -I$(PUFFERLIB) -Itrain train/policy_eval.c -o $(OUT)/shim_search_check_policy.o
	$(CC) $(CFLAGS) -std=gnu11 $(INCLUDES) -c train/bopl_shadow.c -o $(OUT)/shim_search_check_shadow.o
	$(CC) $(CFLAGS) -c $(ABI)/bopl_rl.c -o $(OUT)/shim_search_check_rl.o
	$(CC) $(OUT)/shim_search_check.o $(OUT)/shim_search_check_policy.o $(OUT)/shim_search_check_shadow.o $(OUT)/shim_search_check_rl.o -ldl -lm -lpthread -o $@
	rm -f $(OUT)/shim_search_check.o $(OUT)/shim_search_check_policy.o $(OUT)/shim_search_check_shadow.o $(OUT)/shim_search_check_rl.o

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

.PHONY: shim-search-check divergence-rate divergence-controls divergence-report ability-sweep all research map-ab map-sweep-3000 map-sweep-endgame bounds-ability-ab bounds-case-trace scale-boundary-ab sudden-death-chunk-ab sudden-death-publication-ab rust-ab-suite rust-ab-trace replay-ab replay-sweep replay-visible replay-selfcheck replay-roundtrip rust-q32 writer-bench env-bench env-interleave env-ilv-train encode-check encode-check-neg fwd-bench agent agent-mac agent-shadow-lib agent-shadow-lib-mac shadow-replay check-reward check-routing check-learner-first check-bank-kits style-check style-score style-rates style-rates-check

# Reward semantics on hand-built terminal states (eval/check_reward.c):
# the five outcomes a match can have, and the incentive ordering between them.
# No game process and no oracle - both backends already agree on the counters
# this reads, so what is under test is only what the reward makes of them.
# Style programs (docs/STYLES.md): the C VM (train/bopl_style.h, built as the
# trainer builds it) and eval/style_score.py's numpy interpreter must agree bit
# for bit on every decision of the human replay dumps, with and without
# (synthetic) events. STYLE_CHECK_GAMES=0 runs every dump.
STYLE_FILES := $(filter-out styles/categories.toml,$(wildcard styles/*.toml))
STYLE_CHECK_GAMES ?= 0
$(OUT)/libstyle_check.so: eval/style_check.c train/bopl_style.h $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	clang $(CFLAGS) -march=znver4 -ffp-contract=off -std=gnu11 -Wno-unused-function -fPIC -shared $(INCLUDES) eval/style_check.c -lm -o $@

# One program per style: all of styles/ together exceeds the VM's 32-term limit.
style-check: $(OUT)/libstyle_check.so
	set -e; for style in $(STYLE_FILES); do \
	  native/artifacts/rl/venv/bin/python eval/style_check.py --lib $(abspath $<) --games $(STYLE_CHECK_GAMES) \
	    --json $(abspath $(OUT))/style-check-$$(basename $$style .toml).json $$style eval/style_check.toml; done

# Human/bot rates of every style over the replay dumps (the report's table).
style-score:
	for style in $(STYLE_FILES); do native/artifacts/rl/venv/bin/python eval/style_score.py $$style \
	  --json $(abspath $(OUT))/style-score-$$(basename $$style .toml).json || exit 1; done

# A league member's (or checkpoint's) style rates in its own candidate games
# (league/style_rates.py, docs/STYLES.md section 4), and the proof that measuring
# changes no game and agrees with eval/style_score.py on every decision
# (eval/style_rates_check.py). STYLE_RATES_CONDITION picks the ratings that choose
# the band when the tree's sim has no games of its own yet.
STYLE_RATES_TARGET ?= l1-main-8000m
STYLE_RATES_GAMES ?= 480
STYLE_RATES_CONDITION ?=
style-rates: $(OUT)/libbopl_research.so
	native/artifacts/rl/venv/bin/python league style-rates $(STYLE_RATES_TARGET) --styles $(STYLE_FILES) \
	  --games $(STYLE_RATES_GAMES) $(if $(STYLE_RATES_CONDITION),--condition $(STYLE_RATES_CONDITION)) \
	  --json $(abspath $(OUT))/style-rates/$(notdir $(STYLE_RATES_TARGET)).json

style-rates-check: $(OUT)/libbopl_research.so
	native/artifacts/rl/venv/bin/python eval/style_rates_check.py --target $(STYLE_RATES_TARGET) --members 24 \
	  $(if $(STYLE_RATES_CONDITION),--condition $(STYLE_RATES_CONDITION)) --styles $(STYLE_FILES) \
	  --json $(abspath $(OUT))/style-rates/check-$(notdir $(STYLE_RATES_TARGET)).json

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
# PYTHONPATH pins this checkout's binding: without it the venv's editable
# finder resolves pufferlib to whichever checkout installed it.
check-routing: $(RUST_ROLLOUT)
	PYTHONPATH=$(abspath native/artifacts/rl/dependencies/PufferLib-42f70d6932c30ac977736f861006809c50168ba9) \
	  native/artifacts/rl/venv/bin/python train/check_routing.py \
	  --game $(abspath $(REFERENCE)) --output $(abspath $(OUT))/check-routing-oracle
	PYTHONPATH=$(abspath native/artifacts/rl/dependencies/PufferLib-42f70d6932c30ac977736f861006809c50168ba9) \
	  native/artifacts/rl/venv/bin/python train/check_routing.py \
	  --game $(abspath $(RUST_ROLLOUT)) --output $(abspath $(OUT))/check-routing-port

# train.py --loadout-learner-first: on a frozen-bank vector the learner seat
# always holds the first half of its lineup row, on both sides, and without
# the flag it does not (train/check_learner_first.py). Needs the binding
# rebuilt after C changes (train/build.py --device cuda --binding-only).
check-learner-first:
	PYTHONPATH=$(abspath native/artifacts/rl/dependencies/PufferLib-42f70d6932c30ac977736f861006809c50168ba9) \
	  native/artifacts/rl/venv/bin/python train/check_learner_first.py

# Kit pinning (train.py --pinned-opponents, docs/LEAGUE.md "Kit pinning"): a
# pinned bank's frozen seat always holds one of its member's league.json kits,
# on both sides and across a mid-run swap; a scripted bank's bot holds its own
# kit and never a space map; without the flag, or with it and no rows, every
# match is dealt as before. Rust backend, both reset paths.
check-bank-kits:
	BOPL_RL_BACKEND=rust PYTHONPATH=$(abspath native/artifacts/rl/dependencies/PufferLib-42f70d6932c30ac977736f861006809c50168ba9) \
	  native/artifacts/rl/venv/bin/python train/check_bank_kits.py

$(OUT)/check_reward: eval/check_reward.c train/bopl_puffer.h train/bopl_style.h train/bopl_obs_v4.h train/bopl_rays_v4.h train/bopl_obs_pack.h train/bopl_layout_v4.h train/bopl_bots.h train/bopl_layout.h train/bopl_aim.h $(ABI)/bopl_rl.c $(ABI)/bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -std=gnu11 -Wno-unused-function $(INCLUDES) eval/check_reward.c $(ABI)/bopl_rl.c -ldl -lm -lpthread -o $@
