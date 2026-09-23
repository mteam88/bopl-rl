CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Werror
OUT ?= ../artifacts/rl

all: $(OUT)/libbopl_rl.so $(OUT)/benchmark $(OUT)/coverage $(OUT)/record $(OUT)/reset_check $(OUT)/sampled_check $(OUT)/reset_profile $(OUT)/frame_profile

# Standalone evaluation/extraction binding; does not require rebuilding CUDA.
research: $(OUT)/libbopl_research.so

map-ab: $(OUT)/map_ab

# `dotnet publish` refreshes publish/; the sibling native/ directory keeps whatever
# an earlier build left there, so pointing at it silently compares against a stale
# oracle - it held an ABI 2 library for nine days after ABI 3 landed.
REFERENCE ?= ../../sim/Native/bin/Release/net8.0/linux-x64/publish/BoplSimNative.so
RUST_ROLLOUT ?= ../target/release/libbopl_sim.so
# The default `cargo build --release` stores `Fix` as an f64 (native/src/fixed.rs)
# and is what training uses. It is NOT bit-comparable with the .NET oracle: its
# multiplications and divisions round to the nearest double instead of
# truncating to 2^-32, so every differential probe below sees an arithmetic
# delta of a few raw units at tick 0 and reports that instead of its own
# subject. Every A/B gate must therefore run the `q32` library, which is the
# reference's exact Q32.32 arithmetic. It is built into its own target
# directory so it does not fight the f64 build for `native/target`.
RUST_ROLLOUT_Q32 ?= ../target-q32/release/libbopl_sim.so

rust-q32:
	cargo build --release --features q32 --manifest-path ../Cargo.toml --target-dir ../target-q32

# Exhaustive authored-map gate: all 124 maps, both 2- and 6-player layouts,
# 3000 ticks per live case, with discrete player/slime state required exact.
map-sweep-3000: $(OUT)/map_ab rust-q32
	MAP_AB_STRICT_ACTORS=1 $(abspath $(OUT)/map_ab) $(abspath $(REFERENCE)) $(abspath $(RUST_ROLLOUT_Q32)) 3000

$(OUT)/map_ab: research/rust_ab/map_ab.c bopl_rl.c bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -I. research/rust_ab/map_ab.c bopl_rl.c -ldl -lm -o $@

# DestroyIfOutsideSceneBounds gate: walks a player off the edge of map 0 and
# engages each ability form at three depths relative to the kill plane, then
# requires the reference and the port to agree on the death tick, the death
# position, and the cause of death. Catches ability bodies that hover below
# the plane (GunTransform's slow-mo family) and abilities entered on the tick
# the slime is already out of bounds.
bounds-ability-ab: $(OUT)/bounds_ability_ab rust-q32
	$(abspath $(OUT)/bounds_ability_ab) $(abspath $(REFERENCE)) $(abspath $(RUST_ROLLOUT_Q32))

$(OUT)/bounds_ability_ab: research/rust_ab/bounds_ability_ab.c bopl_rl.c bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -I. research/rust_ab/bounds_ability_ab.c bopl_rl.c -ldl -lm -o $@

# Collider-rescale vs entry-pass boundary gate: a grounded player pressing a
# rescaling ability slot (Dash) while sudden death has carried it under the
# kill plane. The entry-pass check runs Move on a COPY of the body, and the
# copy does not re-sync the collider, so clone and real Move can disagree.
# Currently FAILS on map 2: the reference lets Dash claim the body and kills
# it as an ability body (cause Other, ungrounded 0.05 off the surface) while
# the port pre-empts and kills the ordinary slime (cause Accident).
scale-boundary-ab: $(OUT)/scale_boundary_ab rust-q32
	$(abspath $(OUT)/scale_boundary_ab) $(abspath $(REFERENCE)) $(abspath $(RUST_ROLLOUT_Q32))

$(OUT)/scale_boundary_ab: research/rust_ab/scale_boundary_ab.c bopl_rl.c bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -I. research/rust_ab/scale_boundary_ab.c bopl_rl.c -ldl -lm -o $@

# Sudden-death publication gate: the tick each backend first PUBLISHES
# header.sudden_death, on maps 0..4. GameSessionHandler raises the flag from
# inside the updater pass against a clock the tick has not advanced yet, so the
# port's post-increment session update is a pass ahead of it unless the
# published bit lags by one.
sudden-death-publication-ab: $(OUT)/sudden_death_publication_ab rust-q32
	$(abspath $(OUT)/sudden_death_publication_ab) $(abspath $(REFERENCE)) $(abspath $(RUST_ROLLOUT_Q32))

$(OUT)/sudden_death_publication_ab: research/rust_ab/sudden_death_publication_ab.c bopl_rl.c bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -I. research/rust_ab/sudden_death_publication_ab.c bopl_rl.c -ldl -lm -o $@

# Whole rust_ab differential suite, comparator included. Seventeen probes in
# research/rust_ab only *print* a per-backend trace and always exit 0, so
# checking exit status alone reports them as passing unconditionally; this
# target runs trace_compare.py over their output and fails on any difference.
# See research/rust_ab/README.md "Trace-only probes".
rust-ab-suite: rust-q32
	REFERENCE=$(abspath $(REFERENCE)) RUST=$(abspath $(RUST_ROLLOUT_Q32)) \
	  research/rust_ab/run_suite.sh $(PROBE)

# The comparator on its own, for a capture you already have:
#   make -C native/rl rust-ab-trace CAPTURE=/tmp/out.csv
rust-ab-trace:
	$(PYTHON) research/rust_ab/trace_compare.py $(CAPTURE)

# Real-game replay differential: re-steps saved evaluation games (the .rep
# recordings beside every evaluations/*.json) in both backends with the
# recorded per-tick inputs and reports the first divergence per game.
replay-ab: $(OUT)/replay_ab

$(OUT)/replay_ab: research/rust_ab/replay_ab.c bopl_rl.c bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -I. research/rust_ab/replay_ab.c bopl_rl.c -ldl -lm -o $@

# Whole-directory sweep over one study or sweep evaluation directory:
#   make -C native/rl replay-sweep GAMES=../artifacts/rl/study-grenade/evaluations
GAMES ?= ../artifacts/rl/study-grenade/evaluations
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
# prebuilt `../target/release` in a working checkout is not reliably that, and
# replaying a corpus in a later build fails this by design.
PYTHON ?= python3
LIBRARY ?= $(RUST_ROLLOUT)
replay-roundtrip: $(OUT)/replay_ab
	../artifacts/rl/venv/bin/python research/rust_ab/replay_ab_selfcheck.py \
	  --binary $(abspath $(OUT)/replay_ab) --library $(abspath $(LIBRARY)) $(GAMES)/*/*.rep

# Observation-writer microbenchmark + sim ticks/s on one core (THROUGHPUT_V4.md).
writer-bench: $(OUT)/writer_bench

$(OUT)/writer_bench: writer_bench.c puffer/bopl_puffer.h puffer/bopl_obs_v4.h puffer/bopl_layout_v4.h puffer/bopl_bots.h puffer/bopl_layout.h puffer/bopl_aim.h bopl_rl.c bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -std=gnu11 -Wno-unused-function -I. writer_bench.c bopl_rl.c -lm -lpthread -o $@

$(OUT)/libbopl_research.so: puffer/eval_binding.c puffer/bopl_puffer.h puffer/bopl_obs_v4.h puffer/bopl_layout_v4.h puffer/bopl_bots.h puffer/bopl_bots_v2.h puffer/bopl_bots_v2_play.h puffer/bopl_layout.h puffer/bopl_aim.h bopl_rl.c bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -Wno-unused-function -fPIC -shared -I. puffer/eval_binding.c bopl_rl.c -lm -o $@

# In-game agent shim (puffer/bopl_agent.h): the training decision loop fed by
# a running game, plus the CPU policy evaluator. Loaded by the Unity build
# (Assets/Scripts/Assembly-CSharp/RlAgent.cs) and research/agent_shim.py.
PUFFERLIB ?= $(OUT)/dependencies/PufferLib-42f70d6932c30ac977736f861006809c50168ba9/src
agent: $(OUT)/libbopl_agent.so

$(OUT)/libbopl_agent.so: puffer/bopl_agent.c puffer/bopl_agent.h puffer/policy_eval.c puffer/bopl_set_encoder.h puffer/bopl_v4_eval.h puffer/bopl_obs_v4.h puffer/bopl_puffer.h puffer/bopl_bots.h puffer/bopl_bots_v2.h puffer/bopl_bots_v2_play.h puffer/bopl_layout.h puffer/bopl_layout_v4.h puffer/bopl_aim.h bopl_rl.c bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -std=gnu11 -Wno-unused-function -fPIC -c -I. -Ipuffer puffer/bopl_agent.c -o $(OUT)/bopl_agent.o
	$(CC) $(CFLAGS) -fPIC -c bopl_rl.c -o $(OUT)/bopl_agent_rl.o
	$(CC) -O3 -std=gnu11 -fPIC -c -I$(PUFFERLIB) -Ipuffer puffer/policy_eval.c -o $(OUT)/bopl_agent_policy.o
	$(CC) -shared $(OUT)/bopl_agent.o $(OUT)/bopl_agent_rl.o $(OUT)/bopl_agent_policy.o -lm -lpthread -o $@
	rm -f $(OUT)/bopl_agent.o $(OUT)/bopl_agent_rl.o $(OUT)/bopl_agent_policy.o

# The same shim for the macOS player (arm64, run on the Mac: `make -C native/rl agent-mac`);
# Mono resolves DllImport("bopl_agent") to BoplBattle.app/Contents/Plugins/libbopl_agent.dylib.
agent-mac: $(OUT)/libbopl_agent.dylib

$(OUT)/libbopl_agent.dylib: puffer/bopl_agent.c puffer/bopl_agent.h puffer/policy_eval.c puffer/bopl_set_encoder.h puffer/bopl_v4_eval.h puffer/bopl_obs_v4.h puffer/bopl_puffer.h puffer/bopl_bots.h puffer/bopl_bots_v2.h puffer/bopl_bots_v2_play.h puffer/bopl_layout.h puffer/bopl_layout_v4.h puffer/bopl_aim.h bopl_rl.c bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -std=gnu11 -Wno-unused-function -fPIC -c -I. -Ipuffer puffer/bopl_agent.c -o $(OUT)/bopl_agent.o
	$(CC) $(CFLAGS) -fPIC -c bopl_rl.c -o $(OUT)/bopl_agent_rl.o
	$(CC) -O3 -std=gnu11 -fPIC -c -I$(PUFFERLIB) -Ipuffer puffer/policy_eval.c -o $(OUT)/bopl_agent_policy.o
	$(CC) -dynamiclib $(OUT)/bopl_agent.o $(OUT)/bopl_agent_rl.o $(OUT)/bopl_agent_policy.o -lm -lpthread -install_name @rpath/libbopl_agent.dylib -o $@
	rm -f $(OUT)/bopl_agent.o $(OUT)/bopl_agent_rl.o $(OUT)/bopl_agent_policy.o

$(OUT)/libbopl_rl.so: bopl_rl.c bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -fPIC -shared bopl_rl.c -o $@

$(OUT)/benchmark: benchmark.c bopl_rl.h $(OUT)/libbopl_rl.so
	$(CC) $(CFLAGS) benchmark.c -L$(OUT) -lbopl_rl -Wl,-rpath,'$$ORIGIN' -o $@

$(OUT)/coverage: coverage.c bopl_rl.h $(OUT)/libbopl_rl.so
	$(CC) $(CFLAGS) coverage.c -L$(OUT) -lbopl_rl -Wl,-rpath,'$$ORIGIN' -o $@

$(OUT)/record: record.c bopl_rl.h $(OUT)/libbopl_rl.so
	$(CC) $(CFLAGS) record.c -L$(OUT) -lbopl_rl -Wl,-rpath,'$$ORIGIN' -o $@

$(OUT)/reset_check: reset_check.c bopl_rl.h $(OUT)/libbopl_rl.so
	$(CC) $(CFLAGS) reset_check.c -L$(OUT) -lbopl_rl -Wl,-rpath,'$$ORIGIN' -o $@

$(OUT)/sampled_check: sampled_check.c bopl_rl.h $(OUT)/libbopl_rl.so
	$(CC) $(CFLAGS) sampled_check.c -L$(OUT) -lbopl_rl -Wl,-rpath,'$$ORIGIN' -o $@

$(OUT)/reset_profile: reset_profile.c bopl_rl.h $(OUT)/libbopl_rl.so
	$(CC) $(CFLAGS) -std=gnu11 reset_profile.c -L$(OUT) -lbopl_rl -Wl,-rpath,'$$ORIGIN' -o $@

$(OUT)/frame_profile: frame_profile.c bopl_rl.h $(OUT)/libbopl_rl.so
	$(CC) $(CFLAGS) -std=gnu11 frame_profile.c -L$(OUT) -lbopl_rl -Wl,-rpath,'$$ORIGIN' -o $@

.PHONY: all research map-ab bounds-ability-ab scale-boundary-ab sudden-death-publication-ab rust-ab-suite rust-ab-trace replay-ab replay-sweep replay-visible replay-selfcheck replay-roundtrip rust-q32 writer-bench agent agent-mac
