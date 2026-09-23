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
bounds-ability-ab: $(OUT)/bounds_ability_ab
	$(abspath $(OUT)/bounds_ability_ab) $(abspath $(REFERENCE)) $(abspath $(RUST_ROLLOUT))

$(OUT)/bounds_ability_ab: research/rust_ab/bounds_ability_ab.c bopl_rl.c bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -I. research/rust_ab/bounds_ability_ab.c bopl_rl.c -ldl -lm -o $@

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

.PHONY: all research map-ab bounds-ability-ab writer-bench agent agent-mac
