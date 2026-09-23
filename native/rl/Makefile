CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Werror
OUT ?= ../artifacts/rl

all: $(OUT)/libbopl_rl.so $(OUT)/benchmark $(OUT)/coverage $(OUT)/record $(OUT)/reset_check $(OUT)/sampled_check $(OUT)/reset_profile $(OUT)/frame_profile

# Standalone evaluation/extraction binding; does not require rebuilding CUDA.
research: $(OUT)/libbopl_research.so

map-ab: $(OUT)/map_ab

# The C# NativeAOT oracle. `publish/` is the deployable link; the sibling
# `native/` directory holds the intermediate object of whichever build ran
# last and can be months stale (and on an older bopl_rl.h ABI), so point at
# `publish/`.
REFERENCE ?= ../../sim/Native/bin/Release/net8.0/linux-x64/publish/BoplSimNative.so
RUST_ROLLOUT ?= ../target/release/libbopl_sim.so
# The default `Fix` is an IEEE double behind the Q32.32 API (native/src/fixed.rs:1-30),
# so the default build cannot be compared with the oracle below ~1e-6: every
# multiply and divide rounds differently. Differential work needs the bit-exact
# arithmetic:  cd native && cargo build --release --features q32 --target-dir target-q32
RUST_Q32 ?= ../target-q32/release/libbopl_sim.so

# Exhaustive authored-map gate: all 124 maps, both 2- and 6-player layouts,
# 3000 ticks per live case, with discrete player/slime state required exact.
map-sweep-3000: $(OUT)/map_ab
	MAP_AB_STRICT_ACTORS=1 $(abspath $(OUT)/map_ab) $(abspath $(REFERENCE)) $(abspath $(RUST_ROLLOUT)) 3000

$(OUT)/map_ab: research/rust_ab/map_ab.c bopl_rl.c bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -I. research/rust_ab/map_ab.c bopl_rl.c -ldl -lm -o $@

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
# `RUST=$(RUST_ROLLOUT)` compares the shipping f64 build instead, which only
# makes sense with a loose REPLAY_AB_TOL.
RUST ?= $(RUST_Q32)
replay-sweep: $(OUT)/replay_ab
	REPLAY_AB_TOL=$${REPLAY_AB_TOL-1} $(abspath $(OUT)/replay_ab) $(abspath $(REFERENCE)) $(abspath $(RUST)) $(abspath $(GAMES))

# The same sweep restricted to what a trained policy can actually read
# (bopl_puffer.h:316): the ranking that answers whether a divergence could
# have changed a policy's behaviour.
replay-visible: $(OUT)/replay_ab
	REPLAY_AB_VISIBLE=1 REPLAY_AB_TOL=$${REPLAY_AB_TOL-1} $(abspath $(OUT)/replay_ab) $(abspath $(REFERENCE)) $(abspath $(RUST)) $(abspath $(GAMES))

# Self check: the reference replayed against itself must be bit-identical.
replay-selfcheck: $(OUT)/replay_ab
	REPLAY_AB_TOL=0 $(abspath $(OUT)/replay_ab) $(abspath $(REFERENCE)) $(abspath $(REFERENCE)) $(abspath $(GAMES))
	REPLAY_AB_TOL=0 $(abspath $(OUT)/replay_ab) $(abspath $(RUST)) $(abspath $(RUST)) $(abspath $(GAMES))
# The stronger half of the self check: the recorded inputs, replayed in the
# backend that produced the recording, must reproduce the saved <game>.json.
# LIBRARY must therefore be the build the corpus was recorded with.
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

.PHONY: all research map-ab replay-ab replay-sweep replay-visible replay-selfcheck replay-roundtrip writer-bench agent agent-mac
