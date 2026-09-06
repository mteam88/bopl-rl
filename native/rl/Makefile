CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Werror
OUT ?= ../artifacts/rl

all: $(OUT)/libbopl_rl.so $(OUT)/benchmark $(OUT)/coverage $(OUT)/record $(OUT)/reset_check $(OUT)/sampled_check $(OUT)/reset_profile $(OUT)/frame_profile

# Standalone evaluation/extraction binding; does not require rebuilding CUDA.
research: $(OUT)/libbopl_research.so

$(OUT)/libbopl_research.so: puffer/eval_binding.c puffer/bopl_puffer.h bopl_rl.c bopl_rl.h
	mkdir -p $(OUT)
	$(CC) $(CFLAGS) -Wno-unused-function -fPIC -shared -I. puffer/eval_binding.c bopl_rl.c -lm -o $@

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

.PHONY: all research
