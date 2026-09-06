CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Werror
OUT ?= ../artifacts/rl

all: $(OUT)/libbopl_rl.so $(OUT)/benchmark $(OUT)/coverage $(OUT)/record $(OUT)/reset_check

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

.PHONY: all
