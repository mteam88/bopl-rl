/* Smoke test for the in-process port of the RL ABI (sim/Native ->
 * BoplSimNative.so). dlopens the library, opens an in-process world,
 * resets with the default config for 2 players, steps 240 times with zero
 * actions, and prints ticks/s plus an FNV-1a checksum over every post-step
 * state (header, players, entities, slimes, telemetry). The same sequence
 * through the socket path (bopl_rl.c + Unity player, or bopl_rl.c + a future
 * socket-mode sim/Host) must print the same checksum.
 *
 *   cc -O2 -o native_check native_check.c -ldl
 *   ./native_check <path-to-BoplSimNative.so>
 *
 * The .so locates the export by walking up from its own directory
 * (native/reference-export); override with BOPL_SIM_EXPORT=<repo root>.
 */
#define _GNU_SOURCE
#include "bopl_rl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef uint32_t (*abi_version_fn)(void);
typedef uint32_t (*world_capacity_fn)(void);
typedef void (*default_config_fn)(BoplRlConfig *, uint32_t);
typedef BoplRl *(*open_fn)(const char *, const char *, const char *, uint32_t, char *, size_t);
typedef int (*reset_fn)(BoplRl *, const BoplRlConfig *);
typedef int (*step_fn)(BoplRl *, const BoplRlAction *, size_t, uint32_t);
typedef const BoplRlState *(*state_fn)(const BoplRl *);
typedef const char *(*error_fn)(const BoplRl *);
typedef uint32_t (*count_fn)(const BoplRl *);
typedef void (*close_fn)(BoplRl *);

static uint64_t fnv1a(uint64_t hash, const void *data, size_t size)
{
    const unsigned char *bytes = data;
    for (size_t i = 0; i < size; i++) { hash ^= bytes[i]; hash *= 1099511628211ULL; }
    return hash;
}

static uint64_t hash_state(uint64_t hash, const BoplRlState *state)
{
    const BoplRlHeader *h = &state->header;
    hash = fnv1a(hash, h, sizeof(*h));
    hash = fnv1a(hash, state->players, (size_t)h->player_count * sizeof(BoplRlPlayer));
    hash = fnv1a(hash, state->entities, (size_t)h->entity_count * sizeof(BoplRlEntity));
    hash = fnv1a(hash, state->slimes, (size_t)h->slime_count * sizeof(BoplRlSlime));
    hash = fnv1a(hash, state->telemetry, (size_t)h->player_count * sizeof(BoplRlTelemetry));
    return hash;
}

static int64_t now_ms(void)
{
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (int64_t)value.tv_sec * 1000 + value.tv_nsec / 1000000;
}

int main(int argc, char **argv)
{
    if (argc != 2) { fprintf(stderr, "usage: %s <path-to-BoplSimNative.so>\n", argv[0]); return 2; }
    void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
#define RESOLVE(type, name) type name = (type)dlsym(library, "bopl_rl_" #name); \
    if (!name) { fprintf(stderr, "dlsym bopl_rl_%s: %s\n", #name, dlerror()); return 1; }
    RESOLVE(abi_version_fn, abi_version)
    RESOLVE(world_capacity_fn, world_capacity)
    RESOLVE(default_config_fn, default_config)
    RESOLVE(open_fn, open)
    RESOLVE(reset_fn, reset)
    RESOLVE(step_fn, step)
    RESOLVE(state_fn, state)
    RESOLVE(error_fn, error)
    RESOLVE(count_fn, map_count)
    RESOLVE(count_fn, ability_count)
    RESOLVE(close_fn, close)
#undef RESOLVE

    if (abi_version() != 3) { fprintf(stderr, "unexpected ABI version %u\n", abi_version()); return 1; }

    /* Paths satisfy the shared argument contract; the in-process library only
     * uses the log path (its -logFile equivalent). */
    char message[512] = {0};
    BoplRl *world = open("/usr/bin/true", "/dev/null", "/tmp/bopl-native-check.log", 60000, message, sizeof(message));
    if (!world) { fprintf(stderr, "open failed: %s\n", message); return 1; }
    printf("opened: maps=%u abilities=%u\n", map_count(world), ability_count(world));

    /* NativeAOT supports multiple independent worlds up to its advertised
     * capacity. Exercise a second handle so this smoke test also catches a
     * stale singleton implementation or a broken handle registry. */
    uint32_t capacity_count = world_capacity();
    BoplRl *second = open("/usr/bin/true", "/dev/null", "/tmp/bopl-native-check2.log", 60000,
                          message, sizeof(message));
    if (capacity_count > 1 && !second) {
        fprintf(stderr, "second open failed with capacity %u: %s\n", capacity_count, message); return 1;
    }
    if (capacity_count <= 1 && second) {
        fprintf(stderr, "second open unexpectedly succeeded with capacity %u\n", capacity_count); return 1;
    }
    if (second) close(second);
    printf("world capacity=%u; second handle %s\n", capacity_count,
           second ? "opened and closed" : "correctly rejected");

    BoplRlConfig config;
    default_config(&config, 2);
    int64_t start = now_ms();
    int result = reset(world, &config);
    int64_t reset_elapsed = now_ms() - start;
    if (result != BOPL_RL_OK) { fprintf(stderr, "reset failed (%d): %s\n", result, error(world)); return 1; }
    printf("reset (%u warmup ticks): %lld ms\n", config.warmup_ticks, (long long)reset_elapsed);

    BoplRlAction actions[2] = {{0, 0, 0}, {0, 0, 0}};
    uint64_t checksum = 14695981039346656037ULL;
    int steps = 0;
    start = now_ms();
    for (int i = 0; i < 240; i++) {
        result = step(world, actions, 2, 1);
        if (result == BOPL_RL_ENDED) break;
        if (result != BOPL_RL_OK) { fprintf(stderr, "step %d failed (%d): %s\n", i, result, error(world)); return 1; }
        checksum = hash_state(checksum, state(world));
        steps++;
    }
    int64_t elapsed = now_ms() - start;
    const BoplRlState *final = state(world);
    printf("steps=%d ticks=%u elapsed=%lldms ticks/s=%.0f\n", steps, final->header.tick,
           (long long)elapsed, elapsed > 0 ? steps * 1000.0 / elapsed : 0.0);
    printf("checksum=%016llx\n", (unsigned long long)checksum);
    close(world);
    dlclose(library);
    return 0;
}
