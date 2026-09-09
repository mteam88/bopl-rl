/* Gate 0 checks for the in-process backend (native/rl/SIM_OVERHAUL.md, 1.3
 * and 1.4). Runs one fixed input sequence through the bopl_rl.h ABI and prints
 * FNV-1a checksums over every post-step state, via either backend:
 *
 *   backend_check socket <Host-or-player binary> <bootstrap .rep>
 *   backend_check inproc <BoplSimNative.so> [copies]
 *
 * The socket backend is bopl_rl.c linked in (worker process + unix socket),
 * the in-process backend dlopens the library. With copies > 1, the library is
 * copied to distinct files (glibc de-duplicates dlopen by device/inode) and
 * each copy runs the sequence on its own thread at the same time: several
 * NativeAOT runtimes in one process, which is what one-copy-per-env needs.
 * Every copy must print the checksums of a single copy, which must equal the
 * socket backend's.
 *
 * Sequence: (a) default 2-player config, 240 zero-action ticks (the
 * native_check number); (b) map 1, seed 7, 3 stocks, up to 1500 decisions of 4 ticks
 * with a deterministic pseudo-random input schedule (held stick, occasional
 * jump/ability), stopping early if the episode ends; then (c) = (a) again to check that a
 * dirty reset lands on the same state.
 *
 *   cc -O2 -pthread -o backend_check backend_check.c bopl_rl.c -ldl
 */
#define _GNU_SOURCE
#include "bopl_rl.h"
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    uint32_t (*abi_version)(void);
    void (*default_config)(BoplRlConfig *, uint32_t);
    BoplRl *(*open)(const char *, const char *, const char *, uint32_t, char *, size_t);
    int (*reset)(BoplRl *, const BoplRlConfig *);
    int (*step)(BoplRl *, const BoplRlAction *, size_t, uint32_t);
    const BoplRlState *(*state)(const BoplRl *);
    const char *(*error)(const BoplRl *);
    uint32_t (*map_count)(const BoplRl *);
    void (*close)(BoplRl *);
} Backend;

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

static double now_ms(void)
{
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec * 1000.0 + value.tv_nsec / 1e6;
}

typedef struct {
    const Backend *api;
    BoplRl *world;
    const char *label;
    uint64_t checksum[3];
    int steps[3];
    double reset_ms[3], step_us[3];
    int failed;
} Run;

static int fail(Run *run, const char *what, int result)
{
    fprintf(stderr, "[%s] %s failed (%d): %s\n", run->label, what, result,
            run->world ? run->api->error(run->world) : "");
    run->failed = 1;
    return -1;
}

/* Phase p of the sequence; returns 0 on success. */
static int run_phase(Run *run, int p)
{
    const Backend *api = run->api;
    BoplRlConfig config;
    api->default_config(&config, 2);
    int decisions = 240, repeat = 1;
    uint32_t rng = 0;
    if (p == 1) {
        config.map = 1 % api->map_count(run->world);
        config.seed = 7;
        config.stocks = 3; /* respawns, so the schedule gets a long game */
        decisions = 1500; repeat = 4; rng = 0x9E3779B9u;
    }
    double start = now_ms();
    int result = api->reset(run->world, &config);
    run->reset_ms[p] = now_ms() - start;
    if (result != BOPL_RL_OK) return fail(run, "reset", result);

    uint64_t checksum = 14695981039346656037ULL;
    uint8_t held[2] = {0, 0};
    int steps = 0;
    start = now_ms();
    for (int i = 0; i < decisions; i++) {
        BoplRlAction actions[2] = {{0, 0, 0}, {0, 0, 0}};
        if (rng) {
            for (int k = 0; k < 2; k++) {
                rng = rng * 1664525u + 1013904223u;
                /* Hold a random stick direction for 8 decisions, jump 1 in 8,
                 * an ability 1 in 32 (fully random buttons die within 84 ticks). */
                if ((i & 7) == 0) held[k] = (uint8_t)(rng >> 24);
                actions[k].angle = held[k];
                if (((rng >> 4) & 7) == 0) actions[k].buttons |= 1;
                if (((rng >> 16) & 31) == 0) actions[k].buttons |= (uint16_t)(2u << ((rng >> 21) % 3));
            }
        }
        result = api->step(run->world, actions, 2, (uint32_t)repeat);
        if (result == BOPL_RL_ENDED) break;
        if (result != BOPL_RL_OK) return fail(run, "step", result);
        checksum = hash_state(checksum, api->state(run->world));
        steps++;
    }
    run->step_us[p] = steps ? (now_ms() - start) * 1000.0 / steps : 0;
    run->checksum[p] = checksum;
    run->steps[p] = steps;
    return 0;
}

static void *run_sequence(void *arg)
{
    Run *run = arg;
    for (int p = 0; p < 3 && !run->failed; p++) run_phase(run, p);
    return NULL;
}

static void print_run(const Run *run)
{
    for (int p = 0; p < 3; p++)
        printf("[%s] phase %c: steps=%d reset=%.1fms step=%.1fus checksum=%016llx\n", run->label, 'a' + p,
               run->steps[p], run->reset_ms[p], run->step_us[p], (unsigned long long)run->checksum[p]);
}

static int copy_file(const char *from, const char *to)
{
    int in = open(from, O_RDONLY), out = open(to, O_WRONLY | O_CREAT | O_TRUNC, 0755);
    if (in < 0 || out < 0) return -1;
    char buffer[1 << 16];
    ssize_t n;
    while ((n = read(in, buffer, sizeof(buffer))) > 0)
        if (write(out, buffer, (size_t)n) != n) return -1;
    close(in); close(out);
    return n < 0 ? -1 : 0;
}

static int load_inproc(Backend *api, const char *path)
{
    void *library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!library) { fprintf(stderr, "dlopen %s: %s\n", path, dlerror()); return -1; }
#define RESOLVE(field, name) api->field = (__typeof__(api->field))dlsym(library, "bopl_rl_" #name); \
    if (!api->field) { fprintf(stderr, "dlsym bopl_rl_%s: %s\n", #name, dlerror()); return -1; }
    RESOLVE(abi_version, abi_version) RESOLVE(default_config, default_config) RESOLVE(open, open)
    RESOLVE(reset, reset) RESOLVE(step, step) RESOLVE(state, state) RESOLVE(error, error)
    RESOLVE(map_count, map_count) RESOLVE(close, close)
#undef RESOLVE
    if (api->abi_version() != 2) { fprintf(stderr, "%s: ABI version %u\n", path, api->abi_version()); return -1; }
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s socket <binary> <bootstrap.rep> | inproc <BoplSimNative.so> [copies]\n", argv[0]); return 2; }
    int copies = 1;
    Backend *apis;
    char **labels, **logs;
    char tmpdir[64] = "/tmp/bopl-backend-check-XXXXXX";
    if (!mkdtemp(tmpdir)) { perror("mkdtemp"); return 1; }

    if (!strcmp(argv[1], "socket")) {
        if (argc != 4) { fprintf(stderr, "socket needs <binary> <bootstrap.rep>\n"); return 2; }
        apis = calloc(1, sizeof(Backend));
        *apis = (Backend){bopl_rl_abi_version, bopl_rl_default_config, bopl_rl_open, bopl_rl_reset, bopl_rl_step,
                          bopl_rl_state, bopl_rl_error, bopl_rl_map_count, bopl_rl_close};
    } else if (!strcmp(argv[1], "inproc")) {
        if (argc == 4) copies = atoi(argv[3]);
        if (copies < 1 || copies > 64) { fprintf(stderr, "copies must be 1..64\n"); return 2; }
        apis = calloc((size_t)copies, sizeof(Backend));
        for (int k = 0; k < copies; k++) {
            char path[256];
            snprintf(path, sizeof(path), "%s/BoplSimNative.%d.so", tmpdir, k);
            if (copy_file(argv[2], path)) { perror("copy .so"); return 1; }
            if (load_inproc(&apis[k], path)) return 1;
        }
    } else { fprintf(stderr, "unknown backend %s\n", argv[1]); return 2; }

    labels = calloc((size_t)copies, sizeof(char *));
    logs = calloc((size_t)copies, sizeof(char *));
    Run *runs = calloc((size_t)copies, sizeof(Run));
    for (int k = 0; k < copies; k++) {
        labels[k] = malloc(32); logs[k] = malloc(256);
        snprintf(labels[k], 32, "%s%d", argv[1], k);
        snprintf(logs[k], 256, "%s/%s.log", tmpdir, labels[k]);
        char message[512] = {0};
        const char *binary = argc == 4 && !strcmp(argv[1], "socket") ? argv[2] : "/usr/bin/true";
        const char *replay = !strcmp(argv[1], "socket") ? argv[3] : "/dev/null";
        runs[k].api = &apis[k]; runs[k].label = labels[k];
        double start = now_ms();
        runs[k].world = apis[k].open(binary, replay, logs[k], 60000, message, sizeof(message));
        if (!runs[k].world) { fprintf(stderr, "[%s] open failed: %s\n", labels[k], message); return 1; }
        printf("[%s] opened in %.0f ms (maps=%u)\n", labels[k], now_ms() - start, apis[k].map_count(runs[k].world));
    }

    pthread_t *threads = calloc((size_t)copies, sizeof(pthread_t));
    double start = now_ms();
    for (int k = 0; k < copies; k++) pthread_create(&threads[k], NULL, run_sequence, &runs[k]);
    for (int k = 0; k < copies; k++) pthread_join(threads[k], NULL);
    printf("all sequences: %.0f ms wall\n", now_ms() - start);

    int status = 0;
    for (int k = 0; k < copies; k++) {
        print_run(&runs[k]);
        if (runs[k].failed) status = 1;
        for (int p = 0; p < 3; p++) if (runs[k].checksum[p] != runs[0].checksum[p]) status = 1;
    }
    if (copies > 1) printf("copies agree: %s\n", status ? "NO" : "yes");
    if (runs[0].checksum[0] != runs[0].checksum[2]) { printf("dirty reset (c) != clean reset (a): NO\n"); status = 1; }
    else printf("dirty reset reproduces clean reset: yes\n");
    printf("summary %s a=%016llx b=%016llx c=%016llx\n", argv[1], (unsigned long long)runs[0].checksum[0],
           (unsigned long long)runs[0].checksum[1], (unsigned long long)runs[0].checksum[2]);
    for (int k = 0; k < copies; k++) apis[k].close(runs[k].world);
    if (status) fprintf(stderr, "logs kept in %s\n", tmpdir);
    return status;
}
