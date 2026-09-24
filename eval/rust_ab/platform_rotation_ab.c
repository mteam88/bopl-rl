/* Spawned-platform rotation trace. Opens any number of ABI-compatible
 * backends and prints the newest live rounded rectangle after a Platform cast.
 *
 * cc -O2 -I. -o ../target/release/platform_rotation_ab \
 *   research/rust_ab/platform_rotation_ab.c bopl_rl.c -ldl -lm
 *
 * Usage: platform_rotation_ab <map> <angle> <release_tick> <end_tick>
 *        <label=library> [<label=library> ...]
 */
#include "bopl_rl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_BACKENDS 8

static double f(int64_t raw) { return (double)raw / 4294967296.0; }

static const BoplRlEntity *spawned_platform(const BoplRlState *state,
                                             uint32_t initial_entities) {
    const BoplRlEntity *result = NULL;
    for (uint32_t i = initial_entities; i < state->header.entity_count; i++) {
        const BoplRlEntity *entity = &state->entities[i];
        if (entity->kind == 2 && (entity->flags & 1)) result = entity;
    }
    return result;
}

int main(int argc, char **argv) {
    if (argc < 6 || argc - 5 > MAX_BACKENDS) {
        fprintf(stderr, "usage: %s <map> <angle> <release_tick> <end_tick> "
                        "<label=library> [<label=library> ...]\n", argv[0]);
        return 2;
    }
    const unsigned map = (unsigned)strtoul(argv[1], NULL, 10);
    const unsigned angle = (unsigned)strtoul(argv[2], NULL, 10);
    const unsigned release_tick = (unsigned)strtoul(argv[3], NULL, 10);
    const unsigned end_tick = (unsigned)strtoul(argv[4], NULL, 10);
    const char *jump_text = getenv("PLATFORM_ROTATION_JUMP_TICK");
    const unsigned jump_tick = jump_text ? (unsigned)strtoul(jump_text, NULL, 10) : 0;
    const int count = argc - 5;
    BoplRl *world[MAX_BACKENDS] = {0};
    uint32_t initial[MAX_BACKENDS] = {0};
    char *labels[MAX_BACKENDS] = {0};
    char error[512] = {0};

    for (int i = 0; i < count; i++) {
        char *separator = strchr(argv[i + 5], '=');
        if (!separator) {
            fprintf(stderr, "backend must be label=library: %s\n", argv[i + 5]);
            return 2;
        }
        *separator = '\0';
        labels[i] = argv[i + 5];
        world[i] = bopl_rl_open(separator + 1,
            "/proc/self/cwd/native/fixtures/movement.rep",
            "/proc/self/cwd/native/target/platform_rotation_ab.log",
            120000, error, sizeof error);
        if (!world[i]) {
            fprintf(stderr, "open %s: %s\n", labels[i], error);
            return 1;
        }
    }

    BoplRlConfig config;
    bopl_rl_default_config(&config, 2);
    config.seed = 7;
    config.map = map;
    config.stocks = 10;
    config.warmup_ticks = 240;
    config.max_episode_ticks = 7200;
    for (int player = 0; player < 2; player++) {
        config.player[player].abilities[0] = 14;
        config.player[player].abilities[1] = 2;
        config.player[player].abilities[2] = 4;
    }
    for (int i = 0; i < count; i++) {
        if (bopl_rl_reset(world[i], &config) != BOPL_RL_OK) {
            fprintf(stderr, "reset %s: %s\n", labels[i], bopl_rl_error(world[i]));
            return 1;
        }
        initial[i] = bopl_rl_state(world[i])->header.entity_count;
    }

    puts("backend,map,angle,release_tick,tick,orientation,angular_velocity,x,y,width,height");
    BoplRlAction actions[2] = {{0}};
    actions[0].angle = (uint8_t)angle;
    for (unsigned tick = 1; tick <= end_tick; tick++) {
        actions[0].buttons = (tick == jump_tick ? 1 : 0)
            | (tick >= 276 && tick < release_tick ? 2 : 0);
        for (int i = 0; i < count; i++) {
            int status = bopl_rl_step(world[i], actions, 2, 1);
            if (status != BOPL_RL_OK) {
                fprintf(stderr, "step %s tick %u status %d: %s\n",
                        labels[i], tick, status, bopl_rl_error(world[i]));
                return 1;
            }
            const BoplRlEntity *entity = spawned_platform(
                bopl_rl_state(world[i]), initial[i]);
            if (entity) {
                printf("%s,%u,%u,%u,%u,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f\n",
                       labels[i], map, angle, release_tick, tick,
                       f(entity->rotation), f(entity->angular_velocity),
                       f(entity->x), f(entity->y),
                       hypot(f(entity->right_x), f(entity->right_y)),
                       hypot(f(entity->up_x), f(entity->up_y)));
            }
        }
    }
    for (int i = 0; i < count; i++) bopl_rl_close(world[i]);
    return 0;
}
