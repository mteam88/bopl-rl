/* Sudden-death A/B: print authored platform positions around activation.
 *
 * cc -O2 -I. -o ../target/release/sudden_death_ab \
 *   research/rust_ab/sudden_death_ab.c bopl_rl.c -ldl -lm
 */
#include "bopl_rl.h"
#include <stdio.h>
#include <stdlib.h>

static double f(int64_t raw) { return (double)raw / 4294967296.0; }

static const BoplRlEntity *platform(const BoplRlState *state, unsigned ordinal) {
    for (uint32_t i = 0; i < state->header.entity_count; i++) {
        const BoplRlEntity *entity = &state->entities[i];
        if (entity->kind == 2 && (entity->flags & 1)) {
            if (ordinal-- == 0) return entity;
        }
    }
    return NULL;
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    char error[512] = {0};
    BoplRl *world[2];
    for (int b = 0; b < 2; b++) {
        world[b] = bopl_rl_open(argv[b + 1],
            "/proc/self/cwd/native/fixtures/movement.rep",
            "/proc/self/cwd/native/target/sudden_death_ab.log",
            120000, error, sizeof error);
        if (!world[b]) { fprintf(stderr, "%s\n", error); return 1; }
    }
    BoplRlConfig config;
    bopl_rl_default_config(&config, 2);
    config.seed = 7; config.map = 0; config.stocks = 100;
    config.warmup_ticks = 240; config.max_episode_ticks = 9000;
    for (int p = 0; p < 2; p++) {
        config.player[p].abilities[0] = 14;
        config.player[p].abilities[1] = 2;
        config.player[p].abilities[2] = 4;
    }
    for (int b = 0; b < 2; b++)
        if (bopl_rl_reset(world[b], &config) != BOPL_RL_OK) return 1;

    puts("tick,ordinal,net_x,net_y,rust_x,rust_y");
    BoplRlAction actions[2] = {{0}};
    while (bopl_rl_state(world[0])->header.tick < 7270) {
        for (int b = 0; b < 2; b++)
            if (bopl_rl_step(world[b], actions, 2, 1) != BOPL_RL_OK) return 1;
        const BoplRlState *n = bopl_rl_state(world[0]);
        const BoplRlState *r = bopl_rl_state(world[1]);
        if (n->header.tick < 7180) continue;
        for (unsigned i = 0; i < 3; i++) {
            const BoplRlEntity *np = platform(n, i), *rp = platform(r, i);
            if (!np || !rp) continue;
            printf("%u,%u,%.12f,%.12f,%.12f,%.12f\n", n->header.tick, i,
                   f(np->x), f(np->y), f(rp->x), f(rp->y));
        }
    }
    bopl_rl_close(world[0]);
    bopl_rl_close(world[1]);
    return 0;
}
