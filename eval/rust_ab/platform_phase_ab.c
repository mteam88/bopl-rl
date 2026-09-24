/* Platform ability phase A/B: one row per tick around button registration,
 * anticipation callback, initial resize, and transformed-player following.
 *
 * cc -O2 -I sim/abi -o ../target/release/platform_phase_ab \
 *   eval/rust_ab/platform_phase_ab.c sim/abi/bopl_rl.c -ldl -lm
 */
#include "bopl_rl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double f(int64_t raw) { return (double)raw / 4294967296.0; }

typedef struct {
    double px, py, slime_x, slime_y, sx, sy, radius, width, height, inverse_mass;
    unsigned player_flags, slime_flags, ability_flags, circle_flags, spawned;
} Sample;

static Sample sample(const BoplRlState *state, uint32_t initial_entities) {
    Sample out = {
        .px = f(state->players[0].x), .py = f(state->players[0].y),
        .player_flags = state->players[0].flags,
        .sx = NAN, .sy = NAN, .radius = NAN, .width = NAN, .height = NAN,
        .inverse_mass = NAN,
    };
    for (uint32_t i = 0; i < state->header.slime_count; i++) {
        if (state->slimes[i].owner != 1) continue;
        out.slime_flags = state->slimes[i].flags;
        out.ability_flags = state->slimes[i].ability_flags[0];
        out.slime_x = f(state->slimes[i].x);
        out.slime_y = f(state->slimes[i].y);
    }
    for (uint32_t i = 0; i < state->header.entity_count; i++) {
        const BoplRlEntity *e = &state->entities[i];
        if (i < initial_entities && e->kind == 0 && out.circle_flags == 0)
            out.circle_flags = e->flags;
        if (i < initial_entities || e->kind != 2 || !(e->flags & 1)) continue;
        out.spawned++;
        out.sx = f(e->x); out.sy = f(e->y); out.radius = f(e->radius);
        out.width = hypot(f(e->right_x), f(e->right_y));
        out.height = hypot(f(e->up_x), f(e->up_y));
        out.inverse_mass = f(e->inverse_mass);
    }
    return out;
}

static void row(unsigned tick, BoplRl **world, const uint32_t *initial) {
    Sample n = sample(bopl_rl_state(world[0]), initial[0]);
    Sample r = sample(bopl_rl_state(world[1]), initial[1]);
    printf("%u,%u,%u,%u,%u,%u,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,"
           "%u,%u,%u,%u,%u,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f\n",
           tick, n.player_flags, n.slime_flags, n.ability_flags, n.circle_flags, n.spawned,
           n.px, n.py, n.slime_x, n.slime_y, n.sx, n.sy, n.radius, n.width, n.height, n.inverse_mass,
           r.player_flags, r.slime_flags, r.ability_flags, r.circle_flags, r.spawned,
           r.px, r.py, r.slime_x, r.slime_y, r.sx, r.sy, r.radius, r.width, r.height, r.inverse_mass);
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    char error[512] = {0};
    BoplRl *world[2];
    for (int b = 0; b < 2; b++) {
        world[b] = bopl_rl_open(argv[b + 1],
            "/proc/self/cwd/native/fixtures/movement.rep",
            "/proc/self/cwd/native/target/platform_phase_ab.log",
            120000, error, sizeof error);
        if (!world[b]) { fprintf(stderr, "%s\n", error); return 1; }
    }
    BoplRlConfig c;
    bopl_rl_default_config(&c, 2);
    c.seed = 7; c.map = 0; c.stocks = 0; c.warmup_ticks = 240;
    c.max_episode_ticks = 7200;
    for (int p = 0; p < 2; p++) {
        c.player[p].abilities[0] = 14;
        c.player[p].abilities[1] = 2;
        c.player[p].abilities[2] = 4;
    }
    uint32_t initial[2];
    for (int b = 0; b < 2; b++) {
        if (bopl_rl_reset(world[b], &c) != BOPL_RL_OK) return 1;
        initial[b] = bopl_rl_state(world[b])->header.entity_count;
    }
    puts("tick,net_player_flags,net_slime_flags,net_ability_flags,net_circle_flags,net_spawned,"
         "net_px,net_py,net_slime_x,net_slime_y,net_sx,net_sy,net_radius,net_width,net_height,net_inverse_mass,"
         "rust_player_flags,rust_slime_flags,rust_ability_flags,rust_circle_flags,rust_spawned,"
         "rust_px,rust_py,rust_slime_x,rust_slime_y,rust_sx,rust_sy,rust_radius,rust_width,rust_height,rust_inverse_mass");
    BoplRlAction actions[2] = {{0}};
    actions[0].angle = 65;
    for (unsigned tick = 1; tick <= 360; tick++) {
        actions[0].buttons = tick >= 276 && tick <= 332 ? 2 : 0;
        for (int b = 0; b < 2; b++)
            if (bopl_rl_step(world[b], actions, 2, 1) != BOPL_RL_OK) return 1;
        if (tick >= 270) row(tick, world, initial);
    }
    bopl_rl_close(world[0]); bopl_rl_close(world[1]);
    return 0;
}
