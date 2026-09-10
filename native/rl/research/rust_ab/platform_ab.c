/* Platform-body A/B: grounded transform overlapping terrain, then a cold
 * spawn falling onto an authored platform. One output row per simulation tick.
 *
 * cc -O2 -I. -o ../target/release/platform_ab \
 *   research/rust_ab/platform_ab.c bopl_rl.c -ldl -lm
 */
#include "bopl_rl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double f(int64_t raw) { return (double)raw / 4294967296.0; }

typedef struct {
    double px, py;
    double map_x, map_y, map_vx, map_vy;
    double spawn_x, spawn_y, spawn_vx, spawn_vy;
} Sample;

static Sample sample(const BoplRlState *state, uint32_t initial_entities) {
    Sample out = {
        .px = f(state->players[0].x), .py = f(state->players[0].y),
        .map_x = NAN, .map_y = NAN, .map_vx = NAN, .map_vy = NAN,
        .spawn_x = NAN, .spawn_y = NAN, .spawn_vx = NAN, .spawn_vy = NAN,
    };
    double nearest = INFINITY;
    for (uint32_t i = 0; i < state->header.entity_count; i++) {
        const BoplRlEntity *entity = &state->entities[i];
        if (entity->kind != 2 || !(entity->flags & 1)) continue;
        const double x = f(entity->x), y = f(entity->y);
        if (i >= initial_entities) {
            out.spawn_x = x; out.spawn_y = y;
            out.spawn_vx = f(entity->vx); out.spawn_vy = f(entity->vy);
        } else {
            const double distance = hypot(x - out.px, y - out.py);
            if (distance < nearest) {
                nearest = distance;
                out.map_x = x; out.map_y = y;
                out.map_vx = f(entity->vx); out.map_vy = f(entity->vy);
            }
        }
    }
    return out;
}

static Sample sample_last_map(const BoplRlState *state, uint32_t initial_entities) {
    Sample out = sample(state, initial_entities);
    for (uint32_t i = 0; i < state->header.entity_count && i < initial_entities; i++) {
        const BoplRlEntity *entity = &state->entities[i];
        if (entity->kind != 2 || !(entity->flags & 1)) continue;
        out.map_x = f(entity->x); out.map_y = f(entity->y);
        out.map_vx = f(entity->vx); out.map_vy = f(entity->vy);
    }
    return out;
}

static void reset(BoplRl **world, BoplRlConfig *config, uint32_t *initial) {
    for (int backend = 0; backend < 2; backend++) {
        if (bopl_rl_reset(world[backend], config) != BOPL_RL_OK) {
            fprintf(stderr, "reset backend %d: %s\n", backend,
                    bopl_rl_error(world[backend]));
            exit(1);
        }
        initial[backend] = bopl_rl_state(world[backend])->header.entity_count;
    }
}

static void row(const char *scenario, unsigned tick, BoplRl **world,
                const uint32_t *initial) {
    Sample n = scenario[0] == 'o' && scenario[1] == 'r'
        ? sample_last_map(bopl_rl_state(world[0]), initial[0])
        : sample(bopl_rl_state(world[0]), initial[0]);
    Sample r = scenario[0] == 'o' && scenario[1] == 'r'
        ? sample_last_map(bopl_rl_state(world[1]), initial[1])
        : sample(bopl_rl_state(world[1]), initial[1]);
    printf("%s,%u,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,"
           "%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f\n",
           scenario, tick,
           n.px, n.py, n.map_x, n.map_y, n.map_vx, n.map_vy,
           n.spawn_x, n.spawn_y, n.spawn_vx, n.spawn_vy,
           r.px, r.py, r.map_x, r.map_y, r.map_vx, r.map_vy,
           r.spawn_x, r.spawn_y, r.spawn_vx, r.spawn_vy);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <BoplSimNative.so> <libbopl_sim.so>\n", argv[0]);
        return 2;
    }
    char error[512] = {0};
    BoplRl *world[2];
    for (int backend = 0; backend < 2; backend++) {
        world[backend] = bopl_rl_open(argv[backend + 1],
            "/proc/self/cwd/native/fixtures/movement.rep",
            "/proc/self/cwd/native/target/platform_ab.log",
            120000, error, sizeof error);
        if (!world[backend]) {
            fprintf(stderr, "open backend %d: %s\n", backend, error);
            return 1;
        }
    }

    puts("scenario,tick,net_px,net_py,net_map_x,net_map_y,net_map_vx,net_map_vy,"
         "net_spawn_x,net_spawn_y,net_spawn_vx,net_spawn_vy,"
         "rust_px,rust_py,rust_map_x,rust_map_y,rust_map_vx,rust_map_vy,"
         "rust_spawn_x,rust_spawn_y,rust_spawn_vx,rust_spawn_vy");
    BoplRlConfig config;
    bopl_rl_default_config(&config, 2);
    config.seed = 7; config.map = 0; config.stocks = 0;
    config.warmup_ticks = 240; config.max_episode_ticks = 7200;
    for (int p = 0; p < 2; p++) {
        config.player[p].abilities[0] = 14;
        config.player[p].abilities[1] = 2;
        config.player[p].abilities[2] = 4;
    }
    uint32_t initial[2];
    reset(world, &config, initial);
    BoplRlAction actions[2] = {{0}};
    actions[0].angle = 65; /* up: retain the authored horizontal platform */
    for (unsigned tick = 1; tick <= 380; tick++) {
        actions[0].buttons = tick >= 276 && tick <= 332 ? 2 : 0;
        for (int backend = 0; backend < 2; backend++)
            if (bopl_rl_step(world[backend], actions, 2, 1) != BOPL_RL_OK) {
                fprintf(stderr, "overlap step backend %d: %s\n", backend,
                        bopl_rl_error(world[backend]));
                return 1;
            }
        if (tick >= 268) row("overlap", tick, world, initial);
    }

    config.warmup_ticks = 240;
    reset(world, &config, initial);
    actions[0].angle = 0;
    for (unsigned tick = 1; tick <= 180; tick++) {
        actions[0].buttons = tick == 1 ? 1 : 0;
        for (int backend = 0; backend < 2; backend++)
            if (bopl_rl_step(world[backend], actions, 2, 1) != BOPL_RL_OK) {
                fprintf(stderr, "landing step backend %d: %s\n", backend,
                        bopl_rl_error(world[backend]));
                return 1;
            }
        row("landing", tick, world, initial);
    }

    config.map = 2; config.stocks = 10;
    reset(world, &config, initial);
    for (unsigned tick = 1; tick <= 1620; tick++) {
        for (int backend = 0; backend < 2; backend++)
            if (bopl_rl_step(world[backend], actions, 2, 1) != BOPL_RL_OK) {
                fprintf(stderr, "orbiter step backend %d: %s\n", backend,
                        bopl_rl_error(world[backend]));
                return 1;
            }
        if ((tick >= 650 && tick <= 720) || tick >= 1540)
            row("orbiter", tick, world, initial);
    }

    bopl_rl_close(world[0]);
    bopl_rl_close(world[1]);
    return 0;
}
