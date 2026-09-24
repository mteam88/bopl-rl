/* Platform recast A/B: cast twice and retain the evicted platform while its
 * authored removal curve shrinks the still-collidable body.
 *
 * cc -O2 -I sim/abi -o ../target/release/recast_ab \
 *   eval/rust_ab/recast_ab.c sim/abi/bopl_rl.c -ldl -lm
 */
#include "bopl_rl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double f(int64_t raw) { return (double)raw / 4294967296.0; }

typedef struct {
    unsigned count;
    double old_x, old_y, old_radius, old_inverse_mass;
    double new_x, new_y, new_radius, new_inverse_mass;
} Sample;

static Sample sample(const BoplRlState *state, uint32_t initial_entities) {
    Sample out = {
        .old_x = NAN, .old_y = NAN, .old_radius = NAN,
        .old_inverse_mass = NAN, .new_x = NAN, .new_y = NAN,
        .new_radius = NAN, .new_inverse_mass = NAN,
    };
    for (uint32_t i = initial_entities; i < state->header.entity_count; i++) {
        const BoplRlEntity *entity = &state->entities[i];
        if (entity->kind != 2 || !(entity->flags & 1)) continue;
        double *x = out.count == 0 ? &out.old_x : &out.new_x;
        double *y = out.count == 0 ? &out.old_y : &out.new_y;
        double *radius = out.count == 0 ? &out.old_radius : &out.new_radius;
        double *mass = out.count == 0 ? &out.old_inverse_mass : &out.new_inverse_mass;
        *x = f(entity->x); *y = f(entity->y); *radius = f(entity->radius);
        *mass = f(entity->inverse_mass);
        out.count++;
    }
    return out;
}

static void row(unsigned tick, BoplRl **world, const uint32_t *initial) {
    Sample n = sample(bopl_rl_state(world[0]), initial[0]);
    Sample r = sample(bopl_rl_state(world[1]), initial[1]);
    printf("%u,%u,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,"
           "%u,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f\n",
           tick, n.count, n.old_x, n.old_y, n.old_radius, n.old_inverse_mass,
           n.new_x, n.new_y, n.new_radius, n.new_inverse_mass,
           r.count, r.old_x, r.old_y, r.old_radius, r.old_inverse_mass,
           r.new_x, r.new_y, r.new_radius, r.new_inverse_mass);
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
            "/proc/self/cwd/native/target/recast_ab.log",
            120000, error, sizeof error);
        if (!world[backend]) {
            fprintf(stderr, "open backend %d: %s\n", backend, error);
            return 1;
        }
    }

    BoplRlConfig config;
    bopl_rl_default_config(&config, 2);
    config.seed = 7; config.map = 0; config.stocks = 0;
    config.warmup_ticks = 240; config.max_episode_ticks = 12000;
    for (int p = 0; p < 2; p++) {
        config.player[p].abilities[0] = 14;
        config.player[p].abilities[1] = 2;
        config.player[p].abilities[2] = 4;
    }
    uint32_t initial[2];
    for (int backend = 0; backend < 2; backend++) {
        if (bopl_rl_reset(world[backend], &config) != BOPL_RL_OK) {
            fprintf(stderr, "reset backend %d: %s\n", backend,
                    bopl_rl_error(world[backend]));
            return 1;
        }
        initial[backend] = bopl_rl_state(world[backend])->header.entity_count;
    }

    puts("tick,net_count,net_old_x,net_old_y,net_old_radius,net_old_inverse_mass,"
         "net_new_x,net_new_y,net_new_radius,net_new_inverse_mass,"
         "rust_count,rust_old_x,rust_old_y,rust_old_radius,rust_old_inverse_mass,"
         "rust_new_x,rust_new_y,rust_new_radius,rust_new_inverse_mass");
    BoplRlAction actions[2] = {{0}};
    actions[0].angle = 65;
    for (unsigned tick = 1; tick <= 850; tick++) {
        actions[0].buttons = ((tick >= 276 && tick <= 332) ||
                              (tick >= 710 && tick <= 766)) ? 2 : 0;
        for (int backend = 0; backend < 2; backend++) {
            if (bopl_rl_step(world[backend], actions, 2, 1) != BOPL_RL_OK) {
                fprintf(stderr, "step backend %d: %s\n", backend,
                        bopl_rl_error(world[backend]));
                return 1;
            }
        }
        if (tick >= 700) row(tick, world, initial);
    }
    bopl_rl_close(world[0]);
    bopl_rl_close(world[1]);
    return 0;
}
