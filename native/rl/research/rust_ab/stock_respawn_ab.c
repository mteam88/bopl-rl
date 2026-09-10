/* Stock respawn A/B: walk player 1 into the water, select the right-hand
 * point, and confirm early with a jump edge.
 *
 * cc -O2 -I. -o ../target/release/stock_respawn_ab \
 *   research/rust_ab/stock_respawn_ab.c bopl_rl.c -ldl -lm
 */
#include "bopl_rl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double f(int64_t raw) { return (double)raw / 4294967296.0; }

static const BoplRlSlime *slime(const BoplRlState *state, uint32_t owner) {
    for (uint32_t i = 0; i < state->header.slime_count; i++)
        if (state->slimes[i].owner == owner) return &state->slimes[i];
    return NULL;
}

static void row(unsigned tick, BoplRl **world) {
    const BoplRlState *n = bopl_rl_state(world[0]);
    const BoplRlState *r = bopl_rl_state(world[1]);
    const BoplRlSlime *ns = slime(n, 1), *rs = slime(r, 1);
    printf("%u,%u,%u,%u,%.12f,%.12f,%u,%u,%.12f,"
           "%u,%u,%u,%.12f,%.12f,%u,%u,%.12f\n",
           tick, n->players[0].deaths, n->players[0].stocks,
           n->players[0].cause_of_death, f(n->players[0].x), f(n->players[0].y),
           ns ? ns->flags : 0, ns ? ns->state : 999, ns ? f(ns->timers[0]) : NAN,
           r->players[0].deaths, r->players[0].stocks,
           r->players[0].cause_of_death, f(r->players[0].x), f(r->players[0].y),
           rs ? rs->flags : 0, rs ? rs->state : 999, rs ? f(rs->timers[0]) : NAN);
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
            "/proc/self/cwd/native/target/stock_respawn_ab.log",
            120000, error, sizeof error);
        if (!world[backend]) {
            fprintf(stderr, "open backend %d: %s\n", backend, error);
            return 1;
        }
    }

    BoplRlConfig config;
    bopl_rl_default_config(&config, 2);
    config.seed = 7; config.map = 0; config.stocks = 2;
    config.warmup_ticks = 240; config.max_episode_ticks = 12000;
    for (int p = 0; p < 2; p++) {
        config.player[p].abilities[0] = 2;
        config.player[p].abilities[1] = 4;
        config.player[p].abilities[2] = 14;
    }
    for (int backend = 0; backend < 2; backend++) {
        if (bopl_rl_reset(world[backend], &config) != BOPL_RL_OK) {
            fprintf(stderr, "reset backend %d: %s\n", backend,
                    bopl_rl_error(world[backend]));
            return 1;
        }
    }

    puts("tick,net_deaths,net_stocks,net_cause,net_x,net_y,net_slime_flags,"
         "net_state,net_timer0,rust_deaths,rust_stocks,rust_cause,rust_x,rust_y,"
         "rust_slime_flags,rust_state,rust_timer0");
    BoplRlAction actions[2] = {{0}};
    unsigned both_dead_tick = 0;
    for (unsigned tick = 1; tick <= 1800; tick++) {
        const BoplRlState *n = bopl_rl_state(world[0]);
        const BoplRlState *r = bopl_rl_state(world[1]);
        if (!both_dead_tick && n->players[0].deaths && r->players[0].deaths)
            both_dead_tick = tick;
        if (!both_dead_tick) {
            actions[0].angle = 129; /* left */
            actions[0].buttons = 2; /* dash left whenever the slot is ready */
        } else {
            unsigned since = tick - both_dead_tick;
            actions[0].angle = since >= 2 ? 1 : 0; /* select right */
            actions[0].buttons = since == 8 ? 1 : 0; /* early confirm */
        }
        for (int backend = 0; backend < 2; backend++) {
            if (bopl_rl_step(world[backend], actions, 2, 1) != BOPL_RL_OK) {
                fprintf(stderr, "step backend %d: %s\n", backend,
                        bopl_rl_error(world[backend]));
                return 1;
            }
        }
        if (n->players[0].deaths || r->players[0].deaths || both_dead_tick)
            row(tick, world);
        if (both_dead_tick && tick > both_dead_tick + 50) break;
    }
    if (!both_dead_tick) {
        fprintf(stderr, "script did not kill both player bodies\n");
        return 1;
    }
    bopl_rl_close(world[0]);
    bopl_rl_close(world[1]);
    return 0;
}
