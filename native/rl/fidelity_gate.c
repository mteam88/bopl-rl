/* Statistical fidelity gate between two bopl_rl.h backends.
 *
 *   fidelity_gate <reference.so> <candidate.so> [episodes] [seed0]
 *
 * Drives both backends with identical seeded pseudo-random action streams
 * (sticky inputs: direction held ~2 s, jump/ability presses at game-like
 * rates) using the run-007 training loadout (2 players, dash/bow/platform,
 * stocks 0, warmup 240, limit 7200, 4-tick action repeat), cycling maps.
 *
 * Reports two things:
 *   1. Short-horizon divergence: mean |Δposition| between backends per
 *      decision over the first 150 decisions (600 ticks), before chaos
 *      dominates. Physics bugs show up here as fast systematic drift.
 *   2. Outcome distributions per backend over all episodes: episode length,
 *      timeout rate, deaths by cause, ability activations, arrows
 *      fired/hit. Transferable fidelity requires these to agree within
 *      sampling noise even though individual long episodes diverge.
 *
 *   cc -O2 -o fidelity_gate fidelity_gate.c bopl_rl.c -ldl -lm
 */
#include "bopl_rl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PLAYERS 2
#define REPEAT 4
#define HORIZON 150 /* decisions measured for divergence */

typedef struct {
    uint64_t s;
} Rng;

static uint64_t rng_next(Rng *r) {
    uint64_t x = r->s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    r->s = x;
    return x;
}

/* Sticky action stream: per player, hold a direction for ~30 decisions,
 * press jump ~6% and each ability ~3% of decisions. */
typedef struct {
    uint8_t angle;
} Sticky;

static void sample_actions(Rng *rng, Sticky *st, BoplRlAction *actions) {
    for (int p = 0; p < PLAYERS; p++) {
        uint64_t r = rng_next(rng);
        if ((r & 31) == 0) { /* re-roll direction; often rest in place */
            uint64_t n = rng_next(rng);
            st[p].angle = (n % 100 < 40) ? 0 : (uint8_t)(1 + (n >> 8) % 255);
        }
        uint16_t buttons = 0;
        if ((r >> 8) % 100 < 6) buttons |= 1;  /* jump */
        if ((r >> 16) % 100 < 3) buttons |= 2; /* slot 0: dash */
        if ((r >> 24) % 100 < 3) buttons |= 4; /* slot 1: bow */
        if ((r >> 32) % 100 < 3) buttons |= 8; /* slot 2: platform */
        actions[p].angle = st[p].angle;
        actions[p].reserved = 0;
        actions[p].buttons = buttons;
    }
}

typedef struct {
    long episodes, timeouts, decisions;
    long deaths_env, deaths_opp, deaths_self;
    long activations[3];
    long arrows_fired, arrow_hits;
    double episode_len_sum;
} Stats;

static void accumulate(Stats *st, const BoplRlState *s, int decisions) {
    const BoplRlHeader *h = &s->header;
    st->episodes++;
    st->decisions += decisions;
    st->episode_len_sum += decisions;
    if (h->flags & BOPL_RL_TRUNCATED) st->timeouts++;
    for (uint32_t p = 0; p < h->player_count; p++) {
        const BoplRlTelemetry *t = &s->telemetry[p];
        st->deaths_env += t->deaths_environment;
        st->deaths_opp += t->deaths_opponent;
        st->deaths_self += t->deaths_self;
        for (int a = 0; a < 3; a++) st->activations[a] += t->ability_activations[a];
        st->arrows_fired += t->arrows_fired;
        st->arrow_hits += t->arrow_hits;
    }
}

static double fix_f(int64_t raw) { return (double)raw / 4294967296.0; }

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <reference.so> <candidate.so> [episodes] [seed0]\n", argv[0]);
        return 2;
    }
    int episodes = argc > 3 ? atoi(argv[3]) : 50;
    uint32_t seed0 = argc > 4 ? (uint32_t)atoi(argv[4]) : 1000;

    char error[512] = {0};
    BoplRl *world[2];
    for (int b = 0; b < 2; b++) {
        world[b] = bopl_rl_open(argv[1 + b], "/tmp/gate_bootstrap.rep",
                                "/tmp/fidelity_gate.log", 120000, error, sizeof error);
        if (!world[b]) {
            fprintf(stderr, "open %s failed: %s\n", argv[1 + b], error);
            return 1;
        }
    }

    Stats stats[2] = {0};
    double div_sum[HORIZON] = {0};
    long div_n[HORIZON] = {0};
    long winner_agree = 0, winner_total = 0;

    for (int e = 0; e < episodes; e++) {
        uint32_t map = (uint32_t)(e % 5);
        BoplRlConfig config;
        bopl_rl_default_config(&config, PLAYERS);
        config.seed = seed0 + (uint32_t)e;
        config.map = map;
        config.stocks = 0;
        config.warmup_ticks = 240;
        config.max_episode_ticks = 7200;
        for (int p = 0; p < PLAYERS; p++) {
            config.player[p].abilities[0] = 2;
            config.player[p].abilities[1] = 4;
            config.player[p].abilities[2] = 14;
        }
        for (int b = 0; b < 2; b++) {
            if (bopl_rl_reset(world[b], &config) != BOPL_RL_OK) {
                fprintf(stderr, "reset %d failed: %s\n", b, bopl_rl_error(world[b]));
                return 1;
            }
        }

        /* Both backends replay the identical action stream. */
        Rng rng = {.s = 0x9e3779b97f4a7c15ull ^ (seed0 + (uint64_t)e * 2654435761ull)};
        Sticky st[PLAYERS] = {0};
        int done[2] = {0, 0}, decisions[2] = {0, 0}, winner[2] = {-1, -1};
        for (int d = 1; !(done[0] && done[1]) && d <= 1800; d++) {
            BoplRlAction actions[PLAYERS];
            sample_actions(&rng, st, actions);
            for (int b = 0; b < 2; b++) {
                if (done[b]) continue;
                if (bopl_rl_step(world[b], actions, PLAYERS, REPEAT) != BOPL_RL_OK) {
                    fprintf(stderr, "step failed (%s): %s\n", argv[1 + b],
                            bopl_rl_error(world[b]));
                    return 1;
                }
                decisions[b] = d;
                const BoplRlState *s = bopl_rl_state(world[b]);
                if (s->header.flags & (BOPL_RL_TERMINAL | BOPL_RL_TRUNCATED)) {
                    done[b] = 1;
                    accumulate(&stats[b], s, d);
                    for (uint32_t p = 0; p < s->header.player_count; p++)
                        if (s->players[p].flags & 2 /* WonThisRound */) winner[b] = (int)p;
                }
            }
            if (!done[0] && !done[1] && d <= HORIZON) {
                const BoplRlState *a = bopl_rl_state(world[0]);
                const BoplRlState *c = bopl_rl_state(world[1]);
                for (int p = 0; p < PLAYERS; p++) {
                    double dx = fix_f(a->players[p].x) - fix_f(c->players[p].x);
                    double dy = fix_f(a->players[p].y) - fix_f(c->players[p].y);
                    div_sum[d - 1] += sqrt(dx * dx + dy * dy);
                    div_n[d - 1]++;
                }
            }
        }
        if (done[0] && done[1]) {
            winner_total++;
            if (winner[0] == winner[1]) winner_agree++;
        }
        printf("episode %2d map %u: len %4d/%4d decisions, winner %d/%d\n", e, map,
               decisions[0], decisions[1], winner[0], winner[1]);
    }

    printf("\n== short-horizon divergence (mean |dpos| units, both players) ==\n");
    for (int d = 0; d < HORIZON; d += 15) {
        double m = div_n[d] ? div_sum[d] / div_n[d] : 0;
        printf("decision %3d (tick %4d): %.4f\n", d + 1, (d + 1) * REPEAT, m);
    }

    const char *name[2] = {"reference", "candidate"};
    printf("\n== outcome distributions ==\n");
    printf("%-26s %12s %12s\n", "", name[0], name[1]);
    for (int b = 0; b < 2; b++) stats[b].episode_len_sum /= stats[b].episodes ? stats[b].episodes : 1;
    printf("%-26s %12ld %12ld\n", "episodes", stats[0].episodes, stats[1].episodes);
    printf("%-26s %12.1f %12.1f\n", "mean length (decisions)", stats[0].episode_len_sum,
           stats[1].episode_len_sum);
    printf("%-26s %12ld %12ld\n", "timeouts", stats[0].timeouts, stats[1].timeouts);
    printf("%-26s %12ld %12ld\n", "deaths: environment", stats[0].deaths_env, stats[1].deaths_env);
    printf("%-26s %12ld %12ld\n", "deaths: opponent", stats[0].deaths_opp, stats[1].deaths_opp);
    printf("%-26s %12ld %12ld\n", "deaths: self", stats[0].deaths_self, stats[1].deaths_self);
    printf("%-26s %12ld %12ld\n", "dash activations", stats[0].activations[0], stats[1].activations[0]);
    printf("%-26s %12ld %12ld\n", "bow activations", stats[0].activations[1], stats[1].activations[1]);
    printf("%-26s %12ld %12ld\n", "platform activations", stats[0].activations[2],
           stats[1].activations[2]);
    printf("%-26s %12ld %12ld\n", "arrows fired", stats[0].arrows_fired, stats[1].arrows_fired);
    printf("%-26s %12ld %12ld\n", "arrow hits", stats[0].arrow_hits, stats[1].arrow_hits);
    printf("\nwinner agreement: %ld/%ld\n", winner_agree, winner_total);

    bopl_rl_close(world[0]);
    bopl_rl_close(world[1]);
    return 0;
}
