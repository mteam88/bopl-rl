/* Dump BoplRlState as JSON lines from any bopl_rl.h backend.
 *
 *   dump_state <game.so-or-binary> <map> [ticks] [every] [seed]
 *
 * Resets with the run-007 training loadout (2 players, abilities 2/4/14,
 * stocks 0, warmup 240, limit 7200), prints the reset state, then steps
 * `ticks` decisions of 4 ticks with zero actions, printing every `every`-th
 * state. Ground truth for the Rust rollout backend (native/rl/RUST_ROLLOUTS.md).
 *
 *   cc -O2 -o dump_state dump_state.c bopl_rl.c -ldl -lm
 */
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_state(const BoplRlState *s, const char *label) {
    const BoplRlHeader *h = &s->header;
    printf("{\"label\":\"%s\",\"tick\":%u,\"episode_ticks\":%u,\"flags\":%u,"
           "\"players\":%u,\"entities\":%u,\"slimes\":%u,\"warmup\":%u,"
           "\"sim_time\":%" PRId64 ",\"time_scale\":%" PRId64 ",\n",
           label, h->tick, h->episode_ticks, h->flags, h->player_count,
           h->entity_count, h->slime_count, h->warmup_ticks,
           h->simulation_time, h->player_time_scale);
    printf(" \"player\":[");
    for (uint32_t i = 0; i < h->player_count; i++) {
        const BoplRlPlayer *p = &s->players[i];
        printf("%s{\"id\":%u,\"team\":%u,\"flags\":%u,\"bodies\":%u,\"stocks\":%u,"
               "\"kills\":%u,\"deaths\":%u,\"cod\":%u,\"x\":%" PRId64 ",\"y\":%" PRId64
               ",\"scale\":%" PRId64 ",\"invis\":%" PRId64 ",\"atime\":%" PRId64 "}",
               i ? "," : "", p->id, p->team, p->flags, p->alive_bodies, p->stocks,
               p->kills, p->deaths, p->cause_of_death, p->x, p->y, p->scale,
               p->invisible_time, p->ability_time);
    }
    printf("],\n \"entity\":[");
    for (uint32_t i = 0; i < h->entity_count; i++) {
        const BoplRlEntity *e = &s->entities[i];
        printf("%s{\"id\":%u,\"kind\":%u,\"layer\":%u,\"owner\":%u,\"flags\":%u,"
               "\"hier\":%u,\"x\":%" PRId64 ",\"y\":%" PRId64 ",\"vx\":%" PRId64
               ",\"vy\":%" PRId64 ",\"rx\":%" PRId64 ",\"ry\":%" PRId64
               ",\"ux\":%" PRId64 ",\"uy\":%" PRId64 ",\"radius\":%" PRId64
               ",\"rot\":%" PRId64 ",\"av\":%" PRId64 ",\"imass\":%" PRId64 "}",
               i ? "," : "", e->id, e->kind, e->layer, e->owner, e->flags, e->hierarchy,
               e->x, e->y, e->vx, e->vy, e->right_x, e->right_y, e->up_x, e->up_y,
               e->radius, e->rotation, e->angular_velocity, e->inverse_mass);
        if (i % 4 == 3) printf("\n ");
    }
    printf("],\n \"slime\":[");
    for (uint32_t i = 0; i < h->slime_count; i++) {
        const BoplRlSlime *m = &s->slimes[i];
        printf("%s{\"id\":%u,\"owner\":%u,\"flags\":%u,\"state\":%u,\"x\":%" PRId64
               ",\"y\":%" PRId64 ",\"vx\":%" PRId64 ",\"vy\":%" PRId64
               ",\"rot\":%" PRId64 ",\"scale\":%" PRId64
               ",\"ab\":[%u,%u,%u],\"abf\":[%u,%u,%u],\"cd\":[%" PRId64 ",%" PRId64
               ",%" PRId64 "],\"tm\":[%" PRId64 ",%" PRId64 ",%" PRId64 "]}",
               i ? "," : "", m->id, m->owner, m->flags, m->state, m->x, m->y,
               m->vx, m->vy, m->rotation, m->scale,
               m->abilities[0], m->abilities[1], m->abilities[2],
               m->ability_flags[0], m->ability_flags[1], m->ability_flags[2],
               m->cooldowns[0], m->cooldowns[1], m->cooldowns[2],
               m->timers[0], m->timers[1], m->timers[2]);
        if (i % 3 == 2) printf("\n ");
    }
    printf("],\n \"telemetry\":[");
    for (uint32_t i = 0; i < h->player_count; i++) {
        const BoplRlTelemetry *t = &s->telemetry[i];
        printf("%s{\"act\":[%u,%u,%u],\"fired\":%u,\"hits\":%u,\"denv\":%u,\"dopp\":%u,"
               "\"dself\":%u,\"dother\":%u}", i ? "," : "",
               t->ability_activations[0], t->ability_activations[1],
               t->ability_activations[2], t->arrows_fired, t->arrow_hits,
               t->deaths_environment, t->deaths_opponent, t->deaths_self, t->deaths_other);
    }
    printf("]}\n");
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <game> <map> [decisions] [every] [seed]\n", argv[0]); return 2; }
    const char *game = argv[1];
    uint32_t map = (uint32_t)atoi(argv[2]);
    int decisions = argc > 3 ? atoi(argv[3]) : 0;
    int every = argc > 4 ? atoi(argv[4]) : 1;
    uint32_t seed = argc > 5 ? (uint32_t)atoi(argv[5]) : 37;

    const char *replay = argc > 6 ? argv[6] : "/tmp/bootstrap.rep";
    const char *log = argc > 7 ? argv[7] : "/tmp/dump_state.log";
    char error[512] = {0};
    BoplRl *world = bopl_rl_open(game, replay, log, 120000, error, sizeof error);
    if (!world) { fprintf(stderr, "open failed: %s\n", error); return 1; }

    BoplRlConfig config;
    bopl_rl_default_config(&config, 2);
    config.seed = seed; config.map = map; config.stocks = 0;
    config.warmup_ticks = 240; config.max_episode_ticks = 7200;
    for (int p = 0; p < 2; p++) {
        config.player[p].abilities[0] = 2;   /* dash */
        config.player[p].abilities[1] = 4;   /* bow */
        config.player[p].abilities[2] = 14;  /* platform */
    }
    int rc = bopl_rl_reset(world, &config);
    if (rc != BOPL_RL_OK) { fprintf(stderr, "reset failed (%d): %s\n", rc, bopl_rl_error(world)); return 1; }
    print_state(bopl_rl_state(world), "reset");

    BoplRlAction actions[2] = {0};
    for (int d = 1; d <= decisions; d++) {
        rc = bopl_rl_step(world, actions, 2, 4);
        if (rc != BOPL_RL_OK) { fprintf(stderr, "step %d failed (%d): %s\n", d, rc, bopl_rl_error(world)); return 1; }
        const BoplRlState *s = bopl_rl_state(world);
        if (d % every == 0 || (s->header.flags & BOPL_RL_TERMINAL)) {
            char label[32];
            snprintf(label, sizeof label, "step%d", d);
            print_state(s, label);
        }
        if (s->header.flags & BOPL_RL_TERMINAL) break;
    }
    bopl_rl_close(world);
    return 0;
}
