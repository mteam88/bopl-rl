// Scripted bow test against the Rust rollout backend through bopl_rl.c.
// P1 waits for the bow cooldown, charges fully, fires right at P2.
#include <stdio.h>
#include <string.h>
#include "bopl_rl.h"

static const BoplRlState *S;
static void show(const char *label, const BoplRl *w) {
    S = bopl_rl_state((BoplRl *)w);
    printf("%s: tick=%u ep=%u flags=%u entities=%u slimes=%u\n", label,
           S->header.tick, S->header.episode_ticks, S->header.flags,
           S->header.entity_count, S->header.slime_count);
    for (unsigned i = 0; i < S->header.player_count; i++) {
        const BoplRlPlayer *p = &S->players[i];
        printf("  p%u flags=%u stocks=%u kills=%u deaths=%u cod=%u pos=(%.3f,%.3f)\n",
               p->id, p->flags, p->stocks, p->kills, p->deaths, p->cause_of_death,
               p->x / 4294967296.0, p->y / 4294967296.0);
    }
    for (unsigned i = 0; i < S->header.slime_count; i++) {
        const BoplRlSlime *s = &S->slimes[i];
        printf("  slime o=%u st=%u abf=[%u,%u,%u] t1=%.2f\n", s->owner, s->state,
               s->ability_flags[0], s->ability_flags[1], s->ability_flags[2],
               s->timers[1] / 4294967296.0);
    }
    for (unsigned i = 0; i < S->header.entity_count; i++) {
        const BoplRlEntity *e = &S->entities[i];
        if (e->layer == 9)
            printf("  arrow id=%u flags=%u pos=(%.2f,%.2f) v=(%.1f,%.1f)\n", e->id, e->flags,
                   e->x / 4294967296.0, e->y / 4294967296.0,
                   e->vx / 4294967296.0, e->vy / 4294967296.0);
    }
    printf("  telemetry p0: fired=%u hits=%u | p1 deaths: env=%u opp=%u\n",
           S->telemetry[0].arrows_fired, S->telemetry[0].arrow_hits,
           S->telemetry[1].deaths_environment, S->telemetry[1].deaths_opponent);
}

int main(int argc, char **argv) {
    char error[256];
    BoplRl *w = bopl_rl_open(argv[1], "/tmp/bootstrap.rep", "/tmp/bow.log", 30000, error, sizeof(error));
    if (!w) { fprintf(stderr, "open: %s\n", error); return 1; }
    BoplRlConfig c; bopl_rl_default_config(&c, 2);
    c.map = 0; c.seed = 37; c.stocks = 1; // one death allowed: watch respawn skip? stocks=1 -> 0 kept
    c.stocks = 0;
    for (int i = 0; i < 2; i++) { c.player[i].abilities[0]=2; c.player[i].abilities[1]=4; c.player[i].abilities[2]=14; }
    int r = bopl_rl_reset(w, &c);
    if (r) { fprintf(stderr, "reset: %d %s\n", r, bopl_rl_error(w)); return 1; }
    show("reset", w);
    BoplRlAction idle[2] = {{0,0,0},{0,0,0}};
    // Wait 70 ticks: bow ready (timer 1.4667 + 70/60 > 2.5).
    for (int t = 0; t < 70; t++) if ((r = bopl_rl_step(w, idle, 2, 1))) { fprintf(stderr, "step: %d\n", r); return 1; }
    show("pre-bow", w);
    // Hold bow (buttons bit2=4) aiming right (angle byte 1 => 0 rad).
    BoplRlAction hold[2] = {{1,0,4},{0,0,0}};
    for (int t = 0; t < 60; t++) if ((r = bopl_rl_step(w, hold, 2, 1))) { fprintf(stderr, "hold: %d\n", r); return 1; }
    show("charged", w);
    // Release: full-power shot to the right.
    for (int t = 0; t < 40 && !(bopl_rl_state(w)->header.flags & 1); t++)
        if ((r = bopl_rl_step(w, idle, 2, 1))) { fprintf(stderr, "release: %d (%s)\n", r, bopl_rl_error(w)); break; }
    show("after-release", w);
    bopl_rl_close(w);
    return 0;
}
