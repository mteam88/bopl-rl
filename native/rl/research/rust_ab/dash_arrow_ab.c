/* Incoming-arrow A/B while the target enters Dash.
 * Unity calls Arrow.OnCollide with the Player-layer Dash hitbox before it
 * asks PlayerCollision to kill the dasher. Because Dash owns PlayerBody, not
 * BoplBody, the arrow callback returns without landing or destroying it. */
#include "bopl_rl.h"
#include <stdio.h>

typedef struct { int death_tick; unsigned victim; unsigned arrows_at_death; } Result;

static Result run(BoplRl *world, int jump_tick, int dash_tick) {
    BoplRlConfig c;
    bopl_rl_default_config(&c, 2);
    c.seed = 7; c.map = 0; c.stocks = 0; c.warmup_ticks = 240;
    for (int p = 0; p < 2; p++) {
        c.player[p].abilities[0] = 2;
        c.player[p].abilities[1] = 4;
        c.player[p].abilities[2] = 14;
    }
    if (bopl_rl_reset(world, &c)) return (Result){.death_tick = -2};
    Result out = {.death_tick = -1};
    BoplRlAction a[2] = {{0}};
    for (int tick = 1; tick <= 350 && out.death_tick < 0; tick++) {
        /* Player 1 jumps, charges Bow, and fires right. */
        a[0].angle = 1;
        a[0].buttons = (tick == jump_tick ? 1 : 0) |
                       ((tick >= 220 && tick <= 279) ? 4 : 0);
        /* Player 2 dashes left through the incoming arrow. */
        a[1].angle = tick >= dash_tick ? 104 : 0;
        a[1].buttons = (tick == jump_tick ? 1 : 0) | (tick == dash_tick ? 2 : 0);
        if (bopl_rl_step(world, a, 2, 1)) break;
        const BoplRlState *s = bopl_rl_state(world);
        unsigned arrows = 0;
        for (unsigned i = 0; i < s->header.entity_count; i++)
            if (s->entities[i].layer == 9 && (s->entities[i].flags & 1)) arrows++;
        for (unsigned p = 0; p < 2; p++) if (s->players[p].deaths) {
            out.death_tick = tick;
            out.victim = s->players[p].id;
            out.arrows_at_death = arrows;
        }
    }
    return out;
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    char error[512] = {0}; BoplRl *world[2];
    for (int b = 0; b < 2; b++) {
        world[b] = bopl_rl_open(argv[b + 1], "/tmp/gate_bootstrap.rep",
                                "/tmp/dash_arrow_ab.log", 120000,
                                error, sizeof error);
        if (!world[b]) { fprintf(stderr, "%s\n", error); return 1; }
    }
    for (int dash_tick = 270; dash_tick <= 310; dash_tick++) {
        Result n = run(world[0], 180, dash_tick);
        Result r = run(world[1], 180, dash_tick);
        if (n.death_tick >= 0 || r.death_tick >= 0)
            printf("dash=%d net=%d/%u arrows=%u rust=%d/%u arrows=%u%s\n",
                   dash_tick, n.death_tick, n.victim, n.arrows_at_death,
                   r.death_tick, r.victim, r.arrows_at_death,
                   (n.death_tick != r.death_tick || n.victim != r.victim ||
                    n.arrows_at_death != r.arrows_at_death) ? " mismatch" : "");
    }
    for (int b = 0; b < 2; b++) bopl_rl_close(world[b]);
    return 0;
}
