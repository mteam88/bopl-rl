/* Tick-by-tick first-divergence trace for ONE bounds_ability_ab case.
 *
 * bounds_ability_ab.c reports only the boundary kill: the tick and position at
 * which the out-of-bounds component fired. When a kit's kill tick differs by
 * more than a tick or two the interesting question is not the kill, it is the
 * much earlier tick at which the two simulations stopped agreeing. This runs
 * the same script as that probe for a single case, steps both backends in
 * lockstep, and reports the first tick at which any exported state field
 * differs, using eval/divergence/divergence_compare.h so the field set and the
 * tolerance are exactly the ones the rest of the harness uses.
 *
 * Both backends are driven from the SAME script. bounds_ability_ab lets each
 * backend take its own engage decision from its own pre-step state; here the
 * decision is taken once, from the reference, and replayed into the candidate,
 * so that a divergence in the trace is a divergence in the simulation and not
 * in the driver. The candidate's own decision is reported when it differs.
 *
 * Environment:
 *   BCT_ABILITIES=7,2,14   loadout for both players (default the gust kit)
 *   BCT_SLOT=0             ability slot held once engaged, -1 to fall plainly
 *   BCT_AIM=64             aim byte held while engaged
 *   BCT_ENGAGE_Y=-10.9     engage once player 0 is below this y ("brink")
 *   BCT_WALK_TICK=280      tick from which player 0 walks off the left edge;
 *                          raising it lets a long-cooldown slot come up
 *   BCT_TICKS=1600
 *   BCT_DUMP=a:b           per-tick dump of player 0 / slime 0 / the owned
 *                          entity rows over the inclusive tick range [a,b]
 *   BCT_DUMP_ALL=1         dump every active entity row, not just owned ones
 *   BCT_ALL=1              every differing field of the first diverging tick,
 *                          not just the first one the accumulator keeps
 *   BCT_IGNORE=a,b         generalized field names to exclude from the FIRST
 *                          lines, as DR_IGNORE; the excluded pair is printed
 *                          alongside so nothing is hidden silently
 */
#include "bopl_rl.h"
#include "../divergence/divergence_compare.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static double f(int64_t v) { return (double)v / 4294967296.0; }

/* Every differing field of the FIRST diverging tick, under BCT_ALL=1.
 * dr_compare walks header, then players, then slimes, then entities, and the
 * accumulator keeps only the first hit, so a header count or an aimed gun's
 * basis vector hides everything compared after it for the rest of the episode.
 * Later ticks are dropped: by then the two runs have different state and every
 * field differs, which says nothing about the cause. */
static unsigned bct_all_tick;
static int bct_all_started;
static void bct_report_all(const DrHit *hit) {
    if (bct_all_started && hit->tick != bct_all_tick) return;
    if (!bct_all_started) {
        bct_all_started = 1;
        bct_all_tick = hit->tick;
        printf("ALL tick %u, every differing field:\n", hit->tick);
    }
    /* The raw gap is what separates the cause from its wake: a real defect
     * stands orders of magnitude above the last-bit noise it feeds into
     * everything downstream of it, and eyeballing two 20-digit columns hides
     * that. */
    int64_t gap = hit->left > hit->right ? hit->left - hit->right : hit->right - hit->left;
    printf("  %-28s %-8s ref %20" PRId64 " cand %20" PRId64 " gap %14" PRId64
           " (%.6f)%s [%s]\n",
           hit->field, hit->discrete ? "DISCRETE" : "NUMERIC",
           hit->left, hit->right, gap, f(gap),
           hit->ignored ? " ignored" : "", hit->context);
}

static unsigned env_u(const char *name, unsigned fallback) {
    const char *v = getenv(name);
    return v && *v ? (unsigned)strtoul(v, NULL, 10) : fallback;
}
static double env_d(const char *name, double fallback) {
    const char *v = getenv(name);
    return v && *v ? strtod(v, NULL) : fallback;
}

/* 1 = dump every active entity row, not only the ones the case player owns.
 * A divergence the report-all mode finds on an unowned row (a level affector,
 * a moving wall) is invisible otherwise. */
static int dump_all_rows;

static void dump_side(const char *tag, const BoplRlState *s) {
    const BoplRlPlayer *p = &s->players[0];
    const BoplRlSlime *m = &s->slimes[0];
    printf("  %-9s player (%.6f,%.6f) deaths %u cause %u | slime flags %u state %u "
           "(%.6f,%.6f) v(%.6f,%.6f) rot %.6f scale %.6f | timers %.4f/%.4f/%.4f\n",
           tag, f(p->x), f(p->y), p->deaths, p->cause_of_death, m->flags, m->state,
           f(m->x), f(m->y), f(m->vx), f(m->vy), f(m->rotation), f(m->scale),
           f(m->timers[0]), f(m->timers[1]), f(m->timers[2]));
    for (uint32_t i = 0; i < s->header.entity_count; i++) {
        const BoplRlEntity *e = &s->entities[i];
        if (!(e->flags & BOPL_ENTITY_ACTIVE)) continue;
        if (!dump_all_rows && e->owner != p->id && e->reserved[0] == 0) continue;
        printf("      e%-3u id %-4u kind %u layer %u owner %u ab %u flags %u timer %u "
               "(%.6f,%.6f) v(%.6f,%.6f) r %.6f rot %.6f right(%.6f,%.6f) up(%.6f,%.6f)\n",
               i, e->id, e->kind, e->layer, e->owner, e->reserved[0], e->flags,
               e->reserved[1], f(e->x), f(e->y), f(e->vx), f(e->vy), f(e->radius),
               f(e->rotation), f(e->right_x), f(e->right_y), f(e->up_x), f(e->up_y));
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <BoplSimNative.so> <candidate.so>\n", argv[0]);
        return 2;
    }
    unsigned abilities[3] = {7, 2, 14};
    const char *kit = getenv("BCT_ABILITIES");
    if (kit && *kit)
        sscanf(kit, "%u,%u,%u", &abilities[0], &abilities[1], &abilities[2]);
    int slot = (int)strtol(getenv("BCT_SLOT") ? getenv("BCT_SLOT") : "0", NULL, 10);
    uint8_t aim = (uint8_t)env_u("BCT_AIM", 64);
    double engage_y = env_d("BCT_ENGAGE_Y", -10.9);
    unsigned ticks = env_u("BCT_TICKS", 1600);
    const unsigned walk_tick = env_u("BCT_WALK_TICK", 280);
    unsigned dump_lo = 0, dump_hi = 0;
    const char *dump = getenv("BCT_DUMP");
    if (dump && *dump) sscanf(dump, "%u:%u", &dump_lo, &dump_hi);
    const char *ignore = getenv("BCT_IGNORE");
    if (ignore && *ignore) dr_ignore_list = ignore;
    if (env_u("BCT_ALL", 0)) dr_report_all = bct_report_all;
    dump_all_rows = (int)env_u("BCT_DUMP_ALL", 0);

    char error[512] = {0};
    BoplRl *world[2];
    for (int backend = 0; backend < 2; backend++) {
        world[backend] = bopl_rl_open(argv[backend + 1], "/tmp/gate_bootstrap.rep",
                                      "/tmp/bounds_case_trace.log", 120000, error,
                                      sizeof error);
        if (!world[backend]) {
            fprintf(stderr, "open failed: %s\n", error);
            return 1;
        }
    }
    BoplRlConfig config;
    bopl_rl_default_config(&config, 2);
    config.seed = 123;
    config.map = 0;
    config.stocks = 5;
    config.warmup_ticks = 240;
    config.max_episode_ticks = 3000;
    for (int player = 0; player < 2; player++)
        for (int i = 0; i < 3; i++)
            config.player[player].abilities[i] = abilities[i];
    for (int backend = 0; backend < 2; backend++) {
        int status = bopl_rl_reset(world[backend], &config);
        if (status != BOPL_RL_OK) {
            fprintf(stderr, "reset %d failed: %s\n", backend, bopl_rl_error(world[backend]));
            return 1;
        }
    }
    printf("kit %u/%u/%u slot %d aim %u engage_y %.3f\n", abilities[0], abilities[1],
           abilities[2], slot, aim, engage_y);

    const BoplRlState *state[2] = {bopl_rl_state(world[0]), bopl_rl_state(world[1])};
    unsigned before_deaths = state[0]->players[0].deaths;
    unsigned engaged = 0, engaged_candidate = 0;
    unsigned death[2] = {0, 0};
    DrCombo combo;
    memset(&combo, 0, sizeof combo);
    dr_compare(state[0], state[1], 0, 0, 0, &combo);

    for (unsigned tick = 1; tick <= ticks; tick++) {
        BoplRlAction action[2] = {{0}};
        for (int backend = 0; backend < 2; backend++) {
            const BoplRlSlime *slime = &state[backend]->slimes[0];
            unsigned *flag = backend ? &engaged_candidate : &engaged;
            if (!*flag && slot >= 0 && tick > walk_tick &&
                slime->timers[slot] > slime->cooldowns[slot] &&
                state[backend]->players[0].y < (int64_t)(engage_y * 4294967296.0))
                *flag = tick;
        }
        if (engaged) {
            action[0].angle = aim;
            action[0].buttons = (uint16_t)(2u << slot);
        } else if (tick > walk_tick) {
            action[0].angle = 129;
            if ((tick - walk_tick) % 20 == 0) action[0].buttons = 1;
        }
        int ended = 0;
        for (int backend = 0; backend < 2; backend++) {
            int status = bopl_rl_step(world[backend], action, 2, 1);
            if (status == BOPL_RL_ENDED) ended = 1;
            else if (status != BOPL_RL_OK) {
                fprintf(stderr, "step %d tick %u: %s\n", backend, tick,
                        bopl_rl_error(world[backend]));
                return 1;
            }
            state[backend] = bopl_rl_state(world[backend]);
        }
        dr_compare(state[0], state[1], tick, 0, 0, &combo);
        for (int backend = 0; backend < 2; backend++)
            if (!death[backend] && state[backend]->players[0].deaths > before_deaths)
                death[backend] = tick;
        if (dump_hi && tick >= dump_lo && tick <= dump_hi) {
            printf("tick %u\n", tick);
            dump_side("reference", state[0]);
            dump_side("candidate", state[1]);
        }
        if (ended) break;
        if (death[0] && death[1]) break;
    }

    printf("engaged reference %u candidate %u\n", engaged, engaged_candidate);
    printf("death reference %u candidate %u\n", death[0], death[1]);
    if (combo.any.found)
        printf("FIRST any      tick %u %s %s ref %" PRId64 " (%.6f) cand %" PRId64
               " (%.6f) [%s]\n",
               combo.any.tick, combo.any.discrete ? "DISCRETE" : "NUMERIC",
               combo.any.field, combo.any.left, f(combo.any.left), combo.any.right,
               f(combo.any.right), combo.any.context);
    else
        printf("FIRST any      none\n");
    if (combo.disc.found)
        printf("FIRST discrete tick %u %s ref %" PRId64 " cand %" PRId64 " [%s]\n",
               combo.disc.tick, combo.disc.field, combo.disc.left, combo.disc.right,
               combo.disc.context);
    else
        printf("FIRST discrete none\n");
    if (dr_ignore_list && *dr_ignore_list) {
        printf("ignoring %s\n", dr_ignore_list);
        if (combo.any_ex.found)
            printf("FIRST kept     tick %u %s %s ref %" PRId64 " (%.6f) cand %" PRId64
                   " (%.6f) [%s]\n",
                   combo.any_ex.tick, combo.any_ex.discrete ? "DISCRETE" : "NUMERIC",
                   combo.any_ex.field, combo.any_ex.left, f(combo.any_ex.left),
                   combo.any_ex.right, f(combo.any_ex.right), combo.any_ex.context);
        else
            printf("FIRST kept     none\n");
    }
    for (int i = 0; i < 2; i++) bopl_rl_close(world[i]);
    return 0;
}
