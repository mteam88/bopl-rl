/* Number-representation A/B: drive two builds of the Rust backend (Q32.32
 * `--features q32` vs the default f64 `Fix`) with the fidelity gate's action
 * stream at action repeat 1, compare the complete observation every tick and
 * report the first tick where any field differs by more than a tolerance, then
 * how the divergence grows. Also usable against .NET as the first library.
 *
 *   cc -O2 -I. -o /tmp/repr_ab research/rust_ab/repr_ab.c bopl_rl.c -ldl -lm
 *   /tmp/repr_ab <a.so> <b.so> [episodes=10] [seed0=1000] [tol_raw=4096] [warmup=240]
 *
 * With warmup 0 every simulated tick is observable, so the first differing
 * field can be attributed to the operation that produced it. REPR_AB_TRACE=<e>
 * prints, for episode e, every tick where the worst player position gap grows
 * by 4x or more, with both players' positions, velocities, states and flags,
 * to locate the event that amplified a rounding-level difference.
 *
 * Discrete fields (flags, state, ability flags, entity flags, telemetry) use
 * tolerance zero. Continuous fields are compared in raw Q32.32 units; the
 * default tolerance 4096 raw = 2^-20 ~ 1e-6 units flags anything beyond the
 * accumulated 2^-32 rounding of a few hundred operations.
 */
#include "bopl_rl.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PLAYERS 2

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
typedef struct {
    uint8_t angle;
} Sticky;
/* Same stream as fidelity_gate.c, sampled once per 4 ticks so the per-tick
 * comparison sees exactly the decisions the gate would issue. */
static void sample_actions(Rng *rng, Sticky *st, BoplRlAction *actions) {
    for (int p = 0; p < PLAYERS; p++) {
        uint64_t r = rng_next(rng);
        if ((r & 31) == 0) {
            uint64_t n = rng_next(rng);
            st[p].angle = (n % 100 < 40) ? 0 : (uint8_t)(1 + (n >> 8) % 255);
        }
        uint16_t buttons = 0;
        if ((r >> 8) % 100 < 6) buttons |= 1;
        if ((r >> 16) % 100 < 3) buttons |= 2;
        if ((r >> 24) % 100 < 3) buttons |= 4;
        if ((r >> 32) % 100 < 3) buttons |= 8;
        actions[p].angle = st[p].angle;
        actions[p].reserved = 0;
        actions[p].buttons = buttons;
    }
}

static double fix_f(int64_t raw) { return (double)raw / 4294967296.0; }

typedef struct {
    int found;
    char what[256];
    int64_t a, b;
} Diff;

static int cont(Diff *d, const char *what, int64_t a, int64_t b, int64_t tol) {
    int64_t delta = a > b ? a - b : b - a;
    if (delta > tol) {
        if (!d->found) {
            d->found = 1;
            snprintf(d->what, sizeof d->what, "%s", what);
            d->a = a;
            d->b = b;
        }
        return 1;
    }
    return 0;
}
static int disc(Diff *d, const char *what, uint32_t a, uint32_t b) {
    if (a != b) {
        if (!d->found) {
            d->found = 1;
            snprintf(d->what, sizeof d->what, "%s", what);
            d->a = a;
            d->b = b;
        }
        return 1;
    }
    return 0;
}

static const BoplRlEntity *entity_by_id(const BoplRlState *s, uint32_t id) {
    for (uint32_t i = 0; i < s->header.entity_count; i++)
        if (s->entities[i].id == id) return &s->entities[i];
    return NULL;
}

/* Compare every field; return the number of differing fields and fill the first. */
static int compare(const BoplRlState *a, const BoplRlState *b, int64_t tol, Diff *d) {
    int n = 0;
    char what[128];
    n += disc(d, "header.flags", a->header.flags, b->header.flags);
    n += disc(d, "header.entity_count", a->header.entity_count, b->header.entity_count);
    n += disc(d, "header.slime_count", a->header.slime_count, b->header.slime_count);
    n += cont(d, "header.simulation_time", a->header.simulation_time, b->header.simulation_time, 0);
    for (uint32_t p = 0; p < a->header.player_count; p++) {
        const BoplRlPlayer *pa = &a->players[p], *pb = &b->players[p];
        snprintf(what, sizeof what, "player%u.flags", p);
        n += disc(d, what, pa->flags, pb->flags);
        snprintf(what, sizeof what, "player%u.stocks", p);
        n += disc(d, what, pa->stocks, pb->stocks);
        snprintf(what, sizeof what, "player%u.deaths", p);
        n += disc(d, what, pa->deaths, pb->deaths);
        snprintf(what, sizeof what, "player%u.cause_of_death", p);
        n += disc(d, what, pa->cause_of_death, pb->cause_of_death);
        snprintf(what, sizeof what, "player%u.x", p);
        n += cont(d, what, pa->x, pb->x, tol);
        snprintf(what, sizeof what, "player%u.y", p);
        n += cont(d, what, pa->y, pb->y, tol);
        snprintf(what, sizeof what, "player%u.ability_time", p);
        n += cont(d, what, pa->ability_time, pb->ability_time, 0);
        const BoplRlTelemetry *ta = &a->telemetry[p], *tb = &b->telemetry[p];
        for (int s = 0; s < 3; s++) {
            snprintf(what, sizeof what, "telemetry%u.activations[%d]", p, s);
            n += disc(d, what, ta->ability_activations[s], tb->ability_activations[s]);
        }
        snprintf(what, sizeof what, "telemetry%u.arrows_fired", p);
        n += disc(d, what, ta->arrows_fired, tb->arrows_fired);
        snprintf(what, sizeof what, "telemetry%u.arrow_hits", p);
        n += disc(d, what, ta->arrow_hits, tb->arrow_hits);
        snprintf(what, sizeof what, "telemetry%u.deaths_env", p);
        n += disc(d, what, ta->deaths_environment, tb->deaths_environment);
        snprintf(what, sizeof what, "telemetry%u.deaths_self", p);
        n += disc(d, what, ta->deaths_self, tb->deaths_self);
        snprintf(what, sizeof what, "telemetry%u.deaths_opp", p);
        n += disc(d, what, ta->deaths_opponent, tb->deaths_opponent);
    }
    uint32_t slimes = a->header.slime_count < b->header.slime_count ? a->header.slime_count
                                                                     : b->header.slime_count;
    for (uint32_t i = 0; i < slimes; i++) {
        const BoplRlSlime *sa = &a->slimes[i], *sb = &b->slimes[i];
        snprintf(what, sizeof what, "slime%u.flags", i);
        n += disc(d, what, sa->flags, sb->flags);
        snprintf(what, sizeof what, "slime%u.state", i);
        n += disc(d, what, sa->state, sb->state);
        snprintf(what, sizeof what, "slime%u.x", i);
        n += cont(d, what, sa->x, sb->x, tol);
        snprintf(what, sizeof what, "slime%u.y", i);
        n += cont(d, what, sa->y, sb->y, tol);
        snprintf(what, sizeof what, "slime%u.vx", i);
        n += cont(d, what, sa->vx, sb->vx, tol);
        snprintf(what, sizeof what, "slime%u.vy", i);
        n += cont(d, what, sa->vy, sb->vy, tol);
        snprintf(what, sizeof what, "slime%u.rotation", i);
        n += cont(d, what, sa->rotation, sb->rotation, tol);
        for (int s = 0; s < 3; s++) {
            snprintf(what, sizeof what, "slime%u.ability_flags[%d]", i, s);
            n += disc(d, what, sa->ability_flags[s], sb->ability_flags[s]);
            snprintf(what, sizeof what, "slime%u.timers[%d]", i, s);
            n += cont(d, what, sa->timers[s], sb->timers[s], 0);
            snprintf(what, sizeof what, "slime%u.cooldowns[%d]", i, s);
            n += cont(d, what, sa->cooldowns[s], sb->cooldowns[s], 0);
        }
    }
    for (uint32_t i = 0; i < a->header.entity_count; i++) {
        const BoplRlEntity *ea = &a->entities[i];
        const BoplRlEntity *eb = entity_by_id(b, ea->id);
        snprintf(what, sizeof what, "entity%u.present", ea->id);
        if (disc(d, what, 1, eb != NULL)) {
            n++;
            continue;
        }
        snprintf(what, sizeof what, "entity%u.flags", ea->id);
        n += disc(d, what, ea->flags, eb->flags);
        snprintf(what, sizeof what, "entity%u.kind", ea->id);
        n += disc(d, what, ea->kind, eb->kind);
        if (!(ea->flags & 1) || !(eb->flags & 1)) continue; /* inactive entities are zeroed */
        snprintf(what, sizeof what, "entity%u.x", ea->id);
        n += cont(d, what, ea->x, eb->x, tol);
        snprintf(what, sizeof what, "entity%u.y", ea->id);
        n += cont(d, what, ea->y, eb->y, tol);
        snprintf(what, sizeof what, "entity%u.vx", ea->id);
        n += cont(d, what, ea->vx, eb->vx, tol);
        snprintf(what, sizeof what, "entity%u.vy", ea->id);
        n += cont(d, what, ea->vy, eb->vy, tol);
        snprintf(what, sizeof what, "entity%u.right_x", ea->id);
        n += cont(d, what, ea->right_x, eb->right_x, tol);
        snprintf(what, sizeof what, "entity%u.right_y", ea->id);
        n += cont(d, what, ea->right_y, eb->right_y, tol);
        snprintf(what, sizeof what, "entity%u.up_x", ea->id);
        n += cont(d, what, ea->up_x, eb->up_x, tol);
        snprintf(what, sizeof what, "entity%u.up_y", ea->id);
        n += cont(d, what, ea->up_y, eb->up_y, tol);
        snprintf(what, sizeof what, "entity%u.radius", ea->id);
        n += cont(d, what, ea->radius, eb->radius, tol);
        snprintf(what, sizeof what, "entity%u.rotation", ea->id);
        n += cont(d, what, ea->rotation, eb->rotation, tol);
        snprintf(what, sizeof what, "entity%u.angular_velocity", ea->id);
        n += cont(d, what, ea->angular_velocity, eb->angular_velocity, tol);
        snprintf(what, sizeof what, "entity%u.inverse_mass", ea->id);
        n += cont(d, what, ea->inverse_mass, eb->inverse_mass, tol);
    }
    return n;
}

/* Largest position error (units) between the two backends over all players. */
static double position_gap(const BoplRlState *a, const BoplRlState *b) {
    double worst = 0;
    for (uint32_t p = 0; p < a->header.player_count; p++) {
        double dx = fix_f(a->players[p].x) - fix_f(b->players[p].x);
        double dy = fix_f(a->players[p].y) - fix_f(b->players[p].y);
        double g = sqrt(dx * dx + dy * dy);
        if (g > worst) worst = g;
    }
    return worst;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <a.so> <b.so> [episodes] [seed0] [tol_raw]\n", argv[0]);
        return 2;
    }
    int episodes = argc > 3 ? atoi(argv[3]) : 10;
    uint32_t seed0 = argc > 4 ? (uint32_t)atoi(argv[4]) : 1000;
    int64_t tol = argc > 5 ? strtoll(argv[5], NULL, 10) : 4096;
    uint32_t warmup = argc > 6 ? (uint32_t)atoi(argv[6]) : 240;
    char error[512] = {0};
    BoplRl *w[2];
    for (int b = 0; b < 2; b++) {
        w[b] = bopl_rl_open(argv[1 + b], "/tmp/gate_bootstrap.rep", "/tmp/repr_ab.log", 120000,
                            error, sizeof error);
        if (!w[b]) {
            fprintf(stderr, "open %s failed: %s\n", argv[1 + b], error);
            return 1;
        }
    }
    long identical_ticks = 0, total_ticks = 0, agree_len = 0, agree_winner = 0;
    const char *trace_env = getenv("REPR_AB_TRACE");
    int trace = trace_env ? atoi(trace_env) : -1;
    for (int e = 0; e < episodes; e++) {
        BoplRlConfig config;
        bopl_rl_default_config(&config, PLAYERS);
        config.seed = seed0 + (uint32_t)e;
        config.map = (uint32_t)(e % 5);
        config.stocks = 0;
        config.warmup_ticks = warmup;
        config.max_episode_ticks = 7200;
        for (int p = 0; p < PLAYERS; p++) {
            config.player[p].abilities[0] = 2;
            config.player[p].abilities[1] = 4;
            config.player[p].abilities[2] = 14;
        }
        for (int b = 0; b < 2; b++)
            if (bopl_rl_reset(w[b], &config) != BOPL_RL_OK) {
                fprintf(stderr, "reset failed: %s\n", bopl_rl_error(w[b]));
                return 1;
            }
        Rng rng = {.s = 0x9e3779b97f4a7c15ull ^ (seed0 + (uint64_t)e * 2654435761ull)};
        Sticky st[PLAYERS] = {0};
        BoplRlAction actions[PLAYERS];
        int done[2] = {0, 0}, len[2] = {0, 0}, winner[2] = {-1, -1};
        int first_tick = -1;
        double gap_at[6] = {0}; /* gap 60, 120, 240, 480, 960, 1920 ticks after first divergence */
        const int gap_ticks[6] = {60, 120, 240, 480, 960, 1920};
        double traced_gap = 0;
        const BoplRlState *sa = bopl_rl_state(w[0]), *sb = bopl_rl_state(w[1]);
        Diff d0 = {0};
        int n0 = compare(sa, sb, tol, &d0);
        if (n0) {
            first_tick = 0;
            printf("episode %2d map %u: differs after reset (%d fields): %s a=%" PRId64
                   " b=%" PRId64 " (delta %.3e units)\n",
                   e, config.map, n0, d0.what, d0.a, d0.b, fix_f(d0.a - d0.b));
        }
        for (int t = 1; !(done[0] && done[1]) && t <= 7200; t++) {
            if ((t - 1) % 4 == 0) sample_actions(&rng, st, actions);
            for (int b = 0; b < 2; b++) {
                if (done[b]) continue;
                if (bopl_rl_step(w[b], actions, PLAYERS, 1) != BOPL_RL_OK) {
                    fprintf(stderr, "step failed (%s): %s\n", argv[1 + b], bopl_rl_error(w[b]));
                    return 1;
                }
                const BoplRlState *s = bopl_rl_state(w[b]);
                if (s->header.flags & (BOPL_RL_TERMINAL | BOPL_RL_TRUNCATED)) {
                    done[b] = 1;
                    len[b] = t;
                    for (uint32_t p = 0; p < s->header.player_count; p++)
                        if (s->players[p].flags & 2) winner[b] = (int)p;
                }
            }
            if (done[0] || done[1]) continue;
            sa = bopl_rl_state(w[0]);
            sb = bopl_rl_state(w[1]);
            total_ticks++;
            Diff d = {0};
            int n = compare(sa, sb, tol, &d);
            if (!n) identical_ticks++;
            if (n && first_tick < 0) {
                first_tick = t;
                printf("episode %2d map %u: first divergence at tick %d (%d fields): %s a=%" PRId64
                       " b=%" PRId64 " (%.10f vs %.10f, delta %.3e units)\n",
                       e, config.map, t, n, d.what, d.a, d.b, fix_f(d.a), fix_f(d.b),
                       fix_f(d.a - d.b));
            }
            if (first_tick >= 0)
                for (int g = 0; g < 6; g++)
                    if (t == first_tick + gap_ticks[g]) gap_at[g] = position_gap(sa, sb);
            if (e == trace) {
                double gap = position_gap(sa, sb);
                if (gap > 4 * traced_gap && gap > 1e-9) {
                    printf("  t%5d gap %.3e", t, gap);
                    for (uint32_t i = 0; i < sa->header.slime_count && i < sb->header.slime_count; i++)
                        printf(" | s%u st %u/%u fl %x/%x ab %x%x%x/%x%x%x pos (%.6f,%.6f)/(%.6f,%.6f) v (%.4f,%.4f)/(%.4f,%.4f)",
                               i, sa->slimes[i].state, sb->slimes[i].state, sa->slimes[i].flags,
                               sb->slimes[i].flags, sa->slimes[i].ability_flags[0],
                               sa->slimes[i].ability_flags[1], sa->slimes[i].ability_flags[2],
                               sb->slimes[i].ability_flags[0], sb->slimes[i].ability_flags[1],
                               sb->slimes[i].ability_flags[2], fix_f(sa->slimes[i].x),
                               fix_f(sa->slimes[i].y), fix_f(sb->slimes[i].x), fix_f(sb->slimes[i].y),
                               fix_f(sa->slimes[i].vx), fix_f(sa->slimes[i].vy), fix_f(sb->slimes[i].vx),
                               fix_f(sb->slimes[i].vy));
                    printf("\n");
                    traced_gap = gap;
                }
            }
        }
        if (first_tick < 0) printf("episode %2d map %u: identical for all %d ticks\n", e, config.map, len[0]);
        else {
            printf("    gap after +60/+120/+240/+480/+960/+1920 ticks:");
            for (int g = 0; g < 6; g++) printf(" %.2e", gap_at[g]);
            printf("\n");
        }
        printf("    length %d/%d ticks, winner %d/%d%s\n", len[0], len[1], winner[0], winner[1],
               (len[0] == len[1] && winner[0] == winner[1]) ? "" : "  <-- outcome differs");
        agree_len += len[0] == len[1];
        agree_winner += winner[0] == winner[1];
    }
    printf("\nticks compared %ld, within tolerance %ld (%.1f%%); episodes with equal length %ld/%d, "
           "equal winner %ld/%d\n",
           total_ticks, identical_ticks, total_ticks ? 100.0 * identical_ticks / total_ticks : 0,
           agree_len, episodes, agree_winner, episodes);
    bopl_rl_close(w[0]);
    bopl_rl_close(w[1]);
    return 0;
}
