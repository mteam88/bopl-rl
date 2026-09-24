/* Mine owner immunity and owner ignition differential.
 *
 * Mine.ScanForPlayers skips the owner outright:
 *
 *   if (player == null || player.isInvisible ||
 *       component.GetPlayerId() == player2.Id || player2.Team == player.Team)
 *       { num2++; continue; }
 *
 * so an armed mine never chases or proximity-detonates on the player that
 * threw it, however close it hovers (the prefab's scan radius is 6, not the
 * class default 3). Mine.OnCollide has no such exemption: any Igniters-layer
 * contact, the owner's own hurtbox included, sets
 * `timeSinceAnticipation = anticipationDuration * 0.8` and detonates ~0.072s
 * later, and Detonate gives the Explosion `PlayerOwnerId = item.OwnerId`, so
 * PlayerCollision resolves `num == playerId && flag3` to CauseOfDeath.Accident
 * and Player.Kill credits the owner a kill for its own death (`player.Kills++`
 * runs whenever GetPlayer(idOfKiller) is non-null; only the achievement hook is
 * gated on `idOfKiller != Id`).
 *
 * This is the mechanism behind the self-kills in the Teleport/Mine/Tesla study
 * evaluations, so both halves are pinned here: the mine sits inside its own
 * scan radius of the owner for the whole armed window without reacting, and
 * then the owner jumps into it and dies by its own hand.
 *
 * Entities are matched by hierarchy number rather than by observation id: the
 * two objects created on the fatal tick (the explosion and the dead player's
 * dropped ability pickup) receive swapped ids because deferred prefab
 * initialization is drained in the port's type-category order instead of
 * Updater's global hierarchy order. Every
 * other field, including both objects' own state, is compared exactly.
 *
 * cc -O2 -Wall -Wextra -Werror -I. -o /tmp/mine_owner_contact_ab \
 *   eval/rust_ab/mine_owner_contact_ab.c sim/abi/bopl_rl.c -ldl -lm
 */
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#define THROW_DOWN 192u      /* aim straight down: minThrowStrength, mine settles in reach */
#define CAST_HOLD 30u
#define JUMP_TICK 340u       /* the mine has been armed since ~tick 335 */
#define TOTAL_TICKS 400u
#define MINE_LAYER 10u
#define SCAN_RADIUS_RAW 25769803776LL /* rollout.json mine.scan_radius = 6.0 */

static int64_t distance(int64_t a, int64_t b) {
    return a > b ? a - b : b - a;
}

static int entity_equal(const BoplRlEntity *a, const BoplRlEntity *b) {
    return a->kind == b->kind && a->layer == b->layer && a->owner == b->owner &&
           a->flags == b->flags && distance(a->x, b->x) <= 64 &&
           distance(a->y, b->y) <= 64 && distance(a->vx, b->vx) <= 64 &&
           distance(a->vy, b->vy) <= 64 &&
           distance(a->right_x, b->right_x) <= 1 &&
           distance(a->right_y, b->right_y) <= 1 &&
           distance(a->up_x, b->up_x) <= 1 &&
           distance(a->up_y, b->up_y) <= 1 &&
           distance(a->radius, b->radius) <= 1 &&
           distance(a->rotation, b->rotation) <= 64 &&
           distance(a->angular_velocity, b->angular_velocity) <= 64 &&
           a->inverse_mass == b->inverse_mass;
}

static const BoplRlEntity *by_hierarchy(const BoplRlState *state,
                                        unsigned hierarchy) {
    for (unsigned i = 0; i < state->header.entity_count; i++)
        if (state->entities[i].hierarchy == hierarchy) return &state->entities[i];
    return NULL;
}

static int compare(const BoplRlState *a, const BoplRlState *b, unsigned tick) {
    if (a->header.entity_count != b->header.entity_count ||
        a->header.slime_count != b->header.slime_count ||
        a->header.tick != b->header.tick || a->header.flags != b->header.flags) {
        fprintf(stderr,
                "tick %u header differs: entities %u/%u slimes %u/%u flags %u/%u\n",
                tick, a->header.entity_count, b->header.entity_count,
                a->header.slime_count, b->header.slime_count, a->header.flags,
                b->header.flags);
        return 1;
    }
    for (unsigned i = 0; i < a->header.player_count; i++) {
        if (memcmp(&a->players[i], &b->players[i], sizeof a->players[i]) != 0 ||
            memcmp(&a->telemetry[i], &b->telemetry[i],
                   sizeof a->telemetry[i]) != 0) {
            fprintf(stderr,
                    "tick %u player %u differs: position (%" PRId64 ",%" PRId64
                    ")/(%" PRId64 ",%" PRId64 ") kills %u/%u deaths %u/%u "
                    "cause %u/%u self %u/%u\n",
                    tick, i, a->players[i].x, a->players[i].y, b->players[i].x,
                    b->players[i].y, a->players[i].kills, b->players[i].kills,
                    a->players[i].deaths, b->players[i].deaths,
                    a->players[i].cause_of_death, b->players[i].cause_of_death,
                    a->telemetry[i].deaths_self, b->telemetry[i].deaths_self);
            return 1;
        }
    }
    for (unsigned i = 0; i < a->header.slime_count; i++) {
        if (memcmp(&a->slimes[i], &b->slimes[i], sizeof a->slimes[i]) != 0) {
            fprintf(stderr,
                    "tick %u slime %u differs: state %u/%u position (%" PRId64
                    ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ")\n",
                    tick, i, a->slimes[i].state, b->slimes[i].state,
                    a->slimes[i].x, a->slimes[i].y, b->slimes[i].x,
                    b->slimes[i].y);
            return 1;
        }
    }
    for (unsigned i = 0; i < a->header.entity_count; i++) {
        const BoplRlEntity *left = &a->entities[i];
        const BoplRlEntity *right = by_hierarchy(b, left->hierarchy);
        if (!right) {
            fprintf(stderr, "tick %u hierarchy %u missing from the port\n", tick,
                    left->hierarchy);
            return 1;
        }
        if (!entity_equal(left, right)) {
            fprintf(stderr,
                    "tick %u hierarchy %u differs: kind %u/%u layer %u/%u "
                    "owner %u/%u flags %u/%u position (%" PRId64 ",%" PRId64
                    ")/(%" PRId64 ",%" PRId64 ")\n",
                    tick, left->hierarchy, left->kind, right->kind, left->layer,
                    right->layer, left->owner, right->owner, left->flags,
                    right->flags, left->x, left->y, right->x, right->y);
            return 1;
        }
    }
    return 0;
}

/* Q32.32 squared magnitude needs 128 bits: 6.0 raw is 2.6e10. */
static int within_scan_radius(int64_t dx, int64_t dy) {
    __int128 sum = (__int128)dx * dx + (__int128)dy * dy;
    __int128 limit = (__int128)SCAN_RADIUS_RAW * SCAN_RADIUS_RAW;
    return sum < limit;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <BoplSimNative.so> <libbopl_sim.so>\n",
                argv[0]);
        return 2;
    }
    char error[512] = {0};
    BoplRl *world[2];
    for (int backend = 0; backend < 2; backend++) {
        world[backend] =
            bopl_rl_open(argv[backend + 1], "/tmp/gate_bootstrap.rep",
                         "/tmp/mine_owner_contact_ab.log", 120000, error,
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
    config.stocks = 0; /* one life, as in the study evaluations: the fatal tick
                        * keeps CauseOfDeath instead of respawning immediately */
    config.warmup_ticks = 240;
    config.max_episode_ticks = 2000;
    config.player[0].abilities[0] = 24;
    config.player[0].abilities[1] = 2;
    config.player[0].abilities[2] = 14;
    config.player[1].abilities[0] = 2;
    config.player[1].abilities[1] = 4;
    config.player[1].abilities[2] = 14;
    for (int backend = 0; backend < 2; backend++) {
        int status = bopl_rl_reset(world[backend], &config);
        if (status != BOPL_RL_OK) {
            fprintf(stderr, "reset backend %d failed (%d): %s\n", backend, status,
                    bopl_rl_error(world[backend]));
            return 1;
        }
    }
    if (compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), 0)) return 1;

    unsigned cast_tick = 0, death_tick = 0, in_range_ticks = 0;
    /* Sampled on the fatal tick: with stocks left the player respawns and
     * cause_of_death goes back to NotDeadYet. */
    unsigned cause[2] = {0}, kills[2] = {0}, deaths_self[2] = {0};
    unsigned opponent_kills[2] = {0}, opponent_deaths[2] = {0};
    for (unsigned tick = 1; tick <= TOTAL_TICKS; tick++) {
        const BoplRlState *before = bopl_rl_state(world[0]);
        BoplRlAction action[2] = {{0}};
        if (!cast_tick &&
            before->slimes[0].timers[0] > before->slimes[0].cooldowns[0])
            cast_tick = tick;
        if (cast_tick && tick < cast_tick + CAST_HOLD) {
            action[0].buttons = 2;
            action[0].angle = (unsigned char)THROW_DOWN;
        }
        if (tick >= JUMP_TICK) action[0].buttons |= 1;
        for (int backend = 0; backend < 2; backend++) {
            int status = bopl_rl_step(world[backend], action, 2, 1);
            if (status != BOPL_RL_OK && status != BOPL_RL_ENDED) {
                fprintf(stderr, "tick %u backend %d failed (%d): %s\n", tick,
                        backend, status, bopl_rl_error(world[backend]));
                return 1;
            }
        }
        const BoplRlState *reference = bopl_rl_state(world[0]);
        if (compare(reference, bopl_rl_state(world[1]), tick)) return 1;

        /* Before the jump, the hovering mine must stay inside its own scan
         * radius of the owner without ever reacting to it. */
        if (tick < JUMP_TICK) {
            for (unsigned i = 0; i < reference->header.entity_count; i++) {
                const BoplRlEntity *entity = &reference->entities[i];
                if (entity->layer != MINE_LAYER || !(entity->flags & 1)) continue;
                if (within_scan_radius(entity->x - reference->players[0].x,
                                       entity->y - reference->players[0].y))
                    in_range_ticks++;
            }
        }
        if (reference->header.flags & (BOPL_RL_TERMINAL | BOPL_RL_TRUNCATED)) {
            if (!reference->players[0].deaths) {
                fprintf(stderr, "episode ended at tick %u without the owner dying\n",
                        tick);
                return 1;
            }
        }
        if (!death_tick && reference->players[0].deaths) {
            death_tick = tick;
            for (int backend = 0; backend < 2; backend++) {
                const BoplRlState *state = bopl_rl_state(world[backend]);
                cause[backend] = state->players[0].cause_of_death;
                kills[backend] = state->players[0].kills;
                deaths_self[backend] = state->telemetry[0].deaths_self;
                opponent_kills[backend] = state->players[1].kills;
                opponent_deaths[backend] = state->players[1].deaths;
            }
        }
        if (reference->header.flags & (BOPL_RL_TERMINAL | BOPL_RL_TRUNCATED)) break;
    }
    for (int backend = 0; backend < 2; backend++) bopl_rl_close(world[backend]);

    if (!cast_tick || !death_tick) {
        fprintf(stderr, "incomplete scenario cast=%u death=%u\n", cast_tick,
                death_tick);
        return 1;
    }
    if (in_range_ticks < 60) {
        fprintf(stderr,
                "the mine never lingered inside its scan radius of the owner "
                "(%u ticks); owner immunity was not exercised\n",
                in_range_ticks);
        return 1;
    }
    for (int backend = 0; backend < 2; backend++) {
        /* Accident, self-credited: the owner's own explosion killed it and
         * Player.Kill still incremented the killer's (its own) Kills. */
        if (cause[backend] != 2 || kills[backend] != 1 ||
            deaths_self[backend] != 1) {
            fprintf(stderr,
                    "backend %d owner outcome wrong: cause %u kills %u "
                    "deaths_self %u (expected 2/1/1)\n",
                    backend, cause[backend], kills[backend],
                    deaths_self[backend]);
            return 1;
        }
        if (opponent_kills[backend] || opponent_deaths[backend]) {
            fprintf(stderr,
                    "backend %d opponent was involved: kills %u deaths %u\n",
                    backend, opponent_kills[backend], opponent_deaths[backend]);
            return 1;
        }
    }
    printf("MINE_OWNER_CONTACT_GOLDEN_OK cast=%u death=%u in_range=%u\n",
           cast_tick, death_tick, in_range_ticks);
    return 0;
}
