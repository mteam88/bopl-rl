/* Final-death DynamicAbilityPickup golden. A third player keeps the match
 * alive while player zero leaves the map, then player one collects the drop
 * after its first seven-second ability reroll.
 */
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static int64_t delta(int64_t a, int64_t b) { return a > b ? a - b : b - a; }

static int entity_equal(const BoplRlEntity *a, const BoplRlEntity *b,
                        int64_t *max_vector_delta) {
    if (a->id != b->id || a->kind != b->kind || a->layer != b->layer ||
        a->owner != b->owner || a->flags != b->flags ||
        a->hierarchy != b->hierarchy || delta(a->right_x, b->right_x) > 1 ||
        delta(a->right_y, b->right_y) > 1 || delta(a->up_x, b->up_x) > 1 ||
        delta(a->up_y, b->up_y) > 1 || delta(a->radius, b->radius) > 1 ||
        a->rotation != b->rotation ||
        a->angular_velocity != b->angular_velocity ||
        a->inverse_mass != b->inverse_mass)
        return 0;
    int64_t d[4] = {delta(a->x, b->x), delta(a->y, b->y),
                    delta(a->vx, b->vx), delta(a->vy, b->vy)};
    for (int i = 0; i < 4; i++)
        if (d[i] > *max_vector_delta) *max_vector_delta = d[i];
    if (a->layer == 13) {
        /* Q32 Vec2.NormalizedSafe evaluates the sampled launch direction
         * with a few raw units of error. Drag carries that bounded error;
         * it never changes a collision or lifecycle boundary. */
        return d[0] <= 4096 && d[1] <= 4096 && d[2] <= 4096 && d[3] <= 4096;
    }
    return d[0] == 0 && d[1] == 0 && d[2] == 0 && d[3] == 0;
}

static int compare(const BoplRlState *a, const BoplRlState *b,
                   unsigned episode_tick, int64_t *max_vector_delta) {
    if (memcmp(&a->header, &b->header, sizeof a->header) != 0) {
        fprintf(stderr, "tick %u header differs (level %u/%u entities %u/%u)\n",
                episode_tick, a->header.tick, b->header.tick,
                a->header.entity_count, b->header.entity_count);
        return 1;
    }
    for (unsigned i = 0; i < a->header.player_count; i++) {
        if (memcmp(&a->players[i], &b->players[i], sizeof a->players[i]) != 0 ||
            memcmp(&a->telemetry[i], &b->telemetry[i],
                   sizeof a->telemetry[i]) != 0) {
            fprintf(stderr, "tick %u player %u differs at (%" PRId64 ",%" PRId64
                            ")/(%" PRId64 ",%" PRId64 ") deaths %u/%u\n",
                    episode_tick, i, a->players[i].x, a->players[i].y,
                    b->players[i].x, b->players[i].y,
                    a->players[i].deaths, b->players[i].deaths);
            return 1;
        }
    }
    for (unsigned i = 0; i < a->header.slime_count; i++) {
        if (memcmp(&a->slimes[i], &b->slimes[i], sizeof a->slimes[i]) != 0) {
            fprintf(stderr, "tick %u slime %u differs at (%" PRId64 ",%" PRId64
                            ")/(%" PRId64 ",%" PRId64 ")\n",
                    episode_tick, i, a->slimes[i].x, a->slimes[i].y,
                    b->slimes[i].x, b->slimes[i].y);
            return 1;
        }
    }
    for (unsigned i = 0; i < a->header.entity_count; i++) {
        if (!entity_equal(&a->entities[i], &b->entities[i], max_vector_delta)) {
            fprintf(stderr, "tick %u entity %u differs: id %u/%u layer %u/%u "
                            "flags %u/%u hierarchy %u/%u p (%" PRId64
                            ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ") v "
                            "(%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ")\n",
                    episode_tick, i, a->entities[i].id, b->entities[i].id,
                    a->entities[i].layer, b->entities[i].layer,
                    a->entities[i].flags, b->entities[i].flags,
                    a->entities[i].hierarchy, b->entities[i].hierarchy,
                    a->entities[i].x, a->entities[i].y,
                    b->entities[i].x, b->entities[i].y,
                    a->entities[i].vx, a->entities[i].vy,
                    b->entities[i].vx, b->entities[i].vy);
            return 1;
        }
    }
    return 0;
}

static int reset(BoplRl *world, BoplRlConfig *config, const char *label) {
    int status = bopl_rl_reset(world, config);
    if (status != BOPL_RL_OK) {
        fprintf(stderr, "%s reset failed (%d): %s\n", label, status,
                bopl_rl_error(world));
        return 1;
    }
    return 0;
}

static const BoplRlSlime *slime_for(const BoplRlState *state, unsigned owner) {
    for (unsigned i = 0; i < state->header.slime_count; i++)
        if (state->slimes[i].owner == owner) return &state->slimes[i];
    return NULL;
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
            "/proc/self/cwd/native/target/ability_pickup_death_drop_ab.log",
            120000, error, sizeof error);
        if (!world[backend]) {
            fprintf(stderr, "open failed: %s\n", error);
            return 1;
        }
    }
    BoplRlConfig config;
    bopl_rl_default_config(&config, 3);
    config.seed = 7;
    config.map = 0;
    config.stocks = 0;
    config.warmup_ticks = 240;
    config.max_episode_ticks = 1600;
    for (int player = 0; player < 3; player++) {
        config.player[player].keyboard = 1;
        config.player[player].abilities[0] = 14;
        config.player[player].abilities[1] = 2;
        config.player[player].abilities[2] = 4;
    }
    if (reset(world[0], &config, "reference") ||
        reset(world[1], &config, "rust")) return 1;

    BoplRlAction action[3] = {{0}};
    int64_t max_vector_delta = 0;
    unsigned death_tick = 0, collect_tick = 0;
    unsigned dropped_id = 0, dropped_hierarchy = 0, acquired_ability = 0;
    int saw_reroll_window = 0, jump_cooldown = 0;
    for (unsigned episode_tick = 0; episode_tick < 1300; episode_tick++) {
        memset(action, 0, sizeof action);
        if (!death_tick) {
            action[0].buttons = 128;
            if (episode_tick % 50 == 0) action[0].buttons |= 1;
        }
        const BoplRlState *before = bopl_rl_state(world[0]);
        const BoplRlEntity *pickup = NULL;
        for (unsigned i = 0; i < before->header.entity_count; i++)
            if (before->entities[i].layer == 13 && (before->entities[i].flags & 1)) {
                pickup = &before->entities[i];
                break;
            }
        /* Wait past the strict seven-second swap boundary, then pursue the
         * hovering pickup with a remaining ordinary slime. */
        if (death_tick && episode_tick >= 510 && pickup) {
            saw_reroll_window = 1;
            int64_t dx = pickup->x - before->players[1].x;
            if (dx > INT64_C(2000000000)) action[1].buttons |= 512;
            else if (dx < -INT64_C(2000000000)) action[1].buttons |= 128;
            if (jump_cooldown > 0) jump_cooldown--;
            if (jump_cooldown == 0) {
                action[1].buttons |= 1;
                jump_cooldown = 45;
            }
        }
        for (int backend = 0; backend < 2; backend++) {
            int status = bopl_rl_step(world[backend], action, 3, 1);
            if (status != BOPL_RL_OK) {
                fprintf(stderr, "tick %u backend %d failed: %s\n",
                        episode_tick + 1, backend, bopl_rl_error(world[backend]));
                return 1;
            }
        }
        const BoplRlState *state[2] = {
            bopl_rl_state(world[0]), bopl_rl_state(world[1])};
        if (compare(state[0], state[1], episode_tick + 1,
                    &max_vector_delta)) return 1;

        const BoplRlEntity *live = NULL;
        for (unsigned i = 0; i < state[0]->header.entity_count; i++)
            if (state[0]->entities[i].layer == 13 &&
                (state[0]->entities[i].flags & 1)) {
                live = &state[0]->entities[i];
                break;
            }
        if (!death_tick && state[0]->players[0].deaths == 1) {
            if (!live || live->flags != 1 || live->id != 20 ||
                live->hierarchy != 100000095 ||
                live->x != INT64_C(-157721924043) ||
                live->y != INT64_C(-50279367391)) {
                fprintf(stderr, "death-drop payload mismatch at level tick %u\n",
                        state[0]->header.tick);
                return 1;
            }
            death_tick = state[0]->header.tick;
            dropped_id = live->id;
            dropped_hierarchy = live->hierarchy;
        }
        if (death_tick && !live && !collect_tick) {
            const BoplRlSlime *collector = slime_for(state[0], 2);
            if (!collector) {
                fprintf(stderr, "collector slime missing at pickup removal\n");
                return 1;
            }
            acquired_ability = collector->abilities[2];
            collect_tick = state[0]->header.tick;
            if (!saw_reroll_window || acquired_ability == 4) {
                fprintf(stderr, "pickup disappeared before a verified reroll/collection "
                                "at %u collector ability %u\n",
                        collect_tick, acquired_ability);
                return 1;
            }
            break;
        }
    }
    if (death_tick != 318 || !collect_tick) {
        fprintf(stderr, "incomplete death-drop lifecycle death=%u collect=%u\n",
                death_tick, collect_tick);
        return 1;
    }
    printf("ABILITY_PICKUP_DEATH_DROP_GOLDEN_OK death=%u id=%u hierarchy=%u "
           "collect=%u ability=%u max_q32_delta=%" PRId64 "\n",
           death_tick, dropped_id, dropped_hierarchy, collect_tick,
           acquired_ability, max_vector_delta);
    bopl_rl_close(world[0]);
    bopl_rl_close(world[1]);
    return 0;
}
