/* AbilitySpawner/DynamicAbilityPickup A/B. Default RL sessions keep the
 * authored spawner disabled. BOPL_RL_ABILITY_SPAWNS=1 enables it in both
 * backends and compares the first spawn, delayed collider activation, launch,
 * hover and observation row through 2700 episode ticks.
 */
#include "bopl_rl.h"
#include <stdio.h>
#include <stdlib.h>

static const BoplRlEntity *pickup(const BoplRlState *state) {
    for (uint32_t i = 0; i < state->header.entity_count; i++) {
        const BoplRlEntity *entity = &state->entities[i];
        if (entity->layer == 13 && (entity->flags & 1)) return entity;
    }
    return NULL;
}

static long long gap(long long a, long long b) { return llabs(a - b); }

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
            "/proc/self/cwd/native/target/ability_spawner_ab.log",
            120000, error, sizeof error);
        if (!world[backend]) { fprintf(stderr, "%s\n", error); return 1; }
    }
    BoplRlConfig config;
    bopl_rl_default_config(&config, 2);
    config.seed = 7; config.map = 0; config.stocks = 100;
    config.warmup_ticks = 240; config.max_episode_ticks = 4000;
    for (int player = 0; player < 2; player++) {
        config.player[player].abilities[0] = 14;
        config.player[player].abilities[1] = 2;
        config.player[player].abilities[2] = 4;
    }
    for (int backend = 0; backend < 2; backend++)
        if (bopl_rl_reset(world[backend], &config) != BOPL_RL_OK) return 1;

    BoplRlAction actions[2] = {{0}};
    unsigned maximum[2] = {0}, first_tick[2] = {0}, enabled_tick[2] = {0};
    long long max_position_gap = 0, max_velocity_gap = 0;
    for (int tick = 0; tick < 2700; tick++) {
        for (int backend = 0; backend < 2; backend++)
            if (bopl_rl_step(world[backend], actions, 2, 1) != BOPL_RL_OK) return 1;
        const BoplRlState *state[2] = {bopl_rl_state(world[0]), bopl_rl_state(world[1])};
        const BoplRlEntity *item[2] = {pickup(state[0]), pickup(state[1])};
        for (int backend = 0; backend < 2; backend++) {
            unsigned count = item[backend] != NULL;
            if (count > maximum[backend]) maximum[backend] = count;
            if (item[backend] && first_tick[backend] == 0) first_tick[backend] = state[backend]->header.tick;
            if (item[backend] && (item[backend]->flags & 2) && enabled_tick[backend] == 0)
                enabled_tick[backend] = state[backend]->header.tick;
        }
        if (!!item[0] != !!item[1]) {
            fprintf(stderr, "pickup presence mismatch at episode tick %d\n", tick + 1);
            return 1;
        }
        if (!item[0]) continue;
        if (item[0]->id != item[1]->id || item[0]->flags != item[1]->flags ||
            item[0]->hierarchy != item[1]->hierarchy || item[0]->radius != item[1]->radius ||
            item[0]->inverse_mass != item[1]->inverse_mass) {
            fprintf(stderr, "pickup metadata mismatch at episode tick %d\n", tick + 1);
            return 1;
        }
        long long pg = gap(item[0]->x, item[1]->x);
        long long ygap = gap(item[0]->y, item[1]->y);
        long long vg = gap(item[0]->vx, item[1]->vx);
        long long vygap = gap(item[0]->vy, item[1]->vy);
        if (ygap > pg) pg = ygap;
        if (vygap > vg) vg = vygap;
        if (pg > max_position_gap) max_position_gap = pg;
        if (vg > max_velocity_gap) max_velocity_gap = vg;
    }
    const char *setting = getenv("BOPL_RL_ABILITY_SPAWNS");
    unsigned expected = setting && setting[0] == '1' ? 1 : 0;
    if (maximum[0] != expected || maximum[1] != expected ||
        first_tick[0] != first_tick[1] || enabled_tick[0] != enabled_tick[1] ||
        max_position_gap > 100000 || max_velocity_gap > 100000) {
        fprintf(stderr, "spawner mismatch net/rust=%u/%u first=%u/%u enabled=%u/%u pos=%lld vel=%lld\n",
                maximum[0], maximum[1], first_tick[0], first_tick[1], enabled_tick[0],
                enabled_tick[1], max_position_gap, max_velocity_gap);
        return 1;
    }
    printf("ABILITY_SPAWNER_GOLDEN_OK pickups=%u first=%u enabled=%u pos_gap=%lld vel_gap=%lld\n",
           expected, first_tick[0], enabled_tick[0], max_position_gap, max_velocity_gap);
    bopl_rl_close(world[0]); bopl_rl_close(world[1]);
    return 0;
}
