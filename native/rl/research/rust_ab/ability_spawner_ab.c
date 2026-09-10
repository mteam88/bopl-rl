/* RL ability-spawner configuration A/B. With the default environment both
 * backends must expose no pickup bodies after the authored 40-second spawn
 * boundary. BOPL_RL_ABILITY_SPAWNS=1 should make only the .NET oracle expose
 * its authored item-layer pickup, proving that the opt-in remains functional.
 */
#include "bopl_rl.h"
#include <stdio.h>

static unsigned pickups(const BoplRlState *state) {
    unsigned count = 0;
    for (uint32_t i = 0; i < state->header.entity_count; i++) {
        const BoplRlEntity *entity = &state->entities[i];
        if (entity->layer == 13 && (entity->flags & 1)) count++;
    }
    return count;
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
    unsigned maximum[2] = {0};
    for (int tick = 0; tick < 2700; tick++) {
        for (int backend = 0; backend < 2; backend++) {
            if (bopl_rl_step(world[backend], actions, 2, 1) != BOPL_RL_OK) return 1;
            unsigned count = pickups(bopl_rl_state(world[backend]));
            if (count > maximum[backend]) maximum[backend] = count;
        }
    }
    printf("net_pickups=%u rust_pickups=%u\n", maximum[0], maximum[1]);
    bopl_rl_close(world[0]);
    bopl_rl_close(world[1]);
    return 0;
}
