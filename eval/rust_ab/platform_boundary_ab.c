/* Platform-form boundary A/B: create a platform just before sudden death and
 * hold it while the sink carries it below water. The ability body's boundary
 * component kills/exits the transformed player, but the separate spawned
 * ResizablePlatform has no boundary component and must remain collidable.
 */
#include "bopl_rl.h"
#include <stdio.h>

static unsigned active_rr(const BoplRlState *state) {
    unsigned count = 0;
    for (uint32_t i = 0; i < state->header.entity_count; i++)
        if (state->entities[i].kind == 2 && (state->entities[i].flags & 1)) count++;
    return count;
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    char error[512] = {0}; BoplRl *world[2];
    for (int backend = 0; backend < 2; backend++) {
        world[backend] = bopl_rl_open(argv[backend + 1],
            "/proc/self/cwd/native/fixtures/movement.rep",
            "/proc/self/cwd/native/target/platform_boundary_ab.log",
            120000, error, sizeof error);
        if (!world[backend]) { fprintf(stderr, "%s\n", error); return 1; }
    }
    BoplRlConfig config; bopl_rl_default_config(&config, 2);
    config.seed = 7; config.map = 0; config.stocks = 100;
    config.warmup_ticks = 240; config.max_episode_ticks = 9000;
    for (int player = 0; player < 2; player++) {
        config.player[player].abilities[0] = 2;
        config.player[player].abilities[1] = 4;
        config.player[player].abilities[2] = 14;
    }
    for (int backend = 0; backend < 2; backend++)
        if (bopl_rl_reset(world[backend], &config) != BOPL_RL_OK) return 1;

    BoplRlAction action[2] = {{0}};
    unsigned max_rr[2] = {0}, final_rr[2] = {0}, death_tick[2] = {0};
    double min_y[2] = {1000, 1000}, final_y[2] = {0};
    int running[2] = {1, 1};
    for (unsigned tick = 1; tick <= 900; tick++) {
        /* After cooldown, walk off map 0's left platform and transform while
         * falling through the water line. */
        action[0].angle = tick >= 365 ? 192 : (tick >= 300 ? 128 : 0);
        action[0].buttons = (tick == 300 ? 1 : 0) | (tick >= 365 ? 8 : 0);
        for (int backend = 0; backend < 2; backend++) {
            if (!running[backend]) continue;
            if (bopl_rl_step(world[backend], action, 2, 1) != BOPL_RL_OK) return 1;
            const BoplRlState *state = bopl_rl_state(world[backend]);
            unsigned count = active_rr(state);
            if (count > max_rr[backend]) max_rr[backend] = count;
            final_rr[backend] = count;
            final_y[backend] = state->players[0].y / 4294967296.0;
            if (final_y[backend] < min_y[backend]) min_y[backend] = final_y[backend];
            if (!death_tick[backend] && state->players[0].deaths) death_tick[backend] = tick;
            if (state->header.flags & 1) running[backend] = 0;
        }
        if (!running[0] && !running[1]) break;
    }
    printf("net_rr=%u/%u net_death=%u y=%.3f/%.3f rust_rr=%u/%u rust_death=%u y=%.3f/%.3f\n",
        max_rr[0], final_rr[0], death_tick[0], min_y[0], final_y[0],
        max_rr[1], final_rr[1], death_tick[1], min_y[1], final_y[1]);
    bopl_rl_close(world[0]); bopl_rl_close(world[1]); return 0;
}
