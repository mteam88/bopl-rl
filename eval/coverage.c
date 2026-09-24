/* Integration exercise, not a claim that every interaction was reached. */
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t random32(uint32_t *x) { *x^=*x<<13; *x^=*x>>17; *x^=*x<<5; return *x; }
int main(int argc, char **argv) {
    if (argc!=4) { fprintf(stderr, "coverage <absolute game> <absolute bootstrap.rep> <absolute game.log>\n"); return 2; }
    char error[512];
    BoplRl *world=bopl_rl_open(argv[1], argv[2], argv[3], 30000, error, sizeof(error));
    if (!world) { fprintf(stderr, "%s\n", error); return 1; }
    uint32_t maps=bopl_rl_map_count(world), abilities=bopl_rl_ability_count(world), failures=0;
    uint32_t cases=maps+abilities+6*4*3;
    uint64_t total_ticks=0;
    for (uint32_t test=0; test<cases; test++) {
        BoplRlConfig config; bopl_rl_default_config(&config, 6);
        config.seed=125432+test; config.max_episode_ticks=480;
        if (test<maps) config.map=test;
        else if (test<maps+abilities) {
            uint32_t ability=test-maps;
            for (uint32_t i=0; i<6; i++) for (int slot=0; slot<3; slot++) config.player[i].abilities[slot]=ability;
        } else {
            uint32_t variant=test-maps-abilities;
            config.players=1+variant%6; config.mode=(variant/6)%4; config.slots=1+variant/24;
            config.map=variant%maps;
            for (uint32_t i=0; i<config.players; i++) for (uint32_t j=0; j<config.slots; j++)
                config.player[i].abilities[j]=1+(variant+3*i+j)%(abilities-1);
        }
        int status=bopl_rl_reset(world, &config);
        uint32_t rng=675321+test, ticks=0, max_entities=0, max_slimes=0;
        uint32_t cast_transitions=0; int64_t timers[6][3]={{0}};
        while (!status) {
            const BoplRlState *state=bopl_rl_state(world);
            if (state->header.flags&BOPL_RL_TERMINAL) break;
            if (state->header.entity_count>max_entities) max_entities=state->header.entity_count;
            if (state->header.slime_count>max_slimes) max_slimes=state->header.slime_count;
            for (uint32_t i=0; i<state->header.slime_count; i++) {
                const BoplRlSlime *slime=&state->slimes[i];
                if (slime->owner<1 || slime->owner>6) continue;
                for (int j=0; j<3; j++) {
                    if (slime->timers[j]<timers[slime->owner-1][j]) cast_transitions++;
                    timers[slime->owner-1][j]=slime->timers[j];
                }
            }
            BoplRlAction actions[6]={{0}};
            for (uint32_t i=0; i<config.players; i++) {
                actions[i].angle=(uint8_t)random32(&rng);
                actions[i].buttons=(uint16_t)(random32(&rng)&15);
            }
            status=bopl_rl_step(world, actions, config.players, 4);
            if (!status) ticks=bopl_rl_state(world)->header.episode_ticks;
        }
        total_ticks+=ticks;
        printf("{\"case\":%u,\"map\":%u,\"players\":%u,\"mode\":%u,\"slots\":%u,\"ability_case\":%d,\"ticks\":%u,\"timer_decreases\":%u,\"max_entities\":%u,\"max_slimes\":%u,\"status\":%d}\n",
               test, config.map, config.players, config.mode, config.slots,
               test>=maps && test<maps+abilities?(int)(test-maps):-1, ticks, cast_transitions, max_entities, max_slimes, status);
        fflush(stdout);
        int expected_placeholder_rejection=test==maps && status==BOPL_RL_ARGUMENT;
        if (status && !expected_placeholder_rejection) { failures++; fprintf(stderr, "case %u failed: %s\n", test, bopl_rl_error(world)); }
        if (status==BOPL_RL_IO) break;
    }
    printf("{\"cases\":%u,\"maps\":%u,\"abilities\":%u,\"simulation_ticks\":%"PRIu64",\"failures\":%u}\n", cases, maps, abilities, total_ticks, failures);
    bopl_rl_close(world);
    return failures?1:0;
}
