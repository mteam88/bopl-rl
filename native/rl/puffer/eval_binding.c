/* The evaluator uses exactly the training observation, delay and reward code. */
#include "bopl_puffer.h"
int research_obs_size(void) { return BOPL_OBS_SIZE; }
int research_telemetry_size(void) { return BOPL_TELEMETRY_SIZE; }
BoplPuffer *research_open(uint32_t seed, uint32_t map, uint32_t limit, int delay) {
    BoplPuffer *env=calloc(1, sizeof(*env));
    if (!env) return NULL;
    env->num_agents=2; env->repeat=4; env->observation_delay=delay;
    env->no_autoreset=1; env->rng=seed;
    bopl_rl_default_config(&env->config, 2);
    env->config.map=map; env->config.stocks=3; env->config.max_episode_ticks=limit;
    for (int i=0; i<2; i++) {
        env->config.player[i].abilities[1]=4;
        env->config.player[i].abilities[2]=14;
    }
    env->observations=calloc(2*BOPL_OBS_SIZE, sizeof(float));
    env->actions=calloc(4, sizeof(float)); env->rewards=calloc(2, sizeof(float)); env->terminals=calloc(2, sizeof(float));
    if (!env->observations || !env->actions || !env->rewards || !env->terminals) abort();
    init(env); c_reset(env); return env;
}
void research_reset(BoplPuffer *env, uint32_t seed, uint32_t map) {
    env->rng=seed; env->config.map=map; c_reset(env);
    memset(env->last_telemetry, 0, sizeof(env->last_telemetry));
}
void research_step(BoplPuffer *env, const float *actions) { memcpy(env->actions, actions, 4*sizeof(float)); c_step(env); }
const float *research_obs(BoplPuffer *env) { return env->observations; }
const float *research_telemetry(BoplPuffer *env) { return env->last_telemetry; }
const BoplRlState *research_state(BoplPuffer *env) { return bopl_rl_state(env->world); }
int research_replay(BoplPuffer *env, const char *path) {
    const void *bytes; size_t length;
    if (bopl_rl_replay(env->world, &bytes, &length)) return 1;
    FILE *file=fopen(path, "wb"); if (!file) return 1;
    int status=fwrite(bytes, 1, length, file)!=length;
    return fclose(file)?1:status;
}
void research_close(BoplPuffer *env) {
    if (!env) return;
    c_close(env); free(env->observations); free(env->actions); free(env->rewards); free(env->terminals); free(env);
}
