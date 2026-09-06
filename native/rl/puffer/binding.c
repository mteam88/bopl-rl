#include "bopl_puffer.h"
#define OBS_SIZE BOPL_OBS_SIZE
#define NUM_ATNS 2
#define ACT_SIZES {256, 16}
#define OBS_TENSOR_T FloatTensor
#define Env BoplPuffer
#include "vecenv.h"

int bopl_puffer_telemetry_size(void) { return BOPL_TELEMETRY_SIZE; }

static int option(Dict *kwargs, const char *name, int fallback) {
    DictItem *item=dict_get_unsafe(kwargs, name);
    return item?(int)item->value:fallback;
}
void my_init(Env *env, Dict *kwargs) {
    unsigned int index=env->rng; // PufferLib assigns the physical environment index before my_init.
    env->rng^=(unsigned int)option(kwargs, "seed", 73);
    env->num_agents=option(kwargs, "players", 6);
    DictItem *telemetry=dict_get_unsafe(kwargs, "telemetry_ptr");
    if (telemetry && telemetry->value) {
        if ((size_t)(index+1)*env->num_agents>(size_t)option(kwargs,"telemetry_agents",0)) abort();
        env->telemetry=(float*)(uintptr_t)telemetry->value+(size_t)index*env->num_agents*BOPL_TELEMETRY_SIZE;
    }
    env->observation_delay=option(kwargs, "observation_delay", 9);
    env->repeat=option(kwargs, "action_repeat", 4);
    env->random_maps=option(kwargs, "random_maps", 0);
    env->training_map_mask=(uint32_t)option(kwargs, "training_map_mask", 0);
    if (env->training_map_mask&0x80000000u) abort();
    env->random_loadouts=option(kwargs, "random_loadouts", 0);
    bopl_rl_default_config(&env->config, (uint32_t)env->num_agents);
    env->config.map=(uint32_t)option(kwargs, "map", 0);
    env->config.slots=(uint32_t)option(kwargs, "slots", 3);
    env->config.stocks=(uint32_t)option(kwargs, "stocks", 3);
    env->config.mode=(uint32_t)option(kwargs, "mode", 0);
    env->config.warmup_ticks=(uint32_t)option(kwargs, "warmup_ticks", 240);
    env->config.max_episode_ticks=(uint32_t)option(kwargs, "max_episode_ticks", 3600);
    if (env->num_agents<1 || env->num_agents>6 || env->repeat<1 || env->repeat>16) abort();
    for (int i=0; i<env->num_agents; i++) {
        env->config.player[i].abilities[0]=(uint32_t)option(kwargs, "ability1", 2);
        env->config.player[i].abilities[1]=(uint32_t)option(kwargs, "ability2", 4);
        env->config.player[i].abilities[2]=(uint32_t)option(kwargs, "ability3", 14);
    }
    init(env);
}
void my_log(Log *log, Dict *out) {
    /* PufferLib retains key pointers and starts with room for only 32 fields. */
    static const char *map_keys[31]={
        "map_0_fraction", "map_1_fraction", "map_2_fraction", "map_3_fraction",
        "map_4_fraction", "map_5_fraction", "map_6_fraction", "map_7_fraction",
        "map_8_fraction", "map_9_fraction", "map_10_fraction", "map_11_fraction",
        "map_12_fraction", "map_13_fraction", "map_14_fraction", "map_15_fraction",
        "map_16_fraction", "map_17_fraction", "map_18_fraction", "map_19_fraction",
        "map_20_fraction", "map_21_fraction", "map_22_fraction", "map_23_fraction",
        "map_24_fraction", "map_25_fraction", "map_26_fraction", "map_27_fraction",
        "map_28_fraction", "map_29_fraction", "map_30_fraction"
    };
    if (out->capacity<40) {
        DictItem *items=realloc(out->items,40*sizeof(DictItem));
        if (!items) abort();
        out->items=items;out->capacity=40;
    }
    dict_set(out, "perf", log->perf); dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return); dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "environment_ticks", log->environment_ticks);
    dict_set(out, "observation_overflow", log->observation_overflow); dict_set(out, "game_errors", log->game_errors);
    for (int map=0; map<31; map++) if (log->map_fraction[map]) {
        dict_set(out, map_keys[map], log->map_fraction[map]);
    }
}
