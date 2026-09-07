#include "bopl_puffer.h"
#define OBS_SIZE BOPL_OBS_SIZE
#define NUM_ATNS 2
#define ACT_SIZES {256, 16}
#define OBS_TENSOR_T FloatTensor
#define Env BoplPuffer
/* Every player is routed through per-slot pointers (my_setup_perm), so the
 * trainer's agent permutation and the historical-opponent slots actually take
 * effect; without MY_USES_PERM the vector silently ignores set_agent_perm. */
#define MY_USES_PERM
#define MY_USES_TAGS
/* The upstream constructor/destructor are wrapped below so ctypes callers can
 * find a vector from Python; upstream code and the pybind module keep calling
 * create_static_vec/static_vec_close and see identical behaviour. */
#define create_static_vec bopl_upstream_create_static_vec
#define static_vec_close bopl_upstream_static_vec_close
#include "vecenv.h"
#undef create_static_vec
#undef static_vec_close

int bopl_puffer_telemetry_size(void) { return BOPL_TELEMETRY_SIZE; }

/* Which frozen banks currently hold a usable snapshot, shared by every
 * environment of this process; bit b = bank b. Bank environments draw their
 * opponent from this set at each match start (c_reset). */
static unsigned int bopl_valid_banks=0;
void bopl_set_valid_banks(unsigned int mask) { __atomic_store_n(&bopl_valid_banks, mask, __ATOMIC_SEQ_CST); }
unsigned int bopl_get_valid_banks(void) { return __atomic_load_n(&bopl_valid_banks, __ATOMIC_SEQ_CST); }

static int option(Dict *kwargs, const char *name, int fallback) {
    DictItem *item=dict_get_unsafe(kwargs, name);
    return item?(int)item->value:fallback;
}
/* Environment layout comes from three counts: `players` per game, `banks`
 * frozen opponent banks, and `historical_every` (0 = none; k = every k-th
 * environment is a historical one: its learner plays a frozen snapshot and it
 * consumes 1+banks slots, one per bank, so all bank rows stay adjacent to the
 * primary rows that PufferLib's bank_layout slices per buffer). The trainer
 * must build the agent permutation so slot 1+b of every historical
 * environment lands in bank b's slice of its buffer. */
void my_init(Env *env, Dict *kwargs) {
    unsigned int index=env->rng; // PufferLib assigns the physical environment index before my_init.
    env->rng^=(unsigned int)option(kwargs, "seed", 73);
    env->players=option(kwargs, "players", 6);
    int banks=option(kwargs, "banks", 0), every=option(kwargs, "historical_every", 0);
    env->banks=(every>0 && banks>0 && index%(unsigned)every==(unsigned)every-1)?banks:0;
    env->num_agents=env->banks?1+env->banks:env->players;
    env->tag=env->banks?1:0;
    env->valid_banks=&bopl_valid_banks;
    /* Side and bank draws use their own stream so the game setup sequence
     * (seed/map/loadouts from env->rng) is identical with or without banks. */
    env->side_rng=(index*2654435761u)^(unsigned int)option(kwargs, "seed", 73)^0x9e3779b9u;
    if (!env->side_rng) env->side_rng=1;
    DictItem *telemetry=dict_get_unsafe(kwargs, "telemetry_ptr");
    if (telemetry && telemetry->value) {
        if ((size_t)(index+1)*env->players>(size_t)option(kwargs,"telemetry_agents",0)) abort();
        env->telemetry=(float*)(uintptr_t)telemetry->value+(size_t)index*env->players*BOPL_TELEMETRY_SIZE;
    }
    env->observation_delay=option(kwargs, "observation_delay", 9);
    env->repeat=option(kwargs, "action_repeat", 4);
    env->random_maps=option(kwargs, "random_maps", 0);
    env->training_map_mask=(uint32_t)option(kwargs, "training_map_mask", 0);
    if (env->training_map_mask&0x80000000u) abort();
    env->random_loadouts=option(kwargs, "random_loadouts", 0);
    bopl_rl_default_config(&env->config, (uint32_t)env->players);
    env->config.map=(uint32_t)option(kwargs, "map", 0);
    env->config.slots=(uint32_t)option(kwargs, "slots", 3);
    env->config.stocks=(uint32_t)option(kwargs, "stocks", 3);
    env->config.mode=(uint32_t)option(kwargs, "mode", 0);
    env->config.warmup_ticks=(uint32_t)option(kwargs, "warmup_ticks", 240);
    env->config.max_episode_ticks=(uint32_t)option(kwargs, "max_episode_ticks", 3600);
    /* Rewards arrive scaled by 1000 because PufferLib's kwargs are integers
     * once they pass through its ini/argparse layer: reward_kill_milli=100 is
     * the old trainer's 0.1 kill/death shaping, reward_timeout_milli=-500 the
     * final trainer's timeout penalty. */
    env->reward_kill=(float)option(kwargs, "reward_kill_milli", 100)/1000.0f;
    env->reward_timeout=(float)option(kwargs, "reward_timeout_milli", 0)/1000.0f;
    if (env->players<1 || env->players>6 || env->repeat<1 || env->repeat>16) abort();
    for (int i=0; i<env->players; i++) {
        env->config.player[i].abilities[0]=(uint32_t)option(kwargs, "ability1", 2);
        env->config.player[i].abilities[1]=(uint32_t)option(kwargs, "ability2", 4);
        env->config.player[i].abilities[2]=(uint32_t)option(kwargs, "ability3", 14);
    }
    init(env);
}
/* Called by the vector at creation and after every set_agent_perm: slot s of
 * this environment is physical agent agent_perm[slot_base+s]. */
void my_setup_perm(StaticVec *vec, Env *env, int slot_base) {
    for (int s=0; s<env->num_agents; s++) {
        int phys=vec->agent_perm?vec->agent_perm[slot_base+s]:slot_base+s;
        if (phys<0 || phys>=vec->total_agents) { fprintf(stderr, "Bopl RL: agent permutation out of range\n"); abort(); }
        env->obs_ptr[s]=(float*)vec->observations+(size_t)phys*OBS_SIZE;
        env->action_ptr[s]=vec->actions+(size_t)phys*NUM_ATNS;
        env->reward_ptr[s]=vec->rewards+phys;
        env->terminal_ptr[s]=vec->terminals+phys;
    }
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
    if (out->capacity<48) {
        DictItem *items=realloc(out->items,48*sizeof(DictItem));
        if (!items) abort();
        out->items=items;out->capacity=48;
    }
    dict_set(out, "perf", log->perf); dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return); dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "environment_ticks", log->environment_ticks);
    dict_set(out, "observation_overflow", log->observation_overflow); dict_set(out, "game_errors", log->game_errors);
    dict_set(out, "learner_episodes", log->learner_episodes); dict_set(out, "learner_wins", log->learner_wins);
    dict_set(out, "historical_episodes", log->historical_episodes); dict_set(out, "historical_wins", log->historical_wins);
    for (int map=0; map<31; map++) if (log->map_fraction[map]) {
        dict_set(out, map_keys[map], log->map_fraction[map]);
    }
}

/* ctypes entry points on the loaded _C module. Vectors are identified by the
 * device observation pointer Python already wraps (VecEnv.gpu_obs_ptr). */
enum { BOPL_MAX_VECS=16 };
static StaticVec *bopl_registered[BOPL_MAX_VECS];
static pthread_mutex_t bopl_registry_lock=PTHREAD_MUTEX_INITIALIZER;
static StaticVec *bopl_find_vec(const void *gpu_observations) {
    StaticVec *found=NULL;
    pthread_mutex_lock(&bopl_registry_lock);
    for (int i=0; i<BOPL_MAX_VECS; i++)
        if (bopl_registered[i] && bopl_registered[i]->gpu_observations==gpu_observations) found=bopl_registered[i];
    pthread_mutex_unlock(&bopl_registry_lock);
    return found;
}
StaticVec *create_static_vec(int total_agents, int num_buffers, int gpu, Dict *vec_kwargs, Dict *env_kwargs) {
    StaticVec *vec=bopl_upstream_create_static_vec(total_agents, num_buffers, gpu, vec_kwargs, env_kwargs);
    pthread_mutex_lock(&bopl_registry_lock);
    int slot=-1;
    for (int i=0; i<BOPL_MAX_VECS && slot<0; i++) if (!bopl_registered[i]) slot=i;
    if (slot>=0) bopl_registered[slot]=vec;
    pthread_mutex_unlock(&bopl_registry_lock);
    if (slot<0) { fprintf(stderr, "Bopl vector registry full (%d live vectors)\n", BOPL_MAX_VECS); abort(); }
    return vec;
}
void static_vec_close(StaticVec *vec) {
    pthread_mutex_lock(&bopl_registry_lock);
    for (int i=0; i<BOPL_MAX_VECS; i++) if (bopl_registered[i]==vec) bopl_registered[i]=NULL;
    pthread_mutex_unlock(&bopl_registry_lock);
    bopl_upstream_static_vec_close(vec);
}
/* Vectors alive in this process, as the device observation pointers that key
 * every other ctypes entry point; the trainer's PuffeRL object does not expose
 * its vector, so the trainer looks it up here (exactly one vector while
 * training). Returns the count, filling up to `capacity` handles. */
int bopl_vec_handles(const void **out, int capacity) {
    int n=0;
    pthread_mutex_lock(&bopl_registry_lock);
    for (int i=0; i<BOPL_MAX_VECS; i++) if (bopl_registered[i]) { if (n<capacity) out[n]=bopl_registered[i]->gpu_observations; n++; }
    pthread_mutex_unlock(&bopl_registry_lock);
    return n;
}
/* Start every environment again, after the trainer installed its agent
 * permutation and tags: PufferLib resets the vector inside create_pufferl with
 * identity routing, so the first observations sit on the wrong rows until the
 * games are reset once more. Only between rollouts. */
int bopl_vec_reset(const void *gpu_observations) {
    StaticVec *vec=bopl_find_vec(gpu_observations);
    if (!vec) return -1;
    static_vec_reset(vec);
    /* Every row starts a fresh episode: flag it so the first rollout step
     * zeroes the recurrent state of every bank before consuming observation 0
     * (the trainer's warmup/capture passes leave those states dirty, and the
     * upstream reset drops the opponent-row flags c_reset raised). */
    for (int i=0; i<vec->total_agents; i++) vec->terminals[i]=1;
    if (vec->gpu) {
        if (cudaMemcpy(vec->gpu_terminals, vec->terminals, (size_t)vec->total_agents*sizeof(float), cudaMemcpyHostToDevice)!=cudaSuccess) return -2;
        cudaDeviceSynchronize();
    }
    return 0;
}
/* Host mirrors of the observation, reward and terminal rows (the games write
 * them; the vector uploads them each step), for checks between rollouts.
 * Returns total_agents, or -1 for an unknown vector. */
int bopl_vec_host(const void *gpu_observations, const void **obs, const void **rewards, const void **terminals) {
    StaticVec *vec=bopl_find_vec(gpu_observations);
    if (!vec) return -1;
    if (obs) *obs=vec->observations;
    if (rewards) *rewards=vec->rewards;
    if (terminals) *terminals=vec->terminals;
    return vec->total_agents;
}
/* Step transport actually in use by the vector's games (sampled after real
 * stepping so an old game build cannot be mistaken for the new path):
 *  2 = every game steps through the single-roundtrip sampled command
 *  1 = the two-roundtrip oracle was forced with BOPL_SPLIT_STEP
 *  0 = at least one game fell back to two step commands (old game build)
 * -1 = unknown vector. */
int bopl_vec_step_mode(const void *gpu_observations) {
    StaticVec *vec=bopl_find_vec(gpu_observations);
    if (!vec) return -1;
    Env *envs=(Env*)vec->envs;
    int split=0;
    for (int i=0; i<vec->size; i++) {
        if (envs[i].split_step) split++;
        else if (!bopl_rl_sampled_native(envs[i].world)) return 0;
    }
    return split==vec->size?1:2;
}
/* Routing introspection for tests and the trainer's sidecar: per environment,
 * the physical agent index of each slot (-1 past num_agents), the current
 * side, active bank, tag, and counters. `out` holds 8+BOPL_MAX_SLOTS ints per
 * environment; returns the environment count or -1. */
int bopl_vec_layout(const void *gpu_observations, int *out, int capacity) {
    StaticVec *vec=bopl_find_vec(gpu_observations);
    if (!vec) return -1;
    int stride=8+BOPL_MAX_SLOTS;
    if (capacity<vec->size*stride) return -1;
    Env *envs=(Env*)vec->envs;
    for (int i=0; i<vec->size; i++) {
        int *row=out+i*stride;
        row[0]=envs[i].num_agents; row[1]=envs[i].players; row[2]=envs[i].banks; row[3]=envs[i].tag;
        row[4]=envs[i].side; row[5]=envs[i].active_bank; row[6]=(int)envs[i].matches; row[7]=(int)envs[i].stepped;
        for (int s=0; s<BOPL_MAX_SLOTS; s++)
            row[8+s]=s<envs[i].num_agents?(int)(envs[i].obs_ptr[s]-(float*)vec->observations)/OBS_SIZE:-1;
    }
    return vec->size;
}
/* The same vector-level routing controls PufferLib exposes on its trainer, for
 * vectors created without one (correctness checks, evaluation). */
int bopl_vec_set_perm(const void *gpu_observations, const int *perm) {
    StaticVec *vec=bopl_find_vec(gpu_observations);
    if (!vec) return -1;
    static_vec_set_perm(vec, perm); return 0;
}
int bopl_vec_set_tags(const void *gpu_observations, const int *tags) {
    StaticVec *vec=bopl_find_vec(gpu_observations);
    if (!vec) return -1;
    static_vec_set_env_tags(vec, tags); return 0;
}
/* Cumulative counters summed over the vector's environments (BOPL_COUNTER_SIZE
 * doubles in BoplCounters order: learner_steps, game_steps, matches,
 * opponent_game_decisions, kills, buttons[4], telemetry[16], map_matches[31]).
 * Read between rollouts, when no environment thread is stepping. */
int bopl_vec_counters(const void *gpu_observations, double *out, int capacity) {
    StaticVec *vec=bopl_find_vec(gpu_observations);
    if (!vec || capacity<BOPL_COUNTER_SIZE) return -1;
    Env *envs=(Env*)vec->envs;
    memset(out, 0, BOPL_COUNTER_SIZE*sizeof(double));
    for (int i=0; i<vec->size; i++) {
        const double *c=(const double*)&envs[i].counters;
        for (int k=0; k<BOPL_COUNTER_SIZE; k++) out[k]+=c[k];
    }
    return BOPL_COUNTER_SIZE;
}
int bopl_counter_size(void) { return BOPL_COUNTER_SIZE; }
int bopl_vec_count_aligned(const void *gpu_observations, int tag, int reset_flags) {
    StaticVec *vec=bopl_find_vec(gpu_observations);
    if (!vec) return -1;
    return static_vec_count_aligned(vec, tag, reset_flags);
}
