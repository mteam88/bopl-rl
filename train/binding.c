/* vecenv.h's env pool pins threads (pthread_setaffinity_np, sched_getaffinity). */
#define _GNU_SOURCE 1
#include "bopl_puffer.h"
/* Row memory fits either observation layout (bopl_layout_v4.h); the obs_layout
 * kwarg selects which one every player of the vector is encoded in. */
#define OBS_SIZE BOPL_OBS_STRIDE
#define NUM_ATNS BOPL_NUM_ATNS
#define ACT_SIZES {BOPL_ACT_AIM, BOPL_ACT_BUTTONS, BOPL_ACT_HOLD}
#define OBS_TENSOR_T FloatTensor
/* Per-agent action mask over the concatenated heads (see BoplPuffer.action_mask). */
#define MY_ACTION_MASK BOPL_LOGITS
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

/* Packed observation transport (bopl_obs_pack.h): the trainer (bopl_rollout.cu,
 * which does not see the env struct) registers the row buffer, the slot
 * buffer and the live row range of one buffer with its envs [start, start+count). */
void bopl_pack_register(void *envs, int start, int count, const float *base, float *out, long row_begin, long row_end) {
    BoplPuffer *e=(BoplPuffer*)envs;
    for (int i=start; i<start+count; i++) {
        e[i].pack_base=base; e[i].pack_out=out; e[i].pack_row_begin=row_begin; e[i].pack_row_end=row_end;
    }
}

static int option(Dict *kwargs, const char *name, int fallback) {
    DictItem *item=dict_get_unsafe(kwargs, name);
    return item?(int)item->value:fallback;
}
/* Environment layout comes from three counts: `players` per game, `banks`
 * frozen opponent banks, and `historical_every` (0 = none; k = every k-th
 * environment is a historical one: its learner plays a frozen snapshot and it
 * consumes 1+banks slots, one per bank, so all bank rows stay adjacent to the
 * primary rows that PufferLib's bank_layout slices per buffer). Historical
 * environments are dealt to the banks round-robin (the j-th historical
 * environment plays bank j%banks) and tagged 1+bank for PufferLib's selfplay
 * pool. The trainer must build the agent permutation so slot 1+b of every
 * historical environment lands in bank b's slice of its buffer.
 *
 * `team_every` (0 = none; k = the self-play game of every k-th group, a group
 * being `historical_every` environments) makes that self-play game a 2v2 team
 * game: four learner players, seats 0,1 against 2,3 (BoplPuffer.team_play,
 * docs/TEAM_PLAY.md). Historical games stay 1v1. NativeVector mirrors this. */
void my_init(Env *env, Dict *kwargs) {
    unsigned int index=env->rng; // PufferLib assigns the physical environment index before my_init.
    env->rng^=(unsigned int)option(kwargs, "seed", 73);
    env->players=option(kwargs, "players", 6);
    int banks=option(kwargs, "banks", 0), every=option(kwargs, "historical_every", 0);
    env->banks=(every>0 && banks>0 && index%(unsigned)every==(unsigned)every-1)?banks:0;
    int team_every=option(kwargs, "team_every", 0);
    if (team_every<0 || (team_every && every!=2)) { fprintf(stderr, "Bopl RL: team_every needs historical_every 2\n"); abort(); }
    env->team_play=team_every && !env->banks && (index/(unsigned)every)%(unsigned)team_every==(unsigned)team_every-1;
    if (env->team_play) env->players=4;
    env->num_agents=env->banks?1+env->banks:env->players;
    env->active_bank=env->banks?(int)((index/(unsigned)every)%(unsigned)env->banks):-1;
    env->tag=env->banks?1+env->active_bank:0;
    /* Side and bank draws use their own stream so the game setup sequence
     * (seed/map/loadouts from env->rng) is identical with or without banks. */
    env->side_rng=(index*2654435761u)^(unsigned int)option(kwargs, "seed", 73)^0x9e3779b9u;
    if (!env->side_rng) env->side_rng=1;
    DictItem *telemetry=dict_get_unsafe(kwargs, "telemetry_ptr");
    if (telemetry && telemetry->value) {
        if (team_every) { fprintf(stderr, "Bopl RL: per-player telemetry rows assume two players\n"); abort(); }
        if ((size_t)(index+1)*env->players>(size_t)option(kwargs,"telemetry_agents",0)) abort();
        env->telemetry=(float*)(uintptr_t)telemetry->value+(size_t)index*env->players*BOPL_TELEMETRY_SIZE;
    }
    env->observation_delay=option(kwargs, "observation_delay", 9);
    env->repeat=option(kwargs, "action_repeat", 4);
    env->random_maps=option(kwargs, "random_maps", 0);
    /* option() travels through a double. Four exact 31-bit chunks reconstruct
     * the native 128-bit selection without crossing the 53-bit integer limit. */
    const char *mask_keys[4]={"training_map_mask", "training_map_mask_hi",
                              "training_map_mask_2", "training_map_mask_3"};
    for (uint32_t chunk=0; chunk<4; chunk++) {
        uint32_t bits=(uint32_t)option(kwargs, mask_keys[chunk], 0);
        if (bits&0x80000000u) abort();
        for (uint32_t bit=0; bit<31; bit++) if (bits&(1u<<bit))
            bopl_map_select(&env->training_map_mask, chunk*31+bit);
    }
    env->random_loadouts=option(kwargs, "random_loadouts", 0);
    uint64_t pool_lo=(uint32_t)option(kwargs,"loadout_pool",0);
    uint64_t pool_hi=(uint32_t)option(kwargs,"loadout_pool_hi",0);
    if ((pool_lo|pool_hi)&0x80000000ull) abort();
    env->loadout_pool=pool_lo|(pool_hi<<31);
    env->loadout_no_dup=option(kwargs,"loadout_no_dup",0);
    env->mirror_share_ppm=option(kwargs,"mirror_share_ppm",0);
    env->loadout_set_count=bopl_loadout_set_count;
    if (env->loadout_set_count<0 || env->loadout_set_count>BOPL_LOADOUT_SET_MAX) abort();
    if (env->loadout_set_count) memcpy(env->loadout_set,bopl_loadout_sets,(size_t)env->loadout_set_count*6);
    if (env->loadout_pool && env->loadout_set_count) abort();
    env->loadout_weighted=0;
    for (int ability=0; ability<64; ability++) {
        env->loadout_weight[ability]=bopl_loadout_weights[ability];
        env->loadout_weighted|=bopl_loadout_weights[ability]!=0;
    }
    if (env->loadout_weighted) for (uint32_t ability=1; ability<BOPL_ABILITY_COUNT; ability++)
        if (((env->loadout_pool>>ability)&1u) && !env->loadout_weight[ability]) abort();
    env->loadout_mode=env->loadout_set_count?BOPL_LOADOUT_SET:(env->loadout_pool?BOPL_LOADOUT_POOL:BOPL_LOADOUT_FIXED);
    bopl_rl_default_config(&env->config, (uint32_t)env->players);
    if (env->team_play) for (int i=0; i<env->players; i++) env->config.player[i].team=(uint32_t)(i/2);
    env->config.map=(uint32_t)option(kwargs, "map", 0);
    env->config.slots=(uint32_t)option(kwargs, "slots", 3);
    env->config.stocks=(uint32_t)option(kwargs, "stocks", 0);
    env->config.mode=(uint32_t)option(kwargs, "mode", 0);
    env->config.warmup_ticks=(uint32_t)option(kwargs, "warmup_ticks", 240);
    env->config.max_episode_ticks=(uint32_t)option(kwargs, "max_episode_ticks", 3600);
    /* Rewards arrive scaled by 1000 because PufferLib's kwargs are integers
     * once they pass through its ini/argparse layer: reward_kill_milli=100 is
     * the old trainer's 0.1 kill/death shaping, reward_timeout_milli=-500 the
     * final trainer's timeout penalty. */
    env->reward_kill=(float)option(kwargs, "reward_kill_milli", 100)/1000.0f;
    env->reward_timeout=(float)option(kwargs, "reward_timeout_milli", 0)/1000.0f;
    env->reward_self_death=(float)option(kwargs, "reward_self_death_milli", 0)/1000.0f;
    /* Scripted opponent banks (bopl_bots.h): nibble b of scripted_kinds is the
     * bot kind bank b plays instead of frozen weights (0 = policy bank). */
    env->scripted_kinds=(uint32_t)option(kwargs, "scripted_kinds", 0);
    env->scripted_level=option(kwargs, "scripted_level", BOPL_BOT_HARD);
    if (env->players<1 || env->players>6 || env->repeat<1 || env->repeat>16) abort();
    int obs_layout=option(kwargs, "obs_layout", 3);
    if (obs_layout!=3 && obs_layout!=BOPL4_LAYOUT_VERSION) { fprintf(stderr, "Bopl RL: unknown obs_layout %d\n", obs_layout); abort(); }
    for (int i=0; i<BOPL_MAX_PLAYERS; i++) env->layout[i]=obs_layout;
    for (int i=0; i<env->players; i++) {
        env->config.player[i].abilities[0]=(uint32_t)option(kwargs, "ability1", 2);
        env->config.player[i].abilities[1]=(uint32_t)option(kwargs, "ability2", 4);
        env->config.player[i].abilities[2]=(uint32_t)option(kwargs, "ability3", 14);
    }
    if (!bopl_loadouts_valid(env)) abort();
    init(env);
}
/* Called by the vector at creation and after every set_agent_perm: slot s of
 * this environment is physical agent agent_perm[slot_base+s]. */
void my_setup_perm(StaticVec *vec, Env *env, int slot_base) {
    bopl_written_forget(env);  /* the rows this environment wrote may now belong to another */
    for (int s=0; s<env->num_agents; s++) {
        int phys=vec->agent_perm?vec->agent_perm[slot_base+s]:slot_base+s;
        if (phys<0 || phys>=vec->total_agents) { fprintf(stderr, "Bopl RL: agent permutation out of range\n"); abort(); }
        env->obs_ptr[s]=(float*)vec->observations+(size_t)phys*OBS_SIZE;
        env->action_ptr[s]=vec->actions+(size_t)phys*NUM_ATNS;
        env->reward_ptr[s]=vec->rewards+phys;
        env->terminal_ptr[s]=vec->terminals+phys;
        env->mask_ptr[s]=vec->action_mask+(size_t)phys*MY_ACTION_MASK;
    }
}
void my_log(Log *log, Dict *out) {
    /* PufferLib retains key pointers and starts with room for only 32 fields. */
    static char map_keys[BOPL_COUNTER_MAPS][24]; static int map_keys_ready;
    if (!map_keys_ready) {
        for (int map=0; map<BOPL_COUNTER_MAPS; map++)
            snprintf(map_keys[map], sizeof(map_keys[map]), "map_%d_fraction", map);
        map_keys_ready=1;
    }
    if (out->capacity<320) {
        DictItem *items=realloc(out->items,320*sizeof(DictItem));
        if (!items) abort();
        out->items=items;out->capacity=320;
    }
    dict_set(out, "perf", log->perf); dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return); dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "environment_ticks", log->environment_ticks);
    dict_set(out, "observation_overflow", log->observation_overflow); dict_set(out, "game_errors", log->game_errors);
    dict_set(out, "learner_episodes", log->learner_episodes); dict_set(out, "learner_wins", log->learner_wins);
    dict_set(out, "historical_episodes", log->historical_episodes); dict_set(out, "historical_wins", log->historical_wins);
    static const char *hold_keys[BOPL_ACT_HOLD]={"hold_1","hold_2","hold_4","hold_8","hold_16"};
    for (int h=0; h<BOPL_ACT_HOLD; h++) dict_set(out, hold_keys[h], log->hold_choice[h]);
    dict_set(out, "latched_decisions", log->latched_decisions);
    static const char *bank_n[8]={"hist_n_bank_0","hist_n_bank_1","hist_n_bank_2","hist_n_bank_3","hist_n_bank_4","hist_n_bank_5","hist_n_bank_6","hist_n_bank_7"};
    static const char *bank_score[8]={"hist_score_bank_0","hist_score_bank_1","hist_score_bank_2","hist_score_bank_3","hist_score_bank_4","hist_score_bank_5","hist_score_bank_6","hist_score_bank_7"};
    static const char *bank_decided[8]={"hist_decided_bank_0","hist_decided_bank_1","hist_decided_bank_2","hist_decided_bank_3","hist_decided_bank_4","hist_decided_bank_5","hist_decided_bank_6","hist_decided_bank_7"};
    for (int bank=0; bank<8; bank++) if (log->hist_n_bank[bank]) {
        dict_set(out, bank_n[bank], log->hist_n_bank[bank]); dict_set(out, bank_score[bank], log->hist_score_bank[bank]);
        dict_set(out, bank_decided[bank], log->hist_decided_bank[bank]);
    }
    static const char *bot_n[8]={"bot_n_bank_0","bot_n_bank_1","bot_n_bank_2","bot_n_bank_3","bot_n_bank_4","bot_n_bank_5","bot_n_bank_6","bot_n_bank_7"};
    static const char *bot_score[8]={"bot_score_bank_0","bot_score_bank_1","bot_score_bank_2","bot_score_bank_3","bot_score_bank_4","bot_score_bank_5","bot_score_bank_6","bot_score_bank_7"};
    for (int bank=0; bank<8; bank++) if (log->bot_n_bank[bank]) {
        dict_set(out, bot_n[bank], log->bot_n_bank[bank]); dict_set(out, bot_score[bank], log->bot_score_bank[bank]);
    }
    static char entry_n[32][24], entry_score[32][28], entry_decided[32][32];
    static int entry_keys;
    if (!entry_keys) {
        for (int i=0; i<32; i++) {
            snprintf(entry_n[i], sizeof(entry_n[i]), "bot_n_entry_%d", i);
            snprintf(entry_score[i], sizeof(entry_score[i]), "bot_score_entry_%d", i);
            snprintf(entry_decided[i], sizeof(entry_decided[i]), "bot_decided_entry_%d", i);
        }
        entry_keys=1;
    }
    for (int i=0; i<32; i++) if (log->bot_n_entry[i]) {
        dict_set(out, entry_n[i], log->bot_n_entry[i]); dict_set(out, entry_score[i], log->bot_score_entry[i]);
        dict_set(out, entry_decided[i], log->bot_decided_entry[i]);
    }
    for (int map=0; map<BOPL_COUNTER_MAPS; map++) if (log->map_fraction[map]) {
        dict_set(out, map_keys[map], log->map_fraction[map]);
    }
    if (log->team_episodes || log->team_kills) {
        dict_set(out, "team_episodes", log->team_episodes); dict_set(out, "team_wins", log->team_wins);
        dict_set(out, "team_kills", log->team_kills);
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
        bopl_pack_sync(vec, 1);  /* the batched forward's packed copy: rewards 0, terminals 1, mask */
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
 * opponent_game_decisions, kills, buttons[4], telemetry[16], ability_activations[39],
 * entity_rows[39], entity_overflow[39], map_matches[124]).
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
/* Style programs (bopl_style.h). Install before creating the vector: every
 * environment created afterwards runs it on its learner decisions. Returns 0,
 * or -1 with the reason in `error`. */
int bopl_style_install(const void *bytes, long length, char *error, int capacity) {
    if (bopl_style_loaded) { bopl_style_free(&bopl_style_program); bopl_style_loaded=0; }
    if (!bytes || length<=0) return 0;   /* uninstall */
    if (bopl_style_load(&bopl_style_program, bytes, (size_t)length, error, (size_t)(capacity>0?capacity:0))) return -1;
    bopl_style_loaded=1;
    return 0;
}
/* The fade: the multiplier on every style payout from the next step on. */
void bopl_style_set_scale(float scale) { bopl_style_scale=scale; }
int bopl_style_terms(void) { return bopl_style_loaded?(int)bopl_style_program.n_terms:0; }
/* Cumulative learner style counters summed over the vector's environments:
 * decisions evaluated, then per term the unscaled payout sum and the number
 * of decisions it paid on (1+2*n_terms doubles). Between rollouts only. */
int bopl_vec_style(const void *gpu_observations, double *out, int capacity) {
    StaticVec *vec=bopl_find_vec(gpu_observations);
    int terms=bopl_style_terms(), size=1+2*terms;
    if (!vec || capacity<size) return -1;
    Env *envs=(Env*)vec->envs;
    memset(out, 0, (size_t)size*sizeof(double));
    for (int i=0; i<vec->size; i++) {
        out[0]+=envs[i].style_decisions;
        for (int t=0; t<terms; t++) { out[1+t]+=envs[i].style_pay[t]; out[1+terms+t]+=envs[i].style_fires[t]; }
    }
    return size;
}
