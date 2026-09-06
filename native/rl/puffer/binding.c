#include "bopl_puffer.h"
#define OBS_SIZE BOPL_OBS_SIZE
#define NUM_ATNS 2
#define ACT_SIZES {256, 16}
#define OBS_TENSOR_T FloatTensor
#define Env BoplPuffer
/* The upstream constructor is wrapped below so grouped stepping can find the
 * vector from Python; upstream code and the pybind module keep calling
 * create_static_vec and see identical behaviour. */
#define create_static_vec bopl_upstream_create_static_vec
#define static_vec_close bopl_upstream_static_vec_close
#include "vecenv.h"
#undef create_static_vec
#undef static_vec_close

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

/* ---------------------------------------------------------------------------
 * Grouped asynchronous stepping (research/vector.py `groups`).
 *
 * The default path is the pinned upstream synchronous gpu_vec_step; nothing
 * below runs unless bopl_groups_create is called. A group is a contiguous
 * range of environments (physical index order, so worker w stays agents 2w and
 * 2w+1). Each group owns one pthread, one non-blocking CUDA stream and a
 * private device action buffer. The trainer:
 *   1. copies the group's chosen actions into the group's device buffer on the
 *      torch stream (bopl_group_actions_ptr), then calls bopl_group_launch with
 *      a CUDA event recorded after that copy on the torch stream;
 *   2. later calls bopl_group_wait, after which the group's slices of
 *      gpu_observations, gpu_rewards, gpu_terminals, the host terminals and the
 *      telemetry buffer are final and visible to the calling (torch) stream
 *      through cudaStreamWaitEvent on the done event.
 * Group threads only ever touch their own environments and buffer slices, so
 * the per-step observation/reward/reset semantics of c_step are unchanged.
 * ------------------------------------------------------------------------- */
typedef struct CUevent_st* cudaEvent_t;
extern cudaError_t cudaEventCreateWithFlags(cudaEvent_t*, unsigned int);
extern cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t);
extern cudaError_t cudaEventDestroy(cudaEvent_t);
extern cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned int);
extern cudaError_t cudaStreamDestroy(cudaStream_t);
extern cudaError_t cudaGetLastError(void);
#define cudaEventDisableTiming 2

typedef struct {
    StaticVec *vec;
    int index, env_start, env_count, agent_start, agent_count, workers;
    cudaStream_t stream;
    cudaEvent_t ready, done;   /* ready: actions copy finished (torch); done: outputs uploaded (ours) */
    float *gpu_actions;        /* private device copy, agent_count*NUM_ATNS floats */
    pthread_t thread;
    pthread_mutex_t lock;
    pthread_cond_t wake;
    int state;                 /* 0 idle, 1 launched, 2 shutdown, 3 finished but not yet collected by wait */
    int jitter_us;             /* per-group copy of BOPL_GROUP_JITTER_US, fixed before the thread starts */
    unsigned long launches;
    double env_ms, total_ms;
} BoplGroup;

typedef struct {
    StaticVec *vec;
    int count;
    BoplGroup *groups;
} BoplGroups;

static double bopl_now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec*1000.0+t.tv_nsec/1e6;
}
static void bopl_cuda_check(cudaError_t error, const char *what) {
    if (error!=cudaSuccess) { fprintf(stderr, "Bopl group CUDA failure in %s: %s\n", what, cudaGetErrorString(error)); abort(); }
}
/* Test hook: BOPL_GROUP_JITTER_US=<max> sleeps a pseudo-random time before and
 * after each group's environment step, to shake up interleavings. Unset in
 * training; correctness must not depend on it. */
static void bopl_jitter(BoplGroup *g) {
    if (g->jitter_us<=0) return;
    unsigned int x=(unsigned int)(g->launches*2654435761u+(unsigned)g->index*40503u+1u);
    x^=x<<13; x^=x>>17; x^=x<<5;
    usleep(x%(unsigned)g->jitter_us);
}
static void bopl_group_run(BoplGroup *g) {
    StaticVec *vec=g->vec; Env *envs=(Env*)vec->envs;
    double started=bopl_now_ms();
    bopl_jitter(g);
    bopl_cuda_check(cudaStreamWaitEvent(g->stream, g->ready, 0), "wait actions");
    bopl_cuda_check(cudaMemcpyAsync(&vec->actions[g->agent_start*NUM_ATNS], g->gpu_actions,
        (size_t)g->agent_count*NUM_ATNS*sizeof(float), cudaMemcpyDeviceToHost, g->stream), "actions D2H");
    bopl_cuda_check(cudaStreamSynchronize(g->stream), "actions sync");
    double env_started=bopl_now_ms();
    /* c_step clears its own rewards/terminals slice before writing them. */
    #pragma omp parallel for schedule(static) num_threads(g->workers)
    for (int i=g->env_start; i<g->env_start+g->env_count; i++) c_step(&envs[i]);
    g->env_ms+=bopl_now_ms()-env_started;
    bopl_jitter(g);
    size_t obs_bytes=(size_t)g->agent_count*OBS_SIZE*obs_element_size();
    bopl_cuda_check(cudaMemcpyAsync((char*)vec->gpu_observations+(size_t)g->agent_start*OBS_SIZE*obs_element_size(),
        (char*)vec->observations+(size_t)g->agent_start*OBS_SIZE*obs_element_size(), obs_bytes, cudaMemcpyHostToDevice, g->stream), "obs H2D");
    bopl_cuda_check(cudaMemcpyAsync(&vec->gpu_rewards[g->agent_start], &vec->rewards[g->agent_start],
        (size_t)g->agent_count*sizeof(float), cudaMemcpyHostToDevice, g->stream), "rewards H2D");
    bopl_cuda_check(cudaMemcpyAsync(&vec->gpu_terminals[g->agent_start], &vec->terminals[g->agent_start],
        (size_t)g->agent_count*sizeof(float), cudaMemcpyHostToDevice, g->stream), "terminals H2D");
    bopl_cuda_check(cudaEventRecord(g->done, g->stream), "done event");
    /* Host-side completion: the trainer reads host terminals/telemetry after wait. */
    bopl_cuda_check(cudaStreamSynchronize(g->stream), "outputs sync");
    g->total_ms+=bopl_now_ms()-started; g->launches++;
}
static void *bopl_group_thread(void *arg) {
    BoplGroup *g=arg;
    for (;;) {
        pthread_mutex_lock(&g->lock);
        while (g->state==0 || g->state==3) pthread_cond_wait(&g->wake, &g->lock);
        int state=g->state;
        pthread_mutex_unlock(&g->lock);
        if (state==2) return NULL;
        bopl_group_run(g);
        pthread_mutex_lock(&g->lock);
        g->state=3; pthread_cond_broadcast(&g->wake);
        pthread_mutex_unlock(&g->lock);
    }
}
/* Splits the vector's environments into `count` contiguous groups. The
 * environment count must divide evenly so every group has the same batch size
 * (the trainer slices tensors by agent ranges). Returns an opaque handle. */
/* The pybind VecEnv does not expose its StaticVec; Python identifies the
 * vector by the device observation pointer it already wraps. Vectors register
 * themselves here from my_vec_register, called by the trainer after creation. */
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
/* Paired with registration: the pybind close calls this wrapper, so a freed
 * vector can never be found by bopl_groups_open. Groups must be closed first. */
void static_vec_close(StaticVec *vec) {
    pthread_mutex_lock(&bopl_registry_lock);
    for (int i=0; i<BOPL_MAX_VECS; i++) if (bopl_registered[i]==vec) bopl_registered[i]=NULL;
    pthread_mutex_unlock(&bopl_registry_lock);
    bopl_upstream_static_vec_close(vec);
}
BoplGroups *bopl_groups_create(StaticVec *vec, int count, int threads) {
    if (!vec || !vec->gpu || vec->buffers!=1 || count<1 || vec->size%count) return NULL;
    BoplGroups *groups=calloc(1, sizeof(*groups));
    if (!groups) abort();
    groups->vec=vec; groups->count=count; groups->groups=calloc((size_t)count, sizeof(BoplGroup));
    if (!groups->groups) abort();
    const char *jitter=getenv("BOPL_GROUP_JITTER_US");
    int jitter_us=jitter?atoi(jitter):0;
    Env *envs=(Env*)vec->envs;
    int per_group=vec->size/count, agent=0;
    for (int i=0; i<count; i++) {
        BoplGroup *g=&groups->groups[i];
        g->vec=vec; g->index=i; g->jitter_us=jitter_us; g->env_start=i*per_group; g->env_count=per_group; g->agent_start=agent;
        for (int e=g->env_start; e<g->env_start+g->env_count; e++) agent+=envs[e].num_agents;
        g->agent_count=agent-g->agent_start;
        g->workers=threads/count>0?threads/count:1;
        bopl_cuda_check(cudaStreamCreateWithFlags(&g->stream, cudaStreamNonBlocking), "stream");
        bopl_cuda_check(cudaEventCreateWithFlags(&g->ready, cudaEventDisableTiming), "ready event");
        bopl_cuda_check(cudaEventCreateWithFlags(&g->done, cudaEventDisableTiming), "done event");
        bopl_cuda_check(cudaMalloc((void**)&g->gpu_actions, (size_t)g->agent_count*NUM_ATNS*sizeof(float)), "actions");
        bopl_cuda_check(cudaMemset(g->gpu_actions, 0, (size_t)g->agent_count*NUM_ATNS*sizeof(float)), "actions clear");
        pthread_mutex_init(&g->lock, NULL); pthread_cond_init(&g->wake, NULL);
        if (pthread_create(&g->thread, NULL, bopl_group_thread, g)) abort();
    }
    if (agent!=vec->total_agents) abort();
    return groups;
}
int bopl_groups_count(const BoplGroups *groups) { return groups?groups->count:0; }
int bopl_group_agent_start(const BoplGroups *groups, int i) { return groups->groups[i].agent_start; }
int bopl_group_agent_count(const BoplGroups *groups, int i) { return groups->groups[i].agent_count; }
void *bopl_group_actions_ptr(const BoplGroups *groups, int i) { return groups->groups[i].gpu_actions; }
/* Records the group's ready event on `stream` (the torch stream that produced
 * the group's action copy) and wakes the group thread. Not reentrant per group. */
void bopl_group_launch(BoplGroups *groups, int i, cudaStream_t stream) {
    BoplGroup *g=&groups->groups[i];
    pthread_mutex_lock(&g->lock);
    if (g->state!=0) { fprintf(stderr, "Bopl group %d launched twice without wait\n", i); abort(); }
    bopl_cuda_check(cudaEventRecord(g->ready, stream), "ready record");
    g->state=1; pthread_cond_broadcast(&g->wake);
    pthread_mutex_unlock(&g->lock);
}
/* Blocks until the group's step has finished on the host, then makes `stream`
 * wait for the device uploads. Returns 1 when this call collected a launched
 * step (exactly one wait per launch does), 0 when there was nothing to collect. */
int bopl_group_wait(BoplGroups *groups, int i, cudaStream_t stream) {
    BoplGroup *g=&groups->groups[i];
    pthread_mutex_lock(&g->lock);
    while (g->state==1) pthread_cond_wait(&g->wake, &g->lock);
    int collected=g->state==3;
    if (collected) g->state=0;
    pthread_mutex_unlock(&g->lock);
    /* Exactly one wait per launch collects it: the caller's stream is ordered
     * after the group's uploads, and the group may be launched again. */
    if (collected) bopl_cuda_check(cudaStreamWaitEvent(stream, g->done, 0), "done wait");
    return collected;
}
int bopl_group_pending(BoplGroups *groups, int i) {
    BoplGroup *g=&groups->groups[i];
    pthread_mutex_lock(&g->lock); int pending=g->state==1||g->state==3; pthread_mutex_unlock(&g->lock);
    return pending;
}
double bopl_group_env_ms(const BoplGroups *groups, int i) { return groups->groups[i].env_ms; }
double bopl_group_total_ms(const BoplGroups *groups, int i) { return groups->groups[i].total_ms; }
unsigned long bopl_group_launches(const BoplGroups *groups, int i) { return groups->groups[i].launches; }
/* Waits for every pending group, then joins threads and frees resources. The
 * StaticVec itself stays owned by the caller. */
void bopl_groups_close(BoplGroups *groups) {
    if (!groups) return;
    for (int i=0; i<groups->count; i++) {
        BoplGroup *g=&groups->groups[i];
        pthread_mutex_lock(&g->lock);
        while (g->state==1) pthread_cond_wait(&g->wake, &g->lock);
        g->state=2; pthread_cond_broadcast(&g->wake);
        pthread_mutex_unlock(&g->lock);
        pthread_join(g->thread, NULL);
        cudaStreamSynchronize(g->stream);
        cudaEventDestroy(g->ready); cudaEventDestroy(g->done); cudaStreamDestroy(g->stream); cudaFree(g->gpu_actions);
        pthread_mutex_destroy(&g->lock); pthread_cond_destroy(&g->wake);
    }
    free(groups->groups); free(groups);
}
/* Python entry points (ctypes on the loaded _C module). */
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
BoplGroups *bopl_groups_open(const void *gpu_observations, int count, int threads) {
    StaticVec *vec=bopl_find_vec(gpu_observations);
    return vec?bopl_groups_create(vec, count, threads):NULL;
}
