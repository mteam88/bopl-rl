/* The trainer's sample + finish (bopl_step_sample, bopl_step_finish: the
 * delayed v4 rows shipped or written, the tail, masks, the privileged block,
 * the pack) on a replay corpus, with the trainer's cache behaviour: K worlds
 * are open at once and stepped round-robin, one decision each, like a pool
 * thread stepping its share of the envs, each with its own BoplPuffer (delay
 * copies, written records), its rows in one shared observation buffer and
 * its slots in one shared pack buffer, as the vector lays them out. A world
 * that finishes its game takes the next one from the corpus (the episode
 * start is bopl_reset_observations, c_reset's decision-loop half). The
 * replay's inputs are applied unlatched (BoplStepPhase as bopl_step_begin
 * leaves it for a fresh hold-1 decision).
 *   env_ilv_train <libbopl_sim.so> <corpus_dir> [max_games] [worlds=23]
 * Prints begin_cyc (bopl_step_clear), sample_cyc and finish_cyc (TSC cycles
 * per decision, rdtsc around each call) and the sim's. BOPL_ENV_ILV_HASH=1 folds every decision's
 * slots, masks, rewards and terminals into per-game hashes (the exact gate
 * between two writer builds; off for timing, the hashing costs more than
 * the writer). Pin it (taskset). */
#define ENV_BENCH_NO_MAIN
#include "env_bench.c"
#include <x86intrin.h>

typedef struct {
    BoplRl *world;
    BoplPuffer *env;
    Replay replay;
    int game;         /* corpus index, -1 when idle */
    uint32_t tick;
    uint64_t hash;
    float rewards[PLAYERS], terminals[PLAYERS];
} Slot;

int main(int argc, char **argv) {
    if (argc<3 || argc>5) { fprintf(stderr, "env_ilv_train <libbopl_sim.so> <corpus_dir> [max_games] [worlds=23]\n"); return 2; }
    int max_games=argc>3?atoi(argv[3]):0, k=argc>4?atoi(argv[4]):23;
    const uint32_t warmup=240, limit=10800;
    int hashing=getenv("BOPL_ENV_ILV_HASH")!=NULL;
    DIR *dir=opendir(argv[2]);
    if (!dir) { fprintf(stderr, "%s: cannot open\n", argv[2]); return 1; }
    char **names=NULL; int n=0, cap=0; struct dirent *e;
    while ((e=readdir(dir))) {
        size_t len=strlen(e->d_name);
        if (len<5 || strcmp(e->d_name+len-4, ".rep")) continue;
        if (n==cap) { cap=cap?cap*2:256; names=realloc(names, sizeof(*names)*(size_t)cap); }
        names[n++]=strdup(e->d_name);
    }
    closedir(dir);
    qsort(names, (size_t)n, sizeof(*names), cmp_name);
    if (max_games>0 && n>max_games) n=max_games;
    if (!n || k<1) { fprintf(stderr, "no .rep files in %s (or no worlds)\n", argv[2]); return 1; }
    if (k>n) k=n;

    uint64_t *game_hash=calloc((size_t)n, sizeof(uint64_t));
    Slot *slots=calloc((size_t)k, sizeof(Slot));
    size_t rows=(size_t)k*PLAYERS;
    float *obs=aligned_alloc(64, rows*BOPL_OBS_STRIDE*sizeof(float)), *pack=aligned_alloc(64, rows*BOPL4_PACK_P*sizeof(float));
    unsigned char *masks=aligned_alloc(64, rows*BOPL_LOGITS);
    memset(obs, 0, rows*BOPL_OBS_STRIDE*sizeof(float)); memset(pack, 0, rows*BOPL4_PACK_P*sizeof(float)); memset(masks, 0, rows*BOPL_LOGITS);
    char error[512], log[64]; snprintf(log, sizeof(log), "/tmp/env-ilv-%d.log", (int)getpid());
    for (int s=0; s<k; s++) {
        slots[s].world=bopl_rl_open(argv[1], "/dev/null", log, 30000, error, sizeof(error));
        if (!slots[s].world) { fprintf(stderr, "open failed: %s\n", error); return 1; }
        slots[s].env=calloc(1, sizeof(BoplPuffer));
        slots[s].game=-1;
    }
    int next=0, live=0;
    uint64_t decisions=0, sim=0, sample=0, finish=0, begin=0;
    double wall=seconds();
    for (;;) {
        for (int s=0; s<k; s++) {
            Slot *slot=&slots[s];
            BoplPuffer *env=slot->env;
            if (slot->game<0) {
                if (next>=n) continue;
                char path[4096]; snprintf(path, sizeof(path), "%s/%s", argv[2], names[next]);
                if (replay_decode(path, warmup, limit, &slot->replay)) return 1;
                if (bopl_rl_reset(slot->world, &slot->replay.config)!=BOPL_RL_OK) { fprintf(stderr, "%s: reset failed: %s\n", path, bopl_rl_error(slot->world)); return 1; }
                /* A fresh env per game, as c_reset leaves one (its delay copies kept). */
                BoplStateCopy delayed[2]={env->delayed[0], env->delayed[1]}, sample_copy=env->sample_copy;
                memset(env, 0, sizeof(*env));
                env->delayed[0]=delayed[0]; env->delayed[1]=delayed[1]; env->sample_copy=sample_copy;
                env->players=PLAYERS; env->num_agents=PLAYERS; env->world=slot->world; env->config=slot->replay.config;
                env->observation_delay=1; env->no_autoreset=1; env->repeat=4;
                env->pack_base=obs; env->pack_out=pack; env->pack_row_begin=0; env->pack_row_end=(long)rows;
                for (int i=0; i<PLAYERS; i++) {
                    size_t row=(size_t)s*PLAYERS+(size_t)i;
                    env->layout[i]=BOPL4_LAYOUT_VERSION;
                    env->obs_ptr[i]=obs+row*BOPL_OBS_STRIDE; env->mask_ptr[i]=masks+row*BOPL_LOGITS;
                    env->reward_ptr[i]=&slot->rewards[i]; env->terminal_ptr[i]=&slot->terminals[i];
                }
                bopl_reset_observations(env, bopl_rl_state(slot->world));
                slot->game=next++; slot->tick=1; live++;
                slot->hash=state_hash(0x243F6A8885A308D3ull, bopl_rl_state(slot->world));
            }
            const BoplRlState *state=bopl_rl_state(slot->world);
            const Replay *replay=&slot->replay;
            if (!(state->header.flags&BOPL_RL_TERMINAL) && warmup+slot->tick-1<replay->tick_count) {
                BoplStepPhase phase; memset(&phase, 0, sizeof(phase));
                memcpy(phase.actions, &replay->actions[(size_t)(warmup+slot->tick-1)*PLAYERS], PLAYERS*sizeof(BoplRlAction));
                for (int i=0; i<PLAYERS; i++) phase.fresh[i]=1;
                phase.ticks=state->header.episode_ticks;
                uint64_t tb=__rdtsc();
                bopl_step_clear(env);   /* bopl_step_begin_from's outputs cleared, as before every step */
                uint64_t t0=__rdtsc();
                begin+=t0-tb;
                if (bopl_rl_step_sampled(slot->world, phase.actions, PLAYERS, 3, 1)) { fprintf(stderr, "%s: step failed: %s\n", names[slot->game], bopl_rl_error(slot->world)); return 1; }
                uint64_t t1=__rdtsc();
                bopl_step_sample(env, bopl_rl_sample(slot->world));
                uint64_t t2=__rdtsc();
                bopl_step_finish(env, &phase);
                uint64_t t3=__rdtsc();
                sim+=t1-t0; sample+=t2-t1; finish+=t3-t2;
                state=bopl_rl_state(slot->world);
                if (hashing) {
                    slot->hash=state_hash(slot->hash, state);
                    for (int i=0; i<PLAYERS; i++) {
                        size_t row=(size_t)s*PLAYERS+(size_t)i;
                        const float *slot_words=pack+row*BOPL4_PACK_P;
                        uint32_t mode=((const uint32_t*)slot_words)[0];
                        /* A shipped state's slot is its used words (the rest is stale); any other slot whole. */
                        size_t words=mode==BOPL4_PACK_MODE_STATE?((const Bopl4StateHead*)slot_words)->words:BOPL4_PACK_P;
                        slot->hash=mix_words(slot->hash, slot_words, words*sizeof(float));
                        if (mode!=BOPL4_PACK_MODE_STATE) slot->hash=mix_words(slot->hash, obs+row*BOPL_OBS_STRIDE, BOPL_OBS_STRIDE*sizeof(float));
                        slot->hash=mix_words(slot->hash, masks+row*BOPL_LOGITS, BOPL_LOGITS);
                        slot->hash=mix_words(mix_words(slot->hash, &slot->rewards[i], sizeof(float)), &slot->terminals[i], sizeof(float));
                    }
                    slot->hash=mix_words(mix(slot->hash, (uint64_t)env->log.observation_overflow), &env->counters, sizeof(env->counters));
                }
                slot->tick+=state->header.episode_ticks-phase.ticks;
                decisions++;
                if (!(state->header.flags&BOPL_RL_TERMINAL) && warmup+slot->tick-1<replay->tick_count) continue;
            }
            game_hash[slot->game]=slot->hash;
            free(slot->replay.actions);
            slot->game=-1; live--;
        }
        if (next>=n && live==0) break;
    }
    wall=seconds()-wall;
    uint64_t corpus_hash=0;
    for (int g=0; g<n; g++) corpus_hash=mix(corpus_hash, game_hash[g]);
    printf("{\"games\": %d, \"worlds\": %d, \"decisions\": %"PRIu64", \"wall_s\": %.2f, \"hash\": \"%016"PRIx64"\", \"sim_cyc\": %.1f, \"begin_cyc\": %.1f, \"sample_cyc\": %.1f, \"finish_cyc\": %.1f, \"load1\": %.2f}\n",
           n, k, decisions, wall, hashing?corpus_hash:0, (double)sim/(double)decisions, (double)begin/(double)decisions, (double)sample/(double)decisions, (double)finish/(double)decisions, load1());
    for (int s=0; s<k; s++) bopl_rl_close(slots[s].world);
    return 0;
}
