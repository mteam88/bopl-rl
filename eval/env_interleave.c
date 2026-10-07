/* The trainer's sim cost on a replay corpus, with the trainer's cache
 * behaviour: K worlds are open at once and stepped round-robin, one decision
 * each, the way a pool thread steps its share of the 256 envs. A world that
 * finishes its game takes the next one from the corpus. Sim only (step and
 * sample copy, no writers). Each game's state hash is the same as in
 * eval/env_bench.c with BOPL_ENV_BENCH_SIM_ONLY, and the corpus hash
 * combines them in corpus order, so it does not depend on K.
 *   env_interleave <libbopl_sim.so> <corpus_dir> [max_games] [worlds=23]
 * Run one copy per hardware thread (eval/perf/env_ilv_ab.sh) to add the SMT
 * sharing too. */
#define ENV_BENCH_NO_MAIN
#include "env_bench.c"

typedef struct {
    BoplRl *world;
    BoplStateCopy sample;
    Replay replay;
    int game;         /* corpus index, -1 when idle */
    uint32_t tick;
    uint64_t hash;
} Slot;

int main(int argc, char **argv) {
    if (argc<3 || argc>5) { fprintf(stderr, "env_interleave <libbopl_sim.so> <corpus_dir> [max_games] [worlds=23]\n"); return 2; }
    int max_games=argc>3?atoi(argv[3]):0, k=argc>4?atoi(argv[4]):23;
    const uint32_t warmup=240, limit=10800;
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
    char error[512], log[64]; snprintf(log, sizeof(log), "/tmp/env-ilv-%d.log", (int)getpid());
    for (int s=0; s<k; s++) {
        slots[s].world=bopl_rl_open(argv[1], "/dev/null", log, 30000, error, sizeof(error));
        if (!slots[s].world) { fprintf(stderr, "open failed: %s\n", error); return 1; }
        slots[s].game=-1;
    }
    int next=0, live=0;
    uint64_t decisions=0;
    double sim=0, wall=seconds();
    for (;;) {
        for (int s=0; s<k; s++) {
            Slot *slot=&slots[s];
            if (slot->game<0) {
                if (next>=n) continue;
                char path[4096]; snprintf(path, sizeof(path), "%s/%s", argv[2], names[next]);
                if (replay_decode(path, warmup, limit, &slot->replay)) return 1;
                if (bopl_rl_reset(slot->world, &slot->replay.config)!=BOPL_RL_OK) { fprintf(stderr, "%s: reset failed: %s\n", path, bopl_rl_error(slot->world)); return 1; }
                slot->game=next++; slot->tick=1; live++;
                slot->hash=state_hash(0x243F6A8885A308D3ull, bopl_rl_state(slot->world));
            }
            const BoplRlState *state=bopl_rl_state(slot->world);
            const Replay *replay=&slot->replay;
            if (!(state->header.flags&BOPL_RL_TERMINAL) && warmup+slot->tick-1<replay->tick_count) {
                BoplRlAction actions[PLAYERS];
                memcpy(actions, &replay->actions[(size_t)(warmup+slot->tick-1)*PLAYERS], sizeof(actions));
                uint32_t before=state->header.episode_ticks;
                double t0=seconds();
                if (bopl_rl_step_sampled(slot->world, actions, PLAYERS, 3, 1)) { fprintf(stderr, "%s: step failed: %s\n", names[slot->game], bopl_rl_error(slot->world)); return 1; }
                bopl_state_copy(&slot->sample, bopl_rl_sample(slot->world));
                sim+=seconds()-t0;
                state=bopl_rl_state(slot->world);
                slot->hash=state_hash(state_hash(slot->hash, bopl_rl_sample(slot->world)), state);
                slot->tick+=state->header.episode_ticks-before;
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
    printf("{\"games\": %d, \"worlds\": %d, \"decisions\": %"PRIu64", \"wall_s\": %.2f, \"hash\": \"%016"PRIx64"\", \"sim_us\": %.3f, \"load1\": %.2f}\n",
           n, k, decisions, wall, corpus_hash, sim*1e6/(double)decisions, load1());
    for (int s=0; s<k; s++) bopl_rl_close(slots[s].world);
    return 0;
}
