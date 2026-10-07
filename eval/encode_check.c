/* The GPU v4 row writer (pack mode 4: bopl_rays_v4.cu br4_encode) against
 * the CPU writer (bopl_obs_v4.h bopl_encode_v4_row), bit for bit, over a
 * replay corpus (env_bench's decoder and step loop). Per decision and per
 * writer ablation, two envs see the same delayed state and the same tail:
 * one writes its rows on the CPU; the other ships them as state the way
 * bopl_step_finish does (bopl4_defer, bopl_pack_rows). The GPU writes the
 * shipped rows into a buffer filled with NaN bits, so a float the kernel
 * forgets to write shows as a mismatch, and every float of BOPL4_OBS_SIZE is
 * compared, and the overflow telemetry the host keeps against the CPU
 * writer's.
 *
 *   encode_check <libbopl_sim.so> <corpus_dir> [max_games=0 (all)] [ablations=0,2,8,10,1,4]
 *
 * Exit 0 only with zero mismatches. Build: make encode-check. */
#define ENV_BENCH_NO_MAIN
#include "env_bench.c"
#include <cuda_runtime_api.h>

void bopl4_rays_init(void);
void bopl4_rays_launch(const float *packed, float *dense, int rows, cudaStream_t stream);

enum { BATCH=1024, MAX_ABLATIONS=8 };
typedef struct { int game, ablation, player; uint64_t decision; } RowInfo;
static float *batch_slots, *batch_ref, *batch_out, *d_slots, *d_dense;
static RowInfo batch_info[BATCH];
static int batch_n;
static uint64_t rows_checked, rows_mismatched, floats_mismatched, rows_fallback, launches;
static double kernel_ms;
static long region_bad[6];

static const char *region_name(int i) {
    if (i<BOPL4_OBS_OPPONENT_OFFSET) return i<BOPL4_OBS_SELF_OFFSET?"global":"self";
    if (i<BOPL4_OBS_PREVIOUS_OFFSET) return "opponent";
    if (i<BOPL4_FLAT) return i<BOPL4_OBS_TERRAIN_OFFSET?"previous":"terrain/rays";
    if (i<BOPL4_OBS_PRIVILEGED_OFFSET) return "sets";
    return "privileged";
}
static int region_index(int i) {
    if (i<BOPL4_OBS_OPPONENT_OFFSET) return 0;
    if (i<BOPL4_OBS_PREVIOUS_OFFSET) return 1;
    if (i<BOPL4_OBS_TERRAIN_OFFSET) return 2;
    if (i<BOPL4_FLAT) return 3;
    if (i<BOPL4_OBS_PRIVILEGED_OFFSET) return 4;
    return 5;
}
static void cuda_check(cudaError_t e, const char *what) {
    if (e!=cudaSuccess) { fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); exit(1); }
}
static void flush(char **names) {
    if (!batch_n) return;
    cuda_check(cudaMemcpy(d_slots, batch_slots, (size_t)batch_n*BOPL4_PACK_P*sizeof(float), cudaMemcpyHostToDevice), "slots");
    cuda_check(cudaMemset(d_dense, 0xFF, (size_t)batch_n*BOPL_OBS_STRIDE*sizeof(float)), "memset");
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a, 0);
    bopl4_rays_launch(d_slots, d_dense, batch_n, 0);
    cudaEventRecord(b, 0);
    cuda_check(cudaDeviceSynchronize(), "kernel");
    float ms; cudaEventElapsedTime(&ms, a, b); kernel_ms+=ms; launches++;
    cudaEventDestroy(a); cudaEventDestroy(b);
    cuda_check(cudaMemcpy(batch_out, d_dense, (size_t)batch_n*BOPL_OBS_STRIDE*sizeof(float), cudaMemcpyDeviceToHost), "rows");
    for (int r=0; r<batch_n; r++) {
        const uint32_t *got=(const uint32_t*)(batch_out+(size_t)r*BOPL_OBS_STRIDE), *want=(const uint32_t*)(batch_ref+(size_t)r*BOPL4_OBS_SIZE);
        int bad=0;
        for (int i=0; i<BOPL4_OBS_SIZE; i++) if (got[i]!=want[i]) {
            if (rows_mismatched<5 && bad<6) {
                float g, w; memcpy(&g, &got[i], 4); memcpy(&w, &want[i], 4);
                fprintf(stderr, "mismatch %s decision %"PRIu64" ablation %d player %d: [%d] (%s) gpu %.9g (%08x) cpu %.9g (%08x)\n",
                        names[batch_info[r].game], batch_info[r].decision, batch_info[r].ablation, batch_info[r].player,
                        i, region_name(i), g, got[i], w, want[i]);
            }
            bad++; region_bad[region_index(i)]++;
        }
        rows_checked++;
        if (bad) { rows_mismatched++; floats_mismatched+=(uint64_t)bad; }
    }
    batch_n=0;
}
/* The kernel's latency at the trainer's launch size: `rows` rows of the last batch. */
static void time_launch(int rows) {
    if (batch_n<rows) return;
    cuda_check(cudaMemcpy(d_slots, batch_slots, (size_t)rows*BOPL4_PACK_P*sizeof(float), cudaMemcpyHostToDevice), "slots");
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i=0; i<5; i++) bopl4_rays_launch(d_slots, d_dense, rows, 0);
    cudaEventRecord(a, 0);
    for (int i=0; i<50; i++) bopl4_rays_launch(d_slots, d_dense, rows, 0);
    cudaEventRecord(b, 0);
    cuda_check(cudaDeviceSynchronize(), "timing");
    float ms; cudaEventElapsedTime(&ms, a, b);
    printf("{\"kernel_rows\": %d, \"kernel_us\": %.1f}\n", rows, ms*1e3/50);
    cudaEventDestroy(a); cudaEventDestroy(b);
}
int main(int argc, char **argv) {
    if (argc<3 || argc>5) { fprintf(stderr, "encode_check <libbopl_sim.so> <corpus_dir> [max_games] [ablations]\n"); return 2; }
    int max_games=argc>3?atoi(argv[3]):0;
    int ablations[MAX_ABLATIONS], na=0;
    {
        char list[256]; snprintf(list, sizeof(list), "%s", argc>4?argv[4]:"0,2,8,10,1,4");
        for (char *tok=strtok(list, ","); tok && na<MAX_ABLATIONS; tok=strtok(NULL, ",")) ablations[na++]=atoi(tok);
    }
    uint32_t warmup=240, limit=10800;
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
    if (!n) { fprintf(stderr, "no .rep files in %s\n", argv[2]); return 1; }

    int failures=0;
    bopl4_rays_init();
    batch_slots=malloc((size_t)BATCH*BOPL4_PACK_P*sizeof(float));
    batch_ref=malloc((size_t)BATCH*BOPL4_OBS_SIZE*sizeof(float));
    batch_out=malloc((size_t)BATCH*BOPL_OBS_STRIDE*sizeof(float));
    cuda_check(cudaMalloc((void**)&d_slots, (size_t)BATCH*BOPL4_PACK_P*sizeof(float)), "malloc");
    cuda_check(cudaMalloc((void**)&d_dense, (size_t)BATCH*BOPL_OBS_STRIDE*sizeof(float)), "malloc");

    char error[512], log[64]; snprintf(log, sizeof(log), "/tmp/encode-check-%d.log", (int)getpid());
    BoplRl *world=bopl_rl_open(argv[1], "/dev/null", log, 30000, error, sizeof(error));
    if (!world) { fprintf(stderr, "open failed: %s\n", error); return 1; }
    static float scratch_ref[MAX_ABLATIONS][PLAYERS*BOPL_OBS_STRIDE], scratch_st[MAX_ABLATIONS][PLAYERS*BOPL_OBS_STRIDE];
    static float pack_st[MAX_ABLATIONS][PLAYERS*BOPL4_PACK_P];
    static unsigned char masks[2*MAX_ABLATIONS][PLAYERS][BOPL_LOGITS];
    BoplPuffer *ref[MAX_ABLATIONS], *st[MAX_ABLATIONS];
    for (int k=0; k<na; k++) {
        ref[k]=calloc(1, sizeof(BoplPuffer)); st[k]=calloc(1, sizeof(BoplPuffer));
        env_init(ref[k], world, scratch_ref[k], NULL, masks[2*k], ablations[k]);
        env_init(st[k], world, scratch_st[k], pack_st[k], masks[2*k+1], ablations[k]);
    }
    double cpu_ref=0.0, cpu_state=0.0; uint64_t encodes=0;
    BoplStateCopy delayed[2]={0}, sample={0};
    for (int g=0; g<n; g++) {
        char path[4096]; snprintf(path, sizeof(path), "%s/%s", argv[2], names[g]);
        Replay replay;
        if (replay_decode(path, warmup, limit, &replay)) return 1;
        if (bopl_rl_reset(world, &replay.config)!=BOPL_RL_OK) { fprintf(stderr, "%s: reset failed: %s\n", path, bopl_rl_error(world)); return 1; }
        BoplPuffer *all[2*MAX_ABLATIONS];
        for (int k=0; k<na; k++) { all[2*k]=ref[k]; all[2*k+1]=st[k]; }
        for (int i=0; i<2*na; i++) {
            all[i]->config=replay.config; bopl_written_forget(all[i]);
            memset(all[i]->action_history, 0, sizeof(all[i]->action_history)); memset(all[i]->applied, 0, sizeof(all[i]->applied));
            memset(all[i]->hold_left, 0, sizeof(all[i]->hold_left)); memset(all[i]->hold_run, 0, sizeof(all[i]->hold_run));
        }
        const BoplRlState *state=bopl_rl_state(world);
        for (int i=0; i<2; i++) bopl_state_copy(&delayed[i], state);
        int cursor=0; uint64_t decisions=0; uint32_t tick=1;
        while (!(state->header.flags&BOPL_RL_TERMINAL) && warmup+tick-1<replay.tick_count) {
            BoplRlAction actions[PLAYERS];
            memcpy(actions, &replay.actions[(size_t)(warmup+tick-1)*PLAYERS], sizeof(actions));
            uint32_t before=state->header.episode_ticks;
            if (bopl_rl_step_sampled(world, actions, PLAYERS, 3, 1)) { fprintf(stderr, "%s: step failed: %s\n", path, bopl_rl_error(world)); return 1; }
            bopl_state_copy(&sample, bopl_rl_sample(world));
            state=bopl_rl_state(world);
            tick+=state->header.episode_ticks-before;
            decisions++;
            if (state->header.flags&BOPL_RL_TERMINAL) break;
            const BoplRlState *old=bopl_state_copy_get(&delayed[cursor]);
            for (int k=0; k<na; k++) {
                for (int i=0; i<PLAYERS; i++) ref[k]->applied[i]=st[k]->applied[i]=actions[i];   /* bopl_step_finish_from: before the rows */
                double t0=seconds();
                bopl_encode(ref[k], old, 1);
                double t1=seconds();
                bopl4_defer=1; bopl4_pending_count=0; bopl4_state_count=0;
                bopl_encode(st[k], old, 1);
                double t2=seconds();
                cpu_ref+=t1-t0; cpu_state+=t2-t1; encodes++;
                for (int v=0; v<2; v++) {
                    BoplPuffer *env=v?st[k]:ref[k];
                    for (int i=0; i<PLAYERS; i++) {
                        float *previous=bopl_obs_row(env, i)+bopl_previous_offset(env, i);
                        bopl_previous_action(previous, actions[i], env->hold_left[i], env->hold_run[i]);
                        bopl4_previous_history(previous, env->action_history[i], actions[i]);
                        bopl_write_mask(env, i);
                    }
                    bopl_encode_privileged(env, state, actions);
                }
                bopl4_defer=0;
                bopl_pack_rows(st[k]);
                if (ref[k]->log.observation_overflow!=st[k]->log.observation_overflow
                    || memcmp(ref[k]->counters.entity_rows, st[k]->counters.entity_rows, sizeof(ref[k]->counters.entity_rows))
                    || memcmp(ref[k]->counters.entity_overflow, st[k]->counters.entity_overflow, sizeof(ref[k]->counters.entity_overflow))) {
                    fprintf(stderr, "%s decision %"PRIu64" ablation %d: overflow telemetry differs\n", names[g], decisions, ablations[k]);
                    failures++;
                    ref[k]->log.observation_overflow=st[k]->log.observation_overflow;
                    memcpy(st[k]->counters.entity_rows, ref[k]->counters.entity_rows, sizeof(ref[k]->counters.entity_rows));
                    memcpy(st[k]->counters.entity_overflow, ref[k]->counters.entity_overflow, sizeof(ref[k]->counters.entity_overflow));
                }
                for (int a=0; a<PLAYERS; a++) {
                    const float *slot=pack_st[k]+(size_t)a*BOPL4_PACK_P;
                    if (((const uint32_t*)slot)[0]!=BOPL4_PACK_MODE_STATE) { rows_fallback++; continue; }
                    memcpy(batch_slots+(size_t)batch_n*BOPL4_PACK_P, slot, BOPL4_PACK_P*sizeof(float));
                    memcpy(batch_ref+(size_t)batch_n*BOPL4_OBS_SIZE, scratch_ref[k]+(size_t)a*BOPL_OBS_STRIDE, BOPL4_OBS_SIZE*sizeof(float));
                    batch_info[batch_n]=(RowInfo){g, ablations[k], a, decisions};
                    if (++batch_n==BATCH) {
                        if (launches==8) { time_launch(256); time_launch(384); }
                        flush(names);
                    }
                }
            }
            BoplStateCopy published=delayed[cursor]; delayed[cursor]=sample; sample=published; cursor^=1;
        }
        free(replay.actions);
    }
    flush(names);
    bopl_rl_close(world);
    printf("{\"games\": %d, \"ablations\": %d, \"rows_checked\": %"PRIu64", \"rows_mismatched\": %"PRIu64", \"floats_mismatched\": %"PRIu64", "
           "\"rows_fallback\": %"PRIu64", \"kernel_us_per_%d\": %.1f, \"cpu_rows_us\": %.3f, \"cpu_state_us\": %.3f}\n",
           n, na, rows_checked, rows_mismatched, floats_mismatched, rows_fallback, BATCH, launches?kernel_ms*1e3/(double)launches:0.0,
           cpu_ref*1e6/(double)(encodes?encodes:1), cpu_state*1e6/(double)(encodes?encodes:1));
    if (floats_mismatched)
        printf("{\"bad_global_self\": %ld, \"bad_opponent\": %ld, \"bad_previous\": %ld, \"bad_terrain_rays\": %ld, \"bad_sets\": %ld, \"bad_privileged\": %ld}\n",
               region_bad[0], region_bad[1], region_bad[2], region_bad[3], region_bad[4], region_bad[5]);
    return failures || rows_mismatched ? 1 : 0;
}
