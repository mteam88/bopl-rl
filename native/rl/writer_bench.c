/* Observation-writer microbenchmark and sim ticks/s on one core (THROUGHPUT_V4.md).
 *
 * Phase 1 plays 2-player episodes on the Rust sim with random held inputs (the
 * benchmark.c convention: 4-tick decisions, new inputs every few decisions,
 * episodes reset at the terminal), timing bopl_rl_step alone -> ticks/s, and
 * records every k-th decision's state (bopl_state_copy) until `states` are held.
 * Phase 2 replays the recorded states through the writers, interleaving the
 * variants per repeat so a drifting host load hits them alike:
 *   v3            layout 3 (bopl_encode_v3_row)
 *   v4-aim        layout 4, writer bits 8 (production: fan + terrain + aim/LOS casts)
 *   v4-aim-only   bits 10 (aim/LOS casts, no fan, no terrain block)
 *   v4-no-rays    bits 2  (no geometry channel at all)
 *   v4-priv       bopl4_privileged, the critic block written once more per step from the undelayed state
 * so geometry = v4-aim - v4-aim-only, casts = v4-aim-only - v4-no-rays, and
 * the v4 global/self/opponent/set rows with their relational columns =
 * v4-no-rays - v3. bopl4_relation is timed on its own (relational bit 1 zeroes
 * after computing, so no ablation isolates it). Row counts per set of the
 * recorded states (valid rows of player 0's v4 row) are reported against the
 * trainer's packed budget (bopl_v4.cu DEFAULT_BUDGET 6,12,20,8) and capacity.
 *
 *   writer_bench <libbopl_sim.so> [ticks=130000] [loadout=default|random] [repeats=5] [states=4096] [seed=37]
 *
 * Episodes cycle maps 0, 1, 2 (bench_shape.sh's --training-maps).
 * default = the training loadout 2/4/14 (Dash, Bow, Platform; binding.c);
 * random = binding.c's random_loadouts draw per episode. Pin and nice it
 * (taskset -c 11 nice -n 19), Q32 build (just build-q32). One JSON line per
 * result plus a summary line; the load average travels with every line.
 * Build: make -C native/rl writer-bench. */
#define _GNU_SOURCE
#include "puffer/bopl_puffer.h"
#include <inttypes.h>
#include <time.h>

static double seconds(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec+t.tv_nsec/1e9;
}
static uint32_t random_u32(uint32_t *s) { uint32_t x=*s; x^=x<<13; x^=x>>17; x^=x<<5; return *s=x; }
static double load1(void) { double l[1]; return getloadavg(l, 1)==1?l[0]:-1.0; }
static int cmp_double(const void *a, const void *b) { double x=*(const double*)a, y=*(const double*)b; return (x>y)-(x<y); }
static double median(double *v, int n) { qsort(v, (size_t)n, sizeof(*v), cmp_double); return n&1?v[n/2]:0.5*(v[n/2-1]+v[n/2]); }
static int cmp_int(const void *a, const void *b) { return *(const int*)a-*(const int*)b; }
static void quantiles(int *v, int n, int *p50, int *p90, int *max) {
    qsort(v, (size_t)n, sizeof(*v), cmp_int);
    *p50=v[n/2]; *p90=v[(int)(n*0.9)]; *max=v[n-1];
}

typedef struct { const char *name; int layout, ablate; } Variant;
static const Variant VARIANTS[]={{"v3", 3, -1}, {"v4-aim", 4, 8}, {"v4-aim-only", 4, 10}, {"v4-no-rays", 4, 2}};
enum { N_VARIANTS=sizeof(VARIANTS)/sizeof(*VARIANTS), PLAYERS=2, TICKS_PER_DECISION=4 };

int main(int argc, char **argv) {
    if (argc<2 || argc>7) { fprintf(stderr, "writer_bench <libbopl_sim.so> [ticks] [default|random] [repeats] [states] [seed]\n"); return 2; }
    uint64_t wanted=argc>2?strtoull(argv[2], NULL, 10):130000;
    int random_loadouts=argc>3 && !strcmp(argv[3], "random");
    int repeats=argc>4?atoi(argv[4]):5, n_states=argc>5?atoi(argv[5]):4096;
    uint32_t seed=argc>6?(uint32_t)strtoul(argv[6], NULL, 10):37;
    const char *loadout=random_loadouts?"random":"default";
    char error[512];
    char log[64]; snprintf(log, sizeof(log), "/tmp/writer-bench-%d.log", (int)getpid());
    BoplRl *world=bopl_rl_open(argv[1], "/dev/null", log, 30000, error, sizeof(error));
    if (!world) { fprintf(stderr, "open failed: %s\n", error); return 1; }
    if (bopl_rl_ability_count(world)!=BOPL_ABILITY_COUNT) { fprintf(stderr, "ability catalog %u != %d\n", bopl_rl_ability_count(world), BOPL_ABILITY_COUNT); return 1; }

    /* ---- phase 1: play, time the sim, record states ---- */
    BoplRlConfig config; bopl_rl_default_config(&config, PLAYERS);
    config.seed=seed; config.stocks=0; config.slots=3; config.max_episode_ticks=7200;   /* the trainer's episode (binding.c defaults, --episode-ticks 7200) */
    uint32_t rng=seed*2654435761u^0x9E3779B9u;
    for (int i=0; i<PLAYERS; i++) { config.player[i].abilities[0]=2; config.player[i].abilities[1]=4; config.player[i].abilities[2]=14; }
    if (random_loadouts) for (int i=0; i<PLAYERS; i++) for (uint32_t j=0; j<config.slots; j++)
        config.player[i].abilities[j]=1+random_u32(&rng)%(BOPL_ABILITY_COUNT-1);
    if (bopl_rl_reset(world, &config)!=BOPL_RL_OK) { fprintf(stderr, "reset failed: %s\n", bopl_rl_error(world)); return 1; }
    BoplStateCopy *states=calloc((size_t)n_states, sizeof(*states));
    if (!states) return 1;
    uint64_t decisions_total=wanted/TICKS_PER_DECISION, stride=decisions_total/(uint64_t)n_states; if (!stride) stride=1;
    BoplRlAction actions[PLAYERS]={{0}};
    uint64_t ticks=0, decisions=0, resets=1; int recorded=0;
    uint32_t max_entities=0, max_slimes=0;
    double step_seconds=0, reset_seconds=0;
    while (ticks<wanted) {
        const BoplRlState *state=bopl_rl_state(world);
        if (state->header.flags&BOPL_RL_TERMINAL) {
            config.seed=random_u32(&rng); config.map=(uint32_t)(resets%3);   /* the bench's training maps 0, 1, 2 in turn */
            if (random_loadouts) for (int i=0; i<PLAYERS; i++) for (uint32_t j=0; j<config.slots; j++)
                config.player[i].abilities[j]=1+random_u32(&rng)%(BOPL_ABILITY_COUNT-1);
            double t=seconds();
            if (bopl_rl_reset(world, &config)!=BOPL_RL_OK) { fprintf(stderr, "episode reset failed: %s\n", bopl_rl_error(world)); return 1; }
            reset_seconds+=seconds()-t; resets++;
            state=bopl_rl_state(world);
        }
        if (decisions%4==0) for (int i=0; i<PLAYERS; i++) {   /* held ~1 s: aim, jump/abilities, some movement */
            uint32_t r=random_u32(&rng);
            actions[i].angle=(uint8_t)(1+(r>>16)%255); actions[i].buttons=(uint16_t)((r&15)|((r>>4)&(64|128|256|512)));
        }
        uint32_t before=state->header.episode_ticks;
        double t=seconds();
        if (bopl_rl_step(world, actions, PLAYERS, TICKS_PER_DECISION)!=BOPL_RL_OK) { fprintf(stderr, "step failed: %s\n", bopl_rl_error(world)); return 1; }
        step_seconds+=seconds()-t;
        state=bopl_rl_state(world);
        ticks+=state->header.episode_ticks-before; decisions++;
        if (state->header.entity_count>max_entities) max_entities=state->header.entity_count;
        if (state->header.slime_count>max_slimes) max_slimes=state->header.slime_count;
        if (recorded<n_states && decisions%stride==0) bopl_state_copy(&states[recorded++], state);
    }
    printf("{\"bench\": \"sim\", \"loadout\": \"%s\", \"players\": %d, \"ticks\": %"PRIu64", \"decisions\": %"PRIu64", \"resets\": %"PRIu64
           ", \"ticks_per_second\": %.0f, \"ticks_per_second_with_resets\": %.0f, \"us_per_decision\": %.2f, \"max_entities\": %u, \"max_slimes\": %u, \"recorded_states\": %d, \"load1\": %.2f}\n",
           loadout, PLAYERS, ticks, decisions, resets, ticks/step_seconds, ticks/(step_seconds+reset_seconds), step_seconds*1e6/decisions,
           max_entities, max_slimes, recorded, load1());
    fflush(stdout);
    bopl_rl_close(world);

    /* ---- phase 2: the writers over the recorded states ---- */
    static float scratch[BOPL_MAX_PLAYERS*BOPL_OBS_STRIDE];
    BoplPuffer env={0};
    env.config=config; env.players=PLAYERS; env.num_agents=PLAYERS; env.scratch=scratch;
    for (int i=0; i<PLAYERS; i++) env.applied[i].angle=(uint8_t)(64+61*i);   /* an aimed decision, so the aim ray is cast (98% of real rows) */
    /* The vector march (bopl4_march_avx512) against the scalar one
     * (bopl4_march_scalar: bot_raycast's steps): every production row (bits
     * 8) of every recorded state, both players, bit for bit. */
    {
        static float reference[BOPL_MAX_PLAYERS*BOPL_OBS_STRIDE];
        long mismatches=0, rows=0;
        for (int i=0; i<PLAYERS; i++) { env.layout[i]=BOPL4_LAYOUT_VERSION; bopl_set_ablate_of(&env, i, 8); }
        for (int s=0; s<recorded; s++) {
            const BoplRlState *state=bopl_state_copy_get(&states[s]);
            bopl4_march_scalar_only=1; bopl_written_forget(&env); compute_observations_from(&env, state);
            memcpy(reference, scratch, sizeof(float)*(size_t)PLAYERS*BOPL_OBS_STRIDE);
            bopl4_march_scalar_only=0; bopl_written_forget(&env); compute_observations_from(&env, state);
            for (int i=0; i<PLAYERS; i++) {
                rows++;
                if (memcmp(reference+(size_t)i*BOPL_OBS_STRIDE, scratch+(size_t)i*BOPL_OBS_STRIDE, sizeof(float)*BOPL4_OBS_SIZE)) mismatches++;
            }
        }
        printf("{\"bench\": \"march-check\", \"loadout\": \"%s\", \"rows\": %ld, \"mismatches\": %ld, \"load1\": %.2f}\n", loadout, rows, mismatches, load1());
        fflush(stdout);
        if (mismatches) { fprintf(stderr, "vector march changed %ld of %ld rows\n", mismatches, rows); return 1; }
    }
    double us[N_VARIANTS+1][64]; if (repeats>64) repeats=64;
    for (int r=0; r<repeats; r++) {
        for (int k=0; k<=N_VARIANTS; k++) {
            int v=(k+r)%(N_VARIANTS+1);   /* rotate the order per repeat */
            double t=seconds();
            if (v<N_VARIANTS) {
                for (int i=0; i<PLAYERS; i++) { env.layout[i]=VARIANTS[v].layout==4?BOPL4_LAYOUT_VERSION:3; bopl_set_ablate_of(&env, i, VARIANTS[v].ablate); }
                bopl_written_forget(&env);
                for (int s=0; s<recorded; s++) compute_observations_from(&env, bopl_state_copy_get(&states[s]));
            } else {
                for (int i=0; i<PLAYERS; i++) env.layout[i]=BOPL4_LAYOUT_VERSION;
                for (int s=0; s<recorded; s++) bopl_encode_privileged(&env, bopl_state_copy_get(&states[s]), NULL);
            }
            us[v][r]=(seconds()-t)*1e6/((double)recorded*PLAYERS);
        }
    }
    double med[N_VARIANTS+1];
    for (int v=0; v<=N_VARIANTS; v++) {
        double lo=us[v][0], hi=us[v][0];
        for (int r=1; r<repeats; r++) { if (us[v][r]<lo) lo=us[v][r]; if (us[v][r]>hi) hi=us[v][r]; }
        med[v]=median(us[v], repeats);
        printf("{\"bench\": \"writer\", \"loadout\": \"%s\", \"variant\": \"%s\", \"layout\": %d, \"ablate\": %d, \"us_per_row\": %.3f, \"min\": %.3f, \"max\": %.3f, \"repeats\": %d, \"states\": %d, \"load1\": %.2f}\n",
               loadout, v<N_VARIANTS?VARIANTS[v].name:"v4-priv", v<N_VARIANTS?VARIANTS[v].layout:4, v<N_VARIANTS?VARIANTS[v].ablate:8, med[v], lo, hi, repeats, recorded, load1());
    }
    /* bopl4_relation alone: the 12 relational columns of one row, plain and ballistic. */
    {
        static float out[BOPL4_RELATION]; volatile float sink=0; uint32_t s=seed; double t, ns[2];
        for (int ballistic=0; ballistic<2; ballistic++) {
            t=seconds();
            for (int i=0; i<1000000; i++) {
                double dx=(random_u32(&s)%2000)/10.0-100, dy=(random_u32(&s)%2000)/10.0-100, vx=(random_u32(&s)%200)-100, vy=(random_u32(&s)%200)-100;
                bopl4_relation(out, dx, dy, vx, vy, 0.5, ballistic); sink+=out[8];
            }
            ns[ballistic]=(seconds()-t)*1e3;
        }
        printf("{\"bench\": \"relation\", \"ns_per_call\": %.1f, \"ns_per_call_ballistic\": %.1f, \"load1\": %.2f, \"sink\": %.3g}\n", ns[0], ns[1], load1(), (double)sink);
    }
    /* Valid rows per set of player 0's production row over the recorded states. */
    {
        static const struct { const char *name; int offset, rows, width, budget; } sets[4]={
            {"players", BOPL4_OBS_PLAYER_OFFSET, BOPL4_PLAYERS, BOPL4_PLAYER_FEATURES, 6}, {"slots", BOPL4_OBS_SLOT_OFFSET, BOPL4_SLOTS, BOPL4_SLOT_FEATURES, 12},
            {"entities", BOPL4_OBS_ENTITY_OFFSET, BOPL4_ENTITIES, BOPL4_ENTITY_FEATURES, 20}, {"slimes", BOPL4_OBS_SLIME_OFFSET, BOPL4_SLIMES, BOPL4_SLIME_FEATURES, 8}};
        int *counts=malloc(sizeof(int)*(size_t)recorded);
        for (int i=0; i<PLAYERS; i++) { env.layout[i]=BOPL4_LAYOUT_VERSION; bopl_set_ablate_of(&env, i, 8); }
        bopl_written_forget(&env);
        for (int k=0; k<4; k++) {
            double sum=0;
            for (int s=0; s<recorded; s++) {
                compute_observations_from(&env, bopl_state_copy_get(&states[s]));
                int n=0; for (int r=0; r<sets[k].rows; r++) n+=scratch[sets[k].offset+r*sets[k].width]!=0.0f;
                counts[s]=n; sum+=n;
            }
            int p50, p90, max; quantiles(counts, recorded, &p50, &p90, &max);
            printf("{\"bench\": \"rows\", \"loadout\": \"%s\", \"set\": \"%s\", \"mean\": %.2f, \"p50\": %d, \"p90\": %d, \"max\": %d, \"budget\": %d, \"capacity\": %d}\n",
                   loadout, sets[k].name, sum/recorded, p50, p90, max, sets[k].budget, sets[k].rows);
        }
        free(counts);
    }
    printf("{\"bench\": \"summary\", \"loadout\": \"%s\", \"v3_us\": %.3f, \"v4_aim_us\": %.3f, \"v4_over_v3\": %.2f, \"geometry_us\": %.3f, \"casts_us\": %.3f, \"v4_rows_over_v3_us\": %.3f, \"privileged_us\": %.3f, \"load1\": %.2f}\n",
           loadout, med[0], med[1], med[1]/med[0], med[1]-med[2], med[2]-med[3], med[3]-med[0], med[4], load1());
    for (int s=0; s<recorded; s++) bopl_state_copy_free(&states[s]);
    free(states);
    return 0;
}
