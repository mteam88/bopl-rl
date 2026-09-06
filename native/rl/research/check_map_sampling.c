/* Exercise reset selection without Unity; live geometry/delay checks are separate. */
#define bopl_rl_state sampling_state
#define bopl_rl_reset sampling_reset
#define bopl_rl_map_count sampling_map_count
#define bopl_rl_error sampling_error
#include "../puffer/bopl_puffer.h"
#include <assert.h>

static BoplRlState state;
static BoplRlConfig last;
const BoplRlState *sampling_state(const BoplRl *world) { (void)world; return &state; }
const char *sampling_error(const BoplRl *world) { (void)world; return "sampling test"; }
uint32_t sampling_map_count(const BoplRl *world) { (void)world; return 5; }
int sampling_reset(BoplRl *world, const BoplRlConfig *config) {
    (void)world; last=*config; memset(&state,0,sizeof(state));
    state.header.player_count=2;
    return 0;
}

int main(int argc, char **argv) {
    BoplPuffer env={0};env.num_agents=2;env.config.max_episode_ticks=3600;
    env.observations=calloc(2*BOPL_OBS_SIZE,sizeof(float));
    env.history=calloc(4*BOPL_OBS_SIZE,sizeof(float));
    if (argc==2 && strcmp(argv[1],"overflow")==0) {
        BoplRlEntity entities[BOPL_OBS_ENTITIES+1]={{0}};
        state.header.player_count=2;state.header.entity_count=BOPL_OBS_ENTITIES+1;
        state.entities=entities;
        for (int i=0;i<BOPL_OBS_ENTITIES+1;i++) entities[i].flags=1;
        compute_observations(&env);
        return 2; /* Capacity overflow must fail before any episode is logged. */
    }
    assert(argc==1);
    /* The old single-map default consumes exactly its one episode-seed draw. */
    env.rng=73;BoplPuffer old=env;uint32_t expected=bopl_random(&old);
    c_reset(&env);assert(env.rng==old.rng && last.seed==expected && last.map==0);
    /* A sparse set must never silently broaden to a range or the whole catalog. */
    env.training_map_mask=(1u<<0)|(1u<<2);env.rng=901;
    unsigned int counts[5]={0};uint32_t maps[200],seeds[200];
    for (int i=0;i<200;i++) {
        c_reset(&env);assert(last.map==0 || last.map==2);
        counts[last.map]++;maps[i]=last.map;seeds[i]=last.seed;
    }
    assert(counts[0]>0 && counts[2]>0);
    env.rng=901;
    for (int i=0;i<200;i++) {
        c_reset(&env);assert(last.map==maps[i] && last.seed==seeds[i]);
    }
    env.training_map_mask=7;env.rng=902;memset(counts,0,sizeof(counts));
    for (int i=0;i<200;i++) { c_reset(&env);assert(last.map<3);counts[last.map]++; }
    for (int i=0;i<3;i++) assert(counts[i]>0);
    free(env.observations);free(env.history);
    puts("EXPLICIT_MAP_SELECTION_AND_DEFAULT_RNG_OK");
}
