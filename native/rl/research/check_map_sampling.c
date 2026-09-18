/* Exercise reset selection without Unity; live geometry/delay checks are separate. */
#define bopl_rl_state sampling_state
#define bopl_rl_reset sampling_reset
#define bopl_rl_map_count sampling_map_count
#define bopl_rl_error sampling_error
#define bopl_rl_ability_count sampling_ability_count
#define bopl_rl_reset_pending sampling_reset_pending
#define bopl_rl_reset_begin sampling_reset_begin
#define bopl_rl_reset_finish sampling_reset_finish
#include "../puffer/bopl_puffer.h"
#include <assert.h>

static BoplRlState state;
static BoplRlConfig last;
const BoplRlState *sampling_state(const BoplRl *world) { (void)world; return &state; }
const char *sampling_error(const BoplRl *world) { (void)world; return "sampling test"; }
uint32_t sampling_map_count(const BoplRl *world) { (void)world; return 5; }
uint32_t sampling_ability_count(const BoplRl *world) { (void)world; return BOPL_ABILITY_COUNT; }
int sampling_reset_pending(const BoplRl *world) { (void)world; return 0; }
int sampling_reset_begin(BoplRl *world, const BoplRlConfig *config) { return sampling_reset(world,config); }
int sampling_reset_finish(BoplRl *world) { (void)world; return 0; }
int sampling_reset(BoplRl *world, const BoplRlConfig *config) {
    (void)world; last=*config; memset(&state,0,sizeof(state));
    state.header.player_count=2;
    return 0;
}

int main(int argc, char **argv) {
    BoplPuffer env={0};env.num_agents=2;env.players=2;env.config.players=2;env.config.slots=3;env.config.max_episode_ticks=3600;
    env.world=(BoplRl*)1;env.world_checked=1;
    env.observations=calloc(2*BOPL_OBS_STRIDE,sizeof(float));
    env.scratch=env.observations;
    if (argc==2 && strcmp(argv[1],"overflow")==0) {
        BoplRlEntity entities[BOPL_OBS_ENTITIES+1]={{0}};
        state.header.player_count=2;state.header.entity_count=BOPL_OBS_ENTITIES+1;
        state.entities=entities;
        for (int i=0;i<BOPL_OBS_ENTITIES+1;i++) entities[i].flags=1;
        compute_observations(&env);
        return env.log.observation_overflow==2 ? 2 : 3; /* One count per encoded player row; never fatal. */
    }
    assert(argc==1);
    /* The old single-map default consumes exactly its one episode-seed draw. */
    env.rng=73;BoplPuffer old=env;uint32_t expected=bopl_random(&old);
    c_reset(&env);assert(env.rng==old.rng && last.seed==expected && last.map==0);
    /* A sparse set must never silently broaden to a range or the whole catalog. */
    bopl_map_select(&env.training_map_mask,0);bopl_map_select(&env.training_map_mask,2);env.rng=901;
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
    env.training_map_mask=(BoplMapMask){0};for (uint32_t map=0;map<3;map++) bopl_map_select(&env.training_map_mask,map);env.rng=902;memset(counts,0,sizeof(counts));
    for (int i=0;i<200;i++) { c_reset(&env);assert(last.map<3);counts[last.map]++; }
    for (int i=0;i<3;i++) assert(counts[i]>0);
    /* Ability 38 survives the native pool sampler; no-dup draws without replacement. */
    env.loadout_mode=BOPL_LOADOUT_POOL;env.loadout_pool=1ull<<38;env.rng=903;
    c_reset(&env);
    for (int player=0;player<2;player++) for (int slot=0;slot<3;slot++) assert(last.player[player].abilities[slot]==38);
    env.loadout_pool=(1ull<<1)|(1ull<<2)|(1ull<<38);env.loadout_no_dup=1;env.rng=904;
    c_reset(&env);
    for (int player=0;player<2;player++) {
        uint64_t seen=0;
        for (int slot=0;slot<3;slot++) { uint32_t id=last.player[player].abilities[slot];assert(id==1||id==2||id==38);assert(!(seen&(1ull<<id)));seen|=1ull<<id; }
    }
    /* Explicit lineups and mirror share are selected in C, not Python. */
    env.loadout_mode=BOPL_LOADOUT_SET;env.loadout_set_count=1;env.loadout_no_dup=0;
    uint8_t lineup[6]={4,4,38,2,14,5};memcpy(env.loadout_set[0],lineup,6);env.mirror_share_ppm=0;env.rng=905;
    c_reset(&env);for (int player=0;player<2;player++) for (int slot=0;slot<3;slot++) assert(last.player[player].abilities[slot]==lineup[player*3+slot]);
    env.mirror_share_ppm=1000000;env.rng=906;c_reset(&env);
    for (int slot=0;slot<3;slot++) assert(last.player[0].abilities[slot]==last.player[1].abilities[slot]);
    /* Every active Bow slot participates in forced-release masking. */
    BoplRlSlime slime={0};state.header.player_count=2;state.header.slime_count=1;state.slimes=&slime;
    state.players[0].id=1;slime.owner=1;slime.abilities[0]=4;slime.abilities[1]=2;slime.abilities[2]=4;
    slime.ability_flags[0]=slime.ability_flags[1]=slime.ability_flags[2]=1;
    assert(bopl_bow_buttons(&env,0)==((1u<<1)|(1u<<3)));
    unsigned char mask[BOPL_LOGITS];env.scratch=NULL;env.mask_ptr[0]=mask;env.hold_run[0][1]=env.hold_run[0][3]=1;
    bopl_write_mask(&env,0);
    assert(!mask[BOPL_ACT_AIM+2] && !mask[BOPL_ACT_AIM+8] && !mask[BOPL_ACT_AIM+10] && mask[BOPL_ACT_AIM+4]);
    free(env.observations);
    puts("MAP_123_ABILITY_38_LOADOUT_AND_BOW_MASK_OK");
}
