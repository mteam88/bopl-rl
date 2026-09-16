/* Synthetic crowded layout-v3 fixture: duplicated grenades, clone inventories,
 * rope collapse, invisibility, live disabled smoke and a far pickup. */
#define bopl_rl_state observation_test_state
#include "../puffer/bopl_puffer.h"
#undef bopl_rl_state
#include <assert.h>

#define Q(value) ((int64_t)(value) << 32)
static BoplRlState state;
const BoplRlState *observation_test_state(const BoplRl *world) { (void)world; return &state; }

static float *entity_with_source(float *obs, int source) {
    for (int i=0; i<BOPL_OBS_ENTITIES; i++) {
        float *row=obs+BOPL_OBS_ENTITY_OFFSET+(size_t)i*BOPL_ENTITY_FEATURES;
        if (row[0] && row[BOPL_ENTITY_ABILITY_ID]==source) return row;
    }
    return NULL;
}
static int entity_source_count(float *obs, int source) {
    int count=0;
    for (int i=0; i<BOPL_OBS_ENTITIES; i++) {
        float *row=obs+BOPL_OBS_ENTITY_OFFSET+(size_t)i*BOPL_ENTITY_FEATURES;
        if (row[0] && row[BOPL_ENTITY_ABILITY_ID]==source) count++;
    }
    return count;
}
static int valid_rows(float *obs, int offset, int count, int width) {
    int valid=0;
    for (int i=0; i<count; i++) valid+=obs[offset+(size_t)i*width]!=0;
    return valid;
}

/* A row changes layout only with its written-record dropped and its memory
 * cleared (research_set_layout does the same). */
#define SWITCH_LAYOUT(v) do { env.layout[0]=env.layout[1]=(v); bopl_written_forget(&env); \
    memset(env.observations, 0, 2*BOPL_OBS_STRIDE*sizeof(float)); compute_observations(&env); obs=env.observations; } while (0)
int main(int argc, char **argv) {
    BoplPuffer env={0}; env.num_agents=2; env.players=2; env.config.max_episode_ticks=3600;
    env.observations=calloc(2*BOPL_OBS_STRIDE, sizeof(float)); env.scratch=env.observations;
    float *v4_rows=calloc(2*BOPL4_OBS_SIZE, sizeof(float)); assert(v4_rows);
    BoplRlEntity entities[96]={{0}}; BoplRlSlime slimes[17]={{0}};
    state.header.version=3; state.header.player_count=2; state.header.slime_count=17;
    state.header.episode_ticks=900; state.header.simulation_time=Q(15);
    state.header.simulation_time_since_level_loaded=Q(110);
    state.entities=entities; state.slimes=slimes;
    state.players[0]=(BoplRlPlayer){.id=1,.team=0,.flags=1,.alive_bodies=16,.scale=Q(1)};
    state.players[1]=(BoplRlPlayer){.id=2,.team=1,.flags=1|4,.alive_bodies=1,
        .x=Q(30),.scale=Q(1),.invisible_time=Q(2),.ability_time=Q(0)};
    /* ability_time=0.5: EaseInOut alpha is 0.5, so the invisible opponent is
     * still spatially visible while using an ability. */
    state.players[1].ability_time=Q(1)/2;

    for (int i=0; i<16; i++) {
        slimes[i].id=100+i; slimes[i].owner=1; slimes[i].flags=BOPL_SLIME_ACTIVE|BOPL_SLIME_SPAWNED;
        slimes[i].x=Q(i==0?10:i==1?2:i==2?5:20+i); slimes[i].scale=Q(1);
        if (i<9 && i!=1) {
            slimes[i].abilities[0]=2; slimes[i].abilities[1]=4; slimes[i].abilities[2]=i==2?7:14;
            slimes[i].ability_flags[0]=slimes[i].ability_flags[1]=slimes[i].ability_flags[2]=1|2|4;
            slimes[i].cooldowns[0]=slimes[i].cooldowns[1]=slimes[i].cooldowns[2]=Q(10);
            slimes[i].timers[0]=slimes[i].timers[1]=slimes[i].timers[2]=Q(5);
        } else slimes[i].flags|=BOPL_SLIME_ABILITYLESS;
    }
    slimes[0].flags|=BOPL_SLIME_SUPPRESS_INPUT; /* first match must lose to slime 2 */
    slimes[2].flags|=2u<<BOPL_SLIME_PICKUP_SLOT_SHIFT; /* pickup replaced slot 2 */
    slimes[16]=(BoplRlSlime){.id=200,.owner=2,.flags=BOPL_SLIME_ACTIVE|BOPL_SLIME_SPAWNED,
        .x=Q(30),.scale=Q(1),.abilities={12,4,14},.ability_flags={7,7,7},
        .cooldowns={Q(10),Q(10),Q(10)},.timers={Q(5),Q(5),Q(5)}};

    int n=0;
    for (int i=0; i<20; i++) entities[n++]=(BoplRlEntity){.id=(uint32_t)n,.kind=0,.layer=9,.owner=1,
        .flags=BOPL_ENTITY_ACTIVE|BOPL_ENTITY_ENABLED|BOPL_ENTITY_EXPLOSIVE,
        .reserved={5,0},.x=Q(i+1),.radius=Q(1),.inverse_mass=Q(1)};
    /* Three hooks, four raw spans apiece. Each contiguous hierarchy reset is
     * one rope and must collapse to exactly three evenly spaced spans. */
    for (int hook=0; hook<3; hook++) for (int span=0; span<4; span++)
        entities[n++]=(BoplRlEntity){.id=(uint32_t)n,.kind=3,.owner=1,.flags=BOPL_ENTITY_ACTIVE|BOPL_ENTITY_ENABLED,
            .hierarchy=(uint32_t)span,.reserved={9,0},.x=Q(2+hook*5+span),.right_x=Q(1),.radius=Q(1)/10,
            .inverse_mass=Q(1)};
    entities[n++]=(BoplRlEntity){.id=(uint32_t)n,.kind=0,.layer=9,.owner=2,
        .flags=BOPL_ENTITY_ACTIVE|BOPL_ENTITY_FLAMMABLE_SMOKE,.reserved={12,0},.x=Q(31),.inverse_mass=Q(1)};
    entities[n++]=(BoplRlEntity){.id=(uint32_t)n,.kind=1,.layer=8,.owner=2,
        .flags=BOPL_ENTITY_ACTIVE|BOPL_ENTITY_ENABLED|BOPL_ENTITY_TRIGGER,.reserved={3,0},.x=Q(30)};
    /* Farther than all clutter but force-inserted because it is a pickup. */
    entities[n++]=(BoplRlEntity){.id=(uint32_t)n,.kind=0,.layer=9,
        .flags=BOPL_ENTITY_ACTIVE|BOPL_ENTITY_PICKUP|BOPL_ENTITY_TIMER_KNOWN,
        .reserved={7,300},.x=Q(1000),.radius=Q(1),.inverse_mass=Q(1),.clock_rate=Q(1)};
    for (int i=0; i<40; i++) entities[n++]=(BoplRlEntity){.id=(uint32_t)n,.kind=2,.layer=11,
        .flags=BOPL_ENTITY_ACTIVE|BOPL_ENTITY_ENABLED,.x=Q(50+i)};
    state.header.entity_count=(uint32_t)n;

    compute_observations(&env);
    float *obs=env.observations;
    assert(env.log.observation_overflow==2);
    /* The same state in layout v4 (player 0's row), for the v4 encoder parity harness. */
    SWITCH_LAYOUT(BOPL4_LAYOUT_VERSION);
    memcpy(v4_rows, obs, BOPL4_OBS_SIZE*sizeof(float));
    assert(env.log.observation_overflow==4);
    assert(valid_rows(obs,BOPL4_OBS_SLOT_OFFSET,BOPL4_SLOTS,BOPL4_SLOT_FEATURES)==BOPL4_SLOTS);
    assert(valid_rows(obs,BOPL4_OBS_ENTITY_OFFSET,BOPL4_ENTITIES,BOPL4_ENTITY_FEATURES)==BOPL4_ENTITIES);
    assert(obs[BOPL4_OBS_OPPONENT_OFFSET+16]==1);   /* opponent position known while mid-ability */
    assert(obs[BOPL4_OBS_OPPONENT_OFFSET+BOPL4_SELF]>0); /* its distance */
    SWITCH_LAYOUT(3);
    env.log.observation_overflow=2;
    float *fixture_rows=calloc(2*BOPL_OBS_SIZE, sizeof(float));
    assert(fixture_rows); memcpy(fixture_rows, obs, BOPL_OBS_SIZE*sizeof(float));
    assert(BOPL_OBS_SIZE==3690 && BOPL_OBS_ENTITY_OFFSET==542);
    assert(obs[10]==(float)(10.0/120.0) && obs[11]==0 && obs[13]==0.75f);
    assert(obs[14+4]==20.0f/BOPL_COUNT_SCALE); /* all 20 grenades, before truncation */
    assert(obs[14+8]==9.0f/BOPL_COUNT_SCALE);  /* 12 raw rope spans -> 9 */
    assert(obs[53]==31.0f/BOPL_COUNT_SCALE && obs[54]==1.0f/BOPL_COUNT_SCALE);
    assert(obs[55+4]==31.0f/BOPL_COUNT_SCALE); /* nonstatic objects in +x sector */
    assert(obs[65]==1.0f);                     /* controlling slime pickup target slot 2 */
    assert(obs[BOPL_OBS_SELF_OFFSET+19]==0.5f);/* 16 owned bodies / 32 */
    assert(valid_rows(obs,BOPL_OBS_SLOT_OFFSET,BOPL_OBS_SLOTS,BOPL_SLOT_FEATURES)==BOPL_OBS_SLOTS);
    for (int slot=0; slot<3; slot++) {
        float *row=obs+BOPL_OBS_SLOT_OFFSET+(size_t)slot*BOPL_SLOT_FEATURES;
        assert(row[11]==5.0f/64.0f); /* controlling, not first or nearest abilityless */
    }
    assert(obs[BOPL_OBS_SLOT_OFFSET+2*BOPL_SLOT_FEATURES+BOPL_SLOT_ABILITY_ID]==7.0f);
    assert(entity_source_count(obs,9)==9);
    float *pickup=entity_with_source(obs,7), *smoke=entity_with_source(obs,12), *body=entity_with_source(obs,3);
    assert(pickup && pickup[25]==1 && pickup[22]==0.5f);
    assert(smoke && smoke[18]==0 && smoke[28]==1); /* disabled-but-alive smoke stays */
    assert(body);                                  /* opponent is visible mid-ability */
    assert(obs[BOPL_OBS_PLAYER_OFFSET+BOPL_PLAYER_FEATURES+16]==1);

    /* Once its effective alpha drops below 0.2, hide opponent coordinates,
     * slime/slot/body rows while keeping the independently owned smoke. */
    state.players[1].ability_time=Q(1);
    compute_observations(&env); obs=env.observations;
    assert(env.log.observation_overflow==4);
    memcpy(fixture_rows+BOPL_OBS_SIZE, obs, BOPL_OBS_SIZE*sizeof(float));
    assert(obs[BOPL_OBS_PLAYER_OFFSET+BOPL_PLAYER_FEATURES+16]==0);
    assert(entity_with_source(obs,3)==NULL && entity_with_source(obs,12)!=NULL);
    assert(valid_rows(obs,BOPL_OBS_SLIME_OFFSET,BOPL_OBS_SLIMES,BOPL_SLIME_FEATURES)==16);
    SWITCH_LAYOUT(BOPL4_LAYOUT_VERSION);
    memcpy(v4_rows+BOPL4_OBS_SIZE, obs, BOPL4_OBS_SIZE*sizeof(float));
    assert(obs[BOPL4_OBS_OPPONENT_OFFSET+16]==0 && obs[BOPL4_OBS_OPPONENT_OFFSET+BOPL4_SELF]==0); /* hidden: relation zero */
    assert(valid_rows(obs,BOPL4_OBS_SLIME_OFFSET,BOPL4_SLIMES,BOPL4_SLIME_FEATURES)==16);
    /* The hidden opponent's controlling slime keeps its slot rows, position unknown. */
    { int enemy_slots=0; for (int i=0; i<BOPL4_SLOTS; i++) { float *row=obs+BOPL4_OBS_SLOT_OFFSET+(size_t)i*BOPL4_SLOT_FEATURES;
        if (row[0] && row[15]) { enemy_slots++; assert(row[14]==0 && row[11]==0 && row[12]==0); } }
      assert(enemy_slots==3); }
    if (argc==3 && !strcmp(argv[1], "--dump-v4")) {
        FILE *file=fopen(argv[2], "wb"); assert(file);
        assert(fwrite(v4_rows, sizeof(float), 2*BOPL4_OBS_SIZE, file)==2*BOPL4_OBS_SIZE);
        assert(fclose(file)==0);
    } else if (argc==3 && !strcmp(argv[1], "--dump")) {
        FILE *file=fopen(argv[2], "wb"); assert(file);
        assert(fwrite(fixture_rows, sizeof(float), 2*BOPL_OBS_SIZE, file)==2*BOPL_OBS_SIZE);
        assert(fclose(file)==0);
    } else assert(argc==1);
    puts("OBSERVATION_LAYOUT_V3_CROWDED_FIXTURE_OK");
    free(fixture_rows);
    free(env.observations);
}
