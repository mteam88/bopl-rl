/* Hidden-coordinate changes must not affect any feature or nearest-item order. */
#define bopl_rl_state observation_test_state
#include "../puffer/bopl_puffer.h"
#undef bopl_rl_state
#include <assert.h>
static BoplRlState state;
const BoplRlState *observation_test_state(const BoplRl *world) { (void)world; return &state; }
int main(void) {
    BoplPuffer env={0}; env.num_agents=2; env.config.max_episode_ticks=3600;
    env.observations=calloc(2*BOPL_OBS_SIZE,sizeof(float));
    float *before=calloc(2*BOPL_OBS_SIZE,sizeof(float));
    BoplRlEntity entities[2]={{0}}; BoplRlSlime slimes[2]={{0}};
    state.header.player_count=2; state.header.entity_count=2; state.header.slime_count=2;
    state.entities=entities; state.slimes=slimes;
    for(int i=0;i<2;i++) {
        state.players[i].id=i+1; state.players[i].team=i; state.players[i].flags=1;
        slimes[i].owner=i+1; slimes[i].flags=3;
        slimes[i].abilities[0]=2; slimes[i].abilities[1]=4; slimes[i].abilities[2]=14;
        slimes[i].cooldowns[0]=2LL<<32; slimes[i].timers[0]=1LL<<32;
        entities[i].owner=i+1; entities[i].flags=3;
    }
    state.players[1].flags|=4;
    compute_observations(&env); memcpy(before,env.observations,2*BOPL_OBS_SIZE*sizeof(float));
    state.players[1].x=12345LL<<32; state.players[1].y=-(5432LL<<32); state.players[1].scale=7LL<<32;
    entities[1].x=123LL<<32; entities[1].vx=999LL<<32;
    slimes[1].x=456LL<<32; slimes[1].vx=999LL<<32;
    compute_observations(&env);
    assert(memcmp(before,env.observations,BOPL_OBS_SIZE*sizeof(float))==0);
    /* Its owner still sees its own position; public cooldowns remain visible. */
    assert(memcmp(before+BOPL_OBS_SIZE,env.observations+BOPL_OBS_SIZE,BOPL_OBS_SIZE*sizeof(float))!=0);
    size_t cooldown=10+BOPL_PLAYER_FEATURES+16+40;
    assert(env.observations[cooldown]==0.5f);
    slimes[1].timers[0]=2LL<<32; compute_observations(&env);
    assert(env.observations[cooldown]==1.0f);
    puts("OBSERVATION_PRIVACY_AND_PUBLIC_COOLDOWNS_OK");
    free(before); free(env.observations);
}
