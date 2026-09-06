#ifndef BOPL_PUFFER_H
#define BOPL_PUFFER_H
#include "bopl_rl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { BOPL_OBS_ENTITIES=96, BOPL_OBS_SLIMES=12, BOPL_ABILITY_COUNT=39,
       BOPL_PLAYER_FEATURES=148, BOPL_ENTITY_FEATURES=21, BOPL_SLIME_FEATURES=28,
       BOPL_OBS_SIZE=10+6*BOPL_PLAYER_FEATURES+BOPL_OBS_ENTITIES*BOPL_ENTITY_FEATURES+BOPL_OBS_SLIMES*BOPL_SLIME_FEATURES+7 };
enum { BOPL_EVENT_FIELDS=9, BOPL_TELEMETRY_SIZE=16,
       BOPL_REWARD_OUTCOME=9, BOPL_REWARD_KILLS, BOPL_REWARD_DEATHS,
       BOPL_EPISODES, BOPL_WINS, BOPL_TIMEOUTS, BOPL_TICKS };
typedef struct {
    float perf, score, episode_return, episode_length, environment_ticks, observation_overflow, game_errors, n;
} Log;
typedef struct {
    Log log;
    float *observations, *actions, *rewards, *terminals;
    int num_agents;
    unsigned int rng;
    BoplRl *world;
    BoplRlConfig config;
    int repeat, random_maps, random_loadouts, failed;
    int observation_delay, delay_cursor, no_autoreset;
    float *history, *sample;
    float returns[6];
    /* Optional host buffer, separate from every tensor visible to the policy. */
    float *telemetry;
    float last_telemetry[6*BOPL_TELEMETRY_SIZE];
    BoplRlTelemetry previous_telemetry[6];
    uint32_t deaths[6], kills[6], steps;
    char error[512];
} BoplPuffer;

static float bopl_clip(double value) { return (float)fmax(-1.0, fmin(1.0, value)); }
static double bopl_fix(int64_t value) { return value/4294967296.0; }
static uint32_t bopl_random(BoplPuffer *env) {
    uint32_t x=env->rng; x^=x<<13; x^=x>>17; x^=x<<5; return env->rng=x;
}
static int bopl_ally(const BoplRlState *state, uint32_t owner, const BoplRlPlayer *self) {
    for (uint32_t i=0; i<state->header.player_count; i++)
        if (state->players[i].id==owner) return state->players[i].team==self->team;
    return 0;
}
/* Stable bounded nearest-neighbor selection. Overflow is an explicit feature
 * and metric; the C game state itself retains all entities. */
static void bopl_nearest(uint32_t *indices, double *distances, int *count, int capacity, uint32_t index, double distance) {
    int position=*count;
    if (position==capacity && distance>=distances[position-1]) return;
    if (position<capacity) (*count)++; else position--;
    while (position>0 && distance<distances[position-1]) {
        distances[position]=distances[position-1]; indices[position]=indices[position-1]; position--;
    }
    distances[position]=distance; indices[position]=index;
}
static int bopl_hidden(const BoplRlState *state, uint32_t owner, uint32_t self) {
    if (owner==self) return 0;
    for (uint32_t i=0; i<state->header.player_count; i++)
        if (state->players[i].id==owner) return (state->players[i].flags&4)!=0;
    return 0;
}
static void compute_observations(BoplPuffer *env) {
    memset(env->observations, 0, (size_t)env->num_agents*BOPL_OBS_SIZE*sizeof(float));
    const BoplRlState *state=bopl_rl_state(env->world);
    if (!state) return;
    for (int a=0; a<env->num_agents; a++) {
        const BoplRlPlayer *self=&state->players[a];
        float *obs=env->observations+(size_t)a*BOPL_OBS_SIZE;
        double self_x=bopl_fix(self->x), self_y=bopl_fix(self->y);
        uint32_t nearest[BOPL_OBS_ENTITIES]; double distances[BOPL_OBS_ENTITIES];
        int count=0, active_count=0;
        for (uint32_t i=0; i<state->header.entity_count; i++) {
            const BoplRlEntity *entity=&state->entities[i];
            if (!(entity->flags&1) || (entity->flags&4) || bopl_hidden(state, entity->owner, self->id)) continue;
            double dx=bopl_fix(entity->x)-self_x, dy=bopl_fix(entity->y)-self_y;
            bopl_nearest(nearest, distances, &count, BOPL_OBS_ENTITIES, i, dx*dx+dy*dy);
            active_count++;
        }
        obs[0]=bopl_clip(bopl_fix(state->header.simulation_time)/120.0);
        obs[1]=bopl_clip(bopl_fix(state->header.player_time_scale));
        obs[2]=bopl_clip((double)state->header.episode_ticks/env->config.max_episode_ticks);
        obs[3]=state->header.player_count/6.0f;
        obs[4]=bopl_clip((double)active_count/BOPL_OBS_ENTITIES);
        obs[5]=bopl_clip((double)(active_count-count)/BOPL_OBS_ENTITIES);
        int visible_slimes=0;
        for (uint32_t i=0; i<state->header.slime_count; i++)
            if (!bopl_hidden(state, state->slimes[i].owner, self->id)) visible_slimes++;
        obs[6]=bopl_clip((double)visible_slimes/BOPL_OBS_SLIMES);
        obs[7]=bopl_clip((double)(visible_slimes>BOPL_OBS_SLIMES?visible_slimes-BOPL_OBS_SLIMES:0)/BOPL_OBS_SLIMES);
        if (active_count>count || visible_slimes>BOPL_OBS_SLIMES) env->log.observation_overflow++;
        obs[8]=bopl_clip(self_x/64.0); obs[9]=bopl_clip(self_y/64.0);
        for (uint32_t i=0; i<state->header.player_count; i++) {
            uint32_t index=i==0?(uint32_t)a:(i<=(uint32_t)a?i-1:i);
            const BoplRlPlayer *player=&state->players[index];
            float *out=obs+10+i*BOPL_PLAYER_FEATURES;
            out[0]=1; out[1]=player->id==self->id; out[2]=player->team==self->team;
            out[3]=(player->flags&1)!=0; out[4]=(player->flags&2)!=0;
            out[5]=bopl_clip(player->alive_bodies/6.0); out[6]=bopl_clip(player->stocks/10.0);
            out[7]=bopl_clip(player->kills/10.0); out[8]=bopl_clip(player->deaths/10.0);
            if (!bopl_hidden(state, player->id, self->id)) {
            out[9]=bopl_clip((bopl_fix(player->x)-self_x)/64.0); out[10]=bopl_clip((bopl_fix(player->y)-self_y)/64.0);
            out[11]=bopl_clip(bopl_fix(player->scale)/3.0); } out[12]=(player->flags&4)!=0;
            out[13]=(player->flags&8)!=0; out[14]=(player->flags&16)!=0; out[15]=(player->flags&(32|64))!=0;
            const BoplRlSlime *slime=NULL;
            for (uint32_t j=0; j<state->header.slime_count; j++)
                if (state->slimes[j].owner==player->id) { slime=&state->slimes[j]; break; }
            if (slime) {
            for (int slot=0; slot<3; slot++) {
                float *ability=out+16+slot*44;
                uint32_t id=slime->abilities[slot]; ability[id<BOPL_ABILITY_COUNT?id:BOPL_ABILITY_COUNT]=1;
                double cooldown=bopl_fix(slime->cooldowns[slot]), timer=bopl_fix(slime->timers[slot]);
                ability[40]=bopl_clip(cooldown>0?timer/cooldown:0);
                ability[41]=bopl_clip(cooldown/60.0);
                ability[42]=(slime->ability_flags[slot]&2)!=0; ability[43]=(slime->ability_flags[slot]&4)!=0;
            }
            } else for (int slot=0; slot<3; slot++) {
                uint32_t id=env->config.player[index].abilities[slot];
                out[16+slot*44+(id<BOPL_ABILITY_COUNT?id:BOPL_ABILITY_COUNT)]=1;
            }
        }
        for (int i=0; i<count; i++) {
            const BoplRlEntity *entity=&state->entities[nearest[i]];
            float *out=obs+10+6*BOPL_PLAYER_FEATURES+i*BOPL_ENTITY_FEATURES;
            out[0]=1; if (entity->kind<5) out[1+entity->kind]=1;
            out[6]=entity->owner==self->id; out[7]=bopl_ally(state, entity->owner, self);
            out[8]=bopl_clip((bopl_fix(entity->x)-self_x)/64.0); out[9]=bopl_clip((bopl_fix(entity->y)-self_y)/64.0);
            out[10]=bopl_clip(bopl_fix(entity->vx)/256.0); out[11]=bopl_clip(bopl_fix(entity->vy)/256.0);
            out[12]=bopl_clip(bopl_fix(entity->right_x)/64.0); out[13]=bopl_clip(bopl_fix(entity->right_y)/64.0);
            out[14]=bopl_clip(bopl_fix(entity->up_x)/64.0); out[15]=bopl_clip(bopl_fix(entity->up_y)/64.0);
            out[16]=bopl_clip(bopl_fix(entity->radius)/16.0); out[17]=(entity->flags&2)!=0;
            out[18]=(entity->flags&8)!=0; out[19]=(entity->flags&16)!=0; out[20]=entity->layer/32.0f;
        }
        uint32_t nearby_slimes[BOPL_OBS_SLIMES]; double slime_distances[BOPL_OBS_SLIMES]; int slime_count=0;
        for (uint32_t i=0; i<state->header.slime_count; i++) {
            const BoplRlSlime *slime=&state->slimes[i];
            if (bopl_hidden(state, slime->owner, self->id)) continue;
            double dx=bopl_fix(slime->x)-self_x, dy=bopl_fix(slime->y)-self_y;
            bopl_nearest(nearby_slimes, slime_distances, &slime_count, BOPL_OBS_SLIMES, i, dx*dx+dy*dy);
        }
        for (int i=0; i<slime_count; i++) {
            const BoplRlSlime *slime=&state->slimes[nearby_slimes[i]];
            float *out=obs+10+6*BOPL_PLAYER_FEATURES+BOPL_OBS_ENTITIES*BOPL_ENTITY_FEATURES+i*BOPL_SLIME_FEATURES;
            out[0]=1; out[1]=slime->owner==self->id; out[2]=bopl_ally(state, slime->owner, self);
            out[3]=(slime->flags&1)!=0; out[4]=(slime->flags&2)!=0; out[5]=(slime->flags&4)!=0; out[6]=(slime->flags&8)!=0;
            out[7]=bopl_clip((bopl_fix(slime->x)-self_x)/64.0); out[8]=bopl_clip((bopl_fix(slime->y)-self_y)/64.0);
            out[9]=bopl_clip(bopl_fix(slime->vx)/256.0); out[10]=bopl_clip(bopl_fix(slime->vy)/256.0);
            out[11]=bopl_clip(bopl_fix(slime->scale)/3.0);
            if (slime->state<11) out[12+slime->state]=1;
            out[23]=sinf((float)bopl_fix(slime->rotation));
            out[24]=cosf((float)bopl_fix(slime->rotation));
        }
    }
}
static void bopl_failed(BoplPuffer *env, const char *operation) {
    env->failed=1; env->log.game_errors++;
    snprintf(env->error, sizeof(env->error), "%s: %s", operation, bopl_rl_error(env->world));
    fprintf(stderr, "Bopl RL stopped: %s\n", env->error);
    /* Do not silently turn failed simulation into successful training episodes. */
    abort();
}
static void init(BoplPuffer *env) {
    const char *game=getenv("BOPL_RL_GAME"), *replay=getenv("BOPL_RL_BOOTSTRAP"), *logs=getenv("BOPL_RL_LOG_DIR");
    if (!game || !replay || !logs) { fprintf(stderr, "Set BOPL_RL_GAME, BOPL_RL_BOOTSTRAP and BOPL_RL_LOG_DIR to absolute paths\n"); abort(); }
    char log_path[4096];
    int length=snprintf(log_path, sizeof(log_path), "%s/worker-%ld-%u.log", logs, (long)getpid(), env->rng);
    if (length<0 || (size_t)length>=sizeof(log_path)) abort();
    env->world=bopl_rl_open(game, replay, log_path, 30000, env->error, sizeof(env->error));
    if (!env->world) { fprintf(stderr, "Bopl RL startup: %s\n", env->error); abort(); }
    if (bopl_rl_ability_count(env->world)!=BOPL_ABILITY_COUNT) { fprintf(stderr, "Bopl RL policy catalog changed\n"); abort(); }
    env->history=calloc((size_t)2*env->num_agents*BOPL_OBS_SIZE, sizeof(float));
    env->sample=calloc((size_t)env->num_agents*BOPL_OBS_SIZE, sizeof(float));
    if (!env->history || !env->sample) abort();
    if (env->observation_delay!=0 && (env->observation_delay!=9 || env->repeat!=4)) abort();
    env->rng=env->rng*747796405u+2891336453u;
}
static void c_reset(BoplPuffer *env) {
    env->config.seed=bopl_random(env);
    if (env->random_maps) env->config.map=bopl_random(env)%bopl_rl_map_count(env->world);
    if (env->random_loadouts) for (int i=0; i<env->num_agents; i++) for (uint32_t j=0; j<env->config.slots; j++)
        env->config.player[i].abilities[j]=1+bopl_random(env)%(BOPL_ABILITY_COUNT-1); // Exclude the source's NullAbility placeholder.
    if (bopl_rl_reset(env->world, &env->config)) bopl_failed(env, "reset");
    const BoplRlState *state=bopl_rl_state(env->world);
    memset(env->returns, 0, sizeof(env->returns)); env->steps=0;
    memcpy(env->previous_telemetry, state->telemetry, sizeof(env->previous_telemetry));
    for (int i=0; i<env->num_agents; i++) { env->deaths[i]=state->players[i].deaths; env->kills[i]=state->players[i].kills; }
    compute_observations(env);
    size_t bytes=(size_t)env->num_agents*BOPL_OBS_SIZE*sizeof(float);
    for (int i=0; i<2; i++) memcpy(env->history+i*env->num_agents*BOPL_OBS_SIZE, env->observations, bytes);
    env->delay_cursor=0;
    for (int i=0; i<env->num_agents; i++) env->observations[(i+1)*BOPL_OBS_SIZE-7]=1;
}
static void c_step(BoplPuffer *env) {
    float *telemetry=env->telemetry?env->telemetry:env->last_telemetry;
    memset(telemetry, 0, (size_t)env->num_agents*BOPL_TELEMETRY_SIZE*sizeof(float));
    memset(env->rewards, 0, (size_t)env->num_agents*sizeof(float));
    memset(env->terminals, 0, (size_t)env->num_agents*sizeof(float));
    BoplRlAction actions[6]={{0}};
    for (int i=0; i<env->num_agents; i++) {
        float angle=env->actions[2*i], buttons=env->actions[2*i+1];
        if (!isfinite(angle) || !isfinite(buttons) || angle<0 || angle>255 || buttons<0 || buttons>15
            || floorf(angle)!=angle || floorf(buttons)!=buttons) { fprintf(stderr, "Invalid PufferLib action\n"); abort(); }
        actions[i].angle=(uint8_t)angle; actions[i].buttons=(uint16_t)buttons;
    }
    const BoplRlState *before=bopl_rl_state(env->world);
    if (!before) bopl_failed(env, "state");
    uint32_t ticks=before->header.episode_ticks;
    // Sample at phase 3 of each four-tick decision. At t=12 return the
    // observation from t=3, giving an exact nine-tick delay thereafter.
    uint32_t first=env->observation_delay?3:(uint32_t)env->repeat;
    if (bopl_rl_step(env->world, actions, env->num_agents, first)) bopl_failed(env, "step");
    if (env->observation_delay) {
        compute_observations(env);
        memcpy(env->sample, env->observations, (size_t)env->num_agents*BOPL_OBS_SIZE*sizeof(float));
        if (!(bopl_rl_state(env->world)->header.flags&BOPL_RL_TERMINAL)
            && bopl_rl_step(env->world, actions, env->num_agents, 1)) bopl_failed(env, "step phase");
    }
    const BoplRlState *state=bopl_rl_state(env->world);
    env->log.environment_ticks+=state->header.episode_ticks-ticks; env->steps++;
    int terminal=(state->header.flags&BOPL_RL_TERMINAL)!=0;
    for (int i=0; i<env->num_agents; i++) {
        const BoplRlPlayer *player=&state->players[i];
        float *stats=telemetry+i*BOPL_TELEMETRY_SIZE;
        uint32_t events[BOPL_EVENT_FIELDS], previous[BOPL_EVENT_FIELDS];
        memcpy(events, &state->telemetry[i], sizeof(events));
        memcpy(previous, &env->previous_telemetry[i], sizeof(previous));
        for (int field=0; field<BOPL_EVENT_FIELDS; field++) {
            if (events[field]<previous[field]) bopl_failed(env, "telemetry counter regressed");
            stats[field]=(float)(events[field]-previous[field]);
        }
        stats[BOPL_REWARD_KILLS]=0.1f*(float)((int64_t)player->kills-env->kills[i]);
        stats[BOPL_REWARD_DEATHS]=-0.1f*(float)((int64_t)player->deaths-env->deaths[i]);
        stats[BOPL_REWARD_OUTCOME]=terminal && !(state->header.flags&BOPL_RL_TRUNCATED)?((player->flags&2)?1.0f:-1.0f):0;
        stats[BOPL_EPISODES]=(float)terminal;
        stats[BOPL_WINS]=terminal && !(state->header.flags&BOPL_RL_TRUNCATED) && (player->flags&2);
        stats[BOPL_TIMEOUTS]=terminal && (state->header.flags&BOPL_RL_TRUNCATED);
        stats[BOPL_TICKS]=(float)(state->header.episode_ticks-ticks);
        env->previous_telemetry[i]=state->telemetry[i];
        /* Retain the original arithmetic/order for the actual PPO reward. */
        float reward=0.1f*((float)((int64_t)player->kills-env->kills[i])-(float)((int64_t)player->deaths-env->deaths[i]));
        if (terminal && !(state->header.flags&BOPL_RL_TRUNCATED)) reward+=(player->flags&2)?1.0f:-1.0f;
        env->rewards[i]=reward; env->returns[i]+=reward;
        env->deaths[i]=player->deaths; env->kills[i]=player->kills;
        if (terminal) {
            env->terminals[i]=1;
            env->log.perf+=(player->flags&2)!=0; env->log.score+=player->kills;
            env->log.episode_return+=env->returns[i]; env->log.episode_length+=env->steps; env->log.n++;
        }
    }
    if (terminal && !env->no_autoreset) c_reset(env);
    else if (env->observation_delay) {
        size_t bytes=(size_t)env->num_agents*BOPL_OBS_SIZE*sizeof(float);
        float *old=env->history+(size_t)env->delay_cursor*env->num_agents*BOPL_OBS_SIZE;
        memcpy(env->observations, old, bytes); memcpy(old, env->sample, bytes);
        env->delay_cursor=(env->delay_cursor+1)%2;
    } else compute_observations(env);
    if (!terminal) for (int i=0; i<env->num_agents; i++) {
        // Own button history is known immediately; it is not world information.
        float *previous=env->observations+(size_t)(i+1)*BOPL_OBS_SIZE-7;
        double angle=(actions[i].angle-1)*6.283185307179586/255.0;
        previous[0]=actions[i].angle==0;
        previous[1]=actions[i].angle?cos(angle):0; previous[2]=actions[i].angle?sin(angle):0;
        for (int bit=0; bit<4; bit++) previous[3+bit]=(actions[i].buttons&(1<<bit))!=0;
    }
}
static void c_render(BoplPuffer *env) {
    (void)env;
    fprintf(stderr, "This worker is headless; export an episode with bopl_rl_replay and play it in the game.\n");
}
static void c_close(BoplPuffer *env) { bopl_rl_close(env->world); env->world=NULL; free(env->history); free(env->sample); }
#endif
