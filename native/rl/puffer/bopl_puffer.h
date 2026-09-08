#ifndef BOPL_PUFFER_H
#define BOPL_PUFFER_H
#include "bopl_rl.h"
#include <math.h>
#include <pthread.h>
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
    /* Learner-player outcomes, split by opponent kind (each averaged by n like
     * every other field, so win rate = wins/episodes). */
    float learner_episodes, learner_wins, historical_episodes, historical_wins;
    /* Per frozen bank, the fields PufferLib's selfplay pool reads (n-averaged
     * like the rest): matches the learner completed against bank b and the
     * learner's score in them. A match whose opponent weights were swapped
     * while it ran (see c_step) counts in neither. */
    float hist_n_bank[8], hist_score_bank[8];
    float map_fraction[31];
} Log;
enum { BOPL_MAX_PLAYERS=6, BOPL_MAX_BANKS=8, BOPL_MAX_SLOTS=BOPL_MAX_PLAYERS+BOPL_MAX_BANKS };
enum { BOPL_COUNTER_MAPS=31, BOPL_COUNTER_SIZE=5+4+BOPL_TELEMETRY_SIZE+BOPL_COUNTER_MAPS };
typedef struct {
    double learner_steps;            /* learner decisions (self-play: every player; historical: the learner side) */
    double game_steps;               /* physical decisions of every player, learner or frozen */
    double matches;                  /* completed matches, once each */
    double opponent_game_decisions;  /* frozen-opponent decisions (game_steps - learner_steps) */
    double kills;                    /* game-credited kills by learner players */
    double buttons[4];               /* learner decisions holding jump, dash, arrow, platform */
    double telemetry[BOPL_TELEMETRY_SIZE]; /* learner-only sums in telemetry FIELDS order */
    double map_matches[BOPL_COUNTER_MAPS]; /* completed matches per map */
} BoplCounters;
_Static_assert(sizeof(BoplCounters)==BOPL_COUNTER_SIZE*sizeof(double), "BoplCounters layout");
typedef struct {
    Log log;
    /* Base pointers assigned by PufferLib's vector (slot 0 of this environment).
     * Everything is routed through the per-slot pointers below instead, filled
     * by my_setup_perm from the vector's agent permutation. */
    float *observations, *actions, *rewards, *terminals;
    /* Vector slots this environment consumes (PufferLib's "num_agents"). A
     * self-play or evaluation environment uses one slot per player. A
     * historical environment (banks>0, two players) uses slot 0 for the
     * learner and slots 1..banks for its opponent, one per frozen weight bank;
     * only the slot of this environment's bank (1+active_bank) is ever active,
     * the rest idle. Player p writes to and reads from slot bopl_slot(env, p). */
    int num_agents, players, banks;
    float *obs_ptr[BOPL_MAX_SLOTS], *action_ptr[BOPL_MAX_SLOTS], *reward_ptr[BOPL_MAX_SLOTS], *terminal_ptr[BOPL_MAX_SLOTS];
    /* PufferLib selfplay-pool protocol (MY_USES_TAGS): tag 0 self-play,
     * 1+active_bank historical. boundary_reached is raised at every match
     * start and cleared by the trainer (count_aligned reset) right after it
     * swapped this bank's weights, so at match end it tells whether the
     * opponent was one policy for the whole match. */
    int tag, boundary_reached;
    /* The frozen bank this environment's opponent runs on, fixed for the
     * environment's lifetime (bank = historical index modulo banks). Which
     * player the learner controls (side) is drawn per match in c_reset from a
     * stream separate from the game setup draws, so the map/seed/loadout
     * sequence is unaffected. */
    int side, active_bank;
    unsigned int side_rng;
    unsigned long matches, stepped;
    /* Player-ordered observation block; rows are published to the slots. */
    float *scratch;
    unsigned int rng;
    BoplRl *world;
    BoplRlConfig config;
    /* Second game process of every auto-resetting (training) environment.
     * While an episode runs on `world`, `spare` is already loading the
     * following episode's setup so the terminal step swaps processes instead
     * of waiting ~35 ms for a scene load. Setups are drawn one episode ahead
     * from the same sequence of bopl_random draws, so every episode still
     * gets exactly the setup it would have had on one process. Evaluation
     * environments (no_autoreset) run a single process. */
    BoplRl *spare;
    BoplRlConfig spare_config;
    pthread_t spare_thread;
    int spare_opening;
    char spare_log[4096];
    /* The world process too is opened on a thread at init and joined at the
     * first reset, so a vector's games boot concurrently
     * instead of one after another (PufferLib calls my_init serially). Which
     * process an episode runs on never affects its outcome, and the log path
     * and rng advance are fixed before the thread starts. */
    pthread_t world_thread;
    int world_opening, world_checked;
    char world_log[4096], world_error[512];
    /* Many worlds per game process (kwarg `worlds`, default 1): physical
     * environment `index` runs world index%worlds of process index/worlds.
     * The first environment of a process opens it (bopl_rl_open_worlds on a
     * thread, so processes still boot concurrently); the others take their
     * handle from the shared group at their first reset. Processes with
     * several worlds have no spare: their resets overlap with the other
     * worlds' play instead. */
    unsigned int index;
    int worlds;
    struct BoplWorldGroup *group;
    int repeat, random_maps, random_loadouts, failed;
    uint32_t training_map_mask;
    int observation_delay, delay_cursor, no_autoreset;
    float *history, *sample;
    float returns[6];
    /* Optional host buffer, separate from every tensor visible to the policy. */
    float *telemetry;
    float last_telemetry[6*BOPL_TELEMETRY_SIZE];
    BoplRlTelemetry previous_telemetry[6];
    uint32_t deaths[6], kills[6], steps;
    /* Reward shaping. Phase-1 comparison keeps the old trainer's values (kill
     * 0.1, timeout 0); the final trainer uses sparse outcome only, with a
     * timeout penalty paid by both players. */
    float reward_kill, reward_timeout;
    /* Process-lifetime cumulative counters over decisions this environment
     * actually played (warmup runs inside the game's reset and never reaches
     * c_step, so it is excluded by construction). Unfinished episodes are
     * included, unlike Log, whose unfinished-episode contributions PufferLib's
     * static_vec_log drops. Learner-only where noted; frozen opponent rows
     * never count as learner work. Read at the rollout barrier by
     * bopl_vec_counters. */
    BoplCounters counters;
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
static void compute_observations_from(BoplPuffer *env, const BoplRlState *state) {
    memset(env->scratch, 0, (size_t)env->players*BOPL_OBS_SIZE*sizeof(float));
    if (!state) return;
    for (int a=0; a<env->players; a++) {
        const BoplRlPlayer *self=&state->players[a];
        float *obs=env->scratch+(size_t)a*BOPL_OBS_SIZE;
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
        if (active_count>count || visible_slimes>BOPL_OBS_SLIMES) {
            env->log.observation_overflow++;
            /* Vector logging skips unfinished episodes, so a log-only error
             * could disappear before the trainer sees it. Never train on it. */
            fprintf(stderr, "Bopl policy observation overflow: %d entities, %d slimes\n", active_count, visible_slimes);
            abort();
        }
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
static void compute_observations(BoplPuffer *env) { compute_observations_from(env, bopl_rl_state(env->world)); }
static void bopl_previous_action(float *previous, BoplRlAction action) {
    double angle=(action.angle-1)*6.283185307179586/255.0;
    previous[0]=action.angle==0;
    previous[1]=action.angle?cos(angle):0; previous[2]=action.angle?sin(angle):0;
    for (int bit=0; bit<4; bit++) previous[3+bit]=(action.buttons&(1<<bit))!=0;
}
/* The vector slot player p is routed through this match. */
static int bopl_slot(const BoplPuffer *env, int player) {
    if (!env->banks) return player;
    return player==env->side?0:1+env->active_bank;
}
static uint32_t bopl_side_random(BoplPuffer *env) {
    uint32_t x=env->side_rng; x^=x<<13; x^=x>>17; x^=x<<5; return env->side_rng=x;
}
/* Copy the player-ordered observations into the routed vector slots. */
static void bopl_publish(BoplPuffer *env) {
    for (int p=0; p<env->players; p++)
        memcpy(env->obs_ptr[bopl_slot(env, p)], env->scratch+(size_t)p*BOPL_OBS_SIZE, BOPL_OBS_SIZE*sizeof(float));
}
static void bopl_failed_world(BoplPuffer *env, const BoplRl *world, const char *operation) {
    env->failed=1; env->log.game_errors++;
    snprintf(env->error, sizeof(env->error), "%s: %s", operation, bopl_rl_error(world));
    fprintf(stderr, "Bopl RL stopped: %s\n", env->error);
    /* Do not silently turn failed simulation into successful training episodes. */
    abort();
}
static void bopl_failed(BoplPuffer *env, const char *operation) { bopl_failed_world(env, env->world, operation); }
/* One game process shared by `worlds` environments of one vector.
 * bopl_group_epoch is bumped by every create_static_vec so environments of
 * different vectors never share a process. */
typedef struct BoplWorldGroup {
    unsigned int epoch, id;
    int worlds, refs, opening, ready;
    pthread_t thread;
    pthread_mutex_t lock;
    BoplRl *handles[64];
    char log[4096], error[512];
    struct BoplWorldGroup *next;
} BoplWorldGroup;
static unsigned int bopl_group_epoch;
static BoplWorldGroup *bopl_groups;
static pthread_mutex_t bopl_groups_lock=PTHREAD_MUTEX_INITIALIZER;
static void *bopl_group_open(void *arg) {
    BoplWorldGroup *group=(BoplWorldGroup*)arg;
    const char *game=getenv("BOPL_RL_GAME"), *replay=getenv("BOPL_RL_BOOTSTRAP");
    bopl_rl_open_worlds(game, replay, group->log, 30000, (uint32_t)group->worlds, group->handles, group->error, sizeof(group->error));
    return NULL;
}
/* Finds or creates the environment's process group (my_init is serial). */
static BoplWorldGroup *bopl_group_acquire(BoplPuffer *env, const char *logs) {
    unsigned int id=env->index/(unsigned)env->worlds;
    pthread_mutex_lock(&bopl_groups_lock);
    BoplWorldGroup *group=bopl_groups;
    while (group && !(group->epoch==bopl_group_epoch && group->id==id && group->worlds==env->worlds)) group=group->next;
    if (!group) {
        group=calloc(1, sizeof(*group));
        if (!group) abort();
        group->epoch=bopl_group_epoch; group->id=id; group->worlds=env->worlds;
        pthread_mutex_init(&group->lock, NULL);
        int length=snprintf(group->log, sizeof(group->log), "%s/worker-%ld-%u-p%u.log", logs, (long)getpid(), bopl_group_epoch, id);
        if (length<0 || (size_t)length>=sizeof(group->log)) abort();
        if (pthread_create(&group->thread, NULL, bopl_group_open, group)) abort();
        group->opening=1;
        group->next=bopl_groups; bopl_groups=group;
    }
    group->refs++;
    pthread_mutex_unlock(&bopl_groups_lock);
    return group;
}
/* The group's process is open (joined once, by whichever environment asks first). */
static void bopl_group_ready(BoplWorldGroup *group) {
    pthread_mutex_lock(&group->lock);
    if (group->opening) { pthread_join(group->thread, NULL); group->opening=0; }
    if (!group->handles[0]) { fprintf(stderr, "Bopl RL startup (%d worlds): %s\n", group->worlds, group->error); abort(); }
    pthread_mutex_unlock(&group->lock);
}
static void bopl_group_release(BoplWorldGroup *group) {
    pthread_mutex_lock(&bopl_groups_lock);
    int last=--group->refs==0;
    if (last) {
        BoplWorldGroup **link=&bopl_groups;
        while (*link && *link!=group) link=&(*link)->next;
        if (*link) *link=group->next;
    }
    pthread_mutex_unlock(&bopl_groups_lock);
    if (last) { pthread_mutex_destroy(&group->lock); free(group); }
}
static void *bopl_spare_open(void *arg) {
    BoplPuffer *env=(BoplPuffer*)arg;
    const char *game=getenv("BOPL_RL_GAME"), *replay=getenv("BOPL_RL_BOOTSTRAP");
    env->spare=bopl_rl_open(game, replay, env->spare_log, 30000, env->error, sizeof(env->error));
    return NULL;
}
static void *bopl_world_open(void *arg) {
    BoplPuffer *env=(BoplPuffer*)arg;
    const char *game=getenv("BOPL_RL_GAME"), *replay=getenv("BOPL_RL_BOOTSTRAP");
    env->world=bopl_rl_open(game, replay, env->world_log, 30000, env->world_error, sizeof(env->world_error));
    return NULL;
}
/* The world process is usable: joined if it was opened on a thread, and its
 * catalog checked once (formerly inline in init). */
static void bopl_world_ready(BoplPuffer *env) {
    if (env->world_opening) { pthread_join(env->world_thread, NULL); env->world_opening=0; }
    if (env->group && !env->world) {
        bopl_group_ready(env->group);
        env->world=env->group->handles[env->index%(unsigned)env->worlds];
    }
    if (!env->world) { fprintf(stderr, "Bopl RL startup: %s\n", env->world_error); abort(); }
    if (env->world_checked) return;
    env->world_checked=1;
    for (uint32_t map=0; map<31; map++)
        if ((env->training_map_mask&(1u<<map)) && map>=bopl_rl_map_count(env->world)) {
            fprintf(stderr, "Training map outside game catalog\n"); abort();
        }
    if (bopl_rl_ability_count(env->world)!=BOPL_ABILITY_COUNT) { fprintf(stderr, "Bopl RL policy catalog changed\n"); abort(); }
}
static void init(BoplPuffer *env) {
    const char *game=getenv("BOPL_RL_GAME"), *replay=getenv("BOPL_RL_BOOTSTRAP"), *logs=getenv("BOPL_RL_LOG_DIR");
    if (!game || !replay || !logs) { fprintf(stderr, "Set BOPL_RL_GAME, BOPL_RL_BOOTSTRAP and BOPL_RL_LOG_DIR to absolute paths\n"); abort(); }
    int length=snprintf(env->world_log, sizeof(env->world_log), "%s/worker-%ld-%u.log", logs, (long)getpid(), env->rng);
    if (length<0 || (size_t)length>=sizeof(env->world_log)) abort();
    if (env->training_map_mask && env->random_maps) {
        fprintf(stderr, "Choose explicit training maps or all random maps\n"); abort();
    }
    if (env->worlds<1 || env->worlds>64) { fprintf(stderr, "Bopl RL: 1..64 worlds per process\n"); abort(); }
    if (env->worlds>1) env->group=bopl_group_acquire(env, logs);
    else {
        if (pthread_create(&env->world_thread, NULL, bopl_world_open, env)) abort();
        env->world_opening=1;
    }
    if (env->players<1 || env->players>BOPL_MAX_PLAYERS || env->banks<0 || env->banks>BOPL_MAX_BANKS
        || (env->banks && env->players!=2) || env->num_agents!=(env->banks?1+env->banks:env->players)) {
        fprintf(stderr, "Bopl RL slot layout: players %d, banks %d, slots %d\n", env->players, env->banks, env->num_agents); abort();
    }
    env->history=calloc((size_t)2*env->players*BOPL_OBS_SIZE, sizeof(float));
    env->sample=calloc((size_t)env->players*BOPL_OBS_SIZE, sizeof(float));
    env->scratch=calloc((size_t)env->players*BOPL_OBS_SIZE, sizeof(float));
    if (!env->history || !env->sample || !env->scratch) abort();
    if (env->banks && (env->active_bank<0 || env->active_bank>=env->banks)) { fprintf(stderr, "Bopl RL: historical environment without a bank\n"); abort(); }
    if (!isfinite(env->reward_kill) || !isfinite(env->reward_timeout) || env->reward_timeout>0) { fprintf(stderr, "Bopl RL reward config\n"); abort(); }
    if (env->observation_delay!=0 && (env->observation_delay!=9 || env->repeat!=4)) abort();
    if (!env->no_autoreset && env->worlds==1) {
        length=snprintf(env->spare_log, sizeof(env->spare_log), "%s/worker-%ld-%u-spare.log", logs, (long)getpid(), env->rng);
        if (length<0 || (size_t)length>=sizeof(env->spare_log)) abort();
        /* Game start-up takes seconds; overlap it with the other workers' start-up. */
        if (pthread_create(&env->spare_thread, NULL, bopl_spare_open, env)) abort();
        env->spare_opening=1;
    }
    env->rng=env->rng*747796405u+2891336453u;
}
static void bopl_spare_ready(BoplPuffer *env) {
    if (!env->spare_opening) return;
    pthread_join(env->spare_thread, NULL); env->spare_opening=0;
    if (!env->spare) { fprintf(stderr, "Bopl RL spare startup: %s\n", env->error); abort(); }
    if (bopl_rl_map_count(env->spare)!=bopl_rl_map_count(env->world)
        || bopl_rl_ability_count(env->spare)!=bopl_rl_ability_count(env->world)) { fprintf(stderr, "Bopl RL spare catalog differs\n"); abort(); }
}
/* The setup of the next episode: the same draws, in the same order, as the
 * single-process path makes at the start of that episode. */
static void bopl_draw_config(BoplPuffer *env, BoplRlConfig *config) {
    config->seed=bopl_random(env);
    if (env->training_map_mask) {
        uint32_t count=0;
        for (uint32_t map=0; map<31; map++) count+=(env->training_map_mask>>map)&1u;
        uint32_t selected=bopl_random(env)%count;
        for (uint32_t map=0; map<31; map++) if (env->training_map_mask&(1u<<map)) {
            if (selected==0) { config->map=map; break; }
            selected--;
        }
    } else if (env->random_maps) config->map=bopl_random(env)%bopl_rl_map_count(env->world);
    if (env->random_loadouts) for (int i=0; i<env->players; i++) for (uint32_t j=0; j<config->slots; j++)
        config->player[i].abilities[j]=1+bopl_random(env)%(BOPL_ABILITY_COUNT-1); // Exclude the source's NullAbility placeholder.
}
static void c_reset(BoplPuffer *env) {
    bopl_world_ready(env);
    bopl_spare_ready(env);
    if (env->spare && bopl_rl_reset_pending(env->spare)) {
        /* The spare has been loading this episode's setup since the previous
         * episode started; take it over and let the finished process load the
         * episode after this one. */
        if (bopl_rl_reset_finish(env->spare)) bopl_failed_world(env, env->spare, "spare reset");
        BoplRl *loaded=env->spare; env->spare=env->world; env->world=loaded;
        env->config=env->spare_config;
    } else {
        bopl_draw_config(env, &env->config);
        if (bopl_rl_reset(env->world, &env->config)) bopl_failed(env, "reset");
    }
    if (env->spare) {
        env->spare_config=env->config;
        bopl_draw_config(env, &env->spare_config);
        if (bopl_rl_reset_begin(env->spare, &env->spare_config)) bopl_failed_world(env, env->spare, "spare reset request");
    }
    const BoplRlState *state=bopl_rl_state(env->world);
    memset(env->returns, 0, sizeof(env->returns)); env->steps=0;
    memcpy(env->previous_telemetry, state->telemetry, sizeof(env->previous_telemetry));
    for (int i=0; i<env->players; i++) { env->deaths[i]=state->players[i].deaths; env->kills[i]=state->players[i].kills; }
    if (env->banks) {
        /* Every bank row goes idle (blank observation; the finished match's
         * opponent state was reset by the terminal flag) and the new match
         * draws its side. Blanking all rows, not just the active one, also
         * clears rows that held another slot's observation before the
         * trainer's agent permutation was installed (create_pufferl resets
         * under identity routing). */
        for (int bank=0; bank<env->banks; bank++) memset(env->obs_ptr[1+bank], 0, BOPL_OBS_SIZE*sizeof(float));
        env->side=(int)(bopl_side_random(env)&1u);
        /* A fresh recurrent state for the row this match's opponent runs on. */
        *env->terminal_ptr[1+env->active_bank]=1;
    }
    env->boundary_reached=1; env->matches++;
    compute_observations(env);
    size_t bytes=(size_t)env->players*BOPL_OBS_SIZE*sizeof(float);
    for (int i=0; i<2; i++) memcpy(env->history+i*env->players*BOPL_OBS_SIZE, env->scratch, bytes);
    env->delay_cursor=0;
    for (int i=0; i<env->players; i++) env->scratch[(i+1)*BOPL_OBS_SIZE-7]=1;
    bopl_publish(env);
}
static void c_step(BoplPuffer *env) {
    float *telemetry=env->telemetry?env->telemetry:env->last_telemetry;
    memset(telemetry, 0, (size_t)env->players*BOPL_TELEMETRY_SIZE*sizeof(float));
    for (int s=0; s<env->num_agents; s++) { *env->reward_ptr[s]=0; *env->terminal_ptr[s]=0; }
    env->stepped++;
    BoplRlAction actions[6]={{0}};
    for (int i=0; i<env->players; i++) {
        const float *action=env->action_ptr[bopl_slot(env, i)];
        float angle=action[0], buttons=action[1];
        if (!isfinite(angle) || !isfinite(buttons) || angle<0 || angle>255 || buttons<0 || buttons>15
            || floorf(angle)!=angle || floorf(buttons)!=buttons) { fprintf(stderr, "Invalid PufferLib action\n"); abort(); }
        actions[i].angle=(uint8_t)angle; actions[i].buttons=(uint16_t)buttons;
    }
    const BoplRlState *before=bopl_rl_state(env->world);
    if (!before) bopl_failed(env, "state");
    uint32_t ticks=before->header.episode_ticks;
    // Sample at phase 3 of each four-tick decision. At t=12 return the
    // observation from t=3, giving an exact nine-tick delay thereafter.
    if (env->observation_delay) {
        // Both phases in one game roundtrip: the phase-3 state is returned
        // alongside the final state (identical when the episode ends in phase 1).
        if (bopl_rl_step_sampled(env->world, actions, env->players, 3, 1)) bopl_failed(env, "sampled step");
        compute_observations_from(env, bopl_rl_sample(env->world));
        memcpy(env->sample, env->scratch, (size_t)env->players*BOPL_OBS_SIZE*sizeof(float));
    } else if (bopl_rl_step(env->world, actions, env->players, (uint32_t)env->repeat)) bopl_failed(env, "step");
    const BoplRlState *state=bopl_rl_state(env->world);
    env->log.environment_ticks+=state->header.episode_ticks-ticks; env->steps++;
    int terminal=(state->header.flags&BOPL_RL_TERMINAL)!=0;
    for (int i=0; i<env->players; i++) {
        const BoplRlPlayer *player=&state->players[i];
        int slot=bopl_slot(env, i), learner=!env->banks || i==env->side;
        float *stats=telemetry+i*BOPL_TELEMETRY_SIZE;
        uint32_t events[BOPL_EVENT_FIELDS], previous[BOPL_EVENT_FIELDS];
        memcpy(events, &state->telemetry[i], sizeof(events));
        memcpy(previous, &env->previous_telemetry[i], sizeof(previous));
        for (int field=0; field<BOPL_EVENT_FIELDS; field++) {
            if (events[field]<previous[field]) bopl_failed(env, "telemetry counter regressed");
            stats[field]=(float)(events[field]-previous[field]);
        }
        int truncated=(state->header.flags&BOPL_RL_TRUNCATED)!=0;
        stats[BOPL_REWARD_KILLS]=env->reward_kill*(float)((int64_t)player->kills-env->kills[i]);
        stats[BOPL_REWARD_DEATHS]=-env->reward_kill*(float)((int64_t)player->deaths-env->deaths[i]);
        /* Terminal reward as paid: the win/loss outcome, or the timeout penalty
         * both players receive on truncation (0 in the phase-1 comparison). */
        stats[BOPL_REWARD_OUTCOME]=terminal?(truncated?env->reward_timeout:((player->flags&2)?1.0f:-1.0f)):0;
        stats[BOPL_EPISODES]=(float)terminal;
        stats[BOPL_WINS]=terminal && !(state->header.flags&BOPL_RL_TRUNCATED) && (player->flags&2);
        stats[BOPL_TIMEOUTS]=terminal && (state->header.flags&BOPL_RL_TRUNCATED);
        stats[BOPL_TICKS]=(float)(state->header.episode_ticks-ticks);
        env->previous_telemetry[i]=state->telemetry[i];
        /* Retain the original arithmetic/order for the actual PPO reward. */
        float reward=env->reward_kill*((float)((int64_t)player->kills-env->kills[i])-(float)((int64_t)player->deaths-env->deaths[i]));
        if (terminal) reward+=truncated?env->reward_timeout:((player->flags&2)?1.0f:-1.0f);
        *env->reward_ptr[slot]=reward; env->returns[i]+=reward;
        BoplCounters *c=&env->counters;
        c->game_steps++;
        if (learner) {
            c->learner_steps++;
            c->kills+=(double)((int64_t)player->kills-env->kills[i]);
            for (int field=0; field<BOPL_TELEMETRY_SIZE; field++) c->telemetry[field]+=stats[field];
            for (int bit=0; bit<4; bit++) c->buttons[bit]+=(actions[i].buttons>>bit)&1u;
        } else c->opponent_game_decisions++;
        env->deaths[i]=player->deaths; env->kills[i]=player->kills;
        if (terminal) {
            if (env->config.map<31) env->log.map_fraction[env->config.map]++;
            *env->terminal_ptr[slot]=1;
            int won=!truncated && (player->flags&2)!=0;
            env->log.perf+=won; env->log.score+=player->kills;
            env->log.episode_return+=env->returns[i]; env->log.episode_length+=env->steps; env->log.n++;
            if (learner) {
                env->log.learner_episodes++; env->log.learner_wins+=won;
                if (env->banks) {
                    env->log.historical_episodes++; env->log.historical_wins+=won;
                    /* The pool's Elo/swap statistics skip a match whose opponent
                     * weights the trainer replaced while it ran (the swap
                     * cleared boundary_reached; the next c_reset raises it). */
                    if (env->boundary_reached) { env->log.hist_n_bank[env->active_bank]++; env->log.hist_score_bank[env->active_bank]+=won; }
                }
            }
        }
    }
    if (terminal) { env->counters.matches++; if (env->config.map<BOPL_COUNTER_MAPS) env->counters.map_matches[env->config.map]++; }
    if (terminal && !env->no_autoreset) { c_reset(env); return; }
    if (env->observation_delay) {
        size_t bytes=(size_t)env->players*BOPL_OBS_SIZE*sizeof(float);
        float *old=env->history+(size_t)env->delay_cursor*env->players*BOPL_OBS_SIZE;
        memcpy(env->scratch, old, bytes); memcpy(old, env->sample, bytes);
        env->delay_cursor=(env->delay_cursor+1)%2;
    } else compute_observations(env);
    if (!terminal) for (int i=0; i<env->players; i++) {
        // Own button history is known immediately; it is not world information.
        float *previous=env->scratch+(size_t)(i+1)*BOPL_OBS_SIZE-7;
        bopl_previous_action(previous, actions[i]);
    }
    bopl_publish(env);
}
static void c_render(BoplPuffer *env) {
    (void)env;
    fprintf(stderr, "This worker is headless; export an episode with bopl_rl_replay and play it in the game.\n");
}
static void c_close(BoplPuffer *env) {
    if (env->world_opening) { pthread_join(env->world_thread, NULL); env->world_opening=0; }
    if (env->spare_opening) { pthread_join(env->spare_thread, NULL); env->spare_opening=0; }
    if (env->group && !env->world) { bopl_group_ready(env->group); env->world=env->group->handles[env->index%(unsigned)env->worlds]; }
    bopl_rl_close(env->world); env->world=NULL;
    bopl_rl_close(env->spare); env->spare=NULL;
    if (env->group) { bopl_group_release(env->group); env->group=NULL; }
    free(env->history); free(env->sample); free(env->scratch);
}
#endif
