#ifndef BOPL_PUFFER_H
#define BOPL_PUFFER_H
#include "bopl_rl.h"
#include "bopl_layout.h"
#include "bopl_layout_v4.h"
#include "bopl_bots.h"
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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
    /* Matches the learner played against a scripted bot in bank b (SCRIPTED_BOTS.md),
     * kept out of hist_*_bank so the pool's swap/Elo statistics only see the
     * bank's frozen weights. */
    float bot_n_bank[8], bot_score_bank[8];
    /* Decided learner-vs-roster-entry games this window (timeouts excluded).
     * Index is the process-wide roster slot pushed from train.py. */
    float bot_n_entry[32], bot_score_entry[32];
    /* Hold head (learner decisions, n-averaged): how often each hold length
     * was chosen when the button state was free, and how many decisions the
     * button head was ignored because a latch was in force. */
    float hold_choice[BOPL_ACT_HOLD], latched_decisions;
    float map_fraction[64];   /* BOPL_COUNTER_MAPS */
} Log;
enum { BOPL_MAX_PLAYERS=6, BOPL_MAX_BANKS=8, BOPL_MAX_SLOTS=BOPL_MAX_PLAYERS+BOPL_MAX_BANKS };
/* 64 maps: training_map_mask is 64 bits wide (bits 0-61 reach the binding as two 31-bit halves). */
enum { BOPL_COUNTER_MAPS=64, BOPL_COUNTER_SIZE=5+4+BOPL_TELEMETRY_SIZE+BOPL_COUNTER_MAPS };
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
/* Fraction of a scripted bank's matches that the bot plays; the rest fall to
 * the bank's frozen weights. Process-wide and exported: train.py writes it
 * between updates through ctypes (`bopl_scripted_share` in the trainer
 * module), which is why it is a plain global read once per c_reset rather
 * than a kwarg. Share is Elo-informativeness, not a time decay. */
float bopl_scripted_share=1.0f;
/* Process-wide roster of (kind, level) weighted for the per-episode draw on
 * a scripted bank. train.py pushes Elos-derived weights the same way as share.
 * n=0 keeps the old fixed nibble kind + scripted_level. */
enum { BOPL_ROSTER_MAX=32 };
int bopl_roster_n=0;
uint8_t bopl_roster_kind[BOPL_ROSTER_MAX];
uint8_t bopl_roster_level[BOPL_ROSTER_MAX];
float bopl_roster_weight[BOPL_ROSTER_MAX];
/* A world state that outlives the game's own state buffer: header, players and
 * telemetry by value, entity and slime rows in owned buffers. */
typedef struct {
    BoplRlState state;
    BoplRlEntity *entities; BoplRlSlime *slimes;
    uint32_t entity_capacity, slime_capacity;
    int valid;
} BoplStateCopy;
static void bopl_state_copy(BoplStateCopy *copy, const BoplRlState *state) {
    if (!state) { copy->valid=0; return; }
    uint32_t entities=state->header.entity_count, slimes=state->header.slime_count;
    if (entities>copy->entity_capacity) {
        uint32_t capacity=entities<64?64:entities*2;
        BoplRlEntity *grown=realloc(copy->entities, (size_t)capacity*sizeof(*grown));
        if (!grown) abort();
        copy->entities=grown; copy->entity_capacity=capacity;
    }
    if (slimes>copy->slime_capacity) {
        uint32_t capacity=slimes<12?12:slimes*2;
        BoplRlSlime *grown=realloc(copy->slimes, (size_t)capacity*sizeof(*grown));
        if (!grown) abort();
        copy->slimes=grown; copy->slime_capacity=capacity;
    }
    copy->state=*state;
    if (entities) memcpy(copy->entities, state->entities, (size_t)entities*sizeof(*copy->entities));
    if (slimes) memcpy(copy->slimes, state->slimes, (size_t)slimes*sizeof(*copy->slimes));
    copy->state.entities=copy->entities; copy->state.slimes=copy->slimes;
    copy->valid=1;
}
static const BoplRlState *bopl_state_copy_get(const BoplStateCopy *copy) { return copy->valid?&copy->state:NULL; }
static void bopl_state_copy_free(BoplStateCopy *copy) { free(copy->entities); free(copy->slimes); memset(copy, 0, sizeof(*copy)); }
typedef struct {
    Log log;
    /* Base pointers assigned by PufferLib's vector (slot 0 of this environment).
     * Everything is routed through the per-slot pointers below instead, filled
     * by my_setup_perm from the vector's agent permutation. */
    float *observations, *actions, *rewards, *terminals;
    /* PufferLib action mask (MY_ACTION_MASK = BOPL_LOGITS bytes per agent, 1 =
     * allowed). While a player's buttons are latched, the button head is
     * masked to the latched state and the hold head to length 1, so the
     * sampler returns them with log-probability 0 and the loss kernel gives
     * those heads no gradient or entropy: latched decisions are not decisions. */
    unsigned char *action_mask;
    /* Vector slots this environment consumes (PufferLib's "num_agents"). A
     * self-play or evaluation environment uses one slot per player. A
     * historical environment (banks>0, two players) uses slot 0 for the
     * learner and slots 1..banks for its opponent, one per frozen weight bank;
     * only the slot of this environment's bank (1+active_bank) is ever active,
     * the rest idle. Player p writes to and reads from slot bopl_slot(env, p). */
    int num_agents, players, banks;
    float *obs_ptr[BOPL_MAX_SLOTS], *action_ptr[BOPL_MAX_SLOTS], *reward_ptr[BOPL_MAX_SLOTS], *terminal_ptr[BOPL_MAX_SLOTS];
    unsigned char *mask_ptr[BOPL_MAX_SLOTS];
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
    /* Player-ordered output block for standalone encoders (research_pack,
     * research/check_observations.c). NULL in an environment: its rows are
     * encoded straight into the routed slots (bopl_obs_row), once. */
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
    int repeat, random_maps, random_loadouts, failed;
    uint64_t training_map_mask;
    int observation_delay, delay_cursor, no_autoreset;
    /* Observation delay (nine ticks = two decisions at repeat 4): the world
     * states sampled at phase 3 of the last two decisions, encoded into the
     * slots two decisions later. Keeping ~3 KB of state per slot instead of a
     * 13 KB observation per player, and encoding at publish time, replaces
     * the old encode -> sample -> history -> publish copy chain. */
    BoplStateCopy delayed[2], sample_copy;
    /* How many player, entity and slime rows the last encode into each
     * observation row left non-zero (bopl_encode); the unused slots behind
     * them were zeroed then and stay zero, so the next encode zeroes only the
     * slots it uncovers. bopl_written_forget drops the record when the row
     * memory may have been written by anything else (a routing change). */
    struct { float *row; int players, slots, entities, slimes; } written[BOPL_MAX_SLOTS];
    int written_count;
    /* Observation layout version of each player's row (3 or 4, bopl_layout_v4.h).
     * Training sets one for every player (obs_layout kwarg); evaluation selects
     * per player from the checkpoint's architecture (research_set_layout). */
    int layout[BOPL_MAX_PLAYERS];
    float returns[6];
    /* Optional host buffer, separate from every tensor visible to the policy. */
    float *telemetry;
    float last_telemetry[6*BOPL_TELEMETRY_SIZE];
    BoplRlTelemetry previous_telemetry[6];
    uint32_t deaths[6], kills[6], steps;
    /* Button latch (hold head). hold_left: decisions the latched button state
     * still applies after this one; held: that state; hold_run: consecutive
     * decisions each button has been down (0 when up). Aim is never latched. */
    int hold_left[6]; uint16_t held[6]; int hold_run[6][4];
    /* The decisions applied before the last one (layout v4's own-action
     * history, BOPL4_HISTORY): [p][0] the previous decision, [p][1] the one
     * before. Zero (no aim, no buttons) at the episode start. */
    BoplRlAction action_history[6][2];
    /* The decision applied this step, set before the rows are encoded: layout
     * v4.1's aim ray (bopl_obs_v4.h bopl4_aim_rays). Zero at the episode start. */
    BoplRlAction applied[6];
    /* Scripted opponents (bopl_bots.h, SCRIPTED_BOTS.md). Training: banks whose
     * nibble in scripted_kinds is set play a bot instead of frozen weights (the
     * bank's GPU rows still run; their sampled actions are ignored). Evaluation:
     * research_bot pins a bot on a player. bot_select is the per-player
     * assignment of the current match; bots draw from their own rng stream so
     * a scripted run's map/seed/side sequences match an unscripted one. */
    uint32_t scripted_kinds; int scripted_level;
    uint8_t bot_select[6], bot_select_level[6];
    BoplBot bots[6];
    uint32_t bot_rng;
    /* Roster slot drawn for this match (-1 = not a bot match / no roster). */
    int roster_pick;
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

/* fmax(-1, fmin(1, value)) without the libm calls (gcc will not inline them
 * for their NaN rules): the same result for every input, NaN included (1). */
static inline float bopl_clip(double value) {
    if (!(value<=1.0)) value=1.0;
    if (value<-1.0) value=-1.0;
    return (float)value;
}
static double bopl_fix(int64_t value) { return value/4294967296.0; }
static uint32_t bopl_random(BoplPuffer *env) {
    uint32_t x=env->rng; x^=x<<13; x^=x>>17; x^=x<<5; return env->rng=x;
}
/* The seed a scripted bot on `player` starts the current episode with: a pure
 * function of the episode seed (env->config.seed, drawn by bopl_draw_config
 * from the environment stream and handed to the game by bopl_rl_reset) and
 * the player, so a game is a function of (seed, map, bots) however many
 * episodes the environment played before and whether the bot was selected
 * before or after the reset (research_bot re-arms with the same value). */
static uint32_t bopl_bot_seed(const BoplPuffer *env, int player) {
    uint32_t x=(env->config.seed^0x9E3779B9u)*2654435761u+(uint32_t)(player+1)*0x85EBCA6Bu;
    x^=x>>16; x*=0x7FEB352Du; x^=x>>15; x*=0x846CA68Bu; x^=x>>16;
    return x;
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
static const BoplRlPlayer *bopl_player_by_id(const BoplRlState *state, uint32_t id) {
    for (uint32_t i=0; i<state->header.player_count; i++)
        if (state->players[i].id==id) return &state->players[i];
    return NULL;
}
/* Unity's Invisibility fade is EaseInOut(0,1,1,0): a smoothstep from opaque
 * to transparent. During the final invisible second it fades back in using
 * 5-timeSpentInvisible, exactly like SetAlphaBasedOnPlayerInvisibiltyTime. */
static double bopl_invisibility_alpha(const BoplRlPlayer *player) {
    double remaining=5.0-bopl_fix(player->invisible_time);
    double t=remaining<=1.0?remaining:bopl_fix(player->ability_time);
    if (t<0.0) t=0.0; else if (t>1.0) t=1.0;
    return 1.0-3.0*t*t+2.0*t*t*t;
}
static int bopl_spatial_hidden(const BoplRlState *state, uint32_t owner, uint32_t self) {
    if (!owner || owner==self) return 0;
    const BoplRlPlayer *player=bopl_player_by_id(state, owner);
    return player && (player->flags&4) && bopl_invisibility_alpha(player)<0.2;
}
static int bopl_entity_visible(const BoplRlState *state, const BoplRlEntity *entity, uint32_t self) {
    if (!(entity->flags&BOPL_ENTITY_ACTIVE) || (entity->flags&BOPL_ENTITY_DESTROYED)) return 0;
    /* Only the invisible player's Player-layer body/hurtbox hierarchy hides.
     * Independent owned objects remain visible, including disabled smoke. */
    if (entity->layer==8 && bopl_spatial_hidden(state, entity->owner, self)
        && !(entity->flags&(BOPL_ENTITY_PICKUP|BOPL_ENTITY_PROJECTILE|BOPL_ENTITY_SPARK_NODE
                            |BOPL_ENTITY_FLAMMABLE_SMOKE|BOPL_ENTITY_EXPLOSIVE))) return 0;
    return 1;
}
static int bopl_slime_inventory(const BoplRlSlime *slime) {
    int count=0;
    for (int slot=0; slot<3; slot++)
        if ((slime->ability_flags[slot]&1) && slime->abilities[slot]>0
            && slime->abilities[slot]<=BOPL_ABILITY_COUNT) count++;
    return count;
}
static const BoplRlSlime *bopl_controlling_slime(const BoplRlState *state,
                                                  uint32_t owner, double x, double y) {
    const BoplRlSlime *best=NULL; double best_distance=0.0; int preferred=0;
    for (uint32_t i=0; i<state->header.slime_count; i++) {
        const BoplRlSlime *slime=&state->slimes[i];
        if (slime->owner!=owner || !bopl_slime_inventory(slime)) continue;
        int candidate_preferred=(slime->flags&BOPL_SLIME_SUPPRESS_INPUT)==0;
        double dx=bopl_fix(slime->x)-x, dy=bopl_fix(slime->y)-y, distance=dx*dx+dy*dy;
        if (!best || candidate_preferred>preferred
            || (candidate_preferred==preferred && distance<best_distance)) {
            best=slime; best_distance=distance; preferred=candidate_preferred;
        }
    }
    return best;
}
static int bopl_rope_group_end(const BoplRlState *state, uint32_t begin) {
    const BoplRlEntity *first=&state->entities[begin];
    uint32_t end=begin+1, hierarchy=first->hierarchy;
    while (end<state->header.entity_count) {
        const BoplRlEntity *next=&state->entities[end];
        if (next->kind!=3 || next->owner!=first->owner || next->hierarchy<=hierarchy) break;
        hierarchy=next->hierarchy; end++;
    }
    return (int)end;
}
static int bopl_rope_span_kept(int ordinal, int count) {
    if (count<=3) return 1;
    return ordinal==0 || ordinal==(count-1)/2 || ordinal==count-1;
}
static double bopl_surface_distance(const BoplRlEntity *entity, double x, double y) {
    double px=bopl_fix(entity->x), py=bopl_fix(entity->y), radius=bopl_fix(entity->radius);
    if (entity->kind==3 || entity->kind==4) {
        double sx=bopl_fix(entity->right_x), sy=bopl_fix(entity->right_y);
        double length=sx*sx+sy*sy, t=length>0?((x-px)*sx+(y-py)*sy)/length:0.0;
        if (t<0.0) t=0.0; else if (t>1.0) t=1.0;
        double dx=x-(px+t*sx), dy=y-(py+t*sy);
        return sqrt(dx*dx+dy*dy)-radius;
    }
    double dx=px-x, dy=py-y;
    return sqrt(dx*dx+dy*dy)-radius;
}
/* The vector slot player p is routed through this match. */
static int bopl_slot(const BoplPuffer *env, int player) {
    if (!env->banks) return player;
    return player==env->side?0:1+env->active_bank;
}
/* Where player p's observation row is encoded. */
static float *bopl_obs_row(BoplPuffer *env, int player) {
    return env->scratch?env->scratch+(size_t)player*BOPL_OBS_STRIDE:env->obs_ptr[bopl_slot(env, player)];
}
/* Layout version of player p's row: 4 (bopl_layout_v4.h) or 3 (the default). */
static int bopl_layout_of(const BoplPuffer *env, int player) { return env->layout[player]==BOPL4_LAYOUT_VERSION?4:3; }
static int bopl_obs_size_of(const BoplPuffer *env, int player) { return bopl_layout_of(env, player)==4?BOPL4_OBS_SIZE:BOPL_OBS_SIZE; }
static int bopl_previous_offset(const BoplPuffer *env, int player) {
    return bopl_layout_of(env, player)==4?BOPL4_OBS_PREVIOUS_OFFSET:BOPL_OBS_PREVIOUS_OFFSET;
}
/* The policy sees the nearest BOPL_OBS_ENTITIES entities plus the overflow
 * fraction (obs[5]/obs[7]); the count is logged per episode. Crowded Platform
 * rounds (>96 live bodies) are a real, rare game state, not an error: the
 * 2026-09-08 500M run died on one at 55M learner steps. Report once per env,
 * keep training. Counted when a state is sampled (bopl_step_sample) or
 * encoded undelayed, once per state, as before the delayed states were
 * encoded at publish time. */
static void bopl_note_overflow(BoplPuffer *env, int active_count, int count,
                               int visible_slimes, int visible_slots) {
    if (active_count>count || visible_slimes>BOPL_OBS_SLIMES || visible_slots>BOPL_OBS_SLOTS) {
        if (env->log.observation_overflow==0)
            fprintf(stderr, "Bopl policy observation overflow: %d entities, %d slimes, %d slots (truncated)\n",
                    active_count, visible_slimes, visible_slots);
        env->log.observation_overflow++;
    }
}
/* Every visible-entity and visible-slime count of `state` per observer, for
 * the overflow telemetry of a state that is encoded later. */
static void bopl_count_overflow(BoplPuffer *env, const BoplRlState *state) {
    if (!state) return;
    for (int a=0; a<env->players; a++) {
        const BoplRlPlayer *self=&state->players[a];
        int active_count=0, visible_slimes=0, visible_slots=0;
        for (uint32_t i=0; i<state->header.entity_count;) {
            const BoplRlEntity *entity=&state->entities[i];
            if (entity->kind==3) {
                int end=bopl_rope_group_end(state, i), count=end-(int)i;
                for (int j=0; j<count; j++)
                    if (bopl_rope_span_kept(j, count)
                        && bopl_entity_visible(state, &state->entities[i+j], self->id)) active_count++;
                i=(uint32_t)end;
            } else {
                if (bopl_entity_visible(state, entity, self->id)) active_count++;
                i++;
            }
        }
        for (uint32_t i=0; i<state->header.slime_count; i++) {
            const BoplRlSlime *slime=&state->slimes[i];
            if (bopl_spatial_hidden(state, slime->owner, self->id)) continue;
            visible_slimes++; visible_slots+=bopl_slime_inventory(slime);
        }
        bopl_note_overflow(env, active_count,
                           active_count<BOPL_OBS_ENTITIES?active_count:BOPL_OBS_ENTITIES,
                           visible_slimes, visible_slots);
    }
}
/* The row counts the last encode left in observation row `row`, or NULL when
 * the row's memory is not known to hold a previous encode (then every unused
 * slot is zeroed and the row is recorded). */
static void bopl_written_forget(BoplPuffer *env) { env->written_count=0; }
static void bopl_written_forget_row(BoplPuffer *env, const float *row) {
    for (int i=0; i<env->written_count; i++) if (env->written[i].row==row) { env->written[i]=env->written[--env->written_count]; return; }
}
/* Zero slots [count, previous) of a block of `features`-float slots: with a
 * record, the slots from the previous encode's count on are already zero. */
static void bopl_zero_tail(float *block, int features, int count, int previous, int capacity) {
    int end=previous<0?capacity:previous;
    if (count<end) memset(block+(size_t)count*features, 0, (size_t)(end-count)*features*sizeof(float));
}
static void bopl_player_features(float *out, const BoplRlState *state,
                                 const BoplRlPlayer *player, const BoplRlPlayer *self,
                                 double self_x, double self_y) {
    memset(out, 0, BOPL_PLAYER_FEATURES*sizeof(float));
    int hidden=bopl_spatial_hidden(state, player->id, self->id), bodies=0;
    for (uint32_t i=0; i<state->header.slime_count; i++)
        if (state->slimes[i].owner==player->id) bodies++;
    out[0]=1; out[1]=player->id==self->id; out[2]=player->team==self->team;
    out[3]=(player->flags&1)!=0; out[4]=(player->flags&2)!=0;
    out[5]=bopl_clip(player->alive_bodies/6.0); out[6]=bopl_clip(player->stocks/10.0);
    out[7]=bopl_clip(player->kills/10.0); out[8]=bopl_clip(player->deaths/10.0);
    if (!hidden) {
        out[9]=bopl_clip((bopl_fix(player->x)-self_x)/64.0);
        out[10]=bopl_clip((bopl_fix(player->y)-self_y)/64.0);
        out[11]=bopl_clip(bopl_fix(player->scale)/3.0);
        out[16]=1;
    }
    out[12]=(player->flags&4)!=0; out[13]=(player->flags&8)!=0;
    out[14]=(player->flags&16)!=0; out[15]=(player->flags&(32|64))!=0;
    out[17]=bopl_clip(bopl_fix(player->invisible_time)/5.0);
    out[18]=bopl_clip(bopl_fix(player->ability_time)/5.0);
    out[19]=bopl_clip(bodies/(double)BOPL_OBS_SLIMES);
}
/* Encode `state` into player a's layout-v3 row (bopl_obs_row), writing the row
 * exactly once: every feature is stored, the unused entity/slime slots and
 * the own-action tail are zeroed. (The body of the former bopl_encode loop;
 * bopl_encode dispatches per player's layout.) */
static void bopl_encode_v3_row(BoplPuffer *env, const BoplRlState *state, int a, int count_overflow) {
    {
        float *obs=bopl_obs_row(env, a);
        int w=-1;
        for (int i=0; i<env->written_count; i++) if (env->written[i].row==obs) { w=i; break; }
        int prev_players=w<0?-1:env->written[w].players;
        int prev_slots=w<0?-1:env->written[w].slots;
        int prev_entities=w<0?-1:env->written[w].entities;
        int prev_slimes=w<0?-1:env->written[w].slimes;
        const BoplRlPlayer *self=&state->players[a];
        double self_x=bopl_fix(self->x), self_y=bopl_fix(self->y);
        const BoplRlSlime *controller=bopl_controlling_slime(state, self->id, self_x, self_y);

        uint32_t nearest[BOPL_OBS_ENTITIES]; double distances[BOPL_OBS_ENTITIES];
        int count=0, active_count=0, n_nonstatic=0, n_pickup=0, sectors[8]={0}, source_counts[BOPL_ABILITY_COUNT]={0};
        uint32_t time_stop_remaining=0; int time_stop_known=0;
        for (uint32_t i=0; i<state->header.entity_count;) {
            int end=(state->entities[i].kind==3)?bopl_rope_group_end(state, i):(int)i+1;
            int group=end-(int)i;
            for (int j=0; j<group; j++) {
                const BoplRlEntity *entity=&state->entities[i+j];
                if (entity->kind==3 && !bopl_rope_span_kept(j, group)) continue;
                if (!bopl_entity_visible(state, entity, self->id)) continue;
                double distance=bopl_surface_distance(entity, self_x, self_y);
                if (entity->flags&BOPL_ENTITY_PICKUP) distance-=1.0e12;
                bopl_nearest(nearest, distances, &count, BOPL_OBS_ENTITIES, i+(uint32_t)j, distance);
                active_count++;
                uint32_t source=entity->reserved[0];
                if (source>0 && source<=BOPL_ABILITY_COUNT) source_counts[source-1]++;
                int nonstatic=entity->inverse_mass>0;
                if (nonstatic) {
                    n_nonstatic++;
                    double dx=bopl_fix(entity->x)-self_x, dy=bopl_fix(entity->y)-self_y;
                    double angle=atan2(dy, dx)+3.14159265358979323846;
                    int sector=(int)(angle*(8.0/6.28318530717958647692));
                    if (sector<0) sector=0; else if (sector>7) sector=7;
                    sectors[sector]++;
                }
                if (entity->flags&BOPL_ENTITY_PICKUP) n_pickup++;
                if (!time_stop_known && state->header.player_time_scale==0
                    && (entity->flags&BOPL_ENTITY_TIMER_KNOWN) && entity->clock_rate>0) {
                    time_stop_remaining=entity->reserved[1]; time_stop_known=1;
                }
            }
            i=(uint32_t)end;
        }

        uint32_t nearby_slimes[BOPL_OBS_SLIMES]; double slime_distances[BOPL_OBS_SLIMES];
        int slime_count=0, visible_slimes=0;
        uint32_t slot_indices[BOPL_OBS_SLOTS]; double slot_distances[BOPL_OBS_SLOTS];
        int slot_count=0, visible_slots=0;
        for (uint32_t i=0; i<state->header.slime_count; i++) {
            const BoplRlSlime *slime=&state->slimes[i];
            if (bopl_spatial_hidden(state, slime->owner, self->id)) continue;
            double dx=bopl_fix(slime->x)-self_x, dy=bopl_fix(slime->y)-self_y;
            double distance=dx*dx+dy*dy;
            bopl_nearest(nearby_slimes, slime_distances, &slime_count,
                         BOPL_OBS_SLIMES, i, distance);
            visible_slimes++;
            for (int slot=0; slot<3; slot++) {
                uint32_t id=slime->abilities[slot], flags=slime->ability_flags[slot];
                if (!(flags&1) || id==0 || id>BOPL_ABILITY_COUNT) continue;
                double priority=distance+slot*1.0e-9;
                if (slime==controller) priority-=1.0e12;
                bopl_nearest(slot_indices, slot_distances, &slot_count,
                             BOPL_OBS_SLOTS, i*3u+(uint32_t)slot, priority);
                visible_slots++;
            }
        }

        obs[0]=bopl_clip(bopl_fix(state->header.simulation_time)/120.0);
        obs[1]=bopl_clip(bopl_fix(state->header.player_time_scale));
        obs[2]=bopl_clip((double)state->header.episode_ticks/env->config.max_episode_ticks);
        obs[3]=state->header.player_count/6.0f;
        obs[4]=bopl_clip((double)active_count/BOPL_OBS_ENTITIES);
        obs[5]=bopl_clip((double)(active_count-count)/BOPL_OBS_ENTITIES);
        obs[6]=bopl_clip((double)visible_slimes/BOPL_OBS_SLIMES);
        obs[7]=bopl_clip((double)(visible_slimes>BOPL_OBS_SLIMES?visible_slimes-BOPL_OBS_SLIMES:0)/BOPL_OBS_SLIMES);
        obs[8]=bopl_clip(self_x/64.0); obs[9]=bopl_clip(self_y/64.0);
        double level_time=bopl_fix(state->header.simulation_time_since_level_loaded);
        obs[10]=bopl_clip(fmax(0.0, 120.0-level_time)/120.0);
        obs[11]=state->header.sudden_death!=0;
        obs[12]=state->header.sudden_death?bopl_clip(fmax(0.0, level_time-120.0)/30.0):0.0f;
        obs[13]=bopl_clip((env->config.max_episode_ticks-(double)state->header.episode_ticks)
                          /env->config.max_episode_ticks);
        for (int id=0; id<BOPL_ABILITY_COUNT; id++) obs[14+id]=bopl_clip(source_counts[id]/(double)BOPL_COUNT_SCALE);
        obs[53]=bopl_clip(n_nonstatic/(double)BOPL_OBS_ENTITIES);
        obs[54]=bopl_clip(n_pickup/(double)BOPL_COUNT_SCALE);
        for (int sector=0; sector<8; sector++) obs[55+sector]=bopl_clip(sectors[sector]/(double)BOPL_COUNT_SCALE);
        obs[63]=time_stop_known?bopl_clip(time_stop_remaining/300.0):0.0f;
        obs[64]=time_stop_known;
        obs[65]=controller?((controller->flags>>BOPL_SLIME_PICKUP_SLOT_SHIFT)&BOPL_SLIME_PICKUP_SLOT_MASK)/2.0f:0.0f;
        if (count_overflow) bopl_note_overflow(env, active_count, count, visible_slimes, visible_slots);

        bopl_player_features(obs+BOPL_OBS_SELF_OFFSET, state, self, self, self_x, self_y);
        uint32_t player_count=state->header.player_count<BOPL_OBS_PLAYERS?state->header.player_count:BOPL_OBS_PLAYERS;
        for (uint32_t i=0; i<player_count; i++) {
            uint32_t index=i==0?(uint32_t)a:(i<=(uint32_t)a?i-1:i);
            bopl_player_features(obs+BOPL_OBS_PLAYER_OFFSET+(size_t)i*BOPL_PLAYER_FEATURES,
                                 state, &state->players[index], self, self_x, self_y);
        }
        bopl_zero_tail(obs+BOPL_OBS_PLAYER_OFFSET, BOPL_PLAYER_FEATURES,
                       (int)player_count, prev_players, BOPL_OBS_PLAYERS);

        for (int i=0; i<slot_count; i++) {
            uint32_t packed=slot_indices[i], slime_index=packed/3, slot=packed%3;
            const BoplRlSlime *slime=&state->slimes[slime_index];
            uint32_t flags=slime->ability_flags[slot], id=slime->abilities[slot];
            double cooldown=bopl_fix(slime->cooldowns[slot]), timer=bopl_fix(slime->timers[slot]);
            float row[BOPL_SLOT_FEATURES]={0};
            row[0]=1; row[1]=slime->owner==self->id; row[2]=bopl_ally(state, slime->owner, self);
            row[3]=slot/2.0f; row[4]=bopl_clip(cooldown>0?timer/cooldown:0.0);
            row[5]=bopl_clip(cooldown/60.0); row[6]=(flags&2)!=0; row[7]=(flags&4)!=0;
            row[8]=(flags&BOPL_ABILITY_ACTIVE)!=0;
            row[10]=id==4;
            row[9]=row[10]?((flags>>BOPL_ABILITY_CHARGE_SHIFT)&BOPL_ABILITY_CHARGE_MASK)/4.0f:0.0f;
            row[11]=bopl_clip((bopl_fix(slime->x)-self_x)/64.0);
            row[12]=bopl_clip((bopl_fix(slime->y)-self_y)/64.0);
            row[BOPL_SLOT_ABILITY_ID]=(float)id;
            memcpy(obs+BOPL_OBS_SLOT_OFFSET+(size_t)i*BOPL_SLOT_FEATURES, row, sizeof(row));
        }
        bopl_zero_tail(obs+BOPL_OBS_SLOT_OFFSET, BOPL_SLOT_FEATURES,
                       slot_count, prev_slots, BOPL_OBS_SLOTS);

        for (int i=0; i<count; i++) {
            const BoplRlEntity *entity=&state->entities[nearest[i]];
            float row[BOPL_ENTITY_FEATURES]={0};
            row[0]=1; if (entity->kind<5) row[1+entity->kind]=1;
            row[6]=entity->owner==self->id; row[7]=bopl_ally(state, entity->owner, self);
            row[8]=entity->owner!=0 && !row[6] && !row[7];
            row[9]=bopl_clip((bopl_fix(entity->x)-self_x)/64.0);
            row[10]=bopl_clip((bopl_fix(entity->y)-self_y)/64.0);
            row[11]=bopl_clip(bopl_fix(entity->vx)/256.0); row[12]=bopl_clip(bopl_fix(entity->vy)/256.0);
            row[13]=bopl_clip(bopl_fix(entity->right_x)/64.0); row[14]=bopl_clip(bopl_fix(entity->right_y)/64.0);
            row[15]=bopl_clip(bopl_fix(entity->up_x)/64.0); row[16]=bopl_clip(bopl_fix(entity->up_y)/64.0);
            row[17]=bopl_clip(bopl_fix(entity->radius)/16.0);
            row[18]=(entity->flags&BOPL_ENTITY_ENABLED)!=0;
            row[19]=(entity->flags&BOPL_ENTITY_TRIGGER)!=0;
            row[20]=(entity->flags&BOPL_ENTITY_COMPOSITE)!=0;
            row[21]=entity->layer/32.0f;
            row[22]=(entity->flags&BOPL_ENTITY_TIMER_KNOWN)?bopl_clip(entity->reserved[1]/(double)BOPL_TIMER_TICKS):0.0f;
            row[23]=(entity->flags&BOPL_ENTITY_TIMER_KNOWN)!=0;
            row[24]=bopl_clip(bopl_fix(entity->clock_rate));
            row[25]=(entity->flags&BOPL_ENTITY_PICKUP)!=0;
            row[26]=(entity->flags&BOPL_ENTITY_PROJECTILE)!=0;
            row[27]=(entity->flags&BOPL_ENTITY_SPARK_NODE)!=0;
            row[28]=(entity->flags&BOPL_ENTITY_FLAMMABLE_SMOKE)!=0;
            row[29]=(entity->flags&BOPL_ENTITY_EXPLOSIVE)!=0;
            row[30]=(entity->flags&BOPL_ENTITY_HITBOX)!=0;
            row[31]=(entity->flags&BOPL_ENTITY_CONTACT_DAMAGE)!=0;
            row[32]=(entity->flags&BOPL_ENTITY_CONTACT_DAMAGE_KNOWN)!=0;
            row[BOPL_ENTITY_ABILITY_ID]=entity->reserved[0]<=BOPL_ABILITY_COUNT?(float)entity->reserved[0]:0.0f;
            memcpy(obs+BOPL_OBS_ENTITY_OFFSET+(size_t)i*BOPL_ENTITY_FEATURES, row, sizeof(row));
        }
        bopl_zero_tail(obs+BOPL_OBS_ENTITY_OFFSET, BOPL_ENTITY_FEATURES,
                       count, prev_entities, BOPL_OBS_ENTITIES);

        for (int i=0; i<slime_count; i++) {
            const BoplRlSlime *slime=&state->slimes[nearby_slimes[i]];
            float row[BOPL_SLIME_FEATURES]={0};
            row[0]=1; row[1]=slime->owner==self->id; row[2]=bopl_ally(state, slime->owner, self);
            row[3]=(slime->flags&BOPL_SLIME_ACTIVE)!=0; row[4]=(slime->flags&BOPL_SLIME_SPAWNED)!=0;
            row[5]=(slime->flags&BOPL_SLIME_IN_ABILITY)!=0; row[6]=(slime->flags&BOPL_SLIME_SUPPRESS_INPUT)!=0;
            row[7]=bopl_clip((bopl_fix(slime->x)-self_x)/64.0); row[8]=bopl_clip((bopl_fix(slime->y)-self_y)/64.0);
            row[9]=bopl_clip(bopl_fix(slime->vx)/256.0); row[10]=bopl_clip(bopl_fix(slime->vy)/256.0);
            row[11]=bopl_clip(bopl_fix(slime->scale)/3.0);
            if (slime->state<11) row[12+slime->state]=1;
            row[23]=sinf((float)bopl_fix(slime->rotation)); row[24]=cosf((float)bopl_fix(slime->rotation));
            row[28]=(slime->flags&BOPL_SLIME_ABILITYLESS)!=0; row[29]=1;
            memcpy(obs+BOPL_OBS_SLIME_OFFSET+(size_t)i*BOPL_SLIME_FEATURES, row, sizeof(row));
        }
        bopl_zero_tail(obs+BOPL_OBS_SLIME_OFFSET, BOPL_SLIME_FEATURES,
                       slime_count, prev_slimes, BOPL_OBS_SLIMES);
        memset(obs+BOPL_OBS_PREVIOUS_OFFSET, 0, BOPL_PREVIOUS_FEATURES*sizeof(float));
        if (w<0) {
            if (env->written_count==BOPL_MAX_SLOTS) env->written[0]=env->written[--env->written_count];
            w=env->written_count++; env->written[w].row=obs;
        }
        env->written[w].players=(int)player_count; env->written[w].slots=slot_count;
        env->written[w].entities=count; env->written[w].slimes=slime_count;
    }
}
/* Encode `state` into every player's row in that player's layout. A NULL
 * state encodes as all zeros. Defined after the v4 writer below. */
static void bopl_encode(BoplPuffer *env, const BoplRlState *state, int count_overflow);
/* The undelayed encoder: the state as observed now, overflow counted. */
static void compute_observations_from(BoplPuffer *env, const BoplRlState *state) { bopl_encode(env, state, 1); }
static void compute_observations(BoplPuffer *env) { compute_observations_from(env, bopl_rl_state(env->world)); }
/* Own-action tail (BOPL_PREVIOUS_FEATURES): the decision as applied to the game
 * (latched buttons, live aim), the latch remaining after it, and how long each
 * button has been held. hold_left/hold_run may be NULL (no latch state). */
static float bopl_aim_table[256][2];
static pthread_once_t bopl_aim_table_once=PTHREAD_ONCE_INIT;
static void bopl_aim_table_init(void) {
    for (int angle=1; angle<256; angle++) {
        double radians=(angle-1)*6.283185307179586/255.0;
        bopl_aim_table[angle][0]=cos(radians); bopl_aim_table[angle][1]=sin(radians);
    }
}
static void bopl_previous_action(float *previous, BoplRlAction action, int hold_left, const int *hold_run) {
    /* The 255 aim directions are fixed: the same cos/sin, evaluated once. */
    pthread_once(&bopl_aim_table_once, bopl_aim_table_init);
    previous[0]=action.angle==0;
    previous[1]=bopl_aim_table[action.angle][0]; previous[2]=bopl_aim_table[action.angle][1];
    for (int bit=0; bit<4; bit++) previous[3+bit]=(action.buttons&(1<<bit))!=0;
    previous[7]=bopl_clip((double)hold_left/BOPL_HOLD_MAX);
    for (int bit=0; bit<4; bit++) previous[8+bit]=bopl_clip((hold_run?hold_run[bit]:0)/(double)BOPL_HOLD_MAX);
}
/* Layout v4: the two decisions before the last one behind the v3 tail (7
 * floats each: no-aim flag, aim cos/sin, four buttons), then the history
 * shifts to include `applied`. */
static void bopl4_previous_history(float *previous, BoplRlAction *history, BoplRlAction applied) {
    for (int h=0; h<2; h++) {
        float *out=previous+BOPL_PREVIOUS_FEATURES+7*h;
        out[0]=history[h].angle==0;
        out[1]=bopl_aim_table[history[h].angle][0]; out[2]=bopl_aim_table[history[h].angle][1];
        for (int bit=0; bit<4; bit++) out[3+bit]=(history[h].buttons&(1<<bit))!=0;
    }
    history[1]=history[0]; history[0]=applied;
}
#include "bopl_obs_v4.h"
static void bopl_encode(BoplPuffer *env, const BoplRlState *state, int count_overflow) {
    for (int a=0; a<env->players; a++) {
        if (!state) {
            float *obs=bopl_obs_row(env, a);
            memset(obs, 0, bopl_obs_size_of(env, a)*sizeof(float)); bopl_written_forget_row(env, obs);
        } else if (bopl_layout_of(env, a)==4) bopl_encode_v4_row(env, state, a, count_overflow);
        else bopl_encode_v3_row(env, state, a, count_overflow);
    }
}
/* The critic-only block of every v4 row from the undelayed state `now`
 * (bopl4_privileged); `actions` may be NULL (episode start). */
static void bopl_encode_privileged(BoplPuffer *env, const BoplRlState *now, const BoplRlAction *actions) {
    for (int a=0; a<env->players; a++) {
        if (bopl_layout_of(env, a)!=4) continue;
        int enemy=-1;
        if (now) for (int i=0; i<env->players && enemy<0; i++)
            if (i!=a && now->players[i].team!=now->players[a].team) enemy=i;
        bopl4_privileged(env, now, a, enemy>=0 && actions?&actions[enemy]:NULL,
                         enemy>=0?&env->hold_left[enemy]:NULL, enemy>=0?&env->held[enemy]:NULL);
    }
}
/* Apply the hold head: a free button state adopts the head's buttons and
 * latches them for BOPL_HOLD_LENGTHS[hold] decisions; a latched one ignores
 * the button head. Returns the buttons to send; updates latch bookkeeping. */
static uint16_t bopl_latch(BoplPuffer *env, int player, uint16_t buttons, int hold, int *fresh) {
    if (env->hold_left[player]>0) { env->hold_left[player]--; *fresh=0; }
    else { env->held[player]=buttons; env->hold_left[player]=BOPL_HOLD_LENGTHS[hold]-1; *fresh=1; }
    uint16_t applied=env->held[player];
    for (int bit=0; bit<4; bit++) env->hold_run[player][bit]=(applied&(1<<bit))?env->hold_run[player][bit]+1:0;
    return applied;
}
static int bopl_bow_button(const BoplPuffer *env, int player) {
    const BoplRlState *state=bopl_rl_state(env->world);
    if (!state || player<0 || player>=(int)state->header.player_count) return -1;
    const BoplRlPlayer *self=&state->players[player];
    const BoplRlSlime *controller=bopl_controlling_slime(
        state, self->id, bopl_fix(self->x), bopl_fix(self->y));
    if (!controller) return -1;
    for (int slot=0; slot<3; slot++)
        if ((controller->ability_flags[slot]&1) && controller->abilities[slot]==4) return slot+1;
    return -1;
}
/* Publish the action mask for player i's next decision from its latch state. */
static void bopl_write_mask(BoplPuffer *env, int player) {
    unsigned char *mask=env->mask_ptr[bopl_slot(env, player)];
    if (!mask) return;
    memset(mask, 1, BOPL_LOGITS);
    if (env->hold_left[player]>0) {
        memset(mask+BOPL_ACT_AIM, 0, BOPL_ACT_BUTTONS+BOPL_ACT_HOLD);
        mask[BOPL_ACT_AIM+env->held[player]]=1;
        mask[BOPL_ACT_AIM+BOPL_ACT_BUTTONS]=1;
    } else {
        int bow=bopl_bow_button(env, player);
        if (bow>=0 && env->hold_run[player][bow]>0)
            for (int b=0; b<BOPL_ACT_BUTTONS; b++)
                if (b&(1<<bow)) mask[BOPL_ACT_AIM+b]=0;
    }
}
static uint32_t bopl_side_random(BoplPuffer *env) {
    uint32_t x=env->side_rng; x^=x<<13; x^=x>>17; x^=x<<5; return env->side_rng=x;
}
static void bopl_failed_world(BoplPuffer *env, const BoplRl *world, const char *operation) {
    env->failed=1; env->log.game_errors++;
    snprintf(env->error, sizeof(env->error), "%s: %s", operation, bopl_rl_error(world));
    fprintf(stderr, "Bopl RL stopped: %s\n", env->error);
    /* Do not silently turn failed simulation into successful training episodes. */
    abort();
}
static void bopl_failed(BoplPuffer *env, const char *operation) { bopl_failed_world(env, env->world, operation); }
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
    if (!env->world) { fprintf(stderr, "Bopl RL startup: %s\n", env->world_error); abort(); }
    if (env->world_checked) return;
    env->world_checked=1;
    for (uint32_t map=0; map<BOPL_COUNTER_MAPS; map++)
        if ((env->training_map_mask&(1ull<<map)) && map>=bopl_rl_map_count(env->world)) {
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
    if (pthread_create(&env->world_thread, NULL, bopl_world_open, env)) abort();
    env->world_opening=1;
    if (env->players<1 || env->players>BOPL_MAX_PLAYERS || env->banks<0 || env->banks>BOPL_MAX_BANKS
        || (env->banks && env->players!=2) || env->num_agents!=(env->banks?1+env->banks:env->players)) {
        fprintf(stderr, "Bopl RL slot layout: players %d, banks %d, slots %d\n", env->players, env->banks, env->num_agents); abort();
    }
    if (env->banks && (env->active_bank<0 || env->active_bank>=env->banks)) { fprintf(stderr, "Bopl RL: historical environment without a bank\n"); abort(); }
    for (int bank=0; bank<BOPL_MAX_BANKS; bank++) {
        uint32_t kind=(env->scripted_kinds>>(4*bank))&15u;
        if (kind>BOPL_BOT_KINDS) { fprintf(stderr, "Bopl RL: unknown scripted bot kind %u\n", kind); abort(); }
    }
    if (env->scripted_level<0 || env->scripted_level>=BOPL_BOT_LEVELS) { fprintf(stderr, "Bopl RL: scripted bot level\n"); abort(); }
    if (!isfinite(env->reward_kill) || !isfinite(env->reward_timeout) || env->reward_timeout>0) { fprintf(stderr, "Bopl RL reward config\n"); abort(); }
    if (env->observation_delay!=0 && (env->observation_delay!=9 || env->repeat!=4)) abort();
    /* The spare is a second runtime (~240 MB) for an in-process game, the
     * same memory as its second process for a socket game; BOPL_RL_INPROC_SPARE=0
     * drops it and pays every reset (~8 ms) on the lockstep instead. */
    const char *inproc_spare=getenv("BOPL_RL_INPROC_SPARE");
    if (!env->no_autoreset && !(bopl_rl_in_process(game) && inproc_spare && inproc_spare[0]=='0')) {
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
        for (uint32_t map=0; map<BOPL_COUNTER_MAPS; map++) count+=(uint32_t)((env->training_map_mask>>map)&1u);
        uint32_t selected=bopl_random(env)%count;
        for (uint32_t map=0; map<BOPL_COUNTER_MAPS; map++) if (env->training_map_mask&(1ull<<map)) {
            if (selected==0) { config->map=map; break; }
            selected--;
        }
    } else if (env->random_maps) config->map=bopl_random(env)%bopl_rl_map_count(env->world);
    if (env->random_loadouts) for (int i=0; i<env->players; i++) for (uint32_t j=0; j<config->slots; j++)
        config->player[i].abilities[j]=1+bopl_random(env)%(BOPL_ABILITY_COUNT-1); // Exclude the source's NullAbility placeholder.
}
/* The decision-loop half of c_reset, from the episode's first state: first
 * observation, the delay history primed with it, latches cleared, the neutral
 * own-action tail and the masks. Shared with the agent shim (bopl_agent.c),
 * which is fed states from a running game instead of a world it steps. */
static void bopl_reset_observations(BoplPuffer *env, const BoplRlState *state) {
    memset(env->applied, 0, sizeof(env->applied));   /* before the rows: no aim ray at the episode start */
    compute_observations_from(env, state);
    /* The first two delayed decisions observe this same first state. */
    for (int i=0; i<2; i++) bopl_state_copy(&env->delayed[i], state);
    env->delay_cursor=0;
    memset(env->hold_left, 0, sizeof(env->hold_left)); memset(env->held, 0, sizeof(env->held)); memset(env->hold_run, 0, sizeof(env->hold_run));
    memset(env->action_history, 0, sizeof(env->action_history));
    for (int i=0; i<env->players; i++) {
        float *previous=bopl_obs_row(env, i)+bopl_previous_offset(env, i);
        previous[0]=1;
        if (bopl_layout_of(env, i)==4) for (int h=0; h<BOPL4_HISTORY; h++) previous[BOPL_PREVIOUS_FEATURES+BOPL4_HISTORY_FEATURES*h]=1;
        bopl_write_mask(env, i);
    }
    bopl_encode_privileged(env, state, NULL);
}
/* Weighted roster draw from the bot stream. Same LCG as the share coin-flip
 * so (seed, map, weights) still determine the opponent. */
static int bopl_draw_roster(uint32_t *rng) {
    int n=bopl_roster_n;
    if (n<1) return -1;
    if (n>BOPL_ROSTER_MAX) n=BOPL_ROSTER_MAX;
    float total=0.f;
    for (int i=0; i<n; i++) total+=bopl_roster_weight[i];
    *rng=*rng*747796405u+2891336453u;
    float u=(float)(*rng>>8)*(1.0f/16777216.0f);
    if (!(total>0.f) || !isfinite(total)) return n-1;
    float x=u*total, acc=0.f;
    for (int i=0; i<n; i++) {
        acc+=bopl_roster_weight[i];
        if (x<acc) return i;
    }
    return n-1;
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
    /* The scripted-bot stream (the share draw below) restarts from this
     * episode's seed, like the bot seeds (bopl_bot_seed): nothing a bot does
     * depends on the episodes played before. */
    env->bot_rng=env->config.seed*2654435761u^0x9E3779B9u;
    if (env->banks) {
        /* Every bank row goes idle (blank observation; the finished match's
         * opponent state was reset by the terminal flag) and the new match
         * draws its side. Blanking all rows, not just the active one, also
         * clears rows that held another slot's observation before the
         * trainer's agent permutation was installed (create_pufferl resets
         * under identity routing). */
        for (int bank=0; bank<env->banks; bank++) {
            memset(env->obs_ptr[1+bank], 0, BOPL_OBS_STRIDE*sizeof(float)); bopl_written_forget_row(env, env->obs_ptr[1+bank]);
            if (env->mask_ptr[1+bank]) memset(env->mask_ptr[1+bank], 1, BOPL_LOGITS);
        }
        env->side=(int)(bopl_side_random(env)&1u);
        /* A fresh recurrent state for the row this match's opponent runs on. */
        *env->terminal_ptr[1+env->active_bank]=1;
        /* A scripted bank drives this match's opponent player from C; the
         * bank's GPU row still steps (fixed graph) but its actions are unread. */
        memset(env->bot_select, 0, sizeof(env->bot_select));
        env->roster_pick=-1;
        uint32_t kind=(env->scripted_kinds>>(4*env->active_bank))&15u;
        if (kind) {
            /* Below full share, a draw from the bot stream decides whether this
             * match is the bot's or the bank's frozen weights'. */
            float share=bopl_scripted_share;
            if (share<1.0f) {
                env->bot_rng=env->bot_rng*747796405u+2891336453u;
                if ((float)(env->bot_rng>>8)*(1.0f/16777216.0f)>=share) kind=0;
            }
        }
        if (kind) {
            int pick=bopl_draw_roster(&env->bot_rng);
            if (pick>=0) {
                kind=bopl_roster_kind[pick];
                env->bot_select_level[1-env->side]=(uint8_t)bopl_roster_level[pick];
                env->roster_pick=pick;
            } else {
                env->bot_select_level[1-env->side]=(uint8_t)env->scripted_level;
            }
            env->bot_select[1-env->side]=(uint8_t)kind;
        }
    }
    for (int i=0; i<env->players; i++) if (env->bot_select[i])
        bopl_bot_reset(&env->bots[i], env->bot_select[i], env->bot_select_level[i], bopl_bot_seed(env, i));
    env->boundary_reached=1; env->matches++;
    bopl_reset_observations(env, state);
}
/* c_step in three phases, so the research binding can advance the world one
 * tick at a time (research_step_tick, for callers that need per-tick state)
 * with exactly the training bookkeeping: begin resets the per-decision
 * telemetry/rewards and turns the routed or scripted actions into latched
 * game actions; advance runs the decision's ticks (sampling the observation
 * at phase 3 when delayed); finish credits rewards, telemetry, terminals, the
 * delayed observation and the masks. c_step = begin, advance, finish. */
typedef struct { BoplRlAction actions[6]; int fresh[6], holds[6]; uint32_t ticks; } BoplStepPhase;
/* The _from variants take the world state explicitly (the state before the
 * decision for begin, the state after its ticks for finish) so the agent shim
 * (bopl_agent.c) can run this exact bookkeeping on states a running game
 * hands it; the plain variants read it from env->world. */
static void bopl_step_begin_from(BoplPuffer *env, BoplStepPhase *phase, const BoplRlState *before) {
    float *telemetry=env->telemetry?env->telemetry:env->last_telemetry;
    memset(telemetry, 0, (size_t)env->players*BOPL_TELEMETRY_SIZE*sizeof(float));
    for (int s=0; s<env->num_agents; s++) { *env->reward_ptr[s]=0; *env->terminal_ptr[s]=0; }
    env->stepped++;
    memset(phase, 0, sizeof(*phase));
    BoplRlAction *actions=phase->actions;
    int *fresh=phase->fresh, *holds=phase->holds;
    for (int i=0; i<env->players; i++) {
        float angle, buttons, hold;
        if (env->bot_select[i]) {
            /* Scripted player: decide from the true current state through the
             * same latch/mask constraints; the routed slot's action is unread. */
            uint8_t decided[3];
            env->bots[i].world=env->world;  /* v2 kinds fork this for rollouts (bopl_bots_v2.h) */
            bopl_bot_decide(&env->bots[i], before, i, env->hold_left[i], env->held[i], env->hold_run[i], decided);
            angle=decided[0]; buttons=decided[1]; hold=decided[2];
        } else {
            const float *action=env->action_ptr[bopl_slot(env, i)];
            angle=action[0]; buttons=action[1]; hold=action[2];
        }
        if (!isfinite(angle) || !isfinite(buttons) || !isfinite(hold) || angle<0 || angle>255 || buttons<0 || buttons>15
            || hold<0 || hold>BOPL_ACT_HOLD-1 || floorf(angle)!=angle || floorf(buttons)!=buttons || floorf(hold)!=hold) {
            fprintf(stderr, "Invalid PufferLib action\n"); abort();
        }
        holds[i]=(int)hold;
        actions[i].angle=(uint8_t)angle;
        actions[i].buttons=bopl_latch(env, i, (uint16_t)buttons, holds[i], &fresh[i]);
    }
    /* The inputs as sent this step, for the next decision's foe model (v2 kinds only). */
    for (int i=0; i<env->players; i++) if (env->bot_select[i]) memcpy(env->bots[i].applied, phase->actions, sizeof(phase->actions));
    phase->ticks=before->header.episode_ticks;
}
static void bopl_step_begin(BoplPuffer *env, BoplStepPhase *phase) {
    const BoplRlState *before=bopl_rl_state(env->world);
    if (!before) bopl_failed(env, "state");
    bopl_step_begin_from(env, phase, before);
}
/* The delayed observation's source: the world state at phase 3 of the decision
 * (or the final state when the episode ended earlier). */
static void bopl_step_sample(BoplPuffer *env, const BoplRlState *state) {
    bopl_state_copy(&env->sample_copy, state);
    bopl_count_overflow(env, state);
}
static void bopl_step_advance(BoplPuffer *env, const BoplStepPhase *phase) {
    const BoplRlAction *actions=phase->actions;
    // Sample at phase 3 of each four-tick decision. At t=12 return the
    // observation from t=3, giving an exact nine-tick delay thereafter.
    if (env->observation_delay) {
        // Both phases in one game roundtrip: the phase-3 state is returned
        // alongside the final state (identical when the episode ends in phase 1).
        if (bopl_rl_step_sampled(env->world, actions, env->players, 3, 1)) bopl_failed(env, "sampled step");
        bopl_step_sample(env, bopl_rl_sample(env->world));
    } else if (bopl_rl_step(env->world, actions, env->players, (uint32_t)env->repeat)) bopl_failed(env, "step");
}
static void bopl_step_finish_from(BoplPuffer *env, const BoplStepPhase *phase, const BoplRlState *state) {
    float *telemetry=env->telemetry?env->telemetry:env->last_telemetry;
    const BoplRlAction *actions=phase->actions;
    const int *fresh=phase->fresh, *holds=phase->holds;
    uint32_t ticks=phase->ticks;
    for (int i=0; i<env->players; i++) env->applied[i]=actions[i];
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
            if (fresh[i]) env->log.hold_choice[holds[i]]++; else env->log.latched_decisions++;
        } else c->opponent_game_decisions++;
        env->deaths[i]=player->deaths; env->kills[i]=player->kills;
        if (terminal) {
            if (env->config.map<BOPL_COUNTER_MAPS) env->log.map_fraction[env->config.map]++;
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
                    int scripted=0;
                    for (int p=0; p<env->players; p++) scripted|=env->bot_select[p];
                    if (scripted) {
                        env->log.bot_n_bank[env->active_bank]++; env->log.bot_score_bank[env->active_bank]+=won;
                        /* Timeouts are not roster games: Elo and per-entry
                         * winrate count decided matches only. */
                        if (!truncated && env->roster_pick>=0 && env->roster_pick<BOPL_ROSTER_MAX) {
                            env->log.bot_n_entry[env->roster_pick]++; env->log.bot_score_entry[env->roster_pick]+=won;
                        }
                    }
                    else if (env->boundary_reached) { env->log.hist_n_bank[env->active_bank]++; env->log.hist_score_bank[env->active_bank]+=won; }
                }
            }
        }
    }
    if (terminal) { env->counters.matches++; if (env->config.map<BOPL_COUNTER_MAPS) env->counters.map_matches[env->config.map]++; }
    if (terminal && !env->no_autoreset) { c_reset(env); return; }
    if (env->observation_delay) {
        /* Publish the state sampled two decisions ago (already counted for
         * overflow when sampled), then keep this decision's sample in its
         * place: the buffers swap, nothing is copied. */
        BoplStateCopy *old=&env->delayed[env->delay_cursor];
        bopl_encode(env, bopl_state_copy_get(old), 0);
        BoplStateCopy published=*old; *old=env->sample_copy; env->sample_copy=published;
        env->delay_cursor=(env->delay_cursor+1)%2;
    } else compute_observations_from(env, state);
    if (!terminal) for (int i=0; i<env->players; i++) {
        // Own button history is known immediately; it is not world information.
        float *previous=bopl_obs_row(env, i)+bopl_previous_offset(env, i);
        bopl_previous_action(previous, actions[i], env->hold_left[i], env->hold_run[i]);
        if (bopl_layout_of(env, i)==4) bopl4_previous_history(previous, env->action_history[i], actions[i]);
        bopl_write_mask(env, i);
    }
    /* The critic's privileged block sees the world as it is now, not as the
     * actor's delayed rows do. */
    bopl_encode_privileged(env, state, actions);
}
static void bopl_step_finish(BoplPuffer *env, const BoplStepPhase *phase) {
    bopl_step_finish_from(env, phase, bopl_rl_state(env->world));
}
static void c_step(BoplPuffer *env) {
    BoplStepPhase phase;
    bopl_step_begin(env, &phase);
    bopl_step_advance(env, &phase);
    bopl_step_finish(env, &phase);
}
static void c_render(BoplPuffer *env) {
    (void)env;
    fprintf(stderr, "This worker is headless; export an episode with bopl_rl_replay and play it in the game.\n");
}
static void c_close(BoplPuffer *env) {
    if (env->world_opening) { pthread_join(env->world_thread, NULL); env->world_opening=0; }
    if (env->spare_opening) { pthread_join(env->spare_thread, NULL); env->spare_opening=0; }
    bopl_rl_close(env->world); env->world=NULL;
    bopl_rl_close(env->spare); env->spare=NULL;
    for (int i=0; i<BOPL_MAX_PLAYERS; i++) { bopl_rl_close(env->bots[i].fork); env->bots[i].fork=NULL; }
    for (int i=0; i<2; i++) bopl_state_copy_free(&env->delayed[i]);
    bopl_state_copy_free(&env->sample_copy);
}
#endif
