#ifndef BOPL_PUFFER_H
#define BOPL_PUFFER_H
#include "bopl_rl.h"
#include "bopl_layout.h"
#include "bopl_layout_v4.h"
#include "bopl_obs_pack.h"
#include "bopl_rows_v4.h"
#include "bopl_bots.h"
#include "bopl_style.h"
#include <math.h>
#include <stdint.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
/* Builds that must outlive a failed step define BOPL_PUFFER_RECOVER: the
 * in-game agent shim (bopl_agent.c) runs its decision-time search on forks of
 * a shadow sim inside the running game, where a broken rollout must turn the
 * search off, not abort the game. A failure then longjmps to
 * bopl_recover_point when the caller armed one; everything else aborts as
 * before. */
#ifdef BOPL_PUFFER_RECOVER
#include <setjmp.h>
static _Thread_local jmp_buf *bopl_recover_point;
#define BOPL_RECOVER() do { if (bopl_recover_point) longjmp(*bopl_recover_point, 1); } while (0)
#else
#define BOPL_RECOVER() do {} while (0)
#endif

enum { BOPL_EVENT_FIELDS=9, BOPL_TELEMETRY_SIZE=16,
       BOPL_REWARD_OUTCOME=9, BOPL_REWARD_KILLS, BOPL_REWARD_DEATHS,
       BOPL_EPISODES, BOPL_WINS, BOPL_TIMEOUTS, BOPL_TICKS };
typedef struct {
    float perf, score, episode_return, episode_length, environment_ticks, observation_overflow, game_errors, n;
    /* Learner-player outcomes, split by opponent kind (each averaged by n like
     * every other field, so win rate = wins/episodes). */
    float learner_episodes, learner_wins, historical_episodes, historical_wins;
    /* Per frozen bank, the fields the self-play pool reads (n-averaged like the
     * rest): matches the learner completed against bank b, the learner's wins
     * in them, and how many of those matches were decided. A match whose
     * opponent weights were swapped while it ran (see c_step) counts in none.
     * wins = hist_score_bank, draws = hist_n_bank - hist_decided_bank,
     * losses = hist_decided_bank - hist_score_bank: an undecided round (mutual
     * death) and a timeout are draws in the opponent statistic, not losses.
     * This is only how the pool CHOOSES opponents; the reward an undecided
     * round pays is unchanged (bopl_outcome: -1 for everyone in it). */
    float hist_n_bank[8], hist_score_bank[8], hist_decided_bank[8];
    /* Matches the learner played against a scripted bot in bank b,
     * kept out of hist_*_bank so the pool's swap/Elo statistics only see the
     * bank's frozen weights. */
    float bot_n_bank[8], bot_score_bank[8];
    /* Learner-vs-roster-entry games this window, and how many of them were
     * decided. Index is the process-wide roster slot pushed from train.py.
     * Same shape as hist_*_bank above and for the same reason: a timeout and an
     * undecided round (mutual death) are DRAWS against the entry, not losses
     * and not absences, so both must reach the denominator. Reading a rate as
     * score/n with timeouts dropped from n instead scores the entry only on the
     * games it managed to decide. */
    float bot_n_entry[32], bot_score_entry[32], bot_decided_entry[32];
    /* Hold head (learner decisions, n-averaged): how often each hold length
     * was chosen when the button state was free, and how many decisions the
     * button head was ignored because a latch was in force. */
    float hold_choice[BOPL_ACT_HOLD], latched_decisions;
    float map_fraction[124];   /* BOPL_COUNTER_MAPS */
    /* Team games (BoplPuffer.team_play), learner players, n-averaged like the
     * rest: episodes, wins, and kills of a teammate (bopl_team_kills). Always
     * zero without team play, and my_log only reports them when non-zero. */
    float team_episodes, team_wins, team_kills;
} Log;
enum { BOPL_MAX_PLAYERS=6, BOPL_MAX_BANKS=8, BOPL_MAX_SLOTS=BOPL_MAX_PLAYERS+BOPL_MAX_BANKS };
/* The authored catalogue is dense at 0..123. Selection is a real 128-bit mask;
 * binding.c reconstructs it from four exact 31-bit option chunks. */
enum { BOPL_COUNTER_MAPS=124,
       BOPL_COUNTER_SIZE=5+4+BOPL_TELEMETRY_SIZE+3*BOPL_ABILITY_COUNT+BOPL_COUNTER_MAPS };
typedef struct { uint64_t lo, hi; } BoplMapMask;
static int bopl_map_selected(BoplMapMask mask, uint32_t map) {
    return map<64?(int)((mask.lo>>map)&1u):(map<128?(int)((mask.hi>>(map-64))&1u):0);
}
static void bopl_map_select(BoplMapMask *mask, uint32_t map) {
    if (map<64) mask->lo|=1ull<<map; else if (map<128) mask->hi|=1ull<<(map-64);
}
static int bopl_maps_selected(BoplMapMask mask) { return mask.lo!=0 || mask.hi!=0; }
/* The LevelType.space maps, from native/data/map-set.json's level_type field
 * (22 of 124; eval/map_set.space_maps is the same set, and
 * eval/check_maps.py asserts they agree). Baked game data, as fixed as
 * BOPL_COUNTER_MAPS, so it is compiled in rather than passed per env: an
 * evaluation path that forgot to pass it would silently tell every policy it
 * was on a grass map. Read at the point of use from env->config.map, never
 * cached on the env: env->config is the live world's config on every path
 * (c_reset promotes spare_config into it on takeover), but a cached flag would
 * be wrong three ways - the spare pre-draws the NEXT episode's map over it, the
 * takeover path draws no config at all, and research_reset sets config.map
 * directly without drawing one. Space halves the gravity modifier on players and on
 * ropes, removes water, and moves the lower blast boundary to Camera_YMin;
 * snow is mechanically identical to grass, so only space is represented. */
static const BoplMapMask BOPL_SPACE_MAPS={0xfc00007e00000000ull, 0x0ffc000000000000ull};
typedef struct {
    double learner_steps;            /* learner decisions (self-play: every player; historical: the learner side) */
    double game_steps;               /* physical decisions of every player, learner or frozen */
    double matches;                  /* completed matches, once each */
    double opponent_game_decisions;  /* frozen-opponent decisions (game_steps - learner_steps) */
    double kills;                    /* game-credited kills by learner players */
    double buttons[4];               /* learner decisions holding jump, dash, arrow, platform */
    double telemetry[BOPL_TELEMETRY_SIZE]; /* learner-only sums in telemetry FIELDS order */
    double ability_activations[BOPL_ABILITY_COUNT]; /* learner slot presses, keyed by equipped ability id */
    double entity_rows[BOPL_ABILITY_COUNT];         /* visible candidate entity rows by source ability id */
    double entity_overflow[BOPL_ABILITY_COUNT];     /* candidates dropped beyond the active layout budget */
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
/* Explicit two-player lineup table installed by train.py before vector creation.
 * Three-ID file rows are expanded to the same six-byte representation there. */
enum { BOPL_LOADOUT_SET_MAX=256, BOPL_LOADOUT_FIXED=0, BOPL_LOADOUT_POOL=1, BOPL_LOADOUT_SET=2 };
int bopl_loadout_set_count=0;
uint8_t bopl_loadout_sets[BOPL_LOADOUT_SET_MAX*6];
/* Per-ability weights for pool draws, installed by train.py (--loadout-weights).
 * All zero keeps the uniform draw, whose random stream is unchanged. */
uint16_t bopl_loadout_weights[64];
/* The style program (bopl_style.h) every environment created after
 * bopl_style_install runs, and the fade scale its payouts are paid at (set
 * each update). Without a program nothing below touches the reward. */
BoplStyleProgram bopl_style_program;
int bopl_style_loaded=0;
float bopl_style_scale=1.0f;
/* A world state that outlives the game's own state buffer: header, players and
 * telemetry by value, entity and slime rows in owned buffers. Without a rope
 * (no kind-3 entity, dead or alive) the copy keeps the live entities alone
 * (active and not destroyed), in order, and its entity_count says how many: a
 * dead entity is in no row of either layout and in no overflow count (never
 * visible, bopl_entity_visible; no wall or water, bopl4_scene_class), and the
 * writers relate entities by position only (bopl_nearest keeps ties in index
 * order), so every row and count from the copy is the state's. Only rope
 * pieces see their neighbours (bopl_rope_group_end, bopl4_rope_continues), so
 * with a rope every entity is kept. Copied once per decision while the
 * sampled state is warm, the copy is then what a shipped row needs as it is:
 * one run of entities and, in source_mask/source_counts, the census of its
 * first BOPL4_ENTITIES entities by source ability (bit and index s-1 for
 * source s in 1..BOPL_ABILITY_COUNT), so publishing it two decisions later
 * reads no entity it does not ship (bopl4_shared_from_copy). The rows sit in
 * one buffer, image, laid out as a shipped state is in its slot after the
 * header and players: the entities, the slimes, then room for the slimes'
 * sinf/cosf the publish writes; a shipped row is the header and players
 * (fixed size, copied inline) and one copy of the image (bopl4_encode_state).
 * The state itself (header, players, telemetry) heads the same allocation,
 * so the copy is a few words and the publish's swap of three copies moves
 * no state. */
enum { BOPL_COPY_HEAD=(int)(sizeof(BoplRlHeader)+6*sizeof(BoplRlPlayer)), BOPL_COPY_STATE=(int)((sizeof(BoplRlState)+63)&~(size_t)63) };
_Static_assert(offsetof(BoplRlState, players)==sizeof(BoplRlHeader) && sizeof(((BoplRlState*)0)->players)==6*sizeof(BoplRlPlayer),
               "header and players are contiguous in the state, as in the slot");
_Static_assert(BOPL_COPY_HEAD%8==0 && sizeof(BoplRlEntity)%8==0 && sizeof(BoplRlSlime)%8==0, "the image's rows stay aligned");
typedef struct {
    BoplRlState *state;      /* the allocation's head; image follows it */
    unsigned char *image; size_t image_capacity;
    int valid, compact;
    uint64_t source_mask; uint8_t source_counts[BOPL_ABILITY_COUNT];
    /* To 128 bytes: the publish's swap of three copies (bopl_step_finish_from)
     * is then whole 64-byte moves, where 80 bytes went as two overlapping ones
     * through a stack temporary that the reload could not forward from. */
    uint8_t pad[128-40-BOPL_ABILITY_COUNT];
} BoplStateCopy;
_Static_assert(sizeof(BoplStateCopy)==128, "the copy is two 64-byte moves");
_Static_assert(BOPL_ABILITY_COUNT<=64, "source_mask holds every source ability");
static void bopl_state_copy(BoplStateCopy *copy, const BoplRlState *state) {
    if (!state) { copy->valid=0; return; }
    uint32_t entities=state->header.entity_count, slimes=state->header.slime_count;
    size_t need=(size_t)entities*sizeof(BoplRlEntity)+(size_t)slimes*(sizeof(BoplRlSlime)+2*sizeof(float));
    if (!copy->state || need>copy->image_capacity) {
        size_t capacity=need<16384?16384:need*2;
        unsigned char *grown=realloc(copy->state, BOPL_COPY_STATE+capacity);
        if (!grown) abort();
        copy->state=(BoplRlState*)grown; copy->image=grown+BOPL_COPY_STATE; copy->image_capacity=capacity;
    }
    *copy->state=*state;
    BoplRlEntity *to=(BoplRlEntity*)copy->image;
    const BoplRlEntity *from=state->entities;
    uint32_t kept=0; int ropes=0; uint64_t mask=0;
    memset(copy->source_counts, 0, sizeof(copy->source_counts));
    for (uint32_t i=0; i<entities; i++) {
        const BoplRlEntity *entity=&from[i];
        ropes|=entity->kind==3;
        if ((entity->flags&(BOPL_ENTITY_ACTIVE|BOPL_ENTITY_DESTROYED))!=BOPL_ENTITY_ACTIVE) continue;
        if (kept<BOPL4_ENTITIES) {
            uint32_t source=entity->reserved[0];
            if (source>0 && source<=BOPL_ABILITY_COUNT) { copy->source_counts[source-1]++; mask|=1ull<<(source-1); }
        }
        to[kept++]=*entity;
    }
    if (ropes) { memcpy(to, from, (size_t)entities*sizeof(*to)); kept=entities; }
    copy->state->header.entity_count=kept;
    copy->compact=!ropes; copy->source_mask=mask;
    BoplRlSlime *slime_rows=(BoplRlSlime*)(to+kept);
    if (slimes) memcpy(slime_rows, state->slimes, (size_t)slimes*sizeof(*slime_rows));
    copy->state->entities=to; copy->state->slimes=slime_rows;
    copy->valid=1;
}
static const BoplRlState *bopl_state_copy_get(const BoplStateCopy *copy) { return copy->valid?copy->state:NULL; }
static void bopl_state_copy_free(BoplStateCopy *copy) { free(copy->state); memset(copy, 0, sizeof(*copy)); }
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
     * eval/check_observations.c). NULL in an environment: its rows are
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
    BoplMapMask training_map_mask;
    int loadout_mode, loadout_no_dup, mirror_share_ppm, loadout_set_count;
    /* Learner-first lineups (train.py --loadout-learner-first): in a bank
     * environment a set row is [learner kit | opponent kit], so an exploiter
     * always plays its own kit against its target's. The match's side is
     * drawn with the config (from side_rng, one draw per match as without
     * the flag) and the row's halves are seated to match; config_side and
     * spare_side are the sides of env->config and env->spare_config. */
    int loadout_learner_first, config_side, spare_side;
    uint64_t loadout_pool;
    uint8_t loadout_set[BOPL_LOADOUT_SET_MAX][6];
    int loadout_weighted;
    uint16_t loadout_weight[64];
    int observation_delay, delay_cursor, no_autoreset;
    /* Observation delay (nine ticks = two decisions at repeat 4): the world
     * states sampled at phase 3 of the last two decisions, encoded into the
     * slots two decisions later. Keeping ~3 KB of state per slot instead of a
     * 13 KB observation per player, and encoding at publish time, replaces
     * the old encode -> sample -> history -> publish copy chain. */
    BoplStateCopy delayed[2], sample_copy;
    /* Whether each held state has been through overflow telemetry: a delayed
     * state is counted when it is encoded, and the samples an episode end
     * leaves unpublished are counted then (bopl_count_unpublished). */
    int delayed_counted[2], sample_counted;
    /* How many player, entity and slime rows the last encode into each
     * observation row left non-zero (bopl_encode); the unused slots behind
     * them were zeroed then and stay zero, so the next encode zeroes only the
     * slots it uncovers. bopl_written_forget drops the record when the row
     * memory may have been written by anything else (a routing change). */
    struct { float *row; int players, slots, entities, slimes; } written[BOPL_MAX_SLOTS];
    int written_count;
    /* Packed observation transport (bopl_obs_pack.h): the trainer registers
     * the shared row buffer, the row range the GPU reads from this env's
     * buffer and the slot buffer; c_step packs each of its rows in that range
     * (bopl_pack_rows). NULL pack_out = dense rows as before. */
    const float *pack_base; float *pack_out; long pack_row_begin, pack_row_end;
    /* Observation layout version of each player's row (3 or 4, bopl_layout_v4.h).
     * Training sets one for every player (obs_layout kwarg); evaluation selects
     * per player from the checkpoint's architecture (research_set_layout). */
    int layout[BOPL_MAX_PLAYERS];
    /* Writer ablation bits (bopl_obs_v4.h BOPL4_ABLATE_*) of each player's v4
     * row, stored as mask+1: 0 (every zeroed environment, training) = the
     * process default BOPL4_ABLATE, else the mask of that player's checkpoint
     * (research_set_ablation, the in-game shim), so checkpoints of different
     * writer families share a game. bopl_ablate_of / bopl_set_ablate_of. */
    int ablate[BOPL_MAX_PLAYERS];
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
    /* Scripted opponents (bopl_bots.h). Training: banks whose
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
     * timeout penalty paid by both players. reward_self_death is an extra
     * penalty (>= 0) for each death the game files as self-inflicted or
     * environmental (own explosives, drowning, falling out), on top of the
     * kill/death shaping: a strong net's losses to the uniform anchor are mostly
     * these (sweeps/random-loss-study, 2026-10-13). 0 pays nothing. */
    float reward_kill, reward_timeout, reward_self_death;
    /* Team play (docs/TEAM_PLAY.md): 0 = every player its own team, the
     * historical 1v1 game; else the config's teams are the game's (2v2 self-play
     * environments: seats 0,1 against 2,3, binding.c team_every) and a kill of a
     * teammate is not paid as a kill but charged as one (bopl_team_kills, read
     * from the game's death events). Set once, before the first reset. */
    int team_play;
    /* Process-lifetime cumulative counters over decisions this environment
     * actually played (warmup runs inside the game's reset and never reaches
     * c_step, so it is excluded by construction). Unfinished episodes are
     * included, unlike Log, whose unfinished-episode contributions PufferLib's
     * static_vec_log drops. Learner-only where noted; frozen opponent rows
     * never count as learner work. Read at the rollout barrier by
     * bopl_vec_counters. */
    BoplCounters counters;
    /* Style bonus (NULL without a program): history slots per player
     * (style->hist_floats each, zeroed at every episode start) and learner-only
     * cumulative per-term payouts (unscaled), decisions where a term paid, and
     * the decisions evaluated. Read at the rollout barrier by bopl_vec_style. */
    const BoplStyleProgram *style;
    float *style_slots;
    double style_pay[BS_MAX_TERMS], style_fires[BS_MAX_TERMS], style_decisions;
    char error[512];
} BoplPuffer;

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
static int bopl_ablate_of(const BoplPuffer *env, int player) { return env->ablate[player]-1; }   /* -1: the process default */
static void bopl_set_ablate_of(BoplPuffer *env, int player, int mask) { env->ablate[player]=mask<0?0:mask+1; }
static int bopl_obs_size_of(const BoplPuffer *env, int player) { return bopl_layout_of(env, player)==4?BOPL4_OBS_SIZE:BOPL_OBS_SIZE; }
static int bopl_previous_offset(const BoplPuffer *env, int player) {
    return bopl_layout_of(env, player)==4?BOPL4_OBS_PREVIOUS_OFFSET:BOPL_OBS_PREVIOUS_OFFSET;
}
/* The policy sees the nearest BOPL_OBS_ENTITIES entities plus the overflow
 * fraction (obs[5]/obs[7]); the count is logged per episode. Crowded Platform
 * rounds (>96 live bodies) are a real, rare game state, not an error: the
 * 2026-09-08 500M run died on one at 55M learner steps. Report once per env,
 * keep training. Counted once per sampled state: when it is encoded, or at
 * the episode end for the samples still waiting in the delay. */
static void bopl_note_overflow(BoplPuffer *env, int active_count, int count,
                               int visible_slimes, int visible_slots) {
    if (active_count>count || visible_slimes>BOPL_OBS_SLIMES || visible_slots>BOPL_OBS_SLOTS) {
        if (env->log.observation_overflow==0)
            fprintf(stderr, "Bopl policy observation overflow: %d entities, %d slimes, %d slots (truncated)\n",
                    active_count, visible_slimes, visible_slots);
        env->log.observation_overflow++;
    }
}
static void bopl_note_entity_rows(BoplPuffer *env, const BoplRlState *state,
                                  const uint32_t *nearest, int count, const int *source_counts) {
    int kept[BOPL_ABILITY_COUNT]={0};
    for (int i=0; i<count; i++) {
        uint32_t source=state->entities[nearest[i]].reserved[0];
        if (source>0 && source<BOPL_ABILITY_COUNT) kept[source]++;
    }
    for (int ability=1; ability<BOPL_ABILITY_COUNT; ability++) {
        int total=source_counts[ability-1];
        env->counters.entity_rows[ability]+=total;
        env->counters.entity_overflow[ability]+=total-kept[ability];
    }
}
/* Every visible-entity and visible-slime count of `state` per observer, for
 * the overflow telemetry of a sampled state that is never encoded. */
static void bopl_count_overflow(BoplPuffer *env, const BoplRlState *state) {
    if (!state) return;
    for (int a=0; a<env->players; a++) {
        const BoplRlPlayer *self=&state->players[a];
        uint32_t nearest[BOPL_OBS_ENTITIES]; double distances[BOPL_OBS_ENTITIES];
        int capacity=bopl_layout_of(env,a)==4?BOPL4_ENTITIES:BOPL_OBS_ENTITIES;
        int count=0, active_count=0, visible_slimes=0, visible_slots=0;
        int source_counts[BOPL_ABILITY_COUNT]={0};
        for (uint32_t i=0; i<state->header.entity_count;) {
            const BoplRlEntity *entity=&state->entities[i];
            if (entity->kind==3) {
                int end=bopl_rope_group_end(state, i), group_count=end-(int)i;
                for (int j=0; j<group_count; j++) if (bopl_rope_span_kept(j, group_count)) {
                    const BoplRlEntity *entity=&state->entities[i+j];
                    if (!bopl_entity_visible(state,entity,self->id)) continue;
                    double distance=bopl_surface_distance(entity,bopl_fix(self->x),bopl_fix(self->y));
                    if (entity->flags&BOPL_ENTITY_PICKUP) distance-=1.0e12;
                    bopl_nearest(nearest,distances,&count,capacity,i+(uint32_t)j,distance);active_count++;
                    if (entity->reserved[0]>0 && entity->reserved[0]<BOPL_ABILITY_COUNT) source_counts[entity->reserved[0]-1]++;
                }
                i=(uint32_t)end;
            } else {
                if (bopl_entity_visible(state, entity, self->id)) {
                    double distance=bopl_surface_distance(entity,bopl_fix(self->x),bopl_fix(self->y));
                    if (entity->flags&BOPL_ENTITY_PICKUP) distance-=1.0e12;
                    bopl_nearest(nearest,distances,&count,capacity,i,distance);active_count++;
                    if (entity->reserved[0]>0 && entity->reserved[0]<BOPL_ABILITY_COUNT) source_counts[entity->reserved[0]-1]++;
                }
                i++;
            }
        }
        for (uint32_t i=0; i<state->header.slime_count; i++) {
            const BoplRlSlime *slime=&state->slimes[i];
            if (bopl_spatial_hidden(state, slime->owner, self->id)) continue;
            visible_slimes++; visible_slots+=bopl_slime_inventory(slime);
        }
        bopl_note_overflow(env, active_count, count, visible_slimes, visible_slots);
        bopl_note_entity_rows(env,state,nearest,count,source_counts);
    }
}
/* The row counts the last encode left in observation row `row`, or NULL when
 * the row's memory is not known to hold a previous encode (then every unused
 * slot is zeroed and the row is recorded). */
static void bopl_written_forget(BoplPuffer *env) { env->written_count=0; }
static void bopl_written_forget_row(BoplPuffer *env, const float *row) {
    for (int i=0; i<env->written_count; i++) if (env->written[i].row==row) { env->written[i]=env->written[--env->written_count]; return; }
}
/* The row was just zeroed whole: record empty prefixes (the next encode
 * zeroes nothing, the packed transport sends the header alone). */
static void bopl_written_zeroed(BoplPuffer *env, float *row) {
    int w=-1;
    for (int i=0; i<env->written_count; i++) if (env->written[i].row==row) { w=i; break; }
    if (w<0) {
        if (env->written_count==BOPL_MAX_SLOTS) env->written[0]=env->written[--env->written_count];
        w=env->written_count++; env->written[w].row=row;
    }
    env->written[w].players=env->written[w].slots=env->written[w].entities=env->written[w].slimes=0;
}
/* The GPU writes the v4 rows a packed transport ships (bopl_obs_pack.h).
 * While bopl_step_finish writes an env's rows (bopl4_defer), a row whose
 * state fits its slot is not encoded here at all: bopl4_encode_state ships
 * the state (pack mode 4) and the GPU writes every feature from it
 * (bopl_rays_v4.cu, from the source both writers share, bopl_rows_v4.h);
 * the host keeps the own-action and critic blocks and the overflow
 * telemetry (bopl4_state_rows). A row whose state does not fit is encoded
 * here without its terrain block and rays: the writer keeps the row's scene
 * (bopl4_pending) and bopl_pack_rows ships it (mode 2, or 3 with the dense
 * row). Rows outside the pack range are written whole on the CPU. Per
 * thread: an env's rows are written and packed on the thread that steps it.
 * BOPL_RAYS_CHECK=1 (train.py --dump-rows) writes every row on the CPU as
 * well, so the transport check compares the GPU's. */
static __thread int bopl4_defer, bopl4_pending_count, bopl4_state_count;
static __thread struct { const float *row; Bopl4Scene scene; } bopl4_pending[BOPL_MAX_PLAYERS];
/* The rows shipped as state this step: the row, its slot, and the critic
 * block's undelayed enemy offsets (bopl4_privileged; enemy 0 = none). */
static __thread struct { const float *row; float *slot; int enemy; double enemy_dx, enemy_dy; } bopl4_state_rows[BOPL_MAX_PLAYERS];
static int bopl4_rays_check=-1;
static int bopl4_check_rows(void) {
    if (bopl4_rays_check<0) { const char *v=getenv("BOPL_RAYS_CHECK"); bopl4_rays_check=v&&v[0]=='1'; }
    return bopl4_rays_check;
}
static Bopl4Scene *bopl4_pending_scene(const float *row) {
    for (int i=0; i<bopl4_pending_count; i++) if (bopl4_pending[i].row==row) return &bopl4_pending[i].scene;
    if (bopl4_pending_count==BOPL_MAX_PLAYERS) return NULL;
    bopl4_pending[bopl4_pending_count].row=row;
    return &bopl4_pending[bopl4_pending_count++].scene;
}
static int bopl4_state_row(const float *row) {
    for (int i=0; i<bopl4_state_count; i++) if (bopl4_state_rows[i].row==row) return i;
    return -1;
}
/* The packed slot of observation row `obs`, or NULL outside the env's live range. */
static float *bopl4_pack_slot(const BoplPuffer *env, const float *obs) {
    if (!env->pack_out || (uintptr_t)obs<(uintptr_t)env->pack_base) return NULL;
    long g=(long)(((uintptr_t)obs-(uintptr_t)env->pack_base)/(BOPL_OBS_STRIDE*sizeof(float)));
    return g<env->pack_row_begin || g>=env->pack_row_end ? NULL : env->pack_out+g*BOPL4_PACK_P;
}
static const Bopl4Dirs *bopl4_dirs(void) {
    static __thread Bopl4Dirs dirs;
    static __thread int ready=0;   /* per thread: no synchronisation */
    if (!ready) { bopl4f_dirs_init(&dirs); ready=1; }
    return &dirs;
}
/* Pack this env's v4 rows in the trainer's live range (bopl_obs_pack.h): a
 * row with a scene goes in mode 2 when it fits, else mode 3 (dense + scene);
 * one without (the writer cast its rays) in mode 0, or dense (mode 1) without
 * a writer record, as a v3 row, or when it does not fit. Rows outside the
 * range get their terrain and rays written here. */
static void bopl_pack_rows(BoplPuffer *env) {
    int check=bopl4_check_rows();
    for (int a=0; a<env->players; a++) {
        float *obs=bopl_obs_row(env, a);
        int state_row=bopl4_state_row(obs);
        if (state_row>=0) {
            /* Mode 4: the head and state went in at the encode; now the blocks the host wrote since. */
            float *slot=bopl4_state_rows[state_row].slot;
            Bopl4StateHead *head=(Bopl4StateHead*)slot;
            memcpy(slot+BOPL4_STATE_FIXED_OFFSET, obs+BOPL4_OBS_PREVIOUS_OFFSET, BOPL4_PREVIOUS*sizeof(float));
            /* The critic block went straight to the slot (bopl4_privileged) unless the host row is checked. */
            if (check) memcpy(slot+BOPL4_STATE_FIXED_OFFSET+BOPL4_PREVIOUS, obs+BOPL4_OBS_PRIVILEGED_OFFSET, BOPL4_PRIVILEGED*sizeof(float));
            if (bopl4_state_rows[state_row].enemy) {
                head->flags|=BOPL4_STATE_ENEMY;
                head->enemy_dx=bopl4_state_rows[state_row].enemy_dx; head->enemy_dy=bopl4_state_rows[state_row].enemy_dy;
            }
            continue;
        }
        Bopl4Scene *scene=NULL;
        for (int i=0; i<bopl4_pending_count; i++) if (bopl4_pending[i].row==obs) { scene=&bopl4_pending[i].scene; break; }
        float *slot=bopl4_pack_slot(env, obs);
        if (!slot) {
            if (scene) bopl4f_write(scene, bopl4_dirs(), obs+BOPL4_OBS_TERRAIN_OFFSET);
            continue;
        }
        int counts[4]={-1, -1, -1, -1};
        if (bopl_layout_of(env, a)==4)
            for (int i=0; i<env->written_count; i++) if (env->written[i].row==obs) {
                counts[0]=env->written[i].players; counts[1]=env->written[i].slots;
                counts[2]=env->written[i].entities; counts[3]=env->written[i].slimes; break;
            }
        if (scene) {
            if (check) bopl4f_write(scene, bopl4_dirs(), obs+BOPL4_OBS_TERRAIN_OFFSET);
            if (!bopl_pack_row_scene(obs, counts, scene, slot)) bopl_pack_row_dense_scene(scene, slot);
            continue;
        }
        bopl_pack_row(obs, counts, slot);
    }
    bopl4_pending_count=0; bopl4_state_count=0;
}
/* Zero slots [count, previous) of a block of `features`-float slots: with a
 * record, the slots from the previous encode's count on are already zero. */
static void bopl_zero_tail(float *block, int features, int count, int previous, int capacity) {
    int end=previous<0?capacity:previous;
    if (count<end) memset(block+(size_t)count*features, 0, (size_t)(end-count)*features*sizeof(float));
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
        if (count_overflow) {
            bopl_note_overflow(env, active_count, count, visible_slimes, visible_slots);
            bopl_note_entity_rows(env,state,nearest,count,source_counts);
        }

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
/* The overflow telemetry of player a's v4 row from `state` (what
 * bopl_encode_v4_row counts) without the row: the visible entities after rope
 * thinning, the visible slimes and their slots. Only for states of at most
 * BOPL4_ENTITIES entities, so every visible entity is kept. */
typedef struct { int count, visible_slimes, visible_slots, source_counts[BOPL_ABILITY_COUNT]; uint64_t source_mask; } Bopl4Census;
static void bopl4_census(const BoplRlState *state, int a, Bopl4Census *c) {
    const BoplRlPlayer *self=&state->players[a];
    memset(c, 0, sizeof(*c));
    for (uint32_t i=0; i<state->header.entity_count;) {
        int end=(state->entities[i].kind==3)?bopl_rope_group_end(state, i):(int)i+1;
        int group=end-(int)i;
        for (int j=0; j<group; j++) {
            const BoplRlEntity *entity=&state->entities[i+j];
            if (entity->kind==3 && !bopl_rope_span_kept(j, group)) continue;
            if (!bopl_entity_visible(state, entity, self->id)) continue;
            c->count++;
            uint32_t source=entity->reserved[0];
            if (source>0 && source<=BOPL_ABILITY_COUNT) { c->source_counts[source-1]++; c->source_mask|=1ull<<(source-1); }
        }
        i=(uint32_t)end;
    }
    /* A hidden slime's slots still count when it is the primary enemy's
     * body (the observer's own is never hidden). */
    int enemy_known=0;
    const BoplRlSlime *enemy_controller=NULL;
    for (uint32_t i=0; i<state->header.slime_count; i++) {
        const BoplRlSlime *slime=&state->slimes[i];
        int hidden=bopl_spatial_hidden(state, slime->owner, self->id);
        if (hidden) {
            if (!enemy_known) {
                double self_x=bopl_fix(self->x), self_y=bopl_fix(self->y);
                Bopl4Enemy enemy=bopl4_primary_enemy(state, self, self_x, self_y);
                enemy_controller=enemy.player?bopl_controlling_slime(state, enemy.player->id, enemy.x, enemy.y):NULL;
                enemy_known=1;
            }
            if (slime!=enemy_controller) continue;
        } else c->visible_slimes++;
        for (int slot=0; slot<3; slot++) {
            uint32_t id=slime->abilities[slot], flags=slime->ability_flags[slot];
            if ((flags&1) && id!=0 && id<=BOPL_ABILITY_COUNT) c->visible_slots++;
        }
    }
}
/* bopl_note_overflow and bopl_note_entity_rows of a census: every visible
 * entity is kept, so no ability's entities overflow (entity_overflow gains
 * 0) and entity_rows gains each ability's count, the integer sums exact in
 * any order (the zero additions left out: the counters are never -0). */
static void bopl4_census_note(BoplPuffer *env, const Bopl4Census *c) {
    bopl_note_overflow(env, c->count, c->count, c->visible_slimes, c->visible_slots);
    for (uint64_t mask=c->source_mask; mask; mask&=mask-1) {
        int ability=1+__builtin_ctzll(mask);   /* source_mask: the abilities with a nonzero count */
        if (ability<BOPL_ABILITY_COUNT) env->counters.entity_rows[ability]+=c->source_counts[ability-1];
    }
}
/* bopl4_census_note for a row of a state that ships more than
 * BOPL4_ENTITIES entities: the census, and when more than BOPL4_ENTITIES of
 * them are visible, the nearest BOPL4_ENTITIES as bopl_encode_v4_row keeps
 * them (its pass: rope thinning, visibility, the surface distance, a pickup's
 * offset, bopl_nearest), whose sources say what each ability overflows. */
static void bopl4_crowded_note(BoplPuffer *env, const BoplRlState *state, int a) {
    Bopl4Census c; bopl4_census(state, a, &c);
    if (c.count<=BOPL4_ENTITIES) { bopl4_census_note(env, &c); return; }
    const BoplRlPlayer *self=&state->players[a];
    double self_x=bopl_fix(self->x), self_y=bopl_fix(self->y);
    uint32_t nearest[BOPL4_ENTITIES]; double distances[BOPL4_ENTITIES];
    int count=0;
    for (uint32_t i=0; i<state->header.entity_count;) {
        int end=(state->entities[i].kind==3)?bopl_rope_group_end(state, i):(int)i+1;
        int group=end-(int)i;
        for (int j=0; j<group; j++) {
            const BoplRlEntity *entity=&state->entities[i+j];
            if (entity->kind==3 && !bopl_rope_span_kept(j, group)) continue;
            if (!bopl_entity_visible(state, entity, self->id)) continue;
            double distance=bopl_surface_distance(entity, self_x, self_y);
            if (entity->flags&BOPL_ENTITY_PICKUP) distance-=1.0e12;
            bopl_nearest(nearest, distances, &count, BOPL4_ENTITIES, i+(uint32_t)j, distance);
        }
        i=(uint32_t)end;
    }
    bopl_note_overflow(env, c.count, count, c.visible_slimes, c.visible_slots);
    bopl_note_entity_rows(env, state, nearest, count, c.source_counts);
}
_Static_assert(BOPL4_STATE_OFFSET+BOPL4_STATE_BASE_WORDS+BOPL4_STATE_ENTITIES*BOPL4_STATE_ENTITY_WORDS+2*BOPL4_STATE_SLIME_WORDS
               +(BOPL4_STATE_ENTITIES+3)/4<=BOPL4_PACK_P, "a state of BOPL4_STATE_ENTITIES entities and 2 slimes fits a slot");
/* What the rows of one encode share when they ship their state, found by
 * the first row that tries (state NULL until then):
 * - The entities shipped (runs of entity indices, in order): every one but
 *   the dead (not active, or destroyed). A dead entity is in no row: never
 *   visible (bopl_entity_visible), so neither kept, counted nor among the
 *   nearest, and no wall or water (bopl4_scene_class), so the GPU writes the
 *   same row without it, and a state crowded with dead bodies fits a slot.
 *   Only rope pieces see their neighbours (a rope group is a run of them,
 *   bopl4_entity_kept), so a state with a rope ships every entity.
 * - The slimes' sinf and cosf.
 * - The overflow census when no player is invisible (flags bit 4: then
 *   bopl_spatial_hidden is 0 for every pair, so every observer sees the
 *   same entities and slimes and the census is the state's). Without a
 *   rope as well, the census is the shipped scan's: every live entity is
 *   visible and kept, every slime visible. */
typedef struct {
    const BoplRlState *state;
    const BoplStateCopy *copy;   /* the state's copy, when it is one (bopl_encode_copy) */
    const unsigned char *image; size_t image_bytes;   /* its image with the trig (fits), else NULL */
    /* With image: the shipped entities' bopl_fix(x), bopl_fix(y) and whether
     * they move (inverse_mass > 0), for every row's sectors. */
    double entity_x[BOPL4_STATE_ENTITIES], entity_y[BOPL4_STATE_ENTITIES]; uint8_t movable[BOPL4_STATE_ENTITIES];
    int fits, shared_census, census_ready;
    uint32_t shipped, runs, run_begin[BOPL4_STATE_ENTITIES], run_count[BOPL4_STATE_ENTITIES];
    Bopl4Census census;
    float trig[2*BOPL4_SLIMES];
} Bopl4Shared;
static void bopl4_shared_init(Bopl4Shared *shared, const BoplRlState *state) {
    shared->image=NULL;
    uint32_t entities=state->header.entity_count, slimes=state->header.slime_count;
    shared->state=state; shared->census_ready=0; shared->shared_census=1;
    for (uint32_t i=0; i<state->header.player_count; i++) if (state->players[i].flags&4) shared->shared_census=0;
    uint32_t shipped=0, runs=0, ropes=0;
    Bopl4Census *c=&shared->census;
    if (shared->shared_census) { memset(c->source_counts, 0, sizeof(c->source_counts)); c->source_mask=0; }
    for (uint32_t i=0; i<entities; i++) {
        const BoplRlEntity *entity=&state->entities[i];
        ropes|=entity->kind==3;
        if ((entity->flags&(BOPL_ENTITY_ACTIVE|BOPL_ENTITY_DESTROYED))!=BOPL_ENTITY_ACTIVE || shipped>BOPL4_STATE_ENTITIES) continue;
        if (shipped<BOPL4_STATE_ENTITIES) {
            uint32_t source=entity->reserved[0];
            if (shared->shared_census && source>0 && source<=BOPL_ABILITY_COUNT) { c->source_counts[source-1]++; c->source_mask|=1ull<<(source-1); }
            if (runs && shared->run_begin[runs-1]+shared->run_count[runs-1]==i) shared->run_count[runs-1]++;
            else { shared->run_begin[runs]=i; shared->run_count[runs]=1; runs++; }
        }
        shipped++;
    }
    if (ropes) { shipped=entities; runs=1; shared->run_begin[0]=0; shared->run_count[0]=entities; }
    shared->shipped=shipped; shared->runs=runs;
    shared->fits=shipped<=BOPL4_STATE_ENTITIES && slimes<=BOPL4_SLIMES && bopl4_state_words(shipped, slimes)<=BOPL4_PACK_P;
    if (!shared->fits) return;
    for (uint32_t i=0; i<slimes; i++) bopl4_sincos(&state->slimes[i], &shared->trig[2*i], &shared->trig[2*i+1]);
    if (shared->shared_census && !ropes) {
        c->count=(int)shipped; c->visible_slimes=(int)slimes; c->visible_slots=0;
        for (uint32_t i=0; i<slimes; i++) c->visible_slots+=bopl_slime_inventory(&state->slimes[i]);
        shared->census_ready=1;
    }
}
/* bopl4_shared_init of a copy (bopl_state_copy), from what the copy noted:
 * its entities are what ships (the live ones, or every one with a rope), one
 * run, and without a rope their census is the copy's. */
static void bopl4_shared_from_copy(Bopl4Shared *shared, const BoplStateCopy *copy) {
    shared->image=NULL;
    const BoplRlState *state=copy->state;
    uint32_t entities=state->header.entity_count, slimes=state->header.slime_count;
    shared->state=state; shared->census_ready=0; shared->shared_census=1;
    for (uint32_t i=0; i<state->header.player_count; i++) if (state->players[i].flags&4) shared->shared_census=0;
    shared->shipped=entities; shared->runs=1; shared->run_begin[0]=0; shared->run_count[0]=entities;
    shared->fits=entities<=BOPL4_STATE_ENTITIES && slimes<=BOPL4_SLIMES && bopl4_state_words(entities, slimes)<=BOPL4_PACK_P;
    if (!shared->fits) return;
    float *trig=(float*)(copy->image+(size_t)entities*sizeof(BoplRlEntity)+(size_t)slimes*sizeof(BoplRlSlime));
    for (uint32_t i=0; i<slimes; i++) bopl4_sincos(&state->slimes[i], &trig[2*i], &trig[2*i+1]);
    shared->image=copy->image; shared->image_bytes=(size_t)((unsigned char*)(trig+2*slimes)-copy->image);
    for (uint32_t i=0; i<entities; i++) {
        const BoplRlEntity *entity=&state->entities[i];
        shared->entity_x[i]=bopl_fix(entity->x); shared->entity_y[i]=bopl_fix(entity->y); shared->movable[i]=entity->inverse_mass>0;
    }
    if (shared->shared_census && copy->compact) {
        Bopl4Census *c=&shared->census;
        c->count=(int)entities; c->visible_slimes=(int)slimes; c->visible_slots=0;
        for (uint32_t i=0; i<slimes; i++) c->visible_slots+=bopl_slime_inventory(&state->slimes[i]);
        c->source_mask=copy->source_mask;
        for (uint64_t mask=copy->source_mask; mask; mask&=mask-1) { int k=__builtin_ctzll(mask); c->source_counts[k]=copy->source_counts[k]; }
        shared->census_ready=1;
    }
}
static void bopl4_count_row(BoplPuffer *env, const BoplRlState *state, int a, Bopl4Shared *shared) {
    if (shared->shipped>BOPL4_ENTITIES) { bopl4_crowded_note(env, state, a); return; }
    if (!shared->shared_census) { Bopl4Census c; bopl4_census(state, a, &c); bopl4_census_note(env, &c); return; }
    if (!shared->census_ready) { bopl4_census(state, a, &shared->census); shared->census_ready=1; }
    bopl4_census_note(env, &shared->census);
}
/* Ship player a's v4 row as its state (pack mode 4, bopl_rows_v4.h
 * Bopl4StateHead) when bopl_step_finish writes the env's rows, the row is in
 * the pack range and the state fits the slot: the GPU writes the row. Here:
 * the overflow telemetry, the slot's head and state, and the row's
 * own-action block cleared for the tail. Returns 0 to encode on the CPU. */
static int bopl4_encode_state(BoplPuffer *env, const BoplRlState *state, int a, int count_overflow, Bopl4Shared *shared) {
    if (!bopl4_defer || bopl4_state_count==BOPL_MAX_PLAYERS) return 0;
    float *obs=bopl_obs_row(env, a), *slot=bopl4_pack_slot(env, obs);
    if (!slot) return 0;
    if (shared->state!=state) { if (shared->copy) bopl4_shared_from_copy(shared, shared->copy); else bopl4_shared_init(shared, state); }
    if (!shared->fits) return 0;
    uint32_t entities=shared->shipped, slimes=state->header.slime_count;
    if (bopl4_check_rows()) {
        /* The CPU's row as well, for the transport check (rays included). */
        bopl4_defer=0; bopl_encode_v4_row(env, state, a, count_overflow); bopl4_defer=1;
    } else if (count_overflow) bopl4_count_row(env, state, a, shared);
    bopl_written_forget_row(env, obs);   /* the host row is not written */
    memset(obs+BOPL4_OBS_PREVIOUS_OFFSET, 0, BOPL4_PREVIOUS*sizeof(float));
    int ablation=bopl_ablate_of(env, a);
    Bopl4StateHead *head=(Bopl4StateHead*)slot;
    memset(head, 0, sizeof(*head));
    head->mode=BOPL4_PACK_MODE_STATE; head->player=(uint32_t)a;
    head->ablation=(uint32_t)(ablation>=0?ablation:bopl4_ablation_default());
    head->flags=bopl_map_selected(BOPL_SPACE_MAPS, env->config.map)?BOPL4_STATE_SPACE:0;
    if (env->applied[a].angle) {
        pthread_once(&bopl_aim_table_once, bopl_aim_table_init);
        head->flags|=BOPL4_STATE_AIMED;
        head->aim_dx=bopl_aim_table[env->applied[a].angle][0]; head->aim_dy=bopl_aim_table[env->applied[a].angle][1];
    }
    head->max_episode_ticks=env->config.max_episode_ticks;
    head->words=(uint32_t)bopl4_state_words(entities, slimes); head->entity_count=entities; head->slime_count=slimes;
    char *out=(char*)(slot+BOPL4_STATE_OFFSET);
    const BoplRlEntity *shipped=(const BoplRlEntity*)(out+BOPL_COPY_HEAD);
    unsigned char *sectors;
    if (shared->image) {   /* a copy: the header (its entity_count the kept count) and players, then the image */
#if defined(__has_builtin) && __has_builtin(__builtin_memcpy_inline)
        __builtin_memcpy_inline(out, &state->header, BOPL_COPY_HEAD);
#else
        memcpy(out, &state->header, BOPL_COPY_HEAD);
#endif
        memcpy(out+BOPL_COPY_HEAD, shared->image, shared->image_bytes);
        sectors=(unsigned char*)out+BOPL_COPY_HEAD+shared->image_bytes;
    } else {
        BoplRlHeader *header=(BoplRlHeader*)out;
        memcpy(out, &state->header, sizeof(state->header)); out+=sizeof(state->header);
        header->entity_count=entities;   /* the entities shipped (bopl4_entity_kept reads the count) */
        memcpy(out, state->players, sizeof(state->players)); out+=sizeof(state->players);
        for (uint32_t r=0; r<shared->runs; r++) {
            memcpy(out, &state->entities[shared->run_begin[r]], (size_t)shared->run_count[r]*sizeof(BoplRlEntity));
            out+=(size_t)shared->run_count[r]*sizeof(BoplRlEntity);
        }
        memcpy(out, state->slimes, (size_t)slimes*sizeof(BoplRlSlime)); out+=(size_t)slimes*sizeof(BoplRlSlime);
        float *trig=(float*)out;
        memcpy(trig, shared->trig, (size_t)slimes*2*sizeof(float));
        sectors=(unsigned char*)(trig+2*slimes);
    }
    double self_x=bopl_fix(state->players[a].x), self_y=bopl_fix(state->players[a].y);
    /* One run (a copy): the sectors read the source, not the slot just stored. */
    const BoplRlEntity *source=shared->runs==1?&state->entities[shared->run_begin[0]]:shipped;
    if (shared->image) {
        /* bopl4_sector without its branches, over the positions found once
         * for every row: off the octant edges, its sign/steep case as a
         * select; an entity near an edge gets 8 here and bopl4_sector after. */
        unsigned char edges=0;
#ifdef __clang__
#pragma clang loop interleave_count(1)   /* ~18 entities: an 8-wide body, not 32 */
#endif
        for (uint32_t i=0; i<entities; i++) {
            double dx=shared->entity_x[i]-self_x, dy=shared->entity_y[i]-self_y;
            double ax=fabs(dx), ay=fabs(dy), big=fmax(ax, ay);
            int edge=(fmin(ax, ay)<=1e-6*big) | (fabs(ax-ay)<=1e-6*big);
            int steep=ay>ax;
            int sector=dy<0 ? (dx<0?steep:3-steep) : (dx>0?4+steep:7-steep);
            unsigned char value=shared->movable[i]?(unsigned char)(edge?8:sector):0;
            sectors[i]=value; edges|=value==8;
        }
        if (edges) for (uint32_t i=0; i<entities; i++)
            if (sectors[i]==8) sectors[i]=(unsigned char)bopl4_sector(shared->entity_x[i]-self_x, shared->entity_y[i]-self_y);
    } else for (uint32_t i=0; i<entities; i++) {
        const BoplRlEntity *entity=&source[i];
        sectors[i]=entity->inverse_mass>0?(unsigned char)bopl4_sector(bopl_fix(entity->x)-self_x, bopl_fix(entity->y)-self_y):0;
    }
    int k=bopl4_state_count++;
    bopl4_state_rows[k].row=obs; bopl4_state_rows[k].slot=slot; bopl4_state_rows[k].enemy=0;
    return 1;
}
static void bopl_encode_shared(BoplPuffer *env, const BoplRlState *state, int count_overflow, Bopl4Shared *shared) {
    bopl4_relations_open(state);   /* the CPU rows of this state share their entity relations */
    for (int a=0; a<env->players; a++) {
        if (!state) {
            float *obs=bopl_obs_row(env, a);
            memset(obs, 0, bopl_obs_size_of(env, a)*sizeof(float)); bopl_written_zeroed(env, obs);
        } else if (bopl_layout_of(env, a)==4) {
            if (!bopl4_encode_state(env, state, a, count_overflow, shared)) bopl_encode_v4_row(env, state, a, count_overflow);
        } else bopl_encode_v3_row(env, state, a, count_overflow);
    }
    bopl4_relations_close();
}
static void bopl_encode(BoplPuffer *env, const BoplRlState *state, int count_overflow) {
    Bopl4Shared shared; shared.state=NULL; shared.copy=NULL; shared.image=NULL;
    bopl_encode_shared(env, state, count_overflow, &shared);
}
/* bopl_encode of a copy's state (none: an invalid copy), whose shipped part
 * comes from the copy (bopl4_shared_from_copy). */
static void bopl_encode_copy(BoplPuffer *env, const BoplStateCopy *copy, int count_overflow) {
    Bopl4Shared shared; shared.state=NULL; shared.copy=copy; shared.image=NULL;
    bopl_encode_shared(env, bopl_state_copy_get(copy), count_overflow, &shared);
}
/* The critic-only block of every v4 row from the undelayed state `now`
 * (bopl4_privileged); `actions` may be NULL (episode start). */
/* Every player's controlling slime in `now` (NULL past its players). */
static void bopl_controlling_bodies(const BoplRlState *now, const BoplRlSlime **bodies) {
    for (int i=0; i<BOPL_MAX_PLAYERS; i++) bodies[i]=NULL;
    if (now) for (uint32_t i=0; i<now->header.player_count && i<BOPL_MAX_PLAYERS; i++)
        bodies[i]=bopl_controlling_slime(now, now->players[i].id, bopl_fix(now->players[i].x), bopl_fix(now->players[i].y));
}
static void bopl_encode_privileged_bodies(BoplPuffer *env, const BoplRlState *now, const BoplRlAction *actions,
                                          const BoplRlSlime *const *bodies) {
    for (int a=0; a<env->players; a++) {
        if (bopl_layout_of(env, a)!=4) continue;
        int enemy=-1;
        if (now) for (int i=0; i<env->players && enemy<0; i++)
            if (i!=a && now->players[i].team!=now->players[a].team) enemy=i;
        bopl4_privileged(env, now, a, bodies, enemy>=0 && actions?&actions[enemy]:NULL,
                         enemy>=0?&env->hold_left[enemy]:NULL, enemy>=0?&env->held[enemy]:NULL);
    }
}
static void bopl_encode_privileged(BoplPuffer *env, const BoplRlState *now, const BoplRlAction *actions) {
    const BoplRlSlime *bodies[BOPL_MAX_PLAYERS];
    bopl_controlling_bodies(now, bodies);
    bopl_encode_privileged_bodies(env, now, actions, bodies);
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
static uint16_t bopl_bow_buttons_of(const BoplRlSlime *controller) {
    if (!controller) return 0;
    uint16_t buttons=0;
    for (int slot=0; slot<3; slot++)
        if ((controller->ability_flags[slot]&1) && controller->abilities[slot]==4) buttons|=(uint16_t)(1u<<(slot+1));
    return buttons;
}
static uint16_t bopl_bow_buttons(const BoplPuffer *env, int player) {
    const BoplRlState *state=bopl_rl_state(env->world);
    if (!state || player<0 || player>=(int)state->header.player_count) return 0;
    const BoplRlPlayer *self=&state->players[player];
    return bopl_bow_buttons_of(bopl_controlling_slime(state, self->id, bopl_fix(self->x), bopl_fix(self->y)));
}
/* Publish the action mask for player i's next decision from its latch state.
 * bodies: every player's controlling slime in bopl_rl_state(env->world)
 * (bopl_controlling_bodies), or NULL to look the player's up there. */
static void bopl_write_mask_bodies(BoplPuffer *env, int player, const BoplRlSlime *const *bodies) {
    unsigned char *mask=env->mask_ptr[bopl_slot(env, player)];
    if (!mask) return;
    memset(mask, 1, BOPL_LOGITS);
    if (env->hold_left[player]>0) {
        memset(mask+BOPL_ACT_AIM, 0, BOPL_ACT_BUTTONS+BOPL_ACT_HOLD);
        mask[BOPL_ACT_AIM+env->held[player]]=1;
        mask[BOPL_ACT_AIM+BOPL_ACT_BUTTONS]=1;
    } else {
        uint16_t bows=bodies?(player>=0 && player<BOPL_MAX_PLAYERS?bopl_bow_buttons_of(bodies[player]):0)
                            :bopl_bow_buttons(env, player), latched=0;
        for (int slot=0; slot<3; slot++) if ((bows&(1u<<(slot+1))) && env->hold_run[player][slot+1]>0)
            latched|=(uint16_t)(1u<<(slot+1));
        if (latched)
            for (int b=0; b<BOPL_ACT_BUTTONS; b++)
                if (b&latched) mask[BOPL_ACT_AIM+b]=0;
    }
}
static void bopl_write_mask(BoplPuffer *env, int player) { bopl_write_mask_bodies(env, player, NULL); }
static uint32_t bopl_side_random(BoplPuffer *env) {
    uint32_t x=env->side_rng; x^=x<<13; x^=x>>17; x^=x<<5; return env->side_rng=x;
}
static void bopl_failed_world(BoplPuffer *env, const BoplRl *world, const char *operation) {
    env->failed=1; env->log.game_errors++;
    snprintf(env->error, sizeof(env->error), "%s: %s", operation, bopl_rl_error(world));
    fprintf(stderr, "Bopl RL stopped: %s\n", env->error);
    BOPL_RECOVER();
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
        if (bopl_map_selected(env->training_map_mask, map) && map>=bopl_rl_map_count(env->world)) {
            fprintf(stderr, "Training map outside game catalog\n"); abort();
        }
    if (bopl_rl_ability_count(env->world)!=BOPL_ABILITY_COUNT) { fprintf(stderr, "Bopl RL policy catalog changed\n"); abort(); }
}
/* The world-open threads. macOS gives a secondary thread 512 KB of stack,
 * less than the in-process Rust world's constructor needs (RolloutWorld::new
 * overflows it); Linux's default (8 MB) is kept. */
#ifdef __APPLE__
static pthread_attr_t bopl_open_attr_value;
static void bopl_open_attr_init(void) { pthread_attr_init(&bopl_open_attr_value); pthread_attr_setstacksize(&bopl_open_attr_value, (size_t)16<<20); }
static const pthread_attr_t *bopl_open_attr(void) {
    static pthread_once_t once=PTHREAD_ONCE_INIT;
    pthread_once(&once, bopl_open_attr_init);
    return &bopl_open_attr_value;
}
#else
static const pthread_attr_t *bopl_open_attr(void) { return NULL; }
#endif
static void init(BoplPuffer *env) {
    const char *game=getenv("BOPL_RL_GAME"), *replay=getenv("BOPL_RL_BOOTSTRAP"), *logs=getenv("BOPL_RL_LOG_DIR");
    if (!game || !replay || !logs) { fprintf(stderr, "Set BOPL_RL_GAME, BOPL_RL_BOOTSTRAP and BOPL_RL_LOG_DIR to absolute paths\n"); abort(); }
    int length=snprintf(env->world_log, sizeof(env->world_log), "%s/worker-%ld-%u.log", logs, (long)getpid(), env->rng);
    if (length<0 || (size_t)length>=sizeof(env->world_log)) abort();
    if (bopl_maps_selected(env->training_map_mask) && env->random_maps) {
        fprintf(stderr, "Choose explicit training maps or all random maps\n"); abort();
    }
    if (pthread_create(&env->world_thread, bopl_open_attr(), bopl_world_open, env)) abort();
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
    if (!isfinite(env->reward_kill) || !isfinite(env->reward_timeout) || env->reward_timeout>0
        || !isfinite(env->reward_self_death) || env->reward_self_death<0) { fprintf(stderr, "Bopl RL reward config\n"); abort(); }
    if (env->observation_delay!=0 && (env->observation_delay!=9 || env->repeat!=4)) abort();
    /* The spare is a second runtime (~240 MB) for an in-process game, the
     * same memory as its second process for a socket game; BOPL_RL_INPROC_SPARE=0
     * drops it and pays every reset (~8 ms) on the lockstep instead. */
    const char *inproc_spare=getenv("BOPL_RL_INPROC_SPARE");
    if (!env->no_autoreset && !(bopl_rl_in_process(game) && inproc_spare && inproc_spare[0]=='0')) {
        length=snprintf(env->spare_log, sizeof(env->spare_log), "%s/worker-%ld-%u-spare.log", logs, (long)getpid(), env->rng);
        if (length<0 || (size_t)length>=sizeof(env->spare_log)) abort();
        /* Game start-up takes seconds; overlap it with the other workers' start-up. */
        if (pthread_create(&env->spare_thread, bopl_open_attr(), bopl_spare_open, env)) abort();
        env->spare_opening=1;
    }
    env->rng=env->rng*747796405u+2891336453u;
    if (bopl_style_loaded) {
        env->style=&bopl_style_program;
        env->style_slots=calloc((size_t)env->players*env->style->hist_floats, sizeof(float));
        if (!env->style_slots) abort();
    }
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
    if (bopl_maps_selected(env->training_map_mask)) {
        uint32_t count=0;
        for (uint32_t map=0; map<BOPL_COUNTER_MAPS; map++) count+=(uint32_t)bopl_map_selected(env->training_map_mask, map);
        uint32_t selected=bopl_random(env)%count;
        for (uint32_t map=0; map<BOPL_COUNTER_MAPS; map++) if (bopl_map_selected(env->training_map_mask, map)) {
            if (selected==0) { config->map=map; break; }
            selected--;
        }
    } else if (env->random_maps) config->map=bopl_random(env)%bopl_rl_map_count(env->world);
    int mode=env->loadout_mode;
    uint64_t pool=env->loadout_pool;
    if (env->random_loadouts && mode==BOPL_LOADOUT_FIXED) {
        mode=BOPL_LOADOUT_POOL;
        for (uint32_t ability=1; ability<BOPL_ABILITY_COUNT; ability++) pool|=1ull<<ability;
    }
    if (mode!=BOPL_LOADOUT_FIXED) {
        int mirror=env->mirror_share_ppm>=1000000;
        if (!mirror && env->mirror_share_ppm>0)
            mirror=(int)(bopl_random(env)%1000000u)<env->mirror_share_ppm;
        if (mode==BOPL_LOADOUT_SET) {
            const uint8_t *row=env->loadout_set[bopl_random(env)%(uint32_t)env->loadout_set_count];
            for (int player=0; player<env->players; player++) for (uint32_t slot=0; slot<config->slots; slot++)
                config->player[player].abilities[slot]=row[(mirror && player==1?0:player)*3+(int)slot];
            if (env->loadout_learner_first && env->banks && env->players==2) {
                int side=(int)(bopl_side_random(env)&1u);
                if (side) for (uint32_t slot=0; slot<config->slots; slot++) {
                    uint32_t held=config->player[0].abilities[slot];
                    config->player[0].abilities[slot]=config->player[1].abilities[slot];
                    config->player[1].abilities[slot]=held;
                }
                if (config==&env->spare_config) env->spare_side=side; else env->config_side=side;
            }
        } else {
            for (int player=0; player<env->players; player++) {
                if (mirror && player==1) {
                    for (uint32_t slot=0; slot<config->slots; slot++)
                        config->player[player].abilities[slot]=config->player[0].abilities[slot];
                    continue;
                }
                uint64_t available=pool;
                for (uint32_t slot=0; slot<config->slots && env->loadout_weighted; slot++) {
                    uint32_t total=0, chosen=0;
                    for (uint32_t ability=1; ability<BOPL_ABILITY_COUNT; ability++)
                        if (available&(1ull<<ability)) total+=env->loadout_weight[ability];
                    uint32_t pick=bopl_random(env)%total;
                    for (uint32_t ability=1; ability<BOPL_ABILITY_COUNT; ability++) if (available&(1ull<<ability)) {
                        if (pick<env->loadout_weight[ability]) { chosen=ability; break; }
                        pick-=env->loadout_weight[ability];
                    }
                    config->player[player].abilities[slot]=chosen;
                    if (env->loadout_no_dup) available&=~(1ull<<chosen);
                }
                for (uint32_t slot=0; slot<config->slots && !env->loadout_weighted; slot++) {
                    uint32_t count=0;
                    for (uint32_t ability=1; ability<BOPL_ABILITY_COUNT; ability++) count+=(uint32_t)((available>>ability)&1u);
                    uint32_t pick=bopl_random(env)%count, chosen=0;
                    for (uint32_t ability=1; ability<BOPL_ABILITY_COUNT; ability++) if (available&(1ull<<ability)) {
                        if (pick--==0) { chosen=ability; break; }
                    }
                    config->player[player].abilities[slot]=chosen;
                    if (env->loadout_no_dup) available&=~(1ull<<chosen);
                }
            }
        }
    }
}
static int bopl_loadouts_valid(const BoplPuffer *env) {
    if (env->mirror_share_ppm<0 || env->mirror_share_ppm>1000000
        || env->loadout_set_count<0 || env->loadout_set_count>BOPL_LOADOUT_SET_MAX
        || (env->loadout_mode==BOPL_LOADOUT_SET && env->players!=2)) return 0;
    int pool_count=0;
    for (uint32_t ability=1; ability<BOPL_ABILITY_COUNT; ability++) pool_count+=(int)((env->loadout_pool>>ability)&1u);
    if (env->loadout_mode==BOPL_LOADOUT_POOL && (!pool_count || (env->loadout_no_dup && pool_count<(int)env->config.slots))) return 0;
    for (int player=0; player<env->players; player++) for (uint32_t slot=0; slot<env->config.slots; slot++) {
        uint32_t ability=env->config.player[player].abilities[slot];
        if (ability==0 || ability>=BOPL_ABILITY_COUNT) return 0;
        if (env->loadout_no_dup) for (uint32_t prior=0; prior<slot; prior++)
            if (env->config.player[player].abilities[prior]==ability) return 0;
    }
    for (int row=0; row<env->loadout_set_count; row++) for (int player=0; player<2; player++) for (int slot=0; slot<3; slot++) {
        uint8_t ability=env->loadout_set[row][player*3+slot];
        if (ability==0 || ability>=BOPL_ABILITY_COUNT) return 0;
        if (env->loadout_no_dup) for (int prior=0; prior<slot; prior++)
            if (env->loadout_set[row][player*3+prior]==ability) return 0;
    }
    return 1;
}
/* The decision-loop half of c_reset, from the episode's first state: first
 * observation, the delay history primed with it, latches cleared, the neutral
 * own-action tail and the masks. Shared with the agent shim (bopl_agent.c),
 * which is fed states from a running game instead of a world it steps. */
static void bopl_reset_observations(BoplPuffer *env, const BoplRlState *state) {
    memset(env->applied, 0, sizeof(env->applied));   /* before the rows: no aim ray at the episode start */
    compute_observations_from(env, state);
    /* The first two delayed decisions observe this same first state. */
    for (int i=0; i<2; i++) { bopl_state_copy(&env->delayed[i], state); env->delayed_counted[i]=1; }
    env->delay_cursor=0; env->sample_counted=1;
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
/* Reward semantics. The simulation is the game's and stays the game's; what we
 * choose to pay for is ours, and these two derivations are where that choice
 * lives.
 *
 * Self-kills. Player.Kill credits the kill to whoever the killer id resolves
 * to, and for a suicide that is the victim itself (Player.cs:449-455), so a
 * player's own kill count rises when they blow themselves up. RlTelemetry.Death
 * files that same death under deaths_self, and it is the only classification
 * that coincides with a credited kill: an environment or "other" death resolves
 * to no Player and credits nobody. Subtracting deaths_self from kills therefore
 * leaves exactly the kills earned against another player.
 *
 * Undecided rounds. When the last teams are eliminated together the game runs
 * DeclareGameUndecided: it awards WonThisRound to whoever died on the greatest
 * SimulationTicks and expands that to their whole team, so a tie hands the win
 * flag to every side at once. Outlasting the opponent by even one tick is a
 * real win and stays one; sharing the final tick is not, and the flag spanning
 * more than one team is precisely that case. It catches the whole family of
 * simultaneous endings -- a mine trade, a double drowning in sudden death, one
 * player falling off the level as the other eats a mine -- and none of them
 * earns a winner's +1.
 *
 * Telling that family apart is the kill term's job, not the outcome term's:
 * whoever actually killed an opponent on the way out keeps the +0.1 for it and
 * whoever drowned does not. A trade therefore pays a little above a plain loss
 * and far below a clean win, which is the ordering we want. */
static int64_t bopl_opponent_kills(const BoplRlState *state, int player) {
    return (int64_t)state->players[player].kills-(int64_t)state->telemetry[player].deaths_self;
}
static int bopl_undecided(const BoplRlState *state, int players) {
    uint32_t team=0; int seen=0;
    for (int i=0; i<players; i++) if (state->players[i].flags&2) {
        if (!seen) { team=state->players[i].team; seen=1; }
        else if (state->players[i].team!=team) return 1;
    }
    return 0;
}
/* The match ended with a winner: a terminal that is neither a timeout nor an
 * undecided round. */
static int bopl_decided(const BoplRlState *state, int players) {
    return (state->header.flags&BOPL_RL_TERMINAL) && !(state->header.flags&BOPL_RL_TRUNCATED)
        && !bopl_undecided(state, players);
}
/* Deaths the game filed as self-inflicted (own weapon or ability) or
 * environmental (water, falling) since `before`: what reward_self_death
 * charges, on top of reward_kill's charge for every death. */
static int64_t bopl_self_deaths(const BoplRlTelemetry *now, const BoplRlTelemetry *before) {
    return (int64_t)now->deaths_self-before->deaths_self
        +(int64_t)now->deaths_environment-before->deaths_environment;
}
/* Team play: each player's kills of a teammate during the decision, from the
 * game's death events (both phases of a sampled step). Player.Kill credits
 * them like any kill, so bopl_opponent_kills counts them too and the counters
 * alone cannot tell which victim a kill was. Self-kills stay deaths_self's.
 * A backend without events cannot score a team game: it fails the env. */
static int bopl_team_kills(const BoplRl *world, const BoplRlState *state, int players, int64_t *team_kills) {
    for (int i=0; i<players; i++) team_kills[i]=0;
    uint32_t count=0;
    if (!world || !bopl_rl_event_support(world)) return -1;
    const BoplRlEvent *events=bopl_rl_events(world, &count);
    for (uint32_t e=0; e<count && events; e++) {
        const BoplRlEvent *event=&events[e];
        if (event->kind!=BOPL_EVENT_DEATH || !event->actor || event->actor==event->subject) continue;
        int actor=-1, subject=-1;
        for (int i=0; i<players; i++) {
            if (state->players[i].id==event->actor) actor=i;
            if (state->players[i].id==event->subject) subject=i;
        }
        if (actor>=0 && subject>=0 && state->players[actor].team==state->players[subject].team) team_kills[actor]++;
    }
    return 0;
}
/* Everything one decision pays one player. `previous_kills` and
 * `previous_deaths` are that player's counters as of the previous decision;
 * kills are opponent kills, never the game's self-crediting count. */
typedef struct {
    int64_t kills, deaths;   /* what this decision added */
    int terminal, truncated, won;
    float shaping, outcome;  /* kill/death shaping, and the terminal payment */
} BoplOutcome;
static BoplOutcome bopl_outcome(const BoplRlState *state, int i, int decided,
                                int64_t previous_kills, int64_t previous_deaths,
                                float reward_kill, float reward_timeout) {
    BoplOutcome o;
    o.kills=bopl_opponent_kills(state, i)-previous_kills;
    o.deaths=(int64_t)state->players[i].deaths-previous_deaths;
    o.terminal=(state->header.flags&BOPL_RL_TERMINAL)!=0;
    o.truncated=(state->header.flags&BOPL_RL_TRUNCATED)!=0;
    o.won=decided && (state->players[i].flags&2)!=0;
    o.shaping=reward_kill*((float)o.kills-(float)o.deaths);
    /* An undecided round is a loss for everyone in it. */
    o.outcome=o.terminal?(o.truncated?reward_timeout:(o.won?1.0f:-1.0f)):0.0f;
    return o;
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
        env->config=env->spare_config; env->config_side=env->spare_side;
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
    if (env->style_slots) memset(env->style_slots, 0, (size_t)env->players*env->style->hist_floats*sizeof(float));
    memcpy(env->previous_telemetry, state->telemetry, sizeof(env->previous_telemetry));
    for (int i=0; i<env->players; i++) { env->deaths[i]=state->players[i].deaths; env->kills[i]=(uint32_t)bopl_opponent_kills(state, i); }
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
            memset(env->obs_ptr[1+bank], 0, BOPL_OBS_STRIDE*sizeof(float)); bopl_written_zeroed(env, env->obs_ptr[1+bank]);
            if (env->mask_ptr[1+bank]) memset(env->mask_ptr[1+bank], 1, BOPL_LOGITS);
        }
        env->side=env->loadout_learner_first && env->loadout_mode==BOPL_LOADOUT_SET && env->players==2
            ?env->config_side:(int)(bopl_side_random(env)&1u);
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
/* The step's outputs cleared before the sim's step (bopl_step_begin_from;
 * eval/env_ilv_train.c calls it alone, as it builds the phase itself). */
static void bopl_step_clear(BoplPuffer *env) {
    float *telemetry=env->telemetry?env->telemetry:env->last_telemetry;
    memset(telemetry, 0, (size_t)env->players*BOPL_TELEMETRY_SIZE*sizeof(float));
    for (int s=0; s<env->num_agents; s++) { *env->reward_ptr[s]=0; *env->terminal_ptr[s]=0; }
    env->stepped++;
}
static void bopl_step_begin_from(BoplPuffer *env, BoplStepPhase *phase, const BoplRlState *before) {
    bopl_step_clear(env);
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
            fprintf(stderr, "Invalid PufferLib action\n"); BOPL_RECOVER(); abort();
        }
        holds[i]=(int)hold;
        actions[i].angle=(uint8_t)angle;
        actions[i].buttons=bopl_latch(env, i, (uint16_t)buttons, holds[i], &fresh[i]);
        if (!env->banks || i==env->side) {
            const BoplRlPlayer *player=&before->players[i];
            const BoplRlSlime *controller=bopl_controlling_slime(before,player->id,bopl_fix(player->x),bopl_fix(player->y));
            if (controller) for (int slot=0; slot<3; slot++) if (actions[i].buttons&(1u<<(slot+1))) {
                uint32_t ability=controller->abilities[slot];
                if ((controller->ability_flags[slot]&1) && ability>0 && ability<BOPL_ABILITY_COUNT)
                    env->counters.ability_activations[ability]++;
            }
        }
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
    env->sample_counted=0;
}
/* At an episode end: count the samples the delay never publishes. */
static void bopl_count_unpublished(BoplPuffer *env) {
    for (int i=0; i<2; i++) if (!env->delayed_counted[i]) {
        bopl_count_overflow(env, bopl_state_copy_get(&env->delayed[i])); env->delayed_counted[i]=1;
    }
    if (!env->sample_counted) { bopl_count_overflow(env, bopl_state_copy_get(&env->sample_copy)); env->sample_counted=1; }
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
/* world: `state` is bopl_rl_state(env->world) (bopl_step_finish), so the
 * masks take the controlling slimes the critic block finds in it; the agent
 * shim's states are not its world's (it has none) and its masks are its own. */
/* The style terms of player i's decision on the terminal or ongoing `state`
 * (before any autoreset), summed into the learner's payout counters. */
static float bopl_style_reward(BoplPuffer *env, const BoplRlState *state, const BoplRlEvent *events,
                               uint32_t count, int support, int i) {
    float terms[BS_MAX_TERMS];
    const BoplStyleProgram *p=env->style;
    float total=bopl_style_step(p, state, events, count, support, i, env->style_slots+(size_t)i*p->hist_floats, terms);
    for (uint32_t t=0; t<p->n_terms; t++) { env->style_pay[t]+=terms[t]; env->style_fires[t]+=terms[t]!=0.0f; }
    env->style_decisions++;
    return total;
}
static void bopl_step_finish_with(BoplPuffer *env, const BoplStepPhase *phase, const BoplRlState *state, int world) {
    float *telemetry=env->telemetry?env->telemetry:env->last_telemetry;
    const BoplRlAction *actions=phase->actions;
    const int *fresh=phase->fresh, *holds=phase->holds;
    uint32_t ticks=phase->ticks;
    for (int i=0; i<env->players; i++) env->applied[i]=actions[i];
    env->log.environment_ticks+=state->header.episode_ticks-ticks; env->steps++;
    int terminal=(state->header.flags&BOPL_RL_TERMINAL)!=0;
    int decided=bopl_decided(state, env->players);
    const BoplRlEvent *style_events=NULL; uint32_t style_event_count=0; int style_support=0;
    if (env->style && world) style_events=bopl_style_events(env->world, &style_event_count, &style_support);
    int64_t team_kills[BOPL_MAX_PLAYERS]={0};
    if (env->team_play && bopl_team_kills(world?env->world:NULL, state, env->players, team_kills))
        bopl_failed(env, "team play needs the game's death events");
    for (int i=0; i<env->players; i++) {
        const BoplRlPlayer *player=&state->players[i];
        int slot=bopl_slot(env, i), learner=!env->banks || i==env->side;
        float *stats=telemetry+i*BOPL_TELEMETRY_SIZE;
        uint32_t events[BOPL_EVENT_FIELDS], previous[BOPL_EVENT_FIELDS];
        memcpy(events, &state->telemetry[i], sizeof(events));
        memcpy(previous, &env->previous_telemetry[i], sizeof(previous));
        for (int field=0; field<BOPL_EVENT_FIELDS; field++)
            if (events[field]<previous[field]) bopl_failed(env, "telemetry counter regressed");
        BoplCounters *c=&env->counters;
        /* Each field is summed into the learner's counters as it is stored
         * (the same single add per field and decision), not reloaded from the
         * row after: a vector load of fields just stored one by one waits for
         * the stores to retire instead of forwarding. */
        for (int field=0; field<BOPL_EVENT_FIELDS; field++) {
            float value=(float)(events[field]-previous[field]);
            stats[field]=value;
            if (learner) c->telemetry[field]+=value;
        }
        BoplOutcome o=bopl_outcome(state, i, decided, env->kills[i], env->deaths[i],
                                   env->reward_kill, env->reward_timeout);
        if (env->team_play) {
            /* A teammate's death is not a kill: it is charged like the player's own. */
            o.kills-=team_kills[i];
            o.shaping=env->reward_kill*((float)o.kills-(float)o.deaths-(float)team_kills[i]);
        }
        /* Read before previous_telemetry advances below. */
        int64_t self_deaths=bopl_self_deaths(&state->telemetry[i], &env->previous_telemetry[i]);
        if (o.kills<0) bopl_failed(env, "opponent kill counter regressed");
        /* Terminal reward as paid: the win/loss outcome, or the timeout penalty
         * both players receive on truncation (0 in the phase-1 comparison). */
        float tail[BOPL_TELEMETRY_SIZE-BOPL_EVENT_FIELDS];
        tail[BOPL_REWARD_OUTCOME-BOPL_EVENT_FIELDS]=o.outcome;
        tail[BOPL_REWARD_KILLS-BOPL_EVENT_FIELDS]=env->reward_kill*((float)o.kills-(float)team_kills[i]);
        tail[BOPL_REWARD_DEATHS-BOPL_EVENT_FIELDS]=-env->reward_kill*(float)o.deaths-env->reward_self_death*(float)self_deaths;
        tail[BOPL_EPISODES-BOPL_EVENT_FIELDS]=(float)terminal;
        tail[BOPL_WINS-BOPL_EVENT_FIELDS]=(float)o.won;
        tail[BOPL_TIMEOUTS-BOPL_EVENT_FIELDS]=terminal && o.truncated;
        tail[BOPL_TICKS-BOPL_EVENT_FIELDS]=(float)(state->header.episode_ticks-ticks);
        for (int field=BOPL_EVENT_FIELDS; field<BOPL_TELEMETRY_SIZE; field++) {
            stats[field]=tail[field-BOPL_EVENT_FIELDS];
            if (learner) c->telemetry[field]+=tail[field-BOPL_EVENT_FIELDS];
        }
        env->previous_telemetry[i]=state->telemetry[i];
        /* Retain the original arithmetic/order for the actual PPO reward. */
        float reward=o.shaping+o.outcome;
        reward-=env->reward_self_death*(float)self_deaths;
        if (env->style && learner)
            reward+=bopl_style_scale*bopl_style_reward(env, state, style_events, style_event_count, style_support, i);
        *env->reward_ptr[slot]=reward; env->returns[i]+=reward;
        c->game_steps++;
        if (learner) {
            c->learner_steps++;
            c->kills+=(double)o.kills;
            for (int bit=0; bit<4; bit++) c->buttons[bit]+=(actions[i].buttons>>bit)&1u;
            if (fresh[i]) env->log.hold_choice[holds[i]]++; else env->log.latched_decisions++;
            env->log.team_kills+=(float)team_kills[i];
        } else c->opponent_game_decisions++;
        env->deaths[i]=player->deaths; env->kills[i]=(uint32_t)bopl_opponent_kills(state, i);
        if (terminal) {
            if (env->config.map<BOPL_COUNTER_MAPS) env->log.map_fraction[env->config.map]++;
            *env->terminal_ptr[slot]=1;
            env->log.perf+=o.won; env->log.score+=(float)bopl_opponent_kills(state, i);
            env->log.episode_return+=env->returns[i]; env->log.episode_length+=env->steps; env->log.n++;
            if (learner) {
                env->log.learner_episodes++; env->log.learner_wins+=o.won;
                if (env->team_play) { env->log.team_episodes++; env->log.team_wins+=o.won; }
                if (env->banks) {
                    env->log.historical_episodes++; env->log.historical_wins+=o.won;
                    /* The pool's Elo/swap statistics skip a match whose opponent
                     * weights the trainer replaced while it ran (the swap
                     * cleared boundary_reached; the next c_reset raises it). */
                    int scripted=0;
                    for (int p=0; p<env->players; p++) scripted|=env->bot_select[p];
                    if (scripted) {
                        env->log.bot_n_bank[env->active_bank]++; env->log.bot_score_bank[env->active_bank]+=o.won;
                        /* Every roster match counts. o.won is already gated on
                         * `decided`, so a timeout or a mutual death adds to n
                         * and to decided's complement, and python reads the
                         * draw-aware rate (score + 0.5*(n-decided))/n. */
                        if (env->roster_pick>=0 && env->roster_pick<BOPL_ROSTER_MAX) {
                            env->log.bot_n_entry[env->roster_pick]++;
                            env->log.bot_score_entry[env->roster_pick]+=o.won;
                            env->log.bot_decided_entry[env->roster_pick]+=(float)(decided!=0);
                        }
                    }
                    else if (env->boundary_reached) {
                        /* o.won is already gated on `decided`, so an undecided
                         * round used to land here as a loss. Count the decided
                         * matches too and python recovers wins/losses/draws. */
                        env->log.hist_n_bank[env->active_bank]++;
                        env->log.hist_score_bank[env->active_bank]+=o.won;
                        env->log.hist_decided_bank[env->active_bank]+=(float)(decided!=0);
                    }
                }
            }
        }
    }
    if (terminal) { env->counters.matches++; if (env->config.map<BOPL_COUNTER_MAPS) env->counters.map_matches[env->config.map]++; }
    if (terminal && env->observation_delay) bopl_count_unpublished(env);
    if (terminal && !env->no_autoreset) { c_reset(env); return; }
    if (env->observation_delay) {
        /* Publish the state sampled two decisions ago (counted for overflow
         * as it is encoded), then keep this decision's sample in its place:
         * the buffers swap, nothing is copied. */
        int cursor=env->delay_cursor;
        BoplStateCopy *old=&env->delayed[cursor];
        bopl_encode_copy(env, old, !env->delayed_counted[cursor]);
        BoplStateCopy published=*old; *old=env->sample_copy; env->sample_copy=published;
        env->delayed_counted[cursor]=env->sample_counted; env->sample_counted=1;
        env->delay_cursor=(env->delay_cursor+1)%2;
    } else compute_observations_from(env, state);
    const BoplRlSlime *bodies[BOPL_MAX_PLAYERS];
    bopl_controlling_bodies(state, bodies);
    if (!terminal) for (int i=0; i<env->players; i++) {
        // Own button history is known immediately; it is not world information.
        float *previous=bopl_obs_row(env, i)+bopl_previous_offset(env, i);
        bopl_previous_action(previous, actions[i], env->hold_left[i], env->hold_run[i]);
        if (bopl_layout_of(env, i)==4) bopl4_previous_history(previous, env->action_history[i], actions[i]);
        bopl_write_mask_bodies(env, i, world?bodies:NULL);
    }
    /* The critic's privileged block sees the world as it is now, not as the
     * actor's delayed rows do. */
    bopl_encode_privileged_bodies(env, state, actions, bodies);
}
static void bopl_step_finish_from(BoplPuffer *env, const BoplStepPhase *phase, const BoplRlState *state) {
    bopl_step_finish_with(env, phase, state, 0);
}
static void bopl_step_finish(BoplPuffer *env, const BoplStepPhase *phase) {
    bopl4_defer=env->pack_out!=NULL; bopl4_pending_count=0; bopl4_state_count=0;
    bopl_step_finish_with(env, phase, bopl_rl_state(env->world), 1);
    bopl4_defer=0;
    if (env->pack_out) bopl_pack_rows(env);   /* the rows are final for this step */
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
    free(env->style_slots); env->style_slots=NULL; env->style=NULL;
}
#endif
