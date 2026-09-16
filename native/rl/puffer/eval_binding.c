/* The evaluator uses exactly the training observation, delay and reward code. */
#include "bopl_puffer.h"
/* The layout-v3 row width (the default layout) and the row stride of the
 * observation buffer, which holds either layout (bopl_layout_v4.h). */
int research_obs_size(void) { return BOPL_OBS_SIZE; }
int research_obs_stride(void) { return BOPL_OBS_STRIDE; }
int research_obs_size_of(int layout) { return layout==BOPL4_LAYOUT_VERSION?BOPL4_OBS_SIZE:BOPL_OBS_SIZE; }
int research_telemetry_size(void) { return BOPL_TELEMETRY_SIZE; }
/* Replay extraction shares the production feature encoder. It supplies the
 * delayed world snapshot and immediate previous decision separately. Rows are
 * compact (players x the layout's width); `layout` 3 or 4. */
void research_pack_layout(const BoplRlState *state, const BoplRlConfig *config, float *observations, int layout) {
    BoplPuffer env={0};
    float scratch[BOPL_MAX_PLAYERS*BOPL_OBS_STRIDE];
    env.config=*config; env.players=config->players; env.num_agents=env.players; env.scratch=scratch;
    for (int i=0; i<BOPL_MAX_PLAYERS; i++) env.layout[i]=layout;
    compute_observations_from(&env, state);
    int width=research_obs_size_of(layout);
    for (int i=0; i<env.players; i++) memcpy(observations+(size_t)i*width, scratch+(size_t)i*BOPL_OBS_STRIDE, (size_t)width*sizeof(float));
}
void research_pack(const BoplRlState *state, const BoplRlConfig *config, float *observations) {
    research_pack_layout(state, config, observations, 3);
}
/* Select player p's observation layout (3 or 4). Call after research_reset,
 * like research_bot: the row is re-encoded from the current world state (the
 * episode's first state, which is what the reset row shows) with the neutral
 * own-action tail. Persists across resets. Returns -1 for a bad player or layout. */
int research_set_layout(BoplPuffer *env, int player, int layout) {
    if (player<0 || player>=env->players || (layout!=3 && layout!=BOPL4_LAYOUT_VERSION)) return -1;
    if (bopl_layout_of(env, player)==layout) return 0;
    env->layout[player]=layout;
    float *row=bopl_obs_row(env, player);
    memset(row, 0, BOPL_OBS_STRIDE*sizeof(float)); bopl_written_forget_row(env, row);
    const BoplRlState *now=bopl_rl_state(env->world);
    if (layout==BOPL4_LAYOUT_VERSION) bopl_encode_v4_row(env, now, player, 0); else bopl_encode_v3_row(env, now, player, 0);
    row[bopl_previous_offset(env, player)]=1;
    bopl_encode_privileged(env, now, NULL);
    return 0;
}
int research_get_layout(BoplPuffer *env, int player) { return player<0 || player>=env->players?-1:bopl_layout_of(env, player); }
/* Standalone environments route every player straight to its own row. */
static void research_route(BoplPuffer *env) {
    for (int s=0; s<env->num_agents; s++) {
        env->obs_ptr[s]=env->observations+(size_t)s*BOPL_OBS_STRIDE; env->action_ptr[s]=env->actions+BOPL_NUM_ATNS*s;
        env->reward_ptr[s]=env->rewards+s; env->terminal_ptr[s]=env->terminals+s;
        env->mask_ptr[s]=env->action_mask+(size_t)s*BOPL_LOGITS;
    }
}
/* Action mask of the next decision, 2 x BOPL_LOGITS bytes (1 = allowed). The
 * evaluator applies it before sampling exactly as the trainer's sampler does. */
const unsigned char *research_mask(BoplPuffer *env) { return env->action_mask; }
int research_mask_size(void) { return BOPL_LOGITS; }
/* Replay extraction: the tail is rebuilt from the recorded inputs alone. Hold
 * state is reconstructed as "consecutive ticks down / repeat" by the caller and
 * passed in hold_run (4 ints per player, may be NULL); no latch is in force. */
void research_previous(float *observations, const BoplRlAction *actions, const int *hold_run, int count) {
    for (int i=0; i<count; i++)
        bopl_previous_action(observations+(size_t)(i+1)*BOPL_OBS_SIZE-BOPL_PREVIOUS_FEATURES, actions[i], 0, hold_run?hold_run+4*i:NULL);
}
int research_layout(int *out, int capacity) {
    int values[]={BOPL_OBS_SIZE, BOPL_OBS_PREFIX, BOPL_SELF_FEATURES,
                  BOPL_OBS_PLAYERS, BOPL_PLAYER_FEATURES, BOPL_OBS_SLOTS, BOPL_SLOT_FEATURES,
                  BOPL_OBS_ENTITIES, BOPL_ENTITY_FEATURES, BOPL_OBS_SLIMES, BOPL_SLIME_FEATURES,
                  BOPL_PREVIOUS_FEATURES, BOPL_OBS_SELF_OFFSET, BOPL_OBS_PLAYER_OFFSET,
                  BOPL_OBS_SLOT_OFFSET, BOPL_OBS_ENTITY_OFFSET, BOPL_OBS_SLIME_OFFSET,
                  BOPL_OBS_PREVIOUS_OFFSET, BOPL_SLOT_ABILITY_ID, BOPL_ENTITY_ABILITY_ID,
                  BOPL_EMBEDDING_IDS, BOPL_ACT_AIM, BOPL_ACT_BUTTONS, BOPL_ACT_HOLD, BOPL_HOLD_MAX};
    int count=(int)(sizeof(values)/sizeof(values[0]));
    if (capacity<count) return -count;
    memcpy(out, values, sizeof(values)); return count;
}
/* The compiled layout v4 (bopl_layout_v4.h), checked against research/layout.py. */
int research_layout_v4(int *out, int capacity) {
    int values[]={BOPL4_OBS_SIZE, BOPL4_GLOBAL, BOPL4_SELF, BOPL4_OPPONENT, BOPL4_PREVIOUS, BOPL4_RAYS, BOPL4_RAY_FEATURES,
                  BOPL4_FLAT, BOPL4_PLAYERS, BOPL4_PLAYER_FEATURES, BOPL4_SLOTS, BOPL4_SLOT_FEATURES,
                  BOPL4_ENTITIES, BOPL4_ENTITY_FEATURES, BOPL4_SLIMES, BOPL4_SLIME_FEATURES, BOPL4_PRIVILEGED,
                  BOPL4_OBS_SELF_OFFSET, BOPL4_OBS_OPPONENT_OFFSET, BOPL4_OBS_PREVIOUS_OFFSET, BOPL4_OBS_RAY_OFFSET,
                  BOPL4_OBS_PLAYER_OFFSET, BOPL4_OBS_SLOT_OFFSET, BOPL4_OBS_ENTITY_OFFSET, BOPL4_OBS_SLIME_OFFSET,
                  BOPL4_OBS_PRIVILEGED_OFFSET, BOPL4_SLOT_ABILITY_ID, BOPL4_ENTITY_ABILITY_ID, BOPL_OBS_STRIDE};
    int count=(int)(sizeof(values)/sizeof(values[0]));
    if (capacity<count) return -count;
    memcpy(out, values, sizeof(values)); return count;
}
int research_num_atns(void) { return BOPL_NUM_ATNS; }
int research_check(BoplPuffer *env) { return env->log.game_errors; } /* observation overflow is a logged, non-fatal truncation */
unsigned long research_overflow(BoplPuffer *env) { return env->log.observation_overflow; }
/* The research environment carries the state of a decision split into ticks
 * (research_step_begin/tick/finish). BoplPuffer is the first member, so every
 * BoplPuffer* handed out by research_open is also a ResearchEnv*. */
typedef struct {
    BoplPuffer env;
    BoplStepPhase phase;
    int open, tick, sampled;
} ResearchEnv;
BoplPuffer *research_open(uint32_t seed, uint32_t map, uint32_t limit, int delay) {
    ResearchEnv *research=calloc(1, sizeof(*research));
    if (!research) return NULL;
    BoplPuffer *env=&research->env;
    env->players=2; env->num_agents=2; env->repeat=4; env->observation_delay=delay;
    env->no_autoreset=1; env->rng=seed;
    env->reward_kill=0.1f; env->reward_timeout=0;  /* evaluation reports the phase-1 reward telemetry */
    bopl_rl_default_config(&env->config, 2);
    env->config.map=map; env->config.stocks=0; /* vanilla Bopl: one life per round, matching training */ env->config.max_episode_ticks=limit;
    for (int i=0; i<2; i++) {
        env->config.player[i].abilities[1]=4;
        env->config.player[i].abilities[2]=14;
    }
    env->observations=calloc(2*BOPL_OBS_STRIDE, sizeof(float));
    env->actions=calloc(2*BOPL_NUM_ATNS, sizeof(float)); env->rewards=calloc(2, sizeof(float)); env->terminals=calloc(2, sizeof(float));
    env->action_mask=calloc(2*BOPL_LOGITS, 1);
    if (!env->observations || !env->actions || !env->rewards || !env->terminals || !env->action_mask) abort();
    research_route(env);
    init(env); c_reset(env); return env;
}
/* Reward telemetry configuration for an open evaluator: the kill/death
 * shaping magnitude and the timeout outcome (<=0, both finite). Takes effect
 * from the next step; research_open keeps the phase-1 defaults (0.1, 0). */
int research_rewards(BoplPuffer *env, float kill, float timeout) {
    if (!isfinite(kill) || !isfinite(timeout) || timeout>0) return -1;
    env->reward_kill=kill; env->reward_timeout=timeout; return 0;
}
void research_reset(BoplPuffer *env, uint32_t seed, uint32_t map) {
    env->rng=seed; env->config.map=map; c_reset(env);
    memset(env->last_telemetry, 0, sizeof(env->last_telemetry));
    ((ResearchEnv*)env)->open=0;  /* a reset discards an unfinished tick-split decision */
}
/* Scripted opponents (bopl_bots.h): pin a bot on one player. It decides inside
 * research_step from the true game state; the caller's action floats for that
 * player are ignored. kind 0 detaches. Persists across research_reset (each
 * match re-seeds the bot from the episode seed, bopl_bot_seed); this call
 * also re-arms the bot for the already-reset current episode with that same
 * seed, so the game does not depend on whether the bot was selected before or
 * after the reset. */
int research_bot(BoplPuffer *env, int player, int kind, int level) {
    if (player<0 || player>=env->players || kind<0 || kind>BOPL_BOT_KINDS
        || level<0 || level>=BOPL_BOT_LEVELS) return -1;
    env->bot_select[player]=(uint8_t)kind;
    env->bot_select_level[player]=(uint8_t)level;
    if (kind) bopl_bot_reset(&env->bots[player], kind, level, bopl_bot_seed(env, player));
    return 0;
}
const char *research_bot_name(int kind) {
    return kind>=0 && kind<=BOPL_BOT_KINDS?BOPL_BOT_NAMES[kind]:NULL;
}
/* Whether a kind runs rollouts on a forked game (bopl_rl_fork), which only the
 * in-process Rust sim provides; research_forks says whether this environment's
 * game does. Callers skip such kinds on other backends instead of aborting. */
int research_bot_forks(int kind) {
    return kind==BOPL_BOT_SEARCH;
}
int research_forks(BoplPuffer *env) {
    BoplRl *fork=bopl_rl_fork(env->world, NULL);
    if (!fork) return 0;
    bopl_rl_close(fork);
    return 1;
}
void research_step(BoplPuffer *env, const float *actions) {
    if (((ResearchEnv*)env)->open) { fprintf(stderr, "research_step during a tick-split decision\n"); abort(); }
    memcpy(env->actions, actions, 2*BOPL_NUM_ATNS*sizeof(float)); c_step(env);
}
/* One decision split into its ticks, for real-time play that shows every
 * tick: begin fixes the decision (scripted bots, latch) exactly as
 * research_step does; each tick advances the world one game tick with the
 * latched actions and returns the ticks still to run (0 when the four are
 * done or the episode ended); finish completes the decision's bookkeeping.
 * The observation sample is taken after phase 3 (or at the end of the episode
 * within the first three ticks), as bopl_rl_step_sampled(3, 1) returns it, so
 * observations, masks, telemetry, rewards and the replay are identical to
 * research_step's; check_tick_step.py proves it on fixed seeds. Returns -1 on
 * misuse (ticks left, no decision open, or a decision already open). */
int research_step_begin(BoplPuffer *env, const float *actions) {
    ResearchEnv *research=(ResearchEnv*)env;
    if (research->open) return -1;
    memcpy(env->actions, actions, 2*BOPL_NUM_ATNS*sizeof(float));
    bopl_step_begin(env, &research->phase);
    research->open=1; research->tick=0; research->sampled=0;
    return 0;
}
int research_step_tick(BoplPuffer *env) {
    ResearchEnv *research=(ResearchEnv*)env;
    if (!research->open) return -1;
    const BoplRlState *state=bopl_rl_state(env->world);
    if (research->tick>=env->repeat || (state->header.flags&BOPL_RL_TERMINAL)) return 0;
    if (bopl_rl_step(env->world, research->phase.actions, env->players, 1)) bopl_failed(env, "tick");
    research->tick++;
    state=bopl_rl_state(env->world);
    int ended=(state->header.flags&BOPL_RL_TERMINAL)!=0;
    if (env->observation_delay && !research->sampled && (research->tick==env->repeat-1 || ended)) {
        bopl_step_sample(env, state); research->sampled=1;
    }
    return ended?0:env->repeat-research->tick;
}
int research_step_finish(BoplPuffer *env) {
    ResearchEnv *research=(ResearchEnv*)env;
    if (!research->open) return -1;
    const BoplRlState *state=bopl_rl_state(env->world);
    if (research->tick<env->repeat && !(state->header.flags&BOPL_RL_TERMINAL)) return -1;
    if (env->observation_delay && !research->sampled) { fprintf(stderr, "tick-split decision without a sample\n"); abort(); }
    bopl_step_finish(env, &research->phase);
    research->open=0;
    return 0;
}
const float *research_obs(BoplPuffer *env) { return env->observations; }
const float *research_telemetry(BoplPuffer *env) { return env->last_telemetry; }
const BoplRlState *research_state(BoplPuffer *env) { return bopl_rl_state(env->world); }
int research_replay(BoplPuffer *env, const char *path) {
    const void *bytes; size_t length;
    if (bopl_rl_replay(env->world, &bytes, &length)) return 1;
    FILE *file=fopen(path, "wb"); if (!file) return 1;
    int status=fwrite(bytes, 1, length, file)!=length;
    return fclose(file)?1:status;
}
void research_close(BoplPuffer *env) {
    if (!env) return;
    c_close(env); free(env->observations); free(env->actions); free(env->rewards); free(env->terminals); free(env->action_mask); free(env);
}
