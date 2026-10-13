/* What the reward pays, case by case.
 *
 * The simulation is the game's and is held exact against the .NET oracle, so
 * these fixtures are hand-built published states rather than played episodes:
 * each one is the terminal state of a match, with the episode's whole kill and
 * death history on it, so the reward computed from a zero baseline is the
 * episode return. The point of the file is the incentive ordering -- a trade
 * must be worth less than a clean win, and a suicide must not be cheaper than
 * being killed -- which no differential probe can check, because both backends
 * agree on the counters and disagree with nobody.
 *
 * The shapes are not invented: two idle players on map 17 drown on tick 7128
 * and the published terminal is exactly the `double drowning` row -- both
 * flagged winners, one environment death each, on opposing teams.
 *
 * Build and run: make check-reward */
#include "bopl_puffer.h"
#include <assert.h>

#define REWARD_KILL 0.1f
#define REWARD_TIMEOUT (-0.5f)
#define REWARD_SELF_DEATH 0.5f

typedef struct {
    uint32_t team, flags;              /* flags bit 1 is alive, bit 2 WonThisRound */
    uint32_t kills, deaths;            /* the game's counters, self-kills included */
    uint32_t deaths_self, deaths_environment, deaths_opponent;
    float expected;
} Row;
typedef struct { const char *name, *why; uint32_t flags; int players; Row row[4]; } Case;

/* Every fixture is two players on opposing teams unless it says otherwise.
 * `flags` is the header's: BOPL_RL_TERMINAL, plus BOPL_RL_TRUNCATED for a
 * timeout. */
static const Case cases[]={
{"clean win", "one player outlives the other; the kill was earned", BOPL_RL_TERMINAL, 2, {
    {0, 1|2, 1, 0, 0, 0, 0, +1.1f},
    {1, 0,   0, 1, 0, 0, 1, -1.1f}}},

{"mutual death", "one mine kills both, so the game credits its owner with two "
                 "kills (the opponent and itself) and flags both players winners",
 BOPL_RL_TERMINAL, 2, {
    {0, 2, 2, 1, 1, 0, 0, -1.0f},
    {1, 2, 0, 1, 0, 0, 1, -1.1f}}},

{"lone suicide", "the bomber dies to its own mine and the opponent walks away",
 BOPL_RL_TERMINAL, 2, {
    {0, 0,   1, 1, 1, 0, 0, -1.1f},
    {1, 1|2, 0, 0, 0, 0, 0, +1.0f}}},

{"timeout", "the episode hit max_episode_ticks with both players alive",
 BOPL_RL_TERMINAL|BOPL_RL_TRUNCATED, 2, {
    {0, 1, 0, 0, 0, 0, 0, REWARD_TIMEOUT},
    {1, 1, 0, 0, 0, 0, 0, REWARD_TIMEOUT}}},

{"double drowning", "sudden death takes both players on the same tick; nobody "
                    "killed anybody", BOPL_RL_TERMINAL, 2, {
    {0, 2, 0, 1, 0, 1, 0, -1.1f},
    {1, 2, 0, 1, 0, 1, 0, -1.1f}}},

{"split causes", "player 0 falls off the level on the tick its mine kills "
                 "player 1: simultaneous, but only one of them earned a kill",
 BOPL_RL_TERMINAL, 2, {
    {0, 2, 1, 1, 0, 1, 0, -1.0f},
    {1, 2, 0, 1, 0, 0, 1, -1.1f}}},

{"outlasted", "both players drown, one tick apart, so the game awards the round "
              "to the later death alone -- a decided win, not a trade",
 BOPL_RL_TERMINAL, 2, {
    {0, 2, 0, 1, 0, 1, 0, +0.9f},
    {1, 0, 0, 1, 0, 1, 0, -1.1f}}},

{"team win", "teams 0 and 1, two players each: the whole winning team carries "
             "the flag, dead teammate included, which is not a contested round",
 BOPL_RL_TERMINAL, 4, {
    {0, 1|2, 2, 0, 0, 0, 0, +1.2f},
    {0, 2,   0, 1, 0, 0, 1, +0.9f},
    {1, 0,   1, 1, 0, 0, 1, -1.0f},
    {1, 0,   0, 1, 0, 0, 1, -1.1f}}},

{"kill in progress", "a stock traded mid-episode: shaping only, no outcome",
 0, 2, {
    {0, 1, 1, 0, 0, 0, 0, +0.1f},
    {1, 1, 0, 1, 0, 0, 1, -0.1f}}},
};

static void build(BoplRlState *state, const Case *c) {
    memset(state, 0, sizeof(*state));
    state->header.flags=c->flags;
    state->header.player_count=(uint32_t)c->players;
    for (int i=0; i<c->players; i++) {
        state->players[i].id=(uint32_t)i+1;
        state->players[i].team=c->row[i].team;
        state->players[i].flags=c->row[i].flags;
        state->players[i].kills=c->row[i].kills;
        state->players[i].deaths=c->row[i].deaths;
        state->telemetry[i].deaths_self=c->row[i].deaths_self;
        state->telemetry[i].deaths_environment=c->row[i].deaths_environment;
        state->telemetry[i].deaths_opponent=c->row[i].deaths_opponent;
    }
}
/* The episode return: bopl_step_finish_from's arithmetic with the reset
 * baselines still at zero. */
static float reward_of(const BoplRlState *state, int players, int i) {
    BoplOutcome o=bopl_outcome(state, i, bopl_decided(state, players), 0, 0, REWARD_KILL, REWARD_TIMEOUT);
    return o.shaping+o.outcome;
}
static float reward_for(const Case *c, int i) {
    BoplRlState state;
    build(&state, c);
    return reward_of(&state, c->players, i);
}
/* The same return with reward_self_death on: bopl_step_finish_from charges it
 * per self-inflicted or environmental death, after the outcome. */
static float penalized_for(const Case *c, int i) {
    BoplRlState state;
    BoplRlTelemetry zero={0};
    build(&state, c);
    return reward_of(&state, c->players, i)-REWARD_SELF_DEATH*(float)bopl_self_deaths(&state.telemetry[i], &zero);
}
static const Case *find(const char *name) {
    for (size_t i=0; i<sizeof(cases)/sizeof(*cases); i++)
        if (!strcmp(cases[i].name, name)) return &cases[i];
    assert(0); return NULL;
}

int main(void) {
    int failures=0;
    for (size_t i=0; i<sizeof(cases)/sizeof(*cases); i++) {
        const Case *c=&cases[i];
        BoplRlState state;
        build(&state, c);
        printf("%-16s %s\n", c->name, c->why);
        for (int p=0; p<c->players; p++) {
            float reward=reward_of(&state, c->players, p);
            int ok=fabsf(reward-c->row[p].expected)<1e-6f;
            failures+=!ok;
            printf("  player %d team %u: reward %+.4f expected %+.4f %s\n",
                   p, c->row[p].team, (double)reward, (double)c->row[p].expected, ok?"ok":"FAIL");
        }
    }
    /* The ordering the reward exists to produce. Stated as comparisons rather
     * than numbers so it keeps its meaning if reward_kill is retuned. */
    float clean_win=reward_for(find("clean win"), 0), clean_loss=reward_for(find("clean win"), 1);
    float trade_bomber=reward_for(find("mutual death"), 0), trade_victim=reward_for(find("mutual death"), 1);
    float suicide=reward_for(find("lone suicide"), 0), survivor=reward_for(find("lone suicide"), 1);
    struct { const char *claim; int holds; } order[]={
        {"a trade is worth less to the bomber than a clean win", trade_bomber<clean_win},
        {"a trade is worth less to its victim than surviving",   trade_victim<survivor},
        {"being killed is worse than winning",                   clean_loss<clean_win},
        {"a suicide is not cheaper than being killed",           suicide<=clean_loss},
        {"a suicide is worse than killing the opponent first",   suicide<clean_win},
        /* The whole margin a trade has over a plain loss is the kill shaping.
         * With shaping off (the trainer's default) the two are equal, and a
         * trade pays exactly what losing pays. */
        {"a trade beats a plain loss by the kill shaping, no more",
         fabsf((trade_bomber-clean_loss)-REWARD_KILL)<1e-6f},
        /* reward_self_death: a death the player caused costs more than one the
         * opponent caused, and nobody else's return moves. */
        {"with the self-death penalty, a suicide is worse than being killed",
         penalized_for(find("lone suicide"), 0)<penalized_for(find("clean win"), 1)},
        {"with the self-death penalty, a trade costs the bomber the penalty",
         fabsf(penalized_for(find("mutual death"), 0)-(trade_bomber-REWARD_SELF_DEATH))<1e-6f},
        {"with the self-death penalty, drowning costs the penalty",
         fabsf(penalized_for(find("double drowning"), 0)-(reward_for(find("double drowning"), 0)-REWARD_SELF_DEATH))<1e-6f},
        {"the self-death penalty leaves wins and opponent kills alone",
         penalized_for(find("clean win"), 0)==clean_win && penalized_for(find("clean win"), 1)==clean_loss
         && penalized_for(find("mutual death"), 1)==trade_victim && penalized_for(find("lone suicide"), 1)==survivor},
    };
    printf("ordering\n");
    for (size_t i=0; i<sizeof(order)/sizeof(*order); i++) {
        failures+=!order[i].holds;
        printf("  %-55s %s\n", order[i].claim, order[i].holds?"ok":"FAIL");
    }
    if (failures) printf("check_reward: %d FAILED\n", failures);
    else printf("check_reward: pass\n");
    return failures!=0;
}
