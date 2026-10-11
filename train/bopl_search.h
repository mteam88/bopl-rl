/* Decision-time search, phase 1: a policy+value guided lookahead on forks of
 * the live game, played on top of a trained checkpoint.
 *
 * Every decision the search draws K candidate (aim, buttons, hold) tuples from
 * the policy's masked prior without replacement (Gumbel-top-k over the joint
 * head distribution), scores each by N rollouts of D decisions on a private
 * fork of the world where both players follow the policy (the opponent is the
 * same checkpoint fed the opponent's real rows: a self-model, with its own
 * recurrent state tracked across the real game), bootstraps the leaf with the
 * critic (the privileged block is encoded from the fork's current state, as
 * in training) and commits to the candidate with the best mean return.
 * Optional sequential halving drops the worse half of the candidates between
 * rounds. The chosen action's latch then runs in the real environment as it
 * would for the raw policy: only the aim is free while buttons are latched,
 * and the search picks it.
 *
 * Rollouts run on shadow BoplPuffer environments: a struct copy of the live
 * one whose world is a fork and whose delayed state copies, rows, masks and
 * latches are deep copies, stepped with the same begin/advance/finish
 * bookkeeping as training, so rollout observations (nine-tick delay, own
 * action tail), masks and rewards are exactly the trained ones. The game is
 * deterministic given the inputs (the fork copies the RNG stream), so all
 * search randomness comes from policy sampling.
 *
 * For phase 2 (expert iteration) every candidate keeps its visit count, mean
 * return and prior log-probability (research_search_stats). */
#ifndef BOPL_SEARCH_H
#define BOPL_SEARCH_H
#include "bopl_puffer.h"
#include <time.h>

/* train/policy_eval.c, linked into the research library (Makefile). */
void *bopl_policy_eval_create_v4(const float *data, size_t count, int batch, int hidden, int layers,
                                 int aim_fourier, int aim_ar, const int *shape);
void bopl_policy_eval_step_raw(void *handle, float *observations, const float *state, float *rows, float *next_state);
void bopl_policy_eval_game_logits(void *handle, const float *rows, const int *buttons, float *out);
void bopl_policy_eval_close(void *handle);
int bopl_policy_eval_width(void *handle);

enum { BOPL_SEARCH_MAX_K=64, BOPL_SEARCH_STAT_WIDTH=7 };

typedef struct {
    int aim, buttons, hold;
    int visits; double sum;
    float prior;        /* log-probability under the masked root prior */
    float gumbel;       /* the Gumbel-top-k score the candidate was drawn with */
    int alive;
} BoplSearchCand;

typedef struct {
    BoplPuffer *env;                /* the live research environment (not owned) */
    float *weights; size_t weight_count; int hidden, layers, aim_fourier, aim_ar, shape[8];
    void *policy;                   /* batch-2 evaluator: row 0 player 0, row 1 player 1 */
    int width, state_floats;        /* decoder width; layers*2*hidden recurrent floats */
    int rollouts;                   /* shadows allocated (max rollouts of one round) */
    BoplPuffer *shadow;
    float *roll_state, *roll_next;  /* rollouts x state_floats */
    float *rows, *logits;           /* 2 x (width+1), 2 x BOPL_DECODER scratch */
    float *root_state, *root_next;  /* the real recurrent states of both players */
    float *root_logits;             /* 2 x BOPL_DECODER at the root */
    /* Budget knobs. k candidates, n rollouts per candidate and round, depth
     * decisions per rollout, halving rounds, gamma, reward scale (kill/timeout
     * as trained, so returns match the value head), search_latched: run the
     * search while the buttons are latched (only aim is free) or sample. */
    int k, n, depth, halving, search_latched;
    float gamma, kill, timeout;
    uint64_t rng;
    BoplSearchCand cand[BOPL_SEARCH_MAX_K]; int cands, chosen;
    float root_value[2];
    /* Counters since reset: decisions, searched decisions, rollouts, decision
     * steps, policy forwards (rows) and wall seconds in research_search_act. */
    double seconds; long decisions, searched, rollouts_run, steps_run, forwards_run;
    char error[256];
} BoplSearch;

static double bopl_search_now(void) {
    struct timespec ts; timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec+1e-9*(double)ts.tv_nsec;
}
static uint64_t bopl_search_mix(uint64_t *s) {
    uint64_t z=(*s+=0x9E3779B97F4A7C15ull);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ull; z=(z^(z>>27))*0x94D049BB133111EBull;
    return z^(z>>31);
}
static double bopl_search_uniform(uint64_t *s) { return ((double)(bopl_search_mix(s)>>11)+0.5)*(1.0/9007199254740992.0); }
static double bopl_search_gumbel(uint64_t *s) { return -log(-log(bopl_search_uniform(s))); }

/* Masked log-softmax; -INFINITY where masked. Returns the allowed count. */
static int bopl_search_log_softmax(const float *logits, const unsigned char *mask, int count, float *out) {
    float best=-INFINITY; int allowed=0;
    for (int i=0; i<count; i++) if (mask[i] && logits[i]>best) best=logits[i];
    for (int i=0; i<count; i++) if (mask[i]) allowed++;
    if (!allowed) { for (int i=0; i<count; i++) out[i]=-INFINITY; return 0; }
    double total=0;
    for (int i=0; i<count; i++) if (mask[i]) total+=exp((double)(logits[i]-best));
    float lz=(float)log(total);
    for (int i=0; i<count; i++) out[i]=mask[i]?logits[i]-best-lz:-INFINITY;
    return allowed;
}
static int bopl_search_sample_head(const float *logits, const unsigned char *mask, int count, uint64_t *rng) {
    float best=-INFINITY; int allowed=0, last=0;
    for (int i=0; i<count; i++) if (mask[i]) { allowed++; last=i; if (logits[i]>best) best=logits[i]; }
    if (!allowed) return 0;
    double total=0;
    for (int i=0; i<count; i++) if (mask[i]) total+=exp((double)(logits[i]-best));
    double u=bopl_search_uniform(rng)*total, acc=0;
    for (int i=0; i<count; i++) if (mask[i]) { acc+=exp((double)(logits[i]-best)); if (u<acc) return i; }
    return last;
}
/* The three heads of player `p` sampled from game logits `row` under `mask`. */
static void bopl_search_sample_action(const float *row, const unsigned char *mask, uint64_t *rng, float *out) {
    out[0]=(float)bopl_search_sample_head(row, mask, BOPL_ACT_AIM, rng);
    out[1]=(float)bopl_search_sample_head(row+BOPL_ACT_AIM, mask+BOPL_ACT_AIM, BOPL_ACT_BUTTONS, rng);
    out[2]=(float)bopl_search_sample_head(row+BOPL_ACT_AIM+BOPL_ACT_BUTTONS, mask+BOPL_ACT_AIM+BOPL_ACT_BUTTONS, BOPL_ACT_HOLD, rng);
}

static void bopl_search_route(BoplPuffer *env) {
    for (int s=0; s<env->num_agents; s++) {
        env->obs_ptr[s]=env->observations+(size_t)s*BOPL_OBS_STRIDE; env->action_ptr[s]=env->actions+BOPL_NUM_ATNS*s;
        env->reward_ptr[s]=env->rewards+s; env->terminal_ptr[s]=env->terminals+s;
        env->mask_ptr[s]=env->action_mask+(size_t)s*BOPL_LOGITS;
    }
}
static int bopl_search_shadow_alloc(BoplPuffer *sh) {
    memset(sh, 0, sizeof(*sh));
    sh->observations=calloc(2*BOPL_OBS_STRIDE, sizeof(float));
    sh->actions=calloc(2*BOPL_NUM_ATNS, sizeof(float)); sh->rewards=calloc(2, sizeof(float)); sh->terminals=calloc(2, sizeof(float));
    sh->action_mask=calloc(2*BOPL_LOGITS, 1);
    return sh->observations && sh->actions && sh->rewards && sh->terminals && sh->action_mask?0:-1;
}
static void bopl_search_shadow_free(BoplPuffer *sh) {
    if (sh->world) bopl_rl_close(sh->world);
    for (int i=0; i<2; i++) bopl_state_copy_free(&sh->delayed[i]);
    bopl_state_copy_free(&sh->sample_copy);
    free(sh->observations); free(sh->actions); free(sh->rewards); free(sh->terminals); free(sh->action_mask);
    memset(sh, 0, sizeof(*sh));
}
/* The shadow becomes the live environment as of now: its bookkeeping (latches,
 * action history, kill/death counters, delay cursor) by struct copy, its world
 * a fork, its rows, masks and delayed state copies deep copies. Returns -1
 * when the world cannot fork. */
static int bopl_search_shadow_reset(BoplSearch *s, BoplPuffer *sh) {
    BoplPuffer *real=s->env;
    float *obs=sh->observations, *act=sh->actions, *rew=sh->rewards, *ter=sh->terminals;
    unsigned char *mask=sh->action_mask;
    BoplStateCopy d0=sh->delayed[0], d1=sh->delayed[1], sc=sh->sample_copy;
    BoplRl *fork=sh->world;
    *sh=*real;
    sh->observations=obs; sh->actions=act; sh->rewards=rew; sh->terminals=ter; sh->action_mask=mask;
    sh->num_agents=sh->players; sh->banks=0; sh->side=0;
    bopl_search_route(sh);
    sh->delayed[0]=d0; sh->delayed[1]=d1; sh->sample_copy=sc;
    for (int i=0; i<2; i++) bopl_state_copy(&sh->delayed[i], bopl_state_copy_get(&real->delayed[i]));
    bopl_state_copy(&sh->sample_copy, bopl_state_copy_get(&real->sample_copy));
    sh->world=bopl_rl_fork(real->world, fork);
    if (!sh->world) { sh->world=fork; return -1; }
    sh->spare=NULL; sh->spare_opening=0; sh->world_opening=0;
    sh->style=NULL; sh->style_slots=NULL; sh->pack_out=NULL; sh->pack_base=NULL; sh->telemetry=NULL; sh->scratch=NULL;
    memset(sh->bot_select, 0, sizeof(sh->bot_select));
    sh->written_count=0;   /* the rows are this shadow's: the next encode rewrites them whole */
    memcpy(obs, real->observations, (size_t)2*BOPL_OBS_STRIDE*sizeof(float));
    memcpy(mask, real->action_mask, (size_t)2*BOPL_LOGITS);
    sh->reward_kill=s->kill; sh->reward_timeout=s->timeout;
    sh->no_autoreset=1;
    return 0;
}
/* One decision of a shadow with the actions in its action buffer. Returns the
 * terminal flag; the rewards are in sh->rewards. */
static int bopl_search_shadow_step(BoplPuffer *sh) {
    BoplStepPhase phase;
    bopl_step_begin_from(sh, &phase, bopl_rl_state(sh->world));
    bopl_step_advance(sh, &phase);
    const BoplRlState *state=bopl_rl_state(sh->world);
    bopl_step_finish_with(sh, &phase, state, 1);
    return (state->header.flags&BOPL_RL_TERMINAL)!=0;
}

static void bopl_search_close(BoplSearch *s) {
    if (!s) return;
    if (s->shadow) for (int r=0; r<s->rollouts; r++) bopl_search_shadow_free(&s->shadow[r]);
    free(s->shadow); free(s->roll_state); free(s->roll_next); free(s->rows); free(s->logits);
    free(s->root_state); free(s->root_next); free(s->root_logits); free(s->weights);
    if (s->policy) bopl_policy_eval_close(s->policy);
    free(s);
}
static BoplSearch *bopl_search_open(BoplPuffer *env, const float *weights, size_t count, int hidden, int layers,
                                    int aim_fourier, int aim_ar, const int *shape, int rollouts, uint64_t seed) {
    if (!env || !weights || rollouts<1 || hidden<1 || layers<1) return NULL;
    BoplSearch *s=calloc(1, sizeof(*s)); if (!s) return NULL;
    s->env=env; s->rollouts=rollouts; s->hidden=hidden; s->layers=layers; s->aim_fourier=aim_fourier; s->aim_ar=aim_ar;
    memcpy(s->shape, shape, sizeof(s->shape));
    s->weights=malloc(count*sizeof(float)); s->weight_count=count;
    if (!s->weights) { bopl_search_close(s); return NULL; }
    memcpy(s->weights, weights, count*sizeof(float));
    s->policy=bopl_policy_eval_create_v4(s->weights, count, 2, hidden, layers, aim_fourier, aim_ar, shape);
    if (!s->policy) { bopl_search_close(s); return NULL; }
    s->width=bopl_policy_eval_width(s->policy); s->state_floats=layers*2*hidden;
    s->shadow=calloc((size_t)rollouts, sizeof(BoplPuffer));
    s->roll_state=calloc((size_t)rollouts*s->state_floats, sizeof(float));
    s->roll_next=calloc((size_t)rollouts*s->state_floats, sizeof(float));
    s->rows=calloc((size_t)2*(s->width+1), sizeof(float)); s->logits=calloc(2*BOPL_DECODER, sizeof(float));
    s->root_state=calloc((size_t)s->state_floats, sizeof(float)); s->root_next=calloc((size_t)s->state_floats, sizeof(float));
    s->root_logits=calloc(2*BOPL_DECODER, sizeof(float));
    if (!s->shadow || !s->roll_state || !s->roll_next || !s->rows || !s->logits || !s->root_state || !s->root_next || !s->root_logits) { bopl_search_close(s); return NULL; }
    for (int r=0; r<rollouts; r++) if (bopl_search_shadow_alloc(&s->shadow[r])) { bopl_search_close(s); return NULL; }
    s->k=8; s->n=2; s->depth=8; s->halving=0; s->search_latched=1;
    s->gamma=0.995f; s->kill=0.2883693f; s->timeout=-1.0f;
    s->rng=seed?seed:0x5EEDull;
    return s;
}
/* A new game: the recurrent states restart, the counters too. */
static void bopl_search_reset(BoplSearch *s, uint64_t seed) {
    memset(s->root_state, 0, (size_t)s->state_floats*sizeof(float));
    s->seconds=0; s->decisions=s->searched=s->rollouts_run=s->steps_run=s->forwards_run=0;
    s->cands=0; s->chosen=-1;
    if (seed) s->rng=seed;
}
static int bopl_search_configure(BoplSearch *s, int k, int n, int depth, int halving, int search_latched,
                                 float gamma, float kill, float timeout) {
    if (k<1 || k>BOPL_SEARCH_MAX_K || n<1 || depth<1 || k*n>s->rollouts || !(gamma>0) || gamma>1 || !isfinite(kill) || !isfinite(timeout)) return -1;
    s->k=k; s->n=n; s->depth=depth; s->halving=halving!=0; s->search_latched=search_latched!=0;
    s->gamma=gamma; s->kill=kill; s->timeout=timeout;
    return 0;
}

/* Policy forward of a shadow's two rows; the sampled next actions of both
 * players go to its action buffer, `value` gets player `me`'s critic value. */
static void bopl_search_forward_shadow(BoplSearch *s, BoplPuffer *sh, float *state, float *next, int me, float *value) {
    bopl_policy_eval_step_raw(s->policy, sh->observations, state, s->rows, next);
    bopl_policy_eval_game_logits(s->policy, s->rows, NULL, s->logits);
    for (int p=0; p<2; p++) {
        const float *row=s->logits+(size_t)p*BOPL_DECODER;
        if (s->aim_ar) {
            /* Buttons first, then the aim head of that combo (bopl_aim.h). */
            int combos[2]={0, 0};
            float *a=sh->actions+BOPL_NUM_ATNS*p;
            a[1]=(float)bopl_search_sample_head(row+BOPL_ACT_AIM, sh->action_mask+(size_t)p*BOPL_LOGITS+BOPL_ACT_AIM, BOPL_ACT_BUTTONS, &s->rng);
            combos[p]=(int)a[1];
            bopl_policy_eval_game_logits(s->policy, s->rows, combos, s->logits);
            a[0]=(float)bopl_search_sample_head(row, sh->action_mask+(size_t)p*BOPL_LOGITS, BOPL_ACT_AIM, &s->rng);
            a[2]=(float)bopl_search_sample_head(row+BOPL_ACT_AIM+BOPL_ACT_BUTTONS, sh->action_mask+(size_t)p*BOPL_LOGITS+BOPL_ACT_AIM+BOPL_ACT_BUTTONS, BOPL_ACT_HOLD, &s->rng);
        } else bopl_search_sample_action(row, sh->action_mask+(size_t)p*BOPL_LOGITS, &s->rng, sh->actions+BOPL_NUM_ATNS*p);
    }
    *value=s->logits[(size_t)me*BOPL_DECODER+BOPL_LOGITS];
    s->forwards_run+=2;
}

/* Gumbel-top-k over the joint masked prior of player `me` at the root. */
static int bopl_search_candidates(BoplSearch *s, int me, int k) {
    const float *row=s->root_logits+(size_t)me*BOPL_DECODER;
    const unsigned char *mask=s->env->action_mask+(size_t)me*BOPL_LOGITS;
    float la[BOPL_ACT_AIM], lb[BOPL_ACT_BUTTONS], lh[BOPL_ACT_HOLD];
    int na=bopl_search_log_softmax(row, mask, BOPL_ACT_AIM, la);
    int nb=bopl_search_log_softmax(row+BOPL_ACT_AIM, mask+BOPL_ACT_AIM, BOPL_ACT_BUTTONS, lb);
    int nh=bopl_search_log_softmax(row+BOPL_ACT_AIM+BOPL_ACT_BUTTONS, mask+BOPL_ACT_AIM+BOPL_ACT_BUTTONS, BOPL_ACT_HOLD, lh);
    if (!na || !nb || !nh) return 0;
    int count=0, lowest=0;
    for (int b=0; b<BOPL_ACT_BUTTONS; b++) {
        if (!mask[BOPL_ACT_AIM+b]) continue;
        for (int h=0; h<BOPL_ACT_HOLD; h++) {
            if (!mask[BOPL_ACT_AIM+BOPL_ACT_BUTTONS+h]) continue;
            float lbh=lb[b]+lh[h];
            for (int a=0; a<BOPL_ACT_AIM; a++) {
                if (!mask[a]) continue;
                float prior=la[a]+lbh, score=prior+(float)bopl_search_gumbel(&s->rng);
                if (count==k && score<=s->cand[lowest].gumbel) continue;
                int at=count<k?count++:lowest;
                s->cand[at].aim=a; s->cand[at].buttons=b; s->cand[at].hold=h;
                s->cand[at].prior=prior; s->cand[at].gumbel=score;
                s->cand[at].visits=0; s->cand[at].sum=0; s->cand[at].alive=1;
                if (count==k) { lowest=0; for (int i=1; i<k; i++) if (s->cand[i].gumbel<s->cand[lowest].gumbel) lowest=i; }
            }
        }
    }
    return count;
}

/* `n` rollouts of every alive candidate, batched by depth. */
static int bopl_search_round(BoplSearch *s, int me, int n) {
    int used=0;
    static _Thread_local int owner[4096];
    static _Thread_local float ret[4096], disc[4096], value[4096];
    static _Thread_local unsigned char done[4096];
    if (s->rollouts>4096) return -1;
    const unsigned char *opp_mask=s->env->action_mask+(size_t)(1-me)*BOPL_LOGITS;
    const float *opp_row=s->root_logits+(size_t)(1-me)*BOPL_DECODER;
    for (int c=0; c<s->cands; c++) {
        if (!s->cand[c].alive) continue;
        for (int i=0; i<n && used<s->rollouts; i++) {
            BoplPuffer *sh=&s->shadow[used];
            if (bopl_search_shadow_reset(s, sh)) { snprintf(s->error, sizeof(s->error), "the world cannot fork"); return -1; }
            memcpy(s->roll_state+(size_t)used*s->state_floats, s->root_next, (size_t)s->state_floats*sizeof(float));
            float *mine=sh->actions+BOPL_NUM_ATNS*me, *theirs=sh->actions+BOPL_NUM_ATNS*(1-me);
            mine[0]=(float)s->cand[c].aim; mine[1]=(float)s->cand[c].buttons; mine[2]=(float)s->cand[c].hold;
            if (s->aim_ar) {
                int combos[2]={0, 0};
                theirs[1]=(float)bopl_search_sample_head(opp_row+BOPL_ACT_AIM, opp_mask+BOPL_ACT_AIM, BOPL_ACT_BUTTONS, &s->rng);
                combos[1-me]=(int)theirs[1];
                bopl_policy_eval_game_logits(s->policy, s->rows, combos, s->logits);   /* s->rows still holds the root rows */
                const float *r=s->logits+(size_t)(1-me)*BOPL_DECODER;
                theirs[0]=(float)bopl_search_sample_head(r, opp_mask, BOPL_ACT_AIM, &s->rng);
                theirs[2]=(float)bopl_search_sample_head(r+BOPL_ACT_AIM+BOPL_ACT_BUTTONS, opp_mask+BOPL_ACT_AIM+BOPL_ACT_BUTTONS, BOPL_ACT_HOLD, &s->rng);
            } else bopl_search_sample_action(opp_row, opp_mask, &s->rng, theirs);
            owner[used]=c; ret[used]=0; disc[used]=1; done[used]=0; value[used]=0;
            used++;
        }
    }
    for (int d=0; d<s->depth; d++) {
        for (int r=0; r<used; r++) {
            if (done[r]) continue;
            BoplPuffer *sh=&s->shadow[r];
            int terminal=bopl_search_shadow_step(sh);
            s->steps_run++;
            ret[r]+=disc[r]*sh->rewards[me]; disc[r]*=s->gamma;
            if (terminal) { done[r]=1; value[r]=0; continue; }
            /* Next decision's actions and, at the horizon, the leaf value. */
            float *state=s->roll_state+(size_t)r*s->state_floats, *next=s->roll_next+(size_t)r*s->state_floats;
            bopl_search_forward_shadow(s, sh, state, next, me, &value[r]);
            memcpy(state, next, (size_t)s->state_floats*sizeof(float));
        }
    }
    for (int r=0; r<used; r++) {
        double g=ret[r]+(done[r]?0.0:(double)disc[r]*value[r]);
        s->cand[owner[r]].sum+=g; s->cand[owner[r]].visits++;
    }
    s->rollouts_run+=used;
    return used;
}

/* The decision of player `me` on the live environment's current rows: the
 * real policy forward of both players (their recurrent states advance here,
 * once per decision), then the search. out[3] = aim, buttons, hold. Returns
 * 0, or -1 with s->error set. `searched` tells whether the search ran (a
 * latched decision with search_latched off, or a budget of one, samples). */
static int bopl_search_act(BoplSearch *s, int me, int *out, int *searched) {
    double t0=bopl_search_now();
    BoplPuffer *env=s->env;
    if (me<0 || me>1) { snprintf(s->error, sizeof(s->error), "player %d", me); return -1; }
    for (int p=0; p<2; p++) if (bopl_layout_of(env, p)!=4) {
        snprintf(s->error, sizeof(s->error), "player %d's row is layout %d; the search models both players with a v4 checkpoint", p, bopl_layout_of(env, p));
        return -1;
    }
    bopl_policy_eval_step_raw(s->policy, env->observations, s->root_state, s->rows, s->root_next);
    bopl_policy_eval_game_logits(s->policy, s->rows, NULL, s->root_logits);
    s->forwards_run+=2;
    for (int p=0; p<2; p++) s->root_value[p]=s->root_logits[(size_t)p*BOPL_DECODER+BOPL_LOGITS];
    const unsigned char *mask=env->action_mask+(size_t)me*BOPL_LOGITS;
    int latched=env->hold_left[me]>0;
    int run=s->k>1 && (s->search_latched || !latched);
    s->cands=0; s->chosen=-1;
    if (run) {
        s->cands=bopl_search_candidates(s, me, s->k);
        run=s->cands>1;
    }
    if (!run) {
        float a[3];
        const float *row=s->root_logits+(size_t)me*BOPL_DECODER;
        if (s->aim_ar) {
            int combos[2]={0, 0};
            a[1]=(float)bopl_search_sample_head(row+BOPL_ACT_AIM, mask+BOPL_ACT_AIM, BOPL_ACT_BUTTONS, &s->rng);
            combos[me]=(int)a[1];
            bopl_policy_eval_game_logits(s->policy, s->rows, combos, s->logits);
            const float *r=s->logits+(size_t)me*BOPL_DECODER;
            a[0]=(float)bopl_search_sample_head(r, mask, BOPL_ACT_AIM, &s->rng);
            a[2]=(float)bopl_search_sample_head(r+BOPL_ACT_AIM+BOPL_ACT_BUTTONS, mask+BOPL_ACT_AIM+BOPL_ACT_BUTTONS, BOPL_ACT_HOLD, &s->rng);
        } else bopl_search_sample_action(row, mask, &s->rng, a);
        out[0]=(int)a[0]; out[1]=(int)a[1]; out[2]=(int)a[2];
        if (s->cands==1) { out[0]=s->cand[0].aim; out[1]=s->cand[0].buttons; out[2]=s->cand[0].hold; s->chosen=0; }
    } else {
        int alive=s->cands;
        for (;;) {
            if (bopl_search_round(s, me, s->n)<0) return -1;
            if (!s->halving || alive<=1) break;
            /* Keep the better half by mean return. */
            int keep=(alive+1)/2;
            for (int kept=0; kept<keep; kept++) {
                int best=-1;
                for (int c=0; c<s->cands; c++) if (s->cand[c].alive==1 && (best<0 || s->cand[c].sum/s->cand[c].visits>s->cand[best].sum/s->cand[best].visits)) best=c;
                s->cand[best].alive=2;
            }
            for (int c=0; c<s->cands; c++) s->cand[c].alive=s->cand[c].alive==2;
            alive=keep;
            if (alive<=1) break;
        }
        int best=-1;
        for (int c=0; c<s->cands; c++) if (s->cand[c].alive && s->cand[c].visits
            && (best<0 || s->cand[c].sum/s->cand[c].visits>s->cand[best].sum/s->cand[best].visits)) best=c;
        if (best<0) best=0;
        s->chosen=best;
        out[0]=s->cand[best].aim; out[1]=s->cand[best].buttons; out[2]=s->cand[best].hold;
        s->searched++;
    }
    /* The real recurrent states advance. */
    float *swap=s->root_state; s->root_state=s->root_next; s->root_next=swap;
    s->decisions++;
    s->seconds+=bopl_search_now()-t0;
    if (searched) *searched=run;
    return 0;
}

/* Per-candidate rows of BOPL_SEARCH_STAT_WIDTH floats: aim, buttons, hold,
 * visits, mean return, prior log-probability, chosen. Returns the row count. */
static int bopl_search_stats(const BoplSearch *s, float *out, int capacity) {
    if (capacity<s->cands*BOPL_SEARCH_STAT_WIDTH) return -s->cands*BOPL_SEARCH_STAT_WIDTH;
    for (int c=0; c<s->cands; c++) {
        float *row=out+(size_t)c*BOPL_SEARCH_STAT_WIDTH;
        row[0]=(float)s->cand[c].aim; row[1]=(float)s->cand[c].buttons; row[2]=(float)s->cand[c].hold;
        row[3]=(float)s->cand[c].visits; row[4]=s->cand[c].visits?(float)(s->cand[c].sum/s->cand[c].visits):0.0f;
        row[5]=s->cand[c].prior; row[6]=c==s->chosen;
    }
    return s->cands;
}

/* Cost probes (seconds per operation, wall clock). what: 0 fork (reused), 1
 * one four-tick decision of a fork with the live policy's sampled actions
 * (no forward), 2 a batch-2 policy forward on the live rows, 3 a full shadow
 * decision (reset+fork, step, finish, forward), 4 a forward of an evaluator
 * of batch `batch` (per row). */
static double bopl_search_bench(BoplSearch *s, int what, int iterations, int batch) {
    if (iterations<1) return -1;
    BoplPuffer *sh=&s->shadow[0];
    double t0;
    switch (what) {
    case 0: {
        t0=bopl_search_now();
        for (int i=0; i<iterations; i++) { sh->world=bopl_rl_fork(s->env->world, sh->world); if (!sh->world) return -1; }
        return (bopl_search_now()-t0)/iterations;
    }
    case 1: {
        if (bopl_search_shadow_reset(s, sh)) return -1;
        float *v=sh->actions; for (int i=0; i<2*BOPL_NUM_ATNS; i++) v[i]=0;
        t0=bopl_search_now();
        for (int i=0; i<iterations; i++) {
            if (bopl_search_shadow_step(sh)) { if (bopl_search_shadow_reset(s, sh)) return -1; }
        }
        return (bopl_search_now()-t0)/iterations;
    }
    case 2: {
        t0=bopl_search_now();
        for (int i=0; i<iterations; i++) {
            bopl_policy_eval_step_raw(s->policy, s->env->observations, s->root_state, s->rows, s->root_next);
            bopl_policy_eval_game_logits(s->policy, s->rows, NULL, s->logits);
        }
        return (bopl_search_now()-t0)/iterations;
    }
    case 3: {
        float value;
        t0=bopl_search_now();
        for (int i=0; i<iterations; i++) {
            if (bopl_search_shadow_reset(s, sh)) return -1;
            bopl_search_shadow_step(sh);
            bopl_search_forward_shadow(s, sh, s->roll_state, s->roll_next, 0, &value);
        }
        return (bopl_search_now()-t0)/iterations;
    }
    case 4: {
        if (batch<1) return -1;
        void *policy=bopl_policy_eval_create_v4(s->weights, s->weight_count, batch, s->hidden, s->layers, s->aim_fourier, s->aim_ar, s->shape);
        if (!policy) return -1;
        int width=bopl_policy_eval_width(policy);
        float *obs=calloc((size_t)batch*BOPL4_OBS_SIZE, sizeof(float)), *state=calloc((size_t)batch*s->layers*s->hidden, sizeof(float));
        float *next=calloc((size_t)batch*s->layers*s->hidden, sizeof(float)), *rows=calloc((size_t)batch*(width+1), sizeof(float));
        float *logits=calloc((size_t)batch*BOPL_DECODER, sizeof(float));
        if (!obs || !state || !next || !rows || !logits) { free(obs); free(state); free(next); free(rows); free(logits); bopl_policy_eval_close(policy); return -1; }
        for (int b=0; b<batch; b++) memcpy(obs+(size_t)b*BOPL4_OBS_SIZE, s->env->observations, BOPL4_OBS_SIZE*sizeof(float));
        t0=bopl_search_now();
        for (int i=0; i<iterations; i++) {
            bopl_policy_eval_step_raw(policy, obs, state, rows, next);
            bopl_policy_eval_game_logits(policy, rows, NULL, logits);
        }
        double per=(bopl_search_now()-t0)/iterations/batch;
        free(obs); free(state); free(next); free(rows); free(logits); bopl_policy_eval_close(policy);
        return per;
    }
    default: return -1;
    }
}
#endif
