#ifndef BOPL_BOTS_H
#define BOPL_BOTS_H
/* Scripted opponents for the Dash/Bow/Platform loadout (SCRIPTED_BOTS.md).
 *
 * A bot replaces one player's policy: it reads the true current BoplRlState
 * (perfect information, no observation delay - deliberate cheating for a
 * training/evaluation anchor) but acts through exactly the policy interface:
 * one (aim byte, button bits, hold index) decision per four ticks, run through
 * the same bopl_latch, so button holds pin its buttons and an arrow press is a
 * charge that fires when the latch expires (bopl_write_mask's release rule).
 *
 * Game facts the bots rely on (measured in-sim, native/rl/SCRIPTED_BOTS.md):
 *   arrows   kill on contact; launch speed (charge+1)*15 with charge = held
 *            ticks mapped {<32:0, 32:1, 64:4} by the hold head (only holds of
 *            1|2|4, 8 and 16 decisions exist), gravity 40 u/s^2, no drag,
 *            fire ~8-12 ticks after release; bow cooldown 2.5 s from morph
 *            exit; the bow morph hovers (no fall) and drifts with the aim.
 *   dash     13-unit raycast teleport in the aim direction at the press
 *            decision, wall hits attach, exit slide ~7 u; body contact barely
 *            pushes (<0.5 u/s), so dash is mobility, not a weapon; 4 s cooldown.
 *   platform morph freezes the body in place (~72 ticks, hurtbox disabled
 *            after ~24), spawns a platform there and ejects upward ~4.5 u;
 *            6 s cooldown. The freeze is both recovery and a panic shield.
 *   slime    walk 19 u/s, air accel ~3 u/tick capped ~22, jump ~42 u/s with
 *            heavy drag (apex ~7 u, ~26 ticks), per-tick fall
 *            v' = v*(1-1.7/45) - 1.7 (terminal 45); walking off an edge wraps
 *            around the platform's rounded corner and sticks - leaving ground
 *            requires a jump or a dash.
 *   death    water y < -11.3 (non-space maps), blast |x| > 131.25, y > 72.5.
 */
#include "bopl_rl.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
/* The terrain SDF (bot_sdf and the two wall pickers below it) is where the
 * bots spend their time: every raycast, drop probe, foe prediction and shot
 * simulation evaluates the rounded-rect distance of every wall, thousands of
 * times per decision. On x86-64 with AVX-512 those evaluate eight walls per
 * instruction from a SoA mirror of BotView.walls (bot_sdf_avx512); the lane
 * arithmetic is the scalar code's operations in the scalar code's order
 * (sub, mul, add, abs, max, sqrt, no fused multiply-add: fp-contract is off
 * for the kernels), so the vector and scalar paths return the same bits and
 * the frozen v1 fingerprints (SCRIPTED_BOTS.md, Versioning) are unchanged.
 * Other targets, and CPUs without AVX-512, run the scalar loops
 * (-DBOT_SDF_AVX512=0 forces them, to check the two agree). */
#ifndef BOT_SDF_AVX512
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define BOT_SDF_AVX512 1
#else
#define BOT_SDF_AVX512 0
#endif
#endif
#if BOT_SDF_AVX512
#include <immintrin.h>
static int bot_cpu_avx512=-1;
#endif

/* Kind table (SCRIPTED_BOTS.md "Versioning"). Ids, names and behaviour are
 * append-only: kinds 1..BOPL_BOT_V1_KINDS are the frozen v1 personalities
 * whose code paths (everything up to bopl_bot_decide's v1 body) never change
 * again, so scores against them stay comparable across time. An improved or
 * new bot is a NEW kind appended here and to evaluation_protocol.SCRIPTED_KINDS
 * with the same id, named `<family>[-vN]` (the level stays the last token of
 * the public `kind-level` name: mixed-v2-hard, search-hard), and dispatched
 * from bopl_bot_decide_v2 (bopl_bots_v2.h). Training banks pack a kind per
 * nibble of scripted_kinds, so ids stay below 16. */
enum { BOPL_BOT_NONE=0, BOPL_BOT_ARCHER=1, BOPL_BOT_PURSUER=2, BOPL_BOT_EVASIVE=3, BOPL_BOT_MIXED=4, BOPL_BOT_V1_KINDS=4,
       BOPL_BOT_SEARCH=5,  /* rollout search on a forked Rust sim (bopl_bots_v2.h) */
       BOPL_BOT_ARCHER_V2=6, BOPL_BOT_PURSUER_V2=7, BOPL_BOT_EVASIVE_V2=8, BOPL_BOT_MIXED_V2=9,  /* bopl_bots_v2_play.h */
       BOPL_BOT_KINDS=9 };
enum { BOPL_BOT_EASY=0, BOPL_BOT_MEDIUM=1, BOPL_BOT_HARD=2, BOPL_BOT_LEVELS=3 };

/* Per-bot memory of the v2 kinds (bopl_bots_v2_play.h); v1 kinds never read it. */
typedef struct BotV2Mem {
    uint32_t foe_plat_id, my_plat_id;  /* entity ids of the platforms each side spawned last (0 none) */
    double foe_plat_x, foe_plat_y;     /* where the foe froze for its current/last platform morph */
    double haz_prev_dist;              /* my distance to the foe's platform hazard last decision */
    int foe_plat_decisions;            /* consecutive decisions the foe has been in platform morph */
    int foe_bow_idle;                  /* consecutive decisions the foe's bow has been ready unused */
    int foe_charge_decisions;          /* consecutive decisions the foe has been in bow morph */
    int in_hazard;                     /* last decision was spent escaping a platform hazard */
    int combo;                         /* decisions left to complete a dash->arrow attack */
    int hold_off;                      /* pursuer back-off hysteresis (decisions) */
    int since_dash, since_platform;
    int self_still;                    /* search: consecutive decisions I moved < .15 u */
} BotV2Mem;

typedef struct BoplBot {
    int kind, level;
    uint32_t rng;
    /* v2-only inputs, written by c_step before each decision; v1 kinds never
     * read them. world: the live game this bot plays in (bopl_rl_fork clones
     * it for rollouts). applied: every player's input as sent on the previous
     * step (the foe model of a rollout starts from it). */
    BoplRl *world;
    BoplRlAction applied[6];
    /* Lookahead fork owned by a v2 kind (bopl_rl_fork reuse target); survives
     * bopl_bot_reset, closed by the environment (c_close). */
    BoplRl *fork;
    int decisions;
    /* Passivity detector: consecutive decisions the enemy moved < .15 u. */
    double foe_prev_x, foe_prev_y; int foe_still;
    /* Own velocity estimate (units/s) from decision-to-decision displacement:
     * morph bodies report zero velocity while drifting, and the arrow inherits
     * the shooter's self-imposed velocity, so the solver compensates with this. */
    double self_prev_x, self_prev_y; int have_prev;
    /* Mixed-mode hysteresis. */
    int mode, mode_hold;
    /* Reaction latency at lower levels: a threat must persist this many
     * decisions before the bot responds. */
    int threat_seen;
    /* Stalemate / cling bookkeeping: decisions since an arrow press, and
     * since a jump, so a bot stuck on a wall or behind a slab does not
     * jump-spam or idle until sudden death. */
    int since_shot, since_jump;
    uint8_t last[3];
    BotV2Mem v2;
} BoplBot;

/* Indexed by kind; v2 kinds append after "mixed" (bopl_bots_v2.h checks the length). */
static const char *BOPL_BOT_NAMES[]={"none", "archer", "pursuer", "evasive", "mixed", "search",
                                     "archer-v2", "pursuer-v2", "evasive-v2", "mixed-v2"};
/* v2 kinds decide here (bopl_bots_v2.h); the v1 body of bopl_bot_decide is frozen. */
static void bopl_bot_decide_v2(struct BoplBot *bot, const BoplRlState *state, int player_index,
                               int hold_left, uint16_t held, const int hold_run[4], uint8_t out[3]);

/* ---- constants -------------------------------------------------------- */
#define BOT_TICK (1.0/60.0)
#define BOT_ARROW_G 40.0
#define BOT_ARROW_RADIUS 0.61
#define BOT_SLIME_RADIUS 0.76
#define BOT_WATER (-11.3)
#define BOT_BLAST_X 131.25
#define BOT_BLAST_Y 72.5
#define BOT_WALK 19.0
#define BOT_DASH_LENGTH 13.0
#define BOT_SLOT_DASH 0
#define BOT_SLOT_BOW 1
#define BOT_SLOT_PLATFORM 2
#define BOT_JUMP 1u
#define BOT_DASH 2u
#define BOT_ARROW 4u
#define BOT_PLATFORM 8u
/* Hold indices (BOPL_HOLD_LENGTHS {1,2,4,8,16}) and the arrow speed each
 * charge reaches: index<=2 -> 15 u/s, 3 -> 30 u/s, 4 -> 75 u/s. */
#define BOT_HOLD_QUICK 0
#define BOT_HOLD_MID 3
#define BOT_HOLD_FULL 4
static const double BOT_ARROW_SPEED[5]={15, 15, 15, 30, 75};
/* Ticks from the press decision until the arrow leaves, per hold index
 * (latch length * 4 + ~8 measured). */
static const double BOT_FIRE_DELAY[5]={12, 16, 24, 40, 72};

/* ---- small math ------------------------------------------------------- */
typedef struct { double x, y; } BotV2;
static double bot_fix(int64_t v) { return v/4294967296.0; }
static uint32_t bot_rand(BoplBot *bot) {
    uint32_t x=bot->rng; x^=x<<13; x^=x>>17; x^=x<<5; return bot->rng=x;
}
static double bot_unit(BoplBot *bot) { return bot_rand(bot)/4294967296.0; }
static double bot_len(double x, double y) { return sqrt(x*x+y*y); }
static uint8_t bot_aim_byte(double dx, double dy) {
    double a=atan2(dy, dx);
    if (a<0) a+=2*M_PI;
    long b=lround(a/(2*M_PI)*255.0)%255;
    return (uint8_t)(1+b);
}
#define BOT_AIM_UP bot_aim_byte(0, 1)

/* ---- world view ------------------------------------------------------- */
/* Rounded rect: unit axes (rx,ry)/(ux,uy) with core half extents ex/ey; the
 * radius pads the surface around the core (verified against standing slimes).
 * top starts a downward surface probe; spanx is the horizontal reach of the
 * walkable top (tilted slabs walk their length, vertical pillars their cap). */
typedef struct { double cx, cy, rx, ry, ux, uy, ex, ey, radius, top, spanx; int standable; } BotWall;
typedef struct { double x, y, vx, vy; int leaving; } BotArrow;
typedef struct {
    const BoplRlState *state;
    const BoplRlPlayer *me, *foe;
    const BoplRlSlime *my_slime, *foe_slime;
    double x, y, vx, vy;          /* my body (slime or morph) */
    double fx, fy, fvx, fvy;      /* foe body */
    int alive, dead_phase, morphing, charging, grounded, rot_side;
    int foe_alive, foe_morphing, foe_grounded, foe_charging, foe_platforming;
    int dash_ready, bow_ready, platform_ready;
    double bow_timer_left;        /* seconds until the bow is ready (0 if ready) */
    double svx, svy;              /* own velocity estimate (see BoplBot) */
    BotWall walls[48]; int wall_count;
    BotArrow arrows[12]; int arrow_count;
    /* SoA mirror of walls[] (the fields bot_wall_sdf reads) for the vector
     * SDF kernels; filled alongside walls[] in bot_view. */
    struct { double cx[48], cy[48], rx[48], ry[48], ux[48], uy[48], ex[48], ey[48], radius[48], top[48]; } soa;
} BotView;

static const BoplRlSlime *bot_slime(const BoplRlState *state, uint32_t id) {
    for (uint32_t i=0; i<state->header.slime_count; i++)
        if (state->slimes[i].owner==id) return &state->slimes[i];
    return NULL;
}
static const BoplRlEntity *bot_morph_entity(const BoplRlState *state, uint32_t id) {
    for (uint32_t i=0; i<state->header.entity_count; i++) {
        const BoplRlEntity *e=&state->entities[i];
        if ((e->flags&1) && !(e->flags&4) && e->owner==id && e->layer==8) return e;
    }
    return NULL;
}
/* SlimeController.State: 2 fall, 5 ability, 8 nothing, 10 ragdoll; the rest grounded. */
static int bot_state_grounded(uint32_t s) { return s!=2 && s!=5 && s!=8 && s!=10; }

static int bot_view(BotView *view, const BoplRlState *state, int index) {
#if BOT_SDF_AVX512
    if (bot_cpu_avx512<0) bot_cpu_avx512=__builtin_cpu_supports("avx512f")?1:0;
#endif
    /* Only the scalar fields are cleared: walls, arrows and the SoA mirror are
     * read up to their counts, which start at zero. */
    memset(view, 0, offsetof(BotView, walls));
    view->wall_count=0; view->arrow_count=0;
    view->state=state;
    if (!state || (uint32_t)index>=state->header.player_count || state->header.player_count<2) return 0;
    view->me=&state->players[index];
    for (uint32_t i=0; i<state->header.player_count; i++)
        if ((int)i!=index && (state->players[i].flags&1)) { view->foe=&state->players[i]; break; }
    view->my_slime=bot_slime(state, view->me->id);
    view->alive=(view->me->flags&1)!=0;
    if (!view->alive || !view->my_slime) return 0;
    const BoplRlSlime *slime=view->my_slime;
    if (slime->flags&1) {
        view->x=bot_fix(slime->x); view->y=bot_fix(slime->y);
        view->vx=bot_fix(slime->vx); view->vy=bot_fix(slime->vy);
        view->grounded=bot_state_grounded(slime->state);
        view->rot_side=fabs(sin(bot_fix(slime->rotation)))>0.5;
    } else {
        view->charging=(slime->ability_flags[BOT_SLOT_BOW]&8)!=0;
        const BoplRlEntity *body=bot_morph_entity(state, view->me->id);
        if (body) {
            view->morphing=1;
            view->x=bot_fix(body->x); view->y=bot_fix(body->y);
            view->vx=bot_fix(body->vx); view->vy=bot_fix(body->vy);
        } else {
            /* Slime inactive with no morph body: a lost life awaiting respawn
             * (the record persists with its flags cleared). */
            view->dead_phase=1;
            view->x=bot_fix(view->me->x); view->y=bot_fix(view->me->y);
        }
    }
    view->dash_ready=slime->timers[BOT_SLOT_DASH]>slime->cooldowns[BOT_SLOT_DASH];
    view->bow_ready=slime->timers[BOT_SLOT_BOW]>slime->cooldowns[BOT_SLOT_BOW];
    view->platform_ready=slime->timers[BOT_SLOT_PLATFORM]>slime->cooldowns[BOT_SLOT_PLATFORM];
    double left=bot_fix(slime->cooldowns[BOT_SLOT_BOW])-bot_fix(slime->timers[BOT_SLOT_BOW]);
    view->bow_timer_left=left>0?left:0;
    /* The morph body cannot start another ability. */
    if (view->morphing) view->dash_ready=view->bow_ready=view->platform_ready=0;
    if (view->foe) {
        view->foe_alive=1;
        view->foe_slime=bot_slime(state, view->foe->id);
        const BoplRlSlime *foe=view->foe_slime;
        if (foe && (foe->flags&1)) {
            view->fx=bot_fix(foe->x); view->fy=bot_fix(foe->y);
            view->fvx=bot_fix(foe->vx); view->fvy=bot_fix(foe->vy);
            view->foe_grounded=bot_state_grounded(foe->state);
        } else if (foe) {
            view->foe_morphing=1;
            view->foe_charging=(foe->ability_flags[BOT_SLOT_BOW]&8)!=0;
            view->foe_platforming=(foe->ability_flags[BOT_SLOT_PLATFORM]&8)!=0;
            const BoplRlEntity *body=bot_morph_entity(state, view->foe->id);
            if (body) { view->fx=bot_fix(body->x); view->fy=bot_fix(body->y); view->fvx=bot_fix(body->vx); view->fvy=bot_fix(body->vy); }
            else { view->fx=bot_fix(view->foe->x); view->fy=bot_fix(view->foe->y); }
        } else { view->fx=bot_fix(view->foe->x); view->fy=bot_fix(view->foe->y); }
    }
    for (uint32_t i=0; i<state->header.entity_count; i++) {
        const BoplRlEntity *e=&state->entities[i];
        if (!(e->flags&1) || (e->flags&4)) continue;
        /* Terrain: rounded rects on wall/terrain layers (11 wall, 16 lethal, 21 non-lethal). */
        if (e->kind==2 && (e->layer==11 || e->layer==16 || e->layer==21) && view->wall_count<48) {
            BotWall *w=&view->walls[view->wall_count++];
            w->cx=bot_fix(e->x); w->cy=bot_fix(e->y);
            double rx=bot_fix(e->right_x), ry=bot_fix(e->right_y);
            double ux=bot_fix(e->up_x), uy=bot_fix(e->up_y);
            double fex=bot_len(rx, ry), fey=bot_len(ux, uy);
            w->radius=bot_fix(e->radius);
            if (fex>1e-6) { w->rx=rx/fex; w->ry=ry/fex; } else { w->rx=1; w->ry=0; }
            if (fey>1e-6) { w->ux=ux/fey; w->uy=uy/fey; } else { w->ux=-w->ry; w->uy=w->rx; }
            w->ex=fex; w->ey=fey;
            w->top=w->cy+w->ex*fabs(w->ry)+w->ey*fabs(w->uy)+w->radius;
            w->spanx=w->ex*fabs(w->rx)+w->ey*fabs(w->ux);
            /* Every rounded top is standable (slimes stick to curves); steep
             * long slopes still walk poorly, so cap the usable span. */
            w->standable=e->layer!=16 && w->top>BOT_WATER+1.5;
            if (fabs(w->ry)>0.5) w->spanx=fmin(w->spanx, 1.0);
            int k=view->wall_count-1;
            view->soa.cx[k]=w->cx; view->soa.cy[k]=w->cy; view->soa.rx[k]=w->rx; view->soa.ry[k]=w->ry;
            view->soa.ux[k]=w->ux; view->soa.uy[k]=w->uy; view->soa.ex[k]=w->ex; view->soa.ey[k]=w->ey;
            view->soa.radius[k]=w->radius; view->soa.top[k]=w->top;
        }
        /* Live arrows still in flight. Own arrows count too - bots have flown
         * into their own rising mortars. One still adjacent and moving away is
         * just the shot leaving the bow: `leaving` marks it so the passive
         * threat sims skip it (else a point-blank shot dodges itself), while
         * the launch/dash gates still respect it (a jump can outrun it). */
        if (e->kind==0 && e->layer==9 && view->arrow_count<12) {
            double x=bot_fix(e->x), y=bot_fix(e->y);
            double vx=bot_fix(e->vx), vy=bot_fix(e->vy);
            double relx=x-view->x, rely=y-view->y;
            if (bot_len(vx, vy)>2.0) {
                BotArrow *a=&view->arrows[view->arrow_count++];
                a->x=x; a->y=y; a->vx=vx; a->vy=vy;
                a->leaving=e->owner==view->me->id && bot_len(relx, rely)<3.0
                        && relx*vx+rely*vy>0;
            }
        }
    }
    return view->foe_alive;
}

/* Signed distance from a point to a wall's rounded-rect surface. The
 * scalar and eight-lane versions below perform the same IEEE operations in
 * the same order (see the note at the top of the file); no fused multiply-add
 * may be introduced into either. */
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif
static double bot_wall_sdf(const BotWall *w, double x, double y) {
    double px=x-w->cx, py=y-w->cy;
    double lx=px*w->rx+py*w->ry, ly=px*w->ux+py*w->uy;
    double qx=fabs(lx)-w->ex, qy=fabs(ly)-w->ey;
    double ox=qx>0?qx:0, oy=qy>0?qy:0;
    /* fmin(fmax(qx, qy), 0) without the libm calls: qx and qy are finite,
     * and where the tie rules could differ (a +-0 pair) the sum below is
     * +0 either way, so the result bits are the same. */
    double m=qx>=qy?qx:qy;
    double outside=sqrt(ox*ox+oy*oy), inside=m<=0?m:0;
    return outside+inside-w->radius;
}
static double bot_sdf_scalar(const BotView *view, double x, double y) {
    double best=1e9;
    for (int i=0; i<view->wall_count; i++) {
        double d=bot_wall_sdf(&view->walls[i], x, y);
        if (d<best) best=d;
    }
    return best;
}
#if BOT_SDF_AVX512
/* Lanes k..k+7 of the SoA walls (mask m selects the ones that exist). */
__attribute__((target("avx512f"))) static inline __m512d bot_sdf_lanes(const BotView *view, int k, __mmask8 m, __m512d x, __m512d y) {
    __m512d px=_mm512_sub_pd(x, _mm512_maskz_loadu_pd(m, view->soa.cx+k));
    __m512d py=_mm512_sub_pd(y, _mm512_maskz_loadu_pd(m, view->soa.cy+k));
    __m512d lx=_mm512_add_pd(_mm512_mul_pd(px, _mm512_maskz_loadu_pd(m, view->soa.rx+k)),
                             _mm512_mul_pd(py, _mm512_maskz_loadu_pd(m, view->soa.ry+k)));
    __m512d ly=_mm512_add_pd(_mm512_mul_pd(px, _mm512_maskz_loadu_pd(m, view->soa.ux+k)),
                             _mm512_mul_pd(py, _mm512_maskz_loadu_pd(m, view->soa.uy+k)));
    __m512d qx=_mm512_sub_pd(_mm512_abs_pd(lx), _mm512_maskz_loadu_pd(m, view->soa.ex+k));
    __m512d qy=_mm512_sub_pd(_mm512_abs_pd(ly), _mm512_maskz_loadu_pd(m, view->soa.ey+k));
    __m512d zero=_mm512_setzero_pd();
    /* max/min return the second operand on a tie, which only differs from
     * the scalar ?: in the sign of a zero, and outside+inside is +0 then. */
    __m512d ox=_mm512_max_pd(qx, zero), oy=_mm512_max_pd(qy, zero);
    __m512d outside=_mm512_sqrt_pd(_mm512_add_pd(_mm512_mul_pd(ox, ox), _mm512_mul_pd(oy, oy)));
    __m512d inside=_mm512_min_pd(_mm512_max_pd(qx, qy), zero);
    return _mm512_sub_pd(_mm512_add_pd(outside, inside), _mm512_maskz_loadu_pd(m, view->soa.radius+k));
}
__attribute__((target("avx512f"))) static double bot_sdf_avx512(const BotView *view, double x, double y) {
    __m512d best=_mm512_set1_pd(1e9), vx=_mm512_set1_pd(x), vy=_mm512_set1_pd(y);
    for (int k=0; k<view->wall_count; k+=8) {
        int n=view->wall_count-k;
        __mmask8 m=n>=8?(__mmask8)0xFF:(__mmask8)((1u<<n)-1u);
        best=_mm512_mask_min_pd(best, m, best, bot_sdf_lanes(view, k, m, vx, vy));
    }
    return _mm512_reduce_min_pd(best);
}
/* Index of the first wall with the smallest SDF among the lanes `admit`
 * selects (a bit per wall), or -1 when none; *out_d gets its distance. The
 * order rule matches the scalar `if (sd<d)` scan: the first index reaching
 * the minimum wins. */
__attribute__((target("avx512f"))) static int bot_min_wall_avx512(const BotView *view, double x, double y, uint64_t admit, double *out_d) {
    __m512d vx=_mm512_set1_pd(x), vy=_mm512_set1_pd(y);
    double best=1e9; int index=-1;
    for (int k=0; k<view->wall_count; k+=8) {
        int n=view->wall_count-k;
        __mmask8 m=(__mmask8)((n>=8?0xFFu:((1u<<n)-1u))&(unsigned)(admit>>k));
        if (!m) continue;
        __m512d d=bot_sdf_lanes(view, k, m, vx, vy);
        __mmask8 lower=_mm512_mask_cmp_pd_mask(m, d, _mm512_set1_pd(best), _CMP_LT_OQ);
        if (!lower) continue;
        best=_mm512_mask_reduce_min_pd(lower, d);
        __mmask8 hit=_mm512_mask_cmp_pd_mask(lower, d, _mm512_set1_pd(best), _CMP_EQ_OQ);
        index=k+__builtin_ctz((unsigned)hit);
    }
    *out_d=best;
    return index;
}
#endif
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
static double bot_sdf(const BotView *view, double x, double y) {
#if BOT_SDF_AVX512
    if (bot_cpu_avx512>0) return bot_sdf_avx512(view, x, y);
#endif
    return bot_sdf_scalar(view, x, y);
}
/* Sphere-march a ray; returns distance to the first surface (or max). */
static double bot_raycast(const BotView *view, double x, double y, double dx, double dy, double max) {
    double t=0;
    for (int i=0; i<32 && t<max; i++) {
        double d=bot_sdf(view, x+dx*t, y+dy*t);
        if (d<0.05) return t;
        t+=d;
    }
    return max;
}
/* Sphere-tracing skip for the tick simulations below, which test
 * `bot_sdf(p) < limit` at every tick of a path. The SDF is 1-Lipschitz, so
 * after an evaluation `known` at some point, every point within
 * `known - limit` of it has an SDF of at least `limit`, and the test is
 * false without evaluating it. `travel` bounds the path length flown since
 * that evaluation (the sum of |dx|+|dy| per tick is at least the distance);
 * the slack covers the rounding in the SDF and in the bound (both ~1e-12 at
 * arena coordinates). The outcome of every test, and so of every caller,
 * is the one the per-tick evaluation gives. */
typedef struct { double known, travel; } BotSkip;
#define BOT_SKIP_INIT {-1, 0}
static int bot_skip(BotSkip *s, double dx, double dy, double limit) {
    s->travel+=fabs(dx)+fabs(dy);
    return s->known>=0 && s->travel+1e-6<s->known-limit;
}
static double bot_sdf_known(BotSkip *s, const BotView *view, double x, double y) {
    s->known=bot_sdf(view, x, y); s->travel=0;
    return s->known;
}
/* Vertical drop to the nearest surface below (cap 80). */
static double bot_drop(const BotView *view, double x, double y) {
    return bot_raycast(view, x, y, 0, -1, 80);
}
/* Slime-center height standing on a wall's surface at x (walls can be tilted
 * or capsule-capped, so probe the real surface instead of trusting corners). */
static int bot_surface_y(const BotView *view, const BotWall *w, double x, double *out_y) {
    double y=w->top+6;
    /* The probe origin can start inside an overhanging wall: sink below it
     * first, else the march would report a phantom surface in mid-air. */
    double bottom=w->cy-w->ex*fabs(w->ry)-w->ey*fabs(w->uy)-w->radius;
    int guard=0;
    while (bot_sdf(view, x, y)<0.2 && guard++<64) {
        y-=0.5;
        if (y<bottom) return 0;
    }
    double max=y-bottom+1;
    double d=bot_raycast(view, x, y, 0, -1, max);
    if (d>=max) return 0;
    *out_y=y-d+BOT_SLIME_RADIUS;
    return 1;
}

/* ---- prediction ------------------------------------------------------- */
/* Advance a slime's free fall by one tick (measured PlayerPhysics numbers). */
static void bot_fall_tick(double *vy) { *vy=*vy*(1.0-1.7/45.0)-1.7; if (*vy<-45) *vy=-45; }
/* Foe position after `ticks`, holding vx and falling if airborne; stops at terrain. */
static void bot_predict_foe(const BotView *view, double ticks, double *out_x, double *out_y) {
    double x=view->fx, y=view->fy, vx=view->fvx, vy=view->foe_grounded?0:view->fvy;
    if (view->foe_morphing) { *out_x=x; *out_y=y; return; }  /* morphs stay put */
    int grounded=view->foe_grounded;
    BotSkip skip=BOT_SKIP_INIT;
    for (int t=0; t<(int)ticks; t++) {
        if (!grounded) bot_fall_tick(&vy);
        x+=vx*BOT_TICK; y+=vy*BOT_TICK;
        int far=bot_skip(&skip, vx*BOT_TICK, vy*BOT_TICK, BOT_SLIME_RADIUS);
        if (!grounded && vy<0 && !far && bot_sdf_known(&skip, view, x, y)<BOT_SLIME_RADIUS) {
            grounded=1; vx=0; vy=0;
            /* Landed: the remaining ticks add +0 to x and y, which changes
             * neither unless one is -0 (then it would become +0). */
            if (x!=0 && y!=0) break;
        }
    }
    *out_x=x; *out_y=y;
}
/* My landing point if I fall while steering steer (-1/0/+1); 0 when no dry
 * landing. Air control is ~3 u/s per tick capped near walk speed; simulate
 * conservatively so "safe" verdicts stay safe. */
static int bot_landing_steer(const BotView *view, int steer, double *out_x, double *out_y) {
    double x=view->x, y=view->y, vx=view->vx, vy=view->vy;
    for (int t=0; t<240; t++) {
        bot_fall_tick(&vy);
        /* Measured air control ~0.9 u/s per tick; assume less so a "lands"
         * verdict is trustworthy. */
        vx+=steer*0.7; if (vx>19) vx=19; if (vx<-19) vx=-19;
        x+=vx*BOT_TICK; y+=vy*BOT_TICK;
        if (y<BOT_WATER+0.8 || fabs(x)>BOT_BLAST_X-2) return 0;
        /* Demand real penetration: a graze within centimeters of the surface
         * radius is a miss under the real (stronger-falling) physics. */
        if (vy<0 && bot_sdf(view, x, y)<BOT_SLIME_RADIUS-0.25) { *out_x=x; *out_y=y; return 1; }
    }
    return 0;
}

/* Ballistic solve: direction and flight seconds to hit (tx,ty) from (px,py)
 * at speed s under arrow gravity. Picks the flat solution. */
static int bot_ballistic_arc(double px, double py, double s, double tx, double ty,
                             int high, double *dir_x, double *dir_y, double *seconds) {
    double dx=tx-px, dy=ty-py, g=BOT_ARROW_G;
    double a=g*g/4.0, b=g*dy-s*s, c=dx*dx+dy*dy;
    double disc=b*b-4*a*c;
    if (disc<0) return 0;
    double T=high?(-b+sqrt(disc))/(2*a):(-b-sqrt(disc))/(2*a);
    if (T<=0) T=(-b+sqrt(disc))/(2*a);
    if (T<=0) return 0;
    double t=sqrt(T);
    *dir_x=dx/(s*t); *dir_y=(dy+g*T/2.0)/(s*t); *seconds=t;
    return 1;
}
static int bot_ballistic(double px, double py, double s, double tx, double ty,
                         double *dir_x, double *dir_y, double *seconds) {
    return bot_ballistic_arc(px, py, s, tx, ty, 0, dir_x, dir_y, seconds);
}
/* As above but the arrow inherits the shooter's velocity (measured: launch
 * velocity = dir*speed + self velocity): aim at the target displaced by
 * -v_self*t, iterating on the flight time. */
static int bot_ballistic_moving(double px, double py, double svx, double svy, double s, int high,
                                double tx, double ty, double *dir_x, double *dir_y, double *seconds) {
    double t=bot_len(tx-px, ty-py)/s;
    for (int i=0; i<4; i++)
        if (!bot_ballistic_arc(px, py, s, tx-svx*t, ty-svy*t, high, dir_x, dir_y, &t)) return 0;
    *seconds=t;
    return 1;
}
/* Arrow path clear of terrain up to the target vicinity? Grazing a platform
 * lip is the main systematic miss, so terrain clearance is generous until the
 * final approach. */
static int bot_closest_wall(const BotView *view, double x, double y) {
    int best=-1; double d=1e9;
#if BOT_SDF_AVX512
    if (bot_cpu_avx512>0) return bot_min_wall_avx512(view, x, y, ~0ull, &d);
#endif
    for (int i=0; i<view->wall_count; i++) {
        double sd=bot_wall_sdf(&view->walls[i], x, y);
        if (sd<d) { d=sd; best=i; }
    }
    return best;
}
static int bot_support_wall(const BotView *view, double x, double y) {
    int best=-1; double d=1e9;
#if BOT_SDF_AVX512
    if (bot_cpu_avx512>0) {
        uint64_t admit=0;
        /* Overhead slabs are cover, not a floor. */
        for (int i=0; i<view->wall_count; i++) if (!(view->soa.top[i]>y+1.2)) admit|=1ull<<i;
        best=bot_min_wall_avx512(view, x, y, admit, &d);
        return (best>=0 && d<2.5)?best:-1;
    }
#endif
    for (int i=0; i<view->wall_count; i++) {
        const BotWall *w=&view->walls[i];
        /* Overhead slabs are cover, not a floor. */
        if (w->top>y+1.2) continue;
        double sd=bot_wall_sdf(w, x, y);
        if (sd<d) { d=sd; best=i; }
    }
    return (best>=0 && d<2.5)?best:-1;
}
static int bot_shot_clear(const BotView *view, double px, double py, double vx0, double vy0,
                          double seconds, double tx, double ty) {
    double x=px, y=py, vx=vx0, vy=vy0;
    int ticks=(int)(seconds*60)+2;
    int home=bot_support_wall(view, px, py);
    const BotWall *hw=home>=0?&view->walls[home]:NULL;
    BotSkip skip=BOT_SKIP_INIT;
    for (int t=0; t<ticks; t++) {
        vy-=BOT_ARROW_G*BOT_TICK;
        x+=vx*BOT_TICK; y+=vy*BOT_TICK;
        if (y>BOT_BLAST_Y-0.5 || y<BOT_WATER || fabs(x)>BOT_BLAST_X) return 0;
        /* to_target<1.1 (bot_len(tx-x, ty-y)<1.1): the square root is only
         * taken where the squared distance leaves the verdict open. */
        double qx=tx-x, qy=ty-y, q=qx*qx+qy*qy;
        if (q<1.3 && bot_len(qx, qy)<1.1) return 1;
        if (bot_skip(&skip, vx*BOT_TICK, vy*BOT_TICK, BOT_ARROW_RADIUS+0.2)) continue;
        double d=bot_sdf_known(&skip, view, x, y);
        /* Both margins are at most BOT_ARROW_RADIUS+0.2, so d below that is
         * the only case where which margin applies can matter. */
        if (d>=BOT_ARROW_RADIUS+0.2) continue;
        double to_target=bot_len(tx-x, ty-y), from_start=bot_len(x-px, y-py);
        double margin=(to_target<2.5 || from_start<3.0)?BOT_ARROW_RADIUS*0.7:BOT_ARROW_RADIUS+0.2;
        if (d<margin) {
            /* The deck underfoot overlaps the slime center, so the first
             * ticks of a shot along it would otherwise collide at t≈0. Ignore
             * the home wall only while the arrow is still on/above that deck;
             * a downward shot that drops through the floor is a real hit.
             * Skipping until near the target (the old rule) certified shots
             * that stuck in the floor in-game. */
            if (hw && bot_closest_wall(view, x, y)==home && y>=hw->top-0.08) continue;
            return 0;
        }
    }
    return 1;
}
/* Would this arrow cross back through my own post-release path before it
 * reaches the target? The bow morph hovers while charging, then the slime
 * resumes free fall - an airborne bot firing a steep upward shot drops
 * straight into its own rising arrow (fatal: owner immunity is unreliable). */
static int bot_self_cross(const BotView *view, double ux, double uy, double flight) {
    double ax=view->x, ay=view->y, avx=ux, avy=uy;
    double my=view->y, mvy=0;
    int separated=0, ticks=(int)(flight*60)-3;
    if (ticks>60) ticks=60;
    for (int t=1; t<=ticks; t++) {
        avy-=BOT_ARROW_G*BOT_TICK; ax+=avx*BOT_TICK; ay+=avy*BOT_TICK;
        if (!view->grounded) {
            bot_fall_tick(&mvy);
            my+=mvy*BOT_TICK;
            if (mvy<0 && bot_sdf(view, view->x, my)<BOT_SLIME_RADIUS) mvy=0;
        }
        double d=bot_len(ax-view->x, ay-my);
        if (d>3.2) separated=1;
        else if (separated && d<2.2) return 1;
    }
    return 0;
}
static int bot_solve_shot(const BotView *view, int hold, double lead_ticks, int allow_high,
                          int ignore_terrain, uint8_t *aim, double *flight) {
    double s=BOT_ARROW_SPEED[hold];
    /* Entering the bow morph discards the slime's velocity (measured), but the
     * morph's own hover drift persists into the arrow: compensate only that. */
    double svx=view->morphing?view->svx:0, svy=view->morphing?view->svy:0;
    for (int high=0; high<=(allow_high?1:0); high++) {
        double tx, ty, ux=0, uy=0, t=bot_len(view->fx-view->x, view->fy-view->y)/s;
        int ok=1;
        for (int i=0; i<4 && ok; i++) {
            bot_predict_foe(view, lead_ticks+t*60, &tx, &ty);
            /* A grounded slime's center sits at its radius; bias the aim point
             * up so flat shots clear the platform lip under the target. */
            double ay=ty+(view->foe_grounded?0.35:0);
            ok=bot_ballistic_moving(view->x, view->y, svx, svy, s, high, tx, ay, &ux, &uy, &t);
            ty=ay;
        }
        if (!ok) continue;
        /* Mortar lobs fly long; skip ones that hang too long to trust. Sky
         * mortars that peak in the blast zone never land (shot_clear). */
        if (high && t>4.0) continue;
        if (!ignore_terrain
                && !bot_shot_clear(view, view->x, view->y, ux*s+svx, uy*s+svy, t, tx, ty)) continue;
        if (bot_self_cross(view, ux*s+svx, uy*s+svy, t)) continue;
        *aim=bot_aim_byte(ux, uy); *flight=t;
        return 1;
    }
    return 0;
}
/* A standable spot from which some charge has a clear ballistic path to the
 * foe. Searches map and player platforms; my own next platform morph also
 * lands here later. Returns 1 with the spot; prefers near me, not too near
 * the foe, and slightly higher ground. */
static int bot_dash_reaches(const BotView *view, double tx, double ty);
static int bot_nearest_perch(const BotView *view, double x, double y, double *px, double *py);
/* A drop compared against a threshold below `cap` needs no march past the
 * cap: the march up to it is the same, and everything the 80-unit probe
 * could still return is on the same side of the threshold as the cap. */
static double bot_drop_capped(const BotView *view, double x, double y, double cap) {
    return bot_raycast(view, x, y, 0, -1, cap);
}
static int bot_deck_walkable(const BotView *view, double tx, double ty) {
    if (fabs(ty-view->y)>3.2) return 0;
    double step=tx>=view->x?2.0:-2.0;
    for (double x=view->x; (step>0)?(x<tx):(x>tx); x+=step)
        if (bot_drop_capped(view, x, view->y+0.6, 6.0)>5.0) return 0;
    return bot_drop_capped(view, tx, view->y+0.6, 6.0)<5.0;
}
/* The dash ray from (ox,oy) toward a target, as bot_dash_reaches_from and
 * bot_dash_hits_perch both cast it: the unit direction and distance, and the
 * raycast along it computed at most once between the two. Targets farther
 * than the dash length plus a verdict's tolerance get that verdict without
 * the cast: the dash ends within BOT_DASH_LENGTH of its origin (the ray
 * direction is a unit vector to rounding, hence the 0.1 slack), so its end
 * point cannot come within the tolerance of the target. */
typedef struct { double ox, oy, dx, dy, dist, hit; int cast; } BotDashRay;
static void bot_dash_ray(double ox, double oy, double tx, double ty, BotDashRay *r) {
    r->ox=ox; r->oy=oy; r->dx=tx-ox; r->dy=ty-oy; r->dist=bot_len(r->dx, r->dy); r->cast=0; r->hit=0;
    if (r->dist<1e-6) return;
    r->dx/=r->dist; r->dy/=r->dist;
}
static double bot_dash_ray_hit(const BotView *view, BotDashRay *r) {
    if (!r->cast) { r->hit=bot_raycast(view, r->ox, r->oy, r->dx, r->dy, BOT_DASH_LENGTH); r->cast=1; }
    return r->hit;
}
static int bot_dash_reaches_ray(const BotView *view, double tx, double ty, BotDashRay *r) {
    if (r->dist<1e-6) return 1;
    if (r->dist>BOT_DASH_LENGTH+3.5+0.1) return 0;
    double hit=bot_dash_ray_hit(view, r);
    double end=fmin(hit, BOT_DASH_LENGTH);
    return bot_len(tx-(r->ox+r->dx*end), ty-(r->oy+r->dy*end))<3.5;
}
static int bot_dash_hits_perch_ray(const BotView *view, double tx, double ty, BotDashRay *r) {
    if (r->dist<1e-6) return 1;
    if (r->dist>BOT_DASH_LENGTH+8.0+0.1) return 0;
    double hit=bot_dash_ray_hit(view, r);
    if (hit>=BOT_DASH_LENGTH) return 0;
    double hx=r->ox+r->dx*hit, hy=r->oy+r->dy*hit;
    /* Hitting the underside of the target pancake is how pursuers clung
     * under map-1's top slab for two minutes. */
    if (hy<ty-3.5) return 0;
    return bot_len(tx-hx, ty-hy)<8.0;
}
static int bot_dash_hits_perch(const BotView *view, double tx, double ty) {
    BotDashRay r; bot_dash_ray(view->x, view->y, tx, ty, &r);
    return bot_dash_hits_perch_ray(view, tx, ty, &r);
}
/* bot_dash_reaches(view, tx, ty) || bot_dash_hits_perch(view, tx, ty), one cast. */
static int bot_dash_reaches_or_hits_perch(const BotView *view, double tx, double ty) {
    BotDashRay r; bot_dash_ray(view->x, view->y, tx, ty, &r);
    return bot_dash_reaches_ray(view, tx, ty, &r) || bot_dash_hits_perch_ray(view, tx, ty, &r);
}
static int bot_can_reach(const BotView *view, double px, double py) {
    double dx=px-view->x, dy=py-view->y;
    if (bot_len(dx, dy)<2.2) return 1;
    if (bot_deck_walkable(view, px, py)) return 1;
    if (bot_dash_reaches_or_hits_perch(view, px, py)) return 1;
    /* A jump reaches ~7 u up and ~7-10 u across. */
    if (dy>1.5 && dy<8.0 && fabs(dx)<12.0 && bot_drop_capped(view, px, py+0.6, 6.0)<6.0) return 1;
    return 0;
}
static int bot_firing_spot(const BotView *view, double *sx, double *sy) {
    double tx=view->fx, ty=view->fy+(view->foe_grounded?0.35:0);
    double best=1e18; int found=0;
    static const int holds[2]={BOT_HOLD_MID, BOT_HOLD_FULL};
    for (int i=0; i<view->wall_count; i++) {
        const BotWall *w=&view->walls[i];
        if (!w->standable) continue;
        if (w->spanx+w->radius<4.0 && bot_len(w->cx-view->fx, w->top-view->fy)>6) continue;
        /* Include the rounded cap: the old spanx-0.8 samples sat inboard of
         * the lip, and only the lip clears a downward shot off a pancake. */
        double span=fmax(w->spanx+w->radius*0.35, 0);
        for (int k=-4; k<=4; k++) {
            double px=w->cx+(span*k)/4.0, py;
            if (!bot_surface_y(view, w, px, &py)) continue;
            double range=bot_len(tx-px, ty-py);
            if (range<2) continue;
            if (!bot_can_reach(view, px, py)) continue;
            int cleared=0;
            for (int h=0; h<2 && !cleared; h++) {
                double s=BOT_ARROW_SPEED[holds[h]], ux, uy, t;
                if (s*s/BOT_ARROW_G<=range*0.9) continue;
                for (int high=0; high<2 && !cleared; high++) {
                    if (!bot_ballistic_arc(px, py, s, tx, ty, high, &ux, &uy, &t)) continue;
                    if (high && t>4.0) continue;
                    cleared=bot_shot_clear(view, px, py, ux*s, uy*s, t, tx, ty);
                }
            }
            if (cleared) {
                double score=bot_len(px-view->x, py-view->y)
                             +(range<9?2*(9-range):0)-(py-view->y>0?2:0);
                if (score<best) { best=score; *sx=px; *sy=py; found=1; }
            }
        }
    }
    return found;
}

/* ---- threats ---------------------------------------------------------- */
/* Earliest predicted arrow impact on me (ticks); 1e9 when clear. */
static double bot_arrow_eta(const BotView *view, double *threat_dir_x) {
    double best=1e9;
    for (int i=0; i<view->arrow_count; i++) {
        if (view->arrows[i].leaving) continue;
        double x=view->arrows[i].x, y=view->arrows[i].y;
        double vx=view->arrows[i].vx, vy=view->arrows[i].vy;
        double mx=view->x, my=view->y, mvx=view->vx, mvy=view->grounded?0:view->vy;
        int my_grounded=view->grounded;
        for (int t=1; t<=60; t++) {
            vy-=BOT_ARROW_G*BOT_TICK; x+=vx*BOT_TICK; y+=vy*BOT_TICK;
            if (!my_grounded) { bot_fall_tick(&mvy); }
            mx+=mvx*BOT_TICK; my+=mvy*BOT_TICK;
            if (!my_grounded && mvy<0 && bot_sdf(view, mx, my)<BOT_SLIME_RADIUS) { my_grounded=1; mvx=0; mvy=0; }
            if (bot_sdf(view, x, y)<BOT_ARROW_RADIUS*0.5) break;  /* hits terrain first */
            if (bot_len(x-mx, y-my)<1.9) {
                if (t<best) { best=t; if (threat_dir_x) *threat_dir_x=vx>=0?1:-1; }
                break;
            }
        }
    }
    return best;
}

/* Would falling while steering steer fly me into an arrow (mine included)?
 * Complements bot_arrow_eta, which extrapolates my current velocity and so
 * cannot see a collision my own intended steering is about to cause. */
static int bot_steer_hits_arrow(const BotView *view, int steer) {
    for (int i=0; i<view->arrow_count; i++) {
        if (view->arrows[i].leaving) continue;
        double ax=view->arrows[i].x, ay=view->arrows[i].y;
        double avx=view->arrows[i].vx, avy=view->arrows[i].vy;
        double x=view->x, y=view->y, vx=view->vx, vy=view->vy;
        for (int t=0; t<50; t++) {
            avy-=BOT_ARROW_G*BOT_TICK; ax+=avx*BOT_TICK; ay+=avy*BOT_TICK;
            bot_fall_tick(&vy);
            vx+=steer*0.9; if (vx>19) vx=19; if (vx<-19) vx=-19;
            x+=vx*BOT_TICK; y+=vy*BOT_TICK;
            if (bot_len(ax-x, ay-y)<2.1) return 1;
            if (bot_sdf(view, ax, ay)<BOT_ARROW_RADIUS*0.5) break;
        }
    }
    return 0;
}

/* Would launching upward from the ground (slime jump or platform ejection,
 * both ~58 u/s initial rise) fly me into an arrow? Checks ALL arrows,
 * including a `leaving` one: a launch catches up with an arrow that a
 * standing bot never would (this exact chase has killed an evasive bot). */
static int bot_launch_hits_arrow(const BotView *view, double vx) {
    for (int i=0; i<view->arrow_count; i++) {
        double ax=view->arrows[i].x, ay=view->arrows[i].y;
        double avx=view->arrows[i].vx, avy=view->arrows[i].vy;
        double x=view->x, y=view->y, mvx=vx, mvy=58;
        for (int t=1; t<=55; t++) {
            avy-=BOT_ARROW_G*BOT_TICK; ax+=avx*BOT_TICK; ay+=avy*BOT_TICK;
            bot_fall_tick(&mvy);
            x+=mvx*BOT_TICK; y+=mvy*BOT_TICK;
            if (bot_len(ax-x, ay-y)<2.4) return 1;
            if (bot_sdf(view, ax, ay)<BOT_ARROW_RADIUS*0.5) break;
        }
    }
    return 0;
}

/* Can a launch (jump/ejection, ~58 u/s rise) with horizontal outcome vx0
 * steer to a dry landing? Slope jumps launch along the surface normal, so
 * the actual vx0 is unknowable at the press; callers try several. Mirrors
 * bot_landing_steer's conservative physics. */
static int bot_launch_lands(const BotView *view, double vx0) {
    for (int steer=-1; steer<=1; steer++) {
        double x=view->x, y=view->y, vx=vx0, vy=58;
        int dead=0, landed=0;
        for (int t=0; t<300 && !dead && !landed; t++) {
            bot_fall_tick(&vy);
            vx+=steer*0.7; if (vx>19) vx=19; if (vx<-19) vx=-19;
            x+=vx*BOT_TICK; y+=vy*BOT_TICK;
            if (y<BOT_WATER+0.8 || fabs(x)>BOT_BLAST_X-2) dead=1;
            else if (vy<0 && bot_sdf(view, x, y)<BOT_SLIME_RADIUS-0.25) landed=1;
        }
        if (landed) return 1;
    }
    return 0;
}

/* ---- movement --------------------------------------------------------- */
/* Is a dash toward (dx,dy) survivable? Either it hits terrain (attaches) or
 * its endpoint plus slide has dry ground below. */
static int bot_dash_safe(const BotView *view, double dx, double dy) {
    double n=bot_len(dx, dy); if (n<1e-6) return 0;
    dx/=n; dy/=n;
    double hit=bot_raycast(view, view->x, view->y, dx, dy, BOT_DASH_LENGTH);
    /* Never dash through an arrow's path (a pursuer once dashed at the foe
     * straight into its own rising mortar). Dash covers ~13u in ~0.25s. */
    for (int i=0; i<view->arrow_count; i++) {
        double ax=view->arrows[i].x, ay=view->arrows[i].y;
        double avx=view->arrows[i].vx, avy=view->arrows[i].vy;
        double x=view->x, y=view->y, reach=fmin(hit, BOT_DASH_LENGTH);
        for (int t=1; t<=18; t++) {
            avy-=BOT_ARROW_G*BOT_TICK; ax+=avx*BOT_TICK; ay+=avy*BOT_TICK;
            double along=fmin(55.0*t*BOT_TICK, reach);
            x=view->x+dx*along; y=view->y+dy*along;
            if (bot_len(ax-x, ay-y)<2.4) return 0;
        }
    }
    if (hit<BOT_DASH_LENGTH) {
        double hx=view->x+dx*hit, hy=view->y+dy*hit;
        /* Standing slimes that dash into a pancake underside cling and
         * never reach the top (map-1 pursuer: 28 dashes, 0 arrows). */
        if (!view->rot_side) {
            double px=hx, py=hy;
            if (bot_nearest_perch(view, hx, hy+8, &px, &py)>=0 && hy<py-3.0) return 0;
        }
        return 1;
    }
    double ex=view->x+dx*(BOT_DASH_LENGTH+5), ey=view->y+dy*(BOT_DASH_LENGTH+5);
    if (ey<BOT_WATER+1.5 || fabs(ex)>BOT_BLAST_X-10) return 0;
    double drop=bot_drop(view, ex, ey);
    return ey-drop>BOT_WATER || drop<12;
}
/* Does a dash from (ox,oy) actually end near (tx,ty)? A dash that attaches to
 * a wall far short of the target (e.g. the underside of the foe's platform)
 * is a wasted cooldown. */
static int bot_dash_reaches_from(const BotView *view, double ox, double oy, double tx, double ty) {
    BotDashRay r; bot_dash_ray(ox, oy, tx, ty, &r);
    return bot_dash_reaches_ray(view, tx, ty, &r);
}
static int bot_dash_reaches(const BotView *view, double tx, double ty) {
    return bot_dash_reaches_from(view, view->x, view->y, tx, ty);
}
/* One-level route search: a standable spot I can plausibly get to by
 * walk/jump (or a dash) from which the target is one dash or jump away. */
static int bot_waypoint(const BotView *view, double tx, double ty, double *wx, double *wy) {
    int found=0; double best=1e18;
    for (int i=0; i<view->wall_count; i++) {
        const BotWall *w=&view->walls[i];
        if (!w->standable) continue;
        double span=fmax(w->spanx-0.8, 0);
        for (int side=-1; side<=1; side++) {
            double px=w->cx+side*span, py;
            if (!bot_surface_y(view, w, px, &py)) continue;
            if (bot_len(px-view->x, py-view->y)<1.5) continue;   /* here already */
            if (bot_len(px-tx, py-ty)<2.0) continue;             /* the target itself */
            /* Enables the goal: one dash lands by the target, or it is a jump away. */
            int enables=bot_dash_reaches_from(view, px, py, tx, ty)
                        || (py>ty-6.0 && fabs(px-tx)<7);
            if (!enables) continue;
            /* Plausibly reachable from here without further routing. */
            int reachable=py-view->y<8.0 || bot_dash_reaches_or_hits_perch(view, px, py);
            if (!reachable) continue;
            double score=bot_len(px-view->x, py-view->y)+0.5*bot_len(px-tx, py-ty);
            if (score<best) { best=score; *wx=px; *wy=py; found=1; }
        }
    }
    return found;
}
/* Nearest standable top point; returns wall index or -1. */
static int bot_nearest_perch(const BotView *view, double x, double y, double *px, double *py) {
    int best=-1; double best_d=1e9;
    for (int i=0; i<view->wall_count; i++) {
        const BotWall *w=&view->walls[i];
        if (!w->standable) continue;
        double cx=w->cx, span=fmax(w->spanx-0.8, 0);
        double tx=fmin(fmax(x, cx-span), cx+span), ty;
        if (!bot_surface_y(view, w, tx, &ty)) continue;
        double d=bot_len(tx-x, ty-y);
        /* Prefer perches at or above me: falling past a low perch is fatal. */
        if (ty<y-2) d+=8;
        if (d<best_d) { best_d=d; best=i; *px=tx; *py=ty; }
    }
    return best;
}
static int bot_foe_deck(const BotView *view, double *px, double *py) {
    int best=-1; double best_d=1e9;
    for (int i=0; i<view->wall_count; i++) {
        const BotWall *w=&view->walls[i];
        if (!w->standable) continue;
        double span=fmax(w->spanx-0.8, 0);
        double tx=fmin(fmax(view->fx, w->cx-span), w->cx+span), ty;
        if (!bot_surface_y(view, w, tx, &ty)) continue;
        double d=bot_len(tx-view->fx, ty-view->fy);
        if (d<best_d) { best_d=d; best=i; *px=tx; *py=ty; }
    }
    return best;
}
static int bot_highest_perch(const BotView *view, double *px, double *py) {
    int found=0; double best=-1e9;
    for (int i=0; i<view->wall_count; i++) {
        const BotWall *w=&view->walls[i];
        if (!w->standable) continue;
        double tx=w->cx, ty;
        if (!bot_surface_y(view, w, tx, &ty)) continue;
        if (ty>best) { best=ty; *px=tx; *py=ty; found=1; }
    }
    return found;
}
/* Walk/jump/dash toward (tx,ty). Fills aim and may set button bits. */
static void bot_nav(const BotView *view, BoplBot *bot, double tx, double ty,
                    uint8_t *aim, uint16_t *buttons) {
    double dx=tx-view->x, dy=ty-view->y;
    if (!view->grounded) { *aim=bot_aim_byte(dx>=0?1:-1, 0); return; }
    /* Clinging to a side or ceiling: dash up to the standable top. Jumping
     * every decision (the old behaviour) kept an archer on its own spawned
     * platform for two minutes. */
    if (view->rot_side) {
        double px=view->x, py=view->y;
        bot_nearest_perch(view, view->x, view->y+10, &px, &py);
        double ux=px-view->x, uy=py-view->y;
        if (view->dash_ready && bot_dash_safe(view, ux, uy+1.5))
            { *buttons|=BOT_DASH; *aim=bot_aim_byte(ux, uy+1.5); return; }
        if (view->dash_ready && bot_dash_safe(view, dx, dy))
            { *buttons|=BOT_DASH; *aim=bot_aim_byte(dx, dy); return; }
        if (view->platform_ready)
            { *buttons|=BOT_PLATFORM; *aim=BOT_AIM_UP; return; }
        *aim=bot_aim_byte(ux>=0?0.15:-0.15, 1);
        if (bot->since_jump>=8) *buttons|=BOT_JUMP;
        return;
    }
    /* Tiny stub: dash off toward the foe's deck instead of walking in a 1u loop. */
    {   int home=bot_support_wall(view, view->x, view->y);
        if (home>=0 && view->walls[home].spanx+view->walls[home].radius<4.0) {
            double px=tx, py=ty;
            bot_foe_deck(view, &px, &py);
            double ux=px-view->x, uy=py-view->y;
            if (view->dash_ready && bot_dash_safe(view, ux, uy))
                { *buttons|=BOT_DASH; *aim=bot_aim_byte(ux, uy); return; }
            *aim=bot_aim_byte(ux>=0?1:-1, 0.2);
            if (bot->since_jump>=4) *buttons|=BOT_JUMP;
            return;
        }
    }
    if (fabs(dx)<0.9 && fabs(dy)<3) { *aim=BOT_AIM_UP; return; }  /* arrived: stand */
    /* Stacked above the target: walk to the nearer lip and drop. Always
     * the nearer lip, never "toward the foe" — that flipped at |dx|≈3 and
     * walked a 1u loop until sudden death. */
    if (dy<-4.0) {
        int to_edge=1;
        int home=bot_support_wall(view, view->x, view->y);
        if (home>=0) {
            const BotWall *w=&view->walls[home];
            to_edge=(view->x - (w->cx-w->spanx) < (w->cx+w->spanx)-view->x)?-1:1;
        }
        double ahead=bot_drop(view, view->x+to_edge*2.2, view->y+0.5);
        if (ahead>3.5) {
            double lx, ly;
            if (bot_landing_steer(view, to_edge, &lx, &ly) && ly>BOT_WATER+1.5)
                { *buttons|=BOT_JUMP; *aim=bot_aim_byte(to_edge, -0.2); return; }
            if (view->dash_ready && bot_dash_safe(view, to_edge*0.2, -1))
                { *buttons|=BOT_DASH; *aim=bot_aim_byte(to_edge, -1); return; }
            *buttons|=BOT_JUMP; *aim=bot_aim_byte(to_edge, -0.2); return;
        }
        *aim=bot_aim_byte(to_edge, 0); return;
    }
    /* A jump reaches ~7 units; higher targets need the dash (13 units), the
     * platform morph's ejection, or a route through an intermediate perch.
     * Do not require the target to be nearly overhead: map-1's high pancake
     * is ~20 u across and 10 u up, and the old |dx|<dy+9 gate walked under. */
    for (int depth=0; depth<2 && dy>4.5; depth++) {
        if (bot_dash_reaches_or_hits_perch(view, tx, ty)) {
            if (view->dash_ready && bot_dash_safe(view, dx, dy))
                { *buttons|=BOT_DASH; *aim=bot_aim_byte(dx, dy); return; }
        }
        /* At the climb angle: spend a resource. Walking with dx≈0 used
         * sign(0)=+1 then sign(−)=−1, a 0.5u loop until cooldown. */
        if (fabs(dx)<2.5 && dy>5.0) {
            double overhead=bot_raycast(view, view->x, view->y, 0, 1, 12);
            if (view->dash_ready && bot_dash_safe(view, dx, dy))
                { *buttons|=BOT_DASH; *aim=bot_aim_byte(dx, dy); return; }
            if (view->platform_ready && overhead>6.0)
                { *buttons|=BOT_PLATFORM; *aim=BOT_AIM_UP; return; }
            *aim=bot_aim_byte(dx>=0?0.4:-0.4, 1);
            if (bot->since_jump>=6) *buttons|=BOT_JUMP;
            return;
        }
        if (dy>6 && dy<18 && fabs(view->fx-view->x)>=4 && view->platform_ready
                && bot_raycast(view, view->x, view->y, 0, 1, 12)>6.0)
            { *buttons|=BOT_PLATFORM; *aim=BOT_AIM_UP; return; }
        double wx_, wy_;
        if (depth==0 && bot_waypoint(view, tx, ty, &wx_, &wy_)) {
            tx=wx_; ty=wy_; dx=tx-view->x; dy=ty-view->y;
            if (fabs(dx)<0.9 && fabs(dy)<3) { *aim=BOT_AIM_UP; return; }
            continue;
        }
        break;
    }
    double step=dx>=0?1:-1;
    /* Ahead of my feet: a gap means this walk would wrap around the edge. */
    double ahead=bot_drop(view, view->x+step*2.2, view->y+0.5);
    int gap=ahead>4.5;
    int target_above=dy>2.5 && dy<7 && fabs(dx)<10;
    if (gap) {
        /* Foe / target is below this slab: drop if the landing is dry. */
        if (dy<-3.0) {
            double lx, ly;
            if (bot_landing_steer(view, (int)step, &lx, &ly) && ly>BOT_WATER+1.5)
                { *buttons|=BOT_JUMP; *aim=bot_aim_byte(step, -0.15); return; }
            if (view->dash_ready && bot_dash_safe(view, step*0.2, -1))
                { *buttons|=BOT_DASH; *aim=bot_aim_byte(step, -1); return; }
        }
        /* Only commit if the far side is landable: a jump carries ~7 units,
         * a dash ~18. Otherwise hold the edge (the layers above shoot). */
        double jump_far=bot_drop(view, view->x+step*7, view->y+2);
        if (jump_far<9) { *buttons|=BOT_JUMP; *aim=bot_aim_byte(step, 0.3); return; }
        /* Island hop: a dash aimed at the target (not just along the ground)
         * is how two same-height pancakes get bridged. Standing at the lip
         * was the map-0 camp. */
        if (view->dash_ready && bot_dash_safe(view, dx, dy))
            { *buttons|=BOT_DASH; *aim=bot_aim_byte(dx, fmax(dy, 0.15)); return; }
        if (view->dash_ready && fabs(dx)>6 && bot_dash_safe(view, step, 0.05))
            { *buttons|=BOT_DASH; *aim=bot_aim_byte(step, 0.05); return; }
        *aim=bot_aim_byte(step, 0.2);
        if (bot->since_jump>=6) *buttons|=BOT_JUMP;
        return;
    }
    if (target_above) { *buttons|=BOT_JUMP; *aim=bot_aim_byte(step, 0.3); return; }
    (void)bot;
    *aim=bot_aim_byte(step, 0);
}

/* ---- decision layers --------------------------------------------------- */
typedef struct { uint8_t aim; uint16_t buttons; uint8_t hold; int done; } BotAction;

static void bot_act(BotAction *act, uint8_t aim, uint16_t buttons, uint8_t hold) {
    act->aim=aim; act->buttons=buttons; act->hold=hold; act->done=1;
}

/* Recovery: airborne and no steering choice lands. Spend a resource. */
static void bot_recover(const BotView *view, BoplBot *bot, int can_arrow, BotAction *act) {
    double px=view->x, py=view->y;
    int perch=bot_nearest_perch(view, view->x, view->y, &px, &py);
    double dx=px-view->x, dy=py-view->y;
    /* Dash reaches the perch or its airspace: take it. */
    if (perch>=0 && view->dash_ready && bot_len(dx, dy)<BOT_DASH_LENGTH+6 && bot_dash_safe(view, dx, dy)) {
        bot_act(act, bot_aim_byte(dx, dy), BOT_DASH, 0); return;
    }
    /* Platform freezes the fall and pops us onto a fresh platform. Spend it
     * once steering can no longer make the perch (or the fall is fast/low). */
    int low=view->y<BOT_WATER+9 || view->vy<-30;
    int unreachable=perch<0 || fabs(dx)>fabs(dy)*0.8+6;
    if (view->platform_ready && (low || unreachable)) {
        bot_act(act, BOT_AIM_UP, BOT_PLATFORM, 0); return;
    }
    /* Bow stalls the fall (morph hovers) and the shot's recoil pushes up when
     * fired downward. A last resort before steering alone. */
    if (can_arrow && view->bow_ready && view->vy<-10 && view->y<BOT_WATER+12) {
        bot_act(act, bot_aim_byte(0, -1), BOT_ARROW, BOT_HOLD_QUICK); return;
    }
    bot_act(act, bot_aim_byte(dx>=0?1:-1, 0), 0, 0);
    (void)bot;
}

/* Dodge an incoming arrow. */
static void bot_dodge(const BotView *view, BoplBot *bot, BotAction *act) {
    double dir=1;
    double eta=bot_arrow_eta(view, &dir);
    int window=bot->level>=BOPL_BOT_HARD?26:bot->level==BOPL_BOT_MEDIUM?18:10;
    if (eta>window) { bot->threat_seen=0; return; }
    bot->threat_seen++;
    int wait=bot->level>=BOPL_BOT_HARD?0:bot->level==BOPL_BOT_MEDIUM?1:2;
    if (bot->threat_seen<=wait) return;
    /* Perpendicular escape, biased away from the arrow's travel. */
    double away=-dir;
    if (view->grounded) {
        if (view->dash_ready && eta<12 && bot_dash_safe(view, away, 0.35))
            { bot_act(act, bot_aim_byte(away, 0.35), BOT_DASH, 0); return; }
        double ahead=bot_drop(view, view->x+away*2.2, view->y+0.5);
        if (ahead>4.5) away=-away;  /* do not dodge into a gap */
        bot_act(act, bot_aim_byte(away, 0.2), BOT_JUMP, 0); return;
    }
    if (view->dash_ready && eta<12 && bot_dash_safe(view, away, -0.2))
        { bot_act(act, bot_aim_byte(away, -0.2), BOT_DASH, 0); return; }
    if (view->platform_ready && eta<8)
        { bot_act(act, BOT_AIM_UP, BOT_PLATFORM, 0); return; }  /* freeze: hurtbox vanishes */
    bot_act(act, bot_aim_byte(away, 0), 0, 0);
}

/* Charge choice: fastest total time whose speed covers the range, preferring
 * heavier charges on targets that cannot dodge (still, morphing, far). */
static int bot_pick_hold(const BoplBot *bot, const BotView *view, double range) {
    int stuck=view->foe_morphing || bot->foe_still>=12;
    if (range<5.0 && !stuck) return BOT_HOLD_QUICK;
    if (stuck && range>7) return range<20?BOT_HOLD_MID:BOT_HOLD_FULL;
    if (range<16) return BOT_HOLD_MID;
    if (range<21 && bot->level<BOPL_BOT_HARD) return BOT_HOLD_MID;
    return BOT_HOLD_FULL;
}

/* Try to fire: pick a charge for the range, solve, emit. */
static int bot_try_shot(const BotView *view, BoplBot *bot, int forced_release, BotAction *act) {
    if (!view->bow_ready || view->morphing || forced_release) return 0;
    /* Flying fast (dash exit, platform ejection): the morph-entry position is
     * unsettled, so the solve would be from the wrong origin. Walking is fine
     * (morph entry kills the slime's velocity). */
    if (!view->grounded && bot_len(view->svx, view->svy)>14) return 0;
    double range=bot_len(view->fx-view->x, view->fy-view->y);
    int hold=bot_pick_hold(bot, view, range);
    int desperate=bot->since_shot>40;
    uint8_t aim; double flight;
    static const int order[3]={BOT_HOLD_FULL, BOT_HOLD_MID, BOT_HOLD_QUICK};
    int start=hold==BOT_HOLD_QUICK?2:hold==BOT_HOLD_MID?1:0;
    for (int high_pass=0; high_pass<2; high_pass++) {
        int allow_high=high_pass || view->foe_morphing || bot->foe_still>=8 || desperate;
        for (int k=start; k<3; k++) {
            int h=order[k];
            if (BOT_ARROW_SPEED[h]*BOT_ARROW_SPEED[h]/BOT_ARROW_G<=range*0.85) continue;
            if (bot_solve_shot(view, h, BOT_FIRE_DELAY[h], allow_high, 0, &aim, &flight)) {
                bot_act(act, aim, BOT_ARROW, (uint8_t)h);
                return 1;
            }
        }
        start=0;
    }
    return 0;
}

/* Live aim while a charge is latched: re-solve every decision so the final
 * latched decision (whose aim the fire uses) tracks the target. */
static uint8_t bot_charge_aim(const BotView *view, BoplBot *bot, int hold_left, const int hold_run[4]) {
    int charge_ticks=hold_run[2]*4;  /* decisions the arrow has been down */
    double speed=charge_ticks+hold_left*4>=56?75:charge_ticks+hold_left*4>=28?30:15;
    double lead=hold_left*4+8;
    double fx=0, fy=0; int have_flat=0;
    for (int high=0; high<2; high++) {
        double tx, ty, ux, uy, t=bot_len(view->fx-view->x, view->fy-view->y)/speed;
        int ok=1;
        for (int i=0; i<4 && ok; i++) {
            bot_predict_foe(view, lead+t*60, &tx, &ty);
            if (view->foe_grounded) ty+=0.35;
            ok=bot_ballistic_moving(view->x, view->y, view->svx, view->svy, speed, high, tx, ty, &ux, &uy, &t);
        }
        if (!ok || (high && t>4.0)) continue;
        if (!high) { fx=ux; fy=uy; have_flat=1; }
        if (bot_shot_clear(view, view->x, view->y, ux*speed+view->svx, uy*speed+view->svy, t, tx, ty))
            return bot_aim_byte(ux, uy);
    }
    (void)bot;
    /* Nothing certifies mid-charge (the foe moved behind cover): hold the
     * flat solution rather than waste the arrow entirely. */
    if (have_flat) return bot_aim_byte(fx, fy);
    return bot_aim_byte(view->fx-view->x, view->fy-view->y);
}

static void bot_nav_foe(const BotView *view, BoplBot *bot, uint8_t *aim, uint16_t *buttons) {
    double px=view->fx, py=view->fy;
    int wi=bot_foe_deck(view, &px, &py);
    if (wi>=0 && py>view->y+4.0) {
        const BotWall *w=&view->walls[wi];
        double span=fmax(w->spanx-0.5, 0);
        double left=w->cx-span, right=w->cx+span;
        int lok=bot_drop(view, left, view->y+0.5)<8.0;
        int rok=bot_drop(view, right, view->y+0.5)<8.0;
        if (lok && !rok) px=left-2.5;
        else if (rok && !lok) px=right+2.5;
        else px=(fabs(view->x-left)<=fabs(view->x-right))?left-2.5:right+2.5;
        if (!bot_surface_y(view, w, fmin(fmax(px, left), right), &py)) py=w->top+BOT_SLIME_RADIUS;
    } else if (wi<0) { px=view->fx; py=view->fy; }
    bot_nav(view, bot, px, py, aim, buttons);
}

/* ---- personalities ----------------------------------------------------- */
static void bot_archer(const BotView *view, BoplBot *bot, int forced_release, BotAction *act) {
    double range=bot_len(view->fx-view->x, view->fy-view->y);
    /* Too close to a live threat: make space (dash away, or retreat). */
    if (range<7 && !view->foe_morphing && bot->foe_still<8) {
        double away=view->x>=view->fx?1:-1;
        if (view->dash_ready && bot_dash_safe(view, away, 0.5))
            { bot_act(act, bot_aim_byte(away, 0.5), BOT_DASH, 0); return; }
        uint8_t aim=0; uint16_t buttons=0;
        double px=view->x+away*14, py=view->y;
        int perch=bot_nearest_perch(view, px, py+4, &px, &py);
        if (perch<0) { px=view->x+away*10; py=view->y; }
        bot_nav(view, bot, px, py, &aim, &buttons);
        bot_act(act, aim, buttons, 0); return;
    }
    if (bot_try_shot(view, bot, forced_release, act)) return;
    /* No shot from here. Walk to a certified lip if one is reachable;
     * otherwise close on the foe's deck. Standing in the 10-24 band with
     * no solution is how map-0 timed out for two minutes. */
    uint8_t aim=BOT_AIM_UP; uint16_t buttons=0;
    double sx, sy;
    if (view->bow_timer_left<1.5 && view->fy>view->y-4
            && bot_firing_spot(view, &sx, &sy)
            && bot_len(sx-view->x, sy-view->y)>1.8) {
        bot_nav(view, bot, sx, sy, &aim, &buttons);
        bot_act(act, aim, buttons, 0); return;
    }
    if (view->bow_timer_left<1.5 && view->grounded && view->fy>view->y+5
            && view->fy<view->y+11 && fabs(view->fx-view->x)<8
            && view->platform_ready && (range>11 || view->foe_morphing)) {
        bot_act(act, BOT_AIM_UP, BOT_PLATFORM, 0); return;
    }
    bot_nav_foe(view, bot, &aim, &buttons);
    bot_act(act, aim, buttons, 0);
}

static void bot_pursuer(const BotView *view, BoplBot *bot, int forced_release, BotAction *act) {
    double dx=view->fx-view->x, dy=view->fy-view->y;
    double range=bot_len(dx, dy);
    /* Edge-guard: the foe is falling past the stage - meet them with arrows. */
    int foe_falling_out=!view->foe_grounded && !view->foe_morphing && view->fvy<-8 && view->fy<BOT_WATER+14;
    if (foe_falling_out && bot->level>=BOPL_BOT_MEDIUM && bot_try_shot(view, bot, forced_release, act)) return;
    /* In blade range: quick arrow (a 15 u/s lob covers ~5 units). */
    if (range<5.5 && bot_try_shot(view, bot, forced_release, act)) return;
    /* Mid range with a stationary or morphing target: take the free hit. */
    if ((view->foe_morphing || bot->foe_still>=10) && bot_try_shot(view, bot, forced_release, act)) return;
    /* Any certified shot before closing - vertical splits used to skip this. */
    if (bot_try_shot(view, bot, forced_release, act)) return;
    /* Foe holding high (or low) ground: go to their deck, not under the slab. */
    if (fabs(dy)>4.0) {
        uint8_t aim=0; uint16_t buttons=0;
        bot_nav_foe(view, bot, &aim, &buttons);
        bot_act(act, aim, buttons, 0); return;
    }
    /* Close the distance. Dash covers 13-19 units instantly, but only spend
     * it when it ends near the foe. */
    if (view->dash_ready && !view->morphing && range>9 && range<21
            && bot_dash_reaches(view, view->fx, view->fy) && bot_dash_safe(view, dx, dy)) {
        bot_act(act, bot_aim_byte(dx, dy), BOT_DASH, 0); return;
    }
    if (range<8.5 && view->bow_ready) {
        uint8_t aim=0; uint16_t buttons=0;
        bot_nav(view, bot, view->fx, view->fy, &aim, &buttons);
        bot_act(act, aim, buttons, 0); return;
    }
    if (range<9 && !view->bow_ready) {
        double orbit=(bot->decisions/8)%2?1:-1;
        uint8_t aim=0; uint16_t buttons=0;
        bot_nav(view, bot, view->fx+orbit*6, view->y, &aim, &buttons);
        bot_act(act, aim, buttons, 0); return;
    }
    uint8_t aim=0; uint16_t buttons=0;
    bot_nav_foe(view, bot, &aim, &buttons);
    bot_act(act, aim, buttons, 0);
}

static void bot_evasive(const BotView *view, BoplBot *bot, int forced_release, BotAction *act) {
    double range=bot_len(view->fx-view->x, view->fy-view->y);
    /* A stalemate is worse than a contested shot. */
    if (bot->since_shot>40 && bot_try_shot(view, bot, forced_release, act)) return;
    /* Punish from safety only. */
    if (range>18 && (view->foe_morphing || bot->foe_still>=16 || range>30)
            && bot_try_shot(view, bot, forced_release, act)) return;
    /* A foe that refuses to engage forfeits the safety rule: no shot certifies
     * from the keep-away perch, so walk to one that does and take it. */
    if (bot->foe_still>=30 && view->bow_timer_left<1.5) {
        double sx, sy;
        if (bot_try_shot(view, bot, forced_release, act)) return;
        if (bot_firing_spot(view, &sx, &sy)) {
            uint8_t aim=0; uint16_t buttons=0;
            bot_nav(view, bot, sx, sy, &aim, &buttons);
            bot_act(act, aim, buttons, 0); return;
        }
    }
    /* Keep away: run to the standable perch that maximizes distance. */
    double best_x=view->x, best_y=view->y, best_score=-1e9;
    for (int i=0; i<view->wall_count; i++) {
        const BotWall *w=&view->walls[i];
        if (!w->standable) continue;
        for (int side=-1; side<=1; side++) {
            double span=fmax(w->spanx-0.8, 0);
            double px=w->cx+side*span, py;
            if (!bot_surface_y(view, w, px, &py)) continue;
            double score=bot_len(px-view->fx, py-view->fy)-0.35*bot_len(px-view->x, py-view->y)+0.5*py;
            if (score>best_score) { best_score=score; best_x=px; best_y=py; }
        }
    }
    /* Escape dash when cornered. */
    if (range<8 && view->dash_ready) {
        double ex=best_x-view->x, ey=best_y-view->y+2;
        if (bot_dash_safe(view, ex, ey)) { bot_act(act, bot_aim_byte(ex, ey), BOT_DASH, 0); return; }
        double away=view->x>=view->fx?1:-1;
        if (bot_dash_safe(view, away, 0.6)) { bot_act(act, bot_aim_byte(away, 0.6), BOT_DASH, 0); return; }
    }
    uint8_t aim=0; uint16_t buttons=0;
    bot_nav(view, bot, best_x, best_y, &aim, &buttons);
    bot_act(act, aim, buttons, 0);
}

static void bot_mixed(const BotView *view, BoplBot *bot, int forced_release, BotAction *act) {
    double range=bot_len(view->fx-view->x, view->fy-view->y);
    int foe_falling_out=!view->foe_grounded && !view->foe_morphing && view->fvy<-8 && view->fy<BOT_WATER+14;
    int mode=bot->mode;
    if (bot->mode_hold>0) bot->mode_hold--;
    else {
        if (foe_falling_out || view->foe_morphing || bot->foe_still>=12) mode=BOPL_BOT_PURSUER;
        else if (!view->bow_ready && view->bow_timer_left>1.2 && range<14 && !view->dash_ready) mode=BOPL_BOT_EVASIVE;
        else mode=BOPL_BOT_ARCHER;
        if (mode!=bot->mode) { bot->mode=mode; bot->mode_hold=4; }
    }
    switch (bot->mode) {
    case BOPL_BOT_PURSUER: bot_pursuer(view, bot, forced_release, act); return;
    case BOPL_BOT_EVASIVE: bot_evasive(view, bot, forced_release, act); return;
    default: bot_archer(view, bot, forced_release, act); return;
    }
}

/* ---- entry points ------------------------------------------------------ */
static void bopl_bot_reset(BoplBot *bot, int kind, int level, uint32_t seed) {
    BoplRl *fork=bot->fork;
    memset(bot, 0, sizeof(*bot));
    bot->fork=fork;
    bot->kind=kind; bot->level=level;
    bot->rng=seed?seed:1;
    bot->last[0]=BOT_AIM_UP;
}

/* One decision. hold_left/held/hold_run mirror the environment's latch state
 * BEFORE this decision (bopl_latch applies the result exactly like a policy
 * action). out = {aim byte, button bits, hold index}. */
static void bopl_bot_decide(BoplBot *bot, const BoplRlState *state, int player_index,
                            int hold_left, uint16_t held, const int hold_run[4],
                            uint8_t out[3]) {
    out[0]=BOT_AIM_UP; out[1]=0; out[2]=0;
    if (bot->kind>BOPL_BOT_V1_KINDS) { bopl_bot_decide_v2(bot, state, player_index, hold_left, held, hold_run, out); return; }
    /* ---- v1 (kinds 1-4): frozen; see the kind table above. ---- */
    BotView view;
    int live=bot_view(&view, state, player_index);
    bot->decisions++;
    if (!view.alive || !view.my_slime || view.dead_phase) {
        bot->have_prev=0;
        return;
    }
    /* Own velocity estimate from displacement (15 decisions/second). */
    if (bot->have_prev) { view.svx=(view.x-bot->self_prev_x)*15; view.svy=(view.y-bot->self_prev_y)*15; }
    bot->self_prev_x=view.x; bot->self_prev_y=view.y; bot->have_prev=1;
    /* Passivity tracking. */
    if (live) {
        double moved=bot_len(view.fx-bot->foe_prev_x, view.fy-bot->foe_prev_y);
        bot->foe_still=(moved<0.15 && bot->decisions>1)?bot->foe_still+1:0;
        bot->foe_prev_x=view.fx; bot->foe_prev_y=view.fy;
    }
    /* Latched: only the aim matters. Track the firing solution while charging;
     * otherwise steer toward the foe (or the map) so a dash/jump latch is not
     * wasted aim time. */
    if (hold_left>0) {
        if (live && (held&BOT_ARROW)) out[0]=bot_charge_aim(&view, bot, hold_left, hold_run);
        else if (live) out[0]=bot_aim_byte(view.fx-view.x>=0?1:-1, 0);
        out[1]=held; out[2]=0;
        bot->last[0]=out[0];
        return;
    }
    /* The decision right after an arrow latch must release it (the bow fires
     * during these four ticks; its aim was fixed on the previous decision). */
    int forced_release=hold_run[2]>0;
    if (!live) return;  /* foe dead or gone: stand */
    /* Difficulty: occasionally replay the previous decision (reaction lag). */
    double skip=bot->level>=BOPL_BOT_HARD?0.0:bot->level==BOPL_BOT_MEDIUM?0.08:0.22;
    if (bot->decisions>1 && bot_unit(bot)<skip && !(bot->last[1]&BOT_ARROW)) {
        out[0]=bot->last[0]; out[1]=bot->last[1]&(uint16_t)~BOT_ARROW; out[2]=0;
        return;
    }
    BotAction act; memset(&act, 0, sizeof(act));
    if (view.morphing) {
        /* Still in the bow morph with the arrow loaded (the shot leaves a few
         * ticks after the latch expires): THIS aim is the one the arrow gets,
         * so keep tracking the firing solution until it actually fires. */
        if (view.charging) bot_act(&act, bot_charge_aim(&view, bot, 0, hold_run), 0, 0);
        /* Committed otherwise (dash tail, platform morph): aim somewhere useful. */
        else bot_act(&act, bot_aim_byte(view.fx-view.x>=0?1:-1, 0), 0, 0);
    }
    if (!act.done) bot_dodge(&view, bot, &act);
    if (!act.done) switch (bot->kind) {
    case BOPL_BOT_ARCHER: bot_archer(&view, bot, forced_release, &act); break;
    case BOPL_BOT_PURSUER: bot_pursuer(&view, bot, forced_release, &act); break;
    case BOPL_BOT_EVASIVE: bot_evasive(&view, bot, forced_release, &act); break;
    case BOPL_BOT_MIXED: bot_mixed(&view, bot, forced_release, &act); break;
    default: bot_act(&act, BOT_AIM_UP, 0, 0); break;
    }
    /* Sinking stage: climb if we're near the water. Don't cancel a shot. */
    if (view.grounded && !(act.buttons&BOT_ARROW) && view.y<BOT_WATER+8
            && view.state->header.episode_ticks>90*60) {
        double hx, hy;
        if (bot_highest_perch(&view, &hx, &hy) && hy>view.y+1) {
            uint8_t aim=0; uint16_t buttons=0;
            bot_nav(&view, bot, hx, hy, &aim, &buttons);
            bot_act(&act, aim, buttons, 0);
        } else if (view.platform_ready)
            bot_act(&act, BOT_AIM_UP, BOT_PLATFORM, 0);
    }
    /* Survival veto: while airborne, whatever the personality wants to do must
     * still land somewhere dry. If its steering direction is doomed, try the
     * other steers; if nothing lands, spend a recovery resource. (An arrow
     * press is exempt: the bow morph stops the fall entirely.) */
    if (!view.morphing && !view.grounded && !(act.buttons&(BOT_DASH|BOT_PLATFORM|BOT_ARROW))) {
        double ang=((int)act.aim-1)/255.0*2*M_PI, ax=cos(ang), lx, ly;
        int steer=ax>0.33?1:ax<-0.33?-1:0;
        int doomed=!bot_landing_steer(&view, steer, &lx, &ly);
        int shot_down=bot_steer_hits_arrow(&view, steer);
        if (doomed || shot_down) {
            static const int opts[3]={1, -1, 0};
            int found=0;
            for (int i=0; i<3 && !found; i++)
                if (opts[i]!=steer && bot_landing_steer(&view, opts[i], &lx, &ly)
                        && !bot_steer_hits_arrow(&view, opts[i])) {
                    act.aim=opts[i]?bot_aim_byte(opts[i], 0):BOT_AIM_UP;
                    found=1;
                }
            if (!found && shot_down && !doomed && view.platform_ready) {
                /* Boxed in by an arrow: the platform freeze drops the hurtbox. */
                bot_act(&act, BOT_AIM_UP, BOT_PLATFORM, 0);
            } else if (!found && doomed) {
                BotAction rescue; memset(&rescue, 0, sizeof(rescue));
                bot_recover(&view, bot, !forced_release, &rescue);
                if (rescue.done) act=rescue;
            }
        }
    }
    /* Launch veto: a grounded jump (or the platform ejection that follows a
     * platform press) rises at ~58 u/s and can fly into an arrow - even into
     * the bot's own mortar still climbing away from it. Jumps off slopes
     * launch along the surface normal, not the aim (an archer aiming up-left
     * was thrown up-RIGHT into its arrow), so any horizontal outcome vetoes. */
    if (view.grounded && (act.buttons&(BOT_JUMP|BOT_PLATFORM)) && view.arrow_count) {
        if (bot_launch_hits_arrow(&view, -22) || bot_launch_hits_arrow(&view, 0)
                || bot_launch_hits_arrow(&view, 22))
            act.buttons&=(uint16_t)~(BOT_JUMP|BOT_PLATFORM);
    }
    /* Doomed-jump veto: with dash and platform both on cooldown, a jump whose
     * possible launches (slope normals again) cannot all steer to dry ground
     * is unrecoverable - an evasive bot jumped off a low corner near water
     * with both resources spent and the bow stall only delayed the splash. */
    if (view.grounded && (act.buttons&BOT_JUMP) && !view.dash_ready && !view.platform_ready) {
        if (!bot_launch_lands(&view, -22) || !bot_launch_lands(&view, 0)
                || !bot_launch_lands(&view, 22))
            act.buttons&=(uint16_t)~BOT_JUMP;
    }
    /* Difficulty: aim noise. */
    if (bot->level<BOPL_BOT_HARD && act.aim) {
        double sigma=bot->level==BOPL_BOT_MEDIUM?2.0:5.0;  /* aim-byte steps, ~1.4deg each */
        double u1=bot_unit(bot), u2=bot_unit(bot);
        if (u1>1e-9) {
            int wobble=(int)lround(sqrt(-2*log(u1))*cos(2*M_PI*u2)*sigma);
            int value=((int)act.aim-1+wobble)%255; if (value<0) value+=255;
            act.aim=(uint8_t)(1+value);
        }
    }
    /* Never emit an arrow press on a forced-release decision (mask parity). */
    if (forced_release) act.buttons&=(uint16_t)~BOT_ARROW;
    bot->since_shot=(act.buttons&BOT_ARROW)?0:bot->since_shot+1;
    bot->since_jump=(act.buttons&BOT_JUMP)?0:bot->since_jump+1;
    static int bot_debug=-1;
    if (bot_debug<0) bot_debug=getenv("BOPL_BOT_DEBUG")!=NULL;
    if (bot_debug) {
        double sx=0, sy=0; int spot=bot_firing_spot(&view, &sx, &sy);
        double lx, ly;
        int lminus=bot_landing_steer(&view, -1, &lx, &ly), lzero=bot_landing_steer(&view, 0, &lx, &ly),
            lplus=bot_landing_steer(&view, 1, &lx, &ly);
        fprintf(stderr, "[bot %d] d%03d me(%.1f,%.1f) v(%.1f,%.1f) g%d m%d foe(%.1f,%.1f) mode%d bow%d dash%d aim%3d btn%x hold%d spot%d(%.1f,%.1f) land[%d%d%d] arr%d died%d\n",
                player_index, bot->decisions, view.x, view.y, view.vx, view.vy, view.grounded, view.morphing,
                view.fx, view.fy, bot->mode, view.bow_ready, view.dash_ready,
                act.aim, act.buttons, act.hold, spot, sx, sy, lminus, lzero, lplus,
                view.arrow_count, 0);
    }
    /* A latch is only ever intentional: anything but an arrow charge stays free. */
    if (!(act.buttons&BOT_ARROW)) act.hold=0;
    out[0]=act.aim; out[1]=(uint8_t)(act.buttons&15u); out[2]=act.hold;
    memcpy(bot->last, out, 3);
}

#include "bopl_bots_v2.h"
#endif
