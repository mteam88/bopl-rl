#ifndef BOPL_BOTS_V2_H
#define BOPL_BOTS_V2_H
/* Scripted bot kinds after the frozen v1 set.
 * Included at the end of bopl_bots.h, so the v1 helpers (BotView, ballistic
 * solver, navigation) are available to build on; the v1 functions themselves
 * are never edited. Adding a kind: append its id to the enum and its name to
 * BOPL_BOT_NAMES in bopl_bots.h, mirror both in
 * evaluation_protocol.SCRIPTED_KINDS, and dispatch it below.
 *
 * ---- search (kind 5): rollout search on a forked sim ------------------
 * Model-predictive control: every decision, fork the live game
 * (bopl_rl_fork: a private copy with the same random stream, so the same
 * inputs give the same future), play a set of candidate action sequences on
 * the fork for a horizon, score each one, and emit the first decision of the
 * best. Candidates are macros: stand, walk, jump, dash in a direction, dash
 * then shoot, shoot (the v1 ballistic solver's aims, one set per charge) and
 * platform. The rollout runs the real sim, so everything the v1 bots
 * approximate (own fire-delayed arrows, morph timing, terrain, squish) is
 * exact: a shoot-then-dash into the own arrow simply scores -1.
 *
 * Foe model in the rollout: the foe keeps the movement direction of its last
 * applied input with every button released (a pending bow charge therefore
 * fires at once, at its current aim), and dodges like a hard v1 bot when an
 * arrow threatens it (bot_dodge: dash away if the dash is ready, else jump /
 * platform freeze) - a foe on cooldown cannot dodge, which is the cooldown
 * window humans attack in.
 *
 * Score: own death -1 (+0.25 * t/H, later is better), foe death +1
 * (-0.25 * t/H, sooner is better), both -1; otherwise a heuristic on the end
 * state (arrow threat ETA, dry-landing survivability, own resources ready,
 * foe dash/platform on cooldown, height, distance band). Every candidate is
 * scored at the level's horizon H unless its own action outlasts it (charge
 * + flight, the platform morph): the heuristic is not stationary in H (the
 * foe model shoots once, then its arrow is gone and its dash spent), so a
 * candidate that merely looks further ahead scores higher - six random
 * sequences whose arrows ran to H + charge + 60 caused half of search-hard's
 * losses that way (Levels). Deterministic: no RNG drawn.
 * Levels are the horizon: hard 128 ticks, medium 64, easy 32 (longer is
 * stronger against policies). Requires a backend
 * with bopl_rl_fork (the Rust sim); anything else aborts. */
#include "bopl_layout.h"
#include "bopl_bots_v2_play.h"

_Static_assert(sizeof(BOPL_BOT_NAMES)/sizeof(BOPL_BOT_NAMES[0])==BOPL_BOT_KINDS+1, "one name per bot kind");

typedef struct { uint8_t aim; uint16_t buttons; uint8_t hold; } SearchDecision;
enum { SEARCH_HOLD=0, SEARCH_DASH_SHOOT=1 };
typedef struct {
    SearchDecision first;   /* played if this candidate wins */
    uint8_t after_aim;      /* aim once the first decision (and its latch) is over; buttons 0 */
    int macro, horizon;     /* continuation rule, rollout ticks */
    double score;
} SearchCandidate;
enum { SEARCH_MAX_CANDIDATES=64 };

static double search_clamp(double v, double lo, double hi) { return v<lo?lo:v>hi?hi:v; }
static int search_foe_index(const BoplRlState *s, int me) {
    for (uint32_t i=0; i<s->header.player_count; i++)
        if ((int)i!=me && (s->players[i].flags&1)) return (int)i;
    return -1;
}
static int search_arrows_live(const BoplRlState *s) {
    for (uint32_t i=0; i<s->header.entity_count; i++) {
        const BoplRlEntity *e=&s->entities[i];
        if ((e->flags&1) && !(e->flags&4) && e->kind==0 && e->layer==9) return 1;
    }
    return 0;
}
/* Does any charge certify a terrain-clear shot at the foe from this view? */
static int search_shot_exists(const BotView *v) {
    static const int holds[3]={BOT_HOLD_FULL, BOT_HOLD_MID, BOT_HOLD_QUICK};
    double range=bot_len(v->fx-v->x, v->fy-v->y), flight; uint8_t aim;
    for (int k=0; k<3; k++) {
        int h=holds[k];
        if (BOT_ARROW_SPEED[h]*BOT_ARROW_SPEED[h]/BOT_ARROW_G<=range*0.85) continue;
        if (bot_solve_shot(v, h, BOT_FIRE_DELAY[h], 1, 0, &aim, &flight)) return 1;
    }
    return 0;
}
/* Per-decision context shared by every candidate's heuristic: a firing spot
 * (v1 bot_firing_spot) from the real state, when no shot certifies here.
 * Without it the search camps: within a 1.5 s horizon nothing beats standing
 * on a perch with every resource ready, even with the foe idle below a deck
 * (measured: search-hard vs idle on map 1 stood 7000 ticks). */
typedef struct { int have_spot, foe_passive, stalled; double sx, sy; } SearchContext;
/* Heuristic value of a non-terminal end state for player `me`. */
static double search_heuristic(const BoplRlState *s, int me, const SearchContext *ctx) {
    BotView v;
    int live=bot_view(&v, s, me);
    if (!v.alive || !v.my_slime || v.dead_phase) return -1;
    if (!live) return 1;
    double h=0;
    /* A certified shot from here is the next decision's kill; otherwise
     * progress toward a spot that has one. */
    double d=bot_len(v.fx-v.x, v.fy-v.y);
    if (!v.morphing && !ctx->stalled && search_shot_exists(&v)) h+=0.2;
    else if (ctx->have_spot) h-=0.008*bot_len(ctx->sx-v.x, ctx->sy-v.y);
    /* No known firing spot either: close in (the v1 pursuer's fallback).
     * Without a gradient the bot camps wherever it stands, e.g. on map 0's
     * middle platform 16 u from an idle foe, until the round times out. */
    else h-=0.008*d;
    /* Standing where the foe's bow (ready or charging) has a certified shot
     * at me is the opposite: seek cover - unless the foe is passive. */
    if (v.foe_slime && v.foe && !ctx->foe_passive) {
        const BoplRlSlime *f=v.foe_slime;
        int foe_bow=(f->flags&1)?f->timers[BOT_SLOT_BOW]>f->cooldowns[BOT_SLOT_BOW]:(f->ability_flags[BOT_SLOT_BOW]&BOPL_ABILITY_ACTIVE)!=0;
        BotView fv;
        if (foe_bow && bot_view(&fv, s, (int)(v.foe-s->players)) && search_shot_exists(&fv)) h-=0.2;
    }
    double eta=bot_arrow_eta(&v, NULL);
    if (eta<60) h-=0.5*(1-eta/60);
    if (!v.grounded && !v.morphing) {
        double lx, ly; int lands=0;
        for (int steer=-1; steer<=1 && !lands; steer++) lands=bot_landing_steer(&v, steer, &lx, &ly);
        h-=lands?0.03:(v.dash_ready||v.platform_ready)?0.15:0.6;
    }
    h+=0.06*v.bow_ready+0.05*v.dash_ready+0.04*v.platform_ready;
    if (v.foe_slime) {
        const BoplRlSlime *f=v.foe_slime;
        int foe_dash=f->timers[BOT_SLOT_DASH]>f->cooldowns[BOT_SLOT_DASH];
        int foe_platform=f->timers[BOT_SLOT_PLATFORM]>f->cooldowns[BOT_SLOT_PLATFORM];
        h+=0.06*!foe_dash+0.03*!foe_platform;
    }
    h+=0.015*search_clamp(v.y-v.fy, -6, 6);
    h-=0.012*fmax(0, 7-d)+0.006*fmax(0, d-20);
    return h;
}
/* Coarse "the foe can shoot me from here": within bow reach and a straight
 * line of sight (one sphere-marched ray, ~5 us; the certified v1 solve is
 * ~10x that and too slow to run every rollout decision). */
static int search_coarse_shot(const BoplRlState *s, int foe, double fx, double fy, double mx, double my) {
    double dx=mx-fx, dy=my-fy, d=bot_len(dx, dy);
    if (d<1e-6 || d>45) return 0;
    BotView fv;
    if (!bot_view(&fv, s, foe)) return 0;
    return bot_raycast(&fv, fx, fy, dx/d, dy/d, d-1.0)>=d-1.0;
}
/* Foe model state for one rollout. */
/* dodge: 0 v1 hard bot_dodge; 1 dash away at first sight (jump if the dash
 * is down). Two more (platform freeze, no reaction) were hard-only and cost
 * ~35 Elo for twice the rollouts per shot. */
typedef struct { int hold_left, fired, dashed, dodge, passive; double speed; } SearchFoe;
/* The foe's input for one rollout decision (see the header comment). A foe
 * already charging either releases now (mode 0) or keeps charging with aim
 * tracking me and releases at full charge (mode 1). A foe whose bow is (or
 * becomes) ready shoots once, at the first decision with a coarse shot at me
 * (search_coarse_shot): charge picked by range like v1 bot_pick_hold, aim a
 * flat arc at my current position every decision. Otherwise it keeps its
 * last movement direction, buttons up, and dodges arrows like a hard v1 bot. */
static void search_foe_action(const BoplRlState *s, int foe, int me, uint8_t base_aim, int mode,
                              SearchFoe *fm, BoplRlAction *a) {
    a->angle=base_aim; a->reserved=0; a->buttons=0;
    const BoplRlSlime *fs=bot_slime(s, s->players[foe].id);
    if (!fs) return;
    const BoplRlEntity *body=(fs->flags&1)?NULL:bot_morph_entity(s, s->players[foe].id);
    double fx=body?bot_fix(body->x):(fs->flags&1)?bot_fix(fs->x):bot_fix(s->players[foe].x);
    double fy=body?bot_fix(body->y):(fs->flags&1)?bot_fix(fs->y):bot_fix(s->players[foe].y);
    double mx=bot_fix(s->players[me].x), my=bot_fix(s->players[me].y), ux, uy, t;
    int charging=!(fs->flags&1) && (fs->ability_flags[BOT_SLOT_BOW]&BOPL_ABILITY_ACTIVE);
    /* A foe already growing a platform keeps holding it (up to the 4 s
     * max_size_time), so the fork's platform grows as the real one would
     * rather than ejecting at once: squish geometry inside the horizon is
     * then exact (PlatformTransform.cs; rollout/mod.rs PLATFORM_SIZE). */
    if (!(fs->flags&1) && (fs->ability_flags[BOT_SLOT_PLATFORM]&BOPL_ABILITY_ACTIVE)) {
        a->buttons=BOT_PLATFORM;
        return;
    }
    if (fm->passive) return;  /* still for 20 decisions (SearchContext): stands, no shot, no dodge */
    if (charging && mode==1 && !fm->fired
            && ((fs->ability_flags[BOT_SLOT_BOW]>>BOPL_ABILITY_CHARGE_SHIFT)&BOPL_ABILITY_CHARGE_MASK)<4) {
        a->buttons=BOT_ARROW;
        a->angle=bot_ballistic_arc(fx, fy, 75, mx, my, 0, &ux, &uy, &t)?bot_aim_byte(ux, uy):bot_aim_byte(mx-fx, my-fy);
        return;
    }
    if (fm->hold_left>0) {
        fm->hold_left--;
        a->buttons=BOT_ARROW;
        a->angle=bot_ballistic_arc(fx, fy, fm->speed, mx, my, 0, &ux, &uy, &t)?bot_aim_byte(ux, uy):bot_aim_byte(mx-fx, my-fy);
        return;
    }
    if (!fm->fired && !charging && (fs->flags&1) && fs->timers[BOT_SLOT_BOW]>fs->cooldowns[BOT_SLOT_BOW]
            && search_coarse_shot(s, foe, fx, fy, mx, my)) {
        double range=bot_len(mx-fx, my-fy);
        int hold=range<5?BOT_HOLD_QUICK:range<16?BOT_HOLD_MID:BOT_HOLD_FULL;
        fm->fired=1; fm->hold_left=BOPL_HOLD_LENGTHS[hold]-1; fm->speed=BOT_ARROW_SPEED[hold];
        a->buttons=BOT_ARROW;
        a->angle=bot_ballistic_arc(fx, fy, fm->speed, mx, my, 0, &ux, &uy, &t)?bot_aim_byte(ux, uy):bot_aim_byte(mx-fx, my-fy);
        return;
    }
    /* Not modelled: the foe planting a squish platform. Tried (a ready
     * platform within 9 u is planted, the fork grows it exactly): +4 wins
     * vs the squisher trial 001 but -8..-10 vs mixed-hard, trial 012 and
     * run007 — the bot fled every close fight. */
    /* No shot and a ready bow: the foe closes in (v1 pursuit / firing-spot
     * navigation, coarsely); in mode 2 it dashes in when its dash is ready. */
    if (!fm->fired && !charging && (fs->flags&1) && fs->timers[BOT_SLOT_BOW]>fs->cooldowns[BOT_SLOT_BOW]
            && bot_len(mx-fx, my-fy)>6) {
        a->angle=bot_aim_byte(mx>fx?1:-1, 0);
        if (mode==2 && !fm->dashed && fs->timers[BOT_SLOT_DASH]>fs->cooldowns[BOT_SLOT_DASH]) {
            fm->dashed=1; a->buttons=BOT_DASH; a->angle=bot_aim_byte(mx-fx, my-fy+2);
            return;
        }
    }
    int airborne=(fs->flags&1) && !(fs->flags&2);
    int arrows=search_arrows_live(s);
    if (!airborne && !arrows) return;
    BotView fv;
    if (!bot_view(&fv, s, foe) || fv.morphing) return;
    BoplBot scratch; memset(&scratch, 0, sizeof(scratch)); scratch.level=BOPL_BOT_HARD;
    BotAction act; memset(&act, 0, sizeof(act));
    if (arrows) {
        double dir=1, eta=bot_arrow_eta(&fv, &dir);
        if (fm->dodge==0) bot_dodge(&fv, &scratch, &act);
        else if (eta<=26) {
            if (fv.dash_ready) bot_act(&act, bot_aim_byte(-dir, 0.3), BOT_DASH, 0);
            else if (fv.grounded) bot_act(&act, bot_aim_byte(-dir, 0.2), BOT_JUMP, 0);
        }
    }
    /* Airborne: steer to a landing like v1, or spend a resource to recover.
     * Without this the rollouts predicted "the foe falls to its death
     * whatever I do" and the bot started a full charge; the real foe dashed
     * back and shot it inside the latch. */
    if (!act.done && !fv.grounded) {
        double lx, ly; int steer, lands=0;
        for (steer=0; steer>=-1 && !lands; steer=steer==0?1:steer==1?-1:-2) lands=bot_landing_steer(&fv, steer, &lx, &ly);
        if (lands) { a->angle=bot_aim_byte(steer==0?0:steer, 0); return; }
        bot_recover(&fv, &scratch, 0, &act);
    }
    if (act.done) { a->angle=act.aim; a->buttons=(uint16_t)(act.buttons&15u); }
}
static int search_trace=0;  /* BOPL_BOT_DEBUG>=3: trace the winning candidate's rollout */
/* Play one candidate on a fresh fork; returns its score. */
static double search_rollout(BoplBot *bot, const BoplRlState *real, int me, const SearchCandidate *c,
                             int hold_left, uint16_t held, const int hold_run[4], const SearchContext *ctx, int foe_mode, int dodge) {
    BoplRl *f=bopl_rl_fork(bot->world, bot->fork);
    if (!f && bot->fork) { bopl_rl_close(bot->fork); bot->fork=NULL; f=bopl_rl_fork(bot->world, NULL); }
    if (!f) { fprintf(stderr, "Bopl RL: the search bot needs a backend with bopl_rl_fork (the Rust sim)\n"); abort(); }
    bot->fork=f;
    int players=(int)real->header.player_count, foe=search_foe_index(real, me);
    if (foe<0) return 0;
    int hl=hold_left, hr[4]; uint16_t hd=held;
    memcpy(hr, hold_run, sizeof(hr));
    uint8_t foe_aim=bot->applied[foe].angle;
    BoplRlAction acts[6]; memset(acts, 0, sizeof(acts));
    int shot=0;
    SearchFoe fm; memset(&fm, 0, sizeof(fm)); fm.dodge=dodge; fm.passive=ctx->foe_passive;
    for (int t=0; t<c->horizon; t+=4) {
        const BoplRlState *s=bopl_rl_state(f);
        SearchDecision d;
        if (t==0) d=c->first;
        else { d.aim=c->after_aim; d.buttons=0; d.hold=0; }
        if (t>0 && c->macro==SEARCH_DASH_SHOOT && !shot && hl==0) {
            /* The dash tail is over: fire the v1 solver's shot from where we are. */
            BotView v;
            if (bot_view(&v, s, me) && !v.morphing && v.bow_ready) {
                double range=bot_len(v.fx-v.x, v.fy-v.y), flight; uint8_t aim;
                static const int order[3]={BOT_HOLD_FULL, BOT_HOLD_MID, BOT_HOLD_QUICK};
                for (int k=0; k<3 && !shot; k++) {
                    int h=order[k];
                    if (BOT_ARROW_SPEED[h]*BOT_ARROW_SPEED[h]/BOT_ARROW_G<=range*0.85) continue;
                    if (bot_solve_shot(&v, h, BOT_FIRE_DELAY[h], 1, 0, &aim, &flight)) {
                        d.aim=aim; d.buttons=BOT_ARROW; d.hold=(uint8_t)h; shot=1;
                    }
                }
            }
        }
        /* Latch emulation (bopl_latch + bopl_write_mask's arrow-release rule). */
        uint16_t buttons=d.buttons;
        if (hl==0 && hr[BOPL_BUTTON_ARROW]>0) buttons&=(uint16_t)~BOT_ARROW;
        uint16_t applied;
        if (hl>0) { hl--; applied=hd; }
        else { hd=buttons; hl=BOPL_HOLD_LENGTHS[d.hold]-1; applied=hd; }
        for (int b=0; b<4; b++) hr[b]=(applied&(1<<b))?hr[b]+1:0;
        acts[me].angle=d.aim; acts[me].reserved=0; acts[me].buttons=applied;
        search_foe_action(s, foe, me, foe_aim, foe_mode, &fm, &acts[foe]);
        if (search_trace) {
            const BoplRlSlime *fs=bot_slime(s, s->players[foe].id);
            fprintf(stderr, "      t%3d me(%.1f,%.1f) a%3d/%x foe(%.1f,%.1f) a%3d/%x foe bow %s arrows %d\n", t,
                    bot_fix(s->players[me].x), bot_fix(s->players[me].y), acts[me].angle, acts[me].buttons,
                    bot_fix(s->players[foe].x), bot_fix(s->players[foe].y), acts[foe].angle, acts[foe].buttons,
                    !fs?"?":!(fs->flags&1)?"morph":fs->timers[BOT_SLOT_BOW]>fs->cooldowns[BOT_SLOT_BOW]?"ready":"cooldown", search_arrows_live(s));
        }
        if (bopl_rl_step(f, acts, (size_t)players, 4)) return -1;
        s=bopl_rl_state(f);
        int me_dead=!(s->players[me].flags&1), foe_dead=!(s->players[foe].flags&1);
        double frac=(double)(t+4)/c->horizon;
        if (me_dead) return -1+(foe_dead?0:0.25*frac);
        if (foe_dead) return 1-0.25*frac;
        if (s->header.flags&BOPL_RL_TERMINAL) break;
    }
    return search_heuristic(bopl_rl_state(f), me, ctx);
}
static uint8_t search_aim_offset(uint8_t aim, int steps) {
    int value=((int)aim-1+steps)%255; if (value<0) value+=255;
    return (uint8_t)(1+value);
}
static void search_add(SearchCandidate *cands, int *n, uint8_t aim, uint16_t buttons, uint8_t hold,
                       uint8_t after_aim, int macro, int horizon) {
    if (*n>=SEARCH_MAX_CANDIDATES) return;
    SearchCandidate *c=&cands[(*n)++];
    c->first.aim=aim; c->first.buttons=buttons; c->first.hold=hold;
    c->after_aim=after_aim; c->macro=macro; c->horizon=horizon; c->score=0;
}
/* Shot candidates for one charge: the solver's aims (terrain-checked, then
 * flat regardless of terrain); the horizon covers charge + fire delay +
 * flight so the hit or miss is inside the rollout. Random aims (one or two
 * per charge) were tried and cost both levels ~100 Elo: a full charge at an
 * aim the solver rejected wins the argmax now and then on the end-state
 * heuristic, and the bot dies inside it. */
static void search_add_shots(SearchCandidate *cands, int *n, const BotView *view, int hold) {
    double s=BOT_ARROW_SPEED[hold], range=bot_len(view->fx-view->x, view->fy-view->y);
    if (s*s/BOT_ARROW_G<=range*0.7) return;
    uint8_t aims[2], aim; double flights[2], flight; int count=0;
    if (bot_solve_shot(view, hold, BOT_FIRE_DELAY[hold], 1, 0, &aim, &flight)) { aims[count]=aim; flights[count++]=flight; }
    if (bot_solve_shot(view, hold, BOT_FIRE_DELAY[hold], 0, 1, &aim, &flight) && (!count || aim!=aims[0])) { aims[count]=aim; flights[count++]=flight; }
    /* Each aim's own flight time: a high lob hangs up to 4 s, and scored at
     * the flat arc's horizon it ended mid-flight on the heuristic - never a
     * kill, so the bot sat on a "certified" shot it never took. */
    for (int i=0; i<count; i++)
        search_add(cands, n, aims[i], BOT_ARROW, (uint8_t)hold, 0, SEARCH_HOLD,
                   (int)search_clamp(4*BOPL_HOLD_LENGTHS[hold]+16+flights[i]*60+12, 64, 300));
}
static void bot_search(BoplBot *bot, const BoplRlState *state, int me, int hold_left, uint16_t held,
                       const int hold_run[4], uint8_t out[3]) {
    BotView view;
    int live=bot_view(&view, state, me);
    if (!bot->world || !view.alive || !view.my_slime || view.dead_phase || !live) return;  /* stand */
    int H=bot->level>=BOPL_BOT_HARD?128:bot->level==BOPL_BOT_MEDIUM?64:32;  /* the level is the horizon */
    /* Stalled (20 decisions within 0.15 u): a greedy search has fixed
     * points - a jump across a gap that lands outside H, or a v1-certified
     * shot the exact rollout never confirms (a lob the map blocks), which
     * the heuristic keeps rewarding. For the decision the horizon doubles
     * and the certified-shot bonus is withheld (SearchContext.stalled). */
    bot->v2.self_still=bot->have_prev && bot_len(view.x-bot->self_prev_x, view.y-bot->self_prev_y)<0.15?bot->v2.self_still+1:0;
    bot->self_prev_x=view.x; bot->self_prev_y=view.y; bot->have_prev=1;
    int stalled=bot->v2.self_still>=20;
    if (stalled) H*=2;
    double dx=view.fx-view.x, dy=view.fy-view.y;
    SearchCandidate cands[SEARCH_MAX_CANDIDATES]; int n=0;
    if (hold_left>0) {
        /* Latched: only the aim is free. An arrow charge searches its aim
         * around the live solution; anything else steers. */
        if (held&BOT_ARROW) {
            uint8_t live_aim=bot_charge_aim(&view, bot, hold_left, hold_run);
            int charge=(hold_run[BOPL_BUTTON_ARROW]+hold_left)*4;
            int horizon=hold_left*4+16+(int)search_clamp(bot_len(dx, dy)/(charge>=56?75:charge>=28?30:15)*60, 10, 120)+12;
            static const int offsets[3]={0, 2, -2};
            for (int i=0; i<3; i++) search_add(cands, &n, search_aim_offset(live_aim, offsets[i]), held, 0, 0, SEARCH_HOLD, horizon);
            search_add(cands, &n, bot->last[0], held, 0, 0, SEARCH_HOLD, horizon);
        } else {
            search_add(cands, &n, bot_aim_byte(dx>=0?1:-1, 0), held, 0, bot_aim_byte(dx>=0?1:-1, 0), SEARCH_HOLD, H);
            search_add(cands, &n, bot_aim_byte(dx>=0?-1:1, 0), held, 0, bot_aim_byte(dx>=0?-1:1, 0), SEARCH_HOLD, H);
            search_add(cands, &n, BOT_AIM_UP, held, 0, BOT_AIM_UP, SEARCH_HOLD, H);
        }
    } else {
        int forced_release=hold_run[BOPL_BUTTON_ARROW]>0;
        search_add(cands, &n, BOT_AIM_UP, 0, 0, BOT_AIM_UP, SEARCH_HOLD, H);
        for (int side=-1; side<=1; side+=2) {
            uint8_t a=bot_aim_byte(side, 0);
            search_add(cands, &n, a, 0, 0, a, SEARCH_HOLD, H);
            search_add(cands, &n, bot_aim_byte(side, 0.3), BOT_JUMP, 0, a, SEARCH_HOLD, H);
        }
        search_add(cands, &n, bot_aim_byte(dx>=0?0.3:-0.3, 1), BOT_JUMP, 0, BOT_AIM_UP, SEARCH_HOLD, H);
        if (view.dash_ready && !view.morphing) {
            /* Up, down-diagonal and straight-down dashes plus sideways
             * dash-then-shoot candidates cost 130 Elo. */
            static const double dirs[4][2]={{1,0},{-1,0},{1,1},{-1,1}};
            for (int i=0; i<4; i++) search_add(cands, &n, bot_aim_byte(dirs[i][0], dirs[i][1]), BOT_DASH, 0, BOT_AIM_UP, SEARCH_HOLD, H);
            search_add(cands, &n, bot_aim_byte(dx, dy), BOT_DASH, 0, BOT_AIM_UP, SEARCH_HOLD, H);
            search_add(cands, &n, bot_aim_byte(dx, dy+4), BOT_DASH, 0, BOT_AIM_UP, SEARCH_HOLD, H);
            if (view.bow_ready) {
                int horizon=H+4*BOPL_HOLD_LENGTHS[BOT_HOLD_MID]+60;
                search_add(cands, &n, bot_aim_byte(dx, dy), BOT_DASH, 0, BOT_AIM_UP, SEARCH_DASH_SHOOT, horizon);
                search_add(cands, &n, bot_aim_byte(dx, dy+4), BOT_DASH, 0, BOT_AIM_UP, SEARCH_DASH_SHOOT, horizon);
            }
        }
        if (view.bow_ready && !view.morphing && !forced_release) {
            search_add_shots(cands, &n, &view, BOT_HOLD_FULL);
            search_add_shots(cands, &n, &view, BOT_HOLD_MID);
            search_add_shots(cands, &n, &view, BOT_HOLD_QUICK);
        }
        if (view.platform_ready && !view.morphing) {
            /* The morph lasts ~72 ticks and the ejection is where v1 bots
             * got squished: the horizon must reach past it. */
            search_add(cands, &n, BOT_AIM_UP, BOT_PLATFORM, 0, BOT_AIM_UP, SEARCH_HOLD, H+96);
            search_add(cands, &n, bot_aim_byte(dx, dy), BOT_PLATFORM, 0, bot_aim_byte(dx, dy), SEARCH_HOLD, H+96);
        }
    }
    SearchContext ctx; memset(&ctx, 0, sizeof(ctx)); ctx.stalled=stalled;
    if (!view.morphing && !stalled && !search_shot_exists(&view)) ctx.have_spot=bot_firing_spot(&view, &ctx.sx, &ctx.sy);
    /* Passivity (the v2 bots' foe_still): a foe that has neither moved nor
     * morphed for 20 decisions is modelled as it behaves - standing, no shot,
     * no dodge (SearchFoe.passive) - and the cover penalty against its ready
     * bow is dropped. Without it the deterministic search camped out of an
     * idle foe's line of sight until sudden death, or shot at where the
     * modelled foe would have walked to; pool-v3 opponents camp too. */
    double moved=bot_len(view.fx-bot->foe_prev_x, view.fy-bot->foe_prev_y);
    bot->foe_still=(moved<0.15 && !view.foe_morphing && bot->decisions>1)?bot->foe_still+1:0;
    bot->foe_prev_x=view.fx; bot->foe_prev_y=view.fy;
    ctx.foe_passive=bot->foe_still>=20;
    /* A charging foe is scored under both release timings (now / at full
     * charge) and the candidate keeps the worse: the release-now model alone
     * had the bot start a 64-tick full charge at 3.6 u from a foe whose real
     * charge landed first. */
    static int debug=-1;
    if (debug<0) debug=getenv("BOPL_BOT_DEBUG")?atoi(getenv("BOPL_BOT_DEBUG")):0;
    int modes=view.foe_charging?2:1, best=0;
    for (int i=0; i<n; i++) {
        double score=1e9, part[3]={9, 9, 9};
        /* My arrow in the rollout: the foe gets its best dodge in hindsight
         * (min over the dodge variants), so a shot only scores as a kill when
         * the foe cannot escape it - down, mid-air without a dash, or on
         * cooldown. That is the window humans attack in. */
        int shot=(cands[i].first.buttons&BOT_ARROW) || cands[i].macro==SEARCH_DASH_SHOOT || (held&BOT_ARROW);
        int dodges=shot && !ctx.foe_passive?2:1;
        for (int mode=0; mode<modes; mode++)
            for (int dodge=0; dodge<dodges; dodge++) {
                double r=search_rollout(bot, state, me, &cands[i], hold_left, held, hold_run, &ctx, mode, dodge);
                if (dodge==0) part[mode]=r;
                score=fmin(score, r);
            }
        /* A long commitment (latching the arrow for 8+ decisions) is also
         * scored against a foe that dashes in and shoots: 22 of 40 losses in
         * a 120-game set died inside a full charge the base model called safe. */
        if (hold_left==0 && (cands[i].first.buttons&BOT_ARROW) && cands[i].first.hold>=3)
            score=fmin(score, part[2]=search_rollout(bot, state, me, &cands[i], hold_left, held, hold_run, &ctx, 2, 0));
        /* Commitment margin: a latch of L decisions pins the buttons for L
         * decisions the foe model may get wrong; it has to earn its keep. */
        if (hold_left==0) score-=0.004*(BOPL_HOLD_LENGTHS[cands[i].first.hold]-1);
        cands[i].score=score;
        if (cands[i].score>cands[best].score) best=i;
        if (debug>1) fprintf(stderr, "    c%02d aim%3d btn%x hold%d macro%d H%d -> %.3f (%.2f %.2f %.2f)\n", i, cands[i].first.aim,
                             cands[i].first.buttons, cands[i].first.hold, cands[i].macro, cands[i].horizon, score, part[0], part[1], part[2]);
    }
    if (debug) fprintf(stderr, "[search %d] d%03d me(%.1f,%.1f) foe(%.1f,%.1f) spot%d(%.1f,%.1f) cands %d best %d aim%3d btn%x hold%d score %.3f\n",
                       me, bot->decisions, view.x, view.y, view.fx, view.fy, ctx.have_spot, ctx.sx, ctx.sy, n, best,
                       cands[best].first.aim, cands[best].first.buttons, cands[best].first.hold, cands[best].score);
    if (debug>2 && hold_left==0) {
        search_trace=1;
        double r=search_rollout(bot, state, me, &cands[best], hold_left, held, hold_run, &ctx, 0, 0);
        search_trace=0;
        fprintf(stderr, "      -> mode0 %.3f\n", r);
    }
    out[0]=cands[best].first.aim; out[1]=(uint8_t)(cands[best].first.buttons&15u);
    out[2]=(cands[best].first.buttons&BOT_ARROW)?cands[best].first.hold:0;
}

static void bopl_bot_decide_v2(BoplBot *bot, const BoplRlState *state, int player_index,
                               int hold_left, uint16_t held, const int hold_run[4], uint8_t out[3]) {
    switch (bot->kind) {
    case BOPL_BOT_ARCHER_V2: case BOPL_BOT_PURSUER_V2: case BOPL_BOT_EVASIVE_V2: case BOPL_BOT_MIXED_V2:
        /* bopl_bots_v2_play.h: counts its own decisions, applies the mask parity and records last. */
        bot2_decide(bot, state, player_index, hold_left, held, hold_run, out); return;
    case BOPL_BOT_UNIFORM:
        /* elo_ladder.Random inside the environment: a uniform aim byte and button
         * state, hold index 0, no mask. Every level plays the same; any kit, any map. */
        bot->decisions++;
        out[0]=(uint8_t)(bot_rand(bot)&255u); out[1]=(uint8_t)(bot_rand(bot)&15u); out[2]=0;
        return;
    default: break;
    }
    bot->decisions++;
    out[0]=BOT_AIM_UP; out[1]=hold_left>0?(uint8_t)held:0; out[2]=0;
    switch (bot->kind) {
    case BOPL_BOT_SEARCH: bot_search(bot, state, player_index, hold_left, held, hold_run, out); break;
    default: break;  /* unknown kind: stand */
    }
    /* Mask parity: the decision after an arrow latch releases the arrow; a
     * latched decision only carries its aim. */
    if (hold_left>0) { out[1]=(uint8_t)held; out[2]=0; }
    else if (hold_run[BOPL_BUTTON_ARROW]>0) out[1]&=(uint8_t)~BOT_ARROW;
    if (!(out[1]&BOT_ARROW)) out[2]=0;
    memcpy(bot->last, out, 3);
}

#endif
