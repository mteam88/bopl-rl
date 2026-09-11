#ifndef BOPL_BOTS_V2_PLAY_H
#define BOPL_BOTS_V2_PLAY_H
/* v2 scripted bots: archer-v2 / pursuer-v2 / evasive-v2 / mixed-v2 (kinds 5-8).
 *
 * Built on the v1 helpers of bopl_bots.h (never edited) plus the layers below,
 * which fix the three measured v1 failure modes (loss taxonomy, 2026-09-11,
 * 816 games): 66% of v1 non-wins were platform squishes, 10% own arrows (the
 * shoot->dash pattern and returning mortars), and ~15% attacking into a ready
 * or aimed bow. Every game fact used here is cited to the Unity source in
 * Assets/Scripts/Assembly-CSharp and the rollout data (native/data/rollout.json,
 * Q32.32 raw values / 2^32):
 *
 * Squish  PlayerCollision.cs:326-336,:454 (CauseOfDeath.Squished): a grounded
 *         player whose attached ground is a different body than a colliding
 *         StickyRoundedRectangle dies when that rectangle contains its body
 *         centre (platformSquishPointOffset is zero on the player prefab).
 *         Rust: native/src/rollout/mod.rs:2334-2353. A morph body is not
 *         squishable (PlatformTransform.cs:143 sets isSquishable=false on the
 *         morph's PlayerCollision only), the slime that reappears at ejection is.
 * Spawn   PlatformTransform.cs:132 Anticipation anim (8+8+8 = 24 ticks) then
 *         CreatePlatform (:139-165): the StickyRect spawns AT the frozen body
 *         position with resizable_platform extents (1.1, 0.78) and, during the
 *         Rotation state (Enter anim 4x7 = 28 ticks, :162), a radius that
 *         follows enter_scale_curve (:276): 2.0 -> 3.27 at 0.236 s -> 3.0, while
 *         its up vector slerps toward the owner's aim (:280, rotation_speed 50).
 *         So the fresh platform is a spinning 8.2 x 7.6 rounded box: any point
 *         within sqrt(1.1^2+0.78^2)+3.27 = 4.62 u of the plant point can be
 *         inside it. Size state (:291-416): each half extent grows by up to
 *         radius*SizeChangeSpeed*dt = 3*5/60 = 0.25 u/tick with the anchored
 *         corner fixed, so the far edge advances up to 30 u/s toward the cursor
 *         (cursorSensitivity 25 u/s along the aim), capped by maxArea 250
 *         (:358-368; ~14 u half width along one axis) and maxSizeTime 4 s
 *         (:407). Exit (:418-428): the body rides up the platform's up vector
 *         by radius+y = 3.78 u (EjectionOffset is 0 in rollout.json) and the
 *         slime reappears there, grounded on its own platform.
 * Dash    Dash.cs:245-258: after pauseLength 0.01 s (1 tick) performDash
 *         raycasts dashLength 13 u along the aim on a mask that includes the
 *         Projectile layer (Rust mod.rs:1230-1273): an arrow on the ray kills
 *         the dasher unless it is its own arrow that cannot hurt its owner yet.
 *         Exit slide: selfImposedVelocity = aim * leaveSpeed 60 (Dash.cs:228).
 * Arrow   BowTransform.cs:183 DelayedEnableHurtOwner(TimeBeforeArrowsHurtOwner
 *         * scale / |v|) with time_before_arrows_hurt_owner = 5.0: owner
 *         immunity ends once the arrow has travelled 5 u (4 ticks at 75 u/s).
 *         Bow morph airborne: gravity_modifier 0.03 and velocity x0.1, capped
 *         at maxSlowMoSpeed 9 while loading (BowTransform.cs:93-97, :247-250):
 *         a bot may enter the bow at any airborne speed (dash slide included);
 *         the arrow inherits only the residual selfImposedVelocity (:182).
 *         Hurtbox: PlatformTransform.cs:141 disables the platform morph's
 *         hurtbox at CreatePlatform, so a foe in platform morph can only be hit
 *         during its first 24 ticks.
 * Cooldowns  timers[s] reset at activation and count only outside abilities
 *         (mod.rs:1062-1097); ready when timer > cooldown (bow 2.5 s, dash 4 s,
 *         platform 6 s). The bow hurtbox entity (layer 8, owner = player) carries
 *         the bow aim as its up vector (mod.rs:2680-2695); ability_flags[bow]
 *         bits 4-6 are loadingFrame and bit 7 hasFired (mod.rs:2797-2806).
 */

#define BOT2_PLAT_EX 1.1
#define BOT2_PLAT_EY 0.78
#define BOT2_PLAT_R 3.0
#define BOT2_PLAT_R_PEAK 3.27
#define BOT2_PLAT_DISC 4.62            /* farthest point of the spinning fresh platform from its plant point */
#define BOT2_PLAT_EJECT 5.78           /* ejection along the platform up: radius + half height + EjectionOffset = 3+0.78+2 (PlatformTransform.cs:418-428, :95) */
#define BOT2_PLAT_ANTICIPATION 24
#define BOT2_PLAT_ENTER 28
#define BOT2_PLAT_EDGE_SPEED 30.0      /* far-edge advance while sizing, u/s */
#define BOT2_PLAT_HALF_MAX 17.0        /* maxArea cap: half width 14 + radius 3 along one axis */
#define BOT2_HAZARD_MARGIN 3.0         /* keep this much beyond the spawn disc / live rect */
#define BOT2_ARROW_OWNER_SAFE 5.0
#define BOT2_DASH_LEAVE 60.0
#define BOT2_READY_BOW_RANGE 8.5       /* do not charge or close inside this while the foe's bow is ready */
#define BOT2_PASSIVE_BOW 45            /* decisions a ready bow may idle before the foe counts as passive */

/* ---- v2 view: what v1's BotView lacks ------------------------------------ */
typedef struct {
    int foe_bow_ready, foe_dash_ready, foe_platform_ready;
    double foe_bow_left, foe_dash_left;      /* seconds until ready (0 if ready) */
    int foe_loading;                         /* bow loadingFrame while the foe charges */
    int have_foe_aim; double foe_aim_x, foe_aim_y;
    double foe_aim_cos;                      /* cos(angle between the foe's bow aim and the direction to me) */
    int foe_aiming_me;                       /* charging with the aim on me at a range it can reach */
    int ground_wall;                         /* wall index I stand on (-1 airborne / unknown) */
    /* Foe platform hazard. */
    int haz_active;                          /* foe is in platform morph */
    int haz_wall;                            /* its platform's wall index once spawned (-1 during anticipation) */
    int haz_mine;                            /* I stand on that very platform (it cannot squish its own rider) */
    double haz_x, haz_y;                     /* plant point */
    double haz_dist;                         /* my distance to the hazard surface (negative: inside) */
    double haz_ticks;                        /* ticks the morph has lasted */
    uint32_t wall_id[48], wall_hier[48];     /* entity id / hierarchy per view->walls index */
    int passive;                             /* foe kept a ready bow idle for BOT2_PASSIVE_BOW decisions */
} BotView2;

/* A player's ability hurtbox entity by prefab slot offset (mod.rs:584-600:
 * hierarchy 100000000 + {6 body, 15 dash, 22 bow} + 30*player). */
static const BoplRlEntity *bot2_owner_entity(const BoplRlState *state, uint32_t owner, uint32_t offset) {
    for (uint32_t i=0; i<state->header.entity_count; i++) {
        const BoplRlEntity *e=&state->entities[i];
        if (e->owner==owner && e->layer==8 && (e->flags&1) && !(e->flags&4)
                && e->hierarchy>=100000000u && (e->hierarchy-100000000u)%30==offset) return e;
    }
    return NULL;
}

static void bot2_view(BotView2 *v, const BotView *view, BoplBot *bot) {
    memset(v, 0, sizeof(*v));
    v->ground_wall=-1; v->haz_wall=-1;
    const BoplRlState *state=view->state;
    /* Wall bookkeeping: same filter and order as bot_view's wall loop. */
    int n=0;
    for (uint32_t i=0; i<state->header.entity_count && n<48; i++) {
        const BoplRlEntity *e=&state->entities[i];
        if (!(e->flags&1) || (e->flags&4)) continue;
        if (e->kind==2 && (e->layer==11 || e->layer==16 || e->layer==21)) { v->wall_id[n]=e->id; v->wall_hier[n]=e->hierarchy; n++; }
    }
    if (view->grounded) {
        /* The wall I touch whose surface normal points up (v1's
         * bot_support_wall skips steep slabs whose top is far above me). */
        double best=BOT_SLIME_RADIUS+0.35;
        for (int i=0; i<view->wall_count; i++) {
            const BotWall *w=&view->walls[i];
            double sd=bot_wall_sdf(w, view->x, view->y);
            if (sd>=best) continue;
            if (bot_wall_sdf(w, view->x, view->y+0.15)-bot_wall_sdf(w, view->x, view->y-0.15)<=0) continue;
            best=sd; v->ground_wall=i;
        }
    }
    const BoplRlSlime *foe=view->foe_slime;
    if (foe) {
        v->foe_bow_ready=foe->timers[BOT_SLOT_BOW]>foe->cooldowns[BOT_SLOT_BOW];
        v->foe_dash_ready=foe->timers[BOT_SLOT_DASH]>foe->cooldowns[BOT_SLOT_DASH];
        v->foe_platform_ready=foe->timers[BOT_SLOT_PLATFORM]>foe->cooldowns[BOT_SLOT_PLATFORM];
        double l=bot_fix(foe->cooldowns[BOT_SLOT_BOW])-bot_fix(foe->timers[BOT_SLOT_BOW]); v->foe_bow_left=l>0?l:0;
        l=bot_fix(foe->cooldowns[BOT_SLOT_DASH])-bot_fix(foe->timers[BOT_SLOT_DASH]); v->foe_dash_left=l>0?l:0;
        if (view->foe_morphing) v->foe_bow_ready=v->foe_dash_ready=v->foe_platform_ready=0;
        if (view->foe_charging) {
            v->foe_loading=(int)((foe->ability_flags[BOT_SLOT_BOW]>>4)&7);
            const BoplRlEntity *bow=bot2_owner_entity(state, view->foe->id, 22);
            if (bow) {
                double ux=bot_fix(bow->up_x), uy=bot_fix(bow->up_y), m=bot_len(ux, uy);
                if (m>1e-6) { v->have_foe_aim=1; v->foe_aim_x=ux/m; v->foe_aim_y=uy/m; }
            }
        }
    }
    double dx=view->x-view->fx, dy=view->y-view->fy, range=bot_len(dx, dy);
    if (v->have_foe_aim && range>1e-6) {
        v->foe_aim_cos=(v->foe_aim_x*dx+v->foe_aim_y*dy)/range;
        /* A lob aims above its target: the quick shot's launch angle at 5 u is
         * ~25 deg above the line, so accept a generous cone. */
        v->foe_aiming_me=v->foe_aim_cos>0.72 && range<15.0;
    }
    /* Passivity of the foe's bow: ready and unused for a while. */
    if (v->foe_bow_ready) bot->v2.foe_bow_idle++; else bot->v2.foe_bow_idle=0;
    v->passive=bot->v2.foe_bow_idle>=BOT2_PASSIVE_BOW || bot->foe_still>=20;
    if (view->foe_charging) bot->v2.foe_charge_decisions++; else bot->v2.foe_charge_decisions=0;
    /* Foe platform morph: the plant point is the frozen body; once spawned, the
     * platform is the spawned wall (hierarchy >= 300000000, mod.rs:1701) whose
     * centre the morph body sits on (Exit rides up its up axis). */
    if (view->foe_platforming) {
        v->haz_active=1;
        if (bot->v2.foe_plat_decisions==0) { bot->v2.foe_plat_x=view->fx; bot->v2.foe_plat_y=view->fy; bot->v2.foe_plat_id=0; }
        bot->v2.foe_plat_decisions++;
        v->haz_ticks=bot->v2.foe_plat_decisions*4.0;
        v->haz_x=bot->v2.foe_plat_x; v->haz_y=bot->v2.foe_plat_y;
        int best=-1; double best_d=1e9;
        for (int i=0; i<view->wall_count && i<n; i++) {
            if (v->wall_hier[i]<300000000u) continue;
            const BotWall *w=&view->walls[i];
            double px=view->fx-w->cx, py=view->fy-w->cy;
            double along=px*w->ux+py*w->uy, across=px*w->rx+py*w->ry;
            if (fabs(across)>0.6 || along<-0.6 || along>BOT2_PLAT_EJECT+1.5) continue;
            if (v->wall_id[i]==bot->v2.my_plat_id) continue;
            double d=fabs(across)+fabs(along);
            if (d<best_d) { best_d=d; best=i; }
        }
        if (best>=0) {
            v->haz_wall=best; bot->v2.foe_plat_id=v->wall_id[best];
            v->haz_x=view->walls[best].cx; v->haz_y=view->walls[best].cy;
            v->haz_dist=bot_wall_sdf(&view->walls[best], view->x, view->y);
            v->haz_mine=v->ground_wall==best;
        } else {
            v->haz_dist=bot_len(view->x-v->haz_x, view->y-v->haz_y)-BOT2_PLAT_DISC;
        }
    } else {
        bot->v2.foe_plat_decisions=0;
        v->haz_dist=1e9;
    }
    /* Remember my own platform while I morph on it. */
    if (view->morphing && !view->charging && (view->my_slime->ability_flags[BOT_SLOT_PLATFORM]&8)) {
        for (int i=0; i<view->wall_count && i<n; i++) {
            if (v->wall_hier[i]<300000000u) continue;
            const BotWall *w=&view->walls[i];
            if (bot_len(view->x-w->cx, view->y-w->cy)<BOT2_PLAT_EJECT+1.0) { bot->v2.my_plat_id=v->wall_id[i]; break; }
        }
    }
}

/* ---- arrows: real dash ray, own-walk-aware ETA -------------------------- */
/* Would a dash along (dx,dy) cross an arrow? Dash.cs:245-258 / mod.rs:1230-1273:
 * the 13 u ray is cast one tick after the press against every arrow's circle
 * (radius 0.61), own arrows included once they travelled 5 u
 * (BowTransform.cs:183). The press lands 1-4 ticks from now and the ray is
 * cast a tick later, so the arrow is tested at its positions over the next 6
 * ticks; own arrows are lethal from birth here (the immunity is 4 ticks for a
 * full-charge shot, shorter than the decision itself). The exit slide
 * (leaveSpeed 60, ~7 u) is checked against every arrow too. */
static int bot2_dash_ray_clear(const BotView *view, double dx, double dy) {
    double n=bot_len(dx, dy); if (n<1e-6) return 0;
    dx/=n; dy/=n;
    double wall=bot_raycast(view, view->x, view->y, dx, dy, BOT_DASH_LENGTH);
    double end=fmin(wall, BOT_DASH_LENGTH);
    for (int i=0; i<view->arrow_count; i++) {
        double ax=view->arrows[i].x, ay=view->arrows[i].y, avx=view->arrows[i].vx, avy=view->arrows[i].vy;
        for (int t=0; t<=6; t++) {
            double rx=ax-view->x, ry=ay-view->y;
            double along=rx*dx+ry*dy, perp=fabs(rx*dy-ry*dx);
            if (along>-1.2 && along<end+1.2 && perp<BOT_ARROW_RADIUS+BOT_SLIME_RADIUS+0.3) return 0;
            avy-=BOT_ARROW_G*BOT_TICK; ax+=avx*BOT_TICK; ay+=avy*BOT_TICK;
        }
        /* Slide after the teleport: ~7 u along the aim over the next ~20 ticks. */
        if (wall>=BOT_DASH_LENGTH) {
            ax=view->arrows[i].x; ay=view->arrows[i].y; avx=view->arrows[i].vx; avy=view->arrows[i].vy;
            for (int t=1; t<=26; t++) {
                avy-=BOT_ARROW_G*BOT_TICK; ax+=avx*BOT_TICK; ay+=avy*BOT_TICK;
                if (t<5) continue;
                double s=7.0*(1.0-pow(0.85, t-4));
                double mx=view->x+dx*(end+s), my=view->y+dy*(end+s);
                if (bot_len(ax-mx, ay-my)<2.2) return 0;
                if (bot_sdf(view, ax, ay)<BOT_ARROW_RADIUS*0.5) break;
            }
        }
    }
    return 1;
}
static int bot2_dash_safe(const BotView *view, double dx, double dy) {
    return bot_dash_safe(view, dx, dy) && bot2_dash_ray_clear(view, dx, dy);
}

/* Earliest arrow impact (ticks) if I move with horizontal velocity mvx
 * (grounded: my walk; airborne: steering added to the fall). Unlike
 * bot_arrow_eta this sees my own intended walk (a v1 bot walked under its own
 * returning mortar: eta=inf, then 5) and looks 90 ticks out for lobs. */
/* Ground tangent under me (rightward), from the sdf gradient of the wall I
 * stand on; world axes when airborne/unknown. */
static void bot2_ground_tangent(const BotView *view, const BotView2 *v, double *tx, double *ty) {
    *tx=1; *ty=0;
    if (!view->grounded || v->ground_wall<0) return;
    const BotWall *w=&view->walls[v->ground_wall];
    double nx=bot_wall_sdf(w, view->x+0.15, view->y)-bot_wall_sdf(w, view->x-0.15, view->y);
    double ny=bot_wall_sdf(w, view->x, view->y+0.15)-bot_wall_sdf(w, view->x, view->y-0.15);
    double n=bot_len(nx, ny);
    if (n<1e-6 || ny<=0) return;
    nx/=n; ny/=n;
    *tx=ny; *ty=-nx;
}

/* Grounded walk velocity a move input produces (mod.rs:804-812, SlimeController
 * flag5): the stick must align with the ground tangent by >= 0.2, then the
 * slime runs along the tangent at BOT_WALK. On a steep slab that is mostly
 * vertical motion: the ETA below must see it, or a lobbed shot fired uphill is
 * walked into (the "own-arrow-fresh" family). */
static void bot2_walk_velocity(const BotView *view, const BotView2 *v, uint8_t aim, double *wvx, double *wvy) {
    *wvx=0; *wvy=0;
    if (!aim) return;
    double ang=((int)aim-1)/255.0*2*M_PI, dx=cos(ang), dy=sin(ang), tx, ty;
    bot2_ground_tangent(view, v, &tx, &ty);
    double align=dx*tx+dy*ty;
    if (fabs(align)<0.2) return;
    double s=align>=0?BOT_WALK:-BOT_WALK;
    *wvx=tx*s; *wvy=ty*s;
}

/* Ticks until an arrow reaches me given the motion I am about to make:
 * grounded (mvx,mvy) is my walk along the ground (bot2_walk_velocity),
 * airborne mvx is the steer target. Own arrows count (immunity ends 5 u out,
 * BowTransform.cs:183). 1e9 when none within horizon. */
static double bot2_arrow_eta(const BotView *view, double mvx, double mvy, int horizon, double *dir_x) {
    double best=1e9;
    for (int i=0; i<view->arrow_count; i++) {
        double x=view->arrows[i].x, y=view->arrows[i].y, vx=view->arrows[i].vx, vy=view->arrows[i].vy;
        /* A shot just leaving my bow is harmless until it has flown 5 u
         * (BowTransform.cs:183); it left from about where I stand, so the
         * remaining immune path is 5 u minus its current offset. */
        double immune=view->arrows[i].leaving?BOT2_ARROW_OWNER_SAFE-bot_len(x-view->x, y-view->y):0, path=0;
        double mx=view->x, my=view->y, cvy=view->grounded?mvy:view->vy, cvx=view->grounded?mvx:view->vx;
        int grounded=view->grounded, stopped=0;
        for (int t=1; t<=horizon; t++) {
            vy-=BOT_ARROW_G*BOT_TICK; x+=vx*BOT_TICK; y+=vy*BOT_TICK; path+=bot_len(vx, vy)*BOT_TICK;
            if (!grounded) { bot_fall_tick(&cvy); cvx+=mvx*0.05; if (cvx>19) cvx=19; if (cvx<-19) cvx=-19; }
            if (!stopped) {
                double nx=mx+cvx*BOT_TICK, ny=my+cvy*BOT_TICK;
                /* A walker stops at a wall and does not leave its slab upward. */
                if (grounded && (bot_sdf(view, nx, ny)<BOT_SLIME_RADIUS-0.25 || bot_sdf(view, nx, ny)>BOT_SLIME_RADIUS+0.9)) stopped=1;
                else { mx=nx; my=ny; }
            }
            if (!grounded && cvy<0 && bot_sdf(view, mx, my)<BOT_SLIME_RADIUS) { grounded=1; cvx=mvx; cvy=0; stopped=1; }
            if (bot_sdf(view, x, y)<BOT_ARROW_RADIUS*0.5) break;
            if (path>=immune && bot_len(x-mx, y-my)<1.9) { if (t<best) { best=t; if (dir_x) *dir_x=vx>=0?1:-1; } break; }
        }
    }
    return best;
}

/* ---- platform press safety ------------------------------------------------ */
/* May I press platform here? The ejected slime reappears 3.78 u above the
 * plant point grounded on its own platform (PlatformTransform.cs:420-428), so
 * any other wall containing that point squishes it (PlayerCollision.cs:326);
 * the fresh platform's spinning disc must also not overlap the foe's growing
 * one (both rectangles are walls to the other's rider). */
static int bot2_platform_press_safe(const BotView *view, const BotView2 *v) {
    double px=view->x+(view->grounded?0:view->vx*BOT_TICK), py=view->y+(view->grounded?0:view->vy*BOT_TICK);
    double ey=py+BOT2_PLAT_EJECT;
    if (v->haz_active && bot_len(px-v->haz_x, py-v->haz_y)<BOT2_PLAT_DISC*2+2.0) return 0;
    /* The slime rides the up axis to +5.78 u (flat platform: straight up),
     * then drops onto its platform's top at +3.78+0.76. Every point on that
     * path must be clear of any other rect by a slime radius plus margin, and
     * nothing may hang over the plant point within reach. */
    for (double k=2.0; k<=BOT2_PLAT_EJECT+0.01; k+=0.9)
        if (bot_sdf(view, px, ey-BOT2_PLAT_EJECT+k)<BOT_SLIME_RADIUS+0.7) return 0;
    if (bot_raycast(view, px, py, 0, 1, BOT2_PLAT_EJECT+1.8)<BOT2_PLAT_EJECT+1.8) return 0;
    if (bot_raycast(view, px-1.2, py, 0, 1, BOT2_PLAT_EJECT+1.0)<BOT2_PLAT_EJECT+1.0) return 0;
    if (bot_raycast(view, px+1.2, py, 0, 1, BOT2_PLAT_EJECT+1.0)<BOT2_PLAT_EJECT+1.0) return 0;
    if (ey>BOT_BLAST_Y-4) return 0;
    return 1;
}

/* ---- platform hazard layer -------------------------------------------------- */
/* Retreat target: a standable spot at least min_dist from (hx,hy), preferring
 * short travel and higher ground; 0 if none certifies as reachable. */
static int bot2_retreat_spot(const BotView *view, double hx, double hy, double min_dist, double *tx, double *ty) {
    int found=0; double best=-1e18;
    for (int i=0; i<view->wall_count; i++) {
        const BotWall *w=&view->walls[i];
        if (!w->standable) continue;
        double span=fmax(w->spanx-0.8, 0);
        for (int k=-2; k<=2; k++) {
            double px=w->cx+span*k/2.0, py;
            if (!bot_surface_y(view, w, px, &py)) continue;
            double d=bot_len(px-hx, py-hy);
            if (d<min_dist) continue;
            if (!bot_can_reach(view, px, py)) continue;
            double travel=bot_len(px-view->x, py-view->y);
            /* Height separation counts extra: sizing reaches ~19 u along one
             * axis but only ~11 u diagonally (maxArea). */
            double score=fmin(d, 16.0)+0.5*fmin(fabs(py-hy), 8.0)-0.45*travel+(py>view->y+1?1.5:0)-(py<view->y-3?3:0);
            if (score>best) { best=score; *tx=px; *ty=py; found=1; }
        }
    }
    return found;
}

/* Escape the foe's platform if it can reach me. Fills act when it acts.
 *
 * Timeline (PlatformTransform.cs): 24 ticks of anticipation with no rect, 28
 * ticks of Rotation with the 4.62 u spinning disc, then Size for as long as
 * the button stays down (<= 4 s): the far edge advances up to 30 u/s toward
 * the cursor (measured in the taxonomy replays: half extents +15 u/s, centre
 * 28.6 u/s, half width up to 16+3 u). Walking (19 u/s) cannot outrun a
 * sizing edge, so the layer leaves the disc during anticipation, prefers a
 * different height (diagonal growth is capped near 11 u by maxArea), and once
 * an edge is observed closing in makes itself airborne: an airborne player is
 * not grounded and cannot be squished, and one landing on the platform rides
 * it (its own ground cannot squish it). */
static void bot2_platform_threat(const BotView *view, const BotView2 *v, BoplBot *bot, BotAction *act, int *dbg) {
    int was=bot->v2.in_hazard;
    bot->v2.in_hazard=0;
    if (!v->haz_active) { bot->v2.haz_prev_dist=1e9; return; }
    double approach=bot->v2.haz_prev_dist<1e8?bot->v2.haz_prev_dist-v->haz_dist:0;  /* u closed per decision */
    bot->v2.haz_prev_dist=v->haz_dist;
    if (v->haz_mine) {
        /* Riding the foe's platform: safe from it, but its top rises with any
         * upward growth and a ceiling would squish me. Leave low corridors. */
        if (bot_raycast(view, view->x, view->y, 0, 1, 7.0)>=7.0) return;
    }
    double margin=BOT2_HAZARD_MARGIN+(was?2.0:0);
    if (v->haz_ticks>=BOT2_PLAT_ANTICIPATION+BOT2_PLAT_ENTER-8) margin+=1.5;       /* sizing possible */
    if (approach>0.3) margin+=approach*3.0;                                          /* an edge is closing in */
    if (v->haz_dist>=margin && !v->haz_mine) return;
    bot->v2.in_hazard=1;
    /* Away: the sdf gradient of the live rect, else away from the plant point. */
    double gx, gy;
    if (v->haz_wall>=0) {
        const BotWall *w=&view->walls[v->haz_wall];
        gx=bot_wall_sdf(w, view->x+0.2, view->y)-bot_wall_sdf(w, view->x-0.2, view->y);
        gy=bot_wall_sdf(w, view->x, view->y+0.2)-bot_wall_sdf(w, view->x, view->y-0.2);
    } else { gx=view->x-v->haz_x; gy=view->y-v->haz_y; }
    double away=gx>=0?1:-1;
    if (fabs(gx)<0.05*fabs(gy)+1e-6) away=view->x>=view->fx?1:-1;
    if (!view->grounded) {
        /* Airborne: un-squishable now; land as far from the plant point as any steer allows. */
        int best=0; double best_d=-1;
        for (int s=-1; s<=1; s++) {
            double lx, ly;
            if (!bot_landing_steer(view, s, &lx, &ly)) continue;
            double d=bot_len(lx-v->haz_x, ly-v->haz_y);
            if (d>best_d) { best_d=d; best=s; }
        }
        bot_act(act, best?bot_aim_byte(best, 0):BOT_AIM_UP, 0, 0); *dbg=4; return;
    }
    double tx, ty;
    int spot=bot2_retreat_spot(view, v->haz_x, v->haz_y, BOT2_PLAT_DISC+BOT2_HAZARD_MARGIN+4.0, &tx, &ty);
    /* Imminent: the rect is about to contain me. Dash clears 13 u at once;
     * otherwise a jump makes me un-squishable while airborne. */
    int imminent=v->haz_dist<1.6 || (approach>0.3 && v->haz_dist<1.0+approach*1.8);
    if (imminent) {
        if (view->dash_ready && spot && bot2_dash_safe(view, tx-view->x, ty-view->y+1.0))
            { bot_act(act, bot_aim_byte(tx-view->x, ty-view->y+1.0), BOT_DASH, 0); *dbg=1; return; }
        if (view->dash_ready && bot2_dash_safe(view, away, 0.35))
            { bot_act(act, bot_aim_byte(away, 0.35), BOT_DASH, 0); *dbg=1; return; }
        bot_act(act, bot_aim_byte(away, 0.25), BOT_JUMP, 0); *dbg=2; return;
    }
    uint8_t aim=0; uint16_t buttons=0;
    if (spot) {
        bot_nav(view, bot, tx, ty, &aim, &buttons);
        /* bot_nav may stand ("arrived") when the spot is where I am: keep walking away. */
        if (aim==BOT_AIM_UP && !buttons) aim=bot_aim_byte(away, 0);
    } else {
        double ahead=bot_drop(view, view->x+away*2.5, view->y+0.5);
        if (ahead>4.5 && view->dash_ready && bot2_dash_safe(view, away, 0.3)) { bot_act(act, bot_aim_byte(away, 0.3), BOT_DASH, 0); *dbg=1; return; }
        if (ahead>4.5) away=-away;
        aim=bot_aim_byte(away, 0);
    }
    bot_act(act, aim, buttons, 0); *dbg=(buttons&BOT_DASH)?1:(buttons&BOT_JUMP)?2:4;
}

/* ---- arrow dodge (v1 rule with the own-walk-aware ETA) ---------------------- */
static void bot2_dodge(const BotView *view, BoplBot *bot, BotAction *act, double *eta_out, int *dbg) {
    double dir=1;
    double eta=bot_arrow_eta(view, &dir);
    *eta_out=eta;
    int window=bot->level>=BOPL_BOT_HARD?26:bot->level==BOPL_BOT_MEDIUM?18:10;
    if (eta>window) { bot->threat_seen=0; return; }
    bot->threat_seen++;
    int wait=bot->level>=BOPL_BOT_HARD?0:bot->level==BOPL_BOT_MEDIUM?1:2;
    if (bot->threat_seen<=wait) return;
    double away=-dir;
    if (view->grounded) {
        if (view->dash_ready && eta<12 && bot2_dash_safe(view, away, 0.35))
            { bot_act(act, bot_aim_byte(away, 0.35), BOT_DASH, 0); *dbg=1; return; }
        double ahead=bot_drop(view, view->x+away*2.2, view->y+0.5);
        if (ahead>4.5) away=-away;
        /* Jump only when the launch clears every arrow; else run. */
        if (!bot_launch_hits_arrow(view, -22) && !bot_launch_hits_arrow(view, 0) && !bot_launch_hits_arrow(view, 22))
            { bot_act(act, bot_aim_byte(away, 0.2), BOT_JUMP, 0); *dbg=2; return; }
        double other=-away;
        double e_a=bot2_arrow_eta(view, away*19, 0, 40, NULL), e_b=bot2_arrow_eta(view, other*19, 0, 40, NULL);
        bot_act(act, bot_aim_byte(e_a>=e_b?away:other, 0.2), 0, 0); *dbg=4; return;
    }
    if (view->dash_ready && eta<12 && bot2_dash_safe(view, away, -0.2))
        { bot_act(act, bot_aim_byte(away, -0.2), BOT_DASH, 0); *dbg=1; return; }
    if (view->platform_ready && eta<8)
        { bot_act(act, BOT_AIM_UP, BOT_PLATFORM, 0); *dbg=3; return; }  /* freeze: hurtbox vanishes */
    bot_act(act, bot_aim_byte(away, 0), 0, 0); *dbg=4;
}

/* ---- shooting with cooldown awareness ---------------------------------------- */
/* Heaviest charge worth starting now. A charge is a hovering target for its
 * whole fire delay (12/40/72 ticks), so against a READY bow inside 16 u only
 * quick/mid shots are taken, and none inside BOT2_READY_BOW_RANGE; against a
 * bow on cooldown the charge may last until roughly when the foe's bow returns. */
static int bot2_max_hold(const BotView *view, const BotView2 *v, double range) {
    if (view->foe_platforming) return -1;                      /* hurtbox gone after 24 ticks; the arrow hits the platform */
    if (v->foe_aiming_me && !v->passive) return -1;             /* the duel goes to whoever charged first: not us */
    if (view->foe_charging) return BOT_HOLD_MID;                /* aimed elsewhere: free hit, but it exits soon */
    if (v->passive) return BOT_HOLD_FULL;
    if (v->foe_bow_ready) {
        if (range<BOT2_READY_BOW_RANGE) return -1;
        if (range<16) return BOT_HOLD_MID;
        return BOT_HOLD_FULL;
    }
    /* Bow down: allowed if the charge fires before the foe can answer. */
    double budget=v->foe_bow_left*60+30;
    if (BOT_FIRE_DELAY[BOT_HOLD_FULL]<=budget || range>18) return BOT_HOLD_FULL;
    if (BOT_FIRE_DELAY[BOT_HOLD_MID]<=budget || range>=BOT2_READY_BOW_RANGE) return BOT_HOLD_MID;
    return BOT_HOLD_QUICK;
}

/* v1's shot search with a hold cap and an airborne origin correction: the bow
 * may be entered at any speed (BowTransform.cs:93-97), the morph nearly hovers,
 * and bot_charge_aim re-solves every decision from the real morph position, so
 * certify from where the morph will start (one tick of flight ahead) rather
 * than refuse as v1 does. */
static int bot2_try_shot(const BotView *view, const BotView2 *v, BoplBot *bot, int forced_release, int max_hold, BotAction *act) {
    if (!view->bow_ready || view->morphing || forced_release || max_hold<0) return 0;
    if (bot->v2.in_hazard) return 0;
    BotView tmp; const BotView *use=view;
    if (!view->grounded && bot_len(view->svx, view->svy)>14) {
        tmp=*view; tmp.x+=view->vx*BOT_TICK; tmp.y+=view->vy*BOT_TICK; tmp.svx=tmp.svy=0; use=&tmp;
        if (bot_sdf(use, tmp.x, tmp.y)<BOT_SLIME_RADIUS) return 0;
    }
    double range=bot_len(use->fx-use->x, use->fy-use->y);
    int hold=bot_pick_hold(bot, use, range);
    if (hold>max_hold) hold=max_hold;
    int desperate=bot->since_shot>40;
    uint8_t aim; double flight;
    static const int order[3]={BOT_HOLD_FULL, BOT_HOLD_MID, BOT_HOLD_QUICK};
    int start=hold==BOT_HOLD_QUICK?2:hold==BOT_HOLD_MID?1:0;
    for (int high_pass=0; high_pass<2; high_pass++) {
        int allow_high=high_pass || view->foe_charging || bot->foe_still>=8 || desperate;
        for (int k=start; k<3; k++) {
            int h=order[k];
            if (h>max_hold) continue;
            if (BOT_ARROW_SPEED[h]*BOT_ARROW_SPEED[h]/BOT_ARROW_G<=range*0.85) continue;
            if (bot_solve_shot(use, h, BOT_FIRE_DELAY[h], allow_high, 0, &aim, &flight)) {
                bot_act(act, aim, BOT_ARROW, (uint8_t)h);
                return 1;
            }
        }
        start=0;
    }
    (void)v;
    return 0;
}

/* Squish kill: my platform spawns at my centre and, 24 ticks later, covers a
 * 4.6 u disc; a grounded foe still inside it on other ground dies at spawn.
 * Only when my own ejection is safe. */
static int bot2_try_squish(const BotView *view, const BotView2 *v, BoplBot *bot, BotAction *act) {
    if (!view->platform_ready || !view->grounded || !view->foe_grounded || view->foe_morphing) return 0;
    double px, py;
    bot_predict_foe(view, BOT2_PLAT_ANTICIPATION, &px, &py);
    double d_now=bot_len(view->fx-view->x, view->fy-view->y), d_then=bot_len(px-view->x, py-view->y);
    if (d_now>4.2 || d_then>3.6 || fabs(view->fy-view->y)>2.5) return 0;
    if (v->ground_wall>=0 && bot_wall_sdf(&view->walls[v->ground_wall], view->fx, view->fy)>BOT_SLIME_RADIUS+0.5) return 0;  /* different decks */
    if (!bot2_platform_press_safe(view, v)) return 0;
    (void)bot;
    bot_act(act, BOT_AIM_UP, BOT_PLATFORM, 0);
    return 1;
}

/* Make space from a foe whose bow is ready (or aimed at me): dash away when
 * close, else walk/jump to a perch away from it. */
static void bot2_back_off(const BotView *view, const BotView2 *v, BoplBot *bot, BotAction *act) {
    double away=view->x>=view->fx?1:-1;
    double range=bot_len(view->fx-view->x, view->fy-view->y);
    if (range<7 && view->dash_ready && bot2_dash_safe(view, away, 0.5))
        { bot_act(act, bot_aim_byte(away, 0.5), BOT_DASH, 0); return; }
    double tx, ty;
    if (bot2_retreat_spot(view, view->fx, view->fy, fmax(range+3, 10.0), &tx, &ty)) {
        uint8_t aim=0; uint16_t buttons=0;
        bot_nav(view, bot, tx, ty, &aim, &buttons);
        bot_act(act, aim, buttons, 0); return;
    }
    double ahead=bot_drop(view, view->x+away*2.2, view->y+0.5);
    if (ahead>4.5) {
        /* Cornered at a lip: jump straight up (out of a flat shot's line) rather than walk into the foe. */
        bot_act(act, bot_aim_byte(-away*0.2, 1), BOT_JUMP, 0); return;
    }
    bot_act(act, bot_aim_byte(away, 0.1), 0, 0);
    (void)v;
}

/* Sidestep a bow that is charging at me: leave its aim line now instead of
 * waiting for the arrow (a quick shot at 5 u flies 16 ticks: no dodge then). */
static int bot2_evade_aim(const BotView *view, const BotView2 *v, BoplBot *bot, BotAction *act) {
    if (!v->foe_aiming_me || v->passive) return 0;
    double range=bot_len(view->fx-view->x, view->fy-view->y);
    if (range>12) return 0;
    double away=view->x>=view->fx?1:-1;
    if (view->grounded) {
        if (view->dash_ready && range<9 && bot2_dash_safe(view, away, 0.4))
            { bot_act(act, bot_aim_byte(away, 0.4), BOT_DASH, 0); return 1; }
        double ahead=bot_drop(view, view->x+away*2.2, view->y+0.5);
        if (ahead>4.5) away=-away;
        if (bot->since_jump>=5 && !bot_launch_hits_arrow(view, -22) && !bot_launch_hits_arrow(view, 0) && !bot_launch_hits_arrow(view, 22))
            { bot_act(act, bot_aim_byte(away, 0.3), BOT_JUMP, 0); return 1; }
        bot_act(act, bot_aim_byte(away, 0), 0, 0); return 1;
    }
    if (view->dash_ready && range<9 && bot2_dash_safe(view, away, -0.1))
        { bot_act(act, bot_aim_byte(away, -0.1), BOT_DASH, 0); return 1; }
    bot_act(act, bot_aim_byte(away, 0), 0, 0); return 1;
}

/* ---- personalities ------------------------------------------------------------ */
static void bot2_archer(const BotView *view, const BotView2 *v, BoplBot *bot, int forced_release, BotAction *act) {
    double range=bot_len(view->fx-view->x, view->fy-view->y);
    int max_hold=bot2_max_hold(view, v, range);
    if (bot2_evade_aim(view, v, bot, act)) return;
    if (bot2_try_squish(view, v, bot, act)) return;
    /* Too close to a ready bow: make space first. */
    if (range<BOT2_READY_BOW_RANGE && v->foe_bow_ready && !v->passive && !view->foe_morphing)
        { bot2_back_off(view, v, bot, act); return; }
    if (range<6 && !view->foe_morphing && !v->passive && v->foe_bow_left<0.6)
        { bot2_back_off(view, v, bot, act); return; }
    if (bot2_try_shot(view, v, bot, forced_release, max_hold, act)) return;
    uint8_t aim=BOT_AIM_UP; uint16_t buttons=0;
    double sx, sy;
    if (view->bow_timer_left<1.5 && view->fy>view->y-4 && !view->foe_platforming
            && bot_firing_spot(view, &sx, &sy) && bot_len(sx-view->x, sy-view->y)>1.8
            && !(v->foe_bow_ready && !v->passive && bot_len(sx-view->fx, sy-view->fy)<BOT2_READY_BOW_RANGE)) {
        bot_nav(view, bot, sx, sy, &aim, &buttons);
        bot_act(act, aim, buttons, 0); return;
    }
    if (view->bow_timer_left<1.5 && view->grounded && view->fy>view->y+5 && view->fy<view->y+11
            && fabs(view->fx-view->x)<8 && view->platform_ready && range>11 && !view->foe_platforming
            && bot2_platform_press_safe(view, v)) {
        bot_act(act, BOT_AIM_UP, BOT_PLATFORM, 0); return;
    }
    /* Do not walk into a ready bow or a platform morph; hold range instead. */
    if ((v->foe_bow_ready && !v->passive && range<BOT2_READY_BOW_RANGE+2.5) || (view->foe_platforming && range<BOT2_PLAT_DISC*2+3)) {
        bot_act(act, bot_aim_byte(view->fx>=view->x?0.2:-0.2, 1), 0, 0); return;
    }
    bot_nav_foe(view, bot, &aim, &buttons);
    bot_act(act, aim, buttons, 0);
}

static void bot2_pursuer(const BotView *view, const BotView2 *v, BoplBot *bot, int forced_release, BotAction *act) {
    double dx=view->fx-view->x, dy=view->fy-view->y, range=bot_len(dx, dy);
    int max_hold=bot2_max_hold(view, v, range);
    int foe_falling_out=!view->foe_grounded && !view->foe_morphing && view->fvy<-8 && view->fy<BOT_WATER+14;
    if (foe_falling_out && bot->level>=BOPL_BOT_MEDIUM && bot2_try_shot(view, v, bot, forced_release, BOT_HOLD_FULL, act)) return;
    if (bot2_evade_aim(view, v, bot, act)) return;
    if (bot2_try_squish(view, v, bot, act)) return;
    /* Attack window: the foe cannot answer (bow down, passive, or charging elsewhere). */
    int window=!v->foe_bow_ready || v->passive || (view->foe_charging && !v->foe_aiming_me);
    if (view->foe_platforming) window=0;
    /* Completing a dash->arrow: the slide is airborne and fast; shoot anyway. */
    if (bot->v2.combo>0 && bot2_try_shot(view, v, bot, forced_release, max_hold>=0?max_hold:BOT_HOLD_MID, act)) { bot->v2.combo=0; return; }
    if (range<5.5 && window && bot2_try_shot(view, v, bot, forced_release, BOT_HOLD_QUICK, act)) return;
    if (bot2_try_shot(view, v, bot, forced_release, max_hold, act)) return;
    if (view->foe_platforming) {
        /* Frozen and unhittable: hold range with a ready bow for the ejection. */
        if (range<BOT2_PLAT_DISC*2+3) { bot_act(act, bot_aim_byte(dx>=0?0.2:-0.2, 1), 0, 0); return; }
        uint8_t aim=0; uint16_t buttons=0;
        bot_nav_foe(view, bot, &aim, &buttons); bot_act(act, aim, buttons, 0); return;
    }
    if (!window) {
        /* Ready bow: never inside 8.5 u; hold 9-14 u and wait for it to fire. */
        if (bot->v2.hold_off>0) bot->v2.hold_off--;
        if (range<BOT2_READY_BOW_RANGE || bot->v2.hold_off>0) {
            if (range<BOT2_READY_BOW_RANGE) bot->v2.hold_off=3;
            bot2_back_off(view, v, bot, act); return;
        }
        if (range<14 && fabs(dy)<5) { bot_act(act, bot_aim_byte(dx>=0?0.2:-0.2, 1), 0, 0); return; }
        uint8_t aim=0; uint16_t buttons=0;
        bot_nav_foe(view, bot, &aim, &buttons); bot_act(act, aim, buttons, 0); return;
    }
    /* Window open: close in. Dash-in when it ends near the foe and it cannot
     * dash away (its dash down), then shoot from the slide. */
    if (fabs(dy)>4.0) {
        uint8_t aim=0; uint16_t buttons=0;
        bot_nav_foe(view, bot, &aim, &buttons);
        bot_act(act, aim, buttons, 0); return;
    }
    if (view->dash_ready && !view->morphing && range>8 && range<21 && view->bow_ready
            && (!v->foe_dash_ready || v->passive || range>13)
            && bot_dash_reaches(view, view->fx, view->fy) && bot2_dash_safe(view, dx, dy)) {
        bot->v2.combo=3;
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

static void bot2_evasive(const BotView *view, const BotView2 *v, BoplBot *bot, int forced_release, BotAction *act) {
    double range=bot_len(view->fx-view->x, view->fy-view->y);
    int max_hold=bot2_max_hold(view, v, range);
    if (bot2_evade_aim(view, v, bot, act)) return;
    if (bot2_try_squish(view, v, bot, act)) return;
    /* A stalemate is worse than a contested shot. */
    if (bot->since_shot>40 && bot2_try_shot(view, v, bot, forced_release, max_hold, act)) return;
    /* Punish from safety: far, or while the foe's bow is down. */
    int window=!v->foe_bow_ready || v->passive || (view->foe_charging && !v->foe_aiming_me);
    if ((range>18 || (window && range>BOT2_READY_BOW_RANGE) || view->foe_charging || bot->foe_still>=16 || range>30)
            && bot2_try_shot(view, v, bot, forced_release, max_hold, act)) return;
    if (bot->foe_still>=30 && view->bow_timer_left<1.5) {
        double sx, sy;
        if (bot2_try_shot(view, v, bot, forced_release, max_hold, act)) return;
        if (bot_firing_spot(view, &sx, &sy)) {
            uint8_t aim=0; uint16_t buttons=0;
            bot_nav(view, bot, sx, sy, &aim, &buttons);
            bot_act(act, aim, buttons, 0); return;
        }
    }
    /* Keep away: the perch maximizing distance (also from a foe platform's plant point). */
    double best_x=view->x, best_y=view->y, best_score=-1e9;
    for (int i=0; i<view->wall_count; i++) {
        const BotWall *w=&view->walls[i];
        if (!w->standable) continue;
        for (int side=-1; side<=1; side++) {
            double span=fmax(w->spanx-0.8, 0);
            double px=w->cx+side*span, py;
            if (!bot_surface_y(view, w, px, &py)) continue;
            double score=bot_len(px-view->fx, py-view->fy)-0.35*bot_len(px-view->x, py-view->y)+0.5*py;
            if (v->haz_active) score+=fmin(bot_len(px-v->haz_x, py-v->haz_y), 14.0);
            if (score>best_score) { best_score=score; best_x=px; best_y=py; }
        }
    }
    if (range<8 && view->dash_ready) {
        double ex=best_x-view->x, ey=best_y-view->y+2;
        if (bot2_dash_safe(view, ex, ey)) { bot_act(act, bot_aim_byte(ex, ey), BOT_DASH, 0); return; }
        double away=view->x>=view->fx?1:-1;
        if (bot2_dash_safe(view, away, 0.6)) { bot_act(act, bot_aim_byte(away, 0.6), BOT_DASH, 0); return; }
    }
    uint8_t aim=0; uint16_t buttons=0;
    bot_nav(view, bot, best_x, best_y, &aim, &buttons);
    bot_act(act, aim, buttons, 0);
}

static void bot2_mixed(const BotView *view, const BotView2 *v, BoplBot *bot, int forced_release, BotAction *act) {
    double range=bot_len(view->fx-view->x, view->fy-view->y);
    int foe_falling_out=!view->foe_grounded && !view->foe_morphing && view->fvy<-8 && view->fy<BOT_WATER+14;
    int window=!v->foe_bow_ready || v->passive || (view->foe_charging && !v->foe_aiming_me);
    int mode=bot->mode;
    if (bot->mode_hold>0) bot->mode_hold--;
    else {
        if (foe_falling_out || (window && !view->foe_platforming && range<18) || bot->foe_still>=12) mode=BOPL_BOT_PURSUER;
        else if (!view->bow_ready && view->bow_timer_left>1.2 && range<14 && !view->dash_ready) mode=BOPL_BOT_EVASIVE;
        else mode=BOPL_BOT_ARCHER;
        if (mode!=bot->mode) { bot->mode=mode; bot->mode_hold=4; }
    }
    switch (bot->mode) {
    case BOPL_BOT_PURSUER: bot2_pursuer(view, v, bot, forced_release, act); return;
    case BOPL_BOT_EVASIVE: bot2_evasive(view, v, bot, forced_release, act); return;
    default: bot2_archer(view, v, bot, forced_release, act); return;
    }
}

/* ---- entry point ------------------------------------------------------------------ */
static void bot2_decide(BoplBot *bot, const BoplRlState *state, int player_index,
                        int hold_left, uint16_t held, const int hold_run[4], uint8_t out[3]) {
    out[0]=BOT_AIM_UP; out[1]=0; out[2]=0;
    BotView view;
    int live=bot_view(&view, state, player_index);
    bot->decisions++;
    bot->v2.since_dash++; bot->v2.since_platform++;
    if (bot->v2.combo>0) bot->v2.combo--;
    if (!view.alive || !view.my_slime || view.dead_phase) { bot->have_prev=0; return; }
    if (bot->have_prev) { view.svx=(view.x-bot->self_prev_x)*15; view.svy=(view.y-bot->self_prev_y)*15; }
    bot->self_prev_x=view.x; bot->self_prev_y=view.y; bot->have_prev=1;
    if (live) {
        double moved=bot_len(view.fx-bot->foe_prev_x, view.fy-bot->foe_prev_y);
        bot->foe_still=(moved<0.15 && bot->decisions>1)?bot->foe_still+1:0;
        bot->foe_prev_x=view.fx; bot->foe_prev_y=view.fy;
    }
    BotView2 v2; memset(&v2, 0, sizeof(v2)); v2.ground_wall=v2.haz_wall=-1; v2.haz_dist=1e9;
    if (live) bot2_view(&v2, &view, bot);
    static int bot_debug=-1;
    if (bot_debug<0) bot_debug=getenv("BOPL_BOT_DEBUG")!=NULL;
    if (hold_left>0) {
        if (live && (held&BOT_ARROW)) out[0]=bot_charge_aim(&view, bot, hold_left, hold_run);
        else if (live) out[0]=bot_aim_byte(view.fx-view.x>=0?1:-1, 0);
        out[1]=held; out[2]=0;
        bot->last[0]=out[0];
        if (bot_debug) {
            double eta=bot_arrow_eta(&view, NULL);
            fprintf(stderr, "[bot %d] d%03d t%u me(%.1f,%.1f) v(%.1f,%.1f) g%d m%d foe(%.1f,%.1f) mode%d bow%d dash%d LATCHED held%x left%d aim%3d eta%.0f arr%d\n",
                    player_index, bot->decisions, state->header.episode_ticks, view.x, view.y, view.vx, view.vy, view.grounded, view.morphing,
                    view.fx, view.fy, bot->mode, view.bow_ready, view.dash_ready, held, hold_left, out[0], eta, view.arrow_count);
        }
        return;
    }
    int forced_release=hold_run[2]>0;
    if (!live) return;
    double skip=bot->level>=BOPL_BOT_HARD?0.0:bot->level==BOPL_BOT_MEDIUM?0.08:0.22;
    if (bot->decisions>1 && bot_unit(bot)<skip && !(bot->last[1]&BOT_ARROW)) {
        out[0]=bot->last[0]; out[1]=bot->last[1]&(uint16_t)~BOT_ARROW; out[2]=0;
        return;
    }
    BotAction act; memset(&act, 0, sizeof(act));
    double dbg_eta=1e9; int dbg_dodge=0, dbg_veto=0;
    if (view.morphing) {
        if (view.charging) bot_act(&act, bot_charge_aim(&view, bot, 0, hold_run), 0, 0);
        else if (view.my_slime->ability_flags[BOT_SLOT_PLATFORM]&8) {
            /* Own platform morph. The aim is the platform's controls: during
             * Rotation the platform's up slerps toward the aim vector
             * (PlatformTransform.cs:276-289) and the Exit ejects the slime
             * along that up by radius+halfHeight+EjectionOffset = 3+0.78+2 u
             * (:418-428, EjectionOffset :95). v1 aims at the foe here, so its
             * platform stands on end and throws the slime sideways into the
             * nearest wall: that is the "own ejection" squish family. A zero
             * aim vector (byte 0, world.rs:89) is below DEADZONE and leaves
             * the platform flat: the slime pops straight up onto its own
             * platform, which cannot squish its rider. */
            bot_act(&act, 0, 0, 0);
        }
        else bot_act(&act, bot_aim_byte(view.fx-view.x>=0?1:-1, 0), 0, 0);
    }
    /* Threat layers: the foe's platform first (its window is seconds and the
     * penalty certain), then arrows. */
    if (!act.done) bot2_platform_threat(&view, &v2, bot, &act, &dbg_dodge);
    if (!act.done) bot2_dodge(&view, bot, &act, &dbg_eta, &dbg_dodge);
    else if (!view.morphing) dbg_eta=bot_arrow_eta(&view, NULL);
    if (!act.done) switch (bot->kind) {
    case BOPL_BOT_ARCHER_V2: bot2_archer(&view, &v2, bot, forced_release, &act); break;
    case BOPL_BOT_PURSUER_V2: bot2_pursuer(&view, &v2, bot, forced_release, &act); break;
    case BOPL_BOT_EVASIVE_V2: bot2_evasive(&view, &v2, bot, forced_release, &act); break;
    case BOPL_BOT_MIXED_V2: bot2_mixed(&view, &v2, bot, forced_release, &act); break;
    default: bot_act(&act, BOT_AIM_UP, 0, 0); break;
    }
    /* Sinking stage (v1 rule). */
    if (view.grounded && !(act.buttons&BOT_ARROW) && view.y<BOT_WATER+8 && view.state->header.episode_ticks>90*60) {
        double hx, hy;
        if (bot_highest_perch(&view, &hx, &hy) && hy>view.y+1) {
            uint8_t aim=0; uint16_t buttons=0;
            bot_nav(&view, bot, hx, hy, &aim, &buttons);
            bot_act(&act, aim, buttons, 0);
        } else if (view.platform_ready && bot2_platform_press_safe(&view, &v2))
            bot_act(&act, BOT_AIM_UP, BOT_PLATFORM, 0);
    }
    /* Survival veto (v1): airborne steering must land dry and clear of arrows. */
    if (!view.morphing && !view.grounded && !(act.buttons&(BOT_DASH|BOT_PLATFORM|BOT_ARROW))) {
        double ang=((int)act.aim-1)/255.0*2*M_PI, ax=cos(ang), lx, ly;
        int steer=ax>0.33?1:ax<-0.33?-1:0;
        int doomed=!bot_landing_steer(&view, steer, &lx, &ly);
        int shot_down=bot_steer_hits_arrow(&view, steer);
        if (doomed || shot_down) {
            static const int opts[3]={1, -1, 0};
            int found=0;
            for (int i=0; i<3 && !found; i++)
                if (opts[i]!=steer && bot_landing_steer(&view, opts[i], &lx, &ly) && !bot_steer_hits_arrow(&view, opts[i])) {
                    act.aim=opts[i]?bot_aim_byte(opts[i], 0):BOT_AIM_UP; found=1;
                }
            if (!found && shot_down && !doomed && view.platform_ready && bot2_platform_press_safe(&view, &v2)) {
                bot_act(&act, BOT_AIM_UP, BOT_PLATFORM, 0);
            } else if (!found && doomed) {
                BotAction rescue; memset(&rescue, 0, sizeof(rescue));
                bot_recover(&view, bot, !forced_release, &rescue);
                if (rescue.done) act=rescue;
            }
            dbg_veto=1;
        }
    }
    /* Real dash safety (Dash.cs:245-258): the ray must not cross an arrow. Any
     * dash the layers above chose through v1's glide model is re-checked here. */
    if ((act.buttons&BOT_DASH) && !view.morphing) {
        double ang=((int)act.aim-1)/255.0*2*M_PI;
        if (!bot2_dash_ray_clear(&view, cos(ang), sin(ang))) {
            act.buttons&=(uint16_t)~BOT_DASH; dbg_veto=1;
            if (!view.grounded) {
                /* A recovery dash was the plan: fall back to the platform freeze. */
                if (view.platform_ready && bot2_platform_press_safe(&view, &v2)) act.buttons|=BOT_PLATFORM;
            }
        } else bot->v2.since_dash=0;
    }
    /* Platform press safety: ejection point clear, no growing foe platform near. */
    if (act.buttons&BOT_PLATFORM) {
        if (!bot2_platform_press_safe(&view, &v2)) { act.buttons&=(uint16_t)~BOT_PLATFORM; dbg_veto=1; }
        else bot->v2.since_platform=0;
    }
    /* Launch veto (v1): a grounded jump/ejection must not fly into an arrow. */
    if (view.grounded && (act.buttons&(BOT_JUMP|BOT_PLATFORM)) && view.arrow_count) {
        if (bot_launch_hits_arrow(&view, -22) || bot_launch_hits_arrow(&view, 0) || bot_launch_hits_arrow(&view, 22))
            { act.buttons&=(uint16_t)~(BOT_JUMP|BOT_PLATFORM); dbg_veto=1; }
    }
    /* Doomed-jump veto (v1). */
    if (view.grounded && (act.buttons&BOT_JUMP) && !view.dash_ready && !view.platform_ready) {
        if (!bot_launch_lands(&view, -22) || !bot_launch_lands(&view, 0) || !bot_launch_lands(&view, 22))
            { act.buttons&=(uint16_t)~BOT_JUMP; dbg_veto=1; }
    }
    /* Walk veto: a grounded walk must not step under an arrow (own mortars and
     * fresh uphill lobs included): try the other way along the ground, then
     * standing still. */
    if (view.grounded && !(act.buttons&(BOT_DASH|BOT_PLATFORM|BOT_ARROW|BOT_JUMP)) && view.arrow_count) {
        double wvx, wvy;
        bot2_walk_velocity(&view, &v2, act.aim, &wvx, &wvy);
        if (bot2_arrow_eta(&view, wvx, wvy, 45, NULL)<=45) {
            double tx, ty; bot2_ground_tangent(&view, &v2, &tx, &ty);
            uint8_t opts[3]={bot_aim_byte(tx, ty), bot_aim_byte(-tx, -ty), bot_aim_byte(-ty, tx)};  /* right, left, still (along the normal) */
            double best=-1; uint8_t pick=act.aim;
            for (int i=0; i<3; i++) {
                bot2_walk_velocity(&view, &v2, opts[i], &wvx, &wvy);
                double e=bot2_arrow_eta(&view, wvx, wvy, 45, NULL);
                if (e>best) { best=e; pick=opts[i]; }
            }
            act.aim=pick; dbg_veto=1;
        }
    }
    /* Never charge inside the foe platform's reach. */
    if ((act.buttons&BOT_ARROW) && bot->v2.in_hazard) act.buttons&=(uint16_t)~BOT_ARROW;
    /* Difficulty: aim noise (v1 knob). */
    if (bot->level<BOPL_BOT_HARD && act.aim) {
        double sigma=bot->level==BOPL_BOT_MEDIUM?2.0:5.0;
        double u1=bot_unit(bot), u2=bot_unit(bot);
        if (u1>1e-9) {
            int wobble=(int)lround(sqrt(-2*log(u1))*cos(2*M_PI*u2)*sigma);
            int value=((int)act.aim-1+wobble)%255; if (value<0) value+=255;
            act.aim=(uint8_t)(1+value);
        }
    }
    if (forced_release) act.buttons&=(uint16_t)~BOT_ARROW;
    bot->since_shot=(act.buttons&BOT_ARROW)?0:bot->since_shot+1;
    bot->since_jump=(act.buttons&BOT_JUMP)?0:bot->since_jump+1;
    if (bot_debug) {
        double dbg_wvx, dbg_wvy; bot2_walk_velocity(&view, &v2, act.aim, &dbg_wvx, &dbg_wvy);
        double dbg_weta=view.grounded?bot2_arrow_eta(&view, dbg_wvx, dbg_wvy, 45, NULL):1e9;
        double sx=0, sy=0; int spot=bot_firing_spot(&view, &sx, &sy);
        double lx, ly;
        int lminus=bot_landing_steer(&view, -1, &lx, &ly), lzero=bot_landing_steer(&view, 0, &lx, &ly), lplus=bot_landing_steer(&view, 1, &lx, &ly);
        fprintf(stderr, "[bot %d] d%03d t%u me(%.1f,%.1f) v(%.1f,%.1f) g%d m%d foe(%.1f,%.1f) mode%d bow%d dash%d plat%d aim%3d btn%x hold%d spot%d(%.1f,%.1f) land[%d%d%d] arr%d eta%.0f dodge%d veto%d still%d haz%.1f fbow%d faim%d gw%d wv(%.1f,%.1f) weta%.0f\n",
                player_index, bot->decisions, state->header.episode_ticks, view.x, view.y, view.vx, view.vy, view.grounded, view.morphing,
                view.fx, view.fy, bot->mode, view.bow_ready, view.dash_ready, view.platform_ready,
                act.aim, act.buttons, act.hold, spot, sx, sy, lminus, lzero, lplus,
                view.arrow_count, dbg_eta, dbg_dodge, dbg_veto, bot->foe_still,
                v2.haz_active?v2.haz_dist:99.0, v2.foe_bow_ready, v2.foe_aiming_me, v2.ground_wall, dbg_wvx, dbg_wvy, dbg_weta);
    }
    if (!(act.buttons&BOT_ARROW)) act.hold=0;
    out[0]=act.aim; out[1]=(uint8_t)(act.buttons&15u); out[2]=act.hold;
    memcpy(bot->last, out, 3);
}

#endif
