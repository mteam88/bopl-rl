/* Observation writer for layout v4 (bopl_layout_v4.h).
 * Included by bopl_puffer.h after the v3 writer's helpers; shares its
 * visibility, delay and truncation rules (bopl_spatial_hidden,
 * bopl_entity_visible, bopl_nearest, rope thinning). Nothing here is read by
 * the v3 path. */
#ifndef BOPL_OBS_V4_H
#define BOPL_OBS_V4_H

/* Writer ablations, a bitmask in BOPL4_ABLATE
 * read once per process (train.py --v4-ablate, recorded in the checkpoint's
 * v4_shape so the evaluator applies the same one). The row keeps its shape;
 * the ablated block is zero. */
/* BOPL4_ABLATE_AIM is layout v4.1, not an ablation:
 * the two down rays that duplicate around rays (-135 and -45 degrees are
 * around rays 20 and 28) carry the aim ray and the line of sight instead. */
enum { BOPL4_ABLATE_RELATIONAL=1, BOPL4_ABLATE_RAYS=2, BOPL4_ABLATE_RAYS16=4, BOPL4_ABLATE_AIM=8 };
enum { BOPL4_AIM_RAY=BOPL4_RAYS_AROUND+0, BOPL4_LOS_RAY=BOPL4_RAYS_AROUND+3 };
static int bopl4_ablation_value=-1;
/* The process default (BOPL4_ABLATE, read once): every row of a training run. */
static void bopl4_set_ablation(int bits) { bopl4_ablation_value=bits; }
static int bopl4_ablation_default(void) {
    if (bopl4_ablation_value<0) { const char *v=getenv("BOPL4_ABLATE"); bopl4_ablation_value=v&&v[0]?atoi(v):0; }
    return bopl4_ablation_value;
}
/* The bits of the row being written on this thread (bopl_encode_v4_row sets
 * them from env->ablate[player]; -1 = the default), so players whose
 * checkpoints trained under different writer families share one game. Rows
 * are written one at a time per thread (the env pool's workers, the research
 * environment, the shim). */
static __thread int bopl4_row_ablation=-1;
static int bopl4_ablation(void) {
    return bopl4_row_ablation>=0?bopl4_row_ablation:bopl4_ablation_default();
}

/* Relational block (BOPL4_RELATION) of a body at relative position r moving
 * at relative velocity v (both versus the observer), with `radius` taken off
 * the distance. `ballistic` adds the gravity-arc closest approach (projectiles). */
static void bopl4_relation(float *out, double dx, double dy, double dvx, double dvy, double radius, int ballistic) {
    double r=sqrt(dx*dx+dy*dy), rr=r>1e-9?1.0/r:0.0;
    double rv=dx*dvx+dy*dvy, vv=dvx*dvx+dvy*dvy;
    double tca=vv>1e-9?-rv/vv:0.0;
    if (tca<0.0) tca=0.0; else if (tca>BOPL4_HORIZON) tca=BOPL4_HORIZON;
    double mx=dx+dvx*tca, my=dy+dvy*tca, miss=sqrt(mx*mx+my*my)-radius;
    out[0]=bopl_clip((r-radius)/BOPL4_POSITION_SCALE);
    out[1]=bopl_clip(dy*rr); out[2]=bopl_clip(dx*rr);
    out[3]=bopl_clip(dvx/BOPL4_VELOCITY_SCALE); out[4]=bopl_clip(dvy/BOPL4_VELOCITY_SCALE);
    out[5]=bopl_clip(-rv*rr/BOPL4_VELOCITY_SCALE);                 /* closing speed (>0 approaching) */
    out[6]=bopl_clip((dx*dvy-dy*dvx)*rr/BOPL4_VELOCITY_SCALE);     /* tangential speed */
    out[7]=bopl_clip(tca/BOPL4_HORIZON);
    out[8]=bopl_clip(miss/16.0);
    out[9]=bopl_clip(sqrt(vv)/BOPL4_VELOCITY_SCALE);
    out[10]=0; out[11]=0;
    if (ballistic) {
        /* Arc under gravity (observer assumed on the ground), sampled every
         * 1/15 s to the horizon; the nearest sample's distance and time. */
        double best=r-radius, best_t=0.0;
        for (int k=1; k<=30; k++) {
            double t=k/15.0, px=dx+dvx*t, py=dy+dvy*t-0.5*BOPL4_GRAVITY*t*t, d=sqrt(px*px+py*py)-radius;
            if (d<best) { best=d; best_t=t; }
        }
        out[10]=bopl_clip(best/16.0); out[11]=bopl_clip(best_t/BOPL4_HORIZON);
    }
    if (bopl4_ablation()&BOPL4_ABLATE_RELATIONAL) memset(out, 0, BOPL4_RELATION*sizeof(float));
}
/* Controlling-slime block (BOPL4_BODY); NULL or hidden bodies are all zero. */
static void bopl4_body(float *out, const BoplRlSlime *slime) {
    memset(out, 0, BOPL4_BODY*sizeof(float));
    if (!slime) return;
    double vx=bopl_fix(slime->vx), vy=bopl_fix(slime->vy);
    out[0]=1; out[1]=bopl_clip(vx/BOPL4_VELOCITY_SCALE); out[2]=bopl_clip(vy/BOPL4_VELOCITY_SCALE);
    out[3]=bopl_clip(sqrt(vx*vx+vy*vy)/BOPL4_VELOCITY_SCALE);
    out[4]=sinf((float)bopl_fix(slime->rotation)); out[5]=cosf((float)bopl_fix(slime->rotation));
    if (slime->state<11) out[6+slime->state]=1;
    out[17]=(slime->flags&BOPL_SLIME_IN_ABILITY)!=0; out[18]=(slime->flags&BOPL_SLIME_SUPPRESS_INPUT)!=0;
    out[19]=(slime->flags&BOPL_SLIME_SPAWNED)!=0; out[20]=(slime->flags&BOPL_SLIME_ABILITYLESS)!=0;
    out[21]=bopl_clip(bopl_fix(slime->scale)/3.0);
    out[22]=slime->vx==0 && slime->vy==0;
    out[23]=((slime->flags>>BOPL_SLIME_PICKUP_SLOT_SHIFT)&BOPL_SLIME_PICKUP_SLOT_MASK)/2.0f;
}
/* The primary enemy of `self`: the enemy player whose controlling slime is
 * nearest among the visible ones, else the first enemy (position unknown). */
typedef struct {
    const BoplRlPlayer *player; const BoplRlSlime *slime;
    int known; double x, y, vx, vy;
} Bopl4Enemy;
static Bopl4Enemy bopl4_primary_enemy(const BoplRlState *state, const BoplRlPlayer *self, double self_x, double self_y) {
    Bopl4Enemy best={0}; double best_distance=0.0;
    for (uint32_t i=0; i<state->header.player_count; i++) {
        const BoplRlPlayer *player=&state->players[i];
        if (player->id==self->id || player->team==self->team) continue;
        int hidden=bopl_spatial_hidden(state, player->id, self->id);
        double x=bopl_fix(player->x), y=bopl_fix(player->y);
        const BoplRlSlime *slime=bopl_controlling_slime(state, player->id, x, y);
        double dx=x-self_x, dy=y-self_y, distance=dx*dx+dy*dy;
        int candidate_known=!hidden;
        if (!best.player || candidate_known>best.known || (candidate_known==best.known && distance<best_distance)) {
            best.player=player; best.slime=hidden?NULL:slime; best.known=candidate_known;
            best.x=x; best.y=y; best.vx=slime?bopl_fix(slime->vx):0.0; best.vy=slime?bopl_fix(slime->vy):0.0;
            best_distance=distance;
        }
    }
    return best;
}
typedef struct {
    int axis_count, rot_count;
    double axis_cx[48], axis_cy[48], axis_ex[48], axis_ey[48], axis_radius[48];
    double rot_cx[48], rot_cy[48], rot_rx[48], rot_ry[48], rot_ux[48], rot_uy[48], rot_ex[48], rot_ey[48], rot_radius[48];
} Bopl4Walls;
/* ---- terrain: the static-collider SDF the scripted bots use (bopl_bots.h
 * bot_wall_sdf / bot_sdf / bot_raycast), filled here for the observer. ---- */
typedef struct {
    BotView view;                 /* only walls[], wall_count and soa are filled */
    double water_top;             /* y of the water surface, -1e9 if no water entity */
    int has_water;
    /* Dynamic, visible entities and enemy bodies as circles for the ray casts;
     * cclass: 0 hostile projectile / object, 1 opponent body, 2 own object. */
    int circle_count;
    double cx[BOPL4_ENTITIES+BOPL4_SLIMES], cy[BOPL4_ENTITIES+BOPL4_SLIMES], cr[BOPL4_ENTITIES+BOPL4_SLIMES];
    double cvx[BOPL4_ENTITIES+BOPL4_SLIMES], cvy[BOPL4_ENTITIES+BOPL4_SLIMES];
    int cclass[BOPL4_ENTITIES+BOPL4_SLIMES];
    /* The walls again for the vector march, split by frame: the axis-aligned
     * ones (right = (+-1, 0), up = (0, +-1), or the 90-degree turn of that
     * with the extents swapped) and the rotated rest. For an aligned wall the
     * frame rotation rpx*rx + rpy*ry is rpx*+-1 + rpy*0, whose IEEE value is
     * +-rpx (+-0 added to a double leaves it, and a +0/-0 difference dies in
     * the abs that follows), so |lx| = |rpx| and |ly| = |rpy| exactly without
     * the four products: the same per-wall distance as bot_wall_sdf's. The
     * min over the walls is taken in a different order (aligned first), which
     * does not change a minimum of doubles (the +0/-0 tie caveat above). */
    Bopl4Walls walls;
    /* bopl4_terrain_anchor: each wall's SDF at the body (axis_body, rot_body),
     * from which bopl4_march culls the walls no ray of a group can reach. */
    int anchored;
    double bx, by, axis_body[48], rot_body[48];
} Bopl4Terrain;
static void bopl4_terrain_fill(Bopl4Terrain *t, const BoplRlState *state, const uint32_t *nearest, int count,
                               const uint32_t *nearby_slimes, int slime_count, const BoplRlPlayer *self) {
    BotView *view=&t->view;
#if BOT_SDF_AVX512
    /* bot_view's detection, so the SDF dispatch is settled here too: a row
     * written before any scripted bot has moved (every reset row in a game
     * without one) otherwise marches the scalar SDF. */
    if (bot_cpu_avx512<0) bot_cpu_avx512=__builtin_cpu_supports("avx512f")?1:0;
#endif
    view->wall_count=0; t->has_water=0; t->water_top=-1e9; t->circle_count=0; t->walls.axis_count=0; t->walls.rot_count=0; t->anchored=0;
    for (uint32_t i=0; i<state->header.entity_count; i++) {
        const BoplRlEntity *e=&state->entities[i];
        if (!(e->flags&1) || (e->flags&4)) continue;
        if (e->layer==BOPL4_WATER_LAYER) {
            double top=bopl_fix(e->y)+fabs(bopl_fix(e->up_y))+fabs(bopl_fix(e->right_y));
            if (!t->has_water || top>t->water_top) t->water_top=top;
            t->has_water=1;
            continue;
        }
        /* Terrain: rounded rects on wall/terrain layers (11 wall, 16 lethal, 21 non-lethal). */
        if (e->kind==2 && (e->layer==11 || e->layer==16 || e->layer==21) && view->wall_count<48) {
            BotWall *w=&view->walls[view->wall_count++];
            w->cx=bopl_fix(e->x); w->cy=bopl_fix(e->y);
            double rx=bopl_fix(e->right_x), ry=bopl_fix(e->right_y), ux=bopl_fix(e->up_x), uy=bopl_fix(e->up_y);
            double fex=sqrt(rx*rx+ry*ry), fey=sqrt(ux*ux+uy*uy);
            w->radius=bopl_fix(e->radius);
            if (fex>1e-6) { w->rx=rx/fex; w->ry=ry/fex; } else { w->rx=1; w->ry=0; }
            if (fey>1e-6) { w->ux=ux/fey; w->uy=uy/fey; } else { w->ux=-w->ry; w->uy=w->rx; }
            w->ex=fex; w->ey=fey;
            w->top=w->cy+w->ex*fabs(w->ry)+w->ey*fabs(w->uy)+w->radius;
            w->spanx=w->ex*fabs(w->rx)+w->ey*fabs(w->ux);
            w->standable=e->layer==16?-1:1;   /* -1 marks lethal terrain for the ray class */
            int k=view->wall_count-1;
            view->soa.cx[k]=w->cx; view->soa.cy[k]=w->cy; view->soa.rx[k]=w->rx; view->soa.ry[k]=w->ry;
            view->soa.ux[k]=w->ux; view->soa.uy[k]=w->uy; view->soa.ex[k]=w->ex; view->soa.ey[k]=w->ey;
            view->soa.radius[k]=w->radius; view->soa.top[k]=w->top;
            int straight=(w->rx==1||w->rx==-1) && w->ry==0 && w->ux==0 && (w->uy==1||w->uy==-1);
            int turned=w->rx==0 && (w->ry==1||w->ry==-1) && (w->ux==1||w->ux==-1) && w->uy==0;
            Bopl4Walls *v=&t->walls;
            if (straight||turned) {
                int a=v->axis_count++;
                v->axis_cx[a]=w->cx; v->axis_cy[a]=w->cy; v->axis_radius[a]=w->radius;
                v->axis_ex[a]=straight?w->ex:w->ey; v->axis_ey[a]=straight?w->ey:w->ex;
            } else {
                int r=v->rot_count++;
                v->rot_cx[r]=w->cx; v->rot_cy[r]=w->cy; v->rot_rx[r]=w->rx; v->rot_ry[r]=w->ry;
                v->rot_ux[r]=w->ux; v->rot_uy[r]=w->uy; v->rot_ex[r]=w->ex; v->rot_ey[r]=w->ey; v->rot_radius[r]=w->radius;
            }
        }
    }
    /* The observed (visible, nearest) dynamic bodies, as circles. */
    for (int i=0; i<count; i++) {
        const BoplRlEntity *e=&state->entities[nearest[i]];
        if (e->inverse_mass<=0 || e->kind==3 || e->kind==4) continue;
        int k=t->circle_count++;
        t->cx[k]=bopl_fix(e->x); t->cy[k]=bopl_fix(e->y); t->cr[k]=fmax(bopl_fix(e->radius), 0.25);
        t->cvx[k]=bopl_fix(e->vx); t->cvy[k]=bopl_fix(e->vy);
        t->cclass[k]=e->owner==self->id?2:0;
    }
    /* Enemy bodies (the visible slime rows of other teams). */
    for (int i=0; i<slime_count; i++) {
        const BoplRlSlime *s=&state->slimes[nearby_slimes[i]];
        if (!(s->flags&BOPL_SLIME_ACTIVE)) continue;
        const BoplRlPlayer *owner=bopl_player_by_id(state, s->owner);
        if (!owner || owner->id==self->id || owner->team==self->team) continue;
        int k=t->circle_count++;
        double r=BOT_SLIME_RADIUS*bopl_fix(s->scale);
        t->cx[k]=bopl_fix(s->x); t->cy[k]=bopl_fix(s->y); t->cr[k]=r>0?r:BOT_SLIME_RADIUS;
        t->cvx[k]=bopl_fix(s->vx); t->cvy[k]=bopl_fix(s->vy);
        t->cclass[k]=1;
    }
}
/* Nearest wall's index at a point (for the lethal class). */
static int bopl4_nearest_wall(const BotView *view, double x, double y) {
#if BOT_SDF_AVX512
    if (bot_cpu_avx512>0) { double d; return bot_min_wall_avx512(view, x, y, ~0ull, &d); }   /* the same first-minimum rule */
#endif
    int best=-1; double best_d=1e9;
    for (int i=0; i<view->wall_count; i++) { double d=bot_wall_sdf(&view->walls[i], x, y); if (d<best_d) { best_d=d; best=i; } }
    return best;
}
/* One ray: distance to the first surface among walls, water and circles
 * (BOPL4_RAY_LENGTH when nothing), the class and, for a circle, its closing
 * speed along the ray (relative to the observer's velocity). */
/* Ray k's direction: BOPL4_RAYS_AROUND uniform from +x, then the fan about
 * straight down (cell centres of a 120-degree span: -135, -105, -75, -45). */
static void bopl4_ray_direction_compute(int k, double *dx, double *dy) {
    double a;
    if (k<BOPL4_RAYS_AROUND) a=6.28318530717958647692*k/BOPL4_RAYS_AROUND;
    else { double span=6.28318530717958647692/3.0; a=-1.57079632679489661923-span/2+span*(k-BOPL4_RAYS_AROUND+0.5)/BOPL4_RAYS_DOWN; }
    *dx=cos(a); *dy=sin(a);
}
/* The 36 directions are constants: computed once per process by the same
 * cos / sin calls (so the same doubles), not per ray per row. */
static void bopl4_ray_direction(int k, double *dx, double *dy) {
    static __thread double table[BOPL4_RAYS][2];
    static __thread int ready=0;   /* per thread: no synchronisation, 576 bytes */
    if (!ready) {
        for (int i=0; i<BOPL4_RAYS; i++) bopl4_ray_direction_compute(i, &table[i][0], &table[i][1]);
        ready=1;
    }
    *dx=table[k][0]; *dy=table[k][1];
}
/* bot_raycast's result compared against `limit`, without finishing the march
 * once the outcome is fixed: the march's t only grows and it returns either
 * the t it is at or `max`, so once t > limit the result exceeds limit
 * whatever happens next (if limit >= max the loop cannot reach t > limit
 * before it ends with max, which is then <= limit: the same answer as the
 * full march). Returns 1 when the full raycast would be > limit. */
static int bot_raycast_exceeds(const BotView *view, double x, double y, double dx, double dy, double max, double limit) {
    double t=0;
    for (int i=0; i<32 && t<max; i++) {
        if (t>limit) return 1;
        double d=bot_sdf(view, x+dx*t, y+dy*t);
        if (d<0.05) return t>limit;
        t+=d;
    }
    return max>limit;
}
/* The raycast's t when it is < `limit`, else any value >= limit (the caller
 * only uses the result through a `< limit` comparison). Same early exit. */
static double bot_raycast_below(const BotView *view, double x, double y, double dx, double dy, double max, double limit) {
    double t=0;
    for (int i=0; i<32 && t<max; i++) {
        if (t>=limit) return limit;
        double d=bot_sdf(view, x+dx*t, y+dy*t);
        if (d<0.05) return t;
        t+=d;
    }
    return max;
}
/* bot_raycast_below for up to BOPL4_MARCH rays (own origins), marched
 * together: each ray's steps are exactly bot_raycast's (the same SDF calls on
 * the same points, in the same order per ray). The scalar body interleaves
 * the rays so the core overlaps their independent chains; the AVX-512 march
 * below keeps 8 rays in the lanes and refills a lane when its ray ends. */
#define BOPL4_MARCH 40
#define BOPL4_MARCH_BODY(SDF) \
    double t[BOPL4_MARCH]; \
    uint64_t live=0; \
    for (int i=0; i<n; i++) { t[i]=0; out[i]=max; live|=1ull<<i; } \
    for (int iter=0; iter<32 && live; iter++) { \
        for (int i=0; i<n; i++) { \
            if (!(live&(1ull<<i))) continue; \
            if (!(t[i]<max)) { live&=~(1ull<<i); continue; }               /* the loop condition t < max: out stays max */ \
            if (t[i]>=limit[i]) { out[i]=limit[i]; live&=~(1ull<<i); continue; } \
            double d=SDF(view, ox[i]+dx[i]*t[i], oy[i]+dy[i]*t[i]); \
            if (d<0.05) { out[i]=t[i]; live&=~(1ull<<i); continue; } \
            t[i]+=d; \
        } \
    } \
    /* Rays still live after 32 steps: bot_raycast returns max (out[i] already is). */
#if BOT_SDF_AVX512
/* Compiled with the SDF's own optimisation options: fp-contract=off keeps
 * x+dx*t and the SDF's sums from becoming FMAs now that the target has one,
 * so the sample points and distances stay bot_raycast's. */
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif
/* The same march with rays in the lanes and the walls in the loop: one step
 * of 8 rays is wall_count evaluations of bot_sdf_lanes' arithmetic (the same
 * IEEE operations on the same operands per (ray, wall), so the same per-wall
 * distance) and an elementwise min over the walls (the minimum of a set of
 * doubles does not depend on the order it is taken in; a tie between +0 and
 * -0 can only change the sign of a d that is a hit either way, and t + -0 ==
 * t + +0 for the t >= +0 here). Each lane runs the scalar body's control
 * flow for its ray, in the same order: the 32-step cap, t < max, t >= limit,
 * the hit test, t += d. A lane whose ray has ended takes the next ray, so
 * the lanes stay full while rays of very different lengths (a ray grazing
 * the floor takes all 32 steps, one into the sky 5) march side by side;
 * the sample points of a ray do not depend on which lane it runs in. */
__attribute__((target("avx512f"))) static void bopl4_march_avx512(const Bopl4Walls *tr, const double *ox, const double *oy, int n,
        const double *dx, const double *dy, double max, const double *limit, double *out) {
    __m512d vmax=_mm512_set1_pd(max), zero=_mm512_setzero_pd(), hit_at=_mm512_set1_pd(0.05), far=_mm512_set1_pd(1e9);
    __m512d vox=zero, voy=zero, vdx=zero, vdy=zero, vlimit=zero, t=zero, vout=vmax;
    __m512i steps=_mm512_setzero_si512(), cap=_mm512_set1_epi64(32), one=_mm512_set1_epi64(1);
    int slot[8], next=0;
    __mmask8 live=0;
    for (;;) {
        /* Free lanes take the next rays. */
        for (unsigned free=(unsigned)(~live&0xFFu); free && next<n; free&=free-1u, next++) {
            int l=__builtin_ctz(free); __mmask8 m=(__mmask8)(1u<<l);
            slot[l]=next;
            vox=_mm512_mask_mov_pd(vox, m, _mm512_set1_pd(ox[next])); voy=_mm512_mask_mov_pd(voy, m, _mm512_set1_pd(oy[next]));
            vdx=_mm512_mask_mov_pd(vdx, m, _mm512_set1_pd(dx[next])); vdy=_mm512_mask_mov_pd(vdy, m, _mm512_set1_pd(dy[next]));
            vlimit=_mm512_mask_mov_pd(vlimit, m, _mm512_set1_pd(limit[next]));
            t=_mm512_mask_mov_pd(t, m, zero); vout=_mm512_mask_mov_pd(vout, m, vmax);
            steps=_mm512_mask_mov_epi64(steps, m, _mm512_setzero_si512());
            live|=m;
        }
        if (!live) break;
        __mmask8 before=live;
        live=_mm512_mask_cmp_epi64_mask(live, steps, cap, _MM_CMPINT_LT);       /* the 32-step cap: out stays max */
        live=_mm512_mask_cmp_pd_mask(live, t, vmax, _CMP_LT_OQ);                /* !(t<max): out stays max */
        __mmask8 reach=_mm512_mask_cmp_pd_mask(live, t, vlimit, _CMP_GE_OQ);    /* t>=limit: out=limit */
        vout=_mm512_mask_mov_pd(vout, reach, vlimit); live=(__mmask8)(live&~reach);
        if (live) {
            __m512d px=_mm512_add_pd(vox, _mm512_mul_pd(vdx, t)), py=_mm512_add_pd(voy, _mm512_mul_pd(vdy, t));
            __m512d d=far;
            for (int k=0; k<tr->axis_count; k++) {   /* aligned: |lx| = |rpx|, |ly| = |rpy| (Bopl4Terrain) */
                __m512d qx=_mm512_sub_pd(_mm512_abs_pd(_mm512_sub_pd(px, _mm512_set1_pd(tr->axis_cx[k]))), _mm512_set1_pd(tr->axis_ex[k]));
                __m512d qy=_mm512_sub_pd(_mm512_abs_pd(_mm512_sub_pd(py, _mm512_set1_pd(tr->axis_cy[k]))), _mm512_set1_pd(tr->axis_ey[k]));
                __m512d o_x=_mm512_max_pd(qx, zero), o_y=_mm512_max_pd(qy, zero);
                __m512d outside=_mm512_sqrt_pd(_mm512_add_pd(_mm512_mul_pd(o_x, o_x), _mm512_mul_pd(o_y, o_y)));
                __m512d inside=_mm512_min_pd(_mm512_max_pd(qx, qy), zero);
                d=_mm512_min_pd(d, _mm512_sub_pd(_mm512_add_pd(outside, inside), _mm512_set1_pd(tr->axis_radius[k])));
            }
            for (int k=0; k<tr->rot_count; k++) {
                __m512d rpx=_mm512_sub_pd(px, _mm512_set1_pd(tr->rot_cx[k])), rpy=_mm512_sub_pd(py, _mm512_set1_pd(tr->rot_cy[k]));
                __m512d lx=_mm512_add_pd(_mm512_mul_pd(rpx, _mm512_set1_pd(tr->rot_rx[k])), _mm512_mul_pd(rpy, _mm512_set1_pd(tr->rot_ry[k])));
                __m512d ly=_mm512_add_pd(_mm512_mul_pd(rpx, _mm512_set1_pd(tr->rot_ux[k])), _mm512_mul_pd(rpy, _mm512_set1_pd(tr->rot_uy[k])));
                __m512d qx=_mm512_sub_pd(_mm512_abs_pd(lx), _mm512_set1_pd(tr->rot_ex[k]));
                __m512d qy=_mm512_sub_pd(_mm512_abs_pd(ly), _mm512_set1_pd(tr->rot_ey[k]));
                __m512d o_x=_mm512_max_pd(qx, zero), o_y=_mm512_max_pd(qy, zero);
                __m512d outside=_mm512_sqrt_pd(_mm512_add_pd(_mm512_mul_pd(o_x, o_x), _mm512_mul_pd(o_y, o_y)));
                __m512d inside=_mm512_min_pd(_mm512_max_pd(qx, qy), zero);
                d=_mm512_min_pd(d, _mm512_sub_pd(_mm512_add_pd(outside, inside), _mm512_set1_pd(tr->rot_radius[k])));
            }
            __mmask8 hit=_mm512_mask_cmp_pd_mask(live, d, hit_at, _CMP_LT_OQ);   /* d<0.05: out=t */
            vout=_mm512_mask_mov_pd(vout, hit, t); live=(__mmask8)(live&~hit);
            t=_mm512_mask_add_pd(t, live, t, d);
            steps=_mm512_mask_add_epi64(steps, live, steps, one);
        }
        /* Rays that ended this step: their results, then (top of the loop) new rays. */
        unsigned ended=(unsigned)(before&~live);
        if (ended) {
            double res[8]; _mm512_storeu_pd(res, vout);
            for (; ended; ended&=ended-1u) { int l=__builtin_ctz(ended); out[slot[l]]=res[l]; }
        }
    }
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
#endif
/* The scalar march: bot_raycast's steps, one ray after another (the CPUs
 * without AVX-512, and the reference writer_bench checks the vector march
 * against, row for row, through bopl4_march_scalar_only). */
static void bopl4_march_scalar(const BotView *view, const double *ox, const double *oy, int n, const double *dx, const double *dy,
                               double max, const double *limit, double *out) {
    BOPL4_MARCH_BODY(bot_sdf_scalar)
}
static __thread int bopl4_march_scalar_only=0;
/* Each wall's SDF at the body (x, y), for bopl4_march's culling. A rotated
 * wall whose frame is not orthogonal is never culled (its SDF need not be
 * 1-Lipschitz). */
static void bopl4_terrain_anchor(Bopl4Terrain *t, double x, double y) {
    const Bopl4Walls *v=&t->walls;
    t->anchored=1; t->bx=x; t->by=y;
    for (int k=0; k<v->axis_count; k++) {
        double qx=fabs(x-v->axis_cx[k])-v->axis_ex[k], qy=fabs(y-v->axis_cy[k])-v->axis_ey[k];
        double o_x=fmax(qx, 0), o_y=fmax(qy, 0);
        t->axis_body[k]=sqrt(o_x*o_x+o_y*o_y)+fmin(fmax(qx, qy), 0)-v->axis_radius[k];
    }
    for (int k=0; k<v->rot_count; k++) {
        double rpx=x-v->rot_cx[k], rpy=y-v->rot_cy[k];
        double qx=fabs(rpx*v->rot_rx[k]+rpy*v->rot_ry[k])-v->rot_ex[k], qy=fabs(rpx*v->rot_ux[k]+rpy*v->rot_uy[k])-v->rot_ey[k];
        double o_x=fmax(qx, 0), o_y=fmax(qy, 0);
        t->rot_body[k]=sqrt(o_x*o_x+o_y*o_y)+fmin(fmax(qx, qy), 0)-v->rot_radius[k];
        if (fabs(v->rot_rx[k]*v->rot_ux[k]+v->rot_ry[k]*v->rot_uy[k])>1e-9) t->rot_body[k]=-HUGE_VAL;
    }
}
/* The marches, the vector one over only the walls some ray of the group can
 * reach. A ray i samples points at t < R_i = min(max, limit[i]) from its
 * origin O_i, so within R = max_i(R_i + |O_i - B|) of the body B; a wall
 * with SDF(B) > R + 0.1 is, at every sample, more than R_i - t + 0.1 away
 * (the SDF is 1-Lipschitz). Such a wall is never a hit (> 0.1 > 0.05), and
 * when it is the minimum the step takes t past R_i with or without it, so
 * the ray ends on the next check either way: with max both ways when
 * R_i = max, else with limit or max, which every caller reads only through
 * `< limit` (cast_group's wall_d <= best < limit, the probes' d > threshold
 * with limit just above it) or passes limit == max. Whenever the culled wall
 * is not the minimum the minimum is unchanged. */
static void bopl4_march(const Bopl4Terrain *t, const double *ox, const double *oy, int n, const double *dx, const double *dy,
                        double max, const double *limit, double *out) {
#if BOT_SDF_AVX512
    if (bot_cpu_avx512>0 && !bopl4_march_scalar_only) {
        const Bopl4Walls *v=&t->walls;
        Bopl4Walls kept;
        if (t->anchored) {
            double reach=0;
            for (int i=0; i<n; i++) {
                double ex=ox[i]-t->bx, ey=oy[i]-t->by, r=(limit[i]<max?limit[i]:max)+sqrt(ex*ex+ey*ey);
                if (r>reach) reach=r;
            }
            reach+=0.1;
            kept.axis_count=0; kept.rot_count=0;
            for (int k=0; k<v->axis_count; k++) if (!(t->axis_body[k]>reach)) {
                int a=kept.axis_count++;
                kept.axis_cx[a]=v->axis_cx[k]; kept.axis_cy[a]=v->axis_cy[k]; kept.axis_ex[a]=v->axis_ex[k];
                kept.axis_ey[a]=v->axis_ey[k]; kept.axis_radius[a]=v->axis_radius[k];
            }
            for (int k=0; k<v->rot_count; k++) if (!(t->rot_body[k]>reach)) {
                int r=kept.rot_count++;
                kept.rot_cx[r]=v->rot_cx[k]; kept.rot_cy[r]=v->rot_cy[k]; kept.rot_rx[r]=v->rot_rx[k]; kept.rot_ry[r]=v->rot_ry[k];
                kept.rot_ux[r]=v->rot_ux[k]; kept.rot_uy[r]=v->rot_uy[k]; kept.rot_ex[r]=v->rot_ex[k]; kept.rot_ey[r]=v->rot_ey[k];
                kept.rot_radius[r]=v->rot_radius[k];
            }
            if (kept.axis_count+kept.rot_count<v->axis_count+v->rot_count) v=&kept;
        }
        bopl4_march_avx512(v, ox, oy, n, dx, dy, max, limit, out);
        return;
    }
#endif
    bopl4_march_scalar(&t->view, ox, oy, n, dx, dy, max, limit, out);
}
/* The eighth of the circle (atan2(dy, dx)+pi)*8/(2 pi) falls in, clamped to
 * 0-7. Away from the octant edges (the axes and diagonals) it is read off the
 * signs and which of |dx|, |dy| is larger: there the float atan2 is ~1e-16
 * from the exact angle and the edges are 1e-6 away, so the two agree. Near an
 * edge it takes the atan2 path as written. */
static inline int bopl4_sector(double dx, double dy) {
    double ax=fabs(dx), ay=fabs(dy), big=fmax(ax, ay);
    if (fmin(ax, ay)<=1e-6*big || fabs(ax-ay)<=1e-6*big) {
        double angle=atan2(dy, dx)+3.14159265358979323846;
        int sector=(int)(angle*(8.0/6.28318530717958647692));
        if (sector<0) sector=0; else if (sector>7) sector=7;
        return sector;
    }
    int steep=ay>ax;
    if (dy<0) return dx<0 ? (steep?1:0) : (steep?2:3);
    return dx>0 ? (steep?5:4) : (steep?6:7);
}
/* Classes: 0 wall, 1 lethal wall, 2 water / blast plane, 3 hostile projectile
 * or object, 4 opponent body, 5 own object; out[7] the hit's closing speed
 * along the ray for classes 3-5. The surfaces other than walls are closed
 * form (bopl4_ray_other); the walls' march comes from bopl4_march. */
/* The circles relative to a group's origin, taken once for its rays:
 * centre offset, its squared length, and the squared radius. */
typedef struct { double px[BOPL4_ENTITIES+BOPL4_SLIMES], py[BOPL4_ENTITIES+BOPL4_SLIMES],
                 pp[BOPL4_ENTITIES+BOPL4_SLIMES], r2[BOPL4_ENTITIES+BOPL4_SLIMES]; } Bopl4Circles;
static void bopl4_circles_from(const Bopl4Terrain *t, double x, double y, Bopl4Circles *c) {
    for (int i=0; i<t->circle_count; i++) {
        double px=t->cx[i]-x, py=t->cy[i]-y;
        c->px[i]=px; c->py[i]=py; c->pp[i]=px*px+py*py; c->r2[i]=t->cr[i]*t->cr[i];
    }
}
static double bopl4_ray_other(const Bopl4Terrain *t, const Bopl4Circles *c, double x, double y, double dx, double dy, double svx, double svy,
                              int *class, double *closing) {
    double best=BOPL4_RAY_LENGTH; *class=-1; *closing=0;
    if (t->has_water && dy<-1e-9) {
        double d=(t->water_top-y)/dy;
        if (d>=0 && d<best) { best=d; *class=2; }
    }
    if (dy>1e-9) { double d=(BOT_BLAST_Y-y)/dy; if (d>=0 && d<best) { best=d; *class=2; } }
    if (dx>1e-9) { double d=(BOT_BLAST_X-x)/dx; if (d>=0 && d<best) { best=d; *class=2; } }
    if (dx<-1e-9) { double d=(-BOT_BLAST_X-x)/dx; if (d>=0 && d<best) { best=d; *class=2; } }
    for (int i=0; i<t->circle_count; i++) {
        double along=c->px[i]*dx+c->py[i]*dy;
        if (along<0) continue;
        double perp2=c->pp[i]-along*along, r2=c->r2[i];
        if (perp2>r2) continue;
        double d=along-sqrt(r2-perp2);
        if (d<0) d=0;
        if (d<best) { best=d; *class=3+t->cclass[i]; *closing=-((t->cvx[i]-svx)*dx+(t->cvy[i]-svy)*dy); }
    }
    return best;
}
/* Finish a ray given the other surfaces' best and the walls' marched distance. */
static void bopl4_ray_finish(const Bopl4Terrain *t, double x, double y, double dx, double dy,
                             double best, int class, double closing, double wall_d, float *out) {
    if (t->view.wall_count && wall_d<BOPL4_RAY_LENGTH && wall_d<=best) {
        /* Walls were tested first originally, the other surfaces after with a
         * strict `d < best`: a wall at exactly a later surface's distance kept
         * the wall, hence <= here. */
        best=wall_d;
        int w=bopl4_nearest_wall(&t->view, x+dx*wall_d, y+dy*wall_d);
        class=(w>=0 && t->view.walls[w].standable<0)?1:0;
        closing=0;
    }
    out[0]=(float)(best/BOPL4_RAY_LENGTH);
    for (int c=0; c<6; c++) out[1+c]=class==c;
    out[7]=class>=3?bopl_clip(closing/BOPL4_VELOCITY_SCALE):0.0f;
}
/* Up to BOPL4_MARCH rays from (x, y) along (dx, dy)[i] into out[i]: the
 * closed-form surfaces per ray, then the wall marches together, each cut off
 * once it passes the ray's nearest other surface. */
static void bopl4_cast_group(const Bopl4Terrain *t, double x, double y, double svx, double svy, int n,
                             const double *dx, const double *dy, float *const *out) {
    double best[BOPL4_MARCH], closing[BOPL4_MARCH], limit[BOPL4_MARCH], wall[BOPL4_MARCH], ox[BOPL4_MARCH], oy[BOPL4_MARCH];
    int class[BOPL4_MARCH];
    Bopl4Circles circles;
    bopl4_circles_from(t, x, y, &circles);
    for (int i=0; i<n; i++) {
        ox[i]=x; oy[i]=y;
        best[i]=bopl4_ray_other(t, &circles, x, y, dx[i], dy[i], svx, svy, &class[i], &closing[i]);
        limit[i]=best[i]+1.0;
        wall[i]=BOPL4_RAY_LENGTH;
    }
    if (t->view.wall_count) bopl4_march(t, ox, oy, n, dx, dy, BOPL4_RAY_LENGTH, limit, wall);
    for (int i=0; i<n; i++) bopl4_ray_finish(t, x, y, dx[i], dy[i], best[i], class[i], closing[i], wall[i], out[i]);
}
/* `count` rays of the fan (indices k[]) from (x, y), in groups of BOPL4_MARCH. */
static void bopl4_rays(const Bopl4Terrain *t, double x, double y, double svx, double svy, const int *k, int count, float *obs) {
    for (int start=0; start<count; start+=BOPL4_MARCH) {
        int n=count-start<BOPL4_MARCH?count-start:BOPL4_MARCH;
        double dx[BOPL4_MARCH], dy[BOPL4_MARCH]; float *out[BOPL4_MARCH];
        for (int i=0; i<n; i++) {
            bopl4_ray_direction(k[start+i], &dx[i], &dy[i]);
            out[i]=obs+BOPL4_OBS_RAY_OFFSET+(size_t)k[start+i]*BOPL4_RAY_FEATURES;
        }
        bopl4_cast_group(t, x, y, svx, svy, n, dx, dy, out);
    }
}
/* Layout v4.1 (BOPL4_ABLATE_AIM): in ray slot BOPL4_AIM_RAY the aim ray, cast
 * from the body along the aim of the decision applied this step (the direction
 * the next arrow follows; zero when that decision had no aim), the fan's 8
 * fields: its class one-hot says what the shot meets first, the opponent
 * body class being "the shot is on target". In slot BOPL4_LOS_RAY the line of
 * sight: the cast toward the opponent's centre, so the opponent-body class is
 * a clear line and any other class names what blocks it (a wall or lethal
 * wall, water, a projectile or object) at that distance; zero when the
 * opponent's position is unknown. */
static void bopl4_aim_rays(const Bopl4Terrain *t, double x, double y, double svx, double svy, BoplRlAction applied,
                           const Bopl4Enemy *enemy, float *obs) {
    double dx[2], dy[2]; float *out[2]; int n=0;
    float *aim=obs+BOPL4_OBS_RAY_OFFSET+(size_t)BOPL4_AIM_RAY*BOPL4_RAY_FEATURES;
    float *los=obs+BOPL4_OBS_RAY_OFFSET+(size_t)BOPL4_LOS_RAY*BOPL4_RAY_FEATURES;
    memset(aim, 0, BOPL4_RAY_FEATURES*sizeof(float)); memset(los, 0, BOPL4_RAY_FEATURES*sizeof(float));
    if (applied.angle) {
        pthread_once(&bopl_aim_table_once, bopl_aim_table_init);   /* the first step's rows precede the tail that fills it */
        dx[n]=bopl_aim_table[applied.angle][0]; dy[n]=bopl_aim_table[applied.angle][1]; out[n++]=aim;
    }
    if (enemy->known) {
        double ex=enemy->x-x, ey=enemy->y-y, len=sqrt(ex*ex+ey*ey);
        if (len>1e-9) { dx[n]=ex/len; dy[n]=ey/len; out[n++]=los; }
    }
    if (n) bopl4_cast_group(t, x, y, svx, svy, n, dx, dy, out);
}
static double bopl4_drop(const Bopl4Terrain *t, double x, double y, double cap) {
    double d=t->view.wall_count?bot_raycast(&t->view, x, y, 0, -1, cap):cap;
    if (t->has_water && y>t->water_top && y-t->water_top<d) d=y-t->water_top;
    return d;
}
/* bopl4_drop(...) > limit, and bopl4_drop(...) <= limit, without the full
 * march: the drop is min(raycast, water distance), so it exceeds the limit
 * iff both do. */
static int bopl4_drop_exceeds(const Bopl4Terrain *t, double x, double y, double cap, double limit) {
    if (t->has_water && y>t->water_top && !(y-t->water_top>limit)) return 0;
    return t->view.wall_count?bot_raycast_exceeds(&t->view, x, y, 0, -1, cap, limit):cap>limit;
}
/* The 16 dash directions of the landing test, as bopl4_terrain computed them. */
static void bopl4_dash_direction(int k, double *dx, double *dy) {
    static __thread double table[16][2];
    static __thread int ready=0;
    if (!ready) {
        for (int i=0; i<16; i++) { double angle=6.28318530717958647692*i/16.0; table[i][0]=cos(angle); table[i][1]=sin(angle); }
        ready=1;
    }
    *dx=table[k][0]; *dy=table[k][1];
}
/* The egocentric terrain block (BOPL4_TERRAIN) at body position (x, y).
 * Every raycast here is one of bot_raycast's, marched in bopl4_march groups
 * where the casts are independent; "exceeds" tests march with the limit just
 * above the threshold (out > threshold then decides: a cut-off march has
 * t > threshold, a finished one its exact result), and probes the original
 * stopped before are evaluated without changing what it wrote. */
static void bopl4_terrain(const Bopl4Terrain *t, double x, double y, float *out) {
    const BotView *view=&t->view;
    double drop=32.0, ceiling=32.0;
    if (view->wall_count) {
        double ox[2]={x, x}, oy[2]={y, y}, dx[2]={0, 0}, dy[2]={-1, 1}, limit[2]={32.0, 32.0}, d[2];
        bopl4_march(t, ox, oy, 2, dx, dy, 32.0, limit, d);
        drop=d[0]; ceiling=d[1];
    }
    if (t->has_water && y>t->water_top && y-t->water_top<drop) drop=y-t->water_top;
    out[0]=(float)(drop/32.0); out[1]=(float)(ceiling/32.0);
    /* Edges: walk left/right in 1.5-unit steps to one dash length; the edge
     * is where the floor drops away by more than 4 units (or ends in water),
     * a wall is where the step lands inside terrain. */
    {
        int first_wall[2]={9, 9};
        for (int side=0; side<2 && view->wall_count; side++) {
            double sign=side?1.0:-1.0;
            for (int k=1; k<=8; k++) if (bot_sdf(view, x+sign*1.5*k, y)<0.3) { first_wall[side]=k; break; }
        }
        int first_drop[2]={9, 9};
        double threshold=drop+4.0;
        /* bopl4_drop_exceeds: water within the threshold below this height
         * settles every probe (no edge by drop); no walls, the cap decides. */
        if (t->has_water && y>t->water_top && !(y-t->water_top>threshold)) {
        } else if (!view->wall_count) {
            if (32.0>threshold) first_drop[0]=first_drop[1]=1;
        } else {
            double above=nextafter(threshold, HUGE_VAL);
            double ox[BOPL4_MARCH], oy[BOPL4_MARCH], dx[BOPL4_MARCH], dy[BOPL4_MARCH], limit[BOPL4_MARCH], d[BOPL4_MARCH];
            int side_of[BOPL4_MARCH], k_of[BOPL4_MARCH], n=0;
            for (int side=0; side<2; side++) {
                double sign=side?1.0:-1.0;
                for (int k=1; k<first_wall[side]; k++) {
                    ox[n]=x+sign*1.5*k; oy[n]=y; dx[n]=0; dy[n]=-1; limit[n]=above; side_of[n]=side; k_of[n]=k; n++;
                    if (n==BOPL4_MARCH || (side==1 && k==first_wall[1]-1)) {
                        bopl4_march(t, ox, oy, n, dx, dy, 32.0, limit, d);
                        for (int i=0; i<n; i++) if (d[i]>threshold && k_of[i]<first_drop[side_of[i]]) first_drop[side_of[i]]=k_of[i];
                        n=0;
                    }
                }
            }
            if (n) {
                bopl4_march(t, ox, oy, n, dx, dy, 32.0, limit, d);
                for (int i=0; i<n; i++) if (d[i]>threshold && k_of[i]<first_drop[side_of[i]]) first_drop[side_of[i]]=k_of[i];
            }
        }
        for (int side=0; side<2; side++) {
            int k=first_wall[side]<first_drop[side]?first_wall[side]:first_drop[side];
            out[2+side]=(float)((k<=8?1.5*k:BOPL4_DASH_LENGTH)/BOPL4_DASH_LENGTH);
            out[4+side]=first_wall[side]<=8 && first_wall[side]<=first_drop[side];
        }
    }
    out[6]=t->has_water?bopl_clip((y-t->water_top)/32.0):1.0f;
    double lethal=16.0;
    for (int i=0; i<view->wall_count; i++) if (view->walls[i].standable<0) {
        double d=bot_wall_sdf(&view->walls[i], x, y); if (d<lethal) lethal=d;
    }
    out[7]=(float)(lethal/16.0);
    /* Ground within one dash: some dash endpoint (16 directions, stopped at
     * terrain) has a surface within 4 units below it. Groups of BOPL4_MARCH
     * directions in order, stopping after the group that lands. */
    int landing=drop<=4.0;
    for (int start=0; start<16 && !landing; start+=BOPL4_MARCH) {
        int n=16-start<BOPL4_MARCH?16-start:BOPL4_MARCH;
        double ox[BOPL4_MARCH], oy[BOPL4_MARCH], dx[BOPL4_MARCH], dy[BOPL4_MARCH], limit[BOPL4_MARCH], d[BOPL4_MARCH];
        for (int i=0; i<n; i++) { bopl4_dash_direction(start+i, &dx[i], &dy[i]); ox[i]=x; oy[i]=y; limit[i]=BOPL4_DASH_LENGTH; d[i]=BOPL4_DASH_LENGTH; }
        if (view->wall_count) bopl4_march(t, ox, oy, n, dx, dy, BOPL4_DASH_LENGTH, limit, d);
        double ex[BOPL4_MARCH], ey[BOPL4_MARCH], zero[BOPL4_MARCH], down[BOPL4_MARCH], above[BOPL4_MARCH], fall[BOPL4_MARCH];
        int m=0;
        for (int i=0; i<n; i++) {
            double px=x+dx[i]*(d[i]-0.5), py=y+dy[i]*(d[i]-0.5);
            if (t->has_water && py<=t->water_top) continue;
            if (t->has_water && py>t->water_top && !(py-t->water_top>4.0)) { landing=1; continue; }   /* water within 4: a landing */
            if (!view->wall_count) continue;                                                        /* 8 > 4: no landing */
            ex[m]=px; ey[m]=py; zero[m]=0; down[m]=-1; above[m]=nextafter(4.0, HUGE_VAL); m++;
        }
        if (m && !landing) {
            bopl4_march(t, ex, ey, m, zero, down, 8.0, above, fall);
            for (int i=0; i<m; i++) if (!(fall[i]>4.0)) landing=1;
        }
    }
    out[8]=landing;
    double sdf=view->wall_count?bot_sdf(view, x, y):8.0;
    out[9]=bopl_clip(sdf/8.0);
    if (view->wall_count) {
        double h=0.05, gx=(bot_sdf(view, x+h, y)-bot_sdf(view, x-h, y))/(2*h), gy=(bot_sdf(view, x, y+h)-bot_sdf(view, x, y-h))/(2*h);
        out[10]=bopl_clip(gx); out[11]=bopl_clip(gy);
    } else out[10]=out[11]=0;
}
/* Encode `state` into player a's v4 row. Same contract as the v3 row writer:
 * every feature stored, unused set slots zeroed (with the bopl_written record
 * of the previous encode), the own-action tail zeroed and the privileged block
 * left for bopl4_privileged. */
static void bopl_encode_v4_row(BoplPuffer *env, const BoplRlState *state, int a, int count_overflow) {
    bopl4_row_ablation=bopl_ablate_of(env, a);   /* this player's writer family for the whole row */
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
    double self_vx=controller?bopl_fix(controller->vx):0.0, self_vy=controller?bopl_fix(controller->vy):0.0;
    Bopl4Enemy enemy=bopl4_primary_enemy(state, self, self_x, self_y);
    const BoplRlSlime *enemy_controller=enemy.player?bopl_controlling_slime(state, enemy.player->id, enemy.x, enemy.y):NULL;

    uint32_t nearest[BOPL4_ENTITIES]; double distances[BOPL4_ENTITIES];
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
            bopl_nearest(nearest, distances, &count, BOPL4_ENTITIES, i+(uint32_t)j, distance);
            active_count++;
            uint32_t source=entity->reserved[0];
            if (source>0 && source<=BOPL_ABILITY_COUNT) source_counts[source-1]++;
            if (entity->inverse_mass>0) {
                n_nonstatic++;
                double dx=bopl_fix(entity->x)-self_x, dy=bopl_fix(entity->y)-self_y;
                sectors[bopl4_sector(dx, dy)]++;
            }
            if (entity->flags&BOPL_ENTITY_PICKUP) n_pickup++;
            if (!time_stop_known && state->header.player_time_scale==0
                && (entity->flags&BOPL_ENTITY_TIMER_KNOWN) && entity->clock_rate>0) {
                time_stop_remaining=entity->reserved[1]; time_stop_known=1;
            }
        }
        i=(uint32_t)end;
    }

    uint32_t nearby_slimes[BOPL4_SLIMES]; double slime_distances[BOPL4_SLIMES];
    int slime_count=0, visible_slimes=0;
    uint32_t slot_indices[BOPL4_SLOTS]; double slot_distances[BOPL4_SLOTS];
    int slot_count=0, visible_slots=0;
    for (uint32_t i=0; i<state->header.slime_count; i++) {
        const BoplRlSlime *slime=&state->slimes[i];
        int hidden=bopl_spatial_hidden(state, slime->owner, self->id);
        int forced=slime==controller || slime==enemy_controller;
        if (hidden && !forced) continue;
        double dx=bopl_fix(slime->x)-self_x, dy=bopl_fix(slime->y)-self_y;
        double distance=dx*dx+dy*dy;
        if (!hidden) {
            bopl_nearest(nearby_slimes, slime_distances, &slime_count, BOPL4_SLIMES, i, distance);
            visible_slimes++;
        }
        for (int slot=0; slot<3; slot++) {
            uint32_t id=slime->abilities[slot], flags=slime->ability_flags[slot];
            if (!(flags&1) || id==0 || id>BOPL_ABILITY_COUNT) continue;
            double priority=distance+slot*1.0e-9;
            if (slime==controller) priority-=1.0e12; else if (slime==enemy_controller) priority-=1.0e11;
            bopl_nearest(slot_indices, slot_distances, &slot_count, BOPL4_SLOTS, i*3u+(uint32_t)slot, priority);
            visible_slots++;
        }
    }

    /* Global block: the v3 prefix. */
    obs[0]=bopl_clip(bopl_fix(state->header.simulation_time)/120.0);
    obs[1]=bopl_clip(bopl_fix(state->header.player_time_scale));
    obs[2]=bopl_clip((double)state->header.episode_ticks/env->config.max_episode_ticks);
    obs[3]=state->header.player_count/6.0f;
    obs[4]=bopl_clip((double)active_count/BOPL4_ENTITIES);
    obs[5]=bopl_clip((double)(active_count-count)/BOPL4_ENTITIES);
    obs[6]=bopl_clip((double)visible_slimes/BOPL4_SLIMES);
    obs[7]=bopl_clip((double)(visible_slimes>BOPL4_SLIMES?visible_slimes-BOPL4_SLIMES:0)/BOPL4_SLIMES);
    /* v3's absolute self position was a map lookup key and is dropped (the
     * terrain block replaces it). obs[8] reuses the first dead column for the
     * one mechanical map property: LevelType.space, which halves the gravity
     * modifier on players and ropes, removes water and moves the lower blast
     * boundary. Every non-space map leaves it 0, exactly as the dead column
     * was, so nothing measured on grass or snow changes. Projectile arcs are
     * NOT affected by space (gravityAccel carries no level-type factor), so
     * BOPL4_GRAVITY stays one constant. */
    obs[8]=(float)bopl_map_selected(BOPL_SPACE_MAPS, env->config.map); obs[9]=0;
    double level_time=bopl_fix(state->header.simulation_time_since_level_loaded);
    obs[10]=bopl_clip(fmax(0.0, 120.0-level_time)/120.0);
    obs[11]=state->header.sudden_death!=0;
    obs[12]=state->header.sudden_death?bopl_clip(fmax(0.0, level_time-120.0)/30.0):0.0f;
    obs[13]=bopl_clip((env->config.max_episode_ticks-(double)state->header.episode_ticks)/env->config.max_episode_ticks);
    for (int id=0; id<BOPL_ABILITY_COUNT; id++) obs[14+id]=bopl_clip(source_counts[id]/(double)BOPL_COUNT_SCALE);
    obs[53]=bopl_clip(n_nonstatic/(double)BOPL4_ENTITIES);
    obs[54]=bopl_clip(n_pickup/(double)BOPL_COUNT_SCALE);
    for (int sector=0; sector<8; sector++) obs[55+sector]=bopl_clip(sectors[sector]/(double)BOPL_COUNT_SCALE);
    obs[63]=time_stop_known?bopl_clip(time_stop_remaining/300.0):0.0f;
    obs[64]=time_stop_known;
    obs[65]=controller?((controller->flags>>BOPL_SLIME_PICKUP_SLOT_SHIFT)&BOPL_SLIME_PICKUP_SLOT_MASK)/2.0f:0.0f;
    if (count_overflow) {
        bopl_note_overflow(env, active_count, count, visible_slimes, visible_slots);
        bopl_note_entity_rows(env,state,nearest,count,source_counts);
    }

    /* Self: the v3 player row about oneself, then the controlling body. */
    float *self_block=obs+BOPL4_OBS_SELF_OFFSET;
    bopl_player_features(self_block, state, self, self, self_x, self_y);
    bopl4_body(self_block+BOPL_PLAYER_FEATURES, controller);
    /* Opponent: the primary enemy's player row and body, its relation to us. */
    float *opp=obs+BOPL4_OBS_OPPONENT_OFFSET;
    memset(opp, 0, BOPL4_OPPONENT*sizeof(float));
    if (enemy.player) {
        bopl_player_features(opp, state, enemy.player, self, self_x, self_y);
        bopl4_body(opp+BOPL_PLAYER_FEATURES, enemy.slime);
        if (enemy.known) bopl4_relation(opp+BOPL4_SELF, enemy.x-self_x, enemy.y-self_y, enemy.vx-self_vx, enemy.vy-self_vy, 0.0, 0);
    }
    memset(obs+BOPL4_OBS_PREVIOUS_OFFSET, 0, BOPL4_PREVIOUS*sizeof(float));
    /* Terrain and rays: the egocentric view of the map from the body. */
    if (bopl4_ablation()&BOPL4_ABLATE_RAYS) {
        memset(obs+BOPL4_OBS_TERRAIN_OFFSET, 0, (BOPL4_TERRAIN+BOPL4_RAYS*BOPL4_RAY_FEATURES)*sizeof(float));
        /* Bits 2|8 (v4_ablate 10): the aim ray and line of sight alone, no fan
         * and no terrain block (the two casts on an otherwise empty channel). */
        if (bopl4_ablation()&BOPL4_ABLATE_AIM) {
            Bopl4Terrain terrain;
            bopl4_terrain_fill(&terrain, state, nearest, count, nearby_slimes, slime_count, self);
            bopl4_terrain_anchor(&terrain, self_x, self_y);
            bopl4_aim_rays(&terrain, self_x, self_y, self_vx, self_vy, env->applied[a], &enemy, obs);
        }
    } else {
        Bopl4Terrain terrain;
        int rays16=bopl4_ablation()&BOPL4_ABLATE_RAYS16;   /* every other of the 32, none of the down fan */
        bopl4_terrain_fill(&terrain, state, nearest, count, nearby_slimes, slime_count, self);
        bopl4_terrain_anchor(&terrain, self_x, self_y);
        bopl4_terrain(&terrain, self_x, self_y, obs+BOPL4_OBS_TERRAIN_OFFSET);
        int wanted[BOPL4_RAYS], n_wanted=0;
        for (int k=0; k<BOPL4_RAYS; k++) {
            if (rays16 && (k>=BOPL4_RAYS_AROUND || (k&1))) { memset(obs+BOPL4_OBS_RAY_OFFSET+(size_t)k*BOPL4_RAY_FEATURES, 0, BOPL4_RAY_FEATURES*sizeof(float)); continue; }
            wanted[n_wanted++]=k;
        }
        bopl4_rays(&terrain, self_x, self_y, self_vx, self_vy, wanted, n_wanted, obs);
        if (bopl4_ablation()&BOPL4_ABLATE_AIM) bopl4_aim_rays(&terrain, self_x, self_y, self_vx, self_vy, env->applied[a], &enemy, obs);
    }

    uint32_t player_count=state->header.player_count<BOPL4_PLAYERS?state->header.player_count:BOPL4_PLAYERS;
    for (uint32_t i=0; i<player_count; i++) {
        uint32_t index=i==0?(uint32_t)a:(i<=(uint32_t)a?i-1:i);
        const BoplRlPlayer *player=&state->players[index];
        float *row=obs+BOPL4_OBS_PLAYER_OFFSET+(size_t)i*BOPL4_PLAYER_FEATURES;
        bopl_player_features(row, state, player, self, self_x, self_y);
        memset(row+BOPL_PLAYER_FEATURES, 0, BOPL4_RELATION*sizeof(float));
        if (!bopl_spatial_hidden(state, player->id, self->id)) {
            double x=bopl_fix(player->x), y=bopl_fix(player->y);
            const BoplRlSlime *body=bopl_controlling_slime(state, player->id, x, y);
            bopl4_relation(row+BOPL_PLAYER_FEATURES, x-self_x, y-self_y,
                           (body?bopl_fix(body->vx):0.0)-self_vx, (body?bopl_fix(body->vy):0.0)-self_vy, 0.0, 0);
        }
    }
    bopl_zero_tail(obs+BOPL4_OBS_PLAYER_OFFSET, BOPL4_PLAYER_FEATURES, (int)player_count, prev_players, BOPL4_PLAYERS);

    for (int i=0; i<slot_count; i++) {
        uint32_t packed=slot_indices[i], slime_index=packed/3, slot=packed%3;
        const BoplRlSlime *slime=&state->slimes[slime_index];
        uint32_t flags=slime->ability_flags[slot], id=slime->abilities[slot];
        double cooldown=bopl_fix(slime->cooldowns[slot]), timer=bopl_fix(slime->timers[slot]);
        int hidden=bopl_spatial_hidden(state, slime->owner, self->id);
        double dx=bopl_fix(slime->x)-self_x, dy=bopl_fix(slime->y)-self_y;
        float row[BOPL4_SLOT_FEATURES]={0};
        row[0]=1; row[1]=slime->owner==self->id; row[2]=bopl_ally(state, slime->owner, self);
        row[3]=slot/2.0f; row[4]=bopl_clip(cooldown>0?timer/cooldown:0.0);
        row[5]=bopl_clip(cooldown/60.0); row[6]=(flags&2)!=0; row[7]=(flags&4)!=0;
        row[8]=(flags&BOPL_ABILITY_ACTIVE)!=0;
        row[10]=id==4;
        row[9]=row[10]?((flags>>BOPL_ABILITY_CHARGE_SHIFT)&BOPL_ABILITY_CHARGE_MASK)/4.0f:0.0f;
        if (!hidden) { row[11]=bopl_clip(dx/64.0); row[12]=bopl_clip(dy/64.0); row[16]=bopl_clip(sqrt(dx*dx+dy*dy)/64.0); }
        row[13]=bopl_clip(fmax(0.0, cooldown-timer)/10.0);
        row[14]=!hidden;
        row[15]=enemy.player && slime->owner==enemy.player->id;
        row[BOPL4_SLOT_ABILITY_ID]=(float)id;
        memcpy(obs+BOPL4_OBS_SLOT_OFFSET+(size_t)i*BOPL4_SLOT_FEATURES, row, sizeof(row));
    }
    bopl_zero_tail(obs+BOPL4_OBS_SLOT_OFFSET, BOPL4_SLOT_FEATURES, slot_count, prev_slots, BOPL4_SLOTS);

    for (int i=0; i<count; i++) {
        const BoplRlEntity *entity=&state->entities[nearest[i]];
        float row[BOPL4_ENTITY_FEATURES]={0};
        double ex=bopl_fix(entity->x), ey=bopl_fix(entity->y), evx=bopl_fix(entity->vx), evy=bopl_fix(entity->vy);
        double radius=bopl_fix(entity->radius);
        row[0]=1; if (entity->kind<5) row[1+entity->kind]=1;
        row[6]=entity->owner==self->id; row[7]=bopl_ally(state, entity->owner, self);
        row[8]=entity->owner!=0 && !row[6] && !row[7];
        row[9]=bopl_clip((ex-self_x)/64.0); row[10]=bopl_clip((ey-self_y)/64.0);
        row[11]=bopl_clip(evx/BOPL4_VELOCITY_SCALE); row[12]=bopl_clip(evy/BOPL4_VELOCITY_SCALE);
        row[13]=bopl_clip(bopl_fix(entity->right_x)/64.0); row[14]=bopl_clip(bopl_fix(entity->right_y)/64.0);
        row[15]=bopl_clip(bopl_fix(entity->up_x)/64.0); row[16]=bopl_clip(bopl_fix(entity->up_y)/64.0);
        row[17]=bopl_clip(radius/16.0);
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
        int projectile=(entity->flags&BOPL_ENTITY_PROJECTILE)!=0 && entity->inverse_mass>0;
        /* Relation to the observer: for segments (ropes, platform edges) the
         * surface distance is the segment's, the direction is to the centre. */
        bopl4_relation(row+33, ex-self_x, ey-self_y, evx-self_vx, evy-self_vy, radius, projectile);
        row[33]=bopl_clip(bopl_surface_distance(entity, self_x, self_y)/64.0);
        if (enemy.known) {
            /* The same body seen from the primary enemy: is it about to hit them. */
            float versus[BOPL4_RELATION];
            bopl4_relation(versus, ex-enemy.x, ey-enemy.y, evx-enemy.vx, evy-enemy.vy, radius, projectile);
            row[45]=versus[0]; row[46]=projectile?versus[10]:versus[8];
        }
        row[BOPL4_ENTITY_ABILITY_ID]=entity->reserved[0]<=BOPL_ABILITY_COUNT?(float)entity->reserved[0]:0.0f;
        memcpy(obs+BOPL4_OBS_ENTITY_OFFSET+(size_t)i*BOPL4_ENTITY_FEATURES, row, sizeof(row));
    }
    bopl_zero_tail(obs+BOPL4_OBS_ENTITY_OFFSET, BOPL4_ENTITY_FEATURES, count, prev_entities, BOPL4_ENTITIES);

    for (int i=0; i<slime_count; i++) {
        const BoplRlSlime *slime=&state->slimes[nearby_slimes[i]];
        const BoplRlPlayer *owner=bopl_player_by_id(state, slime->owner);
        float row[BOPL4_SLIME_FEATURES]={0};
        double sx=bopl_fix(slime->x), sy=bopl_fix(slime->y), svx=bopl_fix(slime->vx), svy=bopl_fix(slime->vy);
        row[0]=1; row[1]=slime->owner==self->id; row[2]=bopl_ally(state, slime->owner, self);
        row[3]=(slime->flags&BOPL_SLIME_ACTIVE)!=0; row[4]=(slime->flags&BOPL_SLIME_SPAWNED)!=0;
        row[5]=(slime->flags&BOPL_SLIME_IN_ABILITY)!=0; row[6]=(slime->flags&BOPL_SLIME_SUPPRESS_INPUT)!=0;
        row[7]=bopl_clip((sx-self_x)/64.0); row[8]=bopl_clip((sy-self_y)/64.0);
        row[9]=bopl_clip(svx/BOPL4_VELOCITY_SCALE); row[10]=bopl_clip(svy/BOPL4_VELOCITY_SCALE);
        row[11]=bopl_clip(bopl_fix(slime->scale)/3.0);
        if (slime->state<11) row[12+slime->state]=1;
        row[23]=sinf((float)bopl_fix(slime->rotation)); row[24]=cosf((float)bopl_fix(slime->rotation));
        row[25]=owner?bopl_clip(owner->stocks/10.0):0.0f;
        row[26]=enemy.player && slime->owner==enemy.player->id;
        row[27]=slime==controller || slime==enemy_controller;   /* a controlling body */
        row[28]=(slime->flags&BOPL_SLIME_ABILITYLESS)!=0; row[29]=1;
        bopl4_relation(row+30, sx-self_x, sy-self_y, svx-self_vx, svy-self_vy, 0.0, 0);
        memcpy(obs+BOPL4_OBS_SLIME_OFFSET+(size_t)i*BOPL4_SLIME_FEATURES, row, sizeof(row));
    }
    bopl_zero_tail(obs+BOPL4_OBS_SLIME_OFFSET, BOPL4_SLIME_FEATURES, slime_count, prev_slimes, BOPL4_SLIMES);
    if (w<0) {
        if (env->written_count==BOPL_MAX_SLOTS) env->written[0]=env->written[--env->written_count];
        w=env->written_count++; env->written[w].row=obs;
    }
    env->written[w].players=(int)player_count; env->written[w].slots=slot_count;
    env->written[w].entities=count; env->written[w].slimes=slime_count;
}
/* The critic-only block of player a's v4 row from the undelayed `state`:
 * the primary enemy's true state and its latch (held buttons, hold left, last
 * aim; NULLs when unknown), and the observer's own undelayed position and
 * velocity next to the delayed one the actor sees. */
static void bopl4_privileged(BoplPuffer *env, const BoplRlState *state, int a,
                             const BoplRlAction *enemy_action, const int *enemy_hold_left, const uint16_t *enemy_held) {
    float *out=bopl_obs_row(env, a)+BOPL4_OBS_PRIVILEGED_OFFSET;
    memset(out, 0, BOPL4_PRIVILEGED*sizeof(float));
    if (!state) return;
    const BoplRlPlayer *self=&state->players[a];
    double self_x=bopl_fix(self->x), self_y=bopl_fix(self->y);
    const BoplRlSlime *controller=bopl_controlling_slime(state, self->id, self_x, self_y);
    const BoplRlPlayer *enemy=NULL;
    for (uint32_t i=0; i<state->header.player_count && !enemy; i++)
        if (state->players[i].id!=self->id && state->players[i].team!=self->team) enemy=&state->players[i];
    if (enemy) {
        double x=bopl_fix(enemy->x), y=bopl_fix(enemy->y);
        const BoplRlSlime *body=bopl_controlling_slime(state, enemy->id, x, y);
        out[0]=1; out[1]=bopl_clip((x-self_x)/64.0); out[2]=bopl_clip((y-self_y)/64.0);
        if (body) {
            out[3]=bopl_clip(bopl_fix(body->vx)/BOPL4_VELOCITY_SCALE); out[4]=bopl_clip(bopl_fix(body->vy)/BOPL4_VELOCITY_SCALE);
            out[5]=sinf((float)bopl_fix(body->rotation)); out[6]=cosf((float)bopl_fix(body->rotation));
            if (body->state<11) out[7+body->state]=1;
            out[18]=(body->flags&BOPL_SLIME_IN_ABILITY)!=0; out[19]=(body->flags&BOPL_SLIME_SUPPRESS_INPUT)!=0;
            out[20]=(body->flags&BOPL_SLIME_SPAWNED)!=0;
            for (int slot=0; slot<3; slot++) {
                uint32_t flags=body->ability_flags[slot];
                double cooldown=bopl_fix(body->cooldowns[slot]), timer=bopl_fix(body->timers[slot]);
                float *s=out+23+4*slot;
                if (!(flags&1)) continue;
                s[0]=bopl_clip(fmax(0.0, cooldown-timer)/10.0); s[1]=timer>=cooldown;
                s[2]=(flags&BOPL_ABILITY_ACTIVE)!=0; s[3]=((flags>>BOPL_ABILITY_CHARGE_SHIFT)&BOPL_ABILITY_CHARGE_MASK)/4.0f;
            }
        }
        out[21]=(enemy->flags&4)?(float)bopl_invisibility_alpha(enemy):1.0f; out[22]=(enemy->flags&4)!=0;
        if (enemy_held) for (int bit=0; bit<4; bit++) out[35+bit]=(*enemy_held&(1<<bit))!=0;
        if (enemy_hold_left) out[39]=bopl_clip(*enemy_hold_left/(double)BOPL_HOLD_MAX);
        if (enemy_action) {
            pthread_once(&bopl_aim_table_once, bopl_aim_table_init);
            out[40]=enemy_action->angle==0;
            out[41]=bopl_aim_table[enemy_action->angle][0]; out[42]=bopl_aim_table[enemy_action->angle][1];
        }
        out[47]=(enemy->flags&1)!=0;
    }
    /* How stale the actor's delayed view of the enemy is: true relative
     * position now minus the opponent block's delayed one. */
    const float *opp=bopl_obs_row(env, a)+BOPL4_OBS_OPPONENT_OFFSET;
    if (enemy) {
        out[43]=bopl_clip((bopl_fix(enemy->x)-self_x)/64.0-opp[9]); out[44]=bopl_clip((bopl_fix(enemy->y)-self_y)/64.0-opp[10]);
    }
    if (controller) { out[45]=bopl_clip(bopl_fix(controller->vx)/BOPL4_VELOCITY_SCALE); out[46]=bopl_clip(bopl_fix(controller->vy)/BOPL4_VELOCITY_SCALE); }
}

#endif
