/* Representation-probe binding: the research evaluator (eval_binding.c, one
 * translation unit so the environment globals exist once) plus privileged
 * labels computed from the full sim state, the observation-row -> entity
 * mapping, relational geometry and raycast features. Everything here is
 * research tooling; nothing is read by the trainer. See PROBES.md.
 *
 * Build: research/probes/build.py (cc -I native/rl -I native/rl/puffer). */
#include "eval_binding.c"
#include <time.h>

/* ---- which state the published observation describes ------------------- */
/* With the nine-tick delay the row a player sees was encoded from the state
 * bopl_step_finish swapped into sample_copy; right after a reset (steps==0)
 * it is the live state. */
const BoplRlState *probe_observed_state(BoplPuffer *env) {
    if (!env->observation_delay || env->steps==0) return bopl_rl_state(env->world);
    return bopl_state_copy_get(&env->sample_copy);
}
/* Encode any state undelayed into `out` (players x BOPL_OBS_SIZE), the same
 * encoder as the published rows; the own-action tail is left zero. */
void probe_pack(BoplPuffer *env, const BoplRlState *state, float *out) {
    BoplPuffer scratch={0};
    scratch.config=env->config; scratch.players=env->players; scratch.num_agents=env->players; scratch.scratch=out;
    compute_observations_from(&scratch, state);
}
/* The entity ids behind the entity rows of player a's observation of `state`,
 * in row order: the selection loop of bopl_encode (visibility, rope collapse,
 * surface distance, forced pickups). Returns the row count. */
int probe_entity_rows(const BoplRlState *state, int a, uint32_t *ids, int capacity) {
    if (!state || a<0 || (uint32_t)a>=state->header.player_count) return 0;
    const BoplRlPlayer *self=&state->players[a];
    double self_x=bopl_fix(self->x), self_y=bopl_fix(self->y);
    uint32_t nearest[BOPL_OBS_ENTITIES]; double distances[BOPL_OBS_ENTITIES];
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
            bopl_nearest(nearest, distances, &count, BOPL_OBS_ENTITIES, i+(uint32_t)j, distance);
        }
        i=(uint32_t)end;
    }
    int n=count<capacity?count:capacity;
    for (int i=0; i<n; i++) ids[i]=state->entities[nearest[i]].id;
    return count;
}

/* ---- privileged labels from a forked sim -------------------------------- */
enum { PROBE_CAUSE_ALIVE=0, PROBE_CAUSE_ENVIRONMENT=1, PROBE_CAUSE_SQUISHED=2,
       PROBE_CAUSE_OPPONENT=3, PROBE_CAUSE_SELF=4, PROBE_CAUSE_OTHER=5 };
/* Per player: death_tick (-1 alive through the horizon), cause, killer arrow
 * id (nearest live arrow one tick before death, 0 none), killer arrow x, y,
 * vx, vy at the fork start (for row matching), x/y at ticks 10 and 30 of the
 * idle future, plus "own arrow in flight at fork start". */
enum { PROBE_IDLE_FIELDS=13 };
static BoplRl *probe_fork_cache=NULL;
static char probe_error[512];
const char *probe_last_error(void) { return probe_error; }
/* One fork per binding process; environments are stepped one at a time per
 * Python thread, so callers must serialize (generate.py uses processes). */
static int probe_nearest_arrow(const BoplRlState *s, uint32_t self_id, double x, double y, int hostile,
                               double *ax, double *ay, double *avx, double *avy) {
    int best=-1; double best_d=1e18;
    for (uint32_t i=0; i<s->header.entity_count; i++) {
        const BoplRlEntity *e=&s->entities[i];
        if (!(e->flags&BOPL_ENTITY_ACTIVE) || (e->flags&BOPL_ENTITY_DESTROYED)) continue;
        if (!(e->flags&BOPL_ENTITY_PROJECTILE) && !(e->kind==0 && e->layer==9)) continue;
        if (hostile ? e->owner==self_id : e->owner!=self_id) continue;
        double dx=bopl_fix(e->x)-x, dy=bopl_fix(e->y)-y, d=dx*dx+dy*dy;
        if (d<best_d) { best_d=d; best=(int)i; }
    }
    if (best<0) return 0;
    const BoplRlEntity *e=&s->entities[best];
    if (ax) { *ax=bopl_fix(e->x); *ay=bopl_fix(e->y); *avx=bopl_fix(e->vx); *avy=bopl_fix(e->vy); }
    return (int)e->id;
}
int probe_idle_future(BoplPuffer *env, int horizon, float *out) {
    const BoplRlState *real=bopl_rl_state(env->world);
    if (!real || horizon<1 || horizon>600) return -1;
    int players=(int)real->header.player_count;
    BoplRl *f=bopl_rl_fork(env->world, probe_fork_cache);
    if (!f && probe_fork_cache) { bopl_rl_close(probe_fork_cache); probe_fork_cache=NULL; f=bopl_rl_fork(env->world, NULL); }
    if (!f) return -2;
    probe_fork_cache=f;
    BoplRlAction idle[6]; memset(idle, 0, sizeof(idle));
    BoplRlTelemetry before[6]; memcpy(before, real->telemetry, sizeof(before));
    for (int p=0; p<players; p++) {
        float *o=out+p*PROBE_IDLE_FIELDS;
        for (int k=0; k<PROBE_IDLE_FIELDS; k++) o[k]=0;
        o[0]=-1;
        double ax=0, ay=0, avx=0, avy=0;
        int own=probe_nearest_arrow(real, real->players[p].id, bopl_fix(real->players[p].x), bopl_fix(real->players[p].y), 0, &ax, &ay, &avx, &avy);
        o[12]=own?1.f:0.f;
    }
    int alive[6]; for (int p=0; p<players; p++) alive[p]=(real->players[p].flags&1)!=0;
    /* The fork's state buffer is overwritten by every step: keep a copy of
     * the previous tick for the killer lookup. */
    static BoplStateCopy prev_copy;
    bopl_state_copy(&prev_copy, real);
    const BoplRlState *prev=bopl_state_copy_get(&prev_copy);
    for (int t=1; t<=horizon; t++) {
        if (bopl_rl_step(f, idle, (size_t)players, 1)) { snprintf(probe_error, sizeof(probe_error), "fork step: %s", bopl_rl_error(f)); return -3; }
        const BoplRlState *s=bopl_rl_state(f);
        for (int p=0; p<players; p++) {
            float *o=out+p*PROBE_IDLE_FIELDS;
            if (t==10) { o[8]=(float)bopl_fix(s->players[p].x); o[9]=(float)bopl_fix(s->players[p].y); }
            if (t==30) { o[10]=(float)bopl_fix(s->players[p].x); o[11]=(float)bopl_fix(s->players[p].y); }
            if (!alive[p]) continue;
            if (!(s->players[p].flags&1)) {
                alive[p]=0; o[0]=(float)t;
                const BoplRlTelemetry *a=&s->telemetry[p], *b=&before[p];
                int cause=PROBE_CAUSE_OTHER;
                if (a->deaths_environment>b->deaths_environment) cause=s->players[p].cause_of_death==1?PROBE_CAUSE_SQUISHED:PROBE_CAUSE_ENVIRONMENT;
                else if (a->deaths_opponent>b->deaths_opponent) cause=PROBE_CAUSE_OPPONENT;
                else if (a->deaths_self>b->deaths_self) cause=PROBE_CAUSE_SELF;
                o[1]=(float)cause;
                if (cause==PROBE_CAUSE_OPPONENT || cause==PROBE_CAUSE_SELF) {
                    /* The arrow nearest to the victim one tick before the hit,
                     * hostile for an opponent kill, own for a self kill. */
                    double x=bopl_fix(prev->players[p].x), y=bopl_fix(prev->players[p].y), ax, ay, avx, avy;
                    int id=probe_nearest_arrow(prev, prev->players[p].id, x, y, cause==PROBE_CAUSE_OPPONENT, &ax, &ay, &avx, &avy);
                    o[2]=(float)id;
                    /* Its kinematics at the fork start (the label consumer matches rows there). */
                    for (uint32_t i=0; i<real->header.entity_count; i++) if (real->entities[i].id==(uint32_t)id) {
                        const BoplRlEntity *e=&real->entities[i];
                        o[3]=(float)bopl_fix(e->x); o[4]=(float)bopl_fix(e->y); o[5]=(float)bopl_fix(e->vx); o[6]=(float)bopl_fix(e->vy); o[7]=1;
                    }
                }
            }
        }
        if (s->header.flags&BOPL_RL_TERMINAL) {
            /* Positions after the end hold the terminal state. */
            for (int p=0; p<players; p++) {
                float *o=out+p*PROBE_IDLE_FIELDS;
                if (t<10) { o[8]=(float)bopl_fix(s->players[p].x); o[9]=(float)bopl_fix(s->players[p].y); }
                if (t<30) { o[10]=(float)bopl_fix(s->players[p].x); o[11]=(float)bopl_fix(s->players[p].y); }
            }
            break;
        }
        bopl_state_copy(&prev_copy, s);
        prev=bopl_state_copy_get(&prev_copy);
    }
    return players;
}

/* ---- analytic geometry labels (bopl_bots.h world view) ------------------ */
enum { PROBE_GEOM_FIELDS=20 };
/* [0] view ok, [1] grounded, [2] drop to surface below (cap 80), [3] support
 * wall present, [4] horizontal distance to the support wall's nearer edge,
 * [5] lands dry if falling with no steer, [6] lands dry with some steer,
 * [7] nearest terrain within one dash (13 u) in any of 16 directions,
 * [8] min terrain distance over the 16 rays, [9] my vx, [10] my vy,
 * [11] foe vx, [12] foe vy, [13] foe grounded, [14..16] foe dash/bow/platform
 * ready, [17..19] foe remaining cooldown seconds per slot. */
int probe_geometry(const BoplRlState *state, int player, float *out) {
    for (int i=0; i<PROBE_GEOM_FIELDS; i++) out[i]=0;
    BotView view;
    int ok=bot_view(&view, state, player);
    if (!view.alive || !view.my_slime) return 0;
    out[0]=1; out[1]=(float)view.grounded;
    out[2]=(float)bot_drop(&view, view.x, view.y);
    int support=bot_support_wall(&view, view.x, view.y);
    if (support>=0) {
        const BotWall *w=&view.walls[support];
        double px=view.x-w->cx, py=view.y-w->cy;
        double lx=px*w->rx+py*w->ry;
        out[3]=1; out[4]=(float)((w->ex+w->radius)-fabs(lx));
    }
    double lx, ly;
    out[5]=(float)bot_landing_steer(&view, 0, &lx, &ly);
    out[6]=(float)(out[5]>0 || bot_landing_steer(&view, -1, &lx, &ly) || bot_landing_steer(&view, 1, &lx, &ly));
    double best=1e9;
    for (int i=0; i<16; i++) {
        double a=i*(2*M_PI/16), d=bot_raycast(&view, view.x, view.y, cos(a), sin(a), 40);
        if (d<best) best=d;
    }
    out[7]=best<=BOT_DASH_LENGTH; out[8]=(float)best;
    out[9]=(float)view.vx; out[10]=(float)view.vy;
    if (ok && view.foe) {
        out[11]=(float)view.fvx; out[12]=(float)view.fvy; out[13]=(float)view.foe_grounded;
        const BoplRlSlime *foe=view.foe_slime;
        if (foe) for (int slot=0; slot<3; slot++) {
            double cd=bopl_fix(foe->cooldowns[slot]), timer=bopl_fix(foe->timers[slot]);
            out[14+slot]=timer>cd; out[17+slot]=(float)(cd-timer>0?cd-timer:0);
        }
    }
    return 1;
}

/* ---- raycasts ------------------------------------------------------------ */
/* Hit types, one-hot in the ray record. */
enum { RAY_WALL=0, RAY_KILL=1, RAY_HOSTILE_PROJECTILE=2, RAY_OPPONENT=3, RAY_OWN=4, RAY_NONE=5, RAY_TYPES=6 };
enum { RAY_FIELDS=1+RAY_TYPES+1 };   /* distance/range, type one-hot, hit velocity along the ray / 64 */
typedef struct { double t; int type; double v_along; } RayHit;
/* Ray (ox,oy)+t(dx,dy), |d|=1, vs circle: smallest t>=0 of entry (0 if inside). */
static int ray_circle(double ox, double oy, double dx, double dy, double cx, double cy, double r, double *t) {
    double fx=ox-cx, fy=oy-cy, b=fx*dx+fy*dy, c=fx*fx+fy*fy-r*r;
    if (c<=0) { *t=0; return 1; }
    double disc=b*b-c;
    if (disc<0) return 0;
    double s=-b-sqrt(disc);
    if (s<0) return 0;
    *t=s; return 1;
}
/* Ray vs oriented box (unit axes (rx,ry),(ux,uy), half extents ex,ey). */
static int ray_box(double ox, double oy, double dx, double dy, double cx, double cy,
                   double rx, double ry, double ux, double uy, double ex, double ey, double *t) {
    double px=ox-cx, py=oy-cy;
    double lx=px*rx+py*ry, ly=px*ux+py*uy, ldx=dx*rx+dy*ry, ldy=dx*ux+dy*uy;
    double tmin=0, tmax=1e18;
    double l[2]={lx, ly}, ld[2]={ldx, ldy}, e[2]={ex, ey};
    for (int i=0; i<2; i++) {
        if (fabs(ld[i])<1e-12) { if (fabs(l[i])>e[i]) return 0; continue; }
        double t1=(-e[i]-l[i])/ld[i], t2=(e[i]-l[i])/ld[i];
        if (t1>t2) { double s=t1; t1=t2; t2=s; }
        if (t1>tmin) tmin=t1;
        if (t2<tmax) tmax=t2;
        if (tmin>tmax) return 0;
    }
    if (tmax<0) return 0;
    *t=tmin; return 1;
}
/* Rounded rect = union of two boxes (inflated along each axis) and four corner circles: exact. */
static int ray_rounded(double ox, double oy, double dx, double dy, double cx, double cy,
                       double rx, double ry, double ux, double uy, double ex, double ey, double r, double *t) {
    double best=1e18, s; int hit=0;
    if (ray_box(ox, oy, dx, dy, cx, cy, rx, ry, ux, uy, ex+r, ey, &s) && s<best) { best=s; hit=1; }
    if (ray_box(ox, oy, dx, dy, cx, cy, rx, ry, ux, uy, ex, ey+r, &s) && s<best) { best=s; hit=1; }
    if (r>0) for (int sx=-1; sx<=1; sx+=2) for (int sy=-1; sy<=1; sy+=2) {
        double kx=cx+sx*ex*rx+sy*ey*ux, ky=cy+sx*ex*ry+sy*ey*uy;
        if (ray_circle(ox, oy, dx, dy, kx, ky, r, &s) && s<best) { best=s; hit=1; }
    }
    if (hit) *t=best;
    return hit;
}
/* A shape the rays can hit, prepared once per observation (unit axes, half
 * extents, a bounding radius for the broad phase). kind 0 circle, 1 box,
 * 2 rounded rect, 3/4 capsule (segment + radius) mapped onto a rounded rect. */
typedef struct { int kind, type; double cx, cy, rx, ry, ux, uy, ex, ey, r, bound, vx, vy; } RayShape;
static void ray_shape_entity(RayShape *s, const BoplRlEntity *e, int type, double svx, double svy) {
    double cx=bopl_fix(e->x), cy=bopl_fix(e->y), r=bopl_fix(e->radius);
    double rx=bopl_fix(e->right_x), ry=bopl_fix(e->right_y), ux=bopl_fix(e->up_x), uy=bopl_fix(e->up_y);
    double ex=bot_len(rx, ry), ey=bot_len(ux, uy);
    if (ex>1e-9) { rx/=ex; ry/=ex; } else { rx=1; ry=0; }
    if (ey>1e-9) { ux/=ey; uy/=ey; } else { ux=-ry; uy=rx; }
    s->kind=e->kind==1?1:e->kind==0?0:2; s->type=type;
    s->cx=cx; s->cy=cy; s->rx=rx; s->ry=ry; s->ux=ux; s->uy=uy; s->ex=ex; s->ey=ey; s->r=r;
    if (e->kind==3 || e->kind==4) {
        /* Segment from the entity to entity+right, thickened by the radius. */
        s->cx=cx+0.5*bopl_fix(e->right_x); s->cy=cy+0.5*bopl_fix(e->right_y);
        s->ux=-ry; s->uy=rx; s->ex=0.5*ex; s->ey=0;
    }
    if (s->kind==0) { s->ex=0; s->ey=0; }
    s->bound=bot_len(s->ex, s->ey)+s->r;
    s->vx=bopl_fix(e->vx)-svx; s->vy=bopl_fix(e->vy)-svy;
}
static int ray_shape(const RayShape *s, double ox, double oy, double dx, double dy, double *t) {
    /* Broad phase: the ray misses the bounding circle, or the shape is entirely behind the origin. */
    double fx=s->cx-ox, fy=s->cy-oy, along=fx*dx+fy*dy;
    if (along<-s->bound) return 0;
    double perp=fx*dy-fy*dx;
    if (fabs(perp)>s->bound) return 0;
    switch (s->kind) {
    case 0: return ray_circle(ox, oy, dx, dy, s->cx, s->cy, s->r, t);
    case 1: return ray_box(ox, oy, dx, dy, s->cx, s->cy, s->rx, s->ry, s->ux, s->uy, s->ex, s->ey, t);
    default: return ray_rounded(ox, oy, dx, dy, s->cx, s->cy, s->rx, s->ry, s->ux, s->uy, s->ex, s->ey, s->r, t);
    }
}
static void ray_insert(RayHit *hits, int *n, int capacity, double t, int type, double v_along) {
    int pos=*n;
    if (pos==capacity && t>=hits[pos-1].t) return;
    if (pos<capacity) (*n)++; else pos--;
    while (pos>0 && t<hits[pos-1].t) { hits[pos]=hits[pos-1]; pos--; }
    hits[pos].t=t; hits[pos].type=type; hits[pos].v_along=v_along;
}
/* Ray angles: n uniform from +x counter-clockwise, then `extra_down` rays
 * fanned about straight down (the fall/edge questions). */
static void ray_direction(int i, int n, int extra_down, double *dx, double *dy) {
    double a;
    if (i<n) a=i*(2*M_PI/n);
    else {
        /* Cell centres of a 120-degree fan about straight down (4 extra rays:
         * -135, -105, -75, -45 degrees); the uniform fan already holds -90. */
        int k=i-n; double span=2*M_PI/3;
        a=-M_PI/2-span/2+span*(k+0.5)/extra_down;
    }
    *dx=cos(a); *dy=sin(a);
}
/* `out`: (n+extra_down) x hits_per_ray x RAY_FIELDS floats. Origin: the
 * player's controlling body (slime or morph). Returns the ray count. */
int probe_rays(const BoplRlState *state, int player, int n, int extra_down, int hits_per_ray, float range, float *out) {
    int rays=n+extra_down;
    if (!state || player<0 || (uint32_t)player>=state->header.player_count || n<1 || hits_per_ray<1 || hits_per_ray>4 || range<=0) return -1;
    const BoplRlPlayer *self=&state->players[player];
    double ox=bopl_fix(self->x), oy=bopl_fix(self->y);
    double svx=0, svy=0;
    const BoplRlSlime *me=bot_slime(state, self->id);
    if (me && (me->flags&1)) { ox=bopl_fix(me->x); oy=bopl_fix(me->y); svx=bopl_fix(me->vx); svy=bopl_fix(me->vy); }
    else { const BoplRlEntity *body=bot_morph_entity(state, self->id); if (body) { ox=bopl_fix(body->x); oy=bopl_fix(body->y); } }
    /* Shapes once per observation: terrain, kill volumes, projectiles, opponent bodies. */
    enum { RAY_MAX_SHAPES=160 };
    RayShape shapes[RAY_MAX_SHAPES]; int shape_count=0;
    for (uint32_t k=0; k<state->header.entity_count && shape_count<RAY_MAX_SHAPES; k++) {
        const BoplRlEntity *e=&state->entities[k];
        if (!(e->flags&BOPL_ENTITY_ACTIVE) || (e->flags&BOPL_ENTITY_DESTROYED) || !(e->flags&BOPL_ENTITY_ENABLED)) continue;
        int type;
        if (e->kind==2 && (e->layer==11 || e->layer==21)) type=RAY_WALL;
        else if (e->layer==4 || (e->kind==2 && e->layer==16)) type=RAY_KILL;
        else if ((e->flags&BOPL_ENTITY_PROJECTILE) || (e->kind==0 && e->layer==9)) type=e->owner==self->id?RAY_OWN:RAY_HOSTILE_PROJECTILE;
        else continue;  /* player hurtboxes come from the slime rows below */
        ray_shape_entity(&shapes[shape_count++], e, type, svx, svy);
    }
    for (uint32_t k=0; k<state->header.slime_count && shape_count<RAY_MAX_SHAPES; k++) {
        const BoplRlSlime *s=&state->slimes[k];
        if (!(s->flags&BOPL_SLIME_ACTIVE) || s->owner==self->id) continue;
        if (bopl_spatial_hidden(state, s->owner, self->id)) continue;
        double r=BOT_SLIME_RADIUS*bopl_fix(s->scale);
        RayShape *sh=&shapes[shape_count++];
        memset(sh, 0, sizeof(*sh));
        sh->kind=0; sh->type=RAY_OPPONENT; sh->cx=bopl_fix(s->x); sh->cy=bopl_fix(s->y); sh->r=r>0?r:BOT_SLIME_RADIUS; sh->bound=sh->r;
        sh->vx=bopl_fix(s->vx)-svx; sh->vy=bopl_fix(s->vy)-svy;
    }
    for (int i=0; i<rays; i++) {
        double dx, dy; ray_direction(i, n, extra_down, &dx, &dy);
        RayHit hits[4]; int count=0; double t;
        for (int k=0; k<shape_count; k++) {
            const RayShape *s=&shapes[k];
            if (!ray_shape(s, ox, oy, dx, dy, &t) || t>range) continue;
            ray_insert(hits, &count, hits_per_ray, t, s->type, s->vx*dx+s->vy*dy);
        }
        /* Blast planes and the water line: implicit kill zones. */
        if (dy<0) { t=(BOT_WATER-oy)/dy; if (t>=0 && t<=range) ray_insert(hits, &count, hits_per_ray, t, RAY_KILL, 0); }
        if (dy>0) { t=(BOT_BLAST_Y-oy)/dy; if (t>=0 && t<=range) ray_insert(hits, &count, hits_per_ray, t, RAY_KILL, 0); }
        if (dx>0) { t=(BOT_BLAST_X-ox)/dx; if (t>=0 && t<=range) ray_insert(hits, &count, hits_per_ray, t, RAY_KILL, 0); }
        if (dx<0) { t=(-BOT_BLAST_X-ox)/dx; if (t>=0 && t<=range) ray_insert(hits, &count, hits_per_ray, t, RAY_KILL, 0); }
        for (int h=0; h<hits_per_ray; h++) {
            float *o=out+((size_t)i*hits_per_ray+h)*RAY_FIELDS;
            for (int f=0; f<RAY_FIELDS; f++) o[f]=0;
            if (h<count) { o[0]=(float)(hits[h].t/range); o[1+hits[h].type]=1; o[1+RAY_TYPES]=bopl_clip(hits[h].v_along/64.0); }
            else { o[0]=1; o[1+RAY_NONE]=1; }
        }
    }
    return rays;
}
/* Nanoseconds per probe_rays call, averaged over `iterations`. */
double probe_rays_bench(const BoplRlState *state, int player, int n, int extra_down, int hits_per_ray, float range, int iterations) {
    float *out=malloc((size_t)(n+extra_down)*hits_per_ray*RAY_FIELDS*sizeof(float));
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (int i=0; i<iterations; i++) probe_rays(state, player, n, extra_down, hits_per_ray, range, out);
    clock_gettime(CLOCK_MONOTONIC, &b);
    free(out);
    return ((b.tv_sec-a.tv_sec)*1e9+(b.tv_nsec-a.tv_nsec))/iterations;
}
/* Nanoseconds per undelayed observation encode (both players), for the cost comparison. */
double probe_pack_bench(BoplPuffer *env, int iterations) {
    float *out=malloc((size_t)env->players*BOPL_OBS_SIZE*sizeof(float));
    const BoplRlState *state=bopl_rl_state(env->world);
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (int i=0; i<iterations; i++) probe_pack(env, state, out);
    clock_gettime(CLOCK_MONOTONIC, &b);
    free(out);
    return ((b.tv_sec-a.tv_sec)*1e9+(b.tv_nsec-a.tv_nsec))/iterations;
}
int probe_idle_fields(void) { return PROBE_IDLE_FIELDS; }
int probe_geometry_fields(void) { return PROBE_GEOM_FIELDS; }
int probe_ray_fields(void) { return RAY_FIELDS; }
