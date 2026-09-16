/* Observation writer for layout v4 (bopl_layout_v4.h, POLICY_V4_PLAN.md).
 * Included by bopl_puffer.h after the v3 writer's helpers; shares its
 * visibility, delay and truncation rules (bopl_spatial_hidden,
 * bopl_entity_visible, bopl_nearest, rope thinning). Nothing here is read by
 * the v3 path. */
#ifndef BOPL_OBS_V4_H
#define BOPL_OBS_V4_H

/* Writer ablations (POLICY_V4_PLAN.md section 7), a bitmask in BOPL4_ABLATE
 * read once per process (train.py --v4-ablate, recorded in the checkpoint's
 * v4_shape so the evaluator applies the same one). The row keeps its shape;
 * the ablated block is zero. */
enum { BOPL4_ABLATE_RELATIONAL=1, BOPL4_ABLATE_RAYS=2, BOPL4_ABLATE_RAYS16=4 };
static int bopl4_ablation(void) {
    static int value=-1;
    if (value<0) { const char *v=getenv("BOPL4_ABLATE"); value=v&&v[0]?atoi(v):0; }
    return value;
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
} Bopl4Terrain;
static void bopl4_terrain_fill(Bopl4Terrain *t, const BoplRlState *state, const uint32_t *nearest, int count,
                               const uint32_t *nearby_slimes, int slime_count, const BoplRlPlayer *self) {
    BotView *view=&t->view;
    view->wall_count=0; t->has_water=0; t->water_top=-1e9; t->circle_count=0;
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
    int best=-1; double best_d=1e9;
    for (int i=0; i<view->wall_count; i++) { double d=bot_wall_sdf(&view->walls[i], x, y); if (d<best_d) { best_d=d; best=i; } }
    return best;
}
/* One ray: distance to the first surface among walls, water and circles
 * (BOPL4_RAY_LENGTH when nothing), the class and, for a circle, its closing
 * speed along the ray (relative to the observer's velocity). */
/* Ray k's direction: BOPL4_RAYS_AROUND uniform from +x, then the fan about
 * straight down (cell centres of a 120-degree span: -135, -105, -75, -45). */
static void bopl4_ray_direction(int k, double *dx, double *dy) {
    double a;
    if (k<BOPL4_RAYS_AROUND) a=6.28318530717958647692*k/BOPL4_RAYS_AROUND;
    else { double span=6.28318530717958647692/3.0; a=-1.57079632679489661923-span/2+span*(k-BOPL4_RAYS_AROUND+0.5)/BOPL4_RAYS_DOWN; }
    *dx=cos(a); *dy=sin(a);
}
/* Classes: 0 wall, 1 lethal wall, 2 water / blast plane, 3 hostile projectile
 * or object, 4 opponent body, 5 own object; out[7] the hit's closing speed
 * along the ray for classes 3-5. */
static void bopl4_ray(const Bopl4Terrain *t, double x, double y, double dx, double dy, double svx, double svy, float *out) {
    double best=BOPL4_RAY_LENGTH; int class=-1; double closing=0;
    if (t->view.wall_count) {
        double d=bot_raycast(&t->view, x, y, dx, dy, BOPL4_RAY_LENGTH);
        if (d<best) {
            best=d;
            int w=bopl4_nearest_wall(&t->view, x+dx*d, y+dy*d);
            class=(w>=0 && t->view.walls[w].standable<0)?1:0;
        }
    }
    if (t->has_water && dy<-1e-9) {
        double d=(t->water_top-y)/dy;
        if (d>=0 && d<best) { best=d; class=2; }
    }
    /* The blast planes (bopl_bots.h): implicit kill zones like the water. */
    if (dy>1e-9) { double d=(BOT_BLAST_Y-y)/dy; if (d>=0 && d<best) { best=d; class=2; } }
    if (dx>1e-9) { double d=(BOT_BLAST_X-x)/dx; if (d>=0 && d<best) { best=d; class=2; } }
    if (dx<-1e-9) { double d=(-BOT_BLAST_X-x)/dx; if (d>=0 && d<best) { best=d; class=2; } }
    for (int i=0; i<t->circle_count; i++) {
        double px=t->cx[i]-x, py=t->cy[i]-y, along=px*dx+py*dy;
        if (along<0) continue;
        double perp2=px*px+py*py-along*along, r2=t->cr[i]*t->cr[i];
        if (perp2>r2) continue;
        double d=along-sqrt(r2-perp2);
        if (d<0) d=0;
        if (d<best) { best=d; class=3+t->cclass[i]; closing=-((t->cvx[i]-svx)*dx+(t->cvy[i]-svy)*dy); }
    }
    out[0]=(float)(best/BOPL4_RAY_LENGTH);
    for (int c=0; c<6; c++) out[1+c]=class==c;
    out[7]=class>=3?bopl_clip(closing/BOPL4_VELOCITY_SCALE):0.0f;
}
static double bopl4_drop(const Bopl4Terrain *t, double x, double y, double cap) {
    double d=t->view.wall_count?bot_raycast(&t->view, x, y, 0, -1, cap):cap;
    if (t->has_water && y>t->water_top && y-t->water_top<d) d=y-t->water_top;
    return d;
}
/* The egocentric terrain block (BOPL4_TERRAIN) at body position (x, y). */
static void bopl4_terrain(const Bopl4Terrain *t, double x, double y, float *out) {
    const BotView *view=&t->view;
    double drop=bopl4_drop(t, x, y, 32.0);
    double ceiling=view->wall_count?bot_raycast(view, x, y, 0, 1, 32.0):32.0;
    out[0]=(float)(drop/32.0); out[1]=(float)(ceiling/32.0);
    /* Edges: walk left/right in 1.5-unit steps to one dash length; the edge
     * is where the floor drops away by more than 4 units (or ends in water),
     * a wall is where the step lands inside terrain. */
    for (int side=0; side<2; side++) {
        double sign=side?1.0:-1.0, edge=BOPL4_DASH_LENGTH; int wall=0;
        for (int k=1; k<=8; k++) {
            double px=x+sign*1.5*k;
            if (view->wall_count && bot_sdf(view, px, y)<0.3) { edge=1.5*k; wall=1; break; }
            if (bopl4_drop(t, px, y, 32.0)>drop+4.0) { edge=1.5*k; break; }
        }
        out[2+side]=(float)(edge/BOPL4_DASH_LENGTH); out[4+side]=wall;
    }
    out[6]=t->has_water?bopl_clip((y-t->water_top)/32.0):1.0f;
    double lethal=16.0;
    for (int i=0; i<view->wall_count; i++) if (view->walls[i].standable<0) {
        double d=bot_wall_sdf(&view->walls[i], x, y); if (d<lethal) lethal=d;
    }
    out[7]=(float)(lethal/16.0);
    /* Ground within one dash: some dash endpoint (16 directions, stopped at
     * terrain) has a surface within 4 units below it. */
    int landing=drop<=4.0;
    for (int k=0; k<16 && !landing; k++) {
        double angle=6.28318530717958647692*k/16.0, dx=cos(angle), dy=sin(angle);
        double d=view->wall_count?bot_raycast(view, x, y, dx, dy, BOPL4_DASH_LENGTH):BOPL4_DASH_LENGTH;
        double ex=x+dx*(d-0.5), ey=y+dy*(d-0.5);
        if (t->has_water && ey<=t->water_top) continue;
        landing=bopl4_drop(t, ex, ey, 8.0)<=4.0;
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
    obs[8]=0; obs[9]=0;   /* v3's absolute self position: a map lookup key, dropped (terrain block instead) */
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
    if (count_overflow) bopl_note_overflow(env, active_count, count, visible_slimes, visible_slots);

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
    } else {
        Bopl4Terrain terrain;
        int rays16=bopl4_ablation()&BOPL4_ABLATE_RAYS16;   /* every other of the 32, none of the down fan */
        bopl4_terrain_fill(&terrain, state, nearest, count, nearby_slimes, slime_count, self);
        bopl4_terrain(&terrain, self_x, self_y, obs+BOPL4_OBS_TERRAIN_OFFSET);
        for (int k=0; k<BOPL4_RAYS; k++) {
            float *ray=obs+BOPL4_OBS_RAY_OFFSET+(size_t)k*BOPL4_RAY_FEATURES;
            if (rays16 && (k>=BOPL4_RAYS_AROUND || (k&1))) { memset(ray, 0, BOPL4_RAY_FEATURES*sizeof(float)); continue; }
            double dx, dy; bopl4_ray_direction(k, &dx, &dy);
            bopl4_ray(&terrain, self_x, self_y, dx, dy, self_vx, self_vy, ray);
        }
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
