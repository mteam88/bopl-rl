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
        double best=r-radius, best_t=0.0, best_squared=dx*dx+dy*dy;
        for (int k=1; k<=30; k++) {
            double t=k/15.0, px=dx+dvx*t, py=dy+dvy*t-0.5*BOPL4_GRAVITY*t*t, squared=px*px+py*py;
            /* sqrt and -radius are monotone, so a sample can only beat `best`
             * when its square is smaller: skip the sqrt otherwise. */
            if (squared>=best_squared) continue;
            double d=sqrt(squared)-radius;
            if (d<best) { best=d; best_t=t; best_squared=squared; }
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
#include "bopl_rays_v4.h"
/* The rays' scene for `self`'s row (bopl_rays_v4.h), computed in double and
 * stored as float: the walls are the rounded rects on the wall and terrain
 * layers (11 wall, 16 lethal, 21 non-lethal), water the highest water
 * entity's top, the circles the observed dynamic bodies (the nearest visible
 * entities with mass, not ropes or platforms) and the visible enemy slimes. */
static void bopl4_scene_fill(Bopl4Scene *s, const BoplRlState *state, const uint32_t *nearest, int count,
                             const uint32_t *nearby_slimes, int slime_count, const BoplRlPlayer *self, int ablation,
                             double x, double y, double svx, double svy, BoplRlAction applied, const Bopl4Enemy *enemy) {
    s->flags=(ablation&BOPL4_ABLATE_RAYS?0:BOPL4F_FAN)|(ablation&BOPL4_ABLATE_RAYS16?BOPL4F_HALF:0)|(ablation&BOPL4_ABLATE_AIM?BOPL4F_AIM:0);
    s->x=(float)x; s->y=(float)y; s->svx=(float)svx; s->svy=(float)svy;
    s->aim_dx=s->aim_dy=0.0f;
    if (applied.angle) {
        pthread_once(&bopl_aim_table_once, bopl_aim_table_init);   /* the first step's rows precede the tail that fills it */
        s->flags|=BOPL4F_AIMED; s->aim_dx=bopl_aim_table[applied.angle][0]; s->aim_dy=bopl_aim_table[applied.angle][1];
    }
    s->enemy_x=s->enemy_y=0.0f;
    if (enemy->known) { s->flags|=BOPL4F_ENEMY; s->enemy_x=(float)enemy->x; s->enemy_y=(float)enemy->y; }
    s->axis_count=s->rot_count=s->circle_count=0; s->axis_lethal=s->rot_lethal=s->rot_skew=0;
    double water_top=-1e9; int has_water=0;
    for (uint32_t i=0; i<state->header.entity_count; i++) {
        const BoplRlEntity *e=&state->entities[i];
        if (!(e->flags&1) || (e->flags&4)) continue;
        if (e->layer==BOPL4_WATER_LAYER) {
            double top=bopl_fix(e->y)+fabs(bopl_fix(e->up_y))+fabs(bopl_fix(e->right_y));
            if (!has_water || top>water_top) water_top=top;
            has_water=1;
            continue;
        }
        if (e->kind!=2 || !(e->layer==11 || e->layer==16 || e->layer==21) || s->axis_count+s->rot_count>=BOPL4F_WALLS) continue;
        double rx=bopl_fix(e->right_x), ry=bopl_fix(e->right_y), ux=bopl_fix(e->up_x), uy=bopl_fix(e->up_y);
        double ex=sqrt(rx*rx+ry*ry), ey=sqrt(ux*ux+uy*uy);
        if (ex>1e-6) { rx/=ex; ry/=ex; } else { rx=1; ry=0; }
        if (ey>1e-6) { ux/=ey; uy/=ey; } else { ux=-ry; uy=rx; }
        uint64_t lethal=e->layer==16;
        int straight=(rx==1||rx==-1) && ry==0 && ux==0 && (uy==1||uy==-1);
        int turned=rx==0 && (ry==1||ry==-1) && (ux==1||ux==-1) && uy==0;
        if (straight||turned) {
            int k=s->axis_count++;
            s->axis_cx[k]=(float)bopl_fix(e->x); s->axis_cy[k]=(float)bopl_fix(e->y); s->axis_radius[k]=(float)bopl_fix(e->radius);
            s->axis_ex[k]=(float)(straight?ex:ey); s->axis_ey[k]=(float)(straight?ey:ex);
            s->axis_lethal|=lethal<<k;
        } else {
            int k=s->rot_count++;
            s->rot_cx[k]=(float)bopl_fix(e->x); s->rot_cy[k]=(float)bopl_fix(e->y); s->rot_radius[k]=(float)bopl_fix(e->radius);
            s->rot_rx[k]=(float)rx; s->rot_ry[k]=(float)ry; s->rot_ux[k]=(float)ux; s->rot_uy[k]=(float)uy;
            s->rot_ex[k]=(float)ex; s->rot_ey[k]=(float)ey;
            s->rot_lethal|=lethal<<k;
            s->rot_skew|=(uint64_t)(fabs(rx*ux+ry*uy)>1e-9)<<k;
        }
    }
    if (has_water) { s->flags|=BOPL4F_WATER; s->water_top=(float)water_top; } else s->water_top=-1e9f;
    for (int i=0; i<count; i++) {
        const BoplRlEntity *e=&state->entities[nearest[i]];
        if (e->inverse_mass<=0 || e->kind==3 || e->kind==4) continue;
        int k=s->circle_count++;
        s->cx[k]=(float)bopl_fix(e->x); s->cy[k]=(float)bopl_fix(e->y); s->cr[k]=(float)fmax(bopl_fix(e->radius), 0.25);
        s->cvx[k]=(float)bopl_fix(e->vx); s->cvy[k]=(float)bopl_fix(e->vy);
        s->cclass[k]=e->owner==self->id?2:0;
    }
    for (int i=0; i<slime_count; i++) {
        const BoplRlSlime *slime=&state->slimes[nearby_slimes[i]];
        if (!(slime->flags&BOPL_SLIME_ACTIVE)) continue;
        const BoplRlPlayer *owner=bopl_player_by_id(state, slime->owner);
        if (!owner || owner->id==self->id || owner->team==self->team) continue;
        int k=s->circle_count++;
        double r=BOT_SLIME_RADIUS*bopl_fix(slime->scale);
        s->cx[k]=(float)bopl_fix(slime->x); s->cy[k]=(float)bopl_fix(slime->y); s->cr[k]=(float)(r>0?r:BOT_SLIME_RADIUS);
        s->cvx[k]=(float)bopl_fix(slime->vx); s->cvy[k]=(float)bopl_fix(slime->vy);
        s->cclass[k]=1;
    }
}
_Static_assert(BOPL4F_BLAST_X==BOT_BLAST_X && BOPL4F_BLAST_Y==BOT_BLAST_Y, "the rays' blast planes are the bots'");
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
    /* Written here, or by the GPU from the scene (bopl_pack_rows). */
    {
        Bopl4Scene local, *scene=bopl4_defer?bopl4_pending_scene(obs):NULL;
        bopl4_scene_fill(scene?scene:&local, state, nearest, count, nearby_slimes, slime_count, self, bopl4_ablation(),
                         self_x, self_y, self_vx, self_vy, env->applied[a], &enemy);
        if (!scene) bopl4f_write(&local, bopl4_dirs(), obs+BOPL4_OBS_TERRAIN_OFFSET);
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
