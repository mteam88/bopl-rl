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

/* sinf and cosf of a slime's rotation (zero for none): the body block's and
 * slime row's columns, which the GPU writer takes from the host. */
static void bopl4_sincos(const BoplRlSlime *slime, float *sin_r, float *cos_r) {
    if (!slime) { *sin_r=*cos_r=0.0f; return; }
    *sin_r=sinf((float)bopl_fix(slime->rotation)); *cos_r=cosf((float)bopl_fix(slime->rotation));
}
/* The rays' scene for `self`'s row (bopl_rays_v4.h), computed in double and
 * stored as float: the walls are the rounded rects on the wall and terrain
 * layers (11 wall, 16 lethal, 21 non-lethal), water the highest water
 * entity's top, the circles the observed dynamic bodies (the nearest visible
 * entities with mass, not ropes or platforms) and the visible enemy slimes. */
static void bopl4_scene_fill(Bopl4Scene *s, const BoplRlState *state, const uint32_t *nearest, int count,
                             const uint32_t *nearby_slimes, int slime_count, const BoplRlPlayer *self, int ablation,
                             double x, double y, double svx, double svy, BoplRlAction applied, const Bopl4Enemy *enemy) {
    float aim_dx=0.0f, aim_dy=0.0f;
    if (applied.angle) {
        pthread_once(&bopl_aim_table_once, bopl_aim_table_init);   /* the first step's rows precede the tail that fills it */
        aim_dx=bopl_aim_table[applied.angle][0]; aim_dy=bopl_aim_table[applied.angle][1];
    }
    bopl4_scene_head(s, ablation, x, y, svx, svy, applied.angle!=0, aim_dx, aim_dy, enemy);
    double water_top=-1e9; int has_water=0;
    for (uint32_t i=0; i<state->header.entity_count; i++) {
        float w[9]; double top=0.0; int lethal=0, skew=0;
        int kind=bopl4_scene_entity(&state->entities[i], w, &top, &lethal, &skew);
        if (kind==BOPL4_SCENE_WATER) {
            if (!has_water || top>water_top) water_top=top;
            has_water=1;
            continue;
        }
        if (kind==BOPL4_SCENE_NONE || s->axis_count+s->rot_count>=BOPL4F_WALLS) continue;
        if (kind==BOPL4_SCENE_AXIS) bopl4_scene_axis(s, s->axis_count++, w, lethal);
        else bopl4_scene_rot(s, s->rot_count++, w, lethal, skew);
    }
    if (has_water) { s->flags|=BOPL4F_WATER; s->water_top=(float)water_top; } else s->water_top=-1e9f;
    for (int i=0; i<count; i++) {
        float c[5]; int32_t cls;
        if (bopl4_entity_circle(&state->entities[nearest[i]], self, c, &cls)) bopl4_scene_circle(s, s->circle_count++, c, cls);
    }
    for (int i=0; i<slime_count; i++) {
        float c[5]; int32_t cls;
        if (bopl4_slime_circle(state, &state->slimes[nearby_slimes[i]], self, c, &cls)) bopl4_scene_circle(s, s->circle_count++, c, cls);
    }
}
_Static_assert(BOPL4F_BLAST_X==BOT_BLAST_X && BOPL4F_BLAST_Y==BOT_BLAST_Y, "the rays' blast planes are the bots'");
_Static_assert(BOPL4_SLIME_RADIUS==BOT_SLIME_RADIUS, "the scene's slime radius is the bots'");
/* Encode `state` into player a's v4 row. Same contract as the v3 row writer:
 * every feature stored, unused set slots zeroed (with the bopl_written record
 * of the previous encode), the own-action tail zeroed and the privileged block
 * left for bopl4_privileged. */
static void bopl_encode_v4_row(BoplPuffer *env, const BoplRlState *state, int a, int count_overflow) {
    bopl4_row_ablation=bopl_ablate_of(env, a);   /* this player's writer family for the whole row */
    int ablation=bopl4_ablation();
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
    Bopl4Counts c; memset(&c, 0, sizeof(c));
    for (uint32_t i=0; i<state->header.entity_count;) {
        int end=(state->entities[i].kind==3)?bopl_rope_group_end(state, i):(int)i+1;
        int group=end-(int)i;
        for (int j=0; j<group; j++) {
            const BoplRlEntity *entity=&state->entities[i+j];
            if (entity->kind==3 && !bopl_rope_span_kept(j, group)) continue;
            if (!bopl_entity_visible(state, entity, self->id)) continue;
            double distance=bopl_surface_distance(entity, self_x, self_y);
            if (entity->flags&BOPL_ENTITY_PICKUP) distance-=1.0e12;
            bopl_nearest(nearest, distances, &c.count, BOPL4_ENTITIES, i+(uint32_t)j, distance);
            c.active_count++;
            uint32_t source=entity->reserved[0];
            if (source>0 && source<=BOPL_ABILITY_COUNT) c.source_counts[source-1]++;
            if (entity->inverse_mass>0) {
                c.n_nonstatic++;
                double dx=bopl_fix(entity->x)-self_x, dy=bopl_fix(entity->y)-self_y;
                c.sectors[bopl4_sector(dx, dy)]++;
            }
            if (entity->flags&BOPL_ENTITY_PICKUP) c.n_pickup++;
            if (!c.time_stop_known && state->header.player_time_scale==0
                && (entity->flags&BOPL_ENTITY_TIMER_KNOWN) && entity->clock_rate>0) {
                c.time_stop_remaining=entity->reserved[1]; c.time_stop_known=1;
            }
        }
        i=(uint32_t)end;
    }
    int count=c.count;

    uint32_t nearby_slimes[BOPL4_SLIMES], slot_indices[BOPL4_SLOTS];
    int slime_count, slot_count, visible_slots;
    bopl4_slime_pass(state, self, self_x, self_y, controller, enemy_controller,
                     nearby_slimes, &slime_count, slot_indices, &slot_count, &c.visible_slimes, &visible_slots);

    /* Global block: the v3 prefix. */
    bopl4_global_block(obs, state, &c, env->config.max_episode_ticks, bopl_map_selected(BOPL_SPACE_MAPS, env->config.map), controller, 0, 1);
    if (count_overflow) {
        bopl_note_overflow(env, c.active_count, count, c.visible_slimes, visible_slots);
        bopl_note_entity_rows(env,state,nearest,count,c.source_counts);
    }

    /* Self: the v3 player row about oneself, then the controlling body. */
    float sin_r, cos_r;
    float *self_block=obs+BOPL4_OBS_SELF_OFFSET;
    bopl_player_features(self_block, state, self, self, self_x, self_y);
    bopl4_sincos(controller, &sin_r, &cos_r);
    bopl4_body(self_block+BOPL_PLAYER_FEATURES, controller, sin_r, cos_r);
    /* Opponent: the primary enemy's player row and body, its relation to us. */
    bopl4_sincos(enemy.slime, &sin_r, &cos_r);
    bopl4_opponent_block(obs+BOPL4_OBS_OPPONENT_OFFSET, state, &enemy, self, self_x, self_y, self_vx, self_vy, sin_r, cos_r, ablation);
    memset(obs+BOPL4_OBS_PREVIOUS_OFFSET, 0, BOPL4_PREVIOUS*sizeof(float));
    /* Terrain and rays: the egocentric view of the map from the body. */
    /* Written here, or by the GPU from the scene (bopl_pack_rows). */
    {
        Bopl4Scene local, *scene=bopl4_defer?bopl4_pending_scene(obs):NULL;
        bopl4_scene_fill(scene?scene:&local, state, nearest, count, nearby_slimes, slime_count, self, ablation,
                         self_x, self_y, self_vx, self_vy, env->applied[a], &enemy);
        if (!scene) bopl4f_write(&local, bopl4_dirs(), obs+BOPL4_OBS_TERRAIN_OFFSET);
    }

    uint32_t player_count=state->header.player_count<BOPL4_PLAYERS?state->header.player_count:BOPL4_PLAYERS;
    for (uint32_t i=0; i<player_count; i++) {
        uint32_t index=i==0?(uint32_t)a:(i<=(uint32_t)a?i-1:i);
        bopl4_player_row(obs+BOPL4_OBS_PLAYER_OFFSET+(size_t)i*BOPL4_PLAYER_FEATURES, state, &state->players[index], self,
                         self_x, self_y, self_vx, self_vy, ablation);
    }
    bopl_zero_tail(obs+BOPL4_OBS_PLAYER_OFFSET, BOPL4_PLAYER_FEATURES, (int)player_count, prev_players, BOPL4_PLAYERS);

    for (int i=0; i<slot_count; i++)
        bopl4_slot_row(obs+BOPL4_OBS_SLOT_OFFSET+(size_t)i*BOPL4_SLOT_FEATURES, state, &state->slimes[slot_indices[i]/3],
                       slot_indices[i]%3, self, self_x, self_y, &enemy);
    bopl_zero_tail(obs+BOPL4_OBS_SLOT_OFFSET, BOPL4_SLOT_FEATURES, slot_count, prev_slots, BOPL4_SLOTS);

    for (int i=0; i<count; i++)
        bopl4_entity_row(obs+BOPL4_OBS_ENTITY_OFFSET+(size_t)i*BOPL4_ENTITY_FEATURES, state, &state->entities[nearest[i]], self,
                         self_x, self_y, self_vx, self_vy, &enemy, ablation);
    bopl_zero_tail(obs+BOPL4_OBS_ENTITY_OFFSET, BOPL4_ENTITY_FEATURES, count, prev_entities, BOPL4_ENTITIES);

    for (int i=0; i<slime_count; i++) {
        const BoplRlSlime *slime=&state->slimes[nearby_slimes[i]];
        bopl4_sincos(slime, &sin_r, &cos_r);
        bopl4_slime_row(obs+BOPL4_OBS_SLIME_OFFSET+(size_t)i*BOPL4_SLIME_FEATURES, state, slime, self, self_x, self_y, self_vx, self_vy,
                        &enemy, slime==controller || slime==enemy_controller, sin_r, cos_r, ablation);
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
    int state_row=bopl4_state_row(bopl_obs_row(env, a));
    if (state_row>=0) {
        /* The GPU writes the opponent block, so 43/44 too (bopl_rows_v4.h bopl4_stale_enemy). */
        bopl4_state_rows[state_row].enemy=enemy!=NULL;
        if (enemy) { bopl4_state_rows[state_row].enemy_dx=bopl_fix(enemy->x)-self_x; bopl4_state_rows[state_row].enemy_dy=bopl_fix(enemy->y)-self_y; }
    } else if (enemy) {
        out[43]=bopl4_stale_enemy(bopl_fix(enemy->x)-self_x, opp[9]); out[44]=bopl4_stale_enemy(bopl_fix(enemy->y)-self_y, opp[10]);
    }
    if (controller) { out[45]=bopl_clip(bopl_fix(controller->vx)/BOPL4_VELOCITY_SCALE); out[46]=bopl_clip(bopl_fix(controller->vy)/BOPL4_VELOCITY_SCALE); }
}

#endif
