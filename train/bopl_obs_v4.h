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
 * around rays 20 and 28) carry the aim ray and the line of sight instead.
 * BOPL4_ABLATE_OBSFIX is layout v4.2, not an ablation either: creators as
 * owners, smoke out of the rays and last in the rows, Spike's spikes lethal
 * walls (bopl_rows_v4.h). */
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
 * layers (11 wall, 16 lethal, 21 non-lethal) and in v4.2 the boxes on 16, water the highest water
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
        int kind=bopl4_scene_entity(&state->entities[i], w, &top, &lethal, &skew, ablation);
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
        if (bopl4_entity_circle(&state->entities[nearest[i]], self, c, &cls, ablation)) bopl4_scene_circle(s, s->circle_count++, c, cls);
    }
    for (int i=0; i<slime_count; i++) {
        float c[5]; int32_t cls;
        if (bopl4_slime_circle(state, &state->slimes[nearby_slimes[i]], self, c, &cls)) bopl4_scene_circle(s, s->circle_count++, c, cls);
    }
}
_Static_assert(BOPL4F_BLAST_X==BOT_BLAST_X && BOPL4F_BLAST_Y==BOT_BLAST_Y, "the rays' blast planes are the bots'");
_Static_assert(BOPL4_SLIME_RADIUS==BOT_SLIME_RADIUS, "the scene's slime radius is the bots'");
/* One state's entity relations (bopl4_relation of an entity versus a
 * player), kept across that state's rows on this thread. An entity row needs
 * its body versus the observer (part 1) and versus the observer's primary
 * enemy (part 3), and the enemy is another player whose own row asks the
 * same relation as its part 1: in 1v1 each player's enemy is the other, so
 * every entity both rows keep was related twice to each player. The
 * relation is a pure function of its arguments, so a key (player, entity)
 * serves both parts when the player's position and body velocity are the
 * same doubles in both (the row's self_x, self_y, self_vx, self_vy; the
 * enemy's x, y, vx, vy: bopl_fix of the same fields and the same
 * bopl_controlling_slime), which `observer` checks bit for bit, and the
 * ablation's one relational bit agrees. Live between bopl4_relations_open
 * and _close (bopl_encode_shared, one state's rows); stamp = generation
 * marks a stored relation. */
static __thread struct {
    const BoplRlState *state;
    uint64_t generation, *stamp;
    float (*relation)[BOPL4_RELATION];
    size_t capacity;
    uint32_t entities;
    int relational;                         /* the ablation bit of the stored relations, -1 none yet */
    int observer_set[BOPL_MAX_PLAYERS];
    double observer[BOPL_MAX_PLAYERS][4];
} bopl4_relations;
static void bopl4_relations_open(const BoplRlState *state) {
    bopl4_relations.state=NULL;
    if (!state || state->header.player_count>BOPL_MAX_PLAYERS) return;
    size_t need=(size_t)state->header.player_count*state->header.entity_count;
    if (need>bopl4_relations.capacity) {
        size_t capacity=need<1024?1024:need*2;
        uint64_t *stamp=realloc(bopl4_relations.stamp, capacity*sizeof(*stamp));
        if (!stamp) return;
        bopl4_relations.stamp=stamp;
        memset(stamp, 0, capacity*sizeof(*stamp));   /* generation starts at 1: nothing stored */
        float (*relation)[BOPL4_RELATION]=realloc(bopl4_relations.relation, capacity*sizeof(*relation));
        if (!relation) { bopl4_relations.capacity=0; return; }
        bopl4_relations.relation=relation; bopl4_relations.capacity=capacity;
    }
    bopl4_relations.generation++;
    bopl4_relations.state=state; bopl4_relations.entities=state->header.entity_count;
    bopl4_relations.relational=-1;
    memset(bopl4_relations.observer_set, 0, sizeof(bopl4_relations.observer_set));
}
static void bopl4_relations_close(void) { bopl4_relations.state=NULL; }
/* The cache key of `player` (an index into state->players) as an observer at
 * (x, y, vx, vy) for this row, or -1 to relate without the cache. */
static int bopl4_relation_key(const BoplRlState *state, const BoplRlPlayer *player, double x, double y, double vx, double vy, int ablation) {
    if (bopl4_relations.state!=state || !player) return -1;
    int relational=(ablation&BOPL4_ABLATE_RELATIONAL)!=0;
    if (bopl4_relations.relational>=0 && bopl4_relations.relational!=relational) return -1;
    int p=(int)(player-state->players);
    if (p<0 || p>=(int)state->header.player_count) return -1;
    double at[4]={x, y, vx, vy};
    if (!bopl4_relations.observer_set[p]) {
        memcpy(bopl4_relations.observer[p], at, sizeof(at)); bopl4_relations.observer_set[p]=1;
    } else if (memcmp(bopl4_relations.observer[p], at, sizeof(at))) return -1;
    bopl4_relations.relational=relational;
    return p;
}
/* bopl4_entity_part's relation of entity `index` versus an observer at
 * (x, y, vx, vy), from the cache under `key` when there is one. */
static const float *bopl4_entity_relation(float *scratch, const BoplRlState *state, uint32_t index, int key,
                                          double x, double y, double vx, double vy, int ablation) {
    size_t at=(size_t)key*bopl4_relations.entities+index;
    if (key>=0 && bopl4_relations.stamp[at]==bopl4_relations.generation) return bopl4_relations.relation[at];
    const BoplRlEntity *entity=&state->entities[index];
    double ex=bopl_fix(entity->x), ey=bopl_fix(entity->y), evx=bopl_fix(entity->vx), evy=bopl_fix(entity->vy);
    double radius=bopl_fix(entity->radius);
    int projectile=(entity->flags&BOPL_ENTITY_PROJECTILE)!=0 && entity->inverse_mass>0;
    float *out=key>=0?bopl4_relations.relation[at]:scratch;
    bopl4_relation(out, ex-x, ey-y, evx-vx, evy-vy, radius, projectile, ablation);
    if (key>=0) bopl4_relations.stamp[at]=bopl4_relations.generation;
    return out;
}
/* bopl4_entity_row (parts 0-3) with parts 1 and 3's relations through the
 * cache (self_key and enemy_key from bopl4_relation_key) and part 2's
 * surface distance the one the row's nearest pass found (`surface`:
 * bopl_surface_distance of the entity from self_x, self_y). */
static void bopl4_entity_row_keyed(float *row, const BoplRlState *state, uint32_t index, const BoplRlPlayer *self,
                                   double self_x, double self_y, double self_vx, double self_vy, const Bopl4Enemy *enemy, int ablation,
                                   int self_key, int enemy_key, double surface) {
    const BoplRlEntity *entity=&state->entities[index];
    float scratch[BOPL4_RELATION];
    bopl4_entity_part(row, 0, state, entity, self, self_x, self_y, self_vx, self_vy, enemy, ablation);
    const float *relation=bopl4_entity_relation(scratch, state, index, self_key, self_x, self_y, self_vx, self_vy, ablation);
    for (int f=1; f<BOPL4_RELATION; f++) row[33+f]=relation[f];
    row[33]=bopl_clip(surface/64.0);   /* part 2 */
    row[45]=row[46]=0.0f;
    if (enemy->known) {
        int projectile=(entity->flags&BOPL_ENTITY_PROJECTILE)!=0 && entity->inverse_mass>0;
        const float *versus=bopl4_entity_relation(scratch, state, index, enemy_key, enemy->x, enemy->y, enemy->vx, enemy->vy, ablation);
        row[45]=versus[0]; row[46]=projectile?versus[10]:versus[8];
    }
}
/* Each visible entity's surface distance from the observer, by entity index,
 * as the row's nearest pass computes it (this thread's current row). */
static __thread double *bopl4_surface; static __thread size_t bopl4_surface_capacity;
/* Layout v4.3 (BOPL4_ABLATE_DRILL), every field's width and meaning kept:
 * - a slime in an ability whose player's live Drill form exports its body
 *   (BOPL_ENTITY_FORM_BODY, sim/abi/bopl_rl.h; the training library with
 *   BOPL_RL_FORM_BODY=1) reads the form's body: vx, vy the drill body's real
 *   velocity (the form row's rotation, angular_velocity), its rotation the
 *   form's (atan2 of its right axis; the heading is -up), where the slime's
 *   record is frozen at the form's entry. That is the observer's own body
 *   block and its relations' velocity, the opponent's body block, relation
 *   and the rays' closing speeds, and the slime rows' velocity and rotation;
 *   positions are the players' as before. The critic's block is untouched.
 * - the fan's and the aim's rays from inside the ground march to where they
 *   leave it (bopl_rays_v4.h bopl4f_march, BOPL4F_INSIDE).
 * - a wall or water of the scene is not also a circle (bopl4_entity_circle).
 * Without the bit every row is what v4.2 writes; with the bit but without the
 * export, the rays and circles change and the body blocks do not. */
/* The form body of `slime` (an ability's, its owner's FORM_BODY row nearest
 * its record), or NULL. */
static const BoplRlEntity *bopl4_form_body(const BoplRlState *state, const BoplRlSlime *slime) {
    if (!(slime->flags&BOPL_SLIME_IN_ABILITY)) return NULL;
    const BoplRlEntity *best=NULL; double best_distance=0.0;
    double sx=bopl_fix(slime->x), sy=bopl_fix(slime->y);
    for (uint32_t i=0; i<state->header.entity_count; i++) {
        const BoplRlEntity *e=&state->entities[i];
        if (!(e->flags&BOPL_ENTITY_FORM_BODY) || e->owner!=slime->owner) continue;
        double dx=bopl_fix(e->x)-sx, dy=bopl_fix(e->y)-sy, distance=dx*dx+dy*dy;
        if (!best || distance<best_distance) { best=e; best_distance=distance; }
    }
    return best;
}
/* Slime record `slime` as layout v4.3 reads it given its form body. */
static void bopl4_form_slime(BoplRlSlime *slime, const BoplRlEntity *form) {
    slime->vx=form->rotation; slime->vy=form->angular_velocity;
    double angle=atan2(bopl_fix(form->right_y), bopl_fix(form->right_x));
    slime->rotation=(int64_t)llround(angle*4294967296.0);
}
/* Whether any slime of `state` has a form body (any FORM_BODY row at all,
 * the common case being none). */
static int bopl4_form_any(const BoplRlState *state) {
    for (uint32_t i=0; i<state->header.entity_count; i++)
        if (state->entities[i].flags&BOPL_ENTITY_FORM_BODY) return 1;
    return 0;
}
/* `state` with its slimes as layout v4.3 reads them in *view (its slimes in
 * this thread's buffer), or `state` itself when no slime has a form body. */
static __thread BoplRlSlime *bopl4_form_slimes; static __thread size_t bopl4_form_capacity;
static const BoplRlState *bopl4_form_view(const BoplRlState *state, BoplRlState *view) {
    if (!bopl4_form_any(state)) return state;
    uint32_t n=state->header.slime_count;
    if (n>bopl4_form_capacity) {
        size_t capacity=n<16?16:(size_t)n*2;
        BoplRlSlime *grown=realloc(bopl4_form_slimes, capacity*sizeof(*grown));
        if (!grown) abort();
        bopl4_form_slimes=grown; bopl4_form_capacity=capacity;
    }
    int patched=0;
    for (uint32_t i=0; i<n; i++) {
        bopl4_form_slimes[i]=state->slimes[i];
        const BoplRlEntity *form=bopl4_form_body(state, &state->slimes[i]);
        if (form) { bopl4_form_slime(&bopl4_form_slimes[i], form); patched=1; }
    }
    if (!patched) return state;
    *view=*state; view->slimes=bopl4_form_slimes;
    return view;
}
/* Encode `state` into player a's v4 row. Same contract as the v3 row writer:
 * every feature stored, unused set slots zeroed (with the bopl_written record
 * of the previous encode), the own-action tail zeroed and the privileged block
 * left for bopl4_privileged. */
static void bopl_encode_v4_row(BoplPuffer *env, const BoplRlState *state, int a, int count_overflow) {
    bopl4_row_ablation=bopl_ablate_of(env, a);   /* this player's writer family for the whole row */
    int ablation=bopl4_ablation();
    BoplRlState form_view;
    if (ablation&BOPL4_ABLATE_DRILL) state=bopl4_form_view(state, &form_view);
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
    if (state->header.entity_count>bopl4_surface_capacity) {
        size_t capacity=state->header.entity_count<256?256:(size_t)state->header.entity_count*2;
        double *grown=realloc(bopl4_surface, capacity*sizeof(*grown));
        if (!grown) abort();
        bopl4_surface=grown; bopl4_surface_capacity=capacity;
    }
    for (uint32_t i=0; i<state->header.entity_count;) {
        int end=(state->entities[i].kind==3)?bopl_rope_group_end(state, i):(int)i+1;
        int group=end-(int)i;
        for (int j=0; j<group; j++) {
            const BoplRlEntity *entity=&state->entities[i+j];
            if (entity->kind==3 && !bopl_rope_span_kept(j, group)) continue;
            if (!bopl_entity_visible(state, entity, self->id)) continue;
            double distance=bopl_surface_distance(entity, self_x, self_y);
            bopl4_surface[i+(uint32_t)j]=distance;
            bopl_nearest(nearest, distances, &c.count, BOPL4_ENTITIES, i+(uint32_t)j, bopl4_rank_distance(entity, distance, ablation));
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

    int self_key=bopl4_relation_key(state, self, self_x, self_y, self_vx, self_vy, ablation);
    int enemy_key=enemy.known?bopl4_relation_key(state, enemy.player, enemy.x, enemy.y, enemy.vx, enemy.vy, ablation):-1;
    for (int i=0; i<count; i++)
        bopl4_entity_row_keyed(obs+BOPL4_OBS_ENTITY_OFFSET+(size_t)i*BOPL4_ENTITY_FEATURES, state, nearest[i], self,
                               self_x, self_y, self_vx, self_vy, &enemy, ablation, self_key, enemy_key, bopl4_surface[nearest[i]]);
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
 * velocity next to the delayed one the actor sees. bodies[i]: player i's
 * controlling slime (bopl_controlling_slime at its position), found once
 * for every row of the state. */
static void bopl4_privileged(BoplPuffer *env, const BoplRlState *state, int a, const BoplRlSlime *const *bodies,
                             const BoplRlAction *enemy_action, const int *enemy_hold_left, const uint16_t *enemy_held) {
    /* A row shipped as its state (pack mode 4) is not read on the host
     * unless checked (BOPL_RAYS_CHECK): its block is written in the slot,
     * just stored, and not in the row's cold lines for bopl_pack_rows to copy. */
    float *row=bopl_obs_row(env, a);
    int state_row=bopl4_state_row(row);
    float *out=state_row>=0 && !bopl4_check_rows()?bopl4_state_rows[state_row].slot+BOPL4_STATE_FIXED_OFFSET+BOPL4_PREVIOUS
                                                  :row+BOPL4_OBS_PRIVILEGED_OFFSET;
    memset(out, 0, BOPL4_PRIVILEGED*sizeof(float));
    if (!state) return;
    const BoplRlPlayer *self=&state->players[a];
    double self_x=bopl_fix(self->x), self_y=bopl_fix(self->y);
    const BoplRlSlime *controller=bodies[a];
    const BoplRlPlayer *enemy=NULL; const BoplRlSlime *body=NULL;
    for (uint32_t i=0; i<state->header.player_count && !enemy; i++)
        if (state->players[i].id!=self->id && state->players[i].team!=self->team) { enemy=&state->players[i]; body=bodies[i]; }
    if (enemy) {
        double x=bopl_fix(enemy->x), y=bopl_fix(enemy->y);
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
    const float *opp=row+BOPL4_OBS_OPPONENT_OFFSET;
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
