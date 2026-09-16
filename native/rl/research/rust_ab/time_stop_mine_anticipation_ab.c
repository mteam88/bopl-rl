/* Time Stop + Mine golden: collision-triggered anticipation, frozen item
 * clock/body, then restored MineExplosion registration.
 *
 * The Mine form collider center now preserves FixTransform's exact offset
 * basis separately from the angle-derived box axes.
 * Other entity position, velocity and rotation fields allow 256 raw units
 * (5.96e-8) after chase normalization and its shockwave; discrete state,
 * collider geometry, discrete player/slime state and telemetry remain exact;
 * post-explosion player/slime positions and velocities use the same 256-raw
 * bound because they inherit the shockwave's fixed-point grouping drift.
 */
#include "bopl_rl.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static int64_t distance(int64_t a, int64_t b) {
    return a > b ? a - b : b - a;
}

static int entity_equal(const BoplRlEntity *a, const BoplRlEntity *b) {
    return a->id == b->id && a->kind == b->kind && a->layer == b->layer &&
           a->owner == b->owner && a->flags == b->flags &&
           a->hierarchy == b->hierarchy && distance(a->x, b->x) <= 256 && distance(a->y, b->y) <= 256 &&
           distance(a->vx, b->vx) <= 256 && distance(a->vy, b->vy) <= 256 &&
           distance(a->right_x, b->right_x) <= 1 &&
           distance(a->right_y, b->right_y) <= 1 &&
           distance(a->up_x, b->up_x) <= 1 &&
           distance(a->up_y, b->up_y) <= 1 &&
           distance(a->radius, b->radius) <= 1 &&
           distance(a->rotation, b->rotation) <= 256 &&
           distance(a->angular_velocity, b->angular_velocity) <= 256 &&
           a->inverse_mass == b->inverse_mass;
}

static int player_equal(const BoplRlPlayer *a, const BoplRlPlayer *b) {
    BoplRlPlayer ac = *a, bc = *b;
    if (distance(ac.x, bc.x) > 256 || distance(ac.y, bc.y) > 256)
        return 0;
    ac.x = bc.x; ac.y = bc.y;
    return memcmp(&ac, &bc, sizeof ac) == 0;
}

static int slime_equal(const BoplRlSlime *a, const BoplRlSlime *b) {
    BoplRlSlime ac = *a, bc = *b;
    if (distance(ac.x, bc.x) > 256 || distance(ac.y, bc.y) > 256 ||
        distance(ac.vx, bc.vx) > 256 || distance(ac.vy, bc.vy) > 256)
        return 0;
    ac.x = bc.x; ac.y = bc.y; ac.vx = bc.vx; ac.vy = bc.vy;
    return memcmp(&ac, &bc, sizeof ac) == 0;
}

static int reset(BoplRl *world, BoplRlConfig *config, const char *label) {
    int status = bopl_rl_reset(world, config);
    if (status != BOPL_RL_OK) {
        fprintf(stderr, "%s reset failed (%d): %s\n", label, status,
                bopl_rl_error(world));
        return 1;
    }
    return 0;
}

static const BoplRlSlime *slime_for(const BoplRlState *state, uint32_t owner) {
    for (unsigned i = 0; i < state->header.slime_count; i++)
        if (state->slimes[i].owner == owner) return &state->slimes[i];
    return NULL;
}

static int compare(const BoplRlState *a, const BoplRlState *b, unsigned tick) {
    if (memcmp(&a->header, &b->header, sizeof a->header) != 0) {
        fprintf(stderr,
                "tick %u header differs: ticks %u/%u entities %u/%u slimes %u/%u\n",
                tick, a->header.tick, b->header.tick, a->header.entity_count,
                b->header.entity_count, a->header.slime_count,
                b->header.slime_count);
        return 1;
    }
    for (unsigned i = 0; i < a->header.player_count; i++) {
        if (!player_equal(&a->players[i], &b->players[i]) ||
            memcmp(&a->telemetry[i], &b->telemetry[i],
                   sizeof a->telemetry[i]) != 0) {
            fprintf(stderr,
                    "tick %u player %u differs: flags %u/%u alive %u/%u "
                    "position (%" PRId64 ",%" PRId64 ")/(%" PRId64
                    ",%" PRId64 ") scale %" PRId64 "/%" PRId64
                    " activations %u/%u\n",
                    tick, i, a->players[i].flags, b->players[i].flags,
                    a->players[i].alive_bodies, b->players[i].alive_bodies,
                    a->players[i].x, a->players[i].y, b->players[i].x,
                    b->players[i].y, a->players[i].scale,
                    b->players[i].scale,
                    a->telemetry[i].ability_activations[0],
                    b->telemetry[i].ability_activations[0]);
            for (unsigned z = 0; z < a->header.slime_count; z++) fprintf(stderr, "  slime %u owner %u/%u pos (%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ") vel (%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ") flags %u/%u\n", z, a->slimes[z].owner, b->slimes[z].owner, a->slimes[z].x,a->slimes[z].y,b->slimes[z].x,b->slimes[z].y,a->slimes[z].vx,a->slimes[z].vy,b->slimes[z].vx,b->slimes[z].vy,a->slimes[z].ability_flags[0],b->slimes[z].ability_flags[0]);
            return 1;
        }
    }
    for (unsigned i = 0; i < a->header.slime_count; i++) {
        if (!slime_equal(&a->slimes[i], &b->slimes[i])) {
            fprintf(stderr,
                    "tick %u slime %u differs: position (%" PRId64 ",%" PRId64
                    ")/(%" PRId64 ",%" PRId64 ") ability flags %u/%u\n",
                    tick, i, a->slimes[i].x, a->slimes[i].y, b->slimes[i].x,
                    b->slimes[i].y, a->slimes[i].ability_flags[0],
                    b->slimes[i].ability_flags[0]);
            fprintf(stderr,
                    "  velocity (%" PRId64 ",%" PRId64 ")/(%" PRId64
                    ",%" PRId64 ") state %u/%u\n",
                    a->slimes[i].vx, a->slimes[i].vy, b->slimes[i].vx,
                    b->slimes[i].vy, a->slimes[i].state, b->slimes[i].state);
            const unsigned char *aa=(const unsigned char *)&a->slimes[i];
            const unsigned char *bb=(const unsigned char *)&b->slimes[i];
            for (unsigned z=0; z<sizeof a->slimes[i]; z++) if (aa[z]!=bb[z]) fprintf(stderr, "  byte %u %u/%u\n",z,aa[z],bb[z]);
            for (unsigned e = 0; e < a->header.entity_count; e++) {
                if (a->entities[e].kind == 2)
                    fprintf(stderr,
                            "  platform %u p (%" PRId64 ",%" PRId64
                            ")/(%" PRId64 ",%" PRId64 ") v (%" PRId64
                            ",%" PRId64 ")/(%" PRId64 ",%" PRId64 ")\n",
                            e, a->entities[e].x, a->entities[e].y,
                            b->entities[e].x, b->entities[e].y,
                            a->entities[e].vx, a->entities[e].vy,
                            b->entities[e].vx, b->entities[e].vy);
            }
            return 1;
        }
    }
    for (unsigned i = 0; i < a->header.entity_count; i++) {
        if (!entity_equal(&a->entities[i], &b->entities[i])) {
            fprintf(stderr,
                    "tick %u entity %u differs: id %u/%u kind %u/%u flags %u/%u "
                    "position (%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64
                    ") velocity (%" PRId64 ",%" PRId64 ")/(%" PRId64
                    ",%" PRId64 ")\n",
                    tick, i, a->entities[i].id, b->entities[i].id,
                    a->entities[i].kind, b->entities[i].kind,
                    a->entities[i].flags, b->entities[i].flags,
                    a->entities[i].x, a->entities[i].y, b->entities[i].x,
                    b->entities[i].y, a->entities[i].vx, a->entities[i].vy,
                    b->entities[i].vx, b->entities[i].vy);
            fprintf(stderr,
                    "  hierarchy %u/%u layer %u/%u owner %u/%u right "
                    "(%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64
                    ") up (%" PRId64 ",%" PRId64 ")/(%" PRId64 ",%" PRId64
                    ") radius %" PRId64 "/%" PRId64 " rotation %" PRId64
                    "/%" PRId64 " angular %" PRId64 "/%" PRId64
                    " inverse_mass %" PRId64 "/%" PRId64 "\n",
                    a->entities[i].hierarchy, b->entities[i].hierarchy,
                    a->entities[i].layer, b->entities[i].layer,
                    a->entities[i].owner, b->entities[i].owner,
                    a->entities[i].right_x, a->entities[i].right_y,
                    b->entities[i].right_x, b->entities[i].right_y,
                    a->entities[i].up_x, a->entities[i].up_y,
                    b->entities[i].up_x, b->entities[i].up_y,
                    a->entities[i].radius, b->entities[i].radius,
                    a->entities[i].rotation, b->entities[i].rotation,
                    a->entities[i].angular_velocity,
                    b->entities[i].angular_velocity,
                    a->entities[i].inverse_mass, b->entities[i].inverse_mass);
            return 1;
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    char error[512] = {0};
    BoplRl *world[2];
    for (int backend = 0; backend < 2; backend++) {
        world[backend] = bopl_rl_open(argv[backend + 1], "/tmp/gate_bootstrap.rep",
                                      "/tmp/time_stop_mine_chase_ab.log", 120000,
                                      error, sizeof error);
        if (!world[backend]) { fprintf(stderr, "open: %s\n", error); return 1; }
    }
    BoplRlConfig config;
    bopl_rl_default_config(&config, 6);
    config.seed=123; config.map=0; config.stocks=5;
    config.warmup_ticks=240; config.max_episode_ticks=2000;
    config.player[0].abilities[0]=24;
    config.player[0].abilities[1]=2;
    config.player[0].abilities[2]=14;
    config.player[1].abilities[0]=2;
    config.player[1].abilities[1]=4;
    config.player[1].abilities[2]=14;
    for (int p=2; p<6; p++) {
        config.player[p].abilities[0]=2;
        config.player[p].abilities[1]=4;
        config.player[p].abilities[2]=14;
    }
    config.player[5].abilities[0]=12;
    config.player[5].abilities[1]=2;
    config.player[5].abilities[2]=14;
    if (reset(world[0], &config, "reference") || reset(world[1], &config, "rust")) return 1;
    if (compare(bopl_rl_state(world[0]), bopl_rl_state(world[1]), 0)) return 1;
    unsigned cast=0, stop_cast=0;
    unsigned mine_tick=0, stop_tick=0, restore_tick=0, explosion_tick=0;
    unsigned frozen_ticks=0;
    int saw_mine=0, saw_explosion=0, saw_disabled_explosion=0;
    int saw_stopped=0, saw_restored=0, saw_protection=0;
    int saw_protected_cooldown=0, froze_mine=0;
    int64_t frozen_x=0, frozen_y=0, frozen_vx=0, frozen_vy=0;
    int64_t frozen_rotation=0, frozen_angular=0;
    int64_t frozen_target_x=0, frozen_target_y=0, frozen_target_timers[3]={0};
    int64_t caster_other_timer=0;
    uint32_t frozen_mine_id=0;
    for (unsigned tick=1; tick<=1460; tick++) {
        const BoplRlState *before=bopl_rl_state(world[0]);
        BoplRlAction action[6]={{0}};
        if (!cast && tick >= 550 && before->slimes[0].timers[0] > before->slimes[0].cooldowns[0]) cast=tick;
        if (cast && tick <= cast+30) { action[0].buttons=2; action[0].angle=128; }
        if (!stop_cast && tick > 180) {
            for (unsigned i=0;i<before->header.slime_count;i++)
                if (before->slimes[i].owner==6 &&
                    before->slimes[i].timers[0] > before->slimes[i].cooldowns[0]) {
                    stop_cast=tick;
                    break;
                }
        }
        if (stop_cast && tick < stop_cast+600) action[5].buttons=2;
        if (tick == 725) action[4].buttons=1;
        for (int b=0;b<2;b++) if (bopl_rl_step(world[b],action,6,1)!=BOPL_RL_OK) return 1;
        const BoplRlState *reference=bopl_rl_state(world[0]);
        if (compare(reference,bopl_rl_state(world[1]),tick)) return 1;
        const BoplRlSlime *caster=slime_for(reference,6);
        const BoplRlSlime *target=slime_for(reference,5);
        if (!caster || !target) {
            fprintf(stderr,"tick %u missing Time Stop caster or Mine target\n",tick);
            return 1;
        }
        const BoplRlEntity *mine=NULL;
        for (unsigned e=0;e<reference->header.entity_count;e++) {
            const BoplRlEntity *entity=&reference->entities[e];
            if (entity->kind==0 && entity->layer==10 && (entity->flags & 1)) {
                saw_mine=1;
                if (!mine_tick) mine_tick=tick;
                mine=entity;
            }
            if (entity->kind==1 && entity->layer==18) {
                saw_explosion=1;
                if (!explosion_tick) explosion_tick=tick;
                if (entity->flags==9) saw_disabled_explosion=1;
            }
        }
        if (reference->header.player_time_scale==0) {
            if (!saw_stopped) {
                if (!mine || saw_explosion) {
                    fprintf(stderr,"tick %u stop did not begin during live Mine anticipation\n",tick);
                    return 1;
                }
                saw_stopped=1;
                stop_tick=tick;
                frozen_mine_id=mine->id;
                frozen_x=mine->x; frozen_y=mine->y;
                frozen_vx=mine->vx; frozen_vy=mine->vy;
                frozen_rotation=mine->rotation;
                frozen_angular=mine->angular_velocity;
                frozen_target_x=target->x; frozen_target_y=target->y;
                memcpy(frozen_target_timers,target->timers,sizeof frozen_target_timers);
                caster_other_timer=caster->timers[1];
                froze_mine=1;
            } else if (!mine || mine->id!=frozen_mine_id ||
                       mine->x!=frozen_x || mine->y!=frozen_y ||
                       mine->vx!=frozen_vx || mine->vy!=frozen_vy ||
                       mine->rotation!=frozen_rotation ||
                       mine->angular_velocity!=frozen_angular ||
                       target->x!=frozen_target_x || target->y!=frozen_target_y ||
                       memcmp(target->timers,frozen_target_timers,
                              sizeof frozen_target_timers)!=0 || saw_explosion) {
                fprintf(stderr,"tick %u Mine or target advanced during stop\n",tick);
                return 1;
            }
            frozen_ticks++;
            if ((reference->players[5].flags & 8) &&
                !(reference->players[4].flags & 8))
                saw_protection=1;
            if (caster->timers[0]==0 && caster->timers[1]>caster_other_timer)
                saw_protected_cooldown=1;
        } else if (saw_stopped) {
            saw_restored=1;
            if (!restore_tick) restore_tick=tick;
        }
    }
    const BoplRlState *final=bopl_rl_state(world[0]);
    if (!cast || !stop_cast || !saw_mine || !saw_stopped || !saw_restored ||
        !saw_protection || !saw_protected_cooldown || !froze_mine ||
        !saw_explosion || !saw_disabled_explosion || frozen_ticks!=690 ||
        stop_tick-mine_tick<=120 || explosion_tick-restore_tick!=5 ||
        final->header.player_time_scale!=(1LL<<32)) {
        fprintf(stderr,"incomplete cast=%u stop_cast=%u mine=%d stopped=%d "
                "restored=%d protection=%d cooldown=%d froze=%d explosion=%d "
                "disabled=%d mine_tick=%u stop=%u restore=%u explosion_tick=%u "
                "frozen=%u scale=%" PRId64 "\n",
                cast,stop_cast,saw_mine,saw_stopped,saw_restored,saw_protection,
                saw_protected_cooldown,froze_mine,saw_explosion,
                saw_disabled_explosion,mine_tick,stop_tick,restore_tick,
                explosion_tick,frozen_ticks,final->header.player_time_scale);
        return 1;
    }
    for (int b=0;b<2;b++) bopl_rl_close(world[b]);
    printf("TIME_STOP_MINE_ANTICIPATION_GOLDEN_OK ticks=1460 frozen=%u post_restore=%u\n",
           frozen_ticks,explosion_tick-restore_tick);
    return 0;
}
