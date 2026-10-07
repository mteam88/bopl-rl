// The GPU writer of the v4 row's terrain block and rays (bopl_rays_v4.h):
// rows the env workers shipped with a scene (pack modes 2 and 3,
// bopl_obs_pack.h) get them here, bit for bit what bopl4f_write gives on the
// CPU.
//
// Its own translation unit, built with -fmad=false (train/build.py): no
// multiply-add contraction anywhere in it, as the CPU writer is built
// without; the policy's kernels keep their FMAs. sqrt and division go through
// __fsqrt_rn / __fdiv_rn (bopl4f_sqrt, bopl4f_div).
//
// Built for latency: the trainer launches a few hundred rows per buffer step,
// so a row's critical path is the cost, not the throughput. One block per
// row, the scene in shared memory, BR4_LANES threads per ray splitting each
// step's walls (and the surfaces' circles) between them. The CPU writer's
// march groups (each culled at its own reach, bopl4f_march_group) run
// concurrently where they do not depend on each other:
//   warps 0-4  the fan with the drop and ceiling probes
//   warps 5-6  the dash rays, then their falls (whether or not the drop
//              probe leaves them needed: speculative)
//   warp 7     the aim and line of sight, and the single SDFs (edge walls,
//              the body's SDF and gradient, the lethal distance)
// then, given the drop probe, the edge probes on warps 0-1. Exact: a group's
// reach and each step's SDF are minima / maxima of floats that are never NaN
// or -0 (bopl4f_box), so any order gives the CPU's bits; the surfaces' and
// lethal_at's first strict minimum reduce as (value, index).
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "bopl_layout.h"
#include "bopl_obs_pack.h"
#include "bopl_rows_v4.h"

constexpr int BR4_LANES = 4;                                      // threads per ray
constexpr int BR4_FAN_SLOTS = 40;                                 // the fan's rays and the two probes
constexpr int BR4_FAN_THREADS = BR4_FAN_SLOTS * BR4_LANES;        // warps 0-4
constexpr int BR4_DASH_THREADS = 16 * BR4_LANES;                  // warps 5-6
constexpr int BR4_EDGE_THREADS = 16 * BR4_LANES;                  // warps 0-1, after the join
constexpr int BR4_MISC = BR4_FAN_THREADS + BR4_DASH_THREADS;      // warp 7's first thread
constexpr int BR4_RAYS_THREADS = BR4_MISC + 32;
constexpr int BR4_WALLS = 2 * BOPL4F_WALLS;
static_assert(BR4_FAN_SLOTS >= BOPL4F_GROUP, "a slot per ray of the fan");
static_assert(BR4_FAN_THREADS % 32 == 0 && BR4_DASH_THREADS % 32 == 0 && BR4_EDGE_THREADS % 32 == 0, "groups are whole warps");
static_assert(32 % BR4_LANES == 0 && 2 * BR4_LANES <= 8, "a ray's lanes in one warp; the aim's two rays in lanes 0-7");
// Named barriers (0 is __syncthreads) and reach slots, one per march group.
enum { BR4_FAN = 1, BR4_DASH = 2, BR4_FALL = 3, BR4_AIM = 4, BR4_EDGE = 5, BR4_GROUPS = 6 };

__constant__ Bopl4Dirs br4_dirs;

// The row encode's working set (pack mode 4, br4_encode): the state the slot
// carries, and per entity what the passes find. Dead once the scene is
// built, so it shares its memory with the rays' walls.
struct Br4Encode {
    BoplRlState st;                             // header and players; entities and slimes point into Br4Shared::raw
    Bopl4Enemy enemy;
    const BoplRlPlayer* self;
    const BoplRlSlime *controller, *enemy_controller;
    double x, y, vx, vy;                        // the observer's position and its body's velocity
    Bopl4Counts c;
    int time_stop, nan, slime_count, slot_count;
    uint32_t nearest[BOPL4_ENTITIES], nearby_slimes[BOPL4_SLIMES], slot_indices[BOPL4_SLOTS];
    double distance[BOPL4_ENTITIES], top[BOPL4_ENTITIES];
    float wall[BOPL4_ENTITIES][9];
    unsigned char kept[BOPL4_ENTITIES], kind[BOPL4_ENTITIES], lethal[BOPL4_ENTITIES], skew[BOPL4_ENTITIES];
};
struct Br4Walls {
    // Every wall as a rotated one (aligned: right (1, 0), up (0, 1); the same
    // bits, see br4_wall_sdf), aligned walls first: index j is lethal_at's order.
    float cx[BR4_WALLS], cy[BR4_WALLS], rx[BR4_WALLS], ry[BR4_WALLS], ux[BR4_WALLS], uy[BR4_WALLS],
          ex[BR4_WALLS], ey[BR4_WALLS], radius[BR4_WALLS];
    unsigned char list[BR4_GROUPS][BR4_WALLS];  // a group's culled walls
};
struct Br4Shared {
    // The slot from the scene on (modes 2, 3) or the state (mode 4), one coalesced read.
    __align__(16) float raw[BOPL4_PACK_P - BOPL4_PACK_HEADER];
    Bopl4Scene s;
    Bopl4Anchor a;
    union { Br4Walls w; Br4Encode e; };
    unsigned reach[BR4_GROUPS];                 // a group's reach, as the bits of a non-negative float
    int listn[BR4_GROUPS];
    float probes[2], sdf_at[5], lethal, aim_out[2][BOPL4_RAY_FEATURES];
    int hit[16], edge_drop[16], dash_landing;
};
static_assert(sizeof(Br4Shared) <= 48 * 1024, "static shared memory");
static_assert(BOPL4_PACK_P - BOPL4_STATE_OFFSET <= BOPL4_PACK_P - BOPL4_PACK_HEADER, "a state slot fits raw");

__device__ __forceinline__ void br4_bar(int id, int threads) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory"); }
__device__ __forceinline__ int br4_bar_or(int id, int threads, int p) {
    int out;
    asm volatile("{ .reg .pred q, r; setp.ne.s32 q, %1, 0; bar.red.or.pred r, %2, %3, q; selp.s32 %0, 1, 0, r; }"
                 : "=r"(out) : "r"(p), "r"(id), "r"(threads) : "memory");
    return out;
}
// The shuffle mask of this thread's ray (its BR4_LANES lanes).
__device__ __forceinline__ unsigned br4_ray_mask() {
    return ((1u << BR4_LANES) - 1u) << ((threadIdx.x & 31) & ~(BR4_LANES - 1));
}

// bopl4f_rot_sdf of unified wall j. For an aligned wall px*1 + py*0 is px
// (+-0 added; fabsf drops the sign of a zero) and so equals bopl4f_axis_sdf.
__device__ __forceinline__ float br4_wall_sdf(const Br4Shared& sh, int j, float x, float y) {
    const float px = x - sh.w.cx[j], py = y - sh.w.cy[j];
    const float lx = px * sh.w.rx[j] + py * sh.w.ry[j], ly = px * sh.w.ux[j] + py * sh.w.uy[j];
    return bopl4f_box(fabsf(lx) - sh.w.ex[j], fabsf(ly) - sh.w.ey[j], sh.w.radius[j]);
}

// The scene from its packed form (bopl4f_scene_write), by the whole block:
// the slot's words from the scene to its end copied in one pass (one memory
// round trip), then taken apart in shared memory.
__device__ static void br4_scene_load(const float* slot, int at, Br4Shared& sh) {
    for (int i = threadIdx.x; i < BOPL4_PACK_P - at; i += blockDim.x) sh.raw[i] = slot[at + i];
    __syncthreads();
    const float* in = sh.raw;
    Bopl4Scene* s = &sh.s;
    const int32_t* w = (const int32_t*)in;
    const int na = w[1], nr = w[2], nc = w[3];
    if (threadIdx.x == 0) {
        s->flags = w[0]; s->axis_count = na; s->rot_count = nr; s->circle_count = nc;
        s->x = in[4]; s->y = in[5]; s->svx = in[6]; s->svy = in[7]; s->water_top = in[8];
        s->aim_dx = in[9]; s->aim_dy = in[10]; s->enemy_x = in[11]; s->enemy_y = in[12];
        const uint32_t* m = (const uint32_t*)in + 13;
        s->axis_lethal = m[0] | (uint64_t)m[1] << 32; s->rot_lethal = m[2] | (uint64_t)m[3] << 32; s->rot_skew = m[4] | (uint64_t)m[5] << 32;
    }
    in += BOPL4F_SCENE_HEAD;
    float* axis[5] = {s->axis_cx, s->axis_cy, s->axis_ex, s->axis_ey, s->axis_radius};
    float* rot[9] = {s->rot_cx, s->rot_cy, s->rot_rx, s->rot_ry, s->rot_ux, s->rot_uy, s->rot_ex, s->rot_ey, s->rot_radius};
    float* circles[5] = {s->cx, s->cy, s->cr, s->cvx, s->cvy};
    for (int f = 0; f < 5; f++, in += na) for (int i = threadIdx.x; i < na; i += blockDim.x) axis[f][i] = in[i];
    for (int f = 0; f < 9; f++, in += nr) for (int i = threadIdx.x; i < nr; i += blockDim.x) rot[f][i] = in[i];
    for (int f = 0; f < 5; f++, in += nc) for (int i = threadIdx.x; i < nc; i += blockDim.x) circles[f][i] = in[i];
    for (int i = threadIdx.x; i < nc; i += blockDim.x) s->cclass[i] = ((const int32_t*)in)[i];
    __syncthreads();
}
// The walls unified for the marches and their SDFs at the body (the anchors).
__device__ static void br4_scene_prepare(Br4Shared& sh) {
    const Bopl4Scene* s = &sh.s;
    const int na = s->axis_count, nr = s->rot_count;
    if (threadIdx.x < BR4_GROUPS) sh.reach[threadIdx.x] = 0u;
    for (int j = threadIdx.x; j < na + nr; j += blockDim.x) {
        if (j < na) {
            sh.w.cx[j] = s->axis_cx[j]; sh.w.cy[j] = s->axis_cy[j]; sh.w.rx[j] = 1.0f; sh.w.ry[j] = 0.0f; sh.w.ux[j] = 0.0f; sh.w.uy[j] = 1.0f;
            sh.w.ex[j] = s->axis_ex[j]; sh.w.ey[j] = s->axis_ey[j]; sh.w.radius[j] = s->axis_radius[j];
            sh.a.axis[j] = bopl4f_axis_sdf(s, j, s->x, s->y);
        } else {
            const int k = j - na;
            sh.w.cx[j] = s->rot_cx[k]; sh.w.cy[j] = s->rot_cy[k]; sh.w.rx[j] = s->rot_rx[k]; sh.w.ry[j] = s->rot_ry[k];
            sh.w.ux[j] = s->rot_ux[k]; sh.w.uy[j] = s->rot_uy[k]; sh.w.ex[j] = s->rot_ex[k]; sh.w.ey[j] = s->rot_ey[k]; sh.w.radius[j] = s->rot_radius[k];
            sh.a.rot[k] = bopl4f_rot_sdf(s, k, s->x, s->y);
        }
    }
    __syncthreads();
}

// ---- The row from its state (pack mode 4) ----
// bopl_encode_v4_row's passes split across the block, every feature from the
// functions the CPU writer calls (bopl_rows_v4.h), so the row is its bits:
// the entity pass per entity (the counts are integers, so any order sums
// alike; the nearest are all kept entities, as there are at most
// BOPL4_ENTITIES, ranked by (distance, index), which is bopl_nearest's
// stable order unless a distance is NaN, when it runs as written), the
// slime pass and the scene's walls and circles as written, then a thread per
// row. The previous and critic blocks come from the slot as the host wrote
// them, except critic 43 and 44, which read the opponent block. Returns with
// the scene in sh.s, the dense row written but for the terrain and rays.
constexpr int BR4_ENTITY_THREAD = 64;   // warps 2-3: the per-entity passes
__device__ static void br4_encode(const float* slot, float* out, Br4Shared& sh) {
    const int t = threadIdx.x;
    const Bopl4StateHead* head = (const Bopl4StateHead*)slot;
    const int words = (int)head->words - BOPL4_STATE_OFFSET, a = (int)head->player, ablation = (int)head->ablation;
    const uint32_t E = head->entity_count, S = head->slime_count;
    Br4Encode& e = sh.e;
    for (int i = t; i < words; i += blockDim.x) sh.raw[i] = slot[BOPL4_STATE_OFFSET + i];
    for (int i = t; i < BOPL4_PREVIOUS; i += blockDim.x) out[BOPL4_OBS_PREVIOUS_OFFSET + i] = slot[BOPL4_STATE_FIXED_OFFSET + i];
    for (int i = t; i < BOPL4_PRIVILEGED; i += blockDim.x)
        out[BOPL4_OBS_PRIVILEGED_OFFSET + i] = slot[BOPL4_STATE_FIXED_OFFSET + BOPL4_PREVIOUS + i];
    if (t == 0) {
        memset(&e.c, 0, sizeof(e.c));
        e.time_stop = BOPL4_ENTITIES; e.nan = 0;
    }
    __syncthreads();
    const char* raw = (const char*)sh.raw;
    const BoplRlEntity* entities = (const BoplRlEntity*)(raw + BOPL4_STATE_BASE_WORDS * 4);
    const BoplRlSlime* slimes = (const BoplRlSlime*)(raw + BOPL4_STATE_BASE_WORDS * 4 + E * sizeof(BoplRlEntity));
    const float* trig = (const float*)(raw + BOPL4_STATE_BASE_WORDS * 4 + E * sizeof(BoplRlEntity) + S * sizeof(BoplRlSlime));
    const unsigned char* sectors = (const unsigned char*)(trig + 2 * S);
    // The header and players (the state's first words), and each entity's part of the scene.
    for (int i = t; i < BOPL4_STATE_BASE_WORDS; i += blockDim.x) ((float*)&e.st)[i] = sh.raw[i];
    if (t == 0) { e.st.entities = entities; e.st.slimes = slimes; }
    if (t >= BR4_ENTITY_THREAD && t < BR4_ENTITY_THREAD + (int)E) {
        const int k = t - BR4_ENTITY_THREAD;
        double top = 0.0; int lethal = 0, skew = 0;
        e.kind[k] = (unsigned char)bopl4_scene_entity(&entities[k], e.wall[k], &top, &lethal, &skew);
        e.top[k] = top; e.lethal[k] = (unsigned char)lethal; e.skew[k] = (unsigned char)skew;
    }
    __syncthreads();
    const BoplRlState* state = &e.st;
    const BoplRlPlayer* self = &state->players[a];
    const double self_x = bopl_fix(self->x), self_y = bopl_fix(self->y);
    if (t == 0) {
        // The observer, its body, the primary enemy, and the scene's head.
        const BoplRlSlime* controller = bopl_controlling_slime(state, self->id, self_x, self_y);
        e.self = self; e.controller = controller; e.x = self_x; e.y = self_y;
        e.vx = controller ? bopl_fix(controller->vx) : 0.0; e.vy = controller ? bopl_fix(controller->vy) : 0.0;
        e.enemy = bopl4_primary_enemy(state, self, self_x, self_y);
        e.enemy_controller = e.enemy.player ? bopl_controlling_slime(state, e.enemy.player->id, e.enemy.x, e.enemy.y) : NULL;
        bopl4_scene_head(&sh.s, ablation, self_x, self_y, e.vx, e.vy, (head->flags & BOPL4_STATE_AIMED) != 0, head->aim_dx, head->aim_dy, &e.enemy);
    } else if (t >= BR4_ENTITY_THREAD && t < BR4_ENTITY_THREAD + (int)E) {
        // The entity pass (bopl_encode_v4_row): kept, distance, and the counts.
        const int k = t - BR4_ENTITY_THREAD;
        const BoplRlEntity* entity = &entities[k];
        const int kept = bopl4_entity_kept(state, (uint32_t)k, self->id);
        e.kept[k] = (unsigned char)kept;
        if (kept) {
            double distance = bopl_surface_distance(entity, self_x, self_y);
            if (entity->flags & BOPL_ENTITY_PICKUP) distance -= 1.0e12;
            e.distance[k] = distance;
            if (distance != distance) e.nan = 1;
            atomicAdd(&e.c.active_count, 1);
            const uint32_t source = entity->reserved[0];
            if (source > 0 && source <= BOPL_ABILITY_COUNT) atomicAdd(&e.c.source_counts[source - 1], 1);
            if (entity->inverse_mass > 0) {
                atomicAdd(&e.c.n_nonstatic, 1);
                atomicAdd(&e.c.sectors[sectors[k]], 1);
            }
            if (entity->flags & BOPL_ENTITY_PICKUP) atomicAdd(&e.c.n_pickup, 1);
            if (state->header.player_time_scale == 0 && (entity->flags & BOPL_ENTITY_TIMER_KNOWN) && entity->clock_rate > 0)
                atomicMin(&e.time_stop, k);
        }
    }
    __syncthreads();
    if (t == 0) {
        // The slime pass, as written.
        int visible_slots;
        bopl4_slime_pass(state, self, self_x, self_y, e.controller, e.enemy_controller,
                         e.nearby_slimes, &e.slime_count, e.slot_indices, &e.slot_count, &e.c.visible_slimes, &visible_slots);
        e.c.count = e.c.active_count;
        if (e.time_stop < BOPL4_ENTITIES) { e.c.time_stop_known = 1; e.c.time_stop_remaining = entities[e.time_stop].reserved[1]; }
    } else if (t == 32) {
        // The scene's walls and water (bopl4_scene_fill), in entity order.
        Bopl4Scene* s = &sh.s;
        double water_top = -1e9; int has_water = 0;
        for (uint32_t k = 0; k < E; k++) {
            const int kind = e.kind[k];
            if (kind == BOPL4_SCENE_WATER) {
                if (!has_water || e.top[k] > water_top) water_top = e.top[k];
                has_water = 1;
                continue;
            }
            if (kind == BOPL4_SCENE_NONE || s->axis_count + s->rot_count >= BOPL4F_WALLS) continue;
            if (kind == BOPL4_SCENE_AXIS) bopl4_scene_axis(s, s->axis_count++, e.wall[k], e.lethal[k]);
            else bopl4_scene_rot(s, s->rot_count++, e.wall[k], e.lethal[k], e.skew[k]);
        }
        if (has_water) { s->flags |= BOPL4F_WATER; s->water_top = (float)water_top; } else s->water_top = -1e9f;
    } else if (t >= BR4_ENTITY_THREAD && t < BR4_ENTITY_THREAD + (int)E) {
        // The nearest: every kept entity, at its rank by (distance, index).
        const int k = t - BR4_ENTITY_THREAD;
        if (e.kept[k]) {
            const double d = e.distance[k];
            int rank = 0;
            for (int j = 0; j < (int)E; j++)
                if (e.kept[j] && (e.distance[j] < d || (e.distance[j] == d && j < k))) rank++;
            e.nearest[rank] = (uint32_t)k;
        }
    }
    __syncthreads();
    if (e.nan) {
        if (t == 0) {
            double distances[BOPL4_ENTITIES];
            int count = 0;
            for (uint32_t k = 0; k < E; k++) if (e.kept[k]) bopl_nearest(e.nearest, distances, &count, BOPL4_ENTITIES, k, e.distance[k]);
        }
        __syncthreads();
    }
    // The rows, a thread each; the unused set slots zeroed by everyone.
    const int count = e.c.count, slime_count = e.slime_count, slot_count = e.slot_count;
    const uint32_t player_count = state->header.player_count < BOPL4_PLAYERS ? state->header.player_count : BOPL4_PLAYERS;
    {
        const int sets[4][3] = {{BOPL4_OBS_PLAYER_OFFSET, BOPL4_PLAYER_FEATURES * (int)player_count, BOPL4_PLAYER_FEATURES * BOPL4_PLAYERS},
                                {BOPL4_OBS_SLOT_OFFSET, BOPL4_SLOT_FEATURES * slot_count, BOPL4_SLOT_FEATURES * BOPL4_SLOTS},
                                {BOPL4_OBS_ENTITY_OFFSET, BOPL4_ENTITY_FEATURES * count, BOPL4_ENTITY_FEATURES * BOPL4_ENTITIES},
                                {BOPL4_OBS_SLIME_OFFSET, BOPL4_SLIME_FEATURES * slime_count, BOPL4_SLIME_FEATURES * BOPL4_SLIMES}};
        for (int q = 0; q < 4; q++) for (int i = sets[q][1] + t; i < sets[q][2]; i += blockDim.x) out[sets[q][0] + i] = 0.0f;
    }
    const Bopl4Enemy* enemy = &e.enemy;
    const double svx = e.vx, svy = e.vy;
    if (t < BOPL4_ENTITIES) {
        if (t < count)
            bopl4_entity_row(out + BOPL4_OBS_ENTITY_OFFSET + (size_t)t * BOPL4_ENTITY_FEATURES, state, &entities[e.nearest[t]], self,
                             self_x, self_y, svx, svy, enemy, ablation);
    } else if (t >= 64 && t < 64 + slime_count) {
        const int i = t - 64;
        const BoplRlSlime* slime = &slimes[e.nearby_slimes[i]];
        const uint32_t j = e.nearby_slimes[i];
        bopl4_slime_row(out + BOPL4_OBS_SLIME_OFFSET + (size_t)i * BOPL4_SLIME_FEATURES, state, slime, self, self_x, self_y, svx, svy,
                        enemy, slime == e.controller || slime == e.enemy_controller, trig[2 * j], trig[2 * j + 1], ablation);
    } else if (t >= 96 && t < 96 + slot_count) {
        const int i = t - 96;
        bopl4_slot_row(out + BOPL4_OBS_SLOT_OFFSET + (size_t)i * BOPL4_SLOT_FEATURES, state, &slimes[e.slot_indices[i] / 3],
                       e.slot_indices[i] % 3, self, self_x, self_y, enemy);
    } else if (t >= 128 && t < 128 + (int)player_count) {
        const uint32_t i = (uint32_t)(t - 128), index = i == 0 ? (uint32_t)a : (i <= (uint32_t)a ? i - 1 : i);
        bopl4_player_row(out + BOPL4_OBS_PLAYER_OFFSET + (size_t)i * BOPL4_PLAYER_FEATURES, state, &state->players[index], self,
                         self_x, self_y, svx, svy, ablation);
    } else if (t >= 160 && t < 192) {
        bopl4_global_block(out, state, &e.c, head->max_episode_ticks, (head->flags & BOPL4_STATE_SPACE) != 0, e.controller, t - 160, 32);
    } else if (t == 192) {
        float* self_block = out + BOPL4_OBS_SELF_OFFSET;
        bopl_player_features(self_block, state, self, self, self_x, self_y);
        const int c = e.controller ? (int)(e.controller - slimes) : -1;
        bopl4_body(self_block + BOPL_PLAYER_FEATURES, e.controller, c < 0 ? 0.0f : trig[2 * c], c < 0 ? 0.0f : trig[2 * c + 1]);
    } else if (t == 224) {
        float opp[BOPL4_OPPONENT];
        const int c = enemy->slime ? (int)(enemy->slime - slimes) : -1;
        bopl4_opponent_block(opp, state, enemy, self, self_x, self_y, svx, svy,
                             c < 0 ? 0.0f : trig[2 * c], c < 0 ? 0.0f : trig[2 * c + 1], ablation);
        for (int i = 0; i < BOPL4_OPPONENT; i++) out[BOPL4_OBS_OPPONENT_OFFSET + i] = opp[i];
        if (head->flags & BOPL4_STATE_ENEMY) {
            out[BOPL4_OBS_PRIVILEGED_OFFSET + 43] = bopl4_stale_enemy(head->enemy_dx, opp[9]);
            out[BOPL4_OBS_PRIVILEGED_OFFSET + 44] = bopl4_stale_enemy(head->enemy_dy, opp[10]);
        }
    } else if (t > 192 && t < 224) {
        // The scene's circles (bopl4_scene_fill): the nearest entities', then the
        // nearby slimes', 31 candidates at a time, placed by a ballot's prefix.
        const int lane = t & 31, total = count + slime_count;
        const unsigned group = 0xfffffffeu;
        int base = 0;
        for (int i0 = 0; i0 < total; i0 += 31) {
            const int i = i0 + lane - 1;
            float c[5]; int32_t cls = 0; int has = 0;
            if (i < count) has = bopl4_entity_circle(&entities[e.nearest[i]], self, c, &cls);
            else if (i < total) has = bopl4_slime_circle(state, &slimes[e.nearby_slimes[i - count]], self, c, &cls);
            const unsigned ballot = __ballot_sync(group, has);
            if (has) bopl4_scene_circle(&sh.s, base + __popc(ballot & ((1u << lane) - 1u)), c, cls);
            base += __popc(ballot);
        }
        if (lane == 1) sh.s.circle_count = base;
    }
    __syncthreads();
}

// bopl4f_surfaces with the circles split across the ray's lanes; every lane
// gets the result.
__device__ static float br4_surfaces(const Br4Shared& sh, int lane, float dx, float dy, int* cls, float* closing) {
    const Bopl4Scene* s = &sh.s;
    const float x = s->x, y = s->y;
    float best = (float)BOPL4_RAY_LENGTH;
    *cls = -1; *closing = 0.0f;
    if ((s->flags & BOPL4F_WATER) && dy < -1e-9f) { float d = bopl4f_div(s->water_top - y, dy); if (d >= 0.0f && d < best) { best = d; *cls = 2; } }
    if (dy > 1e-9f) { float d = bopl4f_div(BOPL4F_BLAST_Y - y, dy); if (d >= 0.0f && d < best) { best = d; *cls = 2; } }
    if (dx > 1e-9f) { float d = bopl4f_div(BOPL4F_BLAST_X - x, dx); if (d >= 0.0f && d < best) { best = d; *cls = 2; } }
    if (dx < -1e-9f) { float d = bopl4f_div(-BOPL4F_BLAST_X - x, dx); if (d >= 0.0f && d < best) { best = d; *cls = 2; } }
    // The first circle at the least distance below the planes' best.
    float cbest = best;
    int win = -1;
    for (int i = lane; i < s->circle_count; i += BR4_LANES) {
        float px = s->cx[i] - x, py = s->cy[i] - y;
        float along = px * dx + py * dy;
        if (along < 0.0f) continue;
        float perp2 = (px * px + py * py) - along * along, r2 = s->cr[i] * s->cr[i];
        if (perp2 > r2) continue;
        float d = along - bopl4f_sqrt(r2 - perp2);
        d = d < 0.0f ? 0.0f : d;
        if (d < cbest) { cbest = d; win = i; }
    }
    const unsigned m = br4_ray_mask();
    #pragma unroll
    for (int o = 1; o < BR4_LANES; o <<= 1) {
        const float od = __shfl_xor_sync(m, cbest, o);
        const int ow = __shfl_xor_sync(m, win, o);
        if (ow >= 0 && (od < cbest || (od == cbest && (win < 0 || ow < win)))) { cbest = od; win = ow; }
    }
    if (win >= 0) {
        best = cbest; *cls = 3 + s->cclass[win];
        *closing = -((s->cvx[win] - s->svx) * dx + (s->cvy[win] - s->svy) * dy);
    }
    return best;
}

// bopl4f_lethal_at across the ray's lanes: the first strict minimum in wall order.
__device__ static int br4_lethal_at(const Br4Shared& sh, int lane, float x, float y) {
    const int na = sh.s.axis_count, n = na + sh.s.rot_count;
    float best = 1e9f;
    int at = -1;
    for (int j = lane; j < n; j += BR4_LANES) { const float d = br4_wall_sdf(sh, j, x, y); if (d < best) { best = d; at = j; } }
    const unsigned m = br4_ray_mask();
    #pragma unroll
    for (int o = 1; o < BR4_LANES; o <<= 1) {
        const float od = __shfl_xor_sync(m, best, o);
        const int oa = __shfl_xor_sync(m, at, o);
        if (oa >= 0 && (od < best || (od == best && (at < 0 || oa < at)))) { best = od; at = oa; }
    }
    if (at < 0) return 0;
    return at < na ? (int)((sh.s.axis_lethal >> at) & 1ull) : (int)((sh.s.rot_lethal >> (at - na)) & 1ull);
}

// bopl4f_finish by the ray's lanes, lane 0 writing.
__device__ static void br4_finish(const Br4Shared& sh, int lane, float dx, float dy, float best, int cls, float closing, float wall, float* out) {
    const Bopl4Scene* s = &sh.s;
    if (s->axis_count + s->rot_count && wall < (float)BOPL4_RAY_LENGTH && wall <= best) {
        best = wall; cls = br4_lethal_at(sh, lane, s->x + dx * wall, s->y + dy * wall); closing = 0.0f;
    }
    if (lane) return;
    out[0] = bopl4f_div(best, (float)BOPL4_RAY_LENGTH);
    for (int c = 0; c < 6; c++) out[1 + c] = cls == c;
    out[7] = cls >= 3 ? bopl4f_clip(bopl4f_div(closing, (float)BOPL4_VELOCITY_SCALE)) : 0.0f;
}

// bopl4f_march over group g's walls, each step's SDF split across the ray's lanes.
__device__ static float br4_march(const Br4Shared& sh, int g, int lane, float ox, float oy, float dx, float dy, float max, float limit) {
    const unsigned m = br4_ray_mask();
    const unsigned char* list = sh.w.list[g];
    const int n = sh.listn[g];
    float t = 0.0f;
    for (int i = 0; i < 32; i++) {
        if (!(t < max)) return max;
        if (t >= limit) return limit;
        const float px = ox + dx * t, py = oy + dy * t;
        float d = 1e9f;
        for (int j = lane; j < n; j += BR4_LANES) { const float w = br4_wall_sdf(sh, list[j], px, py); d = w < d ? w : d; }
        #pragma unroll
        for (int o = 1; o < BR4_LANES; o <<= 1) { const float e = __shfl_xor_sync(m, d, o); d = e < d ? e : d; }
        if (d < 0.05f) return t;
        t += d;
    }
    return max;
}

// bopl4f_march_group: called by all `threads` threads of group g (barrier g),
// gt the thread's index in it; the ray of an active thread is marched with
// the walls culled at the group's reach.
__device__ static float br4_group(Br4Shared& sh, int g, int threads, int gt, bool active,
                                  float ox, float oy, float dx, float dy, float max, float limit) {
    const Bopl4Scene* s = &sh.s;
    if (!(s->axis_count + s->rot_count)) return max;
    const int lane = gt % BR4_LANES;
    if (active && lane == 0) {
        const float r = bopl4f_reach(s, ox, oy, max, limit);
        if (r > 0.0f) atomicMax(&sh.reach[g], __float_as_uint(r));   // reach=r>reach?r:reach from 0
    }
    br4_bar(g, threads);
    if (gt == 0) {
        const Bopl4Mask m = bopl4f_cull(s, &sh.a, __uint_as_float(sh.reach[g]) + 0.1f);
        int n = 0;
        for (uint64_t b = m.axis; b; b &= b - 1ull) sh.w.list[g][n++] = (unsigned char)bopl4f_ctz(b);
        for (uint64_t b = m.rot; b; b &= b - 1ull) sh.w.list[g][n++] = (unsigned char)(s->axis_count + bopl4f_ctz(b));
        sh.listn[g] = n;
    }
    br4_bar(g, threads);
    return active ? br4_march(sh, g, lane, ox, oy, dx, dy, max, limit) : 0.0f;
}

// The aim and line-of-sight casts (bopl4f_aim_rays) by warp 7, into out[2][8].
__device__ static void br4_aim_rays(Br4Shared& sh, int gt, float (*out)[BOPL4_RAY_FEATURES]) {
    const Bopl4Scene* s = &sh.s;
    const int ray = gt / BR4_LANES, lane = gt % BR4_LANES;
    float dx = 0.0f, dy = 0.0f;
    bool active = false;
    if (ray == 0 && (s->flags & BOPL4F_AIMED)) { dx = s->aim_dx; dy = s->aim_dy; active = true; }
    if (ray == 1) active = bopl4f_los(s, &dx, &dy);
    int cls = -1;
    float closing = 0.0f, best = 0.0f;
    if (active) best = br4_surfaces(sh, lane, dx, dy, &cls, &closing);
    const float wall = br4_group(sh, BR4_AIM, 32, gt, active, s->x, s->y, dx, dy, (float)BOPL4_RAY_LENGTH, best + 1.0f);
    if (active) br4_finish(sh, lane, dx, dy, best, cls, closing, wall, out[ray]);
}

__global__ void __launch_bounds__(BR4_RAYS_THREADS) bopl4_rays_rows(const float* __restrict__ packed, float* __restrict__ dense, int rows) {
    __shared__ Br4Shared sh;
    const int r = blockIdx.x;
    if (r >= rows) return;
    const float* slot = packed + (long)r * BOPL4_PACK_P;
    const unsigned* hdr = (const unsigned*)slot;
    const unsigned mode = hdr[0];
    if (mode != BOPL4_PACK_MODE_SCENE && mode != BOPL4_PACK_MODE_DENSE_SCENE && mode != BOPL4_PACK_MODE_STATE) return;
    float* region = dense + (long)r * BOPL_OBS_STRIDE + BOPL4_OBS_TERRAIN_OFFSET;
    float* rays = region + BOPL4_TERRAIN;
    float* aim = rays + BOPL4_AIM_RAY * BOPL4_RAY_FEATURES;
    float* los = rays + BOPL4_LOS_RAY * BOPL4_RAY_FEATURES;
    const int i = threadIdx.x;
    if (i < 2 * BOPL4_RAY_FEATURES) sh.aim_out[i / BOPL4_RAY_FEATURES][i % BOPL4_RAY_FEATURES] = 0.0f;
    if (mode == BOPL4_PACK_MODE_STATE) br4_encode(slot, dense + (long)r * BOPL_OBS_STRIDE, sh);
    else br4_scene_load(slot, (int)hdr[5], sh);
    br4_scene_prepare(sh);
    const Bopl4Scene& s = sh.s;
    if (!(s.flags & BOPL4F_FAN)) {
        for (int k = i; k < BOPL4_TERRAIN + BOPL4_RAYS * BOPL4_RAY_FEATURES; k += blockDim.x) region[k] = 0.0f;
        if (!(s.flags & BOPL4F_AIM)) return;
        if (i >= BR4_MISC) br4_aim_rays(sh, i - BR4_MISC, sh.aim_out);
        __syncthreads();
        if (i < 2 * BOPL4_RAY_FEATURES) (i < BOPL4_RAY_FEATURES ? aim : los)[i % BOPL4_RAY_FEATURES] = sh.aim_out[i / BOPL4_RAY_FEATURES][i % BOPL4_RAY_FEATURES];
        return;
    }
    const float x = s.x, y = s.y, wt = s.water_top;
    const int walls = s.axis_count + s.rot_count, water = (s.flags & BOPL4F_WATER) != 0;

    if (i < BR4_FAN_THREADS) {
        // The fan (every other around ray under BOPL4F_HALF), then the drop and ceiling probes.
        const int half = (s.flags & BOPL4F_HALF) != 0, n = half ? BOPL4_RAYS_AROUND / 2 : BOPL4_RAYS;
        if (half) for (int k = i; k < BOPL4_RAYS * BOPL4_RAY_FEATURES; k += BR4_FAN_THREADS) {
            const int ray = k / BOPL4_RAY_FEATURES;
            if (ray >= BOPL4_RAYS_AROUND || (ray & 1)) rays[k] = 0.0f;
        }
        const int ray = i / BR4_LANES, lane = i % BR4_LANES, k = half ? 2 * ray : ray;
        float dx = 0.0f, dy = 0.0f, best = 0.0f, closing = 0.0f, max = 0.0f, limit = 0.0f;
        int cls = -1;
        if (ray < n) {
            dx = br4_dirs.fan[k][0]; dy = br4_dirs.fan[k][1];
            best = br4_surfaces(sh, lane, dx, dy, &cls, &closing);
            limit = best + 1.0f; max = (float)BOPL4_RAY_LENGTH;
        } else if (ray < n + 2) {
            dy = ray == n ? -1.0f : 1.0f; limit = max = 32.0f;
        }
        const float wall = br4_group(sh, BR4_FAN, BR4_FAN_THREADS, i, ray < n + 2, x, y, dx, dy, max, limit);
        if (lane == 0 && ray == n) sh.probes[0] = wall;
        if (lane == 0 && ray == n + 1) sh.probes[1] = wall;
        if (ray < n) br4_finish(sh, lane, dx, dy, best, cls, closing, wall, rays + k * BOPL4_RAY_FEATURES);
    } else if (i < BR4_MISC) {
        // Ground within one dash (bopl4f_terrain), needed when the drop probe is over 4.
        const int gt = i - BR4_FAN_THREADS, ray = gt / BR4_LANES;
        const float dx = br4_dirs.dash[ray][0], dy = br4_dirs.dash[ray][1];
        const float d = br4_group(sh, BR4_DASH, BR4_DASH_THREADS, gt, true, x, y, dx, dy, (float)BOPL4_DASH_LENGTH, (float)BOPL4_DASH_LENGTH);
        const float px = x + dx * (d - 0.5f), py = y + dy * (d - 0.5f);
        bool fall = false, wet = false;
        if (!(water && py <= wt)) {
            if (water && !(py - wt > 4.0f)) wet = true;
            else if (walls) fall = true;
        }
        int landing = br4_bar_or(BR4_DASH, BR4_DASH_THREADS, wet);
        if (!landing && br4_bar_or(BR4_DASH, BR4_DASH_THREADS, fall)) {
            const float f = br4_group(sh, BR4_FALL, BR4_DASH_THREADS, gt, fall, px, py, 0.0f, -1.0f, 8.0f, bopl4f_next_up(4.0f));
            landing = br4_bar_or(BR4_DASH, BR4_DASH_THREADS, fall && !(f > 4.0f));
        }
        if (gt == 0) sh.dash_landing = landing;
    } else {
        // The single SDFs, then the aim and line of sight.
        const int gt = i - BR4_MISC;
        if (gt >= 8 && gt < 24) {
            const int e = gt - 8, side = e / 8, k = e % 8 + 1;
            sh.hit[e] = walls && bopl4f_sdf(&s, bopl4f_all(&s), x + (side ? 1.5f : -1.5f) * (float)k, y) < 0.3f;
        } else if (gt >= 24 && gt < 29 && walls) {
            const int e = gt - 24;
            const float h = 0.05f;
            const float px = e == 1 ? x + h : e == 2 ? x - h : x, py = e == 3 ? y + h : e == 4 ? y - h : y;
            sh.sdf_at[e] = bopl4f_sdf(&s, bopl4f_all(&s), px, py);
        } else if (gt == 29) {
            float lethal = 16.0f;
            const Bopl4Mask lethal_walls = {s.axis_lethal, s.rot_lethal};
            if (lethal_walls.axis | lethal_walls.rot) { const float d = bopl4f_sdf(&s, lethal_walls, x, y); lethal = d < lethal ? d : lethal; }
            sh.lethal = lethal;
        }
        if (s.flags & BOPL4F_AIM) br4_aim_rays(sh, gt, sh.aim_out);
    }
    __syncthreads();

    // The terrain block (bopl4f_terrain) from the probes, then the aim rays over the fan's slots.
    float drop = sh.probes[0];
    if (water && y > wt && y - wt < drop) drop = y - wt;
    int first_wall[2] = {9, 9}, first_drop[2] = {9, 9};
    for (int side = 0; side < 2; side++)
        for (int k = 1; k <= 8; k++) if (sh.hit[side * 8 + k - 1]) { first_wall[side] = k; break; }
    const float threshold = drop + 4.0f;
    const bool edges = !(water && y > wt && !(y - wt > threshold)) && walls;
    if (!(water && y > wt && !(y - wt > threshold)) && !walls && 32.0f > threshold) first_drop[0] = first_drop[1] = 1;
    if (edges && i < BR4_EDGE_THREADS) {
        // Ray e: side 0 steps 1 .. first_wall[0]-1, then side 1's.
        const int e = i / BR4_LANES, left = first_wall[0] - 1, n = left + first_wall[1] - 1;
        const int side = e < left ? 0 : 1, k = side ? e - left + 1 : e + 1;
        const float d = br4_group(sh, BR4_EDGE, BR4_EDGE_THREADS, i, e < n, x + (side ? 1.5f : -1.5f) * (float)k, y, 0.0f, -1.0f,
                                  32.0f, bopl4f_next_up(threshold));
        if (i % BR4_LANES == 0) sh.edge_drop[e] = e < n && d > threshold;
        br4_bar(BR4_EDGE, BR4_EDGE_THREADS);
    }
    if (i == 0) {
        if (edges) {
            const int left = first_wall[0] - 1, n = left + first_wall[1] - 1;
            for (int j = 0; j < n; j++) if (sh.edge_drop[j]) {
                const int sj = j < left ? 0 : 1, kj = sj ? j - left + 1 : j + 1;
                if (kj < first_drop[sj]) first_drop[sj] = kj;
            }
        }
        float* out = region;
        for (int side = 0; side < 2; side++) {
            const int k = first_wall[side] < first_drop[side] ? first_wall[side] : first_drop[side];
            out[2 + side] = bopl4f_div(k <= 8 ? 1.5f * (float)k : (float)BOPL4_DASH_LENGTH, (float)BOPL4_DASH_LENGTH);
            out[4 + side] = first_wall[side] <= 8 && first_wall[side] <= first_drop[side];
        }
        out[0] = bopl4f_div(drop, 32.0f); out[1] = bopl4f_div(sh.probes[1], 32.0f);
        out[6] = water ? bopl4f_clip(bopl4f_div(y - wt, 32.0f)) : 1.0f;
        out[7] = bopl4f_div(sh.lethal, 16.0f);
        out[8] = drop <= 4.0f || sh.dash_landing;
        out[9] = walls ? bopl4f_clip(bopl4f_div(sh.sdf_at[0], 8.0f)) : 1.0f;
        const float h = 0.05f;
        out[10] = walls ? bopl4f_clip(bopl4f_div(sh.sdf_at[1] - sh.sdf_at[2], 2.0f * h)) : 0.0f;
        out[11] = walls ? bopl4f_clip(bopl4f_div(sh.sdf_at[3] - sh.sdf_at[4], 2.0f * h)) : 0.0f;
    }
    if ((s.flags & BOPL4F_AIM) && i >= BR4_MISC && i < BR4_MISC + 2 * BOPL4_RAY_FEATURES) {
        const int e = i - BR4_MISC;
        (e < BOPL4_RAY_FEATURES ? aim : los)[e % BOPL4_RAY_FEATURES] = sh.aim_out[e / BOPL4_RAY_FEATURES][e % BOPL4_RAY_FEATURES];
    }
}

// The directions, once before any launch (a stream capture cannot copy to a symbol).
extern "C" void bopl4_rays_init(void) {
    Bopl4Dirs dirs;
    bopl4f_dirs_init(&dirs);
    if (cudaMemcpyToSymbol(br4_dirs, &dirs, sizeof(dirs)) != cudaSuccess) { fprintf(stderr, "bopl4_rays_init failed\n"); abort(); }
}
// After bopl4_unpack_rows on the same rows: the terrain and rays of the rows with a scene.
extern "C" void bopl4_rays_launch(const float* packed, float* dense, int rows, cudaStream_t stream) {
    bopl4_rays_rows<<<rows, BR4_RAYS_THREADS, 0, stream>>>(packed, dense, rows);
}
