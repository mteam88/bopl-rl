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
//   warps 0-4  the fan with the drop and ceiling probes (the probes count
//              toward its reach; warp 7 marches them)
//   warps 5-6  the dash rays, then their falls (whether or not the drop
//              probe leaves them needed: speculative), then, given the drop
//              probe, the edge probes and the terrain block
//   warp 7     the single SDFs (edge walls, the body's SDF and gradient, the
//              lethal distance), the drop and ceiling probes (16 lanes a
//              probe) once the fan's reach is in, then the aim and line of
//              sight
// (the edge probes ran after a join of the whole block, behind the fan's
// slowest warp, which no longer waits for them). Exact: a group's
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
constexpr int BR4_MISC = BR4_FAN_THREADS + BR4_DASH_THREADS;      // warp 7's first thread
constexpr int BR4_RAYS_THREADS = BR4_MISC + 32;
constexpr int BR4_WALLS = 2 * BOPL4F_WALLS;
static_assert(BR4_FAN_SLOTS >= BOPL4F_GROUP, "a slot per ray of the fan");
static_assert(BR4_FAN_THREADS % 32 == 0 && BR4_DASH_THREADS % 32 == 0, "groups are whole warps");
static_assert(BR4_DASH_THREADS == 16 * BR4_LANES, "the 16 edge probes on the dash warps");
static_assert(32 % BR4_LANES == 0 && 2 * BR4_LANES <= 8, "a ray's lanes in one warp; the aim's two rays in lanes 0-7");
// Named barriers (0 is __syncthreads) and reach slots, one per march group.
enum { BR4_FAN = 1, BR4_DASH = 2, BR4_FALL = 3, BR4_AIM = 4, BR4_EDGE = 5, BR4_GROUPS = 6 };

__constant__ Bopl4Dirs br4_dirs;

struct Br4Walls {
    // Every wall as a rotated one (aligned: right (1, 0), up (0, 1); the same
    // bits, see br4_wall_sdf), in the order of their cull keys: a wall's SDF
    // at the body, -inf for one every group keeps (a skewed wall; a NaN SDF,
    // which bopl4f_cull keeps too). A group's culled walls (bopl4f_cull: the
    // keys at or below its reach) are then the first n, and the march's
    // walls are contiguous; past the walls the keys are +inf.
    float4 c[BR4_WALLS], u[BR4_WALLS];          // (cx, cy, rx, ry), (ux, uy, ex, ey)
    float radius[BR4_WALLS], key[BR4_WALLS];
    unsigned char index[BR4_WALLS];             // the wall's index in lethal_at's order (aligned walls first)
};
struct Br4Shared {
    // The slot from the scene on (modes 2, 3), one coalesced read; then the walls' keys unsorted.
    float raw[BOPL4_PACK_P - BOPL4_PACK_HEADER];
    Bopl4Scene s;
    Br4Walls w;
    // Per circle what no ray changes: its offset from the body (px, py), px * px + py * py and its radius squared,
    // padded for the surfaces' unrolled reads.
    float4 circle[BOPL4F_CIRCLES + 4 * BR4_LANES];
    unsigned reach[BR4_GROUPS];                 // a group's reach, as the bits of a non-negative float
    unsigned fan_cull;                          // the fan's cull bound (its reach + 0.1, positive) once final; 0 before
    float probes[2], sdf_at[5], lethal, aim_out[2][BOPL4_RAY_FEATURES];
    int hit[16], edge_drop[16], dash_landing;
};
static_assert(sizeof(Br4Shared) <= 48 * 1024, "static shared memory");
static_assert(BR4_RAYS_THREADS >= BR4_WALLS && BR4_WALLS % 32 == 0, "a thread per wall; the keys in whole warps");
static_assert(offsetof(Bopl4StateHead, words) == BOPL4_PACK_STATE_WORDS * 4, "the unpack reads a mode-4 slot's length there");

__device__ __forceinline__ void br4_bar(int id, int threads) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory"); }
__device__ __forceinline__ void br4_arrive(int id, int threads) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(threads) : "memory"); }
__device__ __forceinline__ int br4_bar_or(int id, int threads, int p) {
    int out;
    asm volatile("{ .reg .pred q, r; setp.ne.s32 q, %1, 0; bar.red.or.pred r, %2, %3, q; selp.s32 %0, 1, 0, r; }"
                 : "=r"(out) : "r"(p), "r"(id), "r"(threads) : "memory");
    return out;
}
// The shuffle mask of this thread's ray (its L lanes).
template <int L = BR4_LANES>
__device__ __forceinline__ unsigned br4_ray_mask() {
    return ((1u << L) - 1u) << ((threadIdx.x & 31) & ~(L - 1));
}

// bopl4f_rot_sdf of unified wall j. For an aligned wall px*1 + py*0 is px
// (+-0 added; fabsf drops the sign of a zero) and so equals bopl4f_axis_sdf.
__device__ __forceinline__ float br4_wall_sdf(const Br4Shared& sh, int j, float x, float y) {
    const float4 c = sh.w.c[j], u = sh.w.u[j];
    const float px = x - c.x, py = y - c.y;
    const float lx = px * c.z + py * c.w, ly = px * u.x + py * u.y;
    return bopl4f_box(fabsf(lx) - u.z, fabsf(ly) - u.w, sh.w.radius[j]);
}

// The scene from its packed form (bopl4f_scene_write), by the whole block:
// the slot's words from the scene to its end copied in one pass (one memory
// round trip: every load in flight before any store; a load and its store an
// iteration took eight round trips, ~3 us), then taken apart in shared memory.
__device__ static void br4_scene_load(const float* __restrict__ slot, int at, Br4Shared& sh) {
    constexpr int T = BR4_RAYS_THREADS, NL = (BOPL4_PACK_P - BOPL4_PACK_HEADER + T - 1) / T;
    float v[NL];
    #pragma unroll
    for (int t = 0; t < NL; t++) if (t * T + (int)threadIdx.x < BOPL4_PACK_P - at) v[t] = slot[at + t * T + threadIdx.x];
    #pragma unroll
    for (int t = 0; t < NL; t++) if (t * T + (int)threadIdx.x < BOPL4_PACK_P - at) sh.raw[t * T + threadIdx.x] = v[t];
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
// The walls unified for the marches, sorted by their cull keys (their SDFs
// at the body, the anchors): thread j wall j, its rank among the keys, ties
// by index. The circles' terms of bopl4f_surfaces that no ray changes.
__device__ static void br4_scene_prepare(Br4Shared& sh) {
    const Bopl4Scene* s = &sh.s;
    const int na = s->axis_count, n = na + s->rot_count, j = threadIdx.x;
    const float inf = __int_as_float(0x7f800000);
    if (j < BR4_GROUPS) sh.reach[j] = 0u;
    if (j == BR4_GROUPS) sh.fan_cull = 0u;
    for (int i = j; i < s->circle_count; i += blockDim.x) {
        const float px = s->cx[i] - s->x, py = s->cy[i] - s->y;
        sh.circle[i] = make_float4(px, py, px * px + py * py, s->cr[i] * s->cr[i]);
    }
    float key = inf;
    if (j < na) {
        const float a = bopl4f_axis_sdf(s, j, s->x, s->y);
        key = a != a ? -inf : a;
    } else if (j < n) {
        const int k = j - na;
        const float a = bopl4f_rot_sdf(s, k, s->x, s->y);
        key = ((s->rot_skew >> k) & 1ull) || a != a ? -inf : a;
    }
    if (j < n) sh.raw[j] = key;
    else if (j < BR4_WALLS) sh.w.key[j] = inf;
    __syncthreads();
    if (j < n) {
        int rank = 0;
        for (int k = 0; k < n; k++) { const float o = sh.raw[k]; rank += o < key || (o == key && k < j); }
        float4 c, u;
        float radius;
        if (j < na) {
            c = make_float4(s->axis_cx[j], s->axis_cy[j], 1.0f, 0.0f); u = make_float4(0.0f, 1.0f, s->axis_ex[j], s->axis_ey[j]);
            radius = s->axis_radius[j];
        } else {
            const int k = j - na;
            c = make_float4(s->rot_cx[k], s->rot_cy[k], s->rot_rx[k], s->rot_ry[k]); u = make_float4(s->rot_ux[k], s->rot_uy[k], s->rot_ex[k], s->rot_ey[k]);
            radius = s->rot_radius[k];
        }
        sh.w.c[rank] = c; sh.w.u[rank] = u; sh.w.radius[rank] = radius; sh.w.key[rank] = key; sh.w.index[rank] = (unsigned char)j;
    }
    __syncthreads();
}

// ---- The row from its state (pack mode 4): bopl4_encode_rows ----
// bopl_encode_v4_row's passes for BR4E_ROWS rows per block, every feature
// from the functions the CPU writer calls (bopl_rows_v4.h), so the row is its
// bits. A GPU thread runs its instructions in order at tens of cycles each
// (FP64 at 1/64 rate: a dependent DFMA is ~63 cycles, a division or square
// root ~650), so the work is cut into short independent items over many
// threads, one kind of item per warp (a warp runs one function):
//   S   the scene's own threads, beside P1 and P2 (only P3 reads it): per
//       (row, entity, component) bopl4_scene_entity's steps, a wall's axis
//       lengths and unit vectors a square root and a quotient a thread,
//       then the wall in the entity's first thread
//   P1  the other threads, per (row, entity): the circle; per (row,
//       entity): kept, the distance and the counts (integers: any
//       order sums alike); per (row, slime): its circle and what the slime
//       pass needs of it (bopl4_slime_seen); per (row, player): its
//       controlling body and the player as the primary enemy's candidate
//       (a named barrier among these threads ends it)
//   P2  per row (a lane each): the observer, its body, the primary enemy, the
//       scene's head and the slime pass, from P1's; per (row, entity): the
//       nearest, every kept entity at its rank by (distance, index), which is
//       bopl_nearest's stable order (there are at most BOPL4_ENTITIES),
//       unless a distance is NaN, when the row runs bopl_nearest as written
//   P3  (every thread) a warp per row places the scene's circles (the nearest entities',
//       then the nearby slimes', by a ballot's prefix), a warp per row its
//       walls (the first BOPL4F_WALLS candidates in entity order, likewise)
//       and water; the unused set slots zeroed; then what the row items
//       (br4_items) read goes to global memory.
// The previous and critic blocks come from the slot as the host wrote them,
// except critic 43 and 44, which read the opponent block. The scene goes to
// the row's Bopl4Scene in global memory for the rays.
// The scene's threads (four an entity: a square root and a quotient each) and the others.
constexpr int BR4E_ROWS = 1, BR4E_SCENE = 4 * BOPL4_ENTITIES * BR4E_ROWS, BR4E_REST = 128, BR4E_THREADS = BR4E_SCENE + BR4E_REST;
constexpr int BR4E_SEGS = 7, BR4E_KINDS = 4;
static_assert(BR4E_SCENE % 32 == 0, "the scene's warps are whole");
static_assert(BR4E_REST >= 3 * 32 && BR4E_ROWS <= 32, "P2's warps");
// What the row items (br4_items) read of a row, left in global memory by
// bopl4_encode_rows; enemy.player points into this st.
struct Br4Meta {
    BoplRlState st;                             // header and players; entities and slimes point into the slot
    Bopl4Enemy enemy;
    const BoplRlSlime *controller, *enemy_controller;
    const float* trig;                          // sinf, cosf per slime
    float* out;
    double x, y, vx, vy, enemy_dx, enemy_dy;
    Bopl4Counts c;
    int a, ablation, flags, max_ticks, slime_count, slot_count;
    uint32_t nearest[BOPL4_ENTITIES], nearby[BOPL4_SLIMES], slots[BOPL4_SLOTS];
};
static_assert(sizeof(Br4Meta) % 8 == 0, "copied by words");
struct Br4ERow : Br4Meta {
    const unsigned char* sectors;               // per entity
    Bopl4Scene* scene;
    int E, S, P, time_stop, nan;
    float aim_dx, aim_dy;
    const BoplRlSlime* bodies[BOPL4_PLAYERS];   // per player: bopl_controlling_slime
    Bopl4Enemy candidate[BOPL4_PLAYERS];        // per player: bopl4_enemy_candidate
    double candidate_distance[BOPL4_PLAYERS], body_vx[BOPL4_PLAYERS], body_vy[BOPL4_PLAYERS];
    int is_candidate[BOPL4_PLAYERS];
    Bopl4SlimeSeen seen[BOPL4_SLIMES];
    double distance[BOPL4_ENTITIES], top[BOPL4_ENTITIES];
    float wall[BOPL4_ENTITIES][9], ecirc[BOPL4_ENTITIES][5], scirc[BOPL4_SLIMES][5];
    int32_t ecls[BOPL4_ENTITIES], scls[BOPL4_SLIMES];
    unsigned char kept[BOPL4_ENTITIES], kind[BOPL4_ENTITIES], lethal[BOPL4_ENTITIES], skew[BOPL4_ENTITIES], ehas[BOPL4_ENTITIES],
                  shas[BOPL4_SLIMES];
};
struct Br4EShared {
    Br4ERow row[BR4E_ROWS];
    int valid[BR4E_ROWS];
    int off[BR4E_KINDS * BR4E_ROWS + 1], cnt[BR4E_KINDS * BR4E_ROWS];   // P1's items: kind-major, then row; off[last] the end
};
static_assert(sizeof(Br4EShared) <= 48 * 1024, "static shared memory");
static_assert(BOPL4_PLAYERS == 6, "BoplRlState holds six players");

// The order of a non-NaN double as an integer, -0 as +0 (they compare equal).
__device__ __forceinline__ unsigned long long br4e_key(double d) {
    unsigned long long u = (unsigned long long)__double_as_longlong(d);
    if (u == 0x8000000000000000ull) u = 0ull;
    return (u >> 63) ? ~u : u | 0x8000000000000000ull;
}
// The OR of v over the warp.
__device__ __forceinline__ unsigned long long br4e_or(unsigned long long v) {
    #pragma unroll
    for (int o = 16; o; o >>= 1) v |= __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__global__ void __launch_bounds__(BR4E_THREADS, 3) bopl4_encode_rows(const float* __restrict__ packed, float* __restrict__ dense, int rows,
                                                                Bopl4Scene* __restrict__ scenes, Br4Meta* __restrict__ metas,
                                                                int* __restrict__ counts) {
    __shared__ Br4EShared sh;
    const int t = threadIdx.x, r0 = blockIdx.x * BR4E_ROWS, warp = t >> 5, lane = t & 31;
    // P0: the slots' heads, and P1's items.
    if (t < BR4E_ROWS) {
        const int r = r0 + t;
        Br4ERow& w = sh.row[t];
        const float* slot = packed + (long)r * BOPL4_PACK_P;
        const int valid = r < rows && ((const unsigned*)slot)[0] == BOPL4_PACK_MODE_STATE;
        sh.valid[t] = valid;
        w.E = w.S = w.P = 0;
        if (valid) {
            const Bopl4StateHead* head = (const Bopl4StateHead*)slot;
            w.a = (int)head->player; w.ablation = (int)head->ablation; w.flags = (int)head->flags; w.max_ticks = (int)head->max_episode_ticks;
            w.E = (int)head->entity_count; w.S = (int)head->slime_count;
            w.aim_dx = head->aim_dx; w.aim_dy = head->aim_dy; w.enemy_dx = head->enemy_dx; w.enemy_dy = head->enemy_dy;
            const BoplRlHeader* header = (const BoplRlHeader*)(slot + BOPL4_STATE_OFFSET);
            w.P = header->player_count < BOPL4_PLAYERS ? (int)header->player_count : BOPL4_PLAYERS;
            const BoplRlEntity* entities = (const BoplRlEntity*)(slot + BOPL4_STATE_OFFSET + BOPL4_STATE_BASE_WORDS);
            const BoplRlSlime* slimes = (const BoplRlSlime*)(entities + w.E);
            w.st.entities = entities; w.st.slimes = slimes;
            w.trig = (const float*)(slimes + w.S);
            w.sectors = (const unsigned char*)(w.trig + 2 * w.S);
            w.out = dense + (long)r * BOPL_OBS_STRIDE;
            w.scene = scenes + r;
            for (int i = 0; i < (int)(sizeof(w.c) / 4); i++) ((int32_t*)&w.c)[i] = 0;   // words: memset would store bytes
            w.time_stop = BOPL4_ENTITIES; w.nan = 0;
        }
    }
    __syncthreads();
    if (t == 0) {
        int n = 0;
        for (int q = 0; q < BR4E_KINDS; q++) {
            n = (n + 31) & ~31;
            for (int lr = 0; lr < BR4E_ROWS; lr++) {
                const Br4ERow& w = sh.row[lr];
                const int c = q < 2 ? w.E : q == 2 ? w.S : w.P;
                sh.off[q * BR4E_ROWS + lr] = n; sh.cnt[q * BR4E_ROWS + lr] = c;
                n += c;
            }
        }
        sh.off[BR4E_KINDS * BR4E_ROWS] = n;
    }
    // The header and players; the previous and critic blocks as the host wrote them.
    for (int i = t; i < BR4E_ROWS * BOPL4_STATE_BASE_WORDS; i += BR4E_THREADS) {
        const int lr = i / BOPL4_STATE_BASE_WORDS, j = i % BOPL4_STATE_BASE_WORDS;
        if (sh.valid[lr]) ((float*)&sh.row[lr].st)[j] = packed[(long)(r0 + lr) * BOPL4_PACK_P + BOPL4_STATE_OFFSET + j];
    }
    constexpr int FIXED = BOPL4_PREVIOUS + BOPL4_PRIVILEGED;
    for (int i = t; i < BR4E_ROWS * FIXED; i += BR4E_THREADS) {
        const int lr = i / FIXED, j = i % FIXED;
        if (sh.valid[lr])
            sh.row[lr].out[j < BOPL4_PREVIOUS ? BOPL4_OBS_PREVIOUS_OFFSET + j : BOPL4_OBS_PRIVILEGED_OFFSET + j - BOPL4_PREVIOUS] =
                packed[(long)(r0 + lr) * BOPL4_PACK_P + BOPL4_STATE_FIXED_OFFSET + j];
    }
    __syncthreads();

    if (t < BR4E_SCENE) {
        // Entity k's part of the scene (walls and water), beside P1 and P2 (only P3 reads it): bopl4_scene_entity's
        // steps, a wall's axis lengths and unit vectors over four threads, component j each, then the wall in the first.
        const int lr = t / (4 * BOPL4_ENTITIES), k = t % (4 * BOPL4_ENTITIES) >> 2, j = t & 3;
        Br4ERow& w = sh.row[lr];
        const int on = sh.valid[lr] && k < w.E;
        const BoplRlEntity* e = on ? &w.st.entities[k] : nullptr;
        int kind = BOPL4_SCENE_NONE;
        double top = 0.0, length = 0.0, unit = 0.0;
        if (on && (kind = bopl4_scene_class(e, &top)) == BOPL4_SCENE_WALL) {
            const double x = bopl_fix(j < 2 ? e->right_x : e->up_x), y = bopl_fix(j < 2 ? e->right_y : e->up_y);
            length = bopl4_axis_length(x, y);
            if (length > 1e-6) unit = (j & 1 ? y : x) / length;
        }
        const double u1 = __shfl_down_sync(0xffffffffu, unit, 1), u2 = __shfl_down_sync(0xffffffffu, unit, 2);
        const double u3 = __shfl_down_sync(0xffffffffu, unit, 3), ey = __shfl_down_sync(0xffffffffu, length, 2);
        if (on && j == 0) {
            int lethal = 0, skew = 0;
            if (kind == BOPL4_SCENE_WALL) {
                const double ex = length;
                double rx, ry, ux, uy;
                if (ex > 1e-6) { rx = unit; ry = u1; } else { rx = 1; ry = 0; }
                if (ey > 1e-6) { ux = u2; uy = u3; } else { ux = -ry; uy = rx; }
                kind = bopl4_scene_wall(e, rx, ry, ux, uy, ex, ey, w.wall[k], &lethal, &skew);
            }
            w.kind[k] = (unsigned char)kind; w.top[k] = top; w.lethal[k] = (unsigned char)lethal; w.skew[k] = (unsigned char)skew;
        }
    } else {
        const int u = t - BR4E_SCENE, uwarp = u >> 5;   // the thread and warp among the others
        // P1
        for (int i = u; i < sh.off[BR4E_KINDS * BR4E_ROWS]; i += BR4E_REST) {
            int at = 0;
            while (sh.off[at + 1] <= i) at++;
            const int q = at / BR4E_ROWS, k = i - sh.off[at];
            if (k >= sh.cnt[at]) continue;   // the padding to the next kind's warp
            Br4ERow& w = sh.row[at % BR4E_ROWS];
            const BoplRlState* state = &w.st;
            const BoplRlPlayer* self = &state->players[w.a];
            if (q == 0) {
                // Entity k's circle.
                const BoplRlEntity* entity = &state->entities[k];
                int32_t cls = 0;
                w.ehas[k] = (unsigned char)bopl4_entity_circle(entity, self, w.ecirc[k], &cls);
                w.ecls[k] = cls;
            } else if (q == 1) {
                // The entity pass.
                const BoplRlEntity* entity = &state->entities[k];
                const int kept = bopl4_entity_kept(state, (uint32_t)k, self->id);
                w.kept[k] = (unsigned char)kept;
                if (kept) {
                    double distance = bopl_surface_distance(entity, bopl_fix(self->x), bopl_fix(self->y));
                    if (entity->flags & BOPL_ENTITY_PICKUP) distance -= 1.0e12;
                    w.distance[k] = distance;
                    if (distance != distance) w.nan = 1;
                    atomicAdd(&w.c.active_count, 1);
                    const uint32_t source = entity->reserved[0];
                    if (source > 0 && source <= BOPL_ABILITY_COUNT) atomicAdd(&w.c.source_counts[source - 1], 1);
                    if (entity->inverse_mass > 0) {
                        atomicAdd(&w.c.n_nonstatic, 1);
                        atomicAdd(&w.c.sectors[w.sectors[k]], 1);
                    }
                    if (entity->flags & BOPL_ENTITY_PICKUP) atomicAdd(&w.c.n_pickup, 1);
                    if (state->header.player_time_scale == 0 && (entity->flags & BOPL_ENTITY_TIMER_KNOWN) && entity->clock_rate > 0)
                        atomicMin(&w.time_stop, k);
                }
            } else if (q == 2) {
                // Slime k: its circle, what the slime pass needs of it.
                const BoplRlSlime* slime = &state->slimes[k];
                int32_t cls = 0;
                w.shas[k] = (unsigned char)bopl4_slime_circle(state, slime, self, w.scirc[k], &cls);
                w.scls[k] = cls;
                bopl4_slime_seen(state, slime, self, bopl_fix(self->x), bopl_fix(self->y), &w.seen[k]);
            } else {
                // Player k: its controlling body and its velocity, and the player as the primary enemy's candidate.
                const BoplRlPlayer* player = &state->players[k];
                const int hidden = bopl_spatial_hidden(state, player->id, self->id);
                const BoplRlSlime* body = bopl_controlling_slime(state, player->id, bopl_fix(player->x), bopl_fix(player->y));
                w.bodies[k] = body;
                w.body_vx[k] = body ? bopl_fix(body->vx) : 0.0; w.body_vy[k] = body ? bopl_fix(body->vy) : 0.0;
                w.is_candidate[k] = bopl4_enemy_candidate(state, self, bopl_fix(self->x), bopl_fix(self->y), (uint32_t)k, hidden, body,
                                                          &w.candidate[k], &w.candidate_distance[k]);
            }
        }
        br4_bar(1, BR4E_REST);

        // P2
        if (uwarp == 0) {
            if (lane < BR4E_ROWS && sh.valid[lane]) {
                // The observer, its body, the primary enemy, the scene's head, the slime pass.
                Br4ERow& w = sh.row[lane];
                const BoplRlState* state = &w.st;
                const BoplRlPlayer* self = &state->players[w.a];
                const double self_x = bopl_fix(self->x), self_y = bopl_fix(self->y);
                const BoplRlSlime* controller = w.bodies[w.a];
                w.x = self_x; w.y = self_y; w.controller = controller;
                w.vx = w.body_vx[w.a]; w.vy = w.body_vy[w.a];
                Bopl4Enemy enemy{};   // zero, as bopl4_primary_enemy's memset (in registers)
                double enemy_distance = 0.0;
                for (int i = 0; i < w.P; i++)
                    if (w.is_candidate[i]) bopl4_enemy_consider(&enemy, &enemy_distance, &w.candidate[i], w.candidate_distance[i]);
                w.enemy = enemy;
                w.enemy_controller = w.enemy.player ? w.bodies[w.enemy.player - state->players] : NULL;
                bopl4_scene_head(w.scene, w.ablation, self_x, self_y, w.vx, w.vy, (w.flags & BOPL4_STATE_AIMED) != 0, w.aim_dx, w.aim_dy, &w.enemy);
                int visible_slots;
                bopl4_slime_pass_of(state, self, self_x, self_y, controller, w.enemy_controller, w.seen,
                                    w.nearby, &w.slime_count, w.slots, &w.slot_count, &w.c.visible_slimes, &visible_slots);
            }
        } else if (uwarp == 1) {
            if (lane < BR4E_ROWS && sh.valid[lane]) {
                // The counts' rest; the nearest if a distance is NaN.
                Br4ERow& w = sh.row[lane];
                w.c.count = w.c.active_count;
                if (w.time_stop < BOPL4_ENTITIES) { w.c.time_stop_known = 1; w.c.time_stop_remaining = w.st.entities[w.time_stop].reserved[1]; }
                if (w.nan) {
                    double distances[BOPL4_ENTITIES];
                    int count = 0;
                    for (int k = 0; k < w.E; k++) if (w.kept[k]) bopl_nearest(w.nearest, distances, &count, BOPL4_ENTITIES, (uint32_t)k, w.distance[k]);
                }
            }
        } else {
            // The nearest: every kept entity at its rank by (distance, index).
            const int at0 = BR4E_ROWS, i0 = sh.off[at0];   // the entity pass's items
            for (int i = u - 64; i < sh.off[2 * BR4E_ROWS] - i0; i += BR4E_REST - 64) {
                int at = at0;
                while (sh.off[at + 1] <= i + i0) at++;
                const int k = i + i0 - sh.off[at];
                Br4ERow& w = sh.row[at - at0];
                if (k >= sh.cnt[at] || w.nan || !w.kept[k]) continue;
                const unsigned long long d = br4e_key(w.distance[k]);
                int rank = 0;
                for (int j = 0; j < w.E; j++)
                    if (w.kept[j]) { const unsigned long long e = br4e_key(w.distance[j]); rank += e < d || (e == d && j < k); }
                w.nearest[rank] = (uint32_t)k;
            }
        }
    }
    __syncthreads();

    // P3: a warp per row the circles, a warp per row the walls and water.
    if (warp < BR4E_ROWS && sh.valid[warp]) {
        Br4ERow& w = sh.row[warp];
        Bopl4Scene* s = w.scene;
        const int count = w.c.active_count, total = count + w.slime_count;
        int base = 0;
        for (int i0 = 0; i0 < total; i0 += 32) {
            const int i = i0 + lane;
            const float* c = nullptr; int32_t cls = 0; int has = 0;
            if (i < count) { const uint32_t k = w.nearest[i]; has = w.ehas[k]; c = w.ecirc[k]; cls = w.ecls[k]; }
            else if (i < total) { const uint32_t k = w.nearby[i - count]; has = w.shas[k]; c = w.scirc[k]; cls = w.scls[k]; }
            const unsigned ballot = __ballot_sync(0xffffffffu, has);
            if (has) bopl4_scene_circle(s, base + __popc(ballot & ((1u << lane) - 1u)), c, cls);
            base += __popc(ballot);
        }
        if (lane == 0) s->circle_count = base;
    } else if (warp >= BR4E_ROWS && warp < 2 * BR4E_ROWS && sh.valid[warp - BR4E_ROWS]) {
        // bopl4_scene_fill's walls: the first BOPL4F_WALLS candidates in entity order.
        const Br4ERow& w = sh.row[warp - BR4E_ROWS];
        Bopl4Scene* s = w.scene;
        int na = 0, nr = 0;
        unsigned long long axis_lethal = 0, rot_lethal = 0, rot_skew = 0;
        for (int k0 = 0; k0 < w.E; k0 += 32) {
            const int k = k0 + lane, kind = k < w.E ? w.kind[k] : BOPL4_SCENE_NONE;
            const unsigned below = (1u << lane) - 1u;
            const unsigned axis = __ballot_sync(0xffffffffu, kind == BOPL4_SCENE_AXIS), rot = __ballot_sync(0xffffffffu, kind == BOPL4_SCENE_ROT);
            const int ia = na + __popc(axis & below), ir = nr + __popc(rot & below);
            if ((kind == BOPL4_SCENE_AXIS || kind == BOPL4_SCENE_ROT) && ia + ir < BOPL4F_WALLS) {
                // The masks by the warp's OR below (bopl4_scene_axis and _rot OR them in).
                if (kind == BOPL4_SCENE_AXIS) { bopl4_scene_axis(s, ia, w.wall[k], 0); if (w.lethal[k]) axis_lethal |= 1ull << ia; }
                else {
                    bopl4_scene_rot(s, ir, w.wall[k], 0, 0);
                    if (w.lethal[k]) rot_lethal |= 1ull << ir;
                    if (w.skew[k]) rot_skew |= 1ull << ir;
                }
            }
            // Past the cap no candidate is kept: the counts stop there (the walls are taken in entity order).
            const unsigned kept = __ballot_sync(0xffffffffu, (kind == BOPL4_SCENE_AXIS || kind == BOPL4_SCENE_ROT) && ia + ir < BOPL4F_WALLS);
            na += __popc(kept & axis); nr += __popc(kept & rot);
        }
        axis_lethal = br4e_or(axis_lethal); rot_lethal = br4e_or(rot_lethal); rot_skew = br4e_or(rot_skew);
        if (lane == 0) {
            s->axis_count = na; s->rot_count = nr;
            s->axis_lethal = axis_lethal; s->rot_lethal = rot_lethal; s->rot_skew = rot_skew;
            double water_top = -1e9; int has_water = 0;
            for (int k = 0; k < w.E; k++)
                if (w.kind[k] == BOPL4_SCENE_WATER) {
                    if (!has_water || w.top[k] > water_top) water_top = w.top[k];
                    has_water = 1;
                }
            if (has_water) { s->flags |= BOPL4F_WATER; s->water_top = (float)water_top; } else s->water_top = -1e9f;
        }
    }
    // The unused set slots zeroed.
    for (int lr = 0; lr < BR4E_ROWS; lr++) {
        if (!sh.valid[lr]) continue;
        const Br4ERow& w = sh.row[lr];
        const int sets[4][3] = {{BOPL4_OBS_PLAYER_OFFSET, BOPL4_PLAYER_FEATURES * w.P, BOPL4_PLAYER_FEATURES * BOPL4_PLAYERS},
                                {BOPL4_OBS_SLOT_OFFSET, BOPL4_SLOT_FEATURES * w.slot_count, BOPL4_SLOT_FEATURES * BOPL4_SLOTS},
                                {BOPL4_OBS_ENTITY_OFFSET, BOPL4_ENTITY_FEATURES * w.c.active_count, BOPL4_ENTITY_FEATURES * BOPL4_ENTITIES},
                                {BOPL4_OBS_SLIME_OFFSET, BOPL4_SLIME_FEATURES * w.slime_count, BOPL4_SLIME_FEATURES * BOPL4_SLIMES}};
        for (int q = 0; q < 4; q++) for (int i = sets[q][1] + t; i < sets[q][2]; i += BR4E_THREADS) w.out[sets[q][0] + i] = 0.0f;
    }
    // The row's items' counts and what they read, for br4_items.
    if (t < BR4E_ROWS && r0 + t < rows) {
        const int lr = t;
        Br4ERow& w = sh.row[lr];
        const int valid = sh.valid[lr];
        const int n[BR4E_SEGS] = {valid ? w.c.active_count : 0, valid ? w.slime_count : 0, valid ? w.slot_count : 0, valid ? w.P : 0,
                                  valid, valid, valid};
        for (int q = 0; q < BR4E_SEGS; q++) counts[(r0 + lr) * 8 + q] = n[q];
        if (valid && w.enemy.player) w.enemy.player = &metas[r0 + lr].st.players[w.enemy.player - w.st.players];
    }
    __syncthreads();
    constexpr int MW = (int)(sizeof(Br4Meta) / 4);
    for (int i = t; i < BR4E_ROWS * MW; i += BR4E_THREADS) {
        const int lr = i / MW, j = i % MW;
        if (sh.valid[lr]) ((int32_t*)&metas[r0 + lr])[j] = ((const int32_t*)static_cast<const Br4Meta*>(&sh.row[lr]))[j];
    }
}

// ---- The row items (pack mode 4): br4_items ----
// Per (row, item, part) over every row of the launch, one kind after
// another, a kind from a warp's first thread (a warp runs one function): the
// entity, slime, slot and player rows, then the self, opponent and global
// blocks, each row or block cut into the parts its writer function names
// (bopl4_entity_part ...): a thread runs its part as one dependent chain at
// FP64 latency, so the parts run side by side.
// `blocks` blocks of bopl4_rays_items beside the rays' (independent work:
// the two overlap), the items dealt out a warp at a time round the blocks so
// the entity rows spread over the SMs. Each block scans the counts
// bopl4_encode_rows left (BR4R_MAX_ROWS rows at most per launch).
constexpr int BR4R_BLOCKS = 64, BR4R_MAX_ROWS = 256;
// The kinds: (count, part) per segment, a count of BR4E_SEGS (per row) each.
constexpr int BR4R_KINDS = BOPL4_ENTITY_PARTS + BOPL4_SLIME_PARTS + 1 + BOPL4_PLAYER_PARTS + 2 + BOPL4_OPPONENT_PARTS + 1;
__constant__ unsigned char br4r_count[BR4R_KINDS] = {0, 0, 0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 5, 6};
__constant__ unsigned char br4r_part[BR4R_KINDS] = {0, 1, 2, 3, 0, 1, 0, 0, 1, 0, 1, 0, 1, 2, 0};
static_assert(BR4R_KINDS == 15, "the tables above");
__host__ __device__ constexpr int br4r_seg(int k) { return k < 4 ? 0 : k < 6 ? 1 : k < 7 ? 2 : k < 9 ? 3 : k < 11 ? 4 : k < 14 ? 5 : 6; }   // br4r_count
struct Br4IShared { int pre[BR4E_SEGS][BR4R_MAX_ROWS + 1], base[BR4R_KINDS + 1]; int4 wsum[BR4_RAYS_THREADS / 32][2]; };
// Columns [LO, HI) of an item's row from the thread's copy to dst, A the
// floats dst is aligned to (16 bytes: 4): each store as wide as the
// alignment allows. A writer function's scattered float stores to the row
// (one sector a lane, the parts' rows lying apart) clogged the SM's L1 for
// every warp on it, the rays' shared loads too.
template <int LO, int HI, int A>
__device__ __forceinline__ void br4_put(float* __restrict__ dst, const float* row) {
    #pragma unroll
    for (int i = LO; i < HI;) {
        if (A == 4 && i % 4 == 0 && i + 4 <= HI) { *(float4*)(dst + i) = *(const float4*)(row + i); i += 4; }
        else if (i % 2 == 0 && i + 2 <= HI) { *(float2*)(dst + i) = *(const float2*)(row + i); i += 2; }
        else { dst[i] = row[i]; i += 1; }
    }
}
static_assert(BOPL_OBS_STRIDE % 4 == 0 && BOPL4_OBS_ENTITY_OFFSET % 4 == 0 && BOPL4_ENTITY_FEATURES % 4 == 0 && BOPL4_OBS_PLAYER_OFFSET % 4 == 0 &&
              BOPL4_PLAYER_FEATURES % 4 == 0 && BOPL4_OBS_GLOBAL_OFFSET % 4 == 0, "the 16-byte rows");
static_assert(BOPL4_OBS_SLIME_OFFSET % 2 == 0 && BOPL4_SLIME_FEATURES % 2 == 0 && BOPL4_OBS_SLOT_OFFSET % 2 == 0 && BOPL4_SLOT_FEATURES % 2 == 0 &&
              BOPL4_OBS_SELF_OFFSET % 2 == 0 && BOPL4_OBS_OPPONENT_OFFSET % 2 == 0, "the 8-byte rows");
static_assert(BOPL4_ENTITY_FEATURES == 48 && BOPL4_SLIME_FEATURES == 42 && BOPL4_SLOT_FEATURES == 18 && BOPL4_PLAYER_FEATURES == 32 &&
              BOPL_PLAYER_FEATURES == 20 && BOPL4_SELF == 44 && BOPL4_OPPONENT == 56 && BOPL4_GLOBAL == 66, "the parts' columns below");
__device__ static void br4_items(const Br4Meta* __restrict__ metas, const int* __restrict__ counts, int rows, int block, int blocks,
                                 Br4IShared& sh) {
    constexpr int T = BR4_RAYS_THREADS, W = T / 32;
    static_assert(BR4R_MAX_ROWS <= T && BR4E_SEGS <= 8, "a thread per row, its counts two int4s");
    const int t = threadIdx.x, lane = t & 31, wid = t >> 5;
    // The segments' exclusive prefixes over the rows, all seven at once (one
    // memory round and one barrier; a segment at a time took ~8 us).
    int c[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (t < rows) {
        const int4 a = ((const int4*)counts)[2 * t], b = ((const int4*)counts)[2 * t + 1];
        c[0] = a.x; c[1] = a.y; c[2] = a.z; c[3] = a.w; c[4] = b.x; c[5] = b.y; c[6] = b.z; c[7] = b.w;
    }
    int x[BR4E_SEGS];
    #pragma unroll
    for (int q = 0; q < BR4E_SEGS; q++) x[q] = c[q];
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1)
        #pragma unroll
        for (int q = 0; q < BR4E_SEGS; q++) { const int y = __shfl_up_sync(0xffffffffu, x[q], o); if (lane >= o) x[q] += y; }
    if (lane == 31) { sh.wsum[wid][0] = make_int4(x[0], x[1], x[2], x[3]); sh.wsum[wid][1] = make_int4(x[4], x[5], x[6], 0); }
    __syncthreads();
    // Every thread its warp's offsets and the totals from the warps' sums,
    // every load in flight at once (a thread's serial sum over the warps
    // below and thread 0's pass over the kinds behind its own barrier were
    // ~0.6 us of the items' start); thread 0 the kinds' bases.
    int below[BR4E_SEGS], total[BR4E_SEGS];
    #pragma unroll
    for (int q = 0; q < BR4E_SEGS; q++) below[q] = total[q] = 0;
    #pragma unroll
    for (int v = 0; v < W; v++) {
        const int4 a = sh.wsum[v][0], b = sh.wsum[v][1];
        const int y[BR4E_SEGS] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z};
        #pragma unroll
        for (int q = 0; q < BR4E_SEGS; q++) { total[q] += y[q]; below[q] += v < wid ? y[q] : 0; }
    }
    #pragma unroll
    for (int q = 0; q < BR4E_SEGS; q++) if (t < rows) sh.pre[q][t] = below[q] + x[q] - c[q];
    if (t == 0) {
        int base = 0;
        sh.base[0] = 0;
        #pragma unroll
        for (int q = 0; q < BR4E_SEGS; q++) sh.pre[q][rows] = total[q];
        #pragma unroll
        for (int k = 0; k < BR4R_KINDS; k++) sh.base[k + 1] = base = (base + total[br4r_seg(k)] + 31) & ~31;
    }
    __syncthreads();
    const int end = sh.base[BR4R_KINDS];
    for (int it = 0; ((block + blocks * (wid + W * it)) << 5) < end; it++) {
        const int i = ((block + blocks * (wid + W * it)) << 5) + lane;
        if (i >= end) continue;
        int kind = 0;
        #pragma unroll
        for (int k = 1; k < BR4R_KINDS; k++) kind += sh.base[k] <= i;
        const int q = br4r_count[kind], part = br4r_part[kind];
        const int j = i - sh.base[kind];
        if (j >= sh.pre[q][rows]) continue;   // the padding to the next kind's warp
        int rl = 0, rh = rows;                 // the last row whose items start at or before j
        while (rh - rl > 1) { const int m = (rl + rh) >> 1; if (sh.pre[q][m] <= j) rl = m; else rh = m; }
        const int k = j - sh.pre[q][rl];
        const Br4Meta& w = metas[rl];
        const BoplRlState* state = &w.st;
        const BoplRlPlayer* self = &state->players[w.a];
        const BoplRlEntity* entities = state->entities;
        const BoplRlSlime* slimes = state->slimes;
        const Bopl4Enemy* enemy = &w.enemy;
        float* out = w.out;
        // The part is written to the thread's row, then its columns to the row in memory.
        alignas(16) float row[BOPL4_GLOBAL];
        if (q == 0) {
            float* dst = out + BOPL4_OBS_ENTITY_OFFSET + (size_t)k * BOPL4_ENTITY_FEATURES;
            bopl4_entity_part(row, part, state, &entities[w.nearest[k]], self, w.x, w.y, w.vx, w.vy, enemy, w.ablation);
            if (part == 0) { br4_put<0, 33, 4>(dst, row); br4_put<47, 48, 4>(dst, row); }
            else if (part == 1) br4_put<34, 45, 4>(dst, row);
            else if (part == 2) br4_put<33, 34, 4>(dst, row);
            else br4_put<45, 47, 4>(dst, row);
        } else if (q == 1) {
            float* dst = out + BOPL4_OBS_SLIME_OFFSET + (size_t)k * BOPL4_SLIME_FEATURES;
            const uint32_t j = w.nearby[k];
            const BoplRlSlime* slime = &slimes[j];
            bopl4_slime_part(row, part, state, slime, self, w.x, w.y, w.vx, w.vy, enemy, slime == w.controller || slime == w.enemy_controller,
                             w.trig[2 * j], w.trig[2 * j + 1], w.ablation);
            if (part == 0) br4_put<0, 30, 2>(dst, row);
            else br4_put<30, 42, 2>(dst, row);
        } else if (q == 2) {
            bopl4_slot_row(row, state, &slimes[w.slots[k] / 3], w.slots[k] % 3, self, w.x, w.y, enemy);
            br4_put<0, BOPL4_SLOT_FEATURES, 2>(out + BOPL4_OBS_SLOT_OFFSET + (size_t)k * BOPL4_SLOT_FEATURES, row);
        } else if (q == 3) {
            float* dst = out + BOPL4_OBS_PLAYER_OFFSET + (size_t)k * BOPL4_PLAYER_FEATURES;
            const uint32_t index = k == 0 ? (uint32_t)w.a : ((uint32_t)k <= (uint32_t)w.a ? (uint32_t)k - 1 : (uint32_t)k);
            bopl4_player_part(row, part, state, &state->players[index], self, w.x, w.y, w.vx, w.vy, w.ablation);
            if (part == 0) br4_put<0, 20, 4>(dst, row);
            else br4_put<20, 32, 4>(dst, row);
        } else if (q == 4) {
            float* dst = out + BOPL4_OBS_SELF_OFFSET;
            const int c = w.controller ? (int)(w.controller - slimes) : -1;
            if (part == 0) { bopl_player_features(row, state, self, self, w.x, w.y); br4_put<0, 20, 2>(dst, row); }
            else {
                bopl4_body(row + BOPL_PLAYER_FEATURES, w.controller, c < 0 ? 0.0f : w.trig[2 * c], c < 0 ? 0.0f : w.trig[2 * c + 1]);
                br4_put<20, 44, 2>(dst, row);
            }
        } else if (q == 5) {
            float* dst = out + BOPL4_OBS_OPPONENT_OFFSET;
            const int c = enemy->slime ? (int)(enemy->slime - slimes) : -1;
            bopl4_opponent_part(row, part, state, enemy, self, w.x, w.y, w.vx, w.vy,
                                c < 0 ? 0.0f : w.trig[2 * c], c < 0 ? 0.0f : w.trig[2 * c + 1], w.ablation);
            if (part == 0) {
                br4_put<0, 20, 2>(dst, row);
                if (w.flags & BOPL4_STATE_ENEMY) {
                    out[BOPL4_OBS_PRIVILEGED_OFFSET + 43] = bopl4_stale_enemy(w.enemy_dx, row[9]);
                    out[BOPL4_OBS_PRIVILEGED_OFFSET + 44] = bopl4_stale_enemy(w.enemy_dy, row[10]);
                }
            } else if (part == 1) br4_put<20, 44, 2>(dst, row);
            else br4_put<44, 56, 2>(dst, row);
        } else {
            bopl4_global_block(row, state, &w.c, (uint32_t)w.max_ticks, (w.flags & BOPL4_STATE_SPACE) != 0, w.controller, 0, 1);
            br4_put<0, BOPL4_GLOBAL, 4>(out + BOPL4_OBS_GLOBAL_OFFSET, row);
        }
    }
}

// The scene bopl4_encode_rows left in global memory, by the whole block:
// the whole struct in one pass, every load in flight at once (the unused
// capacity comes along and is never read). Copying the used parts sized by
// the counts took five dependent memory rounds, ~6 us of a ~24 us ray block.
__device__ static void br4_scene_fetch(const Bopl4Scene* __restrict__ g, Br4Shared& sh) {
    constexpr int W = (int)(sizeof(Bopl4Scene) / 8), T = BR4_RAYS_THREADS, NL = (W + T - 1) / T;
    static_assert(sizeof(Bopl4Scene) % 8 == 0, "copied by 8-byte words");
    const uint2* src = (const uint2*)g;
    uint2* dst = (uint2*)&sh.s;
    uint2 v[NL];
    #pragma unroll
    for (int t = 0; t < NL; t++) if (t * T + (int)threadIdx.x < W) v[t] = src[t * T + threadIdx.x];
    #pragma unroll
    for (int t = 0; t < NL; t++) if (t * T + (int)threadIdx.x < W) dst[t * T + threadIdx.x] = v[t];
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
    // The first circle at the least distance below the planes' best: four circles a lane at once, without
    // branches (a circle the ray misses, or past the count, is dropped by the select).
    float cbest = best;
    int win = -1;
    for (int i0 = lane; i0 < s->circle_count; i0 += 4 * BR4_LANES) {
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            const int i = i0 + k * BR4_LANES;
            const float4 c = sh.circle[i];
            const float along = c.x * dx + c.y * dy, perp2 = c.z - along * along;
            float d = along - bopl4f_sqrt(c.w - perp2);
            d = d < 0.0f ? 0.0f : d;
            const bool take = i < s->circle_count && !(along < 0.0f) && !(perp2 > c.w) && d < cbest;
            cbest = take ? d : cbest; win = take ? i : win;
        }
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

// bopl4f_sdf at (x, y) over the n walls, or over the lethal ones, by one
// thread from the unified walls four at a time (their loads and square roots
// in flight together; bopl4f_sdf's loop over the mask's bits ran them one
// after another): the minimum in any order is the same float (no SDF is -0,
// bopl4f_box, and a NaN is never taken).
__device__ static float br4_sdf(const Br4Shared& sh, int n, bool lethal, float x, float y) {
    const int na = sh.s.axis_count;
    float d = 1e9f;
    for (int j = 0; j < n; j += 4) {
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            const float w = br4_wall_sdf(sh, j + k, x, y);
            const int at = sh.w.index[j + k];
            const bool in = j + k < n && (!lethal || (at < na ? (sh.s.axis_lethal >> at) & 1ull : (sh.s.rot_lethal >> (at - na)) & 1ull));
            d = in && w < d ? w : d;
        }
    }
    return d;
}

// bopl4f_lethal_at across the ray's lanes: the first strict minimum in wall
// order, the least (SDF, index) below 1e9 over the sorted walls.
__device__ static int br4_lethal_at(const Br4Shared& sh, int lane, float x, float y) {
    const int na = sh.s.axis_count, n = na + sh.s.rot_count;
    float best = 1e9f;
    int at = -1;
    for (int j = lane; j < n; j += BR4_LANES) {
        const float d = br4_wall_sdf(sh, j, x, y);
        const int k = sh.w.index[j];
        if (d < best || (d == best && at >= 0 && k < at)) { best = d; at = k; }
    }
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

// bopl4f_march over the first n walls, each step's SDF split across the
// ray's L lanes: K walls a lane unrolled, every load and SDF of a step in
// flight at once (a lane past n computes on a wall it then drops), the rest
// (n > 4 L) in a loop. The whole warp steps until its last live ray ends
// (the lanes ran that long anyway), so the shuffles take the full mask: a
// ray's own mask cost a convergence check per shuffle. A step's SDF at t is
// taken before t's end checks and the warp's exit branch, so the branch
// resolves beside the SDF instead of in front of it (~50 cycles a step; the
// last step's SDF is wasted). The minima are fminf, one instruction:
// w < d ? w : d is the same float, as d is never NaN (fminf drops a NaN w as
// the select does) and neither is -0 (bopl4f_box).
template <int L, int K>
__device__ static float br4_march(const Br4Shared& sh, int n, int lane, bool live, float ox, float oy, float dx, float dy, float max, float limit) {
    float t = 0.0f, out = 0.0f;
    for (int i = 0; i < 32; i++) {
        const float px = ox + dx * t, py = oy + dy * t;
        float d = 1e9f;
        #pragma unroll
        for (int k = 0; k < K; k++) { const int j = lane + k * L; const float w = br4_wall_sdf(sh, j, px, py); d = j < n ? fminf(w, d) : d; }
        if (K == 4) for (int j = lane + K * L; j < n; j += L) d = fminf(br4_wall_sdf(sh, j, px, py), d);
        #pragma unroll
        for (int o = 1; o < L; o <<= 1) d = fminf(__shfl_xor_sync(0xffffffffu, d, o), d);
        if (live && !(t < max)) { out = max; live = false; }
        if (live && t >= limit) { out = limit; live = false; }
        if (!__any_sync(0xffffffffu, live)) return out;
        if (live && d < 0.05f) { out = t; live = false; }
        t += d;
    }
    return live ? max : out;
}

// The walls culled at a cull bound (a group's reach + 0.1): the keys at or
// below it, counted by the calling warp (a ballot per 32).
__device__ __forceinline__ int br4_culled(const Br4Shared& sh, float bound) {
    int n = 0;
    #pragma unroll
    for (int c = 0; c < BR4_WALLS; c += 32) n += __popc(__ballot_sync(0xffffffffu, sh.w.key[c + (threadIdx.x & 31)] <= bound));
    return n;
}
// br4_march with the walls a lane (the same in the whole warp) as its K.
template <int L>
__device__ static float br4_marches(const Br4Shared& sh, int n, int lane, bool live, float ox, float oy, float dx, float dy, float max, float limit) {
    switch ((n + L - 1) / L) {
    case 0: case 1: return br4_march<L, 1>(sh, n, lane, live, ox, oy, dx, dy, max, limit);
    case 2: return br4_march<L, 2>(sh, n, lane, live, ox, oy, dx, dy, max, limit);
    case 3: return br4_march<L, 3>(sh, n, lane, live, ox, oy, dx, dy, max, limit);
    default: return br4_march<L, 4>(sh, n, lane, live, ox, oy, dx, dy, max, limit);
    }
}

// bopl4f_march_group: called by all `threads` threads of group g (barrier g,
// whole warps), gt the thread's index in it; an active thread's ray counts
// toward the group's reach, and a live one's (active too) is marched with the
// walls culled at it, by its L lanes. Each warp counts the culled walls
// itself: the same n in every warp, no second barrier. The fan's bound goes
// to sh.fan_cull for warp 7's probes.
template <int L = BR4_LANES>
__device__ static float br4_group(Br4Shared& sh, int g, int threads, int gt, bool active, bool live,
                                  float ox, float oy, float dx, float dy, float max, float limit) {
    const Bopl4Scene* s = &sh.s;
    if (!(s->axis_count + s->rot_count)) return max;
    const int lane = gt % L;
    if (active && lane == 0) {
        const float r = bopl4f_reach(s, ox, oy, max, limit);
        if (r > 0.0f) atomicMax(&sh.reach[g], __float_as_uint(r));   // reach=r>reach?r:reach from 0
    }
    br4_bar(g, threads);
    const float bound = __uint_as_float(sh.reach[g]) + 0.1f;
    if (g == BR4_FAN && gt == 0) *(volatile unsigned*)&sh.fan_cull = __float_as_uint(bound);
    return br4_marches<L>(sh, br4_culled(sh, bound), lane, live, ox, oy, dx, dy, max, limit);
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
    const float wall = br4_group(sh, BR4_AIM, 32, gt, active, active, s->x, s->y, dx, dy, (float)BOPL4_RAY_LENGTH, best + 1.0f);
    if (active) br4_finish(sh, lane, dx, dy, best, cls, closing, wall, out[ray]);
}

// The terrain block (bopl4f_terrain) by the dash warps (gt their thread),
// given the probes, the dash's landing and warp 7's SDFs: the edge probes,
// ray e side 0's steps 1 .. first_wall[0]-1, then side 1's, then thread 0
// writes the block.
__device__ static void br4_terrain(Br4Shared& sh, int gt, float* out) {
    const Bopl4Scene& s = sh.s;
    const float x = s.x, y = s.y, wt = s.water_top;
    const int walls = s.axis_count + s.rot_count, water = (s.flags & BOPL4F_WATER) != 0;
    float drop = sh.probes[0];
    if (water && y > wt && y - wt < drop) drop = y - wt;
    int first_wall[2] = {9, 9}, first_drop[2] = {9, 9};
    for (int side = 0; side < 2; side++)
        for (int k = 1; k <= 8; k++) if (sh.hit[side * 8 + k - 1]) { first_wall[side] = k; break; }
    const float threshold = drop + 4.0f;
    const bool edges = !(water && y > wt && !(y - wt > threshold)) && walls;
    if (!(water && y > wt && !(y - wt > threshold)) && !walls && 32.0f > threshold) first_drop[0] = first_drop[1] = 1;
    if (edges) {
        const int e = gt / BR4_LANES, left = first_wall[0] - 1, n = left + first_wall[1] - 1;
        const int side = e < left ? 0 : 1, k = side ? e - left + 1 : e + 1;
        const float d = br4_group(sh, BR4_EDGE, BR4_DASH_THREADS, gt, e < n, e < n, x + (side ? 1.5f : -1.5f) * (float)k, y,
                                  0.0f, -1.0f, 32.0f, bopl4f_next_up(threshold));
        if (gt % BR4_LANES == 0) sh.edge_drop[e] = e < n && d > threshold;
        br4_bar(BR4_EDGE, BR4_DASH_THREADS);
    }
    if (gt == 0) {
        if (edges) {
            const int left = first_wall[0] - 1, n = left + first_wall[1] - 1;
            for (int j = 0; j < n; j++) if (sh.edge_drop[j]) {
                const int sj = j < left ? 0 : 1, kj = sj ? j - left + 1 : j + 1;
                if (kj < first_drop[sj]) first_drop[sj] = kj;
            }
        }
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
}

__device__ static void br4_rays(const float* __restrict__ packed, float* __restrict__ dense, int rows, const Bopl4Scene* __restrict__ scenes,
                                int r, Br4Shared& sh) {
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
    if (mode == BOPL4_PACK_MODE_STATE) br4_scene_fetch(scenes + r, sh);
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
        // The fan (every other around ray under BOPL4F_HALF), the drop and ceiling probes counting
        // toward its reach. Under BOPL4F_AIM warp 7 writes the aim and line-of-sight slots, so the fan
        // leaves them alone (it wrote them first, then the block's join ordered warp 7's after).
        const int half = (s.flags & BOPL4F_HALF) != 0, n = half ? BOPL4_RAYS_AROUND / 2 : BOPL4_RAYS;
        const int aimed = (s.flags & BOPL4F_AIM) != 0;
        if (half) for (int k = i; k < BOPL4_RAYS * BOPL4_RAY_FEATURES; k += BR4_FAN_THREADS) {
            const int ray = k / BOPL4_RAY_FEATURES;
            if ((ray >= BOPL4_RAYS_AROUND || (ray & 1)) && !(aimed && (ray == BOPL4_AIM_RAY || ray == BOPL4_LOS_RAY))) rays[k] = 0.0f;
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
        const float wall = br4_group(sh, BR4_FAN, BR4_FAN_THREADS, i, ray < n + 2, ray < n, x, y, dx, dy, max, limit);
        if (ray < n && !(aimed && (k == BOPL4_AIM_RAY || k == BOPL4_LOS_RAY))) br4_finish(sh, lane, dx, dy, best, cls, closing, wall, rays + k * BOPL4_RAY_FEATURES);
    } else if (i < BR4_MISC) {
        // Ground within one dash (bopl4f_terrain), needed when the drop probe is over 4.
        const int gt = i - BR4_FAN_THREADS, ray = gt / BR4_LANES;
        const float dx = br4_dirs.dash[ray][0], dy = br4_dirs.dash[ray][1];
        const float d = br4_group(sh, BR4_DASH, BR4_DASH_THREADS, gt, true, true, x, y, dx, dy, (float)BOPL4_DASH_LENGTH, (float)BOPL4_DASH_LENGTH);
        const float px = x + dx * (d - 0.5f), py = y + dy * (d - 0.5f);
        bool fall = false, wet = false;
        if (!(water && py <= wt)) {
            if (water && !(py - wt > 4.0f)) wet = true;
            else if (walls) fall = true;
        }
        int landing = br4_bar_or(BR4_DASH, BR4_DASH_THREADS, wet);
        if (!landing && br4_bar_or(BR4_DASH, BR4_DASH_THREADS, fall)) {
            const float f = br4_group(sh, BR4_FALL, BR4_DASH_THREADS, gt, fall, fall, px, py, 0.0f, -1.0f, 8.0f, bopl4f_next_up(4.0f));
            landing = br4_bar_or(BR4_DASH, BR4_DASH_THREADS, fall && !(f > 4.0f));
        }
        if (gt == 0) sh.dash_landing = landing;
        br4_bar(BR4_EDGE, BR4_DASH_THREADS + 32);   // warp 7's probes and hits
        br4_terrain(sh, gt, region);
    } else {
        // The single SDFs on one path, a thread each (as three branches the warp ran them one after
        // another): the edge walls' (8-23), the body's and its gradient's (24-28), the lethal distance (29);
        // with no walls (lethal ones) 1e9: no hit, the lethal distance 16, the body's unused.
        const int gt = i - BR4_MISC;
        if (gt >= 8 && gt < 30) {
            float px = x, py = y;
            if (gt < 24) {
                const int e = gt - 8, side = e / 8, k = e % 8 + 1;
                px = x + (side ? 1.5f : -1.5f) * (float)k;
            } else if (gt < 29) {
                const int e = gt - 24;
                const float h = 0.05f;
                px = e == 1 ? x + h : e == 2 ? x - h : x; py = e == 3 ? y + h : e == 4 ? y - h : y;
            }
            const float d = br4_sdf(sh, walls, gt == 29, px, py);
            if (gt < 24) sh.hit[gt - 8] = d < 0.3f;
            else if (gt < 29) sh.sdf_at[gt - 24] = d;
            else sh.lethal = d < 16.0f ? d : 16.0f;
        }
        // The probes ride in the fan's group: its culled walls, once the fan's barrier has its reach
        // (by the time the SDFs are done, as a rule). Without walls they end at their max. Then the dash
        // warps' terrain block may go on, and warp 7 casts the aim and line of sight.
        float probe = 32.0f;
        if (walls) {
            unsigned bound;
            while (!(bound = *(volatile unsigned*)&sh.fan_cull)) {}
            probe = br4_marches<16>(sh, br4_culled(sh, __uint_as_float(bound)), gt % 16, true, x, y, 0.0f, gt < 16 ? -1.0f : 1.0f, 32.0f, 32.0f);
        }
        if (gt % 16 == 0) sh.probes[gt / 16] = probe;
        br4_arrive(BR4_EDGE, BR4_DASH_THREADS + 32);
        if (s.flags & BOPL4F_AIM) br4_aim_rays(sh, gt, sh.aim_out);
        __syncwarp();
        if ((s.flags & BOPL4F_AIM) && gt < 2 * BOPL4_RAY_FEATURES)
            (gt < BOPL4_RAY_FEATURES ? aim : los)[gt % BOPL4_RAY_FEATURES] = sh.aim_out[gt / BOPL4_RAY_FEATURES][gt % BOPL4_RAY_FEATURES];
    }
}

// The row items and the rays in one launch: blocks [0, item_blocks) the
// items (br4_items), the rest a row's rays each.
union Br4FusedShared { Br4Shared rays; Br4IShared items; };
__global__ void __launch_bounds__(BR4_RAYS_THREADS, 4) bopl4_rays_items(const float* __restrict__ packed, float* __restrict__ dense, int rows,
                                                                       const Bopl4Scene* __restrict__ scenes, const Br4Meta* __restrict__ metas,
                                                                       const int* __restrict__ counts, int item_blocks) {
    __shared__ Br4FusedShared sh;
    if ((int)blockIdx.x < item_blocks) br4_items(metas, counts, rows, blockIdx.x, item_blocks, sh.items);
    else br4_rays(packed, dense, rows, scenes, blockIdx.x - item_blocks, sh.rays);
}

// The directions, once before any launch (a stream capture cannot copy to a
// symbol), and a scene per slot of the packed buffer `packed` (`slots`
// slots) for the rows encoded from their state: rows of different buffers
// (streams) have their own.
static const float* br4_packed;
static long br4_slots;
static Bopl4Scene* br4_scenes;
static Br4Meta* br4_metas;
static int* br4_counts;
extern "C" void bopl4_rays_init(const float* packed, long slots) {
    Bopl4Dirs dirs;
    bopl4f_dirs_init(&dirs);
    if (cudaMemcpyToSymbol(br4_dirs, &dirs, sizeof(dirs)) != cudaSuccess ||
        cudaMalloc((void**)&br4_scenes, (size_t)slots * sizeof(Bopl4Scene)) != cudaSuccess ||
        cudaMalloc((void**)&br4_metas, (size_t)slots * sizeof(Br4Meta)) != cudaSuccess ||
        cudaMalloc((void**)&br4_counts, (size_t)slots * 8 * sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "bopl4_rays_init failed\n");
        abort();
    }
    br4_packed = packed; br4_slots = slots;
}
// After bopl4_unpack_rows on the same rows: the rows shipped as their state
// (pack mode 4), then the terrain and rays of every row with a scene.
extern "C" void bopl4_rays_launch(const float* packed, float* dense, int rows, cudaStream_t stream) {
    for (int at = 0; at < rows; at += BR4R_MAX_ROWS) {
        const int n = rows - at < BR4R_MAX_ROWS ? rows - at : BR4R_MAX_ROWS;
        const float* p = packed + (long)at * BOPL4_PACK_P;
        float* d = dense + (long)at * BOPL_OBS_STRIDE;
        const long first = (long)(p - br4_packed) / BOPL4_PACK_P;
        if (!br4_scenes || p < br4_packed || first + n > br4_slots) { fprintf(stderr, "bopl4_rays_launch: rows outside the packed buffer\n"); abort(); }
        bopl4_encode_rows<<<(n + BR4E_ROWS - 1) / BR4E_ROWS, BR4E_THREADS, 0, stream>>>(p, d, n, br4_scenes + first, br4_metas + first,
                                                                                         br4_counts + first * 8);
        bopl4_rays_items<<<BR4R_BLOCKS + n, BR4_RAYS_THREADS, 0, stream>>>(p, d, n, br4_scenes + first, br4_metas + first,
                                                                           br4_counts + first * 8, BR4R_BLOCKS);
    }
}
