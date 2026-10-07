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

struct Br4Shared {
    float raw[BOPL4_PACK_P - BOPL4_PACK_HEADER];   // the slot from the scene on, one coalesced read
    Bopl4Scene s;
    Bopl4Anchor a;
    // Every wall as a rotated one (aligned: right (1, 0), up (0, 1); the same
    // bits, see br4_wall_sdf), aligned walls first: index j is lethal_at's order.
    float cx[BR4_WALLS], cy[BR4_WALLS], rx[BR4_WALLS], ry[BR4_WALLS], ux[BR4_WALLS], uy[BR4_WALLS],
          ex[BR4_WALLS], ey[BR4_WALLS], radius[BR4_WALLS];
    unsigned reach[BR4_GROUPS];                 // a group's reach, as the bits of a non-negative float
    unsigned char list[BR4_GROUPS][BR4_WALLS];  // its culled walls
    int listn[BR4_GROUPS];
    float probes[2], sdf_at[5], lethal, aim_out[2][BOPL4_RAY_FEATURES];
    int hit[16], edge_drop[16], dash_landing;
};

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
    const float px = x - sh.cx[j], py = y - sh.cy[j];
    const float lx = px * sh.rx[j] + py * sh.ry[j], ly = px * sh.ux[j] + py * sh.uy[j];
    return bopl4f_box(fabsf(lx) - sh.ex[j], fabsf(ly) - sh.ey[j], sh.radius[j]);
}

// The scene from its packed form (bopl4f_scene_write), by the whole block:
// the slot's words from the scene to its end copied in one pass (one memory
// round trip), then taken apart in shared memory.
__device__ static void br4_scene_read(const float* slot, int at, Br4Shared& sh) {
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
    if (threadIdx.x < BR4_GROUPS) sh.reach[threadIdx.x] = 0u;
    in += BOPL4F_SCENE_HEAD;
    float* axis[5] = {s->axis_cx, s->axis_cy, s->axis_ex, s->axis_ey, s->axis_radius};
    float* rot[9] = {s->rot_cx, s->rot_cy, s->rot_rx, s->rot_ry, s->rot_ux, s->rot_uy, s->rot_ex, s->rot_ey, s->rot_radius};
    float* circles[5] = {s->cx, s->cy, s->cr, s->cvx, s->cvy};
    for (int f = 0; f < 5; f++, in += na) for (int i = threadIdx.x; i < na; i += blockDim.x) axis[f][i] = in[i];
    for (int f = 0; f < 9; f++, in += nr) for (int i = threadIdx.x; i < nr; i += blockDim.x) rot[f][i] = in[i];
    for (int f = 0; f < 5; f++, in += nc) for (int i = threadIdx.x; i < nc; i += blockDim.x) circles[f][i] = in[i];
    for (int i = threadIdx.x; i < nc; i += blockDim.x) s->cclass[i] = ((const int32_t*)in)[i];
    __syncthreads();
    for (int j = threadIdx.x; j < na + nr; j += blockDim.x) {
        if (j < na) {
            sh.cx[j] = s->axis_cx[j]; sh.cy[j] = s->axis_cy[j]; sh.rx[j] = 1.0f; sh.ry[j] = 0.0f; sh.ux[j] = 0.0f; sh.uy[j] = 1.0f;
            sh.ex[j] = s->axis_ex[j]; sh.ey[j] = s->axis_ey[j]; sh.radius[j] = s->axis_radius[j];
            sh.a.axis[j] = bopl4f_axis_sdf(s, j, s->x, s->y);
        } else {
            const int k = j - na;
            sh.cx[j] = s->rot_cx[k]; sh.cy[j] = s->rot_cy[k]; sh.rx[j] = s->rot_rx[k]; sh.ry[j] = s->rot_ry[k];
            sh.ux[j] = s->rot_ux[k]; sh.uy[j] = s->rot_uy[k]; sh.ex[j] = s->rot_ex[k]; sh.ey[j] = s->rot_ey[k]; sh.radius[j] = s->rot_radius[k];
            sh.a.rot[k] = bopl4f_rot_sdf(s, k, s->x, s->y);
        }
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
    const unsigned char* list = sh.list[g];
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
        for (uint64_t b = m.axis; b; b &= b - 1ull) sh.list[g][n++] = (unsigned char)bopl4f_ctz(b);
        for (uint64_t b = m.rot; b; b &= b - 1ull) sh.list[g][n++] = (unsigned char)(s->axis_count + bopl4f_ctz(b));
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
    if (hdr[0] != BOPL4_PACK_MODE_SCENE && hdr[0] != BOPL4_PACK_MODE_DENSE_SCENE) return;
    float* region = dense + (long)r * BOPL_OBS_STRIDE + BOPL4_OBS_TERRAIN_OFFSET;
    float* rays = region + BOPL4_TERRAIN;
    float* aim = rays + BOPL4_AIM_RAY * BOPL4_RAY_FEATURES;
    float* los = rays + BOPL4_LOS_RAY * BOPL4_RAY_FEATURES;
    const int i = threadIdx.x;
    if (i < 2 * BOPL4_RAY_FEATURES) sh.aim_out[i / BOPL4_RAY_FEATURES][i % BOPL4_RAY_FEATURES] = 0.0f;
    br4_scene_read(slot, (int)hdr[5], sh);
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
