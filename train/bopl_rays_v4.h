/* The v4 row's terrain block and rays (BOPL4_TERRAIN, BOPL4_RAYS) in single
 * precision (2026-10-06): one source for the CPU writer and the trainer's
 * CUDA writer, which agree bit for bit. Only IEEE single operations: no FMA
 * contraction (the pragma below; nvcc -fmad=false), sqrt and division
 * correctly rounded on both sides, min and max written as selects, and the
 * trig constants (fan, dash and aim directions) computed once on the host.
 * FP64 put this block out of the GPU's reach: the RTX 5070 retires 234 G
 * FP64 add/mul and 27 G sqrt per second, so the writer's ~90k FP64 operations
 * per row would have taken most of it at 1M rows per second.
 *
 * The input is a Bopl4Scene, which bopl4_scene_fill (bopl_obs_v4.h) builds
 * from the delayed state: the walls (rounded rects on the terrain layers),
 * water, the dynamic bodies as circles, and the body's position and velocity.
 * Every march is a group of rays that culls the walls none of its rays can
 * reach from the SDF at the body (bopl4f_cull); the CPU and the CUDA writer
 * form the same groups, so they cull the same walls. */
#ifndef BOPL_RAYS_V4_H
#define BOPL_RAYS_V4_H
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "bopl_layout_v4.h"

#ifdef __CUDACC__
#define BOPL4F_FN static __host__ __device__ inline
#else
#define BOPL4F_FN static inline
#endif
#if defined(__clang__) && !defined(__CUDACC__)
#pragma clang fp contract(off)
#elif defined(__GNUC__) && !defined(__CUDACC__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif

enum { BOPL4F_WALLS=48, BOPL4F_CIRCLES=BOPL4_ENTITIES+BOPL4_SLIMES, BOPL4F_GROUP=BOPL4_RAYS+2 };
/* Scene flags: the fan and terrain block (off under BOPL4_ABLATE_RAYS), every
 * other around ray only (BOPL4_ABLATE_RAYS16), the aim and line-of-sight
 * slots (BOPL4_ABLATE_AIM), water present, a decision with an aim applied,
 * the primary enemy's position known. */
enum { BOPL4F_FAN=1, BOPL4F_HALF=2, BOPL4F_AIM=4, BOPL4F_WATER=8, BOPL4F_AIMED=16, BOPL4F_ENEMY=32 };
#define BOPL4F_BLAST_X 131.25f   /* bopl_bots.h BOT_BLAST_X, BOT_BLAST_Y */
#define BOPL4F_BLAST_Y 72.5f

/* Walls split by frame: axis-aligned (right = (+-1, 0), up = (0, +-1), or
 * turned with the extents swapped, so |local| = |p - centre| without the
 * products) and the rotated rest. Lethal walls (layer 16) and rotated walls
 * whose frame is not orthogonal (never culled: their SDF need not be
 * 1-Lipschitz) are bit masks. */
typedef struct {
    int32_t flags, axis_count, rot_count, circle_count;
    float x, y, svx, svy, water_top, aim_dx, aim_dy, enemy_x, enemy_y;
    uint64_t axis_lethal, rot_lethal, rot_skew;
    float axis_cx[BOPL4F_WALLS], axis_cy[BOPL4F_WALLS], axis_ex[BOPL4F_WALLS], axis_ey[BOPL4F_WALLS], axis_radius[BOPL4F_WALLS];
    float rot_cx[BOPL4F_WALLS], rot_cy[BOPL4F_WALLS], rot_rx[BOPL4F_WALLS], rot_ry[BOPL4F_WALLS], rot_ux[BOPL4F_WALLS],
          rot_uy[BOPL4F_WALLS], rot_ex[BOPL4F_WALLS], rot_ey[BOPL4F_WALLS], rot_radius[BOPL4F_WALLS];
    /* Circles: cclass 0 hostile projectile / object, 1 opponent body, 2 own object. */
    float cx[BOPL4F_CIRCLES], cy[BOPL4F_CIRCLES], cr[BOPL4F_CIRCLES], cvx[BOPL4F_CIRCLES], cvy[BOPL4F_CIRCLES];
    int32_t cclass[BOPL4F_CIRCLES];
} Bopl4Scene;
/* The fan's and the landing test's directions (host-computed constants). */
typedef struct { float fan[BOPL4_RAYS][2], dash[16][2]; } Bopl4Dirs;
/* Each wall's SDF at the body, for the culling. */
typedef struct { float axis[BOPL4F_WALLS], rot[BOPL4F_WALLS]; } Bopl4Anchor;
typedef struct { uint64_t axis, rot; } Bopl4Mask;

/* Ray k: BOPL4_RAYS_AROUND uniform from +x, then the fan about straight down
 * (cell centres of a 120-degree span); dash k: 2 pi k / 16. */
static void bopl4f_dirs_init(Bopl4Dirs *dirs) {
    for (int k=0; k<BOPL4_RAYS; k++) {
        double a;
        if (k<BOPL4_RAYS_AROUND) a=6.28318530717958647692*k/BOPL4_RAYS_AROUND;
        else { double span=6.28318530717958647692/3.0; a=-1.57079632679489661923-span/2+span*(k-BOPL4_RAYS_AROUND+0.5)/BOPL4_RAYS_DOWN; }
        dirs->fan[k][0]=(float)cos(a); dirs->fan[k][1]=(float)sin(a);
    }
    for (int k=0; k<16; k++) { double a=6.28318530717958647692*k/16.0; dirs->dash[k][0]=(float)cos(a); dirs->dash[k][1]=(float)sin(a); }
}

/* On the GPU __fsqrt_rn's own steps without its branch to the slow path
 * (taken for 0, under 2^-101, inf, NaN and negatives): one reconvergence
 * point per call kept a march step's walls from overlapping. Tiny inputs are
 * scaled by 2^100 (exact, the root by 2^-50), 0 and inf are their own roots,
 * the rest is NaN as the steps leave it: equal to __fsqrt_rn on all 2^32
 * inputs (checked exhaustively on the RTX 5070). */
BOPL4F_FN float bopl4f_sqrt(float v) {
#ifdef __CUDA_ARCH__
    const bool tiny = v < 0x1p-100f;
    const float x = tiny ? v * 0x1p100f : v;
    float r;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    const float y = __fmul_rn(x, r), e = __fmaf_rn(-y, y, x), s = __fmaf_rn(e, __fmul_rn(r, 0.5f), y);
    return x == 0.0f || x == __int_as_float(0x7f800000) ? v : tiny ? s * 0x1p-50f : s;
#else
    return sqrtf(v);
#endif
}
BOPL4F_FN float bopl4f_div(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fdiv_rn(a, b);
#else
    return a/b;
#endif
}
BOPL4F_FN int bopl4f_ctz(uint64_t bits) {
#ifdef __CUDA_ARCH__
    return __ffsll((long long)bits)-1;
#else
    return __builtin_ctzll(bits);
#endif
}
BOPL4F_FN float bopl4f_clip(float v) { v=v<1.0f?v:1.0f; return v>-1.0f?v:-1.0f; }
/* The next float above a positive finite v. */
BOPL4F_FN float bopl4f_next_up(float v) { uint32_t b; memcpy(&b, &v, 4); b++; memcpy(&v, &b, 4); return v; }
BOPL4F_FN Bopl4Mask bopl4f_all(const Bopl4Scene *s) {
    Bopl4Mask m={(1ull<<s->axis_count)-1ull, (1ull<<s->rot_count)-1ull};
    return m;
}

/* A rounded rect's SDF from its local offsets past the extents. The selects
 * are x86's max/min (the second operand on a tie or NaN). */
BOPL4F_FN float bopl4f_box(float qx, float qy, float radius) {
    float ox=qx>0.0f?qx:0.0f, oy=qy>0.0f?qy:0.0f;
    float m=qx>qy?qx:qy, inside=m<0.0f?m:0.0f;
    return bopl4f_sqrt(ox*ox+oy*oy)+inside-radius;
}
BOPL4F_FN float bopl4f_axis_sdf(const Bopl4Scene *s, int k, float x, float y) {
    return bopl4f_box(fabsf(x-s->axis_cx[k])-s->axis_ex[k], fabsf(y-s->axis_cy[k])-s->axis_ey[k], s->axis_radius[k]);
}
BOPL4F_FN float bopl4f_rot_sdf(const Bopl4Scene *s, int k, float x, float y) {
    float px=x-s->rot_cx[k], py=y-s->rot_cy[k];
    float lx=px*s->rot_rx[k]+py*s->rot_ry[k], ly=px*s->rot_ux[k]+py*s->rot_uy[k];
    return bopl4f_box(fabsf(lx)-s->rot_ex[k], fabsf(ly)-s->rot_ey[k], s->rot_radius[k]);
}
/* The terrain SDF over the walls in m: the minimum, taken as w < d ? w : d
 * (x86 min(w, d)). */
BOPL4F_FN float bopl4f_sdf(const Bopl4Scene *s, Bopl4Mask m, float x, float y) {
    float d=1e9f;
    for (uint64_t b=m.axis; b; b&=b-1ull) { float w=bopl4f_axis_sdf(s, bopl4f_ctz(b), x, y); d=w<d?w:d; }
    for (uint64_t b=m.rot; b; b&=b-1ull) { float w=bopl4f_rot_sdf(s, bopl4f_ctz(b), x, y); d=w<d?w:d; }
    return d;
}
/* Whether the nearest wall at (x, y) is lethal: the first strict minimum,
 * aligned walls then rotated ones. */
BOPL4F_FN int bopl4f_lethal_at(const Bopl4Scene *s, float x, float y) {
    float best=1e9f; int lethal=0;
    for (int k=0; k<s->axis_count; k++) { float d=bopl4f_axis_sdf(s, k, x, y); if (d<best) { best=d; lethal=(int)((s->axis_lethal>>k)&1ull); } }
    for (int k=0; k<s->rot_count; k++) { float d=bopl4f_rot_sdf(s, k, x, y); if (d<best) { best=d; lethal=(int)((s->rot_lethal>>k)&1ull); } }
    return lethal;
}
/* A march group's walls: a ray samples points at t < min(max, limit) from its
 * origin O, so within that plus |O - B| of the body B; a wall whose SDF at B
 * exceeds the group's largest such reach by 0.1 is never a hit, and when it
 * is the minimum the step takes the ray past its end either way. */
BOPL4F_FN float bopl4f_reach(const Bopl4Scene *s, float ox, float oy, float max, float limit) {
    float ex=ox-s->x, ey=oy-s->y;
    return (limit<max?limit:max)+bopl4f_sqrt(ex*ex+ey*ey);
}
BOPL4F_FN Bopl4Mask bopl4f_cull(const Bopl4Scene *s, const Bopl4Anchor *a, float reach) {
    Bopl4Mask m={0, 0};
    for (int k=0; k<s->axis_count; k++) if (!(a->axis[k]>reach)) m.axis|=1ull<<k;
    for (int k=0; k<s->rot_count; k++) if (((s->rot_skew>>k)&1ull) || !(a->rot[k]>reach)) m.rot|=1ull<<k;
    return m;
}
/* bot_raycast's sphere march: the distance to the walls in m along (dx, dy),
 * max when the ray ends first or after 32 steps, limit once t reaches it. */
BOPL4F_FN float bopl4f_march(const Bopl4Scene *s, Bopl4Mask m, float ox, float oy, float dx, float dy, float max, float limit) {
    float t=0.0f;
    for (int i=0; i<32; i++) {
        if (!(t<max)) return max;
        if (t>=limit) return limit;
        float d=bopl4f_sdf(s, m, ox+dx*t, oy+dy*t);
        if (d<0.05f) return t;
        t+=d;
    }
    return max;
}
/* A ray's surfaces other than walls, in order, each taken on a strict d <
 * best: water's top (rays going up into it from below are not cut), the blast
 * bounds, then the circles. Class 2 for a plane, 3 + cclass for a circle, -1
 * when nothing within the ray; closing is a circle's speed toward the body
 * along the ray. */
BOPL4F_FN float bopl4f_surfaces(const Bopl4Scene *s, float dx, float dy, int *cls, float *closing) {
    float x=s->x, y=s->y, best=(float)BOPL4_RAY_LENGTH;
    *cls=-1; *closing=0.0f;
    if ((s->flags&BOPL4F_WATER) && dy<-1e-9f) { float d=bopl4f_div(s->water_top-y, dy); if (d>=0.0f && d<best) { best=d; *cls=2; } }
    if (dy>1e-9f) { float d=bopl4f_div(BOPL4F_BLAST_Y-y, dy); if (d>=0.0f && d<best) { best=d; *cls=2; } }
    if (dx>1e-9f) { float d=bopl4f_div(BOPL4F_BLAST_X-x, dx); if (d>=0.0f && d<best) { best=d; *cls=2; } }
    if (dx<-1e-9f) { float d=bopl4f_div(-BOPL4F_BLAST_X-x, dx); if (d>=0.0f && d<best) { best=d; *cls=2; } }
    int win=-1;
    for (int i=0; i<s->circle_count; i++) {
        float px=s->cx[i]-x, py=s->cy[i]-y;
        float along=px*dx+py*dy;
        if (along<0.0f) continue;
        float perp2=(px*px+py*py)-along*along, r2=s->cr[i]*s->cr[i];
        if (perp2>r2) continue;
        float d=along-bopl4f_sqrt(r2-perp2);
        d=d<0.0f?0.0f:d;
        if (d<best) { best=d; win=i; }
    }
    if (win>=0) { *cls=3+s->cclass[win]; *closing=-((s->cvx[win]-s->svx)*dx+(s->cvy[win]-s->svy)*dy); }
    return best;
}
/* A ray's 8 fields given its other surfaces and the walls' marched distance:
 * distance over the length, the class one-hot, the closing speed. A wall at
 * exactly another surface's distance takes the ray. */
BOPL4F_FN void bopl4f_finish(const Bopl4Scene *s, float dx, float dy, float best, int cls, float closing, float wall, float *out) {
    if (s->axis_count+s->rot_count && wall<(float)BOPL4_RAY_LENGTH && wall<=best) {
        best=wall; cls=bopl4f_lethal_at(s, s->x+dx*wall, s->y+dy*wall); closing=0.0f;
    }
    out[0]=bopl4f_div(best, (float)BOPL4_RAY_LENGTH);
    for (int c=0; c<6; c++) out[1+c]=cls==c;
    out[7]=cls>=3?bopl4f_clip(bopl4f_div(closing, (float)BOPL4_VELOCITY_SCALE)):0.0f;
}
/* The line of sight's direction, toward the enemy's centre; 0 when the enemy
 * is unknown or on the body. */
BOPL4F_FN int bopl4f_los(const Bopl4Scene *s, float *dx, float *dy) {
    if (!(s->flags&BOPL4F_ENEMY)) return 0;
    float ex=s->enemy_x-s->x, ey=s->enemy_y-s->y, len=bopl4f_sqrt(ex*ex+ey*ey);
    if (!(len>1e-9f)) return 0;
    *dx=bopl4f_div(ex, len); *dy=bopl4f_div(ey, len);
    return 1;
}
/* The scene as shipped in a packed slot (bopl_obs_pack.h): BOPL4F_SCENE_HEAD
 * words (flags, the three counts, the 9 scalars, the three masks as low/high
 * words), then each array cut to its count: the aligned walls' 5, the rotated
 * walls' 9, the circles' 5 and their classes. */
enum { BOPL4F_SCENE_HEAD=19 };
BOPL4F_FN int bopl4f_scene_words(const Bopl4Scene *s) {
    return BOPL4F_SCENE_HEAD+5*s->axis_count+9*s->rot_count+6*s->circle_count;
}
static inline void bopl4f_scene_put(float *out, const void *src, int words) { memcpy(out, src, (size_t)words*4); }
static void bopl4f_scene_write(const Bopl4Scene *s, float *out) {
    int na=s->axis_count, nr=s->rot_count, nc=s->circle_count;
    uint32_t masks[6]={(uint32_t)s->axis_lethal, (uint32_t)(s->axis_lethal>>32), (uint32_t)s->rot_lethal,
                       (uint32_t)(s->rot_lethal>>32), (uint32_t)s->rot_skew, (uint32_t)(s->rot_skew>>32)};
    bopl4f_scene_put(out, &s->flags, 4);   /* flags and the counts */
    bopl4f_scene_put(out+4, &s->x, 9);     /* x .. enemy_y */
    bopl4f_scene_put(out+13, masks, 6);
    out+=BOPL4F_SCENE_HEAD;
    const float *axis[5]={s->axis_cx, s->axis_cy, s->axis_ex, s->axis_ey, s->axis_radius};
    const float *rot[9]={s->rot_cx, s->rot_cy, s->rot_rx, s->rot_ry, s->rot_ux, s->rot_uy, s->rot_ex, s->rot_ey, s->rot_radius};
    const float *circles[5]={s->cx, s->cy, s->cr, s->cvx, s->cvy};
    for (int f=0; f<5; f++, out+=na) bopl4f_scene_put(out, axis[f], na);
    for (int f=0; f<9; f++, out+=nr) bopl4f_scene_put(out, rot[f], nr);
    for (int f=0; f<5; f++, out+=nc) bopl4f_scene_put(out, circles[f], nc);
    bopl4f_scene_put(out, s->cclass, nc);
}
BOPL4F_FN void bopl4f_anchor(const Bopl4Scene *s, Bopl4Anchor *a) {
    for (int k=0; k<s->axis_count; k++) a->axis[k]=bopl4f_axis_sdf(s, k, s->x, s->y);
    for (int k=0; k<s->rot_count; k++) a->rot[k]=bopl4f_rot_sdf(s, k, s->x, s->y);
}

#ifndef __CUDACC__
/* ---- The CPU writer: the groups one ray at a time. ---- */
/* March n rays as one group into out[] (max[i] for every ray without walls). */
static void bopl4f_march_group(const Bopl4Scene *s, const Bopl4Anchor *a, int n, const float *ox, const float *oy,
                               const float *dx, const float *dy, const float *max, const float *limit, float *out) {
    if (!(s->axis_count+s->rot_count)) { for (int i=0; i<n; i++) out[i]=max[i]; return; }
    float reach=0.0f;
    for (int i=0; i<n; i++) { float r=bopl4f_reach(s, ox[i], oy[i], max[i], limit[i]); reach=r>reach?r:reach; }
    Bopl4Mask m=bopl4f_cull(s, a, reach+0.1f);
    for (int i=0; i<n; i++) out[i]=bopl4f_march(s, m, ox[i], oy[i], dx[i], dy[i], max[i], limit[i]);
}
/* n rays from the body into out[i]; with probes, the drop and ceiling casts
 * (straight down and up, 32 units) ride in the same group. */
static void bopl4f_cast_group(const Bopl4Scene *s, const Bopl4Anchor *a, int n, const float *dx, const float *dy,
                              float *const *out, float *probes) {
    float best[BOPL4F_GROUP], closing[BOPL4F_GROUP], ox[BOPL4F_GROUP], oy[BOPL4F_GROUP], mx[BOPL4F_GROUP], my[BOPL4F_GROUP];
    float max[BOPL4F_GROUP], limit[BOPL4F_GROUP], wall[BOPL4F_GROUP];
    int cls[BOPL4F_GROUP], marched=n;
    for (int i=0; i<n; i++) {
        best[i]=bopl4f_surfaces(s, dx[i], dy[i], &cls[i], &closing[i]);
        ox[i]=s->x; oy[i]=s->y; mx[i]=dx[i]; my[i]=dy[i]; limit[i]=best[i]+1.0f; max[i]=(float)BOPL4_RAY_LENGTH;
    }
    if (probes) for (int i=0; i<2; i++, marched++) {
        ox[marched]=s->x; oy[marched]=s->y; mx[marched]=0.0f; my[marched]=i?1.0f:-1.0f; limit[marched]=max[marched]=32.0f;
    }
    bopl4f_march_group(s, a, marched, ox, oy, mx, my, max, limit, wall);
    if (probes) { probes[0]=wall[n]; probes[1]=wall[n+1]; }
    for (int i=0; i<n; i++) bopl4f_finish(s, dx[i], dy[i], best[i], cls[i], closing[i], wall[i], out[i]);
}
/* The egocentric terrain block at the body, given the drop and ceiling casts. */
static void bopl4f_terrain(const Bopl4Scene *s, const Bopl4Dirs *dirs, const Bopl4Anchor *a, const float *drop_ceiling, float *out) {
    float x=s->x, y=s->y, wt=s->water_top;
    int walls=s->axis_count+s->rot_count, water=(s->flags&BOPL4F_WATER)!=0;
    Bopl4Mask all=bopl4f_all(s);
    float drop=drop_ceiling[0];
    if (water && y>wt && y-wt<drop) drop=y-wt;
    out[0]=bopl4f_div(drop, 32.0f); out[1]=bopl4f_div(drop_ceiling[1], 32.0f);
    /* Edges: walk left/right in 1.5-unit steps to one dash length; the edge
     * is where the floor drops away by more than 4 units (or ends in water),
     * a wall is where the step lands inside terrain. */
    int first_wall[2]={9, 9}, first_drop[2]={9, 9};
    for (int side=0; side<2 && walls; side++) {
        float step=side?1.5f:-1.5f;
        for (int k=1; k<=8; k++) if (bopl4f_sdf(s, all, x+step*(float)k, y)<0.3f) { first_wall[side]=k; break; }
    }
    float threshold=drop+4.0f;
    if (water && y>wt && !(y-wt>threshold)) {
        /* water within the threshold below this height: no edge by drop */
    } else if (!walls) {
        if (32.0f>threshold) first_drop[0]=first_drop[1]=1;
    } else {
        float ox[16], oy[16], dx[16], dy[16], limit[16], cap[16], d[16];
        int side_of[16], k_of[16], n=0;
        for (int side=0; side<2; side++) {
            float step=side?1.5f:-1.5f;
            for (int k=1; k<first_wall[side]; k++) {
                ox[n]=x+step*(float)k; oy[n]=y; dx[n]=0.0f; dy[n]=-1.0f; limit[n]=bopl4f_next_up(threshold); cap[n]=32.0f;
                side_of[n]=side; k_of[n]=k; n++;
            }
        }
        bopl4f_march_group(s, a, n, ox, oy, dx, dy, cap, limit, d);
        for (int i=0; i<n; i++) if (d[i]>threshold && k_of[i]<first_drop[side_of[i]]) first_drop[side_of[i]]=k_of[i];
    }
    for (int side=0; side<2; side++) {
        int k=first_wall[side]<first_drop[side]?first_wall[side]:first_drop[side];
        out[2+side]=bopl4f_div(k<=8?1.5f*(float)k:(float)BOPL4_DASH_LENGTH, (float)BOPL4_DASH_LENGTH);
        out[4+side]=first_wall[side]<=8 && first_wall[side]<=first_drop[side];
    }
    out[6]=water?bopl4f_clip(bopl4f_div(y-wt, 32.0f)):1.0f;
    float lethal=16.0f;
    Bopl4Mask lethal_walls={s->axis_lethal, s->rot_lethal};
    if (lethal_walls.axis|lethal_walls.rot) { float d=bopl4f_sdf(s, lethal_walls, x, y); lethal=d<lethal?d:lethal; }
    out[7]=bopl4f_div(lethal, 16.0f);
    /* Ground within one dash: some dash endpoint (16 directions, stopped at
     * terrain) has a surface (or water) within 4 units below it. */
    int landing=drop<=4.0f;
    if (!landing) {
        float ox[16], oy[16], dx[16], dy[16], limit[16], d[16];
        for (int i=0; i<16; i++) { ox[i]=x; oy[i]=y; dx[i]=dirs->dash[i][0]; dy[i]=dirs->dash[i][1]; limit[i]=(float)BOPL4_DASH_LENGTH; }
        bopl4f_march_group(s, a, 16, ox, oy, dx, dy, limit, limit, d);
        float ex[16], ey[16], zero[16], down[16], above[16], cap[16], fall[16];
        int m=0;
        for (int i=0; i<16; i++) {
            float px=x+dx[i]*(d[i]-0.5f), py=y+dy[i]*(d[i]-0.5f);
            if (water && py<=wt) continue;
            if (water && !(py-wt>4.0f)) { landing=1; continue; }
            if (!walls) continue;
            ex[m]=px; ey[m]=py; zero[m]=0.0f; down[m]=-1.0f; above[m]=bopl4f_next_up(4.0f); cap[m]=8.0f; m++;
        }
        if (m && !landing) {
            bopl4f_march_group(s, a, m, ex, ey, zero, down, cap, above, fall);
            for (int i=0; i<m; i++) if (!(fall[i]>4.0f)) landing=1;
        }
    }
    out[8]=landing;
    out[9]=walls?bopl4f_clip(bopl4f_div(bopl4f_sdf(s, all, x, y), 8.0f)):1.0f;
    if (walls) {
        float h=0.05f;
        out[10]=bopl4f_clip(bopl4f_div(bopl4f_sdf(s, all, x+h, y)-bopl4f_sdf(s, all, x-h, y), 2.0f*h));
        out[11]=bopl4f_clip(bopl4f_div(bopl4f_sdf(s, all, x, y+h)-bopl4f_sdf(s, all, x, y-h), 2.0f*h));
    } else out[10]=out[11]=0.0f;
}
/* Layout v4.1 (BOPL4_ABLATE_AIM): in ray slot BOPL4_AIM_RAY the aim ray, cast
 * along the aim of the decision applied this step (zero when it had none); in
 * slot BOPL4_LOS_RAY the line of sight toward the primary enemy's centre
 * (zero when its position is unknown). */
static void bopl4f_aim_rays(const Bopl4Scene *s, const Bopl4Anchor *a, float *rays) {
    float dx[2], dy[2], *out[2]; int n=0;
    float *aim=rays+(size_t)BOPL4_AIM_RAY*BOPL4_RAY_FEATURES, *los=rays+(size_t)BOPL4_LOS_RAY*BOPL4_RAY_FEATURES;
    memset(aim, 0, BOPL4_RAY_FEATURES*sizeof(float)); memset(los, 0, BOPL4_RAY_FEATURES*sizeof(float));
    if (s->flags&BOPL4F_AIMED) { dx[n]=s->aim_dx; dy[n]=s->aim_dy; out[n++]=aim; }
    if (bopl4f_los(s, &dx[n], &dy[n])) out[n++]=los;
    if (n) bopl4f_cast_group(s, a, n, dx, dy, out, NULL);
}
/* The terrain block and the rays (BOPL4_TERRAIN + BOPL4_RAYS*BOPL4_RAY_FEATURES
 * floats from obs+BOPL4_OBS_TERRAIN_OFFSET). */
static void bopl4f_write(const Bopl4Scene *s, const Bopl4Dirs *dirs, float *region) {
    float *rays=region+BOPL4_TERRAIN;
    Bopl4Anchor anchor;
    if (!(s->flags&BOPL4F_FAN)) {
        memset(region, 0, (BOPL4_TERRAIN+BOPL4_RAYS*BOPL4_RAY_FEATURES)*sizeof(float));
        if (s->flags&BOPL4F_AIM) { bopl4f_anchor(s, &anchor); bopl4f_aim_rays(s, &anchor, rays); }
        return;
    }
    bopl4f_anchor(s, &anchor);
    float dx[BOPL4_RAYS], dy[BOPL4_RAYS], *out[BOPL4_RAYS], probes[2];
    int n=0;
    for (int k=0; k<BOPL4_RAYS; k++) {
        float *ray=rays+(size_t)k*BOPL4_RAY_FEATURES;
        if ((s->flags&BOPL4F_HALF) && (k>=BOPL4_RAYS_AROUND || (k&1))) { memset(ray, 0, BOPL4_RAY_FEATURES*sizeof(float)); continue; }
        dx[n]=dirs->fan[k][0]; dy[n]=dirs->fan[k][1]; out[n++]=ray;
    }
    bopl4f_cast_group(s, &anchor, n, dx, dy, out, probes);
    bopl4f_terrain(s, dirs, &anchor, probes, region);
    if (s->flags&BOPL4F_AIM) bopl4f_aim_rays(s, &anchor, rays);
}
#endif

#if defined(__GNUC__) && !defined(__clang__) && !defined(__CUDACC__)
#pragma GCC pop_options
#endif
#endif
