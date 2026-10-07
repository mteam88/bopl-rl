/* Packed transport of the policy v4 observation row between the env
 * workers and the GPU (2026-09-19).
 *
 * The dense row is BOPL_OBS_STRIDE = 4812 floats of which the four set blocks
 * are 4272 floats of slot capacity (6 players, 24 slots, 48 entities, 32
 * slimes) against a handful of present rows (at most 2 / 6 / 11 / 2 seen in
 * training). The writer fills each set as a prefix of its slots and zeroes
 * the slots behind the prefix (bopl_encode_v4_row's bopl_written record), so
 * the row is exactly: the flat block, the four prefixes, zeros, the
 * privileged block. The pack keeps that and nothing else:
 *
 *   header  BOPL4_PACK_HEADER words: mode, then the four prefix lengths
 *   fixed   BOPL4_FLAT flat floats then BOPL4_PRIVILEGED privileged floats
 *   rows    the prefixes, set by set, each row its full width
 *
 * in a slot of BOPL4_PACK_P floats per agent row (8 KB against 19 KB). A row
 * whose prefixes do not fit the slot, or whose writer record is missing
 * (right after a routing change), is sent in mode 1: the header alone, and
 * the GPU reads the dense row from the mapped host buffer. Either way the
 * device row the chain reads is bit for bit the host row.
 *
 * Modes 2 and 3 (2026-10-06) are 0 and 1 for a row whose terrain block and
 * rays the GPU writes (bopl_rays_v4.cu): the host leaves them out and the
 * slot carries the row's ray scene (bopl_rays_v4.h, at hdr[5]) instead; mode
 * 2 also drops them from the fixed block. The rays kernel runs after the
 * unpack, so the device row is the host row as the CPU writer would leave it.
 *
 * Host side: bopl_pack_row (C, called by c_step for its live rows).
 * Device side: bopl4_unpack_rows in bopl_rollout_v4.cu. */
#ifndef BOPL_OBS_PACK_H
#define BOPL_OBS_PACK_H
#include <string.h>
#include "bopl_layout_v4.h"
#include "bopl_rays_v4.h"

enum {
    BOPL4_PACK_MODE_ROWS=0,                 /* prefixes in the slot */
    BOPL4_PACK_MODE_DENSE=1,                /* slot holds the header only; read the dense host row */
    BOPL4_PACK_MODE_SCENE=2,                /* mode 0 without the terrain and rays, then the scene (hdr[5]: its offset) */
    BOPL4_PACK_MODE_DENSE_SCENE=3,          /* mode 1, the scene after the header (hdr[5]) */
    BOPL4_PACK_MODE_STATE=4,                /* the row's delayed state; the GPU writes the row (bopl_rows_v4.h Bopl4StateHead) */
    BOPL4_PACK_HEADER=8,                    /* words: mode, 4 counts, 3 spare */
    BOPL4_PACK_FIXED=BOPL4_FLAT+BOPL4_PRIVILEGED,             /* 540 */
    BOPL4_PACK_ROWS_OFFSET=BOPL4_PACK_HEADER+BOPL4_PACK_FIXED, /* 548 */
    BOPL4_PACK_P=2048,                      /* floats per slot: the prefixes (to ~1600 with 23 entities) and a scene (to ~420) */
    BOPL4_PACK_SCENE_FIXED=BOPL4_OBS_TERRAIN_OFFSET+BOPL4_PRIVILEGED,             /* 240 */
    BOPL4_PACK_SCENE_ROWS_OFFSET=BOPL4_PACK_HEADER+BOPL4_PACK_SCENE_FIXED         /* 248 */
};
/* The four sets in row order: offset in the dense row, row width, slots. */
#define BOPL4_PACK_SETS 4
#define BOPL4_PACK_TABLE { \
    {BOPL4_OBS_PLAYER_OFFSET, BOPL4_PLAYER_FEATURES, BOPL4_PLAYERS}, \
    {BOPL4_OBS_SLOT_OFFSET, BOPL4_SLOT_FEATURES, BOPL4_SLOTS}, \
    {BOPL4_OBS_ENTITY_OFFSET, BOPL4_ENTITY_FEATURES, BOPL4_ENTITIES}, \
    {BOPL4_OBS_SLIME_OFFSET, BOPL4_SLIME_FEATURES, BOPL4_SLIMES} }
typedef struct { int offset, width, slots; } Bopl4PackSet;

/* Pack one dense v4 row given its prefix lengths (players, slots, entities,
 * slimes; a negative count = no record, dense mode). Returns the mode. */
static inline int bopl_pack_row(const float *obs, const int counts[4], float *slot) {
    static const Bopl4PackSet sets[BOPL4_PACK_SETS]=BOPL4_PACK_TABLE;
    unsigned *hdr=(unsigned*)slot;
    int total=0;
    for (int s=0; s<BOPL4_PACK_SETS; s++) {
        if (counts[s]<0 || counts[s]>sets[s].slots) { hdr[0]=BOPL4_PACK_MODE_DENSE; return BOPL4_PACK_MODE_DENSE; }
        total+=counts[s]*sets[s].width;
    }
    if (BOPL4_PACK_ROWS_OFFSET+total>BOPL4_PACK_P) { hdr[0]=BOPL4_PACK_MODE_DENSE; return BOPL4_PACK_MODE_DENSE; }
    hdr[0]=BOPL4_PACK_MODE_ROWS;
    for (int s=0; s<BOPL4_PACK_SETS; s++) hdr[1+s]=(unsigned)counts[s];
    hdr[5]=hdr[6]=hdr[7]=0;
    memcpy(slot+BOPL4_PACK_HEADER, obs, BOPL4_FLAT*sizeof(float));
    memcpy(slot+BOPL4_PACK_HEADER+BOPL4_FLAT, obs+BOPL4_OBS_PRIVILEGED_OFFSET, BOPL4_PRIVILEGED*sizeof(float));
    float *out=slot+BOPL4_PACK_ROWS_OFFSET;
    for (int s=0; s<BOPL4_PACK_SETS; s++) {
        size_t n=(size_t)counts[s]*sets[s].width;
        memcpy(out, obs+sets[s].offset, n*sizeof(float));
        out+=n;
    }
    return BOPL4_PACK_MODE_ROWS;
}
/* Mode 2: the GPU writes the row's terrain block and rays from the scene
 * (bopl_rays_v4.h, bopl4_rays_rows): the flat block before them
 * (BOPL4_OBS_TERRAIN_OFFSET floats; terrain and rays end the flat block), the
 * privileged block, the prefixes, the scene. Returns 0 and writes nothing when
 * it does not fit (or a count is missing). */
static inline int bopl_pack_row_scene(const float *obs, const int counts[4], const Bopl4Scene *scene, float *slot) {
    static const Bopl4PackSet sets[BOPL4_PACK_SETS]=BOPL4_PACK_TABLE;
    int total=0;
    for (int s=0; s<BOPL4_PACK_SETS; s++) {
        if (counts[s]<0 || counts[s]>sets[s].slots) return 0;
        total+=counts[s]*sets[s].width;
    }
    if (BOPL4_PACK_SCENE_ROWS_OFFSET+total+bopl4f_scene_words(scene)>BOPL4_PACK_P) return 0;
    unsigned *hdr=(unsigned*)slot;
    hdr[0]=BOPL4_PACK_MODE_SCENE;
    for (int s=0; s<BOPL4_PACK_SETS; s++) hdr[1+s]=(unsigned)counts[s];
    hdr[5]=(unsigned)(BOPL4_PACK_SCENE_ROWS_OFFSET+total); hdr[6]=hdr[7]=0;
    memcpy(slot+BOPL4_PACK_HEADER, obs, BOPL4_OBS_TERRAIN_OFFSET*sizeof(float));
    memcpy(slot+BOPL4_PACK_HEADER+BOPL4_OBS_TERRAIN_OFFSET, obs+BOPL4_OBS_PRIVILEGED_OFFSET, BOPL4_PRIVILEGED*sizeof(float));
    float *out=slot+BOPL4_PACK_SCENE_ROWS_OFFSET;
    for (int s=0; s<BOPL4_PACK_SETS; s++) {
        size_t n=(size_t)counts[s]*sets[s].width;
        memcpy(out, obs+sets[s].offset, n*sizeof(float));
        out+=n;
    }
    bopl4f_scene_write(scene, out);
    return 1;
}
/* Mode 3: a row mode 2 does not fit; the scene (at most ~420 words) always does. */
static inline void bopl_pack_row_dense_scene(const Bopl4Scene *scene, float *slot) {
    unsigned *hdr=(unsigned*)slot;
    hdr[0]=BOPL4_PACK_MODE_DENSE_SCENE;
    for (int i=1; i<BOPL4_PACK_HEADER; i++) hdr[i]=0;
    hdr[5]=BOPL4_PACK_HEADER;
    bopl4f_scene_write(scene, slot+BOPL4_PACK_HEADER);
}
#endif
