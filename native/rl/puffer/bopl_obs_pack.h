/* Packed transport of the policy v4 observation row between the env
 * workers and the GPU (THROUGHPUT_V4.md, notes-v4.md 2026-09-19).
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
 * in a slot of BOPL4_PACK_P floats per agent row (6 KB against 19 KB). A row
 * whose prefixes do not fit the slot, or whose writer record is missing
 * (right after a routing change), is sent in mode 1: the header alone, and
 * the GPU reads the dense row from the mapped host buffer. Either way the
 * device row the chain reads is bit for bit the host row.
 *
 * Host side: bopl_pack_row (C, called by c_step for its live rows).
 * Device side: bopl4_unpack_rows in bopl_rollout_v4.cu. */
#ifndef BOPL_OBS_PACK_H
#define BOPL_OBS_PACK_H
#include <string.h>
#include "bopl_layout_v4.h"

enum {
    BOPL4_PACK_MODE_ROWS=0,                 /* prefixes in the slot */
    BOPL4_PACK_MODE_DENSE=1,                /* slot holds the header only; read the dense host row */
    BOPL4_PACK_HEADER=8,                    /* words: mode, 4 counts, 3 spare */
    BOPL4_PACK_FIXED=BOPL4_FLAT+BOPL4_PRIVILEGED,             /* 540 */
    BOPL4_PACK_ROWS_OFFSET=BOPL4_PACK_HEADER+BOPL4_PACK_FIXED, /* 548 */
    BOPL4_PACK_P=1536                       /* floats per slot: 988 for the prefixes, ~20 entities with the usual rest */
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
#endif
