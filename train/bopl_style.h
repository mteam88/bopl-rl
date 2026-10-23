/* Style programs (docs/STYLES.md part 3): the stack VM that runs a compiled
 * style (train/style_compile.py) over the ABI state, the step's event stream
 * and per-player history slots, once per learner decision.
 *
 * bopl_style_load copies and verifies a program (the only allocation);
 * bopl_style_step allocates nothing, uses only float arithmetic, and is a pure
 * function of (program, state, events, slots). eval/style_score.py interprets
 * the same bytes in numpy with the same float32 operation order, and
 * eval/style_check.c holds the two to bitwise agreement on the replay dumps. */
#ifndef BOPL_STYLE_H
#define BOPL_STYLE_H

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "bopl_rl.h"

/* The events ABI (docs/STYLES.md part 1) lands with the game-events branch,
 * which defines BOPL_RL_EVENTS in bopl_rl.h. Until then the struct is declared
 * here and the accessors are weak: unresolved, they read as "no events". */
#ifndef BOPL_RL_EVENTS
#define BOPL_RL_EVENTS 1
typedef struct {
    uint32_t kind, tick, subject, actor, entity, source_ability, cause, flags;
    uint32_t related, related_actor, related_tick, creator;
    int64_t x, y, vx, vy;
    int64_t ax, ay, avx, avy;
} BoplRlEvent;
enum BoplRlEventKind { BOPL_EVENT_DEATH=1, BOPL_EVENT_BLINK=2 };
enum BoplRlDeathCause {
    BOPL_DEATH_UNKNOWN=0, BOPL_DEATH_CONTACT=1, BOPL_DEATH_PROJECTILE=2, BOPL_DEATH_EXPLOSION=3,
    BOPL_DEATH_BEAM=4, BOPL_DEATH_SQUISHED=5, BOPL_DEATH_EATEN=6, BOPL_DEATH_RING_OUT=7,
    BOPL_DEATH_SELF=8, BOPL_DEATH_OTHER=9
};
enum BoplRlDeathFlags { BOPL_DEATH_ACTOR_IN_FORM=1, BOPL_DEATH_LAST_BODY=2, BOPL_DEATH_RELATED_BLINKED=4 };
const BoplRlEvent *bopl_rl_events(const BoplRl *world, uint32_t *count) __attribute__((weak));
int bopl_rl_event_support(const BoplRl *world) __attribute__((weak));
#define BOPL_STYLE_WEAK_EVENTS 1
#endif

/* Events of the world's last command; NULL/0 when the backend has none. */
static inline const BoplRlEvent *bopl_style_events(const BoplRl *world, uint32_t *count, int *support) {
    *count=0; *support=0;
#ifdef BOPL_STYLE_WEAK_EVENTS
    if (!bopl_rl_events || !bopl_rl_event_support) return NULL;
#endif
    if (!world) return NULL;
    *support=bopl_rl_event_support(world)!=0;
    const BoplRlEvent *events=bopl_rl_events(world, count);
    if (!events) *count=0;
    return events;
}

enum {
    BS_END, BS_CONST, BS_SELF, BS_OPP, BS_VAR, BS_HEADER, BS_PFIELD, BS_BFIELD, BS_BFIELDI, BS_EVFIELD,
    BS_EFIELD, BS_SFIELD, BS_SFIELDI, BS_ADD, BS_SUB, BS_MUL, BS_DIV, BS_LT, BS_LE, BS_GT, BS_GE, BS_EQ, BS_NE,
    BS_BAND, BS_BOR, BS_NOT, BS_NEG, BS_ABS, BS_SQRT, BS_MIN, BS_MAX, BS_HYPOT, BS_JZ, BS_JNZ, BS_TRUTH,
    BS_READY, BS_HASAB, BS_INCAT, BS_INFORM, BS_FORMAB, BS_PDIST, BS_EDIST, BS_SDIST, BS_HIST, BS_DT,
    BS_EVSUP, BS_AGG, BS_LOOP, BS_TELEM, BS_OP_COUNT,
    /* Load-time rewrites of the private code copy (bs_rewrite), never in
     * style.bin: CONST k fused into the binary operator after it, SELF/OPP
     * fused into the PFIELD/TELEM after it. The fused word skips the word it
     * absorbed, which stays intact for any jump that lands on it. */
    BS_XK_ADD=BS_OP_COUNT, BS_XK_SUB, BS_XK_MUL, BS_XK_DIV, BS_XK_LT, BS_XK_LE, BS_XK_GT, BS_XK_GE, BS_XK_EQ, BS_XK_NE,
    BS_XK_BAND, BS_XK_BOR, BS_XK_MIN, BS_XK_MAX, BS_XS_PFIELD, BS_XO_PFIELD, BS_XS_TELEM, BS_XO_TELEM
};
enum { BS_AGG_COUNT_ENT, BS_AGG_COUNT_SLIME, BS_AGG_COUNT_EV, BS_AGG_MINDIST_ENT, BS_AGG_SOURCES_ENT,
       BS_AGG_COVER_ENT, BS_AGG_MINDIST_SLIME, BS_AGG_KINDS };
enum { BS_LOOP_SUM, BS_LOOP_ANY, BS_LOOP_ALL, BS_LOOP_MAX, BS_LOOP_MIN, BS_LOOP_COUNT, BS_LOOP_MODES };
enum { BS_SLOT_PREV=1, BS_SLOT_SINCE, BS_SLOT_COUNT_OVER, BS_SLOT_MAX_OVER, BS_SLOT_EMA };
#define BS_CAT_ANY 0xFFFFFFu
#define BS_STACK 32
#define BS_VARS 4
#define BS_DEPTH 4          /* chunk nesting: a term or slot, plus aggregates/loops inside */
#define BS_HIDDEN 2         /* per-player history header: [has a decision, its tick] */
#define BS_NEVER 1e9f
#define BS_MAX_TERMS 32

typedef struct { uint32_t kind, offset, pc, cap; float param; } BoplStyleSlot;
/* An entity aggregate's native prefilter, found at load: the leading
 * conjuncts of its filter that only compare one entity field (`owner == self`,
 * `flags & K`, `not flags & K`, `kind == 3`...). A failed check is the
 * filter's 0 (the conjunct's JZ chain ends at END), so the VM is entered only
 * for entities that pass, at the first conjunct not checked natively (`resume`),
 * or not at all when every conjunct was (`all`). Same result, bit for bit. */
#define BS_PRE_CHECKS 6
enum { BS_PRE_EQ, BS_PRE_NE, BS_PRE_ANY, BS_PRE_NONE };
enum { BS_REF_CONST, BS_REF_SELF, BS_REF_OPP, BS_REF_VAR };
typedef struct { uint8_t kind, field, ref, var; uint32_t mask; float value; } BoplStyleCheck;
typedef struct { uint32_t resume; uint8_t n, all; BoplStyleCheck check[BS_PRE_CHECKS]; } BoplStylePre;
typedef struct { uint32_t kind, when, value, reserved; } BoplStyleTerm;
typedef struct {
    uint32_t n_consts, n_code, n_slots, n_terms, n_cats, hist_floats, max_stack;
    float fade_start, fade_end;
    const float *consts;
    const uint64_t *cats;
    const BoplStyleSlot *slots;
    const BoplStyleTerm *terms;
    const uint32_t *code;
    const uint16_t *pre_index;      /* per code word: 1 + its prefilter for an entity AGG, else 0 */
    const BoplStylePre *pre;
    void *storage, *pre_storage;
} BoplStyleProgram;

/* ---- loading ------------------------------------------------------------- */
static int bs_fail(char *error, size_t capacity, const char *message, uint32_t at) {
    if (error && capacity) snprintf(error, capacity, "style program: %s (at %u)", message, at);
    return -1;
}
static int bs_extra(uint32_t op) { return op==BS_AGG || op==BS_LOOP ? 2 : 0; }
static int bs_field_limit(uint32_t op) {
    switch (op) {
    case BS_HEADER: return 7;  case BS_PFIELD: return 13; case BS_BFIELD: case BS_SFIELD: return 12;
    case BS_BFIELDI: case BS_SFIELDI: return 4; case BS_EVFIELD: return 21; case BS_EFIELD: return 23;
    case BS_TELEM: return 9; default: return -1;
    }
}
/* Walks the chunk at pc: argument ranges, stack balance (one value at END),
 * forward jumps that land with a consistent depth, and the chunks it calls. */
static int bs_verify(const BoplStyleProgram *p, uint32_t pc, int level, char *error, size_t capacity) {
    if (level>=BS_DEPTH) return bs_fail(error, capacity, "nesting too deep", pc);
    struct { uint32_t target; int depth; } pending[BS_STACK]; int npending=0, depth=0;
    for (;;) {
        if (pc>=p->n_code) return bs_fail(error, capacity, "code runs past the end", pc);
        for (int i=0; i<npending; i++) if (pending[i].target==pc) {
            if (pending[i].depth!=depth) return bs_fail(error, capacity, "jump lands at a different depth", pc);
            pending[i]=pending[--npending]; i--;
        }
        uint32_t word=p->code[pc], op=word&0xff, arg=word>>8;
        if (op>=BS_OP_COUNT) return bs_fail(error, capacity, "unknown opcode", pc);
        if (pc+(uint32_t)bs_extra(op)>=p->n_code) return bs_fail(error, capacity, "truncated instruction", pc);
        int limit=bs_field_limit(op), pops=0, pushes=1;
        if (limit>=0 && arg>=(uint32_t)limit) return bs_fail(error, capacity, "field out of range", pc);
        switch (op) {
        case BS_END:
            if (depth!=1 || npending) return bs_fail(error, capacity, "unbalanced chunk", pc);
            return 0;
        case BS_CONST: if (arg>=p->n_consts) return bs_fail(error, capacity, "constant out of range", pc); break;
        case BS_SELF: case BS_OPP: case BS_HEADER: case BS_EVFIELD: case BS_EFIELD: case BS_SFIELD: case BS_DT: case BS_EVSUP: break;
        case BS_VAR: if (arg>=BS_VARS) return bs_fail(error, capacity, "loop variable out of range", pc); break;
        case BS_HIST: if (arg>=p->n_slots) return bs_fail(error, capacity, "history slot out of range", pc); break;
        case BS_PFIELD: case BS_BFIELD: case BS_SFIELDI: case BS_TELEM: case BS_NOT: case BS_NEG: case BS_ABS:
        case BS_SQRT: case BS_TRUTH: case BS_INFORM: case BS_FORMAB: case BS_EDIST: case BS_SDIST: pops=1; break;
        case BS_READY: case BS_HASAB: case BS_INCAT:
            if (arg!=BS_CAT_ANY && arg>=p->n_cats) return bs_fail(error, capacity, "category out of range", pc);
            pops=1; break;
        case BS_JZ: case BS_JNZ:
            if (arg<=pc+1 || arg>=p->n_code || npending>=BS_STACK || depth<1) return bs_fail(error, capacity, "bad jump", pc);
            pending[npending].target=arg; pending[npending].depth=depth; npending++;
            pops=1; pushes=0; break;
        case BS_AGG: case BS_LOOP: {
            uint32_t sub=p->code[pc+1], extra=p->code[pc+2];
            if (op==BS_AGG) {
                if (arg>=BS_AGG_KINDS) return bs_fail(error, capacity, "unknown aggregate", pc);
                pops=arg==BS_AGG_MINDIST_ENT || arg==BS_AGG_MINDIST_SLIME ? 1 : arg==BS_AGG_COVER_ENT ? 2 : 0;
                if (arg==BS_AGG_COVER_ENT) {
                    uint32_t sectors=extra&0xff, first=extra>>8;
                    if (sectors<1 || sectors>32 || (sectors>1 && first+sectors-1>p->n_consts))
                        return bs_fail(error, capacity, "bad coverage sectors", pc);
                }
            } else if (arg>=BS_LOOP_MODES || extra>=BS_VARS) return bs_fail(error, capacity, "bad loop", pc);
            if (bs_verify(p, sub, level+1, error, capacity)) return -1;
            break;
        }
        default: pops=2; break;   /* binary operators, BFIELDI, PDIST */
        }
        if (depth<pops) return bs_fail(error, capacity, "stack underflow", pc);
        depth+=pushes-pops;
        if (depth>BS_STACK) return bs_fail(error, capacity, "stack overflow", pc);
        pc+=1+(uint32_t)bs_extra(op);
    }
}
static void bopl_style_free(BoplStyleProgram *p) { if (p) { free(p->storage); free(p->pre_storage); memset(p, 0, sizeof(*p)); } }
static int bs_entity_agg(uint32_t kind);
/* One natively checkable conjunct at pc (a verified chunk): EFIELD f with a
 * SELF/OPP/VAR/CONST under EQ/NE (either order), or EFIELD f and a CONST
 * under BAND, optionally NOT; an optional TRUTH. Returns the next pc or 0. */
static uint32_t bs_pre_conjunct(const BoplStyleProgram *p, uint32_t pc, BoplStyleCheck *out) {
    const uint32_t *code=p->code;
#define BS_OP_AT(i) ((i)<p->n_code ? code[i]&0xff : (uint32_t)BS_OP_COUNT)
    if (pc+3>=p->n_code) return 0;   /* three words and what follows them */
    uint32_t o0=code[pc]&0xff, o1=code[pc+1]&0xff, o2=code[pc+2]&0xff;
    uint32_t a0=code[pc]>>8, a1=code[pc+1]>>8;
    uint32_t field, other, other_arg;
    if (o0==BS_EFIELD) { field=a0; other=o1; other_arg=a1; }
    else if (o1==BS_EFIELD) { field=a1; other=o0; other_arg=a0; }
    else return 0;
    memset(out, 0, sizeof(*out));
    out->field=(uint8_t)field;
    if (o2==BS_EQ || o2==BS_NE) {
        out->kind=o2==BS_EQ?BS_PRE_EQ:BS_PRE_NE;
        switch (other) {
        case BS_CONST: out->ref=BS_REF_CONST; out->value=p->consts[other_arg]; break;
        case BS_SELF: out->ref=BS_REF_SELF; break;
        case BS_OPP: out->ref=BS_REF_OPP; break;
        case BS_VAR: out->ref=BS_REF_VAR; out->var=(uint8_t)other_arg; break;
        default: return 0;
        }
        pc+=3;
    } else if (o2==BS_BAND && other==BS_CONST) {
        float k=p->consts[other_arg];
        out->mask=!(k>=0.0f)?0u:k>=4294967296.0f?0xffffffffu:(uint32_t)k;
        out->kind=BS_PRE_ANY; pc+=3;
        if (BS_OP_AT(pc)==BS_NOT) { out->kind=BS_PRE_NONE; pc++; }
    } else return 0;
    if (BS_OP_AT(pc)==BS_TRUTH) pc++;
    return pc<p->n_code ? pc : 0;
#undef BS_OP_AT
}
static void bs_pre_analyse(const BoplStyleProgram *p, uint32_t sub, BoplStylePre *out) {
    const uint32_t *code=p->code;
    uint32_t pc=sub;
    memset(out, 0, sizeof(*out));
    while (out->n<BS_PRE_CHECKS) {
        BoplStyleCheck check;
        uint32_t next=bs_pre_conjunct(p, pc, &check);
        if (!next) break;
        uint32_t op=code[next]&0xff;
        if (op==BS_END) { out->check[out->n++]=check; out->all=1; pc=next; break; }
        if (op!=BS_JZ) break;
        uint32_t t=code[next]>>8;     /* the false path: JZs keep the 0 down to END */
        while (t<p->n_code && (code[t]&0xff)==BS_JZ && (code[t]>>8)>t) t=code[t]>>8;
        if (t>=p->n_code || (code[t]&0xff)!=BS_END) break;
        out->check[out->n++]=check;
        pc=next+1;
    }
    out->resume=pc;
}
/* Exact rewrites of the verified private code, after the prefilters were
 * read from it (they match the original forms). Positions never move, so
 * execution may start at any instruction (a jump target, a prefilter's
 * resume) and see the original semantics:
 *  - a comparison, NOT or TRUTH followed by TRUTH gets arg 1: it skips that
 *    TRUTH, a no-op on its 0/1 result;
 *  - JZ/JNZ targets are threaded through TRUTH (a no-op on the 0 or 1 a
 *    taken jump leaves) and through jumps of the same sense, which take
 *    that value on in turn;
 *  - CONST k before a binary operator and SELF/OPP before PFIELD/TELEM fuse. */
static int bs_bool_op(uint32_t op) { return (op>=BS_LT && op<=BS_NE) || op==BS_NOT || op==BS_TRUTH; }
static void bs_rewrite(BoplStyleProgram *p) {
    uint32_t *code=(uint32_t *)p->code, n=p->n_code;
    for (uint32_t pc=0; pc<n; pc+=1+(uint32_t)bs_extra(code[pc]&0xff)) {
        uint32_t op=code[pc]&0xff;
        /* Operators read no argument: clear it, then mark the TRUTH skip. */
        if ((op>=BS_ADD && op<=BS_HYPOT) || op==BS_TRUTH) code[pc]=op;
        if (bs_bool_op(op) && pc+1<n && (code[pc+1]&0xff)==BS_TRUTH) code[pc]=op|1u<<8;
    }
    for (uint32_t pc=0; pc<n; pc+=1+(uint32_t)bs_extra(code[pc]&0xff)) {
        uint32_t op=code[pc]&0xff;
        if (op!=BS_JZ && op!=BS_JNZ) continue;
        uint32_t t=code[pc]>>8;
        for (;;) {
            if (t>=n) break;
            uint32_t to=code[t]&0xff;
            if (to==BS_TRUTH) { t++; continue; }
            if (to==op && (code[t]>>8)>t) { t=code[t]>>8; continue; }
            break;
        }
        if (t<n) code[pc]=op|t<<8;
    }
    for (uint32_t pc=0; pc+1<n; pc+=1+(uint32_t)bs_extra(code[pc]&0xff)) {
        uint32_t op=code[pc]&0xff, next=code[pc+1]&0xff, arg=code[pc]>>8;
        if (op==BS_CONST) {
            static const uint8_t fuse[BS_OP_COUNT]={[BS_ADD]=BS_XK_ADD, [BS_SUB]=BS_XK_SUB, [BS_MUL]=BS_XK_MUL, [BS_DIV]=BS_XK_DIV,
                [BS_LT]=BS_XK_LT, [BS_LE]=BS_XK_LE, [BS_GT]=BS_XK_GT, [BS_GE]=BS_XK_GE, [BS_EQ]=BS_XK_EQ, [BS_NE]=BS_XK_NE,
                [BS_BAND]=BS_XK_BAND, [BS_BOR]=BS_XK_BOR, [BS_MIN]=BS_XK_MIN, [BS_MAX]=BS_XK_MAX};
            if (next<BS_OP_COUNT && fuse[next]) code[pc]=fuse[next]|arg<<8;
        } else if ((op==BS_SELF || op==BS_OPP) && (next==BS_PFIELD || next==BS_TELEM)) {
            uint32_t fused=next==BS_PFIELD?(op==BS_SELF?BS_XS_PFIELD:BS_XO_PFIELD):(op==BS_SELF?BS_XS_TELEM:BS_XO_TELEM);
            code[pc]=fused|(code[pc+1]>>8)<<8;
        }
    }
}
static int bs_pre_build(BoplStyleProgram *p) {
    uint32_t count=0;
    for (uint32_t pc=0; pc<p->n_code; pc++)
        if ((p->code[pc]&0xff)==BS_AGG && pc+2<p->n_code && bs_entity_agg(p->code[pc]>>8)) count++;
    if (count>65534) return -1;
    size_t index_bytes=(sizeof(uint16_t)*(size_t)p->n_code+15)&~(size_t)15;
    unsigned char *s=calloc(1, index_bytes+sizeof(BoplStylePre)*(count+1));
    if (!s) return -1;
    uint16_t *index=(uint16_t *)s; BoplStylePre *pre=(BoplStylePre *)(s+index_bytes);
    /* Walk instruction boundaries: AGG/LOOP carry two extra words. */
    uint32_t n=0;
    for (uint32_t pc=0; pc<p->n_code; pc++) {
        uint32_t op=p->code[pc]&0xff;
        if (op==BS_AGG && pc+2<p->n_code && bs_entity_agg(p->code[pc]>>8)) {
            if (p->code[pc+1]<p->n_code) bs_pre_analyse(p, p->code[pc+1], &pre[n]);
            else pre[n].resume=p->code[pc+1];
            index[pc]=(uint16_t)(++n);
        }
        if (op==BS_AGG || op==BS_LOOP) pc+=2;
    }
    p->pre_storage=s; p->pre_index=index; p->pre=pre;
    return 0;
}
static int bopl_style_load(BoplStyleProgram *p, const void *bytes, size_t length, char *error, size_t capacity) {
    memset(p, 0, sizeof(*p));
    if (error && capacity) error[0]=0;
    const unsigned char *b=bytes;
    uint32_t h[9]; float fade[2];
    if (length<48 || memcmp(b, "BSTY", 4)) return bs_fail(error, capacity, "bad magic", 0);
    memcpy(h, b+4, sizeof(h)); memcpy(fade, b+40, sizeof(fade));
    if (h[0]!=1) return bs_fail(error, capacity, "unsupported version", h[0]);
    uint32_t nc=h[1], ncode=h[2], ns=h[3], nt=h[4], ncat=h[5];
    if (nc>1024 || ncode>(1u<<16) || ns>64 || nt>BS_MAX_TERMS || ncat>32 || h[6]>4096 || h[6]<BS_HIDDEN)
        return bs_fail(error, capacity, "program too large", 0);
    size_t need=48+4*(size_t)nc+8*(size_t)ncat+20*(size_t)ns+16*(size_t)nt+4*(size_t)ncode;
    if (length!=need) return bs_fail(error, capacity, "length mismatch", (uint32_t)length);
    /* One aligned copy: consts, cats, slots, terms, code (all 4- or 8-byte fields). */
    size_t cats_at=0, consts_at=8*(size_t)ncat, slots_at=consts_at+4*(size_t)nc,
           terms_at=slots_at+sizeof(BoplStyleSlot)*ns, code_at=terms_at+sizeof(BoplStyleTerm)*nt,
           total=code_at+4*(size_t)ncode+8;
    unsigned char *s=calloc(1, total);
    if (!s) return bs_fail(error, capacity, "out of memory", 0);
    const unsigned char *at=b+48;
    memcpy(s+consts_at, at, 4*(size_t)nc); at+=4*(size_t)nc;
    memcpy(s+cats_at, at, 8*(size_t)ncat); at+=8*(size_t)ncat;
    for (uint32_t i=0; i<ns; i++, at+=20) memcpy(s+slots_at+sizeof(BoplStyleSlot)*i, at, 20);
    memcpy(s+terms_at, at, 16*(size_t)nt); at+=16*(size_t)nt;
    memcpy(s+code_at, at, 4*(size_t)ncode);
    p->storage=s; p->n_consts=nc; p->n_code=ncode; p->n_slots=ns; p->n_terms=nt; p->n_cats=ncat;
    p->hist_floats=h[6]; p->max_stack=h[7]; p->fade_start=fade[0]; p->fade_end=fade[1];
    p->consts=(const float *)(s+consts_at); p->cats=(const uint64_t *)(s+cats_at);
    p->slots=(const BoplStyleSlot *)(s+slots_at); p->terms=(const BoplStyleTerm *)(s+terms_at);
    p->code=(const uint32_t *)(s+code_at);
    for (uint32_t i=0; i<ns; i++) {
        const BoplStyleSlot *slot=&p->slots[i];
        uint32_t size=slot->kind==BS_SLOT_SINCE?2:slot->kind==BS_SLOT_PREV||slot->kind==BS_SLOT_EMA?3:2+2*slot->cap;
        if (slot->kind<BS_SLOT_PREV || slot->kind>BS_SLOT_EMA
            || ((slot->kind==BS_SLOT_COUNT_OVER || slot->kind==BS_SLOT_MAX_OVER) && (slot->cap<1 || slot->cap>1024))
            || slot->offset<BS_HIDDEN || slot->offset+size>p->hist_floats || bs_verify(p, slot->pc, 0, error, capacity)) {
            if (!error || !capacity || !error[0]) bs_fail(error, capacity, "bad history slot", i);
            bopl_style_free(p); return -1;
        }
    }
    for (uint32_t i=0; i<nt; i++) {
        if (bs_verify(p, p->terms[i].when, 0, error, capacity) || bs_verify(p, p->terms[i].value, 0, error, capacity)) {
            bopl_style_free(p); return -1;
        }
    }
    if (bs_pre_build(p)) { bopl_style_free(p); return bs_fail(error, capacity, "out of memory", 0); }
    bs_rewrite(p);
    return 0;
}

/* ---- evaluation ---------------------------------------------------------- */
typedef struct {
    const BoplStyleProgram *p;
    const BoplRlState *s;
    const BoplRlEvent *events, *ev;
    const BoplRlEntity *ent;
    const BoplRlSlime *slime;
    const float *hist;
    uint32_t n_events, self, opp, players, vars[BS_VARS];
    float dt, support;
    const BoplRlSlime *body[6];
    uint8_t body_known[6];
} BoplStyleCtx;

static inline float bs_q(int64_t v) { return (float)v*0x1p-32f; }
static inline float bs_truth(float v) { return v!=0.0f ? 1.0f : 0.0f; }
static inline float bs_hypot(float a, float b) { return sqrtf(a*a+b*b); }
static inline uint32_t bs_u32(float v) {
    if (!(v>=0.0f)) return 0;
    if (v>=4294967296.0f) return 0xffffffffu;
    return (uint32_t)v;
}
/* Player id -> index, or -1 (ids are 1-based: players[i].id == i+1). */
static inline int bs_player(const BoplStyleCtx *c, float v) {
    if (!(v>=1.0f && v<=(float)c->players)) return -1;
    return (int)v-1;
}
static inline float bs_px(const BoplStyleCtx *c, int i) { return i<0?0.0f:bs_q(c->s->players[i].x); }
static inline float bs_py(const BoplStyleCtx *c, int i) { return i<0?0.0f:bs_q(c->s->players[i].y); }
/* The controlling slime, as bopl_controlling_slime picks it (in float): an
 * owned slime with an equipped ability, preferring one that takes input, then
 * the nearest to the player's position, then the first. */
static const BoplRlSlime *bs_body(BoplStyleCtx *c, int i) {
    if (i<0) return NULL;
    if (c->body_known[i]) return c->body[i];
    const BoplRlState *s=c->s; const BoplRlSlime *best=NULL; float best_d=0.0f; int best_pref=0;
    uint32_t owner=(uint32_t)i+1; float px=bs_px(c, i), py=bs_py(c, i);
    for (uint32_t k=0; k<s->header.slime_count; k++) {
        const BoplRlSlime *sl=&s->slimes[k];
        if (sl->owner!=owner) continue;
        int inventory=0;
        for (int slot=0; slot<3; slot++) inventory|=(sl->ability_flags[slot]&1) && sl->abilities[slot]>0 && sl->abilities[slot]<=38;
        if (!inventory) continue;
        int pref=(sl->flags&8)==0;
        float dx=bs_q(sl->x)-px, dy=bs_q(sl->y)-py, d=dx*dx+dy*dy;
        if (!best || pref>best_pref || (pref==best_pref && d<best_d)) { best=sl; best_d=d; best_pref=pref; }
    }
    c->body[i]=best; c->body_known[i]=1;
    return best;
}
static float bs_slime_field(const BoplRlSlime *sl, uint32_t f) {
    if (!sl) return 0.0f;
    switch (f) {
    case 0: return 1.0f;
    case 1: return (float)sl->id;  case 2: return (float)sl->owner; case 3: return (float)sl->flags;
    case 4: return (float)sl->state; case 5: return bs_q(sl->x); case 6: return bs_q(sl->y);
    case 7: return bs_q(sl->vx); case 8: return bs_q(sl->vy); case 9: return bs_q(sl->rotation);
    case 10: return bs_q(sl->scale);
    default: return bs_hypot(bs_q(sl->vx), bs_q(sl->vy));
    }
}
static float bs_slime_array(const BoplRlSlime *sl, uint32_t f, float index) {
    if (!sl || !(index>=0.0f && index<3.0f)) return 0.0f;
    int k=(int)index;
    switch (f) {
    case 0: return (float)sl->abilities[k];
    case 1: return (float)sl->ability_flags[k];
    case 2: return bs_q(sl->cooldowns[k]);
    default: return bs_q(sl->timers[k]);
    }
}
static float bs_entity_field(const BoplRlEntity *e, uint32_t f) {
    if (!e) return 0.0f;
    switch (f) {
    case 0: return (float)e->id; case 1: return (float)e->kind; case 2: return (float)e->layer;
    /* flags: bits 0-23 (a float holds them exactly; the training export's
     * bits 24-25 would round the low bits away), UNDERGROUND as its own field. */
    case 3: return (float)e->owner; case 4: return (float)(e->flags&0x00ffffffu); case 5: return (float)e->hierarchy;
    case 6: return (float)e->reserved[0]; case 7: return (float)e->reserved[1];
    case 8: return bs_q(e->x); case 9: return bs_q(e->y); case 10: return bs_q(e->vx); case 11: return bs_q(e->vy);
    case 12: return bs_q(e->right_x); case 13: return bs_q(e->right_y); case 14: return bs_q(e->up_x);
    case 15: return bs_q(e->up_y); case 16: return bs_q(e->radius); case 17: return bs_q(e->rotation);
    case 18: return bs_q(e->angular_velocity); case 19: return bs_q(e->inverse_mass); case 20: return bs_q(e->clock_rate);
    case 22: return (e->flags&BOPL_ENTITY_UNDERGROUND)?1.0f:0.0f;
    default: return bs_hypot(bs_q(e->vx), bs_q(e->vy));
    }
}
static float bs_event_field(const BoplRlEvent *e, uint32_t f) {
    if (!e) return 0.0f;
    switch (f) {
    case 0: return (float)e->kind; case 1: return (float)e->tick; case 2: return (float)e->subject;
    case 3: return (float)e->actor; case 4: return (float)e->entity; case 5: return (float)e->source_ability;
    case 6: return (float)e->cause; case 7: return (float)e->flags; case 8: return (float)e->related;
    case 9: return (float)e->related_actor; case 10: return (float)e->related_tick;
    case 11: return bs_q(e->x); case 12: return bs_q(e->y); case 13: return bs_q(e->vx); case 14: return bs_q(e->vy);
    case 15: return bs_q(e->ax); case 16: return bs_q(e->ay); case 17: return bs_q(e->avx); case 18: return bs_q(e->avy);
    case 19: return bs_hypot(bs_q(e->vx), bs_q(e->vy));
    default: return bs_hypot(bs_q(e->avx), bs_q(e->avy));
    }
}
static float bs_header_field(const BoplRlHeader *h, uint32_t f) {
    switch (f) {
    case 0: return (float)h->tick; case 1: return (float)h->episode_ticks; case 2: return (float)h->flags;
    case 3: return (float)h->player_count; case 4: return (float)h->entity_count; case 5: return (float)h->slime_count;
    default: return (float)h->sudden_death;
    }
}
static float bs_player_field(const BoplStyleCtx *c, int i, uint32_t f) {
    if (i<0) return 0.0f;
    const BoplRlPlayer *pl=&c->s->players[i];
    switch (f) {
    case 0: return (float)pl->id; case 1: return (float)pl->team; case 2: return (float)pl->flags;
    case 3: return (float)pl->alive_bodies; case 4: return (float)pl->stocks; case 5: return (float)pl->kills;
    case 6: return (float)pl->deaths; case 7: return (float)pl->cause_of_death;
    case 8: return bs_q(pl->x); case 9: return bs_q(pl->y); case 10: return bs_q(pl->scale);
    case 11: return bs_q(pl->invisible_time); default: return bs_q(pl->ability_time);
    }
}
static float bs_telemetry_field(const BoplStyleCtx *c, int i, uint32_t f) {
    if (i<0) return 0.0f;
    const BoplRlTelemetry *t=&c->s->telemetry[i];
    if (f<3) return (float)t->ability_activations[f];
    const uint32_t v[6]={t->arrows_fired, t->arrow_hits, t->deaths_environment, t->deaths_opponent, t->deaths_self, t->deaths_other};
    return (float)v[f-3];
}
static int bs_in_mask(const BoplStyleProgram *p, uint32_t cat, uint32_t ability) {
    if (cat==BS_CAT_ANY) return ability>0;
    return ability<64 && ((p->cats[cat]>>ability)&1u);
}
/* Rational pseudo-angle in [0, 4), monotone in the direction's angle. */
static inline float bs_pseudo_angle(float dx, float dy) {
    float ax=fabsf(dx), ay=fabsf(dy), s=ax+ay;
    if (s==0.0f) return 0.0f;
    if (dy>=0.0f) return dx>=0.0f ? dy/s : 1.0f+ax/s;
    return dx<0.0f ? 2.0f+ay/s : 3.0f+ax/s;
}
static inline int bs_popcount(uint64_t v) { return __builtin_popcountll(v); }

static float bs_eval(BoplStyleCtx *c, uint32_t pc);
static int bs_entity_agg(uint32_t kind) {
    return kind==BS_AGG_COUNT_ENT || kind==BS_AGG_MINDIST_ENT || kind==BS_AGG_SOURCES_ENT || kind==BS_AGG_COVER_ENT;
}
_Static_assert(offsetof(BoplRlEntity, owner)==12 && offsetof(BoplRlEntity, reserved)==24 && offsetof(BoplRlEntity, x)==32,
               "bs_pre_pass reads entity fields 0-7 as consecutive uint32 words");
/* The prefilter with its references resolved for one aggregate call (self,
 * opp and loop variables are fixed while it runs over the entities). */
typedef struct { uint32_t n, kind[BS_PRE_CHECKS], field[BS_PRE_CHECKS], mask[BS_PRE_CHECKS]; float ref[BS_PRE_CHECKS]; } BoplStyleBound;
static inline void bs_pre_bind(const BoplStyleCtx *c, const BoplStylePre *pre, BoplStyleBound *b) {
    b->n=pre->n;
    for (uint32_t i=0; i<pre->n; i++) {
        const BoplStyleCheck *k=&pre->check[i];
        b->kind[i]=k->kind; b->field[i]=k->field; b->mask[i]=k->mask;
        b->ref[i]=k->ref==BS_REF_CONST?k->value:k->ref==BS_REF_SELF?(float)c->self
                 :k->ref==BS_REF_OPP?(float)c->opp:(float)c->vars[k->var];
    }
}
static inline int bs_pre_pass(const BoplStyleBound *b, const BoplRlEntity *e) {
    for (uint32_t i=0; i<b->n; i++) {
        uint32_t f=b->field[i];
        /* Fields 0-7 are the uint32 words (id ... reserved[1]); EFIELD reads them as (float), flags' bits 0-23. */
        float v=f<8?(float)(((const uint32_t *)e)[f]&(f==4?0x00ffffffu:0xffffffffu)):bs_entity_field(e, f);
        switch (b->kind[i]) {
        case BS_PRE_EQ: if (!(v==b->ref[i])) return 0; break;
        case BS_PRE_NE: if (v==b->ref[i]) return 0; break;
        case BS_PRE_ANY: if (!(bs_u32(v)&b->mask[i])) return 0; break;
        default: if (bs_u32(v)&b->mask[i]) return 0; break;
        }
    }
    return 1;
}
static float bs_aggregate(BoplStyleCtx *c, uint32_t kind, uint32_t sub, uint32_t extra, float a, float b,
                          const BoplStylePre *pre) {
    const BoplRlState *s=c->s;
    if (kind==BS_AGG_COUNT_SLIME || kind==BS_AGG_MINDIST_SLIME) {
        const BoplRlSlime *saved=c->slime; float acc=kind==BS_AGG_COUNT_SLIME?0.0f:BS_NEVER;
        int target=bs_player(c, a); float px=bs_px(c, target), py=bs_py(c, target);
        for (uint32_t k=0; k<s->header.slime_count; k++) {
            c->slime=&s->slimes[k];
            if (bs_eval(c, sub)==0.0f) continue;
            if (kind==BS_AGG_COUNT_SLIME) acc+=1.0f;
            else { float d=bs_hypot(bs_q(c->slime->x)-px, bs_q(c->slime->y)-py); if (d<acc) acc=d; }
        }
        c->slime=saved; return acc;
    }
    if (kind==BS_AGG_COUNT_EV) {
        const BoplRlEvent *saved=c->ev; float acc=0.0f;
        for (uint32_t k=0; k<c->n_events; k++) {
            if (extra && c->events[k].kind!=extra) continue;
            c->ev=&c->events[k];
            if (bs_eval(c, sub)!=0.0f) acc+=1.0f;
        }
        c->ev=saved; return acc;
    }
    /* Entities. COVER_ENT pushed (player, radius): a=player, b=radius. */
    const BoplRlEntity *saved=c->ent;
    int target=bs_player(c, a); float px=bs_px(c, target), py=bs_py(c, target);
    float acc=kind==BS_AGG_MINDIST_ENT?BS_NEVER:0.0f; uint64_t bits=0;
    uint32_t sectors=extra&0xff, first=extra>>8;
    uint32_t resume=pre?pre->resume:sub; int native=pre&&pre->all;
    BoplStyleBound bound; bound.n=0;
    if (pre) bs_pre_bind(c, pre, &bound);
    for (uint32_t k=0; k<s->header.entity_count; k++) {
        const BoplRlEntity *e=&s->entities[k];
        if (!bs_pre_pass(&bound, e)) continue;
        c->ent=e;
        if (!native && bs_eval(c, resume)==0.0f) continue;
        switch (kind) {
        case BS_AGG_COUNT_ENT: acc+=1.0f; break;
        case BS_AGG_SOURCES_ENT: if (e->reserved[0]<64) bits|=1ull<<e->reserved[0]; break;
        case BS_AGG_MINDIST_ENT: { float d=bs_hypot(bs_q(e->x)-px, bs_q(e->y)-py); if (d<acc) acc=d; break; }
        default: {
            float dx=bs_q(e->x)-px, dy=bs_q(e->y)-py;
            if (!(bs_hypot(dx, dy)<=b)) break;
            float angle=bs_pseudo_angle(dx, dy); uint32_t sector=0;
            for (uint32_t j=1; j<sectors; j++) sector+=angle>=c->p->consts[first+j-1];
            bits|=1ull<<sector;
        }
        }
    }
    c->ent=saved;
    if (kind==BS_AGG_SOURCES_ENT) return (float)bs_popcount(bits);
    if (kind==BS_AGG_COVER_ENT) return (float)bs_popcount(bits)/(float)sectors;
    return acc;
}
static float bs_loop(BoplStyleCtx *c, uint32_t mode, uint32_t sub, uint32_t var) {
    int self=bs_player(c, (float)c->self);
    uint32_t team=self<0?0xffffffffu:c->s->players[self].team, saved=c->vars[var];
    float acc=mode==BS_LOOP_ALL?1.0f:mode==BS_LOOP_MAX?-BS_NEVER:mode==BS_LOOP_MIN?BS_NEVER:0.0f;
    for (uint32_t i=0; i<c->players; i++) {
        if (c->s->players[i].team==team) continue;
        c->vars[var]=i+1;
        float v=bs_eval(c, sub);
        switch (mode) {
        case BS_LOOP_SUM: acc+=v; break;
        case BS_LOOP_ANY: if (v!=0.0f) acc=1.0f; break;
        case BS_LOOP_ALL: if (v==0.0f) acc=0.0f; break;
        case BS_LOOP_MAX: if (v>acc) acc=v; break;
        case BS_LOOP_MIN: if (v<acc) acc=v; break;
        default: if (v!=0.0f) acc+=1.0f; break;
        }
    }
    c->vars[var]=saved;
    return acc;
}
static float bs_slot_value(const BoplStyleCtx *c, uint32_t index) {
    const BoplStyleSlot *slot=&c->p->slots[index]; const float *h=c->hist+slot->offset;
    float now=(float)c->s->header.episode_ticks;
    switch (slot->kind) {
    case BS_SLOT_PREV: return h[1];
    case BS_SLOT_SINCE: return h[0]!=0.0f ? now-h[1] : BS_NEVER;
    case BS_SLOT_EMA: return h[1];
    default: {
        uint32_t cap=slot->cap, head=(uint32_t)h[0], n=(uint32_t)h[1];
        const float *ticks=h+2, *vals=h+2+cap;
        float acc=slot->kind==BS_SLOT_COUNT_OVER?0.0f:-BS_NEVER;
        for (uint32_t i=0; i<n; i++) {
            uint32_t k=(head+cap-i)%cap;
            if (!(now-ticks[k]<slot->param)) continue;
            if (slot->kind==BS_SLOT_COUNT_OVER) acc+=vals[k];
            else if (vals[k]>acc) acc=vals[k];
        }
        return acc;
    }
    }
}
static float bs_eval(BoplStyleCtx *c, uint32_t pc) {
    const uint32_t *code=c->p->code;
    float stack[BS_STACK]; int sp=0;
#define BS_POP() (stack[--sp])
#define BS_TOP() (stack[sp-1])
    for (;;) {
        uint32_t word=code[pc], op=word&0xff, arg=word>>8;
        pc++;
        switch (op) {
        case BS_END: return stack[0];
        case BS_CONST: stack[sp++]=c->p->consts[arg]; break;
        case BS_SELF: stack[sp++]=(float)c->self; break;
        case BS_OPP: stack[sp++]=(float)c->opp; break;
        case BS_VAR: stack[sp++]=(float)c->vars[arg]; break;
        case BS_HEADER: stack[sp++]=bs_header_field(&c->s->header, arg); break;
        case BS_PFIELD: BS_TOP()=bs_player_field(c, bs_player(c, BS_TOP()), arg); break;
        case BS_TELEM: BS_TOP()=bs_telemetry_field(c, bs_player(c, BS_TOP()), arg); break;
        case BS_BFIELD: BS_TOP()=bs_slime_field(bs_body(c, bs_player(c, BS_TOP())), arg); break;
        case BS_BFIELDI: { float index=BS_POP(); BS_TOP()=bs_slime_array(bs_body(c, bs_player(c, BS_TOP())), arg, index); break; }
        case BS_EVFIELD: stack[sp++]=bs_event_field(c->ev, arg); break;
        case BS_EFIELD: stack[sp++]=bs_entity_field(c->ent, arg); break;
        case BS_SFIELD: stack[sp++]=bs_slime_field(c->slime, arg); break;
        case BS_SFIELDI: BS_TOP()=bs_slime_array(c->slime, arg, BS_TOP()); break;
        case BS_ADD: { float b=BS_POP(); BS_TOP()=BS_TOP()+b; break; }
        case BS_SUB: { float b=BS_POP(); BS_TOP()=BS_TOP()-b; break; }
        case BS_MUL: { float b=BS_POP(); BS_TOP()=BS_TOP()*b; break; }
        case BS_DIV: { float b=BS_POP(); BS_TOP()=BS_TOP()/b; break; }
        case BS_LT: { float b=BS_POP(); BS_TOP()=BS_TOP()<b?1.0f:0.0f; pc+=arg; break; }
        case BS_LE: { float b=BS_POP(); BS_TOP()=BS_TOP()<=b?1.0f:0.0f; pc+=arg; break; }
        case BS_GT: { float b=BS_POP(); BS_TOP()=BS_TOP()>b?1.0f:0.0f; pc+=arg; break; }
        case BS_GE: { float b=BS_POP(); BS_TOP()=BS_TOP()>=b?1.0f:0.0f; pc+=arg; break; }
        case BS_EQ: { float b=BS_POP(); BS_TOP()=BS_TOP()==b?1.0f:0.0f; pc+=arg; break; }
        case BS_NE: { float b=BS_POP(); BS_TOP()=BS_TOP()!=b?1.0f:0.0f; pc+=arg; break; }
#define BS_XK(OP, EXPR) case OP: { float a=BS_TOP(), b=c->p->consts[arg]; (void)a; BS_TOP()=(EXPR); pc+=1+(code[pc]>>8); break; }
        BS_XK(BS_XK_ADD, a+b) BS_XK(BS_XK_SUB, a-b) BS_XK(BS_XK_MUL, a*b) BS_XK(BS_XK_DIV, a/b)
        BS_XK(BS_XK_LT, a<b?1.0f:0.0f) BS_XK(BS_XK_LE, a<=b?1.0f:0.0f) BS_XK(BS_XK_GT, a>b?1.0f:0.0f)
        BS_XK(BS_XK_GE, a>=b?1.0f:0.0f) BS_XK(BS_XK_EQ, a==b?1.0f:0.0f) BS_XK(BS_XK_NE, a!=b?1.0f:0.0f)
        BS_XK(BS_XK_BAND, (float)(bs_u32(a)&bs_u32(b))) BS_XK(BS_XK_BOR, (float)(bs_u32(a)|bs_u32(b)))
        BS_XK(BS_XK_MIN, a<b?a:b) BS_XK(BS_XK_MAX, a>b?a:b)
#undef BS_XK
        case BS_XS_PFIELD: stack[sp++]=bs_player_field(c, bs_player(c, (float)c->self), arg); pc++; break;
        case BS_XO_PFIELD: stack[sp++]=bs_player_field(c, bs_player(c, (float)c->opp), arg); pc++; break;
        case BS_XS_TELEM: stack[sp++]=bs_telemetry_field(c, bs_player(c, (float)c->self), arg); pc++; break;
        case BS_XO_TELEM: stack[sp++]=bs_telemetry_field(c, bs_player(c, (float)c->opp), arg); pc++; break;
        case BS_BAND: { float b=BS_POP(); BS_TOP()=(float)(bs_u32(BS_TOP())&bs_u32(b)); break; }
        case BS_BOR: { float b=BS_POP(); BS_TOP()=(float)(bs_u32(BS_TOP())|bs_u32(b)); break; }
        case BS_NOT: BS_TOP()=BS_TOP()==0.0f?1.0f:0.0f; pc+=arg; break;
        case BS_NEG: BS_TOP()=-BS_TOP(); break;
        case BS_ABS: BS_TOP()=fabsf(BS_TOP()); break;
        case BS_SQRT: BS_TOP()=sqrtf(BS_TOP()); break;
        case BS_MIN: { float b=BS_POP(), a=BS_TOP(); BS_TOP()=a<b?a:b; break; }
        case BS_MAX: { float b=BS_POP(), a=BS_TOP(); BS_TOP()=a>b?a:b; break; }
        case BS_HYPOT: { float b=BS_POP(); BS_TOP()=bs_hypot(BS_TOP(), b); break; }
        case BS_JZ: if (BS_TOP()==0.0f) { BS_TOP()=0.0f; pc=arg; } else sp--; break;
        case BS_JNZ: if (BS_TOP()!=0.0f) { BS_TOP()=1.0f; pc=arg; } else sp--; break;
        case BS_TRUTH: BS_TOP()=bs_truth(BS_TOP()); pc+=arg; break;
        case BS_READY: case BS_HASAB: {
            const BoplRlSlime *sl=bs_body(c, bs_player(c, BS_TOP())); float n=0.0f;
            if (sl) for (int k=0; k<3; k++)
                if (bs_in_mask(c->p, arg, sl->abilities[k]) && (op==BS_HASAB || (sl->ability_flags[k]&2))) n+=1.0f;
            BS_TOP()=n; break;
        }
        case BS_INCAT: { float v=BS_TOP(); BS_TOP()=(v>=1.0f && v<64.0f && bs_in_mask(c->p, arg, (uint32_t)v))?1.0f:0.0f; break; }
        case BS_INFORM: case BS_FORMAB: {
            const BoplRlSlime *sl=bs_body(c, bs_player(c, BS_TOP())); float v=0.0f;
            if (sl) for (int k=0; k<3; k++) if (sl->ability_flags[k]&8) { v=op==BS_INFORM?1.0f:(float)sl->abilities[k]; break; }
            BS_TOP()=v; break;
        }
        case BS_PDIST: {
            int b=bs_player(c, BS_POP()), a=bs_player(c, BS_TOP());
            BS_TOP()=bs_hypot(bs_px(c, a)-bs_px(c, b), bs_py(c, a)-bs_py(c, b)); break;
        }
        case BS_EDIST: case BS_SDIST: {
            int a=bs_player(c, BS_TOP());
            float x=op==BS_EDIST?(c->ent?bs_q(c->ent->x):0.0f):(c->slime?bs_q(c->slime->x):0.0f);
            float y=op==BS_EDIST?(c->ent?bs_q(c->ent->y):0.0f):(c->slime?bs_q(c->slime->y):0.0f);
            BS_TOP()=bs_hypot(x-bs_px(c, a), y-bs_py(c, a)); break;
        }
        case BS_HIST: stack[sp++]=bs_slot_value(c, arg); break;
        case BS_DT: stack[sp++]=c->dt; break;
        case BS_EVSUP: stack[sp++]=c->support; break;
        case BS_AGG: {
            uint32_t sub=code[pc], extra=code[pc+1]; pc+=2;
            float a=0.0f, b=0.0f;
            if (arg==BS_AGG_COVER_ENT) { b=BS_POP(); a=BS_POP(); }
            else if (arg==BS_AGG_MINDIST_ENT || arg==BS_AGG_MINDIST_SLIME) a=BS_POP();
            uint32_t pre=c->p->pre_index?c->p->pre_index[pc-3]:0;
            stack[sp++]=bs_aggregate(c, arg, sub, extra, a, b, pre?&c->p->pre[pre-1]:NULL); break;
        }
        case BS_LOOP: { uint32_t sub=code[pc], var=code[pc+1]; pc+=2; stack[sp++]=bs_loop(c, arg, sub, var); break; }
        default: return 0.0f;   /* unreachable: verified */
        }
    }
#undef BS_POP
#undef BS_TOP
}
/* The nearest living opponent (player positions, float distance, lowest index
 * on ties); the nearest of any when none lives; 0 without opponents. */
static uint32_t bs_nearest_opponent(BoplStyleCtx *c, int self) {
    uint32_t best=0; int best_alive=0; float best_d=0.0f;
    float sx=bs_px(c, self), sy=bs_py(c, self);
    for (uint32_t i=0; i<c->players; i++) {
        const BoplRlPlayer *pl=&c->s->players[i];
        if ((int)i==self || pl->team==c->s->players[self].team) continue;
        int alive=(pl->flags&1)!=0;
        float dx=bs_q(pl->x)-sx, dy=bs_q(pl->y)-sy, d=dx*dx+dy*dy;
        if (!best || alive>best_alive || (alive==best_alive && d<best_d)) { best=i+1; best_alive=alive; best_d=d; }
    }
    return best;
}

/* One decision for player index `player`: advance its history slots (`slots`,
 * program->hist_floats floats, zeroed at episode start), then sum the terms.
 * Event terms run once per event in order; decision terms once. `terms`
 * (n_terms floats, may be NULL) receives each term's payout. */
static float bopl_style_step(const BoplStyleProgram *p, const BoplRlState *s, const BoplRlEvent *events,
                             uint32_t n_events, int event_support, int player, float *slots, float *terms) {
    BoplStyleCtx c;
    memset(&c, 0, sizeof(c));
    c.p=p; c.s=s; c.events=events; c.n_events=events?n_events:0; c.hist=slots;
    c.players=s->header.player_count<6?s->header.player_count:6;
    c.support=event_support?1.0f:0.0f;
    if (player<0 || (uint32_t)player>=c.players) return 0.0f;
    c.self=(uint32_t)player+1;
    c.opp=bs_nearest_opponent(&c, player);
    float now=(float)s->header.episode_ticks;
    c.dt=slots[0]!=0.0f ? now-slots[1] : now;
    for (uint32_t k=0; k<p->n_slots; k++) {
        const BoplStyleSlot *slot=&p->slots[k];
        float v=bs_eval(&c, slot->pc), *h=slots+slot->offset;
        switch (slot->kind) {
        case BS_SLOT_PREV: h[1]=h[0]!=0.0f?h[2]:v; h[2]=v; h[0]=1.0f; break;
        case BS_SLOT_SINCE: if (v!=0.0f) { h[0]=1.0f; h[1]=now; } break;
        case BS_SLOT_EMA:
            if (h[0]==0.0f) h[1]=v;
            else { float a=c.dt/(slot->param+c.dt); h[1]=h[1]+a*(v-h[1]); }
            h[0]=1.0f; break;
        default: {
            uint32_t cap=slot->cap, head=((uint32_t)h[0]+1)%cap, n=(uint32_t)h[1];
            h[2+head]=now;
            h[2+cap+head]=slot->kind==BS_SLOT_COUNT_OVER?bs_truth(v):v;
            h[0]=(float)head; h[1]=(float)(n<cap?n+1:cap);
        }
        }
    }
    float total=0.0f;
    for (uint32_t t=0; t<p->n_terms; t++) {
        const BoplStyleTerm *term=&p->terms[t];
        float pay=0.0f;
        if (term->kind==0) {
            c.ev=NULL;
            if (bs_eval(&c, term->when)!=0.0f) pay=bs_eval(&c, term->value);
        } else for (uint32_t k=0; k<c.n_events; k++) {
            if (events[k].kind!=term->kind) continue;
            c.ev=&events[k];
            if (bs_eval(&c, term->when)!=0.0f) pay+=bs_eval(&c, term->value);
        }
        c.ev=NULL;
        if (terms) terms[t]=pay;
        total+=pay;
    }
    slots[0]=1.0f; slots[1]=now;
    return total;
}
#endif
