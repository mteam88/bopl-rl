/* CPU reference forward for policy v4 (eval/policy_v4.py is the torch
 * twin, train/bopl_v4.cu the trainer).
 * Include after upstream puffernet.h. Weights are the trainer's flat dump read
 * back to back (every tensor is a multiple of 4 floats, bopl_v4.cu
 * reg_aligned): per set W1 (inner, features) and W2 (set_dim, inner), the
 * embedding (ids, inner), flat (flat_dim, BOPL4_FLAT), fusion (hidden, flat_dim
 * + 8 set_dim), the MinGRU layers (upstream make_mingru), then ffn_in / ffn_out
 * (when ffn_dim), actor (width, hidden), critic_in (critic_dim, hidden [+ 48]),
 * critic_b1, critic_b2 (critic_dim, critic_dim), critic_out (critic_dim).
 * fp32 throughout: the trainer runs the row MLPs in bf16, so the two agree to
 * bf16 rounding, as the v3 CPU path agrees with the TF32 trainer. */
#ifndef BOPL_V4_EVAL_H
#define BOPL_V4_EVAL_H
#include <math.h>
#include <string.h>
#include "bopl_layout_v4.h"

typedef struct {
    int set_dim, set_inner, flat_dim, ffn_dim, critic_dim, critic_privileged, critic_detach, mean_pool;
} BoplV4Shape;
static const BoplV4Shape BOPL_V4_DEFAULT_SHAPE={BOPL4_SET_DIM, BOPL4_SET_INNER, BOPL4_FLAT_DIM, BOPL4_FFN_DIM, BOPL4_CRITIC_DIM, 1, 0, 1};

typedef struct { int offset, rows, width, id_column, features; } BoplV4Set;
static const BoplV4Set BOPL_V4_SETS[4]={
    {BOPL4_OBS_PLAYER_OFFSET, BOPL4_PLAYERS, BOPL4_PLAYER_FEATURES, -1, BOPL4_PLAYER_FEATURES},
    {BOPL4_OBS_SLOT_OFFSET, BOPL4_SLOTS, BOPL4_SLOT_FEATURES, BOPL4_SLOT_ABILITY_ID, BOPL4_SLOT_FEATURES-1},
    {BOPL4_OBS_ENTITY_OFFSET, BOPL4_ENTITIES, BOPL4_ENTITY_FEATURES, BOPL4_ENTITY_ABILITY_ID, BOPL4_ENTITY_FEATURES-1},
    {BOPL4_OBS_SLIME_OFFSET, BOPL4_SLIMES, BOPL4_SLIME_FEATURES, -1, BOPL4_SLIME_FEATURES},
};

typedef struct {
    BoplV4Shape shape;
    int batch, hidden, width;
    const float *w1[4], *w2[4], *embedding, *flat, *fusion, *ffn_in, *ffn_out, *actor, *critic_in, *critic_b1, *critic_b2, *critic_out;
    float *joined, *fused, *h1, *h2, *y, *ln, *f, *zin, *z, *b, *z2;   /* scratch */
    /* bopl_v4_fast: column-major copies of the matrices above (NULL until then) and
     * whether the critic runs (the value column is 0 when it does not). */
    float *t_w1[4], *t_w2[4], *t_flat, *t_fusion, *t_ffn_in, *t_ffn_out, *t_actor, *t_critic_in, *t_critic_b1, *t_critic_b2;
    int skip_value;
} BoplV4Eval;

static int bopl_v4_shape_valid(const BoplV4Shape *s) {
    return s->set_dim>=8 && s->set_inner>=8 && s->flat_dim>=8 && s->ffn_dim>=0 && s->critic_dim>=0
        && (s->critic_privileged==0 || s->critic_privileged==1) && (s->mean_pool==0 || s->mean_pool==1);
}
/* Encoder + decoder weights (without the MinGRU layers), for the aim head's decoder width. */
static size_t bopl_v4_weight_count(const BoplV4Shape *s, int hidden, int width) {
    size_t n=0;
    for (int i=0; i<4; i++) n+=(size_t)s->set_inner*BOPL_V4_SETS[i].features+(size_t)s->set_dim*s->set_inner;
    n+=(size_t)BOPL_EMBEDDING_IDS*s->set_inner+(size_t)s->flat_dim*BOPL4_FLAT+(size_t)hidden*(s->flat_dim+8*s->set_dim);
    if (s->ffn_dim) n+=2*(size_t)s->ffn_dim*hidden;
    n+=(size_t)(width+(s->critic_dim?0:1))*hidden;   /* cdim 0: the value is the actor's last row */
    if (s->critic_dim) n+=(size_t)s->critic_dim*(hidden+(s->critic_privileged?BOPL4_PRIVILEGED:0))+2*(size_t)s->critic_dim*s->critic_dim+(size_t)s->critic_dim;
    return n;
}
/* Reads the encoder tensors from `weights` (the MinGRU is read by the caller
 * with make_mingru in between), then bopl_v4_read_decoder the rest. */
static BoplV4Eval *bopl_v4_create(Weights *weights, int batch, int hidden, int width, const BoplV4Shape *shape) {
    if (!weights || batch<1 || hidden<8 || width<1 || !bopl_v4_shape_valid(shape)) return NULL;
    BoplV4Eval *e=calloc(1, sizeof(*e));
    if (!e) return NULL;
    e->shape=*shape; e->batch=batch; e->hidden=hidden; e->width=width;
    int D=shape->set_dim, I=shape->set_inner, C=shape->critic_dim, W=hidden+(shape->critic_privileged?BOPL4_PRIVILEGED:0);
    size_t joined=(size_t)shape->flat_dim+8*D;
    e->joined=calloc((size_t)batch*joined, sizeof(float)); e->fused=calloc((size_t)batch*hidden, sizeof(float));
    e->h1=calloc(I, sizeof(float)); e->h2=calloc(D, sizeof(float));
    /* ln: (batch, hidden) pre-norm fusion rows in the encoder; one row of up to
     * max(W, hidden, ffn_dim, critic_dim) in the decoder. */
    size_t ln=(size_t)batch*(W>hidden?W:hidden);
    if ((size_t)shape->ffn_dim>ln) ln=shape->ffn_dim;
    if ((size_t)C>ln) ln=C;
    e->y=calloc((size_t)batch*hidden, sizeof(float)); e->ln=calloc(ln, sizeof(float));
    e->f=calloc((size_t)(shape->ffn_dim?shape->ffn_dim:1), sizeof(float));
    e->zin=calloc(W, sizeof(float)); e->z=calloc((size_t)(C?C:1), sizeof(float)); e->b=calloc((size_t)(C?C:1), sizeof(float)); e->z2=calloc((size_t)(C?C:1), sizeof(float));
    if (!e->joined || !e->fused || !e->h1 || !e->h2 || !e->y || !e->ln || !e->f || !e->zin || !e->z || !e->b || !e->z2) { free(e); return NULL; }
    for (int i=0; i<4; i++) {
        e->w1[i]=get_weights(weights, I*BOPL_V4_SETS[i].features);
        e->w2[i]=get_weights(weights, D*I);
    }
    e->embedding=get_weights(weights, BOPL_EMBEDDING_IDS*I);
    e->flat=get_weights(weights, shape->flat_dim*BOPL4_FLAT);
    e->fusion=get_weights(weights, hidden*(int)joined);
    return e;
}
static void bopl_v4_read_decoder(BoplV4Eval *e, Weights *weights) {
    int H=e->hidden, C=e->shape.critic_dim, W=H+(e->shape.critic_privileged?BOPL4_PRIVILEGED:0);
    if (e->shape.ffn_dim) { e->ffn_in=get_weights(weights, e->shape.ffn_dim*H); e->ffn_out=get_weights(weights, H*e->shape.ffn_dim); }
    e->actor=get_weights(weights, (e->width+(C?0:1))*H);   /* cdim 0: the value is the actor's last row */
    if (C) {
        e->critic_in=get_weights(weights, C*W); e->critic_b1=get_weights(weights, C*C);
        e->critic_b2=get_weights(weights, C*C); e->critic_out=get_weights(weights, C);
    }
}
static void bopl_v4_free(BoplV4Eval *e) {
    if (!e) return;
    free(e->joined); free(e->fused); free(e->h1); free(e->h2); free(e->y); free(e->ln); free(e->f);
    free(e->zin); free(e->z); free(e->b); free(e->z2);
    for (int i=0; i<4; i++) { free(e->t_w1[i]); free(e->t_w2[i]); }
    free(e->t_flat); free(e->t_fusion); free(e->t_ffn_in); free(e->t_ffn_out); free(e->t_actor);
    free(e->t_critic_in); free(e->t_critic_b1); free(e->t_critic_b2); free(e);
}

/* out[o] = sum_i W[o][i] x[i] (W row-major (rows, cols)), fixed order. */
static void bopl_v4_matvec(float *out, const float *W, const float *x, int rows, int cols) {
    for (int o=0; o<rows; o++) {
        const float *w=W+(size_t)o*cols; float sum=0.0f;
        for (int i=0; i<cols; i++) sum+=w[i]*x[i];
        out[o]=sum;
    }
}
/* bopl_v4_matvec from WT = W transposed (cols, rows): every out[o] still adds its
 * products in order i = 0..cols-1, so the result is bit-identical, but the rows run
 * side by side (vectorised) instead of as one serial add chain each. A column whose
 * x[i] is +-0 is skipped: with finite weights its products are +-0, and adding +-0
 * leaves a sum that started at +0 unchanged (round-to-nearest never yields -0 there).
 * Must not be compiled with -ffast-math or FMA contraction. */
static inline void bopl_v4_matvec_t(float *restrict out, const float *restrict WT, const float *restrict x, int rows, int cols) {
    for (int o=0; o<rows; o++) out[o]=0.0f;
    for (int i=0; i<cols; i++) {
        float xi=x[i];
        if (xi==0.0f) continue;
        const float *w=WT+(size_t)i*rows;
        for (int o=0; o<rows; o++) out[o]+=w[o]*xi;
    }
}
static inline float *bopl_v4_transpose(const float *W, int rows, int cols) {
    float *t=malloc((size_t)rows*cols*sizeof(float));
    if (t) for (int o=0; o<rows; o++) for (int i=0; i<cols; i++) t[(size_t)i*rows+o]=W[(size_t)o*cols+i];
    return t;
}
#ifdef __FAST_MATH__
/* -ffast-math builds (the research library's search) keep exactly the original loops. */
#define bopl_v4_mv(out, W, WT, x, rows, cols) bopl_v4_matvec(out, W, x, rows, cols)
#else
static inline void bopl_v4_mv(float *out, const float *W, const float *WT, const float *x, int rows, int cols) {
    if (WT) bopl_v4_matvec_t(out, WT, x, rows, cols); else bopl_v4_matvec(out, W, x, rows, cols);
}
#endif
/* Switch to the transposed matrices; skip_value also drops the critic. 0 on success,
 * -1 under -ffast-math (where the transposed sums would not be bit-identical). */
static inline int bopl_v4_fast(BoplV4Eval *e, int skip_value) {
#ifdef __FAST_MATH__
    (void)e; (void)skip_value;
    return -1;
#endif
    int H=e->hidden, F=e->shape.ffn_dim, C=e->shape.critic_dim, D=e->shape.set_dim, I=e->shape.set_inner;
    int W=H+(e->shape.critic_privileged?BOPL4_PRIVILEGED:0);
    e->skip_value=skip_value;
    if (e->t_flat) return 0;
    int ok=1;
    for (int s=0; s<4; s++) {
        ok&=(e->t_w1[s]=bopl_v4_transpose(e->w1[s], I, BOPL_V4_SETS[s].features))!=NULL;
        ok&=(e->t_w2[s]=bopl_v4_transpose(e->w2[s], D, I))!=NULL;
    }
    ok&=(e->t_flat=bopl_v4_transpose(e->flat, e->shape.flat_dim, BOPL4_FLAT))!=NULL;
    ok&=(e->t_fusion=bopl_v4_transpose(e->fusion, H, e->shape.flat_dim+8*D))!=NULL;
    if (F) {
        ok&=(e->t_ffn_in=bopl_v4_transpose(e->ffn_in, F, H))!=NULL;
        ok&=(e->t_ffn_out=bopl_v4_transpose(e->ffn_out, H, F))!=NULL;
    }
    ok&=(e->t_actor=bopl_v4_transpose(e->actor, e->width+(C?0:1), H))!=NULL;
    if (C) {
        ok&=(e->t_critic_in=bopl_v4_transpose(e->critic_in, C, W))!=NULL;
        ok&=(e->t_critic_b1=bopl_v4_transpose(e->critic_b1, C, C))!=NULL;
        ok&=(e->t_critic_b2=bopl_v4_transpose(e->critic_b2, C, C))!=NULL;
    }
    return ok?0:-1;
}
static void bopl_v4_layer_norm(float *out, const float *x, int n) {
    float mean=0.0f;
    for (int i=0; i<n; i++) mean+=x[i];
    mean/=n;
    float var=0.0f;
    for (int i=0; i<n; i++) { float d=x[i]-mean; var+=d*d; }
    var/=n;
    float r=1.0f/sqrtf(var+1e-5f);
    for (int i=0; i<n; i++) out[i]=(x[i]-mean)*r;
}
static int bopl_v4_embedding_id(float value) {
    int id=(int)value;
    return id>=0 && id<BOPL_EMBEDDING_IDS && value==(float)id?id:0;
}
/* max || mean over the valid rows of one set: out[0..D) max, out[D..2D) mean. */
static void bopl_v4_pool(BoplV4Eval *e, int set, const float *obs, float *out) {
    const BoplV4Set *s=&BOPL_V4_SETS[set];
    int D=e->shape.set_dim, I=e->shape.set_inner, n=0;
    memset(out, 0, 2*D*sizeof(float));
    for (int r=0; r<s->rows; r++) {
        const float *row=obs+s->offset+(size_t)r*s->width;
        if (row[0]==0.0f) continue;
        bopl_v4_mv(e->h1, e->w1[set], e->t_w1[set], row, I, s->features);
        if (s->id_column>=0) {
            const float *emb=e->embedding+(size_t)bopl_v4_embedding_id(row[s->id_column])*I;
            for (int i=0; i<I; i++) e->h1[i]+=emb[i];
        }
        for (int i=0; i<I; i++) e->h1[i]=fmaxf(0.0f, e->h1[i]);
        bopl_v4_mv(e->h2, e->w2[set], e->t_w2[set], e->h1, D, I);
        for (int c=0; c<D; c++) {
            float v=fmaxf(0.0f, e->h2[c]);
            if (v>out[c]) out[c]=v;
            out[D+c]+=v;
        }
        n++;
    }
    if (n && e->shape.mean_pool) for (int c=0; c<D; c++) out[D+c]/=n;
    else memset(out+D, 0, D*sizeof(float));
}
/* Encoder: observations (batch, BOPL4_OBS_SIZE) -> LN(fusion) (batch, hidden), the MinGRU input. */
static float *bopl_v4_encode(BoplV4Eval *e, const float *observations) {
    int H=e->hidden, FD=e->shape.flat_dim, D=e->shape.set_dim, joined_width=FD+8*D;
    for (int b=0; b<e->batch; b++) {
        const float *obs=observations+(size_t)b*BOPL4_OBS_SIZE;
        float *joined=e->joined+(size_t)b*joined_width;
        bopl_v4_mv(joined, e->flat, e->t_flat, obs, FD, BOPL4_FLAT);
        for (int c=0; c<FD; c++) joined[c]=fmaxf(0.0f, joined[c]);
        for (int s=0; s<4; s++) bopl_v4_pool(e, s, obs, joined+FD+2*D*s);
        bopl_v4_mv(e->ln+(size_t)b*H, e->fusion, e->t_fusion, joined, H, joined_width);
        bopl_v4_layer_norm(e->fused+(size_t)b*H, e->ln+(size_t)b*H, H);
    }
    return e->fused;
}
/* Decoder: h (batch, hidden) from the MinGRU, observations for the privileged block ->
 * rows (batch, width + 1): [logits, value]. */
static void bopl_v4_decode(BoplV4Eval *e, const float *h, const float *observations, float *rows) {
    int H=e->hidden, F=e->shape.ffn_dim, C=e->shape.critic_dim, P=e->shape.critic_privileged?BOPL4_PRIVILEGED:0, W=H+P;
    for (int b=0; b<e->batch; b++) {
        const float *hb=h+(size_t)b*H; float *y=e->y+(size_t)b*H, *row=rows+(size_t)b*(e->width+1);
        memcpy(y, hb, H*sizeof(float));
        if (F) {
            bopl_v4_layer_norm(e->ln, hb, H);
            bopl_v4_mv(e->f, e->ffn_in, e->t_ffn_in, e->ln, F, H);
            for (int i=0; i<F; i++) e->f[i]=fmaxf(0.0f, e->f[i]);
            bopl_v4_mv(e->ln, e->ffn_out, e->t_ffn_out, e->f, H, F);
            for (int i=0; i<H; i++) y[i]+=e->ln[i];
        }
        bopl_v4_mv(row, e->actor, e->t_actor, y, e->width+(C?0:1), H);
        float value=C?0.0f:row[e->width];
        if (C && !e->skip_value) {
            memcpy(e->zin, y, H*sizeof(float));
            if (P) memcpy(e->zin+H, observations+(size_t)b*BOPL4_OBS_SIZE+BOPL4_OBS_PRIVILEGED_OFFSET, P*sizeof(float));
            bopl_v4_layer_norm(e->ln, e->zin, W);
            bopl_v4_mv(e->z, e->critic_in, e->t_critic_in, e->ln, C, W);
            bopl_v4_layer_norm(e->ln, e->z, C);
            bopl_v4_mv(e->b, e->critic_b1, e->t_critic_b1, e->ln, C, C);
            for (int i=0; i<C; i++) e->b[i]=fmaxf(0.0f, e->b[i]);
            bopl_v4_mv(e->z2, e->critic_b2, e->t_critic_b2, e->b, C, C);
            for (int i=0; i<C; i++) e->z2[i]+=e->z[i];
            bopl_v4_layer_norm(e->ln, e->z2, C);
            for (int i=0; i<C; i++) value+=e->critic_out[i]*e->ln[i];
        }
        row[e->width]=value;
    }
}
#endif
