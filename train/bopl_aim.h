/* Aim head parameterisations (experiment of 2026-09-12), shared by the CUDA
 * trainer (pufferlib.cu, bopl_rollout.cu), the CPU evaluator (policy_eval.c)
 * and the in-game agent (bopl_agent.c). Python mirrors the widths in
 * eval/layout.py. Both are off by default and selected per run:
 *
 *   BOPL_AIM_FOURIER=K  (train.py --aim-fourier K)
 *     The 256 aim logits are not decoder outputs. The decoder emits one aim
 *     block of 1 + 2K numbers: a neutral logit (bin 0) and K harmonics of
 *     cos/sin coefficients; bin i >= 1 (angle theta_i = 2 pi (i-1)/255) gets
 *       logit_i = sum_k (a_k cos(k theta_i) + b_k sin(k theta_i)) / sqrt(K).
 *     Neighbouring bins share parameters and the 0/360 wrap is free. The
 *     1/sqrt(K) keeps the initial logit variance equal to a raw bin's.
 *
 *   BOPL_AIM_AR=1  (train.py --aim-ar 1)
 *     The aim head is conditioned on the sampled button combo: the decoder
 *     emits 16 aim blocks (raw 256 bins or Fourier), one per combo, and
 *     buttons are sampled before aim. Bow, dash and platform aims no longer
 *     have to share one distribution.
 *
 * Decoder row: [aim block(s)] [16 button logits] [5 hold logits] [value].
 * The environment interface (277 game logits, action mask, actions) does not
 * change; the samplers materialise the 277 logits from the row. */
#ifndef BOPL_AIM_H
#define BOPL_AIM_H
#include "bopl_layout.h"
#include <math.h>

#ifdef __CUDACC__
#define BOPL_AIM_FN static inline __host__ __device__
#else
#define BOPL_AIM_FN static inline
#endif

enum { BOPL_AIM_MAX_HARMONICS=64 };

/* Width of one aim block in the decoder row. */
BOPL_AIM_FN int bopl_aim_block_width(int harmonics) { return harmonics > 0 ? 1 + 2 * harmonics : BOPL_ACT_AIM; }
/* Number of aim blocks: one per button combo when autoregressive. */
BOPL_AIM_FN int bopl_aim_blocks(int autoregressive) { return autoregressive ? BOPL_ACT_BUTTONS : 1; }
/* Column of the button head in the decoder row (the hold head follows it). */
BOPL_AIM_FN int bopl_decoder_buttons_offset(int harmonics, int autoregressive) {
    return bopl_aim_blocks(autoregressive) * bopl_aim_block_width(harmonics);
}
/* Decoder logit columns; the value is column bopl_decoder_logits(). BOPL_LOGITS when both are off. */
BOPL_AIM_FN int bopl_decoder_logits(int harmonics, int autoregressive) {
    return bopl_decoder_buttons_offset(harmonics, autoregressive) + BOPL_ACT_BUTTONS + BOPL_ACT_HOLD;
}
/* Column of the aim block that serves a sampled button combo. */
BOPL_AIM_FN int bopl_aim_block_offset(int harmonics, int autoregressive, int buttons) {
    return (autoregressive ? buttons : 0) * bopl_aim_block_width(harmonics);
}
BOPL_AIM_FN float bopl_aim_fourier_scale(int harmonics) { return 1.0f / sqrtf((float)harmonics); }

/* cos/sin of k*theta_bin for k = 1, 2, ... by rotation; bin 0 (neutral) has no angle and yields zeros. */
typedef struct { float c1, s1, ck, sk; } BoplAimBasis;
BOPL_AIM_FN BoplAimBasis bopl_aim_basis(int bin) {
    BoplAimBasis b = {0.0f, 0.0f, 0.0f, 0.0f};
    if (bin > 0) {
        float theta = 6.283185307179586f * (float)(bin - 1) / 255.0f;
        b.c1 = b.ck = cosf(theta);
        b.s1 = b.sk = sinf(theta);
    }
    return b;
}
BOPL_AIM_FN void bopl_aim_basis_next(BoplAimBasis* b) {
    float c = b->ck * b->c1 - b->sk * b->s1;
    b->sk = b->sk * b->c1 + b->ck * b->s1;
    b->ck = c;
}

/* One aim logit from a Fourier block (block[0] neutral, block[2k-1] cos_k, block[2k] sin_k). */
BOPL_AIM_FN float bopl_aim_fourier_logit(const float* block, int harmonics, int bin) {
    if (bin <= 0) return block[0];
    BoplAimBasis b = bopl_aim_basis(bin);
    float sum = 0.0f;
    for (int k = 1; k <= harmonics; k++) {
        sum += block[2 * k - 1] * b.ck + block[2 * k] * b.sk;
        bopl_aim_basis_next(&b);
    }
    return sum * bopl_aim_fourier_scale(harmonics);
}

/* The BOPL_LOGITS game logits of one decoder row for a (sampled) button combo. */
BOPL_AIM_FN void bopl_aim_game_logits(const float* row, int harmonics, int autoregressive, int buttons, float* out) {
    const float* block = row + bopl_aim_block_offset(harmonics, autoregressive, buttons);
    if (harmonics > 0) for (int i = 0; i < BOPL_ACT_AIM; i++) out[i] = bopl_aim_fourier_logit(block, harmonics, i);
    else for (int i = 0; i < BOPL_ACT_AIM; i++) out[i] = block[i];
    const float* rest = row + bopl_decoder_buttons_offset(harmonics, autoregressive);
    for (int i = 0; i < BOPL_ACT_BUTTONS + BOPL_ACT_HOLD; i++) out[BOPL_ACT_AIM + i] = rest[i];
}

#endif
