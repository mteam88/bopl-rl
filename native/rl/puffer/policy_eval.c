/* Evaluation only: expose upstream PufferNet's CPU forward pass with explicit
 * recurrent state. Training and optimization remain in PufferLib's CUDA backend.
 * Compile with the pinned PufferLib src directory on the include path.
 *
 * The decoder row is [aim block(s)] [buttons] [hold] [value] (bopl_aim.h): a
 * plain 277 + 1 row unless the checkpoint trained with a Fourier aim block
 * and/or one aim block per button combo. bopl_policy_eval_step_raw returns
 * the row; bopl_policy_eval_game_logits turns it into the BOPL_LOGITS game
 * logits (plus the value) for a button combo, the one materialisation the
 * CPU evaluator (research/native_policy.py) and the in-game agent
 * (bopl_agent.c) share, so their logits are bit-identical. */
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include "puffernet.h"
#include "bopl_set_encoder.h"
#include "bopl_aim.h"

typedef struct {
    Weights *weights;
    PufferNet *net;
    BoplSetEncoder *set_encoder;
    size_t state_size;
    int aim_fourier, aim_ar, width;  /* width: decoder logit columns; the row is width + 1 */
} BoplPolicyEval;

void *bopl_policy_eval_create_aim(const float *data, size_t count, int batch, int observations,
                                  int hidden, int layers, int use_set, int aim_fourier, int aim_ar) {
    /* Multiples of eight have identical flat layouts under upstream's fp32
     * and bf16 allocators, and its CPU reader. Reject ambiguous layouts. */
    if (!data || batch < 1 || observations < 1 || hidden < 8 || hidden % 8 || layers < 1)
        return NULL;
    if (aim_fourier < 0 || aim_fourier > BOPL_AIM_MAX_HARMONICS || aim_ar < 0 || aim_ar > 1) return NULL;
    if (use_set && observations != BOPL_SET_OBS) return NULL;
    int width = bopl_decoder_logits(aim_fourier, aim_ar);
    size_t encoder_count = use_set ? bopl_set_weight_count(hidden) : (size_t)hidden * observations;
    size_t expected = encoder_count + (size_t)hidden * (width + 1)
                    + (size_t)layers * 3 * hidden * hidden;
    size_t states = (size_t)layers * batch * hidden;
    if (count != expected || count > INT_MAX - 7 || states > INT_MAX / 3
        || (size_t)batch * observations > INT_MAX || (size_t)batch * (width + 1) > INT_MAX)
        return NULL;
    for (size_t i = 0; i < count; i++) if (!isfinite(data[i])) return NULL;
    BoplPolicyEval *eval = calloc(1, sizeof(*eval));
    if (!eval) return NULL;
    eval->weights = calloc(1, sizeof(Weights) + (count + 7) * sizeof(float));
    if (!eval->weights) { free(eval); return NULL; }
    eval->weights->data = (float *)(eval->weights + 1);
    eval->weights->size = (int)count + 7;
    memcpy(eval->weights->data, data, count * sizeof(float));
    eval->aim_fourier = aim_fourier; eval->aim_ar = aim_ar; eval->width = width;
    /* Built by hand (not make_puffernet): the decoder is width + 1 wide, and
     * the sampler lives here, not in upstream's Multidiscrete. */
    eval->net = calloc(1, sizeof(PufferNet));
    if (!eval->net) { free(eval->weights); free(eval); return NULL; }
    eval->net->num_agents = batch;
    if (use_set) {
        eval->set_encoder = bopl_set_create(eval->weights, batch, hidden);
        if (!eval->set_encoder) { free(eval->net); free(eval->weights); free(eval); return NULL; }
    } else {
        eval->net->obs = calloc((size_t)batch * observations, sizeof(float));
        eval->net->encoder = make_linear(eval->weights, batch, observations, hidden);
    }
    eval->net->decoder = make_linear(eval->weights, batch, hidden, width + 1);
    eval->net->mingru = make_mingru(eval->weights, batch, hidden, layers);
    eval->state_size = states;
    return eval;
}

void *bopl_policy_eval_create(const float *data, size_t count,
                            int batch, int observations, int hidden, int layers) {
    return bopl_policy_eval_create_aim(data, count, batch, observations, hidden, layers, 0, 0, 0);
}

void *bopl_policy_eval_create_set(const float *data, size_t count,
                                int batch, int observations, int hidden, int layers) {
    return bopl_policy_eval_create_aim(data, count, batch, observations, hidden, layers, 1, 0, 0);
}

int bopl_policy_eval_width(void *handle) { return handle ? ((BoplPolicyEval *)handle)->width : -1; }
int bopl_policy_eval_aim_fourier(void *handle) { return handle ? ((BoplPolicyEval *)handle)->aim_fourier : -1; }
int bopl_policy_eval_aim_ar(void *handle) { return handle ? ((BoplPolicyEval *)handle)->aim_ar : -1; }

/* rows: (batch, width + 1) raw decoder rows. */
void bopl_policy_eval_step_raw(void *handle, float *observations, const float *state,
                               float *rows, float *next_state) {
    BoplPolicyEval *eval = handle;
    PufferNet *net = eval->net;
    memcpy(net->mingru->state, state, eval->state_size * sizeof(float));
    float *encoded;
    if (eval->set_encoder) {
        encoded = bopl_set_forward(eval->set_encoder, observations);
    } else {
        linear(net->encoder, observations);
        encoded = net->encoder->output;
    }
    mingru(net->mingru, encoded);
    linear(net->decoder, net->mingru->output);
    memcpy(rows, net->decoder->output, (size_t)net->num_agents * (eval->width + 1) * sizeof(float));
    memcpy(next_state, net->mingru->state, eval->state_size * sizeof(float));
}

/* The BOPL_LOGITS game logits then the value of every row, for the row's
 * button combo (buttons[b], ignored unless the aim head is autoregressive). */
void bopl_policy_eval_game_logits(void *handle, const float *rows, const int *buttons, float *out) {
    BoplPolicyEval *eval = handle;
    int stride = eval->width + 1;
    for (int b = 0; b < eval->net->num_agents; b++) {
        const float *row = rows + (size_t)b * stride;
        int combo = buttons ? buttons[b] : 0;
        if (combo < 0 || combo >= BOPL_ACT_BUTTONS) combo = 0;
        bopl_aim_game_logits(row, eval->aim_fourier, eval->aim_ar, combo, out + (size_t)b * BOPL_DECODER);
        out[(size_t)b * BOPL_DECODER + BOPL_LOGITS] = row[eval->width];
    }
}

/* Raw logits for button combo 0 (the only combo when not autoregressive) so
 * the evaluator owns sampling and its seeded RNG. */
void bopl_policy_eval_step(void *handle, float *observations, const float *state,
                          float *logits_and_value, float *next_state) {
    BoplPolicyEval *eval = handle;
    float *rows = malloc((size_t)eval->net->num_agents * (eval->width + 1) * sizeof(float));
    bopl_policy_eval_step_raw(handle, observations, state, rows, next_state);
    bopl_policy_eval_game_logits(handle, rows, NULL, logits_and_value);
    free(rows);
}

void bopl_policy_eval_close(void *handle) {
    if (!handle) return;
    BoplPolicyEval *eval = handle;
    free_puffernet(eval->net);
    bopl_set_free(eval->set_encoder);
    free(eval->weights);
    free(eval);
}
