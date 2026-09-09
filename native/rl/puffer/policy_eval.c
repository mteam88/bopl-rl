/* Evaluation only: expose upstream PufferNet's CPU forward pass with explicit
 * recurrent state. Training and optimization remain in PufferLib's CUDA backend.
 * Compile with the pinned PufferLib src directory on the include path. */
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include "puffernet.h"
#include "bopl_set_encoder.h"

typedef struct {
    Weights *weights;
    PufferNet *net;
    BoplSetEncoder *set_encoder;
    size_t state_size;
} BoplPolicyEval;

static void *bopl_policy_eval_create_impl(const float *data, size_t count,
                            int batch, int observations, int hidden, int layers, int use_set) {
    /* Multiples of eight have identical flat layouts under upstream's fp32
     * and bf16 allocators, and its CPU reader. Reject ambiguous layouts. */
    if (!data || batch < 1 || observations < 1 || hidden < 8 || hidden % 8 || layers < 1)
        return NULL;
    if (use_set && observations != BOPL_SET_OBS) return NULL;
    size_t encoder_count = use_set ? bopl_set_weight_count(hidden) : (size_t)hidden * observations;
    size_t expected = encoder_count + (size_t)hidden * BOPL_DECODER
                    + (size_t)layers * 3 * hidden * hidden;
    size_t states = (size_t)layers * batch * hidden;
    if (count != expected || count > INT_MAX - 7 || states > INT_MAX / 3
        || (size_t)batch * observations > INT_MAX)
        return NULL;
    for (size_t i = 0; i < count; i++) if (!isfinite(data[i])) return NULL;
    BoplPolicyEval *eval = calloc(1, sizeof(*eval));
    if (!eval) return NULL;
    eval->weights = calloc(1, sizeof(Weights) + (count + 7) * sizeof(float));
    if (!eval->weights) { free(eval); return NULL; }
    eval->weights->data = (float *)(eval->weights + 1);
    eval->weights->size = (int)count + 7;
    memcpy(eval->weights->data, data, count * sizeof(float));
    int heads[] = {BOPL_ACT_AIM, BOPL_ACT_BUTTONS, BOPL_ACT_HOLD};
    if (use_set) {
        eval->set_encoder = bopl_set_create(eval->weights, batch, hidden);
        if (!eval->set_encoder) { free(eval->weights); free(eval); return NULL; }
        eval->net = calloc(1, sizeof(PufferNet));
        if (!eval->net) {
            bopl_set_free(eval->set_encoder); free(eval->weights); free(eval); return NULL;
        }
        eval->net->num_agents = batch;
        eval->net->decoder = make_linear(eval->weights, batch, hidden, BOPL_DECODER);
        eval->net->mingru = make_mingru(eval->weights, batch, hidden, layers);
    } else {
        eval->net = make_puffernet(eval->weights, batch, observations, hidden, layers, heads, BOPL_NUM_ATNS);
    }
    eval->state_size = states;
    return eval;
}

void *bopl_policy_eval_create(const float *data, size_t count,
                            int batch, int observations, int hidden, int layers) {
    return bopl_policy_eval_create_impl(data, count, batch, observations, hidden, layers, 0);
}

void *bopl_policy_eval_create_set(const float *data, size_t count,
                                int batch, int observations, int hidden, int layers) {
    return bopl_policy_eval_create_impl(data, count, batch, observations, hidden, layers, 1);
}

void bopl_policy_eval_step(void *handle, float *observations, const float *state,
                          float *logits_and_value, float *next_state) {
    BoplPolicyEval *eval = handle;
    PufferNet *net = eval->net;
    memcpy(net->mingru->state, state, eval->state_size * sizeof(float));
    /* Return raw logits so the evaluator owns sampling and its seeded RNG. */
    float *encoded;
    if (eval->set_encoder) {
        encoded = bopl_set_forward(eval->set_encoder, observations);
    } else {
        linear(net->encoder, observations);
        encoded = net->encoder->output;
    }
    mingru(net->mingru, encoded);
    linear(net->decoder, net->mingru->output);
    memcpy(logits_and_value, net->decoder->output, (size_t)net->num_agents * BOPL_DECODER * sizeof(float));
    memcpy(next_state, net->mingru->state, eval->state_size * sizeof(float));
}

void bopl_policy_eval_close(void *handle) {
    if (!handle) return;
    BoplPolicyEval *eval = handle;
    free_puffernet(eval->net);
    bopl_set_free(eval->set_encoder);
    free(eval->weights);
    free(eval);
}
