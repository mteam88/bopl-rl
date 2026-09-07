/* Evaluation only: expose upstream PufferNet's CPU forward pass with explicit
 * recurrent state. Training and optimization remain in PufferLib's CUDA backend.
 * Compile with the pinned PufferLib src directory on the include path. */
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include "puffernet.h"

typedef struct {
    Weights *weights;
    PufferNet *net;
    size_t state_size;
} BoplPolicyEval;

void *bopl_policy_eval_create(const float *data, size_t count,
                            int batch, int observations, int hidden, int layers) {
    /* Multiples of eight have identical flat layouts under upstream's fp32
     * and bf16 allocators, and its CPU reader. Reject ambiguous layouts. */
    if (!data || batch < 1 || observations < 1 || hidden < 8 || hidden % 8 || layers < 1)
        return NULL;
    size_t expected = (size_t)hidden * (observations + (size_t)273)
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
    int heads[] = {256, 16};
    eval->net = make_puffernet(eval->weights, batch, observations, hidden, layers, heads, 2);
    eval->state_size = states;
    return eval;
}

void bopl_policy_eval_step(void *handle, float *observations, const float *state,
                          float *logits_and_value, float *next_state) {
    BoplPolicyEval *eval = handle;
    PufferNet *net = eval->net;
    memcpy(net->mingru->state, state, eval->state_size * sizeof(float));
    /* Return raw logits so the evaluator owns sampling and its seeded RNG. */
    linear(net->encoder, observations);
    mingru(net->mingru, net->encoder->output);
    linear(net->decoder, net->mingru->output);
    memcpy(logits_and_value, net->decoder->output, (size_t)net->num_agents * 273 * sizeof(float));
    memcpy(next_state, net->mingru->state, eval->state_size * sizeof(float));
}

void bopl_policy_eval_close(void *handle) {
    if (!handle) return;
    BoplPolicyEval *eval = handle;
    free_puffernet(eval->net);
    free(eval->weights);
    free(eval);
}
