/* CPU forward for Bopl's set encoder. Include after upstream puffernet.h.
 * All projections are bias-free, matching upstream Linear. Parameter order:
 * flat [H,905], entity [64,21], slime [64,28], fusion [H,H+128].
 * This encoder is followed by the unchanged upstream decoder/MinGRU weights.
 */
#ifndef BOPL_SET_ENCODER_H
#define BOPL_SET_ENCODER_H

enum {
    BOPL_SET_PREFIX = 10 + 6 * 148,
    BOPL_SET_ENTITIES = 96,
    BOPL_SET_ENTITY_WIDTH = 21,
    BOPL_SET_SLIMES = 12,
    BOPL_SET_SLIME_WIDTH = 28,
    BOPL_SET_PREVIOUS = 7,
    BOPL_SET_FLAT = BOPL_SET_PREFIX + BOPL_SET_PREVIOUS,
    BOPL_SET_DIM = 64,
    BOPL_SET_SLIME_OFFSET = BOPL_SET_PREFIX + BOPL_SET_ENTITIES * BOPL_SET_ENTITY_WIDTH,
    BOPL_SET_PREVIOUS_OFFSET = BOPL_SET_SLIME_OFFSET + BOPL_SET_SLIMES * BOPL_SET_SLIME_WIDTH,
    BOPL_SET_OBS = BOPL_SET_PREVIOUS_OFFSET + BOPL_SET_PREVIOUS
};

typedef struct {
    Linear *flat;
    Linear *fusion;
    float *entity_weights;
    float *slime_weights;
    float *flat_input;
    float *joined;
    int batch;
    int hidden;
} BoplSetEncoder;

static size_t bopl_set_weight_count(int hidden) {
    return (size_t)hidden * BOPL_SET_FLAT
         + BOPL_SET_DIM * (BOPL_SET_ENTITY_WIDTH + BOPL_SET_SLIME_WIDTH)
         + (size_t)hidden * (hidden + (size_t)2 * BOPL_SET_DIM);
}

static BoplSetEncoder *bopl_set_create(Weights *weights, int batch, int hidden) {
    if (!weights || batch < 1 || hidden < 8 || hidden % 8
        || (size_t)batch * (hidden + (size_t)2 * BOPL_SET_DIM) > INT_MAX
        || (size_t)batch * BOPL_SET_FLAT > INT_MAX
        || bopl_set_weight_count(hidden) > INT_MAX - 7)
        return NULL;
    size_t scratch = (size_t)batch * (BOPL_SET_FLAT + hidden + (size_t)2 * BOPL_SET_DIM);
    BoplSetEncoder *encoder = calloc(1, sizeof(*encoder) + scratch * sizeof(float));
    if (!encoder) return NULL;
    encoder->batch = batch;
    encoder->hidden = hidden;
    encoder->flat_input = (float *)(encoder + 1);
    encoder->joined = encoder->flat_input + (size_t)batch * BOPL_SET_FLAT;
    encoder->flat = make_linear(weights, batch, BOPL_SET_FLAT, hidden);
    encoder->entity_weights = get_weights_aligned(weights, BOPL_SET_DIM * BOPL_SET_ENTITY_WIDTH);
    encoder->slime_weights = get_weights_aligned(weights, BOPL_SET_DIM * BOPL_SET_SLIME_WIDTH);
    encoder->fusion = make_linear(weights, batch, hidden + 2 * BOPL_SET_DIM, hidden);
    return encoder;
}

static void bopl_set_pool(float *output, const float *rows, const float *weights,
                          int count, int width) {
    /* ReLU makes every valid candidate nonnegative. Zero initialization gives
     * an empty set exactly zero. Absent rows never enter the reduction. */
    memset(output, 0, BOPL_SET_DIM * sizeof(float));
    for (int row = 0; row < count; row++) {
        const float *input = rows + (size_t)row * width;
        if (input[0] == 0.0f) continue;
        for (int channel = 0; channel < BOPL_SET_DIM; channel++) {
            const float *projection = weights + (size_t)channel * width;
            float sum = 0.0f;
            for (int feature = 0; feature < width; feature++)
                sum += input[feature] * projection[feature];
            if (sum > output[channel]) output[channel] = sum;
        }
    }
}

static float *bopl_set_forward(BoplSetEncoder *encoder, const float *observations) {
    int hidden = encoder->hidden;
    int joined_width = hidden + 2 * BOPL_SET_DIM;
    for (int batch = 0; batch < encoder->batch; batch++) {
        const float *obs = observations + (size_t)batch * BOPL_SET_OBS;
        float *flat = encoder->flat_input + (size_t)batch * BOPL_SET_FLAT;
        memcpy(flat, obs, BOPL_SET_PREFIX * sizeof(float));
        memcpy(flat + BOPL_SET_PREFIX, obs + BOPL_SET_PREVIOUS_OFFSET,
               BOPL_SET_PREVIOUS * sizeof(float));
    }
    linear(encoder->flat, encoder->flat_input);
    for (int batch = 0; batch < encoder->batch; batch++) {
        const float *obs = observations + (size_t)batch * BOPL_SET_OBS;
        const float *flat = encoder->flat->output + (size_t)batch * hidden;
        float *joined = encoder->joined + (size_t)batch * joined_width;
        for (int channel = 0; channel < hidden; channel++)
            joined[channel] = fmaxf(0.0f, flat[channel]);
        bopl_set_pool(joined + hidden, obs + BOPL_SET_PREFIX,
                      encoder->entity_weights, BOPL_SET_ENTITIES, BOPL_SET_ENTITY_WIDTH);
        bopl_set_pool(joined + hidden + BOPL_SET_DIM, obs + BOPL_SET_SLIME_OFFSET,
                      encoder->slime_weights, BOPL_SET_SLIMES, BOPL_SET_SLIME_WIDTH);
    }
    linear(encoder->fusion, encoder->joined);
    float *output = encoder->fusion->output;
    for (size_t index = 0; index < (size_t)encoder->batch * hidden; index++)
        output[index] = fmaxf(0.0f, output[index]);
    return output;
}

static void bopl_set_free(BoplSetEncoder *encoder) {
    if (!encoder) return;
    free(encoder->flat);
    free(encoder->fusion);
    free(encoder);
}

#endif
