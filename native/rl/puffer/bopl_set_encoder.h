/* CPU reference forward for observation layout v3. Include after upstream
 * puffernet.h. Parameter order is flat, player, slot, entity, slime, E, fusion.
 * All projections are bias-free. Slot/entity ids are gathered from E[40,64]
 * before ReLU and never enter their linear projections. */
#ifndef BOPL_SET_ENCODER_H
#define BOPL_SET_ENCODER_H
#include "bopl_layout.h"

enum {
    BOPL_SET_DIM=64,
    BOPL_SET_FLAT=BOPL_OBS_PREFIX+BOPL_SELF_FEATURES+BOPL_PREVIOUS_FEATURES,
    BOPL_SET_PLAYER_WIDTH=BOPL_PLAYER_FEATURES,
    BOPL_SET_SLOT_WIDTH=BOPL_SLOT_FEATURES-1,
    BOPL_SET_ENTITY_WIDTH=BOPL_ENTITY_FEATURES-1,
    BOPL_SET_SLIME_WIDTH=BOPL_SLIME_FEATURES,
    BOPL_SET_JOINED=4*BOPL_SET_DIM,
    BOPL_SET_OBS=BOPL_OBS_SIZE
};

typedef struct {
    Linear *flat;
    Linear *fusion;
    float *player_weights;
    float *slot_weights;
    float *entity_weights;
    float *slime_weights;
    float *ability_embeddings;
    float *flat_input;
    float *joined;
    int batch;
    int hidden;
} BoplSetEncoder;

static size_t bopl_set_weight_count(int hidden) {
    return (size_t)hidden*BOPL_SET_FLAT
         + BOPL_SET_DIM*(BOPL_SET_PLAYER_WIDTH+BOPL_SET_SLOT_WIDTH
                         +BOPL_SET_ENTITY_WIDTH+BOPL_SET_SLIME_WIDTH)
         + BOPL_EMBEDDING_IDS*BOPL_SET_DIM
         + (size_t)hidden*(hidden+BOPL_SET_JOINED);
}

static BoplSetEncoder *bopl_set_create(Weights *weights, int batch, int hidden) {
    if (!weights || batch<1 || hidden<8 || hidden%8
        || (size_t)batch*(hidden+BOPL_SET_JOINED)>INT_MAX
        || (size_t)batch*BOPL_SET_FLAT>INT_MAX
        || bopl_set_weight_count(hidden)>INT_MAX-7)
        return NULL;
    size_t scratch=(size_t)batch*(BOPL_SET_FLAT+hidden+BOPL_SET_JOINED);
    BoplSetEncoder *encoder=calloc(1, sizeof(*encoder)+scratch*sizeof(float));
    if (!encoder) return NULL;
    encoder->batch=batch; encoder->hidden=hidden;
    encoder->flat_input=(float *)(encoder+1);
    encoder->joined=encoder->flat_input+(size_t)batch*BOPL_SET_FLAT;
    encoder->flat=make_linear(weights, batch, BOPL_SET_FLAT, hidden);
    encoder->player_weights=get_weights_aligned(weights, BOPL_SET_DIM*BOPL_SET_PLAYER_WIDTH);
    encoder->slot_weights=get_weights_aligned(weights, BOPL_SET_DIM*BOPL_SET_SLOT_WIDTH);
    encoder->entity_weights=get_weights_aligned(weights, BOPL_SET_DIM*BOPL_SET_ENTITY_WIDTH);
    encoder->slime_weights=get_weights_aligned(weights, BOPL_SET_DIM*BOPL_SET_SLIME_WIDTH);
    encoder->ability_embeddings=get_weights_aligned(weights, BOPL_EMBEDDING_IDS*BOPL_SET_DIM);
    encoder->fusion=make_linear(weights, batch, hidden+BOPL_SET_JOINED, hidden);
    return encoder;
}

static int bopl_set_embedding_id(float value) {
    int id=(int)value;
    return id>=0 && id<BOPL_EMBEDDING_IDS && value==(float)id?id:0;
}

static void bopl_set_pool(float *output, const float *rows, const float *weights,
                          const float *embeddings, int count, int row_width,
                          int projection_width, int id_column) {
    memset(output, 0, BOPL_SET_DIM*sizeof(float));
    for (int row=0; row<count; row++) {
        const float *input=rows+(size_t)row*row_width;
        if (input[0]==0.0f) continue;
        int id=id_column>=0?bopl_set_embedding_id(input[id_column]):0;
        for (int channel=0; channel<BOPL_SET_DIM; channel++) {
            const float *projection=weights+(size_t)channel*projection_width;
            float sum=embeddings?embeddings[(size_t)id*BOPL_SET_DIM+channel]:0.0f;
            for (int feature=0; feature<projection_width; feature++)
                sum+=input[feature]*projection[feature];
            if (sum>output[channel]) output[channel]=sum;
        }
    }
}

static float *bopl_set_forward(BoplSetEncoder *encoder, const float *observations) {
    int hidden=encoder->hidden, joined_width=hidden+BOPL_SET_JOINED;
    for (int batch=0; batch<encoder->batch; batch++) {
        const float *obs=observations+(size_t)batch*BOPL_SET_OBS;
        float *flat=encoder->flat_input+(size_t)batch*BOPL_SET_FLAT;
        memcpy(flat, obs, BOPL_OBS_PREFIX*sizeof(float));
        memcpy(flat+BOPL_OBS_PREFIX, obs+BOPL_OBS_SELF_OFFSET, BOPL_SELF_FEATURES*sizeof(float));
        memcpy(flat+BOPL_OBS_PREFIX+BOPL_SELF_FEATURES, obs+BOPL_OBS_PREVIOUS_OFFSET,
               BOPL_PREVIOUS_FEATURES*sizeof(float));
    }
    linear(encoder->flat, encoder->flat_input);
    for (int batch=0; batch<encoder->batch; batch++) {
        const float *obs=observations+(size_t)batch*BOPL_SET_OBS;
        const float *flat=encoder->flat->output+(size_t)batch*hidden;
        float *joined=encoder->joined+(size_t)batch*joined_width;
        for (int channel=0; channel<hidden; channel++) joined[channel]=fmaxf(0.0f, flat[channel]);
        float *sets=joined+hidden;
        bopl_set_pool(sets, obs+BOPL_OBS_PLAYER_OFFSET, encoder->player_weights, NULL,
                      BOPL_OBS_PLAYERS, BOPL_PLAYER_FEATURES, BOPL_SET_PLAYER_WIDTH, -1);
        bopl_set_pool(sets+BOPL_SET_DIM, obs+BOPL_OBS_SLOT_OFFSET, encoder->slot_weights,
                      encoder->ability_embeddings, BOPL_OBS_SLOTS, BOPL_SLOT_FEATURES,
                      BOPL_SET_SLOT_WIDTH, BOPL_SLOT_ABILITY_ID);
        bopl_set_pool(sets+2*BOPL_SET_DIM, obs+BOPL_OBS_ENTITY_OFFSET, encoder->entity_weights,
                      encoder->ability_embeddings, BOPL_OBS_ENTITIES, BOPL_ENTITY_FEATURES,
                      BOPL_SET_ENTITY_WIDTH, BOPL_ENTITY_ABILITY_ID);
        bopl_set_pool(sets+3*BOPL_SET_DIM, obs+BOPL_OBS_SLIME_OFFSET, encoder->slime_weights, NULL,
                      BOPL_OBS_SLIMES, BOPL_SLIME_FEATURES, BOPL_SET_SLIME_WIDTH, -1);
    }
    linear(encoder->fusion, encoder->joined);
    float *output=encoder->fusion->output;
    for (size_t i=0; i<(size_t)encoder->batch*hidden; i++) output[i]=fmaxf(0.0f, output[i]);
    return output;
}

static void bopl_set_free(BoplSetEncoder *encoder) {
    if (!encoder) return;
    free(encoder->flat); free(encoder->fusion); free(encoder);
}

#endif
