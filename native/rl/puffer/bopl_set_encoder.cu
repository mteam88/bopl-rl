// PufferLib custom Encoder. Include after models.cu, then call
// bopl_create_set_encoder(enc) from the bopl branch of create_custom_encoder.
// Layout matches bopl_set_encoder.h: flat, entity, slime, fusion; no biases.
#ifndef BOPL_SET_ENCODER_CU
#define BOPL_SET_ENCODER_CU
#include <cmath>
#include <initializer_list>
#include <stdexcept>

static constexpr int BS_PREFIX=898, BS_FLAT=905, BS_OBS=3257, BS_E=64;
static constexpr int BS_ENTITY_OFFSET=898, BS_SLIME_OFFSET=2914, BS_PREVIOUS_OFFSET=3250;

struct BoplSetWeights {
    PrecisionTensor flat, entity, slime, fusion;
    int hidden;
};
struct BoplSetActs {
    PrecisionTensor flat_input, entity_input, slime_input;
    PrecisionTensor flat, entity, slime, joined, out;
    IntTensor entity_winner, slime_winner;
    PrecisionTensor grad_out, grad_joined, grad_flat, grad_entity, grad_slime;
    PrecisionTensor wgrad_flat, wgrad_entity, wgrad_slime, wgrad_fusion;
};

__global__ void bs_gather_flat(precision_t* dst, const precision_t* obs, int B) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=B*BS_FLAT) return;
    int b=i/BS_FLAT, c=i%BS_FLAT;
    dst[i]=obs[b*BS_OBS+(c<BS_PREFIX ? c : BS_PREVIOUS_OFFSET+c-BS_PREFIX)];
}
template<int N, int W, int OFFSET>
__global__ void bs_gather_set(precision_t* dst, const precision_t* obs, int B) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=B*N*W) return;
    int b=i/(N*W), r=(i/W)%N;
    const precision_t* row=obs+b*BS_OBS+OFFSET+r*W;
    // Mask before projection as well: arbitrary absent features must have no effect.
    dst[i]=to_float(row[0])!=0.0f ? row[i%W] : from_float(0.0f);
}
__global__ void bs_join_flat(precision_t* joined, const precision_t* flat, int B, int H) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<B*H) joined[(i/H)*(H+2*BS_E)+i%H]=from_float(fmaxf(0.0f,to_float(flat[i])));
}
template<int N, int W>
__global__ void bs_pool(precision_t* joined, int* winner, const precision_t* projected,
                        const precision_t* input, int B, int H, int offset) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=B*BS_E) return;
    int b=i/BS_E, c=i%BS_E, best=-1;
    float maximum=0.0f;
    for(int r=0;r<N;r++) {
        if(to_float(input[(b*N+r)*W])==0.0f) continue;
        float value=to_float(projected[(b*N+r)*BS_E+c]);
        if(value>maximum) {maximum=value; best=r;}
    }
    joined[b*(H+2*BS_E)+offset+c]=from_float(maximum);
    winner[i]=best;
}
__global__ void bs_relu(precision_t* x, int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) x[i]=from_float(fmaxf(0.0f,to_float(x[i])));
}
__global__ void bs_relu_grad(precision_t* dst, const precision_t* grad,
                             const precision_t* output, int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) dst[i]=to_float(output[i])>0.0f ? grad[i] : from_float(0.0f);
}
__global__ void bs_flat_grad(precision_t* dst, const precision_t* joined_grad,
                             const precision_t* flat, int B, int H) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<B*H) dst[i]=to_float(flat[i])>0.0f ? joined_grad[(i/H)*(H+2*BS_E)+i%H] : from_float(0.0f);
}
template<int N>
__global__ void bs_pool_grad(precision_t* dst, const precision_t* joined_grad,
                             const int* winner, int B, int H, int offset) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=B*N*BS_E) return;
    int b=i/(N*BS_E), r=(i/BS_E)%N, c=i%BS_E;
    dst[i]=winner[b*BS_E+c]==r ? joined_grad[b*(H+2*BS_E)+offset+c] : from_float(0.0f);
}

static PrecisionTensor bs_forward(void* weights, void* activations, PrecisionTensor input, cudaStream_t stream) {
    auto* w=(BoplSetWeights*)weights; auto* a=(BoplSetActs*)activations;
    int B=a->out.shape[0], H=w->hidden;
    bs_gather_flat<<<grid_size(B*BS_FLAT),BLOCK_SIZE,0,stream>>>(a->flat_input.data,input.data,B);
    bs_gather_set<96,21,BS_ENTITY_OFFSET><<<grid_size(B*96*21),BLOCK_SIZE,0,stream>>>(a->entity_input.data,input.data,B);
    bs_gather_set<12,28,BS_SLIME_OFFSET><<<grid_size(B*12*28),BLOCK_SIZE,0,stream>>>(a->slime_input.data,input.data,B);
    puf_mm(&a->flat_input,&w->flat,&a->flat,stream);
    puf_mm(&a->entity_input,&w->entity,&a->entity,stream);
    puf_mm(&a->slime_input,&w->slime,&a->slime,stream);
    bs_join_flat<<<grid_size(B*H),BLOCK_SIZE,0,stream>>>(a->joined.data,a->flat.data,B,H);
    bs_pool<96,21><<<grid_size(B*BS_E),BLOCK_SIZE,0,stream>>>(a->joined.data,a->entity_winner.data,a->entity.data,a->entity_input.data,B,H,H);
    bs_pool<12,28><<<grid_size(B*BS_E),BLOCK_SIZE,0,stream>>>(a->joined.data,a->slime_winner.data,a->slime.data,a->slime_input.data,B,H,H+BS_E);
    puf_mm(&a->joined,&w->fusion,&a->out,stream);
    bs_relu<<<grid_size(B*H),BLOCK_SIZE,0,stream>>>(a->out.data,B*H);
    return a->out;
}
static void bs_backward(void* weights, void* activations, PrecisionTensor grad, cudaStream_t stream) {
    auto* w=(BoplSetWeights*)weights; auto* a=(BoplSetActs*)activations;
    int B=a->out.shape[0], H=w->hidden;
    bs_relu_grad<<<grid_size(B*H),BLOCK_SIZE,0,stream>>>(a->grad_out.data,grad.data,a->out.data,B*H);
    puf_mm_tn(&a->grad_out,&a->joined,&a->wgrad_fusion,stream);
    puf_mm_nn(&a->grad_out,&w->fusion,&a->grad_joined,stream);
    bs_flat_grad<<<grid_size(B*H),BLOCK_SIZE,0,stream>>>(a->grad_flat.data,a->grad_joined.data,a->flat.data,B,H);
    bs_pool_grad<96><<<grid_size(B*96*BS_E),BLOCK_SIZE,0,stream>>>(a->grad_entity.data,a->grad_joined.data,a->entity_winner.data,B,H,H);
    bs_pool_grad<12><<<grid_size(B*12*BS_E),BLOCK_SIZE,0,stream>>>(a->grad_slime.data,a->grad_joined.data,a->slime_winner.data,B,H,H+BS_E);
    puf_mm_tn(&a->grad_flat,&a->flat_input,&a->wgrad_flat,stream);
    puf_mm_tn(&a->grad_entity,&a->entity_input,&a->wgrad_entity,stream);
    puf_mm_tn(&a->grad_slime,&a->slime_input,&a->wgrad_slime,stream);
}
static void bs_init(void* weights, ulong* seed, cudaStream_t stream) {
    auto* w=(BoplSetWeights*)weights;
    for(auto* tensor:{&w->flat,&w->entity,&w->slime,&w->fusion})
        puf_kaiming_init(tensor,std::sqrt(2.0f),(*seed)++,stream);
}
static void bs_reg_params(void* weights, Allocator* alloc) {
    auto* w=(BoplSetWeights*)weights; int H=w->hidden;
    w->flat={.shape={H,BS_FLAT}}; w->entity={.shape={BS_E,21}};
    w->slime={.shape={BS_E,28}}; w->fusion={.shape={H,H+2*BS_E}};
    // Registration order is also the on-disk and optimizer gradient order.
    for(auto* tensor:{&w->flat,&w->entity,&w->slime,&w->fusion}) alloc_register(alloc,tensor);
}
static void bs_reg_rollout(void* weights, void* activations, Allocator* alloc, int B) {
    auto* w=(BoplSetWeights*)weights; auto* a=(BoplSetActs*)activations; int H=w->hidden;
    a->flat_input={.shape={B,BS_FLAT}}; a->entity_input={.shape={B*96,21}}; a->slime_input={.shape={B*12,28}};
    a->flat={.shape={B,H}}; a->entity={.shape={B*96,BS_E}}; a->slime={.shape={B*12,BS_E}};
    a->joined={.shape={B,H+2*BS_E}}; a->out={.shape={B,H}};
    a->entity_winner={.shape={B,BS_E}}; a->slime_winner={.shape={B,BS_E}};
    for(auto* tensor:{&a->flat_input,&a->entity_input,&a->slime_input,&a->flat,&a->entity,&a->slime,&a->joined,&a->out})
        alloc_register(alloc,tensor);
    alloc_register(alloc,&a->entity_winner); alloc_register(alloc,&a->slime_winner);
}
static void bs_reg_train(void* weights, void* activations, Allocator* acts, Allocator* grads, int B) {
    auto* w=(BoplSetWeights*)weights; auto* a=(BoplSetActs*)activations; int H=w->hidden;
    bs_reg_rollout(weights,activations,acts,B);
    a->grad_out={.shape={B,H}}; a->grad_joined={.shape={B,H+2*BS_E}};
    a->grad_flat={.shape={B,H}}; a->grad_entity={.shape={B*96,BS_E}}; a->grad_slime={.shape={B*12,BS_E}};
    for(auto* tensor:{&a->grad_out,&a->grad_joined,&a->grad_flat,&a->grad_entity,&a->grad_slime}) alloc_register(acts,tensor);
    a->wgrad_flat={.shape={H,BS_FLAT}}; a->wgrad_entity={.shape={BS_E,21}};
    a->wgrad_slime={.shape={BS_E,28}}; a->wgrad_fusion={.shape={H,H+2*BS_E}};
    for(auto* tensor:{&a->wgrad_flat,&a->wgrad_entity,&a->wgrad_slime,&a->wgrad_fusion}) alloc_register(grads,tensor);
}
static void* bs_create(void* self) {
    auto* encoder=(Encoder*)self;
    if(encoder->in_dim!=BS_OBS) throw std::runtime_error("Bopl SET expects 3257 observation features");
    auto* w=(BoplSetWeights*)calloc(1,sizeof(BoplSetWeights));
    w->hidden=encoder->out_dim; return w;
}
static void bs_free(void* value) {free(value);}
static void bopl_create_set_encoder(Encoder* encoder) {
    *encoder=Encoder{
        .forward=bs_forward,.backward=bs_backward,.init_weights=bs_init,.reg_params=bs_reg_params,
        .reg_train=bs_reg_train,.reg_rollout=bs_reg_rollout,.create_weights=bs_create,
        .free_weights=bs_free,.free_activations=bs_free,.in_dim=encoder->in_dim,.out_dim=encoder->out_dim,
        .activation_size=sizeof(BoplSetActs),
    };
}
#endif
