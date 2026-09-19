// PufferLib custom encoder for observation layout v3. Include after models.cu.
// Each set branch fuses projection, optional E[id] gather, ReLU and max-pool;
// only the pooled value and winning row are materialized. Backward follows the
// winner indices and deterministically accumulates each parameter without
// atomics. Parameter order matches bopl_set_encoder.h and layout.py exactly.
#ifndef BOPL_SET_ENCODER_CU
#define BOPL_SET_ENCODER_CU
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include "bopl_layout.h"
#include "bopl_layout_v4.h"
#include "bopl_obs_pack.h"

static_assert(!USE_BF16, "bopl_set_encoder.cu requires PRECISION_FLOAT");

static constexpr int BS_E=64;
/* Row stride of every observation buffer: the larger of the layouts
 * (bopl_layout_v4.h); the v3 columns this encoder reads are the first
 * BOPL_OBS_SIZE of each row. */
static constexpr int BS_OBS=BOPL_OBS_STRIDE;
static constexpr int BS_PREFIX=BOPL_OBS_PREFIX;
static constexpr int BS_SELF=BOPL_SELF_FEATURES;
static constexpr int BS_PREVIOUS=BOPL_PREVIOUS_FEATURES;
static constexpr int BS_FLAT=BS_PREFIX+BS_SELF+BS_PREVIOUS;
static constexpr int BS_PLAYER_OFFSET=BOPL_OBS_PLAYER_OFFSET;
static constexpr int BS_SLOT_OFFSET=BOPL_OBS_SLOT_OFFSET;
static constexpr int BS_ENTITY_OFFSET=BOPL_OBS_ENTITY_OFFSET;
static constexpr int BS_SLIME_OFFSET=BOPL_OBS_SLIME_OFFSET;
static constexpr int BS_PREVIOUS_OFFSET=BOPL_OBS_PREVIOUS_OFFSET;
static constexpr int BS_N_PLAYER=BOPL_OBS_PLAYERS;
static constexpr int BS_N_SLOT=BOPL_OBS_SLOTS;
static constexpr int BS_N_ENT=BOPL_OBS_ENTITIES;
static constexpr int BS_N_SLIME=BOPL_OBS_SLIMES;
static constexpr int BS_W_PLAYER=BOPL_PLAYER_FEATURES;
static constexpr int BS_W_SLOT=BOPL_SLOT_FEATURES;
static constexpr int BS_W_ENT=BOPL_ENTITY_FEATURES;
static constexpr int BS_W_SLIME=BOPL_SLIME_FEATURES;
static constexpr int BS_K_PLAYER=BS_W_PLAYER;
static constexpr int BS_K_SLOT=BS_W_SLOT-1;
static constexpr int BS_K_ENT=BS_W_ENT-1;
static constexpr int BS_K_SLIME=BS_W_SLIME;
static constexpr int BS_POOLS=4;
static constexpr int BS_JOINED_POOLS=BS_POOLS*BS_E;
static constexpr int BS_BWD_ROWS=4;
static constexpr int BS_BWD_THREADS=BS_BWD_ROWS*BS_E;
static constexpr int BS_BWD_MAX_BLOCKS=192;
static constexpr int BS_GRAD_PLAYER=0;
static constexpr int BS_GRAD_SLOT=BS_GRAD_PLAYER+BS_E*BS_K_PLAYER;
static constexpr int BS_GRAD_ENTITY=BS_GRAD_SLOT+BS_E*BS_K_SLOT;
static constexpr int BS_GRAD_SLIME=BS_GRAD_ENTITY+BS_E*BS_K_ENT;
static constexpr int BS_GRAD_EMBEDDING=BS_GRAD_SLIME+BS_E*BS_K_SLIME;
static constexpr int BS_SET_GRAD_FLOATS=BS_GRAD_EMBEDDING+BOPL_EMBEDDING_IDS*BS_E;

static_assert(BS_N_ENT==64 && BS_N_SLIME==32 && BS_N_SLOT==24, "layout-v3 set caps");
static_assert((BS_K_ENT+7)/8==5, "layout-v3 entity tile uses KSTEPS=5");
static_assert(BOPL_SLOT_ABILITY_ID==BS_K_SLOT && BOPL_ENTITY_ABILITY_ID==BS_K_ENT,
              "ability id is excluded from slot/entity projections");

struct BsGather {
    const float* obs;
    const int* idx;
    int T, B_total;
};

struct BoplSetWeights {
    PrecisionTensor flat, player, slot, entity, slime, embedding, fusion;
    int hidden;
};

struct BoplSetActs {
    PrecisionTensor flat_input, flat_output, pooled, joined, out;
    IntTensor player_winner, slot_winner, entity_winner, slime_winner;
    PrecisionTensor grad_out, grad_joined, grad_flat;
    PrecisionTensor wgrad_partials;
    PrecisionTensor wgrad_flat, wgrad_player, wgrad_slot, wgrad_entity,
                    wgrad_slime, wgrad_embedding, wgrad_fusion;
    BsGather gather;
    const float* saved_input;
};

__device__ __forceinline__ const float* bs_row_ptr(const float* input, const BsGather& g, int row) {
    if(g.obs==nullptr) return input+(long)row*BS_OBS;
    int segment=row/g.T, tick=row-segment*g.T;
    return g.obs+((long)tick*g.B_total+g.idx[segment])*BS_OBS;
}

__device__ __forceinline__ int bs_embedding_id(float value) {
    int id=(int)value;
    return id>=0 && id<BOPL_EMBEDDING_IDS && value==(float)id ? id : 0;
}

__global__ void bs_pack_flat(const float* input, BsGather gather, float* flat, int B) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=B*BS_FLAT) return;
    int row=i/BS_FLAT, feature=i-row*BS_FLAT;
    const float* src=bs_row_ptr(input,gather,row);
    if(feature<BS_PREFIX) flat[i]=src[feature];
    else if(feature<BS_PREFIX+BS_SELF)
        flat[i]=src[BOPL_OBS_SELF_OFFSET+feature-BS_PREFIX];
    else flat[i]=src[BS_PREVIOUS_OFFSET+feature-BS_PREFIX-BS_SELF];
}

template<int POOL, int N, int W, int K, int OFFSET, int ID_COLUMN, bool PACKED>
__global__ void bs_pool_forward(const float* input, BsGather gather, int B,
                                const float* weights, const float* embedding,
                                float* pooled, int* winner) {
    int row=blockIdx.x, channel=threadIdx.x;
    if(row>=B || channel>=BS_E) return;
    const float* block=bs_row_ptr(input,gather,row)+OFFSET;
    float best=0.0f; int best_row=-1;
    #pragma unroll 1
    for(int slot=0;slot<N;slot++) {
        const float* x=block+slot*W;
        if(x[0]==0.0f) { if(PACKED) break; else continue; }
        float sum=0.0f;
        if constexpr(ID_COLUMN>=0) {
            int id=bs_embedding_id(x[ID_COLUMN]);
            sum=embedding[id*BS_E+channel];
        }
        constexpr int KSTEPS=(K+7)/8;
        #pragma unroll
        for(int step=0;step<KSTEPS;step++) {
            #pragma unroll
            for(int lane=0;lane<8;lane++) {
                int k=step*8+lane;
                if(k<K) sum=fmaf(x[k],weights[channel*K+k],sum);
            }
        }
        if(sum>best) { best=sum; best_row=slot; }
    }
    pooled[(long)row*BS_JOINED_POOLS+POOL*BS_E+channel]=best;
    winner[(long)row*BS_E+channel]=best_row;
}

__global__ void bs_join(const float* flat, const float* pooled, float* joined, int B, int H) {
    int width=H+BS_JOINED_POOLS;
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=B*width) return;
    int row=i/width, column=i-row*width;
    joined[i]=column<H ? fmaxf(0.0f,flat[(long)row*H+column])
                       : pooled[(long)row*BS_JOINED_POOLS+column-H];
}

__global__ void bs_relu(precision_t* values, int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) values[i]=from_float(fmaxf(0.0f,to_float(values[i])));
}

__global__ void bs_relu_grad(precision_t* dst, const precision_t* grad,
                             const precision_t* output, int count) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) dst[i]=to_float(output[i])>0.0f ? grad[i] : from_float(0.0f);
}

__global__ void bs_flat_grad(float* dst, const float* grad_joined,
                             const float* flat_output, int B, int H) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=B*H) return;
    int row=i/H, channel=i-row*H;
    dst[i]=flat_output[i]>0.0f ? grad_joined[(long)row*(H+BS_JOINED_POOLS)+channel] : 0.0f;
}

template<int N, int W, int K, int OFFSET, int BASE>
__global__ void __launch_bounds__(BS_BWD_THREADS, 2) bs_pool_weight_partials(
        const float* input, BsGather gather, int B, int H, int pool,
        const int* winner, const float* grad_joined, float* partials) {
    extern __shared__ float reduction[];
    int row_group=threadIdx.x/BS_E, channel=threadIdx.x%BS_E;
    float sums[K];
    #pragma unroll
    for(int feature=0;feature<K;feature++) sums[feature]=0.0f;
    int blocks=gridDim.x;
    for(int row=blockIdx.x*BS_BWD_ROWS+row_group;row<B;row+=blocks*BS_BWD_ROWS) {
        int slot=winner[(long)row*BS_E+channel];
        if(slot>=0 && slot<N) {
            const float* x=bs_row_ptr(input,gather,row)+OFFSET+slot*W;
            float g=grad_joined[(long)row*(H+BS_JOINED_POOLS)+H+pool*BS_E+channel];
            #pragma unroll
            for(int feature=0;feature<K;feature++) sums[feature]=fmaf(g,x[feature],sums[feature]);
        }
    }
    #pragma unroll
    for(int feature=0;feature<K;feature++)
        reduction[(row_group*BS_E+channel)*K+feature]=sums[feature];
    __syncthreads();
    for(int i=threadIdx.x;i<BS_E*K;i+=BS_BWD_THREADS) {
        float sum=0.0f;
        #pragma unroll
        for(int group=0;group<BS_BWD_ROWS;group++) sum+=reduction[group*BS_E*K+i];
        partials[(long)blockIdx.x*BS_SET_GRAD_FLOATS+BASE+i]=sum;
    }
}

__global__ void bs_embedding_partials(const float* input, BsGather gather, int B, int H,
                                      const int* slot_winner, const int* entity_winner,
                                      const float* grad_joined, float* partials, int blocks) {
    int i=blockIdx.y*blockDim.x+threadIdx.x;
    if(i>=BOPL_EMBEDDING_IDS*BS_E) return;
    int id=i/BS_E, channel=i-id*BS_E;
    float sum=0.0f;
    for(int row=blockIdx.x;row<B;row+=blocks) {
        const float* obs=bs_row_ptr(input,gather,row);
        int slot=slot_winner[(long)row*BS_E+channel];
        if(slot>=0 && bs_embedding_id(obs[BS_SLOT_OFFSET+slot*BS_W_SLOT+BOPL_SLOT_ABILITY_ID])==id)
            sum+=grad_joined[(long)row*(H+BS_JOINED_POOLS)+H+BS_E+channel];
        int entity=entity_winner[(long)row*BS_E+channel];
        if(entity>=0 && bs_embedding_id(obs[BS_ENTITY_OFFSET+entity*BS_W_ENT+BOPL_ENTITY_ABILITY_ID])==id)
            sum+=grad_joined[(long)row*(H+BS_JOINED_POOLS)+H+2*BS_E+channel];
    }
    partials[(long)blockIdx.x*BS_SET_GRAD_FLOATS+BS_GRAD_EMBEDDING+i]=sum;
}

__global__ void bs_reduce_partials(const float* partials, int blocks, int base,
                                   int count, float* gradient) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    float sum=0.0f;
    for(int block=0;block<blocks;block++) sum+=partials[(long)block*BS_SET_GRAD_FLOATS+base+i];
    gradient[i]=sum;
}

static int g_bs_packed=-1;
static bool bs_packed() {
    if(g_bs_packed<0) { const char* value=getenv("BOPL_SET_PACKED"); g_bs_packed=!(value && value[0]=='0'); }
    return g_bs_packed!=0;
}
void bs_set_packed(int packed) { g_bs_packed=packed ? 1 : 0; }
// Kept for the qualification API. Dense GEMMs still follow BOPL_TF32; the
// fused pools use fp32 so E[id] and winner selection have one implementation.
void bs_set_mma(int) {}
void bs_set_flat_bf16(int) {}

template<bool PACKED>
static void bs_launch_pools(BoplSetWeights* w, BoplSetActs* a, const float* input,
                            int B, cudaStream_t stream) {
    bs_pool_forward<0,BS_N_PLAYER,BS_W_PLAYER,BS_K_PLAYER,BS_PLAYER_OFFSET,-1,PACKED>
        <<<B,BS_E,0,stream>>>(input,a->gather,B,w->player.data,nullptr,a->pooled.data,a->player_winner.data);
    bs_pool_forward<1,BS_N_SLOT,BS_W_SLOT,BS_K_SLOT,BS_SLOT_OFFSET,BOPL_SLOT_ABILITY_ID,PACKED>
        <<<B,BS_E,0,stream>>>(input,a->gather,B,w->slot.data,w->embedding.data,a->pooled.data,a->slot_winner.data);
    bs_pool_forward<2,BS_N_ENT,BS_W_ENT,BS_K_ENT,BS_ENTITY_OFFSET,BOPL_ENTITY_ABILITY_ID,PACKED>
        <<<B,BS_E,0,stream>>>(input,a->gather,B,w->entity.data,w->embedding.data,a->pooled.data,a->entity_winner.data);
    bs_pool_forward<3,BS_N_SLIME,BS_W_SLIME,BS_K_SLIME,BS_SLIME_OFFSET,-1,PACKED>
        <<<B,BS_E,0,stream>>>(input,a->gather,B,w->slime.data,nullptr,a->pooled.data,a->slime_winner.data);
}

static PrecisionTensor bs_forward(void* weights, void* activations, PrecisionTensor input, cudaStream_t stream) {
    auto* w=(BoplSetWeights*)weights; auto* a=(BoplSetActs*)activations;
    int B=a->out.shape[0], H=w->hidden;
    a->saved_input=input.data;
    bs_pack_flat<<<grid_size(B*BS_FLAT),BLOCK_SIZE,0,stream>>>(input.data,a->gather,a->flat_input.data,B);
    puf_mm(&a->flat_input,&w->flat,&a->flat_output,stream);
    if(bs_packed()) bs_launch_pools<true>(w,a,input.data,B,stream);
    else bs_launch_pools<false>(w,a,input.data,B,stream);
    bs_join<<<grid_size(B*(H+BS_JOINED_POOLS)),BLOCK_SIZE,0,stream>>>(
        a->flat_output.data,a->pooled.data,a->joined.data,B,H);
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
    bs_flat_grad<<<grid_size(B*H),BLOCK_SIZE,0,stream>>>(a->grad_flat.data,a->grad_joined.data,a->flat_output.data,B,H);
    puf_mm_tn(&a->grad_flat,&a->flat_input,&a->wgrad_flat,stream);
    int blocks=(B+BS_BWD_ROWS-1)/BS_BWD_ROWS;
    if(blocks>BS_BWD_MAX_BLOCKS) blocks=BS_BWD_MAX_BLOCKS;
    bs_pool_weight_partials<BS_N_PLAYER,BS_W_PLAYER,BS_K_PLAYER,BS_PLAYER_OFFSET,BS_GRAD_PLAYER>
        <<<blocks,BS_BWD_THREADS,BS_BWD_ROWS*BS_E*BS_K_PLAYER*sizeof(float),stream>>>(
            a->saved_input,a->gather,B,H,0,a->player_winner.data,a->grad_joined.data,a->wgrad_partials.data);
    bs_pool_weight_partials<BS_N_SLOT,BS_W_SLOT,BS_K_SLOT,BS_SLOT_OFFSET,BS_GRAD_SLOT>
        <<<blocks,BS_BWD_THREADS,BS_BWD_ROWS*BS_E*BS_K_SLOT*sizeof(float),stream>>>(
            a->saved_input,a->gather,B,H,1,a->slot_winner.data,a->grad_joined.data,a->wgrad_partials.data);
    bs_pool_weight_partials<BS_N_ENT,BS_W_ENT,BS_K_ENT,BS_ENTITY_OFFSET,BS_GRAD_ENTITY>
        <<<blocks,BS_BWD_THREADS,BS_BWD_ROWS*BS_E*BS_K_ENT*sizeof(float),stream>>>(
            a->saved_input,a->gather,B,H,2,a->entity_winner.data,a->grad_joined.data,a->wgrad_partials.data);
    bs_pool_weight_partials<BS_N_SLIME,BS_W_SLIME,BS_K_SLIME,BS_SLIME_OFFSET,BS_GRAD_SLIME>
        <<<blocks,BS_BWD_THREADS,BS_BWD_ROWS*BS_E*BS_K_SLIME*sizeof(float),stream>>>(
            a->saved_input,a->gather,B,H,3,a->slime_winner.data,a->grad_joined.data,a->wgrad_partials.data);
    dim3 embedding_grid(blocks,grid_size(BOPL_EMBEDDING_IDS*BS_E));
    bs_embedding_partials<<<embedding_grid,BLOCK_SIZE,0,stream>>>(
        a->saved_input,a->gather,B,H,a->slot_winner.data,a->entity_winner.data,
        a->grad_joined.data,a->wgrad_partials.data,blocks);
    bs_reduce_partials<<<grid_size(BS_E*BS_K_PLAYER),BLOCK_SIZE,0,stream>>>(
        a->wgrad_partials.data,blocks,BS_GRAD_PLAYER,BS_E*BS_K_PLAYER,a->wgrad_player.data);
    bs_reduce_partials<<<grid_size(BS_E*BS_K_SLOT),BLOCK_SIZE,0,stream>>>(
        a->wgrad_partials.data,blocks,BS_GRAD_SLOT,BS_E*BS_K_SLOT,a->wgrad_slot.data);
    bs_reduce_partials<<<grid_size(BS_E*BS_K_ENT),BLOCK_SIZE,0,stream>>>(
        a->wgrad_partials.data,blocks,BS_GRAD_ENTITY,BS_E*BS_K_ENT,a->wgrad_entity.data);
    bs_reduce_partials<<<grid_size(BS_E*BS_K_SLIME),BLOCK_SIZE,0,stream>>>(
        a->wgrad_partials.data,blocks,BS_GRAD_SLIME,BS_E*BS_K_SLIME,a->wgrad_slime.data);
    bs_reduce_partials<<<grid_size(BOPL_EMBEDDING_IDS*BS_E),BLOCK_SIZE,0,stream>>>(
        a->wgrad_partials.data,blocks,BS_GRAD_EMBEDDING,BOPL_EMBEDDING_IDS*BS_E,a->wgrad_embedding.data);
}

static void bs_init(void* weights, ulong* seed, cudaStream_t stream) {
    auto* w=(BoplSetWeights*)weights;
    for(auto* tensor:{&w->flat,&w->player,&w->slot,&w->entity,&w->slime,&w->embedding,&w->fusion})
        puf_kaiming_init(tensor,std::sqrt(2.0f),(*seed)++,stream);
}

static void bs_reg_params(void* weights, Allocator* alloc) {
    auto* w=(BoplSetWeights*)weights; int H=w->hidden;
    w->flat={.shape={H,BS_FLAT}};
    w->player={.shape={BS_E,BS_K_PLAYER}};
    w->slot={.shape={BS_E,BS_K_SLOT}};
    w->entity={.shape={BS_E,BS_K_ENT}};
    w->slime={.shape={BS_E,BS_K_SLIME}};
    w->embedding={.shape={BOPL_EMBEDDING_IDS,BS_E}};
    w->fusion={.shape={H,H+BS_JOINED_POOLS}};
    for(auto* tensor:{&w->flat,&w->player,&w->slot,&w->entity,&w->slime,&w->embedding,&w->fusion})
        alloc_register(alloc,tensor);
}

static void bs_reg_forward(void* weights, void* activations, Allocator* alloc, int B) {
    auto* w=(BoplSetWeights*)weights; auto* a=(BoplSetActs*)activations; int H=w->hidden;
    a->flat_input={.shape={B,BS_FLAT}}; a->flat_output={.shape={B,H}};
    a->pooled={.shape={B,BS_JOINED_POOLS}}; a->joined={.shape={B,H+BS_JOINED_POOLS}};
    a->out={.shape={B,H}};
    a->player_winner={.shape={B,BS_E}}; a->slot_winner={.shape={B,BS_E}};
    a->entity_winner={.shape={B,BS_E}}; a->slime_winner={.shape={B,BS_E}};
    for(auto* tensor:{&a->flat_input,&a->flat_output,&a->pooled,&a->joined,&a->out}) alloc_register(alloc,tensor);
    for(auto* tensor:{&a->player_winner,&a->slot_winner,&a->entity_winner,&a->slime_winner}) alloc_register(alloc,tensor);
    a->gather=BsGather{};
}

static void bs_reg_rollout(void* weights, void* activations, Allocator* alloc, int B) {
    bs_reg_forward(weights,activations,alloc,B);
}

static void bs_reg_train(void* weights, void* activations, Allocator* acts, Allocator* grads, int B) {
    auto* w=(BoplSetWeights*)weights; auto* a=(BoplSetActs*)activations; int H=w->hidden;
    bs_reg_forward(weights,activations,acts,B);
    a->grad_out={.shape={B,H}}; a->grad_joined={.shape={B,H+BS_JOINED_POOLS}};
    a->grad_flat={.shape={B,H}};
    a->wgrad_partials={.shape={BS_BWD_MAX_BLOCKS,BS_SET_GRAD_FLOATS}};
    for(auto* tensor:{&a->grad_out,&a->grad_joined,&a->grad_flat,&a->wgrad_partials}) alloc_register(acts,tensor);
    a->wgrad_flat={.shape={H,BS_FLAT}}; a->wgrad_player={.shape={BS_E,BS_K_PLAYER}};
    a->wgrad_slot={.shape={BS_E,BS_K_SLOT}}; a->wgrad_entity={.shape={BS_E,BS_K_ENT}};
    a->wgrad_slime={.shape={BS_E,BS_K_SLIME}};
    a->wgrad_embedding={.shape={BOPL_EMBEDDING_IDS,BS_E}};
    a->wgrad_fusion={.shape={H,H+BS_JOINED_POOLS}};
    for(auto* tensor:{&a->wgrad_flat,&a->wgrad_player,&a->wgrad_slot,&a->wgrad_entity,
                      &a->wgrad_slime,&a->wgrad_embedding,&a->wgrad_fusion}) alloc_register(grads,tensor);
}

static void bs_bind_gather(void* activations, const precision_t* rollout_obs,
                           const int* idx, int T, int B_total) {
    ((BoplSetActs*)activations)->gather=BsGather{rollout_obs,idx,T,B_total};
}

static void* bs_create(void* self) {
    auto* encoder=(Encoder*)self;
    if(encoder->in_dim!=BS_OBS)
        throw std::runtime_error("Bopl SET expects "+std::to_string(BS_OBS)+" observation features per row (layout v3 in a BOPL_OBS_STRIDE row)");
    auto* weights=(BoplSetWeights*)calloc(1,sizeof(BoplSetWeights));
    weights->hidden=encoder->out_dim;
    return weights;
}
static void bs_free(void* value) { free(value); }

static void bopl_create_set_encoder(Encoder* encoder) {
    *encoder=Encoder{
        .forward=bs_forward,.backward=bs_backward,.init_weights=bs_init,.reg_params=bs_reg_params,
        .reg_train=bs_reg_train,.reg_rollout=bs_reg_rollout,.create_weights=bs_create,
        .free_weights=bs_free,.free_activations=bs_free,.in_dim=encoder->in_dim,.out_dim=encoder->out_dim,
        .activation_size=sizeof(BoplSetActs),
    };
}
#endif
