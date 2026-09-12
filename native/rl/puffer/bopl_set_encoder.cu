// PufferLib custom Encoder. Include after models.cu, then call
// bopl_create_set_encoder(enc) from the bopl branch of create_custom_encoder.
// Layout matches bopl_set_encoder.h: flat, entity, slime, fusion; no biases.
//
// The entity (96 x 21 -> 64, max-pool) and slime (12 x 28 -> 64, max-pool)
// branches are fused: one kernel reads each observation row once, computes
// every slot's projection in registers, max-pools in-kernel and stores only
// the pooled B x 64 outputs plus the winning slot per (row, channel). The
// B x 96 x 64 activation is never materialised. The backward recomputes
// nothing: the gradient of a max-pool flows to the winning slot only, so
// dW = sum_rows g[row, c] * x[row, winner[row, c], :], accumulated per block
// in registers and reduced deterministically (no atomics).
//
// Training rows can be read straight out of the (T, B_total, obs) rollout
// buffer through the minibatch's segment indices (bs_bind_gather), which
// removes the per-update observation transpose and the per-minibatch
// select_copy of the observations.
#ifndef BOPL_SET_ENCODER_CU
#define BOPL_SET_ENCODER_CU
#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include "bopl_layout.h"

static_assert(!USE_BF16, "bopl_set_encoder.cu: the fused kernels assume PRECISION_FLOAT");

// Layout v2 (bopl_layout.h): prefix 934, flat 946, obs 3298.
static constexpr int BS_PREFIX=BOPL_OBS_PREFIX, BS_FLAT=BOPL_OBS_PREFIX+BOPL_PREVIOUS_FEATURES, BS_OBS=BOPL_OBS_SIZE, BS_E=64;
static constexpr int BS_ENTITY_OFFSET=BOPL_OBS_ENTITY_OFFSET, BS_SLIME_OFFSET=BOPL_OBS_SLIME_OFFSET, BS_PREVIOUS_OFFSET=BOPL_OBS_PREVIOUS_OFFSET;
static constexpr int BS_N_ENT=BOPL_OBS_ENTITIES, BS_W_ENT=BOPL_ENTITY_FEATURES, BS_N_SLIME=BOPL_OBS_SLIMES, BS_W_SLIME=BOPL_SLIME_FEATURES;
static constexpr int BS_PREVIOUS=BOPL_PREVIOUS_FEATURES;
static constexpr int BS_ENT_FLOATS=BS_N_ENT*BS_W_ENT, BS_SLIME_FLOATS=BS_N_SLIME*BS_W_SLIME;
static constexpr int BS_WGRAD_FLOATS=BS_E*BS_W_ENT+BS_E*BS_W_SLIME;  // one partial dW block

// Forward tiling: a row is handled by BS_FWD_TPR threads; thread (sg, cg)
// owns slots [8 sg, 8 sg + 8) x channels [4 cg, 4 cg + 4) u [32 + 4 cg, 36 + 4 cg)
// so both operands are read from shared memory as conflict-free float4s.
// Entity features are staged k-major (Xt[k][slot], row stride BS_XT_STRIDE).
static constexpr int BS_FWD_ROWS=2, BS_FWD_TPR=96, BS_FWD_THREADS=BS_FWD_ROWS*BS_FWD_TPR;
static constexpr int BS_FWD_SLOTS=8, BS_FWD_CH=8, BS_FWD_SG=BS_N_ENT/BS_FWD_SLOTS;  // 12 slot groups
static constexpr int BS_XT_STRIDE=100;  // >= 96, multiple of 4, spreads the staging scatter over banks
static constexpr int BS_FWD_MAX_BLOCKS=1536;
__device__ __forceinline__ int bs_fwd_channel(int cg, int j) {return (j<4 ? 0 : 32-4)+cg*4+j;}
static constexpr int BS_WSL_STRIDE=BS_W_SLIME+1;  // 29: conflict-free per-channel reads
// Backward tiling: thread = channel, BS_BWD_ROWS rows per block in flight.
static constexpr int BS_BWD_ROWS=4, BS_BWD_THREADS=BS_BWD_ROWS*BS_E, BS_BWD_MAX_BLOCKS=192;
static_assert(BS_FWD_SG*BS_FWD_CH==BS_FWD_TPR, "forward tiling");
static_assert(BS_ENT_FLOATS%2==0 && BS_SLIME_FLOATS%2==0 && BS_PREFIX%2==0 && BS_PREVIOUS%2==0, "float2 staging");
static_assert(BS_ENTITY_OFFSET%2==0 && BS_SLIME_OFFSET%2==0 && BS_PREVIOUS_OFFSET%2==0 && BS_OBS%2==0 && BS_FLAT%2==0, "float2 alignment");

// Where a training row lives: obs == nullptr -> contiguous input rows;
// otherwise row i = (segment i / T, step i % T) of the (T, B_total, obs)
// rollout buffer through idx[segment].
struct BsGather {
    const float* obs;
    const int* idx;
    int T, B_total;
};

struct BoplSetWeights {
    PrecisionTensor flat, entity, slime, fusion;
    int hidden;
};
struct BoplSetActs {
    PrecisionTensor flat_input;              // (B, 946)
    PrecisionTensor joined, out;             // (B, H+128), (B, H)
    IntTensor entity_winner, slime_winner;   // (B, 64) slot index or -1
    PrecisionTensor grad_out, grad_joined, grad_flat;
    PrecisionTensor wgrad_partials;          // (BS_BWD_MAX_BLOCKS, BS_WGRAD_FLOATS)
    PrecisionTensor wgrad_flat, wgrad_entity, wgrad_slime, wgrad_fusion;
    BsGather gather;                         // training only; zero = contiguous input
    const float* saved_input;                // contiguous input of the last forward (for backward)
};

__device__ __forceinline__ const float* bs_row_ptr(const float* input, const BsGather& g, int row) {
    if(g.obs==nullptr) return input+(long)row*BS_OBS;
    int seg=row/g.T, t=row-seg*g.T;
    return g.obs+((long)t*g.B_total+g.idx[seg])*BS_OBS;
}
// Cooperative float2 copies with compile-time trip counts: every load is
// issued before the first store, so a row costs one memory latency, not one
// per element.
template<int COUNT2, int THREADS>
__device__ __forceinline__ void bs_copy2(float* dst, const float* src, int tid) {
    constexpr int ITERS=(COUNT2+THREADS-1)/THREADS;
    const float2* s=(const float2*)src; float2* d=(float2*)dst;
    float2 v[ITERS];
    #pragma unroll
    for(int it=0;it<ITERS;it++) {int i=tid+it*THREADS; if(i<COUNT2) v[it]=s[i];}
    #pragma unroll
    for(int it=0;it<ITERS;it++) {int i=tid+it*THREADS; if(i<COUNT2) d[i]=v[it];}
}
// Entity block of one row -> k-major shared tile Xt[k][slot].
template<int THREADS>
__device__ __forceinline__ void bs_stage_entities(float* xt, const float* src, int tid) {
    constexpr int COUNT2=BS_ENT_FLOATS/2, ITERS=(COUNT2+THREADS-1)/THREADS;
    const float2* s=(const float2*)(src+BS_ENTITY_OFFSET);
    float2 v[ITERS];
    #pragma unroll
    for(int it=0;it<ITERS;it++) {int i=tid+it*THREADS; if(i<COUNT2) v[it]=s[i];}
    #pragma unroll
    for(int it=0;it<ITERS;it++) {
        int i=tid+it*THREADS;
        if(i<COUNT2) {
            int e=2*i, slot=e/BS_W_ENT, k=e-slot*BS_W_ENT;
            xt[k*BS_XT_STRIDE+slot]=v[it].x;
            if(++k==BS_W_ENT) {k=0; slot++;}
            xt[k*BS_XT_STRIDE+slot]=v[it].y;
        }
    }
}
// (value, slot) order of a max-pool: larger value wins, ties go to the lower
// slot; -1 only ever pairs with 0 (no valid slot beat the ReLU floor).
__device__ __forceinline__ void bs_better(float& v, int& i, float ov, int oi) {
    if(ov>v || (ov==v && oi!=-1 && oi<i)) {v=ov; i=oi;}
}

__global__ void __launch_bounds__(BS_FWD_THREADS, 2) bs_fused_forward(
        const float* __restrict__ input, BsGather g, int B, int H,
        const float* __restrict__ w_ent, const float* __restrict__ w_slime,
        float* __restrict__ flat_input, float* __restrict__ joined,
        int* __restrict__ ent_winner, int* __restrict__ slime_winner) {
    __shared__ __align__(16) float Xt[BS_FWD_ROWS][BS_W_ENT*BS_XT_STRIDE];  // [k][slot]
    __shared__ __align__(16) float Ss[BS_FWD_ROWS][BS_SLIME_FLOATS];
    __shared__ __align__(16) float Wt[BS_W_ENT][BS_E];                    // [k][channel]
    __shared__ float Wsl[BS_E*BS_WSL_STRIDE];
    __shared__ float red_v[BS_FWD_ROWS][2][BS_E];
    __shared__ int red_i[BS_FWD_ROWS][2][BS_E];

    const int tid=threadIdx.x, rg=tid/BS_FWD_TPR, tr=tid-rg*BS_FWD_TPR;
    const int warp_in_row=tr>>5, lane=tr&31;
    const int sg=tr>>3, cg=tr&7;
    const int JW=H+2*BS_E;
    for(int i=tid;i<BS_E*BS_W_ENT;i+=BS_FWD_THREADS) Wt[i%BS_W_ENT][i/BS_W_ENT]=w_ent[i];
    for(int i=tid;i<BS_E*BS_W_SLIME;i+=BS_FWD_THREADS) Wsl[(i/BS_W_SLIME)*BS_WSL_STRIDE+i%BS_W_SLIME]=w_slime[i];

    for(int row0=blockIdx.x*BS_FWD_ROWS;row0<B;row0+=gridDim.x*BS_FWD_ROWS) {
        const bool live=row0+rg<B;
        const int row=live ? row0+rg : B-1;
        const float* src=bs_row_ptr(input,g,row);
        __syncthreads();  // previous iteration is done with Xt/Ss/red
        bs_stage_entities<BS_FWD_TPR>(Xt[rg],src,tr);
        bs_copy2<BS_SLIME_FLOATS/2,BS_FWD_TPR>(Ss[rg],src+BS_SLIME_OFFSET,tr);
        if(live) {
            float* flat=flat_input+(long)row*BS_FLAT;
            bs_copy2<BS_PREFIX/2,BS_FWD_TPR>(flat,src,tr);
            bs_copy2<BS_PREVIOUS/2,BS_FWD_TPR>(flat+BS_PREFIX,src+BS_PREVIOUS_OFFSET,tr);
        }
        __syncthreads();

        // Entity projections for this thread's 8 slots x 8 channels.
        float acc[BS_FWD_SLOTS][BS_FWD_CH];
        #pragma unroll
        for(int s=0;s<BS_FWD_SLOTS;s++)
            #pragma unroll
            for(int j=0;j<BS_FWD_CH;j++) acc[s][j]=0.0f;
        const float* xt=Xt[rg]+sg*BS_FWD_SLOTS;
        #pragma unroll
        for(int k=0;k<BS_W_ENT;k++) {
            float4 w0=*(const float4*)&Wt[k][cg*4], w1=*(const float4*)&Wt[k][32+cg*4];
            float4 x0=*(const float4*)&xt[k*BS_XT_STRIDE], x1=*(const float4*)&xt[k*BS_XT_STRIDE+4];
            const float w[BS_FWD_CH]={w0.x,w0.y,w0.z,w0.w,w1.x,w1.y,w1.z,w1.w};
            const float x[BS_FWD_SLOTS]={x0.x,x0.y,x0.z,x0.w,x1.x,x1.y,x1.z,x1.w};
            #pragma unroll
            for(int s=0;s<BS_FWD_SLOTS;s++)
                #pragma unroll
                for(int j=0;j<BS_FWD_CH;j++) acc[s][j]=fmaf(x[s],w[j],acc[s][j]);
        }
        // Masked max-pool over this thread's slots (strict >, ReLU floor 0).
        float bv[BS_FWD_CH]; int bi[BS_FWD_CH];
        #pragma unroll
        for(int j=0;j<BS_FWD_CH;j++) {bv[j]=0.0f; bi[j]=-1;}
        #pragma unroll
        for(int s=0;s<BS_FWD_SLOTS;s++) {
            bool valid=xt[s]!=0.0f;  // feature 0 of the slot
            #pragma unroll
            for(int j=0;j<BS_FWD_CH;j++)
                if(valid && acc[s][j]>bv[j]) {bv[j]=acc[s][j]; bi[j]=sg*BS_FWD_SLOTS+s;}
        }
        // Across the 4 slot groups of this warp (lanes differ in bits 3, 4).
        #pragma unroll
        for(int off=8;off<=16;off<<=1)
            #pragma unroll
            for(int j=0;j<BS_FWD_CH;j++) {
                float ov=__shfl_xor_sync(0xffffffffu,bv[j],off);
                int oi=__shfl_xor_sync(0xffffffffu,bi[j],off);
                bs_better(bv[j],bi[j],ov,oi);
            }
        if(warp_in_row>0 && lane<8) {
            #pragma unroll
            for(int j=0;j<BS_FWD_CH;j++) {
                int c=bs_fwd_channel(cg,j);
                red_v[rg][warp_in_row-1][c]=bv[j]; red_i[rg][warp_in_row-1][c]=bi[j];
            }
        }
        // Slime: thread = channel for the first 64 threads of the row.
        float sv=0.0f; int si=-1;
        if(tr<BS_E) {
            const float* wsl=Wsl+tr*BS_WSL_STRIDE;
            #pragma unroll 1
            for(int s=0;s<BS_N_SLIME;s++) {
                const float* xs=Ss[rg]+s*BS_W_SLIME;
                if(xs[0]==0.0f) continue;
                float a=0.0f;
                #pragma unroll
                for(int k=0;k<BS_W_SLIME;k++) a=fmaf(xs[k],wsl[k],a);
                if(a>sv) {sv=a; si=s;}
            }
        }
        __syncthreads();
        if(live) {
            if(warp_in_row==0 && lane<8) {
                #pragma unroll
                for(int j=0;j<BS_FWD_CH;j++) {
                    int c=bs_fwd_channel(cg,j);
                    bs_better(bv[j],bi[j],red_v[rg][0][c],red_i[rg][0][c]);
                    bs_better(bv[j],bi[j],red_v[rg][1][c],red_i[rg][1][c]);
                    joined[(long)row*JW+H+c]=bv[j];
                    ent_winner[(long)row*BS_E+c]=bi[j];
                }
            }
            if(tr<BS_E) {
                joined[(long)row*JW+H+BS_E+tr]=sv;
                slime_winner[(long)row*BS_E+tr]=si;
            }
        }
    }
}

// dW partials: partials[block][c][f] = sum over the block's rows of
// g[row, c] * x[row, winner[row, c], f] for the entity block, then slime.
__global__ void __launch_bounds__(BS_BWD_THREADS) bs_fused_backward(
        const float* __restrict__ input, BsGather g, int B, int H,
        const float* __restrict__ grad_joined, const int* __restrict__ ent_winner,
        const int* __restrict__ slime_winner, float* __restrict__ partials) {
    __shared__ __align__(16) float Xs[BS_BWD_ROWS][BS_ENT_FLOATS];
    __shared__ __align__(16) float Ss[BS_BWD_ROWS][BS_SLIME_FLOATS];
    const int tid=threadIdx.x, rg=tid/BS_E, c=tid-rg*BS_E;
    const int JW=H+2*BS_E;
    float ae[BS_W_ENT], as[BS_W_SLIME];
    #pragma unroll
    for(int f=0;f<BS_W_ENT;f++) ae[f]=0.0f;
    #pragma unroll
    for(int f=0;f<BS_W_SLIME;f++) as[f]=0.0f;

    for(int row0=blockIdx.x*BS_BWD_ROWS;row0<B;row0+=gridDim.x*BS_BWD_ROWS) {
        const bool live=row0+rg<B;
        const int row=live ? row0+rg : B-1;
        const float* src=bs_row_ptr(input,g,row);
        __syncthreads();
        bs_copy2<BS_ENT_FLOATS/2,BS_E>(Xs[rg],src+BS_ENTITY_OFFSET,c);
        bs_copy2<BS_SLIME_FLOATS/2,BS_E>(Ss[rg],src+BS_SLIME_OFFSET,c);
        __syncthreads();
        if(!live) continue;
        int we=ent_winner[(long)row*BS_E+c];
        if(we>=0) {
            float ge=grad_joined[(long)row*JW+H+c];
            const float* x=Xs[rg]+we*BS_W_ENT;
            #pragma unroll
            for(int f=0;f<BS_W_ENT;f++) ae[f]=fmaf(ge,x[f],ae[f]);
        }
        int ws=slime_winner[(long)row*BS_E+c];
        if(ws>=0) {
            float gs=grad_joined[(long)row*JW+H+BS_E+c];
            const float* x=Ss[rg]+ws*BS_W_SLIME;
            #pragma unroll
            for(int f=0;f<BS_W_SLIME;f++) as[f]=fmaf(gs,x[f],as[f]);
        }
    }
    // Reduce the BS_BWD_ROWS row groups through shared memory (fixed order).
    float* red=&Xs[0][0];  // BS_BWD_ROWS*BS_ENT_FLOATS floats of scratch
    float* out=partials+(long)blockIdx.x*BS_WGRAD_FLOATS;
    __syncthreads();
    #pragma unroll
    for(int f=0;f<BS_W_ENT;f++) red[(rg*BS_E+c)*BS_W_ENT+f]=ae[f];
    __syncthreads();
    for(int i=tid;i<BS_E*BS_W_ENT;i+=BS_BWD_THREADS) {
        float s=0.0f;
        for(int r=0;r<BS_BWD_ROWS;r++) s+=red[r*BS_E*BS_W_ENT+i];
        out[i]=s;
    }
    __syncthreads();
    #pragma unroll
    for(int f=0;f<BS_W_SLIME;f++) red[(rg*BS_E+c)*BS_W_SLIME+f]=as[f];
    __syncthreads();
    for(int i=tid;i<BS_E*BS_W_SLIME;i+=BS_BWD_THREADS) {
        float s=0.0f;
        for(int r=0;r<BS_BWD_ROWS;r++) s+=red[r*BS_E*BS_W_SLIME+i];
        out[BS_E*BS_W_ENT+i]=s;
    }
}
__global__ void bs_reduce_partials(const float* __restrict__ partials, int blocks,
                                   float* __restrict__ wgrad_ent, float* __restrict__ wgrad_slime) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=BS_WGRAD_FLOATS) return;
    float s=0.0f;
    for(int b=0;b<blocks;b++) s+=partials[(long)b*BS_WGRAD_FLOATS+i];
    if(i<BS_E*BS_W_ENT) wgrad_ent[i]=s; else wgrad_slime[i-BS_E*BS_W_ENT]=s;
}
// ReLU in place on the flat columns [0, H) of joined (B, H+128).
__global__ void bs_relu_flat(float* joined, int B, int H) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=B*H) return;
    float* p=joined+(long)(i/H)*(H+2*BS_E)+i%H;
    *p=fmaxf(0.0f,*p);
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
// grad_flat = grad_joined[:, :H] masked by the ReLU output stored in joined.
__global__ void bs_flat_grad(precision_t* dst, const precision_t* joined_grad,
                             const precision_t* joined, int B, int H) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=B*H) return;
    long j=(long)(i/H)*(H+2*BS_E)+i%H;
    dst[i]=to_float(joined[j])>0.0f ? joined_grad[j] : from_float(0.0f);
}

// out(B, N) = a(B, K) @ w(N, K)^T written with row stride ldc (cuBLAS, same
// handle/workspace and compute type as puf_mm).
static void bs_mm_ldc(PrecisionTensor* a, PrecisionTensor* w, precision_t* out, int ldc, cudaStream_t stream) {
    int M=a->shape[0], K=a->shape[1], N=w->shape[0];
    float alpha=1.0f, beta=0.0f;
    cublasHandle_t handle=cublas_get_handle(stream);
    cublasSetStream(handle,stream);
    cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&alpha,
        w->data,CUBLAS_PRECISION,K,a->data,CUBLAS_PRECISION,K,&beta,
        out,CUBLAS_PRECISION,ldc,CUBLAS_COMPUTE_PRECISION,CUBLAS_GEMM_DEFAULT);
}
static int bs_fwd_blocks(int B) {int n=(B+BS_FWD_ROWS-1)/BS_FWD_ROWS; return n<BS_FWD_MAX_BLOCKS ? n : BS_FWD_MAX_BLOCKS;}
static int bs_bwd_blocks(int B) {int n=(B+BS_BWD_ROWS-1)/BS_BWD_ROWS; return n<BS_BWD_MAX_BLOCKS ? n : BS_BWD_MAX_BLOCKS;}

static PrecisionTensor bs_forward(void* weights, void* activations, PrecisionTensor input, cudaStream_t stream) {
    auto* w=(BoplSetWeights*)weights; auto* a=(BoplSetActs*)activations;
    int B=a->out.shape[0], H=w->hidden;
    a->saved_input=input.data;
    bs_fused_forward<<<bs_fwd_blocks(B),BS_FWD_THREADS,0,stream>>>(
        input.data,a->gather,B,H,w->entity.data,w->slime.data,
        a->flat_input.data,a->joined.data,a->entity_winner.data,a->slime_winner.data);
    bs_mm_ldc(&a->flat_input,&w->flat,a->joined.data,H+2*BS_E,stream);
    bs_relu_flat<<<grid_size(B*H),BLOCK_SIZE,0,stream>>>(a->joined.data,B,H);
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
    bs_flat_grad<<<grid_size(B*H),BLOCK_SIZE,0,stream>>>(a->grad_flat.data,a->grad_joined.data,a->joined.data,B,H);
    puf_mm_tn(&a->grad_flat,&a->flat_input,&a->wgrad_flat,stream);
    int blocks=bs_bwd_blocks(B);
    bs_fused_backward<<<blocks,BS_BWD_THREADS,0,stream>>>(
        a->saved_input,a->gather,B,H,
        a->grad_joined.data,a->entity_winner.data,a->slime_winner.data,a->wgrad_partials.data);
    bs_reduce_partials<<<grid_size(BS_WGRAD_FLOATS),BLOCK_SIZE,0,stream>>>(
        a->wgrad_partials.data,blocks,a->wgrad_entity.data,a->wgrad_slime.data);
}
static void bs_init(void* weights, ulong* seed, cudaStream_t stream) {
    auto* w=(BoplSetWeights*)weights;
    for(auto* tensor:{&w->flat,&w->entity,&w->slime,&w->fusion})
        puf_kaiming_init(tensor,std::sqrt(2.0f),(*seed)++,stream);
}
static void bs_reg_params(void* weights, Allocator* alloc) {
    auto* w=(BoplSetWeights*)weights; int H=w->hidden;
    w->flat={.shape={H,BS_FLAT}}; w->entity={.shape={BS_E,BS_W_ENT}};
    w->slime={.shape={BS_E,BS_W_SLIME}}; w->fusion={.shape={H,H+2*BS_E}};
    // Registration order is also the on-disk and optimizer gradient order.
    for(auto* tensor:{&w->flat,&w->entity,&w->slime,&w->fusion}) alloc_register(alloc,tensor);
}
static void bs_reg_rollout(void* weights, void* activations, Allocator* alloc, int B) {
    auto* w=(BoplSetWeights*)weights; auto* a=(BoplSetActs*)activations; int H=w->hidden;
    a->flat_input={.shape={B,BS_FLAT}};
    a->joined={.shape={B,H+2*BS_E}}; a->out={.shape={B,H}};
    a->entity_winner={.shape={B,BS_E}}; a->slime_winner={.shape={B,BS_E}};
    for(auto* tensor:{&a->flat_input,&a->joined,&a->out}) alloc_register(alloc,tensor);
    alloc_register(alloc,&a->entity_winner); alloc_register(alloc,&a->slime_winner);
    a->gather=BsGather{};
}
static void bs_reg_train(void* weights, void* activations, Allocator* acts, Allocator* grads, int B) {
    auto* w=(BoplSetWeights*)weights; auto* a=(BoplSetActs*)activations; int H=w->hidden;
    bs_reg_rollout(weights,activations,acts,B);
    a->grad_out={.shape={B,H}}; a->grad_joined={.shape={B,H+2*BS_E}}; a->grad_flat={.shape={B,H}};
    a->wgrad_partials={.shape={BS_BWD_MAX_BLOCKS,BS_WGRAD_FLOATS}};
    for(auto* tensor:{&a->grad_out,&a->grad_joined,&a->grad_flat,&a->wgrad_partials}) alloc_register(acts,tensor);
    a->wgrad_flat={.shape={H,BS_FLAT}}; a->wgrad_entity={.shape={BS_E,BS_W_ENT}};
    a->wgrad_slime={.shape={BS_E,BS_W_SLIME}}; a->wgrad_fusion={.shape={H,H+2*BS_E}};
    for(auto* tensor:{&a->wgrad_flat,&a->wgrad_entity,&a->wgrad_slime,&a->wgrad_fusion}) alloc_register(grads,tensor);
}
// Training: read minibatch row (segment s, step t) from rollout_obs (T,
// B_total, obs) at [t][idx[s]] instead of the contiguous input tensor. The
// pointers are baked into the captured train graph, so they must be the
// trainer's fixed buffers. Pass obs = nullptr to go back to contiguous input.
static void bs_bind_gather(void* activations, const precision_t* rollout_obs, const int* idx, int T, int B_total) {
    auto* a=(BoplSetActs*)activations;
    a->gather=BsGather{rollout_obs,idx,T,B_total};
}
static void* bs_create(void* self) {
    auto* encoder=(Encoder*)self;
    if(encoder->in_dim!=BS_OBS) throw std::runtime_error("Bopl SET expects "+std::to_string(BS_OBS)+" observation features (layout v2)");
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
