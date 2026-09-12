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
// Training pads the flat block from 946 to 948 columns (two zero columns and a
// zero-padded copy of the flat weight) so cuBLAS can pick 16-byte-aligned
// tensor-op kernels: with K = 946 the flat forward fell back to an fp32 SIMT
// kernel (1.07 ms) and the flat weight gradient to an align1 kernel (0.79 ms)
// per 24576-row minibatch on 16 SMs. The rollout forward keeps the unpadded path.
static constexpr int BS_FLAT_PAD=(BS_FLAT+3)/4*4;

// Forward tiling: a row is handled by BS_FWD_TPR threads; thread (sg, cg)
// owns channels [4 cg, 4 cg + 4) u [32 + 4 cg, 36 + 4 cg) of one valid slot per
// pass: the valid slots (feature 0 != 0) of a row are compacted in slot order
// and pass p projects valid slots [12 p, 12 p + 12). Bopl rows carry ~10 valid
// entity slots of 96 and ~2 slimes of 12, so projecting every slot (the
// previous tiling) spent ~8x the FMAs the pool consumes. Weights are read from
// shared memory as conflict-free float4s; entity features stay in the row's
// natural (slot, k) layout: a warp reads 4 slots per k, each a broadcast to 8
// lanes. With the compute this light the kernel is bound by row-load latency,
// hence BS_FWD_ROWS rows per block, two blocks per SM and the register
// prefetch of the next rows (BsRowRegs) while the current ones compute.
static constexpr int BS_FWD_ROWS=4, BS_FWD_TPR=96, BS_FWD_THREADS=BS_FWD_ROWS*BS_FWD_TPR;
static constexpr int BS_FWD_CH=8, BS_FWD_SG=BS_FWD_TPR/BS_FWD_CH;  // 12 slot lanes per pass
static constexpr int BS_FWD_MAX_BLOCKS=512;  // 16 iterations of 4 rows per block on 16 SMs (weight tiles amortized)
__device__ __forceinline__ int bs_fwd_channel(int cg, int j) {return (j<4 ? 0 : 32-4)+cg*4+j;}
// Backward tiling: thread = channel, BS_BWD_ROWS rows per block in flight.
static constexpr int BS_BWD_ROWS=4, BS_BWD_THREADS=BS_BWD_ROWS*BS_E, BS_BWD_MAX_BLOCKS=192;
static_assert(BS_FWD_SG*BS_FWD_CH==BS_FWD_TPR, "forward tiling");
static_assert(BS_FWD_TPR==BS_N_ENT && BS_N_ENT==96 && BS_N_SLIME<=32, "validity ballots: 3 entity warps, one slime warp");
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
    PrecisionTensor flat_input;              // (B, 946) rollout / (B, 948) training
    PrecisionTensor joined, out;             // (B, H+128), (B, H)
    IntTensor entity_winner, slime_winner;   // (B, 64) slot index or -1
    PrecisionTensor grad_out, grad_joined, grad_flat;
    PrecisionTensor wgrad_partials;          // (BS_BWD_MAX_BLOCKS, BS_WGRAD_FLOATS)
    PrecisionTensor wgrad_flat, wgrad_entity, wgrad_slime, wgrad_fusion;
    PrecisionTensor flat_pad, wgrad_flat_pad; // training: (H, 948) zero-padded weight / gradient scratch
    BsGather gather;                         // training only; zero = contiguous input
    const float* saved_input;                // contiguous input of the last forward (for backward)
    int train;                               // registered by bs_reg_train (padded flat block)
};

__device__ __forceinline__ const float* bs_row_ptr(const float* input, const BsGather& g, int row) {
    if(g.obs==nullptr) return input+(long)row*BS_OBS;
    int seg=row/g.T, t=row-seg*g.T;
    return g.obs+((long)t*g.B_total+g.idx[seg])*BS_OBS;
}
// Packed rows (BOPL_SET_PACKED=1, the default): bopl_encode fills entity slots
// [0, count) and zeroes the rest, so a row whose slot BS_ENT_HEAD-1 is empty
// has no valid slot beyond it and only its first BS_ENT_HEAD slots (1.3 KB of
// the 8 KB block) are read; rows with more valid slots fetch the remainder on
// a slow path. Bopl rows carry ~10 valid entities. BOPL_SET_PACKED=0 reads
// every slot (arbitrary masks, e.g. the random rows of check_set_encoder).
static constexpr int BS_ENT_HEAD=16;
static_assert(BS_ENT_HEAD<=32 && BS_N_ENT%BS_ENT_HEAD==0, "packed head");
// One observation row split across BS_FWD_TPR threads, held in registers
// between its global loads and its shared / flat_input stores so the loads of
// the next rows are in flight while the current rows compute (the forward is
// a serial load -> compute chain per block otherwise; on a 16-SM partition
// the two resident blocks per SM cannot hide the ~1.5 us row latency).
template<bool PACKED, bool FLAT=true> struct BsRowRegs {
    static constexpr int ENT_FLOATS=PACKED ? BS_ENT_HEAD*BS_W_ENT : BS_ENT_FLOATS;
    static constexpr int ENT2=(ENT_FLOATS/2+BS_FWD_TPR-1)/BS_FWD_TPR;
    static constexpr int SL2=(BS_SLIME_FLOATS/2+BS_FWD_TPR-1)/BS_FWD_TPR;
    static constexpr int PRE2=FLAT ? (BS_PREFIX/2+BS_FWD_TPR-1)/BS_FWD_TPR : 1;
    static constexpr int PREV2=FLAT ? (BS_PREVIOUS/2+BS_FWD_TPR-1)/BS_FWD_TPR : 1;
    float2 ent[ENT2], sl[SL2], pre[PRE2], prev[PREV2];
};
template<bool PACKED, bool FLAT>
__device__ __forceinline__ void bs_row_load(BsRowRegs<PACKED,FLAT>& r, const float* src, int tr) {
    typedef BsRowRegs<PACKED,FLAT> R;
    const float2* e=(const float2*)(src+BS_ENTITY_OFFSET);
    const float2* sl=(const float2*)(src+BS_SLIME_OFFSET);
    const float2* pre=(const float2*)src;
    const float2* prev=(const float2*)(src+BS_PREVIOUS_OFFSET);
    #pragma unroll
    for(int it=0;it<R::ENT2;it++) {int i=tr+it*BS_FWD_TPR; if(i<R::ENT_FLOATS/2) r.ent[it]=e[i];}
    #pragma unroll
    for(int it=0;it<R::SL2;it++) {int i=tr+it*BS_FWD_TPR; if(i<BS_SLIME_FLOATS/2) r.sl[it]=sl[i];}
    if(FLAT) {
        #pragma unroll
        for(int it=0;it<R::PRE2;it++) {int i=tr+it*BS_FWD_TPR; if(i<BS_PREFIX/2) r.pre[it]=pre[i];}
        #pragma unroll
        for(int it=0;it<R::PREV2;it++) {int i=tr+it*BS_FWD_TPR; if(i<BS_PREVIOUS/2) r.prev[it]=prev[i];}
    }
}
// Entities -> Xs (natural layout), slimes -> Ss, flat block -> flat_input row (if live).
template<bool PACKED, bool FLAT>
__device__ __forceinline__ void bs_row_store(const BsRowRegs<PACKED,FLAT>& r, float* xs, float* ss, float* flat, int flat_width, bool live, int tr) {
    typedef BsRowRegs<PACKED,FLAT> R;
    float2* x2=(float2*)xs;
    #pragma unroll
    for(int it=0;it<R::ENT2;it++) {int i=tr+it*BS_FWD_TPR; if(i<R::ENT_FLOATS/2) x2[i]=r.ent[it];}
    float2* s2=(float2*)ss;
    #pragma unroll
    for(int it=0;it<R::SL2;it++) {int i=tr+it*BS_FWD_TPR; if(i<BS_SLIME_FLOATS/2) s2[i]=r.sl[it];}
    if(FLAT && live) {
        float2* f2=(float2*)flat; float2* p2=(float2*)(flat+BS_PREFIX);
        #pragma unroll
        for(int it=0;it<R::PRE2;it++) {int i=tr+it*BS_FWD_TPR; if(i<BS_PREFIX/2) f2[i]=r.pre[it];}
        #pragma unroll
        for(int it=0;it<R::PREV2;it++) {int i=tr+it*BS_FWD_TPR; if(i<BS_PREVIOUS/2) p2[i]=r.prev[it];}
        if(tr<flat_width-BS_FLAT) flat[BS_FLAT+tr]=0.0f;  // zero pad columns (training)
    }
}
// Flat block (prefix + previous action) of a row staged through shared memory
// with cp.async: no registers are held across the compute, so ptxas cannot
// spill a prefetched value (a spill store waits for its load, which puts the
// whole row latency back on the critical path).
static constexpr int BS_FLAT_STAGE=BS_PREFIX+BS_PREVIOUS;
__device__ __forceinline__ void bs_cp_async8(void* smem, const void* gmem) {
    unsigned s=(unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(s), "l"(gmem));
}
__device__ __forceinline__ void bs_flat_stage_issue(float* fs, const float* src, int tr) {
    #pragma unroll
    for(int it=0;it<(BS_PREFIX/2+BS_FWD_TPR-1)/BS_FWD_TPR;it++) {int i=tr+it*BS_FWD_TPR; if(i<BS_PREFIX/2) bs_cp_async8(fs+2*i,src+2*i);}
    #pragma unroll
    for(int it=0;it<(BS_PREVIOUS/2+BS_FWD_TPR-1)/BS_FWD_TPR;it++) {int i=tr+it*BS_FWD_TPR; if(i<BS_PREVIOUS/2) bs_cp_async8(fs+BS_PREFIX+2*i,src+BS_PREVIOUS_OFFSET+2*i);}
    asm volatile("cp.async.commit_group;");
}
__device__ __forceinline__ void bs_flat_stage_wait() {asm volatile("cp.async.wait_group 0;" ::: "memory");}
__device__ __forceinline__ void bs_flat_stage_store(const float* fs, float* flat, int flat_width, int tr) {
    const float2* f2=(const float2*)fs; float2* o2=(float2*)flat;
    #pragma unroll
    for(int it=0;it<(BS_FLAT_STAGE/2+BS_FWD_TPR-1)/BS_FWD_TPR;it++) {int i=tr+it*BS_FWD_TPR; if(i<BS_FLAT_STAGE/2) o2[i]=f2[i];}
    if(tr<flat_width-BS_FLAT) flat[BS_FLAT+tr]=0.0f;  // zero pad columns (training)
}
static_assert(BS_FLAT_STAGE==BS_FLAT && BS_FLAT_STAGE%2==0, "flat block = prefix + previous");

// (value, slot) order of a max-pool: larger value wins, ties go to the lower
// slot; -1 only ever pairs with 0 (no valid slot beat the ReLU floor).
__device__ __forceinline__ void bs_better(float& v, int& i, float ov, int oi) {
    if(ov>v || (ov==v && oi!=-1 && oi<i)) {v=ov; i=oi;}
}

template<bool PACKED>
__global__ void __launch_bounds__(BS_FWD_THREADS, 2) bs_fused_forward(
        const float* __restrict__ input, BsGather g, int B, int H,
        const float* __restrict__ w_ent, const float* __restrict__ w_slime,
        float* __restrict__ flat_input, int flat_width, float* __restrict__ joined,
        int* __restrict__ ent_winner, int* __restrict__ slime_winner) {
    // PACKED: Xs holds one chunk of BS_ENT_HEAD slots per row (the head, then
    // the remaining chunks of a crowded row on the slow path); every slot else.
    constexpr int CHUNK=PACKED ? BS_ENT_HEAD : BS_N_ENT, NCHUNK=BS_N_ENT/CHUNK;
    __shared__ __align__(16) float Xs[BS_FWD_ROWS][CHUNK*BS_W_ENT];          // [slot][k]
    __shared__ __align__(16) float Ss[BS_FWD_ROWS][BS_SLIME_FLOATS];
    __shared__ __align__(16) float Wt[BS_W_ENT][BS_E];                    // [k][channel]
    // Slime weights transposed (PACKED only: the every-slot Xs leaves no room
    // under the 48 KB static limit, that variant reads them through L1).
    __shared__ __align__(16) float Wst[PACKED ? BS_W_SLIME : 1][BS_E];       // [k][channel]
    __shared__ float red_v[BS_FWD_ROWS][2][BS_E];
    __shared__ int red_i[BS_FWD_ROWS][2][BS_E];

    const int tid=threadIdx.x, rg=tid/BS_FWD_TPR, tr=tid-rg*BS_FWD_TPR;
    const int warp_in_row=tr>>5, lane=tr&31;
    const int sg=tr>>3, cg=tr&7;
    const int JW=H+2*BS_E;
    // Transposed weight tiles: consecutive threads store consecutive channels
    // (the [channel][k] -> [k][channel] scatter of a coalesced read is a
    // 32-way bank conflict; the strided L2 reads of 12 KB per block are not).
    for(int i=tid;i<BS_E*BS_W_ENT;i+=BS_FWD_THREADS) Wt[i/BS_E][i%BS_E]=__ldg(w_ent+(i%BS_E)*BS_W_ENT+i/BS_E);
    if(PACKED) for(int i=tid;i<BS_E*BS_W_SLIME;i+=BS_FWD_THREADS) Wst[i/BS_E][i%BS_E]=__ldg(w_slime+(i%BS_E)*BS_W_SLIME+i/BS_E);

    const int stride=gridDim.x*BS_FWD_ROWS;
    BsRowRegs<PACKED> regs;
    if(blockIdx.x*BS_FWD_ROWS<B) {
        int r=blockIdx.x*BS_FWD_ROWS+rg; if(r>=B) r=B-1;
        bs_row_load<PACKED,true>(regs,bs_row_ptr(input,g,r),tr);
    }
    for(int row0=blockIdx.x*BS_FWD_ROWS;row0<B;row0+=stride) {
        const bool live=row0+rg<B;
        const int row=live ? row0+rg : B-1;
        const float* src=bs_row_ptr(input,g,row);
        __syncthreads();  // previous iteration is done with Xs/Ss/red
        bs_row_store<PACKED,true>(regs,Xs[rg],Ss[rg],flat_input+(long)row*flat_width,flat_width,live,tr);
        if(row0+stride<B) {  // prefetch the next rows (in flight during the compute below)
            int r=row0+stride+rg; if(r>=B) r=B-1;
            bs_row_load<PACKED,true>(regs,bs_row_ptr(input,g,r),tr);
        }
        __syncthreads();
        const float* xs=Xs[rg];
        // Masked max-pool over the valid slots (strict >, ReLU floor 0): this
        // thread's slot lane sg takes valid slots sg, sg + 12, ... in slot order,
        // so a strict > keeps the lowest slot on ties like the pooled reference.
        float bv[BS_FWD_CH]; int bi[BS_FWD_CH];
        #pragma unroll
        for(int j=0;j<BS_FWD_CH;j++) {bv[j]=0.0f; bi[j]=-1;}
        // Slow path (PACKED): slot BS_ENT_HEAD-1 of the head is occupied, so
        // the row has valid slots beyond it; the block walks the remaining
        // chunks, rows that ended in the head sit out (the pool state carries
        // across chunks in slot order).
        bool more=PACKED && xs[(CHUNK-1)*BS_W_ENT]!=0.0f;
        for(int chunk=0;chunk<NCHUNK;chunk++) {
            if(chunk>0) {
                if(!__syncthreads_or(more)) break;
                if(more) {
                    const float2* e=(const float2*)(src+BS_ENTITY_OFFSET+chunk*CHUNK*BS_W_ENT); float2* x2=(float2*)Xs[rg];
                    for(int i=tr;i<CHUNK*BS_W_ENT/2;i+=BS_FWD_TPR) x2[i]=e[i];
                }
                __syncthreads();
                if(!more) continue;
                more=xs[(CHUNK-1)*BS_W_ENT]!=0.0f;
            }
            // Valid slots of this chunk as 32-bit masks (slot = 32 w + bit);
            // every warp of the row computes the same masks, so no shared exchange.
            constexpr int NW=(CHUNK+31)/32;
            unsigned em[NW]; int ec[NW];
            #pragma unroll
            for(int w=0;w<NW;w++) {
                int sl=32*w+lane;
                em[w]=__ballot_sync(0xffffffffu,sl<CHUNK && xs[sl*BS_W_ENT]!=0.0f);
                ec[w]=__popc(em[w]);
            }
            int nvalid=0;
            #pragma unroll
            for(int w=0;w<NW;w++) nvalid+=ec[w];
            // Packed rows: the valid slots are [0, nvalid) and the n-th is n;
            // arbitrary masks (BOPL_SET_PACKED=0, or a non-packed producer) fall
            // back to a bit search.
            bool contiguous=(em[0]&(em[0]+1u))==0u;
            #pragma unroll
            for(int w=1;w<NW;w++) contiguous=contiguous && em[w]==0u;
            for(int n=sg;n<nvalid;n+=BS_FWD_SG) {
                int slot=n;
                if(!contiguous) {
                    int rem=n;
                    #pragma unroll
                    for(int w=0;w<NW;w++) {
                        if(rem<ec[w]) {slot=32*w+__fns(em[w],0,rem+1); break;}
                        rem-=ec[w];
                    }
                }
                const float* x_slot=xs+slot*BS_W_ENT;
                int cgl=cg; asm volatile("" : "+r"(cgl));  // re-read Wt per pass: hoisting its 168 floats spills
                float acc[BS_FWD_CH];
                #pragma unroll
                for(int j=0;j<BS_FWD_CH;j++) acc[j]=0.0f;
                #pragma unroll 7  // fully unrolled, ptxas hoists all 42 float4 weight loads and spills
                for(int k=0;k<BS_W_ENT;k++) {
                    float4 w0=*(const float4*)&Wt[k][cgl*4], w1=*(const float4*)&Wt[k][32+cgl*4];
                    float x=x_slot[k];
                    const float w[BS_FWD_CH]={w0.x,w0.y,w0.z,w0.w,w1.x,w1.y,w1.z,w1.w};
                    #pragma unroll
                    for(int j=0;j<BS_FWD_CH;j++) acc[j]=fmaf(x,w[j],acc[j]);
                }
                slot+=chunk*CHUNK;
                #pragma unroll
                for(int j=0;j<BS_FWD_CH;j++)
                    if(acc[j]>bv[j]) {bv[j]=acc[j]; bi[j]=slot;}
            }
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
        // Slime: thread = channel for the first 64 threads of the row, valid
        // slots in slot order from the ballot (~2 valid slimes per row).
        const unsigned smask=__ballot_sync(0xffffffffu,lane<BS_N_SLIME && Ss[rg][lane*BS_W_SLIME]!=0.0f);
        float sv=0.0f; int si=-1;
        if(tr<BS_E) {
            unsigned m=smask;
            while(m) {
                int s=__ffs(m)-1; m&=m-1;
                const float* xsl=Ss[rg]+s*BS_W_SLIME;
                float a=0.0f;
                #pragma unroll
                for(int k=0;k<BS_W_SLIME;k++) a=fmaf(xsl[k],PACKED ? Wst[k][tr] : __ldg(w_slime+tr*BS_W_SLIME+k),a);
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

// ---- Tensor-core forward (training default path) ----------------------------
// The fp32 kernel above is bound by shared-memory instruction issue on the
// 16-SM partition (per row and valid slot, 64 channels x 21 features of FMAs
// fed from shared memory: ~1 us of the ~2.7 us per 4-row iteration). This
// variant projects a 16-slot tile against the 64 channels with
// mma.sync m16n8k8 TF32 (inputs rounded to a 10-bit mantissa, fp32
// accumulate, same precision class as the TF32 flat GEMM of the default path)
// then pools the accumulator fragments over the valid slots with warp
// shuffles. One warp per row does the entity block (two: one per half of the
// channels), a third the slimes. BOPL_TF32=0 (exact) keeps the fp32 kernel.
__device__ __forceinline__ unsigned bs_tf32(float x) {
    unsigned r; asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x)); return r;
}
__device__ __forceinline__ void bs_mma_tf32(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// One 16-slot tile of a row as the B operand (k x slot) of m16n8k8: two
// n-tiles of 8 slots, KSTEPS k-steps. Weights are the A operand (channel x k),
// so D = [channel][slot] and the pool over slots runs along the fragment
// columns: a lane holds 4 slots of 2 channels per 16-channel m-tile and the
// tile's max needs two shuffle rounds (over the lane's gc bits).
template<int K, int NSLOT> struct BsMmaTile {
    static constexpr int KSTEPS=(K+7)/8, KPAD=KSTEPS*8;
    static constexpr int WSTRIDE=KPAD+4;  // weight row stride in shared memory: conflict-free A fragments
    unsigned b[2][KSTEPS][2];
    unsigned valid;
    // xs = [slot][K] features (rows >= NSLOT read as 0), TF32-rounded.
    __device__ __forceinline__ void load(const float* xs, unsigned valid_, int lane) {
        const int gr=lane>>2, gc=lane&3;
        valid=valid_;
        #pragma unroll
        for(int nt=0;nt<2;nt++) {
            const int slot=nt*8+gr;
            #pragma unroll
            for(int ks=0;ks<KSTEPS;ks++) {
                int k0=ks*8+gc;
                b[nt][ks][0]=bs_tf32(k0<K && slot<NSLOT ? xs[slot*K+k0] : 0.0f);
                b[nt][ks][1]=bs_tf32(k0+4<K && slot<NSLOT ? xs[slot*K+k0+4] : 0.0f);
            }
        }
    }
    // Channels [m0, m0 + 16) against w = [64][WSTRIDE] TF32 weights ([channel][k],
    // k >= K zero; WG: from global memory [channel][K] instead), pooled over
    // the tile's valid slots (slot index base + column). FIRST: this is the
    // first tile pooled into pool_v/pool_i for these channels (plain store),
    // else merge with the state there.
    template<bool WG, bool FIRST>
    __device__ __forceinline__ void pool(const float* w, int m0, int base, float* pool_v, int* pool_i, int lane) const {
        const int gr=lane>>2, gc=lane&3;
        unsigned a[KSTEPS][4];
        #pragma unroll
        for(int ks=0;ks<KSTEPS;ks++) {
            int k0=ks*8+gc;
            if(WG) {
                a[ks][0]=bs_tf32(k0<K ? __ldg(w+(m0+gr)*K+k0) : 0.0f);
                a[ks][1]=bs_tf32(k0<K ? __ldg(w+(m0+gr+8)*K+k0) : 0.0f);
                a[ks][2]=bs_tf32(k0+4<K ? __ldg(w+(m0+gr)*K+k0+4) : 0.0f);
                a[ks][3]=bs_tf32(k0+4<K ? __ldg(w+(m0+gr+8)*K+k0+4) : 0.0f);
            } else {
                a[ks][0]=__float_as_uint(w[(m0+gr)*WSTRIDE+k0]);
                a[ks][1]=__float_as_uint(w[(m0+gr+8)*WSTRIDE+k0]);
                a[ks][2]=__float_as_uint(w[(m0+gr)*WSTRIDE+k0+4]);
                a[ks][3]=__float_as_uint(w[(m0+gr+8)*WSTRIDE+k0+4]);
            }
        }
        float d[2][4];
        #pragma unroll
        for(int nt=0;nt<2;nt++) {
            d[nt][0]=d[nt][1]=d[nt][2]=d[nt][3]=0.0f;
            #pragma unroll
            for(int ks=0;ks<KSTEPS;ks++) bs_mma_tf32(d[nt],a[ks],b[nt][ks][0],b[nt][ks][1]);
        }
        // d[nt][j]: channel m0 + gr, slot 8 nt + 2 gc + j; d[nt][2 + j]: channel
        // m0 + gr + 8. ReLU floor and validity, strict max (lowest slot on
        // ties) over the lane's 4 slots, then over the 4 lanes sharing gr.
        #pragma unroll
        for(int h=0;h<2;h++) {
            float bv=0.0f; int bi=-1;
            #pragma unroll
            for(int nt=0;nt<2;nt++)
                #pragma unroll
                for(int j=0;j<2;j++) {
                    int sl=8*nt+2*gc+j; float v=d[nt][2*h+j];
                    bool ok=((valid>>sl)&1u) && v>0.0f;
                    bs_better(bv,bi,ok ? v : 0.0f,ok ? base+sl : -1);
                }
            #pragma unroll
            for(int off=1;off<=2;off<<=1) {
                float ov=__shfl_xor_sync(0xffffffffu,bv,off); int oi=__shfl_xor_sync(0xffffffffu,bi,off);
                bs_better(bv,bi,ov,oi);
            }
            if(gc==0) {
                int ch=m0+gr+8*h;
                if(FIRST) {pool_v[ch]=bv; pool_i[ch]=bi;}
                else bs_better(pool_v[ch],pool_i[ch],bv,bi);
            }
        }
    }
};

__device__ __forceinline__ bool gc0(int lane) {return (lane&3)==0;}
template<bool PACKED>
__global__ void __launch_bounds__(BS_FWD_THREADS, 2) bs_fused_forward_mma(
        const float* __restrict__ input, BsGather g, int B, int H,
        const float* __restrict__ w_ent, const float* __restrict__ w_slime,
        float* __restrict__ flat_input, int flat_width, float* __restrict__ joined,
        int* __restrict__ ent_winner, int* __restrict__ slime_winner) {
    constexpr int CHUNK=PACKED ? BS_ENT_HEAD : BS_N_ENT, NCHUNK=BS_N_ENT/CHUNK;
    typedef BsMmaTile<BS_W_ENT,16> EntTile; typedef BsMmaTile<BS_W_SLIME,BS_N_SLIME> SlimeTile;
    constexpr int NTILE=2*BS_E/16;  // 4 entity + 4 slime 16-channel tiles per row
    __shared__ __align__(16) float Xs[BS_FWD_ROWS][CHUNK*BS_W_ENT];          // [slot][k]
    __shared__ __align__(16) float Ss[BS_FWD_ROWS][BS_SLIME_FLOATS];
    __shared__ __align__(16) float We[BS_E][EntTile::WSTRIDE];              // [channel][k], TF32, k >= 21 zero
    __shared__ __align__(16) float Ws[PACKED ? BS_E : 1][SlimeTile::WSTRIDE]; // (static limit, see above)
    __shared__ float pool_v[BS_FWD_ROWS][2*BS_E];
    __shared__ int pool_i[BS_FWD_ROWS][2*BS_E];
    constexpr bool STAGE=PACKED;  // flat block staging (cp.async); the every-slot Xs leaves no static room
    __shared__ __align__(16) float Fs[STAGE ? BS_FWD_ROWS : 1][STAGE ? BS_FLAT_STAGE : 2];

    const int tid=threadIdx.x, rg=tid/BS_FWD_TPR, tr=tid-rg*BS_FWD_TPR;
    const int warp_in_row=tr>>5, lane=tr&31;
    const int JW=H+2*BS_E;
    for(int i=tid;i<BS_E*EntTile::WSTRIDE;i+=BS_FWD_THREADS) {
        int c=i/EntTile::WSTRIDE, kk=i%EntTile::WSTRIDE;
        We[c][kk]=kk<BS_W_ENT ? __uint_as_float(bs_tf32(__ldg(w_ent+c*BS_W_ENT+kk))) : 0.0f;
    }
    if(PACKED) for(int i=tid;i<BS_E*SlimeTile::WSTRIDE;i+=BS_FWD_THREADS) {
        int c=i/SlimeTile::WSTRIDE, kk=i%SlimeTile::WSTRIDE;
        Ws[c][kk]=kk<BS_W_SLIME ? __uint_as_float(bs_tf32(__ldg(w_slime+c*BS_W_SLIME+kk))) : 0.0f;
    }

    const int stride=gridDim.x*BS_FWD_ROWS;
    BsRowRegs<PACKED,!STAGE> regs;
    if(blockIdx.x*BS_FWD_ROWS<B) {
        int r=blockIdx.x*BS_FWD_ROWS+rg; if(r>=B) r=B-1;
        bs_row_load<PACKED,!STAGE>(regs,bs_row_ptr(input,g,r),tr);
        if constexpr(STAGE) bs_flat_stage_issue(Fs[rg],bs_row_ptr(input,g,r),tr);
    }
    // The 8 channel tiles of a row (4 entity, 4 slime) go round-robin over
    // its 3 warps; a channel is pooled by exactly one warp.
    float* pv=pool_v[rg]; int* pi=pool_i[rg];
    for(int row0=blockIdx.x*BS_FWD_ROWS;row0<B;row0+=stride) {
        const bool live=row0+rg<B;
        const int row=live ? row0+rg : B-1;
        const float* src=bs_row_ptr(input,g,row);
        if constexpr(STAGE) bs_flat_stage_wait();
        __syncthreads();  // previous iteration is done with Xs/Ss/pool; Fs landed
        bs_row_store<PACKED,!STAGE>(regs,Xs[rg],Ss[rg],flat_input+(long)row*flat_width,flat_width,live,tr);
        if constexpr(STAGE) {if(live) bs_flat_stage_store(Fs[rg],flat_input+(long)row*flat_width,flat_width,tr);}
        const bool prefetch=row0+stride<B;
        int rnext=row0+stride+rg; if(rnext>=B) rnext=B-1;
        if(prefetch) bs_row_load<PACKED,!STAGE>(regs,bs_row_ptr(input,g,rnext),tr);  // in flight during the compute
        __syncthreads();  // Fs read out: restage
        if constexpr(STAGE) {if(prefetch) bs_flat_stage_issue(Fs[rg],bs_row_ptr(input,g,rnext),tr);}
        const float* xs=Xs[rg];
        bool more=PACKED && xs[(CHUNK-1)*BS_W_ENT]!=0.0f;
        for(int chunk=0;chunk<NCHUNK;chunk++) {
            if(chunk>0) {
                if(!__syncthreads_or(more)) break;
                if(more) {
                    const float2* e=(const float2*)(src+BS_ENTITY_OFFSET+chunk*CHUNK*BS_W_ENT); float2* x2=(float2*)Xs[rg];
                    for(int i=tr;i<CHUNK*BS_W_ENT/2;i+=BS_FWD_TPR) x2[i]=e[i];
                }
                __syncthreads();
                if(!more) continue;
                more=xs[(CHUNK-1)*BS_W_ENT]!=0.0f;
            }
            SlimeTile slime;
            const unsigned svalid=chunk==0 ? __ballot_sync(0xffffffffu,lane<BS_N_SLIME && Ss[rg][lane*BS_W_SLIME]!=0.0f) : 0u;
            if(svalid) slime.load(Ss[rg],svalid,lane);
            #pragma unroll 1
            for(int t=0;t<CHUNK/16;t++) {
                const unsigned valid=__ballot_sync(0xffffffffu,lane<16 && xs[(t*16+lane)*BS_W_ENT]!=0.0f);
                EntTile ent;
                if(valid) ent.load(xs+t*16*BS_W_ENT,valid,lane);
                const bool first=chunk==0 && t==0;
                #pragma unroll
                for(int tile=warp_in_row;tile<NTILE;tile+=3) {
                    if(tile<NTILE/2) {
                        const int m0=tile*16, base=chunk*CHUNK+t*16;
                        if(valid) {if(first) ent.template pool<false,true>(&We[0][0],m0,base,pv,pi,lane); else ent.template pool<false,false>(&We[0][0],m0,base,pv,pi,lane);}
                        else if(first && gc0(lane)) {pv[m0+(lane>>2)]=0.0f; pi[m0+(lane>>2)]=-1; pv[m0+8+(lane>>2)]=0.0f; pi[m0+8+(lane>>2)]=-1;}
                    } else if(first) {
                        const int m0=(tile-NTILE/2)*16;
                        if(svalid) {
                            if(PACKED) slime.template pool<false,true>(&Ws[0][0],m0,0,pv+BS_E,pi+BS_E,lane);
                            else slime.template pool<true,true>(w_slime,m0,0,pv+BS_E,pi+BS_E,lane);
                        } else if(gc0(lane)) {pv[BS_E+m0+(lane>>2)]=0.0f; pi[BS_E+m0+(lane>>2)]=-1; pv[BS_E+m0+8+(lane>>2)]=0.0f; pi[BS_E+m0+8+(lane>>2)]=-1;}
                    }
                }
            }
        }
        __syncthreads();
        if(live && warp_in_row<2) {
            // Warp 0 writes the entity channels, warp 1 the slimes.
            const int o=warp_in_row*BS_E;
            #pragma unroll
            for(int h=0;h<2;h++) {
                int c=32*h+lane;
                joined[(long)row*JW+H+o+c]=pv[o+c];
                (warp_in_row ? slime_winner : ent_winner)[(long)row*BS_E+c]=pi[o+c];
            }
        }
    }
}

// dW partials: partials[block][c][f] = sum over the block's rows of
// g[row, c] * x[row, winner[row, c], f] for the entity block, then slime.
// Only the winning slots are read (a row's 64 channels pick among its ~10
// valid slots), straight from the rollout buffer: staging the whole 8 KB
// entity block per row made this pass read 230 MB per minibatch.
__global__ void __launch_bounds__(BS_BWD_THREADS, 3) bs_fused_backward(
        const float* __restrict__ input, BsGather g, int B, int H,
        const float* __restrict__ grad_joined, const int* __restrict__ ent_winner,
        const int* __restrict__ slime_winner, float* __restrict__ partials) {
    __shared__ __align__(16) float red[BS_BWD_ROWS*BS_E*BS_W_SLIME];
    const int tid=threadIdx.x, rg=tid/BS_E, c=tid-rg*BS_E;
    const int JW=H+2*BS_E;
    float ae[BS_W_ENT], as[BS_W_SLIME];
    #pragma unroll
    for(int f=0;f<BS_W_ENT;f++) ae[f]=0.0f;
    #pragma unroll
    for(int f=0;f<BS_W_SLIME;f++) as[f]=0.0f;

    // The pass is a chain of dependent loads per row (winner index, then the
    // slot's features); the next row's indices and gradients are fetched while
    // the current row's features are in flight so each row costs one latency.
    const int stride=gridDim.x*BS_BWD_ROWS;
    int row=blockIdx.x*BS_BWD_ROWS+rg;
    int we=-1, ws=-1; float ge=0.0f, gs=0.0f;
    if(row<B) {
        we=ent_winner[(long)row*BS_E+c]; ws=slime_winner[(long)row*BS_E+c];
        ge=grad_joined[(long)row*JW+H+c]; gs=grad_joined[(long)row*JW+H+BS_E+c];
    }
    for(;row<B;row+=stride) {
        const float* src=bs_row_ptr(input,g,row);
        const int cwe=we, cws=ws; const float cge=ge, cgs=gs;
        const int next=row+stride;
        if(next<B) {
            we=ent_winner[(long)next*BS_E+c]; ws=slime_winner[(long)next*BS_E+c];
            ge=grad_joined[(long)next*JW+H+c]; gs=grad_joined[(long)next*JW+H+BS_E+c];
        }
        if(cwe>=0) {
            const float* x=src+BS_ENTITY_OFFSET+cwe*BS_W_ENT;
            float xv[BS_W_ENT];
            #pragma unroll
            for(int f=0;f<BS_W_ENT;f++) xv[f]=__ldg(x+f);
            #pragma unroll
            for(int f=0;f<BS_W_ENT;f++) ae[f]=fmaf(cge,xv[f],ae[f]);
        }
        if(cws>=0) {
            const float* x=src+BS_SLIME_OFFSET+cws*BS_W_SLIME;
            float xv[BS_W_SLIME];
            #pragma unroll
            for(int f=0;f<BS_W_SLIME;f++) xv[f]=__ldg(x+f);
            #pragma unroll
            for(int f=0;f<BS_W_SLIME;f++) as[f]=fmaf(cgs,xv[f],as[f]);
        }
    }
    // Reduce the BS_BWD_ROWS row groups through shared memory (fixed order).
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
// dst (H, 948) = w (H, 946) with two zero columns; and back for the gradient.
__global__ void bs_pad_rows(float* __restrict__ dst, const float* __restrict__ src, int H) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=H*BS_FLAT_PAD) return;
    int r=i/BS_FLAT_PAD, c=i-r*BS_FLAT_PAD;
    dst[i]=c<BS_FLAT ? src[r*BS_FLAT+c] : 0.0f;
}
__global__ void bs_unpad_rows(float* __restrict__ dst, const float* __restrict__ src, int H) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=H*BS_FLAT) return;
    int r=i/BS_FLAT, c=i-r*BS_FLAT;
    dst[i]=src[r*BS_FLAT_PAD+c];
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
// handle/workspace as puf_mm). Training passes puf_mm's compute type (TF32
// unless BOPL_TF32=0): with exact fp32 this 24576 x 948 x 128 GEMM ran an
// fp32 SIMT kernel, 1.07 ms of the minibatch on 16 SMs, 0.6 ms as TF32. The
// rollout forward keeps exact fp32 (its 112-row GEMM is a few microseconds).
static void bs_mm_ldc(PrecisionTensor* a, PrecisionTensor* w, precision_t* out, int ldc, cudaStream_t stream,
                      cublasComputeType_t compute=CUBLAS_COMPUTE_PRECISION) {
    int M=a->shape[0], K=a->shape[1], N=w->shape[0];
    float alpha=1.0f, beta=0.0f;
    cublasHandle_t handle=cublas_get_handle(stream);
    cublasSetStream(handle,stream);
    cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,N,M,K,&alpha,
        w->data,CUBLAS_PRECISION,K,a->data,CUBLAS_PRECISION,K,&beta,
        out,CUBLAS_PRECISION,ldc,compute,CUBLAS_GEMM_DEFAULT);
}
static int bs_fwd_blocks(int B) {int n=(B+BS_FWD_ROWS-1)/BS_FWD_ROWS; return n<BS_FWD_MAX_BLOCKS ? n : BS_FWD_MAX_BLOCKS;}
// BOPL_SET_PACKED (default 1): rows follow bopl_encode's packed slot layout;
// bs_set_packed overrides it (check_set_encoder exercises both kernels).
static int g_bs_packed=-1;
static bool bs_packed() {
    if(g_bs_packed<0) {const char* e=getenv("BOPL_SET_PACKED"); g_bs_packed=!(e && e[0]=='0');}
    return g_bs_packed!=0;
}
void bs_set_packed(int packed) {g_bs_packed=packed ? 1 : 0;}
// BOPL_SET_MMA: training forward on tensor cores (TF32). Default: follows
// BOPL_TF32 (on unless exact fp32 GEMMs were requested). The rollout forward
// always uses the fp32 kernel. bs_set_mma overrides (check_set_encoder).
static int g_bs_mma=-1;
static bool bs_mma() {
    if(g_bs_mma<0) {
        const char* e=getenv("BOPL_SET_MMA");
        g_bs_mma=e && e[0] ? atoi(e)!=0 : bopl_cublas_compute_type()==CUBLAS_COMPUTE_32F_FAST_TF32;
    }
    return g_bs_mma!=0;
}
void bs_set_mma(int mma) {g_bs_mma=mma ? 1 : 0;}
static int bs_bwd_blocks(int B) {int n=(B+BS_BWD_ROWS-1)/BS_BWD_ROWS; return n<BS_BWD_MAX_BLOCKS ? n : BS_BWD_MAX_BLOCKS;}

static PrecisionTensor bs_forward(void* weights, void* activations, PrecisionTensor input, cudaStream_t stream) {
    auto* w=(BoplSetWeights*)weights; auto* a=(BoplSetActs*)activations;
    int B=a->out.shape[0], H=w->hidden;
    a->saved_input=input.data;
    int flat_width=a->train ? BS_FLAT_PAD : BS_FLAT;
    auto kernel=a->train && bs_mma() ? (bs_packed() ? bs_fused_forward_mma<true> : bs_fused_forward_mma<false>)
                                     : (bs_packed() ? bs_fused_forward<true> : bs_fused_forward<false>);
    kernel<<<bs_fwd_blocks(B),BS_FWD_THREADS,0,stream>>>(
        input.data,a->gather,B,H,w->entity.data,w->slime.data,
        a->flat_input.data,flat_width,a->joined.data,a->entity_winner.data,a->slime_winner.data);
    if(a->train) {
        bs_pad_rows<<<grid_size(H*BS_FLAT_PAD),BLOCK_SIZE,0,stream>>>(a->flat_pad.data,w->flat.data,H);
        bs_mm_ldc(&a->flat_input,&a->flat_pad,a->joined.data,H+2*BS_E,stream,bopl_cublas_compute_type());
    } else {
        bs_mm_ldc(&a->flat_input,&w->flat,a->joined.data,H+2*BS_E,stream);
    }
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
    if(a->train) {
        puf_mm_tn(&a->grad_flat,&a->flat_input,&a->wgrad_flat_pad,stream);
        bs_unpad_rows<<<grid_size(H*BS_FLAT),BLOCK_SIZE,0,stream>>>(a->wgrad_flat.data,a->wgrad_flat_pad.data,H);
    } else {
        puf_mm_tn(&a->grad_flat,&a->flat_input,&a->wgrad_flat,stream);
    }
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
static void bs_reg_forward(void* weights, void* activations, Allocator* alloc, int B, int train) {
    auto* w=(BoplSetWeights*)weights; auto* a=(BoplSetActs*)activations; int H=w->hidden;
    a->train=train;
    a->flat_input={.shape={B,train ? BS_FLAT_PAD : BS_FLAT}};
    a->joined={.shape={B,H+2*BS_E}}; a->out={.shape={B,H}};
    a->entity_winner={.shape={B,BS_E}}; a->slime_winner={.shape={B,BS_E}};
    for(auto* tensor:{&a->flat_input,&a->joined,&a->out}) alloc_register(alloc,tensor);
    alloc_register(alloc,&a->entity_winner); alloc_register(alloc,&a->slime_winner);
    a->gather=BsGather{};
}
static void bs_reg_rollout(void* weights, void* activations, Allocator* alloc, int B) {
    bs_reg_forward(weights,activations,alloc,B,0);
}
static void bs_reg_train(void* weights, void* activations, Allocator* acts, Allocator* grads, int B) {
    auto* w=(BoplSetWeights*)weights; auto* a=(BoplSetActs*)activations; int H=w->hidden;
    bs_reg_forward(weights,activations,acts,B,1);
    a->flat_pad={.shape={H,BS_FLAT_PAD}}; a->wgrad_flat_pad={.shape={H,BS_FLAT_PAD}};
    a->grad_out={.shape={B,H}}; a->grad_joined={.shape={B,H+2*BS_E}}; a->grad_flat={.shape={B,H}};
    a->wgrad_partials={.shape={BS_BWD_MAX_BLOCKS,BS_WGRAD_FLOATS}};
    for(auto* tensor:{&a->flat_pad,&a->wgrad_flat_pad,&a->grad_out,&a->grad_joined,&a->grad_flat,&a->wgrad_partials}) alloc_register(acts,tensor);
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
