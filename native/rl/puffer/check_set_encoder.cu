// Standalone qualification harness: no game or trainer is linked.
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "models.cu"
#include "bopl_set_encoder.cu"
#include <cuda.h>
static void checked(cudaError_t result);

// BOPL_CHECK_SMS=<n>: run the harness stream in a CUDA green context of n SMs
// (the trainer's 16-SM training partition) so timings match the trainer.
static cudaStream_t check_make_stream() {
    const char* sms=getenv("BOPL_CHECK_SMS");
    if(sms && atoi(sms)>0) {
        CUdevice dev; CUdevResource all, part, rest; unsigned groups=1; CUdevResourceDesc desc; CUgreenCtx green;
        if(cuInit(0)==CUDA_SUCCESS && cuDeviceGet(&dev,0)==CUDA_SUCCESS
           && cuDeviceGetDevResource(dev,&all,CU_DEV_RESOURCE_TYPE_SM)==CUDA_SUCCESS
           && cuDevSmResourceSplitByCount(&part,&groups,&all,&rest,0,(unsigned)atoi(sms))==CUDA_SUCCESS && groups>=1
           && cuDevResourceGenerateDesc(&desc,&part,1)==CUDA_SUCCESS
           && cuGreenCtxCreate(&green,desc,dev,CU_GREEN_CTX_DEFAULT_STREAM)==CUDA_SUCCESS) {
            CUstream s;
            if(cuGreenCtxStreamCreate(&s,green,CU_STREAM_NON_BLOCKING,0)==CUDA_SUCCESS) {
                fprintf(stderr,"check_set_encoder: %u-SM green context\n",part.sm.smCount);
                return (cudaStream_t)s;
            }
        }
        throw std::runtime_error("BOPL_CHECK_SMS: green context creation failed");
    }
    cudaStream_t s; checked(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking)); return s;
}

static void checked(cudaError_t result) {
    if(result!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
struct SetCheck {
    BoplSetWeights weights{};
    BoplSetActs acts{};
    Allocator params{}, activations{}, gradients{};
    PrecisionTensor obs{}, incoming{};
    // Gather mode: observations live in a (T, B_total, obs) buffer, rows are
    // read through idx (segments = batch / T), like the trainer's minibatch.
    float* obs_tb=nullptr; int* idx=nullptr; long obs_tb_elems=0; int gather_T=0, gather_B=0;
    cudaStream_t stream=nullptr;
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t executable=nullptr;
    ~SetCheck() {
        if(stream) cudaStreamSynchronize(stream);
        if(executable) cudaGraphExecDestroy(executable);
        if(graph) cudaGraphDestroy(graph);
        alloc_free(&params); alloc_free(&activations); alloc_free(&gradients);
        if(obs_tb) cudaFree(obs_tb); if(idx) cudaFree(idx);
        if(stream) cudaStreamDestroy(stream);
    }
    void pass() {
        bs_forward(&weights,&acts,obs,stream);
        bs_backward(&weights,&acts,incoming,stream);
    }
    void reset_graph() {
        if(executable) {cudaGraphExecDestroy(executable); executable=nullptr;}
        if(graph) {cudaGraphDestroy(graph); graph=nullptr;}
    }
};
static thread_local std::string last_error;
extern "C" const char* bopl_set_check_error() {return last_error.c_str();}
extern "C" void* bopl_set_check_create(int batch, int hidden) {
    auto* check=new SetCheck;
    try {
        if(batch<1 || hidden<8 || hidden%8) throw std::runtime_error("Invalid batch/hidden");
        check->weights.hidden=hidden;
        bs_reg_params(&check->weights,&check->params);
        bs_reg_train(&check->weights,&check->acts,&check->activations,&check->gradients,batch);
        check->obs={.shape={batch,BS_OBS}}; check->incoming={.shape={batch,hidden}};
        alloc_register(&check->activations,&check->obs); alloc_register(&check->activations,&check->incoming);
        checked(alloc_create(&check->params)); checked(alloc_create(&check->activations)); checked(alloc_create(&check->gradients));
        if(check->params.total_bytes!=check->gradients.total_bytes)
            throw std::runtime_error("Weight and gradient layout differ");
        check->stream=check_make_stream();
        return check;
    } catch(const std::exception& e) {last_error=e.what(); delete check; return nullptr;}
}
extern "C" long bopl_set_check_weight_count(void* context) {
    return ((SetCheck*)context)->params.total_bytes/sizeof(float);
}
// Switch the context to gather mode (T > 0) or back to contiguous (T == 0).
// idx has batch / T entries in [0, B_total). Drops any captured graph.
extern "C" int bopl_set_check_gather(void* context, int T, int B_total, const int* idx) {
    auto* check=(SetCheck*)context;
    try {
        check->reset_graph();
        int batch=check->obs.shape[0];
        if(T==0) {bs_bind_gather(&check->acts,nullptr,nullptr,0,0); check->gather_T=0; return 0;}
        if(T<1 || batch%T || B_total<1) throw std::runtime_error("gather: batch must be a multiple of T");
        long elems=(long)T*B_total*BS_OBS;
        if(elems!=check->obs_tb_elems) {
            if(check->obs_tb) cudaFree(check->obs_tb); if(check->idx) cudaFree(check->idx);
            checked(cudaMalloc(&check->obs_tb,elems*sizeof(float)));
            checked(cudaMalloc(&check->idx,(batch/T)*sizeof(int)));
            check->obs_tb_elems=elems;
        }
        checked(cudaMemcpy(check->idx,idx,(batch/T)*sizeof(int),cudaMemcpyHostToDevice));
        check->gather_T=T; check->gather_B=B_total;
        bs_bind_gather(&check->acts,check->obs_tb,check->idx,T,B_total);
        return 0;
    } catch(const std::exception& e) {last_error=e.what(); return 1;}
}
// obs is (batch, obs) in contiguous mode and (T, B_total, obs) in gather mode.
extern "C" int bopl_set_check_run(void* context, const float* weights, const float* obs,
                                   const float* incoming, float* output, float* gradients, int graph) {
    auto* check=(SetCheck*)context;
    try {
        checked(cudaMemcpyAsync(check->params.mem,weights,check->params.total_bytes,cudaMemcpyHostToDevice,check->stream));
        if(check->gather_T>0)
            checked(cudaMemcpyAsync(check->obs_tb,obs,check->obs_tb_elems*sizeof(float),cudaMemcpyHostToDevice,check->stream));
        else
            checked(cudaMemcpyAsync(check->obs.data,obs,numel(check->obs.shape)*sizeof(float),cudaMemcpyHostToDevice,check->stream));
        checked(cudaMemcpyAsync(check->incoming.data,incoming,numel(check->incoming.shape)*sizeof(float),cudaMemcpyHostToDevice,check->stream));
        if(graph && !check->executable) {
            check->pass(); // Warm cuBLAS before graph capture.
            checked(cudaStreamSynchronize(check->stream));
            checked(cudaStreamBeginCapture(check->stream,cudaStreamCaptureModeThreadLocal));
            check->pass();
            checked(cudaStreamEndCapture(check->stream,&check->graph));
            checked(cudaGraphInstantiateWithFlags(&check->executable,check->graph,0));
        }
        if(graph) checked(cudaGraphLaunch(check->executable,check->stream));
        else check->pass();
        checked(cudaGetLastError());
        checked(cudaStreamSynchronize(check->stream));
        checked(cudaMemcpy(output,check->acts.out.data,numel(check->acts.out.shape)*sizeof(float),cudaMemcpyDeviceToHost));
        checked(cudaMemcpy(gradients,check->gradients.mem,check->gradients.total_bytes,cudaMemcpyDeviceToHost));
        return 0;
    } catch(const std::exception& e) {last_error=e.what(); return 1;}
}
// Time forward+backward on whatever is resident (ms per pass, GPU time).
extern "C" double bopl_set_check_time(void* context, int iterations, int forward_only) {
    auto* check=(SetCheck*)context;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    for(int i=0;i<3;i++) {if(forward_only) bs_forward(&check->weights,&check->acts,check->obs,check->stream); else check->pass();}
    cudaEventRecord(start,check->stream);
    for(int i=0;i<iterations;i++) {if(forward_only) bs_forward(&check->weights,&check->acts,check->obs,check->stream); else check->pass();}
    cudaEventRecord(stop,check->stream);
    cudaEventSynchronize(stop);
    float ms=0.0f; cudaEventElapsedTime(&ms,start,stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms/iterations;
}
// Select the packed-row forward (BOPL_SET_PACKED) or the every-slot forward.
extern "C" void bopl_set_check_set_packed(int packed) {bs_set_packed(packed);}
// Select the tensor-core (TF32) training forward or the fp32 kernel.
extern "C" void bopl_set_check_set_mma(int mma) {bs_set_mma(mma);}
// bf16 flat block + flat GEMMs (training); takes effect for Checks created afterwards.
extern "C" void bopl_set_check_set_flat_bf16(int on) {bs_set_flat_bf16(on);}
#ifdef BS_PROF
extern "C" void bopl_set_check_prof(double* out) {
    unsigned long long h[4]; cudaMemcpyFromSymbol(h,bs_prof,sizeof(h)); unsigned long long z[4]={0,0,0,0}; cudaMemcpyToSymbol(bs_prof,z,sizeof(z));
    for(int i=0;i<4;i++) out[i]=(double)h[i];
}
#endif
extern "C" void bopl_set_check_close(void* context) {delete (SetCheck*)context;}
