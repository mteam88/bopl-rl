// Standalone qualification harness: no game or trainer is linked.
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "models.cu"
#include "bopl_set_encoder.cu"

static void checked(cudaError_t result) {
    if(result!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
struct SetCheck {
    BoplSetWeights weights{};
    BoplSetActs acts{};
    Allocator params{}, activations{}, gradients{};
    PrecisionTensor obs{}, incoming{};
    cudaStream_t stream=nullptr;
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t executable=nullptr;
    ~SetCheck() {
        if(stream) cudaStreamSynchronize(stream);
        if(executable) cudaGraphExecDestroy(executable);
        if(graph) cudaGraphDestroy(graph);
        alloc_free(&params); alloc_free(&activations); alloc_free(&gradients);
        if(stream) cudaStreamDestroy(stream);
    }
    void pass() {
        bs_forward(&weights,&acts,obs,stream);
        bs_backward(&weights,&acts,incoming,stream);
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
        checked(cudaStreamCreateWithFlags(&check->stream,cudaStreamNonBlocking));
        return check;
    } catch(const std::exception& e) {last_error=e.what(); delete check; return nullptr;}
}
extern "C" long bopl_set_check_weight_count(void* context) {
    return ((SetCheck*)context)->params.total_bytes/sizeof(float);
}
extern "C" int bopl_set_check_run(void* context, const float* weights, const float* obs,
                                   const float* incoming, float* output, float* gradients, int graph) {
    auto* check=(SetCheck*)context;
    try {
        checked(cudaMemcpyAsync(check->params.mem,weights,check->params.total_bytes,cudaMemcpyHostToDevice,check->stream));
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
extern "C" void bopl_set_check_close(void* context) {delete (SetCheck*)context;}
