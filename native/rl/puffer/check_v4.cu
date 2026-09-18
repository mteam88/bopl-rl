// Standalone qualification harness for policy v4 (bopl_v4.cu): the encoder
// and the decoder back to back (the MinGRU between them is upstream's, already
// qualified by check_recurrent). No game or trainer is linked.
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "models.cu"
#include "bopl_set_encoder.cu"
#include "bopl_v4.cu"

static void checked(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
struct V4Check {
    Encoder enc{}; Decoder dec{};
    void *ew = nullptr, *dw = nullptr;
    v4::EncActs* ea = nullptr; v4::DecActs* da = nullptr;
    Allocator params{}, activations{}, gradients{};
    PrecisionTensor obs{};
    FloatTensor grad_logits{}, grad_value{}, none{};
    float* obs_tb = nullptr; int* idx = nullptr; long obs_tb_elems = 0; int gather_T = 0, gather_B = 0;
    int batch = 0, hidden = 0, od = 0;
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr; cudaGraphExec_t executable = nullptr;
    ~V4Check() {
        if (stream) cudaStreamSynchronize(stream);
        reset_graph();
        alloc_free(&params); alloc_free(&activations); alloc_free(&gradients);
        if (obs_tb) cudaFree(obs_tb); if (idx) cudaFree(idx);
        if (ew) v4::enc_free(ew); if (dw) v4::dec_free(dw); free(ea); free(da);
    }
    void forward() {
        PrecisionTensor h = v4::enc_forward(ew, ea, obs, stream);
        v4::dec_forward(dw, da, h, stream);
    }
    void pass() {
        forward();
        PrecisionTensor g = v4::dec_backward(dw, da, grad_logits, none, grad_value, stream);
        v4::enc_backward(ew, ea, g, stream);
    }
    void reset_graph() {
        if (executable) { cudaGraphExecDestroy(executable); executable = nullptr; }
        if (graph) { cudaGraphDestroy(graph); graph = nullptr; }
    }
};
static thread_local std::string last_error;
static cudaStream_t g_stream = nullptr;
extern "C" const char* bopl_v4_check_error() { return last_error.c_str(); }
extern "C" void* bopl_v4_check_create(int batch, int hidden, int od) {
    auto* c = new V4Check;
    try {
        if (batch < 1 || hidden < 64 || hidden % 64 || od < 1) throw std::runtime_error("Invalid batch/hidden/od");
        c->batch = batch; c->hidden = hidden; c->od = od;
        c->enc.in_dim = BOPL_OBS_STRIDE; c->enc.out_dim = hidden;
        bopl_create_v4_encoder(&c->enc);
        c->dec.hidden_dim = hidden; c->dec.output_dim = od; c->dec.continuous = false;
        bopl_create_v4_decoder(&c->dec);
        c->ew = c->enc.create_weights(&c->enc); c->dw = c->dec.create_weights(&c->dec);
        c->ea = (v4::EncActs*)calloc(1, sizeof(v4::EncActs)); c->da = (v4::DecActs*)calloc(1, sizeof(v4::DecActs));
        c->enc.reg_params(c->ew, &c->params); c->dec.reg_params(c->dw, &c->params);
        c->enc.reg_train(c->ew, c->ea, &c->activations, &c->gradients, batch);
        c->dec.reg_train(c->dw, c->da, &c->activations, &c->gradients, batch);
        c->obs = {.shape = {batch, BOPL_OBS_STRIDE}}; alloc_register(&c->activations, &c->obs);
        c->grad_logits = {.shape = {batch, od}}; alloc_register(&c->activations, &c->grad_logits);
        c->grad_value = {.shape = {batch}}; alloc_register(&c->activations, &c->grad_value);
        checked(alloc_create(&c->params)); checked(alloc_create(&c->activations)); checked(alloc_create(&c->gradients));
        checked(cudaDeviceSynchronize());
        if (c->params.total_bytes != c->gradients.total_bytes) throw std::runtime_error("Weight and gradient layout differ");
        if (!g_stream) checked(cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking));
        c->stream = g_stream;
        return c;
    } catch (const std::exception& e) { last_error = e.what(); delete c; return nullptr; }
}
extern "C" long bopl_v4_check_weight_count(void* context) { return ((V4Check*)context)->params.total_bytes / sizeof(float); }
extern "C" int bopl_v4_check_out_width(void* context) { return ((V4Check*)context)->da->base.od1_pad; }
// Parameter tensors in allocator order: writes (rows, cols) pairs, returns the count.
extern "C" int bopl_v4_check_param_shapes(void* context, long* out, int max_pairs) {
    auto* c = (V4Check*)context;
    for (int i = 0; i < c->params.num_regs && i < max_pairs; i++) {
        int64_t* shape = c->params.regs[i].shape;
        out[2 * i] = shape[0]; out[2 * i + 1] = shape[1] ? shape[1] : 1;
    }
    return c->params.num_regs;
}
extern "C" int bopl_v4_check_gather(void* context, int T, int B_total, const int* idx) {
    auto* c = (V4Check*)context;
    try {
        c->reset_graph();
        if (T == 0) { bopl_v4_bind_gather(c->ea, nullptr, nullptr, 0, 0); c->gather_T = 0; return 0; }
        if (T < 1 || c->batch % T || B_total < 1) throw std::runtime_error("gather: batch must be a multiple of T");
        long elems = (long)T * B_total * BOPL_OBS_STRIDE;
        if (elems != c->obs_tb_elems) {
            if (c->obs_tb) cudaFree(c->obs_tb); if (c->idx) cudaFree(c->idx);
            checked(cudaMalloc(&c->obs_tb, elems * sizeof(float)));
            checked(cudaMalloc(&c->idx, (c->batch / T) * sizeof(int)));
            c->obs_tb_elems = elems;
        }
        checked(cudaMemcpy(c->idx, idx, (c->batch / T) * sizeof(int), cudaMemcpyHostToDevice));
        c->gather_T = T; c->gather_B = B_total;
        bopl_v4_bind_gather(c->ea, c->obs_tb, c->idx, T, B_total);
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return 1; }
}
// obs is (batch, stride) or, in gather mode, (T, B_total, stride). Outputs: the
// encoder output (batch, hidden), the decoder rows (batch, out_width), the
// weight gradients (weight_count) and the per-set overflow (4).
extern "C" int bopl_v4_check_run(void* context, const float* weights, const float* obs, const float* grad_logits,
                                 const float* grad_value, float* enc_out, float* dec_out, float* gradients, int* overflow, int graph) {
    auto* c = (V4Check*)context;
    try {
        checked(cudaMemcpyAsync(c->params.mem, weights, c->params.total_bytes, cudaMemcpyHostToDevice, c->stream));
        if (c->gather_T > 0) checked(cudaMemcpyAsync(c->obs_tb, obs, c->obs_tb_elems * sizeof(float), cudaMemcpyHostToDevice, c->stream));
        else checked(cudaMemcpyAsync(c->obs.data, obs, numel(c->obs.shape) * sizeof(float), cudaMemcpyHostToDevice, c->stream));
        checked(cudaMemcpyAsync(c->grad_logits.data, grad_logits, numel(c->grad_logits.shape) * sizeof(float), cudaMemcpyHostToDevice, c->stream));
        checked(cudaMemcpyAsync(c->grad_value.data, grad_value, numel(c->grad_value.shape) * sizeof(float), cudaMemcpyHostToDevice, c->stream));
        if (graph && !c->executable) {
            c->pass();
            checked(cudaStreamSynchronize(c->stream));
            checked(cudaStreamBeginCapture(c->stream, cudaStreamCaptureModeThreadLocal));
            c->pass();
            checked(cudaStreamEndCapture(c->stream, &c->graph));
            checked(cudaGraphInstantiateWithFlags(&c->executable, c->graph, 0));
        }
        if (graph) checked(cudaGraphLaunch(c->executable, c->stream));
        else c->pass();
        checked(cudaGetLastError());
        checked(cudaStreamSynchronize(c->stream));
        checked(cudaMemcpy(enc_out, c->ea->out.data, numel(c->ea->out.shape) * sizeof(float), cudaMemcpyDeviceToHost));
        checked(cudaMemcpy(dec_out, c->da->base.out.data, numel(c->da->base.out.shape) * sizeof(float), cudaMemcpyDeviceToHost));
        checked(cudaMemcpy(gradients, c->gradients.mem, c->gradients.total_bytes, cudaMemcpyDeviceToHost));
        checked(cudaMemcpy(overflow, c->ea->overflow.data, v4::SETS * sizeof(int), cudaMemcpyDeviceToHost));
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return 1; }
}
// ms per pass (GPU time) on whatever is resident; forward_only skips the backward.
extern "C" double bopl_v4_check_time(void* context, int iterations, int forward_only) {
    auto* c = (V4Check*)context;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    for (int i = 0; i < 3; i++) { if (forward_only) c->forward(); else c->pass(); }
    cudaEventRecord(start, c->stream);
    for (int i = 0; i < iterations; i++) { if (forward_only) c->forward(); else c->pass(); }
    cudaEventRecord(stop, c->stream);
    cudaEventSynchronize(stop);
    float ms = 0.0f; cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms / iterations;
}
// Stage times of one forward+backward pass (ms since the previous mark); names into `names` (64 chars each).
extern "C" int bopl_v4_check_profile(void* context, char* names, double* ms, int max_marks) {
    auto* c = (V4Check*)context;
    c->pass(); cudaStreamSynchronize(c->stream);
    v4::g_prof.on = true; v4::g_prof.n = 0;
    v4::prof_mark("start", c->stream);
    c->pass();
    cudaStreamSynchronize(c->stream);
    v4::g_prof.on = false;
    int n = 0;
    for (int i = 1; i < v4::g_prof.n && n < max_marks; i++, n++) {
        float t = 0; cudaEventElapsedTime(&t, v4::g_prof.ev[i - 1], v4::g_prof.ev[i]);
        ms[n] = t; strncpy(names + 64 * n, v4::g_prof.name[i], 63); names[64 * n + 63] = 0;
    }
    return n;
}
extern "C" void bopl_v4_check_close(void* context) { delete (V4Check*)context; }
