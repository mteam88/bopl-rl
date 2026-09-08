// Standalone check of the terminal-masked MinGRU training scan (patched
// models.cu): runs policy_forward_train / policy_backward for one linear
// encoder + MinGRU + decoder policy and exposes outputs, final states, weight
// gradients and initial-state gradients. No game or trainer is linked.
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "models.cu"

static void checked(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

struct RecurrentCheck {
    Policy policy{};
    PolicyWeights weights{};
    PolicyActivations acts{};
    Allocator params{}, activations{}, gradients{};
    PrecisionTensor x{}, state{}, reset{};
    FloatTensor grad_logits{}, grad_value{};
    PrecisionTensor dec_out{};
    int obs = 0, hidden = 0, layers = 0, T = 0, B = 0, logits = 0;
    cudaStream_t stream = nullptr;
    ~RecurrentCheck() {
        if (stream) cudaStreamSynchronize(stream);
        alloc_free(&params); alloc_free(&activations); alloc_free(&gradients);
        if (stream) cudaStreamDestroy(stream);
    }
};

static thread_local std::string last_error;
extern "C" const char* bopl_recurrent_check_error() { return last_error.c_str(); }

extern "C" void* bopl_recurrent_check_create(int obs, int hidden, int layers, int T, int B, int logits) {
    auto* c = new RecurrentCheck;
    try {
        if (obs < 1 || hidden < 8 || layers < 1 || T < 1 || B < 1 || logits < 1)
            throw std::runtime_error("Invalid shape");
        c->obs = obs; c->hidden = hidden; c->layers = layers; c->T = T; c->B = B; c->logits = logits;
        Encoder encoder = {
            .forward = encoder_forward, .backward = encoder_backward,
            .init_weights = encoder_init_weights, .reg_params = encoder_reg_params,
            .reg_train = encoder_reg_train, .reg_rollout = encoder_reg_rollout,
            .create_weights = encoder_create_weights, .free_weights = encoder_free_weights,
            .free_activations = encoder_free_activations,
            .in_dim = obs, .out_dim = hidden, .activation_size = sizeof(EncoderActivations),
        };
        Decoder decoder = {
            .forward = decoder_forward, .backward = decoder_backward,
            .init_weights = decoder_init_weights, .reg_params = decoder_reg_params,
            .reg_train = decoder_reg_train, .reg_rollout = decoder_reg_rollout,
            .create_weights = decoder_create_weights, .free_weights = decoder_free_weights,
            .free_activations = decoder_free_activations,
            .hidden_dim = hidden, .output_dim = logits, .continuous = false,
            .activation_size = (int)sizeof(DecoderActivations),
        };
        Network network = {
            .forward = mingru_forward, .forward_train = mingru_forward_train,
            .backward = mingru_backward, .init_weights = mingru_init_weights,
            .reg_params = mingru_reg_params, .reg_train = mingru_reg_train,
            .reg_rollout = mingru_reg_rollout, .create_weights = mingru_create_weights,
            .free_weights = mingru_free_weights, .free_activations = mingru_free_activations,
            .hidden = hidden, .num_layers = layers, .horizon = T,
        };
        c->policy = Policy{.encoder = encoder, .decoder = decoder, .network = network,
            .input_dim = obs, .hidden_dim = hidden, .output_dim = logits, .num_atns = logits};
        c->weights = policy_weights_create(&c->policy, &c->params);
        c->acts = policy_reg_train(&c->policy, c->weights, &c->activations, &c->gradients, B * T);
        c->x = {.shape = {B, T, obs}};
        c->state = {.shape = {layers, B, hidden}};
        c->reset = {.shape = {B, T}};
        c->grad_logits = {.shape = {B, T, logits}};
        c->grad_value = {.shape = {B, T, 1}};
        alloc_register(&c->activations, &c->x);
        alloc_register(&c->activations, &c->state);
        alloc_register(&c->activations, &c->reset);
        alloc_register(&c->activations, &c->grad_logits);
        alloc_register(&c->activations, &c->grad_value);
        checked(alloc_create(&c->params));
        checked(alloc_create(&c->activations));
        checked(alloc_create(&c->gradients));
        if (c->params.total_bytes != c->gradients.total_bytes)
            throw std::runtime_error("Weight and gradient layout differ");
        checked(cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking));
        return c;
    } catch (const std::exception& e) { last_error = e.what(); delete c; return nullptr; }
}

extern "C" long bopl_recurrent_check_weight_count(void* context) {
    return ((RecurrentCheck*)context)->params.total_bytes / sizeof(float);
}

// x: (B, T, obs); state: (layers, B, hidden); reset: (B, T);
// grad_logits: (B, T, logits); grad_value: (B, T).
// out: (B, T, logits + 1); final_state: (layers, B, hidden);
// weight_grads: weight_count; state_grads: (layers, B, hidden).
extern "C" int bopl_recurrent_check_run(void* context, const float* weights, const float* x,
        const float* state, const float* reset, const float* grad_logits, const float* grad_value,
        float* out, float* final_state, float* weight_grads, float* state_grads) {
    auto* c = (RecurrentCheck*)context;
    try {
        auto up = [&](void* dst, const void* src, size_t bytes) {
            checked(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, c->stream));
        };
        up(c->params.mem, weights, c->params.total_bytes);
        up(c->x.data, x, numel(c->x.shape) * sizeof(float));
        up(c->state.data, state, numel(c->state.shape) * sizeof(float));
        up(c->reset.data, reset, numel(c->reset.shape) * sizeof(float));
        up(c->grad_logits.data, grad_logits, numel(c->grad_logits.shape) * sizeof(float));
        up(c->grad_value.data, grad_value, numel(c->grad_value.shape) * sizeof(float));
        checked(cudaMemsetAsync(c->gradients.mem, 0, c->gradients.total_bytes, c->stream));
        mingru_set_train_reset(c->reset.data);
        c->dec_out = policy_forward_train(&c->policy, c->weights, c->acts, c->x, c->state, c->stream);
        FloatTensor no_logstd{};
        policy_backward(&c->policy, c->weights, c->acts, c->grad_logits, no_logstd, c->grad_value, c->stream);
        mingru_set_train_reset(nullptr);
        checked(cudaGetLastError());
        checked(cudaStreamSynchronize(c->stream));
        checked(cudaMemcpy(out, c->dec_out.data, numel(c->dec_out.shape) * sizeof(float), cudaMemcpyDeviceToHost));
        checked(cudaMemcpy(weight_grads, c->gradients.mem, c->gradients.total_bytes, cudaMemcpyDeviceToHost));
        auto* a = (MinGRUActivations*)c->acts.network;
        long per_layer = (long)c->B * c->hidden;
        for (int i = 0; i < c->layers; i++) {
            checked(cudaMemcpy(final_state + i * per_layer, a->scan_bufs[i].next_state.data,
                per_layer * sizeof(float), cudaMemcpyDeviceToHost));
            checked(cudaMemcpy(state_grads + i * per_layer, a->scan_bufs[i].grad_state.data,
                per_layer * sizeof(float), cudaMemcpyDeviceToHost));
        }
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return 1; }
}

extern "C" void bopl_recurrent_check_close(void* context) { delete (RecurrentCheck*)context; }
