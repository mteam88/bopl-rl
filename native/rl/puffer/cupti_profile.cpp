// Kernel-level GPU profile of the trainer without nsys: a CUPTI injection
// library that aggregates every kernel / memcpy / memset by (stream, name).
//
// Build (RL venv, CUDA 13 from the nvidia wheels):
//   V=native/artifacts/rl/venv/lib/python3.11/site-packages/nvidia/cu13
//   g++ -O2 -shared -fPIC -I$V/include native/rl/puffer/cupti_profile.cpp \
//       -L$V/lib -l:libcupti.so.13 -Wl,-rpath,$V/lib -o /tmp/libcuptiprof.so
// Run:
//   CUDA_INJECTION64_PATH=/tmp/libcuptiprof.so BOPL_CUPTI_OUT=/tmp/prof.txt \
//       python native/rl/puffer/train.py ...
// Output: one table per stream (busy = union of kernel intervals on that
// stream; sum = sum of kernel durations), rows sorted by total time. The
// trainer's train phase runs on the legacy default stream; the rollout
// forwards run on the per-buffer streams and the bank side streams.
#include <cupti.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <algorithm>

// BOPL_CUPTI_TRACE=<path> additionally keeps the last BOPL_CUPTI_TRACE_N
// (default 400000) records and writes them at exit as
// "start_ns end_ns stream kind name" lines (kind K kernel, H HtoD, D DtoH,
// C DtoD, S memset), for a per-step timeline (native/rl/puffer/cupti_timeline.py).
namespace {
struct Agg { uint64_t count = 0, total = 0, mn = ~0ull, mx = 0; };
struct Rec { uint64_t start, end; uint32_t stream; char kind; const char* name; };
std::vector<Rec> g_trace;
size_t g_trace_cap = 0, g_trace_next = 0;
bool g_trace_wrapped = false;
std::map<std::string, const char*> g_names;  // interned kernel names

void trace(uint32_t stream, char kind, const char* name, uint64_t start, uint64_t end) {
    if (g_trace_cap == 0) return;
    auto it = g_names.find(name);
    if (it == g_names.end()) it = g_names.emplace(name, strdup(name)).first;
    Rec r{start, end, stream, kind, it->second};
    if (g_trace.size() < g_trace_cap) { g_trace.push_back(r); return; }
    g_trace[g_trace_next] = r;
    if (++g_trace_next == g_trace_cap) { g_trace_next = 0; g_trace_wrapped = true; }
}
struct StreamAgg {
    std::map<std::string, Agg> kernels;
    uint64_t busy = 0, last_end = 0, first_start = ~0ull, last = 0, records = 0;
};
std::mutex g_mutex;
std::map<uint32_t, StreamAgg> g_streams;
bool g_flushed = false;

void add(uint32_t stream, const char* name, uint64_t start, uint64_t end) {
    StreamAgg& s = g_streams[stream];
    Agg& a = s.kernels[name];
    uint64_t d = end - start;
    a.count++; a.total += d; a.mn = std::min(a.mn, d); a.mx = std::max(a.mx, d);
    if (start < s.first_start) s.first_start = start;
    if (end > s.last) s.last = end;
    // Approximate interval union (records on one stream arrive nearly in order).
    uint64_t from = std::max(start, s.last_end);
    if (end > from) s.busy += end - from;
    if (end > s.last_end) s.last_end = end;
    s.records++;
}

void CUPTIAPI buffer_requested(uint8_t** buffer, size_t* size, size_t* max_records) {
    *size = 8 << 20;
    *buffer = (uint8_t*)aligned_alloc(8, *size);
    *max_records = 0;
}

const char* memcpy_kind(uint8_t kind) {
    switch (kind) {
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: return "memcpy HtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH: return "memcpy DtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD: return "memcpy DtoD";
        default: return "memcpy other";
    }
}

void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t valid) {
    std::lock_guard<std::mutex> lock(g_mutex);
    CUpti_Activity* record = nullptr;
    while (cuptiActivityGetNextRecord(buffer, valid, &record) == CUPTI_SUCCESS) {
        switch (record->kind) {
            case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
            case CUPTI_ACTIVITY_KIND_KERNEL: {
                auto* k = (CUpti_ActivityKernel10*)record;
                add(k->streamId, k->name, k->start, k->end);
                trace(k->streamId, 'K', k->name, k->start, k->end);
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMCPY: {
                auto* m = (CUpti_ActivityMemcpy6*)record;
                add(m->streamId, memcpy_kind(m->copyKind), m->start, m->end);
                char kind = m->copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_HTOD ? 'H'
                          : m->copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_DTOH ? 'D'
                          : m->copyKind == CUPTI_ACTIVITY_MEMCPY_KIND_DTOD ? 'C' : 'M';
                char buf[64];
                snprintf(buf, sizeof buf, "memcpy_%llu", (unsigned long long)m->bytes);
                trace(m->streamId, kind, buf, m->start, m->end);
                break;
            }
            case CUPTI_ACTIVITY_KIND_MEMSET: {
                auto* m = (CUpti_ActivityMemset4*)record;
                add(m->streamId, "memset", m->start, m->end);
                trace(m->streamId, 'S', "memset", m->start, m->end);
                break;
            }
            default: break;
        }
    }
    free(buffer);
}

std::string demangle_short(const std::string& name) {
    // Keep the mangled name but trim after 90 chars for table width.
    return name.size() > 90 ? name.substr(0, 87) + "..." : name;
}

void flush() {
    if (g_flushed) return;
    g_flushed = true;
    cuptiActivityFlushAll(1);
    std::lock_guard<std::mutex> lock(g_mutex);
    const char* path = getenv("BOPL_CUPTI_OUT");
    FILE* out = path ? fopen(path, "w") : stderr;
    if (!out) out = stderr;
    std::vector<std::pair<uint32_t, StreamAgg*>> streams;
    for (auto& kv : g_streams) streams.push_back({kv.first, &kv.second});
    std::sort(streams.begin(), streams.end(), [](auto& a, auto& b) { return a.second->busy > b.second->busy; });
    for (auto& [id, s] : streams) {
        uint64_t sum = 0;
        for (auto& kv : s->kernels) sum += kv.second.total;
        fprintf(out, "\n== stream %u: %llu records, %zu distinct, busy %.3f ms, sum %.3f ms, span %.3f ms\n",
                id, (unsigned long long)s->records, s->kernels.size(), s->busy / 1e6, sum / 1e6,
                (s->last - s->first_start) / 1e6);
        std::vector<std::pair<std::string, Agg>> rows(s->kernels.begin(), s->kernels.end());
        std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.second.total > b.second.total; });
        fprintf(out, "%12s %9s %10s %9s %9s  %s\n", "total_ms", "count", "avg_us", "min_us", "max_us", "name");
        for (auto& [name, a] : rows) {
            fprintf(out, "%12.3f %9llu %10.2f %9.2f %9.2f  %s\n", a.total / 1e6, (unsigned long long)a.count,
                    a.total / 1e3 / a.count, a.mn / 1e3, a.mx / 1e3, demangle_short(name).c_str());
        }
    }
    if (out != stderr) fclose(out);
    if (const char* tpath = getenv("BOPL_CUPTI_TRACE")) {
        FILE* t = fopen(tpath, "w");
        if (t) {
            size_t n = g_trace.size();
            for (size_t i = 0; i < n; i++) {
                const Rec& r = g_trace[g_trace_wrapped ? (g_trace_next + i) % n : i];
                fprintf(t, "%llu %llu %u %c %s\n", (unsigned long long)r.start, (unsigned long long)r.end,
                        r.stream, r.kind, r.name);
            }
            fclose(t);
        }
    }
}
}  // namespace

extern "C" int InitializeInjection(void) {
    if (getenv("BOPL_CUPTI_TRACE")) {
        const char* n = getenv("BOPL_CUPTI_TRACE_N");
        g_trace_cap = n ? strtoull(n, nullptr, 10) : 400000;
        g_trace.reserve(g_trace_cap);
    }
    cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
    atexit(flush);
    return 1;
}
