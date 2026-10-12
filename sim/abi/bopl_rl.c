#define _GNU_SOURCE
#include "bopl_rl.h"
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <libgen.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#ifdef __APPLE__
/* macOS (the in-game agent shim is built there too): no copy_file_range,
 * accept4 or SOCK_CLOEXEC/SOCK_NONBLOCK flags; set the descriptor flags after
 * the call instead. Byte copies only, which is what non-reflink Linux does. */
#include <sys/uio.h>
static ssize_t copy_file_range(int in, void *a, int out, void *b, size_t n, unsigned f) {
    (void)in; (void)a; (void)out; (void)b; (void)n; (void)f; errno=ENOSYS; return -1;
}
static int socket_cloexec_nonblock(int domain, int type) {
    int fd=socket(domain, type, 0);
    if (fd>=0 && (fcntl(fd, F_SETFD, FD_CLOEXEC) || fcntl(fd, F_SETFL, fcntl(fd, F_GETFL)|O_NONBLOCK))) { close(fd); return -1; }
    return fd;
}
static int accept_cloexec_nonblock(int listener) {
    int fd=accept(listener, NULL, NULL);
    if (fd>=0 && (fcntl(fd, F_SETFD, FD_CLOEXEC) || fcntl(fd, F_SETFL, fcntl(fd, F_GETFL)|O_NONBLOCK))) { close(fd); return -1; }
    return fd;
}
#else
static int socket_cloexec_nonblock(int domain, int type) { return socket(domain, type|SOCK_CLOEXEC|SOCK_NONBLOCK, 0); }
static int accept_cloexec_nonblock(int listener) { return accept4(listener, NULL, NULL, SOCK_CLOEXEC|SOCK_NONBLOCK); }
#endif

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "The RL wire format currently requires a little-endian host"
#endif
_Static_assert(sizeof(BoplRlConfig)==176, "config layout");
_Static_assert(sizeof(BoplRlAction)==4, "action layout");
_Static_assert(sizeof(BoplRlHeader)==64, "header layout");
_Static_assert(sizeof(BoplRlPlayer)==72, "player layout");
_Static_assert(sizeof(BoplRlEntity)==136, "entity layout");
_Static_assert(sizeof(BoplRlSlime)==136, "slime layout");
_Static_assert(sizeof(BoplRlTelemetry)==36, "telemetry layout");

enum { MAGIC=0x314c5242, MAX_PAYLOAD=104000000 };
typedef struct { uint32_t magic, operation, sequence, length; } Frame;
typedef struct {
    BoplRlState state;
    BoplRlEntity *entities;
    BoplRlSlime *slimes;
    size_t entity_capacity, slime_capacity;
} Snapshot;
/* In-process backend: the game is a NativeAOT shared library (sim/Native ->
 * BoplSimNative.so) exporting this same ABI, and every call is forwarded to
 * it. A runtime (one dlopen of the file) serves bopl_rl_world_capacity()
 * worlds - one for library builds without that export - and handles are
 * packed into runtimes up to that (or BOPL_RL_WORLDS_PER_RUNTIME, if lower).
 * Further runtimes dlopen private copies of the file (glibc de-duplicates by
 * device/inode; the copy is a reflink where the filesystem supports it and
 * is unlinked once mapped). Verified by eval/backend_check.c. */
typedef struct InprocLib {
    struct InprocLib *next;
    dev_t device;
    ino_t inode;
    uint32_t capacity, used;
    uint32_t (*abi_version)(void);
    uint32_t (*world_capacity)(void);
    BoplRl *(*open)(const char *, const char *, const char *, uint32_t, char *, size_t);
    int (*reset_begin)(BoplRl *, const BoplRlConfig *);
    int (*reset_finish)(BoplRl *);
    int (*reset_pending)(const BoplRl *);
    int (*step)(BoplRl *, const BoplRlAction *, size_t, uint32_t);
    int (*step_sampled)(BoplRl *, const BoplRlAction *, size_t, uint32_t, uint32_t);
    const BoplRlState *(*state)(const BoplRl *);
    const BoplRlState *(*sample)(const BoplRl *);
    int (*replay)(BoplRl *, const void **, size_t *);
    const char *(*error)(const BoplRl *);
    const char *(*source_hash)(const BoplRl *);
    const char *(*catalog_hash)(const BoplRl *);
    uint32_t (*map_count)(const BoplRl *);
    uint32_t (*ability_count)(const BoplRl *);
    void (*close)(BoplRl *);
    /* Optional (NULL when the library has no lookahead forks). */
    BoplRl *(*fork)(const BoplRl *, BoplRl *);
    /* Optional (NULL when the library records no game events). */
    const BoplRlEvent *(*events)(const BoplRl *, uint32_t *);
    int (*event_support)(const BoplRl *);
} InprocLib;
typedef struct {
    InprocLib *lib;
    BoplRl *inner;
    /* The library's reset_begin does the whole reset (scene build + warmup
     * ticks, ~8 ms warm, 55-90 ms on a runtime's first load of a map) before
     * returning, so begin runs it on a helper thread and finish joins: the
     * spare world of bopl_puffer.h then overlaps resets with play exactly as
     * with a worker process. Measured without this: one reset per ~10-20
     * lockstep steps at 8 ms each was 0.4-0.8 ms per step, the whole gap
     * between the socket and in-process paths. With many worlds per runtime
     * the helper thread runs its world while the caller's threads step the
     * others: the library's per-match statics are thread-local. */
    pthread_t reset_thread;
    bool reset_pending_flag;
    int reset_result;
    BoplRlConfig reset_config;
} Inproc;
static pthread_mutex_t inproc_libs_lock=PTHREAD_MUTEX_INITIALIZER;
static InprocLib *inproc_libs;
struct BoplRl {
    int socket, listener;
    pid_t child;
    uint32_t timeout_ms, sequence, maps, abilities;
    bool valid, broken, sample_valid, reset_pending;
    uint32_t reset_sequence;
    char directory[64], endpoint[108], error[512], source[65], catalog[65];
    unsigned char *buffer;
    size_t capacity;
    Snapshot main, sample;
    Inproc *inproc;
};
extern char **environ;

static int64_t now_ms(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (int64_t)value.tv_sec*1000 + value.tv_nsec/1000000;
}
static void error_text(char *out, size_t capacity, const char *message) {
    if (out && capacity) snprintf(out, capacity, "%s", message);
}
static int fail(BoplRl *w, const char *message) {
    snprintf(w->error, sizeof(w->error), "%s: %s", message, strerror(errno));
    w->broken=true; w->valid=false;
    return BOPL_RL_IO;
}
static int wait_ready(int fd, short events, int64_t deadline) {
    for (;;) {
        int64_t left=deadline-now_ms();
        if (left<=0) { errno=ETIMEDOUT; return -1; }
        struct pollfd item={.fd=fd, .events=events};
        int result=poll(&item, 1, (int)left);
        if (result<0 && errno==EINTR) continue;
        if (result==0) { errno=ETIMEDOUT; return -1; }
        if (result<0) return -1;
        if (item.revents & events) return 0;
        errno=ECONNRESET; return -1;
    }
}
static int transfer(BoplRl *w, void *data, size_t size, bool send_data, int64_t deadline) {
    unsigned char *bytes=data;
    /* Try the non-blocking syscall first and poll only when it would block,
     * instead of polling before every chunk. */
    bool ready=true;
    while (size) {
        if (!ready && wait_ready(w->socket, send_data?POLLOUT:POLLIN, deadline)<0) return fail(w, "RL transport wait");
        ssize_t count=send_data?send(w->socket, bytes, size, MSG_NOSIGNAL):recv(w->socket, bytes, size, 0);
        if (count<0 && (errno==EINTR || errno==EAGAIN)) { ready=false; continue; }
        if (count<=0) { if (!count) errno=ECONNRESET; return fail(w, "RL transport transfer"); }
        bytes+=count; size-=(size_t)count;
    }
    return BOPL_RL_OK;
}
/* Sends a request frame and its payload as one write (two for oversized payloads). */
static int send_request(BoplRl *w, const Frame *frame, const void *data, int64_t deadline) {
    unsigned char packed[sizeof(*frame)+4096];
    if (frame->length<=sizeof(packed)-sizeof(*frame)) {
        memcpy(packed, frame, sizeof(*frame));
        if (frame->length) memcpy(packed+sizeof(*frame), data, frame->length);
        return transfer(w, packed, sizeof(*frame)+frame->length, true, deadline);
    }
    int result=transfer(w, (void*)frame, sizeof(*frame), true, deadline);
    if (!result && frame->length) result=transfer(w, (void*)data, frame->length, true, deadline);
    return result;
}
static int receive_frame(BoplRl *w, uint32_t operation, uint32_t sequence, uint32_t *length, int64_t deadline) {
    Frame frame;
    int result=transfer(w, &frame, sizeof(frame), false, deadline);
    if (result) return result;
    if (frame.magic!=MAGIC || frame.sequence!=sequence || frame.length>MAX_PAYLOAD
        || (frame.operation!=operation && frame.operation!=UINT32_MAX)) {
        errno=EPROTO; return fail(w, "Invalid RL response frame");
    }
    if (frame.length>w->capacity) {
        void *buffer=realloc(w->buffer, frame.length);
        if (!buffer) return fail(w, "RL response allocation");
        w->buffer=buffer; w->capacity=frame.length;
    }
    result=transfer(w, w->buffer, frame.length, false, deadline);
    if (result) return result;
    *length=frame.length;
    if (frame.operation==UINT32_MAX) {
        snprintf(w->error, sizeof(w->error), "%.*s", (int)frame.length, (char*)w->buffer);
        w->valid=false;
        return BOPL_RL_GAME_ERROR;
    }
    return BOPL_RL_OK;
}
static int command(BoplRl *w, uint32_t operation, const void *data, uint32_t length, uint32_t *response_length) {
    if (w->broken) return BOPL_RL_IO;
    int64_t deadline=now_ms()+w->timeout_ms;
    Frame frame={MAGIC, operation, ++w->sequence, length};
    int result=send_request(w, &frame, data, deadline);
    if (!result) result=receive_frame(w, operation, frame.sequence, response_length, deadline);
    return result;
}
/* Decodes one self-delimiting state record from bytes. *consumed receives its
 * length, which may be shorter than available when records are concatenated. */
static int decode_snapshot(BoplRl *w, Snapshot *s, const unsigned char *bytes, uint32_t available, uint32_t *consumed) {
    BoplRlHeader header;
    if (available<sizeof(header)) { errno=EPROTO; return fail(w, "Short RL state"); }
    memcpy(&header, bytes, sizeof(header));
    size_t player_bytes=(size_t)header.player_count*sizeof(BoplRlPlayer);
    size_t entity_bytes=(size_t)header.entity_count*sizeof(BoplRlEntity);
    size_t slime_bytes=(size_t)header.slime_count*sizeof(BoplRlSlime);
    size_t telemetry_bytes=(size_t)header.player_count*sizeof(BoplRlTelemetry);
    size_t total=sizeof(header)+player_bytes+entity_bytes+slime_bytes+telemetry_bytes;
    if (header.version!=3 || header.player_count<1 || header.player_count>6 || total>available) {
        errno=EPROTO; return fail(w, "Invalid RL state layout");
    }
    if (header.slime_count>s->slime_capacity) {
        void *slimes=realloc(s->slimes, slime_bytes);
        if (!slimes) return fail(w, "RL slime allocation");
        s->slimes=slimes; s->slime_capacity=header.slime_count;
    }
    if (header.entity_count>s->entity_capacity) {
        void *entities=realloc(s->entities, entity_bytes);
        if (!entities) return fail(w, "RL entity allocation");
        s->entities=entities; s->entity_capacity=header.entity_count;
    }
    s->state.header=header;
    memset(s->state.players, 0, sizeof(s->state.players));
    memcpy(s->state.players, bytes+sizeof(header), player_bytes);
    if (entity_bytes) memcpy(s->entities, bytes+sizeof(header)+player_bytes, entity_bytes);
    if (slime_bytes) memcpy(s->slimes, bytes+sizeof(header)+player_bytes+entity_bytes, slime_bytes);
    memset(s->state.telemetry, 0, sizeof(s->state.telemetry));
    memcpy(s->state.telemetry, bytes+sizeof(header)+player_bytes+entity_bytes+slime_bytes, telemetry_bytes);
    s->state.entities=s->entities; s->state.slimes=s->slimes;
    *consumed=(uint32_t)total;
    return BOPL_RL_OK;
}
static int decode_state(BoplRl *w, uint32_t length) {
    uint32_t consumed;
    int result=decode_snapshot(w, &w->main, w->buffer, length, &consumed);
    if (result) return result;
    if (consumed!=length) { errno=EPROTO; return fail(w, "Invalid RL state layout"); }
    w->valid=true; w->error[0]=0;
    return BOPL_RL_OK;
}
static int copy_snapshot(BoplRl *w, Snapshot *to, const Snapshot *from) {
    const BoplRlHeader *header=&from->state.header;
    size_t entity_bytes=(size_t)header->entity_count*sizeof(BoplRlEntity);
    size_t slime_bytes=(size_t)header->slime_count*sizeof(BoplRlSlime);
    if (header->slime_count>to->slime_capacity) {
        void *slimes=realloc(to->slimes, slime_bytes);
        if (!slimes) return fail(w, "RL slime allocation");
        to->slimes=slimes; to->slime_capacity=header->slime_count;
    }
    if (header->entity_count>to->entity_capacity) {
        void *entities=realloc(to->entities, entity_bytes);
        if (!entities) return fail(w, "RL entity allocation");
        to->entities=entities; to->entity_capacity=header->entity_count;
    }
    to->state=from->state;
    if (entity_bytes) memcpy(to->entities, from->entities, entity_bytes);
    if (slime_bytes) memcpy(to->slimes, from->slimes, slime_bytes);
    to->state.entities=to->entities; to->state.slimes=to->slimes;
    return BOPL_RL_OK;
}

uint32_t bopl_rl_abi_version(void) { return 3; }
void bopl_rl_default_config(BoplRlConfig *config, uint32_t players) {
    if (!config) return;
    memset(config, 0, sizeof(*config));
    config->seed=37; config->players=players; config->slots=3;
    config->warmup_ticks=240; config->max_episode_ticks=7200;
    for (uint32_t i=0; i<6; i++) {
        config->player[i].color=i%4; config->player[i].team=i;
        config->player[i].abilities[0]=2; // Dash
        config->player[i].abilities[1]=7; // Gust
        config->player[i].abilities[2]=19; // Invisibility
    }
}
int bopl_rl_in_process(const char *executable) {
    size_t length=executable?strlen(executable):0;
    return (length>3 && !strcmp(executable+length-3, ".so"))
        || (length>6 && !strcmp(executable+length-6, ".dylib"));  /* the q32 Rust sim on the Mac shim */
}
/* Copies `from` to `to`; a reflink on Btrfs/XFS, a byte copy elsewhere. */
static int copy_file(const char *from, const char *to) {
    int in=open(from, O_RDONLY|O_CLOEXEC), out=in<0?-1:open(to, O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC, 0700);
    int status=-1;
    if (in>=0 && out>=0) {
        struct stat info;
        if (!fstat(in, &info)) {
            off_t left=info.st_size;
            status=0;
            while (left>0) {
                ssize_t moved=copy_file_range(in, NULL, out, NULL, (size_t)left, 0);
                if (moved<0 && (errno==EXDEV || errno==ENOSYS || errno==EOPNOTSUPP)) {
                    char chunk[1<<16]; ssize_t got;
                    while ((got=read(in, chunk, sizeof(chunk)))>0)
                        if (write(out, chunk, (size_t)got)!=got) { got=-1; break; }
                    if (got<0) status=-1;
                    left=0; break;
                }
                if (moved<=0) { if (!moved) errno=EIO; status=-1; break; }
                left-=moved;
            }
        }
    }
    if (in>=0) close(in);
    if (out>=0) close(out);
    return status;
}
/* dlopen one runtime of the library: the file itself for the first, a
 * private copy for every further one. Returns NULL with w->error set. */
static InprocLib *load_inproc_lib(BoplRl *w, const char *library, bool first,
                                  dev_t device, ino_t inode) {
    InprocLib *lib=calloc(1, sizeof(*lib));
    if (!lib) { fail(w, "RL library table"); return NULL; }
    lib->device=device; lib->inode=inode;
    char parent[4096], path[4096+80], scratch[4096+40];
    void *handle=NULL;
    const char *keep=getenv("BOPL_RL_INPROC_KEEP");
    if (first) handle=dlopen(library, RTLD_NOW|RTLD_LOCAL);
    else {
        /* Private copy next to the library (same filesystem, so a reflink where
         * possible); /tmp when that directory is not writable. */
        snprintf(parent, sizeof(parent), "%s", library);
        snprintf(scratch, sizeof(scratch), "%s/.bopl-rl-XXXXXX", dirname(parent));
        char *directory=mkdtemp(scratch);
        if (!directory) { snprintf(scratch, sizeof(scratch), "/tmp/bopl-rl-XXXXXX"); directory=mkdtemp(scratch); }
        if (!directory) { fail(w, "RL library copy directory"); free(lib); return NULL; }
        snprintf(path, sizeof(path), "%s/BoplSimNative.so", directory);
        if (copy_file(library, path)) { fail(w, "RL library copy"); unlink(path); rmdir(directory); free(lib); return NULL; }
        handle=dlopen(path, RTLD_NOW|RTLD_LOCAL);
        /* The mapping outlives the name; nothing is left on disk per runtime.
         * BOPL_RL_INPROC_KEEP=1 keeps the copies (plus the .dbg symbol file) so
         * perf can symbolize them; the caller cleans the directories up. */
        if (keep && keep[0]=='1') {
            char debug_from[4096+8], debug_to[4096+88];
            snprintf(debug_from, sizeof(debug_from), "%s.dbg", library);
            snprintf(debug_to, sizeof(debug_to), "%s.dbg", path);
            (void)copy_file(debug_from, debug_to);
        } else { unlink(path); rmdir(directory); }
    }
    if (!handle) { snprintf(w->error, sizeof(w->error), "RL library dlopen: %s", dlerror()); free(lib); return NULL; }
#define RESOLVE(field) if (!(lib->field=(__typeof__(lib->field))dlsym(handle, "bopl_rl_" #field))) { \
        snprintf(w->error, sizeof(w->error), "RL library symbol bopl_rl_%s: %s", #field, dlerror()); free(lib); return NULL; }
    RESOLVE(abi_version) RESOLVE(open) RESOLVE(reset_begin) RESOLVE(reset_finish) RESOLVE(reset_pending)
    RESOLVE(step) RESOLVE(step_sampled) RESOLVE(state) RESOLVE(sample) RESOLVE(replay) RESOLVE(error)
    RESOLVE(source_hash) RESOLVE(catalog_hash) RESOLVE(map_count) RESOLVE(ability_count) RESOLVE(close)
#undef RESOLVE
    if (lib->abi_version()!=3) { snprintf(w->error, sizeof(w->error), "RL library ABI version %u", lib->abi_version()); free(lib); return NULL; }
    /* Optional: builds before many-worlds host one world per runtime. */
    lib->world_capacity=(uint32_t (*)(void))dlsym(handle, "bopl_rl_world_capacity");
    lib->fork=(BoplRl *(*)(const BoplRl *, BoplRl *))dlsym(handle, "bopl_rl_fork");
    lib->events=(const BoplRlEvent *(*)(const BoplRl *, uint32_t *))dlsym(handle, "bopl_rl_events");
    lib->event_support=(int (*)(const BoplRl *))dlsym(handle, "bopl_rl_event_support");
    lib->capacity=lib->world_capacity?lib->world_capacity():1;
    if (lib->capacity<1) lib->capacity=1;
    const char *limit=getenv("BOPL_RL_WORLDS_PER_RUNTIME");
    if (limit && limit[0]) {
        unsigned long value=strtoul(limit, NULL, 10);
        if (value>=1 && value<lib->capacity) lib->capacity=(uint32_t)value;
    }
    /* A NativeAOT runtime cannot be unloaded; the mapping stays for the
     * process lifetime, like the library's own statics. No dlclose. */
    return lib;
}
static BoplRl *open_inproc(BoplRl *w, const char *library, const char *replay, const char *log_path,
                           uint32_t timeout_ms, char *error, size_t capacity) {
    Inproc *p=calloc(1, sizeof(*p));
    if (!p) { error_text(error, capacity, "RL handle allocation failed"); free(w); return NULL; }
    w->inproc=p;
    snprintf(w->directory, sizeof(w->directory), "%s", "");
    struct stat identity;
    if (stat(library, &identity)) {
        fail(w, "RL library identity");
        error_text(error, capacity, w->error); free(p); free(w); return NULL;
    }
    pthread_mutex_lock(&inproc_libs_lock);
    InprocLib *lib=inproc_libs;
    while (lib && (lib->device!=identity.st_dev || lib->inode!=identity.st_ino
                   || lib->used>=lib->capacity)) lib=lib->next;
    if (!lib) {
        lib=load_inproc_lib(w, library, inproc_libs==NULL,
                            identity.st_dev, identity.st_ino);
        if (!lib) { pthread_mutex_unlock(&inproc_libs_lock); goto failed; }
        lib->next=inproc_libs; inproc_libs=lib;
    }
    lib->used++; /* reserve the slot; the open itself runs unlocked */
    pthread_mutex_unlock(&inproc_libs_lock);
    /* The library validates the paths like this file does and uses only the
     * log path (its -logFile). Its open() writes the error text itself. A
     * world's open boots its scene (~0.5 s); the library's open is thread-safe
     * (its process-wide boot is behind its own gate), so the trainer's workers
     * open their worlds in parallel. */
    p->inner=lib->open(library, replay, log_path, timeout_ms, error, capacity);
    if (!p->inner) {
        pthread_mutex_lock(&inproc_libs_lock); lib->used--; pthread_mutex_unlock(&inproc_libs_lock);
        free(p); free(w); return NULL;
    }
    p->lib=lib;
    error_text(error, capacity, ""); return w;
failed:
    error_text(error, capacity, w->error); free(p); free(w); return NULL;
}
static void *inproc_reset_worker(void *arg) {
    Inproc *p=arg;
    int result=p->lib->reset_begin(p->inner, &p->reset_config);
    if (!result) result=p->lib->reset_finish(p->inner);
    p->reset_result=result;
    return NULL;
}
static int inproc_reset_begin(BoplRl *w, const BoplRlConfig *config) {
    Inproc *p=w->inproc;
    if (p->reset_pending_flag) { errno=EBUSY; return fail(w, "RL reset already in progress"); }
    p->reset_config=*config;
    int created=pthread_create(&p->reset_thread, NULL, inproc_reset_worker, p);
    if (created) { errno=created; return fail(w, "RL reset thread"); }
    p->reset_pending_flag=true; w->error[0]=0;
    return BOPL_RL_OK;
}
static int inproc_reset_finish(BoplRl *w) {
    Inproc *p=w->inproc;
    if (!p->reset_pending_flag) { errno=EINVAL; return fail(w, "No RL reset in progress"); }
    pthread_join(p->reset_thread, NULL);
    p->reset_pending_flag=false;
    return p->reset_result;
}
#define INPROC(w, call) if ((w) && (w)->inproc) return (w)->inproc->lib->call
BoplRl *bopl_rl_open(const char *executable, const char *replay, const char *log_path,
                     uint32_t timeout_ms, char *error, size_t capacity) {
    if (!executable || executable[0]!='/' || !replay || replay[0]!='/' || !log_path || log_path[0]!='/'
        || timeout_ms<1 || timeout_ms>3600000) {
        error_text(error, capacity, "Absolute executable/replay/log paths and timeout 1..3600000 ms required"); return NULL;
    }
    BoplRl *w=calloc(1, sizeof(*w));
    if (!w) { error_text(error, capacity, "RL handle allocation failed"); return NULL; }
    w->socket=w->listener=-1; w->timeout_ms=timeout_ms;
    if (bopl_rl_in_process(executable)) return open_inproc(w, executable, replay, log_path, timeout_ms, error, capacity);
    snprintf(w->directory, sizeof(w->directory), "/tmp/bopl-rl-XXXXXX");
    if (!mkdtemp(w->directory)) { fail(w, "RL socket directory"); goto cleanup; }
    snprintf(w->endpoint, sizeof(w->endpoint), "%s/game.sock", w->directory);
    w->listener=socket_cloexec_nonblock(AF_UNIX, SOCK_STREAM);
    if (w->listener<0) { fail(w, "RL listening socket"); goto cleanup; }
    struct sockaddr_un address={.sun_family=AF_UNIX};
    snprintf(address.sun_path, sizeof(address.sun_path), "%s", w->endpoint);
    if (bind(w->listener, (struct sockaddr*)&address, sizeof(address)) || listen(w->listener, 1)) {
        fail(w, "RL socket bind/listen"); goto cleanup;
    }
    size_t env_count=0;
    while (environ[env_count]) env_count++;
    char **child_env=calloc(env_count+2, sizeof(char*));
    if (!child_env) { fail(w, "RL environment allocation"); goto cleanup; }
    char endpoint_env[128];
    snprintf(endpoint_env, sizeof(endpoint_env), "BOPL_RL_SOCKET=%s", w->endpoint);
    size_t copied=0;
    for (size_t i=0; i<env_count; i++)
        if (strncmp(environ[i], "BOPL_RL_SOCKET=", 15)) child_env[copied++]=environ[i];
    child_env[copied]=endpoint_env;
    char *argv[]={(char*)executable, "-batchmode", "-nographics", "-logFile", (char*)log_path, (char*)replay, NULL};
    posix_spawnattr_t attributes;
    posix_spawn_file_actions_t files;
    posix_spawnattr_init(&attributes);
    posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&attributes, 0);
    posix_spawn_file_actions_init(&files);
    posix_spawn_file_actions_addopen(&files, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addopen(&files, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    int spawned=posix_spawn(&w->child, executable, &files, &attributes, argv, child_env);
    posix_spawn_file_actions_destroy(&files); posix_spawnattr_destroy(&attributes); free(child_env);
    if (spawned) { errno=spawned; fail(w, "RL game spawn"); goto cleanup; }
    int64_t deadline=now_ms()+timeout_ms;
    if (wait_ready(w->listener, POLLIN, deadline)<0) { fail(w, "RL game startup"); goto cleanup; }
    w->socket=accept_cloexec_nonblock(w->listener);
    if (w->socket<0) { fail(w, "RL game connection"); goto cleanup; }
    close(w->listener); w->listener=-1;
    uint32_t length=0;
    if (receive_frame(w, 0, 0, &length, deadline)) goto cleanup;
    uint32_t version;
    if (length!=140) { errno=EPROTO; fail(w, "Invalid RL handshake length"); goto cleanup; }
    memcpy(&version, w->buffer, 4);
    if (version!=3) { errno=EPROTO; fail(w, "Unsupported RL protocol"); goto cleanup; }
    memcpy(w->source, w->buffer+4, 64); memcpy(w->catalog, w->buffer+68, 64);
    memcpy(&w->maps, w->buffer+132, 4); memcpy(&w->abilities, w->buffer+136, 4);
    error_text(error, capacity, ""); return w;
cleanup:
    error_text(error, capacity, w->error); bopl_rl_close(w); return NULL;
}
int bopl_rl_reset_begin(BoplRl *w, const BoplRlConfig *config) {
    if (w && w->inproc) return config ? inproc_reset_begin(w, config) : BOPL_RL_ARGUMENT;
    if (!w || !config || config->players<1 || config->players>6 || config->slots<1 || config->slots>3
        || config->map>=w->maps || config->mode>3 || config->warmup_ticks>3600
        || config->max_episode_ticks<1 || config->max_episode_ticks>2160000 || config->stocks>INT32_MAX) return BOPL_RL_ARGUMENT;
    for (uint32_t i=0; i<config->players; i++) {
        if (config->player[i].team>=6 || config->player[i].keyboard>1 || config->player[i].color>255) return BOPL_RL_ARGUMENT;
        for (uint32_t j=0; j<config->slots; j++)
            if (!config->player[i].abilities[j] || config->player[i].abilities[j]>=w->abilities) return BOPL_RL_ARGUMENT;
    }
    if (w->reset_pending) { errno=EBUSY; return fail(w, "RL reset already in progress"); }
    if (w->broken) return BOPL_RL_IO;
    /* The game answers only after loading the scene and running the warmup
     * ticks; a caller may do other work in between (see bopl_rl_reset_finish). */
    int64_t deadline=now_ms()+w->timeout_ms;
    Frame frame={MAGIC, 1, ++w->sequence, sizeof(*config)};
    w->sample_valid=false; w->valid=false;
    int result=send_request(w, &frame, config, deadline);
    if (result) return result;
    w->reset_pending=true; w->reset_sequence=frame.sequence;
    return BOPL_RL_OK;
}
int bopl_rl_reset_finish(BoplRl *w) {
    if (w && w->inproc) return inproc_reset_finish(w);
    if (!w) return BOPL_RL_ARGUMENT;
    if (w->broken) return BOPL_RL_IO;
    if (!w->reset_pending) { errno=EINVAL; return fail(w, "No RL reset in progress"); }
    w->reset_pending=false;
    uint32_t length;
    int result=receive_frame(w, 1, w->reset_sequence, &length, now_ms()+w->timeout_ms);
    if (!result) result=decode_state(w, length);
    return result;
}
int bopl_rl_reset_pending(const BoplRl *w) { return w && (w->inproc ? w->inproc->reset_pending_flag : w->reset_pending); }
int bopl_rl_reset(BoplRl *w, const BoplRlConfig *config) {
    if (w && w->inproc) {
        /* Synchronous: run the library's reset on this thread. The begin/finish
         * pair spawns a thread so a caller can overlap a slow (.NET, ~8 ms)
         * reset with other work; spawning and joining one just to wait for it
         * costs ~50 us, more than the Rust backend's whole reset (~25 us). */
        Inproc *p=w->inproc;
        if (!config) return BOPL_RL_ARGUMENT;
        if (p->reset_pending_flag) { errno=EBUSY; return fail(w, "RL reset already in progress"); }
        w->error[0]=0;
        int result=p->lib->reset_begin(p->inner, config);
        return result?result:p->lib->reset_finish(p->inner);
    }
    int result=bopl_rl_reset_begin(w, config);
    return result?result:bopl_rl_reset_finish(w);
}
static int check_actions(const BoplRl *w, const BoplRlAction *actions, size_t count, BoplRlAction *out) {
    if (!w || !actions || !w->valid || count!=w->main.state.header.player_count) return BOPL_RL_ARGUMENT;
    if (w->main.state.header.flags & BOPL_RL_TERMINAL) return BOPL_RL_ENDED;
    for (size_t i=0; i<count; i++) {
        if (actions[i].reserved || (actions[i].buttons & ~1023)) return BOPL_RL_ARGUMENT;
        out[i]=actions[i];
    }
    return BOPL_RL_OK;
}
int bopl_rl_step(BoplRl *w, const BoplRlAction *actions, size_t count, uint32_t repeat) {
    INPROC(w, step(w->inproc->inner, actions, count, repeat));
    struct { uint32_t repeat; BoplRlAction actions[6]; } request={.repeat=repeat};
    int result=check_actions(w, actions, count, request.actions);
    if (result) return result;
    if (repeat<1 || repeat>16) return BOPL_RL_ARGUMENT;
    w->sample_valid=false;
    uint32_t length;
    result=command(w, 2, &request, sizeof(request), &length);
    if (!result) result=decode_state(w, length);
    return result;
}
int bopl_rl_step_sampled(BoplRl *w, const BoplRlAction *actions, size_t count, uint32_t first, uint32_t second) {
    INPROC(w, step_sampled(w->inproc->inner, actions, count, first, second));
    struct { uint32_t first, second; BoplRlAction actions[6]; } request={.first=first, .second=second};
    int result=check_actions(w, actions, count, request.actions);
    if (result) return result;
    if (first<1 || first>16 || second<1 || second>16 || first+second>16) return BOPL_RL_ARGUMENT;
    w->sample_valid=false;
    uint32_t length;
    result=command(w, 6, &request, sizeof(request), &length);
    if (result) return result;
    uint32_t consumed;
    result=decode_snapshot(w, &w->sample, w->buffer, length, &consumed);
    if (result) return result;
    if (consumed==length) {
        if (!(w->sample.state.header.flags & BOPL_RL_TERMINAL)) { errno=EPROTO; return fail(w, "Missing final RL state"); }
        result=copy_snapshot(w, &w->main, &w->sample);
    } else {
        uint32_t rest;
        result=decode_snapshot(w, &w->main, w->buffer+consumed, length-consumed, &rest);
        if (!result && consumed+rest!=length) { errno=EPROTO; return fail(w, "Invalid RL state layout"); }
    }
    if (result) return result;
    w->valid=true; w->sample_valid=true; w->error[0]=0;
    return BOPL_RL_OK;
}
const BoplRlState *bopl_rl_state(const BoplRl *w) { INPROC(w, state(w->inproc->inner)); return w && w->valid ? &w->main.state : NULL; }
const BoplRlState *bopl_rl_sample(const BoplRl *w) { INPROC(w, sample(w->inproc->inner)); return w && w->valid && w->sample_valid ? &w->sample.state : NULL; }
int bopl_rl_replay(BoplRl *w, const void **bytes, size_t *length) {
    INPROC(w, replay(w->inproc->inner, bytes, length));
    if (!w || !bytes || !length) return BOPL_RL_ARGUMENT;
    uint32_t received=0;
    int result=command(w, 5, NULL, 0, &received);
    if (result) return result;
    *bytes=w->buffer; *length=received; w->error[0]=0;
    return BOPL_RL_OK;
}
BoplRl *bopl_rl_fork(const BoplRl *w, BoplRl *reuse) {
    if (!w || !w->inproc || !w->inproc->lib->fork || w->inproc->reset_pending_flag) return NULL;
    if (reuse && (!reuse->inproc || reuse->inproc->lib!=w->inproc->lib || reuse==w)) return NULL;
    BoplRl *f=reuse;
    if (!f) {
        f=calloc(1, sizeof(*f));
        Inproc *p=f?calloc(1, sizeof(*p)):NULL;
        if (!p) { free(f); return NULL; }
        f->socket=f->listener=-1; f->timeout_ms=w->timeout_ms; f->inproc=p; p->lib=w->inproc->lib;
    }
    BoplRl *inner=w->inproc->lib->fork(w->inproc->inner, f->inproc->inner);
    if (!inner) { if (!reuse) { free(f->inproc); free(f); } return NULL; }
    f->inproc->inner=inner;
    return f;
}
/* Game events: only an in-process library exporting them has any; the
 * socket/.NET game reports none and no support. */
const BoplRlEvent *bopl_rl_events(const BoplRl *w, uint32_t *count) {
    if (count) *count=0;
    if (!w || !w->inproc || !w->inproc->lib->events || w->inproc->reset_pending_flag) return NULL;
    uint32_t n=0;
    const BoplRlEvent *events=w->inproc->lib->events(w->inproc->inner, &n);
    if (!events || !n) return NULL;
    if (count) *count=n;
    return events;
}
int bopl_rl_event_support(const BoplRl *w) {
    if (!w || !w->inproc || !w->inproc->lib->events || !w->inproc->lib->event_support) return 0;
    return w->inproc->lib->event_support(w->inproc->inner) ? 1 : 0;
}
const char *bopl_rl_error(const BoplRl *w) {
    if (!w) return "NULL RL handle";
    if (w->inproc) return w->error[0] ? w->error : w->inproc->lib->error(w->inproc->inner);
    return w->error;
}
const char *bopl_rl_source_hash(const BoplRl *w) { INPROC(w, source_hash(w->inproc->inner)); return w?w->source:NULL; }
const char *bopl_rl_catalog_hash(const BoplRl *w) { INPROC(w, catalog_hash(w->inproc->inner)); return w?w->catalog:NULL; }
uint32_t bopl_rl_map_count(const BoplRl *w) { INPROC(w, map_count(w->inproc->inner)); return w?w->maps:0; }
uint32_t bopl_rl_ability_count(const BoplRl *w) { INPROC(w, ability_count(w->inproc->inner)); return w?w->abilities:0; }
void bopl_rl_close(BoplRl *w) {
    if (!w) return;
    if (w->inproc) {
        if (w->inproc->reset_pending_flag) (void)inproc_reset_finish(w);
        w->inproc->lib->close(w->inproc->inner); free(w->inproc); free(w); return;
    }
    if (w->socket>=0 && !w->broken) {
        uint32_t length; uint32_t timeout=w->timeout_ms;
        w->timeout_ms=1000;
        /* Drain an unfinished reset so the close reply is the next frame. */
        if (w->reset_pending) (void)bopl_rl_reset_finish(w);
        if (!w->broken) (void)command(w, 4, NULL, 0, &length);
        w->timeout_ms=timeout;
    }
    if (w->socket>=0) close(w->socket);
    if (w->listener>=0) close(w->listener);
    if (w->child>0) {
        int status; int64_t deadline=now_ms()+2000;
        pid_t result;
        while ((result=waitpid(w->child, &status, WNOHANG))==0 && now_ms()<deadline) usleep(10000);
        if (result==0) {
            kill(-w->child, SIGTERM); deadline=now_ms()+2000;
            while ((result=waitpid(w->child, &status, WNOHANG))==0 && now_ms()<deadline) usleep(10000);
            if (result==0) { kill(-w->child, SIGKILL); while (waitpid(w->child, &status, 0)<0 && errno==EINTR) {} }
        }
    }
    if (w->endpoint[0]) unlink(w->endpoint);
    if (w->directory[0]) rmdir(w->directory);
    free(w->main.slimes); free(w->main.entities); free(w->sample.slimes); free(w->sample.entities); free(w->buffer); free(w);
}
