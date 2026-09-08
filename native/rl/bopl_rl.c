#define _GNU_SOURCE
#include "bopl_rl.h"
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "The RL wire format currently requires a little-endian host"
#endif
_Static_assert(sizeof(BoplRlConfig)==176, "config layout");
_Static_assert(sizeof(BoplRlAction)==4, "action layout");
_Static_assert(sizeof(BoplRlHeader)==48, "header layout");
_Static_assert(sizeof(BoplRlPlayer)==72, "player layout");
_Static_assert(sizeof(BoplRlEntity)==128, "entity layout");
_Static_assert(sizeof(BoplRlSlime)==136, "slime layout");
_Static_assert(sizeof(BoplRlTelemetry)==36, "telemetry layout");

enum { MAGIC=0x314c5242, MAX_PAYLOAD=104000000, WORLD_FLAG=0x100, MAX_WORLDS=64 };
typedef struct { uint32_t magic, operation, sequence, length; } Frame;
typedef struct {
    BoplRlState state;
    BoplRlEntity *entities;
    BoplRlSlime *slimes;
    size_t entity_capacity, slime_capacity;
} Snapshot;
/* One game process serving `worlds` handles over one socket. Requests are
 * serialized by send_lock; replies may arrive in any order across worlds and
 * are matched to the waiting handle by sequence (reply_lock + reply_cond,
 * one reader at a time). Every handle has at most one request outstanding. */
typedef struct Process {
    int socket, listener;
    pid_t child;
    uint32_t timeout_ms, sequence, maps, abilities, worlds, refs;
    bool broken, legacy, reading;
    char directory[64], endpoint[108], error[512], source[65], catalog[65];
    pthread_mutex_t send_lock, reply_lock;
    pthread_cond_t reply_cond;
    BoplRl *handles[MAX_WORLDS];
} Process;
struct BoplRl {
    Process *process;
    uint32_t index;
    bool valid, sample_valid, reset_pending;
    /* Outstanding request (under process->reply_lock). */
    bool waiting, delivered;
    uint32_t waiting_sequence, waiting_operation, delivered_operation, delivered_length;
    char error[512];
    unsigned char *buffer;
    size_t capacity;
    Snapshot main, sample;
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
/* A transport failure breaks the whole process: every handle sees BOPL_RL_IO
 * and waiters are woken. Must not be called with reply_lock held. */
static int fail_process(Process *p, BoplRl *w, const char *message) {
    pthread_mutex_lock(&p->reply_lock);
    if (!p->broken) snprintf(p->error, sizeof(p->error), "%s: %s", message, strerror(errno));
    p->broken=true;
    for (uint32_t i=0; i<p->worlds; i++) if (p->handles[i]) p->handles[i]->valid=false;
    if (w) snprintf(w->error, sizeof(w->error), "%s", p->error);
    pthread_cond_broadcast(&p->reply_cond);
    pthread_mutex_unlock(&p->reply_lock);
    return BOPL_RL_IO;
}
static int fail(BoplRl *w, const char *message) { return fail_process(w->process, w, message); }
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
static int transfer(Process *p, BoplRl *w, void *data, size_t size, bool send_data, int64_t deadline) {
    unsigned char *bytes=data;
    /* Try the non-blocking syscall first and poll only when it would block,
     * instead of polling before every chunk. */
    bool ready=true;
    while (size) {
        if (!ready && wait_ready(p->socket, send_data?POLLOUT:POLLIN, deadline)<0) return fail_process(p, w, "RL transport wait");
        ssize_t count=send_data?send(p->socket, bytes, size, MSG_NOSIGNAL):recv(p->socket, bytes, size, 0);
        if (count<0 && (errno==EINTR || errno==EAGAIN)) { ready=false; continue; }
        if (count<=0) { if (!count) errno=ECONNRESET; return fail_process(p, w, "RL transport transfer"); }
        bytes+=count; size-=(size_t)count;
    }
    return BOPL_RL_OK;
}
/* Sends a request frame and its payload as one write (two for oversized payloads). */
static int send_request(Process *p, BoplRl *w, const Frame *frame, const void *data, int64_t deadline) {
    unsigned char packed[sizeof(*frame)+4096];
    if (frame->length<=sizeof(packed)-sizeof(*frame)) {
        memcpy(packed, frame, sizeof(*frame));
        if (frame->length) memcpy(packed+sizeof(*frame), data, frame->length);
        return transfer(p, w, packed, sizeof(*frame)+frame->length, true, deadline);
    }
    int result=transfer(p, w, (void*)frame, sizeof(*frame), true, deadline);
    if (!result && frame->length) result=transfer(p, w, (void*)data, frame->length, true, deadline);
    return result;
}
static int grow(BoplRl *w, size_t needed) {
    if (needed<=w->capacity) return BOPL_RL_OK;
    void *buffer=realloc(w->buffer, needed);
    if (!buffer) return fail(w, "RL response allocation");
    w->buffer=buffer; w->capacity=needed;
    return BOPL_RL_OK;
}
/* Registers the request on the handle and sends it. World-addressed builds get
 * op|0x100 with the world index prefixed; close (4) is process-wide. */
static int submit(BoplRl *w, uint32_t operation, const void *data, uint32_t length, int64_t deadline) {
    Process *p=w->process;
    unsigned char prefixed[4+sizeof(BoplRlConfig)];
    if (length>sizeof(prefixed)-4) { errno=EINVAL; return fail(w, "RL request too large"); }
    uint32_t wire=operation;
    if (!p->legacy && operation!=4) {
        wire|=WORLD_FLAG;
        memcpy(prefixed, &w->index, 4);
        if (length) memcpy(prefixed+4, data, length);
        data=prefixed; length+=4;
    }
    pthread_mutex_lock(&p->send_lock);
    Frame frame={MAGIC, wire, ++p->sequence, length};
    pthread_mutex_lock(&p->reply_lock);
    w->waiting=true; w->delivered=false; w->waiting_sequence=frame.sequence; w->waiting_operation=wire;
    pthread_mutex_unlock(&p->reply_lock);
    int result=send_request(p, w, &frame, data, deadline);
    pthread_mutex_unlock(&p->send_lock);
    return result;
}
/* Reads one reply frame and hands it to the handle that waits for its
 * sequence. Called by the current reader without reply_lock held. */
static int read_reply(Process *p, BoplRl *reader, int64_t deadline) {
    Frame frame;
    int result=transfer(p, reader, &frame, sizeof(frame), false, deadline);
    if (result) return result;
    if (frame.magic!=MAGIC || frame.length>MAX_PAYLOAD) { errno=EPROTO; return fail_process(p, reader, "Invalid RL response frame"); }
    pthread_mutex_lock(&p->reply_lock);
    BoplRl *target=NULL;
    for (uint32_t i=0; i<p->worlds; i++) {
        BoplRl *candidate=p->handles[i];
        if (candidate && candidate->waiting && candidate->waiting_sequence==frame.sequence) { target=candidate; break; }
    }
    bool matches=target && (frame.operation==target->waiting_operation || frame.operation==UINT32_MAX);
    pthread_mutex_unlock(&p->reply_lock);
    if (!matches) { errno=EPROTO; return fail_process(p, reader, "Unexpected RL response frame"); }
    if (frame.length>target->capacity) {
        void *buffer=realloc(target->buffer, frame.length);
        if (!buffer) return fail_process(p, reader, "RL response allocation");
        target->buffer=buffer; target->capacity=frame.length;
    }
    result=transfer(p, reader, target->buffer, frame.length, false, deadline);
    if (result) return result;
    pthread_mutex_lock(&p->reply_lock);
    target->waiting=false; target->delivered=true;
    target->delivered_operation=frame.operation; target->delivered_length=frame.length;
    pthread_cond_broadcast(&p->reply_cond);
    pthread_mutex_unlock(&p->reply_lock);
    return BOPL_RL_OK;
}
/* Waits for the handle's outstanding reply, reading frames for the whole
 * process while no other thread does. */
static int await(BoplRl *w, uint32_t *length, int64_t deadline) {
    Process *p=w->process;
    struct timespec until;
    clock_gettime(CLOCK_MONOTONIC, &until);
    int64_t left=deadline-now_ms(); if (left<0) left=0;
    until.tv_sec+=left/1000; until.tv_nsec+=(left%1000)*1000000;
    if (until.tv_nsec>=1000000000) { until.tv_sec++; until.tv_nsec-=1000000000; }
    pthread_mutex_lock(&p->reply_lock);
    while (!w->delivered) {
        if (p->broken) {
            snprintf(w->error, sizeof(w->error), "%s", p->error); w->valid=false; w->waiting=false;
            pthread_mutex_unlock(&p->reply_lock); return BOPL_RL_IO;
        }
        if (!p->reading) {
            p->reading=true;
            pthread_mutex_unlock(&p->reply_lock);
            int result=read_reply(p, w, deadline);
            pthread_mutex_lock(&p->reply_lock);
            p->reading=false;
            pthread_cond_broadcast(&p->reply_cond);
            if (result) { w->waiting=false; pthread_mutex_unlock(&p->reply_lock); return result; }
            continue;
        }
        int rc=pthread_cond_timedwait(&p->reply_cond, &p->reply_lock, &until);
        if (rc==ETIMEDOUT && !w->delivered && !p->broken) {
            pthread_mutex_unlock(&p->reply_lock);
            errno=ETIMEDOUT; return fail(w, "RL reply wait");
        }
    }
    w->delivered=false;
    uint32_t operation=w->delivered_operation;
    *length=w->delivered_length;
    pthread_mutex_unlock(&p->reply_lock);
    if (operation==UINT32_MAX) {
        snprintf(w->error, sizeof(w->error), "%.*s", (int)*length, (char*)w->buffer);
        w->valid=false;
        return BOPL_RL_GAME_ERROR;
    }
    return BOPL_RL_OK;
}
static int broken(BoplRl *w) {
    if (!w->process->broken) return BOPL_RL_OK;
    if (!w->error[0]) snprintf(w->error, sizeof(w->error), "%s", w->process->error);
    return BOPL_RL_IO;
}
static int command(BoplRl *w, uint32_t operation, const void *data, uint32_t length, uint32_t *response_length) {
    if (broken(w)) return BOPL_RL_IO;
    int64_t deadline=now_ms()+w->process->timeout_ms;
    int result=submit(w, operation, data, length, deadline);
    if (!result) result=await(w, response_length, deadline);
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
    if (header.version!=2 || header.player_count<1 || header.player_count>6 || total>available) {
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

uint32_t bopl_rl_abi_version(void) { return 2; }
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
static BoplRl *new_handle(Process *p, uint32_t index) {
    BoplRl *w=calloc(1, sizeof(*w));
    if (!w) return NULL;
    w->process=p; w->index=index;
    return w;
}
static void free_handle(BoplRl *w) {
    if (!w) return;
    free(w->main.slimes); free(w->main.entities); free(w->sample.slimes); free(w->sample.entities); free(w->buffer); free(w);
}
static void reap(Process *p) {
    if (p->socket>=0) close(p->socket);
    if (p->listener>=0) close(p->listener);
    if (p->child>0) {
        int status; int64_t deadline=now_ms()+2000;
        pid_t result;
        while ((result=waitpid(p->child, &status, WNOHANG))==0 && now_ms()<deadline) usleep(10000);
        if (result==0) {
            kill(-p->child, SIGTERM); deadline=now_ms()+2000;
            while ((result=waitpid(p->child, &status, WNOHANG))==0 && now_ms()<deadline) usleep(10000);
            if (result==0) { kill(-p->child, SIGKILL); while (waitpid(p->child, &status, 0)<0 && errno==EINTR) {} }
        }
    }
    if (p->endpoint[0]) unlink(p->endpoint);
    if (p->directory[0]) rmdir(p->directory);
    pthread_mutex_destroy(&p->send_lock); pthread_mutex_destroy(&p->reply_lock); pthread_cond_destroy(&p->reply_cond);
    free(p);
}
int bopl_rl_open_worlds(const char *executable, const char *replay, const char *log_path, uint32_t timeout_ms,
                        uint32_t worlds, BoplRl **out, char *error, size_t capacity) {
    if (!executable || executable[0]!='/' || !replay || replay[0]!='/' || !log_path || log_path[0]!='/'
        || timeout_ms<1 || timeout_ms>3600000 || worlds<1 || worlds>MAX_WORLDS || !out) {
        error_text(error, capacity, "Absolute executable/replay/log paths, timeout 1..3600000 ms and 1..64 worlds required");
        return BOPL_RL_ARGUMENT;
    }
    for (uint32_t i=0; i<worlds; i++) out[i]=NULL;
    Process *p=calloc(1, sizeof(*p));
    if (!p) { error_text(error, capacity, "RL process allocation failed"); return BOPL_RL_IO; }
    p->socket=p->listener=-1; p->timeout_ms=timeout_ms; p->worlds=worlds;
    pthread_condattr_t clock;
    pthread_condattr_init(&clock); pthread_condattr_setclock(&clock, CLOCK_MONOTONIC);
    pthread_mutex_init(&p->send_lock, NULL); pthread_mutex_init(&p->reply_lock, NULL);
    pthread_cond_init(&p->reply_cond, &clock); pthread_condattr_destroy(&clock);
    BoplRl *first=new_handle(p, 0);
    if (!first) { error_text(error, capacity, "RL handle allocation failed"); reap(p); return BOPL_RL_IO; }
    p->handles[0]=first; p->refs=1;
    snprintf(p->directory, sizeof(p->directory), "/tmp/bopl-rl-XXXXXX");
    if (!mkdtemp(p->directory)) { p->directory[0]=0; fail(first, "RL socket directory"); goto cleanup; }
    snprintf(p->endpoint, sizeof(p->endpoint), "%s/game.sock", p->directory);
    p->listener=socket(AF_UNIX, SOCK_STREAM|SOCK_CLOEXEC|SOCK_NONBLOCK, 0);
    if (p->listener<0) { fail(first, "RL listening socket"); goto cleanup; }
    struct sockaddr_un address={.sun_family=AF_UNIX};
    snprintf(address.sun_path, sizeof(address.sun_path), "%s", p->endpoint);
    if (bind(p->listener, (struct sockaddr*)&address, sizeof(address)) || listen(p->listener, 1)) {
        fail(first, "RL socket bind/listen"); goto cleanup;
    }
    size_t env_count=0;
    while (environ[env_count]) env_count++;
    char **child_env=calloc(env_count+3, sizeof(char*));
    if (!child_env) { fail(first, "RL environment allocation"); goto cleanup; }
    char endpoint_env[128], worlds_env[32];
    snprintf(endpoint_env, sizeof(endpoint_env), "BOPL_RL_SOCKET=%s", p->endpoint);
    snprintf(worlds_env, sizeof(worlds_env), "BOPL_RL_WORLDS=%u", worlds);
    size_t copied=0;
    for (size_t i=0; i<env_count; i++)
        if (strncmp(environ[i], "BOPL_RL_SOCKET=", 15) && strncmp(environ[i], "BOPL_RL_WORLDS=", 15)) child_env[copied++]=environ[i];
    child_env[copied++]=endpoint_env;
    child_env[copied]=worlds_env;
    char *argv[]={(char*)executable, "-batchmode", "-nographics", "-logFile", (char*)log_path, (char*)replay, NULL};
    posix_spawnattr_t attributes;
    posix_spawn_file_actions_t files;
    posix_spawnattr_init(&attributes);
    posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP);
    posix_spawnattr_setpgroup(&attributes, 0);
    posix_spawn_file_actions_init(&files);
    posix_spawn_file_actions_addopen(&files, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addopen(&files, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    int spawned=posix_spawn(&p->child, executable, &files, &attributes, argv, child_env);
    posix_spawn_file_actions_destroy(&files); posix_spawnattr_destroy(&attributes); free(child_env);
    if (spawned) { errno=spawned; fail(first, "RL game spawn"); goto cleanup; }
    int64_t deadline=now_ms()+timeout_ms;
    if (wait_ready(p->listener, POLLIN, deadline)<0) { fail(first, "RL game startup"); goto cleanup; }
    p->socket=accept4(p->listener, NULL, NULL, SOCK_CLOEXEC|SOCK_NONBLOCK);
    if (p->socket<0) { fail(first, "RL game connection"); goto cleanup; }
    close(p->listener); p->listener=-1;
    /* Handshake: sequence 0, op 0, 140 bytes (single-world build) or 144 (+worlds). */
    Frame frame;
    if (transfer(p, first, &frame, sizeof(frame), false, deadline)) goto cleanup;
    if (frame.magic!=MAGIC || frame.operation!=0 || frame.sequence!=0 || (frame.length!=140 && frame.length!=144)) {
        errno=EPROTO; fail(first, "Invalid RL handshake"); goto cleanup;
    }
    if (grow(first, frame.length)) goto cleanup;
    if (transfer(p, first, first->buffer, frame.length, false, deadline)) goto cleanup;
    uint32_t version;
    memcpy(&version, first->buffer, 4);
    if (version!=2) { errno=EPROTO; fail(first, "Unsupported RL protocol"); goto cleanup; }
    memcpy(p->source, first->buffer+4, 64); memcpy(p->catalog, first->buffer+68, 64);
    memcpy(&p->maps, first->buffer+132, 4); memcpy(&p->abilities, first->buffer+136, 4);
    p->legacy=frame.length==140;
    uint32_t allocated=1;
    if (!p->legacy) memcpy(&allocated, first->buffer+140, 4);
    if (allocated!=worlds) { errno=EPROTO; fail(first, p->legacy?"Game build has no world support":"Game allocated a different world count"); goto cleanup; }
    for (uint32_t i=1; i<worlds; i++) {
        p->handles[i]=new_handle(p, i);
        if (!p->handles[i]) { fail(first, "RL handle allocation"); goto cleanup; }
        p->refs++;
    }
    for (uint32_t i=0; i<worlds; i++) out[i]=p->handles[i];
    error_text(error, capacity, ""); return BOPL_RL_OK;
cleanup:
    error_text(error, capacity, first->error);
    for (uint32_t i=0; i<worlds; i++) { free_handle(p->handles[i]); p->handles[i]=NULL; }
    reap(p);
    return BOPL_RL_IO;
}
BoplRl *bopl_rl_open(const char *executable, const char *replay, const char *log_path,
                     uint32_t timeout_ms, char *error, size_t capacity) {
    BoplRl *out[1];
    return bopl_rl_open_worlds(executable, replay, log_path, timeout_ms, 1, out, error, capacity)?NULL:out[0];
}
uint32_t bopl_rl_world_count(const BoplRl *w) { return w?w->process->worlds:0; }
uint32_t bopl_rl_world_index(const BoplRl *w) { return w?w->index:0; }
int bopl_rl_reset_begin(BoplRl *w, const BoplRlConfig *config) {
    if (!w || !config || config->players<1 || config->players>6 || config->slots<1 || config->slots>3
        || config->map>=w->process->maps || config->mode>3 || config->warmup_ticks>3600
        || config->max_episode_ticks<1 || config->max_episode_ticks>2160000 || config->stocks>INT32_MAX) return BOPL_RL_ARGUMENT;
    for (uint32_t i=0; i<config->players; i++) {
        if (config->player[i].team>=6 || config->player[i].keyboard>1 || config->player[i].color>255) return BOPL_RL_ARGUMENT;
        for (uint32_t j=0; j<config->slots; j++)
            if (!config->player[i].abilities[j] || config->player[i].abilities[j]>=w->process->abilities) return BOPL_RL_ARGUMENT;
    }
    if (w->reset_pending) { errno=EBUSY; return fail(w, "RL reset already in progress"); }
    if (broken(w)) return BOPL_RL_IO;
    /* The game answers only after loading the scene and running the warmup
     * ticks; a caller may do other work in between (see bopl_rl_reset_finish). */
    w->sample_valid=false; w->valid=false;
    int result=submit(w, 1, config, sizeof(*config), now_ms()+w->process->timeout_ms);
    if (result) return result;
    w->reset_pending=true;
    return BOPL_RL_OK;
}
int bopl_rl_reset_finish(BoplRl *w) {
    if (!w) return BOPL_RL_ARGUMENT;
    if (broken(w)) return BOPL_RL_IO;
    if (!w->reset_pending) { errno=EINVAL; return fail(w, "No RL reset in progress"); }
    w->reset_pending=false;
    uint32_t length;
    int result=await(w, &length, now_ms()+w->process->timeout_ms);
    if (!result) result=decode_state(w, length);
    return result;
}
int bopl_rl_reset_pending(const BoplRl *w) { return w && w->reset_pending; }
int bopl_rl_reset(BoplRl *w, const BoplRlConfig *config) {
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
const BoplRlState *bopl_rl_state(const BoplRl *w) { return w && w->valid ? &w->main.state : NULL; }
const BoplRlState *bopl_rl_sample(const BoplRl *w) { return w && w->valid && w->sample_valid ? &w->sample.state : NULL; }
int bopl_rl_replay(BoplRl *w, const void **bytes, size_t *length) {
    if (!w || !bytes || !length) return BOPL_RL_ARGUMENT;
    uint32_t received=0;
    int result=command(w, 5, NULL, 0, &received);
    if (result) return result;
    *bytes=w->buffer; *length=received; w->error[0]=0;
    return BOPL_RL_OK;
}
const char *bopl_rl_error(const BoplRl *w) { return w?w->error:"NULL RL handle"; }
const char *bopl_rl_source_hash(const BoplRl *w) { return w?w->process->source:NULL; }
const char *bopl_rl_catalog_hash(const BoplRl *w) { return w?w->process->catalog:NULL; }
uint32_t bopl_rl_map_count(const BoplRl *w) { return w?w->process->maps:0; }
uint32_t bopl_rl_ability_count(const BoplRl *w) { return w?w->process->abilities:0; }
void bopl_rl_close(BoplRl *w) {
    if (!w) return;
    Process *p=w->process;
    /* Drain an unfinished reset so the process's reply stream stays in step. */
    if (w->reset_pending && !p->broken) {
        uint32_t timeout=p->timeout_ms; p->timeout_ms=1000;
        (void)bopl_rl_reset_finish(w);
        p->timeout_ms=timeout;
    }
    pthread_mutex_lock(&p->reply_lock);
    bool last=--p->refs==0;
    if (!last) p->handles[w->index]=NULL;
    pthread_mutex_unlock(&p->reply_lock);
    if (last) {
        if (p->socket>=0 && !p->broken) {
            uint32_t length; uint32_t timeout=p->timeout_ms;
            p->timeout_ms=1000;
            (void)command(w, 4, NULL, 0, &length);
            p->timeout_ms=timeout;
        }
        p->handles[w->index]=NULL;
        reap(p);
    }
    free_handle(w);
}
