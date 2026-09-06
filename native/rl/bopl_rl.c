#define _GNU_SOURCE
#include "bopl_rl.h"
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
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

enum { MAGIC=0x314c5242, MAX_PAYLOAD=104000000 };
typedef struct { uint32_t magic, operation, sequence, length; } Frame;
struct BoplRl {
    int socket, listener;
    pid_t child;
    uint32_t timeout_ms, sequence, maps, abilities;
    bool valid, broken;
    char directory[64], endpoint[108], error[512], source[65], catalog[65];
    unsigned char *buffer;
    size_t capacity, entity_capacity, slime_capacity;
    BoplRlEntity *entities;
    BoplRlSlime *slimes;
    BoplRlState state;
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
    while (size) {
        if (wait_ready(w->socket, send_data?POLLOUT:POLLIN, deadline)<0) return fail(w, "RL transport wait");
        ssize_t count=send_data?send(w->socket, bytes, size, MSG_NOSIGNAL):recv(w->socket, bytes, size, 0);
        if (count<0 && (errno==EINTR || errno==EAGAIN)) continue;
        if (count<=0) { if (!count) errno=ECONNRESET; return fail(w, "RL transport transfer"); }
        bytes+=count; size-=(size_t)count;
    }
    return BOPL_RL_OK;
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
    int result=transfer(w, &frame, sizeof(frame), true, deadline);
    if (!result) result=transfer(w, (void*)data, length, true, deadline);
    if (!result) result=receive_frame(w, operation, frame.sequence, response_length, deadline);
    return result;
}
static int decode_state(BoplRl *w, uint32_t length) {
    BoplRlHeader header;
    if (length<sizeof(header)) { errno=EPROTO; return fail(w, "Short RL state"); }
    memcpy(&header, w->buffer, sizeof(header));
    size_t player_bytes=(size_t)header.player_count*sizeof(BoplRlPlayer);
    size_t entity_bytes=(size_t)header.entity_count*sizeof(BoplRlEntity);
    size_t slime_bytes=(size_t)header.slime_count*sizeof(BoplRlSlime);
    size_t telemetry_bytes=(size_t)header.player_count*sizeof(BoplRlTelemetry);
    if (header.version!=2 || header.player_count<1 || header.player_count>6
        || sizeof(header)+player_bytes+entity_bytes+slime_bytes+telemetry_bytes!=length) {
        errno=EPROTO; return fail(w, "Invalid RL state layout");
    }
    if (header.slime_count>w->slime_capacity) {
        void *slimes=realloc(w->slimes, slime_bytes);
        if (!slimes) return fail(w, "RL slime allocation");
        w->slimes=slimes; w->slime_capacity=header.slime_count;
    }
    if (header.entity_count>w->entity_capacity) {
        void *entities=realloc(w->entities, entity_bytes);
        if (!entities) return fail(w, "RL entity allocation");
        w->entities=entities; w->entity_capacity=header.entity_count;
    }
    w->state.header=header;
    memset(w->state.players, 0, sizeof(w->state.players));
    memcpy(w->state.players, w->buffer+sizeof(header), player_bytes);
    if (entity_bytes) memcpy(w->entities, w->buffer+sizeof(header)+player_bytes, entity_bytes);
    if (slime_bytes) memcpy(w->slimes, w->buffer+sizeof(header)+player_bytes+entity_bytes, slime_bytes);
    memset(w->state.telemetry, 0, sizeof(w->state.telemetry));
    memcpy(w->state.telemetry, w->buffer+sizeof(header)+player_bytes+entity_bytes+slime_bytes, telemetry_bytes);
    w->state.entities=w->entities; w->state.slimes=w->slimes; w->valid=true; w->error[0]=0;
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
BoplRl *bopl_rl_open(const char *executable, const char *replay, const char *log_path,
                     uint32_t timeout_ms, char *error, size_t capacity) {
    if (!executable || executable[0]!='/' || !replay || replay[0]!='/' || !log_path || log_path[0]!='/'
        || timeout_ms<1 || timeout_ms>3600000) {
        error_text(error, capacity, "Absolute executable/replay/log paths and timeout 1..3600000 ms required"); return NULL;
    }
    BoplRl *w=calloc(1, sizeof(*w));
    if (!w) { error_text(error, capacity, "RL handle allocation failed"); return NULL; }
    w->socket=w->listener=-1; w->timeout_ms=timeout_ms;
    snprintf(w->directory, sizeof(w->directory), "/tmp/bopl-rl-XXXXXX");
    if (!mkdtemp(w->directory)) { fail(w, "RL socket directory"); goto cleanup; }
    snprintf(w->endpoint, sizeof(w->endpoint), "%s/game.sock", w->directory);
    w->listener=socket(AF_UNIX, SOCK_STREAM|SOCK_CLOEXEC|SOCK_NONBLOCK, 0);
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
    w->socket=accept4(w->listener, NULL, NULL, SOCK_CLOEXEC|SOCK_NONBLOCK);
    if (w->socket<0) { fail(w, "RL game connection"); goto cleanup; }
    close(w->listener); w->listener=-1;
    uint32_t length=0;
    if (receive_frame(w, 0, 0, &length, deadline)) goto cleanup;
    uint32_t version;
    if (length!=140) { errno=EPROTO; fail(w, "Invalid RL handshake length"); goto cleanup; }
    memcpy(&version, w->buffer, 4);
    if (version!=2) { errno=EPROTO; fail(w, "Unsupported RL protocol"); goto cleanup; }
    memcpy(w->source, w->buffer+4, 64); memcpy(w->catalog, w->buffer+68, 64);
    memcpy(&w->maps, w->buffer+132, 4); memcpy(&w->abilities, w->buffer+136, 4);
    error_text(error, capacity, ""); return w;
cleanup:
    error_text(error, capacity, w->error); bopl_rl_close(w); return NULL;
}
int bopl_rl_reset(BoplRl *w, const BoplRlConfig *config) {
    if (!w || !config || config->players<1 || config->players>6 || config->slots<1 || config->slots>3
        || config->map>=w->maps || config->mode>3 || config->warmup_ticks>3600
        || config->max_episode_ticks<1 || config->max_episode_ticks>2160000 || config->stocks>INT32_MAX) return BOPL_RL_ARGUMENT;
    for (uint32_t i=0; i<config->players; i++) {
        if (config->player[i].team>=6 || config->player[i].keyboard>1 || config->player[i].color>255) return BOPL_RL_ARGUMENT;
        for (uint32_t j=0; j<config->slots; j++)
            if (!config->player[i].abilities[j] || config->player[i].abilities[j]>=w->abilities) return BOPL_RL_ARGUMENT;
    }
    uint32_t length;
    int result=command(w, 1, config, sizeof(*config), &length);
    if (!result) result=decode_state(w, length);
    return result;
}
int bopl_rl_step(BoplRl *w, const BoplRlAction *actions, size_t count, uint32_t repeat) {
    if (!w || !actions || !w->valid || count!=w->state.header.player_count || repeat<1 || repeat>16) return BOPL_RL_ARGUMENT;
    if (w->state.header.flags & BOPL_RL_TERMINAL) return BOPL_RL_ENDED;
    struct { uint32_t repeat; BoplRlAction actions[6]; } request={.repeat=repeat};
    for (size_t i=0; i<count; i++) {
        if (actions[i].reserved || (actions[i].buttons & ~1023)) return BOPL_RL_ARGUMENT;
        request.actions[i]=actions[i];
    }
    uint32_t length;
    int result=command(w, 2, &request, sizeof(request), &length);
    if (!result) result=decode_state(w, length);
    return result;
}
const BoplRlState *bopl_rl_state(const BoplRl *w) { return w && w->valid ? &w->state : NULL; }
int bopl_rl_replay(BoplRl *w, const void **bytes, size_t *length) {
    if (!w || !bytes || !length) return BOPL_RL_ARGUMENT;
    uint32_t received=0;
    int result=command(w, 5, NULL, 0, &received);
    if (result) return result;
    *bytes=w->buffer; *length=received; w->error[0]=0;
    return BOPL_RL_OK;
}
const char *bopl_rl_error(const BoplRl *w) { return w?w->error:"NULL RL handle"; }
const char *bopl_rl_source_hash(const BoplRl *w) { return w?w->source:NULL; }
const char *bopl_rl_catalog_hash(const BoplRl *w) { return w?w->catalog:NULL; }
uint32_t bopl_rl_map_count(const BoplRl *w) { return w?w->maps:0; }
uint32_t bopl_rl_ability_count(const BoplRl *w) { return w?w->abilities:0; }
void bopl_rl_close(BoplRl *w) {
    if (!w) return;
    if (w->socket>=0 && !w->broken) {
        uint32_t length; uint32_t timeout=w->timeout_ms;
        w->timeout_ms=1000; (void)command(w, 4, NULL, 0, &length); w->timeout_ms=timeout;
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
    free(w->slimes); free(w->entities); free(w->buffer); free(w);
}
