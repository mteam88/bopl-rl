/* Agent shim; see bopl_agent.h. The decision loop is bopl_puffer.h's own
 * code (bopl_reset_observations, bopl_step_begin_from, bopl_step_sample,
 * bopl_step_finish_from) run on caller-supplied states, the way
 * eval_binding.c's research_step_begin/tick/finish run it on a stepped world.
 * This file only adds: the sidecar reader, the policy forward pass via
 * policy_eval.c, the sampler, and the wire-format decoder. */
#include "bopl_agent.h"
#include "bopl_puffer.h"
#include "bopl_aim.h"
#include <errno.h>
#include <stdarg.h>

/* policy_eval.c (no header of its own) */
void *bopl_policy_eval_create_aim(const float *data, size_t count, int batch, int observations, int hidden, int layers,
                                  int use_set, int aim_fourier, int aim_ar);
int bopl_policy_eval_width(void *handle);
void bopl_policy_eval_step_raw(void *handle, float *observations, const float *state, float *rows, float *next_state);
void bopl_policy_eval_game_logits(void *handle, const float *rows, const int *buttons, float *out);
void bopl_policy_eval_close(void *handle);

#define BOPL_AGENT_ABI 1u
/* The training loadout when a spec carries none (bots): Dash, Bow, Platform,
 * the research_open / binding.c defaults (bopl.ini ability1..3). */
static const int BOPL_AGENT_DEFAULT_ABILITIES[3]={2, 4, 14};
enum { BOPL_AGENT_DEFAULT_WARMUP=240, BOPL_AGENT_DEFAULT_LIMIT=7200, BOPL_AGENT_DEFAULT_DELAY=9, BOPL_AGENT_DEFAULT_REPEAT=4 };

struct BoplAgent {
    BoplPuffer env;          /* first: the decision-loop state */
    BoplStepPhase phase;
    int open, tick, sampled, primed, ended, armed;
    uint32_t expected;
    int players, player;
    /* policy */
    void *policy; int hidden, layers, is_policy, aim_fourier, aim_ar;
    float *rstate, *rnext, *decoder, *row;  /* decoder: BOPL_LOGITS game logits + value; row: the raw decoder row */
    uint64_t rng;
    int decision[3], decisions;
    /* scripted */
    int kind, level;
    /* configuration */
    int ability[3], warmup, repeat, delay, limit;
    char id[96];
    /* wire decode */
    BoplRlState decoded; BoplRlEntity *entities; BoplRlSlime *slimes; uint32_t ecap, scap;
};

uint32_t bopl_agent_abi_version(void) { return BOPL_AGENT_ABI; }
int bopl_agent_obs_size(void) { return BOPL_OBS_SIZE; }
int bopl_agent_logits_size(void) { return BOPL_LOGITS; }

static void bopl_agent_error(char *error, size_t capacity, const char *format, ...) {
    if (!error || !capacity) return;
    va_list args; va_start(args, format);
    vsnprintf(error, capacity, format, args);
    va_end(args);
}

/* ---- SHA-256 (FIPS 180-4), to verify and name the weights ---- */
static const uint32_t bopl_sha_k[64]={
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
#define ROTR(x,n) (((x)>>(n))|((x)<<(32-(n))))
static void bopl_sha_block(uint32_t h[8], const unsigned char *p) {
    uint32_t w[64];
    for (int i=0; i<16; i++) w[i]=(uint32_t)p[4*i]<<24|(uint32_t)p[4*i+1]<<16|(uint32_t)p[4*i+2]<<8|p[4*i+3];
    for (int i=16; i<64; i++) {
        uint32_t s0=ROTR(w[i-15],7)^ROTR(w[i-15],18)^(w[i-15]>>3), s1=ROTR(w[i-2],17)^ROTR(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for (int i=0; i<64; i++) {
        uint32_t t1=hh+(ROTR(e,6)^ROTR(e,11)^ROTR(e,25))+((e&f)^(~e&g))+bopl_sha_k[i]+w[i];
        uint32_t t2=(ROTR(a,2)^ROTR(a,13)^ROTR(a,22))+((a&b)^(a&c)^(b&c));
        hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
}
static void bopl_sha256_hex(const unsigned char *data, size_t length, char out[65]) {
    uint32_t h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    size_t i=0;
    for (; i+64<=length; i+=64) bopl_sha_block(h, data+i);
    unsigned char tail[128]; size_t rest=length-i; memcpy(tail, data+i, rest); tail[rest++]=0x80;
    size_t padded=rest<=56?64:128; memset(tail+rest, 0, padded-rest);
    uint64_t bits=(uint64_t)length*8;
    for (int b=0; b<8; b++) tail[padded-1-b]=(unsigned char)(bits>>(8*b));
    for (size_t j=0; j<padded; j+=64) bopl_sha_block(h, tail+j);
    for (int j=0; j<8; j++) snprintf(out+8*j, 9, "%08x", h[j]);
}

/* ---- sidecar reader: the few keys the loop depends on ----
 * A key is located as the quoted name followed by ':' (so "episode_ticks"
 * does not match inside "max_episode_ticks"); the first occurrence wins,
 * which is the top-level one for the keys that also appear nested. */
static const char *bopl_json_find(const char *json, const char *key) {
    size_t n=strlen(key);
    for (const char *p=json; (p=strstr(p, key)); p+=n) {
        if (p==json || p[-1]!='"' || p[n]!='"') continue;
        const char *q=p+n+1; while (*q==' '||*q=='\t'||*q=='\n'||*q=='\r') q++;
        if (*q!=':') continue;
        q++; while (*q==' '||*q=='\t'||*q=='\n'||*q=='\r') q++;
        return q;
    }
    return NULL;
}
static int bopl_json_int(const char *json, const char *key, long *out) {
    const char *v=bopl_json_find(json, key); if (!v) return 0;
    char *end; errno=0; long value=strtol(v, &end, 10);
    if (end==v || errno) return 0;
    if (*end=='.') { /* 7200.0 */ strtod(v, &end); }
    *out=value; return 1;
}
static int bopl_json_string(const char *json, const char *key, char *out, size_t capacity) {
    const char *v=bopl_json_find(json, key); if (!v || *v!='"') return 0;
    v++; const char *end=strchr(v, '"'); if (!end || (size_t)(end-v)>=capacity) return 0;
    memcpy(out, v, (size_t)(end-v)); out[end-v]=0; return 1;
}
static int bopl_json_ints(const char *json, const char *key, long *out, int count) {
    const char *v=bopl_json_find(json, key); if (!v || *v!='[') return 0;
    v++;
    for (int i=0; i<count; i++) {
        char *end; out[i]=strtol(v, &end, 10); if (end==v) return 0;
        v=end; while (*v==' '||*v==',') v++;
    }
    return 1;
}
static unsigned char *bopl_read_file(const char *path, size_t *length) {
    FILE *f=fopen(path, "rb"); if (!f) return NULL;
    if (fseek(f, 0, SEEK_END)) { fclose(f); return NULL; }
    long size=ftell(f); if (size<0) { fclose(f); return NULL; }
    rewind(f);
    unsigned char *data=malloc((size_t)size+1); if (!data) { fclose(f); return NULL; }
    if (fread(data, 1, (size_t)size, f)!=(size_t)size) { free(data); fclose(f); return NULL; }
    fclose(f); data[size]=0; *length=(size_t)size; return data;
}

/* ---- construction ---- */
static int bopl_agent_env_alloc(BoplAgent *a) {
    BoplPuffer *env=&a->env;
    int n=a->players;
    env->players=n; env->num_agents=n; env->repeat=a->repeat; env->observation_delay=a->delay;
    env->no_autoreset=1;
    env->reward_kill=0.1f; env->reward_timeout=0;
    bopl_rl_default_config(&env->config, (uint32_t)n);
    env->config.stocks=0; env->config.max_episode_ticks=(uint32_t)a->limit; env->config.warmup_ticks=(uint32_t)a->warmup;
    for (int i=0; i<n; i++) for (int s=0; s<3; s++) env->config.player[i].abilities[s]=(uint32_t)a->ability[s];
    env->observations=calloc((size_t)n*BOPL_OBS_SIZE, sizeof(float));
    env->actions=calloc((size_t)n*BOPL_NUM_ATNS, sizeof(float));
    env->rewards=calloc((size_t)n, sizeof(float)); env->terminals=calloc((size_t)n, sizeof(float));
    env->action_mask=calloc((size_t)n*BOPL_LOGITS, 1);
    if (!env->observations || !env->actions || !env->rewards || !env->terminals || !env->action_mask) return -1;
    for (int s=0; s<n; s++) {
        env->obs_ptr[s]=env->observations+(size_t)s*BOPL_OBS_SIZE; env->action_ptr[s]=env->actions+BOPL_NUM_ATNS*s;
        env->reward_ptr[s]=env->rewards+s; env->terminal_ptr[s]=env->terminals+s;
        env->mask_ptr[s]=env->action_mask+(size_t)s*BOPL_LOGITS;
    }
    return 0;
}

static int bopl_agent_open_policy(BoplAgent *a, const char *path, char *error, size_t capacity) {
    size_t length=0, json_length=0;
    unsigned char *data=bopl_read_file(path, &length);
    if (!data) { bopl_agent_error(error, capacity, "cannot read checkpoint %s", path); return -1; }
    char sidecar[4096]; snprintf(sidecar, sizeof(sidecar), "%s.json", path);
    unsigned char *json=bopl_read_file(sidecar, &json_length);
    if (!json) { free(data); bopl_agent_error(error, capacity, "cannot read sidecar %s", sidecar); return -1; }
    const char *j=(const char*)json;
    char architecture[64]="", sha[80]="", precision[32]="";
    long obs=0, hidden=0, layers=0, sizes[3]={0,0,0}, value=0;
    int ok=bopl_json_string(j, "architecture", architecture, sizeof(architecture))
        && bopl_json_int(j, "obs_size", &obs) && bopl_json_int(j, "hidden", &hidden)
        && bopl_json_int(j, "num_layers", &layers) && bopl_json_ints(j, "action_sizes", sizes, 3)
        && bopl_json_string(j, "weights_sha256", sha, sizeof(sha));
    if (!ok) { free(data); free(json); bopl_agent_error(error, capacity, "sidecar %s lacks architecture/obs_size/hidden/num_layers/action_sizes/weights_sha256", sidecar); return -1; }
    if (bopl_json_string(j, "precision", precision, sizeof(precision)) && strcmp(precision, "float32")) {
        free(data); free(json); bopl_agent_error(error, capacity, "sidecar precision %s (float32 expected)", precision); return -1;
    }
    if (obs!=BOPL_OBS_SIZE || sizes[0]!=BOPL_ACT_AIM || sizes[1]!=BOPL_ACT_BUTTONS || sizes[2]!=BOPL_ACT_HOLD) {
        free(data); free(json);
        bopl_agent_error(error, capacity, "checkpoint layout obs %ld heads %ld/%ld/%ld; this library is obs %d heads %d/%d/%d",
                         obs, sizes[0], sizes[1], sizes[2], BOPL_OBS_SIZE, BOPL_ACT_AIM, BOPL_ACT_BUTTONS, BOPL_ACT_HOLD);
        return -1;
    }
    char actual[65]; bopl_sha256_hex(data, length, actual);
    if (strcmp(actual, sha)) { free(data); free(json); bopl_agent_error(error, capacity, "weights sha256 %s does not match sidecar %s", actual, sha); return -1; }
    if (length%sizeof(float)) { free(data); free(json); bopl_agent_error(error, capacity, "checkpoint size %zu is not float32", length); return -1; }
    /* The training configuration the checkpoint carries. */
    if (bopl_json_int(j, "observation_delay_ticks", &value)) a->delay=(int)value;
    if (bopl_json_int(j, "action_repeat", &value)) a->repeat=(int)value;
    if (bopl_json_int(j, "warmup_ticks", &value)) a->warmup=(int)value;
    if (bopl_json_int(j, "max_episode_ticks", &value)) a->limit=(int)value;
    else if (bopl_json_int(j, "episode_ticks", &value)) a->limit=(int)value;
    const char *slots[3]={"ability1", "ability2", "ability3"};
    for (int s=0; s<3; s++) if (bopl_json_int(j, slots[s], &value)) a->ability[s]=(int)value;
    /* Aim head parameterisation (bopl_aim.h); absent in checkpoints from before 2026-09-12 = plain. */
    if (bopl_json_int(j, "aim_fourier", &value)) a->aim_fourier=(int)value;
    if (bopl_json_int(j, "aim_ar", &value)) a->aim_ar=(int)value;
    free(json);
    if (a->aim_fourier<0 || a->aim_fourier>BOPL_AIM_MAX_HARMONICS || a->aim_ar<0 || a->aim_ar>1) {
        free(data); bopl_agent_error(error, capacity, "unsupported aim_fourier %d / aim_ar %d", a->aim_fourier, a->aim_ar); return -1;
    }
    if (a->delay!=0 && (a->delay!=9 || a->repeat!=4)) { free(data); bopl_agent_error(error, capacity, "unsupported observation delay %d / repeat %d", a->delay, a->repeat); return -1; }
    int set=!strcmp(architecture, BOPL_ARCHITECTURE), linear=!strcmp(architecture, "puffernet-mingru");
    if (!set && !linear) { free(data); bopl_agent_error(error, capacity, "unknown architecture %s", architecture); return -1; }
    a->policy=bopl_policy_eval_create_aim((const float*)data, length/sizeof(float), 1, BOPL_OBS_SIZE, (int)hidden, (int)layers, set, a->aim_fourier, a->aim_ar);
    free(data);
    if (!a->policy) { bopl_agent_error(error, capacity, "weights (%zu floats) do not fit %s hidden %ld layers %ld aim_fourier %d aim_ar %d", length/sizeof(float), architecture, hidden, layers, a->aim_fourier, a->aim_ar); return -1; }
    a->hidden=(int)hidden; a->layers=(int)layers; a->is_policy=1;
    a->rstate=calloc((size_t)layers*hidden, sizeof(float)); a->rnext=calloc((size_t)layers*hidden, sizeof(float));
    a->decoder=calloc(BOPL_DECODER, sizeof(float));
    a->row=calloc((size_t)bopl_policy_eval_width(a->policy)+1, sizeof(float));
    if (!a->rstate || !a->rnext || !a->decoder || !a->row) { bopl_agent_error(error, capacity, "out of memory"); return -1; }
    snprintf(a->id, sizeof(a->id), "policy:%s", sha);
    return 0;
}

static int bopl_agent_open_bot(BoplAgent *a, const char *spec, char *error, size_t capacity) {
    static const char *levels[BOPL_BOT_LEVELS]={"easy", "medium", "hard"};
    for (int kind=1; kind<=BOPL_BOT_KINDS; kind++) for (int level=0; level<BOPL_BOT_LEVELS; level++) {
        char name[64]; snprintf(name, sizeof(name), "%s-%s", BOPL_BOT_NAMES[kind], levels[level]);
        if (strcmp(name, spec)) continue;
        if (kind==BOPL_BOT_SEARCH) {
            bopl_agent_error(error, capacity, "the search bot plans by forking the in-process simulator every decision; it cannot follow a game it does not step. Pick a checkpoint or one of the scripted kinds (archer|pursuer|evasive|mixed[-v2]-easy|medium|hard).");
            return -1;
        }
        a->kind=kind; a->level=level;
        snprintf(a->id, sizeof(a->id), "bot:%s", name);
        return 0;
    }
    bopl_agent_error(error, capacity, "unknown agent '%s': give a policy-*.bin checkpoint (with its .bin.json sidecar) or a scripted bot <archer|pursuer|evasive|mixed>[-v2]-<easy|medium|hard>", spec);
    return -1;
}

BoplAgent *bopl_agent_open(const char *spec, int players, int player, char *error, size_t capacity) {
    if (error && capacity) error[0]=0;
    if (!spec || !*spec) { bopl_agent_error(error, capacity, "empty agent spec"); return NULL; }
    if (players<1 || players>BOPL_MAX_PLAYERS || player<0 || player>=players) { bopl_agent_error(error, capacity, "players %d / player %d", players, player); return NULL; }
    BoplAgent *a=calloc(1, sizeof(*a));
    if (!a) { bopl_agent_error(error, capacity, "out of memory"); return NULL; }
    a->players=players; a->player=player;
    memcpy(a->ability, BOPL_AGENT_DEFAULT_ABILITIES, sizeof(a->ability));
    a->warmup=BOPL_AGENT_DEFAULT_WARMUP; a->limit=BOPL_AGENT_DEFAULT_LIMIT; a->delay=BOPL_AGENT_DEFAULT_DELAY; a->repeat=BOPL_AGENT_DEFAULT_REPEAT;
    size_t n=strlen(spec);
    int status=n>4 && !strcmp(spec+n-4, ".bin")?bopl_agent_open_policy(a, spec, error, capacity):bopl_agent_open_bot(a, spec, error, capacity);
    if (status || bopl_agent_env_alloc(a)) { if (!status) bopl_agent_error(error, capacity, "out of memory"); bopl_agent_close(a); return NULL; }
    if (a->kind) { a->env.bot_select[player]=(uint8_t)a->kind; a->env.bot_select_level[player]=(uint8_t)a->level; }
    return a;
}

void bopl_agent_close(BoplAgent *a) {
    if (!a) return;
    bopl_policy_eval_close(a->policy);
    free(a->rstate); free(a->rnext); free(a->decoder); free(a->row);
    BoplPuffer *env=&a->env;
    free(env->observations); free(env->actions); free(env->rewards); free(env->terminals); free(env->action_mask);
    for (int i=0; i<2; i++) bopl_state_copy_free(&env->delayed[i]);
    bopl_state_copy_free(&env->sample_copy);
    free(a->entities); free(a->slimes);
    free(a);
}

void bopl_agent_reset(BoplAgent *a, uint32_t seed) {
    if (!a) return;
    BoplPuffer *env=&a->env;
    env->config.seed=seed; env->rng=seed;
    env->bot_rng=env->config.seed*2654435761u^0x9E3779B9u;
    if (a->kind) bopl_bot_reset(&env->bots[a->player], a->kind, a->level, bopl_bot_seed(env, a->player));
    /* splitmix64 stream for the sampler, seeded per round like evaluate.py's
     * torch.Generator().manual_seed(seed*7+side). */
    a->rng=(uint64_t)seed*7u+(uint64_t)a->player;
    if (a->rstate) memset(a->rstate, 0, (size_t)a->layers*a->hidden*sizeof(float));
    a->open=0; a->tick=0; a->sampled=0; a->primed=0; a->ended=0; a->armed=1; a->expected=0; a->decisions=0;
    memset(a->decision, 0, sizeof(a->decision));
    memset(&a->phase, 0, sizeof(a->phase));
    memset(env->actions, 0, (size_t)a->players*BOPL_NUM_ATNS*sizeof(float));
}

static uint64_t bopl_splitmix(uint64_t *state) {
    uint64_t z=(*state+=0x9E3779B97F4A7C15ull);
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ull; z=(z^(z>>27))*0x94D049BB133111EBull;
    return z^(z>>31);
}
/* One categorical draw from softmax(logits) over the allowed indices, the
 * distribution torch.multinomial samples from in evaluate.py (masked entries
 * -> -inf). Inverse-CDF on a uniform in [0,1). */
static int bopl_sample_head(const float *logits, const unsigned char *mask, int count, uint64_t *rng) {
    float best=-INFINITY; int allowed=0, last=-1;
    for (int i=0; i<count; i++) if (mask[i]) { allowed++; last=i; if (logits[i]>best) best=logits[i]; }
    if (!allowed) return 0;
    double total=0;
    for (int i=0; i<count; i++) if (mask[i]) total+=exp((double)(logits[i]-best));
    double u=(double)(bopl_splitmix(rng)>>11)*(1.0/9007199254740992.0)*total, acc=0;
    for (int i=0; i<count; i++) if (mask[i]) { acc+=exp((double)(logits[i]-best)); if (u<acc) return i; }
    return last;
}

static void bopl_agent_decide(BoplAgent *a, const BoplRlState *before) {
    BoplPuffer *env=&a->env;
    memset(env->actions, 0, (size_t)a->players*BOPL_NUM_ATNS*sizeof(float));
    if (a->is_policy) {
        float *obs=env->observations+(size_t)a->player*BOPL_OBS_SIZE;
        const unsigned char *mask=env->action_mask+(size_t)a->player*BOPL_LOGITS;
        bopl_policy_eval_step_raw(a->policy, obs, a->rstate, a->row, a->rnext);
        float *swap=a->rstate; a->rstate=a->rnext; a->rnext=swap;
        /* Head order as the trainer draws them: buttons first when the aim
         * head is conditioned on the sampled combo (bopl_aim.h), else aim,
         * buttons, hold. The button and hold logits do not depend on the combo. */
        bopl_policy_eval_game_logits(a->policy, a->row, NULL, a->decoder);
        if (a->aim_ar) {
            a->decision[1]=bopl_sample_head(a->decoder+BOPL_ACT_AIM, mask+BOPL_ACT_AIM, BOPL_ACT_BUTTONS, &a->rng);
            bopl_policy_eval_game_logits(a->policy, a->row, &a->decision[1], a->decoder);
            a->decision[0]=bopl_sample_head(a->decoder, mask, BOPL_ACT_AIM, &a->rng);
        } else {
            a->decision[0]=bopl_sample_head(a->decoder, mask, BOPL_ACT_AIM, &a->rng);
            a->decision[1]=bopl_sample_head(a->decoder+BOPL_ACT_AIM, mask+BOPL_ACT_AIM, BOPL_ACT_BUTTONS, &a->rng);
        }
        a->decision[2]=bopl_sample_head(a->decoder+BOPL_ACT_AIM+BOPL_ACT_BUTTONS, mask+BOPL_ACT_AIM+BOPL_ACT_BUTTONS, BOPL_ACT_HOLD, &a->rng);
        float *action=env->actions+(size_t)a->player*BOPL_NUM_ATNS;
        for (int h=0; h<3; h++) action[h]=(float)a->decision[h];
    }
    bopl_step_begin_from(env, &a->phase, before);
    if (!a->is_policy) {
        /* Report what the bot chose (bopl_step_begin_from ran bopl_bot_decide). */
        a->decision[0]=(int)a->phase.actions[a->player].angle; a->decision[1]=(int)(a->phase.actions[a->player].buttons&15u);
        a->decision[2]=a->phase.holds[a->player];
    }
    a->open=1; a->tick=0; a->sampled=0; a->decisions++;
}

int bopl_agent_tick(BoplAgent *a, const BoplRlState *state, BoplRlAction *out) {
    if (out) memset(out, 0, sizeof(*out));
    if (!a || !state || !out || !a->armed) return -1;
    if ((int)state->header.player_count!=a->players) return -1;
    if (a->ended) return 1;
    BoplPuffer *env=&a->env;
    uint32_t e=state->header.episode_ticks;
    int terminal=(state->header.flags&BOPL_RL_TERMINAL)!=0;
    if (!a->primed) {
        if (e!=0) return -1;
        /* c_reset's bookkeeping from the episode's first state. */
        memset(env->returns, 0, sizeof(env->returns)); env->steps=0;
        memcpy(env->previous_telemetry, state->telemetry, sizeof(env->previous_telemetry));
        for (int i=0; i<a->players; i++) { env->deaths[i]=state->players[i].deaths; env->kills[i]=state->players[i].kills; }
        memset(env->last_telemetry, 0, sizeof(env->last_telemetry));
        bopl_reset_observations(env, state);
        a->primed=1;
    }
    if (e!=a->expected) return -1;
    a->expected=e+1;
    if (a->open) {
        a->tick=(int)(e-a->phase.ticks);
        if (env->observation_delay && !a->sampled && (a->tick==env->repeat-1 || terminal)) { bopl_step_sample(env, state); a->sampled=1; }
        if (a->tick>=env->repeat || terminal) {
            if (env->observation_delay && !a->sampled) return -1;
            bopl_step_finish_from(env, &a->phase, state);
            a->open=0;
        }
    }
    if (terminal) { a->ended=1; return 1; }
    if (!a->open) bopl_agent_decide(a, state);
    *out=a->phase.actions[a->player];
    return 0;
}

/* RlObservation.cs / bopl_rl_state wire format: header, player_count players,
 * entity_count entities, slime_count slimes, player_count telemetry rows. */
int bopl_agent_tick_bytes(BoplAgent *a, const void *bytes, size_t length, BoplRlAction *out) {
    if (out) memset(out, 0, sizeof(*out));
    if (!a || !bytes || !out) return -1;
    if (length<sizeof(BoplRlHeader)) return -2;
    const unsigned char *p=bytes;
    BoplRlState *s=&a->decoded;
    memcpy(&s->header, p, sizeof(BoplRlHeader)); p+=sizeof(BoplRlHeader);
    uint32_t players=s->header.player_count, entities=s->header.entity_count, slimes=s->header.slime_count;
    if (players>BOPL_MAX_PLAYERS) return -2;
    size_t need=sizeof(BoplRlHeader)+(size_t)players*sizeof(BoplRlPlayer)+(size_t)entities*sizeof(BoplRlEntity)
               +(size_t)slimes*sizeof(BoplRlSlime)+(size_t)players*sizeof(BoplRlTelemetry);
    if (length!=need) return -2;
    memcpy(s->players, p, (size_t)players*sizeof(BoplRlPlayer)); p+=(size_t)players*sizeof(BoplRlPlayer);
    if (entities>a->ecap) { free(a->entities); a->entities=calloc(entities, sizeof(BoplRlEntity)); a->ecap=a->entities?entities:0; if (!a->entities) return -1; }
    if (slimes>a->scap) { free(a->slimes); a->slimes=calloc(slimes, sizeof(BoplRlSlime)); a->scap=a->slimes?slimes:0; if (!a->slimes) return -1; }
    memcpy(a->entities, p, (size_t)entities*sizeof(BoplRlEntity)); p+=(size_t)entities*sizeof(BoplRlEntity);
    memcpy(a->slimes, p, (size_t)slimes*sizeof(BoplRlSlime)); p+=(size_t)slimes*sizeof(BoplRlSlime);
    memcpy(s->telemetry, p, (size_t)players*sizeof(BoplRlTelemetry));
    s->entities=a->entities; s->slimes=a->slimes;
    return bopl_agent_tick(a, s, out);
}

int bopl_agent_ability(const BoplAgent *a, int slot) { return a && slot>=0 && slot<3?a->ability[slot]:-1; }
int bopl_agent_warmup_ticks(const BoplAgent *a) { return a?a->warmup:-1; }
int bopl_agent_repeat(const BoplAgent *a) { return a?a->repeat:-1; }
int bopl_agent_observation_delay(const BoplAgent *a) { return a?a->delay:-1; }
int bopl_agent_max_episode_ticks(const BoplAgent *a) { return a?a->limit:-1; }
const char *bopl_agent_id(const BoplAgent *a) { return a?a->id:""; }
int bopl_agent_is_policy(const BoplAgent *a) { return a?a->is_policy:0; }
const float *bopl_agent_observation(const BoplAgent *a) { return a?a->env.observations+(size_t)a->player*BOPL_OBS_SIZE:NULL; }
const unsigned char *bopl_agent_mask(const BoplAgent *a) { return a?a->env.action_mask+(size_t)a->player*BOPL_LOGITS:NULL; }
const float *bopl_agent_logits(const BoplAgent *a) { return a?a->decoder:NULL; }
void bopl_agent_decision(const BoplAgent *a, int out[3]) { if (a && out) memcpy(out, a->decision, sizeof(a->decision)); else if (out) memset(out, 0, 3*sizeof(int)); }
int bopl_agent_decisions(const BoplAgent *a) { return a?a->decisions:0; }
