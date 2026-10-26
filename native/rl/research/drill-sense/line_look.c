/* Layout v4.4's line to the enemy (writer bit 64, bopl_rays_v4.h bopl4f_line), looked at.
 *
 *   line_look synthetic                                  hand-built scenes with their expected fields
 *   line_look <libbopl_sim.so> <corpus> <games> <bits>   real rows: per category a few rows with their scene
 *
 * One JSON line per row: the case, the body, the enemy, the slot's 8 fields as written (bopl4f_write, the
 * CPU writer the CUDA writer equals bit for bit: eval/encode_check.c), the terrain SDF at the body, the walls.
 * Synthetic lines carry the hand-computed fields (`want`). render_line_look.py draws them and recomputes the
 * fields from the walls in float64. Build (from the checkout):
 *   clang -O2 -march=znver4 -ffp-contract=off -std=gnu11 -Wno-unused-function -DBOPL4_SCENE_TAP=line_tap \
 *     -Itrain -Isim/abi native/rl/research/drill-sense/line_look.c sim/abi/bopl_rl.c -lm -lpthread -o line_look */
#define ENV_BENCH_NO_MAIN
static void line_tap(void *env, int a, const void *scene, const float *obs);
#include "../../../../eval/env_bench.c"

static Bopl4Scene tapped[PLAYERS];
static void line_tap(void *env, int a, const void *scene, const float *obs) {
    (void)env; (void)obs;
    if (a>=0 && a<PLAYERS) tapped[a]=*(const Bopl4Scene *)scene;
}

static void emit(const char *what, const char *where, const Bopl4Scene *s, const float *region, const float *want) {
    const float *line=region+BOPL4_TERRAIN+BOPL4_LINE_RAY*BOPL4_RAY_FEATURES;
    printf("{\"case\": \"%s\", \"where\": \"%s\", \"flags\": %d, \"x\": %.9g, \"y\": %.9g, \"ex\": %.9g, \"ey\": %.9g, \"sdf\": %.9g, \"line\": [",
           what, where, s->flags, s->x, s->y, s->enemy_x, s->enemy_y, region[9]*8.0f);
    for (int c=0; c<BOPL4_RAY_FEATURES; c++) printf("%s%.9g", c?", ":"", line[c]);
    printf("]");
    if (want) printf(", \"want\": [%.9g, %.9g, %.9g]", want[0], want[1], want[2]);
    printf(", \"axis\": [");
    for (int k=0; k<s->axis_count; k++)
        printf("%s[%.9g, %.9g, %.9g, %.9g, %.9g]", k?", ":"", s->axis_cx[k], s->axis_cy[k], s->axis_ex[k], s->axis_ey[k], s->axis_radius[k]);
    printf("], \"rot\": [");
    for (int k=0; k<s->rot_count; k++)
        printf("%s[%.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g, %.9g]", k?", ":"", s->rot_cx[k], s->rot_cy[k], s->rot_rx[k], s->rot_ry[k],
               s->rot_ux[k], s->rot_uy[k], s->rot_ex[k], s->rot_ey[k], s->rot_radius[k]);
    printf("]}\n");
}

/* A scene of aligned boxes (cx, cy, half extents, radius 0), the body and the enemy. */
static void synth(const char *what, int bits, float x, float y, int enemy, float ex, float ey, int n, const float (*box)[4], const float *want) {
    Bopl4Scene s; memset(&s, 0, sizeof(s));
    Bopl4Enemy e; memset(&e, 0, sizeof(e));
    e.known=enemy; e.x=ex; e.y=ey;
    bopl4_scene_head(&s, bits, x, y, 0.0, 0.0, 0, 0.0f, 0.0f, &e);
    for (int k=0; k<n; k++) {
        s.axis_cx[k]=box[k][0]; s.axis_cy[k]=box[k][1]; s.axis_ex[k]=box[k][2]; s.axis_ey[k]=box[k][3]; s.axis_radius[k]=0.0f;
    }
    s.axis_count=n;
    float region[BOPL4_TERRAIN+BOPL4_RAYS*BOPL4_RAY_FEATURES];
    bopl4f_write(&s, bopl4_dirs(), region);
    emit(what, "synthetic", &s, region, want);
}

static int synthetic(void) {
    /* A platform 20 wide, 5 deep, its top at y = 0; a second one at y = 15..20 (for the tunnel case). */
    static const float platform[1][4]={{0.0f, -2.5f, 10.0f, 2.5f}};
    static const float two[2][4]={{0.0f, -2.5f, 10.0f, 2.5f}, {0.0f, 17.5f, 10.0f, 2.5f}};
    static const float wall[1][4]={{10.0f, 5.0f, 1.0f, 10.0f}};   /* a pillar x 9..11, y -5..15 */
    /* Hand-computed: [0] = min(cast, L) / 40; samples t_k = L (k + 0.5) / 32. */
    float w[3];
    /* 1. Open air: both above the platform, 20 apart, nothing between: cast 40, L 20. */
    w[0]=20.0f/40.0f; w[1]=0.0f; w[2]=1.0f; synth("open air", 112, -10.0f, 5.0f, 1, 10.0f, 5.0f, 1, platform, w);
    /* 2. Opponent beyond 40 u in open air: cast 40, L 50 -> 1. */
    w[0]=1.0f; w[1]=0.0f; w[2]=1.0f; synth("open air, 50 u away", 112, -25.0f, 30.0f, 1, 25.0f, 30.0f, 1, platform, w);
    /* 3. Behind a platform: body (0, 10), enemy (0, -10) under it: the cast hits the top at 10; samples in
     *    y in (-5, 0) <=> t in (10, 15) <=> k + 0.5 in (16, 24): k = 16..23, 8 of 32; after the last, 31 - 23 = 8. */
    w[0]=10.0f/40.0f; w[1]=8.0f/32.0f; w[2]=8.0f/32.0f; synth("behind a platform (enemy below it)", 112, 0.0f, 10.0f, 1, 0.0f, -10.0f, 1, platform, w);
    /* 4. Behind a pillar: body (0, 5), enemy (20, 5): the cast hits x = 9 at 9; samples in x in (9, 11) <=>
     *    k + 0.5 in (14.4, 17.6): k = 14..17, 4; after 31 - 17 = 14. */
    w[0]=9.0f/40.0f; w[1]=4.0f/32.0f; w[2]=14.0f/32.0f; synth("behind a pillar", 112, 0.0f, 5.0f, 1, 20.0f, 5.0f, 1, wall, w);
    /* 5. Driller underground (body (0, -2.5), inside the platform) facing the opponent above at (0, 10), bit 32:
     *    the cast marches out to the top at 2.5; L 12.5, samples in y < 0 <=> t < 2.5 <=> k + 0.5 < 6.4: k = 0..5,
     *    6; after 31 - 5 = 26. */
    w[0]=2.5f/40.0f; w[1]=6.0f/32.0f; w[2]=26.0f/32.0f; synth("driller underground, opponent above (bits 112)", 112, 0.0f, -2.5f, 1, 0.0f, 10.0f, 1, platform, w);
    /* 6. The same row without bit 32 (bits 80 = 64 | 16): no exit march, the cast stops at once. */
    w[0]=0.0f; w[1]=6.0f/32.0f; w[2]=26.0f/32.0f; synth("driller underground, opponent above (bits 80, no exit march)", 80, 0.0f, -2.5f, 1, 0.0f, 10.0f, 1, platform, w);
    /* 7. Driller underground under the opponent's platform: body (0, -2.5), enemy (0, 21) inside the upper
     *    platform (y 15..20 is ground; the enemy's centre above at 21 would be clear: put it at 17 to be in
     *    ground): L 19.5, ground t < 2.5 (k + 0.5 < 4.10: k 0..3) and t in (17.5, 22.5) (k + 0.5 > 28.72: k 29..31),
     *    7 of 32, the last sample in ground: run-in 0. The cast exits at 2.5. */
    w[0]=2.5f/40.0f; w[1]=7.0f/32.0f; w[2]=0.0f; synth("driller underground, opponent in the ground above", 112, 0.0f, -2.5f, 1, 0.0f, 17.0f, 2, two, w);
    /* 8. Tunnel route: body (-9, -2.5) and enemy (9, -2.5) both inside the platform: every sample in ground;
     *    the cast exits upward? No: along +x it stays inside to x = 10 (the side), 19 u: [0] = min(19, 18) / 40. */
    w[0]=18.0f/40.0f; w[1]=1.0f; w[2]=0.0f; synth("both underground (tunnel)", 112, -9.0f, -2.5f, 1, 9.0f, -2.5f, 1, platform, w);
    /* 9. Enemy unknown, and enemy on the body: the slot is zero. */
    w[0]=0.0f; w[1]=0.0f; w[2]=0.0f; synth("enemy unknown", 112, -10.0f, 5.0f, 0, 0.0f, 0.0f, 1, platform, w);
    synth("enemy on the body", 112, -10.0f, 5.0f, 1, -10.0f, 5.0f, 1, platform, w);
    /* 10. Bit 64 with the fan ablated (bits 66): zero. */
    synth("fan ablated (bits 66)", 66, 0.0f, 10.0f, 1, 0.0f, -10.0f, 1, platform, w);
    return 0;
}

/* Categories of real rows: the slot and the body's SDF. */
enum { C_OPEN, C_BEHIND, C_END_GROUND, C_UNDER_CLEAR, C_UNDER_TUNNEL, C_FAR, C_COUNT };
static const char *category[C_COUNT]={"open air", "behind terrain", "enemy end in ground", "underground, clear after the exit",
                                      "underground, ground again later", "open air, enemy beyond 40 u"};
static int classify(const Bopl4Scene *s, const float *region, unsigned *ground_out) {
    const float *line=region+BOPL4_TERRAIN+BOPL4_LINE_RAY*BOPL4_RAY_FEATURES;
    float dx, dy, len;
    if (!bopl4f_enemy_line(s, &dx, &dy, &len)) return -1;
    unsigned ground=0;
    Bopl4Mask all=bopl4f_all(s);
    for (int k=0; k<32; k++) { float x, y; bopl4f_line_point(s, dx, dy, len, k, &x, &y); if (bopl4f_sdf(s, all, x, y)<0.0f) ground|=1u<<k; }
    *ground_out=ground;
    int inside=region[9]<0.0f;
    if (inside) {
        unsigned run=ground&1u ? (ground==0xffffffffu ? ground : ((ground+1u)&~ground)-1u) : 0u;   /* the first run of ground */
        return ground==run ? C_UNDER_CLEAR : C_UNDER_TUNNEL;
    }
    if (line[1]==0.0f) return len>40.0f ? C_FAR : C_OPEN;
    return line[2]==0.0f ? C_END_GROUND : C_BEHIND;
}

static int corpus(const char *lib, const char *dir_path, int max_games, int bits) {
    DIR *dir=opendir(dir_path);
    if (!dir) { fprintf(stderr, "%s: cannot open\n", dir_path); return 1; }
    char **names=NULL; int n=0, cap=0; struct dirent *e;
    while ((e=readdir(dir))) {
        size_t len=strlen(e->d_name);
        if (len<5 || strcmp(e->d_name+len-4, ".rep")) continue;
        if (n==cap) { cap=cap?cap*2:256; names=realloc(names, sizeof(*names)*(size_t)cap); }
        names[n++]=strdup(e->d_name);
    }
    closedir(dir);
    qsort(names, (size_t)n, sizeof(*names), cmp_name);
    if (max_games>0 && n>max_games) n=max_games;
    char error[512], log[64]; snprintf(log, sizeof(log), "/tmp/line-look-%d.log", (int)getpid());
    BoplRl *world=bopl_rl_open(lib, "/dev/null", log, 30000, error, sizeof(error));
    if (!world) { fprintf(stderr, "open failed: %s\n", error); return 1; }
    static float scratch[PLAYERS*BOPL_OBS_STRIDE];
    static unsigned char masks[PLAYERS][BOPL_LOGITS];
    BoplPuffer *env=calloc(1, sizeof(BoplPuffer));
    env_init(env, world, scratch, NULL, masks, bits);
    uint64_t counts[C_COUNT]={0}, rows=0, inside_rows=0;
    int shown[C_COUNT]={0}, per_game=0;
    const char *every_env=getenv("LINE_LOOK_EVERY");
    int every=every_env?atoi(every_env):0;
    BoplStateCopy delayed[2]={0}, sample={0};
    for (int g=0; g<n; g++) {
        char path[4096]; snprintf(path, sizeof(path), "%s/%s", dir_path, names[g]);
        Replay replay;
        if (replay_decode(path, 240, 10800, &replay)) return 1;
        if (bopl_rl_reset(world, &replay.config)!=BOPL_RL_OK) { fprintf(stderr, "%s: reset failed\n", path); return 1; }
        env->config=replay.config; bopl_written_forget(env);
        memset(env->action_history, 0, sizeof(env->action_history)); memset(env->applied, 0, sizeof(env->applied));
        const BoplRlState *state=bopl_rl_state(world);
        for (int i=0; i<2; i++) bopl_state_copy(&delayed[i], state);
        int cursor=0; uint32_t tick=1; uint64_t decisions=0;
        per_game=0;
        while (!(state->header.flags&BOPL_RL_TERMINAL) && 240+tick-1<replay.tick_count) {
            BoplRlAction actions[PLAYERS];
            memcpy(actions, &replay.actions[(size_t)(240+tick-1)*PLAYERS], sizeof(actions));
            uint32_t before=state->header.episode_ticks;
            if (bopl_rl_step_sampled(world, actions, PLAYERS, 3, 1)) { fprintf(stderr, "%s: step failed\n", path); return 1; }
            bopl_state_copy(&sample, bopl_rl_sample(world));
            state=bopl_rl_state(world);
            tick+=state->header.episode_ticks-before; decisions++;
            if (state->header.flags&BOPL_RL_TERMINAL) break;
            for (int i=0; i<PLAYERS; i++) env->applied[i]=actions[i];
            bopl_encode(env, bopl_state_copy_get(&delayed[cursor]), 1);
            for (int a=0; a<PLAYERS; a++) {
                const float *region=bopl_obs_row(env, a)+BOPL4_OBS_TERRAIN_OFFSET;
                unsigned ground=0;
                int c=classify(&tapped[a], region, &ground);
                if (c<0) continue;
                rows++; counts[c]++; inside_rows+=region[9]<0.0f;
                if (every>0 && rows%(uint64_t)every==0) {   /* LINE_LOOK_EVERY=N: every Nth row, for a bulk recompute */
                    char where[256]; snprintf(where, sizeof(where), "%s decision %" PRIu64 " player %d", names[g], decisions, a);
                    emit(category[c], where, &tapped[a], region, NULL);
                    continue;
                }
                /* Up to 3 a category, one a game, from decision 40 on, every 7th candidate. */
                if (shown[c]<3 && per_game<2 && decisions>40 && counts[c]%7==3) {
                    char where[256]; snprintf(where, sizeof(where), "%s decision %" PRIu64 " player %d", names[g], decisions, a);
                    emit(category[c], where, &tapped[a], region, NULL);
                    shown[c]++; per_game++;
                }
            }
            BoplStateCopy published=delayed[cursor]; delayed[cursor]=sample; sample=published; cursor^=1;
        }
        free(replay.actions);
    }
    printf("{\"summary\": true, \"games\": %d, \"bits\": %d, \"rows\": %" PRIu64 ", \"inside\": %" PRIu64, n, bits, rows, inside_rows);
    for (int c=0; c<C_COUNT; c++) printf(", \"%s\": %" PRIu64, category[c], counts[c]);
    printf("}\n");
    bopl_rl_close(world);
    return 0;
}

int main(int argc, char **argv) {
    if (argc==2 && !strcmp(argv[1], "synthetic")) return synthetic();
    if (argc==5) return corpus(argv[1], argv[2], atoi(argv[3]), atoi(argv[4]));
    fprintf(stderr, "line_look synthetic | line_look <libbopl_sim.so> <corpus> <games> <bits>\n");
    return 2;
}
