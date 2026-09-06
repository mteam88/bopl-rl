/* The single-roundtrip sampled step must reproduce two step commands exactly:
 * the sample equals the state after the first phase, the final state equals
 * the state after the second phase, and an episode that ends during the first
 * phase yields identical terminal sample and final states. With an older game
 * build the client must fall back transparently and report it. */
#include "bopl_rl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { unsigned char *bytes; size_t length; } Snapshot;
static Snapshot snapshot(const BoplRlState *state) {
    size_t entities=state->header.entity_count*sizeof(BoplRlEntity);
    size_t slimes=state->header.slime_count*sizeof(BoplRlSlime);
    Snapshot result={NULL, sizeof(state->header)+sizeof(state->players)+entities+slimes+sizeof(state->telemetry)};
    result.bytes=malloc(result.length);
    if (!result.bytes) abort();
    unsigned char *at=result.bytes;
    memcpy(at, &state->header, sizeof(state->header)); at+=sizeof(state->header);
    memcpy(at, state->players, sizeof(state->players)); at+=sizeof(state->players);
    if (entities) memcpy(at, state->entities, entities);
    at+=entities;
    if (slimes) memcpy(at, state->slimes, slimes);
    at+=slimes;
    memcpy(at, state->telemetry, sizeof(state->telemetry));
    return result;
}
static int same(Snapshot a, Snapshot b) { return a.length==b.length && !memcmp(a.bytes, b.bytes, a.length); }
static void actions_for(BoplRlAction *actions, unsigned int step) {
    for (uint32_t i=0; i<2; i++) {
        actions[i].angle=(uint8_t)((step*17+i*39)%256);
        actions[i].buttons=(uint16_t)((step+3*i)%16);
        actions[i].reserved=0;
    }
}
static int fail(const char *what) { fprintf(stderr, "FAIL: %s\n", what); return 1; }

/* Run one episode with the two-command reference, then the same episode with
 * sampled steps; every decision's (sample, final) pair must match exactly. */
static int compare_episode(BoplRl *world, BoplRlConfig *config, int expect_native, unsigned int *decisions_out, int *phase1_terminal) {
    Snapshot phase[2][2048]; unsigned int count=0;
    if (bopl_rl_reset(world, config)) return fail(bopl_rl_error(world));
    while (!(bopl_rl_state(world)->header.flags&BOPL_RL_TERMINAL)) {
        BoplRlAction actions[2]; actions_for(actions, count);
        if (count>=2048) return fail("episode too long for the test buffer");
        if (bopl_rl_step(world, actions, 2, 3)) return fail(bopl_rl_error(world));
        phase[0][count]=snapshot(bopl_rl_state(world));
        if (!(bopl_rl_state(world)->header.flags&BOPL_RL_TERMINAL) && bopl_rl_step(world, actions, 2, 1)) return fail(bopl_rl_error(world));
        phase[1][count]=snapshot(bopl_rl_state(world));
        count++;
    }
    if (bopl_rl_reset(world, config)) return fail(bopl_rl_error(world));
    if (bopl_rl_sample(world)) return fail("sample valid after reset");
    unsigned int step=0;
    while (!(bopl_rl_state(world)->header.flags&BOPL_RL_TERMINAL)) {
        BoplRlAction actions[2]; actions_for(actions, step);
        if (step>=count) return fail("sampled episode longer than the reference");
        if (bopl_rl_step_sampled(world, actions, 2, 3, 1)) return fail(bopl_rl_error(world));
        const BoplRlState *sample=bopl_rl_sample(world), *final=bopl_rl_state(world);
        if (!sample || !final) return fail("missing sample or final state");
        Snapshot s=snapshot(sample), f=snapshot(final);
        if (!same(s, phase[0][step])) return fail("sample differs from the two-command phase-1 state");
        if (!same(f, phase[1][step])) return fail("final state differs from the two-command phase-2 state");
        if ((sample->header.flags&BOPL_RL_TERMINAL) && !same(s, f)) return fail("terminal in phase 1 but sample and final differ");
        if ((sample->header.flags&BOPL_RL_TERMINAL)) *phase1_terminal=1;
        if (final->header.episode_ticks-sample->header.episode_ticks>1) return fail("second phase ticked more than once");
        free(s.bytes); free(f.bytes);
        step++;
    }
    if (step!=count) return fail("sampled episode shorter than the reference");
    if (bopl_rl_sampled_native(world)!=expect_native) return fail(expect_native?"client fell back to two commands on the new build":"client did not fall back on the old build");
    for (unsigned int i=0; i<count; i++) { free(phase[0][i].bytes); free(phase[1][i].bytes); }
    *decisions_out=count;
    return 0;
}

int main(int argc, char **argv) {
    if (argc!=5) { fprintf(stderr, "sampled_check <absolute game> <absolute bootstrap.rep> <absolute game.log> <expect_native 0|1>\n"); return 2; }
    int expect_native=atoi(argv[4]);
    char error[512];
    BoplRl *world=bopl_rl_open(argv[1], argv[2], argv[3], 30000, error, sizeof(error));
    if (!world) { fprintf(stderr, "%s\n", error); return 1; }
    BoplRlConfig config; bopl_rl_default_config(&config, 2);
    for (int i=0; i<2; i++) { config.player[i].abilities[0]=2; config.player[i].abilities[1]=4; config.player[i].abilities[2]=14; }
    config.stocks=3;
    BoplRlAction actions[2]; actions_for(actions, 0);
    /* Argument guards, including the overflow case the parent review flagged. */
    config.seed=1; config.max_episode_ticks=600;
    if (bopl_rl_reset(world, &config)) return fail(bopl_rl_error(world));
    if (bopl_rl_step_sampled(world, actions, 2, 0, 1)!=BOPL_RL_ARGUMENT) return fail("first=0 accepted");
    if (bopl_rl_step_sampled(world, actions, 2, 3, 0)!=BOPL_RL_ARGUMENT) return fail("second=0 accepted");
    if (bopl_rl_step_sampled(world, actions, 2, 8, 9)!=BOPL_RL_ARGUMENT) return fail("first+second=17 accepted");
    if (bopl_rl_step_sampled(world, actions, 2, 0xffffffffu, 2)!=BOPL_RL_ARGUMENT) return fail("uint32 overflow accepted");
    if (bopl_rl_state(world)->header.episode_ticks!=0) return fail("rejected arguments advanced the game");
    unsigned int total=0, decisions; int phase1_terminal=0, seen_phase1=0;
    /* Full matches on three seeds (600-tick limit keeps this quick; timeouts end on a multiple of 4). */
    for (uint32_t seed=101; seed<104; seed++) {
        config.seed=seed; config.max_episode_ticks=600;
        if (compare_episode(world, &config, expect_native, &decisions, &phase1_terminal)) return 1;
        total+=decisions;
    }
    /* Episode limits that are not multiples of four end inside the first phase (3 ticks). */
    for (uint32_t limit=9; limit<=11; limit++) {
        config.seed=7; config.max_episode_ticks=limit; phase1_terminal=0;
        if (compare_episode(world, &config, expect_native, &decisions, &phase1_terminal)) return 1;
        if (!phase1_terminal) return fail("expected the episode to end during the first phase");
        if (!(bopl_rl_state(world)->header.flags&BOPL_RL_TRUNCATED)) return fail("expected truncation");
        seen_phase1++; total+=decisions;
    }
    /* Stepping a finished episode is rejected the same way for both commands. */
    if (bopl_rl_step_sampled(world, actions, 2, 3, 1)!=BOPL_RL_ENDED) return fail("sampled step after terminal accepted");
    if (bopl_rl_step(world, actions, 2, 3)!=BOPL_RL_ENDED) return fail("step after terminal accepted");
    printf("{\"status\":\"pass\",\"decisions\":%u,\"phase1_terminal_episodes\":%d,\"sampled_native\":%d}\n", total, seen_phase1, bopl_rl_sampled_native(world));
    bopl_rl_close(world);
    return 0;
}
