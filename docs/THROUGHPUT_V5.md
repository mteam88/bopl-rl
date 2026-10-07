# Throughput campaign 4: the specialist workload

Started 2026-10-06. The target is the 15-kit specialist retrain on the merged
sim (sim32b8446a): native/data/lineups-matthew.txt kits, the 111 training
maps, 3-minute games (10800 ticks), delay 9, self-play pool. generalist-111,
the closest measured run, trained at ~310k sps (232k learner) and was
env-CPU bound: per update, rollout 0.202 s, of which eval_env 0.163 s, against
eval_gpu 0.036 s and train 0.043 s. Campaign 3's 645k was a different
workload (kit 2/4/14 on maps 0-2, lite writer), so it is not a baseline here.

## The bench

Random-input benches (writer_bench, benchmark.c) do not visit the states a
trained policy does: sps moved 2x across generalist-111's own run. So the
bench replays a corpus of trained-policy games:

- `eval/perf/record_corpus.py` records BOPLREP2 games of generalist-111's
  final checkpoint against itself: the 15 kits cross-paired, the training
  maps, 10800-tick limit, delay 9. There are 400 games (2.08M ticks) in
  `native/artifacts/rl/env-corpus/all`. `train/` holds games 0-199 (the PGO
  profile draws from these) and `test/` holds 200-399, held out.
  All 400 replay to their recorded length.
- `eval/env_bench.c` (`make env-bench`) re-steps the games through c_step's
  phases on one core and times each phase on its own: reset, sim
  (`bopl_rl_step_sampled(.., 3, 1)`), sample copy and overflow count, the lite
  (bits 10) and full (bits 0) writers on the delayed state, tail, privileged
  block and pack. Every game's hash folds every sampled and final state
  (replay_ab.c's strict state hash) and both writers' rows. Equal hashes mean
  the change is exact.
- `eval/perf/env_ab.sh base.so cand.so` interleaves the two builds on one
  core and reports the hash agreement and the median sim ratio.
- `eval/perf/exact_gate.sh base.so cand.so out` is the all-ability gate: 12.4k
  random-play divergence_rate games, all 38 abilities on all 124 maps, 2p and
  6p, DR_TOL=0. It flags all 372 games when run against the q32 build, so it
  does detect divergence.
- For profiles, build with `-C force-frame-pointers=yes` and line tables, use
  `perf record --call-graph=fp`, and aggregate with `eval/perf/callees.py`.
  DWARF unwinding does not get through the LTO'd sim.

## Where the time went (full corpus, µs per env decision)

| | base | now |
| --- | ---: | ---: |
| sim | 112.3 | 22.5 |
| writer, lite (bits 10) | 4.7 | 4.5 |
| writer, full (bits 0) | 14.7 | 14.5 |
| sample + tail + priv + pack | 0.8 | 0.8 |
| env step, lite writer | 118.0 | 28.0 |
| env step, full writer | 128.0 | 37.9 |

The writer was not the cost. The sim was 88% of the env step, and 40% of the
sim was the C#-order Updater scheduler from the fidelity merge.

## Levers kept (all exact: identical env_bench hash and gate)

1. **`updater_order` takes each key once.** The scheduler sorted pools with
   `sort_unstable_by_key(|i| (key(pool[i]), i))`. Each `updater_key` does two
   binary searches of Updater's list and a linear scan of a pass's batch, and
   a sort runs about 2n·log n of them. Now it computes n keys, then sorts the
   (key, slot) pairs. The pairs are distinct, so the order is the same.
2. **A smoke cloud's turn sorts only its own cloud.** `run_spawned_before`
   reaches kind 9 once per cloud per pass, and each time
   `smoke_clouds_update_early` sorted every cloud to run the one whose rows
   are in `spawned`. Now it filters first (`updater_order_where`).
   Levers 1 and 2 together cut the sim 2.65x.
3. **Pushes are prefiltered by append pass.** An Updater key leads with its
   append pass, so an object appended after this pass, after `before`'s pass,
   or before the cursor's pass cannot be pushed. It now skips the key search.
   91% of the 6.5 `run_spawned_before` calls per tick queue nothing.
   This gave 5%.
4. **`remove_destroyed_colliders` scans once when there is nothing to drop.**
   It used to make four passes over every entity row on every query.
5. **`step_sampled` publishes the main observation once.** It used to publish
   it at phase 3 and again at the end, and the sample rebuilt the same rows in
   between. Numbering is unchanged, because the sample numbers the same rows
   in the same order. This gave 1%.
6. **PGO** (`native/tools/pgo_build.sh`, now `just build` and the Makefile's
   f64 rule). The sim is instrumented, re-steps 40 train games
   (native/pgo/corpus.tar.xz, 180 KB), then rebuilds with the profile.
   This gave 22% on the held-out games, and the hash is unchanged. It needs
   `rustup component add llvm-tools`.

7. **The full writer's march skips walls no ray of a group can reach.**
   `bopl4_terrain_anchor` takes each wall's SDF at the body once per row.
   `bopl4_march` keeps a wall only if its body SDF is at most
   max_i(min(max, limit_i) + |O_i - B|) + 0.1. A culled wall is never a hit, and
   when it would be the minimum the ray ends on the next check either way
   (the proof is at `bopl4_march`). The 400-game hash was unchanged and
   writer_bench's march-check (since replaced by its rows hash) passed. Of the 12.5 walls a map has, 6.5 are
   kept per march, and the full writer is 8-10% faster.
   Culling cannot get tighter: the first step's disk around the origin binds
   for every direction.
8. **The scheduler caches each pass's batch, and the sweep reads pool rows'
   colliders directly.** `updater_pass_batch` did two binary searches per
   call; a cached range is reused when its edges still split the
   pass-sorted list at that pass. `take_sweep` resolved every row through
   `current_bounds`, which searches the pools; only pool-owned rows need
   that now. The gate is clean, and the sim is 7% faster.
9. **The physics groups take each part's rank, layer, composite and swept
   bounds once.** The Circle/Circle, Box/Box and Circle/Box pair loops
   recomputed them per pair, and the stable sort per comparison. That is
   O(n^2) in a group's parts, and a Smoke cast (ability 13) fills the
   Circle/Circle group with clouds. Kits with Smoke cost 2.5x the sim of the
   rest (61 vs 24 us per decision) and carry 25% of the corpus's sim time.
   The sim is 29% faster on Smoke games and 10% faster over the held-out
   games.
10. **A Smoke cloud's early turn runs that cloud alone.** Every cloud's
    prefab base is unique (`register_prefab` advances by SmokeCircle's five
    rows), so the turn `run_spawned_before` queued for cloud i is cloud i's.
    It no longer filters and keys every cloud to find it. About 32 such turns
    run per tick with 34 clouds alive. Smoke games retire 8% fewer sim
    instructions, and the held-out games 3% fewer.
11. **Updater places are memoized under a list version.** A key's place
    in its pass's batch depends on the list alone, so `updater_owner_position`
    and `updater_member_key` cache it by (pass, owner or member) in 256-slot
    tables. The version bumps on every append, removal, reset or fork.
    Smoke games retire 7.5% fewer sim instructions, and the held-out games
    3% fewer. (A first try hashed `owner.rotate_left(7)`. Bases step by 5,
    so it used 2 of the 256 slots and measured 0.7%.)
12. **The writer bins dynamic entities into octants without atan2.** It
    reads the octant off the signs and whichever of |dx| and |dy| is larger,
    and takes the old atan2 path within 1e-6 (relative) of an octant edge.
    The 400-game hash is unchanged, and the writer is 4-8% faster.
13. **A ray group takes its circles' offsets once.** Every ray of a group
    leaves the body. Each circle's offset, squared length and squared
    radius are now computed per group, not per ray, using the same
    expressions; the C build has no FMA, so nothing contracts. The hash is
    unchanged, and the writer retires about 3.5% fewer instructions.
14. **A wall group asks each wall's side once per pass.** wall_group used to
    look up each wall's collides flag, layer and swept bounds once per
    (part, wall) pair. It now looks them up once per wall and reuses them
    for every part. The gate and the hash are unchanged, and the sim retires
    0.6% fewer instructions on the test games.
15. **A Shockwave's captured targets live beside it, not in it.** Each
    Shockwave carried its 50-entry CircleCastAll lists inline, making it
    5.1 KB. Each pass scans the 36 gust slots and the meteor waves about 7
    times (run_spawned_before, register_pending_objects,
    shockwaves_update), so every check of `active` touched a separate
    cache line 5 KB apart. The lists now live in a parallel WaveTargets
    pool, and a Shockwave is about 100 bytes. The gate and the hash are
    unchanged. Sim instructions are flat (-0.06%) and sim time is -0.9%
    (interleaved, 200 test games).
16. **run_spawned_before skips pool members it would drop before keying
    them.** It takes about 5 walks per tick. Each walk scanned all 36 gust
    slots, though only `3 * players.len()` can hold a wave: `players` only
    grows within a game, and reset clears both. It also computed every
    spawned platform's Updater key and every thrown item's batch lookup
    before the `appended` prefilter that rejects them. Both now come after
    the prefilter, and the gust scans (here, in register_pending_objects
    and in shockwaves_update) cover only the fillable slots. The gate and
    the hash are unchanged, and the sim retires 2.8% fewer instructions on
    the test games.
17. **bopl_clip is two selects.** Its first `if` compiled to a branch whose
    direction depends on the data. As `v<1?v:1` then `v>-1?v:-1` it becomes
    minsd/maxsd, which give the same value for every input (NaN still
    maps to 1). The hash is unchanged. Interleaved, the full writer is 1.4%
    faster and the lite writer 4.5%.
18. **A ray group meets the circles 8 rays at a time.** The circle test (each
    entity and slime as a disc) ran ray by ray with two data-dependent
    skips per (ray, circle). bopl4_circles_avx512 puts 8 of the group's rays
    in the lanes and walks the circles in order. Each (ray, circle) gets the
    same IEEE operations, the skips become negated compares (so a NaN goes
    on as before), and each lane keeps the strict `d < best` in circle
    order. It records the winning circle, and the class and closing speed
    are read from it after the loop as before. writer_bench's vector-vs-scalar
    check (now covering the circles too) finds 0 mismatches in 8192 rows for
    both loadouts, and the hash is unchanged. Interleaved on the test games,
    the full writer goes from 12.25 to 11.07 us (-9.6%).
19. **The drop and ceiling casts ride in the fan's march.** bopl4_terrain
    marched its two vertical casts alone, 2 of 8 lanes for ~10 steps (11%
    of all march iterations). The march now takes a per-ray max, and the
    fan's first group carries the two casts. A ray's march does not depend
    on the others in its call, and the culling reach only grows. The hash
    and writer_bench's rows hash are unchanged. Full writer: 11.07 to
    10.49 us (-5.2%).
20. **Clang builds the writer.** train/build.py compiled binding.c (the
    writer) with gcc -O3 and no -march. Clang -O3 -march=znver4
    -ffp-contract=off (no FMA contraction, so exact: same env_bench hash on
    test/100 and all/400) runs it faster. With 12 concurrent env_bench
    copies, the full writer goes from 16.0 to 14.2 us (-11%) and the lite
    writer from 7.0 to 5.5 us (-22%). Clang alone gives about 6% and -march
    the rest; gcc with -march gives 4%, and GCC PGO of the writer 1%.
    Clang's OpenMP needs libomp, which the gcc-linked extension lacks, so
    binding.c builds without -fopenmp. Its one parallel loop is the
    single-buffer cpu_step/gpu_step fallback, not the env pool, and it runs
    serially to the same result. `make env-bench` builds the same way.
    End to end (spec, 3 interleaved repeats): 347k to 360k sps (+3.7%),
    every candidate run above every incumbent run.


## Cold caches in the trainer

`BOPL_ENV_BENCH_EVICT_KB` makes env_bench sweep a buffer that size between
decisions, outside the timers, as the env pool does when a thread steps
other worlds. On v7c-pgo with 100 test games, the L2 sweeps cost sim
+16% at 4 MB and +18% at 32 MB, and the full writer +6%. A 128 MB sweep
also flushes L3, and costs the sim +70% and the full writer +27%. With all
12 threads busy (no sweep), the sim takes 28.9 us and the writer 17.7 us,
1.77x and 1.5x solo. The trainer's 35.8 us sim is about the 12-busy figure
times the L2-cold factor, so worlds mostly stay in the X3D's L3. IBS
sampling of the 128 MB case spreads the L3 and DRAM misses over dozens of
sites, with no single structure to repack.

## Measured and not kept

- `overflow-checks = false`: 2%. The checks stay as a safety net.
- A no-hit certificate (skipping the march for a ray whose segment stays
  more than 0.05 from every wall) was not built. 58% of rays end without a
  hit, but they average 6 steps each (the sky, not grazing the floor; only
  1% hit the 32-step cap), so it nets about 10% of the march.

- A 4-lane (AVX-512VL, ymm) march in place of the 8-lane one: 4.47 of 8
  lanes are live per iteration, and Zen 4 runs zmm as two 256-bit halves.
  It was exact but 5% slower on the full writer: the march is bound by the
  per-step latency chain (t, point, SDF, sqrt, min, t), not by lane
  throughput, so fewer lanes only means more steps.
- Skipping a wall's sqrt when its lower bound max(qx, qy) - radius is
  above every live lane's smallest upper bound (exact, and only 31% of
  (wall, step) pairs need the sqrt): 22% slower on the full writer. The
  bound pass lengthens each step's dependent chain. Under trainer-like
  SMT load (12 concurrent env_bench copies, siblings sharing the sqrt
  unit) it also loses: full writer 16.6 to 20.0 us.
- Two independent 8-lane chains in one march loop (exact): 12% slower. It
  spills (54 stack references against 12), and a 16-wide drain wastes
  more lanes. A microbenchmark of the 5-wall step on this core: 74 TSC
  ticks independent and 107 dependent, 41 and 55 without the sqrt. The
  zmm sqrt (~14 ticks each) sets the floor.

- BOLT (llvm-bolt with an LBR profile of env_bench, on top of PGO):
  exact, 2.8% slower solo. I-cache misses fell only 4.5% and op-cache
  misses 2%: the sim's code misses are capacity, not layout.
- `opt-level = "s"`: 12.5% slower; `opt-level = 2`: 6.8% slower.
- Fewer env threads to dodge SMT contention: 12 busy copies on 6 cores
  give 19% more decisions than 6 copies, so the pool keeps all threads.
- Stage-major batching (each sim stage runs over k worlds before the
  next stage): exact. Solo 4.5% faster, but with 12 busy copies only 2-3%
  at k = 2 and 4 and worse at k = 8. I-cache misses halve while L1d
  refills rise 39%. Not worth reworking the pool around it.
- Prefetching the next world's hot regions before stepping it recovers
  about 2% of a 12% world-cycling tax. In the trainer, data dominates the
  misses (about 48 KB of L2 misses per env step, against 30 KB of code),
  spread over many structures.
- Answering a player's own hurtbox in `body_part_circle` without
  `detphysics_collider`'s pool scans: exact (debug assertion over all 400
  games), 0.2% faster. Not worth the special case.

## The writer on the GPU (2026-10-06)

With the sim cut to ~38 us per c_step in the pool, the full writer was a
third of the env step. Matthew chose the full writer on the GPU, in FP32.

- **Why FP32.** The RTX 5070 retires 234 G FP64 add/mul and 27 G sqrt per
  second. The writer's ~90k FP64 operations per row would take most of the
  GPU at 1M rows per second. So the terrain block and the rays
  (`train/bopl_rays_v4.h`) are now single precision, with one source for the
  CPU writer and the CUDA writer:
  - No FMA contraction (the pragma, and nvcc -fmad=false for that unit only).
  - Correctly rounded sqrt and division.
  - Selects for min and max.
  - Host-computed trig constants.

  The two writers agree bit for bit: `writer_bench` and the trainer's dump
  check find 0 mismatched floats in 235,000 rows.
- **Fidelity against the FP64 writer** (117,500 corpus rows, full writer):
  - No ray changes class.
  - Ray distance: max difference 0.00125, p99.99 2.9e-6. 5e-6 of rays differ
    by more than 1e-3.
  - Closing speed: max difference 1.2e-7.
  - Terrain field 0: max difference 1.56e-3, at a 1e-5 rate. Fields 10 and
    11: 1.2e-4 and 1.7e-4.
  - The lite writer's terrain is exact.

  This is a recipe change; checkpoints from the FP64 writer see slightly
  different inputs.
- **Transport.** Pack modes 2 and 3 are modes 0 and 1 without the terrain
  and rays. They carry the row's scene (walls, water, circles, body state,
  at most ~420 floats) instead, and `bopl4_rays_launch` writes the block
  after the unpack. PACK_P went from 1536 to 2048 floats, so that a row's prefixes
  (up to ~1600 with 23 entities) and its scene fit in a slot rather than
  going dense. The policy forward went from 297 to 286 us.
- **The kernel** (`train/bopl_rays_v4.cu`) is built for latency, since a
  launch carries a few hundred rows. It runs one block per row, with the scene
  in shared memory and 4 threads per ray. The fan, the dash rays and the aim
  run as concurrent warp groups.
  - 31.7 us per launch of 224 rows.
  - 78 ns per row at 235k rows (12.8 M rows/s).
- **End to end** (spec): 387k sps (v7d, CPU writer) to 440-460k. Every run
  was under varying host load. The rays run inline on the main stream. A
  side stream beside the set encoders measured no faster (0.95x), so it was
  not kept.

## Open

- **Env CPU is still the bound.** In the pool, c_step is ~45 us: sim 37,
  finish 6.4, sample 1.4. The 11 env threads are 91% busy, and the GPU is
  ~68% busy (nsys).
- **SMT gives only 1.3x.** The sim takes 19.6 us solo, 21 us as 6 copies on
  separate cores, and 32.3 us as 12 copies.
- **1M sps needs a ~2x faster sim.** It needs ~24 us per c_step on 12
  threads.
- **The sim is front-end bound.** Solo it retires 50%; front-end stalls are
  31% and back-end 17%. The op cache misses 40% and the I-cache 52%. Under
  load, throughput tracks instructions retired: 305k per decision.
- **The instruction profile** (30 games):
  - DetPhysics step: 23%.
  - player_tick: 9%.
  - run_spawned_before, inclusive: 13%.
  - platform_turns: 7%.
  - circle_circle_group: 6%.
  - circle_box_group: 6%.
  - The Updater scheduler's helpers together: 9.6% (platform_updates_in,
    updater_key, updater_owner_position, platform_pass, platform_turn_key,
    updater_pass_batch).
