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
21. **v8: eleven small exact cuts to the sim's instructions.** Each was
    found in a callgrind or inline-aware perf profile, and each was checked
    by its instruction count on the 30 pgo-train games. Together they take
    the count from 14.49G to 12.41G (-14.4%):
    - `Fix::checked_rem` skips the software fmod when |a| < 2|m|. A test
      checks it bit for bit against `%`.
    - `platform_pass` is inlined, and the Duplicator rope check moved out
      of line.
    - `platform_turns` finds a platform's first authored platform through a
      per-pass map. Each hit is validated, and a miss falls back to the
      linear find.
    - Wall raycasts purge destroyed colliders once per run, not once per
      cast.
    - `platform_pairs` and the group rank sorts sort (key, index) pairs
      unstably, not indices through a comparator.
    - `updater_version` bumps only when Updater's list actually changes.
    - `push_blasts` uses plain loops, not a chain of `flat_map`s.
    - The Circle/Circle and Box/Box pair loops drop priors that end before
      the current part starts on the sweep axis. The ranks rise along that
      axis, so those pairs fail `swept_meet` anyway, and the order is
      unchanged.
    - `wall_group` reads a wall's own index, not a linear
      `platform_entities` search per wall.
    - `take_sweep` skips bounds for pool-owned rows. Only smoke-cloud and
      blast rows are read back, which a debug_assert checks.
    - `remove_destroyed_updatables` skips its retain over Updater's whole
      list unless `destroy_fix` flagged an entry since the last one.
    With 12 concurrent env_bench copies (sims/v8-pgo.so, test/100 x 2
    rounds), sim us per decision goes from 27.9-32.1 to 25.0-26.7 (-10% to
    -17%; other sessions' load made the two runs differ). The hash is
    unchanged, and the exact gate passes with a debug-assertions candidate.
22. **The writer: overflow telemetry folds into the encode.** With the
    delay, `bopl_step_sample` ran `bopl_count_overflow` on every sampled
    state, which repeats the encoder's visibility and nearest-entity pass,
    only for telemetry. Each delayed state is now counted as it is encoded,
    and at an episode's end `bopl_count_unpublished` counts the samples the
    delay never publishes, so every sampled state is still counted once.
    The one difference: the encoder counts a forced hidden controller's
    slots toward `visible_slots`, which only moves the overflow log. In the
    trainer, sample goes from 1.3 to 0.2 us and finish rises 0.3 us, so
    c_step drops 40.3 to 39.75 us (2 interleaved repeats).
    The ballistic relation also skips the sqrt for an arc sample whose
    squared distance is not below the best so far. sqrt and the radius
    subtraction are monotone, so the chosen sample is unchanged, NaN
    included. The writer gets about 0.3% faster, and the hash is unchanged.
23. **v9: platform bookkeeping and read-back-only sweep rows.** On the 30
    pgo-train games, instructions fall from 12.42G to 11.71G (-5.7%).
    - `platform_kinds` keeps each platform's `PlatformPass` next to the
      platform list, updated wherever a platform is pushed, removed or
      cleared, so `platform_pass` is a load, not a derivation. A
      debug_assert checks it against the old derivation. This saves 1%.
    - `platform_pairs` takes its sweep axis from `sweep_axes`, which
      platforms_tick has just computed from the same centers, rather than
      recomputing the mean and variance. A debug build checks that the two
      agree.
    - `take_sweep` fills only the rows `swept_bounds` reads back (smoke
      clouds, the blast boxes, Meteor smears, beam orbs and Drill
      hitboxes), not one per entity. A debug-only `taken` mask asserts
      that every row read was filled. Together with the axis reuse, this
      saves another 4.8%.
    The hash is unchanged, and the debug-assertions exact gate passes
    (12,400 games). In the trainer (3 interleaved repeats at load ~12),
    the sim time moves within noise.
24. **v10: the part filters ask the collider they know.** `collides` goes
    through `detphysics_collider`, which scans the node, rocket, spike,
    clone, pickup, arrow, item and lightning pools and then the players
    before it builds a collider, only for `is_some()`. A callgrind profile
    of v9 put `cloud_part` at ~6% of sim instructions (738k calls on the 30
    games, ~15 per tick) and `body_part_circle` at 1.8%.
    - `cloud_collides` answers a smoke cloud's circle row from the row
      itself (active, not destroyed or removed, not hidden in a Blink
      tunnel), which is what `row_collider` returns once every scan
      misses.
    - `body_part_circle` asks `hurtbox_collider` for its own slot.
    Both debug_assert that they agree with `collides`. Instructions drop
    3.497G to 3.387G (-3.1%), and cycles (non-PGO, taskset, interleaved)
    drop by about 4%. The PGO env_bench sim goes from 15.60 to 15.15 us, the
    hash is unchanged, and the debug-assertions exact gate passes.
25. **v11: an observation places each entity row once.** Found with a
    cycles profile of the trainer itself (a PGO `.so` with frame pointers
    via `--game`, perf on the live process), not the corpus: there
    `publish_entities` was 3.2% of sim cycles for 1.9% of instructions,
    half of it in the four-pass filtered fallback. A probe build counted
    the trainer's calls: 77% find the table out of shape order (a dynamic
    circle or box appended after the authored walls), none have rope rows.
    Without rope rows every path was the same stable shape partition, so
    it now counts each shape and places each row once, overwriting the
    previous observation's rows instead of refilling. Rope rows keep the
    old path. In the trainer profile the function falls to 2.57%; the PGO
    env_bench sim goes from 16.01 to 15.67 us (200 games), the hash is
    unchanged, and the debug-assertions exact gate passes.
26. **v12: the destroyed-row check is one AVX-512 gather scan.**
    `remove_destroyed_colliders` runs on every collider query and almost
    always finds nothing to drop; its `iter().any` over the 136-byte rows
    was 1.1% of trainer cycles. It now gathers the `flags` word of 16 rows
    per instruction and tests them with a masked compare (scalar fallback
    without AVX-512; a debug assertion checks it against the scalar scan).
    c30 cycles fall 1.5%, and the PGO env_bench sim goes from 15.29 to
    15.04 us. The hash is unchanged, and the exact gate passes.
27. **v13: fewer Updater key searches.** Every authored platform precedes
    the runtime ones (only `reset` pushes authored rows; removal keeps the
    order), so the runtime passes and `run_spawned_before` start their
    platform walks at the first runtime platform. Five per-object loops
    (wall and object tunnels, engines, Duplicator effects, shockwaves)
    searched for the object's Updater key before testing whether it was
    active, which it almost never is; they now test first, and the engine
    order keys only the active engines. Measured on instructions, with
    `BOPL_ENV_BENCH_SIM_ONLY=1` (a new env_bench switch that skips the
    writers, so `perf stat` counts the sim alone; instruction counts are
    deterministic, so a 1% change shows without noise): PGO sim -1.1%.
    The hash is unchanged, and the exact gate passes.
28. **v14: the active smoke clouds are indexed by Updater key.** A Smoke
    cast keeps dozens of clouds alive and each cloud's turn runs
    `run_spawned_before`, which walked every cloud to find the few keyed in
    [cursor, before): O(clouds^2) per pass, 3.1% of the trainer's sim
    instructions. The active clouds' keys are now kept sorted in scratch,
    rebuilt when Updater's list version moves or a cloud spawns or leaves
    (both spawn sites, both deactivations, reset, `fork_from`), and the
    walk binary-searches them. A debug assertion compares the index with a
    fresh one at every use. PGO sim -1.6% instructions and -0.7% cycles
    (7 interleaved rounds); the hash is unchanged, and the exact gate
    passes.
29. **v15: two binary searches in `run_spawned_before` are cheaper.** The
    first runtime platform is now a count of the authored ones, kept at
    reset, push, removal and `fork_from` (a debug assertion checks it against
    the search), and the cloud index holds each (key, slot) packed into one
    u128 that orders the same, so its search compares integers instead of
    tuples. Together they were 1.4% of the trainer's sim instructions.
    PGO sim -0.56% instructions, cycles neutral (7 rounds); the hash is
    unchanged, and the exact gate passes.


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
- BOLT on top of PGO (llvm-bolt 22: ext-tsp blocks, cdsort functions, hot
  and cold split, from an LBR profile of env_bench on the train games): it
  was exact, but cut taken branches only 26% and gave nothing. Solo, sim
  went from 14.69 to 14.63 us; with 12 copies, from 25.45 to 25.55. PGO
  already lays the code out.
- Lockstep worlds (2026-10-07). In the trainer, 43% of instruction fetches
  miss the I-cache (38 per 1k instructions) and 40% miss the op cache, while
  L3 fills are only 5 per 1k. The idea was that a tick reruns a wide code
  path once per world, so stepping several worlds phase by phase would reuse
  that code while it is still hot.
  - **How it worked.** `game_tick` became an async state machine that yielded
    between its 71 top-level phases. The pool claimed 4 envs at once, and
    `bopl_rl_step_sampled_many` polled their worlds round-robin. It was exact.
  - **The probe** (21 or 64 worlds per thread):
    - Solo, I-cache misses halved but L1D refills rose 75%, so it was neutral.
    - With 12 copies it was 7% faster.
  - **The trainer** (spec, A/B): the async state machine alone made the sim
    6-8% slower (32.7 to 34.7 us, and 33.7 to 36.6 us). Lockstep groups of 4
    cost another 6% on top (35.4 to 37.7 us). Coarser claims also lengthened
    eval_gpu, from 0.334 to 0.400 ms.
  - **Net:** 500k sps fell to 445k.
  - **Lesson:** solo env_bench and the probe both missed the async cost. Only
    the trainer A/B showed it.
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
  Re-measured on v8 with PGO and 12 busy copies (2026-10-07): `s` neutral
  (28.19 -> 28.09 us), `2` 3% slower, `-C inline-threshold=100` neutral.
  Shrinking code does not buy back the I-cache misses.
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
- Huge pages (2026-10-07). THP=always alone does nothing on this box:
  free memory has no 2 MB blocks (page cache fills the rest), and
  defrag=madvise never compacts for unadvised faults, so every fault falls
  back. After drop_caches, compact_memory and defrag=always, the trainer's
  cand arm got real huge pages (4-5k THP faults, 0 fallbacks), yet the
  pairs were 513.9k vs 519.0k and 526.1k vs 529.3k sps, within the ~2.5%
  run-to-run noise (an earlier A/A run, with no huge pages in either arm,
  differed by 2.5%). The TLB counts agree: in env_bench with 12 copies,
  L1-iTLB misses that hit the L2 TLB are ~21M and page walks ~0.5M per
  27G cycles, under 1%. So remapping the sim's text to huge pages is not
  worth building either.
- Lazy "any live" bits that let `run_spawned_before` skip its wall-tunnel,
  shockwave and SuperStar scans (86% of its calls list nothing; 5.8 calls
  per tick): exact, -1.1% instructions, and cycles 1.195-1.200G to
  1.208-1.219G. The skipped loads were not on the critical path.
- Answering a player's own hurtbox in `body_part_circle` without
  `detphysics_collider`'s pool scans: exact (debug assertion over all 400
  games), 0.2% faster at the time. It was kept in v10 (item 24), where it
  is -1% instructions next to the smoke-cloud fast path.
- A per-tick cache of `group_part` results in `SweepBounds` (2026-10-07,
  on v10): +3.9% instructions. The linear find over the cache costs more
  than the part lookups it replaces.
- Hot/cold splitting the world's data (2026-10-07). IBS on the trainer
  spreads the L2 misses over dozens of structures with no dominant one, so
  there is no field set worth repacking.
- Idle CPU in the trainer (2026-10-07, 1 Hz /proc/stat over a spec run):
  ~91% busy, background included. Scheduling has at most a few % to give.
  Per thread, the 11 env threads run 91-95% and the Python main thread
  2.3%, so the 12th hardware thread is nearly free. A 12th env thread
  (BOPL_ENV_THREADS=12, spec, 3 interleaved pairs) was neutral: +1.1%,
  -1.4%, -0.2%, with stragglers up from ~75-190 to ~110-170 us.
- Callgrind on the corpus is not the trainer. A cycles profile of the live
  trainer (v11 item) tracks its own instruction profile closely, function
  by function (cycles/instructions 0.8-1.3, front end and all), but it
  ranks lines differently from corpus callgrind: the `any()` scan in
  `remove_destroyed_colliders` is 1.1% of trainer cycles, and observation
  copies weigh more. Pick targets from the trainer profile.

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
- **The whole row on the GPU (pack mode 4, 2026-10-07).** The rest of the v4
  row (players, slimes, walls, items, the enemy pick) is now encoded on the
  device from the shipped state, by `train/bopl_rows_v4.h`, a source shared
  with the CPU writer. `eval/encode_check` compares it with the CPU rows:
  3,142,224 rows, 0 mismatched, and the writer hashes are unchanged. A quiet
  A/B on v15 put trainer sps at 552k to 601k (+7.3%; +10.8% on v10). An
  earlier "neutral" reading was contaminated by concurrent benches.

## Open

- **Env CPU is still the bound.** At v8 (~500k sps) c_step is ~40-43 us in
  the pool: sim 31-35, finish 6.1, sample 1.3. ~10.3 of 12 hardware threads
  are stepping envs, so 1M needs c_step ~20 us. In the trainer the encoder
  (finish + sample) is ~15% of cycles, spread over memsets and small double
  ops with no single hotspot.
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
