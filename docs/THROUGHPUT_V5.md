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
   (the proof is at `bopl4_march`). The 400-game hash is unchanged and
   writer_bench's march-check passes. Of the 12.5 walls a map has, 6.5 are
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

## Measured and not kept

- `overflow-checks = false`: 2%. The checks stay as a safety net.
- A no-hit certificate (skipping the march for a ray whose segment stays
  more than 0.05 from every wall) was not built. 58% of rays end without a
  hit, but they average 6 steps each (the sky, not grazing the floor; only
  1% hit the 32-step cap), so it nets about 10% of the march.

## Open

- End-to-end (`BOPL_BENCH_WORKLOAD=spec train/bench_shape.sh`, after the
  2026-10-06 reboot, load ~24 from other agents): 302k sps, 227k learner.
  Still env-bound: per buffer step eval_env 1.33 ms against eval_gpu
  0.29 ms. In the pool, c_step is 71.7 us (sim 43.5, finish 26.3), about
  1.9x env_bench's single-core numbers, because SMT runs 11 threads on
  6 cores and the box was loaded.
- With the full writer, the writer is now 38% of the env step.
  Full vs lite for the specialists is Matthew's call.
