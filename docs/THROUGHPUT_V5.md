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

## Measured and not kept

- `overflow-checks = false`: 2%. The checks stay as a safety net.
- A no-hit certificate (skipping the march for a ray whose segment stays
  more than 0.05 from every wall) was not built. 58% of rays end without a
  hit, but they average 6 steps each (the sky, not grazing the floor; only
  1% hit the 32-step cap), so it nets about 10% of the march.

## Open

- Measure end-to-end trainer sps once the GPU is back; it faulted at
  14:57 on 2026-10-06 and needs a reboot. The env step is now 3.4-4.2x
  cheaper, so the GPU chain and the 8-vs-11 env threads in sweeps
  (driver.py) are next.
- With the full writer, the writer is now 38% of the env step.
  Full vs lite for the specialists is Matthew's call.
