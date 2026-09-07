# Throughput autoresearch notes

Rift `throughput-autoresearch`, branch `autoresearch/throughput`, starting commit 5931f7c.
Results table: `results.tsv` (rows appended by `record.py` from each benchmark's `summary.json`).
Raw runs: `native/artifacts/rl/autoresearch/<dir>` (gitignored).

## Workloads

### w16-393216-r3 (original, startup-heavy)
`bench_throughput.py --workers 16 --repeats 3 --decisions 393216`: map 0, fresh policy, pool grows
from one version, 96 updates. Wall is ~118 s of which ~41 s is serial game spawning
(16 games x ~2.2 s handshake each). Kept for comparability with the first row.

### sustained-w16-786432-r3 (primary from 2026-09-07)
Resume `dap-multimap-001-s73` (10M-step multimap run, 16 workers, maps 0,1,2) at all-player
step 5017600 with its four most recent historical snapshots
(`sustained-w16-continuation.json`, schema-1 continuation context), maps 0,1,2,
evaluation maps 0,1,2, held-out 3,4, no eval/video, 786432 decisions = 192 updates.
Why: it is the shape of real training. Full 4-version pool -> 8 historical steppers
from the first update; the mid-training policy plays longer matches (episode length
~440-490 ticks vs ~390 from scratch) so reset/spawn costs occur at the real rate; the 100k
snapshot cadence recurs (~1 checkpoint + frozen copy + graph capture per 100k
decisions, ~2 per run); three maps exercise the different entity/slime observation
loads. Startup (~42 s) is ~20% of wall instead of ~35%.
`bench_throughput.py --start-step` makes decision counts relative to the loaded step
and checks the first metrics record actually starts there.
Primary score: median `learner_dps_active` (learner-consumed decisions per second of
train.py's own clock). `learner_dps_wall` (includes startup) is recorded alongside.

## Noise
w16-393216-r3 baseline, 3 runs: wall 2466.52-2516.03 (2.0% spread), active 3923.69-3994.89 (1.8%).
sustained-w16-786432-r3 baseline, 3 runs: active 3578.20-3868.54 (7.8% spread, median 3717.7),
wall 2821.05-3000.65 (median 2901.2). The sustained workload is noisier than the
short one; run 2 was slowest and overlapped a 10-15 s /proc sampling script of mine
(light, but noted). Treat single-run differences under ~8% as noise on this workload;
confirm candidates with alternating incumbent/candidate runs.
Sustained active is *lower* than from-scratch (3718 vs 3965): the full pool runs
8 historical steppers from update 1 and matches are longer, so it is the harder,
more honest target.

## Baseline profile (5931f7c, 16 workers, 2 pipeline groups, CUDA graphs)
Per update: rollout ~0.73-0.78 s, PPO ~0.04 s; collection ~94% of loop time.
Each group launch (one 4-tick decision for 8 games) ~3.1-3.2 ms env time, i.e. the
games themselves dominate; GPU util ~15%, ~6.3 busy cores of 12.
Startup 41 s = `my_vec_init` opening games serially.

## Experiment log
(see results.tsv for numbers; each entry: hypothesis, change, workload, outcome)

### exp01 — slim opponent sampling (commit 82a3e43) — INVALID (interrupted)
- Hypothesis: py-spy showed ~55% of main-thread time inside torch `Categorical` construction/sampling for the historical-opponent steppers. Replace `distribution(logits).sample()` with `policy.sample(logits)` (logsumexp-normalise, softmax, `torch.multinomial`), which replicates Categorical's kernels bit-identically, so opponent draws are unchanged for the same RNG state.
- Workload: `sustained-w16-786432-r3`, arms interleaved base (`BOPL_OPPONENT_SAMPLING=distribution`) / slim.
- Outcome: 2 of 6 runs completed (base-w16-1: 3698 active / 2866 wall; slim-w16-1: 4047 / 3079). base-w16-2 was killed after 150/192 updates when systemd-oomd killed peace.service (host cgroup of the agent shell + benchmark) at 2026-09-07T04:53:35Z (Avg10 pressure 89.5, 12.2G usage, MemoryHigh 12G). Single pair suggests +9.5% active / +7.4% wall, inside the ~8% noise band and possibly distorted by memory pressure inside the cgroup. Recorded as `invalid`; rerun as exp01b from a separate systemd unit (`bopl-throughput-fable.service`).

### exp01b — slim opponent sampling, rerun in the new unit — KEEP
- Same code (82a3e43 + record-keeping commit 56406e4), same arms, interleaved base/slim × 3 in `bopl-throughput-fable.service` (no oomd events).
- base (distribution): active 3802/3848/3850, wall 2962/3004/2979 → median 3848 / 2979. Recorded as `exp01b-control-distribution` (status baseline) because it is the incumbent measured under the new conditions: +3.5% active over the 04:20Z sustained baseline that ran inside the pressured peace.service cgroup, so the two are not directly comparable and the base arm is the control for this and following experiments.
- slim: active 4040/4047/4048, wall 3115/3128/3125 → median 4047 / 3125. **+5.2% active, +4.9% wall**, every slim run above every base run; the slim arm's run-to-run spread is 0.2%, the base arm's 1.3%.
- Lesson: the earlier ~8% noise estimate came from runs in the pressured cgroup; in this unit interleaved triples resolve ~2% differences. Sampling on the critical path is worth removing wholesale: exp02 moves it into the historical steppers' CUDA graphs.
- Record caveat: base-w16-3's `source/` snapshot (copied ~05:29:57Z) contains the exp02 edits to inference.py/selfplay.py/train.py, which I made in the working tree while the last runs were still queued and stashed about a minute later. The run itself behaved exactly like base-w16-1/2 (192 updates, 3802 active / 2979 wall): its train.py had been imported before the edit (the edited train.py at that moment would have raised NameError on `OPPONENT_SAMPLING`), and under `BOPL_OPPONENT_SAMPLING=distribution` the selfplay edits are behaviourally identical to 82a3e43. slim-w16-3's snapshot matches HEAD. Lesson: never edit the tree while a benchmark still has runs to launch; edit in a second rift instead.

### exp02 — in-graph opponent sampling (commit 9feb7c1) — KEEP
- Change: historical steppers are captured with `Stepper(sample='actions')`, so the forward pass *and* `policy.sample` (logsumexp normalise, softmax, one-sample multinomial) replay as one CUDA graph on the padded batch; `Opponents.act` slices the real rows. `BOPL_OPPONENT_SAMPLING=graph|slim|distribution` selects the path (default graph).
- Why it is exact: torch's one-sample multinomial draws element i from Philox subsequence i and advances the generator by a fixed 4 counters regardless of batch size, so the padded rows change neither the real rows' draws nor the generator offset. The padded *forward* is not bitwise equal to an unpadded forward (cuBLAS batch dependence), but train.py already ran the padded forward before this change, so the equivalence claim is "padded forward graph + eager sample of the real rows". `check_padded_sampling.py`: 50 cases (capacities 4/8/16/64, various k, 6 steps) identical actions, state and generator state; `check_pipeline.py` phase F: g2-graphs-slim and g2-graphs-distribution dump-identical to g2-graphs (`native/artifacts/rl/autoresearch/exp02-check-pipeline/result.json`).
- Benchmark (`exp02-graph-sampling/`, interleaved slim/graph × 3, same unit): slim active 4074/4104/4094, wall 3142/3168/3151 → median 4094 / 3151 (reproduces exp01b's slim arm within 1%); graph active 4457/4523/4527, wall 3362/3402/3405 → median 4523 / 3402. **+10.5% active, +8.0% wall**; every graph run above every slim run. Cumulative over the new-unit control: 4523/3848 = +17.5% active, 3402/2979 = +14.2% wall. Busy cores rose 6.6 → 7.3 (the games are fed faster); GPU util unchanged at ~17%.
- Per-update rollout time fell from ~0.72 s to ~0.65 s at unchanged PPO time (~0.04 s); startup is still 41 s of every run (24% of wall at this workload).
- What I learned: the main thread's Python per (t,group) is the lever, and each ~100 µs removed per group step is worth ~4–5% at 16 workers. The remaining Python in the loop is bookkeeping (game_decisions counting, script_workers scan, per-step tensor writes, `collect`), then the two `wait_group` stalls.


### exp03 — spare game process per env, asynchronous reset (commit 1a8468b) — KEEP
- Change: each env owns a second game process (`spare`). At episode start the spare is sent the *next* episode's Reset (`bopl_rl_reset_begin`, send only) and loads its scene while the current episode plays. On terminal, `c_reset` finishes the spare's pending reset (`bopl_rl_reset_finish`), swaps `world`/`spare`, and begins the following episode on the freed process. Configs are drawn one episode ahead from the same per-env xorshift stream, so the sequence of seed/map/loadout per episode is unchanged. `bopl_rl_reset` is now `begin`+`finish`; `bopl_rl_close` drains a pending reset. Gated by `BOPL_SPARE_GAME` (non-empty and ≠"0"), ignored under `no_autoreset` (eval binding). Spares open on a pthread during init; log `worker-<pid>-<rng>-spare.log`.
- Why it is exact: a process's reset outcome does not depend on its history (fresh scene load, seeded), so it does not matter which of the two processes plays an episode; only the config sequence matters, and that is unchanged. Evidence: `check_spare_game.py` (`exp03-spare-check/transport-w8`, `transport-w8-g2`): same random action plan with spare off/on, 64 matches, obs/rewards/dones/telemetry/log identical, 2×workers games while open and 0 after close. `check_pipeline.py` with `BOPL_SPARE_GAME=1` (`exp03-check-pipeline/`): all phases pass and every dump/metrics file is identical to the exp02 spare-off run.
- First attempt (`exp03-spare-game-throttled/`, recorded invalid): the spare arm pushed the unit to 12.9 GB > MemoryHigh 12 GB and took 7331 `memory.high` throttle events (unit anon 11.6 GB: 32 games × ~265 MB anon + trainer ~2.3 GB, plus page cache counting toward memory.current). Even throttled it ran 5394 active vs 4499. Limits raised at runtime (`systemctl --user set-property --runtime bopl-throughput-fable.service MemoryHigh=15G MemoryMax=18G`, not persistent) and rerun.
- Benchmark (`exp03-spare-game/`, interleaved base/spare × 3, no new throttle events, unit peak 13.8 GB): base learner active 4519 median (per-run all-agent dps 6046/6026/6021), wall 3391; spare learner active 5393 median (all-agent 7191/7137/7218), wall 3803. **+19.3% active, +12.1% wall**; every spare run above every base run. Rollout 0.605 → 0.511 s/update at unchanged PPO time (0.038 s). Busy cores 7.2 → 8.9. Cumulative over the new-unit control: 5393/3848 = +40.1% active, 3803/2979 = +27.7% wall.
- What I learned: the reset stall was worth ~16% of rollout time, not the ~23% the serial estimate suggested, because under load a reset (~35 ms isolated) overlaps other envs' steps in the OpenMP barrier only partially. Memory is now the binding resource for adding processes: ~310 MB RSS per game, so 16 workers + spares ≈ 12 GB anon. Contention probe during a base run (17 games): games run 6.1 cores with run-queue wait 1.1× run time (schedstat), main threads 4.8 cores + ~1.7 cores-eq waiting; Unity "Thread Pool Worker" threads (10/game) burn 0.55 cores and switch 66k/s (Mono thread pool spinning), Job.Workers ~0.04 cores each. The trainer process uses only ~0.5 cores. Startup is 42.5 s of a 155 s run (27% of wall at this workload).
- Follow-ups: make the spare the default once the unit's memory limit is raised persistently; cut startup (serial game opening in `my_init`); reduce per-game idle CPU (Mono thread-pool spinning, 11 Job.Worker threads per process on a 6-core machine).

### exp04 — serial rope loops in the game (commit 02d8f24) — KEEP; job-worker-count=1 — DISCARD
- Hypothesis (from the exp03 contention probe): each game's ten Mono "Thread Pool Worker" threads burned 0.55 cores and 66k context switches/s across 17 loaded games. Source: `DetPhysics.SimulateRopes_parallel` and `UpdateRopeMesh_parallell` ran `Parallel.ForEach` over all 128 `RopeBody` slots every simulation tick, although almost all are disabled, so every tick paid a thread-pool fan-out/join. Second candidate: Unity spawns 11 `Job.Worker` threads per process (~0.1 cores/game); `job-worker-count=1` in the built `boot.config` caps them.
- Change: plain `for` loops in both methods (`SimulateRope(body, null)` writes only its own body and reads shared read-only fields, so the serial order is bit-identical to the parallel one). Requires a game rebuild (`Assembly-CSharp.dll` sha256 30429617…; stock 72cb1e03…). `environment.configure` now uses `setdefault` for `BOPL_RL_GAME` so bench arms and checkers can point at build snapshots under `native/artifacts/rl/autoresearch/builds/` (gitignored: `game-1a8468b` stock, `game-serialrope`, `game-serialrope-jobs1`).
- Exactness: `verify_rope.py --cases 2000` on the rebuilt game (Unity rope oracle byte-identical to the pre-change corpus, 15808 actions, 0 Rust mismatches); new `check_game_variant.py` (stock vs serial, stock vs serialjobs1, w8 × 600 decisions: obs/rewards/dones/telemetry/log identical, 64 matches each, 0 leftover processes; `exp04-check-variant-w8/`, `exp04-check-variant-jobs1-w8/`); `check_pipeline.py` with the spare game on the rebuilt build (`exp04-check-pipeline/`): all phases exact and all 11 rollout dumps + metrics identical to `exp03-check-pipeline/`. Isolated probe: 0.935 vs 1.041 ms/step, 47 vs 58 threads, no Thread Pool Worker threads.
- Benchmark (`exp04-serial-rope/`, interleaved stock/serial/serialjobs1 × 3, all with `BOPL_SPARE_GAME=1`): stock active 5363/5381/5314, wall 3774/3789/3755 (reproduces exp03's spare arm, 5393/3803, within 1%); serial active 6700/6737/6538, wall 4384/4424/4323 → **+24.9% active, +16.2% wall**, every serial run above every stock run. Rollout 0.524 → 0.411 s/update; busy cores 8.9 → 8.2 while doing more work; GPU util 19.5 → 24.4%. serialjobs1: active 6784/6953/6495, wall 4436/4503/4314: +1.3%/+1.2% on medians but overlapping (its run 3 is below serial's run 3), busy cores 7.7. Discarded as a throughput change; the freed ~0.5 core might pay at higher worker counts.
- Cumulative over the new-unit control (3848 active / 2979 wall): 6700/3848 = +74% active, 4384/2979 = +47% wall.
- Memory caveat: with three build snapshots' page cache, the unit touched MemoryHigh 15G (peak 15359 MiB, 994 memory.high events over the nine runs; the sampler did not attribute them per run). The stock arm's agreement with exp03 says the effect was not visible, but the limits are now 20G/24G (runtime) and `bench_throughput.py` records per-run `cgroup_memory_peak_mb` / `cgroup_memory_high_events` so future rows can prove it.
- What I learned: per-tick CPU inside the game engine is the largest lever now that the trainer side is lean; the games are ~85% of busy cores. `Client.cs:122 Task.Run` is the only other thread-pool user. Startup is 43 s of 134 s at this workload (32%): `my_init` opens the 16 world games serially (each ~2.4 s, the spare opens on a thread in parallel), so exp05 is parallel game start-up.

### exp05 — parallel game start-up (commit c8dcc5b) — KEEP
- Hypothesis: PufferLib's `vecenv.h` create loop calls `my_init` once per env in sequence, and each `bopl_rl_open` blocks ~2.4 s for Unity boot + handshake, so the 16 world games took 43 s of the 134 s run (32% of wall). The spare game (exp03) already opened on its own thread.
- Change (`bopl_puffer.h`): with `BOPL_PARALLEL_OPEN=1`, `init` starts `bopl_rl_open` for the world game on a pthread and returns; the first `c_reset` joins it (`bopl_world_ready`) and only then runs the map-catalog / ability-count checks and aborts on a failed open. `c_close` joins an unfinished open before closing. Unset/0 keeps the serial path. Nothing about ticks, seeds or observations changes.
- Exactness: `check_game_variant.py` w8 × 600 decisions serial-open vs parallel-open: obs/rewards/dones/telemetry/log identical (64 matches; Vector setup 21.75 s → 5.15 s at w8 with spare on). `check_pipeline.py --workers 8` under `BOPL_SPARE_GAME=1 BOPL_PARALLEL_OPEN=1`: A transport exact, B serial vs async exact, C alignment pass, D cuda graphs exact, F opponent sampling modes exact; all 11 dumps and 11 metrics files identical to `exp04-check-pipeline`.
- Benchmark (`exp05-parallel-open/`, interleaved serialopen/parallelopen × 3, both `BOPL_SPARE_GAME=1` on the serial-rope build): serialopen wall 5854/5830/5866 dps, active 8916/8870/8959, startup 43.0 s (reproduces exp04's serial arm: learner wall 4391 vs 4384); parallelopen wall 7698/7687/7571, active 8897/8913/8720, startup 10.6/10.9/10.4 s. Learner rates: **wall 4391 → 5765 (+31.3%)**, active 6687 → 6673 (−0.2%, noise). Every run 793 matches, 589824 learner steps, 1417 updates. Memory peak 14.8 GB, 0 memory.high events under the 20G/24G limits.
- Cumulative over the new-unit control (3848 active / 2979 wall): active +73%, **wall +94%** (2979 → 5765).
- What remains of startup (samples of parallelopen-w16-1): ~2 s of Python/torch import before the games start, ~6 s with all 12 cores saturated while 32 Unity processes boot together, then ~2 s of checkpoint load and CUDA-graph capture. Booting is now CPU-bound, so the next startup lever would be a lighter boot (fewer scenes/assets loaded at start), not more parallelism.
- What I learned: at this workload the run is now 90% active time; further gains have to come from the per-tick cost. The saved perf profile of one game main thread (`game.perf`, 04:05 UTC) is 64% JIT-compiled managed code, 18% Mono runtime (`mono_gc_wbarrier_set_field` 3.2%, `GC_mark_from` 1.5%, GC-safe-region transitions ~2.5%), 12% UnityPlayer, 3.5% libc. `MONO_ENV_OPTIONS=--jitmap` produces no `/tmp/perf-PID.map` under Unity's embedded Mono, so symbolising the managed 64% needs an in-game phase timer instead.
- Next: instrument `Updater.TickSimulation` phases through the existing `ReferenceScheduler.Capture` / `ReferenceDeathEffects.ComponentPhase` hooks (gated by an env var, profile build only) to attribute tick time per phase and per `MonoUpdatable` type, then attack the largest.
