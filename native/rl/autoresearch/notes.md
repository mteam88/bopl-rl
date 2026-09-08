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

### exp06 — simulation ticks per Unity frame (commit 82a7e15) — KEEP
- Instrumentation first (same commit): `RlProfile.cs` is an env-gated phase timer (`BOPL_RL_PROFILE=<dir>` writes `profile-<pid>.json` per game on close; off, it costs one static bool test per call site). It hooks the existing `ReferenceScheduler.Capture` / `ReferenceDeathEffects.ComponentPhase` seams for per-`MonoUpdatable`-type time, inserts markers into every Unity player-loop subsystem (`loop:<Parent>.<Child>`), and brackets the bridge phases (`bridge:read/command/reset/send/observe`, `tick:*`, `unity:frame`). Profiling on is exact against profiling off (`exp06-profile-check`, w16 × 2000, 416 matches); the profile build with profiling off is what every later arm uses (`builds/game-profile`, identical to `Builds/Linux`).
- Training profile (`exp06-profiles-train`, real `train.py` run, 32 games, 1.77 M ticks): 79.9% of game main-thread time is `bridge:read` (waiting for the trainer's next batch of actions — the games are idle most of the time because the trainer steps all 16 envs in lock step and the slowest game gates each step). Of the non-wait time: SlimeController.UpdateSim 141.6 s (0.080 ms per call), PlayerPhysics 92.0 s (0.052), `tick:physics` (DetPhysics) 89.9 s, observe 25.4 s, send 24.0 s, AnimateVelocity 24.1 s (13 M calls), ResizablePlatform 21.8 s (3.15 M calls — resizing every tick), BowTransform 20.0 s, `EarlyUpdate.UpdatePreloading` 18.1 s, SpriteAnimator 11.0 s.
- Player-loop profile (`exp06-profiles-loop16`, stock 16 ticks per frame): a Unity frame with nothing to render costs ~3.3 ms — `UpdatePreloading` 2.3 ms (asset/scene bookkeeping), `ParticleSystemBeginUpdateAll` 0.33, `UpdateAllRenderers` 0.30, `PlayerUpdateCanvases` 0.14, `UpdateRectTransform` 0.10 — i.e. 0.22 ms per tick at 16 ticks per frame, 12% of the game's active time. Gameplay is frame-independent lockstep (`Updater.TickSimulation` under `RlBridge.Pump`), so the frame is pure overhead.
- Change (`RlBridge.cs`): `CommandsPerFrame` is now `BOPL_RL_TICKS_PER_FRAME` (16..65536, default 16 = stock). At 4096 the bridge runs up to 4096 ticks per `Host.Update` before yielding one frame. Unity's deferred work (object destruction, canvas/renderer bookkeeping, `Time.frameCount`) runs 256× less often; nothing on the simulation path reads frame timing.
- Exactness: `check_game_variant.py` stock vs 256 and vs 4096, w16 × 2000 decisions (416 matches each, `exp06-frame-check`, `exp06-frame4096-check`); long check w16 × 6000 decisions, 2 pipeline groups, 236 matches, all metrics identical (`exp06-check-variant-long`). ms/tick in the check runs: 1.816 (16) → 1.592 (256) → 1.574 (4096); `unity:frame` 0.218 → 0.151 → 0.141 ms/tick (15296 → 1744 → 896 frames per 235520 ticks); `bridge:send` 0.023 → 0.017 (fewer socket writes straddling frames).
- Benchmark (`exp06-ticks-per-frame/`, interleaved frame16/frame4096 × 3, both spare + parallel open on the profile build): frame16 wall 5656/5789/5795, active 6542/6700/6690 (reproduces exp05: 5765/6673); frame4096 wall 6069/6098/6024, active 7077/7093/6992. Learner rates: **wall 5789 → 6069 (+4.8%)**, **active 6690 → 7077 (+5.8%)**; every candidate run above every control run (noise ≈2%). Busy cores 8.2 → 7.5, GPU util 24.4 → 25.9%, memory peak 16.1 → 14.9 GB, 0 memory.high events. Wall 97.2 s of which start-up 10.4–10.7 s.
- Cumulative over the new-unit control (3848 active / 2979 wall): **active +84% (7077)**, **wall +104% (6069)**.
- What I learned: the trainer's main thread is ~66% in `wait_group` (py-spy), and the game is 80% waiting on the trainer — both sides idle because each step is gated by the slowest of 16 games plus the GPU round trip. Per-tick game work is still the lever (a faster tick shortens the slowest game's turn), and the profile now names it: SlimeController + PlayerPhysics + DetPhysics ≈ 70% of managed tick time. `MONO_ENV_OPTIONS=--jitmap` is ignored by Unity's embedded Mono, so this in-game timer is the only managed attribution available.
- Next (exp07): remove per-tick work that has no effect on the simulation in headless RL mode, each gated by an env var and checked bit-exact — ResizablePlatform.UpdateSim resizes (sprite scale, material `_Scale`, child components, two Fix divisions) every tick even when extents are unchanged; SlimeController.OldUpdate sets `_PercentDone` on every ability indicator material every tick; `Debug.DrawLine/DrawRay` in `checkForWall`, AnimateVelocity and DetPhysics (39 sites) each marshal into UnityPlayer; `RemoveDestroyedColliders()` runs on every raycast (3 per player per tick). Then the arithmetic itself: `Fix.operator/` and `Fix.Sqrt` are bit-serial 64-iteration loops that can be replaced by exact 128-bit versions.

### exp07 — headless render stripping (commit 1804c01) — KEEP
- Change: new `RlHeadless.Render` static (`BOPL_RL_HEADLESS=1` → false; default true = stock). Gated: all 117 `Debug.DrawLine/DrawRay` sites in 28 gameplay files (each is a standalone statement with pure-read arguments — a sed prefix `if (RlHeadless.Render)`; heaviest are the two per `checkForWall` raycast, three per `AnimateVelocity` platform tick, and the DetPhysics collision-pair lines); `ResizablePlatform.ResizePlatform` material `_Scale/_BevelRadius/_RHeight/_RWidth` SetFloat (×8 per platform tick) and the shakable-platform copy; `AbilityReadyIndicator.SetCooldownProgress` (`_PercentDone` per ability per player tick); the SlimeController.OldUpdate indicator tint loop. Flag-independent exact change: `LayerMask.NameToLayer("effect")` / `"Player"` cached in DetPhysics (was a string lookup per Simulate call and per CircleCastClosestPlayers).
- Exactness (all against the exp06 build, same spare/parallel/4096 flags): `check_game_variant` w16 × 2000, 2 groups → 416 matches (`exp07-check-variant`); w16 × 8000 → 1696 matches, obs/rewards/dones/telemetry identical (`exp07-check-variant-long`); `check_pipeline --workers 8` phases A–F all exact (`exp07-check-pipeline`) and every dump + metric identical to `exp05-check-pipeline` (11/11).
- Benchmark (`exp07-headless/`, interleaved incumbent/headless × 3): incumbent wall 6054/6037/6060, active 7091/7108/7064 (reproduces exp06 6069/7077); headless wall 6331/6280/6358, active 7488/7480/7411. Learner rates: **wall 6069 → 6331 (+4.6%)**, **active 7077 → 7488 (+5.6%)**; every headless run above every incumbent run (noise ≈2%). Per-pipeline-group env time per launch 2.24 → 2.08 ms (−7%); busy cores 7.4 → 7.2, GPU util 26.3 → 27.6%, memory peak 14.7 → 14.0 GB, 0 memory.high events; wall 93.2 s of which start-up 10.4–11.9 s.
- Cumulative over the new-unit control (3848 active / 2979 wall): **active +95% (7488)**, **wall +113% (6331)**.
- What I learned: the render-only calls are cheap individually but each one marshals into the native player (DrawLine builds a gizmo line; Material.SetFloat hashes the property name and touches the material), and the sim issues hundreds per tick — removing them cut per-tick game time by ~7% with zero semantic risk, because none of them can feed back into Fix state. The remaining game time is genuinely simulation: SlimeController/PlayerPhysics/DetPhysics with Fix arithmetic underneath.
- Next (exp08): the semantics-preserving-but-not-trivially-so removals — `ResizablePlatform.UpdateSim` recomputing InverseMass/MOI/extents/child positions every tick even when the extents haven't changed (skip when unchanged; must verify the child RepositionOnParent path has no other input), and `RemoveDestroyedColliders()` on every raycast (6+ per grounded player tick) hoisted to once per tick — each behind its own env var and checked bit-exact. In parallel, sub-bucket profiling in SlimeController.OldUpdate / PlayerPhysics.Move / DetPhysics.Simulate to see whether Fix.operator/ and Fix.Sqrt (bit-serial 64-iteration loops) are worth an exact 128-bit rewrite.

### exp08 — exact fast Fix arithmetic + rounded-rect raycast culling (commit 9cb0fb2) — KEEP both
- Profiling first (build `game-exp08prof` = d1a4239, instrumentation only, `exp08-profile-train/` + `exp08-profiles-train/`, not a score): sub-buckets inside SlimeController/PlayerPhysics/DetPhysics plus counters. Per tick: 1.554 ms, of which `bridge:read` (waiting on the trainer) 82%; the largest managed bucket is `det:raycast_rr` at 7.7% — ~8 raycasts per tick each scanning ~38 rounded rects (37.5 active) with a Fix-heavy closest-point test; `Fix.operator/` 448 calls/tick and `Fix.Sqrt` 45/tick (both bit-serial 64-iteration loops); `det:narrow_phase` 0.8%, `det:integrate` 0.45%, ResizablePlatform update 0.49%, `det:raycast_remove_destroyed` 0.13%; every `slime:*`/`pp:*` sub-bucket ≤0.1%.
- Change A (`Fix.cs`): `operator/` = 64/64 long division on the raw values (high word `n/d`, low 33 bits by `(r<<33)/d` when `d<2^31`, else a double-seeded quotient corrected by a signed residual loop), then the original `(q+1)>>1` rounding and saturation; `Sqrt` = `Math.Sqrt` seed on the 96-bit scaled value corrected with 128-bit compares, then the original `rem > r` rounding. Inputs with |raw| ≥ 2^60 (div) / 2^61 (sqrt) fall back to the untouched originals (`SlowDiv`/`SlowSqrt`), because the original sqrt's overflow branch rounds non-ideally and the residual must stay within 2^62. No env gate: it is the same function.
- Change B (`DetPhysics.RaycastToClosestRoundedRect`, `BOPL_RL_RAYCULL=1`, default off): compute the ray segment's AABB (origin → origin+dir·distance, ±1 unit) once, and skip any active rect whose centre ± (|right.x|+|up.x|+|radius|, |right.y|+|up.y|+|radius|) box cannot overlap it. The bound is a superset of the rounded rect, so a culled rect cannot be hit by the segment and cannot be the closest hit; the loop's other side effects are unchanged (`RemoveDestroyedColliders` still runs, inactive rects still skipped in the same order). Disabled when `distance ≥ 1e6` to avoid overflow in the bounds arithmetic.
- Exactness: `native/rl/autoresearch/fixtest/` (dotnet 8, verbatim originals vs candidates on raw longs): 224,223,723 cases, 0 mismatches — specials, saturation borderline n≈d·2^31±4, the 2^60/2^61 fallback edges, near-perfect squares, uniform/small/huge sampling (README records the counts; div 21 → 6.5 ns, sqrt 106 → 10 ns in isolation). In-game, `check_game_variant` w16 × 2000, 2 groups, exp07 build vs exp08 build: 416 matches, obs/rewards/dones/telemetry/log identical (`exp08-fixmath-check`); same vs exp08 build + RAYCULL: 416 matches identical (`exp08-raycull-check`). Cull profile (`exp08-profiles-raycull`): 36.0 of 37.5 active rects per raycast culled, `det:raycast_rr` 0.0735 → 0.0125 ms/tick, fix_div 268 → 133 per tick, fix_sqrt 45 → 13, ms/tick 1.376 → 1.118 in the check workload.
- Long confirmation (after the benchmark): `check_game_variant` w16 × 8000 decisions, 2 groups, exp07 build vs exp08 build + RAYCULL → 1696 matches, obs/rewards/dones/telemetry/log identical, step time 24.0 → 17.0 s (`exp08-raycull-check-long`); `check_pipeline --workers 8` under the exp08 build + RAYCULL: phases A–F all exact (`exp08-check-pipeline`), and all 11 rollout dumps + 11 metrics files identical to `exp07-check-pipeline`.
- Benchmark (`exp08-fixmath-raycull/`, interleaved incumbent/fixmath/raycull × 3, incumbent = exp07 build with headless+spare+parallel+4096): incumbent wall 6146/5897/6355, active 7204/6997/7478 (reproduces exp07 6331/7488 within noise); fixmath wall 6287/6458/6505, active 7427/7661/7643; raycull wall 7907/8081/8104, active 9750/10024/10048. Learner rates: fixmath alone **wall 6146 → 6458 (+5.1%)**, **active 7204 → 7643 (+6.1%)**; fixmath + raycull **wall 6146 → 8081 (+31%)**, **active 7204 → 10024 (+39%)**; every raycull run above every fixmath run above every incumbent run. Per-group env time per launch 2.09 → 2.00 → 1.24 ms; wall 96.0 → 91.3 → 73.0 s (start-up unchanged 10.3–11.9 s); busy cores 7.2 → 7.2 → 6.4; GPU util 27 → 28 → 36%; memory peak ≤14.7 GB, 0 memory.high events.
- Cumulative over the new-unit control (3848 active / 2979 wall): **active +160% (10024)**, **wall +171% (8081)**.
- What I learned: the per-tick profile was right that the raycast scan dominated, but it understated it — removing 96% of the rect tests took 40% off env time per launch, far more than the bucket's 7.7% share, because the bucket share was measured on the profile build (whose timers themselves inflate the tick) and because a shorter tick shortens the *slowest* of 16 games, which is what gates the pipelined step. Fast Fix arithmetic alone was worth ~5–6%; after the cull removes most of the divisions it still carries its share of the remaining tick. Exactness of the arithmetic rewrite was only cheap to establish because the original functions are pure on raw longs — the 224M-case harness took minutes, while the in-game checks only cover the values a match happens to produce.
- Next (exp09): with rounded-rect raycasts collapsed, the remaining per-tick managed work is spread thin (integrate, bounding boxes, sync, narrow phase, beams ≈ 0.4–0.5% each in the check profile) and the trainer wait (`bridge:read`) is >75% of the game's time — so the next lever is the trainer's own per-decision latency: the main thread spends its time in `wait_group`, and each group step is gated by the slowest game plus a D2H/H2D round trip and the recurrent forward. Candidates: more pipeline groups (4) so a slow game blocks a quarter of the batch, `GameTime.FixedDeltaTime`'s two linear `GetPlayer` scans per call (50 sites), and skipping ResizablePlatform's per-tick recompute when extents are unchanged. Default-on for the kept flags (spare/parallel/4096/headless/raycull) once the long confirmation of raycull (`exp08-raycull-check-long`, `exp08-check-pipeline`) is in.

### exp09 — socket transport coalescing, game and trainer side (commit 25ab8cc) — KEEP both
- Profiling first (no scores; `exp09-strace/`, `exp09-threads2/`, `exp09-pyspy/`): per step the game's main thread did 5 `recvfrom` + 5 `sendto` (unbuffered `BinaryReader`/`BinaryWriter`: four 4-byte header fields and the ~6.2 KB body each as their own syscall) and the trainer did 4 `poll` + 2 `sendto` + ≥2 `recvfrom` (`transfer` polled before every chunk). One game's main thread per step: mean 0.82 ms blocked in recv (waiting on the trainer), mean 0.84 ms compute (p50 0.59, p90 0.90, p99 3.4 ms; the 63–93 ms tails are scene loads). Game process syscalls over 10 s: futex 56% (Unity's own threads), restart_syscall 35%, recvfrom 9585, sendto 11468, poll 1889. Per-thread CPU over 62 s (32 games): 4.86 cores total, main threads 4.39 (0.12–0.15 each; spares equal actives because roles alternate per episode), AssetGarbageCollector 0.12, Loading.Preload 0.07, Job.Workers 0; trainer 0.95 cores; pipewire-pulse 0.03 (audio is not a lever). py-spy on the trainer main thread (active samples): `opponents.act` 28% (index_copy_ 22%, index_select 21%, graph replay 20%, static copies 15%), `wait_group` 24% (95% cond-wait), `ppo_update` 16%, `step_group` 4%, learner stepper 2%. Pipeline accounting: env 1.24 ms per launch, ~0.18 ms non-env overhead per launch per group. In-game: `EarlyUpdate.UpdatePreloading` 0.111 ms/tick (~29 ms per 4096-tick frame), `bridge:send` 0.017, observe 0.0085, reset 0.0089.
- Change A (`RlBridge.cs`, `BOPL_RL_COALESCE=1`, default off, build `game-exp09`): `Send` packs Magic/operation/sequence/length and the body into one byte array and writes it with a single `Socket.Send` loop (5 syscalls → 1); the command reader wraps the `NetworkStream` in a 64 KB `BufferedStream` (5 reads → 1 for the usual short command). The byte stream on the wire is identical.
- Change B (`bopl_rl.c`, `BOPL_RL_EAGER_IO=1`, default off): `transfer` attempts `send`/`recv` first and only `poll`s after `EAGAIN`; `send_request` writes the request frame and payload (≤4 KB) with one `send` (4 polls + 2 sends → 1 send per step; the response read is 2 recvs as before). Timeouts still apply through the deadline once a call would block.
- Exactness: `check_game_variant` w16 × 2000, 2 groups, exp08 build vs exp09 build + both flags → 416 matches, obs/rewards/dones/telemetry/log identical (`exp09-check-variant`); exp09 build without flags vs with flags → 416 matches identical (`exp09-check-variant-selfflags`); `check_pipeline --workers 8` under exp09 build + both flags: phases A–F exact (`exp09-check-pipeline`) and all 11 rollout dumps + 11 metrics files identical to `exp08-check-pipeline`. Long check after the benchmark: w16 × 8000 decisions, 2 groups, exp08 build vs exp09 build + both flags → 1696 matches, obs/rewards/dones/telemetry/log identical, step time 17.34 vs 17.33 s (`exp09-check-variant-long`).
- Benchmark (`exp09-transport/`, interleaved incumbent/coalesce/both × 3, incumbent = exp08 build + spare/parallel/4096/headless/raycull): incumbent wall 8121/8145/8183, active 10060/10017/10091 (reproduces exp08's 8081/10024); coalesce wall 8285/8204/8317, active 10266/10142/10289; both wall 8301/8259/8345, active 10280/10340/10321. Learner rates: coalesce **wall 8145 → 8285 (+1.7%)**, **active 10060 → 10266 (+2.0%)**; both **wall 8145 → 8301 (+1.9%)**, **active 10060 → 10321 (+2.6%)**; every `both` run above every incumbent run on both rates, and 8 of 9 pairwise coalesce-vs-incumbent comparisons in favour. Per-group env time per launch 1.24 → 1.19 → 1.18 ms; busy cores 6.5 unchanged; GPU util 38 → 39–40%.
- Cumulative over the new-unit control (3848 active / 2979 wall): **active +168% (10321)**, **wall +179% (8301)**.
- What I learned: the whole change is about ten fewer syscalls per decision on each of the 32 game main threads plus four fewer on the trainer, and it bought ~2%: the per-step floor is the sum of one game tick's compute (~0.6–0.8 ms) and the trainer's own per-group turnaround (~0.2 ms + inference), and small syscalls were only ~2% of that. Audio and Unity job workers are already idle; the remaining game-side per-step cost is genuine simulation, and the remaining trainer-side cost is `Opponents.act` (five tensor launches per historical version per group step) and the cond-wait in `wait_group` (the game tick itself).
- Next (exp10): trainer main-thread trims that keep the values bit-identical — gather straight into the historical steppers' static inputs (`index_select(out=)`, halving the launches in `Opponents.act`), cache the torch stream handle in `vector.py`, and try `--pipeline-groups 4` as an explicit semantic-change workload (a slow game then gates a quarter of the batch). Then the default-on flip of all seven kept flags.

### exp10 — trainer main-thread trims: direct gather, cached stream handle (commit 9148677) — DISCARD both

Two small changes aimed at the per-group Python overhead in `Opponents.act` and `Vector.step_group/wait_group`, both env-gated and off by default:

- A `BOPL_OPPONENT_GATHER=direct`: when a historical stepper has a captured graph, gather the group's observations and recurrent state straight into the stepper's static inputs (`index_select(..., out=)`) instead of building temporaries and copying them in. Two launches instead of four per historical version per group step.
- B `BOPL_VECTOR_STREAM_CACHE=1`: resolve `torch.cuda.current_stream().cuda_stream` once per `Vector` instead of on every launch/wait (`current_stream()` builds a fresh Python object each call).

Exactness: `exp10-check-pipeline` with both flags set passed phases A–F, and its 11 rollout dumps and 11 metric files are byte-identical to `exp09-check-pipeline`.

Benchmark `exp10-gather` (workload sustained-w16-786432-r3, incumbent = game-exp09 + seven kept flags, arms interleaved, 3 repeats each):

| arm | median active | median wall | wall s |
|---|---|---|---|
| incumbent | 10314 | 8357 | 70.6 |
| gather (A) | 10356 | 8364 | 70.5 |
| both (A+B) | 10381 | 8379 | 70.4 |

Per-run active: incumbent 10054/10314/10374, gather 10356/10323/10389, both 10383/10378/10381. The incumbent's first run was a slow outlier (11.6 s startup); excluding it the arms are indistinguishable. Rollout seconds 48.3–48.6 for every non-outlier run. Both arms are within the ~2% noise band, so neither is a measured gain. The code stays in the tree, env-gated and off, since it is exact and may matter once the pool holds more historical versions (the sustained workload carries the full pool already, so that is not expected).

Lesson: the trainer's Python launch overhead per group step is no longer on the critical path; the group thread's game step (~48 s of a ~57 s loop) dominates, and the main thread mostly waits in `wait_group`. Remaining levers are the game tick itself, the worker count / group count shape, and the ~7.8 s PPO update.

Next (exp11): explicit semantic-change workload with `--pipeline-groups 4` (distinct workload name; groups=4 changes the (t,group) inference order, so it is not the same training program even though check_pipeline shows g4 exact against g4-serial). Then the default-on flip of the seven kept flags with an exactness re-check.

### exp11 — trainer GPU-path trims and PM QoS (commit d4c64b6) — KEEP the three GPU trims, DISCARD PM QoS

Profiling first (no scores; scripts under the job tmp dir, `exp11-gpu` microbench):
- Isolated graph replays on the idle GPU: learner stepper (batch 16) 0.111 ms / 75 kernels; historical stepper (capacity 4) 0.094 ms / 63 kernels; the per-group serial chain learner + 4 historical replays 0.504 ms. `torch.multinomial(probs,1,True)` inside `Categorical.sample` is 13 kernels per head (two of them `_assert_async` argument checks); the entropy computed by `distribution()` during collection is discarded (`inference.Stepper._run` throws it away).
- PPO update at the real shapes (tmax 128, n 32, sequence 32, 3 epochs, minibatch 32 segments): ≈84 ms isolated (≈40 ms in-run per `train_seconds`), 5425 kernels but only 31.6 ms of GPU time — CPU-bound: `cudnn_rnn_backward` 21 ms CPU, `cudaLaunchKernel` 20.7 ms, 217 `cudaStreamSynchronize` and 336 `aten::item` (the per-minibatch `losses.append(...item())` row plus the two `isfinite` checks). Not changed in this experiment (deferring the reads changes when a nonfinite abort fires).

Changes (all env-gated, default off, trainer only, no game rebuild):
- A `BOPL_FAST_MULTINOMIAL=1` (`policy.multinomial1`): `q=empty_like(probs).exponential_(1); div(probs,q,out=q); argmax(q,-1,keepdim=True)`. This is exactly the kernel sequence torch's `multinomial(…,1,replacement=True)` runs internally (exponential race), so the draws and the generator offset are identical — verified 200 trials × 5 shapes bit-equal — but without the two `_assert_async` checks and the wrapping copies: 13 → 3 kernels per head. Used by `distribution()` (learner) and `sample()` (historical, graph-sampled).
- B `BOPL_ROLLOUT_ENTROPY=0` (`inference.py`): the learner stepper calls `distribution(logits,entropy=False)`; actions/log-probabilities/values unchanged, the PPO update still computes entropy from its own `distribution` call.
- C `BOPL_OPPONENT_STREAMS=1` (`selfplay.Opponents.act`): each historical version's graph replay is issued on its own persistent side stream (`wait_stream` in, `wait_stream` out before the `index_copy_`), so the four historical replays overlap on the GPU instead of running serially after the learner replay. CPU issue order — and with it each generator offset a replay consumes — is unchanged. Isolated chain: 0.504 → 0.234 ms (A+B+C: 0.175 ms).
- D `BOPL_CPU_DMA_LATENCY_US=100` (`train.py`): hold a `/dev/cpu_dma_latency` PM QoS request for the run so idle cores stay out of C3 (350 µs exit latency) while 32 game threads and the trainer ping-pong. Runtime-only. (`/dev/cpu_dma_latency` was made `o+rw` with sudo; the value reverts when the fd closes.)

Exactness: `exp11-check-pipeline` with A+B+C+D set — phases A–F exact, and all 11 rollout dumps + 11 metrics files byte-identical to `exp09-check-pipeline` (`compare_pipeline_dumps.py`).

Benchmark 1 (`exp11-gpu/`, incumbent / gpu(A+B+C) / gpuqos(A+B+C+D) × 3, interleaved) was **disturbed** and is not scored: incumbent-2 and gpu-2 ran with busy cores 7.4/6.95 and load1 14–15 from outside processes, gpu-3 saw ~15 s of foreign GPU utilisation (65–87%) and a 10.4 s train phase. Medians incumbent 10027/8100, gpu 9836/7902, gpuqos 10383/8347 (active/wall); the only arm with three consistent runs was gpuqos (10383/10364/10395). Kept as evidence of what interference looks like; a `.trackerignore` now sits in `native/artifacts/` so localsearch does not index bench output while it is written.

Benchmark 2 (`exp11-gpu2/`, quiet machine, 4 arms × 4 repeats, interleaved, incumbent = game-exp09 + seven kept flags):

| arm | active (4 runs) | median active | wall (4 runs) | median wall | wall s | GPU util |
|---|---|---|---|---|---|---|
| incumbent | 10185 / 10286 / 10359 / 10305 | 10295 | 8169 / 8235 / 8316 / 8306 | 8270 | 71.3 | 36.8% |
| qos (D) | 10193 / 10317 / 10302 / 10335 | 10310 | 8281 / 8312 / 8351 / 8355 | 8331 | 70.8 | 37.1% |
| gpu (A+B+C) | 10387 / 10471 / 10491 / 10488 | **10479** | 8341 / 8419 / 8467 / 8430 | **8425** | 70.0 | 28.9% |
| gpuqos (A+B+C+D) | 10380 / 10284 / 10179 / 10462 | 10332 | 8291 / 8305 / 8197 / 8453 | 8298 | 71.1 | 28.6% |

Learner rates for A+B+C: **active 10295 → 10479 (+1.8%)**, **wall 8270 → 8425 (+1.9%)**; every gpu run above every incumbent run on both rates (worst gpu 10387 > best incumbent 10359). GPU utilisation 37 → 29% for the same work (the removed kernels and the overlap are real). PM QoS alone is inside noise (+0.1% active) and on top of the trims it was worse than the trims alone with higher busy cores (6.5 vs 6.4) — discard; the code stays gated off.

Cumulative over the new-unit control (3848 active / 2979 wall): **active +172% (10479)**, **wall +183% (8425)**.

Where the time goes now (from `pipeline` in `metrics.jsonl`; per decision step t with 2 groups, per-launch group-thread total G, env E, and the main-thread turnaround X = P − G where P = rollout seconds / (updates × 128)):

| arm | P ms/t | G ms/launch | E ms/launch | X = P − G |
|---|---|---|---|---|
| incumbent | 1.98 | 1.36 | 1.18 | 0.61 |
| gpu (A+B+C) | 1.95 | 1.19 | 1.15 | 0.75 |

The step cadence P only fell 0.03 ms although the group thread's own cycle G fell 0.17 ms: the trims shortened the group thread's wait on the *actions* copy (G includes the `cudaStreamWaitEvent` on the trainer's stream, so a faster inference chain shows up inside G), and the cycle is now the sum of one game tick (E ≈ 1.15 ms, its slowest of 8 games) plus ~0.75 ms of main-thread work per group turn. That 0.75 ms — two `wait_group` handoffs, `collect` (six small device ops plus a host read of `cpu_dones`), the obs/reset copies, the learner replay, `Opponents.act`, and the `step_group` launch — is now ~40% of the per-step critical path and is the next lever, together with the game tick.

Lesson: `Opponents.act` and the learner replay were each cheap, but they ran *serially* after each group's game step on the only stream the group thread waits on; overlapping the four historical replays and deleting the argument-check kernels was worth ~2% end to end. The GPU is far from saturated (29%); what matters is the latency of the short chain between `wait_group` returning and `step_group` launching.

Next (exp12): cut the main-thread turnaround X. Candidates, each exact: (1) capture the whole per-group inference chain — learner replay + gathers + historical replays + index_copy — as one CUDA graph per (group, pool key) so a group turn is one launch instead of ~30; (2) move `collect`'s device work off the critical path (bank rewards/dones after `step_group`, since the game only needs actions); (3) `--pipeline-groups 4` as an explicit semantic-change workload (distinct workload name). Then the default-on flip of all ten kept flags with an exactness re-check.

### exp12: one CUDA graph per group turn (`BOPL_TURN_GRAPH=1`) and a graphed collect (`BOPL_COLLECT_GRAPH=1`) — keep both (+2.2% active / +2.1% wall)

Trace first (`BOPL_TRACE_TURN=1`, commit 5387ce5, `exp12-trace`, 64 updates at the sustained workload, medians in ms per group turn): wait_group 0.436, collect 0.138, learner replay 0.071, opponents.refresh 0.007, opponents.act 0.284, stores 0.034, step_group launch 0.026. So of the ~0.75 ms main-thread turnaround, ~0.36 ms was the learner replay plus `Opponents.act` (≈30 small launches: gathers, four historical replays on side streams, `index_copy_`), and ~0.14 ms was `collect`.

Change (`turn.py`, env-gated, trainer only, no game rebuild):
- `TurnGraph` (one per pipeline group): `Opponents.refresh` now also keeps host lists of (version, rows, padded rows). The graph is keyed by the tuple of historical versions playing in the group this turn; its static inputs are three index buffers (padded gather rows, scatter source, row mask) refilled from those lists only when the group's opponent key changes. Captured body: learner `policy.step` on the vector's own observation rows + `distribution` sampling + state copy, then for each version slot `pool[version].step` on the gathered rows, `SAMPLE`, and a `torch.where` scatter into the chosen actions and the frozen state. Capture happens under saved RNG state on a side stream with `capture_error_mode='thread_local'` (the other group's C thread is issuing CUDA calls meanwhile) and everything is restored afterwards, so the replay consumes exactly the Philox offsets the eager turn would. Graphs are pruned when a version leaves the pool. Sampled draws, log-probs, values and state are bit-identical to the eager path.
- `CollectGraph`: the six-op reset/state masking in `collect` (`ended_now`, `reset_before`, `state*=keep`, `frozen*=keep`) as one graph per group.
- Smoke trace (3 updates, ms per turn): learner 0.060 → 0.026, act 0.112 → 0.017.

Crash and fix (the first benchmark `exp12-turn` aborted at 16:43:48 UTC with `Bopl group CUDA failure in actions sync: an illegal memory access was encountered` after 78 updates of the `turn` arm; not scored): the failure always followed a pool prune (graphs dropped) and a fresh capture. Bisection with a repro (`exp12-repro/*`: 8 workers, 200-tick episodes, snapshot every 6144 steps, ~100 prunes in 9 updates): never pruning → fine; pruning but keeping the dead `CUDAGraph` objects alive → fine; keeping only their output tensors or their snapshots → crash; `torch.cuda.synchronize()` around the delete, or a shared mempool for all captures → crash. Standalone (`native/rl/autoresearch/graph_workspace_repro.py`, no game): six graphs of `Policy.step` captured on the same stream, free two, `empty_cache`, replay the rest → illegal access; a plain `F.linear` graph reproduces, a GRU-only or multinomial-only graph does not; a fresh stream per capture passes. Root cause: cuBLAS caches one workspace per (handle, stream) (`CublasHandlePool.cpp`); the first graph captured on a stream allocates that workspace inside its private pool and every later capture on the same stream — torch recycles capture streams — bakes the same pointer into its GEMM kernels, so freeing the first graph frees memory the others still use. Fix (0155929): `torch._C._cuda_clearCublasWorkspaces()` immediately before each `capture_begin`, so every graph allocates and owns its own workspace. `inference.Stepper` gets the same line (it captures under `torch.cuda.graph`, which uses one shared stream: the first historical stepper captured owned the workspace of every later one, and steppers are closed on prune — that path never crashed only because `Stepper.close()` never coincided with an `empty_cache`). The repro then survives 96 prunes / 113 captures; `check_pipeline` A–F exact and 11/11 dumps + metrics byte-identical to `exp09-check-pipeline` both with the flags (`exp12-check-pipeline2`) and without (`exp12-check-incumbent`, covering the Stepper change on the incumbent path).

Benchmark (`exp12-turn2/`, 3 arms × 4 repeats interleaved, incumbent = game-exp09 + ten kept flags, commit 0155929):

| arm | active (4 runs) | median active | wall (4 runs) | median wall | wall s | GPU util |
|---|---|---|---|---|---|---|
| incumbent | 10418 / 10362 / 10434 / 10474 | 10426 | 8345 / 8356 / 8378 / 8313 | 8351 | 70.6 | 28.9% |
| turn (12a) | 10311 / 10491 / 10440 / 10289 | 10375 | 8344 / 8459 / 8357 / 8320 | 8351 | 70.6 | 31.1% |
| turncollect (12b) | 10696 / 10645 / 10661 / 8892* | **10653** | 8527 / 8520 / 8544 / 7338* | **8523** | 69.2 | 32.3% |

\* turncollect-4 saw ~20 s of foreign GPU load (samples: 110 W, 87% util, vs ≤55 W in every other run; its `train_seconds` doubled for ~40 updates). It is counted in the median as the program requires and only lowers it from 10661 to 10653; the three clean runs are all above every incumbent run (worst 10645 > best 10474).

Per-step decomposition (P step cadence, G group-thread cycle per launch, E game env, X = P − G main-thread turnaround; ms):

| arm | P | G | E | X |
|---|---|---|---|---|
| incumbent | 1.96 | 1.20 | 1.16 | 0.76 |
| turn | 1.95 | 1.51 | 1.22 | 0.44 |
| turncollect | 1.89 | 1.52 | 1.21 | 0.37 |

Result: **12a alone is flat** (−0.5% active, +0.0% wall) despite cutting X by 0.32 ms — the whole saving reappeared as a longer group-thread wait (G +0.31 ms), i.e. the turn graph shortened the CPU issue time but the group thread now waits on the GPU chain itself: the single graph serialises the learner step and the historical steps that `BOPL_OPPONENT_STREAMS=1` used to overlap on side streams, and the fixed-slot loop runs every version slot each turn. **12b (turn + collect) is a keep**: active 10426 → 10653 (+2.2%), wall 8351 → 8523 (+2.1%), P 1.96 → 1.89 ms. The collect graph is what moved the cadence: those six ops were on the critical path before the learner step and cannot overlap with anything.

Cumulative over the new-unit control (3848 active / 2979 wall): **active +177% (10653)**, **wall +186% (8523)**. Incumbent from here: game-exp09 build + twelve flags (the ten plus `BOPL_TURN_GRAPH=1 BOPL_COLLECT_GRAPH=1`).

Lessons: (1) a CUDA graph is not free of dependencies on the allocator — every graph must own the cuBLAS workspace it uses, and torch 2.14 does not enforce that when captures share a stream; the standalone repro is checked in. (2) Cutting CPU launch latency only pays when the GPU chain it feeds is not the next bottleneck; the graph made the historical steps serial again. Next: capture the turn so the historical branches keep their fork/join (side streams inside the capture become parallel graph branches), or drop the learner step out of the turn graph and keep only the opponents' part; then the `--pipeline-groups 4` explicit-semantic-change workload and the default-on flip of the twelve flags.

### exp13: historical steps as parallel branches inside the turn graph (`BOPL_TURN_GRAPH_BRANCHES=1`)

Idea: exp12 showed the single turn graph serialises the learner step and the historical opponent steps that `BOPL_OPPONENT_STREAMS=1` had overlapped eagerly (G rose 0.31 ms). In `_run_branches` each historical step is forked onto its own stream from an event recorded before the learner step and joined on the capture stream before the scatters; the kernels are issued in the same CPU order, so the Philox offsets and the values are unchanged (8-worker check_pipeline identical 11/11 vs exp09, `exp13-check-pipeline/`).

First 16-worker bench (`exp13-branches/`, commit 4981af9) **failed**: incumbent-w16-1 finished (10652 active / 8463 wall; P 1.898 G 1.526 E 1.225 X 0.372 ms), branches-w16-1 died at update 86 of 192 (up to there P 1.846 G 1.428 E 1.268 X 0.419 — G 0.10 ms better, as intended). The reported error, `cudaErrorStreamCaptureUnjoined` from `capture_end`, was a mask: the log's first traceback is `torch.OutOfMemoryError` inside `policy.step` during a capture (GPU 11.16 GB in use by the trainer, 9.88 GB of it in CUDA-graph private pools), and the exception skipped the branch joins so `capture_end` reported unjoined work instead. Peak allocated memory in the metrics had grown 294 → 471 MB, but reserved memory (not logged then; `cuda_reserved_mb` is logged from now on) is what filled the card: every `CUDAGraph` keeps its own private pool holding every block its capture touched (cuBLAS workspace, cuDNN GRU scratch, one segment per stream used) until the graph is destroyed, and a freed pool is only unmapped by `empty_cache` or an allocation failure. A standalone test (job tmp `graphmem.py`) measures ~110 MB reserved per plain turn graph and ~320 MB per branched graph (five streams); with ~20 live graphs per group and two groups the branched run needed ~10 GB.

Fix (this commit): all graphs of one `TurnGraph` share one memory pool (`graph_pool_handle()`, forced on with BRANCHES, `BOPL_TURN_GRAPH_POOL=1` for plain graphs). Safe because the graphs of a group replay one at a time on one stream and every output is consumed before the next replay. The standalone test then holds reserved memory flat at 416 MB across 12 branched captures and confirms each graph's replay outputs are unaffected by replaying the others. Also a dedicated capture stream per `TurnGraph` (torch hands streams out round-robin from a pool of 32, so a fresh `Stream()` per capture would eventually alias a branch stream), asserted distinct at construction.

**Second bench (after the shared-pool fix, commit 09ac67b, `exp13-branches2`, 4 runs per arm alternating):**

| arm | median active | median wall | runs active | roll s | train/upd ms | P ms/t | G | E | X=P-G | gpu util |
|---|---|---|---|---|---|---|---|---|---|---|
| incumbent (12 flags) | 10414 | 8296 | 10048-10576 | 46.8-49.5 | 41.0-41.7 | 1.90-2.01 | 1.53-1.62 | 1.22-1.32 | 0.37-0.39 | 31 % |
| branches | 10763 | 8538 | 9175-11042 | 44.6-52.5 | 40.9-43.7 | 1.81-2.14 | 1.42-1.70 | 1.26-1.35 | 0.39-0.43 | 22-41 % |

Kept: +3.4 % active, +2.9 % wall on the paired medians. The three good branches runs are the fastest
sustained runs recorded so far (11021, 11042 active); run 2 was a low outlier (9175, load1 14.9, its
GPU util 41 % and train/upd 43.7 ms say the machine was contended, not the graph). The branched
graph's own time per turn (G) drops from ~1.55 to ~1.42 ms when it is not contended, i.e. the five
historical GRU steps really do overlap with the learner step; the gap is small because each step is
only ~0.2 ms of kernel time and the launch/join overhead of five streams eats part of it. Reserved
CUDA memory stayed flat (shared pool), no prune-related growth. The incumbent is now the thirteen
flags (`BOPL_TURN_GRAPH_BRANCHES=1` added). check_pipeline identical 11/11 against the reference dump.

### IL2CPP player (`builds/game-il2cpp`, Matthew's priority 1)

The Linux IL2CPP module for 2022.3.60f1 was installed and the headless game rebuilt with
`scriptingBackend Standalone=IL2CPP`, `il2cppCompilerConfiguration Release` (ProjectSettings
diff, plus the linux toolchain/sysroot packages in packages-lock.json and `Assets/link.xml` to keep
the reflection-driven RL bridge types from being stripped). Same source as game-exp09.

Exactness: `check_game_variant` Mono exp09 vs IL2CPP, 16 workers x 2000 decisions, 2 groups:
`status exact, matches 416` (observations, rewards, dones, telemetry, episode logs identical).
Env stepping alone 4.52 -> 3.29 s (-27 %), setup 10.3 -> 7.1 s (`il2cpp-check.log`).

Sustained bench (`il2cpp-bench/`, 4 runs per arm alternating, thirteen flags on both, commit 223388b):

| arm | median active | median wall | runs active | wall s | startup s |
|---|---|---|---|---|---|
| incumbent (Mono exp09) | 11044 | 8736 | 10931-11133 | 66.9-68.6 | 10.7-11.5 |
| il2cpp | 13352 | 10595 | 13207-13387 | 55.4-56.2 | 8.3-8.6 |

**Keep: +20.9 % active, +21.3 % wall.** Every IL2CPP run beats every Mono run by a wide margin
(the four-run spread is ~1.5 % on each arm). Matthew's independent audit of the eight runs
confirmed 589824 learner samples and 192 updates each, with 18 learning/gameplay metrics equal
across each pair and the runtime payload hashes matching the files. The IL2CPP build is now the
incumbent game (`BOPL_RL_GAME=.../builds/game-il2cpp/BoplBattle.x86_64`). Cumulative over the
new-unit control (3848 active / 2979 wall): active +247 %, wall +256 %.

### Native PufferLib trainer migration: design (2026-09-07, revised after review)

Program order (program.md, 07d6ff9): native trainer with the upstream linear encoder first,
qualified for throughput and strength per wall hour against the old trainer on sustained-w16;
then the native CUDA Bopl set encoder against the linear native baseline; then the simulator
assessment. All upstream changes go through `build.py` overlay patches on the pinned PufferLib
42f70d6; the Bopl binding opts into the upstream routing hooks.

Matthew's review findings (2026-09-07) and how the design answers them:

1. **Boundary-safe opponent ownership.** `count_aligned` only proves that every env has seen
   some match boundary, and `c_reset` starts the next match immediately, so a host-side bank swap
   after alignment would change an env's opponent mid-match. Fix: the trainer never swaps a bank's
   weights while any env of that bank is mid-match; instead each frozen bank owns a fixed set of
   historical envs, and a bank reload is a two-phase per-env latch. The host stages the new
   snapshot in a bank-local pending slot and marks the bank's envs `pending`; each env applies the
   pending switch inside its own `c_reset` (that is, exactly at its own match end) by flipping to
   the staged weights slot. Concretely the bank holds two weight slots (active, staged) and each
   env carries `slot` (0/1) that selects which slot its forward uses; the host flips the staged
   slot into use only after all envs of the bank report `applied`. Side (which player the frozen
   row drives) is drawn per match in `c_reset`, so side swaps are boundary-aligned by construction.
   This keeps match-long ownership exactly.
2. **Recurrent state for sampled sequences.** `select_copy` never fills `TrainGraph.mb_state`;
   `reset_state=false` merely leaves whatever the previous minibatch left in it. Fix: a per-row
   rollout-start state buffer `{layers, total_agents, hidden}` is snapshotted before each rollout
   (all banks, primary rows are what matter), and a new `select_copy` channel gathers the rows of
   the sampled segments into `mb_state`. Together with the terminal reset-mask channel and the
   MinGRU scan restart at masked steps (forward and backward recompute), each sampled segment
   starts from the same hidden state its rollout forward used, and episodes still reset the
   state. Rollout `buffer_states` are zeroed at terminals for every bank before the next forward.
3. **Learner-only sampling per buffer.** Learner rows are the primary slice of every buffer
   (B=16 per buffer with 12 primary rows means rows 0..11 and 16..27, not 0..23). Fix: the prio
   weights of rows with `row % agents_per_buffer >= bank_layout[1]` are zeroed before the
   normalise/CDF step, so frozen rows can never be sampled; `zero_frozen_advantages_cuda` already
   uses the same per-buffer test. `learner_steps` counts primary rows times horizon per rollout.

Declared semantic differences vs the old trainer (to be identified in the qualification report):
- Opponent assignment: fixed bank per env with bank rotation, instead of resampling a snapshot per
  match. With four banks holding the four most recent snapshots and equal env counts per bank the
  per-time-slice mix over snapshots is uniform like the old `recent` draw, but which env faces
  which snapshot is correlated across matches. Not proven equivalent; measured by strength.
- Recurrent core MinGRU (upstream, log-space scan) instead of standard GRU; optimizer Muon
  instead of Adam; GAE without bootstrap past the horizon (upstream `puff_advantage`) plus
  prioritised segment replay; upstream reward clamp to [-1, 1] removed for Bopl (win plus kill in
  one step is +/-1.1 and must stay distinct from a bare win).
- Rollout horizon/minibatch come from `bopl.ini` (horizon 64 vs old 128 with sequence 32).

Outputs: `puffer/train.py` becomes native-only, writes `policy-<step>.bin` plus the sidecar
(`config.training_maps`, `config.episode_ticks`, learner steps), keeps the snapshot pool for the
frozen banks, and `smoke.py` drops `torch_pufferl`. Old-trainer comparisons use Git revisions.

Ownership while this lands: Matthew owns `policy_eval.c`, `native_policy.py`,
`test_native_policy.py` (CPU set encoder later, weight layout to be coordinated), the C# game
cleanup (bd76347 fix math, 836428b incumbent toggles, both to go into the combined IL2CPP
rebuild), the obsolete UI/report family (5960253) and monitor cleanup (21577dd), and the
replay/runtime-path cleanup (reference.py, replay_data.py, render_video.py,
extract_demonstrations.py, verify_demonstration.py). 6a6dcaf (Bow/imitation/league/extension/
series deletions) applies once the old-runtime qualification no longer needs those files.

Next concrete step: edit `bopl_puffer.h`/`binding.c`/`eval_binding.c` for per-slot pointers,
tag/boundary/latch fields and the per-match side; rebuild with `build.py`; run a routing test
(non-identity permutation, side swaps, two-buffer layout) before the overlay kernels.

### 2026-09-07 21:30 UTC — routing lands (env side of the native migration)

`bopl_puffer.h`, `binding.c`, `eval_binding.c` now implement the per-slot design and
`native/rl/puffer/check_routing.py` proves it (artifact `native-routing/result.json`, all 27
checks pass on the IL2CPP player, host stepping):
- Every env owns `num_agents` adjacent vector slots (self-play: `players`; historical:
  `1 + banks`, slot 0 = learner, slot 1+b = bank b). `MY_USES_PERM` + `my_setup_perm` route each
  slot to the physical row `agent_perm` names, so a trainer-built permutation packs every buffer
  as `[primary rows | bank 0 rows | bank 1 rows ...]`, which is exactly what upstream
  `bank_layout` slices. Verified with a non-identity permutation on 1 and 2 buffers: observations,
  actions (previous-action echo), rewards and terminals all land in the permuted rows.
- Ownership changes only in `c_reset`: side and bank are drawn from a separate xorshift stream
  (`side_rng`), so the game setup draws (seed/map/loadouts from `env->rng`) are unchanged with
  or without banks. The bank is drawn uniformly from the trainer's valid-bank mask
  (`bopl_set_valid_banks`, atomic, process-wide); an empty mask aborts visibly. The chosen
  opponent row gets `terminal=1` at match start (fresh recurrent state), the previous opponent
  row is blanked and stays all-zero while idle. Sides and banks both vary across matches per env.
- `MY_USES_TAGS`: historical envs carry tag 1 and set `boundary_reached` in `c_reset`;
  `count_aligned` reported exactly one boundary per historical match, which is the handshake the
  trainer uses to swap a bank: mark it invalid, wait until every historical env has crossed a
  boundary since (so no live match still uses it), load weights, mark valid.
- Restricting the mask to one bank made every new match pick that bank (checked).
- Log gains learner/historical episodes and wins (learner = self-play rows, or the learner side
  of a historical env). Per Matthew: `static_vec_log` averages finished episodes only and clears
  env logs whenever any env reported, so behaviour telemetry for the dashboard must come from
  independent cumulative learner counters (unfinished episodes included, warmup/frozen excluded),
  not from these Log averages. To be added with the trainer output.
- `eval_binding.c` routes slots identity to its own buffers (`research_route`); evaluation still
  steps in sampled mode and resets cleanly (smoke: 200 steps, two episode ends, pairs consistent).

Matthew's constraints received today (supersede earlier text above where they differ):
- Deliverable ends at a QUALIFIED native SET trainer and handoff (to thr_wq22hezkbf); we do not
  launch the 400M run. Sequence: native build/correctness -> native-vs-old sustained-w16 sanity
  -> full native-only cutover -> set encoder correctness + throughput + short strength vs linear
  -> handoff.
- Final trainer reward: sparse terminal +1/-1, timeout -0.5 for both players, no 0.1 kill/death
  shaping; entropy 0.01 annealed to 0.0005; episode cap 7200 ticks default and configurable;
  truncation/episode telemetry retained. Comparison arms stay matched on reward/cap/entropy
  before treatments change (phase-1 native-vs-old keeps the old reward).
- Learning Lab (serve.py, web/, study_data.py, replay_library.py, port 8876) stays; the
  dashboard UI is owned by thr_iiychwrq9p in the dashboard Rift. We own trainer output, eval and
  checkpoints; a short metrics contract is coming. New native stats requested: per-head entropy
  (aim/buttons), advantage std before normalisation, per-decision evaluation value and
  action-probability dumps. Root owns eval-dump wiring and the Python metric-format helper.
- Do not cherry-pick 5960253. The null-hypothesis service deferral was retracted.
- Stage 2 set encoder ABI (Matthew, CPU side on research/cpu-set cc58b70): bias-free weights in
  order flat[H,905], entity[64,21], slime[64,28], fusion[H,H+128], then upstream decoder and
  MinGRU; H=128; presence != 0; absent sets contribute zero; metadata architecture
  `bopl-set-mingru-v1`. CUDA parity vs the CPU oracle and layout agreement are required before
  integration.

Next: overlay patches in `build.py` (rollout-start state gather into `mb_state`, MinGRU terminal
masking, learner-only prio, learner-step counter, reward clamp removal), then `bopl.ini` banks and
the native-only `puffer/train.py` with sidecars, snapshot pool, bank swaps and metrics output.

## 2026-09-08 native trainer 16w timing (harness retargeted, commit 60632d3)

`bench_throughput.py` now drives `puffer/train.py` (learner-decision budget in `--steps`, no
`--load`/`--no-video`/`--start-step`; `learner_dps_active` uses `result.elapsed`, the same span
the old harness measured; `training_seconds` is the metrics clock from the first rollout).
Workload `native-w16-786432-fresh-sparse`: IL2CPP game, 16 workers, 4 banks, horizon 128, random
init, sparse reward (+-1, timeout -0.5), cap 3600, maps 0,1,2, 786432 learner decisions
(1048576 physical) in 256 updates, incumbent game flags + BOPL_PARALLEL_OPEN. Medians of 3:

| arm | learner dps active | learner dps wall | wall s | rollout s | train s |
| passes 1 (default) | 14689 | 13756 | 57.2 | 44.5 | 0.9 |
| passes 3 | 14396 | 13510 | 58.2 | 44.2 | 2.6 |

Old trainer on sustained-w16 (il2cpp-w16, same game build and flags): 13352 active / 10595 wall,
55.7 s wall for 589824 learner decisions (786432 physical) in 192 updates; rollout 35.4 s, train
8.0 s, start-up 8.4 s. The old run resumed the 5.0M checkpoint with the kill/death +-0.1 reward,
constant lr 3e-4, Adam, GAE 0.95, and did 3 epochs x 4 minibatches of 1024 learner decisions
(12 gradient steps) per 3072-learner-decision update. The native run trains from random weights
with Muon + cosine lr, GAE 0.90 + vtrace, prioritized replay, and 1 (or 3) passes over one
3072-row minibatch per update. So these are different workloads, not a matched speedup:
the native trainer's per-update optimizer work is much smaller (the train phase is 0.9-2.6 s vs
8.0 s) and its rollout phase is slower per learner decision (44.5 s / 786432 vs 35.4 s / 589824,
i.e. 17.7k vs 16.7k learner decisions per rollout second, roughly equal). The wall-rate gain
(+30%) is mostly start-up (3.6 s vs 8.4 s) and the shorter train phase. Cross-run noise is small
(active 14282-14795 across both arms). Passes 3 costs 1.7 s per 786k learner decisions.

Observed on all native 16w runs: joint entropy flat at 8.299 (max 8.318) for the whole 786k
budget, explained variance 0-0.08, lr cosine-annealed to 1e-8 by the last update (`anneal_lr` is
left at PufferLib's default 1 by train.py). These are short random-init runs so no learning is
expected; the lr anneal to zero over a benchmark budget is a config issue reported to root.

Correctness on this branch: SET encoder core hook (`BOPL_NATIVE_ENCODER` read in the overlay's
`create_custom_encoder`, fail-closed on bad values), full CUDA build, `check_set_encoder`
pass_all (forward <= 2.4e-7, grad <= 2.9e-6, graph replay / permutation / masked poison / empty
set), unit tests 6+2+3 OK, smokes linear/set/set16 86 PASS, SET 4w CLI run with two evaluations.

## 2026-09-08 native C/API cutover (commit 90ac98e, source only; no build/GPU until root returns the rebuilt IL2CPP executable)

Merged root a0d513f first (environment.step_mode API, train.py runtime_options flag block, series.py BOPL_SPLIT_STEP env, graph_workspace_repro.py, README old flag exports all gone on root's side). Then in my files:

- `bopl_puffer.h`: BOPL_PARALLEL_OPEN, BOPL_SPARE_GAME and BOPL_SPLIT_STEP gates removed. Every env opens its world process on a thread at init; every auto-resetting env runs a spare process (evaluation envs with `no_autoreset` stay single-process, as before). The delayed observation always takes the single-roundtrip sampled step (3+1 ticks); the two-roundtrip oracle branch and the `split_step` field are deleted.
- `bopl_rl.c/.h`: BOPL_RL_EAGER_IO gate removed; non-blocking-first transport and one-write request framing are the only path. The "Unknown RL command" fallback to two step commands is gone: an old game build is now a hard `BOPL_RL_GAME_ERROR`. `bopl_rl_sampled_native` deleted.
- API: `bopl_vec_step_mode` (binding.c) and `research_step_mode` (eval_binding.c) deleted; `smoke.py` and `check_routing.py` no longer assert on the transport mode (there is only one).
- `sampled_check.c` kept as the exactness oracle (sampled command == two sequential steps; argument guards; phase-1 terminal/truncation). Its `expect_native` argument and output field are gone: `sampled_check <game> <bootstrap.rep> <game.log>`.
- Syntax-checked bopl_rl.c, sampled_check.c, eval_binding.c and binding.c (against the overlay vecenv.h) with -Wall -Wextra -Werror; no full build yet.
- rg audit over the shipped tree (excluding artifacts, evidence, notes, results.tsv, program.md) for torch_pufferl, `from policy import`, Stepper, TurnGraph, BOPL_SPLIT_STEP, --backend, step_mode, sampled_native and every deleted game/trainer flag: zero hits.

Pending once the GPU is back: `build.py --device cuda`, `native_policy.build()`, `make -C native/rl`, then smoke linear/set/set16, check_routing, reset_check, sampled_check, unit tests, on the rebuilt game binary. Exactness evidence for the now-unconditional paths is the earlier flag-on/flag-off comparisons (exp03 spare, exp05 parallel open, exp09 eager I/O, sampled parity-main/parity-split in evidence/throughput).

## 2026-09-08 final build validation, SET throughput, strength protocol (commit f2cbbae + this one)

Game: root's final IL2CPP build from cleaned C# (copied to `native/artifacts/rl/autoresearch/builds/game-il2cpp-final/`,
player sha256 0bfe51325720c950b4ad75ce7c3b65595fd7e939908520ec33600919919327a8, GameAssembly.so
67723a4b5427af191d8c16b6090f74ff523c6cf1401fea6273311e28f23b678d, global-metadata.dat
a4e4759368bde74d17d54f46f30df79b1d58fe491dd1970abdc47bd5c17e1a9e; same bytes as the installed
`ExportedProject/Builds/LinuxIL2CPP/BoplBattle.x86_64`). Full native rebuild after the C/API cutover
(build.py --device cuda, native_policy.build(), make all + make -B research) is clean.

Correctness on the final build, evidence in `native/artifacts/rl/autoresearch/final-core-checks/`:
sampled_check pass (459 decisions, 3 phase-1 terminal episodes); reset_check RESET_ISOLATION_OK
(5738 exact snapshots); smoke linear / set / set16 (16 games, 4 banks) 85 PASS each, 0 failed checks;
check_routing 31/31; unit tests 2 (puffer) + 16 (research) OK; check_set_encoder rebuilt, pass_all
(forward <= 2.4e-7, autograd grad <= 2.9e-6, graph replay, permutation, masked poison, empty set) plus
root's new finite-difference check of the CUDA forward vs the native backward: 16 coordinates in each
of flat / entity / slime / fusion, central eps 1e-3, max abs error 6.8e-5 / 6.7e-5 / 7.5e-5 / 3.0e-5.

Throughput, workload `native-w16-786432-fresh-sparse-7200-final` (final game, 16 workers, 4 banks,
horizon 128, random init, sparse +-1 / timeout -0.5, cap 7200, maps 0,1,2, passes 1, entropy
.01->.0005, 786432 learner decisions in 256 updates, no game env flags: the old toggles are now the
only code path). Medians of 3, `native/artifacts/rl/autoresearch/final-bench-w16/summary.json`:

| arm | learner dps active | learner dps wall | wall s | rollout s | train s | CUDA MB |
| linear | 14957 | 14004 | 56.2 | 43.9 | 0.9 | 982 |
| set | 14616 | 13683 | 57.5 | 45.2 | 1.1 | 1190 |

SET costs 2.3% of both rates (all six runs within 14423-14982 active; zero cgroup memory.high
events). Cap 7200 vs the earlier cap-3600 linear rows (14689/13756): +1.8% active, same wall, i.e.
inside noise; the earlier flag-gated paths and the now-unconditional ones perform the same.
Learning signal in these 786k random-init runs is nil for both arms (entropy 8.30 -> 8.29,
explained variance <= 0.01, tail win rate ~0.5), as expected; lr still cosine-anneals to ~1e-8 by
the last update (PufferLib `anneal_lr` default, reported to root).

Strength protocol (`native/rl/autoresearch/strength_check.py`, source only; root executes it):
per encoder and seed, one fresh train.py run with identical settings except `--encoder`
(16w, 4 banks, horizon 128, passes 1, sparse reward, cap 7200, maps 0,1,2, entropy .01->.0005,
cudagraphs 1, `--no-eval`), arms interleaved per seed; then `evaluate.py --suite final` on the last
checkpoint (random + own initial on maps 0,1,2, random on held-out 3,4; same seeds for every run);
then a side-swapped fixed-seed round robin of all final checkpoints plus the random anchor on maps
0,1,2 at cap 7200 (timeouts are draws; Bradley-Terry Elo anchored at random = 0, same fit as
elo_ladder.py). Primary metric: pooled SET-vs-linear score (win 1, draw 1/2) with a Wilson
interval. Verdict rule fixed before any run: `qualified` iff the interval's upper bound >= 0.5 and
SET's pool score vs random >= linear's minus 0.05. The report also carries per-seed scores, Elo,
final-evaluation win rates, training tail win/timeout rates and training wall-hours per arm.
Self-tested on the two 786k bench checkpoints (`strength-selftest/`: training skipped as complete,
final evaluation 17 games per run, 18-game pool, verdict computed) so only the GPU training stage is
unexercised at the full budget. Pool speed on CPU: ~2000 ticks/s per worker.

Recommended budget: `--steps 10000000 --seeds 73,173,273` (six runs of ~12 min at ~14k learner
dps, ~75 min GPU total; final evaluations and the 1008-game pool are CPU, a few minutes). Fallback
if time is short: `--steps 5000000 --seeds 73,173` (four runs, ~25 min GPU).

Limitations to carry into the handoff: (1) no old-trainer anchor exists (the old torch GRU has no
faithful native export), so strength per wall-hour vs the old trainer is only comparable through
win rates vs random/initial and the training telemetry, not head to head; (2) both arms anneal the
learning rate to zero over their own budget, so a 10M run is not a prefix of the 400M run's
schedule; (3) short budgets favor the simpler encoder if anything, so `not-qualified` on a
short check would mean "re-run longer", not "SET is worse".

## 2026-09-08 parent override, entropy schedule, program rewrite (merge 8dfb4c3 + this commit)

Parent (via root) dropped the matched SET-vs-linear strength check as a launch gate: Matthew already
chose SET, correctness is covered by final-core-checks, and the old 10M runs were flat, so a short
comparison was unlikely to discriminate. `strength_check.py` stays as an optional post-launch A/B
(fail-closed resume landed in c261157; no strength runs were made). No 60M/400M launch from here.

Root landed the parent's entropy requirement in f998744 (merged here): the Python wrapper sets
`trainer.hypers.ent_coef = max(floor, initial * 2^(-learner_steps / half_life))` before each native
`train`, `--entropy-half-life` default 30,000,000 learner decisions, native `anneal_ent_coef=0`; the
native trainer already copies the live scalar into the device buffer each update, so CUDA graph
replay sees it without a recompile. With .01 -> .0005 the floor is reached at ~130M learner
decisions. The learning-rate cosine anneal to zero over the run budget is now explicit
(`anneal_lr=1, min_lr_ratio=0`, unchanged behavior). Root's boundary unit test covers
0/30/60/120/130/400M. config.json gains `entropy_schedule`, `entropy_half_life`,
`learning_rate_schedule`. Consequence for the bench rows above: the 786k runs recorded before f998744
used the cosine-per-budget entropy schedule; throughput is unaffected (one scalar copy per update).

Handoff: root owns `native/rl/NATIVE_HANDOFF.md` (8d5f56f, launch command for thr_wq22hezkbf, evidence,
limits). Root's evaluation plan: full suite is 57 games on three maps, evaluation cadence 5M learner
decisions, checkpoints every 1M independently. This Rift's contribution is provenance only:
`program.md` rewritten around the shipped native design (final audit rg on the merged tree is clean
for every deleted token; the single `scripted_opponents` hit is a historical sentence in the
dashboard-owned METRICS_CONTRACT.md).

Final-core-checks binaries: `make -C native/rl all` writes `sampled_check`/`reset_check` to
`native/artifacts/rl/`, not `native/rl/`.

Merged root 3603dca (native-evaluation Rift, `research/cuda-set`; handoff now `--eval-every 5000000`,
`--entropy-half-life 30000000`, entropy evidence `native/rl/evidence/native-cutover-entropy.json`).
`strength_check.py` now passes `--entropy-half-life` (default 30M = train.py default), records it in
plan.json and requires `entropy_schedule == 'exponential'` plus the same half-life in cached runs'
config.json. Checked: a copy of the pre-f998744 self-test run is rejected with
`entropy_schedule: (None, 'exponential'), entropy_half_life: (None, 30000000)`, and the old
`strength-selftest/plan.json` is rejected for the missing field. So the self-test evidence predates
the schedule change; the script has not been exercised end to end on a post-f998744 run.

## 2026-09-08 constant learning rate (parent override; merge b79640c)

Parent's final config override: the learning rate must stay constant at .0003 so the late,
sparse-reward signal is not frozen by a cosine decay, matching the old trainer. Root's fb0f13e
adds `--anneal-lr {0,1}` (default 0, constant) and `--min-lr-ratio` to train.py, writes the native
`anneal_lr` explicitly and records `learning_rate_schedule` ('constant' or 'cosine') in config.json;
b79640c holds root's six-update check (0.0003 through six native updates with `anneal_lr=0`, entropy
halving through the floor as before). Entropy stays exponential with the 30M half-life.

Merged base main b79640c here (fast-forward of my b5bbe62). `strength_check.py` now passes
`--learning-rate`, `--anneal-lr` and `--min-lr-ratio` explicitly (defaults .0003 / 0 / 0 = train.py),
records them in plan.json, and a cached run must match `learning_rate_schedule` plus the native
`learning_rate`, `anneal_lr` and `min_lr_ratio`. Checked with synthetic run directories shaped like the
current config.json: a constant-LR run passes the settings check (it then reaches the checkpoint
identity check), a cosine-trained run is rejected with `learning_rate_schedule: ('cosine', 'constant'),
anneal_lr: (1, 0)`, and a pre-fb0f13e run is rejected for the missing fields. program.md's
schedule paragraph now says constant by default. All throughput rows in results.tsv and the
final-bench-w16 artifacts were produced under the earlier native cosine schedule (`anneal_lr=1`,
visible in their config.json); they remain valid throughput provenance, since the learning-rate
schedule does not touch the collection path, but they are not learning provenance for the 400M run.

## 2026-09-08 standalone simulator assessment (priority 3, source-only)

Question: given the IL2CPP and native-trainer results, is a game-free simulator worth building?
No full rewrite is requested; this records what the numbers say and when to revisit. No new runs.

**Where the time goes now** (final 16-worker bench, `final-bench-w16`, medians of 3; each run 256
updates, 1,048,576 physical decisions = 131,072 ticks per game):

| quantity | linear | SET |
|---|---:|---:|
| learner decisions/s (active, benchmark stopwatch) | 14,957 | 14,616 |
| rollout seconds of ~45–46 s elapsed | 44.2 | 45.2 |
| nominal game ticks/s during collection, summed over 16 games | 47,484 | 46,414 |
| collection ms per nominal tick per game | 0.337 | 0.345 |
| busy cores during the loop (of 12) | 8.8 | 8.6 |
| GPU utilisation | 37 % | 43 % |

Collection is ~98 % of the logged training-loop time; the PPO update is ~1 s per 45 s.
The tick estimate is physical decisions × 4 action-repeat ticks / 2 players, divided by
rollout seconds; it excludes startup and other outer-loop overhead and assumes full
action repeats. Learner throughput uses the separate benchmark stopwatch, so these
rows must not be divided directly to infer a decision-to-tick ratio. The per-step critical path is the
slowest game's decision step plus the trainer's per-group turnaround (exp09/exp13: game main thread
~0.6–0.8 ms compute per 4-tick step, i.e. ~0.15–0.2 ms per tick, and ~0.75 ms of trainer main-thread
work per group turn). `reset_profile` measured the bare tick at 0.07–0.13 ms in one idle game; the
exp08 profile put the remaining managed cost in DetPhysics rounded-rect raycasts and Fix arithmetic
after the render-only work was removed. The Unity frame around each 4096-tick batch is ~0.11 ms per
tick of `UpdatePreloading` that cannot be removed without leaving the engine. These profiles suggest diminishing returns from the tested Unity optimizations: IL2CPP, frame batching, ray culling and 128-bit Fix math each bought their
few percent and the remainder is genuine simulation plus engine overhead, spread over 16 processes that
between them keep ~9 of 12 cores busy while the GPU idles 60 % of the time.

**What the existing Rust port shows.** `native/` (crate `bopl-sim`, 16,064 lines of Rust) is a
bit-exact port of the Level1 movement/jump/dash/shove/first-death slice for one to six players,
validated tick-for-tick against Unity captures (303k math cases, 24k geometry cases, 5,870 golden
ticks; `native/VALIDATION.md`). Through the C ABI it steps a two-player world at 187,842 steps/s on one
thread (`native/artifacts/benchmark-report.json`, walk/reset workload, no observation encoding or
policy): ~5.3 µs per tick against ~150–200 µs of Unity main-thread compute per tick, roughly 30–40×
per core, or ~4× the whole machine's current 47k ticks/s on a single core. That speed is for the
simple slice; a port carrying arrows, platforms and the full raycast set would be slower, but the
Fix arithmetic and rounded-rect geometry are already the validated parts.

**What it would take.** `native/FULL_COMPATIBILITY.md` says plainly that full-game parity is not
achieved: the C# gameplay assembly is 109,664 lines, the catalog has 39 abilities, and the port covers
none of the abilities beyond dash. The training environment, however, fixes the loadout to Dash (2),
Bow (4) and Platform (14) (`binding.c` defaults; `random_loadouts` is off) on maps 0–4 with two
players and three stocks. A *training-loadout* simulator therefore needs: DetPhysics (3,203 lines C#,
partly ported), SlimeController/PlayerPhysics (1,828 lines, the movement part ported), arrows and
their hits, spawned resizable platforms as physics bodies, the stock/death/respawn loop, the five map
geometries, and the observation/telemetry encoder that `bopl_puffer.h` already expresses over the
`BoplRlState` snapshot. Each new mechanic needs Unity capture goldens the way the movement slice got
them; the validation harness and capture format exist, so the work is porting and golden-collecting,
not inventing a method. The reward for going all the way is not only faster ticks: an in-process
simulator removes the socket bridge and the 16-process lockstep, lets the trainer step thousands of
environments per batch (the PufferLib native-vector shape), and turns the current per-group trainer
turnaround into a cost amortised over a large batch. Order of magnitude, on this machine, that suggests
several hundred thousand ticks/s (12 cores at 20–50 µs per full-game tick) against 47k
today, i.e. a speculative 5–10× gain in learner decisions per hour. This is not a measured
training result: full-mechanic cost, observation encoding, scaling and GPU saturation
remain unverified.

**Risks.** (1) Fidelity: anything not bit-exact reopens the train/deploy gap, and the evaluator,
replays and Learning Lab all run the real game. Exactness has to be proven per mechanic, and the
port's own record shows how much of the cost is in the goldens rather than the code. (2) Scope creep:
the 39-ability catalog and six players are the port's stated target; the training case needs three
abilities and two players, and the assessment only holds for that scope. (3) It is not on the
critical path today: whether the 400M run is limited primarily by learning signal or collection rate
is still unknown. Its sparse reward, constant learning rate and 30M-half-life entropy
configuration should be evaluated before committing to the port.

**Recommendation.** Do not start the rewrite now. Revisit when the 400M run gives evidence that more
decisions per hour would change the outcome (loss curves still improving at budget end, or an A/B
that needs many seeds), and then scope it to the training loadout, in `bopl-sim`, mechanic by mechanic
with Unity goldens, starting with arrows (the largest unported gameplay system the loadout touches).
The tested Unity-side levers show diminishing returns: worker-count and pipeline shapes were
swept (exp09–exp13), the historical ~0.75 ms per group turn came from the earlier trainer and does
not establish the remaining overhead in the shipped native path. A current breakdown
would be needed before claiming an optimization floor.

## 2026-09-08 read-only look at the parent's 400M run (first 8M learner decisions)

Parent launched `native/artifacts/rl/null-hypothesis-native/001-s73` from the canonical checkout
(SET, 16 workers, 4 banks, constant lr .0003, entropy .01 with 30M half-life, sparse reward, cap 7200,
final IL2CPP player 0bfe5132…, binding cff33dad…). Nothing here was run by me; these are metrics.jsonl
and evaluations read from disk while the run is at ~8.2M learner decisions (~480 s, i.e. ~17.6k learner
decisions/s sustained, projecting ~6.5 h to 400M; rollout 0.17 s and train 4.4 ms per update, the same
per-update shape as final-bench-w16).

Two things worth knowing when reading this run's dashboard:

1. **`kl` is near zero and `clip_fraction` is zero before the single optimizer step.** With
   `--learner-passes 1` the native trainer runs `total_minibatches = replay_ratio·batch/minibatch = 1`
   (config: batch 7168, minibatch 3072, replay_ratio 0.4286), and that single minibatch is scored with
   the same weights that collected it, so `ratio = exp(logp − old_logp)` is approximately 1 and nothing is
   clipped. Canonical metrics through 8.2M learner decisions have KL in
   [-8.54e-10, 0], consistent with floating-point roundoff. Each update is one plain policy-gradient step (vtrace/GAE advantages, prioritized rows) and
   PPO's clip is inert. The old trainer did 3 epochs × 4 minibatches (12 steps per 3072 decisions), so
   its KL/clip carried information. On this run a zero KL is not evidence of stability, and tiny nonzero
   KL is not itself a bug. A material discrepancy warrants investigation. Additional
   learner passes can expose changes from preceding optimizer steps; they do not
   guarantee nonzero values. These metrics do not measure this update after its step.
2. **Entropy has not moved.** Joint entropy 8.292 nats at update 1 → 8.291 at 8.2M (max ln 4096 =
   8.318; aim 5.530 of 5.545, buttons 2.762 of 2.773), drifting slightly *up* over the last
   thousand updates (8.2896 → 8.2932) while the coefficient fell only to .0083. Explained variance
   rose to 0.3–0.4 (the critic is learning something from terminal outcomes), gradient norm ~0.15,
   advantage std ~0.11, mean reward 0.000, `policy_loss` median |0.006|. The old 10M run (with ±0.1
   kill/death shaping, cap 3600, Adam, 12 steps per update) had entropy at 7.94 by 500k decisions.
   The first evaluation (about 6.67M physical / 5M learner decisions, 57 games) is at chance: 25–50 % vs random and 37–62 % vs the initial
   checkpoint on maps 0–2, 100 % vs random on map 4 with four games.

Hypothesis: the entropy bonus may be holding the actor near uniform while the critic
learns. Entropy alone does not establish that its gradient dominates the reward gradient;
that would require measuring the separate gradient contributions. The
schedule halves the coefficient every 30M learner decisions and reaches the .0005 floor near 130M, so
the run's own design says "wait": the honest checkpoints are 30M (coefficient .005) and 60M (.0025).
A useful watch rule for parent, not a change request: if joint entropy is still above ~8.2 at 30M
and the 30M evaluation is still at chance against random, there is still no clear evidence of policy improvement. That is a watch point, not
confirmation that sparse rewards cannot learn: evaluations are small, entropy is not
a strength metric, and learning may emerge before or after the floor near 130M.
Assess later evaluations with their sample uncertainty; the remaining wall time is
a throughput projection rather than a fixed duration.
Root owns train.py and parent owns the run; nothing to do from this Rift.
