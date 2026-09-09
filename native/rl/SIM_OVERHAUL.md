# Sim overhaul plan: from 30k to ~150k rows/s (2026-09-08)

Written on branch `trainer-step` after the rollout-step work (`sim/RL.md`,
"Trainer step latency"). Supersedes the Unity-centred parts of
`MANY_WORLDS.md`: the headless .NET port (`sim/`) shipped, the 500M runs use
its NativeAOT build, and `sim/Native` already exports the `bopl_rl.h` ABI as
`BoplSimNative.so`. The static-state inventory and the correctness gates in
`MANY_WORLDS.md` (A1, section 4) still apply and are referenced, not repeated.

## 0. Where the time goes today

Run 003 (16 games x 2 players, 2 buffers, hidden 128, 4 banks), per update:
rollout 0.12 s, train 0.005 s. **Training is 4% of wall.** Everything below
is about the rollout step, which is a lockstep chain per buffer:

```
inference graph (learner + 4 banks)  ~0.45 ms -> now ~0.12 ms (parallel branches)
socket round trip + OpenMP wake      ~0.4 ms   (of which sim compute ~0.1-0.2)
```

Rows per buffer-step: 32 (24 learner). Two buffers overlap, so ~0.47 ms wall
per step, 30k rows/s, 22.6k learner rows/s. Per decision the sim runs 4
ticks (`action_repeat=4`); `native_check` measured ~21 us per quiet 2-player
tick through the full ABI, so a decision is ~0.1-0.3 ms of compute per game
depending on how much is on screen.

Throughput model used throughout:

```
rows/s = rows_per_step / t_step,   t_step ~= t_gpu + t_env
t_gpu  ~0.12-0.15 ms, flat up to thousands of rows (launch-bound)
t_env  = transport + max over games of compute  (lockstep)  ...today
       = transport + games * compute / cores               ...once games > cores
```

Ceiling with the .NET sim on this box: 12 cores / ~0.15 ms per game-decision
= ~80k game-decisions/s = **~160k rows/s** with 2 players per game (5x today).
Past that only a cheaper tick helps (section 6).

## 1. Gate 0: measurements before code (half a day, idle box only)

1. The clean serial-vs-branched A/B (`bopl-trainer-step-bench` unit, writes
   `native/artifacts/rl/trainer-step/summary.txt`).
2. Sim compute per decision under real play: run `sampled_check` /
   `reset_check`-style scripted inputs through `BoplSimNative.so` with a
   timer around `bopl_rl_step`, for each of the 5 maps and the training
   loadout; report the median and p99 per tick. This number sets P and N
   below and decides whether section 6 is ever needed.
3. Two copies of `BoplSimNative.so` `dlopen`ed in one process (distinct
   file copies, since glibc de-duplicates by device/inode), `native_check`
   run on both interleaved from two threads. This is the go/no-go for
   section 2 route A: NativeAOT supports several runtimes per process, but
   signal handlers (null-ref -> SIGSEGV) and `DOTNET_*` env are process-wide,
   and it has to be shown here, not assumed.
4. The pending `native_check` checksum vs the socket path (PORTING.md TODO):
   same 240-step zero-action sequence, FNV-1a over decoded states. Nothing
   in-process ships until this is byte-equal.

**Results for 1.3 and 1.4 (2026-09-08, `native/rl/backend_check.c`, evidence
in `native/rl/evidence/trainer-step/backend_check-2026-09-08.txt`, taken
while run 003 was live so timings are inflated):**

- Socket path (`Host` NativeAOT via `bopl_rl.c`) and in-process
  `BoplSimNative.so` give identical checksums on all three phases: 240
  zero-action ticks `5381bf21573b20a6`, 242 decisions of scripted play with
  respawns on map 1 `8dc13de2cc8c8e7c`, dirty reset back to phase (a)
  reproduces it. 1.4 done.
- 2, 4, 8 and 16 copies of the .so `dlopen`ed in one process (distinct
  files), each running the sequence on its own thread concurrently: every
  copy prints the same checksums, no signal-handler or GC trouble. ~238 MB
  RSS per copy (16 copies 3.8 GB), same as a process today. Opens are
  ~1.1 s each and were serial in the test; the backend should open in
  parallel. 1.3 done: Route A is viable.
- Compute per decision (4 ticks, 2 players, Dash/Gust/Invisibility, map 1)
  through the ABI: ~78 us alone, 85-140 us with 8 copies stepping at once on
  the loaded box; quiet zero-action ticks ~23 us. Reset 8-9 ms warm, 55-90
  ms the first time a map is loaded. Use 0.1 ms per game-decision in the
  section 0 model until measured idle.

## 2. In-process stepping (1-2 days, +~1.5x)

Goal: `t_env` = compute only. Remove the unix socket, the .NET
socket reader, and the OpenMP-thread-to-process wake-up per step.

Design: `bopl_rl.c` grows a backend table. `BoplRl` keeps its public API
(`open/reset_begin/reset_finish/step/step_sampled/state/sample/close`,
`bopl_rl.h` v2) and dispatches to either the socket backend (today's code,
untouched) or an in-process backend that `dlopen`s `BoplSimNative.so` and
calls the exported functions directly. Selection: `BOPL_RL_GAME` ending in
`.so` = in-process; `bopl_puffer.h` does not change except for the spare.

- **Route A (ship first): one .so copy per env.** `sim/Native` is one world
  per runtime (gameplay statics; the open latch). The backend copies the .so
  into a per-run temp dir as `BoplSimNative.<k>.so` and `dlopen`s each copy:
  P independent NativeAOT runtimes, each ~31 MB of private code pages plus
  the same ~200 MB heap a process has today. Memory is unchanged vs
  processes (3.6 GB at 16); what we buy is the transport.
- The spare world (`env->spare`, `bopl_spare_open`) stays. The library's
  `reset_begin` does the whole reset before returning (~8 ms warm, 55-90 ms
  the first time a runtime loads a map), so the backend runs it on a helper
  thread and `reset_finish` joins; the spare then overlaps resets with play
  as with a process. Without that overlap every reset sat on the lockstep:
  measured 0.4-0.8 ms per step (one reset per 10-20 steps at 4-16 games with
  an untrained policy), which alone made in-process *slower* than the socket.
  Memory is the same as today (two runtimes per env, two processes per env).
  `BOPL_RL_INPROC_SPARE=0` drops the spare for memory-constrained tests.
- OpenMP threads call into the runtimes; NativeAOT attaches a thread on first
  entry. Pin nothing; measure whether `OMP_PROC_BIND` matters.
- Logging: `bopl_rl_open`'s log path argument already redirects the shim's
  stderr per world, keep `BOPL_RL_LOG_DIR/worker-<pid>-<k>.log`.

Gate: `check_pipeline.py`-style comparison, obs/rewards/dones/telemetry
identical to the socket path for a fixed seed schedule over >= 20 matches on
all 5 maps; then w16b2 A/B against the socket build on the idle box.
Expected: `t_env` 0.4 -> ~0.15-0.2 ms, `t_step` ~0.55 -> ~0.3 ms.

**Done 2026-09-08 (`bopl_rl.c` in-process backend, `bopl_puffer.h` spare
logic).** `BOPL_RL_GAME=.../BoplSimNative.so` dlopens one reflinked copy per
handle (unlinked once mapped; `BOPL_RL_INPROC_KEEP=1` keeps them with the
`.dbg` for perf). Training is bit-identical to the socket path (every loss
and match count equal over 20-60 updates at w4, w8, w16x2). On the loaded
box (run 003 live): w4 `eval_env` 0.268 -> 0.212 ms (+17% learner/s), w16x2
0.969 -> 0.856 ms (+12%). Smaller than the estimate because `t_env` is the
slowest of the 16 games, and with an untrained policy spamming Bow/Platform
a decision costs ~0.2 ms of sim CPU (the same in Host and in the .so; the
78 us of section 1 was a mild scripted schedule). Transport was only ~50 us
of the step. The clean number comes from the `bopl-trainer-step-bench` unit
(m2 = in-process) once the box is idle.

Two findings that feed section 3:

- Reset overlap is not optional. A world-per-runtime design where a reset
  cannot run concurrently with another world's ticking (the `Bind(k)`
  copy-in/out model with plain statics) puts every reset back on the
  lockstep. Section 3 must either make the swapped statics per-thread
  (`[ThreadStatic]` + `Bind` at hand-over) so a reset thread and step
  threads share a runtime, or make reset itself ~0.1 ms (snapshot), or keep
  a spare runtime per P (memory).
- Compute per decision is ~0.2 ms on spawn-heavy content, so the section 0
  ceiling (12 cores / compute) is ~60k decisions/s = ~120k rows/s for random
  play, higher as policies stop spamming. The shim diet (section 6) is worth
  more than assumed.

## 3. Many worlds per runtime (3-5 days, memory and reset cost)

Goal: P runtimes x N worlds, so 96-192 games fit in ~4 GB instead of the
~225 MB per world Route A costs, and the export (`assets.json` slice index,
scene cache) is loaded P times rather than P*N.

The shim already has the container: `UnityShim.World` owns roots,
behaviours, pending Starts, destroy queue, coroutines, layer tables and the
instance-id counter, with `World.Current` as the single static ("day 4 makes
it per-world", World.cs:9). Scene loading is our code (Export loader), so the
"additive scenes" machinery in MANY_WORLDS.md section 3 is unnecessary: a
world is a scene graph, reset = drop the world's roots and rebuild from the
cached level tree.

Work items:

1. `WorldState` in `sim/Gameplay` holding the ~50 per-match statics from
   MANY_WORLDS.md A1 (Updater 15 incl. `random`, GameSessionHandler 5,
   GameSession 4, DetPhysics.instance + levelTypeThisFrame,
   PlayerHandler.instance, GameTime.PlayerTimeScale, Constants.leveltype,
   Rope.ropeNumber, BlackHole.BlackHoleIndex, HookshotInstant, ShootQuantum,
   ReplaySession 2, SteamManager.startParameters, Host per-match fields,
   RlObservation ids/nextId, RlTelemetry, RlBridge's per-episode fields as
   mirrored in `NativeApi`). `Bind(k)` copies in/out around every ABI call
   for world k, together with `World.Current = worlds[k]`. Gameplay source
   stays byte-identical where possible (the port already deviates from the
   Unity reference; `PORTING.md` records each deviation, add this one there).
2. `NativeApi`: handle table instead of the process latch; `bopl_rl_open`
   returns world k; boot (export load, MainMenu bootstrap) happens once per
   runtime, subsequent opens clone the booted state.
3. Fix the 1-in-17 determinism flake first (PORTING.md TODO: suspected
   `Dictionary<string,_>` iteration on a trace-observable path). Multi-world
   parity is measured against N separate processes, and per-process string
   hash seeds would make that comparison flake for the same reason.
4. Gates, verbatim from MANY_WORLDS.md section 4: single-world parity (N=1
   reproduces Route A byte-for-byte), multi-world parity (N worlds with fixed
   seeds == N processes, per world, across resets while others tick, all 5
   maps), trainer smoke, then throughput. Instance ids are per `World` in the
   shim, so the Unity `GetInstanceID` caveat does not apply.

Serial ticking inside one runtime is required (the port serialised every
`Parallel.ForEach`); parallelism is across runtimes, P ~= cores.

**Done (2026-09-09, commit be0d838). What was built differs from items 1-2
above in one way that mattered:** worlds tick *concurrently* inside one
runtime, on the trainer's own threads, rather than serially behind a
`Bind(k)` copy-in/out. The per-match statics are `[System.ThreadStatic]`
(~110 fields, listed in `sim/Gameplay/RlWorldState.cs`), `World.Current` is
thread-static, and `RlWorldState` snapshots the statics into the `World` on
unbind / restores on bind, so a world may move between threads (step thread,
reset helper) but two threads never share a static. The "serial ticking"
sentence above is therefore void; P is 1.

The part the plan did not foresee is assets. The shim's `Material`,
ScriptableObjects and prefab templates are mutated in place by gameplay
(`SteamFrame.Awake` -> `Material.SetColor`; `AchievementHandler` compares
sprite identity against `SteamManager.abilityIcons`), and in a fresh process
each world owns its copies. So `ExportDatabase` now keeps the *parsed inputs*
shared (assets.json / sprites.json slices parsed once, scene JSON cache, type
lookups) and builds the *objects* per world (`ExportDatabase.Store`, hung on
`World.assetStore`). Getting that split wrong in either direction is visible:
shared objects race (a corrupted `Dictionary` in `Material.SetColor` was the
first symptom), per-world parsing made a 32-world boot GC-bound (28 s, 65% of
samples in `WKS::gc_heap`, until the parse cache brought it to 9 s).

How the races were found: the `.so` publish loop (35 s publish + backend_check)
was too slow to bisect with. `sim/Host worlds --n 16 --episodes 3
--abilities 2,4,14` runs N threads, each with its own `World`, through the
same scripted episode (backend_check phase b) and prints per-world checksums
and the first divergent step; it reproduced every race within 15 s and is the
regression test for this section. Then `backend_check worlds <so> N`.

Gates (item 4): N=1 and N=8/16/32 in one runtime print the socket-path
checksums (`5381bf21573b20a6` / `8dc13de2cc8c8e7c`), dirty reset reproduces
the clean one; `worlds` in the managed build agrees over 16 worlds x 3
episodes on maps 1-3 with spawning abilities; puffer smoke passes with 16 and
32 games and its per-update entropies equal `BOPL_RL_WORLDS_PER_RUNTIME=1`.

Numbers (box loaded by run 003, so relative only): RSS 32 worlds in one
runtime 0.94 GB vs 8 runtimes 1.9 GB (`backend_check`); trainer smoke 16
games (32 worlds incl. spares) 2.6 GB peak vs 8.5 GB. Aggregate stepping
cost per decision with 16-32 concurrent worlds 7-16 us against 17 us for 8
separate runtimes - one heap under 32 allocating threads is not free, and
the honest A/B is m2 vs m3 in `bench_trainer_step.sh` when the box is idle.
Server GC was tried for that and rejected (3.5 GB vs 0.56 GB for 16 worlds,
no throughput gain). Item 3 (the 1-in-17 flake) was not needed: the parity
runs never flaked, the flake is likely a socket-path artefact.

## 4. Scale the lockstep (1-2 days, the actual throughput)

With sections 2-3 the step is CPU-bound and `rows/s` grows with games until
`games * compute / cores` dominates. Sweep `--workers` 32 / 64 / 128 / 192 at
`--buffers` 2 and 3, P = 12 runtimes, N = workers / P.

Trainer-side changes this needs (all in the overlay / `train.py`):

- `train.py` caps `--buffers` at 2 and sizes learner rows as 3/2 x workers;
  generalise, and make `minibatch`/`horizon` math follow rows per buffer.
- `BOPL_MAX_BANK_STREAMS` and the bank slot assignment scale with banks, not
  workers, fine as is; check `agent_perm` and the frozen-pool routing
  (`check_routing.py`) at 192 slots.
- GPU observation buffers are sized per vec; confirm cuBLAS handles per
  stream stay within the workspace budget at 4x the rows.

Expected at 128 games: `t_step` ~1.6 ms, 256 rows per buffer-step, ~150k
rows/s (~110k learner). Report the curve against the model in section 0.

**Done 2026-09-09 (`native/rl/puffer/bench_scale.sh`, results in
`native/artifacts/rl/scale/`, idle box, 300 updates each, `--buffers 2`,
`--banks 4`, medians after update 20). The premise was wrong for this box
and the curve is flat:**

| workers | mode | learner/s | rows/s | ms/step env | load |
|---|---|---|---|---|---|
| 16 | one runtime (.so) | 23.2k | 31.0k | 0.83 | 9 |
| 32 | one runtime (.so) | 21.1k | 28.2k | 2.02 | 16 |
| 64 | one runtime (.so) | 20.7k | 27.6k | 4.14 | 23 |
| 64 | one runtime, `OMP_NUM_THREADS=16` passive | 21.6k | 28.8k | 3.94 | 23 |
| 128 | one runtime (.so) | 16.1k | 21.4k | 10.85 | 41 |
| 128 | one runtime, `OMP_NUM_THREADS=16` passive | 16.8k | 22.4k | 10.54 | 35 |
| 32 | socket Host, one process per world | **30.4k** | **40.5k** | 1.34 | - |
| 16 | socket Host (section 3 A/B, m1) | 34.8k | - | 0.50 | - |

Worlds per runtime (`BOPL_RL_WORLDS_PER_RUNTIME`), same shape:

| workers | worlds per runtime | runtimes | learner/s | rows/s | ms/step env |
|---|---|---|---|---|---|
| 16 | all (32) | 1 | 23.2k | 31.0k | 0.83 |
| 16 | 4 | 8 | **29.3k** | 39.0k | 0.62 |
| 16 | 1 | 32 | 30.2k (section 3 A/B; the rerun here swapped, 8.6 GB, and is void) | - | - |
| 32 | all (64) | 1 | 21.1k | 28.2k | 2.02 |
| 32 | 8 | 8 | 27.3k | 36.4k | 1.47 |
| 32 | 4 | 16 | **28.8k** | 38.3k | 1.41 |
| 32 | socket, 1 per process | 64 | 30.4k | 40.5k | 1.34 |

Reading:

- `t_env` grows linearly with games from 16 up: 0.83 -> 2.02 -> 4.14 ->
  10.9 ms. The sim was already `games * compute / cores`-bound at 16 games.
  The box is a Ryzen 5 7600X3D: **6 cores / 12 threads**, not the 12 cores
  section 0 assumed, and a decision under real play costs far more CPU than
  the 21-78 us of the quiet/scripted measurements (below). Adding games adds
  nothing; oversubscribing OMP (128 threads on 12) costs 25%.
- Separate runtimes beat one runtime by a widening margin: at 16 games the
  socket Host is +50% over one runtime (34.8k vs 23.2k), one runtime per
  world +30% (section 3 A/B); at 32 games the socket Host is +44%. Same game
  code, same ticks: the difference is one workstation GC heap under 32-64
  allocating threads. Every gen0 collection suspends every world
  (`ThreadStore::SuspendAllThreads` 2.3% self, `WKS::gc_heap` 12.5%
  inclusive in the 16-game profile below), and collections come as often as
  the *sum* of the worlds' allocation rates demands, so the stall cost grows
  with worlds. Server GC (per-core heaps) was the textbook answer and cost 6x
  RSS for no gain in section 3; the knob that works is
  `BOPL_RL_WORLDS_PER_RUNTIME`: 4 worlds per heap recovers most of the gap
  (+26% at 16 games, +36% at 32) for ~8 runtimes' worth of code pages and
  parse caches. `train.py` now defaults it to 4 when unset.
- `train.py` learned `BOPL_OMP_THREADS` (override for `OMP_NUM_THREADS`) for
  the oversubscription rows; nothing else in this section's trainer-side list
  was needed, the trainer ran 128 games / 192 learner rows per buffer
  unchanged (peak RSS 13.7 GB at 128 games, 256 worlds).

So section 4 as written (more games) is closed for this hardware: throughput
on this box is set by **CPU per decision and per reset**, and the profile
says where it goes. On a many-core box the same sweep should be rerun; the
memory side is solved (~40 MB per extra world in one runtime).

### 4.1 Where a sim CPU-second goes (perf, 16 games, one runtime, steady state)

`perf record -g` on the m2 trainer for 10 s during rollouts, user space only;
86.7% of samples are in `BoplSimNative.so`, 7.2% libcuda, 2.4% libc, 0.8%
libgomp. Inside the .so, inclusive:

```
c_step (the 4 ticks + observe of a decision)          45.5%
  Updater.TickSimulation from step                     ~39.5%
  NativeApi.Observe / RlObservation.Write               5.4%
inproc_reset_worker (background reset thread)         37.6%
  ExportDatabase.LoadSceneByBuildIndex                 20.3%   scene rebuilt from JSON every episode
    PropertyApplier.Apply (reflection SetValue,
      string Tokenize/Split per property)              17.1%
    MonoBehaviour.CallMessage via DynamicInvoke         2.5%
  240 warmup ticks                                     ~16%
GC (WKS::gc_heap::gc1 inclusive, both threads)         12.5%
static_omp_threadmanager (fork/join, barriers)          6.4%
```

Self time, grouped (top 120 symbols = 76% of the .so):

```
physics + fixed-point math   22%   Fix.ClampSinValue 6.2 (top symbol), DetPhysics.Simulate 2.8,
                                   Fix.op_Multiply 2.5, RaycastToClosestRoundedRect 1.3, SimulateRope 0.9
shim / runtime glue          16%   Object.CompareBaseObjects 4.8 (every `== null`), TickSimulationInner 3.4,
                                   GameObject.activeInHierarchy 1.3, TypeCast.IsInstanceOfAny 1.1,
                                   MonoUpdatable.IsEnabled 0.9, Dictionary lookups, Monitor.Enter/Exit 1.3
GC                           14%
misc runtime                 12%   memmove/memset 2.0, __tls_get_addr 0.9, interface dispatch 0.7
gameplay logic                7%
observation encode            3%   MemoryStream.Write 1.6, RlObservation.Write 0.8
presentation leftovers        5%   DetPhysics.renderBeams 2.9 incl, UpdateRopeMesh_parallell 1.2,
                                   RopeMesh.UpdateMesh 0.6, SyncAllPhysicsObjectsToFixTransformPosition 5.6 incl
```

Per episode in this regime (untrained policy, `matches` counter): 83
decisions = 332 played ticks, preceded by a scene rebuild and 240 warmup
ticks. Resets are 38% of sim CPU because the episode is short; they shrink
as policies improve, but the first hours of every run are in this regime.

## 5. Stage 5: cheaper decision (game + shim optimisation, 3-5 days)

The next step. Ordered by measured share; every item is gated on bit-exact
checksums (`backend_check` phases a/b/c, `Host worlds`, the oracle fixtures
in PORTING.md) and on the m2/socket A/B in `bench_scale.sh` at 16 games.
Ambition: 2x rows/s on this box from CPU per decision alone, and closing the
one-runtime penalty so many-worlds stops costing throughput.

1. **Reset: build the scene once per map, clone per episode (20% of CPU).**
   `LoadSceneByBuildIndex` reparses the scene JSON graph through
   `PropertyApplier` (reflection `SetValue`, `Tokenize` string splits per
   property, `DynamicInvoke` for `Awake`/`Start`) on every reset. Keep the
   per-world *template* hierarchy for each map (built once, never started)
   and instantiate the episode's scene from it with the existing
   `CloneRules.CloneHierarchy` (the code path `Object.Instantiate` already
   uses), then run the messages. Gate: identical checksums for a clean
   reset, a dirty reset and a map change; `Host worlds` over all 5 maps.
   Expected: reset 8 ms -> ~1-2 ms warm.
2. **Reset: snapshot after warmup (16% of CPU).** The 240 warmup ticks are
   deterministic given (map, seed schedule, loadouts, colours). Measure how
   many distinct configurations a run visits (`--training-maps 0,1,2`,
   random loadouts); if the seed enters the warmup, snapshot per (map,
   loadout) at tick 0 instead and run only the ticks the seed influences.
   This is the section-5 item from the original plan, promoted by the
   profile. Needs a deep-copy of the world (`RlWorldState` + shim object
   graph); build it on the clone machinery from item 1. Gate: warmup-then-
   play checksum equals restore-then-play for 20 matches per map.
3. **Allocation diet (GC 12.5%, and the one-runtime penalty).** Find the
   per-tick allocators with the managed build (`Host worlds` under
   `dotnet-trace`/`DOTNET_gcServer=0` + `GC.GetAllocatedBytesForCurrentThread`
   per phase; NativeAOT cannot report them). Known suspects from the
   profile: `List<T>` constructions per tick (1.1% inclusive), `HashMap.Clear`,
   `RlObservation` via `MemoryStream`/`BinaryWriter`, `PropertyApplier`
   strings (item 1 removes them), coroutine/message boxing. Target: a quiet
   tick allocates nothing, a spawn-heavy tick < 1 KB. This is also the fix
   for the one-runtime penalty: fewer collections means fewer all-world
   stalls. Re-run the 16/32-game m1-vs-m2 A/B after; if a gap remains, add
   `GCgen0size` at link time (`RuntimeHostConfigurationOption`) and re-test
   a capped Server GC (`GCHeapCount=4`, `GCConserveMemory`) - the section 3
   rejection tested the default heap count only.
4. **`Fix.ClampSinValue` (6.2% self, top symbol).** Reduces the angle with
   29 dependent 64-bit `%` operations by `2*pi*2^60 >> i`. Exact shortcut:
   when `|angle| < (num >> i)` the remaining moduli are identities, so skip
   to the first `i` whose modulus is `<= |angle|` (leading-zero count).
   An angle already in `[-2pi, 2pi)` then costs zero divisions; how often
   that is the case (callers: `Box.SetRotation` 5% inclusive, `FastSin`,
   `Sin`) decides the win, so count it first. Bit-exact by construction;
   fixture-check anyway.
5. **`Object.CompareBaseObjects` (4.8% self).** The Unity `== null` on
   destroyed objects, not inlined by ILC. `AggressiveInlining` on the
   operators and the helper, and audit hot loops in `DetPhysics` /
   `Updater` that null-test every entity every tick (`RemoveDestroyedColliders`
   1.0% inclusive) to test `destroyed` directly.
6. **Presentation still executing (~5%).** `DetPhysics.renderBeams` (2.9%
   inclusive: instantiates `beamAfterEffect` prefabs, `AudioManager.Play`,
   particle colour gradients), `UpdateRopeMesh_parallell` + `RopeMesh.UpdateMesh`
   (1.8%), and whatever in `SyncAllPhysicsObjectsToFixTransformPosition`
   (5.6% inclusive) only feeds rendering. Guard behind `RlHeadless.Render`
   after confirming nothing gameplay-visible reads the results (the ropes'
   mesh is presentation; `transform.position` sync is read by gameplay, keep
   the Fix -> transform part). Gate: checksums, plus a replay comparison
   against the Unity oracle for a beam-heavy fixture.
7. **Updater passes (~6% self across `TickSimulationInner`, `IsEnabled`,
   `activeInHierarchy`, `isActiveAndEnabled`).** The per-tick walk over
   every behaviour re-derives active state through the transform chain.
   Cache `activeInHierarchy` on the `GameObject` and invalidate on
   `SetActive`/reparent; keep enabled `MonoUpdatable`s in a dense list
   maintained on enable/disable instead of filtering each tick.
8. **Observation encode (5.4% inclusive).** `RlObservation.Write` goes
   through `MemoryStream`/`BinaryWriter` into a managed buffer that is then
   copied to the pinned ABI buffer. Write directly into the pinned span with
   `BinaryPrimitives`; the layout is fixed by `bopl_rl.h`, so
   `backend_check` catches any drift.

Measure after each item with the same `bench_scale.sh` shape (16 games,
one runtime, and 16 games socket as the isolated-runtime reference) and
record the perf grouping above so the next item is chosen by share, not by
guess.

### Done 2026-09-09: items 4 and 5 (bit-exact quick wins)

- `Fix.ClampSinValue`: the 29 moduli `num >> i` decrease with `i` and
  `a %= m` is the identity while `|a| < m`, so the loop now skips straight
  to the first modulus <= |angle| (an angle already inside +-2pi divides
  zero times instead of 29). Bit-exact by construction; checked against
  the original on 10^6 random and 2^k-boundary angles in Python emulation.
- `Fix.operator *`: one `Math.BigMul` (exact 128-bit product) and a shift
  instead of four 32x32 partial products summed modulo 2^64; the original
  sum is the same bits 32..95 of the product. `AggressiveInlining`.
- `UnityShim.Object`: `CompareBaseObjects` and the `==`/`!=`/`bool`
  operators marked `AggressiveInlining` so the ubiquitous `== null` stops
  being a call.
- The trace kept diverging from the Unity oracle at tick 0 on
  `<SuddenDeathInProgress>k__BackingField` / `<PlayerTimeScale>k__BackingField`:
  not the maths, but the many-worlds port turned those auto-properties into
  explicit `[ThreadStatic]` fields and `ReferenceCapture.Static` was
  emitting the new names. It now maps them back to Unity's compiler names.

Gates: `sim/Oracle/checkpoint.sh --full` matches the Unity oracle on all 15
fixtures (l0-l4, groups g0-g4, p3); `backend_check` socket / rl / worlds x16
unchanged at `5381bf21573b20a6` `8dc13de2cc8c8e7c`; `Host worlds --n 8
--episodes 2` agree. Throughput (`bench_scale.sh`, 16 games, 4 worlds per
runtime, 300 s, load ~6): 29.3k -> **34.4k learner decisions/s** (+17%),
env share of the rollout step 0.62 -> 0.48 ms. That is more than the two
symbols' 11% self time; inlining the null-compare helps everything around it.
Next by share: items 1-3 (resets 38%, GC 12.5%). Stop when `c_step` per decision on a spawn-heavy loadout is < 100 us
of CPU or the remaining share is spread thin.

## 6. Horizon

- **Rust sim for rollouts.** ~5 us per tick vs the .NET port's ~100+ us of
  CPU per tick under real play. Bow and Platform are unimplemented and the
  standing decision is that Rust is verified against replays continuously
  (`verify_replay.py`), never trusted blind. Only start after stage 5 has
  taken the .NET path as far as bit-exactness allows and the sweep still
  says throughput limits learning.
- **Async actor/learner (PufferLib 5 / Cleanba style).** Overlapping
  training with rollout saves at most the train share, 4% today. Worth it
  only when the network grows enough that `train_seconds` exceeds ~25% of
  the update (roughly hidden >= 512 or several GRU layers). Track the ratio
  in metrics; do not port ahead of need.
- **More cores.** Section 4's sweep is the recipe: on a 32-core box the
  same one-runtime build runs 64-128 games in ~5-10 GB, and the curve should
  bend where `games * compute / cores` says, once stage 5 has made the
  shared heap cheap.

## Order and ownership

1 -> 2 -> 3 -> 4, each in its own rift (`rift create --copy-all --name
<phase>`), each merged only with its gate passed and the A/B recorded under
`native/artifacts/rl/<phase>/`. Keep the socket path working throughout
(evaluation, replays, the dashboard use it). Coordinate `bopl_rl.c` changes
with the trainer owner per MANY_WORLDS.md section 5; nothing here touches
dashboard files.
