# Faster simulation: many worlds per Unity process (handoff)

Written 2026-09-07 by thread `thr_wq22hezkbf` for a new thread. Decision by Matthew:
plan and execute "many worlds per Unity process"; keep Unity as the authoritative
game for evaluation, replays, videos and the dashboard; the Rust simulator, if it
ever runs training rollouts, is verified against Unity replays continuously
(`native/rl/verify_replay.py`), never trusted blind. A headless .NET port of the
gameplay C# was scouted and rejected (Appendix, section B: rewrite-class work).

## 1. Why, and what the bottleneck actually is

Training steps 16 IL2CPP game processes in lockstep from the native PufferLib
trainer (`native/rl/puffer/train.py` -> `bopl_puffer.h` / `bopl_rl.c` -> socket ->
`Assets/Scripts/Assembly-CSharp/RlBridge.cs`). Measured facts
(`native/rl/autoresearch/notes.md` lines 99-130, `native/rl/evidence/throughput/`):

- A game's main thread spends ~80% of its time in `bridge:read`, waiting for the
  trainer. Per decision step: ~0.82 ms blocked, ~0.84 ms compute. 32 games used
  4.9 CPU cores in total on a 12-core machine. **The machine is latency-bound
  (serial trainer -> games -> trainer dependency, slowest game gates each step), not
  CPU-bound.**
- Bare tick 0.07-0.13 ms; `EarlyUpdate.UpdatePreloading` ~0.11 ms per tick of
  engine bookkeeping that did **not** shrink when ticks per frame went 16 -> 4096
  (`BOPL_RL_TICKS_PER_FRAME`). So it is per-tick work, not per-frame; more worlds
  per frame will not amortise it. Only leaving the engine removes it.
- Reset 13-33 ms (scene load), ~310 MB RSS per process. With one-life rounds
  (stocks 0, since 2026-09-07) games last 5-25 s, so resets are frequent:
  learner throughput dropped from ~18k to ~13k decisions/s when stocks went 3 -> 0.
- The 400M run (2026-09-07, `null-hypothesis-native/001-s73`) runs at ~13k learner
  decisions/s with 16 workers, 2 buffers.

Therefore the lever is **more concurrent games per lockstep**, and the cheapest
form of that is simply more game processes, until memory or CPU say stop. Many
worlds per process is the way to keep scaling past that (less memory per game,
fewer processes, cheaper resets because no scene load per round), plus it removes
the "slowest of 16 processes" skew inside a process. It does **not** make a tick
cheaper.

## 2. Gate 0 (do this first, with existing binaries, no code change)

When the machine is free (the 400M run ends around 07:30 on 2026-09-08; check
`systemctl --user is-active bopl-null-hypothesis.service`), measure learner
decisions/s on the sustained workload as a function of concurrency:

```
--workers 16 / 32 / 48 / 64   x   --buffers 1 / 2
```

using `native/rl/puffer/train.py` with the 400M config but `--steps` ~3M,
`--no-eval`-equivalent (eval-every larger than steps), fresh output dirs, stocks 0,
episode cap 7200. Record: learner_sps, matches/s, RSS total, CPU cores used,
GPU util. Memory budget: 30 GB total, ~310 MB per game process; 64 processes is
~20 GB and is the practical ceiling today.

Decision rule:
- If learner_sps scales close to linearly to 48-64 workers, many-worlds is about
  memory and reset cost, not urgent; report the curve and ask Matthew before
  writing C#.
- If it flattens (trainer main thread, socket transfer, or GPU rollout), profile
  where, and report; many-worlds does not fix a trainer-side limit.
- If it is limited by process count / memory, proceed to section 3.

Note `--buffers` is limited to 1 or 2 in `train.py`; PufferLib supports more, ask
`thr_6bmdfjijez` (trainer owner) before changing that.

### Gate 0 results (2026-09-07 23:05-23:28 EDT, thr_nzrpteey2z)

Driver: `native/rl/many_worlds/gate0.py` (commit 2b21359) in systemd unit `bopl-gate0`
(`MemoryHigh=27G MemoryMax=29G MemorySwapMax=2G`), 3M learner decisions per point,
`launch-001-s73.sh` config otherwise (`--no-eval`, stocks 0, episode cap 7200, 4 banks,
observation delay 9 = one sampled step of 3+1 ticks per decision). Raw data:
`native/artifacts/rl/many-worlds/gate0/{results.jsonl, <point>/samples.jsonl,
<point>/run/metrics.jsonl}`; table from `native/rl/many_worlds/gate0_report.py`.
Machine: Ryzen 5 7600X3D (6 cores / 12 threads), 32 GB, RTX 5070. Nothing else ran.

| point | workers | buffers | games | learner sps (steady) | matches/s | PSS games GB (max) | PSS/game GB | game cores | trainer cores | GPU util % | MemAvail min GB | rollout s/update | train s/update | mem.high / oom_kill | rc |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| w16-b2 | 16 | 2 | 32 | 13,876 | 110.6 | 8.2 | 0.256 | 7.21 | 1.77 | 35 | 12.2 | 0.211 | 0.0044 | 0 / 0 | 0 |
| w16-b1 | 16 | 1 | 32 | 12,961 | 104.1 | 8.2 | 0.257 | 6.69 | 1.50 | 21 | 12.1 | 0.225 | 0.0044 | 0 / 0 | 0 |
| w24-b2 | 24 | 2 | 48 | 13,587 | 108.6 | 12.3 | 0.255 | 7.39 | 1.60 | 24 | 7.3 | 0.321 | 0.0052 | 0 / 0 | 0 |
| w32-b2 | 32 | 2 | 64 | 12,213 | 98.0 | 16.2 | 0.240 | 6.87 | 1.46 | 19 | 3.4 | 0.480 | 0.0059 | 0 / 0 | 0 |
| w32-b1 | 32 | 1 | 64 | 13,951 | 111.4 | 16.3 | 0.254 | 7.79 | 1.41 | 13 | 6.2 | 0.425 | 0.0059 | 0 / 0 | 0 |
| w40-b2 | 40 | 2 | 80 | 13,552 | 107.7 | 20.1 | 0.239 | 7.84 | 1.46 | 16 | 2.2 | 0.548 | 0.0066 | 0 / 0 | 0 |

"learner sps (steady)" drops the first 20% of decisions (CUDA warm-up, first resets).
Cores are medians of 2 s samples over the steady window (game and trainer cgroup
jiffies). PSS from `smaps_rollup`; RSS overstates by ~30% (shared IL2CPP pages).

**Curve: flat.** 13.0-14.0k learner decisions/s from 16 to 40 workers (32 to 80
game processes); 32 and 40 workers are within noise of 16. Rollout wall time per
update grows linearly with workers (0.21 s -> 0.55 s) while train time is 1-3% of it.

**Where it is spent.** The box is CPU-saturated at every point: 10.3 of 12
logical CPUs busy (user 8.0, sys 1.65, idle 1.6, measured with /proc/stat over
10 s during w32-b1). Game processes hold ~7.8 logical CPUs regardless of how many
there are; the trainer ~1.5. Per-thread breakdown across all 64 game processes
(10 s, w32-b1):

| thread | cores | note |
|---|--:|---|
| `BoplBattle.x86_64` main | 5.83 | simulation ticks (~37k ticks/s total -> ~158 us main-thread per tick; isolated single process measures 105-124 us, the rest is SMT/cache sharing) |
| `AssetGarbageCollector` | 1.07 | one per process, Unity's own; pure per-process overhead |
| `Loading.Preload` | 0.53 | spare processes loading the next level scene |
| `threaded-ml`, `BatchDeleteObjects`, `Job.Worker *` | 0.35 | per-process Unity worker threads |

GPU: 13-35% utilization, falling as workers rise (bigger, rarer forward batches);
1.2-1.8 GB. Not a limit. Socket transfer and the trainer main thread are ~1.5 cores
and do not grow with workers; not the limit either.

**Memory.** 0.24-0.257 GB PSS per game process. 80 processes = 20 GB PSS,
MemAvailable min 2.2 GB, ~1.4 GB of game pages already in swap with visible
swap I/O. 48 workers (96 processes, ~24 GB PSS + trainer + desktop) would thrash
and adds nothing since throughput is CPU-bound; both 48-worker points were
**not run** (decision, not a guard skip; the unit was stopped after w40-b2).
`--buffers 1` vs `2` is a wash once CPU is saturated (b2 +7% at 16 workers, b1
+14% at 32).

**Decision rule outcome.** Branch 2 ("flattens") with a twist: the limit is not
the trainer but *simulation CPU*. More processes only time-slice the same 6 cores;
the memory ceiling (~40 workers) arrives before any scaling benefit. Many-worlds
is therefore worth doing for what it removes, not for concurrency:

- ~1.9 of the 7.8 game cores are per-process Unity overhead (asset GC, worker
  threads, spare-process scene loading). Folding N worlds into one process
  divides that by N: **up to ~+25% ticks/s** on this box if the freed CPU goes
  to main threads, plus whatever less L3/TLB thrash from fewer heaps buys.
- Memory drops ~N-fold (one IL2CPP runtime, one asset set per process), which
  removes the 40-worker cap and the swap risk, and lets the same 6 cores serve
  fewer, fatter processes with fewer context switches (~140k ctxt/s today).
- It does **not** change the main-thread per-tick cost (~105-160 us). Anything
  beyond ~+25% needs cheaper ticks (profile `TickSimulationInner`/DetPhysics via
  `BOPL_RL_PROFILE`) or the in-process Rust stepper (Appendix B), or more cores.

Recommendation: proceed to section 3 as decided, target 4-8 worlds per process,
P = 6-12 processes; measure with the same `gate0.py` grid at the end (section 4.4)
expecting ~+20-30%, and budget separate work for per-tick cost.

## 3. Design: static-state swap + additively loaded level scenes

From the scout (Appendix A). Every level scene carries its own `DetPhysics` and
`Updater`; per-match global state is ~50 static fields in 12 classes; there is no
`FixedUpdate`, no coroutine and no `Physics2D` query in the simulation path; the
tick is driven entirely by `RlBridge.Pump` -> `Updater.TickSimulation`.

Approach (i) in the appendix, chosen over "N offset arenas in one scene" (breaks
bit-exactness through absolute `SceneBounds` and broadphase ordering) and over
"thread a World context through 1,000+ call sites" (2-3k line diff against the
reference source whose hash is the replay identity).

1. `WorldState` object holding the per-match statics listed in A1 (Updater 15
   incl. `random`, `prefabsAdded`, deferred-destroy queues; GameSessionHandler 5;
   GameSession 4; DetPhysics.instance + levelTypeThisFrame; PlayerHandler.instance;
   GameTime.PlayerTimeScale; Constants.leveltype; Rope.ropeNumber;
   BlackHole.BlackHoleIndex; HookshotInstant; ShootQuantum; ReplaySession 2;
   SteamManager.startParameters; Host per-match fields; RlObservation ids/nextId).
   `WorldState.Bind(k)` copies in/out. Bind before every tick of world k, and
   before loading world k's scene so `Awake`-time registration lands in the right
   world.
2. Level scenes loaded `LoadSceneMode.Additive`, one world at a time; components
   attributed by `gameObject.scene`; reset = `UnloadSceneAsync(world k)` +
   additive load (replaces today's `LoadScene(Single)` in `Host.ResetRl` /
   `CharacterSelectHandler_online.ForceStartGame` /
   `GameSessionHandler.LoadNextLevelScene`).
3. `RlBridge` multiplexes N worlds on one socket: commands carry a world index
   (`reset(k)`, `step(k)`, batched step-all). `bopl_rl.c` / `bopl_puffer.h`
   address sub-worlds so one process serves N environment slots.
4. The 3 `FindObjectOfType` sites in the sim path become scene-scoped.

Estimate: ~450-550 C# lines + ~250 C lines. Serial ticking inside a process is
required for determinism (all `Parallel.ForEach` were already serialised), so use
P processes x N worlds with P about the core count; the thread picks P, N by
measurement.

### Design decisions (2026-09-08 00:50 EDT, thr_nzrpteey2z; binding-protocol owner thr_6bmdfjijez is archived, so these are decided here)

Inputs folded in from thr_wq22hezkbf (throughput rift, exp08 profiles): per stocks-0
match the game main thread spends ~51 ms playing ~412 ticks and ~32 ms building the
next round (240 warmup ticks ~18 ms + scene load 14-19 ms) = 38% of game CPU; the
reset share grows with workers (3 ms/match at w16 -> 4-5 ms at w32, share .32 -> .49);
per-tick sim cost is flat at ~67 us with no bit-exact trim worth >5%. So the work is
(a) remove per-process overhead and the memory cap (many worlds), and (b) make a
world's match reset cheap (snapshot-restore of the post-warmup state, item D below).
Matthew's decision (00:26): the 500M run waits for both (a) and (b).

#### A. Wire protocol (BRL1 v2 stays; world-addressed commands are added)

- Frame header unchanged: `{magic, operation, sequence, length}`. Ops 0-6 keep their
  exact payloads and semantics and always address world 0, so an old C client keeps
  working against the new build and the new C library keeps working against the old
  build (needed for the old-vs-new bit-exact check).
- Handshake (op 0) payload grows from 140 to 144 bytes: `+u32 worlds` = number of
  worlds this process allocated (from env `BOPL_RL_WORLDS`, default 1). The C library
  accepts 140 (legacy build, 1 world, legacy ops only) or 144.
- World-addressed ops = legacy op | 0x100, payload = `u32 world` + legacy payload:
  0x101 reset(k) (180 B), 0x102 step(k) (32 B), 0x106 step_sampled(k) (36 B),
  0x103 observe(k) (4 B), 0x105 replay(k) (4 B). Responses echo op and sequence;
  the response payload is the legacy one. Op 4 (close) stays process-wide.
- Responses may complete out of request order: a reset(k) reply is produced only after
  k's scene has loaded (a Unity frame boundary) and its warmup ran, while steps of the
  other worlds keep being answered. The C side therefore demultiplexes replies by
  sequence number (B). Reply order for commands of the *same* world is always the
  request order.
- No batched step-all op in this iteration. Steps are one command per world (N
  outstanding at most); PufferLib's `#pragma omp parallel for` over environments already
  issues them concurrently. Revisit only if the remeasure attributes >10% of trainer CPU
  to per-command syscalls.
- Error frames (op 0xffffffff) carry the failing sequence; a game error in world k
  invalidates only world k (`valid=false` on that handle); a transport error breaks the
  whole process (all its handles).

#### B. C library (`bopl_rl.h/.c`): one process, N handles, per-process demux

- `BoplRl` stays the per-world handle and the whole existing API keeps its ABI and
  contract ("serialize calls for one handle; separate handles can run concurrently").
  New: `int bopl_rl_open_worlds(executable, bootstrap, log, timeout_ms, worlds, BoplRl **out, error, cap)`
  spawns one process with `BOPL_RL_WORLDS=worlds` and fills `out[0..worlds-1]`.
  `bopl_rl_open` == `open_worlds(1)`. `bopl_rl_close(handle)` releases a world; the
  process (socket, child, directory) is closed when its last handle is released
  (refcount under the process mutex). Closing is allowed in any order.
- Process object: socket, child pid, `send_lock`, `reply_lock` + condvar, a table of
  pending requests `{sequence -> handle, delivered flag}`, one "reader" at a time
  (leader/follower: a waiter that finds no active reader receives the next frame,
  copies the payload into the owning handle's buffer, marks it delivered, broadcasts,
  and keeps reading until its own reply is in). Timeouts are per wait, per handle.
- `bopl_rl_reset_begin/finish` map directly onto the pending table (begin registers the
  sequence; finish waits for it). The single-world "spare process" pattern in
  `bopl_puffer.h` is kept only for worlds==1; for worlds>=2 the in-process deferral of
  reset replies hides the scene load the same way, without a second process.
- Legacy build detection: handshake length 140 -> `worlds` must be 1 and the handle
  uses ops 1-6; otherwise the k-ops are used for every world including 0.

#### C. Unity side (`RlBridge`, `WorldState`, additive scenes)

- `RlWorld` (per world, index k): today's RlBridge per-match statics (`loading, ready,
  started, terminal, failed, warmup, remainingWarmup, episodeTicks, episodeLimit,
  resetSequence, inputs, loadedFrame, pendingStep*, recording`) plus `scene` (the
  world's level scene), `oldScene` (previous level scene awaiting unload), the
  `RlObservation` id map and `RlTelemetry` counters.
- `WorldState.Bind(k)`: copies the per-match statics out of the classes into the
  currently bound world's record and the record of k back into the classes (reference
  swap for lists/queues/arrays, value copy for scalars). Field list (all "P" rows of
  A1): Updater 16 (`LoadedALevelThisUpdate, toDestroyBuffer, destroyedMonos,
  destroyedObjects, updatables, toBeAddedUpdatables, addedUpdatables, toBeAddedPrefab,
  prefabsAdded, registeringAPrefab, random, hitStopFrames, SimTimePassed,
  SimTimeSinceLevelLoaded, SimulationTicks, gameHasStopped`), `DetPhysics.instance`,
  `DetPhysics.levelTypeThisFrame`, `PlayerHandler.instance`, `GameTime.PlayerTimeScale`,
  GameSessionHandler (`playerSpawns_readonly, selfRef, GameIsPaused,
  SuddenDeathTimeStamp, SuddenDeathInProgress`), GameSession (`currentLevel,
  isInitialized, playedLevelsBag`), `Constants.leveltype`, `Rope.ropeNumber`,
  `BlackHole.BlackHoleIndex`, `HookshotInstant.player1Ropes`,
  `ShootQuantum.spawnedQuantumTunnels`, `Host.level`, `Host.goToNextReplay`.
  Left global (set at reset time, not read mid-match): `SteamManager.startParameters`,
  `ReplaySession.Playing/Recording`, `GameLobby.*`, `Host.replayFrames`,
  `WinnerTriangleCanvas.instance` (shared canvas, presentation only). Scratch statics
  (`MagnetGun.parryHitboxes/hitBuffer`, the `PhysicsParent[]`/`Vec2[]` buffers) stay
  shared because worlds tick serially inside one process.
  Each swapped static gets an `internal` accessor where it is private today; no
  gameplay call site changes.
- Bind points: before every command handled for world k, before issuing world k's
  scene load (and left bound across that frame boundary so the scene's `Awake` and
  `Start` land in k), and while running k's warmup ticks. `SceneManager.SetActiveScene(k.scene)`
  accompanies every bind of a world that has a scene, so `Object.Instantiate` without a
  parent lands in k's scene (needed for scene-scoped Find and for unload).
- Scene flow per reset(k): (1) bind k; (2) `Updater.PreLevelLoad`/`Host.Init`/
  `ForceStartGame` exactly as today up to the scene load; (3) if k has a scene:
  `DestroyImmediate` its root objects now (their `OnDestroy` bodies touch no per-world
  static, verified for all gameplay classes), keep the empty scene loaded, remember it as
  `oldScene`; (4) `SceneManager.LoadScene(6+level, LoadSceneMode.Additive)` (issued from
  `ForceStartGame` when `RlBridge.Enabled`; the `winnerTriangle` additive load stays
  once per process); (5) return to Unity for the frame boundary; (6) next frame:
  `sceneLoaded` gives k its `Scene` and `loadedFrame`; `UnloadSceneAsync(oldScene)`;
  one frame later (same `Time.frameCount > loadedFrame` rule as today) run k's warmup
  ticks and reply. At most one world's load is issued per frame; further resets queue
  and are issued on the following frames, in request order. The bootstrap match's
  level scene (loaded Single by the `.rep` bootstrap) is the initial `scene` of world 0
  and is unloaded by world 0's first reset like any other old scene.
- `Host.Update` RL gate becomes `RlBridge.Enabled && hasBeenInitialized` (the active
  scene is no longer the level).
- Pump loop: (1) for every world whose load completed, run warmup within the per-frame
  tick budget and reply; (2) pending deferred steps; (3) command reads. Reads are
  blocking only while no world owes a reply that needs a frame boundary; otherwise the
  loop drains what is already available (own receive buffer replaces `BufferedStream`)
  and returns, which prevents the deadlock "game blocks on read while the trainer's
  only unfinished environment waits for a reset reply".
- Scene-scoped Find: the 20 sim-path `FindObjectsOfType<SlimeController|Ability|BoplBody>`
  sites (BananaPeel, StockPips, FreezeRayGun, Mjolnir, Portal, PokeBall, SuperStarEffect)
  and the `ShakableCamera`/`PlayerAverageCamera` sites go through `WorldScene.Find*<T>()`
  = Unity's result filtered to `gameObject.scene == bound world's scene`. Filtering keeps
  Unity's relative order, which is a property of the objects themselves, so N=1 is the
  identity and N>=2 preserves single-world order; gate 2 must include loadouts that hit
  these sites (Mjolnir, Portal, PokeBall, BananaPeel, FreezeRayGun, SuperStar).
- Unity instance IDs stay process-global; they are only dictionary keys and equality
  operands in the sim (verified: no ordering on `instanceId`), and `RlObservation`
  remaps them per episode in first-seen order.
- Same code path for N=1 and N>=2 (additive + WorldState always), so gate 1 (N=1 new
  build vs old build) exercises the new machinery directly.

#### D. Snapshot-restore of the post-warmup initial state (own item, own gate)

Goal: a world's match reset costs neither the scene load nor the 240 warmup ticks.

Feasibility (settled 2026-09-08 01:15 EDT): the post-warmup state does **not** depend on the
episode seed. Source reading by thr_wq22hezkbf (autoresearch/throughput notes 01:10): the only
`Updater.random` consumer before tick 240 is `GameSessionHandler.Init`, which draws one
`RandomFix(0,1)` per non-destroyed ground (in `grounds[]` order) into `randomDelay[]`; the delay
only shifts when each ground arrives (ticks 129-165) and the animate-in curve clamps to exactly
1 at its last key, so every ground lands bit-exactly on `targetPositions`; `randomDelay[]` is
re-read only by `AnimateOutLoop`, which the RL path never reaches. Evidence: their
`native/rl/warmup_check.c` (cherry-picked as bf4c780, plus a fix: it sent 6 actions to a
2-player world, which the C library rejects) run here on the incumbent build,
`native/rl/evidence/many-worlds/warmup-check.out`: 5 maps x 4 seeds, reset state
and state 120 scripted ticks later byte-identical across seeds for every map;
`WARMUP_SEED_INDEPENDENT_OK`. The control with the Random ability (catalog id 1) in every slot
differs, as expected: that ability draws from the seeded stream at `SpawnPlayers`.

Design (outcome A):

1. **Master worlds.** The first match with a given key `(map, players, per-player loadout,
   stocks, mode)` runs the ordinary load+warmup path and, at the end of warmup, its scene is
   promoted to a never-ticked *master* (kept loaded, never stepped) and a fresh clone is made
   for the requesting world. Later resets with the same key restore from the master. Keys whose
   loadout contains the Random ability are never cached (always the ordinary path). Masters
   live in a small LRU (default 8 scenes; the 500M run has fixed loadouts, so at most one per
   map). A miss takes the ordinary path, which stays available and gated, so random-loadout
   runs simply get a lower hit rate.
2. **Restore = live clone.** `Object.Instantiate` of every root of the master scene into world
   k's (empty) scene gives a 1:1 object correspondence; then a reflective *total copy* of every
   instance field of every component and every reachable plain object from master to clone,
   with UnityEngine.Object references inside the master scene remapped to their clones (assets,
   prefabs, DontDestroyOnLoad singletons and the shared WinnerTriangle objects kept), plain
   objects deep-copied with an identity memo, dictionaries/hash sets rebuilt by re-insertion
   (Unity object keys hash by instance id), delegates re-created with remapped targets, native
   buffers (`RopeBodyList`, `RopeMesh` NativeArrays) copied by hand. The per-world statics
   (WorldState record) are copied from the master's record through the same remap
   (`updatables` order, `HierarchyNumber`s, `prefabsAdded`, `Rope.ropeNumber`, ...), and
   `PhysicsBodyList.instanceIdToColliderIndex` plus each collider's `instanceId` are rebuilt
   from the clones' ids. Awake/OnEnable/Start of the clones run before the copy (one frame
   boundary after `Instantiate`), so their side effects are erased by it; the copy is total
   over the world by construction.
3. **RNG rebuild.** After the copy: `Updater.InitSeed(seed)`, then the same loop as
   `GameSessionHandler.Init` (`randomDelay[i] = RandomFix(0,1) * MaxRandomDelay` for each
   non-destroyed ground in `grounds[]` order), which leaves `Updater.random` in exactly the
   state a fresh match has at tick 240 (no other draws before then; gate 3 checks this).
4. **Gate 3** (`reset_check.c` extended, plus a new `restore_check.c`): every observed state of
   a restored match must equal the state of a match started by the ordinary load+warmup path
   in a separate process, for maps 0-4, >=8 seeds, 2 and 6 players, loadouts covering the
   ability families (Mjolnir, Portal, PokeBall, BananaPeel, FreezeRayGun, SuperStar, ropes,
   black hole, quantum tunnel), across repeated restores of the same world and across worlds
   restored from the same master; plus the section 4 replay-hash comparison. Bit-exact or it
   does not ship; a hard blocker (state the copy cannot reach, e.g. native engine state that
   feeds the sim) is reported to thr_wq22hezkbf with an estimate rather than worked around.
5. **Measurement contract** for the remeasure: RlProfile reports per world `reset:load` (scene
   load frames), `reset:warmup` (warmup tick time) and `reset:restore` separately from `tick`
   time, and gate0.py reports reset ms/match next to learner decisions/s. The operating-point
   recommendation for the 500M run is made after D lands, not from the many-worlds-only
   remeasure.

#### E. Trainer plumbing

- `bopl_puffer.h`: env kwarg `worlds` (default 1). Environment i uses process i/worlds,
  world i%worlds; the first environment of a process opens it with `bopl_rl_open_worlds`
  and a vector-wide registry hands the other handles out; process startup stays
  parallel across processes. `worlds>=2` disables the spare. `c_close` releases the
  handle (the process closes with its last handle).
- `native_vector.py`: require `workers % (2*buffers*worlds) == 0` so a process never
  spans two buffers (a process serving two buffers would serialise their rollouts on
  one game thread). With the Elo-pool layout (self-play, historical alternating) an
  even `worlds` puts one of each into every process.
- `train.py`: new flag `--worlds-per-process N` (default 1 = today). `--workers` stays
  the number of games. Log dir gets one game log per process (`worker-<pid>.log`).
- SourceHash: re-pin `Assets/Resources/SimulationReference.txt` with
  `native/tools/pin_reference.py --commit <c>` after the C# commit; the parity gates
  compare states through the C API, not replay files, so the hash change is expected.

## 4. Correctness gates (bit-exact or it does not ship)

The evaluator, replays, videos and the Learning Lab all run the real game, so any
non-exactness is a train/deploy gap. Required, in order:

1. **Single-world parity.** The N-world build with N=1 must reproduce the current
   build exactly: the 12-game comparison used at the native cutover (exact
   fixed-point state hashes, delayed observation hashes, replay bytes; evidence
   format in `native/rl/evidence/native-cutover-game.json`), plus the sparse
   reward / forced timeout / event conservation / reset checks.
2. **Multi-world parity.** N worlds with fixed seeds and scripted inputs must
   produce, per world, byte-identical replays and state hashes to N separate
   single-world processes given the same seeds and inputs, including across
   resets (world k resets while others keep ticking) and across all five training
   maps (0-4). Specifically re-check the five risks in A4: `Updater.random` and
   `prefabsAdded`/`HierarchyNumber` offsets; `Rope.ropeNumber` / `BlackHoleIndex`
   / `RlObservation.nextId`; process-global `GetInstanceID` leaking into
   observations (compare observations, not `ReferenceCapture` traces, which are
   allowed to differ in raw instance ids); unswapped deferred-destroy queues and
   `LoadedALevelThisUpdate`; additive scene-load `Awake` timing vs today's
   `sceneLoaded` + 1 frame + warm-up sequence.
3. **Native trainer smoke** on the new binding shape (`check_game_variant.py`,
   `check_pipeline.py` style: obs/rewards/dones/telemetry identical to the
   single-world path for a fixed seed schedule), then a short training run whose
   metrics match the single-world run at the same seeds for the first updates.
4. **Throughput** on the same sustained workload as Gate 0, reported as learner
   decisions/s against the Gate 0 curve. Success is a measured gain, not a build
   that works.

Replay format (`ReplayArchive`, BOPLREP2) must not change; the dashboard's replay
browser, video renderer and mind dumps depend on it.

## 5. Constraints and house rules

- The GPU and CPU are reserved for the 400M run until it finishes; source work,
  Unity editor scripting and unit-level tests are fine meanwhile, but no IL2CPP
  builds or 16-process benchmarks while `bopl-null-hypothesis.service` is active.
- Work in a rift: `cd /home/mte/Documents/Coding/bopl-re/ripped/ExportedProject &&
  rift create --copy-all --name many-worlds` (copy-all keeps the warm Unity
  `Library/` import cache). Branch before committing. Unity builds need
  `OPENSSL_CONF=~/Documents/Coding/bopl-re/openssl-unity-sha1.cnf`; the IL2CPP
  build recipe is in `native/rl/autoresearch/notes.md` (exp06 onward) and the
  final cutover build is `Builds/LinuxIL2CPP/` in the canonical checkout.
- Ownership: trainer-side C/CUDA/Python (`native/rl/puffer/*`, `bopl_rl.c`) is
  `thr_6bmdfjijez`'s design; coordinate the binding protocol change with it via
  `bb thread tell thr_6bmdfjijez "<msg>"` before editing, and keep the change
  additive (single-world path must keep working). Dashboard files are owned by
  `thr_wq22hezkbf`; nothing here should need them.
- Report to `thr_wq22hezkbf` with `bb thread tell thr_wq22hezkbf "<msg>"` after
  Gate 0 (with the curve and a recommendation), after single-world parity, and at
  the end. Do not launch long training runs.
- Keep Rust in mind but out of scope: the C side you touch (`bopl_rl.c` addressing
  N worlds behind one process) is the same shape a future in-process Rust stepper
  would use; do not design it in a way that assumes a socket per environment.

## Appendix: source scout report (2026-09-07, read-only, rift `csharp-scout`)

### A. Many worlds per Unity process

#### A1. Static mutable state inventory

Method: `rg` for non-`readonly`/non-`const` `static` fields (255 raw hits; `Fix.cs`/`Vec2.cs`/`PhysTools.cs` hits are all static *methods/operators*, not fields — excluded), plus static auto-properties with setters, singletons, `Find*`, `DontDestroyOnLoad`, `[RuntimeInitializeOnLoadMethod]`.

**Per-match state that must be duplicated per world** (P) vs **global config safe to share** (G) vs **scratch buffers safe if worlds are ticked serially** (S):

| Class (file) | # static mut. fields | Fields | Class |
|---|---:|---|---|
| `Updater` (Updater.cs:23-57) | 15 | `LoadedALevelThisUpdate, toDestroyBuffer, destroyedMonos, destroyedObjects, updatables, toBeAddedUpdatables, addedUpdatables, toBeAddedPrefab, prefabsAdded, registeringAPrefab, random (Unity.Mathematics.Random), hitStopFrames, SimTimePassed, SimTimeSinceLevelLoaded, SimulationTicks, gameHasStopped` (+`standardFixedDeltaTime`, `updateComparer` = G) | **P** (all 15; `random` is the sim RNG) |
| `DetPhysics` (DetPhysics.cs:117-133,192,272) | 11 | `instance` (P), 8 layer ints `orbLayer…lethalterrainLayer` (G, from `LayerMask.NameToLayer` in Awake), `levelTypeThisFrame` (P), `framesBeforeDuplicatesStartShaking` (G) | P: 2, G: 9 |
| `PlayerHandler` (PlayerHandler.cs:5) | 1 | `instance` → holds `List<Player>` (all per-match player state: stocks, kills, alive, Position) | **P** |
| `GameTime` (GameTime.cs:8) | 1 | `PlayerTimeScale {get;set;}` (time-stop ability) | **P** |
| `GameSessionHandler` (GameSessionHandler.cs:44-96) | 5 | `playerSpawns_readonly, selfRef, GameIsPaused, SuddenDeathTimeStamp, SuddenDeathInProgress{get;private set;}` | **P** |
| `GameSession` (GameSession.cs:6-12) | 4 | `currentLevel, isInitialized, inMenus, playedLevelsBag` | P (level rotation) |
| `GameLobby` (GameLobby.cs:3-7) | 3 | `isOnlineGame, nrOfAbilities, isPlayingAReplay` | G (same for all RL worlds) |
| `Constants` (Constants.cs:15-42) | 12 non-string | `leveltype` (**P**, set in `GameSessionHandler.Awake`, read via `Constants.isSpaceLevel` → `SceneBounds.WaterHeight`, 22 refs), rest are tunables (G) | P: 1, G: 11 |
| `Host` (Host.cs:21-150) | 13 | `delayBufferSize, recordReplay, host, replaysSaved, replayLevelsLoaded, replayPath, isLoaded, PingMeasurmentsRate, level, disconnectedPlayerName/Id, goToNextReplay` + ~40 instance fields on the `DontDestroyOnLoad` singleton (`replayFrames`, `clients`, `hasBeenInitialized`, …) | P: `level, goToNextReplay, replayFrames, hasBeenInitialized`; rest G |
| `RlBridge` (RlBridge.cs:15-31) | 20 | socket/reader/response buffers (G), `loading, ready, started, terminal, failed, warmup, remainingWarmup, episodeTicks, episodeLimit, resetSequence, inputs, loadedFrame, pendingStep*, recording` | **P** (this class is what would be rewritten) |
| `RlObservation` (RlObservation.cs:12-17) | 5 | `ids, icons, slimes, nextId` | P (`ids/nextId` entity id map) |
| `SteamManager` (SteamManager.cs:35-88) | 7 | `instance, startParameters (StartRequestPacket: level/seed/players), networkClientHandle, …` | `startParameters` P; rest G |
| `Settings` (Settings.cs:13) | 1 | `instance` (ScriptableObject; `Stocks`, `NumberOfAbilities`) | G |
| `BlackHole` (BlackHole.cs:32-77) | 9 | 8 layer ints (G), `BlackHoleIndex` (P, sorting order only) | P: 1 |
| `Rope` (Rope.cs:24) | 1 | `ropeNumber` (uint counter → `RopeBody.number`, used in observation ids; reset in `Host.ResetRl`/`LoadNextLevelScene`) | **P** |
| `HookshotInstant` (:18) | 1 | `player1Ropes` | P |
| `MagnetGun` (:35,126) | 2 | `parryHitboxes` (List, cleared each `StaticUpdate`) — S; `hitBuffer` — S | S |
| `ShootQuantum` (:41) | 1 | `spawnedQuantumTunnels` (List) | P |
| `StockRespawn` (:12) | 1 | `cachedIndicatorPrefab` | G |
| `CharacterSelectHandler_online` (:47-61) | 7 | `localPlayerInit, startButtonAvailable, selfRef, awaitingReadyResponse, hasMods, …` | G for RL |
| `TutorialGameHandler` | 4 | `isInTutorial` etc. | G |
| `CharacterStatsList` | 6 | menu flags | G |
| `Player` (:40) | 1 | `inputVectors` (256-entry LUT) | G |
| `SlimeController` (:104), `StickyRoundedRectangle` (:74), `PlatformTransform` (:81), `Invisibility` (:14) | 1 each | numeric tunables / `AnimationCurve` | G |
| Scratch arrays: `PhysicsTester` (4), `RocketEngine`, `ShootDuplicator`, `Drill`, `DuplicatePlatformEffect`, `TeleportIndicator` (2) | 10 | `PhysicsParent[]`/`Vec2[]` buffers | S |
| `ReplaySession` (:11-15) | 2 props | `Playing, Recording` | P (`Playing` is read by `Host.RestoreReplayInputModes`) |

Totals: **~50 per-match static fields in 12 classes**; the hot ones by call-site count (non-Reference/Rl files):

| Static access | refs | files |
|---|---:|---:|
| `Updater.*` (updatables/SimulationTicks/SimTime*/Random*/RegisterUpdatable/DestroyFix/HitStop…) | 276 | 124 |
| `AudioManager.Get()` (presentation) | 242 | 90 |
| `PlayerHandler.Get()` | 234 | 81 |
| `DetPhysics.Get()` | 200 | 40 |
| `GameTime.*` | 185 | 77 |
| `Constants.*` | 121 | 52 |
| `SteamManager.instance` | 93 | 23 |
| `Settings.Get()` | 81 | 25 |
| `SceneBounds.*` | 61 | 17 |
| `GameLobby.*` | 60 | 19 |
| `GameSession.*` | 33 | 13 |
| `GameSessionHandler.` statics | 23 | 10 |

Singletons: `DetPhysics.Get()` (instance set in `Awake`, DetPhysics.cs:290-292), `PlayerHandler.Get()` (lazy `new`, PlayerHandler.cs:44-51), `Settings.Get()`, `AudioManager.Get()`, `SteamManager.instance`, `GameSessionHandler.selfRef` (set in `Init`, :180), `WinnerTriangleCanvas.instance`. **No static events** in gameplay code. `[RuntimeInitializeOnLoadMethod]`: only `PerfHud.cs:41`. `DontDestroyOnLoad`: `Host.cs:345-346`, `SteamManager.cs:114`, `AudioManager.cs:151`, `SteamFrame.cs:91`, `PerfHud.cs:48`, `MenuNavRepeater.cs:26`. `FindObject(s)OfType`/`GameObject.Find` in gameplay: 23 files; in the sim path only `Updater.ComputeChecksum/SerializeState` (debug), `GameSessionHandler.AnimateOutLevel:777` (`PlayerAverageCamera`), `Host.ResetRl:1165` (`HandleAbilitySelectUI`), `StockPips`, `Beam`, `ControlPlatform`, `Mjolnir`(6), `Portal`(5), `BananaPeel`(3).

#### A2. Match setup / teardown today

- **Scenes**: build order (`ProjectSettings/EditorBuildSettings.asset`): 0 MainMenu, 1 Replay, 2 CharacterSelect, 3 ChSelect_online, 4 WinnerTriangle, 5 Tutorial, **6+ = levels** (`Constants.FirstLevelSceneIndex = 6`; RL `level` index k → scene 6+k → `Assets/Scenes/FFA/Level{k+1}.unity`). Every level scene carries **its own `DetPhysics` and `Updater` components** (verified for all 124 level scenes); `Host` and `SteamManager` live in MainMenu/Replay and survive via `DontDestroyOnLoad`.
- **Reset path** (`RlBridge.Reset` :172-212 → `Host.ResetRl` :1163-1182): builds a `ReplayArchive` setup (seed, level, stocks, mode, players/abilities), `LoadReplay` → `ReplaySession.Load`, sets `GameLobby.isPlayingAReplay=true`, `Updater.PreLevelLoad()`, `Host.Init(new List<Client>(),1,level)` (which calls `Updater.ReInit()`), `BlackHole.OnNewLevel()`, `Rope.ropeNumber=0`, then `CharacterSelectHandler_online.ForceStartGame(playerColors)` (:210-355) which builds the `Player` list, `PlayerHandler.Get().SetPlayerList`, `Updater.InitSeed(seed)` (:219), `GameSession.Init()`, and **`SceneManager.LoadScene(replayLevel + 6)` in `LoadSceneMode.Single`** (+ additive `winnerTriangle` once). `RlBridge` then waits one Unity frame after `sceneLoaded` (:44-46, :97) before running `startTicks` warm-up ticks.
- **Tick loop**: no `FixedUpdate` anywhere. `Host.Update()` (:877-880) calls `RlBridge.Pump(this)` when `RlBridge.Enabled && hasBeenInitialized && buildIndex >= 6`. `Pump` reads up to 4,096 commands per engine frame and for each step calls `Tick` → `players[i].OverrideInputWithNetworkInput(inputs[i])` → **`Updater.TickSimulation(GameTime.FixedTimeStep)`** (Updater.cs:219-354): player input/update, sort+`Init()` new `MonoUpdatable`s by `HierarchyNumber`, `UpdateDuplicatedObjects`, `UpdateSim` for all, `MagnetGun.StaticUpdate`, `DetPhysics.Simulate`, `LateUpdateSim`, rope late/mesh, then **drain `destroyedObjects/destroyedMonos` queues with `Object.DestroyImmediate`** (:339-350). `Updater.Update()` auto-tick is disabled in replay mode (:413). No time scale/`Time.deltaTime` in the sim path (`GameTime.PlayerTimeScale` is a `Fix`).
- **Player/ability instantiation**: `GameSessionHandler.Awake` (:152-176) registers itself, snapshots ground platforms (`Level.transform.root.GetComponentsInChildren<StickyRoundedRectangle>()`), sets `Constants.leveltype`. `Init()` (:178-206) schedules `RoutineQueue` delays (`timeBeforeTwo`=1 s → `SpawnPlayers`, then `timeBeforeGO`=2.5 s). `SpawnPlayers` (:270-362): `FixTransform.InstantiateFixed(PlayerPrefab, pos)` per player, `SetParent(playerSpawnsRoot)`, `InstantiateFixed(ability prefab)` ×3 then `SetActive(false)`, 3× `Object.Instantiate(AbilityReadyIndicators[l])`, `InputUpdater` for local players, `StockPips.CreateForPlayers`. `FixTransform.InstantiateFixed` (:120-177) wraps `Object.Instantiate` in `Updater.Begin/EndRegisterPrefab` so `Awake`-time `RegisterUpdatable` calls get `HierarchyNumber += prefabsAdded + 100000000`.
- **Observation** (`RlObservation.Write` :33-91): reads `PlayerHandler.Get().PlayerList()`, `DetPhysics.Get()` arrays (`circles/boxes/roundedRects.colliders/physicsBodies/physicsParents`, `ropes.ropeBodies`, `beamBodies`, private `totalBeamsLastFrame` via reflection), scans **`Updater.updatables`** for `SlimeController`s, and reads `Updater.SimulationTicks/SimTimePassed`, `GameTime.PlayerTimeScale`, `GameSessionHandler.HasGameEnded()`. Entity ids are keyed on **Unity `GetInstanceID()`** (`PhysicsParent.instanceId`, `slime.gameObject.GetInstanceID()`) remapped through a per-episode dictionary.
- Per-process measured costs (`native/rl/evidence/throughput/reset-profile.txt`): reset 13–33 ms (warmup 0–240), bare tick 0.07–0.13 ms, `UpdatePreloading` ~0.11 ms/tick of engine frame overhead; ~310 MB RSS per game process (`autoresearch/notes.md:75`).

#### A3. Unity engine coupling inside the per-tick path

Grep counts per file (regex families; `Gizmos`/`OnDrawGizmos` are editor-only):

| file | LOC | `.transform` | GetComponent | gameObject/SetActive | Instantiate | Destroy* | Unity lifecycle methods | Coroutine | `Time.` | Physics2D/RaycastHit2D | Resources.Load | Vector2/3/Mathf/Quat | Renderer/Particle/Material/Color/Sprite | AudioManager | Debug. | LayerMask/.layer | RlHeadless.Render |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Updater | 434 | 2 | 4 | 2 | 0 | 6 | 1 (`Update`) | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DetPhysics | 3203 | 11 | 8 | 25 | 5 | 23 | 2 (`Awake`,`OnDestroy`) | 0 | 0 | **2 (`Physics2D.GetIgnoreLayerCollision`)** | 0 | 92 | 80 | 4 | 36 | 67 | 36 |
| PhysTools | 1354 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 | 6 | 0 | 10 | 3 | 6 |
| SlimeController | 1245 | 6 | 20 | 26 | 4 | 0 | 3 (`Awake`,`Update`,`LateUpdate`) | 0 | 0 | 0 | 0 | 6 | 72 | 7 | 3 | 2 | 4 |
| PlayerPhysics | 583 | 10 | 4 | 4 | 0 | 0 | 1 | 0 | 0 | 2 (buffer type only) | 0 | 13 | 2 | 1 | 2 | 1 | 2 |
| PlayerBody | 282 | 4 | 3 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| PlayerCollision | 482 | 4 | 31 | 40 | 6 | 7 | 1 | 0 | 0 | 2 (type) | 0 | 2 | 11 | 3 | 0 | 27 | 0 |
| Player | 643 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 5 | 0 | 0 | 0 | 0 |
| FixTransform | 206 | 5 | 4 | 19 | 6 | 0 | 2 (`Awake`,`LateUpdate`) | 0 | 0 | 0 | 0 | 17 | 0 | 0 | 0 | 0 | 0 |
| DPhysicsRoundedRect | 649 | 10 | 7 | 9 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 10 | 0 | 0 | 0 | 10 | 0 |
| DPhysicsBox | 498 | 3 | 4 | 3 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 9 | 1 | 0 | 0 | 2 | 0 |
| DPhysicsCircle | 438 | 1 | 4 | 4 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 |
| StickyRoundedRectangle | 624 | 2 | 7 | 3 | 0 | 0 | 1 | 0 | 0 | 4 (type) | 0 | 4 | 4 | 0 | 0 | 0 | 0 |
| BoplBody | 346 | 1 | 5 | 7 | 1 | 0 | 2 (`Awake`,`Update`) | 0 | 0 | 0 | 0 | 2 | 4 | 0 | 0 | 0 | 0 |
| Arrow | 144 | 13 | 9 | 3 | 0 | 7 | 2 (`Awake`,`Start`) | 0 | 0 | 0 | 0 | 2 | 0 | 1 | 0 | 9 | 0 |
| BowTransform | 269 | 0 | 9 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 3 | 9 | 3 | 1 | 0 | 1 |
| PlatformTransform | 479 | 0 | 14 | 0 | 3 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 25 | 3 | 0 | 0 | 0 |
| ResizablePlatform | 142 | 9 | 9 | 8 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 6 | 2 | 0 | 0 | 0 | 2 |
| Dash | 293 | 8 | 9 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 11 | 14 | 1 | 2 | 5 | 2 |
| Ability | 170 | 0 | 5 | 4 | 0 | 0 | 2 (`Awake`,`LateUpdate`) | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 |
| InstantAbility | 143 | 0 | 3 | 6 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 3 | 1 | 0 | 0 | 0 |
| GameSessionHandler | 831 | 7 | 19 | 33 | 5 | 0 | 2 (`Awake`,`Update`) | 0 | 0 | 0 | 0 | 8 | 19 | 10 | 3 | 0 | 2 |
| StockRespawn | 227 | 1 | 3 | 4 | 1 | 2 | 1 | 0 | 1 | 0 | 0 | 2 | 4 | 1 | 0 | 0 | 0 |
| Rope | 276 | 2 | 3 | 1 | 0 | 4 | 2 (`Awake`,`Update`) | 0 | 0 | 0 | 0 | 16 | 14 | 0 | 3 | 0 | 3 |
| Vec2 / Fix | 248 / 42156 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 / 0 | 0 | 0 | 0 | 0 | 0 |

Findings:
- **Transforms**: sim positions live in `FixTransform.position/rotation` (`Vec2`/`Fix`); Unity `Transform` is written only in `FixTransform.LateUpdate → SyncTransform` (:110-118) and `SetScale_DPhysicsOnly` (:99-108). But **hierarchy is read for gameplay**: `transform.parent` (`Arrow.cs:39,56` layer/tag checks for landing), `SetParent`/`GetComponentsInChildren` (attachment of arrows/platform children, `GameSessionHandler.Awake`, `Updater.DestroyFix`), `CompareTag("ResizablePlatform")` in 7 files (`DetPhysics.cs:1257`, `BoplBody.cs:212`, `Arrow`, …).
- **Physics2D**: no Unity collision queries; only `Physics2D.GetIgnoreLayerCollision` in the broad phase (`DetPhysics.cs:2183,2252`) → depends on `ProjectSettings/Physics2DSettings.asset` `m_LayerCollisionMatrix` and `LayerMask.NameToLayer` (130 call sites; `TagManager.asset`). `RaycastHit2D[]` appears only as an unused buffer parameter type.
- **Time**: `Time.deltaTime` only in `Updater.Update` (disabled in replay) and `StockRespawn.cs:190` (indicator scale animation, presentation). `Environment.TickCount` only for local replay seeding (`Updater.cs:245`, not RL path).
- **Coroutines**: `StartCoroutine` only in `CharacterStatsList.cs` (5) and `MainMenu.cs` (1) — none in gameplay. Sim-time sequencing uses `RoutineQueue` (Fix-based).
- **Instantiate/Destroy during ticks**: yes, heavily. `FixTransform.InstantiateFixed` 76 calls/48 files; raw `Object.Instantiate` 90 calls/47 files (in sim path: indicators, beam/duplicator effects, `InputUpdater`); `Updater.DestroyFix` 94 calls/57 files (deferred to end-of-tick `DestroyImmediate`), raw `Object.Destroy` 28/19 files. Prefab `Awake` ordering (Unity) determines `RegisterUpdatable` order; `HierarchyNumber` (serialized per component + `prefabsAdded` offset) then sorts `toBeAddedUpdatables` (Updater.cs:270).
- **Resources.Load**: once, `ReplaySession.SourceHash` (:13), not per tick.
- **Active state**: `DetPhysics.IsActive(pp)` = `fixTrans.gameObject.activeInHierarchy && collider.enabled` (:1658-1665); `MonoUpdatable.IsEnabled()` = `isActiveAndEnabled`. Abilities are gated by `SetActive(false/true)` (SlimeController 26, PlatformTransform 12, Dash 8 active/enabled checks).
- **Render stripping already present**: `RlHeadless.Render` (`SystemInfo.graphicsDeviceType != Null`, RlHeadless.cs) guards **121 sites in 29 files** (DetPhysics 36, Tools 13, Roll 8, PhysTools 6…). The game is launched `-batchmode -nographics` (`bopl_rl.c:239`). Env switches: `BOPL_RL_SOCKET`, `BOPL_RL_PROFILE`, `BOPL_STOCKS`, `BOPL_FORCE_LEVEL`, `BOPL_FORCE_ABILITY0`, `BOPL_EXTRA_PLAYERS`, `BOPL_TRACE*`, `BOPL_KERNEL_*`, `BOPL_DEMOLOG`.

#### A4. Smallest change to host N worlds, with estimates

Three candidate designs:

| Approach | Mechanism | Lines touched (est.) | Main risk |
|---|---|---:|---|
| **(i) Static-state swap + N additively loaded level scenes** | Keep all statics; wrap the ~50 per-match statics (Updater 15, DetPhysics 2, PlayerHandler 1, GameTime 1, GameSessionHandler 5, GameSession 4, Constants.leveltype, BlackHole.BlackHoleIndex, Rope.ropeNumber, HookshotInstant, ShootQuantum, ReplaySession 2, SteamManager.startParameters, Host per-match fields) into a `WorldState` object; `RlBridge` calls `WorldState.Bind(k)` (copy in/out) before every tick and before loading world k's scene (so `Awake`-time `DetPhysics.instance=this`/`RegisterUpdatable` land in the right world). Load levels `LoadSceneMode.Additive`, one at a time, tag components by `gameObject.scene`; reset = `UnloadSceneAsync(world k)` + additive load. Rewrite `RlBridge` to multiplex N worlds on one socket (`reset(k)/step(k)`) and extend `bopl_rl.c`/`bopl_puffer.h` to address sub-worlds. | C#: Updater ~80, other statics ~60, Host.ResetRl/ForceStartGame/GameSessionHandler.LoadNextLevelScene scene plumbing ~100, RlBridge ~200, `FindObjectsOfType` sites 3 → scene-scoped ~15; **~450–550 C#** + **~250 C** | missed static (any of the 12 classes), `Object.DestroyImmediate` of another world's objects if `Updater.destroyedObjects` isn't swapped, overlapping arenas are harmless for `DetPhysics` (custom) but `FindObjectOfType<PlayerAverageCamera>` and `WinnerTriangleCanvas.instance` become ambiguous |
| (ii) N offset arenas in one scene, one `DetPhysics`/`Updater` | Shift each world by a large `Vec2` offset; share physics lists | not viable | `SceneBounds` constants (blast zone, `WaterHeight`) are absolute (`SceneBounds.cs:5-21`, 61 refs); broadphase sort-and-sweep ordering, `AABB.id` = list index, `HierarchyNumber` tie-breaks and `RaycastAll` result order would all change → **not bit-exact** |
| (iii) `World` context threaded through every static access | Replace `X.Get()`/`Updater.*`/`GameTime.*` with `world.X` | 1,000–1,300 call sites in ~150 files (276+234+200+185+23+33+22) → **~2,000–3,000 lines** | mechanical but huge diff against the reference source whose hash (`ReplaySession.SourceHash`) is the replay/oracle identity |

**Recommended smallest viable: (i)**. Top risks to bit-exactness:
1. **`Updater.random` (Unity.Mathematics.Random) and `Updater.prefabsAdded/addedUpdatables` counters** — must be per-world; `HierarchyNumber = serialized + prefabsAdded + 1e8` (Updater.cs:148) feeds sort order and `RaycastComparer` tie-breaks (RaycastComparer.cs:12) and `DatedMonoUpdatable.HierarchyNumberIfPermanent`.
2. **`Rope.ropeNumber`, `BlackHole.BlackHoleIndex`, `RlObservation.nextId`** ID counters (observation ids and rope identity).
3. **Unity `GetInstanceID()`** is process-global; today it is only a dictionary key (`PhysicsBodyList.instanceIdToColliderIndex`; `AABB.id` is the array index, comparers use index/shape/hierarchyNumber) so interleaving does not reorder physics — but `RlObservation` and `ShootQuantum.cs:162` compare instance ids, and `ReferenceCapture` traces record them (trace equality vs single-world would differ).
4. **Deferred destruction queues** (`Updater.destroyedObjects/destroyedMonos`, `toDestroyBuffer`) and `LoadedALevelThisUpdate` are ticked at end of `TickSimulationInner` — if not swapped, a world can `DestroyImmediate` another's objects or skip physics on its first tick.
5. **Scene-load timing**: `RlBridge` waits for `sceneLoaded` + 1 frame (`loadedFrame`); additive loads of N scenes must each get their `Awake` with the correct world bound, and `GameSessionHandler.Init` runs from the *first tick* (`toBeAddedUpdatables` order), so warm-up ticks per world must be lockstepped identically to today's single-world sequence. Also `SceneManager.LoadScene(...Single)` currently unloads the previous level for free; additive requires explicit `UnloadSceneAsync` and pooled prefab (`DetPhysics.beamObjectPool` etc.) cleanup.

Memory/throughput note: today 16 processes × ~310 MB RSS; engine frame overhead (`UpdatePreloading` ~0.11 ms/tick) would be amortised over N worlds per frame, but the sim itself is single-threaded per process (all `Parallel.ForEach` were serialised for determinism, DetPhysics.cs:316-325).

---

### B. Running the gameplay C# outside Unity

#### B1. UnityEngine type surface in the simulation-central classes (34 files)

| UnityEngine type | refs | Nature |
|---|---:|---|
| `Vector3` | 167 | trivial (float conversions, gizmos, transform sync) |
| `Color` | 99 | trivial/presentation (beam colour mixing in `DetPhysics.BeamColors` uses `Color.RGBToHSV/HSVToRGB` — float, gameplay-visible for beams only) |
| `Debug` | 63 | trivial |
| `GameObject` | 51 | **structural** (`SetActive`, `activeInHierarchy`, `CompareTag`, `layer`, `GetInstanceID`) |
| `LayerMask` | 47 | data (`NameToLayer`, 130 sites assembly-wide; needs TagManager data) |
| `SpriteRenderer` | 38 | presentation, but `sprite` identity is read for gameplay (below) |
| `Vector2` | 35 | trivial |
| `Gizmos` | 33 | editor-only |
| `ParticleSystem` | 23 | presentation (DetPhysics beam/duplicator effects) |
| `Mathf` | 22 | trivial |
| `Object` (Instantiate/Destroy/DestroyImmediate) | 20 | **structural** |
| `Quaternion` | 18 | trivial |
| `Sprite` | 16 | **structural-ish**: `AbilityReadyIndicator.GetPrimarySprite()` identity selects abilities for clones/revive (`PlayerCollision.cs:257,273`, `SlimeController.cs:298,326`) |
| `MonoBehaviour` / `Component.GetComponent*` | 8 / ~230 | **structural** (component graph lookups everywhere; PlayerCollision 31, SlimeController 20, GameSessionHandler 19) |
| `RaycastHit2D` | 8 | type-only buffers |
| `SceneManager` | 7 | structural (GameSessionHandler) |
| `Material`, `Shader`, `MeshRenderer/MeshFilter`, `Animator` | 6/1/4/2 | presentation |
| `Time` | 2 | trivial |
| `Physics2D` | 2 | data (`GetIgnoreLayerCollision` layer matrix) |
| `AnimationCurve` | 1 | data: `AnimationCurveFixed.Init` converts Unity keyframes to `Fix` once (AnimationCurveFixed.cs:95-108); `Evaluate` is pure Fix |
| `PlayerInput`, `Gamepad` | 2/2 | input (bypassed by `OverrideInputWithNetworkInput`) |
| `UnityEngine.Random` | 2 | `GameSession.RandomBagLevel` (level rotation) — nondeterministic, outside a single round |
| Attributes (`SerializeField`, `HideInInspector`, `NonSerialized`, `RequireComponent`) | 63 | trivial stubs |

`Fix.cs`, `Vec2.cs`, `PhysTools.cs`, `PhysicsBodyList.cs`, `RoutineQueue.cs`, `Player.cs` are essentially engine-free (Vec2 has 6 `Vector2/3` cast operators; PhysTools has debug draw only). `DetPhysics.cs` carries the beam/effect pools (`Instantiate` ×5, `ParticleSystem` ×23, `Color` ×53) interleaved with the solver.

#### B2. MonoBehaviour derivation and lifecycle reliance

- **119 classes derive from `MonoUpdatable : MonoBehaviour`** (MonoUpdatable.cs), **137 derive directly from `MonoBehaviour`**, 7 `ScriptableObject`s (`Settings`, `NamedSpriteList` (ability catalog), `PlayerColors`, `TeamColors`, `BeamColorList`, `AnimationData`, `LocalizationTable`). Every sim-central class except `Updater`-internals, `Player`, `PlayerHandler`, `GameTime`, `GameSession`, `SceneBounds`, `StockRespawn`, `Fix`, `Vec2`, `PhysTools`, `PhysicsBodyList`, `RoutineQueue`, `AnimationCurveFixed` is a `MonoBehaviour`.
- Game logic runs in `Init/UpdateSim/LateUpdateSim/OnDestroyUpdatable` driven by `Updater.TickSimulationInner` — **not** Unity `Update`. Unity lifecycle methods in sim classes: **`Awake` in 20 of the 27 files** (component wiring via `GetComponent`, `Updater.RegisterUpdatable(this)`, `PhysicsParent{instanceId=GetInstanceID(), hierarchyNumber}` construction — e.g. DPhysicsBox.cs:293-305); `Update`/`LateUpdate` in `SlimeController` (:438 alpha, :697 sound), `BoplBody` (:257 force arrow visual), `Rope` (:239 mesh), `Ability` (:114 alpha), `FixTransform` (:110 transform sync), `GameSessionHandler.Update` (:471 async scene-load gate, menu only), `Updater.Update` (local play only). All presentation. **No `FixedUpdate`, no coroutines in gameplay.**
- Structural reliance that *is* game logic: (a) Unity's `Instantiate` → `Awake` order across a prefab's components determines `RegisterUpdatable` order before the `HierarchyNumber` sort; (b) `SetActive`/`activeInHierarchy`/`enabled` as ability and collider gating (`DetPhysics.IsActive`, `MonoUpdatable.IsEnabled`); (c) `transform.parent`/`SetParent`/`GetComponentsInChildren` for platform attachment (Arrow 10 hierarchy refs, ResizablePlatform 3, Updater.DestroyFix); (d) `CompareTag`/`gameObject.layer`; (e) `Object.DestroyImmediate` semantics (`== null` on destroyed objects — RlObservation.cs:100-104 explicitly works around interface-vs-Unity null); (f) `Sprite` identity for clone ability selection; (g) `Physics2D` layer matrix and `LayerMask.NameToLayer`.

#### B3. Map/level geometry source

- Levels are **scene data**, not code. `Level{1..5}.unity` (maps 0–4) contain 67–82 GameObjects, **0 PrefabInstances**, and per level 4/6/8/9/6 platforms (`StickyRoundedRectangle`+`DPhysicsRoundedRect`+`BoplBody`+`FixTransform`+`AnimateVelocity`+`ShakablePlatform`) plus `GameSessionHandler` (`teamSpawns` ×4, `levelType`, `MaxRandomDelay`, animation curves), `DetPhysics` (gravity, rope/beam tunables, layer mask, prefabs) and `Updater`. Platform geometry = `FixTransform._position/_rotation` (serialized `Fix` raw longs) + `DPhysicsRoundedRect` extents/radius; the level "animate-in" moves them from `startPositions` to `serializedPosition` over `LevelAnimationTime` with `Updater.RandomFix` delays (GameSessionHandler.cs:157-176, 692-720).
- **Already extracted as data**: `native/reference-export/map-{0..123}.json` (127 files, schema `bopl-assets-v1`, full component/property dump per scene incl. raw fixed-point ints) + `assets.json` (prefabs), pinned in `manifest.json`; the Rust port embeds `native/data/level1-movement*.json` via `include_str!` (`src/world.rs:93-102`). So yes — the five training maps are extractable and *have been*; what is missing is loading them in a non-Unity runtime for all component types.

#### B4. Consistency with the Rust port's notes

`native/VALIDATION.md` (1,647 lines) and `native/FULL_COMPATIBILITY.md` (1,563 lines) record: bit-exact **Fix math (303,016 cases), geometry (24,000), RNG (65,000 draws incl. `Updater.RandomFix`), ordered `DetPhysics.Simulate` rigid/composite scenes (1,000 scenes/12,000 steps), ropes, the full beam graph/hitbox/effects, `Player.Kill`, `PlayerCollision`, `PlayerPhysics` attach/detach, `SlimeController` locomotion/ragdoll, spawn/revival, scheduler/deferred destruction, Dash, Invisibility clones, Gust, Revival**. Explicitly *not* achieved (FULL_COMPATIBILITY.md:10-17, 112-114; VALIDATION.md:82-94; ability-coverage.json): the public Rust environment is still the "Level1 movement/dash slice"; **Bow (id 4) and Platform (id 14) are `unimplemented`**, Dash is `slice-only`; complete match lifecycle, stocks/respawn, arrows, resizable platforms as bodies and the C environment integration are open. `native/rl/autoresearch/notes.md:570-640` (2026-09-08 assessment) reached the same conclusion this scan supports: Rust world steps ~5.3 µs/tick vs ~150–200 µs Unity main-thread per tick (30–40× per core), and recommends *not* starting the full rewrite yet. The evidence here (engine coupling is in object lifecycle/hierarchy, not in the math) is consistent with why the port needed per-mechanic goldens rather than transliteration.

#### B5. Verdict

The boundary is **not clean**. Trivial references (Vector/Mathf/Debug/Color/attributes ≈ 470 of ~700 UnityEngine references in the sim-central files) are stub-able in a day. The remaining coupling is structural and pervasive: 119 `MonoUpdatable` + 137 `MonoBehaviour` classes whose *registration order* comes from Unity's `Instantiate`/`Awake` sequencing; ~230 `GetComponent*` graph lookups; `SetActive`/`activeInHierarchy`/`enabled` gating; `transform.parent`/`SetParent`/`GetComponentsInChildren` for attachments; `DestroyImmediate` null semantics; prefab/scene asset deserialization (Fix fields, `AnimationCurve`, `NamedSpriteList` catalog, `Sprite` identities, layer matrix). A faithful "headless .NET" run therefore requires a **mini Unity object model** (GameObject/Component/Transform hierarchy, active-state propagation, prefab instantiation with correct Awake order, destroyed-object null semantics, layer/tag tables, YAML/JSON asset loading) — i.e. re-implementing the parts of the engine the gameplay observes, then proving bit-exactness against the existing goldens. That is closer to the Rust port's experience (16,752 lines and still missing Bow/Platform) than to a stub-and-compile job. **Estimate: rewrite-class (months), not stub-and-compile (weeks)**, unless scoped to reuse the Rust port's already-validated math/physics and add only Bow/Platform/stocks.

Five thickest coupling points, with evidence:
1. **Object lifecycle & ordering** — `FixTransform.InstantiateFixed` (76 sites) + raw `Instantiate` (90) + `DestroyFix` (94) + `DestroyImmediate`; `Awake` in 20/27 sim files does registration; `HierarchyNumber` sort (Updater.cs:270).
2. **Component graph** — ~230 `GetComponent*` in sim-central files (PlayerCollision 31, SlimeController 20, GameSessionHandler 19, PlatformTransform 14).
3. **Active-state & hierarchy as game state** — `DetPhysics.IsActive` (:1658), `MonoUpdatable.IsEnabled`, `SetActive` gating (PlatformTransform 12, SlimeController 14, Dash 8), `transform.parent`/`SetParent`/`CompareTag("ResizablePlatform")` (7 files).
4. **Asset data** — scene/prefab-serialized `Fix` fields, `AnimationCurveFixed` from `AnimationCurve`, `NamedSpriteList` catalog + `Sprite` identity used for clone ability selection, `LayerMask.NameToLayer` (130 sites) + `Physics2D.GetIgnoreLayerCollision` matrix.
5. **DetPhysics.cs itself (3,203 lines)** — solver interleaved with beam/duplicator effect pools (`Instantiate` 5, `ParticleSystem` 23, `Color` 53, `RlHeadless.Render` 36 guards).

---

### Bottom line

**A (many worlds):** Smallest viable approach is **(i) static-state swap + additively loaded level scenes**: wrap ~50 per-match statics in 12 classes (Updater 15, GameSessionHandler 5, GameSession 4, DetPhysics 2, PlayerHandler/GameTime/Constants.leveltype/Rope.ropeNumber/BlackHoleIndex/HookshotInstant/ShootQuantum 1 each, ReplaySession 2, SteamManager.startParameters, Host per-match fields) into a bind/unbind `WorldState`, change `LoadScene(Single)` → additive+unload in `Host.ResetRl`/`ForceStartGame`, and multiplex `RlBridge` + `bopl_rl.c` over world indices. **~450–550 C# lines + ~250 C lines.** Top 5 risks: (1) `Updater.random`/`prefabsAdded` counters and `HierarchyNumber` offsets; (2) `Rope.ropeNumber`/`BlackHoleIndex`/`RlObservation.nextId` id counters; (3) process-global `GetInstanceID` leaking into observations/traces; (4) unswapped deferred-destroy queues / `LoadedALevelThisUpdate`; (5) additive scene-load `Awake` timing vs. today's `sceneLoaded`+1-frame+warm-up sequence. Offset-arenas-in-one-scene is ruled out by absolute `SceneBounds` and broadphase ordering.

**B (headless .NET):** **Rewrite-class (months), not stub-and-compile (weeks).** ~2/3 of UnityEngine references are trivial, but the sim's correctness depends on Unity object lifecycle/order, component graph, active-state, hierarchy, and asset deserialization. The five thickest points: object lifecycle & registration order; `GetComponent` graph; active-state/hierarchy-as-game-state; asset data (Fix-serialized scenes, curves, sprite-identity catalog, layer matrix); `DetPhysics.cs` solver/effects entanglement. This matches `FULL_COMPATIBILITY.md` (Bow/Platform unimplemented in Rust) and the 2026-09-08 note recommending against the rewrite until throughput is shown to limit learning.

