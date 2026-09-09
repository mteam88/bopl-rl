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
- The spare world (`env->spare`, `bopl_spare_open`) is a socket-era overlap
  for 13-33 ms Unity resets. In-process resets are synchronous and ~3 ms;
  16 games resetting once per ~20 s of play stall the lockstep 0.25% of the
  time. Route A opens no spare (`env->spare = NULL` path already exists).
- OpenMP threads call into the runtimes; NativeAOT attaches a thread on first
  entry. Pin nothing; measure whether `OMP_PROC_BIND` matters.
- Logging: `bopl_rl_open`'s log path argument already redirects the shim's
  stderr per world, keep `BOPL_RL_LOG_DIR/worker-<pid>-<k>.log`.

Gate: `check_pipeline.py`-style comparison, obs/rewards/dones/telemetry
identical to the socket path for a fixed seed schedule over >= 20 matches on
all 5 maps; then w16b2 A/B against the socket build on the idle box.
Expected: `t_env` 0.4 -> ~0.15-0.2 ms, `t_step` ~0.55 -> ~0.3 ms.

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

## 5. Deprioritised, with the reason

- **Snapshot-restore resets.** Reset is ~3 ms and one-life games last
  300-1500 decisions; lockstep stall from resets is <1% at 16 games and
  ~2-4% at 128 (resets scale with games, the stall does not shrink). Revisit
  only if the section 4 sweep shows reset stalls above 5% of step time; the
  fix is then a world-level snapshot of the post-warmup state per map.
- **Async actor/learner (PufferLib 5 / Cleanba style).** Overlapping
  training with rollout saves at most the train share, 4% today. Worth it
  only when the network grows enough that `train_seconds` exceeds ~25% of
  the update (roughly hidden >= 512 or several GRU layers). Track the ratio
  in metrics; do not port ahead of need.
- **More buffers as a substitute for more games.** Buffers hide GPU latency
  behind env time and vice versa; once both sides are ~0.15 ms the third
  buffer adds little. Sweep it in section 4 and stop there.

## 6. Horizon: a cheaper tick

Section 4's ceiling is the .NET sim's compute per tick. Two ways past it,
both gated on the section 1 measurement:

- **Shim diet.** Profile a tick in the port: the per-frame passes over all
  behaviours (`Update`/`LateUpdate`, pending Starts, destroy queue,
  coroutines), presentation code still executing behind `RlHeadless.Render`
  guards, and Newtonsoft slices on ability spawn. Likely 1.5-2x, zero
  correctness risk when it stays bit-exact against the oracle fixtures.
- **Rust sim for rollouts.** ~5 us per tick vs ~20-80 us, but Bow and
  Platform are unimplemented and the standing decision is that Rust is
  verified against replays continuously (`verify_replay.py`), never trusted
  blind. Only start after the .NET path is CPU-bound at the section 4
  ceiling and the sweep says throughput still limits learning.

## Order and ownership

1 -> 2 -> 3 -> 4, each in its own rift (`rift create --copy-all --name
<phase>`), each merged only with its gate passed and the A/B recorded under
`native/artifacts/rl/<phase>/`. Keep the socket path working throughout
(evaluation, replays, the dashboard use it). Coordinate `bopl_rl.c` changes
with the trainer owner per MANY_WORLDS.md section 5; nothing here touches
dashboard files.
