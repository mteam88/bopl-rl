# RL.md — the .NET sim as the game process of the PufferLib harness

The NativeAOT build of `sim/Host` is a drop-in replacement for the headless
Unity player under `native/rl/bopl_rl.c`. Nothing on the C/Python side knows
the difference: same argv, same `BOPL_RL_SOCKET` unix socket, same BRL1 v2
protocol, because `RlBridge.cs`/`RlObservation.cs`/`RlTelemetry.cs` are the
game's own files compiled into the port unchanged.

## Build

```sh
# JIT (development; also what `dotnet run --project sim/Host` uses)
nice -n 19 dotnet build sim/Host -c Release
#   -> sim/Host/bin/Release/net8.0/Host

# NativeAOT (what training should use: near-instant startup, 3 ms resets)
nice -n 19 dotnet publish sim/Host -c Release -r linux-x64 -p:PublishAot=true
#   -> sim/Host/bin/Release/net8.0/linux-x64/publish/Host   (~31 MB, self-contained)
```

Reflection roots for the string-driven export loader live in `sim/Host/rd.xml`
(see `sim/Oracle/AOT-SPIKE.md`). The binary locates `native/reference-export`
by walking up from the working directory, then from its own location, so it
must stay inside the repo tree (or be run with the repo as cwd).

## Pointing the harness at it

Every launcher already selects the game through `BOPL_RL_GAME` (explicit
`--game` > `BOPL_RL_GAME` > `Builds/LinuxIL2CPP/BoplBattle.x86_64`):

```sh
export BOPL_RL_GAME=$PWD/sim/Host/bin/Release/net8.0/linux-x64/publish/Host

nice -n 19 native/artifacts/rl/venv/bin/python native/rl/puffer/train.py \
    --output /tmp/run --workers 8 --buffers 1 --steps 3000000 --no-eval

nice -n 19 native/artifacts/rl/venv/bin/python native/rl/puffer/smoke.py --output /tmp/smoke
nice -n 19 native/artifacts/rl/venv/bin/python native/rl/puffer/check_routing.py --output /tmp/routing
native/artifacts/rl/reset_check   "$BOPL_RL_GAME" $PWD/native/fixtures/movement.rep /tmp/rc.log
native/artifacts/rl/sampled_check "$BOPL_RL_GAME" $PWD/native/fixtures/movement.rep /tmp/sc.log
```

`train.py` was changed in one place: the runtime-identity hashes skip the
Unity-only files (`UnityPlayer.so`, `Assembly-CSharp.dll`) that do not exist
next to the port; `config.json` then records only `player_binary_sha256`.

## What the process does (`sim/Host/RlMode.cs`)

`Program.Run` hands over to `RlMode` when `BOPL_RL_SOCKET` is set or the first
argument is `rl`. `RlMode`:

1. Accepts the player's argv: `-batchmode -nographics` are ignored,
   `-logFile <path>` becomes the process log (Console.Out/Error and every
   `Debug.Log*` go there; `bopl_rl.c` points the real stdout/stderr at
   `/dev/null`), the trailing `.rep` is the bootstrap replay. Per-env logs
   land in `BOPL_RL_LOG_DIR/worker-<pid>-<n>[-spare].log`; look there first
   when an env dies. `[rl] ready: heap … workingSet …` marks a successful
   boot, `[rl] quit code=…` a clean exit.
2. Boots exactly like `reset`/`replay` mode: loads the export, loads the real
   MainMenu boot scene (`Program.Bootstrap`), then plays the bootstrap replay
   through `Host.ResetRl` so `Host.hasBeenInitialized` is true on a level
   scene — the state in which the Unity player's `Host.Update` starts calling
   `RlBridge.Pump`. `BOPL_RL_BOOTSTRAP` (`native/fixtures/movement.rep`) is a
   legacy-format replay that `ResetRl` cannot take, so a default 2-player
   level-0 setup is used instead; the first harness reset replaces it anyway
   (`reset_check` proves resets are isolated from prior gameplay).
3. Runs the engine-frame loop until `Application.Quit`: per frame
   `RlBridge.Pump(host)` (= the RL branch of `Host.Update`), then the frame
   boundary used by the oracle-verified `Program.Run` loop — raw
   `Object.Destroy` flush, `Time.frameCount++`, queued `Start`s. `RlBridge`
   decides how many ticks fall between boundaries (`CommandsPerFrame`, plus
   one boundary per scene load so its `loadedFrame` gate lets scene `Start`s
   run before the first warmup tick, as in Unity).
4. Exit code = `Application.quitCode`: 0 after the harness's close command
   (`bopl_rl_close`), 1 if the socket drops without one (`RlBridge` logs
   `RL transport stopped`) or the boot fails (stack trace in the log).

Deliberately not done, to stay on the trace-verified path: `Update` on other
behaviours, coroutine stepping, `LateUpdate`, `Time` advance. None of those
run in the `replay`/`reset` loops that match the Unity traces bit-for-bit;
gameplay is driven by `Updater.TickSimulation` inside `RlBridge`. If a
presentation `Update` ever turns out to be sim-observable the oracle will
say so, and `World.RunFrame` in `sim/UnityShim/World.cs` is the full-frame
alternative.

One process per env, single world, as `bopl_puffer.h` expects (world + spare
process per worker).

## Verified (2026-09-08, HEAD of branch `dotnet-sim`, AOT binary unless noted)

| check | result |
| --- | --- |
| `native/rl/reset_check` (6 players, dirty resets over all 38 abilities incl. Platform 14, 5 maps) | `RESET_ISOLATION_OK`, 5738 exact snapshots, 7 s wall (JIT: same) |
| `native/rl/sampled_check` | `pass`, 459 decisions, 3 phase-1 terminal episodes |
| `native/rl/puffer/check_routing.py` | `ROUTING PASS` (b1 + b2 + timeout reward) |
| `native/rl/puffer/smoke.py` (default: 4 games, 2 banks, 3 updates + rotation, loadout 2/4/14) | `passed`, 85/85 checks, 7.3 s wall |
| RL path vs Unity oracle: `native/artifacts/rl/record` through the port (JIT) with `BOPL_TRACE`, 2 players, ability group 0, 480 ticks, map 0 | `episode.rep` byte-identical to `/tmp/oracle/l0-p2-g0/episode.rep`; trace matches Unity's *controlled* (RL-path) trace **509/509 ticks** with `--ignore-presentation` (the same audited projection Unity needs between its own controlled and replayed runs; unfiltered, the first difference is `QuickSpell.ParticleOnCast` at tick 496, a listed presentation field) |
| `train.py --workers 8 --buffers 1 --steps 3000000 --no-eval` | see "Training smoke" below |

### Training smoke

`BOPL_RL_GAME=<AOT Host> train.py --workers 8 --buffers 1 --steps 3000000
--no-eval --checkpoint-every 1000000`, output under `/tmp`, `nice -n 19`, in
its own `systemd-run --user --scope` (the peace.service cgroup has a 12 GiB
`MemoryHigh`; 17 game processes do not fit under it — see gaps).

Result (status `complete`, 1954 updates, 224.7 s wall, seed 73):

| metric | value |
| --- | --- |
| learner decisions | 3,001,344 (physical 4,001,792) |
| learner decisions/s, whole run | **13,357** (physical 17,809) |
| learner decisions/s per update | median 14,789, mean 14,454, min 1,759, max 21,628 |
| rollout / train seconds per update (median) | 0.100 / 0.003 — still rollout-bound |
| matches / episodes | 20,986 / 31,570 (15,759 wins, 0 timeouts) |
| deaths | environment 12,352, self 2,828, by opponent 668; 3,488 kills, 662 arrow hits |
| reward | `reward_outcome` total −52 over the run (nonzero, near zero-sum as expected); kill/death shaping 0 (off) |
| losses | policy −0.0099 → 0.0011, value 0.0263 → 0.0243, entropy 8.299 → 8.300, gradient norm 0.32 → 0.44, all finite |
| game processes | 16 (8 workers × world+spare), all `quit code=0`; logs contain only the three expected boot lines; working set 548 MB at ready → 670–700 MB at exit (live heap flat at 367 MB) |
| checkpoints | `policy-000000000/001335296/002668544/004001792.bin` written, pool size 1 (no swap in 3M decisions) |

For scale: the Unity player's Gate 0 measurement (`native/rl/MANY_WORLDS.md`)
was 12–14k learner decisions/s flat from 16 to 40 workers with the box
CPU-saturated by the game processes. The port reaches the same rate with 8
workers, at `nice -n 19` in a `CPUWeight=10` scope, on a box already running
other work — the trainer, not the sim, is now the ceiling to look at.

## Known gaps

- **Memory: ~570 MB RSS per game process at rest** (Unity player: ~250–330 MB),
  of which ~360 MB is live managed heap — the export's Newtonsoft `JObject`
  trees (`assets.json` 37 MB on disk → `JProperty` 115 MB, strings 139 MB,
  `JToken.LineInfoAnnotation` 53 MB, …) retained by `ExportDatabase` for
  scene/asset builds. Touching every ability (as `reset_check` does) grows
  the live heap to ~745 MB; it is bounded, not a leak (200 same-map resets:
  flat 363 MB). A forced compacting GC after boot returns ~6 MB, so `RlMode`
  does not bother. Consequences: `--workers 16` = 32 processes ≈ 18 GB; the
  two `check_routing` startup timeouts seen while developing were swap thrash
  from that footprint on a loaded box, not protocol failures. Fix belongs in
  `sim/Export` (parse with `JsonLoadSettings.LineInfoHandling = Ignore` for
  an immediate −53 MB; longer term convert to compact records and drop the
  `JObject`s after the assets are built).
- **`native/rl/research/check_telemetry.py`** records
  `environment.game_identity()`, which requires a Mono `Assembly-CSharp.dll`
  or IL2CPP `GameAssembly.so` next to the binary and raises
  `FileNotFoundError` for the port. Not touched. (`evaluate.py` does not use
  it: run directly on the smoke's final checkpoint with `--game <AOT Host>`,
  `--workers 2`, 4/2/4 games, it completed in 3.7 s and wrote
  `evaluations/004001792/result.json` with replays, visuals and mind dumps —
  so `train.py` without `--no-eval` should work too; the smoke used
  `--no-eval` only to isolate throughput.)
- **Bootstrap replay is not actually played.** The player loads the legacy
  `.rep` through its "Replay" scene; the port substitutes a default setup
  (see above). Only the reached state matters for RL and it is identical.
- **Exit code 1 on an abrupt socket close** without a close command — that is
  `RlBridge`'s own `Application.Quit(1)`; the harness always sends close.
- Boot logs always contain three expected `[error]` lines (Steam unavailable
  in sim, the unexported "Replay" scene, `ParticleSystem.randomSeed`
  overflow warnings from the loader). Anything else is news.
- `sceneCountInBuildSettings - 6` maps and the ability catalog come from the
  export (`native/reference-export`); the harness's `BOPL_ABILITY_COUNT` check
  passed, so both match the Unity build the export was taken from.
