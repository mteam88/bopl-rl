# Performance triage, round 1

Port at merge `957af35` (shim `93bc718` + v3 export), .NET 8 CoreCLR, Release,
`nice -n 19`, level 0 / 2 players / no trace. Profiled with
dotnet-trace (sampled thread time) over 41 in-process episodes plus a
`BOPL_SIM_PROFILE` Stopwatch/GC patch added to THIS rift's `sim/Host/Program.cs`
only (marked `PROFILING PATCH`, reset mode only — do not merge into the shim
rift's Host without review).

## Headline: there is no structural problem — "3600 ticks/s" was JIT warmup

| measurement | ticks/s |
|---|---|
| cold process, 240 ticks (their measurement) | 4,100 |
| cold process, 2400 ticks | 12,000 |
| warm in-process episode 2 | ~17,600 |
| warm in-process episode 5+ (plateau) | **85,000–89,000** |
| Unity player marginal (baseline) | ~5,000 |
| Rust port (baseline) | ~188,000 |

Method: `BOPL_SIM_PROFILE=8 dotnet run -c Release --project sim/Host -- reset
--level 0 --players 2 --ticks 4800` — each profile episode re-runs
`Host.ResetRl` + 4800 ticks in the same process with identical inputs; every
episode simulates fully (`simTicks=4800 gameEnded=False`, identical 5.4 MB
allocated per episode). The 5x ramp from episode 1 to 5 is tiered
compilation + OSR + dynamic PGO promoting the tick loop; `DOTNET_TieredCompilation=0`
doubles cold throughput (20.3k first-episode), `DOTNET_TieredPGO=0` still
plateaus ~79k, so tier-1 recompilation is the main lever, PGO minor.
Post-plateau the port is **17x the Unity player** and within **2.2x of Rust**.

Beware measuring with game-over worlds: with idle players the round ends
between tick 4800 and 9600 (sudden death), and post-end ticks are ~3x cheaper
— long `--ticks` runs inflate the average (24000 ticks reports 28k/s).

## Top-10 inclusive frames (dotnet-trace, 41 resets + 197k ticks in one process)

| % incl | % excl | frame |
|---|---|---|
| 61.6 | 8.6 | `Updater.TickSimulation` (8.6% excl ≈ its own Stopwatch + loop) |
| 51.3 | 3.9 | `Updater.TickSimulationInner` |
| 33.6 | 5.2 | `DetPhysics.Simulate` |
| 22.4 | 22.4 | `Buffer._Memmove` (see attribution below) |
| 19.4 | 1.5 | `JContainer.ReadTokenFrom` (Newtonsoft — scene JSON, per reset) |
| 17.8 | 0.3 | `DetPhysics.SortAndSweepXAxis` |
| 17.5 | 0.0 | `Array.Copy` |
| 16.1 | 0.0 | `Host.ResetRl` |
| 15.9 | 0.0 | `CharacterSelectHandler_online.ForceStartGame` (inside ResetRl) |
| 15.8 | 0.0 | `ExportDatabase.LoadSceneByBuildIndex` (inside ResetRl) |

Also notable: `PropertyApplier.Apply` 11.6%, `Dictionary.Resize` +
`List.set_Capacity` ~9% (all under the loader, attribution below),
`Fix.FastCos`/`FastSin` 5.9% excl combined, `SlimeController.UpdateSim` 5.2%,
`DetPhysics.SyncAllPhysicsObjectsToFixTransformPosition` 4.9%,
`Interop.Sys.PRead` 2.1% (scene file reads, per reset), `Stopwatch.Stop`
1.6% (called from `TickSimulation` — the game's own per-tick self-timing;
Unity pays it too).

**Memmove attribution** (stack-walk over the speedscope conversion):
17.1 of the 22.4 points sit under `SortAndSweepXAxis` via `List<int>.RemoveAt`
— the broad-phase sweep's `activeIds.RemoveAt(num)` mid-list shifts
(DetPhysics.cs:2180). This is the game's own algorithm, identical under
Unity; **not** a shim defect and **not** a fix candidate while bit-exactness
is the goal (element order in the potential-collider lists is
gameplay-observable). The remaining ~5 points are Newtonsoft string/buffer
copies during scene parse. `Array.Sort` itself is negligible (0.04%).

**The workload splits ~60/40 tick-simulation vs world-reload.** Per warm
reset: **~15 ms** (measured by the patch: `resetMs=14.9–17.2`), all of it
JSON re-parse + PropertyApplier + loader dictionary churn, because
`ExportDatabase.LoadSceneByBuildIndex` re-reads and re-parses `map-0.json`
(and `Load` re-parses assets) on every reset. At 4800-tick episodes that is
~20% overhead; at the RL config of 240-tick episodes it would be **~83%** of
compute (15 ms reset vs ~3 ms of warm ticks).

## GC (per 1000 ticks, warm episodes, measured by the patch)

| gen0 | gen1 | gen2 | alloc |
|---|---|---|---|
| 0.00–0.21 | 0.00–0.21 | 0.00 | 1.15 KB/tick |

Effectively allocation-free at steady state; the sporadic gen0+gen1 pair
(~1 per 5k ticks) coincides with loader allocations surviving from the
preceding reset. GC is **not** a suspect.

## Suspect checklist (coordinator's list)

- Reflection message dispatch per frame: **absent**. The Host loop never runs
  `Update`/`LateUpdate`; shim `CallMessage` reflection fires only on lifecycle
  events (instantiate/SetActive/destroy). Nothing reflection-shaped appears in
  the tick-path profile.
- Host frame per tick: **no** — 16 ticks/frame default (matches oracle
  `BOPL_TRACE_BATCH=16`); the frame flushes are invisible in the profile.
- Allocation/GC: **cleared** (table above).
- `UpdateRopeMesh_parallell` / NativeArray shim: 0.08% (`SimulateRopes`) — nil
  on level 0.
- `ReferenceCapture` null checks: not visible (bool gate, cheap).
- Debug.Log volume: not per-tick. But each RESET writes ~430 stderr lines
  ("Could not play non-existent sound" from the empty AudioManager bank at
  spawn) — part of the reset cost, worth silencing (fix 3).
- LINQ in shim GetComponent paths: not visible in the tick profile.

## Fixes, ranked by expected gain

1. **Cache the world per level in `ExportDatabase`** (shim thread). Minimum:
   memoize `JArray.Parse(File.ReadAllText(map.File))` per file — removes the
   Newtonsoft parse + PRead + string copies, ~2/3 of the 15 ms. Better: build
   each level's root set once as a pristine inactive template and
   `CloneRules`-clone it on reset (the Instantiate path already exists and
   preserves order/ids); reset should drop to ~2-3 ms. Gain: ~5x episode
   throughput at the RL 240-tick config, ~20% at 4800 ticks. The sound-bank
   warning silencing (fix 3) rides along.
2. **Keep the process warm** (RL harness contract, document in DOTNET_SIM.md):
   the plateau needs ~15k ticks of one process. Batch episodes in-process
   (the Host already supports it — this rift's `BOPL_SIM_PROFILE` shows the
   pattern). For unavoidably short-lived processes, `DOTNET_TieredCompilation=0`
   doubles cold throughput; publishing Host+Gameplay ReadyToRun (or later
   NativeAOT per AOT.md) removes tier-0 entirely.
3. **Dedupe the AudioManager "non-existent sound" warning** (Host bootstrap or
   shim Debug): ~430 stderr writes per reset for a known-empty bank. One line.
4. **Do not touch the broad phase** (`SortAndSweepXAxis` RemoveAt memmove,
   17% CPU): it is the game's own hot loop and order-observable. Revisit only
   after trace parity, as an opt-in fast mode validated by `diff.py` (swap-
   remove or an interval list would change potential-pair order → gameplay).
5. Non-actions, for the record: per-tick `Stopwatch` is gameplay's own
   (1.6%); `Fix.FastCos/Sin` (5.9%) are already LUT-based and bit-exactness
   forbids substitution; `SyncAllPhysicsObjectsToFixTransformPosition` (4.9%)
   is gameplay.

## Reproduce

```sh
nice -n 19 dotnet build -c Release sim/BoplSim.sln
BOPL_SIM_PROFILE=8 nice -n 19 dotnet run -c Release --no-build --project sim/Host -- \
    reset --level 0 --players 2 --ticks 4800          # per-episode ticks/s + GC + resetMs
PATH="$PATH:$HOME/.dotnet/tools"
BOPL_SIM_PROFILE=40 nice -n 19 dotnet-trace collect --profile dotnet-sampled-thread-time \
    -o /tmp/host-steady.nettrace -- dotnet sim/Host/bin/Release/net8.0/Host.dll \
    reset --level 0 --players 2 --ticks 4800
dotnet-trace report /tmp/host-steady.nettrace topN -n 14 --inclusive
```
