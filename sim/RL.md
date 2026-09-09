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

### In-process instead of a process (branch `trainer-step`, 2026-09-08)

The same harness steps the sim inside the trainer when `BOPL_RL_GAME` ends
in `.so`:

```sh
nice -n 19 dotnet publish -c Release -r linux-x64 sim/Native
export BOPL_RL_GAME=$PWD/sim/Native/bin/Release/net8.0/linux-x64/publish/BoplSimNative.so
```

`bopl_rl.c` then `dlopen`s the library once and opens every handle as a
world inside that one NativeAOT runtime (`sim/Native/NativeApi.cs`: handle
table of `RlWorld`, per-match statics `[ThreadStatic]` + `RlWorldState`
snapshot, per-world asset objects over shared parsed JSON - section 3 of
`native/rl/SIM_OVERHAUL.md`), and forwards every ABI call; `reset_begin` runs
the library's synchronous reset on a helper thread so the spare world overlaps
resets with play as before. Training is bit-identical to the socket path (same
losses and match counts for the same seed); `native/rl/backend_check.c
worlds` is the parity test and `dotnet sim/Host/bin/Release/net8.0/Host.dll
worlds --n 16 --episodes 3` the fast managed one. Memory: ~30 MB per world
plus ~0.7 GB per runtime, against ~240 MB per world when each had its own
runtime. Knobs: `BOPL_RL_WORLDS_PER_RUNTIME=N` (pack at most N worlds into a
runtime; further worlds load reflinked copies of the file; `1` restores one
runtime per world), `BOPL_RL_INPROC_SPARE=0` (no spare, every reset stalls the
lockstep ~8 ms), `BOPL_RL_INPROC_KEEP=1` (keep the copies and `.dbg` next to
the library so `perf` can symbolize). All worlds' `Debug.Log` output goes to
the first opened world's log file (one stderr per process).

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

### Worker sweep (unthrottled, 2026-09-08, after the memory fix)

`train.py --steps 2000000 --no-eval --encoder set --hidden 128 --layers 1
--banks 4 --horizon 128 --training-maps 0,1,2 --stocks 0 --seed 73`, no
`nice`, no scope, box otherwise idle (12 threads, 32 GB, ~11 GB used by
other processes). "run" = whole run incl. startup; peak RSS = sum over the
trainer and its game processes, sampled every 2 s.

| workers | buffers | game procs | learner/s run | learner/s median | physical/s run | peak RSS total | per game proc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8 | 1 | 16 | 13,335 | 14,359 | 17,780 | 4.66 GB | 231 MB |
| 8 | 2 | 16 | 14,345 | 15,600 | 19,126 | 4.72 GB | 230 MB |
| 16 | 1 | 32 | 20,574 | 22,677 | 27,432 | 8.61 GB | 239 MB |
| **16** | **2** | 32 | **22,850** | 25,394 | 30,467 | **8.23 GB** | 224 MB |
| 24 | 1 (+eval) | 48 | 19,412* | 25,435 | 25,883 | 11.24 GB | 220 MB |
| 24 | 2 | 48 | 25,265 | 29,155 | 33,687 | 11.84 GB | 226 MB |
| 32 | 1 / 2 | 64 | — | — | — | — | startup timeout |

\* with `--checkpoint-every 1000000 --eval-every 1000000` and the 500M
recipe's evaluation flags (8/4/16 games, 2 eval workers): two evaluation
cycles ran (4.4 s and 4.7 s each, 8 groups, pool still empty) and two
checkpoints were written; the run-level rate includes them.

Reading: buffers 2 is worth ~10% at every width; 8 → 16 workers scales
+70%, 16 → 24 only +10% (rollout 0.116 → 0.152 s per update — the trainer's
per-step latency is the limit now, not the sim). 64 game processes do not
come up within `bopl_puffer.h`'s 30 s startup timeout: 64 boots × ~260 MB
transient on a box with 11 GB already in use swaps (by hand, 48 boots under
24 CPU spinners take 13 s; 64 boots alone did not finish in 2 min). Chosen
for the 500M run: **16 workers, 2 buffers** — 24 is not clearly better for
40% more memory on a shared box. Projected: 500M / 22,850 = 6.1 h of
training, plus evaluations (5 s each with an empty pool, growing with
`--frozen-games` × pool size) — call it ~7 h. Launch recipe:
`native/artifacts/rl/long-run-500m/launch-002-s73-dotnet.sh` (MemoryMax
15G = 8.6 GB peak + 6 GB).

### Trainer step latency (2026-09-08, branch `trainer-step`)

Where the ~0.9 ms per (buffer, step) of the 16x2 run went, from `perf` on the
live 003 trainer and the vector's own timers (now written to `metrics.jsonl`
as `native_perf`: `eval_gpu` = inference graph + action copy-back, `eval_env`
= OpenMP game stepping, seconds per rollout averaged over buffers):

- ~0.45 ms GPU: the per-step cudagraph ran learner + 4 frozen banks as one
  serial chain of ~93 tiny kernels (28 rows per buffer, 4 per bank: pure
  launch latency). The two buffer threads spent ~85% of their CPU spinning
  in `cudaStreamSynchronize` on it.
- ~0.4 ms env: socket round trip to the .NET host plus OpenMP wake-ups.
- One whole core: the main thread hot-spun in `static_vec_omp_step` for the
  entire rollout.

Changes (all in `pufferlib-bopl.patch`; `BOPL_BANK_BRANCHES=0` restores the
serial chain for A/B):

- Banks 1.. run on side streams forked/joined with events, so the captured
  graph has 5 parallel branches. One cuBLAS handle per stream (same
  workspace size, so the same GEMM algorithms).
- `sample_logits` takes the bank's row count. Upstream sized the launch from
  `dec_out.shape[0]`, and the primary policy's rollout activations span the
  whole buffer, so the learner's launch also wrote the frozen banks' rows and
  advanced their Philox states before the banks overwrote them; serial that
  was a hidden order dependence, parallel it was a race. Fixed, serial and
  branched are bit-identical (same checkpoint hash over 20 updates); against
  the old build only the frozen opponents' random draws differ.
- The rollout barrier and the buffer threads' between-rollout wait
  sleep-poll (50 / 20 us) instead of spinning.

Measured on the box while the 003 run (32 game processes) was live, so both
sides are contaminated (load 27, GPU time-sliced with the other context):
w8x2 `eval_gpu` 0.46 -> 0.12 ms per step; w16x2 interleaved A/B x2, medians:
serial 16.3k learner/s (gpu 0.52, env 0.87 ms), branched 17.7k (gpu 0.29,
env 0.96 ms), +9%. The env half is CPU-starved under that load; remeasure on
an idle box (`bopl-trainer-step-bench` unit, results in
`native/artifacts/rl/trainer-step/`). Expected idle: GPU 0.45 -> ~0.12 ms,
step ~0.9 -> ~0.55 ms, with the env round trip as the remaining floor.

## Known gaps

- **Memory** (fixed 2026-09-08, `602709e` + `2c0b01c`): the export's
  `assets.json` (37 MB) used to be held as a ~300 MB Newtonsoft tree per game
  process — 570 MB RSS at rest, 745 MB after touching every ability. Now
  `JsonSliceIndex` keeps the file bytes plus a key → byte-range index and
  parses one entry when `BuildAsset` asks; scene trees drop line info, the
  boot scene is not cached, the level cache is capped at 8 maps; `RlMode`
  runs one compacting GC after boot. At ready: heap 389 → 59 MiB, RSS
  548 → 208 MiB; during training ~225 MB/process; `reset_check` (38
  abilities × maps) peak RSS 475 → 359 MiB. Boot still allocates ~260 MB
  transiently, which is what limits how many processes can start at once
  (see the sweep). Oracle unchanged: l0-p2-g4 720/720, l1–l4 as before.
- **Game identity** (`7e0ca2c`): `environment.game_identity` reports
  `backend: dotnet-aot` with the player hash only when neither a Mono nor an
  IL2CPP payload sits next to the binary; `train.py` likewise records only
  `player_binary_sha256`. `evaluate.py` needed no change; `train.py` with
  evaluation enabled completed its eval cycles on the port (below).
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
