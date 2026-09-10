# Rust rollouts

The Rust simulator (`native/src/rollout`, crate `bopl-sim`) implements the
full `bopl_rl.h` ABI, so the trainer can roll out on Rust instead of the
NativeAOT .NET sim. The goal is transferable fidelity, not byte-exactness: a
policy trained on one backend performs the same when evaluated on the other.
The .NET sim stays the reference for evaluation and replay.

## How to run

```sh
cd native && cargo build --release   # produces target/release/libbopl_sim.so
BOPL_RL_BACKEND=rust native/artifacts/rl/venv/bin/python native/rl/puffer/train.py ...
```

`train.py` resolves the .so itself, sets `BOPL_RL_INPROC_SPARE=0` (Rust
resets are ~1 ms, no spare world needed) and hosts every world in one
runtime. Any tool that takes a `--game`/library path (`evaluate.py`, the
fidelity gate, `benchmark.c`) accepts `native/target/release/libbopl_sim.so`
directly in place of `BoplSimNative.so` — the in-process loader in
`bopl_rl.c` sees the same 17 exported symbols.

## What is implemented

- **ABI**: every `bopl_rl_*` entry point including `step_sampled` (bank
  opponents), `reset_begin`/`reset_finish`, telemetry, and BOPLREP2 replay
  export (`bopl_rl_replay`, byte-compatible with `ReplayArchive`, SHA-256
  trailer), so `evaluate.py`'s per-group replay saves work unchanged.
- **Maps 0-4**: geometry, spawn points, prefab constants and animation
  timings are extracted from the reference export by
  `native/tools/extract_rollout_data.py` into `native/data/rollout.json`
  (embedded via `include_str!`). Map 2's AntiLockPlatform orbiter runs the
  real force/mass/airDrag spring integration against a one-tick-lagged home
  position, and carries grounded players.
- **Episode lifecycle**: 240-tick warmup simulated for real (players fall
  from spawn and land), deaths (environment/arrow/self), stocks with
  respawn holds, first-death terminal at stocks=0, the 7200-tick timeout,
  and sudden-death platform sinking on the extracted curve.
- **Movement**: the verified movement slice (`world.rs`) with the .NET
  one-tick input latency (a new action takes effect on the second tick of
  its repeat window) reproduced exactly.
- **Dash, Bow, Platform** (catalog ids 2/4/14): ported from `sim/Gameplay`
  with measured timings. Bow: transform 30 ticks, loadingFrame =
  passedSprites (1 at load start, then +8/+7/+7), slow-mo aiming physics,
  fire-on-release (early release fires from transform sprite index 2, tick
  11), knockback recoil, arrow gravity -40 (the DetPhysics override, not
  Unity's -9.82), delayed hurt-owner. Platform: anticipation, aim rotation,
  cursor sizing under the area cap, ejection, one-platform-per-player.
- **Observation**: `bopl_layout.h` layout with the .NET entity id
  assignment order (circles, boxes, rounded rects by first-seen), matching
  kinds/layers/owner/flags, slime ability slots, cooldowns and timers
  (spawn timing -1.5 s at warmup tick 62 reproduced), and the
  SlimeController state machine (fall/land 36t/idle/run/stop 6t/turn 12t,
  tangential ground-alignment deadzones). Rewards and telemetry hook up
  through the same header/flag bits the puffer env reads.

## Deliberate .NET quirks mirrored

- The .NET observation's bow charge bits (ability_flags bits 4-6) and the
  0x80 fired bit always read zero — its reflection read never sees the
  live loadingFrame even though launch speeds prove the internal frame
  reaches 4. Rust hides its real charge too, so the policy sees the
  training distribution.
- The .NET `State` enum never takes the `ability` value (nothing assigns
  it): state freezes at its pre-ability value during an ability and
  resumes from `fall` after `ExitAbility` ungrounds. Rust does the same.
- Ability cooldown timers start at -1.5 s when slimes wake at warmup tick
  62, so nothing is ready at the first decision (dash ready at 2.53 s in).

## Known approximations (documented divergence)

- Ability pickups (`AbilitySpawner`, drops at 40 s intervals) are not
  simulated; only ids 2/4/14 have real behavior, other catalog ids tick a
  plausible cooldown and never activate.
- Map platforms are static bodies (no spring bobbing under player weight);
  the map 2 orbiter is the exception and is exact. This is the main
  suspected cause of the residual episode-length gap below.
- Respawns place the player at the team spawn after a fixed hold rather
  than the .NET respawn-point selection (not exercised at stocks=0).
- Sudden-death sinking interpolates the extracted curve linearly (no
  Unity tangents), and spawned-platform springs move the platform
  directly rather than integrating the spring.

## Fidelity gate

`native/rl/fidelity_gate.c` drives both backends with identical seeded
sticky action streams (2 players, abilities 2/4/14, maps 0-2, stocks 0)
and compares short-horizon divergence plus outcome distributions.
At 300 episodes (commit 56bd3b6):

- winner agreement 267/300 (89%); mean |Δposition| < 0.01 through decision
  16, ~0.5 by decision 46 (chaotic divergence, expected without
  bit-exactness).
- outcome distributions (reference vs candidate): mean length 63.0 vs 73.4
  decisions, deaths environment 133 vs 144, self 165 vs 153, opponent 2 vs
  3, arrows fired 484 vs 551, arrow hits 2 vs 3, timeouts 0 vs 0.

Scripted A/B traces (`native/rl/research/rust_ab/*.c`, built against
`bopl_rl.h` the same way as `fidelity_gate.c`) match tick for tick on: movement states including curvature-induced stops, bow flag/timer
sequences, charge launch-speed curves (within 2 ticks at every hold
length), and the map 2 orbiter trajectory.

## Transfer test

The 1B-step .NET-trained checkpoint (`long-run-1b/007-s73-1b-free-rows`,
policy-705335296) evaluated with `evaluate.py` on both backends, 24 games
per opponent/map plus 8 per held-out map:

| | .NET | Rust |
|---|---|---|
| total wins | 154/160 | 153/160 |
| held-out maps 3/4 | 16/16 | 15/16 |

Every group is within its Wilson interval of the .NET result. Residual
behavioral gap: Rust episodes average ~890 ticks vs ~640 on .NET, and the
policy fires ~3.6 arrows/min vs ~2.3 (it takes longer to close out the
random opponent). Win rates and death-cause mixes agree.

## Throughput

Single-thread sim-only A/B (`native/rl/benchmark.c`, 2 players, repeat 4,
200k ticks, both measured under the same background load):

| | .NET | Rust |
|---|---|---|
| ticks/s stepping | 45.5k | 116.6k |
| ticks/s incl. resets | 16.1k | 79.5k |
| reset cost | ~9 ms | ~1 ms |
| library startup | 1.1 s | 0.3 ms |

Trainer-scale A/B (`bench_scale.sh` with `BOPL_RL_BACKEND=rust`): see the
results appended below when the box is idle (the sim-only ratio is the
upper bound; the GPU learner becomes the limit near 100k learner/s).
