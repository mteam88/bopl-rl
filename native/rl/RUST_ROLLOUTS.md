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
  (embedded via `include_str!`). Every authored and spawned platform has its
  extracted `BoplBody`, an `AnimateVelocity` home spring, and dynamic RR-RR
  collision impulse/positional correction. Map 2's AntiLockPlatform uses the
  same body path and matches the reference waypoint trajectory exactly.
- **Episode lifecycle**: 240-tick warmup simulated for real (players fall
  from spawn and land), deaths (environment/arrow/self), stocks with
  respawn holds, first-death terminal at stocks=0, the 7200-tick timeout,
  and sudden-death platform sinking on the extracted curve. Blast-boundary
  deaths are always routed through match bookkeeping, including while an
  ability body is active.
- **Movement**: the verified movement slice (`world.rs`). Registered input is
  consumed by the live body in the current tick, as in Unity. The public
  `Player.Position` is normally the position published by `SlimeController`
  before the later movement/physics work in that pass; the rollout represents
  that report phase separately from the live body.
- **Dash, Bow, Platform** (catalog ids 2/4/14): ported from `sim/Gameplay`
  with measured timings. Bow: transform 30 ticks, loadingFrame =
  passedSprites (1 at load start, then +8/+7/+7), slow-mo aiming physics,
  fire-on-release (early release fires from transform sprite index 2, tick
  11), knockback recoil, arrow gravity -40 (the DetPhysics override, not
  Unity's -9.82), policy-visible animator charge, delayed hurt-owner, the
  20-arrow owner cap, and projectile collision during a dash raycast.
  Platform: anticipation, aim rotation,
  cursor sizing under the area cap, geometry-dependent mass/inertia,
  ejection and same-pass exit reattachment, one-platform-per-player,
  curve-driven old-body removal, transform following, and squish deaths.
- **Observation**: `bopl_layout.h` layout with the .NET entity id
  assignment order (circles, boxes, rounded rects by first-seen), matching
  kinds/layers/owner/flags, slime ability slots, cooldowns and timers
  (spawn timing -1.5 s at warmup tick 62 reproduced), and the
  SlimeController state machine (fall/land 36t/idle/run/stop 6t/turn 12t,
  tangential ground-alignment deadzones). Rewards and telemetry hook up
  through the same header/flag bits the puffer env reads.

## Deliberate .NET quirks mirrored

- `Player.Position` normally reports the body position from before movement
  in the same updater pass, while the slime/entity rows contain the live body.
  Ability callbacks have specific exceptions: Dash publishes after its
  transform, Platform exit publishes the final ejection, and stock selection
  publishes the selected respawn point in that same pass.
- The .NET `State` enum never takes the `ability` value (nothing assigns
  it): state freezes at its pre-ability value during an ability and
  resumes from `fall` after `ExitAbility` ungrounds. Rust does the same.
- Ability cooldown timers start at -1.5 s when slimes wake at warmup tick
  62, so nothing is ready at the first decision (dash ready at 2.53 s in).

## Known approximations (documented divergence)

- Ability pickups (`AbilitySpawner`, drops at 40 s intervals) are not
  simulated; only ids 2/4/14 have real behavior, other catalog ids tick a
  plausible cooldown and never activate. Besides removing a major game system,
  this leaves the shared Unity RNG sequence different after the first spawner
  event, so late sudden-death random delays cannot remain sample-for-sample
  identical. Closing this requires pickup bodies/collisions and the remaining
  catalog abilities rather than a timing-only patch.
- Flying arrows use a swept ray against the supported hurtboxes and platforms;
  Unity integrates the arrow circle and then uses discrete DetPhysics contacts.
  The scripted ordinary hit matrix currently agrees on every hit/death, while a
  deliberately high-speed tunnelling edge can differ. A faithful replacement
  should use the generic circle/RR, circle/box and circle/circle contact order
  and preserve callback destruction semantics.
- Landed arrows are freed from the native entity table instead of remaining
  as attached visual children. The gameplay-relevant projectile collider is
  destroyed in both backends and the 20-arrow live-projectile cap is enforced,
  so the remaining difference is primarily observation/entity history.
- The rollout only contains dynamic bodies needed by ids 2/4/14. Forces from
  unsupported abilities (for example Machoman impulses into a platform) and
  their circle/box/rope collision pairs remain absent.

## Gameplay fidelity audit (2026-09-10)

| Severity | Gap | Result |
|---|---|---|
| exploitable | Spawned/map platforms were immutable surfaces, allowing a transformed player to phase a platform through terrain | fixed: extracted bodies, springs, RR-RR contacts, impulses/correction, riders and transform following |
| exploitable | A movement body crossing a blast boundary was pre-marked dead, then skipped by match death bookkeeping | fixed: the marker is consumed once and crossing on the current tick is caught |
| exploitable | Dash raycasts ignored arrows even though the Unity collision mask includes Projectile | fixed: ordered arrow hits before the first wall now destroy the arrow and kill when owner rules allow |
| behavior-changing | Resizing kept the prefab mass/inertia instead of `10 + MassPerArea * area` and `ApproximateMOI` | fixed, including live observation `inverse_mass` |
| behavior-changing | Bow parsed but ignored `maxNumberOfArrows = 20` | fixed with per-owner oldest-projectile eviction |
| behavior-changing | Ability entry/spawn and Dash/Bow/Platform updater phases were delayed or collapsed into the public player position | fixed: current input drives live bodies immediately while Unity's component-specific report phases remain explicit |
| behavior-changing | Bow used the original slime hurtbox during transforms, lost ordinary-slime pushes, raycasted after aerial control, and exposed charge one animator pass early | fixed with active ability colliders and source updater/animator ordering |
| behavior-changing | Respawn point/timing choice was absent and fast-spawn upward motion lagged one pass | fixed: three directional points, early jump confirm, exact selection publication and launch phase |
| behavior-changing | Curve tangents and old-platform shrink removal were absent | fixed: exported Hermite tangents/exact evaluator and collidable `being_removed` bodies |
| behavior-changing | Rust arrow flight is swept while Unity projectile contacts are discrete | open plan described above; common scripted hit matrix agrees |
| exploitable | Ability pickups and all abilities other than 2/4/14 are absent | open scope described above |
| cosmetic/observation | Stuck arrow children are not retained | not changed; no live collision remains in Unity either |

## Fidelity gate

`native/rl/fidelity_gate.c` drives both backends with identical seeded
sticky action streams (2 players, abilities 2/4/14, maps 0-4, stocks 0)
and compares short-horizon divergence plus outcome distributions.
The A/B loader must keep pools library-specific. Before commit `1cd34b2`, a
second in-process backend could accidentally reuse the first library's runtime
and turn an intended .NET-vs-Rust check into .NET-vs-.NET.

At 300 episodes with that harness fixed (seeds 1000-1299), the base
(`7f6761b`) versus the final platform/phase implementation was:

| | .NET reference | Rust before | Rust after |
|---|---:|---:|---:|
| mean length (decisions) | 63.0 | 73.4 | 66.5 |
| timeouts | 0 | 0 | 0 |
| environment deaths | 133 | 144 | 153 |
| self deaths | 165 | 153 | 146 |
| opponent deaths | 2 | 3 | 1 |
| dash activations | 221 | 284 | 240 |
| bow activations | 498 | 560 | 510 |
| platform activations | 86 | 129 | 113 |
| arrows fired | 484 | 551 | 497 |
| arrow hits | 2 | 3 | 1 |
| winner agreement | - | 267/300 | 284/300 |

The episode-length gap fell from 10.4 to 3.5 decisions (66%). Mean position
divergence remains under 0.002 through decision 16, then grows chaotically
(0.083/0.198 at decisions 31/46 after the change).

Scripted A/B traces (`native/rl/research/rust_ab/*.c`, built against
`bopl_rl.h` the same way as `fidelity_gate.c`) cover movement states, bow
flag/timer and charge launch-speed sequences, active ability hitboxes/pushes,
stock choice, plus platform overlap, recast, landing and map-2 orbiting. Bow
charge has zero flag-mismatch ticks over the measured load sequence. Stock
respawn position/state/flags/timers are identical after the one-tick death
boundary. In `platform_ab`, the orbiter position is identical to 12 decimal
places at every sampled tick and the landing player settles within 0.0001
units. For the overlap scenario, the maximum map/spawned-platform deltas fell
from 1.385/1.743 to 0.047/0.011 units; their final deltas are 0.044/0.002. The
player's maximum/final delta is 0.041/0.041 after matching Platform's
synchronous exit callback (previous maximum: 0.788).

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

For the full change set, an immediately adjacent 2-player/repeat-4/200k-tick
pair, both limited to `CPUQuota=400%`, `CPUWeight=20`, `nice -n 19`, measured
104.9k -> 101.3k stepping ticks/s (-3.5%) and 71.2k -> 68.1k ticks/s including
resets (-4.3%). This was not a quiet box: the live sweep changed available CPU
noticeably even between short runs (a reverse-order adjacent pair measured
98.3k baseline versus 101.7k final), so treat the magnitude as load noise and
only conclude that no large throughput regression was observed. The prior
256-worker learner headline is about 127k learner steps/s; it was deliberately
not rerun against the live sweep.
