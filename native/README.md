# Bopl native foundation

A standalone Rust simulation of **Level1 with one through six equal-size slimes, movement,
jump, dash, player shoves, and the first player death**. It exposes a small C ABI
and Python/Gymnasium adapters. Unity is needed to generate fresh reference data,
not to build or run the native environment.

This is the first verified slice of the port. It is **not the complete game** and
there is no trained bot or PufferLib training integration yet.

The full-game expansion is active: [scope and progress](FULL_COMPATIBILITY.md).
The Gym adapter still controls a two-player environment; the C and ctypes APIs
support all six players in the movement slice.

## Run

From the Unity project root, with Rust, Python 3, a C compiler, and `just`:

```sh
just --justfile native/justfile verify
just --justfile native/justfile benchmark
```

`verify` builds the library, runs Rust checks, compiles a real C client, compares
complete independent episodes against checked-in Unity goldens through the C ABI,
and checks Python reset/error handling. It needs no Unity installation or .NET.

```python
# PYTHONPATH=native/python
from bopl_native import World

with World() as world:
    world.step([[1, 0], [128, 0]])
    print(world.observe())
    print(world.state())  # full native state and map geometry, raw Q32.32 values
```

The C contract is in [include/bopl.h](include/bopl.h); [examples/smoke.c](examples/smoke.c)
is a compiled example. Each handle owns a separate world. Invalid actions leave
state unchanged. Reset returns to the captured state after tick 240 of the pinned
spawn sequence; it does not run Unity or replay the spawn animation. The native
world stops at the first death. Call reset before stepping again.

Actions use the original byte-angle table: zero means neutral, bytes 1–255
represent `(byte-1)*2π/255`. Button bits are jump=1 and dash=2. All players receive
one action each tick, at the game's exact fixed timestep. Other button bits are
rejected. Use `bopl_create_players(1..6)` or `World(players=6)` to select player count. The C API deliberately provides no unverified map/loadout selector.

## RL access

```sh
python3 -m venv native/artifacts/venv
native/artifacts/venv/bin/pip install -r native/python/requirements.txt
PYTHONPATH=native/python native/artifacts/venv/bin/python -c \
  'from gymnasium.utils.env_checker import check_env; from bopl_env import BoplEnv; e=BoplEnv(); check_env(e, skip_render_check=True); e.close()'
```

`BoplEnv` exposes a `MultiDiscrete([256, 4])` action for player 1. Player 2 is
stationary by default, or supplied by an opponent callback. The observation
includes both players' movement state, dash phase, and cached collision geometry.
The starter reward is survival time with a death penalty. The fixed map and spawn
seed are explicit; Gym's seed affects a supplied stochastic opponent, not the
native reset. This adapter is for locomotion experiments, not combat training.

Gym's API contract is documented [here](https://gymnasium.farama.org/api/env/).
For PufferLib's current native backend, connect its environment buffers to the C
ABI directly. Its [environment tutorial](https://puffer.ai/docs.html) uses C
observation/action/reward/terminal buffers; Gym compatibility should not be
assumed for that backend. The bindings here are tested, but that integration and
policy training are still future work.

## What parity means here

The reference is the reconstructed Unity project at
`67484d097087562123f860e1ed209bcb56e99d2e`, Unity 2022.3.60f1, with opt-in recording
instrumentation. It is not a claim about every Steam release or every historical
Mac build. Reference hashes live in [data/reference.json](data/reference.json),
[data/assets-reference.json](data/assets-reference.json), and the golden headers.

There are three separate checks:

1. Original C# arithmetic against Rust: literal table bytes, wrapping versus
   saturating arithmetic, rounding, trigonometry, and expected arithmetic errors.
2. Actual Unity functions against Rust on generated collision and platform
   perimeter queries, including boundaries one raw unit apart.
3. Unity tick traces against independent native trajectories. Only the reset is
   imported; Rust receives inputs afterward and must produce every subsequent
   state itself. A mismatch reports the first differing tick and field.

The native projection compares positions, orientation and up vectors, imposed
and external velocity, grounded/local state, cooldowns, dash clocks/direction,
platform geometry, cached colliders, and alive masks. It excludes presentation,
inactive original-body scratch state, and the original jump-history flags;
subsequent behavior still tests their effects. At death, destroyed bodies are
compared by alive mask and the surviving players' state. Post-death effects,
ability drops, respawns, round scoring, and the next map are outside the episode.

The optional `reference.py compare --gameplay` view excludes only the fields in
[data/presentation-fields.json](data/presentation-fields.json), each with an
audited source rationale. Raw comparison remains the default and original traces
are retained. Never exclude a difference just because it is inconvenient.

The broader Unity recorder includes managed updatables, active and inactive fixed
transforms, facing, players, RNG, pending update queues, live rigid-body lists,
and session fields. It **does not claim a complete serialization of Unity**:
unlisted physics globals, native rendering internals, and symbolic asset identity
remain limitations. A Unity self-match is useful reference validation; it is not
Rust parity for abilities the port does not implement.

Level1's platforms remain stationary in this slice. They have mass in the full
game. The separate rigid/composite solver now has exact Unity scene tests;
integrating it with movement and ability effects remains in progress. The rope
physics stage, beam graph, hitbox pool and target force/charge components are
verified together in the source physics update order. Shared death accounting and
collision-effect selection have separate Unity checks. Ground attachment and
detachment are shared with the movement world; the remaining death/revival
handlers still need integration. Player-body motion and rope transitions have
coupled component tests and actual replay checks. Ability physics handoff and
grounded rope updates have Unity component checks; gravity and air movement are
shared with this C environment. Animation clocks and callback boundaries now
have exact component tests, including the real spawn frame sequences and fast
respawn conversion. Spawn/revival transitions now have coupled tests with the real ground-loss
handler and animator. Connecting them to deaths, stocks, clones and the full
world scheduler remains in progress. Registration, initialization and deferred
destruction order now have exact Unity checks, including callbacks that mutate
the update queues. A composed Rust runner now connects the scheduler, player controller/body, shared
collider transforms and equal-scale shove callbacks. It keeps native state between
ticks and is checked against the original Unity methods with scripted movement
inputs. An optional SlimeController path now drives locomotion and animation
from keyboard input, compared with the original UpdateSim using empty ability
lists and the exported slime animations. Ragdoll impulses, timed recovery and
invalid-ground detachment now run in this same simulation, with commands applied
between scheduled components to test interactions. Ordinary ability cooldowns
and readiness gates now run there too. Invisibility has an executable instant
handler, including expiration and cooldown recovery. Separate ordinary Ability
actors now execute entry, scale/position updates, and returns through the original
SlimeController callback. An attached Dash component now runs windup, quantized
aiming, wall/air raycasts and automatic returns in that scheduler. Damage and rope
transfer are still absent there; unconnected selected abilities stop with an
explicit unimplemented-entry error. It is not yet driving this C environment. Full player-input
episodes, grappling and beam ability lifecycles, collision-driven damage, time
manipulation, scaling, clones, other abilities, stocks and additional maps still
require ports and comparisons. Equal-size slime shoves use the source's distance condition; the
fixed hurtboxes guarantee overlap within that distance. This shortcut must be
revisited when adding scaled or different player shapes.

## Regenerate and extend the evidence

Build the instrumented Linux player with `UNITY_EDITOR` set to a licensed Unity
2022.3.60f1 executable:

```sh
just --justfile native/justfile unity-build
just --justfile native/justfile export-assets
just --justfile native/justfile verify-math
just --justfile native/justfile verify-geometry

dotnet build native/tools/replay/replay.csproj -c Release
python3 native/tools/generate_scenarios.py native/artifacts/random.json --seed 17
python3 native/tools/verify_scenarios.py native/artifacts/random.json
```

On the PowerSpec, run Unity builds in a separate `systemd-run --user --scope`, and
use the existing `openssl-unity-sha1.cnf` via `OPENSSL_CONF` if required by its
license setup. Do not rebuild the player while a reference capture is running.

```sh
python3 native/tools/reference.py capture path/to/game.rep native/artifacts/a.jsonl.gz --ticks 1200 --batch 1
python3 native/tools/reference.py capture path/to/game.rep native/artifacts/b.jsonl.gz --ticks 1200 --batch 16
python3 native/tools/reference.py compare native/artifacts/a.jsonl.gz native/artifacts/b.jsonl.gz
```

Captures require clean process completion, ordered input/state pairs, and a
matching end marker. They include replay, game-assembly, Unity-player, and trace
hashes. Comparison rejects different input or binary provenance. It preserves
first-difference diagnostics rather than hiding tolerances behind a checksum.
Use a fresh output path; captures refuse to overwrite evidence.

`golden.py export` validates a complete capture, checks the reset, and writes a
small compressed projection. `golden.py verify` consumes those projections
through the C library. Full traces and personal replays stay under the ignored
`native/artifacts/` directory; checked-in goldens come from synthetic inputs.

For real games:

```sh
dotnet native/tools/replay/bin/Release/net8.0/replay.dll inspect path/to/replays
dotnet native/tools/replay/bin/Release/net8.0/replay.dll mutate input.rep output.rep 67484
python3 native/tools/verify_reference_suite.py input.rep output.rep
```

The mutator preserves the replay header and mixes original play with seeded
held-input bursts and simultaneous ability presses. Preflight rejects missing
player inputs and unresolved ability ID 255. Legacy recordings have no build ID
and only four player streams: five/six-player matches cannot be reconstructed
faithfully from these files. The new `BOPLREP2` format records all six player inputs, prefab-resolved
loadouts and pinned game/catalog identity; see [the format description](FULL_COMPATIBILITY.md#replay-v2).
Use `tools/replay-codec` and `generate_full_scenarios.py` for new-format scenarios.
The old replay inspection/mutation tool above remains specific to legacy files.

## Extending the port

Add one mechanic at a time, first recording targeted Unity cases. Preserve update
and callback order, cached geometry, exact numeric constants, and meaningful
animation clocks. Promote a mechanic into the native API only after independent
rollouts pass. Keep real-game and mutated-replay failures as regression cases;
shrink failures to the first divergent input prefix before diagnosing them.

Finite tests establish agreement over their corpus, not universal equivalence.
See [VALIDATION.md](VALIDATION.md) for measured results and their limits.


Beam graph, pooled hitboxes and target effects can now be imported from actual
replay physics stages. The checked fixtures include 141 stages with emitted beams
and a completed physics stage immediately before a six-player Unity exception.
These tests supply Unity's pre-state each step; they do not run the full native
Beam ability. See [coverage and source failures](FULL_COMPATIBILITY.md#beam-state-from-actual-gameplay).
