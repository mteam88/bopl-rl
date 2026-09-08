# DOTNET_SIM — running Bopl's gameplay C# outside Unity (NativeAOT)

Status: **day 1**. Owner thread: `thr_5jxhy9dpza` (coordinator). Rifts:
`ripped/.rifts/ExportedProject/dotnet-sim` (shim + compile) and
`ripped/.rifts/ExportedProject/dotnet-sim-oracle` (export + oracle). Branch `dotnet-sim`.

## Why

The RL environment is latency-bound on the Unity player, not on gameplay
arithmetic (`native/rl/autoresearch/notes.md`: `EarlyUpdate.UpdatePreloading`
and player-loop overhead dominate; a tick of actual gameplay is tens of µs).
The Rust port (`native/src`) proved a runtime-free sim is ~100× faster but
required re-deriving every system by hand against traces, and never reached
full coverage.

The game already owns its own deterministic scheduler. `Updater.TickSimulation`
drives `MonoUpdatable.Init/UpdateSim/LateUpdateSim`, `DetPhysics.Simulate`,
and a deferred-destroy queue; `BoplFixedMath.Fix` replaces floats. Unity is
used for: object identity/hierarchy (`GameObject`, `Transform`, `Component`),
`Instantiate`/`Destroy`, serialized field initialisation from scenes/prefabs,
`Resources.Load`, layers/tags, `Physics2D.GetIgnoreLayerCollision`, and
presentation (renderers, audio, particles, UI, Steam) which the sim ignores.

So: **compile the same C# against a small hand-written `UnityEngine` shim**,
load scenes/prefabs from `native/reference-export/*.json` (a complete
`SerializedObject` dump of every scene + prefab + ScriptableObject, produced
by `Assets/Editor/ReferenceExport.cs`), and expose the existing
`native/rl/bopl_rl.h` C ABI via `[UnmanagedCallersOnly]` from a NativeAOT
shared library. Gameplay files are copied, not rewritten. The oracle is the
game's own `ReferenceCapture` (`BOPL_TRACE`) trace, which compiles into the
port unchanged and is compared with `native/tools/reference.py:first_difference`
exactly like `native/rl/verify_replay.py` does today.

## Layout (inside the rift)

```
sim/
  BoplSim.sln
  UnityShim/      UnityEngine.* replacement. Hand-written. Small.
  Gameplay/       COPIES of Assets/Scripts/Assembly-CSharp/*.cs (+ BoplFixedMath).
                  Never edited except through sim/PORTING.md-listed patches.
  Export/         JSON → object graph loader (scenes, prefabs, ScriptableObjects,
                  ProjectSettings layers/tags/collision matrix).
  Host/           Reset/step driver: replay playback + RL bridge, C ABI export.
  Oracle/         Scripts to record Unity traces and diff against the port.
  PORTING.md      Every deviation from Unity semantics and every gameplay patch,
                  with the reason. Reviewer reads this first.
```

## Rules (Bun-rewrite discipline)

1. Gameplay files are mechanical copies. If a file can't compile, the first
   answer is a shim type, not an edit. If an edit is unavoidable (Steam,
   Netcode, UI, `Resources.Load` of audio/sprite), record it in `PORTING.md`
   with file:line and reason. Prefer excluding whole presentation-only files
   over patching.
2. The shim must reproduce Unity *ordering* wherever gameplay can observe it:
   `Instantiate` runs `Awake` (then `OnEnable`) on every component of the new
   hierarchy depth-first in serialized component order before returning;
   `Start` runs once before the object's first `Update`; `Destroy` is
   deferred to end of frame; `GetComponent(s)`/`GetComponentsInChildren`
   return in serialized order; `transform.GetChild(i)` order = serialized
   child order; `GetInstanceID()` is a monotonically increasing counter.
   Anything not reproduced is a documented deviation.
3. `Fix` math is already deterministic C#; do not touch `BoplFixedMath`.
4. Nothing in this work touches the running training process. Build and run
   with `nice -n 19` (`systemd-run --user --scope -p CPUWeight=10` for long
   jobs). Unity editor/player runs go through `native/tools/reference.py`'s
   `run_reference` or an equivalent low-priority scope.
5. Small commits on branch `dotnet-sim`, one concern each. No `git add -A`.
6. When two readings of Unity semantics are plausible, write both down in
   `PORTING.md` and pick the one the trace confirms. The oracle decides.

## Day-1 checkpoint (tonight)

Done means all of these are true and written to `sim/PORTING.md`:

- [ ] `dotnet build sim/BoplSim.sln` succeeds (JIT; NativeAOT not required yet).
- [ ] `Export/` loads `map-0.json` + referenced prefabs into shim objects with
      every serialized field of every gameplay component set (Fix `m_rawValue`
      fields, object references resolved by GlobalObjectId, arrays, curves).
- [ ] A reset on level 0 with 2 players (the RL default) runs `Updater.TickSimulation`
      240 times without throwing.
- [ ] With `BOPL_TRACE` set, the port writes a trace; `first_difference`
      against the Unity trace of the same `episode.rep` reports the first
      diverging tick and field (or none). Record the number: **matched ticks / 240**.
- [ ] Ticks/s for that 240-tick loop under `dotnet run -c Release` (JIT),
      2 players, level 0. Record the number next to Unity's single-process
      figure from `autoresearch/notes.md`.

What is *not* in the checkpoint: NativeAOT, C ABI, many-worlds, PufferLib
wiring, other maps, other abilities. Those are days 2–5.

## Oracle recipe

```
# Unity side (one-off, produces episode.rep + replayed trace):
python native/rl/verify_replay.py --players 2 --ticks 240 --output /tmp/oracle/l0-p2
#   -> /tmp/oracle/l0-p2/episode.rep, controlled.jsonl.gz, replayed.jsonl.gz
# Port side:
BOPL_TRACE=/tmp/oracle/l0-p2/dotnet.jsonl.gz dotnet run --project sim/Host -- replay /tmp/oracle/l0-p2/episode.rep --ticks 240
python sim/Oracle/diff.py /tmp/oracle/l0-p2/replayed.jsonl.gz /tmp/oracle/l0-p2/dotnet.jsonl.gz
```

`sim/Oracle/diff.py` is a thin wrapper over `native/tools/reference.py`
(`read_trace`, `first_difference`, `native/data/presentation-fields.json`).

Known gaps in the export that the oracle rift fixes first:
- `ReferenceExport.Hierarchy` omits the GameObject **tag** (`CompareTag("ResizablePlatform")`
  etc. is used in gameplay). Add `["tag"] = root.tag`, re-export.
- Layers/tags names and the 2D layer collision matrix come from
  `ProjectSettings/TagManager.asset` and `ProjectSettings/Physics2DSettings.asset`
  (YAML; parse directly, no Unity needed). `DetPhysics` calls
  `Physics2D.GetIgnoreLayerCollision` twice.
- Anything `Resources.Load`ed at runtime by gameplay must be in `assets.json`
  or stubbed; grep and list in `PORTING.md`.

## After the checkpoint

Day 2: oracle to 100% on level 0 / dash+bow loadout; add maps 1–4.
Day 3: `Host/` implements `bopl_rl.h` (`bopl_rl_open/reset/step/close`) with
`[UnmanagedCallersOnly]`, `dotnet publish -p:PublishAot=true`, drop-in for
`native/rl/puffer/binding.c`. Day 4: many worlds in one process = many
`World` instances (the static-state swap from `MANY_WORLDS.md` applies
unchanged). Day 5: adversarial review pass over `PORTING.md` deviations,
full-ability coverage sweep with `native/rl/coverage.c`'s loadouts.
