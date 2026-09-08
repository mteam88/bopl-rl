# PORTING.md — deviations from Unity semantics and gameplay patches

Reviewer reads this first. Every entry: what differs, where, why, and what
decides the question (usually: the oracle trace).

## Gameplay file set

- **All 450 files** of `Assets/Scripts/Assembly-CSharp/*.cs` are copied into
  `sim/Gameplay/`, plus `BoplFixedMath/Fix.cs`. **No exclusions, no patches.**
  Everything compiles against the shim unmodified (commit `sim/Gameplay:
  mechanical copy`). Presentation/Steam/menu classes compile against data-only
  stubs and are simply never driven.
- `Properties/AssemblyInfo.cs` was not copied (Unity assembly metadata).

## Shim deviations (sim/UnityShim, ~5.7k LOC)

1. **`UnityEngine.Random` is System.Random**, not Unity's Xorshift128
   (`Math.cs`). Users in the sim set: `Client.cs`, `DeathEffect2.cs`,
   `DuplicatePlatformEffect.cs`, `GameSession.cs`, `Host.cs`, `Portal.cs`,
   `SceneBoundArrowController.cs`, `SuperStarEffect.cs`. All are
   presentation/menu paths or replay-recording seed choice; deterministic
   gameplay randomness goes through `Updater.InitSeed`/`RandomFix`
   (Unity.Mathematics.Random). If a trace diff lands in one of those files,
   this is the first suspect.
2. **`Unity.Mathematics.Random` is bit-exact**: reimplemented in
   `UnityMathematics.cs` from an ilspycmd decompile of the game's shipped
   `Assets/Plugins/Unity.Mathematics.dll` (xorshift, `NextState` returns
   pre-mutation state; `NextInt(min,max)` = `state*range >> 32 + min`).
3. **Instantiate lifecycle order** (`CloneRules.DispatchAwake`): collect all
   components depth-first in serialized order → call **all Awakes**, then
   **all OnEnables**, then queue Starts. Alternative reading: per-component
   Awake+OnEnable pairs. The oracle decides; flip in one place if traces
   disagree.
4. **Instantiate field copy**: ctor + field initialisers run, then serialized
   fields (public or `[SerializeField]`, non-static, non-readonly, serializable
   type — `CloneRules.SerializationRules`) are copied; `UnityEngine.Object`
   refs into the cloned hierarchy are remapped, refs out of it kept;
   `[Serializable]` plain classes/structs deep-copied by value. Root clone name
   gets Unity's `"(Clone)"` suffix; children keep their names.
5. **`FindObjectsOfType` order** = root creation order, depth-first. Unity's
   order is unspecified but stable within a run; `Updater.ComputeChecksum` and
   `Host.ResetRl` iterate results. If checksum/trace diffs appear, compare
   orderings first.
6. **Destroy(obj, t)** ignores the delay; destruction happens at the host
   frame's destroy flush (gameplay uses `Updater.DestroyFix` for deterministic
   destruction anyway). `DestroyImmediate` is immediate, including the
   `(doomed, allowDestroyingAssets: false)` overload Updater uses.
7. **Coroutines** are stepped once per host frame after `Update`, before
   `LateUpdate`. `WaitForFixedUpdate` and `WaitForEndOfFrame` are treated as
   one-frame waits (the sim host has one frame phase, not three).
   `WaitForSeconds` compares against scaled `Time.time`.
8. **`Mathf.PerlinNoise`** is a deterministic value-noise stand-in, not
   Unity's Perlin table (presentation only in this game).
9. **`AnimationCurve.Evaluate` (float)** mirrors the Hermite basis of the
   game's own `AnimationCurveFixed.Evaluate`, including pre/post wrap handling.
   The deterministic gameplay path evaluates through `AnimationCurveFixed`
   which only reads `keys` / `length` / `preWrapMode` / `postWrapMode` from the
   shim type.
10. **Collision matrix**: `Physics2D.GetIgnoreLayerCollision` reads a table
    installed by the Export loader from `sim/Export/settings.json` (oracle
    thread deliverable). Until it exists: all layers collide + one loud
    stderr warning.
11. **Headless input**: `Keyboard.current`, `Mouse.current`, `Gamepad.current`
    are null and `Gamepad.all` empty — same as the real headless RL player.
    All `Keyboard.current != null` guards in gameplay skip, as on Linux
    headless. Input reaches players via `OverrideInputWithNetworkInput`.
12. **`SystemInfo.graphicsDeviceType` = `GraphicsDeviceType.Null`** —
    `RlHeadless.Render` guards its 121 presentation sites on this.
13. **Steamworks / UI / renderers / particles / audio / video** are data-only
    stubs: setters store, getters return defaults, nothing simulates.
    `SteamClient.IsValid` is false; `SteamClient.Init` throws.
14. **`Application.Quit`** sets `Application.quitRequested` + fires `quitting`
    instead of killing the process; the sim Host checks the flag.
15. **`Time`** advances only via `World.RunFrame(dt)` (host-driven).
    `Time.frameCount` increments per host frame, mirroring `RlBridge`'s
    scene-load frame gate.
16. **`InputActionAsset.FromJson`** parses only map/action names + control
    scheme names (enough for the generated `MenuControls` wrapper to
    construct); bindings are not modeled, actions never fire.
17. **UI `EventSystem`/`ExecuteEvents`**: minimal — `ExecuteEvents.Execute`
    dispatches to handler interfaces on the target's own components only (no
    bubbling). Used by menu code only.
18. **`UnityEngine.Cache`**: empty struct; `SteamFrame.cs` declares a field of
    this type but never uses it.
19. **Serialization callbacks**: `ISerializationCallbackReceiver.
    OnAfterDeserialize` must be invoked by the Export loader after populating
    fields (`FixTransform` implements it). TODO tracked in Export.

## Open questions for the oracle

- Awake/OnEnable batching on Instantiate (deviation 3).
- Scene-load Awake order: export loader awakes root-order-then-depth-first;
  Unity's order is officially unspecified (DOTNET_SIM.md says try this first).
- FindObjectsOfType ordering (deviation 5).

## TODO

- [ ] Export loader: call OnAfterDeserialize after field population.
- [ ] settings.json reader (layers/tags/collision matrix) — blocked on oracle
      thread; falls back loudly.
- [ ] GameObject tag from export — blocked on oracle thread re-export;
      defaults "Untagged".
