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
19. **Serialization callbacks**: `FixTransform` is the only
    `ISerializationCallbackReceiver` in the gameplay set and its
    `OnAfterDeserialize` body is empty — the loader does not dispatch it.
    Revisit if a mod/patch adds a non-empty implementation.

## Review round 1 fixes (sim/Oracle/REVIEW-shim-1.md)

- **F1 UnityEvent persistent calls**: `Events.cs` rewritten —
  `UnityEventBase.m_PersistentCalls` (`PersistentCallGroup` →
  `PersistentCall` → `ArgumentCache`) is a serialized field tree the Export
  loader populates via the normal property walker; `Invoke()` runs persistent
  calls (reflection, list order, `Off` state skipped) before runtime
  listeners. Sim consumers: `InstantAbility.funcOnEnter`,
  `QuickSpell.spellFunc`.
- **F2 ScriptableObject.OnEnable**: dispatched on `CreateInstance`, on
  `CloneRules.CloneScriptableObject` after field copy, and by
  `ExportDatabase.BuildAsset` after deserialization (`AnimationData.OnEnable`
  builds its frame dictionaries there).
- **F3 destroyed-object equality**: `Object.CompareBaseObjects` — null
  literal equals any destroyed object, but two references compare by
  identity even when both destroyed (`destroyedA == destroyedB` is false
  unless same instance; Unity compares instance ids).
- **F4 unified activation batching**: scene load (`ActivateLoadedObjects`),
  `Instantiate` (`CloneRules.DispatchAwake`) and `SetActive(true)`
  (`GameObject.NotifyActivation`) all collect the affected subtree
  depth-first, STABLE-sort by script execution order
  (`ScriptExecutionOrder`, table ripped from the script `.meta` files into
  `sim/Export/execution-order.json`), then run all Awakes, then all
  OnEnables. `FlushPendingStarts` sorts the same way.
- **F5 DontDestroyOnLoad**: marks the root GameObject; Single-mode scene
  loads spare marked roots.
- **F6 Color.RGBToHSV/HSVToRGB**: transcribed verbatim from an ilspycmd
  decompile of the game's `UnityEngine.CoreModule.dll`
  (`native/tools/replay/bin/Release/net8.0/`); note Unity's g-dominant branch
  passes `(b, r)` — argument order differs from the textbook form.
  `BeamColors.Mix` runs these on the sim path and results are
  trace-captured.
- **F7 Time.fixedDeltaTime**: loaded from `settings.json` (`0.01666667` →
  0x3C88888A, the TimeManager.asset value), not `1f/60f` (0x3C888889).
- **F9 (partial)**: `AddComponent` now dispatches Awake on active
  GameObjects regardless of the behaviour's `enabled` (OnEnable still gated
  on `enabled`). Other F9 notes (reverse-order child destroy, `enabled`
  during OnDestroy, `CompareTag` on undefined tag, clone `RectTransform`
  fidelity — done via `UseRectTransform`) are documented, unobserved by the
  sim set. F8 items deferred until after the first trace diff.

## Trace-identity fidelity (first oracle diff, fixtures l0-p2)

`ReferenceCapture` prints object identities and reflects PRIVATE fields by
name, so the shim must match Unity's internal layouts wherever captured:

- **Scene membership**: every GameObject carries its scene; scene loads stamp
  the whole subtree, `new GameObject`/clones inherit the active scene, assets
  have an invalid scene (renders as a bare "/Name[0]" path), reparenting
  moves membership, and `DontDestroyOnLoad` moves the root into the
  "DontDestroyOnLoad" pseudo-scene (buildIndex -1) - all visible in
  `ReferenceCapture.PathOf`.
- **Placeholder names**: the rip stores one asset file per guid, so
  unexported assets (sprites, materials) get their real Unity name from
  `sim/Export/asset-names.json` (generated from Assets/**/*.meta).
- **`Keyframe`** and **`RaycastHit2D`** reimplemented with Unity's exact
  private field names (m_Time/m_Value/... and m_Centroid/m_Collider(int)/...)
  behind the public property API.
- **Renderer.material**: get returns the renderer-owned clone named
  "<source> (Instance)" (cloning whatever is assigned unless already such an
  instance); set stores the reference. The serialized m_Materials array IS
  sharedMaterials.
- **Clone roots leak (bug)**: clone children were root-registered and never
  removed (SetParentInternal manageRoots:false), so FindObjectsOfType
  visited them twice - "facing" section doubled at player spawn.
- **Sound bank**: `sim/Export/sound-bank.json` ripped from MainMenu.unity's
  AudioManager (136 entries, order preserved): `InterruptableAudio` stores
  bank indices (Dash.dashSound.index = 14 = "dash") and they are captured.
  `AudioClip.length` defaults to 1 (PlayPrefixWithInterrupts divides by it).

Result on `sim/Oracle/fixtures/l0-p2` (480 ticks, batch 16): **480/480 ticks
match** once two known-blocked field classes are filtered
(`Keyframe.m_InWeight/m_OutWeight` - the exporter drops curve key weights;
`Player.playerDied/playerWon` - HandleAbilitySelectUI lives in the
unexported WinnerTriangle scene). Unfiltered: 0 ticks (first diff is
playerDied at tick 0). Blocked on oracle: re-export with per-key weights
(f32 bits ideally) + export WinnerTriangle.unity.

## Host bootstrap deviations (boot scene is not exported)

The Unity player reaches gameplay through the boot scene → menus; the sim
Host reconstructs the DontDestroyOnLoad singletons that flow leaves behind:

- **Settings** loaded from `Settings.asset` + `Init()`;
  `LocalizedText.localizationTable` from `LocalizationTable.asset`
  (SettingsInitializer.Awake equivalent).
- **SteamManager** with `abilityIconsFull/Demo` assets and
  `networkClient = Host` (boot-scene serialized wiring).
- **AudioManager** with an EMPTY sound bank and one placeholder song: the
  audio bank is not exported (presentation) but `Play()` must run because it
  feeds `ReferencePlayerController.SoundCapture` (trace-captured). Every
  play logs "non-existent sound" and returns null; music paths index the
  placeholder song.
- **WinnerTriangleCanvas** stub hierarchy (canvas → triangle →
  inactive CharacterStatsList): the additive `winnerTriangle` scene
  (201-object UI, DontDestroyOnLoad root) is not exported.
  `GameSessionHandler.prepareWinnerTriangle` needs the two GameObjects at
  round start. Round-END UI (`ActivateWinnerTriangle` →
  `CharacterStatsList.OnEnable` player rows) will NOT reproduce until the
  oracle exports `Assets/Scenes/WinnerTriangle.unity`. TODO(oracle).

## Open questions for the oracle

- Awake/OnEnable batching on Instantiate (deviation 3).
- Scene-load Awake order: export loader awakes root-order-then-depth-first;
  Unity's order is officially unspecified (DOTNET_SIM.md says try this first).
- FindObjectsOfType ordering (deviation 5).

## TODO

- [x] OnAfterDeserialize: not needed — only implementer (`FixTransform`) has
      an empty body (see deviation 19).
- [ ] settings.json reader (layers/tags/collision matrix) — blocked on oracle
      thread; falls back loudly.
- [ ] GameObject tag from export — blocked on oracle thread re-export;
      defaults "Untagged".
- [ ] Oracle: export `Assets/Scenes/WinnerTriangle.unity` (round-end UI) and
      the boot-scene AudioManager sound-name list (SoundCapture parity for
      sounds whose Play sites the sim reaches).
- [ ] F8 items (ParticleSystem.MainModule.duration, timeSinceLevelLoad reset,
      LayerMask.NameToLayer throw, UnloadSceneAsync, AsyncOperation frame
      gap, Instantiate worldSpace throw) — after first trace diff.
