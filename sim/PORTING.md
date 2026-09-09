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
- **Many-worlds deviation (2026-09-09, `native/rl/SIM_OVERHAUL.md` section
  3)**: the per-match static fields (~110, every one carries the comment
  `// per world: RlWorldState`) are `[System.ThreadStatic] internal static`
  and their initialisers moved to `sim/Gameplay/RlWorldState.cs`, which
  restores them when a thread binds a `UnityShim.World` and captures them on
  unbind. Auto-properties with such backing state (`GameTime.PlayerTimeScale`,
  `GameSessionHandler.SuddenDeathInProgress`, `ReplaySession.Playing/Recording`)
  became explicit properties over a thread-static field. `Player.InputProfile.
  GenerateInputVectorTable` fills a local and publishes it whole. Behaviour in
  a single world is unchanged; the change lets many worlds tick concurrently
  in one process. Process-wide constants (layer indices, `Constants.*`,
  `achievementNames`) stayed plain statics on purpose.

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
5. **`FindObjectsOfType` order** = Unity's `FindObjectsSortMode.InstanceID`
   ascending (the 2022.3 legacy API maps to it): runtime-created objects have
   negative, decreasing instance ids, scene-file objects positive in file
   order — so clones enumerate newest-first, before scene objects. The shim
   keeps its positive monotonic counter plus an `Object.persistent` flag set
   by GraphBuilder for scene-file objects, and sorts with key
   `persistent ? +id : -id`. Order is gameplay-visible
   (Mjolnir/Portal/BananaPeel/FreezeRay iterate
   `FindObjectsOfType<SlimeController>`) and trace-visible (ReferenceCapture
   identity ties among same-named root clones). Root cause of the g3
   Mjolnir/Portal divergence (l0 tick 372) and the l3-p2-g4 CursorArrow trace
   tie (tick 649); fixing it took both fixtures to full match.
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
- **AudioClip.length** defaults to 1 (PlayPrefixWithInterrupts divides by it;
  clip assets are placeholders - audio never plays).

Result (v3 export, real boot + WinnerTriangle scenes): **all five fixtures
match UNFILTERED** - l0 480, l1 444, l2 480, l3 480, l4 418 ticks, including
the early-terminating death episodes. No filtered fields remain.

## Host bootstrap (v3 export: the boot scene IS exported)

`Program.Bootstrap` loads MainMenu.unity (build index 0, scene-boot.json)
exactly as the player boots: the real `SteamManager` (Steam init fails
headless, matching the batchmode oracle), `Host` on the "networkClient"
root, `SettingsInitializer` (Settings.Init + LocalizedText wiring),
`AudioManager` (136-sound bank in serialized order), all wired through
serialized references. The `winnerTriangle` scene loads through the game's
own `ForceStartGame` → `LoadScene("winnerTriangle", Additive)` path;
`HandleAbilitySelectUI.Init` subscribes `Player.playerDied/playerWon` from
its `sceneLoaded` handler - the former stub hierarchy, hand-rolled
AudioManager, and `sim/Export/sound-bank.json` are gone (git history).

One deviation: `Host.Awake` early-returns when process argv carries a
`.rep` path (the player's command-line replay flow through the unexported
"Replay" scene). The sim Host takes the RL/ResetRl path instead and
re-applies the `DontDestroyOnLoad(host/DebugTextRenderer)` calls the
no-argv branch would have made (idempotent when Awake did them).

Shim/loader fixes the real scenes forced (all trace-verified):

- **UseRectTransform orphaned the GameObject** (bug): GraphBuilder parents
  the GO before reaching the RectTransform component entry; the swap now
  carries parentField/children/local pose (charStatsList.transform.parent
  was null → prepareWinnerTriangle NRE).
- **Clones lose RectTransform** (bug): CloneGameObjectRecursive now mirrors
  the source transform kind (AbilitySelectCircle.trans is a serialized
  RectTransform field; assigning a Transform threw ArgumentException).
- **Graphic.material never returns null** - unset means the shared
  "Default UI Material" (AbilitySelectController clones it).
- **Instantiate(Material)** supported: copy named "<name>(Clone)".
- **AudioListener** is a Behaviour (component on the boot AudioManager)
  with the static volume/pause API preserved.
- **Texture2D** gained a parameterless ctor for the loader's placeholder
  path (Activator.CreateInstance failed → CursorUpdater.originalTexture
  null → boot Start crash).
- **scenes.json**: non-level scenes (boot 0, WinnerTriangle 4) load by
  buildIndex or name; name lookup is OrdinalIgnoreCase (the game calls
  LoadScene("winnerTriangle") for WinnerTriangle.unity); Additive loads
  leave the active scene untouched and stamp loaded roots explicitly.

## v5 export: real sprites (g4 training loadout)

- **Instantiate clones shim built-in component state** (bug, g2/g4): Unity
  clones native-component state wholesale, but SerializationRules skipped
  the shim's auto-property backing fields (private, no [SerializeField]) -
  cloned SpriteRenderers reset color to white (g2 SimpleSparkNode.origColor
  r=1 vs 0) and dropped their sprite (g4 ResizablePlatform NRE). Fields
  declared by UnityShim-assembly types are now serialized when their TYPE
  is serializable; non-serializable classes (ParticleSystem modules,
  ctor-created with back-references) stay per-instance.
- **sprites.json** (981 sprites keyed by GlobalObjectId): the loader builds
  real Sprites - texture dims, pixelsPerUnit, rect, textureRect, pivot,
  bounds, border, exact f32 bits - with atlas sprites sharing one Texture2D
  per textureId. Sim-path readers: ResizablePlatform (texture.width/height,
  pixelsPerUnit), GrowthFlare (pivot, bounds.size), StretchableLightning
  (textureRect min/max), DuplicatePlatformEffect. Unmatched ids still get
  named placeholders.

Result: **l0-p2-g4 (Dash/Bow/Platform - the live RL training loadout)
matches 720/720 unfiltered**; l0-p2, l4-p2, l0-p2-g0, l0-p3-g0 re-verified
green after the change.

## In-process C ABI (sim/Native -> BoplSimNative.so)

`sim/Native` publishes a NativeAOT shared library exporting the full
`native/rl/bopl_rl.h` ABI (version 2) via `[UnmanagedCallersOnly]`, replacing
the worker process + unix socket with direct calls. Smoke test:
`native/rl/native_check.c` (dlopen, open, reset, 240 zero-action steps).

- **RlBridge's dispatch layer is reimplemented, not called**: all of
  RlBridge's members are private, but everything it dispatches to is reused
  verbatim (`Host.ResetRl`, `Updater.TickSimulation`, `RlObservation.Write`,
  `RlTelemetry`, `ReplayArchive`, `ReferenceCapture`) via a new
  InternalsVisibleTo(BoplSimNative) on Gameplay. The observation wire bytes
  from `RlObservation.Write` are decoded into unmanaged `BoplRlState` blocks
  exactly the way `bopl_rl.c decode_snapshot` does; bopl_rl.c's client-side
  argument validation and status codes are mirrored verbatim.
- **Frame cadence = the Unity RL runtime, not the trace fixtures**: RlBridge
  runs up to `CommandsPerFrame` (4096) simulation ticks per engine frame and
  defers a step that would cross the budget (`pendingStep`), so destroy
  drains/Start flushes happen every ~4096 ticks — unlike the 16-tick cadence
  of the BOPL_TRACE_BATCH=16 fixtures. The .so mirrors the 4096 model
  (boundary after reset, deferral before an oversized step) for bit-parity
  with the socket path, which is what the checksum comparison targets.
- **`BOPL_RL_SOCKET=in-process` is set at open**: `RlTelemetry.Enabled`
  requires `RlBridge.Enabled` (that env var). The only other consumers
  (Host.Update -> Pump, Host.OnLevelWasLoaded) are never dispatched by the
  shim host. Without it the telemetry section of the observation would stay
  zero, unlike the socket path.
- **One world per process**: gameplay statics; a second `bopl_rl_open`
  returns an error (and close does not lift the latch).
- **reset_begin/finish are synchronous**: no socket to overlap with, so begin
  does all the work and finish reports the stashed result; the pending
  protocol (EBUSY/EINVAL -> BOPL_RL_IO, close drains) is preserved.
- **BOPL_TRACE works inside the .so** (`Tick` calls
  `ReferenceCapture.BeforeReplayStep` like RlBridge.Tick, with the
  `!Completed` truncation guard from sim/Host) and the open() log path is the
  `-logFile` equivalent: stderr (shim Debug.Log*) is redirected there.
- native_check on the default config (2 players, l0, Dash/Gust/Invisibility):
  reset 59 ms, 240 steps in 5 ms (~48k ticks/s through the full ABI with
  per-step observation encode/decode), checksum stable across runs and
  unchanged with tracing on. The checksum awaits comparison against the same
  sequence through the RL thread's socket path.

## Stage 5 allocation diet (2026-09-09, branch stage5-perf)

Gameplay-file edits for per-tick allocation, all bit-exact by construction
(same arithmetic, same ordering; verified against the pre-change build:
`Host worlds` checksum `6ba6807c6a980f41` for `--n 8 --episodes 2
--abilities 2,4,14`, `backend_check` `5381bf21573b20a6`/`8dc13de2cc8c8e7c`):

- `DetPhysics.SyncGameObject(IShape ...)` -> generic `<TShape> where TShape :
  struct, IShape`: the end-of-Simulate sync loop boxed every non-trigger
  collider's shape struct every tick (46% of tick allocation).
- `DetPhysics.PrepareCollisionCallBacks(... IShape shapeA, IShape shapeB ...)`
  -> generic `<TA, TB>`: shapes are now boxed only when a side actually has a
  collision callback (body verbatim, including the original's
  `p1.collisionCallback` null-check quirk in the p2 branch).
- `Array.Sort(..., IComparer)` / `List.Sort(IComparer)` allocate a
  `Comparison<T>` wrapper per call inside `ArraySortHelper`; the broadphase
  sorts (`boundingBoxes`, `bb_rope_rr`), the raycast output sort and
  `Updater.toBeAddedUpdatables` now use cached `Comparison<T>` delegates
  (same IntroSort, same ordering).
- `RlObservation` reads of private fields (`isInAbility`,
  `totalBeamsLastFrame`, bow `loadingFrame`/`hasFired`) via
  `[UnsafeAccessor]` instead of `FieldInfo.GetValue` (boxed a primitive per
  read per observation).

Measured with `Host alloc` (new mode, `sim/Host/AllocCheck.cs`): decision
tick 2,077 -> 1,152 B (spawn-heavy loadout 2,4,14 rate 2; default loadout
1,712 -> ~900), observation 86 -> 3.5 B. The remainder is Object.Instantiate
clone machinery on spawns plus ~0.4 KB/tick of contact-callback boxing
(player-platform contacts box the shape into `CollisionInformation.collider`,
which gameplay callbacks read as `IShape`).

### Round 2: contact-callback box pool (bit-exact)

`DetPhysics.RentShapeBox<T>`: pooled boxed shapes per type (Box / Circle /
RoundedRect), rewritten in place with `Unsafe.Unbox<T>` and handed out as
`IShape` per queued callback. Safe because no gameplay code stores an
`IShape` or a `CollisionInformation` beyond the callback dispatch (grepped),
each `PrepareCollisionCallBacks` call gets a distinct pool slot within the
tick, and the pool resets with `ToMakeCallBacks` at the top of `Simulate` -
after the previous tick's dispatch. Checksums identical to baseline
(loadouts 2,7,19 / 3,9,28 / 27,32,28, maps 1+4). Killed the last
steady-state per-tick allocator: `phase:physics` ~157 B/tick -> 0.

### Stage 5 headless split (fidelity notes)

`RlHeadless.Render` is always false in the shim (`GraphicsDeviceType.Null`);
oracle capture keeps presentation alive via `ReferenceCapture.Enabled`
where it is observable. Guarded as presentation, verified bit-exact
(baseline checksums unchanged; beam loadout 28,28,28 `7846c5b21596d0c9`,
rope loadout 27,32,28, maps 1+4):

- `DetPhysics.renderBeams`: after-effect prefab instantiation, particle
  systems, colors/audio, hit-effect pool activation + transforms. The
  `BeamObject` pool and its `DPhysicsBox` hitboxes stay - gameplay.
- Rope mesh floats: `Rope` creation-time `UpdateMesh`/`SetMeshValues`,
  `DetPhysics.UpdateRopeMesh_parallell` body (the `levelTypeThisFrame`
  latch stays - `SimulateRope` reads it for space gravity).
- `AudioManager` "beamSparks" random-prefix pick.

One deliberate inexactness, `DestroyIfOutsideSceneBounds`: boundary death
effects (water splash etc.) are skipped headless UNLESS the prefab carries
`DeathEffect2`, whose `Init()` calls `Updater.HitStop` (gameplay), or
oracle capture is on. Skipping a splash means `Updater.prefabsAdded`
advances less, so every LATER prefab's `HierarchyNumber` (100,000,000 + a
global running count) shifts down by the skipped updatable count. Relative
update ORDER is unchanged (uniform shift), so trajectories are identical -
verified: step counts and episode outcomes match the keep-everything build
across loadouts/maps, and a byte-level wire diff (env `BOPL_WIRE_DIR`, new
debug hook in `WorldsCheck`) over two full episodes showed every differing
word is an entity `hierarchy` field near 1e8 (1,234 + 1,228 words, zero
other diffs). The trainer never reads `hierarchy` (no reference in
`bopl_puffer.h` / `eval_binding.c` / `bopl_set_encoder.cu`), so training
and transfer are unaffected. `Host worlds` checksums DO change vs baseline
(the wire hashes the raw entity record); use `BOPL_KEEP_BOUNDARY_FX=1` to
restore baseline-comparable checksums when bisecting.
`sim/Oracle/checkpoint.sh`: all five fixtures still match (2026-09-09).

## Open questions for the oracle

- Awake/OnEnable batching on Instantiate (deviation 3).
- Scene-load Awake order: export loader awakes root-order-then-depth-first;
  Unity's order is officially unspecified (DOTNET_SIM.md says try this first).
- ~~FindObjectsOfType ordering (deviation 5)~~ — resolved: InstanceID sort
  mode with the negative-runtime-id model; confirmed by the l3-p2-g4
  CursorArrow tie and the g3 fixture going green.

## TODO

- [x] OnAfterDeserialize: not needed — only implementer (`FixTransform`) has
      an empty body (see deviation 19).
- [x] settings.json reader — shipped and consumed (`LoadSettings`: layers,
      tags, collision matrix, fixedDeltaTime, timeScale).
- [x] GameObject tag — in the v2 export, applied by GraphBuilder ("Untagged"
      fallback can only trigger on a v1 export).
- [x] Curve keyframes decode from the flattened `m_Curve.Array.data[N]` f32
      leaves (exact player bits incl. ±Infinity constant-mode tangents and
      key weights; REVIEW-loader-2 L1/L2) — the Fix-raw summary is applied
      first, leaves overwrite.
- [x] WinnerTriangle + boot scenes (v3 export) loaded for real — all five
      fixtures match unfiltered.
- [ ] F8 items (ParticleSystem.MainModule.duration, timeSinceLevelLoad reset,
      LayerMask.NameToLayer throw, UnloadSceneAsync, AsyncOperation frame
      gap, Instantiate worldSpace throw) — after first trace diff.
- [ ] AOT residual risk (coordinator): serialized `List<T>` fields without
      inline initializers on not-yet-exercised maps/loadouts may need rd.xml
      entries — a serialized-field scan generating rd.xml would close it.
- [ ] Determinism flake (1 in ~17 runs): one checkpoint.sh l4 replay diverged
      at tick 0 on `playerDied` + `updatables`, unreproducible in 16 retries
      (10 manual + scripted). Suspect a `Dictionary<string,_>` iteration
      order leak (.NET randomizes string hash seeds per process) somewhere
      on a trace-observable path — audit sim-path dictionary iterations.
- [x] l3-p2-g4 "divergence" at 649/720: NOT physics — both players entered
      the Platform ability the same tick, spawning two root CursorArrow(Clone)
      objects whose ReferenceCapture identities tie ("[0]" both: PathOf skips
      the sibling ordinal for roots), so the trace order fell through to
      FindObjectsOfType enumeration order. Fixed by deviation 5's InstanceID
      sort; the same fix resolved the REAL g3 physics divergence
      (Mjolnir/Portal iterate FindObjectsOfType<SlimeController> newest-first
      in Unity). All 15 committed fixtures now match unfiltered.
- [x] native_check checksum vs the socket path — `native/rl/backend_check.c`
      runs the same sequence through `bopl_rl.c` + `Host` and through the .so:
      identical (`5381bf21573b20a6` for the 240 zero-action ticks, plus a
      242-decision scripted game with respawns and a dirty reset). The same
      tool dlopens up to 16 file copies of the .so in one process, all
      agreeing (2026-09-08, `native/rl/evidence/trainer-step/`).
