# Shim review, round 1

Reviewed: `sim/UnityShim/*.cs` (~5.7k LOC) plus the loader surfaces it depends
on (`sim/Export/*.cs`), working tree of the `dotnet-sim` rift at merge commit
`d8adcf1` (coordinator referenced `8c14c5a`; the tree I read is the live one).
Read-only review against `sim/Oracle/SEMANTICS.md` [EXACT]/[ORDER] and Unity
semantics. Findings ranked by likelihood of causing a trace divergence.
PORTING.md was checked first — findings below are NOT already documented there
unless said so.

---

## F1. UnityEvent persistent calls are silently dropped — ability casts no-op

**Likelihood: certain** for any loadout containing an `InstantAbility` whose
cast is wired through `funcOnEnter` (Teleport, Hookshot, Portal gun, Web swing,
Super star, Invisibility).

- Shim: `Events.cs:12-27` — `UnityEvent` holds runtime listeners only,
  `GetPersistentEventCount() => 0`. Loader: no `m_PersistentCalls` handling
  anywhere in `sim/Export/` (grep empty).
- Export data: ability prefabs in `assets.json` serialize
  `funcOnEnter.m_PersistentCalls.m_Calls.Array.data[0].m_Target =
  <GlobalObjectId of sibling component>` + `m_MethodName = "UseAbility"`.
- Gameplay observers: `InstantAbility.cs:10,80` (`funcOnEnter?.Invoke()` on
  cast — invokes zero listeners, ability does nothing, no error);
  `QuickSpell.cs:14,131` (`spellFunc`). Wired targets:
  `Teleport.cs:35` (`CastAbility`), `HookshotInstant.cs:78`,
  `PortalGun.cs:41`, `WebSwing.cs:113`, `SuperStar.cs:30` (`UseAbility`),
  `Invisibility.cs:21` (`CastInvisibility`).
- Fix: in the loader's `PropertyApplier`, when a field's flattened path
  matches `<field>.m_PersistentCalls.m_Calls.Array.data[N].*`, resolve
  `m_Target` via the existing `db.ResolveReference` and add a listener
  through a static dispatch table `(component type, method name) → invoke`
  for the closed name set (`UseAbility`, `CastAbility`, `CastInvisibility`;
  full set of 26 in `sim/Oracle/AOT.md` §1). Throw on an unknown name.
  Also honour `m_CallState` if present (Off = skip).

## F2. ScriptableObject `OnEnable` never runs — AnimationData tables stay empty

**Likelihood: high** — first `SpriteAnimator.beginAnimation` returns null
frames and crashes or animates nothing, and sprite-frame timing feeds
hitbox-anticipation classes.

- Shim: `Runtime.cs:184-190` (`ScriptableObject.CreateInstance` = bare
  `Activator.CreateInstance`), `CloneRules.cs:138-144`
  (`CloneScriptableObject` — no callback), `World.cs:132-148`
  (`ActivateLoadedObjects` walks GameObjects only). Unity dispatches
  `OnEnable` on ScriptableObjects at load/CreateInstance/clone time.
- Gameplay observer: `AnimationData.cs:22-46` — `OnEnable` populates
  `spriteFrames`/`velocityFrames`/`animation2DFrames` dictionaries from the
  serialized arrays; readers `AnimationData.cs:49,58` TryGetValue → null on
  the empty dictionary. Consumers on the sim path: `SpriteAnimator.cs:459,470`
  (a MonoUpdatable, fields trace-captured), `Katana.cs`, `Beam.cs`,
  `PlayAnimation.cs`, `RocketEngine.cs`, `CastSpell.cs`, `MachoClap.cs`,
  `SimpleSparkNode.cs`.
- Fix: after the loader populates a ScriptableObject's fields, and in
  `CreateInstance`/`CloneScriptableObject`, invoke a parameterless `OnEnable`
  if the type declares one (reuse `MonoBehaviour.MessagesFor`-style lookup or
  a direct `GetMethod`). `AnimationData.OnEnable` is idempotent-hostile
  (`Dictionary.Add` throws on repeat) — dispatch exactly once per instance.

## F3. Two distinct destroyed objects compare `==` equal — Unity says false

**Likelihood: medium** (needs two dead references meeting in a comparison,
but when it happens the divergence is silent and confusing).

- Shim: `Object.cs:30-37` — `CompareBaseObjects` returns true when BOTH sides
  are null-or-destroyed. Unity's semantics: a null literal equals any
  destroyed object, but two object references compare by instance id —
  `destroyedA == destroyedB` is FALSE unless they are the same object.
- Gameplay observers: any reference comparison where both sides can be dead —
  e.g. platform identity checks around `attachedGround`
  (`PlayerPhysics.cs:216,262,287`, `StickyRoundedRectangle.cs:429`,
  `MachoThrow2.cs:294,404` compare/branch on possibly-destroyed platforms),
  `Shockwave.cs:86,182` buffer entries. Most sites compare against `null`
  (unaffected); the risk is object-vs-object.
- Fix (exact Unity behaviour):

  ```csharp
  private static bool CompareBaseObjects(Object a, Object b)
  {
      bool an = a is null, bn = b is null;
      if (an && bn) return true;
      if (bn) return a.destroyed;      // a == null-literal
      if (an) return b.destroyed;      // null-literal == b
      return ReferenceEquals(a, b);    // ids are unique per object
  }
  ```

  `Equals` and `GetHashCode` (instanceId) stay as they are.

## F4. Three different Awake/OnEnable orderings across activation paths

**Likelihood: medium** — divergence appears when a multi-object hierarchy is
activated and one component's `OnEnable` observes another's `Awake` effects.
PORTING.md deviation 3 documents the Instantiate choice but not that the
other two paths disagree with it.

- Shim: three orders for "hierarchy becomes active":
  1. Instantiate: `CloneRules.cs:150-157` — whole-subtree all-Awakes then
     all-OnEnables (matches Unity).
  2. Scene load: `World.cs:138-148` `ActivateRecursive` — per-GameObject
     Awake-batch then OnEnable-batch, children AFTER the parent's OnEnables.
  3. `SetActive(true)`: `GameObject.cs:92-108` `NotifyActivation` →
     `Component.cs:128-133` `OnBecameActive` — per-COMPONENT interleaved
     Awake, OnEnable, Awake, OnEnable…
- Unity runs the whole activated subtree's Awakes, then the subtree's
  OnEnables, for all three operations. Path 3 is the risky one: gameplay
  toggles ability hierarchies every cast (`FixTransform.cs:135-139`
  inactive-prefab toggle; `GameSessionHandler.cs:270-362` deactivates ability
  objects at spawn; Beam/Mjolnir/FreezeRayGun SetActive sites), and sim
  `OnEnable`s exist (`Roll.cs:482`, `Drill.cs:247`, `Missile.cs:149` —
  `Missile.OnEnable` even instantiates a particle object, which consumes an
  instance id; F6 in SEMANTICS §5 applies).
- Fix: make `NotifyActivation` and `ActivateRecursive` collect the newly
  active subtree and call the same two-pass routine `CloneRules.DispatchAwake`
  uses. One code path, one order, and the PORTING.md "oracle decides" question
  then covers all three call sites at once.

## F5. `DontDestroyOnLoad` is a no-op — Host dies on the first level load

**Likelihood: certain on the multi-round path, invisible in single-round
episodes** (the RL reset loads the level before Host would be destroyed only
if load order is right — fragile either way).

- Shim: `Object.cs:43` — empty body. `GameObject.dontDestroyOnLoad`
  (`GameObject.cs:13`) is never set by anything, and
  `World.UnloadCurrentScene` (`World.cs:151-160`) destroys every root that
  doesn't have it.
- Gameplay observers: `Host.cs:345-346` (the driver!), `SteamManager.cs:114`,
  `AudioManager.cs:151`, `MenuNavRepeater.cs:26`, `PerfHud.cs:48`,
  `SteamFrame.cs:91`. A round transition (`GameSessionHandler.cs:423`
  `LoadScene(6 + next level)`) or the reset path's `LoadScene(replayLevel+6)`
  (`CharacterSelectHandler_online.cs:345`) with Single mode destroys Host and
  the session dies.
- Fix: `if (target is GameObject go) go.dontDestroyOnLoad = true; else if
  (target is Component c) c.gameObject.dontDestroyOnLoad = true;`

## F6. `Color.RGBToHSV`/`HSVToRGB` are not Unity's float formulas — beam
colors are trace-captured

**Likelihood: medium, beams-only** (loadouts with beam; colors mix on beam
overlap).

- Shim: `Math.cs:703-714` (`RGBToHSV` via `Mathf.Repeat((g-b)/delta,6f)/6f`)
  and `Math.cs:685-701` (`HSVToRGB` textbook sector switch). Unity's
  decompiled implementations differ in operation order (RGBToHSVHelper
  computes `offset + (colorone-colortwo)/diff`, then `/6f`, then `+1f` if
  negative; HSVToRGB has its own branch structure) — same math, different
  float rounding.
- Gameplay observer: `DetPhysics.cs:52-70` `BeamColors.Mix` uses
  RGBToHSV/HSVToRGB per component on the sim path, and beam color fields are
  NOT in `presentation-fields.json` — bits land in the trace.
- Fix: transcribe `RGBToHSVHelper`/`HSVToRGB` verbatim from an ilspycmd
  decompile of the game's UnityEngine.CoreModule (same procedure already used
  for Unity.Mathematics.Random). Alternatively propose the fields for the
  presentation exclusion list — but they may feed beam gameplay; verify
  before excluding.

## F7. `Time.fixedDeltaTime` bit-differs from the player's

**Likelihood: low** (single consumer, likely presentation) — but the fix is
one loader line.

- Shim: `Runtime.cs:9` — `1f / 60f` = 0x3C888889. The asset
  (`TimeManager.asset` "Fixed Timestep: 0.01666667", recorded in
  `sim/Export/settings.json` as `0.01666667`) parses to 0x3C88888A. One ULP
  apart.
- Gameplay observer: `OrbitItems.cs:95,123` — multiplies it inside
  `UpdateSim` (moves orbit-item Unity transforms; near-presentation, but the
  class is a MonoUpdatable).
- Fix: `ExportDatabase` installs `Time.fixedDeltaTime =
  (float)settings["fixedDeltaTime"]` next to the collision matrix.

## F8. Members that silently return defaults where the sim path reads them

- `ParticleSystem.MainModule.duration = 5f` (`ParticleSystem.cs:79`):
  `RevivePositionIndicator.cs:76` uses it as a delayed-Destroy horizon —
  presentation object, but the 5 s default is a lie; loader should populate
  it from the prefab or the shim should flag reads of never-written modules.
- `Time.timeSinceLevelLoad` is never reset on `LoadScene`
  (`Runtime.cs:22-30` only advances it): consumers are presentation
  (`BeamEffect.cs:81`, `AnimateIn*`), but reset it in
  `SceneManager.NotifySceneLoaded` for fidelity.
- `LayerMask.NameToLayer` returns −1 for an unknown name (`Math.cs:801-807`)
  and callers immediately compute `1 << -1` (= `1 << 31` in C#) or compare
  layers — a typo in settings.json layer names would corrupt masks silently.
  Make it throw; every gameplay name is expected to resolve.
- `SceneManager.UnloadSceneAsync` unloads nothing (`Scene.cs:88-95`): the
  additively-loaded `winnerTriangle` scene stays forever. Harmless today
  (its objects idle), but `Scene.rootCount`/`GetRootGameObjects`
  (`Scene.cs:25-33`) also return every root including DontDestroyOnLoad and
  additive-scene objects — anything iterating scene roots over-counts.
- `AsyncOperation` completes instantly and `completed` fires synchronously on
  subscribe (`Scene.cs:43-49`) — fine for `GameSessionHandler.Update`'s
  `isDone` gate, but it means "wait one frame for the load" code paths run
  zero frames; keep in mind when matching `RlBridge`'s scene-load frame gate.
- `Instantiate(original, parent, instantiateInWorldSpace)`
  (`Object.cs:114,130-147`): the `worldSpace=true` case is accepted and
  ignored (comment admits it). No gameplay caller today (grep) — make it
  throw instead of silently mis-posing, so a future caller is caught.

## F9. Smaller semantic notes (fix cheaply or document)

- `GameObject.AddComponent` (`GameObject.cs:114-129`) skips Awake when the
  added behaviour starts disabled — Unity runs Awake for any component added
  to an ACTIVE GameObject regardless of `enabled`. No current gameplay
  caller adds-then-disables (all `AddComponent` sites — `SlimeController.cs:465`
  `RewindRecorder`, `StockRespawn.cs:69`, `DuplicatePlatformEffect.cs:369`,
  UI — use default-enabled components), but the fix is one line: dispatch
  `DispatchAwakeIfNeeded()` when `activeInHierarchy`, then `DispatchEnable()`
  only if `enabledSelf`.
- `Object.cs:71-72` destroys children in REVERSE sibling order before the
  parent's components. Unity's within-hierarchy OnDestroy order is
  unobserved by gameplay today (all sim OnDestroy bodies are particle
  cleanup, SEMANTICS §2.6) — fine, but note it next to the code.
- `DestroyGameObject` mutates `enabledSelf=false` before OnDestroy
  (`Object.cs:76-78` via `SetEnabledInternal`) — Unity keeps `enabled`
  readable in OnDestroy. No sim OnDestroy reads it; note only.
- `CompareTag` with an undefined tag returns false (`GameObject.cs:66`);
  Unity throws. All sim tags are defined — acceptable, but a thrown error
  would catch loader/tag-table drift.
- `CloneGameObjectRecursive` always creates a plain `Transform` for clones
  (`CloneRules.cs:73` via `CreateDetached`); a prefab whose root/children use
  `RectTransform` (UI indicators) loses the type. `AbilityReadyIndicator`
  prefabs are SpriteRenderer-based (no RectTransform), so no known sim
  breakage — clone the transform type to be safe.
- `Scene == Scene` compares only buildIndex+valid (`Scene.cs:35`) — additive
  scenes share buildIndex semantics loosely; unused by sim code today.

## Confirmed correct (checked, no action)

- `Unity.Mathematics.Random` (`UnityMathematics.cs`): NextState returns the
  PRE-advance state, ctor/InitState advance once, `NextInt/NextUInt(min,max)`
  use the `state*range >> 32` reduction — matches `native/src/random.rs`,
  which is fixture-validated against 65k Unity draws. Bit-exact. ✓
- `Physics2D.GetIgnoreLayerCollision` polarity (clear bit = ignore) and the
  loader's matrix/tags/layers install (`ExportDatabase.cs:92-100`), plus
  `sceneCountInBuildSettings = 6 + 124 = 130` (`ExportDatabase.cs:18,107`). ✓
- Fake-null operators, `GetHashCode == instanceId`, implicit bool
  (`Object.cs:24-41`) — modulo F3. ✓
- `GetComponent(s)` = component-list order, destroyed filtered;
  `GetComponentsInChildren` = self-first depth-first serialized child order;
  `includeInactive` gating matches Unity (`GameObject.cs:142-235`). ✓
- `Transform.SetParent` appends to the new parent's child list (SEMANTICS
  §3), `worldPositionStays` recomputes locals, default `parent` setter =
  worldPositionStays:true (`Transform.cs:19-59`); note the second parameter
  of `SetParentInternal` is `manageRoots`, not worldPositionStays — the
  `SetParent(p, false)` path is correct but reads confusingly.
- `DestroyFix`/end-of-tick drain is game code and works against
  `DestroyImmediate` + fake-null as shimmed; raw `Destroy` deferral to the
  host frame flush (`World.cs:117-126`) is trace-tolerant — the oracle's
  variable ticks-per-frame (RlBridge batches up to 4096 ticks/frame) proves
  frame-cadence-dependent state never reaches the trace projection. ✓
- Start deferral: `QueueStart` on first enable, flushed at frame start before
  Update (`World.cs:102-115`), only-if-still-enabled — matches Unity; the
  only sim-class Start bodies are presentation (`Arrow.cs:54-65` sets Unity
  localScale only). ✓
- `MonoUpdatable.IsEnabled`/`isActiveAndEnabled` chain
  (`Component.cs:69`) and activeSelf/activeInHierarchy propagation
  (`GameObject.cs:68-108`). ✓
- `AnimationCurve` exposes `keys/length/preWrapMode/postWrapMode` which is
  everything `AnimationCurveFixed.Init` reads; float `Evaluate` is
  presentation-only. ✓
- Coroutine semantics (synchronous to first yield, per-frame step, stopped by
  GO deactivation but not component disable) — menus only, over-built but
  harmless. ✓

## Suggested fix order

F1 and F2 are silent gameplay no-ops with loud downstream effects — do them
before the first oracle diff so divergences aren't dominated by missing
casts/animations. F5 next (one line, unblocks multi-round). F3/F4 are
one-liners/refactors best done before debugging any trace mismatch, because
both produce the confusing kind. F6/F7 when a beam/orbit divergence shows up
or preemptively (both are mechanical). F8/F9 as hygiene.
