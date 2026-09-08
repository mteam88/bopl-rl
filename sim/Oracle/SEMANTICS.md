# Unity runtime semantics the simulation can observe

Adversarial review checklist for the UnityEngine shim. Derived by grep + reading
of `Assets/Scripts/Assembly-CSharp` at commit `8a66de5` (391 gameplay files,
`Reference*`/`Rl*` instrumentation excluded unless noted). "Sim path" = code
reachable from `Updater.TickSimulation` / `DetPhysics.Simulate` / the reset path
(`Host.ResetRl` → `CharacterSelectHandler_online.ForceStartGame`), i.e. the 122
transitive `MonoUpdatable` subclasses (119 direct + `Ability`,
`InstantAbility`, `Rewind` via `AbilityMonoBehaviour`) plus `Updater`, `DetPhysics`, `FixTransform`,
`GameSessionHandler`, `PlayerHandler`, `Player`, `GameTime`, `GameSession`,
`MagnetGun`, `RoutineQueue`, `SceneBounds`, `Host` (driver parts).

Each item: what the source does, file:line, what the shim must guarantee.
Groups: **[EXACT]** = values/semantics must be bit-identical; **[ORDER]** =
correct iff a specific enumeration/scheduling order is reproduced;
**[PRESENTATION]** = may be stubbed to a no-op (trace-excluded or
render/audio-only).

---

## 1. The scheduler contract — [EXACT] and [ORDER]

`Updater.TickSimulationInner` (Updater.cs:234-354) is the whole game loop. The
shim does not implement it (it is gameplay code) but everything it calls into
the shim must behave exactly:

1. `Player.UpdateFRAMEDependentInputs` + `UpdatePlayer` per player, in
   `PlayerHandler` list order (Updater.cs:251-257). [ORDER]
2. Pending-updatable sweep: `toBeAddedUpdatables` drops null/IsDestroyed
   entries in REVERSE index order (:258-264), is sorted by
   `UpdatableComparer` = `x.HierarchyNumber - y.HierarchyNumber`
   (Updater.cs:10-21, :270), then `Init()` runs in sorted order (:271-278)
   and entries append to `updatables` (:279-282). [ORDER] — note
   `List<T>.Sort` is an unstable introsort; if two live updatables ever share
   a `HierarchyNumber` the result depends on the BCL's exact sort. The shim
   host must use the same runtime `List<T>.Sort`; flag any duplicate
   HierarchyNumber sighting as a bug.
3. `DetPhysics.UpdateDuplicatedObjects` unless time stopped (:284-288).
4. Destroy sweep #1 over `updatables` in reverse order: entries that are
   Unity-null OR `IsDestroyed` get `OnDestroyUpdatable()` then are removed
   (:289-297). [EXACT] Unity-null: a `DestroyImmediate`d object must compare
   `== null` while the C# reference is alive. The shim must implement the
   fake-null `==`/`!=`/implicit-bool operators on its `UnityEngine.Object`.
5. `UpdateSim(dt)` for every updatable with `!IsDestroyed && IsEnabled()`
   (:298-305); `IsEnabled` = `isActiveAndEnabled` (MonoUpdatable.cs:27-30) =
   active in hierarchy AND component enabled. [EXACT]
6. `MagnetGun.StaticUpdate()` (:307), `DetPhysics.Simulate(dt)` (:317),
   `LateUpdateSim` sweep (:318-325), `VeryLateRopeUpdate` +
   `UpdateRopeMesh_parallell` (:327-329), destroy sweep #2 (:330-338). [ORDER]
7. Deferred destruction drains `destroyedObjects` then `destroyedMonos`
   queues FIFO with `Object.DestroyImmediate` (:339-350). [ORDER]
8. First tick after a level load returns early at :308-315 — after
   `MagnetGun.StaticUpdate`, **before** `DetPhysics.Simulate` — and still
   increments clocks/ticks. [EXACT]

## 2. Group [EXACT] — values and semantics that must be bit-identical

### 2.1 Object lifecycle

- **`Instantiate` runs `Awake` synchronously** on every component of the new
  hierarchy before returning, in serialized component order, parents before
  children (Unity semantics). 126 sim classes have Unity lifecycle methods
  and all but `Updater` define `Awake` (full list from `rg "void Awake"`; e.g.
  DPhysicsBox.cs:293, PlayerPhysics.cs:93, SlimeController.cs:459,
  GameSessionHandler.cs:152, DetPhysics.cs:290). Sim `Awake`s do component
  wiring (`GetComponent`), `Updater.RegisterUpdatable(this)`, and
  `PhysicsParent { instanceId = GetInstanceID(), … }` construction. The
  relative Awake order inside one prefab decides `RegisterUpdatable` order,
  which (with the +100000000 prefab offset, Updater.cs:143-153) decides
  HierarchyNumber, which decides Init/UpdateSim order forever. [also ORDER]
- **Inactive prefab trick**: `FixTransform.InstantiateFixed(GameObject,…)`
  (FixTransform.cs:131-145, :161-177) — when the instantiated root is
  inactive it does `SetActive(true); SetActive(false)`. Unity defers `Awake`
  of inactive objects until first activation; the toggle forces `Awake` (and
  `OnEnable`/`OnDisable`) NOW. The shim must defer Awake for inactive
  instantiations and run it on first activation, or this trick registers
  components twice/never.
- **`Awake`-before-`Start`**: `Start` exists in sim classes Arrow.cs:54,
  Host.cs:278, AirwaveCaster_DEBUG.cs:14, FollowTransform.cs:20 (+58
  menu/UI files). Unity runs `Start` once, before that object's first
  `Update`, only while active. Arrow.Start (:54-66) parents/scale-adjusts
  landed arrows — gameplay. [EXACT]
- **`Updater.DestroyFix(GameObject)`** (Updater.cs:89-110): sets
  `IsDestroyed = true` on all `MonoUpdatable`s from
  `GetComponentsInChildren(includeInactive: true, buffer)` (Unity returns
  self + descendants depth-first, serialized order), on the root's
  `FixTransform`, and on DIRECT children's FixTransforms only (`GetChild(j)`
  loop :101-108 — grandchild FixTransforms are NOT flagged); then enqueues.
  `DestroyFix(MonoBehaviour)` (:112-123) flags only that component. 94 call
  sites / 57 files. [EXACT — including the direct-children-only quirk]
- **Reads between DestroyFix and end-of-frame are normal**: the object stays
  fully alive until the queue drain; gameplay reads `IsDestroyed` everywhere
  (e.g. Shockwave.cs:86, MagnetGun.cs:602, MachoThrow2.cs:294), and destroy
  sweeps call `OnDestroyUpdatable()` on flagged-but-undestroyed objects
  (Updater.cs:293-295, :334-336). `DestroyImmediate` then makes every
  reference fake-null and fires Unity `OnDisable`(if active)+`OnDestroy` on
  the components (see §2.6). [EXACT]
- **Raw `Object.Destroy` is end-of-UNITY-frame, not end-of-tick**: 28 sites /
  19 files (list: /audit — sim-relevant ones: Arrow.cs:132 destroys child
  gameobjects when a platform collects >80 arrows; FollowTransform.cs:81
  self-destroys when leader dies; WaterWaves.cs:55-59; CanSelfDestruct.cs:17-29;
  RevivePositionIndicator.cs:76 uses the DELAYED overload
  `Destroy(go, particle.main.duration)`). With `CommandsPerFrame`/
  `BOPL_TRACE_BATCH` > 1 several ticks run per engine frame, so raw-Destroyed
  objects legally survive additional ticks and appear in traces until the
  frame ends. The shim needs both queues: per-tick (DestroyFix) and per-frame
  (Destroy), drained where Unity drains them. The delayed overload runs on
  float wall-time — presentation object here, stub as end-of-frame is fine
  but document it. [EXACT for lifetime windows]
- **`DontDestroyOnLoad`**: Host.cs:345-346, SteamManager.cs:114,
  AudioManager.cs:151, SteamFrame.cs:91, PerfHud.cs:48, MenuNavRepeater.cs:26.
  Only `Host` matters to the port (survives level loads). [EXACT for Host]

### 2.2 Active state and enabled gating

- `MonoUpdatable.IsEnabled()` = `isActiveAndEnabled` (MonoUpdatable.cs:27-30)
  gates every `UpdateSim`/`LateUpdateSim` (Updater.cs:300, :320). The shim
  must implement activeSelf vs activeInHierarchy propagation through the
  parent chain. [EXACT]
- `DetPhysics.IsActive(pp)` (DetPhysics.cs:1658-1666) =
  `fixTrans != null && gameObject.activeInHierarchy && monobehaviourCollider.enabled`
  — collider participation in the broad phase. [EXACT]
- `SetActive` is gameplay: ability objects are toggled on cast/expiry
  (GameSessionHandler.SpawnPlayers deactivates freshly instantiated ability
  objects, GameSessionHandler.cs:270-362; Beam 11 sites, SlimeController 8,
  Mjolnir 4, FreezeRayGun 4, MagnetGun/PlayerPhysics/RopeAttachment 3 each…).
  `activeSelf` reads gate logic in MagnetGun.cs:602, Mjolnir.cs:201,
  StickyRoundedRectangle.cs:429 etc. Unity fires `OnEnable`/`OnDisable` on
  the whole subtree during SetActive — see §2.6 for who listens. [EXACT]

### 2.3 Tags and layers

- Tag comparisons on the sim path (all sites, from `rg CompareTag`):
  BoplBody.cs:212 (`isResizable`), DetPhysics.cs:1257, Arrow.cs:39,56,
  RopeAttachment.cs:192, BlackHole.cs:434,479,523, Mine.cs:229,
  ShootQuantum.cs:141,179, SlimeController.cs:477, SlimeTrailHandler.cs:27,
  ShootDuplicator.cs:615, Shockwave.cs:193, POCHandler.cs:158,
  TeleportIndicator.cs:124, MagnetGun.cs:602,712, MachoThrow2.cs:294,404,
  DuplicatePlatformEffect.cs:145-146, PlayerCollision.cs:370,382.
  Tags used: `ResizablePlatform`, `boulder`, `ChainLightning`, `beam`,
  `Player` (builtin), `PlayerSprite`, `SlimeTrailBlob`, `InvincibilityZone`,
  `PlayerSpawn`. Source of truth: export `"tag"` field (bopl-assets-v2) +
  `settings.json` tags. `CompareTag` on a destroyed object must throw/behave
  as Unity does — but no sim site does that (all null-check first). [EXACT]
- `gameObject.layer`: read ~67 sites in DetPhysics alone; **written at
  runtime**: SimpleSparkNode.cs:122,135,165,183,258 (ChainLightning toggles
  Lethal↔NonLethalTerrain — gameplay-critical), ThrowItem.cs:187,
  Roll.cs:342, plus raycast-buffer structs (ShootScaleChange.cs:91,
  ShootQuantum.cs:96, ShootDuplicator.cs:165). DPhysicsCircle.cs:421 copies
  `gameObject.layer` into the physics collider each frame. [EXACT]
- `LayerMask.NameToLayer`: 129 sites (MagnetGun 14, ShootDuplicator 11,
  PlayerCollision 9, BlackHole 9, …); `LayerMask.GetMask("wall",
  "LethalTerrain", "NonLethalTerrain")` DetPhysics.cs:2572,2631,2690;
  `LayerMask.LayerToName` FreezeRayGun.cs:94 (log only). Table from
  `settings.json` layers. [EXACT]
- `Physics2D.GetIgnoreLayerCollision(a,b)` DetPhysics.cs:2183,2252 = bit b of
  `collisionMatrix[a]` CLEAR (see extract_settings.py header). [EXACT]

### 2.4 Randomness

- The sim RNG is `Unity.Mathematics.Random` seeded via `Updater.InitSeed`
  (Updater.cs:45, :179-182), drawn through `RandomInt/RandomFix/
  RandomInsideUnitCircle/RandomUnitVector` (:184-214). Pure C# — copy the
  Unity.Mathematics implementation; bit-exact already proven by the Rust
  port (65k draws). [EXACT]
- `UnityEngine.Random` sites: GameSession.cs:75 (`RandomBagLevel` — level
  rotation BETWEEN rounds; replay/RL pins the level, never hit inside an
  episode), Client.cs:116 + Host.cs:575,616 (netcode), DeathEffect2.cs:30-31,
  SceneBoundArrowController.cs:97-98, DuplicatePlatformEffect.cs:256,
  SuperStarEffect.cs:217, Portal.cs:321 (visual scatter — all excluded or
  Unity-Transform-only). Shim `UnityEngine.Random` may be any deterministic
  stub; nothing compared consumes it. [EXACT that it is never needed in-episode;
  PRESENTATION otherwise]
- `Environment.TickCount` Updater.cs:245: local-play replay seeding only
  (guarded by `!GameLobby.isPlayingAReplay`); unreachable under RL/replay.

### 2.5 Time

- The sim clock is `GameTime.FixedTimeStep`/`Fix` + `Updater.SimTimePassed/
  SimTimeSinceLevelLoaded/SimulationTicks` (Updater.cs:308-315, :351-353).
  `GameTime.FixedDeltaTime(id, dt)` (GameTime.cs:41-52) scales by
  `PlayerTimeScale` (Fix) unless time-stop-protected. [EXACT]
- `Time.*` on the sim path: **only two non-cosmetic sites.**
  `OrbitItems.cs:95,123` uses `Time.fixedDeltaTime` inside `UpdateSim` — a
  constant; shim must return exactly `0.01666667f` (TimeManager "Fixed
  Timestep", in settings.json). [EXACT]
  `Missile.cs:50,174` stores `Time.time` (Init) and compares in `Update` —
  gates loop-engine AUDIO only; field `Missile.spawnTime` is already in
  presentation-fields.json. Shim `Time.time` may be a frame counter.
  [PRESENTATION]
- All other `Time.deltaTime/timeSinceLevelLoad/unscaledTime` reads (≈120
  sites: GameTime.cs:32-38,67 float path, SimpleSparkNode.cs:265,
  GrowthFlare.cs:65, beamAfterEffect.cs:33, PokeBall.cs:117,
  Portal.cs:293-316, AbilityReadyIndicator.cs:142, StockRespawn.cs:190,
  menus, Host netcode timers Host.cs:505-1106) live in `Update`/`LateUpdate`
  bodies or excluded fields. [PRESENTATION]

### 2.6 Unity lifecycle callbacks in sim classes (beyond Awake/Start)

`OnEnable`/`OnDisable`/`OnDestroy` in sim classes — all bodies are
particle/audio-only, verified by reading: Roll.cs:482-506, Drill.cs:247-274,
Missile.cs:67-75,141-158, MeteorSmash.cs:111-116 (destroys smoke particle
parent), DetPhysics.OnDestroy DetPhysics.cs:304-307 (**disposes the rope
NativeArrays — the shim/host must call or replicate this on world teardown or
leak**). Missile.OnEnable (:149-158) lazily `Object.Instantiate`s a smoke
particle — creates a presentation-only object (no FixTransform/MonoUpdatable;
does not perturb HierarchyNumber). The shim must still FIRE these callbacks at
the Unity-correct moments (SetActive toggle, DestroyImmediate) because they
run gameplay-adjacent code; stub bodies execute as compiled.
[PRESENTATION bodies, EXACT timing]

**No `FixedUpdate` exists anywhere. No `OnCollisionEnter/OnTriggerEnter/...`
Unity physics callbacks exist anywhere** (grep empty — collision callbacks are
game-owned `ICollisionCallback` via DetPhysics). The shim need not implement
either. [EXACT: their absence]

### 2.7 Coroutines / Invoke

- `StartCoroutine`: MainMenu.cs:92 and CharacterStatsList.cs:493-618 only —
  menus, never on the sim path. `InvokeRepeating`/Unity string `Invoke`:
  none. All `*.Invoke(` matches are C# delegates/UnityEvents
  (InstantAbility.cs:80 `funcOnEnter`, QuickSpell.cs:131 `spellFunc`,
  Player.cs:464,485 `playerDied/playerWon`, UI onClicks). Sim-time delays use
  `RoutineQueue` (Fix-based, gameplay code). Shim may throw on
  StartCoroutine. [EXACT: never called in sim]

### 2.8 SceneManager

- `SceneManager.sceneCountInBuildSettings` GameSession.cs:28,46 — level count
  = `count - 6`; shim must report **130** (124 levels + 6 fixed scenes;
  maps.json has 124 entries, Constants.FirstLevelSceneIndex = 6). [EXACT]
- `GetActiveScene().buildIndex/name`: GameSessionHandler.cs:121 (`< 6` menu
  guard), Host.cs:835,851,877 (RL pump gate `>= 6`), PerfHud (log).
  Scene `name` also appears in every trace state row (`scene` field) and in
  transform-path identities — must be `Level{k+1}` for map k. [EXACT]
- `LoadScene` on the reset path: CharacterSelectHandler_online.cs:345
  (`replayLevel + 6`, Single) + :353 (`winnerTriangle`, Additive);
  GameSessionHandler.cs:423 (next level after a round). Single-mode load
  destroys the previous scene's objects (except DontDestroyOnLoad) WITHOUT
  running per-object Destroy deferral. `HandleAbilitySelectUI.cs:183,221`
  subscribes/unsubscribes `SceneManager.sceneLoaded`. [EXACT for the reset
  sequence; menu loads PRESENTATION]

### 2.9 Statics with sim-visible state

- `Updater` statics (queues, clocks, `random`, `prefabsAdded`,
  `registeringAPrefab`, `LoadedALevelThisUpdate`) — reset by
  `Updater.ReInit()` (:125-136) + `InitSeed`. [EXACT]
- `Rope.ropeNumber` (reset in Host.ResetRl), `BlackHole.OnNewLevel()`,
  `MagnetGun` static lists (StaticUpdate), `ShootDuplicator.updatableIdBuffer`
  (ShootDuplicator.cs:72, scratch), `GameSessionHandler.selfRef` +
  static flags (captured in traces), `GameSession.currentLevel/playedLevelsBag`,
  `GameLobby.*`, `Constants.*` (immutable). The reset path lists what it
  clears (MANY_WORLDS.md §A2); anything not cleared must carry over
  identically. [EXACT]
- `Fix.SinLut/TanLut` (BoplFixedMath/Fix.cs:68, :20661): 205,887-entry
  `static readonly long[]` inline initialisers — deterministic data, no side
  effects. No static constructors with side effects exist in gameplay
  (`rg "static \w+\(\)"` — none). [EXACT, trivially]

## 3. Group [ORDER] — enumeration orders gameplay observes

- **`GetComponents<T>()` serialized order**: Ability.cs:72
  (`GetComponents<IAbilityComponent>()` — ability sub-component execution
  order), Orb.cs:270, Updater.cs:381 (checksum debug). Prefab serialized
  component order is in the export (`components` array). 
- **`GetComponentsInChildren<T>`** = depth-first, self first, serialized
  child order: GameSessionHandler.Awake platform snapshot (:152-176),
  POCHandler.cs:158 (PlayerSpawn scan), Updater.DestroyFix (:91),
  SlimeController.cs:477 (PlayerSprite lookup), Drill/MagnetGun/
  DuplicatePlatformEffect/MachoThrow2 (counts: Drill 8, MagnetGun 5,
  MachoThrow2 4 — see `rg GetComponentsInChildren`).
- **`transform.GetChild(i)` / `childCount`** = serialized order, mutated by
  SetParent/Destroy: gameplay-relevant reads at Arrow.cs:107 (**a platform
  with >80 children sheds arrows — child count is gameplay**), Arrow.cs:130,
  DPhysicsRoundedRect.cs:98-104, ResizablePlatform.cs:71-79,
  MachoThrow2.cs:453, DuplicatePlatformEffect.cs:298,341,
  StretchableLightning.cs:42, Updater.cs:101. New children append at the END
  of the parent's list (Unity SetParent semantics).
- **`transform.parent` / `SetParent`** as gameplay state: Arrow.cs:39-41
  (landing checks parent layer+tag, multiplies scale), :56-58, :100
  (`SetParent(collision platform)`); Spike.cs (5 uses), MachoThrow2.cs (5),
  GameSessionHandler.cs:270-362 (SpawnPlayers parents players to
  `playerSpawnsRoot`), DuplicatePlatformEffect (4), StockPips, ForceArrow,
  Explosion (2 each), NullifyParentRotation.cs:1. Reparenting must preserve
  Unity's "keep world position" default (`SetParent(p)` ≡
  `SetParent(p, worldPositionStays: true)`) — but note sim positions live in
  FixTransform, so only the hierarchy link and Unity-transform locals matter.
- **`FindObjectsOfType` / `Find`**: sim-path sites are
  GameSessionHandler.cs:777 (`PlayerAverageCamera`, presentation),
  Host.cs:1165 (`HandleAbilitySelectUI`, reset), plus StockPips, Beam,
  ControlPlatform, Mjolnir (6), Portal (5), BananaPeel (3) — Unity returns
  these UNORDERED in practice; every gameplay consumer either takes a single
  expected instance or re-sorts. Review each new use adversarially; the trace
  encoder sorts by identity, hiding order differences. Also
  `Resources.FindObjectsOfTypeAll` never appears outside instrumentation.
- **Deferred queues**: `destroyedObjects` before `destroyedMonos`, each FIFO
  (Updater.cs:339-350); `toBeAddedPrefab` flush order in `EndRegisterPrefab`
  (:143-153) — registration order within the prefab = Awake order.
- **Dictionary iteration**: gameplay uses `List`/arrays on the hot path;
  the trace encoder sorts dictionaries — but `DetPhysics`
  `instanceIdToColliderIndex` is a Dictionary used only for lookup (never
  iterated for gameplay). Confirm any NEW iteration during review.

## 4. Group [PRESENTATION] — safe to stub (bodies still compile and run)

- All `Update`/`LateUpdate` bodies in sim classes: SlimeController.cs:697
  (sound) / :438 (alpha), BoplBody.cs:257 (force-arrow visual), Rope.cs:239
  (mesh), Ability.cs:114 (alpha fade), FixTransform.LateUpdate:110
  (`SyncTransform` — copies Fix state into the Unity Transform; the shim can
  make it a no-op as long as `transform.position` is never read back by sim
  code — the only readers are presentation/Missile particles),
  GameSessionHandler.cs:471 (async scene-load gate — menu flow),
  Updater.Update:411 (auto-tick, disabled in replay: `autoUpdateAccum`),
  Beam.cs:467, BlackHole.cs:125, CastSpell.cs:275, Missile.cs:172,
  RocketEngine.cs:196, Mine.cs:100, PokeBall.cs:112, FrozenSolid.cs:91,
  StretchableLightning.cs:81, MagnetGun.cs:1027, MachoThrow2.cs:186,
  SimpleSparkNode.cs:263, DynamicAbilityPickup.cs:80, UseOrb.cs:121,
  TutorialGameHandler.cs:156, InputUpdater.cs:166, Host.cs:817 (netcode; the
  RL pump path within it IS the driver — port replaces it). The port never
  calls these per-frame methods unless it chooses to; presentation fields
  they mutate are trace-excluded (presentation-fields.json). BUT: fields
  only they write that are NOT excluded must stay at init values — matching
  Unity under `-batchmode` where Update still runs. **Preferred shim policy:
  run NO per-frame Update/LateUpdate at all; that matches the trace
  projection because every field such methods mutate is either excluded or
  audio/render-local. Any counterexample found in review = add to
  presentation-fields.json with audit.**
- Renderers/particles/audio/camera: `SpriteRenderer` (38 refs),
  `ParticleSystem` (23 in DetPhysics pools), `AudioManager.Get().Play*`
  (242 sites), ShakableCamera, `Material/Shader/MeshRenderer` — stub types.
  One exception is **`Sprite` identity**: `AbilityReadyIndicator.
  GetPrimarySprite()` results select revive/clone abilities via
  `abilityIcons.IndexOf(sprite)` (PlayerCollision.cs:257,273,
  SlimeController.cs:298,326) — sprite ASSET identity (reference equality
  vs the NamedSpriteList catalog) must be preserved by the loader even
  though rendering is stubbed. [EXACT identity, stub rendering]
- `Debug.Log/DrawLine` (63+ sites, Tools.cs:93-122 gated on
  `RlHeadless.Render`), `Gizmos`/`OnDrawGizmos` (editor-only),
  `OnValidate` (DPhysicsRoundedRect.cs:451, FixTransform.cs:184 — editor-only,
  never call), `FollowTransform` (visual follower; its `Destroy` at :81 only
  ever kills its own presentation object, but its Leader link is read by the
  trace identity `PathOf` — keep the component data, stub the movement),
  `AbilityReadyIndicator.SecondsSinceReset` (plain MonoBehaviour, sprite
  scale only), vibration `Player.<RumbleState>` (already excluded).

## 5. Known traps for the reviewer

1. The `$ref` cycle tokens in traces embed the FIRST JSON path an object was
   seen at — field enumeration order differences that are otherwise harmless
   change traces (see sim/Oracle/README.md).
2. `UpdatableComparer.Compare` (Updater.cs:12-21) calls `GetHashCode()` on
   both updatables and discards the values (dead reads) — Unity returns the
   instance id there; any shim hash is fine BECAUSE the results are unused.
   Do not "optimize away" the subtraction overflow either:
   `x.HierarchyNumber - y.HierarchyNumber` wraps for |Δ| > int.MaxValue
   (cannot happen with the +100000000 scheme unless prefabsAdded explodes).
3. `Updater.DestroyFix(GameObject)` flags direct-children FixTransforms only
   — a grandchild FixTransform stays `IsDestroyed == false` while its object
   is DestroyImmediate'd with the root. Reproduce, don't fix.
4. `GameSessionHandler.Awake` runs at scene load (before the first tick);
   its `Init()` runs from the first tick via `toBeAddedUpdatables` —
   two-phase. Spawning happens on RoutineQueue delays (1 s / 2.5 s of SIM
   time), so players do not exist until tick ~60.
5. Warmup is 240 ticks (`bopl_rl_default_config`) and traces include it.
6. `Physics2DSettings.m_AutoSyncTransforms = 0` and `m_SimulationMode = 0`
   (script-driven) — the shim must never auto-run 2D physics; DetPhysics is
   the only physics.
