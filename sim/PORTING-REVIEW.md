# Porting review checklist (adversarial)

How the shim/port will be judged. Details and every file:line citation live in
`sim/Oracle/SEMANTICS.md` (runtime behaviour contract) and `sim/Oracle/AOT.md`
(NativeAOT hazards); this page is the walk-through order for a reviewer.

The bar: `python sim/Oracle/diff.py <unity replayed.jsonl.gz> <port trace>`
exits 0 for levels 0-4, 2 players, ability group 0, 480 ticks
(fixtures: `sim/Oracle/fixtures/l{0..4}-p2/episode.rep`), then again for
coverage loadouts. Everything below is a known way to fail that bar.

## A. Scheduler and object model (SEMANTICS.md §1, §2.1)

- [ ] `TickSimulationInner` phase order reproduced exactly, including the
      first-tick-after-load early return BEFORE `DetPhysics.Simulate`.
- [ ] `Instantiate` runs `Awake` synchronously, serialized component order,
      parents before children; `Awake` deferred for inactive instantiations
      and fired on first `SetActive(true)` (the `InstantiateFixed` toggle
      trick depends on this).
- [ ] `RegisterUpdatable` ordering: prefab registrations get
      `HierarchyNumber += prefabsAdded + 100000000`; scene objects keep the
      serialized number; pending list sorted with `List<T>.Sort` +
      `x.HierarchyNumber - y.HierarchyNumber`.
- [ ] `DestroyFix` flags `IsDestroyed` exactly as Updater.cs:89-123 does —
      including the direct-children-only FixTransform quirk — and objects
      stay readable until the end-of-tick `DestroyImmediate` drain
      (`destroyedObjects` then `destroyedMonos`, FIFO).
- [ ] Raw `Object.Destroy` defers to end of ENGINE frame, not end of tick
      (visible when the host runs several ticks per frame).
- [ ] Destroyed objects compare `== null`, `!=`, and `(bool)` as fake-null;
      `Equals`/`GetHashCode` semantics match Unity under generic collections.
- [ ] `Start` runs once before the object's first Update-phase, only while
      active (Arrow.Start is gameplay).
- [ ] `OnEnable`/`OnDisable`/`OnDestroy` fire on SetActive toggles and
      DestroyImmediate (bodies are presentation, timing is not).
- [ ] No `FixedUpdate`, no Unity physics callbacks, no coroutines needed —
      shim should throw if gameplay ever reaches them.

## B. State the sim reads (SEMANTICS.md §2.2-2.5, §3)

- [ ] activeSelf vs activeInHierarchy propagation; `isActiveAndEnabled`
      gating of `UpdateSim`; `DetPhysics.IsActive` triple condition.
- [ ] Tags from export `tag` field; layers + collision matrix from
      `sim/Export/settings.json`; runtime `.layer =` writes respected
      (SimpleSparkNode toggles Lethal/NonLethalTerrain).
- [ ] `LayerMask.NameToLayer`/`GetMask` against the settings.json table.
- [ ] RNG = `Unity.Mathematics.Random(seed)` via `Updater.Random*` only;
      `UnityEngine.Random` never drawn inside an episode.
- [ ] `Time.fixedDeltaTime` == 0.01666667f (OrbitItems); no other `Time.*`
      on the sim path (Missile.spawnTime is excluded/audio).
- [ ] `GetComponents*`/`GetChild`/`childCount` follow serialized order;
      new children append last; Arrow's `childCount > 80` platform limit.
- [ ] `transform.parent`/`SetParent` hierarchy links (Arrow landing,
      SpawnPlayers) with worldPositionStays default.
- [ ] `SceneManager.sceneCountInBuildSettings` == 130; active scene name
      `Level{k+1}` for map k (appears in trace identity paths).
- [ ] Sprite ASSET identity preserved (AbilityReadyIndicator.
      GetPrimarySprite → abilityIcons.IndexOf drives revive/clone choice).
- [ ] Statics reset exactly like the reset path (Updater.ReInit, InitSeed,
      Rope.ropeNumber, BlackHole.OnNewLevel); anything else carries over.
- [ ] UnityEvent persistent calls dispatched (UseAbility/CastAbility/
      CastInvisibility) — loader fails loudly on unknown method names.

## C. Things the shim may stub (SEMANTICS.md §4)

- [ ] Per-frame `Update`/`LateUpdate` bodies not run at all (preferred), or
      run with all touched fields staying trace-excluded — any non-excluded
      field written by a per-frame method is a review finding.
- [ ] Renderers, particles, audio, camera, Debug, Gizmos, OnValidate: stubs.
      Verify no stub returns data gameplay branches on (e.g.
      `ParticleSystem.main.duration` feeds a delayed Destroy — presentation
      object, but must not throw).

## D. AOT (AOT.md)

- [ ] No reflection added by the shim itself (UnityEvent = dispatch table;
      NativeArray = managed wrapper; Unity.Mathematics vendored as source).
- [ ] Menus/Steam files (`SettingSelector`, `SteamSocket`, `SteamManager`,
      coroutine users) excluded from the sim compile set, or their
      reflection rooted deliberately.
- [ ] Tracer/oracle builds run on CoreCLR; the NativeAOT binary ships
      without Newtonsoft-based tracing.
- [ ] Fix LUT compile cost acceptable, else swap to embedded `.bin`
      (bit-verified against the arrays).

## E. Review method

1. Diff traces first (`diff.py --summary`), fixtures before fresh runs.
2. On divergence, bisect by tick and section; check the trap list
   (SEMANTICS.md §5) before reading port code: `$ref` path ordering,
   grandchild-FixTransform quirk, first-tick early return, warmup ticks,
   destroy-queue timing are the historical culprits.
3. Grep the port for `TODO`, `NotImplementedException`, and any shim method
   returning a constant where SEMANTICS.md marks [EXACT].
