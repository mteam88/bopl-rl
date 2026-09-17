# Ability-port semantic review resolution

Each entry records an independent reread of the C# oracle and Rust rollout
implementation. Commit hashes are intentionally left to Git history; every
fixed entry is committed separately with its finding id in the subject.

## Guns and spells

### Suspected guns/spells findings

- **S1 — fixed.** Roll's synchronous swept-circle callbacks now run
  immediately after that roller's own updater slice, before later player
  prefabs in global hierarchy order. They are no longer deferred until every
  player's updater has run, so an earlier Roll can kill a later-hierarchy
  player before that player's same-pass state transition.
- **S2 — fixed.** PlatformTransform still returns while its spawned wall is
  inactive in a Blink tunnel, but the separate SpriteAnimator on the ability
  form keeps running. Rust now lets that animator's completion callback move
  Rotation to Size even while the wall is inactive, matching C#.
- **S3 — fixed.** A rope attached to BlackHole now stores and reapplies its
  offset at `FixTransform.Scale == 1`, and initializes rope mass with that
  same scale. BlackHole growth changes the Unity visual scale and DPhysics
  circle radius directly; it never changes the authoritative FixTransform
  scale used by `RopeAttachment`.
- **S4 — fixed.** Portal entry marking and transit now reject every modeled
  Magnet-held BoplBody (arrows, thrown items, missiles, and both spell
  markers). A held object no longer acquires a stale inside bit or teleports,
  matching `Portal.IsPortalable`'s `!body.IsBeingHeld()` predicate.
- **S5 — fixed.** Revival and Teleport markers now pass the same strictly
  positive live `BoplBody.InverseMass` gate as every other portalable body,
  both when recording entry and when performing transit.
- **S6 — fixed by the marker-physics correction.** Teleport markers retain
  their live `BoplBody`, receive gravity, and integrate position on every
  DetPhysics pass alongside Revival markers. They are no longer positionally
  static between placement and activation.
- **S7 — not a divergence in the modeled object set.** Every represented
  movable teleport target owns one registered teleportable collider and thus
  contributes one `PhysicsParent` to each C# `CircleCastAll`. Composite
  platform children are separate transforms and the attachable ones take
  `teleportObject`'s early return; trigger-only auxiliary rows have no
  `SpriteRenderer` and return before mutation. Rust's one entity hit therefore
  produces the same single positional shift for every reachable target.
- **S8 — fixed.** Each equipped Magnet prefab now retains its own mutable
  `FirepointOffset` and previous Ability scale. `MagnetGun.UpdateSim` uses the
  prior value, then the later `Ability.UpdateSim` multiplies it by
  `player.Scale / prevScale`; this preserves both hierarchy timing and Q32
  telescoping instead of recomputing from the authored offset at fire time.
- **S9 — fixed.** Non-player ScaleChanger targets now clamp against their
  authored collider bounds: `0.3..4` for SpikeAttack and duplicated Rock,
  `0.6..5` for SimpleSparkNode, `0.15..1000000` for BlackHole2, and the
  default `0.1..5` elsewhere. Spawned ResizablePlatform uses its authored
  maximum `6` instead of the map-platform default `5`.
- **S10 — not a divergence in the 39-ability lifecycle.** The only reachable
  invulnerability bit while a rollout-visible target collider is active is
  Poke Ball's mask-16 capture, which Rust checks. Spawn mask 4 is confined to
  the hidden replacement/spawn phase, and Machoman's mask-2 block belongs to
  an unported ability (distinct from the ported Macho Throw), so neither can
  be struck by Freeze through this rollout ABI.
- **S11 — not a divergence after the confirmed Engine animator fix.** In the
  firing phase Rust ignites overlapping Smoke, evaluates the end-of-curve and
  water conditions, and only then applies the zero-clock Time Stop return.
  Thus an already-ended or submerged Engine enters its stopping animation
  during Time Stop exactly as `RocketEngine.UpdateSim` does; only animation
  advancement and force application remain frozen.
- **S12 — not a divergence in the exported rollout maps.** Every one of the
  33 extracted platform `AnimateVelocity` components has positive speed and
  serialized `HomeIsMovable = true`, so Rust's firing-phase home-position
  update reaches exactly the same set. `inRotationMode` is private, starts
  false, and `ActivateRotationMode` has no gameplay caller in this build, so
  the source always updates `homeRotation` too.
- **S13 — fixed.** Rebuilding a platform composite now clears the retired
  combined body's accumulated force. C# reconstructs from child bodies whose
  force was cleared on attachment, then explicitly copies only velocity and
  angular velocity from the old composite; carrying Rust's old force applied
  an extra impulse after any child removal.
- **S14 — fixed.** Rocket Engine now ignites the first overlapping Smoke
  circle in DetPhysics registration order (the source's `scanForSmokeBuffer[0]`)
  instead of the first active slot in Rust's reusable smoke pool. Reusing an
  earlier pool slot can no longer make a newer cloud win the overlap.
- **S15 — fixed.** Removing a platform now destroys each attached Tesla node
  through the same disconnect path as `SimpleSparkNode.OnDestroyUpdatable`.
  Partner nodes on surviving platforms lose their parent/child link and
  lethality, and their lightning is retired, instead of retaining a stale
  live connection to an inactive node slot.
- **S16 — fixed.** A grounded Drill hit by Gust now follows Shockwave's
  `PlayerBody` fallback because the Drill prefab has no `PlayerPhysics`.
  Rust no longer applies `groundedPlayerForceMultiplier`; it applies the
  ordinary PlayerBody force conversion exactly like an airborne Drill.
- **S17 — not a divergence.** The suspected target is misclassified:
  `orbAnim.prefab` is serialized on layer 18, not layer 26. Gust's authored
  mask contains bit 26 but not 18, and Magnet's pull/pickup masks likewise do
  not contain bit 18, so neither C# query reaches a live Beam orb. Rust omits
  the same target from both scans.
- **S18 — not a divergence.** `platform_dynamics[i].body` is the live combined
  composite body whenever attachments exist; the uncombined authored child is
  stored separately as `root_body`. Applying Magnet's off-centre impulse to
  `.body` therefore matches `DPhysicsRoundedRect.AddForceAtPosition`, which
  redirects through `compositeParent.combinedBody` in C#.
- **S19 — resolved by confirmed finding #37.** NativeAOT preserves the
  destroyed Revival component wrapper in `Player.RespawnPositions`, and its
  retained `FixTransform` still supplies `GetPos()`. Rust now retains and
  consumes the same stale marker position instead of eliminating the player.
- **S20 — deferred: requires the per-updatable Time Stop model.** The remaining
  transformed forms cannot be corrected safely by another shared branch:
  their `Ability`, `PlayerPhysics`, `PlayerBody`, component, and animator
  updaters opt into `GameTime.FixedDeltaTime` independently. The required
  change is to keep raw `DT` at the scheduler, assign clock ownership to each
  modeled updater, and gate/integrate each component in hierarchy order from
  its prefab's behavior. Gust and Time Stop's confirmed no-gravity symptoms
  are fixed locally; the un-audited forms remain explicitly non-exact.

- **#1 — fixed.** C# keys permanent duplicated platforms by the casting
  Duplicator hierarchy. Each new platform from that gun now schedules the
  previous copy for the same 80-pass delayed removal (`60` shake passes plus
  the authored `20`-pass grace), while the new copy receives the permanent
  `i32::MAX` lifetime. The expiry path also mirrors duplicated-object shake
  displacement and removes the old platform from physics.
- **#37 — fixed.** Death now consumes an owned Revival marker reference even
  when the marker object was already destroyed by a Black Hole. C# leaves the
  destroyed component in `Player.RespawnPositions`, and `killPlayer` tests the
  list count before calling the stale indicator's retained `GetPos()`.
- **#7 — fixed.** Grounded Time Stop now checks button release during the
  initial `idleToFall` animation, matching `CastSpell.UpdateSim`'s release
  check on every pass rather than forcing the first twenty animation ticks.
- **#22 — fixed.** Gust shockwave curves now advance with raw `DT`, including
  during Time Stop. `Shockwave.UpdateSim` receives `simDeltaTime` directly and
  never calls `GameTime.FixedDeltaTime`; only the affected physics integration
  is frozen.
- **#24 — fixed.** Gust now captures layer-22 Smoke circles in its fixed
  target set and applies every curve impulse to the cloud's `BoplBody`.
  Smoke clouds now retain and integrate that body state, so the impulse moves
  the collider on subsequent physics passes exactly like the C# prefab.
- **#25 — fixed.** Gust activation now dispatches non-attachable layer-16
  duplicated Rock bodies through Shockwave's `BoplBody` fallback. They receive
  the source `9000`-multiplied force once and are deliberately not retained
  for later curve samples.
- **#26 — fixed.** Gust now builds the source 50-entry hit buffer before any
  type-specific handling: circles, then boxes, then rounded rectangles, each
  in collider-registration order. Player, unpushable-form, LethalTerrain, and
  otherwise ignored hits consume slots exactly as they do in
  `DetPhysics.CircleCastAll`; abilityless player clones also receive the
  ordinary activation push when retained by that shortlist.
- **#23 — fixed for the confirmed forms.** Frozen airborne Gust and Time Stop
  forms now return before gravity, aerial steering, and velocity damping, as
  `QuickSpell` and `CastSpell` do. Other active forms retain their authored
  per-component frozen behavior rather than applying this exception globally.
- **#41 — fixed.** Rocket Engine start and end SpriteAnimator counters now
  stop when global time is stopped. The firing phase was already gated; all
  three phases now match the environment-owned C# components.
- **#38 — fixed.** Time Stop's exit offset now multiplies by the caster's live
  player scale, matching `CastSpell.ExitAbility`.
- **#6 — fixed.** Roll now scales its physics radius, lethal hurtbox radius,
  swept-contact radius, platform-hop geometry, and exit radius/offset by the
  player's live scale.
- **#4 — fixed.** Grounded Roll now calls the ordinary zero-input
  `PlayerPhysics.Move` path on every charging pass. This decays the inherited
  grounded speed, updates its retained surface-local coordinate, and performs
  the same live-platform/drop checks as C# before release.
- **#5 — fixed.** Grounded Roll exit now replaces the ordinary slime's
  grounded speed with the magnitude of the form's total `PlayerBody.Velocity`
  (self-imposed plus external), matching the `AbilityExitInfo` populated by
  C# instead of retaining pre-entry motion.
- **#2 — fixed (report heading reversed the two values).** Growth/Shrink and
  Blink now self-hit when the initial muzzle is at or below C#
  `SceneBounds.WaterHeight` (`-11.3f`). Rust previously used the exported
  water collider top (`-11.95`), so muzzles in the intervening band fired an
  ordinary ray. The report's described gameplay consequence and cited C#
  lines were correct; only its opening value attribution was backwards.
- **#3 — fixed.** An initial Duplicator shot at or below
  `SceneBounds.WaterHeight` now replaces the ray result with the casting gun's
  own Player body and spawns the self-clone upward. Reflected water hits keep
  the existing recursive no-water-mask path.
- **#13 — fixed.** Growth/Shrink, Blink, and Freeze muzzle point checks now
  keep the last overlapping rounded rectangle in DetPhysics registration
  order. Their prior iterator `find` selected the first; Duplicator already
  implemented the source's overwrite-on-each-overlap behavior correctly.
- **#16 — fixed.** Generic-object ScaleChangers now retain scale by entity
  identity and synchronize concurrent effects after every write. Sequential
  hits therefore start from the victim's live `FixTransform.Scale`, while
  simultaneous Growth/Shrink effects compose additively in updater order as
  C# does instead of advancing independent accumulators from one.
- **#17 — fixed.** Generic circles/boxes and Smoke clouds now reproduce
  `DPhysicsCircle.Scale`'s Q32 grouping exactly: compute
  `(Fix.One / oldScale) * newScale` first, then multiply each collider
  dimension. The algebraically equivalent `new / old` and
  `(radius / old) * new` forms round differently in fixed point.
- **#19 — fixed.** Magnet's three held-item wall rays now exclude the held
  Macho boulder's rounded-platform collider. This mirrors the source
  temporarily deactivating an item GameObject around both `RaycastForWalls`
  and `UpdateHeldPosition`, preventing the boulder from clamping itself.
- **#20 — fixed (the report mislabeled this as Scale-ray/Blink).**
  Duplicator's platform spawn-position helper now replaces a degenerate
  rounded rectangle's rotated axes with world right/up when both extents are
  below `1/20`, exactly as `ShootDuplicator.PlatformSpawnPos` does.
- **#21 — fixed.** Blink's form entity is now included in all three death
  cleanup paths (Revival replacement, stock replacement, and elimination).
  It therefore receives the same removed/destroyed transition as every other
  ability body instead of remaining an active ray-blocking ghost.
- **#27 — fixed.** Scale rays now dispatch Teleport markers through their
  `ISpell.AddScale` semantics: only the private teleport-area/particle scale
  changes, clamped to the authored `0.1..6`; collider radius and inverse mass
  remain untouched. Revival's `AddScale` intentionally delegates to its
  BoplBody scale and retains the ordinary `0.1..5` collider path.
- **#14 — fixed.** When Magnet steals a player from an active ability form,
  the forced exit packet now relocates the restored slime to the stealing
  gun's current firepoint before ragdoll/hold state is applied. Ordinary
  slime victims retain their existing position, matching the source branch.
- **#15 — fixed.** Local C# `(Fix)X` double literals now use
  `Fix::from_f64`, including the four reported clone-pop, duplicated-Tesla,
  Blink-duration, and airborne-Blink-bound constants. A whole-port audit also
  corrected the same error in live Player/ScaleChanger clamps, ability-pickup,
  Roll, Meteor, Mine, Push, Magnet, and Hookshot arithmetic. Authored Unity
  `Fix` fields and explicit `(Fix)Xf` literals remain `Fix::from_f32`, because
  those values originate as single-precision data in the C# oracle.
- **#8 — fixed.** Duplicated Rocket Engines now advance seven world units
  around the parent platform perimeter before attachment. The prior `0.3`
  argument was the separate surface offset, not Duplicator's placement
  distance.
- **#9 — fixed.** Duplicated Black Holes now enter DetPhysics' duplicate
  expiry list with the explicit source lifespan of 480 passes (eight seconds),
  rather than inheriting the generic 1800-pass object lifetime.
- **#10 — fixed.** A duplicated Black Hole now starts with
  `previous_target_size = 100`, matching the `BlackHole.Init` call made by
  Unity instantiation before Duplicator assigns its copied mass. Its first
  growth interpolation therefore has the source radius and pull/eat reach.
- **#11 — fixed.** Grounded Engine and Tesla exits now reproduce
  `PlantObjectOnPlatform.ExitAbility`'s `enterTurn` packet: forward input at
  or beyond the ground-alignment deadzone enters the slime's stop animation,
  while reverse or neutral input enters idle. This preserves the source state
  transition before the next directional cast.
- **#12 — fixed.** Engine ring eviction now happens after the replacement
  engine is attached and the cast impulse is applied. Every fifth cast thus
  computes that impulse against the temporary root-plus-five-child composite,
  then rebuilds the platform without the destroyed oldest child before the
  physics pass, matching `PlaceRocketEngine.Place` and DetPhysics cleanup.
- **#18 — fixed.** Duplicated Engine and Tesla attachments now scale their
  authored surface offsets by the cloned object's scale before rebuilding
  composite-relative geometry.
- **#28 — fixed.** Teleport marker collider radius now remains at its authored
  value when the owner is scaled. The separate teleport-area radius continues
  to use the marker's logical particle scale, matching `TeleportIndicator`.
- **#29 — fixed.** Freeze animation exits and shared gun landing exits now
  restore the ordinary slime's scaled radius/body geometry, matching the
  already-correct scale-ray animation exit.
- **#30 — fixed.** Magnet's layer-27 pull and pickup masks now reach both
  Revival and Teleport marker bodies. Each spell can be pulled, selected by
  nearest FixTransform position, stolen, held without item wall raycasts,
  and fired with `FireForce * BoplBody.Scale` (Revival's live scale and
  Teleport's unchanged transform scale), matching the source spell branch.
- **#31 — fixed.** Blink now recognizes abilityless clone Player entities as
  players rather than generic objects. Hits apply the player cooldown and
  1.2-second player tunnel, detach through the player relocation path, use
  the player release-position RNG/geometry, and reactivate after the same
  strict lifetime boundary instead of taking the four-second object path.
- **#32 — fixed.** Tesla nodes now retain separate raw FixTransform and
  clamped DPhysicsCircle scales. Surface/hitbox offsets use the raw caster
  scale while collider radius honors the authored `0.6..5` clamp.
- **#33 — fixed.** Removing a platform now shifts every pending
  `DuplicatePlatformEffect` source index above it, and cancels an effect whose
  source itself was destroyed. An in-flight effect therefore cannot silently
  retarget the platform that shifted into the removed vector slot.
- **#34 — fixed.** Each live `DuplicatePlatformEffect` now applies C#'s
  per-pass `Fix.Lerp(scale, 1, (Fix)0.2)` downward decay to both its private
  effect scale and the source wall's DPhysics scale. The destination is
  recomputed from the captured center/basis and current effect-scaled
  extents/radius before overlap forces and the eventual clone spawn.
- **#43 — fixed.** Equal-mass Black Hole merges now compare monotonic prefab
  hierarchy allocation order, the rollout equivalent of C#'s incrementing
  `BlackHoleIndex`/sorting order. Reusing an earlier vector slot can no longer
  make an older hole consume the later-created one.
- **#42 — fixed.** Portal shots now retain `Fire`'s requested velocity and
  reassert it on their first scheduled `UpdateSim`, before advancing flight
  time. Birth-pass Black Hole, Magnet, Gust, or contact perturbations no
  longer survive the source's one-time `fireApplied` reset.
- **#40 — fixed.** Ring-evicted Tesla nodes now retain an immutable contact
  snapshot through the current DetPhysics callback phase. The old node circle
  and lightning box therefore remain lethal to players, Mines, and Smoke for
  the eviction pass, then disappear after contacts when C# runs the node's
  `OnDestroyUpdatable` teardown.
- **#44 — fixed.** Grounded Blink release still chooses its destination from
  active rounded rectangles, but the 20 sampled player positions are now
  blocked by every registered rounded rectangle, including inactive ones.
  This mirrors the source's intentionally unfiltered inner overlap loop and
  preserves both fallback placement and its two RNG draws per attempt.
- **#46 — fixed.** A FrozenSolid whose victim leaves playable state now runs
  the same gameplay portion of `Shatter` as an ordinary thaw: it decrements
  the live block count, applies the 0.35-second upward ragdoll packet, and
  resets all three ability cooldown timers. Death no longer merely discards
  the effect state.
- **#47 — fixed.** Pending platform-duplicate effects now use a growable pool,
  and completed-spawn requests use a growable queue. The 33rd concurrent
  effect is instantiated instead of being silently dropped, and more than 32
  effects can complete on one pass without indexing a fixed array out of
  bounds.
- **#45 — fixed.** Duplicate shake vectors are now sampled from the one
  cross-type creation sequence, newest first, before per-type state updates.
  Each live duplicate therefore receives the same RNG draw as its entry in
  `DetPhysics.duplicatedObjects`, while retaining the existing allocation-free
  object pools and type-specific expiry behavior.
- **#35 — fixed.** On the pass after a scale ray changes a spawned
  `ResizablePlatform` collider, its updater now recomputes inverse mass and
  moment of inertia from the live extents and bevel radius. This follows
  `ResizablePlatform.UpdateSim` calling `ResizePlatform` every pass rather
  than applying `BoplBody`'s ordinary linear scale adjustment.

## Projectiles

- **#1 — fixed.** `ThrowItem2.Fire` now applies caster scale to held-item
  dummy poses, release offsets, launch impulse, and the owner-immunity delay
  (with the source's `Max(scale, 1)` rule).
- **#2 — fixed.** Mine's instantiated payload offset now scales both authored
  local axes by the caster's live scale, matching `ThrowItem.Fire`.
- **#3 — fixed.** Guided missiles now inherit the caster's scale for both the
  spawn offset and live projectile collider/visual scale.
- **#4 — fixed.** Meteor's lethal smear half-width now multiplies by the
  caster's live scale; its travel-derived length remains unchanged.
- **#6 — fixed.** Push entry now preserves the captured ordinary slime
  `groundedSpeed` through `PlayerPhysics.SyncPhysicsTo`; only its self-imposed
  and external velocity channels are cleared by `ControlPlatform.OnEnterAbility`.
- **#5 — fixed.** Every active Push pass now calls zero-input
  `PlayerPhysics.Move` immediately after the authored `0.8` external-velocity
  damping. This restores grounded friction, surface-local movement from
  inherited force, destroyed-ground detachment, and platform drop checks.
- **#7 — fixed.** During Time Stop, a protected owner's Mine still advances
  its raw-delta hover clock and can transition from off to armed, but the
  armed, chasing, and anticipation states now honor their separate global
  stop checks. They no longer scan, steer, or count down merely because their
  owner is protected.
- **#8 — fixed.** Missile form now performs its airborne landing ray and
  exits immediately before touching the live spark. Stabilization/steering is
  dispatched only after the form survives `PlayerPhysics.Move`, so a landing
  pass no longer rotates the projectile or clears its instability counter.
- **#9 — fixed.** All five Rock callback paths now gate BounceBall's extra
  Player/Projectile impulse on the Rock having no attached rope. Player,
  abilityless-clone, Rock/Rock, Arrow, and Dash contacts still resolve and
  apply their gameplay callbacks while Web Swing/Grapple owns the body, but
  no longer add the source-suppressed bounce force.
- **#10 — fixed.** Poke Ball's proximity scan and Player-layer collision
  callback now include live Duplicator abilityless clones. Clone captures
  carry their own invulnerability/captor state, follow the ball without
  receiving shared input, hide the owning player as C# does, and use the same
  timed ejection packet as an ordinary slime victim.

### Suspected projectile findings

- **S1 — not a divergence.** Before Rock's late ability update applies its
  next curve value, the player update divides `collider_radius` by the prior
  scale and resets `scale` to one. The late multiplication therefore performs
  the same old-scale removal as `DPhysicsCircle.Scale`; it does not accumulate
  across passes. The arrow contact path's authored-radius expression is also
  equivalent to the live collider radius after that late update.
- **S2 — fixed.** Grounded Rock entry now stores `prevPlayerVel` from the
  unscaled incoming self-imposed plus external velocity before applying the
  authored `0.6667f` body-velocity factor. A rope's first force sample thus
  uses the same delta as `BounceBall.Initialize`.
- **S3 — fixed.** Cast Spike attacks now retain the owning player/ability-slot
  reference, while Duplicator copies explicitly have none. Every destruction
  route clears the slot only when it still points at that pooled entry, so a
  later reuse cannot make the original caster remove another player's Spike.
- **S4 — fixed.** Grounded Spike entry now performs the material part of
  `PlayerPhysics.SyncPhysicsTo`: it clears self-imposed velocity while keeping
  the captured external velocity and grounded speed. A same-pass shove can no
  longer leak into the Spike form's grounded body channel.
- **S5 — fixed.** Drill's moving-platform angular contribution now measures
  the player from the rounded root's synchronized `FixTransform` position,
  not the composite center. Child-collider positional correction can move the
  latter independently during DetPhysics, exactly the case where the two
  source positions differ.
- **S6 — not a divergence.** The authored `drillables` mask is exactly layer
  11. Every gameplay physics collider on that layer (map terrain, Boulder,
  and ResizablePlatform) is a `DPhysicsRoundedRect`; the layer-11 water-effect
  objects have no DetPhysics collider. `CollideBox` therefore reaches only its
  rounded-rectangle loop, whose registration order is the same insertion and
  stable-removal order as Rust's `platforms` vector. Its first hit is always a
  `StickyRoundedRectangle`, so the source's conditional assignment does not
  retain a different previous platform.
- **S7 — fixed.** Arrow overlap is no longer treated as an explosive igniter.
  The Arrow prefab's `Projectile.IgnitesExplosives` is false, so C# never
  detonates Grenade or Smoke from that callback even if an active registered
  arrow survives until the later item-contact pass.
- **S8 — fixed.** A ThrowItem2 form that entered grounded now exits before
  processing release/fire when its live PlayerPhysics has lost the platform.
  Mine's older ThrowItem form instead retains its animation path but now uses
  live grounded state for movement, exit handoff, and `Fire`; an aerial entry
  landing during anticipation therefore launches from the surface normal at
  the grounded strength rather than using the stale aerial packet.

## Cross-cutting

### Suspected cross-cutting findings

- **S1 — fixed.** Each player now records the pass in which a collision
  callback first reached the kill path. Later callbacks for that same body in
  the same pass return, matching `PlayerCollision.wasKilledThisFrame`; this is
  observable when the first contact schedules a Revival clone and restores
  the shared player's alive flag before the queued callback list is exhausted.
- **S2 — fixed after correcting the premise.** `SmearFramePrefab` is
  `MeteorSmear_explo`, instantiated as an unparented root with `Explosion` and
  no `Projectile` or ancestor `Ability`. Its source id therefore remains zero,
  but the row now carries `E_EXPLOSIVE` rather than the incorrect
  `E_PROJECTILE` component flag.
- **S3 — not a divergence.** Beam rows are emitted by `RlObservation` with a
  null transform, so they can never inherit an `Ability`. Tesla's chain
  lightning and BlackHole2 are both instantiated as unparented root prefabs
  and neither prefab contains `Ability`; they likewise do not receive
  `E_CONTACT_DAMAGE_KNOWN` or a source id in C#. Rust's zero metadata matches.
- **S4 — not a divergence.** The `ResizablePlatform` component GUID appears
  only on the spawned ability prefab (and is referenced by Platform ability),
  not on any of the 125 shipped map scenes. Map terrain therefore never enters
  `PlatformAbility`'s ResizablePlatform fallback, and its zero source id is
  correct; spawned ResizablePlatforms retain their explicit ability source.
- **S5 — fixed.** PortalShot and Portal clocks now advance by the already
  player-scaled updater delta exactly once. This matches each C# component's
  raw `SimDeltaTime * GameTime.PlayerTimeScale` expression at fractional
  reference-harness scales instead of squaring the scale.
- **S6 — not a gameplay divergence.** C# does mistakenly test p1's callback
  while deciding whether to enqueue p2. At queue construction, however,
  `RemoveDestroyedColliders` has already removed every collider whose
  GameObject was destroyed before the pass, while destruction from current
  collisions cannot occur until the fully built queue is dispatched. The
  ported gameplay never destroys a callback component independently of its
  collider GameObject, so no live pair can expose the typo.
- **S7 — not a divergence.** `Player.UpdatePlayer` only advances invisibility
  clocks and rotates the position-setter references; it has no death or stock
  path. Every death remains in later updatables, DetPhysics callbacks, or the
  scene-boundary phase, so evaluating Rust's pending match result before its
  player loop cannot delay a result caused by `UpdatePlayer`.

- **C1 — not a divergence.** The `TimeStop` metadata branch in
  `RlObservation.Entity` is unreachable in the C# oracle. `Write` emits only
  DetPhysics circles, boxes, rounded rectangles, rope segments, and beams;
  `Assets/GameObject/TimeStopSphere.prefab` contains only `Transform`,
  `SpriteRenderer`, `TimeStop`, and `FixTransform`, with no DetPhysics
  collider. NativeAOT therefore has no Time Stop entity row either, and its
  `obs[63]` / `obs[64]` remain zero exactly as Rust's do. Adding a synthetic
  Rust-only row would create the distribution split the report intended to
  prevent.

- **C2 — not a divergence.** The report conflated the generic `Orb` used by
  `UseOrb` with Beam's visual orb. Beam instantiates `orbAnim.prefab`, whose
  components include `DPhysicsCircle`, `Explosion`, animation, and
  `DisableIfNotHeld_deter`, but no `Projectile`. A fresh NativeAOT build
  confirms the live layer-18 Beam circle lacks `E_PROJECTILE`, matching Rust.

- **G3 — fixed.** `Player.Kill` returns immediately when `WonThisRound` is set
  (`sim/Gameplay/Player.cs`), before rumble, death accounting, telemetry, or
  killer credit. `RolloutWorld::kill` now performs the same guard before any
  exit/drop behavior. `native/src/player_life.rs` remains the independently
  verified accounting oracle; merging the two implementations would be a
  larger refactor and is not needed for this localized correction.
- **G1 — fixed.** `projectile_credit` now mirrors the C# Hitbox fallback for
  otherwise-unclassified Explosion/LethalTerrain rows by returning the
  entity's published owner id with owner damage enabled. Dropped lethal Rock
  bodies hit by Meteor/Roll therefore credit their player instead of the
  environment.
- **G2 — fixed.** An uncredited Missile explosion now returns owner id zero,
  matching the prefab's untouched `Explosion.PlayerOwnerId`, at the shared
  Meteor/Roll credit dispatcher too. Its victims classify as `deaths_other`
  instead of environmental deaths at all three collision sites.
- **G4 — fixed.** Smoke clouds now retain the spawning ability slot and
  ability-generation identity. Each non-duplicated burst evicts against the
  one 40-entry FIFO shared by that player's live Smoke ability instance,
  rather than a private ring keyed to the individual grenade cast.
- **G6 — fixed.** A last-slot Grappling Hook or Web Swing now suppresses its
  death pickup while C#'s rope-icon override is active: either the live rope
  reports zero cooldown, or another player's box overlaps the authored rope
  proximity circle. The launch-direction RNG sample is still consumed before
  the failed icon lookup, preserving `DropAbilities` ordering exactly.
- **G7 — fixed.** Pickup contact dispatch now applies `AddAdditionalAbility`
  to every ordinary slime whose collision was queued in the physics pass.
  The pickup is destroyed only after those callbacks, so simultaneous
  overlaps instantiate one replacement ability for each player in callback
  order just as C# does.
- **G8 — fixed.** Expiring Time Stop effects now preserve zero player time
  scale and the last protection bits once the session has ended. C# still
  destroys the effect object, but `TimeStop.End` returns before its teardown
  writes when `HasGameEnded()` is true; Rust now makes the same distinction.
- **G9 — fixed.** Time Stop effects are no longer inserted into the first
  inactive vector slot. They append in creation order like Unity's newly
  instantiated updatables, so overlapping effects write time protection in
  hierarchy order and the newest surviving stop wins the final write.
- **G10 — fixed.** A one-player session now terminates on that player's
  elimination and marks the same player as the round winner. This mirrors the
  C# debug-session branch, which calls `WinGame` for the sole player's team
  after its alive count reaches zero.
