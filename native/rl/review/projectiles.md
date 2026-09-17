All four reviews returned and I've re-verified the load-bearing claims myself. Report follows.

---

# Ability port review — Grenade, Missile, Spike, Smoke, Meteor, Throw, Push, Mine, Drill, Poke ball, Tack, Mjolnir, Rock

**Summary.** 10 confirmed divergences (each re-read by me on both sides) and 8 suspected. The dominant theme, and the most consequential finding, is that **the caster's Growth/Shrink scale is not propagated into spawned projectiles and form-local geometry**: `ThrowItem2.Fire` drops three separate `player.Scale` factors (spawn offset, launch impulse, owner-immunity window), `ThrowItem.Fire` drops it from the Mine spawn offset, `SparkShot` spawns its missile with `scale: Fix::ONE` hardcoded, and the Meteor smear's half-width is unscaled. This single family touches 8 of the 13 abilities and fires on **every cast by a scaled player**, changing both where a projectile appears and how fast it leaves. It is not in FULL_COMPATIBILITY.md; the port is internally inconsistent about it (the forced-release sibling path *does* apply scale), which is strong evidence it is an oversight rather than a modelling decision. The second tier is Push, which omits `PlayerPhysics.Move` entirely and fires on every cast begun while moving.

---

## CONFIRMED divergences

**1. `ThrowItem2.Fire` drops caster scale — 3 factors + held-item pose.** Affects Grenade, Smoke, Tack, Mjolnir, Poke ball.
- C# `sim/Gameplay/ThrowItem2.cs:220` `player.Scale * (isGroundedThrow ? gThrowPosition : aThrowPosition)`; `:223` `DelayedEnableHurtOwner(DisableGrenadeHitBoxTime * Fix.Max(body.fixtrans.Scale, Fix.One))`; `:229` `AddForce(body.fixtrans.Scale * fix * vec, Impulse)`; `:290` (`dummyPos`) `player.Scale * groundedDummyPositions[...]`.
- Rust `native/src/rollout/mod.rs:17079-17083` (`let local = ... * local_sign` — no scale), `:17090` (`let impulse_force = direction * force * Fix::from_int(60)`), `:17098` (`item.hurt_owner_delay = gd.hurt_owner_delay`), `:16894-16903` (dummy pose, no scale).
- The item's own mass/radius *are* scaled (`:17020-17022`), so the velocity error is a full factor of `scale`, not just force. `release_held_throw_item_on_exit` at `:17158-17161` applies `scale` correctly — the inconsistency is internal.
- Fires: every Grenade/Smoke/Tack/Mjolnir/Poke ball release at `player_scale != 1`. Grown casters under-throw, shrunk casters over-throw, and grown casters lose owner-immunity too early.

**2. Mine spawn offset drops caster scale.**
- C# `sim/Gameplay/ThrowItem.cs:213` `Vec2 vec = throwPosition * body.fixtrans.Scale;`
- Rust `mod.rs:16740-16742` `let position = ...base.position + side * data.mine.throw_position.x + up * data.mine.throw_position.y;`
- `throw_position` is (1.46, 1.29); at scale 3 the mine should appear ~2.9 units further out. `boplBody.Scale` itself *is* ported (`:16776`).
- Fires: every Mine cast by a scaled caster.

**3. Guided missile ignores caster scale entirely.**
- C# `sim/Gameplay/SparkShot.cs:111-113` `position += Vec2.ComplexMul(new Vec2(body.rotation), body.fixtrans.Scale * projectileOffset); ... sparkBody.Scale = body.fixtrans.Scale;`
- Rust `mod.rs:16400-16404` `position: ...base.position + data.missile.projectile_offset.complex_mul(right), ... scale: Fix::ONE,`
- `GuidedMissile.scale` is thereafter written only by a ray hitting the missile (`:21817`), so a grown caster fires a normal-size spark with a normal-size hitbox (`spark_extents * missile.scale` at `:23228`, `:24695`) from the wrong muzzle point.
- Fires: every Missile cast by a scaled caster. `scale_missile_ab.c` scales the *live* box post-spawn and cannot catch this.

**4. Meteor smear half-width unscaled.**
- C# `sim/Gameplay/MeteorSmash.cs:310-311` `extents.y = Vec2.Magnitude(vec) * (Fix)0.5f; extents.x *= fixTrans.Scale;`
- Rust `mod.rs:8319` `let extents = Vec2::new(data.meteor.smear_extents.x, distance * Fix::from_f32(0.5));`
- Length and derived offset are correct; only width is wrong. Fires: every Meteor dash by a scaled player — grazing kills along the two-tick lethal corridor flip.

**5. Push never calls `PlayerPhysics.Move(zero, dt)`.**
- C# `sim/Gameplay/ControlPlatform.cs:136-137` `body.externalVelocity *= (Fix)0.8; physics.Move(Vec2.zero, simDeltaTime);`
- Rust `mod.rs:8391-8458` (`push_tick`) does the 0.8 damping at `:8392` then goes straight to the ground lookup; no `move_only` call anywhere in the function.
- Drops per-tick `groundedSpeed *= PlatformSlipperyness01`, `moveAlongSurface` (which advances `LocalPlatformPos` using external velocity along the surface), the `UnGround()` on a destroyed ground, and `DropPlatformTest()`.
- Fires: every tick a Push player carries external velocity — Push + Gust / Meteor / any explosion shockwave / slime-vs-slime shove all fail to slide the pusher.

**6. Push entry zeroes `groundedSpeed`; the source preserves it.**
- C# `sim/Gameplay/PlayerPhysics.cs:538` `groundedSpeed = playerInfo.groundedSpeed;` (via `SyncPhysicsTo`, called at `ControlPlatform.cs:82`); `ControlPlatform.cs:85-86` zeroes only the two velocity channels; `:107` hands `groundedSpeed` back on exit.
- Rust `mod.rs:6404-6406` `slot.base.velocity = Vec2::ZERO; slot.base.external_velocity = Vec2::ZERO; slot.base.grounded_speed = Fix::ZERO;`
- Fires: every Push cast begun while running. A tap-and-release returns up to `max_speed*0.5` of grounded speed in the source and 0 in Rust.

**7. Mine is not frozen by Time Stop when its owner is protected.**
- C# `sim/Gameplay/Mine.cs:154` lets a protected owner past the early return, but each state re-tests the *global* stop without the exemption: `:171` `if (!GameTime.IsTimeStopped() && !body.IsBeingHeld())`, `:185` `if (GameTime.IsTimeStopped() || body.IsBeingHeld()) break;`, `:211` `if (!GameTime.IsTimeStopped())`. A protected owner's mine only hovers and may arm.
- Rust `mod.rs:23454` is the sole guard (`if player_time_scale < 0.01 && !time_stop_protected[owner] { continue; }`); scan (`:23474`), chase (`:23489-23516`) and anticipation countdown (`:23519-23537`) then run at full speed.
- Fires: whenever the mine owner is the Time Stop caster. `time_stop_mine_anticipation_ab.c` gives Mine and Time Stop to different players, so both backends take the early return — genuinely unmeasured.

**8. Missile form steers/stabilizes on the tick it lands; the source exits first.**
- C# `sim/Gameplay/SparkShot.cs:144-161` — `if (physics.VelocityBasedRaycasts(true, simDeltaTime)) { ... ability.ExitAbility(info); return; }`, returning *before* `missile.Stabilize()` at `:189` and the steering block.
- Rust `mod.rs:16511` `self.steer_active_missile(index, action, DT, data);` runs before the airborne branch whose landing exit is at `:16547-16555`.
- One extra rotation step plus velocity realignment on the landing tick, and `instability_ticks` resets. Fires: every Missile cast that ends with the form landing while the spark is alive — most grounded casts.

**9. Rock's bounce impulse is not gated on `ropeBody == null`.**
- C# `sim/Gameplay/BounceBall.cs:310-322` — the whole Player/Projectile impulse block sits inside `if (ropeBody == null)`.
- Rust applies it unconditionally at all five sites: `mod.rs:27010-27011` (player), `:27029-27030` (clone), `:26971-26974` (Rock/Rock), `:24602-24605` (arrow, 0.1 branch), `:16203-16204`.
- Fires: Rock + rope only (Web Swing / Grappling Hook), then on every contact.

**10. Poke ball cannot capture an abilityless clone.**
- C# `sim/Gameplay/PokeBall.cs:135` `collision.colliderPP.fixTrans.GetComponent<SlimeController>()` and `:149` `Object.FindObjectsOfType<SlimeController>()` — Duplicator clones are real `SlimeController`s.
- Rust `mod.rs:23722` and `:25765` both iterate `0..self.players.len()` only; `poke_balls_tick` never touches `self.abilityless_clones`, while the sibling `tacks_tick` explicitly does (`:23817`).
- Fires: Duplicator + Poke ball only, then on every pass. `duplicator_poke*_ab.c` tests duplicated *balls*, not clone victims.

---

## SUSPECTED divergences

**S1. Rock collider radius accumulates where the source resets.** Rust `mod.rs:27135-27136` `st.scale = Fix::ONE + amount; st.collider_radius *= st.scale;` — an accumulating product. In C#, `Ability.cs:135` `potentialCollider.Scale = player.Scale;` runs every `UpdateSim`, and the `DPhysicsCircle.Scale` setter (`DPhysicsCircle.cs:214-222`) divides the old scale out, so `BounceBall.cs:302-304`'s `collisionCircle.Scale += fix2` telescopes to `startRadius * (1 + a_n)`. Sub-review's numeric evaluation of the shipped curve gives a peak of 2.504 vs 2.150 and a permanent 2.060 vs 2.100 settle. **Confirmed sub-part:** the Rust is internally inconsistent — `mod.rs:24562` uses `data.rock.radius * player.rock.scale` for arrow-vs-Rock while every other site uses `rock.collider_radius`. Undetermined: whether `rock_ability_ab.c` compares collider radius at all; I did not trace `potentialCollider` binding for the Rock prefab.

**S2. Rock `prevPlayerVel` initialised post-0.6667.** C# `BounceBall.cs:128` assigns it before `:135` scales the grounded velocity; Rust `mod.rs:6480` `previous_player_velocity: body.velocity,` after `:6450-6454` applied 0.6667. Consumed at `:26836/26838`, so the first rope force sample after a grounded Rock entry is off by `0.3333 * v_entry`. Rope + Rock, grounded cast. (Sub-review verified; I did not re-read.)

**S3. Spike pooled-index aliasing.** `spike_attacks` entries are reused by index (`mod.rs:9479-9483`) and `SpawnedSpike` (`:603-623`) carries no owner, but the non-recast death paths (`:11189` black hole, `:22682-22686` platform destruction, `:20380` duplicate expiry) clear only `active`, never the owning player's slot reference. A later recast by the original owner could `remove_spike` a different player's live spike. C# is safe because Unity's overloaded `!= null` is false for a destroyed `SpikeAttack` (`Spike.cs:96-99`). Rare but high-impact. (Sub-review's finding; not re-verified by me.)

**S4. Spike grounded entry skips `SyncPhysicsTo`'s velocity reset.** C# `PlayerPhysics.cs:536-537` zeroes `selfImposedVelocity` and reloads `externalVelocity`; Rust `mod.rs:6633-6646` touches neither. Undetermined: whether a grounded slime's `velocity` is ever non-zero at Spike entry (would need a shockwave/shove landing the same tick).

**S5. Drill moving-platform contribution uses composite vs transform position.** C# `StickyRoundedRectangle.cs:207-213` takes `pos - groundBody.position` where `position` is the rounded rect's own FixTransform (velocity/angular do resolve to the composite). Rust `mod.rs:11598-11604` subtracts `composite_position`, which diverges from `surface.center` after Spike/Engine wall corrections (`:27562`, `:27636`). Undetermined: whether the C# FixTransform receives the same correction delta in-pass.

**S6. Drill platform selection order.** C# `Drill.cs:309-327` takes `collisionBuffer[0]` in DetPhysics registration order and only overwrites `currentPlatform` when the hit carries a `StickyRoundedRectangle`; Rust `mod.rs:11505-11508` takes the first match in `self.platforms` and assigns unconditionally. Undetermined: whether the two orderings coincide. Only bites when a drill box straddles two platforms.

**S7. Arrow contact treated as an explosive igniter.** C# `GrenadeExplode.cs:70-77` gates Projectile-layer detonation on `component2.IgnitesExplosives`, and `Assets/GameObject/Arrow.prefab:144` has `IgnitesExplosives: 0`. Rust `mod.rs:25801-25812` builds `hit_igniting_projectile` from any active registered arrow overlap with no such test. Mitigated because `arrows_contacts` (`:4636`) runs before `thrown_items_contacts` (`:4640`) and destroys the arrow. Undetermined: whether an arrow can be active, registered and overlapping at the later pass — e.g. one whose contact was consumed by the platform branch's `continue`.

**S8. No "grounded throw form lost its ground" exit.** C# `ThrowItem2.cs:298-302` `if (isGroundedThrow && !physics.IsGrounded()) { ExitAbility(); return; }`; Rust `throw_item_tick` (`mod.rs:16822-16997`) branches off the entry-time `throw_item.grounded` at `:16932` with no live re-test, as does `mine_throw_tick` at `:16666`. Related: an airborne Mine that lands inside the 5-tick `aEnter` window should re-read `physics.IsGrounded()` in `Fire()` (`ThrowItem.cs:404-413`, `:243-252`) and launch at `minThrowStrength` from the ground normal; Rust `mod.rs:16722` reads the stale entry flag. Undetermined: which in-game events null the attached ground during a ~15-tick form.

---

## Documented simplifications I do not consider safe

- **`native/FULL_COMPATIBILITY.md:2873-2912` ("Executable Push platform control")** is under-documented rather than wrong. It describes entry as "clears both velocity channels" and exit as preserving the grounded channel, but discloses neither the omitted per-tick `physics.Move(Vec2.zero, simDeltaTime)` (finding 5) nor the zeroed entry `groundedSpeed` (finding 6). Line 2911's list of unmeasured branches does not cover them. The 420-tick `push_broad_ab.sh` gate cannot catch either: a Push entered from standstill has `groundedSpeed == 0`, which makes `Move(zero)` a no-op.
- **`FULL_COMPATIBILITY.md:1261`** ("A final source audit covers every branch in `ThrowItem2`, `Grenade` and `GrenadeExplode` ... scale propagation") is inaccurate: the Fire-path scale factors (finding 1) are absent, and none of the 16 `grenade_broad_ab.sh` scenarios includes Growth or Shrink.
- **`FULL_COMPATIBILITY.md:2707-2710`** is literally true (`scale_missile_ab.c` does cover both ray signs on the live box) but is relied on in the catalog `verification` text as evidence Missile's scale composition is covered; the caster-scaled *spawn* is untested and wrong.

Checks that came back clean and are worth not re-doing: `native/data/rollout.json` extraction is faithful (I spot-checked Meteor's `extraJumpTeleportMultiplier` 2.0, `FixedUpdatesInDash` 2, `vibrationStr`, `groundForce` against `Assets/GameObject/Meteor.prefab`); `smearScaling`/`spinSpeed` are presentation-only or unused; no ability in this set uses object pooling, so "pool exhaustion" is a non-issue; `MeteorSmash.cs:267` really does call `Vec2.NormalizedSafe(Updater.RandomInsideUnitCircle())`, so the documented normalization is correct; Rock's missing `IsCancellable` early-exit is correct because `Rock.prefab:239` has `IsCancellable: 0`; Push's normalized force arm matches because `DPhysicsCircle.n` itself normalizes.

---

## Coverage

- **Grenade (3):** reviewed fully.
- **Smoke (13):** partially — skipped the per-cast 40-cloud ring eviction ordering, the `cloud_enter_frame_ticks` partition index, `duplicate_smoke_cloud`.
- **Mine (24):** partially — skipped the `igniters` layer-mask decomposition, `Mine.cs:229`'s LethalTerrain/ChainLightning exclusion, `register_pending_mines`, mine duplication.
- **Missile (10):** partially — skipped `Explosion.cs` shockwave constants and the Missile/Missile box-pair contact.
- **Spike (11):** partially — skipped `SpikeAttack.OnCollide` knockback translation, `RepositionOnParent`, the Macho-transfer path, the Duplicator clone path.
- **Meteor (20):** partially — skipped `meteor_waves_*` vs `Shockwave.cs`, `DropAllAttachedPlayers`/`ShakablePlatform` internals, `raycastMask`/water constants.
- **Throw (21):** **largely unreviewed** — C# read end to end, but the Rust form was never branch-compared. Skipped entirely: `Fire()` velocity/aim clamp, `heldPos`/`firePos`/`quickThrowFirePos`, `pickUpNearbyAttachables`/`attachPickupedAttachableToBoulder`, `shrinkPlatform`, the Boulder hold/drop state machine. This is the biggest remaining gap and warrants a second pass.
- **Push (22):** reviewed fully.
- **Drill (26):** partially — skipped `drill_contacts` vs `HitboxCombo`, hitbox geometry vs prefab, rope interaction detail.
- **Poke ball (33):** reviewed fully.
- **Tack (35):** fully, except the wall-settle callback (`BananaPeel.cs:126`) — the code setting `settled`/`settle_position` was not located.
- **Mjolnir (37):** partially — skipped Duplicator/Magnet/Portal/Black-Hole shared thrown-item paths and the prefab constant cross-check.
- **Rock (9):** reviewed fully.

Findings 1-10 I re-read on both sides personally. S2, S3 and the Mine-landing half of S8 come from the sub-reviews and I did not independently re-read their line quotes — treat those three as needing one more confirmation pass before anyone acts on them.