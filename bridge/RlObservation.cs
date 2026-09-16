using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using BoplFixedMath;
using UnityEngine;

// Read-only RL features. Raw collider state is retained even during the source's
// deferred removal window. Policy selection/normalization happens in C.
internal static class RlObservation
{
    private static readonly Dictionary<ulong, uint> ids = new Dictionary<ulong, uint>();
    private static readonly Dictionary<Sprite, uint> icons = new Dictionary<Sprite, uint>();
    private static readonly List<SlimeController> slimes = new List<SlimeController>();
    private static readonly FieldInfo inAbility = typeof(SlimeController).GetField("isInAbility", BindingFlags.Instance | BindingFlags.NonPublic);
    private static readonly FieldInfo beamCount = typeof(DetPhysics).GetField("totalBeamsLastFrame", BindingFlags.Instance | BindingFlags.NonPublic);
    // Bow charge for the policy (ability_flags bits 4..7): loadingFrame 0..4 while the bow is in use, 0x80 once fired.
    private static readonly FieldInfo bowLoadingFrame = typeof(BowTransform).GetField("loadingFrame", BindingFlags.Instance | BindingFlags.NonPublic);
    private static readonly FieldInfo bowHasFired = typeof(BowTransform).GetField("hasFired", BindingFlags.Instance | BindingFlags.NonPublic);
    private static readonly FieldInfo pickupSwapTime = typeof(DynamicAbilityPickup).GetField("timeSinceAbilitySwap", BindingFlags.Instance | BindingFlags.NonPublic);
    private static readonly FieldInfo smokeDetonationTime = typeof(FlammableSmoke).GetField("timeSinceDetonated", BindingFlags.Instance | BindingFlags.NonPublic);
    private static readonly FieldInfo smokeDetonationDelay = typeof(FlammableSmoke).GetField("detonationDelay", BindingFlags.Instance | BindingFlags.NonPublic);
    private static readonly FieldInfo timeStopElapsed = typeof(TimeStop).GetField("secondsElapsed", BindingFlags.Instance | BindingFlags.NonPublic);
    private static readonly FieldInfo timeStopDuration = typeof(TimeStop).GetField("duration", BindingFlags.Instance | BindingFlags.NonPublic);
    private static readonly FieldInfo timeStopExitAlpha = typeof(TimeStop).GetField("ExitAlpha", BindingFlags.Instance | BindingFlags.NonPublic);
    private static uint nextId;

    internal static void Reset()
    {
        ids.Clear(); nextId = 0; icons.Clear();
        var catalog = SteamManager.instance.abilityIcons.sprites;
        for (int i = 0; i < catalog.Count; i++)
            if (catalog[i].sprite != null && !icons.ContainsKey(catalog[i].sprite)) icons.Add(catalog[i].sprite, (uint)i);
    }

    private static uint Id(ulong key)
    {
        if (!ids.TryGetValue(key, out uint value)) { value = ++nextId; ids.Add(key, value); }
        return value;
    }

    // simulationTime: header.simulation_time raw value; RlBridge passes the game's
    // SimTimePassed (zeroed per reset), RlAgent passes ticks-since-level-load x
    // FixedTimeStep, which is the same quantity in the stock local game except
    // that its Update loop also advances SimTimePassed on hitstop frames.
    internal static void Write(BinaryWriter output, uint version, int episodeTicks, uint flags, int warmup, long? simulationTime = null)
    {
        var players = PlayerHandler.Get().PlayerList();
        var physics = DetPhysics.Get();
        slimes.Clear();
        foreach (var component in Updater.updatables)
            if (component is SlimeController slime && slime != null && !slime.IsDestroyed && slime.body != null && slime.body.fixtrans != null)
                slimes.Add(slime);
        int beams = (int)beamCount.GetValue(physics);
        int entities = physics.circles.Length + physics.boxes.Length + physics.roundedRects.Length + beams;
        foreach (var rope in physics.ropes.ropeBodies)
            if (rope.enabled) entities += Math.Max(0, rope.segmentCount - 1);
        output.Write(version); output.Write(Updater.SimulationTicks); output.Write(episodeTicks); output.Write(flags);
        output.Write(players.Count); output.Write(entities); output.Write(warmup); output.Write(slimes.Count);
        output.Write(simulationTime ?? Updater.SimTimePassed.m_rawValue); output.Write(GameTime.PlayerTimeScale.m_rawValue);
        output.Write(Updater.SimTimeSinceLevelLoaded.m_rawValue);
        output.Write(GameSessionHandler.SuddenDeathInProgress ? 1u : 0u); output.Write(0u);
        foreach (var player in players) {
            output.Write(player.Id); output.Write(player.Team);
            output.Write((player.IsAlive ? 1u : 0u) | (player.WonThisRound ? 2u : 0u) | (player.isInvisible ? 4u : 0u)
                | (player.isProtectedFromTimeStop ? 8u : 0u) | (player.InSparkShot ? 16u : 0u)
                | (player.InPlatformTransform ? 32u : 0u) | (player.InMachoThrow ? 64u : 0u));
            output.Write(player.playersAndClonesStillAlive); output.Write(player.StocksRemaining);
            output.Write(player.Kills); output.Write(player.Deaths); output.Write((int)player.CauseOfDeath);
            output.Write(player.Position.x.m_rawValue); output.Write(player.Position.y.m_rawValue); output.Write(player.Scale.m_rawValue);
            output.Write(player.timeSpentInvisible.m_rawValue); output.Write(player.timeSinceEnteredAbility.m_rawValue);
        }
        for (int i = 0; i < physics.circles.Length; i++) {
            var shape = physics.circles.colliders[i];
            Collider(output, 0, physics.circles.physicsParents[i], physics.circles.physicsBodies[i], shape.layer,
                shape.center, Vec2.zero, Vec2.zero, shape.radius);
        }
        for (int i = 0; i < physics.boxes.Length; i++) {
            var shape = physics.boxes.colliders[i];
            Collider(output, 1, physics.boxes.physicsParents[i], physics.boxes.physicsBodies[i], shape.layer,
                shape.center, shape.right, shape.up, Fix.Zero);
        }
        for (int i = 0; i < physics.roundedRects.Length; i++) {
            var shape = physics.roundedRects.colliders[i];
            Collider(output, 2, physics.roundedRects.physicsParents[i], physics.roundedRects.physicsBodies[i], shape.layer,
                shape.box.center, shape.box.right, shape.box.up, shape.radius);
        }
        foreach (var rope in physics.ropes.ropeBodies) {
            if (!rope.enabled) continue;
            for (int j = 0; j + 1 < rope.segmentCount; j++) {
                uint id = Id((1UL << 63) | ((ulong)rope.number << 6) | (uint)j);
                Entity(output, null, id, 3, 0, rope.ownerId, 3u | (rope.hasBeenDettached ? 32u : 0u), j,
                    rope.segment[j], Vec2.zero, rope.segment[j + 1] - rope.segment[j], Vec2.zero,
                    physics.ropeSegmentRadius, Fix.Zero, Fix.Zero, Fix.Zero);
            }
        }
        for (int i = 0; i < beams; i++) {
            var beam = physics.beamBodies[i];
            uint id = Id((1UL << 62) | ((ulong)(uint)beam.id << 16) | (uint)i);
            Entity(output, null, id, 4, 0, beam.ownerId, 3, (int)physics.beamRoles[i],
                beam.position, Vec2.zero, beam.direction * beam.distance, Vec2.zero,
                physics.beamWidth * beam.scale, Fix.Zero, Fix.Zero, Fix.Zero);
        }
        foreach (var slime in slimes) Slime(output, slime);
        foreach (var player in players) RlTelemetry.Write(output, player.Id);
    }

    private static void Collider(BinaryWriter output, uint kind, PhysicsParent parent, PhysicsBody body,
        int layer, Vec2 position, Vec2 right, Vec2 up, Fix radius)
    {
        uint id = Id((uint)parent.instanceId);
        var transform = parent.fixTrans;
        var owner = transform == null ? null : transform.GetComponent<IPlayerIdHolder>();
        var collider = parent.monobehaviourCollider;
        // Interface null checks do not apply Unity's destroyed-object semantics.
        var component = collider as MonoBehaviour;
        bool nativeAlive = component != null;
        uint flags = (transform != null && transform.gameObject.activeInHierarchy ? 1u : 0u)
            | (nativeAlive && collider.enabled ? 2u : 0u) | (!nativeAlive || collider.IsDestroyed ? 4u : 0u)
            | (body.IsTrigger() ? 8u : 0u) | (body.IsComposite() ? 16u : 0u);
        Entity(output, transform, id, kind, layer, owner == null ? 0 : owner.GetPlayerId(), flags, parent.hierarchyNumber,
            position, body.velocity, right, up, radius, body.orientation, body.angularVelocity, body.inverseMass);
    }

    private static uint AbilityId(AbilityMonoBehaviour ability)
    {
        if (ability == null || ability.IsDestroyed) return 0;
        foreach (var slime in slimes) {
            if (slime == null || slime.IsDestroyed || slime.abilities == null) continue;
            int count = Math.Min(3, slime.abilities.Count);
            for (int i = 0; i < count; i++)
                if (slime.abilities[i] == ability) return SlimeAbilityId(slime, i);
        }
        return 0;
    }

    private static uint SlimeAbilityId(SlimeController slime, int slot)
    {
        if (slime.AbilityReadyIndicators == null || slot >= slime.AbilityReadyIndicators.Length
            || slime.AbilityReadyIndicators[slot] == null) return 0;
        var icon = slime.AbilityReadyIndicators[slot].GetPrimarySprite();
        return icon != null && icons.TryGetValue(icon, out uint found) ? found : 0;
    }

    private static uint RemainingTicks(Fix seconds)
    {
        if (seconds <= Fix.Zero) return 0;
        long raw = seconds.m_rawValue, step = GameTime.FixedTimeStep.m_rawValue;
        if (step <= 0) return 0;
        ulong ticks = (ulong)((raw - 1) / step + 1);
        return ticks > uint.MaxValue ? uint.MaxValue : (uint)ticks;
    }

    private static uint SourceAbility(FixTransform transform, int owner, DynamicAbilityPickup pickup, Grenade grenade,
        FlammableSmoke smoke, SimpleSparkNode spark)
    {
        if (pickup != null) {
            var renderer = pickup.GetComponent<SpriteRenderer>();
            if (renderer != null && renderer.sprite != null && icons.TryGetValue(renderer.sprite, out uint granted)) return granted;
        }
        var direct = transform.GetComponent<AbilityMonoBehaviour>();
        if (direct == null) direct = transform.GetComponentInParent<AbilityMonoBehaviour>();
        uint id = AbilityId(direct);
        if (id != 0 && id != uint.MaxValue) return id;
        if (grenade != null && grenade.thrower != null && !grenade.thrower.IsDestroyed) {
            id = AbilityId(grenade.thrower.GetComponent<AbilityMonoBehaviour>());
            if (id != 0 && id != uint.MaxValue) return id;
        }
        // Smoke objects retain an exact reference in their casting ThrowItem2's
        // effect buffer. Spark nodes can be tied to an owner's live spark ability.
        foreach (var slime in slimes) {
            if (slime == null || slime.IsDestroyed || slime.abilities == null || slime.GetPlayerId() != owner) continue;
            int count = Math.Min(3, slime.abilities.Count);
            for (int slot = 0; slot < count; slot++) {
                var ability = slime.abilities[slot];
                if (ability == null || ability.IsDestroyed) continue;
                if (smoke != null) {
                    var thrower = ability.GetComponent<ThrowItem2>();
                    var effects = thrower == null ? null : thrower.effectBuffer;
                    if (effects != null)
                        for (int i = 0; i < effects.Length; i++)
                            if (effects[i] == smoke) return SlimeAbilityId(slime, slot);
                }
                if (spark != null && (ability.GetComponent<PlaceSparkNode>() != null || ability.GetComponent<SparkShot>() != null))
                    return SlimeAbilityId(slime, slot);
            }
        }
        return 0;
    }

    private static void Entity(BinaryWriter output, FixTransform transform, uint id, uint kind, int layer, int owner, uint flags, int hierarchy,
        Vec2 position, Vec2 velocity, Vec2 right, Vec2 up, Fix radius, Fix rotation, Fix angularVelocity, Fix inverseMass)
    {
        uint source = 0, remaining = 0;
        Fix clockRate = Fix.Zero;
        if (transform != null) {
            var pickup = transform.GetComponent<DynamicAbilityPickup>();
            var projectile = transform.GetComponent<Projectile>();
            var spark = transform.GetComponent<SimpleSparkNode>();
            var smoke = transform.GetComponent<FlammableSmoke>();
            var grenade = transform.GetComponent<Grenade>();
            var explosion = transform.GetComponent<Explosion>();
            var hitbox = transform.GetComponent<Hitbox>();
            bool explosive = grenade != null || explosion != null || transform.GetComponent<IGrenadeDetonation>() != null;
            if (pickup != null) flags |= 1u << 7;
            if (projectile != null) flags |= 1u << 8;
            if (spark != null) flags |= 1u << 9;
            if (smoke != null) flags |= 1u << 10;
            if (explosive) flags |= 1u << 11;
            if (hitbox != null) flags |= 1u << 12;
            var contact = transform.GetComponent<Ability>();
            if (contact == null) contact = transform.GetComponentInParent<Ability>();
            if (contact != null && !contact.IsDestroyed) {
                flags |= 1u << 14;
                if (contact.KillPlayersOnContact) flags |= 1u << 13;
            }
            source = SourceAbility(transform, owner, pickup, grenade, smoke, spark);
            if (grenade != null && grenade.selfDestructDelay >= Fix.Zero) {
                remaining = RemainingTicks(grenade.selfDestructDelay); clockRate = GameTime.PlayerTimeScale; flags |= 1u << 6;
            } else if (pickup != null) {
                remaining = RemainingTicks(pickup.SwapAbilityPeriod - (Fix)pickupSwapTime.GetValue(pickup));
                clockRate = GameTime.IsTimeStopped() ? Fix.Zero : Fix.One; flags |= 1u << 6;
            } else if (smoke != null) {
                Fix elapsed = (Fix)smokeDetonationTime.GetValue(smoke);
                if (elapsed >= Fix.Zero) {
                    remaining = RemainingTicks((Fix)smokeDetonationDelay.GetValue(smoke) - elapsed);
                    clockRate = GameTime.PlayerTimeScale; flags |= 1u << 6;
                }
            } else {
                var timeStop = transform.GetComponent<TimeStop>();
                if (timeStop != null) {
                    float elapsed = (float)timeStopElapsed.GetValue(timeStop);
                    float duration = (float)timeStopDuration.GetValue(timeStop);
                    var exit = (AnimationCurve)timeStopExitAlpha.GetValue(timeStop);
                    float end = duration + (exit == null || exit.length == 0 ? 0f : exit.keys[exit.length - 1].time);
                    remaining = RemainingTicks((Fix)Math.Max(0f, end - elapsed)); clockRate = Fix.One; flags |= 1u << 6;
                }
            }
        }
        output.Write(id); output.Write(kind); output.Write(layer); output.Write(owner);
        output.Write(flags); output.Write(hierarchy); output.Write(source); output.Write(remaining);
        output.Write(position.x.m_rawValue); output.Write(position.y.m_rawValue); output.Write(velocity.x.m_rawValue); output.Write(velocity.y.m_rawValue);
        output.Write(right.x.m_rawValue); output.Write(right.y.m_rawValue); output.Write(up.x.m_rawValue); output.Write(up.y.m_rawValue);
        output.Write(radius.m_rawValue); output.Write(rotation.m_rawValue); output.Write(angularVelocity.m_rawValue); output.Write(inverseMass.m_rawValue);
        output.Write(clockRate.m_rawValue);
    }

    private static void Slime(BinaryWriter output, SlimeController slime)
    {
        output.Write(Id((uint)slime.gameObject.GetInstanceID())); output.Write(slime.GetPlayerId());
        output.Write((slime.gameObject.activeInHierarchy ? 1u : 0u) | (slime.HasSpawned ? 2u : 0u)
            | ((bool)inAbility.GetValue(slime) ? 4u : 0u) | (slime.suppressInput ? 8u : 0u) | (slime.body.simulated ? 16u : 0u)
            | (slime.abilities == null || slime.abilities.Count == 0 ? 32u : 0u)
            | ((uint)slime.NextAbiltityToChangeOnPickup & 3u) << 6);
        output.Write((int)slime.getCurrentState());
        var position = slime.GetPos(); var velocity = slime.body.Velocity;
        output.Write(position.x.m_rawValue); output.Write(position.y.m_rawValue); output.Write(velocity.x.m_rawValue); output.Write(velocity.y.m_rawValue);
        output.Write(slime.body.rotation.m_rawValue); output.Write(slime.body.fixtrans.Scale.m_rawValue);
        var timers = slime.GetAbilityCooldownTimers();
        for (int i = 0; i < 3; i++) {
            output.Write(SlimeAbilityId(slime, i));
        }
        for (int i = 0; i < 3; i++) {
            var ability = slime.abilities != null && i < slime.abilities.Count ? slime.abilities[i] : null;
            bool present = ability != null && !ability.IsDestroyed;
            bool timer = timers != null && i < timers.Length;
            Fix cooldown = present ? ability.GetCooldown() : Fix.Zero;
            bool active = present && ability.gameObject.activeInHierarchy;
            uint charge = 0;
            if (active) {
                var bow = ability.GetComponent<BowTransform>();
                if (bow != null) {
                    charge = ((uint)(int)bowLoadingFrame.GetValue(bow) & 7u) << 4;
                    if ((bool)bowHasFired.GetValue(bow)) charge |= 0x80u;
                }
            }
            output.Write((present ? 1u : 0u) | (present && timer && timers[i] >= cooldown ? 2u : 0u)
                | (present && timer && timers[i] > cooldown ? 4u : 0u) | (active ? 8u : 0u) | charge);
        }
        for (int i = 0; i < 3; i++) {
            var ability = slime.abilities != null && i < slime.abilities.Count ? slime.abilities[i] : null;
            output.Write(ability != null && !ability.IsDestroyed ? ability.GetCooldown().m_rawValue : 0L);
        }
        for (int i = 0; i < 3; i++) output.Write(timers != null && i < timers.Length ? timers[i].m_rawValue : 0L);
    }
}
