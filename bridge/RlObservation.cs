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
    private static Dictionary<ulong, uint> ids = new Dictionary<ulong, uint>();
    private static readonly Dictionary<Sprite, uint> icons = new Dictionary<Sprite, uint>();
    private static readonly List<SlimeController> slimes = new List<SlimeController>();
    private static readonly FieldInfo inAbility = typeof(SlimeController).GetField("isInAbility", BindingFlags.Instance | BindingFlags.NonPublic);
    private static readonly FieldInfo beamCount = typeof(DetPhysics).GetField("totalBeamsLastFrame", BindingFlags.Instance | BindingFlags.NonPublic);
    private static uint nextId;
    internal static Dictionary<ulong, uint> Ids { get => ids; set => ids = value; }
    internal static uint NextId { get => nextId; set => nextId = value; }

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

    internal static void Write(BinaryWriter output, uint version, int episodeTicks, uint flags, int warmup)
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
        output.Write(Updater.SimTimePassed.m_rawValue); output.Write(GameTime.PlayerTimeScale.m_rawValue);
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
                Entity(output, id, 3, 0, rope.ownerId, 3u | (rope.hasBeenDettached ? 32u : 0u), j,
                    rope.segment[j], Vec2.zero, rope.segment[j + 1] - rope.segment[j], Vec2.zero,
                    physics.ropeSegmentRadius, Fix.Zero, Fix.Zero, Fix.Zero);
            }
        }
        for (int i = 0; i < beams; i++) {
            var beam = physics.beamBodies[i];
            uint id = Id((1UL << 62) | ((ulong)(uint)beam.id << 16) | (uint)i);
            Entity(output, id, 4, 0, beam.ownerId, 3, (int)physics.beamRoles[i],
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
        Entity(output, id, kind, layer, owner == null ? 0 : owner.GetPlayerId(), flags, parent.hierarchyNumber,
            position, body.velocity, right, up, radius, body.orientation, body.angularVelocity, body.inverseMass);
    }

    private static void Entity(BinaryWriter output, uint id, uint kind, int layer, int owner, uint flags, int hierarchy,
        Vec2 position, Vec2 velocity, Vec2 right, Vec2 up, Fix radius, Fix rotation, Fix angularVelocity, Fix inverseMass)
    {
        output.Write(id); output.Write(kind); output.Write(layer); output.Write(owner);
        output.Write(flags); output.Write(hierarchy); output.Write(0); output.Write(0);
        output.Write(position.x.m_rawValue); output.Write(position.y.m_rawValue); output.Write(velocity.x.m_rawValue); output.Write(velocity.y.m_rawValue);
        output.Write(right.x.m_rawValue); output.Write(right.y.m_rawValue); output.Write(up.x.m_rawValue); output.Write(up.y.m_rawValue);
        output.Write(radius.m_rawValue); output.Write(rotation.m_rawValue); output.Write(angularVelocity.m_rawValue); output.Write(inverseMass.m_rawValue);
    }

    private static void Slime(BinaryWriter output, SlimeController slime)
    {
        output.Write(Id((uint)slime.gameObject.GetInstanceID())); output.Write(slime.GetPlayerId());
        output.Write((slime.gameObject.activeInHierarchy ? 1u : 0u) | (slime.HasSpawned ? 2u : 0u)
            | ((bool)inAbility.GetValue(slime) ? 4u : 0u) | (slime.suppressInput ? 8u : 0u) | (slime.body.simulated ? 16u : 0u));
        output.Write((int)slime.getCurrentState());
        var position = slime.GetPos(); var velocity = slime.body.Velocity;
        output.Write(position.x.m_rawValue); output.Write(position.y.m_rawValue); output.Write(velocity.x.m_rawValue); output.Write(velocity.y.m_rawValue);
        output.Write(slime.body.rotation.m_rawValue); output.Write(slime.body.fixtrans.Scale.m_rawValue);
        var timers = slime.GetAbilityCooldownTimers();
        for (int i = 0; i < 3; i++) {
            uint index = uint.MaxValue;
            if (slime.AbilityReadyIndicators != null && i < slime.AbilityReadyIndicators.Length && slime.AbilityReadyIndicators[i] != null) {
                var icon = slime.AbilityReadyIndicators[i].GetPrimarySprite();
                if (icon != null && icons.TryGetValue(icon, out uint found)) index = found;
            }
            output.Write(index);
        }
        for (int i = 0; i < 3; i++) {
            var ability = slime.abilities != null && i < slime.abilities.Count ? slime.abilities[i] : null;
            bool present = ability != null && !ability.IsDestroyed;
            bool timer = timers != null && i < timers.Length;
            Fix cooldown = present ? ability.GetCooldown() : Fix.Zero;
            output.Write((present ? 1u : 0u) | (present && timer && timers[i] >= cooldown ? 2u : 0u)
                | (present && timer && timers[i] > cooldown ? 4u : 0u) | (present && ability.gameObject.activeInHierarchy ? 8u : 0u));
        }
        for (int i = 0; i < 3; i++) {
            var ability = slime.abilities != null && i < slime.abilities.Count ? slime.abilities[i] : null;
            output.Write(ability != null && !ability.IsDestroyed ? ability.GetCooldown().m_rawValue : 0L);
        }
        for (int i = 0; i < 3; i++) output.Write(timers != null && i < timers.Length ? timers[i].m_rawValue : 0L);
    }
}
