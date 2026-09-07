using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using UnityEngine.LowLevel;

// Opt-in wall-clock profile of the RL harness (BOPL_RL_PROFILE=<directory>).
// Time is attributed to one flat bucket at a time: bridge buckets are switched
// explicitly from RlBridge, tick phases come from the ReferenceScheduler.Capture
// hook and per-component phases from ReferenceDeathEffects.ComponentPhase, so
// gameplay code is untouched and nothing the simulation can observe changes.
// A summary is written when the trainer closes the game.
internal static class RlProfile
{
    private static readonly string output = Environment.GetEnvironmentVariable("BOPL_RL_PROFILE");
    internal static readonly bool Enabled = !string.IsNullOrEmpty(output);
    private static readonly Dictionary<string, long[]> buckets = new Dictionary<string, long[]>();
    private static readonly Dictionary<string, long[]> tickBuckets = new Dictionary<string, long[]>();
    private static readonly Dictionary<Type, long[]> components = new Dictionary<Type, long[]>();
    private static readonly string[] phases = { "init", "update", "late", "destroy_updatable" };
    private static long[] current; private static int index; private static long mark;
    private static long ticks, countedTicks, dumped;

    internal static void Install()
    {
        if (!Enabled) return;
        if (ReferenceScheduler.Capture == null) ReferenceScheduler.Capture = (kind, target) => Enter(tickBuckets, kind);
        if (ReferenceDeathEffects.ComponentPhase == null) ReferenceDeathEffects.ComponentPhase = Component;
        InstrumentLoop();
        mark = Stopwatch.GetTimestamp();
    }

    // A marker system before every player-loop subsystem attributes the time
    // between bridge pumps to Unity's own phases (physics, animation, particles...).
    private static void InstrumentLoop()
    {
        var loop = PlayerLoop.GetCurrentPlayerLoop();
        var top = new List<PlayerLoopSystem>();
        foreach (var system in loop.subSystemList) {
            string name = system.type == null ? "?" : system.type.Name;
            var copy = system;
            if (system.subSystemList != null) {
                var inner = new List<PlayerLoopSystem>();
                foreach (var child in system.subSystemList) { inner.Add(Marker(name + "." + (child.type == null ? "?" : child.type.Name))); inner.Add(child); }
                copy.subSystemList = inner.ToArray();
            }
            top.Add(Marker(name)); top.Add(copy);
        }
        top.Add(Marker("end"));
        loop.subSystemList = top.ToArray();
        PlayerLoop.SetPlayerLoop(loop);
    }

    private static PlayerLoopSystem Marker(string name)
    {
        string bucket = "loop:" + name;
        return new PlayerLoopSystem { type = typeof(RlProfile), updateDelegate = () => Enter(buckets, bucket) };
    }

    private static void Switch(long[] slot, int at)
    {
        long now = Stopwatch.GetTimestamp();
        if (current != null) { current[index] += now - mark; current[index + 1]++; }
        current = slot; index = at; mark = now;
    }

    // Names are string literals (no allocation); the dictionary lookup is the only cost.
    internal static void Enter(string name) { if (Enabled) Enter(buckets, name); }

    private static void Enter(Dictionary<string, long[]> table, string name)
    {
        if (!table.TryGetValue(name, out var slot)) table[name] = slot = new long[2];
        Switch(slot, 0);
    }

    private static void Component(string phase, MonoUpdatable target)
    {
        int at = Array.IndexOf(phases, phase);
        if (at < 0 || target == null) { Enter("component:" + phase); return; }
        Type type = target.GetType();
        if (!components.TryGetValue(type, out var slot)) components[type] = slot = new long[phases.Length * 2];
        Switch(slot, at * 2);
    }

    internal static void Ticked(bool counted) { if (!Enabled) return; ticks++; if (counted) countedTicks++; }

    private static IEnumerable<KeyValuePair<string, long[]>> Rows()
    {
        foreach (var entry in buckets) yield return entry;
        foreach (var entry in tickBuckets) yield return new KeyValuePair<string, long[]>("tick:" + entry.Key, entry.Value);
    }

    internal static void Dump()
    {
        if (!Enabled || dumped++ > 0) return;
        Enter("dump");
        double ms = 1000.0 / Stopwatch.Frequency;
        var text = new StringBuilder();
        text.Append("{\"ticks\":").Append(ticks).Append(",\"counted_ticks\":").Append(countedTicks).Append(",\"buckets\":{");
        bool first = true;
        foreach (var entry in Rows()) {
            text.Append(first ? "" : ",").Append('"').Append(entry.Key).Append("\":[").Append((entry.Value[0] * ms).ToString("F3")).Append(',').Append(entry.Value[1]).Append(']');
            first = false;
        }
        text.Append("},\"components\":{");
        first = true;
        foreach (var entry in components) {
            text.Append(first ? "" : ",").Append('"').Append(entry.Key.Name).Append("\":{");
            for (int phase = 0; phase < phases.Length; phase++)
                text.Append(phase == 0 ? "" : ",").Append('"').Append(phases[phase]).Append("\":[").Append((entry.Value[phase * 2] * ms).ToString("F3")).Append(',').Append(entry.Value[phase * 2 + 1]).Append(']');
            text.Append('}');
            first = false;
        }
        text.Append("}}\n");
        Directory.CreateDirectory(output);
        File.WriteAllText(Path.Combine(output, "profile-" + Process.GetCurrentProcess().Id + ".json"), text.ToString());
    }
}
