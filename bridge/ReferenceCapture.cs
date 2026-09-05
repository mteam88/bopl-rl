using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Reflection;
using BoplFixedMath;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;

// Opt-in differential instrumentation. Never used by an ordinary game session.
public static class ReferenceCapture
{
    public const string Schema = "bopl-managed-state-v3";
    private static readonly string output = Environment.GetEnvironmentVariable("BOPL_TRACE");
    private static StreamWriter writer;
    private static int captured;
    private static JToken lastInput;
    private static JArray events = new JArray();
    private static JArray playerLifeStages = new JArray();
    private static JArray playerBodyStages = new JArray();
    internal static void PlayerBodyStage(JObject stage) => playerBodyStages.Add(stage);
    private static JArray playerCollisionStages = new JArray();
    internal static void PlayerCollisionStage(JObject stage) => playerCollisionStages.Add(stage);
    internal static void PlayerLifeStage(JObject stage) => playerLifeStages.Add(stage);
    private static readonly SortedDictionary<string, int> abilityEntries = new SortedDictionary<string, int>(StringComparer.Ordinal);
    private static readonly Dictionary<Type, FieldInfo[]> fields = new Dictionary<Type, FieldInfo[]>();
    private static readonly int limit = ReadPositive("BOPL_TRACE_TICKS", 600);
    internal static Action<int,int,AbilityMonoBehaviour> ActorAbilityCapture;
    public static bool Recording => !string.IsNullOrEmpty(output);
    public static bool Enabled => Recording || ActorAbilityCapture!=null;
    public static bool Completed { get; private set; }
    public static int Batch => ReadPositive("BOPL_TRACE_BATCH", 1);

    private static int ReadPositive(string name, int fallback)
    {
        string value = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrEmpty(value)) return fallback;
        if (!int.TryParse(value, out int result) || result <= 0)
            throw new ArgumentException(name + " must be positive");
        return result;
    }

    public static void BeforeReplayStep(InputPacketQuad input)
    {
        if (!Recording) return;
        EnsureWriter();
        lastInput = new Encoder().Value(input, "input");
        writer.WriteLine(new JObject { ["kind"] = "input", ["index"] = captured,
            ["value"] = lastInput }.ToString(Formatting.None));
    }

    public static void BeforeReplayStep(InputPacket[] input)
    {
        if (!Recording) return;
        EnsureWriter();
        lastInput = new Encoder().Value(input, "input");
        writer.WriteLine(new JObject { ["kind"] = "input", ["index"] = captured,
            ["value"] = lastInput }.ToString(Formatting.None));
    }

    // Keep the completed physics stage even if its subsequent gameplay callback
    // throws. This is separate evidence, never a successful full-tick state.
    public static void FailedTick(Exception failure)
    {
        if (!Recording) return;
        try
        {
            var stage = ReferencePhysicsCapture.LastStep;
            var snapshot = new JObject {
                ["schema"] = "bopl-replay-failure-v1", ["input_index"] = captured,
                ["tick"] = Updater.SimulationTicks, ["input"] = lastInput,
                ["exception"] = failure.GetType().FullName, ["message"] = failure.Message,
                ["stack"] = failure.StackTrace,
                ["physics_stage_complete"] = stage != null && stage["after"] != null && stage["callbacks"] != null,
                ["physicsStage"] = stage, ["playerLifeStages"] = playerLifeStages, ["playerCollisionStages"] = playerCollisionStages, ["playerBodyStages"] = playerBodyStages
            };
            File.WriteAllText(output + ".failure.json", snapshot.ToString(Formatting.None));
            Finish("simulation-exception");
        }
        catch (Exception captureFailure) { Debug.LogException(captureFailure); }
    }

    public static void AbilityEntered(int player, int slot, AbilityMonoBehaviour ability)
    {
        ActorAbilityCapture?.Invoke(player,slot,ability);
        if (!Recording) return;
        string name = ability.gameObject.name.Replace("(Clone)", "");
        abilityEntries.TryGetValue(name, out int count);
        abilityEntries[name] = count + 1;
        events.Add(new JObject { ["kind"] = "ability_entry", ["player"] = player, ["slot"] = slot,
            ["prefab"] = name, ["instance"] = Encoder.Identity(ability) });
    }

    public static void AfterTick()
    {
        if (!Recording) return;
        try
        {
            EnsureWriter();
            var encoder = new Encoder();
            var state = new JObject();
            state["tick"] = Updater.SimulationTicks;
            if (ReferencePhysicsCapture.Enabled) state["physicsStage"] = ReferencePhysicsCapture.Take();
            state["playerBodyStages"] = playerBodyStages;
            playerBodyStages = new JArray();
            state["playerCollisionStages"] = playerCollisionStages;
            playerCollisionStages = new JArray();
            state["playerLifeStages"] = playerLifeStages;
            playerLifeStages = new JArray();
            state["events"] = events;
            events = new JArray();
            state["scene"] = SceneManager.GetActiveScene().name;
            state["updater"] = encoder.Static(typeof(Updater), new[] {
                "random", "hitStopFrames", "SimTimePassed", "SimTimeSinceLevelLoaded",
                "SimulationTicks", "LoadedALevelThisUpdate", "gameHasStopped",
                "addedUpdatables", "prefabsAdded", "registeringAPrefab",
                "toBeAddedUpdatables", "toBeAddedPrefab", "destroyedObjects", "destroyedMonos"
            });
            state["time"] = encoder.Static(typeof(GameTime), null);
            state["players"] = encoder.Value(PlayerHandler.Get().PlayerList(), "players");
            var components = new JArray();
            foreach (MonoUpdatable component in Updater.updatables)
            {
                if (component == null) continue;
                components.Add(new JObject { ["id"] = Encoder.Identity(component),
                    ["active"] = component.gameObject.activeInHierarchy,
                    ["enabled"] = component.enabled,
                    ["state"] = encoder.ObjectFields(component, "updatable/" + Encoder.Identity(component)) });
            }
            state["updatables"] = components;
            var transforms = new JArray();
            foreach (FixTransform transform in UnityEngine.Object.FindObjectsOfType<FixTransform>(true)
                .OrderBy(Encoder.Identity, StringComparer.Ordinal))
                transforms.Add(new JObject { ["id"] = Encoder.Identity(transform),
                    ["state"] = encoder.ObjectFields(transform, "transform/" + Encoder.Identity(transform)) });
            state["transforms"] = transforms;
            state["facing"] = new JArray(UnityEngine.Object.FindObjectsOfType<FlippingTransform>(true)
                .OrderBy(Encoder.Identity, StringComparer.Ordinal).Select(component => new JObject {
                    ["id"] = Encoder.Identity(component),
                    ["state"] = encoder.ObjectFields(component, "facing/" + Encoder.Identity(component)) }));
            var physics = DetPhysics.Get();
            state["circles"] = encoder.Bodies(physics.circles, "circles");
            state["boxes"] = encoder.Bodies(physics.boxes, "boxes");
            state["roundedRects"] = encoder.Bodies(physics.roundedRects, "roundedRects");
            state["session"] = encoder.Static(typeof(GameSessionHandler), new[] { "selfRef", "GameIsPaused", "SuddenDeathTimeStamp", "<SuddenDeathInProgress>k__BackingField" });
            state["sessionInstance"] = encoder.ObjectFields(typeof(GameSessionHandler).GetField("selfRef", BindingFlags.NonPublic | BindingFlags.Static).GetValue(null), "session");
            writer.WriteLine(new JObject { ["kind"] = "state", ["index"] = captured,
                ["value"] = state }.ToString(Formatting.None));
            captured++;
            if (captured >= limit) Finish("tick-limit");
        }
        catch (Exception error)
        {
            Debug.LogException(error);
            writer?.Dispose();
            writer = null;
            Application.Quit(2);
            throw;
        }
    }

    private static void EnsureWriter()
    {
        if (writer != null) return;
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output)));
        Stream stream = File.Create(output);
        if (output.EndsWith(".gz", StringComparison.Ordinal))
            stream = new GZipStream(stream, System.IO.Compression.CompressionLevel.Fastest);
        writer = new StreamWriter(stream);
        writer.WriteLine(new JObject { ["kind"] = "header", ["schema"] = Schema,
            ["reference"] = "67484d097087562123f860e1ed209bcb56e99d2e", ["unity"] = Application.unityVersion,
            ["batch"] = Batch, ["limit"] = limit,
            ["replayFormat"] = ReplaySession.Playing == null ? "legacy" : "BOPLREP2",
            ["setup"] = ReplaySession.Playing == null ? null : JObject.FromObject(new { ReplaySession.Playing.sourceHash, ReplaySession.Playing.catalogHash, ReplaySession.Playing.seed, ReplaySession.Playing.level, ReplaySession.Playing.gameMode, ReplaySession.Playing.stocks, ReplaySession.Playing.players }),
            ["scope"] = "managed updatables, active and inactive fixed transforms, facing, players, live rigid bodies, session; excludes native rendering internals and unlisted physics globals"
        }.ToString(Formatting.None));
    }

    public static void Finish(string reason)
    {
        EnsureWriter();
        writer.WriteLine(new JObject { ["kind"] = "end", ["ticks"] = captured,
            ["reason"] = reason, ["abilityEntries"] = JObject.FromObject(abilityEntries) }.ToString(Formatting.None));
        writer.Dispose();
        writer = null;
        Completed = true;
        Debug.Log("REFERENCE_CAPTURE_COMPLETE ticks=" + captured);
        Application.Quit(0);
    }

    public sealed class Encoder
    {
        private readonly Dictionary<object, string> seen = new Dictionary<object, string>(new ReferenceComparer());

        public JObject Bodies<T>(PhysicsBodyList<T> list, string path)
        {
            var result = new JArray();
            for (int i = 0; i < list.Length; i++)
                result.Add(new JObject { ["collider"] = Value(list.colliders[i], path + "/" + i + "/collider"),
                    ["body"] = Value(list.physicsBodies[i], path + "/" + i + "/body"),
                    ["parent"] = Identity(list.physicsParents[i].fixTrans) });
            return new JObject { ["length"] = list.Length, ["items"] = result };
        }

        public JObject Static(Type type, string[] names)
        {
            var result = new JObject();
            foreach (FieldInfo field in type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
                .Where(f => names == null || names.Contains(f.Name)).OrderBy(f => f.Name, StringComparer.Ordinal))
                result[field.Name] = Value(field.GetValue(null), type.Name + "/" + field.Name);
            return result;
        }

        public JObject ObjectFields(object value, string path)
        {
            if (value == null) return null;
            var result = new JObject { ["$type"] = value.GetType().FullName };
            foreach (FieldInfo field in Fields(value.GetType()))
            {
                if (field.Name == "instanceId" || field.Name == "instanceIdToColliderIndex") continue;
                if (field.DeclaringType == typeof(CompositeBody) && (field.Name == "circleIds" || field.Name == "boxIds" || field.Name == "roundedRectIds"))
                    result[field.DeclaringType.Name + "." + field.Name] = new JArray(((IEnumerable<int>)field.GetValue(value)).Select(BodyIdentity));
                else result[field.DeclaringType.Name + "." + field.Name] = Value(field.GetValue(value), path + "/" + field.Name);
            }
            return result;
        }

        public JToken Value(object value, string path)
        {
            if (value == null) return JValue.CreateNull();
            Type type = value.GetType();
            if (value is Fix fix) return new JObject { ["$fix"] = fix.m_rawValue };
            if (value is float f) return new JObject { ["$f32"] = BitConverter.ToInt32(BitConverter.GetBytes(f), 0) };
            if (value is double d) return new JObject { ["$f64"] = BitConverter.DoubleToInt64Bits(d) };
            if (value is string || value is bool || type.IsPrimitive || value is decimal)
                return new JValue(value);
            if (type.IsEnum) return new JObject { ["$enum"] = type.FullName, ["value"] = Convert.ToInt64(value) };
            if (value is UnityEngine.Object unity)
                return unity == null ? JValue.CreateNull() : new JObject { ["$unity"] = Identity(unity) };
            if (value is AnimationCurve curve)
                return new JObject { ["$curve"] = Value(curve.keys, path + "/keys"),
                    ["preWrap"] = (int)curve.preWrapMode, ["postWrap"] = (int)curve.postWrapMode };
            if (value is Delegate callback)
                return new JArray(callback.GetInvocationList().Select(c => new JObject {
                    ["method"] = c.Method.DeclaringType.FullName + "." + c.Method.Name,
                    ["target"] = Value(c.Target, path + "/target") }));
            if (!type.IsValueType)
            {
                if (seen.TryGetValue(value, out string previous)) return new JObject { ["$ref"] = previous };
                seen.Add(value, path);
            }
            if (value is IDictionary dictionary)
            {
                var entries = new List<JObject>();
                foreach (DictionaryEntry entry in dictionary)
                    entries.Add(new JObject { ["key"] = Value(entry.Key, path + "/key"),
                        ["value"] = Value(entry.Value, path + "/" + Convert.ToString(entry.Key, CultureInfo.InvariantCulture)) });
                return new JArray(entries.OrderBy(e => e["key"].ToString(Formatting.None), StringComparer.Ordinal));
            }
            if (value is IEnumerable enumerable)
            {
                var result = new JArray();
                foreach (object item in enumerable) result.Add(Value(item, path + "/" + result.Count));
                return result;
            }
            if (type.Assembly == typeof(Updater).Assembly || type.Namespace == "Unity.Mathematics" || type.IsValueType)
                return ObjectFields(value, path);
            // Explicit diagnostics: consumers must not interpret this as complete state.
            return new JObject { ["$unsupported"] = type.FullName };
        }

        private static string BodyIdentity(int id)
        {
            var physics = DetPhysics.Get();
            string Find<T>(PhysicsBodyList<T> list)
            {
                for (int i = 0; i < list.Length; i++)
                    if (list.physicsParents[i].instanceId == id) return Identity(list.physicsParents[i].fixTrans);
                return null;
            }
            return Find(physics.circles) ?? Find(physics.boxes) ?? Find(physics.roundedRects)
                ?? throw new InvalidOperationException("Composite refers to an unregistered body: " + id);
        }

        public static string Identity(UnityEngine.Object value)
        {
            if (value == null) return "null";
            if (value is MonoUpdatable updatable)
                return updatable.GetType().Name + ":" + updatable.HierarchyNumber;
            if (value is Component component) return PathOf(component.transform) + ":" + component.GetType().Name;
            if (value is GameObject gameObject) return PathOf(gameObject.transform);
            return value.GetType().FullName + ":" + value.name;
        }

        private static string PathOf(Transform transform)
        {
            string path = "";
            while (transform != null)
            {
                var follower = transform.GetComponent<FollowTransform>();
                if (follower != null && follower.Leader != null && follower.Leader != transform)
                    return "follow:" + PathOf(follower.Leader) + "/" + transform.name + path;
                var anchor = transform.GetComponents<MonoUpdatable>().OrderBy(c => c.HierarchyNumber).FirstOrDefault();
                if (anchor != null && transform.gameObject.scene.IsValid())
                    return "entity:" + anchor.HierarchyNumber + path;
                int ordinal = 0;
                if (transform.parent != null)
                    for (int i = 0; i < transform.GetSiblingIndex(); i++)
                        if (transform.parent.GetChild(i).name == transform.name) ordinal++;
                path = "/" + transform.name + "[" + ordinal + "]" + path;
                if (transform.parent == null)
                    return transform.gameObject.scene.name + path;
                transform = transform.parent;
            }
            throw new InvalidOperationException("missing transform identity");
        }
    }

    private static FieldInfo[] Fields(Type type)
    {
        if (fields.TryGetValue(type, out FieldInfo[] result)) return result;
        var list = new List<FieldInfo>();
        for (Type t = type; t != null && t != typeof(MonoBehaviour) && t != typeof(ScriptableObject) && t != typeof(object); t = t.BaseType)
            list.AddRange(t.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.DeclaredOnly));
        result = list.OrderBy(f => f.DeclaringType.FullName + "." + f.Name, StringComparer.Ordinal).ToArray();
        fields.Add(type, result);
        return result;
    }

    private sealed class ReferenceComparer : IEqualityComparer<object>
    {
        public new bool Equals(object x, object y) => ReferenceEquals(x, y);
        public int GetHashCode(object value) => System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(value);
    }
}
