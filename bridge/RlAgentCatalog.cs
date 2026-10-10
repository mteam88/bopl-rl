using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json.Linq;
using UnityEngine;

// The opponents the in-game picker (RlAgentPicker) offers, and the choice it
// made. BOPL_AGENT_CATALOG names the JSON Tools/bot_catalog.py writes (league
// members with weights, the scripted bots, rosters: spec, kits, rating,
// description); specs that are paths are relative to BOPL_AGENT_ROOT (the
// checkout), and a bot whose checkpoint or sidecar is missing on this machine
// is left out. The choice is a bot plus a kit: one of the bot's kits, or
// Random (a new kit from its set every round, as training drew one per
// episode). It starts from BOPL_AGENT (a launcher argument) and BOPL_AGENT_KIT
// when given, else the last confirmed choice (<persistentDataPath>/
// bopl-agent-choice.json), else the catalog's default.
public static class RlAgentCatalog
{
    public sealed class Bot
    {
        public string Id, Title, Kind, Group, Spec, Description, RatingSource;
        public float? Rating;
        public int[] Interval;
        public List<int[]> Kits = new List<int[]>();     // empty: the shim's own loadout (ad-hoc specs)
        public Dictionary<string, float> KitRatings = new Dictionary<string, float>();
        public string Path;                               // what BOPL_AGENT takes: absolute path or bot name
        public bool CanRandomize => Kits.Count > 1;
    }

    public const int RandomKit = -1;

    public static readonly List<Bot> Bots = new List<Bot>();
    public static readonly List<string> Groups = new List<string>();
    public static readonly Dictionary<int, string> AbilityNames = new Dictionary<int, string>();
    public static string RatingNote;
    public static bool Loaded { get; private set; }
    public static string LoadError { get; private set; }

    public static Bot Chosen { get; private set; }
    public static int ChosenKit { get; private set; } = RandomKit;

    private static readonly string catalogPath = Environment.GetEnvironmentVariable("BOPL_AGENT_CATALOG");
    private static string root;
    private static string ChoicePath => System.IO.Path.Combine(Application.persistentDataPath, "bopl-agent-choice.json");

    public static bool Present => !string.IsNullOrEmpty(catalogPath);

    public static void EnsureLoaded()
    {
        if (Loaded || !Present) return;
        Loaded = true;
        root = Environment.GetEnvironmentVariable("BOPL_AGENT_ROOT");
        if (string.IsNullOrEmpty(root)) root = System.IO.Path.GetFullPath(System.IO.Path.Combine(System.IO.Path.GetDirectoryName(catalogPath), "../../../.."));
        string defaultBot = null, defaultKit = "random";
        try {
            var json = JObject.Parse(File.ReadAllText(catalogPath));
            RatingNote = (string)json["rating_note"];
            foreach (var pair in (JObject)json["abilities"]) AbilityNames[int.Parse(pair.Key)] = (string)pair.Value;
            foreach (var group in (JArray)json["groups"]) Groups.Add((string)group);
            defaultBot = (string)json["default"]?["bot"];
            defaultKit = (string)json["default"]?["kit"] ?? "random";
            foreach (JObject row in (JArray)json["bots"]) {
                var bot = new Bot {
                    Id = (string)row["id"], Title = (string)row["title"], Kind = (string)row["kind"], Group = (string)row["group"],
                    Spec = (string)row["spec"], Description = (string)row["description"], RatingSource = (string)row["rating_source"],
                    Rating = row["rating"]?.Type == JTokenType.Float || row["rating"]?.Type == JTokenType.Integer ? (float?)(float)row["rating"] : null,
                    Interval = row["interval"] is JArray interval ? new[] { (int)interval[0], (int)interval[1] } : null
                };
                foreach (JArray entry in (JArray)row["kits"]) bot.Kits.Add(new[] { (int)entry[0], (int)entry[1], (int)entry[2] });
                if (row["kit_ratings"] is JObject kitRatings)
                    foreach (var pair in kitRatings) bot.KitRatings[pair.Key] = (float)pair.Value;
                bot.Path = Resolve(bot.Spec);
                if (Available(bot.Path)) Bots.Add(bot);
            }
        } catch (Exception exception) {
            LoadError = "bot catalog " + catalogPath + ": " + exception.Message;
            Debug.LogError("BOPL_AGENT: " + LoadError);
        }
        // Starting choice: the launcher's argument, the saved choice, the default.
        string spec = Environment.GetEnvironmentVariable("BOPL_AGENT"), kit = Environment.GetEnvironmentVariable("BOPL_AGENT_KIT");
        if (string.IsNullOrEmpty(spec)) {
            try {
                if (File.Exists(ChoicePath)) {
                    var saved = JObject.Parse(File.ReadAllText(ChoicePath));
                    var bot = Bots.Find(b => b.Id == (string)saved["bot"]);
                    if (bot == null && saved["spec"] != null && Available((string)saved["spec"])) bot = AdHoc((string)saved["spec"]);
                    if (bot != null) Choose(bot, ParseKit(bot, (string)saved["kit"]), save: false);
                }
            } catch (Exception exception) { Debug.LogWarning("BOPL_AGENT: saved choice: " + exception.Message); }
        } else {
            string full = Resolve(spec);
            var bot = Bots.Find(b => b.Path == full || b.Spec == spec || b.Id == spec) ?? AdHoc(full);
            if (bot != null) Choose(bot, ParseKit(bot, kit), save: false);
        }
        if (Chosen == null) {
            var bot = Bots.Find(b => b.Id == defaultBot) ?? (Bots.Count > 0 ? Bots[0] : null);
            if (bot != null) Choose(bot, ParseKit(bot, defaultKit), save: false);
        }
        Debug.Log("BOPL_AGENT: catalog " + catalogPath + ": " + Bots.Count + " bots available; starting with " + Describe());
    }

    public static string Resolve(string spec)
    {
        if (string.IsNullOrEmpty(spec)) return spec;
        bool path = spec.EndsWith(".bin") || spec.EndsWith(".roster");
        return path && !System.IO.Path.IsPathRooted(spec) ? System.IO.Path.GetFullPath(System.IO.Path.Combine(root ?? "", spec)) : spec;
    }

    private static bool Available(string path)
    {
        if (path.EndsWith(".bin")) return File.Exists(path) && File.Exists(path + ".json");
        if (path.EndsWith(".roster")) return File.Exists(path);
        return true;
    }

    // A spec that is not in the catalog (a raw checkpoint or roster given on the
    // command line): listed under "Command line" on the loadout the shim reports.
    private static Bot AdHoc(string spec)
    {
        if (!RlAgent.Probe(spec, out int[] kit, out string error)) {
            Debug.LogError("BOPL_AGENT=" + spec + ": " + error);
            return null;
        }
        string name = spec.EndsWith(".bin") || spec.EndsWith(".roster") ? System.IO.Path.GetFileNameWithoutExtension(spec) : spec;
        var bot = new Bot {
            Id = name, Title = name.Replace('-', ' ').ToUpperInvariant(), Group = "Command line", Spec = spec, Path = spec,
            Kind = spec.EndsWith(".bin") ? "checkpoint" : spec.EndsWith(".roster") ? "roster" : "scripted",
            Description = "Given on the command line: " + spec + ". Plays the loadout it reports."
        };
        bot.Kits.Add(kit);
        if (!Groups.Contains(bot.Group)) Groups.Insert(0, bot.Group);
        Bots.Insert(0, bot);
        return bot;
    }

    private static int ParseKit(Bot bot, string text)
    {
        if (bot.Kits.Count <= 1) return 0;
        if (string.IsNullOrEmpty(text) || text == "random") return RandomKit;
        string[] parts = text.Split(',');
        if (parts.Length == 3)
            for (int i = 0; i < bot.Kits.Count; i++)
                if (KitKey(bot.Kits[i]) == string.Join(",", parts).Replace(" ", "")) return i;
        return RandomKit;
    }

    public static string KitKey(int[] kit) => kit[0] + "," + kit[1] + "," + kit[2];

    // What the results log records as the choice: "random" or the kit.
    public static string KitChoiceText => Chosen == null ? null : ChosenKit == RandomKit ? "random" : Chosen.Kits.Count > 0 ? KitKey(Chosen.Kits[ChosenKit]) : "own";

    public static string Describe() => Chosen == null ? "nothing" : Chosen.Id + " (" + KitChoiceText + ")";

    public static void Choose(Bot bot, int kit, bool save = true)
    {
        Chosen = bot;
        ChosenKit = bot.CanRandomize ? Mathf.Clamp(kit, RandomKit, bot.Kits.Count - 1) : 0;
        if (!save) return;
        try {
            var json = new JObject { ["bot"] = bot.Id, ["spec"] = bot.Path, ["kit"] = KitChoiceText, ["time"] = DateTime.UtcNow.ToString("o") };
            File.WriteAllText(ChoicePath, json.ToString());
        } catch (Exception exception) { Debug.LogWarning("BOPL_AGENT: could not save the choice: " + exception.Message); }
        Debug.Log("BOPL_AGENT: chose " + Describe());
    }

    // The kit for a round: the chosen one, or a draw from the set.
    public static int[] KitFor(System.Random random)
    {
        if (Chosen == null || Chosen.Kits.Count == 0) return null;
        return ChosenKit == RandomKit ? Chosen.Kits[random.Next(Chosen.Kits.Count)] : Chosen.Kits[ChosenKit];
    }
}
