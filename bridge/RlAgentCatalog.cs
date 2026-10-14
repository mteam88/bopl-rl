using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json.Linq;
using UnityEngine;

// The opponents the in-game picker (RlAgentPicker) offers, and the bots it
// seated. BOPL_AGENT_CATALOG names the JSON Tools/bot_catalog.py writes (league
// members with weights, the scripted bots, rosters: spec, kits, pool, rating,
// description); specs that are paths are relative to BOPL_AGENT_ROOT (the
// checkout), and a bot whose checkpoint or sidecar is missing on this machine
// is left out. Each seat is a bot plus a kit: Random (a new kit every round, as
// training drew one per episode: from the bot's lineup set, or three abilities
// from its pool for a bot trained on random kits) or a fixed kit, one of its
// set or any three abilities its pool allows (Bot.Pool: the trained pool, else
// the abilities of its set). The seats start from BOPL_AGENT (a launcher
// argument) and BOPL_AGENT_KIT when given, else the last confirmed seats
// (<persistentDataPath>/bopl-agent-choice.json), else the catalog's default.
public static class RlAgentCatalog
{
    public sealed class Bot
    {
        public string Id, Title, Kind, Group, Spec, Description, RatingSource;
        public float? Rating;
        public int[] Interval;
        public List<int[]> Kits = new List<int[]>();     // empty: the shim's own loadout (ad-hoc specs)
        public Dictionary<string, float> KitRatings = new Dictionary<string, float>();
        public List<int> Pool = new List<int>();          // what a slot can hold; empty: no custom kits
        public bool AnyKit;                               // trained on random kits from Pool
        public string Path;                               // what BOPL_AGENT takes: absolute path or bot name
        public bool CanRandomize => Kits.Count > 1 || AnyKit;
        public bool CanCustomize => Pool.Count > 0 && CanRandomize;
        // Where a custom kit starts: its first listed kit, else the training default.
        public int[] BaseKit => Kits.Count > 0 ? (int[])Kits[0].Clone() : new[] { 2, 4, 14 };
    }

    // One bot at the character select; Kit null is Random (or, for a bot with
    // neither a set nor a pool, the loadout the shim reports).
    public sealed class Seat
    {
        public Bot Bot;
        public int[] Kit;
        public Seat Copy() => new Seat { Bot = Bot, Kit = Kit == null ? null : (int[])Kit.Clone() };
    }

    public const int MaxSeats = 5;   // six select boxes, one at least for a human

    public static readonly List<Bot> Bots = new List<Bot>();
    public static readonly List<string> Groups = new List<string>();
    public static readonly Dictionary<int, string> AbilityNames = new Dictionary<int, string>();
    public static readonly List<Seat> Seats = new List<Seat>();
    public static string RatingNote;
    public static bool Loaded { get; private set; }
    public static string LoadError { get; private set; }
    public static Bot DefaultBot { get; private set; }

    private static readonly string catalogPath = Environment.GetEnvironmentVariable("BOPL_AGENT_CATALOG");
    private static string root, defaultKit = "random";
    private static string ChoicePath => System.IO.Path.Combine(Application.persistentDataPath, "bopl-agent-choice.json");

    public static bool Present => !string.IsNullOrEmpty(catalogPath);

    public static void EnsureLoaded()
    {
        if (Loaded || !Present) return;
        Loaded = true;
        root = Environment.GetEnvironmentVariable("BOPL_AGENT_ROOT");
        if (string.IsNullOrEmpty(root)) root = System.IO.Path.GetFullPath(System.IO.Path.Combine(System.IO.Path.GetDirectoryName(catalogPath), "../../../.."));
        string defaultBot = null;
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
                if (!Available(bot.Path)) continue;
                // A pool from the catalog, else (catalogs older than the field,
                // bots added by hand) from the checkpoint's own sidecar.
                if (row.ContainsKey("pool")) { if (row["pool"] is JArray pool) SetPool(bot, pool); }
                else if (bot.Path.EndsWith(".bin")) SidecarPool(bot);
                if (!bot.AnyKit && bot.Kits.Count > 1) {
                    var union = new SortedSet<int>();
                    foreach (var kit in bot.Kits) foreach (int ability in kit) if (ability > 1) union.Add(ability);
                    bot.Pool.AddRange(union);
                }
                Bots.Add(bot);
            }
        } catch (Exception exception) {
            LoadError = "bot catalog " + catalogPath + ": " + exception.Message;
            Debug.LogError("BOPL_AGENT: " + LoadError);
        }
        DefaultBot = Bots.Find(b => b.Id == defaultBot) ?? (Bots.Count > 0 ? Bots[0] : null);
        // Starting seats: the launcher's argument, the saved seats, the default.
        string spec = Environment.GetEnvironmentVariable("BOPL_AGENT"), kitText = Environment.GetEnvironmentVariable("BOPL_AGENT_KIT");
        if (string.IsNullOrEmpty(spec)) {
            try {
                if (File.Exists(ChoicePath)) {
                    var saved = JObject.Parse(File.ReadAllText(ChoicePath));
                    // "bots": the seats; older files hold one bot at the top level.
                    var entries = saved["bots"] is JArray list ? new List<JObject>(list.Children<JObject>()) : new List<JObject> { saved };
                    foreach (var entry in entries) {
                        if (Seats.Count >= MaxSeats) break;
                        var bot = Bots.Find(b => b.Id == (string)entry["bot"]);
                        if (bot == null && entry["spec"] != null && Available((string)entry["spec"])) bot = AdHoc((string)entry["spec"]);
                        if (bot != null) Seats.Add(new Seat { Bot = bot, Kit = ParseKit(bot, (string)entry["kit"]) });
                    }
                    if (saved["bots"] is JArray none && none.Count == 0) savedEmpty = true;
                }
            } catch (Exception exception) { Debug.LogWarning("BOPL_AGENT: saved choice: " + exception.Message); }
        } else {
            string full = Resolve(spec);
            var bot = Bots.Find(b => b.Path == full || b.Spec == spec || b.Id == spec) ?? AdHoc(full);
            if (bot != null) Seats.Add(new Seat { Bot = bot, Kit = ParseKit(bot, kitText) });
        }
        if (Seats.Count == 0 && !savedEmpty && DefaultBot != null) Seats.Add(NewSeat());
        Debug.Log("BOPL_AGENT: catalog " + catalogPath + ": " + Bots.Count + " bots available; starting with " + Describe());
    }

    // The last confirmed seats were none (every bot removed): start with none.
    private static bool savedEmpty;

    private static void SetPool(Bot bot, JArray pool)
    {
        // 0 is no ability and 1 the select screen's "Random" tile, which no player carries.
        foreach (var ability in pool) if ((int)ability > 1 && !bot.Pool.Contains((int)ability)) bot.Pool.Add((int)ability);
        bot.AnyKit = bot.Pool.Count > 0;
    }

    private static void SidecarPool(Bot bot)
    {
        try {
            var meta = JObject.Parse(File.ReadAllText(bot.Path + ".json"));
            if (meta["config"]?["loadouts"] is JObject loadouts && (string)loadouts["mode"] == "pool" && loadouts["pool"] is JArray pool) SetPool(bot, pool);
        } catch (Exception exception) { Debug.LogWarning("BOPL_AGENT: " + bot.Path + ".json: " + exception.Message); }
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
        if (spec.EndsWith(".bin")) SidecarPool(bot);
        if (!Groups.Contains(bot.Group)) Groups.Insert(0, bot.Group);
        Bots.Insert(0, bot);
        return bot;
    }

    // A seat for a new bot: a copy of from, else the catalog's default.
    public static Seat NewSeat(Seat from = null)
    {
        if (from != null) return from.Copy();
        var seat = new Seat { Bot = DefaultBot };
        seat.Kit = ParseKit(DefaultBot, defaultKit);
        return seat;
    }

    private static int[] ParseKit(Bot bot, string text)
    {
        if (!bot.CanRandomize) return bot.Kits.Count > 0 ? (int[])bot.Kits[0].Clone() : null;
        if (string.IsNullOrEmpty(text) || text == "random") return null;
        string[] parts = text.Replace(" ", "").Split(',');
        if (parts.Length != 3) return null;
        var kit = new int[3];
        for (int i = 0; i < 3; i++) if (!int.TryParse(parts[i], out kit[i])) return null;
        return Allowed(bot, kit) ? kit : null;
    }

    // A kit this bot may be seated with: one of its set, or three of its pool.
    public static bool Allowed(Bot bot, int[] kit)
    {
        if (kit == null) return bot.CanRandomize || bot.Kits.Count == 0;
        if (KitIndex(bot, kit) >= 0) return true;
        return bot.CanCustomize && bot.Pool.Contains(kit[0]) && bot.Pool.Contains(kit[1]) && bot.Pool.Contains(kit[2]);
    }

    public static int KitIndex(Bot bot, int[] kit)
    {
        if (kit == null) return -1;
        for (int i = 0; i < bot.Kits.Count; i++)
            if (bot.Kits[i][0] == kit[0] && bot.Kits[i][1] == kit[1] && bot.Kits[i][2] == kit[2]) return i;
        return -1;
    }

    public static string KitKey(int[] kit) => kit[0] + "," + kit[1] + "," + kit[2];

    // What the results log records as the choice: "random", the kit, or "own".
    public static string KitChoiceText(Seat seat) => seat == null ? null : seat.Kit != null ? KitKey(seat.Kit) : seat.Bot.CanRandomize ? "random" : "own";

    public static string Describe()
    {
        if (Seats.Count == 0) return "no bot";
        var parts = new List<string>();
        foreach (var seat in Seats) parts.Add(seat.Bot.Id + " (" + KitChoiceText(seat) + ")");
        return string.Join(", ", parts);
    }

    // The picker confirmed these seats: keep them for the next match and launch.
    public static void SetSeats(List<Seat> seats, bool save = true)
    {
        Seats.Clear();
        foreach (var seat in seats) Seats.Add(seat.Copy());
        if (!save) return;
        try {
            var list = new JArray();
            foreach (var seat in Seats) list.Add(new JObject { ["bot"] = seat.Bot.Id, ["spec"] = seat.Bot.Path, ["kit"] = KitChoiceText(seat) });
            // The first bot also at the top level, as builds before the seats read it.
            var json = new JObject { ["bots"] = list, ["time"] = DateTime.UtcNow.ToString("o") };
            if (Seats.Count > 0) { json["bot"] = Seats[0].Bot.Id; json["spec"] = Seats[0].Bot.Path; json["kit"] = KitChoiceText(Seats[0]); }
            File.WriteAllText(ChoicePath, json.ToString());
        } catch (Exception exception) { Debug.LogWarning("BOPL_AGENT: could not save the choice: " + exception.Message); }
        Debug.Log("BOPL_AGENT: chose " + Describe());
    }

    // The kit for a round: the seat's, or a draw (from the set, or three from
    // the pool as training drew them: every slot independently, repeats allowed).
    public static int[] KitFor(Seat seat, System.Random random)
    {
        if (seat == null) return null;
        if (seat.Kit != null) return seat.Kit;
        var bot = seat.Bot;
        if (bot.AnyKit) return new[] { bot.Pool[random.Next(bot.Pool.Count)], bot.Pool[random.Next(bot.Pool.Count)], bot.Pool[random.Next(bot.Pool.Count)] };
        return bot.Kits.Count > 0 ? bot.Kits[random.Next(bot.Kits.Count)] : null;
    }
}
