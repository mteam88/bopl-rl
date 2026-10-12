using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using BoplFixedMath;
using Newtonsoft.Json.Linq;
using UnityEngine;

// In-game AI player (native/rl/PLAY_UNITY.md). With BOPL_AGENT set to a policy
// checkpoint (policy-*.bin with its .bin.json sidecar) or a scripted bot name
// (mixed-v2-hard, ...), the local character select adds one AI player without a
// device (CharacterSelectHandler.TryStartGame_inner) carrying the loadout the
// agent trained with, and every simulation tick of a round is run through the
// agent shim libbopl_agent.so (native/rl/puffer/bopl_agent.h), which is the
// training decision loop (bopl_puffer.h) fed by this game's state instead of
// its own simulator:
//   - the first WarmupTicks (240) simulation ticks after a level load are the
//     training warmup: every player, human included, is fed a zero packet
//     (RlBridge.Reset / PumpInner ran them with all-zero inputs);
//   - from then on the state before each tick (RlObservation.Write, the same
//     record RlBridge sends) goes to the shim, which returns the InputPacket
//     the agent applies to that tick via Player.OverrideInputWithNetworkInput,
//     exactly where RlBridge.Tick applies training actions;
//   - Host.CurrentInputAsPacket reports the applied packets so the auto-saved
//     replay (Host.SaveReplay) holds the agent's real inputs;
//   - ability pickups are disabled as in the training host (sim/Host/RlMode.cs)
//     unless BOPL_RL_ABILITY_SPAWNS=1.
// Ticks are counted per TickSimulation call, which is the simulation timeline
// the training sims and the replay share (Updater.Update skips the call on
// hitstop frames; stock, it still advanced the simulation clock there, which
// FreezesHitstopClock stops while the agent plays). Each round writes one JSON line
// to BOPL_AGENT_RESULTS (seed, players, winner, replay file) for
// Tools/play-vs-agent.sh.
// With BOPL_AGENT_CATALOG set (the launchers set it), the opponent and its kit
// come from the in-game picker instead (RlAgentPicker on the character select,
// RlAgentCatalog for the bots and the remembered choice; BOPL_AGENT then only
// preselects): the agent plays the chosen kit, or a kit drawn from its set for
// every round. The shim needs no loadout of its own for that: the observation
// and mask read the abilities from the game state. With neither variable set
// the stock game is untouched.
// With BOPL_AGENT_SHADOW=1 the shim also runs a shadow sim (train/bopl_shadow.h,
// docs/SHADOW.md): the q32 Rust sim reset from this round's own replay setup
// after level tick 1 and stepped with every player's applied packets, compared
// with this game's observation every tick. The first mismatch of a round is
// logged ("BOPL_AGENT_SHADOW: ... DESYNC ...") and shadowing stops for that
// round. While the agent plays, hitstop frames freeze the simulation clock
// (FreezesHitstopClock) as every sim does; BOPL_AGENT_HITSTOP_TIME=1 restores
// the stock clock for an A/B.
// With BOPL_AGENT_SEARCH=k,n,depth (implies the shadow; docs/INGAME_SEARCH.md)
// each of the agent's decisions is searched on forks of the in-sync shadow
// world (train/bopl_search.h, the research library's search, every player
// modelled by the agent's own policy), synchronously inside the shim tick.
// When the shadow is not tracking, the round reaches the agent's episode cap,
// or the search fails, the agent plays the raw policy for the rest of the
// round ("BOPL_AGENT_SEARCH: round ... raw policy ..."). Each round logs the
// decision timing and the frame times, and its result line carries them.
public static class RlAgent
{
    [StructLayout(LayoutKind.Sequential)]
    private struct Action { public byte angle, reserved; public ushort buttons; }

    private const string Library = "bopl_agent";
    [DllImport(Library)] private static extern uint bopl_agent_abi_version();
    [DllImport(Library)] private static extern IntPtr bopl_agent_open([MarshalAs(UnmanagedType.LPStr)] string spec, int players, int player, byte[] error, UIntPtr capacity);
    [DllImport(Library)] private static extern void bopl_agent_reset(IntPtr agent, uint seed);
    [DllImport(Library)] private static extern int bopl_agent_tick_bytes(IntPtr agent, byte[] bytes, UIntPtr length, out Action action);
    [DllImport(Library)] private static extern void bopl_agent_close(IntPtr agent);
    [DllImport(Library)] private static extern int bopl_agent_ability(IntPtr agent, int slot);
    [DllImport(Library)] private static extern int bopl_agent_warmup_ticks(IntPtr agent);
    [DllImport(Library)] private static extern int bopl_agent_max_episode_ticks(IntPtr agent);
    [DllImport(Library)] private static extern int bopl_agent_decisions(IntPtr agent);
    [DllImport(Library)] private static extern IntPtr bopl_agent_id(IntPtr agent);
    [DllImport(Library)] private static extern IntPtr bopl_shadow_open([MarshalAs(UnmanagedType.LPStr)] string library, byte[] error, UIntPtr capacity);
    [DllImport(Library)] private static extern int bopl_shadow_begin(IntPtr shadow, byte[] archive, UIntPtr length, uint warmup);
    [DllImport(Library)] private static extern int bopl_shadow_check_bytes(IntPtr shadow, byte[] bytes, UIntPtr length);
    [DllImport(Library)] private static extern int bopl_shadow_step(IntPtr shadow, Action[] actions, UIntPtr count);
    [DllImport(Library)] private static extern int bopl_shadow_end(IntPtr shadow);
    [DllImport(Library)] private static extern IntPtr bopl_shadow_report(IntPtr shadow);
    [DllImport(Library)] private static extern uint bopl_shadow_compared(IntPtr shadow);
    [DllImport(Library)] private static extern IntPtr bopl_shadow_library(IntPtr shadow);
    [DllImport(Library)] private static extern void bopl_agent_set_map(IntPtr agent, uint map);
    [DllImport(Library)] private static extern int bopl_agent_search(IntPtr agent, IntPtr shadow, [MarshalAs(UnmanagedType.LPStr)] string budget, byte[] error, UIntPtr capacity);
    [DllImport(Library)] private static extern int bopl_agent_search_active(IntPtr agent);
    [DllImport(Library)] private static extern IntPtr bopl_agent_search_note(IntPtr agent);
    [DllImport(Library)] private static extern int bopl_agent_search_stats(IntPtr agent, double[] stats, int capacity);
    private const int SearchStats = 14;
    private const int ShadowOff = 0, ShadowTracking = 1, ShadowDesync = 2, ShadowEnded = 3, ShadowFailed = 4;
    private static readonly string[] ShadowNames = { "off", "tracking", "desync", "ended", "failed" };

    private static readonly string envSpec = Environment.GetEnvironmentVariable("BOPL_AGENT");
    public static string Spec { get; private set; } = envSpec;
    public static bool Enabled => RlAgentCatalog.Present || !string.IsNullOrEmpty(envSpec);
    private static readonly string resultsPath = Environment.GetEnvironmentVariable("BOPL_AGENT_RESULTS");

    private static IntPtr handle;
    private static string id = "";
    private static int[] abilities;          // the shim's loadout; the kit played is kit
    private static int[] kit;
    private static System.Random kitDraw;
    private static RlAgentCatalog.Bot bot;
    private static string kitChoice;
    private static Player agentPlayer;
    private static NamedSpriteList abilityCatalog;
    private static int warmupTicks, maxEpisodeTicks;
    private static string openError;

    // Round state.
    private static Updater updater;          // the level scene's Updater; a new one is a new round
    private static int round = -1, tick, episodeTicks, roundLevel;
    private static uint baseSeed, roundSeed;
    private static bool started, terminal, failed, humansFrozen, resultWritten;
    private static int agentIndex = -1;
    private static readonly MemoryStream observation = new MemoryStream(65536);
    private static readonly BinaryWriter writer = new BinaryWriter(observation, Encoding.UTF8, true);
    private static readonly Dictionary<int, InputPacket> applied = new Dictionary<int, InputPacket>();
    private static InputPacket agentPacket;

    // Shadow sim (BOPL_AGENT_SHADOW=1).
    // Search (BOPL_AGENT_SEARCH=k,n,depth[,...]; "1" means the default 8,1,1).
    private static readonly string searchBudget = SearchBudget(Environment.GetEnvironmentVariable("BOPL_AGENT_SEARCH"));
    public static readonly bool ShadowRequested = Environment.GetEnvironmentVariable("BOPL_AGENT_SHADOW") == "1" || searchBudget != null;
    private static readonly bool stockHitstopClock = Environment.GetEnvironmentVariable("BOPL_AGENT_HITSTOP_TIME") == "1";
    private static IntPtr shadow;
    private static bool shadowOpenTried;
    private static int shadowStatus;
    private static string shadowReport = "";
    private static Action[] shadowPackets = new Action[0];
    // This round's recording. Host.SaveReplay drops ReplaySession.Recording on
    // the tick the round ends, before AfterTick steps the shadow with that tick.
    private static ReplayArchive shadowRecording;

    private static bool searchOn, searchFellBackLogged;
    private static readonly double[] searchStats = new double[SearchStats];
    // This round's wall times: the shim tick on decision ticks and on the
    // others (ms), and the frames (Time.unscaledDeltaTime, ms) while it plays.
    private static readonly List<double> decisionMs = new List<double>(4096), otherMs = new List<double>(16384), frameMs = new List<double>(16384);
    private static int lastFrame = -1;
    private static readonly System.Diagnostics.Stopwatch tickWatch = new System.Diagnostics.Stopwatch();

    private static string SearchBudget(string value)
    {
        if (string.IsNullOrEmpty(value) || value == "0") return null;
        return value == "1" ? "8,1,1" : value;
    }

    public static string Id => id;
    public static int WarmupTicks => warmupTicks;

    // Opens the shim once (players/agent index fixed for the whole match).
    private static bool Open(int players, int player)
    {
        if (handle != IntPtr.Zero) { bopl_agent_close(handle); handle = IntPtr.Zero; }
        var error = new byte[1024];
        try {
            if (bopl_agent_abi_version() != 1) { openError = "libbopl_agent.so ABI mismatch"; return false; }
            handle = bopl_agent_open(Spec, players, player, error, (UIntPtr)error.Length);
        } catch (DllNotFoundException exception) {
            openError = "libbopl_agent.so not found (Tools/play-vs-agent.sh copies it into BoplBattle_Data/Plugins/x86_64): " + exception.Message;
            return false;
        }
        if (handle == IntPtr.Zero) {
            openError = Encoding.UTF8.GetString(error).TrimEnd('\0');
            return false;
        }
        id = Marshal.PtrToStringAnsi(bopl_agent_id(handle));
        abilities = new[] { bopl_agent_ability(handle, 0), bopl_agent_ability(handle, 1), bopl_agent_ability(handle, 2) };
        warmupTicks = bopl_agent_warmup_ticks(handle);
        maxEpisodeTicks = bopl_agent_max_episode_ticks(handle);
        agentIndex = player;
        OpenShadow();
        AttachSearch();
        return true;
    }

    // One shadow world per launch (the library loads once per process).
    private static void OpenShadow()
    {
        if (!ShadowRequested || shadowOpenTried) return;
        shadowOpenTried = true;
        var error = new byte[2048];
        try {
            shadow = bopl_shadow_open(null, error, (UIntPtr)error.Length);
        } catch (EntryPointNotFoundException exception) {
            Debug.LogError("BOPL_AGENT_SHADOW: this libbopl_agent has no shadow (rebuild it: make agent): " + exception.Message);
            return;
        }
        if (shadow == IntPtr.Zero) { Debug.LogError("BOPL_AGENT_SHADOW: " + Encoding.UTF8.GetString(error).TrimEnd('\0')); return; }
        Debug.Log("BOPL_AGENT_SHADOW: shadow sim " + Marshal.PtrToStringAnsi(bopl_shadow_library(shadow)));
    }

    // The search wraps this shim (a new one per match) around the shadow world.
    private static void AttachSearch()
    {
        searchOn = false;
        if (searchBudget == null) return;
        if (shadow == IntPtr.Zero) { Debug.LogError("BOPL_AGENT_SEARCH: no shadow sim, the agent plays the raw policy"); return; }
        var error = new byte[1024];
        try {
            if (bopl_agent_search(handle, shadow, searchBudget, error, (UIntPtr)error.Length) != 0) {
                Debug.LogError("BOPL_AGENT_SEARCH=" + searchBudget + ": " + Encoding.UTF8.GetString(error).TrimEnd('\0') + "; the agent plays the raw policy");
                return;
            }
        } catch (EntryPointNotFoundException exception) {
            Debug.LogError("BOPL_AGENT_SEARCH: this libbopl_agent has no search (rebuild it: make agent): " + exception.Message);
            return;
        }
        searchOn = true;
        Debug.Log("BOPL_AGENT_SEARCH: budget " + searchBudget + " on the shadow world");
    }

    // Updater.Update on a hitstop frame: true when the simulation clock
    // (SimTimePassed, SimTimeSinceLevelLoaded, SimulationTicks) must not advance.
    // The training sims, the replay player and the shadow never see hitstop
    // frames, and sudden death, platform/orbit motion and several abilities
    // read that clock, so while the agent plays a hitstop only freezes the
    // picture. BOPL_AGENT_HITSTOP_TIME=1 keeps the stock behaviour.
    public static bool FreezesHitstopClock => !stockHitstopClock && Active(out _);

    // RlAgentPicker: can the shim open spec? Its loadout comes back in kit.
    public static bool Probe(string spec, out int[] kit, out string error)
    {
        kit = null; error = null;
        var buffer = new byte[1024];
        try {
            if (bopl_agent_abi_version() != 1) { error = "libbopl_agent.so ABI mismatch"; return false; }
            IntPtr probe = bopl_agent_open(spec, 2, 1, buffer, (UIntPtr)buffer.Length);
            if (probe == IntPtr.Zero) { error = Encoding.UTF8.GetString(buffer).TrimEnd('\0'); return false; }
            kit = new[] { bopl_agent_ability(probe, 0), bopl_agent_ability(probe, 1), bopl_agent_ability(probe, 2) };
            bopl_agent_close(probe);
            return true;
        } catch (DllNotFoundException exception) {
            error = "libbopl_agent.so not found: " + exception.Message;
            return false;
        }
    }

    // The colour AddPlayers gives the agent: the first one no human took.
    public static int ColorIndex(List<Material> taken, PlayerColors colors)
    {
        int color = 0;
        while (color < colors.playerColors.Length - 1 && taken.Contains(colors[color].playerMaterial)) color++;
        return color;
    }

    // CharacterSelectHandler.TryStartGame_inner: append the AI player after the
    // human ones. Team and colour are ones no human took; the loadout is the
    // picker's kit (RlAgentCatalog.KitFor), else the agent's training loadout
    // (sidecar ability1..3, or the training defaults).
    public static void AddPlayers(List<Player> list, ref int nextId, PlayerColors colors, NamedSpriteList catalog)
    {
        if (!Enabled) return;
        bot = null; kitChoice = null;
        if (RlAgentCatalog.Present) {
            RlAgentCatalog.EnsureLoaded();
            bot = RlAgentCatalog.Chosen;
            if (bot == null) { Debug.LogError("BOPL_AGENT: no opponent chosen (" + (RlAgentCatalog.LoadError ?? "the catalog has no bot available here") + ")"); return; }
            Spec = bot.Path;
            kitChoice = RlAgentCatalog.KitChoiceText;
        }
        if (!Open(list.Count + 1, list.Count)) {
            Debug.LogError("BOPL_AGENT=" + Spec + ": " + openError);
            return;
        }
        int team = 0;
        while (list.Exists(p => p.Team == team)) team++;
        int color = ColorIndex(list.ConvertAll(p => p.Color), colors);
        var player = new Player(nextId, team) {
            Color = colors[color].playerMaterial, UsesKeyboardAndMouse = false, CanUseAbilities = true,
            IsLocalPlayer = false, IsAgent = true, inputDevice = null,
            Abilities = new List<GameObject>(3), AbilityIcons = new List<Sprite>(3)
        };
        // One seed sequence per launch: a later match (after the picker changed the
        // bot) continues the round count, so no two rounds of a session share a seed.
        if (kitDraw == null) {
            string seed = Environment.GetEnvironmentVariable("BOPL_AGENT_SEED");
            baseSeed = !string.IsNullOrEmpty(seed) && uint.TryParse(seed, out uint parsed) ? parsed : (uint)Environment.TickCount;
            kitDraw = new System.Random(unchecked((int)baseSeed));
        }
        agentPlayer = player; abilityCatalog = catalog;
        SetKit(RlAgentCatalog.Present ? RlAgentCatalog.KitFor(kitDraw) ?? abilities : abilities);
        nextId++;
        list.Add(player);
        updater = null;
        Debug.Log("BOPL_AGENT: " + id + (bot == null ? "" : " (" + bot.Id + ", kit " + kitChoice + ")") + " joins as player " + player.Id
                  + " (team " + team + ", abilities " + string.Join(",", kit) + "), base seed " + baseSeed);
    }

    // Gives the agent's player this kit (ability catalog indices).
    private static void SetKit(int[] next)
    {
        kit = next;
        agentPlayer.Abilities.Clear(); agentPlayer.AbilityIcons.Clear();
        foreach (int ability in kit) {
            if (ability < 0 || ability >= abilityCatalog.sprites.Count || abilityCatalog.sprites[ability].associatedGameObject == null) {
                Debug.LogError("BOPL_AGENT: unknown ability index " + ability); continue;
            }
            agentPlayer.Abilities.Add(abilityCatalog.sprites[ability].associatedGameObject);
            agentPlayer.AbilityIcons.Add(abilityCatalog.sprites[ability].sprite);
        }
    }

    // GameSessionHandler.LoadNextLevelScene: a random kit is drawn again for
    // every round, as training drew one per episode.
    public static void BeforeLevelLoad()
    {
        if (!Enabled || handle == IntPtr.Zero || agentPlayer == null || kitChoice != "random") return;
        if (!PlayerHandler.Get().PlayerList().Contains(agentPlayer)) return;
        SetKit(RlAgentCatalog.KitFor(kitDraw));
        Debug.Log("BOPL_AGENT: next round's kit " + string.Join(",", kit));
    }

    private static bool Active(out List<Player> players)
    {
        players = null;
        if (!Enabled || handle == IntPtr.Zero || GameLobby.isOnlineGame || GameLobby.isPlayingAReplay) return false;
        players = PlayerHandler.Get().PlayerList();
        return agentIndex >= 0 && agentIndex < players.Count && players[agentIndex].IsAgent;
    }

    private static void BeginRound(List<Player> players)
    {
        round++; tick = 0; episodeTicks = 0; roundLevel = GameSession.CurrentLevel();
        started = terminal = failed = resultWritten = false;
        roundSeed = unchecked(baseSeed + (uint)round);
        // The v4 rows' space flag reads the map (Host.BeginReplayRecording's level).
        try { bopl_agent_set_map(handle, (uint)roundLevel); }
        catch (EntryPointNotFoundException) { }
        bopl_agent_reset(handle, roundSeed);
        searchFellBackLogged = false;
        decisionMs.Clear(); otherMs.Clear(); frameMs.Clear(); lastFrame = -1;
        RlObservation.Reset();
        RlObservation.RebaseRound(players);
        applied.Clear();
        agentPacket = default;
        shadowStatus = ShadowOff; shadowReport = ""; shadowRecording = null;
        if (Environment.GetEnvironmentVariable("BOPL_RL_ABILITY_SPAWNS") != "1")
            foreach (var spawner in UnityEngine.Object.FindObjectsOfType<AbilitySpawner>(true)) spawner.enabled = false;
        humansFrozen = true;
        Debug.Log("BOPL_AGENT: round " + round + " level " + roundLevel + " seed " + roundSeed);
    }

    // Updater.Update, immediately before TickSimulation.
    public static void BeforeTick(Updater current)
    {
        if (!Active(out var players)) return;
        // Every level scene carries its own SimUpdater: a new instance is a new round.
        if (round < 0 || !ReferenceEquals(current, updater)) { updater = current; BeginRound(players); }
        InputPacket zero = new InputPacket { seqNumber = (uint)(Updater.SimulationTicks + 1) };
        if (tick < warmupTicks) {
            // Training warmup: nobody moves.
            for (int i = 0; i < players.Count; i++) { players[i].OverrideInputWithNetworkInput(zero); applied[players[i].Id] = zero; }
            agentPacket = zero;
            return;
        }
        if (humansFrozen) {
            for (int i = 0; i < players.Count; i++) if (!players[i].IsAgent) players[i].RestoreLocalInputs();
            applied.Clear();
            humansFrozen = false;
        }
        var agent = players[agentIndex];
        InputPacket packet = zero;
        bool agentDecides = !terminal && !failed;
        if (agentDecides || shadowStatus == ShadowTracking) {
            try {
                observation.SetLength(0);
                uint flags = (terminal ? 1u : 0u) | (started ? 4u : 0u);
                long simulationTime = (long)tick * GameTime.FixedTimeStep.m_rawValue;
                RlObservation.Write(writer, RlBridge.Version, episodeTicks, flags, warmupTicks, simulationTime);
                ShadowCheck();
                if (agentDecides) {
                    int decisionsBefore = bopl_agent_decisions(handle);
                    tickWatch.Restart();
                    int status = bopl_agent_tick_bytes(handle, observation.GetBuffer(), (UIntPtr)observation.Length, out Action action);
                    tickWatch.Stop();
                    double ms = tickWatch.Elapsed.TotalMilliseconds;
                    (bopl_agent_decisions(handle) != decisionsBefore ? decisionMs : otherMs).Add(ms);
                    if (Time.frameCount != lastFrame) { if (lastFrame >= 0) frameMs.Add(Time.unscaledDeltaTime * 1000.0); lastFrame = Time.frameCount; }
                    if (searchOn && !searchFellBackLogged && bopl_agent_search_active(handle) == 0) {
                        searchFellBackLogged = true;
                        Debug.LogWarning("BOPL_AGENT_SEARCH: round " + round + " level " + roundLevel + ": " + Marshal.PtrToStringAnsi(bopl_agent_search_note(handle)));
                    }
                    if (status == 0) packet = ReplayArchive.Input(zero.seqNumber, action.angle, action.buttons);
                    else if (status == 1) terminal = true;
                    else { failed = true; Debug.LogError("BOPL_AGENT: shim tick failed (" + status + ") at episode tick " + episodeTicks); }
                }
            } catch (Exception exception) {
                failed = true; Debug.LogError("BOPL_AGENT: " + exception);
            }
        }
        agentPacket = packet;
        applied[agent.Id] = packet;
        agent.OverrideInputWithNetworkInput(packet);
    }

    // Updater.Update, immediately after TickSimulation.
    public static void AfterTick()
    {
        if (!Active(out var players)) return;
        tick++;
        ShadowAfterTick(players);
        if (tick <= warmupTicks) return;
        episodeTicks++;
        started |= !GameSessionHandler.HasGameEnded();
        bool ended = started && GameSessionHandler.HasGameEnded();
        if (ended && !terminal) {
            // Hand the terminal state to the shim so it closes its decision like training did.
            try {
                observation.SetLength(0);
                long simulationTime = (long)tick * GameTime.FixedTimeStep.m_rawValue;
                RlObservation.Write(writer, RlBridge.Version, episodeTicks, 1u | 4u, warmupTicks, simulationTime);
                bopl_agent_tick_bytes(handle, observation.GetBuffer(), (UIntPtr)observation.Length, out Action _);
                ShadowCheck();
            } catch (Exception exception) { Debug.LogError("BOPL_AGENT: " + exception); }
            terminal = true;
            if (shadowStatus == ShadowTracking) ShadowSettle(bopl_shadow_end(shadow));
        }
        if (terminal && !resultWritten) { resultWritten = true; LogTiming(); WriteResult(players); }
    }

    private static double Percentile(List<double> values, double q)
    {
        if (values.Count == 0) return 0;
        var sorted = new List<double>(values); sorted.Sort();
        return sorted[Math.Min(sorted.Count - 1, (int)Math.Floor(q * (sorted.Count - 1) + 0.5))];
    }

    private static double Mean(List<double> values)
    {
        double sum = 0; foreach (double v in values) sum += v;
        return values.Count == 0 ? 0 : sum / values.Count;
    }

    private static double Max(List<double> values)
    {
        double max = 0; foreach (double v in values) if (v > max) max = v;
        return max;
    }

    private static int Over(List<double> values, double limit)
    {
        int n = 0; foreach (double v in values) if (v > limit) n++;
        return n;
    }

    private static string Ms(double ms) => ms.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture);

    // One line per round: the shim tick's wall time on decision ticks (search
    // included) and on the rest, the frame times while the agent played, and
    // the search's own counters.
    private static void LogTiming()
    {
        string line = "BOPL_AGENT_TIMING: round " + round + " level " + roundLevel
            + ": decision ticks " + decisionMs.Count + " mean " + Ms(Mean(decisionMs)) + " p50 " + Ms(Percentile(decisionMs, 0.5))
            + " p95 " + Ms(Percentile(decisionMs, 0.95)) + " p99 " + Ms(Percentile(decisionMs, 0.99)) + " max " + Ms(Max(decisionMs)) + " ms"
            + "; other ticks mean " + Ms(Mean(otherMs)) + " max " + Ms(Max(otherMs)) + " ms"
            + "; frames " + frameMs.Count + " p50 " + Ms(Percentile(frameMs, 0.5)) + " p95 " + Ms(Percentile(frameMs, 0.95))
            + " p99 " + Ms(Percentile(frameMs, 0.99)) + " max " + Ms(Max(frameMs)) + " ms, " + Over(frameMs, 1000.0 / 30.0) + " over 33 ms";
        if (searchOn && bopl_agent_search_stats(handle, searchStats, SearchStats) == SearchStats)
            line += "; search " + searchStats[1] + "/" + searchStats[0] + " decisions searched, " + searchStats[2] + " raw, search p50 "
                + Ms(searchStats[5] * 1000) + " p95 " + Ms(searchStats[6] * 1000) + " max " + Ms(searchStats[4] * 1000) + " ms"
                + (searchStats[12] >= 0 ? ", raw from episode tick " + searchStats[12] + ": " + Marshal.PtrToStringAnsi(bopl_agent_search_note(handle)) : "");
        Debug.Log(line);
    }

    private static JObject TimingJson()
    {
        var timing = new JObject {
            ["decision_ticks"] = decisionMs.Count, ["decision_ms_mean"] = Mean(decisionMs), ["decision_ms_p50"] = Percentile(decisionMs, 0.5),
            ["decision_ms_p95"] = Percentile(decisionMs, 0.95), ["decision_ms_p99"] = Percentile(decisionMs, 0.99), ["decision_ms_max"] = Max(decisionMs),
            ["other_ms_mean"] = Mean(otherMs), ["other_ms_max"] = Max(otherMs),
            ["frames"] = frameMs.Count, ["frame_ms_p50"] = Percentile(frameMs, 0.5), ["frame_ms_p95"] = Percentile(frameMs, 0.95),
            ["frame_ms_p99"] = Percentile(frameMs, 0.99), ["frame_ms_max"] = Max(frameMs), ["frames_over_33ms"] = Over(frameMs, 1000.0 / 30.0)
        };
        return timing;
    }

    private static JToken SearchJson()
    {
        if (!searchOn || bopl_agent_search_stats(handle, searchStats, SearchStats) != SearchStats) return null;
        return new JObject {
            ["budget"] = searchBudget, ["decisions"] = searchStats[0], ["searched"] = searchStats[1], ["raw"] = searchStats[2],
            ["seconds"] = searchStats[3], ["max_ms"] = searchStats[4] * 1000, ["p50_ms"] = searchStats[5] * 1000, ["p95_ms"] = searchStats[6] * 1000,
            ["p99_ms"] = searchStats[7] * 1000, ["rollouts"] = searchStats[8], ["rollout_steps"] = searchStats[9], ["forward_rows"] = searchStats[10],
            ["root_searched"] = searchStats[11], ["raw_from_episode_tick"] = searchStats[12] >= 0 ? (JToken)searchStats[12] : null,
            ["note"] = Marshal.PtrToStringAnsi(bopl_agent_search_note(handle))
        };
    }

    // Shadow, after level tick `tick` ran: begin from the round's recording after
    // tick 1, confirm the warmup ticks were all-zero, then step the shadow with
    // every player's packet of each episode tick (Recording.frames[tick - 1] is
    // this tick's applied packets, PlayerList order, as the replay holds them).
    private static void ShadowAfterTick(List<Player> players)
    {
        if (shadow == IntPtr.Zero) return;
        if (tick == 1) { ShadowBegin(players); return; }
        if (shadowStatus != ShadowTracking) return;
        var recording = shadowRecording;
        if (recording.frames.Count != tick) {
            ShadowStop(ShadowFailed, "the round's replay recording stopped (" + recording.frames.Count + " frames at level tick " + tick
                       + (ReferenceEquals(recording, ReplaySession.Recording) ? "" : ", recording closed") + ")");
            return;
        }
        var frame = recording.frames[tick - 1];
        if (frame.Length != players.Count) { ShadowStop(ShadowFailed, "recorded frame has " + frame.Length + " packets for " + players.Count + " players"); return; }
        if (tick <= warmupTicks) {
            for (int i = 0; i < frame.Length; i++)
                if (frame[i].joystickAngle != 0 || ReplayArchive.Buttons(frame[i]) != 0) {
                    ShadowStop(ShadowFailed, "warmup tick " + tick + " applied a non-zero packet for player " + players[i].Id + " (the shadow warms up with zero inputs)");
                    return;
                }
            return;
        }
        if (shadowPackets.Length != frame.Length) shadowPackets = new Action[frame.Length];
        for (int i = 0; i < frame.Length; i++)
            shadowPackets[i] = new Action { angle = frame[i].joystickAngle, reserved = 0, buttons = ReplayArchive.Buttons(frame[i]) };
        ShadowSettle(bopl_shadow_step(shadow, shadowPackets, (UIntPtr)shadowPackets.Length));
    }

    private static void ShadowBegin(List<Player> players)
    {
        var recording = ReplaySession.Recording;
        if (warmupTicks < 1) { ShadowStop(ShadowFailed, "the agent has no warmup; the shadow needs one tick of recording before the episode"); return; }
        if (recording == null || recording.frames.Count != 1 || recording.players.Length != players.Count) {
            ShadowStop(ShadowFailed, "no replay recording for this round (Host.recordReplay off?)"); return;
        }
        for (int i = 0; i < players.Count; i++)
            if (recording.players[i].id != players[i].Id) { ShadowStop(ShadowFailed, "recording player order differs from PlayerList"); return; }
        shadowRecording = recording;
        byte[] archive;
        try { archive = recording.Encode(); }
        catch (Exception exception) { ShadowStop(ShadowFailed, "round setup: " + exception.Message); return; }
        int status = bopl_shadow_begin(shadow, archive, (UIntPtr)archive.Length, (uint)warmupTicks);
        shadowReport = Marshal.PtrToStringAnsi(bopl_shadow_report(shadow));
        shadowStatus = status;
        if (status == ShadowTracking) Debug.Log("BOPL_AGENT_SHADOW: round " + round + ": " + shadowReport);
        else Debug.LogError("BOPL_AGENT_SHADOW: round " + round + ": " + shadowReport);
    }

    private static void ShadowCheck()
    {
        if (shadowStatus != ShadowTracking) return;
        int status = bopl_shadow_check_bytes(shadow, observation.GetBuffer(), (UIntPtr)observation.Length);
        if (status == -2) { ShadowStop(ShadowFailed, "malformed observation record at episode tick " + episodeTicks); return; }
        ShadowSettle(status);
    }

    // A status from the shim: log the round's first change away from tracking.
    private static void ShadowSettle(int status)
    {
        if (status == shadowStatus) return;
        shadowStatus = status;
        shadowReport = Marshal.PtrToStringAnsi(bopl_shadow_report(shadow));
        string line = "BOPL_AGENT_SHADOW: round " + round + " level " + roundLevel + " " + ShadowNames[Math.Max(0, Math.Min(4, status))] + ": " + shadowReport;
        if (status == ShadowEnded) Debug.Log(line); else Debug.LogError(line);
    }

    private static void ShadowStop(int status, string reason)
    {
        shadowStatus = status;
        shadowReport = reason;
        Debug.LogError("BOPL_AGENT_SHADOW: round " + round + " level " + roundLevel + " " + ShadowNames[status] + ": " + reason);
    }

    // Host.CurrentInputAsPacket: the packet actually applied this tick for
    // players RlAgent overrides (the agent always; everyone during the warmup).
    public static bool TryAppliedInput(int playerId, uint sequence, out InputPacket packet)
    {
        packet = default;
        if (!Enabled || handle == IntPtr.Zero || !applied.TryGetValue(playerId, out packet)) return false;
        packet.seqNumber = sequence;
        return true;
    }

    private static void WriteResult(List<Player> players)
    {
        if (string.IsNullOrEmpty(resultsPath)) return;
        try {
            var catalog = SteamManager.instance.abilityIcons.sprites;
            var roster = new JArray();
            Player winner = null; int alive = 0;
            foreach (var player in players) {
                if (player.WonThisRound) winner = player;
                if (player.IsAlive) alive++;
                var loadout = new JArray();
                foreach (var ability in player.Abilities) loadout.Add(catalog.FindIndex(entry => entry.associatedGameObject == ability));
                roster.Add(new JObject { ["id"] = player.Id, ["team"] = player.Team, ["agent"] = player.IsAgent,
                    ["keyboard"] = player.UsesKeyboardAndMouse, ["abilities"] = loadout,
                    ["kills"] = player.Kills, ["deaths"] = player.Deaths, ["won_round"] = player.WonThisRound,
                    ["games_won"] = player.GamesWon, ["cause_of_death"] = (int)player.CauseOfDeath });
            }
            var line = new JObject {
                ["kind"] = "round", ["round"] = round, ["level"] = roundLevel, ["seed"] = roundSeed,
                ["agent"] = Spec, ["agent_id"] = id, ["bot"] = bot?.Id, ["bot_kind"] = bot?.Kind, ["kit_choice"] = kitChoice,
                ["kit"] = new JArray(kit[0], kit[1], kit[2]), ["bot_rating"] = bot?.Rating, ["agent_player"] = players[agentIndex].Id, ["agent_index"] = agentIndex,
                ["players"] = roster, ["winner"] = winner == null ? null : (JToken)winner.Id, ["draw"] = winner == null,
                ["agent_won"] = winner != null && winner.IsAgent,
                ["episode_ticks"] = episodeTicks, ["level_ticks"] = tick, ["warmup_ticks"] = warmupTicks,
                ["max_episode_ticks"] = maxEpisodeTicks, ["decisions"] = bopl_agent_decisions(handle),
                ["replay"] = Host.LastSavedReplay, ["replay_number"] = Host.replaysSaved,
                ["stocks"] = Settings.Get().Stocks, ["game_mode"] = (int)Settings.Get().GameMode,
                ["source_hash"] = ReplaySession.SourceHash, ["catalog_hash"] = ReplaySession.CatalogHash,
                ["failed"] = failed, ["time"] = DateTime.UtcNow.ToString("o"),
                ["shadow"] = shadow == IntPtr.Zero ? null : ShadowNames[shadowStatus],
                ["shadow_compared"] = shadow == IntPtr.Zero ? 0u : bopl_shadow_compared(shadow),
                ["shadow_report"] = shadow == IntPtr.Zero ? null : shadowReport.Split('\n')[0],
                ["hitstop_clock"] = stockHitstopClock ? "stock" : "frozen",
                ["timing"] = TimingJson(), ["search"] = SearchJson()
            };
            File.AppendAllText(resultsPath, line.ToString(Newtonsoft.Json.Formatting.None) + "\n");
        } catch (Exception exception) { Debug.LogError("BOPL_AGENT: result log: " + exception); }
    }
}
