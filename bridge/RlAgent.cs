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
// With BOPL_AGENT_CATALOG set (the launchers set it), the opponents and their
// kits come from the in-game picker instead (RlAgentPicker on the character
// select, RlAgentCatalog for the bots and the remembered seats; BOPL_AGENT then
// only preselects one): every seated bot is its own shim handle and player,
// on the team its seat chose (a human's makes it an ally, another bot's teams
// them up; unset, the latest team nobody is on), and plays its chosen kit, or
// a kit drawn for every round (from its set, or from its pool for a net
// trained on random kits). The shim needs no loadout of its own for that: the
// observation and mask read the abilities from the game state, and every
// player's team, so the rows mark teammates as allies and the primary enemy
// is the nearest player on another team. The nets trained 1v1, so with two or
// more bots, or a bot on a human's team, the match is off what they trained
// on (a teammate was never seen); the scripted bots target the first other
// living player whatever its team (train/bopl_bots.h bot_view). With neither
// variable set the stock game is untouched.
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
    public static bool Enabled => RlAgentCatalog.Present || !string.IsNullOrEmpty(envSpec);
    private static readonly string resultsPath = Environment.GetEnvironmentVariable("BOPL_AGENT_RESULTS");

    // One per AI player of the match, in PlayerList order (they join after the humans).
    private sealed class Agent
    {
        public IntPtr Handle;
        public string Id = "", Spec, KitChoice;
        public RlAgentCatalog.Seat Seat;     // null: BOPL_AGENT without the catalog
        public int[] Abilities;              // the shim's loadout; the kit played is Kit
        public int[] Kit;
        public Player Player;
        public int Index, MaxEpisodeTicks;
        public bool Terminal, Failed;
    }

    private static readonly List<Agent> agents = new List<Agent>();
    // The first agent's handle: the shadow and the search attach to it alone.
    private static IntPtr handle => agents.Count > 0 ? agents[0].Handle : IntPtr.Zero;
    private static System.Random kitDraw;
    private static NamedSpriteList abilityCatalog;
    private static int warmupTicks;
    private static string openError;

    // Round state.
    private static Updater updater;          // the level scene's Updater; a new one is a new round
    private static int round = -1, tick, episodeTicks, roundLevel;
    private static uint baseSeed, roundSeed;
    private static bool started, humansFrozen, resultWritten;
    private static readonly MemoryStream observation = new MemoryStream(65536);
    private static readonly BinaryWriter writer = new BinaryWriter(observation, Encoding.UTF8, true);
    private static readonly Dictionary<int, InputPacket> applied = new Dictionary<int, InputPacket>();

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

    public static string Id => agents.Count > 0 ? agents[0].Id : "";
    public static int WarmupTicks => warmupTicks;

    // Opens one agent of the match (players and its index fixed for the whole match).
    private static Agent Open(string spec, int players, int player)
    {
        var error = new byte[1024];
        IntPtr opened;
        try {
            if (bopl_agent_abi_version() != 1) { openError = "libbopl_agent.so ABI mismatch"; return null; }
            opened = bopl_agent_open(spec, players, player, error, (UIntPtr)error.Length);
        } catch (DllNotFoundException exception) {
            openError = "libbopl_agent.so not found (Tools/play-vs-agent.sh copies it into BoplBattle_Data/Plugins/x86_64): " + exception.Message;
            return null;
        }
        if (opened == IntPtr.Zero) {
            openError = Encoding.UTF8.GetString(error).TrimEnd('\0');
            return null;
        }
        return new Agent {
            Handle = opened, Spec = spec, Index = player, Id = Marshal.PtrToStringAnsi(bopl_agent_id(opened)),
            Abilities = new[] { bopl_agent_ability(opened, 0), bopl_agent_ability(opened, 1), bopl_agent_ability(opened, 2) },
            MaxEpisodeTicks = bopl_agent_max_episode_ticks(opened)
        };
    }

    private static void CloseAll()
    {
        foreach (var agent in agents) if (agent.Handle != IntPtr.Zero) bopl_agent_close(agent.Handle);
        agents.Clear();
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

    // CharacterSelectHandler.TryStartGame_inner: append the AI players after the
    // human ones, one per picker seat (RlAgentCatalog.Seats; BOPL_AGENT alone is
    // one). Each gets its seat's team (unset: the latest team no human and no
    // earlier bot is on, RlAgentCatalog.LaterTeam) and a colour nobody else
    // took; the loadout is the seat's kit (RlAgentCatalog.KitFor), else the
    // agent's training loadout (sidecar ability1..3, or the training defaults).
    public static void AddPlayers(List<Player> list, ref int nextId, PlayerColors colors, NamedSpriteList catalog)
    {
        if (!Enabled) return;
        CloseAll();
        updater = null;
        var seats = new List<RlAgentCatalog.Seat>();
        if (RlAgentCatalog.Present) {
            RlAgentCatalog.EnsureLoaded();
            seats.AddRange(RlAgentCatalog.Seats);
            if (seats.Count == 0) { Debug.Log("BOPL_AGENT: no bot seated (" + (RlAgentCatalog.LoadError ?? "a match without bots") + ")"); return; }
        } else seats.Add(null);
        int room = Constants.MAX_PLAYERS - list.Count;
        if (seats.Count > room) {
            Debug.LogWarning("BOPL_AGENT: " + seats.Count + " bots for " + room + " free player slots; the last " + (seats.Count - room) + " sit out");
            seats.RemoveRange(Math.Max(0, room), seats.Count - Math.Max(0, room));
        }
        // Every shim is opened for the final player count: a bot that fails to
        // open (or warms up differently from the first) drops out and the rest
        // are opened again for the smaller match.
        while (seats.Count > 0) {
            int failed = -1;
            for (int i = 0; i < seats.Count && failed < 0; i++) {
                string spec = seats[i]?.Bot.Path ?? envSpec;
                var agent = Open(spec, list.Count + seats.Count, list.Count + i);
                if (agent == null) { Debug.LogError("BOPL_AGENT=" + spec + ": " + openError); failed = i; break; }
                agent.Seat = seats[i];
                agents.Add(agent);
                int warmup = bopl_agent_warmup_ticks(agent.Handle);
                if (i == 0) warmupTicks = warmup;
                else if (warmup != warmupTicks) {
                    Debug.LogError("BOPL_AGENT=" + spec + ": warmup " + warmup + " ticks, the first bot's is " + warmupTicks + " (every player warms up together); it sits out");
                    failed = i;
                }
            }
            if (failed < 0) break;
            CloseAll();
            seats.RemoveAt(failed);
        }
        if (agents.Count == 0) return;
        OpenShadow();
        AttachSearch();
        // One seed sequence per launch: a later match (after the picker changed the
        // bot) continues the round count, so no two rounds of a session share a seed.
        if (kitDraw == null) {
            string seed = Environment.GetEnvironmentVariable("BOPL_AGENT_SEED");
            baseSeed = !string.IsNullOrEmpty(seed) && uint.TryParse(seed, out uint parsed) ? parsed : (uint)Environment.TickCount;
            kitDraw = new System.Random(unchecked((int)baseSeed));
        }
        abilityCatalog = catalog;
        var taken = list.ConvertAll(p => p.Color);
        var teams = list.ConvertAll(p => p.Team);
        foreach (var agent in agents) {
            int team = agent.Seat != null && RlAgentCatalog.ValidTeam(agent.Seat.Team) ? agent.Seat.Team : RlAgentCatalog.LaterTeam(teams);
            teams.Add(team);
            int color = ColorIndex(taken, colors);
            taken.Add(colors[color].playerMaterial);
            agent.Player = new Player(nextId, team) {
                Color = colors[color].playerMaterial, UsesKeyboardAndMouse = false, CanUseAbilities = true,
                IsLocalPlayer = false, IsAgent = true, inputDevice = null,
                Abilities = new List<GameObject>(3), AbilityIcons = new List<Sprite>(3)
            };
            agent.KitChoice = RlAgentCatalog.KitChoiceText(agent.Seat);
            SetKit(agent, RlAgentCatalog.KitFor(agent.Seat, kitDraw) ?? agent.Abilities);
            nextId++;
            list.Add(agent.Player);
            Debug.Log("BOPL_AGENT: " + agent.Id + (agent.Seat == null ? "" : " (" + agent.Seat.Bot.Id + ", kit " + agent.KitChoice + ")") + " joins as player "
                      + agent.Player.Id + " (team " + team + ", abilities " + string.Join(",", agent.Kit) + "), base seed " + baseSeed);
        }
    }

    // Gives the agent's player this kit (ability catalog indices).
    private static void SetKit(Agent agent, int[] next)
    {
        agent.Kit = next;
        var player = agent.Player;
        player.Abilities.Clear(); player.AbilityIcons.Clear();
        foreach (int ability in next) {
            if (ability < 0 || ability >= abilityCatalog.sprites.Count || abilityCatalog.sprites[ability].associatedGameObject == null) {
                Debug.LogError("BOPL_AGENT: unknown ability index " + ability); continue;
            }
            player.Abilities.Add(abilityCatalog.sprites[ability].associatedGameObject);
            player.AbilityIcons.Add(abilityCatalog.sprites[ability].sprite);
        }
    }

    // GameSessionHandler.LoadNextLevelScene: a random kit is drawn again for
    // every round, as training drew one per episode.
    public static void BeforeLevelLoad()
    {
        if (!Enabled) return;
        var list = PlayerHandler.Get().PlayerList();
        foreach (var agent in agents) {
            if (agent.KitChoice != "random" || !list.Contains(agent.Player)) continue;
            SetKit(agent, RlAgentCatalog.KitFor(agent.Seat, kitDraw));
            Debug.Log("BOPL_AGENT: next round's kit for player " + agent.Player.Id + ": " + string.Join(",", agent.Kit));
        }
    }

    private static bool Active(out List<Player> players)
    {
        players = null;
        if (!Enabled || agents.Count == 0 || GameLobby.isOnlineGame || GameLobby.isPlayingAReplay) return false;
        players = PlayerHandler.Get().PlayerList();
        foreach (var agent in agents)
            if (agent.Index >= players.Count || !ReferenceEquals(players[agent.Index], agent.Player)) return false;
        return true;
    }

    private static void BeginRound(List<Player> players)
    {
        round++; tick = 0; episodeTicks = 0; roundLevel = GameSession.CurrentLevel();
        started = resultWritten = false;
        roundSeed = unchecked(baseSeed + (uint)round);
        for (int i = 0; i < agents.Count; i++) {
            var agent = agents[i];
            agent.Terminal = agent.Failed = false;
            // The v4 rows' space flag reads the map (Host.BeginReplayRecording's level).
            try { bopl_agent_set_map(agent.Handle, (uint)roundLevel); }
            catch (EntryPointNotFoundException) { }
            // The first bot on the round seed (as with one bot); the others on
            // their own streams, so two copies of a net do not sample alike.
            bopl_agent_reset(agent.Handle, unchecked(roundSeed + (uint)i * 0x9E3779B9u));
        }
        searchFellBackLogged = false;
        decisionMs.Clear(); otherMs.Clear(); frameMs.Clear(); lastFrame = -1;
        RlObservation.Reset();
        RlObservation.RebaseRound(players);
        applied.Clear();
        shadowStatus = ShadowOff; shadowReport = ""; shadowRecording = null;
        if (Environment.GetEnvironmentVariable("BOPL_RL_ABILITY_SPAWNS") != "1")
            foreach (var spawner in UnityEngine.Object.FindObjectsOfType<AbilitySpawner>(true)) spawner.enabled = false;
        humansFrozen = true;
        Debug.Log("BOPL_AGENT: round " + round + " level " + roundLevel + " seed " + roundSeed + (agents.Count > 1 ? " (" + agents.Count + " bots)" : ""));
    }

    // The observation record of the state before this tick; terminal sets the
    // record's terminal flag (the shim closes its decision on it).
    private static void WriteObservation(bool terminal)
    {
        observation.SetLength(0);
        uint flags = (terminal ? 1u : 0u) | (started ? 4u : 0u);
        long simulationTime = (long)tick * GameTime.FixedTimeStep.m_rawValue;
        RlObservation.Write(writer, RlBridge.Version, episodeTicks, flags, warmupTicks, simulationTime);
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
            return;
        }
        if (humansFrozen) {
            for (int i = 0; i < players.Count; i++) if (!players[i].IsAgent) players[i].RestoreLocalInputs();
            applied.Clear();
            humansFrozen = false;
        }
        bool anyDecides = agents.Exists(a => !a.Terminal && !a.Failed);
        bool written = false, terminalBit = agents[0].Terminal;
        if (anyDecides || shadowStatus == ShadowTracking) {
            try {
                // The shadow compares the record as the first bot's shim sees it.
                WriteObservation(terminalBit);
                written = true;
                if (episodeTicks == 0) Debug.Log("BOPL_AGENT: round " + round + " first row kills/deaths " + RlObservation.RowCounts(players));
                ShadowCheck();
            } catch (Exception exception) {
                written = false;
                foreach (var agent in agents) agent.Failed = true;
                Debug.LogError("BOPL_AGENT: " + exception);
            }
        }
        if (Time.frameCount != lastFrame && anyDecides) { if (lastFrame >= 0) frameMs.Add(Time.unscaledDeltaTime * 1000.0); lastFrame = Time.frameCount; }
        foreach (var agent in agents) {
            InputPacket packet = zero;
            if (written && !agent.Terminal && !agent.Failed) {
                try {
                    if (terminalBit) { WriteObservation(false); terminalBit = false; }
                    int decisionsBefore = bopl_agent_decisions(agent.Handle);
                    tickWatch.Restart();
                    int status = bopl_agent_tick_bytes(agent.Handle, observation.GetBuffer(), (UIntPtr)observation.Length, out Action action);
                    tickWatch.Stop();
                    double ms = tickWatch.Elapsed.TotalMilliseconds;
                    (bopl_agent_decisions(agent.Handle) != decisionsBefore ? decisionMs : otherMs).Add(ms);
                    if (agent == agents[0] && searchOn && !searchFellBackLogged && bopl_agent_search_active(agent.Handle) == 0) {
                        searchFellBackLogged = true;
                        Debug.LogWarning("BOPL_AGENT_SEARCH: round " + round + " level " + roundLevel + ": " + Marshal.PtrToStringAnsi(bopl_agent_search_note(agent.Handle)));
                    }
                    if (status == 0) packet = ReplayArchive.Input(zero.seqNumber, action.angle, action.buttons);
                    else if (status == 1) agent.Terminal = true;
                    else { agent.Failed = true; Debug.LogError("BOPL_AGENT: player " + agent.Player.Id + ": shim tick failed (" + status + ") at episode tick " + episodeTicks); }
                } catch (Exception exception) {
                    agent.Failed = true; Debug.LogError("BOPL_AGENT: " + exception);
                }
            }
            applied[agent.Player.Id] = packet;
            agent.Player.OverrideInputWithNetworkInput(packet);
        }
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
        if (ended && agents.Exists(a => !a.Terminal)) {
            // Hand the terminal state to the shims so they close their decision like training did.
            try {
                WriteObservation(true);
                foreach (var agent in agents)
                    if (!agent.Terminal) bopl_agent_tick_bytes(agent.Handle, observation.GetBuffer(), (UIntPtr)observation.Length, out Action _);
                ShadowCheck();
            } catch (Exception exception) { Debug.LogError("BOPL_AGENT: " + exception); }
            foreach (var agent in agents) agent.Terminal = true;
            if (shadowStatus == ShadowTracking) ShadowSettle(bopl_shadow_end(shadow));
        }
        if (!resultWritten && agents.TrueForAll(a => a.Terminal)) {
            resultWritten = true;
            Debug.Log("BOPL_AGENT: round " + round + " terminal row kills/deaths " + RlObservation.RowCounts(players) + "; won the round: "
                      + string.Join(", ", players.FindAll(p => p.WonThisRound).ConvertAll(p => "p" + p.Id + " (team " + p.Team + (p.IsAgent ? ", bot" : "") + ")")));
            LogTiming(); WriteResult(players);
        }
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

    // One line per round. The top-level agent fields are the first bot's (as
    // with one bot); "agents" lists every bot with its kit and team. The round
    // goes to a team (GameSessionHandler: the last team standing, and every
    // player on it gets WonThisRound); winner_team is that team, null when the
    // flag spans no team or several (DeclareGameUndecided on a tie), which is a
    // draw. winner is the winning team's first player (a human before the bots),
    // human_won / agent_won whether a human's / a bot's team won (both, with an
    // ally bot), and allied whether any bot shares a team with a human.
    private static void WriteResult(List<Player> players)
    {
        if (string.IsNullOrEmpty(resultsPath)) return;
        try {
            var catalog = SteamManager.instance.abilityIcons.sprites;
            var roster = new JArray();
            var winners = players.FindAll(p => p.WonThisRound);
            int winningTeam = winners.Count > 0 && winners.TrueForAll(p => p.Team == winners[0].Team) ? winners[0].Team : -1;
            Player winner = winningTeam >= 0 ? winners[0] : null;
            bool allied = agents.Exists(a => players.Exists(p => !p.IsAgent && p.Team == a.Player.Team));
            foreach (var player in players) {
                var loadout = new JArray();
                foreach (var ability in player.Abilities) loadout.Add(catalog.FindIndex(entry => entry.associatedGameObject == ability));
                roster.Add(new JObject { ["id"] = player.Id, ["team"] = player.Team, ["agent"] = player.IsAgent,
                    ["keyboard"] = player.UsesKeyboardAndMouse, ["abilities"] = loadout,
                    ["kills"] = player.Kills, ["deaths"] = player.Deaths, ["won_round"] = player.WonThisRound,
                    ["games_won"] = player.GamesWon, ["cause_of_death"] = (int)player.CauseOfDeath });
            }
            var bots = new JArray();
            foreach (var agent in agents) {
                var bot = agent.Seat?.Bot;
                bots.Add(new JObject {
                    ["agent"] = agent.Spec, ["agent_id"] = agent.Id, ["bot"] = bot?.Id, ["bot_kind"] = bot?.Kind, ["kit_choice"] = agent.KitChoice,
                    ["kit"] = new JArray(agent.Kit[0], agent.Kit[1], agent.Kit[2]), ["bot_rating"] = bot?.Rating, ["player"] = agent.Player.Id,
                    ["index"] = agent.Index, ["team"] = agent.Player.Team, ["won"] = winningTeam >= 0 && agent.Player.Team == winningTeam,
                    ["ally_of_human"] = players.Exists(p => !p.IsAgent && p.Team == agent.Player.Team),
                    ["decisions"] = bopl_agent_decisions(agent.Handle), ["max_episode_ticks"] = agent.MaxEpisodeTicks, ["failed"] = agent.Failed
                });
            }
            var first = agents[0];
            var line = new JObject {
                ["kind"] = "round", ["round"] = round, ["level"] = roundLevel, ["seed"] = roundSeed,
                ["agent"] = first.Spec, ["agent_id"] = first.Id, ["bot"] = first.Seat?.Bot.Id, ["bot_kind"] = first.Seat?.Bot.Kind, ["kit_choice"] = first.KitChoice,
                ["kit"] = new JArray(first.Kit[0], first.Kit[1], first.Kit[2]), ["bot_rating"] = first.Seat?.Bot.Rating, ["agent_player"] = first.Player.Id, ["agent_index"] = first.Index,
                ["bots"] = agents.Count, ["agents"] = bots,
                ["players"] = roster, ["winner"] = winner == null ? null : (JToken)winner.Id, ["draw"] = winner == null,
                ["winner_team"] = winningTeam >= 0 ? (JToken)winningTeam : null, ["winners"] = new JArray(winners.ConvertAll(p => (object)p.Id).ToArray()),
                ["agent_won"] = winner != null && winners.Exists(p => p.IsAgent), ["human_won"] = winner != null && winners.Exists(p => !p.IsAgent),
                ["allied"] = allied,
                ["episode_ticks"] = episodeTicks, ["level_ticks"] = tick, ["warmup_ticks"] = warmupTicks,
                ["max_episode_ticks"] = first.MaxEpisodeTicks, ["decisions"] = bopl_agent_decisions(first.Handle),
                ["replay"] = Host.LastSavedReplay, ["replay_number"] = Host.replaysSaved,
                ["stocks"] = Settings.Get().Stocks, ["game_mode"] = (int)Settings.Get().GameMode,
                ["source_hash"] = ReplaySession.SourceHash, ["catalog_hash"] = ReplaySession.CatalogHash,
                ["failed"] = agents.Exists(a => a.Failed), ["time"] = DateTime.UtcNow.ToString("o"),
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
