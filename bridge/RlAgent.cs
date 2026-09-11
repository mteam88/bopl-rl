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
// hitstop frames without simulating anything). Each round writes one JSON line
// to BOPL_AGENT_RESULTS (seed, players, winner, replay file) for
// Tools/play-vs-agent.sh. Unset BOPL_AGENT leaves the stock game untouched.
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

    public static readonly string Spec = Environment.GetEnvironmentVariable("BOPL_AGENT");
    public static bool Enabled => !string.IsNullOrEmpty(Spec);
    private static readonly string resultsPath = Environment.GetEnvironmentVariable("BOPL_AGENT_RESULTS");

    private static IntPtr handle;
    private static string id = "";
    private static int[] abilities;
    private static int warmupTicks, maxEpisodeTicks;
    private static string openError;

    // Round state.
    private static Updater updater;          // the level scene's Updater; a new one is a new round
    private static int round = -1, tick, episodeTicks;
    private static uint baseSeed, roundSeed;
    private static bool started, terminal, failed, humansFrozen, resultWritten;
    private static int agentIndex = -1;
    private static readonly MemoryStream observation = new MemoryStream(65536);
    private static readonly BinaryWriter writer = new BinaryWriter(observation, Encoding.UTF8, true);
    private static readonly Dictionary<int, InputPacket> applied = new Dictionary<int, InputPacket>();
    private static InputPacket agentPacket;

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
        return true;
    }

    // CharacterSelectHandler.TryStartGame_inner: append the AI player after the
    // human ones. Team and colour are ones no human took; the loadout is the
    // agent's training loadout (sidecar ability1..3, or the training defaults).
    public static void AddPlayers(List<Player> list, ref int nextId, PlayerColors colors, NamedSpriteList catalog)
    {
        if (!Enabled) return;
        if (!Open(list.Count + 1, list.Count)) {
            Debug.LogError("BOPL_AGENT=" + Spec + ": " + openError);
            return;
        }
        int team = 0, color = 0;
        while (list.Exists(p => p.Team == team)) team++;
        while (color < colors.playerColors.Length - 1 && list.Exists(p => p.Color == colors[color].playerMaterial)) color++;
        var player = new Player(nextId, team) {
            Color = colors[color].playerMaterial, UsesKeyboardAndMouse = false, CanUseAbilities = true,
            IsLocalPlayer = false, IsAgent = true, inputDevice = null,
            Abilities = new List<GameObject>(3), AbilityIcons = new List<Sprite>(3)
        };
        foreach (int ability in abilities) {
            if (ability < 0 || ability >= catalog.sprites.Count || catalog.sprites[ability].associatedGameObject == null) {
                Debug.LogError("BOPL_AGENT: unknown ability index " + ability); continue;
            }
            player.Abilities.Add(catalog.sprites[ability].associatedGameObject);
            player.AbilityIcons.Add(catalog.sprites[ability].sprite);
        }
        nextId++;
        list.Add(player);
        round = -1; updater = null;
        string seed = Environment.GetEnvironmentVariable("BOPL_AGENT_SEED");
        baseSeed = !string.IsNullOrEmpty(seed) && uint.TryParse(seed, out uint parsed) ? parsed : (uint)Environment.TickCount;
        Debug.Log("BOPL_AGENT: " + id + " joins as player " + player.Id + " (team " + team + ", abilities " + string.Join(",", abilities) + "), base seed " + baseSeed);
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
        round++; tick = 0; episodeTicks = 0;
        started = terminal = failed = resultWritten = false;
        roundSeed = unchecked(baseSeed + (uint)round);
        bopl_agent_reset(handle, roundSeed);
        RlObservation.Reset();
        applied.Clear();
        agentPacket = default;
        if (Environment.GetEnvironmentVariable("BOPL_RL_ABILITY_SPAWNS") != "1")
            foreach (var spawner in UnityEngine.Object.FindObjectsOfType<AbilitySpawner>(true)) spawner.enabled = false;
        humansFrozen = true;
        Debug.Log("BOPL_AGENT: round " + round + " level " + GameSession.CurrentLevel() + " seed " + roundSeed);
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
        if (!terminal && !failed) {
            try {
                observation.SetLength(0);
                uint flags = (terminal ? 1u : 0u) | (started ? 4u : 0u);
                long simulationTime = (long)tick * GameTime.FixedTimeStep.m_rawValue;
                RlObservation.Write(writer, RlBridge.Version, episodeTicks, flags, warmupTicks, simulationTime);
                int status = bopl_agent_tick_bytes(handle, observation.GetBuffer(), (UIntPtr)observation.Length, out Action action);
                if (status == 0) packet = ReplayArchive.Input(zero.seqNumber, action.angle, action.buttons);
                else if (status == 1) terminal = true;
                else { failed = true; Debug.LogError("BOPL_AGENT: shim tick failed (" + status + ") at episode tick " + episodeTicks); }
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
            } catch (Exception exception) { Debug.LogError("BOPL_AGENT: " + exception); }
            terminal = true;
        }
        if (terminal && !resultWritten) { resultWritten = true; WriteResult(players); }
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
                ["kind"] = "round", ["round"] = round, ["level"] = (int)GameSession.CurrentLevel(), ["seed"] = roundSeed,
                ["agent"] = Spec, ["agent_id"] = id, ["agent_player"] = players[agentIndex].Id, ["agent_index"] = agentIndex,
                ["players"] = roster, ["winner"] = winner == null ? null : (JToken)winner.Id, ["draw"] = winner == null,
                ["agent_won"] = winner != null && winner.IsAgent,
                ["episode_ticks"] = episodeTicks, ["level_ticks"] = tick, ["warmup_ticks"] = warmupTicks,
                ["max_episode_ticks"] = maxEpisodeTicks, ["decisions"] = bopl_agent_decisions(handle),
                ["replay"] = Host.LastSavedReplay, ["replay_number"] = Host.replaysSaved,
                ["stocks"] = Settings.Get().Stocks, ["game_mode"] = (int)Settings.Get().GameMode,
                ["source_hash"] = ReplaySession.SourceHash, ["catalog_hash"] = ReplaySession.CatalogHash,
                ["failed"] = failed, ["time"] = DateTime.UtcNow.ToString("o")
            };
            File.AppendAllText(resultsPath, line.ToString(Newtonsoft.Json.Formatting.None) + "\n");
        } catch (Exception exception) { Debug.LogError("BOPL_AGENT: result log: " + exception); }
    }
}
