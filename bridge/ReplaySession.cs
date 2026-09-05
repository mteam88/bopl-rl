using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using UnityEngine;

public static class ReplaySession
{
    public static ReplayArchive Playing { get; private set; }
    public static ReplayArchive Recording { get; private set; }
    public static string SourceHash => Resources.Load<TextAsset>("SimulationReference").text.Trim();

    public static string CatalogHash
    {
        get
        {
            var catalog = SteamManager.instance.abilityIcons.sprites;
            string identity = string.Join("\n", catalog.Select((entry, index) => index + ":" + entry.name + ":" + (entry.associatedGameObject == null ? "" : entry.associatedGameObject.name)));
            using var sha = SHA256.Create();
            return BitConverter.ToString(sha.ComputeHash(Encoding.UTF8.GetBytes(identity))).Replace("-", "").ToLowerInvariant();
        }
    }

    public static void Load(byte[] bytes)
    {
        Playing = ReplayArchive.Decode(bytes);
        if (Playing.sourceHash != SourceHash) throw new InvalidDataException("Replay belongs to a different pinned game source/assets state");
    }

    public static void ClearPlayback() => Playing = null;
    public static void ClearRecording() => Recording = null;

    public static StartRequestPacket LegacyStart()
    {
        return new StartRequestPacket { seqNum = 1, seed = Playing.seed, currentLevel = Playing.level,
            nrOfPlayers = (byte)Playing.players.Length, nrOfAbilites = (byte)Playing.players[0].abilities.Length };
    }

    public static List<Player> CreatePlayers(PlayerColors colors)
    {
        if (Playing.catalogHash != CatalogHash) throw new InvalidDataException("Replay ability catalog does not match this game");
        var catalog = SteamManager.instance.abilityIcons.sprites;
        var result = new List<Player>();
        foreach (var saved in Playing.players)
        {
            if (saved.color >= colors.Length) throw new InvalidDataException("Unknown replay player color");
            var player = new Player { Id = saved.id, Team = saved.team, Color = colors[saved.color].playerMaterial,
                IsLocalPlayer = false, UsesKeyboardAndMouse = saved.keyboard };
            foreach (ushort ability in saved.abilities)
            {
                if (ability >= catalog.Count || catalog[ability].associatedGameObject == null) throw new InvalidDataException("Unknown replay ability " + ability);
                player.Abilities.Add(catalog[ability].associatedGameObject);
                player.AbilityIcons.Add(catalog[ability].sprite);
            }
            result.Add(player);
        }
        Settings.Get().NumberOfAbilities = Playing.players[0].abilities.Length;
        Settings.Get().Stocks = Playing.stocks;
        Settings.Get().GameMode = (GameModeSelection)Playing.gameMode;
        return result;
    }

    public static void BeginRecording(uint seed, byte level, List<Player> players, PlayerColors colors)
    {
        var catalog = SteamManager.instance.abilityIcons.sprites;
        Recording = new ReplayArchive { sourceHash = SourceHash, catalogHash = CatalogHash, seed = seed, level = level,
            stocks = Settings.Get().Stocks, gameMode = (byte)Settings.Get().GameMode,
            players = players.Select(player => {
                int color = Array.FindIndex(colors.playerColors, c => c.playerMaterial == player.Color);
                if (color < 0) throw new InvalidDataException("Player color is absent from the replay catalog");
                return new ReplayPlayer { id = checked((byte)player.Id), color = checked((byte)color), team = checked((byte)player.Team),
                    keyboard = player.UsesKeyboardAndMouse, abilities = player.Abilities.Select(ability => {
                        int index = catalog.FindIndex(entry => entry.associatedGameObject == ability);
                        if (index < 0) throw new InvalidDataException("Ability prefab is absent from the replay catalog: " + ability.name);
                        return checked((ushort)index);
                    }).ToArray() };
            }).ToArray() };
    }

    public static void RecordInputs(uint sequence)
    {
        if (Recording == null) throw new InvalidOperationException("Replay recording has no setup");
        Recording.frames.Add(Recording.players.Select(player => Host.CurrentInputAsPacket(player.id, sequence, null)).ToArray());
    }
}
