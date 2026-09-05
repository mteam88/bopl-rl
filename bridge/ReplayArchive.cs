using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

// The network protocol remains independent of the local replay archive.
public sealed class ReplayArchive
{
    public const int MaxPlayers = 6;
    public const int MaxFrames = 2160000;
    private static readonly byte[] Magic = Encoding.ASCII.GetBytes("BOPLREP2");
    public string sourceHash;
    public string catalogHash;
    public uint seed;
    public byte level;
    public byte gameMode;
    public int stocks;
    public ReplayPlayer[] players;
    public List<InputPacket[]> frames = new List<InputPacket[]>();

    public static bool IsArchive(byte[] bytes) => bytes.Length >= Magic.Length && bytes.Take(Magic.Length).SequenceEqual(Magic);

    public void Validate()
    {
        bool Hash(string value) => value != null && value.Length == 64 && value.All(c => (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
        if (!Hash(sourceHash) || !Hash(catalogHash)) throw new InvalidDataException("Replay requires source and ability catalog SHA256 hashes");
        if (players == null || players.Length < 1 || players.Length > MaxPlayers) throw new InvalidDataException("Replay needs one through six players");
        if (players.Any(p => p == null) || players.Select(p => p.id).Distinct().Count() != players.Length) throw new InvalidDataException("Replay player IDs must be unique");
        if (stocks < 0 || gameMode > 3) throw new InvalidDataException("Invalid replay settings");
        foreach (var player in players)
        {
            if (player.id < 1 || player.id > MaxPlayers || player.team >= MaxPlayers) throw new InvalidDataException("Invalid replay player ID or team");
            if (player.abilities == null || player.abilities.Length < 1 || player.abilities.Length > 3) throw new InvalidDataException("Replay needs one through three abilities");
            if (player.abilities.Length != players[0].abilities.Length) throw new InvalidDataException("All players must have the same ability slot count");
        }
        if (frames.Count < 1 || frames.Count > MaxFrames) throw new InvalidDataException("Invalid replay frame count");
        foreach (var frame in frames)
            if (frame == null || frame.Length != players.Length) throw new InvalidDataException("Every frame must contain every player's input");
    }

    public byte[] Encode()
    {
        Validate();
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.UTF8, true);
        writer.Write(Magic);
        writer.Write(Encoding.ASCII.GetBytes(sourceHash));
        writer.Write(Encoding.ASCII.GetBytes(catalogHash));
        writer.Write(seed); writer.Write(level); writer.Write(gameMode); writer.Write(stocks);
        writer.Write((byte)players.Length);
        foreach (var player in players)
        {
            writer.Write(player.id); writer.Write(player.color); writer.Write(player.team);
            writer.Write(player.keyboard); writer.Write((byte)player.abilities.Length);
            foreach (ushort ability in player.abilities) writer.Write(ability);
        }
        writer.Write(frames.Count);
        foreach (var frame in frames)
            foreach (var input in frame)
            {
                writer.Write(input.seqNumber); writer.Write(input.joystickAngle);
                writer.Write(Buttons(input)); writer.Write(input.targetDelayBufferSize);
            }
        writer.Flush();
        using var sha = SHA256.Create();
        var digest = sha.ComputeHash(stream.ToArray());
        writer.Write(digest); writer.Flush();
        return stream.ToArray();
    }

    public static ReplayArchive Decode(byte[] bytes)
    {
        if (!IsArchive(bytes)) throw new InvalidDataException("Not a BOPLREP2 archive");
        if (bytes.Length < 184 || bytes.Length > 104000000) throw new InvalidDataException("Invalid replay byte length");
        using var sha = SHA256.Create();
        var digest = sha.ComputeHash(bytes, 0, bytes.Length - 32);
        if (!digest.SequenceEqual(bytes.Skip(bytes.Length - 32))) throw new InvalidDataException("Replay checksum mismatch");
        using var stream = new MemoryStream(bytes, 0, bytes.Length - 32, false);
        using var reader = new BinaryReader(stream, Encoding.UTF8);
        reader.ReadBytes(8);
        var result = new ReplayArchive {
            sourceHash = Encoding.ASCII.GetString(reader.ReadBytes(64)), catalogHash = Encoding.ASCII.GetString(reader.ReadBytes(64)),
            seed = reader.ReadUInt32(), level = reader.ReadByte(), gameMode = reader.ReadByte(), stocks = reader.ReadInt32()
        };
        int count = reader.ReadByte();
        if (count < 1 || count > MaxPlayers) throw new InvalidDataException("Invalid replay player count");
        result.players = new ReplayPlayer[count];
        for (int p = 0; p < count; p++)
        {
            var player = new ReplayPlayer { id = reader.ReadByte(), color = reader.ReadByte(), team = reader.ReadByte() };
            byte keyboard = reader.ReadByte();
            if (keyboard > 1) throw new InvalidDataException("Invalid keyboard flag");
            player.keyboard = keyboard == 1;
            int slots = reader.ReadByte();
            if (slots < 1 || slots > 3) throw new InvalidDataException("Invalid replay ability count");
            player.abilities = new ushort[slots];
            for (int a = 0; a < slots; a++) player.abilities[a] = reader.ReadUInt16();
            result.players[p] = player;
        }
        int ticks = reader.ReadInt32();
        if (ticks < 1 || ticks > MaxFrames || (long)ticks * count * 8 != stream.Length - stream.Position)
            throw new InvalidDataException("Replay frame length mismatch");
        for (int tick = 0; tick < ticks; tick++)
        {
            var frame = new InputPacket[count];
            for (int p = 0; p < count; p++)
            {
                uint seq = reader.ReadUInt32(); byte angle = reader.ReadByte(); ushort buttons = reader.ReadUInt16();
                frame[p] = Input(seq, angle, buttons, reader.ReadByte());
            }
            result.frames.Add(frame);
        }
        result.Validate();
        return result;
    }

    public static ushort Buttons(InputPacket p) => (ushort)((p.jump ? 1 : 0) | (p.ab1 ? 2 : 0) | (p.ab2 ? 4 : 0) | (p.ab3 ? 8 : 0) |
        (p.start ? 16 : 0) | (p.select ? 32 : 0) | (p.w ? 64 : 0) | (p.a ? 128 : 0) | (p.s ? 256 : 0) | (p.d ? 512 : 0));

    public static InputPacket Input(uint seq, byte angle, ushort buttons, byte delay = 0)
    {
        if ((buttons & ~1023) != 0) throw new InvalidDataException("Unknown replay input bits");
        return new InputPacket { seqNumber = seq, joystickAngle = angle, targetDelayBufferSize = delay,
            jump = (buttons & 1) != 0, ab1 = (buttons & 2) != 0, ab2 = (buttons & 4) != 0, ab3 = (buttons & 8) != 0,
            start = (buttons & 16) != 0, select = (buttons & 32) != 0, w = (buttons & 64) != 0,
            a = (buttons & 128) != 0, s = (buttons & 256) != 0, d = (buttons & 512) != 0 };
    }
}

public sealed class ReplayPlayer
{
    public byte id;
    public byte color;
    public byte team;
    public bool keyboard;
    public ushort[] abilities;
}
