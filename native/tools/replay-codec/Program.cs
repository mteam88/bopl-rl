using System;
using System.IO;
using System.Linq;
using System.Text.Json;

static class Program
{
    static readonly JsonSerializerOptions Json = new JsonSerializerOptions { IncludeFields = true, WriteIndented = true };
    static void Main(string[] args)
    {
        if (args.Length == 0) throw new ArgumentException("replay-codec encode input.json output.rep | decode input.rep output.json | check input.rep | selftest");
        switch (args[0])
        {
            case "encode":
                var archive = JsonSerializer.Deserialize<ReplayArchive>(File.ReadAllText(args[1]), Json);
                using (var output = new FileStream(args[2], FileMode.CreateNew)) { var bytes = archive.Encode(); output.Write(bytes); }
                Console.WriteLine("REPLAY_V2_ENCODED players=" + archive.players.Length + " ticks=" + archive.frames.Count);
                break;
            case "decode":
                File.WriteAllText(args[2], JsonSerializer.Serialize(ReplayArchive.Decode(File.ReadAllBytes(args[1])), Json));
                break;
            case "check":
                var replay = ReplayArchive.Decode(File.ReadAllBytes(args[1]));
                Console.WriteLine(JsonSerializer.Serialize(new { format = "BOPLREP2", replay.sourceHash, replay.catalogHash, replay.seed, replay.level, replay.stocks, replay.gameMode, replay.players, ticks = replay.frames.Count }, Json));
                break;
            case "selftest": SelfTest(); break;
            default: throw new ArgumentException("Unknown command");
        }
    }

    static void SelfTest()
    {
        int cases = 0;
        var rng = new Random(67484);
        for (int players = 1; players <= 6; players++)
        {
            var replay = new ReplayArchive { sourceHash = new string('a', 64), catalogHash = new string('b', 64),
                seed = uint.MaxValue, level = 123, stocks = 5, gameMode = 3,
                players = Enumerable.Range(1, players).Select(i => new ReplayPlayer { id = (byte)(7-i), color = (byte)(i+4), team = (byte)(i%3), keyboard = i%2==0, abilities = new ushort[] { 0, 38, ushort.MaxValue } }).ToArray() };
            for (int tick = 0; tick < 1024; tick++)
                replay.frames.Add(Enumerable.Range(0, players).Select(i => ReplayArchive.Input((uint)(tick+i), (byte)rng.Next(256), (ushort)tick, (byte)i)).ToArray());
            byte[] encoded = replay.Encode();
            var decoded = ReplayArchive.Decode(encoded);
            if (!encoded.SequenceEqual(decoded.Encode()) || JsonSerializer.Serialize(replay, Json) != JsonSerializer.Serialize(decoded, Json)) throw new Exception("Codec roundtrip differs");
            cases += players * 1024;
            for (int length = 0; length < encoded.Length; length += Math.Max(1, encoded.Length/41))
                Reject(() => ReplayArchive.Decode(encoded.Take(length).ToArray()));
            foreach (int index in new[] {0, 8, 72, 146, encoded.Length-33, encoded.Length-1})
            {
                var corrupt = (byte[])encoded.Clone(); corrupt[index] ^= 1;
                Reject(() => ReplayArchive.Decode(corrupt));
            }
            decoded.players[0].id = 0; Reject(() => decoded.Encode()); decoded.players[0].id = replay.players[0].id;
            decoded.frames[0] = new InputPacket[players-1]; Reject(() => decoded.Encode());
        }
        Reject(() => ReplayArchive.Input(0, 0, 1024));
        Console.WriteLine("REPLAY_CODEC_PASS player_inputs=" + cases + " players=1..6 buttons=0..1023 corruption_and_truncation=rejected");
    }
    static void Reject(Action operation)
    {
        try { operation(); } catch (InvalidDataException) { return; }
        throw new Exception("Malformed replay was accepted");
    }
}
