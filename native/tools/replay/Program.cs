using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Linq;
using System.Security.Cryptography;

public static class ReplayTool
{
    private static void Inspect(string directory)
    {
        foreach (var path in (File.Exists(directory)?new[]{directory}:Directory.GetFiles(directory,"*.rep")).OrderByDescending(File.GetLastWriteTimeUtc))
        {
            try {
                byte[] bytes=File.ReadAllBytes(path);
                var frames=NetworkTools.ReadCompressedReplay(bytes,out var header);
                int simultaneous=0,abilityFrames=0,jumpFrames=0;
                foreach (var f in frames) {
                    var inputs=new[]{f.p1,f.p2,f.p3,f.p4}.Take(header.nrOfPlayers);
                    int casting=inputs.Count(p=>p.ab1||p.ab2||p.ab3);
                    if(casting>1) simultaneous++;
                    if(casting>0) abilityFrames++;
                    if(inputs.Any(p=>p.jump)) jumpFrames++;
                }
                var abilities=new[]{new[]{header.p1_ability1,header.p1_ability2,header.p1_ability3},new[]{header.p2_ability1,header.p2_ability2,header.p2_ability3},new[]{header.p3_ability1,header.p3_ability2,header.p3_ability3},new[]{header.p4_ability1,header.p4_ability2,header.p4_ability3}};
                Console.WriteLine(JsonSerializer.Serialize(new {file=Path.GetFileName(path),bytes=bytes.Length,
                    sha256=Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant(),modified_utc=File.GetLastWriteTimeUtc(path),
                    seed=header.seed,level=header.currentLevel,players=header.nrOfPlayers,recorded_player_slots=4,complete_player_inputs=header.nrOfPlayers<=4,has_unresolved_ability=abilities.Take(header.nrOfPlayers).Any(a=>a.Take(header.nrOfAbilites).Contains((byte)255)),ability_slots=header.nrOfAbilites,
                    abilities=abilities.Take(header.nrOfPlayers).Select(a=>a.Take(header.nrOfAbilites)),ticks=frames.Count,
                    simultaneous_ability_frames=simultaneous,ability_frames=abilityFrames,jump_frames=jumpFrames}));
            } catch(Exception error) { Console.WriteLine(JsonSerializer.Serialize(new {file=Path.GetFileName(path),error=error.Message})); }
        }
    }

    private static void Mutate(string source,string destination,int seed)
    {
        byte[] bytes=File.ReadAllBytes(source);
        var original=NetworkTools.ReadCompressedReplay(bytes,out var header).ToArray();
        if(header.nrOfPlayers>4) throw new ArgumentException("legacy replay lacks player 5/6 inputs");
        var rng=new Random(seed);var result=new List<InputPacketQuad>();
        var held=new InputPacket[4];var remaining=new int[4];
        for(int tick=0;tick<original.Length;tick++) {
            var frame=original[tick];var inputs=new[]{frame.p1,frame.p2,frame.p3,frame.p4};
            for(int p=0;p<header.nrOfPlayers;p++) {
                if(tick<240 || tick%180<60) continue; // Keep real play between bursts.
                if(remaining[p]==0) {
                    held[p]=new InputPacket {joystickAngle=(byte)rng.Next(256),jump=rng.Next(3)==0,
                        ab1=rng.Next(3)==0,ab2=rng.Next(3)==0,ab3=rng.Next(3)==0,
                        w=rng.Next(3)==0,a=rng.Next(3)==0,s=rng.Next(3)==0,d=rng.Next(3)==0};
                    remaining[p]=rng.Next(1,31);
                }
                held[p].seqNumber=inputs[p].seqNumber;inputs[p]=held[p];remaining[p]--;
                if(tick%97==0) {inputs[p].jump=true;inputs[p].ab1=true;inputs[p].ab2=true;inputs[p].ab3=true;}
            }
            result.Add(new InputPacketQuad {p1=inputs[0],p2=inputs[1],p3=inputs[2],p4=inputs[3]});
        }
        byte[] encoded=new byte[105];Array.Copy(bytes,encoded,105);
        File.WriteAllBytes(destination,NetworkTools.SerializeReplay_compressed(result,encoded));
        Console.WriteLine("MUTATED_REPLAY frames="+original.Length+" seed="+seed);
    }

    public static void Main(string[] args)
    {
        if (args.Length==4 && args[0]=="mutate") { Mutate(args[1],args[2],int.Parse(args[3]));return; }
        if (args.Length==2 && args[0]=="inspect") { Inspect(args[1]); return; }
        if (args.Length < 2) throw new ArgumentException("replay <actions.json> <output.rep>");
        using var doc = JsonDocument.Parse(File.ReadAllText(args[0]));
        var root = doc.RootElement;
        var setup = root.GetProperty("setup");
        var header = new StartRequestPacket {
            seqNum = 1, seed = setup.GetProperty("seed").GetUInt32(), nrOfPlayers = 2,
            nrOfAbilites = 1, currentLevel = setup.GetProperty("level").GetByte(),
            frameBufferSize = 0, p1_color = 0, p2_color = 1, p1_team = 0, p2_team = 1,
            p1_ability1 = 2, p2_ability1 = 2
        };
        byte[] encoded = new byte[105];
        NetworkTools.EncodeStartRequest(ref encoded, header);
        var inputs = new List<InputPacketQuad>();
        uint sequence = 0;
        foreach (var frame in root.GetProperty("frames").EnumerateArray())
        {
            sequence++;
            InputPacket Packet(JsonElement player) => new InputPacket {
                seqNumber = sequence, joystickAngle = player[0].GetByte(),
                jump = (player[1].GetInt32() & 1) != 0,
                ab1 = (player[1].GetInt32() & 2) != 0
            };
            inputs.Add(new InputPacketQuad { p1 = Packet(frame[0]), p2 = Packet(frame[1]),
                p3 = new InputPacket { seqNumber = sequence }, p4 = new InputPacket { seqNumber = sequence } });
        }
        if (inputs.Count == 0) throw new ArgumentException("empty replay");
        File.WriteAllBytes(args[1], NetworkTools.SerializeReplay_compressed(inputs, encoded));
        Console.WriteLine("REPLAY_GENERATED frames=" + sequence);
    }
}
