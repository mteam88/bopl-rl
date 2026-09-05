using System;
using System.Text.Json;
using BoplFixedMath;

static class Program
{
    static void Main()
    {
        string line;
        while((line=Console.ReadLine())!=null)
        {
            using var doc=JsonDocument.Parse(line);var q=doc.RootElement;
            Updater.InitSeed(q.GetProperty("seed").GetUInt32());
            var field=typeof(Updater).GetField("random",System.Reflection.BindingFlags.NonPublic|System.Reflection.BindingFlags.Static);
            var result=new System.Collections.Generic.List<long[]>();
            foreach(var op in q.GetProperty("operations").EnumerateArray())
            {
                long min=op.GetProperty("min").GetInt64(),max=op.GetProperty("max").GetInt64();
                long value;
                var rng=(Unity.Mathematics.Random)field.GetValue(null);
                switch(op.GetProperty("kind").GetString()) {
                    case "int":value=rng.NextInt((int)min,(int)max);break;
                    case "uint":value=rng.NextUInt((uint)min,(uint)max);break;
                    case "fix":
                        value=Updater.RandomFix(Fix.FromRaw(min),Fix.FromRaw(max)).m_rawValue;
                        rng=(Unity.Mathematics.Random)field.GetValue(null);
                        break;
                    default:throw new ArgumentException("Unknown RNG operation");
                }
                field.SetValue(null,rng);
                result.Add(new long[]{value,rng.state});
            }
            Console.WriteLine(JsonSerializer.Serialize(result));
        }
    }
}
