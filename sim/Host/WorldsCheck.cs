using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using BoplSim.Export;
using UnityEngine;
using UnityShim;

namespace BoplSim.Host
{
	// Many-worlds determinism check on the managed build (fast to iterate;
	// the .so path is native/rl/backend_check.c `worlds`):
	//
	//   Host worlds [--n 8] [--serial] [--level 1] [--decisions 400] [--abilities 2,4,14] [--rate 32]
	//
	// N worlds in this process, each on its own thread with its own
	// UnityShim.World, run the same scripted episode (backend_check phase b's
	// held-stick schedule) and hash every observation. All worlds must print
	// the checksum of a single world, concurrent or not.
	internal static class WorldsCheck
	{
		private sealed class Result { public ulong Checksum; public int Steps; public string Error; public List<ulong> PerStep = new List<ulong>(); }

		// Debug aid: dump each decision's wire payload for byte-level A/B diffs.
		private static readonly string wireDumpDir = Environment.GetEnvironmentVariable("BOPL_WIRE_DIR");
		// Debug aid: game seed (--seed), for measuring whether the seed enters warmup.
		private static uint GameSeed = 7;
		// --vary-seeds: distinct seed per episode - the snapshot cross-seed gate
		// (fresh vs restored episodes under changing seeds must stay bit-exact).
		private static bool VarySeeds;

		// Number of xorshift steps from state `from` to state `to` (each RNG draw
		// advances exactly one step); -1 if not reached within the bound.
		private static long CountSteps(uint from, uint to)
		{
			uint s = from;
			for (long n = 0; n < 10_000_000; n++)
			{
				if (s == to) return n;
				s ^= s << 13; s ^= s >> 17; s ^= s << 5;
			}
			return -1;
		}

		public static int Run(string[] args)
		{
			int n = 8, level = 1, decisions = 400, rate = 32, episodes = 1;
			bool serial = false;
			ushort[] abilities = { 2, 7, 19 };
			for (int i = 1; i < args.Length; i++)
			{
				switch (args[i])
				{
					case "--n": n = int.Parse(args[++i]); break;
					case "--serial": serial = true; break;
					case "--level": level = int.Parse(args[++i]); break;
					case "--decisions": decisions = int.Parse(args[++i]); break;
					case "--rate": rate = int.Parse(args[++i]); break;
					case "--episodes": episodes = int.Parse(args[++i]); break;
					case "--seed": GameSeed = uint.Parse(args[++i]); break;
					case "--vary-seeds": VarySeeds = true; break;
					case "--abilities":
						{
							var parts = args[++i].Split(',');
							abilities = new[] { ushort.Parse(parts[0]), ushort.Parse(parts[1]), ushort.Parse(parts[2]) };
							break;
						}
					default: Console.Error.WriteLine("Unknown argument: " + args[i]); return 2;
				}
			}
			// The main thread was bound to a scratch world by Main; release it so
			// the worker threads' worlds are the only ones.
			World.Unbind();
			if (Environment.GetEnvironmentVariable("BOPL_SNAPSHOT_AUDIT") != null) WorldCopier.CollectSharedTypeNames = true;

			string repoRoot = Program.FindRepoRoot();
			var db = ExportDatabase.Load(
				Path.Combine(repoRoot, "native", "reference-export"),
				Path.Combine(repoRoot, "sim", "Export", "settings.json"),
				repoRoot);

			var results = new Result[n];
			var threads = new Thread[n];
			for (int k = 0; k < n; k++)
			{
				int index = k;
				results[k] = new Result();
				threads[k] = new Thread(() => RunWorld(db, level, decisions, abilities, rate, episodes, results[index]), 64 << 20);
				threads[k].Start();
				if (serial) threads[k].Join();
			}
			if (!serial) foreach (var t in threads) t.Join();

			int status = 0;
			for (int k = 0; k < n; k++)
			{
				var r = results[k];
				Console.WriteLine($"[world{k}] steps={r.Steps} checksum={r.Checksum:x16}{(r.Error != null ? " ERROR " + r.Error : "")}");
				if (r.Error != null || r.Checksum != results[0].Checksum) status = 1;
				if (r.Checksum != results[0].Checksum)
				{
					int first = 0;
					while (first < r.PerStep.Count && first < results[0].PerStep.Count && r.PerStep[first] == results[0].PerStep[first]) first++;
					Console.WriteLine($"[world{k}]   first divergent step: {first}");
				}
			}
			Console.WriteLine("worlds agree: " + (status == 0 ? "yes" : "NO"));
			if (Environment.GetEnvironmentVariable("BOPL_SNAPSHOT_AUDIT") != null)
			{
				Console.WriteLine("[audit] CoreLib collections copied by reflection (need rd.xml entries under NativeAOT):");
				foreach (var name in new SortedSet<string>(WorldCopier.PlannedCollectionNames)) Console.WriteLine("  " + name);
				Console.WriteLine("[audit] types shared, not copied:");
				foreach (var name in new SortedSet<string>(WorldCopier.SharedTypeNames)) Console.WriteLine("  " + name);
				Console.WriteLine("[audit] slot discovery anomalies:");
				foreach (var pair in WorldCopier.AnomalyFields) Console.WriteLine($"  {pair.Value,6}  {pair.Key}");
				Console.WriteLine("[audit] struct array elements:");
				foreach (var pair in WorldCopier.StructArrayElements) Console.WriteLine($"  {pair.Value,6}  {pair.Key}");
			}
			return status;
		}

		private static void RunWorld(ExportDatabase db, int level, int decisions, ushort[] abilities, int rate, int episodes, Result result)
		{
			var world = new World();
			World.Bind(world);
			try
			{
				db.Install();
				var host = Program.Bootstrap(db);
				var wire = new MemoryStream(65536);
				var payload = new BinaryWriter(wire, Encoding.UTF8, true);
				ulong checksum = 14695981039346656037UL;
				int steps = 0;
				var snapshots = new Dictionary<string, WorldSnapshot.Entry>();
				for (int episode = 0; episode < episodes; episode++)
				{
					uint gameSeed = VarySeeds ? GameSeed + (uint)episode * 2654435761u : GameSeed;
					RunEpisode(ref host, level + episode % 3, decisions, abilities, rate, (uint)(0x9E3779B9u + (uint)episode * 7919u), gameSeed,
						wire, payload, snapshots, ref checksum, ref steps, result);
				}
				result.Checksum = checksum;
				result.Steps = steps;
			}
			catch (Exception e) { result.Error = e.ToString(); }
			finally { World.Unbind(); }
		}

		private static void RunEpisode(ref global::Host host, int level, int decisions, ushort[] abilities, int rate, uint rng, uint gameSeed,
			MemoryStream wire, BinaryWriter payload, Dictionary<string, WorldSnapshot.Entry> snapshots, ref ulong checksum, ref int steps, Result result)
		{
			{
				var world = World.Current;
				// Post-warmup snapshot reuse, mirroring NativeApi.DoReset: key is
				// the configuration minus the seed. BOPL_NO_SNAPSHOT=1 for A/B.
				string key = WorldSnapshot.Enabled ? level + ":" + string.Join(",", abilities) : null;
				InputPacket[] inputs;
				List<Player> players;
				if (key != null && snapshots.TryGetValue(key, out var entry))
				{
					World.Unbind();
					var sw = System.Diagnostics.Stopwatch.StartNew();
					world = WorldSnapshot.Restore(entry, gameSeed, out host);
					if (Environment.GetEnvironmentVariable("BOPL_SNAPSHOT_AUDIT") != null)
						Console.WriteLine($"[copy] restore level={level} ms={sw.Elapsed.TotalMilliseconds:f2} objects={WorldCopier.LastObjects} arrays={WorldCopier.LastArrays} structElems={WorldCopier.LastStructElements} slowFields={WorldCopier.LastSlowFields}");
					inputs = new InputPacket[2];
					players = PlayerHandler.Get().PlayerList();
				}
				else
				{
					var setup = new ReplayArchive
					{
						seed = gameSeed, level = (byte)level, stocks = 3, gameMode = 0,
						sourceHash = ReplaySession.SourceHash, catalogHash = ReplaySession.CatalogHash,
						players = new ReplayPlayer[2],
					};
					for (int i = 0; i < 2; i++)
						setup.players[i] = new ReplayPlayer { id = (byte)(i + 1), color = (byte)(i % 4), team = (byte)i, keyboard = false, abilities = (ushort[])abilities.Clone() };
					setup.frames.Add(new InputPacket[2]);
					setup.Validate();
					RlObservation.Reset();
					RlTelemetry.Reset();
					host.ResetRl(setup);
					FrameBoundary(world);
					inputs = new InputPacket[2];
					players = PlayerHandler.Get().PlayerList();
					uint rngBefore = Updater.random.state;
					for (int t = 0; t < 240; t++) Tick(players, inputs, false);
					if (Environment.GetEnvironmentVariable("BOPL_RNG_TRACE") != null)
						Console.WriteLine($"[rng] level={level} seed={gameSeed} before_warmup={rngBefore:x8} after_warmup={Updater.random.state:x8}"
							+ $" steps_to_before={CountSteps(new Unity.Mathematics.Random(gameSeed).state, rngBefore)} steps_in_warmup={CountSteps(rngBefore, Updater.random.state)}");
					if (key != null)
					{
						var captured = WorldSnapshot.Capture(world, gameSeed, null, false);
						if (captured != null) snapshots[key] = captured;
						// Audit gate: a snapshot graph must not reach any live
						// (source) object the copier meant to replace - a miss here
						// is how live worlds leaked into snapshots (copies of
						// copies) before the per-top plan demotion fix.
						if (captured != null && Environment.GetEnvironmentVariable("BOPL_SNAPSHOT_AUDIT") != null)
						{
							Console.WriteLine($"[capture] level={level} objects={WorldCopier.LastObjects} arrays={WorldCopier.LastArrays}");
							foreach (var slot in WorldCopier.LastCopier.FindMissedSlots(captured.graph))
								Console.WriteLine("[missed] level=" + level + " " + slot);
						}
					}
				}

				byte[] held = new byte[2];
				uint mask = (uint)rate - 1u;
				int episodeTicks = 0;
				bool started = false;
				for (int i = 0; i < decisions; i++)
				{
					byte[] angle = new byte[2]; ushort[] buttons = new ushort[2];
					for (int k = 0; k < 2; k++)
					{
						rng = rng * 1664525u + 1013904223u;
						if ((i & 7) == 0) held[k] = (byte)(rng >> 24);
						angle[k] = held[k];
						if (((rng >> 4) & 7) == 0) buttons[k] |= 1;
						if (((rng >> 16) & mask) == 0) buttons[k] |= (ushort)(2u << (int)((rng >> 21) % 3));
					}
					for (int k = 0; k < 2; k++) inputs[k] = ReplayArchive.Input((uint)(Updater.SimulationTicks + 1), angle[k], buttons[k]);
					bool terminal = false;
					for (int r = 0; r < 4 && !terminal; r++)
					{
						Tick(players, inputs, true);
						episodeTicks++;
						started |= !GameSessionHandler.HasGameEnded();
						terminal = started && GameSessionHandler.HasGameEnded();
					}
					wire.SetLength(0);
					RlObservation.Write(payload, 2, episodeTicks, terminal ? 1u : 0u, 240);
					payload.Flush();
					checksum = Fnv(checksum, wire.GetBuffer(), (int)wire.Length);
					if (wireDumpDir != null)
						File.WriteAllBytes(Path.Combine(wireDumpDir, $"step{steps:d4}.bin"), wire.ToArray());
					result.PerStep.Add(checksum);
					steps++;
					if (terminal) break;
				}
				if (Environment.GetEnvironmentVariable("BOPL_RNG_TRACE") != null)
				{
					ulong delays = 14695981039346656037UL;
					var session = GameSessionHandler.selfRef;
					if (session?.introDelayDrawn != null)
						for (int i = 0; i < session.introDelayDrawn.Length; i++)
							if (session.introDelayDrawn[i])
							{
								delays ^= (ulong)session.DelayRaw(i); delays *= 1099511628211UL;
							}
					Console.WriteLine($"[rng-end] level={level} seed={gameSeed} state={Updater.random.state:x8} delays={delays:x16}");
				}
			}
		}

		private static void Tick(List<Player> players, InputPacket[] inputs, bool counted)
		{
			for (int i = 0; i < inputs.Length; i++) inputs[i].seqNumber = (uint)(Updater.SimulationTicks + 1);
			for (int i = 0; i < inputs.Length; i++) players[i].OverrideInputWithNetworkInput(inputs[i]);
			RlTelemetry.Counting = counted;
			try { Updater.TickSimulation(GameTime.FixedTimeStep); }
			finally { RlTelemetry.Counting = false; }
			Updater.LoadedALevelThisUpdate = false;
		}

		private static void FrameBoundary(World world)
		{
			world.FlushDestroyQueue();
			world.frameCount++;
			Time.frameCount = world.frameCount;
			world.FlushPendingStarts();
		}

		private static ulong Fnv(ulong hash, byte[] data, int length)
		{
			for (int i = 0; i < length; i++) { hash ^= data[i]; hash *= 1099511628211UL; }
			return hash;
		}
	}
}
