using System;
using System.Collections.Generic;
using System.Diagnostics.Tracing;
using System.IO;
using System.Linq;
using System.Text;
using BoplSim.Export;
using UnityEngine;
using UnityShim;

namespace BoplSim.Host
{
	// Allocation profile of the tick loop on the managed build (stage 5 item 3,
	// SIM_OVERHAUL.md). Runs the WorldsCheck scripted episode single-world and
	// attributes GC.GetAllocatedBytesForCurrentThread() deltas to the tick
	// phases (ReferenceScheduler.Capture markers) and to updatable component
	// types (ReferenceDeathEffects.ComponentPhase markers).
	//
	//   Host alloc [--level 1] [--decisions 400] [--abilities 2,4,14] [--rate 32] [--episodes 3] [--top 25]
	//
	// Measurement overhead: Capture marker args box ints for the 4 player
	// phases (~24 B each per tick); everything else passes references/null.
	internal static class AllocCheck
	{
		// GCAllocationTick_V4 fires once per ~100 KB allocated and names the
		// type that crossed the threshold: a fair by-bytes sample of allocation
		// types. Attribution is process-wide but the run is single-threaded.
		private sealed class AllocationTickListener : EventListener
		{
			internal readonly Dictionary<string, long> bytesByType = new Dictionary<string, long>();
			internal volatile bool enabled;

			protected override void OnEventSourceCreated(EventSource source)
			{
				if (source.Name == "Microsoft-Windows-DotNETRuntime")
					EnableEvents(source, EventLevel.Verbose, (EventKeywords)0x1); // GC keyword
			}

			protected override void OnEventWritten(EventWrittenEventArgs data)
			{
				if (!enabled || data.EventName != "GCAllocationTick_V4" && data.EventName != "GCAllocationTick_V3") return;
				string type = "?";
				long amount = 0;
				for (int i = 0; i < data.PayloadNames.Count; i++)
				{
					switch (data.PayloadNames[i])
					{
						case "TypeName": type = data.Payload[i]?.ToString() ?? "?"; break;
						case "AllocationAmount64": amount = Convert.ToInt64(data.Payload[i]); break;
					}
				}
				lock (bytesByType)
				{
					bytesByType.TryGetValue(type, out long total);
					bytesByType[type] = total + amount;
				}
			}
		}

		private static readonly Dictionary<string, long> byPhase = new Dictionary<string, long>();
		private static long lastMark;
		private static string currentLabel = "tick_head";
		private static bool measuring;

		private static void Mark(string next)
		{
			if (!measuring) return;
			long now = GC.GetAllocatedBytesForCurrentThread();
			byPhase.TryGetValue(currentLabel, out long total);
			byPhase[currentLabel] = total + (now - lastMark);
			lastMark = now;
			currentLabel = next;
		}

		public static int Run(string[] args)
		{
			int level = 1, decisions = 400, rate = 32, episodes = 3, top = 25;
			bool attrib = true;
			ushort[] abilities = { 2, 7, 19 };
			for (int i = 1; i < args.Length; i++)
			{
				switch (args[i])
				{
					case "--no-attrib": attrib = false; break;
					case "--level": level = int.Parse(args[++i]); break;
					case "--decisions": decisions = int.Parse(args[++i]); break;
					case "--rate": rate = int.Parse(args[++i]); break;
					case "--episodes": episodes = int.Parse(args[++i]); break;
					case "--top": top = int.Parse(args[++i]); break;
					case "--abilities":
						{
							var parts = args[++i].Split(',');
							abilities = new[] { ushort.Parse(parts[0]), ushort.Parse(parts[1]), ushort.Parse(parts[2]) };
							break;
						}
					default: Console.Error.WriteLine("Unknown argument: " + args[i]); return 2;
				}
			}

			var tickListener = new AllocationTickListener();
			string repoRoot = Program.FindRepoRoot();
			var db = ExportDatabase.Load(
				Path.Combine(repoRoot, "native", "reference-export"),
				Path.Combine(repoRoot, "sim", "Export", "settings.json"),
				repoRoot);
			db.Install();
			var host = Program.Bootstrap(db);
			var wire = new MemoryStream(65536);
			var payload = new BinaryWriter(wire, Encoding.UTF8, true);

			// Phase attribution: Capture marks the phase that FOLLOWS the call;
			// ComponentPhase marks the component call that FOLLOWS. Merge both
			// streams: attribute the delta since the previous mark to the
			// previous label, then switch. Labels are cached so the handlers
			// allocate nothing after their first sighting of a type.
			var phaseLabels = new Dictionary<string, string>();
			var componentLabels = new Dictionary<(string, Type), string>();
			if (attrib)
			{
			ReferenceScheduler.Capture = (label, arg) =>
			{
				if (!measuring) return;
				if (!phaseLabels.TryGetValue(label, out var cached)) phaseLabels[label] = cached = "phase:" + label;
				Mark(cached);
			};
			ReferenceDeathEffects.ComponentPhase = (phase, component) =>
			{
				if (!measuring) return;
				var key = (phase, component?.GetType());
				if (!componentLabels.TryGetValue(key, out var cached))
					componentLabels[key] = cached = "component:" + phase + ":" + (component == null ? "<null>" : component.GetType().Name);
				Mark(cached);
			};
			}

			long resetBytes = 0, warmupBytes = 0, tickBytes = 0, observeBytes = 0;
			int warmupTicks = 0, playedTicks = 0, observations = 0, steps = 0;

			for (int episode = 0; episode < episodes; episode++)
			{
				int episodeLevel = level + episode % 3;
				var setup = new ReplayArchive
				{
					seed = 7, level = (byte)episodeLevel, stocks = 3, gameMode = 0,
					sourceHash = ReplaySession.SourceHash, catalogHash = ReplaySession.CatalogHash,
					players = new ReplayPlayer[2],
				};
				for (int i = 0; i < 2; i++)
					setup.players[i] = new ReplayPlayer { id = (byte)(i + 1), color = (byte)(i % 4), team = (byte)i, keyboard = false, abilities = (ushort[])abilities.Clone() };
				setup.frames.Add(new InputPacket[2]);
				setup.Validate();

				long before = GC.GetAllocatedBytesForCurrentThread();
				RlObservation.Reset();
				RlTelemetry.Reset();
				host.ResetRl(setup);
				FrameBoundary(World.Current);
				resetBytes += GC.GetAllocatedBytesForCurrentThread() - before;

				var inputs = new InputPacket[2];
				var players = PlayerHandler.Get().PlayerList();

				before = GC.GetAllocatedBytesForCurrentThread();
				for (int t = 0; t < 240; t++) Tick(players, inputs, false);
				warmupBytes += GC.GetAllocatedBytesForCurrentThread() - before;
				warmupTicks += 240;

				byte[] held = new byte[2];
				uint rng = 0x9E3779B9u + (uint)episode * 7919u;
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
					before = GC.GetAllocatedBytesForCurrentThread();
					lastMark = before;
					currentLabel = "tick_head";
					measuring = true;
					tickListener.enabled = true;
					for (int r = 0; r < 4 && !terminal; r++)
					{
						Tick(players, inputs, true);
						episodeTicks++;
						playedTicks++;
						started |= !GameSessionHandler.HasGameEnded();
						terminal = started && GameSessionHandler.HasGameEnded();
					}
					Mark("tick_head");
					measuring = false;
					tickListener.enabled = false;
					tickBytes += GC.GetAllocatedBytesForCurrentThread() - before;

					before = GC.GetAllocatedBytesForCurrentThread();
					wire.SetLength(0);
					RlObservation.Write(payload, 2, episodeTicks, terminal ? 1u : 0u, 240);
					payload.Flush();
					observeBytes += GC.GetAllocatedBytesForCurrentThread() - before;
					observations++;
					steps++;
					if (terminal) break;
				}
			}

			ReferenceScheduler.Capture = null;
			ReferenceDeathEffects.ComponentPhase = null;

			Console.WriteLine($"episodes={episodes} decisions(steps)={steps} playedTicks={playedTicks} warmupTicks={warmupTicks}");
			Console.WriteLine($"reset            {resetBytes / Math.Max(1, episodes),12:N0} B/episode");
			Console.WriteLine($"warmup ticks     {(double)warmupBytes / Math.Max(1, warmupTicks),12:N2} B/tick   ({warmupBytes / Math.Max(1, episodes):N0} B/episode)");
			Console.WriteLine($"decision ticks   {(double)tickBytes / Math.Max(1, playedTicks),12:N2} B/tick");
			Console.WriteLine($"observation      {(double)observeBytes / Math.Max(1, observations),12:N2} B/obs");
			Console.WriteLine();
			Console.WriteLine($"decision-tick attribution (bytes/tick over {playedTicks} ticks, top {top}):");
			foreach (var pair in byPhase.OrderByDescending(p => p.Value).Take(top))
				Console.WriteLine($"  {pair.Key,-60} {(double)pair.Value / Math.Max(1, playedTicks),10:N2}");
			Console.WriteLine();
			long sampled = tickListener.bytesByType.Values.Sum();
			Console.WriteLine($"decision-tick allocation types (GCAllocationTick sample, {sampled:N0} B seen, top {top}):");
			foreach (var pair in tickListener.bytesByType.OrderByDescending(p => p.Value).Take(top))
				Console.WriteLine($"  {pair.Key,-70} {pair.Value,12:N0}  {100.0 * pair.Value / Math.Max(1, sampled),5:F1}%");
			return 0;
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
	}
}
