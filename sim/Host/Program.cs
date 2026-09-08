using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using BoplSim.Export;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityShim;

namespace BoplSim.Host
{
	// Drives the ported simulation the way RlBridge drives it under Unity:
	//   Host reset  --level 0 --players 2 --ticks 240 [--seed 37] [--stocks 0] [--mode 0]
	//   Host replay <episode.rep> [--ticks N]
	// BOPL_TRACE=/path.jsonl.gz makes ReferenceCapture record a trace, as in Unity.
	public static class Program
	{
		public static int Main(string[] args)
		{
			try { return Run(args); }
			catch (Exception error)
			{
				Console.Error.WriteLine("Host failed: " + error);
				return 1;
			}
		}

		private static int Run(string[] args)
		{
			string mode = args.Length > 0 ? args[0] : "reset";
			string replayPath = null;
			int level = 0, players = 2, ticks = 240, stocks = 0, gameMode = 0;
			// Unity runs several simulation ticks per engine frame (RlBridge batches;
			// the oracle traces with BOPL_TRACE_BATCH=16). Raw Object.Destroy resolves
			// at frame end and Start runs at frame beginnings, so the tick:frame ratio
			// is observable - match the oracle's batch.
			int ticksPerFrame = Math.Max(1, ReferenceCapture.Batch == 1 && Environment.GetEnvironmentVariable("BOPL_TRACE_BATCH") == null ? 16 : ReferenceCapture.Batch);
			uint seed = 37;
			for (int i = 1; i < args.Length; i++)
			{
				switch (args[i])
				{
					case "--level": level = int.Parse(args[++i]); break;
					case "--players": players = int.Parse(args[++i]); break;
					case "--ticks": ticks = int.Parse(args[++i]); break;
					case "--seed": seed = uint.Parse(args[++i]); break;
					case "--stocks": stocks = int.Parse(args[++i]); break;
					case "--mode": gameMode = int.Parse(args[++i]); break;
					case "--frame-ticks": ticksPerFrame = int.Parse(args[++i]); break;
					default:
						if (replayPath == null && !args[i].StartsWith("--")) replayPath = args[i];
						else { Console.Error.WriteLine("Unknown argument: " + args[i]); return 2; }
						break;
				}
			}
			if (mode == "replay" && replayPath == null) { Console.Error.WriteLine("replay mode needs a .rep path"); return 2; }

			string repoRoot = FindRepoRoot();
			var db = ExportDatabase.Load(
				Path.Combine(repoRoot, "native", "reference-export"),
				Path.Combine(repoRoot, "sim", "Export", "settings.json"),
				repoRoot);
			db.Install();
			var host = Bootstrap(db);

			ReplayArchive setup;
			if (mode == "replay")
			{
				setup = ReplayArchive.Decode(File.ReadAllBytes(replayPath));
			}
			else
			{
				// Mirrors RlBridge.Reset with bopl_rl_default_config (native/rl/bopl_rl.c):
				// color=i%4, team=i, abilities Dash/Gust/Invisibility, zero-input first frame.
				setup = new ReplayArchive
				{
					seed = seed,
					level = (byte)level,
					stocks = stocks,
					gameMode = (byte)gameMode,
					sourceHash = ReplaySession.SourceHash,
					catalogHash = ReplaySession.CatalogHash,
					players = new ReplayPlayer[players],
				};
				for (int i = 0; i < players; i++)
					setup.players[i] = new ReplayPlayer
					{
						id = (byte)(i + 1),
						color = (byte)(i % 4),
						team = (byte)i,
						keyboard = false,
						abilities = new ushort[] { 2, 7, 19 },
					};
				setup.frames.Add(new InputPacket[players]);
				setup.Validate();
			}

			host.ResetRl(setup);
			if (Environment.GetEnvironmentVariable("BOPL_DEBUG_ABIL") != null)
			{
				var p0 = PlayerHandler.Get().PlayerList()[0];
				foreach (var abilityGo in p0.Abilities)
					Console.Error.WriteLine($"[debug] ability go={abilityGo.name} isAsset={abilityGo.isAsset} sceneValid={abilityGo.scene.IsValid()} sceneName='{abilityGo.scene.name}' updatables={string.Join(",", Array.ConvertAll(abilityGo.GetComponents<MonoUpdatable>(), m => m.GetType().Name + "#" + m.HierarchyNumber))}");
			}
			// RlBridge lets Unity finish the frame after LoadScene: scene Starts run
			// before the first simulation tick.
			World.Current.FlushPendingStarts();
			World.Current.FlushDestroyQueue();

			int playerCount = setup.players.Length;
			var playerList = PlayerHandler.Get().PlayerList();
			var replayFrames = mode == "replay" ? new Queue<InputPacket[]>(ReplaySession.Playing.frames) : null;
			var inputs = new InputPacket[playerCount];

			var stopwatch = Stopwatch.StartNew();
			int ticked = 0;
			for (; ticked < ticks; ticked++)
			{
				if (replayFrames != null)
				{
					if (replayFrames.Count == 0) break;
					inputs = replayFrames.Dequeue();
				}
				for (int i = 0; i < inputs.Length && i < playerCount; i++)
					inputs[i].seqNumber = (uint)(Updater.SimulationTicks + 1);
				if (ReferenceCapture.Recording) ReferenceCapture.BeforeReplayStep(inputs);
				for (int i = 0; i < inputs.Length && i < playerList.Count; i++)
					playerList[i].OverrideInputWithNetworkInput(inputs[i]);
				Updater.TickSimulation(GameTime.FixedTimeStep);
				Updater.LoadedALevelThisUpdate = false;
				// Engine frame boundary: raw Object.Destroy drains at frame end,
				// queued Starts run at the top of the next frame (before any tick).
				if ((ticked + 1) % ticksPerFrame == 0)
				{
					World.Current.FlushDestroyQueue();
					World.Current.frameCount++;
					Time.frameCount = World.Current.frameCount;
					World.Current.FlushPendingStarts();
				}
			}
			stopwatch.Stop();
			World.Current.FlushDestroyQueue();

			if (ReferenceCapture.Recording && !ReferenceCapture.Completed) ReferenceCapture.Finish("host-done");

			double perSecond = ticked / Math.Max(stopwatch.Elapsed.TotalSeconds, 1e-9);
			Console.WriteLine($"ticks={ticked}/{ticks} elapsed={stopwatch.Elapsed.TotalSeconds:F3}s ticks/s={perSecond:F0} " +
				$"simulationTicks={Updater.SimulationTicks} gameEnded={GameSessionHandler.HasGameEnded()}");
			return 0;
		}

		// Unity boots MainMenu.unity (build index 0, exported as scene-boot.json).
		// Loading the real scene runs the real singleton Awakes - SteamManager
		// (Steam init fails headless, same as the batchmode oracle), Host on the
		// "networkClient" root, SettingsInitializer (Settings.Init +
		// LocalizedText.localizationTable), AudioManager (136-sound bank) - and
		// the serialized references wire themselves (SteamManager.networkClient,
		// Host.playerColors...). Menu UI roots die at the first Single level load.
		private static global::Host Bootstrap(ExportDatabase db)
		{
			SceneManager.LoadScene(0);
			var host = UnityEngine.Object.FindObjectOfType<global::Host>();
			if (host == null) throw new InvalidOperationException("boot scene did not provide a Host (networkClient root)");

			// Host.Awake early-returns when process argv carries a .rep path (the
			// player's command-line replay flow via the unexported "Replay" scene,
			// which we do not take) - skipping its DontDestroyOnLoad calls. Re-apply
			// what its no-argv path does; idempotent when Awake already did it.
			UnityEngine.Object.DontDestroyOnLoad(host.gameObject);
			var debugText = GetPrivateField(host, "DebugTextRenderer") as UnityEngine.Object;
			if (debugText != null) UnityEngine.Object.DontDestroyOnLoad(debugText);

			// The boot frame completes before anything else: Starts run
			// (DontDestroyOnLoad.Start among them), deferred destroys drain.
			World.Current.FlushPendingStarts();
			World.Current.FlushDestroyQueue();
			// Surface loader warnings not yet dumped by a scene load - under
			// NativeAOT reflection failures land here silently (54fe47a).
			db.DumpWarnings("bootstrap");
			return host;
		}

		private static object GetPrivateField(object target, string name)
		{
			for (var type = target.GetType(); type != null; type = type.BaseType)
			{
				var field = type.GetField(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly);
				if (field != null) return field.GetValue(target);
			}
			throw new MissingFieldException(target.GetType().Name, name);
		}

		private static string FindRepoRoot()
		{
			for (var dir = new DirectoryInfo(Directory.GetCurrentDirectory()); dir != null; dir = dir.Parent)
				if (Directory.Exists(Path.Combine(dir.FullName, "native", "reference-export")))
					return dir.FullName;
			for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
				if (Directory.Exists(Path.Combine(dir.FullName, "native", "reference-export")))
					return dir.FullName;
			throw new DirectoryNotFoundException("Cannot locate repo root (native/reference-export)");
		}
	}
}
