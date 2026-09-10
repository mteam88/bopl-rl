using System;
using System.IO;
using BoplSim.Export;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityShim;

namespace BoplSim.Host
{
	// Drop-in replacement for the headless Unity player under native/rl/bopl_rl.c.
	//
	// The harness posix_spawns
	//   <game> -batchmode -nographics -logFile <log> <bootstrap.rep>
	// with BOPL_RL_SOCKET=<unix socket> and stdout/stderr on /dev/null, then
	// expects the game to connect and speak BRL1 v2 (RlBridge.cs, compiled into
	// the port unchanged). Entered when BOPL_RL_SOCKET is set or "rl" is the
	// first argument.
	//
	// Frame model: RlBridge is pumped from Host.Update once the bootstrap
	// replay has loaded a level (hasBeenInitialized && buildIndex >= 6). One
	// host frame here = [Host.Update -> RlBridge.Pump] -> end-of-frame destroy
	// flush -> frameCount++ -> next frame's Starts. That is exactly the frame
	// boundary Program.Run's replay/reset loop uses (oracle-verified against
	// the Unity trace), with RlBridge deciding how many ticks fall between
	// boundaries (CommandsPerFrame, and one boundary per scene load so the
	// loadedFrame gate lets scene Starts run before the first warmup tick).
	//
	// Deliberately NOT done, to stay on the trace-verified path: Update on
	// other behaviours, coroutine stepping, LateUpdate, Time advance. None of
	// them run in Program.Run's oracle loop either; gameplay is driven by
	// Updater.TickSimulation inside RlBridge. See sim/RL.md.
	public static class RlMode
	{
		private static bool AbilitySpawnsEnabled
			=> string.Equals(Environment.GetEnvironmentVariable("BOPL_RL_ABILITY_SPAWNS"), "1", StringComparison.Ordinal);

		// RL rollouts with fixed loadouts must not consume gameplay RNG or add
		// pickup bodies that the Rust slice does not simulate. Keep the authored
		// AbilitySpawner behavior untouched and apply this only at the RL host
		// boundary. An explicit opt-in preserves a way to exercise the full game.
		private static void ConfigureRlScene()
		{
			if (AbilitySpawnsEnabled) return;
			foreach (var spawner in UnityEngine.Object.FindObjectsOfType<AbilitySpawner>(true))
				spawner.enabled = false;
		}

		public static bool Requested(string[] args)
			=> (args.Length > 0 && args[0] == "rl") || !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("BOPL_RL_SOCKET"));

		public static int Run(string[] args)
		{
			string logPath = null, replayPath = null;
			for (int i = 0; i < args.Length; i++)
			{
				switch (args[i])
				{
					case "rl": if (i == 0) break; goto default;
					case "-batchmode":
					case "-nographics":
						break;
					case "-logFile":
						if (i + 1 < args.Length) logPath = args[++i];
						break;
					default:
						if (args[i].EndsWith(".rep", StringComparison.OrdinalIgnoreCase)) replayPath = args[i];
						else Console.Error.WriteLine("[rl] ignoring unknown argument: " + args[i]);
						break;
				}
			}

			// The player writes its Player.log to -logFile; the harness points
			// stdout/stderr at /dev/null. Route Debug.Log*/Console there so a
			// failing env leaves evidence in BOPL_RL_LOG_DIR.
			StreamWriter log = null;
			if (!string.IsNullOrEmpty(logPath))
			{
				try
				{
					log = new StreamWriter(new FileStream(logPath, FileMode.Create, FileAccess.Write, FileShare.Read)) { AutoFlush = true };
					Console.SetError(log);
					Console.SetOut(log);
				}
				catch (Exception error)
				{
					Console.Error.WriteLine("[rl] cannot open -logFile " + logPath + ": " + error.Message);
				}
			}

			try
			{
				if (!RlBridge.Enabled)
				{
					Console.Error.WriteLine("[rl] BOPL_RL_SOCKET is not set; nothing to connect to");
					return 2;
				}
				Console.Error.WriteLine($"[rl] Host RL mode pid={Environment.ProcessId} socket={Environment.GetEnvironmentVariable("BOPL_RL_SOCKET")} bootstrap={replayPath ?? "(none)"}");

				string repoRoot = Program.FindRepoRoot();
				var db = ExportDatabase.Load(
					Path.Combine(repoRoot, "native", "reference-export"),
					Path.Combine(repoRoot, "sim", "Export", "settings.json"),
					repoRoot);
				db.Install();
				// Every reset loads a fresh level. Disable its spawner as soon as
				// scene construction completes, before Starts or warmup ticks run.
				SceneManager.sceneLoaded += (_, mode) => {
					if (mode == LoadSceneMode.Single) ConfigureRlScene();
				};
				var host = Program.Bootstrap(db);

				// The player reaches Host.Update's RL branch by playing the argv
				// replay: Host.Awake -> "Replay" scene -> Init + ForceStartGame on
				// the replay's level. Same end state through ResetRl; the first
				// real reset from the harness replaces it (reset_check.c proves
				// resets are isolated from prior gameplay).
				host.ResetRl(BootstrapSetup(replayPath));
				ConfigureRlScene();
				// Bootstrap scene load completes its frame like every load below.
				EndFrame();
				// Boot leaves ~110 MiB of dead parse trees (boot scene, built
				// assets' JSON) behind a ~60 MiB live heap; with 16-32 game
				// processes per trainer that is worth one blocking compaction
				// before the harness starts timing us (sim/RL.md).
				GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
				Console.Error.WriteLine($"[rl] ready: heap {GC.GetTotalMemory(false) >> 20}MiB workingSet {Environment.WorkingSet >> 20}MiB");

				int frames = 0;
				while (!Application.quitRequested)
				{
					// Host.Update's RL branch (Host.cs: RlBridge.Enabled && hasBeenInitialized && buildIndex >= 6).
					if (host.hasBeenInitialized && SceneManager.GetActiveScene().buildIndex >= 6)
						RlBridge.Pump(host);
					else
					{
						Console.Error.WriteLine("[rl] Host not initialised on a level scene; cannot pump the RL bridge");
						return 1;
					}
					EndFrame();
					frames++;
				}
				Console.Error.WriteLine($"[rl] quit code={Application.quitCode} frames={frames} simulationTicks={Updater.SimulationTicks} " +
					$"heap={GC.GetTotalMemory(false) >> 20}MiB live={GC.GetTotalMemory(true) >> 20}MiB workingSet={Environment.WorkingSet >> 20}MiB gcs={GC.CollectionCount(0)}/{GC.CollectionCount(1)}/{GC.CollectionCount(2)}");
				return Application.quitCode;
			}
			catch (Exception error)
			{
				Console.Error.WriteLine("[rl] Host RL mode failed: " + error);
				return 1;
			}
			finally
			{
				log?.Flush();
			}
		}

		// Engine frame boundary, as in Program.Run: raw Object.Destroy drains at
		// frame end, queued Starts run at the top of the next frame before any
		// Update. Time.frameCount is what RlBridge's loadedFrame gate reads.
		private static void EndFrame()
		{
			World.Current.FlushDestroyQueue();
			World.Current.frameCount++;
			Time.frameCount = World.Current.frameCount;
			World.Current.FlushPendingStarts();
		}

		private static ReplayArchive BootstrapSetup(string replayPath)
		{
			if (!string.IsNullOrEmpty(replayPath))
			{
				try
				{
					byte[] bytes = File.ReadAllBytes(replayPath);
					if (ReplayArchive.IsArchive(bytes)) return ReplayArchive.Decode(bytes);
					// BOPL_RL_BOOTSTRAP (native/fixtures/movement.rep) is a legacy
					// compressed replay; ResetRl takes archives only. Any loaded
					// level does for the bootstrap.
					Console.Error.WriteLine("[rl] bootstrap replay is not a BOPLREP2 archive; using the default 2-player level-0 setup instead");
				}
				catch (Exception error)
				{
					Console.Error.WriteLine("[rl] cannot read bootstrap replay " + replayPath + ": " + error.Message + "; using the default setup");
				}
			}
			// bopl_rl_default_config for 2 players (native/rl/bopl_rl.c).
			var setup = new ReplayArchive
			{
				seed = 37,
				level = 0,
				stocks = 0,
				gameMode = 0,
				sourceHash = ReplaySession.SourceHash,
				catalogHash = ReplaySession.CatalogHash,
				players = new ReplayPlayer[2],
			};
			for (int i = 0; i < setup.players.Length; i++)
				setup.players[i] = new ReplayPlayer { id = (byte)(i + 1), color = (byte)(i % 4), team = (byte)i, keyboard = false, abilities = new ushort[] { 2, 7, 19 } };
			setup.frames.Add(new InputPacket[setup.players.Length]);
			setup.Validate();
			return setup;
		}
	}
}
