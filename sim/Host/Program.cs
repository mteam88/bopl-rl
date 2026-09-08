using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using BoplSim.Export;
using UnityEngine;
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
			var host = Bootstrap(db, repoRoot);

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

		// Recreates what the Unity boot scenes provide before a level loads. The
		// menu scenes are not exported, so SteamManager/Host are constructed
		// directly (no Awake: it only does Steam/netcode/command-line work).
		// Deviations recorded in sim/PORTING.md.
		private static global::Host Bootstrap(ExportDatabase db, string repoRoot)
		{
			// Settings singleton (SettingsInitializer.Awake equivalent).
			var settings = (Settings)db.LoadAssetByPath("Assets/MonoBehaviour/Settings.asset");
			settings.Init();
			LocalizedText.localizationTable = (LocalizationTable)db.LoadAssetByPath("Assets/MonoBehaviour/LocalizationTable.asset");

			// AudioManager singleton (boot scene, DontDestroyOnLoad). The sound bank
			// is not exported (audio is presentation) but Play() must run: it feeds
			// ReferencePlayerController.SoundCapture, which the trace records. Empty
			// bank -> every Play logs "non-existent sound" and returns null.
			var audioGo = new GameObject("AudioManager");
			var audio = (AudioManager)audioGo.AddComponentSilent(typeof(AudioManager));
			// Sound bank ripped from the boot AudioManager (MainMenu.unity ->
			// sim/Export/sound-bank.json): InterruptableAudio stores bank INDICES
			// (Dash.dashSound.index = 14 = "dash") which the trace captures, so the
			// order must be the player's. Clips are placeholders (audio never plays).
			string bankPath = Path.Combine(repoRoot, "sim", "Export", "sound-bank.json");
			var bank = Newtonsoft.Json.Linq.JArray.Parse(File.ReadAllText(bankPath));
			var sounds = new Sound[bank.Count];
			for (int i = 0; i < bank.Count; i++)
				sounds[i] = new Sound
				{
					name = (string)bank[i]["name"],
					volume = (float)bank[i]["volume"],
					pitch = (float)bank[i]["pitch"],
					loop = (bool)bank[i]["loop"],
					fadeLoop = (bool)bank[i]["fadeLoop"],
					clip = new AudioClip { name = (string)bank[i]["name"] },
				};
			audio.sounds = sounds;
			// Music paths (OnMatchStarted/StartMusic) index songs[] unconditionally;
			// one placeholder keeps them running. Music never reaches the trace.
			audio.songs = new[] { new Song { name = "placeholder", volume = 1f, pitch = 1f, introLength = 1f, intro = new AudioClip(), songLoop = new AudioClip() } };
			audio.settings = settings;
			audio.DispatchAwakeIfNeeded();
			audio.DispatchEnable();

			var steamGo = new GameObject("SteamManager");
			var steam = (SteamManager)steamGo.AddComponentSilent(typeof(SteamManager));
			var abilityIcons = db.LoadAssetByPath("Assets/MonoBehaviour/AbilityIcons.asset");
			var abilityIconsDemo = db.LoadAssetByPath("Assets/MonoBehaviour/AbilityIcons_demo.asset");
			SetPrivateField(steam, "abilityIconsFull", abilityIcons);
			SetPrivateField(steam, "abilityIconsDemo", abilityIconsDemo);
			SteamManager.instance = steam;
			UnityEngine.Object.DontDestroyOnLoad(steamGo);

			// The "winnerTriangle" additive scene (201-object UI, DontDestroyOnLoad
			// root) is NOT exported. GameSessionHandler dereferences exactly two
			// things at round start: WinnerTriangleCanvas.instance.gameObject and
			// instance.charStatsList.transform.parent.gameObject. Stub that shape;
			// the stats GO stays inactive so its round-end UI OnEnable never runs.
			// TODO(oracle): export WinnerTriangle.unity for round-end fidelity.
			var winnerCanvasGo = new GameObject("WinnerTriangleCanvas_leaveEnabled");
			var winnerCanvas = (WinnerTriangleCanvas)winnerCanvasGo.AddComponentSilent(typeof(WinnerTriangleCanvas));
			var triangleGo = new GameObject("WinnerCelebration_leaveDisabled");
			triangleGo.transform.SetParent(winnerCanvasGo.transform, false);
			var statsGo = new GameObject("CharacterStatsList");
			statsGo.SetActive(false);
			statsGo.transform.SetParent(triangleGo.transform, false);
			winnerCanvas.charStatsList = (CharacterStatsList)statsGo.AddComponentSilent(typeof(CharacterStatsList));
			winnerCanvas.DispatchAwakeIfNeeded(); // sets WinnerTriangleCanvas.instance
			UnityEngine.Object.DontDestroyOnLoad(winnerCanvasGo);

			var hostGo = new GameObject("Host");
			var host = (global::Host)hostGo.AddComponentSilent(typeof(global::Host));
			SetPrivateField(host, "playerColors", db.LoadAssetByPath("Assets/MonoBehaviour/PlayerColors.asset"));
			UnityEngine.Object.DontDestroyOnLoad(hostGo);
			steam.networkClient = host; // boot scene wires this serialized reference
			return host;
		}

		private static void SetPrivateField(object target, string name, object value)
		{
			for (var type = target.GetType(); type != null; type = type.BaseType)
			{
				var field = type.GetField(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly);
				if (field != null) { field.SetValue(target, value); return; }
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
