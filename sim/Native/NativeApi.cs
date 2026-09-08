using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using BoplSim.Export;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityShim;

namespace BoplSim.Native
{
	// In-process implementation of native/rl/bopl_rl.h (ABI version 2).
	//
	// The socket path is: bopl_rl.c <-> unix socket <-> RlBridge.Pump inside a
	// Unity player. This library removes the process and the socket: it is the
	// re-implementation of RlBridge's private dispatch layer (whose observation,
	// telemetry, reset and tick entry points are reused verbatim via
	// InternalsVisibleTo) plus the client-side half of bopl_rl.c (argument
	// validation, status codes, BoplRlState decoding).
	//
	// One world per process: gameplay lives in statics (Updater, PlayerHandler,
	// GameSessionHandler...), so a second bopl_rl_open in the same process
	// returns an error. BOPL_TRACE keeps working: Tick calls
	// ReferenceCapture.BeforeReplayStep exactly like RlBridge.Tick.
	public static unsafe class NativeApi
	{
		private const uint Version = 2; // RlBridge.Version
		// RlBridge.CommandsPerFrame: simulation ticks per engine frame. Raw
		// Object.Destroy drains at frame end and queued Starts run at the top of
		// the next frame, so the cadence is gameplay-observable and must match
		// the Unity RL runtime (not the 16-tick cadence of the trace fixtures).
		private const int CommandsPerFrame = 4096;

		private const int Ok = 0, Argument = 1, Io = 2, GameError = 3, Ended = 4;

		// ---- handle & world (one per process) -------------------------------
		private static void* handle;
		private static bool openedEver, closed, broken;
		private static byte* errorBuffer;      // 512 bytes, mirrors BoplRl.error
		private static byte* nullHandleText;   // "NULL RL handle"
		private static byte* sourceHashText, catalogHashText;
		private static uint maps, abilityCount;
		private static global::Host host;

		// ---- episode state (mirrors RlBridge's private fields) --------------
		private static bool valid, sampleValid, started, terminal, failed, resetPending;
		private static int warmup, episodeTicks, episodeLimit, ticksThisFrame, resetResult;
		private static InputPacket[] inputs;
		private static ReplayArchive recording;

		private static readonly MemoryStream wire = new MemoryStream(65536);
		private static readonly BinaryWriter payload = new BinaryWriter(wire, Encoding.UTF8, true);

		// ---- BoplRlState layout (bopl_rl.h, _Static_asserts in bopl_rl.c) ---
		private const int HeaderSize = 48, PlayerSize = 72, EntitySize = 128, SlimeSize = 136, TelemetrySize = 36;
		private const int PlayersOffset = HeaderSize;                    // 48
		private const int EntitiesPtrOffset = PlayersOffset + 6 * PlayerSize; // 480
		private const int SlimesPtrOffset = EntitiesPtrOffset + 8;       // 488
		private const int TelemetryOffset = SlimesPtrOffset + 8;         // 496
		private const int StateSize = TelemetryOffset + 6 * TelemetrySize; // 712

		private struct Snapshot
		{
			public byte* state, entities, slimes;
			public uint entityCapacity, slimeCapacity;
		}
		private static Snapshot main, sample;
		private static byte* replayBuffer;
		private static nuint replayCapacity;

		// ================= exports =================

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_abi_version")]
		public static uint AbiVersion() => Version;

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_default_config")]
		public static void DefaultConfig(uint* config, uint players)
		{
			// Verbatim bopl_rl_default_config (bopl_rl.c). Config is 44 uints.
			if (config == null) return;
			new Span<uint>(config, 44).Clear();
			config[0] = 37; config[2] = players; config[3] = 3;
			config[6] = 240; config[7] = 7200;
			for (uint i = 0; i < 6; i++)
			{
				uint* p = config + 8 + i * 6;
				p[0] = i % 4; p[1] = i;
				p[3] = 2; p[4] = 7; p[5] = 19; // Dash / Gust / Invisibility
			}
		}

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_open")]
		public static void* Open(byte* executable, byte* replay, byte* logPath, uint timeoutMs, byte* error, nuint capacity)
		{
			// Same argument contract as the socket loader, for drop-in use.
			// executable/bootstrap replay are unused in-process: there is no
			// worker process and no Replay-scene bootstrap; the world boots from
			// the export like sim/Host. timeout is meaningless without a socket.
			if (!IsAbsolute(executable) || !IsAbsolute(replay) || !IsAbsolute(logPath) || timeoutMs < 1 || timeoutMs > 3600000)
			{
				ErrorText(error, capacity, "Absolute executable/replay/log paths and timeout 1..3600000 ms required");
				return null;
			}
			if (openedEver)
			{
				ErrorText(error, capacity, "One RL world per process (gameplay statics); spawn another process for a second world");
				return null;
			}
			openedEver = true;
			try
			{
				// -logFile equivalent: the shim's Debug.Log* writes to stderr,
				// which belongs to the training process here.
				Console.SetError(new StreamWriter(ReadUtf8(logPath), false) { AutoFlush = true });
				// RlTelemetry.Enabled => RlBridge.Enabled && Counting, and
				// RlBridge.Enabled reads BOPL_RL_SOCKET once. No socket exists
				// in-process; set the flag so telemetry counts like the socket
				// path. The only other consumers (Host.Update -> Pump,
				// Host.OnLevelWasLoaded) are never dispatched by the shim host.
				if (Environment.GetEnvironmentVariable("BOPL_RL_SOCKET") == null)
					Environment.SetEnvironmentVariable("BOPL_RL_SOCKET", "in-process");

				string repoRoot = FindRepoRoot();
				var db = ExportDatabase.Load(
					Path.Combine(repoRoot, "native", "reference-export"),
					Path.Combine(repoRoot, "sim", "Export", "settings.json"),
					repoRoot);
				db.Install();
				host = Bootstrap(db);

				maps = (uint)(SceneManager.sceneCountInBuildSettings - 6);
				abilityCount = (uint)SteamManager.instance.abilityIcons.sprites.Count;
				sourceHashText = MakeCString(ReplaySession.SourceHash);
				catalogHashText = MakeCString(ReplaySession.CatalogHash);
				errorBuffer = (byte*)NativeMemory.AllocZeroed(512);
				handle = NativeMemory.Alloc(8);
				ErrorText(error, capacity, "");
				return handle;
			}
			catch (Exception exception)
			{
				ErrorText(error, capacity, "RL world bootstrap failed: " + exception);
				return null;
			}
		}

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_reset_begin")]
		public static int ResetBegin(void* world, uint* config) => ResetBeginImpl(world, config);

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_reset_finish")]
		public static int ResetFinish(void* world) => ResetFinishImpl(world);

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_reset_pending")]
		public static int ResetPending(void* world) => world != null && world == handle && resetPending ? 1 : 0;

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_reset")]
		public static int Reset(void* world, uint* config)
		{
			int result = ResetBeginImpl(world, config);
			return result != Ok ? result : ResetFinishImpl(world);
		}

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_step")]
		public static int Step(void* world, byte* actions, nuint count, uint repeat)
		{
			int check = CheckActions(world, actions, count);
			if (check != Ok) return check;
			if (repeat < 1 || repeat > 16) return Argument;
			if (broken) return Io;
			sampleValid = false;
			try
			{
				// RlBridge defers a step that would cross the remaining frame
				// budget (pendingStep): the engine frame ends before the step,
				// never inside it.
				if (ticksThisFrame > 0 && ticksThisFrame + repeat > CommandsPerFrame) FrameBoundary();
				ReadActions(actions, (int)count);
				for (int i = 0; i < repeat && !terminal; i++) Tick(true);
				Observe(ref main);
				if (ticksThisFrame >= CommandsPerFrame) FrameBoundary();
				valid = true;
				SetError("");
				return Ok;
			}
			catch (Exception exception) { return GameFailed(exception); }
		}

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_step_sampled")]
		public static int StepSampled(void* world, byte* actions, nuint count, uint first, uint second)
		{
			int check = CheckActions(world, actions, count);
			if (check != Ok) return check;
			if (first < 1 || first > 16 || second < 1 || second > 16 || first + second > 16) return Argument;
			if (broken) return Io;
			sampleValid = false;
			try
			{
				if (ticksThisFrame > 0 && ticksThisFrame + first + second > CommandsPerFrame) FrameBoundary();
				ReadActions(actions, (int)count);
				for (int i = 0; i < first && !terminal; i++) Tick(true);
				Observe(ref sample);
				if (terminal)
				{
					// Episode ended during the first phase: sample and final
					// state are identical (bopl_rl.c copy_snapshot).
					DecodeInto(ref main);
				}
				else
				{
					for (int i = 0; i < second && !terminal; i++) Tick(true);
					Observe(ref main);
				}
				if (ticksThisFrame >= CommandsPerFrame) FrameBoundary();
				valid = true; sampleValid = true;
				SetError("");
				return Ok;
			}
			catch (Exception exception) { return GameFailed(exception); }
		}

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_state")]
		public static void* State(void* world) => world != null && world == handle && valid ? main.state : null;

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_sample")]
		public static void* Sample(void* world) => world != null && world == handle && valid && sampleValid ? sample.state : null;

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_replay")]
		public static int Replay(void* world, void** bytes, nuint* length)
		{
			if (world == null || world != handle || bytes == null || length == null) return Argument;
			if (closed || broken) return Io;
			try
			{
				// RlBridge case 5 throws this as InvalidOperationException.
				if (recording == null || recording.frames.Count == 0)
					throw new InvalidOperationException("Replay requires at least one simulated tick");
				byte[] encoded = recording.Encode();
				if ((nuint)encoded.Length > replayCapacity)
				{
					replayBuffer = (byte*)NativeMemory.Realloc(replayBuffer, (nuint)encoded.Length);
					replayCapacity = (nuint)encoded.Length;
				}
				Marshal.Copy(encoded, 0, (IntPtr)replayBuffer, encoded.Length);
				*bytes = replayBuffer;
				*length = (nuint)encoded.Length;
				SetError("");
				return Ok;
			}
			catch (Exception exception) { return GameFailed(exception); }
		}

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_error")]
		public static byte* Error(void* world)
		{
			if (world != null && world == handle && errorBuffer != null) return errorBuffer;
			if (nullHandleText == null) nullHandleText = MakeCString("NULL RL handle");
			return nullHandleText;
		}

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_source_hash")]
		public static byte* SourceHash(void* world) => world != null && world == handle ? sourceHashText : null;

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_catalog_hash")]
		public static byte* CatalogHash(void* world) => world != null && world == handle ? catalogHashText : null;

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_map_count")]
		public static uint MapCount(void* world) => world != null && world == handle ? maps : 0;

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_ability_count")]
		public static uint AbilityCount(void* world) => world != null && world == handle ? abilityCount : 0;

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_close")]
		public static void Close(void* world)
		{
			if (world == null || world != handle || closed) return;
			closed = true;
			resetPending = false;
			valid = sampleValid = false;
			// RlBridge case 4: finish an active trace instead of quitting.
			try
			{
				if (ReferenceCapture.Recording && !ReferenceCapture.Completed) ReferenceCapture.Finish("rl-closed");
			}
			catch (Exception exception) { Debug.LogError("RL close: " + exception); }
			// Gameplay statics stay dirty; the process-wide open latch stays set.
		}

		// ================= internals =================

		private static int ResetBeginImpl(void* world, uint* config)
		{
			if (world == null || world != handle || closed || config == null) return Argument;
			// Client-side validation, verbatim from bopl_rl_reset_begin.
			uint players = config[2], slots = config[3];
			if (players < 1 || players > 6 || slots < 1 || slots > 3 || config[1] >= maps || config[5] > 3
				|| config[6] > 3600 || config[7] < 1 || config[7] > 2160000 || config[4] > int.MaxValue) return Argument;
			for (uint i = 0; i < players; i++)
			{
				uint* p = config + 8 + i * 6;
				if (p[1] >= 6 || p[2] > 1 || p[0] > 255) return Argument;
				for (uint j = 0; j < slots; j++)
					if (p[3 + j] == 0 || p[3 + j] >= abilityCount) return Argument;
			}
			if (resetPending) return FailBroken("RL reset already in progress");
			if (broken) return Io;
			sampleValid = false; valid = false;
			// The socket path answers at reset_finish time; in-process the work
			// runs synchronously here and finish just reports the stashed result.
			resetResult = DoReset(config);
			resetPending = true;
			return Ok;
		}

		private static int ResetFinishImpl(void* world)
		{
			if (world == null || world != handle) return Argument;
			if (broken) return Io;
			if (!resetPending) return FailBroken("No RL reset in progress");
			resetPending = false;
			return resetResult;
		}

		// Mirrors RlBridge.Reset + the warmup half of RlBridge.PumpInner.
		private static int DoReset(uint* config)
		{
			try
			{
				uint seed = config[0];
				int level = (int)config[1], count = (int)config[2], slots = (int)config[3];
				int stocks = (int)config[4], mode = (int)config[5], startTicks = (int)config[6], maxTicks = (int)config[7];
				var setup = new ReplayArchive
				{
					seed = seed, level = (byte)level, stocks = stocks, gameMode = (byte)mode,
					sourceHash = ReplaySession.SourceHash, catalogHash = ReplaySession.CatalogHash,
					players = new ReplayPlayer[count],
				};
				var catalog = SteamManager.instance.abilityIcons.sprites;
				for (int i = 0; i < 6; i++)
				{
					uint* p = config + 8 + (uint)i * 6;
					var abilities = new ushort[slots];
					for (int j = 0; j < 3; j++)
					{
						int ability = (int)p[3 + j];
						if (i < count && j < slots)
						{
							if (ability < 0 || ability >= catalog.Count || catalog[ability].associatedGameObject == null
								|| catalog[ability].associatedGameObject.GetComponent<AbilityMonoBehaviour>() == null)
								throw new InvalidDataException("Unknown ability index");
							abilities[j] = (ushort)ability;
						}
					}
					if (i < count)
					{
						int color = (int)p[0], team = (int)p[1], keyboard = (int)p[2];
						if (color < 0 || color >= host.RlColorCount || team < 0 || team >= 6 || (keyboard != 0 && keyboard != 1))
							throw new InvalidDataException("Invalid player configuration");
						setup.players[i] = new ReplayPlayer
						{
							id = (byte)(i + 1), color = (byte)color, team = (byte)team,
							keyboard = keyboard != 0, abilities = abilities,
						};
					}
				}
				setup.frames.Add(new InputPacket[count]);
				setup.Validate();
				inputs = new InputPacket[count];
				started = terminal = failed = false;
				warmup = startTicks; episodeTicks = 0; episodeLimit = maxTicks;
				RlObservation.Reset();
				RlTelemetry.Reset();
				host.ResetRl(setup);
				setup.frames.Clear();
				recording = setup;
				// Pump returns after a reset command: the engine frame ends and
				// scene Starts run before the first warmup tick (the
				// loading/loadedFrame gate in RlBridge.PumpInner).
				FrameBoundary();
				// Warmup: RlBridge would frame-split at CommandsPerFrame ticks,
				// but warmup_ticks is capped at 3600 < 4096, so no split occurs;
				// frameTicks keeps accumulating into the following step commands.
				for (int i = 0; i < startTicks; i++) Tick(false);
				Observe(ref main);
				valid = true;
				SetError("");
				return Ok;
			}
			catch (Exception exception) { return GameFailed(exception); }
		}

		// Client-side action validation, verbatim from bopl_rl.c check_actions.
		private static int CheckActions(void* world, byte* actions, nuint count)
		{
			if (world == null || world != handle || closed || actions == null || !valid
				|| count != PlayerCount()) return Argument;
			if ((HeaderFlags() & 1u) != 0) return Ended; // BOPL_RL_TERMINAL
			for (nuint i = 0; i < count; i++)
			{
				byte reserved = actions[i * 4 + 1];
				ushort buttons = (ushort)(actions[i * 4 + 2] | (actions[i * 4 + 3] << 8));
				if (reserved != 0 || (buttons & ~1023) != 0) return Argument;
			}
			return Ok;
		}

		private static uint PlayerCount() => main.state == null ? 0 : *(uint*)(main.state + 16);
		private static uint HeaderFlags() => main.state == null ? 0 : *(uint*)(main.state + 12);

		// Mirrors RlBridge.ReadActions (which reads all six wire slots; the C
		// client sends exactly player_count actions).
		private static void ReadActions(byte* actions, int count)
		{
			for (int i = 0; i < count && i < inputs.Length; i++)
			{
				byte angle = actions[i * 4];
				ushort buttons = (ushort)(actions[i * 4 + 2] | (actions[i * 4 + 3] << 8));
				inputs[i] = ReplayArchive.Input((uint)(Updater.SimulationTicks + 1), angle, buttons);
			}
		}

		// Mirrors RlBridge.Tick, minus profiling.
		private static void Tick(bool counted)
		{
			var players = PlayerHandler.Get().PlayerList();
			for (int i = 0; i < inputs.Length; i++) inputs[i].seqNumber = (uint)(Updater.SimulationTicks + 1);
			if (recording != null) recording.frames.Add((InputPacket[])inputs.Clone());
			// BOPL_TRACE support. Unity exits at Finish (tick limit); in-process
			// another BeforeReplayStep would recreate and truncate the finished
			// file, hence the Completed guard (same as sim/Host).
			if (ReferenceCapture.Recording && !ReferenceCapture.Completed) ReferenceCapture.BeforeReplayStep(inputs);
			for (int i = 0; i < inputs.Length; i++) players[i].OverrideInputWithNetworkInput(inputs[i]);
			RlTelemetry.Counting = counted;
			try { Updater.TickSimulation(GameTime.FixedTimeStep); }
			catch (Exception exception)
			{
				failed = true;
				ReferenceCapture.FailedTick(exception);
				throw;
			}
			finally { RlTelemetry.Counting = false; }
			Updater.LoadedALevelThisUpdate = false;
			started |= !GameSessionHandler.HasGameEnded();
			if (counted) episodeTicks++;
			terminal = (started && GameSessionHandler.HasGameEnded()) || global::Host.goToNextReplay || episodeTicks >= episodeLimit;
			ticksThisFrame++;
		}

		// Engine frame boundary, same order as the sim/Host driver loop.
		private static void FrameBoundary()
		{
			World.Current.FlushDestroyQueue();
			World.Current.frameCount++;
			Time.frameCount = World.Current.frameCount;
			World.Current.FlushPendingStarts();
			ticksThisFrame = 0;
		}

		// RlBridge.Observe: one state record on the wire, then decoded into the
		// C-visible snapshot the way bopl_rl.c decode_snapshot does.
		private static void Observe(ref Snapshot snapshot)
		{
			wire.SetLength(0);
			uint flags = (terminal ? 1u : 0u) | (episodeTicks >= episodeLimit ? 2u : 0u)
				| (started ? 4u : 0u) | (failed ? 8u : 0u);
			RlObservation.Write(payload, Version, episodeTicks, flags, warmup);
			payload.Flush();
			DecodeInto(ref snapshot);
		}

		private static void DecodeInto(ref Snapshot snapshot)
		{
			byte[] bytes = wire.GetBuffer();
			int length = (int)wire.Length;
			uint playerCount = BitConverter.ToUInt32(bytes, 16);
			uint entityCount = BitConverter.ToUInt32(bytes, 20);
			uint slimeCount = BitConverter.ToUInt32(bytes, 28);
			int playerBytes = (int)playerCount * PlayerSize;
			int entityBytes = (int)entityCount * EntitySize;
			int slimeBytes = (int)slimeCount * SlimeSize;
			int telemetryBytes = (int)playerCount * TelemetrySize;
			if (playerCount < 1 || playerCount > 6
				|| HeaderSize + playerBytes + entityBytes + slimeBytes + telemetryBytes != length)
				throw new InvalidDataException("Invalid RL state layout");
			if (snapshot.state == null) snapshot.state = (byte*)NativeMemory.AllocZeroed(StateSize);
			if (entityCount > snapshot.entityCapacity)
			{
				snapshot.entities = (byte*)NativeMemory.Realloc(snapshot.entities, (nuint)entityBytes);
				snapshot.entityCapacity = entityCount;
			}
			if (slimeCount > snapshot.slimeCapacity)
			{
				snapshot.slimes = (byte*)NativeMemory.Realloc(snapshot.slimes, (nuint)slimeBytes);
				snapshot.slimeCapacity = slimeCount;
			}
			fixed (byte* source = bytes)
			{
				Buffer.MemoryCopy(source, snapshot.state, HeaderSize, HeaderSize);
				var players = new Span<byte>(snapshot.state + PlayersOffset, 6 * PlayerSize);
				players.Clear();
				new ReadOnlySpan<byte>(source + HeaderSize, playerBytes).CopyTo(players);
				if (entityBytes > 0)
					Buffer.MemoryCopy(source + HeaderSize + playerBytes, snapshot.entities, entityBytes, entityBytes);
				if (slimeBytes > 0)
					Buffer.MemoryCopy(source + HeaderSize + playerBytes + entityBytes, snapshot.slimes, slimeBytes, slimeBytes);
				var telemetry = new Span<byte>(snapshot.state + TelemetryOffset, 6 * TelemetrySize);
				telemetry.Clear();
				new ReadOnlySpan<byte>(source + HeaderSize + playerBytes + entityBytes + slimeBytes, telemetryBytes).CopyTo(telemetry);
			}
			*(byte**)(snapshot.state + EntitiesPtrOffset) = snapshot.entities;
			*(byte**)(snapshot.state + SlimesPtrOffset) = snapshot.slimes;
		}

		// A game-side exception: RlBridge.Error sends it as an error frame, and
		// bopl_rl.c receive_frame stores the text and invalidates the state.
		private static int GameFailed(Exception exception)
		{
			if (exception is not InvalidDataException && exception is not InvalidOperationException) failed = true;
			Debug.LogError("RL command failed at tick " + Updater.SimulationTicks + ": " + exception);
			SetError(exception.ToString());
			valid = false;
			return GameError;
		}

		// bopl_rl.c fail(): sticky transport failure. In-process these are only
		// the reset_begin/finish protocol misuses.
		private static int FailBroken(string message)
		{
			SetError(message);
			broken = true;
			valid = false;
			return Io;
		}

		// Mirrors BoplRl.error: 512 bytes including the terminator.
		private static void SetError(string message) => WriteCString(errorBuffer, 512, message);

		private static void ErrorText(byte* error, nuint capacity, string message) => WriteCString(error, capacity, message);

		private static void WriteCString(byte* target, nuint capacity, string message)
		{
			if (target == null || capacity == 0) return;
			byte[] bytes = Encoding.UTF8.GetBytes(message);
			nuint count = Math.Min((nuint)bytes.Length, capacity - 1);
			for (nuint i = 0; i < count; i++) target[i] = bytes[i];
			target[count] = 0;
		}

		private static byte* MakeCString(string value)
		{
			byte[] bytes = Encoding.UTF8.GetBytes(value);
			byte* result = (byte*)NativeMemory.Alloc((nuint)bytes.Length + 1);
			for (int i = 0; i < bytes.Length; i++) result[i] = bytes[i];
			result[bytes.Length] = 0;
			return result;
		}

		private static bool IsAbsolute(byte* path) => path != null && *path == (byte)'/';

		private static string ReadUtf8(byte* text) => Marshal.PtrToStringUTF8((IntPtr)text);

		// ---- bootstrap: same sequence as sim/Host Program.Bootstrap ----------

		private static global::Host Bootstrap(ExportDatabase db)
		{
			SceneManager.LoadScene(0);
			var bootHost = UnityEngine.Object.FindObjectOfType<global::Host>();
			if (bootHost == null) throw new InvalidOperationException("boot scene did not provide a Host (networkClient root)");
			// Host.Awake early-returns when process argv carries a .rep path;
			// re-apply its no-argv DontDestroyOnLoad calls (idempotent).
			UnityEngine.Object.DontDestroyOnLoad(bootHost.gameObject);
			var debugText = GetPrivateField(bootHost, "DebugTextRenderer") as UnityEngine.Object;
			if (debugText != null) UnityEngine.Object.DontDestroyOnLoad(debugText);
			World.Current.FlushPendingStarts();
			World.Current.FlushDestroyQueue();
			db.DumpWarnings("bootstrap");
			return bootHost;
		}

		private static object GetPrivateField(object target, string name)
		{
			for (var type = target.GetType(); type != null; type = type.BaseType)
			{
				var field = type.GetField(name, System.Reflection.BindingFlags.Instance
					| System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic
					| System.Reflection.BindingFlags.DeclaredOnly);
				if (field != null) return field.GetValue(target);
			}
			throw new MissingFieldException(target.GetType().Name, name);
		}

		private static string FindRepoRoot()
		{
			// BOPL_SIM_EXPORT points at the repo root explicitly (the .so may be
			// copied anywhere); otherwise walk up from the library and the cwd.
			string explicitRoot = Environment.GetEnvironmentVariable("BOPL_SIM_EXPORT");
			if (!string.IsNullOrEmpty(explicitRoot)) return explicitRoot;
			for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
				if (Directory.Exists(Path.Combine(dir.FullName, "native", "reference-export")))
					return dir.FullName;
			for (var dir = new DirectoryInfo(Directory.GetCurrentDirectory()); dir != null; dir = dir.Parent)
				if (Directory.Exists(Path.Combine(dir.FullName, "native", "reference-export")))
					return dir.FullName;
			throw new DirectoryNotFoundException("Cannot locate repo root (native/reference-export); set BOPL_SIM_EXPORT");
		}
	}
}
