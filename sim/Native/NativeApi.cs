using System;
using System.Collections.Concurrent;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
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
	// Many worlds per process (SIM_OVERHAUL.md section 3): every bopl_rl_open
	// creates an RlWorld (its own UnityShim.World, boot scene and Host); the
	// export database is loaded once. Gameplay's per-match statics are
	// [ThreadStatic] and swapped by World.Bind/Unbind (RlWorldState), so calls
	// on different handles may run on different threads at the same time -
	// the trainer's step threads and the reset helper threads in bopl_rl.c do
	// exactly that. A handle is single-threaded: two concurrent calls on the
	// same handle are rejected with BOPL_RL_IO.
	//
	// BOPL_TRACE keeps working: Tick calls ReferenceCapture.BeforeReplayStep
	// exactly like RlBridge.Tick (trace one world per process; the capture is
	// static).
	public static unsafe class NativeApi
	{
		private const uint Version = 2; // RlBridge.Version
		// RlBridge.CommandsPerFrame: simulation ticks per engine frame. Raw
		// Object.Destroy drains at frame end and queued Starts run at the top of
		// the next frame, so the cadence is gameplay-observable and must match
		// the Unity RL runtime (not the 16-tick cadence of the trace fixtures).
		private const int CommandsPerFrame = 4096;

		private const int Ok = 0, Argument = 1, Io = 2, GameError = 3, Ended = 4;

		// Soft cap on worlds per process, reported by bopl_rl_world_capacity.
		private const uint WorldCapacity = 4096;

		// ---- process-wide ----------------------------------------------------
		private static readonly object bootGate = new object();
		private static ExportDatabase db;
		private static bool booted;
		private static byte* nullHandleText;   // "NULL RL handle"
		private static byte* sourceHashText, catalogHashText;
		private static uint maps, abilityCount;
		private static readonly ConcurrentDictionary<IntPtr, RlWorld> worlds = new ConcurrentDictionary<IntPtr, RlWorld>();

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

		// One RL world: a UnityShim.World with its boot scene + Host, and the
		// episode state RlBridge keeps in private fields.
		private sealed class RlWorld
		{
			public readonly World world = new World();
			public global::Host host;
			public IntPtr handle;
			public int busy;                       // 1 while a thread is inside a call
			public bool closed, broken;
			public readonly byte* errorBuffer = (byte*)NativeMemory.AllocZeroed(512); // mirrors BoplRl.error

			public bool valid, sampleValid, started, terminal, failed, resetPending;
			public int warmup, episodeTicks, episodeLimit, ticksThisFrame, resetResult;
			public InputPacket[] inputs;
			public ReplayArchive recording;

			public readonly MemoryStream wire = new MemoryStream(65536);
			public readonly BinaryWriter payload;
			public Snapshot main, sample;
			public byte* replayBuffer;
			public nuint replayCapacity;

			public RlWorld() { payload = new BinaryWriter(wire, Encoding.UTF8, true); }
		}

		// Scope of one ABI call on a world: binds the thread to it (gameplay
		// statics restored) and releases on dispose (statics captured).
		private readonly ref struct Scope
		{
			private readonly RlWorld w;
			public Scope(RlWorld w) { this.w = w; World.Bind(w.world); }
			public void Dispose() { World.Unbind(); Volatile.Write(ref w.busy, 0); }
		}

		private static RlWorld Lookup(void* handle)
			=> handle != null && worlds.TryGetValue((IntPtr)handle, out var w) ? w : null;

		// Acquire the world for a call. Null with `status` set on a bad handle,
		// a closed world or a concurrent call on the same handle.
		private static RlWorld Acquire(void* handle, out int status)
		{
			var w = Lookup(handle);
			if (w == null || w.closed) { status = Argument; return null; }
			if (Interlocked.CompareExchange(ref w.busy, 1, 0) != 0)
			{
				// Not Scope-guarded (no bind): set the text directly.
				WriteCString(w.errorBuffer, 512, "RL world is busy on another thread");
				w.broken = true;
				status = Io;
				return null;
			}
			status = Ok;
			return w;
		}

		// ================= exports =================

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_abi_version")]
		public static uint AbiVersion() => Version;

		// Many worlds per process: how many bopl_rl_open calls this library
		// serves (the socket loader and older .so builds have no such export:
		// one world per process).
		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_world_capacity")]
		public static uint WorldCapacityExport() => WorldCapacity;

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
			if ((uint)worlds.Count >= WorldCapacity)
			{
				ErrorText(error, capacity, "RL world capacity reached (" + WorldCapacity + " per process)");
				return null;
			}
			try
			{
				lock (bootGate)
				{
					if (!booted)
					{
						// -logFile equivalent: the shim's Debug.Log* writes to stderr,
						// which belongs to the training process here. One stream per
						// process: the first world's log receives every world's output.
						Console.SetError(new StreamWriter(ReadUtf8(logPath), false) { AutoFlush = true });
						// RlTelemetry.Enabled => RlBridge.Enabled && Counting, and
						// RlBridge.Enabled reads BOPL_RL_SOCKET once. No socket exists
						// in-process; set the flag so telemetry counts like the socket
						// path. The only other consumers (Host.Update -> Pump,
						// Host.OnLevelWasLoaded) are never dispatched by the shim host.
						if (Environment.GetEnvironmentVariable("BOPL_RL_SOCKET") == null)
							Environment.SetEnvironmentVariable("BOPL_RL_SOCKET", "in-process");
						RlWorldState.Attach();
						string repoRoot = FindRepoRoot();
						db = ExportDatabase.Load(
							Path.Combine(repoRoot, "native", "reference-export"),
							Path.Combine(repoRoot, "sim", "Export", "settings.json"),
							repoRoot);
						booted = true;
					}
				}

				var w = new RlWorld();
				w.busy = 1;
				using (new Scope(w))
				{
					db.Install();
					w.host = Bootstrap(db);
					lock (bootGate)
					{
						if (sourceHashText == null)
						{
							maps = (uint)(SceneManager.sceneCountInBuildSettings - 6);
							abilityCount = (uint)SteamManager.instance.abilityIcons.sprites.Count;
							catalogHashText = MakeCString(ReplaySession.CatalogHash);
							sourceHashText = MakeCString(ReplaySession.SourceHash);
						}
					}
				}
				var gc = GCHandle.Alloc(w);
				w.handle = GCHandle.ToIntPtr(gc);
				worlds[w.handle] = w;
				ErrorText(error, capacity, "");
				return (void*)w.handle;
			}
			catch (Exception exception)
			{
				ErrorText(error, capacity, "RL world bootstrap failed: " + exception);
				return null;
			}
		}

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_reset_begin")]
		public static int ResetBegin(void* world, uint* config)
		{
			var w = Acquire(world, out int status);
			if (w == null) return status;
			using (new Scope(w)) return ResetBeginImpl(w, config);
		}

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_reset_finish")]
		public static int ResetFinish(void* world)
		{
			var w = Acquire(world, out int status);
			if (w == null) return status;
			using (new Scope(w)) return ResetFinishImpl(w);
		}

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_reset_pending")]
		public static int ResetPending(void* world)
		{
			var w = Lookup(world);
			return w != null && w.resetPending ? 1 : 0;
		}

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_reset")]
		public static int Reset(void* world, uint* config)
		{
			var w = Acquire(world, out int status);
			if (w == null) return status;
			using (new Scope(w))
			{
				int result = ResetBeginImpl(w, config);
				return result != Ok ? result : ResetFinishImpl(w);
			}
		}

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_step")]
		public static int Step(void* world, byte* actions, nuint count, uint repeat)
		{
			var w = Acquire(world, out int status);
			if (w == null) return status;
			using (new Scope(w))
			{
				int check = CheckActions(w, actions, count);
				if (check != Ok) return check;
				if (repeat < 1 || repeat > 16) return Argument;
				if (w.broken) return Io;
				w.sampleValid = false;
				try
				{
					// RlBridge defers a step that would cross the remaining frame
					// budget (pendingStep): the engine frame ends before the step,
					// never inside it.
					if (w.ticksThisFrame > 0 && w.ticksThisFrame + repeat > CommandsPerFrame) FrameBoundary(w);
					ReadActions(w, actions, (int)count);
					for (int i = 0; i < repeat && !w.terminal; i++) Tick(w, true);
					Observe(w, ref w.main);
					if (w.ticksThisFrame >= CommandsPerFrame) FrameBoundary(w);
					w.valid = true;
					SetError(w, "");
					return Ok;
				}
				catch (Exception exception) { return GameFailed(w, exception); }
			}
		}

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_step_sampled")]
		public static int StepSampled(void* world, byte* actions, nuint count, uint first, uint second)
		{
			var w = Acquire(world, out int status);
			if (w == null) return status;
			using (new Scope(w))
			{
				int check = CheckActions(w, actions, count);
				if (check != Ok) return check;
				if (first < 1 || first > 16 || second < 1 || second > 16 || first + second > 16) return Argument;
				if (w.broken) return Io;
				w.sampleValid = false;
				try
				{
					if (w.ticksThisFrame > 0 && w.ticksThisFrame + first + second > CommandsPerFrame) FrameBoundary(w);
					ReadActions(w, actions, (int)count);
					for (int i = 0; i < first && !w.terminal; i++) Tick(w, true);
					Observe(w, ref w.sample);
					if (w.terminal)
					{
						// Episode ended during the first phase: sample and final
						// state are identical (bopl_rl.c copy_snapshot).
						DecodeInto(w, ref w.main);
					}
					else
					{
						for (int i = 0; i < second && !w.terminal; i++) Tick(w, true);
						Observe(w, ref w.main);
					}
					if (w.ticksThisFrame >= CommandsPerFrame) FrameBoundary(w);
					w.valid = true; w.sampleValid = true;
					SetError(w, "");
					return Ok;
				}
				catch (Exception exception) { return GameFailed(w, exception); }
			}
		}

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_state")]
		public static void* State(void* world)
		{
			var w = Lookup(world);
			return w != null && w.valid ? w.main.state : null;
		}

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_sample")]
		public static void* Sample(void* world)
		{
			var w = Lookup(world);
			return w != null && w.valid && w.sampleValid ? w.sample.state : null;
		}

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_replay")]
		public static int Replay(void* world, void** bytes, nuint* length)
		{
			if (bytes == null || length == null) return Argument;
			var w = Acquire(world, out int status);
			if (w == null) return status;
			using (new Scope(w))
			{
				if (w.broken) return Io;
				try
				{
					// RlBridge case 5 throws this as InvalidOperationException.
					if (w.recording == null || w.recording.frames.Count == 0)
						throw new InvalidOperationException("Replay requires at least one simulated tick");
					byte[] encoded = w.recording.Encode();
					if ((nuint)encoded.Length > w.replayCapacity)
					{
						w.replayBuffer = (byte*)NativeMemory.Realloc(w.replayBuffer, (nuint)encoded.Length);
						w.replayCapacity = (nuint)encoded.Length;
					}
					Marshal.Copy(encoded, 0, (IntPtr)w.replayBuffer, encoded.Length);
					*bytes = w.replayBuffer;
					*length = (nuint)encoded.Length;
					SetError(w, "");
					return Ok;
				}
				catch (Exception exception) { return GameFailed(w, exception); }
			}
		}

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_error")]
		public static byte* Error(void* world)
		{
			var w = Lookup(world);
			if (w != null) return w.errorBuffer;
			if (nullHandleText == null) nullHandleText = MakeCString("NULL RL handle");
			return nullHandleText;
		}

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_source_hash")]
		public static byte* SourceHash(void* world) => Lookup(world) != null ? sourceHashText : null;

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_catalog_hash")]
		public static byte* CatalogHash(void* world) => Lookup(world) != null ? catalogHashText : null;

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_map_count")]
		public static uint MapCount(void* world) => Lookup(world) != null ? maps : 0;

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_ability_count")]
		public static uint AbilityCount(void* world) => Lookup(world) != null ? abilityCount : 0;

		[UnmanagedCallersOnly(EntryPoint = "bopl_rl_close")]
		public static void Close(void* world)
		{
			var w = Acquire(world, out _);
			if (w == null) return;
			using (new Scope(w))
			{
				w.closed = true;
				w.resetPending = false;
				w.valid = w.sampleValid = false;
				// RlBridge case 4: finish an active trace instead of quitting.
				try
				{
					if (ReferenceCapture.Recording && !ReferenceCapture.Completed) ReferenceCapture.Finish("rl-closed");
				}
				catch (Exception exception) { Debug.LogError("RL close: " + exception); }
			}
			// The handle stays valid (error text readable); the world's objects
			// are dropped so the memory returns to the process.
			w.host = null;
			w.recording = null;
			w.world.gameplayState = null;
		}

		// ================= internals =================

		private static int ResetBeginImpl(RlWorld w, uint* config)
		{
			if (config == null) return Argument;
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
			if (w.resetPending) return FailBroken(w, "RL reset already in progress");
			if (w.broken) return Io;
			w.sampleValid = false; w.valid = false;
			// The socket path answers at reset_finish time; in-process the work
			// runs synchronously here and finish just reports the stashed result.
			w.resetResult = DoReset(w, config);
			w.resetPending = true;
			return Ok;
		}

		private static int ResetFinishImpl(RlWorld w)
		{
			if (w.broken) return Io;
			if (!w.resetPending) return FailBroken(w, "No RL reset in progress");
			w.resetPending = false;
			return w.resetResult;
		}

		// Mirrors RlBridge.Reset + the warmup half of RlBridge.PumpInner.
		private static int DoReset(RlWorld w, uint* config)
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
						if (color < 0 || color >= w.host.RlColorCount || team < 0 || team >= 6 || (keyboard != 0 && keyboard != 1))
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
				w.inputs = new InputPacket[count];
				w.started = w.terminal = w.failed = false;
				w.warmup = startTicks; w.episodeTicks = 0; w.episodeLimit = maxTicks;
				RlObservation.Reset();
				RlTelemetry.Reset();
				w.host.ResetRl(setup);
				setup.frames.Clear();
				w.recording = setup;
				// Pump returns after a reset command: the engine frame ends and
				// scene Starts run before the first warmup tick (the
				// loading/loadedFrame gate in RlBridge.PumpInner).
				FrameBoundary(w);
				// Warmup: RlBridge would frame-split at CommandsPerFrame ticks,
				// but warmup_ticks is capped at 3600 < 4096, so no split occurs;
				// frameTicks keeps accumulating into the following step commands.
				for (int i = 0; i < startTicks; i++) Tick(w, false);
				Observe(w, ref w.main);
				w.valid = true;
				SetError(w, "");
				return Ok;
			}
			catch (Exception exception) { return GameFailed(w, exception); }
		}

		// Client-side action validation, verbatim from bopl_rl.c check_actions.
		private static int CheckActions(RlWorld w, byte* actions, nuint count)
		{
			if (actions == null || !w.valid || count != PlayerCount(w)) return Argument;
			if ((HeaderFlags(w) & 1u) != 0) return Ended; // BOPL_RL_TERMINAL
			for (nuint i = 0; i < count; i++)
			{
				byte reserved = actions[i * 4 + 1];
				ushort buttons = (ushort)(actions[i * 4 + 2] | (actions[i * 4 + 3] << 8));
				if (reserved != 0 || (buttons & ~1023) != 0) return Argument;
			}
			return Ok;
		}

		private static uint PlayerCount(RlWorld w) => w.main.state == null ? 0 : *(uint*)(w.main.state + 16);
		private static uint HeaderFlags(RlWorld w) => w.main.state == null ? 0 : *(uint*)(w.main.state + 12);

		// Mirrors RlBridge.ReadActions (which reads all six wire slots; the C
		// client sends exactly player_count actions).
		private static void ReadActions(RlWorld w, byte* actions, int count)
		{
			for (int i = 0; i < count && i < w.inputs.Length; i++)
			{
				byte angle = actions[i * 4];
				ushort buttons = (ushort)(actions[i * 4 + 2] | (actions[i * 4 + 3] << 8));
				w.inputs[i] = ReplayArchive.Input((uint)(Updater.SimulationTicks + 1), angle, buttons);
			}
		}

		// Mirrors RlBridge.Tick, minus profiling.
		private static void Tick(RlWorld w, bool counted)
		{
			var players = PlayerHandler.Get().PlayerList();
			var inputs = w.inputs;
			for (int i = 0; i < inputs.Length; i++) inputs[i].seqNumber = (uint)(Updater.SimulationTicks + 1);
			if (w.recording != null) w.recording.frames.Add((InputPacket[])inputs.Clone());
			// BOPL_TRACE support. Unity exits at Finish (tick limit); in-process
			// another BeforeReplayStep would recreate and truncate the finished
			// file, hence the Completed guard (same as sim/Host).
			if (ReferenceCapture.Recording && !ReferenceCapture.Completed) ReferenceCapture.BeforeReplayStep(inputs);
			for (int i = 0; i < inputs.Length; i++) players[i].OverrideInputWithNetworkInput(inputs[i]);
			RlTelemetry.Counting = counted;
			try { Updater.TickSimulation(GameTime.FixedTimeStep); }
			catch (Exception exception)
			{
				w.failed = true;
				ReferenceCapture.FailedTick(exception);
				throw;
			}
			finally { RlTelemetry.Counting = false; }
			Updater.LoadedALevelThisUpdate = false;
			w.started |= !GameSessionHandler.HasGameEnded();
			if (counted) w.episodeTicks++;
			w.terminal = (w.started && GameSessionHandler.HasGameEnded()) || global::Host.goToNextReplay || w.episodeTicks >= w.episodeLimit;
			w.ticksThisFrame++;
		}

		// Engine frame boundary, same order as the sim/Host driver loop.
		private static void FrameBoundary(RlWorld w)
		{
			var world = w.world;
			world.FlushDestroyQueue();
			world.frameCount++;
			Time.frameCount = world.frameCount;
			world.FlushPendingStarts();
			w.ticksThisFrame = 0;
		}

		// RlBridge.Observe: one state record on the wire, then decoded into the
		// C-visible snapshot the way bopl_rl.c decode_snapshot does.
		private static void Observe(RlWorld w, ref Snapshot snapshot)
		{
			w.wire.SetLength(0);
			uint flags = (w.terminal ? 1u : 0u) | (w.episodeTicks >= w.episodeLimit ? 2u : 0u)
				| (w.started ? 4u : 0u) | (w.failed ? 8u : 0u);
			RlObservation.Write(w.payload, Version, w.episodeTicks, flags, w.warmup);
			w.payload.Flush();
			DecodeInto(w, ref snapshot);
		}

		private static void DecodeInto(RlWorld w, ref Snapshot snapshot)
		{
			byte[] bytes = w.wire.GetBuffer();
			int length = (int)w.wire.Length;
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
		private static int GameFailed(RlWorld w, Exception exception)
		{
			if (exception is not InvalidDataException && exception is not InvalidOperationException) w.failed = true;
			Debug.LogError("RL command failed at tick " + Updater.SimulationTicks + ": " + exception);
			SetError(w, exception.ToString());
			w.valid = false;
			return GameError;
		}

		// bopl_rl.c fail(): sticky transport failure. In-process these are only
		// the reset_begin/finish protocol misuses.
		private static int FailBroken(RlWorld w, string message)
		{
			SetError(w, message);
			w.broken = true;
			w.valid = false;
			return Io;
		}

		// Mirrors BoplRl.error: 512 bytes including the terminator.
		private static void SetError(RlWorld w, string message) => WriteCString(w.errorBuffer, 512, message);

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
