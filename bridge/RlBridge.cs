using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Text;
using BoplFixedMath;
using UnityEngine;
using UnityEngine.SceneManagement;

// Opt-in transport and observation only. Gameplay runs in the original updater.
// One socket carries commands for N worlds (native/rl/MANY_WORLDS.md section 3):
// legacy ops 1-6 address world 0; op|0x100 carries a u32 world prefix. Replies of
// different worlds may interleave; replies of one world keep request order.
public static class RlBridge
{
    public const uint Magic = 0x314c5242; // BRL1, little endian
    public const uint Version = 2;
    public const uint WorldFlag = 0x100;
    private static readonly string endpoint = Environment.GetEnvironmentVariable("BOPL_RL_SOCKET");
    public static bool Enabled => !string.IsNullOrEmpty(endpoint);
    private static Socket socket;
    private static readonly MemoryStream response = new MemoryStream(65536);
    // Assemble each framed response before sending; command reads use our own buffer
    // so the pump can drain what is already available without blocking.
    private static byte[] sendBuffer = new byte[65536 + 16];
    private static byte[] receive = new byte[65536];
    private static int receiveStart, receiveEnd;
    private static readonly BinaryWriter payload = new BinaryWriter(response, Encoding.UTF8, true);
    private static RlWorld[] worlds;
    // Resets whose scene load has not been issued yet, in request order.
    private static readonly Queue<RlWorld> loadQueue = new Queue<RlWorld>();
    private static RlWorld loadingWorld;
    private static bool closed;
    // Gameplay is driven by simulation ticks. Yield periodically so Unity can
    // finish deferred object destruction and scene bookkeeping.
    private const int CommandsPerFrame = 4096;

    private static void Connect()
    {
        int count = 1;
        string configured = Environment.GetEnvironmentVariable("BOPL_RL_WORLDS");
        if (!string.IsNullOrEmpty(configured) && (!int.TryParse(configured, out count) || count < 1 || count > 64))
            throw new InvalidDataException("BOPL_RL_WORLDS must be 1..64");
        WorldState.Init(count);
        worlds = WorldState.Worlds;
        socket = new Socket(AddressFamily.Unix, SocketType.Stream, ProtocolType.Unspecified);
        socket.Connect(new UnixDomainSocketEndPoint(endpoint));
        Application.targetFrameRate = -1;
        QualitySettings.vSyncCount = 0;
        RlProfile.Install();
        SceneManager.sceneLoaded += OnSceneLoaded;
        response.SetLength(0);
        payload.Write(Version);
        payload.Write(Encoding.ASCII.GetBytes(ReplaySession.SourceHash));
        payload.Write(Encoding.ASCII.GetBytes(ReplaySession.CatalogHash));
        payload.Write(SceneManager.sceneCountInBuildSettings - 6);
        payload.Write(SteamManager.instance.abilityIcons.sprites.Count);
        payload.Write(count);
        Send(0, 0);
    }

    private static void OnSceneLoaded(Scene scene, LoadSceneMode mode)
    {
        var world = loadingWorld;
        if (world == null || !world.loading || scene.buildIndex < 6) return;
        WorldState.EndLoad(world, scene);
        world.loadedFrame = Time.frameCount;
        loadingWorld = null;
        RlProfile.Add("reset:scene", System.Diagnostics.Stopwatch.GetTimestamp() - world.loadStarted);
    }

    private static void Send(uint operation, uint sequence)
    {
        RlProfile.Enter("bridge:send");
        payload.Flush();
        int length = checked((int)response.Length), total = 16 + length;
        if (sendBuffer.Length < total) sendBuffer = new byte[Math.Max(total, sendBuffer.Length * 2)];
        WriteUInt32(sendBuffer, 0, Magic); WriteUInt32(sendBuffer, 4, operation);
        WriteUInt32(sendBuffer, 8, sequence); WriteUInt32(sendBuffer, 12, (uint)length);
        Buffer.BlockCopy(response.GetBuffer(), 0, sendBuffer, 16, length);
        for (int sent = 0; sent < total; ) sent += socket.Send(sendBuffer, sent, total - sent, SocketFlags.None);
    }

    private static void WriteUInt32(byte[] buffer, int offset, uint value)
    {
        buffer[offset] = (byte)value; buffer[offset + 1] = (byte)(value >> 8);
        buffer[offset + 2] = (byte)(value >> 16); buffer[offset + 3] = (byte)(value >> 24);
    }

    private static uint ReadUInt32(int offset)
    {
        return (uint)(receive[offset] | receive[offset + 1] << 8 | receive[offset + 2] << 16 | receive[offset + 3] << 24);
    }

    private static void Error(uint sequence, Exception exception)
    {
        Debug.LogError("RL command failed at tick " + Updater.SimulationTicks + ": " + exception);
        response.SetLength(0);
        payload.Write(Encoding.UTF8.GetBytes(exception.ToString()));
        Send(0xffffffff, sequence);
    }

    // True while some world's reply can only be produced after a Unity frame boundary
    // (a scene load in flight or queued, or a deferred step). Reads must not block then:
    // the trainer may be waiting on exactly that reply before sending anything else.
    private static bool OwesFrame()
    {
        if (loadQueue.Count > 0 || loadingWorld != null) return true;
        foreach (var world in worlds) if (world.loading || world.pendingStep != null) return true;
        return false;
    }

    // Fill the receive buffer until `needed` bytes are available. Returns false when
    // the pump must yield instead of blocking.
    private static bool Fill(int needed)
    {
        if (receiveEnd - receiveStart >= needed) return true;
        if (receiveStart > 0) {
            Buffer.BlockCopy(receive, receiveStart, receive, 0, receiveEnd - receiveStart);
            receiveEnd -= receiveStart; receiveStart = 0;
        }
        if (receive.Length < needed) Array.Resize(ref receive, Math.Max(needed, receive.Length * 2));
        while (receiveEnd < needed) {
            if (OwesFrame() && socket.Available == 0) return false;
            int got = socket.Receive(receive, receiveEnd, receive.Length - receiveEnd, SocketFlags.None);
            if (got <= 0) throw new EndOfStreamException();
            receiveEnd += got;
        }
        return true;
    }

    public static void Pump(Host host)
    {
        RlProfile.Enter("bridge:pump");
        try { PumpInner(host); }
        finally {
            // A scene load completes at the coming frame boundary with whatever world is
            // bound: keep the loading world bound so its Awake/Start land in it.
            if (loadingWorld != null && !closed) WorldState.Bind(loadingWorld);
            RlProfile.Enter("unity:frame");
        }
    }

    private static void PumpInner(Host host)
    {
        try {
            int frameTicks = 0;
            if (socket == null) {
                if (SceneManager.GetActiveScene().buildIndex < 6) return;
                Connect();
            }
            if (closed) return;
            // (1) Worlds whose scene loaded last frame: warmup ticks and the reset reply.
            foreach (var world in worlds) {
                if (!world.loading || world.loadedFrame < 0 || Time.frameCount <= world.loadedFrame) continue;
                WorldState.Bind(world);
                RlProfile.Enter("reset:warmup");
                try {
                    for (; frameTicks < CommandsPerFrame && world.remainingWarmup > 0; world.remainingWarmup--) { Tick(world, false); frameTicks++; }
                    if (world.remainingWarmup > 0) return;
                    world.loading = false; world.ready = true;
                    Observe(world); Send(world.resetOperation, world.resetSequence);
                } catch (Exception error) {
                    world.loading = world.ready = false; world.failed = true;
                    Error(world.resetSequence, error);
                }
                if (frameTicks >= CommandsPerFrame) return;
            }
            // (2) Next queued reset: issue its scene load (at most one per frame).
            if (IssueLoad(host)) return;
            // (3) Steps deferred by the per-frame tick budget.
            foreach (var world in worlds) {
                if (world.pendingStep == null) continue;
                byte[] bytes = world.pendingStep; world.pendingStep = null;
                WorldState.Bind(world);
                using var body = new BinaryReader(new MemoryStream(bytes, false));
                int before = Updater.SimulationTicks;
                try {
                    if ((world.pendingOperation & ~WorldFlag) == 6) StepSampled(world, body); else { Step(world, body); Observe(world); }
                    Send(world.pendingOperation, world.pendingSequence);
                }
                catch (Exception error) { Error(world.pendingSequence, error); }
                frameTicks += Updater.SimulationTicks - before;
                if (frameTicks >= CommandsPerFrame) return;
            }
            // (4) Commands.
            for (int command = 0; command < CommandsPerFrame; command++) {
                RlProfile.Enter("bridge:read");
                if (!Fill(16)) return;
                uint magic = ReadUInt32(receiveStart), operation = ReadUInt32(receiveStart + 4);
                uint sequence = ReadUInt32(receiveStart + 8), length = ReadUInt32(receiveStart + 12);
                if (magic != Magic || length > 4096) throw new InvalidDataException("Invalid RL command frame");
                if (!Fill(16 + (int)length)) return;
                byte[] bytes = new byte[length];
                Buffer.BlockCopy(receive, receiveStart + 16, bytes, 0, (int)length);
                receiveStart += 16 + (int)length;
                RlProfile.Enter("bridge:command");
                uint legacy = operation & ~WorldFlag;
                RlWorld world = worlds[0];
                int skip = 0;
                try {
                    if ((operation & WorldFlag) != 0 && legacy != 4) {
                        if (length < 4) throw new InvalidDataException("World command without world index");
                        uint index = BitConverter.ToUInt32(bytes, 0);
                        if (index >= (uint)worlds.Length) throw new InvalidDataException("World index out of range");
                        world = worlds[(int)index]; skip = 4;
                    } else if ((operation & WorldFlag) != 0) {
                        throw new InvalidDataException("Close takes no world index");
                    }
                    using var body = new BinaryReader(new MemoryStream(bytes, skip, bytes.Length - skip, false));
                    switch (legacy) {
                        case 1:
                            RlProfile.Enter("bridge:reset");
                            WorldState.Bind(world);
                            Reset(host, world, body, operation, sequence);
                            if (IssueLoad(host)) return;
                            break;
                        case 2:
                        case 6:
                            // Both phases of a sampled step (6) count like one step of first+second ticks.
                            if (world.loading || world.pendingStep != null)
                                throw new InvalidOperationException("World is busy with a previous command");
                            if (bytes.Length - skip >= 4 && frameTicks > 0 && RequestedTicks(legacy, bytes, skip) > CommandsPerFrame - frameTicks) {
                                world.pendingStep = Slice(bytes, skip); world.pendingSequence = sequence; world.pendingOperation = operation; return;
                            }
                            WorldState.Bind(world);
                            int before = Updater.SimulationTicks;
                            if (legacy == 6) StepSampled(world, body); else { Step(world, body); Observe(world); }
                            frameTicks += Updater.SimulationTicks - before;
                            Send(operation, sequence);
                            if (frameTicks >= CommandsPerFrame) return;
                            break;
                        case 3:
                            if (body.BaseStream.Length != 0 || !world.ready) throw new InvalidOperationException("Observation requires reset");
                            WorldState.Bind(world);
                            Observe(world); Send(operation, sequence); break;
                        case 4:
                            if (length != 0) throw new InvalidDataException("Invalid close payload");
                            response.SetLength(0); Send(operation, sequence);
                            RlProfile.Dump();
                            closed = true;
                            if (ReferenceCapture.Recording && !ReferenceCapture.Completed) ReferenceCapture.Finish("rl-closed");
                            else Application.Quit();
                            socket.Dispose(); return;
                        case 5:
                            if (body.BaseStream.Length != 0 || world.recording == null || world.recording.frames.Count == 0)
                                throw new InvalidOperationException("Replay requires at least one simulated tick");
                            response.SetLength(0); payload.Write(world.recording.Encode()); Send(operation, sequence); break;
                        default: throw new InvalidDataException("Unknown RL command");
                    }
                } catch (Exception error) {
                    if (error is not InvalidDataException && error is not InvalidOperationException) world.failed = true;
                    Error(sequence, error);
                }
            }
        } catch (Exception error) {
            Debug.LogError("RL transport stopped: " + error);
            Application.Quit(1);
        }
    }

    // Issues the scene load of the oldest queued reset. One per Unity frame: the load
    // completes at the frame boundary and the bound world must stay bound across it
    // so the scene's Awake/Start land in that world. Returns true when a load was issued.
    private static bool IssueLoad(Host host)
    {
        if (loadingWorld != null || loadQueue.Count == 0) return false;
        var world = loadQueue.Dequeue();
        WorldState.Bind(world);
        RlProfile.Enter("reset:load");
        loadingWorld = world;
        world.loadStarted = System.Diagnostics.Stopwatch.GetTimestamp();
        try { host.ResetRl(world.recording); world.recording.frames.Clear(); }
        catch (Exception error) {
            loadingWorld = null; world.loading = world.ready = false; world.failed = true;
            Error(world.resetSequence, error);
            return false;
        }
        return true;
    }

    private static byte[] Slice(byte[] bytes, int skip)
    {
        if (skip == 0) return bytes;
        var result = new byte[bytes.Length - skip];
        Buffer.BlockCopy(bytes, skip, result, 0, result.Length);
        return result;
    }

    // Parses and validates the reset, records it on the world and queues the scene
    // load. The load itself is issued by the pump, one world per Unity frame.
    private static void Reset(Host host, RlWorld world, BinaryReader body, uint operation, uint sequence)
    {
        if (world.loading || world.pendingStep != null) throw new InvalidOperationException("World is busy with a previous command");
        if (body.BaseStream.Length != 176) throw new InvalidDataException("Invalid reset payload length");
        uint seed = body.ReadUInt32(); int level = body.ReadInt32(); int count = body.ReadInt32(); int slots = body.ReadInt32();
        int stocks = body.ReadInt32(); int mode = body.ReadInt32(); int startTicks = body.ReadInt32(); int maxTicks = body.ReadInt32();
        if (count < 1 || count > 6 || slots < 1 || slots > 3 || level < 0 || level >= SceneManager.sceneCountInBuildSettings - 6
            || stocks < 0 || mode < 0 || mode > 3 || startTicks < 0 || startTicks > 3600 || maxTicks < 1 || maxTicks > ReplayArchive.MaxFrames)
            throw new InvalidDataException("Invalid RL configuration");
        var setup = new ReplayArchive { seed = seed, level = (byte)level, stocks = stocks, gameMode = (byte)mode,
            sourceHash = ReplaySession.SourceHash, catalogHash = ReplaySession.CatalogHash, players = new ReplayPlayer[count] };
        var catalog = SteamManager.instance.abilityIcons.sprites;
        for (int i = 0; i < 6; i++) {
            int color = body.ReadInt32(); int team = body.ReadInt32(); int keyboard = body.ReadInt32();
            var abilities = new ushort[slots];
            for (int j = 0; j < 3; j++) {
                int ability = body.ReadInt32();
                if (i < count && j < slots) {
                    if (ability < 0 || ability >= catalog.Count || catalog[ability].associatedGameObject == null
                        || catalog[ability].associatedGameObject.GetComponent<AbilityMonoBehaviour>() == null)
                        throw new InvalidDataException("Unknown ability index");
                    abilities[j] = (ushort)ability;
                }
            }
            if (i < count) {
                if (color < 0 || color >= host.RlColorCount || team < 0 || team >= 6 || (keyboard != 0 && keyboard != 1))
                    throw new InvalidDataException("Invalid player configuration");
                setup.players[i] = new ReplayPlayer { id = (byte)(i + 1), color = (byte)color, team = (byte)team,
                    keyboard = keyboard != 0, abilities = abilities };
            }
        }
        setup.frames.Add(new InputPacket[count]);
        setup.Validate();
        world.inputs = new InputPacket[count];
        world.loading = true; world.ready = world.started = world.terminal = world.failed = false;
        world.loadedFrame = -1; world.remainingWarmup = world.warmup = startTicks;
        world.episodeTicks = 0; world.episodeLimit = maxTicks;
        world.resetSequence = sequence; world.resetOperation = operation;
        world.recording = setup;
        RlObservation.Reset();
        RlTelemetry.Reset();
        loadQueue.Enqueue(world);
    }

    private static int RequestedTicks(uint operation, byte[] bytes, int skip)
    {
        int first = BitConverter.ToInt32(bytes, skip);
        return operation == 6 && bytes.Length - skip >= 8 ? first + BitConverter.ToInt32(bytes, skip + 4) : first;
    }

    private static void Step(RlWorld world, BinaryReader body)
    {
        if (!world.ready || world.failed) throw new InvalidOperationException("Reset required");
        if (world.terminal) throw new InvalidOperationException("Episode ended; reset required");
        if (body.BaseStream.Length != 28) throw new InvalidDataException("Invalid step payload length");
        int repeat = body.ReadInt32();
        if (repeat < 1 || repeat > 16) throw new InvalidDataException("Action repeat must be 1..16");
        ReadActions(world, body);
        for (int i = 0; i < repeat && !world.terminal; i++) Tick(world, true);
    }

    // One command for two consecutive phases of the same actions. The response
    // holds the observation after the first phase and, unless the episode ended
    // during it, the observation after the second phase. Ticks run exactly as two
    // Step commands would.
    private static void StepSampled(RlWorld world, BinaryReader body)
    {
        if (!world.ready || world.failed) throw new InvalidOperationException("Reset required");
        if (world.terminal) throw new InvalidOperationException("Episode ended; reset required");
        if (body.BaseStream.Length != 32) throw new InvalidDataException("Invalid sampled step payload length");
        int first = body.ReadInt32(); int second = body.ReadInt32();
        if (first < 1 || first > 16 || second < 1 || second > 16 || first + second > 16)
            throw new InvalidDataException("Sampled step phases must be 1..16 ticks and at most 16 together");
        ReadActions(world, body);
        for (int i = 0; i < first && !world.terminal; i++) Tick(world, true);
        Observe(world);
        if (world.terminal) return;
        for (int i = 0; i < second && !world.terminal; i++) Tick(world, true);
        Observe(world, true);
    }

    private static void ReadActions(RlWorld world, BinaryReader body)
    {
        var inputs = world.inputs;
        for (int i = 0; i < 6; i++) {
            byte angle = body.ReadByte(); byte reserved = body.ReadByte(); ushort buttons = body.ReadUInt16();
            if (reserved != 0 || (buttons & ~1023) != 0) throw new InvalidDataException("Invalid action bits");
            if (i < inputs.Length) inputs[i] = ReplayArchive.Input((uint)(Updater.SimulationTicks + 1), angle, buttons);
        }
    }

    // Caller has bound `world`.
    private static void Tick(RlWorld world, bool counted)
    {
        RlProfile.Enter("tick:pre");
        var inputs = world.inputs;
        var players = PlayerHandler.Get().PlayerList();
        for (int i = 0; i < inputs.Length; i++) inputs[i].seqNumber = (uint)(Updater.SimulationTicks + 1);
        if (world.recording != null) world.recording.frames.Add((InputPacket[])inputs.Clone());
        if (ReferenceCapture.Recording) ReferenceCapture.BeforeReplayStep(inputs);
        for (int i = 0; i < inputs.Length; i++) players[i].OverrideInputWithNetworkInput(inputs[i]);
        RlTelemetry.Counting = counted;
        try { Updater.TickSimulation(GameTime.FixedTimeStep); }
        catch (Exception error) { world.failed = true; ReferenceCapture.FailedTick(error); throw; }
        finally { RlTelemetry.Counting = false; RlProfile.Enter("tick:post"); RlProfile.Ticked(counted); }
        Updater.LoadedALevelThisUpdate = false;
        world.started |= !GameSessionHandler.HasGameEnded();
        if (counted) world.episodeTicks++;
        world.terminal = (world.started && GameSessionHandler.HasGameEnded()) || Host.goToNextReplay || world.episodeTicks >= world.episodeLimit;
    }

    private static void Observe(RlWorld world, bool append = false)
    {
        RlProfile.Enter("bridge:observe");
        if (!append) response.SetLength(0);
        uint flags = (world.terminal ? 1u : 0u) | (world.episodeTicks >= world.episodeLimit ? 2u : 0u) | (world.started ? 4u : 0u) | (world.failed ? 8u : 0u);
        RlObservation.Write(payload, Version, world.episodeTicks, flags, world.warmup);
    }
}
