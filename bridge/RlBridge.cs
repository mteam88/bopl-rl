using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Text;
using BoplFixedMath;
using UnityEngine;
using UnityEngine.SceneManagement;

// Opt-in transport and observation only. Gameplay runs in the original updater.
public static class RlBridge
{
    public const uint Magic = 0x314c5242; // BRL1, little endian
    public const uint Version = 2;
    private static readonly string endpoint = Environment.GetEnvironmentVariable("BOPL_RL_SOCKET");
    public static bool Enabled => !string.IsNullOrEmpty(endpoint);
    private static Socket socket;
    private static BinaryReader reader;
    private static BinaryWriter writer;
    private static readonly MemoryStream response = new MemoryStream(65536);
    // BOPL_RL_COALESCE=1: one socket write per response (header and body
    // together) and buffered command reads, instead of five small writes and
    // five small reads per step. The byte stream is unchanged.
    private static readonly bool coalesce = Environment.GetEnvironmentVariable("BOPL_RL_COALESCE") == "1";
    private static byte[] sendBuffer = new byte[65536 + 16];
    private static readonly BinaryWriter payload = new BinaryWriter(response, Encoding.UTF8, true);
    private static bool loading, ready, started, terminal, failed;
    private static int warmup, remainingWarmup, episodeTicks, episodeLimit;
    private static uint resetSequence;
    private static InputPacket[] inputs;
    private static int loadedFrame;
    private static byte[] pendingStep;
    private static uint pendingSequence, pendingOperation;
    private static ReplayArchive recording;
    // Simulation ticks the bridge runs before returning to Unity for a frame. The
    // original 16 costs one Unity frame (~3 ms of Update/LateUpdate/animator work
    // with nothing to render) every 16 ticks; gameplay is frame-independent by
    // design (lockstep multiplayer), so a larger value only changes when Unity's
    // deferred bookkeeping runs. BOPL_RL_TICKS_PER_FRAME overrides it.
    private static readonly int CommandsPerFrame = TicksPerFrame();

    private static int TicksPerFrame()
    {
        string text = Environment.GetEnvironmentVariable("BOPL_RL_TICKS_PER_FRAME");
        return int.TryParse(text, out int ticks) && ticks >= 16 && ticks <= 65536 ? ticks : 16;
    }

    private static void Connect()
    {
        socket = new Socket(AddressFamily.Unix, SocketType.Stream, ProtocolType.Unspecified);
        socket.Connect(new UnixDomainSocketEndPoint(endpoint));
        var stream = new NetworkStream(socket, true);
        reader = new BinaryReader(coalesce ? new BufferedStream(stream, 65536) : (Stream)stream, Encoding.UTF8, true);
        writer = new BinaryWriter(stream, Encoding.UTF8, true);
        Application.targetFrameRate = -1;
        QualitySettings.vSyncCount = 0;
        RlProfile.Install();
        SceneManager.sceneLoaded += (_, mode) => {
            if (loading && mode == LoadSceneMode.Single) loadedFrame = Time.frameCount;
        };
        response.SetLength(0);
        payload.Write(Version);
        payload.Write(Encoding.ASCII.GetBytes(ReplaySession.SourceHash));
        payload.Write(Encoding.ASCII.GetBytes(ReplaySession.CatalogHash));
        payload.Write(SceneManager.sceneCountInBuildSettings - 6);
        payload.Write(SteamManager.instance.abilityIcons.sprites.Count);
        Send(0, 0);
    }

    private static void Send(uint operation, uint sequence)
    {
        RlProfile.Enter("bridge:send");
        payload.Flush();
        if (coalesce) {
            int length = checked((int)response.Length), total = 16 + length;
            if (sendBuffer.Length < total) sendBuffer = new byte[Math.Max(total, sendBuffer.Length * 2)];
            WriteUInt32(sendBuffer, 0, Magic); WriteUInt32(sendBuffer, 4, operation);
            WriteUInt32(sendBuffer, 8, sequence); WriteUInt32(sendBuffer, 12, (uint)length);
            Buffer.BlockCopy(response.GetBuffer(), 0, sendBuffer, 16, length);
            for (int sent = 0; sent < total; ) sent += socket.Send(sendBuffer, sent, total - sent, SocketFlags.None);
            return;
        }
        writer.Write(Magic); writer.Write(operation); writer.Write(sequence); writer.Write((uint)response.Length);
        writer.Write(response.GetBuffer(), 0, checked((int)response.Length)); writer.Flush();
    }

    private static void WriteUInt32(byte[] buffer, int offset, uint value)
    {
        buffer[offset] = (byte)value; buffer[offset + 1] = (byte)(value >> 8);
        buffer[offset + 2] = (byte)(value >> 16); buffer[offset + 3] = (byte)(value >> 24);
    }

    private static void Error(uint sequence, Exception exception)
    {
        Debug.LogError("RL command failed at tick " + Updater.SimulationTicks + ": " + exception);
        response.SetLength(0);
        payload.Write(Encoding.UTF8.GetBytes(exception.ToString()));
        Send(0xffffffff, sequence);
    }

    public static void Pump(Host host)
    {
        RlProfile.Enter("bridge:pump");
        try { PumpInner(host); }
        finally { RlProfile.Enter("unity:frame"); }
    }

    private static void PumpInner(Host host)
    {
        try {
            int frameTicks = 0;
            if (socket == null) Connect();
            if (loading) {
                // A LoadScene request completes after returning to Unity. Give all
                // scene Start callbacks a frame before the first simulation tick.
                if (loadedFrame < 0 || Time.frameCount <= loadedFrame) return;
                try {
                    for (int i = 0; i < CommandsPerFrame && remainingWarmup > 0; i++, remainingWarmup--) { Tick(false); frameTicks++; }
                    if (remainingWarmup > 0) return;
                    loading = false; ready = true;
                    Observe(); Send(1, resetSequence);
                } catch (Exception error) {
                    loading = ready = false; failed = true;
                    Error(resetSequence, error); return;
                }
            }
            if (frameTicks >= CommandsPerFrame) return;
            if (pendingStep != null) {
                byte[] bytes = pendingStep; pendingStep = null;
                using var body = new BinaryReader(new MemoryStream(bytes, false));
                int before = Updater.SimulationTicks;
                try {
                    if (pendingOperation == 6) StepSampled(body); else { Step(body); Observe(); }
                    Send(pendingOperation, pendingSequence);
                }
                catch (Exception error) { Error(pendingSequence, error); }
                frameTicks += Updater.SimulationTicks - before;
                if (frameTicks >= CommandsPerFrame) return;
            }
            for (int command = 0; command < CommandsPerFrame; command++) {
                RlProfile.Enter("bridge:read");
                uint magic = reader.ReadUInt32(); uint operation = reader.ReadUInt32();
                uint sequence = reader.ReadUInt32(); uint length = reader.ReadUInt32();
                if (magic != Magic || length > 4096) throw new InvalidDataException("Invalid RL command frame");
                byte[] bytes = reader.ReadBytes((int)length);
                if (bytes.Length != length) throw new EndOfStreamException();
                RlProfile.Enter("bridge:command");
                using var body = new BinaryReader(new MemoryStream(bytes, false));
                try {
                    switch (operation) {
                        case 1: RlProfile.Enter("bridge:reset"); Reset(host, body, sequence); return;
                        case 2:
                        case 6:
                            // Both phases of a sampled step (6) count like one step of first+second ticks.
                            if (length >= 4 && frameTicks > 0 && RequestedTicks(operation, bytes) > CommandsPerFrame - frameTicks) {
                                pendingStep = bytes; pendingSequence = sequence; pendingOperation = operation; return;
                            }
                            int before = Updater.SimulationTicks;
                            if (operation == 6) StepSampled(body); else { Step(body); Observe(); }
                            frameTicks += Updater.SimulationTicks - before;
                            Send(operation, sequence);
                            if (frameTicks >= CommandsPerFrame) return;
                            break;
                        case 3:
                            if (length != 0 || !ready) throw new InvalidOperationException("Observation requires reset");
                            Observe(); Send(operation, sequence); break;
                        case 4:
                            if (length != 0) throw new InvalidDataException("Invalid close payload");
                            response.SetLength(0); Send(operation, sequence);
                            RlProfile.Dump();
                            if (ReferenceCapture.Recording && !ReferenceCapture.Completed) ReferenceCapture.Finish("rl-closed");
                            else Application.Quit();
                            socket.Dispose(); return;
                        case 5:
                            if (length != 0 || recording == null || recording.frames.Count == 0)
                                throw new InvalidOperationException("Replay requires at least one simulated tick");
                            response.SetLength(0); payload.Write(recording.Encode()); Send(operation, sequence); break;
                        default: throw new InvalidDataException("Unknown RL command");
                    }
                } catch (Exception error) {
                    if (error is not InvalidDataException && error is not InvalidOperationException) failed = true;
                    Error(sequence, error);
                }
            }
        } catch (Exception error) {
            Debug.LogError("RL transport stopped: " + error);
            Application.Quit(1);
        }
    }

    private static void Reset(Host host, BinaryReader body, uint sequence)
    {
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
        inputs = new InputPacket[count];
        loading = true; ready = started = terminal = failed = false;
        loadedFrame = -1; remainingWarmup = warmup = startTicks;
        episodeTicks = 0; episodeLimit = maxTicks; resetSequence = sequence;
        RlObservation.Reset();
        RlTelemetry.Reset();
        host.ResetRl(setup);
        setup.frames.Clear(); recording = setup;
    }

    private static int RequestedTicks(uint operation, byte[] bytes)
    {
        int first = BitConverter.ToInt32(bytes, 0);
        return operation == 6 && bytes.Length >= 8 ? first + BitConverter.ToInt32(bytes, 4) : first;
    }

    private static void Step(BinaryReader body)
    {
        if (!ready || failed) throw new InvalidOperationException("Reset required");
        if (terminal) throw new InvalidOperationException("Episode ended; reset required");
        if (body.BaseStream.Length != 28) throw new InvalidDataException("Invalid step payload length");
        int repeat = body.ReadInt32();
        if (repeat < 1 || repeat > 16) throw new InvalidDataException("Action repeat must be 1..16");
        ReadActions(body);
        for (int i = 0; i < repeat && !terminal; i++) Tick(true);
    }

    // One command for two consecutive phases of the same actions. The response
    // holds the observation after the first phase and, unless the episode ended
    // during it, the observation after the second phase. Ticks run exactly as two
    // Step commands would.
    private static void StepSampled(BinaryReader body)
    {
        if (!ready || failed) throw new InvalidOperationException("Reset required");
        if (terminal) throw new InvalidOperationException("Episode ended; reset required");
        if (body.BaseStream.Length != 32) throw new InvalidDataException("Invalid sampled step payload length");
        int first = body.ReadInt32(); int second = body.ReadInt32();
        if (first < 1 || first > 16 || second < 1 || second > 16 || first + second > 16)
            throw new InvalidDataException("Sampled step phases must be 1..16 ticks and at most 16 together");
        ReadActions(body);
        for (int i = 0; i < first && !terminal; i++) Tick(true);
        Observe();
        if (terminal) return;
        for (int i = 0; i < second && !terminal; i++) Tick(true);
        Observe(true);
    }

    private static void ReadActions(BinaryReader body)
    {
        for (int i = 0; i < 6; i++) {
            byte angle = body.ReadByte(); byte reserved = body.ReadByte(); ushort buttons = body.ReadUInt16();
            if (reserved != 0 || (buttons & ~1023) != 0) throw new InvalidDataException("Invalid action bits");
            if (i < inputs.Length) inputs[i] = ReplayArchive.Input((uint)(Updater.SimulationTicks + 1), angle, buttons);
        }
    }

    private static void Tick(bool counted)
    {
        RlProfile.Enter("tick:pre");
        var players = PlayerHandler.Get().PlayerList();
        for (int i = 0; i < inputs.Length; i++) inputs[i].seqNumber = (uint)(Updater.SimulationTicks + 1);
        if (recording != null) recording.frames.Add((InputPacket[])inputs.Clone());
        if (ReferenceCapture.Recording) ReferenceCapture.BeforeReplayStep(inputs);
        for (int i = 0; i < inputs.Length; i++) players[i].OverrideInputWithNetworkInput(inputs[i]);
        RlTelemetry.Counting = counted;
        try { Updater.TickSimulation(GameTime.FixedTimeStep); }
        catch (Exception error) { failed = true; ReferenceCapture.FailedTick(error); throw; }
        finally { RlTelemetry.Counting = false; RlProfile.Enter("tick:post"); RlProfile.Ticked(counted); }
        Updater.LoadedALevelThisUpdate = false;
        started |= !GameSessionHandler.HasGameEnded();
        if (counted) episodeTicks++;
        terminal = (started && GameSessionHandler.HasGameEnded()) || Host.goToNextReplay || episodeTicks >= episodeLimit;
    }

    private static void Observe(bool append = false)
    {
        RlProfile.Enter("bridge:observe");
        if (!append) response.SetLength(0);
        uint flags = (terminal ? 1u : 0u) | (episodeTicks >= episodeLimit ? 2u : 0u) | (started ? 4u : 0u) | (failed ? 8u : 0u);
        RlObservation.Write(payload, Version, episodeTicks, flags, warmup);
    }
}
