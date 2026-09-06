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
    public const uint Version = 1;
    private static readonly string endpoint = Environment.GetEnvironmentVariable("BOPL_RL_SOCKET");
    public static bool Enabled => !string.IsNullOrEmpty(endpoint);
    private static Socket socket;
    private static BinaryReader reader;
    private static BinaryWriter writer;
    private static readonly MemoryStream response = new MemoryStream(65536);
    private static readonly BinaryWriter payload = new BinaryWriter(response, Encoding.UTF8, true);
    private static bool loading, ready, started, terminal, failed;
    private static int warmup, remainingWarmup, episodeTicks, episodeLimit;
    private static uint resetSequence;
    private static InputPacket[] inputs;
    private static int loadedFrame;
    private static byte[] pendingStep;
    private static uint pendingSequence;
    private static ReplayArchive recording;
    private const int CommandsPerFrame = 16;

    private static void Connect()
    {
        socket = new Socket(AddressFamily.Unix, SocketType.Stream, ProtocolType.Unspecified);
        socket.Connect(new UnixDomainSocketEndPoint(endpoint));
        var stream = new NetworkStream(socket, true);
        reader = new BinaryReader(stream, Encoding.UTF8, true);
        writer = new BinaryWriter(stream, Encoding.UTF8, true);
        Application.targetFrameRate = -1;
        QualitySettings.vSyncCount = 0;
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
        payload.Flush();
        writer.Write(Magic); writer.Write(operation); writer.Write(sequence); writer.Write((uint)response.Length);
        writer.Write(response.GetBuffer(), 0, checked((int)response.Length)); writer.Flush();
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
                try { Step(body); Observe(); Send(2, pendingSequence); }
                catch (Exception error) { Error(pendingSequence, error); }
                frameTicks += Updater.SimulationTicks - before;
                if (frameTicks >= CommandsPerFrame) return;
            }
            for (int command = 0; command < CommandsPerFrame; command++) {
                uint magic = reader.ReadUInt32(); uint operation = reader.ReadUInt32();
                uint sequence = reader.ReadUInt32(); uint length = reader.ReadUInt32();
                if (magic != Magic || length > 4096) throw new InvalidDataException("Invalid RL command frame");
                byte[] bytes = reader.ReadBytes((int)length);
                if (bytes.Length != length) throw new EndOfStreamException();
                using var body = new BinaryReader(new MemoryStream(bytes, false));
                try {
                    switch (operation) {
                        case 1: Reset(host, body, sequence); return;
                        case 2:
                            if (length >= 4 && frameTicks > 0 && BitConverter.ToInt32(bytes, 0) > CommandsPerFrame - frameTicks) {
                                pendingStep = bytes; pendingSequence = sequence; return;
                            }
                            int before = Updater.SimulationTicks;
                            Step(body); frameTicks += Updater.SimulationTicks - before;
                            Observe(); Send(operation, sequence);
                            if (frameTicks >= CommandsPerFrame) return;
                            break;
                        case 3:
                            if (length != 0 || !ready) throw new InvalidOperationException("Observation requires reset");
                            Observe(); Send(operation, sequence); break;
                        case 4:
                            if (length != 0) throw new InvalidDataException("Invalid close payload");
                            response.SetLength(0); Send(operation, sequence);
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
        host.ResetRl(setup);
        setup.frames.Clear(); recording = setup;
    }

    private static void Step(BinaryReader body)
    {
        if (!ready || failed) throw new InvalidOperationException("Reset required");
        if (terminal) throw new InvalidOperationException("Episode ended; reset required");
        if (body.BaseStream.Length != 28) throw new InvalidDataException("Invalid step payload length");
        int repeat = body.ReadInt32();
        if (repeat < 1 || repeat > 16) throw new InvalidDataException("Action repeat must be 1..16");
        for (int i = 0; i < 6; i++) {
            byte angle = body.ReadByte(); byte reserved = body.ReadByte(); ushort buttons = body.ReadUInt16();
            if (reserved != 0 || (buttons & ~1023) != 0) throw new InvalidDataException("Invalid action bits");
            if (i < inputs.Length) inputs[i] = ReplayArchive.Input((uint)(Updater.SimulationTicks + 1), angle, buttons);
        }
        for (int i = 0; i < repeat && !terminal; i++) Tick(true);
    }

    private static void Tick(bool counted)
    {
        var players = PlayerHandler.Get().PlayerList();
        for (int i = 0; i < inputs.Length; i++) inputs[i].seqNumber = (uint)(Updater.SimulationTicks + 1);
        if (recording != null) recording.frames.Add((InputPacket[])inputs.Clone());
        if (ReferenceCapture.Recording) ReferenceCapture.BeforeReplayStep(inputs);
        for (int i = 0; i < inputs.Length; i++) players[i].OverrideInputWithNetworkInput(inputs[i]);
        try { Updater.TickSimulation(GameTime.FixedTimeStep); }
        catch (Exception error) { failed = true; ReferenceCapture.FailedTick(error); throw; }
        Updater.LoadedALevelThisUpdate = false;
        started |= !GameSessionHandler.HasGameEnded();
        if (counted) episodeTicks++;
        terminal = (started && GameSessionHandler.HasGameEnded()) || Host.goToNextReplay || episodeTicks >= episodeLimit;
    }

    private static void Observe()
    {
        response.SetLength(0);
        uint flags = (terminal ? 1u : 0u) | (episodeTicks >= episodeLimit ? 2u : 0u) | (started ? 4u : 0u) | (failed ? 8u : 0u);
        RlObservation.Write(payload, Version, episodeTicks, flags, warmup);
    }
}
