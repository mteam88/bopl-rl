using System;
using System.Collections;
using System.Collections.Generic;

namespace Unity.Collections
{
	public enum Allocator { Invalid, None, Temp, TempJob, Persistent }

	public struct NativeArray<T> : IDisposable, IEnumerable<T> where T : struct
	{
		private T[] data;
		public NativeArray(int length, Allocator allocator, NativeArrayOptions options = NativeArrayOptions.ClearMemory)
		{ data = new T[length]; }
		public NativeArray(T[] array, Allocator allocator) { data = (T[])array.Clone(); }
		public int Length => data?.Length ?? 0;
		public bool IsCreated => data != null;
		public T this[int index] { get => data[index]; set => data[index] = value; }
		public void Dispose() => data = null;
		public T[] ToArray() => (T[])(data ?? Array.Empty<T>()).Clone();
		public void CopyFrom(T[] array) => Array.Copy(array, data, array.Length);
		public IEnumerator<T> GetEnumerator() => ((IEnumerable<T>)(data ?? Array.Empty<T>())).GetEnumerator();
		IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
	}

	public enum NativeArrayOptions { UninitializedMemory, ClearMemory }
}

namespace UnityEngine.Video
{
	public class VideoPlayer : Behaviour
	{
		public VideoClip clip { get; set; }
		public string url { get; set; }
		public bool isPlaying { get; private set; }
		public bool isLooping { get; set; }
		public bool playOnAwake { get; set; } = true;
		public bool waitForFirstFrame { get; set; } = true;
		public double time { get; set; }
		public ulong frame { get; set; }
		public float playbackSpeed { get; set; } = 1f;
		public RenderTexture targetTexture { get; set; }
		public VideoRenderMode renderMode { get; set; }
		public VideoAudioOutputMode audioOutputMode { get; set; }
		public event Action<VideoPlayer> loopPointReached { add { } remove { } }
		public event Action<VideoPlayer> prepareCompleted { add { } remove { } }
		public bool isPrepared => true;
		public void Play() => isPlaying = true;
		public void Pause() => isPlaying = false;
		public void Stop() => isPlaying = false;
		public void Prepare() { }
		public void SetDirectAudioMute(ushort track, bool mute) { }
	}

	public class VideoClip : Object
	{
		public double length => 0.0;
		public uint width => 0;
		public uint height => 0;
	}

	public enum VideoRenderMode { CameraFarPlane, CameraNearPlane, RenderTexture, MaterialOverride, APIOnly }
	public enum VideoAudioOutputMode { None, AudioSource, Direct, APIOnly }
}

namespace UnityEngine.LowLevel
{
	public struct PlayerLoopSystem
	{
		public Type type;
		public PlayerLoopSystem[] subSystemList;
		public UpdateFunction updateDelegate;
		public delegate void UpdateFunction();
	}

	public static class PlayerLoop
	{
		private static PlayerLoopSystem current;
		public static PlayerLoopSystem GetCurrentPlayerLoop() => current;
		public static PlayerLoopSystem GetDefaultPlayerLoop() => default;
		public static void SetPlayerLoop(PlayerLoopSystem loop) => current = loop;
	}
}

namespace UnityEngine.Profiling
{
	public static class Profiler
	{
		public static void BeginSample(string name) { }
		public static void EndSample() { }
		public static long GetTotalAllocatedMemoryLong() => GC.GetTotalMemory(false);
	}
}

namespace UnityEngine
{
	public static class Gizmos
	{
		public static Color color { get; set; }
		public static void DrawRay(Vector3 from, Vector3 direction) { }
		public static void DrawRay(Ray r) { }
		public static void DrawLine(Vector3 from, Vector3 to) { }
		public static void DrawSphere(Vector3 center, float radius) { }
		public static void DrawWireSphere(Vector3 center, float radius) { }
		public static void DrawCube(Vector3 center, Vector3 size) { }
		public static void DrawWireCube(Vector3 center, Vector3 size) { }
	}

	public static class Cursor
	{
		public static bool visible { get; set; } = true;
		public static CursorLockMode lockState { get; set; }
		public static void SetCursor(Texture2D texture, Vector2 hotspot, CursorMode cursorMode) { }
	}

	public enum CursorLockMode { None, Locked, Confined }
	public enum CursorMode { Auto, ForceSoftware }

	public class MissingReferenceException : SystemException
	{
		public MissingReferenceException() { }
		public MissingReferenceException(string message) : base(message) { }
	}

	public enum LogType { Error, Assert, Warning, Log, Exception }
	public enum LogOption { None, NoStacktrace }

	public static class Graphics
	{
		public static void CopyTexture(Texture src, Texture dst) { }
		public static void Blit(Texture source, RenderTexture dest) { }
		public static void Blit(Texture source, RenderTexture dest, Material mat) { }
		public static void Blit(Texture source, Material mat) { }
		public static void DrawTexture(Rect screenRect, Texture texture) { }
	}

	public static class Input
	{
		public static bool GetKey(KeyCode key) => false;
		public static bool GetKeyDown(KeyCode key) => false;
		public static bool GetKeyUp(KeyCode key) => false;
		public static bool GetMouseButton(int button) => false;
		public static bool GetMouseButtonDown(int button) => false;
		public static Vector3 mousePosition => Vector3.zero;
		public static float GetAxis(string axisName) => 0f;
		public static float GetAxisRaw(string axisName) => 0f;
		public static bool anyKey => false;
		public static bool anyKeyDown => false;
	}

	public enum KeyCode
	{
		None = 0, Backspace = 8, Tab = 9, Return = 13, Escape = 27, Space = 32,
		A = 97, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
		Alpha0 = 48, Alpha1, Alpha2, Alpha3, Alpha4, Alpha5, Alpha6, Alpha7, Alpha8, Alpha9,
		F1 = 282, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
		UpArrow = 273, DownArrow = 274, RightArrow = 275, LeftArrow = 276,
		LeftShift = 304, RightShift = 303, LeftControl = 306, RightControl = 305
	}

	public class Behaviour_Stub { }
}
