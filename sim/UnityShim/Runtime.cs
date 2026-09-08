using System;
using UnityShim;

namespace UnityEngine
{
	public static class Time
	{
		public static float deltaTime;
		public static float fixedDeltaTime = 1f / 60f;
		public static int frameCount;
		public static float time;
		public static float unscaledTime;
		public static float unscaledDeltaTime;
		public static float timeScale = 1f;
		public static float realtimeSinceStartup => unscaledTime;
		public static float fixedTime;
		public static float smoothDeltaTime;
		public static float maximumDeltaTime = 1f / 3f;
		public static float timeSinceLevelLoad;
		public static float captureDeltaTime;

		internal static void Advance(float unscaledDelta)
		{
			unscaledDeltaTime = unscaledDelta;
			deltaTime = unscaledDelta * timeScale;
			smoothDeltaTime = deltaTime;
			unscaledTime += unscaledDelta;
			time += deltaTime;
			timeSinceLevelLoad += deltaTime;
		}
	}

	public static class Debug
	{
		public static void Log(object message) => Console.Error.WriteLine("[log] " + message);
		public static void Log(object message, Object context) => Log(message);
		public static void LogWarning(object message) => Console.Error.WriteLine("[warn] " + message);
		public static void LogWarning(object message, Object context) => LogWarning(message);
		public static void LogError(object message) => Console.Error.WriteLine("[error] " + message);
		public static void LogError(object message, Object context) => LogError(message);
		public static void LogException(Exception exception) => Console.Error.WriteLine("[exception] " + exception);
		public static void LogException(Exception exception, Object context) => LogException(exception);
		public static void LogFormat(string format, params object[] args) => Log(string.Format(format, args));
		public static void LogFormat(LogType logType, LogOption logOptions, Object context, string format, params object[] args)
			=> Console.Error.WriteLine("[" + logType.ToString().ToLowerInvariant() + "] " + string.Format(format, args));
		public static void LogErrorFormat(string format, params object[] args) => LogError(string.Format(format, args));
		public static void LogWarningFormat(string format, params object[] args) => LogWarning(string.Format(format, args));
		public static void Assert(bool condition) { if (!condition) LogError("Assertion failed"); }
		public static void Assert(bool condition, object message) { if (!condition) LogError("Assertion failed: " + message); }
		public static void DrawLine(Vector3 start, Vector3 end) { }
		public static void DrawLine(Vector3 start, Vector3 end, Color color) { }
		public static void DrawLine(Vector3 start, Vector3 end, Color color, float duration) { }
		public static void DrawRay(Vector3 start, Vector3 dir) { }
		public static void DrawRay(Vector3 start, Vector3 dir, Color color) { }
		public static void DrawRay(Vector3 start, Vector3 dir, Color color, float duration) { }
		public static void Break() { }
		public static bool isDebugBuild => false;
	}

	public static class Application
	{
		public static bool quitRequested;
		public static int quitCode;

		public static string persistentDataPath { get; set; } =
			Environment.GetEnvironmentVariable("BOPL_SIM_DATA") ?? System.IO.Path.Combine(System.IO.Path.GetTempPath(), "bopl-sim-data");
		public static string dataPath { get; set; } = persistentDataPath;
		public static string streamingAssetsPath => System.IO.Path.Combine(dataPath, "StreamingAssets");
		public static string version => "1.3.4-sim";
		public static string unityVersion => "2022.3.60f1";
		public static string productName => "BoplBattle";
		public static string companyName => "Zapray";
		public static bool isPlaying => true;
		public static bool isEditor => false;
		public static bool isFocused => true;
		public static bool runInBackground { get; set; } = true;
		public static int targetFrameRate { get; set; } = -1;
		public static RuntimePlatform platform => RuntimePlatform.LinuxPlayer;
		public static SystemLanguage systemLanguage => SystemLanguage.English;
		public static event Action quitting;
		public static event Func<bool> wantsToQuit;

		public static void Quit() => Quit(0);

		public static void Quit(int exitCode)
		{
			quitRequested = true;
			quitCode = exitCode;
			quitting?.Invoke();
		}

		public static void OpenURL(string url) { }
	}

	public enum RuntimePlatform { WindowsPlayer, WindowsEditor, LinuxPlayer, LinuxEditor, OSXPlayer, OSXEditor }

	public enum SystemLanguage { English, French, German, Spanish, Swedish, Unknown }

	public static class SystemInfo
	{
		public static UnityEngine.Rendering.GraphicsDeviceType graphicsDeviceType => UnityEngine.Rendering.GraphicsDeviceType.Null;
		public static string operatingSystem => "Linux (sim)";
		public static string deviceModel => "sim";
		public static string deviceName => "sim";
		public static int processorCount => Environment.ProcessorCount;
		public static int systemMemorySize => 32768;
		public static string processorType => "sim";
		public static int graphicsMemorySize => 0;
		public static string graphicsDeviceName => "null";
	}

	public static class QualitySettings
	{
		public static int vSyncCount { get; set; }
		public static int maxQueuedFrames { get; set; }
		public static int GetQualityLevel() => 0;
		public static void SetQualityLevel(int index) { }
	}

	public static class Screen
	{
		public static int width { get; set; } = 1920;
		public static int height { get; set; } = 1080;
		public static bool fullScreen { get; set; }
		public static FullScreenMode fullScreenMode { get; set; }
		public static Resolution currentResolution => new Resolution { width = width, height = height, refreshRate = 60 };
		public static Resolution[] resolutions => new[] { currentResolution };
		public static void SetResolution(int w, int h, bool fs) { width = w; height = h; fullScreen = fs; }
		public static void SetResolution(int w, int h, FullScreenMode mode) { width = w; height = h; fullScreenMode = mode; }
	}

	public enum FullScreenMode { ExclusiveFullScreen, FullScreenWindow, MaximizedWindow, Windowed }

	public struct Resolution
	{
		public int width, height, refreshRate;
		public override string ToString() => $"{width} x {height} @ {refreshRate}Hz";
	}

	public static class Physics2D
	{
		public static bool GetIgnoreLayerCollision(int layer1, int layer2)
			=> World.Current.GetIgnoreLayerCollision(layer1, layer2);

		public static void IgnoreLayerCollision(int layer1, int layer2, bool ignore = true)
		{
			var w = World.Current;
			w.collisionMatrix ??= CreateAllCollide();
			if (ignore)
			{
				w.collisionMatrix[layer1] &= ~(1u << layer2);
				w.collisionMatrix[layer2] &= ~(1u << layer1);
			}
			else
			{
				w.collisionMatrix[layer1] |= 1u << layer2;
				w.collisionMatrix[layer2] |= 1u << layer1;
			}
		}

		private static uint[] CreateAllCollide()
		{
			var m = new uint[32];
			for (int i = 0; i < 32; i++) m[i] = 0xffffffffu;
			return m;
		}
	}

	public static class Resources
	{
		public static T Load<T>(string path) where T : Object
			=> (T)World.Current.resourceLoader?.Invoke(path, typeof(T));

		public static Object Load(string path) => World.Current.resourceLoader?.Invoke(path, typeof(Object));

		public static T[] LoadAll<T>(string path) where T : Object => Array.Empty<T>();

		public static T[] FindObjectsOfTypeAll<T>() where T : Object => Object.FindObjectsOfType<T>(true);
		public static Object[] FindObjectsOfTypeAll(Type type) => Object.FindObjectsOfType(type);

		public static void UnloadUnusedAssets() { }
	}

	public class ScriptableObject : Object
	{
		public static T CreateInstance<T>() where T : ScriptableObject => (T)CreateInstance(typeof(T));

		public static ScriptableObject CreateInstance(Type type)
		{
			var so = (ScriptableObject)Activator.CreateInstance(type, true);
			DispatchOnEnable(so);
			return so;
		}

		// Unity fires OnEnable when a ScriptableObject is created or its asset is
		// loaded/deserialized (AnimationData builds lookup tables there). The
		// Export loader calls this after populating serialized fields.
		internal static void DispatchOnEnable(ScriptableObject so)
		{
			var method = so.GetType().GetMethod("OnEnable",
				System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic);
			method?.Invoke(so, null);
		}
	}

	public class TextAsset : Object
	{
		public string text { get; set; } = "";
		public byte[] bytes { get; set; } = Array.Empty<byte>();
		public TextAsset() { }
		public TextAsset(string text) { this.text = text; bytes = System.Text.Encoding.UTF8.GetBytes(text); }
	}

	public static class JsonUtility
	{
		public static string ToJson(object obj) => Newtonsoft.Json.JsonConvert.SerializeObject(obj);
		public static T FromJson<T>(string json) => Newtonsoft.Json.JsonConvert.DeserializeObject<T>(json);
		public static void FromJsonOverwrite(string json, object objectToOverwrite)
			=> Newtonsoft.Json.JsonConvert.PopulateObject(json, objectToOverwrite);
	}

	public static class PlayerPrefs
	{
		private static readonly System.Collections.Generic.Dictionary<string, object> store = new();
		public static int GetInt(string key, int defaultValue = 0) => store.TryGetValue(key, out var v) && v is int i ? i : defaultValue;
		public static void SetInt(string key, int value) => store[key] = value;
		public static float GetFloat(string key, float defaultValue = 0f) => store.TryGetValue(key, out var v) && v is float f ? f : defaultValue;
		public static void SetFloat(string key, float value) => store[key] = value;
		public static string GetString(string key, string defaultValue = "") => store.TryGetValue(key, out var v) && v is string s ? s : defaultValue;
		public static void SetString(string key, string value) => store[key] = value;
		public static bool HasKey(string key) => store.ContainsKey(key);
		public static void DeleteKey(string key) => store.Remove(key);
		public static void DeleteAll() => store.Clear();
		public static void Save() { }
	}
}

namespace UnityEngine.Rendering
{
	public enum GraphicsDeviceType
	{
		OpenGL2 = 0, Direct3D9 = 1, Direct3D11 = 2, PlayStation3 = 3, Null = 4,
		Xbox360 = 6, OpenGLES2 = 8, OpenGLES3 = 11, Metal = 16, OpenGLCore = 17,
		Direct3D12 = 18, Vulkan = 21
	}

	// Present in exported scenes; sorting is presentation-only.
	public class SortingGroup : Behaviour
	{
		public string sortingLayerName { get; set; } = "Default";
		public int sortingLayerID { get; set; }
		public int sortingOrder { get; set; }
	}
}
