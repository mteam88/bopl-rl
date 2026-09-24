using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace UnityShim
{
	// Owns every shim object. One instance per simulated world.
	//
	// Current is thread-local: a thread works on exactly one world at a time,
	// and different threads may drive different worlds of the same runtime
	// concurrently (many worlds per process). All
	// per-world state that Unity keeps in statics (Time, Random, scene events)
	// lives on the World; gameplay's own per-match statics are swapped in and
	// out by the onBind/onUnbind hooks (Gameplay's RlWorldState). Anything the
	// shim keeps genuinely static (ScriptExecutionOrder, execution tables) is
	// read-mostly and locked.
	//
	// The single-world drivers (sim/Host, Oracle) never bind: the first access
	// on a thread lazily creates that thread's world, so they behave as before.
	public class World
	{
		[ThreadStatic] private static World current;

		public static World Current
		{
			get => current ??= new World();
			set => current = value;
		}

		// Gameplay registers these once; they restore/capture its per-match
		// statics around a thread's use of a world.
		public static Action<World> onBind, onUnbind;

		// Make `world` the calling thread's current world. Pair with Unbind on
		// the same thread; nesting is not supported (a thread has one world).
		public static void Bind(World world)
		{
			if (world == null) throw new ArgumentNullException(nameof(world));
			if (current != null && current != world) throw new InvalidOperationException("UnityShim: thread already bound to another world");
			current = world;
			onBind?.Invoke(world);
		}

		public static void Unbind()
		{
			var world = current;
			if (world == null) return;
			onUnbind?.Invoke(world);
			current = null;
		}

		// Slot for the gameplay assembly's per-world statics snapshot.
		public object gameplayState;
		// Export's per-world object graph (built assets, ids); see ExportDatabase.Store.
		public object assetStore;

		// UnityEngine.Time, per world (see Runtime.cs Time forwarding).
		public TimeState time = TimeState.Defaults();

		// UnityEngine.Random (presentation-only in this game; System.Random deviation).
		public System.Random random = new System.Random(42);

		// Scene-object singletons Unity exposes statically.
		public UnityEngine.EventSystems.EventSystem eventSystem;
		public UnityEngine.InputSystem.PlayerInputManager playerInputManager;
		public readonly System.Collections.Generic.Dictionary<string, object> playerPrefs = new System.Collections.Generic.Dictionary<string, object>();

		// SceneManager events: subscribers are this world's objects.
		public Action<Scene, LoadSceneMode> sceneLoaded;
		public Action<Scene> sceneUnloaded;
		public Action<Scene, Scene> activeSceneChanged;

		private int nextInstanceId = 1;

		// Live scene objects, creation order. Roots only; children hang off Transform.
		internal readonly List<GameObject> roots = new List<GameObject>();

		// All live MonoBehaviours in creation order, for Update/LateUpdate passes.
		internal readonly List<MonoBehaviour> behaviours = new List<MonoBehaviour>();

		// Behaviours whose Start has not run yet (enabled at least once).
		internal readonly List<MonoBehaviour> pendingStarts = new List<MonoBehaviour>();

		// Object.Destroy is deferred to end of frame.
		internal readonly List<UnityEngine.Object> destroyQueue = new List<UnityEngine.Object>();

		internal readonly CoroutineScheduler coroutines = new CoroutineScheduler();

		// Layer/tag names + 2D collision matrix, installed by the Export loader
		// from ProjectSettings (settings.json). Defaults: numeric names, all collide.
		public string[] layerNames = new string[32];
		public List<string> tagNames = new List<string> { "Untagged", "Respawn", "Finish", "EditorOnly", "MainCamera", "Player", "GameController" };
		public uint[] collisionMatrix;
		public bool collisionMatrixLoaded;

		// Resources.Load and SceneManager.LoadScene delegate here (Export loader installs).
		public Func<string, Type, UnityEngine.Object> resourceLoader;
		public Action<int, LoadSceneMode> sceneLoader;
		public int sceneCountInBuildSettings = 1;
		public Scene activeScene = new Scene(0, "");

		public int frameCount;

		public World()
		{
			for (int i = 0; i < 32; i++) layerNames[i] = "Layer" + i;
			layerNames[0] = "Default";
		}

		public static int NextInstanceId() => Current.nextInstanceId++;

		public bool GetIgnoreLayerCollision(int a, int b)
		{
			if (collisionMatrix == null)
			{
				if (!collisionMatrixLoaded)
				{
					UnityEngine.Debug.LogError("UnityShim: no collision matrix loaded (settings.json missing) - all layers collide");
					collisionMatrixLoaded = true;
				}
				return false;
			}
			return (collisionMatrix[a] & (1u << b)) == 0;
		}

		internal void RegisterBehaviour(MonoBehaviour mb) => behaviours.Add(mb);

		internal void QueueStart(MonoBehaviour mb)
		{
			if (!mb.startQueued && !mb.startRan)
			{
				mb.startQueued = true;
				pendingStarts.Add(mb);
			}
		}

		public void ScheduleDestroy(UnityEngine.Object o) => destroyQueue.Add(o);

		// One Unity-style presentation frame:
		// pending Starts -> Update -> coroutines -> LateUpdate -> destroy flush.
		public void RunFrame(float deltaTime)
		{
			time.Advance(deltaTime);
			frameCount++;
			Time.frameCount = frameCount;
			FlushPendingStarts();
			var snapshot = behaviours.ToArray();
			foreach (var mb in snapshot)
				if (mb != null && mb.isActiveAndEnabled && mb.startRan)
					mb.DispatchInvokes();
			foreach (var mb in snapshot)
				if (mb != null && mb.isActiveAndEnabled && mb.startRan)
					mb.CallMessage(MessageId.Update);
			coroutines.Step();
			foreach (var mb in snapshot)
				if (mb != null && mb.isActiveAndEnabled && mb.startRan)
					mb.CallMessage(MessageId.LateUpdate);
			FlushDestroyQueue();
		}

		public void FlushPendingStarts()
		{
			// Start batches are ordered by script execution order, like Awake.
			// Starts queued during a Start run after it, same frame (Unity does
			// this); late arrivals keep queue order.
			ScriptExecutionOrder.SortStable(pendingStarts);
			for (int i = 0; i < pendingStarts.Count; i++)
			{
				var mb = pendingStarts[i];
				if (mb == null || mb.destroyed) continue;
				if (!mb.isActiveAndEnabled) { mb.startQueued = false; pendingStarts.RemoveAt(i); i--; continue; }
				mb.startQueued = false;
				mb.startRan = true;
				mb.CallStart();
			}
			pendingStarts.Clear();
		}

		public void FlushDestroyQueue()
		{
			// Destroys queued during destroy callbacks are honoured in the same flush.
			for (int i = 0; i < destroyQueue.Count; i++)
			{
				var o = destroyQueue[i];
				if (o != null && !o.destroyed) UnityEngine.Object.DestroyImmediate(o);
			}
			destroyQueue.Clear();
		}

		// Scene-load activation: Unity awakes freshly loaded objects before any
		// Start runs. Batch = the whole load; behaviours collected depth-first in
		// root/file order, then STABLE-sorted by script execution order (the game
		// uses it heavily: FixTransform -80, PlayerBody 2, SlimeController 269...).
		// All Awakes run in that order, then all OnEnables.
		public void ActivateLoadedObjects(List<GameObject> newRoots)
		{
			var batch = new List<MonoBehaviour>();
			foreach (var root in newRoots)
				if (root != null && !root.destroyed) CollectActiveBehaviours(root, batch);
			DispatchActivationBatch(batch);
		}

		internal static void CollectActiveBehaviours(GameObject go, List<MonoBehaviour> batch)
		{
			if (go.destroyed || !go.activeInHierarchy) return;
			foreach (var c in go.components)
				if (c is MonoBehaviour mb && !mb.destroyed) batch.Add(mb);
			var t = go.transform;
			for (int i = 0; i < t.childCount; i++)
				CollectActiveBehaviours(t.GetChild(i).gameObject, batch);
		}

		internal static void DispatchActivationBatch(List<MonoBehaviour> batch)
		{
			ScriptExecutionOrder.SortStable(batch);
			foreach (var mb in batch)
				if (!mb.destroyed && mb.gameObject.activeInHierarchy) mb.DispatchAwakeIfNeeded();
			foreach (var mb in batch)
				if (!mb.destroyed && mb.enabledSelf && mb.gameObject.activeInHierarchy) mb.DispatchEnable();
		}

		// Single-mode scene load: destroy everything except DontDestroyOnLoad roots.
		public void UnloadCurrentScene()
		{
			foreach (var root in roots.ToArray())
			{
				if (root == null || root.destroyed || root.dontDestroyOnLoad) continue;
				UnityEngine.Object.DestroyImmediate(root);
			}
			FlushDestroyQueue();
			Compact();
		}

		internal void Compact()
		{
			behaviours.RemoveAll(b => b == null || b.destroyed);
			roots.RemoveAll(r => r == null || r.destroyed);
		}

		// Deterministic full reset used between episodes if needed.
		public static void ResetAll()
		{
			Current = new World();
		}
	}

	// UnityEngine.Time's mutable statics, owned by a World.
	public sealed class TimeState
	{
		public float deltaTime, fixedDeltaTime, time, unscaledTime, unscaledDeltaTime, timeScale;
		public float fixedTime, smoothDeltaTime, maximumDeltaTime, timeSinceLevelLoad, captureDeltaTime;
		public int frameCount;

		public static TimeState Defaults() => new TimeState { fixedDeltaTime = 1f / 60f, timeScale = 1f, maximumDeltaTime = 1f / 3f };

		internal void Advance(float unscaledDelta)
		{
			unscaledDeltaTime = unscaledDelta;
			deltaTime = unscaledDelta * timeScale;
			smoothDeltaTime = deltaTime;
			unscaledTime += unscaledDelta;
			time += deltaTime;
			timeSinceLevelLoad += deltaTime;
		}
	}
}
