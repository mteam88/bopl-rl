using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace UnityShim
{
	// Owns every shim object. One instance per simulated world; static Current
	// for now (day 4 makes it per-world, see MANY_WORLDS.md).
	public class World
	{
		public static World Current = new World();

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
			Time.Advance(deltaTime);
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
			// Starts queued during a Start run after it, same frame (Unity does this).
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
}
