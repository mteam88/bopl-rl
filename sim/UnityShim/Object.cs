using System;
using System.Collections.Generic;
using UnityShim;

namespace UnityEngine
{
	public class Object
	{
		internal int instanceId;
		internal bool destroyed;
		internal string objectName = "";
		// Assets (prefab templates, ScriptableObjects from the export) never get
		// lifecycle callbacks and are not part of the live scene.
		internal bool isAsset;
		// Unity instance-id model: objects loaded from a file (scene objects,
		// assets) have POSITIVE ids assigned in file order; objects created at
		// runtime (Instantiate, new GameObject, AddComponent) have NEGATIVE ids
		// that decrease over time. The shim keeps one positive monotonic counter
		// and this flag; FindObjectsOfType sorts with key
		// (persistent ? +instanceId : -instanceId) to reproduce Unity's
		// FindObjectsSortMode.InstanceID ascending order: runtime clones first,
		// newest to oldest, then scene objects in load order. Set by the export
		// loader (GraphBuilder) for scene-file objects.
		internal bool persistent;

		public HideFlags hideFlags;

		public Object() { instanceId = World.NextInstanceId(); }

		public virtual string name { get => objectName; set => objectName = value; }

		public int GetInstanceID() => instanceId;

		public override int GetHashCode() => instanceId;

		public override bool Equals(object other) => CompareBaseObjects(this, other as Object);

		public override string ToString() => string.IsNullOrEmpty(objectName) ? GetType().Name : objectName;

		// F3: a null literal equals any destroyed object, but two live references
		// compare by identity even when both are destroyed - destroyedA ==
		// destroyedB is FALSE unless they are the same object (Unity compares
		// instance ids).
		[System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
		private static bool CompareBaseObjects(Object a, Object b)
		{
			bool an = a is null;
			bool bn = b is null;
			if (an && bn) return true;
			if (an) return b.destroyed;
			if (bn) return a.destroyed;
			return ReferenceEquals(a, b);
		}

		[System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
		public static bool operator ==(Object a, Object b) => CompareBaseObjects(a, b);
		[System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
		public static bool operator !=(Object a, Object b) => !CompareBaseObjects(a, b);
		[System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
		public static implicit operator bool(Object o) => !(o is null) && !o.destroyed;

		public static void DontDestroyOnLoad(Object target)
		{
			// Marks the root GameObject; single-mode scene loads spare it. Unity
			// moves the object into the "DontDestroyOnLoad" pseudo-scene (buildIndex
			// -1) - ReferenceCapture.PathOf prints it as the path root.
			var go = target as GameObject ?? (target as Component)?.gameObject;
			if (go == null) return;
			var root = go.transform.root.gameObject;
			root.dontDestroyOnLoad = true;
			StampSceneRecursive(root, new UnityEngine.SceneManagement.Scene(-1, "DontDestroyOnLoad"));
		}

		private static void StampSceneRecursive(GameObject go, UnityEngine.SceneManagement.Scene scene)
		{
			go.scene = scene;
			var t = go.transform;
			for (int i = 0; i < t.childCount; i++)
				StampSceneRecursive(t.GetChild(i).gameObject, scene);
		}

		public static void Destroy(Object obj) => Destroy(obj, 0f);

		public static void Destroy(Object obj, float t)
		{
			if (obj == null) return;
			// Unity defers to end of frame (delay ignored beyond that; gameplay
			// uses Updater.DestroyFix for deterministic destruction).
			World.Current.ScheduleDestroy(obj);
		}

		public static void DestroyImmediate(Object obj) => DestroyImmediate(obj, false);

		public static void DestroyImmediate(Object obj, bool allowDestroyingAssets)
		{
			if (obj == null || obj.destroyed) return;
			switch (obj)
			{
				case GameObject go: DestroyGameObject(go); break;
				case Component c: DestroyComponent(c); break;
				default: obj.destroyed = true; break;
			}
		}

		private static void DestroyGameObject(GameObject go)
		{
			// Children first (depth-first), disable callbacks, then OnDestroy.
			for (int i = go.transform.childCount - 1; i >= 0; i--)
				DestroyGameObject(go.transform.GetChild(i).gameObject);
			foreach (var c in go.components.ToArray())
			{
				if (c is MonoBehaviour mb && !mb.destroyed && !go.isAsset)
				{
					if (mb.enabledSelf && go.activeInHierarchy && mb.awakeRan)
						mb.SetEnabledInternal(false);
					if (mb.awakeRan) mb.CallMessage(MessageId.OnDestroy);
					mb.CleanupOnDestroy();
				}
				c.destroyed = true;
			}
			if (go.transform.parent != null) go.transform.SetParentInternal(null, false);
			else World.Current.roots.Remove(go);
			go.destroyed = true;
			go.transform.destroyed = true;
			World.Current.Compact();
		}

		private static void DestroyComponent(Component c)
		{
			if (c is Transform) return; // Unity refuses to destroy transforms
			if (c is MonoBehaviour mb && !c.gameObject.isAsset)
			{
				if (mb.enabledSelf && c.gameObject.activeInHierarchy && mb.awakeRan)
					mb.SetEnabledInternal(false);
				if (mb.awakeRan) mb.CallMessage(MessageId.OnDestroy);
				mb.CleanupOnDestroy();
			}
			c.gameObject.components.Remove(c);
			c.destroyed = true;
			World.Current.Compact();
		}

		// ---- Instantiate -------------------------------------------------------

		public static T Instantiate<T>(T original) where T : Object
			=> (T)InstantiateInternal(original, null, false, false, default, default);

		public static T Instantiate<T>(T original, Transform parent) where T : Object
			=> (T)InstantiateInternal(original, parent, false, false, default, default);

		public static T Instantiate<T>(T original, Transform parent, bool instantiateInWorldSpace) where T : Object
			=> (T)InstantiateInternal(original, parent, instantiateInWorldSpace, false, default, default);

		public static T Instantiate<T>(T original, Vector3 position, Quaternion rotation) where T : Object
			=> (T)InstantiateInternal(original, null, false, true, position, rotation);

		public static T Instantiate<T>(T original, Vector3 position, Quaternion rotation, Transform parent) where T : Object
			=> (T)InstantiateInternal(original, parent, false, true, position, rotation);

		public static Object Instantiate(Object original) => InstantiateInternal(original, null, false, false, default, default);
		public static Object Instantiate(Object original, Vector3 position, Quaternion rotation)
			=> InstantiateInternal(original, null, false, true, position, rotation);
		public static Object Instantiate(Object original, Vector3 position, Quaternion rotation, Transform parent)
			=> InstantiateInternal(original, parent, false, true, position, rotation);
		public static Object Instantiate(Object original, Transform parent) => InstantiateInternal(original, parent, false, false, default, default);

		private static Object InstantiateInternal(Object original, Transform parent, bool worldSpace, bool hasPose, Vector3 position, Quaternion rotation)
		{
			if (original == null) throw new ArgumentException("The Object you want to instantiate is null.");
			if (original is ScriptableObject so)
				return CloneRules.CloneScriptableObject(so);
			if (original is Material mat)
				return new Material(mat) { name = mat.name + "(Clone)" };
			GameObject sourceGo = original as GameObject ?? (original as Component)?.gameObject;
			if (sourceGo == null) throw new ArgumentException("Instantiate of unsupported type " + original.GetType().Name);
			GameObject clone = CloneRules.CloneHierarchy(sourceGo, parent);
			clone.objectName = sourceGo.objectName + "(Clone)";
			if (hasPose)
			{
				clone.transform.position = position;
				clone.transform.rotation = rotation;
			}
			else if (parent != null && !worldSpace)
			{
				// keep the prefab's local pose under the new parent (already local-copied)
			}
			CloneRules.DispatchAwake(clone);
			if (original is GameObject) return clone;
			// Return the clone's counterpart of the requested component.
			var comp = (Component)original;
			int goIndex = 0, compIndex = sourceGo.components.IndexOf(comp);
			return clone.components[compIndex];
		}

		// ---- Find --------------------------------------------------------------

		public static T FindObjectOfType<T>() where T : Object => FindObjectOfType<T>(false);

		public static T FindObjectOfType<T>(bool includeInactive) where T : Object
		{
			var all = FindObjectsOfType<T>(includeInactive);
			return all.Length > 0 ? all[0] : null;
		}

		public static T[] FindObjectsOfType<T>() where T : Object => FindObjectsOfType<T>(false);

		public static T[] FindObjectsOfType<T>(bool includeInactive) where T : Object
		{
			var result = new List<T>();
			foreach (var root in World.Current.roots.ToArray())
				CollectComponents(root, typeof(T), includeInactive, result);
			SortByUnityInstanceId(result);
			return result.ToArray();
		}

		public static Object FindObjectOfType(Type type)
		{
			var all = FindObjectsOfType(type);
			return all.Length > 0 ? all[0] : null;
		}

		public static Object[] FindObjectsOfType(Type type)
		{
			var result = new List<Object>();
			foreach (var root in World.Current.roots.ToArray())
				CollectComponents(root, type, false, result);
			SortByUnityInstanceId(result);
			return result.ToArray();
		}

		// The 2022.3 legacy FindObjectsOfType maps to FindObjectsByType(...,
		// FindObjectsSortMode.InstanceID): ascending instance id, where runtime
		// instances are negative and decreasing. Order is gameplay-visible
		// (Mjolnir/Portal/BananaPeel iterate FindObjectsOfType<SlimeController>)
		// and trace-visible (ReferenceCapture identity ties among same-named
		// root clones, e.g. two CursorArrow(Clone) roots on l3-p2-g4 tick 649).
		private static void SortByUnityInstanceId<T>(List<T> objects) where T : class
		{
			objects.Sort((left, right) =>
			{
				var a = (Object)(object)left;
				var b = (Object)(object)right;
				long keyA = a.persistent ? a.instanceId : -(long)a.instanceId;
				long keyB = b.persistent ? b.instanceId : -(long)b.instanceId;
				return keyA.CompareTo(keyB);
			});
		}

		private static void CollectComponents<T>(GameObject go, Type type, bool includeInactive, List<T> result) where T : class
		{
			if (go == null || go.destroyed) return;
			if (!includeInactive && !go.activeInHierarchy) return;
			if (type == typeof(GameObject))
			{
				result.Add(go as T);
			}
			else
			{
				foreach (var c in go.components)
					if (!c.destroyed && type.IsInstanceOfType(c)) result.Add(c as T);
			}
			var t = go.transform;
			for (int i = 0; i < t.childCount; i++)
				CollectComponents(t.GetChild(i).gameObject, type, includeInactive, result);
		}
	}

	public enum HideFlags
	{
		None = 0, HideInHierarchy = 1, HideInInspector = 2, DontSaveInEditor = 4,
		NotEditable = 8, DontSaveInBuild = 16, DontUnloadUnusedAsset = 32,
		DontSave = 52, HideAndDontSave = 61
	}
}
