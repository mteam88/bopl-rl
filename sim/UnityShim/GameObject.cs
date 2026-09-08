using System;
using System.Collections.Generic;
using UnityShim;
using UnityEngine.SceneManagement;

namespace UnityEngine
{
	public sealed class GameObject : Object
	{
		internal readonly List<Component> components = new List<Component>();
		internal bool activeSelfField = true;
		internal string tagField = "Untagged";
		internal bool dontDestroyOnLoad;

		public int layer;
		public Scene scene;

		public Transform transform { get; private set; }

		public GameObject() : this("New Game Object") { }

		public GameObject(string name)
		{
			objectName = name ?? "";
			transform = new Transform();
			transform.AttachTo(this);
			components.Add(transform);
			scene = World.Current.activeScene;
			World.Current.roots.Add(this);
		}

		public GameObject(string name, params Type[] componentTypes) : this(name)
		{
			foreach (var t in componentTypes) AddComponent(t);
		}

		// Used by the Export loader / clone machinery: no auto root registration.
		internal static GameObject CreateDetached(string name, bool asset)
		{
			var go = new GameObject(name);
			go.isAsset = asset;
			// Assets live outside any scene: scene.IsValid() is false and scene.name
			// is null (ReferenceCapture.PathOf renders them as "/Name[i]").
			if (asset) { go.transform.isAsset = true; go.scene = default; World.Current.roots.Remove(go); }
			return go;
		}

		// Export loader: scenes serialize RectTransform as the transform component.
		// Must be called before any children or references exist.
		internal void UseRectTransform()
		{
			if (transform is RectTransform) return;
			var old = transform;
			var rt = new RectTransform();
			rt.AttachTo(this);
			rt.isAsset = old.isAsset;
			// Carry the hierarchy links and pose: GraphBuilder parents the GO
			// before it reaches the RectTransform component entry, so the swap
			// must not orphan it (WinnerTriangle: charStatsList.transform.parent).
			rt.localPositionField = old.localPositionField;
			rt.localRotationField = old.localRotationField;
			rt.localScaleField = old.localScaleField;
			rt.parentField = old.parentField;
			if (old.parentField != null)
				old.parentField.children[old.parentField.children.IndexOf(old)] = rt;
			rt.children.AddRange(old.children);
			foreach (var child in old.children) child.parentField = rt;
			old.children.Clear();
			components[components.IndexOf(old)] = rt;
			transform = rt;
		}

		public GameObject gameObject => this;

		public string tag
		{
			get => tagField;
			set => tagField = value ?? "Untagged";
		}

		public bool CompareTag(string other) => tagField == other;

		public bool activeSelf => activeSelfField;

		public bool activeInHierarchy
		{
			get
			{
				if (!activeSelfField || destroyed) return false;
				var p = transform.parent;
				return p == null || p.gameObject.activeInHierarchy;
			}
		}

		public void SetActive(bool value)
		{
			if (activeSelfField == value) return;
			bool wasActive = activeInHierarchy;
			activeSelfField = value;
			bool nowActive = activeInHierarchy;
			if (wasActive == nowActive || isAsset) return;
			if (nowActive) NotifyActivation(this, true);
			else NotifyActivation(this, false);
		}

		// Called when activeInHierarchy flips for this subtree. The affected
		// behaviours form one batch ordered by script execution order, like
		// Unity's SetActive semantics (Awake for never-awoken ones first).
		internal static void NotifyActivation(GameObject go, bool active)
		{
			var batch = new List<MonoBehaviour>();
			CollectActivationBatch(go, batch);
			UnityShim.ScriptExecutionOrder.SortStable(batch);
			if (active)
			{
				foreach (var mb in batch)
					if (!mb.destroyed && mb.gameObject.activeInHierarchy) mb.DispatchAwakeIfNeeded();
				foreach (var mb in batch)
					if (!mb.destroyed && mb.enabledSelf && mb.gameObject.activeInHierarchy) mb.DispatchEnable();
			}
			else
			{
				foreach (var mb in batch)
					if (!mb.destroyed) mb.OnBecameInactive();
			}
		}

		private static void CollectActivationBatch(GameObject go, List<MonoBehaviour> batch)
		{
			foreach (var c in go.components)
				if (c is MonoBehaviour mb && !mb.destroyed) batch.Add(mb);
			var t = go.transform;
			for (int i = 0; i < t.childCount; i++)
			{
				var child = t.GetChild(i).gameObject;
				if (child.activeSelfField) CollectActivationBatch(child, batch);
			}
		}

		// ---- components ----------------------------------------------------

		public T AddComponent<T>() where T : Component => (T)AddComponent(typeof(T));

		public Component AddComponent(Type componentType)
		{
			var c = (Component)Activator.CreateInstance(componentType);
			c.AttachTo(this);
			components.Add(c);
			if (c is MonoBehaviour mb)
			{
				mb.isAsset = isAsset;
				if (!isAsset)
				{
					World.Current.RegisterBehaviour(mb);
					// F9: Awake fires for components added to an active GameObject
					// regardless of enabled; OnBecameActive gates OnEnable itself.
					if (activeInHierarchy) mb.OnBecameActive();
				}
			}
			return c;
		}

		// Export loader: attach without lifecycle (Awake dispatched later, in order).
		internal Component AddComponentSilent(Type componentType)
		{
			var c = (Component)Activator.CreateInstance(componentType);
			c.AttachTo(this);
			c.isAsset = isAsset;
			components.Add(c);
			if (c is MonoBehaviour mb) mb.isAsset = isAsset;
			return c;
		}

		public T GetComponent<T>()
		{
			foreach (var c in components)
				if (!c.destroyed && c is T t) return t;
			return default;
		}

		public Component GetComponent(Type type)
		{
			foreach (var c in components)
				if (!c.destroyed && type.IsInstanceOfType(c)) return c;
			return null;
		}

		public Component GetComponent(string typeName)
		{
			foreach (var c in components)
				if (!c.destroyed && c.GetType().Name == typeName) return c;
			return null;
		}

		public bool TryGetComponent<T>(out T component)
		{
			component = GetComponent<T>();
			return component != null && !(component is Component c && c.destroyed);
		}

		public T[] GetComponents<T>()
		{
			var list = new List<T>();
			foreach (var c in components)
				if (!c.destroyed && c is T t) list.Add(t);
			return list.ToArray();
		}

		public void GetComponents<T>(List<T> results)
		{
			results.Clear();
			foreach (var c in components)
				if (!c.destroyed && c is T t) results.Add(t);
		}

		public Component[] GetComponents(Type type)
		{
			var list = new List<Component>();
			foreach (var c in components)
				if (!c.destroyed && type.IsInstanceOfType(c)) list.Add(c);
			return list.ToArray();
		}

		public T GetComponentInChildren<T>() => GetComponentInChildren<T>(false);

		public T GetComponentInChildren<T>(bool includeInactive)
		{
			if (includeInactive || activeInHierarchy)
			{
				var own = GetComponent<T>();
				if (own != null) return own;
			}
			for (int i = 0; i < transform.childCount; i++)
			{
				var child = transform.GetChild(i).gameObject;
				if (!includeInactive && !child.activeSelfField) continue;
				var found = child.GetComponentInChildren<T>(includeInactive);
				if (found != null) return found;
			}
			return default;
		}

		public T[] GetComponentsInChildren<T>() => GetComponentsInChildren<T>(false);

		public T[] GetComponentsInChildren<T>(bool includeInactive)
		{
			var list = new List<T>();
			CollectInChildren(this, includeInactive, list);
			return list.ToArray();
		}

		public void GetComponentsInChildren<T>(List<T> results) => GetComponentsInChildren(false, results);

		public void GetComponentsInChildren<T>(bool includeInactive, List<T> results)
		{
			results.Clear();
			CollectInChildren(this, includeInactive, results);
		}

		private static void CollectInChildren<T>(GameObject go, bool includeInactive, List<T> list)
		{
			if (!includeInactive && !go.activeInHierarchy) return;
			foreach (var c in go.components)
				if (!c.destroyed && c is T t) list.Add(t);
			for (int i = 0; i < go.transform.childCount; i++)
				CollectInChildren(go.transform.GetChild(i).gameObject, includeInactive, list);
		}

		public T GetComponentInParent<T>() => GetComponentInParent<T>(false);

		public T GetComponentInParent<T>(bool includeInactive)
		{
			for (var t = transform; t != null; t = t.parent)
			{
				if (!includeInactive && !t.gameObject.activeInHierarchy) continue;
				var c = t.gameObject.GetComponent<T>();
				if (c != null) return c;
			}
			return default;
		}

		public T[] GetComponentsInParent<T>() => GetComponentsInParent<T>(false);

		public T[] GetComponentsInParent<T>(bool includeInactive)
		{
			var list = new List<T>();
			for (var t = transform; t != null; t = t.parent)
			{
				if (!includeInactive && !t.gameObject.activeInHierarchy) continue;
				foreach (var c in t.gameObject.components)
					if (!c.destroyed && c is T x) list.Add(x);
			}
			return list.ToArray();
		}

		// ---- statics --------------------------------------------------------

		public static GameObject Find(string name)
		{
			// Unity searches active objects only, full hierarchy paths allowed.
			foreach (var root in World.Current.roots)
			{
				var found = FindRecursive(root, name);
				if (found != null) return found;
			}
			return null;
		}

		private static GameObject FindRecursive(GameObject go, string name)
		{
			if (go.destroyed || !go.activeInHierarchy) return null;
			if (go.objectName == name) return go;
			for (int i = 0; i < go.transform.childCount; i++)
			{
				var found = FindRecursive(go.transform.GetChild(i).gameObject, name);
				if (found != null) return found;
			}
			return null;
		}

		public static GameObject FindGameObjectWithTag(string tag)
		{
			var all = FindGameObjectsWithTag(tag);
			return all.Length > 0 ? all[0] : null;
		}

		public static GameObject[] FindGameObjectsWithTag(string tag)
		{
			var list = new List<GameObject>();
			foreach (var root in World.Current.roots.ToArray())
				CollectByTag(root, tag, list);
			return list.ToArray();
		}

		private static void CollectByTag(GameObject go, string tag, List<GameObject> list)
		{
			if (go.destroyed || !go.activeInHierarchy) return;
			if (go.tagField == tag) list.Add(go);
			for (int i = 0; i < go.transform.childCount; i++)
				CollectByTag(go.transform.GetChild(i).gameObject, tag, list);
		}

		public void SendMessage(string methodName) => SendMessage(methodName, null, SendMessageOptions.RequireReceiver);
		public void SendMessage(string methodName, object value) => SendMessage(methodName, value, SendMessageOptions.RequireReceiver);

		public void SendMessage(string methodName, object value, SendMessageOptions options)
		{
			foreach (var c in components.ToArray())
			{
				if (c.destroyed || !(c is MonoBehaviour mb)) continue;
				var method = mb.GetType().GetMethod(methodName,
					System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic);
				if (method == null) continue;
				method.Invoke(mb, method.GetParameters().Length == 0 ? null : new[] { value });
			}
		}

		public void BroadcastMessage(string methodName) => BroadcastMessage(methodName, null, SendMessageOptions.DontRequireReceiver);
		public void BroadcastMessage(string methodName, object value) => BroadcastMessage(methodName, value, SendMessageOptions.DontRequireReceiver);

		public void BroadcastMessage(string methodName, object value, SendMessageOptions options)
		{
			SendMessage(methodName, value, options);
			for (int i = 0; i < transform.childCount; i++)
				transform.GetChild(i).gameObject.BroadcastMessage(methodName, value, options);
		}
	}

	public enum SendMessageOptions { RequireReceiver, DontRequireReceiver }
}
