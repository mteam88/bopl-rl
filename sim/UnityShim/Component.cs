using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using UnityShim;

namespace UnityEngine
{
	public class Component : Object
	{
		internal GameObject gameObjectField;

		internal void AttachTo(GameObject go) => gameObjectField = go;

		public GameObject gameObject => gameObjectField;
		public Transform transform => gameObjectField.transform;

		public override string name
		{
			get => gameObjectField != null ? gameObjectField.name : objectName;
			set { if (gameObjectField != null) gameObjectField.name = value; else objectName = value; }
		}

		public string tag
		{
			get => gameObjectField.tag;
			set => gameObjectField.tag = value;
		}

		public bool CompareTag(string other) => gameObjectField.CompareTag(other);

		public T GetComponent<T>() => gameObjectField.GetComponent<T>();
		public Component GetComponent(Type type) => gameObjectField.GetComponent(type);
		public Component GetComponent(string typeName) => gameObjectField.GetComponent(typeName);
		public bool TryGetComponent<T>(out T component) => gameObjectField.TryGetComponent(out component);
		public T[] GetComponents<T>() => gameObjectField.GetComponents<T>();
		public void GetComponents<T>(List<T> results) => gameObjectField.GetComponents(results);
		public T GetComponentInChildren<T>() => gameObjectField.GetComponentInChildren<T>();
		public T GetComponentInChildren<T>(bool includeInactive) => gameObjectField.GetComponentInChildren<T>(includeInactive);
		public T[] GetComponentsInChildren<T>() => gameObjectField.GetComponentsInChildren<T>();
		public T[] GetComponentsInChildren<T>(bool includeInactive) => gameObjectField.GetComponentsInChildren<T>(includeInactive);
		public void GetComponentsInChildren<T>(List<T> results) => gameObjectField.GetComponentsInChildren(results);
		public void GetComponentsInChildren<T>(bool includeInactive, List<T> results) => gameObjectField.GetComponentsInChildren(includeInactive, results);
		public T GetComponentInParent<T>() => gameObjectField.GetComponentInParent<T>();
		public T GetComponentInParent<T>(bool includeInactive) => gameObjectField.GetComponentInParent<T>(includeInactive);
		public T[] GetComponentsInParent<T>() => gameObjectField.GetComponentsInParent<T>();
		public T[] GetComponentsInParent<T>(bool includeInactive) => gameObjectField.GetComponentsInParent<T>(includeInactive);

		public void SendMessage(string methodName) => gameObjectField.SendMessage(methodName);
		public void SendMessage(string methodName, object value) => gameObjectField.SendMessage(methodName, value);
		public void BroadcastMessage(string methodName) => gameObjectField.BroadcastMessage(methodName);
		public void BroadcastMessage(string methodName, object value) => gameObjectField.BroadcastMessage(methodName, value);
	}

	public class Behaviour : Component
	{
		internal bool enabledSelf = true;

		public bool enabled
		{
			get => enabledSelf;
			set
			{
				if (this is MonoBehaviour mb) mb.SetEnabledInternal(value);
				else enabledSelf = value;
			}
		}

		public bool isActiveAndEnabled => enabledSelf && !destroyed && gameObjectField != null && gameObjectField.activeInHierarchy;
	}

	internal enum MessageId { Awake, OnEnable, Start, Update, FixedUpdate, LateUpdate, OnDisable, OnDestroy }

	public class MonoBehaviour : Behaviour
	{
		internal bool awakeRan;        // Awake dispatched (implies OnEnable eligibility)
		internal bool startRan;
		internal bool startQueued;
		internal bool enableRan;       // currently in the OnEnable'd state

		private List<Coroutine> ownedCoroutines;
		private List<PendingInvoke> pendingInvokes;

		private sealed class PendingInvoke
		{
			public string method;
			public float dueTime;
			public float repeatRate; // 0 = one shot
		}

		// ---- reflection message cache ---------------------------------------

		private static readonly Dictionary<Type, MethodInfo[]> messageCache = new Dictionary<Type, MethodInfo[]>();
		private static readonly string[] messageNames = { "Awake", "OnEnable", "Start", "Update", "FixedUpdate", "LateUpdate", "OnDisable", "OnDestroy" };

		private static MethodInfo[] MessagesFor(Type type)
		{
			if (messageCache.TryGetValue(type, out var cached)) return cached;
			var methods = new MethodInfo[messageNames.Length];
			for (var t = type; t != null && t != typeof(MonoBehaviour); t = t.BaseType)
			{
				foreach (var m in t.GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
				{
					if (m.GetParameters().Length != 0) continue;
					int index = Array.IndexOf(messageNames, m.Name);
					if (index >= 0 && methods[index] == null) methods[index] = m;
				}
			}
			messageCache[type] = methods;
			return methods;
		}

		internal void CallMessage(MessageId id)
		{
			if (destroyed && id != MessageId.OnDestroy && id != MessageId.OnDisable) return;
			var m = MessagesFor(GetType())[(int)id];
			if (m == null) return;
			var result = m.Invoke(this, null);
			if (result is IEnumerator routine) StartCoroutine(routine);
		}

		internal void CallStart() => CallMessage(MessageId.Start);

		// ---- lifecycle -------------------------------------------------------

		// GameObject became active in hierarchy (or component added/enabled on an active one).
		// Awake runs even for disabled behaviours (Unity ties Awake to GameObject activity).
		internal void OnBecameActive()
		{
			if (destroyed || isAsset) return;
			DispatchAwakeIfNeeded();
			if (enabledSelf) DispatchEnable();
		}

		internal void OnBecameInactive()
		{
			if (destroyed) return;
			DispatchDisable();
		}

		internal void DispatchAwakeIfNeeded()
		{
			if (awakeRan || isAsset) return;
			awakeRan = true;
			CallMessage(MessageId.Awake);
		}

		internal void DispatchEnable()
		{
			if (enableRan || destroyed) return;
			enableRan = true;
			CallMessage(MessageId.OnEnable);
			World.Current.QueueStart(this);
		}

		internal void DispatchDisable()
		{
			if (!enableRan) return;
			enableRan = false;
			CallMessage(MessageId.OnDisable);
		}

		internal void SetEnabledInternal(bool value)
		{
			if (enabledSelf == value) return;
			enabledSelf = value;
			if (isAsset || destroyed || gameObjectField == null || !gameObjectField.activeInHierarchy) return;
			if (value) { DispatchAwakeIfNeeded(); DispatchEnable(); }
			else DispatchDisable();
		}

		internal void CleanupOnDestroy()
		{
			StopAllCoroutines();
			CancelInvoke();
		}

		// ---- coroutines ------------------------------------------------------

		public Coroutine StartCoroutine(IEnumerator routine)
		{
			if (routine == null) throw new NullReferenceException("routine is null");
			var co = World.Current.coroutines.Start(this, routine);
			(ownedCoroutines ??= new List<Coroutine>()).Add(co);
			return co;
		}

		public Coroutine StartCoroutine(string methodName) => StartCoroutine(methodName, null);

		public Coroutine StartCoroutine(string methodName, object value)
		{
			var m = GetType().GetMethod(methodName, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
			if (m == null) throw new MissingMethodException(GetType().Name, methodName);
			var routine = (IEnumerator)m.Invoke(this, m.GetParameters().Length == 0 ? null : new[] { value });
			var co = World.Current.coroutines.Start(this, routine, methodName);
			(ownedCoroutines ??= new List<Coroutine>()).Add(co);
			return co;
		}

		public void StopCoroutine(Coroutine routine)
		{
			if (routine != null) World.Current.coroutines.Stop(routine);
		}

		public void StopCoroutine(IEnumerator routine)
		{
			if (routine != null) World.Current.coroutines.Stop(this, routine);
		}

		public void StopCoroutine(string methodName) => World.Current.coroutines.Stop(this, methodName);

		public void StopAllCoroutines()
		{
			if (ownedCoroutines == null) return;
			foreach (var co in ownedCoroutines) World.Current.coroutines.Stop(co);
			ownedCoroutines.Clear();
		}

		// ---- Invoke ----------------------------------------------------------

		public void Invoke(string methodName, float time)
			=> (pendingInvokes ??= new List<PendingInvoke>()).Add(new PendingInvoke { method = methodName, dueTime = Time.time + time });

		public void InvokeRepeating(string methodName, float time, float repeatRate)
			=> (pendingInvokes ??= new List<PendingInvoke>()).Add(new PendingInvoke { method = methodName, dueTime = Time.time + time, repeatRate = repeatRate });

		public void CancelInvoke() => pendingInvokes?.Clear();

		public void CancelInvoke(string methodName) => pendingInvokes?.RemoveAll(p => p.method == methodName);

		public bool IsInvoking() => pendingInvokes != null && pendingInvokes.Count > 0;

		public bool IsInvoking(string methodName) => pendingInvokes != null && pendingInvokes.Exists(p => p.method == methodName);

		internal void DispatchInvokes()
		{
			if (pendingInvokes == null || pendingInvokes.Count == 0) return;
			for (int i = 0; i < pendingInvokes.Count; i++)
			{
				var p = pendingInvokes[i];
				if (Time.time < p.dueTime) continue;
				if (p.repeatRate > 0f) p.dueTime += p.repeatRate;
				else { pendingInvokes.RemoveAt(i); i--; }
				var m = GetType().GetMethod(p.method, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
				m?.Invoke(this, null);
			}
		}

		public static void print(object message) => Debug.Log(message);
	}
}
