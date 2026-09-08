using System;
using System.Collections.Generic;

namespace UnityEngine.Events
{
	public delegate void UnityAction();
	public delegate void UnityAction<T0>(T0 arg0);
	public delegate void UnityAction<T0, T1>(T0 arg0, T1 arg1);
	public delegate void UnityAction<T0, T1, T2>(T0 arg0, T1 arg1, T2 arg2);
	public delegate void UnityAction<T0, T1, T2, T3>(T0 arg0, T1 arg1, T2 arg2, T3 arg3);

	public abstract class UnityEventBase
	{
		public void RemoveAllListeners() => ClearAll();
		internal abstract void ClearAll();
		public int GetPersistentEventCount() => 0;
	}

	[Serializable]
	public class UnityEvent : UnityEventBase
	{
		private readonly List<UnityAction> listeners = new List<UnityAction>();
		public void AddListener(UnityAction call) => listeners.Add(call);
		public void RemoveListener(UnityAction call) => listeners.Remove(call);
		public void Invoke() { foreach (var l in listeners.ToArray()) l(); }
		internal override void ClearAll() => listeners.Clear();
	}

	[Serializable]
	public class UnityEvent<T0> : UnityEventBase
	{
		private readonly List<UnityAction<T0>> listeners = new List<UnityAction<T0>>();
		public void AddListener(UnityAction<T0> call) => listeners.Add(call);
		public void RemoveListener(UnityAction<T0> call) => listeners.Remove(call);
		public void Invoke(T0 arg0) { foreach (var l in listeners.ToArray()) l(arg0); }
		internal override void ClearAll() => listeners.Clear();
	}

	[Serializable]
	public class UnityEvent<T0, T1> : UnityEventBase
	{
		private readonly List<UnityAction<T0, T1>> listeners = new List<UnityAction<T0, T1>>();
		public void AddListener(UnityAction<T0, T1> call) => listeners.Add(call);
		public void RemoveListener(UnityAction<T0, T1> call) => listeners.Remove(call);
		public void Invoke(T0 arg0, T1 arg1) { foreach (var l in listeners.ToArray()) l(arg0, arg1); }
		internal override void ClearAll() => listeners.Clear();
	}
}
