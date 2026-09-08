using System;
using System.Collections.Generic;
using System.Reflection;

namespace UnityEngine.Events
{
	public delegate void UnityAction();
	public delegate void UnityAction<T0>(T0 arg0);
	public delegate void UnityAction<T0, T1>(T0 arg0, T1 arg1);
	public delegate void UnityAction<T0, T1, T2>(T0 arg0, T1 arg1, T2 arg2);
	public delegate void UnityAction<T0, T1, T2, T3>(T0 arg0, T1 arg1, T2 arg2, T3 arg3);

	// Persistent (serialized) listener support: ability prefabs wire
	// InstantAbility.funcOnEnter / QuickSpell.spellFunc to UseAbility /
	// CastAbility / CastInvisibility through m_PersistentCalls, which the
	// Export loader deserializes into these fields (sim path - SEMANTICS.md
	// §2.7). Persistent calls run before runtime listeners, in list order.
	public enum UnityEventCallState { Off = 0, EditorAndRuntime = 1, RuntimeOnly = 2 }

	public enum PersistentListenerMode { EventDefined = 0, Void = 1, Object = 2, Int = 3, Float = 4, String = 5, Bool = 6 }

	[Serializable]
	public class ArgumentCache
	{
		public Object m_ObjectArgument;
		public string m_ObjectArgumentAssemblyTypeName;
		public int m_IntArgument;
		public float m_FloatArgument;
		public string m_StringArgument;
		public bool m_BoolArgument;
	}

	[Serializable]
	public class PersistentCall
	{
		public Object m_Target;
		public string m_TargetAssemblyTypeName;
		public string m_MethodName;
		public int m_Mode = (int)PersistentListenerMode.EventDefined;
		public ArgumentCache m_Arguments = new ArgumentCache();
		public int m_CallState = (int)UnityEventCallState.RuntimeOnly;

		internal void Invoke(object[] eventArgs, Type[] eventArgTypes)
		{
			if (m_CallState == (int)UnityEventCallState.Off) return;
			if (m_Target == null || string.IsNullOrEmpty(m_MethodName)) return;
			object[] args;
			Type[] argTypes;
			switch ((PersistentListenerMode)m_Mode)
			{
				case PersistentListenerMode.EventDefined: args = eventArgs; argTypes = eventArgTypes; break;
				case PersistentListenerMode.Void: args = Array.Empty<object>(); argTypes = Type.EmptyTypes; break;
				case PersistentListenerMode.Object: args = new object[] { m_Arguments.m_ObjectArgument }; argTypes = new[] { typeof(Object) }; break;
				case PersistentListenerMode.Int: args = new object[] { m_Arguments.m_IntArgument }; argTypes = new[] { typeof(int) }; break;
				case PersistentListenerMode.Float: args = new object[] { m_Arguments.m_FloatArgument }; argTypes = new[] { typeof(float) }; break;
				case PersistentListenerMode.String: args = new object[] { m_Arguments.m_StringArgument }; argTypes = new[] { typeof(string) }; break;
				case PersistentListenerMode.Bool: args = new object[] { m_Arguments.m_BoolArgument }; argTypes = new[] { typeof(bool) }; break;
				default: return;
			}
			var method = FindMethod(m_Target.GetType(), m_MethodName, argTypes);
			if (method == null)
				throw new MissingMethodException(m_Target.GetType().Name, m_MethodName);
			method.Invoke(m_Target, args);
		}

		private static MethodInfo FindMethod(Type type, string name, Type[] argTypes)
		{
			for (var t = type; t != null; t = t.BaseType)
			{
				var method = t.GetMethod(name,
					BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly,
					null, argTypes, null);
				if (method != null) return method;
			}
			// Object-mode arguments may target a narrower parameter type.
			for (var t = type; t != null; t = t.BaseType)
				foreach (var method in t.GetMethods(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
					if (method.Name == name && method.GetParameters().Length == argTypes.Length) return method;
			return null;
		}
	}

	[Serializable]
	public class PersistentCallGroup
	{
		public List<PersistentCall> m_Calls = new List<PersistentCall>();
	}

	[Serializable]
	public abstract class UnityEventBase
	{
		[SerializeField]
		internal PersistentCallGroup m_PersistentCalls = new PersistentCallGroup();

		public int GetPersistentEventCount() => m_PersistentCalls.m_Calls.Count;
		public Object GetPersistentTarget(int index) => m_PersistentCalls.m_Calls[index].m_Target;
		public string GetPersistentMethodName(int index) => m_PersistentCalls.m_Calls[index].m_MethodName;
		public void SetPersistentListenerState(int index, UnityEventCallState state)
			=> m_PersistentCalls.m_Calls[index].m_CallState = (int)state;

		public void RemoveAllListeners() => ClearAll();
		internal abstract void ClearAll();

		internal void InvokePersistent(object[] args, Type[] argTypes)
		{
			var calls = m_PersistentCalls.m_Calls;
			for (int i = 0; i < calls.Count; i++) calls[i].Invoke(args, argTypes);
		}
	}

	[Serializable]
	public class UnityEvent : UnityEventBase
	{
		private readonly List<UnityAction> listeners = new List<UnityAction>();
		public void AddListener(UnityAction call) => listeners.Add(call);
		public void RemoveListener(UnityAction call) => listeners.Remove(call);
		public void Invoke()
		{
			InvokePersistent(Array.Empty<object>(), Type.EmptyTypes);
			foreach (var l in listeners.ToArray()) l();
		}
		internal override void ClearAll() => listeners.Clear();
	}

	[Serializable]
	public class UnityEvent<T0> : UnityEventBase
	{
		private static readonly Type[] argTypes = { typeof(T0) };
		private readonly List<UnityAction<T0>> listeners = new List<UnityAction<T0>>();
		public void AddListener(UnityAction<T0> call) => listeners.Add(call);
		public void RemoveListener(UnityAction<T0> call) => listeners.Remove(call);
		public void Invoke(T0 arg0)
		{
			InvokePersistent(new object[] { arg0 }, argTypes);
			foreach (var l in listeners.ToArray()) l(arg0);
		}
		internal override void ClearAll() => listeners.Clear();
	}

	[Serializable]
	public class UnityEvent<T0, T1> : UnityEventBase
	{
		private static readonly Type[] argTypes = { typeof(T0), typeof(T1) };
		private readonly List<UnityAction<T0, T1>> listeners = new List<UnityAction<T0, T1>>();
		public void AddListener(UnityAction<T0, T1> call) => listeners.Add(call);
		public void RemoveListener(UnityAction<T0, T1> call) => listeners.Remove(call);
		public void Invoke(T0 arg0, T1 arg1)
		{
			InvokePersistent(new object[] { arg0, arg1 }, argTypes);
			foreach (var l in listeners.ToArray()) l(arg0, arg1);
		}
		internal override void ClearAll() => listeners.Clear();
	}
}
