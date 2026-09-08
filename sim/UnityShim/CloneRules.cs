using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using UnityEngine;

namespace UnityShim
{
	// Which fields Unity serializes: public or [SerializeField], non-static,
	// non-readonly, of a serializable type. Shared by Instantiate cloning and
	// the Export loader.
	public static class SerializationRules
	{
		private static readonly Dictionary<Type, FieldInfo[]> cache = new Dictionary<Type, FieldInfo[]>();

		public static FieldInfo[] SerializedFields(Type type)
		{
			if (cache.TryGetValue(type, out var cached)) return cached;
			var chain = new List<Type>();
			for (var t = type; t != null && t != typeof(object) && t != typeof(MonoBehaviour)
				&& t != typeof(Behaviour) && t != typeof(Component) && t != typeof(UnityEngine.Object)
				&& t != typeof(ScriptableObject); t = t.BaseType)
				chain.Add(t);
			chain.Reverse(); // base-class fields first, like Unity
			var fields = new List<FieldInfo>();
			foreach (var t in chain)
				foreach (var f in t.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
					if (IsSerialized(f)) fields.Add(f);
			var result = fields.ToArray();
			cache[type] = result;
			return result;
		}

		public static bool IsSerialized(FieldInfo f)
		{
			if (f.IsStatic || f.IsInitOnly || f.IsLiteral) return false;
			if (f.IsDefined(typeof(NonSerializedAttribute), false)) return false;
			if (!f.IsPublic && !f.IsDefined(typeof(SerializeField), false)) return false;
			return IsSerializableType(f.FieldType);
		}

		public static bool IsSerializableType(Type t)
		{
			if (t.IsArray) return t.GetArrayRank() == 1 && IsSerializableType(t.GetElementType());
			if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(List<>))
				return IsSerializableType(t.GetGenericArguments()[0]);
			if (t.IsPrimitive || t.IsEnum || t == typeof(string)) return true;
			if (typeof(UnityEngine.Object).IsAssignableFrom(t)) return true;
			if (t.IsGenericType) return false; // Unity doesn't serialize other generics
			if (t.IsDefined(typeof(SerializableAttribute), false)) return true;
			return false;
		}
	}

	// Object.Instantiate semantics: constructor + field initialisers run, then
	// serialized fields are copied; references into the cloned hierarchy are
	// remapped; [Serializable] plain classes are deep-copied by value.
	public static class CloneRules
	{
		public static GameObject CloneHierarchy(GameObject source, Transform parent)
		{
			var map = new Dictionary<UnityEngine.Object, UnityEngine.Object>();
			GameObject clone = CloneGameObjectRecursive(source, map);
			foreach (var pair in map)
				if (pair.Key is Component srcComp && pair.Value is Component dstComp && !(srcComp is Transform))
					CopySerializedFields(srcComp, dstComp, map);
			clone.transform.SetParentInternal(parent, true);
			return clone;
		}

		private static GameObject CloneGameObjectRecursive(GameObject source, Dictionary<UnityEngine.Object, UnityEngine.Object> map)
		{
			var clone = GameObject.CreateDetached(source.objectName, false);
			map[source] = clone;
			map[source.transform] = clone.transform;
			clone.layer = source.layer;
			clone.tagField = source.tagField;
			clone.activeSelfField = source.activeSelfField;
			clone.transform.localPositionField = source.transform.localPositionField;
			clone.transform.localRotationField = source.transform.localRotationField;
			clone.transform.localScaleField = source.transform.localScaleField;
			foreach (var c in source.components)
			{
				if (c is Transform || c.destroyed) continue;
				var copy = clone.AddComponentSilent(c.GetType());
				if (copy is Behaviour b && c is Behaviour sb) b.enabledSelf = sb.enabledSelf;
				map[c] = copy;
			}
			for (int i = 0; i < source.transform.childCount; i++)
			{
				var childClone = CloneGameObjectRecursive(source.transform.GetChild(i).gameObject, map);
				childClone.transform.SetParentInternal(clone.transform, false);
			}
			return clone;
		}

		public static void CopySerializedFields(object source, object target, Dictionary<UnityEngine.Object, UnityEngine.Object> map)
		{
			foreach (var f in SerializationRules.SerializedFields(source.GetType()))
				f.SetValue(target, CloneValue(f.GetValue(source), map));
		}

		public static object CloneValue(object value, Dictionary<UnityEngine.Object, UnityEngine.Object> map)
		{
			if (value == null) return null;
			var t = value.GetType();
			if (t.IsPrimitive || t.IsEnum || value is string) return value;
			if (value is UnityEngine.Object uo)
				return map != null && map.TryGetValue(uo, out var mapped) ? mapped : value;
			if (value is Array array)
			{
				var elem = t.GetElementType();
				var copy = Array.CreateInstance(elem, array.Length);
				for (int i = 0; i < array.Length; i++) copy.SetValue(CloneValue(array.GetValue(i), map), i);
				return copy;
			}
			if (value is IList list && t.IsGenericType)
			{
				var copy = (IList)Activator.CreateInstance(t);
				foreach (var item in list) copy.Add(CloneValue(item, map));
				return copy;
			}
			if (t.IsValueType)
			{
				// structs: copy already made by boxing; clone serializable members in place
				object boxed = value;
				foreach (var f in SerializationRules.SerializedFields(t))
					f.SetValue(boxed, CloneValue(f.GetValue(value), map));
				return boxed;
			}
			// [Serializable] plain class: run ctor, deep-copy serialized fields
			var instance = Activator.CreateInstance(t, true);
			foreach (var f in SerializationRules.SerializedFields(t))
				f.SetValue(instance, CloneValue(f.GetValue(value), map));
			return instance;
		}

		public static ScriptableObject CloneScriptableObject(ScriptableObject source)
		{
			var clone = (ScriptableObject)Activator.CreateInstance(source.GetType(), true);
			clone.objectName = source.objectName + "(Clone)";
			CopySerializedFields(source, clone, null);
			ScriptableObject.DispatchOnEnable(clone); // after fields, like Unity deserialization
			return clone;
		}

		// Awake then OnEnable for a freshly instantiated active hierarchy:
		// collected depth-first, stable-sorted by script execution order, all
		// Awakes then all OnEnables (same batch semantics as scene load).
		public static void DispatchAwake(GameObject root)
		{
			if (!root.activeInHierarchy || root.isAsset) return;
			var behaviours = new List<MonoBehaviour>();
			World.CollectActiveBehaviours(root, behaviours);
			World.DispatchActivationBatch(behaviours);
		}
	}
}
