using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace BoplSim.Export
{
	// Pass 2 of scene/prefab loading: apply exported propertyPath → value pairs
	// onto shim objects by reflection, mirroring Unity's serialized→runtime rules:
	// field initialisers already ran (constructor), serialized values overwrite.
	internal static class PropertyApplier
	{
		// Editor-only / structural paths that must not reach the walker.
		private static readonly string[] SkipPrefixes =
		{
			"m_ObjectHideFlags", "m_CorrespondingSourceObject", "m_PrefabInstance",
			"m_PrefabAsset", "m_GameObject", "m_Script", "m_EditorHideFlags",
			"m_EditorClassIdentifier",
		};

		public static void Apply(object target, JObject properties, ExportDatabase db)
		{
			foreach (var pair in properties)
			{
				try { ApplyOne(target, pair.Key, pair.Value, db); }
				catch (Exception error)
				{
					db.Warn("apply failed " + target.GetType().Name + "." + pair.Key + ": " + error.GetType().Name);
				}
			}
		}

		private static void ApplyOne(object target, string path, JToken value, ExportDatabase db)
		{
			foreach (var prefix in SkipPrefixes)
				if (path.StartsWith(prefix, StringComparison.Ordinal)) return;

			// m_Name: meaningful for standalone assets (ScriptableObjects) only.
			if (path == "m_Name" || path.StartsWith("m_Name.", StringComparison.Ordinal))
			{
				if (path == "m_Name" && target is ScriptableObject so && value.Type == JTokenType.String)
					so.name = (string)value;
				return;
			}

			if (target is Transform transform && ApplyTransform(transform, path, value)) return;

			if (target is Behaviour behaviour && path == "m_Enabled")
			{
				// Direct flag write: lifecycle dispatch happens in the activation pass.
				behaviour.enabledSelf = (bool)value;
				return;
			}

			// Struct markers carry no data (leaves follow as separate entries).
			if (value is JObject marker && marker.Count == 1 && marker["serializedType"] != null) return;

			SetPath(target, path, value, db);
		}

		// Transform serialized fields; hierarchy comes from JSON nesting.
		private static bool ApplyTransform(Transform transform, string path, JToken value)
		{
			switch (path)
			{
				case "m_LocalRotation.x": { var q = transform.localRotation; q.x = Float(value); transform.localRotation = q; return true; }
				case "m_LocalRotation.y": { var q = transform.localRotation; q.y = Float(value); transform.localRotation = q; return true; }
				case "m_LocalRotation.z": { var q = transform.localRotation; q.z = Float(value); transform.localRotation = q; return true; }
				case "m_LocalRotation.w": { var q = transform.localRotation; q.w = Float(value); transform.localRotation = q; return true; }
				case "m_LocalPosition.x": { var p = transform.localPosition; p.x = Float(value); transform.localPosition = p; return true; }
				case "m_LocalPosition.y": { var p = transform.localPosition; p.y = Float(value); transform.localPosition = p; return true; }
				case "m_LocalPosition.z": { var p = transform.localPosition; p.z = Float(value); transform.localPosition = p; return true; }
				case "m_LocalScale.x": { var s = transform.localScale; s.x = Float(value); transform.localScale = s; return true; }
				case "m_LocalScale.y": { var s = transform.localScale; s.y = Float(value); transform.localScale = s; return true; }
				case "m_LocalScale.z": { var s = transform.localScale; s.z = Float(value); transform.localScale = s; return true; }
			}
			if (path.StartsWith("m_Children", StringComparison.Ordinal)) return true;
			if (path == "m_Father" || path == "m_RootOrder" || path == "m_ConstrainProportionsScale") return true;
			if (path.StartsWith("m_LocalEulerAnglesHint", StringComparison.Ordinal)) return true;
			if (path.StartsWith("m_LocalRotation", StringComparison.Ordinal)
				|| path.StartsWith("m_LocalPosition", StringComparison.Ordinal)
				|| path.StartsWith("m_LocalScale", StringComparison.Ordinal)) return true; // struct markers
			return false; // RectTransform anchors etc. fall through to the walker
		}

		// ---- path walking ------------------------------------------------------

		private abstract class Segment { }
		private sealed class NameSegment : Segment { public string Name; }
		private sealed class IndexSegment : Segment { public int Index; }
		private sealed class SizeSegment : Segment { }

		private static List<Segment> Tokenize(string path)
		{
			var tokens = path.Split('.');
			var segments = new List<Segment>(tokens.Length);
			for (int i = 0; i < tokens.Length; i++)
			{
				if (tokens[i] == "Array" && i + 1 < tokens.Length)
				{
					if (tokens[i + 1] == "size") { segments.Add(new SizeSegment()); i++; continue; }
					if (tokens[i + 1].StartsWith("data[", StringComparison.Ordinal))
					{
						int idx = int.Parse(tokens[i + 1].Substring(5, tokens[i + 1].Length - 6));
						segments.Add(new IndexSegment { Index = idx });
						i++;
						continue;
					}
				}
				segments.Add(new NameSegment { Name = tokens[i] });
			}
			return segments;
		}

		private static void SetPath(object target, string path, JToken value, ExportDatabase db)
		{
			var segments = Tokenize(path);
			Walk(target, segments, 0, path, value, db);
		}

		// Recursive descent; returns the (possibly re-boxed) container so value
		// types are written back through the chain.
		private static object Walk(object container, List<Segment> segments, int index,
			string fullPath, JToken value, ExportDatabase db)
		{
			var segment = segments[index];
			bool last = index == segments.Count - 1;

			if (segment is SizeSegment)
			{
				// Sizing applies to the member resolved by the previous segment;
				// handled there (below) - reaching here means path started with size.
				return container;
			}

			if (segment is IndexSegment indexSegment)
			{
				if (container is string) return container; // strings double-exported as char arrays
				if (!(container is IList list) || indexSegment.Index >= list.Count)
				{
					db.Warn("index out of range: " + fullPath);
					return container;
				}
				var elementType = container.GetType().IsArray
					? container.GetType().GetElementType()
					: container.GetType().GetGenericArguments()[0];
				if (last)
				{
					object decoded = Decode(value, elementType, fullPath, db);
					if (!ReferenceEquals(decoded, SkipValue)) list[indexSegment.Index] = decoded;
					return container;
				}
				object element = list[indexSegment.Index];
				if (element == null)
				{
					element = CreateInstance(elementType, fullPath, db);
					if (element == null) return container;
				}
				element = Walk(element, segments, index + 1, fullPath, value, db);
				list[indexSegment.Index] = element;
				return container;
			}

			var name = ((NameSegment)segment).Name;
			var member = ResolveMember(container.GetType(), name);
			if (member == null)
			{
				db.Warn("no member " + container.GetType().Name + "." + name);
				return container;
			}
			Type memberType = member is FieldInfo f ? f.FieldType : ((PropertyInfo)member).PropertyType;

			// Next segment sizes this member (array/list allocation).
			if (!last && segments[index + 1] is SizeSegment)
			{
				if (memberType == typeof(string)) return container; // string char-array shadow
				int size = (int)value;
				object sized = AllocateSized(memberType, GetMember(member, container), size, fullPath, db);
				if (sized != null) SetMember(member, ref container, sized);
				return container;
			}

			if (last)
			{
				object decoded = Decode(value, memberType, fullPath, db);
				if (!ReferenceEquals(decoded, SkipValue)) SetMember(member, ref container, decoded);
				return container;
			}

			object current = GetMember(member, container);
			if (current == null)
			{
				if (memberType == typeof(string)) return container;
				current = CreateInstance(memberType, fullPath, db);
				if (current == null) return container;
			}
			current = Walk(current, segments, index + 1, fullPath, value, db);
			// Write back: value types always; reference types only if freshly created.
			SetMember(member, ref container, current);
			return container;
		}

		private static object AllocateSized(Type memberType, object current, int size, string path, ExportDatabase db)
		{
			if (memberType.IsArray)
			{
				var array = (Array)current;
				if (array != null && array.Length == size) return array;
				var fresh = Array.CreateInstance(memberType.GetElementType(), size);
				if (array != null) Array.Copy(array, fresh, Math.Min(array.Length, size));
				return fresh;
			}
			if (typeof(IList).IsAssignableFrom(memberType) && memberType.IsGenericType)
			{
				var list = (IList)(current ?? Activator.CreateInstance(memberType));
				var elementType = memberType.GetGenericArguments()[0];
				while (list.Count > size) list.RemoveAt(list.Count - 1);
				while (list.Count < size)
					list.Add(elementType.IsValueType ? Activator.CreateInstance(elementType) : null);
				return list;
			}
			db.Warn("cannot size " + memberType.Name + " at " + path);
			return null;
		}

		private static object CreateInstance(Type type, string path, ExportDatabase db)
		{
			if (type.IsAbstract || typeof(UnityEngine.Object).IsAssignableFrom(type))
			{
				db.Warn("cannot create intermediate " + type.Name + " at " + path);
				return null;
			}
			try { return Activator.CreateInstance(type, true); }
			catch { db.Warn("no parameterless ctor for " + type.Name + " at " + path); return null; }
		}

		// ---- member resolution ---------------------------------------------------

		private static readonly Dictionary<(Type, string), MemberInfo> memberCache
			= new Dictionary<(Type, string), MemberInfo>();

		private static MemberInfo ResolveMember(Type type, string name)
		{
			var key = (type, name);
			lock (memberCache)
			{
				if (memberCache.TryGetValue(key, out var cached)) return cached;
			}
			MemberInfo resolved = FindField(type, name);
			if (resolved == null && name.StartsWith("m_", StringComparison.Ordinal) && name.Length > 2)
			{
				// Unity builtin components serialize m_PascalCase for camelCase members.
				string camel = char.ToLowerInvariant(name[2]) + name.Substring(3);
				resolved = FindField(type, camel)
					?? (MemberInfo)type.GetProperty(camel, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
			}
			resolved ??= type.GetProperty(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
			if (resolved is PropertyInfo p && (!p.CanWrite || !p.CanRead)) resolved = null;
			lock (memberCache) memberCache[key] = resolved;
			return resolved;
		}

		private static FieldInfo FindField(Type type, string name)
		{
			for (var t = type; t != null; t = t.BaseType)
			{
				var field = t.GetField(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly);
				if (field != null) return field;
			}
			return null;
		}

		private static object GetMember(MemberInfo member, object container)
			=> member is FieldInfo f ? f.GetValue(container) : ((PropertyInfo)member).GetValue(container);

		private static void SetMember(MemberInfo member, ref object container, object value)
		{
			if (member is FieldInfo f) f.SetValue(container, value);
			else ((PropertyInfo)member).SetValue(container, value);
		}

		// ---- value decoding --------------------------------------------------------

		internal static readonly object SkipValue = new object();

		private static float Float(JToken token)
		{
			if (token is JObject o)
			{
				if (o["f32"] != null) return BitConverter.Int32BitsToSingle((int)(long)o["f32"]);
				if (o["f64"] != null) return (float)BitConverter.Int64BitsToDouble((long)o["f64"]);
			}
			return (float)token;
		}

		private static object Decode(JToken value, Type type, string path, ExportDatabase db)
		{
			if (value.Type == JTokenType.Null) return null;

			if (value is JObject obj)
			{
				if (obj["f32"] != null || obj["f64"] != null)
				{
					double d = obj["f32"] != null
						? BitConverter.Int32BitsToSingle((int)(long)obj["f32"])
						: BitConverter.Int64BitsToDouble((long)obj["f64"]);
					if (type == typeof(float)) return (float)d;
					if (type == typeof(double)) return d;
					return Convert.ChangeType(d, Nullable.GetUnderlyingType(type) ?? type);
				}
				if (obj["keys"] != null) return DecodeCurve(obj, db);
				if (obj["serializedType"] != null) return SkipValue;
				db.Warn("unknown object value at " + path);
				return SkipValue;
			}

			if (value.Type == JTokenType.String)
			{
				string s = (string)value;
				if (s.StartsWith("GlobalObjectId_V1-", StringComparison.Ordinal))
					return CoerceReference(db.ResolveReference(s, type), type, path, db);
				if (type == typeof(string)) return s;
				if (type == typeof(char)) return s.Length > 0 ? s[0] : '\0';
				db.Warn("string into " + type.Name + " at " + path);
				return SkipValue;
			}

			if (value.Type == JTokenType.Boolean) return (bool)value;

			// Numbers: enums, LayerMask, chars (string-as-char-array), primitives.
			var target = Nullable.GetUnderlyingType(type) ?? type;
			if (target.IsEnum) return Enum.ToObject(target, (long)value);
			if (target == typeof(LayerMask)) return (LayerMask)(int)(long)value;
			if (target == typeof(float) || target == typeof(double))
				return Convert.ChangeType((double)value, target);
			return Convert.ChangeType((long)value, target);
		}

		private static object CoerceReference(object resolved, Type type, string path, ExportDatabase db)
		{
			if (resolved == null) return null;
			if (type.IsInstanceOfType(resolved)) return resolved;
			if (resolved is GameObject go && typeof(Component).IsAssignableFrom(type))
			{
				var component = go.GetComponent(type);
				if (component != null) return component;
			}
			if (resolved is Component c && type == typeof(GameObject)) return c.gameObject;
			db.Warn("reference type mismatch at " + path + ": " + resolved.GetType().Name + " into " + type.Name);
			return null;
		}

		// Exported curves store Fix raw values (the exporter applied BoplFixedMath's
		// (Fix)float cast). Recover floats whose (Fix) cast reproduces the exact raw
		// value, so AnimationCurveFixed.Init() is bit-identical to Unity.
		private static AnimationCurve DecodeCurve(JObject obj, ExportDatabase db)
		{
			var keysJson = (JArray)obj["keys"];
			var keys = new Keyframe[keysJson.Count];
			for (int i = 0; i < keys.Length; i++)
			{
				var k = (JObject)keysJson[i];
				keys[i] = new Keyframe(
					FloatFromFixRaw((long)k["time"], db),
					FloatFromFixRaw((long)k["value"], db),
					FloatFromFixRaw((long)k["inTangent"], db),
					FloatFromFixRaw((long)k["outTangent"], db));
			}
			return new AnimationCurve(keys)
			{
				preWrapMode = (WrapMode)(int)obj["preWrap"],
				postWrapMode = (WrapMode)(int)obj["postWrap"],
			};
		}

		// Mirrors Fix's explicit float cast: (long)(value * 4.2949673E+09f).
		private static long FixRawFromFloat(float value) => (long)(value * 4.2949673E+09f);

		internal static float FloatFromFixRaw(long raw, ExportDatabase db)
		{
			float candidate = (float)(raw / 4294967296.0);
			if (FixRawFromFloat(candidate) == raw) return candidate;
			int bits = BitConverter.SingleToInt32Bits(candidate);
			for (int step = 1; step <= 32; step++)
			{
				float up = BitConverter.Int32BitsToSingle(bits + step);
				if (FixRawFromFloat(up) == raw) return up;
				float down = BitConverter.Int32BitsToSingle(bits - step);
				if (FixRawFromFloat(down) == raw) return down;
			}
			db.Warn("no float preimage for Fix raw " + raw);
			return candidate;
		}
	}
}
