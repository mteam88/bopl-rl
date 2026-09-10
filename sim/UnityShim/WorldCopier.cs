using System;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace UnityShim
{
	// Deep copy of a World's object graph, for post-warmup episode snapshots
	// (SIM_OVERHAUL.md stage 5, reset item 2). The world must be UNBOUND (its
	// gameplay statics captured into World.gameplayState) so the graph is
	// self-contained: World -> roots/behaviours/coroutines -> GameObjects ->
	// components -> gameplay objects, plus the RlWorldState statics snapshot
	// hanging off gameplayState.
	//
	// Mechanics: MemberwiseClone per object, then rewrite the slots that hold
	// references through an identity map, so aliasing and cycles inside the
	// graph are preserved exactly. Reference slots are located ONCE per type
	// (probe clone + reflection SetValue + IntPtr diff scan of the probe's own
	// allocation - no garbage bits are ever loaded as object references), and
	// every later copy touches them directly through Unsafe, which is what
	// makes a restore ~milliseconds instead of reflection-slow. Structs are
	// copied bitwise by the clone; their nested reference fields are part of
	// the flattened slot list.
	//
	// What is shared, not copied:
	//   - immutables: strings, Type/MemberInfo, boxed primitives/enums
	//   - isAsset shim objects (prefabs, sprites, scene templates): read-only
	//     by contract, shared across episodes by the normal reset path too
	//   - objects from assemblies other than UnityShim/Gameplay and the
	//     collection whitelist (the Export loader's database, comparers...):
	//     read-mostly services; the diagnostic set below records them
	//   - static-target delegate entries; instance-target entries are
	//     re-created against the copied target
	//   - System.Random: replaced by a fresh System.Random(42). It only feeds
	//     presentation (UnityEngine.Random shim, audio pitch), is already a
	//     documented deviation (PORTING.md), and sharing it would let live
	//     episodes mutate the snapshot.
	//
	// NativeAOT: UnityShim and Gameplay are rooted wholesale (Host/rd.xml), so
	// their fields reflect fine. CoreLib collections (List/Dictionary/Queue/...)
	// need one rd.xml entry per closed instantiation; a missing entry throws
	// here and the caller falls back to a full reset (loud, not wrong).
	//
	// Known limitation: a Dictionary/HashSet keyed by the IDENTITY of a copied
	// (world-owned) object would break - the clone's entries carry the source
	// keys' identity hash codes, and the copied keys hash differently. No such
	// collection exists in world state today (reference keys are shared assets:
	// sprites, textures; other keys are values or strings). The fresh-vs-restore
	// checksum gates would catch one appearing.
	public sealed class WorldCopier
	{
		[UnsafeAccessor(UnsafeAccessorKind.Method, Name = "MemberwiseClone")]
		private static extern object ShallowClone(object obj);

		private readonly Dictionary<object, object> map = new Dictionary<object, object>(1 << 14, ReferenceEqualityComparer.Instance);

		// Distinct type names that were shared because they fall outside the
		// copy set (diagnostics; see WorldSnapshot audit).
		public static readonly HashSet<string> SharedTypeNames = new HashSet<string>();
		public static bool CollectSharedTypeNames;
		// Closed CoreLib collection types the copier reflected over: each needs
		// an rd.xml entry for the NativeAOT build (audit; see BOPL_SNAPSHOT_AUDIT).
		public static readonly HashSet<string> PlannedCollectionNames = new HashSet<string>();
		// Copy-scale counters for the last CopyWorld (audit).
		public static int LastObjects, LastArrays, LastStructElements, LastSlowFields;
		// Debug bisect switches: force the reflection paths.
		private static readonly bool slowObjectFields = Environment.GetEnvironmentVariable("BOPL_COPY_SLOW_FIELDS") == "1";
		private static readonly bool slowArrayElements = Environment.GetEnvironmentVariable("BOPL_COPY_SLOW_ARRAYS") == "1";
		// Audit: discovery anomalies (type.field) and struct-array element types.
		public static readonly Dictionary<string, int> AnomalyFields = new Dictionary<string, int>();
		public static readonly Dictionary<string, int> StructArrayElements = new Dictionary<string, int>();

		// Audit: source-graph parent chains (BOPL_COPY_PARENTS=1), for tracing
		// why an object was reached at all.
		public static readonly bool TrackParents = Environment.GetEnvironmentVariable("BOPL_COPY_PARENTS") == "1";
		public static WorldCopier LastCopier;
		public readonly Dictionary<object, object> ParentOf =
			TrackParents ? new Dictionary<object, object>(1 << 14, ReferenceEqualityComparer.Instance) : null;
		private object auditParent;

		// A copy runs with the calling thread unbound from any live world; but
		// probing a type can run its deferred static constructor (NativeAOT
		// defers them to first touch), and a cctor may construct UnityEngine
		// Objects - Texture2D..cctor does - which ask World.Current for an
		// instance id. Bind a throwaway scratch world for the copy so those
		// side effects land there instead of lazily creating a world that
		// poisons the thread's bind slot (the caller's re-Bind then throws).
		[ThreadStatic] private static World scratchWorld;

		public static World CopyWorld(World source)
		{
			World.Bind(scratchWorld ??= new World()); // throws if the caller left a world bound
			try
			{
				var copier = new WorldCopier();
				var copy = (World)copier.Copy(source);
				LastObjects = copier.objects; LastArrays = copier.arrays;
				LastStructElements = copier.structElements; LastSlowFields = copier.slowFieldFixes;
				LastCopier = copier;
				return copy;
			}
			finally { World.Unbind(); }
		}

		// Walk the recorded parent chain from a copied source object to the root,
		// annotating each hop with the parent field that (currently) holds the child.
		public string PathOf(object source)
		{
			var sb = new System.Text.StringBuilder();
			object node = source;
			for (int i = 0; i < 60 && node != null; i++)
			{
				if (sb.Length > 0) sb.Append(" <- ");
				sb.Append(node.GetType().Name);
				if (node is UnityEngine.Object uo) sb.Append('(').Append(uo.name).Append(')');
				if (!ParentOf.TryGetValue(node, out var parent) || parent == null) break;
				if (!(parent is Array))
				{
					string fieldName = null;
					foreach (var field in AllInstanceFields(parent.GetType()))
						if (!field.FieldType.IsValueType && ReferenceEquals(field.GetValue(parent), node)) { fieldName = field.Name; break; }
					if (fieldName != null) sb.Append(" [.").Append(fieldName).Append(']');
				}
				node = parent;
			}
			return sb.ToString();
		}

		private int objects, arrays, structElements, slowFieldFixes;

		// Audit: walk the CLONE graph by reflection and report fields that still
		// hold a SOURCE object which has a copy in the map - i.e. slots the fast
		// path failed to remap (wrong plan offset).
		public List<string> FindMissedSlots(object cloneRoot)
		{
			var missed = new List<string>();
			var visited = new HashSet<object>(ReferenceEqualityComparer.Instance);
			var queue = new Queue<object>();
			var cameFrom = new Dictionary<object, (object parent, string via)>(1 << 14, ReferenceEqualityComparer.Instance);
			string ChainOf(object o)
			{
				var sb = new System.Text.StringBuilder();
				for (int i = 0; i < 25 && o != null; i++)
				{
					if (!cameFrom.TryGetValue(o, out var link)) break;
					sb.Append(link.via).Append(map.ContainsKey(o) ? "(src)" : "").Append(" <- ");
					o = link.parent;
				}
				return sb.ToString();
			}
			void Push(object o, object parent, string via)
			{
				if (o == null) return;
				var t = o.GetType();
				if (o is string || o is Type || o is MemberInfo || t.IsPrimitive || t.IsEnum) return;
				// Mirror the copier's sharing rules: shared objects legitimately
				// stay identical to the source and are not traversed.
				if (o is UnityEngine.Object shim && shim.isAsset) return;
				if (!(o is Delegate) && !t.IsArray && !t.IsValueType && PlanFor(t, o) == null) return;
				if (!visited.Add(o)) return;
				cameFrom[o] = (parent, via + ":" + t.Name);
				if (map.TryGetValue(o, out var mappedTo) && !ReferenceEquals(mappedTo, o) && missed.Count < 8)
					missed.Add(via + " -> " + t.Name + "  CHAIN: " + ChainOf(o));
				queue.Enqueue(o);
			}
			Push(cloneRoot, null, "root");
			while (queue.Count > 0)
			{
				var node = queue.Dequeue();
				var type = node.GetType();
				if (node is Array arr)
				{
					if (!type.GetElementType().IsValueType)
						foreach (var element in arr) Push(element, node, type.Name + "[]");
					else if (IsRefBearing(type.GetElementType()))
					{
						var et = type.GetElementType();
						string tag = et.DeclaringType != null
							? et.DeclaringType.Name + "<" + string.Join(",", Array.ConvertAll(et.GetGenericArguments(), a => a.Name)) + ">." + et.Name
							: et.Name;
						for (int i = 0; i < arr.Length; i++) WalkStruct(arr.GetValue(i), tag + "[]", (v, via) => Push(v, node, via));
					}
					continue;
				}
				if (node is Delegate d)
				{
					foreach (var entry in d.GetInvocationList()) Push(entry.Target, node, type.Name + ".Target");
					continue;
				}
				foreach (var field in AllInstanceFields(type))
				{
					var ft = field.FieldType;
					if (ft.IsPointer || ft == typeof(IntPtr) || ft == typeof(UIntPtr)) continue;
					if (!ft.IsValueType) Push(field.GetValue(node), node, type.Name + "." + field.Name);
					else if (IsRefBearing(ft)) WalkStruct(field.GetValue(node), type.Name + "." + field.Name, (v, via) => Push(v, node, via));
				}
			}
			return missed;
		}

		private static void WalkStruct(object box, string via, Action<object, string> push)
		{
			if (box == null) return;
			foreach (var field in AllInstanceFields(box.GetType()))
			{
				var ft = field.FieldType;
				if (ft.IsPointer || ft == typeof(IntPtr) || ft == typeof(UIntPtr)) continue;
				if (!ft.IsValueType) push(field.GetValue(box), via + "." + field.Name);
				else if (IsRefBearing(ft)) WalkStruct(field.GetValue(box), via + "." + field.Name, push);
			}
		}

		private object Copy(object source)
		{
			if (source == null) return null;
			if (map.TryGetValue(source, out var existing)) return existing;
			if (TrackParents)
			{
				if (!ParentOf.ContainsKey(source)) ParentOf[source] = auditParent;
				var savedParent = auditParent;
				auditParent = source;
				try { return CopyInner(source); }
				finally { auditParent = savedParent; }
			}
			return CopyInner(source);
		}

		private object CopyInner(object source)
		{
			var type = source.GetType();
			// Immutables and metadata: share.
			if (source is string || source is Type || source is MemberInfo || type.IsPrimitive || type.IsEnum) return source;
			if (source is UnityEngine.Object shimObject && shimObject.isAsset) return source;
			if (source is System.Random)
			{
				var fresh = new System.Random(42);
				map[source] = fresh;
				return fresh;
			}
			if (source is Delegate del) return CopyDelegate(del);
			if (type.IsArray) return CopyArray((Array)source);
			if (type.IsValueType) return source; // boxed struct: gameplay never mutates through the box
			var plan = PlanFor(type, source);
			if (plan == null) // outside the copy set: share
			{
				if (CollectSharedTypeNames) lock (SharedTypeNames) SharedTypeNames.Add(type.FullName);
				return source;
			}
			var clone = ShallowClone(source);
			map[source] = clone;
			objects++;
			var slots = plan.RefSlots;
			for (int i = 0; i < slots.Length; i++)
			{
				ref object slot = ref RefAt(clone, slots[i]);
				if (slot != null) slot = Copy(slot);
			}
			var slow = plan.SlowFields;
			if (slow != null)
				for (int i = 0; i < slow.Length; i++)
				{
					slowFieldFixes++;
					var box = slow[i].GetValue(clone);
					if (box == null) continue;
					FixStructByReflection(box);
					slow[i].SetValue(clone, box);
				}
			var slowRefs = plan.SlowRefFields;
			if (slowRefs != null)
				for (int i = 0; i < slowRefs.Length; i++)
				{
					var value = slowRefs[i].GetValue(clone);
					if (value != null) slowRefs[i].SetValue(clone, Copy(value));
				}
			return clone;
		}

		// Reflection fallback for boxed struct graphs (struct array elements,
		// Nullable-with-references fields; both rare and small).
		private void FixStructByReflection(object box)
		{
			var type = box.GetType();
			foreach (var field in AllInstanceFields(type))
			{
				var ft = field.FieldType;
				if (ft.IsPointer || ft == typeof(IntPtr) || ft == typeof(UIntPtr)) continue;
				if (!ft.IsValueType)
				{
					var value = field.GetValue(box);
					if (value != null) field.SetValue(box, Copy(value));
				}
				else if (IsRefBearing(ft))
				{
					var inner = field.GetValue(box);
					if (inner == null) continue;
					FixStructByReflection(inner);
					field.SetValue(box, inner);
				}
			}
		}

		private object CopyArray(Array array)
		{
			long allocBefore = GC.GetAllocatedBytesForCurrentThread();
			var clone = (Array)array.Clone();
			long allocated = GC.GetAllocatedBytesForCurrentThread() - allocBefore;
			map[array] = clone;
			arrays++;
			var elementType = array.GetType().GetElementType();
			if (elementType.IsPointer) return clone;
			if (!elementType.IsValueType)
			{
				// Every element slot is a reference: walk the payload directly.
				ref object first = ref Unsafe.As<byte, object>(ref MemoryMarshal.GetArrayDataReference(clone));
				int length = clone.Length; // rank-1 and multi-dim payloads are both contiguous
				for (int i = 0; i < length; i++)
				{
					ref object slot = ref Unsafe.Add(ref first, i);
					if (slot != null) slot = Copy(slot);
				}
			}
			else if (IsRefBearing(elementType))
			{
				if (array.Rank != 1) throw new NotSupportedException("WorldCopier: multi-dim struct array " + array.GetType());
				int length = clone.Length;
				if (length == 0) return clone;
				// Measured payload OVERSTATES the true payload when the thread
				// allocated anything else inside Clone, so it is only usable as
				// an upper bound; the stride itself comes from probing.
				long measuredPayload = allocated - 2L * IntPtr.Size;
				var plan = slowArrayElements ? null : ElementPlanFor(elementType, array, measuredPayload);
				if (plan != null && (long)plan.Stride * length <= measuredPayload)
				{
					int stride = plan.Stride;
					int words = stride / IntPtr.Size;
					ref byte payload = ref MemoryMarshal.GetArrayDataReference(clone);
					var offsets = plan.RelOffsets;
					for (int e = 0; e < length; e++)
					{
						int elementStart = e * stride;
						if (offsets != null)
						{
							for (int i = 0; i < offsets.Length; i++)
							{
								ref object slot = ref Unsafe.As<byte, object>(ref Unsafe.AddByteOffset(ref payload, elementStart + offsets[i]));
								if (slot != null) slot = Copy(slot);
							}
							continue;
						}
						// Zero-skip: default (all-zero) elements have no refs.
						bool zero = true;
						for (int wordIndex = 0; wordIndex < words && zero; wordIndex++)
							zero = Unsafe.ReadUnaligned<IntPtr>(ref Unsafe.AddByteOffset(ref payload, elementStart + wordIndex * IntPtr.Size)) == IntPtr.Zero;
						if (zero) continue;
						structElements++;
						var boxed = clone.GetValue(e);
						FixStructByReflection(boxed);
						clone.SetValue(boxed, e);
					}
					return clone;
				}
				lock (StructArrayElements)
				{
					StructArrayElements.TryGetValue(elementType.Name, out int c);
					StructArrayElements[elementType.Name] = c + length;
				}
				for (int i = 0; i < length; i++)
				{
					structElements++;
					var box = clone.GetValue(i);
					if (box == null) continue;
					FixStructByReflection(box);
					clone.SetValue(box, i);
				}
			}
			return clone;
		}

		private object CopyDelegate(Delegate source)
		{
			var list = source.GetInvocationList();
			Delegate result = null;
			for (int i = 0; i < list.Length; i++)
			{
				var entry = list[i];
				var copy = entry;
				var target = entry.Target;
				if (target != null)
				{
					var newTarget = Copy(target);
					if (!ReferenceEquals(newTarget, target)) copy = Delegate.CreateDelegate(entry.GetType(), newTarget, entry.Method);
				}
				result = result == null ? copy : Delegate.Combine(result, copy);
			}
			map[source] = result;
			return result;
		}

		// ---- slot access --------------------------------------------------------

		private sealed class RawData { public byte Data; }

		[MethodImpl(MethodImplOptions.AggressiveInlining)]
		private static ref object RefAt(object o, int offset)
			=> ref Unsafe.As<byte, object>(ref Unsafe.AddByteOffset(ref Unsafe.As<RawData>(o).Data, offset));

		// ---- type plans ---------------------------------------------------------

		private sealed class Plan
		{
			public int[] RefSlots;           // byte offsets of reference slots, from object data start
			public FieldInfo[] SlowFields;   // ref-bearing struct fields handled by reflection (discovery anomaly, Nullable); null when none
			public FieldInfo[] SlowRefFields; // reference fields handled by reflection (discovery anomaly); null when none
		}

		private static readonly Dictionary<Type, Plan> plans = new Dictionary<Type, Plan>();
		private static readonly Dictionary<Type, bool> refBearing = new Dictionary<Type, bool>();

		private static Plan PlanFor(Type type, object sample)
		{
			lock (plans)
			{
				if (plans.TryGetValue(type, out var plan)) return plan;
				plan = ShouldCopy(type) ? BuildPlan(type, sample) : null;
				plans[type] = plan;
				return plan;
			}
		}

		// Reference paths: chains of fields from the object to each slot that
		// stores a reference (descending through plain structs). Fields whose
		// path crosses a Nullable (unprobeable, boxed access returns the value)
		// land in nullableTops as their top-level field. Returns false when the
		// type has no reflectable fields at all.
		private static bool CollectRefPaths(Type type, List<FieldInfo[]> refPaths, List<FieldInfo> nullableTops)
		{
			bool sawAnyField = false;
			void Walk(Type t, List<FieldInfo> prefix)
			{
				foreach (var field in AllInstanceFields(t))
				{
					sawAnyField = true;
					var ft = field.FieldType;
					if (ft.IsPointer || ft == typeof(IntPtr) || ft == typeof(UIntPtr)) continue;
					if (!ft.IsValueType)
					{
						var path = new FieldInfo[prefix.Count + 1];
						prefix.CopyTo(path);
						path[prefix.Count] = field;
						refPaths.Add(path);
					}
					else if (Nullable.GetUnderlyingType(ft) is Type underlying)
					{
						if (IsRefBearing(underlying))
						{
							var top = prefix.Count == 0 ? field : prefix[0];
							if (!nullableTops.Contains(top)) nullableTops.Add(top);
						}
					}
					else if (IsRefBearing(ft))
					{
						prefix.Add(field);
						Walk(ft, prefix);
						prefix.RemoveAt(prefix.Count - 1);
					}
				}
			}
			Walk(type, new List<FieldInfo>());
			return sawAnyField;
		}

		private static Plan BuildPlan(Type type, object sample)
		{
			var refPaths = new List<FieldInfo[]>();
			var slowFields = new List<FieldInfo>();
			bool sawAnyField = CollectRefPaths(type, refPaths, slowFields);
			// A whitelisted CoreLib collection whose fields the AOT trimmer
			// dropped would silently copy nothing and alias its internals:
			// fail loudly instead (caller falls back to full resets).
			if (IsWhitelistedCollection(type))
			{
				if (!sawAnyField)
					throw new NotSupportedException("WorldCopier: no field metadata for " + type + " (add an rd.xml entry)");
				lock (PlannedCollectionNames) PlannedCollectionNames.Add(type.AssemblyQualifiedName);
			}
			if (refPaths.Count == 0)
				return new Plan { RefSlots = Array.Empty<int>(), SlowFields = slowFields.Count == 0 ? null : slowFields.ToArray() };

			// Locate each path's slot: clone the sample, plant a sentinel via
			// reflection, and find which pointer-sized slot changed. Reads are
			// IntPtr loads inside the probe's own allocation (GC-safe).
			long before = GC.GetAllocatedBytesForCurrentThread();
			object sizeProbe = ShallowClone(sample);
			long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
			GC.KeepAlive(sizeProbe);
			int dataBytes = checked((int)allocated) - 2 * IntPtr.Size; // header + method table

			// Probe every path first, then resolve per TOP-LEVEL FIELD: a top is
			// handled either entirely by offsets or entirely by reflection. Mixing
			// the two on one field processed the same slot twice - the reflection
			// pass then re-copied already-remapped values (copies of copies,
			// stitching the half-built clone into itself) while the failed path's
			// real slot kept its source reference, leaking the live world into
			// the snapshot graph.
			var probed = new List<(FieldInfo top, FieldInfo[] path, int offset)>(refPaths.Count);
			foreach (var path in refPaths)
			{
				int offset;
				// Fresh probe per path: probing consumes the sample's values.
				if (slowObjectFields) offset = -1;
				else try { offset = ProbePath(ShallowClone(sample), dataBytes, path); }
				catch { offset = -1; }
				probed.Add((path[0], path, offset));
			}
			var demotedTops = new HashSet<FieldInfo>();
			void Demote(FieldInfo top, FieldInfo[] path, string why)
			{
				if (demotedTops.Add(top))
					lock (AnomalyFields)
					{
						var tag = type.Name + "." + string.Join(".", Array.ConvertAll(path, f => f.Name)) + ":" + why;
						AnomalyFields.TryGetValue(tag, out int c); AnomalyFields[tag] = c + 1;
					}
			}
			foreach (var (top, path, offset) in probed)
				if (offset < 0) Demote(top, path, "lost");
			// A duplicate offset across tops means at least one probe latched onto
			// the wrong slot; neither top can be trusted.
			var offsetOwners = new Dictionary<int, (FieldInfo top, FieldInfo[] path)>();
			foreach (var (top, path, offset) in probed)
			{
				if (offset < 0 || demotedTops.Contains(top)) continue;
				if (offsetOwners.TryGetValue(offset, out var other))
				{
					if (!ReferenceEquals(other.top, top)) Demote(other.top, other.path, "dup");
					Demote(top, path, "dup");
					continue;
				}
				offsetOwners[offset] = (top, path);
			}
			var offsets = new List<int>(refPaths.Count);
			var seen = new HashSet<int>();
			var slowRefFields = new List<FieldInfo>();
			foreach (var (top, path, offset) in probed)
			{
				if (demotedTops.Contains(top)) continue;
				if (seen.Add(offset)) offsets.Add(offset); // same-top dup: slot listed once
			}
			foreach (var top in demotedTops)
			{
				if (!top.FieldType.IsValueType) { if (!slowRefFields.Contains(top)) slowRefFields.Add(top); }
				else if (!slowFields.Contains(top)) slowFields.Add(top);
			}
			return new Plan
			{
				RefSlots = offsets.ToArray(),
				SlowFields = slowFields.Count == 0 ? null : slowFields.ToArray(),
				SlowRefFields = slowRefFields.Count == 0 ? null : slowRefFields.ToArray(),
			};
		}

		// Find the byte offset (from data start) of the reference slot behind
		// `path` on the probe object; -1 when it cannot be determined.
		//
		// Writes are plain reflection SetValue (always type-correct): when the
		// probe's current value is non-null it is swapped for null; when it is
		// null, a synthetic non-null value of the field's declared type is
		// planted instead. Exactly one pointer-sized slot changes.
		private static int ProbePath(object probe, int dataBytes, FieldInfo[] path)
		{
			var leaf = path[path.Length - 1];
			object current = GetPath(probe, path);
			bool nullToValue = current == null;
			object planted = nullToValue ? MakeSentinel(leaf.FieldType) : null;
			if (nullToValue && planted == null) return -1; // nothing safe to write
			int slots = dataBytes / IntPtr.Size;
			var beforeWords = new IntPtr[slots];
			ref byte data = ref Unsafe.As<RawData>(probe).Data;
			for (int i = 0; i < slots; i++)
				beforeWords[i] = Unsafe.ReadUnaligned<IntPtr>(ref Unsafe.AddByteOffset(ref data, i * IntPtr.Size));
			SetPath(probe, path, 0, planted);
			int found = -1;
			for (int i = 0; i < slots; i++)
			{
				var now = Unsafe.ReadUnaligned<IntPtr>(ref Unsafe.AddByteOffset(ref data, i * IntPtr.Size));
				if (now == beforeWords[i]) continue;
				// Filter GC-movement noise: our write goes to/from null.
				if (nullToValue ? beforeWords[i] != IntPtr.Zero : now != IntPtr.Zero) continue;
				if (found >= 0) return -1; // more than one candidate: bail out
				found = i * IntPtr.Size;
			}
			GC.KeepAlive(probe);
			return found;
		}

		// A non-null value assignable to a field of declared type `type`, or
		// null when one cannot be manufactured (abstract classes, interfaces).
		private static object MakeSentinel(Type type)
		{
			if (type == typeof(object)) return new object();
			if (type == typeof(string)) return "";
			if (!type.IsClass || type.IsAbstract || type.IsArray || typeof(Delegate).IsAssignableFrom(type)) return null;
			try { return RuntimeHelpers.GetUninitializedObject(type); }
			catch { return null; }
		}

		private static object GetPath(object target, FieldInfo[] path)
		{
			for (int i = 0; i < path.Length; i++)
			{
				target = path[i].GetValue(target);
				if (target == null) return null;
			}
			return target;
		}

		// ---- struct array element plans ----------------------------------------

		private sealed class ElementPlan
		{
			public int Stride;       // element size in bytes
			// Reference slots relative to element start; null = "zero-skip"
			// mode: elements whose words are all zero are skipped, the rest go
			// through boxed reflection (sound for sparse pools).
			public int[] RelOffsets;
		}

		private static readonly Dictionary<Type, ElementPlan> elementPlans = new Dictionary<Type, ElementPlan>();

		// Slot layout of a ref-bearing struct used as an array element, probed
		// on a scratch clone of a real array: null-swap a path's leaf on two
		// different elements and diff the payload words - the absolute offsets
		// give the stride EXACTLY (allocation-counter sizes are only an upper
		// bound and misled an earlier version into corrupting the heap). Null
		// when the type cannot be probed: those arrays copy by reflection.
		private static ElementPlan ElementPlanFor(Type elementType, Array sample, long measuredPayload)
		{
			lock (elementPlans)
			{
				if (elementPlans.TryGetValue(elementType, out var plan)) return plan;
				// Building needs two probeable elements; retry on later arrays
				// rather than caching failure when this one is too small.
				plan = BuildElementPlan(elementType, sample, measuredPayload);
				if (plan != null || sample.Length >= 2) elementPlans[elementType] = plan;
				return plan;
			}
		}

		private static ElementPlan BuildElementPlan(Type elementType, Array sample, long measuredPayload)
		{
			var refPaths = new List<FieldInfo[]>();
			var nullableTops = new List<FieldInfo>();
			CollectRefPaths(elementType, refPaths, nullableTops);
			if (nullableTops.Count > 0 || refPaths.Count == 0 || sample.Length < 2) return null;
			var probe = (Array)sample.Clone();
			int scanBytes = checked((int)Math.Min(measuredPayload, 1 << 20));
			int probeLimit = Math.Min(probe.Length, 128);

			// Stride: flip a primitive field (always settable - no null games)
			// on elements 0 and 1; the two changed words are one stride apart.
			var primitive = FindPrimitivePath(elementType);
			if (primitive == null) return null;
			int abs0 = FlipPrimitive(probe, scanBytes, primitive, 0);
			int abs1 = FlipPrimitive(probe, scanBytes, primitive, 1);
			if (abs0 < 0 || abs1 <= abs0) return null;
			int stride = abs1 - abs0;
			if (stride % IntPtr.Size != 0 || (long)stride * probe.Length > measuredPayload) return null;

			// Reference slots: null-swap where a non-null value exists in the
			// first elements, else plant a synthetic sentinel on a boxed
			// element (payload = exactly one stride). Unresolved paths demote
			// the plan to zero-skip reflection.
			var offsets = new List<int>(refPaths.Count);
			var seen = new HashSet<int>();
			bool complete = true;
			foreach (var path in refPaths)
			{
				int rel = -1;
				for (int i = 0; i < probeLimit && rel < 0; i++)
				{
					object box = probe.GetValue(i);
					if (GetPath(box, path) == null) continue;
					try
					{
						SetPath(box, path, 0, null);
						int abs = DiffPayload(probe, scanBytes, () => probe.SetValue(box, i), nullToValue: false);
						if (abs >= 0) rel = abs - i * stride;
					}
					catch { }
				}
				if (rel < 0)
				{
					// All-null in the probed range: try a synthetic value on a box.
					try { rel = ProbePath(probe.GetValue(0), stride, path); }
					catch { rel = -1; }
				}
				if (rel < 0 || rel >= stride || rel % IntPtr.Size != 0 || !seen.Add(rel))
				{
					lock (AnomalyFields)
					{
						var tag = elementType.Name + "[]." + string.Join(".", Array.ConvertAll(path, f => f.Name)) + ":lost";
						AnomalyFields.TryGetValue(tag, out int c); AnomalyFields[tag] = c + 1;
					}
					complete = false;
					break;
				}
				offsets.Add(rel);
			}
			if (Environment.GetEnvironmentVariable("BOPL_COPY_TRACE") != null)
			{
				string name = elementType.DeclaringType != null && elementType.IsGenericType
					? elementType.DeclaringType.Name + "<" + string.Join(",", Array.ConvertAll(elementType.GetGenericArguments(), a => a.Name)) + ">." + elementType.Name
					: elementType.Name;
				Console.Error.WriteLine($"[plan] {name}[] stride={stride} len={sample.Length} "
					+ (complete ? $"offsets=[{string.Join(",", offsets)}]" : "zero-skip") + $" paths={refPaths.Count}");
			}
			return new ElementPlan { Stride = stride, RelOffsets = complete ? offsets.ToArray() : null };
		}

		// First chain of fields leading to a primitive/enum leaf, or null.
		private static FieldInfo[] FindPrimitivePath(Type type)
		{
			foreach (var field in AllInstanceFields(type))
			{
				var ft = field.FieldType;
				if (ft.IsPointer || ft == typeof(IntPtr) || ft == typeof(UIntPtr)) continue;
				if (ft.IsPrimitive || ft.IsEnum) return new[] { field };
				if (ft.IsValueType && Nullable.GetUnderlyingType(ft) == null)
				{
					var inner = FindPrimitivePath(ft);
					if (inner != null)
					{
						var path = new FieldInfo[inner.Length + 1];
						path[0] = field;
						inner.CopyTo(path, 1);
						return path;
					}
				}
			}
			return null;
		}

		// Flip a primitive leaf on element `index` and return the byte offset of
		// the word that changed; the value is restored afterwards. -1 on noise.
		private static int FlipPrimitive(Array probe, int scanBytes, FieldInfo[] path, int index)
		{
			object box = probe.GetValue(index);
			var declared = path[path.Length - 1].FieldType;
			var leafType = declared.IsEnum ? Enum.GetUnderlyingType(declared) : declared;
			object old = GetPath(box, path);
			object flipped;
			if (leafType == typeof(bool)) flipped = !(bool)old;
			else
			{
				var invariant = System.Globalization.CultureInfo.InvariantCulture;
				object one = Convert.ChangeType(1, leafType, invariant);
				flipped = one.Equals(Convert.ChangeType(old, leafType, invariant)) ? Convert.ChangeType(2, leafType, invariant) : one;
			}
			if (declared.IsEnum) flipped = Enum.ToObject(declared, flipped);
			SetPath(box, path, 0, flipped);
			int found = DiffPayload(probe, scanBytes, () => probe.SetValue(box, index), nullToValue: null);
			// Restore, so later probes see original values.
			SetPath(box, path, 0, old);
			probe.SetValue(box, index);
			return found;
		}

		// Run `mutate` and return the unique payload offset whose word changed.
		// nullToValue filters candidates: false = value->0, true = 0->value,
		// null = any change (primitive flips; GC does not move value words).
		private static int DiffPayload(Array probe, int scanBytes, Action mutate, bool? nullToValue)
		{
			int slots = scanBytes / IntPtr.Size;
			var beforeWords = new IntPtr[slots];
			ref byte payload = ref MemoryMarshal.GetArrayDataReference(probe);
			for (int i = 0; i < slots; i++)
				beforeWords[i] = Unsafe.ReadUnaligned<IntPtr>(ref Unsafe.AddByteOffset(ref payload, i * IntPtr.Size));
			mutate();
			int found = -1;
			for (int i = 0; i < slots; i++)
			{
				var now = Unsafe.ReadUnaligned<IntPtr>(ref Unsafe.AddByteOffset(ref payload, i * IntPtr.Size));
				if (now == beforeWords[i]) continue;
				// Filter GC-movement noise: reference writes go to/from null.
				if (nullToValue == false && now != IntPtr.Zero) continue;
				if (nullToValue == true && beforeWords[i] != IntPtr.Zero) continue;
				if (found >= 0) return -1;
				found = i * IntPtr.Size;
			}
			GC.KeepAlive(probe);
			return found;
		}

		// field-chain SetValue through nested structs (boxed round trips).
		private static void SetPath(object target, FieldInfo[] path, int index, object value)
		{
			if (index == path.Length - 1)
			{
				path[index].SetValue(target, value);
				return;
			}
			var box = path[index].GetValue(target);
			SetPath(box, path, index + 1, value);
			path[index].SetValue(target, box);
		}

		private static bool IsRefBearing(Type type)
		{
			if (!type.IsValueType || type.IsPrimitive || type.IsEnum || type.IsPointer) return false;
			if (type == typeof(IntPtr) || type == typeof(UIntPtr)) return false;
			var underlying = Nullable.GetUnderlyingType(type);
			if (underlying != null) return IsRefBearing(underlying);
			lock (refBearing)
			{
				if (refBearing.TryGetValue(type, out var known)) return known;
				refBearing[type] = false; // cycle guard (value types cannot truly cycle)
				bool result = false;
				foreach (var field in AllInstanceFields(type))
				{
					var ft = field.FieldType;
					if (ft.IsPointer || ft == typeof(IntPtr) || ft == typeof(UIntPtr)) continue;
					if (!ft.IsValueType || IsRefBearing(ft)) { result = true; break; }
				}
				refBearing[type] = result;
				return result;
			}
		}

		private static IEnumerable<FieldInfo> AllInstanceFields(Type type)
		{
			for (var t = type; t != null && t != typeof(object); t = t.BaseType)
				foreach (var field in t.GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly))
					yield return field;
		}

		private static bool ShouldCopy(Type type)
		{
			var assembly = type.Assembly.GetName().Name;
			return assembly == "UnityShim" || assembly == "Gameplay" || IsWhitelistedCollection(type);
		}

		private static bool IsWhitelistedCollection(Type type)
		{
			if (!type.IsGenericType) return false;
			var def = type.GetGenericTypeDefinition();
			if (def == typeof(List<>) || def == typeof(Dictionary<,>) || def == typeof(Queue<>)
				|| def == typeof(Stack<>) || def == typeof(HashSet<>) || def == typeof(LinkedList<>)
				|| def == typeof(LinkedListNode<>)) return true;
			// Nested helpers (Dictionary<,>.KeyCollection, +Entry...) hold a
			// reference back to their collection and must travel with it.
			var declaring = type.DeclaringType;
			return declaring != null && declaring.IsGenericTypeDefinition
				&& (declaring == typeof(Dictionary<,>) || declaring == typeof(HashSet<>) || declaring == typeof(List<>) || declaring == typeof(Queue<>) || declaring == typeof(Stack<>));
		}
	}
}
