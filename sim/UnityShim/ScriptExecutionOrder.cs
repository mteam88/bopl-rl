using System;
using System.Collections.Generic;
using UnityEngine;

namespace UnityShim
{
	// Unity's Script Execution Order (ProjectSettings/MonoManager, ripped from
	// the script .meta files into sim/Export/execution-order.json). It orders
	// Awake/OnEnable/Start/Update batches; the game leans on it hard:
	// InputUpdater -90, SteamManager -85, FixTransform -80, DetPhysics -75,
	// Updater -60, Host -55, ... PlayerBody 2, ... SlimeController 269.
	// Awake order decides Updater.RegisterUpdatable order and therefore
	// HierarchyNumber and simulation order - gameplay-critical.
	public static class ScriptExecutionOrder
	{
		// Process-wide (the table is the same for every world); the cache is
		// filled lazily from several world threads, hence the lock.
		private static Dictionary<string, int> byTypeName = new Dictionary<string, int>();
		private static readonly Dictionary<Type, int> cache = new Dictionary<Type, int>();

		public static void Install(Dictionary<string, int> table)
		{
			lock (cache)
			{
				byTypeName = table ?? new Dictionary<string, int>();
				cache.Clear();
			}
		}

		public static int Of(Type type)
		{
			lock (cache)
			{
				if (cache.TryGetValue(type, out var cached)) return cached;
				// Unity keys execution order by script (class) name; inheritance does
				// not propagate the setting.
				int order = byTypeName.TryGetValue(type.Name, out var found) ? found : 0;
				cache[type] = order;
				return order;
			}
		}

		// Stable sort by execution order (List<T>.Sort is unstable; ties must
		// keep collection order - that IS Unity's secondary order).
		internal static void SortStable(List<MonoBehaviour> batch)
		{
			if (batch.Count < 2) return;
			var keyed = new (int order, int index, MonoBehaviour mb)[batch.Count];
			for (int i = 0; i < batch.Count; i++)
				keyed[i] = (Of(batch[i].GetType()), i, batch[i]);
			Array.Sort(keyed, (a, b) => a.order != b.order ? a.order.CompareTo(b.order) : a.index.CompareTo(b.index));
			for (int i = 0; i < batch.Count; i++) batch[i] = keyed[i].mb;
		}
	}
}
