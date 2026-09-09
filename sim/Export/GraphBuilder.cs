using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace BoplSim.Export
{
	// Pass 1 of scene/prefab loading: create the GameObject hierarchy and all
	// components (no lifecycle callbacks), registering every exported id so
	// pass 2 can resolve object references. Component and child order follow
	// the JSON exactly (which mirrors the Unity file order).
	internal static class GraphBuilder
	{
		public static GameObject BuildGameObject(JObject node, bool asset, Transform parent,
			List<PendingProperties> pending, ExportDatabase db)
		{
			var go = GameObject.CreateDetached((string)node["name"], asset);
			go.layer = (int?)node["layer"] ?? 0;
			go.tag = (string)node["tag"] ?? "Untagged";
			go.activeSelfField = (bool?)node["active"] ?? true;
			// Scene-file objects have positive Unity instance ids ("persistent");
			// runtime Instantiate clones are negative. FindObjectsOfType order
			// (InstanceID ascending) depends on the distinction. Prefab templates
			// (asset=true) never enter the scene, so their flag is irrelevant,
			// and their clones are runtime objects with fresh ids.
			go.persistent = !asset;
			go.transform.persistent = !asset;
			if (parent != null) go.transform.SetParentInternal(parent);

			string goId = (string)node["id"];
			if (goId != null) db.Register(goId, go, asset);

			if (node["components"] is JArray components)
			{
				foreach (var token in components)
				{
					var comp = (JObject)token;
					string typeName = (string)comp["type"];
					string id = (string)comp["id"];
					object target;
					if (typeName == "UnityEngine.Transform")
					{
						target = go.transform;
					}
					else if (typeName == "UnityEngine.RectTransform")
					{
						go.UseRectTransform();
						target = go.transform;
					}
					else
					{
						var type = db.ResolveType(typeName);
						if (type == null)
						{
							db.Warn("component type missing: " + typeName);
							continue;
						}
						target = go.AddComponentSilent(type);
					}
					if (!asset && target is Object component) component.persistent = true;
					if (id != null) db.Register(id, target, asset);
					if (comp["properties"] is JObject props)
						pending.Add(new PendingProperties { Target = target, Properties = props });
				}
			}

			if (node["children"] is JArray children)
				foreach (var child in children)
					BuildGameObject((JObject)child, asset, go.transform, pending, db);

			return go;
		}
	}
}
