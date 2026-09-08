using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityShim;

namespace BoplSim.Export
{
	// Loads native/reference-export/{maps.json,map-N.json,assets.json,manifest.json}
	// plus sim/Export/settings.json (layers/tags/collision matrix) and installs
	// scene/resource loading into the UnityShim World.
	public sealed class ExportDatabase
	{
		// Levels occupy build indices 6.. (first six scenes are menus, not exported).
		public const int LevelSceneOffset = 6;

		internal sealed class MapEntry
		{
			public int Level;
			public string SceneName;
			public string File;
		}

		private readonly string exportDir;
		private readonly string projectDir;   // Unity project root, for Assets/Resources text files

		internal readonly List<MapEntry> Maps = new List<MapEntry>();
		internal readonly Dictionary<string, JObject> AssetData = new Dictionary<string, JObject>();
		internal readonly Dictionary<string, string> GuidByResourcePath = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
		internal readonly Dictionary<string, UnityEngine.Object> BuiltAssets = new Dictionary<string, UnityEngine.Object>();
		internal readonly Dictionary<string, object> ObjectsById = new Dictionary<string, object>();
		internal readonly Dictionary<string, int> WarningCounts = new Dictionary<string, int>();
		internal readonly Dictionary<string, string> AssetNames = new Dictionary<string, string>();
		private readonly Dictionary<string, JArray> sceneJsonCache = new Dictionary<string, JArray>();
		private readonly Dictionary<string, Type> typeCache = new Dictionary<string, Type>();
		private readonly Dictionary<string, TextAsset> textAssets = new Dictionary<string, TextAsset>();
		private readonly HashSet<string> warnedScenes = new HashSet<string>();

		private static readonly Assembly[] SearchAssemblies =
		{
			typeof(Updater).Assembly,          // Gameplay
			typeof(GameObject).Assembly,       // UnityShim
		};

		public int LevelCount => Maps.Count;

		private ExportDatabase(string exportDir, string projectDir)
		{
			this.exportDir = exportDir;
			this.projectDir = projectDir;
		}

		public static ExportDatabase Load(string exportDir, string settingsPath = null, string projectDir = null)
		{
			var db = new ExportDatabase(exportDir, projectDir);
			foreach (var entry in JArray.Parse(File.ReadAllText(Path.Combine(exportDir, "maps.json"))))
				db.Maps.Add(new MapEntry
				{
					Level = (int)entry["level"],
					SceneName = (string)entry["scene"],
					File = (string)entry["file"],
				});

			var assets = JObject.Parse(File.ReadAllText(Path.Combine(exportDir, "assets.json")));
			foreach (var pair in assets)
			{
				var value = (JObject)pair.Value;
				db.AssetData[pair.Key] = value;
				string path = (string)value["path"];
				const string resourcesPrefix = "Assets/Resources/";
				if (path != null && path.StartsWith(resourcesPrefix, StringComparison.OrdinalIgnoreCase))
				{
					string resourcePath = path.Substring(resourcesPrefix.Length);
					int dot = resourcePath.LastIndexOf('.');
					if (dot >= 0) resourcePath = resourcePath.Substring(0, dot);
					db.GuidByResourcePath[resourcePath] = pair.Key;
				}
			}

			if (settingsPath != null && File.Exists(settingsPath)) db.LoadSettings(settingsPath);
			else Debug.LogError("Export: settings.json missing (" + settingsPath + ") - layers unnamed, all layers collide");

			// Script execution order (from the script .meta files) - decides
			// Awake/Start batch order, hence RegisterUpdatable/HierarchyNumber.
			string namesPath = settingsPath == null ? null
				: Path.Combine(Path.GetDirectoryName(settingsPath), "asset-names.json");
			if (namesPath != null && File.Exists(namesPath))
				foreach (var pair in JObject.Parse(File.ReadAllText(namesPath)))
					db.AssetNames[pair.Key] = (string)pair.Value;

			string orderPath = settingsPath == null ? null
				: Path.Combine(Path.GetDirectoryName(settingsPath), "execution-order.json");
			if (orderPath != null && File.Exists(orderPath))
			{
				var table = new Dictionary<string, int>();
				foreach (var pair in JObject.Parse(File.ReadAllText(orderPath)))
					table[pair.Key] = (int)pair.Value;
				ScriptExecutionOrder.Install(table);
			}
			else Debug.LogError("Export: execution-order.json missing - Awake order will be wrong");
			return db;
		}

		private void LoadSettings(string settingsPath)
		{
			var settings = JObject.Parse(File.ReadAllText(settingsPath));
			var world = World.Current;
			var layers = (JArray)settings["layers"];
			for (int i = 0; i < 32 && i < layers.Count; i++)
				world.layerNames[i] = (string)layers[i] ?? "";
			// Custom tags follow Unity's seven builtin tags (already in tagNames).
			foreach (var tag in (JArray)settings["tags"])
				if (!world.tagNames.Contains((string)tag)) world.tagNames.Add((string)tag);
			var matrix = (JArray)settings["collisionMatrix"];
			world.collisionMatrix = new uint[32];
			for (int i = 0; i < 32 && i < matrix.Count; i++)
				world.collisionMatrix[i] = (uint)matrix[i];
			world.collisionMatrixLoaded = true;
			// OrbitItems reads Time.fixedDeltaTime inside UpdateSim (SEMANTICS.md
			// §2.5): must be the exact TimeManager value (0.01666667f), not 1f/60f.
			if (settings["fixedDeltaTime"] != null) Time.fixedDeltaTime = (float)settings["fixedDeltaTime"];
			if (settings["timeScale"] != null) Time.timeScale = (float)settings["timeScale"];
		}

		// Wire this database into the current World.
		public void Install()
		{
			var world = World.Current;
			world.sceneCountInBuildSettings = LevelSceneOffset + Maps.Count;
			world.sceneLoader = LoadSceneByBuildIndex;
			world.resourceLoader = LoadResource;
			SceneManager.nameToBuildIndex = name =>
			{
				foreach (var map in Maps)
					if (map.SceneName == name) return map.Level + LevelSceneOffset;
				// Menu/UI scenes are not exported; loads become warned no-ops.
				return -1;
			};
		}

		// ---- scene loading ---------------------------------------------------

		private void LoadSceneByBuildIndex(int buildIndex, LoadSceneMode mode)
		{
			int level = buildIndex - LevelSceneOffset;
			if (level < 0 || level >= Maps.Count)
			{
				if (warnedScenes.Add(buildIndex.ToString()))
					Debug.LogError("Export: scene " + buildIndex + " is not exported (menu/UI scene) - LoadScene ignored");
				return;
			}
			var map = Maps[level];
			if (mode == LoadSceneMode.Single) World.Current.UnloadCurrentScene();

			// Set the active scene before building so every GameObject the load (or
			// anything after it) creates inherits it; then stamp the loaded subtrees
			// explicitly (children created before roots would otherwise carry the
			// previous scene).
			var scene = new Scene(buildIndex, map.SceneName);
			World.Current.activeScene = scene;

			// PERF-1 fix 1 (minimum): the parse dominated warm resets (~10 of 15 ms);
			// the JArray is only read by GraphBuilder/PropertyApplier, never mutated.
			if (!sceneJsonCache.TryGetValue(map.File, out var sceneJson))
				sceneJsonCache[map.File] = sceneJson = JArray.Parse(File.ReadAllText(Path.Combine(exportDir, map.File)));
			var pending = new List<PendingProperties>();
			var roots = new List<GameObject>();
			foreach (var node in sceneJson)
				roots.Add(GraphBuilder.BuildGameObject((JObject)node, false, null, pending, this));
			foreach (var item in pending)
				PropertyApplier.Apply(item.Target, item.Properties, this);

			foreach (var root in roots) StampScene(root, scene);
			World.Current.ActivateLoadedObjects(roots);
			SceneManager.NotifySceneLoaded(scene, mode);
			DumpWarnings("scene " + map.SceneName);
		}

		private static void StampScene(GameObject go, Scene scene)
		{
			go.scene = scene;
			var t = go.transform;
			for (int i = 0; i < t.childCount; i++)
				StampScene(t.GetChild(i).gameObject, scene);
		}

		// ---- assets ------------------------------------------------------------

		// Build (once) the prefab template / ScriptableObject stored under guid.
		internal UnityEngine.Object BuildAsset(string guid)
		{
			if (BuiltAssets.TryGetValue(guid, out var built)) return built;
			if (!AssetData.TryGetValue(guid, out var entry)) return null;
			var data = (JObject)entry["data"];
			string path = (string)entry["path"];
			if (data["components"] != null)
			{
				var pending = new List<PendingProperties>();
				var root = GraphBuilder.BuildGameObject(data, true, null, pending, this);
				BuiltAssets[guid] = root; // registered before apply so asset cycles resolve
				foreach (var item in pending)
					PropertyApplier.Apply(item.Target, item.Properties, this);
				return root;
			}
			// ScriptableObject (or other single-object asset).
			string typeName = (string)data["type"];
			var type = ResolveType(typeName);
			if (type == null)
			{
				Warn("asset type missing: " + typeName);
				return null;
			}
			var obj = (UnityEngine.Object)Activator.CreateInstance(type, true);
			obj.isAsset = true;
			obj.name = Path.GetFileNameWithoutExtension(path);
			string id = (string)data["id"];
			if (id != null) ObjectsById[id] = obj;
			BuiltAssets[guid] = obj;
			if (data["properties"] is JObject props) PropertyApplier.Apply(obj, props, this);
			// Unity fires OnEnable after deserializing a ScriptableObject asset.
			if (obj is ScriptableObject so) ScriptableObject.DispatchOnEnable(so);
			return obj;
		}

		// Host bootstrap: load an asset by its Unity project path.
		public UnityEngine.Object LoadAssetByPath(string assetPath)
		{
			foreach (var pair in AssetData)
				if ((string)pair.Value["path"] == assetPath)
					return BuildAsset(pair.Key);
			Warn("no asset at path " + assetPath);
			return null;
		}

		// Resolve a GlobalObjectId_V1-<type>-<guid>-<localId>-<prefabId> string.
		// expectedType (the serialized field's type) picks placeholder shapes for
		// assets the export does not carry (sprites, materials, audio...).
		internal object ResolveReference(string id, Type expectedType)
		{
			if (ObjectsById.TryGetValue(id, out var found)) return found;
			var parts = id.Split('-');
			if (parts.Length >= 3 && AssetData.ContainsKey(parts[2]))
			{
				BuildAsset(parts[2]);
				if (ObjectsById.TryGetValue(id, out found)) return found;
				// Reference to the asset root itself uses the root id already registered
				// during build; anything else inside the asset was registered too. A miss
				// here means the exporter saw a sub-object we did not create (skipped type).
				Warn("unresolved id inside asset " + parts[2]);
				return null;
			}
			// Textures, sprites, materials, audio, animator controllers: not exported.
			// Asset IDENTITY can still be gameplay (abilityIcons.IndexOf(sprite) selects
			// abilities - SEMANTICS.md §4): hand out one stable placeholder per id.
			var placeholder = CreatePlaceholder(id, expectedType);
			if (placeholder != null)
			{
				ObjectsById[id] = placeholder;
				return placeholder;
			}
			Warn("unresolved reference guid " + (parts.Length >= 3 ? parts[2] : id));
			return null;
		}

		private UnityEngine.Object CreatePlaceholder(string id, Type expectedType)
		{
			if (expectedType == null) return null;
			var type = expectedType == typeof(Texture) ? typeof(Texture2D) : expectedType;
			if (type.IsAbstract || !typeof(UnityEngine.Object).IsAssignableFrom(type)) return null;
			if (typeof(Component).IsAssignableFrom(type) || type == typeof(GameObject)) return null;
			string ns = type.Namespace ?? "";
			if (!ns.StartsWith("UnityEngine") && !ns.StartsWith("TMPro")) return null; // gameplay assets are exported; a miss is real
			var placeholder = (UnityEngine.Object)Activator.CreateInstance(type, true);
			placeholder.isAsset = true;
			// ReferenceCapture.Identity prints asset names ("UnityEngine.Sprite:
			// AbilityIconsSheet_3"); the rip stores one asset file per guid, so the
			// real Unity name is the ripped file name (sim/Export/asset-names.json).
			var parts = id.Split('-');
			placeholder.name = parts.Length >= 3 && AssetNames.TryGetValue(parts[2], out var known)
				? known
				: "external:" + id.Substring("GlobalObjectId_V1-".Length);
			return placeholder;
		}

		// ---- Resources.Load ----------------------------------------------------

		private UnityEngine.Object LoadResource(string path, Type type)
		{
			if (GuidByResourcePath.TryGetValue(path, out var guid))
				return BuildAsset(guid);
			if (type == typeof(TextAsset) || type == typeof(UnityEngine.Object))
			{
				if (textAssets.TryGetValue(path, out var cached)) return cached;
				if (projectDir != null)
				{
					string file = Path.Combine(projectDir, "Assets", "Resources", path + ".txt");
					if (File.Exists(file))
					{
						var asset = new TextAsset(File.ReadAllText(file)) { name = path };
						textAssets[path] = asset;
						return asset;
					}
				}
			}
			Warn("Resources.Load miss: " + path);
			return null;
		}

		// ---- type resolution / diagnostics --------------------------------------

		internal Type ResolveType(string fullName)
		{
			if (typeCache.TryGetValue(fullName, out var cached)) return cached;
			Type resolved = null;
			foreach (var assembly in SearchAssemblies)
			{
				resolved = assembly.GetType(fullName);
				if (resolved != null) break;
			}
			typeCache[fullName] = resolved;
			return resolved;
		}

		internal void Warn(string key)
		{
			WarningCounts.TryGetValue(key, out int count);
			WarningCounts[key] = count + 1;
		}

		internal void DumpWarnings(string context)
		{
			if (WarningCounts.Count == 0) return;
			var lines = new List<string>();
			foreach (var pair in WarningCounts) lines.Add(pair.Value + "x " + pair.Key);
			lines.Sort();
			Debug.Log("Export[" + context + "] " + WarningCounts.Count + " warning kinds:\n  " + string.Join("\n  ", lines));
			WarningCounts.Clear();
		}
	}

	internal struct PendingProperties
	{
		public object Target;
		public JObject Properties;
	}
}
