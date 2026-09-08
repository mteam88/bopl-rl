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
		// Non-level scenes from scenes.json (v3 export): boot MainMenu (0) and
		// WinnerTriangle (4). Same file shape as map-N.json.
		internal readonly Dictionary<int, MapEntry> ExtraScenes = new Dictionary<int, MapEntry>();
		// assets.json / sprites.json stay as bytes; entries are parsed when built.
		private JsonSliceIndex assets;
		private JsonSliceIndex sprites;
		internal readonly Dictionary<string, string> GuidByResourcePath = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
		internal readonly Dictionary<string, UnityEngine.Object> BuiltAssets = new Dictionary<string, UnityEngine.Object>();
		internal readonly Dictionary<string, object> ObjectsById = new Dictionary<string, object>();
		internal readonly Dictionary<string, int> WarningCounts = new Dictionary<string, int>();
		internal readonly Dictionary<string, string> AssetNames = new Dictionary<string, string>();
		private readonly Dictionary<string, Texture2D> spriteTextures = new Dictionary<string, Texture2D>();
		// Level scenes only: boot/WinnerTriangle load once, levels on every reset.
		// A level tree is ~10 MB; a training run rotates a handful of maps, a
		// map sweep (reset_check) touches dozens, so the cache is capped.
		private readonly Dictionary<string, JArray> sceneJsonCache = new Dictionary<string, JArray>();
		private const int SceneCacheCapacity = 8;
		private static readonly JsonLoadSettings NoLineInfo = new JsonLoadSettings { LineInfoHandling = LineInfoHandling.Ignore };
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

			string scenesPath = Path.Combine(exportDir, "scenes.json");
			if (File.Exists(scenesPath))
				foreach (var entry in JArray.Parse(File.ReadAllText(scenesPath)))
					db.ExtraScenes[(int)entry["buildIndex"]] = new MapEntry
					{
						Level = (int)entry["buildIndex"],
						SceneName = (string)entry["scene"],
						File = (string)entry["file"],
					};

			db.assets = JsonSliceIndex.Load(Path.Combine(exportDir, "assets.json"), "path");
			foreach (var pair in db.assets.Tracked)
			{
				string path = pair.Value;
				const string resourcesPrefix = "Assets/Resources/";
				if (path.StartsWith(resourcesPrefix, StringComparison.OrdinalIgnoreCase))
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

			// v5 export: per-sprite texture/geometry metrics keyed by GlobalObjectId
			// (ResizablePlatform reads sprite.texture.width + pixelsPerUnit on the
			// sim path). Atlas sprites share one Texture2D via textureId.
			string spritesPath = Path.Combine(exportDir, "sprites.json");
			if (File.Exists(spritesPath)) db.sprites = JsonSliceIndex.Load(spritesPath);

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
				// Unity's name lookup is case-insensitive - the game does
				// LoadScene("winnerTriangle") for WinnerTriangle.unity.
				foreach (var pair in ExtraScenes)
					if (string.Equals(pair.Value.SceneName, name, StringComparison.OrdinalIgnoreCase)) return pair.Key;
				foreach (var map in Maps)
					if (string.Equals(map.SceneName, name, StringComparison.OrdinalIgnoreCase)) return map.Level + LevelSceneOffset;
				// Menu/UI scenes not in the export; loads become warned no-ops.
				return -1;
			};
		}

		// ---- scene loading ---------------------------------------------------

		private void LoadSceneByBuildIndex(int buildIndex, LoadSceneMode mode)
		{
			MapEntry map;
			if (!ExtraScenes.TryGetValue(buildIndex, out map))
			{
				int level = buildIndex - LevelSceneOffset;
				if (level < 0 || level >= Maps.Count)
				{
					if (warnedScenes.Add(buildIndex.ToString()))
						Debug.LogError("Export: scene " + buildIndex + " is not exported (menu/UI scene) - LoadScene ignored");
					return;
				}
				map = Maps[level];
			}
			if (mode == LoadSceneMode.Single) World.Current.UnloadCurrentScene();

			// Single: set the active scene before building so every GameObject the
			// load (or anything after it) creates inherits it; then stamp the loaded
			// subtrees explicitly (children created before roots would otherwise
			// carry the previous scene). Additive: the active scene is untouched;
			// StampScene alone puts the loaded objects in the new scene.
			var scene = new Scene(buildIndex, map.SceneName);
			if (mode == LoadSceneMode.Single) World.Current.activeScene = scene;

			// PERF-1 fix 1 (minimum): the parse dominated warm resets (~10 of 15 ms);
			// the JArray is only read by GraphBuilder/PropertyApplier, never mutated.
			// Non-level scenes are not worth their tree (scene-boot.json alone is 5 MB
			// of text): parse and drop.
			bool isLevel = !ExtraScenes.ContainsKey(buildIndex);
			if (!isLevel || !sceneJsonCache.TryGetValue(map.File, out var sceneJson))
			{
				using (var text = new StreamReader(Path.Combine(exportDir, map.File), System.Text.Encoding.UTF8))
				using (var reader = new Newtonsoft.Json.JsonTextReader(text))
				{
					sceneJson = JArray.Load(reader, NoLineInfo);
					while (reader.Read()) { } // as JArray.Parse: trailing content throws
				}
				if (isLevel)
				{
					if (sceneJsonCache.Count >= SceneCacheCapacity) sceneJsonCache.Clear();
					sceneJsonCache[map.File] = sceneJson;
				}
			}
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
			var entry = assets.Parse(guid);
			if (entry == null) return null;
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
			foreach (var pair in assets.Tracked)
				if (pair.Value == assetPath)
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
			if (parts.Length >= 3 && assets.ContainsKey(parts[2]))
			{
				BuildAsset(parts[2]);
				if (ObjectsById.TryGetValue(id, out found)) return found;
				// Reference to the asset root itself uses the root id already registered
				// during build; anything else inside the asset was registered too. A miss
				// here means the exporter saw a sub-object we did not create (skipped type).
				Warn("unresolved id inside asset " + parts[2]);
				return null;
			}
			// Sprites: v5 exports real metrics (sprites.json) - the sim path reads
			// texture dims, pixelsPerUnit, pivot, bounds, textureRect
			// (ResizablePlatform, GrowthFlare, StretchableLightning...).
			var spriteJson = sprites?.Parse(id);
			if (spriteJson != null)
			{
				var sprite = BuildSprite(spriteJson);
				ObjectsById[id] = sprite;
				return sprite;
			}
			// Textures, materials, audio, animator controllers: not exported.
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

		// sprites.json floats arrive as exact {f32:bits}.
		private static float F32(JToken token)
		{
			if (token == null) return 0f;
			if (token is JObject o && o["f32"] != null)
				return BitConverter.Int32BitsToSingle((int)(long)o["f32"]);
			return (float)token;
		}

		private static Vector2 Vec2(JToken t) => t == null ? default : new Vector2(F32(t["x"]), F32(t["y"]));
		private static Vector3 Vec3(JToken t) => t == null ? default : new Vector3(F32(t["x"]), F32(t["y"]), F32(t["z"]));
		private static Rect RectOf(JToken t) => t == null ? default : new Rect(F32(t["x"]), F32(t["y"]), F32(t["w"]), F32(t["h"]));

		private Sprite BuildSprite(JObject data)
		{
			// Atlas sprites share the shim Texture2D their textureId names.
			string texId = (string)data["textureId"];
			Texture2D texture = null;
			if (texId == null || !spriteTextures.TryGetValue(texId, out texture))
			{
				string path = (string)data["path"] ?? "";
				texture = new Texture2D((int)data["textureWidth"], (int)data["textureHeight"])
				{
					name = Path.GetFileNameWithoutExtension(path),
					isAsset = true,
				};
				if (texId != null) spriteTextures[texId] = texture;
			}
			var bounds = new Bounds { center = Vec3(data["bounds"]?["center"]), extents = Vec3(data["bounds"]?["extents"]) };
			var border = data["border"] == null ? default
				: new Vector4(F32(data["border"]["x"]), F32(data["border"]["y"]), F32(data["border"]["z"]), F32(data["border"]["w"]));
			return new Sprite
			{
				name = (string)data["name"],
				isAsset = true,
				texture = texture,
				rect = RectOf(data["rect"]),
				textureRect = RectOf(data["textureRect"]),
				pivot = Vec2(data["pivot"]),
				pixelsPerUnit = F32(data["pixelsPerUnit"]),
				bounds = bounds,
				border = border,
			};
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

		public void DumpWarnings(string context)
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
