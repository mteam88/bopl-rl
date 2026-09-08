using System;
using System.Collections.Generic;
using UnityShim;

namespace UnityEngine.SceneManagement
{
	public struct Scene
	{
		internal int buildIndexField;
		internal string nameField;
		internal bool valid;

		public Scene(int buildIndex, string name)
		{
			buildIndexField = buildIndex;
			nameField = name;
			valid = true;
		}

		public int buildIndex => buildIndexField;
		public string name { get => nameField; set => nameField = value; }
		public bool isLoaded => valid;
		public bool IsValid() => valid;
		public string path => nameField;
		public int rootCount => World.Current.roots.Count;

		public GameObject[] GetRootGameObjects()
		{
			var list = new List<GameObject>();
			foreach (var root in World.Current.roots)
				if (root != null && !root.destroyed) list.Add(root);
			return list.ToArray();
		}

		public static bool operator ==(Scene a, Scene b) => a.buildIndexField == b.buildIndexField && a.valid == b.valid;
		public static bool operator !=(Scene a, Scene b) => !(a == b);
		public override bool Equals(object other) => other is Scene s && this == s;
		public override int GetHashCode() => buildIndexField;
	}

	public enum LoadSceneMode { Single, Additive }

	public class AsyncOperation : YieldInstruction
	{
		public bool isDone { get; set; } = true;
		public float progress { get; set; } = 1f;
		public bool allowSceneActivation { get; set; } = true;
		public event Action<AsyncOperation> completed { add { value?.Invoke(this); } remove { } }
	}

	public static class SceneManager
	{
		// UnityAction<Scene, LoadSceneMode> in Unity; Action works for gameplay's lambda subscribers.
		public static event Action<Scene, LoadSceneMode> sceneLoaded;
		public static event Action<Scene> sceneUnloaded;
		public static event Action<Scene, Scene> activeSceneChanged;

		public static int sceneCountInBuildSettings => World.Current.sceneCountInBuildSettings;
		public static int sceneCount => 1;

		public static Scene GetActiveScene() => World.Current.activeScene;

		public static void LoadScene(int sceneBuildIndex) => LoadScene(sceneBuildIndex, LoadSceneMode.Single);

		public static void LoadScene(int sceneBuildIndex, LoadSceneMode mode)
		{
			var loader = World.Current.sceneLoader;
			if (loader == null) throw new InvalidOperationException("UnityShim: no scene loader installed (Export not initialised)");
			loader(sceneBuildIndex, mode);
		}

		public static void LoadScene(string sceneName) => LoadScene(sceneName, LoadSceneMode.Single);

		public static void LoadScene(string sceneName, LoadSceneMode mode)
			=> LoadScene(BuildIndexByName(sceneName), mode);

		public static AsyncOperation LoadSceneAsync(int sceneBuildIndex) => LoadSceneAsync(sceneBuildIndex, LoadSceneMode.Single);

		public static AsyncOperation LoadSceneAsync(int sceneBuildIndex, LoadSceneMode mode)
		{
			LoadScene(sceneBuildIndex, mode);
			return new AsyncOperation();
		}

		public static AsyncOperation LoadSceneAsync(string sceneName) => LoadSceneAsync(BuildIndexByName(sceneName), LoadSceneMode.Single);
		public static AsyncOperation LoadSceneAsync(string sceneName, LoadSceneMode mode) => LoadSceneAsync(BuildIndexByName(sceneName), mode);

		public static AsyncOperation UnloadSceneAsync(Scene scene)
		{
			sceneUnloaded?.Invoke(scene);
			return new AsyncOperation();
		}

		public static AsyncOperation UnloadSceneAsync(int sceneBuildIndex) => new AsyncOperation();
		public static AsyncOperation UnloadSceneAsync(string sceneName) => new AsyncOperation();

		public static Scene GetSceneByName(string name) => World.Current.activeScene.name == name ? World.Current.activeScene : default;
		public static Scene GetSceneByBuildIndex(int buildIndex) => World.Current.activeScene.buildIndex == buildIndex ? World.Current.activeScene : default;
		public static Scene GetSceneAt(int index) => World.Current.activeScene;
		public static void SetActiveScene(Scene scene) { }
		public static void MoveGameObjectToScene(GameObject go, Scene scene) { }

		// Scene-name lookup installed by the Export loader.
		public static Func<string, int> nameToBuildIndex;

		private static int BuildIndexByName(string name)
		{
			if (nameToBuildIndex == null) throw new InvalidOperationException("UnityShim: scene name lookup not installed");
			return nameToBuildIndex(name);
		}

		// Export loader calls this once a scene finished instantiating.
		public static void NotifySceneLoaded(Scene scene, LoadSceneMode mode)
		{
			var previous = World.Current.activeScene;
			if (mode == LoadSceneMode.Single) World.Current.activeScene = scene;
			sceneLoaded?.Invoke(scene, mode);
			if (mode == LoadSceneMode.Single) activeSceneChanged?.Invoke(previous, scene);
		}
	}
}
