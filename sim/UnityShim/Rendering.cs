using System;
using System.Collections.Generic;

namespace UnityEngine
{
	// Presentation stubs: data-only, keep the properties gameplay reads/writes.

	public class Texture : Object
	{
		public int width { get; set; }
		public int height { get; set; }
		public FilterMode filterMode { get; set; }
		public TextureWrapMode wrapMode { get; set; }
	}

	public enum FilterMode { Point, Bilinear, Trilinear }
	public enum TextureWrapMode { Repeat, Clamp, Mirror, MirrorOnce }
	public enum TextureFormat { Alpha8 = 1, RGB24 = 3, RGBA32 = 4, ARGB32 = 5, RGB565 = 7 }

	public class Texture2D : Texture
	{
		public Texture2D(int width, int height) { this.width = width; this.height = height; }
		public Texture2D(int width, int height, TextureFormat format, bool mipChain) : this(width, height) { }
		public static Texture2D whiteTexture { get; } = new Texture2D(4, 4);
		public static Texture2D blackTexture { get; } = new Texture2D(4, 4);
		public Color GetPixel(int x, int y) => Color.white;
		public void SetPixel(int x, int y, Color color) { }
		public Color[] GetPixels() => Array.Empty<Color>();
		public Color[] GetPixels(int x, int y, int blockWidth, int blockHeight) => new Color[blockWidth * blockHeight];
		public void SetPixels(Color[] colors) { }
		public void SetPixels(int x, int y, int blockWidth, int blockHeight, Color[] colors) { }
		public void Apply() { }
		public void Apply(bool updateMipmaps) { }
		public void Apply(bool updateMipmaps, bool makeNoLongerReadable) { }
		public bool LoadImage(byte[] data) => true;
	}

	public class RenderTexture : Texture
	{
		public RenderTexture(int width, int height, int depth) { this.width = width; this.height = height; }
		public RenderTexture(int width, int height, int depth, RenderTextureFormat format) : this(width, height, depth) { }
		public static RenderTexture active { get; set; }
		public void Release() { }
		public bool Create() => true;
	}

	public enum RenderTextureFormat { ARGB32, Depth, ARGBHalf, RGB565, ARGB4444, ARGB1555, Default, DefaultHDR = 21 }

	public class Sprite : Object
	{
		// Placeholder sprites (assets not exported) still get poked by presentation
		// Awakes (PlatformSlimeTrailCamera reads sprite.texture.width); lazily
		// self-populate so those run without data.
		private Texture2D textureField;
		public Texture2D texture
		{
			get => textureField ??= new Texture2D(0, 0) { objectName = objectName + ":texture" };
			set => textureField = value;
		}
		public Rect rect { get; set; }
		public Rect textureRect { get; set; }
		public Vector2 pivot { get; set; }
		public float pixelsPerUnit { get; set; } = 100f;
		public Bounds bounds { get; set; }
		public Vector4 border { get; set; }

		public static Sprite Create(Texture2D texture, Rect rect, Vector2 pivot)
			=> new Sprite { texture = texture, rect = rect, pivot = pivot };
		public static Sprite Create(Texture2D texture, Rect rect, Vector2 pivot, float pixelsPerUnit)
			=> new Sprite { texture = texture, rect = rect, pivot = pivot, pixelsPerUnit = pixelsPerUnit };
	}

	public class Shader : Object
	{
		public static Shader Find(string name) => new Shader { objectName = name ?? "" };
		public static int PropertyToID(string name) => name?.GetHashCode() ?? 0;
		public static void SetGlobalFloat(string name, float value) { }
		public static void SetGlobalFloat(int nameID, float value) { }
		public static void SetGlobalTexture(string name, Texture value) { }
		public static void SetGlobalTexture(int nameID, Texture value) { }
		public static void SetGlobalVector(string name, Vector4 value) { }
		public static void SetGlobalVector(int nameID, Vector4 value) { }
		public static void SetGlobalColor(string name, Color value) { }
		public static void SetGlobalInt(string name, int value) { }
	}

	public class Material : Object
	{
		private readonly Dictionary<int, float> floats = new Dictionary<int, float>();
		private readonly Dictionary<int, Color> colors = new Dictionary<int, Color>();
		private readonly Dictionary<int, Vector4> vectors = new Dictionary<int, Vector4>();
		private readonly Dictionary<int, Texture> textures = new Dictionary<int, Texture>();

		public Shader shader { get; set; }
		public Color color { get; set; } = Color.white;
		public Texture mainTexture { get; set; }
		public Vector2 mainTextureOffset { get; set; }
		public Vector2 mainTextureScale { get; set; } = Vector2.one;
		public int renderQueue { get; set; }

		public Material() { }
		public Material(Shader shader) { this.shader = shader; }
		public Material(Material source)
		{
			if (source == null) return;
			shader = source.shader; color = source.color; mainTexture = source.mainTexture;
			foreach (var kv in source.floats) floats[kv.Key] = kv.Value;
			foreach (var kv in source.colors) colors[kv.Key] = kv.Value;
			foreach (var kv in source.vectors) vectors[kv.Key] = kv.Value;
			foreach (var kv in source.textures) textures[kv.Key] = kv.Value;
		}

		public float GetFloat(string name) => GetFloat(Shader.PropertyToID(name));
		public float GetFloat(int nameID) => floats.TryGetValue(nameID, out var v) ? v : 0f;
		public void SetFloat(string name, float value) => SetFloat(Shader.PropertyToID(name), value);
		public void SetFloat(int nameID, float value) => floats[nameID] = value;
		public int GetInt(string name) => (int)GetFloat(name);
		public void SetInt(string name, int value) => SetFloat(name, value);
		public Color GetColor(string name) => GetColor(Shader.PropertyToID(name));
		public Color GetColor(int nameID) => colors.TryGetValue(nameID, out var v) ? v : Color.white;
		public void SetColor(string name, Color value) => SetColor(Shader.PropertyToID(name), value);
		public void SetColor(int nameID, Color value) => colors[nameID] = value;
		public Vector4 GetVector(string name) => GetVector(Shader.PropertyToID(name));
		public Vector4 GetVector(int nameID) => vectors.TryGetValue(nameID, out var v) ? v : Vector4.zero;
		public void SetVector(string name, Vector4 value) => SetVector(Shader.PropertyToID(name), value);
		public void SetVector(int nameID, Vector4 value) => vectors[nameID] = value;
		public Texture GetTexture(string name) => textures.TryGetValue(Shader.PropertyToID(name), out var v) ? v : null;
		public void SetTexture(string name, Texture value) => textures[Shader.PropertyToID(name)] = value;
		public void SetTexture(int nameID, Texture value) => textures[nameID] = value;
		public bool HasProperty(string name) => true;
		public bool HasProperty(int nameID) => true;
		public void SetTextureOffset(string name, Vector2 offset) { }
		public void SetTextureScale(string name, Vector2 scale) { }
		public void EnableKeyword(string keyword) { }
		public void DisableKeyword(string keyword) { }
	}

	public class MaterialPropertyBlock
	{
		public void SetFloat(string name, float value) { }
		public void SetFloat(int nameID, float value) { }
		public void SetColor(string name, Color value) { }
		public void SetColor(int nameID, Color value) { }
		public void SetVector(string name, Vector4 value) { }
		public void SetTexture(string name, Texture value) { }
		public void Clear() { }
	}

	public class Renderer : Component
	{
		private Material materialField;
		private Material[] materialsField = Array.Empty<Material>();

		public bool enabled { get; set; } = true;
		public int sortingOrder { get; set; }
		public string sortingLayerName { get; set; } = "Default";
		public int sortingLayerID { get; set; }
		public Bounds bounds { get; set; }
		public bool isVisible => enabled;

		// Unity's .material clones the shared material on first access.
		public Material material
		{
			get => materialField ??= new Material(sharedMaterial);
			set => materialField = value;
		}

		public Material sharedMaterial { get; set; }

		public Material[] materials
		{
			get => materialsField;
			set => materialsField = value;
		}

		public Material[] sharedMaterials { get; set; } = Array.Empty<Material>();

		public void GetPropertyBlock(MaterialPropertyBlock properties) { }
		public void SetPropertyBlock(MaterialPropertyBlock properties) { }
	}

	public class SpriteRenderer : Renderer
	{
		public Sprite sprite { get; set; }
		public Color color { get; set; } = Color.white;
		public bool flipX { get; set; }
		public bool flipY { get; set; }
		public SpriteDrawMode drawMode { get; set; }
		public Vector2 size { get; set; }
		public SpriteMaskInteraction maskInteraction { get; set; }
	}

	public enum SpriteDrawMode { Simple, Sliced, Tiled }
	public enum SpriteMaskInteraction { None, VisibleInsideMask, VisibleOutsideMask }

	public class LineRenderer : Renderer
	{
		private Vector3[] positions = Array.Empty<Vector3>();
		public int positionCount
		{
			get => positions.Length;
			set { var next = new Vector3[value]; Array.Copy(positions, next, Math.Min(positions.Length, value)); positions = next; }
		}
		public float startWidth { get; set; }
		public float endWidth { get; set; }
		public float widthMultiplier { get; set; } = 1f;
		public Color startColor { get; set; } = Color.white;
		public Color endColor { get; set; } = Color.white;
		public bool useWorldSpace { get; set; } = true;
		public bool loop { get; set; }
		public AnimationCurve widthCurve { get; set; }
		public int numCapVertices { get; set; }
		public int numCornerVertices { get; set; }
		public LineTextureMode textureMode { get; set; }
		public LineAlignment alignment { get; set; }

		public void SetPosition(int index, Vector3 position) { if (index >= 0 && index < positions.Length) positions[index] = position; }
		public Vector3 GetPosition(int index) => index >= 0 && index < positions.Length ? positions[index] : Vector3.zero;
		public void SetPositions(Vector3[] newPositions) => positions = (Vector3[])newPositions.Clone();
		public int GetPositions(Vector3[] output) { Array.Copy(positions, output, Math.Min(positions.Length, output.Length)); return positions.Length; }
	}

	public enum LineTextureMode { Stretch, Tile, DistributePerSegment, RepeatPerSegment }
	public enum LineAlignment { View, TransformZ }

	public class TrailRenderer : Renderer
	{
		public float time { get; set; }
		public float startWidth { get; set; }
		public float endWidth { get; set; }
		public bool emitting { get; set; } = true;
		public void Clear() { }
	}

	public class MeshFilter : Component
	{
		public Mesh mesh { get; set; }
		public Mesh sharedMesh { get; set; }
	}

	public class MeshRenderer : Renderer { }

	public class Mesh : Object
	{
		public Vector3[] vertices { get; set; } = Array.Empty<Vector3>();
		public Vector2[] uv { get; set; } = Array.Empty<Vector2>();
		public int[] triangles { get; set; } = Array.Empty<int>();
		public Color[] colors { get; set; } = Array.Empty<Color>();
		public Vector3[] normals { get; set; } = Array.Empty<Vector3>();
		public Bounds bounds { get; set; }
		public void Clear() { vertices = Array.Empty<Vector3>(); uv = Array.Empty<Vector2>(); triangles = Array.Empty<int>(); }
		public void RecalculateBounds() { }
		public void RecalculateNormals() { }
		public void SetVertices(List<Vector3> inVertices) => vertices = inVertices.ToArray();
		public void SetVertices(Vector3[] inVertices) => vertices = (Vector3[])inVertices.Clone();
		public void SetVertices(Vector3[] inVertices, int start, int length) { var v = new Vector3[length]; Array.Copy(inVertices, start, v, 0, length); vertices = v; }
		public void SetVertices(Unity.Collections.NativeArray<Vector3> inVertices) => vertices = inVertices.ToArray();
		public void SetVertices(Unity.Collections.NativeArray<Vector3> inVertices, int start, int length)
		{ var v = new Vector3[length]; for (int i = 0; i < length; i++) v[i] = inVertices[start + i]; vertices = v; }
		public void SetUVs(int channel, List<Vector2> uvs) => uv = uvs.ToArray();
		public void SetUVs(int channel, Vector2[] uvs) => uv = (Vector2[])uvs.Clone();
		public void SetUVs(int channel, Vector2[] uvs, int start, int length) { var v = new Vector2[length]; Array.Copy(uvs, start, v, 0, length); uv = v; }
		public void SetUVs(int channel, Unity.Collections.NativeArray<Vector2> uvs) => uv = uvs.ToArray();
		public void SetUVs(int channel, Unity.Collections.NativeArray<Vector2> uvs, int start, int length)
		{ var v = new Vector2[length]; for (int i = 0; i < length; i++) v[i] = uvs[start + i]; uv = v; }
		public void SetTriangles(int[] tris, int submesh) => triangles = tris;
		public void SetTriangles(List<int> tris, int submesh) => triangles = tris.ToArray();
		public void SetTriangles(int[] tris, int trianglesStart, int trianglesLength, int submesh) { var t = new int[trianglesLength]; Array.Copy(tris, trianglesStart, t, 0, trianglesLength); triangles = t; }
		public void MarkDynamic() { }
	}

	public class Camera : Behaviour
	{
		public static Camera main
		{
			get
			{
				var go = GameObject.FindGameObjectWithTag("MainCamera");
				return go != null ? go.GetComponent<Camera>() : null;
			}
		}
		public static Camera current => main;
		public static Camera[] allCameras => FindObjectsOfType<Camera>();

		public float orthographicSize { get; set; } = 5f;
		public bool orthographic { get; set; } = true;
		public float fieldOfView { get; set; } = 60f;
		public float nearClipPlane { get; set; } = 0.3f;
		public float farClipPlane { get; set; } = 1000f;
		public Color backgroundColor { get; set; }
		public int cullingMask { get; set; } = -1;
		public float aspect { get; set; } = 16f / 9f;
		public Rect rect { get; set; } = new Rect(0f, 0f, 1f, 1f);
		public RenderTexture targetTexture { get; set; }
		public float depth { get; set; }
		public CameraClearFlags clearFlags { get; set; }
		public int pixelWidth => Screen.width;
		public int pixelHeight => Screen.height;
		public void Render() { }
		public void RenderWithShader(Shader shader, string replacementTag) { }

		public Vector3 WorldToScreenPoint(Vector3 position)
		{
			// orthographic mapping (presentation only)
			float halfH = orthographicSize;
			float halfW = halfH * aspect;
			Vector3 local = transform.InverseTransformPoint(position);
			return new Vector3(
				(local.x / halfW * 0.5f + 0.5f) * Screen.width,
				(local.y / halfH * 0.5f + 0.5f) * Screen.height,
				local.z);
		}

		public Vector3 ScreenToWorldPoint(Vector3 position)
		{
			float halfH = orthographicSize;
			float halfW = halfH * aspect;
			Vector3 local = new Vector3(
				(position.x / Screen.width - 0.5f) * 2f * halfW,
				(position.y / Screen.height - 0.5f) * 2f * halfH,
				position.z);
			return transform.TransformPoint(local);
		}

		public Vector3 ViewportToWorldPoint(Vector3 position)
		{
			float halfH = orthographicSize;
			float halfW = halfH * aspect;
			Vector3 local = new Vector3((position.x - 0.5f) * 2f * halfW, (position.y - 0.5f) * 2f * halfH, position.z);
			return transform.TransformPoint(local);
		}

		public Vector3 WorldToViewportPoint(Vector3 position)
		{
			float halfH = orthographicSize;
			float halfW = halfH * aspect;
			Vector3 local = transform.InverseTransformPoint(position);
			return new Vector3(local.x / halfW * 0.5f + 0.5f, local.y / halfH * 0.5f + 0.5f, local.z);
		}

		public Ray ScreenPointToRay(Vector3 pos) => new Ray(ScreenToWorldPoint(new Vector3(pos.x, pos.y, nearClipPlane)), transform.forward);
	}

	public enum CameraClearFlags { Skybox = 1, Color = 2, SolidColor = 2, Depth = 3, Nothing = 4 }

	[Serializable]
	public struct Ray
	{
		public Vector3 origin, direction;
		public Ray(Vector3 origin, Vector3 direction) { this.origin = origin; this.direction = direction.normalized; }
		public Vector3 GetPoint(float distance) => origin + direction * distance;
	}

	public class Animator : Behaviour
	{
		public RuntimeAnimatorController runtimeAnimatorController { get; set; }
		public float speed { get; set; } = 1f;
		public bool applyRootMotion { get; set; }
		public AnimatorUpdateMode updateMode { get; set; }

		public void Play(string stateName) { }
		public void Play(string stateName, int layer) { }
		public void Play(string stateName, int layer, float normalizedTime) { }
		public void SetTrigger(string name) { }
		public void ResetTrigger(string name) { }
		public void SetBool(string name, bool value) { }
		public bool GetBool(string name) => false;
		public void SetFloat(string name, float value) { }
		public float GetFloat(string name) => 0f;
		public void SetInteger(string name, int value) { }
		public int GetInteger(string name) => 0;
		public AnimatorStateInfo GetCurrentAnimatorStateInfo(int layerIndex) => default;
		public void Rebind() { }
		public void Update(float deltaTime) { }
	}

	public enum AnimatorUpdateMode { Normal, AnimatePhysics, UnscaledTime }

	public struct AnimatorStateInfo
	{
		public float normalizedTime => 0f;
		public float length => 0f;
		public bool IsName(string name) => false;
		public int fullPathHash => 0;
		public int shortNameHash => 0;
	}

	public class RuntimeAnimatorController : Object { }

	public class Animation : Behaviour
	{
		public AnimationClip clip { get; set; }
		public bool isPlaying => false;
		public bool Play() => true;
		public bool Play(string animation) => true;
		public void Stop() { }
	}

	public class AnimationClip : Object
	{
		public float length { get; set; }
		public WrapMode wrapMode { get; set; }
	}

	public class AudioSource : Behaviour
	{
		public AudioClip clip { get; set; }
		public float volume { get; set; } = 1f;
		public float pitch { get; set; } = 1f;
		public bool loop { get; set; }
		public bool playOnAwake { get; set; } = true;
		public bool mute { get; set; }
		public float spatialBlend { get; set; }
		public bool isPlaying { get; private set; }
		public float time { get; set; }
		public UnityEngine.Audio.AudioMixerGroup outputAudioMixerGroup { get; set; }

		public void Play() => isPlaying = true;
		public void Play(ulong delay) => isPlaying = true;
		public void PlayDelayed(float delay) => isPlaying = true;
		public void Stop() => isPlaying = false;
		public void Pause() => isPlaying = false;
		public void UnPause() => isPlaying = true;
		public void PlayOneShot(AudioClip clip) { }
		public void PlayOneShot(AudioClip clip, float volumeScale) { }
		public static void PlayClipAtPoint(AudioClip clip, Vector3 position) { }
		public static void PlayClipAtPoint(AudioClip clip, Vector3 position, float volume) { }
	}

	public class AudioClip : Object
	{
		public float length { get; set; }
		public int samples { get; set; }
		public int channels { get; set; }
		public int frequency { get; set; } = 44100;
	}

	public static class AudioListener
	{
		public static float volume { get; set; } = 1f;
		public static bool pause { get; set; }
	}

	[Serializable]
	public class Gradient
	{
		public GradientColorKey[] colorKeys = Array.Empty<GradientColorKey>();
		public GradientAlphaKey[] alphaKeys = Array.Empty<GradientAlphaKey>();
		public GradientMode mode;

		public Color Evaluate(float time)
		{
			if (colorKeys.Length == 0) return Color.white;
			Color color = colorKeys[colorKeys.Length - 1].color;
			for (int i = 0; i < colorKeys.Length; i++)
			{
				if (time <= colorKeys[i].time)
				{
					if (i == 0) { color = colorKeys[0].color; break; }
					float t = Mathf.InverseLerp(colorKeys[i - 1].time, colorKeys[i].time, time);
					color = Color.Lerp(colorKeys[i - 1].color, colorKeys[i].color, t);
					break;
				}
			}
			float alpha = alphaKeys.Length == 0 ? 1f : alphaKeys[alphaKeys.Length - 1].alpha;
			for (int i = 0; i < alphaKeys.Length; i++)
			{
				if (time <= alphaKeys[i].time)
				{
					if (i == 0) { alpha = alphaKeys[0].alpha; break; }
					float t = Mathf.InverseLerp(alphaKeys[i - 1].time, alphaKeys[i].time, time);
					alpha = Mathf.Lerp(alphaKeys[i - 1].alpha, alphaKeys[i].alpha, t);
					break;
				}
			}
			color.a = alpha;
			return color;
		}

		public void SetKeys(GradientColorKey[] colorKeys, GradientAlphaKey[] alphaKeys)
		{
			this.colorKeys = colorKeys;
			this.alphaKeys = alphaKeys;
		}
	}

	[Serializable]
	public struct GradientColorKey
	{
		public Color color;
		public float time;
		public GradientColorKey(Color color, float time) { this.color = color; this.time = time; }
	}

	[Serializable]
	public struct GradientAlphaKey
	{
		public float alpha;
		public float time;
		public GradientAlphaKey(float alpha, float time) { this.alpha = alpha; this.time = time; }
	}

	public enum GradientMode { Blend, Fixed }
}

namespace UnityEngine.Audio
{
	public class AudioMixer : Object
	{
		public bool SetFloat(string name, float value) => true;
		public bool GetFloat(string name, out float value) { value = 0f; return true; }
	}

	public class AudioMixerGroup : Object
	{
		public AudioMixer audioMixer { get; set; }
	}
}
