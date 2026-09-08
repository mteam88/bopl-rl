using System;

namespace UnityEngine
{
	public interface ISerializationCallbackReceiver
	{
		void OnBeforeSerialize();
		void OnAfterDeserialize();
	}

	[AttributeUsage(AttributeTargets.Class)]
	public class SelectionBaseAttribute : Attribute { }

	// Present in exported scenes; no behaviour.
	public class FlareLayer : Behaviour
	{
	}

	public class TextMesh : Component
	{
		public string text { get; set; } = "";
		public Color color { get; set; } = Color.white;
		public int fontSize { get; set; }
		public float characterSize { get; set; } = 1f;
		public TextAnchor anchor { get; set; }
		public TextAlignment alignment { get; set; }
		public Font font { get; set; }
		public float lineSpacing { get; set; } = 1f;
		public bool richText { get; set; } = true;
	}

	public enum TextAlignment { Left, Center, Right }

	public class SpriteMask : Behaviour
	{
		public Sprite sprite { get; set; }
		public bool isCustomRangeActive { get; set; }
		public int frontSortingOrder { get; set; }
		public int backSortingOrder { get; set; }
		public float alphaCutoff { get; set; } = 0.2f;
	}

	// UnityEngine.Cache (Caching API); declared but unused by gameplay.
	public struct Cache
	{
		public bool valid => false;
		public long spaceOccupied => 0;
		public string path => "";
	}

	// ---- IMGUI (PerfHud only; never rendered headless) ----------------------

	public class GUIStyle
	{
		public GUIStyleState normal { get; set; } = new GUIStyleState();
		public GUIStyleState hover { get; set; } = new GUIStyleState();
		public GUIStyleState active { get; set; } = new GUIStyleState();
		public int fontSize { get; set; }
		public FontStyle fontStyle { get; set; }
		public TextAnchor alignment { get; set; }
		public RectOffset padding { get; set; } = new RectOffset();
		public bool wordWrap { get; set; }
		public bool richText { get; set; }

		public GUIStyle() { }
		public GUIStyle(GUIStyle other)
		{
			if (other == null) return;
			normal = new GUIStyleState { textColor = other.normal.textColor };
			fontSize = other.fontSize;
			fontStyle = other.fontStyle;
			alignment = other.alignment;
		}
	}

	public class GUIStyleState
	{
		public Color textColor { get; set; } = Color.white;
		public Texture2D background { get; set; }
	}

	public class GUIContent
	{
		public string text { get; set; } = "";
		public static GUIContent none { get; } = new GUIContent();
		public GUIContent() { }
		public GUIContent(string text) { this.text = text; }
	}

	public class GUISkin : ScriptableObject
	{
		public GUIStyle label { get; set; } = new GUIStyle();
		public GUIStyle box { get; set; } = new GUIStyle();
		public GUIStyle button { get; set; } = new GUIStyle();
	}

	public static class GUI
	{
		public static GUISkin skin { get; set; } = new GUISkin();
		public static Color color { get; set; } = Color.white;
		public static Color backgroundColor { get; set; } = Color.white;
		public static void Box(Rect position, GUIContent content) { }
		public static void Box(Rect position, string text) { }
		public static void Label(Rect position, string text) { }
		public static void Label(Rect position, string text, GUIStyle style) { }
		public static bool Button(Rect position, string text) => false;
	}

	public static class GUILayout
	{
		public static void Label(string text) { }
		public static bool Button(string text) => false;
	}
}
