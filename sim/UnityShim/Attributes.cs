using System;

namespace UnityEngine
{
	[AttributeUsage(AttributeTargets.Field)]
	public sealed class SerializeField : Attribute { }

	[AttributeUsage(AttributeTargets.Field | AttributeTargets.Class | AttributeTargets.Struct)]
	public sealed class SerializeReference : Attribute { }

	[AttributeUsage(AttributeTargets.Field | AttributeTargets.Property | AttributeTargets.Class | AttributeTargets.Struct)]
	public sealed class HideInInspector : Attribute { }

	[AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
	public sealed class RequireComponent : Attribute
	{
		public Type m_Type0, m_Type1, m_Type2;
		public RequireComponent(Type requiredComponent) { m_Type0 = requiredComponent; }
		public RequireComponent(Type requiredComponent, Type requiredComponent2) { m_Type0 = requiredComponent; m_Type1 = requiredComponent2; }
		public RequireComponent(Type requiredComponent, Type requiredComponent2, Type requiredComponent3) { m_Type0 = requiredComponent; m_Type1 = requiredComponent2; m_Type2 = requiredComponent3; }
	}

	[AttributeUsage(AttributeTargets.Field, AllowMultiple = true)]
	public class HeaderAttribute : PropertyAttribute
	{
		public string header;
		public HeaderAttribute(string header) { this.header = header; }
	}

	[AttributeUsage(AttributeTargets.Field)]
	public sealed class RangeAttribute : PropertyAttribute
	{
		public float min, max;
		public RangeAttribute(float min, float max) { this.min = min; this.max = max; }
	}

	[AttributeUsage(AttributeTargets.Field)]
	public class TooltipAttribute : PropertyAttribute
	{
		public string tooltip;
		public TooltipAttribute(string tooltip) { this.tooltip = tooltip; }
	}

	[AttributeUsage(AttributeTargets.Field, AllowMultiple = true)]
	public class SpaceAttribute : PropertyAttribute
	{
		public float height;
		public SpaceAttribute() { height = 8f; }
		public SpaceAttribute(float height) { this.height = height; }
	}

	[AttributeUsage(AttributeTargets.Field)]
	public class TextAreaAttribute : PropertyAttribute
	{
		public int minLines, maxLines;
		public TextAreaAttribute() { minLines = 3; maxLines = 3; }
		public TextAreaAttribute(int minLines, int maxLines) { this.minLines = minLines; this.maxLines = maxLines; }
	}

	[AttributeUsage(AttributeTargets.Field)]
	public class MultilineAttribute : PropertyAttribute
	{
		public int lines;
		public MultilineAttribute() { lines = 3; }
		public MultilineAttribute(int lines) { this.lines = lines; }
	}

	[AttributeUsage(AttributeTargets.Field)]
	public class MinAttribute : PropertyAttribute
	{
		public float min;
		public MinAttribute(float min) { this.min = min; }
	}

	public abstract class PropertyAttribute : Attribute
	{
		public int order { get; set; }
	}

	[AttributeUsage(AttributeTargets.Class)]
	public class CreateAssetMenuAttribute : Attribute
	{
		public string menuName { get; set; }
		public string fileName { get; set; }
		public int order { get; set; }
	}

	[AttributeUsage(AttributeTargets.Class)]
	public class AddComponentMenu : Attribute
	{
		public AddComponentMenu(string menuName) { }
		public AddComponentMenu(string menuName, int order) { }
	}

	[AttributeUsage(AttributeTargets.Class)]
	public class ExecuteInEditMode : Attribute { }

	[AttributeUsage(AttributeTargets.Class)]
	public class ExecuteAlways : Attribute { }

	[AttributeUsage(AttributeTargets.Class)]
	public class DisallowMultipleComponent : Attribute { }

	[AttributeUsage(AttributeTargets.Class)]
	public class DefaultExecutionOrder : Attribute
	{
		public int order;
		public DefaultExecutionOrder(int order) { this.order = order; }
	}

	[AttributeUsage(AttributeTargets.Method)]
	public class RuntimeInitializeOnLoadMethodAttribute : Attribute
	{
		public RuntimeInitializeOnLoadMethodAttribute() { }
		public RuntimeInitializeOnLoadMethodAttribute(RuntimeInitializeLoadType loadType) { }
	}

	public enum RuntimeInitializeLoadType { AfterSceneLoad, BeforeSceneLoad, AfterAssembliesLoaded, BeforeSplashScreen, SubsystemRegistration }

	[AttributeUsage(AttributeTargets.Field)]
	public class ColorUsageAttribute : PropertyAttribute
	{
		public ColorUsageAttribute(bool showAlpha) { }
		public ColorUsageAttribute(bool showAlpha, bool hdr) { }
	}

	[AttributeUsage(AttributeTargets.Class | AttributeTargets.Struct | AttributeTargets.Field)]
	public class PreferBinarySerialization : Attribute { }
}

namespace UnityEngine.Serialization
{
	[AttributeUsage(AttributeTargets.Field, AllowMultiple = true)]
	public class FormerlySerializedAsAttribute : Attribute
	{
		public string oldName { get; }
		public FormerlySerializedAsAttribute(string oldName) { this.oldName = oldName; }
	}
}
