using System;
using System.Collections.Generic;
using UnityEngine.Events;

namespace UnityEngine
{
	public class RectTransform : Transform
	{
		public Vector2 anchoredPosition { get; set; }
		public Vector3 anchoredPosition3D { get; set; }
		public Vector2 sizeDelta { get; set; }
		public Vector2 anchorMin { get; set; }
		public Vector2 anchorMax { get; set; } = Vector2.one;
		public Vector2 pivot { get; set; } = new Vector2(0.5f, 0.5f);
		public Vector2 offsetMin { get; set; }
		public Vector2 offsetMax { get; set; }
		public Rect rect => new Rect(-sizeDelta.x * pivot.x, -sizeDelta.y * pivot.y, sizeDelta.x, sizeDelta.y);

		public void SetSizeWithCurrentAnchors(Axis axis, float size)
		{
			var s = sizeDelta;
			if (axis == Axis.Horizontal) s.x = size; else s.y = size;
			sizeDelta = s;
		}

		public void GetWorldCorners(Vector3[] fourCornersArray)
		{
			var r = rect;
			fourCornersArray[0] = TransformPoint(new Vector3(r.xMin, r.yMin, 0f));
			fourCornersArray[1] = TransformPoint(new Vector3(r.xMin, r.yMax, 0f));
			fourCornersArray[2] = TransformPoint(new Vector3(r.xMax, r.yMax, 0f));
			fourCornersArray[3] = TransformPoint(new Vector3(r.xMax, r.yMin, 0f));
		}

		public void GetLocalCorners(Vector3[] fourCornersArray)
		{
			var r = rect;
			fourCornersArray[0] = new Vector3(r.xMin, r.yMin, 0f);
			fourCornersArray[1] = new Vector3(r.xMin, r.yMax, 0f);
			fourCornersArray[2] = new Vector3(r.xMax, r.yMax, 0f);
			fourCornersArray[3] = new Vector3(r.xMax, r.yMin, 0f);
		}

		public enum Axis { Horizontal, Vertical }
		public enum Edge { Left, Right, Top, Bottom }
	}

	public class Canvas : Behaviour
	{
		public RenderMode renderMode { get; set; }
		public Camera worldCamera { get; set; }
		public float scaleFactor { get; set; } = 1f;
		public int sortingOrder { get; set; }
		public string sortingLayerName { get; set; }
		public bool overrideSorting { get; set; }
		public float planeDistance { get; set; } = 100f;
		public static void ForceUpdateCanvases() { }
	}

	public enum RenderMode { ScreenSpaceOverlay, ScreenSpaceCamera, WorldSpace }

	public class CanvasGroup : Behaviour
	{
		public float alpha { get; set; } = 1f;
		public bool interactable { get; set; } = true;
		public bool blocksRaycasts { get; set; } = true;
		public bool ignoreParentGroups { get; set; }
	}

	public class CanvasRenderer : Component
	{
		public void SetAlpha(float alpha) { }
		public float GetAlpha() => 1f;
	}
}

namespace UnityEngine.UI
{
	public class Graphic : Behaviour
	{
		public virtual Color color { get; set; } = Color.white;
		// Unity's Graphic.material never returns null: unset means the shared
		// default UI material (AbilitySelectController clones it via Instantiate).
		internal static readonly Material defaultGraphicMaterial = new Material { name = "Default UI Material" };
		private Material materialField;
		public Material material
		{
			get => materialField != null ? materialField : defaultGraphicMaterial;
			set => materialField = value;
		}
		public Material materialForRendering => material;
		public RectTransform rectTransform => transform as RectTransform;
		public bool raycastTarget { get; set; } = true;
		public void SetAllDirty() { }
		public void SetVerticesDirty() { }
		public void CrossFadeAlpha(float alpha, float duration, bool ignoreTimeScale) { }
		public void CrossFadeColor(Color targetColor, float duration, bool ignoreTimeScale, bool useAlpha) { }
	}

	public class MaskableGraphic : Graphic
	{
		public bool maskable { get; set; } = true;
	}

	public class Image : MaskableGraphic
	{
		public Sprite sprite { get; set; }
		public Sprite overrideSprite { get; set; }
		public Type type { get; set; }
		public float fillAmount { get; set; } = 1f;
		public FillMethod fillMethod { get; set; }
		public bool fillClockwise { get; set; } = true;
		public int fillOrigin { get; set; }
		public bool preserveAspect { get; set; }
		public void SetNativeSize() { }

		public enum Type { Simple, Sliced, Tiled, Filled }
		public enum FillMethod { Horizontal, Vertical, Radial90, Radial180, Radial360 }
	}

	public class RawImage : MaskableGraphic
	{
		public Texture texture { get; set; }
		public Rect uvRect { get; set; } = new Rect(0f, 0f, 1f, 1f);
	}

	public class Text : MaskableGraphic
	{
		public string text { get; set; } = "";
		public Font font { get; set; }
		public int fontSize { get; set; } = 14;
		public FontStyle fontStyle { get; set; }
		public TextAnchor alignment { get; set; }
		public bool resizeTextForBestFit { get; set; }
		public float lineSpacing { get; set; } = 1f;
		public bool supportRichText { get; set; } = true;
	}

	public class Selectable : Behaviour,
		UnityEngine.EventSystems.IPointerEnterHandler, UnityEngine.EventSystems.IPointerExitHandler,
		UnityEngine.EventSystems.IPointerDownHandler, UnityEngine.EventSystems.IPointerUpHandler,
		UnityEngine.EventSystems.ISelectHandler, UnityEngine.EventSystems.IDeselectHandler,
		UnityEngine.EventSystems.IMoveHandler
	{
		public bool interactable { get; set; } = true;
		public Navigation navigation { get; set; }
		public Image image { get; set; }
		public Graphic targetGraphic { get; set; }
		public ColorBlock colors { get; set; } = ColorBlock.defaultColorBlock;
		public virtual void Select() { }
		public virtual void OnSelect(UnityEngine.EventSystems.BaseEventData eventData) { }
		public virtual void OnDeselect(UnityEngine.EventSystems.BaseEventData eventData) { }
		public virtual void OnPointerEnter(UnityEngine.EventSystems.PointerEventData eventData) { }
		public virtual void OnPointerExit(UnityEngine.EventSystems.PointerEventData eventData) { }
		public virtual void OnPointerDown(UnityEngine.EventSystems.PointerEventData eventData) { }
		public virtual void OnPointerUp(UnityEngine.EventSystems.PointerEventData eventData) { }
		public virtual void OnMove(UnityEngine.EventSystems.AxisEventData eventData) { }
		public bool IsInteractable() => interactable;
		public Selectable FindSelectableOnUp() => null;
		public Selectable FindSelectableOnDown() => null;
		public Selectable FindSelectableOnLeft() => null;
		public Selectable FindSelectableOnRight() => null;

		public struct Navigation
		{
			public Mode mode { get; set; }
			public Selectable selectOnUp { get; set; }
			public Selectable selectOnDown { get; set; }
			public Selectable selectOnLeft { get; set; }
			public Selectable selectOnRight { get; set; }
			public enum Mode { None = 0, Horizontal = 1, Vertical = 2, Automatic = 3, Explicit = 4 }
		}
	}

	[Serializable]
	public struct ColorBlock
	{
		public Color normalColor, highlightedColor, pressedColor, selectedColor, disabledColor;
		public float colorMultiplier, fadeDuration;
		public static ColorBlock defaultColorBlock => new ColorBlock
		{
			normalColor = Color.white, highlightedColor = Color.white, pressedColor = Color.gray,
			selectedColor = Color.white, disabledColor = Color.gray, colorMultiplier = 1f, fadeDuration = 0.1f
		};
	}

	public class Button : Selectable, UnityEngine.EventSystems.IPointerClickHandler, UnityEngine.EventSystems.ISubmitHandler
	{
		public ButtonClickedEvent onClick { get; set; } = new ButtonClickedEvent();
		public virtual void OnPointerClick(UnityEngine.EventSystems.PointerEventData eventData) => onClick.Invoke();
		public virtual void OnSubmit(UnityEngine.EventSystems.BaseEventData eventData) => onClick.Invoke();
		public class ButtonClickedEvent : UnityEvent { }
	}

	public class Toggle : Selectable
	{
		public bool isOn { get; set; }
		public ToggleEvent onValueChanged { get; set; } = new ToggleEvent();
		public Graphic graphic { get; set; }
		public void SetIsOnWithoutNotify(bool value) => isOn = value;
		public class ToggleEvent : UnityEvent<bool> { }
	}

	public class Slider : Selectable
	{
		public float value { get; set; }
		public float minValue { get; set; }
		public float maxValue { get; set; } = 1f;
		public bool wholeNumbers { get; set; }
		public SliderEvent onValueChanged { get; set; } = new SliderEvent();
		public void SetValueWithoutNotify(float input) => value = input;
		public class SliderEvent : UnityEvent<float> { }
	}

	public class Scrollbar : Selectable
	{
		public float value { get; set; }
		public float size { get; set; } = 1f;
	}

	public class ScrollRect : Behaviour
	{
		public Vector2 normalizedPosition { get; set; }
		public float verticalNormalizedPosition { get; set; }
		public float horizontalNormalizedPosition { get; set; }
		public RectTransform content { get; set; }
		public RectTransform viewport { get; set; }
		public Scrollbar verticalScrollbar { get; set; }
	}

	public class Dropdown : Selectable
	{
		public int value { get; set; }
		public List<OptionData> options { get; set; } = new List<OptionData>();
		public DropdownEvent onValueChanged { get; set; } = new DropdownEvent();
		public void RefreshShownValue() { }
		public void ClearOptions() => options.Clear();
		public void AddOptions(List<string> newOptions) { foreach (var o in newOptions) options.Add(new OptionData(o)); }
		public class OptionData
		{
			public string text { get; set; }
			public Sprite image { get; set; }
			public OptionData() { }
			public OptionData(string text) { this.text = text; }
		}
		public class DropdownEvent : UnityEvent<int> { }
	}

	public class InputField : Selectable
	{
		public string text { get; set; } = "";
		public UnityEvent<string> onValueChanged { get; set; } = new UnityEvent<string>();
		public UnityEvent<string> onEndEdit { get; set; } = new UnityEvent<string>();
	}

	public class LayoutElement : Behaviour
	{
		public float minWidth { get; set; } = -1f;
		public float minHeight { get; set; } = -1f;
		public float preferredWidth { get; set; } = -1f;
		public float preferredHeight { get; set; } = -1f;
		public bool ignoreLayout { get; set; }
	}

	public class LayoutGroup : Behaviour
	{
		public RectOffset padding { get; set; } = new RectOffset();
	}

	public class HorizontalOrVerticalLayoutGroup : LayoutGroup
	{
		public float spacing { get; set; }
		public bool childForceExpandWidth { get; set; } = true;
		public bool childForceExpandHeight { get; set; } = true;
	}

	public class HorizontalLayoutGroup : HorizontalOrVerticalLayoutGroup { }
	public class VerticalLayoutGroup : HorizontalOrVerticalLayoutGroup { }

	public class GridLayoutGroup : LayoutGroup
	{
		public Vector2 cellSize { get; set; }
		public Vector2 spacing { get; set; }
	}

	public class ContentSizeFitter : Behaviour
	{
		public FitMode horizontalFit { get; set; }
		public FitMode verticalFit { get; set; }
		public enum FitMode { Unconstrained, MinSize, PreferredSize }
	}

	public class Mask : Behaviour
	{
		public bool showMaskGraphic { get; set; } = true;
	}

	public class Outline : Behaviour
	{
		public Color effectColor { get; set; }
		public Vector2 effectDistance { get; set; }
	}

	public class CanvasScaler : Behaviour
	{
		public ScaleMode uiScaleMode { get; set; }
		public Vector2 referenceResolution { get; set; } = new Vector2(800f, 600f);
		public float matchWidthOrHeight { get; set; }
		public float scaleFactor { get; set; } = 1f;
		public enum ScaleMode { ConstantPixelSize, ScaleWithScreenSize, ConstantPhysicalSize }
	}

	public static class LayoutRebuilder
	{
		public static void ForceRebuildLayoutImmediate(RectTransform layoutRoot) { }
		public static void MarkLayoutForRebuild(RectTransform rect) { }
	}

	// Present in exported scenes; raycasting is presentation-only.
	public class GraphicRaycaster : UnityEngine.EventSystems.UIBehaviour
	{
		public bool ignoreReversedGraphics { get; set; } = true;
	}
}

namespace UnityEngine
{
	public class Font : Object
	{
		public static Font CreateDynamicFontFromOSFont(string fontname, int size) => new Font();
	}

	public enum FontStyle { Normal, Bold, Italic, BoldAndItalic }

	public enum TextAnchor
	{
		UpperLeft, UpperCenter, UpperRight,
		MiddleLeft, MiddleCenter, MiddleRight,
		LowerLeft, LowerCenter, LowerRight
	}

	public class RectOffset
	{
		public int left { get; set; }
		public int right { get; set; }
		public int top { get; set; }
		public int bottom { get; set; }
		public RectOffset() { }
		public RectOffset(int left, int right, int top, int bottom)
		{ this.left = left; this.right = right; this.top = top; this.bottom = bottom; }
	}
}

namespace UnityEngine.EventSystems
{
	public interface IEventSystemHandler { }

	public class BaseEventData
	{
		public BaseEventData() { }
		public BaseEventData(EventSystem eventSystem) { }
		public GameObject selectedObject { get; set; }
	}

	public class PointerEventData : BaseEventData
	{
		public Vector2 position { get; set; }
		public Vector2 delta { get; set; }
		public GameObject pointerEnter { get; set; }
		public GameObject pointerPress { get; set; }
		public int clickCount { get; set; }
		public InputButton button { get; set; }
		public enum InputButton { Left, Right, Middle }
	}

	public class AxisEventData : BaseEventData
	{
		public Vector2 moveVector { get; set; }
		public MoveDirection moveDir { get; set; }
	}

	public enum MoveDirection { Left, Up, Right, Down, None }

	public interface IPointerEnterHandler : IEventSystemHandler { void OnPointerEnter(PointerEventData eventData); }
	public interface IPointerExitHandler : IEventSystemHandler { void OnPointerExit(PointerEventData eventData); }
	public interface IPointerDownHandler : IEventSystemHandler { void OnPointerDown(PointerEventData eventData); }
	public interface IPointerUpHandler : IEventSystemHandler { void OnPointerUp(PointerEventData eventData); }
	public interface IPointerClickHandler : IEventSystemHandler { void OnPointerClick(PointerEventData eventData); }
	public interface ISelectHandler : IEventSystemHandler { void OnSelect(BaseEventData eventData); }
	public interface IDeselectHandler : IEventSystemHandler { void OnDeselect(BaseEventData eventData); }
	public interface ISubmitHandler : IEventSystemHandler { void OnSubmit(BaseEventData eventData); }
	public interface ICancelHandler : IEventSystemHandler { void OnCancel(BaseEventData eventData); }
	public interface IMoveHandler : IEventSystemHandler { void OnMove(AxisEventData eventData); }
	public interface IDragHandler : IEventSystemHandler { void OnDrag(PointerEventData eventData); }
	public interface IBeginDragHandler : IEventSystemHandler { void OnBeginDrag(PointerEventData eventData); }
	public interface IEndDragHandler : IEventSystemHandler { void OnEndDrag(PointerEventData eventData); }

	public abstract class UIBehaviour : MonoBehaviour
	{
		protected virtual void Awake() { }
		protected virtual void OnEnable() { }
		protected virtual void Start() { }
		protected virtual void OnDisable() { }
		protected virtual void OnDestroy() { }
	}

	public class EventSystem : UIBehaviour
	{
		// Per world: set by this world's EventSystem object.
		public static EventSystem current { get => UnityShim.World.Current.eventSystem; set => UnityShim.World.Current.eventSystem = value; }
		public GameObject currentSelectedGameObject { get; private set; }
		public bool sendNavigationEvents { get; set; } = true;
		public bool alreadySelecting => false;
		protected virtual void Update() { }
		public void SetSelectedGameObject(GameObject selected) => currentSelectedGameObject = selected;
		public void SetSelectedGameObject(GameObject selected, BaseEventData pointer) => currentSelectedGameObject = selected;
		public bool IsPointerOverGameObject() => false;
		public void UpdateModules() { }
	}

	public class EventTrigger : Behaviour
	{
		public List<Entry> triggers { get; set; } = new List<Entry>();
		public class Entry
		{
			public EventTriggerType eventID;
			public TriggerEvent callback = new TriggerEvent();
		}
		public class TriggerEvent : UnityEngine.Events.UnityEvent<BaseEventData> { }
	}

	// Minimal ExecuteEvents: dispatches to handler interfaces on the target's components.
	public static class ExecuteEvents
	{
		public delegate void EventFunction<T1>(T1 handler, BaseEventData eventData);

		public static EventFunction<ISubmitHandler> submitHandler { get; } = (h, d) => h.OnSubmit(d);
		public static EventFunction<ICancelHandler> cancelHandler { get; } = (h, d) => h.OnCancel(d);
		public static EventFunction<IPointerClickHandler> pointerClickHandler { get; } = (h, d) => h.OnPointerClick(d as PointerEventData);
		public static EventFunction<ISelectHandler> selectHandler { get; } = (h, d) => h.OnSelect(d);
		public static EventFunction<IDeselectHandler> deselectHandler { get; } = (h, d) => h.OnDeselect(d);

		public static bool Execute<T>(GameObject target, BaseEventData eventData, EventFunction<T> functor) where T : IEventSystemHandler
		{
			if (target == null) return false;
			bool any = false;
			foreach (var c in target.GetComponents<Component>())
			{
				if (c is T handler)
				{
					functor(handler, eventData);
					any = true;
				}
			}
			return any;
		}
	}

	public enum EventTriggerType
	{
		PointerEnter, PointerExit, PointerDown, PointerUp, PointerClick,
		Drag, Drop, Scroll, UpdateSelected, Select, Deselect, Move,
		InitializePotentialDrag, BeginDrag, EndDrag, Submit, Cancel
	}
}

namespace TMPro
{
	using UnityEngine.UI;

	public class TMP_FontAsset : UnityEngine.Object
	{
		public UnityEngine.Material material { get; set; }
	}

	public abstract class TMP_Text : MaskableGraphic
	{
		public virtual string text { get; set; } = "";
		public float fontSize { get; set; } = 36f;
		public bool enableAutoSizing { get; set; }
		public float fontSizeMin { get; set; }
		public float fontSizeMax { get; set; }
		public TMP_FontAsset font { get; set; }
		public FontStyles fontStyle { get; set; }
		public TextAlignmentOptions alignment { get; set; }
		public override UnityEngine.Color color { get; set; } = UnityEngine.Color.white;
		public float alpha { get; set; } = 1f;
		public bool richText { get; set; } = true;
		public float characterSpacing { get; set; }
		public float lineSpacing { get; set; }
		public bool enableWordWrapping { get; set; } = true;
		public TextOverflowModes overflowMode { get; set; }
		public UnityEngine.Vector4 margin { get; set; }
		public int maxVisibleCharacters { get; set; } = 99999;
		public TMP_TextInfo textInfo { get; } = new TMP_TextInfo();
		public bool havePropertiesChanged { get; set; }
		public UnityEngine.Material fontMaterial { get; set; }
		public UnityEngine.Material fontSharedMaterial { get; set; }
		public float outlineWidth { get; set; }
		public UnityEngine.Color outlineColor { get; set; }
		public void SetText(string text) => this.text = text;
		public void SetText(string text, bool syncTextInputBox) => this.text = text;
		public void ForceMeshUpdate() { }
		public void ForceMeshUpdate(bool ignoreActiveState, bool forceTextReparsing) { }
		public UnityEngine.Vector2 GetPreferredValues() => UnityEngine.Vector2.zero;
		public UnityEngine.Vector2 GetPreferredValues(string text) => UnityEngine.Vector2.zero;
		public void UpdateFontAsset() { }
		public void UpdateMeshPadding() { }
		public float preferredWidth => 0f;
		public float preferredHeight => 0f;
	}

	public class TextMeshProUGUI : TMP_Text { }

	public class TextMeshPro : TMP_Text
	{
		public int sortingOrder { get; set; }
		public string sortingLayerName { get; set; }
	}

	public class TMP_TextInfo
	{
		public int characterCount;
		public TMP_CharacterInfo[] characterInfo = Array.Empty<TMP_CharacterInfo>();
	}

	public struct TMP_CharacterInfo
	{
		public char character;
		public bool isVisible;
	}

	public class TMP_InputField : Selectable
	{
		public string text { get; set; } = "";
		public UnityEngine.Events.UnityEvent<string> onValueChanged { get; set; } = new UnityEngine.Events.UnityEvent<string>();
		public UnityEngine.Events.UnityEvent<string> onEndEdit { get; set; } = new UnityEngine.Events.UnityEvent<string>();
		public UnityEngine.Events.UnityEvent<string> onSubmit { get; set; } = new UnityEngine.Events.UnityEvent<string>();
		public void ActivateInputField() { }
		public void DeactivateInputField() { }
	}

	public class TMP_Dropdown : Dropdown { }

	public enum FontStyles { Normal = 0, Bold = 1, Italic = 2, Underline = 4, Strikethrough = 64, UpperCase = 16 }

	public enum TextAlignmentOptions
	{
		TopLeft = 257, Top = 258, TopRight = 260,
		Left = 513, Center = 514, Right = 516, Midline = 4098,
		BottomLeft = 1025, Bottom = 1026, BottomRight = 1028
	}

	public enum TextOverflowModes { Overflow, Ellipsis, Masking, Truncate, ScrollRect, Page, Linked }
}
