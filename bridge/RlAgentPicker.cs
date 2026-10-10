using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem;
using UnityEngine.UI;

// The opponent picker on the local character select (RlAgent, RlAgentCatalog;
// on whenever BOPL_AGENT_CATALOG is set). The sixth select box makes way for
// an opponent "seat": a select column like a joined player's, showing the bot,
// its rating, the kit it will play (cycling through the set when the kit is
// Random) and its slime in the colour it will get. Gamepad Y, Tab or a click
// on the seat opens the picker over the boxes: the bots grouped as the catalog
// lists them (league, exploiters, specialists, scripted, rosters) on the left,
// the highlighted bot's details and kit choice on the right. Any pad or the
// keyboard drives it: up/down picks a bot, left/right the kit, LB/RB (Q/E)
// jumps a group, A (Enter) chooses, B (Esc) backs out; the mouse clicks,
// double-clicks and scrolls. Choosing opens the bot in the agent shim once to
// check it loads, then saves it as the next match's opponent. Everything is
// built at runtime from the select box's own sprites and font, so it needs no
// scene or prefab changes. While the picker is open (and on the frame it
// closes) BlocksInput keeps the select boxes, the click-to-join and the back
// to menu check from also acting on the same presses.
public class RlAgentPicker : MonoBehaviour
{
    private sealed class Row
    {
        public RectTransform Rect;
        public Image Outer, Inner, Check;
        public TextMeshProUGUI Name, Rating;
        public int Bot = -1;          // -1: a group header
        public float Top, Height;
    }

    private sealed class KitCell
    {
        public RectTransform Rect;
        public Image Outer, Inner;
        public Image[] Icons = new Image[3];
    }

    private const float PanelWidth = 2800f, PanelHeight = 1880f;
    private const float ListWidth = 1000f, ListHeight = 1500f, ListX = -860f, ListY = -10f;
    private const float RowHeight = 112f, HeaderHeight = 92f, RowGap = 12f;
    private const float DetailLeft = -250f, DetailCenter = 560f;
    private const float RepeatDelay = 0.35f, RepeatRateVertical = 0.07f, RepeatRateHorizontal = 0.16f;
    private const int GridColumns = 5, GridCells = 20;

    private static RlAgentPicker instance;
    private bool open;
    private int toggledFrame = -1;

    // The select boxes, the click-to-join and the back-to-menu check skip their
    // input while the picker has it.
    public static bool BlocksInput => instance != null && (instance.open || instance.toggledFrame == Time.frameCount);

    private CharacterSelectHandler handler;
    private PlayerColors playerColors;
    private NamedSpriteList abilities;
    private TMP_FontAsset font;
    private Color blue, edge, orange, deep, light;
    private Sprite rounded, columnFill, columnBorder, ring, disc, arrow, check, platform, slime;
    private List<RlAgentCatalog.Bot> bots;

    // Seat.
    private TextMeshProUGUI seatTitle, seatInfo, seatKit, seatKey, seatKeyAction;
    private readonly Image[] seatIcons = new Image[3];
    private Image seatSlime;
    private float seatCycle;
    private int seatShown;

    // Overlay.
    private CanvasGroup overlay;
    private RectTransform panel, listViewport, listContent;
    private readonly List<Row> rows = new List<Row>();
    private TextMeshProUGUI detailName, detailKind, detailRating, detailDescription, kitHeader, kitOptionText, kitNote, errorText, emptyText;
    private Image kindChip;
    private readonly Image[] kitRings = new Image[3], kitIcons = new Image[3];
    private readonly TextMeshProUGUI[] kitNames = new TextMeshProUGUI[3];
    private RectTransform kitLeft, kitRight;
    private readonly List<KitCell> cells = new List<KitCell>();
    private TextMeshProUGUI[] hintKeys, hintActions;
    private Image[] hintKeyFills;

    private int cursor, kitOption, hoverRow = -1;
    private float scroll, scrollTarget, openTime, lastClickTime;
    private int lastClickRow = -1;
    private bool usingGamepad;
    private int heldVertical, heldHorizontal;
    private float timerVertical, timerHorizontal;

    // CharacterSelectHandler.Start: take over the last select box's slot.
    public static void Attach(CharacterSelectHandler handler, CharacterSelectBox seatBox, PlayerColors colors)
    {
        if (!RlAgentCatalog.Present || SteamManager.instance == null || seatBox == null) return;
        RlAgentCatalog.EnsureLoaded();
        handler.gameObject.AddComponent<RlAgentPicker>().Build(handler, seatBox, colors);
    }

    private void OnDestroy()
    {
        if (instance == this) instance = null;
    }

    private void Build(CharacterSelectHandler handler, CharacterSelectBox seatBox, PlayerColors colors)
    {
        instance = this;
        this.handler = handler;
        playerColors = colors;
        abilities = SteamManager.instance.abilityIcons;
        bots = RlAgentCatalog.Bots;
        blue = handler.blue; orange = handler.orange;
        edge = new Color(0.082f, 0.333f, 0.509f);
        deep = new Color(0.09f, 0.46f, 0.76f);
        light = new Color(0.36f, 0.74f, 0.98f);
        font = seatBox.readyPlayerNameText.font;
        // The select box's sprites, by name (the hierarchy names are not stable).
        var sprites = new Dictionary<string, Sprite>();
        foreach (var image in seatBox.GetComponentsInChildren<Image>(true))
            if (image.sprite != null && !sprites.ContainsKey(image.sprite.name)) sprites[image.sprite.name] = image.sprite;
        Sprite Named(string name)
        {
            if (sprites.TryGetValue(name, out var sprite)) return sprite;
            Debug.LogWarning("RlAgentPicker: no sprite " + name + " in the select box");
            return null;
        }
        columnFill = Named("charSelect_Components_1");
        columnBorder = Named("charSelect_Components_0");
        ring = Named("charSelect_Components_12");
        disc = Named("charSelect_Components_13");
        arrow = Named("charSelect_Components_6");
        check = Named("charSelect_Components_8");
        platform = Named("SmallCapsule");
        slime = Named("idle_final_0");
        rounded = Slice(Named("charSelect_Components_4"), 120f);
        usingGamepad = Gamepad.all.Count > 0;

        // The seat takes the last box's slot; that box can no longer be joined.
        seatBox.gameObject.SetActive(false);
        CharacterSelectBox.occupiedRectangles[seatBox.RectangleIndex] = true;
        BuildSeat(seatBox.GetComponent<RectTransform>());
        BuildOverlay(seatBox.GetComponentInParent<Canvas>().rootCanvas);
        cursor = Mathf.Max(0, bots.IndexOf(RlAgentCatalog.Chosen));
        RefreshSeat();
    }

    // ---- building blocks ----

    private static Sprite Slice(Sprite sprite, float border)
    {
        if (sprite == null) return null;
        Rect rect = sprite.packed ? sprite.textureRect : sprite.rect;
        return Sprite.Create(sprite.texture, rect, new Vector2(0.5f, 0.5f), sprite.pixelsPerUnit, 0, SpriteMeshType.FullRect, new Vector4(border, border, border, border));
    }

    private static RectTransform Node(string name, Transform parent, Vector2 position, Vector2 size)
    {
        var node = new GameObject(name, typeof(RectTransform)).GetComponent<RectTransform>();
        node.SetParent(parent, false);
        node.anchorMin = node.anchorMax = node.pivot = new Vector2(0.5f, 0.5f);
        node.anchoredPosition = position;
        node.sizeDelta = size;
        return node;
    }

    // Anchor to the top of the parent (the list content grows downwards).
    private static void Pin(RectTransform node)
    {
        node.anchorMin = node.anchorMax = new Vector2(0.5f, 1f);
    }

    private static Image Picture(string name, Transform parent, Sprite sprite, Color color, Vector2 position, Vector2 size, bool raycast = false)
    {
        var image = Node(name, parent, position, size).gameObject.AddComponent<Image>();
        image.sprite = sprite;
        image.color = color;
        image.raycastTarget = raycast;
        return image;
    }

    // A rounded box: an outline colour behind a fill inset by stroke.
    private (Image outer, Image inner) Box(string name, Transform parent, Vector2 position, Vector2 size, Color outline, Color fill, float radius, float stroke, bool raycast = false)
    {
        var outer = Picture(name, parent, rounded, outline, position, size, raycast);
        outer.type = Image.Type.Sliced;
        outer.pixelsPerUnitMultiplier = 120f / radius;
        var inner = Picture("fill", outer.transform, rounded, fill, Vector2.zero, size - Vector2.one * (2f * stroke));
        inner.type = Image.Type.Sliced;
        inner.pixelsPerUnitMultiplier = 120f / Mathf.Max(8f, radius - stroke);
        return (outer, inner);
    }

    private TextMeshProUGUI Label(string name, Transform parent, string text, float size, Color color, Vector2 position, Vector2 box, TextAlignmentOptions alignment = TextAlignmentOptions.Center)
    {
        var label = Node(name, parent, position, box).gameObject.AddComponent<TextMeshProUGUI>();
        label.font = font;
        label.fontSize = size;
        label.color = color;
        label.alignment = alignment;
        label.enableWordWrapping = true;
        label.overflowMode = TextOverflowModes.Ellipsis;
        label.raycastTarget = false;
        label.richText = true;
        label.text = text;
        return label;
    }

    private static void OnClick(Graphic target, UnityEngine.Events.UnityAction action)
    {
        target.raycastTarget = true;
        var button = target.gameObject.AddComponent<Button>();
        button.transition = Selectable.Transition.None;
        button.navigation = new Navigation { mode = Navigation.Mode.None };
        button.onClick.AddListener(action);
    }

    private Sprite Icon(int ability) => ability >= 0 && ability < abilities.sprites.Count ? abilities.sprites[ability].sprite : null;

    private string AbilityName(int ability)
    {
        if (ability >= 0 && ability < abilities.sprites.Count && LocalizedText.localizationTable != null)
            return LocalizedText.localizationTable.GetText(abilities.sprites[ability].name.ToLower(), Settings.Get().Language);
        return RlAgentCatalog.AbilityNames.TryGetValue(ability, out string name) ? name : ability.ToString();
    }

    private static string KindText(RlAgentCatalog.Bot bot)
    {
        switch (bot.Kind) {
        case "main": return "league main";
        case "exploiter": return "exploiter";
        case "specialist": return "specialist";
        case "scripted": return "scripted bot";
        case "roster": return "roster";
        default: return bot.Kind ?? "";
        }
    }

    private static string RatingText(RlAgentCatalog.Bot bot) => bot.Rating.HasValue ? Mathf.RoundToInt(bot.Rating.Value).ToString() : "";

    // ---- seat ----

    private void BuildSeat(RectTransform box)
    {
        var seat = Node("RlAgentSeat", box.parent, box.anchoredPosition, box.sizeDelta);
        seat.anchorMin = box.anchorMin; seat.anchorMax = box.anchorMax; seat.pivot = box.pivot;
        seat.anchoredPosition = box.anchoredPosition;
        seat.localScale = box.localScale;
        seat.SetSiblingIndex(box.GetSiblingIndex());
        // The joined column, as the select box draws it (SELECT_COLOR, SELECT_BORDER).
        var fill = Picture("fill", seat, columnFill, blue, new Vector2(-8f, 29.4f), new Vector2(607.1f, 1801.7f), true);
        fill.gameObject.AddComponent<Mask>().showMaskGraphic = true;
        OnClick(fill, () => Open());
        Picture("border", seat, columnBorder, edge, new Vector2(-13.4f, 29.4f), new Vector2(640f, 1801.7f));
        Transform column = fill.transform;
        Label("header", seat, "opponent", 78f, edge, new Vector2(-4.9f, 840.5f), new Vector2(560f, 128f));
        seatTitle = Label("title", column, "", 58f, Color.white, new Vector2(0f, 700f), new Vector2(540f, 130f));
        seatInfo = Label("info", column, "", 48f, edge, new Vector2(0f, 598f), new Vector2(560f, 64f));
        float[] rows = { 334.3f, 18.3f, -296.8f };
        for (int i = 0; i < 3; i++) {
            var circle = Picture("ability" + i, column, ring, edge, new Vector2(-1f, rows[i] + 4f), new Vector2(263.8f, 280.6f));
            seatIcons[i] = Picture("icon", circle.transform, null, Color.white, new Vector2(0.7f, -5.8f), new Vector2(419.9f, 402.4f));
        }
        seatKit = Label("kit", column, "", 46f, Color.white, new Vector2(0f, -480f), new Vector2(570f, 120f));
        Picture("platform", column, platform, Color.white, new Vector2(0.7f, -858f), new Vector2(944.3f, 515f));
        seatSlime = Picture("slime", column, slime, Color.white, new Vector2(0.7f, -651f), new Vector2(240.7f, 165.8f));
        // A badge where the box shows its device, and one where it has the ready button.
        var top = Picture("badge", seat, disc, blue, new Vector2(-8.4f, 1080.4f), new Vector2(260f, 263f));
        Picture("ring", top.transform, ring, edge, Vector2.zero, new Vector2(275f, 266f));
        Label("bot", top.transform, "bot", 84f, Color.white, new Vector2(0f, -4f), new Vector2(240f, 120f));
        var key = Picture("change", seat, disc, blue, new Vector2(0f, -847.2f), new Vector2(236f, 239f), true);
        OnClick(key, () => Open());
        Picture("ring", key.transform, ring, edge, Vector2.zero, new Vector2(250f, 241f));
        seatKey = Label("key", key.transform, "", 84f, Color.white, new Vector2(0f, 22f), new Vector2(220f, 100f));
        seatKeyAction = Label("action", key.transform, "change", 34f, edge, new Vector2(0f, -50f), new Vector2(220f, 50f));
    }

    private void RefreshSeat()
    {
        var bot = RlAgentCatalog.Chosen;
        if (bot == null) {
            seatTitle.text = "no bot";
            seatInfo.text = "";
            seatKit.text = RlAgentCatalog.LoadError != null ? "catalog error" : "none available";
            foreach (var icon in seatIcons) icon.enabled = false;
            return;
        }
        seatTitle.text = bot.Title.ToLower();
        string rating = RatingText(bot);
        seatInfo.text = KindText(bot) + (rating.Length > 0 ? "  <color=#ffffff>" + rating + "</color>" : "");
        if (RlAgentCatalog.ChosenKit == RlAgentCatalog.RandomKit) {
            seatKit.text = "random kit\n<size=85%>1 of " + bot.Kits.Count + " each round</size>";
            ShowSeatKit(bot.Kits[seatShown % bot.Kits.Count], false);
        } else if (bot.Kits.Count > 0) {
            int[] kit = bot.Kits[RlAgentCatalog.ChosenKit];
            seatKit.text = AbilityName(kit[0]) + " / " + AbilityName(kit[1]) + " / " + AbilityName(kit[2]);
            ShowSeatKit(kit, false);
        } else {
            seatKit.text = "its training kit";
            foreach (var icon in seatIcons) icon.enabled = false;
        }
    }

    private void ShowSeatKit(int[] kit, bool pop)
    {
        for (int i = 0; i < 3; i++) {
            seatIcons[i].enabled = true;
            seatIcons[i].sprite = Icon(kit[i]);
            if (pop) seatIcons[i].transform.localScale = Vector3.one * 1.18f;
        }
    }

    private void UpdateSeat(float dt)
    {
        var bot = RlAgentCatalog.Chosen;
        if (bot != null && RlAgentCatalog.ChosenKit == RlAgentCatalog.RandomKit && bot.Kits.Count > 1) {
            seatCycle += dt;
            if (seatCycle > 1.3f) {
                seatCycle = 0f;
                seatShown = (seatShown + 1) % bot.Kits.Count;
                ShowSeatKit(bot.Kits[seatShown], true);
            }
        }
        float ease = 1f - Mathf.Exp(-12f * dt);
        foreach (var icon in seatIcons) icon.transform.localScale = Vector3.Lerp(icon.transform.localScale, Vector3.one, ease);
        // The agent takes the first colour no joined player has (RlAgent.AddPlayers).
        var taken = new List<Material>();
        foreach (var box in handler.characterSelectBoxes)
            if (box.isActiveAndEnabled && (box.menuState == CharSelectMenu.select || box.menuState == CharSelectMenu.ready || box.menuState == CharSelectMenu.abilityGrid))
                taken.Add(playerColors[box.playerInit.color].playerMaterial);
        seatSlime.material = playerColors[RlAgent.ColorIndex(taken, playerColors)].uiMaterial;
        seatKey.text = usingGamepad ? "y" : "tab";
        seatKey.fontSize = usingGamepad ? 96f : 64f;
    }

    // ---- overlay ----

    private void BuildOverlay(Canvas root)
    {
        var canvasObject = new GameObject("RlAgentPicker", typeof(RectTransform), typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster), typeof(CanvasGroup));
        // A scene root of its own: under the select canvas it would be a nested
        // canvas, sized by its parent instead of the screen.
        canvasObject.transform.SetParent(null, false);
        var canvas = canvasObject.GetComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvas.sortingOrder = 500;
        var scaler = canvasObject.GetComponent<CanvasScaler>();
        var rootScaler = root.GetComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = rootScaler != null ? rootScaler.referenceResolution : new Vector2(4096f, 2160f);
        scaler.screenMatchMode = CanvasScaler.ScreenMatchMode.MatchWidthOrHeight;
        scaler.matchWidthOrHeight = rootScaler != null ? rootScaler.matchWidthOrHeight : 0f;
        scaler.referencePixelsPerUnit = 100f;
        overlay = canvasObject.GetComponent<CanvasGroup>();
        var rootRect = canvasObject.GetComponent<RectTransform>();

        var backdrop = Picture("backdrop", rootRect, null, new Color(0.02f, 0.12f, 0.24f, 0.6f), Vector2.zero, Vector2.zero, true);
        var backdropRect = backdrop.rectTransform;
        backdropRect.anchorMin = Vector2.zero; backdropRect.anchorMax = Vector2.one; backdropRect.sizeDelta = Vector2.zero;
        OnClick(backdrop, () => Close(false));

        var (panelOuter, panelInner) = Box("panel", rootRect, new Vector2(0f, -20f), new Vector2(PanelWidth, PanelHeight), edge, blue, 120f, 18f, true);
        panel = panelOuter.rectTransform;
        panelInner.raycastTarget = true;   // clicks on the panel do not reach the backdrop
        Transform p = panel;
        Label("title", p, "choose your opponent", 104f, edge, new Vector2(0f, 842f), new Vector2(2400f, 140f));

        // Left: the bots by group.
        var (listOuter, listInner) = Box("list", p, new Vector2(ListX, ListY), new Vector2(ListWidth + 40f, ListHeight + 40f), edge, deep, 70f, 12f);
        listViewport = Node("viewport", listOuter.transform, Vector2.zero, new Vector2(ListWidth, ListHeight));
        listViewport.gameObject.AddComponent<RectMask2D>();
        listContent = Node("content", listViewport, Vector2.zero, new Vector2(ListWidth, 0f));
        listContent.anchorMin = listContent.anchorMax = listContent.pivot = new Vector2(0.5f, 1f);
        float y = 0f;
        string group = null;
        for (int i = 0; i < bots.Count; i++) {
            var bot = bots[i];
            if (bot.Group != group) {
                group = bot.Group;
                int count = bots.FindAll(b => b.Group == group).Count;
                var header = new Row { Top = y, Height = HeaderHeight };
                header.Rect = Node("group " + group, listContent, new Vector2(0f, -y - HeaderHeight * 0.5f), new Vector2(ListWidth, HeaderHeight));
                Pin(header.Rect);
                header.Name = Label("name", header.Rect, group.ToLower() + "  <size=75%><color=#9fd6ff>" + count + "</color></size>", 52f, edge, new Vector2(10f, -6f), new Vector2(ListWidth - 60f, HeaderHeight), TextAlignmentOptions.Left);
                rows.Add(header);
                y += HeaderHeight;
            }
            var row = new Row { Bot = i, Top = y, Height = RowHeight };
            var (outer, inner) = Box("bot " + bot.Id, listContent, new Vector2(0f, -y - RowHeight * 0.5f), new Vector2(ListWidth - 20f, RowHeight), edge, blue, 46f, 7f, true);
            row.Rect = outer.rectTransform; row.Outer = outer; row.Inner = inner;
            Pin(row.Rect);
            int index = rows.Count;
            OnClick(outer, () => ClickRow(index));
            row.Name = Label("name", outer.transform, bot.Title.ToLower(), 50f, Color.white, new Vector2(-110f, 0f), new Vector2(ListWidth - 300f, RowHeight - 10f), TextAlignmentOptions.Left);
            row.Name.enableWordWrapping = false;
            row.Rating = Label("rating", outer.transform, RatingText(bot), 46f, edge, new Vector2(ListWidth * 0.5f - 110f, 0f), new Vector2(150f, RowHeight - 10f), TextAlignmentOptions.Right);
            row.Check = Picture("chosen", outer.transform, check, edge, new Vector2(ListWidth * 0.5f - 230f, 0f), new Vector2(60f, 60f));
            rows.Add(row);
            y += RowHeight + RowGap;
        }
        listContent.sizeDelta = new Vector2(ListWidth, y);
        emptyText = Label("empty", listViewport, RlAgentCatalog.LoadError ?? "no bots available", 48f, Color.white, Vector2.zero, new Vector2(ListWidth - 80f, 600f));
        emptyText.gameObject.SetActive(bots.Count == 0);

        // Right: the highlighted bot.
        float width = PanelWidth * 0.5f - 60f - DetailLeft;
        detailName = Label("name", p, "", 96f, Color.white, new Vector2(DetailLeft + width * 0.5f, 690f), new Vector2(width, 130f), TextAlignmentOptions.Left);
        detailName.enableWordWrapping = false;
        var chip = Box("kind", p, new Vector2(DetailLeft + 170f, 570f), new Vector2(340f, 84f), edge, edge, 42f, 1f);
        kindChip = chip.outer;
        detailKind = Label("text", kindChip.transform, "", 44f, Color.white, Vector2.zero, new Vector2(320f, 80f));
        detailRating = Label("rating", p, "", 60f, Color.white, new Vector2(DetailLeft + width * 0.5f + 380f, 570f), new Vector2(width - 380f, 90f), TextAlignmentOptions.Left);
        detailDescription = Label("description", p, "", 42f, Color.white, new Vector2(DetailLeft + width * 0.5f, 380f), new Vector2(width, 250f), TextAlignmentOptions.TopLeft);
        detailDescription.lineSpacing = -8f;
        kitHeader = Label("kit", p, "kit", 70f, edge, new Vector2(DetailLeft + 100f, 180f), new Vector2(200f, 100f), TextAlignmentOptions.Left);
        kitOptionText = Label("option", p, "", 48f, Color.white, new Vector2(DetailLeft + width * 0.5f + 100f, 180f), new Vector2(width - 200f, 100f), TextAlignmentOptions.Left);
        for (int i = 0; i < 3; i++) {
            float x = DetailCenter + (i - 1) * 330f;
            kitRings[i] = Picture("ring" + i, p, ring, edge, new Vector2(x, -20f), new Vector2(250f, 266f));
            kitIcons[i] = Picture("icon", kitRings[i].transform, null, Color.white, new Vector2(0.7f, -5.5f), new Vector2(398f, 381f));
            kitNames[i] = Label("name" + i, p, "", 42f, Color.white, new Vector2(x, -205f), new Vector2(320f, 70f));
            kitNames[i].enableWordWrapping = false;
        }
        var left = Picture("left", p, arrow, edge, new Vector2(DetailCenter - 580f, -20f), new Vector2(100f, 198f));
        var right = Picture("right", p, arrow, edge, new Vector2(DetailCenter + 580f, -20f), new Vector2(100f, 198f));
        right.rectTransform.localScale = new Vector3(-1f, 1f, 1f);
        OnClick(left, () => StepKit(-1));
        OnClick(right, () => StepKit(1));
        kitLeft = left.rectTransform; kitRight = right.rectTransform;
        for (int i = 0; i < GridCells; i++) {
            int column = i % GridColumns, line = i / GridColumns;
            var (outer, inner) = Box("option" + i, p, new Vector2(DetailCenter + (column - 2) * 312f, -330f - line * 112f), new Vector2(296f, 100f), edge, deep, 40f, 6f, true);
            var cell = new KitCell { Rect = outer.rectTransform, Outer = outer, Inner = inner };
            for (int k = 0; k < 3; k++) cell.Icons[k] = Picture("icon" + k, outer.transform, null, Color.white, new Vector2((k - 1) * 88f, 0f), new Vector2(118f, 113f));
            int option = i;
            OnClick(outer, () => SetKitOption(option, true));
            cells.Add(cell);
        }
        kitNote = Label("note", p, "", 44f, Color.white, new Vector2(DetailCenter, -400f), new Vector2(width, 160f));
        errorText = Label("error", p, "", 42f, orange, new Vector2(DetailLeft + width * 0.5f, -770f), new Vector2(width, 110f));

        // Footer: what the buttons do, and clickable too.
        hintKeys = new TextMeshProUGUI[3]; hintActions = new TextMeshProUGUI[3]; hintKeyFills = new Image[3];
        string[] actions = { "choose", "back", "group" };
        UnityEngine.Events.UnityAction[] clicks = { () => Confirm(), () => Close(false), () => JumpGroup(1) };
        for (int i = 0; i < 3; i++) {
            var (hint, hintFill) = Box("hint" + i, p, new Vector2((i - 1) * 640f, -872f), new Vector2(560f, 100f), edge, deep, 50f, 6f, true);
            OnClick(hint, clicks[i]);
            var key = Box("key", hint.transform, new Vector2(-150f, 0f), new Vector2(200f, 76f), Color.white, Color.white, 38f, 1f);
            hintKeyFills[i] = key.outer;
            hintKeys[i] = Label("text", key.outer.transform, "", 42f, edge, Vector2.zero, new Vector2(200f, 76f));
            hintActions[i] = Label("action", hint.transform, actions[i], 48f, Color.white, new Vector2(110f, 0f), new Vector2(300f, 90f), TextAlignmentOptions.Left);
        }
        canvasObject.SetActive(false);
    }

    private void RefreshHints()
    {
        string[] keys = usingGamepad ? new[] { "a", "b", "lb rb" } : new[] { "enter", "esc", "q e" };
        for (int i = 0; i < 3; i++) hintKeys[i].text = keys[i];
    }

    private int Options(RlAgentCatalog.Bot bot) => bot.Kits.Count + (bot.CanRandomize ? 1 : 0);

    private int KitOf(RlAgentCatalog.Bot bot, int option) => bot.CanRandomize ? option - 1 : option;

    private void SelectBot(int index, bool sound)
    {
        if (bots.Count == 0) return;
        index = Mathf.Clamp(index, 0, bots.Count - 1);
        if (sound && index != cursor) AudioManager.Get()?.Play("navigateMenu");
        cursor = index;
        var bot = bots[cursor];
        int kit = bot == RlAgentCatalog.Chosen ? RlAgentCatalog.ChosenKit : RlAgentCatalog.RandomKit;
        kitOption = bot.CanRandomize ? kit + 1 : Mathf.Max(0, kit);
        errorText.text = "";
        RefreshDetails(true);
        KeepVisible();
    }

    private void SetKitOption(int option, bool sound)
    {
        if (bots.Count == 0) return;
        var bot = bots[cursor];
        if (option < 0 || option >= Options(bot)) return;
        if (sound && option != kitOption) AudioManager.Get()?.Play("navigateMenu2");
        kitOption = option;
        RefreshDetails(true);
    }

    private void StepKit(int step)
    {
        if (bots.Count == 0) return;
        int count = Options(bots[cursor]);
        if (count > 1) SetKitOption((kitOption + step + count) % count, true);
    }

    private void JumpGroup(int step)
    {
        if (bots.Count == 0) return;
        int GroupStart(int i)
        {
            while (i > 0 && bots[i - 1].Group == bots[i].Group) i--;
            return i;
        }
        if (step > 0) {
            int next = cursor;
            while (next < bots.Count && bots[next].Group == bots[cursor].Group) next++;
            SelectBot(next < bots.Count ? next : 0, true);
        } else {
            // To the group's first bot, or from there to the previous group's.
            int start = GroupStart(cursor);
            SelectBot(start != cursor ? start : GroupStart((start - 1 + bots.Count) % bots.Count), true);
        }
    }

    private void RefreshDetails(bool pop)
    {
        for (int i = 0; i < rows.Count; i++) {
            var row = rows[i];
            if (row.Bot < 0) continue;
            bool selected = row.Bot == cursor, hovered = i == hoverRow;
            row.Outer.color = selected ? orange : edge;
            row.Inner.color = selected ? Color.white : hovered ? light : blue;
            row.Name.color = selected ? edge : Color.white;
            row.Rating.color = selected ? orange : edge;
            row.Check.enabled = bots[row.Bot] == RlAgentCatalog.Chosen;
            row.Check.color = selected ? orange : edge;
        }
        if (bots.Count == 0) {
            detailName.text = ""; detailKind.text = ""; kindChip.enabled = false; detailRating.text = ""; detailDescription.text = "";
            kitHeader.text = ""; kitOptionText.text = ""; kitNote.text = "";
            foreach (var r in kitRings) r.gameObject.SetActive(false);
            foreach (var n in kitNames) n.text = "";
            kitLeft.gameObject.SetActive(false); kitRight.gameObject.SetActive(false);
            foreach (var cell in cells) cell.Rect.gameObject.SetActive(false);
            return;
        }
        var bot = bots[cursor];
        detailName.text = bot.Title.ToLower();
        detailKind.text = KindText(bot);
        kindChip.rectTransform.sizeDelta = new Vector2(detailKind.GetPreferredValues(detailKind.text).x + 70f, 84f);
        detailKind.rectTransform.sizeDelta = kindChip.rectTransform.sizeDelta;
        kindChip.rectTransform.anchoredPosition = new Vector2(DetailLeft + kindChip.rectTransform.sizeDelta.x * 0.5f, 570f);
        detailRating.rectTransform.anchoredPosition = new Vector2(DetailLeft + kindChip.rectTransform.sizeDelta.x + 40f + detailRating.rectTransform.sizeDelta.x * 0.5f, 570f);
        if (bot.Rating.HasValue)
            detailRating.text = Mathf.RoundToInt(bot.Rating.Value) + " <size=70%>elo</size>"
                + (bot.Interval != null ? "  <size=70%><color=#0f5582>" + bot.Interval[0] + " - " + bot.Interval[1] + "</color></size>" : "");
        else detailRating.text = "<size=70%><color=#0f5582>no league rating</color></size>";
        detailDescription.text = bot.Description ?? "";

        int options = Options(bot);
        bool hasKits = bot.Kits.Count > 0;
        kitHeader.text = "kit";
        int kit = hasKits ? KitOf(bot, kitOption) : 0;
        int[] shown = !hasKits ? null : kit == RlAgentCatalog.RandomKit ? new[] { 1, 1, 1 } : bot.Kits[kit];
        for (int i = 0; i < 3; i++) {
            kitRings[i].gameObject.SetActive(hasKits);
            if (!hasKits) { kitNames[i].text = ""; continue; }
            kitIcons[i].sprite = Icon(shown[i]);
            kitNames[i].text = kit != RlAgentCatalog.RandomKit ? AbilityName(shown[i]) : i == 1 ? AbilityName(1) : "";
            if (pop) kitIcons[i].transform.localScale = Vector3.one * 1.15f;
        }
        kitLeft.gameObject.SetActive(options > 1);
        kitRight.gameObject.SetActive(options > 1);
        if (!hasKits) kitOptionText.text = "plays the kit it trained on";
        else if (kit == RlAgentCatalog.RandomKit) kitOptionText.text = "random: a new kit from its " + bot.Kits.Count + " each round";
        else {
            string key = RlAgentCatalog.KitKey(bot.Kits[kit]);
            string rating = bot.KitRatings.TryGetValue(key, out float value) ? "  <color=#0f5582>~" + Mathf.RoundToInt(value) + " elo on this kit</color>" : "";
            kitOptionText.text = options > 1 ? (kitOption + (bot.CanRandomize ? 0 : 1)) + " of " + bot.Kits.Count + rating : "its only kit" + rating;
        }
        // The kit grid: Random first, then every kit of the set.
        bool grid = options > 1;
        for (int i = 0; i < cells.Count; i++) {
            var cell = cells[i];
            bool used = grid && i < options;
            cell.Rect.gameObject.SetActive(used);
            if (!used) continue;
            int cellKit = KitOf(bot, i);
            for (int k = 0; k < 3; k++) {
                cell.Icons[k].sprite = Icon(cellKit == RlAgentCatalog.RandomKit ? 1 : bot.Kits[cellKit][k]);
                cell.Icons[k].enabled = cellKit != RlAgentCatalog.RandomKit || k == 1;
            }
            bool selected = i == kitOption;
            cell.Outer.color = selected ? orange : edge;
            cell.Inner.color = selected ? Color.white : deep;
        }
        if (options > GridCells) kitNote.text = "use the arrows for all " + bot.Kits.Count + " kits";
        else if (grid) kitNote.text = "";
        else if (bot.Kind == "scripted") kitNote.text = "scripted bots are built for dash / bow / platform only";
        else if (bot.Kind == "specialist") kitNote.text = "a specialist plays only the kit it was trained on";
        else if (bot.Kind == "roster") kitNote.text = "every roster member plays this kit";
        else kitNote.text = "";
        kitNote.rectTransform.anchoredPosition = new Vector2(DetailCenter, grid ? -760f : -400f);
    }

    // Scroll the list so the highlighted row and its group header are in view.
    private void KeepVisible()
    {
        Row row = rows.Find(r => r.Bot == cursor);
        if (row == null) return;
        float view = ListHeight, top = row.Top, bottom = row.Top + row.Height;
        int index = rows.IndexOf(row);
        if (index > 0 && rows[index - 1].Bot < 0) top = rows[index - 1].Top;
        if (top - 20f < scrollTarget) scrollTarget = top - 20f;
        if (bottom + 40f > scrollTarget + view) scrollTarget = bottom + 40f - view;
        ClampScroll();
    }

    private void ClampScroll() => scrollTarget = Mathf.Clamp(scrollTarget, 0f, Mathf.Max(0f, listContent.sizeDelta.y - ListHeight));

    private void ClickRow(int index)
    {
        if (!open || index < 0 || index >= rows.Count || rows[index].Bot < 0) return;
        bool again = index == lastClickRow && Time.unscaledTime - lastClickTime < 0.4f;
        lastClickRow = index; lastClickTime = Time.unscaledTime;
        if (rows[index].Bot == cursor && again) { Confirm(); return; }
        SelectBot(rows[index].Bot, true);
    }

    public void Open()
    {
        if (open || Time.frameCount == toggledFrame) return;
        open = true;
        toggledFrame = Time.frameCount;
        EventSystem.current?.SetSelectedGameObject(null);
        overlay.gameObject.SetActive(true);
        overlay.alpha = 0f;
        openTime = 0f;
        heldVertical = heldHorizontal = 0;
        hoverRow = -1;
        RefreshHints();
        cursor = Mathf.Max(0, bots.IndexOf(RlAgentCatalog.Chosen));
        SelectBot(cursor, false);
        scroll = scrollTarget;
        AudioManager.Get()?.Play("accept");
    }

    private void Close(bool chosen)
    {
        if (!open || Time.frameCount == toggledFrame) return;
        open = false;
        toggledFrame = Time.frameCount;
        overlay.gameObject.SetActive(false);
        if (!chosen) AudioManager.Get()?.Play("decline");
        RefreshSeat();
    }

    private void Confirm()
    {
        if (!open || bots.Count == 0) return;
        var bot = bots[cursor];
        // The match opens the bot when it starts: check now that it loads.
        if (!RlAgent.Probe(bot.Path, out _, out string error)) {
            errorText.text = "can't load " + bot.Id + ": " + error;
            Debug.LogError("RlAgentPicker: " + bot.Path + ": " + error);
            AudioManager.Get()?.Play("decline");
            return;
        }
        RlAgentCatalog.Choose(bot, bot.Kits.Count > 0 ? KitOf(bot, kitOption) : 0);
        seatShown = 0; seatCycle = 0f;
        AudioManager.Get()?.Play("accept");
        Close(true);
        if (RlAgentCatalog.Chosen.Kits.Count > 0)
            ShowSeatKit(RlAgentCatalog.ChosenKit == RlAgentCatalog.RandomKit ? RlAgentCatalog.Chosen.Kits[0] : RlAgentCatalog.Chosen.Kits[RlAgentCatalog.ChosenKit], true);
    }

    // ---- input ----

    private void TrackDevice()
    {
        var keyboard = Keyboard.current;
        var mouse = Mouse.current;
        if ((keyboard != null && keyboard.anyKey.wasPressedThisFrame) || (mouse != null && (mouse.leftButton.wasPressedThisFrame || mouse.delta.ReadValue().sqrMagnitude > 25f))) {
            if (usingGamepad) { usingGamepad = false; RefreshHints(); }
            return;
        }
        foreach (var pad in Gamepad.all) {
            if (pad.buttonSouth.wasPressedThisFrame || pad.buttonEast.wasPressedThisFrame || pad.buttonNorth.wasPressedThisFrame || pad.buttonWest.wasPressedThisFrame
                || pad.startButton.wasPressedThisFrame || pad.dpad.ReadValue().sqrMagnitude > 0.25f || pad.leftStick.ReadValue().sqrMagnitude > 0.25f) {
                if (!usingGamepad) { usingGamepad = true; RefreshHints(); }
                return;
            }
        }
    }

    private static bool AnyPad(System.Func<Gamepad, bool> test)
    {
        foreach (var pad in Gamepad.all) if (test(pad)) return true;
        return false;
    }

    private static int Held(bool negative, bool positive) => negative == positive ? 0 : negative ? -1 : 1;

    // Steps while a direction is held; a tap shorter than a frame (its press and
    // release between two polls) still counts once.
    private static int Repeat(int held, int tapped, ref int last, ref float timer, float rate)
    {
        if (tapped != 0) { last = held == tapped ? held : 0; timer = RepeatDelay; return tapped; }
        if (held == 0) { last = 0; return 0; }
        if (held != last) { last = held; timer = RepeatDelay; return held; }
        timer -= Time.unscaledDeltaTime;
        if (timer > 0f) return 0;
        timer += rate;
        return held;
    }

    private void Update()
    {
        float dt = Time.unscaledDeltaTime;
        TrackDevice();
        UpdateSeat(dt);
        var keyboard = Keyboard.current;
        if (!open) {
            bool toggle = (keyboard != null && keyboard.tabKey.wasPressedThisFrame) || AnyPad(pad => pad.buttonNorth.wasPressedThisFrame);
            if (toggle && Time.frameCount != toggledFrame) Open();
            return;
        }
        openTime += dt;
        overlay.alpha = Mathf.Clamp01(openTime / 0.12f);
        panel.localScale = Vector3.one * (Mathf.Lerp(0.96f, 1f, overlay.alpha) * PanelFit());
        if (Time.frameCount == toggledFrame) return;   // the press that opened it

        bool Pressed(Key key) => keyboard != null && keyboard[key].wasPressedThisFrame;
        bool Down(Key key) => keyboard != null && keyboard[key].isPressed;
        if (Pressed(Key.Escape) || Pressed(Key.Backspace) || Pressed(Key.Tab)
            || AnyPad(pad => pad.buttonEast.wasPressedThisFrame || pad.buttonNorth.wasPressedThisFrame || pad.selectButton.wasPressedThisFrame)) {
            Close(false);
            return;
        }
        if (Pressed(Key.Enter) || Pressed(Key.NumpadEnter) || Pressed(Key.Space)
            || AnyPad(pad => pad.buttonSouth.wasPressedThisFrame || pad.startButton.wasPressedThisFrame)) {
            Confirm();
            return;
        }
        if (Pressed(Key.Q) || Pressed(Key.PageUp) || AnyPad(pad => pad.leftShoulder.wasPressedThisFrame)) JumpGroup(-1);
        if (Pressed(Key.E) || Pressed(Key.PageDown) || AnyPad(pad => pad.rightShoulder.wasPressedThisFrame)) JumpGroup(1);
        if (Pressed(Key.Home)) SelectBot(0, true);
        if (Pressed(Key.End)) SelectBot(bots.Count - 1, true);

        bool up = Down(Key.W) || Down(Key.UpArrow) || AnyPad(pad => pad.dpad.up.isPressed || pad.leftStick.ReadValue().y > 0.55f);
        bool down = Down(Key.S) || Down(Key.DownArrow) || AnyPad(pad => pad.dpad.down.isPressed || pad.leftStick.ReadValue().y < -0.55f);
        bool leftHeld = Down(Key.A) || Down(Key.LeftArrow) || AnyPad(pad => pad.dpad.left.isPressed || pad.leftStick.ReadValue().x < -0.55f);
        bool rightHeld = Down(Key.D) || Down(Key.RightArrow) || AnyPad(pad => pad.dpad.right.isPressed || pad.leftStick.ReadValue().x > 0.55f);
        int tapVertical = Held(Pressed(Key.W) || Pressed(Key.UpArrow) || AnyPad(pad => pad.dpad.up.wasPressedThisFrame),
                               Pressed(Key.S) || Pressed(Key.DownArrow) || AnyPad(pad => pad.dpad.down.wasPressedThisFrame));
        int tapHorizontal = Held(Pressed(Key.A) || Pressed(Key.LeftArrow) || AnyPad(pad => pad.dpad.left.wasPressedThisFrame),
                                 Pressed(Key.D) || Pressed(Key.RightArrow) || AnyPad(pad => pad.dpad.right.wasPressedThisFrame));
        int vertical = Repeat(Held(up, down), tapVertical, ref heldVertical, ref timerVertical, RepeatRateVertical);
        if (vertical != 0) SelectBot(cursor + vertical, true);
        int horizontal = Repeat(Held(leftHeld, rightHeld), tapHorizontal, ref heldHorizontal, ref timerHorizontal, RepeatRateHorizontal);
        if (horizontal != 0) StepKit(horizontal);

        // Mouse: wheel over the list scrolls it; hovering lights a row.
        var mouse = Mouse.current;
        if (mouse != null) {
            Vector2 point = mouse.position.ReadValue();
            bool overList = RectTransformUtility.RectangleContainsScreenPoint(listViewport, point, null);
            float wheel = mouse.scroll.ReadValue().y;
            if (overList && wheel != 0f) {
                scrollTarget -= wheel * (Mathf.Abs(wheel) >= 10f ? 1.6f : 60f);
                ClampScroll();
            }
            int hover = -1;
            if (overList)
                for (int i = 0; i < rows.Count; i++)
                    if (rows[i].Bot >= 0 && RectTransformUtility.RectangleContainsScreenPoint(rows[i].Rect, point, null)) { hover = i; break; }
            if (hover != hoverRow) { hoverRow = hover; RefreshDetails(false); }
        }
        scroll = Mathf.Lerp(scroll, scrollTarget, 1f - Mathf.Exp(-18f * dt));
        listContent.anchoredPosition = new Vector2(0f, scroll);
        float ease = 1f - Mathf.Exp(-12f * dt);
        foreach (var icon in kitIcons) icon.transform.localScale = Vector3.Lerp(icon.transform.localScale, Vector3.one, ease);
    }

    // Narrow screens (wider than 16:9 here means a shorter canvas) shrink the panel to fit.
    private float PanelFit()
    {
        var area = ((RectTransform)overlay.transform).rect;
        return Mathf.Min(1f, (area.height - 60f) / PanelHeight, (area.width - 60f) / PanelWidth);
    }
}
