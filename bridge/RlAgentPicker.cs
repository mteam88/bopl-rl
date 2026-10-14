using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem;
using UnityEngine.UI;

// The opponent picker on the local character select (RlAgent, RlAgentCatalog;
// on whenever BOPL_AGENT_CATALOG is set). Every bot is a "seat": a select
// column like a joined player's in a box nobody joined (the rightmost free
// ones), showing the bot, its rating, the kit it will play (cycling through
// example kits when the kit is Random) and its slime in the colour it will
// get, the column in its team's colours as a joined box shows its player's
// team. Up to five bots; each joins the match as its own player, on the team
// its seat chose: any of the six the boxes offer (TeamSelector), so a human's
// team makes it an ally ("ally" over the column) and another bot's teams them
// up. A new bot takes the latest team (the highest index: teal, then yellow,
// pink, green, orange, blue) no joined player and no other bot is on
// (RlAgentCatalog.LaterTeam). While bots are seated the start button also
// needs two teams among the ready players and the bots (TeamsSplit).
//
// It reads only buttons the character select leaves unbound, so it never
// takes a press from a player choosing their own abilities (the boxes read A,
// B, X, Y, Start, Select, the d-pad and the left stick; the match binds A, B,
// X, Y and both sticks' directions, never their presses):
//   L3 (left stick press)       add a bot (a copy of the highlighted one)
//   hold L3                     remove the highlighted bot
//   R3 (right stick press)      edit the highlighted bot (again: done)
//   right stick left / right    highlight another bot
// and in the editor, a panel over the boxes:
//   right stick left / right    field: bot > team > kit > slot 1 > slot 2 > slot 3
//   right stick up / down       change it (held: repeats, faster the longer)
//   R3 done, L3 add another bot, hold L3 remove this one
// The editor lists the bots grouped as the catalog does (league, exploiters,
// specialists, scripted, rosters) with rating, interval and description; the
// kit is Random, one of the bot's trained kits, or custom, every slot any
// ability the bot supports (RlAgentCatalog.Bot.Pool), with icons. The d-pad
// and left stick also move in the editor (the boxes ignore them while it is
// open), but nothing needs them. Keyboard and mouse: Tab edits, Ins adds, Del
// removes, arrows or WASD move, Enter is done, Esc cancels the edit, Q / E jump
// a group; a click on a column edits it, and the rows, kits, abilities, arrows
// and hints are clickable. While a box captures a key binding (KeyBindButton,
// which takes any pad button, L3 and R3 included) the picker ignores input.
// Every change is saved (RlAgentCatalog.SetSeats) for the next match and
// launch; a bot is opened in the agent shim once first to check it loads.
// Everything is built at runtime from the select box's own sprites and font,
// so it needs no scene or prefab changes. While the editor is open (and on any
// frame the picker acted) BlocksInput keeps the select boxes, the click-to-join
// and the back-to-menu check from also acting on the same presses.
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

    private sealed class Cell
    {
        public RectTransform Rect;
        public Image Outer, Inner;
        public Image[] Icons;
        public int Value;             // kit option, or ability
    }

    private sealed class SeatView
    {
        public RlAgentCatalog.Seat Seat;
        public int Box = -1;
        public RectTransform Root;
        public Image Border;
        public TextMeshProUGUI Header, Title, Info, Team, Kit, Badge, Key, KeyAction;
        public readonly List<Graphic> Fills = new List<Graphic>(), Edges = new List<Graphic>();   // in the team's fill / border colour
        public readonly Image[] Icons = new Image[3];
        public Image Slime;
        public float Cycle;
        public int Shown;
    }

    private const float PanelWidth = 2800f, PanelHeight = 1880f;
    private const float ListWidth = 1000f, ListHeight = 1500f, ListX = -860f, ListY = -10f;
    private const float RowHeight = 112f, HeaderHeight = 92f, RowGap = 12f;
    private const float DetailLeft = -250f, DetailCenter = 560f, KitRowY = 180f, TeamRowY = 300f, RingY = -20f;
    private const float RepeatDelay = 0.35f, RepeatRateVertical = 0.11f, RepeatRateHorizontal = 0.22f;
    private const float HoldToRemove = 0.6f, StickPress = 0.55f;
    private const int GridColumns = 5, GridCells = 20, AbilityColumns = 12, AbilityCells = 48;
    private const int FieldBot = 0, FieldTeam = 1, FieldKit = 2, FieldSlot = 3;   // slots: FieldSlot + 0..2

    private static RlAgentPicker instance;
    private bool open;
    private int toggledFrame = -1;

    // The select boxes, the click-to-join and the back-to-menu check skip their
    // input while the editor has it, and on any frame the picker acted.
    public static bool BlocksInput => instance != null && (instance.open || instance.toggledFrame == Time.frameCount);

    // CharacterSelectHandler's start button: with bots seated it counts their teams.
    public static bool Seated => instance != null && instance.seats.Count > 0;

    // Two teams at least among the ready players and the bots (else the round
    // would have nobody to fight).
    public static bool TeamsSplit()
    {
        if (instance == null) return true;
        var teams = new HashSet<int>();
        foreach (var (_, team, ready) in instance.HumanTeams()) if (ready) teams.Add(team);
        foreach (var view in instance.seats) teams.Add(view.Seat.Team);
        return teams.Count > 1;
    }

    private CharacterSelectHandler handler;
    private PlayerColors playerColors;
    private TeamColors teamColors;
    private TeamSelector[] boxTeams;   // per select box: its player's team
    private NamedSpriteList abilities;
    private TMP_FontAsset font;
    private Color blue, edge, orange, deep, light;
    private Sprite rounded, columnFill, columnBorder, ring, disc, arrow, check, platform, slime;
    private List<RlAgentCatalog.Bot> bots;
    private readonly HashSet<string> loads = new HashSet<string>();    // specs the shim opened
    private readonly System.Random cycleDraw = new System.Random();

    // Seats.
    private CharacterSelectBox template;
    private readonly List<SeatView> seats = new List<SeatView>();
    private int focus;
    private float l3Since = -1f;   // L3 held since (unscaled time); -1: not held
    private bool l3Removed;

    // Hint strip under the boxes.
    private CanvasGroup strip;
    private RectTransform stripRow;
    private readonly List<(Image box, TextMeshProUGUI key, TextMeshProUGUI action)> stripHints = new List<(Image, TextMeshProUGUI, TextMeshProUGUI)>();
    private TextMeshProUGUI stripNote;
    private GameObject teamWarning;   // over the strip while everyone is on one team

    // Editor.
    private CanvasGroup overlay;
    private RectTransform panel, listViewport, listContent;
    private Image listFrame, kitRow, teamRow;
    private readonly (Image outer, Image inner, TextMeshProUGUI label)[] teamCells = new (Image, Image, TextMeshProUGUI)[RlAgentCatalog.Teams];
    private TextMeshProUGUI teamHeader, teamStatus;
    private float builtAt;            // Time.unscaledTime the picker was built (the scene opened)
    private readonly List<Row> rows = new List<Row>();
    private TextMeshProUGUI title, detailName, detailKind, detailRating, detailDescription, kitHeader, kitOptionText, kitNote, errorText, emptyText;
    private Image kindChip;
    private readonly Image[] kitRings = new Image[3], kitIcons = new Image[3], slotUp = new Image[3], slotDown = new Image[3];
    private readonly TextMeshProUGUI[] kitNames = new TextMeshProUGUI[3];
    private Image kitUp, kitDown;
    private readonly List<Cell> kitCells = new List<Cell>(), abilityCells = new List<Cell>();
    private TextMeshProUGUI[] hintKeys, hintActions;
    private Image[] hintBoxes;

    private int editing = -1, cursor, field, kitOption, hoverRow = -1;
    private RlAgentCatalog.Seat draft;
    private int[] lastCustom;
    private float scroll, scrollTarget, openTime, lastClickTime;
    private int lastClickRow = -1;
    private bool usingGamepad;
    private int heldVertical, heldHorizontal, heldFocus;
    private float timerVertical, timerHorizontal, timerFocus, verticalFor;

    // CharacterSelectHandler.Start: the seats take the rightmost free boxes.
    public static void Attach(CharacterSelectHandler handler, CharacterSelectBox template, PlayerColors colors)
    {
        if (!RlAgentCatalog.Present || SteamManager.instance == null || template == null) return;
        RlAgentCatalog.EnsureLoaded();
        handler.gameObject.AddComponent<RlAgentPicker>().Build(handler, template, colors);
    }

    private void OnDestroy()
    {
        if (instance == this) instance = null;
    }

    private void Build(CharacterSelectHandler handler, CharacterSelectBox template, PlayerColors colors)
    {
        instance = this;
        this.handler = handler;
        this.template = template;
        playerColors = colors;
        abilities = SteamManager.instance.abilityIcons;
        bots = RlAgentCatalog.Bots;
        boxTeams = new TeamSelector[handler.characterSelectBoxes.Length];
        for (int i = 0; i < boxTeams.Length; i++) boxTeams[i] = handler.characterSelectBoxes[i].GetComponentInChildren<TeamSelector>(true);
        teamColors = template.GetComponentInChildren<TeamSelector>(true)?.teams;
        if (teamColors == null || teamColors.teamColors.Length < RlAgentCatalog.Teams) {
            Debug.LogWarning("RlAgentPicker: no team colours in the select box");
            teamColors = null;
        }
        blue = handler.blue; orange = handler.orange;
        edge = new Color(0.082f, 0.333f, 0.509f);
        deep = new Color(0.09f, 0.46f, 0.76f);
        light = new Color(0.36f, 0.74f, 0.98f);
        font = template.readyPlayerNameText.font;
        // The select box's sprites, by name (the hierarchy names are not stable).
        var sprites = new Dictionary<string, Sprite>();
        foreach (var image in template.GetComponentsInChildren<Image>(true))
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

        var root = template.GetComponentInParent<Canvas>().rootCanvas;
        builtAt = Time.unscaledTime;
        BuildStrip(root);
        BuildOverlay(root);
        // The remembered seats, as many as there are free boxes (players who
        // came back from a match took theirs first).
        var saved = new List<RlAgentCatalog.Seat>(RlAgentCatalog.Seats);
        int room = Mathf.Max(0, FreeBoxes() - (JoinedPlayers() > 0 ? 0 : 1));
        if (saved.Count > room) {
            Debug.LogWarning("RlAgentPicker: " + saved.Count + " bots for " + room + " free boxes; seating " + room);
            saved.RemoveRange(room, saved.Count - room);
            RlAgentCatalog.SetSeats(saved, false);
        }
        // Seats from before teams (or the launcher's) get the default, in order.
        foreach (var seat in saved) {
            if (!RlAgentCatalog.ValidTeam(seat.Team)) seat.Team = DefaultTeam();
            seats.Add(BuildSeat(seat));
        }
        RlAgentCatalog.SetSeats(seats.ConvertAll(view => view.Seat), false);
        Layout();
        foreach (var view in seats) RefreshSeat(view);
        RefreshStrip();
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

    // The box's arrow (it points left) turned to point up (-1) or down (1).
    private Image Arrow(string name, Transform parent, Vector2 position, float size, int down)
    {
        var image = Picture(name, parent, arrow, edge, position, new Vector2(size * 0.5f, size), true);
        image.rectTransform.localRotation = Quaternion.Euler(0f, 0f, down > 0 ? 90f : -90f);
        return image;
    }

    // A canvas of its own at the scene root: under the select canvas it would
    // be a nested canvas, sized by its parent instead of the screen.
    private static RectTransform OverlayCanvas(string name, Canvas root, int order, out CanvasGroup group)
    {
        var canvasObject = new GameObject(name, typeof(RectTransform), typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster), typeof(CanvasGroup));
        canvasObject.transform.SetParent(null, false);
        var canvas = canvasObject.GetComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvas.sortingOrder = order;
        var scaler = canvasObject.GetComponent<CanvasScaler>();
        var rootScaler = root.GetComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = rootScaler != null ? rootScaler.referenceResolution : new Vector2(4096f, 2160f);
        scaler.screenMatchMode = CanvasScaler.ScreenMatchMode.MatchWidthOrHeight;
        scaler.matchWidthOrHeight = rootScaler != null ? rootScaler.matchWidthOrHeight : 0f;
        scaler.referencePixelsPerUnit = 100f;
        group = canvasObject.GetComponent<CanvasGroup>();
        return canvasObject.GetComponent<RectTransform>();
    }

    private Sprite Icon(int ability) => ability >= 0 && ability < abilities.sprites.Count ? abilities.sprites[ability].sprite : null;

    private string AbilityName(int ability)
    {
        if (ability >= 0 && ability < abilities.sprites.Count && LocalizedText.localizationTable != null)
            return LocalizedText.localizationTable.GetText(abilities.sprites[ability].name.ToLower(), Settings.Get().Language);
        return RlAgentCatalog.AbilityNames.TryGetValue(ability, out string name) ? name : ability.ToString();
    }

    private string KitText(int[] kit) => AbilityName(kit[0]) + " / " + AbilityName(kit[1]) + " / " + AbilityName(kit[2]);

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

    // ---- teams ----

    private Color TeamFill(int team) => teamColors != null && RlAgentCatalog.ValidTeam(team) ? teamColors.teamColors[team].fill : blue;
    private Color TeamBorder(int team) => teamColors != null && RlAgentCatalog.ValidTeam(team) ? teamColors.teamColors[team].border : edge;
    private static string TeamName(int team) => RlAgentCatalog.ValidTeam(team) ? RlAgentCatalog.TeamNames[team] : "no";

    // The joined players' boxes: (box, team, ready). A box's team is its
    // TeamSelector's, which it hands the player when it readies.
    private List<(int box, int team, bool ready)> HumanTeams()
    {
        var humans = new List<(int, int, bool)>();
        var boxes = handler.characterSelectBoxes;
        for (int i = 0; i < boxes.Length; i++) {
            var box = boxes[i];
            if (Mine(i) || !box.gameObject.activeSelf || box.menuState == CharSelectMenu.join) continue;
            int team = boxTeams[i] != null ? boxTeams[i].selectedIndex : box.playerInit.team;
            humans.Add((i, team, box.menuState == CharSelectMenu.ready || box.menuState == CharSelectMenu.trueReady));
        }
        return humans;
    }

    // A human's name on the screen: "you" when only one joined, else the box ("p2").
    private static string HumanName(int box, int humans) => humans == 1 ? "you" : "p" + (box + 1);

    // The latest team no joined player and no seated bot (but skip) is on.
    private int DefaultTeam(SeatView skip = null)
    {
        var taken = new List<int>();
        foreach (var (_, team, _) in HumanTeams()) taken.Add(team);
        foreach (var view in seats) if (view != skip) taken.Add(view.Seat.Team);
        return RlAgentCatalog.LaterTeam(taken);
    }

    // Who else is on team: the humans by name, the other bots as "bot k".
    private (List<string> humans, List<string> bots) TeamMates(int team, int self, bool shortNames)
    {
        var humans = new List<string>(); var others = new List<string>();
        var joined = HumanTeams();
        foreach (var (box, t, _) in joined) if (t == team) humans.Add(HumanName(box, joined.Count));
        for (int k = 0; k < seats.Count; k++)
            if (k != self && seats[k].Seat.Team == team) others.Add((shortNames ? "b" : "bot ") + (k + 1));
        return (humans, others);
    }

    // The editor's subtitle: what the draft's team means for the match and the bot.
    private string TeamNote()
    {
        if (draft == null) return "";
        if (OneTeam()) return "<color=#ffb04a>everyone is on one team: the match needs two</color>";
        var (humans, others) = TeamMates(draft.Team, editing, false);
        if (humans.Count + others.Count == 0) return "<color=#ffffff>the nets trained 1 v 1</color>";
        if (draft.Bot.Kind == "scripted") return "<color=#ffb04a>scripted bots target their teammates too</color>";
        return "<color=#ffffff>the nets trained 1 v 1: a teammate is new to them</color>";
    }

    // Everyone (the joined players and the bots, with the edit in progress) on one team.
    private bool OneTeam()
    {
        var teams = new HashSet<int>();
        var joined = HumanTeams();
        if (joined.Count == 0 || seats.Count == 0) return false;
        foreach (var (_, team, _) in joined) teams.Add(team);
        for (int k = 0; k < seats.Count; k++) teams.Add(open && k == editing && draft != null ? draft.Team : seats[k].Seat.Team);
        return teams.Count == 1;
    }

    // ---- seats ----

    private bool Mine(int box) => seats.Exists(view => view.Box == box);

    // Boxes nobody joined and no seat holds.
    private int FreeBoxes()
    {
        int free = 0;
        for (int i = 0; i < handler.characterSelectBoxes.Length; i++) if (!CharacterSelectBox.occupiedRectangles[i] && !Mine(i)) free++;
        return free;
    }

    private int JoinedPlayers()
    {
        int joined = 0;
        for (int i = 0; i < handler.characterSelectBoxes.Length; i++) if (CharacterSelectBox.occupiedRectangles[i] && !Mine(i)) joined++;
        return joined;
    }

    // Another bot needs a free box, and one stays free while nobody has joined.
    private bool CanAdd => bots.Count > 0 && seats.Count < RlAgentCatalog.MaxSeats && FreeBoxes() >= (JoinedPlayers() > 0 ? 1 : 2);

    // The seats hold the rightmost boxes nobody joined, in order; a box a seat
    // lets go of can be joined again.
    private void Layout()
    {
        var boxes = handler.characterSelectBoxes;
        var chosen = new List<int>();
        for (int i = boxes.Length - 1; i >= 0 && chosen.Count < seats.Count; i--)
            if (Mine(i) || !CharacterSelectBox.occupiedRectangles[i]) chosen.Add(i);
        chosen.Sort();
        foreach (var view in seats)
            if (view.Box >= 0 && !chosen.Contains(view.Box)) {
                boxes[view.Box].gameObject.SetActive(true);
                CharacterSelectBox.occupiedRectangles[view.Box] = false;
            }
        while (seats.Count > chosen.Count) RemoveView(seats.Count - 1);
        for (int k = 0; k < seats.Count; k++) {
            var view = seats[k];
            int index = chosen[k];
            if (!Mine(index) || view.Box != index) {
                boxes[index].gameObject.SetActive(false);
                CharacterSelectBox.occupiedRectangles[index] = true;
            }
            view.Box = index;
            var box = boxes[index].GetComponent<RectTransform>();
            view.Root.anchorMin = box.anchorMin; view.Root.anchorMax = box.anchorMax; view.Root.pivot = box.pivot;
            view.Root.anchoredPosition = box.anchoredPosition;
            view.Root.localScale = box.localScale;
            view.Root.SetSiblingIndex(box.GetSiblingIndex());
        }
        focus = Mathf.Clamp(focus, 0, Mathf.Max(0, seats.Count - 1));
    }

    private void RemoveView(int index)
    {
        var view = seats[index];
        if (view.Box >= 0 && !seats.Exists(other => other != view && other.Box == view.Box)) {
            handler.characterSelectBoxes[view.Box].gameObject.SetActive(true);
            CharacterSelectBox.occupiedRectangles[view.Box] = false;
        }
        Destroy(view.Root.gameObject);
        seats.RemoveAt(index);
    }

    private SeatView BuildSeat(RlAgentCatalog.Seat seat)
    {
        var view = new SeatView { Seat = seat };
        var box = template.GetComponent<RectTransform>();
        var root = Node("RlAgentSeat", box.parent, box.anchoredPosition, box.sizeDelta);
        view.Root = root;
        // The joined column, as the select box draws it (SELECT_COLOR, SELECT_BORDER).
        var fill = Picture("fill", root, columnFill, blue, new Vector2(-8f, 29.4f), new Vector2(607.1f, 1801.7f), true);
        fill.gameObject.AddComponent<Mask>().showMaskGraphic = true;
        OnClick(fill, () => EditSeat(view));
        view.Fills.Add(fill);
        view.Border = Picture("border", root, columnBorder, edge, new Vector2(-13.4f, 29.4f), new Vector2(640f, 1801.7f));
        Transform column = fill.transform;
        view.Header = Label("header", root, "opponent", 78f, edge, new Vector2(-4.9f, 840.5f), new Vector2(560f, 128f));
        view.Title = Label("title", column, "", 58f, Color.white, new Vector2(0f, 700f), new Vector2(540f, 130f));
        view.Info = Label("info", column, "", 48f, edge, new Vector2(0f, 598f), new Vector2(560f, 64f));
        view.Edges.Add(view.Info);
        view.Team = Label("team", column, "", 40f, Color.white, new Vector2(0f, 532f), new Vector2(560f, 60f));
        view.Team.enableWordWrapping = false;
        float[] lines = { 334.3f, 18.3f, -296.8f };
        for (int i = 0; i < 3; i++) {
            var circle = Picture("ability" + i, column, ring, edge, new Vector2(-1f, lines[i] + 4f), new Vector2(263.8f, 280.6f));
            view.Edges.Add(circle);
            view.Icons[i] = Picture("icon", circle.transform, null, Color.white, new Vector2(0.7f, -5.8f), new Vector2(419.9f, 402.4f));
        }
        view.Kit = Label("kit", column, "", 46f, Color.white, new Vector2(0f, -480f), new Vector2(570f, 120f));
        Picture("platform", column, platform, Color.white, new Vector2(0.7f, -858f), new Vector2(944.3f, 515f));
        view.Slime = Picture("slime", column, slime, Color.white, new Vector2(0.7f, -651f), new Vector2(240.7f, 165.8f));
        // A badge where the box shows its device, and one where it has the ready button.
        var top = Picture("badge", root, disc, blue, new Vector2(-8.4f, 1080.4f), new Vector2(260f, 263f));
        view.Fills.Add(top);
        view.Edges.Add(Picture("ring", top.transform, ring, edge, Vector2.zero, new Vector2(275f, 266f)));
        view.Badge = Label("bot", top.transform, "bot", 84f, Color.white, new Vector2(0f, -4f), new Vector2(240f, 120f));
        var key = Picture("edit", root, disc, blue, new Vector2(0f, -847.2f), new Vector2(236f, 239f), true);
        OnClick(key, () => EditSeat(view));
        view.Fills.Add(key);
        view.Edges.Add(Picture("ring", key.transform, ring, edge, Vector2.zero, new Vector2(250f, 241f)));
        view.Key = Label("key", key.transform, "", 72f, Color.white, new Vector2(0f, 20f), new Vector2(220f, 100f));
        view.Key.enableWordWrapping = false;
        view.Key.overflowMode = TextOverflowModes.Overflow;
        view.KeyAction = Label("action", key.transform, "edit", 34f, edge, new Vector2(0f, -50f), new Vector2(220f, 50f));
        view.Edges.Add(view.KeyAction);
        return view;
    }

    private void RefreshSeat(SeatView view)
    {
        var seat = view.Seat;
        var bot = seat.Bot;
        view.Title.text = bot.Title.ToLower();
        string rating = RatingText(bot);
        view.Info.text = KindText(bot) + (rating.Length > 0 ? "  <color=#ffffff>" + rating + "</color>" : "");
        if (seat.Kit != null) {
            view.Kit.text = KitText(seat.Kit);
            ShowSeatKit(view, seat.Kit, false);
        } else if (bot.CanRandomize) {
            view.Kit.text = "random kit\n<size=85%>" + (bot.AnyKit ? "any 3 of " + bot.Pool.Count + " abilities" : "1 of " + bot.Kits.Count + " each round") + "</size>";
            ShowSeatKit(view, SeatExample(view), false);
        } else {
            view.Kit.text = "its training kit";
            foreach (var icon in view.Icons) icon.enabled = false;
        }
    }

    // What a random seat shows: its set's kits in turn, or draws from its pool.
    private int[] SeatExample(SeatView view)
    {
        var bot = view.Seat.Bot;
        if (bot.AnyKit) return RlAgentCatalog.KitFor(view.Seat, cycleDraw);
        return bot.Kits[view.Shown % bot.Kits.Count];
    }

    private void ShowSeatKit(SeatView view, int[] kit, bool pop)
    {
        for (int i = 0; i < 3; i++) {
            view.Icons[i].enabled = true;
            view.Icons[i].sprite = Icon(kit[i]);
            if (pop) view.Icons[i].transform.localScale = Vector3.one * 1.18f;
        }
    }

    private void UpdateSeats(float dt)
    {
        float ease = 1f - Mathf.Exp(-12f * dt);
        // The bots take the first colours no joined player has, in order (RlAgent.AddPlayers).
        var taken = new List<Material>();
        foreach (var box in handler.characterSelectBoxes)
            if (box.isActiveAndEnabled && (box.menuState == CharSelectMenu.select || box.menuState == CharSelectMenu.ready || box.menuState == CharSelectMenu.abilityGrid))
                taken.Add(playerColors[box.playerInit.color].playerMaterial);
        float hold = l3Since >= 0f && !l3Removed && !open ? Mathf.Clamp01((Time.unscaledTime - l3Since - 0.15f) / (HoldToRemove - 0.15f)) : 0f;
        for (int i = 0; i < seats.Count; i++) {
            var view = seats[i];
            var bot = view.Seat.Bot;
            if (view.Seat.Kit == null && bot.CanRandomize) {
                view.Cycle += dt;
                if (view.Cycle > 1.3f) {
                    view.Cycle = 0f;
                    view.Shown++;
                    ShowSeatKit(view, SeatExample(view), true);
                }
            }
            foreach (var icon in view.Icons) icon.transform.localScale = Vector3.Lerp(icon.transform.localScale, Vector3.one, ease);
            int color = RlAgent.ColorIndex(taken, playerColors);
            taken.Add(playerColors[color].playerMaterial);
            view.Slime.material = playerColors[color].uiMaterial;
            bool focused = i == focus;
            // The team: the column in its colours, "ally" over it when a human is on it.
            int team = view.Seat.Team;
            Color teamEdge = TeamBorder(team);
            foreach (var graphic in view.Fills) graphic.color = TeamFill(team);
            foreach (var graphic in view.Edges) graphic.color = teamEdge;
            var (humans, others) = TeamMates(team, i, false);
            string role = humans.Count > 0 ? "ally" : "opponent";
            humans.AddRange(others);
            view.Team.text = TeamName(team) + " team" + (humans.Count > 0 ? "  <size=85%>with " + string.Join(", ", humans) + "</size>" : "");
            view.Header.text = seats.Count > 1 ? role + " " + (i + 1) : role;
            // The highlight is orange, white on the orange and yellow teams' columns.
            Color highlight = team == 1 || team == 4 ? Color.white : orange;
            view.Border.color = focused && seats.Count > 1 ? highlight : teamEdge;
            view.Header.color = focused && seats.Count > 1 ? highlight : teamEdge;
            view.Key.text = focused ? (usingGamepad ? "r3" : "tab") : "";
            view.Key.fontSize = usingGamepad ? 80f : 64f;
            // Holding L3 shrinks the bot it will remove.
            float scale = focused ? 1f - 0.08f * hold : 1f;
            var box = handler.characterSelectBoxes[view.Box].GetComponent<RectTransform>();
            view.Root.localScale = box.localScale * scale;
        }
        // Nobody to fight: the start button stays away (TeamsSplit), say why.
        if (teamWarning != null) teamWarning.SetActive(!open && OneTeam());
    }

    private bool Loads(RlAgentCatalog.Bot bot, out string error)
    {
        error = null;
        if (loads.Contains(bot.Path)) return true;
        // The match opens the bot when it starts: check now that it loads.
        if (!RlAgent.Probe(bot.Path, out _, out error)) {
            Debug.LogError("RlAgentPicker: " + bot.Path + ": " + error);
            return false;
        }
        loads.Add(bot.Path);
        return true;
    }

    private void Save()
    {
        RlAgentCatalog.SetSeats(seats.ConvertAll(view => view.Seat));
        RefreshStrip();
    }

    // A new bot: a copy of from, else of the highlighted one, else the default.
    private bool AddSeat(RlAgentCatalog.Seat from = null)
    {
        toggledFrame = Time.frameCount;
        if (!CanAdd) { AudioManager.Get()?.Play("decline"); Flash(seats.Count >= RlAgentCatalog.MaxSeats ? "five bots at most" : "no free box for another bot"); return false; }
        var seat = RlAgentCatalog.NewSeat(from ?? (seats.Count > 0 ? seats[focus].Seat : null));
        if (seat.Bot == null || !Loads(seat.Bot, out string error)) { AudioManager.Get()?.Play("decline"); Flash("can't load the bot"); return false; }
        // A copy keeps the bot and kit but not the team: a new bot takes the latest free one.
        seat.Team = DefaultTeam();
        var view = BuildSeat(seat);
        seats.Add(view);
        focus = seats.Count - 1;
        Layout();
        RefreshSeat(view);
        foreach (var icon in view.Icons) icon.transform.localScale = Vector3.one * 1.18f;
        Save();
        AudioManager.Get()?.Play("accept");
        return true;
    }

    private void RemoveSeat(int index)
    {
        toggledFrame = Time.frameCount;
        if (index < 0 || index >= seats.Count) return;
        RemoveView(index);
        if (focus >= seats.Count) focus = seats.Count - 1;
        if (focus < 0) focus = 0;
        Layout();
        Save();
        AudioManager.Get()?.Play("decline");
    }

    private void Focus(int index)
    {
        if (seats.Count == 0) return;
        index = (index + seats.Count) % seats.Count;
        if (index != focus) AudioManager.Get()?.Play("navigateMenu");
        focus = index;
    }

    private void EditSeat(SeatView view)
    {
        if (open) return;
        focus = Mathf.Max(0, seats.IndexOf(view));
        OpenEditor(focus);
    }

    // ---- hint strip ----

    private void BuildStrip(Canvas root)
    {
        var rootRect = OverlayCanvas("RlAgentPickerHints", root, 400, out strip);
        stripRow = Node("hints", rootRect, new Vector2(0f, 92f), new Vector2(3000f, 104f));
        stripRow.anchorMin = stripRow.anchorMax = new Vector2(0.5f, 0f);
        UnityEngine.Events.UnityAction[] clicks = { () => AddSeat(), () => RemoveSeat(focus), () => OpenEditor(focus), () => Focus(focus + 1) };
        for (int i = 0; i < clicks.Length; i++) {
            var (hint, _) = Box("hint" + i, stripRow, Vector2.zero, new Vector2(520f, 96f), edge, blue, 48f, 6f, true);
            // The results screen's "ability select" sits where these hints do: the
            // click that leaves it must not land on one (it removed a bot) as the scene opens.
            var click = clicks[i];
            OnClick(hint, () => { if (Time.unscaledTime > builtAt + 1f) click(); });
            var key = Box("key", hint.transform, Vector2.zero, new Vector2(200f, 72f), Color.white, Color.white, 36f, 1f);
            var keyText = Label("text", key.outer.transform, "", 42f, edge, Vector2.zero, new Vector2(200f, 72f));
            var action = Label("action", hint.transform, "", 46f, Color.white, Vector2.zero, new Vector2(300f, 90f), TextAlignmentOptions.Left);
            action.enableWordWrapping = false;
            stripHints.Add((hint, keyText, action));
        }
        stripNote = Label("note", stripRow, "", 44f, edge, new Vector2(0f, 0f), new Vector2(1400f, 96f));
        var (warning, _) = Box("teams", rootRect, new Vector2(0f, 214f), new Vector2(1500f, 96f), orange, blue, 48f, 6f);
        warning.rectTransform.anchorMin = warning.rectTransform.anchorMax = new Vector2(0.5f, 0f);
        Label("text", warning.transform, "everyone is on one team: give a bot another team to start", 44f, Color.white, Vector2.zero, new Vector2(1460f, 90f)).enableWordWrapping = false;
        teamWarning = warning.gameObject;
        teamWarning.SetActive(false);
    }

    private void RefreshStrip()
    {
        if (strip == null) return;
        bool pad = usingGamepad;
        var shown = new List<(string key, string action, int hint)>();
        if (CanAdd) shown.Add((pad ? "l3" : "ins", seats.Count == 0 ? "add a bot" : "add bot", 0));
        if (seats.Count > 0) shown.Add((pad ? "hold l3" : "del", "remove", 1));
        if (seats.Count > 0) shown.Add((pad ? "r3" : "tab", "edit bot", 2));
        if (seats.Count > 1 && pad) shown.Add(("r stick", "next bot", 3));
        float x = 0f, gap = 40f;
        var widths = new List<float>();
        foreach (var (key, action, hint) in shown) {
            var (box, keyText, actionText) = stripHints[hint];
            keyText.text = key;
            actionText.text = action;
            float keyWidth = Mathf.Max(110f, keyText.GetPreferredValues(key).x + 50f), actionWidth = actionText.GetPreferredValues(action).x;
            float width = 16f + keyWidth + 24f + actionWidth + 34f;
            var keyBox = (RectTransform)keyText.transform.parent;
            keyBox.sizeDelta = new Vector2(keyWidth, 72f);
            keyBox.GetChild(0).GetComponent<RectTransform>().sizeDelta = new Vector2(keyWidth - 2f, 70f);
            keyText.rectTransform.sizeDelta = new Vector2(keyWidth, 72f);
            keyBox.anchoredPosition = new Vector2(-width * 0.5f + 16f + keyWidth * 0.5f, 0f);
            actionText.rectTransform.sizeDelta = new Vector2(actionWidth + 10f, 90f);
            actionText.rectTransform.anchoredPosition = new Vector2(-width * 0.5f + 16f + keyWidth + 24f + actionWidth * 0.5f + 5f, -2f);
            box.rectTransform.sizeDelta = new Vector2(width, 96f);
            box.transform.GetChild(0).GetComponent<RectTransform>().sizeDelta = new Vector2(width - 12f, 84f);
            widths.Add(width);
        }
        float total = -gap;
        foreach (float width in widths) total += width + gap;
        x = -total * 0.5f;
        for (int i = 0; i < stripHints.Count; i++) stripHints[i].box.gameObject.SetActive(false);
        for (int i = 0; i < shown.Count; i++) {
            var box = stripHints[shown[i].hint].box;
            box.gameObject.SetActive(true);
            box.rectTransform.anchoredPosition = new Vector2(x + widths[i] * 0.5f, 0f);
            x += widths[i] + gap;
        }
        stripNote.text = bots.Count == 0 ? (RlAgentCatalog.LoadError ?? "no bots available here") : "";
    }

    // A short note on the strip (why an add did nothing).
    private float flashUntil;
    private void Flash(string text)
    {
        if (open) { errorText.text = text; return; }
        foreach (var hint in stripHints) hint.box.gameObject.SetActive(false);
        stripNote.text = text;
        stripNote.color = orange;
        flashUntil = Time.unscaledTime + 1.6f;
    }

    // ---- editor ----

    private void BuildOverlay(Canvas root)
    {
        var rootRect = OverlayCanvas("RlAgentPicker", root, 500, out overlay);
        var backdrop = Picture("backdrop", rootRect, null, new Color(0.02f, 0.12f, 0.24f, 0.6f), Vector2.zero, Vector2.zero, true);
        var backdropRect = backdrop.rectTransform;
        backdropRect.anchorMin = Vector2.zero; backdropRect.anchorMax = Vector2.one; backdropRect.sizeDelta = Vector2.zero;
        OnClick(backdrop, () => CloseEditor(false));

        var (panelOuter, panelInner) = Box("panel", rootRect, new Vector2(0f, -20f), new Vector2(PanelWidth, PanelHeight), edge, blue, 120f, 18f, true);
        panel = panelOuter.rectTransform;
        panelInner.raycastTarget = true;   // clicks on the panel do not reach the backdrop
        Transform p = panel;
        title = Label("title", p, "choose your opponent", 104f, edge, new Vector2(0f, 842f), new Vector2(2700f, 140f));
        title.enableWordWrapping = false;

        // Left: the bots by group.
        var (listOuter, _) = Box("list", p, new Vector2(ListX, ListY), new Vector2(ListWidth + 40f, ListHeight + 40f), edge, deep, 70f, 12f);
        listFrame = listOuter;
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

        // Right: the bot being edited.
        float width = PanelWidth * 0.5f - 60f - DetailLeft;
        detailName = Label("name", p, "", 96f, Color.white, new Vector2(DetailLeft + width * 0.5f, 690f), new Vector2(width, 130f), TextAlignmentOptions.Left);
        detailName.enableWordWrapping = false;
        var chip = Box("kind", p, new Vector2(DetailLeft + 170f, 570f), new Vector2(340f, 84f), edge, edge, 42f, 1f);
        kindChip = chip.outer;
        detailKind = Label("text", kindChip.transform, "", 44f, Color.white, Vector2.zero, new Vector2(320f, 80f));
        detailRating = Label("rating", p, "", 60f, Color.white, new Vector2(DetailLeft + width * 0.5f + 380f, 570f), new Vector2(width - 380f, 90f), TextAlignmentOptions.Left);
        detailDescription = Label("description", p, "", 38f, Color.white, new Vector2(DetailLeft + width * 0.5f, 440f), new Vector2(width, 164f), TextAlignmentOptions.TopLeft);
        detailDescription.lineSpacing = -8f;
        // The team row: the six teams in their colours, each with who is on it.
        teamHeader = Label("team", p, "team", 70f, edge, new Vector2(DetailLeft + 90f, TeamRowY), new Vector2(180f, 100f), TextAlignmentOptions.Left);
        float teamLeft = DetailLeft + 190f, teamRight = DetailLeft + width;
        var (teamOuter, _) = Box("teams", p, new Vector2((teamLeft + teamRight) * 0.5f, TeamRowY), new Vector2(teamRight - teamLeft, 104f), edge, deep, 52f, 7f, true);
        teamRow = teamOuter;
        OnClick(teamRow, () => SetField(FieldTeam));
        for (int t = 0; t < RlAgentCatalog.Teams; t++) {
            int team = t;
            var (outer, inner) = Box("team" + t, teamRow.transform, new Vector2(-(teamRight - teamLeft) * 0.5f + 92f + t * 150f, 0f), new Vector2(136f, 80f), TeamBorder(t), TeamFill(t), 40f, 7f, true);
            OnClick(outer, () => { SetField(FieldTeam); SetTeam(team, true); });
            var label = Label("who", outer.transform, "", 34f, Color.white, Vector2.zero, new Vector2(128f, 76f));
            label.enableWordWrapping = false;
            teamCells[t] = (outer, inner, label);
        }
        float statusLeft = -(teamRight - teamLeft) * 0.5f + 92f + RlAgentCatalog.Teams * 150f - 50f, statusWidth = (teamRight - teamLeft) * 0.5f - statusLeft - 30f;
        teamStatus = Label("status", teamRow.transform, "", 38f, Color.white, new Vector2(statusLeft + statusWidth * 0.5f, 0f), new Vector2(statusWidth, 100f), TextAlignmentOptions.Left);
        teamStatus.lineSpacing = -12f;
        kitHeader = Label("kit", p, "kit", 70f, edge, new Vector2(DetailLeft + 90f, KitRowY), new Vector2(180f, 100f), TextAlignmentOptions.Left);
        // The kit row: Random, a trained kit, or custom; up / down steps it.
        float rowLeft = DetailLeft + 190f, rowRight = DetailLeft + width;
        var (rowOuter, _) = Box("option", p, new Vector2((rowLeft + rowRight) * 0.5f, KitRowY), new Vector2(rowRight - rowLeft, 104f), edge, deep, 52f, 7f, true);
        kitRow = rowOuter;
        OnClick(kitRow, () => SetField(FieldKit));
        kitOptionText = Label("text", kitRow.transform, "", 48f, Color.white, new Vector2(-50f, 0f), new Vector2(rowRight - rowLeft - 160f, 96f), TextAlignmentOptions.Left);
        float arrows = (rowRight - rowLeft) * 0.5f - 60f;
        kitUp = Arrow("up", kitRow.transform, new Vector2(arrows - 10f, 30f), 68f, -1);
        kitDown = Arrow("down", kitRow.transform, new Vector2(arrows - 10f, -30f), 68f, 1);
        OnClick(kitUp, () => { SetField(FieldKit); StepValue(-1); });
        OnClick(kitDown, () => { SetField(FieldKit); StepValue(1); });
        // The three slots: each any ability the bot supports.
        for (int i = 0; i < 3; i++) {
            float x = DetailCenter + (i - 1) * 360f;
            int slot = i;
            kitRings[i] = Picture("ring" + i, p, ring, edge, new Vector2(x, RingY), new Vector2(250f, 266f), true);
            OnClick(kitRings[i], () => SetField(FieldSlot + slot));
            kitIcons[i] = Picture("icon", kitRings[i].transform, null, Color.white, new Vector2(0.7f, -5.5f), new Vector2(398f, 381f));
            slotUp[i] = Arrow("up" + i, p, new Vector2(x, RingY + 172f), 104f, -1);
            slotDown[i] = Arrow("down" + i, p, new Vector2(x, RingY - 172f), 104f, 1);
            OnClick(slotUp[i], () => { SetField(FieldSlot + slot); StepValue(-1); });
            OnClick(slotDown[i], () => { SetField(FieldSlot + slot); StepValue(1); });
            kitNames[i] = Label("name" + i, p, "", 42f, Color.white, new Vector2(x, RingY - 236f), new Vector2(350f, 64f));
            kitNames[i].enableWordWrapping = false;
        }
        // Under them: the kit options, or the abilities a slot can take.
        for (int i = 0; i < GridCells; i++) {
            int column = i % GridColumns, line = i / GridColumns;
            var cell = GridCell("option" + i, p, new Vector2(DetailCenter + (column - 2) * 312f, -350f - line * 108f), new Vector2(296f, 96f), 3);
            OnClick(cell.Outer, () => { SetField(FieldKit); SetKitOption(cell.Value, true); });
            kitCells.Add(cell);
        }
        for (int i = 0; i < AbilityCells; i++) {
            int column = i % AbilityColumns, line = i / AbilityColumns;
            var cell = GridCell("ability" + i, p, new Vector2(DetailCenter + (column - (AbilityColumns - 1) * 0.5f) * 120f, -350f - line * 120f), new Vector2(108f, 108f), 1);
            OnClick(cell.Outer, () => SetAbility(field - FieldSlot, cell.Value));
            abilityCells.Add(cell);
        }
        kitNote = Label("note", p, "", 42f, Color.white, new Vector2(DetailLeft + width * 0.5f, -775f), new Vector2(width, 100f));
        errorText = Label("error", p, "", 42f, orange, new Vector2(DetailLeft + width * 0.5f, -775f), new Vector2(width, 100f));

        // Footer: what the controls do, and clickable too.
        hintKeys = new TextMeshProUGUI[5]; hintActions = new TextMeshProUGUI[5]; hintBoxes = new Image[5];
        string[] actions = { "move", "done", "add bot", "remove", "cancel" };
        UnityEngine.Events.UnityAction[] clicks = { () => StepField(1), () => Done(), () => EditorAdd(), () => EditorRemove(), () => CloseEditor(false) };
        for (int i = 0; i < 5; i++) {
            var (hint, _) = Box("hint" + i, p, new Vector2((i - 2) * 540f, -872f), new Vector2(510f, 100f), edge, deep, 50f, 6f, true);
            OnClick(hint, clicks[i]);
            hintBoxes[i] = hint;
            var key = Box("key", hint.transform, new Vector2(-120f, 0f), new Vector2(230f, 76f), Color.white, Color.white, 38f, 1f);
            hintKeys[i] = Label("text", key.outer.transform, "", 42f, edge, Vector2.zero, new Vector2(230f, 76f));
            hintActions[i] = Label("action", hint.transform, actions[i], 46f, Color.white, new Vector2(135f, 0f), new Vector2(230f, 90f), TextAlignmentOptions.Left);
            hintActions[i].enableWordWrapping = false;
        }
        overlay.gameObject.SetActive(false);
    }

    private Cell GridCell(string name, Transform parent, Vector2 position, Vector2 size, int icons)
    {
        var (outer, inner) = Box(name, parent, position, size, edge, deep, 40f, 6f, true);
        var cell = new Cell { Rect = outer.rectTransform, Outer = outer, Inner = inner, Icons = new Image[icons] };
        float icon = icons == 1 ? size.y * 1.25f : 113f;
        for (int k = 0; k < icons; k++) cell.Icons[k] = Picture("icon" + k, outer.transform, null, Color.white, new Vector2((k - (icons - 1) * 0.5f) * 88f, 0f), new Vector2(icon * 1.04f, icon));
        return cell;
    }

    private void RefreshHints()
    {
        string[] keys = usingGamepad ? new[] { "r stick", "r3", "l3", "hold l3", "" } : new[] { "arrows", "enter", "ins", "del", "esc" };
        var shown = new List<int>();
        for (int i = 0; i < 5; i++) {
            hintKeys[i].text = keys[i];
            bool show = keys[i].Length > 0 && (i != 2 || CanAdd);
            hintBoxes[i].gameObject.SetActive(show);
            if (show) shown.Add(i);
        }
        // The shown hints sit centred under the panel.
        for (int k = 0; k < shown.Count; k++)
            hintBoxes[shown[k]].rectTransform.anchoredPosition = new Vector2((k - (shown.Count - 1) * 0.5f) * 540f, -872f);
        RefreshStrip();
    }

    // Kit options: Random (if the bot can draw), its trained kits, custom (if
    // its slots can take other abilities).
    private static int RandomOptions(RlAgentCatalog.Bot bot) => bot.CanRandomize ? 1 : 0;
    private static int KitOptions(RlAgentCatalog.Bot bot) => RandomOptions(bot) + bot.Kits.Count + (bot.CanCustomize ? 1 : 0);
    private static int CustomOption(RlAgentCatalog.Bot bot) => bot.CanCustomize ? RandomOptions(bot) + bot.Kits.Count : -1;

    private static int OptionOf(RlAgentCatalog.Seat seat)
    {
        var bot = seat.Bot;
        if (seat.Kit == null) return 0;
        int set = RlAgentCatalog.KitIndex(bot, seat.Kit);
        if (set >= 0) return RandomOptions(bot) + set;
        return Mathf.Max(0, CustomOption(bot));
    }

    private int[] KitOfOption(RlAgentCatalog.Bot bot, int option)
    {
        if (option < RandomOptions(bot)) return null;
        option -= RandomOptions(bot);
        if (option < bot.Kits.Count) return (int[])bot.Kits[option].Clone();
        return (int[])(lastCustom ?? bot.BaseKit).Clone();
    }

    private bool FieldOpen(int f)
    {
        if (bots.Count == 0 || draft == null) return false;
        if (f == FieldBot || f == FieldTeam) return true;
        if (f == FieldKit) return KitOptions(draft.Bot) > 1;
        return f >= FieldSlot && f < FieldSlot + 3 && draft.Bot.CanCustomize;
    }

    private void SetField(int f)
    {
        if (!open || !FieldOpen(f) || f == field) return;
        AudioManager.Get()?.Play("navigateMenu2");
        field = f;
        RefreshDetails(false);
    }

    private void StepField(int step)
    {
        int f = field + step;
        while (f >= FieldBot && f < FieldSlot + 3 && !FieldOpen(f)) f += step;
        if (f >= FieldBot && f < FieldSlot + 3) SetField(f);
    }

    private void StepValue(int step)
    {
        if (!open || bots.Count == 0) return;
        if (field == FieldBot) SelectBot(cursor + step, true);
        else if (field == FieldTeam) SetTeam((draft.Team + step + RlAgentCatalog.Teams) % RlAgentCatalog.Teams, true);
        else if (field == FieldKit) {
            int count = KitOptions(draft.Bot);
            if (count > 1) SetKitOption((kitOption + step + count) % count, true);
        } else StepSlot(field - FieldSlot, step);
    }

    private void SelectBot(int index, bool sound)
    {
        if (bots.Count == 0) return;
        index = Mathf.Clamp(index, 0, bots.Count - 1);
        if (sound && index != cursor) AudioManager.Get()?.Play("navigateMenu");
        cursor = index;
        var bot = bots[cursor];
        if (bot != draft.Bot) {
            // Keep the kit when the new bot can play it too (moving between
            // nets that draw kits); otherwise its default: Random if it can.
            var previous = draft.Bot;
            int[] kit = previous.CanRandomize && RlAgentCatalog.Allowed(bot, draft.Kit) ? draft.Kit : null;
            if (kit == null && !bot.CanRandomize && bot.Kits.Count > 0) kit = (int[])bot.Kits[0].Clone();
            draft = new RlAgentCatalog.Seat { Bot = bot, Kit = kit == null ? null : (int[])kit.Clone(), Team = draft.Team };
            kitOption = OptionOf(draft);
            if (kitOption != CustomOption(bot)) lastCustom = null;
            if (!FieldOpen(field)) field = FieldBot;
        }
        errorText.text = "";
        RefreshDetails(true);
        KeepVisible();
    }

    private void SetTeam(int team, bool sound)
    {
        if (!open || draft == null || !RlAgentCatalog.ValidTeam(team)) return;
        if (sound && team != draft.Team) AudioManager.Get()?.Play("navigateMenu2");
        draft.Team = team;
        RefreshDetails(false);
        teamCells[team].outer.transform.localScale = Vector3.one * 1.2f;
    }

    private void SetKitOption(int option, bool sound)
    {
        if (bots.Count == 0) return;
        var bot = draft.Bot;
        if (option < 0 || option >= KitOptions(bot)) return;
        if (sound && option != kitOption) AudioManager.Get()?.Play("navigateMenu2");
        kitOption = option;
        draft.Kit = KitOfOption(bot, option);
        RefreshDetails(true);
    }

    // A slot changed: the kit becomes custom (from the kit shown, or the bot's first).
    private void SetSlot(int slot, int ability, bool sound)
    {
        var bot = draft.Bot;
        int[] kit = (int[])(draft.Kit ?? lastCustom ?? bot.BaseKit).Clone();
        if (kit[slot] == ability && draft.Kit != null) return;
        kit[slot] = ability;
        if (sound) AudioManager.Get()?.Play("navigateMenu2");
        draft.Kit = kit;
        lastCustom = (int[])kit.Clone();
        kitOption = CustomOption(bot);
        RefreshDetails(false);
        kitIcons[slot].transform.localScale = Vector3.one * 1.15f;
    }

    private void StepSlot(int slot, int step)
    {
        var pool = draft.Bot.Pool;
        if (pool.Count == 0) return;
        int[] kit = draft.Kit ?? lastCustom ?? draft.Bot.BaseKit;
        int at = pool.IndexOf(kit[slot]);
        at = at < 0 ? 0 : (at + step + pool.Count) % pool.Count;
        SetSlot(slot, pool[at], true);
    }

    private void SetAbility(int slot, int ability)
    {
        if (!open || slot < 0 || slot > 2 || !draft.Bot.Pool.Contains(ability)) return;
        SetSlot(slot, ability, true);
    }

    private void JumpGroup(int step)
    {
        if (bots.Count == 0) return;
        int GroupStart(int i)
        {
            while (i > 0 && bots[i - 1].Group == bots[i].Group) i--;
            return i;
        }
        field = FieldBot;
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
            row.Check.enabled = editing >= 0 && editing < seats.Count && bots[row.Bot] == seats[editing].Seat.Bot;
            row.Check.color = selected ? orange : edge;
        }
        bool ally = draft != null && TeamMates(draft.Team, editing, false).humans.Count > 0;
        title.text = (seats.Count > 1 ? (ally ? "ally " : "opponent ") + (editing + 1) + " of " + seats.Count : ally ? "choose your ally" : "choose your opponent")
            + "  <size=42%>" + TeamNote() + "</size>";
        listFrame.color = field == FieldBot ? orange : edge;
        RefreshTeams();
        if (bots.Count == 0 || draft == null) {
            detailName.text = ""; detailKind.text = ""; kindChip.enabled = false; detailRating.text = ""; detailDescription.text = "";
            kitHeader.text = ""; kitOptionText.text = ""; kitNote.text = ""; kitRow.gameObject.SetActive(false);
            foreach (var r in kitRings) r.gameObject.SetActive(false);
            foreach (var n in kitNames) n.text = "";
            for (int i = 0; i < 3; i++) { slotUp[i].gameObject.SetActive(false); slotDown[i].gameObject.SetActive(false); }
            foreach (var cell in kitCells) cell.Rect.gameObject.SetActive(false);
            foreach (var cell in abilityCells) cell.Rect.gameObject.SetActive(false);
            return;
        }
        var bot = draft.Bot;
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

        // The kit row.
        int options = KitOptions(bot);
        bool kitFocused = field == FieldKit;
        kitRow.gameObject.SetActive(true);
        kitRow.color = kitFocused ? orange : edge;
        kitUp.gameObject.SetActive(options > 1); kitDown.gameObject.SetActive(options > 1);
        kitUp.color = kitDown.color = kitFocused ? orange : edge;
        int custom = CustomOption(bot);
        int set = kitOption - RandomOptions(bot);
        string rating = "";
        if (draft.Kit != null && bot.KitRatings.TryGetValue(RlAgentCatalog.KitKey(draft.Kit), out float value))
            rating = "  <color=#9fd6ff>~" + Mathf.RoundToInt(value) + " elo on this kit</color>";
        if (options == 0) kitOptionText.text = "plays the kit it trained on";
        else if (draft.Kit == null) kitOptionText.text = "random: " + (bot.AnyKit ? "any 3 of its " + bot.Pool.Count + " abilities" : "1 of its " + bot.Kits.Count + " kits") + " each round";
        else if (kitOption == custom) kitOptionText.text = "custom kit" + (bot.AnyKit ? "" : "  <color=#9fd6ff>(not one of its " + bot.Kits.Count + " trained kits)</color>") + rating;
        else if (bot.Kits.Count > 1) kitOptionText.text = "trained kit " + (set + 1) + " of " + bot.Kits.Count + rating;
        else kitOptionText.text = "its only kit" + rating;

        // The slots.
        bool canSlot = bot.CanCustomize, hasKit = draft.Kit != null || bot.CanRandomize || bot.Kits.Count > 0;
        int[] shown = draft.Kit ?? (bot.CanRandomize ? new[] { 1, 1, 1 } : bot.Kits.Count > 0 ? bot.Kits[0] : null);
        for (int i = 0; i < 3; i++) {
            bool focused = field == FieldSlot + i;
            kitRings[i].gameObject.SetActive(hasKit);
            kitRings[i].color = focused ? orange : edge;
            slotUp[i].gameObject.SetActive(hasKit && canSlot); slotDown[i].gameObject.SetActive(hasKit && canSlot);
            slotUp[i].color = slotDown[i].color = focused ? orange : edge;
            if (!hasKit) { kitNames[i].text = ""; continue; }
            kitIcons[i].sprite = Icon(shown[i]);
            kitNames[i].text = draft.Kit != null || !bot.CanRandomize ? AbilityName(shown[i]) : i == 1 ? AbilityName(1) : "";
            kitNames[i].color = focused ? orange : Color.white;
            if (pop) kitIcons[i].transform.localScale = Vector3.one * 1.15f;
        }

        // Under them: the kit options, or the abilities of the focused slot.
        bool slotGrid = field >= FieldSlot && canSlot;
        bool grid = !slotGrid && options > 1;
        for (int i = 0; i < kitCells.Count; i++) {
            var cell = kitCells[i];
            bool used = grid && i < options;
            cell.Rect.gameObject.SetActive(used);
            if (!used) continue;
            cell.Value = i;
            int[] kit = i < RandomOptions(bot) ? null : i == custom ? (lastCustom ?? draft.Kit ?? bot.BaseKit) : bot.Kits[i - RandomOptions(bot)];
            for (int k = 0; k < 3; k++) {
                cell.Icons[k].sprite = Icon(kit == null ? 1 : kit[k]);
                cell.Icons[k].enabled = kit != null || k == 1;
            }
            bool selected = i == kitOption;
            cell.Outer.color = selected ? orange : i == custom ? light : edge;
            cell.Inner.color = selected ? Color.white : deep;
        }
        int slot = field - FieldSlot;
        for (int i = 0; i < abilityCells.Count; i++) {
            var cell = abilityCells[i];
            bool used = slotGrid && i < bot.Pool.Count;
            cell.Rect.gameObject.SetActive(used);
            if (!used) continue;
            cell.Value = bot.Pool[i];
            cell.Icons[0].sprite = Icon(cell.Value);
            bool selected = draft.Kit != null && draft.Kit[slot] == cell.Value;
            cell.Outer.color = selected ? orange : edge;
            cell.Inner.color = selected ? Color.white : deep;
        }
        if (slotGrid) kitNote.text = "slot " + (slot + 1) + ": " + (bot.AnyKit ? "any of the " + bot.Pool.Count + " abilities it trained with" : "the " + bot.Pool.Count + " abilities of its trained kits");
        else if (options > GridCells) kitNote.text = "up / down steps through all " + options + " kit choices";
        else if (grid) kitNote.text = "";
        else if (bot.Kind == "scripted") kitNote.text = "scripted bots are built for dash / bow / platform only";
        else if (bot.Kind == "specialist") kitNote.text = "a specialist plays only the kit it was trained on";
        else if (bot.Kind == "roster") kitNote.text = "every roster member plays this kit";
        else kitNote.text = "";
        kitNote.enabled = errorText.text.Length == 0;
    }

    // The team row: every team with who else is on it, the draft's ringed.
    private void RefreshTeams()
    {
        bool shown = bots.Count > 0 && draft != null;
        teamRow.gameObject.SetActive(shown);
        teamHeader.text = shown ? "team" : "";
        if (!shown) return;
        bool focused = field == FieldTeam;
        teamRow.color = focused ? orange : edge;
        teamHeader.color = focused ? orange : edge;
        for (int t = 0; t < RlAgentCatalog.Teams; t++) {
            var (outer, inner, label) = teamCells[t];
            bool selected = t == draft.Team;
            outer.color = selected ? (focused ? orange : Color.white) : TeamBorder(t);
            inner.color = TeamFill(t);
            var (humans, others) = TeamMates(t, editing, true);
            humans.AddRange(others);
            label.text = humans.Count > 0 ? string.Join(" ", humans) : "";
            label.fontSize = humans.Count > 1 ? 28f : 34f;
            if (!selected) outer.transform.localScale = Vector3.one;
            else if (outer.transform.localScale.x < 1.1f) outer.transform.localScale = Vector3.one * 1.1f;
        }
        // Two short lines (the warnings go in the title's subtitle, TeamNote).
        var (allies, mates) = TeamMates(draft.Team, editing, true);
        bool allied = allies.Count > 0;
        allies.AddRange(mates);
        string with = allied ? "ally of " + string.Join(", ", allies) : mates.Count > 0 ? "with " + string.Join(", ", mates) : "on its own";
        teamStatus.text = "<color=#9fd6ff>" + TeamName(draft.Team) + " team</color>\n<size=85%>" + with + "</size>";
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
        field = FieldBot;
        if (rows[index].Bot == cursor && again) { Done(); return; }
        SelectBot(rows[index].Bot, true);
    }

    // Edit seat index (with no bot seated, seat one first).
    private void OpenEditor(int index)
    {
        if (open || Time.frameCount == toggledFrame && seats.Count > 0) return;
        if (seats.Count == 0) {
            if (!AddSeat()) return;
            index = 0;
        }
        open = true;
        toggledFrame = Time.frameCount;
        EventSystem.current?.SetSelectedGameObject(null);
        overlay.gameObject.SetActive(true);
        overlay.alpha = 0f;
        openTime = 0f;
        strip.alpha = 0f; strip.blocksRaycasts = false;
        AudioManager.Get()?.Play("accept");
        BeginEdit(index);
    }

    private void BeginEdit(int index)
    {
        editing = focus = Mathf.Clamp(index, 0, seats.Count - 1);
        draft = seats[editing].Seat.Copy();
        cursor = Mathf.Max(0, bots.IndexOf(draft.Bot));
        kitOption = OptionOf(draft);
        lastCustom = kitOption == CustomOption(draft.Bot) && draft.Kit != null ? (int[])draft.Kit.Clone() : null;
        field = FieldBot;
        heldVertical = heldHorizontal = 0;
        hoverRow = -1;
        errorText.text = "";
        RefreshHints();
        SelectBot(cursor, false);
        scroll = scrollTarget;
    }

    private void CloseEditor(bool done)
    {
        if (!open || Time.frameCount == toggledFrame) return;
        open = false;
        toggledFrame = Time.frameCount;
        overlay.gameObject.SetActive(false);
        strip.alpha = 1f; strip.blocksRaycasts = true;
        if (!done) AudioManager.Get()?.Play("decline");
        editing = -1;
        draft = null;
        foreach (var view in seats) RefreshSeat(view);
        RefreshStrip();
    }

    // Keep the edited bot; false (with the reason shown) when it does not load.
    private bool Commit()
    {
        if (draft == null || editing < 0 || editing >= seats.Count) return false;
        if (!Loads(draft.Bot, out string error)) {
            errorText.text = "can't load " + draft.Bot.Id + ": " + error;
            kitNote.enabled = false;
            AudioManager.Get()?.Play("decline");
            return false;
        }
        var view = seats[editing];
        view.Seat = draft.Copy();
        view.Shown = 0; view.Cycle = 0f;
        RefreshSeat(view);
        if (view.Seat.Kit != null) ShowSeatKit(view, view.Seat.Kit, true);
        Save();
        return true;
    }

    private void Done()
    {
        if (!open || bots.Count == 0) return;
        if (!Commit()) return;
        AudioManager.Get()?.Play("accept");
        CloseEditor(true);
    }

    // In the editor: keep this bot and edit a new copy of it.
    private void EditorAdd()
    {
        if (!open || !Commit()) return;
        if (AddSeat(seats[editing].Seat)) BeginEdit(seats.Count - 1);
    }

    private void EditorRemove()
    {
        if (!open) return;
        int index = editing;
        draft = null;
        RemoveSeat(index);
        if (seats.Count == 0) { toggledFrame = -1; CloseEditor(true); return; }
        BeginEdit(Mathf.Min(index, seats.Count - 1));
    }

    // ---- input ----

    private void TrackDevice()
    {
        var keyboard = Keyboard.current;
        var mouse = Mouse.current;
        bool was = usingGamepad;
        if ((keyboard != null && keyboard.anyKey.wasPressedThisFrame) || (mouse != null && (mouse.leftButton.wasPressedThisFrame || mouse.delta.ReadValue().sqrMagnitude > 25f)))
            usingGamepad = false;
        else
            foreach (var pad in Gamepad.all)
                if (pad.leftStickButton.wasPressedThisFrame || pad.rightStickButton.wasPressedThisFrame || pad.rightStick.ReadValue().sqrMagnitude > 0.25f
                    || pad.buttonSouth.wasPressedThisFrame || pad.startButton.wasPressedThisFrame || pad.dpad.ReadValue().sqrMagnitude > 0.25f || pad.leftStick.ReadValue().sqrMagnitude > 0.25f) {
                    usingGamepad = true;
                    break;
                }
        if (was != usingGamepad) { if (open) RefreshHints(); else RefreshStrip(); }
    }

    private static bool AnyPad(System.Func<Gamepad, bool> test)
    {
        foreach (var pad in Gamepad.all) if (test(pad)) return true;
        return false;
    }

    private static int Held(bool negative, bool positive) => negative == positive ? 0 : negative ? -1 : 1;

    // The way a stick points along its stronger axis, past the press point.
    private static Vector2Int Pointing(Vector2 stick)
    {
        if (stick.sqrMagnitude < StickPress * StickPress) return Vector2Int.zero;
        return Mathf.Abs(stick.x) > Mathf.Abs(stick.y) ? new Vector2Int(stick.x > 0f ? 1 : -1, 0) : new Vector2Int(0, stick.y > 0f ? 1 : -1);
    }

    private static Vector2Int AnyPointing(System.Func<Gamepad, Vector2> stick)
    {
        foreach (var pad in Gamepad.all) {
            var way = Pointing(stick(pad));
            if (way != Vector2Int.zero) return way;
        }
        return Vector2Int.zero;
    }

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

    // L3: a tap (released before HoldToRemove) adds a bot; holding it removes
    // the highlighted one (in the editor: the one being edited).
    private void LeftStickPress(bool blocked)
    {
        if (blocked) { l3Since = -1f; return; }
        if (AnyPad(pad => pad.leftStickButton.wasPressedThisFrame)) { l3Since = Time.unscaledTime; l3Removed = false; }
        if (l3Since < 0f) return;
        if (AnyPad(pad => pad.leftStickButton.isPressed)) {
            if (!l3Removed && Time.unscaledTime - l3Since >= HoldToRemove && seats.Count > 0) {
                l3Removed = true;
                if (open) EditorRemove(); else RemoveSeat(focus);
            }
            return;
        }
        if (!l3Removed) { if (open) EditorAdd(); else AddSeat(); }
        l3Since = -1f;
    }

    private void Update()
    {
        float dt = Time.unscaledDeltaTime;
        TrackDevice();
        UpdateSeats(dt);
        if (flashUntil > 0f && Time.unscaledTime > flashUntil) { flashUntil = 0f; stripNote.color = edge; RefreshStrip(); }
        // A box capturing a binding takes every button (KeyBindButton).
        bool rebinding = KeyBindButton.RebindFrame >= Time.frameCount - 1;
        var keyboard = Keyboard.current;
        bool Pressed(Key key) => keyboard != null && keyboard[key].wasPressedThisFrame;
        bool Down(Key key) => keyboard != null && keyboard[key].isPressed;
        if (!open) {
            LeftStickPress(rebinding || handler == null);
            if (rebinding || open) return;
            if (Pressed(Key.Insert) || Pressed(Key.NumpadPlus)) AddSeat();
            else if (Pressed(Key.Delete) && seats.Count > 0) RemoveSeat(focus);
            else if ((Pressed(Key.Tab) || AnyPad(pad => pad.rightStickButton.wasPressedThisFrame)) && Time.frameCount != toggledFrame) OpenEditor(focus);
            int horizontal = Repeat(AnyPointing(pad => pad.rightStick.ReadValue()).x, 0, ref heldFocus, ref timerFocus, RepeatRateHorizontal * 1.5f);
            if (horizontal != 0 && seats.Count > 1) { toggledFrame = Time.frameCount; Focus(focus + horizontal); }
            return;
        }
        openTime += dt;
        overlay.alpha = Mathf.Clamp01(openTime / 0.12f);
        panel.localScale = Vector3.one * (Mathf.Lerp(0.96f, 1f, overlay.alpha) * PanelFit());
        if (Time.frameCount == toggledFrame) return;   // the press that opened it
        LeftStickPress(rebinding);
        if (!open) return;
        if (Pressed(Key.Escape) || Pressed(Key.Backspace)) { CloseEditor(false); return; }
        if (Pressed(Key.Enter) || Pressed(Key.NumpadEnter) || Pressed(Key.Space) || Pressed(Key.Tab) || AnyPad(pad => pad.rightStickButton.wasPressedThisFrame)) { Done(); return; }
        if (Pressed(Key.Insert) || Pressed(Key.NumpadPlus)) { EditorAdd(); return; }
        if (Pressed(Key.Delete)) { EditorRemove(); return; }
        if (Pressed(Key.Q) || Pressed(Key.PageUp)) JumpGroup(-1);
        if (Pressed(Key.E) || Pressed(Key.PageDown)) JumpGroup(1);
        if (Pressed(Key.Home)) { field = FieldBot; SelectBot(0, true); }
        if (Pressed(Key.End)) { field = FieldBot; SelectBot(bots.Count - 1, true); }

        // The right stick; the d-pad, left stick and keys too while the editor has them.
        var way = AnyPointing(pad => pad.rightStick.ReadValue());
        if (way == Vector2Int.zero) way = AnyPointing(pad => pad.leftStick.ReadValue());
        bool up = way.y > 0 || Down(Key.W) || Down(Key.UpArrow) || AnyPad(pad => pad.dpad.up.isPressed);
        bool down = way.y < 0 || Down(Key.S) || Down(Key.DownArrow) || AnyPad(pad => pad.dpad.down.isPressed);
        bool leftHeld = way.x < 0 || Down(Key.A) || Down(Key.LeftArrow) || AnyPad(pad => pad.dpad.left.isPressed);
        bool rightHeld = way.x > 0 || Down(Key.D) || Down(Key.RightArrow) || AnyPad(pad => pad.dpad.right.isPressed);
        int tapVertical = Held(Pressed(Key.W) || Pressed(Key.UpArrow) || AnyPad(pad => pad.dpad.up.wasPressedThisFrame),
                               Pressed(Key.S) || Pressed(Key.DownArrow) || AnyPad(pad => pad.dpad.down.wasPressedThisFrame));
        int tapHorizontal = Held(Pressed(Key.A) || Pressed(Key.LeftArrow) || AnyPad(pad => pad.dpad.left.wasPressedThisFrame),
                                 Pressed(Key.D) || Pressed(Key.RightArrow) || AnyPad(pad => pad.dpad.right.wasPressedThisFrame));
        // Held up / down repeats faster the longer it is held (long lists).
        int heldWay = Held(up, down);
        verticalFor = heldWay != 0 ? verticalFor + dt : 0f;
        float rate = RepeatRateVertical * Mathf.Lerp(1f, 0.35f, Mathf.Clamp01((verticalFor - RepeatDelay) / 1.5f));
        int vertical = Repeat(heldWay, tapVertical, ref heldVertical, ref timerVertical, rate);
        if (vertical != 0) StepValue(vertical);   // up is -1: the previous bot, kit or ability
        int horizontal2 = Repeat(Held(leftHeld, rightHeld), tapHorizontal, ref heldHorizontal, ref timerHorizontal, RepeatRateHorizontal);
        if (horizontal2 != 0) StepField(horizontal2);

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
        if (draft != null && RlAgentCatalog.ValidTeam(draft.Team)) {
            var chosen = teamCells[draft.Team].outer.transform;
            chosen.localScale = Vector3.Lerp(chosen.localScale, Vector3.one * 1.1f, ease);
        }
    }

    // Narrow screens (wider than 16:9 here means a shorter canvas) shrink the panel to fit.
    private float PanelFit()
    {
        var area = ((RectTransform)overlay.transform).rect;
        return Mathf.Min(1f, (area.height - 60f) / PanelHeight, (area.width - 60f) / PanelWidth);
    }
}
