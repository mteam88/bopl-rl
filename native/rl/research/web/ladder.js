"use strict";
const $ = (id) => document.getElementById(id);
const fmt = new Intl.NumberFormat("en-US");
let ladders = {}, name = new URLSearchParams(location.search).get("ladder") || "", loading = false;
let sortKey = "rating", sortDir = -1;
const expanded = new Set();
// Scripted bot names end in a level (evaluation_protocol.opponents); random is the anchor.
const isBot = (n) => n === "random" || /-(easy|medium|hard)$/.test(n);
const COLUMNS = [
  ["name", "Participant"], ["kind", "Kind"], ["rating", "Elo"], ["interval", "95% interval"],
  ["games", "Games"], ["old_score", "Anchors score"], ["flags", "Flags"],
];
function element(tag, text = "", className = "") {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
}
function connect(ok) {
  $("connection").textContent = ok ? "Live · refreshes every 30s" : "Connection interrupted";
  $("connection").parentElement.classList.toggle("online", ok);
}
async function refresh() {
  if (loading || document.hidden) return;
  loading = true;
  try {
    const response = await fetch("/api/ladders", { cache: "no-store" });
    if (!response.ok) throw Error("Ladder data unavailable");
    ladders = await response.json();
    const names = Object.keys(ladders);
    if (!names.length) throw Error("No ladder is registered (serve.py --ladder NAME=DIR).");
    if (!ladders[name]) name = names[0];
    $("ladderPickerLabel").hidden = names.length < 2;
    $("ladderPicker").replaceChildren(...names.map((n) => Object.assign(element("option", n), { value: n })));
    $("ladderPicker").value = name;
    connect(true);
    $("notice").textContent = "";
    render();
  } catch (error) {
    connect(false);
    $("notice").textContent = error.message;
  } finally {
    loading = false;
  }
}
// One row per ladder participant; finals are joined to their rescore trial by checkpoint hash.
function rows() {
  const { ladder, rescore } = ladders[name];
  const trials = new Map((rescore?.trials || []).map((t) => [t.sha256, t]));
  return ladder.ratings.map((r) => {
    const trial = trials.get(r.sha256);
    return {
      ...r, trial,
      kind: trial ? (r.kind === "anchor" ? "final · pool" : "final") : isBot(r.name) ? "bot" : "anchor",
      old_score: trial?.old_score, flags: trial?.flags || [],
    };
  });
}
function render() {
  const { ladder, rescore } = ladders[name];
  const all = rows();
  const finals = all.filter((r) => r.trial);
  $("ladderTag").textContent = `${fmt.format(ladder.games)} games · ${all.length} participants · ${finals.length} finals`;
  $("ladderHead").replaceChildren(...COLUMNS.map(([key, label]) => {
    const th = element("th", label, "sortable" + (key === sortKey ? " sort-active" : ""));
    th.scope = "col";
    if (key === sortKey) th.textContent += sortDir < 0 ? " ↓" : " ↑";
    th.onclick = () => {
      sortDir = key === sortKey ? -sortDir : (key === "name" || key === "kind" ? 1 : -1);
      sortKey = key;
      render();
    };
    return th;
  }));
  const value = (r) => sortKey === "interval" ? r.rating : sortKey === "flags" ? r.flags.length : r[sortKey];
  const sorted = [...all].sort((a, b) => {
    const [x, y] = [value(a), value(b)];
    if (x === undefined || x === null) return 1;
    if (y === undefined || y === null) return -1;
    return (typeof x === "string" ? x.localeCompare(y) : x - y) * sortDir;
  });
  $("ladderRows").replaceChildren(...sorted.flatMap((r) => {
    const tr = element("tr", "", r.trial ? "final" + (expanded.has(r.name) ? " open" : "") : "");
    tr.append(element("td", r.name), element("td", r.kind, "kind-" + r.kind.split(" ")[0]),
      element("td", Math.round(r.rating).toString()),
      element("td", r.interval ? `${Math.round(r.interval[0])} – ${Math.round(r.interval[1])}` : "—"),
      element("td", fmt.format(r.games)),
      element("td", r.old_score === undefined ? "—" : r.old_score.toFixed(3)),
      element("td", r.flags.join(" · ")));
    if (!r.trial) return [tr];
    tr.title = "Trial " + r.trial.id + " · click for hyperparameters";
    tr.onclick = () => { expanded.has(r.name) ? expanded.delete(r.name) : expanded.add(r.name); render(); };
    if (!expanded.has(r.name)) return [tr];
    const detail = element("tr", "", "hypers");
    const cell = element("td");
    cell.colSpan = COLUMNS.length;
    const t = r.trial;
    const lastK = t.last_k.map((k) => `${fmt.format(k.learner_steps)}: ${Math.round(k.rating)} [${Math.round(k.interval[0])}, ${Math.round(k.interval[1])}]`).join(" · ");
    cell.append(element("p", `Trial ${t.id} · fixed-pool Elo of the final ${Math.round(t.rating)} [${Math.round(t.interval[0])}, ${Math.round(t.interval[1])}] over ${t.games} games · IQM${t.last_k.length} ${Math.round(t.iqm)} · last evaluations ${lastK}`));
    const list = element("dl");
    for (const [key, v] of Object.entries(t.hypers).sort()) {
      const item = element("div");
      item.append(element("dt", key), element("dd", typeof v === "number" ? (Number.isInteger(v) ? fmt.format(v) : v.toPrecision(4)) : String(v)));
      list.append(item);
    }
    cell.append(list);
    detail.append(cell);
    return [tr, detail];
  }));
  $("ladderNote").textContent = `${ladder.note}. ${ladder.games_per_pair} games per pair on maps ${ladder.maps.join(", ")}, ${ladder.episode_ticks / 60} s cap` +
    (rescore ? `; finals joined to ${rescore.trials.length} rescored trials of ${rescore.sweep.split("/").at(-1)}.` : ".");
  $("scatterSection").hidden = !rescore;
  if (rescore) {
    const inLadder = new Set(finals.map((r) => r.trial.id));
    $("scatterTag").textContent = `${rescore.trials.length} finals · ${inLadder.size} in the ladder`;
    scatter($("scatter"), rescore.trials, inLadder);
  }
  $("updated").textContent = `Ladder fitted ${new Date(ladder.updated * 1000).toLocaleString()} · read ${new Date().toLocaleTimeString()}`;
}
function scatter(canvas, trials, highlighted) {
  const r = canvas.getBoundingClientRect(), dpr = devicePixelRatio || 1;
  canvas.width = Math.round(r.width * dpr);
  canvas.height = Math.round(r.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const w = r.width, h = r.height, left = 44, right = 12, top = 10, bottom = 26;
  const xs = trials.map((t) => t.old_score), lows = trials.map((t) => t.interval[0]), highs = trials.map((t) => t.interval[1]);
  const xmin = Math.max(0, Math.min(...xs) - 0.05), xmax = Math.min(1, Math.max(...xs) + 0.05);
  const ymin = Math.floor(Math.min(...lows) / 50) * 50, ymax = Math.ceil(Math.max(...highs) / 50) * 50;
  const x = (v) => left + ((w - left - right) * (v - xmin)) / (xmax - xmin || 1);
  const y = (v) => top + (h - top - bottom) * (1 - (v - ymin) / (ymax - ymin || 1));
  ctx.font = '9px "Trebuchet MS",sans-serif';
  ctx.lineWidth = 1;
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const v = ymin + ((ymax - ymin) * i) / 4;
    ctx.strokeStyle = "#dfe2d5";
    ctx.beginPath(); ctx.moveTo(left, y(v)); ctx.lineTo(w - right, y(v)); ctx.stroke();
    ctx.fillStyle = "#868b7c"; ctx.textAlign = "right";
    ctx.fillText(Math.round(v).toString(), left - 7, y(v));
  }
  ctx.textBaseline = "bottom"; ctx.textAlign = "center";
  for (let i = 0; i <= 4; i++) {
    const v = xmin + ((xmax - xmin) * i) / 4;
    ctx.fillText(v.toFixed(2), x(v), h);
  }
  ctx.textBaseline = "top";
  ctx.textAlign = "left"; ctx.fillText("↑ reference Elo", left + 4, top);
  ctx.textAlign = "right"; ctx.fillText("anchors score →", w - right, top);
  for (const t of trials) {
    const inLadder = highlighted.has(t.id), color = inLadder ? "#2846d4" : "#868b7c";
    ctx.strokeStyle = color + "66";
    ctx.beginPath(); ctx.moveTo(x(t.old_score), y(t.interval[0])); ctx.lineTo(x(t.old_score), y(t.interval[1])); ctx.stroke();
    ctx.beginPath(); ctx.arc(x(t.old_score), y(t.rating), 3, 0, Math.PI * 2);
    ctx.fillStyle = inLadder ? color : "#f6f3e9"; ctx.fill();
    ctx.strokeStyle = color; ctx.stroke();
    ctx.fillStyle = color; ctx.textAlign = "left"; ctx.textBaseline = "middle";
    ctx.fillText(t.id, x(t.old_score) + 5, y(t.rating));
  }
}
$("ladderPicker").onchange = () => {
  name = $("ladderPicker").value;
  history.replaceState(null, "", "/ladder?ladder=" + encodeURIComponent(name));
  render();
};
window.addEventListener("resize", () => { if (ladders[name]) render(); });
document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });
refresh();
setInterval(refresh, 30000);
