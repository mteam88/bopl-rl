"use strict";
const $ = (id) => document.getElementById(id);
const colors = [
  "#f4d93f",
  "#b99be8",
  "#ffca6a",
  "#80c7e0",
  "#d9afee",
  "#ef9e70",
];
let data = null,
  runName = "",
  replay = null,
  selectedEvaluation = null,
  selectedGame = null,
  frameIndex = 0,
  playing = false,
  lastFrameTime = 0,
  view = "physics";
const fmt = new Intl.NumberFormat("en-US");
const short = (n) =>
  n >= 1e6
    ? (n / 1e6).toFixed(1) + "m"
    : n >= 1000
      ? (n / 1000).toFixed(n < 10000 ? 1 : 0) + "k"
      : Math.round(n).toString();
const percent = (n) => Math.round(100 * n) + "%";
const duration = (n) =>
  `${Math.floor(n / 60)}:${Math.floor(n % 60)
    .toString()
    .padStart(2, "0")}`;
const fileUrl = (evaluation, file) =>
  `/runs/${encodeURIComponent(runName)}/evaluations/${evaluation.directory}/${encodeURIComponent(file)}`;
function replaceOptions(select, items) {
  const previous = select.value;
  select.replaceChildren(
    ...items.map(([value, text]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = text;
      return option;
    }),
  );
  if (items.some((x) => x[0] === previous)) select.value = previous;
}
function connect(ok) {
  $("connection").textContent = ok
    ? "Live · updates every 5s"
    : "Connection interrupted";
  $("connection").parentElement.classList.toggle("online", ok);
}
async function refresh() {
  try {
    const runs = await (await fetch("/api/runs")).json();
    if (!runs.length) {
      connect(true);
      return;
    }
    if (!runName) {
      const query = new URLSearchParams(location.search).get("run");
      runName = runs.some((r) => r.name === query) ? query : runs[0].name;
    }
    replaceOptions(
      $("run"),
      runs.map((r) => [r.name, r.name.replaceAll("-", " ")]),
    );
    $("run").value = runName;
    const requestedRun = runName;
    const response = await fetch(
      "/api/run/" + encodeURIComponent(requestedRun),
    );
    if (!response.ok) throw Error("Run unavailable");
    const nextData = await response.json();
    if (requestedRun !== runName) return;
    data = nextData;
    connect(true);
    update();
  } catch (error) {
    connect(false);
    $("notice").textContent = error.message;
  }
}
function update() {
  const c = data.config,
    s = data.status,
    m = data.metrics.at(-1) || {};
  const step = s.step ?? m.step ?? 0;
  $("steps").textContent = fmt.format(step);
  $("target").textContent = `of ${fmt.format(c.steps)} planned decisions`;
  $("progress").style.width = Math.min(100, (100 * step) / c.steps) + "%";
  const recent = data.metrics.slice(-10);
  const sps = recent.length
    ? recent.reduce((a, b) => a + b.sps, 0) / recent.length
    : 0;
  $("sps").textContent = sps ? fmt.format(Math.round(sps)) : "—";
  $("matches").textContent = fmt.format(s.matches ?? m.matches ?? 0);
  $("elapsed").textContent =
    duration(s.elapsed ?? m.elapsed ?? 0) + " elapsed · two players per match";
  $("runState").textContent = s.status || "starting";
  $("notice").textContent =
    data.video?.status === "failed"
      ? "Video capture failed; native and physics replays remain available."
      : s.failure
        ? `Run stopped: ${s.failure}`
        : runName.includes("smoke")
          ? "Validation run — these short jobs check the setup, not policy strength."
          : "";
  $("kl").textContent = m.kl?.toFixed(4) ?? "—";
  $("clip").textContent =
    m.clip_fraction === undefined ? "—" : percent(m.clip_fraction);
  $("memory").textContent = m.cuda_memory_mb
    ? Math.round(m.cuda_memory_mb) + " MB"
    : "—";
  $("configLink").href = `/runs/${encodeURIComponent(runName)}/config.json`;
  $("updated").textContent =
    `Last checked ${new Date().toLocaleTimeString()} · local GPU`;
  const latest = data.evaluations.at(-1);
  const initial = latest?.groups.find(
    (g) => g.opponent === "initial" && g.map === 0,
  );
  $("win").textContent = initial ? percent(initial.win_rate) : "—";
  $("winDetail").textContent = initial
    ? `${initial.wins} / ${initial.games} wins · 95% interval ${percent(initial.interval[0])}–${percent(initial.interval[1])}`
    : "Fixed evaluation · both player sides";
  const evals = data.evaluations;
  const makeSeries = (opponent, map, color) => ({
    color,
    points: evals.flatMap((e) => {
      const g = e.groups.find((g) => g.opponent === opponent && g.map === map);
      return g
        ? [{ x: e.step, y: g.win_rate, lo: g.interval[0], hi: g.interval[1] }]
        : [];
    }),
  });
  const unseen = {
    color: "#71816a",
    points: evals.map((e) => {
      const gs = e.groups.filter((g) => g.map !== 0);
      return {
        x: e.step,
        y:
          gs.reduce((n, g) => n + g.wins, 0) /
          Math.max(
            1,
            gs.reduce((n, g) => n + g.games, 0),
          ),
      };
    }),
  };
  chart(
    "winsChart",
    [
      makeSeries("random", 0, "#2846d4"),
      makeSeries("initial", 0, "#e87350"),
      unseen,
    ],
    { min: 0, max: 1, format: percent },
  );
  chart(
    "speedChart",
    [
      {
        color: "#2846d4",
        points: data.metrics.map((m) => ({ x: m.step, y: m.sps })),
      },
    ],
    { min: 0, format: short },
  );
  chart(
    "entropyChart",
    [
      {
        color: "#71816a",
        points: data.metrics.map((m) => ({ x: m.step, y: m.entropy })),
      },
    ],
    { min: 0, max: Math.log(256) + Math.log(16), format: (n) => n.toFixed(1) },
  );
  chart(
    "lossChart",
    [
      {
        color: "#e87350",
        points: data.metrics.map((m) => ({ x: m.step, y: m.value_loss })),
      },
    ],
    { min: 0, format: (n) => n.toFixed(3) },
  );
  $("actionBars").replaceChildren(
    ...["jump", "dash", "arrow", "platform"].map((name) => {
      const rate = m.action_rates?.[name] ?? 0;
      const row = document.createElement("div");
      row.className = "bar";
      const label = document.createElement("span");
      label.textContent = name[0].toUpperCase() + name.slice(1);
      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill";
      fill.style.width = percent(rate);
      track.append(fill);
      const value = document.createElement("span");
      value.className = "bar-number";
      value.textContent = percent(rate);
      row.append(label, track, value);
      return row;
    }),
  );
  const old = $("checkpoint").value;
  replaceOptions(
    $("checkpoint"),
    evals
      .map((e) => [e.directory, `${fmt.format(e.step)} decisions`])
      .reverse(),
  );
  if (
    evals.length &&
    (!selectedEvaluation ||
      !evals.some((e) => e.directory === selectedEvaluation.directory) ||
      ($("followLatest").checked &&
        selectedEvaluation.directory !== evals.at(-1).directory))
  ) {
    $("checkpoint").value = evals.at(-1).directory;
    selectEvaluation();
  } else if (selectedEvaluation) {
    selectedEvaluation = evals.find(
      (e) => e.directory === selectedEvaluation.directory,
    );
    const game = selectedEvaluation.games.find(
      (g) => g.name === selectedGame?.name,
    );
    if (game?.video && !selectedGame.video) {
      selectedGame = game;
      setVideo();
    }
  }
}
function chart(id, series, options = {}) {
  const canvas = $(id),
    r = canvas.getBoundingClientRect(),
    dpr = devicePixelRatio || 1;
  canvas.width = Math.round(r.width * dpr);
  canvas.height = Math.round(r.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const w = r.width,
    h = r.height,
    left = 40,
    right = 8,
    top = 8,
    bottom = 20;
  const points = series
    .flatMap((s) => s.points)
    .filter((p) => Number.isFinite(p.y));
  const xmax = Math.max(1, ...points.map((p) => p.x)),
    ymin = options.min ?? Math.min(0, ...points.map((p) => p.y)),
    ymax =
      options.max ?? Math.max(0.001, ...points.map((p) => p.hi ?? p.y)) * 1.1;
  const x = (v) => left + ((w - left - right) * v) / xmax,
    y = (v) => top + (h - top - bottom) * (1 - (v - ymin) / (ymax - ymin || 1));
  ctx.font = '9px "Trebuchet MS",sans-serif';
  ctx.lineWidth = 1;
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 2; i++) {
    const value = ymin + ((ymax - ymin) * i) / 2,
      py = y(value);
    ctx.strokeStyle = "#dfe2d5";
    ctx.beginPath();
    ctx.moveTo(left, py);
    ctx.lineTo(w - right, py);
    ctx.stroke();
    ctx.fillStyle = "#868b7c";
    ctx.textAlign = "right";
    ctx.fillText((options.format || short)(value), left - 7, py);
  }
  ctx.textBaseline = "bottom";
  ctx.textAlign = "left";
  ctx.fillText("0", left, h);
  ctx.textAlign = "right";
  ctx.fillText(short(xmax) + " decisions", w - right, h);
  if (!points.length) {
    ctx.fillStyle = "#9b9e93";
    ctx.textAlign = "center";
    ctx.fillText("Waiting for measurements", w / 2, h / 2);
    return;
  }
  for (const s of series) {
    const pts = s.points.filter((p) => Number.isFinite(p.y));
    if (!pts.length) continue;
    if (pts.every((p) => p.lo !== undefined)) {
      ctx.fillStyle = s.color + "13";
      ctx.beginPath();
      pts.forEach((p, i) =>
        i ? ctx.lineTo(x(p.x), y(p.lo)) : ctx.moveTo(x(p.x), y(p.lo)),
      );
      [...pts].reverse().forEach((p) => ctx.lineTo(x(p.x), y(p.hi)));
      ctx.closePath();
      ctx.fill();
    }
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    pts.forEach((p, i) =>
      i ? ctx.lineTo(x(p.x), y(p.y)) : ctx.moveTo(x(p.x), y(p.y)),
    );
    ctx.stroke();
    if (pts.length < 30) {
      ctx.fillStyle = s.color;
      pts.forEach((p) => {
        if (p.lo !== undefined) {
          ctx.strokeStyle = s.color + "66";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x(p.x), y(p.lo));
          ctx.lineTo(x(p.x), y(p.hi));
          ctx.moveTo(x(p.x) - 3, y(p.lo));
          ctx.lineTo(x(p.x) + 3, y(p.lo));
          ctx.moveTo(x(p.x) - 3, y(p.hi));
          ctx.lineTo(x(p.x) + 3, y(p.hi));
          ctx.stroke();
        }
        ctx.beginPath();
        ctx.arc(x(p.x), y(p.y), 2.6, 0, Math.PI * 2);
        ctx.fill();
      });
    }
  }
}
function selectEvaluation() {
  selectedEvaluation = data.evaluations.find(
    (e) => e.directory === $("checkpoint").value,
  );
  if (!selectedEvaluation) return;
  replaceOptions(
    $("match"),
    selectedEvaluation.games
      .filter((g) => g.visual)
      .sort((a, b) =>
        a.name === "selfplay" ? -1 : b.name === "selfplay" ? 1 : 0,
      )
      .map((g) => [
        g.name,
        `${g.opponent === "selfplay" ? "Self-play" : g.opponent === "initial" ? "vs initial policy" : "vs random"} · map ${g.map + 1}`,
      ]),
  );
  selectMatch();
}
async function selectMatch() {
  selectedGame = selectedEvaluation?.games.find(
    (g) => g.name === $("match").value,
  );
  if (!selectedGame) return;
  const requested = selectedGame;
  playing = false;
  const response = await fetch(
    fileUrl(selectedEvaluation, selectedGame.visual),
  );
  const value = await response.json();
  if (requested !== selectedGame) return;
  replay = value;
  frameIndex = 0;
  $("emptyReplay").style.display = "none";
  $("play").disabled = false;
  $("play").textContent = "▶";
  $("scrub").max = replay.frames.length - 1;
  $("scrub").value = "0";
  $("replayTag").textContent =
    selectedGame.opponent === "selfplay"
      ? "Self-play match"
      : selectedGame.timeout
        ? "Time limit"
        : selectedGame.win
          ? "Policy wins"
          : "Policy loses";
  $("download").hidden = false;
  $("download").href = fileUrl(selectedEvaluation, selectedGame.replay);
  $("download").download = selectedGame.replay;
  setVideo();
  setView(selectedGame.video ? "video" : "physics");
  drawFrame();
}
function setVideo() {
  $("viewVideo").disabled = !selectedGame?.video;
  if (selectedGame?.video) {
    $("video").src = fileUrl(selectedEvaluation, selectedGame.video);
    $("video").poster = selectedGame.poster
      ? fileUrl(selectedEvaluation, selectedGame.poster)
      : "";
  } else {
    $("video").removeAttribute("src");
    $("video").load();
  }
}
function setView(next) {
  view = next;
  document.querySelector(".replay-controls").style.display =
    next === "physics" ? "flex" : "none";
  $("players").hidden = next !== "physics";
  $("video").pause();
  if (next === "video") playing = false;
  $("video").hidden = next !== "video";
  $("arena").style.display = next === "physics" ? "block" : "none";
  $("arenaLabel").style.display = next === "physics" ? "block" : "none";
  $("viewVideo").classList.toggle("active", next === "video");
  $("viewPhysics").classList.toggle("active", next === "physics");
}
function drawFrame() {
  if (!replay) return;
  const frame = replay.frames[Math.floor(frameIndex)],
    canvas = $("arena"),
    ctx = canvas.getContext("2d"),
    w = canvas.width,
    h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#18263a";
  ctx.fillRect(0, 0, w, h);
  const visible = replay.frames[0].entities.filter((e) => !(e.flags & 8));
  const xs = visible.flatMap((e) => [
    e.x - Math.abs(e.rx) - Math.abs(e.ux) - e.radius,
    e.x + Math.abs(e.rx) + Math.abs(e.ux) + e.radius,
  ]);
  const span = Math.max(
    54,
    Math.min(110, Math.max(...xs) - Math.min(...xs) + 12),
  );
  const scale = w / span,
    cx = w / 2,
    cy = h * 0.49;
  const point = (x, y) => [cx + x * scale, cy - y * scale];
  ctx.strokeStyle = "#243348";
  ctx.lineWidth = 1;
  for (let x = -60; x <= 60; x += 5) {
    const [px] = point(x, 0);
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, h);
    ctx.stroke();
  }
  for (let y = -30; y < 30; y += 5) {
    const [, py] = point(0, y);
    ctx.beginPath();
    ctx.moveTo(0, py);
    ctx.lineTo(w, py);
    ctx.stroke();
  }

  for (const e of frame.entities) {
    const [x, y] = point(e.x, e.y);
    const owned = e.owner > 0;
    if (
      e.layer === 8 &&
      owned &&
      frame.players.some(
        (p) => p.id === e.owner && Math.hypot(p.x - e.x, p.y - e.y) < 2,
      )
    )
      continue;
    ctx.fillStyle = owned
      ? colors[(e.owner - 1) % 6]
      : e.layer === 4
        ? "#274c64"
        : "#8b9e87";
    ctx.strokeStyle = "#c4d4b6";
    if (e.kind === 1 || e.kind === 2) {
      const corners = [
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
      ].map(([a, b]) =>
        point(e.x + a * e.rx + b * e.ux, e.y + a * e.ry + b * e.uy),
      );
      ctx.beginPath();
      corners.forEach(([px, py], i) =>
        i ? ctx.lineTo(px, py) : ctx.moveTo(px, py),
      );
      ctx.closePath();
      ctx.lineJoin = "round";
      ctx.lineWidth = Math.max(1, e.radius * scale * 2);
      ctx.strokeStyle = ctx.fillStyle;
      ctx.stroke();
      ctx.fill();
    } else if (e.kind === 0) {
      const isBody = frame.players.some(
        (p) =>
          p.flags & 1 &&
          p.id === e.owner &&
          Math.hypot(p.x - e.x, p.y - e.y) < 2,
      );
      if (isBody) continue;
      ctx.beginPath();
      ctx.arc(x, y, Math.max(2, Math.abs(e.radius) * scale), 0, Math.PI * 2);
      ctx.fill();
    } else {
      const [x2, y2] = point(e.x + e.rx, e.y + e.ry);
      ctx.strokeStyle = owned ? colors[(e.owner - 1) % 6] : "#bcd6d8";
      ctx.lineWidth = Math.max(2, e.radius * scale);
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
  }
  for (const p of frame.players) {
    if (!(p.flags & 1)) continue;
    const [x, y] = point(p.x, p.y),
      r = Math.max(8, p.scale * scale * 0.75);
    ctx.fillStyle = colors[(p.id - 1) % 6];
    ctx.strokeStyle = "#101d2b";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.ellipse(x, y, r, r * 0.86, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#152333";
    for (const dx of [-0.27, 0.29]) {
      ctx.beginPath();
      ctx.arc(x + dx * r, y - r * 0.15, Math.max(2, r * 0.11), 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = "#dce5e5";
    ctx.font = '12px "Trebuchet MS",sans-serif';
    ctx.textAlign = "center";
    ctx.fillText("P" + p.id, x, y - r - 9);
  }
  $("scrub").value = Math.floor(frameIndex);
  $("replayTime").textContent =
    duration(frame.t / 60) + " / " + duration(replay.frames.at(-1).t / 60);
  $("players").replaceChildren(
    ...frame.players.map((p) => {
      const box = document.createElement("div");
      box.className = "player";
      box.style.setProperty("--player-color", colors[(p.id - 1) % 6]);
      const label = document.createElement("div");
      label.className = "player-name";
      const name = document.createElement("b");
      name.textContent =
        "Player " + p.id + (p.id === selectedGame.side + 1 ? " · policy" : "");
      const score = document.createElement("span");
      score.textContent = `${p.kills} kills · ${p.stocks} reserves`;
      label.append(name, score);
      box.append(label);
      const abilities = document.createElement("div");
      abilities.className = "abilities";
      const slime = frame.slimes.find((s) => s.owner === p.id);
      ["Dash", "Arrow", "Platform"].forEach((name, i) => {
        const item = document.createElement("div");
        item.className = "ability";
        item.textContent = name;
        const track = document.createElement("span"),
          fill = document.createElement("i");
        const cd = slime?.cooldowns[i] ?? 0,
          timer = slime?.timers[i] ?? 0;
        fill.style.width =
          Math.min(100, Math.max(0, cd > 0 ? (timer / cd) * 100 : 0)) + "%";
        track.append(fill);
        item.append(track);
        abilities.append(item);
      });
      box.append(abilities);
      return box;
    }),
  );
}
function animate(now) {
  if (playing && replay && view === "physics") {
    frameIndex +=
      Math.min(0.1, (now - lastFrameTime) / 1000) *
      15 *
      Number($("speed").value);
    if (frameIndex >= replay.frames.length - 1) {
      frameIndex = replay.frames.length - 1;
      playing = false;
      $("play").textContent = "▶";
    }
    drawFrame();
  }
  lastFrameTime = now;
  requestAnimationFrame(animate);
}
$("run").onchange = () => {
  runName = $("run").value;
  data = null;
  selectedEvaluation = null;
  selectedGame = null;
  replay = null;
  playing = false;
  $("video").pause();
  $("emptyReplay").style.display = "flex";
  history.replaceState(null, "", "?run=" + encodeURIComponent(runName));
  refresh();
};
$("checkpoint").onchange = () => {
  $("followLatest").checked = false;
  selectEvaluation();
};
$("match").onchange = selectMatch;
$("play").onclick = () => {
  if (!replay) return;
  setView("physics");
  if (frameIndex >= replay.frames.length - 1) frameIndex = 0;
  playing = !playing;
  $("play").textContent = playing ? "Ⅱ" : "▶";
};
$("scrub").oninput = () => {
  frameIndex = Number($("scrub").value);
  drawFrame();
};
$("viewPhysics").onclick = () => setView("physics");
$("viewVideo").onclick = () => setView("video");
window.addEventListener("resize", () => {
  if (data) update();
});
refresh();
setInterval(refresh, 5000);
requestAnimationFrame(animate);
