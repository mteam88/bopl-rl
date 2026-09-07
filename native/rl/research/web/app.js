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
  study = null,
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
    const [runs, studies] = await Promise.all(
      ["/api/runs", "/api/series"].map(async (url) => {
        const response = await fetch(url);
        if (!response.ok) throw Error("Dashboard data unavailable");
        return response.json();
      }),
    );
    const query = new URLSearchParams(location.search);
    study =
      studies.find((s) => s.name === query.get("series")) ||
      studies.find((s) =>
        s.runs.some((r) => r.name === (query.get("run") || runName)),
      ) ||
      null;
    if (study && query.has("series") && !query.has("run")) {
      const active = study.status.active_run;
      const target = runs.some((r) => r.name === active)
        ? active
        : study.runs.find((r) => runs.some((x) => x.name === r.name))?.name;
      if (target && target !== runName) resetRun(target);
    }
    updateSeries();
    if (!runs.length) {
      connect(true);
      return;
    }
    if (!runName) {
      const requested = query.get("run");
      if (requested && !runs.some((r) => r.name === requested)) {
        connect(true);
        $("notice").textContent = `Waiting for ${requested} to start…`;
        return;
      }
      if (study && query.has("series")) {
        connect(true);
        $("notice").textContent =
          "The series is prepared. Its first run will appear here when training starts.";
        return;
      }
      runName = requested || runs[0].name;
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
function updateSeries() {
  $("seriesSection").hidden = !study;
  if (!study) return;
  const palette = ["#2846d4", "#e87350", "#71816a"];
  $("seriesState").textContent = study.status.status || "prepared";
  $("seriesTitle").textContent = `${study.runs.length} training seeds. One protocol.`;
  $("seriesRuns").replaceChildren(
    ...study.runs.map((r, i) => {
      const card = document.createElement("a");
      card.className = "series-run" + (r.name === runName ? " active" : "");
      card.style.setProperty("--seed-color", palette[i % palette.length]);
      card.href = "?run=" + encodeURIComponent(r.name);
      const title = document.createElement("b");
      title.textContent = "Seed " + r.seed;
      const progress = document.createElement("small");
      progress.textContent = `${r.status.status} · ${short(r.status.step || 0)} / ${short(study.steps)} decisions`;
      const result = document.createElement("small");
      const g = r.final_groups?.find(
        (g) => g.opponent === "initial" && g.map === 0,
      );
      result.textContent = g
        ? `Final map 0: ${g.wins}/${g.games} wins · ${percent(g.win_rate)}`
        : "Fresh-seed final evaluation pending";
      card.append(title, progress, result);
      return card;
    }),
  );
  chart(
    "seriesChart",
    study.runs.map((r, i) => ({
      color: palette[i % palette.length],
      points: r.evaluations.flatMap((e) => {
        const g = e.groups.find((g) => g.opponent === "initial" && g.map === 0);
        return g ? [{ x: e.step, y: g.win_rate }] : [];
      }),
    })),
    { min: 0, max: 1, format: percent },
  );
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
  const final = data.final_evaluation;
  const initial = (final || data.validation_evaluation || latest)?.groups.find(
    (g) => g.opponent === "initial" && g.map === 0,
  );
  $("win").textContent = initial ? percent(initial.win_rate) : "—";
  $("baselineLabel").textContent = c.evaluation_baseline
    ? "Against the fixed baseline · map 0"
    : "Against the initial policy · map 0";
  $("winDetail").textContent = initial
    ? `${initial.wins} / ${initial.games} wins · ${final ? "fresh final seeds · " : data.validation_evaluation ? "expanded evaluation · " : ""}95% interval ${percent(initial.interval[0])}–${percent(initial.interval[1])}`
    : "Fixed evaluation · both player sides";
  const evals = data.evaluations;
  replaceOptions($("winMap"), (c.eval_maps || [0, 1, 2]).map((m) => [String(m), `Map ${m}`]));
  const chartMap = Number($("winMap").value);
  const makeSeries = (opponent, map, color) => ({
    color,
    points: evals.flatMap((e) => {
      const g = e.groups.find((g) => g.opponent === opponent && g.map === map);
      return g
        ? [{ x: e.step, y: g.win_rate, lo: g.interval[0], hi: g.interval[1] }]
        : [];
    }),
  });
  const frozenNames = new Set((latest?.frozen_opponents || c.frozen_opponents || []).map((p) => p.name));
  $("thirdWinLegend").textContent = frozenNames.size ? "Previous trained policies · mean" : "Unseen maps · mean";
  const comparison = {
    color: "#71816a",
    points: evals.flatMap((e) => {
      const gs = e.groups.filter((g) => frozenNames.size
        ? g.map === chartMap && frozenNames.has(g.opponent)
        : !(c.training_maps || [0]).includes(g.map) && g.opponent === "random");
      if (!gs.length) return [];
      return [{
        x: e.step,
        y:
          gs.reduce((n, g) => n + g.wins, 0) /
          Math.max(
            1,
            gs.reduce((n, g) => n + g.games, 0),
          ),
      }];
    }),
  };
  chart(
    "winsChart",
    [
      makeSeries("random", chartMap, "#2846d4"),
      makeSeries("initial", chartMap, "#e87350"),
      comparison,
    ],
    { min: 0, max: 1, format: percent },
  );
  const matrix = final || data.validation_evaluation || latest;
  $("evaluationMatrix").hidden = !matrix;
  if (matrix) {
    $("matrixSuite").textContent = final ? "Final · fresh game seeds" : `Checkpoint · ${short(matrix.step)} decisions`;
    const played = {};
    for (const row of data.metrics) for (const [map, count] of Object.entries(row.map_matches || {})) played[map] = (played[map] || 0) + count;
    $("mapCoverage").textContent = `Training maps: ${(c.training_maps || [0]).join(", ")}.` +
      (Object.keys(played).length ? ` Completed matches: ${Object.entries(played).map(([map,count]) => `map ${map}: ${fmt.format(count)}`).join(" · ")}.` : "");
    const opponentName = (name) => name === "random" ? "Random buttons" : name === "initial" ? "Starting policy" : name.replace(/^trained-s/, "Previous seed ");
    $("evaluationRows").replaceChildren(...matrix.groups.map((g) => {
      const row = document.createElement("tr");
      const trained = (c.training_maps || [0]).includes(g.map);
      const values = [opponentName(g.opponent), `${g.map} · ${trained ? "trained" : "untrained"}`,
        `${g.wins}/${g.games} · ${(100*g.win_rate).toFixed(1)}%`,
        `${(100*g.interval[0]).toFixed(1)}–${(100*g.interval[1]).toFixed(1)}%`, String(g.timeouts)];
      values.forEach((value, i) => {
        const cell = document.createElement("td"); cell.textContent = value;
        if (i === 1 && !trained) cell.className = "untrained-map";
        row.append(cell);
      });
      return row;
    }));
  }
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
  const measured = data.metrics.filter((m) => m.behavior?.schema === 1);
  $("behaviorNotice").textContent = measured.length
    ? "Rates use simulated time for the learning players. Historical opponents and warmup are excluded."
    : c.telemetry_schema === 1
      ? "Gameplay counters will appear after the first rollout."
      : "This run predates gameplay counters. New experiments record activations, arrow hits, deaths and reward components.";
  const rolling = rollingBehavior(measured);
  const activity = (keys, palette, reward = false) =>
    keys.map((key, i) => ({
      color: palette[i],
      points: rolling.map((m) => ({
        x: m.step,
        y: reward
          ? (m.totals[key] * 1000) / Math.max(1, m.decisions)
          : (m.totals[key] * 60) / Math.max(1e-9, m.seconds),
      })),
    }));
  chart(
    "abilitiesChart",
    activity(
      ["dash_activations", "bow_activations", "platform_activations"],
      ["#2846d4", "#e87350", "#71816a"],
    ),
    { min: 0, format: (n) => n.toFixed(1) },
  );
  chart(
    "arrowsChart",
    activity(["arrows_fired", "arrow_hits"], ["#2846d4", "#e87350"]),
    { min: 0, format: (n) => n.toFixed(1) },
  );
  chart(
    "deathsChart",
    activity(
      ["deaths_environment", "deaths_opponent", "deaths_self", "deaths_other"],
      ["#2846d4", "#e87350", "#71816a", "#ab8a5a"],
    ),
    { min: 0, format: (n) => n.toFixed(1) },
  );
  chart(
    "rewardsChart",
    activity(
      ["reward_outcome", "reward_kills", "reward_deaths"],
      ["#2846d4", "#71816a", "#e87350"],
      true,
    ),
    { format: (n) => n.toFixed(2) },
  );
  $("actionNote").textContent =
    c.telemetry_schema === 1
      ? "Last 20 updates, weighted by learning-player decisions. Button presses are attempts, not confirmed activations."
      : "Legacy run: cumulative button presses across both players, including historical opponents.";
  const actionWindow = c.telemetry_schema === 1 ? data.metrics.slice(-20) : [m];
  $("actionBars").replaceChildren(
    ...["jump", "dash", "arrow", "platform"].map((name) => {
      const weight = actionWindow.reduce(
        (n, m) => n + (m.behavior?.decisions || 1),
        0,
      );
      const rate =
        actionWindow.reduce(
          (n, m) =>
            n + (m.action_rates?.[name] || 0) * (m.behavior?.decisions || 1),
          0,
        ) / Math.max(1, weight);
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
function rollingBehavior(rows) {
  const result = [],
    totals = {};
  let decisions = 0,
    seconds = 0;
  const stride = Math.max(1, Math.ceil(rows.length / 600));
  function add(m, sign) {
    decisions += sign * m.behavior.decisions;
    seconds += sign * m.behavior.seconds;
    for (const [k, v] of Object.entries(m.behavior.totals))
      totals[k] = (totals[k] || 0) + sign * v;
  }
  rows.forEach((m, i) => {
    add(m, 1);
    if (i >= 20) add(rows[i - 20], -1);
    if (i % stride === 0 || i === rows.length - 1)
      result.push({ step: m.step, decisions, seconds, totals: { ...totals } });
  });
  return result;
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
        `${g.opponent === "selfplay" ? "Self-play" : g.opponent === "initial" ? "vs starting policy" : g.opponent === "random" ? "vs random" : "vs " + g.opponent.replace(/^trained-s/, "previous seed ")} · map ${g.map}`,
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
  const frame = replay.frames[Math.floor(frameIndex)];
  drawBoplArena($("arena"),frame,replay.frames[0]);
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
function resetRun(name) {
  runName = name;
  data = null;
  selectedEvaluation = null;
  selectedGame = null;
  replay = null;
  playing = false;
  $("video").pause();
  $("emptyReplay").style.display = "flex";
}
$("run").onchange = () => {
  resetRun($("run").value);
  history.replaceState(null, "", "?run=" + encodeURIComponent(runName));
  refresh();
};
$("checkpoint").onchange = () => {
  $("followLatest").checked = false;
  selectEvaluation();
};
$("match").onchange = selectMatch;
$("winMap").onchange = () => { if (data) update(); };
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
  updateSeries();
  if (data) update();
});
refresh();
setInterval(refresh, 5000);
requestAnimationFrame(animate);
