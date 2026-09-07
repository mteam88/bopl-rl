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
  mind = null,
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
  $("explained").textContent =
    m.explained_variance === undefined ? "—" : m.explained_variance.toFixed(3);
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
  const perHead = data.metrics.some((m) => m.entropy_aim !== undefined);
  chart(
    "entropyChart",
    [
      {
        color: "#2846d4",
        points: data.metrics.map((m) => ({ x: m.step, y: m.entropy_aim })),
      },
      {
        color: "#e87350",
        points: data.metrics.map((m) => ({ x: m.step, y: m.entropy_buttons })),
      },
      {
        color: "#71816a",
        points: data.metrics.map((m) => ({ x: m.step, y: m.entropy })),
      },
    ],
    {
      min: 0,
      max: (Math.log(256) + Math.log(16)) * 1.04,
      format: (n) => n.toFixed(1),
      refs: [
        { y: Math.log(256) + Math.log(16), color: "#71816a" },
        ...(perHead
          ? [
              { y: Math.log(256), color: "#2846d4" },
              { y: Math.log(16), color: "#e87350" },
            ]
          : []),
      ],
    },
  );
  $("entropyNote").textContent = perHead
    ? "Dotted lines are uniform-policy maxima. A head parked at its line is not learning; falling entropy alone does not prove better play."
    : "This run reports only joint entropy (max 8.32). Per-head entropy arrives with the native trainer.";
  chart(
    "scheduleChart",
    [
      {
        color: "#2846d4",
        points: data.metrics.map((m) => ({
          x: m.step,
          y: m.entropy_coefficient,
        })),
      },
      {
        color: "#e87350",
        points: data.metrics.map((m) => ({
          x: m.step,
          y: m.learning_rate ?? c.learning_rate,
        })),
      },
    ],
    { log: true, format: (n) => n.toExponential(0) },
  );
  chart(
    "advantageChart",
    [
      {
        color: "#71816a",
        points: data.metrics.map((m) => ({ x: m.step, y: m.advantage_std })),
      },
    ],
    { min: 0, format: (n) => n.toFixed(3) },
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
  const killReward = c.reward?.kill || 0;
  const kills = (t) =>
    t.kills !== undefined
      ? t.kills
      : killReward
        ? t.reward_kills / killReward
        : undefined;
  const perEpisode = (value) => (m) => ({
    x: m.step,
    y:
      m.totals.episodes > 0 ? value(m.totals) / m.totals.episodes : undefined,
  });
  chart(
    "endingsChart",
    [
      { color: "#2846d4", points: rolling.map(perEpisode((t) => t.timeouts)) },
      {
        color: "#e87350",
        points: rolling.map(perEpisode((t) => t.deaths_environment)),
      },
      { color: "#71816a", points: rolling.map(perEpisode(kills)) },
    ],
    { min: 0, format: (n) => n.toFixed(2) },
  );
  chart(
    "accuracyChart",
    [
      {
        color: "#e87350",
        points: rolling.map((m) => ({
          x: m.step,
          y:
            m.totals.arrows_fired > 0
              ? m.totals.arrow_hits / m.totals.arrows_fired
              : undefined,
        })),
      },
    ],
    { min: 0, max: Math.max(0.05, ...rolling.map((m) => m.totals.arrows_fired > 0 ? m.totals.arrow_hits / m.totals.arrows_fired : 0)) * 1.2, format: percent },
  );
  const capSeconds = (c.episode_ticks || 3600) / 60;
  chart(
    "episodeLengthChart",
    [
      {
        color: "#2846d4",
        points: rolling.map(perEpisode((t) => t.ticks / 60)),
      },
    ],
    {
      min: 0,
      max: capSeconds * 1.1,
      format: (n) => Math.round(n) + "s",
      refs: [{ y: capSeconds, color: "#e87350", label: "episode cap" }],
    },
  );
  const window20 = measured.slice(-20);
  const episodeCount = window20.reduce(
    (n, m) => n + m.behavior.totals.episodes,
    0,
  );
  const timeoutCount = window20.reduce(
    (n, m) => n + m.behavior.totals.timeouts,
    0,
  );
  $("truncations").textContent = episodeCount
    ? percent(timeoutCount / episodeCount)
    : "—";
  $("truncationDetail").textContent = episodeCount
    ? `${fmt.format(timeoutCount)} of ${fmt.format(episodeCount)} recent episodes hit the ${Math.round(capSeconds)} s cap`
    : "Timeouts per episode · rolling 20 updates";
  const gaugeSeries = (key) => [
    {
      color: "#2846d4",
      points: (data.insights || []).map((r) => ({
        x: r.step,
        y: r.gauges[key],
      })),
    },
  ];
  chart("bowHoldChart", gaugeSeries("bow_hold_seconds"), {
    min: 0,
    max: 1,
    format: (n) => n.toFixed(2),
    refs: [{ y: 0.84, color: "#71816a", label: "human" }],
  });
  chart("aimChangeChart", gaugeSeries("aim_change_degrees"), {
    min: 0,
    max: 100,
    format: (n) => Math.round(n) + "°",
    refs: [{ y: 2.8, color: "#71816a", label: "human" }],
  });
  const latestGauges = (data.insights || []).at(-1);
  $("gaugeNote").textContent = latestGauges
    ? `Latest checkpoint: hold ${latestGauges.gauges.bow_hold_seconds?.toFixed(2) ?? "—"} s over ${latestGauges.gauges.completed_holds} completed holds · aim change ${latestGauges.gauges.aim_change_degrees?.toFixed(1) ?? "—"}° over ${fmt.format(latestGauges.gauges.decisions)} decisions, from ${latestGauges.games} saved games. Human references: 47-recording demonstration batch (IMITATION.md).`
    : "Measured from saved evaluation replays once the first evaluation completes.";
  updateLadder();
  updateHeatmapOptions();
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
    .filter((p) => Number.isFinite(p.y) && (!options.log || p.y > 0));
  const refValues = (options.refs || []).map((ref) => ref.y);
  const xmax = Math.max(1, ...points.map((p) => p.x)),
    ymin =
      options.log && points.length
        ? Math.min(...points.map((p) => p.y), ...refValues) / 1.5
        : (options.min ?? Math.min(0, ...points.map((p) => p.y))),
    ymax =
      options.max ??
      Math.max(0.001, ...points.map((p) => p.hi ?? p.y), ...refValues) * 1.1;
  const scale = options.log ? Math.log10 : (v) => v,
    lo = scale(Math.max(ymin, options.log ? 1e-12 : -Infinity)),
    hi = scale(ymax);
  const x = (v) => left + ((w - left - right) * v) / xmax,
    y = (v) =>
      top + (h - top - bottom) * (1 - (scale(v) - lo) / (hi - lo || 1));
  ctx.font = '9px "Trebuchet MS",sans-serif';
  ctx.lineWidth = 1;
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 2; i++) {
    const value = options.log
      ? 10 ** (lo + ((hi - lo) * i) / 2)
      : ymin + ((ymax - ymin) * i) / 2,
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
  for (const ref of options.refs || []) {
    const py = y(ref.y);
    ctx.strokeStyle = ref.color || "#868b7c";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(left, py);
    ctx.lineTo(w - right, py);
    ctx.stroke();
    ctx.setLineDash([]);
    if (ref.label) {
      ctx.fillStyle = ref.color || "#868b7c";
      ctx.textAlign = "left";
      ctx.fillText(ref.label, left + 4, py - 6);
    }
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
function updateLadder() {
  const ladder = data.ladder;
  $("ladderSection").hidden = !ladder;
  if (!ladder) return;
  const checkpoints = ladder.ratings
    .filter((r) => r.kind === "checkpoint")
    .sort((a, b) => a.step - b.step);
  const anchors = ladder.ratings.filter((r) => r.kind === "anchor");
  const anchorPalette = ["#71816a", "#e87350", "#ab8a5a", "#80c7e0", "#b99be8", "#868b7c"];
  $("ladderTag").textContent = `${fmt.format(ladder.games)} rated games`;
  chart(
    "ladderChart",
    [
      {
        color: "#2846d4",
        points: checkpoints.map((r) => ({ x: r.step, y: r.rating })),
      },
    ],
    {
      format: (n) => Math.round(n).toString(),
      refs: anchors.map((a, i) => ({
        y: a.rating,
        color: anchorPalette[i % anchorPalette.length],
        label: a.name,
      })),
    },
  );
  $("ladderLegend").replaceChildren(
    ...[
      Object.assign(document.createElement("span"), {
        textContent: "Checkpoints",
      }),
      ...anchors.map((a, i) => {
        const span = document.createElement("span");
        span.textContent = `${a.name} · ${Math.round(a.rating)} · ${a.games} games`;
        span.style.setProperty("--series", anchorPalette[i % anchorPalette.length]);
        return span;
      }),
    ],
  );
  $("ladderLegend").firstChild.style.setProperty("--series", "#2846d4");
}
const heatmapCache = {};
let heatmapRun = "";
function updateHeatmapOptions() {
  const insights = data.insights || [];
  $("heatmapSection").hidden = !insights.length;
  if (!insights.length) return;
  if (heatmapRun !== runName) {
    heatmapRun = runName;
    for (const key of Object.keys(heatmapCache)) delete heatmapCache[key];
  }
  const previous = $("heatmapEval").value;
  replaceOptions(
    $("heatmapEval"),
    insights
      .map((r) => [r.directory, `${fmt.format(r.step)} decisions`])
      .reverse(),
  );
  if (!insights.some((r) => r.directory === previous))
    $("heatmapEval").value = insights.at(-1).directory;
  loadHeatmap();
}
async function loadHeatmap() {
  const directory = $("heatmapEval").value;
  if (!directory) return;
  const key = `${runName}/${directory}`;
  if (!heatmapCache[key]) {
    try {
      const response = await fetch(
        `/api/insights/${encodeURIComponent(runName)}/${encodeURIComponent(directory)}`,
      );
      if (!response.ok) return;
      heatmapCache[key] = await response.json();
    } catch {
      return;
    }
  }
  if ($("heatmapEval").value !== directory) return;
  const maps = Object.keys(heatmapCache[key].maps).sort(
    (a, b) => Number(a) - Number(b),
  );
  replaceOptions($("heatmapMap"), maps.map((m) => [m, `Map ${m}`]));
  drawHeatmap();
}
function drawHeatmap() {
  const key = `${runName}/${$("heatmapEval").value}`;
  const insight = heatmapCache[key];
  const cell = insight?.maps[$("heatmapMap").value];
  const canvas = $("heatmap"),
    ctx = canvas.getContext("2d"),
    w = canvas.width,
    h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!cell) return;
  const grid = insight.grid;
  const worldW = grid.w * grid.cell,
    worldH = grid.h * grid.cell;
  const px = (x) => ((x - grid.x0) / worldW) * w;
  const py = (y) => h - ((y - grid.y0) / worldH) * h;
  ctx.fillStyle = "#18263a";
  ctx.fillRect(0, 0, w, h);
  for (const outline of cell.outline || []) {
    const corners = [
      [-1, -1],
      [1, -1],
      [1, 1],
      [-1, 1],
    ].map(([a, b]) => [
      px(outline.x + a * outline.rx + b * outline.ux),
      py(outline.y + a * outline.ry + b * outline.uy),
    ]);
    ctx.beginPath();
    corners.forEach(([cx2, cy2], i) =>
      i ? ctx.lineTo(cx2, cy2) : ctx.moveTo(cx2, cy2),
    );
    ctx.closePath();
    ctx.lineJoin = "round";
    ctx.lineWidth = Math.max(2, ((outline.radius * 2) / worldW) * w);
    ctx.strokeStyle = "#33475f";
    ctx.fillStyle = "#33475f";
    ctx.stroke();
    ctx.fill();
  }
  const peak = Math.max(1, ...cell.occupancy.map(([, count]) => count));
  const cw = (grid.cell / worldW) * w,
    ch = (grid.cell / worldH) * h;
  for (const [index, count] of cell.occupancy) {
    const column = index % grid.w,
      row = Math.floor(index / grid.w);
    ctx.fillStyle = `rgba(244,217,63,${(0.85 * Math.sqrt(count / peak)).toFixed(3)})`;
    ctx.fillRect(
      px(grid.x0 + column * grid.cell),
      py(grid.y0 + (row + 1) * grid.cell),
      cw + 0.5,
      ch + 0.5,
    );
  }
  ctx.strokeStyle = "#ef5d3c";
  ctx.lineWidth = 2.5;
  for (const [x, y] of cell.deaths) {
    const cx2 = px(x),
      cy2 = py(y);
    ctx.beginPath();
    ctx.moveTo(cx2 - 6, cy2 - 6);
    ctx.lineTo(cx2 + 6, cy2 + 6);
    ctx.moveTo(cx2 - 6, cy2 + 6);
    ctx.lineTo(cx2 + 6, cy2 - 6);
    ctx.stroke();
  }
  $("heatmapNote").textContent = `${cell.games} saved game${cell.games === 1 ? "" : "s"} on map ${$("heatmapMap").value} at this checkpoint · ${cell.deaths.length} deaths marked. Occupancy is the policy player only; platforms drawn for reference.`;
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
  mind = null;
  $("mindPanel").hidden = true;
  if (selectedGame.mind) {
    fetch(fileUrl(selectedEvaluation, selectedGame.mind))
      .then((r) => (r.ok ? r.json() : null))
      .then((value) => {
        if (requested !== selectedGame || !value) return;
        mind = value;
        $("mindPanel").hidden = false;
        drawMind();
      })
      .catch(() => {});
  }
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
function mindIndex() {
  if (!mind) return -1;
  // Legacy visuals record the action that led into a frame; native visuals
  // (actions_timing=observation-before-action) record the action taken from it.
  const offset = replay?.actions_timing === "observation-before-action" ? 0 : 1;
  return Math.min(
    mind.decisions.length - 1,
    Math.max(0, Math.floor(frameIndex) - offset),
  );
}
function drawMind() {
  if (!mind) return;
  const decisions = mind.decisions,
    cursor = mindIndex();
  const canvas = $("mindValue"),
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
    top = 6,
    bottom = 14;
  const values = decisions.map((d) => d.value);
  const vmax = Math.max(0.1, ...values.map(Math.abs)) * 1.1;
  const x = (i) =>
    left + ((w - left - right) * i) / Math.max(1, decisions.length - 1);
  const y = (v) => top + ((h - top - bottom) * (1 - v / vmax)) / 2;
  ctx.font = '9px "Trebuchet MS",sans-serif';
  ctx.textBaseline = "middle";
  ctx.lineWidth = 1;
  for (const level of [-vmax / 1.1, 0, vmax / 1.1]) {
    ctx.strokeStyle = level === 0 ? "#c9cdbc" : "#dfe2d5";
    ctx.beginPath();
    ctx.moveTo(left, y(level));
    ctx.lineTo(w - right, y(level));
    ctx.stroke();
    ctx.fillStyle = "#868b7c";
    ctx.textAlign = "right";
    ctx.fillText(level.toFixed(1), left - 7, y(level));
  }
  ctx.strokeStyle = "#2846d4";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  values.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
  ctx.stroke();
  if (cursor >= 0) {
    ctx.strokeStyle = "#e87350";
    ctx.beginPath();
    ctx.moveTo(x(cursor), top);
    ctx.lineTo(x(cursor), h - bottom);
    ctx.stroke();
  }
  ctx.fillStyle = "#868b7c";
  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  ctx.fillText("V(s)", left, h);
  const decision = decisions[Math.max(0, cursor)];
  if (decision) {
    $("mindReadout").textContent =
      `V(s) ${decision.value.toFixed(2)} at ${duration(decision.t / 60)}`;
    drawDistribution(
      "mindButtons",
      decision.buttons,
      decision.action?.[1],
      (i) =>
        ["J", "D", "A", "P"].filter((_, bit) => i & (1 << bit)).join("") ||
        "·",
    );
    // Aim index 0 is "no joystick direction", not zero degrees; bins 1..255
    // map to (index-1)*360/255 degrees. Bin 0 is drawn apart in gray.
    drawDistribution("mindAim", decision.aim, decision.action
      ? Math.round((decision.action[0] * decision.aim.length) / 256)
      : undefined, (i) => (i === 0 ? "∅" : ""), 0);
  }
}
function drawDistribution(id, probabilities, chosen, label, special) {
  const canvas = $(id),
    r = canvas.getBoundingClientRect(),
    dpr = devicePixelRatio || 1;
  canvas.width = Math.round(r.width * dpr);
  canvas.height = Math.round(r.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const w = r.width,
    h = r.height,
    bottom = label ? 12 : 4;
  const peak = Math.max(1e-6, ...probabilities);
  const bar = w / probabilities.length;
  probabilities.forEach((p, i) => {
    const height = (h - bottom - 4) * (p / peak);
    ctx.fillStyle =
      i === chosen ? "#e87350" : i === special ? "#868b7c" : "#2846d4";
    ctx.fillRect(i * bar + 0.5, h - bottom - height, Math.max(1, bar - 1), height);
  });
  if (label) {
    ctx.fillStyle = "#868b7c";
    ctx.font = '8px "Trebuchet MS",sans-serif';
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    probabilities.forEach((_, i) => {
      if ((bar >= 14 || i === chosen || i === special) && label(i))
        ctx.fillText(label(i), i * bar + bar / 2, h - 1);
    });
  }
}
function drawFrame() {
  if (!replay) return;
  const frame = replay.frames[Math.floor(frameIndex)];
  drawBoplArena($("arena"),frame,replay.frames[0]);
  if (mind && !$("mindPanel").hidden) drawMind();
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
  mind = null;
  playing = false;
  $("video").pause();
  $("mindPanel").hidden = true;
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
$("heatmapEval").onchange = loadHeatmap;
$("heatmapMap").onchange = drawHeatmap;
window.addEventListener("resize", () => {
  updateSeries();
  if (data) update();
});
refresh();
setInterval(refresh, 5000);
requestAnimationFrame(animate);
