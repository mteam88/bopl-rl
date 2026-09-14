"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const colors = ["#d6ff32", "#3ee0c5", "#ff7a45", "#9ec0ff", "#ff5d73", "#b6e05a"];
  const openSweeps = new Set();
  let ladder = null, data = null, sweep = null;
  let selectedEvaluation = null, selectedGame = null, replay = null;
  let playing = false, frameIndex = 0, lastTick = 0, view = "ladder";

  function route() {
    const path = location.pathname;
    if (path.startsWith("/run/")) return { view: "run", name: decodeURIComponent(path.slice(5)) };
    if (path.startsWith("/sweep/")) return { view: "sweep", name: decodeURIComponent(path.slice(7)) };
    return { view: "ladder" };
  }
  function fmt(n, d = 0) {
    return n == null || Number.isNaN(n) ? "—" : Number(n).toFixed(d);
  }
  function elo(n) { return fmt(n, 0); }
  function ci(interval) {
    if (!interval) return "—";
    return `${elo(interval[0])}–${elo(interval[1])}`;
  }
  const mem = new Map();
  const inflight = new Map();
  let loadToken = 0, replayLoading = false;
  function apply(url, payload) {
    const r = route();
    if (url === "/api/ladder" && r.view === "ladder") {
      ladder = payload;
      renderLadder();
    } else if (r.view === "run" && url === `/api/run/${encodeURIComponent(r.name)}`) {
      data = payload;
      renderRun();
    } else if (r.view === "sweep" && url === `/api/sweep/${encodeURIComponent(r.name)}`) {
      sweep = payload;
      renderSweep();
    }
  }
  async function readNdjson(response, onPartial) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let replay = null;
    const frames = [];
    const consume = (line) => {
      if (!line) return;
      const obj = JSON.parse(line);
      if (replay == null) {
        replay = obj;
        replay.frames = frames;
      } else {
        frames.push(obj);
      }
    };
    while (true) {
      const { done, value } = await reader.read();
      buf += decoder.decode(value || new Uint8Array(), { stream: !done });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        consume(buf.slice(0, nl));
        buf = buf.slice(nl + 1);
        if (replay && frames.length === 1) onPartial?.(replay);
        else if (replay && frames.length % 48 === 0) onPartial?.(replay);
      }
      if (done) break;
    }
    if (buf.trim()) consume(buf.trim());
    onPartial?.(replay);
    return replay;
  }
  async function json(url, { onUpdate } = {}) {
    const hit = mem.get(url);
    if (onUpdate && hit?.data) onUpdate(hit.data);
    const flight = inflight.get(url);
    if (flight) {
      if (onUpdate) flight.listeners.push(onUpdate);
      if (hit?.data) return hit.data;
      return flight.promise;
    }
    const headers = { Accept: "application/json, application/x-ndjson" };
    if (hit?.etag) headers["If-None-Match"] = hit.etag;
    const listeners = onUpdate ? [onUpdate] : [];
    const promise = fetch(url, { headers }).then(async (r) => {
      if (r.status === 304 && hit) return hit.data;
      if (!r.ok) throw Error(`${url} ${r.status}`);
      const ctype = r.headers.get("content-type") || "";
      const etag = r.headers.get("ETag");
      const emit = (partial) => {
        mem.set(url, { data: partial, etag });
        for (const fn of listeners) fn(partial);
      };
      let payload;
      if (ctype.includes("ndjson")) payload = await readNdjson(r, emit);
      else payload = await r.json();
      mem.set(url, { data: payload, etag });
      return payload;
    }).finally(() => inflight.delete(url));
    inflight.set(url, { promise, listeners });
    if (hit?.data) {
      promise.then((payload) => { if (payload !== hit.data) apply(url, payload); }).catch(() => {});
      return hit.data;
    }
    return promise;
  }
  function fileUrl(evaluation, file) {
    return `/runs/${encodeURIComponent(data.name)}/evaluations/${evaluation.directory}/${encodeURIComponent(file)}`;
  }
  function duration(seconds) {
    const s = Math.max(0, Math.floor(seconds));
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }
  function spark(canvas, points) {
    const r = canvas.getBoundingClientRect(), dpr = devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(r.width * dpr));
    canvas.height = Math.max(1, Math.round(r.height * dpr));
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const w = r.width, h = r.height;
    if (!points || points.length < 2) return;
    const xs = points.map((p) => p.steps), ys = points.map((p) => p.rating);
    const x0 = Math.min(...xs), x1 = Math.max(...xs) || 1;
    const y0 = Math.min(...ys), y1 = Math.max(...ys);
    const pad = 2;
    const x = (v) => pad + ((w - 2 * pad) * (v - x0)) / (x1 - x0 || 1);
    const y = (v) => h - pad - ((h - 2 * pad) * (v - y0)) / (y1 - y0 || 1);
    ctx.strokeStyle = "#d6ff32";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    points.forEach((p, i) => (i ? ctx.lineTo(x(p.steps), y(p.rating)) : ctx.moveTo(x(p.steps), y(p.rating))));
    ctx.stroke();
  }
  function chart(canvas, series, opts = {}) {
    const r = canvas.getBoundingClientRect(), dpr = devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(r.width * dpr));
    canvas.height = Math.max(1, Math.round(r.height * dpr));
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const w = r.width, h = r.height, left = 44, right = 8, top = 8, bottom = 18;
    const pts = series.filter((p) => p.y != null);
    const secondary = (opts.secondary || []).filter((p) => p.y != null);
    if (!pts.length && !secondary.length) return;
    const marks = (opts.mark || []).filter((p) => p.y != null);
    const xs = [...pts, ...secondary].map((p) => p.x);
    const ys = [...pts, ...secondary, ...marks].flatMap((p) => [p.y, p.lo, p.hi].filter((v) => v != null));
    const x0 = Math.min(...xs), x1 = Math.max(...xs) || 1;
    const y0 = Math.min(...ys), y1 = Math.max(...ys);
    const x = (v) => left + ((w - left - right) * (v - x0)) / (x1 - x0 || 1);
    const y = (v) => top + ((h - top - bottom) * (1 - (v - y0) / (y1 - y0 || 1)));
    ctx.strokeStyle = "#262b33";
    ctx.fillStyle = "#8b9283";
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const t of [y0, (y0 + y1) / 2, y1]) {
      ctx.beginPath(); ctx.moveTo(left, y(t)); ctx.lineTo(w - right, y(t)); ctx.stroke();
      ctx.fillText(t.toFixed(0), left - 6, y(t));
    }
    if (secondary.length) {
      // A cheaper, differently-sourced estimate (in-run bots-only fits): dashed and dim,
      // so a source switch along the curve is not read as a jump in strength.
      ctx.strokeStyle = "#8b9283";
      ctx.lineWidth = 1.2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      secondary.forEach((p, i) => (i ? ctx.lineTo(x(p.x), y(p.y)) : ctx.moveTo(x(p.x), y(p.y))));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#8b9283";
      for (const p of secondary) { ctx.beginPath(); ctx.arc(x(p.x), y(p.y), 2, 0, 6.3); ctx.fill(); }
    }
    if (pts.some((p) => p.lo != null)) {
      ctx.fillStyle = "#d6ff3222";
      ctx.beginPath();
      pts.forEach((p, i) => (i ? ctx.lineTo(x(p.x), y(p.hi)) : ctx.moveTo(x(p.x), y(p.hi))));
      for (let i = pts.length - 1; i >= 0; i--) ctx.lineTo(x(pts[i].x), y(pts[i].lo));
      ctx.closePath(); ctx.fill();
    }
    ctx.strokeStyle = opts.color || "#d6ff32";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    pts.forEach((p, i) => (i ? ctx.lineTo(x(p.x), y(p.y)) : ctx.moveTo(x(p.x), y(p.y))));
    ctx.stroke();
    if (pts.length === 1) { ctx.fillStyle = ctx.strokeStyle; ctx.beginPath(); ctx.arc(x(pts[0].x), y(pts[0].y), 3, 0, 6.3); ctx.fill(); }
    ctx.fillStyle = opts.color || "#d6ff32";
    for (const p of pts) { ctx.beginPath(); ctx.arc(x(p.x), y(p.y), 2.5, 0, 6.3); ctx.fill(); }
    if (marks.length) {
      ctx.fillStyle = "#3ee0c5";
      for (const p of marks) {
        ctx.beginPath(); ctx.arc(x(p.x), y(p.y), 3.5, 0, 6.3); ctx.fill();
      }
    }
  }

  function show(name) {
    view = name;
    $("ladderView").hidden = name !== "ladder";
    $("runView").hidden = name !== "run";
    $("sweepView").hidden = name !== "sweep";
    document.querySelectorAll("nav a").forEach((a) => a.classList.toggle("active", a.dataset.nav === (name === "run" ? "ladder" : name)));
  }

  function kindClass(row) {
    if (row.kind === "pool") return "pool";
    if (row.kind === "human") return "human";
    if (row.unrated) return "unrated";
    return "";
  }

  function renderLadder() {
    show("ladder");
    if (!ladder) return;
    const pool = ladder.pool || {};
    $("poolIdentity").innerHTML = pool.sha8
      ? `<b>pool</b> ${pool.sha8}<br><b>sim</b> ${(pool.sim_sha256 || "").slice(0, 8)}<br><b>maps</b> ${(pool.maps || []).join(",")}${
          (ladder.human && ladder.human.caveats)
            ? "<br><b>human</b> " + ladder.human.caveats.join(" ")
            : ""}`
      : "No pool registered (--pool)";
    $("ladderTag").textContent = `${(ladder.rows || []).length} runs · ${(ladder.members || []).length} members`;
    const hidePool = $("hidePool").checked;
    const sort = $("sort").value;
    const items = [];
    for (const m of ladder.members || []) {
      if (hidePool && m.kind === "pool") continue;
      items.push(m);
    }
    const bySweep = {};
    for (const row of ladder.rows || []) {
      if (row.sweep) (bySweep[row.sweep] ||= []).push(row);
      else items.push(row);
    }
    const cmp = (a, b) => {
      if (sort === "name") return String(a.name).localeCompare(String(b.name));
      if (sort === "kind") return String(a.kind).localeCompare(String(b.kind)) || (b.rating || -1e9) - (a.rating || -1e9);
      return (b.rating ?? -1e9) - (a.rating ?? -1e9);
    };
    items.sort(cmp);
    const body = $("ladderRows");
    body.replaceChildren();
    const addRow = (row, href) => {
      const tr = document.createElement("tr");
      tr.className = [kindClass(row), row.game_mismatch ? "mismatch" : ""].join(" ");
      const name = document.createElement("td");
      if (href) {
        const a = document.createElement("a");
        a.href = href; a.textContent = row.name;
        name.append(a);
      } else name.textContent = row.name;
      const kind = document.createElement("td");
      kind.textContent = row.member_kind || row.kind;
      const rating = document.createElement("td");
      rating.className = "num";
      rating.textContent = row.unrated ? "unrated" : elo(row.rating);
      const interval = document.createElement("td");
      interval.className = "num";
      if (row.unrated) {
        interval.className = "num cmd-cell";
        interval.textContent = row.rate_command || "unrated on v5";
      } else interval.textContent = ci(row.interval);
      const sparkTd = document.createElement("td");
      if (row.spark && row.spark.length > 1) {
        const c = document.createElement("canvas");
        c.className = "spark";
        sparkTd.append(c);
        requestAnimationFrame(() => spark(c, row.spark));
      }
      const flags = document.createElement("td");
      flags.className = "flags";
      for (const f of row.flags || []) {
        const s = document.createElement("span"); s.textContent = f; s.className = "warn"; flags.append(s);
      }
      tr.append(name, kind, rating, interval, sparkTd, flags);
      body.append(tr);
    };
    for (const row of items) {
      const href = row.kind === "run" || row.kind === "sweep-trial" ? `/run/${encodeURIComponent(row.name)}` : null;
      addRow(row, href);
    }
    for (const [sweepName, rows] of Object.entries(bySweep).sort()) {
      const group = document.createElement("tr");
      group.className = "group";
      const cell = document.createElement("td");
      cell.colSpan = 6;
      cell.textContent = `sweep ${sweepName} · ${rows.length} trials`;
      group.append(cell);
      body.append(group);
      const kids = rows.slice().sort(cmp);
      group.onclick = () => {
        const open = !group.classList.contains("open");
        group.classList.toggle("open", open);
        if (open) openSweeps.add(sweepName); else openSweeps.delete(sweepName);
        if (!open) {
          kids.forEach((row) => { if (row._tr) row._tr.hidden = true; });
          return;
        }
        if (kids[0]._tr) {
          kids.forEach((row) => { row._tr.hidden = false; });
          return;
        }
        let after = group;
        for (const row of kids) {
          addRow(row, `/run/${encodeURIComponent(row.name)}`);
          row._tr = body.lastElementChild;
          after.after(row._tr);
          after = row._tr;
        }
      };
      if (openSweeps.has(sweepName)) group.click();
    }
    $("ladderNote").textContent = "Pool members are frozen. Runs show IQM of the last 3 rated evaluation checkpoints. Unrated rows need the one-line rate-runs command (hover the CI). ≠sim means the games were not played on the pool's sim.";
    $("connectionBox").classList.add("online");
    $("connection").textContent = "Live";
    $("updated").textContent = new Date().toISOString().replace("T", " ").slice(0, 19);
    const firstSweep = (ladder.rows || []).find((r) => r.sweep);
    $("sweepNav").hidden = !firstSweep;
    if (firstSweep) $("sweepNav").href = `/sweep/${encodeURIComponent(firstSweep.sweep)}`;
  }

  function renderWinTable() {
    const report = data?.evaluations?.at(-1);
    if (!report) { $("winRows").replaceChildren(); return; }
    const map = Number($("winMap").value);
    $("winRows").replaceChildren(...(report.groups || []).filter((g) => g.map === map).map((g) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${g.opponent}</td><td class="num">${(100 * g.win_rate).toFixed(0)}%</td><td class="num">${g.games}</td>`;
      return tr;
    }));
    const metrics = report.flag_metrics || {};
    $("behavior").replaceChildren();
    for (const [k, v] of Object.entries(metrics)) {
      if (k === "games") continue;
      const dt = document.createElement("dt"); dt.textContent = k.replaceAll("_", " ");
      const dd = document.createElement("dd"); dd.textContent = typeof v === "number" ? v.toFixed(2) : v;
      $("behavior").append(dt, dd);
    }
  }

  function renderRun() {
    show("run");
    if (!data) return;
    $("runTitle").textContent = data.name;
    $("runEyebrow").textContent = data.sweep ? `SWEEP ${data.sweep} · ${data.trial}` : "RUN";
    const iqm = data.iqm;
    $("runElo").textContent = data.unrated ? "—" : elo(iqm?.score);
    $("runInterval").textContent = data.unrated
      ? "unrated on this pool"
      : `IQM3  ${ci(iqm?.interval)}${data.game_mismatch ? "  ·  sim mismatch" : ""}`;
    const cmd = $("rateCommand");
    if (data.unrated && data.rate_command) {
      cmd.hidden = false;
      cmd.textContent = data.rate_command;
    } else cmd.hidden = true;
    $("runFlags").replaceChildren(...(data.flags || []).map((f) => {
      const s = document.createElement("span"); s.textContent = f; s.className = "warn"; return s;
    }));
    const status = data.status || {};
    $("runStats").innerHTML = [
      ["status", status.status || "—"],
      ["steps", (status.learner_steps || 0).toLocaleString()],
      ["evals", String((data.evaluations || []).length)],
      ["rated", String((data.ratings || []).length)],
    ].map(([k, v]) => `<article><p>${k}</p><strong>${v}</strong></article>`).join("");
    const point = (r) => ({ x: r.learner_steps, y: r.rating, lo: r.interval[0], hi: r.interval[1] });
    const ratings = data.ratings || [];
    const full = ratings.filter((r) => (r.source || "band") !== "evaluation");
    const cheap = ratings.filter((r) => r.source === "evaluation");
    chart($("eloChart"), full.map(point), { secondary: cheap.map(point) });
    $("eloCaption").textContent = data.unrated
      ? "No rating files for this pool yet."
      : `Solid: ${full.length} checkpoint${full.length === 1 ? "" : "s"} rated by 480 pool games (band = 95% bootstrap CI). `
        + `Dashed grey: ${cheap.length} in-run fits from the diagnostic games vs the 5 scripted pool members only; `
        + "these run 100-300 Elo high for bot-heavy curricula and are not comparable to the solid points.";
    $("ratingRows").replaceChildren(...(data.ratings || []).map((r) => {
      const tr = document.createElement("tr");
      if (r.game_mismatch) tr.className = "mismatch";
      tr.innerHTML = `<td>${r.checkpoint}</td><td class="num">${(r.learner_steps || 0).toLocaleString()}</td><td class="num">${elo(r.rating)}</td><td class="num">${ci(r.interval)}</td><td>${r.source || "band"}</td>`;
      return tr;
    }));
    const maps = [...new Set((data.evaluations.at(-1)?.groups || []).map((g) => g.map))];
    $("winMap").replaceChildren(...maps.map((m) => {
      const o = document.createElement("option"); o.value = m; o.textContent = `map ${m}`; return o;
    }));
    $("checkpoint").replaceChildren(...(data.evaluations || []).map((e) => {
      const o = document.createElement("option"); o.value = e.directory; o.textContent = e.directory; return o;
    }));
    if (data.evaluations?.length) $("checkpoint").value = data.evaluations.at(-1).directory;
    if (data.sweep) {
      $("sweepNav").hidden = false;
      $("sweepNav").href = `/sweep/${encodeURIComponent(data.sweep)}`;
    }
    renderWinTable();
    selectEvaluation();
    $("connectionBox").classList.add("online");
    $("connection").textContent = "Live";
  }

  function renderSweep() {
    show("sweep");
    if (!sweep) return;
    $("sweepTitle").textContent = sweep.name;
    const scored = (sweep.trials || []).filter((t) => t.score != null);
    const best = scored.reduce((b, t) => (b == null || t.score > b.score ? t : b), null);
    $("sweepLede").textContent = `${(sweep.trials || []).length} trials · ${scored.length} scored${best ? ` · best ${best.id} at ${fmt(best.score, 0)}` : ""}`;
    const params = Object.keys(((sweep.trials || []).find((t) => t.hypers) || {}).hypers || {});
    $("sweepHead").innerHTML = ["trial", ...params, "cost", "score", "IQM3", "flags"]
      .map((h) => `<th>${h}</th>`).join("");
    $("sweepRows").replaceChildren(...(sweep.trials || []).map((t) => {
      const tr = document.createElement("tr");
      const cells = [`<td><a href="/run/${encodeURIComponent(sweep.name + "-" + t.id)}">${t.id}</a></td>`];
      for (const p of params) {
        const v = (t.hypers || {})[p];
        cells.push(`<td class="num">${typeof v === "number" ? v.toPrecision(3) : (v ?? "")}</td>`);
      }
      cells.push(`<td class="num">${t.cost == null ? "—" : Number(t.cost).toExponential(2)}</td>`);
      cells.push(`<td class="num">${fmt(t.score, 1)}</td>`);
      cells.push(`<td class="num">${t.unrated ? "unrated" : elo(t.rating)}</td>`);
      cells.push(`<td>${(t.flags || []).join(" ")}</td>`);
      tr.innerHTML = cells.join("");
      return tr;
    }));
    // x = position in launch order (sweep.json order), line = best score so far, dots = each trial.
    let running = null;
    const points = scored.map((t, i) => ({ x: i, y: t.score }));
    const bestSoFar = scored.map((t, i) => ({ x: i, y: (running = running == null ? t.score : Math.max(running, t.score)) }));
    chart($("scoreChart"), bestSoFar, { color: "#9ec0ff", mark: points });
    $("connectionBox").classList.add("online");
    $("connection").textContent = "Live";
  }

  async function selectEvaluation() {
    const dir = $("checkpoint").value;
    let report = (data?.evaluations || []).find((e) => e.directory === dir);
    if (data && dir && report && !('games' in report)) {
      report = await json(`/api/run/${encodeURIComponent(data.name)}/evaluation/${encodeURIComponent(dir)}`);
      const i = data.evaluations.findIndex((e) => e.directory === dir);
      if (i >= 0) data.evaluations[i] = report;
    }
    selectedEvaluation = report;
    if (!selectedEvaluation) return;
    const games = (selectedEvaluation.games || []).filter((g) => g.visual);
    $("match").replaceChildren(...games.map((g) => {
      const o = document.createElement("option");
      o.value = g.name;
      o.textContent = `${g.opponent} · map ${g.map}`;
      return o;
    }));
    selectMatch();
  }
  function prefetchVisual(evaluation, game) {
    if (!data || !evaluation || !game?.visual) return;
    json(fileUrl(evaluation, game.visual)).catch(() => {});
  }
  async function selectMatch() {
    selectedGame = selectedEvaluation?.games.find((g) => g.name === $("match").value);
    playing = false;
    if (!selectedGame?.visual) {
      $("play").disabled = true;
      $("replayTag").textContent = "No replay";
      $("emptyReplay").style.display = "";
      return;
    }
    const token = ++loadToken;
    const url = fileUrl(selectedEvaluation, selectedGame.visual);
    const games = (selectedEvaluation.games || []).filter((g) => g.visual);
    const i = games.findIndex((g) => g.name === selectedGame.name);
    if (i >= 0 && games[i + 1]) prefetchVisual(selectedEvaluation, games[i + 1]);
    $("replayTag").textContent = "Loading…";
    let drawn = false;
    const adopt = (value) => {
      if (token !== loadToken || !value?.frames) return;
      replay = value;
      $("emptyReplay").style.display = "none";
      $("play").disabled = false;
      $("scrub").max = Math.max(0, replay.frames.length - 1);
      $("replayTag").textContent = selectedGame.timeout ? "Time limit" : selectedGame.win ? "Policy wins" : "Policy loses";
      $("download").hidden = !selectedGame.replay;
      if (selectedGame.replay) {
        $("download").href = fileUrl(selectedEvaluation, selectedGame.replay);
        $("download").download = selectedGame.replay;
      }
      if (!drawn && replay.frames.length) {
        drawn = true;
        frameIndex = 0;
        $("play").textContent = "▶";
        drawFrame();
      }
    };
    replayLoading = true;
    try {
      const value = await json(url, { onUpdate: adopt });
      if (token === loadToken) {
        if (value) adopt(value);
        else $("replayTag").textContent = "Replay unavailable";
      }
    } catch {
      if (token === loadToken) $("replayTag").textContent = "Replay unavailable";
    } finally {
      if (token === loadToken) replayLoading = false;
    }
  }
  function drawFrame() {
    if (!replay) return;
    const frame = replay.frames[Math.floor(frameIndex)];
    if (!frame) return;
    window.drawBoplArena($("arena"), frame, replay.frames[0]);
    $("scrub").value = Math.floor(frameIndex);
    $("replayTime").textContent = `${duration(frame.t / 60)} / ${duration(replay.frames.at(-1).t / 60)}`;
    $("players").replaceChildren(...frame.players.map((p) => {
      const box = document.createElement("div");
      box.className = "player";
      box.innerHTML = `<b>P${p.id}${p.id === selectedGame.side + 1 ? " · policy" : ""}</b> ${p.kills} kills`;
      box.style.borderColor = colors[(p.id - 1) % 6];
      return box;
    }));
  }
  function tick(now) {
    if (playing && replay) {
      const dt = Math.min(0.05, (now - lastTick) / 1000);
      frameIndex = Math.min(replay.frames.length - 1, frameIndex + dt * 60 * Number($("speed").value));
      if (frameIndex >= replay.frames.length - 1 && !replayLoading) playing = false;
      drawFrame();
    }
    lastTick = now;
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  async function load() {
    $("notice").textContent = "";
    const r = route();
    try {
      if (r.view === "ladder") {
        ladder = await json("/api/ladder");
        renderLadder();
      } else if (r.view === "run") {
        data = await json(`/api/run/${encodeURIComponent(r.name)}`);
        renderRun();
      } else {
        sweep = await json(`/api/sweep/${encodeURIComponent(r.name)}`);
        $("sweepNav").hidden = false;
        $("sweepNav").href = `/sweep/${encodeURIComponent(r.name)}`;
        renderSweep();
      }
    } catch (error) {
      $("notice").textContent = error.message;
      $("connection").textContent = "Error";
    }
  }

  $("sort").onchange = $("hidePool").onchange = () => { if (ladder) renderLadder(); };
  $("winMap").onchange = renderWinTable;
  $("checkpoint").onchange = selectEvaluation;
  $("match").onchange = selectMatch;
  $("play").onclick = () => { playing = !playing; $("play").textContent = playing ? "❚❚" : "▶"; };
  $("scrub").oninput = () => { frameIndex = Number($("scrub").value); drawFrame(); };
  addEventListener("popstate", load);
  document.addEventListener("click", (event) => {
    const a = event.target.closest("a");
    if (!a || event.metaKey || event.ctrlKey) return;
    const href = a.getAttribute("href");
    if (!href || !href.startsWith("/") || href.startsWith("/runs/") || href === "/sweep/") return;
    event.preventDefault();
    history.pushState(null, "", href);
    load();
  });
  load();
})();
