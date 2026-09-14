"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const colors = ["#d6ff32", "#3ee0c5", "#ff7a45", "#9ec0ff", "#ff5d73", "#b6e05a"];
  let ladder = null, index = null, data = null, sweep = null;
  let selectedEvaluation = null, selectedGame = null, replay = null;
  let playing = false, frameIndex = 0, lastTick = 0, view = "index";
  let familyFilter = "";

  function route() {
    const path = location.pathname;
    if (path.startsWith("/run/")) return { view: "run", name: decodeURIComponent(path.slice(5)) };
    if (path.startsWith("/sweep/")) return { view: "sweep", name: decodeURIComponent(path.slice(7)) };
    if (path.startsWith("/ladder")) return { view: "ladder" };
    return { view: "index" };
  }
  function fmt(n, d = 0) {
    return n == null || Number.isNaN(n) ? "—" : Number(n).toFixed(d);
  }
  function elo(n) { return fmt(n, 0); }
  function ci(interval) {
    if (!interval) return "—";
    return `${elo(interval[0])}–${elo(interval[1])}`;
  }
  function wld(row) {
    if (row.wins == null) return "—";
    return `${row.wins}–${row.losses}–${row.draws}`;
  }
  function poolScore(row, label) {
    return (row.pools || []).find((p) => p.label === label);
  }
  const mem = new Map();
  const inflight = new Map();
  let loadToken = 0, replayLoading = false;
  function apply(url, payload) {
    const r = route();
    if (url === "/api/index" && r.view === "index") {
      index = payload;
      renderIndex();
    } else if (url === "/api/ladder" && r.view === "ladder") {
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
    ctx.font = "10px ui-monospace, monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const t of [y0, (y0 + y1) / 2, y1]) {
      ctx.beginPath(); ctx.moveTo(left, y(t)); ctx.lineTo(w - right, y(t)); ctx.stroke();
      ctx.fillText(t.toFixed(0), left - 6, y(t));
    }
    if (secondary.length) {
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
    $("indexView").hidden = name !== "index";
    $("ladderView").hidden = name !== "ladder";
    $("runView").hidden = name !== "run";
    $("sweepView").hidden = name !== "sweep";
    document.querySelectorAll("nav a").forEach((a) => {
      const nav = a.dataset.nav;
      a.classList.toggle("active", nav === name || (name === "run" && nav === "index"));
    });
  }

  function live() {
    $("connectionBox").classList.add("online");
    $("connection").textContent = "Live";
    $("updated").textContent = new Date().toISOString().replace("T", " ").slice(0, 19);
  }

  function setSweepNav(name) {
    if (!name) {
      const first = (index?.families || []).find((f) => f.sweep) || (index?.runs || []).find((r) => r.sweep);
      name = first?.sweep || first?.name;
    }
    $("sweepNav").hidden = !name;
    if (name) $("sweepNav").href = `/sweep/${encodeURIComponent(name)}`;
  }

  function renderIndex() {
    show("index");
    if (!index) return;
    const runs = index.runs || [];
    const families = index.families || [];
    const stats = index.joint?.stats;
    $("indexIdentity").innerHTML = [
      `<b>runs</b> ${runs.length}`,
      `<b>families</b> ${families.length}`,
      stats ? `<b>joint</b> ${stats.participants} players · ${stats.games} games · ${fmt(stats.elapsed, 2)}s` : "<b>joint</b> fitting…",
      (index.pools || []).map((p) => `<b>pool ${p.label}</b> ${p.sha8}`).join("<br>"),
    ].filter(Boolean).join("<br>");
    const famSel = $("filterFamily");
    const keep = famSel.value;
    famSel.replaceChildren(new Option("All", ""));
    for (const f of families) famSel.append(new Option(`${f.name} (${f.count})`, f.name));
    famSel.value = keep || familyFilter;
    $("families").replaceChildren(...families.map((f) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "family" + ((familyFilter === f.name) ? " active" : "");
      b.innerHTML = `<span class="kind">${f.kind}</span><b>${f.name}</b><span>${f.count} runs · ${f.rated} rated${f.running ? ` · ${f.running} live` : ""}</span>`;
      b.onclick = () => {
        familyFilter = familyFilter === f.name ? "" : f.name;
        $("filterFamily").value = familyFilter;
        renderIndex();
      };
      return b;
    }));
    const q = ($("runSearch").value || "").trim().toLowerCase();
    const status = $("filterStatus").value;
    const rating = $("filterRating").value;
    const fam = $("filterFamily").value || familyFilter;
    const sort = $("runSort").value;
    const items = runs.filter((row) => {
      if (fam && row.family !== fam && row.sweep !== fam && row.name !== fam) return false;
      if (status === "other" && ["running", "complete", "interrupted"].includes(row.status)) return false;
      if (status && status !== "other" && row.status !== status) return false;
      if (rating === "joint" && !row.joint) return false;
      if (rating === "v6" && !poolScore(row, "v6")) return false;
      if (rating === "v5" && !poolScore(row, "v5")) return false;
      if (rating === "none" && row.has_rating) return false;
      if (q) {
        const blob = [row.name, row.family, row.sweep, row.status, ...(row.flags || [])].join(" ").toLowerCase();
        if (!blob.includes(q)) return false;
      }
      return true;
    });
    const val = (row) => {
      if (sort === "name") return row.name;
      if (sort === "steps") return row.steps ?? -1;
      if (sort === "updated") return row.updated ?? 0;
      if (sort === "v6") return poolScore(row, "v6")?.rating ?? -1e9;
      if (sort === "v5") return poolScore(row, "v5")?.rating ?? -1e9;
      return row.joint?.rating ?? -1e9;
    };
    items.sort((a, b) => sort === "name" ? String(a.name).localeCompare(b.name) : val(b) - val(a));
    $("indexTag").textContent = `${items.length} / ${runs.length} runs`;
    const body = $("runRows");
    body.replaceChildren();
    for (const row of items) {
      const tr = document.createElement("tr");
      if (row.unrated) tr.className = "unrated";
      const name = document.createElement("td");
      const a = document.createElement("a");
      a.href = `/run/${encodeURIComponent(row.name)}`;
      a.textContent = row.name;
      name.append(a);
      const family = document.createElement("td");
      family.className = "wide";
      if (row.sweep) {
        const s = document.createElement("a");
        s.href = `/sweep/${encodeURIComponent(row.sweep)}`;
        s.textContent = row.sweep;
        family.append(s);
      } else family.textContent = row.family || "—";
      const st = document.createElement("td");
      st.textContent = row.status || "—";
      const steps = document.createElement("td");
      steps.className = "num";
      steps.textContent = row.steps == null ? "—" : Number(row.steps).toLocaleString();
      const joint = document.createElement("td");
      joint.className = "num";
      joint.textContent = row.joint ? elo(row.joint.rating) : "—";
      const v6 = document.createElement("td");
      v6.className = "num wide";
      v6.textContent = poolScore(row, "v6") ? elo(poolScore(row, "v6").rating) : "—";
      const v5 = document.createElement("td");
      v5.className = "num wide";
      v5.textContent = poolScore(row, "v5") ? elo(poolScore(row, "v5").rating) : "—";
      const banks = document.createElement("td");
      banks.className = "wide";
      const share = (row.share_min != null || row.share_max != null)
        ? `${fmt(row.share_min, 2)}–${fmt(row.share_max, 2)}`
        : "—";
      banks.textContent = `${row.scripted_banks ?? "—"} / ${share}`;
      const flags = document.createElement("td");
      flags.className = "flags";
      for (const f of row.flags || []) {
        const s = document.createElement("span"); s.textContent = f; s.className = "warn"; flags.append(s);
      }
      tr.append(name, family, st, steps, joint, v6, v5, banks, flags);
      body.append(tr);
    }
    $("indexNote").textContent = "Joint Elo is the last well-played checkpoint on the joint fit (random = 0). Pool v5 / v6 columns are IQM3 on that frozen pool and are not comparable to each other or to joint Elo.";
    setSweepNav();
    live();
  }

  function kindClass(row) {
    if (row.kind === "human") return "human";
    if (row.kind === "scripted" || row.kind === "random") return row.kind;
    if (row.unrated) return "unrated";
    return "";
  }

  function renderLadder() {
    show("ladder");
    if (!ladder) return;
    const stats = ladder.stats || {};
    $("poolIdentity").innerHTML = [
      stats.participants != null ? `<b>players</b> ${stats.participants}` : "",
      stats.games != null ? `<b>games</b> ${stats.games}` : "",
      stats.pairs != null ? `<b>pairs</b> ${stats.pairs}` : "",
      stats.elapsed != null ? `<b>fit</b> ${fmt(stats.elapsed, 2)}s · ${stats.iterations} iter · ${stats.ci}` : "",
      `<b>anchor</b> random = 0`,
      `<b>prior</b> 1 virtual draw / pair`,
      (ladder.pools || []).map((p) => `<b>pool ${p.label}</b> ${p.sha8}`).join("<br>"),
      ladder.building ? "<b>status</b> fitting…" : "",
      ladder.error ? `<b>error</b> ${ladder.error}` : "",
    ].filter(Boolean).join("<br>");
    const hideBots = $("hideBots").checked;
    const allCkpt = $("allCheckpoints").checked;
    const sort = $("sort").value;
    let items = (ladder.rows || []).slice();
    if (!allCkpt) {
      const best = new Map();
      const kept = [];
      for (const row of items) {
        if (row.kind !== "checkpoint" || !row.run) {
          kept.push(row);
          continue;
        }
        const prev = best.get(row.run);
        if (!prev || (row.games || 0) > (prev.games || 0) ||
            ((row.games || 0) === (prev.games || 0) && (row.steps || 0) > (prev.steps || 0))) {
          best.set(row.run, row);
        }
      }
      items = [...kept, ...best.values()];
    }
    if (hideBots) items = items.filter((row) => row.kind !== "scripted");
    items.sort((a, b) => {
      if (sort === "name") return String(a.name).localeCompare(String(b.name));
      if (sort === "kind") return String(a.kind).localeCompare(String(b.kind)) || (b.rating || -1e9) - (a.rating || -1e9);
      if (sort === "games") return (b.games || 0) - (a.games || 0);
      return (b.rating ?? -1e9) - (a.rating ?? -1e9);
    });
    $("ladderTag").textContent = `${items.length} shown · ${(ladder.rows || []).length} fitted`;
    const body = $("ladderRows");
    body.replaceChildren();
    items.forEach((row, i) => {
      const tr = document.createElement("tr");
      tr.className = kindClass(row);
      const rank = document.createElement("td");
      rank.className = "num";
      rank.textContent = String(row.rank ?? i + 1);
      const name = document.createElement("td");
      if (row.run) {
        const a = document.createElement("a");
        a.href = `/run/${encodeURIComponent(row.run)}`;
        a.textContent = row.name;
        name.append(a);
      } else name.textContent = row.name;
      const kind = document.createElement("td");
      kind.className = "wide";
      kind.textContent = row.kind || "";
      const rating = document.createElement("td");
      rating.className = "num";
      rating.textContent = elo(row.rating);
      const interval = document.createElement("td");
      interval.className = "num wide";
      interval.textContent = ci(row.interval);
      const rec = document.createElement("td");
      rec.className = "num";
      rec.textContent = wld(row);
      const games = document.createElement("td");
      games.className = "num wide";
      games.textContent = row.games == null ? "—" : String(row.games);
      tr.append(rank, name, kind, rating, interval, rec, games);
      body.append(tr);
    });
    const human = ladder.human;
    $("humanPanel").hidden = !human;
    if (human) {
      $("humanLede").textContent = `Matthew  ${elo(human.rating)}  ${ci(human.interval)}  ·  ${wld(human)} in ${human.games} games (excluded rounds dropped).`;
      $("humanRows").replaceChildren(...(human.records || []).map((rec) => {
        const tr = document.createElement("tr");
        const href = rec.id && rec.id.length === 64
          ? (ladder.rows || []).find((r) => r.id === rec.id)?.run
          : null;
        tr.innerHTML = `<td>${href ? `<a href="/run/${encodeURIComponent(href)}">${rec.name}</a>` : rec.name}</td><td class="num">${rec.wins}–${rec.losses}–${rec.draws}</td><td class="num">${rec.games}</td>`;
        return tr;
      }));
    }
    $("ladderNote").textContent = "Joint MLE, random pinned at 0, one virtual draw per played pair. 95% CI is a diagonal Wald interval on the observed information — not a pool-v5 or pool-v6 number.";
    const firstSweep = (index?.families || []).find((f) => f.sweep)?.sweep;
    setSweepNav(firstSweep);
    live();
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

  function selectedPool() {
    const sha8 = $("poolPick").value;
    return (data?.pools || []).find((p) => p.sha8 === sha8) || (data?.pools || [])[0];
  }

  function renderRun() {
    show("run");
    if (!data) return;
    $("runTitle").textContent = data.name;
    $("runEyebrow").textContent = data.sweep ? `SWEEP ${data.sweep} · ${data.trial}` : (data.family ? `FAMILY ${data.family}` : "RUN");
    $("runLede").textContent = "Hero number is joint Elo (random = 0). Side cards are frozen-pool IQM3 and are not on that axis.";
    const joint = data.joint;
    $("runElo").textContent = joint ? elo(joint.rating) : "—";
    $("runInterval").textContent = joint
      ? `joint  ${ci(joint.interval)}`
      : "no joint games yet";
    $("runFlags").replaceChildren(...(data.flags || []).map((f) => {
      const s = document.createElement("span"); s.textContent = f; s.className = "warn"; return s;
    }));
    $("runHero").replaceChildren(...(data.pools || []).map((p) => {
      const art = document.createElement("article");
      art.innerHTML = `<p>pool ${p.label} IQM3</p><strong>${p.unrated ? "—" : elo(p.rating)}</strong>`;
      return art;
    }));
    const cmd = $("rateCommand");
    const picked = selectedPool() || (data.pools || []).find((p) => !p.unrated);
    if (data.unrated && data.rate_command) {
      cmd.hidden = false;
      cmd.textContent = data.rate_command;
    } else cmd.hidden = true;
    const status = data.status || {};
    $("runStats").innerHTML = [
      ["status", status.status || "—"],
      ["steps", (status.learner_steps || 0).toLocaleString()],
      ["evals", String((data.evaluations || []).length)],
      ["joint ckpts", String((joint?.ratings || []).length)],
    ].map(([k, v]) => `<article><p>${k}</p><strong>${v}</strong></article>`).join("");
    const human = data.human;
    $("humanRunPanel").hidden = !human;
    if (human) {
      $("humanRunLede").textContent = `Matthew is ${elo(human.rating)} joint (${ci(human.interval)}). Record vs this run:`;
      $("humanRunRows").replaceChildren(...(human.records || []).map((rec) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${rec.name}</td><td class="num">${rec.wins}–${rec.losses}–${rec.draws}</td><td class="num">${rec.games}</td>`;
        return tr;
      }));
    }
    const jpts = (joint?.ratings || []).map((r) => ({
      x: r.learner_steps, y: r.rating,
      lo: (r.interval || [])[0], hi: (r.interval || [])[1],
    }));
    chart($("jointChart"), jpts, { color: "#d6ff32" });
    $("jointCaption").textContent = jpts.length
      ? `${jpts.length} checkpoint${jpts.length === 1 ? "" : "s"} on the joint fit (every pool match + candidate record + human game, random = 0).`
      : "This run has no games in the joint fit yet.";
    const pick = $("poolPick");
    const keep = pick.value;
    pick.replaceChildren(...(data.pools || []).map((p) => {
      const o = document.createElement("option");
      o.value = p.sha8;
      o.textContent = `pool ${p.label}`;
      return o;
    }));
    if ([...pick.options].some((o) => o.value === keep)) pick.value = keep;
    else if (data.primary?.sha8) pick.value = data.primary.sha8;
    renderPoolChart();
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
    live();
  }

  function renderPoolChart() {
    if (!data) return;
    const pool = selectedPool();
    $("poolChartEyebrow").textContent = pool ? `FROZEN POOL ${String(pool.label).toUpperCase()}` : "FROZEN POOL";
    const ratings = pool?.ratings || [];
    const point = (r) => ({ x: r.learner_steps, y: r.rating, lo: (r.interval || [])[0], hi: (r.interval || [])[1] });
    const full = ratings.filter((r) => (r.source || "band") !== "evaluation");
    const cheap = ratings.filter((r) => r.source === "evaluation");
    chart($("eloChart"), full.map(point), { secondary: cheap.map(point), color: "#3ee0c5" });
    $("eloCaption").textContent = !pool
      ? "No frozen-pool ratings on this run."
      : `Solid: ${full.length} checkpoint${full.length === 1 ? "" : "s"} on pool ${pool.label} (${pool.sha8}). `
        + `Dashed grey: ${cheap.length} in-run bots-only fits. These numbers are not on the joint axis.`;
    $("ratingRows").replaceChildren(...ratings.map((r) => {
      const tr = document.createElement("tr");
      if (r.game_mismatch) tr.className = "mismatch";
      tr.innerHTML = `<td>${r.checkpoint}</td><td class="num">${(r.learner_steps || 0).toLocaleString()}</td><td class="num">${elo(r.rating)}</td><td class="num">${ci(r.interval)}</td><td>${r.source || "band"} · pool ${pool.label}</td>`;
      return tr;
    }));
  }

  function renderSweep() {
    show("sweep");
    if (!sweep) return;
    $("sweepTitle").textContent = sweep.name;
    const scored = (sweep.trials || []).filter((t) => t.score != null);
    const best = scored.reduce((b, t) => (b == null || t.score > b.score ? t : b), null);
    $("sweepLede").textContent = `${(sweep.trials || []).length} trials · ${scored.length} scored${best ? ` · best ${best.id} at ${fmt(best.score, 0)}` : ""}`;
    $("sweepIdentity").innerHTML = sweep.score_label
      ? `<b>objective</b> ${sweep.score_label}`
      : "No scoring pool in sweep.json";
    const params = Object.keys(((sweep.trials || []).find((t) => t.hypers) || {}).hypers || {});
    const iqmLabel = sweep.score_label || "IQM3";
    $("sweepHead").innerHTML = ["trial", ...params, "cost", "score", iqmLabel, "joint", "flags"]
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
      cells.push(`<td class="num">${t.joint ? elo(t.joint.rating) : "—"}</td>`);
      cells.push(`<td>${(t.flags || []).join(" ")}</td>`);
      tr.innerHTML = cells.join("");
      return tr;
    }));
    let running = null;
    const points = scored.map((t, i) => ({ x: i, y: t.score }));
    const bestSoFar = scored.map((t, i) => ({ x: i, y: (running = running == null ? t.score : Math.max(running, t.score)) }));
    chart($("scoreChart"), bestSoFar, { color: "#9ec0ff", mark: points });
    $("sweepNav").hidden = false;
    $("sweepNav").href = `/sweep/${encodeURIComponent(sweep.name)}`;
    live();
  }

  async function selectEvaluation() {
    const dir = $("checkpoint").value;
    let report = (data?.evaluations || []).find((e) => e.directory === dir);
    if (data && dir && report && !("games" in report)) {
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
      if (r.view === "index") {
        index = await json("/api/index");
        renderIndex();
      } else if (r.view === "ladder") {
        const [ladderPayload, indexPayload] = await Promise.all([
          json("/api/ladder"),
          index ? Promise.resolve(index) : json("/api/index"),
        ]);
        ladder = ladderPayload;
        index = indexPayload;
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

  $("sort").onchange = $("hideBots").onchange = $("allCheckpoints").onchange = () => { if (ladder) renderLadder(); };
  $("runSearch").oninput = $("filterFamily").onchange = $("filterStatus").onchange = $("filterRating").onchange = $("runSort").onchange = () => {
    familyFilter = $("filterFamily").value;
    if (index) renderIndex();
  };
  $("poolPick").onchange = renderPoolChart;
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
