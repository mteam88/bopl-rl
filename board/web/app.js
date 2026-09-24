"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  // Validated for dark-surface CVD separation; see the palette check in the
  // build notes. Kind is also always written out as text, never colour alone.
  const KIND = {
    main: "#b8e03a", exploiter: "#ff6b9d", specialist: "#3ee0c5",
    scripted: "#6aa8ff", random: "#c08cff",
  };
  const KIND_ORDER = ["main", "exploiter", "specialist", "scripted", "random"];
  const PLAYERS = ["#f4d93f", "#b99be8", "#ffca6a", "#80c7e0", "#d9afee", "#ef9e70"];

  let league = null, agent = null, run = null, runs = null, metrics = null;
  let condition = null;
  let evaluation = null, game = null, replay = null, values = null;
  let playing = false, frameIndex = 0, lastTick = 0;
  let loadToken = 0, replayLoading = false;

  // ------------------------------------------------------------ helpers
  const esc = (t) => String(t ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? "—" : Number(n).toFixed(d));
  const pct = (v, d = 0) => (v == null ? "—" : `${(100 * v).toFixed(d)}%`);
  const big = (n) => (n == null ? "—" : Number(n).toLocaleString());
  const short = (id) => (id && id.length > 12 ? id.slice(0, 8) : id || "—");
  const duration = (s) => { const v = Math.max(0, Math.floor(s)); return `${Math.floor(v / 60)}:${String(v % 60).padStart(2, "0")}`; };
  const kindColor = (k) => KIND[k] || "#8d917c";
  const name = (m) => m?.name || short(m?.id);

  // Diverging: one pole per direction, neutral at an even split. Never a rainbow.
  function payoffColor(score) {
    if (score == null) return "transparent";
    const d = Math.min(1, Math.abs(score - 0.5) * 2);
    const hue = score >= 0.5 ? "62,224,197" : "255,93,115";
    return `rgba(${hue},${(0.10 + 0.78 * d).toFixed(3)})`;
  }

  function stat(label, value, note) {
    return `<article><p>${esc(label)}</p><strong>${value}</strong>${note ? `<small>${esc(note)}</small>` : ""}</article>`;
  }

  // ------------------------------------------------------------- tooltip
  const tip = () => $("tip");
  function showTip(event, html) {
    const box = tip();
    box.innerHTML = html;
    box.hidden = false;
    const w = box.offsetWidth, h = box.offsetHeight;
    let x = event.clientX + 14, y = event.clientY + 14;
    if (x + w > innerWidth - 8) x = event.clientX - w - 14;
    if (y + h > innerHeight - 8) y = event.clientY - h - 14;
    box.style.left = `${Math.max(8, x)}px`;
    box.style.top = `${Math.max(8, y)}px`;
  }
  const hideTip = () => { tip().hidden = true; };
  document.addEventListener("mouseover", (event) => {
    const host = event.target.closest("[data-tip]");
    if (host) showTip(event, host.dataset.tip); else hideTip();
  });
  document.addEventListener("mousemove", (event) => {
    const host = event.target.closest("[data-tip]");
    if (host) showTip(event, host.dataset.tip);
  });

  // --------------------------------------------------------- data access
  const mem = new Map(), inflight = new Map();
  async function readNdjson(response, onPartial) {
    const reader = response.body.getReader(), decoder = new TextDecoder();
    let buf = "", head = null;
    const frames = [];
    const consume = (line) => {
      if (!line) return;
      const obj = JSON.parse(line);
      if (head == null) { head = obj; head.frames = frames; } else frames.push(obj);
    };
    for (;;) {
      const { done, value } = await reader.read();
      buf += decoder.decode(value || new Uint8Array(), { stream: !done });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        consume(buf.slice(0, nl));
        buf = buf.slice(nl + 1);
        if (head && (frames.length === 1 || frames.length % 48 === 0)) onPartial?.(head);
      }
      if (done) break;
    }
    if (buf.trim()) consume(buf.trim());
    onPartial?.(head);
    return head;
  }
  async function json(url, { onUpdate } = {}) {
    const hit = mem.get(url);
    if (onUpdate && hit?.data) onUpdate(hit.data);
    const flight = inflight.get(url);
    if (flight) {
      if (onUpdate) flight.listeners.push(onUpdate);
      return hit?.data ?? flight.promise;
    }
    const headers = { Accept: "application/json, application/x-ndjson" };
    if (hit?.etag) headers["If-None-Match"] = hit.etag;
    const listeners = onUpdate ? [onUpdate] : [];
    const promise = fetch(url, { headers }).then(async (r) => {
      if (r.status === 304 && hit) return hit.data;
      if (!r.ok) throw Error(`${url} ${r.status}`);
      const etag = r.headers.get("ETag");
      const emit = (partial) => { mem.set(url, { data: partial, etag }); for (const fn of listeners) fn(partial); };
      const payload = (r.headers.get("content-type") || "").includes("ndjson")
        ? await readNdjson(r, emit) : await r.json();
      mem.set(url, { data: payload, etag });
      return payload;
    }).finally(() => inflight.delete(url));
    inflight.set(url, { promise, listeners });
    return hit?.data ?? promise;
  }

  // --------------------------------------------------------------- chart
  // One y-axis, always. Two measures of different scale get two charts.
  function chart(canvas, series, opts = {}) {
    const r = canvas.getBoundingClientRect(), dpr = devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(r.width * dpr));
    canvas.height = Math.max(1, Math.round(r.height * dpr));
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    const w = r.width, h = r.height, left = 48, right = 10, top = 10, bottom = 20;
    const lines = series.filter((s) => s.points?.some((p) => p.y != null));
    if (!lines.length) {
      ctx.fillStyle = "#8d917c"; ctx.font = "12px system-ui";
      ctx.fillText("no data", left, h / 2);
      return;
    }
    const all = lines.flatMap((s) => s.points.filter((p) => p.y != null));
    const xs = all.map((p) => p.x), ys = all.flatMap((p) => [p.y, p.lo, p.hi].filter((v) => v != null));
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    let y0 = opts.lo ?? Math.min(...ys), y1 = opts.hi ?? Math.max(...ys);
    if (y0 === y1) { y0 -= 1; y1 += 1; }
    const X = (v) => left + ((w - left - right) * (v - x0)) / (x1 - x0 || 1);
    const Y = (v) => top + (h - top - bottom) * (1 - (v - y0) / (y1 - y0 || 1));
    ctx.strokeStyle = "#20242c"; ctx.fillStyle = "#8d917c";
    ctx.font = "10px ui-monospace, monospace"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const t of [y0, (y0 + y1) / 2, y1]) {
      ctx.beginPath(); ctx.moveTo(left, Y(t)); ctx.lineTo(w - right, Y(t)); ctx.stroke();
      ctx.fillText(opts.format ? opts.format(t) : t.toFixed(2), left - 6, Y(t));
    }
    if (opts.base != null && opts.base > y0 && opts.base < y1) {
      ctx.save(); ctx.strokeStyle = "#4a5160"; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(left, Y(opts.base)); ctx.lineTo(w - right, Y(opts.base)); ctx.stroke(); ctx.restore();
    }
    for (const s of lines) {
      const pts = s.points.filter((p) => p.y != null);
      if (pts.some((p) => p.lo != null)) {
        ctx.fillStyle = (s.color || "#b8e03a") + "22";
        ctx.beginPath();
        pts.forEach((p, i) => (i ? ctx.lineTo(X(p.x), Y(p.hi ?? p.y)) : ctx.moveTo(X(p.x), Y(p.hi ?? p.y))));
        for (let i = pts.length - 1; i >= 0; i--) ctx.lineTo(X(pts[i].x), Y(pts[i].lo ?? pts[i].y));
        ctx.closePath(); ctx.fill();
      }
      ctx.strokeStyle = s.color || "#b8e03a";
      ctx.lineWidth = s.width || 2;
      if (s.dash) ctx.setLineDash(s.dash);
      ctx.beginPath();
      pts.forEach((p, i) => (i ? ctx.lineTo(X(p.x), Y(p.y)) : ctx.moveTo(X(p.x), Y(p.y))));
      ctx.stroke(); ctx.setLineDash([]);
      if (pts.length < 60) {
        ctx.fillStyle = s.color || "#b8e03a";
        for (const p of pts) { ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), 2.5, 0, 6.3); ctx.fill(); }
      }
    }
    if (lines.length > 1) {
      ctx.textAlign = "left"; ctx.textBaseline = "top";
      let ly = top;
      for (const s of lines) {
        ctx.fillStyle = s.color || "#b8e03a";
        ctx.fillRect(left + 8, ly + 4, 9, 2.5);
        ctx.fillStyle = "#c9ccbe";
        ctx.fillText(s.label, left + 22, ly);
        ly += 13;
      }
    }
  }

  // ======================================================= 1. LEAGUE
  function conditionBar(host, payload, onPick) {
    const list = payload.conditions || [];
    if (!list.length) { host.innerHTML = ""; return; }
    const chips = list.map((c) => {
      const on = c.condition === payload.condition;
      return `<button class="chip${on ? " on" : ""}" data-condition="${esc(c.condition)}">${esc(c.condition)}<span>${big(c.games)}</span></button>`;
    }).join("");
    host.innerHTML = `<p class="eyebrow">CONDITION</p><div class="chips">${chips}</div>`
      + (list.length > 1
        ? `<p class="caption warn-text">${list.length} conditions on record. Counts are never pooled across them —
             a condition fixes the simulator hash, the reward semantics and the loadout set, and mixing two of them is
             how a 62% win rate once read as 92%. You are looking at <b>${esc(payload.condition)}</b> only.</p>`
        : "");
    host.querySelectorAll("[data-condition]").forEach((b) => {
      b.onclick = () => { condition = b.dataset.condition; onPick(); };
    });
  }

  function renderLeague() {
    const p = league;
    $("emptyLeague").hidden = !p.empty;
    $("matrixPanel").hidden = !!p.empty;
    $("cyclePanel").hidden = !!p.empty;
    if (p.empty) {
      $("emptyLeagueWhy").textContent =
        "No league.json was found, or it holds no members. This is the normal state before the first "
        + "checkpoint is admitted — the page will fill in as members join.";
    }
    conditionBar($("conditionBar"), p, load);

    const warn = [];
    if (p.mirrored) warn.push(`Every ordered pair in <b>${esc(p.condition)}</b> has an exact mirror row.
      Read literally that doubles every game count. Either the writer mirrors each game into both rows —
      in which case every total here is 2x — or both sides genuinely played equal schedules. Worth confirming.`);
    if (p.members?.length > 1 && p.coverage < 0.5) warn.push(`Only ${pct(p.coverage)} of ordered pairs have been played.
      The matrix is mostly empty, and both ratings are extrapolating across the gaps.`);
    $("leagueWarnings").innerHTML = warn.map((w) => `<div class="warn">${w}</div>`).join("");

    const members = p.members || [];
    const rated = members.filter((m) => m.rating != null);
    $("leagueHero").innerHTML =
      `<div class="hero-elo"><strong>${members.length}</strong><small>members</small></div>`
      + `<div class="hero-side">`
      + stat("Pairs played", big(p.pairs), `${pct(p.coverage)} coverage`)
      + stat("Games", big(p.games))
      + `</div>`
      + `<div class="hero-side">`
      + stat("Cycles", big(p.cycles?.count ?? 0), p.cycles?.count ? "rating order is not the whole story" : "none found")
      + stat("Rating spread", rated.length > 1 ? fmt(p.saturation?.rating_range, 0) : "—", "Elo")
      + `</div>`;

    renderMatrix();
    renderCycles();
    renderRatingTable();
    renderSaturation();
  }

  function orderedMembers() {
    const members = [...(league.members || [])];
    const how = $("matrixSort").value;
    const key = {
      rating: (m) => -(m.rating ?? -1e9),
      nash: (m) => -(m.nash_value ?? -1e9),
      added: (m) => String(m.added ?? ""),
      kind: (m) => KIND_ORDER.indexOf(m.kind),
    }[how] || ((m) => -(m.rating ?? -1e9));
    members.sort((a, b) => {
      if ($("groupKind").checked) {
        const d = KIND_ORDER.indexOf(a.kind) - KIND_ORDER.indexOf(b.kind);
        if (d) return d;
      }
      const ka = key(a), kb = key(b);
      return ka < kb ? -1 : ka > kb ? 1 : 0;
    });
    return members;
  }

  function provenanceTip(m) {
    const o = m.origin || {};
    return `<b>${esc(name(m))}</b><br><span class="k">${esc(m.kind || "?")}</span>`
      + `<table>`
      + `<tr><td>id</td><td>${esc(short(m.id))}</td></tr>`
      + `<tr><td>origin</td><td>${esc(o.run || "—")}${o.step != null ? ` @ ${big(o.step)}` : ""}</td></tr>`
      + `<tr><td>sim</td><td>${esc(short(m.sim))}</td></tr>`
      + `<tr><td>reward</td><td>${esc(m.reward || "—")}</td></tr>`
      + `<tr><td>loadouts</td><td>${esc(m.loadouts ?? "—")}</td></tr>`
      + `<tr><td>added</td><td>${esc(m.added || "—")}</td></tr>`
      + `<tr><td>parent</td><td>${esc(short(m.parent))}</td></tr>`
      + `</table>`;
  }

  function renderMatrix() {
    const members = orderedMembers();
    const index = new Map(league.ids.map((id, i) => [id, i]));
    $("matrixScale").innerHTML =
      `<span>loses</span><i style="background:${payoffColor(0)}"></i><i style="background:${payoffColor(0.25)}"></i>`
      + `<i style="background:${payoffColor(0.5)}"></i><i style="background:${payoffColor(0.75)}"></i>`
      + `<i style="background:${payoffColor(1)}"></i><span>wins</span>`
      + `<em>0%</em><em style="margin-left:auto">50%</em><em>100%</em>`;

    const head = members.map((m) =>
      `<th data-tip="${esc(provenanceTip(m))}"><span style="color:${kindColor(m.kind)}">${esc(name(m))}</span></th>`).join("");
    const body = members.map((rowMember) => {
      const i = index.get(rowMember.id);
      const cells = members.map((colMember) => {
        const j = index.get(colMember.id);
        if (rowMember.id === colMember.id) return `<td class="self"></td>`;
        const cell = league.cells[i]?.[j];
        if (!cell) return `<td class="none" data-tip="<b>${esc(name(rowMember))}</b> vs <b>${esc(name(colMember))}</b><br>never played"></td>`;
        const tipHtml = `<b>${esc(name(rowMember))}</b> vs <b>${esc(name(colMember))}</b>`
          + `<table><tr><td>score</td><td>${pct(cell.score, 1)}</td></tr>`
          + `<tr><td>W-L-D</td><td>${cell.wins}–${cell.losses}–${cell.draws}</td></tr>`
          + `<tr><td>games</td><td>${big(cell.games)}</td></tr></table>`;
        return `<td style="background:${payoffColor(cell.score)}" data-tip="${esc(tipHtml)}">${(100 * cell.score).toFixed(0)}</td>`;
      }).join("");
      return `<tr><th data-tip="${esc(provenanceTip(rowMember))}">`
        + `<a href="/agent/${encodeURIComponent(rowMember.id)}" style="color:${kindColor(rowMember.kind)}">${esc(name(rowMember))}</a>`
        + `</th>${cells}</tr>`;
    }).join("");
    $("matrixWrap").innerHTML = members.length
      ? `<table class="matrix"><thead><tr><th class="corner"></th>${head}</tr></thead><tbody>${body}</tbody></table>`
      : `<p class="caption">No members to compare yet.</p>`;
  }

  function renderCycles() {
    const c = league.cycles || { count: 0, top: [] };
    $("cycleTitle").textContent = c.count
      ? `${c.count} cycle${c.count === 1 ? "" : "s"}.`
      : "No cycles.";
    if (!c.count) {
      $("cycleList").innerHTML = `<p class="caption">Every decided pair agrees with a single order, so on the
        evidence so far the rating is a fair summary. That can stop being true the moment a member joins.</p>`;
      return;
    }
    const by = new Map((league.members || []).map((m) => [m.id, m]));
    $("cycleList").innerHTML = `<div class="cycles">` + c.top.map((cycle) => {
      const parts = cycle.ids.map((id, i) => {
        const m = by.get(id);
        return `<a href="/agent/${encodeURIComponent(id)}" style="color:${kindColor(m?.kind)}">${esc(name(m))}</a>`
          + `<i>${pct(cycle.scores[i], 0)}</i>`;
      }).join("<u>beats</u>");
      return `<div class="cycle">${parts}<u>beats</u><span class="wrap">↺</span>
        <span class="strength">+${pct(cycle.strength, 1)} weakest edge</span></div>`;
    }).join("") + `</div>`;
  }

  function renderRatingTable() {
    const members = orderedMembers();
    const rows = members.map((m) => {
      const iv = m.interval ? `${fmt(m.interval[0], 0)}–${fmt(m.interval[1], 0)}` : "—";
      return `<tr>`
        + `<td><a href="/agent/${encodeURIComponent(m.id)}">${esc(name(m))}</a></td>`
        + `<td><span class="pill" style="color:${kindColor(m.kind)};border-color:${kindColor(m.kind)}44">${esc(m.kind || "?")}</span></td>`
        + `<td class="num">${fmt(m.rating, 0)}</td><td class="num dim">${iv}</td>`
        + `<td class="num">${pct(m.nash_value, 1)}</td>`
        + `<td class="num">${m.nash_weight > 0.001 ? pct(m.nash_weight, 1) : "—"}</td>`
        + `<td class="num">${pct(m.score, 1)}</td>`
        + `<td class="num dim">${big(m.games)}</td>`
        + `<td class="num dim">${m.opponents ?? 0}</td></tr>`;
    }).join("");
    $("ratingTable").innerHTML =
      `<thead><tr><th>Member</th><th>Kind</th><th class="num">Elo</th><th class="num">95%</th>`
      + `<th class="num">Nash value</th><th class="num">Nash mass</th><th class="num">Score</th>`
      + `<th class="num">Games</th><th class="num">Opp</th></tr></thead><tbody>${rows}</tbody>`;
  }

  function renderSaturation() {
    const s = league.saturation || {};
    const kinds = Object.entries(s.kinds || {}).sort((a, b) => b[1] - a[1])
      .map(([k, n]) => `<span class="pill" style="color:${kindColor(k)};border-color:${kindColor(k)}44">${esc(k)} ${n}</span>`).join(" ");
    $("saturationStats").innerHTML =
      stat("Win-rate spread", fmt(s.spread, 3), s.spread == null ? "needs 2+ rated members"
        : s.spread < 0.05 ? "collapsing — nobody exploits anybody" : "healthy")
      + stat("Score range", s.range == null ? "—" : pct(s.range, 1), "worst to best")
      + stat("Members", big(s.members))
      + stat("Largest origin run", s.top_run ? pct(s.top_run_share, 0) : "—",
        s.top_run ? `${s.top_run} — skew toward the learner` : "no origins recorded")
      + `<article class="wide"><p>Composition</p><div class="pills">${kinds || "—"}</div></article>`;
  }

  // ======================================================== 2. AGENT
  function renderAgent() {
    const p = agent, m = p.member;
    $("agentNav").hidden = false;
    $("agentNav").href = `/agent/${encodeURIComponent(m.id)}`;
    $("agentKind").textContent = (m.kind || "member").toUpperCase();
    $("agentTitle").textContent = name(m);
    $("agentTitle").style.color = kindColor(m.kind);
    $("agentLede").textContent = `Everything below is condition ${p.condition} only.`;
    conditionBar($("agentConditionBar"), p, load);
    $("agentHero").innerHTML =
      `<div class="hero-elo"><strong>${fmt(m.rating, 0)}</strong><small>Elo${m.interval ? ` · ${fmt(m.interval[0], 0)}–${fmt(m.interval[1], 0)}` : ""}</small></div>`
      + `<div class="hero-side">${stat("Nash value", pct(m.nash_value, 1))}${stat("Score", pct(m.score, 1))}</div>`
      + `<div class="hero-side">${stat("Games", big(m.games))}${stat("Opponents", m.opponents ?? 0)}</div>`;

    const o = m.origin || {};
    $("agentProvenance").innerHTML =
      `<b>id</b> ${esc(m.id)}<br>`
      + `<b>origin</b> ${o.run ? `<a href="/run/${o.run.split("/").map(encodeURIComponent).join("/")}">${esc(o.run)}</a>` : "—"}`
      + `${o.step != null ? ` @ step ${big(o.step)}` : ""}<br>`
      + `<b>sim</b> ${esc(m.sim || "—")}<br><b>reward</b> ${esc(m.reward || "—")}<br>`
      + `<b>loadouts</b> ${esc(typeof m.loadouts === "object" ? JSON.stringify(m.loadouts) : m.loadouts ?? "—")}<br>`
      + `<b>added</b> ${esc(m.added || "—")}<br>`
      + `<b>parent</b> ${m.parent ? `<a href="/agent/${encodeURIComponent(m.parent)}">${esc(m.parent)}</a>` : "— (fresh init)"}`;

    const history = (p.history || []).filter((h) => h.step != null);
    chart($("agentHistory"), [
      { label: "Elo", color: "#b8e03a", points: history.map((h) => ({ x: h.step, y: h.rating })) },
    ], { format: (v) => v.toFixed(0) });
    $("agentHistoryNote").textContent = history.length > 1
      ? `${history.length} checkpoints of ${(m.origin || {}).run} are in the league; this is their rating by training step.`
      : "A member has one rating. A trajectory needs several checkpoints of the same run in the league — there are not enough yet.";

    const row = p.row || [];
    $("agentRow").innerHTML = row.length
      ? `<thead><tr><th>Opponent</th><th>Kind</th><th class="num">Score</th><th class="num">W-L-D</th><th class="num">Games</th></tr></thead><tbody>`
        + row.map((r) => `<tr><td><a href="/agent/${encodeURIComponent(r.id)}">${esc(r.name || short(r.id))}</a></td>`
          + `<td><span class="pill" style="color:${kindColor(r.kind)};border-color:${kindColor(r.kind)}44">${esc(r.kind || "?")}</span></td>`
          + `<td class="num" style="color:${r.score < 0.5 ? "#ff6b9d" : "#3ee0c5"}">${pct(r.score, 1)}</td>`
          + `<td class="num dim">—</td><td class="num dim">${big(r.games)}</td></tr>`).join("")
        + `</tbody>`
      : `<tbody><tr><td class="caption">No games recorded against any member in this condition.</td></tr></tbody>`;

    const ex = p.exploiters || { runs: [] };
    $("exploitNote").innerHTML = ex.runs.length
      ? `Each exploiter run's score against this member, by the exploiter's training step.
         <b>The target is inferred</b> — league.json has no field naming who an exploiter was pointed at —
         so a run that also played a broad slate can be misattributed.`
      : `No exploiter has been trained against this member yet, or none is in the league.
         <b>Note:</b> league.json records no target for an exploiter, so this panel has to infer one
         from who the exploiter played most; a <code>target</code> field would make it exact.`;
    $("exploitList").innerHTML = ex.runs.map((r) => {
      const marks = ["0.55", "0.7"].map((k) => {
        const at = r.marks?.[k];
        return `<span class="pill">${Math.round(100 * Number(k))}% ${at == null ? "not reached" : `at ${big(at)}`}</span>`;
      }).join(" ");
      return `<div class="exploit"><b>${esc(r.run)}</b> ${marks}
        <span class="dim">best ${pct(r.best, 1)} over ${r.points.length} checkpoints</span>
        </div>`;
    }).join("") || "";

    const reg = p.specialists || [];
    $("regretList").innerHTML = reg.length
      ? `<div class="bars">` + reg.map((r) => {
        const width = Math.min(100, Math.max(0, r.regret * 200));
        return `<div class="bar" data-tip="${esc(`<b>${r.name || short(r.id)}</b><br>score ${pct(r.score, 1)} over ${big(r.games)} games`)}">`
          + `<span class="bar-label">${esc(r.name || short(r.id))}<i>${esc(typeof r.loadouts === "object" ? JSON.stringify(r.loadouts) : r.loadouts ?? "")}</i></span>`
          + `<span class="bar-track"><span class="bar-fill ${r.regret > 0 ? "bad" : "ok"}" style="width:${width}%"></span></span>`
          + `<span class="bar-value">${r.regret > 0 ? "−" : "+"}${pct(Math.abs(r.regret), 1)}</span></div>`;
      }).join("") + `</div>`
      : `<p class="caption">No specialists in the league yet, so there is no per-kit regret to measure.
          Specialists are identified by <code>kind: "specialist"</code> and their kit by the <code>loadouts</code> descriptor.</p>`;
  }

  // ========================================================== 3. RUN
  function fillRunSelect(select, current) {
    const filter = ($("runSearch")?.value || "").toLowerCase();
    const list = (runs || []).filter((r) => !filter || r.name.toLowerCase().includes(filter));
    select.innerHTML = list.slice(0, 400).map((r) =>
      `<option value="${esc(r.name)}"${r.name === current ? " selected" : ""}>${esc(r.name)} · ${esc(r.status || "?")}</option>`).join("");
  }

  function renderRun() {
    fillRunSelect($("runSelect"), run?.name);
    if (!run) {
      $("runTitle").textContent = "Pick a run.";
      $("runLede").textContent = `${(runs || []).length} runs found.`;
      ["opponentPanel", "runMembersPanel"].forEach((id) => ($(id).hidden = true));
      return;
    }
    ["opponentPanel", "runMembersPanel"].forEach((id) => ($(id).hidden = false));
    const s = run.status || {};
    $("runTitle").textContent = run.name;
    $("runLede").textContent = `${s.status || "?"} · ${big(s.learner_steps || s.step)} steps`;
    $("runHero").innerHTML =
      `<div class="hero-elo"><strong>${big(s.learner_steps || s.step || 0)}</strong><small>learner steps</small></div>`
      + `<div class="hero-side">${stat("Status", esc(s.status || "?"))}${stat("Evaluations", (run.evaluations || []).length)}</div>`
      + `<div class="hero-side">${stat("League rating", run.league?.length ? run.league[run.league.length - 1].rating.toFixed(0) : "—", "latest rated checkpoint")}${stat("League members", (run.members || []).length, "checkpoints admitted")}</div>`;

    renderOpponents();

    const members = run.members || [];
    const admitted = members.filter((m) => m.step != null && m.rating != null);
    const rated = run.league || [];
    const latest = rated[rated.length - 1];
    $("runRatingNote").innerHTML = rated.length || admitted.length
      ? `This run's checkpoints on the league's scale: each one plays the members on the kits the run trained on,
         on random maps, and is rated with the members held fixed (<code>league candidate</code>). The band is the 95%
         interval. <b>in-run</b> points are the trainer's small rating of every evaluation; <b>band</b> points are the
         scorer's full one.${latest ? ` Latest: <b>${latest.rating.toFixed(0)}</b> [${latest.interval[0].toFixed(0)},
         ${latest.interval[1].toFixed(0)}] over ${big(latest.games)} games.` : ""}`
      : `No checkpoint of this run has been rated against the league yet.`;
    chart($("runRating"), [
      { label: "In-run rating", color: "#6b8a2a",
        points: rated.filter((r) => r.source === "in-run").map((r) => ({ x: r.step, y: r.rating, lo: r.interval[0], hi: r.interval[1] })) },
      { label: "Band rating", color: "#b8e03a",
        points: rated.filter((r) => r.source === "band").map((r) => ({ x: r.step, y: r.rating, lo: r.interval[0], hi: r.interval[1] })) },
      { label: "League member", color: "#3ee0c5",
        points: admitted.map((m) => ({ x: m.step, y: m.rating, lo: m.interval?.[0], hi: m.interval?.[1] })) },
    ].filter((series) => series.points.length), { format: (v) => v.toFixed(0) });

    $("runMembers").innerHTML = members.length
      ? `<div class="chips">` + members.map((m) =>
        `<a class="chip" href="/agent/${encodeURIComponent(m.id)}" style="color:${kindColor(m.kind)}">${esc(m.name || short(m.id))}<span>${big(m.step)}</span></a>`).join("") + `</div>`
      : `<p class="caption">No checkpoint from this run has been admitted to the league yet.</p>`;

    const evals = (run.evaluations || []).filter((e) => e.score != null && e.step != null);
    chart($("runEval"), [{ label: "Evaluation score", color: "#3ee0c5", points: evals.map((e) => ({ x: e.step, y: e.score })) }],
      { lo: 0, hi: 1, base: 0.5, format: (v) => `${(100 * v).toFixed(0)}%` });

    if (metrics?.name !== run.name) loadMetrics();
    else renderMetric();
  }

  function renderOpponents() {
    const o = run.opponents || {};
    $("opponentUnavailable").hidden = !!o.available;
    $("opponentStats").innerHTML = "";
    $("opponentBars").innerHTML = "";
    if (!o.available) {
      $("opponentNote").textContent = "";
      $("opponentUnavailable").textContent = o.reason || "Not recorded.";
      return;
    }
    $("opponentNote").innerHTML =
      `Realized counts come from the pool ledger and the scripted roster. Matchmaking is <b>PFSP</b>
       (<code>f_${o.pfsp_f ?? "?"}</code>, p = ${o.pfsp_p ?? "?"}) over ${o.max_size ?? "?"} live snapshots plus every league
       member this run named, served by ${o.banks ?? "?"} opponent slots, so the intended weight vector is real and
       recomputed here from the ledger.
       Intended is a wish, the ${big(o.pfsp_draws)} draws are what it actually picked, and the bars are games played
       — the population saturates when those three come apart.`;
    const degenerate = o.distinct_serving != null && o.banks && o.distinct_serving < o.banks;
    $("opponentStats").innerHTML =
      stat("Distinct opponents played", big(o.distinct), `${big(o.pooled)} candidates ever`)
      + stat("Slots → distinct opponents", `${o.banks ?? "—"} → ${o.distinct_serving ?? "—"}`,
        degenerate ? "slots have collapsed onto the same opponents" : "slots are serving distinct opponents")
      + stat("Candidates never played", `${o.live_unplayed ?? "—"} / ${o.live ?? "—"}`,
        o.live_unplayed ? "these are candidates but contribute nothing" : "every live candidate is being played")
      + stat("Top 4 share", pct(o.top_share, 1), "of all games")
      + stat("Intended vs drawn", o.pfsp_divergence == null ? "—" : o.pfsp_divergence.toFixed(3),
        "total variation — 0 is PFSP getting the curriculum it asked for")
      + stat("Draw entropy",
        `${o.pfsp_drawn_entropy == null ? "—" : o.pfsp_drawn_entropy.toFixed(2)} / ${(o.pfsp_intended_entropy ?? 0).toFixed(2)}`,
        "realized vs intended, nats — a collapsing population loses the left number first")
      + stat("Scripted realized", pct(o.scripted_realized_lifetime, 1), "lifetime")
      + stat("Scripted intended", pct(o.scripted_intended_final, 1), "final value — adaptive, not a lifetime target");

    const entries = (o.entries || []).slice(0, 40);
    const peak = Math.max(...entries.map((e) => e.share), 0.0001);
    $("opponentBars").innerHTML = `<div class="bars">` + entries.map((e) => {
      const tipHtml = `<b>${esc(e.id)}</b><br>${esc(e.kind)}${e.step != null ? ` @ ${big(e.step)}` : ""}`
        + `<table><tr><td>games</td><td>${big(e.games)}</td></tr>`
        + `<tr><td>share</td><td>${pct(e.share, 2)}</td></tr>`
        + `<tr><td>learner wr</td><td>${e.winrate == null ? "—" : pct(e.winrate, 1)}</td></tr>`
        + `<tr><td>score (draws half)</td><td>${e.score == null ? "—" : pct(e.score, 1)}</td></tr>`
        + `<tr><td>intended</td><td>${e.intended == null ? "—" : pct(e.intended, 1)}</td></tr>`
        + `<tr><td>drawn</td><td>${e.drawn == null ? "—" : pct(e.drawn, 1)}</td></tr>`
        + `<tr><td>a candidate</td><td>${e.live ? "yes" : "evicted"}</td></tr></table>`;
      return `<div class="bar" data-tip="${esc(tipHtml)}">`
        + `<span class="bar-label">${esc(e.id)}<i>${e.serving ? "serving now" : e.live ? "a candidate" : "evicted"}</i></span>`
        + `<span class="bar-track"><span class="bar-fill" style="width:${(100 * e.share / peak).toFixed(1)}%;`
        + `background:${e.kind === "scripted" ? "#6aa8ff" : e.kind === "member" ? "#e0a33a" : e.live ? "#b8e03a" : "#4a5160"}"></span></span>`
        + `<span class="bar-value">${pct(e.share, 2)}</span></div>`;
    }).join("") + `</div>`
      + `<p class="caption">Bars are the realized share of all training games. Green is one of this run\u2019s own snapshots,
         amber a league member, blue a scripted bot, grey a snapshot that has since been evicted. A long grey tail with a few
         tall bars is the degeneration to watch for.</p>`;
  }

  const METRIC_KEYS = ["policy_loss", "value_loss", "entropy", "kl", "clip_fraction",
    "gradient_norm", "explained_variance", "mean_reward", "sps", "learner_sps",
    "advantage_std", "learning_rate", "cuda_memory_mb"];

  async function loadMetrics() {
    const target = run.name;
    try {
      const payload = await json(`/api/run/${target.split("/").map(encodeURIComponent).join("/")}/metrics`);
      metrics = { name: target, ...payload };
    } catch { metrics = { name: target, metrics: [] }; }
    if (run?.name === target) renderMetric();
  }

  function renderMetric() {
    const rows = metrics?.metrics || [];
    const present = METRIC_KEYS.filter((k) => rows.some((r) => r[k] != null));
    const pick = $("metricPick");
    if (pick.dataset.run !== run.name) {
      pick.innerHTML = present.map((k) => `<option value="${k}">${k}</option>`).join("");
      pick.dataset.run = run.name;
    }
    const key = pick.value || present[0];
    $("metricTitle").textContent = key ? key.replace(/_/g, " ") : "Training.";
    const xKey = rows.some((r) => r.learner_steps != null) ? "learner_steps" : "step";
    chart($("runMetric"), [{
      label: key || "—", color: "#b8e03a",
      points: rows.filter((r) => r[key] != null).map((r) => ({ x: r[xKey] ?? 0, y: r[key] })),
    }], { format: (v) => (Math.abs(v) >= 1000 ? v.toExponential(1) : v.toFixed(3)) });
  }

  // ======================================================== 4. WATCH
  async function renderWatch() {
    fillRunSelect($("watchRun"), run?.name);
    if (!run) return;
    const evals = run.evaluations || [];
    $("checkpoint").innerHTML = evals.map((e) =>
      `<option value="${esc(e.directory)}">${big(e.step)} · ${pct(e.score, 0)}</option>`).join("");
    if (evals.length) {
      if (!evaluation || !evals.some((e) => e.directory === evaluation.directory)) {
        $("checkpoint").value = evals[evals.length - 1].directory;
      }
      await selectEvaluation();
    }
  }

  const runPath = (n) => n.split("/").map(encodeURIComponent).join("/");
  const fileUrl = (ev, file) => `/runs/${runPath(run.name)}/evaluations/${ev.directory}/${encodeURIComponent(file)}`;

  async function selectEvaluation() {
    const folder = $("checkpoint").value;
    if (!folder) return;
    evaluation = await json(`/api/run/${runPath(run.name)}/evaluation/${encodeURIComponent(folder)}`);
    const games = (evaluation.games || []).filter((g) => g.visual);
    $("match").innerHTML = games.map((g) =>
      `<option value="${esc(g.name)}">${esc(g.name)} · ${g.timeout ? "timeout" : g.win ? "win" : "loss"}</option>`).join("");
    if (games.length) await selectMatch();
    else { $("emptyReplay").style.display = ""; $("play").disabled = true; $("replayTag").textContent = "No replay"; }
  }

  async function selectMatch() {
    game = (evaluation?.games || []).find((g) => g.name === $("match").value);
    playing = false;
    if (!game?.visual) {
      $("play").disabled = true; $("replayTag").textContent = "No replay";
      $("emptyReplay").style.display = ""; return;
    }
    const token = ++loadToken;
    const url = fileUrl(evaluation, game.visual);
    $("replayTag").textContent = "Loading…";
    let drawn = false;
    const adopt = (value) => {
      if (token !== loadToken || !value?.frames) return;
      replay = value;
      $("emptyReplay").style.display = "none";
      $("play").disabled = false;
      $("scrub").max = Math.max(0, replay.frames.length - 1);
      $("replayTag").textContent = game.timeout ? "Time limit" : game.win ? "Policy wins" : "Policy loses";
      $("download").hidden = !game.replay;
      if (game.replay) {
        $("download").href = fileUrl(evaluation, game.replay);
        $("download").download = game.replay;
      }
      if (!drawn && replay.frames.length) { drawn = true; frameIndex = 0; $("play").textContent = "▶"; drawFrame(); }
    };
    replayLoading = true;
    try {
      const value = await json(url, { onUpdate: adopt });
      if (token === loadToken) { if (value) adopt(value); else $("replayTag").textContent = "Replay unavailable"; }
    } catch {
      if (token === loadToken) $("replayTag").textContent = "Replay unavailable";
    } finally { if (token === loadToken) replayLoading = false; }
    loadValues();
  }

  async function loadValues() {
    const target = game?.name;
    values = null;
    $("valueStrip").innerHTML = "";
    try {
      values = await json(`/api/run/${runPath(run.name)}/values/${encodeURIComponent(target)}`);
    } catch { values = null; }
    if (game?.name !== target) return;
    renderValues();
    drawFrame();
  }

  function renderValues() {
    const traces = values?.traces || [];
    $("valueNote").innerHTML = traces.length
      ? `The value head's own estimate for <b>${esc(values.game)}</b>, at ${traces.length} checkpoint${traces.length === 1 ? "" : "s"}.
         A sharpening prediction commits earlier and harder; a flat line means the policy cannot tell how the game is going.`
      : `No <code>.mind.json.gz</code> for this match, so the value head was never dumped.`;
    chart($("valueChart"), traces.map((t, i) => ({
      label: big(t.step),
      color: traces.length === 1 ? "#b8e03a"
        : `rgba(184,224,58,${(0.25 + 0.75 * (i / Math.max(1, traces.length - 1))).toFixed(2)})`,
      width: i === traces.length - 1 ? 2.4 : 1.2,
      points: t.points.map((p) => ({ x: p[0], y: p[1] })),
    })), { base: 0, format: (v) => v.toFixed(2) });
  }

  function currentTrace() {
    const traces = values?.traces || [];
    return traces.find((t) => t.directory === evaluation?.directory) || null;
  }

  function drawStrip() {
    const trace = currentTrace();
    const host = $("valueStrip");
    if (!trace || !replay?.frames?.length) { host.innerHTML = ""; return; }
    const last = replay.frames.at(-1).t || 1;
    const pts = trace.points;
    const lo = Math.min(...pts.map((p) => p[1]), -0.05);
    const hi = Math.max(...pts.map((p) => p[1]), 0.05);
    const X = (t) => (100 * t) / last;
    const Y = (v) => 28 - (26 * (v - lo)) / (hi - lo || 1);
    const line = pts.map((p) => `${X(p[0]).toFixed(2)},${Y(p[1]).toFixed(2)}`).join(" ");
    const now = replay.frames[Math.floor(frameIndex)]?.t ?? 0;
    const zero = Y(0);
    host.innerHTML =
      `<svg viewBox="0 0 100 30" preserveAspectRatio="none" aria-label="value head over the match">`
      + `<line x1="0" y1="${zero.toFixed(2)}" x2="100" y2="${zero.toFixed(2)}" stroke="#39404c" stroke-dasharray="2 2" vector-effect="non-scaling-stroke"/>`
      + `<polyline points="${line}" fill="none" stroke="#b8e03a" stroke-width="1.5" vector-effect="non-scaling-stroke"/>`
      + `<line x1="${X(now).toFixed(2)}" y1="0" x2="${X(now).toFixed(2)}" y2="30" stroke="#f3f1e3" stroke-width="1" vector-effect="non-scaling-stroke"/>`
      + `</svg><span class="striplabel">value head</span>`;
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
      box.innerHTML = `<b>P${p.id}${p.id === game.side + 1 ? " · policy" : ""}</b> ${p.kills} kills`;
      box.style.borderColor = PLAYERS[(p.id - 1) % 6];
      return box;
    }));
    drawStrip();
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

  // ======================================================== routing
  function route() {
    const path = location.pathname;
    if (path.startsWith("/agent/")) return { view: "agent", id: decodeURIComponent(path.slice(7)) };
    if (path.startsWith("/run/")) return { view: "run", name: decodeURIComponent(path.slice(5)) };
    if (path === "/run") return { view: "run", name: null };
    if (path.startsWith("/watch")) return { view: "watch", name: new URLSearchParams(location.search).get("run") };
    return { view: "league" };
  }

  function show(view) {
    for (const id of ["leagueView", "agentView", "runView", "watchView"]) $(id).hidden = id !== `${view}View`;
    document.querySelectorAll("nav a").forEach((a) => a.classList.toggle("active", a.dataset.nav === view));
  }

  const query = () => (condition ? `?condition=${encodeURIComponent(condition)}` : "");

  async function load() {
    $("notice").textContent = "";
    const r = route();
    show(r.view);
    try {
      if (r.view === "league") {
        league = await json(`/api/league${query()}`);
        condition = league.condition;
        renderLeague();
      } else if (r.view === "agent") {
        agent = await json(`/api/agent/${encodeURIComponent(r.id)}${query()}`);
        condition = agent.condition;
        renderAgent();
      } else {
        runs = runs || await json("/api/runs");
        const want = r.name || (r.view === "watch" ? runs[0]?.name : null);
        if (want && run?.name !== want) {
          run = await json(`/api/run/${runPath(want)}`);
          metrics = null; evaluation = null; game = null; replay = null;
        } else if (!want) run = null;
        if (r.view === "run") renderRun(); else await renderWatch();
      }
      $("connection").textContent = "Live";
      $("connectionBox").classList.add("online");
    } catch (error) {
      $("notice").textContent = error.message;
      $("connection").textContent = "Error";
      $("connectionBox").classList.remove("online");
    }
  }

  const go = (href) => { history.pushState(null, "", href); load(); };
  $("matrixSort").onchange = $("groupKind").onchange = () => { if (league) { renderMatrix(); renderRatingTable(); } };
  $("runSearch").oninput = () => fillRunSelect($("runSelect"), run?.name);
  $("runSelect").onchange = () => go(`/run/${runPath($("runSelect").value)}`);
  $("watchRun").onchange = () => go(`/watch?run=${encodeURIComponent($("watchRun").value)}`);
  $("metricPick").onchange = renderMetric;
  $("checkpoint").onchange = selectEvaluation;
  $("match").onchange = selectMatch;
  $("play").onclick = () => { playing = !playing; $("play").textContent = playing ? "❚❚" : "▶"; };
  $("scrub").oninput = () => { frameIndex = Number($("scrub").value); drawFrame(); };
  addEventListener("popstate", load);
  addEventListener("resize", () => { if (route().view === "run" && metrics) renderMetric(); });
  document.addEventListener("click", (event) => {
    const a = event.target.closest("a");
    if (!a || event.metaKey || event.ctrlKey) return;
    const href = a.getAttribute("href");
    if (!href || !href.startsWith("/") || href.startsWith("/runs/")) return;
    event.preventDefault();
    go(href);
  });
  load();
  // Revalidate while the tab is visible. Every payload is ETagged, so a poll
  // that finds nothing new costs one 304. Watch is excluded: reloading it
  // would fight the replay that is currently playing.
  setInterval(() => {
    if (document.hidden || route().view === "watch") return;
    load();
  }, 10000);
})();
