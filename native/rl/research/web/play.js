"use strict";
// Browser side of play.py: lobby, live match (keyboard/mouse state -> WebSocket,
// frames -> arena.js), result tally, and a small viewer for recorded games.
(() => {
const $ = (id) => document.getElementById(id);
// Key-state bits shared with play.py (HumanInput): policy buttons in
// bopl_layout.h order, then Unity's W/A/S/D movement actions.
const KEY = { jump: 1, dash: 2, arrow: 4, platform: 8, W: 16, A: 32, S: 64, D: 128 };
// Unity default KeyboardAndMouse bindings (Assets/MonoBehaviour/SlimeControls.asset):
// W/A/S/D and the arrow keys move, Space jumps, mouse left/right/middle are
// abilities 0/1/2. J/K/L are extra keyboard aliases for the mouse buttons.
const CODES = {
  KeyW: KEY.W, ArrowUp: KEY.W, KeyA: KEY.A, ArrowLeft: KEY.A, KeyS: KEY.S, ArrowDown: KEY.S, KeyD: KEY.D, ArrowRight: KEY.D,
  Space: KEY.jump, KeyJ: KEY.dash, KeyK: KEY.arrow, KeyL: KEY.platform,
};
const MOUSE = { 0: KEY.dash, 2: KEY.arrow, 1: KEY.platform, 3: KEY.platform, 4: KEY.platform };
const ABILITY_NAMES = { 2: "Dash", 4: "Bow", 14: "Platform" };
const COLORS = ["#f4d93f", "#b99be8"];
const DECISION_S = 4 / 60;

let ws = null, lobby = null, match = null, lastRecord = null, holdLengths = [1, 2, 4, 8, 16];
const status = $("status");

// ---- WebSocket ---------------------------------------------------------------
function connect() {
  ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  ws.addEventListener("open", () => { status.textContent = "connected"; status.className = "status ok"; });
  ws.addEventListener("close", () => {
    status.textContent = "disconnected"; status.className = "status bad";
    if (match) { match.live = false; }
    setTimeout(connect, 2000);
  });
  ws.addEventListener("message", (event) => {
    if (match) match.bytes += event.data.length;
    const message = JSON.parse(event.data);
    switch (message.T) {
      case "start": startMatch(message); break;
      case "f": if (match) match.frame(message); break;
      case "end": endMatch(message); break;
      case "aborted": showLobby(); break;
      case "pong": if (match) match.rtt.push(performance.now() - message.c); break;
      case "error": alert(message.message); showLobby(); break;
    }
  });
}
function send(message) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(message)); }

// ---- lobby -----------------------------------------------------------------
async function loadLobby() {
  lobby = await (await fetch("/api/lobby")).json();
  holdLengths = lobby.hold_lengths;
  const opponent = $("opponent"), previous = opponent.value;
  opponent.innerHTML = "";
  const scripted = document.createElement("optgroup"); scripted.label = "Scripted bots (bopl_bots.h)";
  for (const name of lobby.scripted) scripted.append(new Option(name, name));
  const checkpoints = document.createElement("optgroup"); checkpoints.label = "Checkpoints";
  for (const c of lobby.checkpoints) checkpoints.append(new Option(`${c.label}  (step ${c.step.toLocaleString()}, ${c.id})`, c.id));
  opponent.append(scripted, checkpoints);
  if (previous) opponent.value = previous; else opponent.value = "mixed-hard";
  const map = $("map"); if (!map.options.length) for (const m of lobby.maps) map.append(new Option(`map ${m}`, m));
  fillTally($("tally"), lobby.tally);
  const rows = $("recent").querySelector("tbody"); rows.innerHTML = "";
  for (const g of [...lobby.games].reverse().slice(0, 20)) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${g.timestamp.replace("T", " ")}</td><td>${g.opponent.label || g.b}</td><td>${g.map}</td><td>P${g.side + 1}</td>` +
      `<td class="${g.result}">${g.result}</td><td>${g.seconds.toFixed(1)}</td><td></td>`;
    const watch = document.createElement("button"); watch.textContent = "watch"; watch.className = "secondary";
    watch.addEventListener("click", () => openViewer(g));
    tr.lastElementChild.append(watch);
    rows.append(tr);
  }
  const bindings = $("bindings").querySelector("tbody"); bindings.innerHTML = "";
  for (const b of lobby.bindings) {
    const tr = document.createElement("tr"); tr.innerHTML = `<td>${b.action}</td><td>${b.keys}</td><td>${b.unity}</td>`; bindings.append(tr);
  }
  $("lobbyNote").textContent = `Sim ${lobby.game.path} (${lobby.game.sha256.slice(0, 12)}). Next alternating side: P${lobby.next_side + 1}. ` +
    `Records: games.jsonl next to this server's --out; replays (.rep + physics-view .json) beside it.`;
}
function fillTally(table, tally) {
  const body = table.querySelector("tbody"); body.innerHTML = "";
  for (const [name, t] of Object.entries(tally).sort()) {
    const tr = document.createElement("tr");
    const decided = t.wins + t.losses;
    tr.innerHTML = `<td>${t.label && t.label !== name ? `${t.label} <span class=fine>${name}</span>` : name}</td><td>${t.games}</td><td>${t.wins}</td><td>${t.losses}</td><td>${t.draws}</td><td>${decided ? (100 * t.wins / decided).toFixed(0) + "%" : "–"}</td>`;
    body.append(tr);
  }
  if (!body.children.length) body.innerHTML = "<tr><td colspan=6 class=fine>no games yet</td></tr>";
}
function show(id) { for (const p of ["lobby", "game", "result", "viewer"]) $(p).hidden = p !== id; }
function showLobby() { match = null; show("lobby"); loadLobby(); }
$("setup").addEventListener("submit", (event) => {
  event.preventDefault();
  send({ T: "start", opponent: $("opponent").value, map: Number($("map").value), side: $("side").value, mode: $("mode").value, seed: $("seed").value });
});
$("again").addEventListener("click", () => {
  const side = $("side").value === "alternate" ? "alternate" : String(1 - Number($("side").value));
  $("side").value = side;
  send({ T: "start", opponent: $("opponent").value, map: Number($("map").value), side, mode: $("mode").value, seed: "" });
});
$("toLobby").addEventListener("click", showLobby);
$("watchLast").addEventListener("click", () => lastRecord && openViewer(lastRecord));

// ---- live match ---------------------------------------------------------------
function expand(m) {
  return {
    t: m.t, flags: m.g,
    players: m.p.map(([id, flags, x, y, scale, kills, deaths]) => ({ id, flags, x, y, scale, kills, deaths })),
    entities: m.e.map(([kind, layer, owner, x, y, rx, ry, ux, uy, radius, vx, vy]) => ({ kind, layer, owner, x, y, rx, ry, ux, uy, radius, vx, vy })),
    slimes: m.s.map(([owner, abilities, cooldown, flags]) => ({ owner, abilities, cooldown, flags })),
  };
}
function percentile(values, q) {
  if (!values.length) return NaN;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];
}

class Match {
  constructor(message) {
    this.side = message.side; this.map = message.map; this.seed = message.seed; this.opponent = message.opponent;
    this.mode = message.mode; this.limit = message.limit; this.live = true;
    this.initial = expand(message.frame); this.last = this.initial; this.lastAt = performance.now(); this.hud = { s: 0, l: 0, b: 0, a: 0, c: 4, m: this.mode };
    this.down = 0; this.pressed = 0; this.charge = 4; this.seq = 0; this.sent = new Map(); this.aim = null; this.lastAimSent = 0;
    this.latency = []; this.intervals = []; this.rtt = []; this.bytes = 0; this.bytesAt = performance.now(); this.rate = 0; this.frames = 0;
    this.mouse = null;
    this.canvas = $("arena");
    this.pingTimer = setInterval(() => send({ T: "ping", c: performance.now() }), 1000);
    this.myId = this.initial.players[this.side].id;
    this.raf = requestAnimationFrame(() => this.draw());
  }
  frame(m) {
    const now = performance.now();
    if (this.frames) this.intervals.push(now - this.lastAt);
    this.frames++;
    this.last = expand(m); this.lastAt = now; this.hud = m.h;
    // Input->visible latency: the frame acknowledges the newest input sequence applied to it.
    for (const [seq, at] of this.sent) if (seq <= m.h.s) { if (seq === m.h.s) this.latency.push(now - at); this.sent.delete(seq); }
    if (this.latency.length > 200) this.latency.splice(0, this.latency.length - 200);
    if (this.intervals.length > 200) this.intervals.splice(0, this.intervals.length - 200);
    if (now - this.bytesAt > 1000) { this.rate = this.bytes * 1000 / (now - this.bytesAt); this.bytes = 0; this.bytesAt = now; }
  }
  // Keyboard/mouse state -> one input message (sent on change; the server samples it every decision).
  push(extra = {}) {
    this.seq++;
    this.sent.set(this.seq, performance.now());
    if (this.sent.size > 500) this.sent.delete(this.sent.keys().next().value);
    send({ T: "i", k: this.down, p: this.pressed, a: this.aim, c: this.charge, s: this.seq, ...extra });
    this.pressed = 0;
  }
  key(bit, isDown) {
    const before = this.down;
    if (isDown) { this.down |= bit; this.pressed |= bit; } else this.down &= ~bit;
    if (isDown || before !== this.down) this.push();
  }
  release() { if (this.down) { this.down = 0; this.push(); } }
  // Mouse aim, Unity's keyboard scheme (InputUpdater.cs Update): the byte of the
  // vector from the player's position to the mouse's world position; zero inside
  // sqrMagnitude 0.2. The camera is arena.js's (canvas._boplCamera).
  updateAim() {
    if (this.mode !== "mouse" || !this.mouse) return;
    const cam = this.canvas._boplCamera; if (!cam) return;
    const me = this.display().players.find((p) => p.id === this.myId); if (!me) return;
    const w = this.canvas.width, h = this.canvas.height, dpr = w / this.canvas.getBoundingClientRect().width;
    const scale = Math.min(w / cam.sx, h / cam.sy);
    const wx = cam.cx + (this.mouse[0] * dpr - w / 2) / scale, wy = cam.cy - (this.mouse[1] * dpr - h / 2) / scale;
    const dx = wx - me.x, dy = wy - me.y;
    let byte = 0;
    if (dx * dx + dy * dy >= 0.2) {
      // Player.InputProfile.JoyStickAngleByteFromVector: int(angle / 2pi * 255) % 255 + 1.
      const angle = (Math.atan2(dy, dx) + 2 * Math.PI) % (2 * Math.PI);
      byte = (Math.floor(angle / (2 * Math.PI) * 255) % 255) + 1;
    }
    if (byte !== this.aim) { this.aim = byte; const now = performance.now(); if (now - this.lastAimSent > 15) { this.lastAimSent = now; this.push(); } }
  }
  // Frame to show now: the last decision's state, optionally moved along each
  // body's velocity for the time since it arrived (at most one decision), so 15
  // decisions/s look continuous without adding display delay.
  display() {
    if (!$("smooth").checked || !this.live) return this.last;
    const dt = Math.min(DECISION_S, (performance.now() - this.lastAt) / 1000);
    const entities = this.last.entities.map((e) => ({ ...e, x: e.x + e.vx * dt, y: e.y + e.vy * dt }));
    const players = this.last.players.map((p) => {
      const body = this.last.entities.find((e) => e.kind === 0 && e.owner === p.id && Math.hypot(e.x - p.x, e.y - p.y) < 2);
      return body ? { ...p, x: p.x + body.vx * dt, y: p.y + body.vy * dt } : p;
    });
    return { ...this.last, entities, players };
  }
  draw() {
    if (!match || match !== this) return;
    this.raf = requestAnimationFrame(() => this.draw());
    const frame = this.display();
    window.drawBoplArena(this.canvas, frame, this.initial);
    const ctx = this.canvas.getContext("2d"), cam = this.canvas._boplCamera, w = this.canvas.width, h = this.canvas.height;
    const scale = Math.min(w / cam.sx, h / cam.sy), ui = w / this.canvas.getBoundingClientRect().width;
    const point = (x, y) => [w / 2 + (x - cam.cx) * scale, h / 2 - (y - cam.cy) * scale];
    const me = frame.players.find((p) => p.id === this.myId);
    if (me && me.flags & 1) {
      // Aim as applied at the last decision (HUD a), plus the live mouse aim in mouse mode.
      const [x, y] = point(me.x, me.y);
      const drawAim = (byte, color, len) => {
        if (!byte) return;
        const a = (byte - 1) * 2 * Math.PI / 255;
        ctx.strokeStyle = color; ctx.lineWidth = 1.5 * ui; ctx.beginPath(); ctx.moveTo(x, y);
        ctx.lineTo(x + Math.cos(a) * len * scale, y - Math.sin(a) * len * scale); ctx.stroke();
      };
      drawAim(this.hud.a, "rgba(255,255,255,0.75)", 4);
      if (this.mode === "mouse") drawAim(this.aim, "rgba(255,255,255,0.3)", 6);
      ctx.strokeStyle = COLORS[this.side]; ctx.lineWidth = 2 * ui; ctx.setLineDash([4 * ui, 3 * ui]);
      ctx.beginPath(); ctx.arc(x, y, 3.2 * scale * me.scale, 0, 2 * Math.PI); ctx.stroke(); ctx.setLineDash([]);
    }
    this.updateAim();
    this.drawHud(frame);
  }
  drawHud(frame) {
    const me = frame.players[this.side], foe = frame.players[1 - this.side];
    $("hudMatch").innerHTML = `<b style="color:${COLORS[this.side]}">You: P${this.side + 1}</b> vs <b>${this.opponent.name}</b>` +
      `${this.opponent.kind === "checkpoint" ? " (checkpoint step " + this.opponent.step.toLocaleString() + ")" : ""} · map ${this.map} · seed ${this.seed}<br>` +
      `t ${(frame.t / 60).toFixed(1)} s / ${(this.limit / 60).toFixed(0)} s · kills ${me.kills}–${foe.kills}` +
      (this.hud.l ? ` · <span style="color:var(--accent)">latched ${this.hud.l} more decision${this.hud.l > 1 ? "s" : ""}</span>` : "");
    const slime = frame.slimes.find((s) => s.owner === this.myId);
    const box = $("hudAbilities");
    if (slime) {
      box.innerHTML = slime.abilities.map((id, i) => {
        const f = slime.flags[i], ready = f & 2, active = f & 8, charge = (f >> 4) & 7;
        const label = ABILITY_NAMES[id] || `ability ${id}`;
        const state = active ? (id === 4 ? `charge ${charge}/4` : "active") : ready ? "ready" : `${(100 * slime.cooldown[i]).toFixed(0)}%`;
        return `<div class="ability ${active ? "active" : ready ? "ready" : ""}"><div>${label} <span class="fine">${state}</span></div><div class="bar"><i style="width:${active ? 100 : 100 * slime.cooldown[i]}%"></i></div></div>`;
      }).join("");
    }
    const hold = holdLengths[this.charge];
    $("hudCharge").textContent = `bow charge: hold ${hold} decision${hold > 1 ? "s" : ""} (${(hold * DECISION_S).toFixed(2)} s, ${hold >= 16 ? "full" : hold >= 8 ? "medium" : "quick"}) · aim: ${this.mode}`;
    $("hudLatency").textContent = `input→frame p50 ${percentile(this.latency, 0.5).toFixed(0)} ms  p95 ${percentile(this.latency, 0.95).toFixed(0)} ms\n` +
      `frame interval ${percentile(this.intervals, 0.5).toFixed(1)} ms (p95 ${percentile(this.intervals, 0.95).toFixed(1)})\n` +
      `rtt ${percentile(this.rtt.slice(-10), 0.5).toFixed(1)} ms · ${(this.rate / 1024).toFixed(1)} KiB/s · ${this.frames} frames`;
    $("netline").textContent = "";
  }
  stats() {
    return { frames: this.frames, latency_p50_ms: percentile(this.latency, 0.5), latency_p95_ms: percentile(this.latency, 0.95),
      interval_p50_ms: percentile(this.intervals, 0.5), interval_p95_ms: percentile(this.intervals, 0.95), interval_max_ms: Math.max(...this.intervals),
      rtt_p50_ms: percentile(this.rtt, 0.5), bytes_per_s: this.rate, inputs_sent: this.seq };
  }
  stop() { clearInterval(this.pingTimer); cancelAnimationFrame(this.raf); this.live = false; }
}

function startMatch(message) {
  if (match) match.stop();
  match = new Match(message);
  show("game");
  window.boplMatch = match; // for the headless check
}
function endMatch(message) {
  if (!match) return;
  const stats = match.stats();
  match.stop();
  const r = message.record; lastRecord = r;
  $("resultTitle").textContent = r.result === "win" ? "You won" : r.result === "loss" ? "You lost" : "Timeout";
  $("resultDetail").textContent = `vs ${r.opponent.label || r.b} on map ${r.map}, seed ${r.seed}, you were P${r.side + 1}; ${r.seconds.toFixed(1)} s. ` +
    `Arrows: you ${r.human.arrows_fired} fired / ${r.human.arrow_hits} hit, opponent ${r.opponent.arrows_fired} / ${r.opponent.arrow_hits}. ` +
    `Replay ${r.replay}.`;
  fillTally($("resultTally"), message.tally);
  $("resultTiming").textContent = `Sim ran at ${r.timing.realtime_ratio.toFixed(3)}x real time, step ${r.timing.step_ms_mean.toFixed(2)} ms mean / ${r.timing.step_ms_max.toFixed(1)} ms max, ` +
    `${r.timing.reanchored} re-anchors, ${r.timing.dropped_frames} dropped frames. Client: input→frame p50 ${stats.latency_p50_ms.toFixed(0)} ms, p95 ${stats.latency_p95_ms.toFixed(0)} ms; ` +
    `frame interval p50 ${stats.interval_p50_ms.toFixed(1)} ms; rtt ${stats.rtt_p50_ms.toFixed(1)} ms; ${(stats.bytes_per_s / 1024).toFixed(1)} KiB/s.`;
  window.boplLastStats = { ...stats, record: r };
  match = null;
  show("result");
}

// ---- input --------------------------------------------------------------------
window.addEventListener("keydown", (event) => {
  if (!match || !match.live) return;
  if (event.code === "Escape") { send({ T: "abort" }); return; }
  if (/^Digit[1-5]$/.test(event.code)) { match.charge = Number(event.code[5]) - 1; match.push(); event.preventDefault(); return; }
  const bit = CODES[event.code]; if (!bit) return;
  event.preventDefault();
  if (event.repeat) return;
  match.key(bit, true);
});
window.addEventListener("keyup", (event) => {
  if (!match) return;
  const bit = CODES[event.code]; if (!bit) return;
  event.preventDefault(); match.key(bit, false);
});
window.addEventListener("blur", () => match && match.release());
const arena = $("arena");
arena.addEventListener("contextmenu", (e) => e.preventDefault());
arena.addEventListener("mousedown", (e) => { e.preventDefault(); if (match && match.live && MOUSE[e.button] !== undefined) match.key(MOUSE[e.button], true); });
window.addEventListener("mouseup", (e) => { if (match && MOUSE[e.button] !== undefined) match.key(MOUSE[e.button], false); });
arena.addEventListener("auxclick", (e) => e.preventDefault());
arena.addEventListener("mousemove", (e) => { if (match) { const r = arena.getBoundingClientRect(); match.mouse = [e.clientX - r.left, e.clientY - r.top]; } });
arena.addEventListener("wheel", (e) => {
  if (!match || !match.live) return;
  e.preventDefault();
  match.charge = Math.max(0, Math.min(holdLengths.length - 1, match.charge + (e.deltaY < 0 ? 1 : -1))); match.push();
}, { passive: false });

// ---- viewer ---------------------------------------------------------------------
let viewer = null;
async function openViewer(record) {
  const visual = await (await fetch("/human/" + record.visual)).json();
  show("viewer");
  const frames = visual.frames, canvas = $("viewerCanvas"), scrub = $("viewerScrub");
  viewer = { frames, index: 0, playing: false, timer: null };
  $("viewerTitle").textContent = `${record.timestamp.replace("T", " ")} · human (P${record.side + 1}) vs ${record.b} · map ${record.map} · ${record.result}`;
  scrub.max = frames.length - 1; scrub.value = 0;
  const render = () => {
    const f = frames[viewer.index]; window.drawBoplArena(canvas, f, frames[0]);
    $("viewerTime").textContent = `${(f.t / 60).toFixed(2)} s`; scrub.value = viewer.index;
  };
  scrub.oninput = () => { viewer.index = Number(scrub.value); render(); };
  const tick = () => { if (!viewer.playing) return; viewer.index = Math.min(frames.length - 1, viewer.index + 1); render(); if (viewer.index >= frames.length - 1) { viewer.playing = false; $("viewerPlay").textContent = "Play"; } };
  $("viewerPlay").onclick = () => {
    viewer.playing = !viewer.playing; $("viewerPlay").textContent = viewer.playing ? "Pause" : "Play";
    clearInterval(viewer.timer); if (viewer.playing) { if (viewer.index >= frames.length - 1) viewer.index = 0; viewer.timer = setInterval(tick, 1000 * DECISION_S); }
  };
  render();
}
$("viewerBack").addEventListener("click", () => { if (viewer) { viewer.playing = false; clearInterval(viewer.timer); } lastRecord ? show("result") : showLobby(); });

connect();
loadLobby();
})();
