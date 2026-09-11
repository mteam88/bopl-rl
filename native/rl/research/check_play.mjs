// Play page smoke check over WebDriver BiDi (Firefox), like check_dashboard.mjs:
//   firefox --headless --remote-debugging-port 9225 --profile <fresh> --no-remote
//   node check_play.mjs <url> <output-dir>
// Loads the lobby, starts a match against mixed-hard, mashes keys for a few
// seconds, waits for the match to end, and checks: frames arrive at the
// decision rate (15/s), the record and replay were written, no console errors.
import fs from "node:fs/promises";
const url = process.argv[2] || "http://127.0.0.1:8891/";
const out = process.argv[3] || "native/artifacts/rl";
const port = process.env.BIDI_PORT || "9225";

const ws = new WebSocket(`ws://127.0.0.1:${port}/session`);
await new Promise((resolve, reject) => {
  ws.addEventListener("open", resolve, { once: true });
  ws.addEventListener("error", () => reject(Error("BiDi port unreachable")), { once: true });
});
let id = 0;
const pending = new Map();
const consoleErrors = [];
ws.addEventListener("message", (event) => {
  const message = JSON.parse(event.data);
  if (message.id && pending.has(message.id)) {
    const item = pending.get(message.id);
    pending.delete(message.id);
    clearTimeout(item.timer);
    message.type === "error" ? item.reject(Error(JSON.stringify(message))) : item.resolve(message);
  } else if (message.method === "log.entryAdded" && message.params.level === "error")
    consoleErrors.push({ context: message.params.source?.context, text: message.params.text });
});
function call(method, params = {}) {
  return new Promise((resolve, reject) => {
    const next = ++id;
    const timer = setTimeout(() => { pending.delete(next); reject(Error(method + " timed out")); }, 30000);
    pending.set(next, { resolve, reject, timer });
    ws.send(JSON.stringify({ id: next, method, params }));
  });
}
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
await call("session.new", { capabilities: {} });
await call("session.subscribe", { events: ["log.entryAdded"] });
const context = (await call("browsingContext.getTree", {})).result.contexts[0].context;
async function evaluate(expression) {
  const response = await call("script.evaluate", {
    expression: `JSON.stringify((() => (${expression}))() ?? null)`, target: { context }, awaitPromise: true,
  });
  if (response.result.type !== "success") throw Error("evaluate failed: " + JSON.stringify(response.result).slice(0, 500));
  return JSON.parse(response.result.result.value);
}
async function run(expression) {
  const response = await call("script.evaluate", { expression: `(async () => { ${expression} })()`, target: { context }, awaitPromise: true });
  if (response.result.type !== "success") throw Error("run failed: " + JSON.stringify(response.result).slice(0, 500));
}
async function screenshot(path) {
  const shot = await call("browsingContext.captureScreenshot", { context, origin: "document" });
  await fs.writeFile(path, Buffer.from(shot.result.data, "base64"));
}

await call("browsingContext.setViewport", { context, viewport: { width: 1280, height: 900 } });
await call("browsingContext.navigate", { context, url, wait: "complete" });
for (let i = 0; i < 40; i++) {
  await sleep(250);
  if (await evaluate("document.querySelector('#opponent').options.length > 0 && document.querySelector('#status').textContent === 'connected'")) break;
}
const lobby = await evaluate("({opponents: document.querySelector('#opponent').options.length, bindings: document.querySelectorAll('#bindings tbody tr').length, status: document.querySelector('#status').textContent})");
if (!lobby.opponents || !lobby.bindings || lobby.status !== "connected") throw Error("Lobby did not load: " + JSON.stringify(lobby));
const gamesBefore = (await (await fetch(new URL("/api/games", url))).json()).length;
await screenshot(out + "/play-lobby.png");

// Start against mixed-hard as P1 in keys mode, then mash movement/jump/bow like a novice.
await run("document.querySelector('#opponent').value = 'mixed-hard'; document.querySelector('#map').value = '0'; document.querySelector('#side').value = '0'; document.querySelector('#mode').value = 'keys'; document.querySelector('#start').click();");
for (let i = 0; i < 40; i++) { await sleep(250); if (await evaluate("!!window.boplMatch && !document.querySelector('#game').hidden")) break; }
if (!(await evaluate("!!window.boplMatch"))) throw Error("Match did not start");
const key = (code, type) => `window.dispatchEvent(new KeyboardEvent('${type}', {code: '${code}', bubbles: true}))`;
const script = ["KeyD", "Space", "KeyK", "KeyA", "Space", "KeyD", "KeyJ", "KeyK", "KeyW", "KeyD"];
let ended = false, midStats = null;
for (let round = 0; round < 240 && !ended; round++) {
  const code = script[round % script.length];
  await run(`${key(code, "keydown")}; await new Promise(r => setTimeout(r, 120)); ${key(code, "keyup")};`);
  await sleep(130);
  if (round === 12) { midStats = await evaluate("window.boplMatch ? window.boplMatch.stats() : null"); await screenshot(out + "/play-match.png"); }
  ended = await evaluate("!!window.boplLastStats");
}
if (!ended) throw Error("Match did not end within the key script");
const stats = await evaluate("window.boplLastStats");
await screenshot(out + "/play-result.png");
const record = stats.record;
if (!(stats.frames > 10)) throw Error("Too few frames: " + JSON.stringify(stats));
// One frame per 4-tick decision at 60 Hz: 66.7 ms; allow scheduling jitter on a loaded box.
if (!(stats.interval_p50_ms > 60 && stats.interval_p50_ms < 74)) throw Error("Frame interval off the decision rate: " + JSON.stringify(stats));
if (record.ticks !== stats.frames * 4 && Math.abs(record.ticks - stats.frames * 4) > 4) throw Error("Frames do not cover the episode ticks: " + JSON.stringify({ ticks: record.ticks, frames: stats.frames }));
const games = await (await fetch(new URL("/api/games", url))).json();
if (games.length !== gamesBefore + 1 || games.at(-1).replay !== record.replay) throw Error("games.jsonl did not gain the record");
const replay = await fetch(new URL("/human/" + record.replay, url));
const replayBytes = (await replay.arrayBuffer()).byteLength;
if (!replay.ok || replayBytes < 100) throw Error("Replay missing: " + record.replay);
const visual = await (await fetch(new URL("/human/" + record.visual, url))).json();
if (visual.frames.length !== stats.frames + 1) throw Error("Visual frame count differs from streamed frames: " + JSON.stringify({ visual: visual.frames.length, streamed: stats.frames }));
// Viewer renders the recorded game.
await run("document.querySelector('#watchLast').click(); await new Promise(r => setTimeout(r, 1500));");
const viewer = await evaluate("({hidden: document.querySelector('#viewer').hidden, max: Number(document.querySelector('#viewerScrub').max), painted: document.getElementById('viewerCanvas').width > 0})");
if (viewer.hidden || viewer.max < 10 || !viewer.painted) throw Error("Viewer did not render: " + JSON.stringify(viewer));

// Second match in mouse mode as P2: aim with the mouse and press the right button (bow).
await run("document.querySelector('#viewerBack').click(); document.querySelector('#toLobby').click(); await new Promise(r => setTimeout(r, 800)); document.querySelector('#side').value = '1'; document.querySelector('#mode').value = 'mouse'; document.querySelector('#start').click();");
for (let i = 0; i < 40; i++) { await sleep(250); if (await evaluate("!!window.boplMatch && window.boplMatch.mode === 'mouse'")) break; }
if (!(await evaluate("!!window.boplMatch && window.boplMatch.mode === 'mouse'"))) throw Error("Mouse-mode match did not start");
const mouse = (type, x, y, button) => `(() => { const c = document.getElementById('arena'), r = c.getBoundingClientRect(); c.dispatchEvent(new MouseEvent('${type}', {clientX: r.left + r.width * ${x}, clientY: r.top + r.height * ${y}, button: ${button}, bubbles: true})); })()`;
let mouseEnded = false, aimSeen = false;
for (let round = 0; round < 240 && !mouseEnded; round++) {
  const x = 0.3 + 0.4 * ((round % 7) / 6), y = 0.2 + 0.5 * ((round % 5) / 4);
  await run(`${mouse("mousemove", x, y, 0)}; await new Promise(r => setTimeout(r, 40)); ${round % 6 === 0 ? mouse("mousedown", x, y, 2) : key("KeyA", "keydown")}; await new Promise(r => setTimeout(r, 120)); ${round % 6 === 0 ? mouse("mouseup", x, y, 2) : key("KeyA", "keyup")};`);
  await sleep(90);
  aimSeen ||= await evaluate("window.boplMatch ? window.boplMatch.aim !== null : false");
  mouseEnded = await evaluate("!!window.boplLastStats && window.boplLastStats.record.human.input_mode === 'mouse'");
}
if (!mouseEnded) throw Error("Mouse-mode match did not end");
const mouseStats = await evaluate("window.boplLastStats");
if (!aimSeen || !(mouseStats.record.human.aim_source_decisions.mouse > 0)) throw Error("Mouse aim never reached the action: " + JSON.stringify({ aimSeen, sources: mouseStats.record.human.aim_source_decisions }));
if (mouseStats.record.side !== 1) throw Error("Side selection ignored");
// Only our page's errors: the tab's Firefox start page logs its own React
// hydration noise into the same browsing context before we navigate away.
const pageErrors = consoleErrors.filter((e) => e.context === context && !/^Hydration completed/.test(e.text)).map((e) => e.text);
if (pageErrors.length) throw Error("Browser errors: " + JSON.stringify(pageErrors));
console.log(JSON.stringify({
  status: "PLAY_OK", lobby, mid_match: midStats,
  frames: stats.frames, interval_p50_ms: stats.interval_p50_ms, interval_p95_ms: stats.interval_p95_ms, interval_max_ms: stats.interval_max_ms,
  input_to_frame_p50_ms: stats.latency_p50_ms, input_to_frame_p95_ms: stats.latency_p95_ms, rtt_p50_ms: stats.rtt_p50_ms,
  bytes_per_s: stats.bytes_per_s, inputs_sent: stats.inputs_sent,
  result: record.result, ticks: record.ticks, replay: record.replay, replay_bytes: replayBytes, timing: record.timing,
  mouse_mode: { result: mouseStats.record.result, ticks: mouseStats.record.ticks, aim_sources: mouseStats.record.human.aim_source_decisions,
    arrows_fired: mouseStats.record.human.arrows_fired, input_to_frame_p50_ms: mouseStats.latency_p50_ms, frames: mouseStats.frames },
  console_errors: pageErrors.length,
}));
ws.close();
process.exit(0);
