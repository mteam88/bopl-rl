// Dashboard smoke check over WebDriver BiDi (Firefox).
//   firefox --headless --remote-debugging-port 9225 --profile <fresh> --no-remote
//   node check_dashboard.mjs <url> <output-dir>
import fs from "node:fs/promises";
const url = process.argv[2] || "http://127.0.0.1:8890/";
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
    const timer = setTimeout(() => {
      pending.delete(next);
      reject(Error(method + " timed out"));
    }, 30000);
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
    expression: `JSON.stringify((() => (${expression}))() ?? null)`,
    target: { context },
    awaitPromise: true,
  });
  if (response.result.type !== "success")
    throw Error("evaluate failed: " + JSON.stringify(response.result).slice(0, 500));
  return JSON.parse(response.result.result.value);
}
async function run(expression) {
  const response = await call("script.evaluate", {
    expression: `(async () => { ${expression} })()`,
    target: { context },
    awaitPromise: true,
  });
  if (response.result.type !== "success")
    throw Error("run failed: " + JSON.stringify(response.result).slice(0, 500));
}
async function screenshot(path) {
  const shot = await call("browsingContext.captureScreenshot", { context, origin: "document" });
  await fs.writeFile(path, Buffer.from(shot.result.data, "base64"));
}
async function viewport(width, height) {
  await call("browsingContext.setViewport", { context, viewport: { width, height } });
}

await viewport(1440, 1400);
await call("browsingContext.navigate", { context, url, wait: "complete" });
consoleErrors.length = 0;
for (let i = 0; i < 40; i++) {
  await sleep(250);
  if (await evaluate("document.querySelectorAll('#runRows tr').length > 0")) break;
}
const loaded = await evaluate(`({
  rows: document.querySelectorAll('#runRows tr').length,
  families: document.querySelectorAll('#families .family').length,
  wide: document.documentElement.scrollWidth > innerWidth,
  tag: document.querySelector('#indexTag')?.textContent ?? null
})`);
if (!loaded.rows || loaded.wide) throw Error("Index load failed: " + JSON.stringify(loaded));
await screenshot(out + "/dashboard-index.png");

const ladderUrl = new URL("/ladder", url).href;
await call("browsingContext.navigate", { context, url: ladderUrl, wait: "complete" });
for (let i = 0; i < 40; i++) {
  await sleep(250);
  if (await evaluate("document.querySelectorAll('#ladderRows tr').length > 0")) break;
}
const ladderLoaded = await evaluate(`({
  rows: document.querySelectorAll('#ladderRows tr').length,
  human: [...document.querySelectorAll('#ladderRows tr')].some(r => r.classList.contains('human')),
  wide: document.documentElement.scrollWidth > innerWidth,
  tag: document.querySelector('#ladderTag')?.textContent ?? null
})`);
if (!ladderLoaded.rows || ladderLoaded.wide) throw Error("Ladder load failed: " + JSON.stringify(ladderLoaded));

const runHref = await evaluate(`[...document.querySelectorAll('#ladderRows a')].map(a => a.getAttribute('href')).find(h => h && h.startsWith('/run/')) ?? null`);
await run("document.querySelector('#sort').value='name'; document.querySelector('#sort').dispatchEvent(new Event('change'));");
const names = await evaluate("[...document.querySelectorAll('#ladderRows tr')].slice(0,5).map(r => r.children[1]?.textContent)");
await screenshot(out + "/dashboard-ladder.png");
let runView = null;
if (runHref) {
  await call("browsingContext.navigate", { context, url: new URL(runHref, url).href, wait: "complete" });
  for (let i = 0; i < 40; i++) {
    await sleep(250);
    if (await evaluate("!document.querySelector('#runView').hidden")) break;
  }
  runView = await evaluate(`({
    elo: document.querySelector('#runElo')?.textContent,
    chart: document.querySelector('#eloChart')?.width > 0,
    replay: !document.querySelector('#play')?.disabled,
    wide: document.documentElement.scrollWidth > innerWidth
  })`);
  if (!runView.elo || runView.wide) throw Error("Run view failed: " + JSON.stringify(runView));
  if (runView.replay) {
    await run("document.querySelector('#play').click()");
    await sleep(400);
    await run("document.querySelector('#play').click()");
  }
  await screenshot(out + "/dashboard-run.png");
}

const sweepHref = await evaluate("document.querySelector('#sweepNav:not([hidden])') ? document.querySelector('#sweepNav').href : null");
let sweepView = null;
if (sweepHref) {
  await call("browsingContext.navigate", { context, url: sweepHref, wait: "complete" });
  for (let i = 0; i < 40; i++) {
    await sleep(250);
    if (await evaluate("!document.querySelector('#sweepView').hidden && document.querySelectorAll('#sweepRows tr').length > 0")) break;
  }
  sweepView = await evaluate(`({
    rows: document.querySelectorAll('#sweepRows tr').length,
    chart: document.querySelector('#scoreChart')?.width > 0,
    wide: document.documentElement.scrollWidth > innerWidth
  })`);
  if (!sweepView.rows || sweepView.wide) throw Error("Sweep view failed: " + JSON.stringify(sweepView));
  await screenshot(out + "/dashboard-sweep.png");
}

const lineupsUrl = new URL("/lineups", url).href;
await call("browsingContext.navigate", { context, url: lineupsUrl, wait: "complete" });
for (let i = 0; i < 40; i++) {
  await sleep(250);
  if (await evaluate("!document.querySelector('#lineupsView').hidden && document.querySelectorAll('#matrixRows tr').length > 0")) break;
}
const lineupsView = await evaluate(`({
  runs: document.querySelectorAll('#matrixRows tr').length,
  columns: document.querySelectorAll('#matrixHead th.lineup-col').length,
  comparable: document.querySelectorAll('#matrixHead th.comparable').length,
  wide: document.documentElement.scrollWidth > innerWidth
})`);
if (lineupsView.wide) throw Error("Lineups view failed: " + JSON.stringify(lineupsView));
if (lineupsView.runs) await screenshot(out + "/dashboard-lineups.png");
for (const metric of ["random", "rating", "games", "scripted"]) {
  await run(`const s = document.querySelector('#matrixMetric'); s.value = ${JSON.stringify(metric)}; s.dispatchEvent(new Event('change'));`);
  const filled = await evaluate("document.querySelectorAll('#matrixRows td.num').length");
  if (lineupsView.runs && !filled) throw Error("Lineups metric empty: " + metric);
}
// A multi-loadout run: the loadouts section must open, not collapse to its one line.
const multiHref = await evaluate(`[...document.querySelectorAll('#matrixRows tr')]
  .filter(r => r.querySelector('.pill.set, .pill.pool'))
  .map(r => r.querySelector('a')?.getAttribute('href'))[0] ?? null`);
let loadoutView = null;
if (multiHref) {
  await call("browsingContext.navigate", { context, url: new URL(multiHref, url).href, wait: "complete" });
  for (let i = 0; i < 40; i++) {
    await sleep(250);
    if (await evaluate("document.querySelectorAll('#abilityRows tr').length > 0")) break;
  }
  loadoutView = await evaluate(`({
    badge: document.querySelector('#loadoutBadge')?.textContent,
    abilities: document.querySelectorAll('#abilityRows tr').length,
    silent: document.querySelectorAll('#abilityRows tr.silent').length,
    lineups: document.querySelectorAll('#runLineupRows tr').length,
    sparks: document.querySelectorAll('#runLineupRows svg.spark').length,
    detail: !document.querySelector('#loadoutDetail').hidden,
    wide: document.documentElement.scrollWidth > innerWidth
  })`);
  if (!loadoutView.detail || !loadoutView.lineups || loadoutView.wide)
    throw Error("Loadout section failed: " + JSON.stringify(loadoutView));
  await screenshot(out + "/dashboard-loadouts.png");
}

await viewport(390, 844);
await call("browsingContext.navigate", { context, url, wait: "complete" });
for (let i = 0; i < 40; i++) {
  await sleep(250);
  if (await evaluate("document.querySelectorAll('#runRows tr').length > 0")) break;
}
const overflow = await evaluate("document.documentElement.scrollWidth > innerWidth");
if (overflow) throw Error("Mobile horizontal overflow");
await screenshot(out + "/dashboard-mobile.png");
if (multiHref) {
  await call("browsingContext.navigate", { context, url: new URL(multiHref, url).href, wait: "complete" });
  for (let i = 0; i < 40; i++) {
    await sleep(250);
    if (await evaluate("document.querySelectorAll('#runLineupRows tr').length > 0")) break;
  }
  if (await evaluate("document.documentElement.scrollWidth > innerWidth"))
    throw Error("Mobile horizontal overflow on the loadouts section");
  await screenshot(out + "/dashboard-mobile-loadouts.png");
}

const pageErrors = consoleErrors.filter((e) => e.context === context).map((e) => e.text);
if (pageErrors.length) throw Error("Browser errors: " + JSON.stringify(pageErrors));
console.log(JSON.stringify({
  status: "DASHBOARD_OK",
  loaded,
  ladderLoaded,
  names,
  runView,
  sweepView,
  lineupsView,
  loadoutView,
  mobile_overflow: overflow,
  console_errors: pageErrors.length,
}));
// Firefox allows one BiDi session at a time; end it so a rerun can connect.
await call("session.end", {}).catch(() => {});
ws.close();
process.exit(0);
