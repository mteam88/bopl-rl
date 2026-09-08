// Dashboard smoke check over WebDriver BiDi (Firefox).
//
// Chrome 152 headless cannot commit any http(s) navigation on this machine,
// so the check drives Firefox instead:
//   firefox --headless --remote-debugging-port 9225 --profile <fresh> --no-remote
// Firefox allows one BiDi session per browser process; restart it between runs.
//   node check_dashboard.mjs <url> <output-dir>
import fs from "node:fs/promises";
const url = process.argv[2] || "http://127.0.0.1:8876/?run=research-smoke-2";
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
for (let i = 0; i < 40; i++) {
  await sleep(250);
  if (await evaluate("!!(typeof data !== 'undefined' && data && data.metrics)")) break;
}
// Runs without saved evaluation games legitimately have no replay to play.
const expectReplay = await evaluate(
  "(data.evaluations || []).some(e => (e.games || []).some(g => g.save))",
);
if (expectReplay)
  for (let i = 0; i < 40; i++) {
    await sleep(250);
    if (await evaluate("!document.querySelector('#play').disabled")) break;
  }
const loaded = await evaluate(
  "({steps: document.querySelector('#steps').textContent, replay: !document.querySelector('#play').disabled, wide: document.documentElement.scrollWidth > innerWidth, truncations: document.querySelector('#truncations')?.textContent ?? null})",
);
if ((expectReplay && !loaded.replay) || loaded.wide)
  throw Error("Desktop load failed: " + JSON.stringify(loaded));

// Physics replay advances.
let before = null,
  after = null;
if (expectReplay) {
  await run(
    "document.querySelector('#viewPhysics').click(); const scrub = document.querySelector('#scrub'); scrub.value = 60; scrub.dispatchEvent(new Event('input'));",
  );
  before = await evaluate("document.querySelector('#replayTime').textContent");
  await run("document.querySelector('#play').click()");
  await sleep(400);
  after = await evaluate("document.querySelector('#replayTime').textContent");
  if ((await evaluate("Number(document.querySelector('#scrub').value)")) <= 60)
    throw Error("Replay did not advance");
  await run("document.querySelector('#play').click()");
}

// Native video, tolerant to a headless browser without the codec.
let video = null;
if (expectReplay && (await evaluate("!document.querySelector('#viewVideo').disabled"))) {
  await run(
    "document.querySelector('#viewVideo').click(); const v = document.querySelector('#video'); v.muted = true; v.play();",
  );
  for (let i = 0; i < 50; i++) {
    await sleep(200);
    video = await evaluate(
      "(() => { const v = document.querySelector('#video'); return { time: v.currentTime, ready: v.readyState, paused: v.paused, error: v.error?.code ?? null }; })()",
    );
    if (video.error || (video.ready >= 2 && !video.paused && video.time > 0.5)) break;
  }
  if (video.error) video = { skipped: "codec unsupported in this browser", ...video };
  else if (video.ready < 2 || video.time <= 0.5)
    throw Error("Native video did not play: " + JSON.stringify(video));
  await run("document.querySelector('#video').pause()");
}

// Evaluation matrix consistency.
const matrix = await evaluate(`(() => {
  if (!(data.evaluations || []).length) return null;
  const maps = [...document.querySelector('#winMap').options].map(o => Number(o.value));
  const expected = data.config.eval_maps || (data.config.evaluation_maps ? data.config.evaluation_maps.concat(data.config.heldout_maps || []) : [0, 1, 2]);
  const report = data.final_evaluation || data.validation_evaluation || data.evaluations.at(-1);
  return {maps, expected, rows: document.querySelectorAll('#evaluationRows tr').length,
    groups: report?.groups.length || 0, coverage: document.querySelector('#mapCoverage').textContent,
    frozen: (data.evaluations.at(-1)?.frozen_opponents || data.config.frozen_opponents || []).length,
    legend: document.querySelector('#thirdWinLegend').textContent};
})()`);
if (matrix && (JSON.stringify(matrix.maps) !== JSON.stringify(matrix.expected) || matrix.rows !== matrix.groups))
  throw Error("Evaluation matrix mismatch: " + JSON.stringify(matrix));
if (matrix?.frozen && !matrix.legend.includes("Previous trained policies"))
  throw Error("Frozen opponent legend missing");

// Learning-health panels: entropy by head, schedules, advantage spread.
const health = await evaluate(
  "({hidden: document.querySelector('#healthSection').hidden, charts: ['entropyChart','scheduleChart','advantageChart','lossChart','episodeLengthChart'].map(id => document.getElementById(id).width > 0)})",
);
if (health.hidden || !health.charts.every(Boolean))
  throw Error("Health panels did not render: " + JSON.stringify(health));

// Behavior-over-time charts.
const behavioral = await evaluate(
  "document.querySelector('#behaviorSection') && !document.querySelector('#behaviorSection').hidden ? ({notice: document.querySelector('#behaviorNotice').textContent, charts: ['abilitiesChart','arrowsChart','deathsChart','rewardsChart','endingsChart','accuracyChart'].every(id => document.getElementById(id).width > 0)}) : null",
);
if (behavioral && !behavioral.charts) throw Error("Behavioral charts did not render");

// Human-comparison gauges and heatmaps follow insights availability.
const insights = await evaluate(
  "({available: (data.insights || []).length > 0, gauges: ['bowHoldChart','aimChangeChart'].map(id => document.getElementById(id).width > 0), heatmapHidden: document.querySelector('#heatmapSection').hidden, heatmapPainted: (() => { const c = document.getElementById('heatmap'); return c.width > 0 && c.getContext('2d').getImageData(0, 0, c.width, c.height).data.some(v => v > 0); })()})",
);
if (insights.available && (!insights.gauges.every(Boolean) || insights.heatmapHidden || !insights.heatmapPainted))
  throw Error("Insights panels did not render: " + JSON.stringify(insights));

// ELO ladder renders when ladder.json exists.
const ladder = await evaluate(
  "data.ladder ? ({hidden: document.querySelector('#ladderSection').hidden, painted: document.getElementById('ladderChart').width > 0, tag: document.querySelector('#ladderTag').textContent}) : null",
);
if (ladder && (ladder.hidden || !ladder.painted))
  throw Error("Ladder did not render: " + JSON.stringify(ladder));

// Policy-mind strip renders when the selected game has a decision dump.
let mind = null;
if (await evaluate("(data.evaluations.at(-1)?.games || []).some(g => g.mind)")) {
  await run(
    "const match = document.getElementById('match'); const withMind = data.evaluations.at(-1).games.filter(g => g.mind && g.save).map(g => g.name); match.value = withMind[0]; match.dispatchEvent(new Event('change')); await new Promise(r => setTimeout(r, 2500)); const scrub = document.getElementById('scrub'); scrub.value = Math.floor(scrub.max / 2); scrub.dispatchEvent(new Event('input')); await new Promise(r => setTimeout(r, 400));",
  );
  mind = await evaluate(
    "({hidden: document.getElementById('mindPanel').hidden, readout: document.getElementById('mindReadout').textContent, painted: ['mindValue','mindButtons','mindAim'].every(id => document.getElementById(id).width > 0)})",
  );
  if (mind.hidden || !mind.painted || !mind.readout.includes("V(s)"))
    throw Error("Mind panel did not render: " + JSON.stringify(mind));
}

await screenshot(out + "/dashboard-desktop.png");
if (behavioral) {
  await run("document.querySelector('#behaviorSection').scrollIntoView()");
  await screenshot(out + "/dashboard-behavior.png");
  await run("scrollTo(0, 0)");
}

await viewport(393, 852);
await sleep(400);
const overflow = await evaluate("document.documentElement.scrollWidth > innerWidth");
if (overflow) throw Error("Mobile horizontal overflow");
await screenshot(out + "/dashboard-mobile.png");
// Only errors from our own page; the browser's startup tab logs its own noise.
const pageErrors = consoleErrors.filter((e) => e.context === context).map((e) => e.text);
if (pageErrors.length) throw Error("Browser errors: " + JSON.stringify(pageErrors));
console.log(
  JSON.stringify({
    status: "DASHBOARD_OK",
    ...loaded,
    before,
    after,
    video,
    matrix,
    health: !health.hidden,
    behavioral: !!behavioral,
    insights,
    ladder,
    mind,
    mobile_overflow: overflow,
    console_errors: pageErrors.length,
  }),
);
ws.close();
process.exit(0);
