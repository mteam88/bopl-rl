// Run against an isolated Chrome --headless --remote-debugging-port=9224.
import fs from "node:fs/promises";
const url =
  process.argv[2] || "http://100.100.57.32:8876/?run=research-smoke-2";
const out = process.argv[3] || "native/artifacts/rl";
const pages = await (await fetch("http://127.0.0.1:9224/json/list")).json();
const page = pages.find((p) => p.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((resolve) =>
  ws.addEventListener("open", resolve, { once: true }),
);
let id = 0;
const pending = new Map(),
  errors = [];
ws.addEventListener("message", (event) => {
  const message = JSON.parse(event.data);
  if (message.id) {
    const item = pending.get(message.id);
    if (item) {
      pending.delete(message.id);
      clearTimeout(item.timer);
      message.error ? item.reject(message.error) : item.resolve(message.result);
    }
  } else if (message.method === "Runtime.exceptionThrown")
    errors.push(message.params.exceptionDetails);
});
function call(method, params = {}) {
  return new Promise((resolve, reject) => {
    const next = ++id;
    const timer = setTimeout(() => {
      pending.delete(next);
      reject(Error(method + " timed out"));
    }, 10000);
    pending.set(next, { resolve, reject, timer });
    ws.send(JSON.stringify({ id: next, method, params }));
  });
}
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
async function evaluate(expression) {
  const result = await call("Runtime.evaluate", {
    expression,
    returnByValue: true,
    awaitPromise: true,
  });
  if (result.exceptionDetails)
    throw Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
await call("Page.stopLoading");
await call("Page.navigate", { url: "about:blank" });
await call("Runtime.enable");
await call("Page.enable");
await call("Emulation.setDeviceMetricsOverride", {
  width: 1440,
  height: 1400,
  deviceScaleFactor: 1,
  mobile: false,
});
await call("Page.navigate", { url });
for (let i = 0; i < 30; i++) {
  await sleep(200);
  if (
    await evaluate(
      "document.querySelector('#play') && !document.querySelector('#play').disabled",
    )
  )
    break;
}
const loaded = await evaluate(
  "({steps:document.querySelector('#steps').textContent,replay:!document.querySelector('#play').disabled,wide:document.documentElement.scrollWidth>innerWidth})",
);
if (!loaded.replay || loaded.wide)
  throw Error("Desktop load failed: " + JSON.stringify(loaded));
await evaluate(
  "document.querySelector('#viewPhysics').click();document.querySelector('#scrub').value=60;document.querySelector('#scrub').dispatchEvent(new Event('input'))",
);
const before = await evaluate(
  "document.querySelector('#replayTime').textContent",
);
await evaluate("document.querySelector('#play').click()");
await sleep(400);
const after = await evaluate(
  "document.querySelector('#replayTime').textContent",
);
if ((await evaluate("Number(document.querySelector('#scrub').value)")) <= 60)
  throw Error("Replay did not advance");
await evaluate("document.querySelector('#play').click()");
const videoAvailable = await evaluate(
  "!document.querySelector('#viewVideo').disabled",
);
if (videoAvailable) {
  await evaluate(
    "document.querySelector('#viewVideo').click();document.querySelector('#video').muted=true;document.querySelector('#video').play()",
  );
  await sleep(600);
  const targetTime = await evaluate("document.querySelector('#video').currentTime=Math.min(7,document.querySelector('#video').duration/2)");
  let video;
  for (let i = 0; i < 50; i++) {
    await sleep(200);
    video = await evaluate(
      "(() => {const v=document.querySelector('#video');return {time:v.currentTime,ready:v.readyState,seeking:v.seeking,paused:v.paused}})()",
    );
    if (video.ready >= 2 && !video.seeking && !video.paused && video.time > targetTime + .05) break;
  }
  if (video.ready < 2 || video.seeking || video.paused || video.time <= targetTime + .05)
    throw Error("Native video did not play: " + JSON.stringify(video));
  await evaluate("document.querySelector('#video').pause()");
}
const matrix = await evaluate(`(() => {
  const maps = [...document.querySelector('#winMap').options].map(o => Number(o.value));
  const expected = data.config.eval_maps || [0, 1, 2];
  const report = data.final_evaluation || data.validation_evaluation || data.evaluations.at(-1);
  return {maps, expected, rows: document.querySelectorAll('#evaluationRows tr').length,
    groups: report?.groups.length || 0, coverage: document.querySelector('#mapCoverage').textContent,
    frozen: (data.evaluations.at(-1)?.frozen_opponents || data.config.frozen_opponents || []).length,
    legend: document.querySelector('#thirdWinLegend').textContent};
})()`);
if (JSON.stringify(matrix.maps) !== JSON.stringify(matrix.expected) || matrix.rows !== matrix.groups)
  throw Error("Evaluation matrix mismatch: " + JSON.stringify(matrix));
if (matrix.frozen && !matrix.legend.includes("Previous trained policies"))
  throw Error("Frozen opponent legend missing");
if (matrix.maps.length > 1) {
  await evaluate(`document.querySelector('#winMap').value='${matrix.maps[1]}';document.querySelector('#winMap').dispatchEvent(new Event('change'))`);
  if (await evaluate("Number(document.querySelector('#winMap').value)") !== matrix.maps[1])
    throw Error("Map selector did not retain selection");
  await evaluate(`document.querySelector('#winMap').value='${matrix.maps[0]}';document.querySelector('#winMap').dispatchEvent(new Event('change'))`);
}
const desktop = await call("Page.captureScreenshot", { format: "png" });
await fs.writeFile(
  out + "/dashboard-desktop.png",
  Buffer.from(desktop.data, "base64"),
);
const behavioral = await evaluate(
  "document.querySelector('#behaviorSection') ? ({notice:document.querySelector('#behaviorNotice').textContent, charts:['abilitiesChart','arrowsChart','deathsChart','rewardsChart'].every(id=>document.getElementById(id).width>0&&document.getElementById(id).height>0)}) : null",
);
if (behavioral) {
  if (!behavioral.charts) throw Error("Behavioral charts did not render");
  await evaluate("document.querySelector('#behaviorSection').scrollIntoView()");
  const behavior = await call("Page.captureScreenshot", { format: "png" });
  await fs.writeFile(out + "/dashboard-behavior.png", Buffer.from(behavior.data, "base64"));
  await evaluate("scrollTo(0,0)");
}
await call("Emulation.setDeviceMetricsOverride", {
  width: 393,
  height: 852,
  deviceScaleFactor: 1,
  mobile: true,
});
await sleep(300);
const overflow = await evaluate(
  "document.documentElement.scrollWidth>innerWidth",
);
if (overflow) throw Error("Mobile horizontal overflow");
const mobile = await call("Page.captureScreenshot", {
  format: "png",
  captureBeyondViewport: true,
});
await fs.writeFile(
  out + "/dashboard-mobile.png",
  Buffer.from(mobile.data, "base64"),
);
if (errors.length) throw Error("Browser errors: " + JSON.stringify(errors));
console.log(
  JSON.stringify({
    status: "DASHBOARD_OK",
    ...loaded,
    before,
    after,
    video_playback: videoAvailable,
    behavioral,
    matrix,
    mobile_overflow: overflow,
    console_errors: errors.length,
  }),
);
ws.close();
