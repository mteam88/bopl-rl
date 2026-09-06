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
  const video = await evaluate(
    "({time:document.querySelector('#video').currentTime,ready:document.querySelector('#video').readyState})",
  );
  if (video.ready < 2 || video.time <= 0)
    throw Error("Native video did not play: " + JSON.stringify(video));
  await evaluate("document.querySelector('#video').pause()");
}
const desktop = await call("Page.captureScreenshot", { format: "png" });
await fs.writeFile(
  out + "/dashboard-desktop.png",
  Buffer.from(desktop.data, "base64"),
);
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
    mobile_overflow: overflow,
    console_errors: errors.length,
  }),
);
ws.close();
