"use strict";
(() => {
  const node = (id) => document.getElementById(id);
  const filters = {Run:"run", Checkpoint:"checkpoint", Suite:"suite", Map:"map", Opponent:"opponent", Outcome:"outcome"};
  let study = "", library = [], visible = [], frames = [], playing = false, position = 0, lastTime = 0, ticket = 0, selected = null;
  const label = (key, value) => key === "checkpoint" ? `${value / 1e6}M` : String(value);
  const option = (value, text) => { const o = document.createElement("option");o.value = value;o.textContent = text;return o; };
  const message = (text) => {node("matchStatus").textContent = text;};
  async function json(url, init) {const r = await fetch(url, init);if (!r.ok) throw Error(`Replay request failed (${r.status})`);return r.json();}
  async function refresh() {
    if (!study) return;
    const requested = study;
    try {
      const data = await json(`/api/matches/${encodeURIComponent(requested)}`);
      if (study !== requested) return;
      library = data.matches;
      node("matchCoverage").textContent = `${data.saved.toLocaleString()} saved matches from ${data.evaluated.toLocaleString()} evaluated games. Only games with recorded inputs appear here. Results are from the listed run’s perspective.`;
      for (const [id, key] of Object.entries(filters)) {
        const select = node(`match${id}`), previous = select.value;
        const values = [...new Set(library.map((m) => m[key]))].sort((a,b) => typeof a === "number" ? a-b : a.localeCompare(b));
        select.replaceChildren(option("", "All"), ...values.map((v) => option(v, label(key,v))));
        if (values.some((v) => String(v) === previous)) select.value = previous;
      }
      filter();
    } catch (error) {message(error.message);}
  }
  function filter() {
    visible = library.filter((m) => Object.entries(filters).every(([id,key]) => !node(`match${id}`).value || String(m[key]) === node(`match${id}`).value));
    const previous = node("matchPicker").value;
    node("matchPicker").replaceChildren(...visible.map((m) => option(m.id, `${m.checkpoint/1e6}M · ${m.run} vs ${m.opponent} · map ${m.map} · ${m.outcome} · seed ${m.seed} · P${m.side+1}`)));
    if (visible.some((m) => m.id === previous)) node("matchPicker").value = previous;
    node("matchWatch").disabled = !visible.length;node("matchNext").disabled = !visible.length;
    if (!visible.length) message("No saved matches match these filters.");
  }
  function pause() {playing = false;node("matchPlay").textContent = "Play";}
  function draw() {
    if (!frames.length) return;
    const index = Math.min(frames.length-1, Math.floor(position)), frame = frames[index];
    window.drawBoplArena(node("matchCanvas"), frame, frames[0]);
    node("matchSeek").value = index;
    node("matchTime").textContent = `${(frame.t/60).toFixed(1)} / ${(frames.at(-1).t/60).toFixed(1)}s`;
    node("matchHud").replaceChildren(...frame.players.map((p,i) => {
      const span = document.createElement("span");
      span.textContent = `${i ? "Purple" : "Yellow"} · ${selected.players[i]} · ${p.stocks} reserve stocks · ${p.kills} kills · ${p.deaths} deaths${!(p.flags & 1) ? " · out / respawning" : ""}`;
      return span;
    }));
  }
  async function watch() {
    const match = visible.find((m) => m.id === node("matchPicker").value);if (!match) return;
    pause();const own = ++ticket, requested = study;node("matchStage").hidden = true;frames = [];
    const url = `/api/replay/${encodeURIComponent(study)}/${match.id}`;
    message("Preparing full match from its recorded inputs…");
    try {
      let result = await json(url, {method:"POST"});
      while (result.status === "queued" || result.status === "preparing") {
        message(result.status === "queued" ? "Replay queued; one match is prepared at a time." : "Replaying recorded inputs and checking the final state…");
        await new Promise((resolve) => setTimeout(resolve, 1000));
        if (own !== ticket || requested !== study) return;
        result = await json(url);
      }
      if (result.status !== "ready") throw Error(result.error || `Replay is ${result.status}`);
      const value = await json(result.url);
      if (own !== ticket || requested !== study) return;
      frames = value.frames;selected = match;position = 0;
      node("matchSeek").max = frames.length-1;node("matchStage").hidden = false;
      node("matchIdentity").textContent = `Yellow: ${match.players[0]} · Purple: ${match.players[1]} · ${match.checkpoint/1e6}M · Map ${match.map}`;
      node("matchDownload").href = match.replay_url;
      message(`Verified full match · ${match.outcome} for ${match.run} · ${match.name}`);draw();
    } catch (error) {if (own === ticket) message(error.message);}
  }
  node("matchRefresh").onclick = refresh;
  for (const id of Object.keys(filters)) node(`match${id}`).onchange = filter;
  node("matchWatch").onclick = watch;
  node("matchNext").onclick = () => {if (!visible.length) return;node("matchPicker").selectedIndex = (node("matchPicker").selectedIndex+1)%visible.length;watch();};
  node("matchPlay").onclick = () => {if (!frames.length) return;if (playing) pause();else {if (position >= frames.length-1) position=0;playing=true;lastTime=performance.now();node("matchPlay").textContent="Pause";}};
  node("matchSeek").oninput = () => {pause();position=Number(node("matchSeek").value);draw();};
  node("matchBack").onclick = () => {pause();position=Math.max(0,Math.floor(position)-1);draw();};
  node("matchForward").onclick = () => {pause();position=Math.min(frames.length-1,Math.floor(position)+1);draw();};
  function animate(now) {
    if (playing && frames.length) {
      position = Math.min(frames.length-1,position+(now-lastTime)/1000*15*Number(node("matchSpeed").value));
      if (position >= frames.length-1) pause();draw();
    }
    lastTime=now;requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);
  window.addEventListener("study-data", (event) => {
    if (study === event.detail.name) return;
    study=event.detail.name;ticket++;pause();frames=[];library=[];node("matchStage").hidden=true;
    for (const id of Object.keys(filters)) node(`match${id}`).replaceChildren();
    refresh();
  });
  window.addEventListener("resize", draw);
  setInterval(refresh,30000);
})();
