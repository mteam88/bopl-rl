"use strict";
const $ = (id) => document.getElementById(id);
const palette = { A: "#61798d", B: "#df872d", C: "#287f72" };
const count = new Intl.NumberFormat("en-US");
const pct = (n) => `${(100 * n).toFixed(1)}%`;
const million = (n) => `${(n / 1e6).toFixed(2)}M`;
let currentStudy = null, studyName = new URLSearchParams(location.search).get("study") || "", loading = false;
function element(tag, text = "", className = "") {
  const node = document.createElement(tag); node.textContent = text;
  if (className) node.className = className;
  return node;
}
function options(select, entries) {
  const previous = select.value;
  select.replaceChildren(...entries.map(([value, label]) => {
    const option = element("option", label); option.value = value; return option;
  }));
  if (entries.some(([value]) => value === previous)) select.value = previous;
}
function connection(ok) {
  $("connection").textContent = ok ? "Live · refreshes every 5s" : "Connection interrupted";
  $("connection").parentElement.classList.toggle("online", ok);
}
async function refresh() {
  if (loading) return;
  loading = true;
  try {
    const response = await fetch("/api/studies");
    if (!response.ok) throw Error("Study list unavailable");
    const studies = await response.json();
    if (!studies.length) throw Error("No opponent studies found yet.");
    if (!studyName) studyName = studies[0].name;
    options($("studyPicker"), studies.map((s) => [s.name, s.title]));
    $("studyPicker").value = studyName;
    const requested = studyName;
    const result = await fetch(`/api/study/${encodeURIComponent(requested)}`);
    if (!result.ok) throw Error("Study data unavailable");
    const next = await result.json();
    if (requested !== studyName) return;
    currentStudy = next; connection(true); render();
  } catch (error) {
    connection(false); $("studyNotice").textContent = error.message; $("studyNotice").classList.add("error");
  } finally { loading = false; }
}
function render() {
  const d = currentStudy, extension = d.kind === "extension";
  $("studyDescription").textContent = extension
    ? "Continue A and B to four million learner decisions each. Keep the settings fixed. Check the games along the way."
    : "Three starting conditions, three seeds each. Scripted opponents helped; this Bow initialization did not.";
  $("studyState").textContent = d.status;
  const active = d.runs.filter((r) => r.status === "running"), completed = d.runs.filter((r) => r.status === "complete");
  const stale = active.filter((r) => r.update_age_seconds > 90);
  $("studyNotice").classList.toggle("error", d.status === "needs attention" || stale.length > 0);
  $("studyNotice").textContent = d.evaluation.error || d.training.error || (stale.length
    ? `No new training metrics for ${stale.map((r) => r.name).join(", ")} in over 90 seconds.`
    : extension ? "Continuation keeps weights, Adam and recent opponents. Matches and recurrent memories restart at the 1M boundary."
    : "Completed comparison. Every final checkpoint was evaluated; none was selected for having the best score.");
  $("studySteps").textContent = million(d.total_learner_steps);
  $("studyBudget").textContent = `${count.format(d.total_learner_steps)} / ${count.format(d.total_target)} learner decisions${extension ? " · includes the first 1M per run" : ""}`;
  $("studyProgress").style.width = `${Math.min(100, 100 * d.total_learner_steps / d.total_target)}%`;
  $("trainingCount").textContent = `${completed.length} / ${d.runs.length}`;
  $("activeTraining").textContent = active.length ? `${active.map((r) => r.name).join(", ")} training · ${d.runs.filter((r) => r.status === "queued").length} queued` : completed.length === d.runs.length ? "All training runs complete" : "Waiting for the next run";
  $("evaluationCount").textContent = count.format(d.evaluation.completed_games + d.evaluation.active_games);
  $("evaluationProgress").textContent = `of ${count.format(d.evaluation.expected_games)} scheduled matches · ${d.evaluation.completed_suites}/${d.evaluation.expected_suites} suites complete`;
  $("runsHeading").textContent = extension ? "Two conditions. Three seeds." : "Three conditions. Three seeds.";
  $("budgetTag").textContent = extension ? "4M learner decisions per run" : "1M learner decisions per run";
  $("conditionNote").textContent = "A: current and historical policies. B: adds scripted opponents." + (extension ? " Same model, reward, learning rate and opponent settings as before." : " C: Bow teacher initialization with the same mixture as B.");
  $("studyRuns").replaceChildren(...d.runs.map((r) => {
    const card = element("article", "", `study-run ${r.status}`);card.style.setProperty("--condition", palette[r.condition]);
    const title = element("div", "", "run-title");title.append(element("b", `${r.condition} · Seed ${r.seed}`), element("span", r.status));
    const bar = element("div", "", "progress"), fill = element("span");fill.style.width = `${Math.min(100, 100 * r.learner_steps / r.target)}%`;bar.append(fill);
    card.append(title, element("div", `${million(r.learner_steps)} / ${million(r.target)}`, "run-budget"), element("small", "learner decisions"), bar);
    card.append(element("small", r.status === "running" ? `${count.format(Math.round(r.sps || 0))} learner decisions/s · last update ${Math.round(r.update_age_seconds || 0)}s ago` : r.status === "queued" ? "Continues from its saved 1M policy" : r.status === "complete" ? "Budget reached · evaluation shown below" : r.failure || "Check the training process"));
    if (r.config_url) { const link = element("a", "Exact configuration ↗");link.href = r.config_url;card.append(link); }
    return card;
  }));
  $("crossplaySection").hidden = !extension;
  if (extension) {
    options($("crossplayMilestone"), Object.entries(d.crossplay).map(([n, value]) => [n, `${Number(n) / 1e6}M · ${value.complete ? "complete" : value.games ? "in progress" : "pending"}`]));
    renderCrossplay();
  }
  const conditions = [...new Set(d.runs.map((r) => r.condition))];
  $("conditionLegend").replaceChildren(...conditions.map((c) => { const span = element("span", c === "A" ? "A · self-play" : c === "B" ? "B · script mixture" : "C · Bow + mixture");span.style.setProperty("--series", palette[c]);return span; }));
  chart("milestoneChart", d.runs.map((r) => ({color:palette[r.condition],points:Object.entries(r.checkpoints).flatMap(([m,e]) => e.development ? [{x:Number(m),y:e.development.win_rate}] : [])})), 1, pct, extension ? 4e6 : 1e6);
  chart("trainingHitsChart", d.runs.map((r) => ({color:palette[r.condition],points:r.curve.map((p) => ({x:p.steps,y:p.hits}))})), null, (n) => count.format(Math.round(n)), extension ? 4e6 : 1e6);
  const milestones = extension ? [1,2,3,4] : [1];
  const header = element("tr");["Run", ...milestones.map((m) => `Dev ${m}M`), "Reserved 1M", ...(extension ? ["Reserved 4M"] : [])].forEach((s) => header.append(element("th", s)));$("milestoneHead").replaceChildren(header);
  $("milestoneRows").replaceChildren(...d.runs.map((r) => {
    const tr = element("tr");tr.append(element("th", r.name));
    [...milestones.map((m) => r.checkpoints[String(m*1e6)].development), r.checkpoints["1000000"].transfer, ...(extension ? [r.checkpoints["4000000"].transfer] : [])].forEach((result) => tr.append(element("td", result ? `${pct(result.win_rate)} · ${result.wins}/${result.games}` : "Pending", result ? "" : "pending")));
    return tr;
  }));
  $("mapNote").textContent = `Training/development maps: ${d.training_maps.join(", ")}. Reserved maps: ${d.transfer_maps.join(", ")}. Draws count as non-wins. Reserve-map opponents differ from development opponents.`;
  options($("detailRun"), d.runs.map((r) => [r.name, r.name]));options($("detailMilestone"), milestones.map((m) => [String(m*1e6), `${m}M`]));renderDetails();
  options($("studyClip"), d.clips.length ? d.clips.map((c) => [c.url, c.title]) : [["", "No clips yet"]]);renderClip();
  $("failureSection").hidden = !d.failure_groups;
  if (d.failure_groups) {
    const b = d.failure_groups.B, scripts = d.failure_groups.script;
    const findings = [
      [`${b.activation_to_fire.median} ticks`, "Bow is released very quickly", `Median activation-to-shot time in B’s selected failures; scripts take ${scripts.activation_to_fire.median} ticks. Median held-Bow input runs last ${b.holds.median} ticks.`],
      [`${b.projectile_speeds.median.toFixed(1)} vs ${scripts.projectile_speeds.median.toFixed(1)}`, "Arrows are much slower", "Median observed projectile speeds, B versus scripts, in world units per second. Brief charging is a concrete weakness to investigate."],
      [`${b.environment_deaths} environment · ${b.self_deaths} self`, "Survival remains unreliable", `${b.opponent_deaths} deaths were credited to the opponent. Credits do not establish the full cause: an uncredited shove can still be environmental.`],
    ];
    $("failureFindings").replaceChildren(...findings.map(([value,title,detail]) => { const a = element("article");a.append(element("strong",value),element("h3",title),element("p",detail));return a; }));
  }
  $("retentionSection").hidden = !d.retention.length;
  $("retentionText").textContent = d.retention.length ? d.retention.map((r) => `${r.run}: ${r.successes}/${r.trials}`).join(" · ") + ". The unrestricted Bow initialization passed 32/32 before training." : "";
  chart("studySpeedChart", d.runs.map((r) => ({color:palette[r.condition],points:r.curve.map((p) => ({x:p.steps,y:p.sps}))})), null, (n) => count.format(Math.round(n)), extension ? 4e6 : 1e6);
  $("healthDetail").textContent = active.length ? active.map((r) => `${r.name}: policy change ${r.kl?.toFixed(4) ?? "—"}, entropy ${r.entropy?.toFixed(2) ?? "—"}, latest data ${Math.round(r.update_age_seconds || 0)}s ago.`).join(" ") : "No training run is currently active. Saved throughput remains visible for comparison.";
  $("studyLinks").replaceChildren(...d.links.map((item) => { const a = element("a", `${item.label} ↗`);a.href=item.url;return a; }));
  $("studyUpdated").textContent = `Read ${new Date(d.updated_at * 1000).toLocaleTimeString()}`;
}
function renderCrossplay() {
  const c = currentStudy.crossplay[$("crossplayMilestone").value];if (!c) return;
  $("crossA").textContent = c.games ? pct(c.A_wins/c.games) : "—";$("crossB").textContent = c.games ? pct(c.B_wins/c.games) : "—";
  $("crossplayBar").replaceChildren(...[[c.A_wins,"#61798d"],[c.draws,"#d5d9c9"],[c.B_wins,"#df872d"]].map(([n,color]) => { const span=element("span");span.style.width=`${c.games ? 100*n/c.games : 0}%`;span.style.background=color;return span; }));
  $("crossplayDetail").textContent = c.games ? `${c.A_wins} A wins · ${c.B_wins} B wins · ${c.draws} draws. ${c.games}/${c.expected_games} matches${c.complete ? " complete." : " complete; partial pairings are not a final result."}` : "This comparison starts when both policies reach the selected checkpoint.";
  $("pairingRows").replaceChildren(...[73,173,273].map((a) => { const tr=element("tr");tr.append(element("th",String(a)));[73,173,273].forEach((b) => { const r=c.cells.find((x) => x.a===a && x.b===b)?.result;const td=element("td",r ? pct(r.losses/r.games) : "—");if(r){td.style.background=`rgba(223,135,45,${.05+.55*r.losses/r.games})`;td.title=`B ${r.losses} wins / A ${r.wins} wins / ${r.draws} draws in ${r.games} games`;}tr.append(td); });return tr; }));
}
function renderDetails() {
  const run=currentStudy.runs.find((r) => r.name===$("detailRun").value);const report=run?.checkpoints[$("detailMilestone").value]?.[$("detailSuite").value];
  $("detailRows").replaceChildren(...(report?.cells || []).map((c) => { const tr=element("tr");[c.map,c.opponent,`${c.wins}/${c.games} (${pct(c.wins/c.games)})`,c.losses,c.draws,c.hits,c.deaths].forEach((v) => tr.append(element("td",String(v))));return tr; }));
  $("detailNote").textContent=report ? "Both player sides are included in each row." : "This checkpoint/suite is pending or is not scheduled. Reserved-map evaluation runs at 1M and 4M.";
}
function renderClip() {
  const clip=currentStudy.clips.find((c) => c.url===$("studyClip").value);const video=$("studyVideo");
  $("videoPending").hidden=Boolean(clip);video.hidden=!clip;$("clipDetail").textContent=clip?.detail || "";
  if (clip && video.dataset.url!==clip.url) { video.pause();video.src=clip.url;video.poster=clip.poster;video.dataset.url=clip.url; }
  if (!clip && video.dataset.url) { video.pause();video.removeAttribute("src");delete video.dataset.url;video.load(); }
}
function chart(id,series,maximum,format,xMaximum) {
  const canvas=$(id),width=canvas.clientWidth,height=canvas.clientHeight;if (!width || !height) return;
  const ratio=devicePixelRatio || 1;canvas.width=width*ratio;canvas.height=height*ratio;
  const ctx=canvas.getContext("2d");ctx.scale(ratio,ratio);const pad={left:48,right:12,top:12,bottom:30};
  const values=series.flatMap((s) => s.points.map((p) => p.y)).filter(Number.isFinite);const max=maximum || Math.max(1,...values)*1.1;
  const x=(n) => pad.left+n/xMaximum*(width-pad.left-pad.right),y=(n) => height-pad.bottom-n/max*(height-pad.top-pad.bottom);
  ctx.font='10px "Trebuchet MS",sans-serif';ctx.textAlign="right";ctx.fillStyle="#787d71";ctx.lineWidth=1;
  for(let i=0;i<=4;i++){const value=max*i/4;ctx.strokeStyle="#d9dccc";ctx.beginPath();ctx.moveTo(pad.left,y(value));ctx.lineTo(width-pad.right,y(value));ctx.stroke();ctx.fillText(format(value),pad.left-8,y(value)+3);}
  ctx.textAlign="center";for(let i=0;i<=4;i++)ctx.fillText(`${(xMaximum*i/4/1e6).toFixed(xMaximum===1e6?2:0)}M`,x(xMaximum*i/4),height-9);
  series.forEach((s,index) => { const points=s.points.filter((p) => Number.isFinite(p.x)&&Number.isFinite(p.y));ctx.strokeStyle=s.color;ctx.fillStyle=s.color;ctx.lineWidth=1.5;ctx.setLineDash(index%3===1?[5,3]:index%3===2?[2,3]:[]);ctx.beginPath();points.forEach((p,i) => i?ctx.lineTo(x(p.x),y(p.y)):ctx.moveTo(x(p.x),y(p.y)));ctx.stroke();ctx.setLineDash([]);if(points.length<10)points.forEach((p) => {ctx.beginPath();ctx.arc(x(p.x),y(p.y),3,0,Math.PI*2);ctx.fill();});});
  if(!values.length){ctx.fillStyle="#787d71";ctx.textAlign="center";ctx.fillText("Waiting for measured results",width/2,height/2);}
}
$("studyPicker").onchange=() => {studyName=$("studyPicker").value;history.replaceState(null,"",`/study?study=${encodeURIComponent(studyName)}`);$("studyVideo").pause();if(!loading)refresh();};
$("crossplayMilestone").onchange=renderCrossplay;
for(const id of ["detailRun","detailMilestone","detailSuite"])$(id).onchange=renderDetails;
$("studyClip").onchange=renderClip;
window.addEventListener("resize",() => {if(currentStudy)render();});
refresh();setInterval(refresh,5000);
