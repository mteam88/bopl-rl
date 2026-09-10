"use strict";
(() => {
const colors = ["#f4d93f","#b99be8","#ffca6a","#80c7e0","#d9afee","#ef9e70"];
// Survivable area, from Assets/Scripts/Assembly-CSharp/SceneBounds.cs and the
// death tests in DestroyIfOutsideSceneBounds.UpdateSim: the blast zone kills at
// |x| > 131.25 and y > 72.5 on every level; the bottom is WaterHeight (-11.3)
// on grass/snow levels and Camera_YMin (-26) on space levels. Every map in
// native/data/rollout.json is level_type 0 (grass), so the space bounds only
// apply when a replay's first frame says `level_type: 2` (Unity LevelType.space).
const BOUNDS = {
  grass: { x0: -131.25, x1: 131.25, y0: -11.3, y1: 72.5, water: true },
  space: { x0: -131.25, x1: 131.25, y0: -26, y1: 72.5, water: false },
};
const MARGIN = 4; // world units of breathing room around the death rectangle
// Player hurtbox half-extents at scale 1 (rollout.json prefabs.player.hurtbox_extents).
const HURTBOX = [1.33, 0.81];
// The framed area is 270.5 x 91.8 units, so .stage.physics and #matchCanvas use 3:1.

window.drawBoplArena = function(canvas,frame,initial) {
  // Render at device resolution: the whole level is small at this zoom, so
  // sub-pixel geometry matters more than it did when the camera hugged it.
  const rect = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  if (rect.width > 0 && rect.height > 0) {
    const W = Math.round(rect.width * dpr), H = Math.round(rect.height * dpr);
    if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  }
  const ctx=canvas.getContext("2d"),w=canvas.width,h=canvas.height;
  const ui = rect.width > 0 ? w / rect.width : 1; // CSS px -> canvas px
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0e1723";
  ctx.fillRect(0, 0, w, h);
  // Camera: the whole survivable rectangle plus a margin, fixed for the
  // replay. Geometry never drives the frame, so objects parked off-map (map 2
  // keeps a level object at x=-100) simply show where they are.
  const b = initial?.level_type === 2 ? BOUNDS.space : BOUNDS.grass;
  const spanX = b.x1 - b.x0 + 2 * MARGIN, spanY = b.y1 - b.y0 + 2 * MARGIN;
  const scale = Math.min(w / spanX, h / spanY),
    mx = (b.x0 + b.x1) / 2,
    my = (b.y0 + b.y1) / 2;
  const point = (x, y) => [w / 2 + (x - mx) * scale, h / 2 - (y - my) * scale];
  const [left, top] = point(b.x0, b.y1), [right, bottom] = point(b.x1, b.y0);
  ctx.fillStyle = "#18263a";
  ctx.fillRect(left, top, right - left, bottom - top);
  if (b.water) {
    ctx.fillStyle = "#173a52";
    ctx.fillRect(0, bottom, w, h - bottom);
  }

  // Grid: 10-unit minor lines, 50-unit major lines labelled at the edges.
  const font = (px) => `${Math.round(px * ui)}px "Trebuchet MS",sans-serif`;
  const gx0 = Math.floor((mx - w / scale / 2) / 10) * 10, gx1 = mx + w / scale / 2;
  const gy0 = Math.floor((my - h / scale / 2) / 10) * 10, gy1 = my + h / scale / 2;
  for (const major of [false, true]) {
    ctx.strokeStyle = major ? "#31445c" : "#1f2d40";
    ctx.lineWidth = Math.max(1, (major ? 1 : 0.6) * ui);
    for (let x = gx0; x <= gx1; x += 10) {
      if ((x % 50 === 0) !== major) continue;
      const [px] = point(x, 0);
      ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.stroke();
    }
    for (let y = gy0; y <= gy1; y += 10) {
      if ((y % 50 === 0) !== major) continue;
      const [, py] = point(0, y);
      ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(w, py); ctx.stroke();
    }
  }
  // Grid labels share the bottom edge with the boundary labels (below); on
  // narrow screens skip the ones that would collide with them.
  ctx.font = font(9);
  const lowLabel = `x ${b.x0}`, waterLabel = `${b.water ? "water" : "blast zone"}  y ${b.y0}`;
  const labelLeft = left + 4 * ui + ctx.measureText(lowLabel).width + 6 * ui;
  const labelRight = right - 4 * ui - ctx.measureText(waterLabel).width - 6 * ui;
  ctx.fillStyle = "#6f8398";
  ctx.textAlign = "center"; ctx.textBaseline = "bottom";
  for (let x = Math.ceil(gx0 / 50) * 50; x <= gx1; x += 50) {
    const [px] = point(x, 0), half = ctx.measureText(String(x)).width / 2;
    if (Math.abs(x) < b.x1 && px - half > labelLeft && px + half < labelRight)
      ctx.fillText(String(x), px, bottom - 2 * ui);
  }
  ctx.textAlign = "left"; ctx.textBaseline = "middle";
  for (let y = Math.ceil(gy0 / 50) * 50; y <= gy1; y += 50) {
    const [, py] = point(0, y);
    if (y > b.y0 && y < b.y1 && py < bottom - 16 * ui) ctx.fillText(String(y), left + 4 * ui, py);
  }

  // Death boundary: dashed blast-zone edges (sides and top), solid water line.
  ctx.strokeStyle = "#e0655a";
  ctx.lineWidth = Math.max(1, 1.2 * ui);
  ctx.setLineDash([6 * ui, 4 * ui]);
  ctx.beginPath();
  ctx.moveTo(left, bottom); ctx.lineTo(left, top); ctx.lineTo(right, top); ctx.lineTo(right, bottom);
  if (!b.water) ctx.lineTo(left, bottom);
  ctx.stroke();
  ctx.setLineDash([]);
  if (b.water) {
    ctx.strokeStyle = "#5fb8d9";
    ctx.lineWidth = Math.max(1, 1.6 * ui);
    ctx.beginPath(); ctx.moveTo(0, bottom); ctx.lineTo(w, bottom); ctx.stroke();
  }
  // Boundary values sit in the corners; the top-left is left free for the
  // index page's PHYSICS REPLAY caption.
  ctx.font = font(9);
  ctx.fillStyle = "#e0655a";
  ctx.textAlign = "right"; ctx.textBaseline = "top";
  ctx.fillText(`blast zone  y ${b.y1}`, right - 4 * ui, top + 3 * ui);
  ctx.fillText(`x ${b.x1}`, right - 4 * ui, top + 14 * ui);
  ctx.textAlign = "left"; ctx.textBaseline = "bottom";
  ctx.fillText(lowLabel, left + 4 * ui, bottom - 2 * ui);
  ctx.fillStyle = b.water ? "#5fb8d9" : "#e0655a";
  ctx.textAlign = "right";
  ctx.fillText(waterLabel, right - 4 * ui, bottom - 2 * ui);

  for (const e of frame.entities) {
    const [x, y] = point(e.x, e.y);
    const owned = e.owner > 0;
    if (
      e.layer === 8 &&
      owned &&
      frame.players.some(
        (p) => p.id === e.owner && Math.hypot(p.x - e.x, p.y - e.y) < 2,
      )
    )
      continue;
    ctx.fillStyle = owned
      ? colors[(e.owner - 1) % 6]
      : e.layer === 4
        ? "#274c64"
        : "#8b9e87";
    ctx.strokeStyle = "#c4d4b6";
    if (e.kind === 1 || e.kind === 2) {
      const corners = [
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
      ].map(([a, b]) =>
        point(e.x + a * e.rx + b * e.ux, e.y + a * e.ry + b * e.uy),
      );
      ctx.beginPath();
      corners.forEach(([px, py], i) =>
        i ? ctx.lineTo(px, py) : ctx.moveTo(px, py),
      );
      ctx.closePath();
      ctx.lineJoin = "round";
      // Rounded-rectangle platforms: the corner radius is the half thickness.
      ctx.lineWidth = Math.max(1, e.radius * scale * 2);
      ctx.strokeStyle = ctx.fillStyle;
      ctx.stroke();
      ctx.fill();
    } else if (e.kind === 0) {
      const isBody = frame.players.some(
        (p) =>
          p.flags & 1 &&
          p.id === e.owner &&
          Math.hypot(p.x - e.x, p.y - e.y) < 2,
      );
      if (isBody) continue;
      // True radius; a hairline outline keeps tiny objects (arrows) visible.
      const r = Math.abs(e.radius) * scale;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
      if (r < 2 * ui) {
        ctx.strokeStyle = ctx.fillStyle;
        ctx.lineWidth = Math.max(1, 0.8 * ui);
        ctx.stroke();
      }
    } else {
      const [x2, y2] = point(e.x + e.rx, e.y + e.ry);
      ctx.strokeStyle = owned ? colors[(e.owner - 1) % 6] : "#bcd6d8";
      ctx.lineWidth = Math.max(1, e.radius * scale);
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
  }
  for (const p of frame.players) {
    if (!(p.flags & 1)) continue;
    const [x, y] = point(p.x, p.y),
      rx = HURTBOX[0] * p.scale * scale,
      ry = HURTBOX[1] * p.scale * scale;
    ctx.fillStyle = colors[(p.id - 1) % 6];
    ctx.strokeStyle = "#101d2b";
    ctx.lineWidth = Math.max(1, 1 * ui);
    ctx.beginPath();
    ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    if (rx > 6 * ui) {
      ctx.fillStyle = "#152333";
      for (const dx of [-0.27, 0.29]) {
        ctx.beginPath();
        ctx.arc(x + dx * rx, y - ry * 0.2, rx * 0.11, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.fillStyle = "#dce5e5";
    ctx.font = font(10);
    ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.fillText("P" + p.id, x, y - ry - 3 * ui);
  }
};
})();
