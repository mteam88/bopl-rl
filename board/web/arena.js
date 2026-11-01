"use strict";
(() => {
const colors = ["#f4d93f","#b99be8","#ffca6a","#80c7e0","#d9afee","#ef9e70"];
// Survivable area, from game/Assets/Scripts/Assembly-CSharp/SceneBounds.cs and the
// death tests in DestroyIfOutsideSceneBounds.UpdateSim: the blast zone kills at
// |x| > 131.25 and y > 72.5 on every level; the bottom is WaterHeight (-11.3)
// on grass/snow levels and Camera_YMin (-26) on space levels. The first frame
// carries `level_type: 2` (Unity LevelType.space) on space maps; board/serve.py
// fills it from rollout.json session.level_type for replays that lack it.
const BOUNDS = {
  grass: { x0: -131.25, x1: 131.25, y0: -11.3, y1: 72.5, water: true },
  space: { x0: -131.25, x1: 131.25, y0: -26, y1: 72.5, water: false },
};
const MARGIN = 4; // world units of breathing room around the death rectangle
// Player hurtbox half-extents at scale 1 (rollout.json prefabs.player.hurtbox_extents).
const HURTBOX = [1.33, 0.81];
// The survivable area is 270.5 x 91.8 units; .stage.physics and #matchCanvas use 2:1.

// Auto camera: frame the action (live players and their projectiles) with
// padding, never tighter than FOCUS_MIN, never wider than the survivable
// rectangle, and glide between targets so the view does not jitter.
const FOCUS_PAD = 10;          // world units around the action
const FOCUS_MIN = [80, 40];    // minimum view span (world units): a level plus jump height
const FOCUS_EASE = 0.12;       // per-frame fraction of the remaining move (~0.35 s at 60 Hz)
const FOCUS_SNAP_TICKS = 6;    // a jump in t larger than this is a scrub: snap the camera
// RlObservation collider flags (native/src/rollout/mod.rs E_*).
const E_ENABLED = 2, E_EXPLOSIVE = 1 << 11;
const EXPLOSIVE = "#ff8a3d";

function focusTarget(frame, b, aspect) {
  const xs = [], ys = [];
  for (const p of frame.players) if (p.flags & 1) { xs.push(p.x); ys.push(p.y); }
  // Owned circles away from their owner are projectiles (arrows), and grenades
  // carry no owner; keep both in view.
  for (const e of frame.entities) {
    if (e.kind !== 0 || !(e.owner > 0 || e.flags & E_EXPLOSIVE)) continue;
    if (frame.players.some((p) => p.id === e.owner && Math.hypot(p.x - e.x, p.y - e.y) < 2)) continue;
    xs.push(e.x); ys.push(e.y);
  }
  const full = { cx: (b.x0 + b.x1) / 2, cy: (b.y0 + b.y1) / 2, sx: b.x1 - b.x0 + 2 * MARGIN, sy: b.y1 - b.y0 + 2 * MARGIN };
  if (!xs.length) return full;
  let sx = Math.max(FOCUS_MIN[0], Math.max(...xs) - Math.min(...xs) + 2 * FOCUS_PAD);
  let sy = Math.max(FOCUS_MIN[1], Math.max(...ys) - Math.min(...ys) + 2 * FOCUS_PAD);
  // Match the canvas aspect so the fit is exact on both axes.
  if (sx / sy > aspect) sy = sx / aspect; else sx = sy * aspect;
  if (sx >= full.sx || sy >= full.sy) return full;
  // Keep the view inside the survivable rectangle (plus margin) when possible.
  const cx = Math.min(Math.max((Math.min(...xs) + Math.max(...xs)) / 2, b.x0 - MARGIN + sx / 2), b.x1 + MARGIN - sx / 2);
  const cy = Math.min(Math.max((Math.min(...ys) + Math.max(...ys)) / 2, b.y0 - MARGIN + sy / 2), b.y1 + MARGIN - sy / 2);
  return { cx, cy, sx, sy };
}

function camera(canvas, frame, initial, b, aspect) {
  const target = focusTarget(frame, b, aspect);
  let cam = canvas._boplCamera;
  const snap = !cam || cam.initial !== initial || Math.abs((frame.t ?? 0) - cam.t) > FOCUS_SNAP_TICKS;
  if (snap) cam = { ...target };
  else for (const k of ["cx", "cy", "sx", "sy"]) cam[k] += (target[k] - cam[k]) * FOCUS_EASE;
  cam.initial = initial; cam.t = frame.t ?? 0;
  canvas._boplCamera = cam;
  return cam;
}

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
  // Camera: follows the action, bounded by the survivable rectangle plus a
  // margin (see focusTarget). Geometry never drives the frame, so objects
  // parked off-map (map 2 keeps a level object at x=-100) show where they are.
  const b = initial?.level_type === 2 ? BOUNDS.space : BOUNDS.grass;
  const cam = camera(canvas, frame, initial, b, w / h);
  const scale = Math.min(w / cam.sx, h / cam.sy),
    mx = cam.cx,
    my = cam.cy;
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
  // When the camera is zoomed in, the boundary edges can lie outside the
  // canvas; labels then hug the visible edge instead.
  const edgeL = Math.max(left, 0), edgeR = Math.min(right, w), edgeB = Math.min(bottom, h);
  ctx.font = font(9);
  const lowLabel = `x ${b.x0}`, waterLabel = `${b.water ? "water" : "blast zone"}  y ${b.y0}`;
  const bottomVisible = bottom <= h + 1;
  const labelLeft = left >= -1 && bottomVisible ? left + 4 * ui + ctx.measureText(lowLabel).width + 6 * ui : 0;
  const labelRight = bottomVisible ? edgeR - 4 * ui - ctx.measureText(waterLabel).width - 6 * ui : w;
  ctx.fillStyle = "#6f8398";
  ctx.textAlign = "center"; ctx.textBaseline = "bottom";
  const gridStep = scale * 50 < 70 * ui ? 100 : scale * 10 < 40 * ui ? 50 : 10;
  for (let x = Math.ceil(gx0 / gridStep) * gridStep; x <= gx1; x += gridStep) {
    const [px] = point(x, 0), half = ctx.measureText(String(x)).width / 2;
    if (Math.abs(x) < b.x1 && px - half > labelLeft && px + half < labelRight)
      ctx.fillText(String(x), px, edgeB - 2 * ui);
  }
  ctx.textAlign = "left"; ctx.textBaseline = "middle";
  for (let y = Math.ceil(gy0 / gridStep) * gridStep; y <= gy1; y += gridStep) {
    const [, py] = point(0, y);
    if (y > b.y0 && y < b.y1 && py < edgeB - 16 * ui && py > 8 * ui) ctx.fillText(String(y), edgeL + 4 * ui, py);
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
  ctx.fillText(waterLabel, edgeR - 4 * ui, bottom - 2 * ui);

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
    // Grenades and explosions are explosive; an explosion's hitbox is live for
    // one frame (Explosion.hitBoxActiveOnlyFirstFrame), then only its animation
    // lingers, drawn faint so it does not read as terrain.
    const explosive = e.flags & E_EXPLOSIVE;
    ctx.globalAlpha = explosive && !(e.flags & E_ENABLED) ? 0.25 : 1;
    ctx.fillStyle = explosive
      ? EXPLOSIVE
      : owned
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
  ctx.globalAlpha = 1;
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
