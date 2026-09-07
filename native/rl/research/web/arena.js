"use strict";
(() => {
const colors = ["#f4d93f","#b99be8","#ffca6a","#80c7e0","#d9afee","#ef9e70"];
window.drawBoplArena = function(canvas,frame,initial) {
  const ctx=canvas.getContext("2d"),w=canvas.width,h=canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#18263a";
  ctx.fillRect(0, 0, w, h);
  const visible = initial.entities.filter((e) => !(e.flags & 8));
  const xs = visible.flatMap((e) => [
    e.x - Math.abs(e.rx) - Math.abs(e.ux) - e.radius,
    e.x + Math.abs(e.rx) + Math.abs(e.ux) + e.radius,
  ]);
  const span = Math.max(
    54,
    Math.min(110, Math.max(...xs) - Math.min(...xs) + 12),
  );
  const scale = w / span,
    cx = w / 2,
    cy = h * 0.49;
  const point = (x, y) => [cx + x * scale, cy - y * scale];
  ctx.strokeStyle = "#243348";
  ctx.lineWidth = 1;
  for (let x = -60; x <= 60; x += 5) {
    const [px] = point(x, 0);
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, h);
    ctx.stroke();
  }
  for (let y = -30; y < 30; y += 5) {
    const [, py] = point(0, y);
    ctx.beginPath();
    ctx.moveTo(0, py);
    ctx.lineTo(w, py);
    ctx.stroke();
  }

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
      ctx.beginPath();
      ctx.arc(x, y, Math.max(2, Math.abs(e.radius) * scale), 0, Math.PI * 2);
      ctx.fill();
    } else {
      const [x2, y2] = point(e.x + e.rx, e.y + e.ry);
      ctx.strokeStyle = owned ? colors[(e.owner - 1) % 6] : "#bcd6d8";
      ctx.lineWidth = Math.max(2, e.radius * scale);
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
  }
  for (const p of frame.players) {
    if (!(p.flags & 1)) continue;
    const [x, y] = point(p.x, p.y),
      r = Math.max(8, p.scale * scale * 0.75);
    ctx.fillStyle = colors[(p.id - 1) % 6];
    ctx.strokeStyle = "#101d2b";
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.ellipse(x, y, r, r * 0.86, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#152333";
    for (const dx of [-0.27, 0.29]) {
      ctx.beginPath();
      ctx.arc(x + dx * r, y - r * 0.15, Math.max(2, r * 0.11), 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = "#dce5e5";
    ctx.font = '12px "Trebuchet MS",sans-serif';
    ctx.textAlign = "center";
    ctx.fillText("P" + p.id, x, y - r - 9);
  }
};
})();
