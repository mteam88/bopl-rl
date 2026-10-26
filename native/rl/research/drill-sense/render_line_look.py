"""Draw line_look.c's rows: the walls, the body, the enemy, the 32 samples (ground red, air blue), the cast's end,
and recompute the three fields from the walls in float64 (a second implementation of bopl4f_line: the
rounded-rect SDF, the 32 samples and an uncapped march).

    python3 render_line_look.py <rows.jsonl> <out.png> [--table out.md]
"""
import argparse
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def walls(row):
    """Every wall as (cx, cy, rx, ry, ux, uy, ex, ey, radius)."""
    out = [(cx, cy, 1.0, 0.0, 0.0, 1.0, ex, ey, r) for cx, cy, ex, ey, r in row['axis']]
    return out + [tuple(w) for w in row['rot']]


def sdf(ws, x, y):
    best = 1e9
    for cx, cy, rx, ry, ux, uy, ex, ey, r in ws:
        px, py = x - cx, y - cy
        qx, qy = abs(px * rx + py * ry) - ex, abs(px * ux + py * uy) - ey
        d = math.hypot(max(qx, 0.0), max(qy, 0.0)) + min(max(qx, qy), 0.0) - r
        best = min(best, d)
    return best


def recompute(row):
    """The three fields in float64, from the walls alone."""
    x, y, ex, ey = row['x'], row['y'], row['ex'], row['ey']
    length = math.hypot(ex - x, ey - y)
    # Written only by the fan (BOPL4F_FAN 1) under BOPL4F_LINE (128), for a known enemy (BOPL4F_ENEMY 32).
    if row['flags'] & (1 | 32 | 128) != (1 | 32 | 128) or length <= 1e-9:
        return [0.0, 0.0, 0.0], 0.0, []
    dx, dy = (ex - x) / length, (ey - y) / length
    ws = walls(row)
    inside_mode = (row['flags'] & 64) and sdf(ws, x, y) < 0   # BOPL4F_INSIDE
    # bot_raycast's rule (a hit at SDF < 0.05; out of the walls at the first SDF >= 0 by steps of max(-SDF, 0.05)),
    # uncapped: steps counts how many the march took, the writer takes 32 SDF samples and returns 40 if none hits.
    t, steps = (0.0, 0) if ws else (40.0, 0)
    while ws and t < 40.0 and steps < 100000:
        d = sdf(ws, x + dx * t, y + dy * t)
        if inside_mode:
            if d >= 0:
                break
            t += max(-d, 0.05)
        else:
            if d < 0.05:
                break
            t += d
        steps += 1
    t = min(t, 40.0)
    row['_steps'] = steps
    ground = [sdf(ws, x + dx * length * (k + 0.5) / 32, y + dy * length * (k + 0.5) / 32) < 0 for k in range(32)]
    last = max([k for k in range(32) if ground[k]], default=-1)
    return [min(t, length) / 40.0, sum(ground) / 32.0, (31 - last) / 32.0], length, ground


def rounded_rect(ax, w, **kw):
    cx, cy, rx, ry, ux, uy, ex, ey, r = w
    pts = []
    for sx, sy, a0 in ((1, 1, 0), (-1, 1, 90), (-1, -1, 180), (1, -1, 270)):
        for a in np.linspace(a0, a0 + 90, 8):
            lx = sx * ex + r * math.cos(math.radians(a))
            ly = sy * ey + r * math.sin(math.radians(a))
            pts.append((cx + lx * rx + ly * ux, cy + lx * ry + ly * uy))
    ax.add_patch(plt.Polygon(pts, closed=True, **kw))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('rows')
    p.add_argument('out')
    p.add_argument('--table')
    p.add_argument('--check', action='store_true', help='recompute every row and report differences, no picture')
    a = p.parse_args()
    rows = [json.loads(line) for line in open(a.rows) if '"case"' in line]
    if a.check:
        bad, capped, worst = 0, 0, [0.0, 0.0, 0.0]
        for row in rows:
            want, _, _ = recompute(row)
            err = [abs(g - w) for g, w in zip(row['line'][:3], want)]
            cap = row['_steps'] >= 32 if '_steps' in row else False
            worst = [max(u, v) for u, v in zip(worst, err if not cap else [0.0] + err[1:])]
            capped += cap
            ok = err[0] <= 0.1 / 40 and err[1] <= 1.01 / 32 and err[2] <= 1.01 / 32
            bad += not (ok or (cap and err[1] <= 1.01 / 32))
            if not ok and not cap:
                print('DIFF', row['where'], row['line'][:3], want)
        exact = sum(row['line'][1:3] == recompute(row)[0][1:3] for row in rows)
        print(f'{len(rows)} rows: {bad} differ from the float64 recompute beyond tolerance, {capped} with a march over '
              f'32 steps, [1] and [2] exactly equal on {exact}; worst |diff| [0] (uncapped) {worst[0]:.2e} [1] {worst[1]:.4f} [2] {worst[2]:.4f}')
        raise SystemExit(1 if bad else 0)
    cols = 3
    n = len(rows)
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(6.2 * cols, 4.6 * ((n + cols - 1) // cols)), squeeze=False)
    table = ['| case | where | L (u) | body SDF | written [0] [1] [2] | float64 recompute | hand-computed |',
             '|---|---|---|---|---|---|---|']
    bad = 0
    for ax, row in zip(axes.flat, rows):
        got = row['line'][:3]
        want, length, ground = recompute(row)
        x, y, ex, ey = row['x'], row['y'], row['ex'], row['ey']
        for w in walls(row):
            rounded_rect(ax, w, facecolor='#cfc6b8', edgecolor='#8a7f70', lw=0.8)
        ax.plot([x, ex], [y, ey], color='#555', lw=0.8, zorder=2)
        if length > 0:
            dx, dy = (ex - x) / length, (ey - y) / length
            for k, g in enumerate(ground):
                t = length * (k + 0.5) / 32
                ax.plot(x + dx * t, y + dy * t, 'o', ms=3.2, color='#c0392b' if g else '#2e86de', zorder=3)
            t = got[0] * 40.0
            ax.plot(x + dx * t, y + dy * t, 'x', ms=9, mew=2, color='#111', zorder=4)
        ax.add_patch(plt.Circle((x, y), 0.76, color='#27ae60', zorder=5))
        ax.add_patch(plt.Circle((ex, ey), 0.76, color='#8e44ad', zorder=5))
        pad = 6.0
        lo_x, hi_x = min(x, ex) - pad, max(x, ex) + pad
        lo_y, hi_y = min(y, ey) - pad, max(y, ey) + pad
        span = max(hi_x - lo_x, (hi_y - lo_y) * 1.35)
        mx, my = (lo_x + hi_x) / 2, (lo_y + hi_y) / 2
        ax.set_xlim(mx - span / 2, mx + span / 2)
        ax.set_ylim(my - span / 2.7, my + span / 2.7)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=7)
        ok = abs(got[0] - want[0]) <= 0.1 / 40 and abs(got[1] - want[1]) <= 1.01 / 32 and abs(got[2] - want[2]) <= 1.01 / 32
        capped = row.get('_steps', 0) >= 32
        ok = ok or (capped and abs(got[1] - want[1]) <= 1.01 / 32)
        ax.set_title(f"{row['case']}\n[0] {got[0]:.3f}  [1] {got[1]:.3f}  [2] {got[2]:.3f}   (f64: {want[0]:.3f} {want[1]:.3f} {want[2]:.3f})"
                     f"{'' if ok else '  DIFFERS'}{'  (march over 32 steps)' if capped else ''}", fontsize=8.5)
        where = row['where'].replace('corpus-', '')
        ax.text(0.01, 0.01, where, transform=ax.transAxes, fontsize=6.5, color='#666')
        hand = ' '.join(f'{v:.4f}' for v in row['want']) if 'want' in row else ''
        table.append(f"| {row['case']} | {where} | {length:.2f} | {row['sdf']:.2f} | {got[0]:.4f} {got[1]:.4f} {got[2]:.4f} | "
                     f"{want[0]:.4f} {want[1]:.4f} {want[2]:.4f} | {hand} |")
        bad += not ok
    for ax in list(axes.flat)[n:]:
        ax.axis('off')
    fig.suptitle('Layout v4.4, the line to the opponent (writer bit 64): body green, opponent purple, samples red = ground, '
                 'blue = air, x = the cast\'s end ([0] x 40 u)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(a.out, dpi=110)
    if a.table:
        open(a.table, 'w').write('\n'.join(table) + '\n')
    print('\n'.join(table))
    print(f'{len(rows)} rows, {bad} differ from the float64 recompute')


if __name__ == '__main__':
    main()
