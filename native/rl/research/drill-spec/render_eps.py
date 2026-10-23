"""Contact sheet of drill episodes from drill_look visual games (system python3 + matplotlib).

    python3 render_eps.py LOOKDIR OUT.png [--max 12] [--prefer ug|kill|long]
Each panel: walls (layers 11/16/21) at the episode's first frame, A's path from 1 s before the cast
(slime: grey), drill in air (orange), drill underground (red, by the undelayed SDF), B's path (blue),
B's death (x), A's death (+). Arrow = A's drill heading at each underground decision.
"""
import argparse, gzip, json, math
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load(d):
    decs = [json.loads(l) for l in open(d / 'decisions.jsonl')]
    eps = []; cur = []
    for r in decs:
        if cur and (r['g'] != cur[-1]['g'] or r['d'] != cur[-1]['d'] + 1):
            eps.append(cur); cur = []
        cur.append(r)
    if cur: eps.append(cur)
    return eps


def main():
    p = argparse.ArgumentParser(); p.add_argument('dir', type=Path); p.add_argument('out', type=Path)
    p.add_argument('--max', type=int, default=12); p.add_argument('--prefer', default='ug')
    a = p.parse_args()
    eps = load(a.dir)
    frames = {}
    for f in a.dir.glob('frames-g*.json.gz'):
        frames[int(f.stem.split('-g')[1].split('.')[0])] = f
    eps = [e for e in eps if e[0]['g'] in frames]
    games = {json.loads(l)['g']: json.loads(l) for l in open(a.dir / 'games.jsonl')}

    def key(e):
        ug = sum(r['ug'] for r in e)
        g = games[e[0]['g']]
        kill = any(x['victim'] == 'B' and x['killer'] == 'A' and x['ability'] == 26 and e[0]['d'] <= x['d'] <= e[-1]['d'] + 1 for x in g['deaths'])
        return (ug, kill, len(e)) if a.prefer == 'ug' else (kill, ug, len(e)) if a.prefer == 'kill' else (len(e), ug, kill)
    eps.sort(key=key, reverse=True)
    eps = eps[:a.max]
    cols = 3; rows = max(1, math.ceil(len(eps) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4.2), squeeze=False)
    cache = {}
    for ax, e in zip(axes.flat, eps):
        g = e[0]['g']
        if g not in cache:
            with gzip.open(frames[g], 'rt') as f: cache[g] = json.load(f)
        V = cache[g]; fr = V['frames']; sa = V['a_seat']; sb = 1 - sa
        d0, d1 = e[0]['d'], e[-1]['d']
        F0 = fr[min(d0, len(fr) - 1)]
        W = []
        for ent in F0['entities']:
            if ent['kind'] == 2 and ent['layer'] in (11, 16, 21):
                W.append((ent['x'], ent['y'], ent['rx'], ent['ry'], ent['ux'], ent['uy'], ent['radius'], ent['layer']))
            elif ent['layer'] == 4:
                ax.axhline(ent['y'] + abs(ent['uy']) + abs(ent['ry']), color='#3498db', lw=1)
        W = np.array(W) if W else np.zeros((0, 8))
        lo, hi = max(0, d0 - 15), min(len(fr) - 1, d1 + 8)
        pre = [(fr[i]['players'][sa]['x'], fr[i]['players'][sa]['y']) for i in range(lo, d0 + 1) if fr[i]['players'][sa]['flags'] & 1]
        if pre: ax.plot(*zip(*pre), color='#aaaaaa', lw=1)
        post = [(fr[i]['players'][sa]['x'], fr[i]['players'][sa]['y']) for i in range(d1, hi + 1) if fr[i]['players'][sa]['flags'] & 1]
        if post: ax.plot(*zip(*post), color='#aaaaaa', lw=1, ls=':')
        for r0, r1 in zip(e, e[1:]):
            ax.plot([r0['x'], r1['x']], [r0['y'], r1['y']], color='#e74c3c' if (r0['ug'] and r1['ug']) else '#f39c12', lw=2.2)
        for r in e:
            if r['ug'] and r['hx'] is not None:
                ax.arrow(r['x'], r['y'], r['hx'] * 1.5, r['hy'] * 1.5, head_width=0.4, color='k', lw=0.5)
        bp = [(fr[i]['players'][sb]['x'], fr[i]['players'][sb]['y']) for i in range(lo, hi + 1) if fr[i]['players'][sb]['flags'] & 1]
        if bp: ax.plot(*zip(*bp), color='#2980b9', lw=1.5)
        ax.plot(e[0]['x'], e[0]['y'], 'o', color='#f39c12', ms=5)
        gm = games[g]
        for x in gm['deaths']:
            if d0 - 1 <= x['d'] <= d1 + 8:
                who = sa if x['victim'] == 'A' else sb
                i = min(x['d'] + 1, len(fr) - 1)
                pl = fr[x['d']]['players'][who]
                ax.plot(pl['x'], pl['y'], 'x' if x['victim'] == 'B' else '+', color='k', ms=12, mew=2.5)
        xs = [r['x'] for r in e] + [p[0] for p in bp] + [p[0] for p in pre]
        ys = [r['y'] for r in e] + [p[1] for p in bp] + [p[1] for p in pre]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        span = max(30, max(xs) - min(xs) + 10, (max(ys) - min(ys) + 10) * 1.4)
        ax.set_xlim(cx - span / 2, cx + span / 2); ax.set_ylim(cy - span / 2.8, cy + span / 2.8); ax.set_aspect('equal')
        if len(W):
            gx, gy = np.meshgrid(np.linspace(cx - span / 2, cx + span / 2, 300), np.linspace(cy - span / 2.8, cy + span / 2.8, 220))
            for lay, col in ((11, '#7f8c6d'), (16, '#c0392b'), (21, '#7f8c6d')):
                Wl = W[W[:, 7] == lay]
                if not len(Wl): continue
                best = np.full(gx.shape, 1e9)
                for wx, wy, rx, ry, ux, uy, rad, _ in Wl:
                    ex, ey = math.hypot(rx, ry), math.hypot(ux, uy)
                    ax_, ay_ = (rx / ex, ry / ex) if ex > 1e-6 else (1.0, 0.0)
                    bx_, by_ = (ux / ey, uy / ey) if ey > 1e-6 else (-ay_, ax_)
                    px, py = gx - wx, gy - wy
                    lx = np.abs(px * ax_ + py * ay_) - ex; ly = np.abs(px * bx_ + py * by_) - ey
                    best = np.minimum(best, np.hypot(np.maximum(lx, 0), np.maximum(ly, 0)) + np.minimum(np.maximum(lx, ly), 0) - rad)
                ax.contourf(gx, gy, best, levels=[-1e9, 0], colors=[col], alpha=0.55, zorder=0)
        ug = sum(r['ug'] for r in e)
        res = [f"{x['victim']} died ({x['killer']},{x['ability']},{x['cause']})" for x in gm['deaths'] if d0 - 1 <= x['d'] <= d1 + 8]
        ax.set_title(f"g{g} d{d0}-{d1} {len(e)*4/60:.2f}s ug {ug*4/60:.2f}s map{gm['map']} B{gm['b_kit']}\n{'; '.join(res)}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in list(axes.flat)[len(eps):]: ax.axis('off')
    fig.suptitle(f"{a.dir.name}: drill episodes (orange air drill, red underground, grey slime, blue opponent, x B dies, + A dies)", fontsize=10)
    fig.tight_layout(); fig.savefig(a.out, dpi=110)
    print('wrote', a.out, len(eps))


main()
