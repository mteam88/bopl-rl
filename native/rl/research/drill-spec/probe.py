"""Linear and MLP probes on the v4 flat block (492 columns, what the policy's flat encoder reads) of
truly-underground drill decisions: is the information a long underground route needs in the row?

    probe.py LOOKDIR [LOOKDIR ...] --out probe.json
Targets (undelayed truth from drill_look): ground left along the heading (u, clipped 20), heading
(cos, sin), steps to surfacing (decisions until the episode's next above-ground decision, clipped 30),
opponent dx, dy (sanity: the row's relation block carries it). Feature sets:
  row      the current flat block
  row+mem  plus the flat block of the episode's last above-ground decision (perfect memory of the rays
           and terrain seen before going under; an upper bound on what recurrence could add)
  row+hd   plus the true heading (cos, sin): the one-feature fix for the stale body block
  hd+sdf   heading, the row's delayed SDF and gradient only
Group 5-fold CV by game; R^2 on held-out folds, mean and range over folds.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

TO = 66 + 44 + 56 + 26          # terrain offset in the flat block (layout.V4_TERRAIN_OFFSET)


def load(dirs):
    X, M, src = [], [], []
    for k, d in enumerate(dirs):
        z = np.load(Path(d) / 'rows.npz')
        X.append(z['flat']); m = z['meta'].copy(); m[:, 0] += k * 100000; M.append(m); src += [Path(d).name] * len(m)
    return np.concatenate(X), np.concatenate(M), np.array(src)


def main():
    p = argparse.ArgumentParser(); p.add_argument('dirs', nargs='+'); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    X, M, src = load(a.dirs)
    g, d, sdf, x, y, ox, oy, hx, hy, ahead = M.T
    order = np.lexsort((d, g)); X, M, src = X[order], M[order], src[order]
    g, d, sdf, x, y, ox, oy, hx, hy, ahead = M.T
    n = len(g)
    # episodes, last above-ground row, steps to surface
    ep = np.zeros(n, int); e = 0
    for i in range(1, n):
        if g[i] != g[i - 1] or d[i] != d[i - 1] + 1: e += 1
        ep[i] = e
    ug = sdf < 0
    mem = np.full(n, -1); last = -1
    for i in range(n):
        if i == 0 or ep[i] != ep[i - 1]: last = -1
        if not ug[i]: last = i
        mem[i] = last
    to_surface = np.full(n, 30.0)
    nxt = None
    for i in range(n - 1, -1, -1):
        if i == n - 1 or ep[i] != ep[i + 1]: nxt = None
        if not ug[i]: nxt = i
        to_surface[i] = min(30.0, (nxt - i) if nxt is not None else 30.0)
    sel = ug & (mem >= 0) & (hx ** 2 + hy ** 2 > 0.5)
    print('underground rows', int(ug.sum()), 'with memory and heading', int(sel.sum()), 'episodes', len(np.unique(ep[sel])))
    idx = np.where(sel)[0]
    R = X[idx]; Rm = X[mem[idx]]
    H = np.stack([hx[idx], hy[idx]], 1)
    feats = {
        'row': R,
        'row+mem': np.concatenate([R, Rm], 1),
        'row+hd': np.concatenate([R, H], 1),
        'hd+sdf': np.concatenate([H, R[:, TO + 9:TO + 12]], 1),
    }
    targets = {
        'ground_ahead': np.clip(np.where(ahead[idx] < 0, 20, ahead[idx]), 0, 20),
        'heading_cos': hx[idx], 'heading_sin': hy[idx],
        'steps_to_surface': to_surface[idx],
        'opp_dx': ox[idx] - x[idx], 'opp_dy': oy[idx] - y[idx],
    }
    groups = g[idx]
    cv = GroupKFold(n_splits=5)
    out = dict(rows=int(len(idx)), episodes=int(len(np.unique(ep[idx]))), sources={s: int((src[idx] == s).sum()) for s in np.unique(src[idx])},
               target_sd={k: float(np.std(v)) for k, v in targets.items()}, results={})
    for tn, yv in targets.items():
        for fn, F in feats.items():
            for mn in ('ridge', 'mlp'):
                scores = []
                for tr, te in cv.split(F, yv, groups):
                    if mn == 'ridge':
                        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 13)))
                    else:
                        model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(128,), alpha=1e-2, max_iter=400,
                                                                             early_stopping=True, random_state=0))
                    model.fit(F[tr], yv[tr]); scores.append(r2_score(yv[te], model.predict(F[te])))
                out['results'][f'{tn}|{fn}|{mn}'] = [float(np.mean(scores)), float(np.min(scores)), float(np.max(scores))]
                print(f'{tn:18s} {fn:8s} {mn:5s} R2 {np.mean(scores):+.3f} [{np.min(scores):+.3f}, {np.max(scores):+.3f}]', flush=True)
    a.out.write_text(json.dumps(out, indent=1))


main()
