"""The ground left along the heading, read straight off the row: on truly-underground drill decisions of
drill_look runs (rows.npz), the around ray nearest the true heading against the ground ahead (drill_look's
`ahead`). The row is written from the 9-tick-delayed state, so the ray at decision d is compared with the
truth at decision d - LAG (LAG 2 is the delay); the heading and position are the truth's at d - LAG.

    ray_heading.py LOOKDIR [LOOKDIR ...] [--lags 0,2]
Prints per run and lag: rows, correlation, |error| median, share within 1 u, share of the ray reading
more than 1 u short (something nearer than the exit), and the ray's class there (the one-hot argmax).
"""
import argparse
import sys
from pathlib import Path
import numpy as np

T = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(T / 'eval'))
import layout as L  # noqa

CLASSES = ['wall', 'lethal', 'plane', 'object', 'opponent', 'own']


def main():
    p = argparse.ArgumentParser(); p.add_argument('dirs', nargs='+'); p.add_argument('--lags', default='0,2')
    a = p.parse_args()
    for d in a.dirs:
        z = np.load(Path(d) / 'rows.npz'); X, M = z['flat'], z['meta']
        o = np.lexsort((M[:, 1], M[:, 0])); X, M = X[o], M[o]
        g, dec, sdf, x, y, ox, oy, hx, hy, ahead = M.T
        R = X[:, L.V4_RAY_OFFSET:L.V4_RAY_OFFSET + L.V4_RAYS * L.V4_RAY_FEATURES].reshape(-1, L.V4_RAYS, L.V4_RAY_FEATURES)
        for lag in (int(v) for v in a.lags.split(',')):
            i = np.arange(lag, len(M)); j = i - lag
            ok = (g[i] == g[j]) & (dec[i] == dec[j] + lag) & (sdf[j] < 0) & (hx[j] ** 2 + hy[j] ** 2 > 0.5) & (ahead[j] >= 0)
            i, j = i[ok], j[ok]
            k = np.rint((np.arctan2(hy[j], hx[j]) % (2 * np.pi)) / (2 * np.pi / L.V4_RAYS_AROUND)).astype(int) % L.V4_RAYS_AROUND
            ray = R[i, k]; got = ray[:, 0] * 40; want = ahead[j]; e = got - want
            short = e < -1
            cls = np.bincount(np.argmax(ray[short, 1:7], 1), minlength=6) if short.any() else np.zeros(6, int)
            print(f'{Path(d).name:10s} lag {lag} rows {len(e):5d} corr {np.corrcoef(got, want)[0, 1]:+.3f} '
                  f'|err| median {np.median(np.abs(e)):.2f} u, within 1 u {np.mean(np.abs(e) < 1):.3f}, '
                  f'short > 1 u {short.mean():.3f} (' + ', '.join(f'{c} {n}' for c, n in zip(CLASSES, cls) if n) + ')')


main()
