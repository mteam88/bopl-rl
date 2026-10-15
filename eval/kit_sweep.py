"""Kit sweep: rank kits by how exploitable the main is when its opponent holds them.

    venv/bin/python eval/kit_sweep.py \
        [--main f5=PATH] [--probe g111=PATH --probe a20=PATH] [--kits-random 23] [--rows 10] \
        [--workers 4] [--out native/artifacts/rl/sweeps/exploiters-2/kit-sweep]

For each kit K (Matthew's 16 combos, --extra kits, --kits-random uniform kits), the schedule
has --rows rows [K | main kit]: the mirror and random main kits. Every subject plays the same
rows, seeds and maps against the main (eval/exploit_eval.py): the main itself holding K (the
control: how strong K is against the main when the main plays it) and each probe net holding K.

Per kit:
  control   the main's own weights on K vs the main: high means the main loses to K.
  gap[net]  net on K minus the control on K, paired over seed pairs (normal 95%): positive means
            a known policy plays K better against the main than the main itself does, so an
            exploiter warm-started from the main has a hole to find.
  rank      by best_gap (max over probes), then control.
Writes OUT/games.jsonl (exploit_eval), OUT/schedule.txt and kit-weakness.json beside OUT.
Refresh against a newer main by passing --main NAME=PATH (the probes stay fixed nets).
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / 'native/artifacts/rl/sweeps'
sys.path.insert(0, str(ROOT / 'eval'))
from evaluation_protocol import ability_names  # noqa: E402

POOL = [i for i in range(2, 39) if i not in (12, 15)]


def matthew16():
    out = []
    for line in (ROOT / 'native/data/lineups-matthew.txt').read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            out.append(tuple(int(x) for x in line.split(',')))
    return out


def ci(d):
    n = len(d); m = sum(d) / n
    se = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1) / n) if n > 1 else float('inf')
    return m, [m - 1.96 * se, m + 1.96 * se]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--main', default=f'f5={S}/final-5/run/checkpoints/policy-18666684416.bin')
    p.add_argument('--probe', action='append', default=None)
    p.add_argument('--extra', default='26,3,2 2,10,14 6,14,2 29,2,14 18,2,3', help='extra kits (space-separated)')
    p.add_argument('--kits-random', type=int, default=23)
    p.add_argument('--rows', type=int, default=10)
    p.add_argument('--seed', type=int, default=20261015)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out', type=Path, default=S / 'exploiters-2/kit-sweep')
    p.add_argument('--summarize-only', action='store_true')
    a = p.parse_args()
    probes = a.probe or [f'g111={S}/generalist-111-20b/000/checkpoints/policy-26026704896.bin',
                         f'a20={S}/final-1/anneal4b/checkpoints/policy-26666729472.bin']
    mname, mpath = a.main.split('=', 1)
    rng = random.Random(a.seed)
    kits = list(dict.fromkeys(matthew16() + [tuple(int(x) for x in k.split(',')) for k in a.extra.split()]))
    while len(kits) < len(matthew16()) + len(a.extra.split()) + a.kits_random:
        k = tuple(rng.sample(POOL, 3))
        if k not in kits:
            kits.append(k)
    rows = []
    for k in kits:
        rows.append(list(k) + list(k))
        for _ in range(a.rows - 1):
            rows.append(list(k) + rng.sample(POOL, 3))
    a.out.mkdir(parents=True, exist_ok=True)
    sched = a.out / 'schedule.txt'
    sched.write_text(f'# kit sweep (exploiters-2/kit_sweep.py), {len(kits)} kits x {a.rows} rows [kit | {mname} kit]\n'
                     + '\n'.join(','.join(map(str, r)) for r in rows) + '\n')
    if not a.summarize_only:
        cmd = [sys.executable, str(ROOT / 'eval/exploit_eval.py'), '--subject', f'{mname}={mpath}',
               *sum((['--subject', x] for x in probes), []), '--opponent', f'{mname}={mpath}',
               '--schedule', str(sched), '--label', 'kit-sweep', '--pairs', str(len(rows)),
               '--workers', str(a.workers), '--out', str(a.out)]
        subprocess.run(cmd, check=True, cwd=ROOT)
    games = defaultdict(dict)   # (subject, kit) -> {(seed, side): score}
    for line in (a.out / 'games.jsonl').open():
        r = json.loads(line)
        if r['schedule'] != 'kit-sweep' or r['opponent'] != mname:
            continue
        games[(r['subject'], tuple(r['subject_kit']))][(r['seed'], r['side'])] = r['score']
    names = ability_names()
    table = []
    for k in kits:
        c = games.get((mname, k), {})
        if not c:
            continue
        row = dict(kit=list(k), name='/'.join(names[i] for i in k), games=len(c),
                   control=round(sum(c.values()) / len(c), 4), gap={})
        for pr in probes:
            pn = pr.split('=', 1)[0]
            g = games.get((pn, k), {})
            keys = c.keys() & g.keys()
            seeds = defaultdict(list)
            for s in keys:
                seeds[s[0]].append(g[s] - c[s])
            if len(seeds) > 1:
                m, interval = ci([sum(v) / len(v) for v in seeds.values()])
                row['gap'][pn] = dict(score=round(sum(g[s] for s in keys) / len(keys), 4), diff=round(m, 4),
                                      ci=[round(x, 4) for x in interval])
        row['best_gap'] = max((v['diff'] for v in row['gap'].values()), default=None)
        table.append(row)
    table.sort(key=lambda r: (-(r['best_gap'] or 0), -r['control']))
    out = dict(main=a.main, probes=probes, rows_per_kit=a.rows, kits=table)
    (a.out.parent / 'kit-weakness.json').write_text(json.dumps(out, indent=1))
    print(f"{'kit':<34}{'ctl':>6}  " + '  '.join(f'{pr.split("=")[0]:>22}' for pr in probes))
    for r in table:
        print(f"{r['name']:<34}{r['control']:>6.3f}  " + '  '.join(
            f"{v['diff']:+.3f} [{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}]" for v in r['gap'].values()))


if __name__ == '__main__':
    main()
