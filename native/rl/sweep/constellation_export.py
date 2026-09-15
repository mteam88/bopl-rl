#!/usr/bin/env python3
"""Export sweeps to PufferLib's Constellation viewer.

Constellation (dependencies/PufferLib-*/constellation/) is Puffer's raylib experiment
browser: a 3D scatter over any three columns, a 2D projection, a t-SNE map and a
per-column range plot, with range filters and right-click-to-copy hypers. Its axis
list is hardcoded to PufferLib's train hypers, so this writes a copy of
constellation.c with our columns patched in, the data as
resources/constellation/experiments.json (one point per band-scored checkpoint of
every complete trial), the resources it loads, and a build script that skips the
x86-only SIMD flags so it compiles on Apple Silicon.

    python native/rl/sweep/constellation_export.py --out /tmp/constellation \
        native/artifacts/rl/sweeps/overnight-7 native/artifacts/rl/sweeps/overnight-9
    rsync -a /tmp/constellation/ mac:constellation/ && ssh mac 'cd constellation && bash build.sh && ./seethestars'

Each sweep is one Constellation "env" (its own dropdown entry). Scores are on the
sweep's own pool scale (overnight-7: pool v5, overnight-9: pool v7), so compare within
an env, not across. Columns:
    agent_steps          learner steps of the checkpoint, in millions
    uptime               training seconds at the checkpoint
    env/score            the checkpoint's band Elo on the sweep's pool
    env/perf             env/score min-max scaled to [0, 1] within the sweep
    meta/trial           trial index; meta/gp_phase 1 if Protein's GP suggested it (0 = Sobol/anchor)
    gp/predicted_score   the GP's predicted score at suggestion time (0 when unavailable)
    hypers/*             the trial's hypers; steps and entropy_half_life in millions
    tsne1, tsne2         2D embedding of the normalised hypers (t-SNE if sklearn is
                         installed, else the two leading principal components)
"""
import argparse
import glob
import json
import math
import re
import shutil
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEPS = ROOT / 'native/artifacts/rl/dependencies'

HYPERS = ['steps', 'learning_rate', 'entropy', 'entropy_half_life', 'reward_kill',
          'scripted_banks', 'scripted_share_max', 'scripted_share_min', 'head_scale_aim']
IN_MILLIONS = {'steps', 'entropy_half_life'}
COLUMNS = (['agent_steps', 'uptime', 'env/perf', 'env/score', 'meta/trial', 'meta/gp_phase',
            'gp/predicted_score'] + [f'hypers/{h}' + ('_M' if h in IN_MILLIONS else '') for h in HYPERS])


def pufferlib_dir():
    dirs = sorted(DEPS.glob('PufferLib-*'))
    if not dirs:
        raise SystemExit(f'no PufferLib checkout under {DEPS}')
    return dirs[-1]


def load_sweep(sweep_dir):
    sweep_dir = Path(sweep_dir)
    config = json.load(open(sweep_dir / 'sweep.json'))['config']
    fixed, params = config.get('fixed', {}), config.get('params', {})
    trials = []
    for line in open(sweep_dir / 'results.jsonl'):
        r = json.loads(line)
        if r.get('status') != 'complete' or not r.get('observations'):
            continue
        hypers = {h: r['hypers'].get(h, fixed.get(h)) for h in HYPERS}
        if any(v is None for v in hypers.values()):
            missing = [h for h, v in hypers.items() if v is None]
            raise SystemExit(f'{sweep_dir.name} trial {r["id"]}: no value for {missing}')
        trials.append(dict(index=r['index'], hypers=hypers, observations=r['observations'],
                           gp=bool(r.get('suggested')) and 'score' in (r.get('suggestion_info') or {}),
                           predicted=(r.get('suggestion_info') or {}).get('score') or 0.0))
    return trials, params


def normalised_hypers(trials, params):
    """Each trial's hypers scaled to [0, 1] over the sweep's boxes (log where the box is log)."""
    rows = []
    for t in trials:
        row = []
        for h in HYPERS:
            v = float(t['hypers'][h])
            p = params.get(h)
            if p is None:
                row.append(0.0)
                continue
            lo, hi = float(p['min']), float(p['max'])
            if 'log' in p.get('distribution', ''):
                v, lo, hi = math.log(v), math.log(lo), math.log(hi)
            row.append((v - lo) / (hi - lo) if hi > lo else 0.0)
        rows.append(row)
    return rows


def embed(rows):
    """2D embedding of the normalised hyper rows: t-SNE when sklearn is available, else PCA."""
    import numpy as np
    x = np.array(rows, dtype=float)
    if len(x) < 3:
        return [[0.0, 0.0] for _ in rows]
    try:
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=min(30, len(x) - 1), random_state=0).fit_transform(x).tolist()
    except Exception:
        x = x - x.mean(axis=0)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        return (x @ vt[:2].T).tolist()


def export_env(trials, params):
    columns = {c: [] for c in COLUMNS + ['tsne1', 'tsne2']}
    scores = [o['score'] for t in trials for o in t['observations']]
    lo, hi = min(scores), max(scores)
    embedding = embed(normalised_hypers(trials, params))
    for t, (e1, e2) in zip(trials, embedding):
        for o in t['observations']:
            columns['agent_steps'].append(o['steps'] / 1e6)
            columns['uptime'].append(o.get('elapsed', 0.0))
            columns['env/score'].append(o['score'])
            columns['env/perf'].append((o['score'] - lo) / (hi - lo) if hi > lo else 0.0)
            columns['meta/trial'].append(t['index'])
            columns['meta/gp_phase'].append(1 if t['gp'] else 0)
            columns['gp/predicted_score'].append(t['predicted'])
            for h in HYPERS:
                v = float(t['hypers'][h])
                columns[f'hypers/{h}' + ('_M' if h in IN_MILLIONS else '')].append(v / 1e6 if h in IN_MILLIONS else v)
            columns['tsne1'].append(e1)
            columns['tsne2'].append(e2)
    # Constellation reads every column as one comma-separated string of floats.
    return {k: ','.join(f'{x:.6g}' for x in v) for k, v in columns.items()}


def patch_source(src):
    """Replace Constellation's hardcoded axis list with our columns."""
    block = re.compile(r'int hyper_count = \d+;\nchar \*hyper_key\[\d+\] = \{.*?\};', re.S)
    keys = ''.join(f'    "{c}",\n' for c in COLUMNS)
    new = f'int hyper_count = {len(COLUMNS)};\nchar *hyper_key[{len(COLUMNS)}] = {{\n{keys}}};'
    patched, n = block.subn(new, src, count=1)
    if n != 1:
        raise SystemExit('constellation.c: hyper_key block not found; the vendored source changed')
    return patched.replace('"Puffer Constellation"', '"Puffer Constellation: bopl sweeps"')


BUILD_SH = r'''#!/bin/bash
# Build Constellation for these sweeps. Linux: needs clang, libGL; macOS: Xcode CLT.
# Derived from PufferLib's build.sh constellation --fast, minus the x86-only -mavx2/-mfma.
set -e
cd "$(dirname "$0")"
if [ "$(uname -s)" = Linux ]; then
    RAYLIB=raylib-5.5_linux_amd64; LD=(-lGL -lm -lpthread -ldl -lrt -lX11)
else
    RAYLIB=raylib-5.5_macos; LD=(-framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL -lm -lpthread)
fi
if [ ! -d "$RAYLIB" ]; then
    curl -sL "https://github.com/raysan5/raylib/releases/download/5.5/$RAYLIB.tar.gz" | tar xz
fi
${CC:-clang} -O2 -DNDEBUG -DPLATFORM_DESKTOP -Wno-everything -I"$RAYLIB/include" -I vendor \
    constellation.c vendor/cJSON.c -o seethestars "$RAYLIB/lib/libraylib.a" "${LD[@]}"
echo "built ./seethestars ($(python3 -c "import json;d=json.load(open('resources/constellation/experiments.json'));print(', '.join(f\"{k}: {v['env/score'].count(',')+1} points\" for k,v in d.items()))"))"
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('sweeps', nargs='+', help='sweep directories (each with sweep.json and results.jsonl)')
    ap.add_argument('--out', type=Path, required=True, help='output directory (created)')
    args = ap.parse_args()

    data = {}
    for sweep in args.sweeps:
        trials, params = load_sweep(sweep)
        if not trials:
            print(f'{sweep}: no complete trials, skipped')
            continue
        data[Path(sweep).name] = export_env(trials, params)
        scores = [o['score'] for t in trials for o in t['observations']]
        print(f'{Path(sweep).name}: {len(trials)} trials, {len(scores)} points, '
              f'score {min(scores):.0f}..{max(scores):.0f}, median {statistics.median(scores):.0f}')
    if not data:
        raise SystemExit('nothing to export')

    puffer = pufferlib_dir()
    out = args.out
    (out / 'resources/constellation').mkdir(parents=True, exist_ok=True)
    (out / 'resources/shared').mkdir(parents=True, exist_ok=True)
    (out / 'vendor').mkdir(exist_ok=True)
    (out / 'constellation.c').write_text(patch_source((puffer / 'constellation/constellation.c').read_text()))
    for f in glob.glob(str(puffer / 'resources/constellation/*')):
        if not f.endswith('experiments.json'):
            shutil.copy(f, out / 'resources/constellation')
    for name in ('puffers.png', 'JetBrainsMono-SemiBold.ttf'):
        shutil.copy(puffer / 'resources/shared' / name, out / 'resources/shared')
    for f in glob.glob(str(puffer / 'vendor/*.[ch]')) + glob.glob(str(puffer / 'vendor/KHR')):
        shutil.copy(f, out / 'vendor') if Path(f).is_file() else shutil.copytree(f, out / 'vendor/KHR', dirs_exist_ok=True)
    json.dump(data, open(out / 'resources/constellation/experiments.json', 'w'))
    (out / 'build.sh').write_text(BUILD_SH)
    print(f'wrote {out}: bash build.sh && ./seethestars')


if __name__ == '__main__':
    main()
