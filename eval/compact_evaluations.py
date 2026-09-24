#!/usr/bin/env python3
"""Archive finished runs' evaluation replays in place: `game.json` -> `game.json.gz`.

The physics-state visual JSONs are ~70% of a run directory (1.6 GB of a 1B-step
trial's 2 GB) and gzip ~12x. Every reader goes through research/visuals.py, which
resolves either spelling, so reports keep referring to `game.json` unchanged.

Only runs whose status.json is not "running" (and idle for --min-age hours) are
touched; the sweep driver's active trial is never archived under it.

  python native/rl/research/compact_evaluations.py native/artifacts/rl/sweeps/overnight-6 ...
  python native/rl/research/compact_evaluations.py --dry-run --glob 'native/artifacts/rl/sweeps/*'
"""
import argparse
import glob
import gzip
import json
import os
from pathlib import Path
import time

import packed

SKIP_NAMES = {'result.json', 'insights.json'}


def visual_files(run_dir):
    for path in sorted(run_dir.glob('evaluations/*/*.json')):
        if path.name in SKIP_NAMES or path.name.endswith('.mind.json'):
            continue
        yield path


def run_is_idle(run_dir, min_age):
    status = run_dir / 'status.json'
    try:
        state = json.loads(status.read_text()).get('status')
    except (OSError, ValueError):
        state = None
    if state == 'running':
        return False
    newest = max((p.stat().st_mtime for p in (status, packed.resolved(run_dir / 'metrics.jsonl')) if p.is_file()),
                 default=0)
    return time.time() - newest >= min_age


def archive(path, level=1):
    raw = path.read_bytes()
    json.loads(raw)  # never archive a truncated file
    target = path.with_name(path.name + '.gz')
    tmp = target.with_name(target.name + f'.{os.getpid()}.tmp')
    with gzip.open(tmp, 'wb', compresslevel=level) as handle:
        handle.write(raw)
    os.replace(tmp, target)
    os.utime(target, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns))
    path.unlink()
    return len(raw), target.stat().st_size


def run_directories(roots):
    for root in roots:
        root = Path(root)
        if (root / 'status.json').is_file():
            yield root
            continue
        for child in sorted(root.iterdir()) if root.is_dir() else []:
            if (child / 'status.json').is_file():
                yield child


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('roots', nargs='*', help='run directories or sweep directories of runs')
    parser.add_argument('--glob', action='append', default=[], help='glob of roots (repeatable)')
    parser.add_argument('--min-age', type=float, default=1.0, help='hours since the run last wrote (default 1)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    roots = list(args.roots) + [p for pattern in args.glob for p in sorted(glob.glob(pattern))]
    before = after = files = 0
    for run_dir in run_directories(roots):
        if not run_is_idle(run_dir, args.min_age * 3600):
            print(f'skip {run_dir} (running or recent)')
            continue
        run_before = run_after = run_files = 0
        for path in visual_files(run_dir):
            if args.dry_run:
                run_before += path.stat().st_size
                run_files += 1
                continue
            try:
                size, packed = archive(path)
            except ValueError:
                print(f'  corrupt, left alone: {path}')
                continue
            run_before += size
            run_after += packed
            run_files += 1
        if run_files:
            print(f'{run_dir}: {run_files} files {run_before / 1e9:.2f} GB -> {run_after / 1e9:.2f} GB')
        before += run_before
        after += run_after
        files += run_files
    verb = 'would archive' if args.dry_run else 'archived'
    print(f'{verb} {files} files: {before / 1e9:.1f} GB -> {after / 1e9:.1f} GB')


if __name__ == '__main__':
    main()
