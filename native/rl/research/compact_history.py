#!/usr/bin/env python3
"""Archive finished runs' training history in place: `metrics.jsonl` -> `metrics.jsonl.zst`.

`metrics.jsonl` is the bulk of a finished run (1.5 GB of a 1B-step trial) and zstd
-19 packs it ~10x; the worker `game-logs/*.log` are repetitive engine chatter and
pack far harder. Every reader goes through research/packed.py, which resolves
either spelling, so reports keep referring to `metrics.jsonl` unchanged.

A live run appends to `metrics.jsonl` and a zstd frame cannot be appended to, so
only runs whose status.json is not "running" (and idle for --min-age hours) are
touched; the sweep driver's active trial is never archived under it.

  python native/rl/research/compact_history.py native/artifacts/rl/sweeps/overnight-6 ...
  python native/rl/research/compact_history.py --dry-run --glob 'native/artifacts/rl/*'
"""
import argparse
import glob
import json
import os
from pathlib import Path
import subprocess
import time

import packed
from compact_evaluations import run_directories

STALE = 24 * 3600   # a killed run keeps status "running" for ever; its files stop moving


def run_is_idle(run_dir, min_age):
    """As compact_evaluations, with two differences.

    A packed metrics.jsonl no longer dates the run, so resolve it. And a run that
    was killed (crash, abort, reboot) never rewrites status.json, so "running"
    alone cannot mean live: a status that has not moved for STALE is a dead run,
    not a trainer to protect. Only the writer's own clock decides.
    """
    status = run_dir / 'status.json'
    try:
        state = json.loads(status.read_text()).get('status')
    except (OSError, ValueError):
        state = None
    stamps = [p.stat().st_mtime for p in (status, packed.resolved(run_dir / 'metrics.jsonl')) if p.is_file()]
    idle = time.time() - max(stamps, default=0)
    if state == 'running' and idle < STALE:
        return False
    return idle >= min_age


def history_files(run_dir):
    """The run's metrics stream and logs, including the sibling launch log."""
    for path in [run_dir / 'metrics.jsonl',
                 run_dir.with_name(run_dir.name + '.log'),
                 *sorted(run_dir.rglob('*.log'))]:
        if path.is_file():
            yield path


def intact(path):
    """True when every complete line is JSON; a torn tail is normal and kept."""
    if path.name != 'metrics.jsonl':
        return True
    with path.open('rb') as handle:
        previous = None
        for line in handle:
            if previous is not None:
                json.loads(previous)  # raises on a malformed interior line
            previous = line
    return True


def archive(path, level=19, threads=4):
    target = path.with_name(path.name + '.zst')
    tmp = target.with_name(target.name + f'.{os.getpid()}.tmp')
    subprocess.run(['zstd', f'-{level}', '-T{}'.format(threads), '-q', '-f', '--long',
                    str(path), '-o', str(tmp)], check=True)
    size, packed = path.stat().st_size, tmp.stat().st_size
    os.replace(tmp, target)
    os.utime(target, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns))
    path.unlink()
    return size, packed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('roots', nargs='*', help='run directories or sweep directories of runs')
    parser.add_argument('--glob', action='append', default=[], help='glob of roots (repeatable)')
    parser.add_argument('--min-age', type=float, default=1.0, help='hours since the run last wrote (default 1)')
    parser.add_argument('--level', type=int, default=19)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    roots = list(args.roots) + [p for pattern in args.glob for p in sorted(glob.glob(pattern))]
    before = after = files = 0
    for run_dir in run_directories(roots):
        if not run_is_idle(run_dir, args.min_age * 3600):
            print(f'skip {run_dir} (running or recent)', flush=True)
            continue
        run_before = run_after = run_files = 0
        for path in history_files(run_dir):
            if args.dry_run:
                run_before += path.stat().st_size
                run_files += 1
                continue
            try:
                intact(path)
            except ValueError:
                print(f'  corrupt, left alone: {path}', flush=True)
                continue
            size, packed = archive(path, args.level, args.threads)
            run_before += size
            run_after += packed
            run_files += 1
        if run_files:
            print(f'{run_dir}: {run_files} files {run_before / 1e9:.2f} GB -> {run_after / 1e9:.2f} GB', flush=True)
        before += run_before
        after += run_after
        files += run_files
    verb = 'would archive' if args.dry_run else 'archived'
    print(f'{verb} {files} files: {before / 1e9:.1f} GB -> {after / 1e9:.1f} GB')


if __name__ == '__main__':
    main()
