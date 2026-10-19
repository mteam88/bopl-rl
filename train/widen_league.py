"""A league file whose checkpoint members are widened copies, for a wider run's frozen banks.

    native/artifacts/rl/venv/bin/python train/widen_league.py --names-from COMMAND.txt \\
        --out-dir DIR --league-out DIR/league-w512.json
    BOPL_LEAGUE_JSON=DIR/league-w512.json venv/bin/python train/train.py --hidden 512 ...

The trainer plays frozen opponents with the learner's own kernels and shapes, so a
--hidden 512 run refuses every 256-wide member (registry.opponents). Each named
member is widened by surgery.widen_zero, which keeps its actor exactly (KL ~1e-11 on
real games, train/check_widen.py), so the opponents play the same games as the
originals. Every other member row is kept as it is. A widened row keeps the member's
name, kind, kits and lineage; its id, path (absolute) and metadata_sha256 are the
copy's, and `widened_from` records the source id.
"""
import argparse
import contextlib
import io
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'train'), str(ROOT / 'league')]
import registry  # noqa: E402
import surgery  # noqa: E402


def names_from(command):
    argv = shlex.split(Path(command).read_text())
    return argv[argv.index('--opponents') + 1].split(',')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--names', help='comma-separated league member names')
    p.add_argument('--names-from', type=Path, help='a train.py command line (command.txt): its --opponents')
    p.add_argument('--out-dir', type=Path, required=True, help='where the widened checkpoints go')
    p.add_argument('--league-out', type=Path, required=True)
    p.add_argument('--widen', type=int, default=2)
    p.add_argument('--widen-noise', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(argv)
    names = a.names.split(',') if a.names else names_from(a.names_from)

    league = registry.load()
    rows = registry.by_name(league)
    unknown = [n for n in names if n not in rows]
    if unknown:
        raise SystemExit(f'not league members: {unknown}')
    a.out_dir.mkdir(parents=True, exist_ok=True)
    widened = {}
    for name in names:
        row = rows[name]
        if not row.get('path'):
            continue
        source = Path(row['path']) if Path(row['path']).is_absolute() else registry.ROOT / row['path']
        target = (a.out_dir / f'{name}-w{a.widen}x.bin').resolve()
        if not target.exists():
            with contextlib.redirect_stdout(io.StringIO()):
                surgery.main([str(source), str(target), '--widen', str(a.widen), '--widen-init', 'zero',
                              '--widen-noise', str(a.widen_noise), '--seed', str(a.seed)])
        copy = dict(row, id=registry.sha256_file(target), path=str(target),
                    metadata_sha256=registry.sha256_file(str(target) + '.json'), widened_from=row['id'])
        widened[name] = copy
        print(f'{name}: {target.name}')
    members = [widened.get(row['name'], row) for row in league['members']]
    registry.write(members, league['conditions'], a.league_out)
    registry.load(a.league_out)
    print(f'wrote {a.league_out}: {len(widened)} of {len(members)} members widened')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
