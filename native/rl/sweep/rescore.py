"""Re-score a finished (or running) sweep's trials with the reference scorer.

Reads a sweep tree read-only and writes everything — per-checkpoint rating
files, flags, rescore.json and summary.md — under --out, so a live sweep's
directory is never touched:

    BOPL_RL_BACKEND=rust native/artifacts/rl/venv/bin/python native/rl/sweep/rescore.py \
        --sweep native/artifacts/rl/sweeps/overnight-3 \
        --pool native/artifacts/rl/sweeps/anchors/pool-v5/pool.json \
        --out native/artifacts/rl/sweeps/overnight-3-rescore-v5

Per trial: the old score (the sweep's final observation), the reference
rating of the final evaluation checkpoint with its bootstrap CI, the IQM over
the last k evaluations (the value the reference scorer hands Protein), and
the final evaluation's behaviour flags. Only complete, scored trials are
rescored (a killed trial's partial curve stays with Protein).
"""
import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / 'native/rl/research'), str(ROOT / 'native/tools')]

import flags  # noqa: E402
import reference_pool  # noqa: E402
import scoring  # noqa: E402


def log(message):
    print(f'[rescore {time.strftime("%H:%M:%S")}] {message}', flush=True)


def rescore(sweep_dir, pool_path, out, last_k=3, games=240, first_pass_games=6, band=(.15, .85),
            round_games=48, workers=4, seed_base=70001, rules=None, trials=None, compare=None):
    sweep_dir = Path(sweep_dir).resolve()
    out = Path(out).resolve()
    if out == sweep_dir or out.is_relative_to(sweep_dir):
        raise SystemExit('--out must lie outside the sweep directory (the sweep tree is read-only here)')
    out.mkdir(parents=True, exist_ok=True)
    state = json.loads((sweep_dir / 'sweep.json').read_text())
    pool = reference_pool.load_pool(pool_path)
    rules = dict(rules or state.get('config', {}).get('scoring', {}).get('flags') or flags.DEFAULT_RULES)
    rows = []
    for trial in state['trials']:
        if trials and trial['id'] not in trials:
            continue
        if trial['status'] != 'complete' or not trial.get('observations'):
            continue  # killed trials keep their partial curve for Protein; their last checkpoint is no final
        run_dir = sweep_dir / trial['id']  # the recorded run_dir goes stale when a sweep is moved
        if not run_dir.is_dir():
            run_dir = Path(trial.get('run_dir') or run_dir)
        workspace = out / trial['id'] / 'reference'
        curve = scoring.reference_curve(run_dir, pool, last_k=last_k, games=games, first_pass_games=first_pass_games,
                                        band=band, round_games=round_games, workers=workers, seed_base=seed_base,
                                        workspace=workspace, log=log)
        if not curve:
            log(f"trial {trial['id']}: no evaluated checkpoints, skipped")
            continue
        ratings = []
        for learner_steps, checkpoint in scoring.evaluation_checkpoints(run_dir)[-last_k:]:
            ratings.append(dict(json.loads((workspace / (checkpoint.stem + '.json')).read_text()),
                                learner_steps=learner_steps, checkpoint=checkpoint.name))
        annotated = flags.annotate_run(run_dir, rules, write=False)
        final = ratings[-1]
        row = dict(id=trial['id'], status=trial['status'], hypers=trial['hypers'],
                   old_score=trial['observations'][-1]['score'], old_steps=trial['observations'][-1]['steps'],
                   rating=final['rating'], interval=final['interval'], games=final['games'],
                   sha256=final['checkpoint_identity']['sha256'],
                   iqm=curve[-1].score, last_k=[dict(checkpoint=r['checkpoint'], learner_steps=r['learner_steps'],
                                                     rating=r['rating'], interval=r['interval'], games=r['games'])
                                                for r in ratings],
                   flags=annotated[-1][1]['flags'] if annotated else [],
                   flag_metrics=annotated[-1][1]['flag_metrics'] if annotated else {},
                   flags_by_evaluation={name: a['flags'] for name, a in annotated},
                   game_mismatch=any(r.get('game_mismatch') for r in ratings))
        rows.append(row)
        reference_pool.atomic_json(out / trial['id'] / 'result.json', row)
        log(f"trial {trial['id']}: old {row['old_score']:.3f} -> {row['rating']:.0f} Elo "
            f"[{row['interval'][0]:.0f}, {row['interval'][1]:.0f}], IQM{last_k} {row['iqm']:.0f}"
            + (f"; flags {', '.join(row['flags'])}" if row['flags'] else ''))
    report = dict(schema=1, sweep=str(sweep_dir), pool=pool['path'], pool_sha256=pool['pool_sha256'],
                  created=time.time(), last_k=last_k, games=games, first_pass_games=first_pass_games,
                  band=list(band), round_games=round_games, seed_base=seed_base, flag_rules=rules, trials=rows,
                  compare=str(Path(compare).resolve()) if compare else None)
    reference_pool.atomic_json(out / 'rescore.json', report)
    (out / 'summary.md').write_text(summary(report))
    return report


def comparison(report, previous):
    """Markdown table of this re-score against an earlier rescore.json (another pool
    or budget): ranks are over the trials present in both."""
    before = {r['id']: r for r in previous['trials']}
    shared = [r for r in report['trials'] if r['id'] in before]
    if not shared:
        return ['', f"No trials shared with `{previous.get('pool')}`."]
    rank_before = {r['id']: i + 1 for i, r in enumerate(sorted(shared, key=lambda r: -before[r['id']]['iqm']))}
    rank_after = {r['id']: i + 1 for i, r in enumerate(sorted(shared, key=lambda r: -r['iqm']))}
    k = report['last_k']
    lines = ['', f"## Against the previous re-score", '',
             f"Previous: pool `{previous.get('pool')}` ({str(previous.get('pool_sha256', ''))[:16]}), "
             f"{len(shared)} trials in common; ranks below are within those trials.", '',
             f"| trial | previous IQM{k} | previous rank | IQM{k} | rank | move | rating shift |",
             '|---|---|---|---|---|---|---|']
    for row in sorted(shared, key=lambda r: rank_after[r['id']]):
        old = before[row['id']]
        move = rank_before[row['id']] - rank_after[row['id']]
        lines.append(f"| {row['id']} | {old['iqm']:.0f} | {rank_before[row['id']]} | {row['iqm']:.0f} | {rank_after[row['id']]} | "
                     f"{'+' if move > 0 else ''}{move if move else '='} | {row['iqm'] - old['iqm']:+.0f} |")
    changed = [r['id'] for r in shared if rank_before[r['id']] != rank_after[r['id']]]
    lines += ['', f"Rank changes against the previous re-score: {', '.join(sorted(changed)) if changed else 'none'}."]
    return lines


def summary(report):
    rows = report['trials']
    by_old = sorted(rows, key=lambda r: -r['old_score'])
    by_new = sorted(rows, key=lambda r: -r['iqm'])
    old_rank = {r['id']: i + 1 for i, r in enumerate(by_old)}
    new_rank = {r['id']: i + 1 for i, r in enumerate(by_new)}
    lines = [f"# Reference re-score of {Path(report['sweep']).name}", '',
             f"Pool `{Path(report['pool']).relative_to(ROOT) if Path(report['pool']).is_relative_to(ROOT) else report['pool']}` "
             f"({report['pool_sha256'][:16]}), {report['games']} games per checkpoint (first pass "
             f"{report['first_pass_games']} per member, then the {report['band'][0]:.0%}-{report['band'][1]:.0%} band), "
             f"IQM over the last {report['last_k']} evaluations, 95% bootstrap CI over seed pairs. "
             f"Old score: the anchors scorer's final observation. Flags: {', '.join(f'{k} ({v})' for k, v in report['flag_rules'].items())}.", '',
             f"| trial | old score | old rank | final rating (95% CI) | IQM{report['last_k']} | new rank | move | flags |",
             '|---|---|---|---|---|---|---|---|']
    for row in by_new:
        move = old_rank[row['id']] - new_rank[row['id']]
        lines.append(f"| {row['id']} | {row['old_score']:.3f} | {old_rank[row['id']]} | "
                     f"{row['rating']:.0f} [{row['interval'][0]:.0f}, {row['interval'][1]:.0f}] | {row['iqm']:.0f} | "
                     f"{new_rank[row['id']]} | {'+' if move > 0 else ''}{move if move else '='} | {' '.join(row['flags'])} |")
    changed = [r['id'] for r in rows if old_rank[r['id']] != new_rank[r['id']]]
    lines += ['', f"Rank changes: {', '.join(changed) if changed else 'none'}.", '',
              '## Last-k ratings per trial', '']
    for row in sorted(rows, key=lambda r: r['id']):
        parts = ', '.join(f"{r['checkpoint']}: {r['rating']:.0f} [{r['interval'][0]:.0f}, {r['interval'][1]:.0f}]"
                          for r in row['last_k'])
        lines.append(f"- {row['id']}: {parts}")
    if report.get('compare'):
        lines += comparison(report, json.loads(Path(report['compare']).read_text()))
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sweep', type=Path, required=True, help='Sweep directory (read-only)')
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='Output directory outside the sweep')
    parser.add_argument('--last-k', type=int, default=3)
    parser.add_argument('--games', type=int, default=240)
    parser.add_argument('--first-pass-games', type=int, default=6)
    parser.add_argument('--band', default='0.15,0.85')
    parser.add_argument('--round-games', type=int, default=48)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed-base', type=int, default=70001)
    parser.add_argument('--trials', help='Comma-separated trial ids (default: every scored trial)')
    parser.add_argument('--compare', type=Path, help='Earlier rescore.json to rank against (another pool)')
    args = parser.parse_args()
    low, high = (float(v) for v in args.band.split(','))
    report = rescore(args.sweep, args.pool, args.out, last_k=args.last_k, games=args.games,
                     first_pass_games=args.first_pass_games, band=(low, high), round_games=args.round_games,
                     workers=args.workers, seed_base=args.seed_base,
                     trials=set(args.trials.split(',')) if args.trials else None, compare=args.compare)
    print(summary(report))


if __name__ == '__main__':
    main()
