"""Scorers: read a trial's run directory and produce a (learner_steps, score) curve.

The score is an external yardstick, never the training-time self-play win rate
(which is anchored at ~0.5 by construction). One scorer:

- reference: Bradley-Terry Elo of the last k evaluation checkpoints against a
  frozen reference pool (reference_pool.py: pool ratings held fixed, games
  concentrated on 15-85% matchups, bootstrap CI); the final point is the IQM
  over those k ratings. Unbounded scale: use metric_distribution = "linear".

Each curve point carries the learner steps and the wall-clock elapsed at that
point so the driver can use either as Protein's cost.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class Point:
    __slots__ = ('learner_steps', 'score', 'elapsed')

    def __init__(self, learner_steps, score, elapsed):
        self.learner_steps = int(learner_steps)
        self.score = float(score)
        self.elapsed = float(elapsed)

    def as_dict(self):
        return dict(learner_steps=self.learner_steps, score=self.score, elapsed=self.elapsed)


def _read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, ValueError):
        return default


def _metrics_rows(run_dir, keys):
    """Selected keys from every intact metrics.jsonl row (torn tail tolerated)."""
    rows = []
    try:
        with (Path(run_dir) / 'metrics.jsonl').open() as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    break
                rows.append({key: record.get(key) for key in keys})
    except FileNotFoundError:
        pass
    return rows


def _elapsed_lookup(run_dir):
    """learner_steps -> elapsed seconds, from metrics.jsonl (nearest at or after)."""
    rows = _metrics_rows(run_dir, ('learner_steps', 'elapsed'))
    pairs = [(r['learner_steps'], r['elapsed']) for r in rows
             if r['learner_steps'] is not None and r['elapsed'] is not None]

    def lookup(learner_steps):
        best = None
        for steps, elapsed in pairs:
            if best is None or abs(steps - learner_steps) < abs(best[0] - learner_steps):
                best = (steps, elapsed)
        return best[1] if best else 0.0
    return lookup


def _checkpoint_learner_steps(run_dir, checkpoint_name):
    """learner_steps recorded in the checkpoint sidecar; None when unavailable."""
    sidecar = _read_json(Path(run_dir) / 'checkpoints' / (checkpoint_name + '.json'))
    if sidecar and 'learner_steps' in sidecar:
        return int(sidecar['learner_steps'])
    return None


def final_checkpoint(run_dir):
    best = None
    for path in Path(run_dir).glob('checkpoints/policy-*.bin'):
        match = re.fullmatch(r'policy-(\d+)\.bin', path.name)
        if match and (best is None or int(match.group(1)) > best[0]):
            best = (int(match.group(1)), path)
    return best[1] if best else None


def evaluation_checkpoints(run_dir):
    """[(learner_steps, checkpoint path)] for the run's periodic evaluations in
    step order; the final checkpoint alone when the run has no evaluations."""
    run_dir = Path(run_dir)
    rows = []
    for result_file in sorted(run_dir.glob('evaluations/*/result.json')):
        report = _read_json(result_file)
        if not report or not report.get('checkpoint'):
            continue
        path = run_dir / 'checkpoints' / report['checkpoint']
        learner_steps = _checkpoint_learner_steps(run_dir, report['checkpoint'])
        if path.exists() and learner_steps is not None:
            rows.append((learner_steps, path))
    if not rows:
        final = final_checkpoint(run_dir)
        if final is not None:
            rows.append((_checkpoint_learner_steps(run_dir, final.name)
                         or _read_json(run_dir / 'status.json', {}).get('learner_steps', 0), final))
    return sorted(rows)


def reference_curve(run_dir, pool, last_k=3, games=240, first_pass_games=6, band=(.15, .85),
                    round_games=48, workers=4, seed_base=70001, workspace=None, rate=None, log=None):
    """Fixed-pool Bradley-Terry ratings of the last k evaluation checkpoints.

    Each checkpoint is rated against the frozen reference pool (reference_pool
    .rate_checkpoint, cached in <workspace>/<checkpoint>.json, default
    <run>/reference/). Point i's score is the IQM of the ratings up to i within
    the window, so the final point — the value Protein optimizes — is the IQM
    over the last k evaluations rather than one noisy final evaluation.
    """
    import reference_pool
    run_dir = Path(run_dir)
    rows = evaluation_checkpoints(run_dir)[-max(1, int(last_k)):]
    if not rows:
        return []
    if isinstance(pool, (str, Path)):
        pool = reference_pool.load_pool(pool if Path(pool).is_absolute() else ROOT / pool)
    workspace = Path(workspace) if workspace else run_dir / 'reference'
    rate = rate or reference_pool.rate_checkpoint
    elapsed_at = _elapsed_lookup(run_dir)
    ratings = []
    points = []
    for learner_steps, checkpoint in rows:
        result = rate(checkpoint, pool, workspace, games=games, first_pass_games=first_pass_games,
                      band=tuple(band), round_games=round_games, workers=workers, seed_base=seed_base, log=log)
        ratings.append(result['rating'])
        points.append(Point(learner_steps, reference_pool.iqm(ratings), elapsed_at(learner_steps)))
    return points


def curve(run_dir, scoring):
    """Dispatch on the [scoring] config section."""
    scorer = scoring['scorer']
    if scorer == 'reference':
        return reference_curve(run_dir, scoring['pool'], last_k=int(scoring.get('last_k', 3)),
                               games=int(scoring.get('games', 240)),
                               first_pass_games=int(scoring.get('first_pass_games', 6)),
                               band=tuple(scoring.get('band', (.15, .85))),
                               round_games=int(scoring.get('round_games', 48)),
                               workers=int(scoring.get('workers', 4)),
                               seed_base=int(scoring.get('seed_base', 70001)))
    raise ValueError(f'Unknown scorer {scorer!r}')


def downsample(points, n):
    """PufferLib-style downsample: n bins along learner steps, mean score per
    bin, the final point kept exact. Fewer points than n pass through."""
    if len(points) <= n or n < 2:
        return list(points)
    final = points[-1]
    bin_width = final.learner_steps / (n - 1)
    bins = [[] for _ in range(n - 1)]
    for point in points[:-1]:
        index = min(n - 2, int(point.learner_steps / bin_width))
        bins[index].append(point)
    result = []
    for members in bins:
        if not members:
            continue
        result.append(Point(round(sum(p.learner_steps for p in members) / len(members)),
                            sum(p.score for p in members) / len(members),
                            sum(p.elapsed for p in members) / len(members)))
    result.append(final)
    return result
