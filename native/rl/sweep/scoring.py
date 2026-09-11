"""Scorers: read a trial's run directory and produce a (learner_steps, score) curve.

The score is an external yardstick, never the training-time self-play win rate
(which is anchored at ~0.5 by construction). Three scorers:

- anchors: mean win rate over the run's own periodic evaluations
  (evaluations/*/result.json) against fixed opponents. Timeouts already count
  as losses (evaluate.py: win = flag&2 and not timeout). The opponent list is
  data-driven from the results: every group whose opponent is not 'selfplay',
  so frozen checkpoints from an --evaluation-opponents manifest (and future
  scripted bots) join automatically.
- behaviour: a windowed behaviour rate from metrics.jsonl cumulative counters
  (kills_per_episode, arrow_hit_rate, ...).
- match: head-to-head of the trial's final checkpoint against a fixed enemy
  checkpoint via the native CPU eval library; one point at the final cost.
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


def anchor_curve(run_dir, opponents=(), include_heldout=False):
    """Mean win rate per diagnostic evaluation, over anchor opponent groups.

    opponents: restrict to these opponent names; empty means every opponent
    present except 'selfplay'. include_heldout adds held-out map groups.
    """
    run_dir = Path(run_dir)
    elapsed_at = None
    points = []
    for result_file in sorted(run_dir.glob('evaluations/*/result.json')):
        report = _read_json(result_file)
        if not report:
            continue
        groups = [g for g in report.get('groups', ())
                  if g['opponent'] != 'selfplay'
                  and (not opponents or g['opponent'] in opponents)
                  and (include_heldout or g.get('map_role') != 'untrained')]
        if not groups:
            continue
        score = sum(g['win_rate'] for g in groups) / len(groups)
        learner_steps = _checkpoint_learner_steps(run_dir, report.get('checkpoint', ''))
        if learner_steps is None:
            continue
        if elapsed_at is None:
            elapsed_at = _elapsed_lookup(run_dir)
        points.append(Point(learner_steps, score, elapsed_at(learner_steps)))
    return sorted(points, key=lambda p: p.learner_steps)


BEHAVIOUR_METRICS = ('kills_per_episode', 'arrow_hit_rate', 'arrows_per_episode',
                     'wins_per_episode')


def behaviour_curve(run_dir, metric='kills_per_episode', points=32):
    """Windowed behaviour rate from metrics.jsonl per-update behaviour totals.

    train.py writes each update's behaviour counters for that rollout window
    (not cumulative), so a curve point sums numerator and denominator over the
    rows of its window: the rate over that stretch of training.
    """
    if metric not in BEHAVIOUR_METRICS:
        raise ValueError(f'Unknown behaviour metric {metric!r} (options: {BEHAVIOUR_METRICS})')
    rows = _metrics_rows(run_dir, ('learner_steps', 'elapsed', 'behavior'))
    rows = [r for r in rows if r['behavior'] and r['learner_steps'] is not None]
    if len(rows) < 2:
        return []
    numerator, denominator = {
        'kills_per_episode': ('kills', 'episodes'),
        'arrow_hit_rate': ('arrow_hits', 'arrows_fired'),
        'arrows_per_episode': ('arrows_fired', 'episodes'),
        'wins_per_episode': ('wins', 'episodes'),
    }[metric]
    window = max(1, len(rows) // points)
    curve = []
    for start in range(0, len(rows), window):
        members = rows[start:start + window]
        totals = [m['behavior'].get('totals', {}) for m in members]
        sum_n = sum(t.get(numerator, 0) for t in totals)
        sum_d = sum(t.get(denominator, 0) for t in totals)
        if sum_d <= 0:
            continue
        last = members[-1]
        curve.append(Point(last['learner_steps'], sum_n / sum_d, last['elapsed'] or 0.0))
    return curve


def final_checkpoint(run_dir):
    best = None
    for path in Path(run_dir).glob('checkpoints/policy-*.bin'):
        match = re.fullmatch(r'policy-(\d+)\.bin', path.name)
        if match and (best is None or int(match.group(1)) > best[0]):
            best = (int(match.group(1)), path)
    return best[1] if best else None


def match_curve(run_dir, enemy, games=64, maps=(0,), workers=2, seed_base=60001):
    """One point: final-checkpoint win rate vs a fixed enemy (draws = 0.5).

    Runs the native CPU eval library in-process (research/native_policy.py +
    research/environment.py, the code elo_ladder.py uses). Requires the driver
    process to run under the training venv with BOPL_RL_GAME resolvable.
    """
    run_dir = Path(run_dir)
    checkpoint = final_checkpoint(run_dir)
    if checkpoint is None:
        return []
    status = _read_json(run_dir / 'status.json', {})
    learner_steps = status.get('learner_steps')
    if learner_steps is None:
        learner_steps = _checkpoint_learner_steps(run_dir, checkpoint.name) or 0
    elapsed = status.get('elapsed', 0.0)
    score = play_match(checkpoint, Path(enemy), run_dir / 'match',
                       games=games, maps=maps, workers=workers, seed_base=seed_base)
    return [Point(learner_steps, score, elapsed)]


def play_match(policy_a, policy_b, workspace, games=64, maps=(0,), workers=2,
               seed_base=60001, episode_ticks=None):
    """Side-swapped fixed-seed match; returns A's score in 0..1 (draws 0.5)."""
    from concurrent.futures import ThreadPoolExecutor
    import hashlib
    import numpy as np
    from environment import Environment, configure  # research/ on sys.path
    from native_policy import load
    import layout

    head_bounds = [sum(layout.ACTION_SIZES[:i]) for i in range(layout.NUM_ATNS + 1)]
    models = {}
    for label, path in (('a', Path(policy_a)), ('b', Path(policy_b))):
        policy, meta = load(path)
        models[label] = (policy, meta)
    limit = episode_ticks or models['a'][1]['config']['episode_ticks']

    def actor(label, side, rng):
        import torch
        policy = models[label][0]
        state = policy.initial(2, 'cpu')

        def act(observation, mask):
            nonlocal state
            with torch.no_grad():
                batch = torch.from_numpy(np.ascontiguousarray(observation, dtype=np.float32))
                logits, _, state = policy.step(batch, state)
            row = np.where(np.asarray(mask[side]) != 0,
                           logits.detach().cpu().numpy()[side], -np.inf)
            action = []
            for lo, hi in zip(head_bounds, head_bounds[1:]):
                part = row[lo:hi]
                weights = np.exp(part - part.max())
                weights /= weights.sum()
                action.append(rng.choice(len(part), p=weights))
            return np.asarray(action, dtype=np.int64)
        return act

    tasks = []
    per_map = max(2, games // max(1, len(maps)))
    for map_id in maps:
        for pair in range(per_map // 2):
            seed = seed_base + 1000 * map_id + pair
            tasks.append(dict(map=map_id, seed=seed, side=0))
            tasks.append(dict(map=map_id, seed=seed, side=1))
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    configure(workspace / 'workers')

    def play(bucket):
        outcomes = []
        with Environment(seed=seed_base + len(tasks), limit=limit) as env:
            for task in bucket:
                env.reset(task['seed'], task['map'])
                rng = np.random.default_rng((task['seed'], task['map'], task['side']))
                actors = {task['side']: actor('a', task['side'], rng),
                          1 - task['side']: actor('b', 1 - task['side'], rng)}
                while not env.ended:
                    observation = env.obs.copy()
                    mask = env.mask.copy()
                    env.step(np.stack([actors[0](observation, mask),
                                       actors[1](observation, mask)]))
                final = env.frame()
                timeout = bool(final['flags'] & 2)
                a_won = bool(final['players'][task['side']]['flags'] & 2) and not timeout
                b_won = bool(final['players'][1 - task['side']]['flags'] & 2) and not timeout
                outcomes.append(0.5 if timeout or a_won == b_won else (1.0 if a_won else 0.0))
        return outcomes

    buckets = [tasks[i::workers] for i in range(min(workers, len(tasks)))]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = [o for bucket in pool.map(play, buckets) for o in bucket]
    result = sum(outcomes) / len(outcomes)
    (workspace / 'result.json').write_text(json.dumps(dict(
        schema=1, policy_a=str(policy_a), policy_b=str(policy_b), games=len(outcomes),
        score=result, maps=list(maps), seed_base=seed_base)) + '\n')
    return result


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
    scorer = scoring.get('scorer', 'anchors')
    if scorer == 'anchors':
        return anchor_curve(run_dir, tuple(scoring.get('opponents', ())),
                            bool(scoring.get('include_heldout', False)))
    if scorer == 'behaviour':
        return behaviour_curve(run_dir, scoring.get('metric', 'kills_per_episode'))
    if scorer == 'match':
        return match_curve(run_dir, scoring['enemy'], int(scoring.get('games', 64)),
                           tuple(scoring.get('maps', (0,))), int(scoring.get('workers', 2)))
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
