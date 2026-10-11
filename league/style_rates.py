"""How often a league member (or a checkpoint) plays a style, in the league's own games.

    python league style-rates l1-main-8000m --styles styles/a.toml styles/b.toml [--games 240] [--json out.json]

The target plays exactly the games `league candidate` would give its checkpoint
(candidate.play_band: same kits, maps, seeds and band rounds), with the styles
compiled into one program measured in every game: the C VM (train/bopl_style.h)
runs it for both players after each decision inside the research environment,
as the trainer does for a learner, and only tallies it (eval/environment.py
install_style), so the games are unchanged. The table has eval/style_score.py's
columns, per player-game averages, so the rows line up with the human table:
`member` is the target's side, `opponents` the members' side in the same games
(what is done to it), and `member_won` / `member_lost` split the target's rows
by outcome like style_score's human_winner / human_loser. A last column pools
all fires over all minutes, which short games do not dominate.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / 'eval'), str(ROOT / 'native/tools'), str(ROOT / 'train')]

import registry  # noqa: E402
import candidate  # noqa: E402
import elo_ladder  # noqa: E402
from evaluation_protocol import checkpoint_identity  # noqa: E402

GROUPS = ('member', 'opponents', 'member_won', 'member_lost')


def compile_styles(paths):
    """{data, terms, sha256} of the styles compiled into one program (train.py --style)."""
    import style_compile
    compiled = style_compile.compile_files([Path(p) for p in paths])
    return dict(data=compiled.data, terms=list(compiled.term_names),
                sha256=hashlib.sha256(compiled.data).hexdigest())


def target(token, league):
    """(label, checkpoint path, member-like row for the candidate seat, sha256 to exclude)
    for a member name or a checkpoint path."""
    rows = registry.by_name(league)
    if token in rows:
        row = rows[token]
        if not row.get('path'):
            raise SystemExit(f'{token} is a {row["kind"]} member with no weights; style-rates plays checkpoints')
        checkpoint = registry.ROOT / row['path']
        loadouts = row.get('loadouts')
        label = token
    else:
        checkpoint = Path(token).resolve()
        if not checkpoint.is_file():
            raise SystemExit(f'{token!r} is neither a member name nor a checkpoint file')
        config = json.loads(Path(str(checkpoint) + '.json').read_text()).get('config') or {}
        loadouts = registry.trained_loadouts(config)
        label = checkpoint.name
    identity = checkpoint_identity(checkpoint)
    me = dict(id=candidate.CANDIDATE, name=candidate.CANDIDATE, kind='candidate', loadouts=loadouts)
    return label, checkpoint, me, identity


def rows_of(played, terms):
    """{group: [player-game rows]} from candidate.play_band rows carrying style tallies."""
    per = {g: [] for g in GROUPS}
    for game in played:
        minutes = game['ticks'] / 3600.0
        mine, theirs = game['side'], 1 - game['side']
        for group, seat in (('member', mine), ('opponents', theirs)):
            row = dict(b=game['b'], map=game['map'], seed=game['seed'], side=game['side'], minutes=minutes,
                       decisions=game['style_decisions'][seat],
                       pay=[game['style'][seat][t][0] for t in terms],
                       fires=[game['style'][seat][t][1] for t in terms])
            per[group].append(row)
            if group == 'member' and game['outcome'] != 'draw':
                per['member_won' if game['outcome'] == 'a' else 'member_lost'].append(row)
    return per


def table(per, terms):
    """eval/style_score.py's columns (mean over player-games of each rate), plus the
    pooled fires/min (all fires over all minutes): league games are often short, and a
    single kill in a 20 s game is 3/min, so the per-game mean leans on short games."""
    out = []
    for t, name in enumerate(terms):
        for group in GROUPS:
            items = per[group]
            if not items:
                continue
            minutes = sum(it['minutes'] for it in items)
            out.append(dict(term=name, group=group, n=len(items),
                            fire_fraction=float(np.mean([it['fires'][t] / it['decisions'] for it in items])),
                            fires_per_min=float(np.mean([it['fires'][t] / it['minutes'] for it in items])),
                            pay_per_min=float(np.mean([it['pay'][t] / it['minutes'] for it in items])),
                            games_with_fire=float(np.mean([it['fires'][t] > 0 for it in items])),
                            pooled_fires_per_min=float(sum(it['fires'][t] for it in items) / minutes),
                            minutes=float(minutes)))
    return out


def markdown(rows, groups=('member', 'opponents')):
    lines = ['| term | group | n | fire frac | fires/min | pay/min | games with fire | pooled fires/min |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['group'] in groups:
            lines.append(f"| {r['term']} | {r['group']} | {r['n']} | {r['fire_fraction']:.4f} | "
                         f"{r['fires_per_min']:.3f} | {r['pay_per_min']:.5f} | {r['games_with_fire']:.3f} | "
                         f"{r['pooled_fires_per_min']:.3f} |")
    return '\n'.join(lines)


def run(token, styles, games=240, first_pass_games=6, band=(.15, .85), round_games=48, workers=4,
        seed_base=70001, workspace='/tmp/league-style-rates', condition=None, log=print):
    league = registry.load()
    label, checkpoint, me, identity = target(token, league)
    style = compile_styles(styles)
    key, ratings, rows, maps, used = candidate.opponents(me, identity['sha256'], seed_base, league, condition)
    started = time.monotonic()
    played, rounds = candidate.play_band(elo_ladder.Model(candidate.CANDIDATE, checkpoint, 'candidate'), me, rows,
                                         maps, used, Path(workspace), games, first_pass_games, band, round_games,
                                         workers, seed_base, style=style, log=log, label=label)
    support = [game['event_support'] for game in played]
    sim = registry.rating_condition()[0]
    per = rows_of(played, style['terms'])
    result = dict(target=label, checkpoint=str(checkpoint), checkpoint_identity=identity, condition=key,
                  played_under=sim,
                  styles=[str(p) for p in styles], program_sha256=style['sha256'], terms=style['terms'],
                  seed_base=seed_base, budget=games, band=list(band), first_pass_games=first_pass_games,
                  round_games=round_games, games=len(played),
                  event_support=dict(games=len(support), supported=int(sum(support))),
                  fit=candidate.fit(played, used, 200, seed_base),
                  names={member: rows[member]['name'] for member in used},
                  rounds=rounds, elapsed=time.monotonic() - started,
                  table=table(per, style['terms']), played=played)
    return result


def report(result):
    fit = result['fit']
    support = result['event_support']
    head = (f"{result['target']}: {result['games']} games on {result['played_under']}, "
            f"rated against {result['condition']}: {fit['rating']:.0f} "
            f"[{fit['interval'][0]:.0f}, {fit['interval'][1]:.0f}], program sha256 {result['program_sha256'][:16]}, "
            f"event support {support['supported']}/{support['games']} games")
    if support['supported'] != support['games']:
        head += '\n  WARNING: some games ran without the event stream; event terms read 0 there'
    return head + '\n' + markdown(result['table'])
