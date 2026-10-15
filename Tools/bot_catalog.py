#!/usr/bin/env python3
"""Write the bot catalog the in-game opponent picker reads (RlAgentPicker.cs).

  Tools/bot_catalog.py [--out native/artifacts/rl/human/bot-catalog.json]

One JSON file listing every opponent the agent shim (train/bopl_agent.h) can
run, with what the picker shows and offers for each:

  bots[]   id (league member name, bot name, roster name), title, kind, group,
           spec (what BOPL_AGENT takes: a checkpoint path relative to the
           checkout, a bot name, or a .roster path), description, rating and
           interval (league Elo, random = 0) when the league has one, kits (the
           loadouts the bot can play), pool (for a net trained on random kits:
           the abilities its training drew every slot from, so any three of
           them is a kit it knows; null otherwise) and kit_ratings (per-kit Elo
           from a `league candidate` rating, when there is one)
  default  {bot, kit}: the newest league main, on a random kit (from its set or pool)
  abilities  {id: name} (native/data/rollout.json), for the kit labels

Sources: league/league.json for the members and their kits (`loadouts`, read
as registry.kits does: scripted bots only on 2,4,14, a checkpoint on what it
trained on); ratings are the rating condition's Bradley-Terry fit over
league/payoff.jsonl, else the member's stored `league candidate` rating
(<run>/league/<stem>.json, refitted on the current ratings). Rating needs the
league package's numpy; without it the catalog is written without ratings.
Search bots are left out (the shim refuses them: they fork the simulator) and
so is the random anchor (no weights). Rosters are native/artifacts/rl/human/
rosters/*.roster, on their first member's kit (the shim requires one kit).

The launchers (Tools/play-vs-agent.sh, play-vs-agent-mac.sh) run this on every
launch; paths are relative to the checkout so the file works on the Mac too,
where the picker hides any bot whose checkpoint is not present.
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEAGUE = ROOT / 'league/league.json'
ROSTERS = ROOT / 'native/artifacts/rl/human/rosters'
DEFAULT_OUT = ROOT / 'native/artifacts/rl/human/bot-catalog.json'
SCRIPTED_KIT = [2, 4, 14]   # bopl_bots.h BOT_SLOT_DASH/BOW/PLATFORM

KIND_TEXT = {
    'archer': 'Keeps its distance and fights with charged arrows, aimed by a ballistic solver; dashes and platforms to stay safe.',
    'pursuer': 'Closes in and fights up close: dash-ins, point-blank arrows and platform drops on top of you.',
    'evasive': 'Plays keep-away: dodges incoming arrows and counters when you overcommit.',
    'mixed': 'Switches between archer, pursuer and evasive play with the situation.',
    'rusher': 'Relentless pressure: never backs off a ready bow, hops in on you, dash-ins, point-blank arrows, '
              'and sweeps its platform along the floor at you to squish. Keeps its distance only while its own bow reloads.',
}
LEVEL_TEXT = {
    'easy': 'Easy: skips about a fifth of its decisions, reacts late and aims loosely.',
    'medium': 'Medium: occasional hesitation and slightly loose aim.',
    'hard': 'Hard: reacts at once with exact aim.',
}
GROUPS = ['League', 'Exploiters', 'Specialists', 'Scripted v2', 'Scripted v1', 'Rosters']


def ability_names():
    rows = json.loads((ROOT / 'native/data/rollout.json').read_text())['abilities']
    return {int(row['id']): row['name'] for row in rows}


def league_ratings(members):
    """({name: (elo, [low, high], source)}, {name: {kit: elo}}, condition) or empty on any failure."""
    try:
        sys.path[:0] = [str(ROOT)]
        from league import candidate, rating, registry
        key, _ = registry.rating_condition()
        counts = registry.read_payoff(condition=key)
        ids = {m['id'] for m in members}
        names = sorted({i for pair in counts for i in pair if i in ids})
        fit = rating.ratings(names, {pair: value for pair, value in counts.items()
                                     if pair[0] in ids and pair[1] in ids})
        by_id = {m['id']: m for m in members}
        out = {by_id[i]['name']: (fit['ratings'][i], list(fit['intervals'][i]), 'round robin')
               for i in names}
        per_kit = {}
        reference = (key, fit['ratings'])
        for m in members:
            if not m.get('path'):
                continue
            stored = candidate.result_path(ROOT / m['path'])
            payload = candidate.load(stored, reference, bootstrap=200) if stored.exists() else None
            if payload is None:
                continue
            if m['name'] not in out:
                out[m['name']] = (payload['rating'], list(payload['interval']),
                                  f"{payload.get('source', 'candidate')}, {payload.get('games')} games")
            per_kit[m['name']] = {kit: row['rating'] for kit, row in payload.get('kits', {}).items()}
        return out, per_kit, key
    except Exception as error:   # numpy missing (Mac), no payoff yet, ...
        print(f'bot_catalog: no ratings ({type(error).__name__}: {error})', file=sys.stderr)
        return {}, {}, None


def kit_label(kit, names):
    return ' / '.join(names.get(a, str(a)) for a in kit)


def sidecar(path):
    try:
        return json.loads(Path(str(path) + '.json').read_text())
    except (OSError, ValueError):
        return {}


def first_key(node, key):
    """The shim's sidecar reading (bopl_json_find): the first `key` anywhere, depth first."""
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            found = first_key(value, key)
            if found is not None:
                return found
    return None


def checkpoint_kit(path):
    meta = sidecar(path)
    kit = [first_key(meta, f'ability{s}') for s in (1, 2, 3)]
    return [int(a) for a in kit] if all(a is not None for a in kit) else SCRIPTED_KIT


def pool_of(member):
    """The abilities a net trained on random kits drew from (its sidecar's
    config.loadouts, mode "pool"), without 0 (none) and 1 (the select screen's
    Random tile); None for any other member."""
    if member.get('loadouts') or not member.get('path'):
        return None
    loadouts = sidecar(ROOT / member['path']).get('config', {}).get('loadouts') or {}
    if loadouts.get('mode') != 'pool':
        return None
    return sorted({int(a) for a in loadouts.get('pool') or () if int(a) > 1}) or None


def title_of(name):
    return name.replace('-', ' ').upper()


def describe_learned(member, names):
    kind = member['kind']
    kits = member['loadouts']
    meta = sidecar(ROOT / member['path']) if member.get('path') else {}
    config = meta.get('config', {})
    steps = (member.get('origin') or {}).get('step')
    maps = config.get('training_maps')
    arch = meta.get('architecture', '')
    trained = []
    if steps:
        # origin.step counts physical decisions; the learner's share is the run's own ratio
        if config.get('steps') and config.get('learner_steps'):
            trained.append(f"{steps * config['learner_steps'] / config['steps'] / 1e9:.1f}B learner steps")
        else:
            trained.append(f'{steps / 1e9:.1f}B steps')
    if maps:
        trained.append(f'{len(maps)} maps')
    trained = (' Trained ' + ' on '.join(trained) + '.') if trained else ''
    family = ' v4 policy.' if arch.endswith('v4') or 'v4' in arch else ''
    if kind == 'specialist':
        return (f"Specialist: trained only on its own kit, {kit_label(kits['fixed'], names)}, and frozen as "
                f"the strongest known play on it.{trained}")
    rule = member['added']['rule']
    if kind == 'exploiter':
        target = member.get('target') or 'a league member'
        return f'Exploiter: trained from scratch to beat {target}.{trained} {rule}'.strip()
    if kits and kits['mode'] == 'set':
        plays = f" Plays {len(kits['kits'])} kits (Matthew's lineup set)."
    elif kits and kits['mode'] == 'fixed':
        plays = f" Plays {kit_label(kits['fixed'], names)}."
    else:
        pool = pool_of(member)
        plays = f' Trained on random kits from {len(pool)} abilities: plays any kit.' if pool else ' Plays any kit.'
    origin = rule.split(':', 1)[1].strip() if rule.startswith('promoted') and ':' in rule else rule
    head = 'League main' + (', promoted ' + member['added']['date'] if rule.startswith('promoted') else '')
    return f'{head}.{family}{trained}{plays} {origin[:1].upper() + origin[1:]}.'.replace('..', '.')


def describe_scripted(name):
    parts = name.split('-')
    level, family = parts[-1], parts[0]
    v2 = 'v2' in parts
    text = KIND_TEXT.get(family, '') + ' ' + LEVEL_TEXT.get(level, '')
    if family == 'rusher':
        text += ' Built on v2\'s survival layers (arrow dodge, platform threat, landing checks).'
    else:
        text += (' v2 fixes v1\'s main losses: platform squishes, its own arrows, attacking into a ready bow.' if v2
                 else ' The original v1 bot.')
    return text + ' Hand-coded, sees the true game state; built for Dash / Bow / Platform.'


def roster_entry(path, names):
    members = []
    for line in path.read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if line:
            members.append(line.split()[0])
    if not members:
        return None
    first = members[0]
    if first.endswith('.bin'):
        kit = checkpoint_kit(first if first.startswith('/') else path.parent / first)
    else:
        kit = SCRIPTED_KIT
    return dict(id='roster:' + path.stem, title=title_of(path.stem), kind='roster', group='Rosters',
                spec=str(path.relative_to(ROOT)), rating=None, interval=None, rating_source=None,
                description=f'Rating roster of {len(members)} members: each round draws one, weighted '
                            f'(Tools/rating_roster.py). All play {kit_label(kit, names)}.',
                kits=[kit], pool=None, kit_ratings={})


def build():
    names = ability_names()
    league = json.loads(LEAGUE.read_text())
    members = league['members']
    ratings, per_kit, condition = league_ratings(members)
    bots = []
    for m in members:
        kind = m['kind']
        if kind == 'random' or kind == 'human':
            continue
        if kind == 'scripted':
            if m['name'].startswith('search-'):
                continue
            spec, kits, pool = m['name'], [SCRIPTED_KIT], None
            group = 'Scripted v2' if '-v2-' in m['name'] or m['name'].startswith('rusher-') else 'Scripted v1'
            description = describe_scripted(m['name'])
        else:
            if not m.get('path'):
                continue
            spec = m['path']
            loadouts = m.get('loadouts')
            kits = ([list(loadouts['fixed'])] if loadouts and loadouts['mode'] == 'fixed'
                    else [list(k) for k in loadouts['kits']] if loadouts else [])
            pool = pool_of(m)
            group = {'main': 'League', 'exploiter': 'Exploiters', 'specialist': 'Specialists'}[kind]
            description = describe_learned(m, names)
        elo, interval, source = ratings.get(m['name'], (None, None, None))
        bots.append(dict(id=m['name'], title=title_of(m['name']), kind=kind, group=group, spec=spec,
                         rating=round(elo, 1) if elo is not None else None,
                         interval=[round(v) for v in interval] if interval else None, rating_source=source,
                         description=description, kits=kits, pool=pool,
                         kit_ratings={k: round(v) for k, v in per_kit.get(m['name'], {}).items()},
                         added=m['added']['date']))
    if ROSTERS.is_dir():
        for path in sorted(ROSTERS.glob('*.roster')):
            entry = roster_entry(path, names)
            if entry:
                bots.append(entry)
    # Default: the newest main (members are append-only, so the last one added).
    mains = [b for b in bots if b['kind'] == 'main']
    default = mains[-1] if mains else bots[0]
    order = {g: i for i, g in enumerate(GROUPS)}
    bots.sort(key=lambda b: (order[b['group']], b['id'] != default['id'],
                             -(b['rating'] if b['rating'] is not None else -1e9), b['id']))
    return dict(schema=1, generated=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
                condition=condition, rating_note='league Elo (Bradley-Terry, random = 0)' if condition else None,
                default=dict(bot=default['id'], kit='random' if len(default['kits']) > 1 or default.get('pool') else 'fixed'),
                abilities={str(k): v for k, v in names.items()}, groups=GROUPS, bots=bots)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    catalog = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix('.tmp')
    tmp.write_text(json.dumps(catalog, indent=1) + '\n')
    tmp.replace(args.out)
    rated = sum(b['rating'] is not None for b in catalog['bots'])
    print(f"bot catalog: {len(catalog['bots'])} bots ({rated} rated), default {catalog['default']['bot']} -> {args.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
