"""League training on one GPU: a main agent, main exploiters and league exploiters.

AlphaStar's league (Vinyals et al., Nature 575 (2019), Methods "League
composition" and the Supplementary pseudocode), cut down to one main and run as
turns on one GPU instead of agents on their own accelerators:

  main              never reset. Plays half its games against itself (the
                    vector's learner-vs-learner environments) and the rest by
                    PFSP over its own snapshots and every league member, with a
                    fixed share of banks on the exploiters (--bank-groups;
                    AlphaStar's 35/50/15). A snapshot joins the league after
                    every turn: AlphaStar adds main players every 2e9 steps.
  main exploiter    trains only against the newest main snapshot (and that
                    main's two predecessors while it scores under 20%,
                    AlphaStar's fallback to the main's past checkpoints). Joins
                    the league when it beats its target >70% over a turn, or at
                    its timeout (4e9 steps), and is then reset.
  league exploiter  PFSP over every league member, never its own snapshots.
                    Joins at >70% against the league, or at its 2e9 timeout;
                    then resets with probability 0.25 and otherwise continues.
  style exploiter   not in AlphaStar (docs/STYLES.md): a main exploiter that
                    also gets a small fading bonus for playing one human style
                    (train.py --style). Each one cycles to the next entry of
                    [style-exploiter] styles; it joins like a main exploiter
                    (>70% against the newest main, or its timeout) and resets.

Where this departs from the paper, on purpose:
  - Resets start from the newest main snapshot, not a supervised policy: there
    is no human-data imitation agent here. A random-init exploiter (exploit-g111-1,
    2026-10-09) reached only 0.29 against its main in 6.9B learner steps.
  - Turns, not concurrency: the rotation decides the compute split (default
    main 50%, main exploiter 25%, league exploiter 25%; AlphaStar's per-race
    split is 1:1:2, but it had three main agents to keep honest and we have one).
  - Exploiter "steps" are learner decisions, the unit train.py counts.

Each turn is one train.py process (fresh, or --resume of the agent's run with a
larger --steps), launched in its own systemd scope so it outlives a restart of
this loop; a restarted loop adopts a turn still running, or resumes it. New
league members reach a continuing run through --opponents on --resume
(native_selfplay.reconcile). Everything the loop decides is in <dir>/state.json
and <dir>/turns.jsonl; league members are added with `league add`, so they are
immediately playable from the in-game picker and rateable with `league candidate`.

    python train/league_loop.py train/league_loops/l1.toml

Decisions by evaluation (l2 on; [evals] decide = true). l1 judged an exploiter
by its own training ledger (>70% over a turn); l2 judges it with a held-out,
paired evaluation and never registers what it cannot show:
  main exploiter    each new one takes the next RECIPE from [main-exploiter]
                    recipes (a TOML queue of [[recipe]] name, loadout_set,
                    reward_timeout, styles; re-read at every reset so other
                    agents can append), warm-starts from the newest main and
                    trains with --loadout-learner-first against that main only.
                    After each of its turns eval/exploit_eval.py plays it AND the
                    main itself (the control) from the same seeds and kits
                    against the main; with the paired score difference's 95%
                    interval above [evals] exploit_margin it joins the league
                    (with its kits, so mains meet it pinned to them) and
                    resets; at its timeout it resets unregistered.
  league exploiter  random init with league_exploiter_random_init; PFSP over
                    the whole league; after each turn league/all_combos.py plays
                    it against league_members sampled members, and it joins when
                    the pooled score's interval clears 0.5 + league_margin.
  main              after each turn, besides the snapshot, a cheap gate:
                    league/all_combos.py against [evals] gate_opponents (e.g.
                    random, f5, A20) with intervals, logged to turns.jsonl and
                    one readable line per turn in summary.md.
Evaluations are CPU jobs run one at a time in their own capped units
(state.json `jobs`), overlapping the next GPU turn; an exploiter's next turn
waits for its decision. seed_members = "league" makes every league member the
main's architecture can load an opponent, re-read each turn.
"""
import argparse
import json
import os
from pathlib import Path
import random
import shlex
import signal
import subprocess
import sys
import time
import tomllib
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'train/sweep'), str(ROOT / 'league')]
PUFFERLIB_DIR = sorted((ROOT / 'native/artifacts/rl/dependencies').glob('PufferLib-*'))[-1]

import registry  # noqa: E402
import sweep_config  # noqa: E402

VENV_PYTHON = ROOT / 'native/artifacts/rl/venv/bin/python'
TRAIN = ROOT / 'train/train.py'
LEAGUE = ROOT / 'league'
NTFY = 'http://127.0.0.1:2586/bopl'
AGENTS = ('main', 'main-exploiter', 'league-exploiter', 'style-exploiter')
# Exploiter slot -> run-directory and member-name tag.
TAGS = {'main-exploiter': 'me', 'league-exploiter': 'le', 'style-exploiter': 'se'}
# Flags the loop owns: where a turn writes, how long it runs, what it starts
# from and whom it plays.
OWNED = ('output', 'resume', 'steps', 'init_weights', 'opponents')


def log(message):
    print(f'[league-loop {time.strftime("%m-%d %H:%M:%S")}] {message}', flush=True)


def notify(message, priority='default'):
    try:
        subprocess.run(['curl', '-s', '-m', '10', '-H', 'Title: bopl league', '-H', f'Priority: {priority}',
                        '-d', message, NTFY],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except OSError:
        pass


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, allow_nan=False, indent=1) + '\n')
    temporary.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return default


def unit_active(unit):
    """A turn runs in a .scope and a rating in a .service: name the unit fully,
    or systemctl looks up `<unit>.service` and a running scope reads as inactive."""
    return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit]).returncode == 0


# ---- configuration -----------------------------------------------------------

class Config:
    def __init__(self, path):
        raw = tomllib.loads(Path(path).read_text())
        loop = raw['loop']
        self.path = Path(path)
        self.name = loop['name']
        self.dir = ROOT / loop['dir']
        self.init = (ROOT / loop['init']).resolve()
        self.init_member = loop['init_member']
        # "league": every member the main's architecture can load, re-read each turn.
        self.seed_league = loop.get('seed_members') == 'league'
        self.seed_members = [] if self.seed_league else list(loop.get('seed_members', []))
        self.main_turn = int(loop.get('main_turn', 2_000_000_000))
        self.exploiter_turn = int(loop.get('exploiter_turn', 1_000_000_000))
        self.rotation = list(loop.get('rotation', ['main', 'main-exploiter', 'main', 'league-exploiter']))
        self.main_exploiter_timeout = int(loop.get('main_exploiter_timeout', 4_000_000_000))
        self.league_exploiter_timeout = int(loop.get('league_exploiter_timeout', 2_000_000_000))
        self.win = float(loop.get('win', 0.70))
        self.hopeless = float(loop.get('hopeless', 0.20))
        self.min_games = int(loop.get('min_games', 2000))
        self.league_exploiter_reset = float(loop.get('league_exploiter_reset', 0.25))
        self.main_steps = int(loop.get('main_steps', 100_000_000_000))
        self.memory_max = loop.get('memory_max', '10G')
        self.memory_high = loop.get('memory_high')
        # Extra systemd properties for every turn's scope (e.g. "CPUQuota=400%").
        self.unit_properties = list(loop.get('unit_properties', []))
        self.league_exploiter_random_init = bool(loop.get('league_exploiter_random_init', False))
        self.rating_games = int(loop.get('rating_games', 480))
        self.rating_workers = int(loop.get('rating_workers', 4))
        # A rating to beat (the init checkpoint's): a main snapshot above it notifies at high priority.
        self.beat_rating = loop.get('beat_rating')
        self.style_exploiter_timeout = int(loop.get('style_exploiter_timeout', 2_000_000_000))
        self.train = dict(raw.get('train', {}))
        self.agent = {agent: dict(raw.get(agent, {})) for agent in AGENTS}
        # [style-exploiter] styles: one entry per style exploiter, cycled. An
        # entry is a style TOML or a list of them (one program, docs/STYLES.md).
        self.styles = [[entry] if isinstance(entry, str) else list(entry)
                       for entry in self.agent['style-exploiter'].pop('styles', [])]
        # [main-exploiter] recipes: the queue file each new main exploiter takes the next entry of.
        self.recipes_path = self.agent['main-exploiter'].pop('recipes', None)
        evals = dict(raw.get('evals', {}))
        self.evals = dict(decide=False, workers=4, memory_high='6G', seed_base=191001, gate_pairs=200,
                          gate_seed_base=131001, gate_opponents={}, exploit_pairs=200, exploit_margin=0.0,
                          league_members=6, league_pairs=100, league_margin=0.0, unit_properties=[])
        unknown = set(evals) - set(self.evals)
        if unknown:
            raise SystemExit(f'[evals] has unknown keys {sorted(unknown)}')
        self.evals.update(evals)
        if self.evals['decide'] and 'main-exploiter' in self.rotation and not self.recipes_path:
            raise SystemExit('[evals] decide needs [main-exploiter] recipes')
        if self.recipes_path:
            self.load_recipes()
        if 'style-exploiter' in self.rotation and not self.styles:
            raise SystemExit('style-exploiter is in the rotation but [style-exploiter] styles is empty')
        for entry in self.styles:
            for spec in entry:
                if not (ROOT / spec).is_file():
                    raise SystemExit(f'style {spec} is missing')
        if any(agent not in AGENTS for agent in self.rotation) or 'main' not in self.rotation:
            raise SystemExit(f'rotation must name agents from {AGENTS} and include main')
        for table in [self.train, *self.agent.values()]:
            for key in table:
                if key in OWNED:
                    raise SystemExit(f'{key} is set by the loop, not the config')
                if key not in sweep_config.TRAIN_FLAGS:
                    raise SystemExit(f'{key} is not a train.py flag')

    def load_recipes(self):
        """The main-exploiter queue: [[recipe]] name, loadout_set, reward_timeout, styles
        (a list of style TOMLs, one program), optional timeout (learner steps) and
        train (extra train.py flags). Read fresh at every reset."""
        recipes = tomllib.loads((ROOT / self.recipes_path).read_text()).get('recipe', [])
        if not recipes:
            raise SystemExit(f'{self.recipes_path}: no [[recipe]] entries')
        for recipe in recipes:
            missing = {'name', 'loadout_set', 'reward_timeout'} - set(recipe)
            if missing:
                raise SystemExit(f'{self.recipes_path}: recipe {recipe.get("name")} lacks {sorted(missing)}')
            for path in [recipe['loadout_set'], *recipe.get('styles', [])]:
                if not (ROOT / path).is_file():
                    raise SystemExit(f'{self.recipes_path}: recipe {recipe["name"]}: {path} is missing')
            for key in recipe.get('train', {}):
                if key in OWNED or key not in sweep_config.TRAIN_FLAGS:
                    raise SystemExit(f'{self.recipes_path}: recipe {recipe["name"]}: train key {key} not allowed')
        return recipes

    def flags(self, agent, override=None):
        merged = dict(self.train)
        merged.update(self.agent[agent])
        merged.update(override or {})
        out = []
        for key in sorted(merged):
            value = merged[key]
            flag = '--' + key.replace('_', '-')
            if key in sweep_config.STORE_TRUE:
                if value:
                    out.append(flag)
                continue
            out += [flag, sweep_config._render(value)]
        return out


# ---- the league as this loop sees it ----------------------------------------

def member_paths(names):
    league = registry.load()
    rows = registry.opponents(league, names)
    return {row['name']: row['path'] for row in rows}


def add_member(name, kind, weights, run, step, parent, target, rule):
    command = [str(VENV_PYTHON), str(LEAGUE), 'add', '--name', name, '--kind', kind,
               '--weights', str(weights), '--run', str(Path(run).relative_to(ROOT)), '--step', str(step),
               '--parent', parent, '--rule', rule]
    if target:
        command += ['--target', target]
    subprocess.run(command, check=True, cwd=ROOT)


def window(run, paths, before):
    """(games, score) against `paths` in the run's ledger since `before`."""
    ledger = (read_json(Path(run) / 'pool/state.json', {}) or {}).get('ledger', {})
    games = wins = draws = 0
    for path in paths:
        row = ledger.get(path, {})
        start = before.get(path, [0, 0, 0])
        games += row.get('games', 0) - start[0]
        wins += row.get('wins', 0) - start[1]
        draws += row.get('draws', 0) - start[2]
    return games, ((wins + draws / 2) / games if games else None)


def ledger_counts(run, paths):
    ledger = (read_json(Path(run) / 'pool/state.json', {}) or {}).get('ledger', {})
    return {path: [ledger.get(path, {}).get(key, 0) for key in ('games', 'wins', 'draws')] for path in paths}


# ---- the loop ----------------------------------------------------------------

class Loop:
    def __init__(self, config):
        self.config = config
        self.dir = config.dir
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / 'logs').mkdir(exist_ok=True)
        (self.dir / 'ratings').mkdir(exist_ok=True)
        self.state_path = self.dir / 'state.json'
        self.state = read_json(self.state_path) or self.fresh_state()
        # Loops started before the style exploiter / evaluation jobs existed.
        self.state.setdefault('style_exploiter', self.fresh_state()['style_exploiter'])
        self.state.setdefault('jobs', self.fresh_state()['jobs'])
        self.stopping = False
        self.rng = random.Random()
        if self.state.get('rng'):
            self.rng.setstate(self._tuple(self.state['rng']))
        else:
            self.rng.seed(zlib.crc32(config.name.encode()))

    @staticmethod
    def _tuple(value):
        return tuple(Loop._tuple(v) for v in value) if isinstance(value, list) else value

    def fresh_state(self):
        return dict(schema=1, name=self.config.name, turn=0,
                    main=dict(run=str(self.dir / 'main'), learner_steps=0, snapshots=[]),
                    main_exploiter=dict(index=0, run=None, trained=0, target=None, last_score=None),
                    league_exploiter=dict(index=0, run=None, trained=0, since_add=0),
                    style_exploiter=dict(index=0, run=None, trained=0, target=None, last_score=None, style=None),
                    members=[], ratings=dict(queue=[], running=None, done=[]), current=None, rng=None,
                    jobs=dict(seq=0, queue=[], running=None, failed=[]))

    def save(self):
        self.state['rng'] = self.rng.getstate()
        atomic_json(self.state_path, self.state)

    def record(self, **row):
        row.update(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        with (self.dir / 'turns.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')

    def summary(self, line):
        """One readable line per event in <dir>/summary.md, for a human report."""
        with (self.dir / 'summary.md').open('a') as handle:
            handle.write(f"- {time.strftime('%m-%d %H:%M')} {line}\n")

    # -- who plays whom

    def mains(self):
        return [self.config.init_member] + self.state['main']['snapshots']

    def league_names(self):
        """Every member this loop's agents see: the seeds, the starting main, and what the loop added.

        With seed_members = "league", the seeds are every member the main's
        architecture can load (registry.opponents refuses the others), re-read
        each turn so members other agents register join at the next turn."""
        names = []
        seeds = self.config.seed_members
        if self.config.seed_league:
            shape = registry.policy_shape(str(self.config.init))
            league = registry.load()
            for member in league['members']:
                if not member.get('path'):
                    continue
                try:
                    registry.opponents(league, [member['name']], policy=shape)
                except (ValueError, FileNotFoundError):
                    continue
                seeds = seeds + [member['name']]
        for name in seeds + [self.config.init_member] + self.state['members']:
            if name not in names:
                names.append(name)
        return names

    def plan(self, agent):
        """The turn's run, its learner-step target, its flags beyond the config, and its opponents."""
        state = self.state
        if agent == 'main':
            main = state['main']
            run = Path(main['run'])
            fresh, extra = self.start_flags('main', run)
            return dict(run=run, fresh=fresh, target=main['learner_steps'] + self.config.main_turn,
                        extra=extra, opponents=self.league_names(), measured=[])
        key = agent.replace('-', '_')
        slot = state[key]
        newest = self.mains()[-1]
        fresh = slot['run'] is None
        if fresh:
            slot['index'] += 1
            slot['run'] = str(self.dir / f"{TAGS[agent]}-{slot['index']}")
            slot['trained'] = 0
            slot['init'] = newest
            if agent == 'league-exploiter':
                slot['since_add'] = 0
            else:
                slot['last_score'] = None
            if agent == 'style-exploiter':
                slot['style'] = self.config.styles[(slot['index'] - 1) % len(self.config.styles)]
            if agent == 'main-exploiter' and self.config.recipes_path:
                # The queue is re-read here so recipes appended since are seen; it cycles when exhausted.
                recipes = self.config.load_recipes()
                order = slot.get('recipe_index', 0)
                slot['recipe'] = recipes[order % len(recipes)]
                slot['recipe_index'] = order + 1
        run = Path(slot['run'])
        if agent in ('main-exploiter', 'style-exploiter'):
            slot['target'] = newest
            opponents = [newest]
            if (not self.config.evals['decide'] and slot.get('last_score') is not None
                    and slot['last_score'] < self.config.hopeless):
                opponents = self.mains()[-3:]
            measured = [newest]
        else:
            opponents = self.league_names()
            measured = opponents
        fresh, extra = self.start_flags(agent, run)
        return dict(run=run, fresh=fresh, target=slot['trained'] + self.config.exploiter_turn,
                    extra=extra, opponents=opponents, measured=measured)

    def start_flags(self, agent, run):
        """(fresh, flags): --resume once the run has saved its pool, else a warm start.

        A run directory without pool/state.json died before its first save;
        train.py will not reuse it, so it is moved aside.
        """
        if (run / 'pool/state.json').exists():
            return False, ['--resume']
        if run.exists():
            run.rename(run.with_name(f'{run.name}.failed-{int(time.time())}'))
        if agent == 'main':
            return True, ['--init-weights', str(self.config.init)]
        if agent == 'league-exploiter' and self.config.league_exploiter_random_init:
            return True, []
        init = self.state[agent.replace('-', '_')]['init']
        return True, ['--init-weights', member_paths([init])[init]]

    # -- running a turn

    def command(self, agent, plan, unit):
        seed = self.config.agent[agent].get('seed', self.config.train.get('seed', 73))
        if agent != 'main':
            seed = int(seed) + 100 * self.state[agent.replace('-', '_')]['index']
        flags = [f for f in self.config.flags(agent, self.recipe_override(agent))]
        if agent == 'style-exploiter':
            flags = self.style_flags(flags)
        if '--seed' in flags:
            flags[flags.index('--seed') + 1] = str(seed)
        else:
            flags += ['--seed', str(seed)]
        memory = (['-p', f'MemoryMax={self.config.memory_max}'] if self.config.memory_max else []) + \
                 (['-p', f'MemoryHigh={self.config.memory_high}'] if self.config.memory_high else []) + \
                 [x for prop in self.config.unit_properties for x in ('-p', prop)]
        return ['systemd-run', '--user', '--scope', '--collect', *memory,
                '--unit', unit, '--', 'nice', '-n', '0', str(VENV_PYTHON), str(TRAIN),
                '--output', str(plan['run']), '--steps', str(plan['target']),
                '--opponents', ','.join(plan['opponents']), *plan['extra'], *flags]

    def recipe_override(self, agent):
        """A main exploiter's recipe as train.py flags: its kit table, learner-first
        (the learner always holds a recipe kit, main the other half), its timeout
        reward and its style program, faded over the exploiter's timeout so the
        fade stays monotonic across --resume turns."""
        recipe = self.state['main_exploiter'].get('recipe') if agent == 'main-exploiter' else None
        if not recipe:
            return {}
        out = dict(recipe.get('train', {}), loadout_set=recipe['loadout_set'], loadout_learner_first=True,
                   reward_timeout=recipe['reward_timeout'])
        if recipe.get('styles'):
            out.update(style=','.join(recipe['styles']),
                       style_fade_steps=int(recipe.get('timeout', self.config.main_exploiter_timeout)))
        return out

    def style_flags(self, flags):
        """The style exploiter's --style, its fade horizon (the timeout, so the fade
        stays monotonic across --resume turns) and, when its specs name kits, a
        lineup table that seats one of them in every game.

        Kits are kept only if main trains on them (the config's --loadout-set):
        main meets a frozen exploiter in its own lineups, so a specialty in any
        other kit would never be tested. Each kept kit plays every kit of that
        set. The learner's side is drawn per match, so it holds the style kit in
        about half its games (all of them in the mirror row); the style's terms
        gate on the abilities they need."""
        slot = self.state['style_exploiter']
        flags = flags + ['--style', ','.join(slot['style']),
                         '--style-fade-steps', str(self.config.style_exploiter_timeout)]
        wanted = [tuple(kit) for spec in slot['style'] for kit in tomllib.loads((ROOT / spec).read_text()).get('kits', [])]
        if not wanted or '--loadout-set' not in flags:
            return flags
        base = []
        for line in (ROOT / flags[flags.index('--loadout-set') + 1]).read_text().splitlines():
            ids = [int(x) for x in line.split('#')[0].replace(',', ' ').split()]
            for kit in (tuple(ids[:3]), tuple(ids[3:6])) if len(ids) == 6 else (tuple(ids),) if len(ids) == 3 else ():
                if kit not in base:
                    base.append(kit)
        kits = [kit for kit in base if any(sorted(kit) == sorted(w) for w in wanted)]
        if not kits:
            log(f"style {slot['style']}: none of its kits {wanted} is in main's lineups; every kit plays")
            return flags
        table = self.dir / 'lineups' / f"{Path(slot['run']).name}.txt"
        table.parent.mkdir(exist_ok=True)
        rows = [','.join(map(str, kit + other)) for kit in kits for other in base]
        table.write_text(f"# {Path(slot['run']).name}: style {','.join(slot['style'])}; kits {kits} vs main's\n"
                         + '\n'.join(rows) + '\n')
        flags[flags.index('--loadout-set') + 1] = str(table)
        return flags

    def environment(self):
        env = dict(os.environ)
        env['PYTHONPATH'] = str(PUFFERLIB_DIR)
        env.setdefault('BOPL_RL_BACKEND', 'rust')
        env.setdefault('BOPL_OMP_THREADS', str(os.cpu_count() or 12))
        env.setdefault('OMP_WAIT_POLICY', 'passive')
        env.setdefault('BOPL_ENV_THREADS', str(max(2, (os.cpu_count() or 12) - 4)))
        return env

    def start_turn(self, agent):
        plan = self.plan(agent)
        paths = member_paths(plan['measured']) if plan['measured'] else {}
        unit = f"bopl-{self.config.name}-{agent}-{self.state['turn']}.scope"
        current = dict(agent=agent, unit=unit, run=str(plan['run']), target=plan['target'],
                       fresh=plan['fresh'], opponents=plan['opponents'], measured=list(paths.values()),
                       before=ledger_counts(plan['run'], paths.values()) if not plan['fresh'] else {},
                       started=time.time(), launches=0)
        self.state['current'] = current
        self.save()
        self.launch(current, plan)

    def launch(self, current, plan=None):
        if plan is None:
            # A relaunch after an interrupted turn: the run exists now, so it resumes.
            fresh, extra = self.start_flags(current['agent'], Path(current['run']))
            plan = dict(run=Path(current['run']), target=current['target'], opponents=current['opponents'],
                        fresh=fresh, extra=extra)
        current['launches'] += 1
        self.save()
        log_path = self.dir / 'logs' / f"{Path(current['run']).name}.log"
        command = self.command(current['agent'], plan, current['unit'])
        log(f"turn {self.state['turn']}: {current['agent']} -> {plan['run'].name} to "
            f"{plan['target'] / 1e9:.2f}B learner ({'fresh' if plan['fresh'] else 'resume'}), "
            f"{len(plan['opponents'])} league opponents")
        with log_path.open('a') as handle:
            handle.write(f"\n# {time.strftime('%F %T')} {' '.join(command)}\n")
            handle.flush()
            subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=self.environment(),
                             cwd=ROOT, start_new_session=True)

    def wait(self, current):
        """Until the turn's unit has come and gone; then its status.json."""
        deadline = time.time() + 120
        while not unit_active(current['unit']) and time.time() < deadline and not self.finished(current):
            time.sleep(2)
        while unit_active(current['unit']):
            if self.stopping:
                subprocess.run(['systemctl', '--user', 'stop', current['unit']], check=False)
            self.tend_ratings()
            self.tend_jobs()
            time.sleep(20)
        return read_json(Path(current['run']) / 'status.json', {})

    def finished(self, current):
        status = read_json(Path(current['run']) / 'status.json', {})
        return status.get('status') == 'complete' and status.get('learner_steps', 0) >= current['target']

    def run_turn(self):
        current = self.state['current']
        if current is None:
            agent = self.config.rotation[self.state['turn'] % len(self.config.rotation)]
            # An exploiter whose last turn is still being evaluated cannot know
            # whether it continues or resets: its next turn waits for the job.
            slot = self.state.get(agent.replace('-', '_'), {})
            if agent != 'main' and slot.get('pending'):
                log(f"turn {self.state['turn']}: {agent} waits for its evaluation {slot['pending']}")
            while agent != 'main' and slot.get('pending') and not self.stopping:
                self.tend_jobs()
                time.sleep(15)
            if self.stopping:
                return
            self.start_turn(agent)
            current = self.state['current']
        elif not unit_active(current['unit']) and not self.finished(current):
            log(f"adopting interrupted turn {self.state['turn']} ({current['agent']}): relaunching")
            self.launch(current)
        else:
            log(f"adopting turn {self.state['turn']} ({current['agent']})")
        status = self.wait(current)
        if self.stopping:
            return
        if not self.finished(current):
            if current['launches'] < 3 and status.get('status') != 'failed':
                log(f"turn {self.state['turn']} ended at {status.get('status')} short of its target; relaunching")
                self.launch(current)
                return self.run_turn()
            notify(f"{self.config.name}: turn {self.state['turn']} ({current['agent']}) failed: "
                   f"{status.get('failure') or status.get('status')}. Loop stopped.")
            raise SystemExit(f"turn failed: {status}")
        getattr(self, 'finish_' + current['agent'].replace('-', '_'))(current, status)
        self.state['current'] = None
        self.state['turn'] += 1
        self.save()

    # -- what a finished turn changes

    def finish_main(self, current, status):
        main = self.state['main']
        main['learner_steps'] = status['learner_steps']
        checkpoint = Path(status['checkpoint'])
        step = int(checkpoint.stem.split('-')[1])
        # A restarted main (state main.restart, set by hand: a new run from the
        # init) tags its snapshots so they cannot collide with the earlier run's.
        name = f"{self.config.name}-main{main.get('restart') or ''}-{step // 1_000_000}m"
        add_member(name, 'main', checkpoint, current['run'], step, parent=self.config.init_member, target=None,
                   rule=f'league loop {self.config.name}: main snapshot every {self.config.main_turn / 1e9:g}B learner '
                        f'(AlphaStar adds main players every 2e9 steps)')
        main['snapshots'].append(name)
        self.state['members'].append(name)
        if self.config.rating_games > 0:   # 0: the [evals] gate replaces the 480-game league rating
            self.state['ratings']['queue'].append(dict(name=name, checkpoint=str(checkpoint)))
        self.record(turn=self.state['turn'], agent='main', run=current['run'], learner_steps=status['learner_steps'],
                    added=name)
        log(f'main at {status["learner_steps"] / 1e9:.2f}B learner: added {name}')
        if self.config.evals['gate_opponents']:
            self.queue_gate(name, checkpoint, status['learner_steps'])
        else:
            self.summary(f"turn {self.state['turn']} main: {name} at {status['learner_steps'] / 1e9:.2f}B learner")
        self.tend_ratings()

    def finish_main_exploiter(self, current, status):
        slot = self.state['main_exploiter']
        slot['trained'] = status['learner_steps']
        if self.config.evals['decide']:
            games, score = window(current['run'], current['measured'], current['before'])
            target = self.mains()[-1]
            self.record(turn=self.state['turn'], agent='main-exploiter', run=current['run'],
                        recipe=slot['recipe']['name'], learner_steps=slot['trained'], target=target,
                        training_games=games, training_score=score)
            self.queue_exploit_eval(slot, Path(status['checkpoint']), target)
            return
        games, score = window(current['run'], current['measured'], current['before'])
        slot['last_score'] = score
        won = score is not None and games >= self.config.min_games and score > self.config.win
        timed_out = slot['trained'] >= self.config.main_exploiter_timeout
        added = None
        if won or timed_out:
            added = self.add_exploiter(slot, status, current, target=slot['target'], agent='main-exploiter',
                                       why=f"{'beat' if won else 'timed out against'} {slot['target']} "
                                           f"({score:.2f} over {games} games)" if score is not None else 'timed out')
            slot['run'] = None   # reset: the next main exploiter starts from the newest main
        self.record(turn=self.state['turn'], agent='main-exploiter', run=current['run'], learner_steps=slot['trained'],
                    target=slot['target'], games=games, score=score, won=won, timed_out=timed_out, added=added)
        log(f"main exploiter {Path(current['run']).name}: {score if score is None else round(score, 3)} vs "
            f"{slot['target']} over {games} games{'; added ' + added if added else ''}")

    def finish_style_exploiter(self, current, status):
        slot = self.state['style_exploiter']
        slot['trained'] = status['learner_steps']
        games, score = window(current['run'], current['measured'], current['before'])
        slot['last_score'] = score
        won = score is not None and games >= self.config.min_games and score > self.config.win
        timed_out = slot['trained'] >= self.config.style_exploiter_timeout
        style = Path(slot['style'][0]).stem
        added = None
        if won or timed_out:
            added = self.add_exploiter(slot, status, current, target=slot['target'], agent='style-exploiter',
                                       why=f"style {','.join(slot['style'])}; "
                                           + (f"{'beat' if won else 'timed out against'} {slot['target']} "
                                              f"({score:.2f} over {games} games)" if score is not None else 'timed out'))
            slot['run'] = None
        self.record(turn=self.state['turn'], agent='style-exploiter', run=current['run'], style=slot['style'],
                    learner_steps=slot['trained'], target=slot['target'], games=games, score=score, won=won,
                    timed_out=timed_out, added=added)
        log(f"style exploiter {Path(current['run']).name} ({style}): {score if score is None else round(score, 3)} vs "
            f"{slot['target']} over {games} games{'; added ' + added if added else ''}")

    def finish_league_exploiter(self, current, status):
        slot = self.state['league_exploiter']
        turn = status['learner_steps'] - slot['trained']
        slot['trained'] = status['learner_steps']
        slot['since_add'] += turn
        if self.config.evals['decide']:
            games, score = window(current['run'], current['measured'], current['before'])
            self.record(turn=self.state['turn'], agent='league-exploiter', run=current['run'],
                        learner_steps=slot['trained'], training_games=games, training_score=score)
            self.queue_league_eval(slot, Path(status['checkpoint']))
            return
        games, score = window(current['run'], current['measured'], current['before'])
        won = score is not None and games >= self.config.min_games and score > self.config.win
        timed_out = slot['since_add'] >= self.config.league_exploiter_timeout
        added, reset = None, False
        if won or timed_out:
            added = self.add_exploiter(slot, status, current, target=None, agent='league-exploiter',
                                       why=f"{'beat' if won else 'timed out against'} the league "
                                           f"({score:.2f} over {games} games)" if score is not None else 'timed out')
            slot['since_add'] = 0
            reset = self.rng.random() < self.config.league_exploiter_reset
            if reset:
                slot['run'] = None
        self.record(turn=self.state['turn'], agent='league-exploiter', run=current['run'],
                    learner_steps=slot['trained'], games=games, score=score, won=won, timed_out=timed_out,
                    added=added, reset=reset)
        log(f"league exploiter {Path(current['run']).name}: {score if score is None else round(score, 3)} vs the "
            f"league over {games} games{'; added ' + added if added else ''}{'; reset' if reset else ''}")

    def add_exploiter(self, slot, status, current, target, agent, why):
        checkpoint = Path(status['checkpoint'])
        step = int(checkpoint.stem.split('-')[1])
        style = f"-{Path(slot['style'][0]).stem.replace('_', '-')}" if agent == 'style-exploiter' else ''
        if agent == 'main-exploiter' and slot.get('recipe'):
            style = f"-{slot['recipe']['name']}"
        name = f"{self.config.name}-{TAGS[agent]}{slot['index']}{style}-{step // 1_000_000}m"
        add_member(name, 'exploiter', checkpoint, current['run'], step, parent=slot['init'], target=target,
                   rule=f"league loop {self.config.name}: {agent.replace('-', ' ')}, {why}")
        self.state['members'].append(name)
        notify(f'{self.config.name}: added {name}, {why}')
        return name

    # -- ratings of each main snapshot, one at a time on the CPU

    def tend_ratings(self):
        ratings = self.state['ratings']
        running = ratings['running']
        if running and not unit_active(running['unit']):
            result = None
            out = self.dir / 'ratings' / f"{running['name']}.jsonl"
            for line in (out.read_text().splitlines() if out.exists() else []):
                if line.startswith('{'):
                    result = json.loads(line)
            ratings['running'] = None
            if result:
                ratings['done'].append(dict(name=running['name'], rating=result['rating'],
                                            interval=result.get('interval'), games=result.get('games'),
                                            kits=result.get('kits')))
                trail = ', '.join(f"{row['rating']:.0f}" for row in ratings['done'][-6:])
                beat = self.config.beat_rating is not None and result['rating'] > self.config.beat_rating
                notify(f"{self.config.name}: {running['name']} rated {result['rating']:.0f} "
                       f"{[round(x) for x in result.get('interval') or []]} (recent mains: {trail})"
                       + (f"; beats {self.config.beat_rating:g}" if beat else ''), 'high' if beat else 'default')
            self.save()
        if ratings['running'] is None and ratings['queue'] and not self.stopping:
            job = ratings['queue'].pop(0)
            unit = f"bopl-{self.config.name}-rate-{job['name']}.service"
            out = self.dir / 'ratings' / f"{job['name']}.jsonl"
            shell = (f"{VENV_PYTHON} {LEAGUE} candidate {job['checkpoint']} --games {self.config.rating_games} "
                     f"--workers {self.config.rating_workers} --source band > {out} 2>> {out}.log")
            subprocess.run(['systemd-run', '--user', '--quiet', '--collect', '--unit', unit, '-p', 'MemoryHigh=6G',
                            '-p', 'CPUWeight=30', '-p', f'WorkingDirectory={ROOT}',
                            '--setenv=BOPL_RL_BACKEND=rust', 'nice', '-n', '10', 'bash', '-c', shell], check=True)
            ratings['running'] = dict(job, unit=unit)
            self.save()

    # -- evaluation jobs ([evals]): one CPU job at a time, in its own capped unit

    def queue_job(self, kind, shell, out, meta, first=False):
        jobs = self.state['jobs']
        jobs['seq'] += 1
        job = dict(id=f"{jobs['seq']}-{kind}", kind=kind, shell=shell, out=str(out), meta=meta, tries=0)
        # A decision job goes ahead of gates: an exploiter's next turn waits on it.
        jobs['queue'].insert(0, job) if first else jobs['queue'].append(job)
        self.save()
        self.tend_jobs()
        return job['id']

    def _python(self, script, *args):
        return ' '.join(shlex.quote(str(x)) for x in (VENV_PYTHON, ROOT / script, *args))

    def _all_combos(self, checkpoint, opponent, label, out, pairs, seed_base):
        return self._python('league/all_combos.py', checkpoint, opponent, '--pairs', pairs,
                            '--workers', self.config.evals['workers'], '--seed-base', seed_base,
                            '--label', label, '--out', out)

    def _opponent_spec(self, spec):
        if spec == 'random':
            return spec
        if (ROOT / spec).is_file() or Path(spec).is_file():
            return str((ROOT / spec).resolve())
        return member_paths([spec])[spec]   # a league member name

    def queue_gate(self, name, checkpoint, learner_steps):
        ev = self.config.evals
        out = self.dir / 'gates' / f'{name}.jsonl'
        out.parent.mkdir(exist_ok=True)
        shell = f'rm -f {shlex.quote(str(out))}; ' + '; '.join(
            self._all_combos(checkpoint, self._opponent_spec(spec), f'gate:{name}:{opp}', out, ev['gate_pairs'],
                             ev['gate_seed_base'])
            for opp, spec in ev['gate_opponents'].items())
        self.queue_job('gate', shell, out, dict(member=name, checkpoint=str(checkpoint), turn=self.state['turn'],
                                                learner_steps=learner_steps, opponents=list(ev['gate_opponents'])))

    def queue_exploit_eval(self, slot, checkpoint, target):
        ev = self.config.evals
        recipe = slot['recipe']
        out = self.dir / 'evals' / f"{Path(slot['run']).name}-{checkpoint.stem.split('-')[1]}"
        main = member_paths([target])[target]
        shell = f'rm -rf {shlex.quote(str(out))}; ' + self._python(
            'eval/exploit_eval.py', '--subject', f'exploiter={checkpoint}', '--subject', f'control={main}',
            '--opponent', f'main={main}', '--schedule', ROOT / recipe['loadout_set'], '--label', recipe['name'],
            '--pairs', ev['exploit_pairs'], '--workers', ev['workers'], '--seed-base', ev['seed_base'], '--out', out)
        slot['pending'] = self.queue_job('exploit', shell, out, dict(
            run=slot['run'], index=slot['index'], checkpoint=str(checkpoint), target=target, trained=slot['trained'],
            recipe=recipe['name'], turn=self.state['turn']), first=True)

    def queue_league_eval(self, slot, checkpoint):
        ev = self.config.evals
        names = self.league_names()
        sample = self.rng.sample(names, min(ev['league_members'], len(names)))
        paths = member_paths(sample)
        out = self.dir / 'evals' / f"{Path(slot['run']).name}-{checkpoint.stem.split('-')[1]}.jsonl"
        out.parent.mkdir(exist_ok=True)
        shell = f'rm -f {shlex.quote(str(out))}; ' + '; '.join(
            self._all_combos(checkpoint, paths[name], f'league:{name}', out, ev['league_pairs'], ev['seed_base'])
            for name in sample)
        slot['pending'] = self.queue_job('league', shell, out, dict(
            run=slot['run'], index=slot['index'], checkpoint=str(checkpoint), members=sample,
            since_add=slot['since_add'], turn=self.state['turn']), first=True)

    def tend_jobs(self):
        jobs = self.state['jobs']
        running = jobs['running']
        if running and not unit_active(running['unit']):
            jobs['running'] = None
            try:
                getattr(self, 'done_' + running['kind'])(running)
            except Exception as error:   # an unreadable result: retried once, then decided as failed
                running['tries'] += 1
                log(f"job {running['id']} gave no result ({error!r}); try {running['tries']}")
                if running['tries'] < 2:
                    jobs['queue'].insert(0, running)
                else:
                    jobs['failed'].append(dict(id=running['id'], error=repr(error)))
                    getattr(self, 'done_' + running['kind'])(running, failed=True)
            self.save()
        if jobs['running'] is None and jobs['queue'] and not self.stopping:
            job = jobs['queue'].pop(0)
            unit = f"bopl-{self.config.name}-job-{job['id']}-{job['tries']}.service"
            log_path = f"{job['out']}.log"
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            env = ['--setenv=BOPL_RL_BACKEND=rust', '--setenv=OMP_NUM_THREADS=1']
            if os.environ.get('BOPL_LEAGUE_JSON'):
                env.append(f"--setenv=BOPL_LEAGUE_JSON={os.environ['BOPL_LEAGUE_JSON']}")
            subprocess.run(['systemd-run', '--user', '--quiet', '--collect', '--unit', unit,
                            '-p', f"MemoryHigh={self.config.evals['memory_high']}", '-p', 'CPUWeight=30',
                            '-p', f'WorkingDirectory={ROOT}',
                            *[x for prop in self.config.evals['unit_properties'] for x in ('-p', prop)],
                            *env, 'nice', '-n', '10', 'bash', '-c',
                            f"({job['shell']}) >> {shlex.quote(log_path)} 2>&1"], check=True)
            jobs['running'] = dict(job, unit=unit)
            log(f"job {job['id']} started ({unit})")
            self.save()

    @staticmethod
    def _rows(path, prefix):
        rows = {}
        for line in Path(path).read_text().splitlines():
            if line.startswith('{'):
                row = json.loads(line)
                if row.get('label', '').startswith(prefix):
                    rows[row['label'][len(prefix):]] = row
        return rows

    def done_gate(self, job, failed=False):
        meta = job['meta']
        rows = {} if failed else self._rows(job['out'], f"gate:{meta['member']}:")
        if not failed and set(rows) != set(meta['opponents']):
            raise ValueError(f"gate rows {sorted(rows)} of {meta['opponents']}")
        results = {opp: dict(score=round(rows[opp]['score'], 3), ci=[round(x, 3) for x in rows[opp]['ci']],
                             games=rows[opp]['games']) for opp in rows}
        self.record(kind='gate', turn=meta['turn'], member=meta['member'], learner_steps=meta['learner_steps'],
                    results=results, failed=failed)
        text = ', '.join(f"vs {opp} {r['score']:.3f} [{r['ci'][0]:.3f},{r['ci'][1]:.3f}]" for opp, r in results.items())
        line = (f"turn {meta['turn']} main: {meta['member']} at {meta['learner_steps'] / 1e9:.2f}B learner; "
                f"all-combos gate {text or 'FAILED'}")
        self.summary(line)
        notify(f'{self.config.name}: {line}')

    @staticmethod
    def paired(path, a='exploiter', b='control', samples=2000):
        """Subject a's score, b's, and a - b per seed pair (both seats), each with a bootstrap 95% interval."""
        import numpy as np
        per = {}
        for line in Path(path).read_text().splitlines():
            g = json.loads(line)
            per.setdefault(g['subject'], {}).setdefault(g['seed'], []).append(g['score'])
        seeds = sorted(set(per.get(a, {})) & set(per.get(b, {})))
        if not seeds:
            raise ValueError(f'no seed pairs played by both {a} and {b}')
        rng = np.random.default_rng(0)
        out = {}
        for key, values in ((a, [np.mean(per[a][s]) for s in seeds]), (b, [np.mean(per[b][s]) for s in seeds]),
                            ('diff', [np.mean(per[a][s]) - np.mean(per[b][s]) for s in seeds])):
            values = np.asarray(values)
            boot = [rng.choice(values, values.size).mean() for _ in range(samples)]
            out[key] = [round(float(values.mean()), 4), round(float(np.percentile(boot, 2.5)), 4),
                        round(float(np.percentile(boot, 97.5)), 4)]
        out['pairs'] = len(seeds)
        return out

    def done_exploit(self, job, failed=False):
        meta = job['meta']
        result = None if failed else self.paired(Path(job['out']) / 'games.jsonl')
        slot = self.state['main_exploiter']
        if slot.get('run') != meta['run']:
            log(f"job {job['id']}: {meta['run']} is no longer the main exploiter; result recorded only")
        recipe = slot.get('recipe') or {}
        margin = self.config.evals['exploit_margin']
        significant = result is not None and result['diff'][1] > margin
        timed_out = meta['trained'] >= int(recipe.get('timeout', self.config.main_exploiter_timeout))
        added = None
        if slot.get('run') == meta['run']:
            if significant:
                d = result['diff']
                added = self.add_exploiter(slot, dict(checkpoint=meta['checkpoint']), dict(run=meta['run']),
                                           target=meta['target'], agent='main-exploiter',
                                           why=f"recipe {meta['recipe']}: {result['exploiter'][0]:.3f} vs "
                                               f"{meta['target']} on its kits, {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] "
                                               f"paired over {meta['target']}'s own weights ({result['pairs']} pairs)")
            if significant or timed_out:
                slot['run'] = None   # reset: the next main exploiter takes the next recipe
            slot['pending'] = None
            slot['last_eval'] = result
        self.record(kind='exploiter-eval', turn=meta['turn'], run=meta['run'], recipe=meta['recipe'],
                    learner_steps=meta['trained'], target=meta['target'], result=result, significant=significant,
                    timed_out=timed_out, added=added, failed=failed)
        if result:
            d = result['diff']
            text = (f"{result['exploiter'][0]:.3f} [{result['exploiter'][1]:.3f},{result['exploiter'][2]:.3f}] vs "
                    f"{meta['target']}, control {result['control'][0]:.3f}, paired {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]")
        else:
            text = 'evaluation FAILED'
        outcome = (f'added {added}' if added else 'reset (timed out)' if timed_out and slot.get('run') is None
                   else 'continues')
        self.summary(f"turn {meta['turn']} main exploiter {Path(meta['run']).name} ({meta['recipe']}, "
                     f"{meta['trained'] / 1e9:.2f}B learner): {text}; {outcome}")

    def done_league(self, job, failed=False):
        import math
        meta = job['meta']
        rows = {} if failed else self._rows(job['out'], 'league:')
        if not failed and set(rows) != set(meta['members']):
            raise ValueError(f"league rows {sorted(rows)} of {meta['members']}")
        slot = self.state['league_exploiter']
        pooled = None
        if rows:
            scores = [rows[n]['score'] for n in meta['members']]
            ses = [(rows[n]['ci'][1] - rows[n]['ci'][0]) / (2 * 1.96) for n in meta['members']]
            mean = sum(scores) / len(scores)
            se = math.sqrt(sum(x * x for x in ses)) / len(ses)
            pooled = [round(mean, 4), round(mean - 1.96 * se, 4), round(mean + 1.96 * se, 4)]
        significant = pooled is not None and pooled[1] > 0.5 + self.config.evals['league_margin']
        timed_out = meta['since_add'] >= self.config.league_exploiter_timeout
        added, reset = None, False
        if slot.get('run') == meta['run']:
            if significant:
                added = self.add_exploiter(slot, dict(checkpoint=meta['checkpoint']), dict(run=meta['run']),
                                           target=None, agent='league-exploiter',
                                           why=f"{pooled[0]:.3f} [{pooled[1]:.3f},{pooled[2]:.3f}] against "
                                               f"{len(rows)} league members on all combos")
                slot['since_add'] = 0
                reset = self.rng.random() < self.config.league_exploiter_reset
            elif timed_out:
                reset = True
            if reset:
                slot['run'] = None
            slot['pending'] = None
        self.record(kind='league-exploiter-eval', turn=meta['turn'], run=meta['run'], members=meta['members'],
                    per_member={n: [round(r['score'], 3), *[round(x, 3) for x in r['ci']]] for n, r in rows.items()},
                    pooled=pooled, significant=significant, timed_out=timed_out, added=added, reset=reset,
                    failed=failed)
        text = (f"{pooled[0]:.3f} [{pooled[1]:.3f},{pooled[2]:.3f}] vs {len(rows)} league members"
                if pooled else 'evaluation FAILED')
        self.summary(f"turn {meta['turn']} league exploiter {Path(meta['run']).name}: {text}; "
                     f"{'added ' + added if added else 'continues' if not reset else 'reset'}"
                     f"{'; reset' if added and reset else ''}")

    # -- main

    def run(self):
        signal.signal(signal.SIGTERM, self._stop)
        signal.signal(signal.SIGINT, self._stop)
        log(f"{self.config.name}: turn {self.state['turn']}, main at {self.state['main']['learner_steps'] / 1e9:.2f}B "
            f"learner of {self.config.main_steps / 1e9:g}B, {len(self.state['members'])} members added so far")
        while not self.stopping:
            if self.state['current'] is None and self.state['main']['learner_steps'] >= self.config.main_steps:
                notify(f"{self.config.name}: main reached {self.config.main_steps / 1e9:g}B learner; loop done")
                break
            self.run_turn()
        # A rating still running is left to finish on its own unit; the next
        # start of the loop collects it.
        while not self.stopping and (self.state['ratings']['running'] or self.state['ratings']['queue']
                                     or self.state['jobs']['running'] or self.state['jobs']['queue']):
            self.tend_ratings()
            self.tend_jobs()
            time.sleep(30)
        log('stopped' if self.stopping else 'done')

    def _stop(self, signum, frame):
        log(f'signal {signum}: stopping after the current turn saves')
        self.stopping = True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='print the next turn\'s command and exit')
    args = parser.parse_args()
    loop = Loop(Config(args.config))
    if args.dry_run:
        agent = loop.config.rotation[loop.state['turn'] % len(loop.config.rotation)]
        state = json.loads(json.dumps(loop.state))
        plan = loop.plan(agent)
        # command() reads the slot plan() may have opened (its index, its style);
        # then the state is put back, so a dry run saves nothing but a style
        # exploiter's lineup table.
        print(' '.join(loop.command(agent, plan, f'bopl-{loop.config.name}-dry')))
        loop.state = state
        return
    loop.run()


if __name__ == '__main__':
    main()
