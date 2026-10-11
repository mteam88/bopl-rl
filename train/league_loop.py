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
"""
import argparse
import json
import os
from pathlib import Path
import random
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
AGENTS = ('main', 'main-exploiter', 'league-exploiter')
# Flags the loop owns: where a turn writes, how long it runs, what it starts
# from and whom it plays.
OWNED = ('output', 'resume', 'steps', 'init_weights', 'opponents')


def log(message):
    print(f'[league-loop {time.strftime("%m-%d %H:%M:%S")}] {message}', flush=True)


def notify(message):
    try:
        subprocess.run(['curl', '-s', '-m', '10', '-H', 'Title: bopl league', '-d', message, NTFY],
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
        self.seed_members = list(loop.get('seed_members', []))
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
        self.rating_games = int(loop.get('rating_games', 480))
        self.rating_workers = int(loop.get('rating_workers', 4))
        self.train = dict(raw.get('train', {}))
        self.agent = {agent: dict(raw.get(agent, {})) for agent in AGENTS}
        if any(agent not in AGENTS for agent in self.rotation) or 'main' not in self.rotation:
            raise SystemExit(f'rotation must name agents from {AGENTS} and include main')
        for table in [self.train, *self.agent.values()]:
            for key in table:
                if key in OWNED:
                    raise SystemExit(f'{key} is set by the loop, not the config')
                if key not in sweep_config.TRAIN_FLAGS:
                    raise SystemExit(f'{key} is not a train.py flag')

    def flags(self, agent):
        merged = dict(self.train)
        merged.update(self.agent[agent])
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
                    members=[], ratings=dict(queue=[], running=None, done=[]), current=None, rng=None)

    def save(self):
        self.state['rng'] = self.rng.getstate()
        atomic_json(self.state_path, self.state)

    def record(self, **row):
        row.update(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        with (self.dir / 'turns.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')

    # -- who plays whom

    def mains(self):
        return [self.config.init_member] + self.state['main']['snapshots']

    def league_names(self):
        """Every member this loop's agents see: the seeds, the starting main, and what the loop added."""
        names = []
        for name in self.config.seed_members + [self.config.init_member] + self.state['members']:
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
            slot['run'] = str(self.dir / f"{'me' if agent == 'main-exploiter' else 'le'}-{slot['index']}")
            slot['trained'] = 0
            slot['init'] = newest
            if agent == 'league-exploiter':
                slot['since_add'] = 0
            else:
                slot['last_score'] = None
        run = Path(slot['run'])
        if agent == 'main-exploiter':
            slot['target'] = newest
            opponents = [newest]
            if slot.get('last_score') is not None and slot['last_score'] < self.config.hopeless:
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
        init = self.state[agent.replace('-', '_')]['init']
        return True, ['--init-weights', member_paths([init])[init]]

    # -- running a turn

    def command(self, agent, plan, unit):
        seed = self.config.agent[agent].get('seed', self.config.train.get('seed', 73))
        if agent != 'main':
            seed = int(seed) + 100 * self.state[agent.replace('-', '_')]['index']
        flags = [f for f in self.config.flags(agent)]
        if '--seed' in flags:
            flags[flags.index('--seed') + 1] = str(seed)
        else:
            flags += ['--seed', str(seed)]
        return ['systemd-run', '--user', '--scope', '--collect', '-p', f'MemoryMax={self.config.memory_max}',
                '--unit', unit, '--', 'nice', '-n', '0', str(VENV_PYTHON), str(TRAIN),
                '--output', str(plan['run']), '--steps', str(plan['target']),
                '--opponents', ','.join(plan['opponents']), *plan['extra'], *flags]

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
            time.sleep(20)
        return read_json(Path(current['run']) / 'status.json', {})

    def finished(self, current):
        status = read_json(Path(current['run']) / 'status.json', {})
        return status.get('status') == 'complete' and status.get('learner_steps', 0) >= current['target']

    def run_turn(self):
        current = self.state['current']
        if current is None:
            agent = self.config.rotation[self.state['turn'] % len(self.config.rotation)]
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
        name = f'{self.config.name}-main-{step // 1_000_000}m'
        add_member(name, 'main', checkpoint, current['run'], step, parent=self.config.init_member, target=None,
                   rule=f'league loop {self.config.name}: main snapshot every {self.config.main_turn / 1e9:g}B learner '
                        f'(AlphaStar adds main players every 2e9 steps)')
        main['snapshots'].append(name)
        self.state['members'].append(name)
        self.state['ratings']['queue'].append(dict(name=name, checkpoint=str(checkpoint)))
        self.record(turn=self.state['turn'], agent='main', run=current['run'], learner_steps=status['learner_steps'],
                    added=name)
        log(f'main at {status["learner_steps"] / 1e9:.2f}B learner: added {name}')
        self.tend_ratings()

    def finish_main_exploiter(self, current, status):
        slot = self.state['main_exploiter']
        slot['trained'] = status['learner_steps']
        games, score = window(current['run'], current['measured'], current['before'])
        slot['last_score'] = score
        won = score is not None and games >= self.config.min_games and score > self.config.win
        timed_out = slot['trained'] >= self.config.main_exploiter_timeout
        added = None
        if won or timed_out:
            added = self.add_exploiter(slot, status, current, target=slot['target'],
                                       why=f"{'beat' if won else 'timed out against'} {slot['target']} "
                                           f"({score:.2f} over {games} games)" if score is not None else 'timed out')
            slot['run'] = None   # reset: the next main exploiter starts from the newest main
        self.record(turn=self.state['turn'], agent='main-exploiter', run=current['run'], learner_steps=slot['trained'],
                    target=slot['target'], games=games, score=score, won=won, timed_out=timed_out, added=added)
        log(f"main exploiter {Path(current['run']).name}: {score if score is None else round(score, 3)} vs "
            f"{slot['target']} over {games} games{'; added ' + added if added else ''}")

    def finish_league_exploiter(self, current, status):
        slot = self.state['league_exploiter']
        turn = status['learner_steps'] - slot['trained']
        slot['trained'] = status['learner_steps']
        slot['since_add'] += turn
        games, score = window(current['run'], current['measured'], current['before'])
        won = score is not None and games >= self.config.min_games and score > self.config.win
        timed_out = slot['since_add'] >= self.config.league_exploiter_timeout
        added, reset = None, False
        if won or timed_out:
            added = self.add_exploiter(slot, status, current, target=None,
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

    def add_exploiter(self, slot, status, current, target, why):
        checkpoint = Path(status['checkpoint'])
        step = int(checkpoint.stem.split('-')[1])
        tag = 'me' if target else 'le'
        name = f"{self.config.name}-{tag}{slot['index']}-{step // 1_000_000}m"
        add_member(name, 'exploiter', checkpoint, current['run'], step, parent=slot['init'], target=target,
                   rule=f"league loop {self.config.name}: {'main' if target else 'league'} exploiter, {why}")
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
                notify(f"{self.config.name}: {running['name']} rated {result['rating']:.0f} "
                       f"{[round(x) for x in result.get('interval') or []]} (recent mains: {trail})")
            self.save()
        if ratings['running'] is None and ratings['queue'] and not self.stopping:
            job = ratings['queue'].pop(0)
            unit = f"bopl-{self.config.name}-rate-{job['name']}.service"
            out = self.dir / 'ratings' / f"{job['name']}.jsonl"
            shell = (f"{VENV_PYTHON} {LEAGUE} candidate {job['checkpoint']} --games {self.config.rating_games} "
                     f"--workers {self.config.rating_workers} --source band > {out} 2>> {out}.log")
            subprocess.run(['systemd-run', '--user', '--collect', '--unit', unit, '-p', 'MemoryHigh=6G',
                            '-p', 'CPUWeight=30', '-p', f'WorkingDirectory={ROOT}',
                            '--setenv=BOPL_RL_BACKEND=rust', 'nice', '-n', '10', 'bash', '-c', shell], check=True)
            ratings['running'] = dict(job, unit=unit)
            self.save()

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
        while not self.stopping and (self.state['ratings']['running'] or self.state['ratings']['queue']):
            self.tend_ratings()
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
        loop.state = state   # plan() may open an exploiter slot; a dry run changes nothing
        print(' '.join(loop.command(agent, plan, f'bopl-{loop.config.name}-dry')))
        return
    loop.run()


if __name__ == '__main__':
    main()
