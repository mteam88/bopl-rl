"""Gate 0 of MANY_WORLDS.md: learner decisions/s versus game concurrency, no code change.

Runs the native trainer (train.py) with the 400M null-hypothesis configuration
(--no-eval, ~3M learner decisions) for every (--workers, --buffers) point and
samples game/trainer memory, CPU cores and GPU utilisation from outside.

Facts that size the grid (bopl_puffer.h init/c_reset): the native binding ALWAYS
opens two game processes per worker, the world and a spare that pre-loads the
next episode (there is no BOPL_SPARE_GAME switch in the native path; the spare
is skipped only under no_autoreset, which training never sets). So N workers is
2N Unity processes. RSS per process is ~310 MB but most of it is file-backed
pages shared between processes; the sampler records PSS (proportional set size,
/proc/<pid>/smaps_rollup) so the real per-process cost and the memory ceiling
come from measurement, not from RSS. Points that do not fit are skipped.

Run the driver in its own systemd unit, never from the peace.service cgroup:

  systemd-run --user --unit bopl-gate0 -p MemoryHigh=27G -p MemoryMax=29G \
    --working-directory <checkout> <checkout>/native/artifacts/rl/venv/bin/python \
    native/rl/many_worlds/gate0.py --checkout <checkout> --output <dir>

Game processes are found through the driver's own cgroup (cgroup.procs), so a
smoke test running elsewhere on the machine is neither measured nor killed.
Touch <output>/PAUSE to make the driver wait between points (GPU window for
another thread); remove it to continue.
"""
import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

CLK_TCK = os.sysconf('SC_CLK_TCK')
PAGE = os.sysconf('SC_PAGE_SIZE')
GAME_COMM = 'BoplBattle.x86_'  # /proc comm is truncated to 15 bytes
GB = 1073741824


def cgroup_dir():
    line = Path('/proc/self/cgroup').read_text().strip()
    return Path('/sys/fs/cgroup') / line.split('::', 1)[1].lstrip('/')


def cgroup_pids():
    try:
        return [int(x) for x in (cgroup_dir() / 'cgroup.procs').read_text().split()]
    except FileNotFoundError:
        return []


def cgroup_memory_events():
    try:
        return dict(x.split() for x in (cgroup_dir() / 'memory.events').read_text().splitlines())
    except FileNotFoundError:
        return {}


def mem_available_gb():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024 / GB
    raise RuntimeError('MemAvailable missing')


def comm(pid):
    try:
        return Path(f'/proc/{pid}/comm').read_text().strip()
    except (FileNotFoundError, ProcessLookupError):
        return None


def proc_stat(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
        statm = Path(f'/proc/{pid}/statm').read_text().split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = stat[stat.rindex(')') + 2:].split()
    return int(fields[11]) + int(fields[12]), int(statm[1]) * PAGE / GB


def proc_pss_gb(pid):
    try:
        for line in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines():
            if line.startswith('Pss:'):
                return int(line.split()[1]) * 1024 / GB
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return None


def game_pids():
    return [pid for pid in cgroup_pids() if comm(pid) == GAME_COMM]


def gpu_sample():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5).stdout.strip().split(',')
        return float(out[0]), float(out[1]) / 1024
    except Exception:
        return float('nan'), float('nan')


class Sampler(threading.Thread):
    """Every `interval` s: game count, RSS, CPU cores (games, trainer), GPU; PSS every `pss_every` samples."""

    def __init__(self, trainer_pid, path, interval=2.0, pss_every=5, kill_below_gb=1.5):
        super().__init__(daemon=True)
        self.trainer_pid, self.path, self.interval = trainer_pid, path, interval
        self.pss_every, self.kill_below_gb = pss_every, kill_below_gb
        self.stop, self.oom_guard_fired = threading.Event(), False
        self.samples = []

    def run(self):
        previous, n = None, 0
        with open(self.path, 'w') as out:
            while not self.stop.is_set():
                t = time.monotonic()
                pids = game_pids()
                jiffies_games, rss_games, count = 0, 0.0, 0
                for pid in pids:
                    s = proc_stat(pid)
                    if s:
                        jiffies_games += s[0]; rss_games += s[1]; count += 1
                trainer = proc_stat(self.trainer_pid) or (0, 0.0)
                gpu_util, gpu_mem = gpu_sample()
                avail = mem_available_gb()
                sample = dict(t=t, games=count, rss_games_gb=rss_games, rss_trainer_gb=trainer[1],
                              mem_available_gb=avail, gpu_util=gpu_util, gpu_mem_gb=gpu_mem)
                if n % self.pss_every == 0:
                    pss = [p for p in (proc_pss_gb(pid) for pid in pids) if p is not None]
                    if pss:
                        sample['pss_games_gb'] = sum(pss)
                        sample['pss_game_gb_median'] = statistics.median(pss)
                    tp = proc_pss_gb(self.trainer_pid)
                    if tp is not None:
                        sample['pss_trainer_gb'] = tp
                if previous is not None:
                    dt = t - previous['t']
                    sample['cores_games'] = (jiffies_games - previous['jg']) / CLK_TCK / dt
                    sample['cores_trainer'] = (trainer[0] - previous['jt']) / CLK_TCK / dt
                previous = dict(t=t, jg=jiffies_games, jt=trainer[0])
                self.samples.append(sample)
                out.write(json.dumps(sample) + '\n'); out.flush()
                if avail < self.kill_below_gb and not self.oom_guard_fired:
                    self.oom_guard_fired = True
                    try:
                        os.kill(self.trainer_pid, signal.SIGTERM)  # train.py stops cleanly on SIGTERM
                    except ProcessLookupError:
                        pass
                n += 1
                self.stop.wait(self.interval)


def kill_games():
    pids = game_pids()
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return len(pids)


def train_args(a, workers, buffers, output):
    # Mirrors native/artifacts/rl/null-hypothesis-native/launch-001-s73.sh except
    # steps, eval (off), checkpoint cadence and the swept --workers/--buffers.
    return [str(a.checkout / 'native/artifacts/rl/venv/bin/python'), str(a.checkout / 'native/rl/puffer/train.py'),
            '--output', str(output), '--game', str(a.game),
            '--encoder', 'set', '--hidden', '128', '--layers', '1',
            '--steps', str(a.steps), '--seed', '73', '--workers', str(workers), '--buffers', str(buffers), '--banks', '4',
            '--horizon', '128', '--learner-passes', '1', '--learning-rate', '0.0003', '--anneal-lr', '0', '--min-lr-ratio', '0',
            '--training-maps', '0,1,2', '--evaluation-maps', '0,1,2', '--heldout-maps', '3,4',
            '--episode-ticks', '7200', '--stocks', '0', '--entropy', '0.01', '--entropy-floor', '0.0005',
            '--entropy-half-life', '30000000', '--checkpoint-every', str(a.steps * 10), '--eval-every', str(a.steps * 10),
            '--no-eval']


def summarise(output, samples, oom, returncode, wall, startup_wall, events_before, events_after):
    metrics = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()] \
        if (output / 'metrics.jsonl').exists() else []
    result = dict(updates=len(metrics), returncode=returncode, wall_seconds=wall, startup_seconds=startup_wall,
                  oom_guard=oom,
                  cgroup_memory_high_events=int(events_after.get('high', 0)) - int(events_before.get('high', 0)),
                  cgroup_memory_oom_kills=int(events_after.get('oom_kill', 0)) - int(events_before.get('oom_kill', 0)))
    if metrics:
        final = metrics[-1]
        result['learner_steps'] = final['learner_steps']
        result['matches'] = final['matches']
        result['elapsed_training_seconds'] = final['elapsed']
        result['learner_sps_wall_all'] = final['learner_steps'] / final['elapsed']
        # Steady state: drop the first 20% of learner decisions (first resets,
        # CUDA graph warmup), report the rest as one rate.
        cut = next(i for i, m in enumerate(metrics) if m['learner_steps'] >= 0.2 * final['learner_steps'])
        base = metrics[cut]
        span = final['elapsed'] - base['elapsed']
        if span > 0:
            result['learner_sps_steady'] = (final['learner_steps'] - base['learner_steps']) / span
            result['matches_per_second_steady'] = (final['matches'] - base['matches']) / span
        result['learner_sps_update_median'] = statistics.median(m['learner_sps'] for m in metrics[cut:])
        result['rollout_seconds_median'] = statistics.median(m['rollout_seconds'] for m in metrics[cut:] if 'rollout_seconds' in m)
        result['train_seconds_median'] = statistics.median(m['train_seconds'] for m in metrics[cut:] if 'train_seconds' in m)
        result['steady_window'] = dict(from_update=cut, to_update=len(metrics) - 1)
    if samples:
        steady = [s for s in samples if 'cores_games' in s]
        if len(steady) > 5:
            steady = steady[len(steady) // 5:]
        def agg(key, fn, pool=None):
            # Drop NaN, and negative CPU rates (a sample taken while game
            # processes exit sees the jiffies sum drop).
            values = [s[key] for s in (pool or steady) if key in s and s[key] == s[key] and s[key] >= 0]
            return fn(values) if values else None
        result['game_processes_max'] = max(s['games'] for s in samples)
        result['rss_games_gb_max'] = max(s['rss_games_gb'] for s in samples)
        result['rss_trainer_gb_max'] = max(s['rss_trainer_gb'] for s in samples)
        result['pss_games_gb_max'] = agg('pss_games_gb', max, samples)
        result['pss_game_gb_median'] = agg('pss_game_gb_median', statistics.median, samples)
        result['pss_trainer_gb_max'] = agg('pss_trainer_gb', max, samples)
        result['mem_available_gb_min'] = min(s['mem_available_gb'] for s in samples)
        result['cores_games_mean'] = agg('cores_games', statistics.mean)
        result['cores_games_median'] = agg('cores_games', statistics.median)
        result['cores_trainer_mean'] = agg('cores_trainer', statistics.mean)
        result['cores_trainer_median'] = agg('cores_trainer', statistics.median)
        result['gpu_util_mean'] = agg('gpu_util', statistics.mean)
        result['gpu_util_max'] = agg('gpu_util', max)
        result['gpu_mem_gb_max'] = agg('gpu_mem_gb', max)
    return result


def wait_if_paused(a):
    pause = a.output / 'PAUSE'
    if pause.exists():
        print(json.dumps(dict(paused=True, t=time.time())), flush=True)
        while pause.exists():
            time.sleep(5)
        print(json.dumps(dict(paused=False, t=time.time())), flush=True)


def run_point(a, workers, buffers, per_game_gb):
    name = f'w{workers}-b{buffers}'
    output = a.output / name
    need = workers * 2 * per_game_gb + a.headroom_gb
    avail = mem_available_gb()
    if avail < need:
        return dict(point=name, workers=workers, buffers=buffers, skipped=True,
                    reason=f'MemAvailable {avail:.1f} GB < {need:.1f} GB needed ({workers*2} game processes at {per_game_gb:.3f} GB PSS each)')
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    run = output / 'run'  # train.py creates its own output directory and refuses an existing one
    log = open(output / 'train.log', 'w')
    events_before = cgroup_memory_events()
    started = time.monotonic()
    process = subprocess.Popen(train_args(a, workers, buffers, run), cwd=a.checkout, stdout=log, stderr=subprocess.STDOUT)
    sampler = Sampler(process.pid, output / 'samples.jsonl')
    sampler.start()
    startup_wall = None
    deadline = started + a.timeout
    while process.poll() is None:
        if startup_wall is None and (run / 'metrics.jsonl').exists() and (run / 'metrics.jsonl').stat().st_size:
            startup_wall = time.monotonic() - started
        if time.monotonic() > deadline:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(60)
            except subprocess.TimeoutExpired:
                process.kill()
        time.sleep(1)
    wall = time.monotonic() - started
    sampler.stop.set(); sampler.join()
    log.close()
    time.sleep(2)
    leftover = kill_games()
    result = dict(point=name, workers=workers, buffers=buffers, skipped=False, leftover_games_killed=leftover,
                  **summarise(run, sampler.samples, sampler.oom_guard_fired, process.returncode, wall, startup_wall,
                              events_before, cgroup_memory_events()))
    (output / 'gate0-point.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkout', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--game', type=Path, help='default <checkout>/Builds/LinuxIL2CPP/BoplBattle.x86_64')
    p.add_argument('--steps', type=int, default=3_000_000)
    p.add_argument('--timeout', type=float, default=1500, help='seconds per point')
    p.add_argument('--points', default='16:2,16:1,24:2,32:2,32:1,40:2,48:2,48:1',
                   help='workers:buffers list; workers must be divisible by 2*buffers')
    p.add_argument('--per-game-gb', type=float, default=0.31,
                   help='memory guard: assumed per-process cost until the first point measures PSS')
    p.add_argument('--headroom-gb', type=float, default=3.0)
    a = p.parse_args()
    a.checkout, a.output = a.checkout.resolve(), a.output.resolve()
    a.game = (a.game or a.checkout / 'Builds/LinuxIL2CPP/BoplBattle.x86_64').resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(cgroup=str(cgroup_dir()), mem_available_gb=mem_available_gb(), game=str(a.game))), flush=True)
    results_path = a.output / 'results.jsonl'
    per_game_gb = a.per_game_gb
    for point in a.points.split(','):
        workers, buffers = map(int, point.split(':'))
        wait_if_paused(a)
        result = run_point(a, workers, buffers, per_game_gb)
        with results_path.open('a') as out:
            out.write(json.dumps(result) + '\n')
        print(json.dumps(result), flush=True)
        if result.get('pss_game_gb_median'):
            per_game_gb = result['pss_game_gb_median']
        time.sleep(5)


if __name__ == '__main__':
    main()
