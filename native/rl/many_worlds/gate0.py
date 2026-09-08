"""Gate 0 of MANY_WORLDS.md: learner decisions/s versus game concurrency, no code change.

Runs the native trainer (train.py) with the 400M null-hypothesis configuration
(--no-eval, ~3M learner decisions) for every (--workers, --buffers) point and
samples game/trainer RSS, CPU cores and GPU utilisation from outside. Points
that cannot fit in memory are skipped, not attempted: every training worker
holds two game processes (world + spare), ~310 MB RSS each.

Usage: python gate0.py --checkout <ExportedProject> --output <dir> [--points 16:2,16:1,...]
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
GAME_COMM = 'BoplBattle.x86_'  # /proc comm is truncated to 15 bytes
PER_GAME_GB = 0.34
HEADROOM_GB = 3.0


def mem_available_gb():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1048576
    raise RuntimeError('MemAvailable missing')


def proc_stat(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
        statm = Path(f'/proc/{pid}/statm').read_text().split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = stat[stat.rindex(')') + 2:].split()
    utime, stime = int(fields[11]), int(fields[12])
    rss_gb = int(statm[1]) * os.sysconf('SC_PAGE_SIZE') / 1073741824
    return utime + stime, rss_gb


def game_pids():
    pids = []
    for entry in os.scandir('/proc'):
        if not entry.name.isdigit():
            continue
        try:
            if Path(entry.path, 'comm').read_text().strip() == GAME_COMM:
                pids.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            pass
    return pids


def gpu_sample():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5).stdout.strip().split(',')
        return float(out[0]), float(out[1]) / 1024
    except Exception:
        return float('nan'), float('nan')


class Sampler(threading.Thread):
    def __init__(self, trainer_pid, path, interval=2.0, kill_below_gb=1.5):
        super().__init__(daemon=True)
        self.trainer_pid, self.path, self.interval, self.kill_below_gb = trainer_pid, path, interval, kill_below_gb
        self.stop, self.oom_guard_fired = threading.Event(), False
        self.samples = []

    def run(self):
        previous = None
        with open(self.path, 'w') as out:
            while not self.stop.is_set():
                t = time.monotonic()
                jiffies_games, rss_games, count = 0, 0.0, 0
                for pid in game_pids():
                    s = proc_stat(pid)
                    if s:
                        jiffies_games += s[0]; rss_games += s[1]; count += 1
                # Trainer = the python process and its threads (stat is process-wide).
                trainer = proc_stat(self.trainer_pid) or (0, 0.0)
                gpu_util, gpu_mem = gpu_sample()
                avail = mem_available_gb()
                sample = dict(t=t, games=count, rss_games_gb=rss_games, rss_trainer_gb=trainer[1],
                              mem_available_gb=avail, gpu_util=gpu_util, gpu_mem_gb=gpu_mem)
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
                        os.kill(self.trainer_pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                self.stop.wait(self.interval)


def kill_games():
    for pid in game_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def train_args(a, workers, buffers, output):
    return [str(a.checkout / 'native/artifacts/rl/venv/bin/python'), str(a.checkout / 'native/rl/puffer/train.py'),
            '--output', str(output), '--game', str(a.checkout / 'Builds/LinuxIL2CPP/BoplBattle.x86_64'),
            '--encoder', 'set', '--hidden', '128', '--layers', '1',
            '--steps', str(a.steps), '--seed', '73', '--workers', str(workers), '--buffers', str(buffers), '--banks', '4',
            '--horizon', '128', '--learner-passes', '1', '--learning-rate', '0.0003', '--anneal-lr', '0', '--min-lr-ratio', '0',
            '--training-maps', '0,1,2', '--evaluation-maps', '0,1,2', '--heldout-maps', '3,4',
            '--episode-ticks', '7200', '--stocks', '0', '--entropy', '0.01', '--entropy-floor', '0.0005',
            '--entropy-half-life', '30000000', '--checkpoint-every', str(a.steps * 10), '--eval-every', str(a.steps * 10),
            '--no-eval']


def summarise(output, samples, oom, returncode, wall, startup_wall):
    metrics = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()] \
        if (output / 'metrics.jsonl').exists() else []
    result = dict(updates=len(metrics), returncode=returncode, wall_seconds=wall, startup_seconds=startup_wall, oom_guard=oom)
    if metrics:
        final = metrics[-1]
        result['learner_steps'] = final['learner_steps']
        result['matches'] = final['matches']
        result['elapsed_training_seconds'] = final['elapsed']
        result['learner_sps_wall_all'] = final['learner_steps'] / final['elapsed']
        # Steady state: drop the first 20% of learner decisions (process boot, first
        # resets, CUDA graph warmup), report the rest as one rate.
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
        def agg(key, fn):
            values = [s[key] for s in steady if key in s and s[key] == s[key]]
            return fn(values) if values else None
        result['game_processes_max'] = max(s['games'] for s in samples)
        result['rss_games_gb_max'] = max(s['rss_games_gb'] for s in samples)
        result['rss_trainer_gb_max'] = max(s['rss_trainer_gb'] for s in samples)
        result['mem_available_gb_min'] = min(s['mem_available_gb'] for s in samples)
        result['cores_games_mean'] = agg('cores_games', statistics.mean)
        result['cores_trainer_mean'] = agg('cores_trainer', statistics.mean)
        result['gpu_util_mean'] = agg('gpu_util', statistics.mean)
        result['gpu_util_max'] = agg('gpu_util', max)
        result['gpu_mem_gb_max'] = agg('gpu_mem_gb', max)
    return result


def run_point(a, workers, buffers):
    name = f'w{workers}-b{buffers}'
    output = a.output / name
    need = workers * 2 * PER_GAME_GB + HEADROOM_GB
    avail = mem_available_gb()
    if avail < need:
        return dict(point=name, workers=workers, buffers=buffers, skipped=True,
                    reason=f'MemAvailable {avail:.1f} GB < {need:.1f} GB needed ({workers*2} game processes)')
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    log = open(output / 'train.log', 'w')
    started = time.monotonic()
    process = subprocess.Popen(train_args(a, workers, buffers, output), cwd=a.checkout, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    sampler = Sampler(process.pid, output / 'samples.jsonl')
    sampler.start()
    startup_wall = None
    deadline = started + a.timeout
    while process.poll() is None:
        if startup_wall is None and (output / 'metrics.jsonl').exists() and (output / 'metrics.jsonl').stat().st_size:
            startup_wall = time.monotonic() - started
        if time.monotonic() > deadline:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(30)
            except subprocess.TimeoutExpired:
                process.kill()
        time.sleep(1)
    wall = time.monotonic() - started
    sampler.stop.set(); sampler.join()
    log.close()
    time.sleep(2)
    leftover = game_pids()
    if leftover:
        kill_games()
    result = dict(point=name, workers=workers, buffers=buffers, skipped=False, leftover_games_killed=len(leftover),
                  **summarise(output, sampler.samples, sampler.oom_guard_fired, process.returncode, wall, startup_wall))
    (output / 'gate0-point.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkout', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=3_000_000)
    p.add_argument('--timeout', type=float, default=1500, help='seconds per point')
    p.add_argument('--points', default='16:2,16:1,32:1,32:2,48:1,48:2,64:1,64:2')
    a = p.parse_args()
    a.checkout, a.output = a.checkout.resolve(), a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=True)
    results_path = a.output / 'results.jsonl'
    for point in a.points.split(','):
        workers, buffers = map(int, point.split(':'))
        result = run_point(a, workers, buffers)
        with results_path.open('a') as out:
            out.write(json.dumps(result) + '\n')
        print(json.dumps(result), flush=True)
        time.sleep(5)


if __name__ == '__main__':
    main()
