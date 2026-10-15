"""The per-bank rollout chain must play like the batched chain (2026-10-15).

Runs train.py twice from one checkpoint at learning rate 0 (the policy never
moves), once with the batched rollout forward and once with BOPL_BATCHED_FWD=0
(the per-bank chain), against the same frozen opponent, and compares what the
learner did: deaths and kills per decision (Poisson), episodes per decision,
and its win share. Fails if any rate differs by more than 4 standard errors.

Before the fix in train/bopl_v4.cu (one cuBLASLt workspace per stream), v4
forwards running concurrently on the buffer and bank streams shared one
workspace, and the per-bank chain's learner died about 2.5x as often as the
batched chain's from the same human-BC init.

usage: check_rollout_paths.py [--init CKPT] [--opponent NAME] [--steps N] [--output DIR]
Needs the GPU (about 11.5 GB for the per-bank run) and the binding built
(train/build.py --device cuda)."""
import argparse, json, math, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'native/artifacts/rl'
p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--init', type=Path, default=R / 'human-bc/human-bc.bin')
p.add_argument('--opponent', default='final5-18666m')
p.add_argument('--steps', type=int, default=3_000_000, help='learner steps per run')
p.add_argument('--output', type=Path, default=R / 'check-rollout-paths')
a = p.parse_args()

def run(name, batched):
    out = a.output / name
    subprocess.run(['rm', '-rf', str(out)], check=True)
    env = dict(os.environ, BOPL_RL_BACKEND='rust', BOPL_BATCHED_FWD=str(batched), BOPL_OMP_THREADS='10',
               OMP_WAIT_POLICY='passive', BOPL_ENV_THREADS='7',
               PYTHONPATH=str(R / 'dependencies/PufferLib-42f70d6932c30ac977736f861006809c50168ba9'))
    cmd = [str(R / 'venv/bin/python'), '-u', 'train/train.py', '--output', str(out / 'run'), '--init-weights', str(a.init),
           '--seed', '311', '--encoder', 'v4', '--hidden', '256', '--layers', '1', '--v4-set-inner', '64', '--v4-ffn-dim', '0',
           '--v4-critic-dim', '384', '--v4-ablate', '0', '--aim-fourier', '4',
           '--workers', '256', '--buffers', '4', '--banks', '4', '--horizon', '128', '--learner-passes', '1', '--minibatch-div', '2',
           '--cudagraphs', '1', '--historical-every', '1', '--opponents', a.opponent, '--bank-groups', 'league,league,league,league',
           '--pool-size', '2', '--pfsp-p', '1', '--loadout-set', 'native/data/lineups-human-exploiter.txt', '--loadout-learner-first',
           '--mirror-share', '0', '--training-maps', 'training-111', '--episode-ticks', '10800', '--reward-kill', '0.2883693237829292',
           '--learning-rate', '0', '--steps', str(a.steps), '--checkpoint-every', '1000000000', '--no-eval']
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'train.log', 'w') as log:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    log = (out / 'train.log').read_text()
    if batched and 'batched rollout forward disabled' in log:
        sys.exit(f'{name}: the batched chain did not engage (see {out}/train.log)')
    tot = {}
    for line in open(out / 'run/metrics.jsonl'):
        for k, v in json.loads(line)['behavior']['totals'].items():
            tot[k] = tot.get(k, 0) + v
    tot['decisions'] = tot['ticks'] / 4
    tot['deaths'] = sum(tot[k] for k in ('deaths_environment', 'deaths_opponent', 'deaths_self', 'deaths_other'))
    return tot

b, s = run('batched', 1), run('per-bank', 0)
worst = 0.0
for k in ('deaths', 'kills', 'episodes'):
    rb, rs = b[k] / b['decisions'], s[k] / s['decisions']
    se = math.sqrt(b[k] / b['decisions'] ** 2 + s[k] / s['decisions'] ** 2) or 1e-12
    z = (rs - rb) / se
    worst = max(worst, abs(z))
    print(f'{k:9s} per 1k decisions: batched {1000 * rb:.3f}  per-bank {1000 * rs:.3f}  z {z:+.1f}')
pb, ps = b['wins'] / b['episodes'], s['wins'] / s['episodes']
pp = (b['wins'] + s['wins']) / (b['episodes'] + s['episodes'])
z = (ps - pb) / math.sqrt(pp * (1 - pp) * (1 / b['episodes'] + 1 / s['episodes']))
worst = max(worst, abs(z))
print(f'win share: batched {pb:.3f}  per-bank {ps:.3f}  z {z:+.1f}')
if worst > 4:
    sys.exit(f'FAIL: the per-bank chain plays differently from the batched chain (|z| {worst:.1f} > 4)')
print(f'ok: the two rollout chains agree (max |z| {worst:.1f})')
