"""Record a corpus of trained-policy games on the specialist workload as BOPLREP2
archives: generalist-111's final checkpoint against itself, Matthew's 15 kits
cross-paired, the 111 training maps, 3-minute games, delay 9 (the trainer's
condition). The archives are the input of eval/env_bench.c (sim + writer cost on the
state distribution a trained policy actually visits) and, a sample of them,
of the PGO build (native/pgo/corpus.tar.xz).
usage: record_corpus.py <out_dir> <pairs> <workers> [seed_base]"""
import hashlib, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'league'), str(ROOT / 'eval'), str(ROOT / 'train'), str(ROOT / 'train/sweep')]
import registry, games
out_dir, pairs, workers = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
seed_base = int(sys.argv[4]) if len(sys.argv) > 4 else 91001
out_dir.mkdir(parents=True, exist_ok=True)
import os
run = Path(os.environ.get('BOPL_CORPUS_RUN', ROOT / 'native/artifacts/rl/sweeps/generalist-111/000'))   # the policy and its kits
config = json.loads((run / 'config.json').read_text())
checkpoint = json.loads((run / 'status.json').read_text())['checkpoint']
kits = sorted({tuple(r[:3]) for r in config['loadouts']['lineups']})
row = dict(id='g111', name='generalist-111', kind='main', path=checkpoint, loadouts=dict(mode='set', kits=[list(k) for k in kits]))
games.configure(out_dir / 'workspace')
model = games.participant(row)
tasks = games.matchup(row, dict(row, id='g111b'), config['training_maps'], pairs, seed_base)
from environment import Environment
from evaluation_protocol import DEFAULT_LOADOUT

def play(bucket):
    done = []
    with Environment(seed=seed_base, limit=registry.EPISODE_TICKS, delay=registry.DELAY) as env:
        for i, task in bucket:
            env.configure_loadouts(dict(mode='set', fixed=list(DEFAULT_LOADOUT), pool=[], lineups=[task['kits']],
                                        set_file=None, no_dup=False, mirror_share=0.))
            env.reset(task['seed'], task['map'])
            env.bot(0, 0); env.bot(1, 0)
            for side in (0, 1): model.attach(env, side)
            rng = np.random.default_rng((task['seed'], task['map'], task['side']))
            actors = {s: model.start(task, s, rng) for s in (0, 1)}
            while not env.ended:
                obs = env.rows.copy(); mask = env.mask.copy()
                env.step(np.stack([actors[0](obs, mask), actors[1](obs, mask)]))
            name = f"{i:05d}-map{task['map']:03d}-{'-'.join(map(str, task['kits']))}"
            env.replay(str(out_dir / f'{name}.rep'))
            done.append(dict(name=name, map=task['map'], kits=task['kits'], seed=task['seed'], ticks=env.frame()['t']))
    return done

t0 = time.time()
indexed = list(enumerate(tasks))
with ThreadPoolExecutor(workers) as pool:
    results = [g for part in pool.map(play, [indexed[i::workers] for i in range(workers)]) for g in part]
(out_dir / 'index.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in sorted(results, key=lambda r: r['name'])))
print(json.dumps(dict(games=len(results), seconds=round(time.time() - t0), ticks=sum(r['ticks'] for r in results))))
