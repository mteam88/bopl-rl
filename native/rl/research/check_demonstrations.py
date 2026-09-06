"""Prove replay extraction is bit-identical to live policy observations."""
import argparse
from pathlib import Path
import numpy as np
from environment import Environment, configure
from extract_demonstrations import extract

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=False)
configure(args.output/'workers')
observations, actions = [], []
rng=np.random.default_rng(171)
with Environment(seed=171,limit=640) as env:
    while not env.ended:
        observations.append(env.obs.copy())
        action=np.stack([rng.integers(256,size=2),rng.integers(16,size=2)],axis=-1)
        actions.append(action)
        env.step(action)
    env.replay(args.output/'oracle.rep')
for player in (1,2):
    data=np.load(extract(args.output/'oracle.rep',args.output/'extracted',player,episode_limit=640))
    np.testing.assert_array_equal(data['observations'],np.asarray(observations)[:,player-1])
    np.testing.assert_array_equal(data['actions'],np.asarray(actions)[:,player-1])
print(f'REPLAY_POLICY_CONTRACT_OK: {len(actions)} decisions x 2 players x 3257 features',flush=True)
