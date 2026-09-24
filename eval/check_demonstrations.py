"""Prove replay extraction is bit-identical to live policy observations."""
import argparse
from pathlib import Path
import numpy as np
from environment import Environment, configure, OBS_SIZE
from extract_demonstrations import extract
from layout import ACT_AIM, ACT_BUTTONS

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
        # Hold index 0 (length 1): no latch, so the replay (which has no hold head) is exactly reproducible.
        action=np.stack([rng.integers(ACT_AIM,size=2),rng.integers(ACT_BUTTONS,size=2),np.zeros(2,np.int64)],axis=-1)
        actions.append(action)
        env.step(action)
    env.replay(args.output/'oracle.rep')
for player in (1,2):
    data=np.load(extract(args.output/'oracle.rep',args.output/'extracted',player,episode_limit=640))
    np.testing.assert_array_equal(data['observations'],np.asarray(observations)[:,player-1])
    np.testing.assert_array_equal(data['actions'],np.asarray(actions)[:,player-1,:2])  # replay labels: aim, buttons
print(f'REPLAY_POLICY_CONTRACT_OK: {len(actions)} decisions x 2 players x {OBS_SIZE} features',flush=True)
