"""Check sequence causality, batch isolation and saved control-rate alignment."""
import argparse
from pathlib import Path
import numpy as np
import torch
from policy import Policy

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fifteen',type=Path)
parser.add_argument('--thirty',type=Path)
args=parser.parse_args();torch.set_num_threads(1);torch.manual_seed(73)
model=Policy(3257)
obs=torch.randn(7,2,3257,requires_grad=True)
resets=torch.zeros(7,2,dtype=torch.bool);resets[4,1]=True
logits,_=model.sequence(obs,model.initial(2,'cpu'),resets)
logits[3,0].sum().backward()
assert torch.count_nonzero(obs.grad[4:])==0, 'Future observation leak'
assert torch.count_nonzero(obs.grad[:,1])==0, 'Batch/agent mixing'
assert torch.count_nonzero(obs.grad[:4,0])>0, 'No causal dependence'
state=model.initial(2,'cpu');steps=[]
with torch.no_grad():
    for t in range(7):
        state=state*(~resets[t]).view(1,-1,1)
        result,_,state=model.step(obs[t],state);steps.append(result)
    torch.testing.assert_close(torch.stack(steps),logits,atol=1e-6,rtol=1e-5)
print('CAUSAL_GRADIENT_AND_SEQUENCE_STEP_OK')
if args.fifteen or args.thirty:
    if not args.fifteen or not args.thirty:parser.error('Supply both datasets')
    checked=0
    for fifteen in args.fifteen.glob('*.npz'):
        thirty=args.thirty/fifteen.name
        if not thirty.exists():continue
        with np.load(fifteen) as a,np.load(thirty) as b:
            np.testing.assert_array_equal(a['ticks'],b['ticks'][::2])
            np.testing.assert_array_equal(a['actions'],b['actions'][::2])
            # The world remains nine ticks delayed on the shared time grid.
            np.testing.assert_array_equal(a['observations'][:,:-7],b['observations'][::2,:-7])
            for data in (a,b):
                prior=np.vstack([np.zeros((1,2),np.int64),data['actions'][:-1]])
                history=data['observations'][:,-7:]
                np.testing.assert_array_equal(history[:,0],prior[:,0]==0)
                for bit in range(4):np.testing.assert_array_equal(history[:,3+bit],(prior[:,1]&(1<<bit))!=0)
            checked+=1
    if not checked:raise AssertionError('No matching datasets')
    print(f'CONTROL_RATE_ALIGNMENT_OK: {checked} complete replay pairs')

from replay_data import previous
from policy import previous_actions
history=np.zeros((256,3257),np.float32)
controls=np.stack([np.arange(256),np.arange(256)%16],-1)
previous(history,controls)
np.testing.assert_array_equal(previous_actions(torch.from_numpy(history)).numpy(),controls)
delta=Policy(3257,action_mode='delta')
raw=torch.zeros(256,272);raw[:,0]=10;raw[:,256]=10
mapped=delta.action_logits(raw,torch.from_numpy(history))
np.testing.assert_array_equal(torch.stack([x.argmax(-1) for x in mapped.split([256,16],-1)],-1).numpy(),controls)
raw=torch.arange(272).float().expand(256,-1)
mapped=delta.action_logits(raw,torch.from_numpy(history))
for original,permuted in zip(raw.split([256,16],-1),mapped.split([256,16],-1)):
    torch.testing.assert_close(permuted.sort(-1).values,original)
print('PREVIOUS_INPUT_DECODING_AND_DELTA_BIJECTION_OK: all 256 stick codes and 16 button states')
