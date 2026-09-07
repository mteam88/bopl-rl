"""Check action expansion, causal script behavior and masked PPO updates."""
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
from initialize_league import expand
from policy import Policy
from ppo import update
from scripted_opponents import BotSpec, ScriptedOpponent, SELF, OTHER


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(101)
    model=Policy(3257)
    hidden=torch.randn(64,128)
    original=model.actor(hidden)
    expand(model,.001)
    expanded=model.actor(hidden)
    probabilities=expanded[:,256:].softmax(-1)
    torch.testing.assert_close(expanded[:,:256],original[:,:256],rtol=0,atol=0)
    torch.testing.assert_close(probabilities[:,[4,5,6,7,12,13,14,15]].sum(-1),
                               original[:,[256,260]].softmax(-1)[:,1],rtol=1e-5,atol=1e-7)
    for bit in (1,2,8):
        actual=probabilities[:,[b for b in range(16) if b&bit]].sum(-1)
        torch.testing.assert_close(actual,torch.full_like(actual,.001),rtol=1e-5,atol=1e-7)
    assert (probabilities>0).all()
    probabilities.sum().backward()
    assert model.actor.weight.requires_grad
    o=np.zeros(3257,dtype=np.float32)
    o[SELF+3]=o[OTHER+3]=1;o[OTHER+9]=.25
    for style in ('archer','pursuer','evasive','switcher'):
        spec=BotSpec(style,style)
        a=ScriptedOpponent(spec,99);b=ScriptedOpponent(spec,99)
        for t in range(200):
            x=o.copy();x[SELF+16+44+42]=1
            if t==80:x[SELF+8]=.1
            left=a.act(x);right=b.act(x)
            np.testing.assert_array_equal(left,right)
            assert 0<=left[0]<=255 and 0<=left[1]<=15
    # Cooldown > threshold alone is not a projectile threat.
    calm=o.copy();calm[OTHER+16+44+43]=1
    bot=ScriptedOpponent(BotSpec('e','evasive',distance=10),101)
    assert bot.act(calm)[0]==1
    # Visible enemy projectile travelling toward us produces retreat.
    threat=o.copy();threat[898]=1;threat[898+8]=8/64;threat[898+10]=-40/256;threat[898+20]=9/32
    bot=ScriptedOpponent(BotSpec('e','evasive',distance=10),101)
    assert bot.act(threat)[0] in (128,129)
    print('FULL_ACTION_MARGINALS_AND_NONZERO_SUPPORT; SCRIPT_REPRODUCIBILITY_AND_THREATS_OK')
    if args.rollout:
        from train import advantages
        r=torch.load(args.rollout,map_location='cpu',weights_only=False)
        c=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
        a=Policy(c['obs_size'],c['hidden']);a.load_state_dict(c['policy']);b=copy.deepcopy(a)
        ao=torch.optim.Adam(a.parameters(),lr=3e-4,eps=1e-5)
        bo=torch.optim.Adam(b.parameters(),lr=3e-4,eps=1e-5)
        # Independent targets isolate the optimizer's exclusion of opponents.
        adv=torch.randn_like(r['rewards']);returns=adv+r['values']
        changed=r['actions'].clone();changed[~r['masks']]=torch.tensor([255,15])
        values=[]
        for model,optimizer,actions in ((a,ao,r['actions']),(b,bo,changed)):
            torch.manual_seed(918)
            values.append(update(model,optimizer,r['obs'],actions,r['old_logp'],r['values'],returns,adv,
                                 r['masks'],r['resets'],r['memories'],32,1,8))
        np.testing.assert_array_equal(values[0],values[1])
        for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
        assert r['masks'].sum()==r['masks'].numel()*.75
        print('OPPONENT_ACTIONS_EXCLUDED_FROM_PPO_BIT_EXACT; LEARNER_FRACTION_075')


if __name__=='__main__':main()
