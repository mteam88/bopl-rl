"""Shared recurrent actor/critic for training and replay evaluation."""
import torch
from torch import nn
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, obs_size, hidden=128):
        super().__init__()
        self.obs_size=obs_size; self.hidden=hidden
        self.encoder=nn.Sequential(nn.Linear(obs_size,hidden),nn.Tanh())
        self.memory=nn.GRU(hidden,hidden)
        self.actor=nn.Linear(hidden,272)
        self.critic=nn.Linear(hidden,1)
        nn.init.orthogonal_(self.actor.weight,0.01); nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight,1.0); nn.init.zeros_(self.critic.bias)
    def initial(self,agents,device): return torch.zeros(1,agents,self.hidden,device=device)
    def step(self,obs,state):
        value,state=self.memory(self.encoder(obs).unsqueeze(0),state)
        value=value.squeeze(0)
        return self.actor(value),self.critic(value).squeeze(-1),state
    def sequence(self,obs,state,resets):
        # Reset only the agents that started a new episode. Splitting at the
        # union of boundaries allows cuDNN to process uninterrupted spans.
        encoded=self.encoder(obs)
        boundaries=torch.where(resets.any(dim=1))[0].tolist()
        starts=sorted(set([0,*boundaries,obs.shape[0]]))
        chunks=[]
        for start,end in zip(starts,starts[1:]):
            state=state*(~resets[start]).view(1,-1,1)
            value,state=self.memory(encoded[start:end],state)
            chunks.append(value)
        value=torch.cat(chunks,dim=0)
        return self.actor(value),self.critic(value).squeeze(-1)

def distribution(logits,actions=None):
    angle,buttons=logits.split([256,16],dim=-1)
    a,b=Categorical(logits=angle),Categorical(logits=buttons)
    if actions is None: actions=torch.stack([a.sample(),b.sample()],dim=-1)
    return actions,a.log_prob(actions[...,0])+b.log_prob(actions[...,1]),a.entropy()+b.entropy()

def load(path,device='cpu'):
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    policy=Policy(checkpoint['obs_size'],checkpoint['hidden']).to(device)
    policy.load_state_dict(checkpoint['policy'])
    return policy,checkpoint
