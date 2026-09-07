"""Shared recurrent actor/critic for training and replay evaluation."""
import os
import torch
from torch import nn
from torch.distributions import Categorical

# BOPL_FAST_MULTINOMIAL=1: draw one sample per row with the three kernels torch's
# own n_sample==1 path issues (q ~ Exp(1) per element, argmax(probs/q)), skipping
# only its finiteness / non-negativity / zero-row assert kernels (ten launches per
# call). Softmax outputs are already valid; a non-finite logit surfaces in the
# trainer's finiteness checks instead of a CUDA assert. Same generator advance.
FAST_MULTINOMIAL=os.environ.get('BOPL_FAST_MULTINOMIAL')=='1'

def multinomial1(probs):
    """torch.multinomial(probs,1,True) for a 2-D probability tensor: identical draws."""
    if not FAST_MULTINOMIAL: return torch.multinomial(probs,1,True)
    q=torch.empty_like(probs).exponential_(1)
    torch.div(probs,q,out=q)
    return torch.argmax(q,dim=-1,keepdim=True)

class Policy(nn.Module):
    def __init__(self, obs_size, hidden=128, input_mode="full", action_mode="absolute"):
        super().__init__()
        self.obs_size=obs_size; self.hidden=hidden
        if input_mode not in ("full","zero","previous","world"): raise ValueError(input_mode)
        self.input_mode=input_mode
        if action_mode not in ("absolute","delta"): raise ValueError(action_mode)
        self.action_mode=action_mode
        self.encoder=nn.Sequential(nn.Linear(obs_size,hidden),nn.Tanh())
        self.memory=nn.GRU(hidden,hidden)
        self.actor=nn.Linear(hidden,272)
        self.critic=nn.Linear(hidden,1)
        nn.init.orthogonal_(self.actor.weight,0.01); nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight,1.0); nn.init.zeros_(self.critic.bias)
    def inputs(self,obs):
        if self.input_mode=="full": return obs
        if self.input_mode=="zero": return torch.zeros_like(obs)
        masked=obs.clone()
        if self.input_mode=="previous": masked[...,:-7]=0
        else: masked[...,-7:]=0
        return masked
    def action_logits(self,raw,obs):
        if self.action_mode=="absolute": return raw
        # A permutation, not a new output head: delta zero keeps each control.
        # When the prior stick is neutral, direction codes remain absolute.
        # Otherwise 0..254 rotate on the 255-direction circle; 255 is neutral.
        prior=previous_actions(obs)
        absolute=torch.arange(256,device=obs.device)
        direction=(absolute-prior[...,0,None])%255
        direction=torch.where(absolute==0,255,direction)
        direction=torch.where(prior[...,0,None]==0,absolute,direction)
        buttons=torch.arange(16,device=obs.device)^prior[...,1,None]
        angle_logits,button_logits=raw.split([256,16],-1)
        return torch.cat([angle_logits.gather(-1,direction),button_logits.gather(-1,buttons)],-1)
    def initial(self,agents,device): return torch.zeros(1,agents,self.hidden,device=device)
    def step(self,obs,state):
        value,state=self.memory(self.encoder(self.inputs(obs)).unsqueeze(0),state)
        value=value.squeeze(0)
        return self.action_logits(self.actor(value),obs),self.critic(value).squeeze(-1),state
    def sequence(self,obs,state,resets):
        # Reset only the agents that started a new episode. Splitting at the
        # union of boundaries allows cuDNN to process uninterrupted spans.
        encoded=self.encoder(self.inputs(obs))
        boundaries=torch.where(resets.any(dim=1))[0].tolist()
        starts=sorted(set([0,*boundaries,obs.shape[0]]))
        chunks=[]
        for start,end in zip(starts,starts[1:]):
            state=state*(~resets[start]).view(1,-1,1)
            value,state=self.memory(encoded[start:end],state)
            chunks.append(value)
        value=torch.cat(chunks,dim=0)
        return self.action_logits(self.actor(value),obs),self.critic(value).squeeze(-1)

def previous_actions(obs):
    history=obs[...,-7:]
    angle=(torch.round(torch.remainder(torch.atan2(history[...,2],history[...,1]),2*torch.pi)*255/(2*torch.pi)).long()%255)+1
    angle=torch.where(history[...,0]>.5,0,angle)
    bits=torch.tensor([1,2,4,8],device=obs.device)
    buttons=((history[...,3:]>.5).long()*bits).sum(-1)
    return torch.stack([angle,buttons],-1)

def distribution(logits,actions=None,entropy=True):
    # validate_args=False skips torch's per-call finiteness check, which forces
    # a GPU sync and roughly doubles the cost of each sampling call. Samples,
    # log-probabilities and entropies are computed identically; train.py still
    # checks observations, losses and gradients for finite values.
    angle,buttons=logits.split([256,16],dim=-1)
    a,b=Categorical(logits=angle,validate_args=False),Categorical(logits=buttons,validate_args=False)
    if actions is None:
        if FAST_MULTINOMIAL:
            # Categorical.sample: multinomial over probs reshaped to rows, transposed back.
            actions=torch.stack([multinomial1(a.probs.reshape(-1,256)).T.reshape(angle.shape[:-1]),
                                 multinomial1(b.probs.reshape(-1,16)).T.reshape(buttons.shape[:-1])],dim=-1)
        else: actions=torch.stack([a.sample(),b.sample()],dim=-1)
    logp=a.log_prob(actions[...,0])+b.log_prob(actions[...,1])
    # entropy=False (rollout steppers): the same actions and log-probabilities, without
    # the entropy kernels nobody reads during collection.
    return actions,logp,(a.entropy()+b.entropy()) if entropy else None

def sample(logits):
    """Actions only, drawn with exactly the kernels Categorical(logits).sample()
    issues (normalise by logsumexp, softmax, multinomial with replacement), so
    the draws are bit-identical to distribution(logits)[0] from the same
    generator state. Skips the Distribution objects, log-probabilities and
    entropies that historical opponents never use."""
    angle,buttons=logits.split([256,16],dim=-1)
    angle=angle-angle.logsumexp(dim=-1,keepdim=True); buttons=buttons-buttons.logsumexp(dim=-1,keepdim=True)
    a=multinomial1(torch.softmax(angle,dim=-1).reshape(-1,256)).T.reshape(angle.shape[:-1])
    b=multinomial1(torch.softmax(buttons,dim=-1).reshape(-1,16)).T.reshape(buttons.shape[:-1])
    return torch.stack([a,b],dim=-1)

def load(path,device='cpu'):
    """Load a policy for evaluation: a Torch checkpoint (.pt) or a native
    PufferLib checkpoint (.bin flat float32 weights + <path>.json sidecar,
    see native_policy.py). Both return (policy, metadata) where policy has
    initial()/step()/eval()/obs_size and metadata carries step and config."""
    if str(path).endswith('.bin'):
        from native_policy import load as load_native
        return load_native(path,device)
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    policy=Policy(checkpoint['obs_size'],checkpoint['hidden'],checkpoint.get('input_mode','full'),checkpoint.get('action_mode','absolute')).to(device)
    policy.load_state_dict(checkpoint['policy'])
    return policy,checkpoint
