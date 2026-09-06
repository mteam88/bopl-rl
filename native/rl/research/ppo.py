"""Shared recurrent PPO update, preserving rollout memory and reset ownership."""
import torch
from policy import distribution


def update(policy, optimizer, obs, actions, old_logp, values, returns, adv,
           masks, resets, memories, length, epochs, minibatch,
           entropy_coefficient=.01, action_distribution=distribution):
    tmax, n = obs.shape[:2]
    device = obs.device
    segments=tmax//length
    def pack(x):
        return x.reshape(segments,length,n,*x.shape[2:]).transpose(1,2).reshape(segments*n,length,*x.shape[2:]).transpose(0,1)
    bo,ba,bl,bv,br,bad,bm,bd=map(pack,(obs,actions,old_logp,values,returns,adv,masks,resets))
    initial=torch.cat(list(memories),dim=1)
    indices=torch.where(bm.any(dim=0))[0]
    losses=[]
    for epoch in range(epochs):
        permutation=indices[torch.randperm(len(indices),device=device)]
        for ids in permutation.split(minibatch):
            logits,new_value=policy.sequence(bo[:,ids],initial[:,ids],bd[:,ids])
            _,logp,entropy=action_distribution(logits,ba[:,ids])
            mask=bm[:,ids]
            advantage=bad[:,ids][mask]
            advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
            logratio=(logp-bl[:,ids])[mask]; ratio=logratio.exp()
            pg=torch.maximum(-advantage*ratio,-advantage*ratio.clamp(.8,1.2)).mean()
            value_loss=.5*(new_value[mask]-br[:,ids][mask]).square().mean()
            entropy_loss=entropy[mask].mean()
            loss=pg+.5*value_loss-entropy_coefficient*entropy_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(policy.parameters(),.5)
            if not torch.isfinite(loss) or not torch.isfinite(grad): raise RuntimeError('Nonfinite loss or gradient')
            optimizer.step()
            losses.append([pg.item(),value_loss.item(),entropy_loss.item(),((ratio-1)-logratio).mean().item(),((ratio-1).abs()>.2).float().mean().item(),grad.item()])
    return losses
