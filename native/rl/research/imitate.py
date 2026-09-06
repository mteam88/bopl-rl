"""Small, auditable behavior-cloning fits with the unchanged recurrent policy.

Splits are whole replay files. The GRU sees each episode from its first decision;
only alive frames contribute to loss. Validation never selects the checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from policy import Policy, previous_actions

def read(paths):
    result=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as data:
            result.append(dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                obs=data['observations'].copy(),actions=data['actions'].copy(),alive=data['alive'].copy(),
                metadata=json.loads(str(data['metadata']))))
    return result

def batch(records,device):
    length=max(len(r['actions']) for r in records)
    obs=np.zeros((length,len(records),records[0]['obs'].shape[-1]),np.float32)
    actions=np.zeros((length,len(records),2),np.int64)
    mask=np.zeros((length,len(records)),bool)
    for i,r in enumerate(records):
        n=len(r['actions']);obs[:n,i]=r['obs'];actions[:n,i]=r['actions'];mask[:n,i]=r['alive']
    return tuple(torch.as_tensor(x,device=device) for x in (obs,actions,mask))

def forward(model,data,change_weights=None):
    obs,actions,mask=data
    resets=torch.zeros(obs.shape[:2],device=obs.device,dtype=torch.bool)
    logits,_=model.sequence(obs,model.initial(obs.shape[1],obs.device),resets)
    a,b=logits.split([256,16],dim=-1)
    losses=[F.cross_entropy(part[mask],actions[...,i][mask],reduction='none') for i,part in enumerate((a,b))]
    if change_weights is None: loss=sum(part.mean() for part in losses)
    else:
        changed=(actions!=previous_actions(obs))[mask]
        weights=1+changed*(change_weights-1)
        loss=sum((part*weights[:,i]).sum()/weights[:,i].sum() for i,part in enumerate(losses))
    pred=torch.stack([a.argmax(-1),b.argmax(-1)],-1)
    return loss,pred

def metrics(pred,actions,mask):
    pred=pred[mask].cpu().numpy();actions=actions[mask].cpu().numpy()
    angle=pred[:,0];target=actions[:,0]
    both=(angle>0)&(target>0)
    delta=np.abs(angle[both]-target[both]);delta=np.minimum(delta,255-delta)*360/255
    return dict(decisions=len(actions),angle_accuracy=float(np.mean(angle==target)),
        buttons_accuracy=float(np.mean(pred[:,1]==actions[:,1])),
        joint_accuracy=float(np.mean(np.all(pred==actions,axis=-1))),
        nonneutral_direction_mae_degrees=float(delta.mean()) if len(delta) else None,
        neutral_accuracy=float(np.mean((angle==0)==(target==0))))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',nargs='+',type=Path,required=True)
    parser.add_argument('--validation',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=1000)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--seed',type=int,default=73)
    parser.add_argument('--balance-changes',action='store_true',help='Equal total training weight for changed and unchanged controls, independently per head')
    parser.add_argument('--action-mode',choices=('absolute','delta'),default='absolute')
    parser.add_argument('--input-mode',choices=('full','zero','previous','world'),default='full')
    args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(args.seed);np.random.seed(args.seed)
    training,validation=read(args.train),read(args.validation)
    train_ids={r['metadata']['replay']['sha256'] for r in training}
    val_ids={r['metadata']['replay']['sha256'] for r in validation}
    if train_ids & val_ids: raise ValueError('Replay leakage between train and validation')
    contracts={(r["metadata"]["repeat"],r["metadata"]["delay"],r["metadata"]["origin"],r["metadata"]["episode_limit"]) for r in training+validation}
    if len(contracts)!=1: raise ValueError("Mixed observation/control contracts")
    control_repeat=next(iter(contracts))[0]
    args.output.mkdir(parents=True,exist_ok=False)
    train_batch,val_batch=batch(training,args.device),batch(validation,args.device)
    model=Policy(train_batch[0].shape[-1],input_mode=args.input_mode,action_mode=args.action_mode).to(args.device)
    change_weights=None
    if args.balance_changes:
        obs,actions,mask=train_batch
        rate=(actions!=previous_actions(obs))[mask].float().mean(0)
        if torch.any((rate==0)|(rate==1)): raise ValueError("Cannot balance a constant control")
        change_weights=(1-rate)/rate
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    manifest=dict(change_weights=change_weights.tolist() if change_weights is not None else None,action_mode=args.action_mode,control_repeat=control_repeat,input_mode=args.input_mode,seed=args.seed,epochs=args.epochs,lr=args.lr,architecture='unchanged Policy: 128 tanh + GRU128',
        train=[{k:v for k,v in r.items() if k not in ('obs','actions','alive')} for r in training],
        validation=[{k:v for k,v in r.items() if k not in ('obs','actions','alive')} for r in validation])
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def save(name,epoch):
        torch.save(dict(policy=model.state_dict(),input_mode=args.input_mode,action_mode=args.action_mode,control_repeat=control_repeat,obs_size=model.obs_size,hidden=model.hidden,
            step=epoch,config=dict(episode_ticks=3600,training_maps=sorted({r['metadata']['replay']['level'] for r in training})),
            imitation_manifest=manifest),args.output/name)
    save('initial.pt',0)
    start=time.monotonic()
    with (args.output/'metrics.jsonl').open('w') as log:
        for epoch in range(args.epochs+1):
            if epoch:
                model.train();optimizer.zero_grad(set_to_none=True)
                loss,_=forward(model,train_batch,change_weights)
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            if epoch%100==0 or epoch==args.epochs:
                model.eval()
                with torch.no_grad():
                    record=dict(epoch=epoch,elapsed_seconds=time.monotonic()-start)
                    for name,data in [('train',train_batch),('validation',val_batch)]:
                        loss,pred=forward(model,data)
                        record[name]=dict(loss=float(loss),**metrics(pred,data[1],data[2]))
                print(json.dumps(record),flush=True);log.write(json.dumps(record)+'\n');log.flush()
    save('final.pt',args.epochs)
    predictions={}
    with torch.no_grad():
        for name,records in [('train',training),('validation',validation)]:
            for i,r in enumerate(records):
                data=batch([r],args.device);_,pred=forward(model,data)
                predictions[f'{name}_{i}_predicted']=pred[:,0].cpu().numpy()
                predictions[f'{name}_{i}_target']=r['actions']
    np.savez_compressed(args.output/'predictions.npz',**predictions)
    # Persistence is a strong baseline for held controls; accuracy alone is not
    # evidence of useful imitation. No future input is used for this baseline.
    baselines={}
    for name,records in [('train',training),('validation',validation)]:
        labels=np.concatenate([r['actions'][r['alive']] for r in records])
        prior=np.concatenate([np.vstack([np.zeros((1,2),dtype=np.int64),r['actions'][:-1]])[r['alive']] for r in records])
        mask=torch.ones(len(labels),dtype=torch.bool)
        baselines[name]=metrics(torch.tensor(prior),torch.tensor(labels),mask)
    (args.output/'result.json').write_text(json.dumps(dict(final=record,persistence_baseline=baselines),indent=2)+'\n')

if __name__=='__main__': main()
