"""Score frozen imitation weights on complete, disjoint human recordings."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from imitate import read, batch, forward, metrics
from policy import load

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('demonstrations',type=Path,nargs='+')
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--examples',action='store_true',help='Save exact network input and prediction arrays for visual inspection')
args=parser.parse_args();torch.set_num_threads(1)
model,checkpoint=load(args.checkpoint);model.eval()
trained={r['metadata']['replay']['sha256'] for r in checkpoint['imitation_manifest']['train']}
records=read(args.demonstrations)
if any(r["metadata"]["repeat"]!=checkpoint.get("control_repeat",4) for r in records):
    raise ValueError("Dataset and checkpoint control rates differ")
if any(r['metadata']['replay']['sha256'] in trained for r in records):
    raise ValueError('A scoring replay was used to train this checkpoint')
if args.output.exists():raise FileExistsError(args.output)
predictions=[];targets=[];priors=[];details=[];loss_sum=0;count=0
with torch.no_grad():
    for record in records:
        data=batch([record],'cpu');loss,pred=forward(model,data);mask=data[2]
        n=int(mask.sum());count+=n;loss_sum+=float(loss)*n
        if args.examples:
            example=args.output.parent/(args.output.stem+'-'+Path(record['path']).stem+'.npz')
            example.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(example,observations=record['obs'],inputs=model.inputs(data[0])[:,0].numpy(),
                target=record['actions'],predicted=pred[:,0].numpy(),metadata=json.dumps(record['metadata']))
        predictions.append(pred[mask]);targets.append(data[1][mask])
        prior=torch.from_numpy(np.vstack([np.zeros((1,2),dtype=np.int64),record['actions'][:-1]]))
        priors.append(prior[record['alive']])
        details.append(dict(file=record['path'],sha256=record['sha256'],replay_sha256=record['metadata']['replay']['sha256'],
            map=record['metadata']['replay']['level'],loss=float(loss),**metrics(pred,data[1],mask)))
labels=torch.cat(targets);mask=torch.ones(len(labels),dtype=torch.bool)
report=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    matches=len(records),model=dict(loss=loss_sum/count,**metrics(torch.cat(predictions),labels,mask)),
    persistence=metrics(torch.cat(priors),labels,mask),per_match=details)
prediction=torch.cat(predictions);prior=torch.cat(priors)
events={}
if all(r['metadata']['replay']['players'][r['metadata']['player']-1]['abilities']==[4,14,2] for r in records):
    for name,bit in [('jump',0),('bow',1),('platform',2),('dash',3)]:
        before=(prior[:,1]&(1<<bit))!=0;target=(labels[:,1]&(1<<bit))!=0;chosen=(prediction[:,1]&(1<<bit))!=0
        for event,condition in [('press',~before),('release',before)]:
            actual=condition&(target!=before);predicted=condition&(chosen!=before)
            tp=int((actual&predicted).sum());positives=int(actual.sum());called=int(predicted.sum())
            events[name+'_'+event]=dict(actual=positives,predicted=called,true_positive=tp,
                recall=tp/positives if positives else None,precision=tp/called if called else None)
report['events']=events
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='per_match'}),flush=True)
