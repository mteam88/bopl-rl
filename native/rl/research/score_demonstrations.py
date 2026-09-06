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
args=parser.parse_args();torch.set_num_threads(1)
model,checkpoint=load(args.checkpoint);model.eval()
trained={r['metadata']['replay']['sha256'] for r in checkpoint['imitation_manifest']['train']}
records=read(args.demonstrations)
if any(r['metadata']['replay']['sha256'] in trained for r in records):
    raise ValueError('A scoring replay was used to train this checkpoint')
if args.output.exists():raise FileExistsError(args.output)
predictions=[];targets=[];priors=[];details=[];loss_sum=0;count=0
with torch.no_grad():
    for record in records:
        data=batch([record],'cpu');loss,pred=forward(model,data);mask=data[2]
        n=int(mask.sum());count+=n;loss_sum+=float(loss)*n
        predictions.append(pred[mask]);targets.append(data[1][mask])
        prior=torch.from_numpy(np.vstack([np.zeros((1,2),dtype=np.int64),record['actions'][:-1]]))
        priors.append(prior[record['alive']])
        details.append(dict(file=record['path'],sha256=record['sha256'],replay_sha256=record['metadata']['replay']['sha256'],
            map=record['metadata']['replay']['level'],loss=float(loss),**metrics(pred,data[1],mask)))
labels=torch.cat(targets);mask=torch.ones(len(labels),dtype=torch.bool)
report=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    matches=len(records),model=dict(loss=loss_sum/count,**metrics(torch.cat(predictions),labels,mask)),
    persistence=metrics(torch.cat(priors),labels,mask),per_match=details)
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='per_match'}),flush=True)
