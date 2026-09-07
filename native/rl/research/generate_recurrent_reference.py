"""Independent pinned Torch MinGRU fixture, checked against native CPU steps."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'native/rl/research'))
from native_policy import NativePolicy,REVISION

def main():
    source=ROOT/'native/artifacts/rl/dependencies'/f'PufferLib-{REVISION}'/'pufferlib/models.py'
    spec=importlib.util.spec_from_file_location('puffer_reference',source)
    upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    torch.set_num_threads(1)
    rng=np.random.default_rng(103)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"native/artifacts/rl/native-migration-audit/recurrent-reference")
    args=parser.parse_args()
    outdir=args.output
    outdir.mkdir(parents=True,exist_ok=True)
    summary=[]
    for obs,hidden,layers in [(11,16,2),(3257,128,1),(3257,128,2)]:
        T,B=17,3
        count=hidden*(obs+273)+layers*3*hidden*hidden
        flat_weights=(rng.normal(size=count)*.025).astype(np.float32)
        encoder_end=hidden*obs; decoder_end=encoder_end+273*hidden
        encoder=torch.tensor(flat_weights[:encoder_end].reshape(hidden,obs),requires_grad=True)
        decoder=torch.tensor(flat_weights[encoder_end:decoder_end].reshape(273,hidden),requires_grad=True)
        recurrent=upstream.MinGRU(hidden,layers)
        with torch.no_grad():
            for i,layer in enumerate(recurrent.layers):
                layer.weight.copy_(torch.from_numpy(flat_weights[decoder_end+i*3*hidden*hidden:decoder_end+(i+1)*3*hidden*hidden].reshape(3*hidden,hidden)))
        observations=rng.normal(size=(T,B,obs)).astype(np.float32)
        starts=rng.uniform(.1,.9,size=(layers,B,hidden)).astype(np.float32)
        resets=np.zeros((T,B),np.float32)
        for b,times in enumerate(((0,4,16),(3,7,15),(1,8,12))):
            resets[list(times),b]=1
        initial=torch.tensor(starts,requires_grad=True)
        state=(initial,); outputs=[]; states=[]
        for t in range(T):
            state=(state[0]*(1-torch.from_numpy(resets[t])).reshape(1,B,1),)
            x,state=recurrent.forward_eval(F.linear(torch.from_numpy(observations[t]),encoder),state)
            outputs.append(F.linear(x,decoder));states.append(state[0])
        output=torch.stack(outputs)
        grad_output=torch.tensor(rng.normal(size=output.shape).astype(np.float32))
        variables=[encoder,decoder,*[layer.weight for layer in recurrent.layers],initial]
        gradients=torch.autograd.grad((output*grad_output).sum(),variables,retain_graph=True)
        packed_gradient=torch.cat([g.flatten() for g in gradients[:-1]])
        # Every row reset within the sequence: final-output loss cannot reach initial memory.
        tail_initial=torch.autograd.grad(output[-1].sum(),initial)[0]
        assert torch.count_nonzero(tail_initial)==0
        # Compare each forward result to upstream C inference, carrying the same initial memory.
        policy=NativePolicy(flat_weights,obs,hidden,layers)
        actual_state=torch.from_numpy(starts.copy()); actual=[]
        for t in range(T):
            actual_state=actual_state*(1-torch.from_numpy(resets[t])).reshape(1,B,1)
            logits,values,actual_state=policy.step(torch.from_numpy(observations[t]),actual_state)
            actual.append(torch.cat((logits,values[:,None]),-1))
            torch.testing.assert_close(actual_state,states[t],rtol=2e-5,atol=2e-6)
        actual=torch.stack(actual)
        torch.testing.assert_close(actual,output,rtol=2e-5,atol=2e-6)
        filename=f'obs{obs}-h{hidden}-l{layers}.npz'
        np.savez(outdir/filename,weights=flat_weights,observations=observations,initial_state=starts,
                 reset_before_observation=resets,outputs=output.detach().numpy(),
                 states=torch.stack(states).detach().numpy(),grad_outputs=grad_output.numpy(),
                 weight_gradients=packed_gradient.numpy(),initial_state_gradients=gradients[-1].numpy(),
                 final_output_initial_state_gradient=tail_initial.numpy())
        summary.append(dict(file=filename,obs=obs,hidden=hidden,layers=layers,batch=B,timesteps=T,
                            cpu_forward_max_abs=float((actual-output).abs().max().detach()),
                            reset_stops_gradient=True))
    report=dict(schema=1,source_revision=REVISION,layout='time,batch,feature; state layers,batch,hidden',
                reset_semantics='reset before the observation at the same t',cases=summary)
    (outdir/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=="__main__":
    main()
