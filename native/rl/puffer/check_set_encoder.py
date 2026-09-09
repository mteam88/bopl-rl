"""Qualify CUDA SET output/gradients, masked pooling and CUDA graph buffer reuse.

The Torch expressions here are an independent test oracle, never a training
backend. The production encoder uses PufferLib's native Encoder interface.
"""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sysconfig

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
import sys; sys.path.insert(0,str(ROOT/'native/rl/research'))
from layout import (OBS_SIZE as OBS, OBS_PREFIX as PREFIX, OBS_SLIME_OFFSET as SLIME_OFF, OBS_PREVIOUS_OFFSET as PREV_OFF,
                    SET_FLAT as FLAT, OBS_ENTITIES as N_ENT, ENTITY_FEATURES as W_ENT, OBS_SLIMES as N_SLIME, SLIME_FEATURES as W_SLIME)
REVISION='42f70d6932c30ac977736f861006809c50168ba9'


def build(output, source, compiler):
    nvidia=Path(sysconfig.get_path('purelib'))/'nvidia'
    cuda=nvidia/'cu13'
    obj=output.with_suffix('.o')
    subprocess.run([str(cuda/'bin/nvcc'),'-O3','-std=c++17','-arch=sm_120',
                    '-ccbin',compiler,'-Xcompiler=-fPIC','-DPRECISION_FLOAT',
                    '-I'+str(source/'src'),'-I'+str(cuda/'include'),
                    '-c',str(HERE/'check_set_encoder.cu'),'-o',str(obj)],check=True)
    subprocess.run([compiler,'-shared','-Wl,--no-undefined',str(obj),
                    str(cuda/'lib/libcudart.so.13'),str(cuda/'lib/libcublas.so.13'),
                    str(cuda/'lib/libcurand.so.10'),
                    '-Wl,--disable-new-dtags,-rpath,'+str(cuda/'lib'),
                    '-o',str(output)],check=True)


def oracle(obs, weights, hidden, incoming):
    """Independent functional reference; autograd supplies all four weight gradients."""
    sizes=[hidden*FLAT,64*W_ENT,64*W_SLIME,hidden*(hidden+128)]
    w=torch.tensor(weights,requires_grad=True)
    flat,entity,slime,fusion=w.split(sizes)
    flat=flat.reshape(hidden,FLAT);entity=entity.reshape(64,W_ENT)
    slime=slime.reshape(64,W_SLIME);fusion=fusion.reshape(hidden,hidden+128)
    obs=torch.from_numpy(obs)
    x=torch.cat((obs[:,:PREFIX],obs[:,PREV_OFF:]),-1)
    def pool(rows,projection):
        mask=rows[:,:,0]!=0
        safe=rows.masked_fill(~mask[:,:,None],0)
        values=(safe@projection.T).relu().masked_fill(~mask[:,:,None],-float('inf'))
        pooled=values.max(1).values
        return torch.where(mask.any(1,keepdim=True),pooled,torch.zeros_like(pooled))
    entities=pool(obs[:,PREFIX:SLIME_OFF].reshape(-1,N_ENT,W_ENT),entity)
    slimes=pool(obs[:,SLIME_OFF:PREV_OFF].reshape(-1,N_SLIME,W_SLIME),slime)
    out=(torch.cat(((x@flat.T).relu(),entities,slimes),-1)@fusion.T).relu()
    (out*torch.from_numpy(incoming)).sum().backward()
    return out.detach().numpy(),w.grad.numpy()


class Check:
    def __init__(self, lib, batch, hidden):
        self.lib=lib;self.batch=batch;self.hidden=hidden
        self.context=lib.bopl_set_check_create(batch,hidden)
        if not self.context:raise RuntimeError(lib.bopl_set_check_error().decode())
        expected=hidden*FLAT+64*(W_ENT+W_SLIME)+hidden*(hidden+128)
        assert lib.bopl_set_check_weight_count(self.context)==expected
    def run(self,weights,obs,incoming,graph=False):
        output=np.empty((self.batch,self.hidden),np.float32);gradients=np.empty_like(weights)
        args=[np.ascontiguousarray(a,dtype=np.float32) for a in (weights,obs,incoming)]
        if self.lib.bopl_set_check_run(self.context,*args,output,gradients,int(graph)):
            raise RuntimeError(self.lib.bopl_set_check_error().decode())
        assert np.isfinite(output).all() and np.isfinite(gradients).all()
        return output,gradients
    def close(self):
        if self.context:self.lib.bopl_set_check_close(self.context);self.context=None


def finite_differences(lib):
    """Compare native backward with central differences of native forward.

    Positive weights/inputs keep ReLUs away from zero, and strictly increasing
    row magnitudes give each max pool a stable winner. The random mixed-sign
    corpus above separately checks changing winners and inactive ReLUs against
    autograd. Float64 loss accumulation limits cancellation of float32 outputs.
    """
    batch,hidden=2,8
    rng=np.random.default_rng(91073)
    obs=rng.uniform(.05,.2,size=(batch,OBS)).astype(np.float32)
    for start,stop,n,width in [(PREFIX,SLIME_OFF,N_ENT,W_ENT),(SLIME_OFF,PREV_OFF,N_SLIME,W_SLIME)]:
        rows=obs[:,start:stop].reshape(batch,n,width)
        base=rng.uniform(.05,.2,size=(batch,1,width-1))
        rows[:,:,1:]=base*np.linspace(.2,1.,n)[None,:,None]
        rows[:,:,0]=1
        rows[:,::3,0]=0
    sizes=[hidden*FLAT,64*W_ENT,64*W_SLIME,hidden*(hidden+128)]
    weights=rng.uniform(.01,.04,size=sum(sizes)).astype(np.float32)
    incoming=rng.uniform(-1,1,size=(batch,hidden)).astype(np.float32)
    check=Check(lib,batch,hidden)
    rows=[]
    try:
        _,gradient=check.run(weights,obs,incoming)
        offset=0
        for name,size in zip(('flat','entity','slime','fusion'),sizes):
            indices=offset+rng.choice(size,size=16,replace=False)
            numerical=[]
            for index in indices:
                plus=weights.copy();minus=weights.copy()
                plus[index]+=np.float32(.001);minus[index]-=np.float32(.001)
                high,_=check.run(plus,obs,incoming)
                low,_=check.run(minus,obs,incoming)
                delta=float(plus[index])-float(minus[index])
                numerical.append(np.sum((high.astype(np.float64)-low)*incoming)/delta)
            numerical=np.asarray(numerical)
            analytic=gradient[indices]
            np.testing.assert_allclose(analytic,numerical,rtol=.003,atol=.0003,
                                       err_msg=f'{name} native finite difference')
            rows.append(dict(block=name,coordinates=len(indices),
                             max_absolute_error=float(np.abs(analytic-numerical).max())))
            offset+=size
    finally:check.close()
    return dict(batch=batch,hidden=hidden,epsilon=.001,blocks=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'native/artifacts/rl/native-migration-audit/cuda-set')
    parser.add_argument('--puffer',type=Path,default=ROOT/'native/artifacts/rl/dependencies'/f'PufferLib-{REVISION}')
    parser.add_argument('--build',action='store_true')
    parser.add_argument('--build-only',action='store_true',help='Compile the harness without loading CUDA or running checks')
    parser.add_argument('--cuda-host-compiler',default='g++-15')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    library=args.output/'check_set_encoder.so'
    if args.build or args.build_only:build(library,args.puffer,args.cuda_host_compiler)
    if args.build_only:
        print(library);return
    lib=ctypes.CDLL(str(library))
    array=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    lib.bopl_set_check_create.argtypes=[ctypes.c_int,ctypes.c_int];lib.bopl_set_check_create.restype=ctypes.c_void_p
    lib.bopl_set_check_error.restype=ctypes.c_char_p
    lib.bopl_set_check_weight_count.argtypes=[ctypes.c_void_p];lib.bopl_set_check_weight_count.restype=ctypes.c_long
    lib.bopl_set_check_run.argtypes=[ctypes.c_void_p,*([array]*5),ctypes.c_int];lib.bopl_set_check_run.restype=ctypes.c_int
    lib.bopl_set_check_close.argtypes=[ctypes.c_void_p]
    torch.set_num_threads(1)
    rng=np.random.default_rng(73);results=[]
    for batch,hidden in [(1,8),(3,16),(7,128)]:
        obs=rng.normal(size=(batch,OBS)).astype(np.float32)
        for start,stop,count,width in [(PREFIX,SLIME_OFF,N_ENT,W_ENT),(SLIME_OFF,PREV_OFF,N_SLIME,W_SLIME)]:
            rows=obs[:,start:stop].reshape(batch,count,width)
            rows[:,:,0]=rng.choice([0.,1.],size=(batch,count),p=[.7,.3])
        count=hidden*FLAT+64*(W_ENT+W_SLIME)+hidden*(hidden+128)
        weights=(rng.normal(size=count)*.03).astype(np.float32)
        incoming=rng.normal(size=(batch,hidden)).astype(np.float32)
        check=Check(lib,batch,hidden)
        try:
            expected,expected_grad=oracle(obs,weights,hidden,incoming)
            actual,grad=check.run(weights,obs,incoming)
            np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=1e-5)
            np.testing.assert_allclose(grad,expected_grad,rtol=3e-4,atol=5e-5)
            graph,graph_grad=check.run(weights,obs,incoming,True)
            np.testing.assert_array_equal(graph,actual);np.testing.assert_array_equal(graph_grad,grad)
            # Reorder slots and poison absent rows: predictions must be unaffected.
            changed=obs.copy()
            for start,stop,n,width in [(PREFIX,SLIME_OFF,N_ENT,W_ENT),(SLIME_OFF,PREV_OFF,N_SLIME,W_SLIME)]:
                rows=changed[:,start:stop].reshape(batch,n,width)
                rows[:,:,1:][rows[:,:,0]==0]=1e9
                rows[:]=rows[:,rng.permutation(n)]
            permuted,_=check.run(weights,changed,incoming,True)
            np.testing.assert_allclose(permuted,actual,rtol=0,atol=1e-6)
            # All-absent after populated graph replay must not retain pooled values or gradients.
            empty=changed.copy()
            empty[:,PREFIX:SLIME_OFF].reshape(batch,N_ENT,W_ENT)[:,:,0]=0
            empty[:,SLIME_OFF:PREV_OFF].reshape(batch,N_SLIME,W_SLIME)[:,:,0]=0
            empty_out,empty_grad=check.run(weights,empty,incoming,True)
            want,want_grad=oracle(empty,weights,hidden,incoming)
            np.testing.assert_allclose(empty_out,want,rtol=2e-4,atol=1e-5)
            np.testing.assert_allclose(empty_grad,want_grad,rtol=3e-4,atol=5e-5)
            np.testing.assert_array_equal(empty_grad[hidden*FLAT:hidden*FLAT+64*(W_ENT+W_SLIME)],0)
            # Mutate all inputs to a captured graph, including weights and upstream gradient.
            next_weights=(weights*.8).astype(np.float32);next_incoming=-incoming
            changed_out,changed_grad=check.run(next_weights,obs,next_incoming,True)
            want,want_grad=oracle(obs,next_weights,hidden,next_incoming)
            np.testing.assert_allclose(changed_out,want,rtol=2e-4,atol=1e-5)
            np.testing.assert_allclose(changed_grad,want_grad,rtol=3e-4,atol=5e-5)
            results.append(dict(batch=batch,hidden=hidden,weights=count,
                                forward_max_abs=float(np.abs(actual-expected).max()),
                                gradient_max_abs=float(np.abs(grad-expected_grad).max()),
                                graph_replay=True,permutation=True,masked_poison=True,empty_after_populated=True))
        finally:check.close()
    report=dict(schema=1,pass_all=True,cases=results,finite_differences=finite_differences(lib))
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
