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
    sizes=[hidden*905,64*21,64*28,hidden*(hidden+128)]
    w=torch.tensor(weights,requires_grad=True)
    flat,entity,slime,fusion=w.split(sizes)
    flat=flat.reshape(hidden,905);entity=entity.reshape(64,21)
    slime=slime.reshape(64,28);fusion=fusion.reshape(hidden,hidden+128)
    obs=torch.from_numpy(obs)
    x=torch.cat((obs[:,:898],obs[:,3250:]),-1)
    def pool(rows,projection):
        mask=rows[:,:,0]!=0
        safe=rows.masked_fill(~mask[:,:,None],0)
        values=(safe@projection.T).relu().masked_fill(~mask[:,:,None],-float('inf'))
        pooled=values.max(1).values
        return torch.where(mask.any(1,keepdim=True),pooled,torch.zeros_like(pooled))
    entities=pool(obs[:,898:2914].reshape(-1,96,21),entity)
    slimes=pool(obs[:,2914:3250].reshape(-1,12,28),slime)
    out=(torch.cat(((x@flat.T).relu(),entities,slimes),-1)@fusion.T).relu()
    (out*torch.from_numpy(incoming)).sum().backward()
    return out.detach().numpy(),w.grad.numpy()


class Check:
    def __init__(self, lib, batch, hidden):
        self.lib=lib;self.batch=batch;self.hidden=hidden
        self.context=lib.bopl_set_check_create(batch,hidden)
        if not self.context:raise RuntimeError(lib.bopl_set_check_error().decode())
        expected=hidden*905+64*(21+28)+hidden*(hidden+128)
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
        obs=rng.normal(size=(batch,3257)).astype(np.float32)
        for start,stop,count,width in [(898,2914,96,21),(2914,3250,12,28)]:
            rows=obs[:,start:stop].reshape(batch,count,width)
            rows[:,:,0]=rng.choice([0.,1.],size=(batch,count),p=[.7,.3])
        count=hidden*905+64*49+hidden*(hidden+128)
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
            for start,stop,n,width in [(898,2914,96,21),(2914,3250,12,28)]:
                rows=changed[:,start:stop].reshape(batch,n,width)
                rows[:,:,1:][rows[:,:,0]==0]=1e9
                rows[:]=rows[:,rng.permutation(n)]
            permuted,_=check.run(weights,changed,incoming,True)
            np.testing.assert_allclose(permuted,actual,rtol=0,atol=1e-6)
            # All-absent after populated graph replay must not retain pooled values or gradients.
            empty=changed.copy()
            empty[:,898:2914].reshape(batch,96,21)[:,:,0]=0
            empty[:,2914:3250].reshape(batch,12,28)[:,:,0]=0
            empty_out,empty_grad=check.run(weights,empty,incoming,True)
            want,want_grad=oracle(empty,weights,hidden,incoming)
            np.testing.assert_allclose(empty_out,want,rtol=2e-4,atol=1e-5)
            np.testing.assert_allclose(empty_grad,want_grad,rtol=3e-4,atol=5e-5)
            np.testing.assert_array_equal(empty_grad[hidden*905:hidden*905+64*49],0)
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
    report=dict(schema=1,pass_all=True,cases=results)
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
