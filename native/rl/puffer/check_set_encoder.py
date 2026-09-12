"""Qualify CUDA SET output/gradients, masked pooling and CUDA graph buffer reuse.

The Torch expressions here are an independent test oracle, never a training
backend. The production encoder uses PufferLib's native Encoder interface.
"""
import argparse
import ctypes
import json
import os
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
    # The encoder uses the overlay's per-stream cuBLAS handles (pufferlib-bopl.patch),
    # so compile against the patched overlay build.py generates, like production.
    overlay=ROOT/'native/artifacts/rl/puffer-overlay'
    if not (overlay/'kernels.cu').exists():
        raise SystemExit(f'{overlay} missing: run build.py --device cuda first')
    subprocess.run([str(cuda/'bin/nvcc'),'-O3','-std=c++17','-arch=sm_120',
                    '-ccbin',compiler,'-Xcompiler=-fPIC','-DPRECISION_FLOAT',*os.environ.get('BS_EXTRA','').split(),
                    '-I'+str(overlay),'-I'+str(source/'src'),'-I'+str(cuda/'include'),
                    '-c',str(HERE/'check_set_encoder.cu'),'-o',str(obj)],check=True)
    subprocess.run([compiler,'-shared','-Wl,--no-undefined',str(obj),
                    str(cuda/'lib/libcudart.so.13'),str(cuda/'lib/libcublas.so.13'),
                    str(cuda/'lib/libcurand.so.10'),'-lcuda',
                    '-Wl,--disable-new-dtags,-rpath,'+str(cuda/'lib'),
                    '-o',str(output)],check=True)


def oracle(obs, weights, hidden, incoming, dtype=torch.float32, value_obs=None):
    """Independent functional reference; autograd supplies all four weight gradients.
    float64 for large batches: at 4096 rows the CPU float32 flat gradient can be
    off by 1e-2 relative (cancellation over the batch) where the CUDA fp32 result
    is within 2e-5 of float64. value_obs (the tensor-core forward): the set
    projections take their *values* from these rows (TF32-rounded features) while
    the weight gradients use `obs`, as the CUDA backward reads the raw rows."""
    sizes=[hidden*FLAT,64*W_ENT,64*W_SLIME,hidden*(hidden+128)]
    w=torch.tensor(weights,dtype=dtype,requires_grad=True)
    flat,entity,slime,fusion=w.split(sizes)
    flat=flat.reshape(hidden,FLAT);entity=entity.reshape(64,W_ENT)
    slime=slime.reshape(64,W_SLIME);fusion=fusion.reshape(hidden,hidden+128)
    obs=torch.from_numpy(obs).to(dtype)
    vobs=obs if value_obs is None else torch.from_numpy(value_obs).to(dtype)
    x=torch.cat((obs[:,:PREFIX],obs[:,PREV_OFF:]),-1)
    def pool(rows,vrows,projection):
        mask=rows[:,:,0]!=0
        safe=rows.masked_fill(~mask[:,:,None],0)
        projected=safe@projection.T
        if vrows is not rows:  # value from the rounded rows, gradient through the raw ones
            projected=projected-projected.detach()+(vrows.masked_fill(~mask[:,:,None],0)@projection.T).detach()
        values=projected.relu().masked_fill(~mask[:,:,None],-float('inf'))
        pooled=values.max(1).values
        return torch.where(mask.any(1,keepdim=True),pooled,torch.zeros_like(pooled))
    entities=pool(obs[:,PREFIX:SLIME_OFF].reshape(-1,N_ENT,W_ENT),vobs[:,PREFIX:SLIME_OFF].reshape(-1,N_ENT,W_ENT),entity)
    slimes=pool(obs[:,SLIME_OFF:PREV_OFF].reshape(-1,N_SLIME,W_SLIME),vobs[:,SLIME_OFF:PREV_OFF].reshape(-1,N_SLIME,W_SLIME),slime)
    out=(torch.cat(((x@flat.T).relu(),entities,slimes),-1)@fusion.T).relu()
    (out*torch.from_numpy(incoming).to(dtype)).sum().backward()
    return out.detach().float().numpy(),w.grad.float().numpy()


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
    def gather(self,T,B_total,idx):
        idx=np.ascontiguousarray(idx,dtype=np.int32)
        if self.lib.bopl_set_check_gather(self.context,T,B_total,idx):
            raise RuntimeError(self.lib.bopl_set_check_error().decode())
    def time(self,iterations,forward_only=False):
        return self.lib.bopl_set_check_time(self.context,iterations,int(forward_only))
    def close(self):
        if self.context:self.lib.bopl_set_check_close(self.context);self.context=None


def random_obs(rng,batch):
    obs=rng.normal(size=(batch,OBS)).astype(np.float32)
    for start,stop,count,width in [(PREFIX,SLIME_OFF,N_ENT,W_ENT),(SLIME_OFF,PREV_OFF,N_SLIME,W_SLIME)]:
        rows=obs[:,start:stop].reshape(batch,count,width)
        rows[:,:,0]=rng.choice([0.,1.],size=(batch,count),p=[.7,.3])
    return obs


def packed_obs(rng,batch,counts=None):
    """Rows laid out like bopl_encode: valid slots [0, n) with feature 0 = 1 and an
    all-zero tail. n spans empty, the common few, the 16-slot head boundary of the
    packed forward (15, 16, 17) and full blocks."""
    obs=rng.normal(size=(batch,OBS)).astype(np.float32)
    for start,stop,count,width in [(PREFIX,SLIME_OFF,N_ENT,W_ENT),(SLIME_OFF,PREV_OFF,N_SLIME,W_SLIME)]:
        rows=obs[:,start:stop].reshape(batch,count,width)
        pool=[0,1,2,9,10,12,15,16,17,31,32,33,64,95,96] if count==N_ENT else list(range(count+1))
        n=rng.choice(pool,size=batch) if counts is None else np.asarray(counts)
        n=np.minimum(n,count)
        mask=np.arange(count)[None,:]<n[:,None]
        rows[:,:,0]=mask
        rows[~mask]=0
    return obs


def packed_cases(lib,rng):
    """BOPL_SET_PACKED=1 (the default): on bopl_encode's layout the packed forward
    must match the every-slot forward bit for bit (same arithmetic, fewer bytes
    read), directly and through a captured graph, including rows that take the
    slow path (>= 16 entities). The oracle comparison is limited to 1030 rows: at
    4096 rows with 16..96 random slots a few fp32-vs-float64 near-ties flip a
    pool winner and the entity gradient legitimately differs."""
    results=[]
    for batch,hidden in [(15,8),(1030,32),(4096,128)]:
        obs=packed_obs(rng,batch)
        crowded=packed_obs(rng,batch,counts=rng.integers(16,97,size=batch))
        count=hidden*FLAT+64*(W_ENT+W_SLIME)+hidden*(hidden+128)
        weights=(rng.normal(size=count)*.03).astype(np.float32)
        incoming=rng.normal(size=(batch,hidden)).astype(np.float32)
        check=Check(lib,batch,hidden)
        try:
            lib.bopl_set_check_set_packed(0)
            every,every_grad=check.run(weights,obs,incoming)
            every_slow,every_slow_grad=check.run(weights,crowded,incoming)
            lib.bopl_set_check_set_packed(1)
            packed,packed_grad=check.run(weights,obs,incoming)
            np.testing.assert_array_equal(packed,every);np.testing.assert_array_equal(packed_grad,every_grad)
            graph,graph_grad=check.run(weights,obs,incoming,True)
            np.testing.assert_array_equal(graph,packed);np.testing.assert_array_equal(graph_grad,packed_grad)
            slow,slow_grad=check.run(weights,crowded,incoming,True)  # replay on slow-path rows
            np.testing.assert_array_equal(slow,every_slow);np.testing.assert_array_equal(slow_grad,every_slow_grad)
            report=dict(batch=batch,hidden=hidden,bit_identical=True,graph_replay=True,
                        slow_path_rows=int((obs[:,PREFIX:SLIME_OFF].reshape(batch,N_ENT,W_ENT)[:,:,0].sum(1)>=16).sum()))
            if batch<=1030:
                for got,got_grad,rows in ((packed,packed_grad,obs),(slow,slow_grad,crowded)):
                    want,want_grad=oracle(rows,weights,hidden,incoming,torch.float64)
                    np.testing.assert_allclose(got,want,rtol=2e-4,atol=1e-5)
                    np.testing.assert_allclose(got_grad,want_grad,rtol=3e-4,atol=5e-5)
                report.update(forward_max_abs=float(np.abs(got-want).max()),gradient_max_abs=float(np.abs(got_grad-want_grad).max()))
            results.append(report)
        finally:
            check.close();lib.bopl_set_check_set_packed(0)
    return results


def tf32(x):
    """cvt.rna.tf32.f32: round to a 10-bit mantissa, ties away from zero."""
    bits=np.ascontiguousarray(x,dtype=np.float32).view(np.uint32)
    return ((bits+np.uint32(0x1000))&np.uint32(0xFFFFE000)).view(np.float32)


def mma_cases(lib,rng):
    """BOPL_SET_MMA=1 (the training default with TF32 GEMMs): the tensor-core
    forward projects TF32-rounded features against TF32-rounded entity / slime
    weights with fp32 accumulation, so it is checked against the float64 oracle fed
    the same rounded operands (the flat GEMM stays exact here). Both slot layouts
    (packed head chunks, every slot) must agree bit for bit with each other and
    with a graph replay, on head-only and crowded rows."""
    sizes=lambda hidden:[hidden*FLAT,64*W_ENT,64*W_SLIME,hidden*(hidden+128)]
    def rounded(obs,weights,hidden):
        obs=obs.copy();obs[:,PREFIX:PREV_OFF]=tf32(obs[:,PREFIX:PREV_OFF])
        weights=weights.copy();start=sizes(hidden)[0];stop=start+sizes(hidden)[1]+sizes(hidden)[2]
        weights[start:stop]=tf32(weights[start:stop])  # entity + slime weights
        return obs,weights
    results=[]
    lib.bopl_set_check_set_mma(1)
    try:
        for batch,hidden in [(15,8),(1030,32),(4096,128)]:
            count=sum(sizes(hidden))
            weights=(rng.normal(size=count)*.03).astype(np.float32)
            incoming=rng.normal(size=(batch,hidden)).astype(np.float32)
            report=dict(batch=batch,hidden=hidden,bit_identical=True,graph_replay=True)
            # Random rows have arbitrary slot masks: every-slot layout only. A
            # Check captures one graph, so each row set gets its own.
            for name,rows in (('packed',packed_obs(rng,batch)),('crowded',packed_obs(rng,batch,counts=rng.integers(16,97,size=batch))),('random',random_obs(rng,batch))):
                check=Check(lib,batch,hidden)
                try:
                    lib.bopl_set_check_set_packed(0)
                    every,every_grad=check.run(weights,rows,incoming)
                    lib.bopl_set_check_set_packed(name!='random')
                    packed,packed_grad=check.run(weights,rows,incoming)
                    np.testing.assert_array_equal(packed,every);np.testing.assert_array_equal(packed_grad,every_grad)
                    graph,graph_grad=check.run(weights,rows,incoming,True)
                    np.testing.assert_array_equal(graph,packed);np.testing.assert_array_equal(graph_grad,packed_grad)
                    if batch<=1030:
                        vrows,vweights=rounded(rows,weights,hidden)
                        want,want_grad=oracle(rows,vweights,hidden,incoming,torch.float64,value_obs=vrows)
                        np.testing.assert_allclose(packed,want,rtol=2e-4,atol=1e-5)
                        np.testing.assert_allclose(packed_grad,want_grad,rtol=3e-4,atol=5e-5)
                        report[name+'_forward_max_abs']=float(np.abs(packed-want).max())
                        report[name+'_gradient_max_abs']=float(np.abs(packed_grad-want_grad).max())
                finally:
                    check.close()
            results.append(report)
    finally:
        lib.bopl_set_check_set_mma(0);lib.bopl_set_check_set_packed(0)
    return results


def gather_cases(lib,rng):
    """Training path: rows read from a (T, B_total, obs) rollout buffer through
    segment indices must equal the contiguous path bit for bit (same kernels,
    same arithmetic order) and match the oracle; repeated segments are legal."""
    results=[]
    for T,B_total,segments,hidden in [(16,6,4,128),(8,3,7,16)]:
        batch=T*segments
        obs_tb=random_obs(rng,T*B_total).reshape(T,B_total,OBS)
        idx=rng.choice(B_total,size=segments,replace=True)
        idx[1]=idx[0]  # duplicate segment
        contiguous=np.ascontiguousarray(obs_tb[:,idx].transpose(1,0,2).reshape(batch,OBS))
        count=hidden*FLAT+64*(W_ENT+W_SLIME)+hidden*(hidden+128)
        weights=(rng.normal(size=count)*.03).astype(np.float32)
        incoming=rng.normal(size=(batch,hidden)).astype(np.float32)
        check=Check(lib,batch,hidden)
        try:
            direct,direct_grad=check.run(weights,contiguous,incoming)
            check.gather(T,B_total,idx)
            gathered,gathered_grad=check.run(weights,obs_tb,incoming,True)
            np.testing.assert_array_equal(gathered,direct);np.testing.assert_array_equal(gathered_grad,direct_grad)
            expected,expected_grad=oracle(contiguous,weights,hidden,incoming)
            np.testing.assert_allclose(gathered,expected,rtol=2e-4,atol=1e-5)
            np.testing.assert_allclose(gathered_grad,expected_grad,rtol=3e-4,atol=5e-5)
            # New indices into the same captured graph (idx is read by the kernels).
            idx2=rng.permutation(B_total)[:segments] if segments<=B_total else rng.choice(B_total,size=segments)
            check.gather(T,B_total,idx2)
            contiguous2=np.ascontiguousarray(obs_tb[:,idx2].transpose(1,0,2).reshape(batch,OBS))
            gathered2,gathered2_grad=check.run(weights,obs_tb,incoming,True)
            want,want_grad=oracle(contiguous2,weights,hidden,incoming)
            np.testing.assert_allclose(gathered2,want,rtol=2e-4,atol=1e-5)
            np.testing.assert_allclose(gathered2_grad,want_grad,rtol=3e-4,atol=5e-5)
            results.append(dict(T=T,B_total=B_total,segments=segments,hidden=hidden,bit_identical=True,
                                forward_max_abs=float(np.abs(gathered-expected).max()),
                                gradient_max_abs=float(np.abs(gathered_grad-expected_grad).max())))
        finally:check.close()
    return results


def timing(lib,batch=24576,hidden=128,T=128,B_total=448):
    """GPU ms per encoder forward / forward+backward at the training shape (gather mode)."""
    rng=np.random.default_rng(5)
    segments=batch//T
    check=Check(lib,batch,hidden)
    try:
        count=hidden*FLAT+64*(W_ENT+W_SLIME)+hidden*(hidden+128)
        weights=(rng.normal(size=count)*.03).astype(np.float32)
        obs_tb=random_obs(rng,T*B_total).reshape(T,B_total,OBS)
        idx=rng.choice(B_total,size=segments,replace=False)
        incoming=rng.normal(size=(batch,hidden)).astype(np.float32)
        check.gather(T,B_total,idx)
        check.run(weights,obs_tb,incoming)
        return dict(batch=batch,hidden=hidden,forward_ms=check.time(20,True),forward_backward_ms=check.time(20))
    finally:check.close()


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
    parser.add_argument('--time',action='store_true',help='Also time the encoder at the training shape (24576 x 128)')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    library=args.output/'check_set_encoder.so'
    if args.build or args.build_only:build(library,args.puffer,args.cuda_host_compiler)
    if args.build_only:
        print(library);return
    # The overlay's cuBLAS wrapper defaults to TF32; the oracle comparison is fp32.
    os.environ.setdefault('BOPL_TF32','0')
    lib=ctypes.CDLL(str(library))
    array=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
    lib.bopl_set_check_create.argtypes=[ctypes.c_int,ctypes.c_int];lib.bopl_set_check_create.restype=ctypes.c_void_p
    lib.bopl_set_check_error.restype=ctypes.c_char_p
    lib.bopl_set_check_weight_count.argtypes=[ctypes.c_void_p];lib.bopl_set_check_weight_count.restype=ctypes.c_long
    lib.bopl_set_check_run.argtypes=[ctypes.c_void_p,*([array]*5),ctypes.c_int];lib.bopl_set_check_run.restype=ctypes.c_int
    lib.bopl_set_check_close.argtypes=[ctypes.c_void_p]
    lib.bopl_set_check_gather.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')]
    lib.bopl_set_check_gather.restype=ctypes.c_int
    lib.bopl_set_check_time.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int];lib.bopl_set_check_time.restype=ctypes.c_double
    lib.bopl_set_check_set_packed.argtypes=[ctypes.c_int]
    lib.bopl_set_check_set_mma.argtypes=[ctypes.c_int]
    # The random corpus below has arbitrary slot masks: every-slot forward. The
    # production default (packed rows) is qualified by packed_cases.
    lib.bopl_set_check_set_packed(0)
    torch.set_num_threads(1)
    rng=np.random.default_rng(73);results=[]
    # 1 / 3 / 7 rows exercise partial blocks; 1030 rows (> 4 x 192 backward
    # blocks, not a multiple of the rows per block) exercises the grid-stride
    # loops and the multi-block gradient reduction.
    for batch,hidden in [(1,8),(3,16),(7,128),(1030,32)]:
        obs=random_obs(rng,batch)
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
    report=dict(schema=4,pass_all=True,cases=results,gather=gather_cases(lib,rng),packed=packed_cases(lib,rng),mma=mma_cases(lib,rng),
                finite_differences=finite_differences(lib))
    if args.time:report['timing']=timing(lib)
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
